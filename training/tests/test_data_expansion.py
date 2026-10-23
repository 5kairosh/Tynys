"""New data cannot recycle exposed patients or turn other disease into NORMAL."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
import expand_training_data as prepare
from train_data_expansion import ensure_unopened, promotion_gate


def row(name, labels, patient=None, view='PA'):
    return {'ImageID': name, 'PatientID': patient or name, 'Projection': view, 'Labels': repr(labels)}


def test_pilot_never_maps_other_abnormalities_to_normal_or_keeps_lateral():
    records = [row('normal', ['normal']), row('positive', [' pneumonia', 'infiltrates']),
        row('other', ['cardiomegaly']), row('ambiguous', ['normal', 'pneumonia']),
        row('lateral', ['pneumonia'], view='L'), row('rib', ['normal'], view='COSTAL')]
    selected, excluded = prepare.padchest_candidates(records)
    assert {(r['ImageID'], label) for r,label in selected} == {('normal',0), ('positive',1)}
    assert excluded == {'other_or_ambiguous':2, 'non_frontal':2}


def test_pilot_one_image_per_patient_and_order_independent():
    records = [row('study-a', ['normal'], 'same'), row('study-b', ['normal'], 'same'), row('other', ['pneumonia'])]
    selected, _ = prepare.padchest_candidates(records)
    reversed_selection, _ = prepare.padchest_candidates(list(reversed(records)))
    assert selected == reversed_selection and len(selected) == 2
    with pytest.raises(ValueError, match='grouping'):
        prepare.padchest_candidates([{**records[0], 'PatientID':''}])


def test_new_training_validation_confirmation_exclude_every_old_patient(monkeypatch):
    candidates = [{'patient_group':f'nih:{label}-{i}', 'label':label, 'source_id':f'{label}-{i}'}
                  for label in (0,1) for i in range(12)]
    monkeypatch.setattr(prepare, 'metadata', lambda _: {})
    monkeypatch.setattr(prepare, 'select_cohort', lambda *args: ({'all':candidates}, {}))
    protocol = {'seed':42, 'rsna_selection':{'original_selection_seed':1, 'pool_per_class':12,
        'new_counts_per_class':{'train':4, 'validation':2, 'confirmation':2}}}
    previous = [{'patient_group':f'nih:{label}-{i}'} for label in (0,1) for i in range(3)]
    splits = prepare.fresh_rsna(protocol, previous)
    all_rows = [r for rows in splits.values() for r in rows]
    assert len({r['patient_group'] for r in all_rows}) == len(all_rows) == 16
    assert not {r['patient_group'] for r in all_rows} & {r['patient_group'] for r in previous}
    for rows in splits.values():
        assert sum(r['label']==0 for r in rows) == sum(r['label']==1 for r in rows)
    assert prepare.fresh_rsna(protocol, previous) == splits
    with pytest.raises(ValueError, match='Insufficient fresh'):
        prepare.fresh_rsna(protocol, [{'patient_group':r['patient_group']} for r in candidates])


def test_frozen_candidate_and_opened_confirmation_cannot_be_overwritten(tmp_path):
    ensure_unopened(tmp_path, 'fit')
    (tmp_path/'fit_report.json').write_text('{}')
    with pytest.raises(ValueError, match='Fit already frozen'):
        ensure_unopened(tmp_path, 'fit')
    ensure_unopened(tmp_path, 'evaluate')
    (tmp_path/'evaluation.json').write_text('{}')
    for mode in ('fit','evaluate'):
        with pytest.raises(ValueError, match='Confirmation already opened'):
            ensure_unopened(tmp_path, mode)


def test_high_old_accuracy_cannot_override_failed_fresh_confirmation():
    protocol = json.loads((TRAINING/'data_expansion_20261023/protocol.json').read_text())
    good = {'candidate':{'sensitivity':.99,'specificity':.99}, 'accuracy_gain':.05,
            'balanced_accuracy_gain':.05, 'paired_accuracy_bootstrap':{'lower_95':.01}}
    cohorts = {name:copy.deepcopy(good) for name in ('new_rsna_confirmation','internal_test','external_openi')}
    assert all(promotion_gate(protocol, {'validation_feasible':True}, cohorts).values())
    cohorts['new_rsna_confirmation']['accuracy_gain'] = 0
    cohorts['new_rsna_confirmation']['paired_accuracy_bootstrap']['lower_95'] = -.02
    gates = promotion_gate(protocol, {'validation_feasible':True}, cohorts)
    assert not gates['fresh_accuracy_gain'] and not gates['fresh_paired_gain_lower_bound']
    assert not all(gates.values())


def test_real_expansion_manifest_has_no_prior_or_cross_split_patient_leakage():
    def read(path):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    directory = TRAINING/'data_expansion_20261023'
    prior = [r for name in ('train','validation','holdout','holdout_confirmation')
             for r in read(TRAINING/'generalization_stage6'/(name+'_manifest.csv'))]
    old_groups = {r['patient_group'] for r in prior}
    old_hashes = {r['sha256'] for r in prior}
    for name in ('train','validation','test'):
        old_hashes.update(r['sha256'] for r in read(TRAINING/'manifests'/(name+'_manifest.csv')))
    old_hashes.update(r['sha256'] for r in read(TRAINING/'external_predictions.csv'))
    rows = []
    for split, expected in [('train',1200),('validation',200),('confirmation',400)]:
        part = read(directory/(split+'_manifest.csv'))
        assert len(part) == expected
        assert sum(int(r['label'])==0 for r in part) == expected//2
        assert sum(int(r['label'])==1 for r in part) == expected//2
        assert all(r['split']==split and r['target'] in {'NORMAL','RSNA_LUNG_OPACITY'} for r in part)
        rows.extend(part)
    assert len(rows) == len({r['source_id'] for r in rows}) == len({r['patient_group'] for r in rows}) == len({r['sha256'] for r in rows}) == 1800
    assert not old_groups & {r['patient_group'] for r in rows}
    assert not old_hashes & {r['sha256'] for r in rows}


def test_published_experiment_cannot_hide_a_failed_gate_or_change_frozen_fit():
    directory = TRAINING/'data_expansion_20261023'
    protocol = json.loads((directory/'protocol.json').read_text())
    fitted = json.loads((directory/'fit_report.json').read_text())
    measured = json.loads((directory/'evaluation.json').read_text())
    logical_sha = lambda path: hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    assert fitted['protocol_sha256'] == logical_sha(directory/'protocol.json')
    assert measured['fit_report_sha256'] == logical_sha(directory/'fit_report.json')
    assert fitted['training_code_sha256'] == logical_sha(TRAINING/'train_data_expansion.py')
    assert fitted['evaluation_used_to_fit_or_select'] is False
    assert measured['thresholds_tuned_on_evaluation'] is False
    assert measured['clinically_validated'] is False
    gates = promotion_gate(protocol, fitted, measured['cohorts'])
    assert measured['promotion_gate'] == gates and measured['promotion_passed'] == all(gates.values())
    for cohort, expected in [('new_rsna_confirmation',400), ('internal_test',618), ('external_openi',200)]:
        with (directory/(cohort+'_predictions.csv')).open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len({r['sha256'] for r in rows}) == expected
        for key in ('baseline','candidate'):
            tn = sum(int(r['label'])==0 and int(r[key+'_prediction'])==0 for r in rows)
            fp = sum(int(r['label'])==0 and int(r[key+'_prediction'])==1 for r in rows)
            fn = sum(int(r['label'])==1 and int(r[key+'_prediction'])==0 for r in rows)
            tp = sum(int(r['label'])==1 and int(r[key+'_prediction'])==1 for r in rows)
            stats = measured['cohorts'][cohort][key]
            assert [stats[k] for k in ('TN','FP','FN','TP')] == [tn,fp,fn,tp]
            assert stats['accuracy'] == (tn+tp)/expected
