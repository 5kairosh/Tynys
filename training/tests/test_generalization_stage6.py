"""Leakage, label semantics and decision-preserving calibration regressions."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
from prepare_rsna_stage6 import select_cohort
from adapt_external_stage6 import balanced_domain_weights, calibrated, paired_accuracy_interval


def test_one_patient_cannot_cross_rsna_splits_even_when_findings_change():
    annotations = {'labelGroups':[{'name':'Calculated','labels':[
        {'id':'normal','name':'Normal'},{'id':'opacity','name':'Lung Opacity'},
        {'id':'other','name':'No Lung Opacity / Not Normal'}]}], 'datasets':[{'annotations':[]}]}
    mapping = []
    for patient in range(60):
        # Repeated examinations are deliberately assigned opposite labels.
        for examination in range(2):
            uid=f'study-{patient}-{examination}'
            mapping.append({'subset_img_id':uid,'img_id':f'{patient:08}_00{examination}.png','StudyInstanceUID':uid})
            annotations['datasets'][0]['annotations'].append({'StudyInstanceUID':uid,'labelId':'normal' if examination==0 else 'opacity'})
    protocol={'seed':20261021,'rsna_counts_per_class':{'train':8,'validation':3,'holdout':3}}
    splits,audit=select_cohort(annotations,mapping,protocol)
    rows=sum(splits.values(),[])
    assert len(rows)==28 and len({r['patient_group'] for r in rows})==28
    for part in splits.values():
        assert sum(r['label']==0 for r in part)==sum(r['label']==1 for r in part)
    assert select_cohort(annotations,list(reversed(mapping)),protocol)==(splits,audit)


def test_other_abnormalities_are_never_relabeled_normal():
    labels=[{'id':'n','name':'Normal'},{'id':'p','name':'Lung Opacity'},{'id':'a','name':'No Lung Opacity / Not Normal'}]
    records=[];mapping=[]
    for i in range(40):
        uid=str(i);label='n' if i<15 else 'p' if i<30 else 'a'
        records.append({'StudyInstanceUID':uid,'labelId':label})
        mapping.append({'StudyInstanceUID':uid,'subset_img_id':uid,'img_id':f'{i:08}_000.png'})
    splits,audit=select_cohort({'labelGroups':[{'name':'Calculated','labels':labels}],'datasets':[{'annotations':records}]},mapping,
                               {'seed':42,'rsna_counts_per_class':{'train':5,'validation':2,'holdout':2}})
    assert all(int(r['source_id'])<30 for r in sum(splits.values(),[]))
    assert audit['exclusions']['non_normal_other_finding']==10


def test_domain_class_weights_do_not_let_large_kaggle_stratum_dominate():
    domains=np.array(['kaggle']*12+['rsna']*4)
    labels=np.array([0]*9+[1]*3+[0]+[1]*3)
    weights=balanced_domain_weights(domains,labels)
    totals=[weights[(domains==d)&(labels==label)].sum() for d in ('kaggle','rsna') for label in (0,1)]
    assert np.allclose(totals,[4]*4) and weights.sum()==len(labels)
    with pytest.raises(ValueError):balanced_domain_weights(['kaggle','rsna'],[0,1])


@pytest.mark.parametrize('temperature',[.2,.98,1.,2.,5.])
def test_temperature_transform_preserves_the_selected_operating_decisions(temperature):
    raw=np.array([0.,.01,.1,.4,.5,.8,.99,1.])
    threshold=.405
    from scipy.special import expit,logit
    adjusted_threshold=expit(logit(threshold)/temperature)
    assert np.array_equal(raw>=threshold,calibrated(raw,temperature)>=adjusted_threshold)
    sys.path.insert(0,str(TRAINING.parent/'site'))
    from calibration import calibrate_score
    assert np.allclose(calibrated(raw,temperature),[calibrate_score(float(x),{'method':'temperature','T':temperature}) for x in raw])


def test_paired_gain_interval_cannot_claim_improvement_for_identical_predictions():
    y=np.array([0,0,1,1]);old=np.array([0,1,0,1])
    interval=paired_accuracy_interval(y,old,old,42,repeats=2000)
    assert interval['difference']==interval['lower_95']==interval['upper_95']==0
    worse=paired_accuracy_interval(y,y,1-y,42,repeats=2000)
    assert worse['upper_95']==-1


def test_new_holdout_cannot_be_used_to_select_another_head(monkeypatch,tmp_path):
    import adapt_external_stage6 as module
    monkeypatch.setattr(module,'OUT',tmp_path)
    (tmp_path/'fit_report.json').write_text('{}')
    with pytest.raises(ValueError,match='Fit already frozen'):module.fit()
    (tmp_path/'evaluation.json').write_text('{}')
    with pytest.raises(ValueError,match='Once-only holdout'):module.evaluate()
