import json
from pathlib import Path
import numpy as np
import pytest
from ood import OODDetector

MODEL = Path(__file__).resolve().parents[1]/'model'


def test_published_cutoff_replays_from_validation_only():
    import csv, hashlib, math
    root=MODEL.parents[1]
    with (root/'training/ood_stage5/validation_distances.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    with (root/'training/manifests/validation_manifest.csv').open() as stream:
        validation=list(csv.DictReader(stream))
    assert [r['sha256'] for r in rows]==[r['sha256'] for r in validation]
    report=json.loads((root/'training/ood_stage5/fit_report.json').read_text(encoding='utf-8'))
    assert len(rows)==952 and report['quantile_rank']==math.ceil(953*.95)==906
    assert sorted(float(r['distance']) for r in rows)[905]==report['threshold']
    assert sum(int(r['accepted']) for r in rows)==report['calibration_accepted']==906
    # Stage5 is a historical replay; the active Stage6 parameters are tested
    # against their own disjoint development manifests below.
    protocol=(root/'training/ood_stage5/protocol.json').read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(protocol).hexdigest()==report['protocol_sha256']


def test_published_v2_has_disjoint_fitting_roles_and_finite_cutoff():
    import hashlib, math, csv
    root=MODEL.parents[1]
    report=json.loads((root/'training/generalization_stage6/ood_fit_report.json').read_text())
    model_id=hashlib.sha256((MODEL/'pneumonia_mobilenetv2.keras').read_bytes()).hexdigest()[:12]
    config_sha=hashlib.sha256((MODEL/'model_config.json').read_bytes()).hexdigest()
    detector=OODDetector();detector.load(MODEL/'ood_params.npz',model_id,config_sha)
    s=detector.status()
    assert s['available'],s
    assert s['method']=='train_validation_mahalanobis_v2'
    assert s['reference_count']==5401 and s['calibration_count']==1352
    assert s['target_acceptance']==.95 and s['threshold']>0
    assert detector.metadata['test_or_external_used_to_fit'] is False
    assert detector.metadata['clinical_or_formal_coverage_guarantee'] is False
    assert detector.metadata['quantile_rank']==math.ceil(1353*.95)==1286
    with (root/'training/generalization_stage6/ood_validation_distances.csv').open() as stream:
        distances=list(csv.DictReader(stream))
    with (root/'training/generalization_stage6/ood_validation_manifest.csv').open() as stream:
        manifest=list(csv.DictReader(stream))
    assert len(distances)==1352 and [r['sha256'] for r in distances]==[r['sha256'] for r in manifest]
    assert sorted(float(r['distance']) for r in distances)[1285]==s['threshold']==report['threshold']
    assert hashlib.sha256((MODEL/'ood_params.npz').read_bytes()).hexdigest()==report['parameter_sha256']
    check=detector.check(detector.params[0])
    assert check['within_reference'] is True and check['is_xray'] is True
    assert check['distance']==0 and check['parameter_id']==s['parameter_id']


@pytest.mark.parametrize('bad', ['model','config','legacy','role','rank','nan','shape','asymmetric','not_positive'])
def test_corrupt_or_stale_ood_never_uses_old_threshold(tmp_path,bad):
    with np.load(MODEL/'ood_params.npz',allow_pickle=False) as saved:
        values={k:saved[k] for k in saved.files}
    metadata=json.loads(str(values['metadata']))
    expected_config=metadata['config_sha256']
    if bad=='model':values['model_id']='other'
    if bad=='config':expected_config='other'
    if bad=='legacy':values.pop('metadata')
    if bad=='role':metadata['reference_split']='validation'
    if bad=='rank':metadata['quantile_rank']=1
    if bad=='nan':values['threshold']=np.nan
    if bad=='shape':values['mean']=np.zeros(5)
    if bad=='asymmetric':
        values['precision']=values['precision'].copy();values['precision'][0,1]+=1
    if bad=='not_positive':values['precision']=-np.eye(1280)
    if bad!='legacy':values['metadata']=json.dumps(metadata)
    path=tmp_path/'bad.npz';np.savez_compressed(path,**values)
    detector=OODDetector();detector.load(path,str(values['model_id']) if bad!='model' else 'expected-real-model',expected_config)
    assert detector.status()['available'] is False


@pytest.mark.parametrize('vector', [np.zeros((1,2)),np.array([np.nan,0]),np.array([np.inf,0]),np.zeros(3)])
def test_invalid_feature_vector_cannot_silently_pass(vector):
    detector=OODDetector();detector.params=(np.zeros(2),np.eye(2),3.)
    with pytest.raises(ValueError):detector.check(vector)
