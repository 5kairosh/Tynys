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
    assert hashlib.sha256((MODEL/'ood_params.npz').read_bytes()).hexdigest()==report['parameter_sha256']
    protocol=(root/'training/ood_stage5/protocol.json').read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(protocol).hexdigest()==report['protocol_sha256']


def test_published_v2_has_disjoint_fitting_roles_and_finite_cutoff():
    import hashlib
    config_sha=hashlib.sha256((MODEL/'model_config.json').read_bytes()).hexdigest()
    detector=OODDetector();detector.load(MODEL/'ood_params.npz','ea584b998a45',config_sha)
    s=detector.status()
    assert s['available'],s
    assert s['method']=='train_validation_mahalanobis_v2'
    assert s['reference_count']==3801 and s['calibration_count']==952
    assert s['target_acceptance']==.95 and s['threshold']>0
    assert detector.metadata['test_or_external_used_to_fit'] is False
    assert detector.metadata['clinical_or_formal_coverage_guarantee'] is False
    assert detector.metadata['quantile_rank']==906
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
    detector=OODDetector();detector.load(path,'ea584b998a45',expected_config)
    assert detector.status()['available'] is False


@pytest.mark.parametrize('vector', [np.zeros((1,2)),np.array([np.nan,0]),np.array([np.inf,0]),np.zeros(3)])
def test_invalid_feature_vector_cannot_silently_pass(vector):
    detector=OODDetector();detector.params=(np.zeros(2),np.eye(2),3.)
    with pytest.raises(ValueError):detector.check(vector)
