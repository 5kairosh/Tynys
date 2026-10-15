"""Test actual data separation and threshold-selection contracts without training."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multilabel_experiment import terms, audit_splits, choose_threshold, LABELS


def parts():
    return {name:[{'report_id':name+str(i),'group_id':name+str(i),'sha256':name+str(i),**{label:i for label in LABELS}} for i in (0,1)] for name in ('train','validation','test')}


def test_mesh_aliases_and_absence():
    assert terms({'MeSH':{'major':['<b>Pulmonary Atelectasis</b>/lower lobe','Cardiomegaly'], 'minor':['Pneumonia/focal']}})=={'atelectasis','cardiomegaly','pneumonia'}
    assert 'pneumonia' not in terms({'MeSH':{'major':['Normal'],'minor':None}})


@pytest.mark.parametrize('key',['report_id','group_id','sha256'])
def test_overlap_rejected(key):
    splits=parts();audit_splits(splits)
    splits['test'][0][key]=splits['train'][0][key]
    with pytest.raises(AssertionError):audit_splits(splits)


def test_missing_label_class_rejected():
    splits=parts()
    for r in splits['validation']:r['pneumonia']=0
    with pytest.raises(AssertionError):audit_splits(splits)


def test_per_label_validation_f1_threshold():
    y=np.array([0,0,1,1]);p=np.array([.1,.2,.4,.8])
    threshold=choose_threshold(y,p)
    assert .2<threshold<=.4
    np.testing.assert_array_equal(p>=threshold,y)


def test_saved_report_groups_and_hashes_are_disjoint():
    import csv
    directory=Path(__file__).resolve().parents[1]/'outputs/multilabel'
    if not (directory/'train_manifest.csv').exists():pytest.skip('Experiment not prepared yet.')
    splits={}
    for name in ('train','validation','test'):
        with (directory/(name+'_manifest.csv')).open(encoding='utf-8-sig') as f:splits[name]=list(csv.DictReader(f))
    audit_splits(splits)
    assert all(len(r['sha256'])==64 for rows in splits.values() for r in rows)
