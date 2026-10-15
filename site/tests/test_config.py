import json
import pytest
from app import read_config
from calibration import calibrate_score


@pytest.mark.parametrize('patch',[
    {'classes':['PNEUMONIA','NORMAL']},{'classes':None},
    {'threshold':True},{'threshold':-0.1},{'threshold':1.01},{'threshold':float('nan')},
    {'image_size':[31,224]},{'image_size':[224,1025]},{'image_size':[True,224]},
    {'image_size':[224]},{'calibration':{'method':'temperature','T':0}},
    {'calibration':{'method':'temperature','T':True}},{'calibration':{'method':'unknown','T':1}},
])
def test_invalid(tmp_path,patch):
    value={'classes':['NORMAL','PNEUMONIA'],'image_size':[224,224],'threshold':.74,**patch}
    p=tmp_path/'config.json';p.write_text(json.dumps(value))
    with pytest.raises((ValueError,TypeError)):
        read_config(p)


@pytest.mark.parametrize('threshold',[0,.74,1])
def test_valid(tmp_path,threshold):
    p=tmp_path/'config.json';p.write_text(json.dumps({'classes':['NORMAL','PNEUMONIA'],'image_size':[32,1024],'threshold':threshold}))
    assert read_config(p)['threshold']==threshold


def test_temperature_identity_and_order():
    for p in [0,.01,.5,.993072,1]:
        assert calibrate_score(p)==p
        assert calibrate_score(p,{'method':'temperature','T':1})==p
    assert .5<calibrate_score(.9,{'method':'temperature','T':2})<.9
