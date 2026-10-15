import json
import sys
from pathlib import Path
import numpy as np
import pytest
from ood import OODDetector


def test_ood_unavailable_and_distance(tmp_path):
    detector=OODDetector()
    detector.load(tmp_path/'missing.npz','model')
    assert not detector.check([0,0])['available']
    detector.params=(np.zeros(2),np.eye(2),4.9)
    result=detector.check([3,4])
    assert result['distance']==5 and result['is_xray'] is False
    detector.params=(np.zeros(2),np.eye(2),5.)
    assert detector.check([3,4])['is_xray'] is True
    invalid=tmp_path/'ood.npz'
    np.savez(invalid,mean=np.zeros(2),precision=np.eye(2),threshold=3.,model_id='old')
    detector.load(invalid,'new')
    assert not detector.status()['available']


def test_external_group_sampling():
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'))
    from external_validation import select_groups
    rows=[{'patient_id':str(i),'label':i%2,'path':f'{i}.png'} for i in range(240)]
    rows += [rows[0].copy()]
    chosen=select_groups(rows,200,42)
    assert len(chosen)==len({r['patient_id'] for r in chosen})==200
    assert chosen==select_groups(rows,200,42)
    with pytest.raises(ValueError):
        select_groups(rows+[{'patient_id':'0','label':1,'path':'other.png'}],200,42)
