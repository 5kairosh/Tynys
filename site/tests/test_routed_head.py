"""Check the actual serialized router, chosen margin gradients and provenance."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'training/generalization_stage6'


@pytest.mark.parametrize('domain_logit,selected',[(-5.,'baseline_margin'),(5.,'adapted_margin')])
def test_real_serialized_routing_uses_the_selected_expert_and_its_gradient(client,domain_logit,selected,tmp_path):
    from app import engine
    tf=engine.tf;head=engine.model.layers[-1]
    assert head.name=='domain_routed_head'
    kernel,bias=head.get_layer('source_domain_logit').get_weights()
    vector=((domain_logit-float(bias[0]))*kernel[:,0]/float(np.sum(kernel[:,0]**2))).astype(np.float32)[None,:]
    selected_layer=head.get_layer(selected)
    expected=selected_layer(vector,training=False)
    logits_model=tf.keras.Model(head.input,head.get_layer('probability').input)
    x=tf.convert_to_tensor(vector)
    with tf.GradientTape() as tape:
        tape.watch(x);z=logits_model(x,training=False)
    gradient=tape.gradient(z,x).numpy()
    assert np.allclose(z.numpy(),expected.numpy(),atol=1e-5)
    assert np.allclose(gradient,selected_layer.kernel.numpy().T,atol=1e-6)
    destination=tmp_path/'head.keras';head.save(destination)
    restored=tf.keras.models.load_model(destination,compile=False,safe_mode=True)
    assert np.allclose(restored(vector,training=False),tf.sigmoid(expected),atol=1e-6)


def test_actual_config_and_model_match_the_frozen_passed_router():
    report=json.loads((STUDY/'routing_fit_report.json').read_text())
    gate=json.loads((STUDY/'routing_evaluation.json').read_text())
    assert gate['promotion_passed'] and all(gate['promotion_gate'].values())
    first=json.loads((STUDY/'evaluation.json').read_text())
    assert first['promotion_passed'] is False and first['promotion_gate']['kaggle_balanced_accuracy_noninferiority'] is False
    model=ROOT/'site/model/pneumonia_mobilenetv2.keras'
    assert hashlib.sha256(model.read_bytes()).hexdigest()==report['model_sha256']
    config=json.loads((ROOT/'site/model/model_config.json').read_text(encoding='utf-8'))
    assert config['routing']['head']=='domain_routed_head'
    assert config['threshold']==report['threshold']==.5
    assert config['calibration']['T']==report['temperature']


def test_new_nih_confirmation_patients_and_hashes_never_enter_fitting():
    def rows(name):
        with (STUDY/name).open(encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))
    confirm=rows('holdout_confirmation_manifest.csv')
    seen=[r for split in ('train','validation','holdout') for r in rows(split+'_manifest.csv')]
    assert len(confirm)==len({r['patient_group'] for r in confirm})==400
    assert not {r['patient_group'] for r in confirm}&{r['patient_group'] for r in seen}
    assert not {r['sha256'] for r in confirm}&{r['sha256'] for r in seen}
    train=rows('ood_train_manifest.csv');validation=rows('ood_validation_manifest.csv')
    assert len(train)==5401 and len(validation)==1352
    assert not {r['group'] for r in train}&{r['group'] for r in validation}
    assert not {r['sha256'] for r in confirm}&{r['sha256'] for r in train+validation}


def test_routed_api_preserves_guard_comparison_and_reports_source_selection(client):
    stream=io.BytesIO();Image.fromarray(np.random.default_rng(12).integers(0,256,(100,80,3),dtype=np.uint8)).save(stream,format='PNG')
    response=client.post('/api/predict',files={'file':('synthetic.png',stream.getvalue(),'image/png')})
    assert response.status_code==200
    result=response.json();routing=result['domain_routing']
    assert routing['available'] and routing['selected_expert'] in ('baseline','rsna_adapted')
    assert 0<=routing['rsna_domain_score']<=1
    assert result['threshold']==.5 and result['comparison']['available']
    assert result['assessment']['clinically_validated'] is False
    if result['explanation']['available']:
        assert abs(result['raw_score']-result['explanation']['reconstructed_score'])<1e-5
