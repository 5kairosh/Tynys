"""Stage a complete, internally bound deployment bundle after the frozen gate.

prepare does not change production files. activate backs up and copies only the
verified staged files. Evaluation labels never fit any deployment parameter.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from adapt_external_stage6 import (ROOT,OUT,CACHE,MODEL,CONFIG,Extractor,archived_features,
    digest,save_json,read_rows,balanced_domain_weights,calibrated)
from calibrate_ood import fit_reference, distances, calibrate_cutoff

sys.path.insert(0,str(ROOT/'site'))
from ood import OODDetector
from selective_evaluation import artifact_bindings, summarize, SCHEMA
from assessment import POLICY
from mcnemar_test import calculate_mcnemar
from ensemble_fit import metrics

BUNDLE=CACHE/'deployment_bundle'


def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def stage_json(relative,value):
    p=BUNDLE/relative;p.parent.mkdir(parents=True,exist_ok=True);save_json(p,value)


def prepare():
    gate=json.loads((OUT/'routing_evaluation.json').read_text());assert gate['promotion_passed'] and all(gate['promotion_gate'].values())
    fit=json.loads((OUT/'routing_fit_report.json').read_text())
    assert digest(MODEL)==fit['source_model_sha256'] and digest(CONFIG)==fit['source_config_sha256']
    candidate=CACHE/'routed_mobilenetv2.keras';assert digest(candidate)==fit['model_sha256']
    model_dir=BUNDLE/'site/model';model_dir.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(candidate,model_dir/'pneumonia_mobilenetv2.keras')
    shutil.copyfile(ROOT/'site/model/cnn_best.keras',model_dir/'cnn_best.keras')
    extractor=Extractor();tf=extractor.tf
    routed=tf.keras.models.load_model(candidate,compile=False,safe_mode=True)
    head=routed.layers[-1];config=json.loads(CONFIG.read_text())
    features={};rows={}
    for split in ('train','validation'):
        k,kx=archived_features(split);r,rx=extractor.rsna(split)
        features[split]=np.concatenate([kx,rx]);rows[split]=k+r
        manifest=[{'sha256':item['sha256'],'label':int(item['label']),
                   'group':item.get('group',item.get('patient_group')),
                   'source':'Kaggle' if i<len(k) else 'RSNA'} for i,item in enumerate(k+r)]
        write_rows(OUT/('ood_'+split+'_manifest.csv'),manifest)
    y=np.array([int(r['label']) for r in rows['validation']]);domains=np.array(['kaggle']*952+['rsna']*400)
    weight=balanced_domain_weights(domains,y)
    mobile=calibrated(head(features['validation'],training=False).numpy()[:,0],fit['temperature'])
    cnn=tf.keras.models.load_model(ROOT/'site/model/cnn_best.keras',compile=False,safe_mode=True)
    old_validation=read_rows(ROOT/'training/cache/validation/manifest.csv')
    with np.load(ROOT/'training/cache/validation_outputs.npz',allow_pickle=False) as cached:
        assert list(cached['hashes'])==[r['sha256'] for r in old_validation]
        cnn_lookup=dict(zip(cached['hashes'].tolist(),cached['CNN'].tolist()))
    cnn_parts=[np.array([cnn_lookup[r['sha256']] for r in rows['validation'][:952]])]
    rsna=rows['validation'][952:]
    for start in range(0,len(rsna),8):
        batch=[]
        for row in rsna[start:start+8]:
            p=CACHE/'images'/(row['source_id']+'.png');assert digest(p)==row['sha256']
            image=tf.io.decode_image(p.read_bytes(),channels=3,expand_animations=False)
            batch.append(tf.image.resize_with_pad(tf.cast(image,tf.float32),224,224))
        cnn_parts.append(cnn(tf.stack(batch),training=False).numpy()[:,0])
    cnn_prob=calibrated(np.concatenate(cnn_parts),config['model_calibrations']['CNN']['T'])
    objective=lambda w:float(np.average(-y*np.log(np.clip(w*cnn_prob+(1-w)*mobile,1e-7,1-1e-7))
                                      -(1-y)*np.log1p(-np.clip(w*cnn_prob+(1-w)*mobile,1e-7,1-1e-7)),weights=weight))
    result=minimize_scalar(objective,bounds=(.001,.999),method='bounded');assert result.success
    cnn_weight=float(result.x)
    ids={'CNN':digest(model_dir/'cnn_best.keras')[:12],'MobileNetV2':digest(candidate)[:12]}
    config.update(threshold=.5,selected_stage='domain_routed_frozen_experts',
                  calibration={'method':'temperature','T':fit['temperature']},
                  development_sources={'Kaggle':{'train':3801,'validation':952},'RSNA':{'train':1600,'validation':400}},
                  routing={'head':'domain_routed_head','source_logit_threshold':0.,
                           'selection':'training source similarity, not age or disease diagnosis',
                           'baseline_expert_model_sha256':fit['source_model_sha256'],
                           'rsna_expert_fit':'training/generalization_stage6/fit_report.json'},
                  dataset='Kaggle Chest X-Ray Pneumonia + RSNA-STR 2018 development subset')
    config['model_calibrations']['MobileNetV2']=config['calibration']
    config['ensemble']={'weights':{'CNN':cnn_weight,'MobileNetV2':1-cnn_weight},'threshold':.5,'validation_count':1352,
                        'selection':'domain/class balanced validation NLL; weights [0.001,0.999]; common operating margin threshold 0.5 fixed',
                        'model_ids':ids}
    # Exact config bytes are bound to every dependent artifact.
    stage_json('site/model/model_config.json',config)
    mean,precision=fit_reference(features['train'])
    val_distance=distances(features['validation'],mean,precision)
    cutoff,rank=calibrate_cutoff(val_distance,.95)
    metadata={'schema':2,'method':'train_validation_mahalanobis_v2','reference_split':'train','reference_count':5401,
              'calibration_split':'validation','calibration_count':1352,'target_acceptance':.95,'quantile_rank':rank,
              'model_sha256':digest(candidate),'config_sha256':digest(model_dir/'model_config.json'),
              'protocol_sha256':digest(OUT/'routing_protocol.json',True),
              'train_manifest_sha256':digest(OUT/'ood_train_manifest.csv',True),
              'validation_manifest_sha256':digest(OUT/'ood_validation_manifest.csv',True),
              'covariance_shrinkage':.1,'test_or_external_used_to_fit':False,
              'evaluation_cohorts_used_to_fit':False,'development_sources':['Kaggle train/validation','RSNA train/validation'],
              'clinical_or_formal_coverage_guarantee':False}
    np.savez_compressed(model_dir/'ood_params.npz',mean=mean,precision=precision,threshold=cutoff,
                        model_id=ids['MobileNetV2'],metadata=json.dumps(metadata,sort_keys=True))
    detector=OODDetector();detector.load(model_dir/'ood_params.npz',ids['MobileNetV2'],digest(model_dir/'model_config.json'))
    assert detector.status()['available']
    write_rows(OUT/'ood_validation_distances.csv',[{'sha256':r['sha256'],'distance':float(d),'accepted':int(d<=cutoff)} for r,d in zip(rows['validation'],val_distance)])
    save_json(OUT/'ood_fit_report.json',{**metadata,'threshold':cutoff,'parameter_sha256':digest(model_dir/'ood_params.npz'),
                                       'calibration_accepted':int(sum(val_distance<=cutoff)),
                                       'per_source_acceptance':{d:float(np.mean(val_distance[domains==d]<=cutoff)) for d in ('kaggle','rsna')}})
    cohorts={};metric_rows=[];external_models=None
    for name in ('internal_test','external_openi','rsna_confirmation'):
        if name=='rsna_confirmation':
            part,x=extractor.rsna('holdout_confirmation')
            prediction_rows=read_rows(OUT/'rsna_confirmation_routed_predictions.csv')
        else:
            part,x=archived_features(name)
            prediction_rows=read_rows(OUT/(name+'_routed_predictions.csv'))
        assert [r['sha256'] for r in part]==[r['sha256'] for r in prediction_rows]
        score=np.array([float(r['candidate_score']) for r in prediction_rows]);labels=np.array([int(r['label']) for r in part])
        evaluated=[{'label':int(label),'prediction':int(p>=.5),'xray_check':detector.check(vector)} for label,p,vector in zip(labels,score,x)]
        cohorts[name]=summarize(evaluated)
        if name!='rsna_confirmation':
            combined=[]
            for original,primary in zip(part,prediction_rows):
                row=dict(original);s=float(primary['candidate_score']);cnn_score=float(row['CNN_score'])
                ensemble=cnn_weight*cnn_score+(1-cnn_weight)*s
                row.update(MobileNetV2_score=s,MobileNetV2_prediction=int(s>=.5),score=s,prediction=int(s>=.5),
                           Ensemble_score=ensemble,Ensemble_prediction=int(ensemble>=.5),
                           routed_expert='rsna_adapted' if int(primary['rsna_branch']) else 'baseline')
                combined.append(row)
            relative='training/manifests/test_predictions.csv' if name=='internal_test' else 'training/external_predictions.csv'
            write_rows(BUNDLE/relative,combined)
            results=[metrics(labels,[float(r[n+'_score']) for r in combined],threshold,n) for n,threshold in
                     [('CNN',float(read_rows(ROOT/'site/model/test_metrics.csv')[0]['threshold'])),('MobileNetV2',.5),('Ensemble',.5)]]
            if name=='internal_test':metric_rows=results
            else:external_models=results
        print(name+' OOD '+json.dumps({'coverage':cohorts[name]['coverage'],'accepted':cohorts[name]['accepted']}),flush=True)
    write_rows(model_dir/'test_metrics.csv',metric_rows)
    stage_json('site/model/mcnemar.json',calculate_mcnemar(BUNDLE/'training/manifests/test_predictions.csv',model_dir))
    stage_json('site/model/external_validation.json',{'available':True,'models':external_models,
        'metrics':{**next(m for m in external_models if m['model']=='MobileNetV2'),'count':200,'cohort':'external'},
        'model_id':ids['MobileNetV2'],'config_sha256':digest(model_dir/'model_config.json'),
        'source':'OpenI Indiana, NLM API','normal_count':160,'pneumonia_count':40,
        'patient_grouping':'one report UID per sample; longitudinal patient IDs unverified',
        'threshold_tuned_external':False,'previously_inspected':True})
    baseline=json.loads((ROOT/'site/model/selective_evaluation.json').read_text())
    audit={'schema':SCHEMA,'policy':POLICY,'clinically_validated':False,'thresholds_tuned_on_evaluation':False,
           'bindings':artifact_bindings(model_dir),'classification_threshold':.5,'ood_threshold':cutoff,
           'ood_fitted_on':'Kaggle + RSNA development train reference; separate development validation cutoff; target .95',
           'ood_calibration':metadata,'evaluation_cohorts_previously_inspected':True,
           'fresh_confirmation_cohort':'rsna_confirmation; 400 original NIH patients',
           'comparison_to_stage5':{k:v for k,v in baseline.items() if k in ('policy','ood_threshold','cohorts')},
           'preprocessing':'RGB float32 resize_with_pad224; unchanged MNV2 backbone; frozen domain-routed head',
           'model':'MobileNetV2 domain-routed experts','cohorts':cohorts,
           'source_hash_encoding':'SHA256 UTF-8 logical LF text',
           'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p,True) for p in
                            [OUT/'routing_protocol.json',OUT/'routing_fit_report.json',OUT/'routing_evaluation.json',OUT/'ood_fit_report.json',Path(__file__)]},
           'limitations':['Only RSNA confirmation is fresh for 6B; all old cohorts were inspected.',
                          'RSNA radiographic opacity and OpenI weak report labels are not clinical diagnoses.',
                          'OOD pass is neither correctness nor X-ray authentication. Coverage and accepted error both reported.',
                          'Balanced RSNA quotas do not estimate deployment prevalence. Patient linkage across institutions unverified.'],
           'tensorflow_version':tf.__version__}
    stage_json('site/model/selective_evaluation.json',audit)
    save_json(OUT/'deployment_bundle.json',{'gate_passed':True,'candidate_model_sha256':digest(candidate),
              'config_sha256':digest(model_dir/'model_config.json'),'ood_parameter_id':digest(model_dir/'ood_params.npz')[:12],
              'ensemble':config['ensemble'],'files':{str(p.relative_to(BUNDLE)).replace('\\','/'):digest(p) for p in BUNDLE.rglob('*') if p.is_file()}})
    print('DEPLOYMENT_BUNDLE_READY '+str(BUNDLE),flush=True)


def activate():
    report=json.loads((OUT/'deployment_bundle.json').read_text());assert report['gate_passed']
    fit=json.loads((OUT/'routing_fit_report.json').read_text())
    assert digest(MODEL)==fit['source_model_sha256'] and digest(CONFIG)==fit['source_config_sha256']
    for name,expected in report['files'].items():assert digest(BUNDLE/name)==expected,name
    backup=CACHE/'production_stage5_backup'
    for name in report['files']:
        old=ROOT/name;target=backup/name
        if old.exists() and not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,target)
    for name in report['files']:
        source=BUNDLE/name;target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.stage6-update');shutil.copyfile(source,temporary);temporary.replace(target)
    print('BUNDLE_ACTIVATED Backups '+str(backup),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','activate'])
    args=parser.parse_args();prepare() if args.command=='prepare' else activate()
