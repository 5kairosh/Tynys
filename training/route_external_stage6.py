"""6B: deterministic source router between two frozen Dense experts.

Built entirely from standard Keras layers/ops; safe_mode=True needs no custom
objects, Lambda or code deserialization. Only `fit` reads development features.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import csv
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from adapt_external_stage6 import (ROOT,OUT,CACHE,MODEL,CONFIG,Extractor,archived_features,
    digest,save_json,read_rows,metrics,balanced_domain_weights,calibrated,dense_raw,paired_accuracy_interval)
from prepare_rsna_stage6 import select_cohort,metadata,ANNOTATIONS,MAPPING,download_images,write_csv


def prepare():
    protocol=json.loads((OUT/'protocol.json').read_text())
    protocol['rsna_counts_per_class']['holdout_confirmation']=200
    splits,_=select_cohort(metadata(ANNOTATIONS),metadata(MAPPING),protocol)
    for split in ('train','validation','holdout'):
        old=read_rows(OUT/(split+'_manifest.csv'))
        assert [r['source_id'] for r in splits[split]]==[r['source_id'] for r in old]
    new=splits['holdout_confirmation']
    old_patients={r['patient_group'] for name in ('train','validation','holdout') for r in read_rows(OUT/(name+'_manifest.csv'))}
    assert not old_patients.intersection(r['patient_group'] for r in new)
    selection=[{k:r[k] for k in ('source_id','patient_group','label','split')} for r in new]
    destination=OUT/'confirmation_selection.json'
    if destination.exists():assert json.loads(destination.read_text())==selection
    else:save_json(destination,selection)
    rows,_=download_images({'holdout_confirmation':new},6)
    write_csv(OUT/'holdout_confirmation_manifest.csv',rows['holdout_confirmation'])
    assert not {r['sha256'] for r in rows['holdout_confirmation']}.intersection(
        r['sha256'] for name in ('train','validation','holdout') for r in read_rows(OUT/(name+'_manifest.csv')))
    print('NEW_CONFIRMATION_READY 400 distinct NIH patients',flush=True)


def build_model(tf,baseline,old_kernel,old_bias,new_kernel,new_bias,router_kernel,router_bias):
    x=tf.keras.Input((1280,),name='gap_features')
    router=tf.keras.layers.Dense(1,name='source_domain_logit',trainable=False)
    old=tf.keras.layers.Dense(1,name='baseline_margin',trainable=False)
    new=tf.keras.layers.Dense(1,name='adapted_margin',trainable=False)
    z=tf.keras.ops.where(tf.keras.ops.greater_equal(router(x),0.),new(x),old(x))
    out=tf.keras.layers.Activation('sigmoid',name='probability')(z)
    head=tf.keras.Model(x,out,name='domain_routed_head')
    for layer,kernel,bias in [(router,router_kernel,router_bias),(old,old_kernel,old_bias),(new,new_kernel,new_bias)]:
        layer.set_weights([np.asarray(kernel,np.float32).reshape(-1,1),np.array([bias],np.float32)])
    inputs=tf.keras.Input((224,224,3),name='xray')
    value=inputs
    for layer in baseline.layers[1:-1]:value=layer(value,training=False)
    model=tf.keras.Model(inputs,head(value),name='tynys_domain_routed_mobilenetv2')
    model.trainable=False
    return model,head


def fit():
    if (OUT/'routing_fit_report.json').exists():raise ValueError('Router already frozen; cannot fit again after confirmation exposure.')
    original=json.loads((OUT/'fit_report.json').read_text())
    assert digest(CACHE/'candidate_head.npz')==original['candidate_head_sha256']
    extractor=Extractor();tf=extractor.tf
    development={}
    for split in ('train','validation'):
        k,kx=archived_features(split);r,rx=extractor.rsna(split)
        development[split]=(np.concatenate([kx,rx]),np.array([int(row['label']) for row in k+r]),np.array(['kaggle']*len(k)+['rsna']*len(r)))
    x,y,domains=development['train'];vx,vy,vd=development['validation']
    weights=balanced_domain_weights(domains,y)
    scaler=StandardScaler().fit(x,sample_weight=weights)
    classifier=LogisticRegression(C=.001,max_iter=2000,solver='lbfgs',random_state=20261021)
    classifier.fit(scaler.transform(x),(domains=='rsna').astype(int),sample_weight=weights)
    rk=(classifier.coef_[0]/scaler.scale_).astype(np.float32)
    rb=np.float32(classifier.intercept_[0]-np.dot(classifier.coef_[0],scaler.mean_/scaler.scale_))
    config=json.loads(CONFIG.read_text());old_k,old_b=extractor.model.layers[-1].get_weights()
    old_t=config['calibration']['T'];old_threshold=config['threshold']
    old_k=(old_k[:,0]/old_t).astype(np.float32)
    old_b=np.float32(float(old_b[0])/old_t-logit(old_threshold))
    with np.load(CACHE/'candidate_head.npz',allow_pickle=False) as a:
        new_k=(a['kernel']/float(a['temperature'])).astype(np.float32)
        new_b=np.float32(float(a['bias'])/float(a['temperature'])-logit(float(a['threshold'])))
    model,head=build_model(tf,extractor.model,old_k,old_b,new_k,new_b,rk,rb)
    raw=head(tf.convert_to_tensor(vx,tf.float32),training=False).numpy()[:,0]
    z=logit(np.clip(raw.astype(float),1e-7,1-1e-7));val_weights=balanced_domain_weights(vd,vy)
    objective=lambda t:float(np.average(np.logaddexp(0,z/t)-vy*z/t,weights=val_weights))
    result=minimize_scalar(objective,bounds=(.2,5),method='bounded');assert result.success
    temperature=float(result.x);score=calibrated(raw,temperature)
    candidate=CACHE/'routed_mobilenetv2.keras';model.save(candidate)
    reloaded=tf.keras.models.load_model(candidate,compile=False,safe_mode=True)
    loaded_head=reloaded.layers[-1]
    assert np.allclose(loaded_head(vx[:16],training=False).numpy()[:,0],raw[:16],rtol=0,atol=1e-6)
    router_score=expit((vx.astype(float)@rk)+float(rb));route=router_score>=.5
    validation={d:metrics(vy[vd==d],score[vd==d],.5) for d in ('kaggle','rsna')}
    routing={d:{'count':int(sum(vd==d)),'chosen_rsna':int(sum(route[vd==d])),
                'domain_accuracy':float(np.mean(route[vd==d]==(d=='rsna')))} for d in ('kaggle','rsna')}
    report={'stage':'6B','frozen_at':datetime.now(timezone.utc).isoformat(),
            'routing_protocol_sha256':digest(OUT/'routing_protocol.json',True),
            'base_fit_report_sha256':digest(OUT/'fit_report.json',True),
            'source_model_sha256':digest(MODEL),'source_config_sha256':digest(CONFIG),
            'model_sha256':digest(candidate),'temperature':temperature,'threshold':.5,
            'validation':validation,'validation_routing':routing,'validation_weighted_nll':float(result.fun),
            'router_C':.001,'router_iterations':int(classifier.n_iter_[0]),
            'router_selection':'Fixed C and >=0 domain logit, no hyperparameter search',
            'trained_on':'5401 train images only; domain/class balanced',
            'evaluation_data_used_to_fit':False,'production_model_changed':False,
            'confirmation_manifest_sha256':digest(OUT/'holdout_confirmation_manifest.csv',True)}
    save_json(OUT/'routing_fit_report.json',report)
    print('ROUTER_FROZEN '+json.dumps(report),flush=True)


def evaluate():
    destination=OUT/'routing_evaluation.json'
    if destination.exists():raise ValueError('Confirmation already opened. Do not retune this router.')
    report=json.loads((OUT/'routing_fit_report.json').read_text())
    assert digest(MODEL)==report['source_model_sha256'] and digest(CONFIG)==report['source_config_sha256']
    assert digest(OUT/'routing_protocol.json',True)==report['routing_protocol_sha256']
    assert digest(OUT/'holdout_confirmation_manifest.csv',True)==report['confirmation_manifest_sha256']
    candidate=CACHE/'routed_mobilenetv2.keras';assert digest(candidate)==report['model_sha256']
    extractor=Extractor();tf=extractor.tf
    model=tf.keras.models.load_model(candidate,compile=False,safe_mode=True);head=model.layers[-1]
    config=json.loads(CONFIG.read_text());bk,bb=extractor.model.layers[-1].get_weights()
    cohorts={}
    for name in ('rsna_confirmation','rsna_prior_holdout','internal_test','external_openi'):
        rows,x=(extractor.rsna('holdout_confirmation') if name=='rsna_confirmation' else
                extractor.rsna('holdout') if name=='rsna_prior_holdout' else archived_features(name))
        y=np.array([int(r['label']) for r in rows])
        old_scores=calibrated(dense_raw(tf,x,bk[:,0],bb[0]),config['calibration']['T'])
        raw=head(tf.convert_to_tensor(x,tf.float32),training=False).numpy()[:,0]
        new_scores=calibrated(raw,report['temperature'])
        old=metrics(y,old_scores,config['threshold']);new=metrics(y,new_scores,.5)
        rsna_probability=expit(head.get_layer('source_domain_logit')(x,training=False).numpy()[:,0])
        chosen=rsna_probability>=.5
        value={'baseline':old,'candidate':new,'accuracy_gain':new['accuracy']-old['accuracy'],
               'balanced_accuracy_gain':new['balanced_accuracy']-old['balanced_accuracy'],
               'fresh_holdout':name=='rsna_confirmation',
               'paired_accuracy_bootstrap':paired_accuracy_interval(y,old_scores>=config['threshold'],new_scores>=.5,20261021),
               'routing':{'baseline_expert':int(sum(~chosen)),'adapted_expert':int(sum(chosen))}}
        cohorts[name]=value
        numerical=[{'sha256':r['sha256'],'label':int(label),'baseline_score':float(a),'candidate_raw':float(raw_score),
                    'candidate_score':float(b),'candidate_prediction':int(b>=.5),'rsna_branch':int(branch),'rsna_domain_score':float(d)}
                   for r,label,a,b,raw_score,branch,d in zip(rows,y,old_scores,new_scores,raw,chosen,rsna_probability)]
        write_csv(OUT/(name+'_routed_predictions.csv'),numerical)
        print(name+' '+json.dumps(value),flush=True)
    r,k,o=[cohorts[n] for n in ('rsna_confirmation','internal_test','external_openi')]
    gates={'rsna_accuracy_gain':r['accuracy_gain']>=.05-1e-12,'rsna_balanced_gain':r['balanced_accuracy_gain']>=.05-1e-12,
           'rsna_sensitivity':r['candidate']['sensitivity']>=.80,'rsna_specificity':r['candidate']['specificity']>=.70,
           'paired_gain_lower_bound_positive':r['paired_accuracy_bootstrap']['lower_95']>0,
           'kaggle_balanced_noninferiority':k['balanced_accuracy_gain']>=-.02-1e-12,'kaggle_sensitivity':k['candidate']['sensitivity']>=.90,
           'openi_accuracy_gain':o['accuracy_gain']>=.05-1e-12,'openi_balanced_gain':o['balanced_accuracy_gain']>=.05-1e-12,
           'openi_sensitivity':o['candidate']['sensitivity']>=.60}
    result={'stage':'6B','evaluated_at':datetime.now(timezone.utc).isoformat(),'routing_fit_report_sha256':digest(OUT/'routing_fit_report.json',True),
            'cohorts':cohorts,'promotion_gate':gates,'promotion_passed':bool(all(gates.values())),
            'production_model_changed':False,'thresholds_tuned_on_evaluation':False,
            'clinical_validation':False,'notice':'Only 400-patient RSNA confirmation is fresh for 6B. All other cohorts were inspected before this design. RSNA opacity proxy and OpenI weak labels are not clinical diagnoses.'}
    save_json(destination,result)
    print('ROUTED_PROMOTION_GATE '+json.dumps(gates)+' PASSED='+str(result['promotion_passed']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','fit','evaluate'])
    args=parser.parse_args();{'prepare':prepare,'fit':fit,'evaluate':evaluate}[args.command]()


if __name__=='__main__':main()
