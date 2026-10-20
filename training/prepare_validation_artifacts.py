"""Fit calibration/OOD on SHA-verified original validation without retraining.

Uses raw test scores from the original archived test_predictions.csv only for final
evaluation after fitting has ended. Does not use external data to fit or select.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'site'))
from calibration import calibrate_score
from mcnemar_test import calculate_mcnemar


def read_rows(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_rows(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def choose(y,prob):
    candidates=[]
    for threshold in np.linspace(0,1,201):
        pred=prob>=threshold
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        recall=tp/(tp+fn)
        if recall>=.9:
            candidates.append((tn/(tn+fp),f1_score(y,pred,zero_division=0),float(threshold)))
    return max(candidates)[2]


def main():
    model_dir=ROOT/'site/model'
    manifests=ROOT/'training/manifests'
    frames={k:read_rows(manifests/(k+'_manifest.csv')) for k in ('train','validation','test')}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        for key in ('sha256','group'):
            assert not {r[key] for r in frames[a]} & {r[key] for r in frames[b]},(a,b,key)
    cache=ROOT/'training/cache/validation'
    validation=read_rows(cache/'manifest.csv')
    assert {r['sha256'] for r in validation}=={r['sha256'] for r in frames['validation']}
    assert len(validation)==len(frames['validation'])==952
    canonical={r['sha256']:r for r in frames['validation']}
    assert all(r['label']==canonical[r['sha256']]['label'] and r['group']==canonical[r['sha256']]['group'] for r in validation)
    for row in validation:
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
    original=ROOT/'training/original_evaluation'
    original.mkdir(exist_ok=True)
    for name in ('model_config.json','test_metrics.csv'):
        if not (original/name).exists():
            shutil.copyfile(model_dir/name,original/name)
    if not (original/'test_predictions.csv').exists():
        shutil.copyfile(manifests/'test_predictions.csv',original/'test_predictions.csv')
    source_predictions=read_rows(original/'test_predictions.csv')
    assert [r['sha256'] for r in source_predictions]==[r['sha256'] for r in frames['test']]
    ids={name:hashlib.sha256((model_dir/file).read_bytes()).hexdigest()[:12]
         for name,file in [('CNN','cnn_best.keras'),('MobileNetV2','pneumonia_mobilenetv2.keras')]}
    # These fingerprints identify the weights distributed with the paired raw-score archive.
    assert ids=={'CNN':'19f707027db4','MobileNetV2':'ea584b998a45'},'Raw test-score provenance does not match model weights.'
    import tensorflow as tf
    raw_path=ROOT/'training/cache/validation_outputs.npz'
    if raw_path.exists():
        with np.load(raw_path,allow_pickle=False) as data:
            assert str(data['model_ids'])==json.dumps(ids,sort_keys=True)
            assert list(data['hashes'])==[r['sha256'] for r in validation]
            scores={name:data[name] for name in ids}
            vectors=data['features']
    else:
        mobile=tf.keras.models.load_model(model_dir/'pneumonia_mobilenetv2.keras',compile=False)
        cnn=tf.keras.models.load_model(model_dir/'cnn_best.keras',compile=False)
        layers=mobile.layers
        base_index=next(i for i,l in enumerate(layers) if isinstance(l,tf.keras.Model) and l.name.startswith('mobilenetv2'))
        @tf.function
        def infer(batch):
            x=batch
            for layer in layers[1:base_index+1]:
                x=layer(x,training=False)
            vector=tf.reduce_mean(x,axis=(1,2))
            for layer in layers[base_index+1:]:
                x=layer(x,training=False)
            return x,cnn(batch,training=False),vector
        outputs=[]
        for start in range(0,len(validation),16):
            images=[]
            for row in validation[start:start+16]:
                image=tf.io.decode_image(Path(row['path']).read_bytes(),channels=3,expand_animations=False)
                image.set_shape([None,None,3])
                images.append(tf.image.resize_with_pad(tf.cast(image,tf.float32),224,224))
            outputs.append(tuple(v.numpy() for v in infer(tf.stack(images))))
            print(f'Validation inference: {min(start+16,len(validation))}/{len(validation)}',flush=True)
        scores={'MobileNetV2':np.concatenate([o[0] for o in outputs]).ravel(),
                'CNN':np.concatenate([o[1] for o in outputs]).ravel()}
        vectors=np.concatenate([o[2] for o in outputs]).astype(np.float64)
        np.savez_compressed(raw_path,**scores,features=vectors,model_ids=json.dumps(ids,sort_keys=True),hashes=[r['sha256'] for r in validation])
    y=np.array([int(r['label']) for r in validation])
    calibrations={};thresholds={};diagnostics=[]
    for name,raw in scores.items():
        p=np.clip(raw.astype(np.float64),1e-7,1-1e-7)
        logits=np.log(p)-np.log1p(-p)
        def nll(T):
            z=logits/T
            return float(np.mean(np.logaddexp(0,z)-y*z))
        fit=minimize_scalar(nll,bounds=(.2,5),method='bounded')
        assert fit.success
        T=float(fit.x)
        calibrations[name]={'method':'temperature','T':T}
        thresholds[name]=choose(y,expit(logits/T))
        diagnostics.append(dict(model=name,T=T,threshold=thresholds[name],validation_count=len(y),NLL_raw=nll(1),NLL_calibrated=nll(T)))
    # Fit OOD separately AFTER config serialization; never reuse validation to
    # fit both its reference covariance and its distance cutoff.
    config=json.loads((original/'model_config.json').read_text(encoding='utf-8'))
    config.update(threshold=thresholds['MobileNetV2'],calibration=calibrations['MobileNetV2'],model_calibrations=calibrations)
    test_y=np.array([int(r['label']) for r in source_predictions]);metrics=[]
    predictions=[r.copy() for r in source_predictions]
    for name in ('CNN','MobileNetV2'):
        raw=np.array([float(r[name+'_score']) for r in source_predictions])
        prob=np.array([calibrate_score(float(p),calibrations[name]) for p in raw])
        pred=prob>=thresholds[name]
        tn,fp,fn,tp=map(int,confusion_matrix(test_y,pred,labels=[0,1]).ravel())
        metrics.append(dict(model=name,threshold=thresholds[name],accuracy=accuracy_score(test_y,pred),precision=precision_score(test_y,pred,zero_division=0),
                            recall=recall_score(test_y,pred,zero_division=0),specificity=tn/(tn+fp),f1=f1_score(test_y,pred,zero_division=0),
                            roc_auc=roc_auc_score(test_y,prob),average_precision=average_precision_score(test_y,prob),TN=tn,FP=fp,FN=fn,TP=tp))
        for row,p,pr in zip(predictions,prob,pred):
            row[name+'_score']=p;row[name+'_prediction']=int(pr)
    # All fitting has finished. Publish matching config, thresholds and paired test predictions.
    (model_dir/'model_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    write_rows(model_dir/'test_metrics.csv',metrics)
    write_rows(ROOT/'training/calibration_validation.csv',diagnostics)
    write_rows(manifests/'test_predictions.csv',predictions)
    paired=calculate_mcnemar(manifests/'test_predictions.csv',model_dir)
    (model_dir/'mcnemar.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'training/calibrate_ood.py'),'--publish'],check=True)
    fitted_ood=json.loads((ROOT/'training/ood_stage5/fit_report.json').read_text(encoding='utf-8'))
    report={'validation_calibration':diagnostics,'ood':{**fitted_ood,'validation_count':len(y),'feature_dimension':1280,'percentile':95,'non_xray_validated':False},'test_metrics':metrics,'mcnemar':paired,
            'provenance':'Original Oct13 models and SHA-verified validation. Archived raw test scores evaluated once after fitting.'}
    (ROOT/'training/calibration_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (model_dir/'project_summary.txt').write_text('Validation temperature scaling және OOD есептелді. Оқу прототипі.\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n90% accuracy мақсаты: '+('жетті' if metrics[1]['accuracy']>=.9 else 'жетпеді')+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
