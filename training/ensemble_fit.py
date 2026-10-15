"""Fit strictly positive convex weights and threshold ONLY on original validation."""
import hashlib
import json
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from prepare_validation_artifacts import ROOT, read_rows, write_rows, choose
from calibration import calibrate_score
from mcnemar_test import calculate_mcnemar


def metrics(y, p, threshold, name='Ensemble'):
    pred=np.asarray(p)>=threshold
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    return dict(model=name,threshold=threshold,accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),specificity=tn/(tn+fp),f1=f1_score(y,pred,zero_division=0),roc_auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),TN=tn,FP=fp,FN=fn,TP=tp)


def main():
    directory=ROOT/'site/model'; path=directory/'model_config.json'
    config=json.loads(path.read_text(encoding='utf-8'))
    rows=read_rows(ROOT/'training/cache/validation/manifest.csv')
    canonical=read_rows(ROOT/'training/manifests/validation_manifest.csv')
    assert len(rows)==len(canonical)==952
    assert {r['sha256'] for r in rows}=={r['sha256'] for r in canonical}
    ids={name:hashlib.sha256((directory/file).read_bytes()).hexdigest()[:12] for name,file in [('CNN','cnn_best.keras'),('MobileNetV2','pneumonia_mobilenetv2.keras')]}
    with np.load(ROOT/'training/cache/validation_outputs.npz',allow_pickle=False) as data:
        assert str(data['model_ids'])==json.dumps(ids,sort_keys=True)
        assert list(data['hashes'])==[r['sha256'] for r in rows]
        probs={name:np.array([calibrate_score(float(v),config['model_calibrations'][name]) for v in data[name]]) for name in ids}
    y=np.array([int(r['label']) for r in rows])
    def nll(a):
        p=np.clip(a*probs['CNN']+(1-a)*probs['MobileNetV2'],1e-7,1-1e-7)
        return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))
    fit=minimize_scalar(nll,bounds=(.001,.999),method='bounded')
    assert fit.success
    weight=float(fit.x);p=weight*probs['CNN']+(1-weight)*probs['MobileNetV2']
    ensemble={'weights':{'CNN':weight,'MobileNetV2':1-weight},'threshold':choose(y,p), 'validation_count':952,'selection':'validation NLL, weights bounded [0.001,0.999]; recall>=0.90 then specificity then F1','model_ids':ids}
    config['ensemble']=ensemble
    test=read_rows(ROOT/'training/manifests/test_predictions.csv')
    ty=np.array([int(r['label']) for r in test])
    tp=weight*np.array([float(r['CNN_score']) for r in test])+(1-weight)*np.array([float(r['MobileNetV2_score']) for r in test])
    for r,score in zip(test,tp):r.update(Ensemble_score=float(score),Ensemble_prediction=int(score>=ensemble['threshold']))
    report={'ensemble':ensemble,'validation_NLL':nll(weight),'validation_NLL_mean':nll(.5),'test':metrics(ty,tp,ensemble['threshold']),'external_tuning':False}
    base=[r for r in read_rows(directory/'test_metrics.csv') if r['model']!='Ensemble']
    path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    write_rows(directory/'test_metrics.csv',base+[report['test']])
    write_rows(ROOT/'training/manifests/test_predictions.csv',test)
    (directory/'mcnemar.json').write_text(json.dumps(calculate_mcnemar(ROOT/'training/manifests/test_predictions.csv',directory),ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'training/ensemble_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
