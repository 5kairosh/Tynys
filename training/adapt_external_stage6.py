"""Frozen-backbone external adaptation, with a once-only held-out promotion gate.

--fit never loads evaluation features. --evaluate requires the frozen fit artifacts.
Nothing in this script changes the deployed classifier/config/OOD files.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'training/generalization_stage6'
CACHE = ROOT/'training/cache/external_stage6'
MODEL = ROOT/'site/model/pneumonia_mobilenetv2.keras'
CONFIG = ROOT/'site/model/model_config.json'


def digest(path, text=False):
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b'\r\n', b'\n') if text else data).hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def metrics(y, scores, threshold):
    y = np.asarray(y, int); scores = np.asarray(scores, float)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError('Both labels and finite scores required.')
    predictions = scores >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(y, predictions, labels=[0,1]).ravel())
    recall = tp/(tp+fn); specificity = tn/(tn+fp)
    return {'count': len(y), 'normal': tn+fp, 'positive': tp+fn,
            'accuracy': (tn+tp)/len(y), 'balanced_accuracy': (recall+specificity)/2,
            'sensitivity': recall, 'specificity': specificity,
            'precision': tp/(tp+fp) if tp+fp else 0., 'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'roc_auc': float(roc_auc_score(y,scores)), 'average_precision': float(average_precision_score(y,scores)),
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'errors': fp+fn, 'threshold': float(threshold)}


def balanced_domain_weights(domains, labels):
    domains = np.asarray(domains); labels = np.asarray(labels, int)
    weights = np.zeros(len(labels), float)
    for domain in np.unique(domains):
        for label in (0,1):
            mask = (domains == domain) & (labels == label)
            if not mask.any():
                raise ValueError('Every development domain needs both labels.')
            weights[mask] = 1./mask.sum()
    return weights * len(weights)/weights.sum()


def calibrated(raw, temperature):
    """Same clipping and temperature convention as site.calibration."""
    raw = np.asarray(raw, dtype=float)
    return raw.copy() if temperature == 1 else expit(logit(np.clip(raw,1e-7,1-1e-7))/temperature)


def dense_raw(tf, features, kernel, bias):
    x = tf.convert_to_tensor(features, dtype=tf.float32)
    z = tf.linalg.matmul(x,tf.reshape(tf.convert_to_tensor(kernel,dtype=tf.float32),[-1,1]))
    return tf.sigmoid(z+tf.cast(bias,tf.float32)).numpy()[:,0]


class Extractor:
    def __init__(self):
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        self.tf = tf
        self.model = tf.keras.models.load_model(MODEL, compile=False, safe_mode=True)
        index = next(i for i, layer in enumerate(self.model.layers)
                     if isinstance(layer, tf.keras.Model) and layer.name.startswith('mobilenetv2'))
        self.layers = self.model.layers[1:index+1]
        @tf.function(input_signature=[tf.TensorSpec([None,224,224,3],tf.float32)])
        def forward(images):
            for layer in self.layers:
                images = layer(images, training=False)
            return tf.reduce_mean(images, axis=(1,2))
        self.forward = forward

    def rsna(self, split):
        rows = read_rows(OUT/(split+'_manifest.csv'))
        hashes = [r['sha256'] for r in rows]
        key = json.dumps({'model': digest(MODEL), 'config': digest(CONFIG), 'preprocessing': 'RGB-float32-pad224-GAP-v1'}, sort_keys=True)
        path = CACHE/('features_'+split+'.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if str(saved['bindings']) == key and list(saved['hashes']) == hashes:
                    return rows, saved['features']
        features = []
        for start in range(0,len(rows),8):
            images = []
            for row in rows[start:start+8]:
                file = CACHE/'images'/(row['source_id']+'.png')
                if digest(file) != row['sha256']:
                    raise ValueError('Source image changed before extraction.')
                image = self.tf.io.decode_image(file.read_bytes(), channels=3, expand_animations=False)
                images.append(self.tf.image.resize_with_pad(self.tf.cast(image,self.tf.float32),224,224))
            features.append(self.forward(self.tf.stack(images)).numpy())
            if start%160 == 0 or start+8 >= len(rows):
                print(f'{split} features {min(start+8,len(rows))}/{len(rows)}',flush=True)
        value = np.concatenate(features)
        assert value.shape == (len(rows),1280) and np.isfinite(value).all()
        np.savez_compressed(path, bindings=key, hashes=hashes, features=value)
        return rows, value


def archived_features(name):
    if name in ('train','validation'):
        rows = read_rows(ROOT/'training/manifests'/(name+'_manifest.csv'))
        path = ROOT/'training/cache/ood_stage5'/(name+'_features.npz')
    else:
        rows = read_rows(ROOT/'training/manifests/test_predictions.csv' if name == 'internal_test'
                         else ROOT/'training/external_predictions.csv')
        path = ROOT/'training/cache/selective'/(name+'.npz')
    with np.load(path, allow_pickle=False) as data:
        binding = json.loads(str(data['bindings']))
        assert binding.get('model',binding.get('model_sha256')) == digest(MODEL)
        assert binding.get('config',binding.get('config_sha256')) == digest(CONFIG)
        assert list(data['hashes']) == [r['sha256'] for r in rows]
        features = data['features']
    assert features.shape == (len(rows),1280) and np.isfinite(features).all()
    return rows, features


def choose_threshold(y, score, domains):
    best = None
    for threshold in np.round(np.arange(.01,.99001,.005),6):
        parts = [metrics(y[domains==d],score[domains==d],threshold) for d in ('kaggle','rsna')]
        feasible = parts[0]['sensitivity'] >= .90 and parts[1]['sensitivity'] >= .80
        rank = (feasible, min(p['balanced_accuracy'] for p in parts),
                np.mean([p['balanced_accuracy'] for p in parts]), float(threshold))
        if best is None or rank > best[0]:
            best = rank, float(threshold), parts
    return best


def fit():
    if (OUT/'fit_report.json').exists():
        raise ValueError('Fit already frozen. Refusing a new selection after possible holdout exposure.')
    protocol = json.loads((OUT/'protocol.json').read_text())
    extractor = Extractor()
    development = {}
    for split in ('train','validation'):
        kaggle, kx = archived_features(split)
        rsna, rx = extractor.rsna(split)
        x = np.concatenate([kx,rx]); y = np.array([int(r['label']) for r in kaggle+rsna])
        domains = np.array(['kaggle']*len(kaggle)+['rsna']*len(rsna))
        development[split] = x,y,domains
    x,y,domain = development['train']; vx,vy,vd = development['validation']
    weights = balanced_domain_weights(domain,y)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    transformed = scaler.transform(x)
    summaries = []; selected = None
    for c in protocol['candidates']['C']:
        classifier = LogisticRegression(C=c,max_iter=protocol['candidates']['max_iter'],solver='lbfgs',random_state=protocol['seed'])
        classifier.fit(transformed,y,sample_weight=weights)
        # Fold standardization into Dense weights: original architecture and
        # Grad-CAM remain exact. Only this final kernel/bias would change.
        kernel = (classifier.coef_[0]/scaler.scale_).astype(np.float32)
        bias = np.float32(classifier.intercept_[0]-np.dot(classifier.coef_[0],scaler.mean_/scaler.scale_))
        validation_raw = dense_raw(extractor.tf,vx,kernel,bias)
        validation_logits = logit(np.clip(validation_raw.astype(float),1e-7,1-1e-7))
        rank, threshold, parts = choose_threshold(vy,validation_raw,vd)
        summary = {'C':c,'threshold':threshold,'feasible':bool(rank[0]),'kaggle_validation':parts[0],'rsna_validation':parts[1],
                   'iterations':int(classifier.n_iter_[0]),'converged':int(classifier.n_iter_[0])<protocol['candidates']['max_iter']}
        summaries.append(summary)
        candidate_rank = (*rank[:3],-c,threshold)
        if selected is None or candidate_rank > selected[0]:
            selected = candidate_rank,kernel,bias,validation_logits,summary
        print('Validation candidate '+json.dumps(summary),flush=True)
    _,kernel,bias,logits,summary = selected
    val_weights = balanced_domain_weights(vd,vy)
    objective = lambda temperature: float(np.average(np.logaddexp(0,logits/temperature)-vy*logits/temperature,weights=val_weights))
    result = minimize_scalar(objective,bounds=(.2,5),method='bounded')
    if not result.success or not np.isfinite(result.fun):
        raise ValueError('Temperature optimization failed.')
    temperature = float(result.x)
    threshold = float(expit(logit(summary['threshold'])/temperature))
    candidate = CACHE/'candidate_head.npz'
    np.savez_compressed(candidate,kernel=kernel,bias=bias,temperature=temperature,threshold=threshold,
                        source_model_sha256=digest(MODEL),source_config_sha256=digest(CONFIG))
    extractor.model.layers[-1].set_weights([kernel[:,None],np.array([bias],np.float32)])
    candidate_model = CACHE/'candidate_mobilenetv2.keras'
    extractor.model.save(candidate_model)
    report = {'stage':6,'frozen_at':datetime.now(timezone.utc).isoformat(),'seed':protocol['seed'],
              'protocol_sha256':digest(OUT/'protocol.json',True),'source_model_sha256':digest(MODEL),'source_config_sha256':digest(CONFIG),
              'candidate_head_sha256':digest(candidate),'candidate_model_sha256':digest(candidate_model),
              'selected_C':summary['C'],'validation_feasible':summary['feasible'],
              'threshold_before_temperature':summary['threshold'],'threshold':threshold,'temperature':temperature,
              'validation_candidates':summaries,'validation_weighted_nll':float(result.fun),
              'manifest_sha256':{name:digest(OUT/(name+'_manifest.csv'),True) for name in ('train','validation','holdout')},
              'fit_data':{'kaggle_train':3801,'rsna_train':1600,'kaggle_validation':952,'rsna_validation':400},
              'evaluation_data_used_for_fit_or_selection':False,'production_model_changed':False}
    save_json(OUT/'fit_report.json',report)
    print('HEAD_FROZEN '+json.dumps({k:report[k] for k in ['selected_C','threshold','temperature','validation_feasible','candidate_head_sha256']}),flush=True)


def paired_accuracy_interval(labels, old, new, seed, repeats=10000):
    labels = np.asarray(labels); delta = ((np.asarray(new)==labels).astype(float)-(np.asarray(old)==labels).astype(float))
    rng = np.random.default_rng(seed)
    values = np.concatenate([rng.choice(delta,size=(min(1000,repeats-start),len(delta)),replace=True).mean(axis=1)
                             for start in range(0,repeats,1000)])
    return {'difference':float(delta.mean()),'lower_95':float(np.quantile(values,.025)),
            'upper_95':float(np.quantile(values,.975)),'resamples':repeats,'seed':seed}


def evaluate():
    destination = OUT/'evaluation.json'
    if destination.exists():
        raise ValueError('Once-only holdout already opened; do not repeatedly adapt to this result.')
    report = json.loads((OUT/'fit_report.json').read_text())
    assert report['source_model_sha256']==digest(MODEL) and report['source_config_sha256']==digest(CONFIG)
    assert report['protocol_sha256']==digest(OUT/'protocol.json',True)
    assert all(report['manifest_sha256'][n]==digest(OUT/(n+'_manifest.csv'),True) for n in ('train','validation','holdout'))
    head = CACHE/'candidate_head.npz';assert digest(head)==report['candidate_head_sha256']
    with np.load(head,allow_pickle=False) as data:
        kernel,bias,temperature,threshold = data['kernel'],float(data['bias']),float(data['temperature']),float(data['threshold'])
    assert digest(CACHE/'candidate_mobilenetv2.keras')==report['candidate_model_sha256']
    extractor = Extractor(); baseline_kernel, baseline_bias = extractor.model.layers[-1].get_weights()
    config = json.loads(CONFIG.read_text()); old_t = config['calibration']['T']; old_threshold = config['threshold']
    cohorts = {}
    for name in ('rsna_holdout','internal_test','external_openi'):
        rows,x = extractor.rsna('holdout') if name=='rsna_holdout' else archived_features(name)
        y = np.array([int(r['label']) for r in rows])
        baseline = calibrated(dense_raw(extractor.tf,x,baseline_kernel[:,0],baseline_bias[0]),old_t)
        candidate = calibrated(dense_raw(extractor.tf,x,kernel,bias),temperature)
        if name!='rsna_holdout':
            assert [int(p>=old_threshold) for p in baseline]==[int(float(r['MobileNetV2_score'])>=old_threshold) for r in rows]
        old = metrics(y,baseline,old_threshold); new = metrics(y,candidate,threshold)
        comparison = {'baseline':old,'candidate':new,'accuracy_gain':new['accuracy']-old['accuracy'],
                      'balanced_accuracy_gain':new['balanced_accuracy']-old['balanced_accuracy'],
                      'paired_accuracy_bootstrap':paired_accuracy_interval(y,baseline>=old_threshold,candidate>=threshold,report['seed']),
                      'fresh_holdout':name=='rsna_holdout'}
        cohorts[name] = comparison
        numerical = [{'sha256':r['sha256'],'label':int(label),'baseline_score':float(a),'candidate_score':float(b),
                      'baseline_prediction':int(a>=old_threshold),'candidate_prediction':int(b>=threshold)}
                     for r,label,a,b in zip(rows,y,baseline,candidate)]
        with (OUT/(name+'_predictions.csv')).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(numerical[0]));writer.writeheader();writer.writerows(numerical)
        print(name+' '+json.dumps(comparison),flush=True)
    r,k,o = [cohorts[n] for n in ('rsna_holdout','internal_test','external_openi')]
    gates = {'validation_feasible':report['validation_feasible'],
             'rsna_accuracy_gain':r['accuracy_gain']>=.05-1e-12,
             'rsna_balanced_accuracy_gain':r['balanced_accuracy_gain']>=.05-1e-12,
             'rsna_sensitivity':r['candidate']['sensitivity']>=.80,
             'rsna_specificity':r['candidate']['specificity']>=.70,
             'rsna_paired_gain_lower_bound_positive':r['paired_accuracy_bootstrap']['lower_95']>0,
             'kaggle_balanced_accuracy_noninferiority':k['balanced_accuracy_gain']>=-.02-1e-12,
             'kaggle_sensitivity':k['candidate']['sensitivity']>=.90,
             'openi_accuracy_gain':o['accuracy_gain']>=.05-1e-12,
             'openi_balanced_accuracy_gain':o['balanced_accuracy_gain']>=.05-1e-12,
             'openi_sensitivity':o['candidate']['sensitivity']>=.60}
    value = {'stage':6,'evaluated_at':datetime.now(timezone.utc).isoformat(),'fit_report_sha256':digest(OUT/'fit_report.json',True),
             'cohorts':cohorts,'promotion_gate':gates,'promotion_passed':bool(all(gates.values())),
             'production_model_changed':False,'thresholds_tuned_on_evaluation':False,
             'notice':'RSNA quota-sampled radiographic proxy labels, not clinical pneumonia gold standard. Kaggle/OpenI previously inspected; only RSNA holdout is fresh. No re-fitting after opening it.'}
    save_json(destination,value)
    print('PROMOTION_GATE '+json.dumps(gates)+' PASSED='+str(value['promotion_passed']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['fit','evaluate']);args=parser.parse_args()
    fit() if args.command=='fit' else evaluate()


if __name__=='__main__':
    main()
