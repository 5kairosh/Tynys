"""Train a candidate on expanded development data; open new confirmation once.

Neither mode changes production files. Old evaluations cannot enter fitting.
Cached feature reuse requires identical saved backbone weights/preprocessing.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from adapt_external_stage6 import (ROOT, MODEL, CONFIG, Extractor, balanced_domain_weights,
    calibrated, digest, metrics, paired_accuracy_interval, read_rows, save_json)
from route_external_stage6 import build_model
from prepare_rsna_stage6 import write_csv

OUT = ROOT / 'training/data_expansion_20261023'
CACHE = ROOT / 'training/cache/data_expansion_20261023'
OLD_CACHE = ROOT / 'training/cache/external_stage6'


def backbone_digest(model, tf):
    value = hashlib.sha256()
    for layer in model.layers[1:]:
        if isinstance(layer, tf.keras.Model) and layer.name == 'domain_routed_head':
            break
        # Augmentation/dropout are disabled at inference; include the exact
        # deterministic normalization and feature extractor weights/config.
        if isinstance(layer, tf.keras.layers.Rescaling):
            config = layer.get_config()
            value.update(json.dumps({k:config[k] for k in ('scale', 'offset')}, sort_keys=True).encode())
        if isinstance(layer, tf.keras.Model) and layer.name.startswith('mobilenetv2'):
            for weight in layer.get_weights():
                value.update(str(weight.shape).encode()); value.update(str(weight.dtype).encode())
                value.update(np.ascontiguousarray(weight).tobytes())
            break
    return value.hexdigest()


class Features:
    def __init__(self):
        self.extractor = Extractor()
        self.tf, self.model = self.extractor.tf, self.extractor.model
        self.signature = backbone_digest(self.model, self.tf)
        old = OLD_CACHE / 'production_stage5_backup/site/model/pneumonia_mobilenetv2.keras'
        self.old_hash = digest(old)
        reference = self.tf.keras.models.load_model(old, compile=False, safe_mode=True)
        if backbone_digest(reference, self.tf) != self.signature:
            raise ValueError('Old feature caches belong to different backbone weights or normalization.')
        del reference
        CACHE.mkdir(parents=True, exist_ok=True)

    def previous(self, name):
        if name.startswith('kaggle_'):
            split = name.removeprefix('kaggle_')
            rows = read_rows(ROOT/'training/manifests'/(split+'_manifest.csv'))
            cache = ROOT/'training/cache/ood_stage5'/(split+'_features.npz')
        else:
            split = name.removeprefix('rsna_')
            rows = read_rows(ROOT/'training/generalization_stage6'/(split+'_manifest.csv'))
            cache = OLD_CACHE/('features_'+split+'.npz')
        with np.load(cache, allow_pickle=False) as saved:
            binding = json.loads(str(saved['bindings']))
            if binding.get('model', binding.get('model_sha256')) != self.old_hash:
                raise ValueError('Unrecognized original cache provenance.')
            if list(saved['hashes']) != [r['sha256'] for r in rows]:
                raise ValueError('Cache order does not match canonical patient manifest.')
            x = saved['features'].copy()
        if x.shape != (len(rows), 1280) or not np.isfinite(x).all():
            raise ValueError('Invalid cached features.')
        # Fresh image-based spot verification in addition to whole-backbone identity.
        for index in (0, len(rows)//2, len(rows)-1):
            path = image_path(rows[index], name)
            direct = self.images([rows[index]], [path], cache_name=None)[0]
            np.testing.assert_allclose(x[index], direct, rtol=0, atol=2e-5)
        return rows, x

    def images(self, rows, paths, cache_name):
        hashes = [r['sha256'] for r in rows]
        key = json.dumps({'backbone': self.signature, 'preprocessing': 'RGB-float32-resize_with_pad224-GAP-v1'}, sort_keys=True)
        path = CACHE/(cache_name+'.npz') if cache_name else None
        if path and path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if str(saved['bindings']) == key and list(saved['hashes']) == hashes:
                    return saved['features'].copy()
        parts = []
        for start in range(0, len(rows), 8):
            batch = []
            for row, source in zip(rows[start:start+8], paths[start:start+8]):
                raw = source.read_bytes()
                if hashlib.sha256(raw).hexdigest() != row['sha256']:
                    raise ValueError('Image changed after manifest freeze.')
                image = self.tf.io.decode_image(raw, channels=3, expand_animations=False)
                batch.append(self.tf.image.resize_with_pad(self.tf.cast(image, self.tf.float32), 224, 224))
            parts.append(self.extractor.forward(self.tf.stack(batch)).numpy())
            if cache_name and (start % 160 == 0 or start+8 >= len(rows)):
                print(f'{cache_name}: {min(start+8,len(rows))}/{len(rows)}', flush=True)
        x = np.concatenate(parts)
        if x.shape != (len(rows), 1280) or not np.isfinite(x).all():
            raise ValueError('Feature extraction failed.')
        if path:
            np.savez_compressed(path, hashes=hashes, bindings=key, features=x)
        return x


def image_path(row, cohort):
    if cohort.startswith('kaggle_') or cohort == 'internal_test':
        split = cohort.removeprefix('kaggle_') if cohort != 'internal_test' else 'test'
        return ROOT/'training/cache'/split/(row['sha256']+Path(row['path']).suffix)
    if cohort == 'external_openi':
        return ROOT/'training/cache/openi'/(hashlib.sha256(row['url'].encode()).hexdigest()+'.png')
    return OLD_CACHE/'images'/(row['source_id']+'.png')


def ensure_unopened(directory, mode):
    if (directory/'evaluation.json').exists():
        raise ValueError('Confirmation already opened; do not refit or evaluate it again.')
    if mode == 'fit' and (directory/'fit_report.json').exists():
        raise ValueError('Fit already frozen; cannot overwrite the selected candidate.')


def rank_candidate(y, logits, domains, c, threshold):
    parts = []
    for domain in ('kaggle', 'rsna'):
        mask = domains == domain
        prediction = logits[mask] >= 0
        labels = y[mask]
        recall = float(prediction[labels == 1].mean())
        specificity = float((~prediction[labels == 0]).mean())
        parts.append({'sensitivity': recall, 'specificity': specificity, 'balanced_accuracy': (recall+specificity)/2})
    feasible = parts[0]['sensitivity'] >= .90 and parts[1]['sensitivity'] >= .80
    rank = (feasible, min(p['balanced_accuracy'] for p in parts),
            np.mean([p['balanced_accuracy'] for p in parts]), -c, -abs(threshold-.5))
    return rank, parts


def fit():
    ensure_unopened(OUT, 'fit')
    protocol = json.loads((OUT/'protocol.json').read_text())
    features = Features(); tf = features.tf
    development = {}; manifests = {}
    for split in ('train', 'validation'):
        k, kx = features.previous('kaggle_'+split)
        r, rx = features.previous('rsna_'+split)
        new = read_rows(OUT/(split+'_manifest.csv'))
        nx = features.images(new, [image_path(row, 'rsna') for row in new], 'new_'+split)
        development[split] = (np.concatenate([kx, rx, nx]), np.array([int(row['label']) for row in k+r+new]),
                              np.array(['kaggle']*len(k)+['rsna']*(len(r)+len(new))))
        manifests[split] = {'kaggle': digest(ROOT/'training/manifests'/(split+'_manifest.csv'), True),
                            'old_rsna': digest(ROOT/'training/generalization_stage6'/(split+'_manifest.csv'), True),
                            'new_rsna': digest(OUT/(split+'_manifest.csv'), True)}
    x, y, domains = development['train']; vx, vy, vd = development['validation']
    weights = balanced_domain_weights(domains, y)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    scaled = scaler.transform(x)
    head = features.model.layers[-1]
    old_k, old_b = head.get_layer('baseline_margin').get_weights()
    router_k, router_b = head.get_layer('source_domain_logit').get_weights()
    old_logits = head.get_layer('baseline_margin')(vx, training=False).numpy()[:, 0]
    route = head.get_layer('source_domain_logit')(vx, training=False).numpy()[:, 0] >= 0
    best = None; candidates = []
    for c in protocol['candidate']['C']:
        classifier = LogisticRegression(C=c, max_iter=2000, solver='lbfgs', random_state=protocol['seed'])
        classifier.fit(scaled, y, sample_weight=weights)
        if int(classifier.n_iter_[0]) >= 2000:
            raise ValueError('Unconverged candidate: do not promote.')
        kernel = (classifier.coef_[0]/scaler.scale_).astype(np.float32)
        bias = np.float32(classifier.intercept_[0]-np.dot(classifier.coef_[0], scaler.mean_/scaler.scale_))
        projection = tf.linalg.matmul(tf.convert_to_tensor(vx, tf.float32), kernel[:, None]).numpy()[:, 0]
        for threshold in protocol['candidate']['expert_thresholds']:
            margin_bias = np.float32(bias-np.float32(logit(threshold)))
            logits = np.where(route, projection+margin_bias, old_logits)
            rank, parts = rank_candidate(vy, logits, vd, c, threshold)
            summary = {'C': c, 'expert_threshold': threshold, 'feasible': bool(rank[0]),
                       'kaggle_validation': parts[0], 'rsna_validation': parts[1]}
            candidates.append(summary)
            if best is None or rank > best[0]:
                best = (rank, kernel.copy(), margin_bias, summary)
        print(f'C={c}: fitted on {len(y)} training images', flush=True)
    rank, kernel, bias, summary = best
    model, new_head = build_model(tf, features.model, old_k[:,0], old_b[0], kernel, bias, router_k[:,0], router_b[0])
    raw = new_head(vx, training=False).numpy()[:, 0]
    z = logit(np.clip(raw.astype(float), 1e-7, 1-1e-7))
    val_weights = balanced_domain_weights(vd, vy)
    objective = lambda t: float(np.average(np.logaddexp(0, z/t)-vy*z/t, weights=val_weights))
    temperature_fit = minimize_scalar(objective, bounds=(.2, 5), method='bounded')
    if not temperature_fit.success:
        raise ValueError('Temperature fit failed.')
    temperature = float(temperature_fit.x)
    candidate = CACHE/'candidate.keras'; model.save(candidate)
    reloaded = tf.keras.models.load_model(candidate, compile=False, safe_mode=True)
    np.testing.assert_allclose(reloaded.layers[-1](vx[:32], training=False).numpy(), raw[:32, None], rtol=0, atol=1e-6)
    scores = calibrated(raw, temperature)
    report = {'frozen_at': datetime.now(timezone.utc).isoformat(),
        'training_code_sha256': digest(Path(__file__), True),
        'protocol_sha256': digest(OUT/'protocol.json', True), 'backbone_sha256': features.signature,
        'source_model_sha256': digest(MODEL), 'source_config_sha256': digest(CONFIG),
        'candidate_sha256': digest(candidate), 'candidate_path': str(candidate.relative_to(ROOT)).replace('\\', '/'),
        'selected_C': summary['C'], 'selected_expert_threshold': summary['expert_threshold'],
        'validation_feasible': bool(rank[0]), 'temperature': temperature, 'threshold': .5,
        'validation_weighted_nll': float(temperature_fit.fun), 'train_count': len(y), 'validation_count': len(vy),
        'validation': {d: metrics(vy[vd==d], scores[vd==d], .5) for d in ('kaggle', 'rsna')},
        'selection_candidates': candidates, 'manifests': manifests,
        'confirmation_manifest_sha256': digest(OUT/'confirmation_manifest.csv', True),
        'evaluation_used_to_fit_or_select': False, 'production_model_changed': False}
    save_json(OUT/'fit_report.json', report)
    print('CANDIDATE_FROZEN '+json.dumps({k:report[k] for k in ('selected_C','selected_expert_threshold','validation_feasible','temperature','train_count','validation_count')}), flush=True)


def promotion_gate(protocol, fitted, cohorts):
    rules = protocol['promotion_gate']
    r, k, o = (cohorts[name] for name in ('new_rsna_confirmation', 'internal_test', 'external_openi'))
    return {
        'validation_feasible': fitted['validation_feasible'],
        'fresh_accuracy_gain': r['accuracy_gain'] >= rules['new_confirmation_accuracy_gain_at_least']-1e-12,
        'fresh_sensitivity': r['candidate']['sensitivity'] >= rules['new_confirmation_sensitivity_at_least'],
        'fresh_specificity': r['candidate']['specificity'] >= rules['new_confirmation_specificity_at_least'],
        'fresh_paired_gain_lower_bound': r['paired_accuracy_bootstrap']['lower_95'] > rules['paired_bootstrap_95_lower_gain_above'],
        'kaggle_noninferiority': k['balanced_accuracy_gain'] >= -rules['old_kaggle_balanced_accuracy_drop_at_most']-1e-12,
        'kaggle_sensitivity': k['candidate']['sensitivity'] >= rules['old_kaggle_sensitivity_at_least'],
        'openi_accuracy_gain': o['accuracy_gain'] >= rules['old_openi_accuracy_gain_at_least']-1e-12,
        'openi_sensitivity': o['candidate']['sensitivity'] >= rules['old_openi_sensitivity_at_least'],
    }


def evaluate():
    ensure_unopened(OUT, 'evaluate')
    fitted = json.loads((OUT/'fit_report.json').read_text())
    protocol = json.loads((OUT/'protocol.json').read_text())
    if digest(OUT/'protocol.json', True) != fitted['protocol_sha256'] or digest(MODEL) != fitted['source_model_sha256'] or digest(CONFIG) != fitted['source_config_sha256']:
        raise ValueError('Protocol or baseline changed after freeze.')
    if digest(Path(__file__), True) != fitted['training_code_sha256']:
        raise ValueError('Training/evaluation implementation changed after candidate freeze.')
    if digest(OUT/'confirmation_manifest.csv', True) != fitted['confirmation_manifest_sha256']:
        raise ValueError('Frozen confirmation cohort changed.')
    candidate_path = ROOT/fitted['candidate_path']
    if digest(candidate_path) != fitted['candidate_sha256']:
        raise ValueError('Candidate changed after validation selection.')
    features = Features(); tf = features.tf
    candidate = tf.keras.models.load_model(candidate_path, compile=False, safe_mode=True)
    config = json.loads(CONFIG.read_text())
    cohorts = {}
    for name in ('new_rsna_confirmation', 'internal_test', 'external_openi', 'padchest_pilot'):
        if name == 'new_rsna_confirmation':
            rows = read_rows(OUT/'confirmation_manifest.csv')
            paths = [image_path(row, 'rsna') for row in rows]
        elif name == 'padchest_pilot':
            rows = read_rows(ROOT/'training/cache/padchest_pilot/manifest.csv')
            paths = [ROOT/row['path'] for row in rows]
        else:
            rows = read_rows(ROOT/('training/manifests/test_manifest.csv' if name == 'internal_test' else 'training/external_predictions.csv'))
            paths = [image_path(row, name) for row in rows]
        x = features.images(rows, paths, name)
        y = np.array([int(row['label']) for row in rows])
        old = calibrated(features.model.layers[-1](x, training=False).numpy()[:,0], config['calibration']['T'])
        new = calibrated(candidate.layers[-1](x, training=False).numpy()[:,0], fitted['temperature'])
        baseline, measured = metrics(y, old, config['threshold']), metrics(y, new, .5)
        comparison = {'baseline': baseline, 'candidate': measured,
            'accuracy_gain': measured['accuracy']-baseline['accuracy'],
            'balanced_accuracy_gain': measured['balanced_accuracy']-baseline['balanced_accuracy'],
            'paired_accuracy_bootstrap': paired_accuracy_interval(y, old>=config['threshold'], new>=.5, protocol['seed']),
            'fresh_patients': name in ('new_rsna_confirmation', 'padchest_pilot'),
            'new_institution': name == 'padchest_pilot', 'pilot_only': name == 'padchest_pilot'}
        cohorts[name] = comparison
        if name != 'padchest_pilot':
            write_csv(OUT/(name+'_predictions.csv'), [{'sha256':row['sha256'], 'label':int(label),
                'baseline_score':float(a), 'candidate_score':float(b), 'baseline_prediction':int(a>=config['threshold']),
                'candidate_prediction':int(b>=.5)} for row,label,a,b in zip(rows,y,old,new)])
        print(name+' '+json.dumps(comparison), flush=True)
    gates = promotion_gate(protocol, fitted, cohorts)
    report = {'evaluated_at': datetime.now(timezone.utc).isoformat(), 'fit_report_sha256': digest(OUT/'fit_report.json', True),
        'cohorts': cohorts, 'promotion_gate': gates, 'promotion_passed': bool(all(gates.values())),
        'production_model_changed': False, 'thresholds_tuned_on_evaluation': False, 'clinically_validated': False,
        'limitations': protocol['limitations']}
    save_json(OUT/'evaluation.json', report)
    print('PROMOTION_PASSED='+str(report['promotion_passed']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('fit', 'evaluate'))
    args = parser.parse_args()
    {'fit': fit, 'evaluate': evaluate}[args.mode]()
