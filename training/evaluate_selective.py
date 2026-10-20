"""Measure fixed ood_guard_v1 coverage/errors on SHA-verified held-out images.

No training, threshold search or parameter writes. Raw features/images are cached
only under ignored training/cache. Public output contains numerical aggregates
and image hashes, never image pixels. Re-run after source/parameter changes.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'site'))
import numpy as np
from assessment import POLICY, assess_input
from calibration import calibrate_score
from ood import OODDetector
from selective_evaluation import SCHEMA, artifact_bindings, summarize


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 32:
        raise ValueError('Batch size must be in [1,32].')
    model_dir = ROOT / 'site/model'
    bindings = artifact_bindings(model_dir)
    config = json.loads((model_dir / 'model_config.json').read_text(encoding='utf-8'))
    detector = OODDetector()
    detector.load(model_dir / 'ood_params.npz', bindings['model'][:12])
    if not detector.status()['available']:
        raise ValueError('OOD parameters unavailable: no evaluation published.')
    frames = {key: read_rows(ROOT / 'training/manifests' / (key + '_manifest.csv'))
              for key in ('train', 'validation', 'test')}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        for field in ('sha256', 'group'):
            assert not {r[field] for r in frames[a]} & {r[field] for r in frames[b]}, (a, b, field)
    internal = read_rows(ROOT / 'training/manifests/test_predictions.csv')
    assert [(r['sha256'], r['label']) for r in internal] == [(r['sha256'], r['label']) for r in frames['test']]
    external = read_rows(ROOT / 'training/external_predictions.csv')
    cohorts = {'internal_test': internal, 'external_openi': external}
    all_internal = {r['sha256'] for rows in frames.values() for r in rows}
    assert not all_internal & {r['sha256'] for r in external}
    paths = {}
    for name, rows in cohorts.items():
        assert len({r['sha256'] for r in rows}) == len(rows), 'Duplicate image hash.'
        if name == 'external_openi':
            assert len({r['patient_id'] for r in rows}) == len(rows), 'Duplicate external report group.'
        paths[name] = []
        for row in rows:
            path = (ROOT / 'training/cache/test' / (row['sha256'] + Path(row['path']).suffix)
                    if name == 'internal_test' else ROOT / 'training/cache/openi' / (hashlib.sha256(row['url'].encode()).hexdigest() + '.png'))
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Image SHA-256 mismatch: ' + str(path))
            paths[name].append(path)
        print(f'{name}: {len(rows)} image hashes verified', flush=True)

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    model = tf.keras.models.load_model(model_dir / 'pneumonia_mobilenetv2.keras', compile=False, safe_mode=True)
    base_index = next(i for i, layer in enumerate(model.layers)
                      if isinstance(layer, tf.keras.Model) and layer.name.startswith('mobilenetv2'))

    @tf.function(input_signature=[tf.TensorSpec([None, *config['image_size'], 3], tf.float32)])
    def forward(batch):
        x = batch
        for layer in model.layers[1:base_index + 1]:
            x = layer(x, training=False)
        features = tf.reduce_mean(x, axis=(1, 2))
        for layer in model.layers[base_index + 1:]:
            x = layer(x, training=False)
        return x[:, 0], features

    cache = ROOT / 'training/cache/selective'
    cache.mkdir(parents=True, exist_ok=True)
    output = ROOT / 'training/selective_evaluation'
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, rows in cohorts.items():
        hashes = [r['sha256'] for r in rows]
        # Features depend on weights and preprocessing; bind the whole API source
        # conservatively. A changed source causes a fresh forward pass.
        key = json.dumps({k: bindings[k] for k in ('model', 'config', 'api_pipeline')}, sort_keys=True)
        cache_path = cache / (name + '.npz')
        reused = False
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as saved:
                if str(saved['bindings']) == key and list(saved['hashes']) == hashes:
                    raw_scores, vectors = saved['scores'], saved['features']
                    reused = True
        if not reused:
            raw_parts, feature_parts = [], []
            for start in range(0, len(rows), args.batch_size):
                images = []
                for path in paths[name][start:start + args.batch_size]:
                    image = tf.io.decode_image(path.read_bytes(), channels=3, expand_animations=False)
                    images.append(tf.image.resize_with_pad(tf.cast(image, tf.float32), *config['image_size']))
                scores, features = forward(tf.stack(images))
                raw_parts.append(scores.numpy()); feature_parts.append(features.numpy())
                if start % 80 == 0 or start + args.batch_size >= len(rows):
                    print(f'{name}: forward {min(start + args.batch_size, len(rows))}/{len(rows)}', flush=True)
            raw_scores, vectors = np.concatenate(raw_parts), np.concatenate(feature_parts)
            np.savez_compressed(cache_path, bindings=key, hashes=hashes, scores=raw_scores, features=vectors)
        evaluated, numerical = [], []
        max_score_delta = 0.
        for i, row in enumerate(rows):
            score = calibrate_score(float(raw_scores[i]), config.get('calibration'))
            if not np.isfinite(score) or not np.isfinite(vectors[i]).all():
                raise ValueError('Non-finite inference output.')
            prediction = int(score >= config['threshold'])
            check = detector.check(vectors[i])  # exact live API OOD implementation
            assessment = assess_input(check)    # exact live API disposition
            stored_score = float(row['MobileNetV2_score'])
            max_score_delta = max(max_score_delta, abs(stored_score - score))
            if prediction != int(stored_score >= config['threshold']) or abs(score - stored_score) > 1e-4:
                raise ValueError('Forward pass differs from archived prediction: ' + row['sha256'])
            evaluated.append({'label': int(row['label']), 'prediction': prediction, 'xray_check': check})
            numerical.append({'sha256': row['sha256'], 'label': int(row['label']), 'prediction': prediction,
                              'score': score, 'distance': check['distance'], 'ood_threshold': check['threshold'],
                              'disposition': assessment['status'], 'reason': assessment['reason']})
        summaries[name] = {**summarize(evaluated), 'max_archived_score_delta': max_score_delta}
        target = output / (name + '.csv')
        with target.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(numerical[0]))
            writer.writeheader(); writer.writerows(numerical)
        print(name + ': ' + json.dumps(summaries[name]), flush=True)
    report = {'schema': SCHEMA, 'policy': POLICY, 'clinically_validated': False,
              'thresholds_tuned_on_evaluation': False, 'bindings': bindings,
              'classification_threshold': config['threshold'], 'ood_threshold': detector.params[2],
              'ood_fitted_on': 'validation (952 images); 95th percentile on fitting data',
              'preprocessing': 'decode_image RGB; float32; resize_with_pad 224x224; trained model preprocessing',
              'model': 'MobileNetV2', 'cohorts': summaries,
              'source_hash_encoding': 'SHA256 of UTF-8 text with CRLF normalized to LF',
              'source_sha256': {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                  for path in [ROOT/'training/manifests/test_manifest.csv', ROOT/'training/manifests/test_predictions.csv',
                               ROOT/'training/external_predictions.csv', Path(__file__), *sorted(output.glob('*.csv'))]},
              'limitations': ['Fixed-policy descriptive audit; no causal/clinical reliability claim.',
                  'Accepted-subset accuracy has a different denominator from full-cohort accuracy.',
                  'OOD acceptance is not X-ray authentication or a guarantee of correctness.',
                  'Filename/report UID grouping is not verified longitudinal patient identity.',
                  'OpenI is a selected small cohort with weak report-level MeSH labels, not a clinical gold standard.',
                  'Only MobileNetV2 binary predictions audited; image quality warnings are advisory and do not gate.',
                  'Validation fitted the OOD boundary and is not an independent evaluation cohort.'],
              'tensorflow_version': tf.__version__}
    if artifact_bindings(model_dir) != bindings:
        raise ValueError('Source/parameters changed during evaluation; refusing stale publication.')
    destination = model_dir / 'selective_evaluation.json'
    staged = destination.with_suffix('.tmp')
    staged.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    staged.replace(destination)
    print('Published ' + str(destination), flush=True)


if __name__ == '__main__':
    main()
