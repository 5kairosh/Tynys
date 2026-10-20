"""Fit train-only OOD geometry, calibrate a prespecified cutoff on validation.

Never reads test/OpenI predictions or labels, or writes classifier weights/config.
Images/features remain in ignored cache. Run only with complete SHA-verified data.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'training/ood_stage5/protocol.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def fit_reference(vectors):
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError('Finite training feature matrix required.')
    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False)
    scale = max(float(np.trace(covariance) / len(mean)), 1e-8)
    covariance = .9 * covariance + .1 * scale * np.eye(len(mean))
    precision = np.linalg.solve(covariance, np.eye(len(mean)))
    return mean, (precision + precision.T) / 2


def distances(vectors, mean, precision):
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != mean.shape or not np.isfinite(values).all():
        raise ValueError('Invalid feature matrix.')
    delta = values - mean
    squared = np.einsum('ij,jk,ik->i', delta, precision, delta, optimize=True)
    if not np.isfinite(squared).all() or np.min(squared) < -1e-6:
        raise ValueError('Invalid Mahalanobis distances.')
    return np.sqrt(np.maximum(0, squared))


def calibrate_cutoff(calibration_distances, target=.95):
    values = np.asarray(calibration_distances, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.min(values) < 0 or not 0 < target < 1:
        raise ValueError('Finite nonnegative calibration distances and target in (0,1) required.')
    rank = math.ceil((len(values) + 1) * target)
    if rank > len(values):
        raise ValueError('Calibration set too small for the prespecified target.')
    threshold = float(np.sort(values)[rank - 1])
    if threshold <= 0:
        raise ValueError('Degenerate calibration threshold.')
    return threshold, rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--publish', action='store_true', help='Publish after train/validation fit; do not use holdout outcomes to select a candidate.')
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 32:
        raise ValueError('Batch size in [1,32] required.')
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    assert protocol['reference_count'] == 3801 and protocol['threshold_count'] == 952
    assert protocol['target_reference_acceptance'] == .95
    model_dir = ROOT / 'site/model'
    model_path, config_path = model_dir / 'pneumonia_mobilenetv2.keras', model_dir / 'model_config.json'
    model_sha, config_sha = sha(model_path), sha(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('routing', {}).get('head') == 'domain_routed_head':
        raise ValueError('This is the historical Kaggle-only fitter. Refusing to overwrite Stage6 combined-source OOD. See training/GENERALIZATION_STAGE6_KZ.txt and its frozen replay commits.')
    frames, local_paths = {}, {}
    for name in ('train', 'validation'):
        frames[name] = read_rows(ROOT / 'training/manifests' / (name + '_manifest.csv'))
        assert len(frames[name]) == protocol['reference_count' if name == 'train' else 'threshold_count']
        assert len({r['sha256'] for r in frames[name]}) == len(frames[name])
        local_paths[name] = []
        for row in frames[name]:
            path = ROOT / 'training/cache' / name / (row['sha256'] + Path(row['path']).suffix)
            if sha(path) != row['sha256']:
                raise ValueError('Source image SHA mismatch: ' + str(path))
            local_paths[name].append(path)
        print(f'{name}: {len(frames[name])} hashes verified', flush=True)
    for field in ('sha256', 'group'):
        assert not {r[field] for r in frames['train']} & {r[field] for r in frames['validation']}, field
    # The fixed split manifests were already audited for holdout separation.
    # This fitter deliberately never reads test/OpenI feature or score caches.
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
    end = next(i for i, layer in enumerate(model.layers)
               if isinstance(layer, tf.keras.Model) and layer.name.startswith('mobilenetv2'))

    @tf.function(input_signature=[tf.TensorSpec([None, *config['image_size'], 3], tf.float32)])
    def extract(batch):
        x = batch
        for layer in model.layers[1:end + 1]:
            x = layer(x, training=False)
        return tf.reduce_mean(x, axis=(1, 2))

    cache = ROOT / 'training/cache/ood_stage5'
    cache.mkdir(parents=True, exist_ok=True)
    key = json.dumps({'model_sha256': model_sha, 'config_sha256': config_sha,
                      'preprocessing': 'RGB-float32-resize_with_pad-224-MNV2-GAP-v1', 'tensorflow': tf.__version__}, sort_keys=True)
    features = {}
    for name, rows in frames.items():
        hashes = [r['sha256'] for r in rows]
        destination = cache / (name + '_features.npz')
        if destination.exists():
            with np.load(destination, allow_pickle=False) as saved:
                if str(saved['bindings']) == key and list(saved['hashes']) == hashes:
                    features[name] = saved['features']
        if name not in features:
            chunks = []
            for start in range(0, len(rows), args.batch_size):
                images = []
                for path in local_paths[name][start:start + args.batch_size]:
                    image = tf.io.decode_image(path.read_bytes(), channels=3, expand_animations=False)
                    images.append(tf.image.resize_with_pad(tf.cast(image, tf.float32), *config['image_size']))
                chunks.append(extract(tf.stack(images)).numpy())
                if start % 160 == 0 or start + args.batch_size >= len(rows):
                    print(f'{name}: features {min(start + args.batch_size, len(rows))}/{len(rows)}', flush=True)
            features[name] = np.concatenate(chunks)
            np.savez_compressed(destination, hashes=hashes, bindings=key, features=features[name])
        assert features[name].shape == (len(rows), 1280) and np.isfinite(features[name]).all()
    mean, precision = fit_reference(features['train'])
    val_distances = distances(features['validation'], mean, precision)
    threshold, rank = calibrate_cutoff(val_distances, protocol['target_reference_acceptance'])
    metadata = {'schema': 2, 'method': protocol['version'], 'reference_split': 'train',
                'reference_count': len(frames['train']), 'calibration_split': 'validation',
                'calibration_count': len(frames['validation']), 'target_acceptance': .95,
                'quantile_rank': rank, 'model_sha256': model_sha, 'config_sha256': config_sha,
                'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
                'train_manifest_sha256': hashlib.sha256((ROOT/'training/manifests/train_manifest.csv').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
                'validation_manifest_sha256': hashlib.sha256((ROOT/'training/manifests/validation_manifest.csv').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
                'covariance_shrinkage': .1, 'test_or_external_used_to_fit': False,
                'clinical_or_formal_coverage_guarantee': False}
    candidate = cache / 'ood_params.npz'
    np.savez_compressed(candidate, mean=mean, precision=precision, threshold=threshold,
                        model_id=model_sha[:12], metadata=json.dumps(metadata, sort_keys=True))
    summary = {**metadata, 'threshold': threshold, 'parameter_sha256': sha(candidate),
               'calibration_accepted': int(sum(val_distances <= threshold)),
               'calibration_flagged': int(sum(val_distances > threshold)),
               'classifier_config_sha256_unchanged': config_sha,
               'validation_distance_percentiles': {str(q): float(np.quantile(val_distances, q)) for q in (0,.5,.9,.95,.99,1)},
               'published': args.publish}
    output = ROOT / 'training/ood_stage5'
    with (output/'validation_distances.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sha256','distance','accepted'])
        writer.writeheader(); writer.writerows({'sha256': r['sha256'], 'distance': d, 'accepted': int(d<=threshold)} for r,d in zip(frames['validation'], val_distances))
    assert sha(model_path) == model_sha and sha(config_path) == config_sha, 'Classifier changed during fit.'
    if args.publish:
        sys.path.insert(0, str(ROOT/'site'))
        from ood import OODDetector
        detector = OODDetector(); detector.load(candidate, model_sha[:12], config_sha)
        assert detector.status()['available'], 'Candidate does not pass runtime validation.'
        target = model_dir / 'ood_params.npz'
        staged = target.with_suffix('.tmp'); staged.write_bytes(candidate.read_bytes()); staged.replace(target)
    (output/'fit_report.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
