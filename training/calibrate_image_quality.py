"""Fit only the lower-tail detail reference, not a clinical blur classifier.

Use all SHA-verified validation images with sufficient size and contrast.
The percentile is fixed at 1 before fitting; no disease labels or test images
are used. Quality labels are unavailable, so false-positive rate is unknown.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'site'))
from image_quality import measure_quality, POLICY, FEATURE_SCHEMA, THRESHOLDS


def main():
    import tensorflow as tf
    manifest = ROOT/'training/manifests/validation_manifest.csv'
    with manifest.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    values = []
    for index, row in enumerate(rows, 1):
        path = ROOT/'training/cache/validation'/(row['sha256']+Path(row['path']).suffix)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Validation hash mismatch at row {index}.')
        metrics = measure_quality(tf.io.decode_image(data, channels=3, expand_animations=False).numpy())
        if min(metrics['width'], metrics['height']) >= THRESHOLDS['minimum_side_px'] and metrics['contrast'] >= THRESHOLDS['minimum_p95_p05_contrast']:
            values.append(metrics['normalized_laplacian_variance'])
        if index % 200 == 0:
            print(f'Detail reference: {index}/{len(rows)}', flush=True)
    if len(values) < 100 or not np.isfinite(values).all():
        raise ValueError('Insufficient valid detail measurements.')
    config = {'policy': POLICY, 'feature_schema': FEATURE_SCHEMA,
              'detail_threshold': float(np.percentile(values, 1.)),
              'reference_percentile': 1., 'reference_count': len(rows), 'eligible_count': len(values),
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'fitted_at': datetime.now(timezone.utc).isoformat(),
              'clinically_validated': False, 'quality_ground_truth_available': False,
              'notice': 'Lower 1st percentile of eligible validation detail measurements. This is not a validated blur decision or a measured false-positive rate.'}
    target = ROOT/'site/model/image_quality_config.json'
    target.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
