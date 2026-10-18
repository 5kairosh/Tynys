"""Describe quality-warning frequency on SHA-verified cached validation inputs.

No threshold fitting, image-quality labels, model inference, downloads or test
data are used. Frequencies are not sensitivity/specificity or clinical validation.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'site'))
from image_quality import check_image_quality, POLICY, THRESHOLDS, quality_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=ROOT/'training/cache/validation')
    parser.add_argument('--output', type=Path, default=ROOT/'training/quality_evidence/validation_summary.json')
    args = parser.parse_args()
    import tensorflow as tf

    with (ROOT/'training/manifests/validation_manifest.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Validation manifest is empty.')
    warnings, classes = Counter(), Counter()
    flagged = 0
    for index, row in enumerate(rows, 1):
        path = args.cache/(row['sha256']+Path(row['path']).suffix)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError(f'Validation hash mismatch at row {index}.')
        pixels = tf.io.decode_image(data, channels=3, expand_animations=False).numpy()
        result = check_image_quality(pixels)
        warnings.update(w['code'] for w in result['warnings'])
        classes[row['class']] += 1
        flagged += result['status'] == 'warning'
        if index % 100 == 0:
            print(f'Quality check: {index}/{len(rows)}', flush=True)
    report = {
        'checked_at': datetime.now(timezone.utc).isoformat(), 'policy': POLICY,
        'quality_source_sha256': hashlib.sha256((ROOT/'site/image_quality.py').read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256((ROOT/'training/manifests/validation_manifest.csv').read_bytes()).hexdigest(),
        'count': len(rows), 'sha256_verified': len(rows), 'classes': dict(classes),
        'with_warning': flagged, 'without_warning': len(rows)-flagged,
        'warning_fraction': flagged/len(rows), 'warning_counts': dict(warnings),
        'thresholds': {**THRESHOLDS, 'minimum_normalized_laplacian_variance': quality_reference()['detail_threshold']},
        'detail_reference_fitted': True, 'reference_percentile': 1.,
        'clinically_validated': False, 'quality_ground_truth_available': False,
        'false_positive_rate': None,
        'notice': 'Warning frequency on the reference data only. The detail cutoff is fitted at the fixed lower 1st percentile of eligible validation measurements; other thresholds are engineering constants. No image-quality ground truth: sensitivity, specificity and false-positive rate cannot be estimated.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
