"""Controlled added-blur check on 20 reference images, not clinical validation."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'site'))
from image_quality import check_image_quality


def main():
    import tensorflow as tf
    with (ROOT/'training/manifests/validation_manifest.csv').open(encoding='utf-8') as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: row['sha256'])[:20]
    before, after, decreased = 0, 0, 0
    for row in rows:
        path = ROOT/'training/cache/validation'/(row['sha256']+Path(row['path']).suffix)
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row['sha256']
        pixels = tf.io.decode_image(data, channels=3, expand_animations=False).numpy()
        original = check_image_quality(pixels)
        copy = Image.fromarray(pixels)
        copy.thumbnail((512, 512), Image.Resampling.LANCZOS)
        blurred = check_image_quality(np.asarray(copy.filter(ImageFilter.GaussianBlur(4))))
        before += any(w['code'] == 'low_detail' for w in original['warnings'])
        after += any(w['code'] == 'low_detail' for w in blurred['warnings'])
        decreased += blurred['metrics']['normalized_laplacian_variance'] < original['metrics']['normalized_laplacian_variance']
    result = {'checked_at': datetime.now(timezone.utc).isoformat(), 'reference_images': len(rows),
              'sha256_verified': len(rows), 'transformation': 'thumbnail longest side 512, RGB GaussianBlur radius=4',
              'original_low_detail_warnings': before, 'added_blur_low_detail_warnings': after,
              'detail_measure_decreased': decreased, 'selection': 'first 20 validation SHA256 values in sorted order',
              'independent_of_reference_fitting': False, 'clinically_validated': False,
              'notice': 'Controlled artificial degradation of reference images only. No clinical image-quality labels or clinical sensitivity/specificity estimates.'}
    output = ROOT/'training/quality_evidence/controlled_blur.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
