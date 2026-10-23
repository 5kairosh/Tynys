"""HTTP smoke checks using synthetic inputs only; never uploads private X-rays."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
from pypdf import PdfReader
import requests

from verify_docker_live import dicom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.url.rstrip('/')
    status_response = requests.get(base + '/api/status', timeout=45)
    status_response.raise_for_status()
    status = status_response.json()
    assert status['ready'] and status['comparison_ready']
    for name in ('dicom', 'xray_check', 'mcnemar', 'ensemble', 'selective_evaluation'):
        assert status[name]['available'], name
    html = requests.get(base + '/', timeout=15)
    html.raise_for_status()
    for element in ('analysis-progress', 'retry-analysis', 'replace-image'):
        assert f'id="{element}"' in html.text, element
    script_url = re.search(r'src="(/static/app\.js\?v=[a-f0-9]+)"', html.text)
    assert script_url, 'The browser entrypoint must reference a versioned script.'
    js = requests.get(base + script_url.group(1), timeout=15)
    js.raise_for_status()
    assert 'selectionRevision' in js.text and 'AbortSignal.any' in js.text
    pixels = np.random.default_rng(2026).integers(0, 256, (240, 192, 3), dtype=np.uint8)
    encoded = {}
    for fmt in ('PNG', 'JPEG', 'GIF'):
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format=fmt)
        encoded[fmt] = stream.getvalue()
    checks = []
    cases = [
        ('valid.png', encoded['PNG'], 200), ('valid.jpg', encoded['JPEG'], 200),
        ('cr.dcm', dicom('CR'), 200), ('dx.dcm', dicom('DX'), 200),
        ('ct.dcm', dicom('CT'), 400), ('text.jpg', b'not an image', 400),
        ('empty.png', b'', 400), ('spoof.jpg', encoded['PNG'], 400),
        ('spoof.png', encoded['JPEG'], 400), ('gif.png', encoded['GIF'], 400),
        ('truncated.jpg', encoded['JPEG'][:-20], 400),
        ('image.txt', encoded['PNG'], 400), ('large.png', b'x' * (12 * 1024 * 1024 + 1), 413),
    ]
    for name, data, expected in cases:
        response = requests.post(base + '/api/predict', files={'file': (name, data)}, timeout=190)
        assert response.status_code == expected, (name, response.status_code, response.text[:300])
        value = response.json()
        check = {'file': name, 'http_status': response.status_code}
        if expected == 200:
            assert value['comparison']['available'] and len(value['comparison']['models']) == 3
            assert value['explanation']['available'] and value['xray_check']['available']
            check.update(comparison_models=3, gradcam=True, ood=True)
        else:
            assert isinstance(value['detail'], str) and value['detail']
            check['message'] = value['detail']
        checks.append(check)
        print(f'{name}: {response.status_code}', flush=True)
    files = [('files', ('valid.png', encoded['PNG'])), ('files', ('cr.dcm', dicom('CR')))]
    response = requests.post(base + '/api/report', files=files, timeout=280)
    assert response.status_code == 200 and response.content.startswith(b'%PDF')
    assert len(PdfReader(io.BytesIO(response.content)).pages) == 2
    checks.append({'endpoint': '/api/report', 'http_status': 200, 'pages': 2})
    response = requests.post(base + '/api/report', files=[files[0], ('files', ('bad.jpg', b'bad'))], timeout=190)
    assert response.status_code == 400 and 'ашу' in response.json()['detail']
    checks.append({'endpoint': '/api/report', 'invalid_last_file': True, 'http_status': 400})
    evidence = {'checked_at': datetime.now(timezone.utc).isoformat(), 'url': base,
                'synthetic_inputs_only': True, 'private_images_uploaded': False,
                'model_id': status['model_id'], 'threshold': status['config']['threshold'],
                'frontend_revision_guard': True, 'checks': checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{len(checks)} HTTP checks passed; evidence: {args.output}', flush=True)


if __name__ == '__main__':
    main()
