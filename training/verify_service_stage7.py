"""Small bounded burst against an explicitly named running Tynys server.

Generated engineering pixels only. Never reads or uploads medical source files.
Writes aggregate timings/statuses, not inputs or inference results. Use on an
owned demo instance, with no other users testing it at the same time.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import statistics
import threading
import time

import httpx
from PIL import Image
from pypdf import PdfReader


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--requests', type=int, default=8, choices=range(4, 13))
    args = parser.parse_args()
    base = args.base_url.rstrip('/')
    image = Image.new('RGB', (96, 128))
    image.putdata([((x * 13 + y * 7) % 256,) * 3 for y in range(128) for x in range(96)])
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    pixels = stream.getvalue()
    samples, rows = [], []
    stopped = threading.Event()
    barrier = threading.Barrier(args.requests)

    def poll():
        with httpx.Client(base_url=base, timeout=5, trust_env=False) as client:
            while not stopped.is_set():
                start = time.perf_counter()
                try:
                    response = client.get('/api/health')
                    samples.append({'status': response.status_code,
                                    'ms': round((time.perf_counter() - start) * 1000, 2),
                                    'service': response.json()['service']})
                except Exception as error:
                    samples.append({'error': type(error).__name__})
                stopped.wait(.05)

    def request(index):
        with httpx.Client(base_url=base, timeout=290, trust_env=False) as client:
            barrier.wait(timeout=15)
            start = time.perf_counter()
            report = index == 0
            route = '/api/report' if report else '/api/predict'
            files = [('files', ('control-a.png', pixels, 'image/png')),
                     ('files', ('control-b.png', pixels, 'image/png'))] if report else {
                         'file': ('engineering-control.png', pixels, 'image/png')}
            response = client.post(route, files=files)
            row = {'route': route, 'status': response.status_code,
                   'ms': round((time.perf_counter() - start) * 1000, 2)}
            if response.status_code == 200:
                if report:
                    assert response.content.startswith(b'%PDF')
                    row['pdf_pages'] = len(PdfReader(io.BytesIO(response.content)).pages)
                    assert row['pdf_pages'] == 2
                else:
                    value = response.json()
                    row.update(model_id=value['model_id'], comparison=value['comparison']['available'],
                               gradcam=value['explanation']['available'])
                    assert row['comparison'] and row['gradcam']
            else:
                assert response.status_code == 429, response.text
                assert response.headers.get('retry-after') == '5'
            return row

    with httpx.Client(base_url=base, timeout=30, trust_env=False) as client:
        initial = client.get('/api/status').json()
        assert initial['ready'] and initial['comparison_ready']
        assert initial['service']['policy'] == 'bounded_inference_v1'
        assert initial['service']['admitted'] == 0, 'Run on an idle owned instance.'
        assert client.get('/api/ready').status_code == 200
        monitoring = threading.Thread(target=poll, daemon=True)
        monitoring.start()
        try:
            with ThreadPoolExecutor(max_workers=args.requests) as pool:
                rows = list(pool.map(request, range(args.requests)))
        finally:
            stopped.set()
            monitoring.join(timeout=6)
        assert any(row['status'] == 200 for row in rows)
        assert any(row['status'] == 429 for row in rows), 'Burst did not exercise backpressure.'
        # Test PDF explicitly even if its burst request was declined.
        pdf = client.post('/api/report', files=[('files', ('a.png', pixels, 'image/png')),
                                               ('files', ('b.png', pixels, 'image/png'))], timeout=280)
        assert pdf.status_code == 200 and pdf.content.startswith(b'%PDF')
        pages = len(PdfReader(io.BytesIO(pdf.content)).pages)
        assert pages == 2
        recovered = client.post('/api/predict', files={'file': ('recovery.png', pixels, 'image/png')}, timeout=190)
        assert recovered.status_code == 200 and recovered.json()['model_id'] == initial['model_id']
        final = client.get('/api/health').json()
        assert final['service']['active'] == final['service']['admitted'] == 0
    assert samples and all(s.get('status') == 200 for s in samples), samples
    assert max(s['service']['active'] for s in samples) <= 1
    assert max(s['service']['admitted'] for s in samples) <= 3
    latencies = [s['ms'] for s in samples]
    assert percentile(latencies, .95) < 2000, 'Health p95 exceeded the demo target.'
    report = {'stage': 7, 'checked_at': datetime.now(timezone.utc).isoformat(), 'base_url': base,
              'generated_pixels_only': True, 'medical_accuracy_test': False,
              'model_id': initial['model_id'], 'burst': rows,
              'accepted': sum(r['status'] == 200 for r in rows),
              'busy_rejected': sum(r['status'] == 429 for r in rows),
              'health': {'samples': len(samples), 'all_200': True,
                         'median_ms': round(statistics.median(latencies), 2),
                         'p95_ms': percentile(latencies, .95), 'max_ms': max(latencies),
                         'peak_active': max(s['service']['active'] for s in samples),
                         'peak_admitted': max(s['service']['admitted'] for s in samples)},
              'pdf_pages_after_burst': pages, 'recovery_predict': 200,
              'final_service': final['service'],
              'limits': 'Short demo burst, not a sustained capacity or 24/7 availability guarantee.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
