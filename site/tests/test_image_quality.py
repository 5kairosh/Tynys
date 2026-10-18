import io
import json

import numpy as np
import pytest
from PIL import Image, ImageFilter
from pypdf import PdfReader

from image_quality import check_image_quality, unavailable_quality


def rgb(gray):
    return np.repeat(np.asarray(gray, dtype=np.uint8)[..., None], 3, axis=2)


def codes(result):
    return {w['code'] for w in result['warnings']}


def checkerboard(size=256):
    y, x = np.indices((size, size))
    return rgb(40+((x//16+y//16) % 2)*175)


def test_blurring_removes_detail_without_changing_input():
    source = checkerboard()
    saved = source.copy()
    sharp = check_image_quality(source)
    blurred_pixels = np.asarray(Image.fromarray(source).filter(ImageFilter.GaussianBlur(3)))
    blurred = check_image_quality(blurred_pixels)
    assert codes(sharp) == set()
    assert 'low_detail' in codes(blurred)
    assert blurred['metrics']['contrast'] >= .1
    assert blurred['metrics']['normalized_laplacian_variance'] < sharp['metrics']['normalized_laplacian_variance']
    np.testing.assert_array_equal(source, saved)


@pytest.mark.parametrize('pixels,expected', [
    (rgb(np.full((256, 256), 5)), {'very_dark', 'low_contrast'}),
    (rgb(np.full((256, 256), 250)), {'very_bright', 'low_contrast'}),
    (rgb(np.full((256, 256), 128)), {'low_contrast'}),
    (checkerboard(128), {'low_resolution'}),
    (np.tile(checkerboard(), (1, 3, 1)), {'unusual_shape'}),
])
def test_observable_quality_problems(pixels, expected):
    result = check_image_quality(pixels)
    assert codes(result) == expected
    assert result['status'] == 'warning'
    assert result['clinically_validated'] is False
    assert result['affects_model_score'] is False
    assert result['anatomical_crop_checked'] is False
    assert result['policy'] == 'image_quality_v1'
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('shape', [(1, 1), (2, 17), (2048, 1536)])
def test_small_and_large_inputs_have_finite_bounded_metrics(shape):
    pixels = np.random.default_rng(12).integers(0, 256, (*shape, 3), dtype=np.uint8)
    result = check_image_quality(pixels)
    assert max(result['metrics']['analysis_width'], result['metrics']['analysis_height']) <= 512
    json.dumps(result, allow_nan=False)


def test_api_quality_does_not_change_model_outputs_and_report_includes_it(client, monkeypatch):
    import app

    # Even with OOD in-reference, a quality warning remains advisory.
    monkeypatch.setattr(app.engine.ood, 'status', lambda: {'available': True})
    monkeypatch.setattr(app.engine.ood, 'check', lambda vector: {'available': True, 'is_xray': True})
    raw = io.BytesIO()
    Image.new('RGB', (256, 256), (5, 5, 5)).save(raw, format='PNG')
    files = {'file': ('dark.png', raw.getvalue())}
    response = client.post('/api/predict', files=files)
    assert response.status_code == 200
    result = response.json()
    assert codes(result['quality']) == {'very_dark', 'low_contrast'}
    assert result['assessment']['status'] == 'classified'
    report = client.post('/api/report', files=[('files', ('dark.png', raw.getvalue()))])
    assert report.status_code == 200, report.text
    reader = PdfReader(io.BytesIO(report.content))
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    assert 'Сурет сапасы:' in text and 'тым қараңғы' in text and 'контрасты төмен' in text
    monkeypatch.setattr(app, 'check_image_quality', lambda pixels: unavailable_quality())
    after = client.post('/api/predict', files=files).json()
    for key in ['raw_score', 'pneumonia_score', 'threshold', 'model_class', 'assessment', 'comparison']:
        assert after[key] == result[key]
    assert after['quality']['available'] is False


def test_quality_failure_does_not_break_prediction(client, monkeypatch):
    import app

    def broken_check(pixels):
        raise ValueError('simulated optional quality-check failure')

    monkeypatch.setattr(app, 'check_image_quality', broken_check)
    raw = io.BytesIO()
    Image.new('RGB', (256, 256), (128, 128, 128)).save(raw, format='PNG')
    response = client.post('/api/predict', files={'file': ('sample.png', raw.getvalue())})
    assert response.status_code == 200
    assert response.json()['quality']['status'] == 'unavailable'


def test_missing_detail_reference_is_reported_as_unavailable(client, monkeypatch, tmp_path):
    import image_quality

    monkeypatch.setattr(image_quality, 'REFERENCE_FILE', tmp_path/'missing.json')
    assert client.get('/api/status').json()['quality_check']['available'] is False
    raw = io.BytesIO()
    Image.new('RGB', (256, 256)).save(raw, format='PNG')
    response = client.post('/api/predict', files={'file': ('sample.png', raw.getvalue())})
    assert response.status_code == 200
    assert response.json()['quality']['available'] is False
