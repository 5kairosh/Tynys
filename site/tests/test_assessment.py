import io

import pytest
from PIL import Image
from pypdf import PdfReader

from assessment import assess_input


@pytest.mark.parametrize('check,reason', [
    ({'available': True, 'is_xray': False}, 'ood_shift'),
    ({'available': False}, 'ood_unavailable'),
    ({'available': True}, 'ood_unavailable'),
    ({'available': True, 'is_xray': 'true'}, 'ood_unavailable'),
    (None, 'ood_unavailable'),
])
def test_missing_or_flagged_check_withholds_headline_class(check, reason):
    result = assess_input(check)
    assert result['status'] == 'review_required'
    assert result['reason'] == reason
    assert result['clinically_validated'] is False


@pytest.mark.parametrize('within_reference', [True, False, None])
def test_api_and_pdf_agree_without_changing_scores(client, monkeypatch, within_reference):
    from app import engine

    # Exercise the real API/model/report flow, controlling only the OOD outcome.
    monkeypatch.setattr(engine.ood, 'status', lambda: {'available': within_reference is not None})
    monkeypatch.setattr(engine.ood, 'check', lambda vector: {
        'available': True, 'is_xray': within_reference,
        'distance': 1. if within_reference else 100., 'threshold': 26.34,
    })
    stream = io.BytesIO()
    Image.new('RGB', (80, 100), (90, 110, 120)).save(stream, format='PNG')
    raw = stream.getvalue()
    response = client.post('/api/predict', files={'file': ('sample.png', raw)})
    assert response.status_code == 200, response.text
    result = response.json()
    review = within_reference is not True
    assert (result['assessment']['status'] == 'review_required') is review
    assert result['assessment']['clinically_validated'] is False
    assert result['model_class'] == ('PNEUMONIA' if result['pneumonia_score'] >= result['threshold'] else 'NORMAL')
    assert result['threshold'] == .745
    assert result['comparison']['available']
    assert result['comparison']['models'][1]['pneumonia_score'] == result['pneumonia_score']
    report = client.post('/api/report', files=[('files', ('a.png', raw)), ('files', ('b.png', raw))])
    assert report.status_code == 200, report.text
    reader = PdfReader(io.BytesIO(report.content))
    assert len(reader.pages) == 2
    for page in reader.pages:
        text = page.extract_text()
        assert ('Сенімді бағалау мүмкін емес' in text) is review
        if review:
            assert 'Техникалық санат:' in text
            assert 'Санат бойынша қорытынды берілмеді.' in text
            assert 'клиникалық сенімділікті растамайды' in text
        else:
            assert 'Модель санаты' in text


def test_ood_computation_failure_withholds_classification(client, monkeypatch):
    from app import engine

    monkeypatch.setattr(engine.ood, 'status', lambda: {'available': True})

    def broken_check(vector):
        raise ValueError('simulated OOD failure')

    monkeypatch.setattr(engine.ood, 'check', broken_check)
    stream = io.BytesIO()
    Image.new('RGB', (64, 64)).save(stream, format='PNG')
    response = client.post('/api/predict', files={'file': ('sample.png', stream.getvalue())})
    assert response.status_code == 200
    assert response.json()['assessment']['reason'] == 'ood_unavailable'
    assert response.json()['assessment']['status'] == 'review_required'
