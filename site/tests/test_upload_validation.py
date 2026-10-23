"""Invalid uploads must never reach either classifier or Grad-CAM."""
import io
import hashlib
from pathlib import Path
import re
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from PIL import Image

from app import IMAGE_LIMIT, Engine, validate_upload


def test_changed_frontend_assets_have_current_cache_versions():
    static = Path(__file__).resolve().parents[1] / 'static'
    html = (static / 'index.html').read_text(encoding='utf-8')
    for asset in ('app.js', 'style.css'):
        version = re.search(r'/static/' + re.escape(asset) + r'\?v=([a-f0-9]{12})', html)
        digest = hashlib.sha256((static / asset).read_bytes().replace(b'\r\n', b'\n')).hexdigest()[:12]
        assert version and version.group(1) == digest, 'Browsers must fetch the new asset, not an earlier cached version.'


def image_bytes(fmt='PNG', size=(17, 29), mode='RGB'):
    stream = io.BytesIO()
    Image.new(mode, size, 120).save(stream, format=fmt)
    return stream.getvalue()


@pytest.mark.parametrize('name,data,status,message', [
    ('empty.png', b'', 400, 'бос'),
    ('text.jpg', b'not an image', 400, 'ашу'),
    ('picture.txt', image_bytes(), 400, 'Тек'),
    ('wrong.jpg', image_bytes(), 400, 'сәйкес емес'),
    ('wrong.png', image_bytes('JPEG'), 400, 'сәйкес емес'),
    ('gif.png', image_bytes('GIF'), 400, 'нақты пішімі'),
    ('large.png', b'x' * (IMAGE_LIMIT + 1), 413, '12 МБ'),
    ('broken.png', image_bytes()[:45], 400, 'ашу'),
    ('broken.jpg', image_bytes('JPEG')[:-20], 400, 'ашу'),
    ('large-pixels.png', image_bytes(size=(5001, 4000)), 400, '20 мегапиксель'),
], ids=['empty', 'text', 'unsupported', 'png-disguised', 'jpeg-disguised',
        'gif', 'oversized', 'png-truncated', 'jpeg-truncated', 'pixel-limit'])
def test_reject_before_model(name, data, status, message):
    engine = Engine()
    engine.model = Mock(side_effect=AssertionError('Invalid image reached model'))
    engine.cnn_model = Mock(side_effect=AssertionError('Invalid image reached CNN'))
    with pytest.raises(HTTPException) as error:
        engine.predict(data, name)
    assert error.value.status_code == status
    assert message in error.value.detail
    engine.model.assert_not_called()
    engine.cnn_model.assert_not_called()


def test_animated_png_rejected():
    stream = io.BytesIO()
    Image.new('RGB', (8, 8), 'red').save(stream, format='PNG', save_all=True,
        append_images=[Image.new('RGB', (8, 8), 'blue')], duration=100, loop=0)
    with pytest.raises(HTTPException, match='бір кадрлы'):
        validate_upload(stream.getvalue(), 'animated.png')


@pytest.mark.parametrize('name,fmt', [('X.PNG', 'PNG'), ('X.JpEg', 'JPEG'), ('x.jpg', 'JPEG')])
def test_valid_format_and_dimensions(name, fmt):
    raw = image_bytes(fmt)
    checked = validate_upload(raw, name)
    assert checked.data == raw and (checked.width, checked.height) == (17, 29)
    assert checked.is_dicom is False


def test_tensorflow_decode_error_is_400_before_model(client, monkeypatch):
    import app
    model = Mock(side_effect=AssertionError('Model must not be called'))
    monkeypatch.setattr(app.engine, 'model', model)
    def fail(*args, **kwargs):
        raise app.engine.tf.errors.InvalidArgumentError(None, None, 'bad pixels')
    monkeypatch.setattr(app.engine.tf.io, 'decode_image', fail)
    response = client.post('/api/predict', files={'file': ('valid.png', image_bytes())})
    assert response.status_code == 400 and 'пиксельдерін' in response.json()['detail']
    model.assert_not_called()


@pytest.mark.parametrize('endpoint,field', [('/api/predict', 'file'), ('/api/report', 'files')])
def test_bad_uploads_rejected_by_both_endpoints(client, endpoint, field, monkeypatch):
    import app
    model = Mock(side_effect=AssertionError('Model must not be called'))
    monkeypatch.setattr(app.engine, 'model', model)
    response = client.post(endpoint, files={field: ('spoof.jpg', image_bytes())})
    assert response.status_code == 400 and 'сәйкес емес' in response.json()['detail']
    assert response.headers['cache-control'] == 'no-store'
    model.assert_not_called()


def test_invalid_last_pdf_file_rejected_before_any_inference(client, monkeypatch):
    import app
    model = Mock(side_effect=AssertionError('Batch must be validated first'))
    monkeypatch.setattr(app.engine, 'model', model)
    response = client.post('/api/report', files=[('files', ('ok.png', image_bytes())),
                                                 ('files', ('broken.jpg', b'hello'))])
    assert response.status_code == 400 and 'ашу' in response.json()['detail']
    model.assert_not_called()
