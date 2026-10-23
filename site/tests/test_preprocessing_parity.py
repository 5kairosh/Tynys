"""Compare serving tensors with the actual notebook function, not a retyped copy."""
import ast
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import prepare_image, validate_upload

TRAINING = Path(__file__).resolve().parents[2] / 'training/pneumonia_colab_full_kz.py'


@pytest.mark.parametrize('fmt,mode,size', [
    ('PNG', 'RGB', (71, 113)), ('PNG', 'L', (113, 71)),
    ('PNG', 'RGBA', (17, 61)), ('PNG', 'I;16', (63, 41)),
    ('JPEG', 'RGB', (71, 113)), ('JPEG', 'L', (113, 71)),
    ('JPEG', 'RGB', (224, 224)),
])
def test_notebook_and_serving_tensors_are_identical(client, tmp_path, fmt, mode, size):
    from app import engine
    tf = engine.tf
    tree = ast.parse(TRAINING.read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'decode_image')
    namespace = {'tf': tf, 'IMG_SIZE': tuple(engine.config['image_size'])}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(TRAINING), 'exec'), namespace)
    rng = np.random.default_rng(73)
    shape = (size[1], size[0]) + (() if mode in {'L', 'I;16'} else (4 if mode == 'RGBA' else 3,))
    pixels = rng.integers(0, 65536 if mode == 'I;16' else 256, shape, dtype=np.uint16 if mode == 'I;16' else np.uint8)
    image = Image.fromarray(pixels)
    stream = io.BytesIO()
    # EXIF orientation must be handled exactly as training (no auto rotation).
    options = {'exif': Image.Exif()} if fmt == 'JPEG' else {}
    if options:
        options['exif'][274] = 6
    image.save(stream, format=fmt, **options)
    name = 'sample.' + ('jpg' if fmt == 'JPEG' else 'png')
    path = tmp_path / name
    path.write_bytes(stream.getvalue())
    expected, _ = namespace['decode_image'](tf.constant(str(path)), tf.constant(0))
    checked = validate_upload(path.read_bytes(), name)
    original, actual = prepare_image(tf, checked.data, engine.config['image_size'])
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    assert tuple(actual.shape) == (224, 224, 3) and actual.dtype == tf.float32
    assert 0 <= float(tf.reduce_min(actual)) <= float(tf.reduce_max(actual)) <= 255
    assert tuple(original.shape[:2]) == (size[1], size[0])


def test_normalization_is_inside_each_saved_model(client):
    from app import engine
    tf = engine.tf
    pixels = tf.constant([0., 127.5, 255.])
    for model, scale, offset in [(engine.model, 1 / 127.5, -1), (engine.cnn_model, 1 / 255., 0)]:
        rescalings = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Rescaling)]
        assert len(rescalings) == 1, 'Serving must not add a second normalization.'
        np.testing.assert_allclose(rescalings[0](pixels).numpy(), (pixels * scale + offset).numpy(), atol=1e-7)
