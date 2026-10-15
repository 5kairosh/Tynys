import numpy as np
import pytest
from pad_geometry import pad_geometry,map_point


@pytest.mark.parametrize('h,w',[(224,224),(605,605),(1464,1280),(300,900),(701,299),(1,1),(17,1000)])
def test_geometry(h,w):
    g=pad_geometry(h,w)
    rh,rw=g['resized_h'],g['resized_w']
    assert g['pad_y']>=0 and g['pad_x']>=0
    assert rh+g['pad_y']<=224 and rw+g['pad_x']<=224
    ratio=max(w/224,h/224)
    assert abs(rh-h/ratio)<=1.0001 and abs(rw-w/ratio)<=1.0001
    original=(w*.37,h*.61)
    mapped=map_point(*original,h,w)
    np.testing.assert_allclose(map_point(*mapped,h,w,inverse=True),original,atol=1e-10)
    # Compare exact float32 geometry to TensorFlow's real padding, not just helper arithmetic.
    import tensorflow as tf
    result=tf.image.resize_with_pad(tf.ones([h,w,1]),224,224).numpy()[...,0]
    nonzero=np.argwhere(result>.5)
    assert tuple(nonzero.min(axis=0))==(g['pad_y'],g['pad_x'])
    assert tuple(nonzero.max(axis=0)+1)==(g['pad_y']+rh,g['pad_x']+rw)
