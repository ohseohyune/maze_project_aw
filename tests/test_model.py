import numpy as np

from robot.model import define_model


def test_define_model_tip_offset():
    _, M, _ = define_model(tcp_offset=(0.0, -0.315, 0.0))
    assert np.allclose(M[:3, 3], [0.0, -0.428, 0.7535])


def test_define_model_default_offset_unchanged():
    _, M, _ = define_model()
    assert np.allclose(M[:3, 3], [0.0, -0.222, 0.7535])
