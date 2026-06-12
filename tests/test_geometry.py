import numpy as np

from maze.geometry import cell_to_world


def test_cell_to_world_flat():
    p, n = cell_to_world(0, 1, hover=0.0)
    assert np.allclose(p, [0.34 - 0.18, -0.22, 0.008], atol=1e-3)
    assert np.allclose(n, [0.0, 0.0, 1.0])


def test_cell_to_world_default_hover_is_1mm():
    p, n = cell_to_world(0, 1)
    p_surface, _ = cell_to_world(0, 1, hover=0.0)
    assert np.isclose(np.dot(p - p_surface, n), 0.001)


def test_cell_to_world_folded():
    _, n = cell_to_world(10, 10, hover=0.0)
    assert np.allclose(n, [0.0, -0.5, np.sqrt(3.0) / 2.0], atol=1e-3)


def test_exit_site_alignment():
    p, _ = cell_to_world(11, 10, hover=0.005)
    assert np.allclose(p, [0.52, 0.1880, 0.1223], atol=2e-3)
