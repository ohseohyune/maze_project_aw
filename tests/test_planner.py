import os

from maze.parser import read_occupancy_grid
from maze.planner import (
    bfs_path,
    corner_round,
    line_of_sight_smooth,
    resample_path,
    simplify_collinear,
    validate_free_path,
)


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_bfs_sample_maze():
    grid = read_occupancy_grid(os.path.join(HERE, "maze_occupancy_grid.csv"))
    path = bfs_path(grid, (1, 1), (10, 10))
    assert path[0] == (1, 1)
    assert path[-1] == (10, 10)
    for r, c in path:
        assert grid[r, c] == 0


def test_smoothing_keeps_endpoints():
    grid = read_occupancy_grid(os.path.join(HERE, "maze_occupancy_grid.csv"))
    path = bfs_path(grid, (1, 1), (10, 10))
    smooth = line_of_sight_smooth(path, grid)
    rounded = corner_round(smooth)
    assert smooth[0] == path[0]
    assert smooth[-1] == path[-1]
    assert rounded[0] == (1.0, 1.0)
    assert rounded[-1] == (10.0, 10.0)


def test_simplify_collinear_keeps_corridor_turns():
    path = [(0, 1), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3)]
    simplified = simplify_collinear(path)
    assert simplified == [(0.0, 1.0), (1.0, 1.0), (1.0, 3.0), (3.0, 3.0)]
    for a, b in zip(simplified[:-1], simplified[1:]):
        assert a[0] == b[0] or a[1] == b[1]


def test_resample_path_exact_count_and_endpoints():
    path = [(0.0, 1.0), (1.0, 1.0), (1.0, 3.0)]
    sampled = resample_path(path, target_count=180)
    assert len(sampled) == 180
    assert sampled[0] == path[0]
    assert sampled[-1] == path[-1]


def test_validate_free_path_rejects_wall():
    grid = read_occupancy_grid(os.path.join(HERE, "maze_occupancy_grid.csv"))
    validate_free_path([(0.0, 1.0), (1.0, 1.0)], grid)
    try:
        validate_free_path([(0.0, 1.0), (0.0, 2.0)], grid)
    except ValueError:
        return
    raise AssertionError("Expected wall-intersecting path to be rejected")
