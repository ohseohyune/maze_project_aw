import os

from maze.parser import read_occupancy_grid, validate_start_goal


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_read_and_detect_sample_grid():
    grid = read_occupancy_grid(os.path.join(HERE, "maze_occupancy_grid.csv"))
    assert grid.shape == (12, 12)
    start, goal = validate_start_goal(grid)
    assert start == (1, 1)
    assert goal == (10, 10)
