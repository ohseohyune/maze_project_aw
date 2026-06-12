"""
CSV parser and start/goal validation for occupancy-grid mazes.

- real_occupancy_grid : 2D CSV 파일의 경로를 받아서 배열(grid)로 반환하는 함수
- validate_start_goal : fixed start and goal이 outside of maze에 있지 않은지, 벽으로 감지되지는 않은지 검증하는 함수 
"""

import numpy as np


def read_occupancy_grid(path: str) -> np.ndarray:
    grid = np.loadtxt(path, delimiter=",", dtype=int)
    if grid.ndim != 2:
        raise ValueError("Occupancy grid must be a 2D CSV")
    if not np.isin(grid, [0, 1]).all():
        raise ValueError("Occupancy grid values must be 0 or 1")
    return grid


def validate_start_goal(
    grid: np.ndarray,
    start: tuple[int, int] = (1, 1),
    goal: tuple[int, int] = (10, 10),
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Validate and return the fixed start and goal cells."""
    # 입력값을 NumPy 배열로 바꿔주는 함수
    grid = np.asarray(grid, dtype=int)
    rows, cols = grid.shape
    start = tuple(start)
    goal = tuple(goal)

    for name, cell in (("start", start), ("goal", goal)):
        r, c = cell
        if r < 0 or r >= rows or c < 0 or c >= cols:
            raise ValueError(f"{name} cell {cell} is outside the occupancy grid")
        # start 또는 goal 위치가 벽인지 확인하는 코드
        if grid[r, c] != 0:
            raise ValueError(f"{name} cell {cell} must be free")

    return start, goal
