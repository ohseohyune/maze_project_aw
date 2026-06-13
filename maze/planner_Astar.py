"""
A* grid planner for maze path search.
"""

import heapq
import itertools
from typing import Callable

import numpy as np


GridCell = tuple[int, int]


def manhattan_distance(a: GridCell, b: GridCell) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def astar_path(
    grid: np.ndarray,
    start: GridCell,
    goal: GridCell,
    heuristic: Callable[[GridCell, GridCell], float] = manhattan_distance,
) -> list[GridCell]:
    """
    Find a shortest 4-connected grid path using A*.

    The grid convention matches planner.bfs_path:
      0 = free cell
      nonzero = wall / obstacle
    """
    grid = np.asarray(grid, dtype=int)
    rows, cols = grid.shape
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))

    for name, cell in (("start", start), ("goal", goal)):
        r, c = cell
        if r < 0 or r >= rows or c < 0 or c >= cols:
            raise ValueError(f"{name} cell is outside the grid: {cell}")
        if grid[r, c] != 0:
            raise ValueError(f"{name} cell must be free: {cell}")

    counter = itertools.count()
    open_set = [(heuristic(start, goal), 0.0, next(counter), start)]
    parent: dict[GridCell, GridCell | None] = {start: None}
    g_score = {start: 0.0}
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    while open_set:
        _, cur_g, _, cur = heapq.heappop(open_set)
        if cur_g > g_score[cur]:
            continue
        if cur == goal:
            break

        for dr, dc in moves:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if grid[nr, nc] != 0:
                continue

            tentative_g = cur_g + 1.0
            if tentative_g >= g_score.get(nxt, np.inf):
                continue

            parent[nxt] = cur
            g_score[nxt] = tentative_g
            f_score = tentative_g + heuristic(nxt, goal)
            heapq.heappush(open_set, (f_score, tentative_g, next(counter), nxt))

    if goal not in parent:
        raise ValueError(f"No path found from {start} to {goal}")

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))
