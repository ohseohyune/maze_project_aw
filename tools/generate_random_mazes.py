#!/usr/bin/env python3
"""
Generate random mazes in the project's 12x12 CSV format using a
Recursive-Backtracker (DFS perfect maze) on a 5x5 room grid.

Every generated maze is a *perfect* maze:
  - exactly one path between any two rooms (no loops)
  - all corridor stubs are dead ends
  - every maze looks visually distinct for different seeds

Room grid: 5x5 rooms at odd positions (1,1), (1,3), …, (9,9) inside
the 10x10 inner region.  A 2-cell tail (10,9)→(10,10) connects the
corner room (9,9) to the goal.

  start = (1, 1)
  goal  = (10, 10)
"""

import argparse
import os
import random
from collections import deque

import numpy as np


START = (1, 1)
GOAL = (10, 10)
ENTRANCE = (0, 1)
EXIT = (11, 10)
GOAL_ROOM = (9, 9)

_ROOM_ROWS = range(1, 10, 2)   # 1, 3, 5, 7, 9
_ROOM_COLS = range(1, 10, 2)
ROOMS = {(r, c) for r in _ROOM_ROWS for c in _ROOM_COLS}


def _bfs_path(grid):
    q = deque([START])
    parent = {START: None}
    while q:
        cur = q.popleft()
        if cur == GOAL:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return list(reversed(path))
        r, c = cur
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nxt = (r + dr, c + dc)
            nr, nc = nxt
            if 0 <= nr <= 11 and 0 <= nc <= 11 and nxt not in parent and grid[nxt] == 0:
                parent[nxt] = cur
                q.append(nxt)
    return []


def generate_maze(seed: int) -> np.ndarray:
    """
    Recursive-backtracker on the 5x5 room grid.
    Returns a 12x12 array (1 = wall, 0 = free).
    """
    rng = random.Random(seed)

    grid = np.ones((12, 12), dtype=int)
    grid[ENTRANCE] = 0
    grid[EXIT] = 0

    # Start DFS from (1,1)
    grid[START] = 0
    visited = {START}
    stack = [START]

    while stack:
        r, c = stack[-1]
        # Candidate unvisited rooms reachable in one step (2 cells away)
        unvisited = []
        for dr, dc in [(-2, 0), (0, 2), (2, 0), (0, -2)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in ROOMS and (nr, nc) not in visited:
                unvisited.append((nr, nc))

        if unvisited:
            rng.shuffle(unvisited)
            nr, nc = unvisited[0]
            # Carve the wall cell between current room and chosen room
            grid[(r + nr) // 2, (c + nc) // 2] = 0
            # Open the chosen room
            grid[nr, nc] = 0
            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()

    # Short tail from corner room (9,9) to goal (10,10)
    grid[10, 9] = 0
    grid[10, 10] = 0

    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="generated_mazes")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260526)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.count):
        grid = generate_maze(args.seed + i)
        path = _bfs_path(grid)
        if not path:
            raise RuntimeError(f"No path found for maze {i} (seed={args.seed + i})")
        out = os.path.join(args.out_dir, f"random_maze_{i:02d}.csv")
        np.savetxt(out, grid, fmt="%d", delimiter=",")
        print(f"{out}  bfs_len={len(path)}")


if __name__ == "__main__":
    main()
