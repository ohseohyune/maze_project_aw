"""
Grid planning helpers: BFS, collinear simplification, and corner rounding.
"""

from collections import deque

import numpy as np


def bfs_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    너비 우선 탐색(BFS, Breadth-First Search) 알고리즘을 사용해서 2차원 격자(grid) 위에서 시작점부터 도착점까지의 최단 경로를 찾는 파이썬 함수
    """

    grid = np.asarray(grid, dtype=int)
    rows, cols = grid.shape
    start = tuple(start)
    goal = tuple(goal)
    if grid[start] != 0 or grid[goal] != 0:
        raise ValueError("start and goal must be free cells")

    parent = {start: None}
    queue = deque([start])
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for dr, dc in moves:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if grid[nr, nc] != 0 or nxt in parent:
                continue
            parent[nxt] = cur
            queue.append(nxt)

    if goal not in parent:
        raise ValueError(f"No path found from {start} to {goal}")

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))


def simplify_collinear(path: list) -> list[tuple[float, float]]:
    """
    Remove only redundant points on the same 4-connected corridor.

    Unlike line-of-sight smoothing, this never creates diagonal shortcuts; it
    keeps turns and therefore follows the maze's actual free-cell route.
    """
    if len(path) <= 2:
        return [tuple(map(float, p)) for p in path]

    simplified = [tuple(map(float, path[0]))]
    prev_dir = None
    for a_raw, b_raw in zip(path[:-1], path[1:]):
        a = np.array(a_raw, dtype=float)
        b = np.array(b_raw, dtype=float)
        step = b - a
        if np.linalg.norm(step) < 1e-12:
            continue
        if abs(step[0]) > 1e-9 and abs(step[1]) > 1e-9:
            raise ValueError("simplify_collinear expects a 4-connected path")
        direction = tuple(np.sign(step).astype(int))
        if prev_dir is not None and direction != prev_dir:
            simplified.append(tuple(map(float, a_raw)))
        prev_dir = direction

    simplified.append(tuple(map(float, path[-1])))
    return simplified


def corner_round(path: list, n_subdiv: int = 3) -> list[tuple[float, float]]:
    if len(path) <= 2 or n_subdiv <= 0:
        return [(float(r), float(c)) for r, c in path]

    rounded = [tuple(map(float, path[0]))]
    radius = 0.35

    for i in range(1, len(path) - 1):
        p_prev = np.array(path[i - 1], dtype=float)
        p = np.array(path[i], dtype=float)
        p_next = np.array(path[i + 1], dtype=float)
        d_in = p - p_prev
        d_out = p_next - p
        len_in = np.linalg.norm(d_in)
        len_out = np.linalg.norm(d_out)

        if len_in < 1e-9 or len_out < 1e-9:
            continue
        u_in = d_in / len_in
        u_out = d_out / len_out
        turn_cos = float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))
        if abs(turn_cos - 1.0) < 1e-9:
            rounded.append(tuple(p))
            continue

        turn_angle = float(np.arccos(turn_cos))
        tangent_offset = radius / np.tan(0.5 * turn_angle)
        tangent_offset = min(tangent_offset, 0.45 * len_in, 0.45 * len_out)
        effective_radius = tangent_offset * np.tan(0.5 * turn_angle)
        if effective_radius < 1e-9:
            rounded.append(tuple(p))
            continue

        a = p - tangent_offset * u_in
        b = p + tangent_offset * u_out
        perp_in = np.array([-u_in[1], u_in[0]], dtype=float)
        perp_out = np.array([-u_out[1], u_out[0]], dtype=float)

        center = None
        best_err = np.inf
        for sign_in in (-1.0, 1.0):
            for sign_out in (-1.0, 1.0):
                n_in = sign_in * perp_in
                n_out = sign_out * perp_out
                mat = np.column_stack((n_in, -n_out))
                det = np.linalg.det(mat)
                if abs(det) < 1e-9:
                    continue
                s, t = np.linalg.solve(mat, b - a)
                candidate = a + s * n_in
                err = (
                    abs(np.linalg.norm(candidate - a) - effective_radius)
                    + abs(np.linalg.norm(candidate - b) - effective_radius)
                )
                if err < best_err:
                    best_err = err
                    center = candidate

        if center is None:
            rounded.append(tuple(p))
            continue

        r_a = a - center
        r_b = b - center
        theta_a = float(np.arctan2(r_a[1], r_a[0]))
        theta_b = float(np.arctan2(r_b[1], r_b[0]))
        travel_ccw = np.dot(np.array([-r_a[1], r_a[0]]), u_in) > 0.0

        rounded.append(tuple(a))
        for k in range(1, n_subdiv + 1):
            alpha = k / (n_subdiv + 1)
            if travel_ccw:
                delta = (theta_b - theta_a) % (2.0 * np.pi)
            else:
                delta = -((theta_a - theta_b) % (2.0 * np.pi))
            theta = theta_a + alpha * delta
            q = center + effective_radius * np.array([np.cos(theta), np.sin(theta)])
            rounded.append(tuple(q))
        rounded.append(tuple(b))

    rounded.append(tuple(map(float, path[-1])))
    return rounded


def resample_path(path: list, target_count: int) -> list[tuple[float, float]]:
    """
    Resample a polyline to an exact number of evenly spaced waypoints.

    The first and last points are preserved exactly. This keeps the viewer and
    IK path size predictable without changing the overall corridor route.
    """
    if target_count < 2:
        raise ValueError("target_count must be at least 2")

    pts = np.asarray(path, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("path must be a sequence of (r, c) points")
    if len(pts) == 1:
        return [tuple(pts[0]) for _ in range(target_count)]

    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg_len > 1e-12])
    pts = pts[keep]
    if len(pts) == 1:
        return [tuple(pts[0]) for _ in range(target_count)]

    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    samples = np.linspace(0.0, cum_len[-1], target_count)

    out = []
    j = 0
    for s in samples:
        while j < len(seg_len) - 1 and s > cum_len[j + 1]:
            j += 1
        alpha = 0.0 if seg_len[j] < 1e-12 else (s - cum_len[j]) / seg_len[j]
        q = (1.0 - alpha) * pts[j] + alpha * pts[j + 1]
        out.append((float(q[0]), float(q[1])))

    out[0] = tuple(pts[0])
    out[-1] = tuple(pts[-1])
    return out


def validate_free_path(
    path: list,
    grid: np.ndarray,
    samples_per_cell: int = 4,
    allow_final_exit: bool = True,
) -> None:
    """
    Assert that a float path follows free cells without cutting through walls.

    The optional final exit extension may leave the grid after passing through a
    free boundary cell; all other out-of-grid samples are rejected.
    """
    grid = np.asarray(grid, dtype=int)
    rows, cols = grid.shape
    pts = [np.asarray(p, dtype=float) for p in path]
    if len(pts) < 2:
        return

    for seg_idx, (p0, p1) in enumerate(zip(pts[:-1], pts[1:])):
        dist = float(np.linalg.norm(p1 - p0))
        n = max(1, int(np.ceil(dist * samples_per_cell)))
        for k in range(n + 1):
            p = p0 + (p1 - p0) * (k / n)
            r = int(round(float(p[0])))
            c = int(round(float(p[1])))
            inside = 0 <= r < rows and 0 <= c < cols
            if inside:
                if grid[r, c] != 0:
                    raise ValueError(
                        f"Path intersects wall near segment {seg_idx}: "
                        f"sample=({p[0]:.3f}, {p[1]:.3f}), cell=({r}, {c})"
                    )
                continue

            r_clip = int(np.clip(r, 0, rows - 1))
            c_clip = int(np.clip(c, 0, cols - 1))
            clipped_is_boundary = (
                r_clip in {0, rows - 1} or c_clip in {0, cols - 1}
            )
            exits_through_free_boundary = (
                allow_final_exit
                and clipped_is_boundary
                and grid[r_clip, c_clip] == 0
                and seg_idx >= len(pts) - 4
            )
            if not exits_through_free_boundary:
                raise ValueError(
                    f"Path leaves grid before final exit near segment {seg_idx}: "
                    f"sample=({p[0]:.3f}, {p[1]:.3f})"
                )
