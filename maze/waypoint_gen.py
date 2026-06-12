"""
Convert maze cell paths into SE(3) target poses.
"""

import numpy as np

from maze.geometry import FOLD_ANGLE, FOLD_SWITCH_ROW, cell_to_world

# 로봇의 진행 방향 벡터를 계산하기 위해 단위 벡터 구함
def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return v / n

# 로봇 tool이 미로 표면과 수직하도록 하는 회전 행렬 구하는 함수
def _orientation_from_tool_axis(
    tool_axis_in_ee: np.ndarray,
    target_axis_world: np.ndarray,
    preferred_x_world: np.ndarray = None,
):
    tool_axis_in_ee = _normalize(tool_axis_in_ee)
    target_axis_world = _normalize(target_axis_world)

    if not np.allclose(tool_axis_in_ee, [0.0, 1.0, 0.0]):
        raise NotImplementedError("Only link6 +y tool axis is supported for now")

    # 미로의 법선벡터가 ee의 y축과 평행하도록
    y_axis = target_axis_world
    # 참조 벡터(ref) 결정
    ref = np.array([1.0, 0.0, 0.0])
    if preferred_x_world is not None:
        ref_candidate = np.asarray(preferred_x_world, dtype=float).reshape(3)
        ref_candidate = ref_candidate - np.dot(ref_candidate, y_axis) * y_axis
        if np.linalg.norm(ref_candidate) > 1e-6:
            ref = ref_candidate
    if abs(np.dot(ref, y_axis)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])

    # x,z축 계산
    x_axis = ref - np.dot(ref, y_axis) * y_axis
    x_axis = _normalize(x_axis)
    z_axis = _normalize(np.cross(x_axis, y_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))

    R = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(R) < 0:
        z_axis *= -1.0
        R = np.column_stack([x_axis, y_axis, z_axis])
    # end-effector의 desired 회전행렬
    return R

# flat 구간 → folded 구간으로 넘어갈 때 법선벡터를 부드럽게 전환하는 함수
def _smooth_tool_normal(r: float, n_surface: np.ndarray, transition_half_width: float = 1.50):
    """
    - flat:   n = [0, 0, 1]         ← z축 위
    - folded: n = [0, -0.5, 0.866]  ← 30° 기울어진 방향

    그냥 뚝 바꾸면 로봇 자세가 순간적으로 확 꺾여서 충격이 생겨. 그래서 경첩 주변 ±1.5 행 구간에서 서서히 보간
    
    - row ≤ 4.5  → flat_n 그대로
    - row ≥ 7.0  → folded_n 그대로
    - 그 사이    → 아래 공식으로 보간
    """
    flat_n = np.array([0.0, 0.0, 1.0])
    folded_n = np.array([0.0, -np.sin(FOLD_ANGLE), np.cos(FOLD_ANGLE)])
    r = float(r)
    lo = FOLD_SWITCH_ROW - transition_half_width
    hi = FOLD_SWITCH_ROW + transition_half_width
    if r <= lo:
        return flat_n
    if r >= hi:
        return folded_n
    u = (r - lo) / (hi - lo)
    #  시작과 끝에서 기울기가 0이라 자연스럽게 전환
    s = 3.0 * u**2 - 2.0 * u**3
    return _normalize((1.0 - s) * flat_n + s * folded_n)


def build_se3_targets(
    cell_path: list,
    tool_axis_in_ee: np.ndarray = np.array([0.0, 1.0, 0.0]),
    yaw_strategy: str = "continuous",
) -> list[np.ndarray]:
    """
    cell_path(셀 좌표 목록) → 각 waypoint의 SE(3) 목표 pose 리스트를 반환하는 함수
    """

    if yaw_strategy not in {"fixed", "continuous", "min_joint5"}:
        raise ValueError("yaw_strategy must be 'fixed', 'continuous', or 'min_joint5'")

    targets = []
    prev_x_axis = None
    for r, c in cell_path:
        # 1. 셀 좌표 → 3D 월드 위치 p, 법선벡터 n
        p, n = cell_to_world(r, c)
        # 2. 경첩 근처면 법선을 부드럽게 보간
        n_tool = _smooth_tool_normal(r, n)
        # The tip TCP is at link6 -y.  Aligning link6 +y with +normal makes the
        # physical tip direction (-y) point into the maze plane (-normal).
        preferred_x = prev_x_axis if yaw_strategy in {"continuous", "min_joint5"} else None
        R = _orientation_from_tool_axis(tool_axis_in_ee, n_tool, preferred_x)
        prev_x_axis = R[:, 0].copy()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        targets.append(T)
    return targets
