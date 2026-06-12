"""
SE(3) trajectory utilities.
---------------------------
시간 t
→ 0~1 사이 진행률 s 계산
→ 시작 pose T0와 목표 pose T1 사이를 SE(3)에서 보간
→ 현재 목표 pose T_des 반환
"""

import numpy as np

from reference.dream_math import cubic_spline
from robot.kinematics import T_inv, matrix_exp_se3
from utils.lie_group import matrix_log_se3


def cubic_interp(t: float, t_start: float, t_end: float) -> float:
    """
    t_start 시간에는 s = 0
    t_end 시간에는 s = 1
    시작, 끝 속도는 0 인 cubic spline

    """
    return float(cubic_spline(t, t_start, t_end, 0.0, 1.0, 0.0, 0.0))

# SE(3) interpolation에서 twist vector를 4x4 SE(3) 변환행렬로 바꿔주는 함수
def _se3_exp_vec(twist: np.ndarray) -> np.ndarray:
    # 6차원 twist vector를 SE(3) transformation matrix로 바꾸는 함수
    twist = np.asarray(twist, dtype=float).reshape(6)
    w_th = twist[:3]
    v_th = twist[3:]
    theta = np.linalg.norm(w_th)
    if theta < 1e-12:
        # 회전량이 거의 0이면, 즉 거의 순수 이동이면 특별 처리
        T = np.eye(4)
        T[:3, 3] = v_th
        return T
    S = np.concatenate([w_th / theta, v_th / theta])
    return matrix_exp_se3(S, theta)


def se3_interpolate(T0: np.ndarray, T1: np.ndarray, s: float) -> np.ndarray:
    """
    SE(3) 곡면 → log → se(3) 평면 → s배 스케일링(선형 보간) → exp → SE(3) 곡면 복귀

 
    SE(3)는 곡면이기에 s를 곱하는 선형 보간이 불가능하다. 
    따라서 se(3)로 변환을 해서 선형 보간을 해준 다음에 다시 SE(3)로 바꾼다. 

    s * log_rel는 twist 벡터를 s배 스케일링을 해서  "전체 경로의 s배 만큼을 이동"하게 한다. 
    이는 T0 기준 좌표계에서 연산을 수행했으므로, T0을 다시 곱해서 World 좌표계 기준으로 변환해준다

    T0 : 시작 pose (4x4 SE(3) 행렬)
    T1 : 목표 pose (4x4 SE(3) 행렬)
    """
    # 진행률 클리핑
    s = float(np.clip(s, 0.0, 1.0))
    # T0에서 T1으로 가는 상대 변환 : T0 기준 좌표계"에서 T1이 어디 있는지를 계산
    T_rel = T_inv(T0) @ T1
    # T_rel을 se(3) Lie Algebra로 변환
    log_rel = matrix_log_se3(T_rel)
    return T0 @ _se3_exp_vec(s * log_rel)


def target_trajectory(
    T_start: np.ndarray,
    t_start: float,
    t_end: float,
    t: float,
    delta: float = 0.05,
) -> np.ndarray:
    """
    현재 pose에서 앞쪽으로 delta만큼 직선 이동하는 궤적을 생성
     회전 행렬(R)을 그대로 유지하고 위치만 바꿈
    """
    s = cubic_interp(t, t_start, t_end)
    T_final = np.array(T_start, dtype=float, copy=True)
    T_final[:3, 3] = T_start[:3, 3] + T_start[:3, 0] * delta
    return se3_interpolate(T_start, T_final, s)


def maze_trajectory(t: float, T_waypoints, segment_times) -> np.ndarray:
    """
    여러 waypoint를 시간 순서대로 SE(3) 보간으로 이동하는 궤적 함수
    """

    # SE(3) pose 목록 : [T0, T1, T2, T3]
    T_waypoints = [np.asarray(T, dtype=float).reshape(4, 4) for T in T_waypoints]
    # 각 waypoint에 도달하는 시간 : [t0, t1, t2, t3]
    segment_times = np.asarray(segment_times, dtype=float).reshape(-1)
    if len(T_waypoints) != len(segment_times):
        raise ValueError("T_waypoints and segment_times must have the same length")
    if len(T_waypoints) == 0:
        raise ValueError("At least one waypoint is required")
    if len(T_waypoints) == 1 or t <= segment_times[0]:
        return T_waypoints[0].copy()
    if t >= segment_times[-1]:
        return T_waypoints[-1].copy()

    # 현재 시간이 속한 구간 찾기 
    idx = int(np.searchsorted(segment_times, t, side="right") - 1)
    idx = max(0, min(idx, len(T_waypoints) - 2))
    # 찾은 구간의 시작/끝 시간으로 진행률 s 계산 후 SE(3) 보간
    t0 = segment_times[idx]
    t1 = segment_times[idx + 1]
    s = cubic_interp(t, t0, t1)
    return se3_interpolate(T_waypoints[idx], T_waypoints[idx + 1], s)
