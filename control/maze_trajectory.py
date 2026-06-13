"""
Joint-space trajectory generation for the maze path.
"""

import numpy as np

from reference.dream_math import cubic_spline


def allocate_segment_times(
    q_path: np.ndarray,
    q_dot_max: np.ndarray,
    q_ddot_max: np.ndarray = None,
) -> np.ndarray:
    """
    Joint-space waypoint별 도착 시간을 누적해서 계산한다.

    - 각 segment는 q0 -> q1 이동이다.
    - joint별 이동량 dq를 q_dot_max로 나누면 joint별 필요 시간이 된다.
    - 모든 joint는 동시에 움직이므로, 가장 오래 걸리는 joint가 segment 시간을 결정한다.
    - 예: Joint1이 10 deg, Joint2가 20 deg 움직이고 q_dot_max가 180 deg/s라면
      segment 시간은 max(10/180, 20/180) = 0.111 s이다.
    - 반환값은 누적 시간이다. 예: [0.0, 0.2, 0.5]는 0번 waypoint는 0.0초,
      1번 waypoint는 0.2초, 2번 waypoint는 0.5초에 도착한다는 뜻이다.
    """
    q_path = np.asarray(q_path, dtype=float)
    q_dot_max = np.asarray(q_dot_max, dtype=float).reshape(-1)
    if q_path.ndim != 2:
        raise ValueError("q_path must be a 2D array")
    if len(q_path) == 0:
        raise ValueError("q_path cannot be empty")
    if len(q_path) == 1:
        return np.array([0.0])

    times = [0.0]
    for q0, q1 in zip(q_path[:-1], q_path[1:]):
        # - 현재 segment에서 각 joint가 움직여야 하는 양
        # - 가장 오래 걸리는 joint가 이 segment의 기준 시간이 된다.
        dq = np.abs(q1 - q0)
        vel_time = np.max(dq / np.maximum(q_dot_max, 1e-9))

        if q_ddot_max is not None:
            q_ddot_max = np.asarray(q_ddot_max, dtype=float).reshape(-1)
            # - 가속도 제한이 주어지면 가속도 기준 시간도 계산한다.
            # - 속도 기준 시간과 가속도 기준 시간 중 더 긴 값을 사용한다.
            acc_time = np.max(np.sqrt(2.0 * dq / np.maximum(q_ddot_max, 1e-9)))
            seg_time = max(vel_time, acc_time)
        else:
            seg_time = vel_time

        # - spline은 시작/끝 속도를 부드럽게 만들기 때문에 평균 속도 기준 시간보다 여유가 필요하다.
        # - 1.875는 smooth timing에서 peak velocity가 평균 velocity보다 커지는 비율이다.
        # - 0.02초는 waypoint 차이가 아주 작을 때 segment 시간이 0에 가까워지는 것을 막는다.
        times.append(times[-1] + max(1.875 * seg_time, 0.02))
    return np.asarray(times)


def _waypoint_velocities(q_path: np.ndarray, seg_times: np.ndarray) -> np.ndarray:
    """
    각 waypoint에서 사용할 joint velocity를 추정한다.

    - spline 보간은 위치뿐 아니라 waypoint에서의 속도도 필요하다.
    - 시작점과 끝점은 정지 상태로 두기 위해 velocity를 0으로 둔다.
    - 중간 waypoint는 이전 segment와 다음 segment의 평균 기울기를 사용한다.
    - 단, joint 움직임 방향이 바뀌는 경우에는 overshoot를 막기 위해 velocity를 0으로 둔다.
    """
    qdot = np.zeros_like(q_path)
    if len(q_path) <= 2:
        return qdot

    for i in range(1, len(q_path) - 1):
        # - 현재 waypoint 기준 이전/다음 segment의 duration
        dt_prev = seg_times[i] - seg_times[i - 1]
        dt_next = seg_times[i + 1] - seg_times[i]
        if dt_prev <= 0.0 or dt_next <= 0.0:
            continue

        # - 이전 segment와 다음 segment의 joint-space slope
        # - slope는 대략 joint velocity와 같은 의미이다.
        slope_prev = (q_path[i] - q_path[i - 1]) / dt_prev
        slope_next = (q_path[i + 1] - q_path[i]) / dt_next

        # - 같은 방향으로 계속 움직이는 joint는 평균 속도를 사용한다.
        # - 방향이 바뀌는 joint는 해당 waypoint에서 잠깐 멈추도록 0을 사용한다.
        same_direction = slope_prev * slope_next > 0.0
        avg = (q_path[i + 1] - q_path[i - 1]) / (dt_prev + dt_next)
        qdot[i] = np.where(same_direction, avg, 0.0)

    return qdot


def joint_spline_traj(
    t: float,
    q_path: np.ndarray,
    seg_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    시간 t에서의 joint position, velocity, acceleration을 계산한다.

    - q_path는 waypoint별 joint angle이다.
    - seg_times는 각 waypoint에 도착해야 하는 누적 시간이다.
    - 현재 시간 t가 어느 segment에 있는지 찾는다.
    - 해당 segment 안에서 reference.dream_math.cubic_spline으로 q를 보간한다.
    - 반환값은 q_des, qdot_des, qddot_des이다.
    """
    q_path = np.asarray(q_path, dtype=float)
    seg_times = np.asarray(seg_times, dtype=float).reshape(-1)
    if q_path.ndim != 2:
        raise ValueError("q_path must be a 2D array")
    if len(q_path) != len(seg_times):
        raise ValueError("q_path and seg_times must have matching lengths")
    if len(q_path) == 1 or t <= seg_times[0]:
        # - 시작 전이거나 waypoint가 하나뿐이면 시작 자세에서 정지한다.
        z = np.zeros(q_path.shape[1])
        return q_path[0].copy(), z.copy(), z.copy()
    if t >= seg_times[-1]:
        # - trajectory 종료 후에는 마지막 자세에서 정지한다.
        z = np.zeros(q_path.shape[1])
        return q_path[-1].copy(), z.copy(), z.copy()

    # - 현재 시간 t가 포함되는 segment index를 찾는다.
    # - 예: seg_times = [0.0, 1.0, 3.0]이고 t = 1.5이면 idx = 1이다.
    idx = int(np.searchsorted(seg_times, t, side="right") - 1)
    idx = max(0, min(idx, len(q_path) - 2))
    qdot_path = _waypoint_velocities(q_path, seg_times)

    # - 현재 segment의 시작/끝 시간과 duration
    t0 = seg_times[idx]
    t1 = seg_times[idx + 1]
    h = t1 - t0
    if h <= 0.0:
        z = np.zeros(q_path.shape[1])
        return q_path[idx].copy(), z.copy(), z.copy()

    # - s는 segment 내부 진행률이다.
    # - s = 0이면 q0, s = 1이면 q1이다.
    s = (t - t0) / h
    q0 = q_path[idx]
    q1 = q_path[idx + 1]
    v0 = qdot_path[idx]
    v1 = qdot_path[idx + 1]

    # - reference.dream_math의 cubic_spline을 사용해 q0/q1 위치와 v0/v1 속도를 만족하도록 보간한다.
    q = cubic_spline(t, t0, t1, q0, q1, v0, v1)

    # - cubic_spline은 위치만 반환하므로 같은 3차 다항식 계수에서 qdot/qddot을 해석적으로 계산한다.
    total_q = q1 - q0
    a1 = v0
    a2 = 3.0 * total_q / h**2 - 2.0 * v0 / h - v1 / h
    a3 = -2.0 * total_q / h**3 + (v0 + v1) / h**2
    t_elapsed = t - t0
    qdot = a1 + 2.0 * a2 * t_elapsed + 3.0 * a3 * t_elapsed**2
    qddot = 2.0 * a2 + 6.0 * a3 * t_elapsed

    return q, qdot, qddot
