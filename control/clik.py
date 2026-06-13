"""
Closed-loop inverse kinematics helpers.
"""

import numpy as np

from control.jacobian import body_jacobian
from control.trajectory import target_trajectory
from robot.kinematics import T_inv, body_poe_fk
from utils.lie_group import matrix_log_se3


def compute_body_error(T_cur: np.ndarray, T_des: np.ndarray) -> np.ndarray:
    T_err = T_inv(T_cur) @ T_des
    return matrix_log_se3(T_err)


def _solve_joint_update(J: np.ndarray, control: np.ndarray, method: str, damping: float):
    if method == "pinv":
        return np.linalg.pinv(J) @ control
    if method == "dls":
        eye = np.eye(J.shape[0])
        return J.T @ np.linalg.solve(J @ J.T + damping**2 * eye, control)
    raise ValueError(f"Unknown CLIK method: {method}")


def clik_one_step(
    theta,
    T_des,
    B_list,
    M,
    K_p,
    dt,
    method: str = "dls",
    damping: float = 0.05,
    joint_lower_limits: np.ndarray = None,
    joint_upper_limits: np.ndarray = None,
) -> np.ndarray:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    T_cur = body_poe_fk(theta, B_list, M)
    e_b = compute_body_error(T_cur, T_des)
    Jb = body_jacobian(theta, B_list)
    control = np.asarray(K_p, dtype=float).reshape(6, 6) @ e_b
    theta_dot = _solve_joint_update(Jb, control, method, damping)
    theta_new = theta + theta_dot * dt

    if joint_lower_limits is not None or joint_upper_limits is not None:
        lo = (
            -np.inf
            if joint_lower_limits is None
            else np.asarray(joint_lower_limits, dtype=float)
        )
        hi = (
            np.inf
            if joint_upper_limits is None
            else np.asarray(joint_upper_limits, dtype=float)
        )
        theta_new = np.clip(theta_new, lo, hi)
    return theta_new


def solve_ik(
    T_des,
    B_list,
    M,
    q_init,
    K_p,
    max_iter=200,
    tol=1e-4,
    joint_lower_limits=None,
    joint_upper_limits=None,
    method: str = "dls",
    damping: float = 0.05,
    dt: float = 0.05,
) -> tuple[np.ndarray, bool]:
    q = np.asarray(q_init, dtype=float).reshape(-1).copy()
    best_q = q.copy()
    best_err = np.inf

    for _ in range(max_iter):
        T_cur = body_poe_fk(q, B_list, M)
        err = compute_body_error(T_cur, T_des)
        err_norm = np.linalg.norm(err)
        if err_norm < best_err:
            best_err = err_norm
            best_q = q.copy()
        if err_norm < tol:
            return q, True

        Jb = body_jacobian(q, B_list)
        cond = np.linalg.cond(Jb)
        step_damping = damping if np.isfinite(cond) and cond < 1e4 else max(damping, 0.1)
        q = clik_one_step(
            q,
            T_des,
            B_list,
            M,
            K_p,
            dt,
            method=method,
            damping=step_damping,
            joint_lower_limits=joint_lower_limits,
            joint_upper_limits=joint_upper_limits,
        )

    return best_q, False


def run_clik(
    theta_init,
    B_list,
    M,
    K_p,
    K_i=None,
    t_start=0.0,
    t_end=5.0,
    dt=0.01,
    delta=0.05,
    method: str = "dls",
    damping: float = 0.05,
    joint_lower_limits=None,
    joint_upper_limits=None,
) -> dict:
    theta = np.asarray(theta_init, dtype=float).copy()
    T_start = body_poe_fk(theta, B_list, M)
    time_steps = np.arange(t_start, t_end + dt * 0.5, dt)

    log_time = np.zeros(len(time_steps))
    log_theta = np.zeros((len(time_steps), len(B_list)))
    log_pos_cur = np.zeros((len(time_steps), 3))
    log_pos_des = np.zeros((len(time_steps), 3))
    log_error = np.zeros((len(time_steps), 6))
    log_cond = np.zeros(len(time_steps))

    for k, t in enumerate(time_steps):
        T_des = target_trajectory(T_start, t_start, t_end, t, delta)
        theta = clik_one_step(
            theta,
            T_des,
            B_list,
            M,
            K_p,
            dt,
            method=method,
            damping=damping,
            joint_lower_limits=joint_lower_limits,
            joint_upper_limits=joint_upper_limits,
        )

        T_cur = body_poe_fk(theta, B_list, M)
        e_b = compute_body_error(T_cur, T_des)
        log_time[k] = t
        log_theta[k] = theta
        log_pos_cur[k] = T_cur[:3, 3]
        log_pos_des[k] = T_des[:3, 3]
        log_error[k] = e_b
        log_cond[k] = np.linalg.cond(body_jacobian(theta, B_list))

    return {
        "time": log_time,
        "theta": log_theta,
        "pos_cur": log_pos_cur,
        "pos_des": log_pos_des,
        "error": log_error,
        "cond": log_cond,
    }
