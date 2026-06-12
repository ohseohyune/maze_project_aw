"""
control/jacobian.py
===================
Problem 1: Body Jacobian  J_b(θ) ∈ ℝ^{6×n}

포함 함수:
  body_jacobian : θ, B_list → J_b
"""

import numpy as np
from robot.kinematics import adjoint, T_inv, matrix_exp_se3


def body_jacobian(theta: np.ndarray, B_list: list) -> np.ndarray:
    """
    Body Jacobian  J_b(θ) ∈ ℝ^{6×n}.   (Modern Robotics §5.5)

    관계식
    ──────
      V_b = J_b(θ) · θ̇

    변환 공식
    ─────────
      J_s(θ) = [Ad_Tsb] J_b(θ)
      J_b(θ) = [Ad_Tbs] J_s(θ),   T_bs = T_sb^{-1}

    구현 전략
    ─────────
      먼저 B_list 를 screw axis 로 하는 PoE product

        T_sb = e^[B1]θ1 · ... · e^[Bn]θn

      의 space Jacobian J_s 를 왼쪽 → 오른쪽으로 만든 뒤,

        J_b = Ad_{T_sb^{-1}} · J_s

      로 body Jacobian 으로 변환한다.

    Parameters
    ----------
    theta  : (n,)        현재 관절 각도 [rad]
    B_list : list of (6,) body screw axes

    Returns
    -------
    Jb : (6, n)  body Jacobian
    """
    theta = np.asarray(theta, dtype=float)
    n = len(B_list)

    Js = np.zeros((6, n))
    T_sb = np.eye(4)

    for i, (B, th) in enumerate(zip(B_list, theta)):
        Js[:, i] = adjoint(T_sb) @ np.asarray(B, dtype=float)
        T_sb = T_sb @ matrix_exp_se3(B, th)

    return adjoint(T_inv(T_sb)) @ Js
