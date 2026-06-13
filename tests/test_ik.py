import numpy as np

from control.clik import solve_ik
from maze.waypoint_gen import build_se3_targets
from robot.kinematics import body_poe_fk
from robot.model import define_model


def test_ik_at_entrance_position():
    _, M, B_list = define_model(tcp_offset=(0.0, -0.315, 0.0))
    T_target = build_se3_targets([(0, 1)])[0]
    q_init = np.deg2rad([0, -45, 90, -45, 90, 0])
    joint_lower_limits = np.array(
        [-2 * np.pi, -2 * np.pi, -2.618, -2 * np.pi, -2 * np.pi, -2 * np.pi]
    )
    joint_upper_limits = np.array(
        [2 * np.pi, 2 * np.pi, 2.618, 2 * np.pi, 2 * np.pi, 2 * np.pi]
    )
    q, ok = solve_ik(
        T_target,
        B_list,
        M,
        q_init,
        K_p=np.diag([3.0, 3.0, 3.0, 8.0, 8.0, 8.0]),
        max_iter=800,
        tol=5e-4,
        joint_lower_limits=joint_lower_limits,
        joint_upper_limits=joint_upper_limits,
        damping=0.08,
        dt=0.04,
    )
    assert ok
    T_check = body_poe_fk(q, B_list, M)
    assert np.allclose(T_check[:3, 3], T_target[:3, 3], atol=2e-3)
