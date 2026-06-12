"""
OpenManipulator-Y kinematic model.

The default TCP keeps the original HW2 flange/gripper-base convention.  Maze
code passes tcp_offset=(0, -0.315, 0) to track the physical tip body origin.
"""

import numpy as np

from robot.kinematics import adjoint, T_inv


LINK6_HOME_POS = np.array([0.0, -0.113, 0.7535], dtype=float)
DEFAULT_TCP_OFFSET = np.array([0.0, -0.109, 0.0], dtype=float)


def define_model(tcp_offset=DEFAULT_TCP_OFFSET):
    """
    Return space screws, home TCP pose, and body screws for OMY.

    Parameters
    ----------
    tcp_offset : array-like, shape (3,)
        TCP offset expressed in the link6 frame.  Use (0, -0.315, 0) for the
        maze tip.
    """
    tcp_offset = np.asarray(tcp_offset, dtype=float).reshape(3)

    omega = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
    ]

    q_joints = [
        np.array([0.0, 0.0, 0.1715]),
        np.array([0.0, -0.1215, 0.1715]),
        np.array([0.0, -0.1215, 0.4185]),
        np.array([0.0, 0.0, 0.6380]),
        np.array([0.0, -0.113, 0.6380]),
        np.array([0.0, -0.113, 0.7535]),
    ]

    S_list = []
    for w, q in zip(omega, q_joints):
        S_list.append(np.concatenate([w, -np.cross(w, q)]))

    M = np.eye(4)
    M[:3, 3] = LINK6_HOME_POS + tcp_offset

    Ad_Minv = adjoint(T_inv(M))
    B_list = [Ad_Minv @ S for S in S_list]
    return S_list, M, B_list
