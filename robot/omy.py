import dataclasses
from typing import Optional, Tuple

import mujoco
import numpy as np

from reference.dream_math import T_from_Rp


@dataclasses.dataclass
class OMYConfig:
    xml_path: str
    ee_body: str = "link6"
    base_body: str = "base_unit"
    tcp_offset_in_ee: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0.0, -0.315, 0.0], dtype=float)
    )
    arm_joint_names: Tuple[str, ...] = (
        "Joint1",
        "Joint2",
        "Joint3",
        "Joint4",
        "Joint5",
        "Joint6",
    )
    gripper_act_name: str = "Gripper"
    dt: float = 0.002


class OMYRobot:
    def __init__(self, cfg: OMYConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.cfg.dt = float(self.model.opt.timestep)

        self.ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, cfg.ee_body
        )
        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, cfg.base_body
        )

        self.arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in cfg.arm_joint_names
        ]
        self.arm_dof_ids = np.array(
            [self.model.jnt_dofadr[j] for j in self.arm_joint_ids], dtype=int
        )
        self.arm_qpos_adrs = np.array(
            [self.model.jnt_qposadr[j] for j in self.arm_joint_ids], dtype=int
        )
        self.arm_qpos_lo = np.array(
            [self.model.jnt_range[j, 0] for j in self.arm_joint_ids], dtype=float
        )
        self.arm_qpos_hi = np.array(
            [self.model.jnt_range[j, 1] for j in self.arm_joint_ids], dtype=float
        )

        self.arm_act_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in cfg.arm_joint_names
            ],
            dtype=int,
        )

        self.gripper_act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, cfg.gripper_act_name
        )

        self._jacp = np.zeros((3, self.model.nv), dtype=float)
        self._jacr = np.zeros((3, self.model.nv), dtype=float)
        self._M = np.zeros((self.model.nv, self.model.nv), dtype=float)
        self._Jpos_prev: Optional[np.ndarray] = None

        self.q_home = np.deg2rad(np.array([0, -45, 90, -45, 90, 0], dtype=float))

    def get_q(self) -> np.ndarray:
        return self.data.qpos.copy()

    def set_q(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float).reshape(-1)
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        self._Jpos_prev = None

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        self.data.ctrl[:] = np.asarray(ctrl, dtype=float).reshape(-1)

    def reset_home_keyframe(self, q_home=None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if q_home is not None:
            self.q_home = np.asarray(q_home, dtype=float).reshape(-1)
        self.data.qpos[self.arm_qpos_adrs] = self.q_home
        mujoco.mj_forward(self.model, self.data)
        self._Jpos_prev = None

    def tcp_pose_world(self) -> np.ndarray:
        p = self.data.xpos[self.ee_body_id].copy()
        R = self.data.xmat[self.ee_body_id].reshape(3, 3).copy()
        p_tcp = p + R @ self.cfg.tcp_offset_in_ee.reshape(3)
        return T_from_Rp(R, p_tcp)

    def tcp_position_world(self) -> np.ndarray:
        return self.tcp_pose_world()[:3, 3].copy()

    def forward_kinematics_arm(self, q_arm: np.ndarray) -> np.ndarray:
        q_full = self.get_q()
        q_full[self.arm_qpos_adrs] = np.asarray(q_arm, dtype=float).reshape(-1)
        self.set_q(q_full)
        return self.tcp_pose_world()

    def space_jacobian_tcp(self) -> np.ndarray:
        self._jacp[:] = 0.0
        self._jacr[:] = 0.0
        p_tcp = self.tcp_position_world()
        mujoco.mj_jac(
            self.model, self.data, self._jacp, self._jacr, p_tcp, self.ee_body_id
        )
        return np.vstack([self._jacr, self._jacp])

    def space_jacobian_tcp_arm(self) -> np.ndarray:
        return self.space_jacobian_tcp()[:, self.arm_dof_ids]

    def tcp_twist_world(self) -> np.ndarray:
        return self.space_jacobian_tcp() @ self.data.qvel

    def tcp_linvel_world(self) -> np.ndarray:
        return self.tcp_twist_world()[3:6]

    def mass_matrix_full(self) -> np.ndarray:
        mujoco.mj_fullM(self.model, self._M, self.data.qM)
        return self._M.copy()

    def mass_matrix_arm(self) -> np.ndarray:
        idx = self.arm_dof_ids
        return self.mass_matrix_full()[np.ix_(idx, idx)].copy()

    def bias_forces_full(self) -> np.ndarray:
        return self.data.qfrc_bias.copy()

    def bias_forces_arm(self) -> np.ndarray:
        return self.bias_forces_full()[self.arm_dof_ids].copy()
