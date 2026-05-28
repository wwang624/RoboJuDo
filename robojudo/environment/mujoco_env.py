import logging
import time
from pathlib import Path

import mujoco
import mujoco_viewer
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.utils.util_func import my_quat_rotate_np, quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        self._xml_runtime_path = self._prepare_runtime_xml(cfg_env.xml)
        self.model = mujoco.MjModel.from_xml_path(self._xml_runtime_path)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        self._refresh_dof_addr_arrays()
        self._refresh_soccer_object_handles()
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        self.viewer = mujoco_viewer.MujocoViewer(
            self.model,
            self.data,
            width=1200,
            height=900,
            hide_menus=True,
            diable_key_callbacks=True,
        )
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10.0
        self.viewer.cam.azimuth = 180.0
        # self.viewer._paused = True

        if cfg_env.visualize_extras:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self.last_time = time.time()
        self.random_heading = cfg_env.random_heading

        self._apply_random_heading()
        self.sync_soccer_objects_to_configured_targets()

        self.update()  # get initial state

    def _prepare_runtime_xml(self, xml_path: str) -> str:
        if not self.cfg_env.soccer_objects_enabled:
            return xml_path

        path = Path(xml_path)
        xml_text = path.read_text(encoding="utf-8")
        ball_name = self.cfg_env.soccer_ball_body_name
        has_ball = f'name="{ball_name}"' in xml_text
        has_goal_marker = 'name="soccer_goal_marker"' in xml_text
        if has_ball and has_goal_marker:
            return str(path)

        ball_pos = self.cfg_env.soccer_ball_pos
        goal_pos = self.cfg_env.soccer_goal_marker_pos
        ball_block = "" if has_ball else f"""
    <body name="{ball_name}" pos="{ball_pos[0]:.5f} {ball_pos[1]:.5f} {ball_pos[2]:.5f}" quat="1 0 0 0">
      <joint limited="false" name="{ball_name}" type="free"/>
      <geom name="{ball_name}_geom" type="sphere" size="{self.cfg_env.soccer_ball_radius:.5f}"
            contype="1" conaffinity="1" friction="0.8 0.01 0.0001" density="150"
            solref="0.02 1" solimp="0.9 0.95 0.01" rgba="1 1 1 1"/>
    </body>
"""
        goal_block = "" if has_goal_marker else f"""
    <body name="soccer_goal_marker" mocap="true" pos="{goal_pos[0]:.5f} {goal_pos[1]:.5f} {goal_pos[2]:.5f}">
      <geom name="soccer_goal_marker_geom" type="sphere" size="{self.cfg_env.soccer_goal_marker_radius:.5f}"
            contype="0" conaffinity="0" rgba="1 0 0 1"/>
    </body>
"""
        if "</worldbody>" not in xml_text:
            raise RuntimeError(f"MuJoCo XML at {xml_path} is missing </worldbody>; cannot inject soccer objects.")
        split_token = "</worldbody>"
        head, sep, tail = xml_text.rpartition(split_token)
        if not sep:
            raise RuntimeError(f"MuJoCo XML at {xml_path} is missing </worldbody>; cannot inject soccer objects.")
        xml_text = head + ball_block + goal_block + "\n" + sep + tail

        runtime_path = path.with_name(f"{path.stem}_runtime_soccer.xml")
        runtime_path.write_text(xml_text, encoding="utf-8")
        return str(runtime_path)

    def _refresh_dof_addr_arrays(self):
        qpos_addr = []
        qvel_addr = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"MuJoCo XML is missing configured robot joint '{joint_name}'.")
            qpos_addr.append(int(self.model.jnt_qposadr[joint_id]))
            qvel_addr.append(int(self.model.jnt_dofadr[joint_id]))
        self._dof_qpos_addr = np.asarray(qpos_addr, dtype=np.int32)
        self._dof_qvel_addr = np.asarray(qvel_addr, dtype=np.int32)

    def _refresh_soccer_object_handles(self):
        self._soccer_ball_qpos_addr = None
        self._soccer_ball_qvel_addr = None
        self._soccer_goal_mocap_id = None
        if not self.cfg_env.soccer_objects_enabled:
            return
        ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.cfg_env.soccer_ball_body_name)
        if ball_joint_id >= 0:
            self._soccer_ball_qpos_addr = int(self.model.jnt_qposadr[ball_joint_id])
            self._soccer_ball_qvel_addr = int(self.model.jnt_dofadr[ball_joint_id])
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "soccer_goal_marker")
        if goal_body_id >= 0:
            mocap_id = int(self.model.body_mocapid[goal_body_id])
            self._soccer_goal_mocap_id = mocap_id if mocap_id >= 0 else None

    def _world_pos_from_pelvis_local(self, local_pos: list[float] | np.ndarray) -> np.ndarray:
        pelvis_pos_w = self.data.qpos.astype(np.float32)[:3]
        pelvis_quat_xyzw = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        return pelvis_pos_w + my_quat_rotate_np(pelvis_quat_xyzw, np.asarray(local_pos, dtype=np.float32))

    def sync_soccer_objects_to_configured_targets(self):
        """Sim-only: place soccer objects from configured world or pelvis-local targets."""
        if not self.cfg_env.soccer_objects_enabled:
            return
        from_local = self.cfg_env.soccer_objects_from_local_targets
        ball_pos_w = None
        goal_pos_w = None
        if self._soccer_ball_qpos_addr is not None:
            if from_local and self.cfg_env.soccer_ball_local is not None:
                ball_pos_w = self._world_pos_from_pelvis_local(self.cfg_env.soccer_ball_local)
            elif not from_local:
                ball_pos_w = np.asarray(self.cfg_env.soccer_ball_pos, dtype=np.float32)
        if ball_pos_w is not None:
            self.data.qpos[self._soccer_ball_qpos_addr : self._soccer_ball_qpos_addr + 3] = ball_pos_w
            self.data.qpos[self._soccer_ball_qpos_addr + 3 : self._soccer_ball_qpos_addr + 7] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float32
            )
            if self._soccer_ball_qvel_addr is not None:
                self.data.qvel[self._soccer_ball_qvel_addr : self._soccer_ball_qvel_addr + 6] = 0.0
        if self._soccer_goal_mocap_id is not None:
            if from_local and self.cfg_env.soccer_goal_marker_local is not None:
                goal_pos_w = self._world_pos_from_pelvis_local(self.cfg_env.soccer_goal_marker_local)
            elif not from_local:
                goal_pos_w = np.asarray(self.cfg_env.soccer_goal_marker_pos, dtype=np.float32)
        if goal_pos_w is not None:
            self.data.mocap_pos[self._soccer_goal_mocap_id] = goal_pos_w
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        self._soccer_obs = self._get_soccer_obs()
        logger.info(
            "Synced soccer objects from %s targets: ball_w=%s goal_w=%s",
            "local" if from_local else "world",
            None if ball_pos_w is None else np.round(ball_pos_w, 4).tolist(),
            None if goal_pos_w is None else np.round(goal_pos_w, 4).tolist(),
        )

    def sync_soccer_objects_to_local_targets(self):
        self.sync_soccer_objects_to_configured_targets()

    def _apply_random_heading(self):
        """Rotate the root body by a random yaw if random_heading is enabled."""
        if not self.random_heading:
            return
        yaw = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        q = self.data.qpos[3:7].copy()  # MuJoCo [w, x, y, z]
        # Pre-multiply by yaw rotation q_yaw=[c,0,0,s]: q_new = q_yaw ⊗ q
        self.data.qpos[3] = c * q[0] - s * q[3]
        self.data.qpos[4] = c * q[1] - s * q[2]
        self.data.qpos[5] = c * q[2] + s * q[1]
        self.data.qpos[6] = c * q[3] + s * q[0]

    def reborn(self, init_qpos=None):
        if init_qpos is not None:
            self.data.qpos[0:7] = init_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
            self._apply_random_heading()
        self.sync_soccer_objects_to_configured_targets()
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def set_root_state(self, root_pos: np.ndarray, root_quat_xyzw: np.ndarray, reset_alignment: bool = True):
        self.data.qpos[0:3] = np.asarray(root_pos, dtype=np.float32)
        self.data.qpos[3:7] = np.asarray(root_quat_xyzw, dtype=np.float32)[[3, 0, 1, 2]]
        self.data.qvel[:6] = 0.0
        self.sync_soccer_objects_to_configured_targets()
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        self.update()
        if reset_alignment and self.born_place_align:
            self.set_born_place()
            self.update()

    def set_dof_state(self, dof_pos: np.ndarray, dof_vel: np.ndarray | None = None):
        dof_pos = np.asarray(dof_pos, dtype=np.float32)
        assert len(dof_pos) == self.num_dofs, "dof_pos len should be num_dofs of env"
        self.data.qpos[self._dof_qpos_addr] = dof_pos
        if dof_vel is None:
            self.data.qvel[self._dof_qvel_addr] = 0.0
        else:
            dof_vel = np.asarray(dof_vel, dtype=np.float32)
            assert len(dof_vel) == self.num_dofs, "dof_vel len should be num_dofs of env"
            self.data.qvel[self._dof_qvel_addr] = dof_vel
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        self.update()

    def _get_soccer_obs(self) -> dict | None:
        if self._soccer_ball_qpos_addr is None or self._soccer_goal_mocap_id is None:
            return None

        pelvis_pos_w = self.data.qpos.astype(np.float32)[:3]
        pelvis_quat_wxyz = self.data.qpos.astype(np.float32)[3:7]
        pelvis_quat_xyzw = pelvis_quat_wxyz[[1, 2, 3, 0]]
        ball_pos_w = self.data.qpos[self._soccer_ball_qpos_addr : self._soccer_ball_qpos_addr + 3].astype(np.float32)
        goal_pos_w = self.data.mocap_pos[self._soccer_goal_mocap_id].astype(np.float32)

        return {
            "pelvis_pos_w": pelvis_pos_w,
            "pelvis_quat_xyzw": pelvis_quat_xyzw,
            "base_ang_vel": self.data.qvel.astype(np.float32)[3:6],
            "ball_world": ball_pos_w,
            "ball_local": quat_rotate_inverse_np(pelvis_quat_xyzw, ball_pos_w - pelvis_pos_w).astype(np.float32),
            "goal_local": quat_rotate_inverse_np(pelvis_quat_xyzw, goal_pos_w - pelvis_pos_w).astype(np.float32),
        }

    def set_soccer_goal_marker_world(self, goal_pos_w: np.ndarray):
        if self._soccer_goal_mocap_id is None:
            return
        self.data.mocap_pos[self._soccer_goal_mocap_id] = np.asarray(goal_pos_w, dtype=np.float32)
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        self._soccer_obs = self._get_soccer_obs()

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)
        if hasattr(self, "model"):
            self._refresh_dof_addr_arrays()

    def self_check(self):
        pass

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        dof_pos = self.data.qpos[self._dof_qpos_addr].astype(np.float32)
        dof_vel = self.data.qvel[self._dof_qvel_addr].astype(np.float32)

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        ang_vel = self.data.qvel.astype(np.float32)[3:6]
        base_pos = self.data.qpos.astype(np.float32)[:3]
        lin_vel = self.data.qvel.astype(np.float32)[0:3]

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

        self._soccer_obs = self._get_soccer_obs()

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)

        self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        if self.viewer.is_alive:
            self.viewer.render()

        for _ in range(self.sim_decimation):
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)

            self.data.ctrl = torque

            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.update(simple=True)
        self.update(simple=False)

    def shutdown(self):
        self.viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)
