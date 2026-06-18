from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum, auto

import numpy as np

import robojudo.environment
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlLocoMimicPipelineCfg
from robojudo.pipeline.rl_multi_policy_pipeline import PolicyManager, RlMultiPolicyPipeline
from robojudo.pipeline.rl_pipeline import PolicyWrapper
from robojudo.policy import PolicyCfg
from robojudo.utils.progress import ProgressBar

logger = logging.getLogger(__name__)


class PolicyInterpManager(PolicyManager):
    class InterpState(Enum):
        IDLE = auto()
        START = auto()
        IN_PROGRESS = auto()
        END = auto()

    DURATIONS_LOCO_MIMIC = [0, 75, 25]  # [start, in-progress, end] in steps
    DURATIONS_MIMIC_LOCO = [25, 75, 0]  # [start, in-progress, end] in steps

    def __init__(
        self,
        cfg_policy_loco: PolicyCfg,
        cfg_policies: list[PolicyCfg],
        env: Environment,
        default_override_dof_indices: list[int],
        loco_dof_pos: np.ndarray | None = None,
        device: str = "cpu",
    ):
        cfg_policies_all = [cfg_policy_loco] + cfg_policies
        super().__init__(cfg_policies_all, env, device)

        self.policy_loco_id = 0
        self.policy_mimic_num = len(cfg_policies)
        assert self.policy_mimic_num > 0, "At least one mimic policy is required for switching."
        self.policy_mimic_ids = list(range(1, self.policy_mimic_num + 1))
        self.policy_mimic_idx = 0

        # Interpolation variables
        self.interp_state = self.InterpState.IDLE
        self.interp_timestep = 0
        self.interp_durations = [20, 40, 20]  # [start, in-progress, end] in steps
        self.interp_pbar = None
        self.interp_callback_start = None
        self.interp_callback_end = None

        self.loco_dof_pos = loco_dof_pos if loco_dof_pos is not None else self.env.default_pos.copy()
        self.override_dof_pos = self.loco_dof_pos.copy()
        self.default_override_dof_indices = default_override_dof_indices
        self.override_dof_indices = list(default_override_dof_indices)

    def set_policy(self, policy_id: int):
        if not (0 <= policy_id < self.num_policies):
            raise ValueError(f"Policy id {policy_id} out of range [0, {self.num_policies})")
        self.warmup_policy_indices.discard(policy_id)
        self._current_policy_id = policy_id
        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        logger.warning(f"Switched to policy: {policy_id}: {self.policy.name}")

    def _interpolate_init(
        self,
        get_target_pos: Callable[[], np.ndarray],
        durations: list[int],
        callback_start=None,
        callback_end=None,
    ):
        self.interp_get_target_pos = get_target_pos
        self.interp_durations = durations
        self.interp_callback_start = callback_start
        self.interp_callback_end = callback_end
        self.interp_pbar = ProgressBar("Interpolation", durations[1])

        self.interp_state = self.InterpState.START
        # Starting Tasks
        self.timer.add(self._interpolate_start, delay_steps=durations[0])
        # Ending Tasks
        self.timer.add(self._interpolate_end, delay_steps=sum(durations) + 1)

    def _interpolate_start(self):
        if self.interp_state != self.InterpState.START:
            return
        if self.interp_callback_start is not None:
            self.interp_callback_start()
            self.interp_callback_start = None

        self.interp_start_pos = self.env.dof_pos.copy()
        self.interp_target_pos = self.interp_get_target_pos()
        self.interp_timestep = 0
        self.interp_state = self.InterpState.IN_PROGRESS

        # logger.debug("Interpolation started.")

    def _interpolate_end(self):
        if self.interp_state != self.InterpState.END:
            return
        self.override_dof_pos = self.interp_target_pos.copy()
        if self.interp_pbar:
            self.interp_pbar.close()
            self.interp_pbar = None
        if self.interp_callback_end is not None:
            self.interp_callback_end()
            self.interp_callback_end = None
        self.interp_state = self.InterpState.IDLE

        # logger.debug("Interpolation ended.")

    def _interpolate_step(self):
        if self.interp_state != self.InterpState.IN_PROGRESS:
            return

        if self.interp_pbar:
            self.interp_pbar.set(self.interp_timestep)

        progress = self.interp_timestep / self.interp_durations[1]
        alpha = min(progress, 1.0)
        self.override_dof_pos = (1 - alpha) * self.interp_start_pos + alpha * self.interp_target_pos

        if self.interp_timestep < self.interp_durations[1]:
            self.interp_timestep += 1
        else:
            self.interp_state = self.InterpState.END

    def toggle_mimic_policy(self, delta: int):
        # only switch mimic policy if current policy is locomotion
        if self.current_policy_id != self.policy_loco_id:
            logger.warning("Cannot switch mimic policy when policy is mimic.")
            return

        self.policy_mimic_idx = (self.policy_mimic_idx + delta) % self.policy_mimic_num
        policy_id = self.policy_mimic_ids[self.policy_mimic_idx]
        policy_name = self.policy_by_id(policy_id).name
        logger.info(f"Switch mimic policy to {self.policy_mimic_idx}: {policy_name}")

    def switch_to_loco(self):
        if self.current_policy_id == self.policy_loco_id and self.interp_state == self.InterpState.IDLE:
            logger.warning("Already in locomotion policy.")
            return
        if self.current_policy_id != self.policy_loco_id:
            self.policy_by_id(self.policy_loco_id).reset()
            self.warmup_policy_indices.add(self.policy_loco_id)
        self._interpolate_init(
            get_target_pos=lambda: self.loco_dof_pos,
            durations=self.DURATIONS_MIMIC_LOCO,
            callback_start=lambda: self.set_policy(self.policy_loco_id),
        )

    def _ctrl_ball_local(self, ctrl_data):
        if ctrl_data is None:
            return None
        soccer_obs = ctrl_data.get("soccer_obs", None)
        if soccer_obs is not None:
            ball_local = soccer_obs.get("ball_local", None) if hasattr(soccer_obs, "get") else getattr(soccer_obs, "ball_local", None)
            if ball_local is not None:
                return ball_local
        return ctrl_data.get("ball_local", None)

    def switch_to_mimic(self, env_data=None, ctrl_data=None):
        if self.current_policy_id != self.policy_loco_id:
            logger.warning("Already in mimic policy.")
            return
        policy_mimic_id = self.policy_mimic_ids[self.policy_mimic_idx]
        if getattr(self.policy_by_id(policy_mimic_id).policy, "requires_hard_switch", False):
            self._set_mimic_policy(policy_mimic_id, env_data=env_data, ctrl_data=ctrl_data)
            return
        self.policy_by_id(policy_mimic_id).reset()
        self.warmup_policy_indices.add(policy_mimic_id)
        self._interpolate_init(
            get_target_pos=lambda: self.policy_by_id(policy_mimic_id).get_init_dof_pos(),
            durations=self.DURATIONS_LOCO_MIMIC,
            callback_end=lambda: self._set_mimic_policy(policy_mimic_id, env_data=env_data, ctrl_data=ctrl_data),
        )

    def _set_mimic_policy(self, policy_id: int, env_data=None, ctrl_data=None):
        policy = self.policy_by_id(policy_id)
        init_root_state = getattr(policy, "get_init_root_state", None)
        self.set_policy(policy_id)
        if (
            init_root_state is not None
            and not getattr(policy.policy, "requires_hard_switch", False)
            and hasattr(self.env, "set_root_state")
        ):
            root_pos, root_quat = init_root_state()
            self.env.set_root_state(root_pos, root_quat, reset_alignment=False)  # pyright: ignore[reportAttributeAccessIssue]
        soccer_obs = getattr(self.env, "_soccer_obs", None)
        use_env_soccer_obs = bool(getattr(policy.policy, "use_env_soccer_obs", True))
        target_source = str(getattr(policy.policy, "soccer_target_source", "auto")).lower()
        wants_env_target = use_env_soccer_obs and target_source in {"auto", "env"}
        if wants_env_target and hasattr(self.env, "sync_soccer_objects_to_configured_targets"):
            self.env.sync_soccer_objects_to_configured_targets()  # pyright: ignore[reportAttributeAccessIssue]
            soccer_obs = getattr(self.env, "_soccer_obs", None)
        if hasattr(policy.policy, "reset"):
            try:
                ctrl_ball_local = self._ctrl_ball_local(ctrl_data) if target_source in {"auto", "ctrl"} else None
                base_quat = getattr(env_data, "base_quat", None)
                if ctrl_ball_local is not None:
                    policy.policy.reset(ball_local=ctrl_ball_local, base_quat=base_quat)
                elif soccer_obs is not None and wants_env_target:
                    policy.policy.reset(ball_local=soccer_obs["ball_local"], base_quat=base_quat)
                else:
                    policy.policy.reset(base_quat=base_quat)
            except TypeError:
                pass
        reset_dof_on_switch = bool(getattr(policy.policy, "reset_dof_on_hard_switch", False))
        if (
            (not getattr(policy.policy, "requires_hard_switch", False) or reset_dof_on_switch)
            and hasattr(self.env, "set_dof_state")
        ):
            init_dof_pos = policy.get_init_dof_pos()
            self.env.set_dof_state(init_dof_pos)  # pyright: ignore[reportAttributeAccessIssue]
        self.override_dof_indices = list(self.default_override_dof_indices)

    def step(self, env_data, ctrl_data):
        super().step(env_data, ctrl_data)
        self._interpolate_step()


@pipeline_registry.register
class RlLocoMimicPipeline(RlMultiPolicyPipeline):
    cfg: RlLocoMimicPipelineCfg

    @property
    def policy(self) -> PolicyWrapper:
        return self.policy_manager.policy

    def __init__(self, cfg: RlLocoMimicPipelineCfg):
        # Skip RlMultiPolicyPipeline initialization
        Pipeline.__init__(self, cfg=cfg)

        env_class: type[Environment] = getattr(robojudo.environment, self.cfg.env.env_type)
        self.env: Environment = env_class(cfg_env=self.cfg.env, device=self.device)

        self.ctrl_manager = CtrlManager(cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device)

        # upper body override
        self.num_upper_body_dof = self.cfg.upper_dof_num
        if upper_dof_pos_default := self.cfg.upper_dof_pos_default:
            loco_dof_pos = self.env.default_pos.copy()
            loco_dof_pos[-self.num_upper_body_dof :] = upper_dof_pos_default
            self.loco_dof_pos = loco_dof_pos
        else:
            self.loco_dof_pos = self.env.default_pos
        if override_dof_indices := self.cfg.upper_dof_override_indices:
            self.override_dof_indices = override_dof_indices
        else:
            self.override_dof_indices = list(range(-self.num_upper_body_dof, 0))

        self.policy_manager = PolicyInterpManager(
            cfg_policy_loco=self.cfg.loco_policy,
            cfg_policies=self.cfg.mimic_policies,
            env=self.env,
            default_override_dof_indices=self.override_dof_indices,
            loco_dof_pos=self.loco_dof_pos,
            device=self.device,
        )
        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self.loco_dof_pos = self.policy.get_init_dof_pos()
        self.policy_manager.loco_dof_pos = self.loco_dof_pos.copy()
        self.policy_manager.override_dof_pos = self.loco_dof_pos.copy()
        self.visualizer = self.env.visualizer

        self.freq = self.cfg.loco_policy.freq
        self.dt = 1.0 / self.freq

        self.policy_locomotion_mimic_flag = 0  # 0: locomotion, 1: mimic
        logger.info(f"Initial policy: {self.policy_manager.current_policy_id}: {self.policy.name}")

        self.self_check()
        self.reset()

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1

        commands = ctrl_data.get("COMMANDS", [])

        # Handle policy CALLBACK
        for callback in extras.get("CALLBACK", []):
            if callback == "[MOTION_DONE]":
                if (
                    self.policy_locomotion_mimic_flag == 1
                    and self.policy_manager.interp_state == self.policy_manager.InterpState.IDLE
                ):
                    commands.append("[POLICY_LOCO]")
                    logger.info("Mimic motion done, switch to locomotion policy.")

        for command in commands:
            if command == "[SHUTDOWN]":
                logger.warning("Emergency shutdown!")
                self.env.shutdown()
            elif command == "[SIM_REBORN]":
                if hasattr(self.env, "reborn"):
                    logger.warning("Simulation Env reborn!")
                    self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
            elif command.startswith("[POLICY_SWITCH]"):
                switch_target = command.split(",")[1]
                if switch_target == "NEXT":
                    self.policy_manager.toggle_mimic_policy(1)
                elif switch_target == "LAST":
                    self.policy_manager.toggle_mimic_policy(-1)
            elif command == "[POLICY_LOCO]":
                self.policy_locomotion_mimic_flag = 0
                self._clear_soccer_perception_cache(ctrl_data)
                self.policy_manager.switch_to_loco()
            elif command == "[POLICY_MIMIC]":
                self.policy_locomotion_mimic_flag = 1
                self.policy_manager.switch_to_mimic(env_data=env_data, ctrl_data=ctrl_data)

        soccer_goal_world = extras.get("soccer_goal_world", None)
        if soccer_goal_world is not None and hasattr(self.env, "set_soccer_goal_marker_world"):
            self.env.set_soccer_goal_marker_world(soccer_goal_world)  # pyright: ignore[reportAttributeAccessIssue]

        self.ctrl_manager.post_step_callback(ctrl_data)

        self.policy.post_step_callback(commands)
        if self.visualizer is not None:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)

        # # Handle policy switch after step to avoid mid-step change
        self.policy_manager.step(env_data, ctrl_data)

        self.safety_check()
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )

    def _clear_soccer_perception_cache(self, ctrl_data=None):
        for controller in self.ctrl_manager.controllers.values():
            provider = getattr(controller.inst, "provider", None)
            clear = getattr(provider, "clear", None)
            if callable(clear):
                clear()
        if ctrl_data is not None:
            ctrl_data.pop("soccer_obs", None)
            ctrl_data.pop("ball_local", None)
            ctrl_data["soccer_obs_valid"] = False

    def step(self, dry_run=False):
        self.env.update()
        env_data = self.env.get_data()
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)

        commands = ctrl_data.get("COMMANDS", [])
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")

        if self.policy_manager.current_policy_id == self.policy_manager.policy_loco_id:
            ctrl_data["ref_dof_pos"] = self.policy.obs_adapter.fit(self.policy_manager.override_dof_pos)

        obs, extras = self.policy.get_observation(env_data, ctrl_data)

        pd_target = self.policy.get_pd_target(obs)

        if self.policy_manager.current_policy_id == self.policy_manager.policy_loco_id:
            override_dof_indices = self.policy_manager.override_dof_indices
            pd_target[override_dof_indices] = self.policy_manager.override_dof_pos[override_dof_indices]

        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))
            self.log_soccer_frame(env_data, ctrl_data, obs, extras, pd_target)
            # logger.debug(pd_target)

        self.post_step_callback(env_data, ctrl_data, extras, pd_target)

    def prepare(self):
        init_motor_angle = self.loco_dof_pos.copy()
        super().prepare(init_motor_angle=init_motor_angle)


if __name__ == "__main__":
    pass
