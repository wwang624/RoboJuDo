from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import SelfPolicyCfg
from robojudo.utils.util_func import command_remap, get_gravity_orientation


@policy_registry.register
class SelfPolicy(Policy):
    cfg_policy: SelfPolicyCfg

    def __init__(self, cfg_policy, device):
        self._uses_onnx = str(cfg_policy.policy_file).endswith(".onnx")
        if self._uses_onnx:
            cfg_policy.disable_autoload = True
        super().__init__(cfg_policy=cfg_policy, device=device)

        self.onnx_session: ort.InferenceSession | None = None
        self.onnx_input_name: str | None = None
        if self._uses_onnx:
            policy_file = Path(self.cfg_policy.policy_file)
            self.onnx_session = ort.InferenceSession(str(policy_file), providers=["CPUExecutionProvider"])
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name

        self.obs_scales = self.cfg_policy.obs_scales
        self.max_cmd = np.array(self.cfg_policy.max_cmd)
        self.commands_map = self.cfg_policy.commands_map

        self.num_actions = self.cfg_policy.num_actions
        self.single_obs_dim = self.cfg_policy.single_obs_dim
        self.obs_history_len = self.cfg_policy.obs_history_len
        self.include_phase_obs = bool(getattr(self.cfg_policy, "include_phase_obs", True))
        self.num_obs = self.single_obs_dim * self.obs_history_len
        self.action_scales = np.asarray(self.cfg_policy.action_scales, dtype=np.float32)

        self.obs_history = deque(maxlen=self.obs_history_len)
        self.reset()

    def reset(self):
        self.timestep = 0
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_history.clear()
        for _ in range(self.obs_history_len):
            self.obs_history.append(np.zeros(self.single_obs_dim, dtype=np.float32))

    def post_step_callback(self, commands=None):
        self.timestep += 1

    def _get_phase(self):
        cycle_time = 0.8
        return (self.timestep * self.dt) % cycle_time / cycle_time

    def _get_commands(self, ctrl_data):
        commands = np.zeros(3, dtype=np.float32)
        for key in ctrl_data.keys():
            if key in ["JoystickCtrl", "UnitreeCtrl"]:
                axes = ctrl_data[key]["axes"]
                lx, ly, rx = axes["LeftX"], axes["LeftY"], axes["RightX"]
                commands[0] = command_remap(ly, self.commands_map[0])
                commands[1] = command_remap(lx, self.commands_map[1])
                commands[2] = command_remap(rx, self.commands_map[2])
                break
            if key == "KeyboardCtrl":
                for event in ctrl_data[key]["keyboard_event"]:
                    if event["type"] != "keyboard":
                        continue
                    value = event["pressed"] * 1.5
                    name = event["name"]
                    if name == "w":
                        commands[0] = command_remap(value, self.commands_map[0])
                    elif name == "s":
                        commands[0] = command_remap(-value, self.commands_map[0])
                    elif name == "a":
                        commands[1] = command_remap(-value, self.commands_map[1])
                    elif name == "d":
                        commands[1] = command_remap(value, self.commands_map[1])
                    elif name == "e":
                        commands[2] = command_remap(value, self.commands_map[2])
                    elif name == "q":
                        commands[2] = command_remap(-value, self.commands_map[2])
                break
        return commands

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self._uses_onnx:
            assert self.onnx_session is not None and self.onnx_input_name is not None
            outputs = self.onnx_session.run(None, {self.onnx_input_name: obs.reshape(1, -1).astype(np.float32)})
            actions = np.asarray(outputs[0], dtype=np.float32).squeeze()
        else:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                actions_tensor = self.model(obs_tensor).cpu()
            actions = actions_tensor.numpy().squeeze()
        actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions
        self.last_action = actions.copy()

        processed_actions = actions
        if self.action_clip is not None:
            processed_actions = np.clip(processed_actions, -self.action_clip, self.action_clip)
        return processed_actions * self.action_scales

    def get_observation(self, env_data, ctrl_data):
        phase = self._get_phase()
        commands = self._get_commands(ctrl_data)
        clipped_commands = np.clip(commands, -self.max_cmd, self.max_cmd)
        if np.linalg.norm(commands[:3]) < 0.1:
            phase = 0.0

        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)
        gravity_orientation = get_gravity_orientation(env_data.base_quat)
        dof_pos_rel = (env_data.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dof_vel_rel = env_data.dof_vel * self.obs_scales.dof_vel

        obs_parts = [
            env_data.base_ang_vel * self.obs_scales.ang_vel,
            gravity_orientation,
            clipped_commands * self.obs_scales.command,
            dof_pos_rel,
            dof_vel_rel,
            self.last_action,
        ]
        if self.include_phase_obs:
            obs_parts.append(np.array([sin_phase, cos_phase], dtype=np.float32))
        single_obs = np.concatenate(obs_parts).astype(np.float32)
        if single_obs.shape[0] != self.single_obs_dim:
            raise RuntimeError(
                f"SelfPolicy single obs dim mismatch: built {single_obs.shape[0]}, expected {self.single_obs_dim}. "
                f"include_phase_obs={self.include_phase_obs}"
            )
        self.obs_history.append(single_obs)
        obs = np.concatenate(list(self.obs_history))
        extras = {"phase": phase, "commands": commands}
        return obs, extras

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        base_pos = env_data["base_pos"]
        base_quat = env_data["base_quat"]
        command_x = extras["commands"][0]
        command_y = extras["commands"][1]
        command_yaw = extras["commands"][2]

        visualizer.draw_arrow(base_pos, base_quat, [command_x, 0, 0], color=[1, 0, 0, 1], scale=2, horizontal_only=True, id=0)
        visualizer.draw_arrow(base_pos, base_quat, [0, command_y, 0], color=[0, 1, 0, 1], scale=2, horizontal_only=True, id=1)
        visualizer.draw_arrow(
            base_pos + np.array([0.0, 0, 0.6]),
            base_quat,
            [0, command_yaw, 0],
            color=[1, 1, 1, 1],
            scale=2,
            horizontal_only=True,
            id=2,
        )
