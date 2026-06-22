from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import SoccerPolicyCfg
from robojudo.utils.util_func import calc_heading_quat_np, my_quat_rotate_np, quat_rotate_inverse_np

logger = logging.getLogger(__name__)


CTRL_DETECTOR_WAIT_FRAME = 5


def _csv_to_list(raw: str) -> list[str]:
    if raw is None or raw == "":
        return []
    return [item for item in raw.split(",") if item != ""]


def _csv_to_float_array(raw: str) -> np.ndarray:
    values = _csv_to_list(raw)
    if not values:
        return np.array([], dtype=np.float32)
    return np.asarray([float(v) for v in values], dtype=np.float32)


def _decode_metadata_list(raw: str) -> list[str]:
    if raw is None or raw == "":
        return []
    raw = raw.strip()
    if raw.startswith("["):
        return list(json.loads(raw))
    return _csv_to_list(raw)


def _decode_metadata_array(raw: str) -> np.ndarray:
    if raw is None or raw == "":
        return np.zeros((0,), dtype=np.float32)
    raw = raw.strip()
    if raw.startswith("["):
        return np.asarray(json.loads(raw), dtype=np.float32)
    return _csv_to_float_array(raw)


def _quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.asarray(quat, dtype=np.float32)[[1, 2, 3, 0]]


def _get_optional(mapping, key: str, default=None):
    if mapping is None:
        return default
    if hasattr(mapping, "get"):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


@dataclass
class SoccerPolicyMetadata:
    joint_names: list[str]
    default_joint_pos: np.ndarray
    joint_stiffness: np.ndarray
    joint_damping: np.ndarray
    action_scale: np.ndarray
    observation_names: list[str]
    anchor_body_name: str
    body_names: list[str]
    motion_names: list[str]
    motion_lengths: np.ndarray
    motion_kick_leg_names: list[str]
    final_anchor_pos: np.ndarray


class OnnxSoccerPolicy:
    def __init__(self, onnx_path: str, providers: list[str] | None = None):
        self.session = ort.InferenceSession(onnx_path, providers=providers or ["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.output_index = {name: idx for idx, name in enumerate(self.output_names)}
        self.model_meta = self.session.get_modelmeta().custom_metadata_map
        self.uses_motion_index = "motion_idx" in self.input_names
        self._validate_export_contract(onnx_path)
        self.metadata = SoccerPolicyMetadata(
            joint_names=_decode_metadata_list(self.model_meta.get("joint_names", "")),
            default_joint_pos=_csv_to_float_array(self.model_meta.get("default_joint_pos", "")),
            joint_stiffness=_csv_to_float_array(self.model_meta.get("joint_stiffness", "")),
            joint_damping=_csv_to_float_array(self.model_meta.get("joint_damping", "")),
            action_scale=_csv_to_float_array(self.model_meta.get("action_scale", "")),
            observation_names=_decode_metadata_list(self.model_meta.get("observation_names", "")),
            anchor_body_name=self.model_meta.get("anchor_body_name", "torso_link"),
            body_names=_decode_metadata_list(self.model_meta.get("body_names", "")),
            motion_names=_decode_metadata_list(self.model_meta.get("motion_names", "")),
            motion_lengths=_csv_to_float_array(self.model_meta.get("motion_lengths", "")),
            motion_kick_leg_names=_decode_metadata_list(self.model_meta.get("motion_kick_leg_names", "")),
            final_anchor_pos=_decode_metadata_array(self.model_meta.get("final_anchor_pos", "")),
        )
        self.obs_dim = int(self.session.get_inputs()[0].shape[-1])
        self.is_recurrent = {"h_in", "c_in", "time_step"}.issubset(self.input_names)
        self.recurrent_shape = self._infer_recurrent_shape()
        self.reference = self._precompute_reference_cache()

    def _validate_export_contract(self, onnx_path: str) -> None:
        required_meta = {
            "joint_names",
            "default_joint_pos",
            "joint_stiffness",
            "joint_damping",
            "action_scale",
            "observation_names",
            "anchor_body_name",
            "body_names",
        }
        missing_meta = sorted(key for key in required_meta if key not in self.model_meta)
        if missing_meta:
            raise RuntimeError(f"ONNX model is missing soccer export metadata {missing_meta}: {onnx_path}")

        required_outputs = {
            "actions",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }
        if self.uses_motion_index:
            required_outputs.update({"motion_length", "motion_idx_selected", "time_step_total"})
        else:
            required_outputs.add("time_step_total")
        missing_outputs = sorted(name for name in required_outputs if name not in self.output_index)
        if missing_outputs:
            raise RuntimeError(f"ONNX model is missing soccer reference outputs {missing_outputs}: {onnx_path}")
        if "time_step" not in self.input_names:
            raise RuntimeError(f"ONNX model is missing required `time_step` input: {onnx_path}")

    def _infer_recurrent_shape(self) -> tuple[int, int] | None:
        if not self.is_recurrent:
            return None
        h_shape = self.session.get_inputs()[self.input_names.index("h_in")].shape
        return int(h_shape[0]), int(h_shape[2])

    def _zero_inputs(self, time_step: int, motion_idx: int = 0) -> dict[str, np.ndarray]:
        inputs: dict[str, np.ndarray] = {"obs": np.zeros((1, self.obs_dim), dtype=np.float32)}
        if self.is_recurrent:
            assert self.recurrent_shape is not None
            num_layers, hidden_dim = self.recurrent_shape
            inputs["h_in"] = np.zeros((num_layers, 1, hidden_dim), dtype=np.float32)
            inputs["c_in"] = np.zeros((num_layers, 1, hidden_dim), dtype=np.float32)
        if self.uses_motion_index:
            inputs["motion_idx"] = np.array([[motion_idx]], dtype=np.float32)
        inputs["time_step"] = np.array([[time_step]], dtype=np.float32)
        return inputs

    def _run_raw(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        feed = {name: inputs[name] for name in self.input_names}
        return self.session.run(None, feed)

    def _precompute_reference_cache(self) -> dict[str, np.ndarray]:
        if self.uses_motion_index:
            if self.metadata.motion_lengths.size == 0 or not self.metadata.motion_names:
                raise RuntimeError("Bundled soccer ONNX is missing motion_names/motion_lengths metadata.")
            motion_count = len(self.metadata.motion_names)
            motion_lengths = self.metadata.motion_lengths.astype(np.int32)
            max_time = int(np.max(motion_lengths))
            cache: dict[str, np.ndarray] = {
                "joint_pos": np.zeros((motion_count, max_time, len(self.metadata.joint_names)), dtype=np.float32),
                "joint_vel": np.zeros((motion_count, max_time, len(self.metadata.joint_names)), dtype=np.float32),
                "body_pos_w": np.zeros((motion_count, max_time, len(self.metadata.body_names), 3), dtype=np.float32),
                "body_quat_w": np.zeros((motion_count, max_time, len(self.metadata.body_names), 4), dtype=np.float32),
                "body_ang_vel_w": np.zeros((motion_count, max_time, len(self.metadata.body_names), 3), dtype=np.float32),
            }
            for motion_idx in range(motion_count):
                for step in range(int(motion_lengths[motion_idx])):
                    outputs = self._run_raw(self._zero_inputs(step, motion_idx))
                    for key in cache:
                        cache[key][motion_idx, step] = outputs[self.output_index[key]].squeeze(0).astype(np.float32)
            return {"motion_lengths": motion_lengths, **cache}

        first = self._run_raw(self._zero_inputs(0))
        time_step_total = int(first[self.output_index["time_step_total"]].reshape(-1)[0])
        cache_single: dict[str, list[np.ndarray]] = {
            "joint_pos": [],
            "joint_vel": [],
            "body_pos_w": [],
            "body_quat_w": [],
            "body_ang_vel_w": [],
        }
        for step in range(time_step_total):
            outputs = self._run_raw(self._zero_inputs(step))
            for key in cache_single:
                cache_single[key].append(outputs[self.output_index[key]].squeeze(0).astype(np.float32))
        return {
            "time_step_total": np.array(time_step_total, dtype=np.int32),
            **{key: np.stack(values, axis=0) for key, values in cache_single.items()},
        }

    def act(
        self,
        obs: np.ndarray,
        time_step: int,
        motion_idx: int = 0,
        h_in: np.ndarray | None = None,
        c_in: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        inputs: dict[str, np.ndarray] = {"obs": obs.astype(np.float32), "time_step": np.array([[time_step]], dtype=np.float32)}
        if self.uses_motion_index:
            inputs["motion_idx"] = np.array([[motion_idx]], dtype=np.float32)
        if self.is_recurrent:
            assert h_in is not None and c_in is not None
            inputs["h_in"] = h_in.astype(np.float32)
            inputs["c_in"] = c_in.astype(np.float32)
        outputs = self._run_raw(inputs)
        actions = outputs[self.output_index["actions"]].astype(np.float32)
        if self.is_recurrent:
            h_out = outputs[self.output_index["h_out"]].astype(np.float32)
            c_out = outputs[self.output_index["c_out"]].astype(np.float32)
        else:
            h_out = None
            c_out = None
        return actions.squeeze(0), h_out, c_out


@policy_registry.register
class SoccerPolicy(Policy):
    cfg_policy: SoccerPolicyCfg
    requires_hard_switch = True
    reset_dof_on_hard_switch = True

    def __init__(self, cfg_policy: SoccerPolicyCfg, device):
        if not os.path.isfile(cfg_policy.policy_file):
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")

        logger.debug(f"Loading soccer policy '{cfg_policy.policy_name}' from {cfg_policy.policy_file}")
        self.onnx_policy = OnnxSoccerPolicy(cfg_policy.policy_file, providers=cfg_policy.providers)

        super().__init__(cfg_policy=cfg_policy, device=device)

        policy_joint_names = self.onnx_policy.metadata.joint_names
        cfg_joint_names = self.cfg_action_dof.joint_names
        if policy_joint_names != cfg_joint_names:
            raise RuntimeError(
                "SoccerPolicyCfg.action_dof must match ONNX joint_names exactly. "
                f"cfg={cfg_joint_names}, onnx={policy_joint_names}"
            )
        if self.cfg_obs_dof.joint_names != cfg_joint_names:
            raise RuntimeError("SoccerPolicyCfg.obs_dof and action_dof must use the same joint order.")

        self.default_dof_pos = self.onnx_policy.metadata.default_joint_pos.astype(np.float32)
        self.default_pos = self.default_dof_pos.copy()
        self.action_scale_array = self.onnx_policy.metadata.action_scale.astype(np.float32)
        self.observation_names = self.onnx_policy.metadata.observation_names
        self.anchor_body_index = self.onnx_policy.metadata.body_names.index(self.onnx_policy.metadata.anchor_body_name)
        self.pelvis_body_index = self.onnx_policy.metadata.body_names.index("pelvis")
        self.cfg_ball_local = np.asarray(cfg_policy.ball_local, dtype=np.float32)
        self.ball_to_goal_anchor = np.asarray(cfg_policy.ball_to_goal_anchor, dtype=np.float32)
        self.ball_local = self.cfg_ball_local.copy()
        self.goal_local = self.ball_local + self.ball_to_goal_anchor
        self.goal_anchor_heading_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.soccer_target_source = str(cfg_policy.soccer_target_source).lower()
        if self.soccer_target_source not in {"auto", "env", "ctrl", "cfg"}:
            logger.warning("Unknown soccer_target_source=%s; using auto.", self.soccer_target_source)
            self.soccer_target_source = "auto"
        self.use_env_soccer_obs = bool(cfg_policy.use_env_soccer_obs)
        self.last_target_obs_source: str | None = None
        self.wait_for_ctrl_detector = self.soccer_target_source == "ctrl"
        self.waiting_for_ctrl_detector = False
        self.action_ramp_steps = max(int(cfg_policy.action_ramp_steps), 0)
        self.motion_selection_positions = self._build_motion_selection_positions()

        self.reset()

    def _zero_recurrent_state(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.onnx_policy.is_recurrent:
            return None, None
        assert self.onnx_policy.recurrent_shape is not None
        num_layers, hidden_dim = self.onnx_policy.recurrent_shape
        return (
            np.zeros((num_layers, 1, hidden_dim), dtype=np.float32),
            np.zeros((num_layers, 1, hidden_dim), dtype=np.float32),
        )

    def _motion_length(self, motion_idx: int) -> int:
        if self.onnx_policy.uses_motion_index:
            return int(self.onnx_policy.reference["motion_lengths"][motion_idx])
        return int(self.onnx_policy.reference["time_step_total"])

    def _build_motion_selection_positions(self) -> np.ndarray:
        if self.onnx_policy.uses_motion_index:
            positions = []
            for motion_idx in range(len(self.onnx_policy.metadata.motion_names)):
                motion_len = self._motion_length(motion_idx)
                first_anchor = self.onnx_policy.reference["body_pos_w"][motion_idx, 0, self.anchor_body_index]
                final_anchor = self.onnx_policy.reference["body_pos_w"][motion_idx, motion_len - 1, self.anchor_body_index]
                first_anchor_quat = _quat_wxyz_to_xyzw(
                    self.onnx_policy.reference["body_quat_w"][motion_idx, 0, self.anchor_body_index]
                )
                positions.append(quat_rotate_inverse_np(first_anchor_quat, final_anchor - first_anchor))
            return np.stack(positions, axis=0).astype(np.float32)

        first_anchor = self.onnx_policy.reference["body_pos_w"][0, self.anchor_body_index]
        final_anchor = self.onnx_policy.reference["body_pos_w"][-1, self.anchor_body_index]
        first_anchor_quat = _quat_wxyz_to_xyzw(self.onnx_policy.reference["body_quat_w"][0, self.anchor_body_index])
        return quat_rotate_inverse_np(first_anchor_quat, final_anchor - first_anchor)[None, :].astype(np.float32)

    def select_motion_for_ball(self, ball_local: np.ndarray) -> int:
        if not self.onnx_policy.uses_motion_index:
            return 0
        distances = np.linalg.norm(self.motion_selection_positions[:, :2] - ball_local[:2], axis=1)
        return int(distances.argmin())

    def _motion_name(self, motion_idx: int) -> str:
        if self.onnx_policy.metadata.motion_names and 0 <= motion_idx < len(self.onnx_policy.metadata.motion_names):
            return self.onnx_policy.metadata.motion_names[motion_idx]
        return "single_motion"

    def _log_motion_selection(self, ball_local: np.ndarray):
        if not self.onnx_policy.uses_motion_index:
            logger.info("Soccer motion selected: single_motion ball_local=%s", np.round(ball_local, 3).tolist())
            return
        distances = np.linalg.norm(self.motion_selection_positions[:, :2] - ball_local[:2], axis=1)
        nearest = np.argsort(distances)[: min(5, len(distances))]
        nearest_info = [
            {
                "idx": int(idx),
                "name": self._motion_name(int(idx)),
                "dist": round(float(distances[idx]), 4),
                "target_xy": np.round(self.motion_selection_positions[idx, :2], 3).tolist(),
            }
            for idx in nearest
        ]
        logger.info(
            "Soccer motion selected: idx=%d name=%s length=%d ball_local=%s nearest=%s",
            self.current_motion_idx,
            self._motion_name(self.current_motion_idx),
            self.current_motion_length,
            np.round(ball_local, 3).tolist(),
            nearest_info,
        )

    def reset(self, ball_local: np.ndarray | None = None, base_quat: np.ndarray | None = None):
        self.h_state, self.c_state = self._zero_recurrent_state()
        self.timestep = 0
        self.flag_motion_done = False
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        if base_quat is not None:
            self.goal_anchor_heading_quat = calc_heading_quat_np(np.asarray(base_quat, dtype=np.float32)).astype(np.float32)
        if ball_local is not None:
            self.ball_local = np.asarray(ball_local, dtype=np.float32)
        elif not self.use_env_soccer_obs:
            self.ball_local = self.cfg_ball_local.copy()
        self.goal_local = self._compute_goal_local(self.ball_local, self.goal_anchor_heading_quat)
        self.waiting_for_ctrl_detector = self.wait_for_ctrl_detector
        self.current_motion_idx = self.select_motion_for_ball(self.ball_local)
        self.current_motion_length = self._motion_length(self.current_motion_idx)
        self._log_motion_selection(self.ball_local)

    def post_step_callback(self, commands: list[str] | None = None):
        for command in commands or []:
            if command in ("[MOTION_RESET]", "[MOTION_FADE_IN]"):
                self.reset()
            elif command == "[MOTION_FADE_OUT]":
                self.flag_motion_done = True

    def _current_reference(self) -> dict[str, np.ndarray]:
        idx = CTRL_DETECTOR_WAIT_FRAME if self.waiting_for_ctrl_detector else self.timestep
        idx = min(idx, self.current_motion_length - 1)
        if self.onnx_policy.uses_motion_index:
            return {
                "joint_pos": self.onnx_policy.reference["joint_pos"][self.current_motion_idx, idx],
                "joint_vel": self.onnx_policy.reference["joint_vel"][self.current_motion_idx, idx],
                "body_ang_vel_w": self.onnx_policy.reference["body_ang_vel_w"][self.current_motion_idx, idx],
            }
        return {
            "joint_pos": self.onnx_policy.reference["joint_pos"][idx],
            "joint_vel": self.onnx_policy.reference["joint_vel"][idx],
            "body_ang_vel_w": self.onnx_policy.reference["body_ang_vel_w"][idx],
        }

    def _motion_phase_sin_cos(self) -> np.ndarray:
        time_step = CTRL_DETECTOR_WAIT_FRAME if self.waiting_for_ctrl_detector else self.timestep
        denom = max(float(self.current_motion_length - 1), 1.0)
        phase = np.clip(float(time_step) / denom, 0.0, 1.0)
        angle = 2.0 * np.pi * phase
        return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)

    def _compute_ball_to_goal_local(self, base_quat: np.ndarray) -> np.ndarray:
        ball_to_goal_world = my_quat_rotate_np(self.goal_anchor_heading_quat, self.ball_to_goal_anchor)
        return quat_rotate_inverse_np(np.asarray(base_quat, dtype=np.float32), ball_to_goal_world).astype(np.float32)

    def _compute_goal_local(self, ball_local: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
        return (np.asarray(ball_local, dtype=np.float32) + self._compute_ball_to_goal_local(base_quat)).astype(np.float32)

    def _start_from_detector(self, ball_local: np.ndarray) -> None:
        self.h_state, self.c_state = self._zero_recurrent_state()
        self.timestep = CTRL_DETECTOR_WAIT_FRAME
        self.flag_motion_done = False
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.current_motion_idx = self.select_motion_for_ball(ball_local)
        self.current_motion_length = self._motion_length(self.current_motion_idx)
        self._log_motion_selection(ball_local)

    def _extract_target_obs(self, data, source: str) -> tuple[np.ndarray, object, str] | None:
        soccer_obs = _get_optional(data, "soccer_obs", None)
        if soccer_obs is not None and _get_optional(soccer_obs, "ball_local", None) is not None:
            ball_local = np.asarray(_get_optional(soccer_obs, "ball_local"), dtype=np.float32)
            return ball_local, soccer_obs, f"{source}.soccer_obs"

        ball = _get_optional(data, "ball_local", None)
        if ball is None:
            return None
        return np.asarray(ball, dtype=np.float32), None, f"{source}.target"

    def _resolve_soccer_observation(self, env_data, ctrl_data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        ctrl_target = self._extract_target_obs(ctrl_data, "ctrl")
        env_target = self._extract_target_obs(env_data, "env") if self.use_env_soccer_obs else None

        target = None
        if self.soccer_target_source == "ctrl":
            target = ctrl_target
        elif self.soccer_target_source == "env":
            target = env_target
        elif self.soccer_target_source == "auto":
            target = ctrl_target if ctrl_target is not None else env_target

        base_quat = env_data.base_quat
        base_ang_vel = env_data.base_ang_vel
        if target is None and self.soccer_target_source == "cfg":
            ball_local = self.cfg_ball_local.copy()
            source = "cfg"
        elif target is None and self.wait_for_ctrl_detector:
            ball_local = self.ball_local.copy()
            source = "ctrl.waiting"
        elif target is None:
            raise RuntimeError(
                "Soccer target observation is unavailable. "
                "Provide ctrl_data.soccer_obs/ball_local for real, env_data.soccer_obs for sim, "
                "or set soccer_target_source='cfg' explicitly for a fixed-target test."
            )
        else:
            ball_local, soccer_obs, source = target
            if soccer_obs is not None:
                base_quat = np.asarray(_get_optional(soccer_obs, "pelvis_quat_xyzw", base_quat), dtype=np.float32)
                base_ang_vel = np.asarray(_get_optional(soccer_obs, "base_ang_vel", base_ang_vel), dtype=np.float32)

        goal_local = self._compute_goal_local(ball_local, base_quat)
        if self.wait_for_ctrl_detector:
            soccer_obs_source = _get_optional(soccer_obs, "source", None) if target is not None else None
            was_waiting = self.waiting_for_ctrl_detector
            self.waiting_for_ctrl_detector = source != "ctrl.soccer_obs" or soccer_obs_source != "detector"
            if was_waiting and not self.waiting_for_ctrl_detector:
                self._start_from_detector(ball_local)
                goal_local = self._compute_goal_local(ball_local, base_quat)
            if self.waiting_for_ctrl_detector:
                ball_local = self.ball_local.copy()
                goal_local = self.goal_local.copy()
        if source != self.last_target_obs_source:
            logger.info(
                "Soccer target source: %s ball_local=%s goal_local=%s ball_to_goal_anchor=%s waiting_for_detector=%s",
                source,
                np.round(ball_local, 3).tolist(),
                np.round(goal_local, 3).tolist(),
                np.round(self.ball_to_goal_anchor, 3).tolist(),
                self.waiting_for_ctrl_detector,
            )
            self.last_target_obs_source = source
        return (
            np.asarray(ball_local, dtype=np.float32),
            np.asarray(goal_local, dtype=np.float32),
            np.asarray(base_quat, dtype=np.float32),
            np.asarray(base_ang_vel, dtype=np.float32),
            source,
        )

    def get_observation(self, env_data, ctrl_data):
        ball_local, goal_local, base_quat, base_ang_vel, _ = self._resolve_soccer_observation(env_data, ctrl_data)
        self.ball_local = ball_local
        self.goal_local = goal_local

        ref = self._current_reference()
        dof_pos_minus_default = env_data.dof_pos - self.default_dof_pos
        projected_gravity = quat_rotate_inverse_np(base_quat, np.array([0, 0, -1], dtype=np.float32))

        term_map: dict[str, np.ndarray] = {
            "command": np.concatenate((ref["joint_pos"], ref["joint_vel"]), axis=0).astype(np.float32),
            "projected_gravity": projected_gravity.astype(np.float32),
            "motion_ref_ang_vel": ref["body_ang_vel_w"][self.anchor_body_index].astype(np.float32),
            "base_ang_vel": base_ang_vel.astype(np.float32),
            "joint_pos": dof_pos_minus_default.astype(np.float32),
            "joint_vel": env_data.dof_vel.astype(np.float32),
            "actions": self.last_action.astype(np.float32),
            "target_point_pos": ball_local.astype(np.float32),
            "target_destination_pos_local": goal_local.astype(np.float32),
            "motion_phase": self._motion_phase_sin_cos(),
        }

        obs_terms = []
        missing = []
        for name in self.observation_names:
            term = term_map.get(name)
            if term is None:
                missing.append(name)
                continue
            obs_terms.append(term.reshape(-1))
        if missing:
            raise RuntimeError(f"Soccer observation terms are not implemented in RoboJuDo: {missing}")
        obs = np.concatenate(obs_terms, axis=0).astype(np.float32)
        if obs.shape[0] != self.onnx_policy.obs_dim:
            raise RuntimeError(f"Constructed obs dim {obs.shape[0]} does not match ONNX input dim {self.onnx_policy.obs_dim}.")

        extras = {"CALLBACK": ["[MOTION_DONE]"] if self.flag_motion_done else []}
        ball_world = _get_optional(_get_optional(env_data, "soccer_obs", None), "ball_world", None)
        if ball_world is not None:
            ball_to_goal_world = my_quat_rotate_np(self.goal_anchor_heading_quat, self.ball_to_goal_anchor)
            extras["soccer_goal_world"] = (np.asarray(ball_world, dtype=np.float32) + ball_to_goal_world).astype(np.float32)
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        actions, h_out, c_out = self.onnx_policy.act(
            obs.reshape(1, -1),
            CTRL_DETECTOR_WAIT_FRAME if self.waiting_for_ctrl_detector else self.timestep,
            self.current_motion_idx,
            self.h_state,
            self.c_state,
        )
        if self.action_clip is not None:
            actions = np.clip(actions, -self.action_clip, self.action_clip)
        actions = actions.astype(np.float32)
        if self.waiting_for_ctrl_detector:
            actions = np.zeros_like(actions, dtype=np.float32)
            self.last_action = actions.copy()
            return actions * self.action_scale_array
        if self.action_ramp_steps > 0:
            ramp_alpha = min((self.timestep + 1) / self.action_ramp_steps, 1.0)
            actions = actions * ramp_alpha
        self.h_state, self.c_state = h_out, c_out
        self.last_action = actions.copy()
        self.timestep += 1
        if self.timestep >= self.current_motion_length:
            self.flag_motion_done = True
        return actions * self.action_scale_array

    def get_init_dof_pos(self) -> np.ndarray:
        idx = CTRL_DETECTOR_WAIT_FRAME if self.waiting_for_ctrl_detector else 0
        if self.onnx_policy.uses_motion_index:
            idx = min(idx, self.current_motion_length - 1)
            return self.onnx_policy.reference["joint_pos"][self.current_motion_idx, idx].astype(np.float32).copy()
        return self.onnx_policy.reference["joint_pos"][idx].astype(np.float32).copy()

    def get_init_root_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.onnx_policy.uses_motion_index:
            root_pos = self.onnx_policy.reference["body_pos_w"][self.current_motion_idx, 0, self.pelvis_body_index]
            root_quat_wxyz = self.onnx_policy.reference["body_quat_w"][self.current_motion_idx, 0, self.pelvis_body_index]
        else:
            root_pos = self.onnx_policy.reference["body_pos_w"][0, self.pelvis_body_index]
            root_quat_wxyz = self.onnx_policy.reference["body_quat_w"][0, self.pelvis_body_index]
        return root_pos.astype(np.float32).copy(), _quat_wxyz_to_xyzw(root_quat_wxyz)
