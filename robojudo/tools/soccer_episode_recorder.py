from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np

msgpack_numpy.patch()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    return repr(value)


def _to_array(value: Any, dtype=np.float32) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=dtype).copy()
    except Exception:
        return None


def _optional_get(mapping: Any, key: str, default=None):
    if mapping is None:
        return default
    if hasattr(mapping, "get"):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _sha256(path: str | None) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inner_policy(policy_wrapper: Any) -> Any:
    return getattr(policy_wrapper, "policy", policy_wrapper)


def _policy_entries(policy_source: Any) -> list[Any]:
    if policy_source is None:
        return []
    manager = getattr(policy_source, "policy_manager", None)
    if manager is not None:
        return list(getattr(manager, "policies", []))
    policies = getattr(policy_source, "policies", None)
    if policies is not None:
        return list(policies)
    return [policy_source]


class SoccerEpisodeRecorder:
    """Structured recorder for real/sim soccer debugging.

    The recorder is deliberately policy-agnostic. It always stores the raw obs
    vector and control target, then adds soccer-specific fields only when the
    active policy exposes them.
    """

    def __init__(
        self,
        run_cfg: Any | None = None,
        base_path: str = "logs/soccer",
        pipeline: Any | None = None,
        policy: Any | None = None,
        env: Any | None = None,
        flush_every: int = 50,
        log_hidden_state: bool = False,
    ):
        timestamp = time.strftime("%y%m%d-%H%M%S")
        cfg_name = run_cfg.__class__.__name__ if run_cfg is not None else "run"
        self.log_path = Path(base_path) / f"soccer_{timestamp}_{cfg_name}"
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.frames_file = self.log_path / "frames.msgpack"
        self.metadata_file = self.log_path / "metadata.json"
        self.flush_every = max(int(flush_every), 1)
        self.log_hidden_state = bool(log_hidden_state)

        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._frames_written = 0
        self._thread = threading.Thread(target=self._writer_thread, daemon=True)

        self._write_metadata(run_cfg=run_cfg, pipeline=pipeline, policy=policy, env=env)
        self._thread.start()

    def _policy_metadata(self, policy: Any) -> dict[str, Any]:
        inner = _inner_policy(policy)
        onnx_policy = getattr(inner, "onnx_policy", None)
        cfg_policy = getattr(inner, "cfg_policy", None)
        policy_file = getattr(cfg_policy, "policy_file", None)
        return {
            "wrapper_name": getattr(policy, "name", None),
            "class": inner.__class__.__name__ if inner is not None else None,
            "policy_name": getattr(cfg_policy, "policy_name", None),
            "policy_file": policy_file,
            "policy_file_sha256": _sha256(policy_file),
            "freq": getattr(inner, "freq", None),
            "num_actions": getattr(inner, "num_actions", None),
            "obs_dim": getattr(onnx_policy, "obs_dim", None),
            "is_recurrent": getattr(onnx_policy, "is_recurrent", None),
            "uses_motion_index": getattr(onnx_policy, "uses_motion_index", None),
            "onnx_inputs": getattr(onnx_policy, "input_names", None),
            "onnx_outputs": getattr(onnx_policy, "output_names", None),
            "onnx_metadata": getattr(onnx_policy, "model_meta", None),
            "observation_names": getattr(inner, "observation_names", None),
            "anchor_body_index": getattr(inner, "anchor_body_index", None),
            "ball_to_goal_anchor": getattr(inner, "ball_to_goal_anchor", None),
            "soccer_target_source": getattr(inner, "soccer_target_source", None),
            "use_env_soccer_obs": getattr(inner, "use_env_soccer_obs", None),
            "policy_obs_joint_names": _optional_get(getattr(inner, "cfg_obs_dof", None), "joint_names", None),
            "policy_action_joint_names": _optional_get(getattr(inner, "cfg_action_dof", None), "joint_names", None),
            "default_pos": getattr(inner, "default_pos", None),
            "action_scale": getattr(inner, "action_scale_array", getattr(inner, "action_scale", None)),
        }

    def _write_metadata(self, run_cfg: Any | None, pipeline: Any | None, policy: Any | None, env: Any | None) -> None:
        entries = _policy_entries(pipeline if pipeline is not None else policy)
        current = policy if policy is not None else (entries[0] if entries else None)
        metadata = {
            "schema": "robojudo_soccer_episode_v1",
            "created_time": time.time(),
            "created_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_cfg_class": run_cfg.__class__.__name__ if run_cfg is not None else None,
            "robot": getattr(run_cfg, "robot", None),
            "policy": self._policy_metadata(current) if current is not None else {},
            "policies": [self._policy_metadata(entry) for entry in entries],
            "dof": {
                "env_joint_names": _optional_get(getattr(env, "dof_cfg", None), "joint_names", None),
            },
            "soccer": {
                "note": "Soccer-specific fields are stored per policy in `policies` and per frame in `soccer`.",
            },
            "run_cfg": run_cfg.to_dict() if run_cfg is not None and hasattr(run_cfg, "to_dict") else None,
        }
        with open(self.metadata_file, "w") as f:
            json.dump(_json_safe(metadata), f, indent=2)

    def log(self, *, timestep: int, env_data: Any, ctrl_data: Any, obs: Any, extras: Any, pd_target: Any, policy: Any) -> None:
        inner = _inner_policy(policy)
        onnx_policy = getattr(inner, "onnx_policy", None)
        soccer_obs = _optional_get(ctrl_data, "soccer_obs", None)
        env_soccer_obs = _optional_get(env_data, "soccer_obs", None)

        frame: dict[str, Any] = {
            "time": time.time(),
            "timestep": int(timestep),
            "runtime": {
                "policy_name": getattr(policy, "name", None),
                "policy_class": inner.__class__.__name__,
                "motion_idx": getattr(inner, "current_motion_idx", None),
                "motion_name": self._motion_name(inner),
                "motion_length": getattr(inner, "current_motion_length", None),
                "policy_timestep": getattr(inner, "timestep", None),
                "flag_motion_done": getattr(inner, "flag_motion_done", None),
                "waiting_for_ctrl_detector": getattr(inner, "waiting_for_ctrl_detector", None),
                "target_obs_source": getattr(inner, "last_target_obs_source", None),
                "rnn_h_norm": self._norm(getattr(inner, "h_state", None)),
                "rnn_c_norm": self._norm(getattr(inner, "c_state", None)),
            },
            "state": {
                "base_quat": _to_array(_optional_get(env_data, "base_quat", None)),
                "base_ang_vel": _to_array(_optional_get(env_data, "base_ang_vel", None)),
                "base_lin_vel": _to_array(_optional_get(env_data, "base_lin_vel", None)),
                "root_pos": _to_array(_optional_get(env_data, "root_pos", None)),
                "root_quat": _to_array(_optional_get(env_data, "root_quat", None)),
                "dof_pos": _to_array(_optional_get(env_data, "dof_pos", None)),
                "dof_vel": _to_array(_optional_get(env_data, "dof_vel", None)),
                "obs": _to_array(obs),
                "last_action_raw": _to_array(getattr(inner, "last_action", None)),
                "pd_target": _to_array(pd_target),
            },
            "soccer": {
                "ctrl_soccer_obs": _json_safe(soccer_obs),
                "env_soccer_obs": _json_safe(env_soccer_obs),
                "ctrl_ball_local": _to_array(_optional_get(ctrl_data, "ball_local", None)),
                "ctrl_soccer_obs_valid": _optional_get(ctrl_data, "soccer_obs_valid", None),
                "policy_ball_local": _to_array(getattr(inner, "ball_local", None)),
                "policy_goal_local": _to_array(getattr(inner, "goal_local", None)),
                "ball_to_goal_anchor": _to_array(getattr(inner, "ball_to_goal_anchor", None)),
                "soccer_goal_world": _to_array(_optional_get(extras, "soccer_goal_world", None)),
                "goal_anchor_heading_quat": _to_array(getattr(inner, "goal_anchor_heading_quat", None)),
            },
            "extras": _json_safe(extras),
        }
        if onnx_policy is not None:
            frame["runtime"]["onnx_obs_dim"] = getattr(onnx_policy, "obs_dim", None)
            frame["runtime"]["onnx_is_recurrent"] = getattr(onnx_policy, "is_recurrent", None)
            frame["runtime"]["onnx_uses_motion_index"] = getattr(onnx_policy, "uses_motion_index", None)
        if self.log_hidden_state:
            frame["runtime"]["rnn_h_state"] = _to_array(getattr(inner, "h_state", None))
            frame["runtime"]["rnn_c_state"] = _to_array(getattr(inner, "c_state", None))

        self._queue.put(frame)

    def _motion_name(self, inner: Any) -> str | None:
        motion_idx = getattr(inner, "current_motion_idx", None)
        if motion_idx is None:
            return None
        motion_name_fn = getattr(inner, "_motion_name", None)
        if callable(motion_name_fn):
            try:
                return motion_name_fn(int(motion_idx))
            except Exception:
                return None
        return None

    def _norm(self, value: Any) -> float | None:
        arr = _to_array(value)
        if arr is None:
            return None
        return float(np.linalg.norm(arr))

    def _writer_thread(self) -> None:
        with open(self.frames_file, "ab") as f:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    self._queue.task_done()
                    continue
                packed = msgpack.packb(item, use_bin_type=True)
                f.write(packed)
                self._frames_written += 1
                if self._frames_written % self.flush_every == 0:
                    f.flush()
                    os.fsync(f.fileno())
                self._queue.task_done()
            f.flush()
            os.fsync(f.fileno())

    def close(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join(timeout=2.0)
