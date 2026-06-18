from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np

msgpack_numpy.patch()


def _as_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=np.float32)
    except Exception:
        return None


def _get(mapping: Any, path: str, default=None):
    cur = mapping
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


def _norm(value: Any) -> float | None:
    arr = _as_array(value)
    if arr is None:
        return None
    return float(np.linalg.norm(arr))


def iter_frames(path: Path):
    unpacker = msgpack.Unpacker(raw=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            unpacker.feed(chunk)
            yield from unpacker


def analyze(log_dir: Path, out_dir: Path) -> dict[str, Any]:
    metadata_path = log_dir / "metadata.json"
    frames_path = log_dir / "frames.msgpack"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not frames_path.is_file():
        raise FileNotFoundError(frames_path)

    with open(metadata_path) as f:
        metadata = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ball_jumps = []
    pd_jumps = []
    last_ball = None
    last_pd = None
    frame_count = 0
    valid_count = 0
    motion_counts: dict[str, int] = {}
    target_sources: dict[str, int] = {}

    for frame in iter_frames(frames_path):
        frame_count += 1
        ball = _as_array(_get(frame, "soccer.policy_ball_local"))
        goal = _as_array(_get(frame, "soccer.policy_goal_local"))
        ctrl_ball = _as_array(_get(frame, "soccer.ctrl_ball_local"))
        pd_target = _as_array(_get(frame, "state.pd_target"))
        dof_pos = _as_array(_get(frame, "state.dof_pos"))
        last_action = _as_array(_get(frame, "state.last_action_raw"))
        source = _get(frame, "runtime.target_obs_source")
        motion_name = _get(frame, "runtime.motion_name")
        valid = _get(frame, "soccer.ctrl_soccer_obs_valid")
        conf = _get(frame, "soccer.ctrl_soccer_obs.confidence")
        obs = _as_array(_get(frame, "state.obs"))

        if valid:
            valid_count += 1
        if source is not None:
            target_sources[str(source)] = target_sources.get(str(source), 0) + 1
        if motion_name is not None:
            motion_counts[str(motion_name)] = motion_counts.get(str(motion_name), 0) + 1

        ball_jump = None
        if ball is not None and last_ball is not None:
            ball_jump = float(np.linalg.norm(ball - last_ball))
            ball_jumps.append(ball_jump)
        if ball is not None:
            last_ball = ball.copy()

        pd_jump = None
        if pd_target is not None and last_pd is not None:
            pd_jump = float(np.linalg.norm(pd_target - last_pd))
            pd_jumps.append(pd_jump)
        if pd_target is not None:
            last_pd = pd_target.copy()

        tracking_error = None
        if pd_target is not None and dof_pos is not None and pd_target.shape == dof_pos.shape:
            tracking_error = float(np.linalg.norm(pd_target - dof_pos))

        rows.append(
            {
                "frame": frame_count - 1,
                "time": _get(frame, "time"),
                "pipeline_timestep": _get(frame, "timestep"),
                "policy_timestep": _get(frame, "runtime.policy_timestep"),
                "motion_idx": _get(frame, "runtime.motion_idx"),
                "motion_name": motion_name,
                "motion_length": _get(frame, "runtime.motion_length"),
                "target_source": source,
                "valid": valid,
                "confidence": conf,
                "obs_norm": _norm(obs),
                "action_norm": _norm(last_action),
                "pd_norm": _norm(pd_target),
                "pd_jump": pd_jump,
                "tracking_error_norm": tracking_error,
                "ball_x": None if ball is None else float(ball[0]),
                "ball_y": None if ball is None else float(ball[1]),
                "ball_z": None if ball is None else float(ball[2]),
                "goal_x": None if goal is None else float(goal[0]),
                "goal_y": None if goal is None else float(goal[1]),
                "goal_z": None if goal is None else float(goal[2]),
                "ctrl_ball_x": None if ctrl_ball is None else float(ctrl_ball[0]),
                "ctrl_ball_y": None if ctrl_ball is None else float(ctrl_ball[1]),
                "ctrl_ball_z": None if ctrl_ball is None else float(ctrl_ball[2]),
                "ball_jump": ball_jump,
                "rnn_h_norm": _get(frame, "runtime.rnn_h_norm"),
                "rnn_c_norm": _get(frame, "runtime.rnn_c_norm"),
            }
        )

    csv_path = out_dir / "frame_summary.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "log_dir": log_dir.as_posix(),
        "frames": frame_count,
        "robot": metadata.get("robot"),
        "policy": metadata.get("policy", {}),
        "valid_ratio": None if frame_count == 0 else valid_count / frame_count,
        "target_sources": target_sources,
        "motion_counts": motion_counts,
        "ball_jump": _stats(ball_jumps),
        "pd_jump": _stats(pd_jumps),
        "csv": csv_path.as_posix() if rows else None,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RoboJuDo structured soccer episode logs.")
    parser.add_argument("log_dir", type=Path, help="Directory containing metadata.json and frames.msgpack")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.log_dir / "analysis")
    summary = analyze(args.log_dir, out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
