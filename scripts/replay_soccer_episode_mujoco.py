#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np

from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg
from robojudo.environment.mujoco_env import MujocoEnv
from robojudo.utils.util_func import my_quat_rotate_np, quatToEuler

msgpack_numpy.patch()


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


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=np.float32)
    except Exception:
        return None


def iter_frames(log_dir: Path):
    frames_path = log_dir / "frames.msgpack"
    with open(frames_path, "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        yield from unpacker


def local_to_world(root_pos: np.ndarray, root_quat_xyzw: np.ndarray, local_pos: np.ndarray) -> np.ndarray:
    return root_pos + my_quat_rotate_np(root_quat_xyzw, local_pos)


def yaw_only_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    yaw = float(quatToEuler(quat_xyzw)[2])
    half = yaw * 0.5
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def resolve_robot_state(
    frame: dict[str, Any],
    fallback_root_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
    dof_pos = _array(_get(frame, "state.dof_pos"))
    dof_vel = _array(_get(frame, "state.dof_vel"))
    if dof_pos is None:
        return None

    root_pos = _array(_get(frame, "state.root_pos"))
    root_quat = _array(_get(frame, "state.root_quat"))
    base_quat = _array(_get(frame, "state.base_quat"))

    if root_pos is None:
        root_pos = fallback_root_pos.copy()
    if root_quat is None:
        root_quat = base_quat
    if root_quat is None:
        root_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return root_pos, root_quat, dof_pos, dof_vel


def resolve_soccer_world(frame: dict[str, Any], root_pos: np.ndarray, root_quat: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    ball_w = _array(_get(frame, "soccer.env_soccer_obs.ball_world"))
    goal_w = _array(_get(frame, "soccer.soccer_goal_world"))

    if ball_w is None:
        ball_local = _array(_get(frame, "soccer.policy_ball_local"))
        if ball_local is None:
            ball_local = _array(_get(frame, "soccer.ctrl_ball_local"))
        if ball_local is not None:
            ball_w = local_to_world(root_pos, root_quat, ball_local)

    if goal_w is None:
        goal_local = _array(_get(frame, "soccer.policy_goal_local"))
        if goal_local is not None:
            goal_w = local_to_world(root_pos, root_quat, goal_local)

    return ball_w, goal_w


def project_targets_to_ground(
    ball_w: np.ndarray | None,
    goal_w: np.ndarray | None,
    ball_height: float,
    goal_height: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if ball_w is not None:
        ball_w = ball_w.copy()
        ball_w[2] = ball_height
    if goal_w is not None:
        goal_w = goal_w.copy()
        goal_w[2] = goal_height
    return ball_w, goal_w


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a RoboJuDo soccer episode log in MuJoCo.")
    parser.add_argument("log_dir", type=Path, help="Directory containing metadata.json and frames.msgpack")
    parser.add_argument("--only-soccer", action="store_true", help="Replay only frames where the active policy is SoccerPolicy.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--print-interval", type=int, default=50)
    parser.add_argument(
        "--root-mode",
        choices=["fixed", "integrate"],
        default="integrate",
        help="`fixed` replays posture at a fixed root. `integrate` estimates root xy from logged base_lin_vel.",
    )
    parser.add_argument("--root-height", type=float, default=0.78)
    parser.add_argument("--ball-height", type=float, default=0.11)
    parser.add_argument("--goal-height", type=float, default=0.11)
    parser.add_argument(
        "--raw-target-z",
        action="store_true",
        help="Keep logged target z instead of projecting ball/goal marker to ground height.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["first", "dynamic", "manual"],
        default="first",
        help="How to place soccer ball/goal. `first` locks the first replayed frame target, "
        "`dynamic` updates from each frame, `manual` uses --ball-world/--goal-world.",
    )
    parser.add_argument("--ball-world", type=float, nargs=3, default=None)
    parser.add_argument("--goal-world", type=float, nargs=3, default=None)
    args = parser.parse_args()

    metadata_path = args.log_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    robot = metadata.get("robot")
    if robot != "g1":
        raise RuntimeError(f"Only G1 replay is implemented now, got robot={robot!r}.")

    env_cfg = G1MujocoEnvCfg(
        forward_kinematic=None,
        update_with_fk=False,
        born_place_align=False,
        soccer_objects_enabled=True,
        soccer_objects_from_local_targets=False,
    )
    env = MujocoEnv(cfg_env=env_cfg)
    frame_dt = 1.0 / max(args.fps * args.speed, 1e-6)
    fixed_ball_w = None if args.ball_world is None else np.asarray(args.ball_world, dtype=np.float32)
    fixed_goal_w = None if args.goal_world is None else np.asarray(args.goal_world, dtype=np.float32)
    if fixed_ball_w is not None and not args.raw_target_z:
        fixed_ball_w[2] = args.ball_height
    if fixed_goal_w is not None and not args.raw_target_z:
        fixed_goal_w[2] = args.goal_height
    fixed_target_initialized = args.target_mode == "manual"
    if args.target_mode == "manual" and fixed_ball_w is None:
        raise ValueError("--target-mode manual requires --ball-world")
    replay_root_pos = np.array([0.0, 0.0, args.root_height], dtype=np.float32)
    prev_frame_time = None

    try:
        while True:
            shown = 0
            for idx, frame in enumerate(iter_frames(args.log_dir)):
                if idx < args.start_frame:
                    continue
                policy_name = str(_get(frame, "runtime.policy_name"))
                if args.only_soccer and "SoccerPolicy" not in policy_name:
                    continue

                base_quat = _array(_get(frame, "state.base_quat"))
                base_lin_vel = _array(_get(frame, "state.base_lin_vel"))
                frame_time = _get(frame, "time")
                if args.root_mode == "integrate" and base_quat is not None and base_lin_vel is not None:
                    if prev_frame_time is not None and frame_time is not None:
                        dt = float(frame_time) - float(prev_frame_time)
                        if 0.0 < dt < 0.2:
                            replay_root_pos[:2] += my_quat_rotate_np(yaw_only_quat(base_quat), base_lin_vel)[:2] * dt
                    if frame_time is not None:
                        prev_frame_time = frame_time
                elif frame_time is not None:
                    prev_frame_time = frame_time

                state = resolve_robot_state(frame, replay_root_pos)
                if state is None:
                    continue
                root_pos, root_quat, dof_pos, dof_vel = state
                root_pos[2] = args.root_height
                env.data.qpos[0:3] = root_pos
                env.data.qpos[3:7] = root_quat[[3, 0, 1, 2]]
                env.set_dof_state(dof_pos, dof_vel)

                if args.target_mode == "dynamic":
                    ball_w, goal_w = resolve_soccer_world(frame, root_pos, root_quat)
                elif args.target_mode == "first":
                    if not fixed_target_initialized:
                        fixed_ball_w, fixed_goal_w = resolve_soccer_world(frame, root_pos, root_quat)
                        if not args.raw_target_z:
                            fixed_ball_w, fixed_goal_w = project_targets_to_ground(
                                fixed_ball_w,
                                fixed_goal_w,
                                args.ball_height,
                                args.goal_height,
                            )
                        fixed_target_initialized = True
                        print(
                            "Locked first soccer target: "
                            f"ball_w={None if fixed_ball_w is None else np.round(fixed_ball_w, 4).tolist()} "
                            f"goal_w={None if fixed_goal_w is None else np.round(fixed_goal_w, 4).tolist()}"
                        )
                    ball_w, goal_w = fixed_ball_w, fixed_goal_w
                else:
                    ball_w, goal_w = fixed_ball_w, fixed_goal_w
                if args.target_mode == "dynamic" and not args.raw_target_z:
                    ball_w, goal_w = project_targets_to_ground(ball_w, goal_w, args.ball_height, args.goal_height)
                if ball_w is not None:
                    env.set_soccer_ball_world(ball_w)
                if goal_w is not None:
                    env.set_soccer_goal_marker_world(goal_w)

                env.viewer.cam.lookat = env.data.qpos.astype(np.float32)[:3]
                if env.viewer.is_alive:
                    env.viewer.render()
                else:
                    return

                if shown % max(1, args.print_interval) == 0:
                    print(
                        f"frame={idx} policy={policy_name} "
                        f"source={_get(frame, 'soccer.ctrl_soccer_obs.source')} "
                        f"motion={_get(frame, 'runtime.motion_name')} "
                        f"ball_w={None if ball_w is None else np.round(ball_w, 3).tolist()}"
                    )
                shown += 1
                if args.max_frames is not None and shown >= args.max_frames:
                    return
                time.sleep(frame_dt)
            if not args.loop:
                break
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
