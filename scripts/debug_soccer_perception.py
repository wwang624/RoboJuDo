#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time

import numpy as np

from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg
from robojudo.environment.unitree_cpp_env import UnitreeCppEnv
from robojudo.perception import SoccerPerceptionProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run RoboJuDo soccer ball perception without sending motor commands.")
    parser.add_argument("--detector-model", required=True, help="YOLO .pt/.engine detector model.")
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--device", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--depth-window", type=int, default=7)
    parser.add_argument("--manual-ball-local", type=float, nargs=3, default=None)
    parser.add_argument("--ball-x-range", type=float, nargs=2, default=[0.15, 3.0])
    parser.add_argument("--ball-y-range", type=float, nargs=2, default=[-1.5, 1.5])
    parser.add_argument("--ball-z-range", type=float, nargs=2, default=[-0.9, 0.3])
    parser.add_argument("--net-if", default="eth0")
    parser.add_argument("--display", action="store_true", help="Show OpenCV detection overlay.")
    parser.add_argument("--print-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = UnitreeCppEnv(
        cfg_env=G1RealEnvCfg(
            act=False,
            unitree=G1UnitreeCfg(net_if=args.net_if),
        )
    )
    provider = SoccerPerceptionProvider(
        model_path=args.detector_model,
        class_id=args.class_id,
        confidence_threshold=args.confidence,
        device=args.device,
        resolution=(args.width, args.height),
        fps=args.fps,
        detector_rate=args.rate,
        depth_window=args.depth_window,
        ball_x_range=tuple(args.ball_x_range),
        ball_y_range=tuple(args.ball_y_range),
        ball_z_range=tuple(args.ball_z_range),
        manual_ball_local=args.manual_ball_local,
    )
    period = 1.0 / max(args.rate, 1e-6)
    counter = 0
    try:
        while True:
            start = time.time()
            env.update()
            result = provider.update_from_env(env)
            if result is None:
                result = provider.latest_or_manual()
            if counter % max(1, args.print_interval) == 0:
                if result is None:
                    print("soccer perception: no detection and no manual ball")
                else:
                    age = time.time() - result.timestamp
                    torso = None if result.ball_torso is None else np.round(result.ball_torso, 3).tolist()
                    print(
                        "soccer perception | "
                        f"ball_torso={torso} "
                        f"ball_pelvis={np.round(result.ball_local, 3).tolist()} "
                        f"conf={result.confidence:.3f} valid={result.valid} age={age:.3f}"
                    )
            if args.display:
                image = provider.latest_debug_image()
                if image is not None:
                    import cv2

                    cv2.imshow("RoboJuDo Soccer Perception", image[:, :, ::-1])
                    key = cv2.waitKey(1)
                    if key in (27, ord("q")):
                        break
            counter += 1
            sleep_time = period - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        provider.stop()
        env.shutdown()


if __name__ == "__main__":
    main()
