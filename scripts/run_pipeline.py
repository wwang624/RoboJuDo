from __future__ import annotations

# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import time

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="g1",
        help="Name of the config class to use",
    )
    parser.add_argument(
        "--log-soccer",
        action="store_true",
        help="Enable structured soccer episode recorder.",
    )
    parser.add_argument(
        "--soccer-log-dir",
        type=str,
        default=None,
        help="Directory for structured soccer logs. Defaults to cfg.debug.soccer_log_dir.",
    )
    parser.add_argument(
        "--soccer-log-hidden-state",
        action="store_true",
        help="Also store full recurrent hidden states. This can generate large logs.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info(f"Using config: {args.config}")
    config_manager = ConfigManager(config_name=args.config)

    cfg: RlPipelineCfg = config_manager.get_cfg()
    if args.log_soccer:
        cfg.debug.log_soccer = True
    if args.soccer_log_dir is not None:
        cfg.debug.soccer_log_dir = args.soccer_log_dir
    if args.soccer_log_hidden_state:
        cfg.debug.soccer_log_hidden_state = True

    pipeline_type = cfg.pipeline_type

    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Using pipeline: {pipeline_type} -> {pipeline_class}")

    pipeline = pipeline_class(cfg=cfg)

    if not cfg.env.is_sim:
        pipeline.prepare()
    elif getattr(pipeline, "_has_default_pose_mode", False):
        pipeline._set_default_pose_mode(True)
        logger.warning("Sim mode — holding default pose, press R to start motion")

    try:
        while True:
            time_start = time.time()
            pipeline.step()
            time_end = time.time()
            time_diff = time_end - time_start

            # keep the pipeline running at the desired frequency
            if not cfg.run_fullspeed:
                time_diff = pipeline.dt - time_diff
                if time_diff > 0:
                    time.sleep(time_diff)
                else:
                    if not cfg.env.is_sim:
                        logger.error(f"Warning: frame drop -> {time_diff}")
                        if time_diff < -0.2:
                            logger.critical("Exiting due to excessive frame drop")
                            pipeline.env.shutdown()
                            time.sleep(10)
                            break
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
