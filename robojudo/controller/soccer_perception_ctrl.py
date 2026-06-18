from __future__ import annotations

import logging
import time

import numpy as np

from robojudo.controller import ControllerHook, ctrl_registry
from robojudo.controller.ctrl_cfgs import SoccerPerceptionCtrlCfg
from robojudo.perception import SoccerPerceptionProvider, SoccerPerceptionResult

logger = logging.getLogger(__name__)


@ctrl_registry.register
class SoccerPerceptionCtrl(ControllerHook):
    cfg_ctrl: SoccerPerceptionCtrlCfg

    def __init__(self, cfg_ctrl: SoccerPerceptionCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self.provider = SoccerPerceptionProvider(
            model_path=cfg_ctrl.detector_model,
            class_id=cfg_ctrl.detector_class_id,
            confidence_threshold=cfg_ctrl.detector_confidence,
            device=cfg_ctrl.detector_device,
            resolution=(cfg_ctrl.width, cfg_ctrl.height),
            fps=cfg_ctrl.fps,
            detector_rate=cfg_ctrl.detector_rate,
            depth_window=cfg_ctrl.depth_window,
            ball_x_range=tuple(cfg_ctrl.ball_x_range),
            ball_y_range=tuple(cfg_ctrl.ball_y_range),
            ball_z_range=tuple(cfg_ctrl.ball_z_range),
            manual_ball_local=cfg_ctrl.manual_ball_local,
        )
        self.provider.start(threaded=cfg_ctrl.threaded)
        self.counter = 0
        self.last_log_time = time.time()

    def reset(self):
        self.provider.clear()

    def post_step_callback(self, commands: list[str] | None = None):
        if commands and "[SHUTDOWN]" in commands:
            self.provider.stop()
        if commands and "[POLICY_LOCO]" in commands:
            self.provider.clear()

    def close(self):
        self.provider.stop()

    def get_data_with_hook(self, prior_ctrl_data: dict, env_data: dict):
        result = None
        if self.env is not None:
            result = self.provider.update_from_env(self.env)
            if result is None:
                result = self.provider.latest_or_manual()
        else:
            result = self.provider.latest_or_manual()

        if self.cfg_ctrl.debug_window:
            self._show_debug_window()

        self.counter += 1
        if result is not None and self.counter % max(1, self.cfg_ctrl.print_interval) == 0:
            self._log_result(result)

        if result is None:
            return {"soccer_obs_valid": False}
        soccer_obs = {
            "ball_local": result.ball_local.astype(np.float32),
            "source": "detector" if result.detection is not None else "manual",
            "confidence": float(result.confidence),
            "timestamp": float(result.timestamp),
            "valid": bool(result.valid),
            "in_range": bool(result.in_range),
        }
        return {
            "soccer_obs": soccer_obs,
            "ball_local": soccer_obs["ball_local"],
            "soccer_obs_valid": bool(result.valid),
        }

    def _log_result(self, result: SoccerPerceptionResult) -> None:
        age = time.time() - result.timestamp
        logger.info(
            "Soccer perception ball_local=%s source=%s conf=%.3f valid=%s age=%.3f",
            np.round(result.ball_local, 3).tolist(),
            "detector" if result.detection is not None else "manual",
            result.confidence,
            result.valid,
            age,
        )

    def _show_debug_window(self) -> None:
        image = self.provider.latest_debug_image()
        if image is None:
            return
        try:
            import cv2
        except ImportError:
            return
        cv2.imshow("RoboJuDo Soccer Perception", image[:, :, ::-1])
        cv2.waitKey(1)
