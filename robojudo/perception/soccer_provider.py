from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from robojudo.perception.frames import default_g1_realsense_depth_link_transform, g1_torso_point_to_pelvis
from robojudo.perception.realsense_rgbd import RealSenseRgbdCamera
from robojudo.perception.soccer_ball import BallDetection, SoccerBallDetector

logger = logging.getLogger(__name__)


@dataclass
class SoccerPerceptionResult:
    ball_local: np.ndarray
    confidence: float
    timestamp: float
    valid: bool
    in_range: bool
    detection: BallDetection | None
    ball_torso: np.ndarray | None = None


class SoccerPerceptionProvider:
    def __init__(
        self,
        *,
        model_path: str,
        class_id: int | None = None,
        confidence_threshold: float = 0.35,
        device: str | int | None = None,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
        detector_rate: float = 30.0,
        depth_window: int = 7,
        ball_x_range: tuple[float, float] = (0.15, 3.0),
        ball_y_range: tuple[float, float] = (-1.5, 1.5),
        ball_z_range: tuple[float, float] = (-0.9, 0.3),
        manual_ball_local: list[float] | None = None,
        camera_to_torso: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.manual_ball_local = None if manual_ball_local is None else np.asarray(manual_ball_local, dtype=np.float32)
        self.detector_rate = float(detector_rate)
        self.ball_x_range = ball_x_range
        self.ball_y_range = ball_y_range
        self.ball_z_range = ball_z_range
        self.camera_to_torso = camera_to_torso or default_g1_realsense_depth_link_transform()

        self.camera = RealSenseRgbdCamera(resolution=resolution, fps=fps)
        self.detector = SoccerBallDetector(
            model_path,
            class_id=class_id,
            confidence_threshold=confidence_threshold,
            depth_window=depth_window,
            device=device,
        )
        self._lock = threading.Lock()
        self._joint_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: SoccerPerceptionResult | None = None
        self._latest_color_rgb: np.ndarray | None = None
        self._joint_pos: np.ndarray | None = None
        self._joint_names: list[str] | None = None

    def start(self, *, threaded: bool = True) -> None:
        if not threaded:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="soccer_perception", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.camera.stop()

    def _loop(self) -> None:
        period = 1.0 / max(self.detector_rate, 1e-6)
        while not self._stop.is_set():
            start = time.time()
            try:
                with self._joint_lock:
                    joint_pos = None if self._joint_pos is None else self._joint_pos.copy()
                    joint_names = None if self._joint_names is None else list(self._joint_names)
                self.step(joint_pos=joint_pos, joint_names=joint_names)
            except Exception as exc:
                logger.warning("Soccer perception step failed: %s", exc)
            sleep_time = period - (time.time() - start)
            if sleep_time > 0:
                self._stop.wait(sleep_time)

    def step(self, joint_pos: np.ndarray | None = None, joint_names: list[str] | None = None) -> SoccerPerceptionResult | None:
        camera_data = self.camera.get_camera_data()
        if camera_data is None:
            return self.latest()
        color_rgb, depth_m = camera_data
        detection = self.detector.detect(
            color_rgb,
            depth_m,
            self.camera.color_intrinsics,
            camera_to_torso=self.camera_to_torso,
        )
        self._latest_color_rgb = color_rgb.copy()
        if detection is None or detection.position_torso is None:
            return self.latest()
        if joint_pos is None or joint_names is None:
            return self.latest()
        ball_pelvis = g1_torso_point_to_pelvis(detection.position_torso, joint_pos, joint_names)
        result = self._make_result(ball_pelvis, detection, detection.position_torso)
        with self._lock:
            if result.valid:
                self._latest = result
        return result

    def update_from_env(self, env) -> SoccerPerceptionResult | None:
        joint_pos = env.dof_pos
        joint_names = env.dof_cfg.joint_names
        with self._joint_lock:
            self._joint_pos = joint_pos.copy()
            self._joint_names = list(joint_names)
        if self._thread is not None and self._thread.is_alive():
            return self.latest()
        return self.step(joint_pos=joint_pos, joint_names=joint_names)

    def latest(self) -> SoccerPerceptionResult | None:
        with self._lock:
            return self._latest

    def clear(self) -> None:
        with self._lock:
            self._latest = None

    def latest_or_manual(self) -> SoccerPerceptionResult | None:
        latest = self.latest()
        if latest is not None and latest.valid:
            return latest
        if self.manual_ball_local is None:
            return None
        return SoccerPerceptionResult(
            ball_local=self.manual_ball_local.copy(),
            confidence=0.0,
            timestamp=time.time(),
            valid=True,
            in_range=self._in_range(self.manual_ball_local),
            detection=None,
            ball_torso=None,
        )

    def latest_debug_image(self) -> np.ndarray | None:
        if self._latest_color_rgb is None:
            return None
        image = self._latest_color_rgb.copy()
        latest = self.latest()
        if latest is None or latest.detection is None:
            return image
        try:
            import cv2
        except ImportError:
            return image
        det = latest.detection
        x0, y0, x1, y1 = det.bbox_xyxy.astype(int).tolist()
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.circle(image, tuple(int(v) for v in det.center_uv), 3, (255, 0, 0), -1)
        text = f"pelvis={np.round(latest.ball_local, 2).tolist()} conf={latest.confidence:.2f}"
        cv2.putText(image, text, (max(0, x0), max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return image

    def _make_result(self, ball_local: np.ndarray, detection: BallDetection, ball_torso: np.ndarray) -> SoccerPerceptionResult:
        in_range = self._in_range(ball_local)
        return SoccerPerceptionResult(
            ball_local=ball_local.astype(np.float32),
            confidence=float(detection.confidence),
            timestamp=time.time(),
            valid=bool(in_range),
            in_range=bool(in_range),
            detection=detection,
            ball_torso=ball_torso.astype(np.float32),
        )

    def _in_range(self, ball_local: np.ndarray) -> bool:
        ball = np.asarray(ball_local, dtype=np.float32)
        return bool(
            np.all(np.isfinite(ball))
            and self.ball_x_range[0] <= ball[0] <= self.ball_x_range[1]
            and self.ball_y_range[0] <= ball[1] <= self.ball_y_range[1]
            and self.ball_z_range[0] <= ball[2] <= self.ball_z_range[1]
        )
