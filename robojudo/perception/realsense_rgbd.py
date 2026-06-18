from __future__ import annotations

from dataclasses import dataclass
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    width: int = 0
    height: int = 0
    fx: float = 0.0
    fy: float = 0.0
    ppx: float = 0.0
    ppy: float = 0.0


class RealSenseRgbdCamera:
    def __init__(self, resolution: tuple[int, int], fps: int):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError("RealSenseRgbdCamera requires pyrealsense2.") from exc

        self.rs = rs
        self.resolution = resolution
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, fps)
        self.profile = None
        self.align = None
        self.depth_scale = 0.0
        self.color_intrinsics = CameraIntrinsics()
        self._started = False
        self._last_start_attempt = 0.0
        self._start_retry_interval = 1.0

        self.ensure_started()

    def start(self) -> None:
        if self._started:
            return
        self._last_start_attempt = time.time()
        self.profile = self.pipeline.start(self.config)
        self._started = True
        self.align = self.rs.align(self.rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        color_profile = self.profile.get_stream(self.rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.color_intrinsics = CameraIntrinsics(
            width=int(intr.width),
            height=int(intr.height),
            fx=float(intr.fx),
            fy=float(intr.fy),
            ppx=float(intr.ppx),
            ppy=float(intr.ppy),
        )
        _ = self.pipeline.wait_for_frames(1000)

    def ensure_started(self) -> bool:
        if self._started:
            return True
        now = time.time()
        if now - self._last_start_attempt < self._start_retry_interval:
            return False
        try:
            self.start()
            return True
        except Exception as exc:
            logger.warning("RealSense start failed: %s", exc)
            self._started = False
            return False

    def get_camera_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.ensure_started():
            return None
        timeout_ms = int(1000 / self.fps)
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms * 2)
        except Exception as exc:
            logger.warning("RealSense wait_for_frames failed: %s", exc)
            self._mark_disconnected()
            return None
        if self.align is None:
            return None
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None
        color_rgb = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        depth_m = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale
        return color_rgb, depth_m

    def _mark_disconnected(self) -> None:
        was_started = self._started
        self._started = False
        self.align = None
        self.profile = None
        if was_started:
            try:
                self.pipeline.stop()
            except Exception as exc:
                logger.debug("RealSense stop after failure ignored: %s", exc)

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self.pipeline.stop()
        finally:
            self._started = False
