from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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

        self.resolution = resolution
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, fps)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
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

    def get_camera_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        timeout_ms = int(1000 / self.fps)
        frames = self.pipeline.wait_for_frames(timeout_ms * 2)
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None
        color_rgb = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        depth_m = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale
        return color_rgb, depth_m

    def stop(self) -> None:
        self.pipeline.stop()
