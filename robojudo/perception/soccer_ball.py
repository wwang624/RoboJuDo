from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BallDetection:
    bbox_xyxy: np.ndarray
    confidence: float
    class_id: int
    center_uv: tuple[float, float]
    depth_m: float
    position_optical: np.ndarray
    position_depth_link: np.ndarray
    position_torso: np.ndarray | None = None


class SoccerBallDetector:
    def __init__(
        self,
        model_path: str,
        *,
        class_id: int | None = None,
        confidence_threshold: float = 0.35,
        depth_window: int = 7,
        device: str | int | None = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("SoccerBallDetector requires ultralytics.") from exc

        self.model = YOLO(model_path)
        self.class_id = class_id
        self.confidence_threshold = confidence_threshold
        self.depth_window = max(1, int(depth_window))
        self.device = device

    def detect(
        self,
        color_rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsics,
        *,
        camera_to_torso: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> BallDetection | None:
        results = self.model.predict(color_rgb, verbose=False, device=self.device)
        if not results:
            return None
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
        keep = conf >= self.confidence_threshold
        if self.class_id is not None:
            keep &= cls == self.class_id
        candidate_indices = np.flatnonzero(keep)
        if candidate_indices.size == 0:
            return None

        best = int(candidate_indices[np.argmax(conf[candidate_indices])])
        bbox = xyxy[best].astype(np.float32)
        center_u = float((bbox[0] + bbox[2]) * 0.5)
        center_v = float((bbox[1] + bbox[3]) * 0.5)
        depth = self._median_depth(depth_m, center_u, center_v)
        if depth is None:
            return None

        pos_optical = self._deproject(center_u, center_v, depth, intrinsics)
        pos_depth_link = self._optical_to_depth_link(pos_optical)
        pos_torso = None
        if camera_to_torso is not None:
            translation, rotation_matrix = camera_to_torso
            pos_torso = translation.astype(np.float32) + rotation_matrix.astype(np.float32) @ pos_depth_link

        return BallDetection(
            bbox_xyxy=bbox,
            confidence=float(conf[best]),
            class_id=int(cls[best]),
            center_uv=(center_u, center_v),
            depth_m=float(depth),
            position_optical=pos_optical.astype(np.float32),
            position_depth_link=pos_depth_link.astype(np.float32),
            position_torso=pos_torso.astype(np.float32) if pos_torso is not None else None,
        )

    def _median_depth(self, depth_m: np.ndarray, center_u: float, center_v: float) -> float | None:
        height, width = depth_m.shape
        half = self.depth_window // 2
        u = int(round(center_u))
        v = int(round(center_v))
        u0 = max(0, u - half)
        u1 = min(width, u + half + 1)
        v0 = max(0, v - half)
        v1 = min(height, v + half + 1)
        patch = depth_m[v0:v1, u0:u1]
        valid = patch[np.isfinite(patch) & (patch > 0.05)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    @staticmethod
    def _deproject(u: float, v: float, depth: float, intrinsics) -> np.ndarray:
        x = (u - intrinsics.ppx) * depth / intrinsics.fx
        y = (v - intrinsics.ppy) * depth / intrinsics.fy
        z = depth
        return np.array([x, y, z], dtype=np.float32)

    @staticmethod
    def _optical_to_depth_link(position_optical: np.ndarray) -> np.ndarray:
        return np.array([position_optical[2], -position_optical[0], -position_optical[1]], dtype=np.float32)
