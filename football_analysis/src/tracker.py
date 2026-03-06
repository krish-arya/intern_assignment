import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO

from config import DetectorConfig, TrackerConfig


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    confidence: float
    center: tuple[float, float]


class PlayerTracker:

    def __init__(self, detector_cfg: DetectorConfig, tracker_cfg: TrackerConfig):
        self.detector_cfg = detector_cfg
        self.tracker_cfg = tracker_cfg
        self.model = YOLO(detector_cfg.model_name)

    def track_frame(self, frame: np.ndarray, persist: bool = True) -> list[Track]:
        results = self.model.track(
            frame,
            conf=self.detector_cfg.confidence_threshold,
            iou=self.detector_cfg.iou_threshold,
            imgsz=self.detector_cfg.imgsz,
            classes=[self.detector_cfg.person_class_id],
            tracker=f"{self.tracker_cfg.tracker_type}.yaml",
            persist=persist,
            verbose=False,
        )

        cfg = self.detector_cfg
        frame_h = frame.shape[0]
        roi_top = cfg.pitch_roi_top * frame_h
        roi_bot = cfg.pitch_roi_bottom * frame_h

        tracks = []
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
            for box in boxes:
                if box.id is None:
                    continue
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = bbox
                box_w = x2 - x1
                box_h = y2 - y1

                if box_h < cfg.min_bbox_height:
                    continue

                aspect = box_h / box_w if box_w > 0 else 0
                if aspect < cfg.min_aspect_ratio or aspect > cfg.max_aspect_ratio:
                    continue

                cy = float((y1 + y2) / 2)
                if cy < roi_top or cy > roi_bot:
                    continue

                track_id = int(box.id[0].cpu())
                conf = float(box.conf[0].cpu())
                cx = float((x1 + x2) / 2)
                tracks.append(Track(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=conf,
                    center=(cx, cy),
                ))

        return tracks
