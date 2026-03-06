import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO

from config import DetectorConfig


@dataclass
class Detection:
    bbox: np.ndarray
    confidence: float
    class_id: int


class PlayerDetector:

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model = YOLO(config.model_name)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=[self.config.person_class_id],
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                cls_id = int(box.cls[0].cpu())
                detections.append(Detection(bbox=bbox, confidence=conf, class_id=cls_id))

        return detections
