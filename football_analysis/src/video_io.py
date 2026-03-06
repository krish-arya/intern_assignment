import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    total_frames: int


class VideoReader:

    def __init__(self, video_path: str):
        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")

        self.info = VideoInfo(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self._cap.get(cv2.CAP_PROP_FPS),
            total_frames=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def __iter__(self):
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1

    def release(self):
        self._cap.release()

    def __del__(self):
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()


class VideoWriter:

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, fps, (width, height))

        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {self.path}")

    def write(self, frame: np.ndarray):
        self._writer.write(frame)

    def release(self):
        self._writer.release()

    def __del__(self):
        if hasattr(self, "_writer"):
            self._writer.release()
