import numpy as np
from collections import defaultdict

from config import SpeedConfig


class SpeedEstimator:

    def __init__(self, config: SpeedConfig, fps: float):
        self.config = config
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
        self._positions: dict[int, list[tuple[int, float, float]]] = defaultdict(list)

    def update(self, track_id: int, frame_idx: int, center: tuple[float, float]):
        self._positions[track_id].append((frame_idx, center[0], center[1]))

    def get_instantaneous_speed(self, track_id: int) -> float:
        positions = self._positions.get(track_id, [])
        if len(positions) < 2:
            return 0.0

        window = positions[-self.config.smoothing_window:]
        if len(window) < 2:
            return 0.0

        speeds = []
        for i in range(1, len(window)):
            f_prev, x_prev, y_prev = window[i - 1]
            f_curr, x_curr, y_curr = window[i]

            pixel_dist = np.sqrt((x_curr - x_prev) ** 2 + (y_curr - y_prev) ** 2)
            frame_gap = f_curr - f_prev
            if frame_gap <= 0:
                continue

            time_elapsed = frame_gap * self.dt
            speed_m_s = (pixel_dist * self.config.meters_per_pixel) / time_elapsed
            if speed_m_s <= self.config.max_realistic_speed:
                speeds.append(speed_m_s)

        return float(np.mean(speeds)) if speeds else 0.0

    def get_average_speed(self, track_id: int) -> float:
        positions = self._positions.get(track_id, [])
        if len(positions) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(positions)):
            f_prev, x_prev, y_prev = positions[i - 1]
            f_curr, x_curr, y_curr = positions[i]
            seg_dist_px = np.sqrt((x_curr - x_prev) ** 2 + (y_curr - y_prev) ** 2)
            seg_dist_m = seg_dist_px * self.config.meters_per_pixel
            frame_gap = f_curr - f_prev
            if frame_gap <= 0:
                continue
            seg_time = frame_gap * self.dt
            seg_speed = seg_dist_m / seg_time
            if seg_speed > self.config.max_realistic_speed:
                continue
            total_distance += seg_dist_m

        first_frame = positions[0][0]
        last_frame = positions[-1][0]
        total_time = (last_frame - first_frame) * self.dt

        if total_time <= 0:
            return 0.0

        return total_distance / total_time

    def get_all_average_speeds(self) -> dict[int, float]:
        return {tid: self.get_average_speed(tid) for tid in self._positions}

    @property
    def tracked_ids(self) -> list[int]:
        return list(self._positions.keys())
