import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path

from config import HeatmapConfig


class HeatmapGenerator:

    def __init__(self, config: HeatmapConfig, frame_width: int, frame_height: int):
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._positions: dict[int, list[tuple[float, float]]] = defaultdict(list)

    def update(self, track_id: int, center: tuple[float, float]):
        self._positions[track_id].append(center)

    def generate(self, track_id: int) -> np.ndarray:
        positions = self._positions.get(track_id, [])
        grid_w, grid_h = self.config.resolution

        grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        scale_x = grid_w / self.frame_width
        scale_y = grid_h / self.frame_height

        for cx, cy in positions:
            gx = int(cx * scale_x)
            gy = int(cy * scale_y)
            gx = np.clip(gx, 0, grid_w - 1)
            gy = np.clip(gy, 0, grid_h - 1)
            grid[gy, gx] += 1

        ksize = int(self.config.sigma * 6) | 1
        grid = cv2.GaussianBlur(grid, (ksize, ksize), self.config.sigma)

        if grid.max() > 0:
            grid = (grid / grid.max() * 255).astype(np.uint8)
        else:
            grid = grid.astype(np.uint8)

        heatmap = cv2.applyColorMap(grid, self.config.colormap)
        heatmap = cv2.resize(heatmap, (self.frame_width, self.frame_height))

        return heatmap

    def save_all(self, output_dir: str):
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for track_id in sorted(self._positions.keys()):
            heatmap = self.generate(track_id)
            filename = out_path / f"heatmap_player_{track_id}.png"
            cv2.imwrite(str(filename), heatmap)

        print(f"  Saved {len(self._positions)} heatmaps to {out_path}")

    @property
    def tracked_ids(self) -> list[int]:
        return list(self._positions.keys())
