from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DetectorConfig:
    model_name: str = "yolov8s.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    person_class_id: int = 0
    device: str = "auto"
    imgsz: int = 1280
    min_bbox_height: int = 40
    min_aspect_ratio: float = 1.2
    max_aspect_ratio: float = 5.0
    pitch_roi_top: float = 0.1
    pitch_roi_bottom: float = 0.95


@dataclass
class TrackerConfig:
    tracker_type: str = "botsort"
    track_high_thresh: float = 0.3
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.4
    track_buffer: int = 60
    match_thresh: float = 0.8


@dataclass
class SpeedConfig:
    meters_per_pixel: float = 0.05
    smoothing_window: int = 10
    max_realistic_speed: float = 12.0


@dataclass
class HeatmapConfig:
    resolution: tuple = (120, 80)
    sigma: float = 10.0
    colormap: int = 2


@dataclass
class PipelineConfig:
    input_video: str = ""
    output_dir: str = str(Path("output"))
    output_video_name: str = "annotated_output.mp4"
    stats_file_name: str = "player_stats.csv"
    heatmap_dir_name: str = "heatmaps"

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)

    max_frames: int = 0
    display_live: bool = False
    min_track_frames: int = 60
