import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

from config import PipelineConfig
from src.video_io import VideoReader, VideoWriter
from src.tracker import PlayerTracker
from src.speed_estimator import SpeedEstimator
from src.heatmap import HeatmapGenerator
from src.annotator import annotate_frame


def build_config_from_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Football Video Analysis – Detection & Tracking Pipeline"
    )
    parser.add_argument("input_video", type=str, help="Path to input football video")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory for outputs (default: output)")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        help="YOLO model name/path (default: yolov8s.pt)")
    parser.add_argument("--tracker", type=str, default="botsort",
                        choices=["botsort", "bytetrack"],
                        help="Tracker type (default: botsort)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--meters-per-pixel", type=float, default=0.05,
                        help="Calibration factor for speed estimation (default: 0.05)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames to process, 0 = all (default: 0)")
    parser.add_argument("--display", action="store_true",
                        help="Show live preview during processing")

    args = parser.parse_args()

    config = PipelineConfig()
    config.input_video = args.input_video
    config.output_dir = args.output_dir
    config.detector.model_name = args.model
    config.detector.confidence_threshold = args.conf
    config.tracker.tracker_type = args.tracker
    config.speed.meters_per_pixel = args.meters_per_pixel
    config.max_frames = args.max_frames
    config.display_live = args.display

    return config


def run_pipeline(config: PipelineConfig):
    print("=" * 60)
    print("  Football Video Analysis Pipeline")
    print("=" * 60)

    reader = VideoReader(config.input_video)
    info = reader.info
    print(f"\n  Input  : {config.input_video}")
    print(f"  Size   : {info.width}x{info.height} @ {info.fps:.1f} FPS")
    print(f"  Frames : {info.total_frames}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = str(output_dir / config.output_video_name)
    writer = VideoWriter(output_video_path, info.fps, info.width, info.height)

    tracker = PlayerTracker(config.detector, config.tracker)
    speed_est = SpeedEstimator(config.speed, info.fps)
    heatmap_gen = HeatmapGenerator(config.heatmap, info.width, info.height)

    total = config.max_frames if config.max_frames > 0 else info.total_frames
    start_time = time.time()

    print(f"\n  Processing frames...")
    for frame_idx, frame in reader:
        if config.max_frames > 0 and frame_idx >= config.max_frames:
            break

        tracks = tracker.track_frame(frame)

        inst_speeds = {}
        for t in tracks:
            speed_est.update(t.track_id, frame_idx, t.center)
            heatmap_gen.update(t.track_id, t.center)
            inst_speeds[t.track_id] = speed_est.get_instantaneous_speed(t.track_id)

        annotated = annotate_frame(frame, tracks, inst_speeds)
        writer.write(annotated)

        if config.display_live:
            cv2.imshow("Football Analysis", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  Interrupted by user.")
                break

        if (frame_idx + 1) % 100 == 0 or frame_idx == total - 1:
            elapsed = time.time() - start_time
            fps_proc = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"    Frame {frame_idx + 1}/{total}  "
                  f"({fps_proc:.1f} FPS processing)")

    reader.release()
    writer.release()
    if config.display_live:
        cv2.destroyAllWindows()

    print(f"\n  Output video saved: {output_video_path}")

    all_avg_speeds = speed_est.get_all_average_speeds()
    min_f = config.min_track_frames
    avg_speeds = {
        tid: spd for tid, spd in all_avg_speeds.items()
        if len(speed_est._positions.get(tid, [])) >= min_f
    }
    stats_path = str(output_dir / config.stats_file_name)

    with open(stats_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Player ID", "Avg Speed (m/s)", "Avg Speed (km/h)"])
        for tid in sorted(avg_speeds.keys()):
            spd = avg_speeds[tid]
            csv_writer.writerow([tid, f"{spd:.2f}", f"{spd * 3.6:.2f}"])

    total_detected = len(all_avg_speeds)
    total_kept = len(avg_speeds)
    print(f"\n  Tracks detected: {total_detected}, kept after filtering (>={min_f} frames): {total_kept}")
    print(f"\n  Player Statistics")
    print(f"  {'Player ID':<12} {'Avg Speed (m/s)':<18} {'Avg Speed (km/h)':<18}")
    print(f"  {'-'*48}")
    for tid in sorted(avg_speeds.keys()):
        spd = avg_speeds[tid]
        print(f"  {tid:<12} {spd:<18.2f} {spd * 3.6:<18.2f}")

    print(f"\n  Stats saved: {stats_path}")

    heatmap_dir = str(output_dir / config.heatmap_dir_name)
    heatmap_gen._positions = {
        tid: pos for tid, pos in heatmap_gen._positions.items()
        if tid in avg_speeds
    }
    heatmap_gen.save_all(heatmap_dir)

    elapsed_total = time.time() - start_time
    print(f"\n  Pipeline completed in {elapsed_total:.1f}s")
    print("=" * 60)


def main():
    config = build_config_from_args()
    run_pipeline(config)


if __name__ == "__main__":
    main()
