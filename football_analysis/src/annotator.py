import cv2
import numpy as np

from src.tracker import Track


_PALETTE = [
    (255, 50, 50), (50, 255, 50), (50, 50, 255), (255, 255, 50),
    (255, 50, 255), (50, 255, 255), (200, 100, 50), (50, 100, 200),
    (200, 200, 50), (50, 200, 200), (150, 50, 200), (200, 50, 150),
]


def _get_color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def annotate_frame(
    frame: np.ndarray,
    tracks: list[Track],
    speeds: dict[int, float] | None = None,
) -> np.ndarray:
    for track in tracks:
        color = _get_color(track.track_id)
        x1, y1, x2, y2 = track.bbox.astype(int)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID {track.track_id}"
        if speeds and track.track_id in speeds:
            spd = speeds[track.track_id]
            label += f" | {spd:.1f} m/s"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame
