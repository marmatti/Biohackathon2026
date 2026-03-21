from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from simulator.filament import FilamentEvent, FilamentFrame

def make_frame_label(
    frame_idx: int,
    filament_frame: Optional[FilamentFrame],
    image_shape: tuple[int, int],
) -> Dict[str, Any]:
    """
    Build a JSON-serializable label dictionary for one frame.
    """
    skeleton_mask_2d = filament_pixels_to_mask(
        pixel_indices_rc=None if filament_frame is None else filament_frame.pixel_indices_rc,
        image_shape=image_shape,
    )

    if filament_frame is None:
        thick_mask_2d = np.zeros(image_shape, dtype=np.uint8)

        return {
            "frame": int(frame_idx),
            "filament_present": False,
            "filament_length_3d_px": 0.0,
            "filament_length_2d_px": 0.0,
            "filament_pixel_indices_rc": [],
            "filament_skeleton_mask_2d": skeleton_mask_2d.tolist(),
            "filament_mask_2d": thick_mask_2d.tolist(),
        }

    return {
        "frame": int(frame_idx),
        "filament_present": bool(filament_frame.present),
        "filament_length_3d_px": float(filament_frame.length_3d_px),
        "filament_length_2d_px": float(filament_frame.length_2d_px),
        "filament_pixel_indices_rc": filament_frame.pixel_indices_rc.astype(int).tolist(),
        "filament_skeleton_mask_2d": skeleton_mask_2d.tolist(),
        "filament_mask_2d": filament_frame.mask_2d.astype(np.uint8).tolist(),
    }


def filament_pixels_to_mask(
    pixel_indices_rc: Optional[np.ndarray],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Convert filament pixel indices [row, col] into a dense 2D binary mask.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if pixel_indices_rc is None or len(pixel_indices_rc) == 0:
        return mask

    pixel_indices_rc = np.asarray(pixel_indices_rc, dtype=np.int32)
    rr = pixel_indices_rc[:, 0]
    cc = pixel_indices_rc[:, 1]

    valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    rr = rr[valid]
    cc = cc[valid]

    mask[rr, cc] = 1
    return mask


def make_movie_labels(
    frame_labels: List[Dict[str, Any]],
    event: Optional[FilamentEvent] = None,
    seed: Optional[int] = None,
    image_shape: Optional[tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Build a movie-level JSON-serializable label dictionary.
    """
    positive_frames = [f["frame"] for f in frame_labels if f["filament_present"]]

    if positive_frames:
        appearance_frame = int(min(positive_frames))
        disappearance_frame = int(max(positive_frames))
        lifetime_frames = int(len(positive_frames))
        has_filament = True
    else:
        appearance_frame = None
        disappearance_frame = None
        lifetime_frames = 0
        has_filament = False

    labels: Dict[str, Any] = {
        "seed": None if seed is None else int(seed),
        "image_shape": None if image_shape is None else [int(image_shape[0]), int(image_shape[1])],
        "n_frames": int(len(frame_labels)),
        "has_filament": has_filament,
        "appearance_frame": appearance_frame,
        "disappearance_frame": disappearance_frame,
        "lifetime_frames": lifetime_frames,
        "frames": frame_labels,
    }

    if event is not None:
        labels["event"] = {
            "onset_frame": int(event.onset_frame),
            "growth_duration": int(event.growth_duration),
            "hold_duration": int(event.hold_duration),
            "shrink_duration": int(event.shrink_duration),
            "end_frame": int(event.end_frame),
            "max_length_px": float(event.max_length_px),
        }
    else:
        labels["event"] = None

    return labels


def save_labels_json(
    labels: Dict[str, Any],
    out_path: str | Path,
    indent: int = 2,
) -> Path:
    """
    Save movie labels to a JSON file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=indent)

    return out_path


def summarize_frame_labels(
    frame_labels: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute a lightweight summary from per-frame labels.
    """
    lengths_3d = np.array([f["filament_length_3d_px"] for f in frame_labels], dtype=float)
    lengths_2d = np.array([f["filament_length_2d_px"] for f in frame_labels], dtype=float)
    present = np.array([f["filament_present"] for f in frame_labels], dtype=bool)

    return {
        "n_frames": int(len(frame_labels)),
        "n_positive_frames": int(np.sum(present)),
        "max_length_3d_px": float(np.max(lengths_3d)) if len(lengths_3d) > 0 else 0.0,
        "max_length_2d_px": float(np.max(lengths_2d)) if len(lengths_2d) > 0 else 0.0,
        "mean_length_3d_px_positive": float(np.mean(lengths_3d[present])) if np.any(present) else 0.0,
        "mean_length_2d_px_positive": float(np.mean(lengths_2d[present])) if np.any(present) else 0.0,
    }