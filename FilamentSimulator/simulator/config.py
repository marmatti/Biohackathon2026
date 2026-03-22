from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file into a nested dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must contain a YAML mapping at the top level.")

    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Minimal validation for the simulator config.

    This is intentionally lightweight for the first version.
    """
    required_top_keys = [
        "seed",
        "movie",
        "output",
        "cells",
        "filament_event",
        "filament",
        "optics",
        "noise",
    ]
    for key in required_top_keys:
        if key not in cfg:
            raise KeyError(f"Missing required top-level config key: '{key}'")

    # --- seed ---
    if not isinstance(cfg["seed"], int):
        raise TypeError(f"Config key 'seed' must be an int, got {type(cfg['seed']).__name__}")

    # --- movie ---
    _require_positive_int(cfg["movie"], "n_frames")
    _require_positive_int(cfg["movie"], "height")
    _require_positive_int(cfg["movie"], "width")

    # --- output ---
    _require_key(cfg["output"], "movie_npy_name")
    _require_key(cfg["output"], "clean_movie_npy_name")
    _require_key(cfg["output"], "movie_labels_json_name")
    _require_key(cfg["output"], "frame_labels_json_name")
    _require_key(cfg["output"], "noisy_tiff_dir")
    _require_key(cfg["output"], "clean_tiff_dir")
    
    # --- cells ---
    _require_positive_int(cfg["cells"], "n_cells")
    _require_range(cfg["cells"], "mother_radius_px_range")
    _require_number(cfg["cells"], "bud_probability")
    _require_range(cfg["cells"], "bud_radius_ratio_range")
    _require_range(cfg["cells"], "intensity_range")
    _require_range(cfg["cells"], "edge_softness_px_range")
    _require_range(cfg["cells"], "placement_margin_px_range")
    _require_range(cfg["cells"], "min_center_distance_scale_range")
    _require_range(cfg["cells"], "ellipticity_ratio_range")
    
    bud_probability = cfg["cells"]["bud_probability"]
    if not (0.0 <= bud_probability <= 1.0):
        raise ValueError("cells.bud_probability must be between 0 and 1")

    # --- filament_event ---
    _require_number(cfg["filament_event"], "probability_any_filament")

    p_any = cfg["filament_event"]["probability_any_filament"]
    if not (0.0 <= p_any <= 1.0):
        raise ValueError("filament_event.probability_any_filament must be between 0 and 1")
        
    frac_lo, frac_hi = cfg["filament_event"]["total_duration_fraction_range"]
    if frac_lo <= 0 or frac_hi <= 0:
        raise ValueError("filament_event.total_duration_fraction_range values must be > 0")
        
    # --- filament ---
    _require_range(cfg["filament"], "length_px_range")
    _require_range(cfg["filament"], "radius_px_range")   
    _require_number(cfg["filament"], "intensity_mean")
    _require_positive_number(cfg["filament"], "intensity_std")
    _require_range(cfg["filament"], "z_center_range_px")
    _require_positive_number(cfg["filament"], "rotation_sigma_deg_per_frame")
    _require_positive_int(cfg["filament"], "n_centerline_points")

    _require_key(cfg["filament"], "translation_sigma_xyz_px")
    translation_sigma = cfg["filament"]["translation_sigma_xyz_px"]
    if not isinstance(translation_sigma, (list, tuple)) or len(translation_sigma) != 3:
        raise TypeError("filament.translation_sigma_xyz_px must be a length-3 list or tuple")
    for i, v in enumerate(translation_sigma):
        if not isinstance(v, (int, float)):
            raise TypeError(
                f"filament.translation_sigma_xyz_px[{i}] must be numeric, "
                f"got {type(v).__name__}"
            )
        if v < 0:
            raise ValueError(f"filament.translation_sigma_xyz_px[{i}] must be >= 0")

    # --- optics ---
    _require_positive_number(cfg["optics"], "psf_sigma_xy_px")
    _require_positive_number(cfg["optics"], "defocus_sigma_z_px")

    # --- noise ---
    _require_number(cfg["noise"], "background_offset")
    _require_number(cfg["noise"], "background_gradient_max")
    _require_positive_number(cfg["noise"], "poisson_scale")
    _require_positive_number(cfg["noise"], "gaussian_read_sigma")
    _require_number(cfg["noise"], "clip_min")
    _require_number(cfg["noise"], "clip_max")

    clip_min = cfg["noise"]["clip_min"]
    clip_max = cfg["noise"]["clip_max"]
    if clip_max <= clip_min:
        raise ValueError("noise.clip_max must be greater than noise.clip_min")


def ensure_output_dir(path: str | Path) -> Path:
    """
    Create the output directory if needed and return it as a Path.
    """
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _require_key(d: Dict[str, Any], key: str) -> None:
    if key not in d:
        raise KeyError(f"Missing required config key: '{key}'")


def _require_number(d: Dict[str, Any], key: str) -> None:
    _require_key(d, key)
    if not isinstance(d[key], (int, float)):
        raise TypeError(f"Config key '{key}' must be a number, got {type(d[key]).__name__}")


def _require_positive_number(d: Dict[str, Any], key: str) -> None:
    _require_number(d, key)
    if d[key] <= 0:
        raise ValueError(f"Config key '{key}' must be > 0")


def _require_positive_int(d: Dict[str, Any], key: str) -> None:
    _require_key(d, key)
    if not isinstance(d[key], int):
        raise TypeError(f"Config key '{key}' must be an int, got {type(d[key]).__name__}")
    if d[key] <= 0:
        raise ValueError(f"Config key '{key}' must be > 0")


def _require_range(d: Dict[str, Any], key: str) -> None:
    _require_key(d, key)
    value = d[key]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"Config key '{key}' must be a length-2 list or tuple")
    lo, hi = value
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        raise TypeError(f"Config key '{key}' must contain numeric values")
    if hi < lo:
        raise ValueError(f"Config range '{key}' must satisfy high >= low")