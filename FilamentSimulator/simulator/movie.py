from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import tifffile
import numpy as np
import matplotlib.pyplot as plt

from simulator.config import ensure_output_dir, load_config
from simulator.cells import (
    choose_filament_host_cell,
    render_cells,
    sample_cells,
)
from simulator.filament import (
    build_filament_frame,
    sample_filament_event,
    sample_initial_filament_pose,
    update_filament_pose,
)
from simulator.labels import make_frame_label, make_movie_labels
from simulator.noise import add_noise_and_background
from simulator.optics import render_clean_frame


def simulate_movie(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate one movie and return images plus labels.

    Returns
    -------
    dict
        Dictionary with keys:
        - "clean_movie": np.ndarray of shape (T, H, W)
        - "noisy_movie": np.ndarray of shape (T, H, W)
        - "movie_labels": dict
        - "frame_labels": list[dict]
        - "host_cell_id": int or None
    """
    rng = np.random.default_rng(cfg["seed"])

    image_shape = (int(cfg["movie"]["height"]), int(cfg["movie"]["width"]))
    n_frames = int(cfg["movie"]["n_frames"])

    # --- sample static cell scene ---
    cells = sample_cells(
        cfg_cells=cfg["cells"],
        image_shape=image_shape,
        rng=rng,
    )

    cell_image = render_cells(
        cells=cells,
        image_shape=image_shape,
    )

    # --- sample optional filament event ---
    event = sample_filament_event(
        cfg_filament_event=cfg["filament_event"],
        cfg_filament=cfg["filament"],
        n_frames=n_frames,
        rng=rng,
    )

    if event is None:
        host_cell = None
        pose = None
    else:
        host_cell = choose_filament_host_cell(cells, rng)
        pose = sample_initial_filament_pose(
            host_cell=host_cell,
            cfg_filament=cfg["filament"],
            rng=rng,
        )

    # --- simulate frame by frame ---
    clean_frames = []
    noisy_frames = []
    frame_labels = []

    pose_t = pose

    for t in range(n_frames):
        filament_frame = build_filament_frame(
            event=event,
            pose=pose_t,
            frame_idx=t,
            image_shape=image_shape,
            cfg_filament=cfg["filament"],
        )

        clean_image = render_clean_frame(
            cell_image=cell_image,
            filament_frame=filament_frame,
            image_shape=image_shape,
            cfg_optics=cfg["optics"],
        )

        noisy_image = add_noise_and_background(
            clean_image=clean_image,
            cfg_noise=cfg["noise"],
            rng=rng,
        )

        frame_label = make_frame_label(
            frame_idx=t,
            filament_frame=filament_frame,
            image_shape=image_shape,
        )

        clean_frames.append(clean_image.astype(np.float32))
        noisy_frames.append(noisy_image.astype(np.float32))
        frame_labels.append(frame_label)

        if filament_frame is not None and host_cell is not None and pose_t is not None:
            pose_t = update_filament_pose(
                pose=pose_t,
                host_cell=host_cell,
                cfg_filament=cfg["filament"],
                rng=rng,
            )

    clean_movie = np.stack(clean_frames, axis=0).astype(np.float32)
    noisy_movie = np.stack(noisy_frames, axis=0).astype(np.float32)

    movie_labels = make_movie_labels(
        frame_labels=frame_labels,
        event=event,
        seed=cfg["seed"],
        image_shape=image_shape,
    )

    movie_labels["n_cells"] = int(len(cells))
    movie_labels["host_cell_id"] = None if host_cell is None else int(host_cell.cell_id)

    return {
        "clean_movie": clean_movie,
        "noisy_movie": noisy_movie,
        "movie_labels": movie_labels,
        "frame_labels": frame_labels,
        "host_cell_id": None if host_cell is None else int(host_cell.cell_id),
    }


def float_to_uint16(
    image: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """
    Convert a float image to uint16 using linear scaling.

    If vmin/vmax are not provided, they are inferred from the image.
    """
    image = np.asarray(image, dtype=np.float32)

    if vmin is None:
        vmin = float(np.min(image))
    if vmax is None:
        vmax = float(np.max(image))

    if vmax <= vmin:
        return np.zeros_like(image, dtype=np.uint16)

    scaled = (image - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.round(scaled * 65535.0).astype(np.uint16)


def save_movie_as_uint16_tiff_sequence(
    movie: np.ndarray,
    out_dir: str | Path,
    prefix: str,
    zero_pad: int = 4,
    pmin: float = 1.0,
    pmax: float = 99.5,
) -> Path:
    """
    Save a movie stack as an ordered uint16 TIFF sequence using percentile scaling.

    Files are written as:
        {prefix}_0000.tif
        {prefix}_0001.tif
        ...
    """
    out_dir = ensure_output_dir(out_dir)
    movie = np.asarray(movie, dtype=np.float32)

    vmin = float(np.percentile(movie, pmin))
    vmax = float(np.percentile(movie, pmax))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    n_frames = movie.shape[0]
    for i in range(n_frames):
        frame_u16 = float_to_uint16(movie[i], vmin=vmin, vmax=vmax)
        out_path = out_dir / f"{prefix}_{i:0{zero_pad}d}.tif"
        tifffile.imwrite(out_path, frame_u16)

    return out_dir

def write_simulation_outputs(
    sim_data: Dict[str, Any],
    out_dir: str | Path,
    cfg: Dict[str, Any],
) -> Path:
    """
    Write simulated images and labels to disk.
    
    Files written
    -------------
    - noisy movie stack: output/<movie_npy_name>
    - clean movie stack: output/<clean_movie_npy_name>
    - noisy uint16 TIFF frames: output/<noisy_tiff_dir>/
    - clean uint16 TIFF frames: output/<clean_tiff_dir>/
    - movie labels: output/<movie_labels_json_name>
    - frame labels: output/<frame_labels_json_name>
    """
    out_dir = ensure_output_dir(out_dir)

    output_cfg = cfg.get("output", {})
    noisy_movie_name = output_cfg.get("movie_npy_name", "movie.npy")
    clean_movie_name = output_cfg.get("clean_movie_npy_name", "clean_movie.npy")
    movie_labels_name = output_cfg.get("movie_labels_json_name", "movie_labels.json")
    frame_labels_name = output_cfg.get("frame_labels_json_name", "frame_labels.json")
    noisy_tiff_dir = output_cfg.get("noisy_tiff_dir", "noisy_tiff_frames")
    clean_tiff_dir = output_cfg.get("clean_tiff_dir", "clean_tiff_frames")

    noisy_movie_path = out_dir / noisy_movie_name
    clean_movie_path = out_dir / clean_movie_name
    movie_labels_path = out_dir / movie_labels_name
    frame_labels_path = out_dir / frame_labels_name

    # Save NPY stacks
    np.save(noisy_movie_path, sim_data["noisy_movie"])
    np.save(clean_movie_path, sim_data["clean_movie"])

    # Save ordered PNG sequences
    save_movie_as_uint16_tiff_sequence(
        movie=sim_data["noisy_movie"],
        out_dir=out_dir / noisy_tiff_dir,
        prefix="noisy",
        pmin=1.0,
        pmax=99.5,
    )
    save_movie_as_uint16_tiff_sequence(
        movie=sim_data["clean_movie"],
        out_dir=out_dir / clean_tiff_dir,
        prefix="clean",
        pmin=1.0,
        pmax=99.5,
    )

    movie_labels = dict(sim_data["movie_labels"])
    frame_labels = list(sim_data["frame_labels"])

    if "frames" in movie_labels:
        movie_labels = dict(movie_labels)
        movie_labels.pop("frames")

    with movie_labels_path.open("w", encoding="utf-8") as f:
        json.dump(movie_labels, f, indent=2)

    with frame_labels_path.open("w", encoding="utf-8") as f:
        json.dump(frame_labels, f, indent=2)

    return out_dir


def simulate_and_save_movie(
    config_path: str | Path,
    out_dir: str | Path,
) -> Dict[str, Any]:
    """
    Convenience wrapper: load config, simulate one movie, and save outputs.
    """
    cfg = load_config(config_path)
    sim_data = simulate_movie(cfg)
    write_simulation_outputs(sim_data, out_dir=out_dir, cfg=cfg)
    return sim_data