from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Cell:
    """
    Minimal budding-yeast-like cell model.

    Each cell has a mother blob and optionally a bud blob.
    Some appearance parameters are sampled per cell.
    """
    cell_id: int
    mother_center_rc: Tuple[float, float]
    mother_radius_px: float
    mother_axis_ratio: float
    mother_angle_rad: float

    bud_present: bool
    bud_center_rc: Tuple[float, float] | None
    bud_radius_px: float | None
    bud_axis_ratio: float | None
    bud_angle_rad: float | None

    intensity: float
    edge_softness_px: float


def sample_cells(
    cfg_cells: dict,
    image_shape: Tuple[int, int],
    rng: np.random.Generator,
) -> List[Cell]:
    """
    Sample a population of 2D budding-cell blobs with slight ellipticity.
    """
    height, width = image_shape

    n_cells = int(cfg_cells["n_cells"])
    bud_probability = float(cfg_cells["bud_probability"])

    r_lo, r_hi = cfg_cells["mother_radius_px_range"]
    bud_ratio_lo, bud_ratio_hi = cfg_cells["bud_radius_ratio_range"]
    intensity_lo, intensity_hi = cfg_cells["intensity_range"]
    edge_lo, edge_hi = cfg_cells["edge_softness_px_range"]
    margin_lo, margin_hi = cfg_cells["placement_margin_px_range"]
    dist_lo, dist_hi = cfg_cells["min_center_distance_scale_range"]
    ell_lo, ell_hi = cfg_cells["ellipticity_ratio_range"]

    cells: List[Cell] = []
    max_attempts = 5000
    attempts = 0

    while len(cells) < n_cells and attempts < max_attempts:
        attempts += 1

        mother_radius = float(rng.uniform(r_lo, r_hi))
        mother_axis_ratio = float(rng.uniform(ell_lo, ell_hi))
        mother_angle_rad = float(rng.uniform(0.0, 2.0 * np.pi))

        intensity = float(rng.uniform(intensity_lo, intensity_hi))
        edge_softness_px = float(rng.uniform(edge_lo, edge_hi))
        margin = float(rng.uniform(margin_lo, margin_hi))
        min_center_distance_scale = float(rng.uniform(dist_lo, dist_hi))

        row = float(rng.uniform(margin, height - margin))
        col = float(rng.uniform(margin, width - margin))
        mother_center = (row, col)

        too_close = False
        for existing in cells:
            erow, ecol = existing.mother_center_rc
            min_dist = min_center_distance_scale * (
                mother_radius + existing.mother_radius_px
            )
            if np.hypot(row - erow, col - ecol) < min_dist:
                too_close = True
                break
        if too_close:
            continue

        bud_present = bool(rng.random() < bud_probability)
        bud_center = None
        bud_radius = None
        bud_axis_ratio = None
        bud_angle_rad = None

        if bud_present:
            bud_ratio = float(rng.uniform(bud_ratio_lo, bud_ratio_hi))
            bud_radius = bud_ratio * mother_radius
            bud_axis_ratio = float(rng.uniform(ell_lo, ell_hi))
            bud_angle_rad = float(rng.uniform(0.0, 2.0 * np.pi))

            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            center_dist = mother_radius + 0.55 * bud_radius
            bud_row = row + center_dist * np.sin(theta)
            bud_col = col + center_dist * np.cos(theta)

            if not (
                margin <= bud_row <= height - margin
                and margin <= bud_col <= width - margin
            ):
                continue

            bud_center = (float(bud_row), float(bud_col))

        cells.append(
            Cell(
                cell_id=len(cells) + 1,
                mother_center_rc=mother_center,
                mother_radius_px=mother_radius,
                mother_axis_ratio=mother_axis_ratio,
                mother_angle_rad=mother_angle_rad,
                bud_present=bud_present,
                bud_center_rc=bud_center,
                bud_radius_px=bud_radius,
                bud_axis_ratio=bud_axis_ratio,
                bud_angle_rad=bud_angle_rad,
                intensity=intensity,
                edge_softness_px=edge_softness_px,
            )
        )

    if len(cells) == 0:
        raise RuntimeError("Failed to sample any cells. Try relaxing placement settings.")

    if len(cells) < n_cells:
        raise RuntimeError(
            f"Only sampled {len(cells)} / {n_cells} cells. "
            "Try reducing n_cells, reducing placement margins, or allowing more overlap."
        )

    return cells


def render_cells(
    cells: List[Cell],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Render diffuse fluorescence from all cells as soft blobs.

    Each cell uses its own sampled edge_softness_px.
    """
    rr, cc = np.indices(image_shape, dtype=np.float32)
    image = np.zeros(image_shape, dtype=np.float32)

    for cell in cells:
        mask = cell_soft_mask(cell, rr, cc)
        image += cell.intensity * mask

    return image


def render_cell_id_mask(
    cells: List[Cell],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Render a hard integer mask assigning each pixel to the nearest cell blob
    if it lies inside that cell's mother or bud circle.
    """
    rr, cc = np.indices(image_shape, dtype=np.float32)
    id_mask = np.zeros(image_shape, dtype=np.int32)
    best_score = np.full(image_shape, -np.inf, dtype=np.float32)

    for cell in cells:
        score = cell_membership_score(cell, rr, cc)
        update = score > best_score
        id_mask[update] = cell.cell_id
        best_score[update] = score[update]

    id_mask[best_score < 0.0] = 0
    return id_mask


def choose_filament_host_cell(
    cells: List[Cell],
    rng: np.random.Generator,
) -> Cell:
    idx = int(rng.integers(0, len(cells)))
    return cells[idx]


def cell_soft_mask(
    cell: Cell,
    rr: np.ndarray,
    cc: np.ndarray,
) -> np.ndarray:
    """
    Soft union of mother and optional bud masks using rotated ellipses.
    """
    mother = soft_ellipse(
        rr,
        cc,
        center_rc=cell.mother_center_rc,
        base_radius_px=cell.mother_radius_px,
        axis_ratio=cell.mother_axis_ratio,
        angle_rad=cell.mother_angle_rad,
        edge_softness_px=cell.edge_softness_px,
    )

    if (
        not cell.bud_present
        or cell.bud_center_rc is None
        or cell.bud_radius_px is None
        or cell.bud_axis_ratio is None
        or cell.bud_angle_rad is None
    ):
        return mother

    bud = soft_ellipse(
        rr,
        cc,
        center_rc=cell.bud_center_rc,
        base_radius_px=cell.bud_radius_px,
        axis_ratio=cell.bud_axis_ratio,
        angle_rad=cell.bud_angle_rad,
        edge_softness_px=cell.edge_softness_px,
    )

    return np.maximum(mother, bud)

def cell_membership_score(
    cell: Cell,
    rr: np.ndarray,
    cc: np.ndarray,
) -> np.ndarray:
    """
    Positive inside cell, negative outside; larger means deeper inside.
    """
    mother = signed_ellipse_score(
        rr,
        cc,
        center_rc=cell.mother_center_rc,
        base_radius_px=cell.mother_radius_px,
        axis_ratio=cell.mother_axis_ratio,
        angle_rad=cell.mother_angle_rad,
    )

    if (
        not cell.bud_present
        or cell.bud_center_rc is None
        or cell.bud_radius_px is None
        or cell.bud_axis_ratio is None
        or cell.bud_angle_rad is None
    ):
        return mother

    bud = signed_ellipse_score(
        rr,
        cc,
        center_rc=cell.bud_center_rc,
        base_radius_px=cell.bud_radius_px,
        axis_ratio=cell.bud_axis_ratio,
        angle_rad=cell.bud_angle_rad,
    )
    return np.maximum(mother, bud)


def soft_disk(
    rr: np.ndarray,
    cc: np.ndarray,
    center_rc: Tuple[float, float],
    radius_px: float,
    edge_softness_px: float,
) -> np.ndarray:
    """
    Render a softened disk using a logistic boundary.
    """
    dist = np.hypot(rr - center_rc[0], cc - center_rc[1])
    signed = radius_px - dist
    softness = max(float(edge_softness_px), 1e-3)
    return 1.0 / (1.0 + np.exp(-signed / softness))


def signed_disk_score(
    rr: np.ndarray,
    cc: np.ndarray,
    center_rc: Tuple[float, float],
    radius_px: float,
) -> np.ndarray:
    """
    Positive inside the disk, negative outside.
    """
    dist = np.hypot(rr - center_rc[0], cc - center_rc[1])
    return radius_px - dist

def soft_ellipse(
    rr: np.ndarray,
    cc: np.ndarray,
    center_rc: Tuple[float, float],
    base_radius_px: float,
    axis_ratio: float,
    angle_rad: float,
    edge_softness_px: float,
) -> np.ndarray:
    """
    Render a softened rotated ellipse using a logistic boundary.

    base_radius_px is converted into ellipse semi-axes while approximately
    preserving area:
        a = r * sqrt(q)
        b = r / sqrt(q)
    where q = axis_ratio = a / b.
    """
    q = max(float(axis_ratio), 1e-6)
    a = float(base_radius_px) * np.sqrt(q)
    b = float(base_radius_px) / np.sqrt(q)

    x = cc - center_rc[1]
    y = rr - center_rc[0]

    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    x_rot = cos_t * x + sin_t * y
    y_rot = -sin_t * x + cos_t * y

    rho = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)
    signed = 1.0 - rho

    softness = max(float(edge_softness_px) / max(base_radius_px, 1e-3), 1e-3)
    return 1.0 / (1.0 + np.exp(-signed / softness))


def signed_ellipse_score(
    rr: np.ndarray,
    cc: np.ndarray,
    center_rc: Tuple[float, float],
    base_radius_px: float,
    axis_ratio: float,
    angle_rad: float,
) -> np.ndarray:
    """
    Positive inside the ellipse, negative outside.
    """
    q = max(float(axis_ratio), 1e-6)
    a = float(base_radius_px) * np.sqrt(q)
    b = float(base_radius_px) / np.sqrt(q)

    x = cc - center_rc[1]
    y = rr - center_rc[0]

    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    x_rot = cos_t * x + sin_t * y
    y_rot = -sin_t * x + cos_t * y

    rho = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2)
    return 1.0 - rho