from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class FilamentEvent:
    """
    Time profile of a single optional filament event in one movie.
    """

    onset_frame: int
    growth_duration: int
    hold_duration: int
    shrink_duration: int
    max_length_px: float

    @property
    def end_frame(self) -> int:
        return (
            self.onset_frame
            + self.growth_duration
            + self.hold_duration
            + self.shrink_duration
        )

    def is_present(self, frame_idx: int) -> bool:
        return self.onset_frame <= frame_idx < self.end_frame


@dataclass
class FilamentPose:
    """
    Latent 3D pose of a straight rod filament.

    center_xyz uses the convention:
        x = image column
        y = image row
        z = axial depth
    """
    center_xyz: np.ndarray
    direction_xyz: np.ndarray
    radius_px: float
    intensity: float
    bend_mode_indices: np.ndarray      # shape (K,)
    bend_angles_rad: np.ndarray        # shape (K,)
    bend_amplitudes_px: np.ndarray     # shape (K,)


@dataclass
class FilamentFrame:
    """
    Per-frame latent and projected filament state.
    """
    present: bool
    length_3d_px: float
    length_2d_px: float
    centerline_xyz: np.ndarray          # shape (N, 3)
    centerline_rc: np.ndarray           # shape (N, 2)
    pixel_indices_rc: np.ndarray        # skeleton pixels, shape (M, 2)
    mask_2d: np.ndarray                 # thick projected mask, shape (H, W)
    intensity: float
    radius_px: float


def sample_filament_event(
    cfg_filament_event: dict,
    cfg_filament: dict,
    n_frames: int,
    rng: np.random.Generator,
) -> Optional[FilamentEvent]:
    """
    Sample a filament event automatically from n_frames.

    Design:
    - sample a total event duration as a fraction of n_frames
    - sample onset so the event fits fully inside the movie
    - split total duration into growth / hold / shrink
    - bias hold to be longer on average than growth/shrink
    """
    p_any = float(cfg_filament_event["probability_any_filament"])
    if rng.random() >= p_any:
        return None

    len_lo, len_hi = cfg_filament["length_px_range"]
    max_length_px = float(rng.uniform(len_lo, len_hi))

    frac_lo, frac_hi = cfg_filament_event["total_duration_fraction_range"]

    total_lo = max(6, int(round(float(frac_lo) * n_frames)))
    total_hi = max(total_lo, int(round(float(frac_hi) * n_frames)))
    total_duration = int(rng.integers(total_lo, total_hi + 1))

    # Onset chosen so the full event fits in the movie
    latest_onset = n_frames - total_duration
    onset = int(rng.integers(0, latest_onset + 1))

    # Minimum phase length
    min_phase = max(2, int(round(0.04 * n_frames)))

    # Ensure total_duration can accommodate all 3 phases
    if 3 * min_phase > total_duration:
        min_phase = max(1, total_duration // 3)

    remaining = total_duration - 3 * min_phase

    # Bias: hold tends to be longer than growth/shrink
    props = rng.dirichlet([2.0, 4.0, 2.0])

    raw = props * remaining
    extra = np.floor(raw).astype(int)
    remainder = remaining - int(np.sum(extra))

    frac = raw - np.floor(raw)
    order = np.argsort(-frac)
    for i in range(remainder):
        extra[order[i]] += 1

    growth = int(min_phase + extra[0])
    hold = int(min_phase + extra[1])
    shrink = int(min_phase + extra[2])

    return FilamentEvent(
        onset_frame=onset,
        growth_duration=growth,
        hold_duration=hold,
        shrink_duration=shrink,
        max_length_px=max_length_px,
    )


def sample_initial_filament_pose(
    host_cell,
    cfg_filament: dict,
    rng: np.random.Generator,
) -> FilamentPose:
    z_lo, z_hi = cfg_filament["z_center_range_px"]
    radius_lo, radius_hi = cfg_filament["radius_px_range"]
    radius_px = float(rng.uniform(radius_lo, radius_hi))
    
    intensity_mean = float(cfg_filament["intensity_mean"])
    intensity_std = float(cfg_filament["intensity_std"])
    intensity = max(1.0, float(rng.normal(intensity_mean, intensity_std)))

    row, col = host_cell.mother_center_rc
    z = float(rng.uniform(z_lo, z_hi))

    center_xyz = np.array([col, row, z], dtype=np.float32)
    direction_xyz = random_unit_vector(rng).astype(np.float32)

    bend_lo, bend_hi = cfg_filament["bend_count_range"]
    n_bends = int(rng.integers(bend_lo, bend_hi + 1))

    amp_lo, amp_hi = cfg_filament["curvature_amplitude_px_range"]

    if n_bends == 0:
        bend_mode_indices = np.zeros((0,), dtype=np.int32)
        bend_angles_rad = np.zeros((0,), dtype=np.float32)
        bend_amplitudes_px = np.zeros((0,), dtype=np.float32)
    else:
        bend_mode_indices = np.arange(1, n_bends + 1, dtype=np.int32)
        bend_angles_rad = rng.uniform(0.0, 2.0 * np.pi, size=n_bends).astype(np.float32)
        bend_amplitudes_px = rng.uniform(amp_lo, amp_hi, size=n_bends).astype(np.float32)

    return FilamentPose(
        center_xyz=center_xyz,
        direction_xyz=direction_xyz,
        radius_px=radius_px,
        intensity=intensity,
        bend_mode_indices=bend_mode_indices,
        bend_angles_rad=bend_angles_rad,
        bend_amplitudes_px=bend_amplitudes_px,
    )

def make_curved_rod_centerline(
    center_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    length_px: float,
    n_points: int,
    bend_mode_indices: np.ndarray,
    bend_angles_rad: np.ndarray,
    bend_amplitudes_px: np.ndarray,
) -> np.ndarray:
    """
    Build a slightly curved 3D filament centerline.

    The curve is represented as a straight axis plus 0-3 transverse bend modes.
    """
    u = normalize_vector(direction_xyz)

    # Build an orthonormal transverse basis e1, e2 perpendicular to u
    e1, e2 = make_transverse_basis(u)

    s_vals = np.linspace(-0.5 * length_px, 0.5 * length_px, n_points, dtype=np.float32)
    centerline = center_xyz[None, :] + s_vals[:, None] * u[None, :]

    if len(bend_mode_indices) == 0:
        return centerline.astype(np.float32)

    x = (2.0 * s_vals / max(length_px, 1e-6)).astype(np.float32)  # normalized to [-1, 1]

    bend_disp = np.zeros_like(centerline, dtype=np.float32)

    for mode_idx, theta, amp in zip(bend_mode_indices, bend_angles_rad, bend_amplitudes_px):
        v = np.cos(theta) * e1 + np.sin(theta) * e2
        profile = bend_profile(x, int(mode_idx))[:, None]  # shape (N, 1)
        bend_disp += float(amp) * profile * v[None, :]

    return (centerline + bend_disp).astype(np.float32)

    
def update_filament_pose(
    pose: FilamentPose,
    host_cell,
    cfg_filament: dict,
    rng: np.random.Generator,
    max_translation_tries: int = 20,
) -> FilamentPose:
    """
    Apply a small random 3D rotation and translation to the filament.

    Translation is constrained so that the filament center remains inside the
    chosen host cell in the image plane. The z coordinate is clamped to the
    configured z range.

    Notes
    -----
    This constrains the filament center, not the full rod endpoints.
    """
    # --- rotation ---
    sigma_deg = float(cfg_filament["rotation_sigma_deg_per_frame"])
    sigma_rad = np.deg2rad(sigma_deg)

    delta_angle = float(rng.normal(0.0, sigma_rad))
    axis = random_unit_vector(rng)

    new_dir = rotate_vector_rodrigues(pose.direction_xyz, axis, delta_angle)
    new_dir = normalize_vector(new_dir)

    # --- translation ---
    tx_sigma, ty_sigma, tz_sigma = cfg_filament["translation_sigma_xyz_px"]
    z_lo, z_hi = cfg_filament["z_center_range_px"]

    old_center = pose.center_xyz.copy()
    accepted_center = old_center.copy()

    for _ in range(max_translation_tries):
        delta_center = np.array(
            [
                rng.normal(0.0, tx_sigma),  # x = col
                rng.normal(0.0, ty_sigma),  # y = row
                rng.normal(0.0, tz_sigma),  # z
            ],
            dtype=np.float32,
        )

        candidate = old_center + delta_center
        candidate[2] = np.clip(candidate[2], z_lo, z_hi)

        cand_col = float(candidate[0])
        cand_row = float(candidate[1])

        if point_inside_cell_rc(host_cell, cand_row, cand_col):
            accepted_center = candidate
            break

    # --- curvature drift ---
    amp_sigma = float(cfg_filament.get("curvature_drift_sigma_px", 0.0))
    new_bend_amplitudes = pose.bend_amplitudes_px.copy()
    if len(new_bend_amplitudes) > 0 and amp_sigma > 0:
        new_bend_amplitudes += rng.normal(
            0.0, amp_sigma, size=len(new_bend_amplitudes)
        ).astype(np.float32)
        new_bend_amplitudes = np.clip(new_bend_amplitudes, 0.0, None)
        
    return FilamentPose(
        center_xyz=accepted_center.astype(np.float32),
        direction_xyz=new_dir.astype(np.float32),
        radius_px=float(pose.radius_px),
        intensity=float(pose.intensity),
        bend_mode_indices=pose.bend_mode_indices.copy(),
        bend_angles_rad=pose.bend_angles_rad.copy(),
        bend_amplitudes_px=new_bend_amplitudes.astype(np.float32),
    )

def point_inside_cell_rc(cell, row: float, col: float) -> bool:
    """
    Return True if a 2D point (row, col) lies inside the mother or bud disk.
    """
    if point_inside_disk_rc(row, col, cell.mother_center_rc, cell.mother_radius_px):
        return True

    if (
        cell.bud_present
        and cell.bud_center_rc is not None
        and cell.bud_radius_px is not None
        and point_inside_disk_rc(row, col, cell.bud_center_rc, cell.bud_radius_px)
    ):
        return True

    return False


def point_inside_disk_rc(
    row: float,
    col: float,
    center_rc: tuple[float, float],
    radius_px: float,
) -> bool:
    """
    Return True if a 2D point lies inside a disk.
    """
    return (row - center_rc[0]) ** 2 + (col - center_rc[1]) ** 2 <= radius_px ** 2


def build_filament_frame(
    event: Optional[FilamentEvent],
    pose: Optional[FilamentPose],
    frame_idx: int,
    image_shape: Tuple[int, int],
    cfg_filament: dict,
) -> Optional[FilamentFrame]:
    """
    Build the latent and projected filament state for one frame.
    """
    if event is None or pose is None or not event.is_present(frame_idx):
        return None

    length_3d_px = filament_length_at_frame(event, frame_idx)
    if length_3d_px <= 0.0:
        return None

    n_points = int(cfg_filament["n_centerline_points"])
    centerline_xyz = make_curved_rod_centerline(
        center_xyz=pose.center_xyz,
        direction_xyz=pose.direction_xyz,
        length_px=length_3d_px,
        n_points=n_points,
        bend_mode_indices=pose.bend_mode_indices,
        bend_angles_rad=pose.bend_angles_rad,
        bend_amplitudes_px=pose.bend_amplitudes_px,
    )
    
    length_3d_px_actual = polyline_length(centerline_xyz)
    centerline_rc = project_centerline_xyz_to_rc(centerline_xyz)
    length_2d_px = polyline_length(centerline_rc)
    
    # Skeleton pixels from projected centerline
    pixel_indices_rc = rasterize_polyline_to_pixels(
        centerline_rc=centerline_rc,
        image_shape=image_shape,
    )
    
    # Thick projected filament mask using radius
    mask_2d = rasterize_polyline_to_thick_mask(
        centerline_rc=centerline_rc,
        image_shape=image_shape,
        radius_px=pose.radius_px,
    )
    
    return FilamentFrame(
        present=True,
        length_3d_px=float(length_3d_px_actual),
        length_2d_px=float(length_2d_px),
        centerline_xyz=centerline_xyz,
        centerline_rc=centerline_rc,
        pixel_indices_rc=pixel_indices_rc,
        mask_2d=mask_2d,
        intensity=float(pose.intensity),
        radius_px=float(pose.radius_px),
    )

def filament_length_at_frame(
    event: FilamentEvent,
    frame_idx: int,
) -> float:
    """
    Piecewise-linear growth / hold / shrink length profile.
    """
    t = frame_idx - event.onset_frame

    if t < 0:
        return 0.0

    g = event.growth_duration
    h = event.hold_duration
    s = event.shrink_duration
    lmax = event.max_length_px

    if t < g:
        # Grow from 0 to lmax.
        frac = (t + 1) / max(g, 1)
        return lmax * frac

    t -= g
    if t < h:
        return lmax

    t -= h
    if t < s:
        frac = 1.0 - (t + 1) / max(s, 1)
        return max(0.0, lmax * frac)

    return 0.0


def project_centerline_xyz_to_rc(centerline_xyz: np.ndarray) -> np.ndarray:
    """
    Orthographic projection from 3D (x, y, z) to image coordinates (row, col).
    """
    cols = centerline_xyz[:, 0]
    rows = centerline_xyz[:, 1]
    return np.stack([rows, cols], axis=1).astype(np.float32)


def rasterize_polyline_to_pixels(
    centerline_rc: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Convert a projected polyline into unique integer pixel indices [row, col].
    """
    if centerline_rc.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)

    dense_rc = densify_polyline(centerline_rc, step_px=0.25)
    rr = np.rint(dense_rc[:, 0]).astype(np.int32)
    cc = np.rint(dense_rc[:, 1]).astype(np.int32)

    h, w = image_shape
    valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    rr = rr[valid]
    cc = cc[valid]

    if rr.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    pixels = np.column_stack([rr, cc])
    pixels = np.unique(pixels, axis=0)
    return pixels.astype(np.int32)


def rasterize_polyline_to_thick_mask(
    centerline_rc: np.ndarray,
    image_shape: tuple[int, int],
    radius_px: float,
) -> np.ndarray:
    """
    Rasterize a projected centerline into a thick binary mask using a disk of
    radius `radius_px` around each sampled centerline point.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if centerline_rc.shape[0] == 0:
        return mask

    dense_rc = densify_polyline(centerline_rc, step_px=0.25)

    rr_grid, cc_grid = np.indices((h, w), dtype=np.float32)

    for row, col in dense_rc:
        dist2 = (rr_grid - row) ** 2 + (cc_grid - col) ** 2
        mask[dist2 <= radius_px ** 2] = 1

    return mask

    
def densify_polyline(
    points_rc: np.ndarray,
    step_px: float = 0.25,
) -> np.ndarray:
    """
    Densify a polyline so rasterization leaves fewer gaps.
    """
    if points_rc.shape[0] <= 1:
        return points_rc.astype(np.float32)

    dense = [points_rc[0]]
    for i in range(points_rc.shape[0] - 1):
        p0 = points_rc[i]
        p1 = points_rc[i + 1]
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))

        if seg_len < 1e-8:
            continue

        n_steps = max(2, int(np.ceil(seg_len / step_px)) + 1)
        t_vals = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
        pts = p0[None, :] + t_vals[:, None] * seg[None, :]
        dense.append(pts[1:])

    return np.concatenate(
        [p[None, :] if p.ndim == 1 else p for p in dense],
        axis=0,
    ).astype(np.float32)


def polyline_length(points: np.ndarray) -> float:
    """
    Arc length of an ordered 2D or 3D polyline.
    """
    if points.shape[0] <= 1:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """
    Sample a random 3D unit vector isotropically.
    """
    v = rng.normal(size=3).astype(np.float32)
    return normalize_vector(v)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    """
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / norm).astype(np.float32)


def rotate_vector_rodrigues(
    v: np.ndarray,
    axis: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """
    Rotate vector v around axis by angle_rad using Rodrigues' formula.
    """
    axis = normalize_vector(axis)
    v = v.astype(np.float32)

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    return (
        v * cos_a
        + np.cross(axis, v) * sin_a
        + axis * np.dot(axis, v) * (1.0 - cos_a)
    ).astype(np.float32)

def make_transverse_basis(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return two orthonormal vectors perpendicular to u.
    """
    u = normalize_vector(u)

    # Choose a helper vector not too parallel to u
    if abs(float(u[2])) < 0.9:
        helper = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    e1 = np.cross(u, helper)
    e1 = normalize_vector(e1)
    e2 = np.cross(u, e1)
    e2 = normalize_vector(e2)

    return e1.astype(np.float32), e2.astype(np.float32)


def bend_profile(x: np.ndarray, mode_idx: int) -> np.ndarray:
    """
    Spatial bend profile on normalized arclength x in [-1, 1].

    mode 1: single bowed arc
    mode 2: S-like bend
    mode 3: higher-order waviness
    """
    x = np.asarray(x, dtype=np.float32)

    if mode_idx == 1:
        # symmetric bow, zero at ends
        return (1.0 - x**2).astype(np.float32)

    if mode_idx == 2:
        # antisymmetric S-bend, zero at ends and center
        return (x * (1.0 - x**2)).astype(np.float32)

    if mode_idx == 3:
        # higher-order symmetric bend
        return ((1.0 - x**2) * (2.0 * x**2 - 0.5)).astype(np.float32)

    # fallback
    return (1.0 - x**2).astype(np.float32)