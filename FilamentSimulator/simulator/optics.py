from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from simulator.filament import FilamentFrame


def render_clean_frame(
    cell_image: np.ndarray,
    filament_frame: Optional[FilamentFrame],
    image_shape: Tuple[int, int],
    cfg_optics: dict,
) -> np.ndarray:
    """
    Render a clean fluorescence frame before background/noise.
    """
    if cell_image.shape != image_shape:
        raise ValueError(
            f"cell_image shape {cell_image.shape} does not match image_shape {image_shape}"
        )

    clean = np.array(cell_image, dtype=np.float32, copy=True)

    if filament_frame is not None and filament_frame.present:
        filament_img = render_filament_image(
            filament_frame=filament_frame,
            image_shape=image_shape,
            cfg_optics=cfg_optics,
        )
        clean += filament_img

    clean = apply_gaussian_psf(
        image=clean,
        sigma_xy_px=float(cfg_optics["psf_sigma_xy_px"]),
    )
    return clean.astype(np.float32)


def render_filament_image(
    filament_frame: FilamentFrame,
    image_shape: Tuple[int, int],
    cfg_optics: dict,
) -> np.ndarray:
    """
    Render a filament as a projected thick tube by splatting a soft disk around
    each projected centerline point, then letting the global PSF blur it.

    This uses the filament radius so the projected object has visible width.
    """
    h, w = image_shape
    img = np.zeros((h, w), dtype=np.float32)

    if filament_frame.centerline_xyz.shape[0] == 0:
        return img

    centerline_xyz = filament_frame.centerline_xyz
    centerline_rc = filament_frame.centerline_rc
    z_vals = centerline_xyz[:, 2]

    z_sigma = float(cfg_optics["defocus_sigma_z_px"])
    weights = defocus_attenuation(z_vals, z_sigma)

    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return img

    # Split total intensity across centerline samples
    point_intensities = filament_frame.intensity * (weights / weight_sum)

    for (row, col), val in zip(centerline_rc, point_intensities):
        splat_soft_disk(
            image=img,
            row=float(row),
            col=float(col),
            radius_px=float(filament_frame.radius_px),
            value=float(val),
        )

    return img


def splat_soft_disk(
    image: np.ndarray,
    row: float,
    col: float,
    radius_px: float,
    value: float,
) -> None:
    """
    Add a normalized soft disk centered at (row, col) into `image`.

    The kernel is normalized so the total added intensity equals `value`.
    """
    if radius_px <= 0 or value == 0:
        return

    h, w = image.shape
    pad = int(np.ceil(radius_px + 1.5))

    r0 = max(0, int(np.floor(row)) - pad)
    r1 = min(h, int(np.floor(row)) + pad + 1)
    c0 = max(0, int(np.floor(col)) - pad)
    c1 = min(w, int(np.floor(col)) + pad + 1)

    if r0 >= r1 or c0 >= c1:
        return

    rr, cc = np.indices((r1 - r0, c1 - c0), dtype=np.float32)
    rr = rr + r0
    cc = cc + c0

    dist = np.sqrt((rr - row) ** 2 + (cc - col) ** 2)

    # Soft-edged disk: flat-ish center with smooth decay near edge
    # You can sharpen/soften this later if needed.
    sigma = max(radius_px / 2.0, 1e-3)
    kernel = np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32)
    kernel[dist > radius_px] *= 0.25  # keep a weak tail near edges

    ksum = float(kernel.sum())
    if ksum <= 1e-12:
        return

    image[r0:r1, c0:c1] += value * (kernel / ksum)


def apply_gaussian_psf(
    image: np.ndarray,
    sigma_xy_px: float,
) -> np.ndarray:
    """
    Apply an isotropic 2D Gaussian blur to approximate the microscope PSF.
    """
    if sigma_xy_px <= 0:
        return np.asarray(image, dtype=np.float32)
    return gaussian_filter(image.astype(np.float32), sigma=sigma_xy_px, mode="constant")


def defocus_attenuation(
    z_vals: np.ndarray,
    sigma_z_px: float,
) -> np.ndarray:
    """
    Brightness attenuation as a function of axial distance from focus.
    """
    sigma_z_px = max(float(sigma_z_px), 1e-6)
    z_vals = np.asarray(z_vals, dtype=np.float32)
    return np.exp(-0.5 * (z_vals / sigma_z_px) ** 2).astype(np.float32)