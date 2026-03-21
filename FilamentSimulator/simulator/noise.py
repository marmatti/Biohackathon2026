from __future__ import annotations

import numpy as np


def add_noise_and_background(
    clean_image: np.ndarray,
    cfg_noise: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add background, low-frequency gradient, shot noise, read noise, and clipping.

    Parameters
    ----------
    clean_image : np.ndarray
        Clean fluorescence image before detector/background effects.
    cfg_noise : dict
        The 'noise' section of the config.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Noisy image with the same shape as clean_image.
    """
    image = np.asarray(clean_image, dtype=np.float32).copy()

    image = add_background_offset(
        image=image,
        background_offset=float(cfg_noise["background_offset"]),
    )

    image = add_background_gradient(
        image=image,
        gradient_max=float(cfg_noise["background_gradient_max"]),
        rng=rng,
    )

    image = add_poisson_shot_noise(
        image=image,
        poisson_scale=float(cfg_noise["poisson_scale"]),
        rng=rng,
    )

    image = add_gaussian_read_noise(
        image=image,
        sigma=float(cfg_noise["gaussian_read_sigma"]),
        rng=rng,
    )

    image = clip_image(
        image=image,
        clip_min=float(cfg_noise["clip_min"]),
        clip_max=float(cfg_noise["clip_max"]),
    )

    return image.astype(np.float32)


def add_background_offset(
    image: np.ndarray,
    background_offset: float,
) -> np.ndarray:
    """
    Add a constant detector/background offset to the whole image.
    """
    return image.astype(np.float32) + float(background_offset)


def add_background_gradient(
    image: np.ndarray,
    gradient_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add a simple low-frequency linear background gradient.

    The gradient direction and magnitude are random for each frame.
    """
    h, w = image.shape
    yy, xx = np.indices((h, w), dtype=np.float32)

    # Normalized coordinates centered near zero.
    if h > 1:
        yy = (yy / (h - 1)) - 0.5
    else:
        yy = yy * 0.0

    if w > 1:
        xx = (xx / (w - 1)) - 0.5
    else:
        xx = xx * 0.0

    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    amplitude = float(rng.uniform(-gradient_max, gradient_max))

    gradient = amplitude * (np.cos(theta) * xx + np.sin(theta) * yy)
    return image.astype(np.float32) + gradient.astype(np.float32)


def add_poisson_shot_noise(
    image: np.ndarray,
    poisson_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add Poisson shot noise.

    The image is converted to photon-like counts by dividing by poisson_scale,
    sampled with a Poisson distribution, and then scaled back.
    """
    poisson_scale = max(float(poisson_scale), 1e-6)

    # Poisson rate must be non-negative.
    lam = np.clip(image.astype(np.float32), 0.0, None) / poisson_scale
    noisy = rng.poisson(lam).astype(np.float32) * poisson_scale
    return noisy


def add_gaussian_read_noise(
    image: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add zero-mean Gaussian read noise.
    """
    sigma = max(float(sigma), 0.0)
    if sigma == 0.0:
        return image.astype(np.float32)

    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape).astype(np.float32)
    return image.astype(np.float32) + noise


def clip_image(
    image: np.ndarray,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """
    Clip image intensities to the configured detector range.
    """
    return np.clip(image.astype(np.float32), clip_min, clip_max).astype(np.float32)