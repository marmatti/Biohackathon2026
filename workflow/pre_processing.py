"""
Pre-processing utilities for the filament segmentation workflow.
"""

from typing import Any, List, Tuple

import numpy as np


class DatasetNormalizer:
    """
    Applies per-image robust min-max normalization to all collected samples.

    The input format matches SampleCollector output:
    (img_info, image, ground_truth).
    """

    def __init__(
        self,
        samples: List[Tuple[dict, np.ndarray, Any]],
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        eps: float = 1e-8,
    ):
        self.samples = samples
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.eps = eps

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float32)
        lo = float(np.percentile(arr, self.lower_percentile))
        hi = float(np.percentile(arr, self.upper_percentile))

        if hi - lo <= self.eps:
            return np.zeros_like(arr, dtype=np.float32)

        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)
        return arr.astype(np.float32)

    def run(self) -> List[Tuple[dict, np.ndarray, Any]]:
        """Return a normalized copy of all samples."""
        normalized: List[Tuple[dict, np.ndarray, Any]] = []
        for img_info, img, gt in self.samples:
            normalized.append((img_info, self._normalize_image(img), gt))
        return normalized
