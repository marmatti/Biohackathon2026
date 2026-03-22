"""
Segmentation methods for detecting brighter spots (filaments) in microscopy images.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy import ndimage
from skimage import filters, morphology, segmentation, exposure


class BaseSegmenter(ABC):
    """Base class for segmentation methods."""

    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Produce binary mask from grayscale image.
        Args:
            image: (H, W) grayscale array, uint8 or float
        Returns:
            (H, W) binary mask, 0 or 1
        """
        pass

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.segment(image)


class OtsuSegmenter(BaseSegmenter):
    """
    Otsu's method: global threshold that minimizes intra-class variance.
    Good for images with bimodal intensity distribution.
    """

    def __init__(self, invert: bool = False):
        """
        Args:
            invert: If True, segment dark regions instead of bright.
        """
        self.invert = invert

    def segment(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float64)
        thresh = filters.threshold_otsu(img)
        mask = (img >= thresh).astype(np.uint8)
        if self.invert:
            mask = 1 - mask
        return mask


class AdaptiveThresholdSegmenter(BaseSegmenter):
    """
    Local adaptive thresholding. Handles uneven illumination by computing
    threshold per local neighborhood.
    """

    def __init__(
        self,
        block_size: int = 15,
        c: float = 0,
        method: str = "gaussian",
        invert: bool = False,
    ):
        """
        Args:
            block_size: Size of local neighborhood (odd).
            c: Constant subtracted from mean.
            method: 'gaussian' or 'mean'.
            invert: Segment dark regions if True.
        """
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.c = c
        self.method = method
        self.invert = invert

    def segment(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float64)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        thresh = filters.threshold_local(
            img, block_size=self.block_size, offset=self.c, method=self.method
        )
        mask = (img >= thresh).astype(np.uint8)
        if self.invert:
            mask = 1 - mask
        return mask


class MorphologicalSegmenter(BaseSegmenter):
    """
    Threshold + morphological operations (opening/closing) to clean up
    small noise and fill holes.
    """

    def __init__(
        self,
        threshold_percentile: float = 75,
        open_radius: int = 1,
        close_radius: int = 2,
    ):
        """
        Args:
            threshold_percentile: Percentile for intensity threshold (0-100).
            open_radius: Radius for opening (removes small bright spots).
            close_radius: Radius for closing (fills small holes).
        """
        self.threshold_percentile = threshold_percentile
        self.open_radius = open_radius
        self.close_radius = close_radius

    def segment(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float64)
        thresh = np.percentile(img, self.threshold_percentile)
        mask = (img >= thresh).astype(np.uint8)
        se_open = morphology.disk(self.open_radius)
        se_close = morphology.disk(self.close_radius)
        mask = morphology.binary_opening(mask, se_open).astype(np.uint8)
        mask = morphology.binary_closing(mask, se_close).astype(np.uint8)
        return mask


class TopHatSegmenter(BaseSegmenter):
    """
    White top-hat: extracts small bright structures by subtracting
    morphological opening from original. Ideal for bright spots on dark background.
    """

    def __init__(self, radius: int = 3, threshold_percentile: float = 50):
        """
        Args:
            radius: Structuring element radius for opening.
            threshold_percentile: Percentile to binarize top-hat result.
        """
        self.radius = radius
        self.threshold_percentile = threshold_percentile

    def segment(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float64)
        se = morphology.disk(self.radius)
        opened = ndimage.grey_opening(img, footprint=se)
        tophat = img - opened
        nonzero = tophat[tophat > 0]
        if len(nonzero) == 0:
            return np.zeros_like(img, dtype=np.uint8)
        thresh = np.percentile(nonzero, self.threshold_percentile)
        mask = (tophat >= thresh).astype(np.uint8)
        return mask


class WatershedSegmenter(BaseSegmenter):
    """
    Watershed segmentation from markers. Uses local maxima or intensity-based
    markers to separate overlapping bright regions.
    """

    def __init__(
        self,
        min_distance: int = 5,
        min_size: int = 20,
        sigma: float = 1.0,
    ):
        """
        Args:
            min_distance: Min pixels between markers.
            min_size: Min region size (smaller regions removed).
            sigma: Gaussian blur before finding markers.
        """
        self.min_distance = min_distance
        self.min_size = min_size
        self.sigma = sigma

    def segment(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float64)
        if self.sigma > 0:
            img = ndimage.gaussian_filter(img, self.sigma)
        local_max = morphology.local_maxima(img)
        markers = ndimage.label(local_max)[0]
        if markers.max() == 0:
            # Fallback: use intensity threshold as single marker
            thresh = np.percentile(img, 90)
            markers = np.zeros_like(img, dtype=int)
            markers[img >= thresh] = 1
            markers = ndimage.label(markers)[0]
        labels = segmentation.watershed(-img, markers, mask=img > np.percentile(img, 25))
        # Convert to binary: any labeled region
        mask = (labels > 0).astype(np.uint8)
        # Remove tiny regions
        if self.min_size > 0:
            labeled, num = ndimage.label(mask)
            for i in range(1, num + 1):
                if (labeled == i).sum() < self.min_size:
                    mask[labeled == i] = 0
        return mask


class CLAHEThresholdSegmenter(BaseSegmenter):
    """
    Contrast-limited adaptive histogram equalization (CLAHE) + Otsu.
    Enhances local contrast before thresholding.
    """

    def __init__(self, clip_limit: float = 0.03, tile_size: int = 8):
        self.clip_limit = clip_limit
        self.tile_size = tile_size

    def segment(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float64)
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        enhanced = exposure.equalize_adapthist(
            img, clip_limit=self.clip_limit, kernel_size=(self.tile_size,) * 2
        )
        enhanced = (enhanced * 255).astype(np.uint8)
        thresh = filters.threshold_otsu(enhanced)
        return (enhanced >= thresh).astype(np.uint8)


class FrangiSegmenter(BaseSegmenter):
    """
    Frangi vesselness filter. Analyses eigenvalues of the Hessian matrix to
    detect tubular/elongated structures. Strongly responds to filaments while
    suppressing round blobs (cells).
    """

    def __init__(
        self,
        sigmas: tuple = (1, 2, 3, 4),
        threshold_percentile: float = 80,
        black_ridges: bool = False,
    ):
        """
        Args:
            sigmas: Scale range for Hessian computation (tube widths in pixels).
            threshold_percentile: Percentile of non-zero Frangi response for binarization.
            black_ridges: If True, detect dark ridges on bright background.
        """
        self.sigmas = sigmas
        self.threshold_percentile = threshold_percentile
        self.black_ridges = black_ridges

    def segment(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float64)
        if img.max() > 1.0:
            img = img / img.max()
        response = filters.frangi(img, sigmas=self.sigmas, black_ridges=self.black_ridges)
        nonzero = response[response > 0]
        if len(nonzero) == 0:
            return np.zeros_like(img, dtype=np.uint8)
        thresh = np.percentile(nonzero, self.threshold_percentile)
        return (response >= thresh).astype(np.uint8)


class MeijeringSegmenter(BaseSegmenter):
    """
    Meijering neuriteness filter. Designed for detecting neurites and filaments.
    Based on Hessian eigenvalues, optimised for ridge-like (1D) structures.
    Generally best for thin, elongated filaments.
    """

    def __init__(
        self,
        sigmas: tuple = (1, 2, 3, 4),
        threshold_percentile: float = 80,
        black_ridges: bool = False,
    ):
        """
        Args:
            sigmas: Scale range for Hessian computation.
            threshold_percentile: Percentile of non-zero response for binarization.
            black_ridges: If True, detect dark ridges on bright background.
        """
        self.sigmas = sigmas
        self.threshold_percentile = threshold_percentile
        self.black_ridges = black_ridges

    def segment(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float64)
        if img.max() > 1.0:
            img = img / img.max()
        response = filters.meijering(img, sigmas=self.sigmas, black_ridges=self.black_ridges)
        nonzero = response[response > 0]
        if len(nonzero) == 0:
            return np.zeros_like(img, dtype=np.uint8)
        thresh = np.percentile(nonzero, self.threshold_percentile)
        return (response >= thresh).astype(np.uint8)


class SatoSegmenter(BaseSegmenter):
    """
    Sato tubeness filter. Hessian-based filter for detecting tubular structures.
    Similar to Frangi but with a different response function.
    """

    def __init__(
        self,
        sigmas: tuple = (1, 2, 3, 4),
        threshold_percentile: float = 80,
        black_ridges: bool = False,
    ):
        """
        Args:
            sigmas: Scale range for Hessian computation.
            threshold_percentile: Percentile of non-zero response for binarization.
            black_ridges: If True, detect dark ridges on bright background.
        """
        self.sigmas = sigmas
        self.threshold_percentile = threshold_percentile
        self.black_ridges = black_ridges

    def segment(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float64)
        if img.max() > 1.0:
            img = img / img.max()
        response = filters.sato(img, sigmas=self.sigmas, black_ridges=self.black_ridges)
        nonzero = response[response > 0]
        if len(nonzero) == 0:
            return np.zeros_like(img, dtype=np.uint8)
        thresh = np.percentile(nonzero, self.threshold_percentile)
        return (response >= thresh).astype(np.uint8)
