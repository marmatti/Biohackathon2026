"""
Experiment classes for segmentation pipeline.
All visualization and parameter scan logic lives here; the notebook only calls these classes.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import matplotlib.pyplot as plt

from .segmentation import BaseSegmenter


class SampleCollector:
    """Collects samples from a loader with optional intensity statistics."""

    def __init__(self, loader, max_total: int = 20):
        self.loader = loader
        self.max_total = max_total
        self.samples: List[Tuple[dict, np.ndarray, Any]] = []

    def run(self) -> List[Tuple[dict, np.ndarray, Any]]:
        """Collect samples and optionally print intensity stats."""
        self.samples = []
        for img_info, img, gt in self.loader.iter_samples(max_total=self.max_total):
            self.samples.append((img_info, img, gt))
        if self.samples:
            all_pixels = np.concatenate([s[1].ravel() for s in self.samples])
            print(f"Intensity range: [{all_pixels.min():.1f}, {all_pixels.max():.1f}]")
            print(f"Mean intensity: {all_pixels.mean():.2f}")
            print(f"Std intensity: {all_pixels.std():.2f}")
        else:
            print("No images loaded. Ensure TIFF files are in workflow/data/.")
        return self.samples


class SampleGridVisualizer:
    """Visualizes multiple samples in a single row."""

    def __init__(self, samples: List[Tuple[dict, np.ndarray, Any]], n_display: int = 4):
        self.samples = samples
        self.n_display = n_display

    def run(self) -> None:
        """Display samples in a grid."""
        if not self.samples:
            return
        fig, axes = plt.subplots(1, self.n_display, figsize=(12, 3))
        for i, (img_info, img, _) in enumerate(self.samples[: self.n_display]):
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(f"{img_info['file_name']} f{img_info['frame']}")
            axes[i].axis("off")
        plt.tight_layout()
        plt.show()


class SegmentationVisualizer:
    """Visualizes input image and segmentation output for a single segmenter."""

    def __init__(
        self,
        segmenter: BaseSegmenter,
        title: str = "prediction",
        figsize: Tuple[int, int] = (8, 4),
    ):
        self.segmenter = segmenter
        self.title = title
        self.figsize = figsize

    def run(self, img: np.ndarray) -> None:
        """Display input and segmentation result."""
        pred = self.segmenter.segment(img)
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Input")
        axes[1].imshow(pred, cmap="gray")
        axes[1].set_title(self.title)
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


class ParameterScanExperiment:
    """
    Runs a 2D parameter scan over a segmenter class.
    Rows = first param, columns = second param. Shows input in first column.
    """

    def __init__(
        self,
        segmenter_class: Type[BaseSegmenter],
        row_param: Tuple[str, List[Any]],
        col_param: Tuple[str, List[Any]],
        fixed_params: Optional[Dict[str, Any]] = None,
        figsize_per_cell: Tuple[float, float] = (3, 3),
    ):
        """
        Args:
            segmenter_class: Segmenter class to instantiate.
            row_param: (param_name, [values]) for rows.
            col_param: (param_name, [values]) for columns.
            fixed_params: Params passed to all instantiations.
            figsize_per_cell: Size of each subplot.
        """
        self.segmenter_class = segmenter_class
        self.row_param = row_param
        self.col_param = col_param
        self.fixed_params = fixed_params or {}
        self.figsize_per_cell = figsize_per_cell

    def run(self, img: np.ndarray, suptitle: Optional[str] = None) -> None:
        """Run parameter scan and display grid."""
        row_name, row_values = self.row_param
        col_name, col_values = self.col_param
        n_rows = len(row_values)
        n_cols = len(col_values) + 1  # +1 for input

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                n_cols * self.figsize_per_cell[0],
                n_rows * self.figsize_per_cell[1],
            ),
        )
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, row_val in enumerate(row_values):
            axes[i, 0].imshow(img, cmap="gray")
            axes[i, 0].set_title("Input" if i == 0 else "")
            axes[i, 0].set_ylabel(str(row_val), fontsize=11)
            axes[i, 0].axis("off")

            for j, col_val in enumerate(col_values):
                params = {
                    **self.fixed_params,
                    row_name: row_val,
                    col_name: col_val,
                }
                segmenter = self.segmenter_class(**params)
                pred = segmenter.segment(img)
                axes[i, j + 1].imshow(pred, cmap="gray")
                axes[i, j + 1].set_title(f"{col_name}={col_val}")
                axes[i, j + 1].axis("off")

        if suptitle:
            plt.suptitle(suptitle, y=1.02)
        plt.tight_layout()
        plt.show()


class AdaptiveParameterScan(ParameterScanExperiment):
    """
    Preconfigured parameter scan for AdaptiveThresholdSegmenter.
    Scans over c and method (gaussian vs mean).
    """

    def __init__(
        self,
        c_values: List[float] = None,
        methods: List[str] = None,
        block_size: int = 15,
    ):
        from .segmentation import AdaptiveThresholdSegmenter

        c_values = c_values or [-5, 0, 5, 10]
        methods = methods or ["gaussian", "mean"]
        super().__init__(
            segmenter_class=AdaptiveThresholdSegmenter,
            row_param=("method", methods),
            col_param=("c", c_values),
            fixed_params={"block_size": block_size},
        )

    def run(self, img: np.ndarray) -> None:
        super().run(img, suptitle="Adaptive threshold: parameter scan (c, method)")


class TubenessComparisonExperiment:
    """
    Compare Frangi, Meijering, and Sato tubeness filters side-by-side.
    Shows raw filter response and binarised result for each.
    """

    def __init__(
        self,
        sigmas: tuple = (1, 2, 3, 4),
        threshold_percentile: float = 80,
        black_ridges: bool = False,
    ):
        self.sigmas = sigmas
        self.threshold_percentile = threshold_percentile
        self.black_ridges = black_ridges

    def run(self, img: np.ndarray) -> None:
        from skimage import filters as skf
        from .segmentation import FrangiSegmenter, MeijeringSegmenter, SatoSegmenter

        img_f = np.asarray(img, dtype=np.float64)
        if img_f.max() > 1.0:
            img_f = img_f / img_f.max()

        configs = [
            ("Frangi", skf.frangi, FrangiSegmenter),
            ("Meijering", skf.meijering, MeijeringSegmenter),
            ("Sato", skf.sato, SatoSegmenter),
        ]

        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        for row, (name, raw_fn, seg_cls) in enumerate(configs):
            response = raw_fn(img_f, sigmas=self.sigmas, black_ridges=self.black_ridges)
            segmenter = seg_cls(
                sigmas=self.sigmas,
                threshold_percentile=self.threshold_percentile,
                black_ridges=self.black_ridges,
            )
            mask = segmenter.segment(img)

            axes[row, 0].imshow(img, cmap="gray")
            axes[row, 0].set_title("Input" if row == 0 else "")
            axes[row, 0].set_ylabel(name, fontsize=13, fontweight="bold")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(response, cmap="hot")
            axes[row, 1].set_title("Filter response" if row == 0 else "")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(mask, cmap="gray")
            pct = self.threshold_percentile
            axes[row, 2].set_title(f"Binary (p{pct})" if row == 0 else "")
            axes[row, 2].axis("off")

        plt.suptitle(
            f"Hessian tubeness filter comparison (sigmas={self.sigmas}, p{self.threshold_percentile})",
            y=1.01,
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()


class TubenessThresholdScanExperiment:
    """
    Scan threshold_percentile values for one tubeness segmenter class.
    Displays Input + one binary mask per threshold so the best cutoff can be
    selected visually.
    """

    def __init__(
        self,
        segmenter_class: Type[BaseSegmenter],
        sigmas: tuple = (1, 2, 3, 4),
        threshold_values: Optional[List[float]] = None,
        black_ridges: bool = False,
        figsize_per_cell: Tuple[float, float] = (2.6, 2.6),
    ):
        """
        Args:
            segmenter_class: One of FrangiSegmenter, MeijeringSegmenter, SatoSegmenter.
            sigmas: Scale range for Hessian filters.
            threshold_values: Percentile values to test (default: 70..90 step 5).
            black_ridges: Passed to the segmenter.
            figsize_per_cell: Width/height per subplot in inches.
        """
        self.segmenter_class = segmenter_class
        self.sigmas = sigmas
        self.threshold_values = threshold_values or list(range(70, 91, 5))
        self.black_ridges = black_ridges
        self.figsize_per_cell = figsize_per_cell

    def run(self, img: np.ndarray, suptitle: Optional[str] = None) -> None:
        """Render Input + masks for each threshold value."""
        thresholds = self.threshold_values
        n_cols = len(thresholds) + 1  # +1 for input panel

        fig, axes = plt.subplots(
            1,
            n_cols,
            figsize=(n_cols * self.figsize_per_cell[0], self.figsize_per_cell[1]),
        )
        if n_cols == 1:
            axes = np.array([axes])

        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Input")
        axes[0].axis("off")

        for j, pct in enumerate(thresholds):
            segmenter = self.segmenter_class(
                sigmas=self.sigmas,
                threshold_percentile=pct,
                black_ridges=self.black_ridges,
            )
            pred = segmenter.segment(img)
            axes[j + 1].imshow(pred, cmap="gray")
            axes[j + 1].set_title(f"p{pct:g}")
            axes[j + 1].axis("off")

        name = self.segmenter_class.__name__.replace("Segmenter", "")
        title = suptitle or (
            f"{name} threshold scan (sigmas={self.sigmas}, "
            f"threshold_percentile={thresholds[0]}..{thresholds[-1]})"
        )
        plt.suptitle(title, y=1.03)
        plt.tight_layout()
        plt.show()
