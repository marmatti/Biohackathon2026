"""
Segment tracker for visualizing segmentation across the full dataset.
Supports video export and data export (CSV, results).
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

from .segmentation import BaseSegmenter

ViewMode = Literal["overlay", "triple", "contour", "skeleton", "combined"]


class SegmentTracker:
    """
    Runs a segmenter across a dataset and visualizes results as video.
    All logic for tracking, rendering, and export lives here.
    """

    def __init__(
        self,
        segmenter: BaseSegmenter,
        loader: Any,
        max_frames_per_file: Optional[int] = None,
        max_total: Optional[int] = None,
    ):
        """
        Args:
            segmenter: Segmentation method to apply (e.g. OtsuSegmenter).
            loader: Data loader with iter_samples() yielding (img_info, img, gt).
            max_frames_per_file: Limit frames per TIFF file (None = all).
            max_total: Stop after this many frames total (None = all).
        """
        self.segmenter = segmenter
        self.loader = loader
        self.max_frames_per_file = max_frames_per_file
        self.max_total = max_total
        self._samples: List[Tuple[dict, np.ndarray, Any]] = []
        self._predictions: List[np.ndarray] = []

    def _collect_samples(self) -> List[Tuple[dict, np.ndarray, Any]]:
        """Collect samples from loader (cached)."""
        if self._samples:
            return self._samples
        kwargs = {}
        if self.max_frames_per_file is not None:
            kwargs["max_frames_per_file"] = self.max_frames_per_file
        if self.max_total is not None:
            kwargs["max_total"] = self.max_total
        try:
            self._samples = list(self.loader.iter_samples(**kwargs))
        except TypeError:
            self._samples = list(self.loader.iter_samples())
            if self.max_total is not None:
                self._samples = self._samples[: self.max_total]
        return self._samples

    def _run_segmentation(self) -> List[np.ndarray]:
        """Run segmenter on all collected samples (cached)."""
        samples = self._collect_samples()
        if self._predictions and len(self._predictions) == len(samples):
            return self._predictions
        self._predictions = [self.segmenter.segment(img) for _, img, _ in samples]
        return self._predictions

    def _iter_frames(self) -> Iterator[Tuple[dict, np.ndarray, np.ndarray]]:
        """Yield (img_info, image, prediction) for each frame."""
        samples = self._collect_samples()
        preds = self._run_segmentation()
        for (info, img, gt), pred in zip(samples, preds):
            yield info, img, pred

    def get_results(self) -> List[Dict[str, Any]]:
        """
        Return list of per-frame results for analysis.
        Each dict has: frame_idx, file_name, frame, area_px, centroid_y, centroid_x,
        mean_intensity, max_intensity (optional).
        """
        results = []
        for i, (info, img, pred) in enumerate(self._iter_frames()):
            area = int(pred.sum())
            if area > 0:
                yy, xx = np.where(pred > 0)
                cy, cx = float(yy.mean()), float(xx.mean())
                mean_int = float(img[pred > 0].mean())
                max_int = float(img[pred > 0].max())
            else:
                cy, cx, mean_int, max_int = np.nan, np.nan, np.nan, np.nan
            results.append({
                "frame_idx": i,
                "file_name": info.get("file_name", ""),
                "frame": info.get("frame", i),
                "area_px": area,
                "centroid_y": cy,
                "centroid_x": cx,
                "mean_intensity": mean_int,
                "max_intensity": max_int,
            })
        return results

    def export_csv(
        self,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Export per-frame statistics to CSV.
        Returns path to saved file.
        """
        results = self.get_results()
        if not results:
            return None
        try:
            import csv
        except ImportError:
            return None
        out = Path(output_path) if output_path else Path("segment_tracker_results.csv")
        out = out.resolve()
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            for r in results:
                row = {}
                for k, v in r.items():
                    if isinstance(v, float) and not np.isnan(v):
                        row[k] = f"{v:.4f}"
                    else:
                        row[k] = v
                writer.writerow(row)
        print(f"CSV saved to {out}")
        return out

    def _to_display_rgb(self, img: np.ndarray) -> np.ndarray:
        """Convert grayscale image to RGB uint8 for display."""
        img_norm = np.asarray(img, dtype=np.float64)
        if img_norm.max() > 1.0:
            img_norm = img_norm / img_norm.max()
        rgb = np.stack([img_norm] * 3, axis=-1)
        return (np.clip(rgb * 255, 0, 255)).astype(np.uint8)

    def _render_frame(
        self,
        img: np.ndarray,
        pred: np.ndarray,
        img_info: dict,
        overlay_alpha: float = 0.4,
        overlay_color: str = "green",
    ) -> np.ndarray:
        """
        Render a single frame: original image with segmentation overlay.
        Returns RGB array (H, W, 3) in uint8 for video display.
        """
        h, w = img.shape
        img_rgb = self._to_display_rgb(img).astype(np.float64) / 255.0

        # Color mapping: green, red, cyan, magenta
        color_map = {
            "green": (0, 1, 0),
            "red": (1, 0, 0),
            "cyan": (0, 1, 1),
            "magenta": (1, 0, 1),
            "yellow": (1, 1, 0),
        }
        rgb_val = color_map.get(overlay_color, (0, 1, 0))

        overlay = np.zeros((h, w, 4))
        overlay[:, :, 0] = rgb_val[0] * pred
        overlay[:, :, 1] = rgb_val[1] * pred
        overlay[:, :, 2] = rgb_val[2] * pred
        overlay[:, :, 3] = pred.astype(float) * overlay_alpha

        for c in range(3):
            img_rgb[:, :, c] = (
                img_rgb[:, :, c] * (1 - overlay[:, :, 3])
                + overlay[:, :, c] * overlay[:, :, 3]
            )

        return (np.clip(img_rgb * 255, 0, 255)).astype(np.uint8)

    def _render_contour(
        self,
        img: np.ndarray,
        pred: np.ndarray,
        overlay_color: str = "cyan",
        line_width: int = 2,
    ) -> np.ndarray:
        """Render original with contour outline only (no fill)."""
        from skimage import measure
        from skimage.draw import line

        img_rgb = self._to_display_rgb(img).astype(np.float64) / 255.0
        contours = measure.find_contours(pred.astype(float), 0.5)
        outline = np.zeros(img.shape, dtype=np.uint8)
        for contour in contours:
            for i in range(len(contour) - 1):
                r0, c0 = int(contour[i, 0]), int(contour[i, 1])
                r1, c1 = int(contour[i + 1, 0]), int(contour[i + 1, 1])
                rr, cc = line(r0, c0, r1, c1)
                valid = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
                outline[rr[valid], cc[valid]] = 1
        if line_width > 1:
            outline = morphology.dilation(outline, morphology.disk(line_width // 2))

        color_map = {"green": (0, 1, 0), "red": (1, 0, 0), "cyan": (0, 1, 1), "magenta": (1, 0, 1), "yellow": (1, 1, 0)}
        rgb_val = color_map.get(overlay_color, (0, 1, 1))
        for c in range(3):
            img_rgb[:, :, c] = np.where(outline > 0, rgb_val[c], img_rgb[:, :, c])

        return (np.clip(img_rgb * 255, 0, 255)).astype(np.uint8)

    def _render_skeleton(
        self,
        img: np.ndarray,
        pred: np.ndarray,
        overlay_color: str = "magenta",
    ) -> np.ndarray:
        """Render original with skeleton (thin centerline) overlay."""
        skeleton = morphology.skeletonize(pred > 0).astype(np.uint8)
        return self._render_frame(img, skeleton, {}, overlay_alpha=0.9, overlay_color=overlay_color)

    def _render_triple(
        self,
        img: np.ndarray,
        pred: np.ndarray,
        overlay_alpha: float = 0.4,
        overlay_color: str = "green",
    ) -> np.ndarray:
        """Render three panels: original | binary mask | overlay."""
        overlay = self._render_frame(img, pred, {}, overlay_alpha, overlay_color)
        mask_rgb = np.stack([pred * 255] * 3, axis=-1).astype(np.uint8)
        orig_rgb = self._to_display_rgb(img)
        return np.concatenate([orig_rgb, mask_rgb, overlay], axis=1)

    def _render_combined(
        self,
        img: np.ndarray,
        pred: np.ndarray,
    ) -> np.ndarray:
        """
        Overlay all representations on one frame with consistent colors:
        filled (green), contour (cyan), skeleton (magenta).
        """
        img_rgb = self._to_display_rgb(img).astype(np.float64) / 255.0
        h, w = img.shape

        # 1. Filled overlay (green, alpha 0.25)
        overlay = np.zeros((h, w, 4))
        overlay[:, :, 1] = pred  # Green
        overlay[:, :, 3] = pred.astype(float) * 0.25
        for c in range(3):
            img_rgb[:, :, c] = (
                img_rgb[:, :, c] * (1 - overlay[:, :, 3])
                + (0 if c != 1 else 1) * overlay[:, :, 3]
            )

        # 2. Contour outline (cyan)
        from skimage import measure
        from skimage.draw import line

        contours = measure.find_contours(pred.astype(float), 0.5)
        outline = np.zeros(img.shape, dtype=np.uint8)
        for contour in contours:
            for i in range(len(contour) - 1):
                r0, c0 = int(contour[i, 0]), int(contour[i, 1])
                r1, c1 = int(contour[i + 1, 0]), int(contour[i + 1, 1])
                rr, cc = line(r0, c0, r1, c1)
                valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                outline[rr[valid], cc[valid]] = 1
        outline = morphology.dilation(outline, morphology.disk(1))
        img_rgb[:, :, 0] = np.where(outline > 0, 0, img_rgb[:, :, 0])
        img_rgb[:, :, 1] = np.where(outline > 0, 1, img_rgb[:, :, 1])
        img_rgb[:, :, 2] = np.where(outline > 0, 1, img_rgb[:, :, 2])

        # 3. Skeleton (magenta)
        skeleton = morphology.skeletonize(pred > 0).astype(np.uint8)
        img_rgb[:, :, 0] = np.where(skeleton > 0, 1, img_rgb[:, :, 0])
        img_rgb[:, :, 1] = np.where(skeleton > 0, 0, img_rgb[:, :, 1])
        img_rgb[:, :, 2] = np.where(skeleton > 0, 1, img_rgb[:, :, 2])

        return (np.clip(img_rgb * 255, 0, 255)).astype(np.uint8)

    def run_video(
        self,
        output_path: Optional[Path] = None,
        fps: int = 10,
        overlay_alpha: float = 0.4,
        show_side_by_side: bool = True,
        view_mode: ViewMode = "overlay",
        overlay_color: str = "green",
    ) -> Optional[Path]:
        """
        Create a video of segmentation over the dataset.
        Saves to output_path (default: segment_tracker_output.mp4).

        Args:
            output_path: Where to save the video.
            fps: Frames per second.
            overlay_alpha: Transparency of segmentation overlay (0–1).
            show_side_by_side: If True, show original | view; else view only.
            view_mode: "overlay" (filled), "triple" (orig|mask|overlay),
                "contour" (outline only), "skeleton" (centerline), "combined" (all overlaid).
            overlay_color: "green", "red", "cyan", "magenta", "yellow".

        Returns:
            Path to saved video, or None if no frames.
        """
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio is required for video export. Install with: pip install imageio[ffmpeg]"
            )

        frames_list = list(self._iter_frames())
        if not frames_list:
            print("No frames to render.")
            return None

        out_path = Path(output_path) if output_path else Path("segment_tracker_output.mp4")
        out_path = out_path.resolve()

        print(f"Rendering {len(frames_list)} frames ({view_mode}) to {out_path}...")

        def render_one(img: np.ndarray, pred: np.ndarray) -> np.ndarray:
            if view_mode == "overlay":
                return self._render_frame(img, pred, {}, overlay_alpha, overlay_color)
            if view_mode == "triple":
                return self._render_triple(img, pred, overlay_alpha, overlay_color)
            if view_mode == "contour":
                return self._render_contour(img, pred, overlay_color)
            if view_mode == "skeleton":
                return self._render_skeleton(img, pred, overlay_color)
            if view_mode == "combined":
                return self._render_combined(img, pred)
            return self._render_frame(img, pred, {}, overlay_alpha, overlay_color)

        def frame_generator():
            for info, img, pred in frames_list:
                rendered = render_one(img, pred)
                if show_side_by_side and view_mode != "triple":
                    orig_rgb = self._to_display_rgb(img)
                    combined = np.concatenate([orig_rgb, rendered], axis=1)
                else:
                    combined = rendered
                yield combined

        writer = imageio.get_writer(
            str(out_path),
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )
        for frame in frame_generator():
            writer.append_data(frame)
        writer.close()

        print(f"Video saved to {out_path}")
        return out_path

    def run_animation(
        self,
        overlay_alpha: float = 0.4,
        interval_ms: int = 100,
        repeat: bool = True,
    ) -> None:
        """
        Display an animated loop in the notebook (matplotlib FuncAnimation).
        Useful for quick preview without saving a file.
        """
        from matplotlib.animation import FuncAnimation

        frames_list = list(self._iter_frames())
        if not frames_list:
            print("No frames to animate.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_orig, ax_overlay = axes

        def animate(i):
            info, img, pred = frames_list[i % len(frames_list)]
            rendered = self._render_frame(img, pred, info, overlay_alpha)
            ax_orig.clear()
            ax_orig.imshow(img, cmap="gray")
            ax_orig.set_title(f"Original — {info.get('file_name', '')} f{info.get('frame', i)}")
            ax_orig.axis("off")
            ax_overlay.clear()
            ax_overlay.imshow(rendered)
            ax_overlay.set_title(f"Segmentation — frame {i + 1}/{len(frames_list)}")
            ax_overlay.axis("off")

        anim = FuncAnimation(
            fig,
            animate,
            frames=len(frames_list),
            interval=interval_ms,
            repeat=repeat,
        )
        plt.tight_layout()
        plt.show()
        return anim
