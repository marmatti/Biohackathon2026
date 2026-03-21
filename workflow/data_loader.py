"""
Data loading utilities for filament dataset.
Supports raw multi-frame TIFF files.
"""

from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
from PIL import Image


class RawImageLoader:
    """
    Loads raw microscopy images from a directory of multi-frame TIFF files.
    Each TIFF file typically contains ~900 frames (or fewer). Yields individual
    frames as grayscale arrays for segmentation.
    """

    def __init__(
        self,
        data_dir: str,
        extensions: tuple = (".tif", ".tiff"),
        normalize: bool = True,
    ):
        """
        Args:
            data_dir: Path to directory containing TIFF files.
            extensions: File extensions to load.
            normalize: If True, normalize 16-bit images to [0, 1] float.
        """
        self.data_dir = Path(data_dir)
        self.extensions = extensions
        self.normalize = normalize
        self._files = sorted(self._discover_files())

    def _discover_files(self) -> list:
        """Find all TIFF files in the data directory."""
        files = []
        for ext in self.extensions:
            files.extend(self.data_dir.glob(f"*{ext}"))
        return files

    @property
    def files(self) -> list:
        """List of discovered TIFF file paths."""
        return self._files

    def get_frame_count(self, path: Path) -> int:
        """Get number of frames in a multi-frame TIFF."""
        with Image.open(path) as img:
            n = 0
            try:
                while True:
                    img.seek(n)
                    n += 1
            except EOFError:
                pass
        return n

    def load_frame(self, path: Path, frame_idx: int) -> np.ndarray:
        """
        Load a single frame from a multi-frame TIFF.
        Returns float [0, 1] when normalize=True (compatible with all segmenters).
        """
        with Image.open(path) as img:
            img.seek(frame_idx)
            arr = np.array(img)
        if self.normalize and (
            arr.dtype in (np.uint16, np.int16)
            or (arr.dtype.kind in ("u", "i") and arr.itemsize == 2)
        ):
            arr = arr.astype(np.float64) / 65535.0
        elif self.normalize and (
            arr.dtype == np.uint8 or (arr.dtype.kind == "u" and arr.itemsize == 1)
        ):
            arr = arr.astype(np.float64) / 255.0
        return arr

    def iter_samples(
        self,
        max_frames_per_file: Optional[int] = None,
        max_total: Optional[int] = None,
    ) -> Iterator[Tuple[dict, np.ndarray, None]]:
        """
        Iterate over (img_info, image_array, None) for each frame.
        img_info contains: id, file_name, frame, height, width.

        Args:
            max_frames_per_file: Limit frames per file (e.g. 20 for EDA).
            max_total: Stop after this many frames total.
        """
        sample_id = 0
        for path in self._files:
            try:
                n_frames = self.get_frame_count(path)
            except Exception:
                continue
            limit = max_frames_per_file if max_frames_per_file else n_frames
            for frame_idx in range(min(n_frames, limit)):
                try:
                    img = self.load_frame(path, frame_idx)
                except Exception:
                    continue
                h, w = img.shape
                img_info = {
                    "id": sample_id,
                    "file_name": path.name,
                    "frame": frame_idx,
                    "height": h,
                    "width": w,
                }
                yield img_info, img, None
                sample_id += 1
                if max_total and sample_id >= max_total:
                    return

    def __len__(self) -> int:
        """Total number of frames across all files."""
        total = 0
        for path in self._files:
            try:
                total += self.get_frame_count(path)
            except Exception:
                pass
        return total
