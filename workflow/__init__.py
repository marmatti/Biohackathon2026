"""
Workflow package for filament segmentation.
"""

from .data_loader import RawImageLoader
from .segment_tracker import SegmentTracker
from .segmentation import (
    OtsuSegmenter,
    AdaptiveThresholdSegmenter,
    MorphologicalSegmenter,
    WatershedSegmenter,
    TopHatSegmenter,
    CLAHEThresholdSegmenter,
    FrangiSegmenter,
    MeijeringSegmenter,
    SatoSegmenter,
)
from .evaluation import SegmentationEvaluator

# Lazy import for experiments (avoids Jupyter import issues)
def __getattr__(name):
    if name in ("SampleCollector", "SampleGridVisualizer", "SegmentationVisualizer",
                "ParameterScanExperiment", "AdaptiveParameterScan",
                "TubenessComparisonExperiment"):
        from . import experiments
        return getattr(experiments, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RawImageLoader",
    "SegmentTracker",
    "OtsuSegmenter",
    "AdaptiveThresholdSegmenter",
    "MorphologicalSegmenter",
    "WatershedSegmenter",
    "TopHatSegmenter",
    "CLAHEThresholdSegmenter",
    "FrangiSegmenter",
    "MeijeringSegmenter",
    "SatoSegmenter",
    "SegmentationEvaluator",
    "SampleCollector",
    "SampleGridVisualizer",
    "SegmentationVisualizer",
    "ParameterScanExperiment",
    "AdaptiveParameterScan",
    "TubenessComparisonExperiment",
]
