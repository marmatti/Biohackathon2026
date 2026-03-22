"""
Evaluation metrics for benchmarking segmentation against human labels.
"""

from typing import Dict, List, Optional

import numpy as np


class SegmentationEvaluator:
    """
    Computes metrics comparing predicted masks to ground truth.
    """

    @staticmethod
    def _ensure_binary(mask: np.ndarray) -> np.ndarray:
        """Ensure mask is binary 0/1."""
        m = np.asarray(mask)
        return (m > 0).astype(np.uint8)

    @staticmethod
    def iou(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Intersection over Union (Jaccard index).
        Range [0, 1], higher is better.
        """
        pred = SegmentationEvaluator._ensure_binary(pred)
        gt = SegmentationEvaluator._ensure_binary(gt)
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return float(intersection) / float(union)

    @staticmethod
    def dice(pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Dice coefficient (F1 for binary segmentation).
        2 * |A ∩ B| / (|A| + |B|)
        Range [0, 1], higher is better.
        """
        pred = SegmentationEvaluator._ensure_binary(pred)
        gt = SegmentationEvaluator._ensure_binary(gt)
        intersection = np.logical_and(pred, gt).sum()
        total = pred.sum() + gt.sum()
        if total == 0:
            return 1.0
        return 2.0 * intersection / total

    @staticmethod
    def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple:
        """Returns (precision, recall)."""
        pred = SegmentationEvaluator._ensure_binary(pred)
        gt = SegmentationEvaluator._ensure_binary(gt)
        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, 1 - gt).sum()
        fn = np.logical_and(1 - pred, gt).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return float(precision), float(recall)

    def evaluate(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single image.
        """
        iou_val = self.iou(pred, gt)
        dice_val = self.dice(pred, gt)
        prec, rec = self.precision_recall(pred, gt)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {
            "iou": iou_val,
            "dice": dice_val,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    def evaluate_batch(
        self,
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Aggregate metrics over multiple images (mean).
        """
        if not predictions or not ground_truths:
            return {}
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have same length")
        results = [self.evaluate(p, g) for p, g in zip(predictions, ground_truths)]
        return {
            k: np.mean([r[k] for r in results])
            for k in results[0].keys()
        }
