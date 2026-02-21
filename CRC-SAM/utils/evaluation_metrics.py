# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Metrics for Medical Image Segmentation

Metrics included:
    - Dice Similarity Coefficient (DSC)
    - Intersection over Union (IoU / Jaccard Index)
    - Precision (Positive Predictive Value)
    - Recall (Sensitivity / True Positive Rate)
    - Specificity (True Negative Rate)
    - F1 Score
    - Accuracy
    - Hausdorff Distance (HD95)
    - Average Surface Distance (ASD)

Usage:
    from utils.evaluation_metrics import SegmentationEvaluator

    evaluator = SegmentationEvaluator()
    metrics = evaluator.compute_all_metrics(prediction, ground_truth)

    # Or for batch evaluation
    results = evaluator.evaluate_batch(predictions, ground_truths)
    evaluator.print_results(results)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings


class SegmentationEvaluator:
    """
    Comprehensive segmentation evaluator with multiple metrics.

    Attributes:
        include_surface_metrics: Whether to compute surface-based metrics (slower)
        spacing: Voxel spacing for surface distance computation
    """

    def __init__(
        self,
        include_surface_metrics: bool = False,
        spacing: Tuple[float, ...] = (1.0, 1.0),
    ):
        self.include_surface_metrics = include_surface_metrics
        self.spacing = spacing

    def compute_confusion_matrix(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
    ) -> Dict[str, int]:
        """Compute confusion matrix elements (TP, TN, FP, FN)."""
        pred_flat = pred.flatten().astype(bool)
        gt_flat = gt.flatten().astype(bool)

        TP = np.sum(pred_flat & gt_flat)
        TN = np.sum(~pred_flat & ~gt_flat)
        FP = np.sum(pred_flat & ~gt_flat)
        FN = np.sum(~pred_flat & gt_flat)

        return {"TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN)}

    def compute_dice(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute Dice Similarity Coefficient: DSC = 2|P∩G| / (|P|+|G|)."""
        intersection = np.sum(pred.astype(bool) & gt.astype(bool))
        pred_sum = np.sum(pred.astype(bool))
        gt_sum = np.sum(gt.astype(bool))

        if pred_sum + gt_sum == 0:
            return 1.0
        return (2 * intersection) / (pred_sum + gt_sum)

    def compute_iou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute Intersection over Union (Jaccard Index): IoU = |P∩G| / |P∪G|."""
        pred_bool = pred.astype(bool)
        gt_bool = gt.astype(bool)

        intersection = np.sum(pred_bool & gt_bool)
        union = np.sum(pred_bool | gt_bool)

        if union == 0:
            return 1.0
        return intersection / union

    def compute_precision(self, pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
        """Compute Precision: TP / (TP + FP)."""
        cm = self.compute_confusion_matrix(pred, gt)
        return cm["TP"] / (cm["TP"] + cm["FP"] + eps)

    def compute_recall(self, pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
        """Compute Recall (Sensitivity): TP / (TP + FN)."""
        cm = self.compute_confusion_matrix(pred, gt)
        return cm["TP"] / (cm["TP"] + cm["FN"] + eps)

    def compute_specificity(self, pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
        """Compute Specificity: TN / (TN + FP)."""
        cm = self.compute_confusion_matrix(pred, gt)
        return cm["TN"] / (cm["TN"] + cm["FP"] + eps)

    def compute_f1(self, pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
        """Compute F1 Score: 2*(Precision*Recall)/(Precision+Recall)."""
        precision = self.compute_precision(pred, gt, eps)
        recall = self.compute_recall(pred, gt, eps)
        return (2 * precision * recall) / (precision + recall + eps)

    def compute_accuracy(self, pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
        """Compute Accuracy: (TP + TN) / (TP + TN + FP + FN)."""
        cm = self.compute_confusion_matrix(pred, gt)
        total = cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"]
        return (cm["TP"] + cm["TN"]) / (total + eps)

    def compute_hausdorff_distance(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        percentile: float = 95,
    ) -> float:
        """Compute Hausdorff Distance at given percentile (default HD95)."""
        try:
            from scipy.ndimage import distance_transform_edt, binary_erosion
        except ImportError:
            warnings.warn("scipy not available for Hausdorff distance")
            return float('nan')

        pred_bool = pred.astype(bool)
        gt_bool = gt.astype(bool)

        if not np.any(pred_bool) and not np.any(gt_bool):
            return 0.0
        if not np.any(pred_bool) or not np.any(gt_bool):
            return float('inf')

        pred_surface = pred_bool ^ binary_erosion(pred_bool)
        gt_surface = gt_bool ^ binary_erosion(gt_bool)

        pred_dist = distance_transform_edt(~gt_bool, sampling=self.spacing)
        gt_dist = distance_transform_edt(~pred_bool, sampling=self.spacing)

        pred_to_gt = pred_dist[pred_surface]
        gt_to_pred = gt_dist[gt_surface]

        if len(pred_to_gt) == 0 or len(gt_to_pred) == 0:
            return float('inf')

        all_distances = np.concatenate([pred_to_gt, gt_to_pred])
        return float(np.percentile(all_distances, percentile))

    def compute_average_surface_distance(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
    ) -> float:
        """Compute Average Surface Distance (ASD)."""
        try:
            from scipy.ndimage import distance_transform_edt, binary_erosion
        except ImportError:
            warnings.warn("scipy not available for ASD")
            return float('nan')

        pred_bool = pred.astype(bool)
        gt_bool = gt.astype(bool)

        if not np.any(pred_bool) and not np.any(gt_bool):
            return 0.0
        if not np.any(pred_bool) or not np.any(gt_bool):
            return float('inf')

        pred_surface = pred_bool ^ binary_erosion(pred_bool)
        gt_surface = gt_bool ^ binary_erosion(gt_bool)

        pred_dist = distance_transform_edt(~gt_bool, sampling=self.spacing)
        gt_dist = distance_transform_edt(~pred_bool, sampling=self.spacing)

        pred_to_gt = pred_dist[pred_surface]
        gt_to_pred = gt_dist[gt_surface]

        if len(pred_to_gt) == 0 or len(gt_to_pred) == 0:
            return float('inf')

        return float((np.mean(pred_to_gt) + np.mean(gt_to_pred)) / 2)

    def compute_all_metrics(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all segmentation metrics."""
        metrics = {
            "dice": self.compute_dice(pred, gt),
            "iou": self.compute_iou(pred, gt),
            "precision": self.compute_precision(pred, gt),
            "recall": self.compute_recall(pred, gt),
            "specificity": self.compute_specificity(pred, gt),
            "f1": self.compute_f1(pred, gt),
            "accuracy": self.compute_accuracy(pred, gt),
        }

        cm = self.compute_confusion_matrix(pred, gt)
        metrics.update(cm)

        if self.include_surface_metrics:
            metrics["hd95"] = self.compute_hausdorff_distance(pred, gt, 95)
            metrics["asd"] = self.compute_average_surface_distance(pred, gt)

        return metrics

    def evaluate_batch(
        self,
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray],
        sample_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Evaluate a batch of predictions.

        Returns dict with 'per_sample' and 'aggregated' results.
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions and ground truths must match")

        if sample_names is None:
            sample_names = [f"sample_{i}" for i in range(len(predictions))]

        per_sample = []
        all_metrics = {}

        for pred, gt, name in zip(predictions, ground_truths, sample_names):
            metrics = self.compute_all_metrics(pred, gt)
            per_sample.append({"name": name, **metrics})

            for key, value in metrics.items():
                if key not in ["TP", "TN", "FP", "FN"]:
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

        # Aggregated statistics
        aggregated = {"num_samples": len(predictions)}

        for key, values in all_metrics.items():
            values = np.array(values)
            valid_values = values[np.isfinite(values)]
            if len(valid_values) > 0:
                aggregated[f"mean_{key}"] = float(np.mean(valid_values))
                aggregated[f"std_{key}"] = float(np.std(valid_values))
                aggregated[f"median_{key}"] = float(np.median(valid_values))
                aggregated[f"min_{key}"] = float(np.min(valid_values))
                aggregated[f"max_{key}"] = float(np.max(valid_values))
            else:
                for stat in ["mean", "std", "median", "min", "max"]:
                    aggregated[f"{stat}_{key}"] = float('nan')

        # Precision-recall correlation
        precision_vals = np.array(all_metrics.get("precision", []))
        recall_vals = np.array(all_metrics.get("recall", []))

        if len(precision_vals) > 1:
            valid_mask = np.isfinite(precision_vals) & np.isfinite(recall_vals)
            if np.sum(valid_mask) > 1:
                aggregated["precision_recall_correlation"] = float(
                    np.corrcoef(precision_vals[valid_mask], recall_vals[valid_mask])[0, 1]
                )
            else:
                aggregated["precision_recall_correlation"] = float('nan')

        return {
            "per_sample": per_sample,
            "aggregated": aggregated,
        }

    def print_results(self, results: Dict, show_per_sample: bool = False):
        """Pretty print evaluation results."""
        agg = results["aggregated"]

        print("\n" + "=" * 70)
        print("COMPREHENSIVE SEGMENTATION EVALUATION RESULTS")
        print("=" * 70)
        print(f"Number of samples: {agg['num_samples']}")
        print("-" * 70)

        metric_display = [
            ("Dice (DSC)", "dice"),
            ("IoU (Jaccard)", "iou"),
            ("Precision", "precision"),
            ("Recall (Sensitivity)", "recall"),
            ("Specificity", "specificity"),
            ("F1 Score", "f1"),
            ("Accuracy", "accuracy"),
        ]

        if self.include_surface_metrics:
            metric_display.extend([
                ("HD95", "hd95"),
                ("ASD", "asd"),
            ])

        print(f"{'Metric':<25} {'Mean +/- Std':<20} {'Median':<10} {'Range':<20}")
        print("-" * 70)

        for display_name, key in metric_display:
            mean_val = agg.get(f"mean_{key}", float('nan'))
            std_val = agg.get(f"std_{key}", float('nan'))
            median_val = agg.get(f"median_{key}", float('nan'))
            min_val = agg.get(f"min_{key}", float('nan'))
            max_val = agg.get(f"max_{key}", float('nan'))

            if np.isfinite(mean_val):
                print(f"{display_name:<25} {mean_val:.4f} +/- {std_val:.4f}    "
                      f"{median_val:.4f}     [{min_val:.4f}, {max_val:.4f}]")
            else:
                print(f"{display_name:<25} N/A")

        print("=" * 70)

        pr_corr = agg.get("precision_recall_correlation", float('nan'))
        if np.isfinite(pr_corr):
            print(f"Precision-Recall Correlation: {pr_corr:.4f}")

        if show_per_sample:
            print("\n" + "-" * 70)
            print("PER-SAMPLE RESULTS")
            print("-" * 70)
            for sample in results["per_sample"]:
                print(f"\n{sample['name']}:")
                print(f"  Dice: {sample['dice']:.4f}, IoU: {sample['iou']:.4f}")
                print(f"  Precision: {sample['precision']:.4f}, Recall: {sample['recall']:.4f}")

    def save_results(self, results: Dict, output_path: str, format: str = "json"):
        """Save evaluation results to file."""
        import os
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        if format == "json":
            import json

            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_to_serializable(v) for v in obj]
                return obj

            with open(output_path, "w") as f:
                json.dump(convert_to_serializable(results), f, indent=2)

        elif format == "csv":
            import csv
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                if results["per_sample"]:
                    header = list(results["per_sample"][0].keys())
                    writer.writerow(header)
                    for sample in results["per_sample"]:
                        writer.writerow([sample.get(h, "") for h in header])

        print(f"Results saved to: {output_path}")


# Convenience functions
def compute_metrics_quick(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Quick function to compute all metrics without creating an evaluator."""
    evaluator = SegmentationEvaluator(include_surface_metrics=False)
    return evaluator.compute_all_metrics(pred, gt)


def compute_metrics_with_surface(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0),
) -> Dict[str, float]:
    """Compute all metrics including surface-based metrics."""
    evaluator = SegmentationEvaluator(include_surface_metrics=True, spacing=spacing)
    return evaluator.compute_all_metrics(pred, gt)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice Similarity Coefficient."""
    return SegmentationEvaluator().compute_dice(pred, gt)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Intersection over Union."""
    return SegmentationEvaluator().compute_iou(pred, gt)


def precision_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Precision."""
    return SegmentationEvaluator().compute_precision(pred, gt)


def recall_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Recall (Sensitivity)."""
    return SegmentationEvaluator().compute_recall(pred, gt)
