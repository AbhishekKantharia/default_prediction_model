"""
Model Evaluation Module
Comprehensive evaluation of default prediction models with threshold
optimization, calibration analysis, and loan-type-specific metrics.
"""
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    brier_score_loss, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates default prediction models across multiple metrics."""

    def __init__(self, config: dict):
        self.config = config
        self.metrics_history: List[Dict] = []
        self.optimal_thresholds: Dict[str, float] = {}

    def compute_all_metrics(self, y_true: np.ndarray,
                            y_prob: np.ndarray,
                            model_name: str = "model") -> Dict[str, float]:
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "model_name": model_name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_true, y_prob),
            "auc_pr": average_precision_score(y_true, y_prob),
            "log_loss": log_loss(y_true, y_prob),
            "brier_score": brier_score_loss(y_true, y_prob),
            "true_positives": int(((y_pred == 1) & (y_true == 1)).sum()),
            "true_negatives": int(((y_pred == 0) & (y_true == 0)).sum()),
            "false_positives": int(((y_pred == 1) & (y_true == 0)).sum()),
            "false_negatives": int(((y_pred == 0) & (y_true == 1)).sum()),
            "total_samples": len(y_true),
            "default_rate": y_true.mean()
        }

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0
            metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0

        self.metrics_history.append(metrics)
        logger.info(f"Metrics computed for {model_name}: "
                    f"AUC-ROC={metrics['auc_roc']:.4f}, "
                    f"F1={metrics['f1_score']:.4f}")
        return metrics

    def optimize_threshold(self, y_true: np.ndarray,
                           y_prob: np.ndarray,
                           metric: str = "f1_score") -> float:
        thresholds = np.arange(0.05, 0.95, 0.01)
        best_score = 0
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            if metric == "f1_score":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == "balanced_accuracy":
                score = (recall_score(y_true, y_pred, zero_division=0) +
                         (y_true == 0).mean()) / 2
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        logger.info(f"Optimal threshold for {metric}: {best_threshold:.2f} "
                    f"(score: {best_score:.4f})")
        return best_threshold

    def find_optimal_thresholds(self, y_true: np.ndarray,
                                y_prob: np.ndarray) -> Dict[str, float]:
        thresholds = {}
        for metric in ["f1_score", "precision", "recall", "balanced_accuracy"]:
            thresholds[metric] = self.optimize_threshold(
                y_true, y_prob, metric)
        self.optimal_thresholds = thresholds
        return thresholds

    def calibration_analysis(self, y_true: np.ndarray,
                             y_prob: np.ndarray,
                             n_bins: int = 10) -> pd.DataFrame:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        calibration = []
        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if mask.sum() > 0:
                mean_pred = y_prob[mask].mean()
                mean_true = y_true[mask].mean()
                calibration.append({
                    "bin_center": bin_centers[i],
                    "mean_predicted": mean_pred,
                    "mean_observed": mean_true,
                    "count": mask.sum(),
                    "gap": abs(mean_pred - mean_true)
                })

        df = pd.DataFrame(calibration)
        if len(df) > 0:
            ece = (df["count"] * df["gap"]).sum() / df["count"].sum()
            logger.info(f"Expected Calibration Error: {ece:.4f}")
        return df

    def compute_lift_chart(self, y_true: np.ndarray,
                           y_prob: np.ndarray,
                           n_groups: int = 10) -> pd.DataFrame:
        order = np.argsort(-y_prob)
        y_true_sorted = y_true[order]
        n_per_group = len(y_true) // n_groups

        lift_data = []
        baseline_rate = y_true.mean()

        for i in range(n_groups):
            start = i * n_per_group
            end = start + n_per_group
            group = y_true_sorted[start:end]
            group_rate = group.mean()
            lift = group_rate / baseline_rate if baseline_rate > 0 else 0
            cumulative_rate = y_true_sorted[:end].mean()

            lift_data.append({
                "decile": i + 1,
                "default_rate": group_rate,
                "lift": lift,
                "cumulative_default_rate": cumulative_rate,
                "count": len(group)
            })

        return pd.DataFrame(lift_data)

    def evaluate_by_segment(self, y_true: np.ndarray,
                            y_prob: np.ndarray,
                            segments: pd.Series) -> pd.DataFrame:
        results = []
        for segment in segments.unique():
            mask = segments == segment
            if mask.sum() > 10:
                metrics = self.compute_all_metrics(
                    y_true[mask], y_prob[mask], model_name=str(segment))
                metrics["segment"] = segment
                metrics["segment_size"] = mask.sum()
                results.append(metrics)
        return pd.DataFrame(results)

    def evaluate_by_loan_type(self, y_true: np.ndarray,
                              y_prob: np.ndarray,
                              loan_types: pd.Series) -> pd.DataFrame:
        return self.evaluate_by_segment(y_true, y_prob, loan_types)

    def generate_evaluation_report(self, y_true: np.ndarray,
                                   y_prob: np.ndarray,
                                   model_name: str = "model") -> Dict:
        metrics = self.compute_all_metrics(y_true, y_prob, model_name)
        thresholds = self.find_optimal_thresholds(y_true, y_prob)
        calibration = self.calibration_analysis(y_true, y_prob)
        lift = self.compute_lift_chart(y_true, y_prob)

        report = {
            "model_name": model_name,
            "overall_metrics": metrics,
            "optimal_thresholds": thresholds,
            "calibration_summary": {
                "mean_gap": calibration["gap"].mean() if len(calibration) > 0 else 0,
                "max_gap": calibration["gap"].max() if len(calibration) > 0 else 0,
            },
            "lift_summary": {
                "top_decile_lift": lift.iloc[0]["lift"] if len(lift) > 0 else 0,
                "top_decile_capture": lift.iloc[0]["cumulative_default_rate"] if len(lift) > 0 else 0,
            },
            "confusion_matrix": {
                "TP": metrics.get("true_positives", 0),
                "TN": metrics.get("true_negatives", 0),
                "FP": metrics.get("false_positives", 0),
                "FN": metrics.get("false_negatives", 0),
            }
        }

        logger.info(f"Evaluation report generated for {model_name}")
        return report
