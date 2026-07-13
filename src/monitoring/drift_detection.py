"""
Concept Drift Detection and Automatic Retraining Triggers

Monitors data drift, concept drift, and prediction distribution shifts.
Triggers automatic retraining when degradation is detected.
"""
import logging
import os
import json
import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, log_loss, brier_score_loss
)

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift, concept drift, and prediction distribution shifts."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = self._load_config(config_path)
        self.alert_threshold = self.config.get("monitoring", {}).get(
            "alert_threshold", 0.05
        )
        self.psi_significant = 0.2
        self.psi_moderate = 0.1
        self.retrain_history: List[Dict] = []

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        try:
            import yaml
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Config not found at %s, using defaults", config_path)
            return {"monitoring": {"alert_threshold": 0.05}}

    @staticmethod
    def _compute_psi(reference: np.ndarray, current: np.ndarray,
                     n_bins: int = 10) -> float:
        ref_counts, bin_edges = np.histogram(reference, bins=n_bins, density=True)
        cur_counts, _ = np.histogram(current, bins=bin_edges, density=True)
        ref_pct = np.clip(ref_counts / (ref_counts.sum() + 1e-10), 1e-6, None)
        cur_pct = np.clip(cur_counts / (cur_counts.sum() + 1e-10), 1e-6, None)
        psi_values = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
        return float(np.sum(psi_values))

    @staticmethod
    def _classify_psi(psi: float) -> str:
        if psi >= 0.2:
            return "significant"
        elif psi >= 0.1:
            return "moderate"
        return "stable"

    def detect_data_drift(self, reference_data: pd.DataFrame,
                          current_data: pd.DataFrame,
                          threshold: float = 0.05) -> Dict[str, Any]:
        """
        Detect data drift between reference and current datasets.

        Uses PSI for numeric features and Chi-squared test for categorical
        features. Returns a report with drifted features, PSI values, and
        p-values.

        Args:
            reference_data: Baseline dataset.
            current_data: New dataset to compare against.
            threshold: Significance level for statistical tests.

        Returns:
            Dictionary with drift report per feature and summary.
        """
        report: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "reference_shape": list(reference_data.shape),
            "current_shape": list(current_data.shape),
            "numeric_features": {},
            "categorical_features": {},
            "drifted_features": [],
            "overall_drift_detected": False,
        }

        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
        categorical_cols = reference_data.select_dtypes(
            include=["object", "category"]
        ).columns

        for col in numeric_cols:
            if col not in current_data.columns:
                continue
            ref = reference_data[col].dropna().values
            cur = current_data[col].dropna().values
            if len(ref) == 0 or len(cur) == 0:
                continue

            psi = self._compute_psi(ref, cur)
            ks_stat, ks_p = ks_2samp(ref, cur)
            classification = self._classify_psi(psi)

            report["numeric_features"][col] = {
                "psi": round(psi, 6),
                "classification": classification,
                "ks_statistic": round(ks_stat, 6),
                "ks_p_value": round(ks_p, 6),
                "ref_mean": round(float(np.mean(ref)), 6),
                "cur_mean": round(float(np.mean(cur)), 6),
                "ref_std": round(float(np.std(ref)), 6),
                "cur_std": round(float(np.std(cur)), 6),
                "drifted": classification != "stable" or ks_p < threshold,
            }
            if classification != "stable" or ks_p < threshold:
                report["drifted_features"].append(col)

        for col in categorical_cols:
            if col not in current_data.columns:
                continue
            ref_counts = reference_data[col].fillna("__MISSING__").value_counts()
            cur_counts = current_data[col].fillna("__MISSING__").value_counts()
            all_cats = list(set(ref_counts.index) | set(cur_counts.index))
            ref_arr = np.array([ref_counts.get(c, 0) for c in all_cats])
            cur_arr = np.array([cur_counts.get(c, 0) for c in all_cats])

            if ref_arr.sum() == 0 or cur_arr.sum() == 0:
                continue

            ref_pct = ref_arr / ref_arr.sum()
            cur_pct = cur_arr / cur_arr.sum()
            psi = self._compute_psi(
                np.repeat(np.arange(len(all_cats)), ref_arr.astype(int)),
                np.repeat(np.arange(len(all_cats)), cur_arr.astype(int)),
            )

            try:
                contingency = np.array([ref_arr, cur_arr])
                chi2, chi2_p, _, _ = chi2_contingency(contingency)
            except ValueError:
                chi2, chi2_p = 0.0, 1.0

            classification = self._classify_psi(psi)
            report["categorical_features"][col] = {
                "psi": round(psi, 6),
                "classification": classification,
                "chi2_statistic": round(chi2, 6),
                "chi2_p_value": round(chi2_p, 6),
                "n_categories": len(all_cats),
                "drifted": classification != "stable" or chi2_p < threshold,
            }
            if classification != "stable" or chi2_p < threshold:
                report["drifted_features"].append(col)

        report["overall_drift_detected"] = len(report["drifted_features"]) > 0
        report["drift_fraction"] = (
            len(report["drifted_features"])
            / max(len(numeric_cols) + len(categorical_cols), 1)
        )
        logger.info(
            "Data drift: %d/%d features drifted",
            len(report["drifted_features"]),
            len(numeric_cols) + len(categorical_cols),
        )
        return report

    def detect_concept_drift(
        self,
        reference_predictions: np.ndarray,
        reference_labels: np.ndarray,
        current_predictions: np.ndarray,
        current_labels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Detect concept drift using Page-Hinkley, DDM, and ADWIN-like tests.

        Compares model performance degradation between reference and current
        periods.

        Args:
            reference_predictions: Model predictions on reference data.
            reference_labels: True labels for reference data.
            current_predictions: Model predictions on current data.
            current_labels: True labels for current data.

        Returns:
            Drift report with type, severity, and affected segments.
        """
        report: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "reference_performance": {},
            "current_performance": {},
            "page_hinkley": {},
            "ddm": {},
            "adwin_like": {},
            "drift_detected": False,
            "drift_type": "none",
            "severity": "low",
        }

        ref_auc = roc_auc_score(reference_labels, reference_predictions)
        cur_auc = roc_auc_score(current_labels, current_predictions)
        ref_f1 = f1_score(
            reference_labels, (reference_predictions >= 0.5).astype(int)
        )
        cur_f1 = f1_score(
            current_labels, (current_predictions >= 0.5).astype(int)
        )
        ref_loss = brier_score_loss(reference_labels, reference_predictions)
        cur_loss = brier_score_loss(current_labels, current_predictions)

        report["reference_performance"] = {
            "auc_roc": round(ref_auc, 6),
            "f1_score": round(ref_f1, 6),
            "brier_score": round(ref_loss, 6),
            "n_samples": len(reference_labels),
        }
        report["current_performance"] = {
            "auc_roc": round(cur_auc, 6),
            "f1_score": round(cur_f1, 6),
            "brier_score": round(cur_loss, 6),
            "n_samples": len(current_labels),
        }

        auc_degradation = ref_auc - cur_auc
        f1_degradation = ref_f1 - cur_f1
        loss_increase = cur_loss - ref_loss

        page_hinkley_result = self._page_hinkley_test(
            reference_labels, reference_predictions,
            current_labels, current_predictions,
        )
        report["page_hinkley"] = page_hinkley_result

        ddm_result = self._ddm_test(
            reference_labels, reference_predictions,
            current_labels, current_predictions,
        )
        report["ddm"] = ddm_result

        adwin_result = self._adwin_like_test(
            reference_labels, reference_predictions,
            current_labels, current_predictions,
        )
        report["adwin_like"] = adwin_result

        drift_signals = sum([
            page_hinkley_result.get("drift_detected", False),
            ddm_result.get("drift_detected", False),
            adwin_result.get("drift_detected", False),
        ])

        report["drift_detected"] = drift_signals >= 2 or auc_degradation > 0.05
        report["auc_degradation"] = round(auc_degradation, 6)
        report["f1_degradation"] = round(f1_degradation, 6)
        report["loss_increase"] = round(loss_increase, 6)

        if auc_degradation > 0.1 or drift_signals >= 2:
            report["severity"] = "high"
            report["drift_type"] = "concept_drift"
        elif auc_degradation > 0.05 or drift_signals >= 1:
            report["severity"] = "moderate"
            report["drift_type"] = "gradual_drift"
        elif auc_degradation > 0.02:
            report["severity"] = "low"
            report["drift_type"] = "minor_drift"
        else:
            report["severity"] = "none"
            report["drift_type"] = "stable"

        report["affected_segments"] = self._identify_affected_segments(
            reference_predictions, reference_labels,
            current_predictions, current_labels,
        )

        logger.info(
            "Concept drift: type=%s, severity=%s, AUC degradation=%.4f",
            report["drift_type"],
            report["severity"],
            auc_degradation,
        )
        return report

    def _page_hinkley_test(
        self,
        ref_labels: np.ndarray,
        ref_preds: np.ndarray,
        cur_labels: np.ndarray,
        cur_preds: np.ndarray,
        delta: float = 0.005,
        threshold: float = 50.0,
    ) -> Dict[str, Any]:
        errors = np.concatenate([
            np.abs(ref_labels - ref_preds),
            np.abs(cur_labels - cur_preds),
        ])
        n_ref = len(ref_labels)
        running_sum = 0.0
        min_sum = float("inf")
        max_sum = float("-inf")
        drift_point = -1

        mean_error = errors.mean()
        for i, err in enumerate(errors):
            running_sum += err - mean_error - delta
            min_sum = min(min_sum, running_sum)
            if running_sum - min_sum > threshold:
                drift_point = i
                break

        drift_detected = drift_point >= n_ref
        return {
            "drift_detected": drift_detected,
            "drift_point": int(drift_point) if drift_point >= 0 else None,
            "page_hinkley_statistic": round(running_sum, 6),
            "threshold": threshold,
        }

    def _ddm_test(
        self,
        ref_labels: np.ndarray,
        ref_preds: np.ndarray,
        cur_labels: np.ndarray,
        cur_preds: np.ndarray,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
    ) -> Dict[str, Any]:
        all_labels = np.concatenate([ref_labels, cur_labels])
        all_preds = np.concatenate([ref_preds, cur_preds])
        all_errors = (all_preds >= 0.5).astype(int) != all_labels.astype(int)

        n_ref = len(ref_labels)
        min_samples = max(n_ref, 30)

        p = 0.0
        s = 0.0
        n = 0
        p_min = float("inf")
        s_min = float("inf")
        drift_point = -1
        warning_point = -1

        for i, err in enumerate(all_errors):
            n += 1
            p += (int(err) - p) / n
            s = np.sqrt(p * (1.0 - p) / max(n, 1))

            if n <= min_samples:
                continue

            if p + s < p_min + s_min:
                p_min = p
                s_min = s

            if p + s > p_min + drift_level * s_min and drift_point < 0:
                drift_point = i
            elif p + s > p_min + warning_level * s_min and warning_point < 0:
                warning_point = i

        drift_detected = drift_point >= n_ref
        return {
            "drift_detected": drift_detected,
            "warning_detected": warning_point >= n_ref,
            "drift_point": int(drift_point) if drift_point >= 0 else None,
            "warning_point": int(warning_point) if warning_point >= 0 else None,
            "final_error_rate": round(p, 6),
            "min_error_rate": round(p_min, 6),
        }

    def _adwin_like_test(
        self,
        ref_labels: np.ndarray,
        ref_preds: np.ndarray,
        cur_labels: np.ndarray,
        cur_preds: np.ndarray,
        min_window_size: int = 30,
        delta: float = 0.002,
    ) -> Dict[str, Any]:
        all_errors = np.concatenate([
            np.abs(ref_labels - ref_preds),
            np.abs(cur_labels - cur_preds),
        ])
        n = len(all_errors)
        n_ref = len(ref_labels)
        window = list(all_errors[:min_window_size])
        drift_detected = False
        drift_point = -1

        for i in range(min_window_size, n):
            window.append(all_errors[i])
            mean_w = np.mean(window)
            size_w = len(window)

            split = size_w // 2
            if split < min_window_size:
                continue

            left = np.mean(window[:split])
            right = np.mean(window[split:])
            epsilon = np.sqrt(
                (1.0 / (2.0 * split) + 1.0 / (2.0 * (size_w - split)))
                * 2.0 * np.log(2.0 * n / delta)
            )

            if abs(left - right) >= epsilon:
                drift_detected = True
                drift_point = i
                window = window[split:]
                break

        result = {
            "drift_detected": drift_detected and drift_point >= n_ref,
            "drift_point": int(drift_point) if drift_point >= 0 else None,
            "window_size": len(window),
            "mean_error": round(float(np.mean(window)), 6),
        }
        return result

    def _identify_affected_segments(
        self,
        ref_preds: np.ndarray,
        ref_labels: np.ndarray,
        cur_preds: np.ndarray,
        cur_labels: np.ndarray,
        n_quantiles: int = 4,
    ) -> List[Dict[str, Any]]:
        segments = []
        if len(ref_preds) < n_quantiles * 10:
            return segments

        try:
            bins = np.percentile(
                ref_preds, np.linspace(0, 100, n_quantiles + 1)
            )
        except ValueError:
            return segments

        for i in range(n_quantiles):
            lo, hi = bins[i], bins[i + 1]
            ref_mask = (ref_preds >= lo) & (ref_preds <= hi)
            cur_mask = (cur_preds >= lo) & (cur_preds <= hi)

            if ref_mask.sum() < 5 or cur_mask.sum() < 5:
                continue

            try:
                ref_auc = roc_auc_score(ref_labels[ref_mask], ref_preds[ref_mask])
                cur_auc = roc_auc_score(cur_labels[cur_mask], cur_preds[cur_mask])
                degradation = ref_auc - cur_auc
                segments.append({
                    "segment": f"quantile_{i + 1}",
                    "probability_range": [round(float(lo), 4), round(float(hi), 4)],
                    "reference_auc": round(ref_auc, 6),
                    "current_auc": round(cur_auc, 6),
                    "degradation": round(degradation, 6),
                    "significantly_degraded": degradation > 0.05,
                })
            except ValueError:
                continue

        return segments

    def detect_prediction_drift(
        self,
        reference_probs: np.ndarray,
        current_probs: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Detect distribution shift in model predictions using KS test,
        Wasserstein distance, and Jensen-Shannon divergence.

        Args:
            reference_probs: Prediction probabilities from reference period.
            current_probs: Prediction probabilities from current period.

        Returns:
            Statistical test results and drift indicators.
        """
        ks_stat, ks_p = ks_2samp(reference_probs, current_probs)
        w_dist = wasserstein_distance(reference_probs, current_probs)
        js_div = float(jensenshannon(reference_probs, current_probs) ** 2)

        ref_hist, bin_edges = np.histogram(reference_probs, bins=20, density=True)
        cur_hist, _ = np.histogram(current_probs, bins=bin_edges, density=True)
        ref_hist = np.clip(ref_hist, 1e-10, None)
        cur_hist = np.clip(cur_hist, 1e-10, None)
        ref_hist /= ref_hist.sum()
        cur_hist /= cur_hist.sum()
        kl_div = float(np.sum(ref_hist * np.log(ref_hist / cur_hist)))

        report: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "ks_test": {
                "statistic": round(ks_stat, 6),
                "p_value": round(ks_p, 6),
                "drift_detected": ks_p < self.alert_threshold,
            },
            "wasserstein_distance": {
                "value": round(w_dist, 6),
                "interpretation": (
                    "significant" if w_dist > 0.1
                    else "moderate" if w_dist > 0.05
                    else "stable"
                ),
            },
            "jensen_shannon_divergence": {
                "value": round(js_div, 6),
                "interpretation": (
                    "significant" if js_div > 0.05
                    else "moderate" if js_div > 0.01
                    else "stable"
                ),
            },
            "kl_divergence": round(kl_div, 6),
            "reference_mean": round(float(reference_probs.mean()), 6),
            "current_mean": round(float(current_probs.mean()), 6),
            "reference_std": round(float(reference_probs.std()), 6),
            "current_std": round(float(current_probs.std()), 6),
        }

        drift_count = sum([
            report["ks_test"]["drift_detected"],
            report["wasserstein_distance"]["interpretation"] == "significant",
            report["jensen_shannon_divergence"]["interpretation"] == "significant",
        ])
        report["overall_drift_detected"] = drift_count >= 2
        report["drift_severity"] = (
            "high" if drift_count >= 3
            else "moderate" if drift_count >= 2
            else "low" if drift_count >= 1
            else "none"
        )

        logger.info(
            "Prediction drift: KS p=%.4f, Wasserstein=%.4f, JS=%.6f",
            ks_p, w_dist, js_div,
        )
        return report

    def compute_feature_importance_drift(
        self,
        reference_importances: Dict[str, float],
        current_importances: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Detect changes in feature importance rankings and magnitudes.

        Args:
            reference_importances: Feature importances from the reference model.
            current_importances: Feature importances from the current model.

        Returns:
            Report on importance shifts and re-ranked features.
        """
        all_features = sorted(
            set(reference_importances.keys()) | set(current_importances.keys())
        )
        ref_values = np.array([
            reference_importances.get(f, 0.0) for f in all_features
        ])
        cur_values = np.array([
            current_importances.get(f, 0.0) for f in all_features
        ])

        ref_total = ref_values.sum()
        cur_total = cur_values.sum()
        if ref_total > 0:
            ref_values = ref_values / ref_total
        if cur_total > 0:
            cur_values = cur_values / cur_total

        abs_change = np.abs(cur_values - ref_values)
        relative_change = np.where(
            ref_values > 1e-8, abs_change / ref_values, 0.0
        )

        changes = []
        for i, feat in enumerate(all_features):
            changes.append({
                "feature": feat,
                "reference_importance": round(float(ref_values[i]), 6),
                "current_importance": round(float(cur_values[i]), 6),
                "absolute_change": round(float(abs_change[i]), 6),
                "relative_change": round(float(relative_change[i]), 6),
            })

        changes.sort(key=lambda x: x["absolute_change"], reverse=True)

        ref_rank = {f: i + 1 for i, f in enumerate(
            sorted(all_features, key=lambda x: reference_importances.get(x, 0),
                   reverse=True)
        )}
        cur_rank = {f: i + 1 for i, f in enumerate(
            sorted(all_features, key=lambda x: current_importances.get(x, 0),
                   reverse=True)
        )}

        reranked = [
            {
                "feature": feat,
                "reference_rank": ref_rank.get(feat, len(all_features)),
                "current_rank": cur_rank.get(feat, len(all_features)),
                "rank_change": ref_rank.get(feat, len(all_features))
                - cur_rank.get(feat, len(all_features)),
            }
            for feat in all_features
            if abs(
                ref_rank.get(feat, len(all_features))
                - cur_rank.get(feat, len(all_features))
            ) > 0
        ]
        reranked.sort(key=lambda x: abs(x["rank_change"]), reverse=True)

        spearman_corr, spearman_p = stats.spearmanr(ref_values, cur_values)

        report: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_features": len(all_features),
            "spearman_correlation": {
                "statistic": round(float(spearman_corr), 6),
                "p_value": round(float(spearman_p), 6),
            },
            "max_absolute_change": round(float(abs_change.max()), 6),
            "mean_absolute_change": round(float(abs_change.mean()), 6),
            "top_changed_features": changes[:10],
            "reranked_features": reranked[:10],
            "importance_drift_detected": float(abs_change.max()) > 0.1,
        }

        logger.info(
            "Feature importance drift: max_change=%.4f, spearman=%.4f",
            abs_change.max(), spearman_corr,
        )
        return report

    def generate_drift_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        reference_labels: Optional[np.ndarray] = None,
        reference_predictions: Optional[np.ndarray] = None,
        current_labels: Optional[np.ndarray] = None,
        current_predictions: Optional[np.ndarray] = None,
        reference_probs: Optional[np.ndarray] = None,
        current_probs: Optional[np.ndarray] = None,
        reference_importances: Optional[Dict[str, float]] = None,
        current_importances: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive drift report combining all drift checks.

        Saves the report as a JSON file and returns it.

        Args:
            reference_data: Baseline feature data.
            current_data: Current feature data.
            reference_labels: Baseline true labels.
            reference_predictions: Baseline model predictions.
            current_labels: Current true labels.
            current_predictions: Current model predictions.
            reference_probs: Baseline prediction probabilities.
            current_probs: Current prediction probabilities.
            reference_importances: Baseline feature importances.
            current_importances: Current feature importances.

        Returns:
            Full drift report dictionary.
        """
        report: Dict[str, Any] = {
            "report_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.utcnow().isoformat(),
            "sections": {},
            "summary": {},
        }

        data_drift = self.detect_data_drift(reference_data, current_data)
        report["sections"]["data_drift"] = data_drift

        if (reference_predictions is not None and reference_labels is not None
                and current_predictions is not None
                and current_labels is not None):
            concept_drift = self.detect_concept_drift(
                reference_predictions, reference_labels,
                current_predictions, current_labels,
            )
            report["sections"]["concept_drift"] = concept_drift

        if reference_probs is not None and current_probs is not None:
            pred_drift = self.detect_prediction_drift(
                reference_probs, current_probs,
            )
            report["sections"]["prediction_drift"] = pred_drift

        if reference_importances is not None and current_importances is not None:
            importance_drift = self.compute_feature_importance_drift(
                reference_importances, current_importances,
            )
            report["sections"]["feature_importance_drift"] = importance_drift

        severity_map = {"high": 3, "moderate": 2, "low": 1, "none": 0}
        max_severity = "none"
        for section_name, section_data in report["sections"].items():
            section_severity = section_data.get(
                "severity",
                "high" if section_data.get("drift_detected", False)
                or section_data.get("overall_drift_detected", False)
                else "none",
            )
            if severity_map.get(section_severity, 0) > severity_map.get(
                max_severity, 0
            ):
                max_severity = section_severity

        report["summary"] = {
            "overall_drift_detected": max_severity != "none",
            "max_severity": max_severity,
            "n_drifted_features": len(
                data_drift.get("drifted_features", [])
            ),
            "sections_checked": list(report["sections"].keys()),
            "retrain_recommendation": self.should_retrain(report),
        }

        report_path = os.path.join(
            "data", "monitoring",
            f"drift_report_{report['report_id']}.json",
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Drift report saved to %s", report_path)

        return report

    def should_retrain(self, drift_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decision logic for retraining based on drift report.

        High severity drift -> immediate retrain.
        Moderate -> schedule retrain.
        Low -> continue monitoring.

        Args:
            drift_report: Output from generate_drift_report.

        Returns:
            Dictionary with retrain decision and rationale.
        """
        severity = drift_report.get("summary", {}).get("max_severity", "none")
        recommendation: Dict[str, Any] = {
            "retrain_needed": False,
            "urgency": "none",
            "action": "continue_monitoring",
            "rationale": [],
        }

        if severity == "high":
            recommendation["retrain_needed"] = True
            recommendation["urgency"] = "immediate"
            recommendation["action"] = "retrain_now"
            recommendation["rationale"].append(
                "High severity drift detected across multiple indicators"
            )

            concept = drift_report.get("sections", {}).get("concept_drift", {})
            if concept.get("auc_degradation", 0) > 0.1:
                recommendation["rationale"].append(
                    f"AUC degradation: {concept['auc_degradation']:.4f}"
                )

        elif severity == "moderate":
            recommendation["retrain_needed"] = True
            recommendation["urgency"] = "scheduled"
            recommendation["action"] = "schedule_retrain"
            recommendation["rationale"].append(
                "Moderate drift detected; schedule retraining within 7 days"
            )

        elif severity == "low":
            recommendation["retrain_needed"] = False
            recommendation["urgency"] = "monitor"
            recommendation["action"] = "continue_monitoring"
            recommendation["rationale"].append(
                "Low severity drift; continue monitoring"
            )

        data_drift = drift_report.get("sections", {}).get("data_drift", {})
        drifted_count = len(data_drift.get("drifted_features", []))
        total_features = (
            len(data_drift.get("numeric_features", {}))
            + len(data_drift.get("categorical_features", {}))
        )
        if total_features > 0 and drifted_count / total_features > 0.3:
            recommendation["retrain_needed"] = True
            if recommendation["urgency"] != "immediate":
                recommendation["urgency"] = "scheduled"
                recommendation["action"] = "schedule_retrain"
            recommendation["rationale"].append(
                f"{drifted_count}/{total_features} features show drift "
                f"({drifted_count / total_features:.1%})"
            )

        return recommendation


class AutoRetrainer:
    """Automated model retraining triggered by drift detection."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = self._load_config(config_path)
        self.models_path = self.config.get("api", {}).get(
            "model_path", "models/saved"
        )
        self.retrain_log: List[Dict] = []

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        try:
            import yaml
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Config not found at %s, using defaults", config_path)
            return {"api": {"model_path": "models/saved"}}

    def check_and_retrain(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
        drift_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check drift report and retrain if necessary.

        Retrains the model with combined old and new data, compares against
        the original on a holdout set, and auto-replaces if the new model
        performs better.

        Args:
            model: Current trained model.
            X_train: New training features.
            y_train: New training labels.
            X_val: Validation features.
            y_val: Validation labels.
            X_test: Holdout test features.
            y_test: Holdout test labels.
            feature_names: List of feature names.
            drift_report: Output from DriftDetector.generate_drift_report.

        Returns:
            Dictionary with retraining results and model comparison.
        """
        retrain_result: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "retrain_triggered": False,
            "old_model_metrics": {},
            "new_model_metrics": {},
            "model_replaced": False,
            "retrain_log_entry": {},
        }

        recommendation = drift_report.get("summary", {}).get(
            "retrain_recommendation", {}
        )
        if not recommendation.get("retrain_needed", False):
            logger.info("No retraining needed based on drift report")
            retrain_result["retrain_log_entry"] = {
                "action": "skip",
                "reason": recommendation.get("action", "continue_monitoring"),
            }
            self.retrain_log.append(retrain_result["retrain_log_entry"])
            return retrain_result

        retrain_result["retrain_triggered"] = True
        logger.info(
            "Retraining triggered: urgency=%s",
            recommendation.get("urgency", "unknown"),
        )

        old_probs = model.predict_proba(X_test)
        if old_probs.ndim == 2:
            old_probs = old_probs[:, 1]
        old_preds = (old_probs >= 0.5).astype(int)
        retrain_result["old_model_metrics"] = {
            "auc_roc": round(roc_auc_score(y_test, old_probs), 6),
            "f1_score": round(f1_score(y_test, old_preds, zero_division=0), 6),
            "accuracy": round(accuracy_score(y_test, old_preds), 6),
            "log_loss": round(log_loss(y_test, old_probs), 6),
            "brier_score": round(brier_score_loss(y_test, old_probs), 6),
        }

        new_model = copy.deepcopy(model)
        try:
            if hasattr(new_model, "partial_fit"):
                new_model.partial_fit(X_train, y_train)
                logger.info("Incremental learning applied via partial_fit")
            else:
                combined_X = pd.concat([X_train, X_val], axis=0)
                combined_y = pd.concat([y_train, y_val], axis=0)
                new_model.fit(combined_X, combined_y)
                logger.info("Full retraining with combined data")
        except Exception as e:
            logger.error("Retraining failed: %s", str(e))
            retrain_result["retrain_log_entry"] = {
                "action": "failed",
                "error": str(e),
                "urgency": recommendation.get("urgency"),
            }
            self.retrain_log.append(retrain_result["retrain_log_entry"])
            return retrain_result

        new_probs = new_model.predict_proba(X_test)
        if new_probs.ndim == 2:
            new_probs = new_probs[:, 1]
        new_preds = (new_probs >= 0.5).astype(int)
        retrain_result["new_model_metrics"] = {
            "auc_roc": round(roc_auc_score(y_test, new_probs), 6),
            "f1_score": round(f1_score(y_test, new_preds, zero_division=0), 6),
            "accuracy": round(accuracy_score(y_test, new_preds), 6),
            "log_loss": round(log_loss(y_test, new_probs), 6),
            "brier_score": round(brier_score_loss(y_test, new_probs), 6),
        }

        is_better = self.validate_model_degradation(model, new_model, X_test, y_test)
        retrain_result["model_replaced"] = is_better

        if is_better:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(self.models_path, f"retrained_{timestamp}")
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, "model.joblib")
            joblib.dump(new_model, model_path)

            old_model_path = os.path.join(self.models_path, "model.joblib")
            if os.path.exists(old_model_path):
                backup_path = os.path.join(self.models_path, "model_backup.joblib")
                joblib.dump(model, backup_path)

            joblib.dump(new_model, os.path.join(self.models_path, "model.joblib"))
            logger.info("New model saved and deployed: %s", model_path)
        else:
            logger.info(
                "New model did not outperform old model; keeping original"
            )

        retrain_result["retrain_log_entry"] = {
            "action": "replaced" if is_better else "kept_original",
            "urgency": recommendation.get("urgency"),
            "old_auc": retrain_result["old_model_metrics"]["auc_roc"],
            "new_auc": retrain_result["new_model_metrics"]["auc_roc"],
            "improvement": round(
                retrain_result["new_model_metrics"]["auc_roc"]
                - retrain_result["old_model_metrics"]["auc_roc"],
                6,
            ),
        }
        self.retrain_log.append(retrain_result["retrain_log_entry"])

        return retrain_result

    def schedule_retraining_check(self, schedule: str = "daily") -> Dict[str, Any]:
        """
        Configure periodic retraining check schedule.

        Args:
            schedule: One of 'hourly', 'daily', 'weekly', 'monthly'.

        Returns:
            Schedule configuration dictionary.
        """
        valid_schedules = {"hourly", "daily", "weekly", "monthly"}
        if schedule not in valid_schedules:
            raise ValueError(
                f"Invalid schedule '{schedule}'. Must be one of {valid_schedules}"
            )

        interval_hours = {
            "hourly": 1,
            "daily": 24,
            "weekly": 168,
            "monthly": 720,
        }

        config = {
            "schedule": schedule,
            "interval_hours": interval_hours[schedule],
            "enabled": True,
            "created_at": datetime.utcnow().isoformat(),
            "next_check": None,
            "last_check": None,
        }

        logger.info("Retraining schedule set to %s", schedule)
        return config

    def validate_model_degradation(
        self,
        old_model: Any,
        new_model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> bool:
        """
        Validate that the new model outperforms the old model on the holdout.

        Uses AUC-ROC as primary metric with F1 as secondary tiebreaker.

        Args:
            old_model: Current production model.
            new_model: Newly trained model.
            X_test: Holdout test features.
            y_test: Holdout test labels.

        Returns:
            True if new model is better.
        """
        old_probs = old_model.predict_proba(X_test)
        new_probs = new_model.predict_proba(X_test)

        if old_probs.ndim == 2:
            old_probs = old_probs[:, 1]
        if new_probs.ndim == 2:
            new_probs = new_probs[:, 1]

        old_auc = roc_auc_score(y_test, old_probs)
        new_auc = roc_auc_score(y_test, new_probs)

        old_f1 = f1_score(y_test, (old_probs >= 0.5).astype(int), zero_division=0)
        new_f1 = f1_score(y_test, (new_probs >= 0.5).astype(int), zero_division=0)

        logger.info(
            "Model validation — Old AUC: %.4f, New AUC: %.4f | "
            "Old F1: %.4f, New F1: %.4f",
            old_auc, new_auc, old_f1, new_f1,
        )

        auc_improvement = new_auc > old_auc
        f1_improvement = new_f1 > old_f1

        if auc_improvement:
            return True
        if abs(new_auc - old_auc) < 0.001 and f1_improvement:
            return True

        return False

    def rollback_model(self, previous_model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Rollback to a previous model if the new one fails.

        Args:
            previous_model_path: Path to the model file to restore.
                If None, uses the backup model.

        Returns:
            Rollback status dictionary.
        """
        if previous_model_path is None:
            previous_model_path = os.path.join(
                self.models_path, "model_backup.joblib"
            )

        result: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "rollback_executed": False,
            "source_path": previous_model_path,
            "target_path": os.path.join(self.models_path, "model.joblib"),
        }

        if not os.path.exists(previous_model_path):
            result["error"] = f"Backup model not found at {previous_model_path}"
            logger.error("Rollback failed: %s", result["error"])
            return result

        try:
            model = joblib.load(previous_model_path)
            joblib.dump(model, result["target_path"])
            result["rollback_executed"] = True
            logger.info("Model rolled back from %s", previous_model_path)
        except Exception as e:
            result["error"] = str(e)
            logger.error("Rollback failed: %s", str(e))

        return result
