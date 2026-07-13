"""
Prediction Module
Generates default probability predictions with 12-month horizon estimation
and confidence intervals.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


class DefaultPredictor:
    """Predicts loan default probability 12 months in advance."""

    def __init__(self, config: dict):
        self.config = config
        self.prediction_horizon = config["data"]["prediction_horizon_months"]

    def predict_probability(self, model, X: pd.DataFrame,
                            calibrated: bool = True) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = model.decision_function(X)
            probs = 1 / (1 + np.exp(-probs))
        return probs

    def predict_with_confidence(self, model, X: pd.DataFrame,
                                n_estimators: int = 100) -> pd.DataFrame:
        base_probs = self.predict_probability(model, X)

        if hasattr(model, "estimators_"):
            tree_preds = []
            for estimator in model.estimators_:
                if hasattr(estimator, "predict_proba"):
                    tree_preds.append(estimator.predict_proba(X)[:, 1])
            if tree_preds:
                tree_preds = np.array(tree_preds)
                std_dev = tree_preds.std(axis=0)
            else:
                std_dev = np.zeros(len(X))
        else:
            std_dev = np.full(len(X), np.std(base_probs) * 0.1)

        results = pd.DataFrame({
            "default_probability": base_probs,
            "confidence_lower": np.clip(base_probs - 1.96 * std_dev, 0, 1),
            "confidence_upper": np.clip(base_probs + 1.96 * std_dev, 0, 1),
            "risk_score": (base_probs * 100).round(2),
        })

        results["risk_category"] = pd.cut(
            results["default_probability"],
            bins=[0, 0.1, 0.25, 0.5, 0.75, 1.0],
            labels=["Very Low", "Low", "Medium", "High", "Very High"]
        )

        results["recommended_action"] = results["default_probability"].apply(
            self._get_action_recommendation)

        logger.info(f"Predictions generated for {len(X)} samples")
        return results

    def _get_action_recommendation(self, prob: float) -> str:
        if prob < 0.1:
            return "APPROVE - Standard terms"
        elif prob < 0.25:
            return "APPROVE - Enhanced monitoring recommended"
        elif prob < 0.5:
            return "REVIEW - Additional documentation required"
        elif prob < 0.75:
            return "CONDITIONAL - Higher interest rate / collateral required"
        else:
            return "DECLINE - High default risk"

    def predict_with_ensemble(self, models: Dict[str, Any], X: pd.DataFrame,
                              weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        if weights is None:
            weights = {name: 1.0 / len(models) for name in models}

        all_probs = []
        for name, model in models.items():
            w = weights.get(name, 1.0 / len(models))
            probs = self.predict_probability(model, X)
            all_probs.append(probs * w)

        ensemble_probs = np.sum(all_probs, axis=0)
        individual_probs = {
            name: self.predict_probability(model, X)
            for name, model in models.items()
        }

        results = pd.DataFrame({
            "ensemble_default_probability": ensemble_probs,
            "ensemble_risk_score": (ensemble_probs * 100).round(2),
        })

        for name, probs in individual_probs.items():
            results[f"{name}_probability"] = probs

        results["prediction_variance"] = np.var(
            list(individual_probs.values()), axis=0)

        results["risk_category"] = pd.cut(
            results["ensemble_default_probability"],
            bins=[0, 0.1, 0.25, 0.5, 0.75, 1.0],
            labels=["Very Low", "Low", "Medium", "High", "Very High"]
        )

        logger.info(f"Ensemble predictions generated for {len(X)} samples")
        return results

    def estimate_12_month_default(self, model, X: pd.DataFrame,
                                  current_delinquency_months: int = 0) -> pd.DataFrame:
        results = self.predict_with_confidence(model, X)

        time_adjustment = np.where(
            current_delinquency_months > 0,
            1 + (current_delinquency_months / 12) * 0.3,
            1.0
        )

        results["adjusted_12m_probability"] = np.clip(
            results["default_probability"] * time_adjustment, 0, 1)
        results["adjusted_risk_score"] = (
            results["adjusted_12m_probability"] * 100).round(2)

        results["estimated_default_horizon"] = np.where(
            results["adjusted_12m_probability"] > 0.5,
            "Within 6 months",
            np.where(
                results["adjusted_12m_probability"] > 0.25,
                "Within 12 months",
                "Low risk (12+ months)")
        )

        logger.info("12-month default estimation complete")
        return results

    def batch_predict(self, model, X: pd.DataFrame,
                      batch_size: int = 10000) -> pd.DataFrame:
        n_samples = len(X)
        results_list = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = X.iloc[start:end]
            batch_results = self.predict_with_confidence(model, batch)
            results_list.append(batch_results)
            logger.info(
                f"Batch {start // batch_size + 1} complete "
                f"({end}/{n_samples})")

        results = pd.concat(results_list, ignore_index=True)
        return results
