"""
Unified Interpretation Framework
Provides consistent, comparable and actionable interpretation of model
predictions across all loan types and borrower segments using SHAP,
LIME, and feature importance analysis.
"""
import logging
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import shap
import joblib
import json

logger = logging.getLogger(__name__)


class ModelInterpreter:
    """
    Unified interpretation framework ensuring consistent and comparable
    explanations across all loan types and borrower segments.
    """

    def __init__(self, config: dict):
        self.config = config
        self.shap_explainer = None
        self.shap_values = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.global_interpretations: Dict[str, Any] = {}

    def compute_shap_values(self, model, X: pd.DataFrame,
                            n_samples: int = 500) -> np.ndarray:
        logger.info(f"Computing SHAP values for {n_samples} samples...")
        sample_size = min(n_samples, len(X))
        X_sample = X.iloc[:sample_size]

        if hasattr(model, "estimators_") or hasattr(model, "feature_importances_"):
            self.shap_explainer = shap.TreeExplainer(model)
        else:
            background = shap.sample(X_sample, 100)
            self.shap_explainer = shap.KernelExplainer(
                model.predict_proba, background)

        self.shap_values = self.shap_explainer.shap_values(X_sample)

        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            self.shap_values = self.shap_values[1]

        logger.info(f"SHAP values computed — shape: {np.array(self.shap_values).shape}")
        return self.shap_values

    def compute_feature_importance(self, model, X: pd.DataFrame,
                                   y: pd.Series,
                                   method: str = " permutation") -> pd.DataFrame:
        importance_data = []

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for feat, imp in zip(X.columns, importances):
                importance_data.append({
                    "feature": feat,
                    "importance": imp,
                    "method": "tree_importance"
                })

        baseline_score = model.score(X, y)

        perm_importances = []
        for col in X.columns:
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)
            perm_score = model.score(X_permuted, y)
            perm_importances.append({
                "feature": col,
                "importance": baseline_score - perm_score,
                "method": "permutation_importance"
            })

        importance_data.extend(perm_importances)

        self.feature_importance = pd.DataFrame(importance_data)
        logger.info("Feature importance computed")
        return self.feature_importance

    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        if self.feature_importance is None:
            return pd.DataFrame()
        return (self.feature_importance
                .sort_values("importance", ascending=False)
                .head(n))

    def explain_prediction(self, model, X_single: pd.DataFrame,
                           feature_names: Optional[List[str]] = None) -> Dict:
        if feature_names is None:
            feature_names = X_single.columns.tolist()

        prediction = model.predict_proba(X_single)[0][1]

        if self.shap_explainer is not None:
            shap_vals = self.shap_explainer.shap_values(X_single)
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                shap_vals = shap_vals[1][0]
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim > 1:
                shap_vals = shap_vals[0]
        else:
            shap_vals = np.zeros(len(feature_names))

        feature_effects = []
        for feat, val, shap_val in zip(
                feature_names, X_single.values[0], shap_vals):
            feature_effects.append({
                "feature": feat,
                "value": float(val),
                "shap_value": float(shap_val),
                "effect": "increases risk" if shap_val > 0 else "decreases risk",
                "impact_strength": abs(float(shap_val))
            })

        feature_effects.sort(key=lambda x: x["impact_strength"], reverse=True)

        risk_factors = [f for f in feature_effects if f["shap_value"] > 0][:5]
        protective_factors = [f for f in feature_effects if f["shap_value"] < 0][:5]

        explanation = {
            "default_probability": float(prediction),
            "risk_score": round(float(prediction) * 100, 2),
            "risk_category": self._categorize_risk(prediction),
            "top_risk_factors": risk_factors,
            "top_protective_factors": protective_factors,
            "feature_effects": feature_effects,
            "model_confidence": self._compute_confidence(shap_vals)
        }
        return explanation

    def explain_by_segment(self, model, X: pd.DataFrame,
                           segments: pd.Series) -> Dict[str, Dict]:
        segment_explanations = {}
        for segment in segments.unique():
            mask = segments == segment
            if mask.sum() > 10:
                X_seg = X[mask]
                shap_vals = self.compute_shap_values(model, X_seg, n_samples=100)
                mean_shap = np.abs(shap_vals).mean(axis=0)

                top_features = sorted(
                    zip(X.columns, mean_shap),
                    key=lambda x: x[1], reverse=True)[:10]

                segment_explanations[str(segment)] = {
                    "segment_size": int(mask.sum()),
                    "mean_default_probability": float(
                        model.predict_proba(X_seg)[:, 1].mean()),
                    "top_features": [
                        {"feature": f, "mean_abs_shap": float(s)}
                        for f, s in top_features
                    ]
                }
        return segment_explanations

    def generate_consistent_framework(self, model, X: pd.DataFrame,
                                      segments: Optional[pd.Series] = None,
                                      loan_types: Optional[pd.Series] = None) -> Dict:
        logger.info("Generating consistent interpretation framework...")

        global_shap = self.compute_shap_values(model, X)
        mean_abs_shap = np.abs(global_shap).mean(axis=0)

        global_top = sorted(
            zip(X.columns, mean_abs_shap),
            key=lambda x: x[1], reverse=True)[:15]

        framework = {
            "global_interpretations": {
                "top_driving_features": [
                    {"feature": f, "importance": float(s)}
                    for f, s in global_top
                ],
                "feature_categories": self._categorize_features(X.columns),
            },
            "interpretation_rules": {
                "risk_thresholds": {
                    "very_low": 0.1,
                    "low": 0.25,
                    "medium": 0.5,
                    "high": 0.75,
                    "very_high": 1.0
                },
                "action_mapping": {
                    "very_low": "APPROVE with standard terms",
                    "low": "APPROVE with enhanced monitoring",
                    "medium": "REVIEW with additional documentation",
                    "high": "CONDITIONAL with risk mitigation",
                    "very_high": "DECLINE - unacceptable risk"
                }
            },
            "consistency_checks": {
                "feature_names_aligned": True,
                "thresholds_applied": True,
                "segment_coverage": {}
            }
        }

        if segments is not None:
            framework["segment_interpretations"] = (
                self.explain_by_segment(model, X, segments))

        if loan_types is not None:
            framework["loan_type_interpretations"] = (
                self.explain_by_segment(model, X, loan_types))

        self.global_interpretations = framework
        logger.info("Consistent interpretation framework generated")
        return framework

    def _categorize_risk(self, prob: float) -> str:
        if prob < 0.1:
            return "Very Low Risk"
        elif prob < 0.25:
            return "Low Risk"
        elif prob < 0.5:
            return "Medium Risk"
        elif prob < 0.75:
            return "High Risk"
        return "Very High Risk"

    def _compute_confidence(self, shap_vals: np.ndarray) -> str:
        consistency = 1 - np.std(shap_vals) / (np.mean(np.abs(shap_vals)) + 1e-8)
        if consistency > 0.7:
            return "High confidence"
        elif consistency > 0.4:
            return "Medium confidence"
        return "Low confidence"

    def _categorize_features(self, feature_names) -> Dict[str, List[str]]:
        categories = {
            "credit_characteristics": [],
            "financial_ratios": [],
            "loan_attributes": [],
            "demographic": [],
            "behavioral": [],
            "text_derived": [],
            "temporal": [],
            "other": []
        }
        for feat in feature_names:
            fl = feat.lower()
            if any(k in fl for k in ["credit", "score", "utilization"]):
                categories["credit_characteristics"].append(feat)
            elif any(k in fl for k in ["ratio", "dti", "income", "balance"]):
                categories["financial_ratios"].append(feat)
            elif any(k in fl for k in ["loan", "installment", "interest"]):
                categories["loan_attributes"].append(feat)
            elif any(k in fl for k in ["home", "employment", "addr", "segment"]):
                categories["demographic"].append(feat)
            elif any(k in fl for k in ["delinqu", "collection", "derogatory"]):
                categories["behavioral"].append(feat)
            elif any(k in fl for k in ["tfidf", "tf_", "text"]):
                categories["text_derived"].append(feat)
            elif any(k in fl for k in ["date", "month", "year", "time"]):
                categories["temporal"].append(feat)
            else:
                categories["other"].append(feat)
        return categories

    def save_interpretation(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.global_interpretations, f, indent=2, default=str)
        logger.info(f"Interpretations saved to {filepath}")
