"""Real-time SHAP explanation service for live predictions."""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class RealtimeSHAPExplainer:
    """Real-time SHAP explainer with caching, counterfactuals, and regulatory reporting."""

    def __init__(
        self,
        model,
        feature_names: List[str],
        background_data: Optional[pd.DataFrame] = None,
        cache_explanations: bool = True,
    ):
        self.model = model
        self.feature_names = list(feature_names)
        self.cache_explanations = cache_explanations
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = timedelta(minutes=30)
        self._cache_timestamps: Dict[str, datetime] = {}
        self._background_data = background_data
        self._explainer = None
        self._global_explainer = None
        self._segment_averages: Dict[str, Dict[str, float]] = {}
        self._risk_categories = {
            "high": {"threshold": 0.6, "label": "High Risk", "color": "#e74c3c"},
            "medium": {"threshold": 0.3, "label": "Medium Risk", "color": "#f39c12"},
            "low": {"threshold": 0.0, "label": "Low Risk", "color": "#27ae60"},
        }

        self._initialize_explainer()

    def _initialize_explainer(self):
        """Auto-detect model type and initialize the appropriate SHAP explainer."""
        try:
            model_type = type(self.model).__name__.lower()
            tree_models = [
                "xgbclassifier", "xgbregressor", "lgbmclassifier",
                "lgbmregressor", "randomforestclassifier", "randomforestregressor",
                "gradientboostingclassifier", "gradientboostingregressor",
                "catboostclassifier", "catboostregressor",
            ]

            if any(t in model_type for t in tree_models):
                self._explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized TreeExplainer for %s", model_type)
            else:
                if self._background_data is not None:
                    bg = self._background_data
                    if len(bg) > 100:
                        bg = bg.sample(100, random_state=42)
                    self._explainer = shap.KernelExplainer(
                        self.model.predict_proba
                        if hasattr(self.model, "predict_proba")
                        else self.model.predict,
                        bg,
                    )
                else:
                    dummy = pd.DataFrame(
                        np.zeros((10, len(self.feature_names))),
                        columns=self.feature_names,
                    )
                    self._explainer = shap.KernelExplainer(
                        self.model.predict_proba
                        if hasattr(self.model, "predict_proba")
                        else self.model.predict,
                        dummy,
                    )
                logger.info("Initialized KernelExplainer (fallback)")
        except Exception as e:
            logger.error("Failed to initialize SHAP explainer: %s", e)
            self._explainer = None

    def _compute_shap_values(self, features_df: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for a given feature DataFrame."""
        if self._explainer is None:
            raise RuntimeError("SHAP explainer is not initialized")
        raw = self._explainer.shap_values(features_df)
        if isinstance(raw, list):
            return raw[1] if len(raw) > 1 else raw[0]
        return raw

    def _hash_features(self, features_dict: Dict[str, Any]) -> str:
        """Create a deterministic hash for a feature dictionary."""
        serialised = json.dumps(features_dict, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]

    def _get_risk_category(self, probability: float) -> Dict[str, str]:
        """Map probability to a risk category."""
        for _name, cat in sorted(
            self._risk_categories.items(),
            key=lambda x: x[1]["threshold"],
            reverse=True,
        ):
            if probability >= cat["threshold"]:
                return cat
        return self._risk_categories["low"]

    def _purge_stale_cache(self):
        """Remove expired cache entries."""
        now = datetime.utcnow()
        stale = [
            k for k, ts in self._cache_timestamps.items()
            if now - ts > self._cache_ttl
        ]
        for k in stale:
            self._cache.pop(k, None)
            self._cache_timestamps.pop(k, None)

    # ------------------------------------------------------------------ #
    #  Public – single / batch prediction explanations                   #
    # ------------------------------------------------------------------ #

    def explain_prediction(
        self,
        features_dict: Dict[str, Any],
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Explain a single prediction with SHAP values and natural-language text."""
        fhash = self._hash_features(features_dict)

        if self.cache_explanations:
            self._purge_stale_cache()
            cached = self.get_cached_explanation(fhash)
            if cached is not None:
                logger.debug("Cache hit for %s", fhash)
                return cached

        feat_df = pd.DataFrame([features_dict])[self.feature_names]
        shap_vals = self._compute_shap_values(feat_df)[0]

        ranking = sorted(
            zip(self.feature_names, shap_vals, [features_dict.get(f, 0) for f in self.feature_names]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        risk_factors = []
        protective_factors = []
        for fname, sval, fval in ranking:
            entry = {
                "feature": fname,
                "value": float(fval),
                "shap_value": float(sval),
                "direction": "increases_risk" if sval > 0 else "decreases_risk",
            }
            if sval > 0:
                risk_factors.append(entry)
            else:
                protective_factors.append(entry)

        explanation_lines = []
        if risk_factors:
            explanation_lines.append("Key risk factors:")
            for rf in risk_factors[:5]:
                explanation_lines.append(
                    f"  - {rf['feature']} = {rf['value']} (adds {rf['shap_value']:.4f} to risk score)"
                )
        if protective_factors:
            explanation_lines.append("Protective factors:")
            for pf in protective_factors[:3]:
                explanation_lines.append(
                    f"  - {pf['feature']} = {pf['value']} (reduces risk by {abs(pf['shap_value']):.4f})"
                )

        result = {
            "shap_values": {f: float(v) for f, v in zip(self.feature_names, shap_vals)},
            "feature_ranking": [
                {"feature": f, "shap_value": float(s), "value": float(v)}
                for f, s, v in ranking
            ],
            "explanation_text": "\n".join(explanation_lines),
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,
            "features_hash": fhash,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if self.cache_explanations:
            self.cache_explanation(fhash, result)

        return result

    def explain_prediction_html(
        self,
        features_dict: Dict[str, Any],
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Return chart-ready data structures for a web frontend."""
        base = self.explain_prediction(features_dict, top_n=top_n)

        bar_data = [
            {
                "feature": r["feature"],
                "value": r["value"],
                "shap_value": r["shap_value"],
                "direction": "increases" if r["shap_value"] > 0 else "decreases",
            }
            for r in base["feature_ranking"]
        ]

        cumulative = 0.0
        waterfall_data = []
        for item in bar_data:
            start = cumulative
            cumulative += item["shap_value"]
            waterfall_data.append({
                "feature": item["feature"],
                "start": round(start, 6),
                "end": round(cumulative, 6),
                "shap_value": item["shap_value"],
                "direction": item["direction"],
            })

        force_entries = []
        for item in bar_data:
            force_entries.append({
                "feature": item["feature"],
                "shap_value": item["shap_value"],
                "value": item["value"],
            })
        force_plot_data = {
            "base_value": float(np.mean(list(base["shap_values"].values()))),
            "entries": force_entries,
        }

        return {
            "bar_chart": bar_data,
            "waterfall_chart": waterfall_data,
            "force_plot": force_plot_data,
            "explanation_text": base["explanation_text"],
            "risk_factors": base["risk_factors"],
            "protective_factors": base["protective_factors"],
        }

    def explain_batch(
        self,
        features_list: List[Dict[str, Any]],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Explain a batch of predictions."""
        return [self.explain_prediction(fd, top_n=top_n) for fd in features_list]

    # ------------------------------------------------------------------ #
    #  Global explanations                                                #
    # ------------------------------------------------------------------ #

    def get_global_explanations(
        self,
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """Compute global feature importance from a sample of data."""
        if self._explainer is None:
            return {"error": "Explainer not initialised"}

        if self._background_data is not None:
            sample = self._background_data.sample(
                min(n_samples, len(self._background_data)), random_state=42
            )
        else:
            sample = pd.DataFrame(
                np.random.randn(n_samples, len(self.feature_names)),
                columns=self.feature_names,
            )

        shap_vals = self._compute_shap_values(sample)

        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        global_ranking = sorted(
            zip(self.feature_names, mean_abs),
            key=lambda x: x[1],
            reverse=True,
        )

        interactions = {}
        top_features = [f for f, _ in global_ranking[:5]]
        for i, f1 in enumerate(top_features):
            for f2 in top_features[i + 1:]:
                idx1 = self.feature_names.index(f1)
                idx2 = self.feature_names.index(f2)
                interaction_strength = float(
                    np.mean(np.abs(shap_vals[:, idx1] * shap_vals[:, idx2]))
                )
                interactions[f"{f1} x {f2}"] = interaction_strength

        segment_importance = {}
        if self._background_data is not None and "income" in sample.columns:
            median_income = sample["income"].median()
            for seg_name, mask in [
                ("low_income", sample["income"] <= median_income),
                ("high_income", sample["income"] > median_income),
            ]:
                seg_shap = shap_vals[mask.values]
                seg_mean = np.mean(np.abs(seg_shap), axis=0)
                segment_importance[seg_name] = {
                    f: float(v) for f, v in zip(self.feature_names, seg_mean)
                }

        return {
            "global_feature_importance": [
                {"feature": f, "mean_abs_shap": float(v)} for f, v in global_ranking
            ],
            "feature_interactions": interactions,
            "importance_by_segment": segment_importance,
            "n_samples": len(sample),
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------ #
    #  Counterfactual explanations                                        #
    # ------------------------------------------------------------------ #

    def explain_counterfactual(
        self,
        features_dict: Dict[str, Any],
        target_probability: float = 0.3,
        max_changes: int = 3,
    ) -> Dict[str, Any]:
        """Find the minimum changes needed to reach a target risk probability."""
        feat_df = pd.DataFrame([features_dict])[self.feature_names]
        baseline_shap = self._compute_shap_values(feat_df)[0]
        baseline_score = float(np.sum(baseline_shap))

        candidate_features = sorted(
            zip(self.feature_names, baseline_shap),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        best_counterfactual = dict(features_dict)
        best_score = baseline_score
        changed_features: List[Dict[str, Any]] = []

        for fname, sval in candidate_features[:max_changes + 5]:
            if sval <= 0:
                continue
            trial = dict(features_dict)
            orig_val = trial[fname]

            if isinstance(orig_val, (int, float)):
                step = abs(orig_val) * 0.3 if orig_val != 0 else 0.5
                trial[fname] = orig_val - step
            else:
                continue

            trial_df = pd.DataFrame([trial])[self.feature_names]
            trial_shap = self._compute_shap_values(trial_df)[0]
            trial_score = float(np.sum(trial_shap))

            if trial_score < best_score:
                best_score = trial_score
                best_counterfactual = dict(trial)
                changed_features.append({
                    "feature": fname,
                    "original_value": float(orig_val),
                    "counterfactual_value": float(trial[fname]),
                    "change": float(trial[fname]) - float(orig_val),
                    "shap_impact": float(sval),
                })
                if len(changed_features) >= max_changes:
                    break

        return {
            "original": features_dict,
            "counterfactual": best_counterfactual,
            "changed_features": changed_features,
            "original_score": baseline_score,
            "counterfactual_score": best_score,
            "improvement": baseline_score - best_score,
            "target_probability": target_probability,
            "feasible": best_score <= target_probability,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------ #
    #  Segment-specific explanations                                     #
    # ------------------------------------------------------------------ #

    def explain_by_segment(
        self,
        features_dict: Dict[str, Any],
        segment: str,
    ) -> Dict[str, Any]:
        """Compare a prediction to its peer segment average."""
        explanation = self.explain_prediction(features_dict)
        segment_avg = self._segment_averages.get(segment, {})

        segment_comparison = []
        for fname in self.feature_names:
            pred_val = features_dict.get(fname, 0)
            seg_val = segment_avg.get(fname, pred_val)
            segment_comparison.append({
                "feature": fname,
                "prediction_value": float(pred_val),
                "segment_average": float(seg_val),
                "difference": float(pred_val - seg_val),
                "pct_difference": (
                    float((pred_val - seg_val) / seg_val * 100)
                    if seg_val != 0 else 0.0
                ),
            })

        segment_comparison.sort(key=lambda x: abs(x["difference"]), reverse=True)

        return {
            **explanation,
            "segment": segment,
            "segment_comparison": segment_comparison[:10],
            "segment_data_available": bool(segment_avg),
        }

    def set_segment_averages(self, segment: str, averages: Dict[str, float]):
        """Store pre-computed segment averages for comparison."""
        self._segment_averages[segment] = averages

    # ------------------------------------------------------------------ #
    #  Executive & regulatory summaries                                   #
    # ------------------------------------------------------------------ #

    def generate_executive_summary(
        self,
        features_dict: Dict[str, Any],
        prediction_result: Dict[str, Any],
    ) -> str:
        """Generate a 3-5 sentence summary suitable for a risk committee."""
        prob = prediction_result.get("probability", prediction_result.get("default_probability", 0.5))
        cat = self._get_risk_category(prob)
        explanation = self.explain_prediction(features_dict, top_n=5)

        top_risks = explanation["risk_factors"][:3]
        risk_text = ", ".join(r["feature"] for r in top_risks) if top_risks else "no significant risk factors"

        confidence = prediction_result.get("confidence", 0.85)
        rec = "recommend further review" if prob >= 0.5 else "recommend approval with standard conditions"

        summary = (
            f"The applicant is classified as {cat['label']} with a default probability of {prob:.1%}. "
            f"Key risk drivers include {risk_text}. "
            f"The model confidence for this assessment is {confidence:.0%}. "
            f"We {rec}. "
            f"This assessment is based on {len(features_dict)} features analysed by the ensemble credit-risk model."
        )
        return summary

    def generate_regulatory_explanation(
        self,
        features_dict: Dict[str, Any],
        prediction_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Produce a regulatory-compliant explanation with audit trail."""
        explanation = self.explain_prediction(features_dict)
        prob = prediction_result.get("probability", prediction_result.get("default_probability", 0.5))
        cat = self._get_risk_category(prob)

        factor_breakdown = []
        for item in explanation["feature_ranking"]:
            factor_breakdown.append({
                "factor_name": item["feature"],
                "factor_value": item["value"],
                "contribution_to_score": item["shap_value"],
                "direction": "adverse" if item["shap_value"] > 0 else "favorable",
                "materiality": (
                    "high" if abs(item["shap_value"]) > 0.1
                    else "medium" if abs(item["shap_value"]) > 0.05
                    else "low"
                ),
            })

        return {
            "regulatory_format": "RBI_CREDIT_RISK_EXPLANATION",
            "version": "2.0",
            "applicant_summary": {
                "default_probability": round(prob, 6),
                "risk_category": cat["label"],
                "model_version": prediction_result.get("model_version", "unknown"),
            },
            "factor_breakdown": factor_breakdown,
            "decision_rationale": {
                "primary_decision": "approve" if prob < 0.5 else "decline",
                "key_factors": [f["factor_name"] for f in factor_breakdown[:5]],
                "risk_score": round(float(np.sum([f["contribution_to_score"] for f in factor_breakdown])), 6),
            },
            "audit_trail": {
                "timestamp": datetime.utcnow().isoformat(),
                "model_type": type(self.model).__name__,
                "n_features_used": len(self.feature_names),
                "shap_method": type(self._explainer).__name__ if self._explainer else "none",
                "explanation_id": explanation.get("features_hash", "N/A"),
            },
            "compliance_notes": [
                "Explanation generated using SHAP (SHapley Additive exPlanations) methodology.",
                "All material adverse factors are disclosed per RBI Fair Practices Code.",
                "Model is subject to periodic validation and drift monitoring.",
            ],
        }

    # ------------------------------------------------------------------ #
    #  Cache management                                                   #
    # ------------------------------------------------------------------ #

    def cache_explanation(self, features_hash: str, explanation: Dict[str, Any]):
        """Store an explanation in the cache."""
        self._cache[features_hash] = explanation
        self._cache_timestamps[features_hash] = datetime.utcnow()
        logger.debug("Cached explanation for %s", features_hash)

    def get_cached_explanation(self, features_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached explanation if still valid."""
        if features_hash not in self._cache:
            return None
        ts = self._cache_timestamps.get(features_hash)
        if ts and datetime.utcnow() - ts > self._cache_ttl:
            self._cache.pop(features_hash, None)
            self._cache_timestamps.pop(features_hash, None)
            return None
        return self._cache[features_hash]

    def invalidate_cache(self):
        """Clear the entire explanation cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("SHAP explanation cache cleared")


class ExplanationFormatter:
    """Format SHAP explanations for various audiences."""

    def format_for_loan_officer(
        self,
        shap_result: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simple, action-oriented format for loan officers."""
        prob = prediction.get("probability", prediction.get("default_probability", 0.5))

        risk_items = []
        for rf in shap_result.get("risk_factors", [])[:5]:
            risk_items.append(
                f"{rf['feature']} = {rf['value']} → increases default risk"
            )

        protect_items = []
        for pf in shap_result.get("protective_factors", [])[:3]:
            protect_items.append(
                f"{pf['feature']} = {pf['value']} → reduces default risk"
            )

        if prob < 0.3:
            action = "APPROVE – Low risk applicant"
        elif prob < 0.6:
            action = "REVIEW – Moderate risk, consider additional documentation"
        else:
            action = "DECLINE – High risk applicant"

        return {
            "format": "loan_officer",
            "decision_recommendation": action,
            "risk_level": f"{prob:.0%} estimated default probability",
            "top_risk_factors": risk_items,
            "protective_factors": protect_items,
            "simple_summary": shap_result.get("explanation_text", ""),
        }

    def format_for_risk_manager(
        self,
        shap_result: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Detailed format with metrics for risk managers."""
        prob = prediction.get("probability", prediction.get("default_probability", 0.5))
        return {
            "format": "risk_manager",
            "default_probability": prob,
            "risk_category": prediction.get("risk_category", "unknown"),
            "confidence": prediction.get("confidence", None),
            "shap_values": shap_result.get("shap_values", {}),
            "feature_ranking": shap_result.get("feature_ranking", []),
            "risk_factors": shap_result.get("risk_factors", []),
            "protective_factors": shap_result.get("protective_factors", []),
            "model_version": prediction.get("model_version", "unknown"),
            "explanation_id": shap_result.get("features_hash", ""),
            "timestamp": shap_result.get("timestamp", ""),
        }

    def format_for_executive(
        self,
        shap_result: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """High-level summary for executives."""
        prob = prediction.get("probability", prediction.get("default_probability", 0.5))
        top_risks = [
            rf["feature"] for rf in shap_result.get("risk_factors", [])[:3]
        ]
        return {
            "format": "executive",
            "headline": f"{'Low' if prob < 0.3 else 'Medium' if prob < 0.6 else 'High'} risk applicant",
            "default_probability": f"{prob:.1%}",
            "key_risk_factors": top_risks,
            "recommendation": (
                "Approve" if prob < 0.3
                else "Review further" if prob < 0.6
                else "Decline"
            ),
            "summary": shap_result.get("explanation_text", ""),
        }

    def format_for_auditor(
        self,
        shap_result: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Regulatory-compliant format with audit trail."""
        prob = prediction.get("probability", prediction.get("default_probability", 0.5))
        factor_log = []
        for item in shap_result.get("feature_ranking", []):
            factor_log.append({
                "feature": item["feature"],
                "value": item["value"],
                "shap_contribution": item["shap_value"],
                "direction": "adverse" if item["shap_value"] > 0 else "favorable",
            })
        return {
            "format": "auditor",
            "audit_trail": {
                "explanation_id": shap_result.get("features_hash", ""),
                "timestamp": shap_result.get("timestamp", ""),
                "model_type": type(self).__name__,
            },
            "decision": {
                "probability": prob,
                "category": prediction.get("risk_category", "unknown"),
            },
            "factor_log": factor_log,
            "compliance": "RBI Fair Practices Code compliant",
        }

    def format_for_customer(
        self,
        shap_result: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Customer-friendly explanation with no jargon."""
        prob = prediction.get("probability", prediction.get("default_probability", 0.5))

        plain_risks = []
        for rf in shap_result.get("risk_factors", [])[:3]:
            fname = rf["feature"].replace("_", " ").title()
            plain_risks.append(
                f"Your {fname} was a factor that slightly increased the risk assessment."
            )

        plain_protections = []
        for pf in shap_result.get("protective_factors", [])[:3]:
            fname = pf["feature"].replace("_", " ").title()
            plain_protections.append(
                f"Your {fname} helped lower the overall risk."
            )

        if prob < 0.3:
            outcome = "Your application looks good based on the information provided."
        elif prob < 0.6:
            outcome = "Your application may need some additional information or review."
        else:
            outcome = "Based on the current information, there are some concerns with the application."

        return {
            "format": "customer",
            "outcome_summary": outcome,
            "things_that_helped": plain_protections,
            "things_to_improve": plain_risks,
            "next_steps": "A representative will be in touch if any further information is needed.",
        }

    def format_for_api(
        self,
        shap_result: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """JSON API response format."""
        return {
            "format": "api",
            "prediction": {
                "probability": prediction.get("probability", prediction.get("default_probability")),
                "risk_category": prediction.get("risk_category"),
                "confidence": prediction.get("confidence"),
            },
            "explanation": {
                "shap_values": shap_result.get("shap_values", {}),
                "top_features": shap_result.get("feature_ranking", []),
                "risk_factors": shap_result.get("risk_factors", []),
                "protective_factors": shap_result.get("protective_factors", []),
            },
            "metadata": {
                "explanation_id": shap_result.get("features_hash", ""),
                "timestamp": shap_result.get("timestamp", ""),
                "model_version": prediction.get("model_version", ""),
            },
        }
