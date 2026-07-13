"""
Ensemble Deployment Module for Default Prediction.

Provides a production-grade ensemble combining XGBoost, Random Forest,
and Logistic Regression with weighted averaging and stacking strategies.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnsembleDeployer:
    """Production ensemble of XGBoost, Random Forest, and Logistic Regression.

    Supports weighted averaging, stacking, confidence estimation via model
    disagreement, and full persistence of trained models and metadata.

    Attributes:
        config: Loaded YAML configuration dictionary.
        xgb_model: XGBoost classifier instance.
        rf_model: Random Forest classifier instance.
        lr_model: Logistic Regression classifier instance.
        meta_learner: Stacking meta-learner (LogisticRegression).
        ensemble_weights: Validation-AUC-based weights for averaging.
        scaler: StandardScaler fitted on training features.
        is_trained: Whether the ensemble has been trained.
        training_metadata: Metadata from the most recent training run.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize EnsembleDeployer with project configuration.

        Args:
            config_path: Path to the project YAML configuration file.

        Raises:
            FileNotFoundError: If config_path does not exist.
        """
        self.config: Dict[str, Any] = self._load_config(config_path)
        self.xgb_model: Optional[XGBClassifier] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.lr_model: Optional[LogisticRegression] = None
        self.meta_learner: Optional[LogisticRegression] = None
        self.ensemble_weights: Optional[np.ndarray] = None
        self.scaler: StandardScaler = StandardScaler()
        self.is_trained: bool = False
        self.training_metadata: Dict[str, Any] = {}
        self.individual_metrics: Dict[str, Dict[str, float]] = {}
        self._feature_names: List[str] = []

        self.build_ensemble()
        logger.info("EnsembleDeployer initialized with config from %s", config_path)

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file.

        Args:
            config_path: Path to the YAML config file.

        Returns:
            Parsed configuration dictionary.

        Raises:
            FileNotFoundError: If config file is missing.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def build_ensemble(self) -> None:
        """Create the three base models with production hyperparameters.

        XGBoost is configured for imbalanced classification with
        ``scale_pos_weight``. Random Forest uses balanced class weights.
        Logistic Regression uses L2 penalty with balanced weights.
        """
        self.xgb_model = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            scale_pos_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=42,
            use_label_encoder=False,
            n_jobs=-1,
        )

        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        self.lr_model = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        self.meta_learner = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )

        logger.info(
            "Ensemble built: XGBoost, RandomForest, LogisticRegression + meta-learner"
        )

    def _compute_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute a full suite of classification metrics.

        Args:
            y_true: Ground-truth binary labels.
            y_prob: Predicted probabilities for the positive class.
            y_pred: Hard predictions (0/1).

        Returns:
            Dictionary of metric name to value.
        """
        eps = 1e-15
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "brier_score": float(brier_score_loss(y_true, y_prob_clipped)),
        }

        if len(np.unique(y_true)) > 1:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob_clipped))
            metrics["auc_pr"] = float(average_precision_score(y_true, y_prob_clipped))
            metrics["log_loss"] = float(log_loss(y_true, y_prob_clipped))
        else:
            metrics["auc_roc"] = 0.5
            metrics["auc_pr"] = 0.0
            metrics["log_loss"] = float("inf")

        return metrics

    def train_ensemble(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        use_smote: bool = True,
    ) -> Dict[str, Any]:
        """Train all three base models and build the ensemble.

        Steps:
            1. Optionally apply SMOTE on training data.
            2. Scale features via StandardScaler.
            3. Train XGBoost, RandomForest, LogisticRegression independently.
            4. Compute validation metrics for each model.
            5. Derive weighted-averaging weights from validation AUC.
            6. Build stacking predictions and fit the meta-learner.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            X_val: Validation feature matrix (optional).
            y_val: Validation labels (optional).
            use_smote: Whether to apply SMOTE oversampling on training data.

        Returns:
            Dictionary containing training metadata, metrics, and weights.
        """
        start_time = time.time()

        X_tr = self._to_dataframe(X_train)
        y_tr = self._to_numpy(y_train)
        self._feature_names = list(X_tr.columns)

        if X_val is not None and y_val is not None:
            X_v = self._to_dataframe(X_val)
            y_v = self._to_numpy(y_val)
        else:
            X_v, y_v = None, None

        if use_smote and SMOTE_AVAILABLE:
            logger.info("Applying SMOTE to training data")
            smote = SMOTE(random_state=42)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr.values, y_tr)
            X_tr = pd.DataFrame(X_tr_res, columns=self._feature_names)
            y_tr = y_tr_res
        elif use_smote and not SMOTE_AVAILABLE:
            logger.warning("imbalanced-learn not installed; skipping SMOTE")

        X_tr_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_tr),
            columns=self._feature_names,
            index=X_tr.index,
        )

        models = {
            "xgboost": self.xgb_model,
            "random_forest": self.rf_model,
            "logistic_regression": self.lr_model,
        }

        self.individual_metrics = {}

        for name, model in models.items():
            logger.info("Training %s", name)
            model.fit(X_tr_scaled, y_tr)

        if X_v is not None:
            X_v_scaled = pd.DataFrame(
                self.scaler.transform(X_v),
                columns=self._feature_names,
                index=X_v.index,
            )
            self._compute_validation_metrics(X_v_scaled, y_v, models)
            self._compute_ensemble_weights(X_v_scaled, y_v)
            self._fit_stacking(X_v_scaled, y_v)
        else:
            default_w = np.array([1.0 / len(models)] * len(models))
            self.ensemble_weights = default_w
            logger.info("No validation set; using uniform weights: %s", default_w)

        self.is_trained = True
        elapsed = time.time() - start_time

        self.training_metadata = {
            "trained_at": datetime.now().isoformat(),
            "training_samples": int(len(y_tr)),
            "validation_samples": int(len(y_v)) if y_v is not None else 0,
            "n_features": int(X_tr_scaled.shape[1]),
            "smote_applied": use_smote,
            "training_time_seconds": round(elapsed, 2),
            "ensemble_weights": {
                "xgboost": float(self.ensemble_weights[0]),
                "random_forest": float(self.ensemble_weights[1]),
                "logistic_regression": float(self.ensemble_weights[2]),
            },
        }

        logger.info(
            "Ensemble trained in %.2fs. Weights: %s",
            elapsed,
            self.training_metadata["ensemble_weights"],
        )
        return self.training_metadata

    def _compute_validation_metrics(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        models: Dict[str, Any],
    ) -> None:
        """Evaluate each model on the validation set and store metrics.

        Args:
            X_val: Scaled validation features.
            y_val: Validation labels.
            models: Mapping of model name to fitted estimator.
        """
        for name, model in models.items():
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            self.individual_metrics[name] = self._compute_metrics(y_val, y_prob, y_pred)
            logger.info(
                "%s val AUC-ROC: %.4f", name, self.individual_metrics[name]["auc_roc"]
            )

    def _compute_ensemble_weights(self, X_val: pd.DataFrame, y_val: np.ndarray) -> None:
        """Derive weighted-averaging weights proportional to validation AUC.

        Weights are normalised so that they sum to 1.

        Args:
            X_val: Scaled validation features.
            y_val: Validation labels.
        """
        auc_scores = np.array(
            [
                self.individual_metrics["xgboost"]["auc_roc"],
                self.individual_metrics["random_forest"]["auc_roc"],
                self.individual_metrics["logistic_regression"]["auc_roc"],
            ]
        )

        auc_scores = np.maximum(auc_scores, 1e-10)
        self.ensemble_weights = auc_scores / auc_scores.sum()

        logger.info("Ensemble weights from AUC: %s", self.ensemble_weights.tolist())

    def _fit_stacking(self, X_val: pd.DataFrame, y_val: np.ndarray) -> None:
        """Generate stacking meta-features on validation data and fit meta-learner.

        Args:
            X_val: Scaled validation features.
            y_val: Validation labels.
        """
        meta_features = self._get_stacking_meta_features(X_val)
        self.meta_learner.fit(meta_features, y_val)
        meta_pred = self.meta_learner.predict_proba(meta_features)[:, 1]
        meta_metrics = self._compute_metrics(
            y_val, meta_pred, (meta_pred >= 0.5).astype(int)
        )
        self.individual_metrics["stacking_meta"] = meta_metrics
        logger.info("Stacking meta-learner val AUC-ROC: %.4f", meta_metrics["auc_roc"])

    def _get_stacking_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Produce stacking meta-features (predicted probabilities) from base models.

        Args:
            X: Scaled feature matrix.

        Returns:
            Array of shape (n_samples, 3) with each column being the positive-class
            probability from one base model.
        """
        meta = np.column_stack(
            [
                self.xgb_model.predict_proba(X)[:, 1],
                self.rf_model.predict_proba(X)[:, 1],
                self.lr_model.predict_proba(X)[:, 1],
            ]
        )
        return meta

    def predict_ensemble(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Return both weighted-averaging and stacking ensemble predictions.

        Args:
            X: Feature matrix (raw, will be scaled internally).

        Returns:
            Dictionary with keys ``weighted_proba``, ``stacking_proba``,
            ``weighted_pred``, ``stacking_pred``.
        """
        self._ensure_trained()
        X_scaled = self._scale_features(X)

        base_probas = np.column_stack(
            [
                self.xgb_model.predict_proba(X_scaled)[:, 1],
                self.rf_model.predict_proba(X_scaled)[:, 1],
                self.lr_model.predict_proba(X_scaled)[:, 1],
            ]
        )

        weighted_proba = base_probas @ self.ensemble_weights
        stacking_proba = self.meta_learner.predict_proba(
            self._get_stacking_meta_features(X_scaled)
        )[:, 1]

        return {
            "weighted_proba": weighted_proba,
            "stacking_proba": stacking_proba,
            "weighted_pred": (weighted_proba >= 0.5).astype(int),
            "stacking_pred": (stacking_proba >= 0.5).astype(int),
        }

    def predict_with_confidence(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        confidence_level: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        """Predictions with confidence intervals derived from model disagreement.

        The standard deviation across the three base-model probabilities is used
        as a proxy for prediction uncertainty.

        Args:
            X: Feature matrix.
            confidence_level: Confidence level for the interval (default 0.95).

        Returns:
            Dictionary containing ``mean_proba``, ``std_proba``,
            ``ci_lower``, ``ci_upper``, ``predictions``, and ``confidence``.
        """
        self._ensure_trained()
        X_scaled = self._scale_features(X)

        base_probas = np.column_stack(
            [
                self.xgb_model.predict_proba(X_scaled)[:, 1],
                self.rf_model.predict_proba(X_scaled)[:, 1],
                self.lr_model.predict_proba(X_scaled)[:, 1],
            ]
        )

        mean_proba = base_probas.mean(axis=1)
        std_proba = base_probas.std(axis=1)

        from scipy import stats as sp_stats

        z = sp_stats.norm.ppf(1 - (1 - confidence_level) / 2)
        ci_lower = np.clip(mean_proba - z * std_proba, 0, 1)
        ci_upper = np.clip(mean_proba + z * std_proba, 0, 1)
        ci_width = ci_upper - ci_lower
        confidence = 1.0 - ci_width

        return {
            "mean_proba": mean_proba,
            "std_proba": std_proba,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "predictions": (mean_proba >= 0.5).astype(int),
            "confidence": confidence,
        }

    def get_individual_predictions(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Return predictions from each base model separately.

        Args:
            X: Feature matrix.

        Returns:
            Dictionary mapping model name to its positive-class probabilities.
        """
        self._ensure_trained()
        X_scaled = self._scale_features(X)

        return {
            "xgboost": self.xgb_model.predict_proba(X_scaled)[:, 1],
            "random_forest": self.rf_model.predict_proba(X_scaled)[:, 1],
            "logistic_regression": self.lr_model.predict_proba(X_scaled)[:, 1],
        }

    def compare_models(self) -> pd.DataFrame:
        """Return a DataFrame comparing all model metrics side-by-side.

        Returns:
            DataFrame with metric names as rows and model names as columns.

        Raises:
            RuntimeError: If the ensemble has not been trained yet.
        """
        self._ensure_trained()
        if not self.individual_metrics:
            raise RuntimeError("No metrics available. Train the ensemble first.")

        df = pd.DataFrame(self.individual_metrics)
        df.index.name = "metric"
        return df

    def save_ensemble(self, path: str) -> None:
        """Persist all models, scaler, weights, and metadata to disk.

        The directory structure is::

            <path>/
                xgb_model.joblib
                rf_model.joblib
                lr_model.joblib
                meta_learner.joblib
                scaler.joblib
                ensemble_metadata.json

        Args:
            path: Directory path where artefacts will be saved.
        """
        self._ensure_trained()
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        artefacts = {
            "xgb_model.joblib": self.xgb_model,
            "rf_model.joblib": self.rf_model,
            "lr_model.joblib": self.lr_model,
            "meta_learner.joblib": self.meta_learner,
            "scaler.joblib": self.scaler,
        }

        for filename, obj in artefacts.items():
            joblib.dump(obj, save_dir / filename)
            logger.info("Saved %s", filename)

        metadata = {
            **self.training_metadata,
            "individual_metrics": self.individual_metrics,
            "ensemble_weights": self.ensemble_weights.tolist(),
            "feature_names": self._feature_names,
            "saved_at": datetime.now().isoformat(),
        }

        meta_path = save_dir / "ensemble_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        logger.info("Ensemble saved to %s", save_dir)

    def load_ensemble(self, path: str) -> None:
        """Load previously saved ensemble from disk.

        Args:
            path: Directory containing saved artefacts.

        Raises:
            FileNotFoundError: If any required artefact is missing.
        """
        load_dir = Path(path)

        required_files = [
            "xgb_model.joblib",
            "rf_model.joblib",
            "lr_model.joblib",
            "meta_learner.joblib",
            "scaler.joblib",
            "ensemble_metadata.json",
        ]

        for fname in required_files:
            if not (load_dir / fname).exists():
                raise FileNotFoundError(
                    f"Required artefact not found: {load_dir / fname}"
                )

        self.xgb_model = joblib.load(load_dir / "xgb_model.joblib")
        self.rf_model = joblib.load(load_dir / "rf_model.joblib")
        self.lr_model = joblib.load(load_dir / "lr_model.joblib")
        self.meta_learner = joblib.load(load_dir / "meta_learner.joblib")
        self.scaler = joblib.load(load_dir / "scaler.joblib")

        with open(load_dir / "ensemble_metadata.json", "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        self.ensemble_weights = np.array(metadata["ensemble_weights"])
        self._feature_names = metadata.get("feature_names", [])
        self.individual_metrics = metadata.get("individual_metrics", {})
        self.training_metadata = metadata
        self.is_trained = True

        logger.info("Ensemble loaded from %s", load_dir)

    # -- Internal helpers -------------------------------------------------------

    def _ensure_trained(self) -> None:
        """Raise if ensemble has not been trained or loaded."""
        if not self.is_trained:
            raise RuntimeError(
                "Ensemble is not trained. Call train_ensemble() or load_ensemble() first."
            )

    def _scale_features(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Scale features using the fitted StandardScaler.

        Args:
            X: Raw feature matrix.

        Returns:
            Scaled DataFrame with the same column names.
        """
        df = self._to_dataframe(X)
        scaled = self.scaler.transform(df)
        return pd.DataFrame(scaled, columns=df.columns, index=df.index)

    @staticmethod
    def _to_dataframe(
        X: Union[pd.DataFrame, np.ndarray],
    ) -> pd.DataFrame:
        """Convert input to a pandas DataFrame.

        Args:
            X: Input features.

        Returns:
            DataFrame representation.
        """
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)

    @staticmethod
    def _to_numpy(y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Convert input to a NumPy array.

        Args:
            y: Input labels.

        Returns:
            1-D NumPy array.
        """
        if isinstance(y, pd.Series):
            return y.values
        return np.asarray(y)

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return (
            f"EnsembleDeployer(status={status}, "
            f"models=['xgboost', 'random_forest', 'logistic_regression'])"
        )
