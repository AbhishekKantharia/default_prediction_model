"""
Model Training Module
Trains multiple classification algorithms for default prediction with
loan-type-specific models and ensemble stacking.
"""
import logging
import os
import json
import joblib
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and manages multiple ML models for default prediction."""

    def __init__(self, config: dict):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.calibrated_models: Dict[str, Any] = {}
        self.best_model_name: Optional[str] = None
        self.cv_scores: Dict[str, Dict[str, float]] = {}
        self.models_path = config["api"]["model_path"]
        os.makedirs(self.models_path, exist_ok=True)

    def _get_base_models(self) -> Dict[str, Any]:
        return {
            "logistic_regression": LogisticRegression(
                C=1.0, penalty="l2", solver="lbfgs",
                max_iter=1000, class_weight="balanced",
                random_state=42),

            "random_forest": RandomForestClassifier(
                n_estimators=300, max_depth=12,
                min_samples_split=10, min_samples_leaf=5,
                class_weight="balanced", n_jobs=-1,
                random_state=42),

            "xgboost": xgb.XGBClassifier(
                n_estimators=500, max_depth=6,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, scale_pos_weight=3,
                eval_metric="auc", use_label_encoder=False,
                random_state=42, n_jobs=-1),

            "lightgbm": lgb.LGBMClassifier(
                n_estimators=500, max_depth=8,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, is_unbalance=True,
                random_state=42, n_jobs=-1, verbose=-1),

            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=300, max_depth=5,
                learning_rate=0.05, subsample=0.8,
                min_samples_split=10, random_state=42),
        }

    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        smote = SMOTE(random_state=42, sampling_strategy=0.5)
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(f"SMOTE applied — {X.shape} -> {X_res.shape}")
        return X_res, y_res

    def train_single_model(self, name: str, model, X_train: pd.DataFrame,
                           y_train: pd.Series, X_val: pd.DataFrame,
                           y_val: pd.Series, use_smote: bool = True) -> Dict:
        logger.info(f"Training {name}...")

        if use_smote:
            X_train_res, y_train_res = self.apply_smote(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train

        model.fit(X_train_res, y_train_res)

        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)

        results = {
            "model": model,
            "train_accuracy": train_score,
            "val_accuracy": val_score,
            "name": name
        }

        logger.info(f"{name} — train: {train_score:.4f}, val: {val_score:.4f}")
        return results

    def cross_validate_models(self, X: pd.DataFrame, y: pd.Series,
                              n_splits: int = 5) -> Dict[str, Dict[str, float]]:
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=42)
        base_models = self._get_base_models()
        scores = {}

        for name, model in base_models.items():
            logger.info(f"Cross-validating {name}...")
            cv_results = cross_val_score(
                model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
            scores[name] = {
                "mean_auc": cv_results.mean(),
                "std_auc": cv_results.std(),
                "scores": cv_results.tolist()
            }
            logger.info(f"{name} AUC: {cv_results.mean():.4f} "
                        f"(+/- {cv_results.std():.4f})")

        self.cv_scores = scores
        return scores

    def build_stacking_ensemble(self, base_models: Dict[str, Any]) -> Any:
        estimators = [(name, model) for name, model in base_models.items()]
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                C=1.0, max_iter=1000, random_state=42),
            cv=5, n_jobs=-1
        )
        logger.info("Stacking ensemble created")
        return stacking

    def calibrate_model(self, model, X_val: pd.DataFrame,
                        y_val: pd.Series) -> Any:
        calibrated = CalibratedClassifierCV(
            model, method="isotonic", cv=5)
        calibrated.fit(X_val, y_val)
        return calibrated

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         use_smote: bool = True) -> Dict[str, Dict]:
        base_models = self._get_base_models()
        all_results = {}

        for name, model in base_models.items():
            results = self.train_single_model(
                name, model, X_train, y_train, X_val, y_val, use_smote)
            all_results[name] = results
            self.models[name] = results["model"]

        stacking_model = self.build_stacking_ensemble(base_models)
        stacking_results = self.train_single_model(
            "stacking_ensemble", stacking_model, X_train, y_train,
            X_val, y_val, use_smote)
        all_results["stacking_ensemble"] = stacking_results
        self.models["stacking_ensemble"] = stacking_results["model"]

        best_name = max(
            all_results,
            key=lambda k: all_results[k]["val_accuracy"])
        self.best_model_name = best_name
        logger.info(f"Best model: {best_name} "
                    f"(val_acc: {all_results[best_name]['val_accuracy']:.4f})")

        return all_results

    def train_loan_type_models(self, df: pd.DataFrame,
                               target_col: str,
                               loan_type_col: str = "loan_type") -> Dict:
        loan_models = {}
        if loan_type_col not in df.columns:
            logger.warning("Loan type column not found — skipping loan-type models")
            return loan_models

        for loan_type in df[loan_type_col].unique():
            subset = df[df[loan_type_col] == loan_type]
            if len(subset) < 100:
                logger.info(f"Skipping {loan_type}: insufficient samples ({len(subset)})")
                continue

            X = subset.drop(columns=[target_col, "loan_id",
                                     loan_type_col], errors="ignore")
            y = subset[target_col]

            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=5,
                learning_rate=0.05, eval_metric="auc",
                use_label_encoder=False, random_state=42, verbose=0)
            model.fit(X, y)
            loan_models[loan_type] = model
            logger.info(f"Trained {loan_type} model — samples: {len(subset)}")

        return loan_models

    def save_models(self, tag: str = "latest"):
        save_dir = os.path.join(self.models_path, tag)
        os.makedirs(save_dir, exist_ok=True)

        for name, model in self.models.items():
            path = os.path.join(save_dir, f"{name}.joblib")
            joblib.dump(model, path)
            logger.info(f"Saved model: {path}")

        metadata = {
            "best_model": self.best_model_name,
            "cv_scores": self.cv_scores,
            "model_names": list(self.models.keys())
        }
        meta_path = os.path.join(save_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Models saved to {save_dir}")

    def load_models(self, tag: str = "latest") -> Dict[str, Any]:
        load_dir = os.path.join(self.models_path, tag)
        loaded = {}
        for fname in os.listdir(load_dir):
            if fname.endswith(".joblib"):
                name = fname.replace(".joblib", "")
                loaded[name] = joblib.load(os.path.join(load_dir, fname))
        self.models = loaded
        logger.info(f"Loaded {len(loaded)} models from {load_dir}")
        return loaded
