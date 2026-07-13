"""
MLflow Integration Module for Experiment Tracking and Model Registry.

Provides ``MLflowTracker`` for low-level MLflow operations and
``PipelineMLOps`` for high-level training/evaluation workflow tracking.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import mlflow
    from mlflow.models import ModelSignature
    from mlflow.tracking import MlflowClient
    from mlflow.types import ColSpec, DataType

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Low-level wrapper around MLflow for experiment tracking.

    Handles experiment creation, run lifecycle, parameter/metric logging,
    model logging, feature-importance tables, dataset statistics, and the
    MLflow Model Registry.

    Attributes:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: Local or remote tracking URI.
        client: ``MlflowClient`` instance.
        experiment_id: Numeric ID of the resolved experiment.
        current_run: The active ``mlflow.run`` (if any).
    """

    def __init__(
        self,
        experiment_name: str = "default_prediction",
        tracking_uri: str = "mlruns",
    ) -> None:
        """Initialise the tracker and ensure the experiment exists.

        Args:
            experiment_name: MLflow experiment name.
            tracking_uri: URI for the MLflow tracking server / local dir.
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "mlflow is required. Install it via: pip install mlflow>=2.5.0"
            )

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.experiment_id = self._ensure_experiment()
        self.current_run: Optional[Any] = None

        logger.info(
            "MLflowTracker ready – experiment='%s' (id=%s), uri=%s",
            self.experiment_name,
            self.experiment_id,
            self.tracking_uri,
        )

    def _ensure_experiment(self) -> str:
        """Create the MLflow experiment if it does not already exist.

        Returns:
            The experiment ID.
        """
        exp = self.client.get_experiment_by_name(self.experiment_name)
        if exp is not None:
            return exp.experiment_id  # type: ignore[return-value]
        exp_id = self.client.create_experiment(self.experiment_name)
        logger.info("Created MLflow experiment '%s' (id=%s)", self.experiment_name, exp_id)
        return exp_id

    def start_run(self, run_name: Optional[str] = None) -> Any:
        """Start a new MLflow run.

        Args:
            run_name: Optional human-readable name for the run.

        Returns:
            The active ``mlflow.run`` object.
        """
        self.current_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
        )
        logger.info(
            "MLflow run started: %s (id=%s)",
            run_name,
            self.current_run.info.run_id,
        )
        return self.current_run

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current active run.

        Args:
            status: Termination status (``FINISHED`` or ``FAILED``).
        """
        if self.current_run is not None:
            mlflow.end_run(status=status)
            logger.info("MLflow run ended with status=%s", status)
            self.current_run = None

    def log_params(self, params_dict: Dict[str, Any]) -> None:
        """Log a dictionary of hyperparameters to the active run.

        Args:
            params_dict: Parameter name-value pairs.
        """
        self._ensure_active_run()
        flat = self._flatten_dict(params_dict)
        mlflow.log_params(flat)
        logger.info("Logged %d parameters", len(flat))

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics to the active run.

        Args:
            metrics_dict: Metric name-value pairs.
            step: Optional global step (for iterative training).
        """
        self._ensure_active_run()
        filtered = {k: float(v) for k, v in metrics_dict.items() if np.isfinite(v)}
        if step is not None:
            mlflow.log_metrics(filtered, step=step)
        else:
            mlflow.log_metrics(filtered)
        logger.info("Logged %d metrics", len(filtered))

    def log_model(
        self,
        model: Any,
        model_name: str,
        artifact_path: str = "models",
    ) -> None:
        """Persist and log a trained model as an MLflow artefact.

        Args:
            model: A fitted scikit-learn / XGBoost estimator.
            model_name: Logical name used for the artefact sub-directory.
            artifact_path: Parent artefact path inside the run.
        """
        self._ensure_active_run()
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=os.path.join(artifact_path, model_name),
            registered_model_name=model_name,
        )
        logger.info("Model '%s' logged to artefact path '%s'", model_name, artifact_path)

    def log_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
    ) -> None:
        """Log feature importance as an MLflow table artefact.

        Args:
            feature_names: List of feature names.
            importances: Array of importance values aligned with feature_names.
        """
        self._ensure_active_run()
        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": np.asarray(importances).tolist(),
            }
        ).sort_values("importance", ascending=False)

        table = mlflow.table(df)
        mlflow.log_artifact(
            _dataframe_to_csv_path(df),
            artifact_path="feature_importance",
        )
        logger.info("Feature importance table logged (%d features)", len(df))

    def log_dataset_info(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
    ) -> None:
        """Log summary statistics for train / validation / test sets.

        Args:
            X_train: Training feature matrix.
            X_val: Validation feature matrix.
            X_test: Test feature matrix.
        """
        self._ensure_active_run()

        for label, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
            df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
            stats = {
                f"dataset/{label}/n_samples": int(df.shape[0]),
                f"dataset/{label}/n_features": int(df.shape[1]),
                f"dataset/{label}/mean_abs": float(df.abs().mean().mean()),
                f"dataset/{label}/std": float(df.std().mean()),
                f"dataset/{label}/null_pct": float(df.isnull().mean().mean()),
            }
            mlflow.log_metrics(stats)

        logger.info("Dataset statistics logged for train/val/test")

    def log_artifacts(self, local_dir: str) -> None:
        """Log all files under *local_dir* as MLflow artefacts.

        Args:
            local_dir: Path to a local directory of artefacts.
        """
        self._ensure_active_run()
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"Artefact directory not found: {local_dir}")
        mlflow.log_artifacts(local_dir)
        logger.info("Artefacts logged from %s", local_dir)

    def compare_runs(self, n_runs: int = 10) -> pd.DataFrame:
        """Retrieve and compare the last *n_runs* runs.

        Args:
            n_runs: Number of recent runs to compare.

        Returns:
            DataFrame with one row per run.
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=["start_time DESC"],
            max_results=n_runs,
        )

        rows = []
        for run in runs:
            row: Dict[str, Any] = {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "status": run.info.status,
                "start_time": run.info.start_time,
                "duration_ms": run.info.end_time - run.info.start_time
                if run.info.end_time
                else None,
            }
            row.update(run.data.params)
            row.update(run.data.metrics)
            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info("Compared %d runs", len(df))
        return df

    def get_best_run(self, metric: str = "val_auc") -> Optional[Dict[str, Any]]:
        """Return the best run according to *metric*.

        Args:
            metric: Metric name to optimise (higher-is-better).

        Returns:
            Dictionary of run details or ``None`` if no runs exist.
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if not runs:
            return None

        best = runs[0]
        result: Dict[str, Any] = {
            "run_id": best.info.run_id,
            "metrics": best.data.metrics,
            "params": best.data.params,
            "tags": best.data.tags,
        }
        logger.info("Best run by %s: %s", metric, result["run_id"])
        return result

    def register_model(
        self,
        model_name: str,
        stage: str = "production",
    ) -> None:
        """Transition the latest version of *model_name* to *stage*.

        Args:
            model_name: Registered model name.
            stage: Target lifecycle stage (e.g. ``production``).
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")
        if not versions:
            logger.warning("No model versions found for '%s'", model_name)
            return

        latest = max(versions, key=lambda v: int(v.version))
        self.client.transition_model_version_stage(
            name=model_name,
            version=latest.version,
            stage=stage,
        )
        logger.info(
            "Model '%s' version %s transitioned to '%s'",
            model_name,
            latest.version,
            stage,
        )

    def tag_run(self, tags_dict: Dict[str, str]) -> None:
        """Add arbitrary tags to the current run.

        Args:
            tags_dict: Tag key-value pairs.
        """
        self._ensure_active_run()
        mlflow.set_tags(tags_dict)
        logger.info("Tagged run with %d tags", len(tags_dict))

    # -- Helpers ----------------------------------------------------------------

    def _ensure_active_run(self) -> None:
        """Verify that a run is active; raise otherwise."""
        if self.current_run is None:
            raise RuntimeError("No active MLflow run. Call start_run() first.")

    @staticmethod
    def _flatten_dict(
        d: Dict[str, Any], parent_key: str = "", sep: str = "/"
    ) -> Dict[str, str]:
        """Flatten a nested dict with ``/``-separated keys.

        Args:
            d: Dictionary to flatten.
            parent_key: Prefix for nested keys.
            sep: Separator character.

        Returns:
            Flat dictionary with string values.
        """
        items: List[tuple] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    MLflowTracker._flatten_dict(v, new_key, sep=sep).items()
                )
            else:
                items.append((new_key, str(v)))
        return dict(items)


def _dataframe_to_csv_path(df: pd.DataFrame) -> str:
    """Write a DataFrame to a temporary CSV and return the path.

    Args:
        df: DataFrame to persist.

    Returns:
        Absolute path to the created CSV file.
    """
    tmp_dir = Path("tmp_mlflow_artifacts")
    tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / f"feature_importance_{int(time.time())}.csv"
    df.to_csv(path, index=False)
    return str(path)


class PipelineMLOps:
    """High-level MLOps pipeline that wraps ``MLflowTracker``.

    Provides convenience methods for tracking full training runs,
    evaluation runs, auto-promotion of the best model, and HTML report
    generation.

    Attributes:
        config: Loaded YAML configuration.
        tracker: Underlying ``MLflowTracker`` instance.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialise the MLOps pipeline.

        Args:
            config_path: Path to the project YAML configuration file.

        Raises:
            FileNotFoundError: If *config_path* does not exist.
            ImportError: If ``mlflow`` is not installed.
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "mlflow is required. Install it via: pip install mlflow>=2.5.0"
            )

        self.config = self._load_config(config_path)

        experiment_name = self.config.get("project", {}).get(
            "name", "default_prediction"
        )
        self.tracker = MLflowTracker(
            experiment_name=experiment_name,
            tracking_uri="mlruns",
        )
        logger.info("PipelineMLOps initialised")

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load and return YAML configuration.

        Args:
            config_path: Path to YAML file.

        Returns:
            Parsed configuration dictionary.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def track_training_run(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        feature_names: List[str],
    ) -> str:
        """Track a complete training experiment end-to-end.

        Logs parameters, training/validation/test metrics, the fitted model,
        feature importance, and dataset statistics.

        Args:
            model: Fitted estimator exposing ``get_params()``, ``predict()``,
                ``predict_proba()``, and optionally ``feature_importances_``.
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            X_test: Test features.
            y_test: Test labels.
            feature_names: Ordered list of feature names.

        Returns:
            The MLflow run ID.
        """
        y_train_np = self._to_numpy(y_train)
        y_val_np = self._to_numpy(y_val)
        y_test_np = self._to_numpy(y_test)

        run = self.tracker.start_run(run_name=f"train_{datetime.now():%Y%m%d_%H%M%S}")
        try:
            params = model.get_params()
            self.tracker.log_params(params)

            self.tracker.log_dataset_info(X_train, X_val, X_test)

            X_val_df = pd.DataFrame(X_val) if not isinstance(X_val, pd.DataFrame) else X_val
            X_test_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test

            val_metrics = self._evaluate(model, X_val_df, y_val_np, prefix="val")
            test_metrics = self._evaluate(model, X_test_df, y_test_np, prefix="test")
            self.tracker.log_metrics({**val_metrics, **test_metrics})

            if hasattr(model, "feature_importances_"):
                self.tracker.log_feature_importance(
                    feature_names, model.feature_importances_
                )
            elif hasattr(model, "coef_"):
                self.tracker.log_feature_importance(
                    feature_names, np.abs(model.coef_).flatten()
                )

            model_name = type(model).__name__
            self.tracker.log_model(model, model_name)

            self.tracker.tag_run(
                {
                    "model_type": model_name,
                    "pipeline": "training",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            run_id = run.info.run_id
            logger.info("Training run tracked: %s", run_id)
            return run_id

        except Exception:
            self.tracker.end_run(status="FAILED")
            raise
        else:
            self.tracker.end_run(status="FINISHED")

    def track_evaluation_run(
        self,
        model: Any,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        metrics: Optional[Dict[str, float]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> str:
        """Track a standalone evaluation run.

        Args:
            model: Fitted estimator.
            X_test: Test features.
            y_test: Test labels.
            metrics: Pre-computed metrics (re-computed if ``None``).
            feature_names: Optional list of feature names for importance logging.

        Returns:
            The MLflow run ID.
        """
        y_test_np = self._to_numpy(y_test)

        run = self.tracker.start_run(
            run_name=f"eval_{datetime.now():%Y%m%d_%H%M%S}"
        )
        try:
            X_test_df = (
                pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
            )

            if metrics is None:
                metrics = self._evaluate(model, X_test_df, y_test_np, prefix="test")
            self.tracker.log_metrics(metrics)

            model_name = type(model).__name__
            self.tracker.log_model(model, model_name)

            if feature_names and hasattr(model, "feature_importances_"):
                self.tracker.log_feature_importance(
                    feature_names, model.feature_importances_
                )

            self.tracker.tag_run(
                {
                    "model_type": model_name,
                    "pipeline": "evaluation",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            run_id = run.info.run_id
            logger.info("Evaluation run tracked: %s", run_id)
            return run_id

        except Exception:
            self.tracker.end_run(status="FAILED")
            raise
        else:
            self.tracker.end_run(status="FINISHED")

    def promote_model_if_best(
        self,
        run_id: str,
        metric: str = "val_auc",
        model_name: str = "default_prediction_model",
    ) -> bool:
        """Promote the model to production if this run is the new best.

        Args:
            run_id: The run to evaluate.
            metric: Metric to compare (higher is better).
            model_name: Registered model name in the MLflow Model Registry.

        Returns:
            ``True`` if the model was promoted, ``False`` otherwise.
        """
        best = self.tracker.get_best_run(metric=metric)
        if best is None:
            logger.info("No previous best run; promoting run %s", run_id)
            self.tracker.register_model(model_name, stage="production")
            return True

        if best["run_id"] == run_id:
            self.tracker.register_model(model_name, stage="production")
            logger.info("Run %s is the best – promoted to production", run_id)
            return True

        logger.info(
            "Run %s (%s=%.4f) did not beat best run %s (%s=%.4f) – no promotion",
            run_id,
            metric,
            self._get_metric_from_run(run_id, metric),
            best["run_id"],
            metric,
            best["metrics"].get(metric, float("nan")),
        )
        return False

    def generate_mlflow_report(self, output_path: str = "mlflow_report.html") -> str:
        """Generate a self-contained HTML report of recent experiments.

        Uses basic HTML templating (no external dependencies).

        Args:
            output_path: File path for the generated HTML report.

        Returns:
            The absolute path to the generated report.
        """
        df = self.tracker.compare_runs(n_runs=25)

        metric_cols = [c for c in df.columns if c not in ("run_id", "run_name", "status", "start_time", "duration_ms")]
        display_cols = ["run_id", "run_name", "status"] + metric_cols[:10]
        display_df = df[display_cols].copy()

        for col in display_df.columns:
            if display_df[col].dtype in (np.float64, np.float32):
                display_df[col] = display_df[col].round(4)

        table_html = display_df.to_html(index=False, classes="runs-table", border=0)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLflow Experiment Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         margin: 2rem; background: #fafafa; color: #333; }}
  h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 0.5rem; }}
  .runs-table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
  .runs-table th {{ background: #16213e; color: #fff; padding: 0.6rem 1rem; text-align: left; }}
  .runs-table td {{ padding: 0.5rem 1rem; border-bottom: 1px solid #ddd; }}
  .runs-table tr:hover {{ background: #e8f0fe; }}
  .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 1rem; }}
</style>
</head>
<body>
<h1>MLflow Experiment Report</h1>
<p class="meta">Generated: {datetime.now():%Y-%m-%d %H:%M:%S}</p>
<p class="meta">Experiment: {self.tracker.experiment_name} &middot; Runs shown: {len(df)}</p>
{table_html}
</body>
</html>"""

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        logger.info("MLflow report written to %s", out.resolve())
        return str(out.resolve())

    # -- Internal helpers -------------------------------------------------------

    @staticmethod
    def _evaluate(
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute classification metrics for a fitted model.

        Args:
            model: Fitted estimator with ``predict`` and ``predict_proba``.
            X: Feature matrix.
            y: Ground-truth labels.
            prefix: Optional prefix prepended to metric names.

        Returns:
            Dictionary of prefixed metric names to values.
        """
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y, y_pred, zero_division=0)),
            "brier_score": float(brier_score_loss(y, y_prob)),
        }

        if len(np.unique(y)) > 1:
            metrics["auc_roc"] = float(roc_auc_score(y, y_prob))
            metrics["auc_pr"] = float(average_precision_score(y, y_prob))
            metrics["log_loss"] = float(log_loss(y, y_prob))
        else:
            metrics["auc_roc"] = 0.5
            metrics["auc_pr"] = 0.0
            metrics["log_loss"] = float("inf")

        if prefix:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}

        return metrics

    def _get_metric_from_run(self, run_id: str, metric: str) -> float:
        """Retrieve a single metric value from a specific run.

        Args:
            run_id: MLflow run ID.
            metric: Metric name.

        Returns:
            Metric value or ``float('nan')`` if not found.
        """
        run = self.tracker.client.get_run(run_id)
        return run.data.metrics.get(metric, float("nan"))

    @staticmethod
    def _to_numpy(y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Convert to NumPy array.

        Args:
            y: Input labels.

        Returns:
            1-D NumPy array.
        """
        if isinstance(y, pd.Series):
            return y.values
        return np.asarray(y)
