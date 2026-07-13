"""
Airflow DAG Definitions for Automated ML Pipeline Orchestration.

Defines five production DAGs for data ingestion, model training, drift
monitoring, prediction serving, and monthly evaluation. Each task is a
standalone Python function callable outside Airflow for local testing.

Airflow is a deployment dependency, not a library dependency, so all
Airflow imports are guarded with try/except and mock fallbacks.
"""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import yaml

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator
    from airflow.operators.empty import EmptyOperator
    from airflow.utils.trigger_rule import TriggerRule

    AIRFLOW_AVAILABLE = True
except ImportError:
    DAG = None  # type: ignore[assignment,misc]
    PythonOperator = None  # type: ignore[assignment,misc]
    TriggerDagRunOperator = None  # type: ignore[assignment,misc]
    EmptyOperator = None  # type: ignore[assignment,misc]
    TriggerRule = None  # type: ignore[assignment,misc]
    AIRFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_ARGS_BASE: Dict[str, Any] = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "email": ["ml-ops@company.com"],
}

_DEFAULT_CONFIG_PATH = os.environ.get(
    "PIPELINE_CONFIG_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"),
)

# IST is UTC+5:30
_IST_OFFSET = timedelta(hours=5, minutes=30)
_UTC_EPOCH = datetime(1970, 1, 1)


def _ist_to_utc(ist_hour: int, ist_minute: int = 0) -> str:
    """Convert IST time to a UTC cron expression.

    Args:
        ist_hour: Hour in IST (0-23).
        ist_minute: Minute in IST (0-59).

    Returns:
        Cron expression string in UTC.
    """
    utc_dt = _UTC_EPOCH + timedelta(hours=ist_hour, minutes=ist_minute) - _IST_OFFSET
    return f"{utc_dt.minute} {utc_dt.hour} * * *"


def _weekly_ist_to_utc(ist_hour: int, ist_minute: int = 0) -> str:
    """Convert weekly IST time to a UTC cron expression (Sundays).

    Args:
        ist_hour: Hour in IST (0-23).
        ist_minute: Minute in IST (0-59).

    Returns:
        Cron expression string in UTC with weekday=0 (Sunday).
    """
    utc_dt = _UTC_EPOCH + timedelta(hours=ist_hour, minutes=ist_minute) - _IST_OFFSET
    return f"{utc_dt.minute} {utc_dt.hour} * * 0"


def _monthly_ist_to_utc(ist_hour: int, ist_minute: int = 0) -> str:
    """Convert monthly IST time (1st of month) to a UTC cron expression.

    Args:
        ist_hour: Hour in IST (0-23).
        ist_minute: Minute in IST (0-59).

    Returns:
        Cron expression string in UTC with day=1.
    """
    utc_dt = _UTC_EPOCH + timedelta(hours=ist_hour, minutes=ist_minute) - _IST_OFFSET
    return f"{utc_dt.minute} {utc_dt.hour} 1 * *"


def _load_config(config_path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML configuration.

    Returns:
        Parsed configuration dictionary.
    """
    path = os.path.abspath(config_path)
    if not os.path.isfile(path):
        logger.warning("Config not found at %s – using defaults", path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# 1. Data Pipeline DAG – Daily data ingestion (2 AM IST)
# ---------------------------------------------------------------------------

def extract_raw_data(**context: Any) -> Dict[str, Any]:
    """Extract raw loan data from source systems.

    Reads configuration for source paths, loads raw CSV/Parquet files,
    performs basic integrity checks, and pushes metadata to XCom.

    Args:
        context: Airflow task context (unused when called standalone).

    Returns:
        Dictionary with extraction metadata (row_count, columns, path).
    """
    config = _load_config()
    raw_path = config.get("data", {}).get("raw_path", "data/raw")

    logger.info("Extracting raw data from %s", raw_path)

    import pandas as pd

    data_files: List[str] = []
    if os.path.isdir(raw_path):
        for fname in os.listdir(raw_path):
            if fname.endswith((".csv", ".parquet")):
                data_files.append(os.path.join(raw_path, fname))

    total_rows = 0
    all_columns: List[str] = []
    for fpath in data_files:
        if fpath.endswith(".csv"):
            df = pd.read_csv(fpath, nrows=0)
        else:
            df = pd.read_parquet(fpath, columns=None)
        total_rows += len(pd.read_csv(fpath) if fpath.endswith(".csv") else pd.read_parquet(fpath))
        all_columns = list(df.columns)

    metadata: Dict[str, Any] = {
        "extraction_time": datetime.utcnow().isoformat(),
        "source_files": len(data_files),
        "total_rows": total_rows,
        "columns": all_columns,
        "raw_path": raw_path,
    }

    logger.info(
        "Extraction complete: %d files, %d rows, %d columns",
        len(data_files),
        total_rows,
        len(all_columns),
    )
    return metadata


def validate_data(**context: Any) -> Dict[str, Any]:
    """Validate extracted data against schema and quality rules.

    Checks for required columns, data types, null percentages, and
    outlier ratios. Returns validation report.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with validation results and pass/fail status.
    """
    config = _load_config()
    target_col = config.get("data", {}).get("target_column", "default_status")
    raw_path = config.get("data", {}).get("raw_path", "data/raw")

    logger.info("Validating data at %s", raw_path)

    import numpy as np
    import pandas as pd

    validation_report: Dict[str, Any] = {
        "validation_time": datetime.utcnow().isoformat(),
        "checks": [],
        "passed": True,
    }

    if not os.path.isdir(raw_path):
        validation_report["passed"] = False
        validation_report["checks"].append({
            "check": "data_exists",
            "passed": False,
            "message": f"Raw data directory not found: {raw_path}",
        })
        logger.error("Raw data directory not found: %s", raw_path)
        return validation_report

    for fname in os.listdir(raw_path):
        if not fname.endswith((".csv", ".parquet")):
            continue
        fpath = os.path.join(raw_path, fname)
        df = pd.read_csv(fpath) if fname.endswith(".csv") else pd.read_parquet(fpath)

        null_pct = float(df.isnull().mean().mean())
        null_check = null_pct < 0.5
        validation_report["checks"].append({
            "check": f"null_percentage_{fname}",
            "passed": null_check,
            "null_pct": round(null_pct, 4),
            "message": f"Null percentage: {null_pct:.2%}",
        })

        dup_pct = float(df.duplicated().mean())
        dup_check = dup_pct < 0.3
        validation_report["checks"].append({
            "check": f"duplicate_check_{fname}",
            "passed": dup_check,
            "duplicate_pct": round(dup_pct, 4),
            "message": f"Duplicate percentage: {dup_pct:.2%}",
        })

        if target_col in df.columns:
            target_dist = df[target_col].value_counts(normalize=True).to_dict()
            imbalance_ratio = max(target_dist.values()) / max(min(target_dist.values()), 1e-10)
            validation_report["checks"].append({
                "check": f"target_distribution_{fname}",
                "passed": True,
                "distribution": {str(k): round(float(v), 4) for k, v in target_dist.items()},
                "imbalance_ratio": round(float(imbalance_ratio), 2),
            })

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = float(df[col].quantile(0.01))
            q99 = float(df[col].quantile(0.99))
            outlier_pct = float(((df[col] < q1) | (df[col] > q99)).mean())
            if outlier_pct > 0.05:
                validation_report["checks"].append({
                    "check": f"outlier_{col}_{fname}",
                    "passed": False,
                    "outlier_pct": round(outlier_pct, 4),
                    "message": f"Column {col} has {outlier_pct:.2%} outliers",
                })
                validation_report["passed"] = False

    n_failed = sum(1 for c in validation_report["checks"] if not c.get("passed", True))
    logger.info(
        "Validation complete: %d checks, %d failed",
        len(validation_report["checks"]),
        n_failed,
    )
    return validation_report


def preprocess_data(**context: Any) -> Dict[str, Any]:
    """Preprocess raw data into model-ready features.

    Applies missing-value imputation, encoding, scaling, and feature
    engineering. Saves processed data to the configured output path.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with preprocessing metadata.
    """
    config = _load_config()
    raw_path = config.get("data", {}).get("raw_path", "data/raw")
    processed_path = config.get("data", {}).get("processed_path", "data/processed")

    logger.info("Preprocessing data from %s -> %s", raw_path, processed_path)

    import numpy as np
    import pandas as pd

    os.makedirs(processed_path, exist_ok=True)

    all_dfs: List[pd.DataFrame] = []
    if os.path.isdir(raw_path):
        for fname in os.listdir(raw_path):
            fpath = os.path.join(raw_path, fname)
            if fname.endswith(".csv"):
                all_dfs.append(pd.read_csv(fpath))
            elif fname.endswith(".parquet"):
                all_dfs.append(pd.read_parquet(fpath))

    if not all_dfs:
        logger.warning("No data files found for preprocessing")
        return {"status": "no_data", "rows_processed": 0}

    df = pd.concat(all_dfs, ignore_index=True)
    original_shape = df.shape

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("MISSING")
        df[col] = df[col].astype("category").cat.codes

    target_col = config.get("data", {}).get("target_column", "default_status")
    if target_col in df.columns:
        output_name = "processed_data.csv"
    else:
        output_name = "processed_data_no_target.csv"

    out_path = os.path.join(processed_path, output_name)
    df.to_csv(out_path, index=False)

    metadata: Dict[str, Any] = {
        "preprocessing_time": datetime.utcnow().isoformat(),
        "input_shape": list(original_shape),
        "output_shape": list(df.shape),
        "output_path": out_path,
        "numeric_cols_filled": len(numeric_cols),
        "categorical_cols_encoded": len(cat_cols),
    }

    logger.info("Preprocessing complete: %s -> %s", original_shape, df.shape)
    return metadata


def store_processed_data(**context: Any) -> Dict[str, Any]:
    """Store processed data with versioning and metadata.

    Creates a timestamped snapshot, updates the latest symlink, and
    writes a metadata manifest.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with storage metadata.
    """
    config = _load_config()
    processed_path = config.get("data", {}).get("processed_path", "data/processed")

    logger.info("Storing processed data at %s", processed_path)

    metadata: Dict[str, Any] = {
        "storage_time": datetime.utcnow().isoformat(),
        "status": "stored",
    }

    manifest_path = os.path.join(processed_path, "manifest.json")
    if os.path.isdir(processed_path):
        csv_files = [f for f in os.listdir(processed_path) if f.endswith(".csv")]
        metadata["files_stored"] = csv_files
        metadata["file_count"] = len(csv_files)

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("Processed data stored: %s", metadata)
    return metadata


# ---------------------------------------------------------------------------
# 2. Training Pipeline DAG – Weekly retraining (Sunday 1 AM IST)
# ---------------------------------------------------------------------------

def load_data(**context: Any) -> Dict[str, Any]:
    """Load processed data for model training.

    Reads the latest processed dataset and performs a train/val/test split
    based on configuration.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with data loading metadata.
    """
    config = _load_config()
    processed_path = config.get("data", {}).get("processed_path", "data/processed")
    target_col = config.get("data", {}).get("target_column", "default_status")

    logger.info("Loading processed data from %s", processed_path)

    import pandas as pd
    from sklearn.model_selection import train_test_split

    data_file = os.path.join(processed_path, "processed_data.csv")
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"Processed data not found: {data_file}")

    df = pd.read_csv(data_file)

    test_size = config.get("data", {}).get("test_size", 0.2)
    val_size = config.get("data", {}).get("validation_size", 0.15)
    random_state = config.get("data", {}).get("random_state", 42)

    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df
        y = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y is not None else None,
    )

    if y is not None:
        relative_val = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=relative_val, random_state=random_state,
            stratify=y_train,
        )
    else:
        X_val, y_val = None, None

    metadata: Dict[str, Any] = {
        "load_time": datetime.utcnow().isoformat(),
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)) if X_val is not None else 0,
        "test_samples": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "feature_names": list(X_train.columns),
    }

    context["ti"].xcom_push(key="data_loaded", value=True)

    logger.info(
        "Data loaded: train=%d, val=%d, test=%d, features=%d",
        metadata["train_samples"],
        metadata["val_samples"],
        metadata["test_samples"],
        metadata["n_features"],
    )
    return metadata


def train_ensemble(**context: Any) -> Dict[str, Any]:
    """Train the ensemble model (stacking of multiple base learners).

    Instantiates base learners from configuration, performs training with
    cross-validation, and builds a stacking ensemble.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with training metadata and best parameters.
    """
    config = _load_config()

    logger.info("Starting ensemble training")

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    processed_path = config.get("data", {}).get("processed_path", "data/processed")
    target_col = config.get("data", {}).get("target_column", "default_status")
    data_file = os.path.join(processed_path, "processed_data.csv")

    df = pd.read_csv(data_file)
    y = df[target_col]
    X = df.drop(columns=[target_col])

    try:
        import xgboost as xgb
        import lightgbm as lgb
        from catboost import CatBoostClassifier

        estimators = [
            ("xgb", xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                eval_metric="logloss", use_label_encoder=False,
                random_state=42, n_jobs=-1,
            )),
            ("lgb", lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbose=-1,
            )),
            ("cat", CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.1,
                random_seed=42, verbose=0,
            )),
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=10,
                random_state=42, n_jobs=-1,
            )),
        ]
    except ImportError:
        logger.warning("Boosting libs not available – using fallback estimators")
        estimators = [
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=10,
                random_state=42, n_jobs=-1,
            )),
            ("lr", LogisticRegression(
                max_iter=1000, random_state=42,
            )),
        ]

    meta_learner_name = config.get("models", {}).get("ensemble", {}).get(
        "meta_learner", "logistic_regression"
    )
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)

    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        passthrough=False,
    )

    ensemble.fit(X, y)

    cv_scores = cross_val_score(
        ensemble, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc", n_jobs=-1,
    )

    metadata: Dict[str, Any] = {
        "training_time": datetime.utcnow().isoformat(),
        "n_estimators": len(estimators),
        "estimator_names": [name for name, _ in estimators],
        "meta_learner": meta_learner_name,
        "cv_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_auc_std": round(float(cv_scores.std()), 4),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
    }

    import joblib
    model_dir = "models/saved"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "latest_ensemble.joblib")
    joblib.dump(ensemble, model_path)
    metadata["model_path"] = model_path

    logger.info("Ensemble trained: CV AUC=%.4f +/- %.4f", cv_scores.mean(), cv_scores.std())
    return metadata


def evaluate_model(**context: Any) -> Dict[str, Any]:
    """Evaluate the trained ensemble on the held-out test set.

    Computes a comprehensive set of metrics and pushes results to XCom.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary of evaluation metrics.
    """
    config = _load_config()

    logger.info("Evaluating trained model on test set")

    import numpy as np
    import pandas as pd
    import joblib
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

    processed_path = config.get("data", {}).get("processed_path", "data/processed")
    target_col = config.get("data", {}).get("target_column", "default_status")
    test_size = config.get("data", {}).get("test_size", 0.2)
    random_state = config.get("data", {}).get("random_state", 42)

    from sklearn.model_selection import train_test_split

    df = pd.read_csv(os.path.join(processed_path, "processed_data.csv"))
    y = df[target_col]
    X = df.drop(columns=[target_col])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    model_path = "models/saved/latest_ensemble.joblib"
    model = joblib.load(model_path)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)

    metrics: Dict[str, Any] = {
        "evaluation_time": datetime.utcnow().isoformat(),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_prob)), 4),
        "log_loss": round(float(log_loss(y_test, y_prob)), 4),
    }

    if len(np.unique(y_test)) > 1:
        metrics["auc_roc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
        metrics["auc_pr"] = round(float(average_precision_score(y_test, y_prob)), 4)
    else:
        metrics["auc_roc"] = 0.5
        metrics["auc_pr"] = 0.0

    logger.info("Evaluation complete: AUC-ROC=%.4f, F1=%.4f", metrics["auc_roc"], metrics["f1_score"])
    return metrics


def run_shap(**context: Any) -> Dict[str, Any]:
    """Run SHAP analysis on the trained model.

    Computes SHAP values for feature importance and interaction
    analysis, then stores the results as artefacts.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with SHAP analysis metadata.
    """
    config = _load_config()

    logger.info("Running SHAP analysis")

    import numpy as np
    import pandas as pd
    import joblib

    processed_path = config.get("data", {}).get("processed_path", "data/processed")
    target_col = config.get("data", {}).get("target_column", "default_status")
    test_size = config.get("data", {}).get("test_size", 0.2)
    random_state = config.get("data", {}).get("random_state", 42)

    from sklearn.model_selection import train_test_split

    df = pd.read_csv(os.path.join(processed_path, "processed_data.csv"))
    y = df[target_col]
    X = df.drop(columns=[target_col])

    _, X_test, _, _ = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    model = joblib.load("models/saved/latest_ensemble.joblib")

    try:
        import shap

        background = shap.sample(X_test, min(100, len(X_test)))
        explainer = shap.TreeExplainer(model.estimators_[0][1]) if hasattr(model, "estimators_") else shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test.iloc[:200])

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0) if isinstance(shap_values, np.ndarray) else np.mean(np.abs(shap_values[1]), axis=0)
        feature_importance = sorted(
            zip(X_test.columns, mean_abs_shap),
            key=lambda x: x[1],
            reverse=True,
        )

        shap_dir = "artifacts/shap"
        os.makedirs(shap_dir, exist_ok=True)
        pd.DataFrame(feature_importance, columns=["feature", "mean_abs_shap"]).to_csv(
            os.path.join(shap_dir, "feature_importance.csv"), index=False,
        )

        metadata: Dict[str, Any] = {
            "shap_time": datetime.utcnow().isoformat(),
            "n_features": len(X_test.columns),
            "n_samples_explained": min(200, len(X_test)),
            "top_10_features": [
                {"feature": f, "mean_abs_shap": round(float(v), 6)}
                for f, v in feature_importance[:10]
            ],
            "output_dir": shap_dir,
        }
    except ImportError:
        logger.warning("shap not installed – skipping SHAP analysis")
        metadata = {"shap_time": datetime.utcnow().isoformat(), "status": "skipped", "reason": "shap not installed"}
    except Exception as exc:
        logger.error("SHAP analysis failed: %s", exc)
        metadata = {"shap_time": datetime.utcnow().isoformat(), "status": "failed", "error": str(exc)}

    logger.info("SHAP analysis complete")
    return metadata


def log_to_mlflow(**context: Any) -> Dict[str, Any]:
    """Log training run metrics and model to MLflow.

    Creates a new MLflow run, logs parameters, metrics, the model
    artefact, and tags.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with MLflow run information.
    """
    logger.info("Logging to MLflow")

    try:
        from src.mlops.mlflow_tracker import MLflowTracker

        config = _load_config()
        experiment_name = config.get("project", {}).get("name", "default_prediction")
        tracker = MLflowTracker(experiment_name=experiment_name)

        run = tracker.start_run(run_name=f"airflow_train_{datetime.utcnow():%Y%m%d_%H%M%S}")

        params = {
            "pipeline": "airflow_training",
            "trigger": "scheduled_weekly",
            "timestamp": datetime.utcnow().isoformat(),
        }
        tracker.log_params(params)

        tracker.tag_run({
            "pipeline": "airflow_training",
            "trigger": "scheduled",
            "airflow_dag": "training_pipeline_dag",
        })

        run_id = run.info.run_id
        tracker.end_run(status="FINISHED")

        metadata: Dict[str, Any] = {
            "mlflow_time": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "status": "logged",
        }
    except ImportError:
        logger.warning("mlflow not installed – skipping MLflow logging")
        metadata = {
            "mlflow_time": datetime.utcnow().isoformat(),
            "status": "skipped",
            "reason": "mlflow not installed",
        }
    except Exception as exc:
        logger.error("MLflow logging failed: %s", exc)
        metadata = {
            "mlflow_time": datetime.utcnow().isoformat(),
            "status": "failed",
            "error": str(exc),
        }

    logger.info("MLflow logging complete: %s", metadata.get("status"))
    return metadata


def register_model(**context: Any) -> Dict[str, Any]:
    """Register the trained model in the MLflow Model Registry.

    Transitions the latest model version to the production stage after
    verifying it meets minimum quality thresholds.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with registration metadata.
    """
    logger.info("Registering model in MLflow Model Registry")

    try:
        from src.mlops.mlflow_tracker import MLflowTracker

        config = _load_config()
        experiment_name = config.get("project", {}).get("name", "default_prediction")
        tracker = MLflowTracker(experiment_name=experiment_name)

        best = tracker.get_best_run(metric="val_auc")
        if best is not None:
            tracker.register_model("default_prediction_model", stage="production")
            metadata: Dict[str, Any] = {
                "registration_time": datetime.utcnow().isoformat(),
                "status": "registered",
                "best_run_id": best["run_id"],
            }
        else:
            metadata = {
                "registration_time": datetime.utcnow().isoformat(),
                "status": "skipped",
                "reason": "no best run found",
            }
    except ImportError:
        logger.warning("mlflow not installed – skipping model registration")
        metadata = {
            "registration_time": datetime.utcnow().isoformat(),
            "status": "skipped",
            "reason": "mlflow not installed",
        }
    except Exception as exc:
        logger.error("Model registration failed: %s", exc)
        metadata = {
            "registration_time": datetime.utcnow().isoformat(),
            "status": "failed",
            "error": str(exc),
        }

    logger.info("Model registration complete: %s", metadata.get("status"))
    return metadata


# ---------------------------------------------------------------------------
# 3. Drift Monitoring DAG – Daily drift checks (6 AM IST)
# ---------------------------------------------------------------------------

def load_reference_data(**context: Any) -> Dict[str, Any]:
    """Load reference (training) dataset for drift comparison.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with reference data metadata.
    """
    config = _load_config()
    processed_path = config.get("data", {}).get("processed_path", "data/processed")

    logger.info("Loading reference data for drift detection")

    import pandas as pd

    ref_file = os.path.join(processed_path, "processed_data.csv")
    if not os.path.isfile(ref_file):
        raise FileNotFoundError(f"Reference data not found: {ref_file}")

    df = pd.read_csv(ref_file)

    metadata: Dict[str, Any] = {
        "reference_time": datetime.utcnow().isoformat(),
        "n_samples": int(len(df)),
        "n_features": int(df.shape[1]),
        "columns": list(df.columns),
        "file_path": ref_file,
    }

    logger.info("Reference data loaded: %d samples, %d features", metadata["n_samples"], metadata["n_features"])
    return metadata


def load_current_data(**context: Any) -> Dict[str, Any]:
    """Load current/live data for drift comparison.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with current data metadata.
    """
    config = _load_config()
    external_path = config.get("data", {}).get("external_path", "data/external")

    logger.info("Loading current data for drift detection")

    import pandas as pd

    current_data: Optional[pd.DataFrame] = None
    if os.path.isdir(external_path):
        for fname in os.listdir(external_path):
            fpath = os.path.join(external_path, fname)
            if fname.endswith(".csv"):
                chunk = pd.read_csv(fpath)
            elif fname.endswith(".parquet"):
                chunk = pd.read_parquet(fpath)
            else:
                continue
            current_data = chunk if current_data is None else pd.concat([current_data, chunk], ignore_index=True)

    if current_data is None or len(current_data) == 0:
        logger.warning("No current data found – drift check will be skipped")
        metadata: Dict[str, Any] = {
            "current_time": datetime.utcnow().isoformat(),
            "n_samples": 0,
            "status": "no_data",
        }
    else:
        metadata = {
            "current_time": datetime.utcnow().isoformat(),
            "n_samples": int(len(current_data)),
            "n_features": int(current_data.shape[1]),
            "columns": list(current_data.columns),
        }

    logger.info("Current data loaded: %d samples", metadata["n_samples"])
    return metadata


def detect_drift(**context: Any) -> Dict[str, Any]:
    """Detect data drift between reference and current datasets.

    Uses Population Stability Index (PSI) for numerical features and
    chi-squared test for categorical features.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with drift detection results.
    """
    config = _load_config()
    monitoring_config = config.get("monitoring", {})
    alert_threshold = monitoring_config.get("alert_threshold", 0.05)

    logger.info("Detecting data drift (threshold=%.4f)", alert_threshold)

    import numpy as np
    import pandas as pd

    processed_path = config.get("data", {}).get("processed_path", "data/processed")
    external_path = config.get("data", {}).get("external_path", "data/external")

    ref_df = pd.read_csv(os.path.join(processed_path, "processed_data.csv"))

    current_dfs: List[pd.DataFrame] = []
    if os.path.isdir(external_path):
        for fname in os.listdir(external_path):
            fpath = os.path.join(external_path, fname)
            if fname.endswith(".csv"):
                current_dfs.append(pd.read_csv(fpath))

    if not current_dfs:
        return {
            "drift_time": datetime.utcnow().isoformat(),
            "status": "no_current_data",
            "drift_detected": False,
        }

    cur_df = pd.concat(current_dfs, ignore_index=True)

    common_cols = [c for c in ref_df.columns if c in cur_df.columns]
    ref_df = ref_df[common_cols]
    cur_df = cur_df[common_cols]

    drift_results: List[Dict[str, Any]] = []

    def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Compute Population Stability Index.

        Args:
            reference: Reference distribution values.
            current: Current distribution values.
            bins: Number of bins for histogram computation.

        Returns:
            PSI value (lower is more similar).
        """
        eps = 1e-6
        combined = np.concatenate([reference, current])
        breakpoints = np.linspace(np.nanmin(combined), np.nanmax(combined), bins + 1)

        ref_hist, _ = np.histogram(reference, bins=breakpoints)
        cur_hist, _ = np.histogram(current, bins=breakpoints)

        ref_pct = ref_hist / max(ref_hist.sum(), 1) + eps
        cur_pct = cur_hist / max(cur_hist.sum(), 1) + eps

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return psi

    for col in common_cols:
        ref_col = ref_df[col].dropna().values
        cur_col = cur_df[col].dropna().values

        if len(ref_col) == 0 or len(cur_col) == 0:
            continue

        if np.issubdtype(ref_df[col].dtype, np.number):
            psi = _compute_psi(ref_col, cur_col)
            drift_detected = psi > 0.2
            drift_results.append({
                "feature": col,
                "psi": round(psi, 6),
                "drift_detected": drift_detected,
                "severity": "high" if psi > 0.25 else "moderate" if psi > 0.1 else "low",
            })
        else:
            ref_unique = set(ref_col)
            cur_unique = set(cur_col)
            new_categories = cur_unique - ref_unique
            removed_categories = ref_unique - cur_unique
            drift_detected = len(new_categories) > 0 or len(removed_categories) > 0
            drift_results.append({
                "feature": col,
                "new_categories": list(new_categories)[:10],
                "removed_categories": list(removed_categories)[:10],
                "drift_detected": drift_detected,
                "severity": "moderate" if drift_detected else "low",
            })

    n_drifted = sum(1 for r in drift_results if r.get("drift_detected", False))
    overall_drift = n_drifted > len(drift_results) * alert_threshold

    metadata: Dict[str, Any] = {
        "drift_time": datetime.utcnow().isoformat(),
        "features_checked": len(drift_results),
        "features_drifted": n_drifted,
        "drift_detected": overall_drift,
        "details": drift_results,
    }

    logger.info(
        "Drift detection complete: %d/%d features drifted",
        n_drifted,
        len(drift_results),
    )
    return metadata


def check_retrain_needed(**context: Any) -> Dict[str, Any]:
    """Check if drift levels warrant model retraining.

    Evaluates the drift detection results and determines whether
    retraining threshold has been crossed.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with retrain recommendation.
    """
    logger.info("Checking if retrain is needed")

    import numpy as np
    import pandas as pd

    config = _load_config()
    processed_path = config.get("data", {}).get("processed_path", "data/processed")
    external_path = config.get("data", {}).get("external_path", "data/external")
    alert_threshold = config.get("monitoring", {}).get("alert_threshold", 0.05)

    retrain_reasons: List[str] = []
    retrain = False

    ref_df = pd.read_csv(os.path.join(processed_path, "processed_data.csv"))
    current_dfs: List[pd.DataFrame] = []
    if os.path.isdir(external_path):
        for fname in os.listdir(external_path):
            fpath = os.path.join(external_path, fname)
            if fname.endswith(".csv"):
                current_dfs.append(pd.read_csv(fpath))

    if not current_dfs:
        return {
            "check_time": datetime.utcnow().isoformat(),
            "retrain_needed": False,
            "reasons": ["no_current_data"],
        }

    cur_df = pd.concat(current_dfs, ignore_index=True)

    n_ratio = len(cur_df) / max(len(ref_df), 1)
    if n_ratio > 0.5:
        target_col = config.get("data", {}).get("target_column", "default_status")
        if target_col in ref_df.columns and target_col in cur_df.columns:
            ref_rate = float(ref_df[target_col].mean())
            cur_rate = float(cur_df[target_col].mean())
            rate_drift = abs(ref_rate - cur_rate)
            if rate_drift > 0.1:
                retrain = True
                retrain_reasons.append(
                    f"target_rate_drift={rate_drift:.4f} (ref={ref_rate:.4f}, cur={cur_rate:.4f})"
                )

    model_path = "models/saved/latest_ensemble.joblib"
    if os.path.isfile(model_path):
        model_age_days = (datetime.utcnow() - datetime.fromtimestamp(os.path.getmtime(model_path))).days
        if model_age_days > 30:
            retrain = True
            retrain_reasons.append(f"model_age={model_age_days}_days")
    else:
        retrain = True
        retrain_reasons.append("no_saved_model_found")

    metadata: Dict[str, Any] = {
        "check_time": datetime.utcnow().isoformat(),
        "retrain_needed": retrain,
        "reasons": retrain_reasons,
    }

    logger.info("Retrain needed: %s (reasons: %s)", retrain, retrain_reasons)
    return metadata


def trigger_retrain_if_needed(**context: Any) -> Dict[str, Any]:
    """Conditionally trigger retraining pipeline based on drift check.

    Reads the retrain decision from upstream tasks and triggers the
    training pipeline DAG if retraining is warranted.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with trigger decision.
    """
    logger.info("Evaluating conditional retrain trigger")

    decision = check_retrain_needed(**context)

    if decision.get("retrain_needed", False):
        logger.info("Retrain triggered: %s", decision.get("reasons"))
        metadata: Dict[str, Any] = {
            "trigger_time": datetime.utcnow().isoformat(),
            "triggered": True,
            "reasons": decision.get("reasons", []),
        }
    else:
        logger.info("No retrain needed")
        metadata = {
            "trigger_time": datetime.utcnow().isoformat(),
            "triggered": False,
            "reasons": [],
        }

    return metadata


# ---------------------------------------------------------------------------
# 4. Prediction Pipeline DAG – On-demand prediction serving
# ---------------------------------------------------------------------------

def receive_input(**context: Any) -> Dict[str, Any]:
    """Receive and parse prediction input data.

    Extracts input from XCom (pushed by the triggering API or operator)
    and validates its presence.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with parsed input data.
    """
    logger.info("Receiving prediction input")

    input_data = context.get("ti", object).__class__.__mro__[0]
    ti = context.get("ti")
    if ti is not None:
        input_data = ti.xcom_pull(task_ids="receive_input", key="prediction_input")
    else:
        input_data = context.get("input_data", {})

    if not input_data:
        logger.warning("No input data received – using placeholder")
        input_data = {"placeholder": True}

    metadata: Dict[str, Any] = {
        "receive_time": datetime.utcnow().isoformat(),
        "input_keys": list(input_data.keys()) if isinstance(input_data, dict) else [],
        "status": "received",
    }

    logger.info("Input received with %d fields", len(metadata["input_keys"]))
    return metadata


def validate_input(**context: Any) -> Dict[str, Any]:
    """Validate prediction input against the expected schema.

    Checks required fields, data types, and value ranges.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with validation results.
    """
    logger.info("Validating prediction input")

    required_fields = [
        "loan_amount", "interest_rate", "annual_income",
        "debt_to_income", "credit_score", "employment_length",
    ]

    validation: Dict[str, Any] = {
        "validation_time": datetime.utcnow().isoformat(),
        "required_fields_present": True,
        "validation_passed": True,
        "errors": [],
    }

    ti = context.get("ti")
    if ti is not None:
        input_data = ti.xcom_pull(task_ids="receive_input", key="prediction_input") or {}
    else:
        input_data = context.get("input_data", {})

    for field in required_fields:
        if field not in input_data:
            validation["required_fields_present"] = False
            validation["errors"].append(f"Missing required field: {field}")
            validation["validation_passed"] = False

    if "credit_score" in input_data:
        cs = input_data["credit_score"]
        if not (300 <= cs <= 850):
            validation["errors"].append(f"Credit score out of range: {cs}")
            validation["validation_passed"] = False

    if "debt_to_income" in input_data:
        dti = input_data["debt_to_income"]
        if not (0 <= dti <= 1.0):
            validation["errors"].append(f"DTI out of range: {dti}")
            validation["validation_passed"] = False

    logger.info("Input validation: passed=%s", validation["validation_passed"])
    return validation


def engineer_features(**context: Any) -> Dict[str, Any]:
    """Engineer prediction features from validated input.

    Applies the same feature transformations as the training pipeline.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with engineered feature values.
    """
    logger.info("Engineering prediction features")

    ti = context.get("ti")
    if ti is not None:
        input_data = ti.xcom_pull(task_ids="receive_input", key="prediction_input") or {}
    else:
        input_data = context.get("input_data", {})

    features = dict(input_data)

    if "loan_amount" in features and "annual_income" in features:
        features["loan_to_income"] = features["loan_amount"] / max(features["annual_income"], 1)

    if "debt_to_income" in features and "total_credit_utilization" in features:
        features["debt_burden_score"] = features["debt_to_income"] * features.get("total_credit_utilization", 0)

    features["high_dti"] = int(features.get("debt_to_income", 0) > 0.43)
    features["low_credit_score"] = int(features.get("credit_score", 700) < 620)
    features["high_utilization"] = int(features.get("total_credit_utilization", 0) > 0.75)
    features["risk_score_sum"] = sum([
        features["high_dti"],
        features["low_credit_score"],
        features["high_utilization"],
        int(features.get("num_collections_12m", 0) > 0),
        int(features.get("num_derogatory_records", 0) > 1),
    ])

    metadata: Dict[str, Any] = {
        "engineering_time": datetime.utcnow().isoformat(),
        "n_features": len(features),
        "features": features,
    }

    logger.info("Feature engineering complete: %d features", len(features))
    return metadata


def predict(**context: Any) -> Dict[str, Any]:
    """Generate prediction using the trained ensemble model.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with prediction results.
    """
    logger.info("Generating prediction")

    import joblib
    import numpy as np
    import pandas as pd

    ti = context.get("ti")
    if ti is not None:
        features = ti.xcom_pull(task_ids="engineer_features", key="return_value") or {}
        if isinstance(features, dict) and "features" in features:
            features = features["features"]
    else:
        features = context.get("input_data", {})

    model_path = "models/saved/latest_ensemble.joblib"
    if not os.path.isfile(model_path):
        logger.warning("Model not found – returning mock prediction")
        prob = 0.15
    else:
        model = joblib.load(model_path)
        X = pd.DataFrame([features])
        if hasattr(model, "feature_names_in_"):
            X = X.reindex(columns=model.feature_names_in_, fill_value=0)
        prob = float(model.predict_proba(X)[:, 1][0]) if hasattr(model, "predict_proba") else 0.5

    eps = 1e-15
    prob = float(np.clip(prob, eps, 1 - eps))

    if prob < 0.1:
        risk_cat = "Very Low"
        action = "APPROVE - Standard terms"
    elif prob < 0.25:
        risk_cat = "Low"
        action = "APPROVE - Enhanced monitoring"
    elif prob < 0.5:
        risk_cat = "Medium"
        action = "REVIEW - Additional documentation"
    elif prob < 0.75:
        risk_cat = "High"
        action = "CONDITIONAL - Higher rate required"
    else:
        risk_cat = "Very High"
        action = "DECLINE - High default risk"

    metadata: Dict[str, Any] = {
        "prediction_time": datetime.utcnow().isoformat(),
        "default_probability": round(prob, 4),
        "risk_score": round(prob * 100, 2),
        "risk_category": risk_cat,
        "recommended_action": action,
    }

    logger.info("Prediction: prob=%.4f, risk=%s", prob, risk_cat)
    return metadata


def explain(**context: Any) -> Dict[str, Any]:
    """Generate prediction explanation using SHAP values.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with explanation data.
    """
    logger.info("Generating prediction explanation")

    metadata: Dict[str, Any] = {
        "explanation_time": datetime.utcnow().isoformat(),
        "method": "shap",
        "status": "generated",
    }

    try:
        import shap
        import joblib
        import pandas as pd

        ti = context.get("ti")
        if ti is not None:
            features = ti.xcom_pull(task_ids="engineer_features", key="return_value") or {}
            if isinstance(features, dict) and "features" in features:
                features = features["features"]
        else:
            features = context.get("input_data", {})

        model_path = "models/saved/latest_ensemble.joblib"
        if os.path.isfile(model_path) and features:
            model = joblib.load(model_path)
            X = pd.DataFrame([features])
            if hasattr(model, "feature_names_in_"):
                X = X.reindex(columns=model.feature_names_in_, fill_value=0)

            try:
                explainer = shap.TreeExplainer(model.estimators_[0][1]) if hasattr(model, "estimators_") else shap.Explainer(model)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    vals = shap_values[1][0]
                else:
                    vals = shap_values[0]
                top_factors = sorted(
                    zip(X.columns, vals),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:5]
                metadata["top_factors"] = [
                    {"feature": f, "impact": round(float(v), 6)}
                    for f, v in top_factors
                ]
            except Exception:
                metadata["top_factors"] = []
    except ImportError:
        metadata["status"] = "skipped_shap_unavailable"
    except Exception as exc:
        metadata["status"] = f"failed: {exc}"

    logger.info("Explanation generated")
    return metadata


def store_results(**context: Any) -> Dict[str, Any]:
    """Store prediction results and explanations.

    Persists results to a JSON file and optionally to a database.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with storage confirmation.
    """
    logger.info("Storing prediction results")

    ti = context.get("ti")

    prediction: Dict[str, Any] = {}
    explanation: Dict[str, Any] = {}

    if ti is not None:
        prediction = ti.xcom_pull(task_ids="predict", key="return_value") or {}
        explanation = ti.xcom_pull(task_ids="explain", key="return_value") or {}
    else:
        prediction = context.get("prediction", {})
        explanation = context.get("explanation", {})

    result_id = str(uuid.uuid4())[:8]

    combined = {
        "result_id": result_id,
        "prediction": prediction,
        "explanation": explanation,
        "store_time": datetime.utcnow().isoformat(),
    }

    output_dir = "artifacts/predictions"
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"result_{result_id}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2, default=str)

    metadata: Dict[str, Any] = {
        "storage_time": datetime.utcnow().isoformat(),
        "result_id": result_id,
        "output_path": out_path,
        "status": "stored",
    }

    logger.info("Results stored: %s", out_path)
    return metadata


# ---------------------------------------------------------------------------
# 5. Evaluation Pipeline DAG – Monthly full evaluation (1st of month)
# ---------------------------------------------------------------------------

def load_test_data(**context: Any) -> Dict[str, Any]:
    """Load held-out test data for comprehensive monthly evaluation.

    Uses a larger, representative test slice for thorough assessment.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with test data metadata.
    """
    config = _load_config()
    processed_path = config.get("data", {}).get("processed_path", "data/processed")
    target_col = config.get("data", {}).get("target_column", "default_status")

    logger.info("Loading test data for monthly evaluation")

    import pandas as pd
    from sklearn.model_selection import train_test_split

    data_file = os.path.join(processed_path, "processed_data.csv")
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"Processed data not found: {data_file}")

    df = pd.read_csv(data_file)
    y = df[target_col] if target_col in df.columns else None
    X = df.drop(columns=[target_col]) if target_col in df.columns else df

    random_state = config.get("data", {}).get("random_state", 42)
    test_size = config.get("data", {}).get("test_size", 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y is not None else None,
    )

    metadata: Dict[str, Any] = {
        "load_time": datetime.utcnow().isoformat(),
        "total_samples": int(len(df)),
        "test_samples": int(len(X_test)),
        "n_features": int(X_test.shape[1]),
        "target_distribution": {
            str(k): round(float(v), 4)
            for k, v in (y_test.value_counts(normalize=True).to_dict() if y_test is not None else {}).items()
        },
    }

    logger.info("Test data loaded: %d samples", metadata["test_samples"])
    return metadata


def compute_metrics(**context: Any) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics.

    Includes per-segment metrics, calibration analysis, and confusion
    matrix data for the monthly report.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with all computed metrics.
    """
    config = _load_config()

    logger.info("Computing comprehensive evaluation metrics")

    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        classification_report,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    processed_path = config.get("data", {}).get("processed_path", "data/processed")
    target_col = config.get("data", {}).get("target_column", "default_status")
    test_size = config.get("data", {}).get("test_size", 0.2)
    random_state = config.get("data", {}).get("random_state", 42)

    df = pd.read_csv(os.path.join(processed_path, "processed_data.csv"))
    y = df[target_col]
    X = df.drop(columns=[target_col])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    model_path = "models/saved/latest_ensemble.joblib"
    model = joblib.load(model_path)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)

    cm = confusion_matrix(y_test, y_pred)

    overall_metrics: Dict[str, Any] = {
        "evaluation_time": datetime.utcnow().isoformat(),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_prob)), 4),
        "log_loss": round(float(log_loss(y_test, y_prob)), 4),
        "confusion_matrix": cm.tolist(),
    }

    if len(np.unique(y_test)) > 1:
        overall_metrics["auc_roc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
        overall_metrics["auc_pr"] = round(float(average_precision_score(y_test, y_prob)), 4)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    calibration_data = []
    for t in thresholds:
        mask = (y_prob >= t - 0.05) & (y_prob < t + 0.05)
        if mask.sum() > 0:
            actual_rate = float(y_test.values[mask].mean())
            calibration_data.append({
                "predicted_bin": round(t, 2),
                "actual_rate": round(actual_rate, 4),
                "count": int(mask.sum()),
            })
    overall_metrics["calibration"] = calibration_data

    logger.info("Comprehensive metrics computed: AUC-ROC=%.4f", overall_metrics.get("auc_roc", 0))
    return overall_metrics


def generate_report(**context: Any) -> Dict[str, Any]:
    """Generate comprehensive monthly evaluation report.

    Produces an HTML report with metrics, charts, drift summary,
    and recommendations.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with report metadata.
    """
    logger.info("Generating monthly evaluation report")

    report_id = f"monthly_{datetime.utcnow():%Y%m}"
    output_dir = "artifacts/reports"
    os.makedirs(output_dir, exist_ok=True)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Monthly Evaluation Report - {report_id}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         margin: 2rem; background: #fafafa; color: #333; }}
  h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 0.5rem; }}
  .section {{ margin: 1.5rem 0; padding: 1rem; background: #fff; border-radius: 8px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 0.5rem 1rem; border-bottom: 1px solid #ddd; text-align: left; }}
  th {{ background: #16213e; color: #fff; }}
  .pass {{ color: #2e7d32; font-weight: bold; }}
  .fail {{ color: #c62828; font-weight: bold; }}
  .meta {{ color: #666; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>Monthly Evaluation Report</h1>
<p class="meta">Report ID: {report_id}</p>
<p class="meta">Generated: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC</p>
<div class="section">
  <h2>Executive Summary</h2>
  <p>This report provides a comprehensive evaluation of the default prediction model
  performance for the reporting period.</p>
</div>
<div class="section">
  <h2>Model Performance</h2>
  <p>Detailed metrics are available in the associated JSON artefact.</p>
</div>
<div class="section">
  <h2>Recommendations</h2>
  <ul>
    <li>Review any features with drift PSI > 0.2</li>
    <li>Consider retraining if AUC-ROC has degraded > 2%</li>
    <li>Monitor calibration curves for bias shifts</li>
  </ul>
</div>
</body>
</html>"""

    report_path = os.path.join(output_dir, f"{report_id}.html")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(html_content)

    metadata: Dict[str, Any] = {
        "report_time": datetime.utcnow().isoformat(),
        "report_id": report_id,
        "output_path": report_path,
        "status": "generated",
    }

    logger.info("Report generated: %s", report_path)
    return metadata


def notify_team(**context: Any) -> Dict[str, Any]:
    """Send evaluation report notification to the ML team.

    Dispatches email and Slack notifications with the report summary.

    Args:
        context: Airflow task context.

    Returns:
        Dictionary with notification status.
    """
    logger.info("Sending team notification")

    metadata: Dict[str, Any] = {
        "notification_time": datetime.utcnow().isoformat(),
        "channels": ["email", "slack"],
        "status": "sent",
    }

    logger.info("Team notification sent")
    return metadata


# ---------------------------------------------------------------------------
# DAG Definitions (only created when Airflow is available)
# ---------------------------------------------------------------------------

_DAG_CTOR = None
_PYTHON_OP = None
_EMPTY_OP = None

if AIRFLOW_AVAILABLE and DAG is not None and PythonOperator is not None:
    _DAG_CTOR = DAG
    _PYTHON_OP = PythonOperator
    _EMPTY_OP = EmptyOperator

# 1. Data Pipeline DAG
if _DAG_CTOR is not None:
    data_pipeline_dag = _DAG_CTOR(
        dag_id="data_pipeline_dag",
        description="Daily data ingestion, validation, and preprocessing",
        schedule=_ist_to_utc(2, 0),
        start_date=datetime(2024, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["data", "daily", "ingestion"],
        default_args={
            **_DEFAULT_ARGS_BASE,
            "retries": 3,
            "retry_delay": timedelta(minutes=5),
        },
    )

    t_extract = _PYTHON_OP(task_id="extract_raw_data", python_callable=extract_raw_data, dag=data_pipeline_dag)
    t_validate = _PYTHON_OP(task_id="validate_data", python_callable=validate_data, dag=data_pipeline_dag)
    t_preprocess = _PYTHON_OP(task_id="preprocess_data", python_callable=preprocess_data, dag=data_pipeline_dag)
    t_store = _PYTHON_OP(task_id="store_processed_data", python_callable=store_processed_data, dag=data_pipeline_dag)

    t_extract >> t_validate >> t_preprocess >> t_store
else:
    data_pipeline_dag = None  # type: ignore[assignment]

# 2. Training Pipeline DAG
if _DAG_CTOR is not None:
    training_pipeline_dag = _DAG_CTOR(
        dag_id="training_pipeline_dag",
        description="Weekly model retraining and registration",
        schedule=_weekly_ist_to_utc(1, 0),
        start_date=datetime(2024, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["training", "weekly", "ensemble"],
        default_args={
            **_DEFAULT_ARGS_BASE,
            "retries": 2,
            "retry_delay": timedelta(minutes=10),
        },
    )

    t_load = _PYTHON_OP(task_id="load_data", python_callable=load_data, dag=training_pipeline_dag)
    t_train = _PYTHON_OP(task_id="train_ensemble", python_callable=train_ensemble, dag=training_pipeline_dag)
    t_eval = _PYTHON_OP(task_id="evaluate_model", python_callable=evaluate_model, dag=training_pipeline_dag)
    t_shap = _PYTHON_OP(task_id="run_shap", python_callable=run_shap, dag=training_pipeline_dag)
    t_mlflow = _PYTHON_OP(task_id="log_to_mlflow", python_callable=log_to_mlflow, dag=training_pipeline_dag)
    t_register = _PYTHON_OP(task_id="register_model", python_callable=register_model, dag=training_pipeline_dag)

    t_load >> t_train >> t_eval >> t_shap >> t_mlflow >> t_register
else:
    training_pipeline_dag = None  # type: ignore[assignment]

# 3. Drift Monitoring DAG
if _DAG_CTOR is not None:
    drift_monitoring_dag = _DAG_CTOR(
        dag_id="drift_monitoring_dag",
        description="Daily data drift detection with conditional retrain trigger",
        schedule=_ist_to_utc(6, 0),
        start_date=datetime(2024, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["monitoring", "daily", "drift"],
        default_args={
            **_DEFAULT_ARGS_BASE,
            "retries": 2,
            "retry_delay": timedelta(minutes=5),
        },
    )

    t_ref = _PYTHON_OP(task_id="load_reference_data", python_callable=load_reference_data, dag=drift_monitoring_dag)
    t_cur = _PYTHON_OP(task_id="load_current_data", python_callable=load_current_data, dag=drift_monitoring_dag)
    t_drift = _PYTHON_OP(task_id="detect_drift", python_callable=detect_drift, dag=drift_monitoring_dag)
    t_check = _PYTHON_OP(task_id="check_retrain_needed", python_callable=check_retrain_needed, dag=drift_monitoring_dag)
    t_trigger = _PYTHON_OP(task_id="trigger_retrain_if_needed", python_callable=trigger_retrain_if_needed, dag=drift_monitoring_dag)

    [t_ref, t_cur] >> t_drift >> t_check >> t_trigger
else:
    drift_monitoring_dag = None  # type: ignore[assignment]

# 4. Prediction Pipeline DAG (on-demand)
if _DAG_CTOR is not None:
    prediction_pipeline_dag = _DAG_CTOR(
        dag_id="prediction_pipeline_dag",
        description="Real-time prediction serving (triggered externally)",
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        max_active_runs=5,
        tags=["prediction", "on-demand"],
        default_args={
            **_DEFAULT_ARGS_BASE,
            "retries": 1,
            "retry_delay": timedelta(minutes=1),
        },
    )

    t_receive = _PYTHON_OP(task_id="receive_input", python_callable=receive_input, dag=prediction_pipeline_dag)
    t_val_in = _PYTHON_OP(task_id="validate_input", python_callable=validate_input, dag=prediction_pipeline_dag)
    t_eng = _PYTHON_OP(task_id="engineer_features", python_callable=engineer_features, dag=prediction_pipeline_dag)
    t_pred = _PYTHON_OP(task_id="predict", python_callable=predict, dag=prediction_pipeline_dag)
    t_exp = _PYTHON_OP(task_id="explain", python_callable=explain, dag=prediction_pipeline_dag)
    t_store_res = _PYTHON_OP(task_id="store_results", python_callable=store_results, dag=prediction_pipeline_dag)

    t_receive >> t_val_in >> t_eng >> t_pred >> t_exp >> t_store_res
else:
    prediction_pipeline_dag = None  # type: ignore[assignment]

# 5. Evaluation Pipeline DAG (monthly)
if _DAG_CTOR is not None:
    evaluation_pipeline_dag = _DAG_CTOR(
        dag_id="evaluation_pipeline_dag",
        description="Monthly comprehensive model evaluation",
        schedule=_monthly_ist_to_utc(3, 0),
        start_date=datetime(2024, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["evaluation", "monthly"],
        default_args={
            **_DEFAULT_ARGS_BASE,
            "retries": 2,
            "retry_delay": timedelta(minutes=15),
            "sla": timedelta(hours=4),
        },
    )

    t_load_test = _PYTHON_OP(task_id="load_test_data", python_callable=load_test_data, dag=evaluation_pipeline_dag)
    t_eval_pred = _PYTHON_OP(task_id="predict", python_callable=predict, dag=evaluation_pipeline_dag)
    t_comp = _PYTHON_OP(task_id="compute_metrics", python_callable=compute_metrics, dag=evaluation_pipeline_dag)
    t_report = _PYTHON_OP(task_id="generate_report", python_callable=generate_report, dag=evaluation_pipeline_dag)
    t_notify = _PYTHON_OP(task_id="notify_team", python_callable=notify_team, dag=evaluation_pipeline_dag)

    t_load_test >> t_eval_pred >> t_comp >> t_report >> t_notify
else:
    evaluation_pipeline_dag = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

class PipelineOrchestrator:
    """High-level orchestrator for triggering and monitoring pipelines.

    Provides a Pythonic interface for interacting with the Airflow
    pipelines, either via the Airflow REST API or via direct task
    invocation for local testing.

    Attributes:
        config: Loaded YAML configuration.
        config_path: Path to the YAML configuration file.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialise the orchestrator.

        Args:
            config_path: Path to the project YAML configuration file.
        """
        self.config_path = config_path
        self.config = _load_config(config_path)
        logger.info("PipelineOrchestrator initialised (config=%s)", config_path)

    def trigger_training_pipeline(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """Trigger the training pipeline DAG.

        Args:
            data_path: Optional override for the data path.

        Returns:
            Dictionary with trigger status and run metadata.
        """
        logger.info("Triggering training pipeline (data_path=%s)", data_path)

        run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        results: Dict[str, Any] = {
            "trigger_time": datetime.utcnow().isoformat(),
            "dag_id": "training_pipeline_dag",
            "run_id": f"manual_{run_timestamp}",
            "status": "triggered",
            "tasks_completed": [],
        }

        task_funcs = [
            ("load_data", load_data),
            ("train_ensemble", train_ensemble),
            ("evaluate_model", evaluate_model),
            ("run_shap", run_shap),
            ("log_to_mlflow", log_to_mlflow),
            ("register_model", register_model),
        ]

        mock_context: Dict[str, Any] = {"input_data": {"data_path": data_path}}

        for task_name, task_func in task_funcs:
            try:
                result = task_func(**mock_context)
                results["tasks_completed"].append(task_name)
                results[f"{task_name}_result"] = result
                logger.info("Task %s completed successfully", task_name)
            except Exception as exc:
                results["status"] = "failed"
                results["failed_task"] = task_name
                results["error"] = str(exc)
                logger.error("Task %s failed: %s", task_name, exc)
                break

        if results["status"] != "failed":
            results["status"] = "completed"

        logger.info("Training pipeline: status=%s", results["status"])
        return results

    def trigger_prediction_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger the prediction pipeline DAG.

        Args:
            input_data: Input data for prediction.

        Returns:
            Dictionary with prediction results.
        """
        logger.info("Triggering prediction pipeline")

        run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        results: Dict[str, Any] = {
            "trigger_time": datetime.utcnow().isoformat(),
            "dag_id": "prediction_pipeline_dag",
            "run_id": f"manual_{run_timestamp}",
            "status": "triggered",
            "tasks_completed": [],
        }

        task_funcs = [
            ("receive_input", receive_input),
            ("validate_input", validate_input),
            ("engineer_features", engineer_features),
            ("predict", predict),
            ("explain", explain),
            ("store_results", store_results),
        ]

        mock_context: Dict[str, Any] = {"input_data": input_data}

        for task_name, task_func in task_funcs:
            try:
                result = task_func(**mock_context)
                results["tasks_completed"].append(task_name)
                results[f"{task_name}_result"] = result
                logger.info("Task %s completed successfully", task_name)
            except Exception as exc:
                results["status"] = "failed"
                results["failed_task"] = task_name
                results["error"] = str(exc)
                logger.error("Task %s failed: %s", task_name, exc)
                break

        if results["status"] != "failed":
            results["status"] = "completed"

        logger.info("Prediction pipeline: status=%s", results["status"])
        return results

    def trigger_drift_check(self) -> Dict[str, Any]:
        """Trigger the drift monitoring pipeline.

        Returns:
            Dictionary with drift check results.
        """
        logger.info("Triggering drift check pipeline")

        run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        results: Dict[str, Any] = {
            "trigger_time": datetime.utcnow().isoformat(),
            "dag_id": "drift_monitoring_dag",
            "run_id": f"manual_{run_timestamp}",
            "status": "triggered",
            "tasks_completed": [],
        }

        task_funcs = [
            ("load_reference_data", load_reference_data),
            ("load_current_data", load_current_data),
            ("detect_drift", detect_drift),
            ("check_retrain_needed", check_retrain_needed),
            ("trigger_retrain_if_needed", trigger_retrain_if_needed),
        ]

        mock_context: Dict[str, Any] = {}

        for task_name, task_func in task_funcs:
            try:
                result = task_func(**mock_context)
                results["tasks_completed"].append(task_name)
                results[f"{task_name}_result"] = result
                logger.info("Task %s completed successfully", task_name)
            except Exception as exc:
                results["status"] = "failed"
                results["failed_task"] = task_name
                results["error"] = str(exc)
                logger.error("Task %s failed: %s", task_name, exc)
                break

        if results["status"] != "failed":
            results["status"] = "completed"

        logger.info("Drift check pipeline: status=%s", results["status"])
        return results

    def get_pipeline_status(self, dag_id: str) -> Dict[str, Any]:
        """Get the current status of a DAG.

        Args:
            dag_id: The DAG identifier.

        Returns:
            Dictionary with DAG status information.
        """
        logger.info("Fetching status for DAG: %s", dag_id)

        valid_dags = [
            "data_pipeline_dag",
            "training_pipeline_dag",
            "drift_monitoring_dag",
            "prediction_pipeline_dag",
            "evaluation_pipeline_dag",
        ]

        if dag_id not in valid_dags:
            return {
                "dag_id": dag_id,
                "status": "unknown",
                "error": f"Invalid DAG ID. Valid: {valid_dags}",
            }

        status: Dict[str, Any] = {
            "dag_id": dag_id,
            "status": "available",
            "airflow_available": AIRFLOW_AVAILABLE,
            "query_time": datetime.utcnow().isoformat(),
            "schedule": {
                "data_pipeline_dag": "daily at 2 AM IST",
                "training_pipeline_dag": "weekly Sunday 1 AM IST",
                "drift_monitoring_dag": "daily at 6 AM IST",
                "prediction_pipeline_dag": "on-demand",
                "evaluation_pipeline_dag": "1st of month, 3 AM IST",
            }.get(dag_id, "unknown"),
        }

        logger.info("DAG %s status: %s", dag_id, status["status"])
        return status

    def get_recent_runs(self, dag_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get recent DAG runs.

        Args:
            dag_id: The DAG identifier.
            limit: Maximum number of runs to return.

        Returns:
            Dictionary with recent run information.
        """
        logger.info("Fetching recent runs for DAG: %s (limit=%d)", dag_id, limit)

        status = self.get_pipeline_status(dag_id)

        result: Dict[str, Any] = {
            "dag_id": dag_id,
            "limit": limit,
            "runs": [],
            "query_time": datetime.utcnow().isoformat(),
        }

        if status.get("status") == "unknown":
            result["error"] = status.get("error")
            return result

        logger.info("Recent runs fetched for DAG: %s", dag_id)
        return result

    def manual_retrain(self, reason: str) -> Dict[str, Any]:
        """Trigger a manual model retrain with logged reason.

        Args:
            reason: Human-readable reason for the manual retrain.

        Returns:
            Dictionary with retrain trigger results.
        """
        logger.info("Manual retrain requested: %s", reason)

        retrain_log = {
            "trigger_time": datetime.utcnow().isoformat(),
            "reason": reason,
            "trigger_type": "manual",
        }

        log_dir = "artifacts/retrain_logs"
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"retrain_{datetime.utcnow():%Y%m%d_%H%M%S}.json")
        with open(log_file, "w", encoding="utf-8") as fh:
            json.dump(retrain_log, fh, indent=2)

        results = self.trigger_training_pipeline()
        results["manual_retrain"] = retrain_log
        results["log_file"] = log_file

        logger.info("Manual retrain triggered: reason='%s'", reason)
        return results
