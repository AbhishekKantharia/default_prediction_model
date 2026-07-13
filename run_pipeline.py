"""
Default Prediction Model - Main Pipeline
End-to-end pipeline that orchestrates data ingestion, preprocessing,
feature engineering, model training, evaluation, and interpretation.
"""
import os
import sys
import logging
import yaml
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.ingestion import DataIngestion
from src.data.preprocessing import (
    StructuredDataPreprocessor, UnstructuredDataProcessor, DataLoader
)
from src.features.engineering import FeatureEngineer
from src.models.training import ModelTrainer
from src.models.prediction import DefaultPredictor
from src.evaluation.metrics import ModelEvaluator
from src.interpretation.framework import ModelInterpreter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pipeline")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict):
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("DEFAULT PREDICTION MODEL — FULL PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Data Ingestion ──────────────────────────────────────
    logger.info("\n[1/7] DATA INGESTION")
    ingestion = DataIngestion(config)

    try:
        df = ingestion.load_structured_data("loan_data.csv")
        logger.info("Loaded production data")
    except FileNotFoundError:
        logger.info("No production data found — generating synthetic dataset")
        df = ingestion.generate_synthetic_dataset(n_samples=50000)

    df = ingestion.create_default_label(df)
    logger.info(f"Dataset shape: {df.shape}, Default rate: {df['default_status'].mean():.4f}")

    # ── Step 2: Data Preprocessing ──────────────────────────────────
    logger.info("\n[2/7] DATA PREPROCESSING")
    structured_preprocessor = StructuredDataPreprocessor(config)
    text_processor = UnstructuredDataProcessor()

    text_columns = [c for c in ["loan_description", "borrower_notes", "loan_title"]
                    if c in df.columns]
    if text_columns:
        df = text_processor.process_text_columns(df, text_columns)
        logger.info(f"Text features added — columns: "
                    f"{[c for c in df.columns if c.startswith('tf_')]}")

    df = structured_preprocessor.fit_transform(df)
    logger.info(f"After preprocessing: {df.shape}")

    # ── Step 3: Feature Engineering ─────────────────────────────────
    logger.info("\n[3/7] FEATURE ENGINEERING")
    feature_engineer = FeatureEngineer(config)
    df = feature_engineer.build_all_features(df)

    for col in ["loan_description", "borrower_notes", "loan_title",
                "risk_grade", "home_ownership", "employment_status",
                "verification_status", "addr_state", "loan_purpose",
                "loan_type", "borrower_segment"]:
        if col in df.columns and df[col].dtype == "object":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    logger.info(f"Final features: {df.shape[1]} columns")

    # ── Step 4: Data Splitting ──────────────────────────────────────
    logger.info("\n[4/7] DATA SPLITTING")
    data_loader = DataLoader(config)
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(df)
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ── Step 5: Model Training ──────────────────────────────────────
    logger.info("\n[5/7] MODEL TRAINING")
    trainer = ModelTrainer(config)
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    logger.info(f"Best model: {trainer.best_model_name}")

    # Cross-validation
    logger.info("\nCross-validation...")
    cv_scores = trainer.cross_validate_models(
        pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    # Loan-type models
    if "loan_type" in df.columns:
        logger.info("\nLoan-type-specific models...")
        loan_models = trainer.train_loan_type_models(
            df, config["data"]["target_column"])

    trainer.save_models()
    logger.info("Models saved")

    # ── Step 6: Model Evaluation ────────────────────────────────────
    logger.info("\n[6/7] MODEL EVALUATION")
    evaluator = ModelEvaluator(config)
    predictor = DefaultPredictor(config)

    best_model = trainer.models[trainer.best_model_name]
    test_probs = predictor.predict_probability(best_model, X_test)
    test_metrics = evaluator.compute_all_metrics(y_test, test_probs,
                                                 trainer.best_model_name)

    logger.info(f"\n{'='*40}")
    logger.info(f"TEST SET RESULTS — {trainer.best_model_name}")
    logger.info(f"{'='*40}")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    logger.info(f"  AUC-ROC:   {test_metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR:    {test_metrics['auc_pr']:.4f}")
    logger.info(f"  Brier:     {test_metrics['brier_score']:.4f}")

    # Threshold optimization
    thresholds = evaluator.find_optimal_thresholds(y_test.values, test_probs)
    logger.info(f"  Optimal thresholds: {thresholds}")

    # Evaluation report
    report = evaluator.generate_evaluation_report(
        y_test.values, test_probs, trainer.best_model_name)

    # Per-segment evaluation
    if "borrower_segment" in X_test.columns:
        segment_eval = evaluator.evaluate_by_segment(
            y_test.values, test_probs,
            df.loc[X_test.index, "borrower_segment"])
        logger.info(f"\nSegment evaluation:\n{segment_eval.to_string()}")

    # ── Step 7: Interpretation ──────────────────────────────────────
    logger.info("\n[7/7] MODEL INTERPRETATION")
    interpreter = ModelInterpreter(config)
    shap_values = interpreter.compute_shap_values(best_model, X_test)
    feature_imp = interpreter.compute_feature_importance(
        best_model, X_test, y_test)

    top_features = interpreter.get_top_features(20)
    logger.info(f"\nTop 10 features:\n{top_features.head(10).to_string()}")

    framework = interpreter.generate_consistent_framework(
        best_model, X_test)
    interpreter.save_interpretation("models/artifacts/interpretations.json")

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Best model: {trainer.best_model_name}")
    logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1_score']:.4f}")
    logger.info(f"Models saved to: {config['api']['model_path']}")
    logger.info("=" * 60)

    return {
        "metrics": report,
        "cv_scores": cv_scores,
        "top_features": top_features.to_dict(),
        "thresholds": thresholds,
        "framework": framework,
        "elapsed_time": elapsed
    }


if __name__ == "__main__":
    config = load_config()
    results = run_pipeline(config)

    with open("models/artifacts/pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
