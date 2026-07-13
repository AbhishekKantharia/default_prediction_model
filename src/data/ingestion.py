"""
Data Ingestion Module
Handles loading of structured data (CSV, Parquet), unstructured text data,
and external reference datasets for the Default Prediction Model.
"""
import os
import logging
from typing import Tuple, Dict, Optional, List
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataIngestion:
    """Central data ingestion class for structured and unstructured data."""

    def __init__(self, config: dict):
        self.config = config
        self.raw_path = Path(config["data"]["raw_path"])
        self.processed_path = Path(config["data"]["processed_path"])
        self.target_column = config["data"]["target_column"]
        self.prediction_horizon = config["data"]["prediction_horizon_months"]

    def load_structured_data(self, filename: str) -> pd.DataFrame:
        filepath = self.raw_path / filename
        ext = filepath.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(filepath, low_memory=False)
        elif ext == ".parquet":
            df = pd.read_parquet(filepath)
        elif ext in (".xls", ".xlsx"):
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        logger.info(f"Loaded structured data: {filepath} — shape {df.shape}")
        return df

    def load_text_data(self, filename: str) -> pd.DataFrame:
        filepath = self.raw_path / filename
        df = pd.read_csv(filepath, low_memory=False)
        text_cols = [c for c in df.columns if any(k in c.lower() for k in
                     ["description", "notes", "title", "text", "comment"])]
        logger.info(f"Loaded text data with columns: {text_cols}")
        return df

    def load_external_data(self, filename: str) -> pd.DataFrame:
        filepath = Path(self.config["data"]["external_path"]) / filename
        df = pd.read_csv(filepath)
        logger.info(f"Loaded external data: {filepath}")
        return df

    def merge_datasets(self, structured: pd.DataFrame, text: pd.DataFrame,
                       on: str = "loan_id") -> pd.DataFrame:
        merged = structured.merge(text, on=on, how="left", suffixes=("", "_text"))
        logger.info(f"Merged dataset shape: {merged.shape}")
        return merged

    def create_default_label(self, df: pd.DataFrame,
                             months_col: str = "months_since_issue",
                             status_col: str = "loan_status") -> pd.DataFrame:
        """
        Create binary default label based on the prediction horizon.
        A loan is labeled as default (1) if it charged off or was 90+ days
        delinquent within the prediction horizon window.
        """
        default_statuses = [
            "Charged Off", "Default", "Late (16-30 days)",
            "Late (31-120 days)", "In Grace Period",
            "Default Registered", "Defaulted"
        ]
        mask_horizon = df[months_col] <= self.prediction_horizon
        mask_default = df[status_col].isin(default_statuses)
        df[self.target_column] = (mask_horizon & mask_default).astype(int)
        default_rate = df[self.target_column].mean()
        logger.info(f"Default label created — rate: {default_rate:.4f}")
        return df

    def generate_synthetic_dataset(self, n_samples: int = 50000,
                                   seed: int = 42) -> pd.DataFrame:
        """
        Generate a synthetic loan dataset for development/testing.
        Simulates realistic distributions for all feature categories.
        """
        rng = np.random.RandomState(seed)
        n = n_samples

        data = {
            "loan_id": range(1, n + 1),
            "loan_amount": rng.lognormal(mean=9.5, sigma=0.6, size=n).astype(int),
            "interest_rate": rng.uniform(3.5, 28.0, n).round(2),
            "annual_income": rng.lognormal(mean=11, sigma=0.5, size=n).astype(int),
            "debt_to_income": rng.beta(2, 5, n).round(4),
            "credit_score": rng.normal(680, 80, n).clip(300, 850).astype(int),
            "employment_length": rng.choice(
                [0, 1, 2, 3, 5, 7, 10, 15, 20], size=n,
                p=[0.05, 0.08, 0.10, 0.12, 0.15, 0.15, 0.15, 0.12, 0.08]),
            "months_since_last_delinquency": rng.exponential(24, n).clip(0, 120).astype(int),
            "open_credit_lines": rng.poisson(8, n).clip(1, 30),
            "total_credit_utilization": rng.beta(3, 3, n).round(4),
            "revolving_balance": rng.lognormal(mean=8, sigma=1.2, size=n).astype(int),
            "installment_amount": rng.lognormal(mean=6.5, sigma=0.5, size=n).astype(int),
            "annual_payment": rng.lognormal(mean=8.5, sigma=0.6, size=n).astype(int),
            "total_revolving_bal": rng.lognormal(mean=8.5, sigma=1.0, size=n).astype(int),
            "total_acc": rng.poisson(15, n).clip(1, 60),
            "num_derogatory_records": rng.poisson(0.5, n).clip(0, 10),
            "num_collections_12m": rng.poisson(0.3, n).clip(0, 5),
            "total_current_balance": rng.lognormal(mean=10, sigma=1.0, size=n).astype(int),
            "total_revolving_credit_limit": rng.lognormal(mean=9.5, sigma=0.8, size=n).astype(int),
            "months_since_first_credit_line": rng.uniform(6, 360, n).astype(int),
            "delinquency_2yrs": rng.poisson(0.8, n).clip(0, 10),
            "loan_type": rng.choice(["personal", "mortgage", "auto", "business",
                                     "student", "credit_card"], size=n,
                                    p=[0.25, 0.20, 0.18, 0.12, 0.10, 0.15]),
            "home_ownership": rng.choice(["OWN", "MORTGAGE", "RENT", "OTHER"],
                                         size=n, p=[0.10, 0.45, 0.40, 0.05]),
            "loan_purpose": rng.choice(["debt_consolidation", "credit_card", "home_improvement",
                                        "major_purchase", "medical", "small_business",
                                        "education", "other"], size=n),
            "employment_status": rng.choice(["employed", "self_employed", "retired",
                                             "unemployed"], size=n,
                                            p=[0.65, 0.15, 0.12, 0.08]),
            "verification_status": rng.choice(["verified", "source_verified", "not_verified"],
                                              size=n, p=[0.35, 0.30, 0.35]),
            "addr_state": rng.choice(["CA", "NY", "TX", "FL", "IL", "PA", "OH",
                                      "GA", "NC", "MI", "Other"], size=n),
            "borrower_segment": rng.choice(["prime", "subprime", "near_prime", "deep_subprime"],
                                           size=n, p=[0.35, 0.20, 0.30, 0.15]),
            "risk_grade": rng.choice(["A", "B", "C", "D", "E", "F", "G"],
                                     size=n, p=[0.15, 0.25, 0.25, 0.18, 0.10, 0.05, 0.02]),
            "loan_description": [f"Loan for {purpose} - {segment} segment"
                                 for purpose, segment in zip(
                                     rng.choice(["debt consolidation", "home improvement",
                                                 "business expansion", "education", "medical"],
                                                size=n),
                                     rng.choice(["prime", "subprime", "near_prime"], size=n))],
            "borrower_notes": [f"Borrower has {rng.choice(['stable', 'variable', 'improving'])} "
                               f"income with {rng.choice(['good', 'fair', 'poor'])} credit history"
                               for _ in range(n)],
            "loan_title": rng.choice(["Personal Loan", "Home Loan", "Auto Loan",
                                      "Business Loan", "Education Loan"], size=n),
            "issue_date": pd.date_range("2020-01-01", periods=n, freq="h"),
            "last_payment_date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "next_payment_date": pd.date_range("2024-06-01", periods=n, freq="D"),
            "earliest_credit_line": pd.date_range("2005-01-01", periods=n, freq="3D"),
            "months_since_issue": rng.uniform(1, 60, n).astype(int),
            "loan_status": rng.choice(
                ["Fully Paid", "Current", "Late (16-30 days)", "Late (31-120 days)",
                 "Charged Off", "Default", "In Grace Period"],
                size=n, p=[0.35, 0.30, 0.08, 0.07, 0.10, 0.05, 0.05]),
        }

        df = pd.DataFrame(data)
        df["annual_income"] = np.maximum(df["annual_income"], 20000)
        df["debt_to_income"] = np.clip(df["debt_to_income"], 0, 1)
        df["total_credit_utilization"] = np.clip(df["total_credit_utilization"], 0, 1)

        logger.info(f"Generated synthetic dataset: {df.shape}")
        return df

    def get_train_test_split_config(self) -> dict:
        return {
            "test_size": self.config["data"]["test_size"],
            "validation_size": self.config["data"]["validation_size"],
            "random_state": self.config["data"]["random_state"]
        }
