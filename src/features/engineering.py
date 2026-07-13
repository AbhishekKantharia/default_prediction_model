"""
Feature Engineering Module
Creates advanced features for default prediction including interaction terms,
aggregations, temporal features, and loan-type-specific transformations.
"""
import logging
from typing import List, Optional, Dict

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for default prediction."""

    def __init__(self, config: dict):
        self.config = config
        self.feature_names: List[str] = []
        self.feature_importance_: Optional[pd.DataFrame] = None

    def create_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["loan_to_income"] = np.where(
            df["annual_income"] > 0,
            df["loan_amount"] / df["annual_income"], 0)
        df["installment_to_income"] = np.where(
            df["annual_income"] > 0,
            (df["installment_amount"] * 12) / df["annual_income"], 0)
        df["revolving_utilization"] = np.where(
            df["total_revolving_credit_limit"] > 0,
            df["total_revolving_bal"] / df["total_revolving_credit_limit"], 0)
        df["balance_to_credit_limit"] = np.where(
            df["total_revolving_credit_limit"] > 0,
            df["total_current_balance"] / df["total_revolving_credit_limit"], 0)
        df["income_per_account"] = np.where(
            df["total_acc"] > 0,
            df["annual_income"] / df["total_acc"], 0)
        df["debt_burden_score"] = (
            df["debt_to_income"] * df["total_credit_utilization"])
        df["credit_efficiency"] = np.where(
            df["open_credit_lines"] > 0,
            df["total_acc"] / df["open_credit_lines"], 0)

        logger.info("Financial ratios created")
        return df

    def create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["high_dti"] = (df["debt_to_income"] > 0.43).astype(int)
        df["low_credit_score"] = (df["credit_score"] < 620).astype(int)
        df["high_utilization"] = (df["total_credit_utilization"] > 0.75).astype(int)
        df["short_credit_history"] = (
            df["months_since_first_credit_line"] < 24).astype(int)
        df["many_delinquencies"] = (df["delinquency_2yrs"] > 2).astype(int)
        df["recent_delinquency"] = (
            df["months_since_last_delinquency"] < 6).astype(int)
        df["high_collections"] = (df["num_collections_12m"] > 0).astype(int)
        df["high_derogatory"] = (df["num_derogatory_records"] > 1).astype(int)
        df["new_borrower"] = (
            df["employment_length"] <= 1).astype(int)

        risk_cols = [
            "high_dti", "low_credit_score", "high_utilization",
            "short_credit_history", "many_delinquencies",
            "recent_delinquency", "high_collections",
            "high_derogatory", "new_borrower"]
        df["risk_score_sum"] = df[risk_cols].sum(axis=1)

        logger.info("Risk indicators created")
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["credit_x_income"] = df["credit_score"] * df["annual_income"]
        df["dti_x_utilization"] = (
            df["debt_to_income"] * df["total_credit_utilization"])
        df["loan_x_interest"] = df["loan_amount"] * df["interest_rate"]
        df["income_stability"] = (
            df["employment_length"] * df["annual_income"])
        df["credit_momentum"] = np.where(
            df["months_since_last_delinquency"] > 0,
            df["credit_score"] / df["months_since_last_delinquency"], 0)
        df["debt_capacity"] = np.where(
            df["debt_to_income"] > 0,
            df["annual_income"] * (1 - df["debt_to_income"]), 0)

        logger.info("Interaction features created")
        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "months_since_issue" in df.columns:
            df["loan_age"] = df["months_since_issue"]
            df["loan_age_bucket"] = pd.cut(
                df["months_since_issue"],
                bins=[0, 6, 12, 24, 36, 60],
                labels=[0, 1, 2, 3, 4]).astype(float)

        if "months_since_first_credit_line" in df.columns:
            df["credit_history_length"] = df["months_since_first_credit_line"]
            df["credit_history_bucket"] = pd.cut(
                df["months_since_first_credit_line"],
                bins=[0, 12, 24, 60, 120, 360],
                labels=[0, 1, 2, 3, 4]).astype(float)

        logger.info("Temporal features created")
        return df

    def create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        group_cols = {
            "loan_type": "lt_",
            "borrower_segment": "seg_",
            "risk_grade": "rg_"
        }
        numeric_cols = [
            "loan_amount", "interest_rate", "credit_score",
            "debt_to_income"]

        for col, prefix in group_cols.items():
            if col in df.columns:
                for num_col in numeric_cols:
                    if num_col in df.columns:
                        grouped = df.groupby(col)[num_col].agg(
                            ["mean", "std", "median"])
                        grouped.columns = [
                            f"{prefix}{num_col}_{stat}"
                            for stat in ["mean", "std", "median"]]
                        df = df.merge(grouped, on=col, how="left")

        logger.info("Aggregation features created")
        return df

    def create_polynomial_features(self, df: pd.DataFrame,
                                   columns: Optional[List[str]] = None,
                                   degree: int = 2) -> pd.DataFrame:
        if columns is None:
            columns = ["credit_score", "debt_to_income",
                       "total_credit_utilization"]

        existing_cols = [c for c in columns if c in df.columns]
        if not existing_cols:
            return df

        poly = PolynomialFeatures(
            degree=degree, interaction_only=False, include_bias=False)
        poly_features = poly.fit_transform(df[existing_cols].fillna(0))
        feature_names = poly.get_feature_names_out(existing_cols)

        new_features = pd.DataFrame(
            poly_features[:, len(existing_cols):],
            columns=[f"poly_{n}" for n in feature_names[len(existing_cols):]],
            index=df.index
        )
        df = pd.concat([df, new_features], axis=1)
        logger.info(f"Polynomial features created: {new_features.shape[1]}")
        return df

    def create_loan_type_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "loan_type" in df.columns:
            # Mortgage-specific
            mortgage_mask = df["loan_type"] == "mortgage"
            if mortgage_mask.any():
                df.loc[mortgage_mask, "home_equity_ratio"] = np.where(
                    df.loc[mortgage_mask, "annual_income"] > 0,
                    df.loc[mortgage_mask, "loan_amount"] / (
                        df.loc[mortgage_mask, "annual_income"] * 4), 0)

            # Auto-specific
            auto_mask = df["loan_type"] == "auto"
            if auto_mask.any():
                df.loc[auto_mask, "auto_loan_burden"] = np.where(
                    df.loc[auto_mask, "annual_income"] > 0,
                    df.loc[auto_mask, "installment_amount"] * 12 / (
                        df.loc[auto_mask, "annual_income"]), 0)

            # Business-specific
            biz_mask = df["loan_type"] == "business"
            if biz_mask.any():
                df.loc[biz_mask, "business_risk_score"] = (
                    df.loc[biz_mask, "debt_to_income"] * (
                        1 + df.loc[biz_mask, "delinquency_2yrs"]))

        logger.info("Loan-type-specific features created")
        return df

    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Building all features — input shape: {df.shape}")

        df = self.create_financial_ratios(df)
        df = self.create_risk_indicators(df)
        df = self.create_interaction_features(df)
        df = self.create_temporal_features(df)
        df = self.create_loan_type_specific_features(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != self.config["data"]["target_column"]]
        df = self.create_aggregation_features(df)

        df = df.replace([np.inf, -np.inf], 0).fillna(0)

        self.feature_names = df.columns.tolist()
        logger.info(f"Feature engineering complete — output shape: {df.shape}")
        return df
