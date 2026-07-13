"""
Data Preprocessing Module
Handles cleaning, encoding, imputation, and transformation of both
structured and unstructured data for the Default Prediction Model.
"""
import logging
from typing import Tuple, Dict, Optional, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, OneHotEncoder,
    OrdinalEncoder, PowerTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class StructuredDataPreprocessor:
    """Preprocesses structured (tabular) loan data."""

    def __init__(self, config: dict):
        self.config = config
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.power_transformers: Dict[str, PowerTransformer] = {}
        self.fitted = False

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna("Unknown")

        return df

    def remove_outliers_iqr(self, df: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            factor: float = 3.0) -> pd.DataFrame:
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - factor * IQR
                upper = Q3 + factor * IQR
                df[col] = df[col].clip(lower, upper)
        return df

    def encode_categoricals(self, df: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            method: str = "label") -> pd.DataFrame:
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue
            if method == "label":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            elif method == "ordinal":
                oe = OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1)
                df[col] = oe.fit_transform(df[[col]])
                self.encoders[col] = oe
        return df

    def scale_features(self, df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       method: str = "standard") -> pd.DataFrame:
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in df.columns:
                if method == "standard":
                    scaler = StandardScaler()
                    df[col] = scaler.fit_transform(df[[col]])
                    self.scalers[col] = scaler
                elif method == "power":
                    pt = PowerTransformer(method="yeo-johnson")
                    df[col] = pt.fit_transform(df[[col]])
                    self.power_transformers[col] = pt
        return df

    def apply_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        date_cols = ["issue_date", "last_payment_date", "next_payment_date",
                     "earliest_credit_line"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                df.drop(columns=[col], inplace=True)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Preprocessing structured data — shape: {df.shape}")

        df = self.handle_missing_values(df)
        df = self.apply_date_features(df)
        df = self.remove_outliers_iqr(df)
        df = self.encode_categoricals(df)

        self.fitted = True
        logger.info(f"Preprocessing complete — final shape: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        df = self.handle_missing_values(df)
        df = self.apply_date_features(df)
        df = self.remove_outliers_iqr(df)
        for col, le in self.encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col].astype(str))
                except Exception:
                    df[col] = -1
        return df


class UnstructuredDataProcessor:
    """Processes unstructured text data (loan descriptions, borrower notes)."""

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.tfidf_vectorizer = None

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        import re
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_text_features(self, text: str) -> Dict[str, float]:
        features = {}
        if not isinstance(text, str) or not text:
            return {
                "text_length": 0, "word_count": 0, "avg_word_length": 0,
                "num_uppercase": 0, "num_special_chars": 0,
                "sentiment_score": 0, "subjectivity": 0,
                "has_financial_terms": 0, "risk_keywords_count": 0
            }

        words = text.split()
        features["text_length"] = len(text)
        features["word_count"] = len(words)
        features["avg_word_length"] = np.mean([len(w) for w in words]) if words else 0
        features["num_uppercase"] = sum(1 for c in text if c.isupper())
        features["num_special_chars"] = sum(1 for c in text if not c.isalnum() and not c.isspace())

        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            features["sentiment_score"] = blob.sentiment.polarity
            features["subjectivity"] = blob.sentiment.subjectivity
        except ImportError:
            features["sentiment_score"] = 0
            features["subjectivity"] = 0

        financial_terms = ["income", "debt", "credit", "loan", "payment",
                           "interest", "balance", "default", "risk", "score"]
        features["has_financial_terms"] = int(
            any(t in text.lower() for t in financial_terms))

        risk_keywords = ["struggle", "difficulty", "hardship", "late",
                         "missed", "overdue", "delinquent", "past due"]
        features["risk_keywords_count"] = sum(
            1 for kw in risk_keywords if kw in text.lower())

        return features

    def process_text_columns(self, df: pd.DataFrame,
                             text_columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        text_features = []
        for idx, row in df.iterrows():
            combined_text = " ".join(
                str(row.get(col, "")) for col in text_columns)
            cleaned = self.clean_text(combined_text)
            feats = self.extract_text_features(cleaned)
            text_features.append(feats)

        text_df = pd.DataFrame(text_features, index=df.index)
        text_df.columns = [f"tf_{c}" for c in text_df.columns]
        return pd.concat([df, text_df], axis=1)

    def create_tfidf_features(self, df: pd.DataFrame,
                              text_columns: List[str],
                              max_features: int = 50) -> pd.DataFrame:
        from sklearn.feature_extraction.text import TfidfVectorizer

        df = df.copy()
        combined = df[text_columns].apply(
            lambda row: " ".join(str(v) for v in row), axis=1)
        combined = combined.fillna("")

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
            index=df.index
        )
        return pd.concat([df, tfidf_df], axis=1)


class DataLoader:
    """Splits data into train/validation/test sets with stratification."""

    def __init__(self, config: dict):
        self.config = config
        self.target_column = config["data"]["target_column"]

    def split_data(self, df: pd.DataFrame) -> Tuple:
        test_size = self.config["data"]["test_size"]
        val_size = self.config["data"]["validation_size"]
        seed = self.config["data"]["random_state"]

        X = df.drop(columns=[self.target_column, "loan_id"], errors="ignore")
        y = df[self.target_column]

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y)

        relative_val = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=relative_val,
            random_state=seed, stratify=y_trainval)

        logger.info(
            f"Data split — train: {X_train.shape}, val: {X_val.shape}, "
            f"test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test
