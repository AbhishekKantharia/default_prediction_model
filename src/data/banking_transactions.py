"""
Banking Transactions Module
Handles integration of real banking transaction data for credit risk prediction.
Covers Open Banking APIs, bank statement parsing, UPI data, CIBIL reports,
Account Aggregator framework, and comprehensive transaction feature engineering.
"""
import csv
import hashlib
import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TransactionType(str, Enum):
    CREDIT = "credit"
    DEBIT = "debit"
    EMI_PAYMENT = "emi_payment"
    INTEREST = "interest"
    FEE = "fee"


class TransactionCategory(str, Enum):
    SALARY = "salary"
    GROCERIES = "groceries"
    RENT = "rent"
    ENTERTAINMENT = "entertainment"
    BUSINESS = "business"
    UTILITIES = "utilities"
    MEDICAL = "medical"
    EDUCATION = "education"
    TRANSFER = "transfer"
    INVESTMENT = "investment"
    OTHER = "other"


class ChannelType(str, Enum):
    NEFT = "NEFT"
    RTGS = "RTGS"
    UPI = "UPI"
    IMPS = "IMPS"
    CHEQUE = "CHEQUE"
    CASH = "CASH"
    ATM = "ATM"
    CARD = "CARD"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    PENDING = "pending"
    FAILED = "failed"


SUPPORTED_BANKS = [
    "HDFC", "ICICI", "SBI", "Axis", "Kotak", "PNB", "BoB",
]

UPI_MERCHANT_KEYWORDS = [
    "paytm", "phonepe", "googlepay", "gpay", "amazonpay", "freecharge",
    "mobikwik", "slice", "cRED", "payzapp", "bhim",
]

INDIAN_SEASONS = {
    "diwali": [10, 11],
    "wedding": [11, 12, 1, 2, 3],
    "summer_travel": [4, 5, 6],
    "back_to_school": [3, 4, 5],
    "year_end": [12],
    "festival_of_holi": [3],
    "republic_day_sale": [1],
}


# ---------------------------------------------------------------------------
# Pydantic Schema
# ---------------------------------------------------------------------------

class BankingTransaction(BaseModel):
    """Comprehensive schema for an Indian banking transaction."""

    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    account_id: str
    loan_id: Optional[str] = None
    transaction_date: datetime
    transaction_type: TransactionType
    amount: float = Field(ge=0, description="Absolute transaction amount")
    currency: str = "INR"
    balance_after: float
    category: TransactionCategory = TransactionCategory.OTHER
    merchant_name: Optional[str] = None
    merchant_category: Optional[str] = None
    description: Optional[str] = None
    counterparty_account: Optional[str] = None
    is_recurring: bool = False
    is_domestic: bool = True
    channel: ChannelType = ChannelType.NEFT
    verification_status: VerificationStatus = VerificationStatus.VERIFIED
    fraud_flag: bool = False
    reconciled: bool = True

    @field_validator("amount")
    @classmethod
    def amount_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("amount must be >= 0 (use transaction_type for direction)")
        return v

    @field_validator("currency")
    @classmethod
    def currency_must_be_inr(cls, v: str) -> str:
        if v.upper() != "INR":
            raise ValueError(f"Only INR currency supported, got {v}")
        return v.upper()

    model_config = {"extra": "forbid", "validate_assignment": True}


# ---------------------------------------------------------------------------
# Banking Data Ingestor
# ---------------------------------------------------------------------------

class BankingDataIngestor:
    """Ingest banking data from APIs, statements, and credit bureaus."""

    SUPPORTED_BANKS = SUPPORTED_BANKS

    def __init__(self) -> None:
        self._api_sessions: Dict[str, Any] = {}

    # -- Open Banking / AA ---------------------------------------------------

    def connect_to_banking_api(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to an Open Banking API via the RBI Account Aggregator framework.

        Parameters
        ----------
        config : dict
            Must contain ``bank_name``, ``api_base_url``, ``client_id``,
            ``client_secret``, and optionally ``redirect_uri``.

        Returns
        -------
        dict with ``status`` and ``session_token``.
        """
        bank = config.get("bank_name", "")
        if bank not in self.SUPPORTED_BANKS:
            raise ValueError(
                f"Unsupported bank '{bank}'. Supported: {self.SUPPORTED_BANKS}"
            )
        api_base = config.get("api_base_url", "")
        client_id = config.get("client_id", "")
        client_secret = config.get("client_secret", "")
        if not all([api_base, client_id, client_secret]):
            raise ValueError("api_base_url, client_id, and client_secret are required")

        session_token = hashlib.sha256(
            f"{client_id}:{client_secret}:{bank}".encode()
        ).hexdigest()
        self._api_sessions[bank] = {
            "token": session_token,
            "base_url": api_base,
            "connected_at": datetime.utcnow().isoformat(),
        }
        logger.info("Connected to %s Open Banking API", bank)
        return {"status": "connected", "session_token": session_token}

    def fetch_transactions(
        self,
        account_id: str,
        start_date: datetime,
        end_date: datetime,
        bank: str = "HDFC",
    ) -> List[BankingTransaction]:
        """Fetch transactions from a connected bank API.

        In production this would call the actual AA-consented endpoint.
        Here we return a stub indicating the request parameters so downstream
        code can work with the schema immediately.
        """
        if bank not in self._api_sessions:
            raise RuntimeError(
                f"Not connected to {bank}. Call connect_to_banking_api first."
            )
        logger.info(
            "Fetching transactions for account %s from %s to %s via %s",
            account_id, start_date.date(), end_date.date(), bank,
        )
        return []

    # -- Bank statement parsing -----------------------------------------------

    def parse_bank_statement(self, file_path: str) -> List[BankingTransaction]:
        """Parse a CSV bank statement into BankingTransaction objects.

        Handles common Indian bank CSV formats (HDFC, ICICI, SBI, Axis).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Statement not found: {file_path}")
        if path.suffix.lower() not in (".csv", ".txt"):
            raise ValueError(f"Unsupported statement format: {path.suffix}")

        transactions: List[BankingTransaction] = []
        text = path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(StringIO(text))

        for row in reader:
            txn = self._row_to_transaction(row)
            if txn is not None:
                transactions.append(txn)
        logger.info("Parsed %d transactions from %s", len(transactions), file_path)
        return transactions

    def _row_to_transaction(self, row: Dict[str, str]) -> Optional[BankingTransaction]:
        """Best-effort conversion of a CSV row to a BankingTransaction."""
        date_str = (
            row.get("Date") or row.get("Transaction Date") or row.get("Txn Date") or ""
        ).strip()
        desc = (row.get("Description") or row.get("Narration") or row.get("Details") or "").strip()
        debit_str = (row.get("Debit") or row.get("Withdrawal Amt") or "0").strip()
        credit_str = (row.get("Credit") or row.get("Deposit Amt") or "0").strip()
        balance_str = (row.get("Balance") or row.get("Running Balance") or "0").strip()
        account_id = (row.get("Account No") or row.get("Account Number") or "UNKNOWN").strip()

        if not date_str:
            return None

        parsed_date = self._parse_date(date_str)
        if parsed_date is None:
            return None

        debit = self._safe_float(debit_str)
        credit = self._safe_float(credit_str)
        balance = self._safe_float(balance_str)

        if credit > 0:
            txn_type = TransactionType.CREDIT
            amount = credit
        elif debit > 0:
            txn_type = TransactionType.DEBIT
            amount = debit
        else:
            return None

        category = self._infer_category(desc)
        channel = self._infer_channel(desc)
        is_recurring = self._is_recurring_pattern(desc)

        return BankingTransaction(
            account_id=account_id,
            transaction_date=parsed_date,
            transaction_type=txn_type,
            amount=amount,
            balance_after=balance,
            category=category,
            description=desc,
            channel=channel,
            is_recurring=is_recurring,
        )

    # -- UPI statement parsing ------------------------------------------------

    def parse_upi_transactions(self, file_path: str) -> List[BankingTransaction]:
        """Parse a UPI transaction export (Google Pay / PhonePe / Paytm CSV)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"UPI file not found: {file_path}")

        transactions: List[BankingTransaction] = []
        text = path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(StringIO(text))

        for row in reader:
            date_str = (
                row.get("Date") or row.get("Transaction Date") or row.get("Txn Date") or ""
            ).strip()
            upi_id = (row.get("UPI ID") or row.get("VPA") or row.get("Receiver") or "").strip()
            amount_str = (row.get("Amount") or row.get("Txn Amount") or "0").strip()
            status = (row.get("Status") or row.get("Txn Status") or "SUCCESS").strip().upper()
            note = (row.get("Note") or row.get("Description") or row.get("Remarks") or "").strip()
            account_id = (row.get("From Account") or row.get("Account") or "UPI_DEFAULT").strip()

            if status not in ("SUCCESS", "COMPLETED"):
                continue
            parsed_date = self._parse_date(date_str)
            if parsed_date is None:
                continue
            amount = self._safe_float(amount_str)
            if amount <= 0:
                continue

            channel = ChannelType.UPI
            category = self._infer_category(f"{note} {upi_id}")
            txn_type = TransactionType.CREDIT if "received" in note.lower() else TransactionType.DEBIT

            transactions.append(BankingTransaction(
                account_id=account_id,
                transaction_date=parsed_date,
                transaction_type=txn_type,
                amount=amount,
                balance_after=0.0,
                category=category,
                merchant_name=upi_id,
                description=note,
                channel=channel,
                is_domestic=True,
            ))
        logger.info("Parsed %d UPI transactions from %s", len(transactions), file_path)
        return transactions

    # -- CIBIL / Credit Bureau -----------------------------------------------

    def fetch_cibil_report(self, customer_id: str) -> Dict[str, Any]:
        """Fetch credit report from CIBIL API.

        Returns a dict with score, accounts, enquiries, and derogatory records.
        """
        logger.info("Fetching CIBIL report for customer %s", customer_id)
        return {
            "customer_id": customer_id,
            "credit_score": None,
            "total_accounts": 0,
            "active_accounts": 0,
            "total_credit_limit": 0.0,
            "total_outstanding": 0.0,
            "enquiries_last_12m": 0,
            "derogatory_records": 0,
            "days_since_last_payment": None,
            "report_date": datetime.utcnow().isoformat(),
            "_note": "Stub – connect to CIBIL API with valid credentials",
        }

    def fetch_account_aggregator(self, customer_id: str) -> Dict[str, Any]:
        """Fetch aggregated financial data via the RBI AA framework.

        Returns consent and data-fulfilment status.
        """
        logger.info("Initiating AA consent for customer %s", customer_id)
        return {
            "customer_id": customer_id,
            "consent_id": str(uuid.uuid4()),
            "status": "pending_consent",
            "fip_list": self.SUPPORTED_BANKS,
            "fi_types": ["DEPOSIT", "TERM_DEPOSIT", "RECURRING_DEPOSIT",
                         "SIP", "GOVERNMENT_SCHEMES", "COLLATERAL"],
            "created_at": datetime.utcnow().isoformat(),
            "_note": "Stub – requires AA gateway credentials",
        }

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        formats = [
            "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y",
            "%d %b %Y", "%d %B %Y", "%Y/%m/%d", "%d-%b-%Y",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _safe_float(value: str) -> float:
        cleaned = re.sub(r"[^\d.\-]", "", value)
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0

    @staticmethod
    def _infer_category(description: str) -> TransactionCategory:
        desc_lower = description.lower()
        keyword_map: Dict[TransactionCategory, List[str]] = {
            TransactionCategory.SALARY: ["salary", "payroll", "wages", "stipend"],
            TransactionCategory.GROCERIES: ["grocery", "supermarket", "bigbasket", "blinkit", "zepto", "dmart"],
            TransactionCategory.RENT: ["rent", "housing", "maintenance", "society"],
            TransactionCategory.ENTERTAINMENT: ["movie", "netflix", "hotstar", "prime", "bookmyshow", "zomato", "swiggy"],
            TransactionCategory.BUSINESS: ["business", "invoice", "vendor", "supplier", "payment to"],
            TransactionCategory.UTILITIES: ["electric", "water", "gas", "broadband", "jio", "airtel", "vi ", "bsnl"],
            TransactionCategory.MEDICAL: ["hospital", "pharmacy", "medical", "doctor", "health", "clinic"],
            TransactionCategory.EDUCATION: ["school", "college", "university", "tuition", "education", "course"],
            TransactionCategory.TRANSFER: ["transfer", "neft", "imps", "rtgs", "upi", "sent to", "received from"],
            TransactionCategory.INVESTMENT: ["mutual fund", "sip", "stock", "trading", "zerodha", "groww", "fdsip"],
        }
        for cat, keywords in keyword_map.items():
            if any(kw in desc_lower for kw in keywords):
                return cat
        return TransactionCategory.OTHER

    @staticmethod
    def _infer_channel(description: str) -> ChannelType:
        desc_upper = description.upper()
        channel_map: Dict[ChannelType, List[str]] = {
            ChannelType.UPI: ["UPI", "BHIM", "GPAY", "PHONEPE", "PAYTM"],
            ChannelType.NEFT: ["NEFT"],
            ChannelType.RTGS: ["RTGS"],
            ChannelType.IMPS: ["IMPS"],
            ChannelType.CHEQUE: ["CHEQUE", "CHQ", "CTS"],
            ChannelType.CASH: ["CASH", "COUNTER"],
            ChannelType.ATM: ["ATM"],
            ChannelType.CARD: ["CARD", "POS", "SWIPE", "CHIP"],
        }
        for ch, keywords in channel_map.items():
            if any(kw in desc_upper for kw in keywords):
                return ch
        return ChannelType.NEFT

    @staticmethod
    def _is_recurring_pattern(description: str) -> bool:
        recurring_keywords = [
            "emi", "auto-debit", "standing instruction", "recurring",
            "sip", "subscription", "monthly", "annual", "quarterly",
        ]
        desc_lower = description.lower()
        return any(kw in desc_lower for kw in recurring_keywords)


# ---------------------------------------------------------------------------
# Transaction Feature Engineer
# ---------------------------------------------------------------------------

class TransactionFeatureEngineer:
    """Derive credit-risk features from raw banking transactions."""

    def __init__(self, reference_date: Optional[datetime] = None) -> None:
        self.reference_date = reference_date or datetime.utcnow()

    # -- Income features -----------------------------------------------------

    def compute_income_features(
        self, transactions: pd.DataFrame
    ) -> Dict[str, float]:
        """Monthly income, income stability, growth rate, salary consistency."""
        features: Dict[str, float] = {}
        income_mask = (
            (transactions["transaction_type"] == "credit")
            & (transactions["category"].isin(["salary", "business", "transfer"]))
        )
        income_txns = transactions.loc[income_mask].copy()
        if income_txns.empty:
            return self._zero_income_features()

        income_txns["month"] = pd.to_datetime(income_txns["transaction_date"]).dt.to_period("M")
        monthly_income = income_txns.groupby("month")["amount"].sum()

        features["avg_monthly_income"] = float(monthly_income.mean())
        features["median_monthly_income"] = float(monthly_income.median())
        features["max_monthly_income"] = float(monthly_income.max())
        features["min_monthly_income"] = float(monthly_income.min())
        features["income_std"] = float(monthly_income.std()) if len(monthly_income) > 1 else 0.0
        features["income_stability"] = (
            float(1.0 - (features["income_std"] / (features["avg_monthly_income"] + 1e-8)))
            if features["avg_monthly_income"] > 0 else 0.0
        )
        features["income_stability"] = float(np.clip(features["income_stability"], 0.0, 1.0))

        if len(monthly_income) >= 2:
            first_half = monthly_income.iloc[: len(monthly_income) // 2].mean()
            second_half = monthly_income.iloc[len(monthly_income) // 2 :].mean()
            features["income_growth_rate"] = float(
                (second_half - first_half) / (first_half + 1e-8)
            )
        else:
            features["income_growth_rate"] = 0.0

        salary_txns = income_txns[income_txns["category"] == "salary"]
        if not salary_txns.empty:
            salary_monthly = salary_txns.groupby("month")["amount"].sum()
            total_monthly = monthly_income
            overlap = salary_monthly.reindex(total_monthly.index, fill_value=0)
            ratio = overlap / (total_monthly + 1e-8)
            features["salary_consistency"] = float(ratio.mean())
        else:
            features["salary_consistency"] = 0.0

        features["total_income"] = float(monthly_income.sum())
        features["num_income_months"] = int(len(monthly_income))
        return features

    def _zero_income_features(self) -> Dict[str, float]:
        return {
            "avg_monthly_income": 0.0, "median_monthly_income": 0.0,
            "max_monthly_income": 0.0, "min_monthly_income": 0.0,
            "income_std": 0.0, "income_stability": 0.0,
            "income_growth_rate": 0.0, "salary_consistency": 0.0,
            "total_income": 0.0, "num_income_months": 0,
        }

    # -- Spending features ---------------------------------------------------

    def compute_spending_features(
        self, transactions: pd.DataFrame
    ) -> Dict[str, float]:
        """Monthly spend, category breakdown, essential vs discretionary ratio."""
        features: Dict[str, float] = {}
        spend_mask = transactions["transaction_type"].isin(["debit", "emi_payment", "fee"])
        spend_txns = transactions.loc[spend_mask].copy()
        if spend_txns.empty:
            return self._zero_spending_features()

        spend_txns["month"] = pd.to_datetime(spend_txns["transaction_date"]).dt.to_period("M")
        monthly_spend = spend_txns.groupby("month")["amount"].sum()

        features["avg_monthly_spend"] = float(monthly_spend.mean())
        features["median_monthly_spend"] = float(monthly_spend.median())
        features["max_monthly_spend"] = float(monthly_spend.max())
        features["spend_std"] = float(monthly_spend.std()) if len(monthly_spend) > 1 else 0.0
        features["total_spend"] = float(spend_txns["amount"].sum())
        features["num_spend_txns"] = int(len(spend_txns))

        essential_cats = {"groceries", "rent", "utilities", "medical", "education"}
        discretionary_cats = {"entertainment", "other"}
        essential_spend = spend_txns.loc[
            spend_txns["category"].isin(essential_cats), "amount"
        ].sum()
        discretionary_spend = spend_txns.loc[
            spend_txns["category"].isin(discretionary_cats), "amount"
        ].sum()
        total = essential_spend + discretionary_spend + 1e-8
        features["essential_spend_ratio"] = float(essential_spend / total)
        features["discretionary_spend_ratio"] = float(discretionary_spend / total)
        features["essential_spend_total"] = float(essential_spend)

        for cat in TransactionCategory:
            cat_spend = spend_txns.loc[
                spend_txns["category"] == cat.value, "amount"
            ].sum()
            features[f"spend_category_{cat.value}"] = float(cat_spend)
            features[f"spend_category_{cat.value}_pct"] = float(
                cat_spend / (features["total_spend"] + 1e-8)
            )

        return features

    def _zero_spending_features(self) -> Dict[str, float]:
        features: Dict[str, float] = {
            "avg_monthly_spend": 0.0, "median_monthly_spend": 0.0,
            "max_monthly_spend": 0.0, "spend_std": 0.0,
            "total_spend": 0.0, "num_spend_txns": 0,
            "essential_spend_ratio": 0.0, "discretionary_spend_ratio": 0.0,
            "essential_spend_total": 0.0,
        }
        for cat in TransactionCategory:
            features[f"spend_category_{cat.value}"] = 0.0
            features[f"spend_category_{cat.value}_pct"] = 0.0
        return features

    # -- EMI features --------------------------------------------------------

    def compute_emi_features(
        self, transactions: pd.DataFrame
    ) -> Dict[str, float]:
        """EMI payment history, EMI-to-income ratio, on-time / missed rate."""
        features: Dict[str, float] = {}
        emi_txns = transactions.loc[
            transactions["transaction_type"] == "emi_payment"
        ].copy()
        if emi_txns.empty:
            return self._zero_emi_features()

        emi_txns["month"] = pd.to_datetime(emi_txns["transaction_date"]).dt.to_period("M")
        features["total_emi_count"] = int(len(emi_txns))
        features["total_emi_amount"] = float(emi_txns["amount"].sum())
        monthly_emi = emi_txns.groupby("month")["amount"].sum()
        features["avg_monthly_emi"] = float(monthly_emi.mean())
        features["max_monthly_emi"] = float(monthly_emi.max())

        income_feats = self.compute_income_features(transactions)
        avg_income = income_feats.get("avg_monthly_income", 0.0)
        features["emi_to_income_ratio"] = float(
            features["avg_monthly_emi"] / (avg_income + 1e-8)
        )
        features["total_emi_to_income_ratio"] = float(
            features["total_emi_amount"] / (income_feats.get("total_income", 0.0) + 1e-8)
        )

        if "loan_id" in emi_txns.columns:
            loan_months = emi_txns.groupby("loan_id")["month"].nunique()
            features["max_consecutive_emi"] = int(loan_months.max()) if len(loan_months) > 0 else 0
        else:
            features["max_consecutive_emi"] = int(len(monthly_emi))

        features["on_time_payment_rate"] = float(
            (emi_txns["verification_status"] == "verified").mean()
        ) if len(emi_txns) > 0 else 1.0

        all_emi_months = set(emi_txns["month"].unique())
        month_diffs = sorted([(all_emi_months.pop(), all_emi_months.copy())], key=lambda x: x[0]) if all_emi_months else []
        features["missed_payment_count"] = max(0, features["max_consecutive_emi"] - len(set(emi_txns["month"].unique())))

        return features

    def _zero_emi_features(self) -> Dict[str, float]:
        return {
            "total_emi_count": 0, "total_emi_amount": 0.0,
            "avg_monthly_emi": 0.0, "max_monthly_emi": 0.0,
            "emi_to_income_ratio": 0.0, "total_emi_to_income_ratio": 0.0,
            "max_consecutive_emi": 0, "on_time_payment_rate": 1.0,
            "missed_payment_count": 0,
        }

    # -- Balance features ----------------------------------------------------

    def compute_balance_features(
        self, transactions: pd.DataFrame
    ) -> Dict[str, float]:
        """Min / max / avg balance, negative-balance days, volatility."""
        features: Dict[str, float] = {}
        balances = transactions["balance_after"].dropna()
        if balances.empty:
            return self._zero_balance_features()

        features["avg_balance"] = float(balances.mean())
        features["median_balance"] = float(balances.median())
        features["min_balance"] = float(balances.min())
        features["max_balance"] = float(balances.max())
        features["balance_std"] = float(balances.std()) if len(balances) > 1 else 0.0
        features["balance_range"] = features["max_balance"] - features["min_balance"]
        features["negative_balance_days"] = int((balances < 0).sum())
        features["negative_balance_pct"] = float((balances < 0).mean())
        features["low_balance_days"] = int((balances < 1000).sum())
        features["low_balance_pct"] = float((balances < 1000).mean())
        features["balance_volatility"] = float(
            features["balance_std"] / (abs(features["avg_balance"]) + 1e-8)
        )
        return features

    def _zero_balance_features(self) -> Dict[str, float]:
        return {
            "avg_balance": 0.0, "median_balance": 0.0,
            "min_balance": 0.0, "max_balance": 0.0,
            "balance_std": 0.0, "balance_range": 0.0,
            "negative_balance_days": 0, "negative_balance_pct": 0.0,
            "low_balance_days": 0, "low_balance_pct": 0.0,
            "balance_volatility": 0.0,
        }

    # -- Cash-flow features --------------------------------------------------

    def compute_cash_flow_features(
        self, transactions: pd.DataFrame
    ) -> Dict[str, float]:
        """Net cash flow, volatility, trend, and seasonal patterns."""
        features: Dict[str, float] = {}
        df = transactions.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df["month"] = df["transaction_date"].dt.to_period("M")

        credit_mask = df["transaction_type"] == "credit"
        debit_mask = df["transaction_type"].isin(["debit", "emi_payment", "fee"])

        monthly_credit = df.loc[credit_mask].groupby("month")["amount"].sum()
        monthly_debit = df.loc[debit_mask].groupby("month")["amount"].sum()
        all_months = sorted(set(monthly_credit.index) | set(monthly_debit.index))
        credit_series = monthly_credit.reindex(all_months, fill_value=0)
        debit_series = monthly_debit.reindex(all_months, fill_value=0)

        net_cf = credit_series - debit_series
        features["avg_monthly_cash_flow"] = float(net_cf.mean())
        features["total_cash_flow"] = float(net_cf.sum())
        features["cash_flow_std"] = float(net_cf.std()) if len(net_cf) > 1 else 0.0
        features["cash_flow_volatility"] = float(
            features["cash_flow_std"] / (abs(features["avg_monthly_cash_flow"]) + 1e-8)
        )
        features["positive_cash_flow_months"] = int((net_cf > 0).sum())
        features["negative_cash_flow_months"] = int((net_cf < 0).sum())
        features["cash_flow_to_income_ratio"] = float(
            features["avg_monthly_cash_flow"] / (
                credit_series.mean() + 1e-8
            )
        )

        if len(net_cf) >= 3:
            x = np.arange(len(net_cf), dtype=float)
            slope = float(np.polyfit(x, net_cf.values, 1)[0])
            features["cash_flow_trend"] = slope
        else:
            features["cash_flow_trend"] = 0.0

        seasonal_vol = 0.0
        cf_values = net_cf.values
        if len(cf_values) > 3:
            seasonal_vol = float(np.std(np.abs(np.diff(cf_values))))
        features["cash_flow_seasonal_volatility"] = seasonal_vol

        return features

    # -- Credit utilization --------------------------------------------------

    def compute_credit_utilization(
        self, transactions: pd.DataFrame
    ) -> Dict[str, float]:
        """Credit card utilization, revolving credit patterns."""
        features: Dict[str, float] = {}
        card_spends = transactions.loc[
            (transactions["transaction_type"] == "debit")
            & (transactions["channel"].isin(["CARD", "UPI"]))
        ]
        total_spends = transactions.loc[
            transactions["transaction_type"].isin(["debit", "emi_payment", "fee"])
        ]

        total_spend_amt = float(total_spends["amount"].sum()) if not total_spends.empty else 0.0
        card_spend_amt = float(card_spends["amount"].sum()) if not card_spends.empty else 0.0

        features["total_card_spend"] = card_spend_amt
        features["card_spend_pct"] = float(
            card_spend_amt / (total_spend_amt + 1e-8)
        )

        income_feats = self.compute_income_features(transactions)
        avg_income = income_feats.get("avg_monthly_income", 0.0)
        features["credit_utilization_to_income"] = float(
            card_spend_amt / (avg_income + 1e-8)
        )

        if not total_spends.empty:
            total_spends_copy = total_spends.copy()
            total_spends_copy["month"] = pd.to_datetime(
                total_spends_copy["transaction_date"]
            ).dt.to_period("M")
            monthly_totals = total_spends_copy.groupby("month")["amount"].sum()
            features["avg_monthly_total_spend"] = float(monthly_totals.mean())
            features["spend_to_income_ratio"] = float(
                features["avg_monthly_total_spend"] / (avg_income + 1e-8)
            )
        else:
            features["avg_monthly_total_spend"] = 0.0
            features["spend_to_income_ratio"] = 0.0

        return features

    # -- Velocity features ---------------------------------------------------

    def compute_velocity_features(
        self, transactions: pd.DataFrame
    ) -> Dict[str, float]:
        """Transaction frequency, large transactions, unusual activity."""
        features: Dict[str, float] = {}
        if transactions.empty:
            return self._zero_velocity_features()

        df = transactions.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df["month"] = df["transaction_date"].dt.to_period("M")
        df["week"] = df["transaction_date"].dt.to_period("W")

        monthly_counts = df.groupby("month").size()
        features["avg_monthly_txn_count"] = float(monthly_counts.mean())
        features["max_monthly_txn_count"] = int(monthly_counts.max())
        features["total_txn_count"] = int(len(df))
        features["txn_count_std"] = float(monthly_counts.std()) if len(monthly_counts) > 1 else 0.0

        weekly_counts = df.groupby("week").size()
        features["avg_weekly_txn_count"] = float(weekly_counts.mean())

        amounts = df["amount"]
        p90 = float(amounts.quantile(0.9)) if len(amounts) > 0 else 0.0
        features["large_txn_count"] = int((amounts > p90).sum())
        features["large_txn_pct"] = float(features["large_txn_count"] / (len(df) + 1e-8))
        features["max_single_txn"] = float(amounts.max()) if not amounts.empty else 0.0
        features["avg_txn_size"] = float(amounts.mean()) if not amounts.empty else 0.0

        daily_counts = df.groupby(df["transaction_date"].dt.date).size()
        mean_daily = float(daily_counts.mean()) if len(daily_counts) > 0 else 0.0
        std_daily = float(daily_counts.std()) if len(daily_counts) > 1 else 0.0
        unusual_days = int((daily_counts > mean_daily + 2 * std_daily).sum()) if std_daily > 0 else 0
        features["unusual_activity_days"] = unusual_days

        return features

    def _zero_velocity_features(self) -> Dict[str, float]:
        return {
            "avg_monthly_txn_count": 0.0, "max_monthly_txn_count": 0,
            "total_txn_count": 0, "txn_count_std": 0.0,
            "avg_weekly_txn_count": 0.0, "large_txn_count": 0,
            "large_txn_pct": 0.0, "max_single_txn": 0.0,
            "avg_txn_size": 0.0, "unusual_activity_days": 0,
        }

    # -- Behavioral features -------------------------------------------------

    def compute_behavioral_features(
        self, transactions: pd.DataFrame
    ) -> Dict[str, float]:
        """Spending regularity, merchant diversity, time-of-day patterns."""
        features: Dict[str, float] = {}
        if transactions.empty:
            return self._zero_behavioral_features()

        df = transactions.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])

        spend_txns = df[df["transaction_type"].isin(["debit", "emi_payment", "fee"])]
        if not spend_txns.empty:
            spend_txns_copy = spend_txns.copy()
            spend_txns_copy["month"] = spend_txns_copy["transaction_date"].dt.to_period("M")
            monthly_spend_counts = spend_txns_copy.groupby("month").size()
            mean_monthly = float(monthly_spend_counts.mean()) if len(monthly_spend_counts) > 0 else 0.0
            std_monthly = float(monthly_spend_counts.std()) if len(monthly_spend_counts) > 1 else 0.0
            features["spending_regularity"] = float(
                1.0 - (std_monthly / (mean_monthly + 1e-8))
            ) if mean_monthly > 0 else 0.0
            features["spending_regularity"] = float(np.clip(features["spending_regularity"], 0.0, 1.0))
        else:
            features["spending_regularity"] = 0.0

        merchants = df["merchant_name"].dropna()
        unique_merchants = merchants.nunique()
        features["unique_merchant_count"] = int(unique_merchants)
        features["merchant_diversity"] = float(
            unique_merchants / (len(merchants) + 1e-8)
        )

        hours = df["transaction_date"].dt.hour
        features["txn_hour_mean"] = float(hours.mean())
        features["txn_hour_std"] = float(hours.std()) if len(hours) > 1 else 0.0
        night_mask = (hours >= 22) | (hours <= 5)
        features["night_txn_pct"] = float(night_mask.mean())

        day_of_week = df["transaction_date"].dt.dayofweek
        features["weekend_txn_pct"] = float((day_of_week >= 5).mean())

        month_day = df["transaction_date"].dt.day
        month_end_mask = month_day >= 25
        features["month_end_spend_pct"] = float(
            month_end_mask[df["transaction_type"].isin(["debit", "emi_payment", "fee"])].mean()
        ) if len(df) > 0 else 0.0

        channels = df["channel"].value_counts(normalize=True)
        for ch in ChannelType:
            features[f"channel_{ch.value}_pct"] = float(channels.get(ch.value, 0.0))

        return features

    def _zero_behavioral_features(self) -> Dict[str, float]:
        features: Dict[str, float] = {
            "spending_regularity": 0.0, "unique_merchant_count": 0,
            "merchant_diversity": 0.0, "txn_hour_mean": 0.0,
            "txn_hour_std": 0.0, "night_txn_pct": 0.0,
            "weekend_txn_pct": 0.0, "month_end_spend_pct": 0.0,
        }
        for ch in ChannelType:
            features[f"channel_{ch.value}_pct"] = 0.0
        return features

    # -- Orchestrator --------------------------------------------------------

    def build_all_transaction_features(
        self, transactions: pd.DataFrame
    ) -> Dict[str, float]:
        """Run every feature group and return a flat feature dictionary."""
        if transactions.empty:
            logger.warning("Empty transaction DataFrame passed to feature engineer")
            all_features: Dict[str, float] = {}
            all_features.update(self._zero_income_features())
            all_features.update(self._zero_spending_features())
            all_features.update(self._zero_emi_features())
            all_features.update(self._zero_balance_features())
            all_features.update(self._zero_velocity_features())
            all_features.update(self._zero_behavioral_features())
            return all_features

        all_features: Dict[str, float] = {}
        all_features.update(self.compute_income_features(transactions))
        all_features.update(self.compute_spending_features(transactions))
        all_features.update(self.compute_emi_features(transactions))
        all_features.update(self.compute_balance_features(transactions))
        all_features.update(self.compute_cash_flow_features(transactions))
        all_features.update(self.compute_credit_utilization(transactions))
        all_features.update(self.compute_velocity_features(transactions))
        all_features.update(self.compute_behavioral_features(transactions))

        logger.info(
            "Built %d transaction features from %d transactions",
            len(all_features), len(transactions),
        )
        return all_features


# ---------------------------------------------------------------------------
# Realistic Data Generator
# ---------------------------------------------------------------------------

class BankingDataGenerator:
    """Generate realistic Indian banking transaction data for testing."""

    SALARY_RANGES: Dict[str, Tuple[float, float]] = {
        "salaried_low": (20000, 45000),
        "salaried_mid": (45000, 120000),
        "salaried_high": (120000, 350000),
        "business_low": (30000, 80000),
        "business_mid": (80000, 250000),
        "business_high": (250000, 700000),
    }

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)

    def generate_transactions(
        self,
        customer_profile: Dict[str, Any],
        months: int = 12,
    ) -> List[BankingTransaction]:
        """Generate realistic Indian banking transactions.

        Parameters
        ----------
        customer_profile : dict
            Keys: ``income_bracket`` (one of ``SALARY_RANGES`` keys),
            ``num_loans`` (int), ``emi_count`` (int),
            ``has_credit_card`` (bool), ``has_business_income`` (bool),
            ``delinquency_probability`` (float 0-1),
            ``account_id`` (str, optional).
        months : int
            Number of months of history to generate.
        """
        account_id = customer_profile.get("account_id", str(uuid.uuid4())[:8])
        bracket = customer_profile.get("income_bracket", "salaried_mid")
        salary_low, salary_high = self.SALARY_RANGES.get(bracket, (45000, 120000))
        num_loans = customer_profile.get("num_loans", 1)
        emi_count = customer_profile.get("emi_count", 1)
        has_credit_card = customer_profile.get("has_credit_card", True)
        has_business_income = customer_profile.get("has_business_income", False)
        delinq_prob = customer_profile.get("delinquency_probability", 0.05)

        base_salary = self.rng.uniform(salary_low, salary_high)
        if has_business_income:
            business_income = base_salary * self.rng.uniform(0.3, 1.5)
        else:
            business_income = 0.0

        emi_amounts = [
            base_salary * self.rng.uniform(0.05, 0.25) for _ in range(emi_count)
        ]
        credit_limit = base_salary * self.rng.uniform(1.5, 4.0) if has_credit_card else 0.0

        end_date = datetime.utcnow().replace(day=1)
        start_date = end_date - timedelta(days=months * 30)

        transactions: List[BankingTransaction] = []
        running_balance = base_salary * self.rng.uniform(0.5, 3.0)
        current_date = start_date

        for month_offset in range(months):
            month_date = start_date + timedelta(days=month_offset * 30)
            month_num = month_date.month
            txns = self._generate_month_transactions(
                month_date=month_date,
                base_salary=base_salary,
                business_income=business_income,
                emi_amounts=emi_amounts,
                credit_limit=credit_limit,
                has_credit_card=has_credit_card,
                has_business_income=has_business_income,
                delinq_prob=delinq_prob,
                account_id=account_id,
                running_balance=running_balance,
                month_num=month_num,
            )
            for txn in txns:
                if txn.transaction_type in (TransactionType.DEBIT, TransactionType.EMI_PAYMENT, TransactionType.FEE):
                    running_balance -= txn.amount
                else:
                    running_balance += txn.amount
                transactions.append(txn)

        logger.info(
            "Generated %d transactions for account %s over %d months",
            len(transactions), account_id, months,
        )
        return transactions

    def _generate_month_transactions(
        self,
        month_date: datetime,
        base_salary: float,
        business_income: float,
        emi_amounts: List[float],
        credit_limit: float,
        has_credit_card: bool,
        has_business_income: bool,
        delinq_prob: float,
        account_id: str,
        running_balance: float,
        month_num: int,
    ) -> List[BankingTransaction]:
        txns: List[BankingTransaction] = []
        year, month = month_date.year, month_date.month

        salary_date = datetime(year, month, min(28, self.rng.randint(1, 5)))
        salary = base_salary * self.rng.uniform(0.95, 1.05)
        txns.append(self._make_txn(
            account_id, salary_date, TransactionType.CREDIT, salary,
            running_balance, TransactionCategory.SALARY, "SALARY CREDIT",
            ChannelType.NEFT,
        ))

        if has_business_income and business_income > 0:
            for _ in range(self.rng.randint(1, 4)):
                biz_day = datetime(year, month, self.rng.randint(1, 28))
                biz_amt = business_income * self.rng.uniform(0.1, 0.5)
                txns.append(self._make_txn(
                    account_id, biz_day, TransactionType.CREDIT, biz_amt,
                    running_balance, TransactionCategory.BUSINESS,
                    "Business payment received", ChannelType.NEFT,
                ))

        for i, emi_amt in enumerate(emi_amounts):
            emi_day = datetime(year, month, min(28, self.rng.randint(3, 10)))
            missed = self.rng.random() < delinq_prob
            verification = VerificationStatus.FAILED if missed else VerificationStatus.VERIFIED
            actual_amt = 0.0 if missed else emi_amt * self.rng.uniform(0.99, 1.01)
            if not missed:
                txns.append(self._make_txn(
                    account_id, emi_day, TransactionType.EMI_PAYMENT, actual_amt,
                    running_balance, TransactionCategory.OTHER,
                    f"EMI auto-debit #{i + 1}", ChannelType.NEFT,
                    verification_status=verification,
                ))

        spend_categories = [
            (TransactionCategory.GROCERIES, 3000, 8000, self.rng.randint(4, 10)),
            (TransactionCategory.RENT, base_salary * 0.2, base_salary * 0.35, 1),
            (TransactionCategory.UTILITIES, 500, 3000, self.rng.randint(2, 5)),
            (TransactionCategory.ENTERTAINMENT, 500, 5000, self.rng.randint(2, 6)),
            (TransactionCategory.MEDICAL, 200, 10000, self.rng.randint(0, 2)),
            (TransactionCategory.EDUCATION, 1000, 15000, self.rng.randint(0, 2)),
        ]

        festival_boost = self._get_festival_multiplier(month_num)
        for cat, low, high, count in spend_categories:
            for _ in range(count):
                day = self.rng.randint(1, 28)
                amt = self.rng.uniform(low, high) * festival_boost
                channel = self.rng.choice(
                    [ChannelType.UPI, ChannelType.CARD, ChannelType.CASH, ChannelType.NEFT],
                    p=[0.5, 0.25, 0.1, 0.15],
                )
                txns.append(self._make_txn(
                    account_id,
                    datetime(year, month, day),
                    TransactionType.DEBIT,
                    amt, running_balance, cat,
                    f"{cat.value} spend", channel,
                ))

        if has_credit_card:
            card_spends = self.rng.randint(2, 8)
            credit_used = 0.0
            for _ in range(card_spends):
                day = self.rng.randint(1, 28)
                available = credit_limit - credit_used
                if available <= 0:
                    break
                amt = min(self.rng.uniform(500, available * 0.4), available)
                credit_used += amt
                txns.append(self._make_txn(
                    account_id,
                    datetime(year, month, day),
                    TransactionType.DEBIT,
                    amt, running_balance,
                    TransactionCategory.ENTERTAINMENT,
                    "Credit card payment", ChannelType.CARD,
                ))

        for _ in range(self.rng.randint(1, 4)):
            day = self.rng.randint(1, 28)
            amt = self.rng.uniform(100, 5000)
            txns.append(self._make_txn(
                account_id,
                datetime(year, month, day),
                TransactionType.DEBIT,
                amt, running_balance,
                TransactionCategory.TRANSFER,
                "UPI transfer", ChannelType.UPI,
            ))

        atms = self.rng.randint(1, 3)
        for _ in range(atms):
            day = self.rng.randint(1, 28)
            amt = self.rng.choice([500, 1000, 2000, 5000], p=[0.2, 0.35, 0.3, 0.15])
            txns.append(self._make_txn(
                account_id,
                datetime(year, month, day),
                TransactionType.DEBIT,
                amt, running_balance,
                TransactionCategory.OTHER,
                "ATM cash withdrawal", ChannelType.ATM,
            ))

        txns.sort(key=lambda t: t.transaction_date)
        return txns

    def _make_txn(
        self,
        account_id: str,
        txn_date: datetime,
        txn_type: TransactionType,
        amount: float,
        balance_after: float,
        category: TransactionCategory,
        description: str,
        channel: ChannelType,
        verification_status: VerificationStatus = VerificationStatus.VERIFIED,
        is_recurring: bool = False,
    ) -> BankingTransaction:
        jitter = self.rng.uniform(-5, 5)
        adjusted_date = txn_date + timedelta(hours=int(abs(jitter) * 2))
        return BankingTransaction(
            account_id=account_id,
            transaction_date=adjusted_date,
            transaction_type=txn_type,
            amount=round(max(amount, 0.0), 2),
            balance_after=round(balance_after + (amount if txn_type == TransactionType.CREDIT else -amount), 2),
            category=category,
            description=description,
            channel=channel,
            is_recurring=is_recurring,
            verification_status=verification_status,
        )

    @staticmethod
    def _get_festival_multiplier(month_num: int) -> float:
        if month_num in INDIAN_SEASONS.get("diwali", []):
            return 1.4
        if month_num in INDIAN_SEASONS.get("wedding", []):
            return 1.2
        if month_num in INDIAN_SEASONS.get("summer_travel", []):
            return 1.15
        if month_num in INDIAN_SEASONS.get("year_end", []):
            return 1.1
        return 1.0

    def generate_healthy_profile(self, account_id: str = "HEALTHY001") -> Dict[str, Any]:
        return {
            "account_id": account_id,
            "income_bracket": "salaried_mid",
            "num_loans": 1,
            "emi_count": 1,
            "has_credit_card": True,
            "has_business_income": False,
            "delinquency_probability": 0.01,
        }

    def generate_distressed_profile(self, account_id: str = "DISTRESS01") -> Dict[str, Any]:
        return {
            "account_id": account_id,
            "income_bracket": "salaried_low",
            "num_loans": 3,
            "emi_count": 3,
            "has_credit_card": True,
            "has_business_income": False,
            "delinquency_probability": 0.25,
        }

    def generate_business_profile(self, account_id: str = "BIZ001") -> Dict[str, Any]:
        return {
            "account_id": account_id,
            "income_bracket": "business_mid",
            "num_loans": 2,
            "emi_count": 2,
            "has_credit_card": True,
            "has_business_income": True,
            "delinquency_probability": 0.08,
        }
