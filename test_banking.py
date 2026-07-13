import sys
sys.path.insert(0, r"C:\Users\abhis\Downloads\default_prediction_model")
from src.data.banking_transactions import (
    BankingTransaction, BankingDataIngestor, TransactionFeatureEngineer,
    BankingDataGenerator, TransactionType, TransactionCategory, ChannelType
)
import pandas as pd

gen = BankingDataGenerator(seed=42)

# Test healthy profile
profile = gen.generate_healthy_profile()
txns = gen.generate_transactions(profile, months=6)
print(f"Generated {len(txns)} transactions (healthy)")

df_txns = pd.DataFrame([t.model_dump() for t in txns])
fe = TransactionFeatureEngineer()
features = fe.build_all_transaction_features(df_txns)
print(f"Computed {len(features)} features")

# Test distressed profile
dp = gen.generate_distressed_profile()
dtxns = gen.generate_transactions(dp, months=12)
df_dist = pd.DataFrame([t.model_dump() for t in dtxns])
fd = fe.build_all_transaction_features(df_dist)
print(f"Distressed profile: {len(fd)} features")
print(f"  emi_to_income_ratio: {fd['emi_to_income_ratio']:.4f}")
print(f"  on_time_payment_rate: {fd['on_time_payment_rate']:.4f}")

# Test business profile
bp = gen.generate_business_profile()
btxns = gen.generate_transactions(bp, months=12)
print(f"Business profile: {len(btxns)} transactions")

# Test ingestor
ingestor = BankingDataIngestor()
ingestor.connect_to_banking_api({
    "bank_name": "HDFC",
    "api_base_url": "https://api.hdfcbank.com/open-banking",
    "client_id": "test123",
    "client_secret": "secret456",
})
print("Banking API connected")

cibil = ingestor.fetch_cibil_report("CUST001")
print(f"CIBIL report keys: {len(cibil)}")

aa = ingestor.fetch_account_aggregator("CUST001")
print(f"AA consent status: {aa['status']}")

# Test Pydantic validation
txn = BankingTransaction(
    account_id="TEST001",
    transaction_date="2025-01-15T10:00:00",
    transaction_type="credit",
    amount=50000.0,
    balance_after=75000.0,
    category="salary",
    channel="NEFT",
)
print(f"Transaction validated: {txn.transaction_id[:8]}...")

print("\nAll tests passed!")
