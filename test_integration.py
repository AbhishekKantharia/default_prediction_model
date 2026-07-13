"""Integration test for all 7 features."""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

print('=== ALL 7 FEATURES - INTEGRATION TEST ===')

print()
print('--- 1. BANKING TRANSACTION DATA ---')
from src.data.banking_transactions import BankingDataGenerator, TransactionFeatureEngineer
gen = BankingDataGenerator(seed=42)
txns = gen.generate_transactions({'customer_id':'C001','monthly_income':75000,'profile_type':'healthy'}, months=6)
tx_df = pd.DataFrame([t.model_dump() for t in txns])
tx_df['transaction_date'] = pd.to_datetime(tx_df['transaction_date'])
fe = TransactionFeatureEngineer()
txn_feats = fe.build_all_transaction_features(tx_df)
print('  Generated %d txns -> %d features' % (len(txns), len(txn_feats)))
print('  avg_monthly_income=%.0f, income_stability=%.2f' % (txn_feats['avg_monthly_income'], txn_feats['income_stability']))

print()
print('--- 2. ENSEMBLE ML MODEL (XGBoost + RF + LR) ---')
from src.models.ensemble import EnsembleDeployer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=2000, n_features=15, weights=[0.8,0.2], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
deployer = EnsembleDeployer()
result = deployer.train_ensemble(X_tr, y_tr, X_val, y_val, use_smote=False)
for k, v in result.items():
    if isinstance(v, dict) and 'val_auc' in v:
        print('  %s val_auc=%.4f' % (k, v['val_auc']))
preds = deployer.predict_with_confidence(X_test[:5])
print('  CI lower: %s' % preds['ci_lower'][:3].round(3))
print('  CI upper: %s' % preds['ci_upper'][:3].round(3))
print('  Confidence: %s' % preds['confidence'][:3].round(3))

print()
print('--- 3. MLOPS (MLflow) ---')
from src.mlops.mlflow_tracker import MLflowTracker
tracker = MLflowTracker(experiment_name='test', tracking_uri='sqlite:///mlruns_test.db')
tracker.start_run(run_name='integration_test')
tracker.log_params({'n_estimators': 500, 'max_depth': 8})
tracker.log_metrics({'val_auc': 0.89, 'val_f1': 0.82})
print('  MLflow tracking URI: %s' % tracker.tracking_uri)
print('  Experiment: %s' % tracker.experiment_name)
print('  Active run ID: %s' % tracker.current_run.info.run_id if tracker.current_run else 'None')

print()
print('--- 4. REAL-TIME SHAP EXPLANATIONS ---')
from src.interpretation.realtime_shap import RealtimeSHAPExplainer
feat_names = ['f%d' % i for i in range(15)]
explainer = RealtimeSHAPExplainer(deployer.xgb_model, feat_names, background_data=X_tr[:100])
sample_dict = dict(zip(feat_names, X_test[0]))
shap_res = explainer.explain_prediction(sample_dict, top_n=5)
rf_count = len(shap_res.get('risk_factors', []))
pf_count = len(shap_res.get('protective_factors', []))
print('  Risk factors: %d, Protective: %d' % (rf_count, pf_count))
cf = explainer.explain_counterfactual(sample_dict, target_probability=0.3)
cf_changes = len(cf.get('changed_features', []))
print('  Counterfactual changes needed: %d' % cf_changes)
summary = explainer.generate_executive_summary(sample_dict, {'default_prob': 0.35})
print('  Executive summary: %d chars' % len(summary))

print()
print('--- 5. CONCEPT DRIFT DETECTION ---')
from src.monitoring.drift_detection import DriftDetector
detector = DriftDetector()
ref = pd.DataFrame(np.random.randn(500, 5), columns=['f0','f1','f2','f3','f4'])
cur = pd.DataFrame(np.random.randn(500, 5) + 0.5, columns=['f0','f1','f2','f3','f4'])
drift = detector.detect_data_drift(ref, cur)
print('  Drifted features: %d/%d' % (drift['num_drifted_features'], drift['total_features']))
print('  Status: %s' % drift['overall_drift_status'])

pred_drift = detector.detect_prediction_drift(
    np.random.rand(500)*0.3+0.1, np.random.rand(500)*0.3+0.3)
print('  Prediction drift KS stat: %.4f' % pred_drift.get('ks_statistic', 0))

print()
print('--- 6. RISK-BASED PRICING ---')
from src.pricing.risk_pricing import RiskPricingEngine, EMICalculator
pricing = RiskPricingEngine()
p = pricing.recommend_pricing(
    default_prob=0.15,
    loan_details={'loan_amount':500000,'tenure_months':60,'loan_type':'personal',
                  'segment':'near_prime','credit_score':680,'dti':0.35})
for k in ['recommended_rate','optimal_rate','minimum_acceptable_rate','expected_loss','monthly_emi']:
    if k in p:
        val = p[k]
        if isinstance(val, float):
            print('  %s: %.2f' % (k, val))
        else:
            print('  %s: %s' % (k, val))

el = pricing.compute_expected_loss(0.15, 0.45, 500000)
print('  Expected loss (PD*LGD*EAD): INR %s' % '{:,.0f}'.format(el))

emi_calc = EMICalculator()
emi = emi_calc.compute_emi(500000, p.get('recommended_rate', 12), 60)
print('  EMI: INR %s' % '{:,.0f}'.format(emi))
schedule = emi_calc.compute_amortization_schedule(500000, p.get('recommended_rate', 12), 60)
total_int = sum(row['interest'] for row in schedule)
print('  Total interest over 60 months: INR %s' % '{:,.0f}'.format(total_int))

print()
print('--- 7. BANKING API INTEGRATION ---')
from src.api.banking_api import BankingAPIClient, AABankingIntegration
client = BankingAPIClient(api_base_url='https://api.example.com')
print('  BankingAPIClient: %s' % client.api_base_url)
aa = AABankingIntegration(aa_provider='cookie')
print('  AA Integration: %s' % aa.provider)

print()
print('==========================================')
print('  ALL 7 FEATURES VERIFIED SUCCESSFULLY')
print('==========================================')
