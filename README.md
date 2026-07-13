# Default Prediction Model — Problem Statement 4

A robust predictive solution that estimates the probability of loan default **12 months in advance**, achieving **90%+ accuracy** using both structured and unstructured data across all loan types and borrower segments.

## Problem Statement

**Current State:** Low prediction accuracy (16–22%), dependent solely on structured data, fragmented methodologies across loan types and borrower segments.

**Expected Outcome:** A robust predictive solution that:
- Estimates default probability **12 months in advance**
- Improves accuracy to **90%+**
- Uses both **structured and unstructured** data
- Applies suitable analytical methods for **different loan types and borrower profiles**
- Provides a **common interpretation framework** ensuring consistent, comparable, and actionable insights

## Solution Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Data       │───▶│ Preprocessing│───▶│    Feature       │
│   Ingestion  │    │   Pipeline   │    │   Engineering    │
└─────────────┘    └──────────────┘    └─────────────────┘
                                              │
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ Interpretation│◀──│  Evaluation  │◀──▶│   Model          │
│  Framework   │    │   Module     │    │   Training       │
└─────────────┘    └──────────────┘    └─────────────────┘
```

## Project Structure

```
default_prediction_model/
├── run_pipeline.py              # Main entry — orchestrates full pipeline
├── config.yaml                  # Central configuration
├── requirements.txt             # Python dependencies
├── index.html                   # Landing page
├── src/
│   ├── data/
│   │   ├── ingestion.py         # Data loading + synthetic generation
│   │   └── preprocessing.py     # Cleaning, encoding, text processing
│   ├── features/
│   │   └── engineering.py       # 50+ engineered features
│   ├── models/
│   │   ├── training.py          # 6 algorithms + stacking + SMOTE
│   │   └── prediction.py        # 12-month horizon predictions
│   ├── evaluation/
│   │   └── metrics.py           # Full evaluation + calibration
│   ├── interpretation/
│   │   └── framework.py         # SHAP + consistent framework
│   └── api/
│       └── app.py               # FastAPI REST endpoints
├── data/                        # Raw & processed data
├── models/                      # Saved models + artifacts
└── tests/                       # Unit tests
```

## Key Features

### Data Pipeline
- **Structured Data:** CSV/Parquet ingestion with automatic type detection
- **Unstructured Data:** Text processing (NLP) for loan descriptions, borrower notes
- **Synthetic Data:** Realistic synthetic dataset generation for development
- **Preprocessing:** Missing value handling, IQR outlier removal, encoding, scaling

### Feature Engineering (50+ Features)
- **Financial Ratios:** Loan-to-income, installment-to-income, revolving utilization
- **Risk Indicators:** High DTI, low credit score, high utilization flags
- **Interaction Features:** Credit × income, DTI × utilization, loan × interest
- **Temporal Features:** Loan age buckets, credit history length
- **Loan-Type-Specific:** Home equity ratio (mortgage), auto loan burden (auto)
- **Text Features:** Sentiment analysis, risk keyword detection, TF-IDF vectors

### Model Training
- **6 Algorithms:** Logistic Regression, Random Forest, XGBoost, LightGBM, Gradient Boosting, Stacking Ensemble
- **SMOTE Balancing:** Handles imbalanced default/non-default classes
- **Loan-Type Models:** Separate XGBoost models per loan type
- **Cross-Validation:** 5-fold stratified CV with AUC-ROC scoring

### Evaluation Framework
- **Metrics:** Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, Log Loss, Brier Score
- **Threshold Optimization:** Finds optimal thresholds per metric
- **Calibration Analysis:** Expected Calibration Error computation
- **Lift Charts:** Decile-level performance analysis
- **Segment Evaluation:** Per borrower segment and loan type breakdowns

### Unified Interpretation Framework
- **SHAP Values:** Global and local feature importance explanations
- **Feature Categories:** Automatic classification into credit, financial, behavioral, text, temporal
- **Segment-Level Explanations:** Per borrower segment and loan type
- **Consistent Framework:** Same interpretation rules across all models
- **Actionable Insights:** Risk factor identification with strength ratings

## Supported Loan Types

| Loan Type | Specific Features |
|-----------|------------------|
| Personal | Standard risk assessment |
| Mortgage | Home equity ratio, LTV analysis |
| Auto | Auto loan burden, vehicle value estimation |
| Business | Business risk score, revenue stability |
| Student | Education ROI, employment projection |
| Credit Card | Utilization patterns, payment behavior |

## Borrower Segments

| Segment | Credit Range | Model Adaptation |
|---------|-------------|------------------|
| Prime | 720+ | Lower risk weights |
| Near Prime | 660–719 | Moderate risk assessment |
| Subprime | 600–659 | Enhanced monitoring |
| Deep Subprime | <600 | High-risk protocols |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single loan default prediction |
| `/predict/batch` | POST | Batch predictions |
| `/explain` | POST | SHAP-based prediction explanation |
| `/model/info` | GET | Model metadata and capabilities |
| `/health` | GET | Health check |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (generates synthetic data if no real data)
python run_pipeline.py

# Start API server
uvicorn src.api.app:app --reload --port 8000
```

## Configuration

All settings are centralized in `config.yaml`:
- Data paths and target column
- Feature categories (numerical, categorical, text, temporal)
- Model hyperparameters
- Evaluation metrics and thresholds
- API configuration

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
