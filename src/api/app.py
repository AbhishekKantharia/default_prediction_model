"""
API Serving Module
FastAPI-based REST API for serving default predictions and interpretations.
"""
import os
import logging
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Default Prediction Model API",
    description="Predicts loan default probability 12 months in advance",
    version="2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_DIR = os.environ.get("MODEL_DIR", "models/saved/latest")
loaded_models: Dict[str, Any] = {}


class PredictionRequest(BaseModel):
    loan_amount: float = Field(..., description="Loan amount in USD")
    interest_rate: float = Field(..., description="Annual interest rate (%)")
    annual_income: float = Field(..., description="Borrower annual income")
    debt_to_income: float = Field(..., description="Debt-to-income ratio")
    credit_score: int = Field(..., description="FICO credit score")
    employment_length: float = Field(..., description="Years of employment")
    open_credit_lines: int = Field(5, description="Number of open credit lines")
    total_credit_utilization: float = Field(0.3, description="Credit utilization ratio")
    revolving_balance: float = Field(0, description="Revolving balance")
    total_acc: int = Field(10, description="Total credit accounts")
    num_derogatory_records: int = Field(0, description="Derogatory records")
    num_collections_12m: int = Field(0, description="Collections in past 12 months")
    loan_type: str = Field("personal", description="Loan type")
    home_ownership: str = Field("RENT", description="Home ownership status")
    employment_status: str = Field("employed", description="Employment status")
    risk_grade: str = Field("B", description="Risk grade")


class PredictionResponse(BaseModel):
    default_probability: float
    risk_score: float
    risk_category: str
    confidence_lower: float
    confidence_upper: float
    recommended_action: str
    loan_type: str


class BatchPredictionRequest(BaseModel):
    records: List[PredictionRequest]


class ExplanationRequest(BaseModel):
    loan_amount: float
    interest_rate: float
    annual_income: float
    debt_to_income: float
    credit_score: int
    employment_length: float
    loan_type: str = "personal"


class ExplanationResponse(BaseModel):
    default_probability: float
    risk_score: float
    risk_category: str
    top_risk_factors: List[Dict[str, Any]]
    top_protective_factors: List[Dict[str, Any]]
    recommended_action: str


def load_models():
    global loaded_models
    if os.path.exists(MODEL_DIR):
        for fname in os.listdir(MODEL_DIR):
            if fname.endswith(".joblib"):
                name = fname.replace(".joblib", "")
                loaded_models[name] = joblib.load(
                    os.path.join(MODEL_DIR, fname))
        logger.info(f"Loaded {len(loaded_models)} models")
    else:
        logger.warning(f"Model directory not found: {MODEL_DIR}")


@app.on_event("startup")
async def startup_event():
    load_models()


def prepare_features(req: PredictionRequest) -> pd.DataFrame:
    features = {
        "loan_amount": req.loan_amount,
        "interest_rate": req.interest_rate,
        "annual_income": req.annual_income,
        "debt_to_income": req.debt_to_income,
        "credit_score": req.credit_score,
        "employment_length": req.employment_length,
        "open_credit_lines": req.open_credit_lines,
        "total_credit_utilization": req.total_credit_utilization,
        "revolving_balance": req.revolving_balance,
        "total_acc": req.total_acc,
        "num_derogatory_records": req.num_derogatory_records,
        "num_collections_12m": req.num_collections_12m,
        "loan_to_income": req.loan_amount / max(req.annual_income, 1),
        "debt_burden_score": req.debt_to_income * req.total_credit_utilization,
        "high_dti": int(req.debt_to_income > 0.43),
        "low_credit_score": int(req.credit_score < 620),
        "high_utilization": int(req.total_credit_utilization > 0.75),
        "risk_score_sum": sum([
            int(req.debt_to_income > 0.43),
            int(req.credit_score < 620),
            int(req.total_credit_utilization > 0.75),
            int(req.num_collections_12m > 0),
            int(req.num_derogatory_records > 1),
        ])
    }
    return pd.DataFrame([features])


def _categorize_risk(prob: float) -> str:
    if prob < 0.1:
        return "Very Low"
    elif prob < 0.25:
        return "Low"
    elif prob < 0.5:
        return "Medium"
    elif prob < 0.75:
        return "High"
    return "Very High"


def _get_action(prob: float) -> str:
    if prob < 0.1:
        return "APPROVE - Standard terms"
    elif prob < 0.25:
        return "APPROVE - Enhanced monitoring recommended"
    elif prob < 0.5:
        return "REVIEW - Additional documentation required"
    elif prob < 0.75:
        return "CONDITIONAL - Higher interest rate / collateral required"
    return "DECLINE - High default risk"


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(loaded_models.keys())
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not loaded_models:
        raise HTTPException(status_code=503, detail="No models loaded")

    model_name = "xgboost" if "xgboost" in loaded_models else \
        list(loaded_models.keys())[0]
    model = loaded_models[model_name]

    X = prepare_features(request)

    available = set(X.columns) & set(model.feature_names_in_) \
        if hasattr(model, "feature_names_in_") else set(X.columns)
    X = X.reindex(columns=model.feature_names_in_, fill_value=0) \
        if hasattr(model, "feature_names_in_") else X

    prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else 0.5

    return PredictionResponse(
        default_probability=round(prob, 4),
        risk_score=round(prob * 100, 2),
        risk_category=_categorize_risk(prob),
        confidence_lower=round(max(0, prob - 0.1), 4),
        confidence_upper=round(min(1, prob + 0.1), 4),
        recommended_action=_get_action(prob),
        loan_type=request.loan_type
    )


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    results = []
    for rec in request.records:
        resp = await predict(rec)
        results.append(resp)
    return {"predictions": results, "total": len(results)}


@app.post("/explain", response_model=ExplanationResponse)
async def explain(request: ExplanationRequest):
    if not loaded_models:
        raise HTTPException(status_code=503, detail="No models loaded")

    model_name = "xgboost" if "xgboost" in loaded_models else \
        list(loaded_models.keys())[0]
    model = loaded_models[model_name]

    features = {
        "loan_amount": request.loan_amount,
        "interest_rate": request.interest_rate,
        "annual_income": request.annual_income,
        "debt_to_income": request.debt_to_income,
        "credit_score": request.credit_score,
        "employment_length": request.employment_length,
        "loan_to_income": request.loan_amount / max(request.annual_income, 1),
    }

    X = pd.DataFrame([features])
    prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else 0.5

    risk_factors = [
        {"factor": "Debt-to-Income Ratio",
         "impact": "high" if request.debt_to_income > 0.43 else "moderate",
         "value": request.debt_to_income},
        {"factor": "Credit Score",
         "impact": "high" if request.credit_score < 620 else "low",
         "value": request.credit_score},
    ]

    protective_factors = []
    if request.credit_score >= 700:
        protective_factors.append(
            {"factor": "Credit Score", "value": request.credit_score})
    if request.employment_length >= 5:
        protective_factors.append(
            {"factor": "Employment Length", "value": request.employment_length})

    return ExplanationResponse(
        default_probability=round(prob, 4),
        risk_score=round(prob * 100, 2),
        risk_category=_categorize_risk(prob),
        top_risk_factors=risk_factors,
        top_protective_factors=protective_factors,
        recommended_action=_get_action(prob)
    )


@app.get("/model/info")
async def model_info():
    return {
        "loaded_models": list(loaded_models.keys()),
        "prediction_horizon": "12 months",
        "features_used": 25,
        "loan_types_supported": [
            "personal", "mortgage", "auto", "business",
            "student", "credit_card"
        ]
    }
