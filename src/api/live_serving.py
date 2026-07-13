"""Live prediction serving API integrating all 7 features into a unified FastAPI service."""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Optional internal imports with fallbacks                            #
# ------------------------------------------------------------------ #

try:
    from src.models.ensemble import EnsembleDeployer
except Exception:
    EnsembleDeployer = None  # type: ignore[assignment,misc]

try:
    from src.data.banking_transactions import BankingDataIngestor, TransactionFeatureEngineer
except Exception:
    BankingDataIngestor = None  # type: ignore[assignment,misc]
    TransactionFeatureEngineer = None  # type: ignore[assignment,misc]

try:
    from src.interpretation.realtime_shap import RealtimeSHAPExplainer, ExplanationFormatter
except Exception:
    RealtimeSHAPExplainer = None  # type: ignore[assignment,misc]
    ExplanationFormatter = None  # type: ignore[assignment,misc]

try:
    from src.pricing.risk_pricing import RiskPricingEngine, EMICalculator
except Exception:
    RiskPricingEngine = None  # type: ignore[assignment,misc]
    EMICalculator = None  # type: ignore[assignment,misc]

try:
    from src.monitoring.drift_detection import DriftDetector, AutoRetrainer
except Exception:
    DriftDetector = None  # type: ignore[assignment,misc]
    AutoRetrainer = None  # type: ignore[assignment,misc]

try:
    from src.mlops.mlflow_tracker import MLflowTracker, PipelineMLOps
except Exception:
    MLflowTracker = None  # type: ignore[assignment,misc]
    PipelineMLOps = None  # type: ignore[assignment,misc]

try:
    from src.api.banking_api import BankingAPIClient, AABankingIntegration
except Exception:
    BankingAPIClient = None  # type: ignore[assignment,misc]
    AABankingIntegration = None  # type: ignore[assignment,misc]

# ------------------------------------------------------------------ #
#  Pydantic request / response models                                 #
# ------------------------------------------------------------------ #

class LoanApplicationRequest(BaseModel):
    """Input for a full prediction request."""
    customer_id: Optional[str] = None
    age: Optional[float] = Field(None, ge=18, le=120)
    income: Optional[float] = Field(None, ge=0)
    employment_length: Optional[float] = Field(None, ge=0)
    loan_amount: Optional[float] = Field(None, ge=0)
    interest_rate: Optional[float] = Field(None, ge=0, le=100)
    term_months: Optional[int] = Field(None, ge=1)
    debt_to_income: Optional[float] = Field(None, ge=0)
    credit_score: Optional[float] = Field(None, ge=300, le=850)
    num_credit_lines: Optional[int] = Field(None, ge=0)
    num_derogatory_marks: Optional[int] = Field(None, ge=0)
    revolving_utilization: Optional[float] = Field(None, ge=0, le=2)
    total_accounts: Optional[int] = Field(None, ge=0)
    months_since_last_delinquency: Optional[float] = Field(None, ge=0)
    transaction_data: Optional[List[Dict[str, Any]]] = None
    extra_features: Optional[Dict[str, Any]] = None

    @field_validator("extra_features", mode="before")
    @classmethod
    def _ensure_dict(cls, v):
        return v if v is not None else {}


class QuickPredictionRequest(BaseModel):
    """Minimal input for a fast heuristic prediction."""
    age: Optional[float] = None
    income: Optional[float] = None
    loan_amount: Optional[float] = None
    credit_score: Optional[float] = None
    debt_to_income: Optional[float] = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction input."""
    applications: List[LoanApplicationRequest]


class ExplainRequest(BaseModel):
    """Detailed explanation for an existing prediction."""
    features: Dict[str, Any]
    prediction_result: Optional[Dict[str, Any]] = None
    top_n: int = Field(10, ge=1, le=50)
    format_for: str = Field("api", pattern="^(api|loan_officer|risk_manager|executive|auditor|customer)$")


class CounterfactualRequest(BaseModel):
    """Counterfactual explanation input."""
    features: Dict[str, Any]
    target_probability: float = Field(0.3, ge=0, le=1)
    max_changes: int = Field(3, ge=1, le=10)


class PricingRequest(BaseModel):
    """Risk-based pricing request."""
    features: Dict[str, Any]
    probability: Optional[float] = None
    loan_amount: Optional[float] = None
    loan_term_months: Optional[int] = None
    base_rate: Optional[float] = None


class TransactionAnalysisRequest(BaseModel):
    """Banking transaction analysis request."""
    customer_id: Optional[str] = None
    transactions: List[Dict[str, Any]]
    analysis_period_days: int = Field(90, ge=1, le=365)


class DriftCheckRequest(BaseModel):
    """Drift detection trigger request."""
    reference_data_path: Optional[str] = None
    current_data: Optional[List[Dict[str, Any]]] = None


class BankingSubmitRequest(BaseModel):
    """Submit credit decision to the banking system."""
    customer_id: str
    decision: str
    probability: float
    risk_category: str
    explanation_id: Optional[str] = None
    notes: Optional[str] = None


class BankingProfileRequest(BaseModel):
    """Fetch customer profile from banking API."""
    customer_id: str
    include_transactions: bool = False


class LoanProcessRequest(BaseModel):
    """Full loan processing pipeline."""
    application: LoanApplicationRequest
    auto_submit: bool = False
    pricing_override: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    """Structured prediction response."""
    success: bool
    probability: Optional[float] = None
    risk_category: Optional[str] = None
    confidence: Optional[float] = None
    explanation: Optional[Dict[str, Any]] = None
    pricing: Optional[Dict[str, Any]] = None
    drift_warning: Optional[str] = None
    model_version: Optional[str] = None
    timestamp: str = ""
    processing_time_ms: float = 0


class HealthResponse(BaseModel):
    """Health-check response."""
    status: str
    model_loaded: bool
    explainer_loaded: bool
    pricing_loaded: bool
    drift_detector_loaded: bool
    tracker_loaded: bool
    banking_api_loaded: bool
    transaction_processor_loaded: bool
    uptime_seconds: float
    timestamp: str


# ------------------------------------------------------------------ #
#  Application factory & shared state                                 #
# ------------------------------------------------------------------ #

_start_time = time.time()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Credit Risk Prediction API",
        description="Unified API for credit risk prediction, SHAP explanations, risk-based pricing, drift detection, and banking integration.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared component containers (initialised lazily)
    state: Dict[str, Any] = {
        "model": None,
        "explainer": None,
        "formatter": None,
        "pricing_engine": None,
        "drift_detector": None,
        "tracker": None,
        "banking_api": None,
        "txn_processor": None,
        "feature_names": [
            "age", "income", "employment_length", "loan_amount",
            "interest_rate", "term_months", "debt_to_income",
            "credit_score", "num_credit_lines", "num_derogatory_marks",
            "revolving_utilization", "total_accounts",
            "months_since_last_delinquency",
        ],
    }

    def _ensure_model():
        if state["model"] is None:
            if EnsembleDeployer is None:
                raise HTTPException(503, "EnsembleDeployer module not available")
            state["model"] = EnsembleDeployer()
            state["model"].load_ensemble()
        return state["model"]

    def _ensure_explainer():
        if state["explainer"] is None:
            if RealtimeSHAPExplainer is None:
                raise HTTPException(503, "RealtimeSHAPExplainer module not available")
            model = _ensure_model()
            state["explainer"] = RealtimeSHAPExplainer(
                model, state["feature_names"]
            )
        return state["explainer"]

    def _ensure_formatter():
        if state["formatter"] is None:
            if ExplanationFormatter is None:
                raise HTTPException(503, "ExplanationFormatter module not available")
            state["formatter"] = ExplanationFormatter()
        return state["formatter"]

    def _ensure_pricing():
        if state["pricing_engine"] is None:
            if RiskPricingEngine is None:
                raise HTTPException(503, "RiskPricingEngine module not available")
            state["pricing_engine"] = RiskPricingEngine()
        return state["pricing_engine"]

    def _ensure_drift():
        if state["drift_detector"] is None:
            if DriftDetector is None:
                raise HTTPException(503, "DriftDetector module not available")
            state["drift_detector"] = DriftDetector()
        return state["drift_detector"]

    def _ensure_tracker():
        if state["tracker"] is None:
            if MLflowTracker is None:
                raise HTTPException(503, "MLflowTracker module not available")
            state["tracker"] = MLflowTracker()
        return state["tracker"]

    def _ensure_banking_api():
        if state["banking_api"] is None:
            if BankingAPIClient is None:
                raise HTTPException(503, "BankingAPIClient module not available")
            state["banking_api"] = BankingAPIClient(api_base_url="https://api.banking.example.com")
        return state["banking_api"]

    def _ensure_txn_processor():
        if state["txn_processor"] is None:
            if TransactionFeatureEngineer is None:
                raise HTTPException(503, "TransactionFeatureEngineer module not available")
            state["txn_processor"] = TransactionFeatureEngineer()
        return state["txn_processor"]

    def _features_from_request(req: LoanApplicationRequest) -> Dict[str, Any]:
        """Convert a request to a flat feature dict."""
        mapping = {
            "age": req.age,
            "income": req.income,
            "employment_length": req.employment_length,
            "loan_amount": req.loan_amount,
            "interest_rate": req.interest_rate,
            "term_months": req.term_months,
            "debt_to_income": req.debt_to_income,
            "credit_score": req.credit_score,
            "num_credit_lines": req.num_credit_lines,
            "num_derogatory_marks": req.num_derogatory_marks,
            "revolving_utilization": req.revolving_utilization,
            "total_accounts": req.total_accounts,
            "months_since_last_delinquency": req.months_since_last_delinquency,
        }
        if req.extra_features:
            mapping.update(req.extra_features)
        return {k: (float(v) if v is not None else 0.0) for k, v in mapping.items()}

    def _heuristic_predict(features: Dict[str, Any]) -> Dict[str, Any]:
        """Fast heuristic prediction (no model needed)."""
        score = 0.5
        credit_score = features.get("credit_score", 650)
        dti = features.get("debt_to_income", 0.3)
        income = features.get("income", 50000)
        derogs = features.get("num_derogatory_marks", 0)

        score -= (credit_score - 650) / 1000
        score += dti * 0.2
        score -= min(income / 500000, 0.1)
        score += derogs * 0.05
        score = max(0.01, min(0.99, score))

        if score < 0.3:
            cat = "low_risk"
        elif score < 0.6:
            cat = "medium_risk"
        else:
            cat = "high_risk"

        return {
            "probability": round(score, 4),
            "risk_category": cat,
            "method": "heuristic",
        }

    # ------------------------------------------------------------------ #
    #  Middleware – request logging                                       #
    # ------------------------------------------------------------------ #

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000
        logger.info(
            "%s %s → %s (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    # ------------------------------------------------------------------ #
    #  Endpoints                                                          #
    # ------------------------------------------------------------------ #

    @app.post("/api/v1/predict", response_model=PredictionResponse)
    async def predict_full(req: LoanApplicationRequest):
        """Full prediction with explanation + pricing."""
        t0 = time.time()
        try:
            model = _ensure_model()
            features = _features_from_request(req)
            feat_df = pd.DataFrame([features])[state["feature_names"]]
            proba = float(model.predict_proba(feat_df)[0, 1])
            cat = "low_risk" if proba < 0.3 else "medium_risk" if proba < 0.6 else "high_risk"
            confidence = round(1 - abs(proba - 0.5) * 2, 4)

            pred_result = {
                "probability": proba,
                "risk_category": cat,
                "confidence": confidence,
                "model_version": "1.0",
            }

            explanation = None
            try:
                explainer = _ensure_explainer()
                explanation = explainer.explain_prediction(features)
            except Exception as exc:
                logger.warning("SHAP explanation failed: %s", exc)

            pricing = None
            try:
                pricer = _ensure_pricing()
                pricing = pricer.calculate_price(
                    probability=proba,
                    loan_amount=req.loan_amount or 100000,
                    loan_term_months=req.term_months or 60,
                )
            except Exception as exc:
                logger.warning("Pricing failed: %s", exc)

            return PredictionResponse(
                success=True,
                probability=proba,
                risk_category=cat,
                confidence=confidence,
                explanation=explanation,
                pricing=pricing if isinstance(pricing, dict) else {"rate": None},
                model_version="1.0",
                timestamp=datetime.now(timezone.utc).isoformat(),
                processing_time_ms=round((time.time() - t0) * 1000, 2),
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(500, detail=str(exc))

    @app.post("/api/v1/predict/quick", response_model=PredictionResponse)
    async def predict_quick(req: QuickPredictionRequest):
        """Quick heuristic prediction (no model load)."""
        t0 = time.time()
        features = {
            "age": req.age or 35,
            "income": req.income or 50000,
            "loan_amount": req.loan_amount or 100000,
            "credit_score": req.credit_score or 650,
            "debt_to_income": req.debt_to_income or 0.3,
        }
        result = _heuristic_predict(features)
        return PredictionResponse(
            success=True,
            probability=result["probability"],
            risk_category=result["risk_category"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            processing_time_ms=round((time.time() - t0) * 1000, 2),
        )

    @app.post("/api/v1/predict/batch")
    async def predict_batch(req: BatchPredictionRequest):
        """Batch predictions."""
        results = []
        for app_req in req.applications:
            try:
                resp = await predict_full(app_req)
                results.append(resp.model_dump())
            except HTTPException as exc:
                results.append({"success": False, "error": exc.detail})
        return {"results": results, "count": len(results)}

    @app.post("/api/v1/explain")
    async def explain(req: ExplainRequest):
        """Detailed SHAP explanation."""
        t0 = time.time()
        try:
            explainer = _ensure_explainer()
            shap_result = explainer.explain_prediction(req.features, top_n=req.top_n)
            pred = req.prediction_result or {"probability": 0.5, "risk_category": "unknown"}
            formatter = _ensure_formatter()
            fmt_map = {
                "loan_officer": formatter.format_for_loan_officer,
                "risk_manager": formatter.format_for_risk_manager,
                "executive": formatter.format_for_executive,
                "auditor": formatter.format_for_auditor,
                "customer": formatter.format_for_customer,
                "api": formatter.format_for_api,
            }
            formatted = fmt_map.get(req.format_for, formatter.format_for_api)(shap_result, pred)
            return {
                "success": True,
                "explanation": shap_result,
                "formatted": formatted,
                "processing_time_ms": round((time.time() - t0) * 1000, 2),
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Explanation failed")
            raise HTTPException(500, detail=str(exc))

    @app.post("/api/v1/explain/counterfactual")
    async def explain_counterfactual(req: CounterfactualRequest):
        """Counterfactual explanation."""
        t0 = time.time()
        try:
            explainer = _ensure_explainer()
            cf = explainer.explain_counterfactual(
                req.features,
                target_probability=req.target_probability,
                max_changes=req.max_changes,
            )
            return {
                "success": True,
                "counterfactual": cf,
                "processing_time_ms": round((time.time() - t0) * 1000, 2),
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Counterfactual explanation failed")
            raise HTTPException(500, detail=str(exc))

    @app.post("/api/v1/price")
    async def price(req: PricingRequest):
        """Risk-based pricing recommendation."""
        t0 = time.time()
        try:
            pricer = _ensure_pricing()
            kwargs: Dict[str, Any] = {}
            if req.probability is not None:
                kwargs["probability"] = req.probability
            if req.loan_amount is not None:
                kwargs["loan_amount"] = req.loan_amount
            if req.loan_term_months is not None:
                kwargs["loan_term_months"] = req.loan_term_months
            if req.base_rate is not None:
                kwargs["base_rate"] = req.base_rate
            result = pricer.calculate_price(**kwargs)
            return {
                "success": True,
                "pricing": result if isinstance(result, dict) else {"recommendation": result},
                "processing_time_ms": round((time.time() - t0) * 1000, 2),
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Pricing failed")
            raise HTTPException(500, detail=str(exc))

    @app.post("/api/v1/transactions/analyze")
    async def analyze_transactions(req: TransactionAnalysisRequest):
        """Analyse banking transactions."""
        t0 = time.time()
        try:
            proc = _ensure_txn_processor()
            txn_df = pd.DataFrame(req.transactions)
            features = proc.extract_features(txn_df)
            return {
                "success": True,
                "customer_id": req.customer_id,
                "features": features if isinstance(features, dict) else features.to_dict(),
                "n_transactions": len(req.transactions),
                "processing_time_ms": round((time.time() - t0) * 1000, 2),
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Transaction analysis failed")
            raise HTTPException(500, detail=str(exc))

    @app.get("/api/v1/transactions/features/{customer_id}")
    async def get_transaction_features(customer_id: str):
        """Get transaction-based features for a customer."""
        t0 = time.time()
        try:
            proc = _ensure_txn_processor()
            features = proc.get_customer_features(customer_id)
            return {
                "success": True,
                "customer_id": customer_id,
                "features": features if isinstance(features, dict) else features.to_dict(),
                "processing_time_ms": round((time.time() - t0) * 1000, 2),
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Feature fetch failed")
            raise HTTPException(500, detail=str(exc))

    @app.post("/api/v1/drift/check")
    async def drift_check(req: DriftCheckRequest):
        """Trigger drift detection."""
        t0 = time.time()
        try:
            detector = _ensure_drift()
            if req.current_data:
                current_df = pd.DataFrame(req.current_data)
            else:
                current_df = None
            result = detector.check_drift(
                reference_path=req.reference_data_path,
                current_data=current_df,
            )
            return {
                "success": True,
                "drift_result": result if isinstance(result, dict) else {"details": str(result)},
                "processing_time_ms": round((time.time() - t0) * 1000, 2),
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Drift check failed")
            raise HTTPException(500, detail=str(exc))

    @app.get("/api/v1/drift/status")
    async def drift_status():
        """Get drift monitoring status."""
        try:
            detector = _ensure_drift()
            status = detector.get_status() if hasattr(detector, "get_status") else {"status": "available"}
            return {"success": True, "status": status}
        except HTTPException:
            raise
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @app.get("/api/v1/model/compare")
    async def model_compare():
        """Compare ensemble model performance."""
        try:
            model = _ensure_model()
            info = model.get_model_info() if hasattr(model, "get_model_info") else {}
            return {"success": True, "model_info": info}
        except HTTPException:
            raise
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @app.get("/api/v1/market/rates")
    async def market_rates():
        """Current market rates."""
        return {
            "success": True,
            "rates": {
                "repo_rate": 6.50,
                "inflation_rate": 5.1,
                "prime_lending_rate": 9.15,
                "base_rate": 8.75,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    @app.get("/api/v1/market/repo-rate")
    async def repo_rate():
        """RBI repo rate."""
        return {
            "success": True,
            "repo_rate": 6.50,
            "source": "RBI",
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    @app.post("/api/v1/banking/submit-decision")
    async def submit_decision(req: BankingSubmitRequest):
        """Submit credit decision to the banking system."""
        t0 = time.time()
        try:
            api = _ensure_banking_api()
            result = api.submit_decision(
                customer_id=req.customer_id,
                decision=req.decision,
                probability=req.probability,
                risk_category=req.risk_category,
                explanation_id=req.explanation_id,
                notes=req.notes,
            )
            return {
                "success": True,
                "submission_result": result if isinstance(result, dict) else {"status": str(result)},
                "processing_time_ms": round((time.time() - t0) * 1000, 2),
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Decision submission failed")
            raise HTTPException(500, detail=str(exc))

    @app.post("/api/v1/banking/fetch-profile")
    async def fetch_profile(req: BankingProfileRequest):
        """Fetch customer profile from the banking API."""
        t0 = time.time()
        try:
            api = _ensure_banking_api()
            profile = api.fetch_customer_profile(
                req.customer_id,
                include_transactions=req.include_transactions,
            )
            return {
                "success": True,
                "profile": profile if isinstance(profile, dict) else {"data": profile},
                "processing_time_ms": round((time.time() - t0) * 1000, 2),
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Profile fetch failed")
            raise HTTPException(500, detail=str(exc))

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health():
        """Health check."""
        return HealthResponse(
            status="healthy",
            model_loaded=state["model"] is not None,
            explainer_loaded=state["explainer"] is not None,
            pricing_loaded=state["pricing_engine"] is not None,
            drift_detector_loaded=state["drift_detector"] is not None,
            tracker_loaded=state["tracker"] is not None,
            banking_api_loaded=state["banking_api"] is not None,
            transaction_processor_loaded=state["txn_processor"] is not None,
            uptime_seconds=round(time.time() - _start_time, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @app.get("/api/v1/model/info")
    async def model_info():
        """Model information."""
        try:
            model = _ensure_model()
            info = {
                "model_type": type(model).__name__,
                "feature_names": state["feature_names"],
                "n_features": len(state["feature_names"]),
            }
            if hasattr(model, "get_model_info"):
                info.update(model.get_model_info())
            return {"success": True, "model_info": info}
        except HTTPException:
            raise
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @app.post("/api/v1/loan/process")
    async def process_loan(req: LoanProcessRequest):
        """Full loan processing pipeline: predict → price → optionally submit."""
        t0 = time.time()
        pipeline_result: Dict[str, Any] = {
            "steps": [],
            "success": True,
        }

        # Step 1 – Prediction
        try:
            pred_resp = await predict_full(req.application)
            pipeline_result["prediction"] = pred_resp.model_dump()
            pipeline_result["steps"].append("prediction")
        except Exception as exc:
            pipeline_result["success"] = False
            pipeline_result["error"] = f"Prediction failed: {exc}"
            return pipeline_result

        # Step 2 – Pricing
        try:
            price_req = PricingRequest(
                features=_features_from_request(req.application),
                probability=pred_resp.probability,
                loan_amount=req.application.loan_amount,
                loan_term_months=req.application.term_months,
            )
            price_resp = await price(price_req)
            pipeline_result["pricing"] = price_resp
            pipeline_result["steps"].append("pricing")
        except Exception as exc:
            logger.warning("Pricing step failed in pipeline: %s", exc)
            pipeline_result["pricing"] = {"error": str(exc)}

        # Step 3 – Submit (optional)
        if req.auto_submit and req.application.customer_id:
            try:
                submit = BankingSubmitRequest(
                    customer_id=req.application.customer_id,
                    decision="approve" if (pred_resp.probability or 0.5) < 0.5 else "decline",
                    probability=pred_resp.probability or 0.5,
                    risk_category=pred_resp.risk_category or "unknown",
                    explanation_id=None,
                    notes="Auto-submitted via /loan/process pipeline",
                )
                submit_resp = await submit_decision(submit)
                pipeline_result["submission"] = submit_resp
                pipeline_result["steps"].append("submission")
            except Exception as exc:
                logger.warning("Submission step failed in pipeline: %s", exc)
                pipeline_result["submission"] = {"error": str(exc)}

        pipeline_result["processing_time_ms"] = round((time.time() - t0) * 1000, 2)
        pipeline_result["timestamp"] = datetime.now(timezone.utc).isoformat()
        return pipeline_result

    return app


# ------------------------------------------------------------------ #
#  Module-level app instance for uvicorn                              #
# ------------------------------------------------------------------ #

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
