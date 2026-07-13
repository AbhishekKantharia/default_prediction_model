"""
Risk-Based Pricing Engine

Computes risk-adjusted interest rates, expected losses, EMI schedules,
and full pricing recommendations based on predicted default probability.
"""
import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskPricingEngine:
    """Calculates risk-adjusted interest rates and pricing recommendations."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = self._load_config(config_path)
        pricing_config = self.config.get("pricing", {})
        self.repo_rate = pricing_config.get("repo_rate", 0.065)
        self.bank_spread = pricing_config.get("bank_spread", 0.02)
        self.risk_spread_base = pricing_config.get("risk_spread_base", 0.015)
        self.base_rate = self.repo_rate + self.bank_spread + self.risk_spread_base
        self.segment_adjustments = pricing_config.get("segment_adjustments", {
            "prime": -0.015,
            "near_prime": 0.0,
            "subprime": 0.03,
            "deep_subprime": 0.06,
        })
        self.loan_type_adjustments = pricing_config.get("loan_type_adjustments", {
            "mortgage": -0.02,
            "auto": -0.01,
            "business": 0.005,
            "personal": 0.01,
            "student": -0.005,
            "credit_card": 0.02,
        })
        self.min_rate = pricing_config.get("min_rate", 0.05)
        self.max_rate = pricing_config.get("max_rate", 0.28)

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        try:
            import yaml
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Config not found at %s, using defaults", config_path)
            return {}

    @staticmethod
    def _clip_rate(rate: float, min_rate: float, max_rate: float) -> float:
        return float(np.clip(rate, min_rate, max_rate))

    def compute_risk_adjusted_rate(
        self,
        default_prob: float,
        loan_type: str,
        segment: str,
        loan_amount: float,
        tenure_months: int,
        credit_score: int,
        dti: float,
    ) -> Dict[str, Any]:
        """
        Calculate risk-based interest rate for a loan application.

        Args:
            default_prob: Predicted probability of default (0-1).
            loan_type: Type of loan (mortgage, auto, personal, etc.).
            segment: Borrower segment (prime, subprime, near_prime, deep_subprime).
            loan_amount: Requested loan amount.
            tenure_months: Loan tenure in months.
            credit_score: Borrower credit score (300-850).
            dti: Debt-to-income ratio (0-1).

        Returns:
            Dictionary with recommended rate, range, premium, and breakdown.
        """
        risk_premium = default_prob * 2.5
        segment_adj = self.segment_adjustments.get(segment, 0.0)
        loan_type_adj = self.loan_type_adjustments.get(loan_type, 0.0)

        if credit_score >= 750:
            credit_adj = -0.01
        elif credit_score >= 700:
            credit_adj = -0.005
        elif credit_score >= 650:
            credit_adj = 0.0
        elif credit_score >= 600:
            credit_adj = 0.01
        else:
            credit_adj = 0.02

        if dti <= 0.2:
            dti_adj = -0.005
        elif dti <= 0.35:
            dti_adj = 0.0
        elif dti <= 0.5:
            dti_adj = 0.005
        else:
            dti_adj = 0.015

        loan_size_adj = 0.0
        if loan_amount > 500000:
            loan_size_adj = -0.005
        elif loan_amount < 50000:
            loan_size_adj = 0.005

        if tenure_months <= 12:
            tenure_adj = 0.005
        elif tenure_months <= 60:
            tenure_adj = 0.0
        else:
            tenure_adj = 0.003

        recommended_rate = (
            self.base_rate
            + risk_premium
            + segment_adj
            + loan_type_adj
            + credit_adj
            + dti_adj
            + loan_size_adj
            + tenure_adj
        )
        recommended_rate = self._clip_rate(
            recommended_rate, self.min_rate, self.max_rate
        )
        rate_range_min = self._clip_rate(
            recommended_rate - 0.015, self.min_rate, self.max_rate
        )
        rate_range_max = self._clip_rate(
            recommended_rate + 0.015, self.min_rate, self.max_rate
        )

        result: Dict[str, Any] = {
            "recommended_rate": round(recommended_rate, 6),
            "rate_range": {
                "min": round(rate_range_min, 6),
                "max": round(rate_range_max, 6),
            },
            "risk_premium": round(risk_premium, 6),
            "breakdown": {
                "base_rate": round(self.base_rate, 6),
                "repo_rate": round(self.repo_rate, 6),
                "bank_spread": round(self.bank_spread, 6),
                "risk_spread_base": round(self.risk_spread_base, 6),
                "default_risk_premium": round(risk_premium, 6),
                "segment_adjustment": round(segment_adj, 6),
                "loan_type_adjustment": round(loan_type_adj, 6),
                "credit_score_adjustment": round(credit_adj, 6),
                "dti_adjustment": round(dti_adj, 6),
                "loan_size_adjustment": round(loan_size_adj, 6),
                "tenure_adjustment": round(tenure_adj, 6),
            },
            "input_parameters": {
                "default_prob": round(default_prob, 6),
                "loan_type": loan_type,
                "segment": segment,
                "loan_amount": loan_amount,
                "tenure_months": tenure_months,
                "credit_score": credit_score,
                "dti": round(dti, 6),
            },
        }

        logger.info(
            "Risk-adjusted rate: %.4f (PD=%.4f, segment=%s)",
            recommended_rate, default_prob, segment,
        )
        return result

    def compute_expected_loss(
        self,
        default_prob: float,
        lgd: float,
        ead: float,
    ) -> Dict[str, Any]:
        """
        Calculate expected loss = PD * LGD * EAD.

        Args:
            default_prob: Probability of default (0-1).
            lgd: Loss given default (0-1). Typical: 0.45 unsecured, 0.25 secured.
            ead: Exposure at default (loan balance + undrawn commitment).

        Returns:
            Expected loss amount and components.
        """
        expected_loss = default_prob * lgd * ead
        return {
            "expected_loss": round(expected_loss, 2),
            "default_probability": round(default_prob, 6),
            "lgd": round(lgd, 6),
            "ead": round(ead, 2),
            "expected_loss_rate": round(default_prob * lgd, 6),
        }

    def compute_risk_margin(
        self,
        expected_loss: float,
        operating_cost_ratio: float = 0.03,
    ) -> Dict[str, Any]:
        """
        Calculate the required risk margin over base rate.

        Args:
            expected_loss: Expected loss amount.
            operating_cost_ratio: Operating cost as a fraction of loan amount.

        Returns:
            Risk margin details.
        """
        risk_margin = expected_loss * 0.1 + operating_cost_ratio
        return {
            "risk_margin": round(risk_margin, 6),
            "expected_loss_contribution": round(expected_loss * 0.1, 6),
            "operating_cost_contribution": round(operating_cost_ratio, 6),
        }

    def recommend_pricing(
        self,
        default_prob: float,
        loan_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate full pricing recommendation with optimal, competitive, and
        minimum acceptable rates plus EMI comparisons and profitability.

        Args:
            default_prob: Predicted default probability.
            loan_details: Dictionary containing:
                - loan_type: str
                - segment: str
                - loan_amount: float
                - tenure_months: int
                - credit_score: int
                - dti: float
                - lgd: float (optional, defaults based on secured/unsecured)
                - operating_cost_ratio: float (optional, default 0.03)

        Returns:
            Complete pricing recommendation.
        """
        loan_type = loan_details.get("loan_type", "personal")
        segment = loan_details.get("segment", "near_prime")
        loan_amount = loan_details.get("loan_amount", 100000)
        tenure_months = loan_details.get("tenure_months", 60)
        credit_score = loan_details.get("credit_score", 700)
        dti = loan_details.get("dti", 0.3)
        operating_cost_ratio = loan_details.get("operating_cost_ratio", 0.03)

        secured_types = {"mortgage", "auto"}
        lgd = loan_details.get(
            "lgd", 0.25 if loan_type in secured_types else 0.45
        )
        ead = loan_amount * 1.0

        rate_info = self.compute_risk_adjusted_rate(
            default_prob, loan_type, segment,
            loan_amount, tenure_months, credit_score, dti,
        )
        optimal_rate = rate_info["recommended_rate"]

        competitive_rate = self._clip_rate(
            optimal_rate - 0.005, self.min_rate, self.max_rate
        )
        minimum_acceptable_rate = self._clip_rate(
            optimal_rate - 0.02, self.min_rate, self.max_rate
        )

        el_info = self.compute_expected_loss(default_prob, lgd, ead)
        risk_margin_info = self.compute_risk_margin(
            el_info["expected_loss"], operating_cost_ratio
        )

        emi_calc = EMICalculator()
        emi_optimal = emi_calc.compute_emi(loan_amount, optimal_rate, tenure_months)
        emi_competitive = emi_calc.compute_emi(
            loan_amount, competitive_rate, tenure_months
        )
        emi_minimum = emi_calc.compute_emi(
            loan_amount, minimum_acceptable_rate, tenure_months
        )

        total_interest_optimal = emi_optimal * tenure_months - loan_amount
        total_interest_competitive = emi_competitive * tenure_months - loan_amount
        total_interest_minimum = emi_minimum * tenure_months - loan_amount

        avg_annual_balance = loan_amount * 0.6
        interest_income_optimal = loan_amount * optimal_rate * (
            tenure_months / 12.0
        )
        operating_cost = loan_amount * operating_cost_ratio
        expected_loss = el_info["expected_loss"]
        expected_profit_optimal = (
            interest_income_optimal - expected_loss - operating_cost
        )
        risk_adjusted_return = (
            expected_profit_optimal / max(loan_amount, 1)
        )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "optimal_rate": round(optimal_rate, 6),
            "competitive_rate": round(competitive_rate, 6),
            "minimum_acceptable_rate": round(minimum_acceptable_rate, 6),
            "rate_range": rate_info["rate_range"],
            "risk_premium": rate_info["risk_premium"],
            "rate_breakdown": rate_info["breakdown"],
            "emi_comparison": {
                "optimal": {
                    "rate": round(optimal_rate, 6),
                    "monthly_emi": round(emi_optimal, 2),
                    "total_interest": round(total_interest_optimal, 2),
                    "total_payment": round(emi_optimal * tenure_months, 2),
                },
                "competitive": {
                    "rate": round(competitive_rate, 6),
                    "monthly_emi": round(emi_competitive, 2),
                    "total_interest": round(total_interest_competitive, 2),
                    "total_payment": round(emi_competitive * tenure_months, 2),
                },
                "minimum_acceptable": {
                    "rate": round(minimum_acceptable_rate, 6),
                    "monthly_emi": round(emi_minimum, 2),
                    "total_interest": round(total_interest_minimum, 2),
                    "total_payment": round(emi_minimum * tenure_months, 2),
                },
            },
            "profitability_analysis": {
                "expected_loss": el_info["expected_loss"],
                "expected_loss_rate": el_info["expected_loss_rate"],
                "interest_income": round(interest_income_optimal, 2),
                "operating_cost": round(operating_cost, 2),
                "expected_profit": round(expected_profit_optimal, 2),
                "risk_adjusted_return": round(risk_adjusted_return, 6),
                "risk_margin": risk_margin_info,
            },
            "risk_metrics": {
                "default_probability": round(default_prob, 6),
                "lgd": lgd,
                "ead": round(ead, 2),
                "expected_loss": el_info["expected_loss"],
                "credit_score": credit_score,
                "dti": round(dti, 6),
                "segment": segment,
                "loan_type": loan_type,
            },
        }

    def generate_pricing_report(
        self,
        customer_data: Dict[str, Any],
        prediction_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a full pricing report for a customer.

        Args:
            customer_data: Customer information including loan details.
            prediction_results: Model prediction results including default_prob.

        Returns:
            Comprehensive pricing report.
        """
        default_prob = prediction_results.get(
            "default_probability",
            prediction_results.get("probability", 0.05),
        )

        loan_details = {
            "loan_type": customer_data.get("loan_type", "personal"),
            "segment": customer_data.get("borrower_segment", "near_prime"),
            "loan_amount": customer_data.get("loan_amount", 100000),
            "tenure_months": customer_data.get("tenure_months", 60),
            "credit_score": customer_data.get("credit_score", 700),
            "dti": customer_data.get("debt_to_income", 0.3),
        }

        pricing = self.recommend_pricing(default_prob, loan_details)

        emi_calc = EMICalculator()
        amortization = emi_calc.compute_amortization_schedule(
            loan_details["loan_amount"],
            pricing["optimal_rate"],
            loan_details["tenure_months"],
        )

        first_12 = amortization[:12] if len(amortization) >= 12 else amortization
        summary_schedule = []
        for row in first_12:
            summary_schedule.append({
                "month": int(row["month"]),
                "emi": round(float(row["emi"]), 2),
                "principal": round(float(row["principal"]), 2),
                "interest": round(float(row["interest"]), 2),
                "balance": round(float(row["balance"]), 2),
            })

        report: Dict[str, Any] = {
            "report_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.utcnow().isoformat(),
            "customer_id": customer_data.get("customer_id", "unknown"),
            "loan_id": customer_data.get("loan_id", "unknown"),
            "pricing_recommendation": pricing,
            "amortization_summary": summary_schedule,
            "total_months": loan_details["tenure_months"],
            "loan_amount": loan_details["loan_amount"],
        }

        report_path = f"data/pricing/report_{report['report_id']}.json"
        import os
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Pricing report saved to %s", report_path)

        return report

    def compare_with_market(
        self,
        recommended_rate: float,
        competitor_rates: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Compare recommended rate with competitor rates.

        Args:
            recommended_rate: Our recommended interest rate.
            competitor_rates: Dictionary mapping competitor names to rates.

        Returns:
            Comparison analysis.
        """
        rates = list(competitor_rates.values())
        market_mean = np.mean(rates) if rates else 0.0
        market_median = np.median(rates) if rates else 0.0
        market_min = min(rates) if rates else 0.0
        market_max = max(rates) if rates else 0.0

        position = sum(1 for r in rates if recommended_rate <= r) + 1
        percentile = (position / (len(rates) + 1)) * 100

        return {
            "recommended_rate": round(recommended_rate, 6),
            "market_statistics": {
                "mean": round(float(market_mean), 6),
                "median": round(float(market_median), 6),
                "min": round(float(market_min), 6),
                "max": round(float(market_max), 6),
                "n_competitors": len(rates),
            },
            "our_position": {
                "rank": position,
                "percentile": round(float(percentile), 2),
                "below_market_average": recommended_rate < market_mean,
                "difference_from_mean": round(
                    float(recommended_rate - market_mean), 6
                ),
            },
            "competitor_comparison": {
                name: {
                    "rate": round(rate, 6),
                    "difference": round(recommended_rate - rate, 6),
                    "our_rate_is_lower": recommended_rate < rate,
                }
                for name, rate in competitor_rates.items()
            },
            "market_assessment": (
                "competitive" if recommended_rate < market_mean
                else "above_market"
            ),
        }


class EMICalculator:
    """Standard EMI calculation and amortization utilities."""

    @staticmethod
    def compute_emi(
        principal: float,
        annual_rate: float,
        tenure_months: int,
    ) -> float:
        """
        Calculate monthly EMI using the standard formula.

        EMI = P * r * (1+r)^n / ((1+r)^n - 1)

        Args:
            principal: Loan principal amount.
            annual_rate: Annual interest rate (e.g., 0.10 for 10%).
            tenure_months: Loan tenure in months.

        Returns:
            Monthly EMI amount.
        """
        if principal <= 0:
            return 0.0
        if tenure_months <= 0:
            return principal
        if annual_rate <= 0:
            return principal / tenure_months

        monthly_rate = annual_rate / 12.0
        factor = (1 + monthly_rate) ** tenure_months
        emi = principal * monthly_rate * factor / (factor - 1)
        return float(emi)

    def compute_amortization_schedule(
        self,
        principal: float,
        annual_rate: float,
        tenure_months: int,
    ) -> List[Dict[str, float]]:
        """
        Generate full amortization schedule.

        Args:
            principal: Loan principal amount.
            annual_rate: Annual interest rate.
            tenure_months: Loan tenure in months.

        Returns:
            List of monthly payment records.
        """
        emi = self.compute_emi(principal, annual_rate, tenure_months)
        monthly_rate = annual_rate / 12.0 if annual_rate > 0 else 0.0
        balance = principal
        schedule: List[Dict[str, float]] = []

        for month in range(1, tenure_months + 1):
            interest = balance * monthly_rate
            principal_paid = emi - interest
            if principal_paid > balance:
                principal_paid = balance
                emi = principal_paid + interest
            balance -= principal_paid

            schedule.append({
                "month": month,
                "emi": round(emi, 2),
                "principal": round(principal_paid, 2),
                "interest": round(interest, 2),
                "balance": round(max(balance, 0), 2),
                "cumulative_interest": 0.0,
                "cumulative_principal": 0.0,
            })

        cum_interest = 0.0
        cum_principal = 0.0
        for record in schedule:
            cum_interest += record["interest"]
            cum_principal += record["principal"]
            record["cumulative_interest"] = round(cum_interest, 2)
            record["cumulative_principal"] = round(cum_principal, 2)

        return schedule

    def compute_total_interest(
        self,
        principal: float,
        annual_rate: float,
        tenure_months: int,
    ) -> float:
        """
        Compute total interest paid over the life of the loan.

        Args:
            principal: Loan principal amount.
            annual_rate: Annual interest rate.
            tenure_months: Loan tenure in months.

        Returns:
            Total interest amount.
        """
        emi = self.compute_emi(principal, annual_rate, tenure_months)
        total_payment = emi * tenure_months
        return round(total_payment - principal, 2)

    def compute_prepayment_impact(
        self,
        principal: float,
        annual_rate: float,
        tenure_months: int,
        prepayment_amount: float,
        prepayment_month: int,
    ) -> Dict[str, Any]:
        """
        Calculate impact of prepayment on loan.

        Args:
            principal: Original loan principal.
            annual_rate: Annual interest rate.
            tenure_months: Original loan tenure.
            prepayment_amount: Amount to prepay.
            prepayment_month: Month at which prepayment is made.

        Returns:
            Old vs new EMI, tenure, and total interest comparison.
        """
        old_emi = self.compute_emi(principal, annual_rate, tenure_months)
        old_total_interest = self.compute_total_interest(
            principal, annual_rate, tenure_months
        )
        old_total_payment = old_emi * tenure_months

        monthly_rate = annual_rate / 12.0 if annual_rate > 0 else 0.0
        balance = principal

        for month in range(1, min(prepayment_month, tenure_months) + 1):
            interest = balance * monthly_rate
            principal_paid = old_emi - interest
            balance -= principal_paid

        new_balance = max(balance - prepayment_amount, 0)
        remaining_tenure = tenure_months - prepayment_month

        if new_balance <= 0 or remaining_tenure <= 0:
            new_emi = 0.0
            new_tenure = prepayment_month
            new_total_interest = old_total_interest
        else:
            new_emi = self.compute_emi(new_balance, annual_rate, remaining_tenure)
            new_total_interest = self.compute_total_interest(
                new_balance, annual_rate, remaining_tenure
            )
            interest_already_paid = old_total_interest * (
                prepayment_month / tenure_months
            ) if tenure_months > 0 else 0.0
            payments_already_made = old_emi * prepayment_month
            new_total_interest += (
                payments_already_made - principal + new_balance - prepayment_amount
            )
            new_total_interest = max(new_total_interest, 0)
            new_tenure = prepayment_month + remaining_tenure

        interest_saved = old_total_interest - new_total_interest

        return {
            "original": {
                "emi": round(old_emi, 2),
                "tenure_months": tenure_months,
                "total_interest": round(old_total_interest, 2),
                "total_payment": round(old_total_payment, 2),
            },
            "after_prepayment": {
                "new_emi": round(new_emi, 2),
                "new_tenure_months": new_tenure,
                "new_total_interest": round(new_total_interest, 2),
                "new_total_payment": round(
                    new_total_interest + principal - prepayment_amount, 2
                ),
                "remaining_balance": round(new_balance, 2),
            },
            "impact": {
                "prepayment_amount": round(prepayment_amount, 2),
                "interest_saved": round(interest_saved, 2),
                "tenure_reduction_months": max(
                    tenure_months - new_tenure, 0
                ),
                "monthly_emi_reduction": round(
                    old_emi - new_emi if new_emi > 0 else old_emi, 2
                ),
            },
        }
