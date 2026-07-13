"""
Banking API Client for Live Loan Processing Integration.

Provides ``BankingAPIClient`` for interacting with core banking systems
and ``AABankingIntegration`` for the RBI Account Aggregator framework.

All HTTP calls use ``requests`` when available, falling back to stdlib
``urllib`` so the module remains importable in restricted environments.
"""

import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP abstraction (requests or urllib fallback)
# ---------------------------------------------------------------------------

try:
    import requests as _requests

    _HAS_REQUESTS = True
except ImportError:
    _requests = None  # type: ignore[assignment]
    _HAS_REQUESTS = False


def _http_request(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    data: Optional[bytes] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Issue an HTTP request using requests or urllib as fallback.

    Args:
        method: HTTP verb (GET, POST, PUT, DELETE, PATCH).
        url: Full request URL.
        headers: Optional request headers.
        params: Optional query parameters (GET).
        json_body: Optional JSON body (POST/PUT/PATCH).
        data: Optional raw bytes body.
        timeout: Request timeout in seconds.

    Returns:
        Dictionary with ``status_code``, ``body`` (parsed JSON or text),
        and ``headers``.

    Raises:
        ConnectionError: If the request fails at the network level.
    """
    merged_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        merged_headers.update(headers)

    if _HAS_REQUESTS and _requests is not None:
        resp = _requests.request(
            method,
            url,
            headers=merged_headers,
            params=params,
            json=json_body,
            data=data,
            timeout=timeout,
        )
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return {
            "status_code": resp.status_code,
            "body": body,
            "headers": dict(resp.headers),
        }

    # urllib fallback
    full_url = url
    if params:
        full_url = f"{url}?{urlencode(params)}"

    body_bytes = None
    if json_body is not None:
        body_bytes = json.dumps(json_body).encode("utf-8")
    elif data is not None:
        body_bytes = data

    req = Request(full_url, data=body_bytes, headers=merged_headers, method=method.upper())

    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                body = json.loads(raw)
            except Exception:
                body = raw
            return {
                "status_code": resp.status,
                "body": body,
                "headers": dict(resp.headers),
            }
    except HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp else ""
        try:
            body = json.loads(raw)
        except Exception:
            body = raw
        return {
            "status_code": exc.code,
            "body": body,
            "headers": {},
        }
    except URLError as exc:
        raise ConnectionError(f"HTTP request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# BankingAPIClient
# ---------------------------------------------------------------------------

class BankingAPIClient:
    """Client for interacting with the core banking API.

    Handles loan application submission, status tracking, customer
    profile retrieval, credit decisions, disbursements, repayment
    schedules, notifications, and market rate queries.

    Attributes:
        api_base_url: Base URL of the banking API.
        api_key: Optional API key for authentication.
        timeout: Default request timeout in seconds.
    """

    def __init__(self, api_base_url: str, api_key: Optional[str] = None, timeout: int = 30) -> None:
        """Initialise the banking API client.

        Args:
            api_base_url: Base URL of the banking API (e.g. ``https://api.bank.com``).
            api_key: Optional API key for Bearer authentication.
            timeout: Default HTTP timeout in seconds.
        """
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        logger.info("BankingAPIClient initialised (base_url=%s)", self.api_base_url)

    def _build_headers(self) -> Dict[str, str]:
        """Build default request headers with optional API key auth.

        Returns:
            Dictionary of HTTP headers.
        """
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Issue an authenticated request to the banking API.

        Args:
            method: HTTP verb.
            path: API path (appended to base URL).
            params: Optional query parameters.
            json_body: Optional JSON body.
            timeout: Override for the default timeout.

        Returns:
            Parsed response dictionary.

        Raises:
            ConnectionError: On network failure.
            ValueError: On non-2xx responses.
        """
        url = urljoin(self.api_base_url + "/", path.lstrip("/"))
        headers = self._build_headers()

        logger.debug("%s %s", method, url)

        response = _http_request(
            method,
            url,
            headers=headers,
            params=params,
            json_body=json_body,
            timeout=timeout or self.timeout,
        )

        if response["status_code"] >= 400:
            logger.error(
                "API error: %s %s -> %d: %s",
                method,
                url,
                response["status_code"],
                response["body"],
            )

        return response

    def check_health(self) -> Dict[str, Any]:
        """Check API health status.

        Returns:
            Dictionary with health check result.
        """
        logger.info("Checking API health")
        try:
            response = self._request("GET", "/api/v1/health")
            return {
                "status": "healthy" if response["status_code"] == 200 else "unhealthy",
                "status_code": response["status_code"],
                "details": response["body"],
                "check_time": datetime.utcnow().isoformat(),
            }
        except ConnectionError as exc:
            return {
                "status": "unreachable",
                "error": str(exc),
                "check_time": datetime.utcnow().isoformat(),
            }

    def submit_loan_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a loan application to the banking system.

        Validates the input data against the expected schema before
        submission and returns an application ID on success.

        Args:
            application_data: Loan application details including applicant
                information, loan amount, purpose, etc.

        Returns:
            Dictionary with ``application_id``, ``status``, and submission
            details.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        logger.info("Submitting loan application")

        required_fields = [
            "customer_id", "loan_amount", "interest_rate", "loan_type",
            "tenure_months", "employment_status", "annual_income",
        ]
        missing = [f for f in required_fields if f not in application_data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        if application_data["loan_amount"] <= 0:
            raise ValueError("Loan amount must be positive")

        if not (0 < application_data.get("interest_rate", 0) <= 50):
            raise ValueError("Interest rate must be between 0 and 50 percent")

        valid_loan_types = ["personal", "mortgage", "auto", "business", "student", "credit_card"]
        if application_data.get("loan_type") not in valid_loan_types:
            raise ValueError(f"Invalid loan_type. Must be one of: {valid_loan_types}")

        payload = {
            **application_data,
            "submission_timestamp": datetime.utcnow().isoformat(),
            "application_ref": str(uuid.uuid4())[:12],
        }

        response = self._request("POST", "/api/v1/loans/apply", json_body=payload)

        if response["status_code"] in (200, 201):
            result = response["body"] if isinstance(response["body"], dict) else {"raw": response["body"]}
            result.setdefault("application_id", payload["application_ref"])
            result.setdefault("status", "submitted")
            result["submission_time"] = payload["submission_timestamp"]
            logger.info("Loan application submitted: %s", result["application_id"])
            return result

        return {
            "application_id": None,
            "status": "failed",
            "error": response["body"],
            "status_code": response["status_code"],
        }

    def get_loan_status(self, application_id: str) -> Dict[str, Any]:
        """Retrieve the current status of a loan application.

        Args:
            application_id: The application identifier.

        Returns:
            Dictionary with loan status details.
        """
        logger.info("Fetching loan status for %s", application_id)

        response = self._request("GET", f"/api/v1/loans/{application_id}/status")

        if response["status_code"] == 200:
            result = response["body"] if isinstance(response["body"], dict) else {"status": response["body"]}
            result["application_id"] = application_id
            return result

        return {
            "application_id": application_id,
            "status": "unknown",
            "error": response["body"],
        }

    def fetch_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        """Fetch customer profile data from the banking system.

        Args:
            customer_id: Unique customer identifier.

        Returns:
            Dictionary with customer profile information.
        """
        logger.info("Fetching customer profile: %s", customer_id)

        response = self._request("GET", f"/api/v1/customers/{customer_id}")

        if response["status_code"] == 200:
            profile = response["body"] if isinstance(response["body"], dict) else {"data": response["body"]}
            profile["customer_id"] = customer_id
            profile["fetch_time"] = datetime.utcnow().isoformat()
            return profile

        return {
            "customer_id": customer_id,
            "status": "not_found",
            "error": response["body"],
        }

    def fetch_account_balance(self, customer_id: str, account_id: str) -> Dict[str, Any]:
        """Perform a real-time balance check.

        Args:
            customer_id: Unique customer identifier.
            account_id: Unique account identifier.

        Returns:
            Dictionary with current balance information.
        """
        logger.info("Fetching balance for customer=%s account=%s", customer_id, account_id)

        response = self._request(
            "GET",
            f"/api/v1/customers/{customer_id}/accounts/{account_id}/balance",
        )

        if response["status_code"] == 200:
            result = response["body"] if isinstance(response["body"], dict) else {"balance": response["body"]}
            result["customer_id"] = customer_id
            result["account_id"] = account_id
            result["query_time"] = datetime.utcnow().isoformat()
            return result

        return {
            "customer_id": customer_id,
            "account_id": account_id,
            "status": "error",
            "error": response["body"],
        }

    def fetch_transaction_history(
        self,
        customer_id: str,
        account_id: str,
        days: int = 90,
    ) -> Dict[str, Any]:
        """Fetch transaction history for the specified period.

        Args:
            customer_id: Unique customer identifier.
            account_id: Unique account identifier.
            days: Number of historical days to retrieve (default 90).

        Returns:
            Dictionary with transaction list and metadata.
        """
        logger.info(
            "Fetching transactions for customer=%s account=%s (days=%d)",
            customer_id,
            account_id,
            days,
        )

        params = {
            "days": days,
            "from_date": (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d"),
            "to_date": datetime.utcnow().strftime("%Y-%m-%d"),
        }

        response = self._request(
            "GET",
            f"/api/v1/customers/{customer_id}/accounts/{account_id}/transactions",
            params=params,
        )

        if response["status_code"] == 200:
            result = response["body"] if isinstance(response["body"], dict) else {"transactions": response["body"]}
            result["customer_id"] = customer_id
            result["account_id"] = account_id
            result["period_days"] = days
            result["query_time"] = datetime.utcnow().isoformat()
            return result

        return {
            "customer_id": customer_id,
            "account_id": account_id,
            "status": "error",
            "error": response["body"],
        }

    def submit_credit_decision(
        self,
        application_id: str,
        decision: str,
        rate: float,
        reason: str,
    ) -> Dict[str, Any]:
        """Submit a credit decision back to the banking system.

        Args:
            application_id: The loan application identifier.
            decision: Credit decision (``approved``, ``rejected``,
                ``conditional``).
            rate: Offered interest rate (percentage).
            reason: Human-readable decision rationale.

        Returns:
            Dictionary with decision submission confirmation.

        Raises:
            ValueError: If decision or rate are invalid.
        """
        logger.info("Submitting credit decision for %s: %s", application_id, decision)

        valid_decisions = ["approved", "rejected", "conditional", "manual_review"]
        if decision.lower() not in valid_decisions:
            raise ValueError(f"Invalid decision. Must be one of: {valid_decisions}")

        if rate < 0 or rate > 50:
            raise ValueError("Interest rate must be between 0 and 50 percent")

        payload = {
            "application_id": application_id,
            "decision": decision.lower(),
            "offered_rate": rate,
            "reason": reason,
            "decision_timestamp": datetime.utcnow().isoformat(),
            "decision_id": str(uuid.uuid4())[:12],
        }

        response = self._request(
            "POST",
            f"/api/v1/loans/{application_id}/decision",
            json_body=payload,
        )

        if response["status_code"] in (200, 201):
            result = response["body"] if isinstance(response["body"], dict) else {"status": "submitted"}
            result["application_id"] = application_id
            result["decision"] = decision.lower()
            result["decision_id"] = payload["decision_id"]
            logger.info("Credit decision submitted: %s -> %s", application_id, decision)
            return result

        return {
            "application_id": application_id,
            "status": "failed",
            "error": response["body"],
        }

    def trigger_disbursement(
        self,
        loan_id: str,
        amount: float,
        account_details: Dict[str, str],
    ) -> Dict[str, Any]:
        """Trigger loan disbursement to the borrower's account.

        Args:
            loan_id: The approved loan identifier.
            amount: Disbursement amount.
            account_details: Dictionary with ``bank_code``, ``account_number``,
                and ``ifsc_code``.

        Returns:
            Dictionary with disbursement confirmation and transaction ID.

        Raises:
            ValueError: If amount is invalid or account details are incomplete.
        """
        logger.info("Triggering disbursement for loan=%s amount=%.2f", loan_id, amount)

        if amount <= 0:
            raise ValueError("Disbursement amount must be positive")

        required = ["bank_code", "account_number", "ifsc_code"]
        missing = [f for f in required if f not in account_details]
        if missing:
            raise ValueError(f"Missing account details: {missing}")

        payload = {
            "loan_id": loan_id,
            "amount": amount,
            "account_details": account_details,
            "disbursement_timestamp": datetime.utcnow().isoformat(),
            "transaction_ref": str(uuid.uuid4())[:16],
        }

        response = self._request("POST", f"/api/v1/loans/{loan_id}/disburse", json_body=payload)

        if response["status_code"] in (200, 201):
            result = response["body"] if isinstance(response["body"], dict) else {"status": "initiated"}
            result["loan_id"] = loan_id
            result["amount"] = amount
            result["transaction_ref"] = payload["transaction_ref"]
            logger.info("Disbursement triggered: loan=%s, ref=%s", loan_id, payload["transaction_ref"])
            return result

        return {
            "loan_id": loan_id,
            "status": "failed",
            "error": response["body"],
        }

    def create_repayment_schedule(
        self,
        loan_id: str,
        principal: float,
        rate: float,
        tenure: int,
    ) -> Dict[str, Any]:
        """Create an EMI repayment schedule.

        Computes a standard amortisation schedule using the reducing
        balance method and submits it to the banking system.

        Args:
            loan_id: The loan identifier.
            principal: Loan principal amount.
            rate: Annual interest rate (percentage).
            tenure: Loan tenure in months.

        Returns:
            Dictionary with the computed EMI schedule and API submission
            confirmation.

        Raises:
            ValueError: If inputs are invalid.
        """
        logger.info(
            "Creating repayment schedule for loan=%s (principal=%.2f, rate=%.2f%%, tenure=%d)",
            loan_id,
            principal,
            rate,
            tenure,
        )

        if principal <= 0:
            raise ValueError("Principal must be positive")
        if rate <= 0 or rate > 50:
            raise ValueError("Interest rate must be between 0 and 50 percent")
        if tenure <= 0 or tenure > 360:
            raise ValueError("Tenure must be between 1 and 360 months")

        monthly_rate = rate / (12 * 100)
        if monthly_rate > 0:
            emi = principal * monthly_rate * (1 + monthly_rate) ** tenure / (
                (1 + monthly_rate) ** tenure - 1
            )
        else:
            emi = principal / tenure

        schedule: List[Dict[str, Any]] = []
        balance = principal
        total_interest = 0.0

        for month in range(1, tenure + 1):
            interest_component = balance * monthly_rate
            principal_component = emi - interest_component
            balance -= principal_component
            total_interest += interest_component

            schedule.append({
                "month": month,
                "emi": round(emi, 2),
                "principal": round(principal_component, 2),
                "interest": round(interest_component, 2),
                "balance": round(max(balance, 0), 2),
            })

        total_payment = emi * tenure

        schedule_data = {
            "loan_id": loan_id,
            "principal": principal,
            "annual_rate": rate,
            "tenure_months": tenure,
            "emi": round(emi, 2),
            "total_payment": round(total_payment, 2),
            "total_interest": round(total_interest, 2),
            "schedule": schedule,
            "created_at": datetime.utcnow().isoformat(),
        }

        response = self._request(
            "POST",
            f"/api/v1/loans/{loan_id}/repayment-schedule",
            json_body=schedule_data,
        )

        if response["status_code"] in (200, 201):
            logger.info("Repayment schedule created: loan=%s, EMI=%.2f", loan_id, emi)
            schedule_data["api_status"] = "submitted"
        else:
            schedule_data["api_status"] = "failed"
            schedule_data["api_error"] = response["body"]

        return schedule_data

    def send_notification(
        self,
        customer_id: str,
        message: str,
        channel: str = "sms",
    ) -> Dict[str, Any]:
        """Send a notification to a customer.

        Args:
            customer_id: Target customer identifier.
            message: Notification body text.
            channel: Delivery channel (``sms``, ``email``, ``push``,
                ``whatsapp``).

        Returns:
            Dictionary with notification delivery status.

        Raises:
            ValueError: If channel is invalid.
        """
        logger.info("Sending %s notification to customer=%s", channel, customer_id)

        valid_channels = ["sms", "email", "push", "whatsapp"]
        if channel.lower() not in valid_channels:
            raise ValueError(f"Invalid channel. Must be one of: {valid_channels}")

        payload = {
            "customer_id": customer_id,
            "message": message,
            "channel": channel.lower(),
            "sent_at": datetime.utcnow().isoformat(),
            "notification_id": str(uuid.uuid4())[:12],
        }

        response = self._request("POST", f"/api/v1/notifications/send", json_body=payload)

        if response["status_code"] in (200, 201):
            result = response["body"] if isinstance(response["body"], dict) else {"status": "sent"}
            result["notification_id"] = payload["notification_id"]
            result["channel"] = channel.lower()
            logger.info("Notification sent: id=%s", payload["notification_id"])
            return result

        return {
            "notification_id": payload["notification_id"],
            "status": "failed",
            "error": response["body"],
        }

    def get_market_rates(self) -> Dict[str, Any]:
        """Fetch current market interest rates.

        Queries the banking system or external rate providers for
        prevailing lending rates across loan categories.

        Returns:
            Dictionary with market rate data.
        """
        logger.info("Fetching market rates")

        response = self._request("GET", "/api/v1/market/rates")

        if response["status_code"] == 200:
            result = response["body"] if isinstance(response["body"], dict) else {"rates": response["body"]}
            result["fetch_time"] = datetime.utcnow().isoformat()
            return result

        return {
            "status": "error",
            "error": response["body"],
            "fetch_time": datetime.utcnow().isoformat(),
        }

    def fetch_rbi_repo_rate(self) -> Dict[str, Any]:
        """Fetch the current RBI repo rate.

        Returns:
            Dictionary with repo rate value and effective date.
        """
        logger.info("Fetching RBI repo rate")

        response = self._request("GET", "/api/v1/market/rbi-repo-rate")

        if response["status_code"] == 200:
            result = response["body"] if isinstance(response["body"], dict) else {"repo_rate": response["body"]}
            result["fetch_time"] = datetime.utcnow().isoformat()
            return result

        return {
            "repo_rate": None,
            "status": "error",
            "error": response["body"],
            "fetch_time": datetime.utcnow().isoformat(),
        }

    def fetch_exchange_rate(
        self,
        from_currency: str = "USD",
        to_currency: str = "INR",
    ) -> Dict[str, Any]:
        """Fetch the current exchange rate between two currencies.

        Args:
            from_currency: Source currency code (e.g. ``USD``, ``EUR``).
            to_currency: Target currency code (e.g. ``INR``).

        Returns:
            Dictionary with exchange rate and metadata.
        """
        logger.info("Fetching exchange rate: %s -> %s", from_currency, to_currency)

        params = {"from": from_currency.upper(), "to": to_currency.upper()}
        response = self._request("GET", "/api/v1/market/exchange-rate", params=params)

        if response["status_code"] == 200:
            result = response["body"] if isinstance(response["body"], dict) else {"rate": response["body"]}
            result["from_currency"] = from_currency.upper()
            result["to_currency"] = to_currency.upper()
            result["fetch_time"] = datetime.utcnow().isoformat()
            return result

        return {
            "from_currency": from_currency.upper(),
            "to_currency": to_currency.upper(),
            "rate": None,
            "status": "error",
            "error": response["body"],
            "fetch_time": datetime.utcnow().isoformat(),
        }


# ---------------------------------------------------------------------------
# AABankingIntegration – RBI Account Aggregator Framework
# ---------------------------------------------------------------------------

class AABankingIntegration:
    """Integration with the RBI Account Aggregator (AA) framework.

    Implements the consent lifecycle, data fetch, and revocation flows
    required by the RBI AA framework for privacy-preserving financial
    data sharing.

    Supports multiple AA providers (Cookie, Finvu, OneMoney, etc.).

    Attributes:
        aa_provider: AA provider identifier.
        base_url: Base URL of the AA provider's API.
        timeout: Default request timeout in seconds.
    """

    _PROVIDER_URLS: Dict[str, str] = {
        "cookie": "https://aa.cookie.so/api/v1",
        "finvu": "https://aa.finvu.in/api/v1",
        "onemoney": "https://onemoney.in/aa/api/v1",
    }

    def __init__(
        self,
        aa_provider: str = "cookie",
        base_url: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        """Initialise the AA integration client.

        Args:
            aa_provider: Name of the AA provider.
            base_url: Override URL for the AA provider API.
            timeout: Default HTTP timeout in seconds.
        """
        self.aa_provider = aa_provider.lower()
        self.base_url = base_url or self._PROVIDER_URLS.get(
            self.aa_provider,
            f"https://aa.{self.aa_provider}.in/api/v1",
        )
        self.timeout = timeout
        self._session_token: Optional[str] = None
        self._consent_store: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "AABankingIntegration initialised (provider=%s, base_url=%s)",
            self.aa_provider,
            self.base_url,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Issue an authenticated request to the AA provider API.

        Args:
            method: HTTP verb.
            path: API path (appended to base URL).
            params: Optional query parameters.
            json_body: Optional JSON body.

        Returns:
            Parsed response dictionary.
        """
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-AA-Provider": self.aa_provider,
        }
        if self._session_token:
            headers["Authorization"] = f"Bearer {self._session_token}"

        logger.debug("AA %s %s", method, url)

        return _http_request(
            method,
            url,
            headers=headers,
            params=params,
            json_body=json_body,
            timeout=self.timeout,
        )

    def create_consent(
        self,
        customer_id: str,
        data_types: List[str],
        purpose: str,
    ) -> Dict[str, Any]:
        """Create an AA consent request.

        Initiates the consent flow for accessing the customer's financial
        data through the Account Aggregator framework.

        Args:
            customer_id: Customer identifier (mobile number or handle).
            data_types: List of data types to request (e.g.
                ``["deposit", "term_deposit", "recurring_deposit",
                "mutual_fund", "insurance", "government_scheme"]``).
            purpose: Purpose code for the data request.

        Returns:
            Dictionary with ``consent_id``, consent URL, and status.

        Raises:
            ValueError: If data_types or purpose are invalid.
        """
        logger.info(
            "Creating AA consent for customer=%s (data_types=%s)",
            customer_id,
            data_types,
        )

        valid_data_types = [
            "deposit", "term_deposit", "recurring_deposit",
            "mutual_fund", "insurance", "government_scheme",
            "equity", "bond", "debt",
        ]
        invalid = [dt for dt in data_types if dt not in valid_data_types]
        if invalid:
            raise ValueError(f"Invalid data_types: {invalid}. Valid: {valid_data_types}")

        if not purpose or len(purpose) < 3:
            raise ValueError("Purpose must be a non-empty string")

        consent_id = str(uuid.uuid4())

        payload = {
            "consent_id": consent_id,
            "customer_id": customer_id,
            "data_types": data_types,
            "purpose": purpose,
            "aa_provider": self.aa_provider,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat(),
        }

        response = self._request("POST", "/consent/create", json_body=payload)

        if response["status_code"] in (200, 201):
            result = response["body"] if isinstance(response["body"], dict) else {}
            result["consent_id"] = consent_id
            result["status"] = "pending"
            result["customer_id"] = customer_id
            result["consent_url"] = f"{self.base_url}/consent/{consent_id}/authorize"
        else:
            result = {
                "consent_id": consent_id,
                "status": "failed",
                "error": response["body"],
            }

        self._consent_store[consent_id] = {
            **payload,
            "status": result.get("status", "pending"),
        }

        logger.info("AA consent created: id=%s, status=%s", consent_id, result.get("status"))
        return result

    def check_consent_status(self, consent_id: str) -> Dict[str, Any]:
        """Check the status of a consent request.

        Args:
            consent_id: The consent identifier returned by
                ``create_consent``.

        Returns:
            Dictionary with consent status details.
        """
        logger.info("Checking AA consent status: %s", consent_id)

        response = self._request("GET", f"/consent/{consent_id}/status")

        if response["status_code"] == 200:
            result = response["body"] if isinstance(response["body"], dict) else {"status": response["body"]}
            result["consent_id"] = consent_id

            if consent_id in self._consent_store:
                self._consent_store[consent_id]["status"] = result.get("status", "unknown")

            return result

        if consent_id in self._consent_store:
            return {
                "consent_id": consent_id,
                "status": self._consent_store[consent_id].get("status", "unknown"),
                "source": "local_cache",
            }

        return {
            "consent_id": consent_id,
            "status": "unknown",
            "error": response["body"],
        }

    def fetch_account_data(self, consent_id: str) -> Dict[str, Any]:
        """Fetch account data after consent approval.

        Retrieves the customer's financial data as authorised by the
        consent. Only works for consents with status ``active``.

        Args:
            consent_id: The approved consent identifier.

        Returns:
            Dictionary with account data.

        Raises:
            RuntimeError: If consent is not in active status.
        """
        logger.info("Fetching account data via AA (consent=%s)", consent_id)

        status_resp = self.check_consent_status(consent_id)
        status = status_resp.get("status", "unknown")

        if status not in ("active", "approved"):
            raise RuntimeError(
                f"Consent {consent_id} is not active (status={status}). "
                "Cannot fetch data."
            )

        response = self._request("GET", f"/data/{consent_id}/accounts")

        if response["status_code"] == 200:
            result = response["body"] if isinstance(response["body"], dict) else {"data": response["body"]}
            result["consent_id"] = consent_id
            result["fetch_time"] = datetime.utcnow().isoformat()
            return result

        return {
            "consent_id": consent_id,
            "status": "error",
            "error": response["body"],
        }

    def fetch_transactions_from_aa(
        self,
        consent_id: str,
        date_range: Dict[str, str],
    ) -> Dict[str, Any]:
        """Fetch transactions via the AA framework.

        Args:
            consent_id: Active consent identifier.
            date_range: Dictionary with ``start_date`` and ``end_date``
                in ``YYYY-MM-DD`` format.

        Returns:
            Dictionary with transaction data.

        Raises:
            ValueError: If date_range is malformed.
        """
        logger.info(
            "Fetching AA transactions (consent=%s, range=%s)",
            consent_id,
            date_range,
        )

        if "start_date" not in date_range or "end_date" not in date_range:
            raise ValueError("date_range must contain 'start_date' and 'end_date'")

        status_resp = self.check_consent_status(consent_id)
        if status_resp.get("status") not in ("active", "approved"):
            return {
                "consent_id": consent_id,
                "status": "error",
                "error": "Consent not active",
            }

        params = {
            "from_date": date_range["start_date"],
            "to_date": date_range["end_date"],
        }

        response = self._request(
            "GET",
            f"/data/{consent_id}/transactions",
            params=params,
        )

        if response["status_code"] == 200:
            result = response["body"] if isinstance(response["body"], dict) else {"transactions": response["body"]}
            result["consent_id"] = consent_id
            result["date_range"] = date_range
            result["fetch_time"] = datetime.utcnow().isoformat()
            return result

        return {
            "consent_id": consent_id,
            "status": "error",
            "error": response["body"],
        }

    def fetch_balance_from_aa(self, consent_id: str) -> Dict[str, Any]:
        """Fetch account balance via the AA framework.

        Args:
            consent_id: Active consent identifier.

        Returns:
            Dictionary with balance information.
        """
        logger.info("Fetching AA balance (consent=%s)", consent_id)

        status_resp = self.check_consent_status(consent_id)
        if status_resp.get("status") not in ("active", "approved"):
            return {
                "consent_id": consent_id,
                "status": "error",
                "error": "Consent not active",
            }

        response = self._request("GET", f"/data/{consent_id}/balance")

        if response["status_code"] == 200:
            result = response["body"] if isinstance(response["body"], dict) else {"balance": response["body"]}
            result["consent_id"] = consent_id
            result["fetch_time"] = datetime.utcnow().isoformat()
            return result

        return {
            "consent_id": consent_id,
            "status": "error",
            "error": response["body"],
        }

    def revoke_consent(self, consent_id: str) -> Dict[str, Any]:
        """Revoke data access for a consent.

        Permanently revokes the consent and stops further data sharing.
        The customer must re-authorise if access is needed again.

        Args:
            consent_id: The consent identifier to revoke.

        Returns:
            Dictionary with revocation confirmation.
        """
        logger.info("Revoking AA consent: %s", consent_id)

        payload = {
            "consent_id": consent_id,
            "revoked_at": datetime.utcnow().isoformat(),
        }

        response = self._request("POST", f"/consent/{consent_id}/revoke", json_body=payload)

        if response["status_code"] in (200, 204):
            if consent_id in self._consent_store:
                self._consent_store[consent_id]["status"] = "revoked"

            result = response["body"] if isinstance(response["body"], dict) else {}
            result["consent_id"] = consent_id
            result["status"] = "revoked"
            result["revoked_at"] = payload["revoked_at"]
            logger.info("AA consent revoked: %s", consent_id)
            return result

        return {
            "consent_id": consent_id,
            "status": "revoke_failed",
            "error": response["body"],
        }
