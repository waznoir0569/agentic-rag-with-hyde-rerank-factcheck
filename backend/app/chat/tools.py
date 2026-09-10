import logging
from typing import Dict

import requests
from loguru import logger

from app.config import settings
from app.db.pgvector_utils import vector_store

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

# Tavily picks up the key from the TAVILY_API_KEY env var by default.
# If you need to pass it explicitly, set the env var before constructing
# the wrapper rather than passing an unsupported kwarg here.
import os
os.environ.setdefault("TAVILY_API_KEY", settings.tavily_api_key)

tavily = TavilySearchResults(
    max_results=3,
    include_answer=False,
    include_raw_content=False,
    include_images=False,
)


@tool
async def retrieve_user_documents(query: str, config: RunnableConfig) -> str:
    """
    Use this tool to answer questions about the user's uploaded documents.
    It will automatically retrieve documents relevant to the current user and thread.
    """
    user_id = config["configurable"].get("user_id")  # type: ignore
    thread_id = config["configurable"].get("thread_id")  # type: ignore

    logger.info(f"Retrieving documents for user_id: {user_id} and thread_id: {thread_id}")

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3, "filter": {"thread_id": thread_id, "user_id": user_id}}
    )
    result_docs = await retriever.ainvoke(query)

    if not result_docs:
        return "No relevant documents"

    return "\n\n".join([doc.page_content for doc in result_docs])


def behavioral_scoring(
    debtrules3,
    debtnorms8,
    debtrules16,
    debtpers5,
    debtnorms9,
    debtpers9,
    debtpers11,
    debtrules9,
):
    """
    Calculate gamma_short = Debt Aversion, alpha_short = Risk Aversion,
    lambda_short = Loss Aversion, and delta_short = Time Preference.

    Parameters:
        debtrules3, debtnorms8, debtrules16, debtpers5,
        debtnorms9, debtpers9, debtpers11, debtrules9:
            Numeric input values on scale (1 to 5).

    Returns:
        list[float]: [gamma_short, alpha_short, lambda_short, delta_short]
    """
    gamma_short = (
        1.0694
        + 0.0045 * debtrules3
        - 0.0067 * debtnorms8
    )

    alpha_short = (
        0.501796
        + 0.06459 * debtrules16
        - 0.034671 * debtpers5
        - 0.044603 * debtnorms9
    )

    lambda_short = (
        1.177504
        - 0.025919 * debtrules16
        + 0.028538 * debtpers9
    )

    delta_short = (
        0.050555
        - 0.003716 * debtpers9
        + 0.001902 * debtpers11
        - 0.000938 * debtrules9
    )

    return [gamma_short, alpha_short, lambda_short, delta_short]

from typing import Dict


def _score_financial_wellbeing(
    income: float,
    expenses: float,
    savings: float,
    debt: float,
) -> Dict:
    """
    Very simple Financial Well-Being Score (0-100).

    The score is based on:
    1. How much income remains after expenses
    2. How much savings the user has
    3. How much debt the user has relative to income
    """

    if income <= 0:
        return {
            "financial_wellbeing_score": 0,
            "category": "High Risk",
            "breakdown": {},
        }

    # 1. Money left after expenses
    leftover = income - expenses

    if leftover > 0:
        savings_rate = leftover / income
    else:
        savings_rate = 0

    savings_rate = max(0, min(savings_rate, 1))

    # Savings contribution: 0-60 points
    savings_rate_score = savings_rate * 60

    # 2. Emergency savings contribution: 0-30 points
    if savings >= income * 3:
        savings_score = 30
    elif savings >= income:
        savings_score = 20
    elif savings > 0:
        savings_score = 10
    else:
        savings_score = 0

    # 3. Debt contribution: 0-20 points deducted
    debt_ratio = debt / income
    debt_penalty = min(debt_ratio * 20, 20)

    # Final score
    total_score = savings_rate_score + savings_score - debt_penalty

    total_score = round(max(0, min(total_score, 100)), 1)

    # Category
    if total_score >= 80:
        category = "Healthy"
    elif total_score >= 60:
        category = "Moderate"
    elif total_score >= 40:
        category = "Stressed"
    else:
        category = "High Risk"

    return {
        "financial_wellbeing_score": total_score,
        "category": category,
        "breakdown": {
            "savings_rate_score": round(savings_rate_score, 1),
            "savings_score": savings_score,
            "debt_penalty": round(debt_penalty, 1),
        },
        "inputs": {
            "income": income,
            "expenses": expenses,
            "savings": savings,
            "debt": debt,
        },
    }


@tool
def calculate_financial_wellbeing(
    monthly_income: float,
    monthly_expenses: float,
    savings_balance: float,
    total_debt_amount: float = 0.0,
) -> Dict:
    """
    Calculate a simple Financial Well-Being Score (0-100).

    Args:
        monthly_income: Take-home income per month.
        monthly_expenses: Regular monthly spending.
        savings_balance: Current liquid savings.
        total_debt_amount: Total outstanding debt.

    Returns:
        Financial wellbeing score, category, and breakdown.
    """

    return _score_financial_wellbeing(
        income=monthly_income,
        expenses=monthly_expenses,
        savings=savings_balance,
        debt=total_debt_amount,
    )


std_logger = logging.getLogger(__name__)


class MacroEconomicFetcher:
    """
    Fetches macroeconomic data relevant for debt decisions for a given country.
    Uses the World Bank API (no API key required).
    """

    INDICATORS = {
        "inflation_rate": "FP.CPI.TOTL.ZG",
        "unemployment_rate": "SL.UEM.TOTL.ZS",
        "real_interest_rate": "FR.INR.RINR",
        "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    }

    def __init__(self):
        self.base_url = "https://api.worldbank.org/v2/country"

    def _fetch_indicator(self, country_code: str, indicator: str) -> float | None:
        """Fetch latest non-null value for an indicator. Returns None on
        failure or missing data - callers must NOT substitute 0.0, since
        that would present a failed fetch as a real (and misleading)
        economic figure."""
        url = f"{self.base_url}/{country_code}/indicator/{indicator}"
        params = {"format": "json", "per_page": 100}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if len(data) > 1:
                for entry in data[1]:
                    value = entry.get("value")
                    if value is not None:
                        return float(value)

            logger.warning(f"World Bank API returned no data for {indicator} ({country_code})")
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch {indicator} for {country_code}: {e}")
            return None

    def get_macro_context(self, country_code: str = 'gr') -> Dict[str, float | None]:
        """
        Gathers macroeconomic factors for the specified country. Any
        indicator that couldn't be fetched comes back as None rather than
        0.0 - the system prompt is instructed to surface that as
        "data unavailable" instead of a real value.
        """
        macro_values = {}
        for friendly_name, indicator_id in self.INDICATORS.items():
            value = self._fetch_indicator(country_code, indicator_id)
            macro_values[friendly_name] = round(value, 2) if value is not None else None
        return macro_values


@tool
def calculate_amortization(
    principal: float,
    annual_interest_rate: float,
    term_months: int,
) -> Dict:
    """
    Calculate a simple fixed-rate loan amortization summary.

    Args:
        principal: Loan amount (e.g. 20000).
        annual_interest_rate: Annual interest rate as a percent (e.g. 6.5 for 6.5%).
        term_months: Loan term in months (e.g. 60).

    Returns:
        Dict with monthly_payment, total_payment, total_interest, and a
        month-by-month schedule (month, payment, principal, interest, balance).
    """
    monthly_rate = (annual_interest_rate / 100) / 12

    if monthly_rate == 0:
        monthly_payment = principal / term_months
    else:
        monthly_payment = (
            principal * monthly_rate * (1 + monthly_rate) ** term_months
        ) / ((1 + monthly_rate) ** term_months - 1)

    balance = principal
    schedule = []

    for month in range(1, term_months + 1):
        interest_payment = balance * monthly_rate
        principal_payment = monthly_payment - interest_payment
        balance -= principal_payment
        balance = max(balance, 0.0)  # avoid tiny negative rounding

        schedule.append({
            "month": month,
            "payment": round(monthly_payment, 2),
            "principal": round(principal_payment, 2),
            "interest": round(interest_payment, 2),
            "balance": round(balance, 2),
        })

    total_payment = monthly_payment * term_months
    total_interest = total_payment - principal

    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2),
        "schedule": schedule,
    }


macro_economic_fetcher = MacroEconomicFetcher()


@tool
def get_macro_context(country_code: str = "gr") -> Dict[str, float]:
    """
    Fetch macroeconomic indicators (inflation, unemployment, real interest
    rate, GDP growth) for a given country from the World Bank API.
    Defaults to Greece ('gr') if no country is specified or extracted.

    Args:
        country_code: 2- or 3-letter ISO country code (e.g. 'us', 'gb', 'gr').
    """
    return macro_economic_fetcher.get_macro_context(country_code or "gr")


tools = [
    retrieve_user_documents,
    tavily,
    calculate_financial_wellbeing,
    calculate_amortization,
    get_macro_context,
]