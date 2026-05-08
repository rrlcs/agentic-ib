import requests
import os
from observability.logger import get_logger, log_event

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
log = get_logger(__name__)


def get_financials(symbol: str):
    url = "https://www.alphavantage.co/query"
    log_event(log, "financial_api_request_started", symbol=symbol)
    # get overview, income statement, balance sheet, cash flow, earnings
    
    overview_params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": API_KEY
    }

    income_statement_params = {
        "function": "INCOME_STATEMENT",
        "symbol": symbol,
        "apikey": API_KEY
    }

    balance_sheet_params = {
        "function": "BALANCE_SHEET",
        "symbol": symbol,
        "apikey": API_KEY
    }

    cash_flow_params = {
        "function": "CASH_FLOW",
        "symbol": symbol,
        "apikey": API_KEY
    }

    earnings_params = {
        "function": "EARNINGS",
        "symbol": symbol,
        "apikey": API_KEY
    }

    try:
        overview_response = requests.get(url, params=overview_params, timeout=20)
        income_statement_response = requests.get(url, params=income_statement_params, timeout=20)
        balance_sheet_response = requests.get(url, params=balance_sheet_params, timeout=20)
        cash_flow_response = requests.get(url, params=cash_flow_params, timeout=20)
        earnings_response = requests.get(url, params=earnings_params, timeout=20)
        overview_response.raise_for_status()
        income_statement_response.raise_for_status()
        balance_sheet_response.raise_for_status()
        cash_flow_response.raise_for_status()
        earnings_response.raise_for_status()
        overview_data = overview_response.json()
        income_statement_data = income_statement_response.json()
        balance_sheet_data = balance_sheet_response.json()
        cash_flow_data = cash_flow_response.json()
        earnings_data = earnings_response.json()
        data = {
            "overview": overview_data,
            "income_statement": income_statement_data,
            "balance_sheet": balance_sheet_data,
            "cash_flow": cash_flow_data,
            "earnings": earnings_data
        }
    except requests.RequestException as exc:
        log_event(log, "financial_api_request_failed", symbol=symbol, error=str(exc))
        return {
            "name": symbol,
            "market_cap": None,
            "pe_ratio": None,
            "revenue_ttm": None,
            "profit_margin": None,
            "error": f"financial_api_unavailable: {exc}",
        }

    log_event(log, "financial_api_request_completed", symbol=symbol, financials=data)
    return data