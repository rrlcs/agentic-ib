import requests
import os
from observability.logger import get_logger, log_event

API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
log = get_logger(__name__)


def get_financials(symbol: str):
    url = "https://www.alphavantage.co/query"
    log_event(log, "financial_api_request_started", symbol=symbol)

    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
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

    result = {
        "name": data.get("Name"),
        "market_cap": data.get("MarketCapitalization"),
        "pe_ratio": data.get("PERatio"),
        "revenue_ttm": data.get("RevenueTTM"),
        "profit_margin": data.get("ProfitMargin")
    }
    log_event(log, "financial_api_request_completed", symbol=symbol, has_name=bool(result.get("name")))
    return result