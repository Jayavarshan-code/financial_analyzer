"""
data/fetchers/market_rates.py
------------------------------
Fetches live market-wide rates used in WACC calculation.

Risk-Free Rate:
  Fetched live from ^TNX (CBOE 10-Year Treasury Note Yield Index) via yfinance.
  ^TNX reports the yield as a percentage (e.g. 4.50 = 4.50%), so we divide by 100.
  Falls back to DEFAULT_RFR if the fetch fails (network outage, rate-limit, etc.).

Equity Risk Premium:
  NOT auto-fetched. Damodaran's implied ERP requires scraping his site or a
  premium API.  Update DEFAULT_ERP below when Damodaran publishes a new monthly
  estimate. Reference:
    http://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/implprem.html
  As of mid-2025 the implied ERP sits around 4.5%-5.5%; the default below is
  intentionally slightly conservative at 5.5%.

Usage:
    from data.fetchers.market_rates import fetch_risk_free_rate, DEFAULT_ERP

    rfr = fetch_risk_free_rate()          # live 10-yr Treasury
    rfr = fetch_risk_free_rate(cache=True) # re-use if already fetched this session
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable defaults
#
# DEFAULT_ERP: update this when Damodaran publishes a new monthly estimate.
#   The implied ERP for January 2025 (from Damodaran's site) was ~4.6%.
#   We use 5.5% as a slightly conservative buffer.
#
# DEFAULT_RFR: fallback for when ^TNX fetch fails.
#   Set this to whatever the 10-yr yield was on the last date you ran the model.
# ---------------------------------------------------------------------------
DEFAULT_ERP: float = 0.055   # update from Damodaran's site periodically
DEFAULT_RFR: float = 0.045   # fallback only — the live fetch is always preferred

# Module-level cache so a single CLI run doesn't fetch ^TNX multiple times
_rfr_cache: Optional[float] = None
_rfr_cache_ts: float = 0.0
_RFR_CACHE_TTL: float = 3600.0   # 1 hour — refresh if stale


def fetch_risk_free_rate(
    fallback: float = DEFAULT_RFR,
    cache: bool = True,
) -> float:
    """
    Fetch the current 10-year US Treasury yield as a decimal.

    Uses ^TNX (CBOE 10-Year Treasury Note Yield Index). The index price IS the
    yield in percentage points (e.g. 4.35 → 4.35%), so dividing by 100 gives
    the decimal rate.

    Args:
        fallback: Value to return if the fetch fails (default: DEFAULT_RFR).
        cache:    If True, reuse a session-level cached value to avoid repeated
                  network calls during scenario analysis / Monte Carlo runs.

    Returns:
        Current 10-yr yield as a decimal (e.g. 0.043 for 4.3%).
    """
    global _rfr_cache, _rfr_cache_ts

    now = time.monotonic()
    if cache and _rfr_cache is not None and (now - _rfr_cache_ts) < _RFR_CACHE_TTL:
        logger.debug("Using cached RFR: %.2f%%", _rfr_cache * 100)
        return _rfr_cache

    rfr = _fetch_tnx(fallback)

    if cache:
        _rfr_cache = rfr
        _rfr_cache_ts = now

    return rfr


def _fetch_tnx(fallback: float) -> float:
    """Inner fetch — separated so tests can mock it without patching the cache."""
    try:
        import yfinance as yf
        tnx = yf.Ticker("^TNX")
        info = tnx.info or {}
        price = (
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("previousClose")
        )
        if price is not None and float(price) > 0:
            rfr = float(price) / 100.0
            logger.info("Live 10-yr Treasury yield (^TNX): %.2f%%", rfr * 100)
            return rfr
        logger.warning("^TNX returned no price. Using fallback RFR %.2f%%", fallback * 100)
    except Exception as exc:
        logger.warning(
            "Could not fetch ^TNX: %s. Using fallback RFR %.2f%%", exc, fallback * 100
        )
    return fallback


def clear_rfr_cache() -> None:
    """Reset the session cache. Useful in tests."""
    global _rfr_cache, _rfr_cache_ts
    _rfr_cache = None
    _rfr_cache_ts = 0.0
