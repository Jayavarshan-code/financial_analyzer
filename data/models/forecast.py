"""
data/models/forecast.py
------------------------
Pydantic v2 data models for Phase 2 analytics modules.

Model hierarchy:
  ForecastMethod          - enum of revenue forecasting methods
  EarningsForecastMethod  - enum of earnings forecasting methods
  ForecastPoint           - one period (actual or projected) with optional CI
  RevenueForecastResult   - one method's full forecast: actuals + projections + fit stats
  RevenueForecastSuite    - all methods run against one ticker + recommended result
  EarningsForecastPoint   - one projected period: revenue → EBITDA → net income → EPS
  EarningsForecastResult  - one method's full earnings forecast
  EarningsForecastSuite   - all earnings methods + recommended result

Design notes:
  - All monetary values in millions USD (consistent with FullFinancialHistory).
  - EPS values are per-share (not scaled).
  - Confidence intervals use the 95% prediction interval by convention.
    lower_95 / upper_95 = None for historical actuals and when CIs cannot be computed.
  - RevenueForecastSuite.dcf_growth_rates feeds directly into
    DCFAssumptions.revenue_growth_rates for seamless DCF integration.
  - RevenueForecastResult.mape is None when any historical revenue is zero or negative
    (avoids division-by-zero in percentage errors).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ForecastMethod(str, Enum):
    CAGR = "cagr"
    LINEAR_TREND = "linear_trend"
    EXPONENTIAL_TREND = "exponential_trend"
    HOLT_WINTERS = "holt_winters"
    ENSEMBLE = "ensemble"


class EarningsForecastMethod(str, Enum):
    MARGIN_BASED = "margin_based"
    EPS_TREND = "eps_trend"


# ---------------------------------------------------------------------------
# Revenue forecast models
# ---------------------------------------------------------------------------

class ForecastPoint(BaseModel):
    """
    One period of a revenue forecast: actual or projected.

    value:      revenue in millions USD
    is_actual:  True for historical data points
    lower_95 / upper_95: 95% prediction interval (None for actuals or when unavailable)
    """

    year: int
    value: float
    is_actual: bool = False
    lower_95: Optional[float] = None
    upper_95: Optional[float] = None


class RevenueForecastResult(BaseModel):
    """
    Complete output of one revenue forecasting method.

    points:         historical actuals + projected periods (oldest first)
    projected_only: just the forecast periods (length = n_forecast_years)
    cagr_projected: geometric CAGR of the projected revenue over the forecast horizon
    r_squared:      in-sample R² (None for CAGR / Holt-Winters)
    mae:            mean absolute error on training data (millions USD)
    mape:           mean absolute percentage error on training data (decimal, e.g. 0.05)
    metadata:       method-specific info (e.g. slope, alpha parameter)
    """

    method: ForecastMethod
    method_label: str
    points: list[ForecastPoint]
    projected_only: list[ForecastPoint]
    cagr_projected: float
    r_squared: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None
    metadata: dict = Field(default_factory=dict)


class RevenueForecastSuite(BaseModel):
    """
    All revenue forecasting methods run against one company.

    results:         one RevenueForecastResult per method attempted
    recommended:     result with lowest MAPE (or first available if MAPE unavailable)
    dcf_growth_rates: year-by-year growth rates from recommended, ready for DCFAssumptions
    """

    ticker: str
    base_year: str
    base_revenue: float
    n_forecast_years: int
    results: list[RevenueForecastResult]
    recommended: RevenueForecastResult
    dcf_growth_rates: list[float]


# ---------------------------------------------------------------------------
# Earnings forecast models
# ---------------------------------------------------------------------------

class EarningsForecastPoint(BaseModel):
    """
    One projected period: revenue → EBITDA → net income → EPS.

    Revenue and CI bounds come from the underlying revenue forecast.
    Margins are applied deterministically to revenue; CI on EPS inherits from revenue CI.
    """

    year: int
    is_actual: bool = False

    # Revenue (from revenue forecast)
    revenue: float
    revenue_lower_95: Optional[float] = None
    revenue_upper_95: Optional[float] = None

    # P&L build-down
    ebitda_margin: float
    ebitda: float
    net_margin: float
    net_income: float

    # Per-share
    shares_diluted: float               # millions
    eps: float
    eps_lower_95: Optional[float] = None
    eps_upper_95: Optional[float] = None


class EarningsForecastResult(BaseModel):
    """One earnings forecasting method's complete output."""

    method: EarningsForecastMethod
    method_label: str
    points: list[EarningsForecastPoint]
    projected_only: list[EarningsForecastPoint]
    cagr_eps: float
    cagr_net_income: float
    avg_ebitda_margin: float
    avg_net_margin: float


class EarningsForecastSuite(BaseModel):
    """
    All earnings forecasting methods run against one company.

    revenue_forecast: which revenue forecast was used as the base
    results:          one EarningsForecastResult per method attempted
    recommended:      the primary result (margin-based preferred when available)
    """

    ticker: str
    base_year: str
    revenue_forecast: RevenueForecastResult
    results: list[EarningsForecastResult]
    recommended: EarningsForecastResult
