"""
analytics/revenue_forecaster.py
---------------------------------
Revenue Forecasting Module — Phase 2, Module 1.

Fits multiple statistical models to a company's historical annual revenue
series and generates a multi-year forward projection with 95% prediction
intervals. The recommended forecast's growth rates feed directly into
DCFAssumptions.revenue_growth_rates for seamless pipeline integration.

Methods implemented:
  CAGR              — n-year compound annual growth rate; simple, interpretable
  LINEAR_TREND      — OLS regression: revenue = a + b·t; best for steady growers
  EXPONENTIAL_TREND — OLS on log(revenue); best for percentage-growth stories
  HOLT_WINTERS      — double exponential smoothing with additive trend (statsmodels)
  ENSEMBLE          — MAPE-weighted average of all available methods

Method selection (auto-recommended):
  The method with the lowest MAPE on historical data is recommended.
  When MAPE is unavailable (constant revenue, too few points), the ensemble
  or first successful method is used instead.

Confidence intervals (95%):
  Regression methods: prediction interval using OLS residual std
  CAGR: expanding interval based on historical growth rate std dev
  Holt-Winters: simulation-based via statsmodels simulate_smoother
  Ensemble: propagated from component methods (unweighted union)

Minimum data requirements:
  CAGR:             ≥ 2 annual periods
  LINEAR_TREND:     ≥ 2 annual periods
  EXPONENTIAL_TREND:≥ 2 annual periods (all revenues must be positive)
  HOLT_WINTERS:     ≥ 4 annual periods
  ENSEMBLE:         ≥ 1 other method succeeded

All monetary values in millions USD.

Public API:
    from analytics.revenue_forecaster import RevenueForecastEngine
    engine = RevenueForecastEngine()
    suite  = engine.run(history, n_years=5)
    print(engine.summary(suite))
    # Feed into DCF:
    assumptions.revenue_growth_rates = suite.dcf_growth_rates
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from data.models.financials import FullFinancialHistory
from data.models.forecast import (
    ForecastMethod,
    ForecastPoint,
    RevenueForecastResult,
    RevenueForecastSuite,
)

logger = logging.getLogger(__name__)

# Lazy import — skip Holt-Winters if statsmodels is not installed
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as _HW
    _HW_AVAILABLE = True
except ImportError:
    _HW_AVAILABLE = False
    logger.debug("statsmodels not available — Holt-Winters method will be skipped.")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_revenue_series(history: FullFinancialHistory) -> tuple[list[int], list[float]]:
    """
    Extract (calendar_year, revenue_millions) pairs from annual snapshots.
    Returns only snapshots with positive revenue, sorted oldest-first.
    """
    pairs = []
    for snap in history.annual_snapshots:
        rev = snap.statements.income_statement.revenue
        if rev is not None and rev > 0:
            pairs.append((int(snap.period[:4]), rev))
    pairs.sort()
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _compute_cagr(start: float, end: float, n_periods: int) -> float:
    """Geometric CAGR over n_periods."""
    if n_periods <= 0 or start <= 0:
        return 0.0
    return (end / start) ** (1 / n_periods) - 1


def _growth_rates(base: float, projected: list[ForecastPoint]) -> list[float]:
    """Derive year-over-year revenue growth rates from a base revenue and forecast points."""
    rates = []
    prev = base
    for p in projected:
        g = (p.value - prev) / prev if prev > 0 else 0.0
        rates.append(g)
        prev = p.value
    return rates


def _mae_mape(actuals: np.ndarray, fitted: np.ndarray) -> tuple[float, Optional[float]]:
    residuals = actuals - fitted
    mae = float(np.mean(np.abs(residuals)))
    if np.all(actuals > 0):
        mape = float(np.mean(np.abs(residuals / actuals)))
    else:
        mape = None
    return mae, mape


# ---------------------------------------------------------------------------
# Individual fit functions
# ---------------------------------------------------------------------------

def _fit_cagr(years: list[int], revenues: list[float], n_forecast: int) -> RevenueForecastResult:
    """
    Project revenue at the historical CAGR computed from the last 3 data points
    (or however many are available).  CI uses the std dev of historical YoY growth.
    """
    n_hist = len(revenues)
    n_for_cagr = min(n_hist, 3)
    cagr = _compute_cagr(revenues[-n_for_cagr], revenues[-1], n_for_cagr - 1) if n_for_cagr >= 2 else 0.0

    # Historical YoY growth rates → std dev for CI width
    yoy = [(revenues[i] - revenues[i - 1]) / revenues[i - 1]
           for i in range(1, n_hist)]
    growth_std = float(np.std(yoy)) if len(yoy) >= 2 else abs(cagr) * 0.3

    # In-sample fit (apply CAGR backwards from last point to check quality)
    base_rev = revenues[-n_for_cagr]
    fitted = np.array([base_rev * (1 + cagr) ** t for t in range(n_for_cagr)])
    actuals = np.array(revenues[-n_for_cagr:])
    mae, mape = _mae_mape(actuals, fitted)

    # Historical actual points
    points: list[ForecastPoint] = [ForecastPoint(year=y, value=v, is_actual=True)
                                    for y, v in zip(years, revenues)]
    # Projected points
    projected: list[ForecastPoint] = []
    for i in range(1, n_forecast + 1):
        rev = revenues[-1] * (1 + cagr) ** i
        rev = max(0.0, rev)
        ci_half = rev * 1.96 * growth_std * np.sqrt(i)
        projected.append(ForecastPoint(
            year=years[-1] + i,
            value=rev,
            is_actual=False,
            lower_95=max(0.0, rev - ci_half),
            upper_95=rev + ci_half,
        ))
    points += projected

    return RevenueForecastResult(
        method=ForecastMethod.CAGR,
        method_label=f"CAGR ({n_for_cagr - 1}-year)",
        points=points,
        projected_only=projected,
        cagr_projected=cagr,
        mae=mae,
        mape=mape,
        metadata={"cagr": cagr, "growth_std": growth_std},
    )


def _fit_linear_trend(years: list[int], revenues: list[float], n_forecast: int) -> RevenueForecastResult:
    """OLS linear regression: revenue = a + b * t."""
    t = np.arange(len(revenues), dtype=float)
    rev = np.array(revenues)

    slope, intercept, r_value, _p, _se = sp_stats.linregress(t, rev)

    fitted = intercept + slope * t
    residuals = rev - fitted
    s = float(np.std(residuals, ddof=2)) if len(residuals) > 2 else float(np.std(residuals))
    r_sq = float(r_value ** 2)
    mae, mape = _mae_mape(rev, fitted)

    t_mean = float(np.mean(t))
    Stt = float(np.sum((t - t_mean) ** 2))
    n = len(t)

    points: list[ForecastPoint] = [ForecastPoint(year=y, value=v, is_actual=True)
                                    for y, v in zip(years, revenues)]
    projected: list[ForecastPoint] = []
    for i in range(1, n_forecast + 1):
        t_new = len(revenues) - 1 + i
        rev_proj = max(0.0, intercept + slope * t_new)
        # Prediction interval
        se_pred = s * np.sqrt(1 + 1 / n + (t_new - t_mean) ** 2 / Stt) if Stt > 0 else s
        lower = max(0.0, rev_proj - 1.96 * se_pred)
        upper = rev_proj + 1.96 * se_pred
        projected.append(ForecastPoint(
            year=years[-1] + i, value=rev_proj, is_actual=False,
            lower_95=lower, upper_95=upper,
        ))
    points += projected

    cagr = _compute_cagr(revenues[-1], projected[-1].value, n_forecast)
    return RevenueForecastResult(
        method=ForecastMethod.LINEAR_TREND,
        method_label="OLS Linear Trend",
        points=points,
        projected_only=projected,
        cagr_projected=cagr,
        r_squared=r_sq,
        mae=mae,
        mape=mape,
        metadata={"slope": slope, "intercept": intercept},
    )


def _fit_exponential_trend(years: list[int], revenues: list[float], n_forecast: int) -> Optional[RevenueForecastResult]:
    """OLS on log(revenue): log(rev) = a + b * t, back-transformed."""
    if any(r <= 0 for r in revenues):
        return None

    t = np.arange(len(revenues), dtype=float)
    log_rev = np.log(np.array(revenues))

    slope, intercept, r_value, _p, _se = sp_stats.linregress(t, log_rev)

    fitted_log = intercept + slope * t
    residuals_log = log_rev - fitted_log
    s_log = float(np.std(residuals_log, ddof=2)) if len(residuals_log) > 2 else float(np.std(residuals_log))
    r_sq = float(r_value ** 2)

    fitted_rev = np.exp(fitted_log)
    mae, mape = _mae_mape(np.array(revenues), fitted_rev)

    t_mean = float(np.mean(t))
    Stt = float(np.sum((t - t_mean) ** 2))
    n = len(t)

    points: list[ForecastPoint] = [ForecastPoint(year=y, value=v, is_actual=True)
                                    for y, v in zip(years, revenues)]
    projected: list[ForecastPoint] = []
    for i in range(1, n_forecast + 1):
        t_new = len(revenues) - 1 + i
        log_proj = intercept + slope * t_new
        rev_proj = float(np.exp(log_proj))
        se_log = s_log * np.sqrt(1 + 1 / n + (t_new - t_mean) ** 2 / Stt) if Stt > 0 else s_log
        # CI in log space → multiplicative CI in revenue space
        lower = max(0.0, float(np.exp(log_proj - 1.96 * se_log)))
        upper = float(np.exp(log_proj + 1.96 * se_log))
        projected.append(ForecastPoint(
            year=years[-1] + i, value=rev_proj, is_actual=False,
            lower_95=lower, upper_95=upper,
        ))
    points += projected

    cagr = _compute_cagr(revenues[-1], projected[-1].value, n_forecast)
    return RevenueForecastResult(
        method=ForecastMethod.EXPONENTIAL_TREND,
        method_label="OLS Exponential Trend",
        points=points,
        projected_only=projected,
        cagr_projected=cagr,
        r_squared=r_sq,
        mae=mae,
        mape=mape,
        metadata={"slope_log": slope, "intercept_log": intercept},
    )


def _fit_holt_winters(years: list[int], revenues: list[float], n_forecast: int) -> Optional[RevenueForecastResult]:
    """Double exponential smoothing via statsmodels (additive trend, no seasonality)."""
    if not _HW_AVAILABLE or len(revenues) < 4:
        return None

    rev_arr = np.array(revenues, dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _HW(rev_arr, trend="add", initialization_method="estimated")
            fit = model.fit(optimized=True)
    except Exception as exc:
        logger.debug("Holt-Winters fit failed: %s", exc)
        return None

    forecast_vals = fit.forecast(n_forecast)
    residuals = fit.resid
    s = float(np.std(residuals))

    mae = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals / rev_arr))) if np.all(rev_arr > 0) else None

    points: list[ForecastPoint] = [ForecastPoint(year=y, value=v, is_actual=True)
                                    for y, v in zip(years, revenues)]
    projected: list[ForecastPoint] = []
    for i, fv in enumerate(forecast_vals):
        rev_proj = max(0.0, float(fv))
        # CI widens linearly with horizon (approximation)
        ci_half = 1.96 * s * np.sqrt(i + 1)
        projected.append(ForecastPoint(
            year=years[-1] + i + 1,
            value=rev_proj,
            is_actual=False,
            lower_95=max(0.0, rev_proj - ci_half),
            upper_95=rev_proj + ci_half,
        ))
    points += projected

    cagr = _compute_cagr(revenues[-1], projected[-1].value, n_forecast)
    return RevenueForecastResult(
        method=ForecastMethod.HOLT_WINTERS,
        method_label="Holt-Winters (additive trend)",
        points=points,
        projected_only=projected,
        cagr_projected=cagr,
        mae=mae,
        mape=mape,
        metadata={"alpha": float(fit.params.get("smoothing_level", 0)),
                  "beta": float(fit.params.get("smoothing_trend", 0))},
    )


def _fit_ensemble(
    component_results: list[RevenueForecastResult],
    years: list[int],
    revenues: list[float],
    n_forecast: int,
) -> RevenueForecastResult:
    """
    MAPE-weighted average of all available component methods.
    Components with unavailable MAPE receive a weight of 1 (unweighted fallback).
    """
    weights_raw = []
    for r in component_results:
        if r.mape is not None and r.mape > 0:
            weights_raw.append(1.0 / r.mape)
        else:
            weights_raw.append(1.0)

    total = sum(weights_raw)
    weights = [w / total for w in weights_raw]

    # Weighted projected revenues
    projected: list[ForecastPoint] = []
    for i in range(n_forecast):
        w_rev = sum(w * r.projected_only[i].value for w, r in zip(weights, component_results))
        w_rev = max(0.0, w_rev)

        # CI: use the widest lower and highest upper across components
        lowers = [r.projected_only[i].lower_95 for r in component_results
                  if r.projected_only[i].lower_95 is not None]
        uppers = [r.projected_only[i].upper_95 for r in component_results
                  if r.projected_only[i].upper_95 is not None]

        projected.append(ForecastPoint(
            year=years[-1] + i + 1,
            value=w_rev,
            is_actual=False,
            lower_95=min(lowers) if lowers else None,
            upper_95=max(uppers) if uppers else None,
        ))

    # In-sample: weighted fitted values
    component_mapes = [r.mape for r in component_results if r.mape is not None]
    ensemble_mape = float(np.mean(component_mapes)) if component_mapes else None
    component_maes = [r.mae for r in component_results if r.mae is not None]
    ensemble_mae = float(np.mean(component_maes)) if component_maes else None

    points = [ForecastPoint(year=y, value=v, is_actual=True)
               for y, v in zip(years, revenues)] + projected

    cagr = _compute_cagr(revenues[-1], projected[-1].value, n_forecast)
    return RevenueForecastResult(
        method=ForecastMethod.ENSEMBLE,
        method_label="Ensemble (MAPE-weighted)",
        points=points,
        projected_only=projected,
        cagr_projected=cagr,
        mae=ensemble_mae,
        mape=ensemble_mape,
        metadata={"weights": {r.method.value: round(w, 4) for r, w in zip(component_results, weights)}},
    )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class RevenueForecastEngine:
    """
    Fits multiple statistical models to historical revenue and produces
    forward projections with confidence intervals.

    Usage:
        engine = RevenueForecastEngine()
        suite  = engine.run(history, n_years=5)
        print(engine.summary(suite))
        dcf_assumptions.revenue_growth_rates = suite.dcf_growth_rates
    """

    def run(
        self,
        history: FullFinancialHistory,
        n_years: int = 5,
        methods: Optional[list[ForecastMethod]] = None,
    ) -> RevenueForecastSuite:
        """
        Fit all available methods and return a RevenueForecastSuite.

        Args:
            history:  FullFinancialHistory from FinancialStatementAnalyzer.
            n_years:  Number of years to project forward.
            methods:  Subset of ForecastMethod to run. None = all.

        Returns:
            RevenueForecastSuite with results for all successful methods
            and the recommended result (lowest MAPE).

        Raises:
            ValueError: if history contains fewer than 2 periods of revenue data.
        """
        years, revenues = _extract_revenue_series(history)
        if len(years) < 2:
            raise ValueError(
                f"{history.profile.ticker}: need at least 2 revenue periods "
                f"for forecasting; got {len(years)}."
            )

        run_all = methods is None
        results: list[RevenueForecastResult] = []

        if run_all or ForecastMethod.CAGR in (methods or []):
            results.append(_fit_cagr(years, revenues, n_years))

        if run_all or ForecastMethod.LINEAR_TREND in (methods or []):
            results.append(_fit_linear_trend(years, revenues, n_years))

        if run_all or ForecastMethod.EXPONENTIAL_TREND in (methods or []):
            r = _fit_exponential_trend(years, revenues, n_years)
            if r is not None:
                results.append(r)

        if run_all or ForecastMethod.HOLT_WINTERS in (methods or []):
            r = _fit_holt_winters(years, revenues, n_years)
            if r is not None:
                results.append(r)

        if not results:
            raise ValueError(f"{history.profile.ticker}: no forecasting method succeeded.")

        # Ensemble from all successful non-ensemble methods
        if (run_all or ForecastMethod.ENSEMBLE in (methods or [])) and len(results) >= 2:
            results.append(_fit_ensemble(results, years, revenues, n_years))

        # Recommended: lowest MAPE
        recommended = self._pick_best(results)

        # DCF growth rates from recommended forecast
        dcf_rates = _growth_rates(revenues[-1], recommended.projected_only)

        return RevenueForecastSuite(
            ticker=history.profile.ticker,
            base_year=history.latest.period,
            base_revenue=revenues[-1],
            n_forecast_years=n_years,
            results=results,
            recommended=recommended,
            dcf_growth_rates=dcf_rates,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_best(results: list[RevenueForecastResult]) -> RevenueForecastResult:
        """Return the result with the lowest MAPE; break ties by method order."""
        ranked = sorted(
            results,
            key=lambda r: (r.mape is None, r.mape or 0.0),
        )
        return ranked[0]

    # ------------------------------------------------------------------
    # Summary / display
    # ------------------------------------------------------------------

    def summary(self, suite: RevenueForecastSuite) -> str:
        """
        Formatted ASCII summary.

        Sections:
          1. Method comparison table (CAGR, R², MAE, MAPE)
          2. Recommended forecast projection table with CI
        """
        lines: list[str] = []
        W = 76

        lines.append("=" * W)
        lines.append(f"  REVENUE FORECAST  |  {suite.ticker}  |  Base Year: {suite.base_year}")
        lines.append(f"  Base Revenue: ${suite.base_revenue:,.0f}M  |  Horizon: {suite.n_forecast_years} years")
        lines.append("=" * W)
        lines.append("")

        # ---- Method comparison ----
        lines.append("  METHOD COMPARISON")
        lines.append("  " + "-" * (W - 2))
        lines.append(f"  {'Method':<30} {'Proj CAGR':>10} {'R-sq':>7} {'MAE ($M)':>10} {'MAPE':>8}")
        lines.append("  " + "-" * (W - 2))

        for r in suite.results:
            mark = " (*)" if r.method == suite.recommended.method else "    "
            cagr_s = f"{r.cagr_projected * 100:.1f}%"
            rsq_s = f"{r.r_squared:.3f}" if r.r_squared is not None else "N/A"
            mae_s = f"{r.mae:,.0f}" if r.mae is not None else "N/A"
            mape_s = f"{r.mape * 100:.1f}%" if r.mape is not None else "N/A"
            lines.append(
                f"  {r.method_label + mark:<30} {cagr_s:>10} {rsq_s:>7} {mae_s:>10} {mape_s:>8}"
            )

        lines.append(f"\n  (*) recommended\n")

        # ---- Recommended forecast projection ----
        rec = suite.recommended
        lines.append(f"  RECOMMENDED FORECAST  ({rec.method_label})")
        lines.append("  " + "-" * (W - 2))
        lines.append(f"  {'Year':<6} {'Revenue ($M)':>14} {'YoY Growth':>11} {'Lower 95%':>12} {'Upper 95%':>12}")
        lines.append("  " + "-" * (W - 2))

        prev = suite.base_revenue
        for p in rec.projected_only:
            g = (p.value - prev) / prev * 100 if prev > 0 else 0
            lo = f"${p.lower_95:>10,.0f}M" if p.lower_95 is not None else "     N/A    "
            hi = f"${p.upper_95:>10,.0f}M" if p.upper_95 is not None else "     N/A    "
            lines.append(
                f"  {p.year:<6} ${p.value:>12,.0f}M {g:>+10.1f}% {lo} {hi}"
            )
            prev = p.value

        lines.append("")
        lines.append(f"  Projected CAGR ({suite.n_forecast_years}-yr): {rec.cagr_projected * 100:.2f}%")
        lines.append(f"  DCF Growth Rates: {', '.join(f'{r*100:.1f}%' for r in suite.dcf_growth_rates)}")
        lines.append("")
        lines.append("=" * W)
        return "\n".join(lines)
