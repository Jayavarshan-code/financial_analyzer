"""
analytics/earnings_forecaster.py
----------------------------------
Earnings Forecasting Module — Phase 2, Module 2.

Projects net income and EPS forward by combining a revenue forecast with
margin assumptions derived from historical financial data.

Methods implemented:
  MARGIN_BASED — Applies historically-derived average margins to the
                 revenue forecast.  Margin assumptions:
                   ebitda_margin : average of last N years' EBITDA/Revenue
                   net_margin    : average of last N years' Net Income/Revenue
                 CI on EPS inherits directly from revenue CI (same margins applied
                 to lower_95 / upper_95 revenue bounds).

  EPS_TREND    — Fits an OLS linear trend directly to historical EPS,
                 independent of the revenue forecast.  Useful as a
                 cross-check when margins are volatile.

Recommended method: MARGIN_BASED when both revenue and margin data are
available; EPS_TREND as fallback when margin data is sparse.

All monetary values in millions USD.  EPS is per-share (not scaled).

Public API:
    from analytics.earnings_forecaster import EarningsForecastEngine
    from analytics.revenue_forecaster import RevenueForecastEngine

    rev_suite = RevenueForecastEngine().run(history, n_years=5)
    earn_suite = EarningsForecastEngine().run(history, rev_suite.recommended)
    print(EarningsForecastEngine().summary(earn_suite))
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from data.models.financials import FullFinancialHistory
from data.models.forecast import (
    EarningsForecastMethod,
    EarningsForecastPoint,
    EarningsForecastResult,
    EarningsForecastSuite,
    RevenueForecastResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_avg(values: list[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def _compute_cagr(start: float, end: float, n_periods: int) -> float:
    if n_periods <= 0 or start <= 0 or end <= 0:
        return 0.0
    return (end / start) ** (1 / n_periods) - 1


def _margin_ci(rev_ci: Optional[float], margin: float) -> Optional[float]:
    return rev_ci * margin if rev_ci is not None else None


# ---------------------------------------------------------------------------
# Method: Margin-Based
# ---------------------------------------------------------------------------

def _fit_margin_based(
    history: FullFinancialHistory,
    revenue_forecast: RevenueForecastResult,
    n_lookback: int = 3,
) -> Optional[EarningsForecastResult]:
    """
    Apply historically-averaged margins to the revenue forecast.

    Returns None when margin data is unavailable (e.g. company has no EBITDA data).
    """
    snaps = history.annual_snapshots[-n_lookback:]

    ebitda_margins = [s.ratios.profitability.ebitda_margin for s in snaps]
    net_margins = [s.ratios.profitability.net_margin for s in snaps]

    avg_ebitda_margin = _safe_avg(ebitda_margins)
    avg_net_margin = _safe_avg(net_margins)

    if avg_ebitda_margin is None and avg_net_margin is None:
        return None

    # Fall back to direct calculation when ratios module didn't populate
    if avg_ebitda_margin is None:
        raw = []
        for s in snaps:
            rev = s.statements.income_statement.revenue
            ebitda = s.statements.income_statement.ebitda
            if rev and ebitda and rev > 0:
                raw.append(ebitda / rev)
        avg_ebitda_margin = sum(raw) / len(raw) if raw else 0.25

    if avg_net_margin is None:
        raw = []
        for s in snaps:
            rev = s.statements.income_statement.revenue
            ni = s.statements.income_statement.net_income
            if rev and ni and rev > 0:
                raw.append(ni / rev)
        avg_net_margin = sum(raw) / len(raw) if raw else 0.15

    # Diluted shares (use most recent, stable assumption)
    shares = (
        history.latest.statements.income_statement.shares_diluted
        or history.profile.shares_outstanding
        or 1.0
    )

    # Historical actuals as context (last n_lookback years)
    actuals: list[EarningsForecastPoint] = []
    for snap in snaps:
        inc = snap.statements.income_statement
        if inc.revenue is None:
            continue
        rev = inc.revenue
        ebitda = inc.ebitda if inc.ebitda is not None else rev * avg_ebitda_margin
        ni = inc.net_income if inc.net_income is not None else rev * avg_net_margin
        sh = inc.shares_diluted or shares
        eps = ni / sh if sh > 0 else 0.0
        actuals.append(EarningsForecastPoint(
            year=int(snap.period[:4]),
            is_actual=True,
            revenue=rev,
            ebitda_margin=ebitda / rev if rev else avg_ebitda_margin,
            ebitda=ebitda,
            net_margin=ni / rev if rev else avg_net_margin,
            net_income=ni,
            shares_diluted=sh,
            eps=eps,
        ))

    # Projected points
    projected: list[EarningsForecastPoint] = []
    for fp in revenue_forecast.projected_only:
        rev = fp.value
        ebitda = rev * avg_ebitda_margin
        ni = rev * avg_net_margin
        eps = ni / shares if shares > 0 else 0.0

        eps_lower = _margin_ci(fp.lower_95, avg_net_margin) / shares if shares > 0 and fp.lower_95 else None
        eps_upper = _margin_ci(fp.upper_95, avg_net_margin) / shares if shares > 0 and fp.upper_95 else None

        projected.append(EarningsForecastPoint(
            year=fp.year,
            is_actual=False,
            revenue=rev,
            revenue_lower_95=fp.lower_95,
            revenue_upper_95=fp.upper_95,
            ebitda_margin=avg_ebitda_margin,
            ebitda=ebitda,
            net_margin=avg_net_margin,
            net_income=ni,
            shares_diluted=shares,
            eps=eps,
            eps_lower_95=eps_lower,
            eps_upper_95=eps_upper,
        ))

    if not projected:
        return None

    base_eps = actuals[-1].eps if actuals else projected[0].eps
    cagr_eps = _compute_cagr(base_eps, projected[-1].eps, len(projected)) if base_eps > 0 else 0.0
    base_ni = actuals[-1].net_income if actuals else projected[0].net_income
    cagr_ni = _compute_cagr(base_ni, projected[-1].net_income, len(projected)) if base_ni > 0 else 0.0

    return EarningsForecastResult(
        method=EarningsForecastMethod.MARGIN_BASED,
        method_label="Margin-Based",
        points=actuals + projected,
        projected_only=projected,
        cagr_eps=cagr_eps,
        cagr_net_income=cagr_ni,
        avg_ebitda_margin=avg_ebitda_margin,
        avg_net_margin=avg_net_margin,
    )


# ---------------------------------------------------------------------------
# Method: EPS Trend Regression
# ---------------------------------------------------------------------------

def _fit_eps_trend(
    history: FullFinancialHistory,
    n_forecast: int,
) -> Optional[EarningsForecastResult]:
    """
    OLS linear trend on historical EPS: eps = a + b * t.
    Independent of revenue forecast — serves as a cross-check.
    """
    pairs = []
    for snap in history.annual_snapshots:
        eps = snap.statements.income_statement.eps_diluted
        if eps is not None:
            pairs.append((int(snap.period[:4]), eps))
    pairs.sort()

    if len(pairs) < 2:
        return None

    years = [p[0] for p in pairs]
    eps_vals = [p[1] for p in pairs]

    t = np.arange(len(years), dtype=float)
    eps_arr = np.array(eps_vals)

    slope, intercept, r_value, _p, _se = sp_stats.linregress(t, eps_arr)

    fitted = intercept + slope * t
    residuals = eps_arr - fitted
    s = float(np.std(residuals, ddof=2)) if len(residuals) > 2 else float(np.std(residuals))

    t_mean = float(np.mean(t))
    Stt = float(np.sum((t - t_mean) ** 2))
    n = len(t)

    # Use last available shares as a constant for NI derivation
    shares = (
        history.latest.statements.income_statement.shares_diluted
        or history.profile.shares_outstanding
        or 1.0
    )
    last_revenue = history.latest.statements.income_statement.revenue or 0.0
    last_ebitda_margin = history.latest.ratios.profitability.ebitda_margin or 0.0
    last_net_margin = (
        history.latest.ratios.profitability.net_margin
        or (history.latest.statements.income_statement.net_income or 0) / last_revenue
        if last_revenue > 0 else 0.0
    )

    # Historical actuals
    actuals: list[EarningsForecastPoint] = []
    for yr, eps in zip(years, eps_vals):
        actuals.append(EarningsForecastPoint(
            year=yr, is_actual=True,
            revenue=0.0, ebitda_margin=0.0, ebitda=0.0,
            net_margin=0.0, net_income=eps * shares,
            shares_diluted=shares, eps=eps,
        ))

    projected: list[EarningsForecastPoint] = []
    for i in range(1, n_forecast + 1):
        t_new = len(years) - 1 + i
        eps_proj = float(intercept + slope * t_new)
        se_pred = s * np.sqrt(1 + 1 / n + (t_new - t_mean) ** 2 / Stt) if Stt > 0 else s
        eps_lower = eps_proj - 1.96 * se_pred
        eps_upper = eps_proj + 1.96 * se_pred
        ni = eps_proj * shares

        projected.append(EarningsForecastPoint(
            year=years[-1] + i, is_actual=False,
            revenue=0.0,  # not revenue-derived
            ebitda_margin=last_ebitda_margin, ebitda=0.0,
            net_margin=last_net_margin, net_income=ni,
            shares_diluted=shares, eps=eps_proj,
            eps_lower_95=eps_lower, eps_upper_95=eps_upper,
        ))

    if not projected:
        return None

    base_eps = eps_vals[-1] if eps_vals else 0.0
    cagr_eps = _compute_cagr(base_eps, projected[-1].eps, n_forecast) if base_eps > 0 else 0.0
    base_ni = base_eps * shares
    cagr_ni = _compute_cagr(base_ni, projected[-1].net_income, n_forecast) if base_ni > 0 else 0.0

    return EarningsForecastResult(
        method=EarningsForecastMethod.EPS_TREND,
        method_label="EPS OLS Trend",
        points=actuals + projected,
        projected_only=projected,
        cagr_eps=cagr_eps,
        cagr_net_income=cagr_ni,
        avg_ebitda_margin=last_ebitda_margin,
        avg_net_margin=last_net_margin,
    )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class EarningsForecastEngine:
    """
    Projects net income and EPS using margin-based and direct EPS-trend methods.

    Usage:
        from analytics.revenue_forecaster import RevenueForecastEngine
        rev = RevenueForecastEngine().run(history)
        earn = EarningsForecastEngine().run(history, rev.recommended)
        print(EarningsForecastEngine().summary(earn))
    """

    def run(
        self,
        history: FullFinancialHistory,
        revenue_forecast: RevenueForecastResult,
    ) -> EarningsForecastSuite:
        """
        Run all earnings forecasting methods and return a suite.

        Args:
            history:          FullFinancialHistory from FinancialStatementAnalyzer.
            revenue_forecast: RevenueForecastResult to use as the revenue input.

        Returns:
            EarningsForecastSuite with results for available methods.

        Raises:
            ValueError: if no earnings method produces a valid result.
        """
        results: list[EarningsForecastResult] = []
        n_years = len(revenue_forecast.projected_only)

        r_margin = _fit_margin_based(history, revenue_forecast)
        if r_margin is not None:
            results.append(r_margin)

        r_eps = _fit_eps_trend(history, n_years)
        if r_eps is not None:
            results.append(r_eps)

        if not results:
            raise ValueError(
                f"{history.profile.ticker}: no earnings forecasting method succeeded."
            )

        # MARGIN_BASED is preferred when available
        recommended = next(
            (r for r in results if r.method == EarningsForecastMethod.MARGIN_BASED),
            results[0],
        )

        return EarningsForecastSuite(
            ticker=history.profile.ticker,
            base_year=history.latest.period,
            revenue_forecast=revenue_forecast,
            results=results,
            recommended=recommended,
        )

    # ------------------------------------------------------------------
    # Summary / display
    # ------------------------------------------------------------------

    def summary(self, suite: EarningsForecastSuite) -> str:
        """
        Formatted ASCII summary.

        Sections:
          1. Method comparison (EPS CAGR, avg margins)
          2. Recommended forecast table (year, revenue, net income, EPS, CI)
        """
        lines: list[str] = []
        W = 76

        lines.append("=" * W)
        lines.append(
            f"  EARNINGS FORECAST  |  {suite.ticker}  |  Base Year: {suite.base_year}"
        )
        lines.append("=" * W)
        lines.append("")

        # ---- Method comparison ----
        lines.append("  METHOD COMPARISON")
        lines.append("  " + "-" * (W - 2))
        lines.append(
            f"  {'Method':<24} {'EPS CAGR':>9} {'NI CAGR':>9}"
            f" {'EBITDA%':>8} {'Net%':>7}"
        )
        lines.append("  " + "-" * (W - 2))
        for r in suite.results:
            mark = " (*)" if r.method == suite.recommended.method else "    "
            lines.append(
                f"  {r.method_label + mark:<24}"
                f" {r.cagr_eps * 100:>8.1f}%"
                f" {r.cagr_net_income * 100:>8.1f}%"
                f" {r.avg_ebitda_margin * 100:>7.1f}%"
                f" {r.avg_net_margin * 100:>6.1f}%"
            )
        lines.append(f"\n  (*) recommended\n")

        # ---- Recommended projection ----
        rec = suite.recommended
        lines.append(f"  PROJECTED EARNINGS  ({rec.method_label})")
        lines.append("  " + "-" * (W - 2))
        lines.append(
            f"  {'Year':<6} {'Revenue ($M)':>14} {'Net Income ($M)':>16}"
            f" {'EPS':>8} {'EPS Low':>9} {'EPS High':>9}"
        )
        lines.append("  " + "-" * (W - 2))
        for p in rec.projected_only:
            rev_s = f"${p.revenue:,.0f}M" if p.revenue else "N/A"
            ni_s = f"${p.net_income:,.0f}M"
            eps_lo = f"${p.eps_lower_95:.2f}" if p.eps_lower_95 is not None else "N/A"
            eps_hi = f"${p.eps_upper_95:.2f}" if p.eps_upper_95 is not None else "N/A"
            lines.append(
                f"  {p.year:<6} {rev_s:>14} {ni_s:>16}"
                f" ${p.eps:>7.2f} {eps_lo:>9} {eps_hi:>9}"
            )

        lines.append("")
        lines.append(f"  EPS CAGR ({suite.n_forecast_years if hasattr(suite, 'n_forecast_years') else len(rec.projected_only)}-yr): {rec.cagr_eps * 100:.2f}%")
        lines.append(f"  Avg EBITDA Margin: {rec.avg_ebitda_margin * 100:.1f}%  |  Avg Net Margin: {rec.avg_net_margin * 100:.1f}%")
        lines.append("")
        lines.append("=" * W)
        return "\n".join(lines)
