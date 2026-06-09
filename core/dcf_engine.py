"""
core/dcf_engine.py
-------------------
DCF Engine — Phase 1, Module 2.

Responsibilities:
  1. Compute WACC from CAPM inputs or accept a direct override.
  2. Project revenue, EBITDA, EBIT, NOPAT, CapEx, NWC, and Free Cash Flow
     for each year of the explicit forecast period.
  3. Compute terminal value via Gordon Growth Model or Exit EV/EBITDA multiple.
  4. Discount all cash flows back to present value and build the
     EV → equity value → implied share price bridge.
  5. Generate a 2-D sensitivity table varying WACC and terminal growth rate
     (or exit multiple) across ±200 bps and ±100 bps respectively.

FCF build-up (one projected year):
  Revenue            = prior_revenue × (1 + growth_rate)
  EBITDA             = Revenue × ebitda_margin
  D&A                = Revenue × da_as_pct_revenue
  EBIT               = EBITDA − D&A
  NOPAT              = EBIT × (1 − tax_rate)
  CapEx              = Revenue × capex_as_pct_revenue               [outflow, +]
  ΔNWC               = ΔRevenue × nwc_change_as_pct_revenue_delta  [outflow, +]
  Net Investment     = CapEx − D&A + ΔNWC
  FCF                = NOPAT − Net Investment
  PV(FCF)            = FCF / (1 + WACC)^year

Terminal value (Gordon Growth):
  TV  = FCF_n × (1 + tgr) / (WACC − tgr)       [requires WACC > tgr]
  PV(TV) = TV / (1 + WACC)^n

Terminal value (Exit Multiple):
  TV  = EBITDA_n × exit_ev_ebitda_multiple
  PV(TV) = TV / (1 + WACC)^n

EV bridge:
  EV           = Σ PV(FCF_t) + PV(TV)
  Equity Value = EV − net_debt − minority_interest
  Price        = Equity Value / diluted_shares

Sensitivity table:
  Rows: WACC swept from (base − 200 bps) to (base + 200 bps) in 50 bps steps
  Cols: tgr swept from (base − 100 bps) to (base + 100 bps) in 25 bps steps
        (or exit multiple ± 2× in 0.5× steps when exit_multiple method is used)

Public API:
    from core.dcf_engine import DCFEngine
    from core.financial_statements import FinancialStatementAnalyzer

    history = FinancialStatementAnalyzer().analyze("AAPL")
    engine  = DCFEngine()

    # Auto-derive assumptions from history (uses latest actuals as base)
    result  = engine.run(history)

    # Or pass custom assumptions
    from data.models.dcf import DCFAssumptions, WACCInputs
    assumptions = DCFAssumptions(
        revenue_growth_rates=[0.10, 0.09, 0.08, 0.07, 0.06],
        ebitda_margin=0.32,
        wacc_inputs=WACCInputs(beta=1.2, risk_free_rate=0.045),
        terminal_growth_rate=0.025,
    )
    result = engine.run(history, assumptions=assumptions)
    print(engine.summary(result))
"""

from __future__ import annotations

import logging
from typing import Optional

from data.models.dcf import (
    DCFAssumptions,
    DCFBridge,
    DCFResult,
    ProjectedYear,
    SensitivityTable,
    TerminalValueMethod,
    TerminalValueResult,
    WACCInputs,
    WACCResult,
)
from data.models.financials import FullFinancialHistory
from core.wacc_deriver import WACCDeriver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WACC Calculator
# ---------------------------------------------------------------------------

class WACCCalculator:
    """
    Computes WACC from CAPM-based inputs.

    Formula:
      Ke   = Rf + β × ERP
      Kd   = pre_tax_cost_of_debt × (1 − t)
      WACC = Ke × We + Kd × Wd
    """

    def compute(self, inputs: WACCInputs) -> WACCResult:
        ke = inputs.cost_of_equity   # rfr + adj_beta * erp + size_premium
        kd_at = inputs.after_tax_cost_of_debt
        wacc = ke * inputs.equity_weight + kd_at * inputs.debt_weight

        logger.debug(
            "WACC: Ke=%.2f%% Kd(AT)=%.2f%% We=%.0f%% Wd=%.0f%% → WACC=%.2f%%",
            ke * 100, kd_at * 100,
            inputs.equity_weight * 100, inputs.debt_weight * 100,
            wacc * 100,
        )

        return WACCResult(
            wacc=wacc,
            cost_of_equity=ke,
            after_tax_cost_of_debt=kd_at,
            equity_weight=inputs.equity_weight,
            debt_weight=inputs.debt_weight,
            beta=inputs.beta,
            raw_beta=inputs.raw_beta,
            risk_free_rate=inputs.risk_free_rate,
            equity_risk_premium=inputs.equity_risk_premium,
            size_premium=inputs.size_premium,
            pre_tax_cost_of_debt=inputs.cost_of_debt,
            synthetic_rating=inputs.synthetic_rating,
            tax_rate=inputs.tax_rate,
        )


# ---------------------------------------------------------------------------
# FCF Projector
# ---------------------------------------------------------------------------

class FCFProjector:
    """
    Projects Free Cash Flow for each year of the explicit forecast period.

    Takes the base year revenue from the latest FinancialSnapshot and applies
    the assumption set year-by-year to produce a list of ProjectedYear objects.
    """

    def project(
        self,
        base_revenue: float,
        wacc: float,
        assumptions: DCFAssumptions,
        base_calendar_year: Optional[int] = None,
    ) -> list[ProjectedYear]:
        """
        Args:
            base_revenue:        Last reported annual revenue (millions USD).
            wacc:                Discount rate as a decimal.
            assumptions:         Full DCFAssumptions.
            base_calendar_year:  Calendar year of the base period (for labelling).

        Returns:
            List of ProjectedYear, one per projection year, oldest first.
        """
        projected: list[ProjectedYear] = []
        prior_revenue = base_revenue

        for i, g in enumerate(assumptions.revenue_growth_rates):
            year_num = i + 1
            cal_year = (base_calendar_year + year_num) if base_calendar_year else None

            revenue = prior_revenue * (1.0 + g)
            ebitda = revenue * assumptions.ebitda_margin
            da = revenue * assumptions.da_as_pct_revenue
            ebit = ebitda - da
            nopat = ebit * (1.0 - assumptions.tax_rate)

            capex = revenue * assumptions.capex_as_pct_revenue
            delta_rev = revenue - prior_revenue
            delta_nwc = delta_rev * assumptions.nwc_change_as_pct_revenue_delta
            net_investment = capex - da + delta_nwc

            fcf = nopat - net_investment
            discount_factor = 1.0 / ((1.0 + wacc) ** year_num)
            pv_fcf = fcf * discount_factor

            projected.append(ProjectedYear(
                year=year_num,
                calendar_year=cal_year,
                revenue=revenue,
                revenue_growth_rate=g,
                ebitda=ebitda,
                ebitda_margin=assumptions.ebitda_margin,
                depreciation_amortization=da,
                ebit=ebit,
                ebit_margin=ebit / revenue if revenue else 0.0,
                tax_rate=assumptions.tax_rate,
                nopat=nopat,
                capital_expenditures=capex,
                change_in_nwc=delta_nwc,
                net_investment=net_investment,
                free_cash_flow=fcf,
                discount_factor=discount_factor,
                pv_free_cash_flow=pv_fcf,
            ))

            prior_revenue = revenue

        return projected


# ---------------------------------------------------------------------------
# Terminal Value Calculator
# ---------------------------------------------------------------------------

class TerminalValueCalculator:
    """
    Computes terminal value using either Gordon Growth or Exit EV/EBITDA.

    Gordon Growth:
        TV = FCF_n × (1 + tgr) / (WACC − tgr)
        Only valid when WACC > tgr. Raises ValueError otherwise.

    Exit Multiple:
        TV = EBITDA_n × exit_multiple
        Simpler and anchored to current market pricing; less sensitive to
        the WACC vs. tgr spread but requires a defensible comparable multiple.
    """

    def compute(
        self,
        terminal_year: ProjectedYear,
        wacc: float,
        assumptions: DCFAssumptions,
        n: int,
    ) -> TerminalValueResult:
        """
        Args:
            terminal_year:  Last ProjectedYear (year N).
            wacc:           Discount rate.
            assumptions:    Full DCFAssumptions.
            n:              Number of projection years (for discounting TV).
        """
        method = assumptions.terminal_value_method
        tgr = assumptions.terminal_growth_rate

        if method == TerminalValueMethod.GORDON_GROWTH:
            if wacc <= tgr:
                raise ValueError(
                    f"WACC ({wacc:.2%}) must exceed terminal growth rate ({tgr:.2%}) "
                    "for Gordon Growth terminal value to be valid."
                )
            tv = terminal_year.free_cash_flow * (1.0 + tgr) / (wacc - tgr)
            exit_multiple_used = None
        else:
            if assumptions.exit_ev_ebitda_multiple is None:
                raise ValueError("exit_ev_ebitda_multiple must be set for exit_multiple method.")
            tv = terminal_year.ebitda * assumptions.exit_ev_ebitda_multiple
            exit_multiple_used = assumptions.exit_ev_ebitda_multiple

        pv_tv = tv / ((1.0 + wacc) ** n)

        return TerminalValueResult(
            method=method,
            terminal_year_ebitda=terminal_year.ebitda,
            terminal_year_fcf=terminal_year.free_cash_flow,
            terminal_growth_rate=tgr if method == TerminalValueMethod.GORDON_GROWTH else None,
            exit_multiple=exit_multiple_used,
            terminal_value=tv,
            pv_terminal_value=pv_tv,
        )


# ---------------------------------------------------------------------------
# Sensitivity Table Builder
# ---------------------------------------------------------------------------

class SensitivityBuilder:
    """
    Builds a 2-D grid of implied share prices by re-running the DCF core
    (TV + bridge only — no need to re-project FCFs) across WACC × tgr space.

    Row axis: WACC swept ±200 bps around base in 50 bps steps → 9 values
    Col axis: tgr swept ±100 bps around base in 25 bps steps → 9 values
              (or exit multiple ± 2× in 0.5× steps when exit method used)
    """

    def build(
        self,
        projected_years: list[ProjectedYear],
        bridge: DCFBridge,
        assumptions: DCFAssumptions,
        base_wacc: float,
    ) -> SensitivityTable:
        n = len(projected_years)
        terminal_year = projected_years[-1]

        # Row axis: WACC
        wacc_steps = [-0.020, -0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015, 0.020]
        row_axis = [round(base_wacc + s, 5) for s in wacc_steps]

        # Col axis: tgr or exit multiple
        if assumptions.terminal_value_method == TerminalValueMethod.GORDON_GROWTH:
            base_tgr = assumptions.terminal_growth_rate
            tgr_steps = [-0.010, -0.0075, -0.005, -0.0025, 0.0, 0.0025, 0.005, 0.0075, 0.010]
            col_axis = [round(base_tgr + s, 5) for s in tgr_steps]
            col_label = "Terminal Growth Rate"
        else:
            base_mult = assumptions.exit_ev_ebitda_multiple or 10.0
            mult_steps = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
            col_axis = [round(base_mult + s, 1) for s in mult_steps]
            col_label = "Exit EV/EBITDA Multiple"

        prices: list[list[Optional[float]]] = []
        tv_calc = TerminalValueCalculator()

        for wacc in row_axis:
            row: list[Optional[float]] = []
            for col_val in col_axis:
                try:
                    # Re-compute PV of explicit FCFs with this WACC
                    pv_fcfs = sum(
                        py.free_cash_flow / ((1.0 + wacc) ** py.year)
                        for py in projected_years
                    )

                    # Re-compute terminal value
                    sens_assumptions = assumptions.model_copy(deep=True)
                    if assumptions.terminal_value_method == TerminalValueMethod.GORDON_GROWTH:
                        sens_assumptions.terminal_growth_rate = col_val
                    else:
                        sens_assumptions.exit_ev_ebitda_multiple = col_val

                    tv_result = tv_calc.compute(terminal_year, wacc, sens_assumptions, n)

                    ev = pv_fcfs + tv_result.pv_terminal_value
                    equity_val = ev - bridge.net_debt - bridge.minority_interest
                    price = equity_val / bridge.shares_outstanding if bridge.shares_outstanding > 0 else None
                    row.append(round(price, 2) if price is not None else None)

                except (ValueError, ZeroDivisionError):
                    row.append(None)  # invalid combination (e.g. WACC ≤ tgr)

            prices.append(row)

        return SensitivityTable(
            col_label=col_label,
            row_axis=row_axis,
            col_axis=col_axis,
            prices=prices,
        )


# ---------------------------------------------------------------------------
# Main DCF Engine
# ---------------------------------------------------------------------------

class DCFEngine:
    """
    Orchestrates the full DCF pipeline for a given FullFinancialHistory.

    Usage:
        engine = DCFEngine()
        result = engine.run(history)                          # auto assumptions
        result = engine.run(history, assumptions=my_assum)   # custom assumptions
        print(engine.summary(result))
    """

    def __init__(self) -> None:
        self._wacc_calc = WACCCalculator()
        self._fcf_proj = FCFProjector()
        self._tv_calc = TerminalValueCalculator()
        self._sens_builder = SensitivityBuilder()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        history: FullFinancialHistory,
        assumptions: Optional[DCFAssumptions] = None,
        build_sensitivity: bool = True,
    ) -> DCFResult:
        """
        Run the full DCF pipeline.

        Args:
            history:     FullFinancialHistory from FinancialStatementAnalyzer.
            assumptions: Override assumptions. If None, auto-derived from history.

        Returns:
            DCFResult with every intermediate and final value populated.

        Raises:
            ValueError: if history has no snapshots or WACC ≤ tgr.
        """
        if not history.annual_snapshots:
            raise ValueError(f"No financial data available for {history.profile.ticker}")

        ticker = history.profile.ticker

        # Prefer TTM when it is more current than the last annual 10-K.
        # A September 2024 10-K used in June 2025 means Year-1 "growth" already
        # has 9 months of actuals baked in — TTM corrects the base.
        base_snap = history.base_snapshot
        logger.info("Running DCF for %s (base: %s)", ticker, base_snap.period)

        # --- Build assumptions ---
        if assumptions is None:
            assumptions = self._derive_assumptions(history)

        # --- Resolve WACC ---
        wacc_result = self._resolve_wacc(assumptions, history)
        wacc = wacc_result.wacc

        # --- Project FCFs ---
        base_revenue = base_snap.statements.income_statement.revenue
        if base_revenue is None:
            raise ValueError(f"Base snapshot for {ticker} has no revenue data.")

        base_year = int(base_snap.period[:4])
        projected = self._fcf_proj.project(base_revenue, wacc, assumptions, base_year)

        # --- Terminal value ---
        tv_result = self._tv_calc.compute(projected[-1], wacc, assumptions, assumptions.projection_years)

        # --- EV bridge ---
        bridge = self._build_bridge(tv_result, projected, assumptions, history)

        # Fill TV as % of EV
        tv_result.pv_as_pct_of_ev = (
            tv_result.pv_terminal_value / bridge.enterprise_value
            if bridge.enterprise_value else None
        )

        # --- Sensitivity table ---
        if build_sensitivity:
            sensitivity = self._sens_builder.build(projected, bridge, assumptions, wacc)
        else:
            sensitivity = SensitivityTable(
                row_axis=[round(wacc, 5)],
                col_axis=[round(assumptions.terminal_growth_rate, 5)],
                prices=[[round(bridge.implied_share_price, 2)]],
            )

        return DCFResult(
            ticker=ticker,
            base_year=base_snap.period,
            assumptions=assumptions,
            wacc_result=wacc_result,
            projected_years=projected,
            terminal_value_result=tv_result,
            bridge=bridge,
            sensitivity=sensitivity,
        )

    # ------------------------------------------------------------------
    # Auto-derive assumptions from historical data
    # ------------------------------------------------------------------

    def _derive_assumptions(self, history: FullFinancialHistory) -> DCFAssumptions:
        """
        Build DCFAssumptions from historical data using regime-aware growth
        projection and margin-trend extrapolation.

        Growth derivation (4 regimes):
          Regime A — Declining + no recovery signal (last 2 growth rates < 0):
            Do NOT assume mean-reversion to positive GDP growth. Use a
            partial-recovery path: stay negative Y1, modest improvement Y2-Y3,
            conservative recovery Y4-Y5. A dying company should not magically
            reverts to 2.5% growth in year 5 without analyst justification.

          Regime B — Fast deceleration (OLS slope on growth < -2pp/yr):
            Extrapolate the trend 2 years forward before fading to TGR. Avoids
            the error of assuming a decelerating company suddenly stabilises.

          Regime C — Accelerating growth (OLS slope > +1pp/yr):
            Preserve momentum for Y1-Y2 before fading. Avoids punishing a
            high-growth company by immediately dragging its Year-1 rate down
            toward a long-run average.

          Regime D — Stable / mature (default):
            Standard convex fade from exponentially-weighted recent growth to
            TGR. Uses recency-biased weighting (most recent year 2× prior)
            rather than a simple average, so the most recent data carries more
            information about current trajectory.

        Margin derivation:
          Detects OLS trend in EBITDA margin over the last 4 years. Projects
          forward 2 years of that trend before stabilising. An expanding-margin
          business should not be penalised with a flat-average assumption, and
          a structurally deteriorating business should not get false mean-
          reversion.

        Structural ratios (D&A, CapEx, tax rate):
          Simple recent average is appropriate here — these are capital-
          structure / accounting parameters, not business-cycle indicators.
        """
        snaps = history.annual_snapshots
        profile = history.profile
        n_proj = 5
        tgr = 0.025  # long-run nominal GDP-like sustainable growth

        # ------------------------------------------------------------------
        # Step 1: collect historical revenue growth rates
        # ------------------------------------------------------------------
        hist_growths = [
            s.ratios.growth.revenue_growth
            for s in snaps[1:]
            if s.ratios.growth.revenue_growth is not None
        ]

        if not hist_growths:
            base_growth = 0.06
            growth_trend_slope = 0.0
        else:
            n_g = len(hist_growths)
            # Exponential recency weights: [1, 2, 4, ...] — most recent is highest.
            # This means if a company grew 50% two years ago but only 5% last year,
            # the 5% dominates the average rather than being equally diluted.
            weights = [2.0 ** i for i in range(n_g)]
            total_w = sum(weights)
            ewa_growth = sum(w * g for w, g in zip(weights, hist_growths)) / total_w
            base_growth = max(min(ewa_growth, 0.50), -0.20)

            # OLS slope across historical growth rates — detects acceleration/deceleration
            if n_g >= 3:
                t = list(range(n_g))
                t_mean = sum(t) / n_g
                g_mean = sum(hist_growths) / n_g
                sxy = sum((ti - t_mean) * (gi - g_mean) for ti, gi in zip(t, hist_growths))
                sxx = sum((ti - t_mean) ** 2 for ti in t)
                growth_trend_slope = sxy / sxx if sxx > 0 else 0.0
            else:
                growth_trend_slope = 0.0

        # ------------------------------------------------------------------
        # Step 2: select growth regime and build projection
        # ------------------------------------------------------------------
        last_two_both_negative = (
            len(hist_growths) >= 2
            and hist_growths[-1] < 0
            and hist_growths[-2] < 0
        )

        if base_growth < 0 and (last_two_both_negative or growth_trend_slope <= 0):
            # Regime A: persistent decline — no false recovery
            growth_rates = [
                base_growth,
                base_growth * 0.70,
                max(base_growth * 0.40, -0.05),
                max(tgr * 0.50, 0.005),
                tgr,
            ]

        elif growth_trend_slope < -0.02:
            # Regime B: fast deceleration — extrapolate trend before fading
            extrapolated = base_growth + growth_trend_slope * 2
            fade_start = max(extrapolated, tgr)
            growth_rates = [
                base_growth + (fade_start - base_growth) * (i / (n_proj - 1))
                for i in range(n_proj)
            ]

        elif growth_trend_slope > 0.01:
            # Regime C: accelerating growth — preserve momentum Y1-Y2
            peak = min(base_growth + growth_trend_slope * 2, 0.50)
            growth_rates = []
            for i in range(n_proj):
                if i == 0:
                    g = base_growth + growth_trend_slope
                elif i == 1:
                    g = peak
                else:
                    # Linear fade from peak to tgr over remaining years
                    fade = (i - 1) / (n_proj - 2)
                    g = peak + (tgr - peak) * fade
                growth_rates.append(max(min(g, 0.50), -0.20))

        else:
            # Regime D: stable — convex fade from recent base to TGR
            growth_rates = [
                base_growth + (tgr - base_growth) * (i / (n_proj - 1))
                for i in range(n_proj)
            ]

        # ------------------------------------------------------------------
        # Step 3: EBITDA margin — trend-projected, not flat average
        # ------------------------------------------------------------------
        ebitda_margin_series = [
            s.ratios.profitability.ebitda_margin
            for s in snaps[-4:]
            if s.ratios.profitability.ebitda_margin is not None
            and s.ratios.profitability.ebitda_margin > 0
        ]

        if ebitda_margin_series:
            n_m = len(ebitda_margin_series)
            avg_margin = sum(ebitda_margin_series) / n_m
            if n_m >= 3:
                t = list(range(n_m))
                t_mean = sum(t) / n_m
                m_mean = avg_margin
                sxy = sum((ti - t_mean) * (mi - m_mean)
                          for ti, mi in zip(t, ebitda_margin_series))
                sxx = sum((ti - t_mean) ** 2 for ti in t)
                margin_slope = sxy / sxx if sxx > 0 else 0.0
                # Project 2 years of trend from most recent level, then stabilise.
                # Cap at [5%, 60%] — beyond these, the model is unreliable anyway.
                recent_margin = ebitda_margin_series[-1]
                projected = recent_margin + margin_slope * 2
                ebitda_margin = max(min(projected, 0.60), 0.05)
            else:
                ebitda_margin = avg_margin
        else:
            ebitda_margin = 0.20

        # ------------------------------------------------------------------
        # Step 4: structural ratios — simple recent average is fine here
        # ------------------------------------------------------------------
        da_pcts, capex_pcts, tax_rates_raw = [], [], []
        for s in snaps[-3:]:
            inc = s.statements.income_statement
            cf = s.statements.cash_flow_statement
            rev = inc.revenue
            if rev and rev > 0:
                da = inc.depreciation_amortization
                if da:
                    da_pcts.append(abs(da) / rev)
                capex = cf.capital_expenditures
                if capex:
                    capex_pcts.append(abs(capex) / rev)
            if inc.income_tax and inc.pretax_income and inc.pretax_income > 0:
                tax_rates_raw.append(inc.income_tax / inc.pretax_income)

        da_pct = sum(da_pcts) / len(da_pcts) if da_pcts else 0.04
        capex_pct = sum(capex_pcts) / len(capex_pcts) if capex_pcts else 0.04
        tax_rate = sum(tax_rates_raw) / len(tax_rates_raw) if tax_rates_raw else 0.21
        tax_rate = max(min(tax_rate, 0.40), 0.05)

        # ------------------------------------------------------------------
        # Step 5: balance sheet items + market-calibrated WACC inputs
        #
        # WACCDeriver handles: live RFR (^TNX), Blume's beta adjustment,
        # Kroll-style size premium, synthetic Kd from ICR → default spread,
        # and market-value capital weights (shares * price + book debt).
        # Use base_snapshot (TTM when available) for the most current figures.
        # ------------------------------------------------------------------
        base_snap = history.base_snapshot
        net_debt = base_snap.statements.balance_sheet.net_debt
        shares = (
            base_snap.statements.income_statement.shares_diluted
            or profile.shares_outstanding
        )
        wacc_inputs = WACCDeriver().derive(history, tax_rate=round(tax_rate, 4))

        return DCFAssumptions(
            projection_years=n_proj,
            revenue_growth_rates=[round(g, 5) for g in growth_rates],
            ebitda_margin=round(ebitda_margin, 4),
            da_as_pct_revenue=round(da_pct, 4),
            tax_rate=round(tax_rate, 4),
            capex_as_pct_revenue=round(capex_pct, 4),
            wacc_inputs=wacc_inputs,
            terminal_growth_rate=tgr,
            net_debt=net_debt,
            shares_outstanding=shares,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_wacc(
        self, assumptions: DCFAssumptions, history: FullFinancialHistory
    ) -> WACCResult:
        """Return WACC result — either from override or computed via CAPM."""
        if assumptions.wacc_override is not None:
            w = assumptions.wacc_override
            logger.info("Using WACC override: %.2f%%", w * 100)
            # Build a synthetic WACCResult so the report can always show a breakdown
            return WACCResult(
                wacc=w,
                cost_of_equity=w,
                after_tax_cost_of_debt=w,
                equity_weight=1.0,
                debt_weight=0.0,
                beta=history.profile.beta or 1.0,
                risk_free_rate=0.0,
                equity_risk_premium=0.0,
                pre_tax_cost_of_debt=0.0,
                tax_rate=0.0,
            )

        inputs = assumptions.wacc_inputs or WACCInputs()
        result = self._wacc_calc.compute(inputs)
        logger.info("Computed WACC: %.2f%%", result.wacc * 100)
        return result

    def _build_bridge(
        self,
        tv_result: TerminalValueResult,
        projected: list[ProjectedYear],
        assumptions: DCFAssumptions,
        history: FullFinancialHistory,
    ) -> DCFBridge:
        pv_fcfs = sum(py.pv_free_cash_flow for py in projected)
        ev = pv_fcfs + tv_result.pv_terminal_value

        # Use assumption-level net_debt/shares; fall back to latest snapshot
        latest = history.latest
        net_debt = assumptions.net_debt
        if net_debt is None:
            net_debt = latest.statements.balance_sheet.net_debt or 0.0

        shares = assumptions.shares_outstanding
        if shares is None:
            shares = (
                latest.statements.income_statement.shares_diluted
                or history.profile.shares_outstanding
                or 0.0
            )

        equity_val = ev - net_debt - assumptions.minority_interest
        price = equity_val / shares if shares > 0 else 0.0

        current_price = history.profile.current_price
        upside = ((price - current_price) / current_price) if current_price and price else None

        return DCFBridge(
            pv_explicit_fcfs=pv_fcfs,
            pv_terminal_value=tv_result.pv_terminal_value,
            enterprise_value=ev,
            net_debt=net_debt,
            minority_interest=assumptions.minority_interest,
            equity_value=equity_val,
            shares_outstanding=shares,
            implied_share_price=price,
            current_price=current_price,
            upside_downside_pct=upside,
        )

    # ------------------------------------------------------------------
    # Summary / display
    # ------------------------------------------------------------------

    def summary(self, result: DCFResult) -> str:
        """
        Formatted text summary of the DCF result.

        Sections:
          1. Valuation summary (implied price vs current, upside/downside)
          2. WACC build-up
          3. FCF projection table
          4. EV bridge
          5. Sensitivity table (WACC × tgr grid)
        """
        lines: list[str] = []
        b = result.bridge
        w = result.wacc_result
        tv = result.terminal_value_result

        # ---- Header ----
        lines.append("=" * 72)
        lines.append(f"  DCF VALUATION  |  {result.ticker}  |  Base Year: {result.base_year}")
        lines.append("=" * 72)

        # ---- Implied price ----
        lines.append(f"\n  Implied Share Price : ${b.implied_share_price:,.2f}")
        if b.current_price:
            dir_sym = "(+)" if (b.upside_downside_pct or 0) >= 0 else "(-)"
            lines.append(f"  Current Price       : ${b.current_price:,.2f}")
            lines.append(f"  Upside / Downside   : {dir_sym} {abs(b.upside_downside_pct or 0) * 100:.1f}%")

        # ---- WACC build-up ----
        lines.append("\n  WACC Build-Up")
        lines.append("  " + "-" * 44)
        lines.append(f"  Risk-Free Rate (10-yr)    : {w.risk_free_rate * 100:.2f}%")
        if w.raw_beta is not None:
            lines.append(f"  Beta (raw / Yahoo)        : {w.raw_beta:.2f}x")
            lines.append(f"  Beta (Blume adj.)         : {w.beta:.2f}x  (= 0.67*raw + 0.33)")
        else:
            lines.append(f"  Beta                      : {w.beta:.2f}x")
        lines.append(f"  Equity Risk Premium       : {w.equity_risk_premium * 100:.2f}%")
        if w.size_premium > 0:
            lines.append(f"  Size Premium              : {w.size_premium * 100:.2f}%")
        lines.append(f"  Cost of Equity (Ke)       : {w.cost_of_equity * 100:.2f}%")
        kd_label = (
            f"  Pre-Tax Kd [{w.synthetic_rating}]" if w.synthetic_rating
            else "  Pre-Tax Cost of Debt"
        )
        lines.append(f"  {kd_label:<28}: {w.pre_tax_cost_of_debt * 100:.2f}%")
        lines.append(f"  After-Tax Kd              : {w.after_tax_cost_of_debt * 100:.2f}%")
        lines.append(f"  Equity Weight (mkt)       : {w.equity_weight * 100:.0f}%")
        lines.append(f"  Debt Weight (book)        : {w.debt_weight * 100:.0f}%")
        lines.append(f"  " + "-" * 44)
        lines.append(f"  WACC                      : {w.wacc * 100:.2f}%")

        # ---- FCF projection table ----
        lines.append("\n  FCF Projection")
        lines.append("  " + "-" * 80)
        hdr = f"  {'Year':<6} {'Cal Yr':<8} {'Revenue':>10} {'Rev Gr':>7} {'EBITDA':>10} "
        hdr += f"{'EBIT':>10} {'FCF':>10} {'PV(FCF)':>10}"
        lines.append(hdr)
        lines.append("  " + "-" * 80)

        for py in result.projected_years:
            cal = str(py.calendar_year) if py.calendar_year else "-"
            lines.append(
                f"  {py.year:<6} {cal:<8} "
                f"${py.revenue:>9,.0f} "
                f"{py.revenue_growth_rate * 100:>6.1f}% "
                f"${py.ebitda:>9,.0f} "
                f"${py.ebit:>9,.0f} "
                f"${py.free_cash_flow:>9,.0f} "
                f"${py.pv_free_cash_flow:>9,.0f}"
            )

        lines.append(f"\n  Terminal Value  ({tv.method.value})")
        lines.append(f"  Terminal Year EBITDA   : ${tv.terminal_year_ebitda:,.0f}M")
        lines.append(f"  Terminal Year FCF      : ${tv.terminal_year_fcf:,.0f}M")
        if tv.terminal_growth_rate is not None:
            lines.append(f"  Terminal Growth Rate   : {tv.terminal_growth_rate * 100:.2f}%")
        if tv.exit_multiple is not None:
            lines.append(f"  Exit EV/EBITDA         : {tv.exit_multiple:.1f}x")
        lines.append(f"  Terminal Value (undiscounted) : ${tv.terminal_value:,.0f}M")
        lines.append(f"  PV of Terminal Value          : ${tv.pv_terminal_value:,.0f}M")
        if tv.pv_as_pct_of_ev is not None:
            lines.append(f"  TV as % of EV                 : {tv.pv_as_pct_of_ev * 100:.1f}%")

        # ---- EV bridge ----
        lines.append("\n  EV -> Equity Bridge")
        lines.append("  " + "-" * 40)
        lines.append(f"  PV of Explicit FCFs    : ${b.pv_explicit_fcfs:>12,.0f}M")
        lines.append(f"  PV of Terminal Value   : ${b.pv_terminal_value:>12,.0f}M")
        lines.append(f"  Enterprise Value       : ${b.enterprise_value:>12,.0f}M")
        lines.append(f"  (-) Net Debt           : ${b.net_debt:>12,.0f}M")
        lines.append(f"  (-) Minority Interest  : ${b.minority_interest:>12,.0f}M")
        lines.append(f"  Equity Value           : ${b.equity_value:>12,.0f}M")
        lines.append(f"  Shares Outstanding     : {b.shares_outstanding:>12,.0f}M")
        lines.append(f"  Implied Share Price    : ${b.implied_share_price:>12,.2f}")

        # ---- Sensitivity table ----
        s = result.sensitivity
        lines.append(f"\n  Sensitivity Table  (Rows: WACC  |  Cols: {s.col_label})")
        lines.append("  " + "-" * (12 + 10 * len(s.col_axis)))

        # Header row
        col_headers = "  " + " " * 8
        for cv in s.col_axis:
            if s.col_label == "Terminal Growth Rate":
                col_headers += f"{cv * 100:>8.2f}%"
            else:
                col_headers += f"{cv:>8.1f}x"
        lines.append(col_headers)
        lines.append("  " + "-" * (12 + 10 * len(s.col_axis)))

        for i, wacc_val in enumerate(s.row_axis):
            marker = " <-" if abs(wacc_val - result.wacc_result.wacc) < 0.0001 else "   "
            row_str = f"  {wacc_val * 100:>6.2f}%{marker}"
            for price in s.prices[i]:
                if price is None:
                    row_str += f"{'N/A':>9} "
                else:
                    row_str += f"${price:>8,.2f}"
            lines.append(row_str)

        lines.append("\n" + "=" * 72)
        return "\n".join(lines)
