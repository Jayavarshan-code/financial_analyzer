"""
core/scenario_analysis.py
---------------------------
Scenario Analysis Engine — Phase 1, Module 4.

Responsibilities:
  1. Build Bull / Base / Bear DCF assumption sets from a base (auto-derived or
     user-provided) by applying signed ScenarioDelta objects to key drivers.
  2. Run the full DCF pipeline for each scenario and collect results.
  3. Optionally run a fully vectorized Monte Carlo simulation: draw N
     correlated perturbations via explicit Cholesky decomposition and evaluate
     all paths simultaneously as numpy matrix operations — no Python loop.
  4. Render a side-by-side ASCII comparison of the three scenarios plus an
     optional MC distribution summary.

Bull / Bear defaults (overridable per-call via bull_delta / bear_delta):
  Driver               Bull      Bear
  Revenue growth/yr    +3 pp     -3 pp
  EBITDA margin        +2 pp     -2 pp
  WACC (effective)     -50 bp    +50 bp
  Terminal growth      +25 bp    -25 bp

All driver values are clipped to safe ranges after delta application:
  revenue_growth: [-30%, +50%]   ebitda_margin: [1%, 80%]
  WACC:           [4%, 30%]      tgr:           [0%, 5%]
  WACC > tgr is enforced by pulling tgr down 0.5 pp if needed.

Scenarios always use wacc_override (the effective WACC from the base run)
so each scenario is a pure single-variable change from the base, making
scenario comparisons interpretable.

Monte Carlo — vectorized implementation:
  Correlated draws: Σ = diag(σ) × ρ × diag(σ); L = cholesky(Σ); Z ~ N(0,I);
                    draws = Z @ L.T  (shape n×4, cols: growth/margin/wacc/tgr)
  DCF math:         all n simulations evaluated in parallel as numpy matrix
                    operations — revenue cumprod, FCF build-up, discounting,
                    terminal value, and bridge in a single vectorized pass.
  Speedup:          ~50-100× vs. a Python for-loop calling DCFEngine.run().
  Validity filter:  price > 0 and finite; WACC > TGR enforced element-wise.

Public API:
    from core.scenario_analysis import ScenarioAnalyzer
    from data.models.scenario import MonteCarloConfig, ScenarioDelta

    analyzer = ScenarioAnalyzer()
    result = analyzer.run(history)
    result = analyzer.run(
        history,
        bull_delta=ScenarioDelta(growth_delta=0.05, margin_delta=0.03),
        bear_delta=ScenarioDelta(growth_delta=-0.05, margin_delta=-0.04, wacc_delta=0.015),
        mc_config=MonteCarloConfig(n_simulations=2000, seed=42),
    )
    print(analyzer.summary(result))
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from data.models.dcf import DCFAssumptions
from data.models.financials import FullFinancialHistory
from data.models.scenario import (
    MonteCarloConfig,
    MonteCarloResult,
    ScenarioAnalysisResult,
    ScenarioDelta,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioTag,
)
from core.dcf_engine import DCFEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default scenario deltas
# ---------------------------------------------------------------------------

_DEFAULT_BULL_DELTA = ScenarioDelta(
    growth_delta=+0.03,
    margin_delta=+0.02,
    wacc_delta=-0.005,
    tgr_delta=+0.0025,
)
_DEFAULT_BEAR_DELTA = ScenarioDelta(
    growth_delta=-0.03,
    margin_delta=-0.02,
    wacc_delta=+0.005,
    tgr_delta=-0.0025,
)

# Labels and descriptions keyed by tag — independent of delta magnitudes
_SCENARIO_META: dict[ScenarioTag, tuple[str, str]] = {
    ScenarioTag.BULL: (
        "Bull Case",
        "Revenue acceleration, margin expansion, lower cost of capital",
    ),
    ScenarioTag.BASE: (
        "Base Case",
        "Auto-derived assumptions from historical financials",
    ),
    ScenarioTag.BEAR: (
        "Bear Case",
        "Revenue deceleration, margin compression, higher cost of capital",
    ),
}


def _apply_deltas(
    base: DCFAssumptions,
    effective_wacc: float,
    growth_delta: float,
    margin_delta: float,
    wacc_delta: float,
    tgr_delta: float,
) -> DCFAssumptions:
    """
    Apply deltas to base assumptions and return a new DCFAssumptions.

    Always uses wacc_override (the computed effective WACC from the base run)
    so that scenario WACC changes are absolute shifts, not CAPM re-derivations.
    """
    new_growth = [
        max(-0.30, min(0.50, g + growth_delta))
        for g in base.revenue_growth_rates
    ]
    new_margin = max(0.01, min(0.80, base.ebitda_margin + margin_delta))
    new_wacc = max(0.04, min(0.30, effective_wacc + wacc_delta))
    new_tgr = max(0.0, min(0.05, base.terminal_growth_rate + tgr_delta))

    # Ensure WACC strictly exceeds tgr (Gordon Growth requirement)
    if new_wacc <= new_tgr:
        new_tgr = new_wacc - 0.005

    return base.model_copy(update={
        "revenue_growth_rates": new_growth,
        "ebitda_margin": new_margin,
        "wacc_override": round(new_wacc, 6),
        "terminal_growth_rate": round(new_tgr, 6),
    })


# ---------------------------------------------------------------------------
# ScenarioAnalyzer
# ---------------------------------------------------------------------------

class ScenarioAnalyzer:
    """
    Runs Bull / Base / Bear scenarios and an optional Monte Carlo simulation
    for a given FullFinancialHistory.

    Usage:
        analyzer = ScenarioAnalyzer()
        result   = analyzer.run(history)
        # Custom deltas:
        result   = analyzer.run(
            history,
            bull_delta=ScenarioDelta(growth_delta=0.05, margin_delta=0.03),
            bear_delta=ScenarioDelta(growth_delta=-0.05, wacc_delta=0.015),
        )
        print(analyzer.summary(result))
    """

    def __init__(self) -> None:
        self._engine = DCFEngine()

    def run(
        self,
        history: FullFinancialHistory,
        base_assumptions: Optional[DCFAssumptions] = None,
        mc_config: Optional[MonteCarloConfig] = None,
        bull_delta: Optional[ScenarioDelta] = None,
        bear_delta: Optional[ScenarioDelta] = None,
    ) -> ScenarioAnalysisResult:
        """
        Run scenario analysis for the given company history.

        Args:
            history:           FullFinancialHistory from FinancialStatementAnalyzer.
            base_assumptions:  Override the base DCF assumptions. If None,
                               auto-derived from history via DCFEngine._derive_assumptions.
            mc_config:         If provided, run a vectorized Monte Carlo simulation.
            bull_delta:        Custom Bull-case driver shifts. Defaults to
                               +3pp growth / +2pp margin / -50bp WACC / +25bp TGR.
            bear_delta:        Custom Bear-case driver shifts. Defaults to
                               -3pp growth / -2pp margin / +50bp WACC / -25bp TGR.

        Returns:
            ScenarioAnalysisResult with 3 scenario DCF results and optional MC.
        """
        if not history.annual_snapshots:
            raise ValueError(f"No financial data for {history.profile.ticker}")

        # Build base assumptions
        if base_assumptions is None:
            base_assumptions = self._engine._derive_assumptions(history)

        # Run the base DCF once to get the effective WACC
        base_dcf = self._engine.run(history, assumptions=base_assumptions)
        effective_wacc = base_dcf.wacc_result.wacc

        logger.info(
            "ScenarioAnalyzer: %s | base WACC=%.2f%% | base price=$%.2f",
            history.profile.ticker,
            effective_wacc * 100,
            base_dcf.bridge.implied_share_price,
        )

        # Resolve deltas: caller-supplied → defaults
        _bull = bull_delta or _DEFAULT_BULL_DELTA
        _bear = bear_delta or _DEFAULT_BEAR_DELTA
        _base = ScenarioDelta()  # all zeros

        scenario_specs = [
            (ScenarioTag.BULL, _bull),
            (ScenarioTag.BASE, _base),
            (ScenarioTag.BEAR, _bear),
        ]

        # Build and run all three scenarios
        scenario_results: list[ScenarioResult] = []
        for tag, delta in scenario_specs:
            label, desc = _SCENARIO_META[tag]
            adj = _apply_deltas(
                base_assumptions, effective_wacc,
                growth_delta=delta.growth_delta,
                margin_delta=delta.margin_delta,
                wacc_delta=delta.wacc_delta,
                tgr_delta=delta.tgr_delta,
            )
            definition = ScenarioDefinition(
                tag=tag, label=label, description=desc, assumptions=adj
            )
            try:
                dcf_result = self._engine.run(history, assumptions=adj)
            except ValueError as exc:
                logger.warning("Scenario '%s' failed: %s", label, exc)
                continue
            scenario_results.append(
                ScenarioResult(scenario=definition, dcf_result=dcf_result)
            )

        # Monte Carlo (vectorized)
        mc_result: Optional[MonteCarloResult] = None
        if mc_config is not None:
            mc_result = self._run_monte_carlo(
                history, base_assumptions, effective_wacc, mc_config
            )

        return ScenarioAnalysisResult(
            ticker=history.profile.ticker,
            base_year=history.base_snapshot.period,
            scenarios=scenario_results,
            monte_carlo=mc_result,
        )

    # ------------------------------------------------------------------
    # Monte Carlo — fully vectorized
    # ------------------------------------------------------------------

    def _run_monte_carlo(
        self,
        history: FullFinancialHistory,
        base: DCFAssumptions,
        effective_wacc: float,
        config: MonteCarloConfig,
    ) -> MonteCarloResult:
        """
        Vectorized Monte Carlo: all n simulations evaluated simultaneously.

        Algorithm:
          1. Build Σ = diag(σ) × ρ × diag(σ)  from config.correlations.
          2. Factorize via L = cholesky(Σ) — validates positive-definiteness
             upfront, fails fast on a bad correlation matrix.
          3. Draw Z ~ N(0, I) of shape (n, 4); correlated draws = Z @ L.T.
          4. Apply deltas and clip to safe bounds — all (n,) or (n, n_proj) arrays.
          5. Compute full DCF in one numpy pass:
               revenue    = base_rev × cumprod(1 + g_rates, axis=1)
               FCF        = NOPAT − CapEx + D&A − ΔNWC           (n × n_proj)
               PV(FCF)    = (FCF × discount_factors).sum(axis=1)  (n,)
               TV         = FCF_n × (1+tgr) / (wacc−tgr)         (n,)
               price      = (PV(FCF) + PV(TV) − net_debt) / shares
          6. Filter price > 0 and finite; collect distribution statistics.

        Speedup vs. Python for-loop: ~50-100× for n=1000.
        """
        rng = np.random.default_rng(config.seed)
        n = config.n_simulations

        # ── Step 1-2: Cholesky factorization of the covariance matrix ────
        stds = np.array([
            config.revenue_growth_std,
            config.ebitda_margin_std,
            config.wacc_std,
            config.tgr_std,
        ])
        rho = np.array(config.correlations, dtype=float)
        cov = np.outer(stds, stds) * rho
        cov += np.eye(4) * 1e-8  # guarantee numerical positive-definiteness

        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "MonteCarloConfig.correlations is not positive definite — "
                "check that the matrix is symmetric with |ρ[i,j]| < 1."
            ) from exc

        # ── Step 3: correlated draws ─────────────────────────────────────
        # Z shape (n, 4); draws = Z @ L.T gives Σ-correlated perturbations
        Z = rng.standard_normal((n, 4))
        draws = Z @ L.T  # (n, 4): cols = [Δgrowth, Δmargin, Δwacc, Δtgr]

        # ── Step 4: extract base DCF parameters ──────────────────────────
        base_growth = np.array(base.revenue_growth_rates)   # (n_proj,)
        n_proj = len(base_growth)

        base_snap = history.base_snapshot
        base_revenue = base_snap.statements.income_statement.revenue or 0.0
        net_debt = (
            base.net_debt
            if base.net_debt is not None
            else (base_snap.statements.balance_sheet.net_debt or 0.0)
        )
        shares = float(
            base.shares_outstanding
            or base_snap.statements.income_statement.shares_diluted
            or history.profile.shares_outstanding
            or 1.0
        )
        minority = base.minority_interest

        da_pct    = base.da_as_pct_revenue
        capex_pct = base.capex_as_pct_revenue
        nwc_pct   = base.nwc_change_as_pct_revenue_delta
        tax       = base.tax_rate

        # ── Step 4 (cont.): apply deltas and clip ────────────────────────
        # g_rates (n, n_proj): same Δgrowth applied to every projection year
        g_rates = np.clip(
            base_growth[np.newaxis, :] + draws[:, 0:1],
            -0.30, 0.50,
        )
        margins = np.clip(base.ebitda_margin       + draws[:, 1], 0.01, 0.80)
        waccs   = np.clip(effective_wacc            + draws[:, 2], 0.04, 0.30)
        tgrs    = np.clip(base.terminal_growth_rate + draws[:, 3], 0.00, 0.05)

        # Gordon Growth requires WACC > TGR — enforce element-wise
        tgrs = np.where(waccs <= tgrs, waccs - 0.005, tgrs)

        # ── Step 5: vectorized DCF ────────────────────────────────────────
        # Revenue projections  (n, n_proj)
        revenue = base_revenue * np.cumprod(1.0 + g_rates, axis=1)

        # Prior-year revenue for ΔNWC; prepend base_revenue as year-0
        prev_rev = np.concatenate(
            [np.full((n, 1), base_revenue), revenue[:, :-1]], axis=1
        )

        # FCF build-up  (n, n_proj)
        m      = margins[:, np.newaxis]          # (n, 1) → broadcasts over years
        ebitda = revenue * m
        da     = revenue * da_pct
        ebit   = ebitda - da
        nopat  = ebit * (1.0 - tax)
        capex  = revenue * capex_pct
        dnwc   = (revenue - prev_rev) * nwc_pct
        fcf    = nopat - (capex - da + dnwc)     # (n, n_proj)

        # Discount factors  (n, n_proj)
        years = np.arange(1, n_proj + 1, dtype=float)
        disc  = 1.0 / (1.0 + waccs[:, np.newaxis]) ** years
        pv_fcf_sum = (fcf * disc).sum(axis=1)    # (n,)

        # Terminal value  (n,)
        tv    = fcf[:, -1] * (1.0 + tgrs) / (waccs - tgrs)
        pv_tv = tv / (1.0 + waccs) ** n_proj

        # Bridge: EV → equity → price  (n,)
        ev     = pv_fcf_sum + pv_tv
        equity = ev - net_debt - minority
        prices = equity / shares

        # ── Step 6: filter and collect statistics ─────────────────────────
        valid_mask   = (prices > 0) & np.isfinite(prices)
        valid_prices = prices[valid_mask]
        n_valid      = int(valid_mask.sum())

        if n_valid == 0:
            raise ValueError(
                "Monte Carlo produced no valid simulations — check base assumptions."
            )

        current    = history.profile.current_price
        prob_above = (
            float(np.mean(valid_prices > current))
            if current is not None else None
        )

        return MonteCarloResult(
            n_simulations=n,
            n_valid=n_valid,
            mean_price=float(np.mean(valid_prices)),
            median_price=float(np.median(valid_prices)),
            std_price=float(np.std(valid_prices)),
            pct_5=float(np.percentile(valid_prices,  5)),
            pct_10=float(np.percentile(valid_prices, 10)),
            pct_25=float(np.percentile(valid_prices, 25)),
            pct_75=float(np.percentile(valid_prices, 75)),
            pct_90=float(np.percentile(valid_prices, 90)),
            pct_95=float(np.percentile(valid_prices, 95)),
            current_price=current,
            probability_above_current=prob_above,
        )

    # ------------------------------------------------------------------
    # Summary / display
    # ------------------------------------------------------------------

    def summary(self, result: ScenarioAnalysisResult) -> str:
        """
        Formatted ASCII summary.

        Sections:
          1. Side-by-side scenario comparison table
          2. Monte Carlo distribution (if available)
        """
        lines: list[str] = []
        W = 74

        def _bar():
            lines.append("=" * W)

        def _rule():
            lines.append("  " + "-" * (W - 2))

        def _pct(v: Optional[float]) -> str:
            return f"{v * 100:.1f}%" if v is not None else "N/A"

        def _usd(v: Optional[float]) -> str:
            return f"${v:,.2f}" if v is not None else "N/A"

        def _updown(implied: Optional[float], current: Optional[float]) -> str:
            if implied is None or current is None or current == 0:
                return "N/A"
            pct = (implied - current) / current * 100
            sign = "+" if pct >= 0 else ""
            return f"{sign}{pct:.1f}%"

        _bar()
        lines.append(
            f"  SCENARIO ANALYSIS  |  {result.ticker}  |  Base Year: {result.base_year}"
        )
        _bar()
        lines.append("")

        # ---- Scenario comparison table ----------------------------------
        lines.append("  SCENARIO COMPARISON")
        _rule()

        cols = [sr.scenario.label for sr in result.scenarios]
        col_w = 14
        lines.append(f"  {'Metric':<28}" + "".join(f"{c:>{col_w}}" for c in cols))
        _rule()

        def _row(label: str, vals: list[str]) -> str:
            return f"  {label:<28}" + "".join(f"{v:>{col_w}}" for v in vals)

        scenarios = result.scenarios

        # Revenue growth CAGR
        cagrs = []
        for sr in scenarios:
            rates = sr.scenario.assumptions.revenue_growth_rates
            product = 1.0
            for r in rates:
                product *= (1 + r)
            cagr = product ** (1 / len(rates)) - 1
            cagrs.append(_pct(cagr))
        lines.append(_row("Revenue CAGR", cagrs))

        lines.append(_row(
            "EBITDA Margin",
            [_pct(sr.scenario.assumptions.ebitda_margin) for sr in scenarios],
        ))
        lines.append(_row(
            "WACC",
            [_pct(sr.dcf_result.wacc_result.wacc) for sr in scenarios],
        ))
        lines.append(_row(
            "Terminal Growth Rate",
            [_pct(sr.scenario.assumptions.terminal_growth_rate) for sr in scenarios],
        ))
        _rule()

        lines.append(_row(
            "Implied Share Price",
            [_usd(sr.dcf_result.bridge.implied_share_price) for sr in scenarios],
        ))

        current = (
            result.scenarios[0].dcf_result.bridge.current_price
            if result.scenarios else None
        )
        lines.append(_row(
            f"vs Current ({_usd(current)})",
            [_updown(sr.dcf_result.bridge.implied_share_price, current)
             for sr in scenarios],
        ))
        lines.append(_row(
            "TV as % of EV",
            [_pct(sr.dcf_result.terminal_value_result.pv_as_pct_of_ev)
             for sr in scenarios],
        ))
        lines.append("")

        # ---- Monte Carlo ------------------------------------------------
        if result.monte_carlo is not None:
            mc = result.monte_carlo
            lines.append(
                f"  MONTE CARLO SIMULATION  (n = {mc.n_simulations:,} | valid = {mc.n_valid:,})"
            )
            _rule()
            lines.append(f"  {'Mean Implied Price':<30}  {_usd(mc.mean_price)}")
            lines.append(f"  {'Median':<30}  {_usd(mc.median_price)}")
            lines.append(f"  {'Std Dev':<30}  {_usd(mc.std_price)}")
            lines.append(f"  {'5th Percentile':<30}  {_usd(mc.pct_5)}")
            lines.append(f"  {'25th Percentile':<30}  {_usd(mc.pct_25)}")
            lines.append(f"  {'75th Percentile':<30}  {_usd(mc.pct_75)}")
            lines.append(f"  {'95th Percentile':<30}  {_usd(mc.pct_95)}")
            if mc.probability_above_current is not None:
                lines.append(
                    f"  {'P(Price > Current)':<30}  "
                    f"{mc.probability_above_current * 100:.1f}%"
                )
            lines.append("")

        _bar()
        return "\n".join(lines)
