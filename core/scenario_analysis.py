"""
core/scenario_analysis.py
---------------------------
Scenario Analysis Engine — Phase 1, Module 4.

Responsibilities:
  1. Build Bull / Base / Bear DCF assumption sets from a base (auto-derived or
     user-provided) by applying signed deltas to key drivers.
  2. Run the full DCF pipeline for each scenario and collect results.
  3. Optionally run a Monte Carlo simulation: draw N perturbations from
     normal distributions around the base assumptions, run a fast DCF
     (sensitivity table skipped), and compute the implied-price distribution.
  4. Render a side-by-side ASCII comparison of the three scenarios plus an
     optional MC distribution summary.

Bull / Bear construction (deltas applied to base):
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

Monte Carlo:
  The four drivers [growth, margin, wacc, tgr] are drawn jointly from a
  multivariate normal distribution using the covariance matrix implied by
  MonteCarloConfig.correlations × individual std devs.  Independent draws
  would be financially illiterate: an inflation spike simultaneously raises
  WACC, squeezes margins via input-cost pressure, and nudges TGR upward
  through the nominal-GDP channel.  Correlated sampling captures this.
  DCF is run with build_sensitivity=False (skips the 9x9 grid) for speed.
  Simulations where WACC <= tgr after perturbation are discarded.

Public API:
    from core.scenario_analysis import ScenarioAnalyzer
    from data.models.scenario import MonteCarloConfig

    analyzer = ScenarioAnalyzer()
    result = analyzer.run(history)                          # 3 scenarios only
    result = analyzer.run(history, mc_config=MonteCarloConfig(n_simulations=2000, seed=42))
    print(analyzer.summary(result))
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from data.models.dcf import DCFAssumptions
from data.models.financials import FullFinancialHistory
from data.models.scenario import (
    MonteCarloConfig,
    MonteCarloResult,
    ScenarioAnalysisResult,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioTag,
)
from core.dcf_engine import DCFEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario deltas
# ---------------------------------------------------------------------------

_SCENARIO_SPECS: list[tuple[ScenarioTag, str, str, dict]] = [
    (
        ScenarioTag.BULL,
        "Bull Case",
        "Revenue acceleration, margin expansion, lower cost of capital",
        {"growth_delta": +0.03, "margin_delta": +0.02, "wacc_delta": -0.005, "tgr_delta": +0.0025},
    ),
    (
        ScenarioTag.BASE,
        "Base Case",
        "Auto-derived assumptions from historical financials",
        {"growth_delta": 0.0, "margin_delta": 0.0, "wacc_delta": 0.0, "tgr_delta": 0.0},
    ),
    (
        ScenarioTag.BEAR,
        "Bear Case",
        "Revenue deceleration, margin compression, higher cost of capital",
        {"growth_delta": -0.03, "margin_delta": -0.02, "wacc_delta": +0.005, "tgr_delta": -0.0025},
    ),
]


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
        print(analyzer.summary(result))
    """

    def __init__(self) -> None:
        self._engine = DCFEngine()

    def run(
        self,
        history: FullFinancialHistory,
        base_assumptions: Optional[DCFAssumptions] = None,
        mc_config: Optional[MonteCarloConfig] = None,
    ) -> ScenarioAnalysisResult:
        """
        Run scenario analysis for the given company history.

        Args:
            history:           FullFinancialHistory from FinancialStatementAnalyzer.
            base_assumptions:  Override the base DCF assumptions. If None,
                               auto-derived from history via DCFEngine._derive_assumptions.
            mc_config:         If provided, run Monte Carlo simulation with this config.

        Returns:
            ScenarioAnalysisResult containing 3 scenario DCF results and
            an optional MonteCarloResult.
        """
        if not history.annual_snapshots:
            raise ValueError(f"No financial data for {history.profile.ticker}")

        # Build base assumptions
        if base_assumptions is None:
            base_assumptions = self._engine._derive_assumptions(history)

        # Run the base DCF first to get the effective WACC
        base_dcf = self._engine.run(history, assumptions=base_assumptions)
        effective_wacc = base_dcf.wacc_result.wacc

        logger.info(
            "ScenarioAnalyzer: %s | base WACC=%.2f%% | base price=$%.2f",
            history.profile.ticker, effective_wacc * 100, base_dcf.bridge.implied_share_price
        )

        # Build and run all three scenarios
        scenario_results: list[ScenarioResult] = []
        for tag, label, desc, deltas in _SCENARIO_SPECS:
            adj = _apply_deltas(base_assumptions, effective_wacc, **deltas)
            definition = ScenarioDefinition(tag=tag, label=label, description=desc, assumptions=adj)
            try:
                dcf_result = self._engine.run(history, assumptions=adj)
            except ValueError as exc:
                logger.warning("Scenario '%s' failed: %s", label, exc)
                continue
            scenario_results.append(ScenarioResult(scenario=definition, dcf_result=dcf_result))

        # Monte Carlo
        mc_result: Optional[MonteCarloResult] = None
        if mc_config is not None:
            mc_result = self._run_monte_carlo(history, base_assumptions, effective_wacc, mc_config)

        return ScenarioAnalysisResult(
            ticker=history.profile.ticker,
            base_year=history.latest.period,
            scenarios=scenario_results,
            monte_carlo=mc_result,
        )

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def _run_monte_carlo(
        self,
        history: FullFinancialHistory,
        base: DCFAssumptions,
        effective_wacc: float,
        config: MonteCarloConfig,
    ) -> MonteCarloResult:
        """
        Draw N perturbations from a correlated multivariate normal, run a fast
        DCF for each, and compute the distribution of implied share prices.

        The joint draw captures real-world co-movement between drivers: an
        inflation shock simultaneously lifts WACC, squeezes margins (input
        costs), and nudges TGR upward (nominal GDP channel).  Independent draws
        would produce a statistically clean but economically meaningless
        distribution.
        """
        rng = np.random.default_rng(config.seed)
        n = config.n_simulations

        # Build covariance matrix: Σ[i,j] = ρ[i,j] × σ[i] × σ[j]
        stds = np.array([
            config.revenue_growth_std,
            config.ebitda_margin_std,
            config.wacc_std,
            config.tgr_std,
        ])
        rho = np.array(config.correlations, dtype=float)
        cov = np.outer(stds, stds) * rho
        # Small regularisation guarantees positive definiteness for Cholesky
        cov += np.eye(4) * 1e-8

        # Single joint draw: shape (n, 4) — columns are [growth, margin, wacc, tgr]
        draws = rng.multivariate_normal(mean=np.zeros(4), cov=cov, size=n)

        prices: list[float] = []
        for i in range(n):
            try:
                sampled = _apply_deltas(
                    base, effective_wacc,
                    growth_delta=float(draws[i, 0]),
                    margin_delta=float(draws[i, 1]),
                    wacc_delta=float(draws[i, 2]),
                    tgr_delta=float(draws[i, 3]),
                )
                result = self._engine.run(history, assumptions=sampled, build_sensitivity=False)
                p = result.bridge.implied_share_price
                if p is not None and p > 0:
                    prices.append(p)
            except (ValueError, ZeroDivisionError):
                continue

        n_valid = len(prices)
        if n_valid == 0:
            raise ValueError("Monte Carlo produced no valid simulations — check base assumptions.")

        arr = np.array(prices)
        current = history.profile.current_price
        prob_above = (float(np.mean(arr > current)) if current is not None else None)

        return MonteCarloResult(
            n_simulations=n,
            n_valid=n_valid,
            mean_price=float(np.mean(arr)),
            median_price=float(np.median(arr)),
            std_price=float(np.std(arr)),
            pct_5=float(np.percentile(arr, 5)),
            pct_10=float(np.percentile(arr, 10)),
            pct_25=float(np.percentile(arr, 25)),
            pct_75=float(np.percentile(arr, 75)),
            pct_90=float(np.percentile(arr, 90)),
            pct_95=float(np.percentile(arr, 95)),
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

        # Column headers
        cols = [sr.scenario.label for sr in result.scenarios]
        col_w = 14
        lines.append(f"  {'Metric':<28}" + "".join(f"{c:>{col_w}}" for c in cols))
        _rule()

        def _row(label: str, vals: list[str]) -> str:
            return f"  {label:<28}" + "".join(f"{v:>{col_w}}" for v in vals)

        scenarios = result.scenarios
        # Revenue growth CAGR (geometric mean of all years' growth rates)
        cagrs = []
        for sr in scenarios:
            rates = sr.scenario.assumptions.revenue_growth_rates
            import math
            product = 1.0
            for r in rates:
                product *= (1 + r)
            cagr = product ** (1 / len(rates)) - 1
            cagrs.append(_pct(cagr))
        lines.append(_row("Revenue CAGR", cagrs))

        # EBITDA margin
        lines.append(_row(
            "EBITDA Margin",
            [_pct(sr.scenario.assumptions.ebitda_margin) for sr in scenarios]
        ))

        # Effective WACC
        lines.append(_row(
            "WACC",
            [_pct(sr.dcf_result.wacc_result.wacc) for sr in scenarios]
        ))

        # Terminal growth rate
        lines.append(_row(
            "Terminal Growth Rate",
            [_pct(sr.scenario.assumptions.terminal_growth_rate) for sr in scenarios]
        ))

        _rule()

        # Implied prices
        lines.append(_row(
            "Implied Share Price",
            [_usd(sr.dcf_result.bridge.implied_share_price) for sr in scenarios]
        ))

        # Upside/downside vs current
        current = (
            result.scenarios[0].dcf_result.bridge.current_price
            if result.scenarios else None
        )
        lines.append(_row(
            f"vs Current ({_usd(current)})",
            [_updown(sr.dcf_result.bridge.implied_share_price, current) for sr in scenarios]
        ))

        # TV as % of EV
        lines.append(_row(
            "TV as % of EV",
            [
                _pct(sr.dcf_result.terminal_value_result.pv_as_pct_of_ev)
                for sr in scenarios
            ]
        ))

        lines.append("")

        # ---- Monte Carlo ------------------------------------------------
        if result.monte_carlo is not None:
            mc = result.monte_carlo
            lines.append(f"  MONTE CARLO SIMULATION  (n = {mc.n_simulations:,} | valid = {mc.n_valid:,})")
            _rule()
            lines.append(f"  {'Mean Implied Price':<30}  {_usd(mc.mean_price)}")
            lines.append(f"  {'Median':<30}  {_usd(mc.median_price)}")
            lines.append(f"  {'Std Dev':<30}  {_usd(mc.std_price)}")
            lines.append(f"  {'5th Percentile':<30}  {_usd(mc.pct_5)}")
            lines.append(f"  {'25th Percentile':<30}  {_usd(mc.pct_25)}")
            lines.append(f"  {'75th Percentile':<30}  {_usd(mc.pct_75)}")
            lines.append(f"  {'95th Percentile':<30}  {_usd(mc.pct_95)}")
            if mc.probability_above_current is not None:
                lines.append(f"  {'P(Price > Current)':<30}  {mc.probability_above_current * 100:.1f}%")
            lines.append("")

        _bar()
        return "\n".join(lines)
