"""
data/models/scenario.py
------------------------
Pydantic v2 data models for the Scenario Analysis module.

Model hierarchy:
  ScenarioTag             - Bull / Base / Bear / Custom enum label
  ScenarioDefinition      - named DCF assumption set for one scenario
  ScenarioResult          - one DCF run output bound to its ScenarioDefinition
  MonteCarloConfig        - controls for the Monte Carlo simulation
  MonteCarloResult        - full distribution statistics from N simulations
  ScenarioAnalysisResult  - top-level output: 3 scenario results + optional MC

Design notes:
  - Each scenario stores its full DCFAssumptions so reports can always show
    exactly what drove each outcome — no hidden globals.
  - MonteCarloConfig.seed makes simulations reproducible for testing.
  - MonteCarloResult includes pct_5/pct_95 and pct_10/pct_90 for risk reporting.
  - probability_above_current is the fraction of simulations where the implied
    price exceeds the current market price (None if no current price is known).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from data.models.dcf import DCFAssumptions, DCFResult


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

class ScenarioTag(str, Enum):
    BULL = "bull"
    BASE = "base"
    BEAR = "bear"
    CUSTOM = "custom"


class ScenarioDefinition(BaseModel):
    """
    Named DCF assumption set for one scenario.

    label and description appear in printed summaries and HTML reports.
    assumptions is passed directly to DCFEngine.run().
    """

    tag: ScenarioTag
    label: str
    description: str
    assumptions: DCFAssumptions


class ScenarioResult(BaseModel):
    """One DCF run under one named scenario."""

    scenario: ScenarioDefinition
    dcf_result: DCFResult


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

class MonteCarloConfig(BaseModel):
    """
    Controls for the Monte Carlo simulation.

    The four drivers — revenue_growth, ebitda_margin, wacc, tgr — are drawn
    jointly from a multivariate normal distribution governed by `correlations`.
    Independent draws would be financially illiterate: the same macro shock
    (e.g. an inflation spike) simultaneously lifts WACC, squeezes margins via
    input-cost pressure, and nudges the terminal growth rate upward through the
    nominal-GDP channel.  Treating them as independent produces a
    statistically well-shaped bell curve that is economically meaningless.

    Default correlation matrix  (rows/cols: growth, margin, wacc, tgr)
    ─────────────────────────────────────────────────────────────────────
          growth  margin  wacc   tgr
      g [  1.00   0.25  -0.20   0.30 ]
      m [  0.25   1.00  -0.35   0.10 ]
      w [ -0.20  -0.35   1.00   0.40 ]
      t [  0.30   0.10   0.40   1.00 ]

    Economic rationale for each pair:
      growth ↔ margin  (+0.25): operating leverage — revenue growth expands margins
      growth ↔ wacc    (-0.20): recessions compress growth AND widen credit spreads
      growth ↔ tgr     (+0.30): long-run TGR tied to long-run nominal GDP growth
      margin ↔ wacc    (-0.35): inflation spikes raise WACC and squeeze input margins
      margin ↔ tgr     (+0.10): structural profitability weakly predicts longevity
      wacc   ↔ tgr     (+0.40): both driven by nominal rates / inflation expectations

    The matrix must be symmetric and positive semi-definite.  A small diagonal
    regularisation (1e-8 × I) is applied in the engine to guarantee numerical
    stability of the Cholesky decomposition.

    Setting seed makes the sequence reproducible (used in tests).
    """

    n_simulations: int = Field(1000, ge=100, le=50_000, description="Number of Monte Carlo draws")
    revenue_growth_std: float = Field(0.03, description="Std dev for revenue growth perturbation (decimal)")
    ebitda_margin_std: float = Field(0.02, description="Std dev for EBITDA margin perturbation (decimal)")
    wacc_std: float = Field(0.01, description="Std dev for WACC perturbation (decimal)")
    tgr_std: float = Field(0.005, description="Std dev for terminal growth rate perturbation (decimal)")
    seed: Optional[int] = None

    # 4×4 correlation matrix for [revenue_growth, ebitda_margin, wacc, tgr].
    # Override to impose a different macro regime (e.g. stagflation, deflation).
    correlations: list[list[float]] = Field(
        default=[
            [ 1.00,  0.25, -0.20,  0.30],
            [ 0.25,  1.00, -0.35,  0.10],
            [-0.20, -0.35,  1.00,  0.40],
            [ 0.30,  0.10,  0.40,  1.00],
        ],
        description=(
            "4×4 correlation matrix for [revenue_growth, ebitda_margin, wacc, tgr]. "
            "Must be symmetric and positive semi-definite."
        ),
    )


class MonteCarloResult(BaseModel):
    """
    Distribution of implied share prices from Monte Carlo simulation.

    n_valid <= n_simulations because simulations where WACC <= tgr
    (after perturbation) produce no valid price and are excluded from statistics.
    """

    n_simulations: int
    n_valid: int
    mean_price: float
    median_price: float
    std_price: float
    pct_5: float
    pct_10: float
    pct_25: float
    pct_75: float
    pct_90: float
    pct_95: float
    current_price: Optional[float] = None
    probability_above_current: Optional[float] = None


# ---------------------------------------------------------------------------
# Full scenario analysis result
# ---------------------------------------------------------------------------

class ScenarioAnalysisResult(BaseModel):
    """
    Top-level output of one scenario analysis run.

    scenarios:   always 3 entries ordered [Bull, Base, Bear]
    monte_carlo: populated only when mc_config is passed to ScenarioAnalyzer.run()
    """

    ticker: str
    base_year: str
    scenarios: list[ScenarioResult]
    monte_carlo: Optional[MonteCarloResult] = None
