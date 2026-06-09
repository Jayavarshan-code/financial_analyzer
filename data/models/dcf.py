"""
data/models/dcf.py
-------------------
Pydantic v2 data models for the DCF Engine.

Model hierarchy:
  WACCInputs          – raw ingredients for WACC calculation (CAPM-based)
  WACCResult          – computed WACC with component breakdown
  DCFAssumptions      – complete input set for one DCF run
  ProjectedYear       – one year of the explicit forecast period
  TerminalValueResult – terminal value with methodology detail
  DCFBridge           – EV → equity value → implied price bridge
  SensitivityTable    – 2-D grid of implied prices vs. WACC × tgr (or exit multiple)
  DCFResult           – complete output: all of the above in one object

Design notes:
  - All monetary fields in millions USD (matching FinancialSnapshot convention).
  - DCFAssumptions supports two modes:
      (a) Provide wacc_override → skips WACCInputs entirely.
      (b) Provide WACCInputs → engine computes WACC via CAPM + after-tax cost of debt.
  - Revenue growth can be specified as:
      (a) A single flat rate applied to all projection years.
      (b) A list of per-year rates (must equal projection_years in length).
  - Two terminal value methodologies:
      Gordon Growth Model  : TV = FCF_n × (1 + tgr) / (WACC − tgr)
      Exit EV/EBITDA       : TV = EBITDA_n × exit_multiple
  - SensitivityTable axes default to WACC ± 200 bps and tgr ± 100 bps.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TerminalValueMethod(str, Enum):
    GORDON_GROWTH = "gordon_growth"
    EXIT_MULTIPLE = "exit_multiple"


# ---------------------------------------------------------------------------
# WACC components
# ---------------------------------------------------------------------------

class WACCInputs(BaseModel):
    """
    Inputs required to compute WACC via CAPM.

    Cost of equity  = risk_free_rate + beta × equity_risk_premium + size_premium
    After-tax Kd    = cost_of_debt × (1 − tax_rate)
    WACC            = Ke × equity_weight + Kd_at × debt_weight

    The WACCDeriver in core/wacc_deriver.py populates these fields from live
    market data when assumptions are auto-derived. When passing WACCInputs
    manually, the caller is responsible for providing correct values.
    """

    risk_free_rate: float = Field(0.045, description="10-yr gov't bond yield (decimal). Auto-derived from ^TNX.")
    beta: float = Field(1.0, description="Blume-adjusted levered equity beta")
    raw_beta: Optional[float] = Field(None, description="Unadjusted Yahoo Finance beta (before Blume's)")
    equity_risk_premium: float = Field(
        0.055,
        description=(
            "Equity Risk Premium (decimal). Use Damodaran's implied ERP "
            "(pages.stern.nyu.edu/~adamodar/). Update DEFAULT_ERP in "
            "data/fetchers/market_rates.py when market conditions change."
        ),
    )
    size_premium: float = Field(
        0.0,
        description=(
            "Additional return premium for smaller companies (decimal). "
            "Applied to cost of equity on top of CAPM. "
            "Approximates Kroll/Duff & Phelps size premia tiers."
        ),
    )
    cost_of_debt: float = Field(
        0.05,
        description=(
            "Marginal pre-tax cost of debt (decimal). "
            "Auto-derived via synthetic credit rating (ICR → Damodaran default spread)."
        ),
    )
    synthetic_rating: Optional[str] = Field(
        None,
        description="Synthetic bond rating derived from Interest Coverage Ratio (e.g. 'A-', 'BB+').",
    )
    tax_rate: float = Field(0.21, description="Marginal corporate tax rate (decimal)")
    debt_weight: float = Field(0.20, description="Debt / (Debt + Equity) at market value")

    @property
    def equity_weight(self) -> float:
        return 1.0 - self.debt_weight

    @property
    def cost_of_equity(self) -> float:
        return self.risk_free_rate + self.beta * self.equity_risk_premium + self.size_premium

    @property
    def after_tax_cost_of_debt(self) -> float:
        return self.cost_of_debt * (1.0 - self.tax_rate)


class WACCResult(BaseModel):
    """Computed WACC with full component breakdown for transparency."""

    wacc: float
    cost_of_equity: float
    after_tax_cost_of_debt: float
    equity_weight: float
    debt_weight: float
    beta: float                               # Blume-adjusted beta used in calculation
    raw_beta: Optional[float] = None          # Yahoo Finance beta before Blume's
    risk_free_rate: float
    equity_risk_premium: float
    size_premium: float = 0.0                 # small-cap premium added to Ke
    pre_tax_cost_of_debt: float
    synthetic_rating: Optional[str] = None   # e.g. "A-" derived from ICR
    tax_rate: float


# ---------------------------------------------------------------------------
# DCF Assumptions (full input set)
# ---------------------------------------------------------------------------

class DCFAssumptions(BaseModel):
    """
    Complete assumption set for a single DCF run.

    Minimal required inputs for a usable DCF:
      - revenue_growth_rates (or a single float)
      - ebitda_margin
      - wacc_override OR wacc_inputs
      - terminal_growth_rate (if using gordon_growth)

    All other fields default to reasonable values that the engine will
    use and surface in the report so assumptions are always auditable.
    """

    # --- Projection horizon ---
    projection_years: int = Field(5, ge=1, le=20, description="Explicit forecast period in years")

    # --- Revenue growth ---
    # Either a single flat rate or a list of per-year rates
    revenue_growth_rates: list[float] = Field(
        default_factory=lambda: [0.08] * 5,
        description="Per-year revenue growth rates. Length must equal projection_years.",
    )

    # --- Margin assumptions ---
    ebitda_margin: float = Field(0.25, description="Steady-state EBITDA margin (decimal)")
    da_as_pct_revenue: float = Field(0.04, description="D&A as % of revenue (decimal)")
    tax_rate: float = Field(0.21, description="Effective tax rate applied to EBIT (decimal)")

    # --- Reinvestment assumptions ---
    capex_as_pct_revenue: float = Field(0.05, description="CapEx as % of revenue (decimal)")
    nwc_change_as_pct_revenue_delta: float = Field(
        0.01,
        description="Incremental NWC required per unit of revenue growth (decimal). "
                    "Applied to the revenue increment each year.",
    )

    # --- WACC ---
    wacc_override: Optional[float] = Field(
        None,
        description="If set, bypasses WACCInputs and uses this WACC directly (decimal).",
    )
    wacc_inputs: Optional[WACCInputs] = Field(
        default_factory=WACCInputs,
        description="CAPM-based WACC components. Ignored if wacc_override is set.",
    )

    # --- Terminal value ---
    terminal_value_method: TerminalValueMethod = TerminalValueMethod.GORDON_GROWTH
    terminal_growth_rate: float = Field(
        0.025,
        description="Perpetuity growth rate for Gordon Growth terminal value (decimal).",
    )
    exit_ev_ebitda_multiple: Optional[float] = Field(
        None,
        description="Exit EV/EBITDA multiple for terminal value. Required if method=exit_multiple.",
    )

    # --- Balance sheet bridge (EV → equity value) ---
    net_debt: Optional[float] = Field(
        None,
        description="Net debt in millions USD (debt − cash). Pulled from latest snapshot if None.",
    )
    minority_interest: float = Field(0.0, description="Minority interest in millions USD.")
    shares_outstanding: Optional[float] = Field(
        None,
        description="Diluted shares outstanding in millions. Pulled from latest snapshot if None.",
    )

    @model_validator(mode="after")
    def validate_growth_rates_length(self) -> "DCFAssumptions":
        if len(self.revenue_growth_rates) != self.projection_years:
            # Auto-extend or truncate to match projection_years
            rates = self.revenue_growth_rates
            n = self.projection_years
            if len(rates) < n:
                rates = rates + [rates[-1]] * (n - len(rates))
            else:
                rates = rates[:n]
            self.revenue_growth_rates = rates
        return self

    @model_validator(mode="after")
    def validate_terminal_value_inputs(self) -> "DCFAssumptions":
        if (
            self.terminal_value_method == TerminalValueMethod.EXIT_MULTIPLE
            and self.exit_ev_ebitda_multiple is None
        ):
            raise ValueError(
                "exit_ev_ebitda_multiple must be provided when terminal_value_method='exit_multiple'"
            )
        return self


# ---------------------------------------------------------------------------
# Projection outputs
# ---------------------------------------------------------------------------

class ProjectedYear(BaseModel):
    """
    Full P&L and FCF build-up for a single projected year.

    All values in millions USD. Stored for transparency so the caller
    can inspect every line of the FCF bridge, not just the final FCF.
    """

    year: int                           # 1-indexed relative to base year
    calendar_year: Optional[int] = None

    # Revenue build
    revenue: float
    revenue_growth_rate: float

    # EBITDA → EBIT
    ebitda: float
    ebitda_margin: float
    depreciation_amortization: float
    ebit: float
    ebit_margin: float

    # NOPAT
    tax_rate: float
    nopat: float                        # EBIT × (1 − tax_rate)

    # Reinvestment
    capital_expenditures: float         # stored as positive outflow
    change_in_nwc: float                # positive = NWC increased (cash outflow)
    net_investment: float               # capex − D&A + ΔNWC

    # FCF
    free_cash_flow: float               # NOPAT − net_investment
    discount_factor: float              # 1 / (1 + WACC)^year
    pv_free_cash_flow: float            # FCF × discount_factor


class TerminalValueResult(BaseModel):
    """Terminal value calculation detail."""

    method: TerminalValueMethod
    terminal_year_ebitda: float
    terminal_year_fcf: float
    terminal_growth_rate: Optional[float] = None
    exit_multiple: Optional[float] = None
    terminal_value: float               # undiscounted
    pv_terminal_value: float            # discounted back to today
    pv_as_pct_of_ev: Optional[float] = None   # filled after EV is known


class DCFBridge(BaseModel):
    """
    Enterprise Value → Equity Value → Implied Share Price bridge.

    EV = PV(explicit FCFs) + PV(terminal value)
    Equity Value = EV − net_debt − minority_interest
    Implied Price = Equity Value / shares_outstanding
    """

    pv_explicit_fcfs: float
    pv_terminal_value: float
    enterprise_value: float
    net_debt: float
    minority_interest: float
    equity_value: float
    shares_outstanding: float
    implied_share_price: float
    current_price: Optional[float] = None
    upside_downside_pct: Optional[float] = None   # (implied - current) / current


# ---------------------------------------------------------------------------
# Sensitivity table
# ---------------------------------------------------------------------------

class SensitivityTable(BaseModel):
    """
    2-D grid of implied share prices across WACC and terminal growth rate (or exit multiple).

    row_axis:    list of WACC values tested (decimals)
    col_axis:    list of tgr or exit multiple values tested
    col_label:   "Terminal Growth Rate" | "Exit EV/EBITDA"
    prices:      prices[i][j] = implied price at row_axis[i], col_axis[j]
    """

    row_label: str = "WACC"
    col_label: str = "Terminal Growth Rate"
    row_axis: list[float]
    col_axis: list[float]
    prices: list[list[Optional[float]]]   # [row][col], None if WACC ≤ tgr (invalid)


# ---------------------------------------------------------------------------
# Full DCF result
# ---------------------------------------------------------------------------

class DCFResult(BaseModel):
    """
    Complete output of one DCF run.

    Contains every intermediate and final value so the report generator
    and scenario analysis module can consume any part of it.
    """

    ticker: str
    base_year: str                          # period of the last actual snapshot used
    assumptions: DCFAssumptions
    wacc_result: WACCResult
    projected_years: list[ProjectedYear]
    terminal_value_result: TerminalValueResult
    bridge: DCFBridge
    sensitivity: SensitivityTable
