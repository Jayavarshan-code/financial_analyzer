"""
data/models/financials.py
--------------------------
Pydantic v2 data models for all financial statement data and derived metrics.

These models are the shared data contracts used across every module in the engine.
All monetary values are stored in millions USD unless noted otherwise.

Model hierarchy:
  RawStatements          – raw line items fetched from a data provider (3 statements)
    ├── IncomeStatement
    ├── BalanceSheet
    └── CashFlowStatement

  FinancialRatios        – computed from RawStatements by the analyzer
  FinancialSnapshot      – RawStatements + FinancialRatios for a single period
  CompanyProfile         – static company metadata (name, sector, shares outstanding)
  FullFinancialHistory   – ordered list of FinancialSnapshots across periods + profile

Validation rules:
  - All monetary fields default to None so partial data from providers is tolerated.
  - Ratios that cannot be computed (e.g. division by zero) are stored as None, not 0.
  - Periods are stored as ISO-8601 date strings ("2023-09-30") for portability.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Raw Statement Line Items
# ---------------------------------------------------------------------------

class IncomeStatement(BaseModel):
    """Annual or TTM income statement line items (values in millions USD)."""

    period: str = Field(..., description="Period end date, ISO-8601 (e.g. '2023-09-30')")
    period_type: str = Field("annual", description="'annual' | 'ttm' | 'quarterly'")

    # Top-line
    revenue: Optional[float] = None
    cost_of_revenue: Optional[float] = None
    gross_profit: Optional[float] = None

    # Operating
    research_and_development: Optional[float] = None
    selling_general_administrative: Optional[float] = None
    other_operating_expenses: Optional[float] = None
    operating_income: Optional[float] = None          # EBIT

    # Below the line
    interest_expense: Optional[float] = None
    interest_income: Optional[float] = None
    other_non_operating: Optional[float] = None
    pretax_income: Optional[float] = None
    income_tax: Optional[float] = None
    net_income: Optional[float] = None

    # Per-share & share counts
    eps_basic: Optional[float] = None
    eps_diluted: Optional[float] = None
    shares_basic: Optional[float] = None              # millions
    shares_diluted: Optional[float] = None            # millions

    # Derived but often reported directly
    ebitda: Optional[float] = None
    depreciation_amortization: Optional[float] = None


class BalanceSheet(BaseModel):
    """Annual or TTM balance sheet line items (values in millions USD)."""

    period: str
    period_type: str = "annual"

    # Current assets
    cash_and_equivalents: Optional[float] = None
    short_term_investments: Optional[float] = None
    accounts_receivable: Optional[float] = None
    inventory: Optional[float] = None
    other_current_assets: Optional[float] = None
    total_current_assets: Optional[float] = None

    # Non-current assets
    property_plant_equipment_net: Optional[float] = None
    goodwill: Optional[float] = None
    intangible_assets: Optional[float] = None
    long_term_investments: Optional[float] = None
    other_non_current_assets: Optional[float] = None
    total_non_current_assets: Optional[float] = None

    total_assets: Optional[float] = None

    # Current liabilities
    accounts_payable: Optional[float] = None
    short_term_debt: Optional[float] = None
    deferred_revenue_current: Optional[float] = None
    other_current_liabilities: Optional[float] = None
    total_current_liabilities: Optional[float] = None

    # Non-current liabilities
    long_term_debt: Optional[float] = None
    deferred_tax_liabilities: Optional[float] = None
    other_non_current_liabilities: Optional[float] = None
    total_non_current_liabilities: Optional[float] = None

    total_liabilities: Optional[float] = None

    # Equity
    common_stock: Optional[float] = None
    retained_earnings: Optional[float] = None
    accumulated_other_comprehensive_income: Optional[float] = None
    total_stockholders_equity: Optional[float] = None

    # Operating lease liabilities (ASC 842 / IFRS 16) — capitalized separately so the
    # fetcher can fold them into total_debt on a consistent basis across all sectors.
    # Without this, a retailer with heavy operating leases (e.g. Target) appears
    # asset-light vs. a tech company that owns its buildings.
    operating_lease_liability_lt: Optional[float] = None      # non-current operating leases
    operating_lease_liability_current: Optional[float] = None  # current portion

    # Derived
    total_debt: Optional[float] = None               # short_term_debt + long_term_debt + operating leases
    net_debt: Optional[float] = None                 # total_debt - cash_and_equivalents


class CashFlowStatement(BaseModel):
    """Annual or TTM cash flow statement line items (values in millions USD)."""

    period: str
    period_type: str = "annual"

    # Operating
    net_income_cf: Optional[float] = None            # net income as reported in CF (may differ)
    depreciation_amortization_cf: Optional[float] = None
    stock_based_compensation: Optional[float] = None
    change_in_working_capital: Optional[float] = None
    other_operating_activities: Optional[float] = None
    operating_cash_flow: Optional[float] = None      # CFO

    # Investing
    capital_expenditures: Optional[float] = None     # negative by convention
    acquisitions: Optional[float] = None
    purchases_of_investments: Optional[float] = None
    sales_of_investments: Optional[float] = None
    other_investing_activities: Optional[float] = None
    investing_cash_flow: Optional[float] = None

    # Financing
    debt_issuance: Optional[float] = None
    debt_repayment: Optional[float] = None
    common_stock_issued: Optional[float] = None
    common_stock_repurchased: Optional[float] = None
    dividends_paid: Optional[float] = None
    other_financing_activities: Optional[float] = None
    financing_cash_flow: Optional[float] = None

    # Summary
    net_change_in_cash: Optional[float] = None
    free_cash_flow: Optional[float] = None           # OCF + capex (capex is negative)
    free_cash_flow_per_share: Optional[float] = None


# ---------------------------------------------------------------------------
# Computed Financial Ratios
# ---------------------------------------------------------------------------

class ProfitabilityRatios(BaseModel):
    """Margins and returns on capital — how efficiently the company generates profit."""

    gross_margin: Optional[float] = None             # gross_profit / revenue
    operating_margin: Optional[float] = None         # EBIT / revenue
    net_margin: Optional[float] = None               # net_income / revenue
    ebitda_margin: Optional[float] = None            # EBITDA / revenue
    fcf_margin: Optional[float] = None               # FCF / revenue

    return_on_assets: Optional[float] = None         # net_income / total_assets
    return_on_equity: Optional[float] = None         # net_income / stockholders_equity
    return_on_invested_capital: Optional[float] = None  # NOPAT / invested_capital
    return_on_capital_employed: Optional[float] = None  # EBIT / (total_assets - current_liabilities)


class LiquidityRatios(BaseModel):
    """Short-term solvency — can the company meet its near-term obligations?"""

    current_ratio: Optional[float] = None            # current_assets / current_liabilities
    quick_ratio: Optional[float] = None              # (current_assets - inventory) / current_liabilities
    cash_ratio: Optional[float] = None               # cash / current_liabilities
    operating_cash_flow_ratio: Optional[float] = None  # OCF / current_liabilities


class LeverageRatios(BaseModel):
    """Debt burden and interest coverage — long-term solvency signals."""

    debt_to_equity: Optional[float] = None           # total_debt / stockholders_equity
    debt_to_assets: Optional[float] = None           # total_debt / total_assets
    net_debt_to_ebitda: Optional[float] = None       # net_debt / EBITDA
    interest_coverage: Optional[float] = None        # EBIT / interest_expense
    equity_multiplier: Optional[float] = None        # total_assets / stockholders_equity


class EfficiencyRatios(BaseModel):
    """Asset utilization — how productively the company turns assets into revenue."""

    asset_turnover: Optional[float] = None           # revenue / total_assets
    inventory_turnover: Optional[float] = None       # COGS / inventory
    receivables_turnover: Optional[float] = None     # revenue / accounts_receivable
    days_sales_outstanding: Optional[float] = None   # 365 / receivables_turnover
    days_inventory_outstanding: Optional[float] = None  # 365 / inventory_turnover
    days_payable_outstanding: Optional[float] = None    # (accounts_payable / COGS) * 365
    cash_conversion_cycle: Optional[float] = None    # DSO + DIO - DPO


class GrowthRates(BaseModel):
    """Year-over-year growth rates (expressed as decimals, e.g. 0.12 = 12%)."""

    revenue_growth: Optional[float] = None
    gross_profit_growth: Optional[float] = None
    operating_income_growth: Optional[float] = None
    net_income_growth: Optional[float] = None
    ebitda_growth: Optional[float] = None
    fcf_growth: Optional[float] = None
    eps_growth: Optional[float] = None


class FinancialRatios(BaseModel):
    """Aggregated computed ratios for a single period."""

    period: str
    profitability: ProfitabilityRatios = Field(default_factory=ProfitabilityRatios)
    liquidity: LiquidityRatios = Field(default_factory=LiquidityRatios)
    leverage: LeverageRatios = Field(default_factory=LeverageRatios)
    efficiency: EfficiencyRatios = Field(default_factory=EfficiencyRatios)
    growth: GrowthRates = Field(default_factory=GrowthRates)


# ---------------------------------------------------------------------------
# Aggregated Snapshot & History
# ---------------------------------------------------------------------------

class RawStatements(BaseModel):
    """The three financial statements for a single period."""

    income_statement: IncomeStatement
    balance_sheet: BalanceSheet
    cash_flow_statement: CashFlowStatement


class FinancialSnapshot(BaseModel):
    """
    A single-period slice: raw statements + derived ratios.

    This is the fundamental unit of analysis. The DCF engine, comps engine,
    and all analytics modules consume lists of these.
    """

    period: str
    period_type: str = "annual"
    currency: str = "USD"
    statements: RawStatements
    ratios: FinancialRatios


class CompanyProfile(BaseModel):
    """
    Static company metadata fetched once per analysis.

    Market-price-dependent fields (market_cap, price) are snapshot values
    at time of fetch — they are not stored per historical period.
    """

    ticker: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: Optional[str] = None
    currency: str = "USD"
    country: Optional[str] = None
    description: Optional[str] = None

    # Share data
    shares_outstanding: Optional[float] = None       # millions
    float_shares: Optional[float] = None             # millions

    # Current market data (point-in-time)
    current_price: Optional[float] = None
    market_cap: Optional[float] = None               # millions
    enterprise_value: Optional[float] = None         # millions
    beta: Optional[float] = None

    # Trailing multiples
    pe_ratio_ttm: Optional[float] = None
    ps_ratio_ttm: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda_ttm: Optional[float] = None

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()


class FullFinancialHistory(BaseModel):
    """
    Complete financial history for a company: profile + ordered annual snapshots.

    Snapshots are ordered oldest-first. The last element is the most recent period.
    TTM data (if available) is stored separately so it does not break the annual series.
    """

    profile: CompanyProfile
    annual_snapshots: list[FinancialSnapshot] = Field(default_factory=list)
    ttm_snapshot: Optional[FinancialSnapshot] = None
    # Warnings emitted by the fetcher when fields were derived rather than read directly,
    # or when cross-statement inconsistencies were detected.
    data_warnings: list[str] = Field(default_factory=list)

    @property
    def latest(self) -> Optional[FinancialSnapshot]:
        """Most recent annual snapshot."""
        return self.annual_snapshots[-1] if self.annual_snapshots else None

    @property
    def base_snapshot(self) -> Optional[FinancialSnapshot]:
        """
        Most current data point for DCF base-year revenue.

        Returns the TTM snapshot when available and more recent than the last
        annual 10-K — this is the correct base for projections, because the
        last annual filing can be up to 11 months old.  Falls back to the
        latest annual snapshot when TTM is absent or not more current.
        """
        if (
            self.ttm_snapshot is not None
            and self.latest is not None
            and self.ttm_snapshot.period > self.latest.period
        ):
            return self.ttm_snapshot
        return self.latest

    @property
    def periods(self) -> list[str]:
        return [s.period for s in self.annual_snapshots]
