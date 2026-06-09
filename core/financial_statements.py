"""
core/financial_statements.py
------------------------------
Financial Statement Analyzer — Phase 1, Module 1.

Responsibilities:
  1. Orchestrate data fetching for a given ticker via a pluggable fetcher
     (default: YFinanceFetcher).
  2. Compute all financial ratios (profitability, liquidity, leverage,
     efficiency, growth) for every historical period.
  3. Populate the FinancialRatios on each FinancialSnapshot in-place,
     replacing the stub stubs created by the fetcher.
  4. Expose a clean public API: analyze(ticker) → FullFinancialHistory.
  5. Provide a formatted summary table for quick CLI inspection.

Ratio computation logic:
  - All ratios guard against division-by-zero and missing fields, returning
    None rather than raising or returning 0.
  - ROIC = NOPAT / Invested Capital
      NOPAT  = EBIT × (1 − effective_tax_rate)
      Invested Capital = total_equity + total_debt − cash
  - Growth rates are YoY: (current − prior) / |prior|.
    Periods with a negative prior base are set to None (sign-flip cases are
    misleading, e.g. swinging from loss to profit).

Public API:
    analyzer = FinancialStatementAnalyzer()
    history  = analyzer.analyze("AAPL")
    print(analyzer.summary(history))

    # Access a specific period's ratios
    latest = history.latest
    print(latest.ratios.profitability.return_on_invested_capital)
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol

from data.fetchers.yfinance_fetcher import YFinanceFetcher
from data.models.financials import (
    BalanceSheet,
    CashFlowStatement,
    EfficiencyRatios,
    FinancialRatios,
    FinancialSnapshot,
    FullFinancialHistory,
    GrowthRates,
    IncomeStatement,
    LeverageRatios,
    LiquidityRatios,
    ProfitabilityRatios,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetcher protocol — allows swapping yfinance for FMP or a mock in tests
# ---------------------------------------------------------------------------

class FinancialDataFetcher(Protocol):
    def fetch(self, ticker: str) -> FullFinancialHistory: ...


# ---------------------------------------------------------------------------
# Safe arithmetic helpers
# ---------------------------------------------------------------------------

def _div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Safe division; returns None on zero denominator or missing values."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _growth(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    """
    YoY growth rate. Returns None when:
      - Either value is missing
      - Prior is zero (undefined growth)
      - Prior is negative (sign-flip makes % growth misleading)
    """
    if current is None or prior is None or prior <= 0:
        return None
    return (current - prior) / prior


# ---------------------------------------------------------------------------
# Ratio computers — one function per ratio group
# ---------------------------------------------------------------------------

def _compute_profitability(
    inc: IncomeStatement,
    bs: BalanceSheet,
    cf: CashFlowStatement,
) -> ProfitabilityRatios:
    """
    Compute margin and return ratios.

    ROIC methodology:
      effective_tax_rate = income_tax / pretax_income  (clipped to [0, 1])
      NOPAT              = operating_income × (1 − effective_tax_rate)
      invested_capital   = total_equity + total_debt − cash_and_equivalents
    """
    rev = inc.revenue

    # Effective tax rate for NOPAT
    tax_rate: Optional[float] = None
    if inc.income_tax is not None and inc.pretax_income and inc.pretax_income > 0:
        tax_rate = min(max(inc.income_tax / inc.pretax_income, 0.0), 1.0)

    nopat: Optional[float] = None
    if inc.operating_income is not None and tax_rate is not None:
        nopat = inc.operating_income * (1 - tax_rate)

    invested_capital: Optional[float] = None
    equity = bs.total_stockholders_equity
    debt = bs.total_debt
    cash = bs.cash_and_equivalents
    if equity is not None and debt is not None and cash is not None:
        invested_capital = equity + debt - cash

    roic = _div(nopat, invested_capital)

    # ROCE = EBIT / (total_assets - current_liabilities)
    roce: Optional[float] = None
    if inc.operating_income is not None and bs.total_assets is not None and bs.total_current_liabilities is not None:
        capital_employed = bs.total_assets - bs.total_current_liabilities
        roce = _div(inc.operating_income, capital_employed)

    # EBITDA: use reported value or reconstruct from EBIT + D&A
    ebitda = inc.ebitda
    if ebitda is None and inc.operating_income is not None and inc.depreciation_amortization is not None:
        ebitda = inc.operating_income + inc.depreciation_amortization

    return ProfitabilityRatios(
        gross_margin=_div(inc.gross_profit, rev),
        operating_margin=_div(inc.operating_income, rev),
        net_margin=_div(inc.net_income, rev),
        ebitda_margin=_div(ebitda, rev),
        fcf_margin=_div(cf.free_cash_flow, rev),
        return_on_assets=_div(inc.net_income, bs.total_assets),
        return_on_equity=_div(inc.net_income, bs.total_stockholders_equity),
        return_on_invested_capital=roic,
        return_on_capital_employed=roce,
    )


def _compute_liquidity(bs: BalanceSheet, cf: CashFlowStatement) -> LiquidityRatios:
    """
    Short-term solvency ratios.

    Quick ratio excludes inventory (less liquid current asset).
    Operating cash flow ratio uses actual cash generation vs. near-term obligations.
    """
    ca = bs.total_current_assets
    cl = bs.total_current_liabilities
    inv = bs.inventory or 0.0
    cash = bs.cash_and_equivalents

    quick_assets: Optional[float] = None
    if ca is not None:
        quick_assets = ca - inv

    return LiquidityRatios(
        current_ratio=_div(ca, cl),
        quick_ratio=_div(quick_assets, cl),
        cash_ratio=_div(cash, cl),
        operating_cash_flow_ratio=_div(cf.operating_cash_flow, cl),
    )


def _compute_leverage(
    inc: IncomeStatement,
    bs: BalanceSheet,
) -> LeverageRatios:
    """
    Debt burden and interest coverage ratios.

    net_debt_to_ebitda: preferred by credit analysts over gross debt/EBITDA.
    Interest coverage below 1.5× is considered distress territory.
    """
    ebitda = inc.ebitda
    if ebitda is None and inc.operating_income is not None and inc.depreciation_amortization is not None:
        ebitda = inc.operating_income + inc.depreciation_amortization

    # Interest expense stored as positive in our model (expense sign convention)
    int_exp = inc.interest_expense
    if int_exp is not None:
        int_exp = abs(int_exp)

    return LeverageRatios(
        debt_to_equity=_div(bs.total_debt, bs.total_stockholders_equity),
        debt_to_assets=_div(bs.total_debt, bs.total_assets),
        net_debt_to_ebitda=_div(bs.net_debt, ebitda),
        interest_coverage=_div(inc.operating_income, int_exp),
        equity_multiplier=_div(bs.total_assets, bs.total_stockholders_equity),
    )


def _compute_efficiency(
    inc: IncomeStatement,
    bs: BalanceSheet,
) -> EfficiencyRatios:
    """
    Asset utilization and working capital cycle ratios.

    Cash Conversion Cycle (CCC) = DSO + DIO − DPO
      Low CCC → company collects cash quickly and pays suppliers slowly (favorable).
      Negative CCC (e.g. Amazon) → suppliers finance operations.
    """
    cogs = inc.cost_of_revenue
    rev = inc.revenue

    recv_turn = _div(rev, bs.accounts_receivable)
    inv_turn = _div(cogs, bs.inventory)
    pay_turn = _div(cogs, bs.accounts_payable)

    dso = _div(365.0, recv_turn)
    dio = _div(365.0, inv_turn)
    dpo: Optional[float] = None
    if bs.accounts_payable is not None and cogs and cogs > 0:
        dpo = (bs.accounts_payable / cogs) * 365.0

    ccc: Optional[float] = None
    if dso is not None and dio is not None and dpo is not None:
        ccc = dso + dio - dpo

    return EfficiencyRatios(
        asset_turnover=_div(rev, bs.total_assets),
        inventory_turnover=inv_turn,
        receivables_turnover=recv_turn,
        days_sales_outstanding=dso,
        days_inventory_outstanding=dio,
        days_payable_outstanding=dpo,
        cash_conversion_cycle=ccc,
    )


def _compute_growth(
    current: FinancialSnapshot,
    prior: FinancialSnapshot,
) -> GrowthRates:
    """
    YoY growth rates between two consecutive annual snapshots.

    Called with (current_period, prior_period); prior must be exactly one
    year earlier for the rates to be meaningful.
    """
    ci = current.statements.income_statement
    pi = prior.statements.income_statement
    ccf = current.statements.cash_flow_statement
    pcf = prior.statements.cash_flow_statement

    c_ebitda = ci.ebitda
    p_ebitda = pi.ebitda

    return GrowthRates(
        revenue_growth=_growth(ci.revenue, pi.revenue),
        gross_profit_growth=_growth(ci.gross_profit, pi.gross_profit),
        operating_income_growth=_growth(ci.operating_income, pi.operating_income),
        net_income_growth=_growth(ci.net_income, pi.net_income),
        ebitda_growth=_growth(c_ebitda, p_ebitda),
        fcf_growth=_growth(ccf.free_cash_flow, pcf.free_cash_flow),
        eps_growth=_growth(ci.eps_diluted, pi.eps_diluted),
    )


# ---------------------------------------------------------------------------
# Main Analyzer class
# ---------------------------------------------------------------------------

class FinancialStatementAnalyzer:
    """
    Orchestrates data fetch + ratio computation for a single ticker.

    Inject a custom fetcher (must satisfy FinancialDataFetcher protocol) to
    use FMP, mock data, or cached responses instead of live yfinance calls.

        analyzer = FinancialStatementAnalyzer()
        history  = analyzer.analyze("MSFT")

        # With a custom fetcher (e.g. in tests):
        analyzer = FinancialStatementAnalyzer(fetcher=MockFetcher())
    """

    def __init__(self, fetcher: Optional[FinancialDataFetcher] = None) -> None:
        self._fetcher: FinancialDataFetcher = fetcher or YFinanceFetcher()

    def analyze(self, ticker: str) -> FullFinancialHistory:
        """
        Full pipeline: fetch → normalize → compute ratios → return history.

        Args:
            ticker: Stock ticker symbol (case-insensitive).

        Returns:
            FullFinancialHistory with all FinancialSnapshots populated with
            computed FinancialRatios.

        Raises:
            ValueError: if the ticker returns no usable data.
        """
        logger.info("Starting analysis for %s", ticker.upper())

        history = self._fetcher.fetch(ticker)
        self._enrich_ratios(history)

        logger.info(
            "Analysis complete for %s — %d periods",
            history.profile.ticker,
            len(history.annual_snapshots),
        )
        return history

    def _enrich_ratios(self, history: FullFinancialHistory) -> None:
        """
        Compute and attach FinancialRatios to every snapshot in-place.

        Growth rates require a prior-period snapshot, so the first period
        in the series always has empty GrowthRates.  The TTM snapshot (when
        present) is also enriched; its growth rates are computed against the
        most recent annual snapshot.
        """
        snapshots = history.annual_snapshots

        for i, snapshot in enumerate(snapshots):
            inc = snapshot.statements.income_statement
            bs = snapshot.statements.balance_sheet
            cf = snapshot.statements.cash_flow_statement

            profitability = _compute_profitability(inc, bs, cf)
            liquidity = _compute_liquidity(bs, cf)
            leverage = _compute_leverage(inc, bs)
            efficiency = _compute_efficiency(inc, bs)

            # Growth rates need a prior period
            if i > 0:
                growth = _compute_growth(snapshot, snapshots[i - 1])
            else:
                growth = GrowthRates()

            snapshot.ratios = FinancialRatios(
                period=snapshot.period,
                profitability=profitability,
                liquidity=liquidity,
                leverage=leverage,
                efficiency=efficiency,
                growth=growth,
            )

        # Enrich the TTM snapshot if present — compare against last annual
        if history.ttm_snapshot is not None:
            ttm = history.ttm_snapshot
            inc = ttm.statements.income_statement
            bs = ttm.statements.balance_sheet
            cf = ttm.statements.cash_flow_statement

            ttm_growth = GrowthRates()
            if snapshots:
                ttm_growth = _compute_growth(ttm, snapshots[-1])

            ttm.ratios = FinancialRatios(
                period=ttm.period,
                profitability=_compute_profitability(inc, bs, cf),
                liquidity=_compute_liquidity(bs, cf),
                leverage=_compute_leverage(inc, bs),
                efficiency=_compute_efficiency(inc, bs),
                growth=ttm_growth,
            )

    # ------------------------------------------------------------------
    # Summary / display helpers
    # ------------------------------------------------------------------

    def summary(self, history: FullFinancialHistory) -> str:
        """
        Return a formatted multi-line text summary of the analysis.

        Intended for quick CLI inspection — not for the full report (that
        is handled by core/report_generator.py in a later phase).
        """
        p = history.profile
        lines: list[str] = []

        lines.append("=" * 70)
        lines.append(f"  {p.name} ({p.ticker})  |  {p.sector or 'N/A'}  |  {p.exchange or 'N/A'}")
        lines.append("=" * 70)

        if p.current_price:
            lines.append(f"  Current Price : ${p.current_price:,.2f}")
        if p.market_cap:
            lines.append(f"  Market Cap    : ${p.market_cap:,.0f}M")
        if p.enterprise_value:
            lines.append(f"  EV            : ${p.enterprise_value:,.0f}M")
        if p.beta:
            lines.append(f"  Beta          : {p.beta:.2f}")

        lines.append("")
        lines.append(f"  {'Period':<12} {'Revenue':>10} {'Gross Mgn':>10} {'EBIT Mgn':>10} "
                     f"{'Net Mgn':>9} {'ROIC':>8} {'D/E':>8} {'FCF':>10}")
        lines.append("  " + "-" * 80)

        for snap in history.annual_snapshots:
            inc = snap.statements.income_statement
            rat = snap.ratios
            pro = rat.profitability
            lev = rat.leverage

            rev_str = f"${inc.revenue:,.0f}M" if inc.revenue else "N/A"
            gm_str = f"{pro.gross_margin * 100:.1f}%" if pro.gross_margin is not None else "N/A"
            om_str = f"{pro.operating_margin * 100:.1f}%" if pro.operating_margin is not None else "N/A"
            nm_str = f"{pro.net_margin * 100:.1f}%" if pro.net_margin is not None else "N/A"
            roic_str = f"{pro.return_on_invested_capital * 100:.1f}%" if pro.return_on_invested_capital is not None else "N/A"
            de_str = f"{lev.debt_to_equity:.2f}x" if lev.debt_to_equity is not None else "N/A"
            fcf_str = f"${snap.statements.cash_flow_statement.free_cash_flow:,.0f}M" \
                if snap.statements.cash_flow_statement.free_cash_flow else "N/A"

            lines.append(
                f"  {snap.period:<12} {rev_str:>10} {gm_str:>10} {om_str:>10} "
                f"{nm_str:>9} {roic_str:>8} {de_str:>8} {fcf_str:>10}"
            )

        lines.append("")

        # Growth rates for most recent period
        latest = history.latest
        if latest and latest.ratios.growth.revenue_growth is not None:
            g = latest.ratios.growth
            lines.append("  YoY Growth (most recent period):")
            lines.append(f"    Revenue   : {g.revenue_growth * 100:+.1f}%")
            if g.ebitda_growth is not None:
                lines.append(f"    EBITDA    : {g.ebitda_growth * 100:+.1f}%")
            if g.net_income_growth is not None:
                lines.append(f"    Net Income: {g.net_income_growth * 100:+.1f}%")
            if g.eps_growth is not None:
                lines.append(f"    EPS       : {g.eps_growth * 100:+.1f}%")
            if g.fcf_growth is not None:
                lines.append(f"    FCF       : {g.fcf_growth * 100:+.1f}%")

        lines.append("=" * 70)
        return "\n".join(lines)
