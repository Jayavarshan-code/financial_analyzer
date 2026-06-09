"""
data/fetchers/yfinance_fetcher.py
----------------------------------
Fetches and normalizes financial statement data from Yahoo Finance via yfinance.

Responsibilities:
  1. Pull company profile, income statement, balance sheet, and cash flow
     for a given ticker using the yfinance library.
  2. Normalize raw DataFrame column names and values into our typed Pydantic
     models (IncomeStatement, BalanceSheet, CashFlowStatement, CompanyProfile).
  3. Align all three statements by period so each FinancialSnapshot contains
     matched annual data.
  4. Compute balance-sheet-level derived fields (net_debt, total_debt) that
     yfinance does not provide directly.
  5. Normalize operating lease obligations across all sectors (ASC 842 / IFRS 16)
     so that companies with heavy operating leases (retail, airlines) are
     comparable to capital-light tech companies.

Known yfinance limitations (document here so callers are aware):
  - Historical depth is typically 4 annual periods.
  - Column names returned by yfinance shift across library versions.
    All mappings are ordered alias lists: the FIRST matching alias wins so
    you can add new yfinance names at the front without touching existing code.
  - Values are returned in raw units (not millions). This fetcher converts
    everything to millions USD before returning.
  - Rate limits: yfinance uses undocumented Yahoo Finance endpoints and can
    get throttled. Retry logic is left to the caller (CLI has --verbose to
    surface errors).

Usage:
    from data.fetchers.yfinance_fetcher import YFinanceFetcher

    fetcher = YFinanceFetcher()
    history = fetcher.fetch("AAPL")   # returns FullFinancialHistory
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from data.models.financials import (
    BalanceSheet,
    CashFlowStatement,
    CompanyProfile,
    FinancialSnapshot,
    FullFinancialHistory,
    IncomeStatement,
    RawStatements,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field name mappings: ordered list of (yfinance_label, model_field) tuples.
#
# Rules:
#   - First matching alias wins — put the most current yfinance names first.
#   - A model_field that already has a value is NOT overwritten (first-match).
#   - The model_field must exactly match the Pydantic model's field name.
#   - Keep ALL historical aliases so the fetcher continues to work across
#     yfinance version upgrades without silent data loss.
# ---------------------------------------------------------------------------

# Income Statement aliases
_IS_ALIASES: list[tuple[str, str]] = [
    # Revenue (several yfinance names depending on sector/version)
    ("Total Revenue",                         "revenue"),
    ("Revenue",                               "revenue"),
    ("Operating Revenue",                     "revenue"),   # banks, REITs
    ("Gross Revenue",                         "revenue"),

    # Cost structure
    ("Cost Of Revenue",                       "cost_of_revenue"),
    ("Cost of Goods Sold",                    "cost_of_revenue"),
    ("Gross Profit",                          "gross_profit"),

    # Operating expenses
    ("Research And Development",              "research_and_development"),
    ("Research Development",                  "research_and_development"),
    ("Selling General Administrative",        "selling_general_administrative"),
    ("Selling General And Administrative",    "selling_general_administrative"),
    ("Total Operating Expenses",              "other_operating_expenses"),

    # EBIT / Operating Income
    ("Operating Income",                      "operating_income"),
    ("Total Operating Income As Reported",    "operating_income"),

    # Below the line
    ("Interest Expense",                      "interest_expense"),
    ("Interest Expense Non Operating",        "interest_expense"),
    ("Interest Income",                       "interest_income"),
    ("Net Interest Income",                   "interest_income"),
    ("Other Income Expense Net",              "other_non_operating"),
    ("Other Non Operating Income Expenses",   "other_non_operating"),
    ("Income Before Tax",                     "pretax_income"),
    ("Pretax Income",                         "pretax_income"),
    ("Income Tax Expense",                    "income_tax"),
    ("Tax Provision",                         "income_tax"),
    ("Net Income",                            "net_income"),
    ("Net Income Common Stockholders",        "net_income"),

    # Per-share / share counts
    ("Diluted EPS",                           "eps_diluted"),
    ("Basic EPS",                             "eps_basic"),
    ("Diluted Average Shares",                "shares_diluted"),
    ("Basic Average Shares",                  "shares_basic"),

    # EBITDA — prefer reported; fallback computed in _derive_statement_fields
    ("EBITDA",                                "ebitda"),
    ("Normalized EBITDA",                     "ebitda"),
    ("Adjusted EBITDA",                       "ebitda"),

    # D&A — prefer income statement; fallback from CF in _derive_statement_fields
    ("Reconciled Depreciation",               "depreciation_amortization"),
    ("Depreciation And Amortization",         "depreciation_amortization"),
    ("Depreciation",                          "depreciation_amortization"),
    ("Amortization",                          "depreciation_amortization"),
]

# Balance Sheet aliases
_BS_ALIASES: list[tuple[str, str]] = [
    # Cash
    ("Cash And Cash Equivalents",             "cash_and_equivalents"),
    ("Cash Cash Equivalents And Short Term Investments", "cash_and_equivalents"),
    ("Cash And Short Term Investments",       "cash_and_equivalents"),

    # Current assets
    ("Other Short Term Investments",          "short_term_investments"),
    ("Net Receivables",                       "accounts_receivable"),
    ("Receivables",                           "accounts_receivable"),
    ("Other Receivables",                     "accounts_receivable"),
    ("Inventory",                             "inventory"),
    ("Other Current Assets",                  "other_current_assets"),
    ("Total Current Assets",                  "total_current_assets"),

    # Non-current assets
    ("Net PPE",                               "property_plant_equipment_net"),
    ("Net Property Plant And Equipment",      "property_plant_equipment_net"),
    ("Goodwill",                              "goodwill"),
    ("Goodwill And Other Intangible Assets",  "intangible_assets"),
    ("Intangible Assets",                     "intangible_assets"),
    ("Long Term Equity Investment",           "long_term_investments"),
    ("Other Non Current Assets",              "other_non_current_assets"),
    ("Total Non Current Assets",              "total_non_current_assets"),
    ("Total Assets",                          "total_assets"),

    # Current liabilities
    ("Accounts Payable",                      "accounts_payable"),
    ("Current Debt",                          "short_term_debt"),
    ("Short Term Debt",                       "short_term_debt"),
    # When yfinance bundles current lease obligations into short-term debt:
    ("Current Debt And Capital Lease Obligation", "short_term_debt"),
    ("Deferred Revenue",                      "deferred_revenue_current"),
    ("Other Current Liabilities",             "other_current_liabilities"),
    ("Total Current Liabilities",             "total_current_liabilities"),

    # Non-current liabilities
    ("Long Term Debt",                        "long_term_debt"),
    # When yfinance bundles long-term lease obligations into long-term debt:
    ("Long Term Debt And Capital Lease Obligation", "long_term_debt"),
    ("Deferred Tax Liabilities Gross",        "deferred_tax_liabilities"),
    ("Deferred Tax Liabilities",              "deferred_tax_liabilities"),
    ("Other Non Current Liabilities",         "other_non_current_liabilities"),
    ("Total Non Current Liabilities Net Minority Interest", "total_non_current_liabilities"),
    ("Total Non Current Liabilities",         "total_non_current_liabilities"),
    ("Total Liabilities Net Minority Interest", "total_liabilities"),
    ("Total Liabilities",                     "total_liabilities"),

    # -----------------------------------------------------------------------
    # Operating lease liabilities (ASC 842 / IFRS 16)
    # When yfinance separates operating leases from financial debt, we capture
    # them in dedicated fields so _derive_balance_fields can fold them in
    # consistently across all tickers.
    # -----------------------------------------------------------------------
    ("Long Term Lease Obligation",            "operating_lease_liability_lt"),
    ("Operating Lease Long Term Obligation",  "operating_lease_liability_lt"),
    ("Operating Lease Liabilities",           "operating_lease_liability_lt"),
    ("Current Operating Lease",               "operating_lease_liability_current"),
    ("Operating Lease Current Obligation",    "operating_lease_liability_current"),
    ("Operating Lease Liability Current",     "operating_lease_liability_current"),

    # Equity
    ("Common Stock",                          "common_stock"),
    ("Retained Earnings",                     "retained_earnings"),
    ("Accumulated Other Comprehensive Income Loss", "accumulated_other_comprehensive_income"),
    ("Total Equity Gross Minority Interest",  "total_stockholders_equity"),
    ("Stockholders Equity",                   "total_stockholders_equity"),
    ("Total Stockholders Equity",             "total_stockholders_equity"),
]

# Cash Flow Statement aliases
_CF_ALIASES: list[tuple[str, str]] = [
    ("Net Income From Continuing Operations", "net_income_cf"),
    ("Net Income",                            "net_income_cf"),
    ("Depreciation And Amortization",         "depreciation_amortization_cf"),
    ("Depreciation Amortization Depletion",   "depreciation_amortization_cf"),
    ("Reconciled Depreciation",               "depreciation_amortization_cf"),
    ("Depreciation",                          "depreciation_amortization_cf"),
    ("Stock Based Compensation",              "stock_based_compensation"),
    ("Change In Working Capital",             "change_in_working_capital"),
    ("Other Non Cash Items",                  "other_operating_activities"),
    ("Operating Cash Flow",                   "operating_cash_flow"),
    ("Cash Flow From Continuing Operating Activities", "operating_cash_flow"),
    ("Capital Expenditure",                   "capital_expenditures"),
    ("Capital Expenditures",                  "capital_expenditures"),
    ("Purchase Of Business",                  "acquisitions"),
    ("Purchase Of Investment",                "purchases_of_investments"),
    ("Sale Of Investment",                    "sales_of_investments"),
    ("Other Investing Activities",            "other_investing_activities"),
    ("Investing Cash Flow",                   "investing_cash_flow"),
    ("Cash Flow From Continuing Investing Activities", "investing_cash_flow"),
    ("Issuance Of Debt",                      "debt_issuance"),
    ("Repayment Of Debt",                     "debt_repayment"),
    ("Common Stock Issuance",                 "common_stock_issued"),
    ("Repurchase Of Capital Stock",           "common_stock_repurchased"),
    ("Cash Dividends Paid",                   "dividends_paid"),
    ("Other Financing Activities",            "other_financing_activities"),
    ("Financing Cash Flow",                   "financing_cash_flow"),
    ("Cash Flow From Continuing Financing Activities", "financing_cash_flow"),
    ("Changes In Cash",                       "net_change_in_cash"),
    ("Free Cash Flow",                        "free_cash_flow"),
]


def _to_millions(value: Any) -> Optional[float]:
    """Convert a raw yfinance numeric value (in absolute units) to millions."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value) / 1_000_000
    except (TypeError, ValueError):
        return None


def _extract_row(
    df: pd.DataFrame,
    aliases: list[tuple[str, str]],
    period_col: Any,
) -> dict[str, Any]:
    """
    Pull one period's worth of data from a yfinance statement DataFrame.

    yfinance DataFrames have rows = line items, columns = period dates.
    We iterate the alias list in order; the FIRST matching alias wins for each
    model field (so new yfinance names placed earlier take priority).
    """
    result: dict[str, Any] = {}
    for yf_label, model_field in aliases:
        if yf_label in df.index and model_field not in result:
            raw = df.loc[yf_label, period_col]
            result[model_field] = _to_millions(raw)
    return result


class YFinanceFetcher:
    """
    Fetches financial history for a single ticker from Yahoo Finance.

    The main entry point is fetch(ticker) which returns a FullFinancialHistory.
    All internal methods are prefixed with _ and handle one statement type each.
    """

    def fetch(self, ticker: str) -> FullFinancialHistory:
        """
        Fetch complete financial history for the given ticker.

        Returns FullFinancialHistory with:
          - CompanyProfile (metadata + current market data)
          - Up to 4 annual FinancialSnapshots (oldest -> newest)
          - data_warnings: list of derivation/quality issues detected

        Raises:
            ValueError: if the ticker is not found or returns no data.
        """
        ticker = ticker.upper().strip()
        logger.info("Fetching data for %s", ticker)

        yf_ticker = yf.Ticker(ticker)
        profile = self._build_profile(ticker, yf_ticker)

        income_df = self._safe_fetch(yf_ticker, "income_stmt")
        balance_df = self._safe_fetch(yf_ticker, "balance_sheet")
        cashflow_df = self._safe_fetch(yf_ticker, "cashflow")

        if income_df is None or income_df.empty:
            raise ValueError(f"No financial data returned for ticker '{ticker}'")

        snapshots, warnings = self._build_annual_snapshots(income_df, balance_df, cashflow_df)

        if not snapshots:
            raise ValueError(f"Could not parse any annual periods for '{ticker}'")

        logger.info("Fetched %d annual periods for %s", len(snapshots), ticker)
        if warnings:
            logger.debug("Data warnings for %s: %s", ticker, "; ".join(warnings))

        # TTM — best-effort; failure must never block annual data
        ttm_snapshot = None
        try:
            ttm_snapshot = self._build_ttm_snapshot(yf_ticker)
            if ttm_snapshot:
                logger.info("TTM snapshot built for %s (as of %s)", ticker, ttm_snapshot.period)
        except Exception as exc:
            logger.warning("TTM snapshot failed for %s: %s", ticker, exc)

        return FullFinancialHistory(
            profile=profile,
            annual_snapshots=snapshots,
            ttm_snapshot=ttm_snapshot,
            data_warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_fetch(self, yf_ticker: yf.Ticker, attr: str) -> Optional[pd.DataFrame]:
        """Fetch a statement attribute, returning None on any error."""
        try:
            df = getattr(yf_ticker, attr)
            return df if df is not None and not df.empty else None
        except Exception as exc:
            logger.warning("Could not fetch '%s': %s", attr, exc)
            return None

    def _build_profile(self, ticker: str, yf_ticker: yf.Ticker) -> CompanyProfile:
        """Build CompanyProfile from yfinance .info dict."""
        info: dict = {}
        try:
            info = yf_ticker.info or {}
        except Exception as exc:
            logger.warning("Could not fetch .info for %s: %s", ticker, exc)

        def get(key: str) -> Any:
            return info.get(key)

        shares_out = _to_millions(get("sharesOutstanding"))
        float_shares = _to_millions(get("floatShares"))
        market_cap = _to_millions(get("marketCap"))
        ev = _to_millions(get("enterpriseValue"))

        return CompanyProfile(
            ticker=ticker,
            name=get("longName") or get("shortName") or ticker,
            sector=get("sector"),
            industry=get("industry"),
            exchange=get("exchange"),
            currency=get("currency") or "USD",
            country=get("country"),
            description=get("longBusinessSummary"),
            shares_outstanding=shares_out,
            float_shares=float_shares,
            current_price=get("currentPrice") or get("regularMarketPrice"),
            market_cap=market_cap,
            enterprise_value=ev,
            beta=get("beta"),
            pe_ratio_ttm=get("trailingPE"),
            ps_ratio_ttm=get("priceToSalesTrailing12Months"),
            pb_ratio=get("priceToBook"),
            ev_ebitda_ttm=get("enterpriseToEbitda"),
        )

    def _build_annual_snapshots(
        self,
        income_df: pd.DataFrame,
        balance_df: Optional[pd.DataFrame],
        cashflow_df: Optional[pd.DataFrame],
    ) -> tuple[list[FinancialSnapshot], list[str]]:
        """
        Align the three statement DataFrames by their shared period columns and
        produce one FinancialSnapshot per period.

        Returns (snapshots, accumulated_warnings) where warnings are deduplicated
        across all periods.
        """
        snapshots: list[FinancialSnapshot] = []
        all_warnings: list[str] = []
        seen_warnings: set[str] = set()

        periods = sorted(income_df.columns.tolist())

        for period_ts in periods:
            period_str = pd.Timestamp(period_ts).strftime("%Y-%m-%d")
            period_warnings: list[str] = []

            # --- Income Statement ---
            is_data = _extract_row(income_df, _IS_ALIASES, period_ts)

            # --- Cash Flow (needed for derivation before building CF model) ---
            cf_data: dict[str, Any] = {}
            if cashflow_df is not None and period_ts in cashflow_df.columns:
                cf_data = _extract_row(cashflow_df, _CF_ALIASES, period_ts)

            # Derive missing IS fields from components
            is_data, is_warnings = self._derive_statement_fields(is_data, cf_data)
            period_warnings.extend(is_warnings)

            income = IncomeStatement(period=period_str, **is_data)

            # --- Balance Sheet ---
            balance = BalanceSheet(period=period_str)
            if balance_df is not None and period_ts in balance_df.columns:
                bs_data = _extract_row(balance_df, _BS_ALIASES, period_ts)
                balance_raw = BalanceSheet(period=period_str, **bs_data)
                balance, bs_warnings = self._derive_balance_fields(balance_raw)
                period_warnings.extend(bs_warnings)

            # --- Cash Flow ---
            cashflow = CashFlowStatement(period=period_str)
            if cf_data:
                cashflow = CashFlowStatement(period=period_str, **cf_data)
                cashflow = self._derive_cashflow_fields(cashflow, income)

            # Cross-statement consistency check
            cs_warnings = self._cross_check(income, cashflow, period_str)
            period_warnings.extend(cs_warnings)

            raw = RawStatements(
                income_statement=income,
                balance_sheet=balance,
                cash_flow_statement=cashflow,
            )

            from data.models.financials import FinancialRatios
            snapshot = FinancialSnapshot(
                period=period_str,
                statements=raw,
                ratios=FinancialRatios(period=period_str),
            )
            snapshots.append(snapshot)

            # Deduplicate warnings (same derivation note across periods → log once)
            for w in period_warnings:
                if w not in seen_warnings:
                    all_warnings.append(w)
                    seen_warnings.add(w)

        return snapshots, all_warnings

    def _build_ttm_snapshot(self, yf_ticker: yf.Ticker) -> Optional["FinancialSnapshot"]:
        """
        Build a TTM FinancialSnapshot by summing the last 4 quarterly periods.

        Rules:
          - Income statement + cash flow: SUM of last 4 quarters (flow statements).
          - Balance sheet: most recent quarter only (point-in-time stock).
          - EPS / share counts: NOT summed. Use latest quarter's shares; recompute
            TTM EPS as (summed net income) / (latest diluted shares).
          - Returns None when fewer than 4 quarters are available.
        """
        q_income = self._safe_fetch(yf_ticker, "quarterly_income_stmt")
        q_balance = self._safe_fetch(yf_ticker, "quarterly_balance_sheet")
        q_cashflow = self._safe_fetch(yf_ticker, "quarterly_cashflow")

        if q_income is None or q_income.empty or q_income.shape[1] < 4:
            logger.debug("Fewer than 4 quarters available — skipping TTM build")
            return None

        # Sort ascending (oldest first), take last 4
        all_periods = sorted(q_income.columns.tolist())
        ttm_periods = all_periods[-4:]
        latest_q = ttm_periods[-1]
        ttm_date = pd.Timestamp(latest_q).strftime("%Y-%m-%d")

        # ── Income Statement: sum 4 quarters ─────────────────────────────
        # Fields that must NOT be summed (point-in-time or per-share):
        _IS_NO_SUM = {"shares_diluted", "shares_basic", "eps_diluted", "eps_basic"}

        is_ttm: dict[str, Any] = {}
        for period_ts in ttm_periods:
            q_data = _extract_row(q_income, _IS_ALIASES, period_ts)
            for field, value in q_data.items():
                if field in _IS_NO_SUM or value is None:
                    continue
                is_ttm[field] = (is_ttm.get(field) or 0.0) + value

        # Shares: latest quarter only (not summed)
        latest_q_is = _extract_row(q_income, _IS_ALIASES, latest_q)
        for share_field in ("shares_diluted", "shares_basic"):
            v = latest_q_is.get(share_field)
            if v is not None:
                is_ttm[share_field] = v

        # TTM EPS = TTM net income / latest diluted shares
        ni_ttm = is_ttm.get("net_income")
        shares_ttm = is_ttm.get("shares_diluted")
        if ni_ttm is not None and shares_ttm and shares_ttm > 0:
            is_ttm["eps_diluted"] = ni_ttm / shares_ttm

        # ── Cash Flow: sum 4 quarters ─────────────────────────────────────
        cf_ttm: dict[str, Any] = {}
        if q_cashflow is not None and not q_cashflow.empty:
            for period_ts in ttm_periods:
                if period_ts in q_cashflow.columns:
                    q_data = _extract_row(q_cashflow, _CF_ALIASES, period_ts)
                    for field, value in q_data.items():
                        if value is None:
                            continue
                        cf_ttm[field] = (cf_ttm.get(field) or 0.0) + value

        # ── Derive missing IS fields (D&A, operating income, EBITDA) ─────
        is_ttm, _ = self._derive_statement_fields(is_ttm, cf_ttm)

        # ── Balance Sheet: latest quarter (point-in-time) ─────────────────
        bs_data: dict[str, Any] = {}
        if q_balance is not None and not q_balance.empty and latest_q in q_balance.columns:
            bs_data = _extract_row(q_balance, _BS_ALIASES, latest_q)

        from data.models.financials import FinancialRatios

        income = IncomeStatement(period=ttm_date, period_type="ttm", **is_ttm)

        bs_raw = BalanceSheet(period=ttm_date, period_type="ttm", **bs_data)
        balance, _ = self._derive_balance_fields(bs_raw)

        cashflow = CashFlowStatement(period=ttm_date, period_type="ttm")
        if cf_ttm:
            cashflow = CashFlowStatement(period=ttm_date, period_type="ttm", **cf_ttm)
            cashflow = self._derive_cashflow_fields(cashflow, income)

        return FinancialSnapshot(
            period=ttm_date,
            period_type="ttm",
            statements=RawStatements(
                income_statement=income,
                balance_sheet=balance,
                cash_flow_statement=cashflow,
            ),
            ratios=FinancialRatios(period=ttm_date),
        )

    def _derive_statement_fields(
        self,
        is_data: dict[str, Any],
        cf_data: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """
        Fill missing income statement fields by computing them from available data.

        Derivations attempted (in priority order):
          1. D&A: sourced from CF statement when IS D&A is absent.
          2. Operating income: gross_profit - r&d - sga when absent.
          3. EBITDA: operating_income + D&A when absent.

        Returns (enriched_is_data, warnings).
        """
        warnings: list[str] = []

        # Step 1: D&A from CF statement
        if is_data.get("depreciation_amortization") is None:
            da_cf = cf_data.get("depreciation_amortization_cf")
            if da_cf is not None:
                is_data["depreciation_amortization"] = abs(da_cf)
                warnings.append("D&A sourced from cash flow statement (not on income statement)")

        # Step 2: Operating income from gross profit components
        if is_data.get("operating_income") is None:
            gp = is_data.get("gross_profit")
            if gp is not None:
                rd = is_data.get("research_and_development") or 0.0
                sga = is_data.get("selling_general_administrative") or 0.0
                is_data["operating_income"] = gp - rd - sga
                warnings.append("Operating income derived: gross_profit - R&D - SG&A")

        # Step 3: EBITDA from operating income + D&A
        if is_data.get("ebitda") is None:
            op_inc = is_data.get("operating_income")
            da = is_data.get("depreciation_amortization")
            if op_inc is not None and da is not None:
                is_data["ebitda"] = op_inc + abs(da)
                warnings.append("EBITDA derived: operating_income + D&A")

        return is_data, warnings

    def _derive_balance_fields(
        self,
        bs: BalanceSheet,
    ) -> tuple[BalanceSheet, list[str]]:
        """
        Fill total_debt and net_debt from component fields; fold operating lease
        obligations in for cross-sector comparability.

        Operating lease normalization (ASC 842 / IFRS 16):
          A retailer with $10B of store operating leases has the same economic
          obligation as one that owns those stores on a mortgage. Without
          normalization, the retailer appears to have lower leverage — making
          EV/EBITDA, net-debt-to-EBITDA, and DCF net-debt adjustments wrong.
        """
        data = bs.model_dump()
        warnings: list[str] = []

        # Build financial debt from components when not reported directly
        if data.get("total_debt") is None:
            st = data.get("short_term_debt") or 0.0
            lt = data.get("long_term_debt") or 0.0
            data["total_debt"] = st + lt if (st or lt) else None

        # Fold operating lease obligations into total_debt
        ol_lt = data.get("operating_lease_liability_lt") or 0.0
        ol_curr = data.get("operating_lease_liability_current") or 0.0
        if ol_lt or ol_curr:
            data["total_debt"] = (data.get("total_debt") or 0.0) + ol_lt + ol_curr
            warnings.append(
                f"Operating lease obligations ({ol_lt + ol_curr:,.0f}M) "
                "capitalized and folded into total_debt (ASC 842 normalization)"
            )

        # Net debt
        if data.get("net_debt") is None and data.get("total_debt") is not None:
            cash = data.get("cash_and_equivalents") or 0.0
            si = data.get("short_term_investments") or 0.0
            data["net_debt"] = data["total_debt"] - cash - si

        return BalanceSheet(**data), warnings

    def _derive_cashflow_fields(
        self,
        cf: CashFlowStatement,
        income: IncomeStatement,
    ) -> CashFlowStatement:
        """Compute free_cash_flow and FCF/share when not directly provided."""
        data = cf.model_dump()

        if data.get("free_cash_flow") is None:
            ocf = data.get("operating_cash_flow")
            capex = data.get("capital_expenditures")
            if ocf is not None and capex is not None:
                data["free_cash_flow"] = ocf + capex  # capex is negative by convention

        if data.get("free_cash_flow_per_share") is None:
            fcf = data.get("free_cash_flow")
            shares = income.shares_diluted
            if fcf is not None and shares and shares > 0:
                data["free_cash_flow_per_share"] = fcf / shares

        return CashFlowStatement(**data)

    def _cross_check(
        self,
        income: IncomeStatement,
        cf: CashFlowStatement,
        period: str,
    ) -> list[str]:
        """
        Flag large discrepancies between matched line items across statements.

        A >15% gap between IS net income and CF net income signals that one of:
          - Extraordinary items hit CF but not IS (or vice versa)
          - The yfinance taxonomy categorized the lines differently
          - There is a legitimate non-cash adjustment we're not capturing

        These are warnings, not errors — the engine keeps running.
        """
        warnings: list[str] = []
        ni_is = income.net_income
        ni_cf = cf.net_income_cf
        if ni_is and ni_cf and ni_is != 0:
            gap = abs(ni_is - ni_cf) / abs(ni_is)
            if gap > 0.15:
                warnings.append(
                    f"{period}: IS net income ({ni_is:,.0f}M) differs from "
                    f"CF net income ({ni_cf:,.0f}M) by {gap:.0%} — "
                    "possible extraordinary items or taxonomy mismatch"
                )
        return warnings
