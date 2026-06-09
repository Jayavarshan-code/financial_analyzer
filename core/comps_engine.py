"""
core/comps_engine.py
----------------------
Comparable Valuation Engine — Phase 1, Module 3.

Pipeline:
  1. Fetch FullFinancialHistory for subject + all peer tickers via the
     FinancialStatementAnalyzer (pluggable via Protocol — MockFetcher in tests).
  2. Extract trailing multiples from each history.
     Priority: TTM snapshot if available, else latest annual snapshot.
     EV and market cap are taken from CompanyProfile (live market data);
     if missing, estimated from price x shares and net_debt from balance sheet.
  3. Compute cross-sectional statistics (mean / median / p25 / p75) across
     the peer set for each multiple. Peers missing or with a negative/zero
     denominator are excluded from that multiple's statistics without error.
  4. Back-solve implied share price for the subject using each peer-set median:
       EV-based  : Implied EV = median x subject_metric
                   Price = (Implied EV - net_debt) / shares
       Price-based: Price = median x subject_per_share_metric
  5. Render ASCII summary: peer table + stat rows + implied price waterfall.

Multiple definitions (all TTM):
  EV / EBITDA       - most common comparable; not distorted by D&A or capex policy
  EV / Revenue      - used for high-growth / pre-profitability companies
  Price / Earnings  - most-followed; distorted by leverage and non-cash items
  Price / FCF       - cleaner than P/E; market cap / free cash flow
  Price / Book      - useful for financials; price / book value of equity

Public API:
    from core.comps_engine import CompsEngine
    result = CompsEngine().run("AAPL", ["MSFT", "GOOGL", "META", "AMZN"])
    print(CompsEngine().summary(result))
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from data.models.comps import (
    CompsCompany,
    CompsMultiples,
    CompsResult,
    CompsStatistics,
    ImpliedValuation,
    MultipleStats,
)
from data.models.financials import FullFinancialHistory
from core.financial_statements import FinancialStatementAnalyzer


# ---------------------------------------------------------------------------
# Safe arithmetic helpers
# ---------------------------------------------------------------------------

def _div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    """Safe division — returns None when denominator is zero, None, or negative."""
    if num is None or den is None or den <= 0:
        return None
    return num / den


def _stats(values: list[Optional[float]], name: str) -> MultipleStats:
    """
    Compute descriptive statistics for one multiple across the peer set.

    Only positive, non-None values are included.
    """
    valid = np.array([v for v in values if v is not None and v > 0], dtype=float)
    n = len(valid)
    if n == 0:
        return MultipleStats(multiple_name=name, n_valid=0)

    return MultipleStats(
        multiple_name=name,
        n_valid=n,
        mean=float(np.mean(valid)),
        median=float(np.median(valid)),
        p25=float(np.percentile(valid, 25)),
        p75=float(np.percentile(valid, 75)),
        minimum=float(np.min(valid)),
        maximum=float(np.max(valid)),
    )


# ---------------------------------------------------------------------------
# Multiple extractor
# ---------------------------------------------------------------------------

def _extract_multiples(history: FullFinancialHistory) -> CompsMultiples:
    """
    Pull live market data and TTM (or latest annual) financials,
    then compute the 5 standard trading multiples.

    EV / market cap come from CompanyProfile (populated at fetch time from
    live market data). If missing, we estimate:
      market_cap = current_price x shares_outstanding
      enterprise_value = market_cap + net_debt
    """
    profile = history.profile
    snap = history.ttm_snapshot or history.latest
    if snap is None:
        return CompsMultiples(ticker=profile.ticker)

    inc = snap.statements.income_statement
    bs = snap.statements.balance_sheet
    cf = snap.statements.cash_flow_statement

    ev = profile.enterprise_value
    mc = profile.market_cap
    price = profile.current_price
    shares = profile.shares_outstanding       # millions

    # Estimate market cap from price x shares if not directly available
    if mc is None and price is not None and shares is not None:
        mc = price * shares

    # Estimate EV from market cap + net debt if not available
    if ev is None and mc is not None and bs.net_debt is not None:
        ev = mc + bs.net_debt

    revenue = inc.revenue
    ebitda = inc.ebitda
    eps = inc.eps_diluted
    fcf = cf.free_cash_flow
    book_eq = bs.total_stockholders_equity

    m = CompsMultiples(
        ticker=profile.ticker,
        enterprise_value=ev,
        market_cap=mc,
        revenue=revenue,
        ebitda=ebitda,
        net_income=inc.net_income,
        eps_diluted=eps,
        free_cash_flow=fcf,
        book_value_equity=book_eq,
        shares_outstanding=shares,
        current_price=price,
    )

    # Multiples: only set when denominator is strictly positive
    m.ev_to_ebitda = _div(ev, ebitda) if ebitda is not None and ebitda > 0 else None
    m.ev_to_revenue = _div(ev, revenue) if revenue is not None and revenue > 0 else None
    m.price_to_earnings = _div(price, eps) if eps is not None and eps > 0 else None
    m.price_to_fcf = _div(mc, fcf) if fcf is not None and fcf > 0 else None
    m.price_to_book = _div(mc, book_eq) if book_eq is not None and book_eq > 0 else None

    return m


# ---------------------------------------------------------------------------
# CompsEngine
# ---------------------------------------------------------------------------

class CompsEngine:
    """
    Comparable valuation engine.

    A custom fetcher (any object with a .fetch(ticker) method that returns
    FullFinancialHistory) can be injected for testing or alternative data sources.

    Usage:
        engine = CompsEngine()
        result = engine.run("AAPL", ["MSFT", "GOOGL", "META", "AMZN"])
        print(engine.summary(result))
    """

    def __init__(self, fetcher=None) -> None:
        self._fetcher = fetcher

    def _fetch(self, ticker: str) -> Optional[FullFinancialHistory]:
        """Fetch and enrich financial history; return None on any failure."""
        try:
            return FinancialStatementAnalyzer(fetcher=self._fetcher).analyze(ticker)
        except Exception:
            return None

    def run(
        self,
        subject_ticker: str,
        peer_tickers: list[str],
    ) -> CompsResult:
        """
        Build a comparable company analysis for subject_ticker vs peer_tickers.

        Peers that fail to fetch or return no data are silently dropped —
        the result will show fewer peers than requested.

        Args:
            subject_ticker: the company being valued (e.g. "AAPL")
            peer_tickers:   list of comparable company tickers (e.g. ["MSFT", "GOOGL"])

        Returns:
            CompsResult with multiples, statistics, and implied valuations.

        Raises:
            ValueError: if the subject ticker cannot be fetched.
        """
        subject_history = self._fetch(subject_ticker)
        if subject_history is None:
            raise ValueError(f"Could not fetch financial data for subject ticker: {subject_ticker}")

        subject_multiples = _extract_multiples(subject_history)
        subject_company = CompsCompany(
            ticker=subject_history.profile.ticker,
            name=subject_history.profile.name,
            sector=subject_history.profile.sector,
            industry=subject_history.profile.industry,
            multiples=subject_multiples,
        )

        peers: list[CompsCompany] = []
        for ticker in peer_tickers:
            h = self._fetch(ticker)
            if h is None:
                continue
            peers.append(CompsCompany(
                ticker=h.profile.ticker,
                name=h.profile.name,
                sector=h.profile.sector,
                industry=h.profile.industry,
                multiples=_extract_multiples(h),
            ))

        peers.sort(key=lambda c: c.ticker)
        stats = self._compute_statistics(peers)
        implied = self._compute_implied(subject_multiples, stats, subject_history)

        return CompsResult(
            subject=subject_company,
            peers=peers,
            stats=stats,
            implied=implied,
        )

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute_statistics(self, peers: list[CompsCompany]) -> CompsStatistics:
        return CompsStatistics(
            ev_to_ebitda=_stats(
                [p.multiples.ev_to_ebitda for p in peers], "EV/EBITDA"
            ),
            ev_to_revenue=_stats(
                [p.multiples.ev_to_revenue for p in peers], "EV/Revenue"
            ),
            price_to_earnings=_stats(
                [p.multiples.price_to_earnings for p in peers], "P/E"
            ),
            price_to_fcf=_stats(
                [p.multiples.price_to_fcf for p in peers], "P/FCF"
            ),
            price_to_book=_stats(
                [p.multiples.price_to_book for p in peers], "P/B"
            ),
        )

    def _compute_implied(
        self,
        subject: CompsMultiples,
        stats: CompsStatistics,
        history: FullFinancialHistory,
    ) -> ImpliedValuation:
        """
        Back-solve implied share price using each peer-set median multiple.

        EV-based approach:
          implied_ev    = median_multiple x subject_metric
          implied_price = (implied_ev - net_debt) / shares_outstanding
          Returns None if implied equity value (EV - net_debt) would be <= 0.

        Price-based approach:
          implied_price = median_multiple x subject_per_share_metric
        """
        snap = history.ttm_snapshot or history.latest
        bs = snap.statements.balance_sheet if snap else None
        net_debt = (bs.net_debt if bs else None) or 0.0
        shares = subject.shares_outstanding or history.profile.shares_outstanding

        def _ev_implied(median: Optional[float], metric: Optional[float]) -> Optional[float]:
            if median is None or metric is None or metric <= 0:
                return None
            if shares is None or shares <= 0:
                return None
            implied_ev = median * metric
            equity_val = implied_ev - net_debt
            return round(equity_val / shares, 2) if equity_val > 0 else None

        def _price_implied(median: Optional[float], per_share: Optional[float]) -> Optional[float]:
            if median is None or per_share is None or per_share <= 0:
                return None
            return round(median * per_share, 2)

        fcf_per_share = _div(subject.free_cash_flow, shares)
        bvps = _div(subject.book_value_equity, shares)

        return ImpliedValuation(
            implied_from_ev_ebitda=_ev_implied(stats.ev_to_ebitda.median, subject.ebitda),
            implied_from_ev_revenue=_ev_implied(stats.ev_to_revenue.median, subject.revenue),
            implied_from_pe=_price_implied(stats.price_to_earnings.median, subject.eps_diluted),
            implied_from_pfcf=_price_implied(stats.price_to_fcf.median, fcf_per_share),
            implied_from_pb=_price_implied(stats.price_to_book.median, bvps),
            current_price=subject.current_price,
            net_debt=net_debt,
            shares_outstanding=shares,
        )

    # ------------------------------------------------------------------
    # Summary / display
    # ------------------------------------------------------------------

    def summary(self, result: CompsResult) -> str:
        """
        Formatted ASCII summary.

        Sections:
          1. Peer trading multiples table with stat rows (median / mean / p25 / p75)
          2. Subject company's own multiples
          3. Implied share price waterfall (one row per multiple methodology)
        """
        lines: list[str] = []
        W = 74

        def _bar():
            lines.append("=" * W)

        def _rule():
            lines.append("  " + "-" * (W - 2))

        def _fmt_x(v: Optional[float]) -> str:
            return f"{v:.1f}x" if v is not None else "N/A"

        def _fmt_p(v: Optional[float]) -> str:
            return f"${v:,.2f}" if v is not None else "N/A"

        def _updown(implied: Optional[float], current: Optional[float]) -> str:
            if implied is None or current is None or current == 0:
                return ""
            pct = (implied - current) / current * 100
            sign = "+" if pct >= 0 else ""
            return f"  ({sign}{pct:.1f}%)"

        _bar()
        lines.append(
            f"  COMPARABLE VALUATION  |  {result.subject.ticker}"
            f"  |  {result.subject.name}"
        )
        _bar()
        lines.append("")

        # ---- Peer multiples table ----------------------------------------
        lines.append("  PEER TRADING MULTIPLES")
        _rule()
        hdr = (
            f"  {'Ticker':<8} {'Name':<24}"
            f" {'EV/EBITDA':>9} {'EV/Rev':>7} {'P/E':>6} {'P/FCF':>7} {'P/B':>6}"
        )
        lines.append(hdr)
        _rule()

        for p in result.peers:
            m = p.multiples
            lines.append(
                f"  {p.ticker:<8} {p.name[:23]:<24}"
                f" {_fmt_x(m.ev_to_ebitda):>9}"
                f" {_fmt_x(m.ev_to_revenue):>7}"
                f" {_fmt_x(m.price_to_earnings):>6}"
                f" {_fmt_x(m.price_to_fcf):>7}"
                f" {_fmt_x(m.price_to_book):>6}"
            )

        _rule()
        s = result.stats
        for label, attr in [("Median", "median"), ("Mean", "mean"), ("P25", "p25"), ("P75", "p75")]:
            lines.append(
                f"  {label:<8} {'':<24}"
                f" {_fmt_x(getattr(s.ev_to_ebitda, attr)):>9}"
                f" {_fmt_x(getattr(s.ev_to_revenue, attr)):>7}"
                f" {_fmt_x(getattr(s.price_to_earnings, attr)):>6}"
                f" {_fmt_x(getattr(s.price_to_fcf, attr)):>7}"
                f" {_fmt_x(getattr(s.price_to_book, attr)):>6}"
            )
        lines.append("")

        # ---- Subject multiples ------------------------------------------
        lines.append(f"  SUBJECT MULTIPLES  |  {result.subject.ticker}")
        _rule()
        sm = result.subject.multiples
        for label, val, fmt in [
            ("EV / EBITDA", sm.ev_to_ebitda, _fmt_x),
            ("EV / Revenue", sm.ev_to_revenue, _fmt_x),
            ("P / E", sm.price_to_earnings, _fmt_x),
            ("P / FCF", sm.price_to_fcf, _fmt_x),
            ("P / Book", sm.price_to_book, _fmt_x),
            ("Current Price", sm.current_price, _fmt_p),
        ]:
            lines.append(f"  {label:<20}  {fmt(val)}")
        lines.append("")

        # ---- Implied valuations -----------------------------------------
        lines.append("  IMPLIED SHARE PRICE  (peer-set median multiples)")
        _rule()
        iv = result.implied
        current = iv.current_price
        for label, val in [
            ("From EV/EBITDA", iv.implied_from_ev_ebitda),
            ("From EV/Revenue", iv.implied_from_ev_revenue),
            ("From P/E", iv.implied_from_pe),
            ("From P/FCF", iv.implied_from_pfcf),
            ("From P/Book", iv.implied_from_pb),
        ]:
            ud = _updown(val, current)
            lines.append(f"  {label:<22}  {_fmt_p(val)}{ud}")

        lines.append("")
        if current is not None:
            lines.append(f"  Current Market Price  :  {_fmt_p(current)}")
        lines.append("")
        _bar()

        return "\n".join(lines)
