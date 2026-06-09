"""
data/models/comps.py
---------------------
Pydantic v2 data models for the Comparable Valuation Engine.

Model hierarchy:
  CompsMultiples     - trading multiples for one ticker
                       (EV/EBITDA, EV/Revenue, P/E, P/FCF, P/B)
  CompsCompany       - profile metadata + computed multiples for one peer
  MultipleStats      - cross-sectional statistics (mean, median, p25, p75) for one multiple
  CompsStatistics    - MultipleStats for all 5 multiples across the peer set
  ImpliedValuation   - implied share price of the subject from each peer-set median multiple
  CompsResult        - full output: subject + peers + stats + implied valuations

Valuation methodology for implied prices:
  EV-based multiples (EV/EBITDA, EV/Revenue):
    Implied EV    = median_multiple x subject_metric
    Implied Price = (Implied EV - net_debt) / shares_outstanding

  Price-based multiples (P/E, P/FCF, P/B):
    Implied Price = median_multiple x subject_per_share_metric

Design notes:
  - Only peers with positive denominators are included in statistics.
    Negative EBITDA, negative earnings, or negative FCF → None multiple (excluded from stats).
  - All monetary fields in millions USD (matching FullFinancialHistory convention).
  - Multiples are dimensionless ratios stored as plain floats.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Per-company multiples
# ---------------------------------------------------------------------------

class CompsMultiples(BaseModel):
    """
    Trailing twelve-month trading multiples for one company.

    Raw financial inputs (in millions USD) and current market data are stored
    alongside the computed ratios so callers can verify or re-derive multiples.
    """

    ticker: str

    # Raw financial inputs (millions USD, TTM preferred)
    enterprise_value: Optional[float] = None
    market_cap: Optional[float] = None
    revenue: Optional[float] = None
    ebitda: Optional[float] = None
    net_income: Optional[float] = None
    eps_diluted: Optional[float] = None          # per-share
    free_cash_flow: Optional[float] = None       # CFO + CapEx (CapEx negative)
    book_value_equity: Optional[float] = None    # total stockholders equity
    shares_outstanding: Optional[float] = None   # millions
    current_price: Optional[float] = None

    # Computed multiples — None when denominator is zero or negative
    ev_to_ebitda: Optional[float] = None         # EV / EBITDA
    ev_to_revenue: Optional[float] = None        # EV / Revenue
    price_to_earnings: Optional[float] = None    # Price / EPS
    price_to_fcf: Optional[float] = None         # Market Cap / FCF
    price_to_book: Optional[float] = None        # Market Cap / Book Equity


class CompsCompany(BaseModel):
    """One company (subject or peer): profile metadata + computed multiples."""

    ticker: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    multiples: CompsMultiples


# ---------------------------------------------------------------------------
# Cross-sectional statistics across the peer set
# ---------------------------------------------------------------------------

class MultipleStats(BaseModel):
    """
    Descriptive statistics for a single multiple computed across all peers
    that had a valid (positive) value for that multiple.
    """

    multiple_name: str
    n_valid: int                          # number of peers with a valid value
    mean: Optional[float] = None
    median: Optional[float] = None
    p25: Optional[float] = None          # 25th percentile
    p75: Optional[float] = None          # 75th percentile
    minimum: Optional[float] = None
    maximum: Optional[float] = None


class CompsStatistics(BaseModel):
    """Aggregated cross-sectional statistics for all 5 multiples."""

    ev_to_ebitda: MultipleStats
    ev_to_revenue: MultipleStats
    price_to_earnings: MultipleStats
    price_to_fcf: MultipleStats
    price_to_book: MultipleStats


# ---------------------------------------------------------------------------
# Implied valuation of the subject company
# ---------------------------------------------------------------------------

class ImpliedValuation(BaseModel):
    """
    Implied share prices for the subject company, one per multiple methodology.

    EV-based: implied_ev = median_multiple x subject_metric
              implied_price = (implied_ev - net_debt) / shares_outstanding
    Price-based: implied_price = median_multiple x subject_per_share_metric

    None when the subject's metric is unavailable or the peer-set median is None.
    """

    implied_from_ev_ebitda: Optional[float] = None
    implied_from_ev_revenue: Optional[float] = None
    implied_from_pe: Optional[float] = None
    implied_from_pfcf: Optional[float] = None
    implied_from_pb: Optional[float] = None

    # Context stored for report generation
    current_price: Optional[float] = None
    net_debt: Optional[float] = None
    shares_outstanding: Optional[float] = None


# ---------------------------------------------------------------------------
# Full comps result
# ---------------------------------------------------------------------------

class CompsResult(BaseModel):
    """
    Complete output of one comparable company analysis run.

    subject:  the company being valued
    peers:    fetched peer data (sorted by ticker, excludes failed fetches)
    stats:    cross-sectional multiple statistics across the peer set
    implied:  implied share prices for the subject derived from peer-set medians
    """

    subject: CompsCompany
    peers: list[CompsCompany]
    stats: CompsStatistics
    implied: ImpliedValuation
