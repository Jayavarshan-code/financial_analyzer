"""
core/wacc_deriver.py
---------------------
Derives a fully market-calibrated WACCInputs from a FullFinancialHistory.

Problems solved vs the naive approach:
  1. Risk-Free Rate  — live 10-yr Treasury yield instead of a hardcoded number.
  2. Beta            — Blume's adjustment (0.67 * raw + 0.33) so beta regresses
                       toward the market mean over time; raw beta exposed for audit.
  3. Size Premium    — Kroll/Duff & Phelps-style tier: smaller companies carry
                       idiosyncratic risk CAPM alone does not capture.
  4. Cost of Debt    — Synthetic credit rating from ICR (Damodaran default-spread
                       table) instead of a hardcoded 5% or the dangerous
                       "interest expense / total debt" historical average.
  5. Capital weights — Market Value of Equity (shares * price) + Book Value of
                       Debt (standard approximation), not a fixed 20/80 split.

Public API:
    from core.wacc_deriver import WACCDeriver
    from data.fetchers.market_rates import DEFAULT_ERP

    inputs = WACCDeriver().derive(history)
    inputs = WACCDeriver().derive(history, risk_free_rate=0.043, erp=0.05)
"""

from __future__ import annotations

import logging
from typing import Optional

from data.fetchers.market_rates import DEFAULT_ERP, fetch_risk_free_rate
from data.models.dcf import WACCInputs
from data.models.financials import FullFinancialHistory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic credit rating table
#
# Source: Damodaran's interest-coverage-ratio to default-spread mapping.
#   http://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ratings.htm
#
# Each entry: (icr_lower_bound_exclusive, synthetic_rating, default_spread)
# Logic: iterate top-to-bottom; first entry where ICR > threshold wins.
# Spreads are approximations for US investment-grade and high-yield bonds
# as of 2024-2025.
# ---------------------------------------------------------------------------
_ICR_RATING_SPREAD: list[tuple[float, str, float]] = [
    ( 8.50,  "AAA",  0.0063),
    ( 6.50,  "AA",   0.0078),
    ( 5.50,  "A+",   0.0098),
    ( 4.25,  "A",    0.0108),
    ( 3.00,  "A-",   0.0122),
    ( 2.50,  "BBB",  0.0156),
    ( 2.00,  "BB+",  0.0200),
    ( 1.75,  "BB",   0.0240),
    ( 1.50,  "B+",   0.0275),
    ( 1.25,  "B",    0.0440),
    ( 0.80,  "B-",   0.0520),
    ( 0.50,  "CCC",  0.0800),
    ( 0.20,  "CC",   0.1000),
    (-9999,  "D",    0.1400),   # negative ICR → distressed
]


def synthetic_kd(icr: float, risk_free_rate: float) -> tuple[str, float]:
    """
    Map an Interest Coverage Ratio to a (synthetic_rating, cost_of_debt).

    Args:
        icr:            EBIT / Interest Expense. Negative → distressed.
        risk_free_rate: 10-yr gov't bond yield (decimal).

    Returns:
        (rating_label, pre_tax_kd) e.g. ("A-", 0.055)
    """
    for threshold, rating, spread in _ICR_RATING_SPREAD:
        if icr > threshold:
            kd = risk_free_rate + spread
            return rating, kd
    return "D", risk_free_rate + 0.1400


# ---------------------------------------------------------------------------
# Size premium tiers
#
# Approximates Kroll (formerly Duff & Phelps) CRSP Decile premia.
# These are additional returns required by investors in smaller companies
# over and above what CAPM predicts.
#
# Market cap tiers (millions USD):
#   Micro-cap  < $250M   : 4.0%
#   Small-cap  $250M–$2B : 2.0%
#   Mid-small  $2B–$5B   : 1.0%
#   Mid/Large  > $5B     : 0.0%
# ---------------------------------------------------------------------------
_SIZE_TIERS: list[tuple[float, float]] = [
    (     0.0, 0.04),   # micro-cap
    (   250.0, 0.02),   # small-cap
    ( 2_000.0, 0.01),   # mid-small
    ( 5_000.0, 0.00),   # large-cap and above
]


def size_premium(market_cap_millions: Optional[float]) -> float:
    """
    Return the Kroll-style size premium (decimal) for a given market cap.

    Args:
        market_cap_millions: Market capitalisation in millions USD.
                             Returns 0 if None (unknown size → don't penalise).
    """
    if market_cap_millions is None or market_cap_millions <= 0:
        return 0.0

    premium = 0.0
    for cap_threshold, sp in _SIZE_TIERS:
        if market_cap_millions > cap_threshold:
            premium = sp
        else:
            break
    return premium


# ---------------------------------------------------------------------------
# Blume's Beta Adjustment
#
# Raw historical beta naturally reverts toward 1.0 (the market average) over
# time. Blume (1971) showed that using the adjusted figure more accurately
# predicts future beta. Virtually all professional models apply this.
#
#   Adjusted Beta = 0.67 × Raw Beta + 0.33 × 1.0
# ---------------------------------------------------------------------------

def blume_adjust(raw_beta: float) -> float:
    """Apply Blume's adjustment: push raw beta 33% toward the market mean (1.0)."""
    return 0.67 * raw_beta + 0.33 * 1.0


# ---------------------------------------------------------------------------
# Capital structure weights (market-value based)
#
# Market rules:
#   E (market) = Shares Outstanding × Current Price  ← always use market value
#   D (market) ≈ Book Value of Debt                  ← book is accepted approximation
#   V = E + D
#   We = E / V,  Wd = D / V
#
# We never use a hardcoded 20/80 split. If market cap is unavailable we fall
# back to 80/20 equity/debt (sensible default for a profitable company).
# ---------------------------------------------------------------------------

def capital_weights(
    shares_outstanding: Optional[float],
    current_price: Optional[float],
    total_debt: Optional[float],
) -> tuple[float, float]:
    """
    Compute market-value capital structure weights.

    Args:
        shares_outstanding: Diluted shares in millions.
        current_price:      Current share price (USD).
        total_debt:         Book value of total debt in millions USD.

    Returns:
        (debt_weight, equity_weight) both as decimals summing to 1.0.
    """
    equity_mv = (
        shares_outstanding * current_price
        if shares_outstanding is not None and current_price is not None
        and shares_outstanding > 0 and current_price > 0
        else None
    )
    debt_bv = max(total_debt or 0.0, 0.0)  # negative net debt → zero weight

    if equity_mv is None:
        logger.debug("Market cap unavailable — defaulting to 80/20 equity/debt weights")
        return 0.20, 0.80  # (wd, we) fallback

    total_capital = equity_mv + debt_bv
    if total_capital <= 0:
        return 0.20, 0.80

    wd = debt_bv / total_capital
    we = equity_mv / total_capital
    return round(wd, 4), round(we, 4)


# ---------------------------------------------------------------------------
# Main deriver
# ---------------------------------------------------------------------------

class WACCDeriver:
    """
    Derives a market-calibrated WACCInputs from historical financial data.

    All four of the professional refinements are applied:
      - Live RFR from ^TNX (cached per-session)
      - Blume-adjusted beta + size premium
      - Synthetic Kd from ICR → Damodaran default spread
      - Market-value capital weights (shares * price for equity, book for debt)

    Usage:
        inputs = WACCDeriver().derive(history)
        # Override RFR or ERP if you want to freeze an assumption:
        inputs = WACCDeriver().derive(history, risk_free_rate=0.043, erp=0.05)
    """

    def derive(
        self,
        history: FullFinancialHistory,
        risk_free_rate: Optional[float] = None,
        erp: float = DEFAULT_ERP,
        tax_rate: Optional[float] = None,
    ) -> WACCInputs:
        """
        Derive WACCInputs from a FullFinancialHistory.

        Args:
            history:        Fully analysed financial history (ratios populated).
            risk_free_rate: Override the live Treasury fetch (useful for tests).
            erp:            Equity Risk Premium (decimal). Defaults to DEFAULT_ERP.
                            Update DEFAULT_ERP in market_rates.py when Damodaran
                            publishes a new estimate.
            tax_rate:       Override marginal tax rate. If None, derived from IS.

        Returns:
            WACCInputs with all fields populated from market data.
        """
        profile = history.profile
        latest = history.latest

        # ------------------------------------------------------------------
        # 1. Risk-Free Rate — live 10-yr Treasury or explicit override
        # ------------------------------------------------------------------
        rfr = risk_free_rate if risk_free_rate is not None else fetch_risk_free_rate()

        # ------------------------------------------------------------------
        # 2. Beta — Blume-adjusted, clipped to [0.3, 3.0]
        # ------------------------------------------------------------------
        raw_beta = profile.beta
        if raw_beta is None or raw_beta <= 0:
            logger.debug("%s: no beta from profile, using 1.0", profile.ticker)
            raw_beta = 1.0
        raw_beta = max(min(raw_beta, 3.0), 0.3)
        adj_beta = round(blume_adjust(raw_beta), 4)

        # ------------------------------------------------------------------
        # 3. Size premium — Kroll tier from live market cap
        # ------------------------------------------------------------------
        market_cap = profile.market_cap
        sp = size_premium(market_cap)
        if sp > 0:
            logger.debug(
                "%s: applying %.0f%% size premium (market cap $%.0fM)",
                profile.ticker, sp * 100, market_cap,
            )

        # ------------------------------------------------------------------
        # 4. Cost of Debt — synthetic credit rating from interest coverage
        # ------------------------------------------------------------------
        icr = self._get_icr(history)
        if icr is not None:
            rating, kd = synthetic_kd(icr, rfr)
            logger.debug(
                "%s: ICR=%.2f → synthetic rating %s → Kd=%.2f%%",
                profile.ticker, icr, rating, kd * 100,
            )
        else:
            # No interest data → assume investment-grade (AAA spread)
            rating = "AAA*"
            kd = rfr + 0.0063
            logger.debug(
                "%s: no interest data, assuming investment-grade Kd=%.2f%%",
                profile.ticker, kd * 100,
            )

        # ------------------------------------------------------------------
        # 5. Capital structure weights — market equity + book debt
        # ------------------------------------------------------------------
        shares = (
            latest.statements.income_statement.shares_diluted
            or profile.shares_outstanding
        )
        total_debt = latest.statements.balance_sheet.total_debt
        wd, _we = capital_weights(shares, profile.current_price, total_debt)

        # ------------------------------------------------------------------
        # 6. Tax rate — effective rate from IS, clipped
        # ------------------------------------------------------------------
        effective_tax = tax_rate
        if effective_tax is None:
            effective_tax = self._get_tax_rate(history)

        return WACCInputs(
            risk_free_rate=round(rfr, 5),
            beta=adj_beta,
            raw_beta=round(raw_beta, 4),
            equity_risk_premium=erp,
            size_premium=round(sp, 4),
            cost_of_debt=round(kd, 5),
            synthetic_rating=rating,
            tax_rate=round(effective_tax, 4),
            debt_weight=wd,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_icr(self, history: FullFinancialHistory) -> Optional[float]:
        """
        Return the most recent Interest Coverage Ratio (EBIT / Interest Expense).

        Prefers the ratio already computed by FinancialStatementAnalyzer; falls
        back to computing it directly from raw IS data.
        """
        latest = history.latest
        # Try the pre-computed leverage ratio first
        icr = latest.ratios.leverage.interest_coverage
        if icr is not None:
            return icr

        # Fall back to direct computation
        inc = latest.statements.income_statement
        ebit = inc.operating_income
        interest = inc.interest_expense
        if ebit is not None and interest is not None and interest != 0:
            return ebit / abs(interest)

        return None

    def _get_tax_rate(self, history: FullFinancialHistory) -> float:
        """Average effective tax rate from the last 3 years, clipped to [5%, 40%]."""
        rates = []
        for snap in history.annual_snapshots[-3:]:
            inc = snap.statements.income_statement
            if inc.income_tax and inc.pretax_income and inc.pretax_income > 0:
                rates.append(inc.income_tax / inc.pretax_income)
        rate = sum(rates) / len(rates) if rates else 0.21
        return max(min(rate, 0.40), 0.05)
