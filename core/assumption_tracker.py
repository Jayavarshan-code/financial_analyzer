"""
core/assumption_tracker.py
----------------------------
Assumption Tracker — Phase 1, Module 5.

Responsibilities:
  1. Log each DCF assumption set (with its outcome) to a per-ticker JSON file.
  2. Load and replay the full history of assumption changes for any ticker.
  3. Produce a field-level diff between any two logged entries so analysts
     can see exactly what changed between runs and how prices moved.
  4. Render a concise ASCII summary table of the assumption history.

Persistence:
  Each ticker's log is stored as a JSON file at:
    {storage_dir}/{ticker.upper()}_assumptions.json

  The file is an array of serialized AssumptionEntry objects.
  The tracker always loads-then-appends to avoid clobbering parallel writes.

Tracked assumption fields (flattened from DCFAssumptions + WACCInputs):
  Revenue CAGR           ebitda_margin         da_as_pct_revenue
  tax_rate               capex_as_pct_revenue   nwc_change_as_pct_revenue_delta
  wacc_effective         terminal_growth_rate   terminal_value_method
  exit_ev_ebitda_multiple  net_debt             shares_outstanding
  beta                   risk_free_rate         equity_risk_premium
  cost_of_debt           debt_weight

Public API:
    tracker = AssumptionTracker()
    entry = tracker.log(ticker, assumptions, label="auto-derived",
                        implied_price=100.82, wacc_effective=0.0917)
    diff  = tracker.diff_latest(ticker)
    print(tracker.summary(ticker))
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from data.models.assumption import AssumptionDiff, AssumptionEntry, FieldDiff
from data.models.dcf import DCFAssumptions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flat field registry
# Fields are tuples of (field_key, display_label, is_rate)
# is_rate=True: change_abs is in percentage points; is_rate=False: relative %
# ---------------------------------------------------------------------------

_SCALAR_FIELDS: list[tuple[str, str, bool]] = [
    ("ebitda_margin",                   "EBITDA Margin",           True),
    ("da_as_pct_revenue",               "D&A % Revenue",           True),
    ("tax_rate",                        "Tax Rate",                True),
    ("capex_as_pct_revenue",            "CapEx % Revenue",         True),
    ("nwc_change_as_pct_revenue_delta", "NWC Change % Rev Delta",  True),
    ("terminal_growth_rate",            "Terminal Growth Rate",    True),
    ("exit_ev_ebitda_multiple",         "Exit EV/EBITDA Multiple", False),
    ("net_debt",                        "Net Debt ($M)",           False),
    ("shares_outstanding",              "Shares Outstanding ($M)", False),
]

_WACC_FIELDS: list[tuple[str, str, bool]] = [
    ("beta",                  "Beta",                False),
    ("risk_free_rate",        "Risk-Free Rate",      True),
    ("equity_risk_premium",   "Equity Risk Premium", True),
    ("cost_of_debt",          "Cost of Debt",        True),
    ("debt_weight",           "Debt Weight",         True),
]


def _flatten(entry: AssumptionEntry) -> dict[str, tuple[str, object, bool]]:
    """Return {field_key: (label, value, is_rate)} for all tracked fields."""
    a = entry.assumptions
    result: dict[str, tuple[str, object, bool]] = {}

    # Revenue CAGR — geometric mean of all projection year growth rates
    rates = a.revenue_growth_rates
    if rates:
        product = 1.0
        for r in rates:
            product *= (1 + r)
        cagr = product ** (1 / len(rates)) - 1
        result["revenue_cagr"] = ("Revenue CAGR", cagr, True)

    for key, label, is_rate in _SCALAR_FIELDS:
        result[key] = (label, getattr(a, key), is_rate)

    # WACC effective (from run result, not inputs)
    result["wacc_effective"] = ("WACC (effective)", entry.wacc_effective, True)

    # WACCInputs fields
    wi = a.wacc_inputs
    if wi is not None:
        for key, label, is_rate in _WACC_FIELDS:
            result[f"wacc_{key}"] = (label, getattr(wi, key), is_rate)

    # Terminal value method (string)
    result["terminal_value_method"] = ("TV Method", a.terminal_value_method.value, False)

    return result


def _field_diff(field_key: str, label: str, old: object, new: object, is_rate: bool) -> Optional[FieldDiff]:
    """Return a FieldDiff if old != new, else None."""
    if old == new:
        return None

    change_abs = None
    change_pct = None
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        old_f, new_f = float(old), float(new)
        change_abs = new_f - old_f
        if is_rate:
            change_pct = None  # for rates we report abs pp change, not relative %
        else:
            change_pct = change_abs / old_f if old_f != 0 else None

    return FieldDiff(
        field=field_key,
        label=label,
        old_value=old,
        new_value=new,
        change_abs=change_abs,
        change_pct=change_pct,
    )


# ---------------------------------------------------------------------------
# AssumptionTracker
# ---------------------------------------------------------------------------

class AssumptionTracker:
    """
    Logs, loads, diffs, and summarises DCF assumption histories per ticker.

    Args:
        storage_dir: Directory where JSON log files are stored.
                     Defaults to None, which is resolved at first use
                     to settings.data_cache_dir.
    """

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        self._storage_dir = storage_dir
        self._cache: dict[str, list[AssumptionEntry]] = {}

    def _resolve_dir(self) -> Path:
        if self._storage_dir is not None:
            return self._storage_dir
        from config import settings
        return settings.data_cache_dir

    def _path(self, ticker: str) -> Path:
        return self._resolve_dir() / f"{ticker.upper()}_assumptions.json"

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def log(
        self,
        ticker: str,
        assumptions: DCFAssumptions,
        label: str = "run",
        implied_price: Optional[float] = None,
        wacc_effective: Optional[float] = None,
    ) -> AssumptionEntry:
        """
        Create and persist a new AssumptionEntry for ticker.

        The log is loaded fresh from disk before appending so concurrent
        CLI invocations don't overwrite each other's history.

        Returns the newly created entry.
        """
        entry = AssumptionEntry(
            ticker=ticker.upper(),
            label=label,
            assumptions=assumptions,
            implied_price=implied_price,
            wacc_effective=wacc_effective,
        )
        history = self._load(ticker)
        history.append(entry)
        self._cache[ticker.upper()] = history
        self._save(ticker, history)
        logger.info(
            "AssumptionTracker: logged %s [%s] implied=$%s wacc=%.2f%%",
            ticker, label,
            f"{implied_price:.2f}" if implied_price else "N/A",
            (wacc_effective or 0) * 100,
        )
        return entry

    def get_history(self, ticker: str) -> list[AssumptionEntry]:
        """Return all logged entries for ticker, oldest first."""
        return self._load(ticker)

    def diff(self, entry_a: AssumptionEntry, entry_b: AssumptionEntry) -> AssumptionDiff:
        """
        Field-level diff between two entries.

        Only fields with differing values appear in AssumptionDiff.changed.
        unchanged_count counts fields that were identical.
        """
        flat_a = _flatten(entry_a)
        flat_b = _flatten(entry_b)

        changed: list[FieldDiff] = []
        unchanged = 0

        all_keys = set(flat_a) | set(flat_b)
        for key in sorted(all_keys):
            _, val_a, is_rate = flat_a.get(key, ("", None, False))
            label, val_b, _ = flat_b.get(key, ("", None, False))
            fd = _field_diff(key, label, val_a, val_b, is_rate)
            if fd is not None:
                changed.append(fd)
            else:
                unchanged += 1

        return AssumptionDiff(
            ticker=entry_a.ticker,
            entry_a_id=entry_a.id,
            entry_b_id=entry_b.id,
            entry_a_label=entry_a.label,
            entry_b_label=entry_b.label,
            changed=changed,
            unchanged_count=unchanged,
        )

    def diff_latest(self, ticker: str) -> Optional[AssumptionDiff]:
        """
        Diff the two most recent entries for ticker.
        Returns None if fewer than 2 entries exist.
        """
        history = self.get_history(ticker)
        if len(history) < 2:
            return None
        return self.diff(history[-2], history[-1])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self, ticker: str) -> list[AssumptionEntry]:
        """Load from disk (or return cached). Always returns a list."""
        key = ticker.upper()
        p = self._path(key)
        if not p.exists():
            return []
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return [AssumptionEntry.model_validate(item) for item in raw]
        except Exception as exc:
            logger.warning("Could not parse assumption log %s: %s", p, exc)
            return []

    def _save(self, ticker: str, entries: list[AssumptionEntry]) -> None:
        """Persist the entry list to disk as JSON."""
        p = self._path(ticker.upper())
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [e.model_dump(mode="json") for e in entries]
        p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------
    # Summary / display
    # ------------------------------------------------------------------

    def summary(self, ticker: str) -> str:
        """
        Formatted ASCII table of the assumption history for ticker.

        Columns: #, ID, Label, Timestamp, Implied Price, WACC, EBITDA Margin
        """
        history = self.get_history(ticker)
        lines: list[str] = []
        W = 80

        lines.append("=" * W)
        lines.append(f"  ASSUMPTION LOG  |  {ticker.upper()}  ({len(history)} entries)")
        lines.append("=" * W)

        if not history:
            lines.append("  No entries logged yet.")
            lines.append("=" * W)
            return "\n".join(lines)

        lines.append(
            f"  {'#':<4} {'ID':<10} {'Label':<22} {'Date':<12}"
            f" {'Implied $':>10} {'WACC':>7} {'EBITDA%':>8}"
        )
        lines.append("  " + "-" * (W - 2))

        for i, e in enumerate(history, 1):
            ts = e.timestamp.strftime("%Y-%m-%d")
            ip = f"${e.implied_price:,.2f}" if e.implied_price else "N/A"
            w = f"{e.wacc_effective * 100:.2f}%" if e.wacc_effective else "N/A"
            em = f"{e.assumptions.ebitda_margin * 100:.1f}%"
            lines.append(
                f"  {i:<4} {e.id:<10} {e.label[:21]:<22} {ts:<12}"
                f" {ip:>10} {w:>7} {em:>8}"
            )

        lines.append("=" * W)
        return "\n".join(lines)

    def diff_summary(self, diff: AssumptionDiff) -> str:
        """Formatted ASCII diff table."""
        lines: list[str] = []
        W = 76

        def _fmt_val(v: object, fd: FieldDiff) -> str:
            if v is None:
                return "N/A"
            if isinstance(v, float):
                # Heuristic: rates stored as decimals < 1 are shown as pct
                if abs(v) < 5:
                    return f"{v * 100:.2f}%"
                return f"{v:,.2f}"
            return str(v)

        lines.append("=" * W)
        lines.append(f"  ASSUMPTION DIFF  |  {diff.ticker}")
        lines.append(f"  [{diff.entry_a_id}] {diff.entry_a_label}  ->  [{diff.entry_b_id}] {diff.entry_b_label}")
        lines.append("=" * W)

        if not diff.changed:
            lines.append("  No changes detected.")
        else:
            lines.append(f"  {'Field':<30} {'Old':>12} {'New':>12} {'Change':>12}")
            lines.append("  " + "-" * (W - 2))
            for fd in diff.changed:
                old_s = _fmt_val(fd.old_value, fd)
                new_s = _fmt_val(fd.new_value, fd)
                if fd.change_abs is not None:
                    if isinstance(fd.old_value, float) and abs(fd.old_value) < 5:
                        chg = f"{fd.change_abs * 100:+.2f} pp"
                    elif fd.change_pct is not None:
                        chg = f"{fd.change_pct * 100:+.1f}%"
                    else:
                        chg = f"{fd.change_abs:+.2f}"
                else:
                    chg = "(changed)"
                lines.append(f"  {fd.label:<30} {old_s:>12} {new_s:>12} {chg:>12}")

        lines.append(f"\n  {diff.unchanged_count} fields unchanged.")
        lines.append("=" * W)
        return "\n".join(lines)
