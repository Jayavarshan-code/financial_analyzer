"""
data/models/assumption.py
--------------------------
Pydantic v2 data models for the Assumption Tracker.

Model hierarchy:
  AssumptionEntry  - one versioned snapshot of DCFAssumptions with run metadata
  FieldDiff        - one field that changed between two entries
  AssumptionDiff   - complete diff between two AssumptionEntry objects

Design notes:
  - AssumptionEntry.id is a short UUID prefix (8 chars) for human-readable logs.
  - timestamp is UTC so logs remain interpretable across timezones.
  - implied_price and wacc_effective are stored from the DCFResult that consumed
    these assumptions so every log entry shows the valuation outcome as well.
  - FieldDiff.change_pct is populated only for numeric fields and expresses the
    absolute change in percentage-points (not relative %) for rate fields, and
    relative % change for monetary/share-count fields.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from data.models.dcf import DCFAssumptions


class AssumptionEntry(BaseModel):
    """One logged assumption set with outcome metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    ticker: str
    label: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    assumptions: DCFAssumptions
    implied_price: Optional[float] = None    # from DCFResult.bridge.implied_share_price
    wacc_effective: Optional[float] = None   # from DCFResult.wacc_result.wacc


class FieldDiff(BaseModel):
    """One field that changed between two assumption entries."""

    field: str                       # machine-readable field name
    label: str                       # human-readable display label
    old_value: Any
    new_value: Any
    change_abs: Optional[float] = None   # absolute change (new - old) for numeric fields
    change_pct: Optional[float] = None   # relative change ((new-old)/old) for non-rate fields


class AssumptionDiff(BaseModel):
    """
    Field-level diff between two AssumptionEntry objects.

    changed:         fields whose value differed between entry_a and entry_b
    unchanged_count: number of tracked fields that were identical
    """

    ticker: str
    entry_a_id: str
    entry_b_id: str
    entry_a_label: str
    entry_b_label: str
    changed: list[FieldDiff]
    unchanged_count: int
