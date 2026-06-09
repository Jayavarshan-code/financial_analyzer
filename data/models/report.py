"""
data/models/report.py
----------------------
Pydantic v2 data models for the Executive Report Generator.

Model hierarchy:
  ReportFormat  - output format enum (html | pdf | both)
  ReportConfig  - controls output path, format, and which sections to include
  ReportInput   - all module outputs bundled for one report run
  ReportOutput  - paths to generated files + generation metadata

Design notes:
  - ReportConfig.output_dir defaults to None; the generator falls back to
    settings.reports_dir when None is passed.
  - comps_result and scenario_result are optional — the generator renders
    their sections only when present.
  - ReportOutput.pdf_error is populated (instead of raising) when WeasyPrint
    is not installed or fails, so HTML always succeeds independently of PDF.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from data.models.dcf import DCFResult
from data.models.comps import CompsResult
from data.models.financials import FullFinancialHistory
from data.models.scenario import ScenarioAnalysisResult


class ReportFormat(str, Enum):
    HTML = "html"
    PDF = "pdf"
    BOTH = "both"


class ReportConfig(BaseModel):
    """Controls how and where the report is written."""

    format: ReportFormat = ReportFormat.HTML
    output_dir: Optional[Path] = None        # None → use settings.reports_dir
    filename_stem: str = ""                  # empty → "{ticker}_{YYYYMMDD}"
    include_assumptions_section: bool = True
    include_sensitivity_table: bool = True


class ReportInput(BaseModel):
    """All module outputs bundled into one object for report generation."""

    history: FullFinancialHistory
    dcf_result: DCFResult
    comps_result: Optional[CompsResult] = None
    scenario_result: Optional[ScenarioAnalysisResult] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ReportOutput(BaseModel):
    """Paths to generated files and generation metadata."""

    ticker: str
    generated_at: datetime
    html_path: Optional[Path] = None
    pdf_path: Optional[Path] = None
    pdf_error: Optional[str] = None    # populated when PDF generation fails gracefully
