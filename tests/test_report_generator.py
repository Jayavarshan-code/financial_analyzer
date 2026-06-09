"""
tests/test_report_generator.py
--------------------------------
Unit tests for the Executive Report Generator.

Uses the same MockFetcher + DCFEngine + fixture pattern as the other test modules.
All file output goes to tmp_path — no real filesystem state written.

Tests verify:
  - HTML file is created and non-empty
  - Required sections are present in the HTML
  - PDF error is populated gracefully when WeasyPrint is absent
  - Comps and scenario sections appear only when provided
  - Assumption section is optional (can be suppressed via config)
  - ReportOutput contains correct paths

Run with:
    pytest tests/test_report_generator.py -v
"""

import pytest

from data.models.dcf import DCFAssumptions
from data.models.financials import (
    BalanceSheet,
    CashFlowStatement,
    CompanyProfile,
    FinancialRatios,
    FinancialSnapshot,
    FullFinancialHistory,
    IncomeStatement,
    RawStatements,
)
from data.models.report import ReportConfig, ReportFormat, ReportInput
from core.financial_statements import FinancialStatementAnalyzer
from core.dcf_engine import DCFEngine
from core.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Shared mock data (same as test_dcf_engine)
# ---------------------------------------------------------------------------

def _make_snapshot(period, revenue, net_income, total_assets, total_equity,
                   total_debt, cash, current_assets, current_liabilities,
                   ocf, capex, ebitda=0.0, da=0.0,
                   income_tax=0.0, pretax_income=0.0,
                   eps_diluted=0.0, shares_diluted=1.0):
    inc = IncomeStatement(
        period=period, revenue=revenue, net_income=net_income,
        operating_income=ebitda - da if (ebitda and da) else None,
        pretax_income=pretax_income, income_tax=income_tax,
        ebitda=ebitda, depreciation_amortization=da,
        eps_diluted=eps_diluted, shares_diluted=shares_diluted,
        cost_of_revenue=revenue * 0.55 if revenue else None,
    )
    bs = BalanceSheet(
        period=period, total_assets=total_assets,
        total_stockholders_equity=total_equity,
        total_debt=total_debt, net_debt=total_debt - cash,
        cash_and_equivalents=cash,
        total_current_assets=current_assets,
        total_current_liabilities=current_liabilities,
    )
    cf = CashFlowStatement(
        period=period, operating_cash_flow=ocf,
        capital_expenditures=capex,
        free_cash_flow=ocf + capex,
    )
    return FinancialSnapshot(
        period=period,
        statements=RawStatements(income_statement=inc, balance_sheet=bs, cash_flow_statement=cf),
        ratios=FinancialRatios(period=period),
    )


class MockFetcher:
    def fetch(self, ticker: str) -> FullFinancialHistory:
        s1 = _make_snapshot("2022-09-30", revenue=400_000, net_income=80_000,
                            total_assets=350_000, total_equity=70_000,
                            total_debt=100_000, cash=50_000,
                            current_assets=130_000, current_liabilities=120_000,
                            ocf=100_000, capex=-15_000,
                            ebitda=100_000, da=12_000,
                            income_tax=20_000, pretax_income=100_000,
                            eps_diluted=5.0, shares_diluted=16_000)
        s2 = _make_snapshot("2023-09-30", revenue=480_000, net_income=96_000,
                            total_assets=390_000, total_equity=80_000,
                            total_debt=90_000, cash=60_000,
                            current_assets=160_000, current_liabilities=140_000,
                            ocf=120_000, capex=-18_000,
                            ebitda=144_000, da=14_400,
                            income_tax=24_000, pretax_income=120_000,
                            eps_diluted=6.2, shares_diluted=15_500)
        return FullFinancialHistory(
            profile=CompanyProfile(
                ticker=ticker, name="Mock Corp",
                sector="Technology", industry="Software",
                current_price=25.0, shares_outstanding=15_500, beta=1.1,
            ),
            annual_snapshots=[s1, s2],
        )


FIXED_ASSUMPTIONS = DCFAssumptions(
    projection_years=5,
    revenue_growth_rates=[0.10, 0.09, 0.08, 0.07, 0.06],
    ebitda_margin=0.30,
    da_as_pct_revenue=0.03,
    tax_rate=0.20,
    capex_as_pct_revenue=0.04,
    nwc_change_as_pct_revenue_delta=0.01,
    wacc_override=0.09,
    terminal_growth_rate=0.025,
    net_debt=30_000.0,
    shares_outstanding=15_500.0,
)


@pytest.fixture
def history():
    return FinancialStatementAnalyzer(fetcher=MockFetcher()).analyze("MOCK")


@pytest.fixture
def dcf_result(history):
    return DCFEngine().run(history, assumptions=FIXED_ASSUMPTIONS)


@pytest.fixture
def report_input(history, dcf_result):
    return ReportInput(history=history, dcf_result=dcf_result)


# ---------------------------------------------------------------------------
# Basic HTML generation
# ---------------------------------------------------------------------------

class TestHtmlGeneration:
    def test_html_file_created(self, tmp_path, report_input):
        out = ReportGenerator().generate(report_input, ReportConfig(output_dir=tmp_path))
        assert out.html_path is not None
        assert out.html_path.exists()

    def test_html_file_non_empty(self, tmp_path, report_input):
        out = ReportGenerator().generate(report_input, ReportConfig(output_dir=tmp_path))
        assert out.html_path.stat().st_size > 1000

    def test_html_contains_ticker(self, tmp_path, report_input):
        out = ReportGenerator().generate(report_input, ReportConfig(output_dir=tmp_path))
        html = out.html_path.read_text(encoding="utf-8")
        assert "MOCK" in html

    def test_html_contains_company_name(self, tmp_path, report_input):
        out = ReportGenerator().generate(report_input, ReportConfig(output_dir=tmp_path))
        html = out.html_path.read_text(encoding="utf-8")
        assert "Mock Corp" in html

    def test_html_is_valid_html_structure(self, tmp_path, report_input):
        out = ReportGenerator().generate(report_input, ReportConfig(output_dir=tmp_path))
        html = out.html_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_output_ticker_matches(self, tmp_path, report_input):
        out = ReportGenerator().generate(report_input, ReportConfig(output_dir=tmp_path))
        assert out.ticker == "MOCK"


# ---------------------------------------------------------------------------
# Section presence
# ---------------------------------------------------------------------------

class TestSections:
    def _html(self, tmp_path, report_input, **config_kwargs) -> str:
        cfg = ReportConfig(output_dir=tmp_path, **config_kwargs)
        out = ReportGenerator().generate(report_input, cfg)
        return out.html_path.read_text(encoding="utf-8")

    def test_executive_summary_section_present(self, tmp_path, report_input):
        html = self._html(tmp_path, report_input)
        assert "Executive Summary" in html

    def test_dcf_section_present(self, tmp_path, report_input):
        html = self._html(tmp_path, report_input)
        assert "DCF Analysis" in html

    def test_wacc_build_up_present(self, tmp_path, report_input):
        html = self._html(tmp_path, report_input)
        assert "WACC Build-Up" in html

    def test_ev_bridge_present(self, tmp_path, report_input):
        html = self._html(tmp_path, report_input)
        assert "EV Bridge" in html

    def test_sensitivity_table_present_by_default(self, tmp_path, report_input):
        html = self._html(tmp_path, report_input)
        assert "Sensitivity" in html

    def test_sensitivity_table_suppressed_by_config(self, tmp_path, report_input):
        html = self._html(tmp_path, report_input, include_sensitivity_table=False)
        assert "Sensitivity: Implied Price" not in html

    def test_assumptions_section_present_by_default(self, tmp_path, report_input):
        html = self._html(tmp_path, report_input)
        assert "Assumptions Used" in html

    def test_assumptions_section_suppressed(self, tmp_path, report_input):
        html = self._html(tmp_path, report_input, include_assumptions_section=False)
        assert "Assumptions Used" not in html

    def test_comps_section_absent_when_not_provided(self, tmp_path, report_input):
        html = self._html(tmp_path, report_input)
        assert "Comparable Valuation" not in html

    def test_scenario_section_absent_when_not_provided(self, tmp_path, report_input):
        html = self._html(tmp_path, report_input)
        assert "Scenario Analysis" not in html

    def test_financial_highlights_present(self, tmp_path, report_input):
        html = self._html(tmp_path, report_input)
        assert "Financial Highlights" in html


# ---------------------------------------------------------------------------
# Comps and scenario sections
# ---------------------------------------------------------------------------

class TestOptionalSections:
    def test_comps_section_present_when_provided(self, tmp_path, history, dcf_result):
        from tests.test_comps_engine import (
            CompsEngine, MockFetcher as CompsMockFetcher
        )
        comps = CompsEngine(fetcher=CompsMockFetcher()).run("SUBJ", ["PEER1", "PEER2"])
        inp = ReportInput(history=history, dcf_result=dcf_result, comps_result=comps)
        out = ReportGenerator().generate(inp, ReportConfig(output_dir=tmp_path))
        html = out.html_path.read_text(encoding="utf-8")
        assert "Comparable Valuation" in html

    def test_scenario_section_present_when_provided(self, tmp_path, history, dcf_result):
        from core.scenario_analysis import ScenarioAnalyzer
        scenario = ScenarioAnalyzer().run(history)
        inp = ReportInput(history=history, dcf_result=dcf_result, scenario_result=scenario)
        out = ReportGenerator().generate(inp, ReportConfig(output_dir=tmp_path))
        html = out.html_path.read_text(encoding="utf-8")
        assert "Scenario Analysis" in html


# ---------------------------------------------------------------------------
# PDF handling
# ---------------------------------------------------------------------------

class TestPdfHandling:
    def test_pdf_not_generated_for_html_format(self, tmp_path, report_input):
        out = ReportGenerator().generate(report_input, ReportConfig(
            output_dir=tmp_path, format=ReportFormat.HTML
        ))
        assert out.pdf_path is None
        assert out.pdf_error is None

    def test_both_format_attempts_pdf(self, tmp_path, report_input):
        out = ReportGenerator().generate(report_input, ReportConfig(
            output_dir=tmp_path, format=ReportFormat.BOTH
        ))
        # PDF may succeed or fail gracefully (WeasyPrint may not be installed)
        assert out.html_path is not None and out.html_path.exists()
        # If PDF failed, error message is set (not an exception)
        if out.pdf_path is None:
            assert out.pdf_error is not None


# ---------------------------------------------------------------------------
# Filename
# ---------------------------------------------------------------------------

class TestFilename:
    def test_default_filename_contains_ticker(self, tmp_path, report_input):
        out = ReportGenerator().generate(report_input, ReportConfig(output_dir=tmp_path))
        assert "MOCK" in out.html_path.name

    def test_custom_stem_applied(self, tmp_path, report_input):
        out = ReportGenerator().generate(report_input, ReportConfig(
            output_dir=tmp_path, filename_stem="my_custom_report"
        ))
        assert out.html_path.name == "my_custom_report.html"
