"""
core/report_generator.py
--------------------------
Executive Report Generator — Phase 1, Module 6.

Generates a professional HTML report (and optionally PDF via WeasyPrint)
that consolidates all Phase 1 module outputs into a single document.

Report sections:
  1. Header           - company name, ticker, sector/industry, date
  2. Executive Summary - implied price vs current, WACC, TV%, EV/equity bridge
  3. Financial Highlights - last 3 annual periods: revenue, EBITDA, margins, growth
  4. DCF Analysis     - WACC build-up, FCF projection, terminal value, sensitivity table
  5. Comparable Valuation (optional) - peer multiples, stats, implied prices
  6. Scenario Analysis (optional)    - Bull/Base/Bear table, MC distribution
  7. Assumptions      - full DCFAssumptions used in the base DCF run

Output:
  HTML file:  {output_dir}/{ticker}_{YYYYMMDD}.html
  PDF file:   {output_dir}/{ticker}_{YYYYMMDD}.pdf  (only if WeasyPrint is installed)

PDF requires WeasyPrint (pip install weasyprint). If not installed, the generator
logs a warning, populates ReportOutput.pdf_error, and still returns the HTML path.

Public API:
    from core.report_generator import ReportGenerator
    from data.models.report import ReportConfig, ReportInput, ReportFormat

    rpt = ReportGenerator()
    out = rpt.generate(
        ReportInput(history=history, dcf_result=dcf),
        ReportConfig(format=ReportFormat.HTML)
    )
    print(out.html_path)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Environment, BaseLoader

from data.models.report import ReportConfig, ReportFormat, ReportInput, ReportOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Jinja2 format helpers registered as filters
# ---------------------------------------------------------------------------

def _fmt_usd(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"${value:,.{decimals}f}"


def _fmt_m(value: Optional[float]) -> str:
    """Format a millions-USD figure: $1,234,567M."""
    if value is None:
        return "N/A"
    return f"${value:,.0f}M"


def _fmt_pct(value: Optional[float], decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def _fmt_x(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}x"


def _updown_cls(value: Optional[float]) -> str:
    """CSS class for positive/negative values."""
    if value is None:
        return ""
    return "pos" if value >= 0 else "neg"


# ---------------------------------------------------------------------------
# HTML template (Jinja2)
# ---------------------------------------------------------------------------

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{{ profile.ticker }} — Valuation Report</title>
<style>
:root{--navy:#1a2744;--blue:#2563eb;--lblue:#eff6ff;--green:#16a34a;--red:#dc2626;
  --gray:#6b7280;--border:#e5e7eb;--stripe:#f8fafc;}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#111;background:#f1f5f9;}
.page-header{background:var(--navy);color:#fff;padding:20px 32px;}
.page-header h1{font-size:20px;margin-bottom:2px;}
.page-header .meta{color:#94a3b8;font-size:11px;}
.section{background:#fff;border:1px solid var(--border);border-radius:6px;
  margin:14px 28px;padding:18px 22px;}
.section h2{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;
  color:var(--navy);border-bottom:2px solid var(--blue);padding-bottom:6px;margin-bottom:14px;}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;
  margin-bottom:4px;}
.kpi{background:var(--lblue);border-left:4px solid var(--blue);padding:10px 14px;border-radius:4px;}
.kpi-label{font-size:10px;color:var(--gray);margin-bottom:3px;}
.kpi-value{font-size:18px;font-weight:700;}
.kpi-sub{font-size:10px;color:var(--gray);margin-top:2px;}
.pos{color:var(--green);}
.neg{color:var(--red);}
table{width:100%;border-collapse:collapse;font-size:12px;}
th{background:var(--navy);color:#fff;padding:7px 10px;text-align:right;font-weight:600;}
th:first-child{text-align:left;}
td{padding:6px 10px;border-bottom:1px solid var(--border);text-align:right;}
td:first-child{text-align:left;font-weight:500;}
tr:nth-child(even) td{background:var(--stripe);}
tr:last-child td{border-bottom:none;}
.hl td{background:var(--lblue)!important;font-weight:700;}
.sens td{font-size:11px;padding:4px 7px;}
.sens-base-row td{border-top:2px solid var(--blue);border-bottom:2px solid var(--blue);}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
.footer{text-align:center;padding:20px;color:var(--gray);font-size:11px;}
</style>
</head>
<body>

<div class="page-header">
  <h1>{{ profile.name }}  ({{ profile.ticker }})</h1>
  <div class="meta">
    Sector: {{ profile.sector or 'N/A' }} &nbsp;|&nbsp;
    Industry: {{ profile.industry or 'N/A' }} &nbsp;|&nbsp;
    Currency: {{ profile.currency }} &nbsp;|&nbsp;
    Generated: {{ generated_at }}
  </div>
</div>

<!-- ======= EXECUTIVE SUMMARY ======= -->
<div class="section">
  <h2>Executive Summary</h2>
  <div class="kpi-grid">
    <div class="kpi">
      <div class="kpi-label">Implied Share Price</div>
      <div class="kpi-value">{{ bridge.implied_share_price | fmt_usd }}</div>
      {% if bridge.current_price %}
      <div class="kpi-sub">Current: {{ bridge.current_price | fmt_usd }}</div>
      {% endif %}
    </div>
    {% if bridge.upside_downside_pct is not none %}
    <div class="kpi">
      <div class="kpi-label">Upside / Downside</div>
      <div class="kpi-value {{ bridge.upside_downside_pct | updown_cls }}">
        {{ (bridge.upside_downside_pct * 100) | fmt_pct_raw }}%
      </div>
      <div class="kpi-sub">vs current market price</div>
    </div>
    {% endif %}
    <div class="kpi">
      <div class="kpi-label">WACC</div>
      <div class="kpi-value">{{ wacc.wacc | fmt_pct }}</div>
      <div class="kpi-sub">Ke: {{ wacc.cost_of_equity | fmt_pct }} &nbsp; Kd(AT): {{ wacc.after_tax_cost_of_debt | fmt_pct }}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Enterprise Value</div>
      <div class="kpi-value">{{ bridge.enterprise_value | fmt_m }}</div>
      <div class="kpi-sub">TV: {{ tv.pv_as_pct_of_ev | fmt_pct }} of EV</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Equity Value</div>
      <div class="kpi-value">{{ bridge.equity_value | fmt_m }}</div>
      <div class="kpi-sub">Net Debt: {{ bridge.net_debt | fmt_m }}</div>
    </div>
  </div>
</div>

<!-- ======= FINANCIAL HIGHLIGHTS ======= -->
{% if snapshots %}
<div class="section">
  <h2>Financial Highlights  ($ millions)</h2>
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        {% for s in snapshots %}<th>{{ s.period[:4] }}</th>{% endfor %}
      </tr>
    </thead>
    <tbody>
      <tr><td>Revenue</td>
        {% for s in snapshots %}<td>{{ s.statements.income_statement.revenue | fmt_m }}</td>{% endfor %}
      </tr>
      <tr><td>EBITDA</td>
        {% for s in snapshots %}<td>{{ s.statements.income_statement.ebitda | fmt_m }}</td>{% endfor %}
      </tr>
      <tr><td>EBIT (Op. Income)</td>
        {% for s in snapshots %}<td>{{ s.statements.income_statement.operating_income | fmt_m }}</td>{% endfor %}
      </tr>
      <tr><td>Net Income</td>
        {% for s in snapshots %}<td>{{ s.statements.income_statement.net_income | fmt_m }}</td>{% endfor %}
      </tr>
      <tr><td>Free Cash Flow</td>
        {% for s in snapshots %}<td>{{ s.statements.cash_flow_statement.free_cash_flow | fmt_m }}</td>{% endfor %}
      </tr>
      <tr class="hl"><td>EBITDA Margin</td>
        {% for s in snapshots %}<td>{{ s.ratios.profitability.ebitda_margin | fmt_pct }}</td>{% endfor %}
      </tr>
      <tr><td>Net Margin</td>
        {% for s in snapshots %}<td>{{ s.ratios.profitability.net_margin | fmt_pct }}</td>{% endfor %}
      </tr>
      <tr><td>Revenue Growth</td>
        {% for s in snapshots %}<td>{{ s.ratios.growth.revenue_growth | fmt_pct }}</td>{% endfor %}
      </tr>
      <tr><td>ROIC</td>
        {% for s in snapshots %}<td>{{ s.ratios.profitability.return_on_invested_capital | fmt_pct }}</td>{% endfor %}
      </tr>
    </tbody>
  </table>
</div>
{% endif %}

<!-- ======= DCF ANALYSIS ======= -->
<div class="section">
  <h2>DCF Analysis  (base year: {{ dcf_result.base_year }})</h2>

  <div class="two-col">
    <!-- WACC Build-Up -->
    <div>
      <p style="font-weight:700;margin-bottom:8px;font-size:11px;text-transform:uppercase;color:var(--navy)">WACC Build-Up</p>
      <table>
        <tbody>
          <tr><td>Risk-Free Rate</td><td>{{ wacc.risk_free_rate | fmt_pct }}</td></tr>
          <tr><td>Beta</td><td>{{ wacc.beta | fmt_x }}</td></tr>
          <tr><td>Equity Risk Premium</td><td>{{ wacc.equity_risk_premium | fmt_pct }}</td></tr>
          <tr><td>Cost of Equity (Ke)</td><td>{{ wacc.cost_of_equity | fmt_pct }}</td></tr>
          <tr><td>Pre-Tax Cost of Debt</td><td>{{ wacc.pre_tax_cost_of_debt | fmt_pct }}</td></tr>
          <tr><td>After-Tax Kd</td><td>{{ wacc.after_tax_cost_of_debt | fmt_pct }}</td></tr>
          <tr><td>Equity Weight</td><td>{{ wacc.equity_weight | fmt_pct(0) }}</td></tr>
          <tr><td>Debt Weight</td><td>{{ wacc.debt_weight | fmt_pct(0) }}</td></tr>
          <tr class="hl"><td>WACC</td><td>{{ wacc.wacc | fmt_pct }}</td></tr>
        </tbody>
      </table>
    </div>

    <!-- EV Bridge -->
    <div>
      <p style="font-weight:700;margin-bottom:8px;font-size:11px;text-transform:uppercase;color:var(--navy)">EV Bridge</p>
      <table>
        <tbody>
          <tr><td>PV of Explicit FCFs</td><td>{{ bridge.pv_explicit_fcfs | fmt_m }}</td></tr>
          <tr><td>PV of Terminal Value</td><td>{{ bridge.pv_terminal_value | fmt_m }}</td></tr>
          <tr class="hl"><td>Enterprise Value</td><td>{{ bridge.enterprise_value | fmt_m }}</td></tr>
          <tr><td>(-) Net Debt</td><td>{{ bridge.net_debt | fmt_m }}</td></tr>
          <tr><td>(-) Minority Interest</td><td>{{ bridge.minority_interest | fmt_m }}</td></tr>
          <tr class="hl"><td>Equity Value</td><td>{{ bridge.equity_value | fmt_m }}</td></tr>
          <tr><td>Shares Outstanding</td><td>{{ bridge.shares_outstanding | fmt_m }}</td></tr>
          <tr class="hl"><td>Implied Share Price</td><td>{{ bridge.implied_share_price | fmt_usd }}</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- FCF Projection -->
  <p style="font-weight:700;margin:14px 0 8px;font-size:11px;text-transform:uppercase;color:var(--navy)">FCF Projection ($M)</p>
  <table>
    <thead>
      <tr>
        <th>Year</th><th>Calendar</th><th>Revenue</th><th>Rev Growth</th>
        <th>EBITDA</th><th>EBIT</th><th>FCF</th><th>PV(FCF)</th>
      </tr>
    </thead>
    <tbody>
      {% for py in projected_years %}
      <tr>
        <td>{{ py.year }}</td>
        <td>{{ py.calendar_year or '-' }}</td>
        <td>{{ py.revenue | fmt_m }}</td>
        <td>{{ py.revenue_growth_rate | fmt_pct }}</td>
        <td>{{ py.ebitda | fmt_m }}</td>
        <td>{{ py.ebit | fmt_m }}</td>
        <td>{{ py.free_cash_flow | fmt_m }}</td>
        <td>{{ py.pv_free_cash_flow | fmt_m }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <!-- Terminal Value -->
  <p style="font-weight:700;margin:14px 0 8px;font-size:11px;text-transform:uppercase;color:var(--navy)">
    Terminal Value  ({{ tv.method.value | replace('_', ' ') | title }})
  </p>
  <table>
    <tbody>
      <tr><td>Terminal Year EBITDA</td><td>{{ tv.terminal_year_ebitda | fmt_m }}</td></tr>
      <tr><td>Terminal Year FCF</td><td>{{ tv.terminal_year_fcf | fmt_m }}</td></tr>
      {% if tv.terminal_growth_rate is not none %}
      <tr><td>Terminal Growth Rate</td><td>{{ tv.terminal_growth_rate | fmt_pct }}</td></tr>
      {% endif %}
      {% if tv.exit_multiple is not none %}
      <tr><td>Exit EV/EBITDA</td><td>{{ tv.exit_multiple | fmt_x }}</td></tr>
      {% endif %}
      <tr><td>Terminal Value (undiscounted)</td><td>{{ tv.terminal_value | fmt_m }}</td></tr>
      <tr class="hl"><td>PV of Terminal Value</td><td>{{ tv.pv_terminal_value | fmt_m }}</td></tr>
      <tr><td>TV as % of EV</td><td>{{ tv.pv_as_pct_of_ev | fmt_pct }}</td></tr>
    </tbody>
  </table>

  {% if include_sensitivity %}
  <!-- Sensitivity Table -->
  <p style="font-weight:700;margin:14px 0 8px;font-size:11px;text-transform:uppercase;color:var(--navy)">
    Sensitivity: Implied Price  (Rows: WACC  |  Cols: {{ sensitivity.col_label }})
  </p>
  <table class="sens">
    <thead>
      <tr>
        <th>WACC \</th>
        {% for cv in sensitivity.col_axis %}
        <th>
          {% if sensitivity.col_label == 'Terminal Growth Rate' %}{{ (cv * 100) | round(2) }}%
          {% else %}{{ cv }}x{% endif %}
        </th>
        {% endfor %}
      </tr>
    </thead>
    <tbody>
      {% for i, wacc_val in sensitivity.row_axis | enumerate %}
      <tr{% if (wacc_val - dcf_result.wacc_result.wacc) | abs < 0.0001 %} class="sens-base-row"{% endif %}>
        <td>{{ (wacc_val * 100) | round(2) }}%</td>
        {% for price in sensitivity.prices[i] %}
        <td>{% if price is none %}N/A{% else %}{{ price | fmt_usd }}{% endif %}</td>
        {% endfor %}
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}
</div>

<!-- ======= COMPARABLE VALUATION ======= -->
{% if comps %}
<div class="section">
  <h2>Comparable Valuation</h2>
  <table>
    <thead>
      <tr>
        <th>Company</th><th>EV/EBITDA</th><th>EV/Revenue</th><th>P/E</th><th>P/FCF</th><th>P/Book</th>
      </tr>
    </thead>
    <tbody>
      {% for p in comps.peers %}
      <tr>
        <td>{{ p.ticker }} — {{ p.name }}</td>
        <td>{{ p.multiples.ev_to_ebitda | fmt_x }}</td>
        <td>{{ p.multiples.ev_to_revenue | fmt_x }}</td>
        <td>{{ p.multiples.price_to_earnings | fmt_x }}</td>
        <td>{{ p.multiples.price_to_fcf | fmt_x }}</td>
        <td>{{ p.multiples.price_to_book | fmt_x }}</td>
      </tr>
      {% endfor %}
      <tr class="hl">
        <td>Peer Median</td>
        <td>{{ comps.stats.ev_to_ebitda.median | fmt_x }}</td>
        <td>{{ comps.stats.ev_to_revenue.median | fmt_x }}</td>
        <td>{{ comps.stats.price_to_earnings.median | fmt_x }}</td>
        <td>{{ comps.stats.price_to_fcf.median | fmt_x }}</td>
        <td>{{ comps.stats.price_to_book.median | fmt_x }}</td>
      </tr>
    </tbody>
  </table>

  <p style="font-weight:700;margin:14px 0 8px;font-size:11px;text-transform:uppercase;color:var(--navy)">Implied Prices (from peer-set medians)</p>
  <table>
    <tbody>
      <tr><td>From EV/EBITDA</td><td>{{ comps.implied.implied_from_ev_ebitda | fmt_usd }}</td></tr>
      <tr><td>From EV/Revenue</td><td>{{ comps.implied.implied_from_ev_revenue | fmt_usd }}</td></tr>
      <tr><td>From P/E</td><td>{{ comps.implied.implied_from_pe | fmt_usd }}</td></tr>
      <tr><td>From P/FCF</td><td>{{ comps.implied.implied_from_pfcf | fmt_usd }}</td></tr>
      <tr><td>From P/Book</td><td>{{ comps.implied.implied_from_pb | fmt_usd }}</td></tr>
      {% if comps.implied.current_price %}
      <tr class="hl"><td>Current Market Price</td><td>{{ comps.implied.current_price | fmt_usd }}</td></tr>
      {% endif %}
    </tbody>
  </table>
</div>
{% endif %}

<!-- ======= SCENARIO ANALYSIS ======= -->
{% if scenario %}
<div class="section">
  <h2>Scenario Analysis</h2>
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        {% for sr in scenario.scenarios %}<th>{{ sr.scenario.label }}</th>{% endfor %}
      </tr>
    </thead>
    <tbody>
      <tr><td>Revenue CAGR</td>
        {% for sr in scenario.scenarios %}
        <td>{{ sr.scenario.assumptions.revenue_growth_rates | mean_pct }}</td>
        {% endfor %}
      </tr>
      <tr><td>EBITDA Margin</td>
        {% for sr in scenario.scenarios %}<td>{{ sr.scenario.assumptions.ebitda_margin | fmt_pct }}</td>{% endfor %}
      </tr>
      <tr><td>WACC</td>
        {% for sr in scenario.scenarios %}<td>{{ sr.dcf_result.wacc_result.wacc | fmt_pct }}</td>{% endfor %}
      </tr>
      <tr><td>Terminal Growth Rate</td>
        {% for sr in scenario.scenarios %}<td>{{ sr.scenario.assumptions.terminal_growth_rate | fmt_pct }}</td>{% endfor %}
      </tr>
      <tr class="hl"><td>Implied Share Price</td>
        {% for sr in scenario.scenarios %}<td>{{ sr.dcf_result.bridge.implied_share_price | fmt_usd }}</td>{% endfor %}
      </tr>
      <tr><td>TV as % of EV</td>
        {% for sr in scenario.scenarios %}<td>{{ sr.dcf_result.terminal_value_result.pv_as_pct_of_ev | fmt_pct }}</td>{% endfor %}
      </tr>
    </tbody>
  </table>

  {% if scenario.monte_carlo %}
  <p style="font-weight:700;margin:14px 0 8px;font-size:11px;text-transform:uppercase;color:var(--navy)">
    Monte Carlo  (n = {{ scenario.monte_carlo.n_simulations | comma }}, valid = {{ scenario.monte_carlo.n_valid | comma }})
  </p>
  <table>
    <tbody>
      <tr><td>Mean Implied Price</td><td>{{ scenario.monte_carlo.mean_price | fmt_usd }}</td></tr>
      <tr><td>Median</td><td>{{ scenario.monte_carlo.median_price | fmt_usd }}</td></tr>
      <tr><td>Std Dev</td><td>{{ scenario.monte_carlo.std_price | fmt_usd }}</td></tr>
      <tr><td>5th Percentile</td><td>{{ scenario.monte_carlo.pct_5 | fmt_usd }}</td></tr>
      <tr><td>25th Percentile</td><td>{{ scenario.monte_carlo.pct_25 | fmt_usd }}</td></tr>
      <tr><td>75th Percentile</td><td>{{ scenario.monte_carlo.pct_75 | fmt_usd }}</td></tr>
      <tr><td>95th Percentile</td><td>{{ scenario.monte_carlo.pct_95 | fmt_usd }}</td></tr>
      {% if scenario.monte_carlo.probability_above_current is not none %}
      <tr class="hl"><td>P(Price > Current)</td>
        <td>{{ (scenario.monte_carlo.probability_above_current * 100) | round(1) }}%</td>
      </tr>
      {% endif %}
    </tbody>
  </table>
  {% endif %}
</div>
{% endif %}

<!-- ======= ASSUMPTIONS ======= -->
{% if include_assumptions %}
<div class="section">
  <h2>DCF Assumptions Used</h2>
  <div class="two-col">
    <table>
      <tbody>
        <tr><td>Projection Years</td><td>{{ assumptions.projection_years }}</td></tr>
        <tr><td>EBITDA Margin</td><td>{{ assumptions.ebitda_margin | fmt_pct }}</td></tr>
        <tr><td>D&amp;A % Revenue</td><td>{{ assumptions.da_as_pct_revenue | fmt_pct }}</td></tr>
        <tr><td>Tax Rate</td><td>{{ assumptions.tax_rate | fmt_pct }}</td></tr>
        <tr><td>CapEx % Revenue</td><td>{{ assumptions.capex_as_pct_revenue | fmt_pct }}</td></tr>
        <tr><td>NWC Change % Rev Delta</td><td>{{ assumptions.nwc_change_as_pct_revenue_delta | fmt_pct }}</td></tr>
        <tr><td>Terminal Growth Rate</td><td>{{ assumptions.terminal_growth_rate | fmt_pct }}</td></tr>
        <tr><td>TV Method</td><td>{{ assumptions.terminal_value_method.value }}</td></tr>
        <tr><td>Net Debt</td><td>{{ assumptions.net_debt | fmt_m }}</td></tr>
        <tr><td>Shares Outstanding</td><td>{{ assumptions.shares_outstanding | fmt_m }}</td></tr>
      </tbody>
    </table>
    <table>
      <thead><tr><th>Year</th><th>Revenue Growth</th></tr></thead>
      <tbody>
        {% for i, g in assumptions.revenue_growth_rates | enumerate %}
        <tr><td>Year {{ i + 1 }}</td><td>{{ g | fmt_pct }}</td></tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
{% endif %}

<div class="footer">
  DCF Valuation Engine &nbsp;|&nbsp; Generated {{ generated_at }} &nbsp;|&nbsp;
  All monetary values in millions USD. This report is for informational purposes only.
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Generates HTML (and optionally PDF) valuation reports.

    Usage:
        rpt = ReportGenerator()
        out = rpt.generate(
            ReportInput(history=history, dcf_result=dcf, comps_result=comps),
            ReportConfig(format=ReportFormat.HTML)
        )
        print(f"Report saved to {out.html_path}")
    """

    def __init__(self) -> None:
        self._env = self._build_env()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        report_input: ReportInput,
        config: Optional[ReportConfig] = None,
    ) -> ReportOutput:
        """
        Render the report and write output files.

        Args:
            report_input: All module outputs bundled together.
            config:       Output format and path settings. Defaults to HTML.

        Returns:
            ReportOutput with paths to written files.
        """
        if config is None:
            config = ReportConfig()

        output_dir = config.output_dir or self._default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        ticker = report_input.dcf_result.ticker
        stem = config.filename_stem or f"report_{ticker}_{report_input.generated_at.strftime('%Y%m%d')}"

        html_content = self._render_html(report_input, config)

        out = ReportOutput(ticker=ticker, generated_at=report_input.generated_at)

        if config.format in (ReportFormat.HTML, ReportFormat.BOTH):
            html_path = output_dir / f"{stem}.html"
            html_path.write_text(html_content, encoding="utf-8")
            out.html_path = html_path
            logger.info("HTML report written: %s", html_path)

        if config.format in (ReportFormat.PDF, ReportFormat.BOTH):
            pdf_path = output_dir / f"{stem}.pdf"
            out.pdf_path, out.pdf_error = self._render_pdf(html_content, pdf_path)

        return out

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render_html(self, inp: ReportInput, config: ReportConfig) -> str:
        """Render the Jinja2 template to an HTML string."""
        dcf = inp.dcf_result
        hist = inp.history

        snapshots = hist.annual_snapshots[-3:]   # last 3 periods

        context = {
            "profile":          hist.profile,
            "dcf_result":       dcf,
            "projected_years":  dcf.projected_years,
            "wacc":             dcf.wacc_result,
            "tv":               dcf.terminal_value_result,
            "bridge":           dcf.bridge,
            "sensitivity":      dcf.sensitivity,
            "assumptions":      dcf.assumptions,
            "snapshots":        snapshots,
            "comps":            inp.comps_result,
            "scenario":         inp.scenario_result,
            "generated_at":     inp.generated_at.strftime("%Y-%m-%d %H:%M UTC"),
            "include_sensitivity": config.include_sensitivity_table,
            "include_assumptions": config.include_assumptions_section,
        }

        tmpl = self._env.get_template("report")
        return tmpl.render(**context)

    @staticmethod
    def _render_pdf(html: str, output_path: Path) -> tuple[Optional[Path], Optional[str]]:
        """Try to render PDF via WeasyPrint; return (path, error_msg)."""
        try:
            from weasyprint import HTML as WP_HTML  # type: ignore
            WP_HTML(string=html).write_pdf(str(output_path))
            logger.info("PDF report written: %s", output_path)
            return output_path, None
        except ImportError:
            msg = "WeasyPrint not installed — PDF generation skipped. Run: pip install weasyprint"
            logger.warning(msg)
            return None, msg
        except Exception as exc:
            msg = f"PDF generation failed: {exc}"
            logger.warning(msg)
            return None, msg

    @staticmethod
    def _default_output_dir() -> Path:
        from config import settings
        return settings.reports_dir

    # ------------------------------------------------------------------
    # Jinja2 environment
    # ------------------------------------------------------------------

    def _build_env(self) -> Environment:
        env = Environment(loader=_StringTemplateLoader({"report": _TEMPLATE}))

        env.filters["fmt_usd"] = _fmt_usd
        env.filters["fmt_m"] = _fmt_m
        env.filters["fmt_pct"] = _fmt_pct
        env.filters["fmt_x"] = _fmt_x
        env.filters["updown_cls"] = _updown_cls
        env.filters["fmt_pct_raw"] = lambda v: f"{v:.1f}" if v is not None else "N/A"
        env.filters["fmt_pct"] = lambda v, decimals=1: _fmt_pct(v, decimals)
        env.filters["comma"] = lambda v: f"{int(v):,}" if v is not None else "N/A"
        env.filters["mean_pct"] = _mean_pct_filter
        env.filters["enumerate"] = lambda it: list(enumerate(it))

        return env


def _mean_pct_filter(rates: list[float]) -> str:
    """Geometric mean of growth rates formatted as pct (used in scenario table)."""
    if not rates:
        return "N/A"
    import math
    product = 1.0
    for r in rates:
        product *= (1 + r)
    cagr = product ** (1 / len(rates)) - 1
    return _fmt_pct(cagr)


class _StringTemplateLoader(BaseLoader):
    """Minimal Jinja2 loader backed by a plain dict of template strings."""

    def __init__(self, templates: dict[str, str]) -> None:
        self._templates = templates

    def get_source(self, environment: Environment, template: str):
        source = self._templates.get(template)
        if source is None:
            from jinja2 import TemplateNotFound
            raise TemplateNotFound(template)
        return source, None, lambda: True
