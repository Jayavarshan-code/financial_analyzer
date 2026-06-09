# DCF Valuation Engine

A professional-grade equity valuation platform built in Python. It fetches live financial data from Yahoo Finance, applies institutional-quality analytical methods across six engines, and surfaces results through both a Typer CLI and a Streamlit web application.

---

## Table of Contents

1. [Business Context](#business-context)
2. [Architecture Overview](#architecture-overview)
3. [Engine Reference](#engine-reference)
   - [Financial Statement Analyzer](#1-financial-statement-analyzer)
   - [DCF Engine](#2-dcf-engine)
   - [WACC Deriver](#3-wacc-deriver)
   - [Comparable Valuation Engine](#4-comparable-valuation-engine)
   - [Scenario Analysis & Monte Carlo](#5-scenario-analysis--monte-carlo)
   - [Revenue & Earnings Forecasting](#6-revenue--earnings-forecasting)
   - [Report Generator](#7-report-generator)
   - [Assumption Tracker](#8-assumption-tracker)
4. [Key Technical Design Decisions](#key-technical-design-decisions)
5. [Data Models](#data-models)
6. [Installation](#installation)
7. [CLI Usage](#cli-usage)
8. [Streamlit Web App](#streamlit-web-app)
9. [Test Suite](#test-suite)
10. [Project Structure](#project-structure)
11. [Limitations & Known Issues](#limitations--known-issues)

---

## Business Context

Equity valuation is performed daily by hedge funds, investment banks, and independent analysts to decide whether a public company's stock is fairly priced. The dominant methodologies are:

- **DCF (Discounted Cash Flow)** — intrinsic value: what is this business worth if we own all its future free cash flows?
- **Comparable Company Analysis (Comps)** — relative value: how does the market price similar businesses, and what does that imply for this one?
- **Scenario Analysis** — sensitivity: what does the range of plausible outcomes look like?

Commercial platforms (Bloomberg Terminal, FactSet, Capital IQ) charge $25,000–$50,000 per seat per year for this functionality. This project replicates the analytical core at zero cost using public data, with institutional-quality methodology choices throughout.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Entry Points                                │
│            CLI (main.py / Typer)   ·   Web App (streamlit_app.py)  │
└────────────────────────┬───────────────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────────────┐
│                    core/  — Analytical Engines                      │
│                                                                     │
│  FinancialStatementAnalyzer  →  computes ratios, enriches history   │
│           │                                                         │
│           ▼                                                         │
│      DCFEngine  ←──── WACCDeriver  ←──── market_rates (^TNX)       │
│      CompsEngine                                                    │
│      ScenarioAnalyzer  (Bull / Base / Bear + Monte Carlo)           │
│      RevenueForecastEngine  /  EarningsForecastEngine               │
│      ReportGenerator                                                │
│      AssumptionTracker                                              │
└────────────────────────┬───────────────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────────────┐
│                    data/  — Models & Fetchers                       │
│                                                                     │
│  Pydantic v2 models  →  FullFinancialHistory  (shared contract)     │
│  YFinanceFetcher     →  annual snapshots + TTM snapshot             │
│  market_rates        →  live 10-yr Treasury yield (^TNX)            │
└─────────────────────────────────────────────────────────────────────┘
```

**Data flow:**
```
Yahoo Finance API
    → YFinanceFetcher (normalize, derive missing fields, build TTM)
    → FullFinancialHistory
    → FinancialStatementAnalyzer (compute ratios)
    → [DCFEngine | CompsEngine | ScenarioAnalyzer | ForecastEngine]
    → [CLI output | Streamlit tabs | HTML/PDF report]
```

**Shared data contract:** Every engine consumes `FullFinancialHistory`. This means all engines are independently testable with mock data and the pipeline is composable.

---

## Engine Reference

### 1. Financial Statement Analyzer

**File:** `core/financial_statements.py`

Orchestrates the fetch → normalize → compute-ratios pipeline.

**Ratio groups computed per period:**

| Group | Metrics |
|---|---|
| Profitability | Gross margin, operating margin, net margin, EBITDA margin, FCF margin, ROA, ROE, ROIC, ROCE |
| Liquidity | Current ratio, quick ratio, cash ratio, OCF ratio |
| Leverage | Debt/equity, debt/assets, net debt/EBITDA, interest coverage, equity multiplier |
| Efficiency | Asset turnover, inventory turnover, receivables turnover, DSO, DIO, DPO, cash conversion cycle |
| Growth | Revenue, gross profit, operating income, net income, EBITDA, FCF, EPS — all YoY |

**TTM ratio enrichment:** When the fetcher builds a TTM snapshot, the analyzer computes ratios for it and sets YoY growth rates relative to the last annual snapshot. This means the WACC deriver's ICR lookup and the DCF's base-year margin all use the most current 12 months of data, not a potentially 11-month-old 10-K.

---

### 2. DCF Engine

**File:** `core/dcf_engine.py`

Implements the full five-stage DCF pipeline.

#### FCF Build-Up

```
Revenue   = prior_revenue × (1 + growth_rate)
EBITDA    = Revenue × ebitda_margin
D&A       = Revenue × da_as_pct_revenue
EBIT      = EBITDA − D&A
NOPAT     = EBIT × (1 − tax_rate)
CapEx     = Revenue × capex_as_pct_revenue
ΔNWC      = ΔRevenue × nwc_change_as_pct_revenue_delta
FCF       = NOPAT − CapEx + D&A − ΔNWC
PV(FCF)   = FCF / (1 + WACC)^t
```

#### Terminal Value

- **Gordon Growth Model:** `TV = FCF_n × (1 + TGR) / (WACC − TGR)` — requires WACC > TGR
- **Exit EV/EBITDA:** `TV = EBITDA_n × exit_multiple` — anchored to current market pricing

#### EV → Price Bridge

```
EV           = Σ PV(FCF_t) + PV(TV)
Equity Value = EV − Net Debt − Minority Interest
Implied Price = Equity Value / Diluted Shares
```

#### Sensitivity Table

A 9×9 grid sweeping WACC ±200 bps and TGR ±100 bps around the base case. Invalid combinations where WACC ≤ TGR return `None` (displayed as N/A).

#### Regime-Aware Growth Derivation

The naive approach — three-year average growth tapered linearly to 2.5% TGR — is structurally flawed. It punishes high-growth companies by pulling Year-1 down to a historical average, and gives dying companies a false recovery. Four regimes are detected instead:

| Regime | Detection | Behavior |
|---|---|---|
| **A — Persistent Decline** | Base growth < 0 AND (last 2 rates both negative OR OLS slope ≤ 0) | No mean-reversion to GDP growth. Partial recovery path stays negative Y1–Y3. |
| **B — Fast Deceleration** | OLS slope on growth rates < −2 pp/yr | Extrapolate trend 2 years forward before fading to TGR. Avoids assuming the company suddenly stabilizes. |
| **C — Accelerating Growth** | OLS slope > +1 pp/yr | Preserve momentum Y1–Y2 before fading. Avoids penalizing a 30%→35%→40% grower. |
| **D — Stable/Mature** | Default | Convex fade from exponentially-weighted recent growth to TGR. |

**Exponential recency weighting:** `weights = [2^i for i in range(n)]` (oldest=1, newest=2^(n−1)). A company with 50% growth two years ago and 5% last year gets 5% as the dominant signal, not a diluted average.

**Margin trend projection:** OLS on the last 4 years of EBITDA margin. Projects 2 years of that trend before stabilizing. Bounded to [5%, 60%].

#### TTM-Aware Base Year

```python
@property
def base_snapshot(self) -> Optional[FinancialSnapshot]:
    if ttm_snapshot is not None and ttm_snapshot.period > latest.period:
        return ttm_snapshot
    return latest
```

If quarterly data is available, the DCF projects forward from the most current 12-month revenue figure rather than a potentially 11-month-old 10-K. This can change the implied price by 5–15% for companies with strong quarterly momentum.

---

### 3. WACC Deriver

**File:** `core/wacc_deriver.py`

Five professional refinements over the naive hardcoded-constant approach:

#### Risk-Free Rate

Fetches the live 10-year US Treasury yield from `^TNX` via yfinance. Session-level cache (1 hr TTL) prevents redundant network calls during Monte Carlo runs. Hardcoded fallback = 4.5%.

```python
rfr = yf.Ticker("^TNX").info.get("regularMarketPrice") / 100
```

#### Beta — Blume's Adjustment

Raw historical beta naturally reverts toward 1.0 over time. Blume (1971) showed adjusted beta better predicts future beta:

```
Adjusted Beta = 0.67 × Raw Beta + 0.33 × 1.0
```

Both raw and adjusted betas are stored in `WACCResult` for auditability.

#### Size Premium — Kroll-Style Tiers

CAPM underestimates required returns for smaller companies. Approximates Kroll (formerly Duff & Phelps) CRSP Decile premia:

| Market Cap | Premium |
|---|---|
| < $250M (micro-cap) | 4.0% |
| $250M – $2B (small-cap) | 2.0% |
| $2B – $5B (mid-small) | 1.0% |
| > $5B | 0.0% |

#### Cost of Debt — Synthetic Credit Rating

Derives pre-tax Kd from the Interest Coverage Ratio (EBIT / Interest Expense) via Damodaran's default spread table, rather than the dangerous `interest_expense / total_debt` historical average (which is backward-looking and distorted by fixed-rate debt issued in different rate environments).

```
ICR > 8.50 → AAA → Kd = RFR + 0.63%
ICR > 6.50 → AA  → Kd = RFR + 0.78%
ICR > 5.50 → A+  → Kd = RFR + 0.98%
...
ICR < 0.20 → D   → Kd = RFR + 14.0%
```

#### Capital Structure Weights — Market Value

```
E (market) = Shares Outstanding × Current Price
D (market) ≈ Book Value of Debt  [standard approximation]
Wd = D / (E + D)
```

Never uses a hardcoded 20/80 split. Falls back to 80/20 only when price or shares are unavailable.

#### WACC Formula

```
Ke   = RFR + Adjusted_Beta × ERP + Size_Premium
Kd   = Synthetic_Rating_Spread + RFR
WACC = Ke × We + Kd × (1 − tax_rate) × Wd
```

---

### 4. Comparable Valuation Engine

**File:** `core/comps_engine.py`

Fetches live market data and TTM (or latest annual) financials for a subject and peer set, computes five standard trading multiples, and back-solves implied share prices.

**Multiples computed:**

| Multiple | Formula | Notes |
|---|---|---|
| EV/EBITDA | Enterprise Value / TTM EBITDA | Most commonly used; controls for capital structure and D&A policy differences |
| EV/Revenue | Enterprise Value / TTM Revenue | Used for pre-profitability or high-growth companies |
| P/E | Price / TTM EPS (diluted) | Most widely followed; distorted by leverage and non-cash items |
| P/FCF | Market Cap / TTM FCF | Cleaner than P/E; rewards cash generation |
| P/B | Market Cap / Book Equity | Useful for financials where tangible book matters |

**Statistics:** Mean, median, P25, P75, min, max computed via `numpy.percentile` across all peers with a valid (positive) denominator. Peers with negative denominators are excluded from that multiple's statistics without error.

**Implied price derivation:**
- EV-based: `Implied Price = (Median_Multiple × Subject_Metric − Net_Debt) / Shares`
- Price-based: `Implied Price = Median_Multiple × Subject_Per_Share_Metric`

---

### 5. Scenario Analysis & Monte Carlo

**File:** `core/scenario_analysis.py`

#### Bull / Base / Bear Scenarios

Three scenarios constructed by applying signed `ScenarioDelta` objects to the base auto-derived assumptions. Scenarios always use `wacc_override` (the computed effective WACC from the base run) so comparisons are single-variable and interpretable.

Default deltas — fully overridable per call:

| Driver | Bull | Bear |
|---|---|---|
| Revenue growth / yr | +3 pp | −3 pp |
| EBITDA margin | +2 pp | −2 pp |
| WACC | −50 bp | +50 bp |
| Terminal growth rate | +25 bp | −25 bp |

Custom scenarios via `ScenarioDelta`:
```python
from data.models.scenario import ScenarioDelta

result = ScenarioAnalyzer().run(
    history,
    bull_delta=ScenarioDelta(growth_delta=0.06, margin_delta=0.04, wacc_delta=-0.01),
    bear_delta=ScenarioDelta(growth_delta=-0.06, margin_delta=-0.05, wacc_delta=+0.02),
)
```

#### Monte Carlo — Vectorized with Cholesky Correlation

**The independence fallacy:** Drawing from four independent normal distributions is financially illiterate. An inflation spike simultaneously lifts WACC (rates up, credit spreads widen), squeezes margins (input cost pressure), and nudges TGR upward (nominal GDP channel). Treating these as uncorrelated random walks produces a statistically clean but economically meaningless bell curve.

**Correlation matrix (default, overridable via `MonteCarloConfig.correlations`):**

```
              growth   margin    wacc    tgr
growth      [  1.00    0.25   -0.20    0.30 ]
margin      [  0.25    1.00   -0.35    0.10 ]
wacc        [ -0.20   -0.35    1.00    0.40 ]
tgr         [  0.30    0.10    0.40    1.00 ]
```

Economic rationale:
- `growth ↔ margin (+0.25)`: operating leverage — revenue growth expands margins
- `growth ↔ wacc (−0.20)`: recessions compress growth AND widen credit spreads
- `wacc ↔ margin (−0.35)`: inflation spikes raise WACC and squeeze input margins
- `wacc ↔ tgr (+0.40)`: both driven by nominal rates / inflation expectations

**Implementation — explicit Cholesky, no Python loop:**

```python
# Step 1: Build covariance matrix
cov = np.outer(stds, stds) * rho + np.eye(4) * 1e-8

# Step 2: Cholesky — fails fast if correlation matrix is not positive definite
L = np.linalg.cholesky(cov)

# Step 3: Correlated draws — shape (n, 4)
Z = rng.standard_normal((n, 4))
draws = Z @ L.T

# Step 4-5: Entire DCF evaluated in one numpy pass (no Python loop)
revenue    = base_revenue * np.cumprod(1 + g_rates, axis=1)   # (n, n_proj)
fcf        = nopat - (capex - da + dnwc)                       # (n, n_proj)
pv_fcf_sum = (fcf * discount_factors).sum(axis=1)              # (n,)
tv         = fcf[:, -1] * (1 + tgrs) / (waccs - tgrs)         # (n,)
prices     = (pv_fcf_sum + pv_tv - net_debt) / shares          # (n,)
```

**Speedup:** ~50–100× over a Python `for` loop calling `DCFEngine.run()` per simulation. 5,000 correlated simulations run in under 0.1 seconds.

**Output:** Full distribution — mean, median, std dev, P5/P10/P25/P75/P90/P95, and probability of beating current market price.

---

### 6. Revenue & Earnings Forecasting

**Files:** `analytics/revenue_forecaster.py`, `analytics/earnings_forecaster.py`

#### Revenue Forecast Methods

| Method | Approach | Best For |
|---|---|---|
| CAGR | n-year compound annual growth rate | Simple reference point |
| Linear Trend | OLS: `Revenue = a + bt` | Steady, predictable growers |
| Exponential Trend | OLS on `log(Revenue)` | Percentage-growth compounders |
| Holt-Winters | Double exponential smoothing with additive trend (statsmodels) | Trend + level changes |
| Ensemble | MAPE-weighted average of all valid methods | Reduces single-model risk |

**Method selection:** The method with the lowest MAPE (Mean Absolute Percentage Error) on historical in-sample data is recommended. Confidence intervals are method-specific: regression methods use OLS prediction intervals; CAGR uses expanding intervals from historical growth std dev; Holt-Winters uses `simulate_smoother`.

**Pipeline integration:** The recommended forecast's implied growth rates feed directly into `DCFAssumptions.revenue_growth_rates`, creating a data-driven rather than analyst-guessed growth assumption for the DCF.

#### Earnings Forecasting

Two methods:
- **Margin-Based:** Average historical EBITDA and net margin applied to the revenue forecast. CI on EPS inherits from revenue CI.
- **EPS Trend:** OLS linear trend directly on historical EPS. Cross-check when margins are volatile.

---

### 7. Report Generator

**File:** `core/report_generator.py`

Generates complete HTML (and optionally PDF) valuation reports using Jinja2 templates.

**Report sections:**
1. Company overview and current market data
2. Financial statement summary (3-year table: revenue, EBITDA, net income, key ratios)
3. DCF valuation — WACC build-up, FCF projection table, EV bridge, 9×9 sensitivity heatmap
4. Comparable company analysis (if peers provided)
5. Scenario analysis — Bull/Base/Bear comparison, optional Monte Carlo distribution
6. Assumption log — timestamped history of every DCF run for this ticker

---

### 8. Assumption Tracker

**File:** `core/assumption_tracker.py`

Persists every DCF run's assumption set to a local JSON log file. Enables:
- Auditing how assumptions have changed over time for the same ticker
- Field-level diff between the last two runs (`--diff` flag)
- Reproducibility — you can see exactly what inputs drove a previous implied price

---

## Key Technical Design Decisions

### yfinance Field Name Drift

yfinance's column names for financial statement line items shift across library versions without warning. The naive `dict[str, str]` field map silently drops data when a name changes.

**Fix:** All field maps are `list[tuple[str, str]]` — ordered alias lists. The first matching alias wins. New yfinance names are prepended; old names are never removed. The fetcher never silently loses a line item because a column was renamed upstream.

```python
_IS_ALIASES: list[tuple[str, str]] = [
    ("Total Revenue",    "revenue"),
    ("Revenue",          "revenue"),     # older yfinance
    ("Operating Revenue","revenue"),     # banks, REITs
    ("EBITDA",           "ebitda"),
    ("Normalized EBITDA","ebitda"),      # alternate name
    ...
]
```

### Cross-Sector Comparability — Operating Lease Normalization

Under ASC 842 / IFRS 16, operating lease obligations appear on the balance sheet for US GAAP reporters (post-2019) but were historically off-balance-sheet. A retailer with $10B of store operating leases appears debt-free if leases aren't normalized — making EV/EBITDA, net-debt-to-EBITDA, and DCF net-debt adjustments wrong relative to a capital-goods company that owns its facilities.

**Fix:** `BalanceSheet` carries dedicated `operating_lease_liability_lt` and `operating_lease_liability_current` fields. The fetcher folds them into `total_debt` so leverage ratios are comparable across all sectors.

### D&A and EBITDA Derivation Fallbacks

Many non-US and financial-sector companies don't directly report EBITDA on their income statement, and D&A placement varies by reporter.

**Derivation chain:**
1. Use reported EBITDA if present
2. If D&A missing from IS, source it from the cash flow statement
3. Derive operating income from `gross_profit − R&D − SG&A`
4. Derive EBITDA from `operating_income + D&A`

Each derivation appends a warning to `FullFinancialHistory.data_warnings`, which is surfaced in both the CLI output and the Streamlit "Data Quality Warnings" expander.

### TTM Construction

Trailing Twelve Months is built by summing the last four quarterly statements for income statement and cash flow (flow statements) while using the most recent quarter's balance sheet (point-in-time stock).

Special handling:
- **EPS and share counts are NOT summed.** Share counts are point-in-time; TTM EPS is recomputed as `sum(quarterly_net_income) / latest_shares_diluted`.
- The `base_snapshot` property on `FullFinancialHistory` returns TTM only if it's strictly more recent than the last annual — if the most recent annual is December 2024 and TTM ends December 2024, there's no new information and the annual is used.

### Numpy Everywhere

All quantitative operations use numpy arrays. No custom statistical implementations exist — `numpy.percentile`, `numpy.mean`, `numpy.std`, and `numpy.linalg.cholesky` replace hand-rolled equivalents throughout. This eliminates a class of off-by-one percentile errors (nearest-rank vs. linear interpolation) and keeps the dependency surface consistent.

---

## Data Models

All monetary values are in **millions USD** throughout the entire codebase.

```
FullFinancialHistory
├── CompanyProfile          (ticker, name, sector, beta, market_cap, current_price)
├── annual_snapshots: list[FinancialSnapshot]
│   ├── statements: RawStatements
│   │   ├── IncomeStatement
│   │   ├── BalanceSheet
│   │   └── CashFlowStatement
│   └── ratios: FinancialRatios
│       ├── ProfitabilityRatios
│       ├── LiquidityRatios
│       ├── LeverageRatios
│       ├── EfficiencyRatios
│       └── GrowthRates
├── ttm_snapshot: Optional[FinancialSnapshot]
└── data_warnings: list[str]
```

`FullFinancialHistory` is the single shared contract between all engines. Every engine takes it as input and is independently testable against a mock implementation.

---

## Installation

**Requirements:** Python 3.12+

```bash
git clone https://github.com/your-username/financial-analyzer.git
cd financial-analyzer
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Core dependencies:**

| Package | Purpose |
|---|---|
| `yfinance >= 0.2.38` | Financial data from Yahoo Finance |
| `pydantic >= 2.7` | Typed data models and validation |
| `pandas >= 2.2` | DataFrame parsing of yfinance statements |
| `numpy >= 1.26` | All quantitative computations |
| `scipy >= 1.13` | Statistical functions |
| `statsmodels >= 0.14` | Holt-Winters forecasting |
| `streamlit` | Web app |
| `plotly` | Interactive charts |
| `typer >= 0.12` | CLI |
| `rich >= 13.7` | Terminal formatting |
| `jinja2 >= 3.1` | HTML report templates |
| `weasyprint >= 62` | PDF rendering (optional) |

> **Note on WeasyPrint:** PDF generation requires system-level GTK libraries. On Windows, install the GTK runtime from [gtk.org](https://gtk.org). HTML output works without it.

---

## CLI Usage

```bash
python main.py --help
```

### `analyze` — Financial Statement Summary

```bash
python main.py analyze AAPL
python main.py analyze MSFT --verbose
```

Fetches 4 years of financials and prints a formatted table of all ratio groups with growth trends.

### `dcf` — Full DCF Valuation

```bash
# Auto-derive all assumptions
python main.py dcf AAPL

# Override WACC and projection horizon
python main.py dcf AAPL --wacc 0.09 --years 7 --tgr 0.03

# Custom growth rate
python main.py dcf NVDA --growth 0.25 --years 5

# Exit multiple terminal value
python main.py dcf AAPL --method exit_multiple --exit-multiple 18
```

Output: WACC build-up, FCF projection table, EV bridge, 9×9 sensitivity heatmap.

### `comps` — Comparable Valuation

```bash
python main.py comps AAPL MSFT GOOGL META AMZN
python main.py comps NVDA AMD INTC QCOM
```

Output: Peer multiples table, implied prices from each multiple, peer-set statistics.

### `scenario` — Bull/Base/Bear + Monte Carlo

```bash
# 3 scenarios only
python main.py scenario AAPL

# With Monte Carlo (1000 simulations, default)
python main.py scenario AAPL --mc

# Custom simulation count and reproducible seed
python main.py scenario TSLA --mc --mc-n 5000 --mc-seed 42
```

Output: Side-by-side scenario table, optional Monte Carlo distribution (P5/P25/P50/P75/P95).

### `forecast` — Revenue & Earnings

```bash
# Revenue forecast, 5-year horizon
python main.py forecast AAPL

# 7 years + earnings (net income + EPS)
python main.py forecast AAPL --years 7 --earnings

# Force a specific method
python main.py forecast AAPL --method holt_winters
```

Output: Method comparison table (MAPE, R², CAGR), recommended forecast with 95% CI.

### `report` — Full HTML/PDF Report

```bash
# HTML report
python main.py report AAPL

# With comparable valuation and Monte Carlo
python main.py report AAPL --peers MSFT GOOGL META AMZN --mc

# PDF output
python main.py report AAPL --format pdf

# Both HTML and PDF
python main.py report AAPL --format both --peers MSFT GOOGL --mc
```

Reports are written to `reports/AAPL_<timestamp>.html` (and `.pdf`).

### `assumptions` — Audit Log

```bash
# Show full assumption history for AAPL
python main.py assumptions AAPL

# Diff the two most recent runs
python main.py assumptions AAPL --diff
```

---

## Streamlit Web App

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

### Sidebar Controls

| Control | Description |
|---|---|
| **Ticker** | Subject company (default: AAPL) |
| **Peer tickers** | Space-separated list for Comparable Valuation tab |
| **WACC override (%)** | Set 0 to auto-derive from live Treasury + Blume beta + synthetic Kd |
| **Terminal growth rate (%)** | Default 2.5% |
| **Projection years** | 3–10, default 5 |
| **Run Monte Carlo** | Toggle + simulation count slider (500–5000) |
| **Custom Bull / Bear Inputs** | Collapsible expander — override growth and margin shifts for each scenario |
| **Run Analysis** | Triggers all engines; results cached 1 hour per ticker |

### Tabs

| Tab | Contents |
|---|---|
| **Overview** | Company header, key metrics, historical financials bar chart, margin trend line chart, data quality warnings |
| **DCF Valuation** | WACC build-up (raw beta / Blume beta / size premium / synthetic rating), FCF projection chart, EV waterfall, 9×9 sensitivity heatmap (RdYlGn color scale) |
| **Comparable Valuation** | Peer multiples table, implied price bar chart with current-price line, peer-set statistics table |
| **Scenario Analysis** | Bull/Base/Bear comparison table, grouped bar chart, Monte Carlo percentile bar chart with P(price > current) |
| **Revenue & Earnings Forecast** | Method comparison table, revenue forecast line chart with 95% CI ribbon (actuals + projections), earnings forecast table (Revenue → Net Income → EPS with CIs) |

---

## Test Suite

```bash
pytest tests/ -v
```

**259 tests across 9 test files — all passing.**

| Test File | Coverage |
|---|---|
| `test_financial_statements.py` | Ratio computation, growth rates, ROIC, CCC, edge cases |
| `test_dcf_engine.py` | WACC calculator, FCF projector, terminal value, sensitivity table, full pipeline, summary rendering |
| `test_wacc_deriver.py` | Synthetic credit ratings (ICR boundaries), Blume adjustment, size premium tiers, capital weights, full integration |
| `test_comps_engine.py` | Multiple extraction, statistics, implied valuations, peer fetch, summary |
| `test_scenario_analysis.py` | Delta application, clipping, Bull > Base > Bear ordering, Monte Carlo distribution, reproducibility |
| `test_revenue_forecaster.py` | All five methods, CI ordering, MAPE, CAGR, method comparison |
| `test_earnings_forecaster.py` | Margin-based and EPS-trend methods, CI propagation |
| `test_report_generator.py` | HTML generation, section presence, format options |
| `test_assumption_tracker.py` | Log/load round-trip, diff, custom labels |

All tests are **network-free**: live data fetches are replaced with deterministic `MockFetcher` implementations. The RFR is injected via `risk_free_rate` parameter on `WACCDeriver.derive()` so `^TNX` is never called in tests.

---

## Project Structure

```
financial_analyzer/
├── main.py                          # CLI entry point (Typer)
├── streamlit_app.py                 # Web app (Streamlit)
├── requirements.txt
│
├── core/                            # Analytical engines
│   ├── financial_statements.py      # Ratio computation + enrichment
│   ├── dcf_engine.py                # DCF pipeline: WACC, FCF, TV, bridge, sensitivity
│   ├── wacc_deriver.py              # Market-calibrated WACC (^TNX, Blume, synthetic Kd)
│   ├── comps_engine.py              # Comparable company analysis
│   ├── scenario_analysis.py         # Bull/Base/Bear + vectorized Monte Carlo
│   ├── report_generator.py          # HTML/PDF report generation
│   └── assumption_tracker.py        # DCF run audit log
│
├── analytics/                       # Forecasting engines
│   ├── revenue_forecaster.py        # CAGR, OLS, Holt-Winters, Ensemble
│   └── earnings_forecaster.py       # Margin-based + EPS trend
│
├── data/
│   ├── fetchers/
│   │   ├── yfinance_fetcher.py      # Yahoo Finance → FullFinancialHistory + TTM
│   │   └── market_rates.py          # Live 10-yr Treasury (^TNX), ERP constant
│   └── models/
│       ├── financials.py            # FullFinancialHistory + all statement models
│       ├── dcf.py                   # DCFAssumptions, WACCInputs, DCFResult
│       ├── comps.py                 # CompsResult, CompsMultiples, ImpliedValuation
│       ├── scenario.py              # ScenarioDelta, MonteCarloConfig, ScenarioResult
│       ├── forecast.py              # RevenueForecastSuite, EarningsForecastSuite
│       ├── report.py                # ReportConfig, ReportInput, ReportOutput
│       └── assumption.py            # AssumptionEntry, AssumptionLog
│
└── tests/
    ├── test_financial_statements.py
    ├── test_dcf_engine.py
    ├── test_wacc_deriver.py
    ├── test_comps_engine.py
    ├── test_scenario_analysis.py
    ├── test_revenue_forecaster.py
    ├── test_earnings_forecaster.py
    ├── test_report_generator.py
    └── test_assumption_tracker.py
```

---

## Limitations & Known Issues

### Data Source — Yahoo Finance

All financial data is sourced from Yahoo Finance via `yfinance`. This introduces several known constraints:

- **Depth:** Maximum 4 annual periods. Longer histories require a paid data provider (FMP, Polygon, Bloomberg).
- **Rate limiting:** Yahoo Finance's undocumented API endpoints are occasionally throttled. The fetcher logs warnings but does not implement retry logic — add a decorator if needed.
- **International tickers:** Accounting taxonomy inconsistencies are more common for non-US filers, particularly around operating lease presentation (IFRS vs GAAP) and EBITDA reporting conventions.
- **Stale data:** `yfinance` caches `.info` internally per session. Restart the Python process for fresh market data.

### WACC Methodology Boundaries

- **ERP** is a documented configurable constant (`DEFAULT_ERP = 0.055` in `market_rates.py`) pointing to Damodaran's annual survey. It is **not** fetched dynamically — update it manually when Damodaran publishes a new estimate (typically January).
- **Beta** uses only Yahoo Finance's reported figure, adjusted via Blume. Bottom-up beta (unlevering peer betas and relevering at subject's capital structure) is not implemented.
- **Debt market value** uses book value as a standard approximation. For companies with significant off-market-rate debt, this can materially misprice Kd.

### Monte Carlo Correlation Assumptions

The default correlation matrix captures stylized macro factor exposure, not empirically estimated correlations from this specific company's history. For sector-specific analysis (financials, utilities, commodities), the matrix should be overridden via `MonteCarloConfig.correlations`.

### PDF Generation

WeasyPrint requires system-level GTK libraries that are non-trivial to install on Windows. HTML output is fully supported on all platforms. If `weasyprint` raises an import error, PDF generation is silently skipped and a warning is printed; HTML output is unaffected.
