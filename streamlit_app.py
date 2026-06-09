"""
streamlit_app.py
-----------------
Streamlit web interface for the DCF Valuation Engine.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import traceback
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── engine imports ───────────────────────────────────────────────────────────
from analytics.earnings_forecaster import EarningsForecastEngine
from analytics.revenue_forecaster import RevenueForecastEngine
from core.comps_engine import CompsEngine
from core.dcf_engine import DCFEngine
from core.financial_statements import FinancialStatementAnalyzer
from core.scenario_analysis import ScenarioAnalyzer
from data.models.dcf import DCFAssumptions, WACCInputs
from data.models.financials import FullFinancialHistory
from data.models.scenario import MonteCarloConfig, ScenarioDelta

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DCF Valuation Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── tiny CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-label { font-size: 0.78rem; color: #888; }
    .metric-value { font-size: 1.25rem; font-weight: 600; }
    .upside  { color: #22c55e; }
    .downside { color: #ef4444; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_m(v: Optional[float], suffix: str = "M") -> str:
    if v is None:
        return "N/A"
    if abs(v) >= 1_000:
        return f"${v/1_000:,.1f}B"
    return f"${v:,.0f}{suffix}"


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v*100:.1f}%" if v is not None else "N/A"


def _fmt_x(v: Optional[float]) -> str:
    return f"{v:.2f}x" if v is not None else "N/A"


def _color_upside(pct: Optional[float]) -> str:
    if pct is None:
        return "N/A"
    sign = "+" if pct >= 0 else ""
    cls = "upside" if pct >= 0 else "downside"
    return f'<span class="{cls}">{sign}{pct*100:.1f}%</span>'


# ── cached data fetches ───────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_history(ticker: str) -> FullFinancialHistory:
    return FinancialStatementAnalyzer().analyze(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def _run_dcf(ticker: str, wacc_override: Optional[float], tgr: float, years: int) -> object:
    history = _fetch_history(ticker)
    assumptions: Optional[DCFAssumptions] = None
    if wacc_override is not None:
        engine_tmp = DCFEngine()
        auto = engine_tmp._derive_assumptions(history)
        growth = auto.revenue_growth_rates[:years]
        if len(growth) < years:
            growth += [growth[-1]] * (years - len(growth))
        assumptions = auto.model_copy(update={
            "wacc_override": wacc_override,
            "terminal_growth_rate": tgr,
            "projection_years": years,
            "revenue_growth_rates": growth,
        })
    return DCFEngine().run(history, assumptions=assumptions)


@st.cache_data(ttl=3600, show_spinner=False)
def _run_comps(subject: str, peers: list[str]) -> object:
    return CompsEngine().run(subject, peers)


@st.cache_data(ttl=3600, show_spinner=False)
def _run_scenario(
    ticker: str,
    run_mc: bool,
    mc_n: int,
    bull_growth: float = 0.03,
    bull_margin: float = 0.02,
    bear_growth: float = -0.03,
    bear_margin: float = -0.02,
) -> object:
    history = _fetch_history(ticker)
    mc = MonteCarloConfig(n_simulations=mc_n, seed=42) if run_mc else None
    bull = ScenarioDelta(growth_delta=bull_growth, margin_delta=bull_margin,
                         wacc_delta=-0.005, tgr_delta=+0.0025)
    bear = ScenarioDelta(growth_delta=bear_growth, margin_delta=bear_margin,
                         wacc_delta=+0.005, tgr_delta=-0.0025)
    return ScenarioAnalyzer().run(history, mc_config=mc, bull_delta=bull, bear_delta=bear)


@st.cache_data(ttl=3600, show_spinner=False)
def _run_forecast(ticker: str, years: int) -> tuple:
    history = _fetch_history(ticker)
    rev = RevenueForecastEngine().run(history, n_years=years)
    try:
        earn = EarningsForecastEngine().run(history, rev.recommended)
    except Exception:
        earn = None
    return rev, earn


# ── sidebar ───────────────────────────────────────────────────────────────────

def _sidebar() -> dict:
    st.sidebar.title("DCF Valuation Engine")
    st.sidebar.markdown("---")

    ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()
    peers_raw = st.sidebar.text_input(
        "Peer tickers (space-separated)",
        value="MSFT GOOGL META AMZN",
        help="Used for the Comparable Valuation tab.",
    )
    peers = [p.upper().strip() for p in peers_raw.split() if p.strip()]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Overrides (optional)")

    wacc_pct = st.sidebar.number_input(
        "WACC override (%)", min_value=0.0, max_value=30.0,
        value=0.0, step=0.1,
        help="Leave 0 to auto-derive from live Treasury + Blume beta + synthetic Kd.",
    )
    wacc_override = wacc_pct / 100 if wacc_pct > 0 else None

    tgr_pct = st.sidebar.number_input(
        "Terminal growth rate (%)", min_value=0.0, max_value=5.0,
        value=2.5, step=0.1,
    )
    proj_years = st.sidebar.slider("Projection years", 3, 10, 5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Scenario / Monte Carlo")
    run_mc = st.sidebar.checkbox("Run Monte Carlo", value=False)
    mc_n = st.sidebar.slider("Simulations", 500, 5000, 1000, step=500,
                              disabled=not run_mc)

    with st.sidebar.expander("Custom Bull / Bear Inputs"):
        st.caption("Override the default ±3 pp growth / ±2 pp margin shifts.")
        bull_growth = st.number_input(
            "Bull: revenue growth shift (pp)", min_value=-10.0, max_value=20.0,
            value=3.0, step=0.5,
            help="Added to every projection year's growth rate in the bull case.",
        ) / 100
        bull_margin = st.number_input(
            "Bull: EBITDA margin shift (pp)", min_value=-10.0, max_value=20.0,
            value=2.0, step=0.5,
        ) / 100
        bear_growth = st.number_input(
            "Bear: revenue growth shift (pp)", min_value=-20.0, max_value=10.0,
            value=-3.0, step=0.5,
        ) / 100
        bear_margin = st.number_input(
            "Bear: EBITDA margin shift (pp)", min_value=-20.0, max_value=10.0,
            value=-2.0, step=0.5,
        ) / 100

    st.sidebar.markdown("---")
    run_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

    return dict(
        ticker=ticker, peers=peers,
        wacc_override=wacc_override, tgr=tgr_pct / 100,
        proj_years=proj_years,
        run_mc=run_mc, mc_n=mc_n,
        bull_growth=bull_growth, bull_margin=bull_margin,
        bear_growth=bear_growth, bear_margin=bear_margin,
        run_btn=run_btn,
    )


# ── tab renderers ─────────────────────────────────────────────────────────────

def _tab_overview(history: FullFinancialHistory, dcf_result) -> None:
    p = history.profile
    b = dcf_result.bridge
    w = dcf_result.wacc_result

    # Header row
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"## {p.name}  `{p.ticker}`")
        st.caption(f"{p.sector or ''}  ·  {p.industry or ''}  ·  {p.exchange or ''}")
    with col2:
        if p.current_price:
            st.metric("Current Price", f"${p.current_price:,.2f}")
    with col3:
        up = b.upside_downside_pct
        delta_str = f"{up*100:+.1f}%" if up is not None else None
        st.metric("Implied Price", f"${b.implied_share_price:,.2f}", delta=delta_str)

    st.markdown("---")

    # Key metrics row
    m_cols = st.columns(5)
    metrics = [
        ("Market Cap",     _fmt_m(p.market_cap)),
        ("Enterprise Val", _fmt_m(p.enterprise_value)),
        ("WACC",           f"{w.wacc*100:.2f}%"),
        ("Net Debt",       _fmt_m(b.net_debt)),
        ("Shares Out",     f"{b.shares_outstanding:,.0f}M"),
    ]
    for col, (label, val) in zip(m_cols, metrics):
        col.metric(label, val)

    # Historical financials
    st.subheader("Historical Financials")
    rows = []
    for s in history.annual_snapshots:
        inc = s.statements.income_statement
        rat = s.ratios
        rows.append({
            "Year":         s.period[:4],
            "Revenue ($M)": inc.revenue,
            "EBITDA ($M)":  inc.ebitda,
            "Net Income ($M)": inc.net_income,
            "Rev Growth":   rat.growth.revenue_growth,
            "EBITDA Margin": rat.profitability.ebitda_margin,
            "Net Margin":   rat.profitability.net_margin,
        })
    df = pd.DataFrame(rows).set_index("Year")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(x=df.index, y=df["Revenue ($M)"],  name="Revenue",    marker_color="#3b82f6")
        fig.add_bar(x=df.index, y=df["EBITDA ($M)"],   name="EBITDA",     marker_color="#22c55e")
        fig.add_bar(x=df.index, y=df["Net Income ($M)"], name="Net Income", marker_color="#a78bfa")
        fig.update_layout(title="Revenue / EBITDA / Net Income ($M)",
                          barmode="group", height=320,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        pct_df = df[["Rev Growth", "EBITDA Margin", "Net Margin"]].dropna(how="all")
        fig2 = go.Figure()
        for col, color in [("Rev Growth", "#3b82f6"), ("EBITDA Margin", "#22c55e"), ("Net Margin", "#a78bfa")]:
            vals = pct_df[col].dropna()
            fig2.add_scatter(x=vals.index, y=vals * 100, mode="lines+markers",
                             name=col, line=dict(color=color))
        fig2.update_layout(title="Growth & Margin Trends (%)", height=320,
                           yaxis_ticksuffix="%",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

    # Data quality warnings
    if history.data_warnings:
        with st.expander(f"Data Quality Warnings ({len(history.data_warnings)})", expanded=False):
            for w_msg in history.data_warnings:
                st.warning(w_msg, icon="⚠️")


def _tab_dcf(dcf_result) -> None:
    b = dcf_result.bridge
    w = dcf_result.wacc_result
    tv = dcf_result.terminal_value_result

    # ── WACC build-up ──────────────────────────────────────────────────────
    st.subheader("WACC Build-Up")
    wc1, wc2, wc3 = st.columns(3)

    with wc1:
        st.markdown("**Cost of Equity**")
        rows_ke = [
            ("Risk-Free Rate (^TNX)", f"{w.risk_free_rate*100:.2f}%"),
        ]
        if w.raw_beta is not None:
            rows_ke.append(("Beta (raw / Yahoo)",    f"{w.raw_beta:.2f}x"))
            rows_ke.append(("Beta (Blume adj.)",     f"{w.beta:.2f}x"))
        else:
            rows_ke.append(("Beta",                  f"{w.beta:.2f}x"))
        rows_ke.append(("Equity Risk Premium",   f"{w.equity_risk_premium*100:.2f}%"))
        if w.size_premium > 0:
            rows_ke.append(("Size Premium",       f"{w.size_premium*100:.2f}%"))
        rows_ke.append(("**Cost of Equity (Ke)**", f"**{w.cost_of_equity*100:.2f}%**"))
        st.table(pd.DataFrame(rows_ke, columns=["Item", "Value"]).set_index("Item"))

    with wc2:
        st.markdown("**Cost of Debt**")
        rows_kd = []
        if w.synthetic_rating:
            rows_kd.append(("Synthetic Rating",   w.synthetic_rating))
        rows_kd += [
            ("Pre-Tax Kd",        f"{w.pre_tax_cost_of_debt*100:.2f}%"),
            ("Tax Rate",          f"{w.tax_rate*100:.1f}%"),
            ("**After-Tax Kd**",  f"**{w.after_tax_cost_of_debt*100:.2f}%**"),
        ]
        st.table(pd.DataFrame(rows_kd, columns=["Item", "Value"]).set_index("Item"))

    with wc3:
        st.markdown("**Capital Structure & WACC**")
        rows_w = [
            ("Equity Weight (mkt)", f"{w.equity_weight*100:.1f}%"),
            ("Debt Weight (book)",  f"{w.debt_weight*100:.1f}%"),
            ("**WACC**",            f"**{w.wacc*100:.2f}%**"),
        ]
        st.table(pd.DataFrame(rows_w, columns=["Item", "Value"]).set_index("Item"))

    st.markdown("---")

    # ── FCF projections ────────────────────────────────────────────────────
    st.subheader("FCF Projections")
    proj_rows = []
    for py in dcf_result.projected_years:
        proj_rows.append({
            "Year":          py.calendar_year or py.year,
            "Revenue ($M)":  py.revenue,
            "Rev Growth":    f"{py.revenue_growth_rate*100:.1f}%",
            "EBITDA ($M)":   py.ebitda,
            "EBIT ($M)":     py.ebit,
            "FCF ($M)":      py.free_cash_flow,
            "PV(FCF) ($M)":  py.pv_free_cash_flow,
        })
    proj_df = pd.DataFrame(proj_rows).set_index("Year")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(x=proj_df.index, y=proj_df["Revenue ($M)"],
                    name="Revenue", marker_color="#3b82f6", opacity=0.6)
        fig.add_scatter(x=proj_df.index, y=proj_df["FCF ($M)"],
                        name="FCF", mode="lines+markers",
                        line=dict(color="#22c55e", width=2))
        fig.add_scatter(x=proj_df.index, y=proj_df["PV(FCF) ($M)"],
                        name="PV(FCF)", mode="lines+markers",
                        line=dict(color="#f59e0b", width=2, dash="dot"))
        fig.update_layout(title="Revenue vs FCF ($M)", height=320,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.dataframe(
            proj_df.style.format({
                "Revenue ($M)": "${:,.0f}",
                "EBITDA ($M)":  "${:,.0f}",
                "EBIT ($M)":    "${:,.0f}",
                "FCF ($M)":     "${:,.0f}",
                "PV(FCF) ($M)": "${:,.0f}",
            }),
            use_container_width=True,
        )

    st.markdown("---")

    # ── EV bridge ─────────────────────────────────────────────────────────
    st.subheader("EV → Equity Bridge")
    bc1, bc2 = st.columns([1, 2])
    with bc1:
        bridge_rows = [
            ("PV of Explicit FCFs",  b.pv_explicit_fcfs),
            ("PV of Terminal Value", b.pv_terminal_value),
            ("Enterprise Value",     b.enterprise_value),
            ("(-) Net Debt",         b.net_debt),
            ("(-) Minority Interest",b.minority_interest),
            ("Equity Value",         b.equity_value),
        ]
        bridge_df = pd.DataFrame(bridge_rows, columns=["Item", "$M"]).set_index("Item")
        st.dataframe(bridge_df.style.format("${:,.0f}"), use_container_width=True)
        tv_pct = tv.pv_as_pct_of_ev
        if tv_pct:
            st.caption(f"Terminal value = {tv_pct*100:.1f}% of EV  |  "
                       f"TV method: {tv.method.value}")

    with bc2:
        # Waterfall chart
        labels = ["PV FCFs", "PV TV", "Enterprise Value", "Net Debt", "Equity Value"]
        values = [
            b.pv_explicit_fcfs,
            b.pv_terminal_value,
            -(b.net_debt + b.minority_interest),
            0,
            b.equity_value,
        ]
        fig_wf = go.Figure(go.Waterfall(
            name="EV Bridge", orientation="v",
            measure=["relative", "relative", "total", "relative", "total"],
            x=["PV of FCFs", "PV of TV", "EV", "(-) Net Debt & MI", "Equity Value"],
            y=[b.pv_explicit_fcfs, b.pv_terminal_value,
               0, -(b.net_debt + b.minority_interest), 0],
            connector=dict(line=dict(color="rgba(63,63,70,0.3)")),
            decreasing=dict(marker_color="#ef4444"),
            increasing=dict(marker_color="#22c55e"),
            totals=dict(marker_color="#3b82f6"),
        ))
        fig_wf.update_layout(title="EV to Equity Waterfall ($M)", height=320)
        st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown("---")

    # ── Sensitivity table ─────────────────────────────────────────────────
    st.subheader(f"Sensitivity: WACC × {dcf_result.sensitivity.col_label}")
    sens = dcf_result.sensitivity
    curr = history_price_from_session()

    col_labels = [
        f"{v*100:.2f}%" if "Growth" in sens.col_label else f"{v:.1f}x"
        for v in sens.col_axis
    ]
    row_labels = [f"{v*100:.2f}%" for v in sens.row_axis]
    prices_matrix = [[p if p is not None else float("nan") for p in row]
                     for row in sens.prices]

    sens_df = pd.DataFrame(prices_matrix, index=row_labels, columns=col_labels)
    sens_df.index.name = "WACC \\ TGR"

    fig_heat = go.Figure(go.Heatmap(
        z=prices_matrix,
        x=col_labels, y=row_labels,
        colorscale="RdYlGn",
        text=[[f"${v:.0f}" if not pd.isna(v) else "N/A" for v in row]
              for row in prices_matrix],
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="Implied $"),
    ))
    base_wacc_label = f"{dcf_result.wacc_result.wacc*100:.2f}%"
    fig_heat.update_layout(
        title=f"Sensitivity Table — base WACC row: {base_wacc_label}",
        height=380,
        xaxis_title=sens.col_label,
        yaxis_title="WACC",
    )
    st.plotly_chart(fig_heat, use_container_width=True)


def _tab_comps(comps_result) -> None:
    if comps_result is None:
        st.info("No comparable data — add peer tickers in the sidebar and re-run.")
        return

    sub = comps_result.subject
    impl = comps_result.implied
    stats = comps_result.stats

    st.subheader(f"Peer Multiples  —  subject: {sub.ticker}")

    # Peer table
    peer_rows = []
    for c in comps_result.peers:
        m = c.multiples
        peer_rows.append({
            "Ticker": c.ticker,
            "EV/EBITDA": _fmt_x(m.ev_to_ebitda),
            "EV/Revenue": _fmt_x(m.ev_to_revenue),
            "P/E":        _fmt_x(m.price_to_earnings),
            "P/FCF":      _fmt_x(m.price_to_fcf),
            "P/B":        _fmt_x(m.price_to_book),
        })
    if peer_rows:
        st.dataframe(pd.DataFrame(peer_rows).set_index("Ticker"), use_container_width=True)

    st.markdown("---")

    # Implied prices
    st.subheader("Implied Share Price (from peer medians)")
    impl_rows = [
        ("EV/EBITDA", impl.implied_from_ev_ebitda),
        ("EV/Revenue", impl.implied_from_ev_revenue),
        ("P/E",       impl.implied_from_pe),
        ("P/FCF",     impl.implied_from_pfcf),
        ("P/B",       impl.implied_from_pb),
    ]
    valid = [(lbl, v) for lbl, v in impl_rows if v is not None]

    if valid:
        labels_v, values_v = zip(*valid)
        current = sub.multiples.current_price

        fig = go.Figure()
        fig.add_bar(x=list(labels_v), y=list(values_v),
                    marker_color="#3b82f6", name="Implied Price")
        if current:
            fig.add_hline(y=current, line_dash="dash",
                          line_color="#f59e0b",
                          annotation_text=f"Current ${current:.0f}")
        fig.update_layout(title="Implied Prices from Peer Multiples",
                          height=350, yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)

    # Median stats
    st.subheader("Peer Set Median Multiples")
    med_rows = []
    for name, stat in [
        ("EV/EBITDA", stats.ev_to_ebitda), ("EV/Revenue", stats.ev_to_revenue),
        ("P/E", stats.price_to_earnings), ("P/FCF", stats.price_to_fcf), ("P/B", stats.price_to_book),
    ]:
        if stat and stat.median is not None:
            med_rows.append({
                "Multiple": name, "P25": round(stat.p25 or 0, 2),
                "Median": round(stat.median, 2),
                "Mean": round(stat.mean or 0, 2),
                "P75": round(stat.p75 or 0, 2),
            })
    if med_rows:
        st.dataframe(pd.DataFrame(med_rows).set_index("Multiple"), use_container_width=True)


def _tab_scenario(scenario_result) -> None:
    if scenario_result is None:
        st.info("Scenario analysis did not run.")
        return

    st.subheader("Bull / Base / Bear Scenarios")
    scen_rows = []
    for sr in scenario_result.scenarios:
        a = sr.dcf_result.assumptions
        b = sr.dcf_result.bridge
        scen_rows.append({
            "Scenario":    sr.scenario.label,
            "Rev Growth Y1": _fmt_pct(a.revenue_growth_rates[0]),
            "EBITDA Margin": _fmt_pct(a.ebitda_margin),
            "WACC":          f"{sr.dcf_result.wacc_result.wacc*100:.2f}%",
            "TGR":           _fmt_pct(a.terminal_growth_rate),
            "Implied Price": f"${b.implied_share_price:,.2f}",
            "Upside":        _fmt_pct(b.upside_downside_pct),
        })
    st.dataframe(pd.DataFrame(scen_rows).set_index("Scenario"), use_container_width=True)

    # Bar chart
    prices = {sr.scenario.label: sr.dcf_result.bridge.implied_share_price
              for sr in scenario_result.scenarios}
    current = scenario_result.scenarios[0].dcf_result.bridge.current_price

    colors = {"Bull": "#22c55e", "Base": "#3b82f6", "Bear": "#ef4444"}
    fig = go.Figure()
    for label, price in prices.items():
        fig.add_bar(x=[label], y=[price],
                    marker_color=colors.get(label, "#94a3b8"),
                    text=[f"${price:,.0f}"], textposition="outside")
    if current:
        fig.add_hline(y=current, line_dash="dash", line_color="#f59e0b",
                      annotation_text=f"Current ${current:.0f}")
    fig.update_layout(title="Scenario Implied Prices", height=380, showlegend=False,
                      yaxis_title="Implied Share Price ($)")
    st.plotly_chart(fig, use_container_width=True)

    # Monte Carlo
    mc = scenario_result.monte_carlo
    if mc is not None:
        st.markdown("---")
        st.subheader("Monte Carlo Simulation")
        mc_cols = st.columns(4)
        mc_cols[0].metric("Simulations",   f"{mc.n_valid:,} / {mc.n_simulations:,}")
        mc_cols[1].metric("Median",        f"${mc.median_price:,.2f}")
        mc_cols[2].metric("5th / 95th",    f"${mc.pct_5:,.0f} – ${mc.pct_95:,.0f}")
        if mc.probability_above_current is not None:
            mc_cols[3].metric("P(price > current)", f"{mc.probability_above_current*100:.1f}%")

        pct_points = [mc.pct_5, mc.pct_10, mc.pct_25, mc.median_price, mc.pct_75, mc.pct_90, mc.pct_95]
        pct_labels = ["5th", "10th", "25th", "50th", "75th", "90th", "95th"]
        fig_mc = go.Figure()
        fig_mc.add_bar(x=pct_labels, y=pct_points, marker_color="#3b82f6",
                       text=[f"${v:,.0f}" for v in pct_points], textposition="outside")
        if mc.probability_above_current is not None and current:
            fig_mc.add_hline(y=current, line_dash="dash", line_color="#f59e0b",
                             annotation_text=f"Current ${current:.0f}")
        fig_mc.update_layout(title="Monte Carlo Percentile Distribution", height=350,
                             yaxis_title="Implied Price ($)")
        st.plotly_chart(fig_mc, use_container_width=True)


def _tab_forecast(rev_suite, earn_suite) -> None:
    if rev_suite is None:
        st.info("Revenue forecast did not run.")
        return

    st.subheader("Revenue Forecast")

    # Method comparison
    method_rows = []
    for r in rev_suite.results:
        mark = " (*)" if r.method == rev_suite.recommended.method else ""
        method_rows.append({
            "Method":    r.method_label + mark,
            "MAPE":      f"{r.mape*100:.2f}%" if r.mape is not None else "N/A",
            "R²":        f"{r.r_squared:.4f}" if r.r_squared is not None else "N/A",
            f"CAGR":     f"{r.cagr_projected*100:.1f}%" if r.cagr_projected else "N/A",
        })
    st.dataframe(pd.DataFrame(method_rows).set_index("Method"), use_container_width=True)
    st.caption("(*) recommended method")

    # Forecast chart with CI ribbon
    rec = rev_suite.recommended
    actuals = [p for p in rec.points if p.is_actual]
    projected = rec.projected_only

    fig = go.Figure()
    if actuals:
        fig.add_scatter(
            x=[p.year for p in actuals], y=[p.value for p in actuals],
            mode="lines+markers", name="Actual", line=dict(color="#3b82f6", width=2),
        )
    proj_x = [p.year for p in projected]
    proj_y = [p.value for p in projected]
    proj_lo = [p.lower_95 for p in projected]
    proj_hi = [p.upper_95 for p in projected]

    # CI ribbon
    if all(v is not None for v in proj_lo + proj_hi):
        fig.add_scatter(
            x=proj_x + proj_x[::-1],
            y=proj_hi + proj_lo[::-1],
            fill="toself", fillcolor="rgba(34,197,94,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        )
    fig.add_scatter(
        x=proj_x, y=proj_y, mode="lines+markers",
        name=f"Forecast ({rec.method_label})",
        line=dict(color="#22c55e", width=2),
    )
    fig.update_layout(
        title=f"Revenue Forecast — {rev_suite.ticker}  (base ${rev_suite.base_revenue:,.0f}M)",
        height=380, yaxis_title="Revenue ($M)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Earnings
    if earn_suite is not None:
        st.markdown("---")
        st.subheader("Earnings Forecast (EPS)")
        erec = earn_suite.recommended
        earn_rows = []
        for p in erec.projected_only:
            earn_rows.append({
                "Year": p.year,
                "Revenue ($M)": p.revenue,
                "Net Income ($M)": p.net_income,
                "EPS": p.eps,
                "EPS Low (95%)": p.eps_lower_95,
                "EPS High (95%)": p.eps_upper_95,
            })
        earn_df = pd.DataFrame(earn_rows).set_index("Year")
        st.dataframe(
            earn_df.style.format({
                "Revenue ($M)":    "${:,.0f}",
                "Net Income ($M)": "${:,.0f}",
                "EPS":             "${:.2f}",
                "EPS Low (95%)":   "${:.2f}",
                "EPS High (95%)":  "${:.2f}",
            }),
            use_container_width=True,
        )
        st.caption(
            f"Method: {erec.method_label}  |  "
            f"EPS CAGR: {erec.cagr_eps*100:.1f}%  |  "
            f"Avg EBITDA margin: {erec.avg_ebitda_margin*100:.1f}%"
        )


# ── session state helper ──────────────────────────────────────────────────────

def history_price_from_session() -> Optional[float]:
    """Retrieve current price from session state for sensitivity annotation."""
    return st.session_state.get("current_price")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = _sidebar()

    if not cfg["ticker"]:
        st.info("Enter a ticker in the sidebar and click **Run Analysis**.")
        return

    # Auto-run on first load or when explicitly clicked
    run = cfg["run_btn"] or "dcf_result" not in st.session_state

    if run:
        status = st.status(f"Analysing {cfg['ticker']}…", expanded=True)

        try:
            with status:
                st.write("Fetching financial data…")
                history = _fetch_history(cfg["ticker"])
                st.session_state["current_price"] = history.profile.current_price

                st.write("Running DCF…")
                dcf_result = _run_dcf(
                    cfg["ticker"],
                    cfg["wacc_override"],
                    cfg["tgr"],
                    cfg["proj_years"],
                )

                comps_result = None
                if cfg["peers"]:
                    st.write(f"Running comps ({', '.join(cfg['peers'])})…")
                    try:
                        comps_result = _run_comps(cfg["ticker"], cfg["peers"])
                    except Exception as e:
                        st.warning(f"Comps skipped: {e}")

                st.write("Running scenario analysis…")
                try:
                    scenario_result = _run_scenario(
                        cfg["ticker"], cfg["run_mc"], cfg["mc_n"],
                        bull_growth=cfg["bull_growth"],
                        bull_margin=cfg["bull_margin"],
                        bear_growth=cfg["bear_growth"],
                        bear_margin=cfg["bear_margin"],
                    )
                except Exception as e:
                    st.warning(f"Scenario skipped: {e}")
                    scenario_result = None

                st.write("Running revenue forecast…")
                try:
                    rev_suite, earn_suite = _run_forecast(cfg["ticker"], cfg["proj_years"])
                except Exception as e:
                    st.warning(f"Forecast skipped: {e}")
                    rev_suite, earn_suite = None, None

                st.session_state.update(dict(
                    history=history, dcf_result=dcf_result,
                    comps_result=comps_result, scenario_result=scenario_result,
                    rev_suite=rev_suite, earn_suite=earn_suite,
                ))
            status.update(label="Analysis complete", state="complete")

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            return

    if "dcf_result" not in st.session_state:
        return

    history       = st.session_state["history"]
    dcf_result    = st.session_state["dcf_result"]
    comps_result  = st.session_state.get("comps_result")
    scenario_result = st.session_state.get("scenario_result")
    rev_suite     = st.session_state.get("rev_suite")
    earn_suite    = st.session_state.get("earn_suite")

    # Show upside / downside headline
    b = dcf_result.bridge
    if b.current_price:
        up = b.upside_downside_pct or 0
        color = "green" if up >= 0 else "red"
        sign = "+" if up >= 0 else ""
        st.markdown(
            f"**{history.profile.name}** — Implied **${b.implied_share_price:,.2f}**  "
            f"vs current **${b.current_price:,.2f}**  "
            f"<span style='color:{color};font-weight:600'>{sign}{up*100:.1f}%</span>",
            unsafe_allow_html=True,
        )

    tabs = st.tabs(["Overview", "DCF Valuation", "Comparable Valuation",
                    "Scenario Analysis", "Revenue & Earnings Forecast"])

    with tabs[0]:
        _tab_overview(history, dcf_result)
    with tabs[1]:
        _tab_dcf(dcf_result)
    with tabs[2]:
        _tab_comps(comps_result)
    with tabs[3]:
        _tab_scenario(scenario_result)
    with tabs[4]:
        _tab_forecast(rev_suite, earn_suite)


if __name__ == "__main__":
    main()
