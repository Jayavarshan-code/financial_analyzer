"""
main.py
--------
CLI entry point for the DCF Valuation Engine.

Commands:
  analyze     – Financial Statement Analyzer: ratios, margins, growth summary
  dcf         – DCF Engine: WACC, FCF projection, implied price, sensitivity table
  comps       – Comparable Valuation Engine: peer multiples and implied prices
  scenario    – Scenario Analysis: Bull/Base/Bear DCF + optional Monte Carlo
  assumptions – Show / diff the assumption log for a ticker
  report      – Generate a full HTML / PDF valuation report
  forecast    – Revenue + Earnings forecasting (Phase 2)

Usage:
    python main.py analyze AAPL
    python main.py dcf AAPL
    python main.py comps AAPL MSFT GOOGL META AMZN
    python main.py scenario AAPL --mc
    python main.py forecast AAPL
    python main.py forecast AAPL --years 7 --earnings
    python main.py report AAPL --peers MSFT GOOGL --mc
"""

from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console

from core.financial_statements import FinancialStatementAnalyzer
from core.dcf_engine import DCFEngine
from core.comps_engine import CompsEngine
from core.scenario_analysis import ScenarioAnalyzer
from core.assumption_tracker import AssumptionTracker
from core.report_generator import ReportGenerator
from analytics.revenue_forecaster import RevenueForecastEngine
from analytics.earnings_forecaster import EarningsForecastEngine
from data.models.dcf import DCFAssumptions, WACCInputs, TerminalValueMethod
from data.models.forecast import ForecastMethod
from data.models.report import ReportConfig, ReportFormat, ReportInput
from data.models.scenario import MonteCarloConfig

app = typer.Typer(help="DCF Valuation Engine CLI", no_args_is_help=True)
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------

@app.command()
def analyze(
    ticker: str = typer.Argument(..., help="Stock ticker symbol, e.g. AAPL"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Fetch financials and display ratios / growth summary for TICKER."""
    _setup_logging(verbose)

    with console.status(f"[bold green]Fetching financials for {ticker.upper()}..."):
        try:
            analyzer = FinancialStatementAnalyzer()
            history = analyzer.analyze(ticker)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)

    console.print(analyzer.summary(history))


# ---------------------------------------------------------------------------
# dcf command
# ---------------------------------------------------------------------------

@app.command()
def dcf(
    ticker: str = typer.Argument(..., help="Stock ticker symbol, e.g. AAPL"),

    # WACC controls
    wacc: Optional[float] = typer.Option(None, "--wacc", help="Override WACC directly, e.g. 0.09"),
    beta: Optional[float] = typer.Option(None, "--beta", help="Equity beta (used if --wacc not set)"),
    rfr: float = typer.Option(0.045, "--rfr", help="Risk-free rate, e.g. 0.045"),
    erp: float = typer.Option(0.055, "--erp", help="Equity risk premium, e.g. 0.055"),
    debt_weight: float = typer.Option(0.20, "--dw", help="Debt weight in capital structure"),

    # Projection controls
    years: int = typer.Option(5, "--years", "-n", help="Projection years (1–20)"),
    growth: Optional[float] = typer.Option(None, "--growth", "-g",
        help="Flat revenue growth rate for all years, e.g. 0.08. "
             "If omitted, auto-derived from historical CAGR."),

    # Terminal value
    tgr: float = typer.Option(0.025, "--tgr", help="Terminal growth rate (Gordon Growth)"),
    method: TerminalValueMethod = typer.Option(
        TerminalValueMethod.GORDON_GROWTH, "--method",
        help="Terminal value method: gordon_growth | exit_multiple",
    ),
    exit_multiple: Optional[float] = typer.Option(
        None, "--exit-multiple", help="Exit EV/EBITDA multiple (required if --method exit_multiple)"
    ),

    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Run a full DCF valuation for TICKER.

    Auto-derives assumptions from historical financials unless overrides are provided.
    Prints WACC build-up, FCF projection, EV bridge, and 9×9 sensitivity table.
    """
    _setup_logging(verbose)

    # --- Fetch financials ---
    with console.status(f"[bold green]Fetching financials for {ticker.upper()}..."):
        try:
            analyzer = FinancialStatementAnalyzer()
            history = analyzer.analyze(ticker)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)

    # --- Build assumptions (None = auto-derive) ---
    assumptions: Optional[DCFAssumptions] = None

    if any(v is not None for v in [wacc, beta, growth, exit_multiple]) or method != TerminalValueMethod.GORDON_GROWTH:
        latest = history.latest
        net_debt = latest.statements.balance_sheet.net_debt
        shares = (
            latest.statements.income_statement.shares_diluted
            or history.profile.shares_outstanding
        )

        wacc_inputs = WACCInputs(
            risk_free_rate=rfr,
            beta=beta or history.profile.beta or 1.0,
            equity_risk_premium=erp,
            debt_weight=debt_weight,
        )

        growth_rates = [growth] * years if growth is not None else None

        if growth_rates is None:
            # Use auto-derived via engine internals — pass None assumptions
            # and let the engine fill in growth; only override what was specified
            engine = DCFEngine()
            auto = engine._derive_assumptions(history)
            growth_rates = auto.revenue_growth_rates[:years]
            if len(growth_rates) < years:
                growth_rates += [growth_rates[-1]] * (years - len(growth_rates))

        assumptions = DCFAssumptions(
            projection_years=years,
            revenue_growth_rates=growth_rates,
            wacc_override=wacc,
            wacc_inputs=wacc_inputs,
            terminal_growth_rate=tgr,
            terminal_value_method=method,
            exit_ev_ebitda_multiple=exit_multiple,
            net_debt=net_debt,
            shares_outstanding=shares,
        )

    # --- Run DCF ---
    with console.status("[bold green]Running DCF..."):
        try:
            engine = DCFEngine()
            result = engine.run(history, assumptions=assumptions)
        except ValueError as exc:
            console.print(f"[red]DCF Error:[/red] {exc}")
            raise typer.Exit(code=1)

    console.print(engine.summary(result))


# ---------------------------------------------------------------------------
# comps command
# ---------------------------------------------------------------------------

@app.command()
def comps(
    subject: str = typer.Argument(..., help="Subject ticker to value, e.g. AAPL"),
    peers: list[str] = typer.Argument(..., help="Peer ticker(s), e.g. MSFT GOOGL META AMZN"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Run a comparable company analysis for SUBJECT vs PEERS.

    Fetches live market data and TTM financials for all tickers, computes
    EV/EBITDA, EV/Revenue, P/E, P/FCF, and P/B multiples across the peer set,
    and back-solves implied share prices for the subject from peer-set medians.
    """
    _setup_logging(verbose)

    all_tickers = [subject.upper()] + [p.upper() for p in peers]
    with console.status(f"[bold green]Fetching data for {', '.join(all_tickers)}..."):
        try:
            engine = CompsEngine()
            result = engine.run(subject, peers)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)

    console.print(engine.summary(result))


# ---------------------------------------------------------------------------
# scenario command
# ---------------------------------------------------------------------------

@app.command()
def scenario(
    ticker: str = typer.Argument(..., help="Stock ticker symbol, e.g. AAPL"),
    mc: bool = typer.Option(False, "--mc", help="Run Monte Carlo simulation"),
    mc_n: int = typer.Option(1000, "--mc-n", help="Number of Monte Carlo simulations"),
    mc_seed: Optional[int] = typer.Option(None, "--mc-seed", help="RNG seed for reproducible MC"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Run Bull / Base / Bear scenario analysis for TICKER.

    Auto-derives base assumptions from historical financials, then applies
    +/-3 pp revenue growth, +/-2 pp EBITDA margin, and +/-50 bp WACC deltas
    to construct Bull and Bear cases. Optionally runs Monte Carlo simulation.
    """
    _setup_logging(verbose)

    with console.status(f"[bold green]Fetching financials for {ticker.upper()}..."):
        try:
            analyzer_fs = FinancialStatementAnalyzer()
            history = analyzer_fs.analyze(ticker)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)

    mc_config = MonteCarloConfig(n_simulations=mc_n, seed=mc_seed) if mc else None

    status_msg = "[bold green]Running scenarios + Monte Carlo..." if mc else "[bold green]Running scenarios..."
    with console.status(status_msg):
        try:
            sa = ScenarioAnalyzer()
            result = sa.run(history, mc_config=mc_config)
        except ValueError as exc:
            console.print(f"[red]Scenario Error:[/red] {exc}")
            raise typer.Exit(code=1)

    console.print(sa.summary(result))


# ---------------------------------------------------------------------------
# assumptions command
# ---------------------------------------------------------------------------

@app.command()
def assumptions(
    ticker: str = typer.Argument(..., help="Stock ticker symbol, e.g. AAPL"),
    diff: bool = typer.Option(False, "--diff", "-d", help="Show diff between last two logged runs"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Show the assumption log for TICKER.

    Displays a table of every DCF run that was logged for this ticker,
    including the implied price and WACC from each run. Use --diff to
    print a field-level comparison of the two most recent entries.
    """
    _setup_logging(verbose)
    tracker = AssumptionTracker()
    console.print(tracker.summary(ticker))
    if diff:
        d = tracker.diff_latest(ticker)
        if d is None:
            console.print("[yellow]Need at least 2 logged entries to diff.[/yellow]")
        else:
            console.print(tracker.diff_summary(d))


# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------

@app.command()
def report(
    ticker: str = typer.Argument(..., help="Stock ticker symbol, e.g. AAPL"),
    peers: Optional[list[str]] = typer.Option(None, "--peers", "-p",
        help="Peer tickers for comparable valuation section, e.g. --peers MSFT GOOGL"),
    mc: bool = typer.Option(False, "--mc", help="Include Monte Carlo in scenario section"),
    mc_n: int = typer.Option(1000, "--mc-n", help="Number of Monte Carlo simulations"),
    format: ReportFormat = typer.Option(ReportFormat.HTML, "--format", "-f",
        help="Output format: html | pdf | both"),
    no_sensitivity: bool = typer.Option(False, "--no-sensitivity",
        help="Omit sensitivity table from report"),
    log_assumptions: bool = typer.Option(True, "--log/--no-log",
        help="Log assumptions to the assumption tracker"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Generate a full HTML (and optionally PDF) valuation report for TICKER.

    Runs the Financial Statement Analyzer, DCF Engine, and optionally the
    Comparable Valuation Engine (if --peers provided) and Scenario Analysis
    (always included). Outputs to the reports/ directory.
    """
    _setup_logging(verbose)

    with console.status(f"[bold green]Fetching financials for {ticker.upper()}..."):
        try:
            analyzer_fs = FinancialStatementAnalyzer()
            history = analyzer_fs.analyze(ticker)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)

    with console.status("[bold green]Running DCF..."):
        try:
            dcf_engine = DCFEngine()
            dcf_result = dcf_engine.run(history)
        except ValueError as exc:
            console.print(f"[red]DCF Error:[/red] {exc}")
            raise typer.Exit(code=1)

    # Log assumptions before anything else fails
    if log_assumptions:
        try:
            AssumptionTracker().log(
                ticker,
                dcf_result.assumptions,
                label="report",
                implied_price=dcf_result.bridge.implied_share_price,
                wacc_effective=dcf_result.wacc_result.wacc,
            )
        except Exception:
            pass  # Never let tracking failure abort the report

    comps_result = None
    if peers:
        with console.status(f"[bold green]Fetching comps ({', '.join(peers)})..."):
            try:
                comps_result = CompsEngine().run(ticker, list(peers))
            except ValueError as exc:
                console.print(f"[yellow]Comps skipped:[/yellow] {exc}")

    mc_config = MonteCarloConfig(n_simulations=mc_n) if mc else None
    with console.status("[bold green]Running scenarios..."):
        try:
            scenario_result = ScenarioAnalyzer().run(history, mc_config=mc_config)
        except ValueError as exc:
            console.print(f"[yellow]Scenario analysis skipped:[/yellow] {exc}")
            scenario_result = None

    with console.status("[bold green]Generating report..."):
        rpt_input = ReportInput(
            history=history,
            dcf_result=dcf_result,
            comps_result=comps_result,
            scenario_result=scenario_result,
        )
        cfg = ReportConfig(
            format=format,
            include_sensitivity_table=not no_sensitivity,
        )
        out = ReportGenerator().generate(rpt_input, cfg)

    if out.html_path:
        console.print(f"[green]HTML report:[/green] {out.html_path}")
    if out.pdf_path:
        console.print(f"[green]PDF report:[/green]  {out.pdf_path}")
    if out.pdf_error:
        console.print(f"[yellow]PDF skipped:[/yellow] {out.pdf_error}")


# ---------------------------------------------------------------------------
# forecast command
# ---------------------------------------------------------------------------

@app.command()
def forecast(
    ticker: str = typer.Argument(..., help="Stock ticker symbol, e.g. AAPL"),
    years: int = typer.Option(5, "--years", "-n", help="Projection horizon (1–20)"),
    method: Optional[str] = typer.Option(
        None, "--method", "-m",
        help="Force a single method: cagr | linear_trend | exponential_trend | holt_winters | ensemble"
    ),
    earnings: bool = typer.Option(False, "--earnings", "-e",
        help="Also run earnings forecast (net income + EPS)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Run revenue (and optionally earnings) forecasting for TICKER.

    Fits CAGR, linear trend, exponential trend, Holt-Winters, and ensemble
    models to historical revenue.  Prints a method comparison table and the
    recommended forecast with 95% confidence intervals.

    The recommended forecast's growth rates can feed directly into DCFAssumptions.
    """
    _setup_logging(verbose)

    with console.status(f"[bold green]Fetching financials for {ticker.upper()}..."):
        try:
            analyzer_fs = FinancialStatementAnalyzer()
            history = analyzer_fs.analyze(ticker)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)

    # Resolve method filter
    methods_filter = None
    if method:
        try:
            methods_filter = [ForecastMethod(method.lower())]
        except ValueError:
            valid = ", ".join(m.value for m in ForecastMethod)
            console.print(f"[red]Unknown method '{method}'. Valid: {valid}[/red]")
            raise typer.Exit(code=1)

    with console.status("[bold green]Fitting revenue forecast models..."):
        try:
            rev_engine = RevenueForecastEngine()
            rev_suite = rev_engine.run(history, n_years=years, methods=methods_filter)
        except ValueError as exc:
            console.print(f"[red]Forecast Error:[/red] {exc}")
            raise typer.Exit(code=1)

    console.print(rev_engine.summary(rev_suite))

    if earnings:
        with console.status("[bold green]Running earnings forecast..."):
            try:
                earn_engine = EarningsForecastEngine()
                earn_suite = earn_engine.run(history, rev_suite.recommended)
                console.print(earn_engine.summary(earn_suite))
            except ValueError as exc:
                console.print(f"[yellow]Earnings forecast skipped:[/yellow] {exc}")


if __name__ == "__main__":
    app()
