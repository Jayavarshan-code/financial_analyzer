"""
config.py
---------
Central configuration for the DCF Valuation Engine.

Loads settings from environment variables (or a .env file at project root).
All modules import from here — never hardcode API keys or tunable constants
directly in module files.

Environment variables (set in .env or shell):
  FMP_API_KEY      Financial Modeling Prep API key (optional; yfinance used by default)
  DEFAULT_CURRENCY  ISO 4217 code, e.g. "USD"
  REPORTS_DIR      Absolute path where generated reports are written
  LOG_LEVEL        Python logging level: DEBUG | INFO | WARNING | ERROR
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Data providers ---
    fmp_api_key: str = ""                          # leave blank to use yfinance only
    default_data_provider: str = "yfinance"        # "yfinance" | "fmp"

    # --- Currency & locale ---
    default_currency: str = "USD"

    # --- DCF defaults (overridable per analysis) ---
    default_projection_years: int = 5
    default_terminal_growth_rate: float = 0.025    # 2.5%
    default_risk_free_rate: float = 0.045          # 10-yr US Treasury proxy
    default_equity_risk_premium: float = 0.055     # Damodaran ERP

    # --- Paths ---
    base_dir: Path = Path(__file__).parent
    reports_dir: Path = base_dir / "reports"
    data_cache_dir: Path = base_dir / ".cache"

    # --- Logging ---
    log_level: str = "INFO"


settings = Settings()

# Ensure output directories exist at import time
settings.reports_dir.mkdir(parents=True, exist_ok=True)
settings.data_cache_dir.mkdir(parents=True, exist_ok=True)
