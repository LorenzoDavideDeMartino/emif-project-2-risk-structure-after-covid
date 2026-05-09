from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs" / "project2"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
DATA_CANDIDATES = [
    DATA_DIR / "Data.xlsx",
    DATA_DIR / "Data (1).xlsx",
]

COVID_BREAK = "2020-03-11"
POST_COVID_START = "2020-01-01"
ANALYSIS_START = "2000-01-01"
TRADING_DAYS = 252
VAR_LEVEL = 0.05
ACF_LAGS = 20
ROLLING_VAR_WINDOW = 21

COLUMN_MAP = {
    "S&P500": "sp500",
    "Eurostoxx 50": "eurostoxx50",
    "Hang Seng": "hang_seng",
    "MSCI EM": "msci_em",
    "SMI": "smi",
    "US T 10-year Yield": "ust10y_yield",
    "German Gov 10-year yield": "bund10y_yield",
    "Oil futures": "oil",
    "Gold": "gold",
    "EURUSD": "eurusd",
    "USDJPY": "usdjpy",
    "US IG Bonds": "us_ig_bonds",
    "US HY Bonds": "us_hy_bonds",
    "USDCHF": "usdchf",
}

PRICE_COLUMNS = [
    "sp500",
    "eurostoxx50",
    "hang_seng",
    "msci_em",
    "smi",
    "oil",
    "gold",
    "eurusd",
    "usdjpy",
    "us_ig_bonds",
    "us_hy_bonds",
    "usdchf",
]

YIELD_COLUMNS = [
    "ust10y_yield",
    "bund10y_yield",
]

DISPLAY_NAMES = {
    "sp500": "S&P 500",
    "eurostoxx50": "Euro Stoxx 50",
    "hang_seng": "Hang Seng",
    "msci_em": "MSCI EM",
    "smi": "SMI",
    "ust10y_yield": "US 10Y yield change (bp)",
    "bund10y_yield": "German 10Y yield change (bp)",
    "oil": "Oil futures",
    "gold": "Gold",
    "eurusd": "EURUSD",
    "usdjpy": "USDJPY",
    "us_ig_bonds": "US IG bonds",
    "us_hy_bonds": "US HY bonds",
    "usdchf": "USDCHF",
}
