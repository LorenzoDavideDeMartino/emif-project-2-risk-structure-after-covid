from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import normal_ad

from project2_config import (
    DATA_CANDIDATES,
    TABLE_DIR,
    FIGURE_DIR,
    COLUMN_MAP,
    PRICE_COLUMNS,
    YIELD_COLUMNS,
    ALL_SERIES_COLUMNS,
    DISPLAY_NAMES,
    COVID_BREAK,
    POST_COVID_START,
    ANALYSIS_START,
    TRADING_DAYS,
    VAR_LEVEL,
    ACF_LAGS,
    ROLLING_VAR_WINDOW,
)


def ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def resolve_data_path() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No Excel workbook found. Expected data/raw/Data.xlsx (or Data (1).xlsx locally)."
    )


def load_raw_data() -> pd.DataFrame:
    data_path = resolve_data_path()
    raw_data = pd.read_excel(data_path)
    raw_data = raw_data.rename(columns={raw_data.columns[0]: "date"})
    raw_data["date"] = pd.to_datetime(raw_data["date"])
    raw_data = raw_data.rename(columns=COLUMN_MAP)
    raw_data = raw_data.sort_values("date").reset_index(drop=True)
    expected_columns = {"date", *ALL_SERIES_COLUMNS}
    missing_columns = expected_columns.difference(raw_data.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"The Excel file is missing the following required columns: {missing_list}")
    return raw_data


def build_aligned_returns(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # We keep the original levels for visual inspection, then build transformed series for econometrics.
    level_data = raw_data.copy()
    transformed_data = pd.DataFrame({"date": level_data["date"]})

    for column in PRICE_COLUMNS:
        # Oil futures turn negative in April 2020, so a standard log return is undefined there.
        # We therefore use simple percentage returns for oil, while keeping log returns for the other price series.
        if column == "oil":
            transformed_data[column] = 100.0 * level_data[column].pct_change()
        else:
            transformed_data[column] = 100.0 * np.log(level_data[column]).diff()

    for column in YIELD_COLUMNS:
        transformed_data[column] = 100.0 * level_data[column].diff()

    transformed_data = transformed_data.loc[transformed_data["date"] >= pd.Timestamp(ANALYSIS_START)].copy()
    transformed_data = transformed_data.dropna(how="any").reset_index(drop=True)
    return level_data, transformed_data


def split_pre_post(aligned_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # We keep the event date (11 March 2020) for break tests and figures,
    # but the sample split itself follows the project brief: 2000-2019 vs 2020 onward.
    sample_split = pd.Timestamp(POST_COVID_START)
    pre_covid = aligned_returns.loc[aligned_returns["date"] < sample_split].copy()
    post_covid = aligned_returns.loc[aligned_returns["date"] >= sample_split].copy()
    return pre_covid, post_covid


def expected_shortfall(series: pd.Series, alpha: float = VAR_LEVEL) -> float:
    cutoff = series.quantile(alpha)
    tail = series.loc[series <= cutoff]
    return float(tail.mean()) if len(tail) > 0 else np.nan


def max_drawdown(return_series: pd.Series) -> float:
    cumulative_wealth = np.exp(return_series.dropna().cumsum() / 100.0)
    running_peak = cumulative_wealth.cummax()
    drawdown = cumulative_wealth / running_peak - 1.0
    return float(drawdown.min())


def describe_one_series(series: pd.Series, asset_name: str, sample_name: str) -> dict:
    # For prices, drawdown has an economic meaning. For yield changes, it does not, so we leave it blank.
    jb_stat, jb_pvalue = stats.jarque_bera(series)
    with np.errstate(divide="ignore", invalid="ignore"):
        ad_stat, ad_pvalue = normal_ad(series)

    row = {
        "sample": sample_name,
        "asset": asset_name,
        "n_obs": int(series.shape[0]),
        "mean_daily": series.mean(),
        "std_daily": series.std(),
        "skewness": series.skew(),
        "excess_kurtosis": series.kurt(),
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_pvalue,
        "anderson_darling_stat": ad_stat,
        "anderson_darling_pvalue": ad_pvalue,
        "var_5pct": series.quantile(VAR_LEVEL),
        "es_5pct": expected_shortfall(series, VAR_LEVEL),
        "max_drawdown": np.nan,
    }

    if asset_name not in YIELD_COLUMNS:
        row["max_drawdown"] = max_drawdown(series)

    return row


def build_stylized_facts_table(pre_covid: pd.DataFrame, post_covid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_name, sample in [("Pre-COVID", pre_covid), ("Post-COVID", post_covid)]:
        for asset_name in ALL_SERIES_COLUMNS:
            rows.append(describe_one_series(sample[asset_name].dropna(), asset_name, sample_name))
    stylized_facts = pd.DataFrame(rows)
    stylized_facts["asset_label"] = stylized_facts["asset"].map(DISPLAY_NAMES)
    return stylized_facts


def variance_chow_test(variance_proxy: pd.Series, break_date: str) -> dict:
    # We test for a break in the mean of squared returns, which is the standard daily proxy for return variance.
    variance_proxy = variance_proxy.dropna().copy()
    break_timestamp = pd.Timestamp(break_date)
    pre = variance_proxy.loc[variance_proxy.index < break_timestamp]
    post = variance_proxy.loc[variance_proxy.index >= break_timestamp]

    if len(pre) < 30 or len(post) < 30:
        raise ValueError("Not enough observations on one side of the break date.")

    full_mean = variance_proxy.mean()
    sse_full = ((variance_proxy - full_mean) ** 2).sum()
    sse_pre = ((pre - pre.mean()) ** 2).sum()
    sse_post = ((post - post.mean()) ** 2).sum()

    k = 1
    n_total = len(variance_proxy)
    f_stat = ((sse_full - (sse_pre + sse_post)) / k) / ((sse_pre + sse_post) / (n_total - 2 * k))
    p_value = 1.0 - stats.f.cdf(f_stat, k, n_total - 2 * k)

    return {
        "break_date": break_timestamp,
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "pre_variance_mean": float(pre.mean()),
        "post_variance_mean": float(post.mean()),
        "n_pre": int(len(pre)),
        "n_post": int(len(post)),
    }


def quandt_andrews_style_variance_test(variance_proxy: pd.Series, trim: float = 0.15) -> pd.DataFrame:
    # A Quandt-Andrews style sup-F scan checks many possible break dates instead of imposing only one candidate date.
    variance_proxy = variance_proxy.dropna().copy()
    n_obs = len(variance_proxy)
    start = int(n_obs * trim)
    end = int(n_obs * (1.0 - trim))

    full_mean = variance_proxy.mean()
    sse_full = ((variance_proxy - full_mean) ** 2).sum()
    rows = []

    for split in range(start, end):
        pre = variance_proxy.iloc[:split]
        post = variance_proxy.iloc[split:]
        sse_pre = ((pre - pre.mean()) ** 2).sum()
        sse_post = ((post - post.mean()) ** 2).sum()
        k = 1
        f_stat = ((sse_full - (sse_pre + sse_post)) / k) / ((sse_pre + sse_post) / (n_obs - 2 * k))
        rows.append({
            "break_date": variance_proxy.index[split],
            "f_stat": float(f_stat),
            "p_value_chow_style": float(1.0 - stats.f.cdf(f_stat, k, n_obs - 2 * k)),
        })

    return pd.DataFrame(rows).sort_values("f_stat", ascending=False).reset_index(drop=True)


def bai_perron_style_breaks(variance_proxy: pd.Series, n_breaks: int = 3, min_size: int = 63) -> pd.DataFrame:
    # The ruptures package gives a practical Bai-Perron-style multiple break search on the variance proxy.
    variance_proxy = variance_proxy.dropna().copy()
    signal = variance_proxy.to_numpy().reshape(-1, 1)
    algo = rpt.Binseg(model="l2", min_size=min_size).fit(signal)
    break_indices = algo.predict(n_bkps=n_breaks)

    rows = []
    for break_index in break_indices:
        if break_index < len(variance_proxy):
            rows.append({
                "break_index": int(break_index),
                "break_date": variance_proxy.index[break_index - 1],
            })
    return pd.DataFrame(rows)


def plot_level_panel(level_data: pd.DataFrame, columns: Iterable[str], title: str) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for axis, column in zip(axes.flatten(), columns):
        axis.plot(level_data["date"], level_data[column], lw=1.2)
        axis.set_title(DISPLAY_NAMES[column])
        axis.grid(alpha=0.2)
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    return figure


def plot_correlation_heatmaps(pre_covid: pd.DataFrame, post_covid: pd.DataFrame) -> plt.Figure:
    pre_corr = pre_covid.drop(columns="date").corr()
    post_corr = post_covid.drop(columns="date").corr()

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(pre_corr, cmap="coolwarm", center=0.0, ax=axes[0])
    axes[0].set_title("Pre-COVID correlation matrix")
    sns.heatmap(post_corr, cmap="coolwarm", center=0.0, ax=axes[1])
    axes[1].set_title("Post-COVID correlation matrix")
    figure.tight_layout()
    return figure


def plot_acf_grid(aligned_returns: pd.DataFrame, title: str, squared: bool = False) -> plt.Figure:
    columns = [column for column in ALL_SERIES_COLUMNS if column in aligned_returns.columns]
    figure, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()

    for axis, column in zip(axes, columns):
        series = aligned_returns[column].copy()
        if squared:
            series = series ** 2
        plot_acf(series, ax=axis, lags=ACF_LAGS, zero=False)
        axis.set_title(DISPLAY_NAMES[column], fontsize=9)
        axis.grid(alpha=0.2)

    for axis in axes[len(columns):]:
        axis.axis("off")

    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    return figure


def plot_variance_breaks(sp500_returns: pd.Series, covid_break: str, break_table: pd.DataFrame) -> plt.Figure:
    variance_proxy = (sp500_returns ** 2).rolling(ROLLING_VAR_WINDOW).mean() * TRADING_DAYS
    variance_proxy = variance_proxy.dropna()

    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(variance_proxy.index, variance_proxy, color="navy", lw=1.2, label=f"{ROLLING_VAR_WINDOW}-day rolling variance")
    axis.axvline(pd.Timestamp(covid_break), color="red", linestyle="--", lw=1.5, label="11 March 2020")

    for _, row in break_table.iterrows():
        axis.axvline(pd.Timestamp(row["break_date"]), color="darkorange", linestyle=":", alpha=0.8)

    axis.set_title("S&P 500 variance proxy and structural break candidates")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def save_dataframe(dataframe: pd.DataFrame, filename: str) -> None:
    ensure_output_dirs()
    dataframe.to_csv(TABLE_DIR / filename, index=False)


def save_figure(figure: plt.Figure, filename: str) -> None:
    ensure_output_dirs()
    figure.savefig(FIGURE_DIR / filename, dpi=200, bbox_inches="tight")
