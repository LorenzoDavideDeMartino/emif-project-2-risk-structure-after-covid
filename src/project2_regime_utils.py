from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from project2_config import COVID_BREAK, POST_COVID_START, TABLE_DIR, TRADING_DAYS
from project2_data_utils import ensure_output_dirs, load_raw_data, build_aligned_returns
from project2_multivariate_utils import rolling_correlation_table, ROLLING_WINDOW

SPX_UST_PAIR = ("sp500", "ust10y_yield")
ROLLING_BREAK_WINDOW = 63
MAX_BREAKS = 3


def load_regime_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_data = load_raw_data()
    _, aligned_returns = build_aligned_returns(raw_data)

    # The regime analysis focuses on two state variables: equity variance and equity/rate dependence.
    rolling_variance = aligned_returns[["date", "sp500"]].copy()
    rolling_variance["sp500_rolling_variance"] = (
        rolling_variance["sp500"].rolling(ROLLING_BREAK_WINDOW).var() * TRADING_DAYS
    )
    rolling_variance = rolling_variance.dropna().reset_index(drop=True)

    rolling_corr_table = rolling_correlation_table(aligned_returns, [SPX_UST_PAIR], window=ROLLING_WINDOW)
    spx_ust_corr = rolling_corr_table.loc[
        (rolling_corr_table["left"] == SPX_UST_PAIR[0]) & (rolling_corr_table["right"] == SPX_UST_PAIR[1]),
        ["date", "rolling_corr"],
    ].rename(columns={"rolling_corr": "spx_ust_rolling_corr"}).reset_index(drop=True)

    return rolling_variance, spx_ust_corr


def fit_two_state_markov(correlation_data: pd.DataFrame):
    # A two-state Markov model is a direct way to ask whether the equity/rate relation alternates between distinct dependence regimes.
    series = correlation_data.set_index("date")["spx_ust_rolling_corr"]
    model = MarkovRegression(series, k_regimes=2, trend="c", switching_variance=True)
    result = model.fit(disp=False, em_iter=10, search_reps=20, search_iter=10)
    return result


def markov_summary_table(markov_result) -> pd.DataFrame:
    params = markov_result.params
    regime_prob = markov_result.smoothed_marginal_probabilities
    regime_means = {
        0: float(params.get("const[0]", np.nan)),
        1: float(params.get("const[1]", np.nan)),
    }
    lower_mean_state = min(regime_means, key=regime_means.get)
    higher_mean_state = max(regime_means, key=regime_means.get)

    return pd.DataFrame([
        {
            "regime": 0,
            "estimated_mean": regime_means[0],
            "estimated_variance": float(params.get("sigma2[0]", np.nan)),
            "avg_smoothed_probability": float(regime_prob[0].mean()),
            "label": "lower-correlation regime" if 0 == lower_mean_state else "higher-correlation regime",
        },
        {
            "regime": 1,
            "estimated_mean": regime_means[1],
            "estimated_variance": float(params.get("sigma2[1]", np.nan)),
            "avg_smoothed_probability": float(regime_prob[1].mean()),
            "label": "lower-correlation regime" if 1 == lower_mean_state else "higher-correlation regime",
        },
        {
            "regime": np.nan,
            "estimated_mean": np.nan,
            "estimated_variance": np.nan,
            "avg_smoothed_probability": float((regime_prob.idxmax(axis=1) == lower_mean_state).mean()),
            "label": "share of time in lower-correlation regime",
        },
    ])


def smoothed_probability_table(markov_result) -> pd.DataFrame:
    probs = markov_result.smoothed_marginal_probabilities.copy()
    probs = probs.rename(columns={0: "regime_0_probability", 1: "regime_1_probability"}).reset_index()
    probs = probs.rename(columns={probs.columns[0]: "date"})
    return probs


def plot_smoothed_probabilities(prob_table: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(prob_table["date"], prob_table["regime_0_probability"], color="navy", lw=1.2)
    axes[0].axvline(pd.Timestamp(COVID_BREAK), color="red", linestyle="--", lw=1.1)
    axes[0].set_title("Smoothed probability of regime 0")
    axes[0].grid(alpha=0.2)

    axes[1].plot(prob_table["date"], prob_table["regime_1_probability"], color="darkorange", lw=1.2)
    axes[1].axvline(pd.Timestamp(COVID_BREAK), color="red", linestyle="--", lw=1.1)
    axes[1].set_title("Smoothed probability of regime 1")
    axes[1].grid(alpha=0.2)

    figure.suptitle("Figure 1. Markov-switching smoothed probabilities for SPX/UST rolling correlation", fontsize=14)
    figure.tight_layout()
    return figure


def run_break_detection(series: pd.Series, n_breaks: int = MAX_BREAKS, min_size: int = ROLLING_BREAK_WINDOW) -> pd.DataFrame:
    # The break search is implemented with ruptures, which provides a practical Bai-Perron-style multiple-break procedure.
    clean_series = series.dropna().copy()
    signal = clean_series.to_numpy().reshape(-1, 1)
    algo = rpt.Binseg(model="l2", min_size=min_size).fit(signal)
    break_indices = algo.predict(n_bkps=n_breaks)

    rows = []
    for break_index in break_indices:
        if break_index < len(clean_series):
            rows.append({
                "break_index": int(break_index),
                "break_date": clean_series.index[break_index - 1],
                "series_mean_before_break": float(clean_series.iloc[:break_index].mean()),
            })
    return pd.DataFrame(rows)


def plot_breaks(series: pd.Series, break_table: pd.DataFrame, title: str, ylabel: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.plot(series.index, series.values, color="navy", lw=1.2)
    axis.axvline(pd.Timestamp(COVID_BREAK), color="red", linestyle="--", lw=1.1, label="11 March 2020")

    for _, row in break_table.iterrows():
        axis.axvline(pd.Timestamp(row["break_date"]), color="darkorange", linestyle=":", lw=1.1)

    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    return figure


def pre_post_break_count(break_table: pd.DataFrame) -> pd.DataFrame:
    sample_split = pd.Timestamp(POST_COVID_START)
    return pd.DataFrame([{
        "n_breaks_pre_covid": int((pd.to_datetime(break_table["break_date"]) < sample_split).sum()),
        "n_breaks_post_covid": int((pd.to_datetime(break_table["break_date"]) >= sample_split).sum()),
    }])


def save_regime_outputs(markov_summary: pd.DataFrame, smoothed_probs: pd.DataFrame, variance_breaks: pd.DataFrame, corr_breaks: pd.DataFrame) -> None:
    ensure_output_dirs()
    markov_summary.to_csv(TABLE_DIR / "05_markov_summary.csv", index=False)
    smoothed_probs.to_csv(TABLE_DIR / "05_markov_smoothed_probabilities.csv", index=False)
    variance_breaks.to_csv(TABLE_DIR / "05_sp500_variance_breaks.csv", index=False)
    corr_breaks.to_csv(TABLE_DIR / "05_spx_ust_corr_breaks.csv", index=False)
