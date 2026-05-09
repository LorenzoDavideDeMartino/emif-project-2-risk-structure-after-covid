from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from arch import arch_model

from project2_config import FIGURE_DIR, TABLE_DIR, COVID_BREAK, TRADING_DAYS
from project2_data_utils import (
    ensure_output_dirs,
    load_raw_data,
    build_aligned_returns,
    split_pre_post,
)

ROLLING_WINDOW = 252
KEY_PAIRS = [
    ("sp500", "ust10y_yield"),
    ("sp500", "us_hy_bonds"),
    ("sp500", "oil"),
    ("gold", "ust10y_yield"),
]
PAIR_LABELS = {
    ("sp500", "ust10y_yield"): "S&P 500 vs US 10Y yield change",
    ("sp500", "us_hy_bonds"): "S&P 500 vs US HY Bonds",
    ("sp500", "oil"): "S&P 500 vs Oil futures",
    ("gold", "ust10y_yield"): "Gold vs US 10Y yield change",
}
DISPLAY = {
    "sp500": "S&P 500",
    "ust10y_yield": "US 10Y yield change",
    "us_hy_bonds": "US HY Bonds",
    "oil": "Oil futures",
    "gold": "Gold",
}


def load_multivariate_samples() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_data = load_raw_data()
    _, aligned_returns = build_aligned_returns(raw_data)
    pre_covid, post_covid = split_pre_post(aligned_returns)
    return aligned_returns, pre_covid, post_covid


def rolling_correlation_table(aligned_returns: pd.DataFrame, pairs: list[tuple[str, str]], window: int = ROLLING_WINDOW) -> pd.DataFrame:
    rows = []
    for left, right in pairs:
        rolling_corr = aligned_returns[left].rolling(window).corr(aligned_returns[right])
        pair_frame = pd.DataFrame({
            "date": aligned_returns["date"],
            "left": left,
            "right": right,
            "pair_label": PAIR_LABELS[(left, right)],
            "rolling_corr": rolling_corr,
        })
        rows.append(pair_frame)
    return pd.concat(rows, axis=0, ignore_index=True).dropna().reset_index(drop=True)


def plot_rolling_correlations(rolling_table: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    for axis, pair in zip(axes, KEY_PAIRS):
        pair_data = rolling_table.loc[
            (rolling_table["left"] == pair[0]) & (rolling_table["right"] == pair[1])
        ]
        axis.plot(pair_data["date"], pair_data["rolling_corr"], lw=1.2, color="navy")
        axis.axvline(pd.Timestamp(COVID_BREAK), color="red", linestyle="--", lw=1.1)
        axis.axhline(0.0, color="black", linestyle=":", lw=0.9)
        axis.set_title(PAIR_LABELS[pair], fontsize=10)
        axis.grid(alpha=0.2)
    figure.suptitle("Figure 1. Rolling 52-week correlations", fontsize=14)
    figure.tight_layout()
    return figure


def summarize_rolling_correlations(rolling_table: pd.DataFrame) -> pd.DataFrame:
    covid_break = pd.Timestamp(COVID_BREAK)
    rows = []
    for pair in KEY_PAIRS:
        pair_data = rolling_table.loc[
            (rolling_table["left"] == pair[0]) & (rolling_table["right"] == pair[1])
        ]
        pre = pair_data.loc[pair_data["date"] < covid_break, "rolling_corr"]
        post = pair_data.loc[pair_data["date"] >= covid_break, "rolling_corr"]
        rows.append({
            "pair": PAIR_LABELS[pair],
            "pre_mean_rolling_corr": pre.mean(),
            "post_mean_rolling_corr": post.mean(),
            "change_post_minus_pre": post.mean() - pre.mean(),
            "pre_std_rolling_corr": pre.std(),
            "post_std_rolling_corr": post.std(),
            "pre_min_rolling_corr": pre.min(),
            "post_min_rolling_corr": post.min(),
            "pre_max_rolling_corr": pre.max(),
            "post_max_rolling_corr": post.max(),
        })
    return pd.DataFrame(rows)


def fit_univariate_garch(series: pd.Series) -> object:
    model = arch_model(series.dropna(), mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    return model.fit(disp="off", show_warning=False)


def standardized_residual_matrix(data: pd.DataFrame) -> pd.DataFrame:
    standardized = {}
    for column in data.columns:
        fitted = fit_univariate_garch(data[column])
        standardized[column] = pd.Series(fitted.std_resid, index=fitted.resid.index)
    return pd.DataFrame(standardized).dropna()


def dcc_negative_loglik(params: np.ndarray, z: np.ndarray, qbar: np.ndarray) -> float:
    alpha, beta = params
    if alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10

    q_t = qbar.copy()
    loss = 0.0
    eye = np.eye(qbar.shape[0])

    for t in range(1, z.shape[0]):
        q_t = (1.0 - alpha - beta) * qbar + alpha * np.outer(z[t - 1], z[t - 1]) + beta * q_t
        scale = np.sqrt(np.diag(q_t))
        r_t = q_t / np.outer(scale, scale)
        sign, logdet = np.linalg.slogdet(r_t)
        if sign <= 0:
            r_t = r_t + 1e-8 * eye
            sign, logdet = np.linalg.slogdet(r_t)
        inv_r_t = np.linalg.pinv(r_t)
        loss += logdet + z[t] @ inv_r_t @ z[t]

    return 0.5 * loss


def fit_bivariate_dcc(pair_data: pd.DataFrame) -> dict:
    if "date" in pair_data.columns:
        pair_data = pair_data.set_index("date")
    z = standardized_residual_matrix(pair_data)
    qbar = z.corr().to_numpy()
    optimum = optimize.minimize(
        dcc_negative_loglik,
        x0=np.array([0.03, 0.94]),
        args=(z.to_numpy(), qbar),
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6},
    )

    alpha, beta = optimum.x
    q_t = qbar.copy()
    dates = z.index
    correlations = []

    for t in range(1, z.shape[0]):
        q_t = (1.0 - alpha - beta) * qbar + alpha * np.outer(z.iloc[t - 1], z.iloc[t - 1]) + beta * q_t
        scale = np.sqrt(np.diag(q_t))
        r_t = q_t / np.outer(scale, scale)
        correlations.append({
            "date": dates[t],
            "dcc_corr": float(r_t[0, 1]),
        })

    dcc_path = pd.DataFrame(correlations)
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "persistence": float(alpha + beta),
        "success": bool(optimum.success),
        "dcc_path": dcc_path,
        "standardized_residuals": z,
    }


def plot_spx_ust_dcc(dcc_path: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.plot(dcc_path["date"], dcc_path["dcc_corr"], color="darkgreen", lw=1.3)
    axis.axvline(pd.Timestamp(COVID_BREAK), color="red", linestyle="--", lw=1.1)
    axis.axhline(0.0, color="black", linestyle=":", lw=0.9)
    axis.set_title("Figure 2. DCC-GARCH correlation: S&P 500 vs US 10Y yield change")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def summarize_dcc_path(dcc_path: pd.DataFrame) -> pd.DataFrame:
    covid_break = pd.Timestamp(COVID_BREAK)
    pre = dcc_path.loc[dcc_path["date"] < covid_break, "dcc_corr"]
    post = dcc_path.loc[dcc_path["date"] >= covid_break, "dcc_corr"]
    return pd.DataFrame([{
        "pre_mean_dcc_corr": pre.mean(),
        "post_mean_dcc_corr": post.mean(),
        "change_post_minus_pre": post.mean() - pre.mean(),
        "pre_std_dcc_corr": pre.std(),
        "post_std_dcc_corr": post.std(),
        "pre_min_dcc_corr": pre.min(),
        "post_min_dcc_corr": post.min(),
        "pre_max_dcc_corr": pre.max(),
        "post_max_dcc_corr": post.max(),
    }])


def forbes_rigobon_adjustment(correlation: float, variance_ratio: float) -> float:
    # Forbes-Rigobon corrects raw correlation when one market becomes much more volatile across samples.
    delta = variance_ratio - 1.0
    denominator = np.sqrt(1.0 + delta * (1.0 - correlation ** 2))
    return float(correlation / denominator)


def build_forbes_rigobon_table(pre_covid: pd.DataFrame, post_covid: pd.DataFrame) -> pd.DataFrame:
    raw_pre_corr = pre_covid[["sp500", "ust10y_yield"]].corr().iloc[0, 1]
    raw_post_corr = post_covid[["sp500", "ust10y_yield"]].corr().iloc[0, 1]
    variance_ratio = post_covid["sp500"].var() / pre_covid["sp500"].var()
    adjusted_post_corr = forbes_rigobon_adjustment(raw_post_corr, variance_ratio)

    return pd.DataFrame([{
        "raw_pre_corr": raw_pre_corr,
        "raw_post_corr": raw_post_corr,
        "sp500_variance_ratio_post_pre": variance_ratio,
        "forbes_rigobon_adjusted_post_corr": adjusted_post_corr,
        "difference_adjusted_post_minus_pre": adjusted_post_corr - raw_pre_corr,
    }])


def save_multivariate_outputs(rolling_summary: pd.DataFrame, dcc_summary: pd.DataFrame, fr_table: pd.DataFrame, dcc_path: pd.DataFrame) -> None:
    ensure_output_dirs()
    rolling_summary.to_csv(TABLE_DIR / "03_rolling_correlation_summary.csv", index=False)
    dcc_summary.to_csv(TABLE_DIR / "03_dcc_summary.csv", index=False)
    fr_table.to_csv(TABLE_DIR / "03_forbes_rigobon_summary.csv", index=False)
    dcc_path.to_csv(TABLE_DIR / "03_spx_ust_dcc_path.csv", index=False)
