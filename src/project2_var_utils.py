from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

from project2_config import TABLE_DIR
from project2_data_utils import ensure_output_dirs, load_raw_data, build_aligned_returns, split_pre_post

VAR_COLUMNS = ["sp500", "ust10y_yield", "oil", "eurusd", "us_hy_bonds"]
VAR_LABELS = {
    "sp500": "S&P 500",
    "ust10y_yield": "US 10Y yield change",
    "oil": "Oil futures",
    "eurusd": "EURUSD",
    "us_hy_bonds": "US HY Bonds",
}
VAR_LAG = 2
IRF_HORIZON = 10


def load_var_samples() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_data = load_raw_data()
    _, aligned_returns = build_aligned_returns(raw_data)
    aligned_var_data = aligned_returns[["date"] + VAR_COLUMNS].dropna().copy()
    pre_covid, post_covid = split_pre_post(aligned_var_data)
    return aligned_var_data, pre_covid, post_covid


def fit_var(sample_data: pd.DataFrame, lag_order: int = VAR_LAG) -> VARResultsWrapper:
    var_input = sample_data.set_index("date")
    fitted_var = VAR(var_input).fit(lag_order)
    return fitted_var


def var_order_selection(sample_data: pd.DataFrame, max_lags: int = 10) -> pd.DataFrame:
    var_input = sample_data.set_index("date")
    selection = VAR(var_input).select_order(maxlags=max_lags)
    return pd.DataFrame([selection.selected_orders]).rename(columns=str.upper)


def var_summary_row(var_result, sample_name: str) -> dict:
    roots = np.abs(var_result.roots)
    return {
        "sample": sample_name,
        "lag_order": int(var_result.k_ar),
        "n_obs": int(var_result.nobs),
        "aic": float(var_result.aic),
        "bic": float(var_result.bic),
        "hqic": float(var_result.hqic),
        "stable": bool(var_result.is_stable()),
        "smallest_root_modulus": float(np.min(roots)),
    }


def granger_table(var_result, sample_name: str) -> pd.DataFrame:
    rows = []
    for caused in VAR_COLUMNS:
        for causing in VAR_COLUMNS:
            if caused == causing:
                continue
            test = var_result.test_causality(caused=caused, causing=[causing], kind="f")
            rows.append({
                "sample": sample_name,
                "caused": caused,
                "causing": causing,
                "granger_f_stat": float(test.test_statistic),
                "granger_pvalue": float(test.pvalue),
            })
    return pd.DataFrame(rows)


def geweke_directional_measure(data: pd.DataFrame, causing: str, caused: str, lag_order: int = VAR_LAG) -> dict:
    lagged_blocks = [
        data[VAR_COLUMNS].shift(lag).add_suffix(f"_l{lag}")
        for lag in range(1, lag_order + 1)
    ]
    regression_frame = pd.concat([data[[caused]], *lagged_blocks], axis=1).dropna()
    dependent = regression_frame[caused]
    regressors = regression_frame.drop(columns=[caused])
    drop_columns = [f"{causing}_l{lag}" for lag in range(1, lag_order + 1)]

    full_fit = OLS(dependent, add_constant(regressors)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    restricted_fit = OLS(
        dependent,
        add_constant(regressors.drop(columns=drop_columns)),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    n_obs = len(dependent)
    full_variance = float(full_fit.ssr / n_obs)
    restricted_variance = float(restricted_fit.ssr / n_obs)
    measure = float(np.log(restricted_variance / full_variance))
    statistic = n_obs * measure
    pvalue = 1.0 - stats.chi2.cdf(statistic, df=lag_order)
    return {
        "geweke_value": measure,
        "test_stat": statistic,
        "pvalue": pvalue,
    }


def geweke_causality_table(var_result, sample_data: pd.DataFrame, sample_name: str, lag_order: int = VAR_LAG) -> pd.DataFrame:
    # Directional Geweke compares a full equation with a restricted one where the lags of one causing variable are removed.
    var_input = sample_data.set_index("date")
    sigma_u = var_result.sigma_u
    n_obs = int(var_result.nobs)
    rows = []

    for causing, caused in itertools.permutations(VAR_COLUMNS, 2):
        directional_result = geweke_directional_measure(sample_data, causing, caused, lag_order=lag_order)
        rows.append({
            "sample": sample_name,
            "measure": "directional",
            "causing": causing,
            "caused": caused,
            **directional_result,
        })

    for left, right in itertools.combinations(VAR_COLUMNS, 2):
        sub_cov = sigma_u.loc[[left, right], [left, right]].to_numpy()
        determinant = max(float(np.linalg.det(sub_cov)), 1e-12)
        instantaneous_measure = float(np.log((sub_cov[0, 0] * sub_cov[1, 1]) / determinant))
        instantaneous_stat = n_obs * instantaneous_measure
        instantaneous_pvalue = 1.0 - stats.chi2.cdf(instantaneous_stat, df=1)
        rows.append({
            "sample": sample_name,
            "measure": "instantaneous",
            "causing": left,
            "caused": right,
            "geweke_value": instantaneous_measure,
            "test_stat": instantaneous_stat,
            "pvalue": instantaneous_pvalue,
        })

    return pd.DataFrame(rows)


def plot_irf_panel(var_result, sample_name: str) -> plt.Figure:
    # We focus on a small set of economically interpretable IRFs rather than plotting the full 5x5 system.
    irf = var_result.irf(IRF_HORIZON)
    orth_irfs = irf.orth_irfs
    horizons = np.arange(orth_irfs.shape[0])
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    selections = [
        ("sp500", "ust10y_yield", "S&P 500 response to a US 10Y yield shock"),
        ("sp500", "oil", "S&P 500 response to an oil shock"),
        ("us_hy_bonds", "sp500", "US HY Bonds response to an equity shock"),
        ("us_hy_bonds", "ust10y_yield", "US HY Bonds response to a US 10Y yield shock"),
    ]

    for axis, (response, impulse, title) in zip(axes.flatten(), selections):
        response_idx = VAR_COLUMNS.index(response)
        impulse_idx = VAR_COLUMNS.index(impulse)
        axis.plot(horizons, orth_irfs[:, response_idx, impulse_idx], lw=1.4, color="navy")
        axis.axhline(0.0, color="black", linestyle=":", lw=0.9)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Days")
        axis.set_ylabel("Response")
        axis.grid(alpha=0.2)

    figure.suptitle(f"Cholesky impulse responses - {sample_name}", fontsize=14)
    figure.tight_layout()
    return figure


def save_var_outputs(var_summary: pd.DataFrame, granger_results: pd.DataFrame, geweke_results: pd.DataFrame) -> None:
    ensure_output_dirs()
    var_summary.to_csv(TABLE_DIR / "04_var_summary.csv", index=False)
    granger_results.to_csv(TABLE_DIR / "04_granger_results.csv", index=False)
    geweke_results.to_csv(TABLE_DIR / "04_geweke_results.csv", index=False)
