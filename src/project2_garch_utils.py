from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model

from project2_config import TABLE_DIR
from project2_data_utils import (
    ensure_output_dirs,
    load_raw_data,
    build_aligned_returns,
    split_pre_post,
)

GARCH_ASSETS = ["sp500", "us_hy_bonds", "oil", "gold"]
GARCH_LABELS = {
    "sp500": "S&P 500",
    "us_hy_bonds": "US HY Bonds",
    "oil": "Oil futures",
    "gold": "Gold",
}
MODEL_SPECS = {
    "GARCH(1,1)": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "normal"},
    "GJR-GARCH": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "normal"},
    "GARCH-t": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "t"},
}
LJUNG_BOX_LAG = 10


def load_garch_samples() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_data = load_raw_data()
    _, aligned_returns = build_aligned_returns(raw_data)
    pre_covid, post_covid = split_pre_post(aligned_returns)
    return aligned_returns, pre_covid, post_covid


def fit_one_garch(series: pd.Series, model_name: str) -> object:
    spec = MODEL_SPECS[model_name]
    model = arch_model(
        series.dropna(),
        mean="Constant",
        vol=spec["vol"],
        p=spec["p"],
        o=spec["o"],
        q=spec["q"],
        dist=spec["dist"],
        rescale=False,
    )
    result = model.fit(disp="off", show_warning=False)
    return result


def compute_persistence(result: object, model_name: str) -> float:
    params = result.params
    alpha = float(params.get("alpha[1]", 0.0))
    beta = float(params.get("beta[1]", 0.0))
    gamma = float(params.get("gamma[1]", 0.0))
    if model_name == "GJR-GARCH":
        return alpha + beta + 0.5 * gamma
    return alpha + beta


def engle_ng_test(result: object) -> dict:
    # Engle-Ng checks whether negative shocks create asymmetric variance effects left unexplained by the model.
    standardized_residuals = pd.Series(result.std_resid).dropna()
    lagged_residuals = standardized_residuals.shift(1)
    negative_dummy = (lagged_residuals < 0).astype(float)

    test_frame = pd.DataFrame({
        "squared_resid": standardized_residuals ** 2,
        "negative_dummy": negative_dummy,
        "negative_size": negative_dummy * lagged_residuals,
        "positive_size": (1.0 - negative_dummy) * lagged_residuals,
    }).dropna()

    regression = OLS(
        test_frame["squared_resid"],
        add_constant(test_frame[["negative_dummy", "negative_size", "positive_size"]]),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    joint_test = regression.f_test("negative_dummy = negative_size = positive_size = 0")

    return {
        "engle_ng_joint_f": float(joint_test.fvalue),
        "engle_ng_joint_pvalue": float(joint_test.pvalue),
        "sign_bias_pvalue": float(regression.pvalues.get("negative_dummy", np.nan)),
        "negative_size_bias_pvalue": float(regression.pvalues.get("negative_size", np.nan)),
        "positive_size_bias_pvalue": float(regression.pvalues.get("positive_size", np.nan)),
    }


def diagnostics_table(result: object) -> dict:
    # We test residual autocorrelation and residual ARCH left in the standardized residuals.
    standardized_residuals = pd.Series(result.std_resid).dropna()
    ljung_resid = acorr_ljungbox(standardized_residuals, lags=[LJUNG_BOX_LAG], return_df=True)
    ljung_sq = acorr_ljungbox(standardized_residuals ** 2, lags=[LJUNG_BOX_LAG], return_df=True)
    engle_ng = engle_ng_test(result)

    diagnostics = {
        "lb_resid_stat_lag10": float(ljung_resid.loc[LJUNG_BOX_LAG, "lb_stat"]),
        "lb_resid_pvalue_lag10": float(ljung_resid.loc[LJUNG_BOX_LAG, "lb_pvalue"]),
        "lb_sq_resid_stat_lag10": float(ljung_sq.loc[LJUNG_BOX_LAG, "lb_stat"]),
        "lb_sq_resid_pvalue_lag10": float(ljung_sq.loc[LJUNG_BOX_LAG, "lb_pvalue"]),
    }
    diagnostics.update(engle_ng)
    return diagnostics


def summarize_garch_result(result: object, asset_name: str, sample_name: str, model_name: str) -> dict:
    params = result.params
    summary = {
        "asset": asset_name,
        "sample": sample_name,
        "model": model_name,
        "n_obs": int(result.nobs),
        "mu": float(params.get("mu", np.nan)),
        "omega": float(params.get("omega", np.nan)),
        "alpha_1": float(params.get("alpha[1]", np.nan)),
        "beta_1": float(params.get("beta[1]", np.nan)),
        "gamma_1": float(params.get("gamma[1]", np.nan)),
        "nu": float(params.get("nu", np.nan)),
        "persistence": compute_persistence(result, model_name),
        "loglik": float(result.loglikelihood),
        "aic": float(result.aic),
        "bic": float(result.bic),
    }
    summary.update(diagnostics_table(result))
    return summary


def estimate_all_garch_models(pre_covid: pd.DataFrame, post_covid: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for sample_name, sample_data in [("Pre-COVID", pre_covid), ("Post-COVID", post_covid)]:
        for asset_name in GARCH_ASSETS:
            for model_name in MODEL_SPECS:
                result = fit_one_garch(sample_data[asset_name], model_name)
                summary_rows.append(summarize_garch_result(result, asset_name, sample_name, model_name))

    return pd.DataFrame(summary_rows)


def plot_conditional_volatility(pre_series: pd.Series, post_series: pd.Series, asset_name: str) -> plt.Figure:
    # A plain GARCH(1,1) volatility plot is enough to compare volatility persistence across the two regimes.
    pre_result = fit_one_garch(pre_series, "GARCH(1,1)")
    post_result = fit_one_garch(post_series, "GARCH(1,1)")

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    axes[0].plot(pre_result.conditional_volatility, color="navy", lw=1.1)
    axes[0].set_title(f"{GARCH_LABELS[asset_name]} conditional volatility - Pre-COVID")
    axes[0].grid(alpha=0.2)

    axes[1].plot(post_result.conditional_volatility, color="firebrick", lw=1.1)
    axes[1].set_title(f"{GARCH_LABELS[asset_name]} conditional volatility - Post-COVID")
    axes[1].grid(alpha=0.2)

    figure.tight_layout()
    return figure


def build_comparison_table(summary_table: pd.DataFrame) -> pd.DataFrame:
    # A compact comparison table makes the pre/post interpretation easier than reading 24 rows directly.
    focus_columns = [
        "asset",
        "sample",
        "model",
        "alpha_1",
        "beta_1",
        "gamma_1",
        "nu",
        "persistence",
        "lb_resid_pvalue_lag10",
        "lb_sq_resid_pvalue_lag10",
        "engle_ng_joint_pvalue",
    ]
    comparison_table = summary_table[focus_columns].copy()
    comparison_table["asset_label"] = comparison_table["asset"].map(GARCH_LABELS)
    return comparison_table


def save_garch_outputs(summary_table: pd.DataFrame) -> None:
    ensure_output_dirs()
    summary_table.to_csv(TABLE_DIR / "02_garch_summary.csv", index=False)
