# EMiF Project 2 - Has the structure of risk in financial markets changed since COVID-19?

This repository contains the full empirical pipeline for Project 2 of **Empirical Methods in Finance (EMiF)**. The analysis uses **only the provided Excel workbook** and stays within the scope of the course: stylized facts, GARCH, DCC, VAR, Geweke causality, Markov switching and structural break analysis.

## Research question

**Has the structure of risk in financial markets changed since COVID-19?**

The project studies this question by comparing the behavior of returns, volatility, dependence, transmission and regimes across two calendar sub-samples:

- **Pre-COVID:** 2000-2019
- **Post-COVID:** 2020 onward

The event date of **11 March 2020** is then treated separately as the key economic breakpoint to test and visualize.

## Data

The project is designed to run with a single input file:

- `data/raw/Data.xlsx`

No external data source is used anywhere in the pipeline.

## Repository structure

```text
.
|-- data/
|   |-- raw/
|   |   `-- Data.xlsx
|   `-- processed/
|-- notebooks/
|   |-- 01_data_prep_and_stylized_facts.ipynb
|   |-- 02_univariate_garch.ipynb
|   |-- 03_multivariate_dcc_erc.ipynb
|   |-- 04_var_geweke_transmission.ipynb
|   |-- 05_regimes_and_breaks.ipynb
|   |-- _notebook_setup.py
|   `-- EMiF_Project_Final.ipynb
|-- outputs/
|   `-- project2/
|-- src/
|   |-- project2_config.py
|   |-- project2_data_utils.py
|   |-- project2_garch_utils.py
|   |-- project2_multivariate_utils.py
|   |-- project2_var_utils.py
|   `-- project2_regime_utils.py
|-- requirements.txt
`-- README.md
```

## Environment setup

```bash
pip install -r requirements.txt
```

## Recommended execution order

Run the notebooks in the following order:

1. `01_data_prep_and_stylized_facts.ipynb`
2. `02_univariate_garch.ipynb`
3. `03_multivariate_dcc_erc.ipynb`
4. `04_var_geweke_transmission.ipynb`
5. `05_regimes_and_breaks.ipynb`
6. `EMiF_Project_Final.ipynb`

If you want a single submission-ready notebook, run `EMiF_Project_Final.ipynb`: it reproduces the full pipeline directly from `Data.xlsx`.

## What each notebook does

### 01_data_prep_and_stylized_facts.ipynb
- Loads the 14 market series from the Excel file
- Builds log returns for price series and first differences in basis points for yields
- Defines the pre-COVID (2000-2019) and post-COVID (2020 onward) samples
- Produces descriptive moments, downside risk measures, correlation heatmaps and autocorrelation plots
- Justifies the COVID break date with variance break tests on the S&P 500

### 02_univariate_garch.ipynb
- Estimates GARCH(1,1), GJR-GARCH and GARCH-t models
- Compares persistence and asymmetry before and after COVID
- Runs standard residual diagnostics

### 03_multivariate_dcc_erc.ipynb
- Studies time-varying dependence through rolling correlations and DCC-GARCH
- Focuses especially on the SPX / UST10Y relationship
- Applies the Forbes-Rigobon adjustment

### 04_var_geweke_transmission.ipynb
- Estimates the five-variable VAR requested in the project brief
- Computes Geweke causality measures and Cholesky IRFs
- Studies whether transmission channels changed after COVID

### 05_regimes_and_breaks.ipynb
- Estimates a two-state Markov switching model on the rolling SPX / UST10Y correlation
- Detects multiple structural breaks with `ruptures`
- Serves as the core notebook for the final research message

### EMiF_Project_Final.ipynb
- Reproduces the full empirical pipeline in one self-contained notebook
- Can be used as the submission-ready master notebook because it reruns the analysis directly from `Data.xlsx`

## Outputs

The notebooks save tables and figures under `outputs/project2/` so that results can be reused in the report without recomputing everything manually.

## Reproducibility notes

- The code is written to run end-to-end with the Excel workbook only.
- No machine learning, copulas, intraday data or BEKK models are used.
- Yield series are never log-transformed; they are converted into daily changes in basis points.
- Oil is treated with simple percentage returns because the April 2020 negative WTI settlement makes a standard log return undefined.
- The sample split is calendar-based (`2000-2019` vs `2020+`), while `11 March 2020` is kept as the tested and plotted COVID event date.
- The analysis is intended to run in less than 15 minutes on a standard laptop.
