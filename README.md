# DSA5205 Project 2 — Fundamental Signals and Portfolio Construction in US Equities

Team 34: Wan Gehong(A0356324E), Wang Tianyi(A0357264X), Wu Jiarui(A0262471J), Yan Tianao(A0354280H).

This project uses US equity datasets together with fundamental and price-volume factors to develop and evaluate a quantitative investment strategy. It compares a six-factor direct score with Ridge, Lasso and XGBoost under common portfolio rules. Both notebooks contain the saved outputs of a complete sequential execution on 11 November 2026.

## Notebooks

Run these notebooks in order, with a fresh kernel for each:

1. [Equity Data Preparation](notebooks/Equity_Data_Preparation.ipynb): source inventory, market-quality checks, dated security mapping, financial and industry events, monthly universe construction, total-return prices, next-close labels and source-coverage audits.
2. [Equity Strategy Research](notebooks/Equity_Strategy_Research.ipynb): input verification, factor screening, feature engineering, chronological CV, validation selection, evaluation, robustness, return attribution and accounting checks.

Data preparation constructs the research dataset directly from raw inputs. It publishes `data/processed/latest_snapshot.json` only after every stage succeeds. The strategy notebook verifies the completed dataset, its checksums, canonical price chain and labels. Each notebook creates a new timestamped run; the strategy notebook trains models afresh. Full execution is not needed to read saved results.

## Local setup

Raw Data Link: https://drive.google.com/file/d/1Dn6wCS3lrVT1TrblWFauMjQE_IWTCkjk/view?usp=share_link

GitHub Link: https://github.com/UtAstN/dsa5205-project2-submission

Clone the repository, then use Python 3.13 from the repository root. The following shell commands use macOS/Linux paths:

```bash
git clone https://github.com/UtAstN/dsa5205-project2-submission.git
cd dsa5205-project2-submission
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m ipykernel install --user --name team34-project2 --display-name "Python (Team 34 Project 2)"
```

Open the notebooks in VS Code with Jupyter support, or an existing Jupyter installation. Select the kernel that uses this repository's `.venv` interpreter. A newly installed kernel appears as **Python (Team 34 Project 2)**; depending on the editor and its saved metadata, the notebooks may initially display **.venv (3.13.0)** or **Python 3** instead. Verify that the selected interpreter is `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows) before running. Use **Restart Kernel → Run All** in each notebook only when repeating the research. On macOS, XGBoost may require `brew install libomp`. On Windows PowerShell, create the environment with `py -3.13 -m venv .venv` and replace `.venv/bin/python` in Python commands with `.venv\Scripts\python.exe`.

Raw data are not included in this repository. To reproduce the experiments, place the required source files in the layout below. Keep the two named Compustat ZIP archives intact because the preparation code reads them directly:

```text
data/
  raw/
    wrds/14_1_crsp_daily/        CRSP daily prices, names and delistings
    wrds/14_2_compustat/         downloaded Compustat extracts
    wrds/14_3_ccm_link/          dated CCM security links
    compustat/quarterly/Comp_Quarterly6126.csv.zip
    compustat/annual/20250102updated north fundamental_dta.zip
    benchmark/SPY.gz            archived raw Yahoo chart response
    benchmark/SPY.json          source provenance
    sina/<capture>/            supplementary raw vendor responses
  processed/
    <data-run>/research_snapshot/
    latest_snapshot.json
```

The quarterly archive supplies the longer financial history and release proxies; the North America annual archive supplies historical SIC. The preparation notebook attempts fresh Sina requests and records failures, so this coverage-audit step can require internet access. Sina turnover/VWAP is excluded from strategy inputs because coverage is incomplete. Liquidity screening uses CRSP unadjusted close × share volume. The archived `SPY.gz` supplies the benchmark; it is not downloaded by the strategy notebook.

Reserve space for raw archives, published panels, and each new run's training matrices and model outputs. `runs/` contains local execution artifacts; `results/` contains compact, shareable summaries. Data, backups, model files and per-security forecasts/holdings are excluded from Git. Saved notebook outputs are retained.

### Use a prepared dataset

A complete dataset produced by the preparation notebook can be used directly with the strategy notebook. Set `DATA_DIRECTORY` in the strategy notebook's first code cell to the folder that directly contains `panels/`, `metadata/` and `checksums.sha256`:

```python
DATA_DIRECTORY = "/your/local/path/research_snapshot"
```

Alternatively set `TEAM34_DATA_DIR` before launching Jupyter. Resolution priority is `DATA_DIRECTORY`, then `TEAM34_DATA_DIR`, then the local published pointer. Leave both explicit settings unset to use the default dataset published by data preparation.

The dataset must include all panels, index files and metadata, including `snapshot.json`, `preprocessing_complete.json` and `activation.json` under `metadata/`. Completion flags, precision policy and checksums are verified before use. Load pickle files only from trusted sources. The default-pointer route also requires the producer run's `complete.json`; an explicit path can be used for a complete dataset stored elsewhere.

## Research design

| Stage | Protocol |
|---|---|
| Data | CRSP 2014–2025, with earlier Compustat history for accounting features. Dated CCM links; explicit quote, missing-data and delisting treatment. |
| Universe | Monthly top 80% by 60-session mean dollar-volume proxy; 252 observed closes; raw close ≥ $5; daily security/price gates. The completed dataset contains 4,366 ever-selected securities. |
| Prices and targets | Causal float64 total-return chains from source returns. Target: T+1 close to T+11 close. Unmatured tail labels remain missing. |
| WQ screening | Directions from 2015–2017; 36 candidates screened in 2018–2019 using RankIC, coverage, annual stability and correlation. Six factors pass. |
| Model tuning | Three feature sets: fundamentals; fundamentals plus WQ; all engineered features. Expanding annual CV folds in 2020–2022 with purged labels and fresh semiannual fits: 63 configurations, 189 scored folds. |
| Portfolio selection | In 2023, compare four scores × three weightings: equal, square-root cap and market cap. Select maximum net Sharpe among finite-Sharpe candidates with mean cash ≤ 5%. |
| Evaluation | Freeze specifications before 2024–2025 evaluation. ML forecasts refit semiannually on the preceding five calendar years using mature labels only. |

Trading uses next-close execution, Top100 selection, a rank-150 retention buffer, ten-session rebalancing, a 2% formation-weight cap and 10 bps per traded dollar. SPY and matched full-pool portfolios are controls. Costs, leave-one-factor-out checks and benchmark-relative return attribution use the six-factor market-cap portfolio; holdings-based risk attribution uses the validation-selected portfolio.

The 4,366 identifiers are the union across the full history; daily eligible membership is about two thousand, not four thousand simultaneously.

WQ price and screening calculations retain float64 precision before final model-feature storage. This avoids changes caused by differently normalized, prematurely rounded adjustment factors.

## Latest results and verification

The 2023 rule selects **six-factor direct score with market-cap weights** (`fundamental_cap`), with validation net Sharpe **1.3325**.

| 2024–2025 | Net CAGR | Annual volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Selected strategy | 24.92% | 16.81% | 1.4074 | −20.76% |
| Ridge / square-root cap | 12.31% | 20.80% | 0.6617 | −26.76% |
| XGBoost / equal weight | 15.72% | 26.82% | 0.6783 | −29.43% |
| SPY | 22.19% | 16.35% | 1.3071 | −18.76% |

Representative weightings are frozen using 2023 validation, not chosen from the evaluation table. Lasso is ineligible in validation because it holds cash; its prediction diagnostics remain available. The selected strategy's CAGR gap to SPY is +2.73 percentage points, with the saved 20-session paired-bootstrap 95% interval [−8.79, +16.40] points.

The notebooks include OOS prediction R², period-level return/risk plots, trading-cost and factor-ablation checks, linked return attribution, and a US adaptation of the repository's Barra-style market/industry/style model. Estimated residual alpha is conditional on that model and is not evidence of a guaranteed tradable alpha.

The reproducibility audit records **74 passing tests, 396 fresh model fits, 38 portfolio ledgers**, source-hash checks, universe truncation checks and an independent source-return check of 8,945,343 labels. A repeated XGBoost fit produced exactly identical predictions. See the [reproduction audit](results/reproduction_audit.json) for the recorded checks.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests -q
```

## Expected outputs

Running the notebooks produces both displayed Notebook outputs and files in the project directory. Each clean execution uses new timestamped run directories, while compact summary files are copied to `results/`.

### Notebook outputs

The data-preparation notebook displays:

- source inventories, row counts, file sizes and SHA-256 hashes;
- market-quality checks, quarantined OHLC anomalies and security-link diagnostics;
- fundamental and industry event summaries;
- universe membership, truncation checks and yearly coverage statistics;
- price, delisting, target-label and supplementary-vendor coverage audits; and
- the paths of the completed research snapshot and activated strategy input.

The strategy-research notebook displays:

- input-integrity, price-precision, corporate-action and target-coverage checks;
- WQ candidate screening and fundamental-factor diagnostics;
- chronological cross-validation scores, selected hyperparameters and training windows;
- validation and evaluation tables for the direct score, Ridge, Lasso and XGBoost portfolios;
- OOS prediction R², wealth, drawdown, return, risk and turnover results;
- trading-cost, factor-ablation and benchmark-relative robustness analyses;
- return attribution and holdings-based market, industry and style-factor attribution; and
- portfolio-accounting checks and the final selected-strategy summary.

### Project file outputs

- `runs/wrds_<timestamp>/`: detailed data-preparation audits, source manifests, intermediate event files, coverage tables and a completion marker.
- `data/processed/<data-run>/research_snapshot/`: the published research dataset, including aligned panels, metadata, label information and `checksums.sha256`.
- `data/processed/latest_snapshot.json`: the pointer used by the strategy notebook to locate the latest successfully completed dataset.
- `results/raw_data_preparation/`: compact, shareable data-quality, source, universe, financial, industry and target-coverage summaries.
- `runs/experiment_<timestamp>/`: the complete strategy run, including factor screens, model-selection records, forecasts, portfolio ledgers, robustness tests, attribution tables and audit files.
- `results/`: compact CSV and JSON summaries of factor screening, cross-validation, frozen model choices, training history, portfolio performance, OOS R², risk attribution and reproducibility checks.

