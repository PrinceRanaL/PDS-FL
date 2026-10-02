# PDS-FL: A Probabilistic Data Structure based Secure Federated Learning Framework for Smart Cities

Reference implementation accompanying the manuscript *"PDS-FL: A
Probabilistic Data Structure based Secure Federated Learning Framework
for Smart Cities"* (Prince Rana, Amritpal Singh). Released under the
MIT license as part of the manuscript's Code Availability declaration.

All results reported in the manuscript are produced by the scripts in this
repository. Because LightGBM's tree ensemble has no fixed-length parameter
vector, FedProx/FedAvg are implemented in their functional
(federated gradient-boosting) form; see the docstring of
`src/phase2/federated.py`.

## Repository structure

```
PDS-FL/
├── src/
│   ├── data/          schema.py (column detection), preprocessing.py (imputation, features, temporal split)
│   ├── phase1/        train_baselines.py (LR, SVR, RF, XGBoost, LightGBM)
│   ├── phase2/        aqf.py (Adaptive Quotient Filter), sembf.py (Semaphore Bloom Filter),
│   │                  federated.py (FedProx / FedAvg federated boosting)
│   └── utils/         metrics.py
├── scripts/           one script per experiment in the paper (see Usage)
├── tests/             test_smoke.py (synthetic data, no dataset needed)
├── data/              place the Kaggle CSV here (not redistributed)
├── results/           JSON/CSV/PNG outputs of the runs reported in the paper
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/[author-github-org]/PDS-FL.git
cd PDS-FL
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

This work uses the publicly available Kaggle dataset **"Air Quality
Dataset: Indian Cities (2022-2025)"**:
<https://www.kaggle.com/datasets/bhautikvekariya21/air-quality-dataset-indian-cities-2022-2025>

Download it yourself and place the CSV at `data/aqi_india.csv` (the
raw CSV is not redistributed in this repository). The pipeline
auto-detects column names via `src/data/schema.py`; if your copy of
the dataset uses different column headers, add the actual names to
the `ALIASES` dict in that file.

## Usage

### 1. Verify the pipeline (no dataset needed)

```bash
pytest tests/test_smoke.py -v
```

### 2. Phase 1 -- centralised model benchmarking

```bash
python scripts/run_phase1.py --csv data/aqi_india.csv --output_dir results/phase1
```
Trains Linear Regression, SVR, Random Forest, XGBoost, and LightGBM,
reports test R2/RMSE/MAE and the train-test bias-variance gap for
each, and saves the best model (by test R2) plus the fitted scaler.

### 3. Phase 2 -- FedProx vs. FedAvg convergence comparison

```bash
python scripts/run_phase2_convergence.py --csv data/aqi_india.csv \
    --output_dir results/phase2 --rounds 100 --mu 1.0
```
Simulates one federated client per city (29 clients), gated by AQF
authentication and SemBF round-scoped deduplication, and runs both
FedAvg (`mu=0`) and FedProx (`mu>0`) for the requested number of
rounds. Saves the round-wise metrics CSV and a convergence plot.

Both the raw per-round metrics and the best-checkpoint (early-stopped)
metrics are saved.

### 4. AQF false-positive-rate measurement

```bash
python scripts/measure_aqf_fpr.py --n_legit 29 --n_attacks 1000000
```
Registers the legitimate client population, then simulates the given
number of unauthorised-client authentication attempts, reporting the
empirical false-positive rate with a 95% Wilson-score confidence
interval alongside AQF's theoretical FPR bound.

### 5. Additional analyses

```bash
# AQF/SemBF scalability, 29 to 10,000 synthetic clients
python scripts/measure_scalability.py --sizes 29 100 500 1000 5000 10000

# Measured bytes/round and estimated transmission time vs. raw-data upload
python scripts/measure_communication_cost.py --csv data/aqi_india.csv --rounds 38

# Multi-run robustness + paired t-test / Wilcoxon / Friedman across models
python scripts/run_multiseed_robustness.py --csv data/aqi_india.csv --n_runs 8 --subsample_size 35000

# Gaussian-mechanism DP ablation (linear-model variant), privacy-utility sweep
python scripts/run_phase2_dp.py --csv data/aqi_india.csv --rounds 60 \
    --epsilons 0.01 0.02 0.05 0.1 0.5 1.0 5.0 --clip 2.0 --delta 1e-5
```

```bash
# SemBF duplicate-detection stress test (100 rounds, ~50% of clients
# also submit 1-3 extra duplicates per round)
python scripts/measure_sembf_dedup.py --rounds 100 --dup_frac 0.5

# Adaptive AQF: false-positive bound and measured rate as enrolment grows
python scripts/measure_aqf_adaptive.py --sizes 29 64 128 256 512 1024 2048 4096

# Per-city test RMSE/R2 of the federated model, and per-city linear
# coefficients used to quantify non-IID heterogeneity
python scripts/per_city_eval.py --csv data/aqi_india.csv --rounds 38 --mu 1.0

# Regenerate the Phase-1 and Phase-2 figures from the saved JSON/CSV results
python scripts/make_phase1_figures.py
python scripts/make_phase2_figures.py
```

Notes: the multi-run study fits each model on a random subsample of the
training set for tractability; the DP ablation uses a linear model (not
LightGBM) because a tree ensemble has no fixed-length parameter vector to
clip and noise; the FedProx-vs-FedAvg comparison was run once per rule.
The result files in `results/` are the outputs of the runs reported in the
manuscript. One multi-run job was interrupted after 7 of 8 runs, so the
saved statistics use those 7 (see the `note` field in the JSON).

Every number stated in the manuscript (Abstract, Tables, Figures) was
regenerated from these scripts against the released dataset; see
`results/*/*.json` for the raw values.

## Citation

If you use this code, please cite the manuscript:

```bibtex
@article{rana_pdsfl,
  title   = {PDS-FL: A Probabilistic Data Structure based Secure Federated Learning Framework for Smart Cities},
  author  = {Rana, Prince and Singh, Amritpal},
  journal = {Journal of Cloud Computing},
}
```

## License

MIT -- see [LICENSE](LICENSE).
