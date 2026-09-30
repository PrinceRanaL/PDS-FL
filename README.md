# PDS-FL: A Probabilistic Data Structure based Secure Federated Learning Framework for Smart Cities

Reference implementation accompanying the manuscript *"PDS-FL: A
Probabilistic Data Structure based Secure Federated Learning Framework
for Smart Cities"* (Prince Rana, Amritpal Singh). Released under the
MIT license as part of the manuscript's Code Availability declaration.

> **Note on provenance.** The authors' original training scripts were
> lost prior to the revision reported here. This repository is a
> from-scratch, faithful re-implementation of the framework described
> in the manuscript (Sections 3-5): Phase-1 centralised model
> benchmarking, and Phase-2 secure federated deployment with the
> Adaptive Quotient Filter (AQF) and Semaphore Bloom Filter (SemBF)
> security layers plus a FedProx-vs-FedAvg federated gradient-boosting
> simulator. Where the exact original mechanism was not recoverable
> (notably, how FedProx's proximal term was realised for a tree-based
> LightGBM model rather than a fixed-length parameter vector), the
> design choice actually implemented is documented in the relevant
> module's docstring -- see `src/phase2/federated.py`.

## Repository structure

```
PDS-FL/
├── src/
│   ├── data/
│   │   ├── schema.py            # column auto-detection / aliasing
│   │   └── preprocessing.py     # cleaning, feature engineering, splits
│   ├── phase1/
│   │   └── train_baselines.py   # LR, SVR, RandomForest, XGBoost, LightGBM benchmarking
│   ├── phase2/
│   │   ├── aqf.py               # Adaptive Quotient Filter (client authentication)
│   │   ├── sembf.py             # Semaphore Bloom Filter (round-scoped dedup)
│   │   └── federated.py         # FedProx / FedAvg federated boosting simulator
│   └── utils/
│       └── metrics.py           # R2 / RMSE / MAE, bias-variance gap
├── scripts/
│   ├── run_phase1.py                  # centralised model selection
│   ├── run_phase2_convergence.py      # FedProx vs FedAvg round-wise comparison
│   └── measure_aqf_fpr.py             # AQF false-positive-rate + confidence interval
├── tests/
│   └── test_smoke.py            # synthetic-data end-to-end tests (no dataset needed)
├── results/                     # generated outputs (git-ignored except the headline plot)
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone <this-repo-url>
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

**Reproducibility note:** on the released dataset, both aggregation
rules improve for roughly the first 35-40 rounds (FedProx reaching a
marginally higher peak test R2 than FedAvg), after which continued
training without early stopping causes both to degrade due to client
drift under the non-IID, 29-city partition -- FedAvg degrades faster
and further than FedProx, consistent with the proximal term's intended
effect. We therefore report **best-checkpoint (early-stopped)**
metrics as the headline convergence result; the raw per-round CSV
(including the post-peak degradation) is also saved for full
transparency.

### 4. AQF false-positive-rate measurement

```bash
python scripts/measure_aqf_fpr.py --n_legit 29 --n_attacks 100000
```
Registers the legitimate client population, then simulates the given
number of unauthorised-client authentication attempts, reporting the
empirical false-positive rate with a 95% Wilson-score confidence
interval alongside AQF's theoretical FPR bound.

### 5. Additional analyses added in the second revision

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
