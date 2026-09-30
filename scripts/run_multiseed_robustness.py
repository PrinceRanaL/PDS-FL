"""
Multi-run robustness study (Reviewer 2, Comments 12-13): repeats Phase-1
model benchmarking over multiple independent train/test splits (distinct
random seeds), reporting mean +/- std and a 95% confidence interval for
each model's test R2/RMSE/MAE, then runs paired significance tests
(paired t-test, Wilcoxon signed-rank) between the best model and every
other candidate, plus a Friedman test across all models jointly.

Runtime note: to make N=`--n_runs` independent full-scale re-fits
tractable on a single CPU core, each run trains on a fixed-size random
subsample (`--subsample_size` rows, stratified by city) of the full
673,728-row Phase-1 training partition, rather than the complete
training set used for the headline Phase-1 numbers
(scripts/run_phase1.py). This is a documented robustness/statistical-
testing protocol, distinct from -- and not a replacement for -- the
main Phase-1 benchmark.

Usage: python scripts/run_multiseed_robustness.py --csv data/aqi_india.csv --n_runs 5
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.data.preprocessing import build_dataset, temporal_split
from src.utils.metrics import regression_metrics


def get_models(seed):
    return {
        "LinearRegression": LinearRegression(),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=18,
                                               n_jobs=-1, random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=seed, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=400, max_depth=-1, num_leaves=63,
                                   learning_rate=0.05, subsample=0.8,
                                   colsample_bytree=0.8, random_state=seed,
                                   n_jobs=-1, verbosity=-1),
    }


def one_run(dataset, seed: int, subsample_size: int, test_frac: float = 0.2):
    train_df, test_df = temporal_split(dataset.df, test_frac=test_frac)
    rng = np.random.RandomState(seed)
    if len(train_df) > subsample_size:
        idx = rng.choice(len(train_df), size=subsample_size, replace=False)
        train_df = train_df.iloc[idx]
    # SVR needs a further subsample to stay tractable; cap at 8000 rows.
    X_train = train_df[dataset.feature_columns].values
    y_train = train_df[dataset.target_column].values
    X_test = test_df[dataset.feature_columns].values
    y_test = test_df[dataset.target_column].values

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    run_results = {}
    for name, model in get_models(seed).items():
        if name == "SVR" and len(X_train_s) > 8000:
            sidx = rng.choice(len(X_train_s), size=8000, replace=False)
            model.fit(X_train_s[sidx], y_train[sidx])
        else:
            model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        run_results[name] = regression_metrics(y_test, pred)
    return run_results


def confidence_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    if n < 2:
        return mean, mean, mean
    sem = stats.sem(values)
    margin = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - margin, mean + margin


def run(csv_path: str, n_runs: int, subsample_size: int, output: str):
    dataset = build_dataset(csv_path)
    all_runs = []  # list of {model: {r2, rmse, mae}} per seed
    for seed in range(n_runs):
        print(f"--- run {seed+1}/{n_runs} (seed={seed}) ---")
        r = one_run(dataset, seed=seed, subsample_size=subsample_size)
        for name, m in r.items():
            print(f"  [{name}] R2={m['r2']:.4f} RMSE={m['rmse']:.3f} MAE={m['mae']:.3f}")
        all_runs.append(r)

    model_names = list(all_runs[0].keys())
    summary = {}
    r2_matrix = {name: [run[name]["r2"] for run in all_runs] for name in model_names}
    rmse_matrix = {name: [run[name]["rmse"] for run in all_runs] for name in model_names}
    mae_matrix = {name: [run[name]["mae"] for run in all_runs] for name in model_names}

    for name in model_names:
        r2_mean, r2_lo, r2_hi = confidence_interval(r2_matrix[name])
        summary[name] = {
            "r2_mean": r2_mean, "r2_std": float(np.std(r2_matrix[name], ddof=1)),
            "r2_95ci": [r2_lo, r2_hi],
            "rmse_mean": float(np.mean(rmse_matrix[name])),
            "rmse_std": float(np.std(rmse_matrix[name], ddof=1)),
            "mae_mean": float(np.mean(mae_matrix[name])),
            "mae_std": float(np.std(mae_matrix[name], ddof=1)),
        }

    best_model = max(model_names, key=lambda n: summary[n]["r2_mean"])

    # Paired significance tests: best model vs every other, on R2 across runs.
    pairwise = {}
    for name in model_names:
        if name == best_model:
            continue
        a = np.array(r2_matrix[best_model])
        b = np.array(r2_matrix[name])
        t_stat, t_p = stats.ttest_rel(a, b)
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        pairwise[f"{best_model}_vs_{name}"] = {
            "paired_t_stat": float(t_stat), "paired_t_pvalue": float(t_p),
            "wilcoxon_stat": float(w_stat), "wilcoxon_pvalue": float(w_p),
        }

    # Friedman test across all models jointly (on R2 across runs).
    friedman_stat, friedman_p = stats.friedmanchisquare(
        *[r2_matrix[name] for name in model_names]
    )

    result = {
        "n_runs": n_runs,
        "subsample_size": subsample_size,
        "best_model": best_model,
        "per_model_summary": summary,
        "pairwise_significance_vs_best": pairwise,
        "friedman_test_across_all_models": {
            "statistic": float(friedman_stat), "pvalue": float(friedman_p)
        },
        "raw_r2_per_run": r2_matrix,
    }
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--subsample_size", type=int, default=60000)
    parser.add_argument("--output", default="results/phase1/multiseed_robustness.json")
    args = parser.parse_args()
    run(args.csv, args.n_runs, args.subsample_size, args.output)
