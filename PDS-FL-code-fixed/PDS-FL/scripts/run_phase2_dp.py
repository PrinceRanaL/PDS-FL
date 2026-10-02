"""
Formal differential-privacy ablation (Reviewer 2, Comment 9).

The main PDS-FL pipeline (src/phase2/federated.py) uses LightGBM tree
ensembles, whose leaf-value parameterisation makes a *clean*,
textbook Gaussian-mechanism DP analysis awkward (tree structure and
leaf count vary run-to-run, so "the parameter vector" is not a fixed-
dimensional quantity to clip and noise directly). Rather than bolt an
ad hoc, hard-to-analyse noise mechanism onto the tree ensemble, this
script demonstrates a formally-analysable DP-FedProx/DP-FedAvg
mechanism on a fixed-length linear model (weight vector w in R^d,
d = number of input features) trained with the SAME client
population, AQF/SemBF gating, and non-IID city partition as the main
pipeline. This isolates and correctly demonstrates the achievable
privacy-utility trade-off; integrating an equivalent per-leaf noise
mechanism into the tree-based pipeline is identified as future work
(see the manuscript's Limitations subsection).

Mechanism (standard DP-FedAvg/FedProx, McMahan et al.-style):
  1. Each round, every authenticated, non-duplicate client computes a
     local gradient-descent update to a *shared-dimension* weight
     vector w (E local epochs of plain/FedProx-regularised SGD on
     squared loss).
  2. The client's total update direction (w_local - w_global) is
     clipped to L2 norm <= C (bounding one client's maximum influence
     on the aggregate).
  3. The server sums the clipped updates and adds Gaussian noise with
     std. dev. sigma = (C / epsilon_round) * sqrt(2 * ln(1.25/delta))
     to the SUM (the standard Gaussian mechanism for a sum query of
     sensitivity C), then divides by the number of participants and
     applies the averaged, noised update to the global model.
  4. Over T rounds, total privacy loss is epsilon_total = T *
     epsilon_round under basic composition (a conservative bound;
     Renyi-DP/moments-accountant composition would be tighter for the
     same noise level).

Usage:
  python scripts/run_phase2_dp.py --csv data/aqi_india.csv --rounds 60 \
      --epsilons 0.05 0.1 0.5 1.0 5.0 --clip 2.0 --delta 1e-5
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np

from src.data.preprocessing import build_dataset, temporal_split, city_partitions
from src.phase2.aqf import AdaptiveQuotientFilter, make_credential
from src.phase2.sembf import SemaphoreBloomFilter
from src.utils.metrics import regression_metrics


def gaussian_sigma(clip: float, epsilon_round: float, delta: float) -> float:
    return (clip / epsilon_round) * math.sqrt(2 * math.log(1.25 / delta))


def local_sgd_update(w, X, y, mu, w_global, local_epochs, lr):
    w_local = w.copy()
    n = len(y)
    for _ in range(local_epochs):
        pred = X @ w_local
        grad = (X.T @ (pred - y)) / n + mu * (w_local - w_global)
        w_local = w_local - lr * grad
    return w_local


def run_dp_linear_fl(train_by_city, X_test, y_test, rounds, mu, lr,
                      local_epochs, clip, epsilon_round, delta, seed=0):
    d = next(iter(train_by_city.values()))[0].shape[1]
    # base_prediction is the known global label mean, fixed as a
    # constant offset (analogous to the tree ensemble's init score in
    # src/phase2/federated.py); w models the deviation from it, so
    # standardised (mean-zero) features can still express the label's
    # non-zero mean level.
    all_y = np.concatenate([y for _, y in train_by_city.values()])
    base_prediction = float(np.mean(all_y))
    w = np.zeros(d)
    aqf = AdaptiveQuotientFilter(capacity=64, target_fpr=0.001)
    for city in train_by_city:
        aqf.register(make_credential(city))
    sembf = SemaphoreBloomFilter(size=4096, num_hashes=4)
    rng = np.random.RandomState(seed)

    sigma = gaussian_sigma(clip, epsilon_round, delta) if epsilon_round else 0.0
    history = []
    for t in range(1, rounds + 1):
        sembf.reset()
        accepted = [c for c in train_by_city
                    if aqf.authenticate(make_credential(c)) and sembf.query_and_lock(c, t)]
        clipped_updates = []
        for city in accepted:
            X_c, y_c = train_by_city[city]
            y_c_centered = y_c - base_prediction
            w_local = local_sgd_update(w, X_c, y_c_centered, mu, w, local_epochs, lr)
            delta_w = w_local - w
            norm = np.linalg.norm(delta_w)
            if norm > clip:
                delta_w = delta_w * (clip / norm)
            clipped_updates.append(delta_w)

        if clipped_updates:
            summed = np.sum(clipped_updates, axis=0)
            if sigma > 0:
                summed = summed + rng.normal(0, sigma, size=summed.shape)
            w = w + summed / len(clipped_updates)

        pred_test = base_prediction + X_test @ w
        m = regression_metrics(y_test, pred_test)
        m["round"] = t
        m["n_participants"] = len(accepted)
        history.append(m)
    return history


def run(csv_path, rounds, epsilons, clip, delta, mu, lr, local_epochs, output):
    dataset = build_dataset(csv_path)
    train_df, test_df = temporal_split(dataset.df, test_frac=0.2)

    # Standardise features (shared scaler fit on the full training
    # partition) so that a single clip norm C is meaningful across
    # features of very different native scales (e.g. CO in ug/m3 vs.
    # AOD in [0,1]).
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train_df[dataset.feature_columns].values)

    train_by_city = {}
    for city, g in city_partitions(train_df):
        X_c = scaler.transform(g[dataset.feature_columns].values)
        y_c = g[dataset.target_column].values
        train_by_city[city] = (X_c, y_c)
    X_test = scaler.transform(test_df[dataset.feature_columns].values)
    y_test = test_df[dataset.target_column].values

    results = {}
    # epsilon_round=None => no-DP (infinite budget) reference baseline.
    for eps in [None] + list(epsilons):
        label = "no_dp" if eps is None else str(eps)
        print(f"\n=== epsilon_round={eps} ===")
        history = run_dp_linear_fl(train_by_city, X_test, y_test, rounds, mu,
                                    lr, local_epochs, clip, eps, delta)
        best = max(history, key=lambda h: h["r2"])
        print(f"  best test R2={best['r2']:.4f} at round {best['round']}")
        results[label] = {
            "epsilon_round": eps,
            "epsilon_total_basic_composition": None if eps is None else eps * rounds,
            "history": history,
            "best": best,
        }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--epsilons", type=float, nargs="+",
                         default=[0.05, 0.1, 0.5, 1.0, 5.0])
    parser.add_argument("--clip", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--output", default="results/phase2/dp_tradeoff.json")
    args = parser.parse_args()
    run(args.csv, args.rounds, args.epsilons, args.clip, args.delta,
        args.mu, args.lr, args.local_epochs, args.output)
