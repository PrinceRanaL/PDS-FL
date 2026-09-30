"""
Runs the FedProx vs. FedAvg round-wise convergence comparison
(Section 5.3.4 of the paper / Fig. 5) and saves:
  - results/phase2/convergence_fedprox_vs_fedavg.csv
  - results/phase2/convergence_fedprox_vs_fedavg.png
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.preprocessing import build_dataset, temporal_split, city_partitions
from src.phase2.federated import FederatedBoostingConfig, FederatedBoostingSimulator


def run_phase2(csv_path: str, output_dir: str = "results/phase2",
               rounds: int = 60, mu: float = 0.5, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    dataset = build_dataset(csv_path)
    train_df, test_df = temporal_split(dataset.df, test_frac=0.2)

    train_by_city = {
        city: (g[dataset.feature_columns].values, g[dataset.target_column].values)
        for city, g in city_partitions(train_df)
    }
    X_test = test_df[dataset.feature_columns].values
    y_test = test_df[dataset.target_column].values

    results = {}
    for label, mu_val in [("FedAvg", 0.0), ("FedProx", mu)]:
        print(f"\n=== Running {label} (mu={mu_val}) for {rounds} rounds ===")
        cfg = FederatedBoostingConfig(mu=mu_val, rounds=rounds, seed=seed)
        sim = FederatedBoostingSimulator(train_by_city, X_test, y_test, cfg)
        history = sim.run(verbose=True)
        results[label] = history

    # Combine into one dataframe for plotting / CSV export.
    rows = []
    for label, history in results.items():
        for h in history:
            rows.append({"aggregation": label, **h})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "convergence_fedprox_vs_fedavg.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    metrics = [("r2", "Test $R^2$"), ("rmse", "Test RMSE"), ("mae", "Test MAE")]
    colors = {"FedProx": "#1f77b4", "FedAvg": "#d62728"}
    for ax, (key, label) in zip(axes, metrics):
        for agg in ["FedProx", "FedAvg"]:
            sub = df[df.aggregation == agg]
            ax.plot(sub["round"], sub[key], label=agg, color=colors[agg], linewidth=1.6)
        ax.set_xlabel("Federated Round $t$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Round-wise Convergence: FedProx vs. FedAvg")
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "convergence_fedprox_vs_fedavg.png")
    fig.savefig(fig_path, dpi=200)
    print(f"\nSaved plot to {fig_path}")

    summary = {}
    for agg in ["FedProx", "FedAvg"]:
        sub = df[df.aggregation == agg]
        final = sub.iloc[-1]
        # round-to-round delta R2 stability (std of consecutive differences)
        delta_r2 = sub["r2"].diff().dropna()
        summary[agg] = {
            "final_r2": float(final["r2"]),
            "final_rmse": float(final["rmse"]),
            "final_mae": float(final["mae"]),
            "delta_r2_std": float(delta_r2.std()),
            "delta_r2_mean_abs_last10": float(delta_r2.tail(10).abs().mean()),
        }
    with open(os.path.join(output_dir, "convergence_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: FedProx vs FedAvg convergence comparison")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_dir", default="results/phase2")
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_phase2(args.csv, args.output_dir, args.rounds, args.mu, args.seed)
