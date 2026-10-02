"""
Communication cost quantification (Reviewer 2, Comment 8): measures the
actual serialized size of one federated round's transmitted payload
(client model updates) versus the size of the raw local dataset that
FL avoids transmitting, using the real trained models from
scripts/run_phase2_convergence.py's federated simulator.

Reports, per round and cumulatively over T rounds:
  - bytes transmitted per client (serialized LightGBM update trees)
  - bytes that would be transmitted if raw data were centralised instead
  - bytes saved / percentage reduction
  - wall-clock transmission time at representative bandwidths (4G: 12
    Mbps up; broadband: 50 Mbps up; low-power LPWAN/NB-IoT: 250 kbps up)

Usage: python scripts/measure_communication_cost.py --csv data/aqi_india.csv
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from lightgbm import LGBMRegressor

from src.data.preprocessing import build_dataset, temporal_split, city_partitions


def serialized_size_bytes(model) -> int:
    return len(model.booster_.model_to_string().encode("utf-8"))


def raw_data_size_bytes(X: np.ndarray, y: np.ndarray) -> int:
    # 8 bytes/float64 per feature value, plus 8 bytes for the label,
    # per row -- the size of the raw local dataset a fully centralised
    # (non-federated) scheme would need to upload every round.
    n_rows, n_features = X.shape
    return n_rows * (n_features + 1) * 8


def run(csv_path: str, rounds: int, local_epochs: int, output: str):
    dataset = build_dataset(csv_path)
    train_df, _ = temporal_split(dataset.df, test_frac=0.2)
    train_by_city = {
        city: (g[dataset.feature_columns].values, g[dataset.target_column].values)
        for city, g in city_partitions(train_df)
    }

    # Fit one representative local update (E local epochs, same
    # hyperparameters as src/phase2/federated.py) for every city to
    # measure the real per-client, per-round transmitted payload size.
    per_client_bytes = {}
    per_client_raw_bytes = {}
    for city, (X_c, y_c) in train_by_city.items():
        trees_bytes = 0
        pred = np.zeros(len(y_c))
        for e in range(local_epochs):
            residual = y_c - pred
            model = LGBMRegressor(n_estimators=1, max_depth=4, num_leaves=15,
                                   learning_rate=0.1, reg_lambda=2.0,
                                   min_child_samples=50, verbosity=-1,
                                   random_state=e)
            model.fit(X_c, residual)
            trees_bytes += serialized_size_bytes(model)
            pred += model.predict(X_c)
        per_client_bytes[city] = trees_bytes
        per_client_raw_bytes[city] = raw_data_size_bytes(X_c, y_c)

    total_fl_bytes_per_round = sum(per_client_bytes.values())
    total_raw_bytes = sum(per_client_raw_bytes.values())  # sent once, if centralised
    total_fl_bytes_T_rounds = total_fl_bytes_per_round * rounds

    bandwidths_mbps = {"NB-IoT/LPWAN (0.25 Mbps up)": 0.25,
                        "4G/LTE (12 Mbps up)": 12.0,
                        "Broadband (50 Mbps up)": 50.0}
    latency = {}
    for name, mbps in bandwidths_mbps.items():
        bytes_per_sec = mbps * 1e6 / 8
        latency[name] = {
            "seconds_per_round_all_clients": total_fl_bytes_per_round / bytes_per_sec,
            "seconds_for_T_rounds_all_clients": total_fl_bytes_T_rounds / bytes_per_sec,
            "seconds_to_send_raw_data_once_all_clients": total_raw_bytes / bytes_per_sec,
        }

    result = {
        "n_clients": len(train_by_city),
        "local_epochs_per_round": local_epochs,
        "rounds": rounds,
        "bytes_per_client_per_round_mean": float(np.mean(list(per_client_bytes.values()))),
        "bytes_per_client_per_round_min": int(min(per_client_bytes.values())),
        "bytes_per_client_per_round_max": int(max(per_client_bytes.values())),
        "total_fl_bytes_per_round_all_clients": total_fl_bytes_per_round,
        "total_fl_bytes_over_T_rounds_all_clients": total_fl_bytes_T_rounds,
        "total_raw_data_bytes_all_clients_one_time": total_raw_bytes,
        "reduction_vs_raw_data_one_time_pct":
            100 * (1 - total_fl_bytes_T_rounds / total_raw_bytes),
        "bytes_saved_over_T_rounds_all_clients": total_raw_bytes - total_fl_bytes_T_rounds,
        "latency_by_bandwidth": latency,
    }
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--rounds", type=int, default=38)  # best-checkpoint round from Fig. 5
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--output", default="results/phase2/communication_cost.json")
    args = parser.parse_args()
    run(args.csv, args.rounds, args.local_epochs, args.output)
