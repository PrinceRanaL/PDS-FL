"""
City-wise evaluation (Section 5.3.8): (a) per-city test RMSE/R2/MAE of the
FedProx global model at a fixed round; (b) heterogeneity indicators: per-city
mean/std of AQI and the per-city local linear-regression coefficients
(standardised features), showing how the pollutant-AQI relationship differs
across cities.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from src.data.preprocessing import build_dataset, temporal_split, city_partitions
from src.phase2.federated import FederatedBoostingConfig, FederatedBoostingSimulator
from src.utils.metrics import regression_metrics


def run(csv, rounds, mu, out):
    ds = build_dataset(csv)
    tr, te = temporal_split(ds.df, 0.2)
    fc, tc = ds.feature_columns, ds.target_column
    train_by_city = {c: (g[fc].values, g[tc].values) for c, g in city_partitions(tr)}
    sim = FederatedBoostingSimulator(train_by_city, te[fc].values, te[tc].values,
                                      FederatedBoostingConfig(mu=mu, rounds=rounds))
    sim.run(verbose=False)
    pred = sim.cum_pred_test
    sc = StandardScaler().fit(tr[fc].values)
    rows = []
    cities = te["city"].values
    for c in sorted(set(cities)):
        m = cities == c
        met = regression_metrics(te[tc].values[m], pred[m])
        g = tr[tr["city"] == c]
        lr = LinearRegression().fit(sc.transform(g[fc].values), g[tc].values)
        rows.append({"city": c, **met, "train_mean_aqi": float(g[tc].mean()), "train_std_aqi": float(g[tc].std()),
                     "test_mean_aqi": float(te[tc].values[m].mean()),
                     "coef": dict(zip(fc, map(float, lr.coef_)))})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"rounds": rounds, "mu": mu, "cities": rows}, open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--rounds", type=int, default=38)
    ap.add_argument("--mu", type=float, default=1.0); ap.add_argument("--output", default="results/phase2/per_city.json")
    a = ap.parse_args(); run(a.csv, a.rounds, a.mu, a.output)
