"""
Smoke tests using small synthetic data (no real dataset required), so
that anyone cloning the repository can verify the pipeline runs
end-to-end before pointing it at the real Kaggle CSV.

Run with: pytest tests/test_smoke.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.data.schema import detect_columns, validate_schema
from src.phase2.aqf import AdaptiveQuotientFilter
from src.phase2.sembf import SemaphoreBloomFilter
from src.phase2.federated import FederatedBoostingConfig, FederatedBoostingSimulator


def make_synthetic_df(n_cities=4, n_per_city=500, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for c in range(n_cities):
        for i in range(n_per_city):
            pm25 = rng.uniform(10, 300)
            pm10 = pm25 * rng.uniform(1.0, 1.5)
            row = {
                "city": f"city_{c}",
                "pm2_5_ugm3": pm25,
                "pm10_ugm3": pm10,
                "so2_ugm3": rng.uniform(1, 50),
                "no2_ugm3": rng.uniform(1, 80),
                "co_ugm3": rng.uniform(100, 2000),
                "o3_ugm3": rng.uniform(1, 100),
                "humidity_percent": rng.uniform(20, 90),
                "pressure_msl_hpa": rng.uniform(990, 1020),
                "wind_gusts_kmh": rng.uniform(0, 40),
                "us_aqi": pm25 * 0.8 + rng.normal(0, 5),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def test_schema_detection():
    df = make_synthetic_df()
    found = detect_columns(df)
    validate_schema(found)
    assert "pm25" in found and found["pm25"] == "pm2_5_ugm3"
    assert "aqi" in found and found["aqi"] == "us_aqi"


def test_aqf_authenticates_registered_and_rejects_unknown():
    aqf = AdaptiveQuotientFilter(capacity=32, target_fpr=0.01)
    for cid in ["city_0", "city_1", "city_2"]:
        aqf.register(cid)
    assert aqf.authenticate("city_0") is True
    assert aqf.authenticate("city_1") is True
    # An unregistered id should almost always be rejected (probabilistic,
    # so we just check it's not universally accepted).
    rejections = sum(not aqf.authenticate(f"intruder_{i}") for i in range(200))
    assert rejections > 190


def test_sembf_blocks_duplicate_within_round_but_not_across_rounds():
    sembf = SemaphoreBloomFilter(size=1024, num_hashes=3)
    sembf.reset()
    assert sembf.query_and_lock("city_0", round_idx=1) is True
    assert sembf.query_and_lock("city_0", round_idx=1) is False  # duplicate, same round
    sembf.reset()
    assert sembf.query_and_lock("city_0", round_idx=2) is True  # new round, allowed again


def test_federated_simulator_runs_and_improves():
    df = make_synthetic_df(n_cities=3, n_per_city=300)
    feature_cols = ["pm2_5_ugm3", "pm10_ugm3", "so2_ugm3", "no2_ugm3",
                     "co_ugm3", "o3_ugm3", "humidity_percent",
                     "pressure_msl_hpa", "wind_gusts_kmh"]
    train_by_city = {}
    test_X, test_y = [], []
    for city, g in df.groupby("city"):
        split = int(len(g) * 0.8)
        train_by_city[city] = (
            g[feature_cols].values[:split], g["us_aqi"].values[:split]
        )
        test_X.append(g[feature_cols].values[split:])
        test_y.append(g["us_aqi"].values[split:])
    X_test = np.concatenate(test_X)
    y_test = np.concatenate(test_y)

    cfg = FederatedBoostingConfig(mu=0.5, rounds=5, local_epochs=2)
    sim = FederatedBoostingSimulator(train_by_city, X_test, y_test, cfg)
    history = sim.run(verbose=False)
    assert len(history) == 5
    # R2 should improve from round 1 to round 5 on this easy synthetic task.
    assert history[-1]["r2"] > history[0]["r2"]


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-v"]))
