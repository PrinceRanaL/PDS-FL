"""
Phase 1: Data Preprocessing and Optimal Model Identification
(Section 4.1 of the paper). Trains five candidate regressors on the
centralised (pooled across all 29 cities) dataset, reports R2/RMSE/MAE
and the train-test bias-variance gap, and selects the best model
(by test R2) for federated deployment in Phase 2.
"""

import argparse
import json
import os

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import build_dataset, temporal_split
from src.utils.metrics import regression_metrics, bias_variance_gap

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


def get_candidate_models(random_state: int = 42) -> dict:
    models = {
        "LinearRegression": LinearRegression(),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1),
        "RandomForest": RandomForestRegressor(
            n_estimators=150, max_depth=18, n_jobs=-1,
            random_state=random_state,
        ),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1,
        )
    if _HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=400, max_depth=-1, num_leaves=63,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, n_jobs=-1,
        )
    return models


def run_phase1(csv_path: str, output_dir: str = "results/phase1",
                test_frac: float = 0.2, random_state: int = 42) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    dataset = build_dataset(csv_path)
    train_df, test_df = temporal_split(dataset.df, test_frac=test_frac)

    X_train = train_df[dataset.feature_columns].values
    y_train = train_df[dataset.target_column].values
    X_test = test_df[dataset.feature_columns].values
    y_test = test_df[dataset.target_column].values

    # Linear/SVR benefit from scaling; tree ensembles do not need it but
    # scaling is harmless for them, so we apply one shared scaler for
    # simplicity and reproducibility.
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    fitted = {}
    for name, model in get_candidate_models(random_state).items():
        if name == "SVR" and len(X_train_s) > 20000:
            # SVR training time is superlinear in sample count; the
            # manuscript's SVR baseline is fit on a fixed-size random
            # subsample of the training set (still evaluated on the
            # full held-out test set) to keep Phase-1 benchmarking
            # tractable at this dataset scale.
            rng = np.random.RandomState(random_state)
            idx = rng.choice(len(X_train_s), size=20000, replace=False)
            model.fit(X_train_s[idx], y_train[idx])
        else:
            model.fit(X_train_s, y_train)
        train_pred = model.predict(X_train_s)
        test_pred = model.predict(X_test_s)
        train_m = regression_metrics(y_train, train_pred)
        test_m = regression_metrics(y_test, test_pred)
        gap = bias_variance_gap(train_m, test_m)
        results[name] = {"train": train_m, "test": test_m, **gap}
        fitted[name] = model
        print(f"[{name}] test R2={test_m['r2']:.4f} "
              f"RMSE={test_m['rmse']:.4f} MAE={test_m['mae']:.4f}")

    best_name = max(results, key=lambda k: results[k]["test"]["r2"])
    print(f"\nBest model by test R2: {best_name}")

    with open(os.path.join(output_dir, "phase1_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(fitted[best_name], os.path.join(output_dir, "best_model.joblib"))
    with open(os.path.join(output_dir, "best_model_name.txt"), "w") as f:
        f.write(best_name)
    joblib.dump(dataset.feature_columns,
                os.path.join(output_dir, "feature_columns.joblib"))

    return {"results": results, "best_model": best_name}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: centralised model benchmarking")
    parser.add_argument("--csv", required=True, help="Path to the AQI dataset CSV")
    parser.add_argument("--output_dir", default="results/phase1")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_phase1(args.csv, args.output_dir, args.test_frac, args.seed)
