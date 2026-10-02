"""
Preprocessing pipeline for the PDS-FL AQI dataset (Section 4.1.2 of the
paper):
  1. Residual missing-value imputation (per-city median, global-median fallback)
  2. Calendar features (month, day of week, hour)
  3. Feature scaling is applied by the Phase-1 / DP scripts (training-set
     statistics only); the federated LightGBM procedure uses unscaled features
  4. Per-city temporally ordered train/test split (last 20% of each city's
     timeline held out)
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from src.data.schema import load_raw

FEATURE_COLUMNS = [
    "pm25", "pm10", "so2", "no2", "co", "o3", "dust", "aod",
    "temperature", "humidity", "dew_point", "wind_speed",
    "pressure", "precipitation", "cloud_cover", "solar_radiation",
]
TARGET_COLUMN = "aqi"


@dataclass
class Dataset:
    df: pd.DataFrame
    feature_columns: list = field(default_factory=lambda: list(FEATURE_COLUMNS))
    target_column: str = TARGET_COLUMN

    @property
    def cities(self):
        return sorted(self.df["city"].unique().tolist())


def _impute(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df.groupby("city")[col].transform(
            lambda s: s.fillna(s.median())
        )
        # Fallback for cities where a whole column is missing.
        df[col] = df[col].fillna(df[col].median())
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month
        df["dayofweek"] = df["date"].dt.dayofweek
        if "hour" not in df.columns:
            df["hour"] = df["date"].dt.hour
    # If the dataset does not already provide a computed AQI column,
    # fall back to a simple max-sub-index proxy so the pipeline still
    # runs end-to-end; replace with the official CPCB AQI sub-index
    # formula if your CSV does not already include an 'aqi' column.
    if TARGET_COLUMN not in df.columns:
        df[TARGET_COLUMN] = df[["pm25", "pm10", "so2", "no2", "co", "o3"]].max(axis=1)
    return df


def build_dataset(csv_path: str) -> Dataset:
    df = load_raw(csv_path)
    cols_to_impute = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = _impute(df, [c for c in cols_to_impute if c in df.columns])
    df = _engineer_features(df)
    df = _impute(df, [TARGET_COLUMN])
    feature_columns = [c for c in FEATURE_COLUMNS if c in df.columns]
    for extra in ("month", "dayofweek", "hour"):
        if extra in df.columns:
            feature_columns.append(extra)
    return Dataset(df=df, feature_columns=feature_columns)


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Per-city temporal split: the last `test_frac` rows (by date, or by
    original row order if no date column) of each city form the test set.
    Mirrors Eq. (split) in the manuscript."""
    sort_col = "date" if "date" in df.columns else None
    train_parts, test_parts = [], []
    for city, g in df.groupby("city"):
        g = g.sort_values(sort_col) if sort_col else g
        n_test = max(1, int(len(g) * test_frac))
        train_parts.append(g.iloc[:-n_test])
        test_parts.append(g.iloc[-n_test:])
    return pd.concat(train_parts).reset_index(drop=True), \
        pd.concat(test_parts).reset_index(drop=True)


def city_partitions(df: pd.DataFrame):
    """Yield (city_name, city_df) for the non-IID federated client split
    used in Phase 2 -- one client per city (Section 3.1)."""
    for city, g in df.groupby("city"):
        yield city, g.reset_index(drop=True)
