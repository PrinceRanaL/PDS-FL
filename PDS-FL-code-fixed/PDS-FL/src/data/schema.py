"""
Column auto-detection for the Kaggle "Air Quality Dataset: Indian Cities
(2022-2025)" source (and similar CPCB/IMD-style CSVs).

The public Kaggle export's exact column names can vary slightly between
versions, so rather than hard-coding names, we match on a set of known
aliases (case-insensitive, punctuation-insensitive). If your CSV uses
different names, add them to the ALIASES dict below -- everything
downstream (preprocessing.py, phase1, phase2) reads from the canonical
names this module produces.
"""

import re
import pandas as pd

# Canonical name -> list of aliases seen in public AQI datasets.
ALIASES = {
    "city": ["city", "station_city", "location", "city_name"],
    "date": ["date", "datetime", "timestamp", "date_time"],
    "hour": ["hour", "hr", "time"],
    "pm25": ["pm2.5", "pm25", "pm_2_5", "pm2_5", "pm2_5_ugm3"],
    "pm10": ["pm10", "pm_10", "pm10_ugm3"],
    "so2": ["so2", "so_2", "so2_ugm3"],
    "no2": ["no2", "no_2", "nox", "no2_ugm3"],
    "co": ["co", "carbon_monoxide", "co_ugm3"],
    "o3": ["o3", "ozone", "o3_ugm3"],
    "dust": ["dust", "dust_ugm3"],
    "aod": ["aod"],
    "humidity": ["humidity", "relative_humidity", "rh", "h", "humidity_percent"],
    "dew_point": ["dew_point", "dew_point_c", "dewpoint"],
    "wind_speed": ["wind_speed", "windspeed", "wind_spd", "ws", "wind_gusts_kmh"],
    "pressure": ["pressure", "mslp", "sea_level_pressure", "slp", "pressure_msl_hpa"],
    "precipitation": ["precipitation", "precipitation_mm", "rainfall"],
    "cloud_cover": ["cloud_cover", "cloud_cover_percent"],
    "solar_radiation": ["solar_radiation", "solar_rad", "radiation", "srad"],
    "temperature": ["temperature", "temp", "temperature_c", "t"],
    "aqi": ["aqi", "air_quality_index", "us_aqi"],
}


def _normalise(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", col.lower())


def detect_columns(df: pd.DataFrame) -> dict:
    """Return {canonical_name: actual_column_name_in_df} for every
    canonical field we can find. Raises with a helpful message listing
    unmatched required fields."""
    normalised_lookup = {_normalise(c): c for c in df.columns}
    found = {}
    for canon, aliases in ALIASES.items():
        for alias in aliases:
            key = _normalise(alias)
            if key in normalised_lookup:
                found[canon] = normalised_lookup[key]
                break
    return found


REQUIRED_FOR_TRAINING = [
    "city", "pm25", "pm10", "so2", "no2", "co", "o3",
    "humidity", "wind_speed", "pressure", "aqi",
]


def validate_schema(found: dict, required=REQUIRED_FOR_TRAINING):
    missing = [f for f in required if f not in found]
    if missing:
        raise ValueError(
            "Could not auto-detect the following required columns: "
            f"{missing}. Please open src/data/schema.py and add the "
            "actual column name(s) from your CSV to the ALIASES dict "
            "for each missing field, e.g. ALIASES['pm25'].append("
            "'YourActualColumnName')."
        )


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    found = detect_columns(df)
    validate_schema(found)
    # Rename to canonical names; keep any extra columns untouched.
    rename_map = {v: k for k, v in found.items()}
    df = df.rename(columns=rename_map)
    return df
