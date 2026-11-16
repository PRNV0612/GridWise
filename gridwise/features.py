"""Feature engineering, shared by training and the API.

Takes the long-format frame from etl.py and produces the model matrix X and
target y = log1p(kwh). Training and serving both import build_features, so
they cannot drift apart.

Feature groups:
- Calendar: hour, day-of-week, month, weekend flag, public holiday flag
  (per site country, via the `holidays` library).
- Weather: air temp, dew point, wind speed, cloud cover, 24h rolling mean
  temperature, cooling degree hours (base 18C), heating degree hours
  (base 15.5C).
- Building: building_id (categorical), primary use (categorical),
  log floor area, building age, site_id (categorical).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gridwise import config

# BDG2 site -> country, for the holidays library. Most sites are US;
# the two non-US sites we know of are mapped explicitly.
SITE_COUNTRY = {
    "Panther": "US", "Fox": "US", "Moose": "US", "Cockatoo": "US",
    "Horse": "US", "Robin": "US", "Wolf": "US", "Bull": "US", "Hog": "US",
    "Bear": "US", "Eagle": "US", "Shrew": "US", "Rat": "US", "Owl": "US",
    "Peacock": "US", "Hawk": "US", "Gator": "US", "Lamb": "CA", "Goby": "IE",
    "Lapwing": "UK",
}
DEFAULT_COUNTRY = "US"

COOLING_BASE_C = 18.0
HEATING_BASE_C = 15.5

CATEGORICAL_COLS = ["building_id", "primaryspaceusage", "site_id"]
FEATURE_COLS = [
    "hour", "dayofweek", "month", "is_weekend", "is_holiday",
    "airTemperature", "dewTemperature", "windSpeed", "cloudCoverage",
    "temp_rolling_24h", "cdh", "hdh",
    "log_sqm", "building_age",
] + CATEGORICAL_COLS
# Lag features are only computable where recent actual load history exists.
# The weather-only feature set serves measurement & verification / anomaly
# detection; the day-ahead set (weather + lags) serves operational forecasting
# and is the one compared against the seasonal-naive baseline.
LAG_FEATURE_COLS = ["lag_24h", "lag_168h", "load_rolling_24h"]
DAY_AHEAD_FEATURE_COLS = FEATURE_COLS + LAG_FEATURE_COLS
TARGET_COL = "log1p_kwh"


def _holiday_flag(df: pd.DataFrame) -> pd.Series:
    """Public-holiday flag per row, resolved by site country."""
    import holidays as holidays_lib

    country = df["site_id"].map(
        lambda s: SITE_COUNTRY.get(str(s).split("_")[0], DEFAULT_COUNTRY)
    )
    out = pd.Series(False, index=df.index, dtype=bool)
    year = df["timestamp"].dt.year
    for yr in sorted(year.unique()):
        for c in country.unique():
            mask = year.eq(yr) & country.eq(c)
            if not mask.any():
                continue
            cal = holidays_lib.country_holidays(c, years=[int(yr)])
            local_dates = df.loc[mask, "timestamp"].dt.date
            out.loc[mask] = local_dates.isin(cal).values
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with FEATURE_COLS + TARGET_COL, index aligned to df.

    `df` must contain: timestamp, building_id, kwh, primaryspaceusage, sqm,
    yearbuilt, site_id, airTemperature, dewTemperature, windSpeed, cloudCoverage.
    Weather must already be joined (etl.py output satisfies this).
    """
    out = pd.DataFrame(index=df.index)

    # Calendar
    ts = df["timestamp"]
    out["hour"] = ts.dt.hour.astype("int8")
    out["dayofweek"] = ts.dt.dayofweek.astype("int8")
    out["month"] = ts.dt.month.astype("int8")
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("int8")
    out["is_holiday"] = _holiday_flag(df).astype("int8")

    # Weather
    out["airTemperature"] = df["airTemperature"].astype("float32")
    out["dewTemperature"] = df["dewTemperature"].astype("float32")
    out["windSpeed"] = df["windSpeed"].astype("float32")
    out["cloudCoverage"] = df["cloudCoverage"].astype("float32")

    # 24h rolling mean temperature per building (min_periods low: cold start ok)
    out["temp_rolling_24h"] = (
        df.groupby("building_id")["airTemperature"]
        .transform(lambda s: s.rolling(24, min_periods=3).mean())
        .astype("float32")
    )
    out["cdh"] = (
        (out["airTemperature"] - COOLING_BASE_C).clip(lower=0).astype("float32")
    )
    out["hdh"] = (
        (HEATING_BASE_C - out["airTemperature"]).clip(lower=0).astype("float32")
    )

    # Building
    out["log_sqm"] = np.log1p(df["sqm"].astype("float32"))
    out["building_age"] = (config.TRAIN_YEAR - df["yearbuilt"]).astype("float32")
    for c in CATEGORICAL_COLS:
        out[c] = df[c].astype("category")

    # Target
    out[TARGET_COL] = np.log1p(df["kwh"].astype("float64"))

    # Lag features (day-ahead model only): previous day, previous week,
    # trailing 24h mean load. Computed on a timestamp-sorted frame per
    # building; gaps from dropped fault hours become NaN and are simply
    # not used as training rows.
    sorted_idx = df.sort_values(["building_id", "timestamp"]).index
    kwh_s = df.loc[sorted_idx, "kwh"].astype("float32")
    grp = df.loc[sorted_idx, "building_id"]
    ts_s = df.loc[sorted_idx, "timestamp"]

    # shift by wall-clock delta so DST/dropped fault hours resolve correctly
    lag24 = kwh_s.copy()
    lag168 = kwh_s.copy()
    roll24 = kwh_s.copy()
    for name, hours, series in (("lag_24h", 24, lag24), ("lag_168h", 168, lag168)):
        shifted = []
        for _, pos in grp.groupby(grp, sort=False).indices.items():
            idx = np.asarray(pos)
            t = ts_s.iloc[idx].to_numpy()
            v = series.iloc[idx].to_numpy()
            target_t = t - np.timedelta64(hours, "h")
            j = np.searchsorted(t, target_t, side="left")
            valid = (j < len(t)) & (j > 0)
            vals = np.full(len(t), np.nan, dtype="float32")
            match = valid & (t[np.clip(j, 0, len(t) - 1)] == target_t)
            vals[match] = v[np.clip(j, 0, len(t) - 1)][match]
            shifted.append(pd.Series(vals, index=series.index[idx]))
        out_col = pd.concat(shifted).reindex(df.index)
        out[name] = out_col.astype("float32").to_numpy()
    # trailing 24h mean (excludes current hour)
    roll = kwh_s.groupby(grp, sort=False).transform(
        lambda s: s.rolling(24, min_periods=6).mean().shift(1)
    )
    out["load_rolling_24h"] = roll.reindex(df.index).astype("float32").to_numpy()

    return out
