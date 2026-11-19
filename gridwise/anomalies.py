"""Anomaly detection: residual-based flags on top of the weather-only model.

Why the weather-only model (not day-ahead): the day-ahead model consumes
recent load lags, so an anomalous reading leaks into its own prediction and
the residual shrinks -- the anomaly hides. The weather-only model predicts
what the building *should* use from exogenous signals only, so the residual
is a clean measurement & verification signal.

Detection layers:
1. Residual rule -- robust z-score of the log-space residual per building,
   z = (r - median_168h) / (1.4826 * MAD_168h), flag z > 3 for >= 3
   consecutive hours. Robust stats so a huge outlier cannot inflate the
   scale and mask itself.
2. Night baseload drift -- 00:00-05:00 hourly load vs the trailing 4-week
   night median; > 25% above flags sustained baseload growth (equipment
   left on, schedule change).
3. Stuck meter -- identical consecutive kWh readings for >= 6h (meter
   quantization repeats 1-2h, real load wobbles).

Run:  .venv/bin/python -m gridwise.anomalies
Writes models/anomalies.parquet (per-building hourly flags) and prints a
summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gridwise import config
from gridwise.features import FEATURE_COLS, TARGET_COL, build_features
from gridwise.train import load_frame

MAD_SCALE = 1.4826  # MAD -> consistent std estimate for Gaussian noise
STUCK_METER_HOURS = 6


def robust_z(resid: pd.Series) -> pd.Series:
    """Rolling robust z-score of log-space residuals, per building.

    Window is trailing 168h and excludes the current hour (shift(1)), so the
    score of hour t uses only the 168 hours before it.
    """
    grouped = resid.groupby(level="building_id", sort=False)

    def _med(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(config.ROLLED_STATS_WINDOW_HOURS,
                                  min_periods=24).median()

    med = grouped.transform(_med)
    mad = grouped.transform(
        lambda s: (s.shift(1) - _med(s)).abs()
        .rolling(config.ROLLED_STATS_WINDOW_HOURS, min_periods=24).median()
    )
    return (resid - med) / (MAD_SCALE * mad.replace(0, np.nan))


def consecutive_run_mask(flags: pd.Series, min_len: int) -> pd.Series:
    """True where flags hold for >= min_len consecutive hours per building."""
    bld = flags.index.get_level_values("building_id")
    run_id = flags.groupby(bld, sort=False).transform(
        lambda v: (v != v.shift(1)).cumsum()
    )
    run_len = flags.groupby([bld, run_id], sort=False).transform("size")
    return flags & (run_len >= min_len)


def night_baseload_flags(df: pd.DataFrame) -> pd.Series:
    """Flag night hours (00:00-05:00) where load exceeds the trailing
    4-week night median by more than NIGHT_BASELOAD_DRIFT."""
    s = df.set_index(["building_id", "timestamp"])["kwh"].astype("float64")
    hours = s.index.get_level_values("timestamp").hour
    s = s[(hours >= 0) & (hours <= 5)]
    med = s.groupby(level="building_id", sort=False).transform(
        lambda v: v.shift(1)
        .rolling(4 * config.ROLLED_STATS_WINDOW_HOURS, min_periods=7 * 24)
        .median()
    )
    drift = (s - med) / med
    return (drift > config.NIGHT_BASELOAD_DRIFT).fillna(False)


def stuck_meter_flags(df: pd.DataFrame) -> pd.Series:
    """Flag >= 6h runs of identical kWh readings per building."""
    flat = df[["building_id", "timestamp", "kwh"]].sort_values(
        ["building_id", "timestamp"]
    )
    grp = flat.groupby("building_id", sort=False)
    is_stuck = flat["kwh"].eq(grp["kwh"].shift(1)) & flat["kwh"].notna()
    run_id = is_stuck.groupby(flat["building_id"], sort=False).transform(
        lambda v: (v != v.shift(1)).cumsum()
    )
    run_len = is_stuck.groupby(
        [flat["building_id"], run_id], sort=False
    ).transform("size")
    flags = is_stuck & run_len.ge(STUCK_METER_HOURS)
    return pd.Series(
        flags.to_numpy(),
        index=pd.MultiIndex.from_frame(flat[["building_id", "timestamp"]]),
    )


def run() -> pd.DataFrame:
    import joblib

    df = load_frame()
    feats = build_features(df)
    for c in ("primaryspaceusage", "building_id", "site_id"):
        feats[c] = feats[c].cat.add_categories([]).fillna("Unknown")

    model = joblib.load(config.MODELS_DIR / "weather_model.joblib")
    resid = feats[TARGET_COL].astype("float64") - pd.Series(
        model.predict(feats[FEATURE_COLS]), index=feats.index
    )
    resid.index = df.set_index(["building_id", "timestamp"]).index

    df_mi = df.set_index(["building_id", "timestamp"]).sort_index()

    z = robust_z(resid.sort_index())
    residual_flags = consecutive_run_mask(
        z > config.ANOMALY_Z_THRESHOLD, config.ANOMALY_MIN_RUN_HOURS
    )

    flat = df_mi.reset_index()
    night = night_baseload_flags(flat)
    stuck = stuck_meter_flags(flat)

    flags = pd.DataFrame(index=df_mi.index)
    flags["resid_log"] = resid.sort_index().astype("float32")
    flags["z"] = z.astype("float32")
    flags["anom_residual"] = residual_flags.reindex(
        flags.index, fill_value=False
    ).fillna(False)
    flags["anom_night_baseload"] = night.reindex(
        flags.index, fill_value=False
    ).fillna(False)
    flags["anom_stuck_meter"] = stuck.reindex(
        flags.index, fill_value=False
    ).fillna(False)
    flags["kwh"] = df_mi["kwh"].astype("float32")
    flags["primaryspaceusage"] = df_mi["primaryspaceusage"]
    flags = flags.reset_index()

    summary = {
        "buildings": int(flags.building_id.nunique()),
        "residual_hours": int(flags.anom_residual.sum()),
        "residual_buildings": int(
            flags[flags.anom_residual].building_id.nunique()
        ),
        "night_baseload_hours": int(flags.anom_night_baseload.sum()),
        "night_baseload_buildings": int(
            flags[flags.anom_night_baseload].building_id.nunique()
        ),
        "stuck_hours": int(flags.anom_stuck_meter.sum()),
        "stuck_buildings": int(
            flags[flags.anom_stuck_meter].building_id.nunique()
        ),
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / "anomalies.parquet"
    flags.to_parquet(path, index=False)
    print(f"[anomalies] wrote {path}")
    print(pd.Series(summary).to_string())
    return flags


if __name__ == "__main__":
    run()
