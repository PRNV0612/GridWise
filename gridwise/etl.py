"""ETL: load BDG2 electricity meters, clean, join weather/metadata, write parquet.

Pipeline:
1. Load `cleaned/electricity_cleaned.csv` (wide: one column per meter) and melt to long.
2. Join building metadata (primary use, floor area, year built, site, timezone).
3. Join site weather: BDG2 weather timestamps are UTC; meters are local time.
   We convert weather to each site's local time before the join, then
   interpolate weather gaps of up to 3 hours.
4. Clean: drop meters with >20% missing hours, long zero runs (>=24h) and
   stuck values (>=48h) as meter faults, clip negative readings to missing.
5. Require >=80% coverage in BOTH 2016 and 2017 for a meter to be kept.
6. Optionally restrict to a subset of buildings for fast iteration.
7. Write data/processed/electricity.parquet (float32) and log kept vs dropped.

Run:  .venv/bin/python -m gridwise.etl [--full]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gridwise import config


def load_wide() -> pd.DataFrame:
    """Load the cleaned electricity file as (timestamp, one column per meter)."""
    df = pd.read_csv(config.BDG2_METERS_CLEANED / "electricity_cleaned.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp").sort_index()


def load_metadata() -> pd.DataFrame:
    m = pd.read_csv(config.BDG2_METADATA / "metadata.csv", low_memory=False)
    return m[
        [
            "building_id",
            "site_id",
            "primaryspaceusage",
            "sqm",
            "yearbuilt",
            "timezone",
        ]
    ]


def load_weather_local() -> dict[str, pd.DataFrame]:
    """Load weather, convert UTC timestamps to each site's local time.

    Returns {site_id: weather_df indexed by local timestamp}.
    """
    w = pd.read_csv(config.BDG2_WEATHER / "weather.csv")
    w["timestamp"] = pd.to_datetime(w["timestamp"], utc=True)
    cols = ["airTemperature", "dewTemperature", "windSpeed", "cloudCoverage"]
    out = {}
    for site, g in w.groupby("site_id"):
        g = g.set_index("timestamp")[cols].sort_index()
        g = g[~g.index.duplicated(keep="first")]
        # interpolate gaps up to 3h
        g = g.interpolate(method="time", limit=3, limit_area="inside")
        out[site] = g.tz_convert(g.index.tz) if g.index.tz is not None else g
    return out


def _localize(weather: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Shift UTC-indexed weather to naive local time for merging with meters."""
    local = weather.tz_convert(tz).tz_localize(None)
    return local[~local.index.duplicated(keep="first")]


def clean_meter_series(s: pd.Series) -> pd.Series:
    """Apply per-meter cleaning rules. Returns NaN where readings are faults."""
    s = s.astype("float32")
    s = s.where(s >= 0)                      # clip negatives to missing

    # zero runs >= 24h -> NaN
    is_zero = s.eq(0)
    zero_run = is_zero.groupby((~is_zero).cumsum()).transform("size") * is_zero
    s = s.mask(zero_run >= config.STUCK_ZERO_RUN_HOURS)

    # stuck values (same reading) >= 48h -> NaN
    is_stuck = s.eq(s.shift()) & s.notna()
    stuck_run = is_stuck.groupby((~is_stuck).cumsum()).transform("size") * is_stuck
    s = s.mask(stuck_run >= config.STUCK_VALUE_RUN_HOURS)
    return s


def run(full: bool = False) -> pd.DataFrame:
    wide = load_wide()
    meta = load_metadata()
    meters = [c for c in wide.columns if c in set(meta.building_id)]

    if not full:
        # Subset: sample buildings across sites and usage types.
        elec_meta = meta[meta.building_id.isin(meters)]
        rng = np.random.default_rng(config.RANDOM_SEED)
        # stratify by site to keep several sites in the subset
        n = min(config.SUBSET_N_BUILDINGS, len(elec_meta))
        idx = (
            elec_meta.groupby("site_id", group_keys=False)
            .apply(lambda g: g.sample(max(1, int(round(n * len(g) / len(elec_meta)))),
                                       random_state=int(rng.integers(1e6))))
        )
        subset_ids = set(idx.building_id)
        meters = [c for c in meters if c in subset_ids]

    print(f"[etl] meters in scope: {len(meters)} (full={full})")
    wide = wide[meters]

    # per-meter cleaning
    wide = wide.apply(clean_meter_series, axis=0)

    # melt to long
    long = wide.reset_index().melt(
        id_vars="timestamp", var_name="building_id", value_name="kwh"
    )
    long = long.dropna(subset=["kwh"])
    long["kwh"] = long["kwh"].astype("float32")

    # coverage filter per meter across both years
    hours_2016, hours_2017 = 8784, 8760  # 2016 is a leap year
    cov = long.assign(
        y2016=long.timestamp.dt.year.eq(2016), y2017=long.timestamp.dt.year.eq(2017)
    ).groupby("building_id")[["y2016", "y2017"]].sum()
    keep = cov[(cov.y2016 >= config.MIN_COVERAGE_BOTH_YEARS * hours_2016)
               & (cov.y2017 >= config.MIN_COVERAGE_BOTH_YEARS * hours_2017)].index
    n_before = long.building_id.nunique()
    long = long[long.building_id.isin(set(keep))]

    # join metadata
    long = long.merge(meta, on="building_id", how="left")

    # join weather per site, aligned to local time
    weather = load_weather_local()
    parts = []
    site_tz = dict(zip(meta.site_id, meta.timezone))
    for site, g in long.groupby("site_id"):
        tz = site_tz.get(site)
        w = weather.get(site)
        if w is None or not tz:
            continue
        wl = _localize(w, tz)
        g = g.merge(
            wl, left_on="timestamp", right_index=True, how="left"
        )
        parts.append(g)
    long = pd.concat(parts, ignore_index=True)

    print(
        f"[etl] kept {long.building_id.nunique()}/{n_before} meters "
        f"after cleaning + coverage; rows={len(long):,}"
    )
    print(
        f"[etl] weather missing after join: "
        f"{long.airTemperature.isna().mean():.1%}"
    )

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    long.to_parquet(config.PARQUET_PATH, index=False)
    print(f"[etl] wrote {config.PARQUET_PATH}")
    return long


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="use all buildings, not the subset")
    args = ap.parse_args()
    run(full=args.full)
