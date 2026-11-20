"""UAE scenario overlay: replay a building's forecast under ERA5 Dubai weather.

This is a SCENARIO, not a measurement — labelled everywhere it appears:
- Load shape: the weather-only model (no lags — this is an exogenous replay,
  so lagged US loads are meaningless under Dubai weather) predicts what the
  building would use under Dubai's cooling-dominated climate.
- Scale: anchored to KAPSARC's DEWA annual consumption series (2013-17,
  ~4.8%/yr growth) extrapolated to 2026 as a labelled projection.
- Carbon: DewaCarbonProvider (solar-heavy daily shape, DEWA mean factor).

Data: ERA5 hourly reanalysis for Dubai via the Open-Meteo archive API
(no key required). Cached to data/uae/era5_dubai_hourly.csv.

Run:  .venv/bin/python -m gridwise.uae_scenario
Writes docs/plots/uae_scenario.png and prints the scenario summary.
"""
from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridwise import config
from gridwise.carbon import DewaCarbonProvider
from gridwise.features import FEATURE_COLS, build_features
from gridwise.shifting import shift_load
from gridwise.train import load_frame

ERA5_CACHE = config.DATA_UAE / "era5_dubai_hourly.csv"
ERA5_URL = (
    "https://archive-api.open-meteo.com/v1/era5"
    "?latitude=25.2&longitude=55.3"
    "&start_date={start}&end_date={end}"
    "&hourly=temperature_2m,cloud_cover,wind_speed_10m,relative_humidity_2m"
)
# full year in Dubai local time: ERA5 is UTC, Dubai is UTC+4, so pad both ends
ERA5_RANGE = ("2016-12-30", "2018-01-02")
SCENARIO_BUILDING = "Hog_office_Joey"  # large office, the demo building
KAPSARC_GROWTH_PER_YEAR = 0.048  # DEWA 2013-17 CAGR from the anchor CSV


def fetch_era5_dubai(year: int = 2017) -> pd.DataFrame:
    """ERA5 hourly weather for Dubai, cached. Columns match feature names."""
    if ERA5_CACHE.exists():
        df = pd.read_csv(ERA5_CACHE, parse_dates=["timestamp"])
        if df.timestamp.dt.year.eq(year).all():
            return df
    import httpx

    r = httpx.get(ERA5_URL.format(start=ERA5_RANGE[0], end=ERA5_RANGE[1]),
                  timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(h["time"]),
        "airTemperature": h["temperature_2m"],
        "cloudCoverage": h["cloud_cover"],
        "windSpeed": h["wind_speed_10m"],
        "relative_humidity_2m": h["relative_humidity_2m"],
    })
    # dew point from RH via the Magnus approximation
    t = df["airTemperature"]
    rh = df["relative_humidity_2m"].clip(1, 100)
    a, b = 17.625, 243.04
    gamma = np.log(rh / 100) + (a * t) / (b + t)
    df["dewTemperature"] = (b * gamma) / (a - gamma)

    df = df[["timestamp", "airTemperature", "dewTemperature",
             "windSpeed", "cloudCoverage"]]
    df = df[df.timestamp.dt.year.eq(year)]  # keep the scenario year only
    df.to_csv(ERA5_CACHE, index=False)
    print(f"[uae] cached ERA5 Dubai -> {ERA5_CACHE}")
    return df


def dewa_growth_factor(from_year: int, to_year: int) -> float:
    """DEWA consumption growth projection, KAPSARC 2013-17 trend extrapolated.

    Labelled projection, not measurement."""
    return (1 + KAPSARC_GROWTH_PER_YEAR) ** (to_year - from_year)


def run() -> dict:
    df = load_frame()
    b = df[df.building_id.eq(SCENARIO_BUILDING)].copy()
    b = b[b.timestamp.dt.year.eq(2017)]  # scenario replays the test year
    if b.empty:
        raise SystemExit(f"building {SCENARIO_BUILDING} not in parquet")

    era5 = fetch_era5_dubai(2017)
    # join on day+hour (meter timestamps are naive local; ERA5 is UTC — the
    # scenario deliberately uses Dubai local clock via UTC+4 shift)
    era5["local"] = era5["timestamp"] + pd.Timedelta(hours=4)
    era5["dh"] = era5["local"].dt.strftime("%m-%d %H")
    b["dh"] = b["timestamp"].dt.strftime("%m-%d %H")
    weather = era5.set_index("dh")

    scenario = b.copy()
    for col in ("airTemperature", "dewTemperature", "windSpeed", "cloudCoverage"):
        scenario[col] = scenario["dh"].map(weather[col]).astype("float32")
    missing = scenario["airTemperature"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} rows without ERA5 match")

    # weather-only model: exogenous replay, lags excluded by design
    feats = build_features(scenario)
    for c in ("primaryspaceusage", "building_id", "site_id"):
        feats[c] = feats[c].cat.add_categories([]).fillna("Unknown")
    model = joblib.load(config.MODELS_DIR / "weather_model.joblib")
    pred = np.expm1(model.predict(feats[FEATURE_COLS]))
    scenario["scenario_kwh"] = pred

    growth = dewa_growth_factor(2017, 2026)
    scenario["scenario_2026_kwh"] = scenario["scenario_kwh"] * growth

    # --- summary + plot -------------------------------------------------
    home = b.set_index("timestamp")["kwh"]
    scen = scenario.set_index("timestamp")["scenario_kwh"]
    monthly = pd.DataFrame({
        "home actual": home.resample("ME").mean(),
        "dubai 2017 scenario": scen.resample("ME").mean(),
        "dubai 2026 scenario": scenario.set_index("timestamp")[
            "scenario_2026_kwh"].resample("ME").mean(),
    })
    summer = monthly.loc["2017-07-31"]
    winter = monthly.loc["2017-01-31"]

    fig, ax = plt.subplots(figsize=(9, 5))
    monthly.plot(ax=ax, color={"home actual": "#94a3b8",
                               "dubai 2017 scenario": "#f59e0b",
                               "dubai 2026 scenario": "#ef4444"})
    ax.set_ylabel("mean hourly kWh")
    ax.set_title(
        f"{SCENARIO_BUILDING}: home climate vs Dubai scenario (labelled projection)"
    )
    fig.tight_layout()
    config.ROOT.joinpath("docs/plots").mkdir(exist_ok=True)
    path = config.ROOT / "docs" / "plots" / "uae_scenario.png"
    fig.savefig(path, dpi=150)
    print(f"[uae] wrote {path}")

    # --- shift on a representative July day, DEWA carbon curve -----------
    july = scenario[scenario.timestamp.dt.strftime("%Y-%m-%d").eq("2017-07-15")]
    load_day = july.set_index("timestamp")["scenario_2026_kwh"]
    ci = DewaCarbonProvider().series(
        load_day.index[0].tz_localize("UTC"),
        load_day.index[-1].tz_localize("UTC"),
    ).tz_convert(None)
    ci.index = load_day.index
    r = shift_load(load_day, ci)

    summary = {
        "building": SCENARIO_BUILDING,
        "annual_kwh_home_actual": round(float(home.sum())),
        "annual_kwh_dubai_2017_scenario": round(float(scen.sum())),
        "summer_winter_ratio_dubai": round(
            float(summer["dubai 2017 scenario"] /
                  winter["dubai 2017 scenario"]), 2),
        "summer_winter_ratio_home": round(
            float(summer["home actual"] / winter["home actual"]), 2),
        "dewa_growth_factor_2017_2026": round(growth, 2),
        "shift_day": "2017-07-15",
        "shift_kwh_moved": round(r.moved_total_kwh, 1),
        "shift_co2_saved_kg": round(r.saved_co2_g / 1e3, 1),
        "shift_co2_saved_pct": round(100 * r.saved_co2_g / r.baseline_co2_g, 1),
    }
    print(pd.Series(summary).to_string())
    print("[uae] NOTE: all Dubai numbers are labelled scenario projections,")
    print("       not measurements (README caveat 2).")
    return summary


if __name__ == "__main__":
    run()
