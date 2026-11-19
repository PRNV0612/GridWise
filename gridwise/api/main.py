"""FastAPI service exposing the three GridWise features:

- GET  /health                              status + model version
- GET  /buildings                           catalogue (id, use, sqm, site)
- GET  /buildings/{id}/forecast?start&end   hourly forecast vs actual
- GET  /buildings/{id}/anomalies?start&end  anomaly flags/events
- POST /buildings/{id}/shift                carbon-aware shift plan
- GET  /carbon?start&hours&provider         the carbon curve used
- GET  /metrics                             contents of models/metrics.json

Run:  .venv/bin/uvicorn gridwise.api.main:app --reload
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from gridwise import config
from gridwise.carbon import get_provider
from gridwise.features import (
    DAY_AHEAD_FEATURE_COLS,
    FEATURE_COLS,
    build_features,
)
from gridwise.shifting import shift_load

app = FastAPI(title="GridWise API", version="0.1.0")

_STATE: dict = {}


@app.on_event("startup")
def _load() -> None:
    """Load parquet, models and anomaly flags once, not per request."""
    df = pd.read_parquet(config.PARQUET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    anomalies = pd.read_parquet(config.MODELS_DIR / "anomalies.parquet")
    anomalies["timestamp"] = pd.to_datetime(anomalies["timestamp"])

    _STATE["df"] = df
    _STATE["anomalies"] = anomalies
    _STATE["day_ahead"] = joblib.load(config.MODEL_PATH)
    _STATE["weather"] = joblib.load(config.MODELS_DIR / "weather_model.joblib")
    with open(config.METRICS_PATH) as f:
        _STATE["metrics"] = json.load(f)

    # Pre-build features + predictions per building (one pass at startup).
    feats = build_features(df)
    for c in ("primaryspaceusage", "building_id", "site_id"):
        feats[c] = feats[c].cat.add_categories([]).fillna("Unknown")
    day_pred = pd.Series(
        _STATE["day_ahead"].predict(feats[DAY_AHEAD_FEATURE_COLS]),
        index=feats.index,
    )
    _STATE["forecast"] = pd.Series(
        day_pred.to_numpy(), index=df.index
    )
    _STATE["forecast"].name = "forecast_kwh"


def _building(id: str) -> pd.DataFrame:
    b = _STATE["df"][_STATE["df"].building_id.eq(id)]
    if b.empty:
        raise HTTPException(404, f"unknown building_id '{id}'")
    return b


def _parse_range(b: pd.DataFrame, start: str | None, end: str | None):
    ts = b["timestamp"]
    if start is not None:
        ts = ts[ts >= pd.Timestamp(start)]
    if end is not None:
        ts = ts[ts <= pd.Timestamp(end)]
    return ts


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": app.version,
        "buildings": int(_STATE["df"].building_id.nunique()),
    }


@app.get("/buildings")
def buildings():
    cat = (
        _STATE["df"][["building_id", "primaryspaceusage", "sqm", "site_id"]]
        .drop_duplicates("building_id")
        .sort_values("building_id")
    )
    return cat.to_dict(orient="records")


@app.get("/buildings/{id}/forecast")
def forecast(id: str, start: str | None = None, end: str | None = None,
             limit: int = Query(500, le=5000)):
    b = _building(id)
    ts = _parse_range(b, start, end)
    rows = b[b.timestamp.isin(ts)].join(
        _STATE["forecast"], how="left"
    )
    return {
        "building_id": id,
        "n_hours": int(len(rows)),
        "hours": rows[["timestamp", "kwh", "forecast_kwh"]]
        .head(limit)
        .assign(
            timestamp=lambda d: d.timestamp.dt.strftime("%Y-%m-%dT%H:%M"),
            kwh=lambda d: d.kwh.round(2),
            forecast_kwh=lambda d: d.forecast_kwh.round(2),
        )
        .to_dict(orient="records"),
    }


@app.get("/buildings/{id}/anomalies")
def anomalies(id: str, start: str | None = None, end: str | None = None):
    a = _STATE["anomalies"][_STATE["anomalies"].building_id.eq(id)]
    if a.empty:
        raise HTTPException(404, f"unknown building_id '{id}'")
    ts = a["timestamp"]
    if start is not None:
        ts = ts[ts >= pd.Timestamp(start)]
    if end is not None:
        ts = ts[ts <= pd.Timestamp(end)]
    a = a[a.timestamp.isin(ts)]
    flagged = a[a[["anom_residual", "anom_night_baseload", "anom_stuck_meter"]]
               .any(axis=1)]

    # merge consecutive flagged hours into events
    events = []
    cur = None
    for r in flagged.itertuples():
        if (cur is not None and r.timestamp - cur["end"] <= pd.Timedelta(hours=1)
                and r.anom_residual == cur["residual"]):
            cur["end"] = r.timestamp
            cur["hours"] += 1
            cur["peak_z"] = max(cur["peak_z"], float(r.z) if r.z == r.z else 0)
            cur["excess_kwh"] += max(float(r.resid_log), 0)
        else:
            if cur is not None:
                events.append(cur)
            cur = {
                "type": "residual" if r.anom_residual else (
                    "night_baseload" if r.anom_night_baseload else "stuck_meter"),
                "residual": bool(r.anom_residual),
                "start": r.timestamp,
                "end": r.timestamp,
                "hours": 1,
                "peak_z": float(r.z) if r.z == r.z else 0,
                "excess_kwh": max(float(r.resid_log), 0),
            }
    if cur is not None:
        events.append(cur)

    for e in events:
        e["start"] = e["start"].strftime("%Y-%m-%dT%H:%M")
        e["end"] = e["end"].strftime("%Y-%m-%dT%H:%M")
        e["peak_z"] = round(e["peak_z"], 2)
        e["excess_kwh"] = round(e["excess_kwh"], 2)
    return {"building_id": id, "n_flagged_hours": int(len(flagged)),
            "events": events}


@app.get("/carbon")
def carbon(start: str, hours: int = Query(24, ge=1, le=48),
           provider: str = "uk"):
    if provider not in ("uk", "dewa", "csv"):
        raise HTTPException(400, "provider must be uk|dewa|csv")
    try:
        prov = get_provider(provider)
        s = prov.series(pd.Timestamp(start), pd.Timestamp(start)
                        + pd.Timedelta(hours=hours - 1))
    except Exception as e:  # live API failure: report, don't crash
        raise HTTPException(502, f"carbon provider '{provider}' failed: {e}")
    return {
        "provider": provider,
        "hours": [
            {"timestamp": t.strftime("%Y-%m-%dT%H:%M"), "gco2_kwh": round(v, 1)}
            for t, v in s.items()
        ],
    }


@app.post("/buildings/{id}/shift")
def shift(id: str, start: str, horizon_h: int = 24,
          flexible_share: float = 0.15, headroom: float = 0.20,
          provider: str = "dewa"):
    if not 0 < flexible_share < 1 or not 0 < headroom <= 1:
        raise HTTPException(400, "flexible_share and headroom must be in (0,1]")
    if horizon_h not in (24, 48):
        raise HTTPException(400, "horizon_h must be 24 or 48")
    b = _building(id)
    start_ts = pd.Timestamp(start)
    win = b[b.timestamp.between(
        start_ts, start_ts + pd.Timedelta(hours=horizon_h - 1)
    )].set_index("timestamp")

    # forecast where available, actual as fallback (scenario replay)
    load = win["kwh"].fillna(
        _STATE["forecast"].reindex(win.index)
    ).astype("float64")
    if load.isna().any():
        raise HTTPException(
            409, f"no load data for {int(load.isna().sum())} of "
                 f"{len(load)} hours in the requested horizon"
        )

    try:
        ci = get_provider(provider).series(
            start_ts.tz_localize("UTC"), (start_ts + pd.Timedelta(
                hours=horizon_h - 1)).tz_localize("UTC")
        ).tz_convert(None)
    except Exception as e:
        raise HTTPException(502, f"carbon provider '{provider}' failed: {e}")
    ci.index = load.index  # meter-local clock for the demo

    r = shift_load(load, ci, flexible_share, headroom)
    return {
        "building_id": id,
        "provider": provider,
        "horizon": [str(t) for t in (r.horizon[0], r.horizon[-1])],
        "baseline_kwh": [round(x, 2) for x in r.baseline_kwh],
        "shifted_kwh": [round(x, 2) for x in r.shifted_kwh],
        "ci_gco2_kwh": [round(x, 1) for x in r.ci_gco2],
        "kwh_moved": round(r.moved_total_kwh, 1),
        "co2_saved_kg": round(r.saved_co2_g / 1e3, 2),
        "co2_saved_pct": round(
            100 * r.saved_co2_g / r.baseline_co2_g, 1
        ) if r.baseline_co2_g else 0,
        "moves": r.moves,
    }


@app.get("/metrics")
def metrics():
    return _STATE["metrics"]
