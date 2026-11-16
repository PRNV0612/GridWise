"""Train the LightGBM load-forecast model and evaluate it on the 2017 test year.

Split (time-based, no shuffling):
- train: Jan-Sep 2016
- validation: Oct-Dec 2016 (early stopping)
- test: 2017 (never seen)

Baseline: seasonal naive — prediction for hour t = actual load at t - 168h.
Every model number is reported against it.

Run:  .venv/bin/python -m gridwise.train
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from gridwise import config
from gridwise.evaluate import all_metrics, by_group
from gridwise.features import (
    DAY_AHEAD_FEATURE_COLS,
    FEATURE_COLS,
    TARGET_COL,
    build_features,
)


def load_frame() -> pd.DataFrame:
    df = pd.read_parquet(config.PARQUET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def seasonal_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Predict kwh(t) = kwh(t - 168h) per building; attach as 'naive_pred'.

    Rows without a lag-168h observation get NaN (dropped from baseline
    scoring) — same rule applies to the model: both are scored on the same
    test rows where the baseline exists, so the comparison is fair.
    """
    lag = (
        df.set_index("timestamp")
        .groupby("building_id")["kwh"]
        .transform(lambda s: s.shift(freq=pd.Timedelta(hours=168)))
    )
    df = df.assign(naive_pred=lag.to_numpy())
    return df


def split(df: pd.DataFrame):
    ts = df["timestamp"]
    train = ts.dt.year.eq(config.TRAIN_YEAR) & ~ts.dt.month.isin(
        config.VALIDATION_MONTHS
    )
    valid = (
        ts.dt.year.eq(config.TRAIN_YEAR)
        & ts.dt.month.isin(config.VALIDATION_MONTHS)
    )
    test = ts.dt.year.eq(config.TEST_YEAR)
    return train.values, valid.values, test.values


LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    num_leaves=255,
    learning_rate=0.05,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    n_estimators=2000,
    verbose=-1,
    seed=config.RANDOM_SEED,
)


def run() -> dict:
    df = load_frame()
    df = seasonal_naive(df)
    feats = build_features(df)
    for c in ("primaryspaceusage", "building_id", "site_id"):
        feats[c] = feats[c].cat.add_categories([]).fillna("Unknown")

    train_m, valid_m, test_m = split(df)
    y = feats[TARGET_COL]
    y_ok = y.notna().values

    X = feats.copy()
    for c in X.select_dtypes("category"):
        X[c] = X[c].astype("category")

    def fit(cols: list[str]):
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X.loc[train_m & y_ok, cols],
            y[train_m & y_ok],
            eval_set=[(X.loc[valid_m & y_ok, cols], y[valid_m & y_ok])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100, verbose=False)],
            categorical_feature="auto",
        )
        return model

    # Model A: weather-only (measurement & verification / anomaly layer)
    weather_model = fit(FEATURE_COLS)
    # Model B: day-ahead (weather + lags) — the operational forecast
    lag_ok = feats[DAY_AHEAD_FEATURE_COLS[-3:]].notna().all(axis=1).values
    day_ahead = fit(DAY_AHEAD_FEATURE_COLS)

    # Score both on the same test rows the baseline can score (lag-168h
    # exists), so the comparison is apples-to-apples.
    usable = df["naive_pred"].notna().values & y_ok
    test_score = test_m & usable

    def score(model, cols):
        pred_kwh = np.expm1(model.predict(X.loc[test_score, cols]))
        return pred_kwh

    actual = df.loc[test_score, "kwh"].to_numpy()
    naive = df.loc[test_score, "naive_pred"].to_numpy()
    weather_pred = score(weather_model, FEATURE_COLS)
    day_ahead_pred = score(day_ahead, DAY_AHEAD_FEATURE_COLS)

    baseline_metrics = all_metrics(actual, naive)
    weather_metrics = all_metrics(actual, weather_pred)
    day_ahead_metrics = all_metrics(actual, day_ahead_pred)
    improvement = round(
        100
        * (baseline_metrics["cv_rmse_pct"] - day_ahead_metrics["cv_rmse_pct"])
        / baseline_metrics["cv_rmse_pct"],
        1,
    )

    results = {
        "n_test_rows": int(test_score.sum()),
        "n_buildings": int(df.loc[test_score, "building_id"].nunique()),
        "model": day_ahead_metrics,               # headline model: day-ahead
        "model_weather_only": weather_metrics,
        "baseline_seasonal_naive": baseline_metrics,
        "cv_rmse_improvement_vs_baseline_pct": improvement,
        "by_usage": by_group(
            df.loc[test_score, "kwh"],
            day_ahead_pred,
            df.loc[test_score, "primaryspaceusage"],
        ),
        "best_iteration_day_ahead": int(day_ahead.best_iteration_ or 0),
        "best_iteration_weather": int(weather_model.best_iteration_ or 0),
        "lgb_params": LGB_PARAMS,
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(day_ahead, config.MODEL_PATH)
    joblib.dump(weather_model, config.MODELS_DIR / "weather_model.joblib")
    with open(config.METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "by_usage"}, indent=2))
    print(f"[train] model -> {config.MODEL_PATH}, metrics -> {config.METRICS_PATH}")
    return results


if __name__ == "__main__":
    run()
