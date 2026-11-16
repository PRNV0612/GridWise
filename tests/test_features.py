"""Tests for feature engineering and metrics."""
import numpy as np
import pandas as pd
import pytest

from gridwise.evaluate import all_metrics, cv_rmse, nmbe
from gridwise.features import build_features


def _toy_frame(n_hours: int = 72):
    ts = pd.date_range("2016-01-01", periods=n_hours, freq="h")
    df = pd.DataFrame({
        "timestamp": ts,
        "building_id": ["B1"] * n_hours,
        "kwh": np.linspace(10.0, 20.0, n_hours),
        "primaryspaceusage": ["Office"] * n_hours,
        "sqm": [500.0] * n_hours,
        "yearbuilt": [2000.0] * n_hours,
        "site_id": ["Panther"] * n_hours,
        "airTemperature": [15.0] * n_hours,
        "dewTemperature": [10.0] * n_hours,
        "windSpeed": [3.0] * n_hours,
        "cloudCoverage": [50.0] * n_hours,
    })
    return df


def test_build_features_columns_and_target():
    out = build_features(_toy_frame())
    assert set(["hour", "dayofweek", "is_weekend", "cdh", "hdh"]).issubset(out.columns)
    # log1p target
    assert out["log1p_kwh"].iloc[0] == pytest.approx(np.log1p(10.0))
    # constant 15C => no cooling, no heating
    assert out["cdh"].max() == 0.0
    assert out["hdh"].max() == 0.5


def test_lag_features_align_with_naive():
    df = _toy_frame(72)
    df.loc[:, "naive_pred"] = np.nan
    out = build_features(df)
    # lag_24h of row i equals kwh of row i-24
    assert out["lag_24h"].iloc[25] == pytest.approx(df["kwh"].iloc[1])
    # first 24 rows have no lag
    assert out["lag_24h"].iloc[:24].isna().all()


def test_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert cv_rmse(y, y) == 0.0
    assert nmbe(y, y) == 0.0
    m = all_metrics(y, y)
    assert m["rmsle"] == 0.0


def test_cv_rmse_scale_free():
    # doubling the scale doubles RMSE but also the mean -> same CV(RMSE)
    y = np.array([10.0, 20.0, 30.0, 40.0])
    p = np.array([11.0, 21.0, 31.0, 41.0])
    assert cv_rmse(y, p) == pytest.approx(cv_rmse(2 * y, 2 * p))


def test_nmbe_sign():
    # over-prediction -> positive bias
    y = np.array([10.0, 10.0])
    assert nmbe(y, y + 1.0) == pytest.approx(10.0)
