"""Metrics: CV(RMSE), NMBE, RMSLE, MAE — ASHRAE Guideline 14 conventions.

All functions operate on kWh (not log space), so model predictions must be
expm1'd back before scoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cv_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of variation of RMSE: RMSE / mean(actual). ASHRAE G14 hourly
    calibration target is <= 30%."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return float(rmse / actual.mean()) * 100.0


def nmbe(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Net Mean Bias Error (%): sum(pred - actual) / sum(actual). G14 hourly
    target is within +/- 10%."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float((predicted - actual).sum() / actual.sum()) * 100.0


def rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared log error — matches the ASHRAE Kaggle metric, so
    results are comparable with published numbers."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(
        np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2))
    )


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(
        np.mean(np.abs(np.asarray(actual, float) - np.asarray(predicted, float)))
    )


def all_metrics(actual, predicted) -> dict:
    return {
        "cv_rmse_pct": round(cv_rmse(actual, predicted), 2),
        "nmbe_pct": round(nmbe(actual, predicted), 2),
        "rmsle": round(rmsle(actual, predicted), 4),
        "mae_kwh": round(mae(actual, predicted), 3),
    }


def by_group(actual: pd.Series, predicted: np.ndarray, groups: pd.Series) -> dict:
    """Metrics per building usage type, for the README results table."""
    out = {}
    for name, idx in groups.groupby(groups).groups.items():
        sub = groups.index.get_indexer(idx)
        out[str(name)] = all_metrics(
            np.asarray(actual)[sub], np.asarray(predicted)[sub]
        )
    return out
