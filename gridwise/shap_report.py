"""SHAP feature importance for the day-ahead model -> docs/plots/shap_importance.png.

Computed on a sample of 2017 test rows (the year the README numbers are
reported on), using TreeExplainer on the LightGBM model directly — exact
Shapley values for tree ensembles, no sampling approximation.

Run:  .venv/bin/python -m gridwise.shap_report
"""
from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from gridwise import config
from gridwise.features import DAY_AHEAD_FEATURE_COLS, build_features
from gridwise.train import load_frame, split


def run(sample_n: int = 20000, seed: int = config.RANDOM_SEED) -> pd.Series:
    df = load_frame()
    feats = build_features(df)
    for c in ("primaryspaceusage", "building_id", "site_id"):
        feats[c] = feats[c].cat.add_categories([]).fillna("Unknown")
    X = feats[DAY_AHEAD_FEATURE_COLS].copy()

    # score on the same test rows as the headline metrics: 2017 with lags
    _, _, test_m = split(df)
    lag_ok = feats[DAY_AHEAD_FEATURE_COLS[-3:]].notna().all(axis=1).values
    X_test = X.loc[test_m & lag_ok]
    rng = np.random.default_rng(seed)
    X_sample = X_test.iloc[rng.choice(len(X_test), size=min(sample_n, len(X_test)),
                                replace=False)]

    model = joblib.load(config.MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    imp = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X_sample.columns
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    top = imp.iloc[::-1]
    colors = []
    for name in top.index:
        if name in ("lag_24h", "lag_168h", "load_rolling_24h"):
            colors.append("#60a5fa")          # lag features
        elif name in ("airTemperature", "dewTemperature", "temp_rolling_24h",
                      "cdh", "hdh", "windSpeed", "cloudCoverage"):
            colors.append("#f59e0b")          # weather
        else:
            colors.append("#34d399")          # calendar/building
    ax.barh(top.index, top.values, color=colors)
    ax.set_xlabel("mean |SHAP value|  (log1p kWh units)")
    ax.set_title("Day-ahead model feature importance (SHAP, 2017 test sample)")
    fig.tight_layout()

    config.ROOT.joinpath("docs/plots").mkdir(exist_ok=True)
    path = config.ROOT / "docs" / "plots" / "shap_importance.png"
    fig.savefig(path, dpi=150)
    print(f"[shap] wrote {path}")
    print(imp.round(4).to_string())
    return imp


if __name__ == "__main__":
    run()
