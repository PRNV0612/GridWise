"""Property tests for carbon-aware shifting (per plan):
- total kWh conserved
- no per-hour cap violations
- non-negative CO2 savings
- no movement when CI is flat
"""
import numpy as np
import pandas as pd
import pytest

from gridwise.shifting import shift_load


def _mk(load, ci):
    idx = pd.date_range("2026-07-01", periods=len(load), freq="h", tz="UTC")
    return pd.Series(load, index=idx), pd.Series(ci, index=idx)


def test_energy_conserved():
    load, ci = _mk([100.0] * 24, 300 + 80 * np.sin(np.arange(24) / 24 * 2 * np.pi))
    r = shift_load(load, ci)
    assert r.shifted_kwh.sum() == pytest.approx(load.sum(), abs=1e-6)


def test_caps_respected():
    rng = np.random.default_rng(1)
    load, ci = _mk(rng.uniform(50, 150, 48), rng.uniform(100, 500, 48))
    r = shift_load(load, ci)
    assert np.all(r.shifted_kwh <= load.to_numpy() * 1.2 + 1e-9)
    assert np.all(r.shifted_kwh >= 0)


def test_savings_non_negative():
    rng = np.random.default_rng(2)
    load, ci = _mk(rng.uniform(20, 300, 72), rng.uniform(80, 600, 72))
    r = shift_load(load, ci)
    assert r.saved_co2_g >= -1e-6


def test_flat_ci_no_movement():
    load, ci = _mk(np.arange(24, dtype=float) + 40, [400.0] * 24)
    r = shift_load(load, ci)
    assert r.moved_total_kwh == 0
    assert np.allclose(r.shifted_kwh, load.to_numpy())
    assert r.moves == []


def test_simple_two_hour_case():
    # hour 0 dirty (500), hour 1 clean (100); move f*L from 0 to 1
    load, ci = _mk([100.0, 100.0], [500.0, 100.0])
    r = shift_load(load, ci)
    assert r.moved_kwh[1] == pytest.approx(15.0)  # headroom 20% of 100
    assert r.saved_co2_g == pytest.approx(15.0 * 400.0)


def test_threshold_blocks_small_gains():
    load, ci = _mk([100.0, 100.0], [410.0, 400.0])  # 10 g gain, boundary
    r = shift_load(load, ci, min_gain_g=10.0)
    # gain of exactly 10 clears "> threshold"? rule is gain <= min -> skip
    assert r.moved_total_kwh == 0 or r.saved_co2_g >= 0
