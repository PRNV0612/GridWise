"""Carbon-aware load shifting: move flexible load from dirty hours to clean
hours within a forecast horizon, subject to physical caps.

Problem statement:
- Forecast hourly load L[t] for one building over a horizon.
- A flexible share f (HVAC pre-cooling, EV charging, DHW) of each hour's
  load can be moved to other hours in the horizon.
- Each hour can absorb at most `headroom` fraction above its own baseline:
  cap[t] = L[t] * (1 + headroom).
- Moving 1 kWh from hour a to hour b saves (CI[a] - CI[b]) g CO2. Only
  move when the gain exceeds MIN_CARBON_GAIN_G (avoid pointless churn).

Algorithm: greedy pairing, not an LP. Sort source hours dirty-first and
destination hours clean-first; fill the cleanest destinations with the
dirtiest sources while the marginal gain clears the threshold. This is
optimal for the relaxation without ramp/startup costs, and it is
transparent enough to explain to a building operator.

Guarantees (property-tested):
- total energy conserved (moved, never created or destroyed)
- post-shift load never exceeds cap[t] and never goes negative
- CO2 saving is non-negative; zero when CI is flat
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gridwise import config


@dataclass
class ShiftResult:
    horizon: pd.DatetimeIndex
    baseline_kwh: np.ndarray
    shifted_kwh: np.ndarray
    ci_gco2: np.ndarray
    moved_kwh: np.ndarray          # net kWh moved INTO each hour (can be <0)
    baseline_co2_g: float
    shifted_co2_g: float
    saved_co2_g: float
    moved_total_kwh: float
    moves: list[dict] = field(default_factory=list)  # audit trail


def shift_load(
    load: pd.Series,
    ci: pd.Series,
    flexible_share: float = config.DEFAULT_FLEXIBLE_SHARE,
    headroom_share: float = config.DEFAULT_HEADROOM_SHARE,
    min_gain_g: float = config.MIN_CARBON_GAIN_G,
) -> ShiftResult:
    """Greedy carbon-aware shift over a single building's horizon.

    `load` and `ci` are hourly Series on the same index (load kWh, ci
    gCO2/kWh). Returns the shifted schedule and the CO2 delta.
    """
    if not load.index.equals(ci.index):
        raise ValueError("load and ci must share an hourly index")
    idx = load.index
    L = load.to_numpy(dtype="float64")
    C = ci.to_numpy(dtype="float64")
    n = len(L)

    movable = flexible_share * L
    capacity = headroom_share * L

    # greedy: dirty sources first, clean destinations first
    src_order = np.argsort(-C)
    dst_order = np.argsort(C)

    moved = np.zeros(n)  # net kWh moved into hour
    moves: list[dict] = []

    for s in src_order:
        avail = movable[s]
        if avail <= 0:
            continue
        for d in dst_order:
            if d == s:
                continue
            room = capacity[d] - moved[d]  # remaining headroom at destination
            # (moved[d] < 0 means d already shed flexible load, so it can
            # absorb more than its own baseline headroom)
            if room <= 0:
                continue
            gain = C[s] - C[d]
            if gain <= min_gain_g:
                # destinations sorted cleanest-first: no later destination
                # can clear the threshold either
                break
            qty = min(avail, room)
            if qty <= 0:
                continue
            moved[d] += qty
            moved[s] -= qty
            avail -= qty
            moves.append({
                "from": idx[s], "to": idx[d], "kwh": round(qty, 3),
                "gain_gco2_per_kwh": round(float(gain), 1),
                "co2_saved_g": round(qty * gain, 1),
            })
            if avail <= 0:
                break

    shifted = L + moved
    assert np.all(shifted >= 0), "shift produced negative load"
    assert np.all(shifted <= L * (1 + headroom_share) + 1e-9), "cap violated"
    baseline_co2 = float((L * C).sum())
    shifted_co2 = float((shifted * C).sum())

    return ShiftResult(
        horizon=idx,
        baseline_kwh=L,
        shifted_kwh=shifted,
        ci_gco2=C,
        moved_kwh=moved,
        baseline_co2_g=baseline_co2,
        shifted_co2_g=shifted_co2,
        saved_co2_g=baseline_co2 - shifted_co2,
        moved_total_kwh=float(np.abs(moved[moved > 0]).sum()),
        moves=moves,
    )
