"""Carbon intensity providers, behind one interface so the shifting logic
never knows where the gCO2/kWh numbers came from.

Providers:
- UKCarbonProvider   -- live UK Carbon Intensity API (default, real signal)
- DewaCarbonProvider -- UAE scenario: annual-mean CI with a midday solar dip
                        (Mohammed bin Rashid Solar Park shape), labelled
                        calibrated projection, not measured data
- CsvCarbonProvider  -- offline fallback (columns: timestamp, gco2_kwh)

All providers return a pandas Series indexed by hourly UTC timestamps,
values in gCO2/kWh.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from gridwise import config

CARBON_COLS = ["timestamp", "gco2_kwh"]


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Coerce a timestamp (naive or aware) to UTC."""
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class CarbonProvider:
    """Base class: subclasses implement `fetch(start, end)`."""

    name = "base"

    def series(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Hourly carbon intensity gCO2/kWh, UTC index, inclusive of end."""
        start = _utc(pd.Timestamp(start))
        end = _utc(pd.Timestamp(end))
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        out = self.fetch(start, end)
        s = out.reindex(idx)
        if s.isna().any():
            raise ValueError(
                f"[carbon:{self.name}] missing intensity for "
                f"{int(s.isna().sum())} hours; refusing to shift on unknown CI"
            )
        return s.astype("float64")

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        raise NotImplementedError


class UKCarbonProvider(CarbonProvider):
    """Live GB grid carbon intensity via carbonintensity.org.uk.

    API returns 30-min periods; we resample to hourly means.
    https://api.carbonintensity.org.uk/documentation
    """

    name = "uk"
    URL = "https://api.carbonintensity.org.uk/intensity/{from}/{to}"

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        fmt = "%Y-%m-%dT%H:%MZ"
        # the API's `to` bound is exclusive, so request one extra period
        url = self.URL.format(
            **{
                "from": start.strftime(fmt),
                "to": (end + pd.Timedelta(hours=1)).strftime(fmt),
            }
        )
        r = httpx.get(url, timeout=15)
        r.raise_for_status()
        periods = r.json()["data"]
        s = pd.Series(
            {pd.Timestamp(p["from"]).tz_convert("UTC"): p["intensity"]["actual"]
             for p in periods},
            dtype="float64",
        )
        s = s[~s.index.duplicated(keep="first")].sort_index()
        hourly = s.resample("h").mean()
        # most recent periods may have forecast-only (null actual) values
        if hourly.isna().any():
            forecast = pd.Series(
                {pd.Timestamp(p["from"]).tz_convert("UTC"): p["intensity"]["forecast"]
                 for p in periods},
                dtype="float64",
            ).sort_index().resample("h").mean()
            hourly = hourly.fillna(forecast)
        return hourly


class DewaCarbonProvider(CarbonProvider):
    """UAE scenario provider (calibrated projection, not measured).

    DEWA's published grid average sits near 0.35 kgCO2/kWh and falls as the
    Mohammed bin Rashid Solar Park scales. We model the *shape* of a
    solar-heavy grid: cleanest around solar noon, dirtiest late evening,
    with the annual mean anchored at DEWA_EMISSION_FACTOR_2026 * 1000.
    """

    name = "dewa"
    # fractional dip at solar noon vs the 24h mean (calibration assumption)
    NOON_DIP = 0.45
    # fractional rise at evening peak (gas-heavy hours)
    EVENING_RISE = 0.25

    def __init__(self, mean_gco2: float | None = None):
        self.mean_gco2 = mean_gco2 or config.DEWA_EMISSION_FACTOR_2026 * 1000

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        idx = pd.date_range(start, end, freq="h", tz="UTC")
        # Dubai local solar noon ~ 08:00-09:00 UTC
        local_hour = (idx + pd.Timedelta(hours=4)).hour.to_numpy()
        shape = (
            -self.NOON_DIP * np.cos((local_hour - 8) / 12 * np.pi)
            + self.EVENING_RISE * np.clip((local_hour - 18) / 6, 0, 1)
        )
        # fractions of the mean, centered, then applied to the mean level
        shape = shape - shape.mean()
        return pd.Series(self.mean_gco2 * (1 + shape), index=idx)


class CsvCarbonProvider(CarbonProvider):
    """Offline fallback: CSV with columns timestamp, gco2_kwh."""

    name = "csv"

    def __init__(self, path: Path):
        self.path = Path(path)

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        df = pd.read_csv(self.path)
        s = pd.Series(
            df.gco2_kwh.to_numpy(dtype="float64"),
            index=pd.to_datetime(df.timestamp, utc=True),
        )
        return s[~s.index.duplicated(keep="first")].sort_index()


def get_provider(name: str, **kwargs) -> CarbonProvider:
    providers = {
        "uk": UKCarbonProvider,
        "dewa": DewaCarbonProvider,
        "csv": CsvCarbonProvider,
    }
    return providers[name](**kwargs)
