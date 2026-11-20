"""API smoke tests: every endpoint answers correctly against the real
parquet/models (uk provider replaced by dewa to avoid network in CI)."""
import pytest
from fastapi.testclient import TestClient

from gridwise.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers startup loading
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_buildings(client):
    r = client.get("/buildings")
    ids = [b["building_id"] for b in r.json()]
    assert r.status_code == 200 and len(ids) > 100


def test_forecast_roundtrip(client):
    b = client.get("/buildings").json()[0]["building_id"]
    r = client.get(f"/buildings/{b}/forecast",
                   params={"start": "2017-06-01", "end": "2017-06-02"})
    j = r.json()
    assert r.status_code == 200
    assert j["n_hours"] > 20
    assert all("forecast_kwh" in h for h in j["hours"])
    # regression: forecasts must be in kWh, not log1p(kwh)
    big = [h for h in j["hours"] if h["kwh"] and h["kwh"] > 100]
    if big:
        assert any(h["forecast_kwh"] > 1.0 for h in big), (
            "forecast looks like log-space values — expm1 conversion missing"
        )


def test_forecast_unknown_building_404(client):
    assert client.get("/buildings/nope/forecast").status_code == 404


def test_anomalies_events(client):
    b = client.get("/buildings").json()[0]["building_id"]
    r = client.get(f"/buildings/{b}/anomalies",
                   params={"start": "2017-01-01", "end": "2017-12-31"})
    assert r.status_code == 200
    for e in r.json()["events"]:
        assert e["end"] >= e["start"] and e["hours"] >= 1


def test_shift_conserves_energy(client):
    b = "Hog_office_Joey"
    r = client.post(f"/buildings/{b}/shift",
                    params={"start": "2017-12-31", "horizon_h": 24,
                            "provider": "dewa"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert sum(j["baseline_kwh"]) == pytest.approx(sum(j["shifted_kwh"]), abs=0.5)
    assert j["co2_saved_kg"] >= 0


def test_shift_bad_params(client):
    r = client.post("/buildings/Hog_office_Joey/shift",
                    params={"start": "2017-12-31", "flexible_share": 1.5})
    assert r.status_code == 400


def test_metrics(client):
    r = client.get("/metrics")
    assert "cv_rmse_improvement_vs_baseline_pct" in r.json()


def test_carbon_dewa(client):
    r = client.get("/carbon", params={"start": "2017-06-01", "hours": 24,
                                      "provider": "dewa"})
    assert r.status_code == 200
    hours = r.json()["hours"]
    assert len(hours) == 24 and all(h["gco2_kwh"] > 0 for h in hours)
