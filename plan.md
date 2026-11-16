# GridWise: AI Building Energy Optimiser (build plan)

**Goal:** Ship a working, public GitHub project by **Sat 28 Nov 2026**, three days before the Schneider Electric Energy Tech Accelerator deadline (1 Dec 2026). The project should give you one strong CV bullet with a real number in it.

**One-line pitch:** GridWise predicts how much electricity a building *should* use, flags when it uses more, and tells the operator when to run flexible loads so they draw the cleanest grid power. It reports the kWh and CO₂ saved.

**Why it fits the role:** The project touches all four themes in the posting: AI (a forecasting model), digitalization (a software-defined service with an API and dashboard), sustainability (carbon-aware load shifting) and buildings and energy, Schneider's core business. It also matches your IoT and SCADA background, since it treats smart-meter data the same way you treat traffic sensor data.

**2026-relevance convention (locked):** the load model stays on BDG2 — it is the field-standard benchmark, and the meter data is 2016–17. Current-year relevance is carried by (a) the **live** carbon signal, (b) the **DEWA grid emission factor** (~0.4 kgCO₂/kWh, falling with the Mohammed bin Rashid Solar Park rollout) and (c) a clearly-labelled **UAE scenario overlay**. Never claim the meter data is 2026. Every UAE-derived number is a calibrated projection, labelled as such.

---

## 1. Scope

### Must have (this is the CV line)
1. **Forecast model:** hourly electricity forecast per building from weather, calendar and building metadata, tested on a held-out year and compared against a naive baseline.
2. **Anomaly detection:** flags hours or days where actual use is far above the forecast.
3. **Carbon-aware load shifting:** takes a 24–48 h load forecast and a carbon-intensity forecast, moves a configurable flexible share of load to cleaner hours, and reports the kWh moved and CO₂ avoided.
4. **API:** FastAPI service that exposes all three features.
5. **Dashboard:** Next.js page with a building picker, forecast vs actual chart, anomaly markers, a shift recommendation and a savings card.
6. **README:** problem, architecture diagram, results table, screenshots, how to run it.

### Nice to have (only if ahead of schedule)
- Nice to have (promoted for the UAE story): **UAE scenario overlay** — rerun a building's forecast under ERA5 Dubai weather (cooling-dominated profile), scale with the KAPSARC/DEWA anchor, and run the shift optimiser against `DewaCarbonProvider` (solar-heavy daytime-clean curve). Labelled "scenario" everywhere it appears.
- Day-ahead model with lag features (load 24 h and 168 h ago) next to the weather-only model.
- SHAP feature-importance chart in the README.
- Docker Compose for one-command start, and a deployed demo (Render or Fly for the API, Vercel for the dashboard).
- LLM "energy report" that turns the week's anomalies and savings into a short plain-English summary for a facility manager. This adds a GenAI angle, but it is optional.

### Out of scope
- Real hardware or BMS integration, user accounts, multi-tenant setup and tariff optimisation.

---

## 2. Data

### Building energy: Building Data Genome Project 2 (BDG2)
- Repo: <https://github.com/buds-lab/building-data-genome-project-2>
- 3,053 meters across 1,636 buildings at 19 sites in North America and Europe, recorded hourly for **2016 and 2017**. This is the superset of the ASHRAE Great Energy Predictor III (Kaggle) data.
- Folders: `data/meters/` (`raw/`, `cleaned/`, `kaggle/`), `data/metadata/` and `data/weather/`.
- **Use:** the electricity meters from `cleaned/`, plus building metadata (primary use, floor area, year built, site, timezone) and site weather.
- **Day 1 check:** confirm how the large files download (the repo has a `.gitattributes`, so it probably uses Git LFS; the wiki has details). Also confirm the units and the timezone column. Do not commit the raw data; add `data/` to `.gitignore`.
- **Fallback:** the [ASHRAE GEPIII Kaggle data](https://www.kaggle.com/c/ashrae-energy-prediction) covers 2016 only. You would need to accept the competition rules on Kaggle yourself. Known quirk: site 0's electricity readings are in kBTU (× 0.2931 converts to kWh).

**ETL decisions (locked before coding):**
- 3,053 meters over 1,636 buildings ⇒ model **per-meter**; aggregate to building level only in the dashboard/anomaly layer.
- Weather↔meter timezone alignment is the known time-sink (per-site local time, DST straddles, some `cleaned_heated` variants not local) — sidestep by starting from the **`kaggle/` folder** (pre-joined weather, 2016-only) for iteration, and only wrestle full 2017 alignment for the final run.
- Require **≥80% coverage in both 2016 and 2017** before a building is quoted in CV(RMSE)/NMBE.

### Carbon intensity: GB Carbon Intensity API (National Grid ESO)
- Docs: <https://carbon-intensity.github.io/api-definitions/>. No API key needed. Half-hourly data in gCO₂/kWh, timestamps in UTC.
- Forecast: `GET https://api.carbonintensity.org.uk/intensity/{from}/fw48h`
- History: `GET /intensity/{from}/{to}` (at most 14 days per request)
- **Design note:** the buildings are in North America and Europe, while this carbon signal covers Great Britain. Keep carbon behind a `CarbonProvider` interface so the source is swappable, and say this openly in the README. The shifting logic works with any carbon curve. Name Electricity Maps or WattTime as drop-in providers for other grids, Dubai included.
  - `UKCarbonProvider` (live, default)
  - `DewaCarbonProvider` (UAE scenario: DEWA's published grid emission factor, optionally a simple day/night or declining-with-solar-share curve)
  - `CsvCarbonProvider` (offline fallback: cache a few weeks of history so the demo works without internet)

### UAE context data (research 16 Nov 2026 — verified against the portals' APIs)
There is **no public BDG2-equivalent for the UAE**, so the UAE layer is synthetic + calibrated, never claimed as measured:
- **KAPSARC** `datasource.kapsarc.org` (Opendatasoft v2.1 API): `electricity-consumption-by-authority` is annual GWh per authority (DEWA/ADWEA/SEWA/FEWA), 3 indicators, **2013–2017 only** (DEWA 2017 = 43,777 GWh, ~4.8%/yr growth). Used as the scaling anchor; check in the pulled records as `data/uae/kapsarc_dewa_annual.csv`.
- **bayanat.ae / data.dubai:** old Dubai Pulse sets are stale annual XLSX; the new data.dubai portal gates DEWA data behind registration + a Data Exchange Hub access request (UAE Pass SSO) — unobtainable before the deadline. `dubaipulse.ae` is dead.
- **ERA5 hourly weather for Dubai** (open, current): drives the cooling-dominated UAE scenario load profile.
- **README line (required):** "UAE figures are calibrated projections; no public hourly smart-meter data exists for the Gulf region."

---

## 3. Architecture

```
            BDG2 CSVs                     GB Carbon Intensity API
                │                                   │
        ┌───────▼────────┐                 ┌────────▼────────┐
        │ ETL (pandas)   │                 │ CarbonProvider  │
        │ clean, join,   │                 │ (live | csv)    │
        │ parquet out    │                 └────────┬────────┘
        └───────┬────────┘                          │
        ┌───────▼────────┐   model.joblib  ┌────────▼────────┐
        │ Train LightGBM ├────────────────►│  FastAPI        │
        │ + evaluate     │                 │  /forecast      │
        └────────────────┘                 │  /anomalies     │
                                           │  /shift         │
                                           └────────┬────────┘
                                                    │ JSON
                                           ┌────────▼────────┐
                                           │ Next.js + charts│
                                           └─────────────────┘
```

## 4. Repo layout

```
gridwise/
├── README.md
├── pyproject.toml            # or requirements.txt
├── .gitignore                # data/, models/, .env, node_modules/
├── data/                     # gitignored: raw/ and processed/ (NOT data/uae/)
│   └── uae/                  # committed: kapsarc_dewa_annual.csv, era5 dubai sample
├── models/                   # gitignored: model.joblib, metrics.json
├── notebooks/
│   └── 01_eda.ipynb          # quick EDA, plots reused in the README
├── gridwise/
│   ├── config.py             # paths, constants, flexible share, thresholds
│   ├── etl.py                # load BDG2, clean, join weather/metadata, write parquet
│   ├── features.py           # feature engineering (shared by training and the API)
│   ├── train.py              # time split, fit, evaluate, save model + metrics
│   ├── evaluate.py           # metrics: CV(RMSE), NMBE, RMSLE, MAE; baseline comparison
│   ├── anomalies.py          # residual-based detection
│   ├── shifting.py           # carbon-aware load-shift optimiser
│   ├── carbon.py             # CarbonProvider interface + UK and CSV providers
│   └── api/
│       ├── main.py           # FastAPI app
│       └── schemas.py        # pydantic models
├── tests/
│   ├── test_features.py
│   ├── test_shifting.py      # shifting conserves energy and never raises CO₂
│   └── test_api.py
├── web/                      # Next.js app (App Router, TypeScript, Recharts)
│   └── app/page.tsx
└── docs/
    ├── architecture.png
    └── screenshots/
```

**Stack:** Python 3.11, pandas, pyarrow, LightGBM, scikit-learn, holidays, FastAPI, pydantic, httpx, joblib, pytest. Frontend: Next.js, TypeScript, Recharts and React Query (which you already use at work).

---

## 5. Technical design

### 5.1 ETL (`etl.py`)
1. Load electricity meters (wide: one column per meter) and **melt** to long format: `building_id, timestamp, kwh`.
2. Join metadata (`primary_use`, `sqm`, `year_built`, `site_id`, `timezone`) and site weather on `site_id + timestamp`.
3. Clean:
   - drop meters with more than 20% missing hours;
   - drop long runs of zeros (24 h or more) and stuck values (the same value for 48 h or more), since these are meter faults, not real use;
   - align weather to each meter's timezone, then interpolate weather gaps of up to 3 h;
   - clip negative readings to missing.
4. Write `data/processed/electricity.parquet` with float32 columns, and log the row and meter counts kept vs dropped.
5. **Memory tip:** start with about 200 buildings across several sites to iterate quickly, then scale to everything for the final run.

### 5.2 Features (`features.py`)
| Group | Features |
|---|---|
| Calendar | hour, day of week, month, weekend flag, public holiday flag (`holidays` library, by site country) |
| Weather | air temperature, dew point, wind speed, cloud cover; 24 h rolling mean temperature; cooling degree hours (base 18 °C); heating degree hours (base 15.5 °C) |
| Building | building_id (categorical), primary_use (categorical), log(floor area), building age, site_id |
| Lag (nice to have) | load 24 h and 168 h ago, rolling 24 h mean load |

- Target: `log1p(kwh)`. Predict, then apply `expm1`.
- The API imports the same `features.py`, so training and serving can't drift apart.

### 5.3 Model (`train.py`)
- **Time split, no shuffling:** train on 2016, validate on Oct–Dec 2016 for early stopping, test on 2017. Never let 2017 data into training.
- LightGBM regressor starting point: `num_leaves=255, learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8, n_estimators=2000`, with early stopping (100 rounds).
- **Baseline to beat:** seasonal naive (same hour one week earlier). Every result is reported against it.
- Save `models/model.joblib` and `models/metrics.json`.

### 5.4 Evaluation (`evaluate.py`)
Report on the 2017 test year, both overall and by building type:

| Metric | Why |
|---|---|
| **CV(RMSE)** | ASHRAE Guideline 14 calibration metric; hourly target is 30% or less |
| **NMBE** | Bias; hourly target is within ±10% |
| RMSLE | Matches the ASHRAE Kaggle competition, so results are comparable |
| MAE (kWh) | Easy to read |
| % improvement vs seasonal naive | The headline number for the CV |

> Use CV(RMSE), not MAPE: buildings with near-zero hours make MAPE blow up.

### 5.5 Anomaly detection (`anomalies.py`)
1. Residual: `r = actual − forecast`.
2. Rolling robust z-score per building: `z = (r − median₁₆₈ₕ) / (1.4826 · MAD₁₆₈ₕ)`.
3. Flag an **anomaly event** when z > 3 for **3 or more hours in a row**. Merge nearby hours into one event with start, end, excess kWh and peak z.
4. Rule-based flags as well:
   - **night baseload drift:** median use from 00:00 to 05:00 is more than 25% above that building's 4-week median;
   - **stuck meter:** the same value for 6 h or more.
5. **Validation:** the data has no labels, so inject synthetic anomalies (for example +40% load for 6 h on random days in the test set) and report precision and recall. That gives the README an honest number.

### 5.6 Carbon-aware load shifting (`shifting.py`)
Inputs: load forecast `L[t]` and carbon intensity `CI[t]` (gCO₂/kWh) over a 24–48 h horizon, flexible share `f` (default 15%, representing HVAC pre-cooling, EV charging, water heating), a per-hour headroom cap `H` (default +20% of `L[t]`) and an optional allowed time window.

Algorithm (greedy, fully explainable):
1. Flexible energy available each hour: `F[t] = f · L[t]`.
2. Sort hours by CI, from dirtiest to cleanest.
3. Move energy from the dirtiest hours into the cleanest hours with headroom left, until `CI_source − CI_target` drops below a minimum gain (for example 10 g/kWh) or all headroom is used.
4. Output the new load profile, the list of moves (from, to, kWh) and `CO₂ saved = Σ moved_kWh × (CI_from − CI_to)`.
- **Tests (property-based):** total kWh is unchanged; no hour goes over its cap; CO₂ saved is never negative; nothing moves when CI is flat.
- Stretch goal: the same problem as a small linear program (`scipy.optimize.linprog`), to show the greedy result matches the optimum.

### 5.7 API (`api/main.py`)
| Method | Path | Returns |
|---|---|---|
| GET | `/health` | status and model version |
| GET | `/buildings` | id, primary use, floor area, site for each building |
| GET | `/buildings/{id}/forecast?start=&end=` | hourly forecast vs actual |
| GET | `/buildings/{id}/anomalies?start=&end=` | anomaly events |
| POST | `/buildings/{id}/shift` | body `{start, horizon_h, flexible_share, headroom}` → new profile, moves, kWh shifted, CO₂ saved |
| GET | `/carbon?start=&hours=` | the carbon curve used |
| GET | `/metrics` | the contents of metrics.json |

- Historical replay: the API serves 2017 test-period data as if it were live. Pick a "now" date in 2017 and step forward. The README must say this is a replay.
- The GB Carbon Intensity API keeps only ~2 years of history and caps requests at 14 days, so a 2017 "carbon replay" is an **hour-of-week proxy**, not a true backcast — label it as such.
- Map the 2017 replay timestamps onto the carbon curve by hour of week, and document that too.

### 5.8 Dashboard (`web/`)
- Header: building picker and date picker.
- **Forecast chart:** actual vs forecast line with shaded anomaly events.
- **Shift panel:** a slider for flexible share (0–30%), then a chart of the original vs shifted load with the carbon intensity line on a second axis.
- **Impact cards:** kWh shifted, kg CO₂ avoided, % of carbon reduced, and the number of anomalies this week.
- **Model card:** CV(RMSE), NMBE and improvement vs baseline, read from `/metrics`.

---

## 6. Schedule (about 6 working days plus a buffer)

| Day | Date | Work | Done when |
|---|---|---|---|
| 1 | Mon 16 Nov | Repo setup, download BDG2, ETL on a 200-building subset (start from `kaggle/`), EDA notebook, KAPSARC anchor CSV | `electricity.parquet` exists; 3 EDA plots saved |
| 2 | Tue 18 Nov | Features, seasonal naive baseline, first LightGBM, evaluation module | `metrics.json` shows the model beating the baseline on 2017 |
| 3 | Fri 20 Nov | Scale to all buildings, light tuning, anomaly detection + synthetic test | Final metrics; anomaly precision and recall |
| 4 | Sat 21 Nov | CarbonProvider, shifting optimiser, tests, FastAPI endpoints | `pytest` passes; `/shift` returns sensible output |
| 5 | Sun 22 Nov | Next.js dashboard wired to the API | All 4 panels render for any building |
| 6 | Wed 25 Nov | README, architecture diagram, screenshots, results table, cleanup | A stranger could run it from the README |
| Buffer | 26–28 Nov | Nice-to-haves (deploy, SHAP, Docker), final CV bullet | Repo public; link on CV and LinkedIn |

Change the days to fit around work; the order matters more than the dates.

---

## 7. Task checklist

### Setup
- [ ] `git init` the GridWise repo (public), Python environment and `.gitignore`
- [ ] Download BDG2; confirm units, timezone column and file paths
- [ ] `config.py` with paths and constants
- [ ] Pull KAPSARC DEWA annual records into `data/uae/kapsarc_dewa_annual.csv`

### Data and model
- [ ] `etl.py`: melt, join, clean, parquet
- [ ] EDA notebook: load profile by building type, temperature vs load, weekday vs weekend
- [ ] `features.py`
- [ ] Seasonal naive baseline
- [ ] `train.py` + `evaluate.py` (time split, metrics, save)
- [ ] Full-data run; record final numbers

### Analytics
- [ ] `anomalies.py` (residual z-score + rules)
- [ ] Synthetic anomaly injection → precision and recall
- [ ] `carbon.py` (UK live + CSV fallback; cache responses)
- [ ] `shifting.py` + property tests

### Service and UI
- [ ] FastAPI endpoints + pydantic schemas + `test_api.py`
- [ ] Next.js dashboard (4 panels)
- [ ] CORS, environment variables, loading and error states

### Ship
- [ ] README: pitch, architecture, results table, screenshots, limitations, how to run
- [ ] Optional: Docker Compose / deploy / SHAP / LLM report
- [ ] Add repo link and final metrics to the CV

---

## 8. README results table (fill in from `metrics.json`)

| Model | CV(RMSE) | NMBE | RMSLE | vs seasonal naive |
|---|---|---|---|---|
| Seasonal naive | – | – | – | – |
| GridWise LightGBM | – | – | – | –% |

Also add: anomaly precision and recall on the synthetic test, and an example shift result ("15% flexible load → X kWh shifted, Y kg CO₂ avoided per week, Z% less carbon").

## 9. CV bullet template (fill in real numbers only)

> **GridWise: AI Building Energy Optimiser** | *Python, LightGBM, FastAPI, Next.js* — Forecast hourly electricity demand for **[N] buildings** (BDG2 dataset), reaching **[X]% CV(RMSE)**, **[Y]% better** than a seasonal baseline. Added anomaly detection (**[P]% precision** on injected faults) and a carbon-aware load-shifting service that cut modelled emissions by **[Z]%** at 15% flexible load.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Dataset too large for the laptop | Use a 200-building subset first, float32, and parquet; for the final run, train per site if needed |
| Raw data is messy (zeros, gaps, unit issues) | Use the `cleaned/` folder, follow the cleaning rules above, and log what was dropped |
| Carbon data is for a different region than the buildings | Swappable provider; state it openly; map by hour of week |
| "Is the UAE part real data?" | It never pretends to be: calibrated projection with a README disclaimer; KAPSARC anchor checked in as CSV |
| Carbon API is down during a demo | CSV fallback provider |
| Running out of time | Nice-to-haves come last; the must-haves alone make the CV line |
| Interviewer asks "is it real-time?" | Be upfront: a historical replay that is designed to plug into live meter feeds (MQTT or BMS), which links back to your IoT work at Scientechnic |

## 11. Interview talking points
- Why a time-based split and why CV(RMSE) and NMBE (the ASHRAE Guideline 14 standard for measurement and verification).
- The trade-off between the weather-only baseline model, used for measurement, verification and anomaly detection, and the lag-feature model, used for operational day-ahead forecasts.
- Why a greedy shift is explainable to facility managers, and what an LP or MPC version would add.
- How this would plug into a building management system (BMS) such as Schneider's EcoStruxure Building, and how your traffic-sensor work at Scientechnic carries over to building telemetry.
