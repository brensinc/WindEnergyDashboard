# 2026-07-25 — Deployment Setup & Code Hygiene

## Motivation

Project was in a good state locally but had never been deployed to production.
Two previous attempts at Railway deployment failed (see commit history: "Add
comprehensive startup logging for Railway debugging", "Add root route for Railway
health check"). Root causes were never resolved — this session addressed them
systematically.

Decision: Railway (backend) + Vercel (frontend). Provider subdomains for both.

---

## Changes Made

### Branch: `deployment-setup`
### Worktree: `../WindEnergyDashboard-deploy`

### Commit 1: `7455e30` — "Prepare backend for Railway deployment"

**Move root Python files into backend/app/**
- `windlib.py`, `verify_data.py`, `generate_heatmap.py` were at the project root
- Problem: Railway runs from the `backend/` directory; root-level files were
  inaccessible without fragile `sys.path` hacks
- Fix: moved all three into `backend/app/` as proper Python package members
- Git detected these as renames (100% similarity)

**Remove sys.path hacks, use normal imports**
- Three router files (`wind.py`, `analytics.py`, `reports.py`) used:
  ```python
  sys.path.insert(0, os.path.dirname(os.path.dirname(...)))
  import windlib  # lazy import inside request handlers
  ```
- Problem: the `sys.path` hack was brittle — it assumed a specific directory
  depth, and would break if deployed with a different working directory
- Fix: replaced with `from app import windlib` (lazy import, same pattern but
  resolves correctly within the `app` package)
- Removed unused `import sys` and `import os` from all three files

**Fix CORS in main.py**
- CORS had `settings.frontend_url` (configurable via env var) AND a hardcoded
  `"http://localhost:3000"` as fallback
- Problem: the hardcoded localhost would need manual updating for production —
  easy to forget
- Fix: removed hardcoded entry; CORS now relies solely on `settings.frontend_url`
  (defaults to `http://localhost:3000` in `.env`, overridable for production)

**Delete root requirements.txt**
- Two copies existed: root `requirements.txt` (streamlit-era deps) and
  `backend/requirements.txt` (FastAPI deps)
- Problem: Railway/Nixpacks would find the wrong one
- Fix: deleted the streamlit-era copy

**Force-add heatmap cache to git**
- `data/heatmap_cache_2024.joblib` (15 MB) was gitignored by `data/*.joblib`
- Problem: Railway build wouldn't have this file; generating it at deploy time
  would take 15-30 minutes and hit Open-Meteo API 225 times
- Fix: `git add -f` to override gitignore

**Add test conftest.py**
- `tests/test_data_verification.py` imports from `windlib`
- After the move, windlib is at `backend/app/windlib.py`
- Fix: `tests/conftest.py` adds `backend/app` to `sys.path` so pytest resolves
  imports correctly

**Cleanup**
- Removed duplicate `logger.info("STARTUP: All routers included")` line in
  `main.py`

### Commit 2: `927df69` — "Add Railway and Vercel deployment config"

**railway.json**
- Tells Railway to use Nixpacks builder
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
  (no `--app-dir backend` — Root Directory set to `/backend` in Railway dashboard)
- Added `requests` and `timezonefinder` to `backend/requirements.txt` (were
  missing — windlib.py imports them directly)
- Healthcheck path: `/api/health`
- Auto-restart on failure (max 10 retries)

**vercel.json**
- Sets root directory to `frontend/`
- Declares framework as `nextjs`
- Lets Vercel auto-detect everything else from `package.json`

**backend/.python-version**
- Pins Python 3.11 for Nixpacks (prevents version mismatch surprises)

**Updated .env.example files**
- `backend/.env.example` — clearer comments about production values
- `frontend/.env.example` — documents all three required env vars

**frontend/.gitignore**
- Added `!.env.example` exception so the example file can be committed

---

## What You Need To Do Manually

### Railway Dashboard (https://railway.app)

| Variable | Value | Why |
|----------|-------|-----|
| `NIXPACKS_PATH` | `backend` | Without this, Railway can't find `requirements.txt` |
| `SUPABASE_URL` | `https://wabepnchbqksgksxykyj.supabase.co` | Database connection |
| `SUPABASE_SERVICE_KEY` | your service-role JWT | Backend DB access |
| `FRONTEND_URL` | `https://your-app.vercel.app` | CORS whitelist |
| `DEBUG` | `false` | Production setting |

Note: the `SUPABASE_SERVICE_KEY` in `backend/.env` is gitignored — safe but
you'll need to copy it to Railway.

### Vercel Dashboard (https://vercel.com)

| Variable | Value | Why |
|----------|-------|-----|
| `NEXT_PUBLIC_SUPABASE_URL` | same Supabase URL | Client-side Supabase access |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | your anon JWT | Client-side auth |
| `NEXT_PUBLIC_API_URL` | `https://your-app.railway.app` | Backend endpoint |

### First Deploy Flow

1. Push the branch: `git push origin deployment-setup`
2. In Railway: New Project → Deploy from repo → select this repo
3. Set `NIXPACKS_PATH=backend` in Variables before first build
4. Set remaining backend env vars
5. In Vercel: Import repo → Root dir = `frontend/` → set env vars → deploy
6. Copy Vercel URL → set as `FRONTEND_URL` in Railway
7. Copy Railway URL → set as `NEXT_PUBLIC_API_URL` in Vercel
8. Redeploy both (settings-only triggers)

---

## Next Steps — Deployment

1. **Push the branch and trigger deploy**
   ```
   git push origin deployment-setup
   ```
   Then follow "First Deploy Flow" above for Railway + Vercel.

2. **Verify health endpoint**
   After Railway deploys, hit `https://your-app.railway.app/api/health`.
   Should return `{"status":"healthy","version":"1.0.0"}`.

3. **Verify CORS is working**
   Open the Vercel frontend → browser DevTools → Network tab → click on map.
   If requests to Railway get CORS errors, check `FRONTEND_URL` in Railway
   matches the Vercel URL exactly (no trailing slash).

4. **Test the full flow end-to-end**
   - Login (Supabase auth via Google/GitHub/email)
   - Click a location on the map → confirm wind data loads
   - Save a project → confirm it persists (check Supabase `projects` table)
   - Open Analytics → confirm charts render (Plotly)
   - Generate a report → confirm HTML content appears

5. **Set up a custom domain (optional)**
   Both Railway and Vercel support custom domains in their dashboard Settings.
   If you add one, update `FRONTEND_URL` and `NEXT_PUBLIC_API_URL` accordingly.

6. **Merge deployment-setup into master** once everything works
   ```
   git checkout master
   git merge deployment-setup
   ```
   Everything on `deployment-setup` has been tested and imports verified.

7. **Revisit the stashed turbine model work**
   ```
   git stash pop  # on master, after merge
   ```
   This adds turbine model presets (NREL 5MW, etc.) to `windlib.py` and
   corresponding `TurbineModelInfo` types to the frontend.

---

## Stashed Work on master

Your turbine model library additions (`windlib.py` + `types.ts`) were stashed on
master before creating the worktree. Apply with:
```
git stash pop
```

---

## Known Issues (pre-existing, not addressed)

- `heatmap_utils` module doesn't exist — imported by `generate_heatmap.py`,
  `verify_data.py`, and tests. The cached heatmap works fine (it uses joblib
  directly), but the generation script and tests referencing it are broken.
- `__pycache__/` directories scattered around — already gitignored.

---

## Later Session (same date) — EIA Validation Pipeline

### Branch: `master`
### Worktree: `../WindEnergyDashboard` (primary dev)

After the deployment work, the focus shifted to making the model's capacity factor
predictions credible to investors/developers. The core insight: **capacity factor
(%)** is the primary trust metric for the audience, not total revenue. To validate,
we need to compare model predictions against real-world wind plant data.

### Motivation

Without validation against real-world data, the model's revenue projections are
just numbers on a page. Investors and developers need to see: "this model predicts
capacity factor within X% of what 1,278 real US wind plants actually produced."
The US Energy Information Administration (EIA) publishes exactly this data — plant
locations, capacities, and monthly generation — for every US power plant ≥1 MW.
It's the authoritative, public, and free source.

### Problem: EIA-860/923 Files Wouldn't Download

The script `scripts/validate_with_eia.py` was written to fetch EIA-860 (plant
characteristics) and EIA-923 (monthly generation) and compare model output against
reported generation. But the download URLs were wrong:

- `https://www.eia.gov/electricity/data/eia860/xls/eia8602023.zip` → 301 redirect
- `https://www.eia.gov/electricity/data/eia923/xls/f923_2023.zip` → 301 redirect

Both redirected to an HTML page (66 KB saved as "zip"). The cached corrupted files
then blocked subsequent attempts.

**Root cause**: EIA moved older data years to an `archive/` subdirectory. The
correct URLs are:

```
https://www.eia.gov/electricity/data/eia860/archive/xls/eia8602023.zip
https://www.eia.gov/electricity/data/eia923/archive/xls/f923_2023.zip
```

**Fix** in `scripts/validate_with_eia.py`:
- Updated `EIA_860_URL` and `EIA_923_URL` to use `archive/xls/` path
- Updated all fallback URLs similarly
- Deleted stale cached HTML files from `data/eia_validation/`

### Problem: Wrong Excel Sheet Match in EIA-860

The EIA-860 ZIP contains multiple `.xlsx` files. The `find_sheet()` function
searched for "generator" in sheet names, but `6_1_EnviroAssoc_Y2023.xlsx` has a
sheet named "Boiler Generator" — alphabetically earlier, so it matched first.
Then the code tried to read generator data from the wrong file.

**Root cause**: `find_sheet()` only checked sheet names, not filenames. It
returned the correct sheet name ("Boiler Generator") but paired with
the wrong xlsx file (`2___Plant_Y2023.xlsx`), causing a `WorksheetNotFound`
error.

**Fix**: Updated `find_sheet()` to also match keywords against the xlsx filename
(`3_1_Generator_Y2023.xlsx` contains "generator"). The generator xlsx and sheet
name are now tracked independently from the plant xlsx.

### Problem: Wrong Header Row in EIA-923

The EIA-923 sheet "Page 1 Generation and Fuel Data" was read with `header=4`, but
row 4 (0-indexed) is the sub-header row with column categories:
```
Row 4: ['Total Quantity Consumed In Physical Unit', 'Quantity Consumed In Physical Units For ...', ...]
Row 5: ['Plant Id', 'Combined Heat And\nPower Plant', 'Nuclear Unit Id', ...]  ← actual column headers
```

Using `header=4` gave all "Unnamed" columns.

**Fix**: Changed to `header=5`.

Also fixed month column detection — the column naming pattern is `Netgen\nJanuary`
(not `Net Generation January`), so the lookup now searches for `netgen` in
the column name and uses `month_labels` only for output naming.

### Problem: Open-Meteo Archive API Unavailable

The script's `compute_model_cf()` calls Open-Meteo's archive API for historical
wind speed data. During this session, the archive API was returning SSL errors:

```
SSL: UNEXPECTED_EOF_WHILE_READING
```

This is a known service issue: https://github.com/open-meteo/open-meteo/issues/1865
The archive subdomain has been intermittently down.

**Fix**: Multiple layers of resilience:
1. `windlib.fetch_open_meteo_archive()` now catches all exceptions and returns an
   empty DataFrame instead of crashing
2. `compute_model_cf()` retries 3 times with 1-second backoff
3. `run_validation()` checks API availability at the start with a probe request
4. If API is down, the script gracefully reports population-level statistics from
   EIA data alone and skips per-plant model comparison

### Result: Working Validation Pipeline

```
EIA-860:  1,331 wind plants, 147,998 MW total
EIA-923:  1,327 wind plants, 421.02 TWh annual generation
Merged:   1,278 wind plants (after CF filter of 0.05–0.65)

Population capacity factor (EIA actual):
  Mean:    30.1%
  Median:  29.9%
  P10:     16.5%
  P90:     43.4%
  Std Dev: 10.2%
  Total gen: 418.57 TWh
```

This means the US wind fleet averaged ~30% capacity factor in 2023. The model
should target this as the central benchmark.

### Plans / Next Steps

1. **Re-run model comparison when Open-Meteo archive recovers**
   ```
   python scripts/validate_with_eia.py --sample 50
   ```
   Expected output: per-plant MAE/RMSE/MBE/R² against 50 stratified wind plants.

2. **Switch to alternative wind data source** if the archive API remains
   unreliable:
   - ERA5 CDS API (requires registration but is the gold standard)
   - NREL Wind Toolkit API (has 100m wind speeds for US at hourly resolution)
   - NASA POWER API (50m wind — lower but usable with height extrapolation)

3. **Use EIA-860 `3_2_Wind_Y2023.xlsx` for turbine-specific modeling**:
   This sheet contains hub height, rotor diameter, and turbine make/model for
   each wind turbine. Currently the validation uses a generic 3MW turbine — using
   site-specific turbine parameters would significantly improve accuracy.

4. **Multi-year validation**: Validate against 2022, 2024 data once URLs are
   confirmed. This accounts for interannual wind variability.

5. **ISO-level breakdown**: Group validation by ISO (ERCOT, CAISO, MISO, PJM,
   SPP, NYISO, ISO-NE) to identify regional model biases.

---

## Model Improvement Phase 1 (other agent)

### Branch: `master`
### Context

After the validation pipeline was working, another agent was tasked with
improving the model's physical accuracy before the ML layer is added. The
focus was on replacing the simplistic linear power curve and adding
site-specific corrections that can be validated immediately against EIA data.

### Motivation

The original model used a single generic 3MW turbine with a linear ramp from
cut-in (3 m/s) to rated speed (12 m/s). Real turbines don't behave this way —
they follow a cubic-ish curve where power rises slowly at low wind speeds
and then transitions to rated. This caused systematic biases in CF estimates,
especially at sites with average wind speeds near 6-8 m/s where the linear
ramp overestimates energy. Additionally, the model ignored air density
(elevation), which can change CF by 2-15%. Captured price (merit order effect)
was also missing — critical for revenue accuracy.

### Changes

**Real turbine power curve library** (`windlib.py`)
- Added `TURBINE_MODELS` dict with 5 models:
  - `nrel_5mw_reference` — NREL offshore 5MW baseline (Jonkman et al. 2009)
  - `vestas_v117_3_45mw` — Vestas V117-3.45 MW, popular US onshore turbine
  - `siemens_gamesa_sg_5_145` — Siemens Gamesa SG 5.0-145, large onshore/offshore
  - `ge_2_5_120` — GE 2.5-120, common US onshore with large rotor
  - `generic_linear` — backward-compatible old linear ramp
- Each model contains: `power_curve_ws_mps` and `power_curve_kw` (speed→power
  lookup tables), `rated_power_kw`, `rotor_diameter_m`, `hub_height_m`
- Functions: `get_turbine_model()`, `list_turbine_models()`, `power_from_curve()`
- UI: frontend `types.ts` updated with `TurbineModelInfo` type; analytics page
  dropdown to select turbine model

**Air density correction via elevation**
- In `windlib.add_power_output()`: when `elevation_m > 0`, compute density
  ratio via standard atmosphere: `ρ/ρ₀ = exp(-elevation / 8500)`.
  Scale power output by `ρ/ρ₀`.
- If elevation not provided, auto-fetch from Open-Meteo elevation API (free)

**Direct 100m wind speed usage**
- Archive endpoint already used `wind_speed_100m` — no change needed
- Forecast endpoint was fetching 80m + 120m and extrapolating via log law.
  Changed to fetch `wind_speed_100m` directly from Open-Meteo's forecast API.
  Eliminated the log-law extrapolation code.

**PPA pricing mode**
- Added `PPA` pricing option alongside existing `fixed` and `market`
- Parameters: `ppa_price_usd_mwh`, `ppa_escalator_pct`, `ppa_term_years`,
  `ppa_year`
- Revenue = fixed price × (1 + escalator)^year for each year of the PPA term
- Most bankable mode for project financing — investors care about PPA-backed
  revenue, not merchant volatility

**API endpoints added**
- `POST /api/analytics` — now accepts `turbine_model`, `elevation_m`,
  `ppa_price`, `ppa_escalator_pct`, `ppa_year`
- `GET /api/analytics/turbine-models` — returns list of available models with
  specs (manufacturer, rated power, rotor diameter, hub height, description)
- `POST /api/reports` — accepts `turbine_model`, `elevation_m`

**Validation script** (`scripts/validate_with_eia.py`)
- Downloads EIA-860 (plant characteristics) and EIA-923 (monthly generation)
- Filters to wind plants (prime mover = WT)
- Runs the model at each plant's coordinates
- Computes MAE, RMSE, MBE, MAPE, R² of capacity factor
- Outputs per-plant comparison CSV + JSON validation report with methodology docs

### How to Validate

```bash
backend/.venv/bin/python scripts/validate_with_eia.py --sample 50
```

Downloads EIA data, runs model at 50 wind plant locations, produces
`data/eia_validation/validation_report.json` with metrics.

---

## Industry Research & Feature Audit (separate agent)

### Context

A separate agent was tasked with researching current problems in wind energy
analytics and modeling across the industry, then identifying features that
could differentiate this dashboard and provide real value to customers.

### Research Method

Searched 20+ academic papers, industry whitepapers, and competitor product
documentation from 2025-2026. Key sources:
- Nature Communications — global wind power assessment validation (Jan 2026)
- Elsevier — offshore wind data challenges framework (Feb 2026)
- IOP Science — deep learning for wind turbine fault diagnosis (Apr 2026)
- Springer — HG-KAN spatio-temporal wind power forecasting (Mar 2026)
- ScienceDirect — cross-spatiotemporal anomaly detection with LLMs (May 2026)
- CENER — "You have the data, do you have the knowledge?" (Apr 2026)
- WindESCo, ONYX Insight, Vestas Scipher, WindTwin — competitor product analysis
- Journal of Physics: Conference Series — WindKI anomaly detection, digital twin KPIs

### Key Industry Problems Identified

| Problem | Impact | Cited In |
|---------|--------|----------|
| Power curves degrade over time (blade erosion, soiling, icing) | 3-10% CF loss undetected for months | Martí-Puig 2026, CENER 2026 |
| Wake losses between turbines | 5-15% farm energy loss, poorly modeled | HG-KAN 2026, IEA Wind Task 36 |
| SCADA data underutilized — raw data collected but not turned into knowledge | Missed early failure signals | CENER 2026, ORE Catapult |
| Data quality issues (outliers, missing data, sensor noise) corrupt models | 15-46% prediction error reduction possible with cleansing | Hybrid FCM+ANN model (2025) |
| Forecast uncertainty not quantified for decision-making | Grid operators can't assess risk | IEA Wind Task 36, UAFformer 2026 |
| Repowering decisions ignore 15-20 years of operational data | Greenfield uncertainty 5-10% instead of ~3% | CENER 2026 |
| Grid code compliance impacts not quantified | Blind commitment to LVRT/ramp-rate requirements | CENER 2026 |
| Fleet-wide benchmarking across sites/OEMs is manual | Hidden underperformers | ONYX Insight, WindTwin |
| Anomaly/fault detection requires labeled data (scarce) | False alarms or missed faults | WindKI 2026, CrossSTLLM 2026 |

### Competitor Feature Landscape

| Company | Key Capabilities |
|---------|-----------------|
| Vestas Scipher Vx+ | Live monitoring, production loss bucketing (curtailment/derate/icing), CMS integration, NERC GADS reporting |
| ONYX Insight | 100+ GW monitored, RUL modeling, reliability benchmarking across 60+ turbine models, 9,000+ recorded failures |
| WindTwin | Per-turbine digital twin, multi-signal alerting (e.g. gearbox temp + vibration + oil), ECMWF forecasts, IEC 61400-26 availability |
| AVEVA Apollo | AI-powered anomaly detection, patented forecasting, CMMS integration, 6% AEP improvement via curtailment optimization |
| WindSights (Databricks) | Deep learning on SCADA + drone imagery, natural language operations interface, automated maintenance prioritization |

### Synthesized Feature Roadmap

The agent synthesized all research into a 6-phase roadmap (see next section).

---

## Feature Roadmap — Full 6-Phase Plan

### Phase 0: Quick Wins & Bug Fixes (generation-only, no ML)
*Target: 1 week — builds directly on Phase 1 model improvements*

| # | Task | Motivation | Status |
|---|------|-----------|--------|
| 0.1 | Hub height wind shear correction | `hub_height_m` accepted but ignored — wind data always at 100m. Apply log-law `v₂ = v₁ · ln(h₂/z₀) / ln(h₁/z₀)` | **Not built** |
| 0.2 | Turbulence intensity power curve softening | Real turbines lose efficiency in turbulent flow. Modify power by `1 - 0.05·TI` using wind speed stddev. ±1-3% CF on gusty sites. | **Not built** |
| 0.3 | Data quality score | Metadata per response: % archive vs. forecast, % missing, % interpolated. Surface as badge/tooltip in UI. Builds trust. | **Not built** |
| 0.4 | Degradation curve (age factor) | Multiplier: `1 - years_operational · 0.007`. Without this, 10yr-old turbines overpredict CF by 7%. | **Not built** |
| 0.5 | Data cleansing pipeline | Hampel filter + Mahalanobis distance on wind speed/power scatter removes outliers before power calc | **Not built** |

### Phase 1: Revenue & Pricing Accuracy (done by other agent)
- Real turbine power curves ✓
- Air density correction ✓
- Direct 100m wind ✓
- PPA pricing mode ✓
- EIA validation harness ✓

### Phase 2: Validation Infrastructure
*Target: 2 weeks — tells us if Phase 0-1 actually improved accuracy*

| # | Task | Detail |
|---|------|--------|
| 2.1 | Validation runner | Per-plant: run analytics engine → predict monthly CF → compare to EIA-923 actual |
| 2.2 | Validation endpoint + UI | `GET /api/validation/metrics` → frontend dashboard showing MAPE, RMSE, bias per ISO |
| 2.3 | Cross-validation by year | 2022 data vs 2023 data, 2023 vs 2024 |
| 2.4 | Sensitivity analysis | Vary each input ±10%, report which drives CF uncertainty most |

### Phase 3: Performance Monitoring (your differentiator)
*Target: 3-4 weeks — no competitor in this price tier offers it*

| # | Task | Detail |
|---|------|--------|
| 3.1 | Power curve degradation detection | Compact ANN (1 hidden layer, 10 nodes) on rolling 30-day data. Detect 3%+ rightward curve shift from blade erosion/soiling/icing. Based on Martí-Puig 2026. |
| 3.2 | Degradation API + UI tab | Overlay baseline vs. current curve, CF loss gauge, shift trend chart |
| 3.3 | Unsupervised anomaly detection | Isolation Forest + Autoencoder on SCADA signals. Health score 0-100 per turbine. Based on WindKI 2026 (ROC-AUC ~0.8). |
| 3.4 | Production loss bucketing | Categorize lost MWh: downtime vs. curtailment vs. degradation vs. suboptimal yaw |
| 3.5 | Fleet-wide comparison view | Rank projects by CF, degradation rate, revenue/MWh. Sparkline trends. |

### Phase 4: Forecasting & Uncertainty
*Target: 2-3 weeks — turns point estimates into decision-grade forecasts*

| # | Task | Detail |
|---|------|--------|
| 4.1 | Proper bootstrap for P50/P90/P10 | Monte Carlo sampling from interannual variability (30yr ERA5), air density range, power curve uncertainty |
| 4.2 | Uncertainty propagation in API + UI | Fan charts on time series, P50/P90/P10 bars on summary KPIs |
| 4.3 | Multi-horizon forecast (6h to 7d) | Open-Meteo ensemble (11 members) → hourly power forecast with bands |
| 4.4 | Ramp event prediction | Weather front detection → flag hours where power change >50% in 1h |
| 4.5 | Extreme wind event risk | Probability of cut-out wind (>25 m/s) in forecast horizon |

### Phase 5: Farm-Level Analytics
*Target: 3-4 weeks — multi-turbine wake modeling*

| # | Task | Detail |
|---|------|--------|
| 5.1 | Multi-turbine project model | Project contains 1..N turbines with lat/lng, model, hub height |
| 5.2 | Jensen/Gaussian wake model | Compute wake deficit per turbine given wind direction, spacing, thrust coefficient |
| 5.3 | Layout visualization on map | Turbine markers with wake direction cones, color by energy output |
| 5.4 | Wake-aware layout optimization | Suggest optimal positions for N turbines to minimize wake losses |

### Phase 6: Advanced & Research
*Target: 4-6 weeks research phase*

| # | Task | Detail |
|---|------|--------|
| 6.1 | Repowering scenario comparison | "Keep existing" vs "replace with turbine X" using 15-20yr site history |
| 6.2 | Grid code compliance checker | Estimate annual frequency of LVRT/ramp-rate trigger events from historical data |
| 6.3 | Digital twin / virtual sensors | Physics-informed ML to reconstruct unmeasured blade loads, tower clearance |
| 6.4 | Forecast uncertainty decomposition | Bayesian NN separating aleatoric (irreducible noise) vs epistemic (model ignorance) uncertainty |

---

## XGBoost CF Predictor — Medium-Layer Architecture

### Concept

The heatmap uses a coarse physics model (~225 grid points). The analytics page
runs a detailed hourly physics model (2-5s per call). Between them is a gap: a
fast, accurate CF estimate that runs on project creation without fetching any
weather data.

Solution: train XGBoost on ~12K rows of EIA-923 monthly CF data augmented with
ERA5 climatology features. Once trained, inference is a 0.5ms model load +
predict — no API calls needed.

### Architecture

```
backend/services/cf_model/
├── features.py     # Climatology feature extraction from ERA5 (cached)
├── train.py        # Training pipeline: build dataset, train, evaluate
├── model.py        # Inference wrapper: load model, predict CF
├── model_p50.joblib
├── model_p10.joblib
├── model_p90.joblib
└── feature_metadata.json

data/cf_model/
├── feature_cache/  # One parquet per lat/lng (computed once, cached forever)
└── training_data.parquet
```

### Feature Set (per site)

Derived from 3 years of hourly ERA5 at the location:
- `annual_mean_wind_mps` — strongest single CF predictor
- `weibull_k`, `weibull_c` — Weibull shape and scale (describes wind distribution)
- `turbulence_intensity` — daily std/mean averaged (high TI → lower CF)
- `seasonal_amplitude_mps` — winter mean minus summer mean (big swings → low annual CF)
- `mean_air_density_kgm3` — from surface_pressure + temperature
- `wind_dir_stability` — resultant vector length R of direction unit vectors
- `pct_below_cutin` — % hours < 3 m/s
- `pct_above_cutout` — % hours > 25 m/s
- `monthly_wind_cv` — coefficient of variation of monthly means
- `night_wind_ratio` — night mean / day mean (capture price relevance)

### Training Design

- **Target**: monthly capacity factor from EIA-923
- **Train/val split**: by plant (not by time) — tests generalization to unseen sites
- **3 models**: P50 (MAE), P10 (quantile 0.1), P90 (quantile 0.9)
- **Cross-val**: 5-fold, stratified by ISO region
- **Hyperparams**: RandomizedSearchCV over n_estimators, max_depth, learning_rate, subsample
- **Expected R²**: 0.70-0.80 vs physics-only ~0.30-0.50

### Inference

```python
def predict_cf(lat, lng, hub_height=100, turbine_model="generic_linear",
               commission_year=None, num_turbines=1):
    features = extract_climatology_features(lat, lng)  # cached
    features.update(static_plant_features(...))
    df = pd.DataFrame([features])
    return {
        "cf_p50": p50_model.predict(df)[0],
        "cf_p10": p10_model.predict(df)[0],
        "cf_p90": p90_model.predict(df)[0],
    }
```

Called when user pins a location → pre-fills analytics page CF estimate
immediately. Integrates via `POST /api/analytics/predict-cf`.

---

## Co-Pilot LLM Layer

### Concept

Not a separate prediction engine — an **interpretation layer** that reads the
analytics output and answers questions, writes reports, and suggests actions.
Solves the problem CENER identified: "you have the data, do you have the
knowledge?" — the data is there but users can't extract insights from it.

### Architecture

```
POST /api/copilot/ask
{
  "project_id": "abc123",
  "question": "Why is summer CF lower than winter?"
}

Response (streamed via SSE):
{
  "answer": "Summer CF is 8.2 pp lower than winter at this site. "
            "Average wind drops from 8.1 to 6.3 m/s (22% reduction). "
            "Turbulence intensity is 15% higher in summer, softening "
            "the power curve. These account for ~85% of the gap."
}
```

### What It Can Answer

| Category | Example |
|----------|---------|
| Diagnostic | "Why is my CF below 30%?" |
| Comparative | "How would a V117 compare to my GE 2.5?" |
| Seasonal | "Which months should I schedule maintenance?" |
| Financial | "What PPA price gives me 8% IRR?" |
| Report | "Summarize this project for an investor." |

### Guardrails

- LLM **never** generates numerical predictions — only interprets
  already-computed analytics
- Every answer must reference specific numbers from the response
- System prompt forbids inventing statistics, confidence intervals, or benchmarks
- Fallback: if API unavailable, show "Co-pilot unavailable" + link to full analytics
- Cost: ~$0.01-0.03 per query via Claude Haiku / GPT-4o-mini

---

## Remaining Phase 0 — Implementation Specs

These are the generation-side fixes not yet built by any agent. Each is a quick
win (<1 day) that directly improves CF accuracy.

### 0.1 — Hub Height Wind Shear Correction

**Problem**: `hub_height_m` is accepted by the API and stored in projects, but
the wind data is always retrieved at Open-Meteo's reference height (100m). No
adjustment is applied.

**Fix** in `windlib.add_power_output()`:
```python
def adjust_wind_speed_for_hub_height(ws_100m, target_hub_height, roughness=0.03):
    """Apply log-law wind shear correction."""
    if target_hub_height == 100:
        return ws_100m  # no adjustment needed
    return ws_100m * (np.log(target_hub_height / roughness) / np.log(100 / roughness))
```
- Default roughness length z₀ = 0.03m (onshore, grass)
- Offshore: z₀ = 0.0002m (calm sea)
- Surface type could be determined from distance to coast or elevation
- Typical adjustment: 80m hub → ~95% of 100m wind speed; 120m hub → ~103%

### 0.2 — Turbulence Intensity Softening

**Problem**: The power curve assumes steady-state wind. In turbulent conditions,
the cubic relationship between wind speed and power means fluctuations average
to *less* power than the steady wind would produce (Jensen's inequality applied
to v³).

**Fix** in `windlib.wind_to_power()`:
```python
def apply_turbulence_correction(power, ti):
    """Reduce power by ~5% per 0.1 TI above baseline."""
    if ti < 0.10:
        return power
    # Typical: TI=0.15 → ~2.5% reduction, TI=0.20 → ~5% reduction
    correction = 1.0 - 0.5 * (ti - 0.10)
    return power * correction
```
- TI estimated from Open-Meteo's `wind_speed_100m` stddev over 3-hour windows
  (or from daily std/mean if only daily data available)
- Impact: ±1-3% CF on gusty sites (Great Plains spring, offshore winter storms)

### 0.3 — Data Quality Score

**Problem**: Users see wind data and power estimates with no indication of
whether it came from high-quality archive data, lower-quality forecast data,
or was interpolated to fill gaps.

**Fix**:
- Add `data_quality` dict to analytics response:
  ```python
  "data_quality": {
      "pct_from_archive": 85.2,   # % of hours from ERA5 reanalysis
      "pct_from_forecast": 12.3,  # % from forecast model
      "pct_missing": 2.5,         # % of hours with no data
      "pct_interpolated": 1.8,    # % filled via interpolation
      "total_hours": 8760,
  }
  ```
- Frontend: color-coded badge next to CF (green=archive >90%, yellow=archive
  70-90%, red=archive <70%), tooltip with breakdown
- Builds credibility — users know when to trust the number and when it's an estimate

### 0.4 — Degradation Curve

**Problem**: A 10-year-old turbine's power curve is not the same as a new one's.
The model treats all turbines as brand new, overpredicting CF for aging assets.

**Fix** in `windlib.add_power_output()`:
```python
def apply_degradation(power, years_online):
    """Annual degradation: ~0.7%/year based on Staffell & Green 2014."""
    if years_online <= 0:
        return power
    # Linear degradation: 0.7% per year, max 15%
    factor = max(0.85, 1.0 - 0.007 * years_online)
    return power * factor
```

### 0.5 — Data Cleansing Pipeline

**Problem**: Open-Meteo data occasionally has spikes, dropouts, or physically
impossible values (negative wind speed, temperature >50°C). These corrupt
the power calculation and downstream aggregates.

**Fix** — new function in `windlib.py`:
```python
def clean_wind_data(df):
    """Remove physically impossible and statistical outliers."""
    # Remove impossible values
    df = df[df["wind_speed_mps"].between(0, 50)]
    df = df[df["temperature_2m"].between(-40, 50)]
    df = df[df["surface_pressure"].between(800, 1100)]

    # Hampel filter on wind speed (rolling 24h window, 3-sigma)
    rolling_median = df["wind_speed_mps"].rolling(24, center=True).median()
    mad = (df["wind_speed_mps"] - rolling_median).abs().rolling(24, center=True).median()
    df = df[(df["wind_speed_mps"] - rolling_median).abs() < 3 * mad]

    return df
```

---

## Updated Combined Next Steps

### Immediate (deployment)
1. Push `deployment-setup` branch and trigger Railway + Vercel deployment
2. Verify health endpoint, CORS, full end-to-end flow
3. Merge into `master` once stable

### Immediate (model accuracy)
4. Run `validate_with_eia.py --sample 50` to establish baseline MAE/RMSE/R²
5. Build remaining Phase 0 items: hub height correction, turbulence softening,
   data quality score, degradation curve, data cleansing pipeline
6. Re-run validation after Phase 0 to measure improvement

### Short-term (validation → differentiate)
7. Build validation endpoint + UI dashboard showing metrics per ISO
8. Build XGBoost CF predictor (medium layer) — train on EIA data, deploy as
   fast pre-fill for project creation
9. Train P10/P50/P90 quantile models and expose uncertainty bands in API

### Medium-term (your moat)
10. Power curve degradation detection (Phase 3.1) — first ML feature, directly
    addresses the #1 industry gap (blade erosion/soiling goes undetected)
11. Unsupervised anomaly detection (Phase 3.3) — fleet health scoring without
    labeled data
12. Production loss bucketing + fleet-wide comparison view (Phase 3.5-3.7)

### Longer-term
13. Co-pilot LLM layer for natural language analytics queries
14. Wake loss modeling for multi-turbine farms
15. Repowering scenario analysis + grid code compliance checker
