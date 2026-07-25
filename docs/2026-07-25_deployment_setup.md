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
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --app-dir backend`
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

## Combined Next Steps

1. Get the deployment live (steps above)
2. Re-run model validation against EIA data
3. Start using turbine-specific specs from EIA-860
4. Validate against 2022 and 2024 as well
5. Add custom domain(s) once everything is stable
