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
