# Render Deployment Setup Guide

**Last Updated**: 2026-03-23
**Purpose**: Deploy the FastAPI backend to Render (current hosting platform)
**Future**: Migrate to VPS (Hetzner + Coolify) after initial ML models are trained — see [Migration to VPS](#migration-to-vps)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Build & Start Configuration](#build--start-configuration)
4. [Environment Variables](#environment-variables)
5. [Free Tier Notes](#free-tier-notes)
6. [Upgrading for ML Models](#upgrading-for-ml-models)
7. [Migration to VPS](#migration-to-vps)

---

## Prerequisites

- Render account (free at [render.com](https://render.com))
- `bouldering-analysis` repository pushed to GitHub
- Supabase project URL and service role key

---

## Initial Setup

1. Log in to [Render Dashboard](https://dashboard.render.com)
2. Click **New +** → **Web Service**
3. Connect your GitHub account and select the `bouldering-analysis` repository
4. Click **Connect**

---

## Build & Start Configuration

Render deploys from the `Dockerfile` in the repository root. No build or start command
configuration is needed — Render detects the Dockerfile automatically.

| Setting | Value |
|---------|-------|
| **Name** | `bouldering-analysis` |
| **Region** | Frankfurt (EU) or closest to your users |
| **Branch** | `main` |
| **Runtime** | Docker |
| **Instance Type** | Free (512 MB) — sufficient without ML models |

> **Why Docker?** The project pins PyTorch to the CUDA 12.8 index. The `Dockerfile`
> overrides this with the CPU-only build (~700 MB vs ~2.9 GB), which is required for
> Render's CPU-only instances and keeps image size manageable.

---

## Environment Variables

Set these in Render Dashboard → your service → **Environment**:

### Required

| Variable | Value |
|----------|-------|
| `BA_SUPABASE_URL` | `https://<your-project-ref>.supabase.co` |
| `BA_SUPABASE_KEY` | Your Supabase service role key |
| `BA_CORS_ORIGINS` | `["https://grade-my-climb.vercel.app"]` |
| `BA_DEBUG` | `false` |
| `BA_API_KEY` | A random secret string |

### Optional (leave unset for defaults)

| Variable | Default | Notes |
|----------|---------|-------|
| `BA_LOG_LEVEL` | `INFO` | |
| `BA_RATE_LIMIT_UPLOAD` | `10` | Requests/IP/min |
| `BA_MAX_UPLOAD_SIZE_MB` | `10` | |

### ML Models (set when model files are available)

Leave these **unset** for now. The `/analyze` endpoint returns `503` gracefully when
model paths are empty. Set them after upgrading to a paid instance with persistent disk.

| Variable | Value (when ready) |
|----------|--------------------|
| `BA_DETECTION_MODEL_PATH` | `/models/detection/best.pt` |
| `BA_CLASSIFICATION_MODEL_PATH` | `/models/classification/best.pt` |
| `BA_ML_GRADE_MODEL_PATH` | `/models/grading/v<version>` |

---

## Free Tier Notes

- **RAM**: 512 MB — sufficient without ML models loaded (PyTorch models not loaded when
  paths are empty, but the library is still imported: expect ~300–400 MB baseline usage)
- **Sleep on idle**: The free instance spins down after 15 minutes of inactivity.
  First request after idle takes ~30 seconds to wake up.
- **Disk**: No persistent disk on free tier — ML model files cannot be stored.

**Workaround for sleep**: The `scripts/ping_supabase.py` keep-alive script exists but
targets Supabase, not Render. A simple cron ping to `GET /health` every 10 minutes
(via UptimeRobot free tier or similar) will prevent sleep.

---

## Upgrading for ML Models

When ML models are trained and ready:

1. Upgrade instance to **Starter ($25/month, 2 GB RAM)** in Render Dashboard
2. Add a **Disk** (Render Dashboard → your service → Disks → Add Disk):
   - Mount path: `/models`
   - Size: 1–2 GB (sufficient for `.pt` files + XGBoost model)
3. SSH into the service or use Render Shell to copy model files:

   ```bash
   # From your local machine
   scp models/detection/best.pt <render-ssh-address>:/models/detection/
   scp models/classification/best.pt <render-ssh-address>:/models/classification/
   ```

4. Set `BA_DETECTION_MODEL_PATH`, `BA_CLASSIFICATION_MODEL_PATH`,
   `BA_ML_GRADE_MODEL_PATH` env vars in Render Dashboard
5. Redeploy

> **Note**: Once ML models are stable, consider migrating to VPS (Hetzner + Coolify)
> for significantly better value — 4 GB RAM at ~€4/month vs $25/month on Render.
> See [Migration to VPS](#migration-to-vps).

---

## Migration to VPS

**When to migrate**: After initial ML models are trained and the Render $25/month cost
becomes a concern.

**Complexity**: Low (~1 hour total). The backend is fully stateless — Supabase owns all
persistent data. Nothing needs to be migrated except the runtime environment.

**Steps**:

1. Spin up a [Hetzner CX22](https://www.hetzner.com/cloud) VPS (€4/month, 4 GB RAM,
   Ubuntu 24.04)
2. Install [Coolify](https://coolify.io) (one command, ~20 min)
3. Connect the same GitHub repo in Coolify — no code changes needed
4. Copy env vars from Render Dashboard → Coolify
5. Transfer ML model files to VPS:

   ```bash
   scp -r models/ root@<vps-ip>:/opt/models/
   ```

6. Update `VITE_API_URL` in Vercel dashboard (new backend URL)
7. Update `BA_CORS_ORIGINS` if backend URL changes
8. Verify health endpoint and smoke test
9. Decommission Render service
