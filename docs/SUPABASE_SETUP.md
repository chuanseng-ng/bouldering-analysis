# Supabase Setup Guide

This guide explains how to connect your Supabase project to the bouldering-analysis application.

## Overview

The application uses Supabase for:

- **Database**: PostgreSQL database for storing route data, analysis results, and metadata
- **Storage**: Object storage for route images and model outputs
- **Authentication** (future): User authentication and authorization

## Prerequisites

- A Supabase account (free tier works fine)
- Python 3.10+ with project dependencies installed

---

## Step 1: Create a Supabase Project

1. **Sign up for Supabase** (if you haven't already):
   - Go to [https://supabase.com](https://supabase.com)
   - Click "Start your project" and create an account

2. **Create a new project**:
   - Click "New Project"
   - Choose your organization
   - Fill in project details:
     - **Name**: `bouldering-analysis` (or your preferred name)
     - **Database Password**: Choose a strong password (save it securely!)
     - **Region**: Select the region closest to you
   - Click "Create new project"
   - Wait 2-3 minutes for Supabase to set up your project

---

## Step 2: Get Your API Credentials

Once your project is ready:

1. **Navigate to Project Settings**:
   - Click the gear icon (⚙️) in the left sidebar
   - Select "API" from the settings menu

2. **Copy your credentials**:
   - **Project URL**: Look for "Project URL" (e.g., `https://abcdefghijklmnop.supabase.co`)
   - **API Key**: Copy the `service_role` key (under "Project API keys")

   ```text
   Project URL: https://xxxxxxxxxxxxx.supabase.co
   service_role key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

   **Important**:
   - ✅ Use the `service_role` key for this **server-side** FastAPI backend
   - ✅ The `service_role` key bypasses Row Level Security — keep it server-side only
   - ❌ NOT the `anon` / `publishable` key — the `anon` key cannot INSERT rows due to
     the RLS policy and is intended for untrusted client-side code (browser, mobile)
   - ❌ Never expose the `service_role` key in client code or commit it to version control

---

## Step 3: Configure Environment Variables

Create a `.env` file in your project root:

```bash
# Create .env file
touch .env
```

Add your Supabase credentials to `.env`:

```bash
# Application Settings
BA_APP_NAME=bouldering-analysis
BA_APP_VERSION=0.1.0
BA_DEBUG=true
BA_LOG_LEVEL=DEBUG
BA_CORS_ORIGINS=["http://localhost:3000"]

# Supabase Configuration
BA_SUPABASE_URL=https://xxxxxxxxxxxxx.supabase.co
BA_SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...  # service_role key
```

**Replace**:
- `xxxxxxxxxxxxx` with your actual project URL
- `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` with your **`service_role`** key

**Note**: The backend uses the `service_role` key to bypass RLS and perform inserts.
This key must be kept strictly server-side. Never include it in frontend code or commits.

**Security Note**:
- ✅ `.env` is in `.gitignore` - your credentials won't be committed
- ❌ Never commit API keys to version control
- 🔐 Keep your `service_role` key separate and secure

---

## Step 4: Set Up Storage Buckets

The application needs storage buckets for images and model outputs.

### Option A: Using Supabase Dashboard (Recommended)

1. **Navigate to Storage**:
   - In your Supabase dashboard, click "Storage" in the left sidebar
   - Click "Create a new bucket"

2. **Create route-images bucket**:
   - **Name**: `route-images`
   - **Public bucket**: ✅ Enable (allows public read access to images)
   - **File size limit**: 50 MB (or adjust as needed)
   - **Allowed MIME types**: `image/jpeg, image/png, image/webp`
   - Click "Create bucket"

3. **Create model-outputs bucket** (optional for future use):
   - **Name**: `model-outputs`
   - **Public bucket**: ❌ Disable (keep model outputs private)
   - Click "Create bucket"

### Option B: Using SQL (Alternative)

You can also create buckets via SQL in the SQL Editor:

```sql
-- Create route-images bucket (public)
INSERT INTO storage.buckets (id, name, public)
VALUES ('route-images', 'route-images', true);

-- Create model-outputs bucket (private)
INSERT INTO storage.buckets (id, name, public)
VALUES ('model-outputs', 'model-outputs', false);
```

---

## Step 5: Set Up Database Tables

The application uses PostgreSQL tables for storing route data and analysis results.

### Create the routes table

The canonical migration file lives at `migrations/sql/001_create_routes_table.sql`.

1. **Navigate to SQL Editor**:
   - In your Supabase dashboard, click "SQL Editor" in the left sidebar
   - Click "New query"

2. **Paste and run** the contents of `migrations/sql/001_create_routes_table.sql`.
   The script is idempotent — safe to re-run.

3. **Verify the table was created**:
   - Navigate to "Table Editor" in the left sidebar
   - You should see the `routes` table listed

4. **Run the verifier script** (optional):

   ```bash
   python scripts/migrations/create_routes_table.py
   ```

   Expected output: `VERIFICATION PASSED — routes table is correctly configured.`

5. **Verify the `updated_at` trigger exists** (SQL Editor):

   ```sql
   SELECT tgname, tgrelid::regclass
   FROM pg_trigger
   WHERE tgname = 'set_routes_updated_at';
   ```

   If this query returns one row, the moddatetime trigger was created successfully.

### Table Schema Reference

| Column | Type | Default | Constraints | Description |
|--------|------|---------|-------------|-------------|
| `id` | UUID | `gen_random_uuid()` | PRIMARY KEY | Unique route identifier |
| `image_url` | TEXT | required | `char_length <= 2048` | Public URL of route image |
| `wall_angle` | FLOAT | NULL | `BETWEEN -90 AND 90` | Wall angle in degrees |
| `status` | VARCHAR(20) | `'pending'` | `IN ('pending','processing','done','failed')` | Processing status |
| `created_at` | TIMESTAMPTZ | `NOW()` | NOT NULL | Creation timestamp |
| `updated_at` | TIMESTAMPTZ | `NOW()` | NOT NULL, auto-updated | Last update timestamp |

---

## Step 6: Test Your Connection

Verify your Supabase connection works:

```bash
python test_supabase_connection.py
```

Expected output:

```text
============================================================
Testing Supabase Connection
============================================================

[1/2] Testing Supabase connection...
[OK] Successfully connected to Supabase!
     Project URL: https://xxxxxxxxxxxxx.supabase.co/

[2/2] Testing Supabase Storage...

[INFO] Checking storage buckets directly...
       (anon key can't list all buckets - this is normal)
       [OK] route-images - accessible
       [OK] model-outputs - accessible

[OK] Found 2 bucket(s): route-images, model-outputs

============================================================
[SUCCESS] All tests passed! Your Supabase setup is working.
============================================================
```

---

## Troubleshooting

### Error: "SUPABASE_URL environment variable is required"

**Cause**: Environment variables not loaded properly.

**Solution**:
1. Verify `.env` file exists in project root
2. Check variable names start with `BA_` prefix
3. Restart your development server

### Error: "Failed to create Supabase client"

**Cause**: Invalid credentials or network issues.

**Solution**:
1. Verify your Supabase project URL is correct
2. Check your API key is the `service_role` key (JWT token starting with `eyJ...`)
3. NOT the `anon` or `publishable` key — the backend requires `service_role`
4. Ensure your Supabase project is active (not paused)
5. Check your internet connection

### Storage buckets not detected

**Cause**: The backend is configured with the `anon` (publishable) key instead of the `service_role` key. The `anon` key cannot list all buckets.

**Solution**:
- Ensure `BA_SUPABASE_KEY` is set to the **service_role** key (JWT starting with `eyJ...`), not the `anon` or publishable key
- The `service_role` key lets the backend list and access all buckets without RLS restrictions
- Note: the `anon` key is for client-side access only — it cannot list buckets, which is expected behaviour for anonymous users but incorrect for the backend
- Verify the environment variable is set to the `service_role` key when running the test script

### Error: "Failed to upload file to bucket"

**Possible causes**:
- **File too large**: Check bucket size limits
- **Invalid MIME type**: Verify allowed file types
- **Permission denied**: Ensure bucket is public or use proper RLS policies

**Solution**:
1. Check file size and type
2. Verify bucket permissions in Supabase dashboard
3. Review Supabase Storage logs for details

---

## Security Best Practices

### API Key Management

- ✅ Use `anon` key for client-side operations
- ✅ Use `service_role` key only for trusted server operations
- ✅ Rotate keys periodically in production
- ❌ Never expose `service_role` key in client code
- ❌ Never commit keys to version control

### Row-Level Security (RLS)

When you add database tables (Milestone 9), enable RLS:

```sql
-- Enable RLS on routes table
ALTER TABLE routes ENABLE ROW LEVEL SECURITY;

-- Allow public read access
CREATE POLICY "Public routes are viewable by everyone"
ON routes FOR SELECT
USING (true);

-- Allow authenticated users to insert (no ownership column; auth presence check only)
CREATE POLICY "Users can create their own routes"
ON routes FOR INSERT
WITH CHECK (auth.uid() IS NOT NULL);
```

### Storage Policies

For storage buckets, set up appropriate policies:

```sql
-- Allow public read access to route-images
CREATE POLICY "Public images are viewable by everyone"
ON storage.objects FOR SELECT
USING (bucket_id = 'route-images');

-- Allow authenticated users to upload
CREATE POLICY "Users can upload route images"
ON storage.objects FOR INSERT
WITH CHECK (
    bucket_id = 'route-images'
    AND auth.role() = 'authenticated'
);
```

---

## Usage Examples

### Upload an Image

```python
from src.database import upload_to_storage

# Upload route image
with open("route_photo.jpg", "rb") as f:
    image_data = f.read()

url = upload_to_storage(
    bucket="route-images",
    file_path="2024/01/route_12345.jpg",
    file_data=image_data,
    content_type="image/jpeg"
)

print(f"Image uploaded to: {url}")
```

### Get Image URL

```python
from src.database import get_storage_url

url = get_storage_url(
    bucket="route-images",
    file_path="2024/01/route_12345.jpg"
)

print(f"Image URL: {url}")
```

### Delete an Image

```python
from src.database import delete_from_storage

delete_from_storage(
    bucket="route-images",
    file_path="2024/01/route_12345.jpg"
)

print("Image deleted successfully")
```

### List Files

```python
from src.database import list_storage_files

files = list_storage_files(
    bucket="route-images",
    path="2024/01/"  # Optional: filter by path prefix
)

for file in files:
    print(f"- {file['name']} ({file['metadata']['size']} bytes)")
```

---

## Preventing Project Pauses (Free Tier)

Supabase pauses free projects after ~1 week of inactivity. To prevent this, send a
periodic request to the health endpoint so the database stays active.

### Option A: External uptime monitor (recommended)

Configure UptimeRobot, Better Uptime, or cron-job.org to hit:

```text
GET https://your-backend.com/api/v1/health/db
```

every 5–6 days. No credentials required — this endpoint returns HTTP 200 when the
database is reachable.

### Option B: GitHub Actions scheduled workflow

Use `scripts/ping_supabase.py` in a scheduled workflow:

```yaml
on:
  schedule:
    - cron: '0 12 * * 1'   # Every Monday at noon UTC

jobs:
  keep-alive:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: python scripts/ping_supabase.py --url ${{ secrets.BACKEND_URL }}
```

Add `BACKEND_URL` as a GitHub Actions secret (Settings → Secrets → Actions).

### Option C: Local cron job

```cron
0 12 * * 1 python /path/to/scripts/ping_supabase.py --url https://your-backend.com
```

---

## Resources

- [Supabase Documentation](https://supabase.com/docs)
- [Supabase Python Client](https://supabase.com/docs/reference/python/introduction)
- [Storage Documentation](https://supabase.com/docs/guides/storage)
- [Row Level Security](https://supabase.com/docs/guides/auth/row-level-security)

---

**Last Updated**: 2026-03-15
**Related Documentation**: [FASTAPI_ROLE.md](FASTAPI_ROLE.md), [DESIGN.md](DESIGN.md)
