-- Migration 001: Create routes table
-- Idempotent — safe to re-run on an existing database.
-- Compatible with Supabase (PostgreSQL 15+).

-- Enable moddatetime extension (idempotent)
CREATE EXTENSION IF NOT EXISTS moddatetime;

-- Routes table
CREATE TABLE IF NOT EXISTS routes (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    image_url   TEXT        NOT NULL
                            CHECK (char_length(image_url) <= 2048),
    -- Range matches the API layer (-90/90). The graph builder uses -15/90 for
    -- biomechanical reasons; values below -15 are stored but rejected at build time.
    wall_angle  FLOAT       CHECK (wall_angle IS NULL OR wall_angle BETWEEN -90 AND 90),
    status      VARCHAR(20) NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Auto-update updated_at on every UPDATE
CREATE OR REPLACE TRIGGER set_routes_updated_at
    BEFORE UPDATE ON routes
    FOR EACH ROW
    EXECUTE FUNCTION moddatetime(updated_at);

-- Index for chronological listing (GET /api/v1/routes pagination)
CREATE INDEX IF NOT EXISTS idx_routes_created_at
    ON routes (created_at DESC);

-- Partial index for background job polling (pending/processing rows only)
CREATE INDEX IF NOT EXISTS idx_routes_status_pending
    ON routes (status)
    WHERE status IN ('pending', 'processing');

-- ── Row Level Security ───────────────────────────────────────────────────────
-- Enable RLS (deny-all by default for anonymous callers).
ALTER TABLE routes ENABLE ROW LEVEL SECURITY;

-- Service role bypasses RLS automatically in Supabase; the policies below
-- grant read access to the PostgREST anon role and restrict writes to the
-- service role (backend only).

-- Allow public read access (e.g. frontend status polling)
CREATE POLICY IF NOT EXISTS "routes_select_public"
    ON routes FOR SELECT
    TO PUBLIC
    USING (true);

-- Restrict INSERT to service role (backend API writes only)
CREATE POLICY IF NOT EXISTS "routes_insert_service"
    ON routes FOR INSERT
    TO service_role
    WITH CHECK (true);

-- Restrict UPDATE to service role (background ML pipeline updates status)
CREATE POLICY IF NOT EXISTS "routes_update_service"
    ON routes FOR UPDATE
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Restrict DELETE to service role
CREATE POLICY IF NOT EXISTS "routes_delete_service"
    ON routes FOR DELETE
    TO service_role
    USING (true);
