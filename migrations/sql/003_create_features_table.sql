-- Migration 003: Create features table
-- Idempotent - safe to re-run on an existing database.
-- Compatible with Supabase (PostgreSQL 15+).
-- Depends on: 001_create_routes_table.sql (routes table must exist first).

-- Enable moddatetime extension (idempotent - also enabled in 001)
CREATE EXTENSION IF NOT EXISTS moddatetime;

-- Features table - persists RouteFeatures instances from the feature extraction pipeline.
-- Features are write-once: re-running feature extraction deletes the row for the route
-- and reinserts a fresh one (see re-run contract in plans/MIGRATION_PLAN.md).
-- No updated_at / trigger: write-once design makes row mutation unnecessary.
--
-- UNIQUE (route_id) enforces one feature vector per route and doubles as the covering
-- index for route-scoped lookups (SELECT * FROM features WHERE route_id = $1), so no
-- separate idx_features_route_id index is needed.
--
-- No CHECK constraints on feature_vector: JSONB contents are validated at the
-- application layer by the RouteFeatures Pydantic model (src/features/assembler.py).
CREATE TABLE IF NOT EXISTS features (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    route_id       UUID        NOT NULL UNIQUE REFERENCES routes(id) ON DELETE CASCADE,
    feature_vector JSONB       NOT NULL,
    extracted_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Row Level Security
-- Enable RLS (deny-all by default for anonymous callers).
ALTER TABLE features ENABLE ROW LEVEL SECURITY;

-- Allow public read access (e.g. frontend feature display)
DROP POLICY IF EXISTS features_select_public ON features;
CREATE POLICY features_select_public
    ON features FOR SELECT
    TO PUBLIC
    USING (true);

-- Restrict INSERT to service role (backend ML pipeline writes only)
DROP POLICY IF EXISTS features_insert_service ON features;
CREATE POLICY features_insert_service
    ON features FOR INSERT
    TO service_role
    WITH CHECK (true);

-- Restrict UPDATE to service role
DROP POLICY IF EXISTS features_update_service ON features;
CREATE POLICY features_update_service
    ON features FOR UPDATE
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Restrict DELETE to service role (used during re-run cycle: delete + reinsert)
DROP POLICY IF EXISTS features_delete_service ON features;
CREATE POLICY features_delete_service
    ON features FOR DELETE
    TO service_role
    USING (true);
