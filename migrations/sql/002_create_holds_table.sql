-- Migration 002: Create holds table
-- Idempotent - safe to re-run on an existing database.
-- Compatible with Supabase (PostgreSQL 15+).
-- Depends on: 001_create_routes_table.sql (routes table must exist first).

-- Enable moddatetime extension (idempotent - also enabled in 001)
CREATE EXTENSION IF NOT EXISTS moddatetime;

-- Holds table - persists ClassifiedHold instances from the inference pipeline.
-- Holds are write-once: re-running detection deletes all holds for the route
-- and reinserts fresh ones (see re-run contract in plans/MIGRATION_PLAN.md).
-- No updated_at / trigger: write-once design makes row mutation unnecessary.
CREATE TABLE IF NOT EXISTS holds (
    id                   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    route_id             UUID        NOT NULL REFERENCES routes(id) ON DELETE CASCADE,
    hold_id              INT         NOT NULL CHECK (hold_id >= 0),
    x_center             FLOAT       NOT NULL CHECK (x_center BETWEEN 0 AND 1),
    y_center             FLOAT       NOT NULL CHECK (y_center BETWEEN 0 AND 1),
    width                FLOAT       NOT NULL CHECK (width BETWEEN 0 AND 1),
    height               FLOAT       NOT NULL CHECK (height BETWEEN 0 AND 1),
    detection_class      VARCHAR(10) NOT NULL CHECK (detection_class IN ('hold', 'volume')),
    detection_confidence FLOAT       NOT NULL CHECK (detection_confidence BETWEEN 0 AND 1),
    hold_type            VARCHAR(20) NOT NULL CHECK (hold_type IN ('jug', 'crimp', 'sloper', 'pinch', 'pocket', 'foothold', 'volume', 'unknown')),
    type_confidence      FLOAT       NOT NULL CHECK (type_confidence BETWEEN 0 AND 1),
    prob_jug             FLOAT       NOT NULL CHECK (prob_jug BETWEEN 0 AND 1),
    prob_crimp           FLOAT       NOT NULL CHECK (prob_crimp BETWEEN 0 AND 1),
    prob_sloper          FLOAT       NOT NULL CHECK (prob_sloper BETWEEN 0 AND 1),
    prob_pinch           FLOAT       NOT NULL CHECK (prob_pinch BETWEEN 0 AND 1),
    prob_pocket          FLOAT       NOT NULL CHECK (prob_pocket BETWEEN 0 AND 1),
    prob_foothold        FLOAT       NOT NULL CHECK (prob_foothold BETWEEN 0 AND 1),
    prob_unknown         FLOAT       NOT NULL CHECK (prob_unknown BETWEEN 0 AND 1),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Enforce local hold ordering uniqueness within a route.
    -- This UNIQUE constraint also doubles as the covering index for
    -- route-scoped lookups (SELECT * FROM holds WHERE route_id = $1 ORDER BY hold_id),
    -- so no separate idx_holds_route_id is needed.
    UNIQUE (route_id, hold_id)
);

-- Row Level Security
-- Enable RLS (deny-all by default for anonymous callers).
ALTER TABLE holds ENABLE ROW LEVEL SECURITY;

-- Allow public read access (e.g. frontend hold visualisation)
DROP POLICY IF EXISTS holds_select_public ON holds;
CREATE POLICY holds_select_public
    ON holds FOR SELECT
    TO PUBLIC
    USING (true);

-- Restrict INSERT to service role (backend ML pipeline writes only)
DROP POLICY IF EXISTS holds_insert_service ON holds;
CREATE POLICY holds_insert_service
    ON holds FOR INSERT
    TO service_role
    WITH CHECK (true);

-- Restrict UPDATE to service role
DROP POLICY IF EXISTS holds_update_service ON holds;
CREATE POLICY holds_update_service
    ON holds FOR UPDATE
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Restrict DELETE to service role (used during re-run cycle: delete + reinsert)
DROP POLICY IF EXISTS holds_delete_service ON holds;
CREATE POLICY holds_delete_service
    ON holds FOR DELETE
    TO service_role
    USING (true);
