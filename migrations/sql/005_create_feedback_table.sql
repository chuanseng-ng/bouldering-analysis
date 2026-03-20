-- Migration 005: Create feedback table
-- Idempotent - safe to re-run on an existing database.
-- Compatible with Supabase (PostgreSQL 15+).
-- Depends on: 001_create_routes_table.sql (routes table must exist first).

-- Feedback table - stores user-submitted grade assessments and comments.
-- Feedback is append-only immutable history: every user submission inserts a new
-- row.  Old rows are never deleted or overwritten.  Multiple feedback entries per
-- route are explicitly allowed (anonymous public submission).
--
-- No UNIQUE on route_id: multiple users can submit feedback for the same route.
-- No updated_at / trigger: append-only design makes row mutation unnecessary.
--
-- user_grade is nullable with an IS NULL OR IN check — the user may omit their
-- grade estimate while still submitting accuracy or comment feedback.
-- is_accurate and comments are plain nullable columns — no CHECK constraint needed.
CREATE TABLE IF NOT EXISTS feedback (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    route_id    UUID        NOT NULL REFERENCES routes(id) ON DELETE CASCADE,
    user_grade  VARCHAR(10) CHECK (user_grade IS NULL OR user_grade IN (
                    'V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17'
                )),
    is_accurate BOOLEAN,
    comments    TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Compound index for sorted per-route listing.
-- Explicit CREATE INDEX (not UNIQUE) — multiple feedback entries per route allowed.
CREATE INDEX IF NOT EXISTS idx_feedback_route_id_created_at
    ON feedback (route_id, created_at DESC);

-- Row Level Security
-- Enable RLS (deny-all by default for anonymous callers).
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;

-- Allow public read access (e.g. frontend feedback display)
DROP POLICY IF EXISTS feedback_select_public ON feedback;
CREATE POLICY feedback_select_public
    ON feedback FOR SELECT
    TO PUBLIC
    USING (true);

-- Allow public INSERT — end users submit feedback directly from the frontend
-- without authentication.  This is intentional: feedback is anonymous and the
-- route_id FK prevents orphaned rows.
DROP POLICY IF EXISTS feedback_insert_public ON feedback;
CREATE POLICY feedback_insert_public
    ON feedback FOR INSERT
    TO PUBLIC
    WITH CHECK (true);

-- Restrict UPDATE to service role (admin data hygiene only — feedback is
-- append-only by contract; the application never issues UPDATE statements against
-- this table.)
DROP POLICY IF EXISTS feedback_update_service ON feedback;
CREATE POLICY feedback_update_service
    ON feedback FOR UPDATE
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Restrict DELETE to service role
DROP POLICY IF EXISTS feedback_delete_service ON feedback;
CREATE POLICY feedback_delete_service
    ON feedback FOR DELETE
    TO service_role
    USING (true);
