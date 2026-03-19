-- Migration 004: Create predictions table
-- Idempotent - safe to re-run on an existing database.
-- Compatible with Supabase (PostgreSQL 15+).
-- Depends on: 001_create_routes_table.sql (routes table must exist first).

-- Predictions table - persists grade prediction results from the grading pipeline.
-- Predictions are append-only immutable history: every analysis run inserts a new
-- row.  Old rows are never deleted or overwritten.  Multiple predictions per route
-- are explicitly allowed (e.g. different model versions, heuristic vs ML comparison).
--
-- No UNIQUE on route_id: use the compound index below for sorted per-route listing.
-- No updated_at / trigger: append-only design makes row mutation unnecessary.
--
-- explanation contents are validated at the application layer by the ExplanationResult
-- Pydantic model (src/explanation/types.py); no CHECK constraint is applied to the
-- JSONB column itself.
CREATE TABLE IF NOT EXISTS predictions (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    route_id         UUID        NOT NULL REFERENCES routes(id) ON DELETE CASCADE,
    estimator_type   VARCHAR(20) NOT NULL CHECK (estimator_type IN ('heuristic', 'ml')),
    grade            VARCHAR(10) NOT NULL CHECK (grade IN (
                         'V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                         'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17'
                     )),
    grade_index      INT         NOT NULL CHECK (grade_index BETWEEN 0 AND 17),
    confidence       FLOAT       NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    difficulty_score FLOAT       NOT NULL CHECK (difficulty_score BETWEEN 0 AND 1),
    -- NULL for all current estimators (reserved for future calibrated uncertainty output).
    uncertainty      FLOAT                CHECK (uncertainty BETWEEN 0 AND 1),
    explanation      JSONB,
    -- NULL for heuristic estimator (no model artifact); set to v<YYYYMMDD_HHMMSS> for ML.
    model_version    VARCHAR(20),
    predicted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Compound index for sorted per-route listing (most common access pattern).
-- No UNIQUE — multiple predictions per route are allowed.
CREATE INDEX IF NOT EXISTS idx_predictions_route_id_predicted_at
    ON predictions (route_id, predicted_at DESC);

-- Row Level Security
-- Enable RLS (deny-all by default for anonymous callers).
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Allow public read access (e.g. frontend grade display)
DROP POLICY IF EXISTS predictions_select_public ON predictions;
CREATE POLICY predictions_select_public
    ON predictions FOR SELECT
    TO PUBLIC
    USING (true);

-- Restrict INSERT to service role (backend ML pipeline writes only)
DROP POLICY IF EXISTS predictions_insert_service ON predictions;
CREATE POLICY predictions_insert_service
    ON predictions FOR INSERT
    TO service_role
    WITH CHECK (true);

-- Restrict UPDATE to service role (operational corrections only — predictions are
-- append-only by contract; the application never issues UPDATE statements against
-- this table. This policy exists for admin data hygiene, consistent with holds
-- and features tables which follow the same four-policy pattern.)
DROP POLICY IF EXISTS predictions_update_service ON predictions;
CREATE POLICY predictions_update_service
    ON predictions FOR UPDATE
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Restrict DELETE to service role
DROP POLICY IF EXISTS predictions_delete_service ON predictions;
CREATE POLICY predictions_delete_service
    ON predictions FOR DELETE
    TO service_role
    USING (true);
