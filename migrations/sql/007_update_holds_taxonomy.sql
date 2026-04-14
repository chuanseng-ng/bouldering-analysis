-- Migration 007: Align holds table with 7-class canonical taxonomy.
-- Idempotent — safe to re-run.
--
-- Changes:
--   1. Add prob_pocket and prob_foothold (new classifier-output columns).
--   2. Drop prob_edges (aliased → crimp; no longer written by the classifier).
--   3. Drop prob_volume (volume is a detection_class only, not a classifier output).
--   4. Widen hold_type CHECK to include new canonical classes while keeping
--      volume as a valid passthrough value from the detector.

-- Add new classifier-output probability columns
ALTER TABLE holds ADD COLUMN IF NOT EXISTS prob_pocket
    FLOAT NOT NULL DEFAULT 0.0 CHECK (prob_pocket BETWEEN 0 AND 1);
ALTER TABLE holds ADD COLUMN IF NOT EXISTS prob_foothold
    FLOAT NOT NULL DEFAULT 0.0 CHECK (prob_foothold BETWEEN 0 AND 1);

-- Drop obsolete probability columns
ALTER TABLE holds DROP COLUMN IF EXISTS prob_edges;
ALTER TABLE holds DROP COLUMN IF EXISTS prob_volume;

-- Widen hold_type CHECK constraint to include new canonical classes;
-- volume is kept for detector-flagged volumes (hold_type passthrough).
ALTER TABLE holds DROP CONSTRAINT IF EXISTS holds_hold_type_check;
ALTER TABLE holds ADD CONSTRAINT holds_hold_type_check
    CHECK (hold_type IN ('jug', 'crimp', 'sloper', 'pinch', 'pocket', 'foothold', 'volume', 'unknown'));
