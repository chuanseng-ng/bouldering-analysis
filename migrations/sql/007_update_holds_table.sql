-- Migration 007: Update holds table for 8-class hold taxonomy
-- Idempotent - safe to re-run on an existing database.
-- Compatible with Supabase (PostgreSQL 15+).
-- Depends on: 002_create_holds_table.sql (holds table must exist first).
--
-- Changes:
--   1. Adds prob_pocket, prob_edges, prob_foothold columns (8-class classifier output).
--   2. Drops legacy prob_volume column (removed from 8-class taxonomy).
--   3. Normalises legacy taxonomy values in existing rows before adding new CHECKs.
--   4. Updates detection_class CHECK to reflect the 8 YOLO class names.
--   5. Updates hold_type CHECK to reflect the 8 normalised hold-type labels.
--   6. Expands detection_class column width to VARCHAR(20) to fit 'Hand-holds'.
--
-- Probability-sum invariant note:
--   The probability columns (prob_jug, prob_crimp, prob_sloper, prob_pinch,
--   prob_pocket, prob_edges, prob_foothold, prob_unknown) are intentionally NOT
--   constrained to sum to 1.0 at the database level.  Floating-point round-trips
--   through the classifier can produce sums of 0.9999... or 1.0001..., which
--   would cause spurious CHECK failures.  The sum invariant is enforced in the
--   application layer by the ClassifiedHold Pydantic validator (src/graph/types.py)
--   before any row is written.

-- ── Step 1: Add new probability columns ─────────────────────────────────────

ALTER TABLE holds
    ADD COLUMN IF NOT EXISTS prob_pocket   FLOAT NOT NULL DEFAULT 0.0
        CHECK (prob_pocket   BETWEEN 0 AND 1);

ALTER TABLE holds
    ADD COLUMN IF NOT EXISTS prob_edges    FLOAT NOT NULL DEFAULT 0.0
        CHECK (prob_edges    BETWEEN 0 AND 1);

ALTER TABLE holds
    ADD COLUMN IF NOT EXISTS prob_foothold FLOAT NOT NULL DEFAULT 0.0
        CHECK (prob_foothold BETWEEN 0 AND 1);

-- ── Step 2: Drop legacy prob_volume column ───────────────────────────────────

ALTER TABLE holds DROP COLUMN IF EXISTS prob_volume;

-- ── Step 3: Normalise legacy taxonomy values before adding new CHECKs ────────
-- Rows written by the old 2-class detection model store detection_class in
-- ('hold', 'volume') and hold_type may include 'volume'.  Map them to the
-- closest canonical value so the new CHECK constraints do not reject them.

-- 'hold' was the generic detection class → map to 'Jug' (most common hold type).
-- 'volume' was the old large-feature class → map to 'Hand-holds' (unknown type).
UPDATE holds SET detection_class = 'Jug'        WHERE detection_class = 'hold';
UPDATE holds SET detection_class = 'Hand-holds' WHERE detection_class = 'volume';

-- hold_type 'volume' no longer exists in the 8-class taxonomy → map to 'unknown'.
UPDATE holds SET hold_type = 'unknown' WHERE hold_type = 'volume';

-- ── Step 4: Update detection_class column and CHECK constraint ───────────────
-- PostgreSQL auto-names inline CHECK constraints as <table>_<column>_check.

ALTER TABLE holds DROP CONSTRAINT IF EXISTS holds_detection_class_check;

-- Widen column to VARCHAR(20) so 'Hand-holds' (10 chars) fits comfortably.
ALTER TABLE holds ALTER COLUMN detection_class TYPE VARCHAR(20);

ALTER TABLE holds
    ADD CONSTRAINT holds_detection_class_check
        CHECK (detection_class IN (
            'Crimp', 'Edges', 'Foothold', 'Hand-holds',
            'Jug', 'Pinch', 'Pocket', 'Sloper'
        ));

-- ── Step 5: Update hold_type CHECK constraint ────────────────────────────────

ALTER TABLE holds DROP CONSTRAINT IF EXISTS holds_hold_type_check;

ALTER TABLE holds
    ADD CONSTRAINT holds_hold_type_check
        CHECK (hold_type IN (
            'jug', 'crimp', 'sloper', 'pinch',
            'pocket', 'edges', 'foothold', 'unknown'
        ));

-- ── Step 6: Verification (informational — confirm schema state after migration)
-- These SELECT statements return one row each; a NULL or false result indicates
-- the migration did not apply correctly.

-- New probability columns present
SELECT
    EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'holds' AND column_name = 'prob_pocket'
    ) AS prob_pocket_exists,
    EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'holds' AND column_name = 'prob_edges'
    ) AS prob_edges_exists,
    EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'holds' AND column_name = 'prob_foothold'
    ) AS prob_foothold_exists,
    NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'holds' AND column_name = 'prob_volume'
    ) AS prob_volume_removed;

-- Updated CHECK constraints present
SELECT
    COUNT(*) = 2 AS both_checks_exist
FROM pg_constraint
WHERE conrelid = 'holds'::regclass
  AND contype = 'c'
  AND conname IN ('holds_detection_class_check', 'holds_hold_type_check');

-- RLS policies still present
SELECT
    COUNT(*) AS rls_policy_count
FROM pg_policies
WHERE tablename = 'holds';
