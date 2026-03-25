-- Migration 007: Update holds table for 8-class hold taxonomy
-- Idempotent - safe to re-run on an existing database.
-- Compatible with Supabase (PostgreSQL 15+).
-- Depends on: 002_create_holds_table.sql (holds table must exist first).
--
-- Changes:
--   1. Adds prob_pocket, prob_edges, prob_foothold columns (8-class classifier output).
--   2. Drops legacy prob_volume column (removed from 8-class taxonomy).
--   3. Updates detection_class CHECK to reflect the 8 YOLO class names.
--   4. Updates hold_type CHECK to reflect the 8 normalised hold-type labels.
--   5. Expands detection_class column width to VARCHAR(20) to fit 'Hand-holds'.

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

-- ── Step 3: Update detection_class column and CHECK constraint ───────────────
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

-- ── Step 4: Update hold_type CHECK constraint ────────────────────────────────

ALTER TABLE holds DROP CONSTRAINT IF EXISTS holds_hold_type_check;

ALTER TABLE holds
    ADD CONSTRAINT holds_hold_type_check
        CHECK (hold_type IN (
            'jug', 'crimp', 'sloper', 'pinch',
            'pocket', 'edges', 'foothold', 'unknown'
        ));
