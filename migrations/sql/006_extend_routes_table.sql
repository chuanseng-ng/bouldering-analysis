-- Migration: Extend routes table with start/finish hold ID arrays
-- PR-10.2: Adds start_hold_ids and finish_hold_ids columns to persist
--          user-selected constraints from the frontend hold annotation screen.
--
-- Re-run safety: all statements use IF NOT EXISTS / conditional DDL.
-- No data loss on re-run.

-- Add INTEGER[] columns to store user-selected constraint hold IDs.
-- NULL = user has not yet set constraints for this route.
ALTER TABLE routes
    ADD COLUMN IF NOT EXISTS start_hold_ids  INTEGER[],
    ADD COLUMN IF NOT EXISTS finish_hold_ids INTEGER[];

-- Partial index for efficiently finding routes that have been annotated
-- with start/finish constraints (used by history + polling queries).
CREATE INDEX IF NOT EXISTS idx_routes_constraints_set
    ON routes (id, updated_at DESC)
    WHERE start_hold_ids IS NOT NULL AND finish_hold_ids IS NOT NULL;
