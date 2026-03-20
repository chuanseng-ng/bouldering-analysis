#!/usr/bin/env python3
"""Verifier script for the predictions table migration (004_create_predictions_table.sql).

Checks that the predictions table, its columns, CHECK constraints, and RLS policies
are all present in the target Supabase database.  Does NOT apply the migration — run
the SQL file directly in the Supabase SQL Editor for that.

Usage:
    python scripts/migrations/create_predictions_table.py           # verify (default)
    python scripts/migrations/create_predictions_table.py --dry-run # print SQL, no DB call

Exit codes:
    0  — verification passed (or dry-run completed)
    1  — verification failed or unexpected error

Supabase PostgREST note:
    Verification queries target ``information_schema`` views and
    ``pg_catalog.pg_policies``.  By default Supabase's PostgREST only
    exposes the ``public`` schema.  To use the live-database verify mode
    you must expose ``information_schema`` and ``pg_catalog`` in your
    Supabase project settings (API → Extra search paths) or run the SQL
    queries directly via the Supabase SQL Editor.  The dry-run mode and all
    unit tests work without this configuration.

Write contract:
    Predictions are append-only immutable history.  Each analysis run inserts
    a new row; old rows are never deleted or overwritten.  Multiple predictions
    per route are explicitly allowed (model versioning / heuristic vs ML comparison).
    There is no re-run contract analogous to features/holds — callers simply INSERT
    a new prediction record on every analysis run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(
    0, str(_PROJECT_ROOT)
)  # ensure src/ is importable when run from any directory

from scripts.migrations._migration_utils import (  # noqa: E402  # pylint: disable=wrong-import-position
    SupabaseClientLike,
    TableVerificationConfig,
    VerificationResult,
    setup_migration_logging,
    verify_table,
)
from src.database.supabase_client import (  # noqa: E402  # pylint: disable=wrong-import-position
    SupabaseClientError,
    get_supabase_client,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SQL_FILE = _PROJECT_ROOT / "migrations" / "sql" / "004_create_predictions_table.sql"

_CONFIG = TableVerificationConfig(
    table_name="predictions",
    expected_columns=(
        "id",
        "route_id",
        "estimator_type",
        "grade",
        "grade_index",
        "confidence",
        "difficulty_score",
        "uncertainty",
        "explanation",
        "model_version",
        "predicted_at",
    ),
    # Predictions are append-only — no updated_at column, no moddatetime trigger.
    trigger_name=None,
    expected_check_constraints=frozenset(
        {
            "predictions_estimator_type_check",
            "predictions_grade_check",
            "predictions_grade_index_check",
            "predictions_confidence_check",
            "predictions_difficulty_score_check",
            "predictions_uncertainty_check",
        }
    ),
    expected_rls_policies=frozenset(
        {
            "predictions_select_public",
            "predictions_insert_service",
            "predictions_update_service",
            "predictions_delete_service",
        }
    ),
)


# ---------------------------------------------------------------------------
# Public verifier
# ---------------------------------------------------------------------------


def verify_predictions_table(client: SupabaseClientLike) -> VerificationResult:
    """Verify the predictions table schema in the connected Supabase database.

    Checks performed:
    1. Table ``predictions`` exists.
    2. All 11 expected columns are present.
    3. All 6 expected CHECK constraints are present.
    4. All 4 expected RLS policies are present.

    (No trigger check: predictions are append-only, so no ``updated_at`` trigger
    is configured.)

    Args:
        client: Supabase client instance (from ``get_supabase_client()``).

    Returns:
        :class:`VerificationResult` with ``success=True`` if all checks pass,
        or ``success=False`` with ``errors`` populated otherwise.
    """
    return verify_table(client, _CONFIG)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Verify that the predictions table migration has been applied.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/migrations/create_predictions_table.py
  python scripts/migrations/create_predictions_table.py --dry-run
""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the migration SQL to stdout without connecting to the database.",
    )
    return parser


def main() -> None:
    """Entry point for the predictions table verifier script.

    Exits with code 0 on success, 1 on failure.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.dry_run:
        # Dry-run: just print the SQL
        if _SQL_FILE.exists():
            print(f"-- SQL file: {_SQL_FILE}\n")
            print(_SQL_FILE.read_text(encoding="utf-8"))
        else:
            logger.error("SQL file not found at %s", _SQL_FILE)
            sys.exit(1)
        sys.exit(0)

    # Verify mode (default)
    logger.info("=" * 60)
    logger.info("Verifying predictions table migration")
    logger.info("=" * 60)

    try:
        client = get_supabase_client()
    except SupabaseClientError as exc:
        logger.error("Could not connect to Supabase: %s", exc)
        sys.exit(1)

    result = verify_predictions_table(client)

    if result.success:
        logger.info("=" * 60)
        logger.info("VERIFICATION PASSED — predictions table is correctly configured.")
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error("=" * 60)
        logger.error("VERIFICATION FAILED — %d issue(s) found:", len(result.errors))
        for i, err in enumerate(result.errors, 1):
            logger.error("  %d. %s", i, err)
        logger.error("=" * 60)
        logger.error("Apply the migration via the Supabase SQL Editor: %s", _SQL_FILE)
        sys.exit(1)


if __name__ == "__main__":
    setup_migration_logging("verify_predictions_table.log")
    main()
