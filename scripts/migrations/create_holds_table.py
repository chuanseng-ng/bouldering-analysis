#!/usr/bin/env python3
"""Verifier script for the holds table migration (002_create_holds_table.sql).

Checks that the holds table, its columns, CHECK constraints, and RLS policies
are all present in the target Supabase database.  Does NOT apply the migration
— run the SQL file directly in the Supabase SQL Editor for that.

Usage:
    python scripts/migrations/create_holds_table.py           # verify (default)
    python scripts/migrations/create_holds_table.py --dry-run # print SQL, no DB call

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

Re-run contract:
    Holds are write-once.  To re-run hold detection for a route, the caller
    must ``DELETE FROM holds WHERE route_id = $1`` before reinserting.
    The ``UNIQUE (route_id, hold_id)`` constraint enforces ordering uniqueness
    within a route and doubles as the covering index for route-scoped reads,
    so no separate ``idx_holds_route_id`` index is created.
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

_SQL_FILE = _PROJECT_ROOT / "migrations" / "sql" / "002_create_holds_table.sql"

_CONFIG = TableVerificationConfig(
    table_name="holds",
    expected_columns=(
        "id",
        "route_id",
        "hold_id",
        "x_center",
        "y_center",
        "width",
        "height",
        "detection_class",
        "detection_confidence",
        "hold_type",
        "type_confidence",
        "prob_jug",
        "prob_crimp",
        "prob_sloper",
        "prob_pinch",
        "prob_pocket",
        "prob_foothold",
        "prob_unknown",
        "created_at",
    ),
    # Holds are write-once — no updated_at column, no moddatetime trigger.
    trigger_name=None,
    expected_check_constraints=frozenset(
        {
            "holds_hold_id_check",
            "holds_x_center_check",
            "holds_y_center_check",
            "holds_width_check",
            "holds_height_check",
            "holds_detection_class_check",
            "holds_detection_confidence_check",
            "holds_hold_type_check",
            "holds_type_confidence_check",
            "holds_prob_jug_check",
            "holds_prob_crimp_check",
            "holds_prob_sloper_check",
            "holds_prob_pinch_check",
            "holds_prob_pocket_check",
            "holds_prob_foothold_check",
            "holds_prob_unknown_check",
        }
    ),
    expected_rls_policies=frozenset(
        {
            "holds_select_public",
            "holds_insert_service",
            "holds_update_service",
            "holds_delete_service",
        }
    ),
)


# ---------------------------------------------------------------------------
# Public verifier
# ---------------------------------------------------------------------------


def verify_holds_table(client: SupabaseClientLike) -> VerificationResult:
    """Verify the holds table schema in the connected Supabase database.

    Checks performed:
    1. Table ``holds`` exists.
    2. All 19 expected columns are present.
    3. All 16 expected CHECK constraints are present.
    4. All 4 expected RLS policies are present.

    (No trigger check: holds are write-once, so no ``updated_at`` trigger
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
        description="Verify that the holds table migration has been applied.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/migrations/create_holds_table.py
  python scripts/migrations/create_holds_table.py --dry-run
""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the migration SQL to stdout without connecting to the database.",
    )
    return parser


def main() -> None:
    """Entry point for the holds table verifier script.

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
    logger.info("Verifying holds table migration")
    logger.info("=" * 60)

    try:
        client = get_supabase_client()
    except SupabaseClientError as exc:
        logger.error("Could not connect to Supabase: %s", exc)
        sys.exit(1)

    result = verify_holds_table(client)

    if result.success:
        logger.info("=" * 60)
        logger.info("VERIFICATION PASSED — holds table is correctly configured.")
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
    setup_migration_logging("verify_holds_table.log")
    main()
