#!/usr/bin/env python3
"""Verifier script for the features table migration (003_create_features_table.sql).

Checks that the features table, its columns, and RLS policies are all present in the
target Supabase database.  Does NOT apply the migration — run the SQL file directly in
the Supabase SQL Editor for that.

Usage:
    python scripts/migrations/create_features_table.py           # verify (default)
    python scripts/migrations/create_features_table.py --dry-run # print SQL, no DB call

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
    Features are write-once.  To re-run feature extraction for a route, the caller
    must ``DELETE FROM features WHERE route_id = $1`` before reinserting.
    The ``UNIQUE (route_id)`` constraint enforces one feature vector per route and
    doubles as the covering index for route-scoped reads, so no separate
    ``idx_features_route_id`` index is created.
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

_SQL_FILE = _PROJECT_ROOT / "migrations" / "sql" / "003_create_features_table.sql"

_CONFIG = TableVerificationConfig(
    table_name="features",
    expected_columns=(
        "id",
        "route_id",
        "feature_vector",
        "extracted_at",
    ),
    # Features are write-once — no updated_at column, no moddatetime trigger.
    trigger_name=None,
    # No column-level CHECK constraints: JSONB contents are validated at the
    # application layer by RouteFeatures Pydantic model.
    expected_check_constraints=frozenset(),
    expected_rls_policies=frozenset(
        {
            "features_select_public",
            "features_insert_service",
            "features_update_service",
            "features_delete_service",
        }
    ),
)


# ---------------------------------------------------------------------------
# Public verifier
# ---------------------------------------------------------------------------


def verify_features_table(client: SupabaseClientLike) -> VerificationResult:
    """Verify the features table schema in the connected Supabase database.

    Checks performed:
    1. Table ``features`` exists.
    2. All 4 expected columns are present.
    3. All 4 expected RLS policies are present.

    (No trigger check: features are write-once, so no ``updated_at`` trigger
    is configured.  No CHECK constraint check: JSONB contents are validated
    at the application layer.)

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
        description="Verify that the features table migration has been applied.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/migrations/create_features_table.py
  python scripts/migrations/create_features_table.py --dry-run
""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the migration SQL to stdout without connecting to the database.",
    )
    return parser


def main() -> None:
    """Entry point for the features table verifier script.

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
    logger.info("Verifying features table migration")
    logger.info("=" * 60)

    try:
        client = get_supabase_client()
    except SupabaseClientError as exc:
        logger.error("Could not connect to Supabase: %s", exc)
        sys.exit(1)

    result = verify_features_table(client)

    if result.success:
        logger.info("=" * 60)
        logger.info("VERIFICATION PASSED — features table is correctly configured.")
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
    setup_migration_logging("verify_features_table.log")
    main()
