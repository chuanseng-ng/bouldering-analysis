#!/usr/bin/env python3
"""Verifier script for the routes table migration (001_create_routes_table.sql).

Checks that the routes table, its columns, CHECK constraints, and the
moddatetime trigger are all present in the target Supabase database.
Does NOT apply the migration — run the SQL file directly in the Supabase
SQL Editor for that.

Usage:
    python scripts/migrations/create_routes_table.py           # verify (default)
    python scripts/migrations/create_routes_table.py --dry-run # print SQL, no DB call

Exit codes:
    0  — verification passed (or dry-run completed)
    1  — verification failed or unexpected error

Supabase PostgREST note:
    Verification queries target ``information_schema`` views.  By default
    Supabase's PostgREST only exposes the ``public`` schema.  To use the
    live-database verify mode, you must expose ``information_schema`` in your
    Supabase project settings (API → Extra search paths) or run the SQL
    queries directly via the Supabase SQL Editor.  The dry-run mode and all
    unit tests work without this configuration.

wall_angle range note:
    The database stores wall_angle in [-90, 90] to match the API layer
    (``src/routes/routes.py``).  The graph builder (``src/graph/route_graph.py``)
    uses a tighter range of [-15, 90] for biomechanical reasons.  Values
    between -90 and -15.1 are valid at the DB level but will be rejected at
    graph-build time; this is intentional — the DB is the wider authority and
    application logic enforces domain constraints.
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

_SQL_FILE = _PROJECT_ROOT / "migrations" / "sql" / "001_create_routes_table.sql"

_CONFIG = TableVerificationConfig(
    table_name="routes",
    expected_columns=(
        "id",
        "image_url",
        "wall_angle",
        "status",
        "created_at",
        "updated_at",
    ),
    trigger_name="set_routes_updated_at",
    expected_check_constraints=frozenset(
        {
            "routes_image_url_check",
            "routes_wall_angle_check",
            "routes_status_check",
        }
    ),
    expected_rls_policies=frozenset(
        {
            "routes_select_public",
            "routes_insert_service",
            "routes_update_service",
            "routes_delete_service",
        }
    ),
)


# ---------------------------------------------------------------------------
# Public verifier
# ---------------------------------------------------------------------------


def verify_routes_table(client: SupabaseClientLike) -> VerificationResult:
    """Verify the routes table schema in the connected Supabase database.

    Checks performed:
    1. Table ``routes`` exists.
    2. All 6 expected columns are present.
    3. All 3 expected CHECK constraints are present
       (``routes_image_url_check``, ``routes_wall_angle_check``,
       ``routes_status_check``).
    4. The ``set_routes_updated_at`` trigger is present.
    5. All 4 expected RLS policies are present.

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
        description="Verify that the routes table migration has been applied.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/migrations/create_routes_table.py
  python scripts/migrations/create_routes_table.py --dry-run
""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the migration SQL to stdout without connecting to the database.",
    )
    return parser


def main() -> None:
    """Entry point for the routes table verifier script.

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
    logger.info("Verifying routes table migration")
    logger.info("=" * 60)

    try:
        client = get_supabase_client()
    except SupabaseClientError as exc:
        logger.error("Could not connect to Supabase: %s", exc)
        sys.exit(1)

    result = verify_routes_table(client)

    if result.success:
        logger.info("=" * 60)
        logger.info("VERIFICATION PASSED — routes table is correctly configured.")
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
    setup_migration_logging("verify_routes_table.log")
    main()
