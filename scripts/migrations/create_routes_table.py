#!/usr/bin/env python3
"""Verifier script for the routes table migration (001_create_routes_table.sql).

Checks that the routes table, its columns, CHECK constraints, and the
moddatetime trigger are all present in the target Supabase database.
Does NOT apply the migration — run the SQL file directly in the Supabase
SQL Editor for that.

Usage:
    python scripts/migrations/create_routes_table.py              # verify (default)
    python scripts/migrations/create_routes_table.py --verify-only
    python scripts/migrations/create_routes_table.py --dry-run    # print SQL, no DB call

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

project_root: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(
    0, str(project_root)
)  # ensure src/ is importable when run from any directory

from src.database.supabase_client import (  # noqa: E402  # pylint: disable=wrong-import-position
    SupabaseClientError,
    get_supabase_client,
)


def setup_migration_logging(log_filename: str) -> None:
    """Configure root logging for a migration script.

    Creates the logs directory if it does not exist and initialises
    ``logging.basicConfig`` with a StreamHandler (stdout) and a FileHandler.

    Args:
        log_filename: Name of the log file (e.g. ``"verify_routes_table.log"``).
    """
    project_root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(project_root / "logs" / log_filename),
        ],
    )


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SQL_FILE = project_root / "migrations" / "sql" / "001_create_routes_table.sql"

_EXPECTED_COLUMNS: tuple[str, ...] = (
    "id",
    "image_url",
    "wall_angle",
    "status",
    "created_at",
    "updated_at",
)

_TRIGGER_NAME = "set_routes_updated_at"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Outcome of a routes-table verification run.

    Attributes:
        success: True if all checks passed.
        errors: Human-readable error messages for each failed check.
    """

    success: bool = True
    errors: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        """Record a failure reason and mark success as False.

        Args:
            message: Description of the failed check.
        """
        self.success = False
        self.errors.append(message)


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def _check_table_exists(client: Any) -> bool:
    """Return True if the routes table exists in information_schema.

    Args:
        client: Supabase client instance.

    Returns:
        True if table is present, False otherwise.
    """
    result = (  # type: ignore[attr-defined]
        client.table("information_schema.tables")
        .select("table_name")
        .eq("table_schema", "public")
        .eq("table_name", "routes")
        .execute()
    )
    return bool(result.data)


def _get_columns(client: Any) -> list[str]:
    """Return the list of column names present in the routes table.

    Args:
        client: Supabase client instance.

    Returns:
        Sorted list of column name strings.
    """
    result = (  # type: ignore[attr-defined]
        client.table("information_schema.columns")
        .select("column_name")
        .eq("table_schema", "public")
        .eq("table_name", "routes")
        .execute()
    )
    return sorted(row["column_name"] for row in (result.data or []))


def _check_constraints(client: Any) -> list[str]:
    """Return the list of CHECK constraint names on the routes table.

    Args:
        client: Supabase client instance.

    Returns:
        List of CHECK constraint name strings.
    """
    result = (  # type: ignore[attr-defined]
        client.table("information_schema.table_constraints")
        .select("constraint_name")
        .eq("table_schema", "public")
        .eq("table_name", "routes")
        .eq("constraint_type", "CHECK")
        .execute()
    )
    return [row["constraint_name"] for row in (result.data or [])]


def _check_trigger_exists(client: Any) -> bool:
    """Return True if the moddatetime trigger is present.

    Args:
        client: Supabase client instance.

    Returns:
        True if trigger is present, False otherwise.
    """
    result = (  # type: ignore[attr-defined]
        client.table("information_schema.triggers")
        .select("trigger_name")
        .eq("event_object_schema", "public")
        .eq("event_object_table", "routes")
        .eq("trigger_name", _TRIGGER_NAME)
        .execute()
    )
    return bool(result.data)


# ---------------------------------------------------------------------------
# Public verifier
# ---------------------------------------------------------------------------


def verify_routes_table(client: Any) -> VerificationResult:
    """Verify the routes table schema in the connected Supabase database.

    Checks performed:
    1. Table ``routes`` exists.
    2. All 6 expected columns are present.
    3. At least one CHECK constraint is defined.
    4. The ``set_routes_updated_at`` trigger is present.

    Args:
        client: Supabase client instance (from ``get_supabase_client()``).

    Returns:
        :class:`VerificationResult` with ``success=True`` if all checks pass,
        or ``success=False`` with ``errors`` populated otherwise.
    """
    result = VerificationResult()

    # 1. Table existence
    try:
        if not _check_table_exists(client):
            result.fail("Table 'routes' does not exist in the public schema.")
            return result  # No point checking further if table is absent
        logger.info("[OK] Table 'routes' exists")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result.fail(f"Failed to check table existence: {exc}")
        return result

    # 2. Column presence
    try:
        present = _get_columns(client)
        missing = [c for c in _EXPECTED_COLUMNS if c not in present]
        if missing:
            result.fail(f"Missing columns in 'routes': {missing}")
        else:
            logger.info("[OK] All %d expected columns present", len(_EXPECTED_COLUMNS))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result.fail(f"Failed to check columns: {exc}")

    # 3. CHECK constraints — query returns only CHECK-type constraints
    try:
        check_constraints = _check_constraints(client)
        if not check_constraints:
            result.fail(
                "No CHECK constraints found on 'routes' table "
                "(expected at least status and image_url CHECK constraints)."
            )
        else:
            logger.info(
                "[OK] Found %d CHECK constraint(s) on 'routes'", len(check_constraints)
            )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result.fail(f"Failed to check constraints: {exc}")

    # 4. Trigger
    try:
        if not _check_trigger_exists(client):
            result.fail(f"Trigger '{_TRIGGER_NAME}' not found on 'routes' table.")
        else:
            logger.info("[OK] Trigger '%s' present", _TRIGGER_NAME)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result.fail(f"Failed to check trigger: {exc}")

    return result


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
  python scripts/migrations/create_routes_table.py --verify-only
  python scripts/migrations/create_routes_table.py --dry-run
""",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify-only",
        action="store_true",
        default=True,
        help="Verify the routes table schema (default mode).",
    )
    mode.add_argument(
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
