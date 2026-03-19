"""Shared PostgREST-based utilities for Supabase migration verifier scripts.

Provides common types, helpers, and a generic ``verify_table()`` entry-point
used by ``create_routes_table.py``, ``create_holds_table.py``,
``create_features_table.py``, ``create_predictions_table.py``, and future
migration scripts.  All helpers query ``information_schema`` (for schema
metadata) or ``pg_catalog.pg_policies`` (for RLS policies) via the Supabase
PostgREST client — no direct database connection is required.

Usage:
    from scripts.migrations._migration_utils import (
        TableVerificationConfig,
        VerificationResult,
        setup_migration_logging,
        verify_table,
    )

Notes:
    - ``information_schema`` and ``pg_catalog`` must be exposed in your
      Supabase project's API search path (Settings → API → Extra search paths)
      for the live-verify mode to work.  Dry-run mode and all unit tests work
      without this configuration.
    - This module is intentionally free of third-party dependencies beyond
      the Supabase client so it can be imported in migration scripts that run
      before the full virtual environment is activated.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# ── project root (scripts/migrations/ → 3 levels up) ─────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed Protocol for the Supabase client surface used by verifier helpers
# ---------------------------------------------------------------------------


class SupabaseClientLike(Protocol):
    """Minimal protocol for the Supabase client used by verifier helpers.

    Only the ``table()`` entry-point is typed here.  The returned builder's
    chain methods (``.select()``, ``.eq()``, ``.execute()``) vary between the
    real ``SyncRequestBuilder`` and test mocks, so the return type is ``Any``
    to remain compatible with both.
    """

    def table(self, table_name: str) -> Any:
        """Return a query builder targeting the named table/view."""
        ...


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------


@dataclass
class TableVerificationConfig:
    """Configuration for a single table verification run.

    Attributes:
        table_name: Name of the table to verify in the ``public`` schema.
        expected_columns: Tuple of column names that must all be present.
        trigger_name: Name of the ``BEFORE UPDATE`` trigger to check, or
            ``None`` to skip the trigger check (e.g. write-once tables).
        expected_check_constraints: frozenset of CHECK constraint names that
            must all be present on the table.
        expected_rls_policies: frozenset of RLS policy names that must all be
            present.  Pass ``frozenset()`` to skip the RLS policy check.
    """

    table_name: str
    expected_columns: tuple[str, ...]
    trigger_name: str | None
    expected_check_constraints: frozenset[str]
    expected_rls_policies: frozenset[str]


@dataclass
class VerificationResult:
    """Outcome of a table verification run.

    Attributes:
        success: ``True`` if all checks passed; ``False`` as soon as
            :meth:`fail` is called.
        errors: Human-readable error messages for each failed check,
            in the order they were recorded.
    """

    success: bool = True
    errors: list[str] = field(default_factory=list)

    def fail(self, message: str) -> "VerificationResult":
        """Record a failure reason and mark success as False.

        Args:
            message: Description of the failed check.

        Returns:
            ``self``, enabling method chaining.
        """
        self.success = False
        self.errors.append(message)
        return self


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_migration_logging(log_filename: str) -> None:
    """Configure root logging for a migration script.

    Creates the ``logs/`` directory under the project root if it does not
    exist and initialises ``logging.basicConfig`` with a
    :class:`~logging.StreamHandler` (stdout) and a
    :class:`~logging.FileHandler`.

    Args:
        log_filename: Name of the log file
            (e.g. ``"verify_holds_table.log"``).
    """
    PROJECT_ROOT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(PROJECT_ROOT / "logs" / log_filename),
        ],
    )


# ---------------------------------------------------------------------------
# Individual check helpers
# ---------------------------------------------------------------------------


def check_table_exists(client: SupabaseClientLike, table_name: str) -> bool:
    """Return True if *table_name* exists in the ``public`` schema.

    Args:
        client: Supabase client instance.
        table_name: Name of the table to look up.

    Returns:
        ``True`` if the table is present in ``information_schema.tables``,
        ``False`` otherwise.
    """
    result = (
        client.table("information_schema.tables")
        .select("table_name")
        .eq("table_schema", "public")
        .eq("table_name", table_name)
        .execute()
    )
    return bool(result.data)


def get_columns(client: SupabaseClientLike, table_name: str) -> list[str]:
    """Return a sorted list of column names for *table_name*.

    Args:
        client: Supabase client instance.
        table_name: Name of the table whose columns to retrieve.

    Returns:
        Sorted list of column name strings from
        ``information_schema.columns``.
    """
    result = (
        client.table("information_schema.columns")
        .select("column_name")
        .eq("table_schema", "public")
        .eq("table_name", table_name)
        .execute()
    )
    return sorted(row["column_name"] for row in (result.data or []))


def check_constraints(
    client: SupabaseClientLike,
    table_name: str,
    expected: frozenset[str],
) -> list[str]:
    """Return error messages for any missing expected CHECK constraints.

    Extra constraints present in the database but not in *expected* are
    silently ignored — we only verify that all required constraints exist.

    Args:
        client: Supabase client instance.
        table_name: Name of the table to inspect.
        expected: frozenset of CHECK constraint names that must be present.

    Returns:
        A list containing one error message if any expected constraints are
        missing, or an empty list if all are present.
    """
    if not expected:
        return []
    result = (
        client.table("information_schema.table_constraints")
        .select("constraint_name")
        .eq("table_schema", "public")
        .eq("table_name", table_name)
        .eq("constraint_type", "CHECK")
        .execute()
    )
    actual = {row["constraint_name"] for row in (result.data or [])}
    missing = expected - actual
    if missing:
        return [f"Missing CHECK constraints on '{table_name}': {sorted(missing)}"]
    return []


def check_trigger(
    client: SupabaseClientLike,
    table_name: str,
    trigger_name: str,
) -> bool:
    """Return True if *trigger_name* exists on *table_name*.

    Args:
        client: Supabase client instance.
        table_name: Name of the table to inspect.
        trigger_name: Name of the trigger to look up in
            ``information_schema.triggers``.

    Returns:
        ``True`` if the trigger is present, ``False`` otherwise.
    """
    result = (
        client.table("information_schema.triggers")
        .select("trigger_name")
        .eq("event_object_schema", "public")
        .eq("event_object_table", table_name)
        .eq("trigger_name", trigger_name)
        .execute()
    )
    return bool(result.data)


def check_rls_policies(
    client: SupabaseClientLike,
    table_name: str,
    expected_policies: frozenset[str],
) -> list[str]:
    """Return error messages for any missing expected RLS policies.

    If *expected_policies* is empty the check is skipped and an empty list
    is returned immediately without a database round-trip.

    Args:
        client: Supabase client instance.
        table_name: Name of the table to inspect.
        expected_policies: frozenset of RLS policy names that must be
            present.  Pass ``frozenset()`` to skip the check.

    Returns:
        A list containing one error message if any expected policies are
        missing, or an empty list if all are present (or the check is
        skipped).
    """
    if not expected_policies:
        return []
    result = (
        client.table("pg_catalog.pg_policies")
        .select("policyname")
        .eq("schemaname", "public")
        .eq("tablename", table_name)
        .execute()
    )
    actual = {row["policyname"] for row in (result.data or [])}
    missing = expected_policies - actual
    if missing:
        return [f"Missing RLS policies on '{table_name}': {sorted(missing)}"]
    return []


# ---------------------------------------------------------------------------
# Generic table verifier
# ---------------------------------------------------------------------------


def verify_table(
    client: SupabaseClientLike,
    config: TableVerificationConfig,
) -> VerificationResult:
    """Verify a table's schema against *config*.

    Checks performed (in order):

    1. Table ``config.table_name`` exists in the ``public`` schema.
       Returns early with a single error if not found.
    2. All columns in ``config.expected_columns`` are present.
    3. All CHECK constraints in ``config.expected_check_constraints`` are
       present.
    4. Trigger ``config.trigger_name`` is present — **skipped** when
       ``config.trigger_name is None``.
    5. All RLS policies in ``config.expected_rls_policies`` are present —
       **skipped** when the frozenset is empty.

    Args:
        client: Supabase client instance (from ``get_supabase_client()``).
        config: Verification configuration describing what to check.

    Returns:
        :class:`VerificationResult` with ``success=True`` if every enabled
        check passes, or ``success=False`` with ``errors`` populated
        otherwise.
    """
    result = VerificationResult()
    table_name = config.table_name

    # 1. Table existence
    try:
        if not check_table_exists(client, table_name):
            result.fail(f"Table '{table_name}' does not exist in the public schema.")
            return result  # No point checking further if the table is absent
        logger.info("[OK] Table '%s' exists", table_name)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result.fail(f"Failed to check table existence: {exc}")
        return result

    # 2. Column presence
    try:
        present = get_columns(client, table_name)
        missing = [c for c in config.expected_columns if c not in present]
        if missing:
            result.fail(f"Missing columns in '{table_name}': {missing}")
        else:
            logger.info(
                "[OK] All %d expected columns present", len(config.expected_columns)
            )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result.fail(f"Failed to check columns: {exc}")

    # 3. CHECK constraints
    try:
        errors = check_constraints(
            client, table_name, config.expected_check_constraints
        )
        for err in errors:
            result.fail(err)
        if not errors:
            logger.info(
                "[OK] All %d expected CHECK constraint(s) present on '%s'",
                len(config.expected_check_constraints),
                table_name,
            )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result.fail(f"Failed to check constraints: {exc}")

    # 4. Trigger (skipped when trigger_name is None)
    if config.trigger_name is not None:
        try:
            if not check_trigger(client, table_name, config.trigger_name):
                result.fail(
                    f"Trigger '{config.trigger_name}' not found on '{table_name}' table."
                )
            else:
                logger.info("[OK] Trigger '%s' present", config.trigger_name)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result.fail(f"Failed to check trigger: {exc}")

    # 5. RLS policies (skipped when expected_rls_policies is empty)
    if config.expected_rls_policies:
        try:
            errors = check_rls_policies(
                client, table_name, config.expected_rls_policies
            )
            for err in errors:
                result.fail(err)
            if not errors:
                logger.info(
                    "[OK] All %d expected RLS policy(s) present on '%s'",
                    len(config.expected_rls_policies),
                    table_name,
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result.fail(f"Failed to check RLS policies: {exc}")

    return result
