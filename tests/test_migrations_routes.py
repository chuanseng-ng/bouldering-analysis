"""Tests for the routes table migration (001_create_routes_table.sql).

Three test layers:
  1. Offline SQL parsing — no database connection required.
  2. Verifier unit tests — mocked Supabase client.
  3. Integration tests — skipped unless real Supabase credentials are present.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SQL_FILE = _PROJECT_ROOT / "migrations" / "sql" / "001_create_routes_table.sql"


# ---------------------------------------------------------------------------
# Layer 1 — Offline SQL parsing
# ---------------------------------------------------------------------------


class TestRoutesMigrationSQL:
    """Offline checks — parse the SQL file without a database connection."""

    @pytest.fixture
    def sql_content(self) -> str:
        """Return the content of the migration SQL file."""
        assert _SQL_FILE.exists(), f"SQL file not found: {_SQL_FILE}"
        return _SQL_FILE.read_text(encoding="utf-8")

    def test_sql_file_exists(self) -> None:
        """Migration SQL file must exist at the expected path."""
        assert _SQL_FILE.exists()
        assert _SQL_FILE.stat().st_size > 0

    def test_creates_routes_table(self, sql_content: str) -> None:
        """SQL must create the routes table."""
        assert "CREATE TABLE IF NOT EXISTS routes" in sql_content

    def test_uses_gen_random_uuid(self, sql_content: str) -> None:
        """SQL must use gen_random_uuid() (no extension dependency)."""
        assert "gen_random_uuid()" in sql_content
        # Explicitly confirm uuid_generate_v4 is NOT used
        assert "uuid_generate_v4" not in sql_content

    def test_status_check_constraint(self, sql_content: str) -> None:
        """SQL must include a CHECK constraint covering all 4 status values."""
        for value in ("pending", "processing", "done", "failed"):
            assert value in sql_content, f"Status value '{value}' missing from SQL"

    def test_wall_angle_check_constraint(self, sql_content: str) -> None:
        """SQL must include a wall_angle BETWEEN -90 AND 90 CHECK constraint."""
        assert "BETWEEN -90 AND 90" in sql_content

    def test_image_url_check_constraint(self, sql_content: str) -> None:
        """SQL must include a char_length <= 2048 CHECK constraint on image_url."""
        assert "char_length(image_url) <= 2048" in sql_content

    def test_moddatetime_trigger(self, sql_content: str) -> None:
        """SQL must define the moddatetime trigger for updated_at."""
        assert "set_routes_updated_at" in sql_content
        assert "moddatetime" in sql_content

    def test_created_at_index(self, sql_content: str) -> None:
        """SQL must create the idx_routes_created_at index."""
        assert "idx_routes_created_at" in sql_content
        assert "created_at DESC" in sql_content

    def test_status_partial_index(self, sql_content: str) -> None:
        """SQL must create the idx_routes_status_pending partial index."""
        assert "idx_routes_status_pending" in sql_content
        # Check it's a partial index
        assert "WHERE status IN" in sql_content

    def test_rls_enabled(self, sql_content: str) -> None:
        """SQL must enable Row Level Security on the routes table."""
        assert "ENABLE ROW LEVEL SECURITY" in sql_content

    def test_rls_policies_defined(self, sql_content: str) -> None:
        """SQL must define RLS policies for select, insert, update, and delete."""
        for policy in (
            "routes_select_public",
            "routes_insert_service",
            "routes_update_service",
            "routes_delete_service",
        ):
            assert policy in sql_content, f"RLS policy '{policy}' missing from SQL"

    def test_idempotent_table_creation(self, sql_content: str) -> None:
        """CREATE TABLE must use IF NOT EXISTS for idempotency."""
        assert "CREATE TABLE IF NOT EXISTS routes" in sql_content

    def test_idempotent_indexes(self, sql_content: str) -> None:
        """CREATE INDEX statements must use IF NOT EXISTS for idempotency."""
        # Count CREATE INDEX occurrences
        create_index_matches = re.findall(r"CREATE INDEX", sql_content, re.IGNORECASE)
        create_index_if_matches = re.findall(
            r"CREATE INDEX IF NOT EXISTS", sql_content, re.IGNORECASE
        )
        # All CREATE INDEX statements must include IF NOT EXISTS
        assert len(create_index_matches) == len(create_index_if_matches)

    def test_moddatetime_extension_enabled(self, sql_content: str) -> None:
        """SQL must enable the moddatetime extension idempotently."""
        assert "CREATE EXTENSION IF NOT EXISTS moddatetime" in sql_content

    def test_status_column_has_default_pending(self, sql_content: str) -> None:
        """status column must default to 'pending'."""
        assert "DEFAULT 'pending'" in sql_content

    def test_updated_at_and_created_at_not_null(self, sql_content: str) -> None:
        """created_at and updated_at must be NOT NULL."""
        assert re.search(r"created_at\s+TIMESTAMPTZ\s+NOT NULL", sql_content)
        assert re.search(r"updated_at\s+TIMESTAMPTZ\s+NOT NULL", sql_content)

    def test_id_is_primary_key_with_uuid_default(self, sql_content: str) -> None:
        """id column must be UUID PRIMARY KEY with a default."""
        assert re.search(
            r"id\s+UUID\s+PRIMARY KEY DEFAULT gen_random_uuid\(\)", sql_content
        )


# ---------------------------------------------------------------------------
# Layer 2 — Verifier unit tests (mocked Supabase)
# ---------------------------------------------------------------------------


def _make_mock_client(
    table_exists: bool = True,
    columns: list[str] | None = None,
    constraints: list[str] | None = None,
    trigger_exists: bool = True,
    policies: list[str] | None = None,
) -> MagicMock:
    """Build a mock Supabase client for verifier unit tests.

    Args:
        table_exists: Whether the routes table query returns rows.
        columns: List of column names to return. Defaults to all 6 expected.
        constraints: List of constraint names. Defaults to all 3 expected.
        trigger_exists: Whether the trigger query returns rows.
        policies: RLS policy names to return. Defaults to all 4 expected.

    Returns:
        Configured mock client.
    """
    if columns is None:
        columns = [
            "id",
            "image_url",
            "wall_angle",
            "status",
            "created_at",
            "updated_at",
        ]
    if constraints is None:
        constraints = [
            "routes_status_check",
            "routes_image_url_check",
            "routes_wall_angle_check",
        ]
    if policies is None:
        policies = [
            "routes_select_public",
            "routes_insert_service",
            "routes_update_service",
            "routes_delete_service",
        ]

    client = MagicMock()

    def _make_chain(data: list[dict[str, Any]]) -> MagicMock:
        """Return a mock that always resolves to the given data."""
        result = MagicMock()
        result.data = data
        chain = MagicMock()
        chain.select.return_value = chain
        chain.eq.return_value = chain
        chain.execute.return_value = result
        return chain

    table_data = [{"table_name": "routes"}] if table_exists else []
    col_data = [{"column_name": c} for c in columns]
    constraint_data = [{"constraint_name": c} for c in constraints]
    trigger_data = [{"trigger_name": "set_routes_updated_at"}] if trigger_exists else []
    policy_data = [{"policyname": p} for p in policies]

    def _table_side_effect(name: str) -> MagicMock:
        if name == "information_schema.tables":
            return _make_chain(table_data)
        if name == "information_schema.columns":
            return _make_chain(col_data)
        if name == "information_schema.table_constraints":
            return _make_chain(constraint_data)
        if name == "information_schema.triggers":
            return _make_chain(trigger_data)
        if name == "pg_catalog.pg_policies":
            return _make_chain(policy_data)
        return _make_chain([])

    client.table.side_effect = _table_side_effect
    return client


class TestCreateRoutesTableVerifier:
    """Unit tests for the verify_routes_table() function with mocked Supabase."""

    def test_verify_success(self) -> None:
        """verify_routes_table returns success when all checks pass."""
        from scripts.migrations.create_routes_table import verify_routes_table

        client = _make_mock_client()
        result = verify_routes_table(client)

        assert result.success is True
        assert result.errors == []

    def test_verify_table_missing(self) -> None:
        """verify_routes_table fails when the routes table does not exist."""
        from scripts.migrations.create_routes_table import verify_routes_table

        client = _make_mock_client(table_exists=False)
        result = verify_routes_table(client)

        assert result.success is False
        assert any("does not exist" in e for e in result.errors)

    def test_verify_column_missing_one(self) -> None:
        """verify_routes_table fails when a column is missing."""
        from scripts.migrations.create_routes_table import verify_routes_table

        # Remove 'status' column
        client = _make_mock_client(
            columns=["id", "image_url", "wall_angle", "created_at", "updated_at"]
        )
        result = verify_routes_table(client)

        assert result.success is False
        assert any("status" in e for e in result.errors)

    def test_verify_all_columns_missing(self) -> None:
        """verify_routes_table reports all missing columns."""
        from scripts.migrations.create_routes_table import verify_routes_table

        client = _make_mock_client(columns=["id"])
        result = verify_routes_table(client)

        assert result.success is False
        # At least one error about missing columns
        assert any("Missing columns" in e for e in result.errors)

    def test_verify_constraint_missing(self) -> None:
        """verify_routes_table fails when expected CHECK constraints are absent."""
        from scripts.migrations.create_routes_table import verify_routes_table

        client = _make_mock_client(constraints=[])
        result = verify_routes_table(client)

        assert result.success is False
        assert any("Missing CHECK constraints" in e for e in result.errors)

    def test_verify_trigger_missing(self) -> None:
        """verify_routes_table fails when the moddatetime trigger is absent."""
        from scripts.migrations.create_routes_table import verify_routes_table

        client = _make_mock_client(trigger_exists=False)
        result = verify_routes_table(client)

        assert result.success is False
        assert any("set_routes_updated_at" in e for e in result.errors)

    def test_verify_table_check_returns_early_when_missing(self) -> None:
        """When table is missing, only the table-existence error is reported."""
        from scripts.migrations.create_routes_table import verify_routes_table

        client = _make_mock_client(table_exists=False)
        result = verify_routes_table(client)

        # Should return early — only one error about table absence
        assert len(result.errors) == 1

    def test_dry_run_prints_sql(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--dry-run mode prints the SQL file contents to stdout."""
        from scripts.migrations.create_routes_table import main

        with patch("sys.argv", ["create_routes_table.py", "--dry-run"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "CREATE TABLE IF NOT EXISTS routes" in captured.out

    def test_verification_result_fail_sets_success_false(self) -> None:
        """VerificationResult.fail() sets success=False and appends the message."""
        from scripts.migrations.create_routes_table import VerificationResult

        result = VerificationResult()
        result.fail("something went wrong")

        assert not result.success
        assert "something went wrong" in result.errors

    def test_verification_result_multiple_failures(self) -> None:
        """VerificationResult records all failure messages."""
        from scripts.migrations.create_routes_table import VerificationResult

        result = VerificationResult()
        result.fail("error one")
        result.fail("error two")

        assert result.success is False
        assert len(result.errors) == 2

    def test_verify_rls_policy_missing(self) -> None:
        """verify_routes_table fails when an RLS policy is absent."""
        from scripts.migrations.create_routes_table import verify_routes_table

        client = _make_mock_client(policies=["routes_select_public"])  # 3 missing
        result = verify_routes_table(client)

        assert result.success is False
        assert any("RLS policies" in e for e in result.errors)

    def test_config_has_all_4_rls_policies(self) -> None:
        """_CONFIG must list all 4 expected RLS policies."""
        from scripts.migrations.create_routes_table import _CONFIG

        assert len(_CONFIG.expected_rls_policies) == 4


# ---------------------------------------------------------------------------
# Layer 3 — Integration tests (skipped without Supabase credentials)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRoutesMigrationIntegration:
    """Integration tests that run against a live Supabase instance.

    Skipped automatically when BA_SUPABASE_URL / BA_SUPABASE_KEY are not set.
    """

    @pytest.fixture(autouse=True)
    def skip_without_credentials(self) -> None:
        """Skip the whole class if Supabase credentials are absent."""
        import os

        if not (
            os.environ.get("BA_SUPABASE_URL") and os.environ.get("BA_SUPABASE_KEY")
        ):
            pytest.skip("BA_SUPABASE_URL / BA_SUPABASE_KEY not configured")

    def test_verifier_against_live_db(self) -> None:
        """Verifier must confirm routes table is correctly set up in Supabase."""
        from scripts.migrations.create_routes_table import verify_routes_table
        from src.database.supabase_client import get_supabase_client

        client = get_supabase_client()
        result = verify_routes_table(client)

        assert result.success, f"Verification failed: {result.errors}"
