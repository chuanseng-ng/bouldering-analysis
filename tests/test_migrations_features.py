"""Tests for the features table migration (003_create_features_table.sql).

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
_SQL_FILE = _PROJECT_ROOT / "migrations" / "sql" / "003_create_features_table.sql"


# ---------------------------------------------------------------------------
# Mock helpers (shared across Layer 2 tests)
# ---------------------------------------------------------------------------


def _make_chain(data: list[dict[str, Any]]) -> MagicMock:
    """Return a mock query-builder that resolves to *data*."""
    result = MagicMock()
    result.data = data
    chain = MagicMock()
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.execute.return_value = result
    return chain


def _make_mock_client(
    table_exists: bool = True,
    columns: list[str] | None = None,
    policies: list[str] | None = None,
) -> MagicMock:
    """Build a mock Supabase client for features verifier unit tests.

    Args:
        table_exists: Whether the features table query returns rows.
        columns: Column names to return.  Defaults to all 4 expected columns.
        policies: RLS policy names.  Defaults to all 4 expected policies.

    Returns:
        Configured mock client.
    """
    if columns is None:
        columns = [
            "id",
            "route_id",
            "feature_vector",
            "extracted_at",
        ]
    if policies is None:
        policies = [
            "features_select_public",
            "features_insert_service",
            "features_update_service",
            "features_delete_service",
        ]

    client = MagicMock()
    table_data = [{"table_name": "features"}] if table_exists else []
    col_data = [{"column_name": c} for c in columns]
    policy_data = [{"policyname": p} for p in policies]

    def _table_side_effect(name: str) -> MagicMock:
        if name == "information_schema.tables":
            return _make_chain(table_data)
        if name == "information_schema.columns":
            return _make_chain(col_data)
        if name == "information_schema.table_constraints":
            # No CHECK constraints on the features table — always empty.
            return _make_chain([])
        if name == "pg_catalog.pg_policies":
            return _make_chain(policy_data)
        return _make_chain([])

    client.table.side_effect = _table_side_effect
    return client


# ---------------------------------------------------------------------------
# Layer 1 — Offline SQL parsing
# ---------------------------------------------------------------------------


class TestFeaturesMigrationSQL:
    """Offline checks — parse the SQL file without a database connection."""

    @pytest.fixture
    def sql_content(self) -> str:
        """Return the content of the features migration SQL file."""
        assert _SQL_FILE.exists(), f"SQL file not found: {_SQL_FILE}"
        return _SQL_FILE.read_text(encoding="utf-8")

    def test_sql_file_exists(self) -> None:
        """Migration SQL file must exist at the expected path."""
        assert _SQL_FILE.exists()
        assert _SQL_FILE.stat().st_size > 0

    def test_creates_features_table(self, sql_content: str) -> None:
        """SQL must create the features table with IF NOT EXISTS."""
        assert "CREATE TABLE IF NOT EXISTS features" in sql_content

    def test_moddatetime_extension_enabled(self, sql_content: str) -> None:
        """SQL must enable the moddatetime extension idempotently."""
        assert "CREATE EXTENSION IF NOT EXISTS moddatetime" in sql_content

    # ── Column presence ───────────────────────────────────────────────────────

    def test_column_id(self, sql_content: str) -> None:
        """id column must be present."""
        assert re.search(r"\bid\b", sql_content)

    def test_column_route_id(self, sql_content: str) -> None:
        """route_id column must be present."""
        assert "route_id" in sql_content

    def test_column_feature_vector(self, sql_content: str) -> None:
        """feature_vector column must be present."""
        assert "feature_vector" in sql_content

    def test_column_extracted_at(self, sql_content: str) -> None:
        """extracted_at column must be present."""
        assert "extracted_at" in sql_content

    # ── Absent columns ────────────────────────────────────────────────────────

    def test_no_updated_at_column(self, sql_content: str) -> None:
        """updated_at must NOT be defined as a column — features are write-once."""
        assert not re.search(r"updated_at\s+TIMESTAMPTZ", sql_content)

    def test_no_updated_at_trigger(self, sql_content: str) -> None:
        """No moddatetime trigger for updated_at should be defined."""
        assert "set_features_updated_at" not in sql_content

    # ── Column types ──────────────────────────────────────────────────────────

    def test_id_is_uuid_primary_key_with_default(self, sql_content: str) -> None:
        """id must be UUID PRIMARY KEY with gen_random_uuid() default."""
        assert re.search(
            r"id\s+UUID\s+PRIMARY KEY DEFAULT gen_random_uuid\(\)", sql_content
        )

    def test_feature_vector_is_jsonb_not_null(self, sql_content: str) -> None:
        """feature_vector must be JSONB NOT NULL."""
        assert re.search(r"feature_vector\s+JSONB\s+NOT NULL", sql_content)

    def test_extracted_at_is_timestamptz_not_null(self, sql_content: str) -> None:
        """extracted_at must be TIMESTAMPTZ NOT NULL."""
        assert re.search(r"extracted_at\s+TIMESTAMPTZ\s+NOT NULL", sql_content)

    # ── Constraints ───────────────────────────────────────────────────────────

    def test_unique_route_id(self, sql_content: str) -> None:
        """route_id must have a UNIQUE constraint (one feature vector per route)."""
        assert re.search(r"route_id\s+UUID\s+NOT NULL UNIQUE", sql_content)

    def test_on_delete_cascade(self, sql_content: str) -> None:
        """route_id FK must use ON DELETE CASCADE."""
        assert "ON DELETE CASCADE" in sql_content

    # ── Idempotency ───────────────────────────────────────────────────────────

    def test_idempotent_table_creation(self, sql_content: str) -> None:
        """CREATE TABLE must use IF NOT EXISTS for idempotency."""
        assert "CREATE TABLE IF NOT EXISTS features" in sql_content

    def test_drop_policy_if_exists_select(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for features_select_public."""
        assert "DROP POLICY IF EXISTS features_select_public" in sql_content

    def test_drop_policy_if_exists_insert(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for features_insert_service."""
        assert "DROP POLICY IF EXISTS features_insert_service" in sql_content

    def test_drop_policy_if_exists_update(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for features_update_service."""
        assert "DROP POLICY IF EXISTS features_update_service" in sql_content

    def test_drop_policy_if_exists_delete(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for features_delete_service."""
        assert "DROP POLICY IF EXISTS features_delete_service" in sql_content

    # ── RLS ───────────────────────────────────────────────────────────────────

    def test_rls_enabled(self, sql_content: str) -> None:
        """SQL must enable Row Level Security on the features table."""
        assert "ALTER TABLE features ENABLE ROW LEVEL SECURITY" in sql_content

    def test_rls_policy_select_public(self, sql_content: str) -> None:
        """features_select_public policy must be defined."""
        assert "features_select_public" in sql_content

    def test_rls_policy_insert_service(self, sql_content: str) -> None:
        """features_insert_service policy must be defined."""
        assert "features_insert_service" in sql_content

    def test_rls_policy_update_service(self, sql_content: str) -> None:
        """features_update_service policy must be defined."""
        assert "features_update_service" in sql_content

    def test_rls_policy_delete_service(self, sql_content: str) -> None:
        """features_delete_service policy must be defined."""
        assert "features_delete_service" in sql_content

    # ── No spurious indexes ───────────────────────────────────────────────────

    def test_no_separate_idx_features_route_id(self, sql_content: str) -> None:
        """No separate idx_features_route_id — UNIQUE covers route-scoped lookups."""
        assert not re.search(
            r"CREATE\s+INDEX\b[^;]*idx_features_route_id", sql_content, re.IGNORECASE
        )

    def test_no_check_constraint_on_feature_vector(self, sql_content: str) -> None:
        """feature_vector must have no CHECK constraint — validated at application layer."""
        assert not re.search(r"CHECK\s*\(\s*feature_vector", sql_content)

    def test_fk_references_routes_id(self, sql_content: str) -> None:
        """route_id FK must reference routes(id)."""
        assert "REFERENCES routes(id)" in sql_content


# ---------------------------------------------------------------------------
# Layer 2 — Verifier unit tests (mocked Supabase)
# ---------------------------------------------------------------------------


class TestCreateFeaturesTableVerifier:
    """Unit tests for verify_features_table() with a mocked Supabase client."""

    def test_verify_success(self) -> None:
        """verify_features_table returns success when all checks pass."""
        from scripts.migrations.create_features_table import verify_features_table

        client = _make_mock_client()
        result = verify_features_table(client)

        assert result.success is True
        assert result.errors == []

    def test_verify_table_missing(self) -> None:
        """verify_features_table fails with early exit when features table is absent."""
        from scripts.migrations.create_features_table import verify_features_table

        client = _make_mock_client(table_exists=False)
        result = verify_features_table(client)

        assert result.success is False
        assert len(result.errors) == 1
        assert "does not exist" in result.errors[0]

    def test_verify_column_missing(self) -> None:
        """verify_features_table fails when a column is missing."""
        from scripts.migrations.create_features_table import verify_features_table

        client = _make_mock_client(columns=["id", "route_id", "extracted_at"])
        result = verify_features_table(client)

        assert result.success is False
        assert any("feature_vector" in e for e in result.errors)

    def test_verify_wrong_columns(self) -> None:
        """verify_features_table fails when only wrong columns are present."""
        from scripts.migrations.create_features_table import verify_features_table

        client = _make_mock_client(columns=["id"])  # only id present
        result = verify_features_table(client)

        assert result.success is False
        assert any("Missing columns" in e for e in result.errors)

    def test_verify_rls_policy_missing(self) -> None:
        """verify_features_table fails when an RLS policy is absent."""
        from scripts.migrations.create_features_table import verify_features_table

        client = _make_mock_client(policies=["features_select_public"])  # 3 missing
        result = verify_features_table(client)

        assert result.success is False
        assert any("RLS policies" in e for e in result.errors)

    def test_dry_run_prints_sql(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--dry-run prints the SQL file contents to stdout."""
        from scripts.migrations.create_features_table import main

        with patch("sys.argv", ["create_features_table.py", "--dry-run"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "CREATE TABLE IF NOT EXISTS features" in captured.out

    def test_main_exits_zero_on_success(self) -> None:
        """main() exits with code 0 when verification passes."""
        from scripts.migrations.create_features_table import main

        mock_client = _make_mock_client()
        with patch("sys.argv", ["create_features_table.py"]):
            with patch(
                "scripts.migrations.create_features_table.get_supabase_client",
                return_value=mock_client,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 0

    def test_main_exits_one_on_failure(self) -> None:
        """main() exits with code 1 when verification fails."""
        from scripts.migrations.create_features_table import main

        mock_client = _make_mock_client(table_exists=False)
        with patch("sys.argv", ["create_features_table.py"]):
            with patch(
                "scripts.migrations.create_features_table.get_supabase_client",
                return_value=mock_client,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    def test_main_exits_one_on_connection_failure(self) -> None:
        """main() exits with code 1 when Supabase connection raises SupabaseClientError."""
        from scripts.migrations.create_features_table import main
        from src.database.supabase_client import SupabaseClientError

        with patch("sys.argv", ["create_features_table.py"]):
            with patch(
                "scripts.migrations.create_features_table.get_supabase_client",
                side_effect=SupabaseClientError("connection refused"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    def test_dry_run_exits_one_when_sql_file_missing(self, tmp_path: Path) -> None:
        """--dry-run exits with code 1 when the SQL file does not exist."""
        from scripts.migrations import create_features_table

        nonexistent = tmp_path / "missing.sql"
        with patch("sys.argv", ["create_features_table.py", "--dry-run"]):
            with patch.object(create_features_table, "_SQL_FILE", nonexistent):
                with pytest.raises(SystemExit) as exc_info:
                    create_features_table.main()

        assert exc_info.value.code == 1

    def test_no_trigger_check_for_features(self) -> None:
        """Features verifier must not check for an updated_at trigger."""
        from scripts.migrations.create_features_table import _CONFIG

        assert _CONFIG.trigger_name is None

    def test_no_check_constraints_for_features(self) -> None:
        """Features verifier must not check for CHECK constraints (none defined)."""
        from scripts.migrations.create_features_table import _CONFIG

        assert _CONFIG.expected_check_constraints == frozenset()

    def test_config_has_all_4_columns(self) -> None:
        """_CONFIG must list all 4 expected columns."""
        from scripts.migrations.create_features_table import _CONFIG

        assert len(_CONFIG.expected_columns) == 4

    def test_config_has_all_4_rls_policies(self) -> None:
        """_CONFIG must list all 4 expected RLS policies."""
        from scripts.migrations.create_features_table import _CONFIG

        assert len(_CONFIG.expected_rls_policies) == 4


# ---------------------------------------------------------------------------
# Layer 3 — Integration tests (skipped without Supabase credentials)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFeaturesMigrationIntegration:
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
        """Verifier must confirm features table is correctly set up in Supabase."""
        from scripts.migrations.create_features_table import verify_features_table
        from src.database.supabase_client import get_supabase_client

        client = get_supabase_client()
        result = verify_features_table(client)

        assert result.success, f"Verification failed: {result.errors}"
