"""Tests for the feedback table migration (005_create_feedback_table.sql).

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

from src.grading.constants import V_GRADES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SQL_FILE = _PROJECT_ROOT / "migrations" / "sql" / "005_create_feedback_table.sql"


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
    constraints: list[str] | None = None,
    policies: list[str] | None = None,
) -> MagicMock:
    """Build a mock Supabase client for feedback verifier unit tests.

    Args:
        table_exists: Whether the feedback table query returns rows.
        columns: Column names to return.  Defaults to all 6 expected columns.
        constraints: CHECK constraint names.  Defaults to all 1 expected.
        policies: RLS policy names.  Defaults to all 4 expected policies.

    Returns:
        Configured mock client.
    """
    if columns is None:
        columns = [
            "id",
            "route_id",
            "user_grade",
            "is_accurate",
            "comments",
            "created_at",
        ]
    if constraints is None:
        constraints = [
            "feedback_user_grade_check",
        ]
    if policies is None:
        policies = [
            "feedback_select_public",
            "feedback_insert_public",
            "feedback_update_service",
            "feedback_delete_service",
        ]

    client = MagicMock()
    table_data = [{"table_name": "feedback"}] if table_exists else []
    col_data = [{"column_name": c} for c in columns]
    constraint_data = [{"constraint_name": c} for c in constraints]
    policy_data = [{"policyname": p} for p in policies]

    def _table_side_effect(name: str) -> MagicMock:
        if name == "information_schema.tables":
            return _make_chain(table_data)
        if name == "information_schema.columns":
            return _make_chain(col_data)
        if name == "information_schema.table_constraints":
            return _make_chain(constraint_data)
        if name == "pg_catalog.pg_policies":
            return _make_chain(policy_data)
        return _make_chain([])

    client.table.side_effect = _table_side_effect
    return client


# ---------------------------------------------------------------------------
# Layer 1 — Offline SQL parsing
# ---------------------------------------------------------------------------


class TestFeedbackMigrationSQL:
    """Offline checks — parse the SQL file without a database connection."""

    @pytest.fixture
    def sql_content(self) -> str:
        """Return the content of the feedback migration SQL file."""
        assert _SQL_FILE.exists(), f"SQL file not found: {_SQL_FILE}"
        return _SQL_FILE.read_text(encoding="utf-8")

    # ── File existence ─────────────────────────────────────────────────────

    def test_sql_file_exists(self) -> None:
        """Migration SQL file must exist at the expected path."""
        assert _SQL_FILE.exists()
        assert _SQL_FILE.stat().st_size > 0

    # ── Table DDL ─────────────────────────────────────────────────────────

    def test_creates_feedback_table(self, sql_content: str) -> None:
        """SQL must create the feedback table with IF NOT EXISTS."""
        assert "CREATE TABLE IF NOT EXISTS feedback" in sql_content

    def test_idempotent_table_creation(self, sql_content: str) -> None:
        """CREATE TABLE must use IF NOT EXISTS for idempotency."""
        assert "IF NOT EXISTS" in sql_content

    def test_no_moddatetime_extension(self, sql_content: str) -> None:
        """SQL must NOT enable moddatetime — feedback table uses no trigger."""
        assert "CREATE EXTENSION IF NOT EXISTS moddatetime" not in sql_content

    # ── Primary key ────────────────────────────────────────────────────────

    def test_id_is_uuid_pk(self, sql_content: str) -> None:
        """id must be UUID PRIMARY KEY with gen_random_uuid() default."""
        assert re.search(
            r"id\s+UUID\s+PRIMARY KEY DEFAULT gen_random_uuid\(\)", sql_content
        )

    # ── Foreign key ────────────────────────────────────────────────────────

    def test_route_id_not_null(self, sql_content: str) -> None:
        """route_id must be NOT NULL."""
        assert re.search(r"route_id\s+UUID\s+NOT NULL", sql_content)

    def test_fk_references_routes_id(self, sql_content: str) -> None:
        """route_id FK must reference routes(id)."""
        assert "REFERENCES routes(id)" in sql_content

    def test_on_delete_cascade(self, sql_content: str) -> None:
        """route_id FK must use ON DELETE CASCADE."""
        assert "ON DELETE CASCADE" in sql_content

    # ── Column presence ────────────────────────────────────────────────────

    def test_column_user_grade(self, sql_content: str) -> None:
        """user_grade column must be present."""
        assert "user_grade" in sql_content

    def test_column_is_accurate(self, sql_content: str) -> None:
        """is_accurate column must be present."""
        assert "is_accurate" in sql_content

    def test_column_comments(self, sql_content: str) -> None:
        """comments column must be present."""
        assert "comments" in sql_content

    def test_column_created_at(self, sql_content: str) -> None:
        """created_at column must be present."""
        assert "created_at" in sql_content

    # ── Column types and nullability ───────────────────────────────────────

    def test_user_grade_is_varchar10(self, sql_content: str) -> None:
        """user_grade must be VARCHAR(10)."""
        assert re.search(r"user_grade\s+VARCHAR\(10\)", sql_content)

    def test_is_accurate_is_boolean(self, sql_content: str) -> None:
        """is_accurate must be BOOLEAN."""
        assert re.search(r"is_accurate\s+BOOLEAN", sql_content)

    def test_comments_is_text(self, sql_content: str) -> None:
        """comments must be TEXT."""
        assert re.search(r"comments\s+TEXT", sql_content)

    def test_created_at_is_timestamptz_not_null(self, sql_content: str) -> None:
        """created_at must be TIMESTAMPTZ NOT NULL."""
        assert re.search(r"created_at\s+TIMESTAMPTZ\s+NOT NULL", sql_content)

    def test_user_grade_is_nullable(self, sql_content: str) -> None:
        """user_grade must be nullable — user may omit grade estimate."""
        assert not re.search(r"user_grade\s+VARCHAR\(10\)\s+NOT NULL", sql_content)

    def test_user_grade_check_allows_null(self, sql_content: str) -> None:
        """user_grade CHECK must include IS NULL OR to permit null values."""
        assert "IS NULL OR" in sql_content

    def test_user_grade_check_contains_all_18_values(self, sql_content: str) -> None:
        """user_grade CHECK must include all 18 V-scale values (V0–V17)."""
        assert all(f"'{g}'" in sql_content for g in V_GRADES)

    # ── Index ──────────────────────────────────────────────────────────────

    def test_explicit_index_present(self, sql_content: str) -> None:
        """Explicit index idx_feedback_route_id_created_at must be defined."""
        assert "idx_feedback_route_id_created_at" in sql_content

    def test_index_covers_route_id_and_created_at(self, sql_content: str) -> None:
        """Index must cover route_id and created_at columns."""
        assert re.search(
            r"idx_feedback_route_id_created_at\s+ON\s+feedback\s*\(\s*route_id\s*,\s*created_at",
            sql_content,
        )

    # ── Design decisions ───────────────────────────────────────────────────

    def test_no_unique_on_route_id(self, sql_content: str) -> None:
        """route_id must NOT have a UNIQUE constraint — multiple feedback per route allowed."""
        assert not re.search(r"route_id\s+UUID\s+NOT NULL\s+UNIQUE", sql_content)
        assert not re.search(r"UNIQUE\s*\(\s*route_id\s*\)", sql_content)

    def test_no_updated_at_column(self, sql_content: str) -> None:
        """updated_at must NOT be defined — feedback is append-only."""
        assert not re.search(r"updated_at\s+TIMESTAMPTZ", sql_content)

    def test_no_trigger_reference(self, sql_content: str) -> None:
        """No CREATE TRIGGER must be defined — feedback is append-only."""
        assert "CREATE TRIGGER" not in sql_content

    def test_append_only_no_delete_before_insert(self, sql_content: str) -> None:
        """SQL must not contain DELETE FROM — feedback rows are never overwritten."""
        assert "DELETE FROM" not in sql_content

    # ── Idempotency ────────────────────────────────────────────────────────

    def test_drop_policy_if_exists_select(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for feedback_select_public."""
        assert "DROP POLICY IF EXISTS feedback_select_public" in sql_content

    def test_drop_policy_if_exists_insert(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for feedback_insert_public."""
        assert "DROP POLICY IF EXISTS feedback_insert_public" in sql_content

    def test_drop_policy_if_exists_update(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for feedback_update_service."""
        assert "DROP POLICY IF EXISTS feedback_update_service" in sql_content

    def test_drop_policy_if_exists_delete(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for feedback_delete_service."""
        assert "DROP POLICY IF EXISTS feedback_delete_service" in sql_content

    # ── RLS ───────────────────────────────────────────────────────────────

    def test_rls_enabled(self, sql_content: str) -> None:
        """SQL must enable Row Level Security on the feedback table."""
        assert "ALTER TABLE feedback ENABLE ROW LEVEL SECURITY" in sql_content

    def test_rls_policy_select_public(self, sql_content: str) -> None:
        """feedback_select_public policy must be defined."""
        assert "feedback_select_public" in sql_content

    def test_rls_policy_insert_public(self, sql_content: str) -> None:
        """feedback_insert_public policy must be defined."""
        assert "feedback_insert_public" in sql_content

    def test_rls_policy_update_service(self, sql_content: str) -> None:
        """feedback_update_service policy must be defined."""
        assert "feedback_update_service" in sql_content

    def test_rls_policy_delete_service(self, sql_content: str) -> None:
        """feedback_delete_service policy must be defined."""
        assert "feedback_delete_service" in sql_content

    def test_rls_insert_is_to_public_not_service(self, sql_content: str) -> None:
        """INSERT policy must use TO PUBLIC, not TO service_role — anonymous submission."""
        # Extract the insert policy block
        insert_block = re.search(
            r"CREATE POLICY feedback_insert_public.*?;",
            sql_content,
            re.DOTALL,
        )
        assert insert_block is not None, "feedback_insert_public policy not found"
        block_text = insert_block.group(0)
        assert "TO PUBLIC" in block_text
        assert "service_role" not in block_text


# ---------------------------------------------------------------------------
# Layer 2 — Verifier unit tests (mocked Supabase)
# ---------------------------------------------------------------------------


class TestCreateFeedbackTableVerifier:
    """Unit tests for verify_feedback_table() with a mocked Supabase client."""

    def test_verify_success(self) -> None:
        """verify_feedback_table returns success when all checks pass."""
        from scripts.migrations.create_feedback_table import verify_feedback_table

        client = _make_mock_client()
        result = verify_feedback_table(client)

        assert result.success is True
        assert result.errors == []

    def test_verify_table_missing(self) -> None:
        """verify_feedback_table fails with early exit when feedback table is absent."""
        from scripts.migrations.create_feedback_table import verify_feedback_table

        client = _make_mock_client(table_exists=False)
        result = verify_feedback_table(client)

        assert result.success is False
        assert len(result.errors) == 1
        assert "does not exist" in result.errors[0]

    def test_verify_column_missing(self) -> None:
        """verify_feedback_table fails when a column is missing."""
        from scripts.migrations.create_feedback_table import verify_feedback_table

        client = _make_mock_client(
            columns=[
                "id",
                "route_id",
                "user_grade",
                "is_accurate",
                "comments",
                # created_at omitted
            ]
        )
        result = verify_feedback_table(client)

        assert result.success is False
        assert any("created_at" in e for e in result.errors)

    def test_verify_wrong_columns(self) -> None:
        """verify_feedback_table fails when only wrong columns are present."""
        from scripts.migrations.create_feedback_table import verify_feedback_table

        client = _make_mock_client(columns=["id"])  # only id present
        result = verify_feedback_table(client)

        assert result.success is False
        assert any("Missing columns" in e for e in result.errors)

    def test_verify_constraint_missing(self) -> None:
        """verify_feedback_table fails when the CHECK constraint is absent."""
        from scripts.migrations.create_feedback_table import verify_feedback_table

        client = _make_mock_client(constraints=[])  # constraint missing
        result = verify_feedback_table(client)

        assert result.success is False
        assert any("CHECK constraints" in e for e in result.errors)

    def test_verify_rls_policy_missing(self) -> None:
        """verify_feedback_table fails when an RLS policy is absent."""
        from scripts.migrations.create_feedback_table import verify_feedback_table

        client = _make_mock_client(policies=["feedback_select_public"])  # 3 missing
        result = verify_feedback_table(client)

        assert result.success is False
        assert any("RLS policies" in e for e in result.errors)

    def test_dry_run_prints_sql(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--dry-run prints the SQL file contents to stdout."""
        from scripts.migrations.create_feedback_table import main

        with patch("sys.argv", ["create_feedback_table.py", "--dry-run"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "CREATE TABLE IF NOT EXISTS feedback" in captured.out

    def test_main_exits_zero_on_success(self) -> None:
        """main() exits with code 0 when verification passes."""
        from scripts.migrations.create_feedback_table import main

        mock_client = _make_mock_client()
        with patch("sys.argv", ["create_feedback_table.py"]):
            with patch(
                "scripts.migrations.create_feedback_table.get_supabase_client",
                return_value=mock_client,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 0

    def test_main_exits_one_on_failure(self) -> None:
        """main() exits with code 1 when verification fails."""
        from scripts.migrations.create_feedback_table import main

        mock_client = _make_mock_client(table_exists=False)
        with patch("sys.argv", ["create_feedback_table.py"]):
            with patch(
                "scripts.migrations.create_feedback_table.get_supabase_client",
                return_value=mock_client,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    def test_main_exits_one_on_connection_failure(self) -> None:
        """main() exits with code 1 when Supabase connection raises SupabaseClientError."""
        from scripts.migrations.create_feedback_table import main
        from src.database.supabase_client import SupabaseClientError

        with patch("sys.argv", ["create_feedback_table.py"]):
            with patch(
                "scripts.migrations.create_feedback_table.get_supabase_client",
                side_effect=SupabaseClientError("connection refused"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    def test_dry_run_exits_one_when_sql_file_missing(self, tmp_path: Path) -> None:
        """--dry-run exits with code 1 when the SQL file does not exist."""
        from scripts.migrations import create_feedback_table

        nonexistent = tmp_path / "missing.sql"
        with patch("sys.argv", ["create_feedback_table.py", "--dry-run"]):
            with patch.object(create_feedback_table, "_SQL_FILE", nonexistent):
                with pytest.raises(SystemExit) as exc_info:
                    create_feedback_table.main()

        assert exc_info.value.code == 1

    def test_no_trigger_check_for_feedback(self) -> None:
        """Feedback verifier must not check for an updated_at trigger."""
        from scripts.migrations.create_feedback_table import _CONFIG

        assert _CONFIG.trigger_name is None

    def test_config_has_all_6_columns(self) -> None:
        """_CONFIG must list all 6 expected columns."""
        from scripts.migrations.create_feedback_table import _CONFIG

        assert len(_CONFIG.expected_columns) == 6

    def test_config_has_1_check_constraint(self) -> None:
        """_CONFIG must list exactly 1 expected CHECK constraint."""
        from scripts.migrations.create_feedback_table import _CONFIG

        assert len(_CONFIG.expected_check_constraints) == 1

    def test_config_has_all_4_rls_policies(self) -> None:
        """_CONFIG must list all 4 expected RLS policies."""
        from scripts.migrations.create_feedback_table import _CONFIG

        assert len(_CONFIG.expected_rls_policies) == 4

    def test_config_insert_policy_is_public(self) -> None:
        """_CONFIG must include feedback_insert_public (not feedback_insert_service)."""
        from scripts.migrations.create_feedback_table import _CONFIG

        assert "feedback_insert_public" in _CONFIG.expected_rls_policies
        assert "feedback_insert_service" not in _CONFIG.expected_rls_policies


# ---------------------------------------------------------------------------
# Layer 3 — Integration tests (skipped without Supabase credentials)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFeedbackMigrationIntegration:
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
        """Verifier must confirm feedback table is correctly set up in Supabase."""
        from scripts.migrations.create_feedback_table import verify_feedback_table
        from src.database.supabase_client import get_supabase_client

        client = get_supabase_client()
        result = verify_feedback_table(client)

        assert result.success, f"Verification failed: {result.errors}"
