"""Tests for the holds table migration (002_create_holds_table.sql).

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
_SQL_FILE = _PROJECT_ROOT / "migrations" / "sql" / "002_create_holds_table.sql"


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
    """Build a mock Supabase client for holds verifier unit tests.

    Args:
        table_exists: Whether the holds table query returns rows.
        columns: Column names to return.  Defaults to all 18 expected columns.
        constraints: CHECK constraint names.  Defaults to all 15 expected.
        policies: RLS policy names.  Defaults to all 4 expected policies.

    Returns:
        Configured mock client.
    """
    if columns is None:
        columns = [
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
            "prob_volume",
            "prob_unknown",
            "created_at",
        ]
    if constraints is None:
        constraints = [
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
            "holds_prob_volume_check",
            "holds_prob_unknown_check",
        ]
    if policies is None:
        policies = [
            "holds_select_public",
            "holds_insert_service",
            "holds_update_service",
            "holds_delete_service",
        ]

    client = MagicMock()
    table_data = [{"table_name": "holds"}] if table_exists else []
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


class TestHoldsMigrationSQL:
    """Offline checks — parse the SQL file without a database connection."""

    @pytest.fixture
    def sql_content(self) -> str:
        """Return the content of the holds migration SQL file."""
        assert _SQL_FILE.exists(), f"SQL file not found: {_SQL_FILE}"
        return _SQL_FILE.read_text(encoding="utf-8")

    def test_sql_file_exists(self) -> None:
        """Migration SQL file must exist at the expected path."""
        assert _SQL_FILE.exists()
        assert _SQL_FILE.stat().st_size > 0

    def test_creates_holds_table(self, sql_content: str) -> None:
        """SQL must create the holds table with IF NOT EXISTS."""
        assert "CREATE TABLE IF NOT EXISTS holds" in sql_content

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

    def test_column_hold_id(self, sql_content: str) -> None:
        """hold_id column must be present."""
        assert "hold_id" in sql_content

    def test_column_x_center(self, sql_content: str) -> None:
        """x_center column must be present."""
        assert "x_center" in sql_content

    def test_column_y_center(self, sql_content: str) -> None:
        """y_center column must be present."""
        assert "y_center" in sql_content

    def test_column_width(self, sql_content: str) -> None:
        """width column must be present."""
        assert re.search(r"\bwidth\b", sql_content)

    def test_column_height(self, sql_content: str) -> None:
        """height column must be present."""
        assert re.search(r"\bheight\b", sql_content)

    def test_column_detection_class(self, sql_content: str) -> None:
        """detection_class column must be present."""
        assert "detection_class" in sql_content

    def test_column_detection_confidence(self, sql_content: str) -> None:
        """detection_confidence column must be present."""
        assert "detection_confidence" in sql_content

    def test_column_hold_type(self, sql_content: str) -> None:
        """hold_type column must be present."""
        assert "hold_type" in sql_content

    def test_column_type_confidence(self, sql_content: str) -> None:
        """type_confidence column must be present."""
        assert "type_confidence" in sql_content

    def test_column_prob_jug(self, sql_content: str) -> None:
        """prob_jug column must be present."""
        assert "prob_jug" in sql_content

    def test_column_prob_crimp(self, sql_content: str) -> None:
        """prob_crimp column must be present."""
        assert "prob_crimp" in sql_content

    def test_column_prob_sloper(self, sql_content: str) -> None:
        """prob_sloper column must be present."""
        assert "prob_sloper" in sql_content

    def test_column_prob_pinch(self, sql_content: str) -> None:
        """prob_pinch column must be present."""
        assert "prob_pinch" in sql_content

    def test_column_prob_volume(self, sql_content: str) -> None:
        """prob_volume column must be present."""
        assert "prob_volume" in sql_content

    def test_column_prob_unknown(self, sql_content: str) -> None:
        """prob_unknown column must be present."""
        assert "prob_unknown" in sql_content

    def test_column_created_at(self, sql_content: str) -> None:
        """created_at column must be present."""
        assert "created_at" in sql_content

    # ── Absent columns ────────────────────────────────────────────────────────

    def test_no_updated_at_column(self, sql_content: str) -> None:
        """updated_at must NOT be defined as a column — holds are write-once."""
        # Check there is no column definition for updated_at (comments may mention it)
        assert not re.search(r"updated_at\s+TIMESTAMPTZ", sql_content)

    def test_no_updated_at_trigger(self, sql_content: str) -> None:
        """No moddatetime trigger for updated_at should be defined."""
        assert "set_holds_updated_at" not in sql_content

    # ── Constraints ───────────────────────────────────────────────────────────

    def test_unique_route_id_hold_id(self, sql_content: str) -> None:
        """UNIQUE (route_id, hold_id) composite constraint must be present."""
        assert re.search(r"UNIQUE\s*\(\s*route_id\s*,\s*hold_id\s*\)", sql_content)

    def test_on_delete_cascade(self, sql_content: str) -> None:
        """route_id FK must use ON DELETE CASCADE."""
        assert "ON DELETE CASCADE" in sql_content

    def test_check_hold_id_non_negative(self, sql_content: str) -> None:
        """hold_id must have a CHECK (hold_id >= 0) constraint."""
        assert "hold_id >= 0" in sql_content

    def test_check_x_center_between(self, sql_content: str) -> None:
        """x_center must have a BETWEEN 0 AND 1 CHECK constraint."""
        assert re.search(r"x_center\s+BETWEEN\s+0\s+AND\s+1", sql_content)

    def test_check_y_center_between(self, sql_content: str) -> None:
        """y_center must have a BETWEEN 0 AND 1 CHECK constraint."""
        assert re.search(r"y_center\s+BETWEEN\s+0\s+AND\s+1", sql_content)

    def test_check_width_between(self, sql_content: str) -> None:
        """width must have a BETWEEN 0 AND 1 CHECK constraint."""
        assert re.search(r"width\s+BETWEEN\s+0\s+AND\s+1", sql_content)

    def test_check_height_between(self, sql_content: str) -> None:
        """height must have a BETWEEN 0 AND 1 CHECK constraint."""
        assert re.search(r"height\s+BETWEEN\s+0\s+AND\s+1", sql_content)

    def test_check_detection_class_values(self, sql_content: str) -> None:
        """detection_class must have a CHECK constraint allowing 'hold' and 'volume'."""
        assert "'hold'" in sql_content
        assert "'volume'" in sql_content

    def test_check_detection_confidence_between(self, sql_content: str) -> None:
        """detection_confidence must have a BETWEEN 0 AND 1 CHECK constraint."""
        assert re.search(r"detection_confidence\s+BETWEEN\s+0\s+AND\s+1", sql_content)

    def test_check_hold_type_values(self, sql_content: str) -> None:
        """hold_type must CHECK all 6 valid values."""
        for value in ("jug", "crimp", "sloper", "pinch", "volume", "unknown"):
            assert value in sql_content, f"hold_type value '{value}' missing from SQL"

    def test_check_type_confidence_between(self, sql_content: str) -> None:
        """type_confidence must have a BETWEEN 0 AND 1 CHECK constraint."""
        assert re.search(r"type_confidence\s+BETWEEN\s+0\s+AND\s+1", sql_content)

    def test_check_prob_jug_between(self, sql_content: str) -> None:
        """prob_jug must have a BETWEEN 0 AND 1 CHECK constraint."""
        assert re.search(r"prob_jug\s+BETWEEN\s+0\s+AND\s+1", sql_content)

    def test_check_prob_crimp_between(self, sql_content: str) -> None:
        """prob_crimp must have a BETWEEN 0 AND 1 CHECK constraint."""
        assert re.search(r"prob_crimp\s+BETWEEN\s+0\s+AND\s+1", sql_content)

    def test_check_prob_sloper_between(self, sql_content: str) -> None:
        """prob_sloper must have a BETWEEN 0 AND 1 CHECK constraint."""
        assert re.search(r"prob_sloper\s+BETWEEN\s+0\s+AND\s+1", sql_content)

    def test_check_prob_pinch_between(self, sql_content: str) -> None:
        """prob_pinch must have a BETWEEN 0 AND 1 CHECK constraint."""
        assert re.search(r"prob_pinch\s+BETWEEN\s+0\s+AND\s+1", sql_content)

    def test_check_prob_volume_between(self, sql_content: str) -> None:
        """prob_volume must have a BETWEEN 0 AND 1 CHECK constraint."""
        assert re.search(r"prob_volume\s+BETWEEN\s+0\s+AND\s+1", sql_content)

    def test_check_prob_unknown_between(self, sql_content: str) -> None:
        """prob_unknown must have a BETWEEN 0 AND 1 CHECK constraint."""
        assert re.search(r"prob_unknown\s+BETWEEN\s+0\s+AND\s+1", sql_content)

    # ── Idempotency ───────────────────────────────────────────────────────────

    def test_idempotent_table_creation(self, sql_content: str) -> None:
        """CREATE TABLE must use IF NOT EXISTS for idempotency."""
        assert "CREATE TABLE IF NOT EXISTS holds" in sql_content

    def test_drop_policy_if_exists_select(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for holds_select_public."""
        assert "DROP POLICY IF EXISTS holds_select_public" in sql_content

    def test_drop_policy_if_exists_insert(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for holds_insert_service."""
        assert "DROP POLICY IF EXISTS holds_insert_service" in sql_content

    def test_drop_policy_if_exists_update(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for holds_update_service."""
        assert "DROP POLICY IF EXISTS holds_update_service" in sql_content

    def test_drop_policy_if_exists_delete(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for holds_delete_service."""
        assert "DROP POLICY IF EXISTS holds_delete_service" in sql_content

    # ── RLS ───────────────────────────────────────────────────────────────────

    def test_rls_enabled(self, sql_content: str) -> None:
        """SQL must enable Row Level Security on the holds table."""
        assert "ALTER TABLE holds ENABLE ROW LEVEL SECURITY" in sql_content

    def test_rls_policy_select_public(self, sql_content: str) -> None:
        """holds_select_public policy must be defined."""
        assert "holds_select_public" in sql_content

    def test_rls_policy_insert_service(self, sql_content: str) -> None:
        """holds_insert_service policy must be defined."""
        assert "holds_insert_service" in sql_content

    def test_rls_policy_update_service(self, sql_content: str) -> None:
        """holds_update_service policy must be defined."""
        assert "holds_update_service" in sql_content

    def test_rls_policy_delete_service(self, sql_content: str) -> None:
        """holds_delete_service policy must be defined."""
        assert "holds_delete_service" in sql_content

    # ── No spurious indexes ───────────────────────────────────────────────────

    def test_no_separate_idx_holds_route_id(self, sql_content: str) -> None:
        """No separate idx_holds_route_id — UNIQUE covers route-scoped lookups."""
        # There must be no CREATE INDEX statement for idx_holds_route_id
        assert not re.search(
            r"CREATE\s+INDEX\b[^;]*idx_holds_route_id", sql_content, re.IGNORECASE
        )

    def test_no_idx_holds_created_at(self, sql_content: str) -> None:
        """No idx_holds_created_at — no pagination use-case for holds."""
        assert "idx_holds_created_at" not in sql_content

    def test_id_is_uuid_primary_key_with_default(self, sql_content: str) -> None:
        """id must be UUID PRIMARY KEY with gen_random_uuid() default."""
        assert re.search(
            r"id\s+UUID\s+PRIMARY KEY DEFAULT gen_random_uuid\(\)", sql_content
        )

    def test_created_at_not_null(self, sql_content: str) -> None:
        """created_at must be TIMESTAMPTZ NOT NULL."""
        assert re.search(r"created_at\s+TIMESTAMPTZ\s+NOT NULL", sql_content)


# ---------------------------------------------------------------------------
# Layer 2 — Verifier unit tests (mocked Supabase)
# ---------------------------------------------------------------------------


class TestCreateHoldsTableVerifier:
    """Unit tests for verify_holds_table() with a mocked Supabase client."""

    def test_verify_success(self) -> None:
        """verify_holds_table returns success when all checks pass."""
        from scripts.migrations.create_holds_table import verify_holds_table

        client = _make_mock_client()
        result = verify_holds_table(client)

        assert result.success is True
        assert result.errors == []

    def test_verify_table_missing(self) -> None:
        """verify_holds_table fails with early exit when holds table is absent."""
        from scripts.migrations.create_holds_table import verify_holds_table

        client = _make_mock_client(table_exists=False)
        result = verify_holds_table(client)

        assert result.success is False
        assert len(result.errors) == 1
        assert "does not exist" in result.errors[0]

    def test_verify_column_missing(self) -> None:
        """verify_holds_table fails when a column is missing."""
        from scripts.migrations.create_holds_table import verify_holds_table

        columns_without_prob_unknown = [
            c
            for c in [
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
                "prob_volume",
                "created_at",
            ]
        ]
        client = _make_mock_client(columns=columns_without_prob_unknown)
        result = verify_holds_table(client)

        assert result.success is False
        assert any("prob_unknown" in e for e in result.errors)

    def test_verify_wrong_columns(self) -> None:
        """verify_holds_table fails when only wrong columns are present."""
        from scripts.migrations.create_holds_table import verify_holds_table

        client = _make_mock_client(columns=["id"])  # only id present
        result = verify_holds_table(client)

        assert result.success is False
        assert any("Missing columns" in e for e in result.errors)

    def test_verify_constraint_missing(self) -> None:
        """verify_holds_table fails when a CHECK constraint is absent."""
        from scripts.migrations.create_holds_table import verify_holds_table

        client = _make_mock_client(constraints=[])
        result = verify_holds_table(client)

        assert result.success is False
        assert any("Missing CHECK constraints" in e for e in result.errors)

    def test_verify_rls_policy_missing(self) -> None:
        """verify_holds_table fails when an RLS policy is absent."""
        from scripts.migrations.create_holds_table import verify_holds_table

        client = _make_mock_client(policies=["holds_select_public"])  # 3 missing
        result = verify_holds_table(client)

        assert result.success is False
        assert any("RLS policies" in e for e in result.errors)

    def test_dry_run_prints_sql(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--dry-run prints the SQL file contents to stdout."""
        from scripts.migrations.create_holds_table import main

        with patch("sys.argv", ["create_holds_table.py", "--dry-run"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "CREATE TABLE IF NOT EXISTS holds" in captured.out

    def test_main_exits_zero_on_success(self) -> None:
        """main() exits with code 0 when verification passes."""
        from scripts.migrations.create_holds_table import main

        mock_client = _make_mock_client()
        with patch("sys.argv", ["create_holds_table.py"]):
            with patch(
                "scripts.migrations.create_holds_table.get_supabase_client",
                return_value=mock_client,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 0

    def test_main_exits_one_on_failure(self) -> None:
        """main() exits with code 1 when verification fails."""
        from scripts.migrations.create_holds_table import main

        mock_client = _make_mock_client(table_exists=False)
        with patch("sys.argv", ["create_holds_table.py"]):
            with patch(
                "scripts.migrations.create_holds_table.get_supabase_client",
                return_value=mock_client,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    def test_no_trigger_check_for_holds(self) -> None:
        """Holds verifier must not check for an updated_at trigger."""
        from scripts.migrations.create_holds_table import _CONFIG

        assert _CONFIG.trigger_name is None

    def test_config_has_all_18_columns(self) -> None:
        """_CONFIG must list all 18 expected columns."""
        from scripts.migrations.create_holds_table import _CONFIG

        assert len(_CONFIG.expected_columns) == 18

    def test_config_has_all_15_check_constraints(self) -> None:
        """_CONFIG must list all 15 expected CHECK constraints."""
        from scripts.migrations.create_holds_table import _CONFIG

        assert len(_CONFIG.expected_check_constraints) == 15

    def test_config_has_all_4_rls_policies(self) -> None:
        """_CONFIG must list all 4 expected RLS policies."""
        from scripts.migrations.create_holds_table import _CONFIG

        assert len(_CONFIG.expected_rls_policies) == 4


# ---------------------------------------------------------------------------
# Layer 3 — Integration tests (skipped without Supabase credentials)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHoldsMigrationIntegration:
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
        """Verifier must confirm holds table is correctly set up in Supabase."""
        from scripts.migrations.create_holds_table import verify_holds_table
        from src.database.supabase_client import get_supabase_client

        client = get_supabase_client()
        result = verify_holds_table(client)

        assert result.success, f"Verification failed: {result.errors}"
