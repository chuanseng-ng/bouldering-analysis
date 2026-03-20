"""Tests for the predictions table migration (004_create_predictions_table.sql).

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
_SQL_FILE = _PROJECT_ROOT / "migrations" / "sql" / "004_create_predictions_table.sql"


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
    """Build a mock Supabase client for predictions verifier unit tests.

    Args:
        table_exists: Whether the predictions table query returns rows.
        columns: Column names to return.  Defaults to all 11 expected columns.
        constraints: CHECK constraint names.  Defaults to all 6 expected.
        policies: RLS policy names.  Defaults to all 4 expected policies.

    Returns:
        Configured mock client.
    """
    if columns is None:
        columns = [
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
        ]
    if constraints is None:
        constraints = [
            "predictions_estimator_type_check",
            "predictions_grade_check",
            "predictions_grade_index_check",
            "predictions_confidence_check",
            "predictions_difficulty_score_check",
            "predictions_uncertainty_check",
        ]
    if policies is None:
        policies = [
            "predictions_select_public",
            "predictions_insert_service",
            "predictions_update_service",
            "predictions_delete_service",
        ]

    client = MagicMock()
    table_data = [{"table_name": "predictions"}] if table_exists else []
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


class TestPredictionsMigrationSQL:
    """Offline checks — parse the SQL file without a database connection."""

    @pytest.fixture
    def sql_content(self) -> str:
        """Return the content of the predictions migration SQL file."""
        assert _SQL_FILE.exists(), f"SQL file not found: {_SQL_FILE}"
        return _SQL_FILE.read_text(encoding="utf-8")

    # ── File existence ─────────────────────────────────────────────────────

    def test_sql_file_exists(self) -> None:
        """Migration SQL file must exist at the expected path."""
        assert _SQL_FILE.exists()
        assert _SQL_FILE.stat().st_size > 0

    # ── Table DDL ─────────────────────────────────────────────────────────

    def test_creates_predictions_table(self, sql_content: str) -> None:
        """SQL must create the predictions table with IF NOT EXISTS."""
        assert "CREATE TABLE IF NOT EXISTS predictions" in sql_content

    def test_idempotent_table_creation(self, sql_content: str) -> None:
        """CREATE TABLE must use IF NOT EXISTS for idempotency."""
        assert "IF NOT EXISTS" in sql_content

    def test_no_moddatetime_extension(self, sql_content: str) -> None:
        """SQL must NOT enable moddatetime — predictions table uses no trigger."""
        assert "CREATE EXTENSION IF NOT EXISTS moddatetime" not in sql_content

    # ── Primary key ────────────────────────────────────────────────────────

    def test_id_is_uuid_primary_key_with_default(self, sql_content: str) -> None:
        """id must be UUID PRIMARY KEY with gen_random_uuid() default."""
        assert re.search(
            r"id\s+UUID\s+PRIMARY KEY DEFAULT gen_random_uuid\(\)", sql_content
        )

    # ── Foreign key ────────────────────────────────────────────────────────

    def test_route_id_column_present(self, sql_content: str) -> None:
        """route_id column must be present."""
        assert "route_id" in sql_content

    def test_on_delete_cascade(self, sql_content: str) -> None:
        """route_id FK must use ON DELETE CASCADE."""
        assert "ON DELETE CASCADE" in sql_content

    def test_fk_references_routes_id(self, sql_content: str) -> None:
        """route_id FK must reference routes(id)."""
        assert "REFERENCES routes(id)" in sql_content

    # ── Column presence ────────────────────────────────────────────────────

    def test_column_estimator_type(self, sql_content: str) -> None:
        """estimator_type column must be present."""
        assert "estimator_type" in sql_content

    def test_column_grade(self, sql_content: str) -> None:
        """grade column must be present."""
        assert re.search(r"\bgrade\b", sql_content)

    def test_column_grade_index(self, sql_content: str) -> None:
        """grade_index column must be present."""
        assert "grade_index" in sql_content

    def test_column_confidence(self, sql_content: str) -> None:
        """confidence column must be present."""
        assert "confidence" in sql_content

    def test_column_difficulty_score(self, sql_content: str) -> None:
        """difficulty_score column must be present."""
        assert "difficulty_score" in sql_content

    def test_column_uncertainty(self, sql_content: str) -> None:
        """uncertainty column must be present."""
        assert "uncertainty" in sql_content

    def test_column_explanation(self, sql_content: str) -> None:
        """explanation column must be present."""
        assert "explanation" in sql_content

    def test_column_model_version(self, sql_content: str) -> None:
        """model_version column must be present."""
        assert "model_version" in sql_content

    def test_column_predicted_at(self, sql_content: str) -> None:
        """predicted_at column must be present."""
        assert "predicted_at" in sql_content

    # ── Column types and nullability ───────────────────────────────────────

    def test_confidence_is_not_null(self, sql_content: str) -> None:
        """confidence must be NOT NULL — always computed by both estimators."""
        assert re.search(r"confidence\s+FLOAT\s+NOT NULL", sql_content)

    def test_explanation_is_jsonb(self, sql_content: str) -> None:
        """explanation must be JSONB (not TEXT) for structured queryability."""
        assert re.search(r"explanation\s+JSONB", sql_content)
        assert not re.search(r"explanation\s+TEXT", sql_content)

    def test_explanation_is_nullable(self, sql_content: str) -> None:
        """explanation must be nullable — explanation generation may be skipped."""
        assert not re.search(r"explanation\s+JSONB\s+NOT NULL", sql_content)

    def test_uncertainty_is_nullable(self, sql_content: str) -> None:
        """uncertainty must be nullable — reserved for future estimator output."""
        assert not re.search(r"uncertainty\s+FLOAT\s+NOT NULL", sql_content)

    def test_model_version_is_nullable(self, sql_content: str) -> None:
        """model_version must be nullable — NULL for heuristic, set for ML."""
        assert not re.search(r"model_version\s+VARCHAR\S*\s+NOT NULL", sql_content)

    def test_predicted_at_is_timestamptz_not_null(self, sql_content: str) -> None:
        """predicted_at must be TIMESTAMPTZ NOT NULL."""
        assert re.search(r"predicted_at\s+TIMESTAMPTZ\s+NOT NULL", sql_content)

    # ── CHECK constraints ──────────────────────────────────────────────────

    def test_grade_check_contains_all_18_values(self, sql_content: str) -> None:
        """grade CHECK must include all 18 V-scale values (V0–V17)."""
        assert all(f"'{g}'" in sql_content for g in V_GRADES)

    def test_grade_index_check_constraint(self, sql_content: str) -> None:
        """grade_index must have a CHECK BETWEEN 0 AND 17."""
        assert re.search(
            r"grade_index\s+INT\s+NOT NULL\s+CHECK\s*\(\s*grade_index\s+BETWEEN\s+0\s+AND\s+17\s*\)",
            sql_content,
        )

    def test_estimator_type_check_constraint(self, sql_content: str) -> None:
        """estimator_type must have a CHECK constraint limiting to known estimators."""
        assert re.search(
            r"estimator_type\s+IN\s*\(\s*'heuristic'\s*,\s*'ml'\s*\)",
            sql_content,
        )

    def test_confidence_check_constraint(self, sql_content: str) -> None:
        """confidence must have a CHECK BETWEEN 0 AND 1."""
        assert re.search(
            r"confidence\s+BETWEEN\s+0\s+AND\s+1",
            sql_content,
        )

    def test_difficulty_score_check_constraint(self, sql_content: str) -> None:
        """difficulty_score must have a CHECK BETWEEN 0 AND 1."""
        assert re.search(
            r"difficulty_score\s+BETWEEN\s+0\s+AND\s+1",
            sql_content,
        )

    def test_uncertainty_check_constraint(self, sql_content: str) -> None:
        """uncertainty must have a CHECK BETWEEN 0 AND 1."""
        assert re.search(
            r"uncertainty\s+BETWEEN\s+0\s+AND\s+1",
            sql_content,
        )

    # ── Index ──────────────────────────────────────────────────────────────

    def test_compound_index_present(self, sql_content: str) -> None:
        """Compound index (route_id, predicted_at DESC) must be defined."""
        assert "idx_predictions_route_id_predicted_at" in sql_content

    def test_index_covers_route_id_and_predicted_at(self, sql_content: str) -> None:
        """Compound index must cover both route_id and predicted_at columns."""
        assert re.search(
            r"idx_predictions_route_id_predicted_at\s+ON\s+predictions\s*\(\s*route_id\s*,\s*predicted_at",
            sql_content,
        )

    # ── Design decisions ───────────────────────────────────────────────────

    def test_no_unique_on_route_id(self, sql_content: str) -> None:
        """route_id must NOT have a UNIQUE constraint — multiple predictions per route allowed."""
        # No inline UNIQUE on the route_id column
        assert not re.search(r"route_id\s+UUID\s+NOT NULL\s+UNIQUE", sql_content)
        # No table-level UNIQUE (route_id) constraint
        assert not re.search(r"UNIQUE\s*\(\s*route_id\s*\)", sql_content)

    def test_no_updated_at_column(self, sql_content: str) -> None:
        """updated_at must NOT be defined — predictions are append-only."""
        assert not re.search(r"updated_at\s+TIMESTAMPTZ", sql_content)

    def test_no_trigger_reference(self, sql_content: str) -> None:
        """No CREATE TRIGGER must be defined — predictions are append-only."""
        assert "CREATE TRIGGER" not in sql_content

    def test_append_only_no_delete_before_insert(self, sql_content: str) -> None:
        """SQL must not contain DELETE FROM — predictions are never overwritten."""
        assert "DELETE FROM" not in sql_content

    # ── Idempotency ────────────────────────────────────────────────────────

    def test_drop_policy_if_exists_select(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for predictions_select_public."""
        assert "DROP POLICY IF EXISTS predictions_select_public" in sql_content

    def test_drop_policy_if_exists_insert(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for predictions_insert_service."""
        assert "DROP POLICY IF EXISTS predictions_insert_service" in sql_content

    def test_drop_policy_if_exists_update(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for predictions_update_service."""
        assert "DROP POLICY IF EXISTS predictions_update_service" in sql_content

    def test_drop_policy_if_exists_delete(self, sql_content: str) -> None:
        """DROP POLICY IF EXISTS guard must be present for predictions_delete_service."""
        assert "DROP POLICY IF EXISTS predictions_delete_service" in sql_content

    # ── RLS ───────────────────────────────────────────────────────────────

    def test_rls_enabled(self, sql_content: str) -> None:
        """SQL must enable Row Level Security on the predictions table."""
        assert "ALTER TABLE predictions ENABLE ROW LEVEL SECURITY" in sql_content

    def test_rls_policy_select_public(self, sql_content: str) -> None:
        """predictions_select_public policy must be defined."""
        assert "predictions_select_public" in sql_content

    def test_rls_policy_insert_service(self, sql_content: str) -> None:
        """predictions_insert_service policy must be defined."""
        assert "predictions_insert_service" in sql_content

    def test_rls_policy_update_service(self, sql_content: str) -> None:
        """predictions_update_service policy must be defined."""
        assert "predictions_update_service" in sql_content

    def test_rls_policy_delete_service(self, sql_content: str) -> None:
        """predictions_delete_service policy must be defined."""
        assert "predictions_delete_service" in sql_content


# ---------------------------------------------------------------------------
# Layer 2 — Verifier unit tests (mocked Supabase)
# ---------------------------------------------------------------------------


class TestCreatePredictionsTableVerifier:
    """Unit tests for verify_predictions_table() with a mocked Supabase client."""

    def test_verify_success(self) -> None:
        """verify_predictions_table returns success when all checks pass."""
        from scripts.migrations.create_predictions_table import verify_predictions_table

        client = _make_mock_client()
        result = verify_predictions_table(client)

        assert result.success is True
        assert result.errors == []

    def test_verify_table_missing(self) -> None:
        """verify_predictions_table fails with early exit when predictions table is absent."""
        from scripts.migrations.create_predictions_table import verify_predictions_table

        client = _make_mock_client(table_exists=False)
        result = verify_predictions_table(client)

        assert result.success is False
        assert len(result.errors) == 1
        assert "does not exist" in result.errors[0]

    def test_verify_column_missing(self) -> None:
        """verify_predictions_table fails when a column is missing."""
        from scripts.migrations.create_predictions_table import verify_predictions_table

        client = _make_mock_client(
            columns=[
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
                # predicted_at omitted
            ]
        )
        result = verify_predictions_table(client)

        assert result.success is False
        assert any("predicted_at" in e for e in result.errors)

    def test_verify_wrong_columns(self) -> None:
        """verify_predictions_table fails when only wrong columns are present."""
        from scripts.migrations.create_predictions_table import verify_predictions_table

        client = _make_mock_client(columns=["id"])  # only id present
        result = verify_predictions_table(client)

        assert result.success is False
        assert any("Missing columns" in e for e in result.errors)

    def test_verify_constraint_missing(self) -> None:
        """verify_predictions_table fails when a CHECK constraint is absent."""
        from scripts.migrations.create_predictions_table import verify_predictions_table

        client = _make_mock_client(
            constraints=["predictions_confidence_check"]  # 5 missing
        )
        result = verify_predictions_table(client)

        assert result.success is False
        assert any("CHECK constraints" in e for e in result.errors)

    def test_verify_rls_policy_missing(self) -> None:
        """verify_predictions_table fails when an RLS policy is absent."""
        from scripts.migrations.create_predictions_table import verify_predictions_table

        client = _make_mock_client(policies=["predictions_select_public"])  # 3 missing
        result = verify_predictions_table(client)

        assert result.success is False
        assert any("RLS policies" in e for e in result.errors)

    def test_dry_run_prints_sql(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--dry-run prints the SQL file contents to stdout."""
        from scripts.migrations.create_predictions_table import main

        with patch("sys.argv", ["create_predictions_table.py", "--dry-run"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "CREATE TABLE IF NOT EXISTS predictions" in captured.out

    def test_main_exits_zero_on_success(self) -> None:
        """main() exits with code 0 when verification passes."""
        from scripts.migrations.create_predictions_table import main

        mock_client = _make_mock_client()
        with patch("sys.argv", ["create_predictions_table.py"]):
            with patch(
                "scripts.migrations.create_predictions_table.get_supabase_client",
                return_value=mock_client,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 0

    def test_main_exits_one_on_failure(self) -> None:
        """main() exits with code 1 when verification fails."""
        from scripts.migrations.create_predictions_table import main

        mock_client = _make_mock_client(table_exists=False)
        with patch("sys.argv", ["create_predictions_table.py"]):
            with patch(
                "scripts.migrations.create_predictions_table.get_supabase_client",
                return_value=mock_client,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    def test_main_exits_one_on_connection_failure(self) -> None:
        """main() exits with code 1 when Supabase connection raises SupabaseClientError."""
        from scripts.migrations.create_predictions_table import main
        from src.database.supabase_client import SupabaseClientError

        with patch("sys.argv", ["create_predictions_table.py"]):
            with patch(
                "scripts.migrations.create_predictions_table.get_supabase_client",
                side_effect=SupabaseClientError("connection refused"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 1

    def test_dry_run_exits_one_when_sql_file_missing(self, tmp_path: Path) -> None:
        """--dry-run exits with code 1 when the SQL file does not exist."""
        from scripts.migrations import create_predictions_table

        nonexistent = tmp_path / "missing.sql"
        with patch("sys.argv", ["create_predictions_table.py", "--dry-run"]):
            with patch.object(create_predictions_table, "_SQL_FILE", nonexistent):
                with pytest.raises(SystemExit) as exc_info:
                    create_predictions_table.main()

        assert exc_info.value.code == 1

    def test_no_trigger_check_for_predictions(self) -> None:
        """Predictions verifier must not check for an updated_at trigger."""
        from scripts.migrations.create_predictions_table import _CONFIG

        assert _CONFIG.trigger_name is None

    def test_config_has_all_11_columns(self) -> None:
        """_CONFIG must list all 11 expected columns."""
        from scripts.migrations.create_predictions_table import _CONFIG

        assert len(_CONFIG.expected_columns) == 11

    def test_config_has_all_6_check_constraints(self) -> None:
        """_CONFIG must list all 6 expected CHECK constraints."""
        from scripts.migrations.create_predictions_table import _CONFIG

        assert len(_CONFIG.expected_check_constraints) == 6

    def test_config_has_all_4_rls_policies(self) -> None:
        """_CONFIG must list all 4 expected RLS policies."""
        from scripts.migrations.create_predictions_table import _CONFIG

        assert len(_CONFIG.expected_rls_policies) == 4


# ---------------------------------------------------------------------------
# Layer 3 — Integration tests (skipped without Supabase credentials)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPredictionsMigrationIntegration:
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
        """Verifier must confirm predictions table is correctly set up in Supabase."""
        from scripts.migrations.create_predictions_table import verify_predictions_table
        from src.database.supabase_client import get_supabase_client

        client = get_supabase_client()
        result = verify_predictions_table(client)

        assert result.success, f"Verification failed: {result.errors}"
