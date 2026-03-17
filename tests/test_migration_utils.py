"""Unit tests for scripts/migrations/_migration_utils.py.

Tests every public helper in the shared migration utilities module with
mocked Supabase clients so no database connection is required.

Test classes:
    TestVerificationResult      — VerificationResult.fail() behaviour
    TestCheckTableExists        — check_table_exists() helper
    TestGetColumns              — get_columns() helper
    TestCheckConstraints        — check_constraints() helper
    TestCheckTrigger            — check_trigger() helper
    TestCheckRlsPolicies        — check_rls_policies() helper
    TestVerifyTable             — verify_table() orchestrator
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain(data: list[dict[str, Any]]) -> MagicMock:
    """Return a mock query-builder chain that resolves to *data*.

    Args:
        data: Rows to return from ``.execute()``.

    Returns:
        Mock builder with ``.select()``, ``.eq()``, ``.execute()`` all
        chaining back to the same object.
    """
    result = MagicMock()
    result.data = data
    chain = MagicMock()
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.execute.return_value = result
    return chain


def _make_client(
    table_data: list[dict[str, Any]] | None = None,
    col_data: list[dict[str, Any]] | None = None,
    constraint_data: list[dict[str, Any]] | None = None,
    trigger_data: list[dict[str, Any]] | None = None,
    policy_data: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a generic mock Supabase client.

    Each positional dataset is keyed by the ``information_schema`` /
    ``pg_catalog`` view it represents.  Unspecified datasets default to
    empty lists (simulating "not found").

    Args:
        table_data: Rows for ``information_schema.tables``.
        col_data: Rows for ``information_schema.columns``.
        constraint_data: Rows for ``information_schema.table_constraints``.
        trigger_data: Rows for ``information_schema.triggers``.
        policy_data: Rows for ``pg_catalog.pg_policies``.

    Returns:
        Configured mock client.
    """
    client = MagicMock()

    mapping = {
        "information_schema.tables": table_data or [],
        "information_schema.columns": col_data or [],
        "information_schema.table_constraints": constraint_data or [],
        "information_schema.triggers": trigger_data or [],
        "pg_catalog.pg_policies": policy_data or [],
    }

    def _side_effect(name: str) -> MagicMock:
        return _make_chain(mapping.get(name, []))

    client.table.side_effect = _side_effect
    return client


# ---------------------------------------------------------------------------
# TestVerificationResult
# ---------------------------------------------------------------------------


class TestVerificationResult:
    """Tests for the VerificationResult dataclass."""

    def test_initial_state_is_success(self) -> None:
        """VerificationResult starts as success with no errors."""
        from scripts.migrations._migration_utils import VerificationResult

        result = VerificationResult()
        assert result.success is True
        assert result.errors == []

    def test_fail_sets_success_false(self) -> None:
        """fail() must set success to False."""
        from scripts.migrations._migration_utils import VerificationResult

        result = VerificationResult()
        result.fail("something broke")
        assert result.success is False

    def test_fail_appends_message(self) -> None:
        """fail() must append the message to errors."""
        from scripts.migrations._migration_utils import VerificationResult

        result = VerificationResult()
        result.fail("error one")
        assert "error one" in result.errors

    def test_fail_returns_self_for_chaining(self) -> None:
        """fail() must return self to enable method chaining."""
        from scripts.migrations._migration_utils import VerificationResult

        result = VerificationResult()
        returned = result.fail("chain test")
        assert returned is result

    def test_fail_chaining_accumulates_errors(self) -> None:
        """Chained fail() calls must accumulate all messages."""
        from scripts.migrations._migration_utils import VerificationResult

        result = VerificationResult()
        result.fail("error one").fail("error two").fail("error three")
        assert len(result.errors) == 3
        assert result.success is False

    def test_success_cannot_be_restored_after_fail(self) -> None:
        """Once fail() is called, success stays False even with no further calls."""
        from scripts.migrations._migration_utils import VerificationResult

        result = VerificationResult()
        result.fail("irreversible")
        # No way to undo a failure; confirm it stays False
        assert result.success is False
        assert len(result.errors) == 1

    def test_multiple_independent_results_are_isolated(self) -> None:
        """Two VerificationResult instances must not share state."""
        from scripts.migrations._migration_utils import VerificationResult

        r1 = VerificationResult()
        r2 = VerificationResult()
        r1.fail("only r1")
        assert r2.success is True
        assert r2.errors == []


# ---------------------------------------------------------------------------
# TestCheckTableExists
# ---------------------------------------------------------------------------


class TestCheckTableExists:
    """Tests for check_table_exists()."""

    def test_returns_true_when_table_found(self) -> None:
        """check_table_exists returns True when the table row is present."""
        from scripts.migrations._migration_utils import check_table_exists

        client = _make_client(
            table_data=[{"table_name": "holds"}],
        )
        assert check_table_exists(client, "holds") is True

    def test_returns_false_when_table_missing(self) -> None:
        """check_table_exists returns False when no rows are returned."""
        from scripts.migrations._migration_utils import check_table_exists

        client = _make_client(table_data=[])
        assert check_table_exists(client, "holds") is False

    def test_propagates_client_error(self) -> None:
        """check_table_exists lets exceptions bubble up to the caller."""
        from scripts.migrations._migration_utils import check_table_exists

        client = MagicMock()
        client.table.side_effect = RuntimeError("connection refused")
        with pytest.raises(RuntimeError, match="connection refused"):
            check_table_exists(client, "holds")


# ---------------------------------------------------------------------------
# TestGetColumns
# ---------------------------------------------------------------------------


class TestGetColumns:
    """Tests for get_columns()."""

    def test_returns_sorted_column_names(self) -> None:
        """get_columns returns column names in alphabetical order."""
        from scripts.migrations._migration_utils import get_columns

        client = _make_client(
            col_data=[
                {"column_name": "z_col"},
                {"column_name": "a_col"},
                {"column_name": "m_col"},
            ]
        )
        assert get_columns(client, "holds") == ["a_col", "m_col", "z_col"]

    def test_returns_empty_list_when_no_columns(self) -> None:
        """get_columns returns [] when the table has no columns (or is absent)."""
        from scripts.migrations._migration_utils import get_columns

        client = _make_client(col_data=[])
        assert get_columns(client, "holds") == []

    def test_propagates_client_error(self) -> None:
        """get_columns lets exceptions bubble up to the caller."""
        from scripts.migrations._migration_utils import get_columns

        client = MagicMock()
        client.table.side_effect = RuntimeError("network error")
        with pytest.raises(RuntimeError, match="network error"):
            get_columns(client, "holds")


# ---------------------------------------------------------------------------
# TestCheckConstraints
# ---------------------------------------------------------------------------


class TestCheckConstraints:
    """Tests for check_constraints()."""

    def test_returns_empty_when_all_found(self) -> None:
        """check_constraints returns [] when all expected constraints are present."""
        from scripts.migrations._migration_utils import check_constraints

        expected = frozenset({"holds_x_center_check", "holds_y_center_check"})
        client = _make_client(
            constraint_data=[
                {"constraint_name": "holds_x_center_check"},
                {"constraint_name": "holds_y_center_check"},
            ]
        )
        assert check_constraints(client, "holds", expected) == []

    def test_returns_error_when_one_missing(self) -> None:
        """check_constraints returns an error message when a constraint is absent."""
        from scripts.migrations._migration_utils import check_constraints

        expected = frozenset({"holds_x_center_check", "holds_y_center_check"})
        client = _make_client(
            constraint_data=[{"constraint_name": "holds_x_center_check"}]
        )
        errors = check_constraints(client, "holds", expected)
        assert len(errors) == 1
        assert "holds_y_center_check" in errors[0]

    def test_extra_db_constraints_are_ignored(self) -> None:
        """Extra constraints in the DB that are not expected cause no error."""
        from scripts.migrations._migration_utils import check_constraints

        expected = frozenset({"holds_x_center_check"})
        client = _make_client(
            constraint_data=[
                {"constraint_name": "holds_x_center_check"},
                {"constraint_name": "unexpected_constraint"},
            ]
        )
        assert check_constraints(client, "holds", expected) == []

    def test_returns_empty_for_empty_expected(self) -> None:
        """check_constraints with an empty frozenset always returns []."""
        from scripts.migrations._migration_utils import check_constraints

        client = _make_client(constraint_data=[])
        assert check_constraints(client, "holds", frozenset()) == []

    def test_error_message_includes_table_name(self) -> None:
        """Error message must contain the table name for clarity."""
        from scripts.migrations._migration_utils import check_constraints

        expected = frozenset({"holds_missing_check"})
        client = _make_client(constraint_data=[])
        errors = check_constraints(client, "holds", expected)
        assert "holds" in errors[0]


# ---------------------------------------------------------------------------
# TestCheckTrigger
# ---------------------------------------------------------------------------


class TestCheckTrigger:
    """Tests for check_trigger()."""

    def test_returns_true_when_trigger_found(self) -> None:
        """check_trigger returns True when the trigger row is present."""
        from scripts.migrations._migration_utils import check_trigger

        client = _make_client(trigger_data=[{"trigger_name": "set_routes_updated_at"}])
        assert check_trigger(client, "routes", "set_routes_updated_at") is True

    def test_returns_false_when_trigger_missing(self) -> None:
        """check_trigger returns False when no rows are returned."""
        from scripts.migrations._migration_utils import check_trigger

        client = _make_client(trigger_data=[])
        assert check_trigger(client, "routes", "set_routes_updated_at") is False

    def test_propagates_client_error(self) -> None:
        """check_trigger lets exceptions bubble up to the caller."""
        from scripts.migrations._migration_utils import check_trigger

        client = MagicMock()
        client.table.side_effect = RuntimeError("timeout")
        with pytest.raises(RuntimeError, match="timeout"):
            check_trigger(client, "routes", "set_routes_updated_at")


# ---------------------------------------------------------------------------
# TestCheckRlsPolicies
# ---------------------------------------------------------------------------


class TestCheckRlsPolicies:
    """Tests for check_rls_policies()."""

    def test_returns_empty_when_all_policies_found(self) -> None:
        """check_rls_policies returns [] when all 4 expected policies are present."""
        from scripts.migrations._migration_utils import check_rls_policies

        expected = frozenset(
            {
                "holds_select_public",
                "holds_insert_service",
                "holds_update_service",
                "holds_delete_service",
            }
        )
        client = _make_client(
            policy_data=[
                {"policyname": "holds_select_public"},
                {"policyname": "holds_insert_service"},
                {"policyname": "holds_update_service"},
                {"policyname": "holds_delete_service"},
            ]
        )
        assert check_rls_policies(client, "holds", expected) == []

    def test_returns_error_when_one_policy_missing(self) -> None:
        """check_rls_policies returns an error when a policy is absent."""
        from scripts.migrations._migration_utils import check_rls_policies

        expected = frozenset(
            {
                "holds_select_public",
                "holds_insert_service",
                "holds_update_service",
                "holds_delete_service",
            }
        )
        client = _make_client(
            policy_data=[
                {"policyname": "holds_select_public"},
                {"policyname": "holds_insert_service"},
                {"policyname": "holds_update_service"},
                # holds_delete_service missing
            ]
        )
        errors = check_rls_policies(client, "holds", expected)
        assert len(errors) == 1
        assert "holds_delete_service" in errors[0]

    def test_skips_check_when_expected_is_empty(self) -> None:
        """check_rls_policies returns [] immediately when expected is empty."""
        from scripts.migrations._migration_utils import check_rls_policies

        # Client must NOT be called when expected is empty
        client = MagicMock()
        result = check_rls_policies(client, "routes", frozenset())
        assert result == []
        client.table.assert_not_called()

    def test_error_message_includes_table_name(self) -> None:
        """Error message must reference the table name for clarity."""
        from scripts.migrations._migration_utils import check_rls_policies

        expected = frozenset({"holds_select_public"})
        client = _make_client(policy_data=[])
        errors = check_rls_policies(client, "holds", expected)
        assert "holds" in errors[0]

    def test_propagates_client_error(self) -> None:
        """check_rls_policies lets exceptions bubble up to the caller."""
        from scripts.migrations._migration_utils import check_rls_policies

        client = MagicMock()
        client.table.side_effect = RuntimeError("db error")
        with pytest.raises(RuntimeError, match="db error"):
            check_rls_policies(client, "holds", frozenset({"holds_select_public"}))


# ---------------------------------------------------------------------------
# TestVerifyTable
# ---------------------------------------------------------------------------


def _default_holds_config() -> Any:
    """Return a TableVerificationConfig for the holds table for use in tests."""
    from scripts.migrations._migration_utils import TableVerificationConfig

    return TableVerificationConfig(
        table_name="holds",
        expected_columns=("id", "route_id", "hold_id", "created_at"),
        trigger_name=None,
        expected_check_constraints=frozenset({"holds_hold_id_check"}),
        expected_rls_policies=frozenset(
            {
                "holds_select_public",
                "holds_insert_service",
                "holds_update_service",
                "holds_delete_service",
            }
        ),
    )


def _full_holds_client(
    table_name: str = "holds",
    columns: list[str] | None = None,
    constraints: list[str] | None = None,
    policies: list[str] | None = None,
) -> MagicMock:
    """Build a success-path mock client for the holds table.

    Args:
        table_name: Table name to report as present.
        columns: Column names to return. Defaults to the minimal test set.
        constraints: CHECK constraint names to return. Defaults to one check.
        policies: RLS policy names to return. Defaults to all 4 holds policies.

    Returns:
        Configured mock client.
    """
    if columns is None:
        columns = ["id", "route_id", "hold_id", "created_at"]
    if constraints is None:
        constraints = ["holds_hold_id_check"]
    if policies is None:
        policies = [
            "holds_select_public",
            "holds_insert_service",
            "holds_update_service",
            "holds_delete_service",
        ]
    return _make_client(
        table_data=[{"table_name": table_name}],
        col_data=[{"column_name": c} for c in columns],
        constraint_data=[{"constraint_name": c} for c in constraints],
        policy_data=[{"policyname": p} for p in policies],
    )


class TestVerifyTable:
    """Tests for the verify_table() orchestrator."""

    def test_success_path(self) -> None:
        """verify_table returns success when all checks pass."""
        from scripts.migrations._migration_utils import verify_table

        config = _default_holds_config()
        client = _full_holds_client()
        result = verify_table(client, config)
        assert result.success is True
        assert result.errors == []

    def test_table_missing_returns_early(self) -> None:
        """verify_table returns after the first error when the table is absent."""
        from scripts.migrations._migration_utils import verify_table

        config = _default_holds_config()
        client = _make_client(table_data=[])  # table absent
        result = verify_table(client, config)
        assert result.success is False
        assert len(result.errors) == 1
        assert "does not exist" in result.errors[0]

    def test_column_mismatch_is_reported(self) -> None:
        """verify_table reports missing columns."""
        from scripts.migrations._migration_utils import verify_table

        config = _default_holds_config()
        # Return only some columns — hold_id missing
        client = _full_holds_client(columns=["id", "route_id", "created_at"])
        result = verify_table(client, config)
        assert result.success is False
        assert any("hold_id" in e for e in result.errors)

    def test_constraint_mismatch_is_reported(self) -> None:
        """verify_table reports missing CHECK constraints."""
        from scripts.migrations._migration_utils import verify_table

        config = _default_holds_config()
        client = _full_holds_client(constraints=[])  # no constraints
        result = verify_table(client, config)
        assert result.success is False
        assert any("holds_hold_id_check" in e for e in result.errors)

    def test_policy_mismatch_is_reported(self) -> None:
        """verify_table reports missing RLS policies."""
        from scripts.migrations._migration_utils import verify_table

        config = _default_holds_config()
        client = _full_holds_client(
            policies=["holds_select_public"]  # 3 policies missing
        )
        result = verify_table(client, config)
        assert result.success is False
        assert any("RLS policies" in e for e in result.errors)

    def test_trigger_name_none_skips_trigger_check(self) -> None:
        """verify_table does not query information_schema.triggers when trigger_name is None."""
        from scripts.migrations._migration_utils import (
            TableVerificationConfig,
            verify_table,
        )

        config = TableVerificationConfig(
            table_name="holds",
            expected_columns=("id",),
            trigger_name=None,  # explicit skip
            expected_check_constraints=frozenset(),
            expected_rls_policies=frozenset(),
        )
        client = _make_client(
            table_data=[{"table_name": "holds"}],
            col_data=[{"column_name": "id"}],
        )
        result = verify_table(client, config)
        assert result.success is True
        # Confirm triggers view was never queried
        called_tables = [call.args[0] for call in client.table.call_args_list]
        assert "information_schema.triggers" not in called_tables

    def test_empty_rls_policies_skips_policy_check(self) -> None:
        """verify_table does not query pg_catalog.pg_policies when expected is empty."""
        from scripts.migrations._migration_utils import (
            TableVerificationConfig,
            verify_table,
        )

        config = TableVerificationConfig(
            table_name="routes",
            expected_columns=("id",),
            trigger_name=None,
            expected_check_constraints=frozenset(),
            expected_rls_policies=frozenset(),  # skip
        )
        client = _make_client(
            table_data=[{"table_name": "routes"}],
            col_data=[{"column_name": "id"}],
        )
        result = verify_table(client, config)
        assert result.success is True
        called_tables = [call.args[0] for call in client.table.call_args_list]
        assert "pg_catalog.pg_policies" not in called_tables

    def test_trigger_missing_is_reported(self) -> None:
        """verify_table reports a missing trigger when trigger_name is set."""
        from scripts.migrations._migration_utils import (
            TableVerificationConfig,
            verify_table,
        )

        config = TableVerificationConfig(
            table_name="routes",
            expected_columns=("id",),
            trigger_name="set_routes_updated_at",
            expected_check_constraints=frozenset(),
            expected_rls_policies=frozenset(),
        )
        client = _make_client(
            table_data=[{"table_name": "routes"}],
            col_data=[{"column_name": "id"}],
            trigger_data=[],  # trigger absent
        )
        result = verify_table(client, config)
        assert result.success is False
        assert any("set_routes_updated_at" in e for e in result.errors)

    def test_multiple_failures_all_reported(self) -> None:
        """verify_table accumulates all failures rather than stopping at the first."""
        from scripts.migrations._migration_utils import verify_table

        config = _default_holds_config()
        # Columns missing + constraints missing + policies missing
        client = _full_holds_client(columns=["id"], constraints=[], policies=[])
        result = verify_table(client, config)
        assert result.success is False
        # At minimum: one column error + one constraint error + one policy error
        assert len(result.errors) >= 3

    def test_table_existence_exception_returns_early(self) -> None:
        """verify_table handles a client error on the table check and returns early."""
        from scripts.migrations._migration_utils import verify_table

        config = _default_holds_config()
        client = MagicMock()
        client.table.side_effect = RuntimeError("timeout")
        result = verify_table(client, config)
        assert result.success is False
        assert len(result.errors) == 1
        assert "Failed to check table existence" in result.errors[0]

    def test_columns_exception_is_recorded(self) -> None:
        """verify_table records an error and continues when column check raises."""
        from scripts.migrations._migration_utils import (
            TableVerificationConfig,
            verify_table,
        )

        config = TableVerificationConfig(
            table_name="holds",
            expected_columns=("id",),
            trigger_name=None,
            expected_check_constraints=frozenset(),
            expected_rls_policies=frozenset(),
        )
        # Table check returns success; column check raises
        call_count = 0

        def _side_effect(name: str) -> Any:
            nonlocal call_count
            call_count += 1
            if name == "information_schema.tables":
                return _make_chain([{"table_name": "holds"}])
            if name == "information_schema.columns":
                raise RuntimeError("columns query failed")
            return _make_chain([])

        client = MagicMock()
        client.table.side_effect = _side_effect
        result = verify_table(client, config)
        assert result.success is False
        assert any("Failed to check columns" in e for e in result.errors)

    def test_constraints_exception_is_recorded(self) -> None:
        """verify_table records an error and continues when constraint check raises."""
        from scripts.migrations._migration_utils import (
            TableVerificationConfig,
            verify_table,
        )

        config = TableVerificationConfig(
            table_name="holds",
            expected_columns=("id",),
            trigger_name=None,
            expected_check_constraints=frozenset({"holds_hold_id_check"}),
            expected_rls_policies=frozenset(),
        )

        def _side_effect(name: str) -> Any:
            if name == "information_schema.tables":
                return _make_chain([{"table_name": "holds"}])
            if name == "information_schema.columns":
                return _make_chain([{"column_name": "id"}])
            if name == "information_schema.table_constraints":
                raise RuntimeError("constraint query failed")
            return _make_chain([])

        client = MagicMock()
        client.table.side_effect = _side_effect
        result = verify_table(client, config)
        assert result.success is False
        assert any("Failed to check constraints" in e for e in result.errors)

    def test_trigger_exception_is_recorded(self) -> None:
        """verify_table records an error and continues when trigger check raises."""
        from scripts.migrations._migration_utils import (
            TableVerificationConfig,
            verify_table,
        )

        config = TableVerificationConfig(
            table_name="routes",
            expected_columns=("id",),
            trigger_name="set_routes_updated_at",
            expected_check_constraints=frozenset(),
            expected_rls_policies=frozenset(),
        )

        def _side_effect(name: str) -> Any:
            if name == "information_schema.tables":
                return _make_chain([{"table_name": "routes"}])
            if name == "information_schema.columns":
                return _make_chain([{"column_name": "id"}])
            if name == "information_schema.triggers":
                raise RuntimeError("trigger query failed")
            return _make_chain([])

        client = MagicMock()
        client.table.side_effect = _side_effect
        result = verify_table(client, config)
        assert result.success is False
        assert any("Failed to check trigger" in e for e in result.errors)

    def test_rls_exception_is_recorded(self) -> None:
        """verify_table records an error when RLS policy check raises."""
        from scripts.migrations._migration_utils import (
            TableVerificationConfig,
            verify_table,
        )

        config = TableVerificationConfig(
            table_name="holds",
            expected_columns=("id",),
            trigger_name=None,
            expected_check_constraints=frozenset(),
            expected_rls_policies=frozenset({"holds_select_public"}),
        )

        def _side_effect(name: str) -> Any:
            if name == "information_schema.tables":
                return _make_chain([{"table_name": "holds"}])
            if name == "information_schema.columns":
                return _make_chain([{"column_name": "id"}])
            if name == "pg_catalog.pg_policies":
                raise RuntimeError("policies query failed")
            return _make_chain([])

        client = MagicMock()
        client.table.side_effect = _side_effect
        result = verify_table(client, config)
        assert result.success is False
        assert any("Failed to check RLS policies" in e for e in result.errors)


# ---------------------------------------------------------------------------
# TestSetupMigrationLogging
# ---------------------------------------------------------------------------


class TestSetupMigrationLogging:
    """Tests for setup_migration_logging()."""

    def test_creates_logs_directory(self, tmp_path: Any, monkeypatch: Any) -> None:
        """setup_migration_logging creates the logs/ directory if absent."""
        import logging

        from scripts.migrations import _migration_utils

        monkeypatch.setattr(_migration_utils, "PROJECT_ROOT", tmp_path)
        # Reset handlers so basicConfig can run (idempotent in test context)
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            _migration_utils.setup_migration_logging("test_verify.log")
            assert (tmp_path / "logs").is_dir()
            assert (tmp_path / "logs" / "test_verify.log").exists()
        finally:
            # Restore original handlers
            for h in root.handlers:
                h.close()
            root.handlers = original_handlers
