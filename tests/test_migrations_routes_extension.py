"""Tests for migration 006: extend routes table (PR-10.2).

Layer 1: Offline SQL parsing (no DB connection).
"""

from pathlib import Path


_SQL_FILE = (
    Path(__file__).parent.parent / "migrations" / "sql" / "006_extend_routes_table.sql"
)


def _read_sql() -> str:
    return _SQL_FILE.read_text(encoding="utf-8")


class TestSqlFileExists:
    """The migration SQL file must be present."""

    def test_sql_file_exists(self) -> None:
        """Migration file 006_extend_routes_table.sql must exist."""
        assert _SQL_FILE.exists(), f"Missing: {_SQL_FILE}"


class TestSqlAddColumns:
    """The SQL must add start_hold_ids and finish_hold_ids columns."""

    def test_adds_start_hold_ids_column(self) -> None:
        """SQL must add start_hold_ids INTEGER[] column."""
        sql = _read_sql()
        assert "start_hold_ids" in sql
        assert "INTEGER[]" in sql

    def test_adds_finish_hold_ids_column(self) -> None:
        """SQL must add finish_hold_ids INTEGER[] column."""
        sql = _read_sql()
        assert "finish_hold_ids" in sql

    def test_alter_table_routes(self) -> None:
        """SQL must target the routes table."""
        sql = _read_sql()
        assert "ALTER TABLE routes" in sql

    def test_uses_if_not_exists(self) -> None:
        """SQL must use ADD COLUMN IF NOT EXISTS for idempotency."""
        sql = _read_sql()
        assert "IF NOT EXISTS" in sql


class TestSqlIndex:
    """The SQL must create a partial index for annotated routes."""

    def test_creates_index(self) -> None:
        """SQL must create idx_routes_constraints_set index."""
        sql = _read_sql()
        assert "idx_routes_constraints_set" in sql

    def test_index_is_partial(self) -> None:
        """Index must be partial (WHERE clause)."""
        sql = _read_sql()
        assert "WHERE" in sql
        assert "start_hold_ids IS NOT NULL" in sql
        assert "finish_hold_ids IS NOT NULL" in sql

    def test_index_uses_if_not_exists(self) -> None:
        """CREATE INDEX must use IF NOT EXISTS."""
        sql = _read_sql()
        assert "CREATE INDEX IF NOT EXISTS" in sql
