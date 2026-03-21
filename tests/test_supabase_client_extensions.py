"""Tests for Supabase client extension functions (PR-10.1).

Covers update_record, select_records, and delete_records.
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.database.supabase_client import (
    SupabaseClientError,
    delete_records,
    get_supabase_client,
    select_records,
    update_record,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_UUID1 = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_UUID2 = "b2c3d4e5-f6a7-8901-bcde-f12345678901"


def _make_client(table_data: Any = None) -> tuple[MagicMock, MagicMock]:
    """Return (mock_client, mock_table) with chained query result."""
    get_supabase_client.cache_clear()
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_result = MagicMock()
    mock_result.data = table_data

    # Chain: .table().update().eq().execute() / .select()...
    mock_table.update.return_value.eq.return_value.execute.return_value = mock_result
    mock_table.select.return_value.eq.return_value.execute.return_value = mock_result
    mock_table.select.return_value.execute.return_value = mock_result
    mock_table.select.return_value.order.return_value.execute.return_value = mock_result
    mock_table.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_result
    mock_table.select.return_value.order.return_value.range.return_value.execute.return_value = mock_result
    mock_table.select.return_value.limit.return_value.execute.return_value = mock_result
    mock_table.select.return_value.range.return_value.execute.return_value = mock_result
    mock_table.delete.return_value.eq.return_value.execute.return_value = mock_result

    mock_client.table.return_value = mock_table
    return mock_client, mock_table


def _patch_client(mock_client: MagicMock) -> Any:
    """Context manager that patches get_supabase_client to return mock_client."""
    return patch(
        "src.database.supabase_client.get_supabase_client", return_value=mock_client
    )


# ---------------------------------------------------------------------------
# TestUpdateRecord
# ---------------------------------------------------------------------------


class TestUpdateRecord:
    """Tests for update_record function."""

    def test_update_record_success(self) -> None:
        """update_record returns the updated row dict."""
        updated_row: dict[str, Any] = {"id": _UUID1, "status": "done"}
        mock_client, _ = _make_client(table_data=[updated_row])

        with _patch_client(mock_client):
            result = update_record("routes", _UUID1, {"status": "done"})

        assert result == updated_row

    def test_update_record_calls_update_eq(self) -> None:
        """update_record chains .update(data).eq('id', record_id)."""
        updated_row: dict[str, Any] = {"id": _UUID1, "status": "done"}
        mock_client, mock_table = _make_client(table_data=[updated_row])

        with _patch_client(mock_client):
            update_record("routes", _UUID1, {"status": "done"})

        mock_table.update.assert_called_once_with({"status": "done"})
        mock_table.update.return_value.eq.assert_called_once_with("id", _UUID1)

    def test_update_record_empty_id_raises(self) -> None:
        """update_record raises SupabaseClientError for empty record_id."""
        with pytest.raises(SupabaseClientError, match="Record ID cannot be empty"):
            update_record("routes", "", {"status": "done"})

    def test_update_record_invalid_uuid_raises(self) -> None:
        """update_record raises SupabaseClientError for non-UUID record_id."""
        with pytest.raises(SupabaseClientError, match="Invalid record ID"):
            update_record("routes", "not-a-uuid", {"status": "done"})

    def test_update_record_empty_data_raises(self) -> None:
        """update_record raises SupabaseClientError when data is empty."""
        with pytest.raises(
            SupabaseClientError, match="Data dictionary cannot be empty"
        ):
            update_record("routes", "a1b2c3d4-e5f6-7890-abcd-ef1234567890", {})

    def test_update_record_not_found_raises(self) -> None:
        """update_record raises SupabaseClientError when no row is returned."""
        mock_client, _ = _make_client(table_data=[])

        with _patch_client(mock_client):
            with pytest.raises(SupabaseClientError, match="not found in table"):
                update_record("routes", _UUID1, {"status": "done"})

    def test_update_record_unknown_table_raises(self) -> None:
        """update_record raises SupabaseClientError for unknown table name."""
        with pytest.raises(SupabaseClientError, match="Unknown table"):
            update_record("unknown", _UUID1, {"x": 1})

    def test_update_record_supabase_error_wrapped(self) -> None:
        """Supabase exceptions are wrapped in SupabaseClientError."""
        mock_client = MagicMock()
        mock_client.table.return_value.update.side_effect = RuntimeError("network")

        with _patch_client(mock_client):
            with pytest.raises(SupabaseClientError, match="Failed to update record"):
                update_record("routes", _UUID1, {"status": "done"})


# ---------------------------------------------------------------------------
# TestSelectRecords
# ---------------------------------------------------------------------------


class TestSelectRecords:
    """Tests for select_records function."""

    def test_select_records_no_filters(self) -> None:
        """select_records with no filters returns all rows."""
        rows: list[dict[str, Any]] = [{"id": "r1"}, {"id": "r2"}]
        mock_client, mock_table = _make_client(table_data=rows)
        mock_table.select.return_value.execute.return_value.data = rows

        with _patch_client(mock_client):
            result = select_records("routes")

        assert result == rows

    def test_select_records_with_filter(self) -> None:
        """select_records applies .eq() for each filter key."""
        rows: list[dict[str, Any]] = [{"id": "r1", "status": "done"}]
        mock_client, mock_table = _make_client(table_data=rows)
        # eq chain: .select().eq().execute()
        mock_table.select.return_value.eq.return_value.execute.return_value.data = rows

        with _patch_client(mock_client):
            result = select_records("routes", filters={"status": "done"})

        assert result == rows
        mock_table.select.return_value.eq.assert_called_with("status", "done")

    def test_select_records_order_by_desc(self) -> None:
        """select_records calls .order(col, desc=True) for '*.desc' syntax."""
        rows: list[dict[str, Any]] = [{"id": "r1"}]
        mock_client, mock_table = _make_client(table_data=rows)
        mock_table.select.return_value.order.return_value.execute.return_value.data = (
            rows
        )

        with _patch_client(mock_client):
            result = select_records("routes", order_by="created_at.desc")

        assert result == rows
        mock_table.select.return_value.order.assert_called_once_with(
            "created_at", desc=True
        )

    def test_select_records_order_by_asc(self) -> None:
        """select_records calls .order(col, desc=False) for '*.asc' syntax."""
        rows: list[dict[str, Any]] = []
        mock_client, mock_table = _make_client(table_data=rows)
        mock_table.select.return_value.order.return_value.execute.return_value.data = (
            rows
        )

        with _patch_client(mock_client):
            select_records("holds", order_by="hold_id.asc")

        mock_table.select.return_value.order.assert_called_once_with(
            "hold_id", desc=False
        )

    def test_select_records_with_limit_only(self) -> None:
        """select_records applies .limit() when limit specified without offset."""
        rows: list[dict[str, Any]] = [{"id": "r1"}]
        mock_client, mock_table = _make_client(table_data=rows)
        mock_table.select.return_value.limit.return_value.execute.return_value.data = (
            rows
        )

        with _patch_client(mock_client):
            result = select_records("routes", limit=5)

        assert result == rows
        mock_table.select.return_value.limit.assert_called_once_with(5)

    def test_select_records_with_limit_and_offset(self) -> None:
        """select_records applies .range(offset, offset+limit-1) when both set."""
        rows: list[dict[str, Any]] = []
        mock_client, mock_table = _make_client(table_data=rows)
        mock_table.select.return_value.range.return_value.execute.return_value.data = (
            rows
        )

        with _patch_client(mock_client):
            select_records("routes", limit=10, offset=20)

        mock_table.select.return_value.range.assert_called_once_with(20, 29)

    def test_select_records_empty_result(self) -> None:
        """select_records returns empty list when data is empty."""
        mock_client, mock_table = _make_client(table_data=[])
        mock_table.select.return_value.execute.return_value.data = []

        with _patch_client(mock_client):
            result = select_records("routes")

        assert result == []

    def test_select_records_none_data_returns_empty(self) -> None:
        """select_records returns empty list when data is None."""
        mock_client, mock_table = _make_client(table_data=None)
        mock_table.select.return_value.execute.return_value.data = None

        with _patch_client(mock_client):
            result = select_records("routes")

        assert result == []

    def test_select_records_unknown_table_raises(self) -> None:
        """select_records raises SupabaseClientError for unknown table name."""
        with pytest.raises(SupabaseClientError, match="Unknown table"):
            select_records("unknown_table")

    def test_select_records_supabase_error_wrapped(self) -> None:
        """Supabase exceptions are wrapped in SupabaseClientError."""
        mock_client = MagicMock()
        mock_client.table.return_value.select.side_effect = RuntimeError("db down")

        with _patch_client(mock_client):
            with pytest.raises(SupabaseClientError, match="Failed to select records"):
                select_records("routes")

    def test_select_records_custom_columns(self) -> None:
        """select_records passes columns string to .select()."""
        rows: list[dict[str, Any]] = [{"id": "r1", "status": "done"}]
        mock_client, mock_table = _make_client(table_data=rows)
        mock_table.select.return_value.execute.return_value.data = rows

        with _patch_client(mock_client):
            select_records("routes", columns="id, status")

        mock_table.select.assert_called_once_with("id, status")


# ---------------------------------------------------------------------------
# TestDeleteRecords
# ---------------------------------------------------------------------------


class TestDeleteRecords:
    """Tests for delete_records function."""

    def test_delete_records_success_returns_count(self) -> None:
        """delete_records returns the number of deleted rows."""
        deleted_rows: list[dict[str, Any]] = [{"id": "r1"}, {"id": "r2"}]
        mock_client, _ = _make_client(table_data=deleted_rows)

        with _patch_client(mock_client):
            count = delete_records("features", {"route_id": _UUID1})

        assert count == 2

    def test_delete_records_calls_eq_per_filter(self) -> None:
        """delete_records chains .eq() for each filter key."""
        deleted_rows: list[dict[str, Any]] = [{"id": "r1"}]
        mock_client, mock_table = _make_client(table_data=deleted_rows)

        with _patch_client(mock_client):
            delete_records("features", {"route_id": _UUID1})

        mock_table.delete.assert_called_once_with()
        mock_table.delete.return_value.eq.assert_called_with("route_id", _UUID1)

    def test_delete_records_empty_filters_raises(self) -> None:
        """delete_records raises SupabaseClientError when filters is empty."""
        with pytest.raises(SupabaseClientError, match="filters cannot be empty"):
            delete_records("features", {})

    def test_delete_records_unknown_table_raises(self) -> None:
        """delete_records raises SupabaseClientError for unknown table name."""
        with pytest.raises(SupabaseClientError, match="Unknown table"):
            delete_records("unknown", {"id": "x"})

    def test_delete_records_no_data_returns_zero(self) -> None:
        """delete_records returns 0 when data is None/empty."""
        mock_client, _ = _make_client(table_data=None)

        with _patch_client(mock_client):
            count = delete_records("features", {"route_id": _UUID1})

        assert count == 0

    def test_delete_records_supabase_error_wrapped(self) -> None:
        """Supabase exceptions are wrapped in SupabaseClientError."""
        mock_client = MagicMock()
        mock_client.table.return_value.delete.side_effect = RuntimeError("network")

        with _patch_client(mock_client):
            with pytest.raises(SupabaseClientError, match="Failed to delete records"):
                delete_records("features", {"route_id": _UUID1})
