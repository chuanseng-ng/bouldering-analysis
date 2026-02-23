"""Tests for Supabase client management."""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest

from src.config import get_settings_override
from src.database.supabase_client import (
    SupabaseClientError,
    _KNOWN_BUCKETS,
    delete_from_storage,
    get_storage_url,
    get_supabase_client,
    insert_record,
    list_storage_files,
    reset_supabase_client_cache,
    select_record_by_id,
    upload_to_storage,
)


class TestGetSupabaseClient:
    """Tests for get_supabase_client function."""

    def test_get_supabase_client_requires_url(self) -> None:
        """Client creation should fail without SUPABASE_URL."""
        get_supabase_client.cache_clear()

        with patch("src.database.supabase_client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = get_settings_override(
                {"supabase_url": "", "supabase_key": "test-key"}
            )

            with pytest.raises(
                SupabaseClientError,
                match="SUPABASE_URL environment variable is required",
            ):
                get_supabase_client()

    def test_get_supabase_client_requires_key(self) -> None:
        """Client creation should fail without SUPABASE_KEY."""
        get_supabase_client.cache_clear()

        with patch("src.database.supabase_client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = get_settings_override(
                {"supabase_url": "https://test.supabase.co", "supabase_key": ""}
            )

            with pytest.raises(
                SupabaseClientError,
                match="SUPABASE_KEY environment variable is required",
            ):
                get_supabase_client()

    def test_get_supabase_client_creates_client_successfully(self) -> None:
        """Client should be created with valid credentials and ClientOptions."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key-12345",
                }
            )
            mock_create_client.return_value = mock_client

            client = get_supabase_client()

            assert client is mock_client
            mock_create_client.assert_called_once_with(
                "https://test.supabase.co", "test-key-12345", options=ANY
            )

    def test_get_supabase_client_caches_result(self) -> None:
        """Client should be cached and reused."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            # First call should create client
            client1 = get_supabase_client()

            # Second call should return cached client
            client2 = get_supabase_client()

            assert client1 is client2
            # create_client should only be called once
            mock_create_client.assert_called_once()

    def test_get_supabase_client_handles_creation_error(self) -> None:
        """Client creation errors should be wrapped in SupabaseClientError."""
        get_supabase_client.cache_clear()

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.side_effect = Exception("Connection failed")

            with pytest.raises(
                SupabaseClientError, match="Failed to create Supabase client"
            ):
                get_supabase_client()

    def test_reset_supabase_client_cache_forces_new_client(self) -> None:
        """reset_supabase_client_cache should clear the lru_cache."""
        get_supabase_client.cache_clear()

        mock_client1 = MagicMock()
        mock_client2 = MagicMock()

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client1
            client1 = get_supabase_client()

            # Clear cache and swap the mock return value
            reset_supabase_client_cache()
            mock_create_client.return_value = mock_client2
            client2 = get_supabase_client()

            assert client1 is not client2
            assert mock_create_client.call_count == 2


class TestUploadToStorage:
    """Tests for upload_to_storage function."""

    def test_upload_to_storage_success(self) -> None:
        """File upload should succeed with valid parameters."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_bucket = MagicMock()

        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.get_public_url.return_value = (
            "https://test.supabase.co/storage/v1/object/public/bucket/file.jpg"
        )

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            file_data = b"test image data"
            url = upload_to_storage(
                "route-images", "path/to/file.jpg", file_data, "image/jpeg"
            )

            assert (
                url
                == "https://test.supabase.co/storage/v1/object/public/bucket/file.jpg"
            )
            mock_client.storage.from_.assert_called_with("route-images")
            mock_bucket.upload.assert_called_once_with(
                path="path/to/file.jpg",
                file=file_data,
                file_options={"content-type": "image/jpeg"},
            )
            mock_bucket.get_public_url.assert_called_once_with("path/to/file.jpg")

    def test_upload_to_storage_without_content_type(self) -> None:
        """File upload should work without explicit content type."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_bucket = MagicMock()

        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.get_public_url.return_value = "https://test.url/file.jpg"

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            file_data = b"test data"
            url = upload_to_storage("route-images", "file.jpg", file_data)

            assert url == "https://test.url/file.jpg"
            mock_bucket.upload.assert_called_once_with(
                path="file.jpg",
                file=file_data,
                file_options=None,
            )

    def test_upload_to_storage_handles_errors(self) -> None:
        """Upload errors should be wrapped in SupabaseClientError."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_bucket = MagicMock()

        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.upload.side_effect = Exception("Upload failed")

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            with pytest.raises(
                SupabaseClientError,
                match="Failed to upload file to bucket 'route-images'",
            ):
                upload_to_storage("route-images", "file.jpg", b"data")


class TestDeleteFromStorage:
    """Tests for delete_from_storage function."""

    def test_delete_from_storage_success(self) -> None:
        """File deletion should succeed with valid parameters."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_bucket = MagicMock()

        mock_client.storage.from_.return_value = mock_bucket

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            delete_from_storage("route-images", "path/to/file.jpg")

            mock_client.storage.from_.assert_called_with("route-images")
            mock_bucket.remove.assert_called_once_with(["path/to/file.jpg"])

    def test_delete_from_storage_handles_errors(self) -> None:
        """Deletion errors should be wrapped in SupabaseClientError."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_bucket = MagicMock()

        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.remove.side_effect = Exception("Delete failed")

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            with pytest.raises(
                SupabaseClientError,
                match="Failed to delete file from bucket 'route-images'",
            ):
                delete_from_storage("route-images", "file.jpg")


class TestGetStorageUrl:
    """Tests for get_storage_url function."""

    def test_get_storage_url_returns_public_url(self) -> None:
        """Function should return public URL for file."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_bucket = MagicMock()

        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.get_public_url.return_value = "https://test.url/file.jpg"

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            url = get_storage_url("route-images", "path/to/file.jpg")

            assert url == "https://test.url/file.jpg"
            mock_client.storage.from_.assert_called_with("route-images")
            mock_bucket.get_public_url.assert_called_once_with("path/to/file.jpg")

    def test_get_storage_url_handles_errors(self) -> None:
        """URL retrieval errors should be wrapped in SupabaseClientError."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_bucket = MagicMock()

        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.get_public_url.side_effect = Exception("URL retrieval failed")

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            with pytest.raises(
                SupabaseClientError,
                match="Failed to get URL for file in bucket 'route-images'",
            ):
                get_storage_url("route-images", "file.jpg")


class TestListStorageFiles:
    """Tests for list_storage_files function."""

    def test_list_storage_files_success(self) -> None:
        """Function should return list of files with default pagination."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_bucket = MagicMock()

        mock_files: list[dict[str, Any]] = [
            {"name": "file1.jpg", "id": "1"},
            {"name": "file2.jpg", "id": "2"},
        ]
        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.list.return_value = mock_files

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            files = list_storage_files("route-images", "2024/")

            assert files == mock_files
            mock_client.storage.from_.assert_called_with("route-images")
            mock_bucket.list.assert_called_once_with(
                "2024/", {"limit": 100, "offset": 0}
            )

    def test_list_storage_files_with_empty_path(self) -> None:
        """Function should work with empty path."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_bucket = MagicMock()

        mock_files: list[dict[str, Any]] = []
        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.list.return_value = mock_files

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            files = list_storage_files("route-images")

            assert not files
            mock_bucket.list.assert_called_once_with("", {"limit": 100, "offset": 0})

    def test_list_storage_files_with_pagination(self) -> None:
        """Function should pass custom limit and offset to Supabase."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_bucket = MagicMock()

        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.list.return_value = []

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            list_storage_files("route-images", "2024/", limit=10, offset=20)

            mock_bucket.list.assert_called_once_with(
                "2024/", {"limit": 10, "offset": 20}
            )

    def test_list_storage_files_handles_errors(self) -> None:
        """Listing errors should be wrapped in SupabaseClientError."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_bucket = MagicMock()

        mock_client.storage.from_.return_value = mock_bucket
        mock_bucket.list.side_effect = Exception("List failed")

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            with pytest.raises(
                SupabaseClientError,
                match="Failed to list files in bucket 'route-images'",
            ):
                list_storage_files("route-images")


class TestValidateTableName:
    """Tests for _validate_table_name via public functions.

    These tests exercise _validate_table_name indirectly through insert_record
    and select_record_by_id to avoid coupling to the private function.
    """

    @pytest.mark.parametrize(
        "table_name",
        ["routes", "holds", "features", "predictions", "feedback"],
    )
    def test_known_table_names_pass_validation(self, table_name: str) -> None:
        """Known table names should pass table name validation.

        Reaching the data-validation error proves table name was accepted.
        """
        with pytest.raises(
            SupabaseClientError, match="Data dictionary cannot be empty"
        ):
            insert_record(table_name, {})

    @pytest.mark.parametrize(
        "table_name,expected_match",
        [
            # Empty / whitespace
            ("", "Table name cannot be empty"),
            (" ", "Invalid table name"),
            # Invalid format — starts with digit or contains disallowed chars
            ("123table", "Invalid table name.*must start with letter"),
            ("table-name", "Invalid table name.*alphanumeric"),
            ("table.name", "Invalid table name.*alphanumeric"),
            # SQL injection attempts (fail at regex step)
            ("routes; DROP TABLE routes", "Invalid table name.*alphanumeric"),
            ("routes--comment", "Invalid table name.*alphanumeric"),
            ("routes' OR '1'='1", "Invalid table name.*alphanumeric"),
            # Valid format but not in _KNOWN_TABLES
            ("users", "Unknown table"),
            ("admin", "Unknown table"),
            ("secrets", "Unknown table"),
            # Unicode (fail at regex step — non-ASCII chars not allowed)
            ("\u0440outes", "Invalid table name"),  # Cyrillic р as first char
            ("r\u043eut\u00e9s", "Invalid table name"),  # Cyrillic о and Latin é
        ],
    )
    def test_invalid_or_unknown_table_names_raise(
        self, table_name: str, expected_match: str
    ) -> None:
        """Invalid or unknown table names should raise SupabaseClientError."""
        with pytest.raises(SupabaseClientError, match=expected_match):
            insert_record(table_name, {})


class TestValidateBucketName:
    """Tests for _validate_bucket_name via public storage functions.

    These tests exercise _validate_bucket_name indirectly through
    upload_to_storage to avoid coupling to the private function.
    """

    @pytest.mark.parametrize("bucket_name", list(_KNOWN_BUCKETS))
    def test_known_bucket_names_pass_validation(self, bucket_name: str) -> None:
        """Known bucket names should pass validation and reach the upload step.

        A SupabaseClientError from the missing Supabase client proves the bucket
        name was accepted (validation did not raise first).
        """
        get_supabase_client.cache_clear()

        with patch("src.database.supabase_client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = get_settings_override(
                {"supabase_url": "", "supabase_key": ""}
            )
            with pytest.raises(SupabaseClientError, match="SUPABASE_URL"):
                upload_to_storage(bucket_name, "file.jpg", b"data")

    @pytest.mark.parametrize(
        "bucket_name,expected_match",
        [
            # Empty
            ("", "Bucket name cannot be empty"),
            # Invalid format — uppercase or starts with digit
            ("Route-Images", "Invalid bucket name"),
            ("1bucket", "Invalid bucket name"),
            ("bucket_name", "Invalid bucket name"),
            ("UPPERCASE", "Invalid bucket name"),
            # Valid format but not in _KNOWN_BUCKETS
            ("my-bucket", "Unknown bucket"),
            ("uploads", "Unknown bucket"),
            ("images", "Unknown bucket"),
        ],
    )
    def test_invalid_or_unknown_bucket_names_raise(
        self, bucket_name: str, expected_match: str
    ) -> None:
        """Invalid or unknown bucket names should raise SupabaseClientError."""
        with pytest.raises(SupabaseClientError, match=expected_match):
            upload_to_storage(bucket_name, "file.jpg", b"data")

    def test_bucket_validation_applies_to_delete(self) -> None:
        """_validate_bucket_name should also guard delete_from_storage."""
        with pytest.raises(SupabaseClientError, match="Unknown bucket"):
            delete_from_storage("not-a-bucket", "file.jpg")

    def test_bucket_validation_applies_to_get_url(self) -> None:
        """_validate_bucket_name should also guard get_storage_url."""
        with pytest.raises(SupabaseClientError, match="Unknown bucket"):
            get_storage_url("not-a-bucket", "file.jpg")

    def test_bucket_validation_applies_to_list(self) -> None:
        """_validate_bucket_name should also guard list_storage_files."""
        with pytest.raises(SupabaseClientError, match="Unknown bucket"):
            list_storage_files("not-a-bucket")


class TestInsertRecord:
    """Tests for insert_record function."""

    def test_insert_record_validates_empty_table_name(self) -> None:
        """Insert should reject empty table name."""
        with pytest.raises(SupabaseClientError, match="Table name cannot be empty"):
            insert_record("", {"image_url": "https://example.com/image.jpg"})

    def test_insert_record_validates_invalid_table_name(self) -> None:
        """Insert should reject invalid table names."""
        with pytest.raises(
            SupabaseClientError, match="Invalid table name.*must start with letter"
        ):
            insert_record("123routes", {"image_url": "https://example.com/image.jpg"})

        with pytest.raises(
            SupabaseClientError, match="Invalid table name.*alphanumeric"
        ):
            insert_record(
                "routes-table", {"image_url": "https://example.com/image.jpg"}
            )

    def test_insert_record_validates_empty_data(self) -> None:
        """Insert should reject empty data dictionary."""
        with pytest.raises(
            SupabaseClientError, match="Data dictionary cannot be empty"
        ):
            insert_record("routes", {})

    def test_insert_record_success(self) -> None:
        """Insert should succeed with valid inputs."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "image_url": "https://example.com/image.jpg",
                "created_at": "2026-01-27T12:00:00Z",
            }
        ]

        mock_client.table.return_value.insert.return_value.execute.return_value = (
            mock_result
        )

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            result = insert_record(
                "routes", {"image_url": "https://example.com/image.jpg"}
            )

            assert result["id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
            assert result["image_url"] == "https://example.com/image.jpg"

    def test_insert_record_raises_when_no_data_returned(self) -> None:
        """Insert should raise SupabaseClientError when result.data is empty."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = []  # empty list — Supabase returned no rows

        mock_client.table.return_value.insert.return_value.execute.return_value = (
            mock_result
        )

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            with pytest.raises(SupabaseClientError, match="returned no data"):
                insert_record("routes", {"image_url": "https://example.com/image.jpg"})

    def test_insert_record_handles_database_error(self) -> None:
        """Insert should raise error on database failure."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_client.table.return_value.insert.return_value.execute.side_effect = (
            Exception("Database error")
        )

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            with pytest.raises(
                SupabaseClientError, match="Failed to insert record into table"
            ):
                insert_record("routes", {"image_url": "https://example.com/image.jpg"})


class TestSelectRecordById:
    """Tests for select_record_by_id function."""

    def test_select_record_validates_empty_table_name(self) -> None:
        """Select should reject empty table name."""
        with pytest.raises(SupabaseClientError, match="Table name cannot be empty"):
            select_record_by_id("", "a1b2c3d4-e5f6-7890-abcd-ef1234567890")

    def test_select_record_validates_invalid_table_name(self) -> None:
        """Select should reject invalid table names."""
        with pytest.raises(
            SupabaseClientError, match="Invalid table name.*must start with letter"
        ):
            select_record_by_id("123routes", "a1b2c3d4-e5f6-7890-abcd-ef1234567890")

    def test_select_record_validates_empty_record_id(self) -> None:
        """Select should reject empty record ID."""
        with pytest.raises(SupabaseClientError, match="Record ID cannot be empty"):
            select_record_by_id("routes", "")

    @pytest.mark.parametrize(
        "record_id",
        [
            "not-a-uuid",
            "12345678",
            "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            "00000000-0000-0000-0000-00000000000Z",
            "plainstring",
        ],
    )
    def test_select_record_validates_invalid_uuid(self, record_id: str) -> None:
        """Select should reject non-UUID record IDs."""
        with pytest.raises(SupabaseClientError, match="must be a valid UUID"):
            select_record_by_id("routes", record_id)

    def test_select_record_accepts_valid_uuid(self) -> None:
        """Valid UUID should pass validation and proceed to client lookup.

        Reaching the SUPABASE_URL error proves UUID validation did not raise.
        """
        get_supabase_client.cache_clear()

        with patch("src.database.supabase_client.get_settings") as mock_get_settings:
            mock_get_settings.return_value = get_settings_override(
                {"supabase_url": "", "supabase_key": ""}
            )
            with pytest.raises(SupabaseClientError, match="SUPABASE_URL"):
                select_record_by_id("routes", "a1b2c3d4-e5f6-7890-abcd-ef1234567890")

    def test_select_record_success(self) -> None:
        """Select should succeed with valid inputs."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "image_url": "https://example.com/image.jpg",
            "created_at": "2026-01-27T12:00:00Z",
        }

        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = mock_result

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            result = select_record_by_id(
                "routes", "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
            )

            assert result is not None
            assert result["id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
            assert result["image_url"] == "https://example.com/image.jpg"

    def test_select_record_with_custom_columns(self) -> None:
        """Select should pass custom columns to the query."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "image_url": "https://example.com/img.jpg",
        }

        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = mock_result

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            result = select_record_by_id(
                "routes", "a1b2c3d4-e5f6-7890-abcd-ef1234567890", columns="id,image_url"
            )

            assert result is not None
            mock_client.table.return_value.select.assert_called_once_with(
                "id,image_url"
            )

    def test_select_record_returns_none_when_not_found(self) -> None:
        """Select should return None when record doesn't exist."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = None

        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = mock_result

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            result = select_record_by_id(
                "routes", "00000000-0000-0000-0000-000000000000"
            )

            assert result is None

    def test_select_record_raises_on_multiple_rows(self) -> None:
        """Select should raise SupabaseClientError when multiple rows are returned.

        PostgREST raises when .maybe_single() receives more than one row.
        """
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.side_effect = Exception(
            "JSON object requested, multiple (or no) rows returned"
        )

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            with pytest.raises(
                SupabaseClientError, match="Failed to select record from table"
            ):
                select_record_by_id("routes", "a1b2c3d4-e5f6-7890-abcd-ef1234567890")

    def test_select_record_handles_database_error(self) -> None:
        """Select should raise error on database failure."""
        get_supabase_client.cache_clear()

        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.side_effect = Exception(
            "Database error"
        )

        with (
            patch("src.database.supabase_client.get_settings") as mock_get_settings,
            patch("src.database.supabase_client.create_client") as mock_create_client,
        ):
            mock_get_settings.return_value = get_settings_override(
                {
                    "supabase_url": "https://test.supabase.co",
                    "supabase_key": "test-key",
                }
            )
            mock_create_client.return_value = mock_client

            with pytest.raises(
                SupabaseClientError, match="Failed to select record from table"
            ):
                select_record_by_id("routes", "a1b2c3d4-e5f6-7890-abcd-ef1234567890")
