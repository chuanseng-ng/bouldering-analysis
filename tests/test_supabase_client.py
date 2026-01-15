"""Tests for Supabase client management."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.config import get_settings_override
from src.database.supabase_client import (
    SupabaseClientError,
    delete_from_storage,
    get_storage_url,
    get_supabase_client,
    list_storage_files,
    upload_to_storage,
)


class TestGetSupabaseClient:
    """Tests for get_supabase_client function."""

    def test_get_supabase_client_requires_url(self) -> None:
        """Client creation should fail without SUPABASE_URL."""
        # Clear cache to force new client creation
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
        """Client should be created with valid credentials."""
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
                "https://test.supabase.co", "test-key-12345"
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
                "test-bucket", "path/to/file.jpg", file_data, "image/jpeg"
            )

            assert (
                url
                == "https://test.supabase.co/storage/v1/object/public/bucket/file.jpg"
            )
            mock_client.storage.from_.assert_called_with("test-bucket")
            mock_bucket.upload.assert_called_once()
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
            url = upload_to_storage("test-bucket", "file.jpg", file_data)

            assert url == "https://test.url/file.jpg"
            mock_bucket.upload.assert_called_once()

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
                match="Failed to upload file to bucket 'test-bucket'",
            ):
                upload_to_storage("test-bucket", "file.jpg", b"data")


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

            delete_from_storage("test-bucket", "path/to/file.jpg")

            mock_client.storage.from_.assert_called_with("test-bucket")
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
                match="Failed to delete file from bucket 'test-bucket'",
            ):
                delete_from_storage("test-bucket", "file.jpg")


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

            url = get_storage_url("test-bucket", "path/to/file.jpg")

            assert url == "https://test.url/file.jpg"
            mock_client.storage.from_.assert_called_with("test-bucket")
            mock_bucket.get_public_url.assert_called_once_with("path/to/file.jpg")


class TestListStorageFiles:
    """Tests for list_storage_files function."""

    def test_list_storage_files_success(self) -> None:
        """Function should return list of files."""
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

            files = list_storage_files("test-bucket", "2024/")

            assert files == mock_files
            mock_client.storage.from_.assert_called_with("test-bucket")
            mock_bucket.list.assert_called_once_with("2024/")

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

            files = list_storage_files("test-bucket")

            assert files == []
            mock_bucket.list.assert_called_once_with("")

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
                match="Failed to list files in bucket 'test-bucket'",
            ):
                list_storage_files("test-bucket")
