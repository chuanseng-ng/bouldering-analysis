"""Supabase client management with connection pooling.

This module provides a centralized Supabase client with caching
and storage bucket access helpers.
"""

import re
import uuid as _uuid_module
from collections.abc import Generator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from supabase import Client, create_client
from supabase.lib.client_options import SyncClientOptions

from src.config import get_settings


class SupabaseClientError(Exception):
    """Raised when Supabase client operations fail."""


_KNOWN_TABLES: tuple[str, ...] = (
    "routes",
    "holds",
    "features",
    "predictions",
    "feedback",
)

_KNOWN_BUCKETS: tuple[str, ...] = (
    "route-images",
    "model-outputs",
)


@contextmanager
def _supabase_op(context: str) -> Generator[None, None, None]:
    """Wrap Supabase operations with consistent error handling.

    Args:
        context: Description of the operation for error messages.

    Yields:
        None

    Raises:
        SupabaseClientError: If any non-SupabaseClientError exception is raised.
    """
    try:
        yield
    except SupabaseClientError:
        raise
    except Exception as e:
        raise SupabaseClientError(f"{context}: {e!s}") from e


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Get cached Supabase client instance.

    The client is created once and cached for the lifetime of the application.
    Connection pooling is handled automatically by the underlying library.

    Returns:
        Configured Supabase client ready for database and storage operations.

    Raises:
        SupabaseClientError: If BA_SUPABASE_URL or BA_SUPABASE_KEY are not configured.

    Example:
        >>> client = get_supabase_client()
        >>> result = client.table("routes").select("*").execute()
    """
    settings = get_settings()

    # Validate required configuration
    if not settings.supabase_url:
        raise SupabaseClientError(
            "SUPABASE_URL environment variable is required but not set. "
            "Set BA_SUPABASE_URL in your environment or .env file."
        )

    if not settings.supabase_key:
        raise SupabaseClientError(
            "SUPABASE_KEY environment variable is required but not set. "
            "Set BA_SUPABASE_KEY in your environment or .env file."
        )

    try:
        options = SyncClientOptions(
            postgrest_client_timeout=settings.supabase_timeout_seconds
        )
        client = create_client(
            settings.supabase_url, settings.supabase_key, options=options
        )
        return client
    except Exception as e:
        raise SupabaseClientError(f"Failed to create Supabase client: {e!s}") from e


def reset_supabase_client_cache() -> None:
    """Clear cached Supabase client (for testing).

    Example:
        >>> reset_supabase_client_cache()
    """
    get_supabase_client.cache_clear()


def upload_to_storage(
    bucket: str,
    file_path: str,
    file_data: bytes,
    content_type: str | None = None,
) -> str:
    """Upload file to Supabase Storage bucket.

    Args:
        bucket: Name of the storage bucket (e.g., "route-images").
        file_path: Destination path within the bucket (e.g., "2024/01/route.jpg").
        file_data: Binary file content to upload.
        content_type: MIME type of the file (e.g., "image/jpeg").
            If None, Supabase will attempt to infer it.

    Returns:
        Public URL of the uploaded file.

    Raises:
        SupabaseClientError: If upload fails or bucket doesn't exist.

    Example:
        >>> with open("route.jpg", "rb") as f:
        ...     url = upload_to_storage(
        ...         "route-images",
        ...         "2024/01/route.jpg",
        ...         f.read(),
        ...         "image/jpeg"
        ...     )
    """
    _validate_bucket_name(bucket)
    client = get_supabase_client()

    with _supabase_op(f"Failed to upload file to bucket '{bucket}'"):
        options: dict[str, Any] = {}
        if content_type:
            options["content-type"] = content_type

        client.storage.from_(bucket).upload(
            path=file_path,
            file=file_data,
            file_options=options if options else None,  # type: ignore[arg-type]
        )

        public_url: str = str(client.storage.from_(bucket).get_public_url(file_path))
        return public_url


def delete_from_storage(bucket: str, file_path: str) -> None:
    """Delete file from Supabase Storage bucket.

    Args:
        bucket: Name of the storage bucket.
        file_path: Path to file within the bucket.

    Raises:
        SupabaseClientError: If deletion fails.

    Example:
        >>> delete_from_storage("route-images", "2024/01/route.jpg")
    """
    _validate_bucket_name(bucket)
    client = get_supabase_client()

    with _supabase_op(f"Failed to delete file from bucket '{bucket}'"):
        client.storage.from_(bucket).remove([file_path])


def get_storage_url(bucket: str, file_path: str) -> str:
    """Get public URL for file in Supabase Storage.

    Args:
        bucket: Name of the storage bucket.
        file_path: Path to file within the bucket.

    Returns:
        Public URL of the file.

    Raises:
        SupabaseClientError: If URL retrieval fails.

    Example:
        >>> url = get_storage_url("route-images", "2024/01/route.jpg")
    """
    _validate_bucket_name(bucket)
    client = get_supabase_client()

    with _supabase_op(f"Failed to get URL for file in bucket '{bucket}'"):
        url: str = str(client.storage.from_(bucket).get_public_url(file_path))
        return url


def list_storage_files(
    bucket: str,
    path: str = "",
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List files in a Supabase Storage bucket.

    Supabase returns at most 100 files by default. Use ``limit`` and ``offset``
    to paginate through larger directories.

    Args:
        bucket: Name of the storage bucket.
        path: Optional path prefix to filter files (e.g., "2024/01/").
        limit: Maximum number of files to return (default 100, matching the
            Supabase default). Reduce to page through large buckets.
        offset: Number of files to skip before returning results (default 0).

    Returns:
        List of file metadata dictionaries with keys like 'name', 'id', etc.

    Raises:
        SupabaseClientError: If listing fails.

    Example:
        >>> files = list_storage_files("route-images", "2024/01/")
        >>> for file in files:
        ...     print(file["name"])
    """
    _validate_bucket_name(bucket)
    client = get_supabase_client()

    with _supabase_op(f"Failed to list files in bucket '{bucket}'"):
        result: list[dict[str, Any]] = client.storage.from_(bucket).list(
            path, {"limit": limit, "offset": offset}
        )
        return result


# =============================================================================
# Database Table Operations
# =============================================================================


def _validate_table_name(table: str) -> None:
    """Validate table name for SQL safety and known-table allowlist.

    Args:
        table: Table name to validate.

    Raises:
        SupabaseClientError: If table name is invalid or not in the known tables
            allowlist.
    """
    if not table:
        raise SupabaseClientError("Table name cannot be empty")

    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table):
        raise SupabaseClientError(
            f"Invalid table name '{table}': must start with letter/underscore "
            "and contain only alphanumeric characters and underscores"
        )

    if table not in _KNOWN_TABLES:
        raise SupabaseClientError(
            f"Unknown table '{table}': must be one of {_KNOWN_TABLES}"
        )


def _validate_bucket_name(bucket: str) -> None:
    """Validate bucket name for storage operations.

    Args:
        bucket: Bucket name to validate.

    Raises:
        SupabaseClientError: If bucket name is invalid or not in the known buckets
            allowlist.
    """
    if not bucket:
        raise SupabaseClientError("Bucket name cannot be empty")

    if not re.match(r"^[a-z][a-z0-9-]*$", bucket):
        raise SupabaseClientError(
            f"Invalid bucket name '{bucket}': must start with a lowercase letter "
            "and contain only lowercase letters, digits, and hyphens"
        )

    if bucket not in _KNOWN_BUCKETS:
        raise SupabaseClientError(
            f"Unknown bucket '{bucket}': must be one of {_KNOWN_BUCKETS}"
        )


def insert_record(table: str, data: dict[str, Any]) -> dict[str, Any]:
    """Insert a record into a Supabase table.

    Args:
        table: Name of the table (e.g., "routes").
        data: Dictionary of column names to values.

    Returns:
        The inserted record with server-generated fields (id, created_at, etc.).

    Raises:
        SupabaseClientError: If insert fails, table doesn't exist, or input is invalid.

    Example:
        >>> record = insert_record("routes", {"image_url": "https://..."})
        >>> print(record["id"])
    """
    # Validate inputs
    _validate_table_name(table)

    if not data:
        raise SupabaseClientError("Data dictionary cannot be empty")

    client = get_supabase_client()

    with _supabase_op(f"Failed to insert record into table '{table}'"):
        result = client.table(table).insert(data).execute()

        if not result.data:
            raise SupabaseClientError(f"Insert to table '{table}' returned no data")

        # result.data[0] is already a dict-like object from Supabase
        record: dict[str, Any] = result.data[0]  # type: ignore[assignment]
        return record


def select_record_by_id(
    table: str,
    record_id: str,
    columns: str = "*",
) -> dict[str, Any] | None:
    """Select a single record by ID.

    Args:
        table: Name of the table.
        record_id: UUID of the record (must be a valid UUID string).
        columns: Comma-separated column names to select (default ``"*"`` for all).
            For tables with large columns (e.g. ``feature_vector JSONB``), prefer
            specifying only the columns you need to avoid over-fetching.

    Returns:
        The record as a dictionary, or None if not found.

    Raises:
        SupabaseClientError: If query fails, multiple rows are returned, or
            input is invalid.

    Example:
        >>> route = select_record_by_id("routes", "uuid-here")
        >>> if route:
        ...     print(route["image_url"])
    """
    # Validate inputs
    _validate_table_name(table)

    if not record_id:
        raise SupabaseClientError("Record ID cannot be empty")

    try:
        _uuid_module.UUID(record_id)
    except ValueError as e:
        raise SupabaseClientError(
            f"Invalid record ID '{record_id}': must be a valid UUID"
        ) from e

    client = get_supabase_client()

    with _supabase_op(f"Failed to select record from table '{table}'"):
        result = (
            client.table(table)
            .select(columns)
            .eq("id", record_id)
            .maybe_single()
            .execute()
        )
        # result.data is None if not found, dict if found;
        # PostgREST raises if >1 row is returned
        return result.data  # type: ignore[no-any-return]
