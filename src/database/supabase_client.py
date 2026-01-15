"""Supabase client management with connection pooling.

This module provides a centralized Supabase client with caching
and storage bucket access helpers.
"""

from functools import lru_cache
from typing import Any

from supabase import Client, create_client

from src.config import get_settings


class SupabaseClientError(Exception):
    """Raised when Supabase client operations fail."""


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
        client = create_client(settings.supabase_url, settings.supabase_key)
        return client
    except Exception as e:
        raise SupabaseClientError(f"Failed to create Supabase client: {e!s}") from e


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
    client = get_supabase_client()

    try:
        # Upload file to storage
        options: dict[str, Any] = {}
        if content_type:
            options["content-type"] = content_type

        client.storage.from_(bucket).upload(
            path=file_path,
            file=file_data,
            file_options=options if options else None,  # type: ignore[arg-type]
        )

        # Get public URL
        public_url: str = str(client.storage.from_(bucket).get_public_url(file_path))
        return public_url

    except Exception as e:
        raise SupabaseClientError(
            f"Failed to upload file to bucket '{bucket}': {e!s}"
        ) from e


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
    client = get_supabase_client()

    try:
        client.storage.from_(bucket).remove([file_path])
    except Exception as e:
        raise SupabaseClientError(
            f"Failed to delete file from bucket '{bucket}': {e!s}"
        ) from e


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
    client = get_supabase_client()

    try:
        url: str = str(client.storage.from_(bucket).get_public_url(file_path))
        return url
    except Exception as e:
        raise SupabaseClientError(
            f"Failed to get URL for file in bucket '{bucket}': {e!s}"
        ) from e


def list_storage_files(bucket: str, path: str = "") -> list[dict[str, Any]]:
    """List files in a Supabase Storage bucket.

    Args:
        bucket: Name of the storage bucket.
        path: Optional path prefix to filter files (e.g., "2024/01/").

    Returns:
        List of file metadata dictionaries with keys like 'name', 'id', etc.

    Raises:
        SupabaseClientError: If listing fails.

    Example:
        >>> files = list_storage_files("route-images", "2024/01/")
        >>> for file in files:
        ...     print(file["name"])
    """
    client = get_supabase_client()

    try:
        result: list[dict[str, Any]] = list(client.storage.from_(bucket).list(path))
        return result
    except Exception as e:
        raise SupabaseClientError(
            f"Failed to list files in bucket '{bucket}': {e!s}"
        ) from e
