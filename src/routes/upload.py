"""Route image upload endpoint.

This module provides endpoints for uploading bouldering route images
to Supabase Storage with validation.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from src.database.supabase_client import SupabaseClientError, upload_to_storage
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["upload"])


class UploadResponse(BaseModel):
    """Response model for successful image upload.

    Attributes:
        file_id: Unique identifier for the uploaded file.
        public_url: Public URL to access the uploaded image.
        file_size: Size of the uploaded file in bytes.
        content_type: MIME type of the uploaded file.
        uploaded_at: ISO 8601 timestamp of upload.
    """

    file_id: str
    public_url: str
    file_size: int
    content_type: str
    uploaded_at: str


class ErrorResponse(BaseModel):
    """Response model for upload errors.

    Attributes:
        detail: Human-readable error message.
        error_code: Machine-readable error code.
    """

    detail: str
    error_code: str


def format_bytes(size_bytes: int) -> str:
    """Convert bytes to human-readable format.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Human-readable size string (e.g., "15.23 MB").

    Example:
        >>> format_bytes(1024)
        '1.00 KB'
        >>> format_bytes(15728640)
        '15.00 MB'
    """
    size = float(size_bytes)
    for unit in ["bytes", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def categorize_storage_error(error: SupabaseClientError) -> str:
    """Categorize storage errors for user-friendly messaging.

    Args:
        error: The storage error to categorize.

    Returns:
        User-friendly error message.

    Example:
        >>> error = SupabaseClientError("Permission denied")
        >>> categorize_storage_error(error)
        'Storage upload failed: Insufficient permissions'
    """
    error_str = str(error).lower()

    if "permission" in error_str or "unauthorized" in error_str:
        return "Storage upload failed: Insufficient permissions"
    if "quota" in error_str or "limit" in error_str:
        return "Storage upload failed: Storage quota exceeded"
    if "network" in error_str or "timeout" in error_str or "connection" in error_str:
        return "Storage upload failed: Network connection error"

    # Generic message for unknown storage errors
    return "Storage upload failed: Unable to save image"


def validate_file_signature(file_data: bytes, content_type: str) -> None:
    """Validate file signature (magic bytes) matches declared content type.

    Args:
        file_data: Raw file bytes to validate.
        content_type: Declared MIME type.

    Raises:
        HTTPException: If file signature doesn't match content type.

    Example:
        >>> validate_file_signature(jpeg_bytes, "image/jpeg")
    """
    if len(file_data) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is too small to be a valid image",
        )

    # JPEG signature: FF D8 FF
    jpeg_signature = b"\xff\xd8\xff"
    # PNG signature: 89 50 4E 47 0D 0A 1A 0A
    png_signature = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"

    file_signature = file_data[:8]

    if content_type == "image/jpeg":
        if not file_data.startswith(jpeg_signature):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File content does not match JPEG signature",
            )
    elif content_type == "image/png":
        if not file_signature == png_signature:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File content does not match PNG signature",
            )


def validate_image_file(file: UploadFile, request: Request) -> None:
    """Validate uploaded image file metadata.

    Args:
        file: Uploaded file from FastAPI.
        request: FastAPI request object (used to access app settings).

    Raises:
        HTTPException: If validation fails (400 Bad Request).

    Example:
        >>> validate_image_file(uploaded_file, request)
    """
    settings = request.app.state.settings

    # Check if file is present
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided",
        )

    # Validate content type
    if file.content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid file type '{file.content_type}'. "
                f"Allowed types: {', '.join(settings.allowed_image_types)}"
            ),
        )

    # Validate file size (check if file has size attribute)
    if hasattr(file, "size") and file.size is not None:
        max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
        if file.size > max_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"File size ({format_bytes(file.size)}) exceeds maximum "
                    f"allowed size ({format_bytes(max_size_bytes)})"
                ),
            )


def generate_file_path(content_type: str) -> tuple[str, str]:
    """Generate unique file path for storage.

    Organizes files by year/month/unique_id.ext pattern.

    Args:
        content_type: MIME type of the file (e.g., "image/jpeg").

    Returns:
        Tuple of (file_id, file_path) where file_id is the UUID
        and file_path is the storage path.

    Example:
        >>> file_id, path = generate_file_path("image/jpeg")
        >>> # Returns: ("uuid-here", "2024/01/uuid-here.jpg")
    """
    # Generate unique ID
    file_id = str(uuid.uuid4())

    # Get file extension from content type
    extension_map = {
        "image/jpeg": "jpg",
        "image/png": "png",
    }
    extension = extension_map.get(content_type, "jpg")

    # Create path with date organization
    now = datetime.now(timezone.utc)
    file_path = f"{now.year}/{now.month:02d}/{file_id}.{extension}"

    return file_id, file_path


@router.post(
    "/routes/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid file or validation error",
        },
        500: {
            "model": ErrorResponse,
            "description": "Server error during upload",
        },
    },
)
async def upload_route_image(
    file: Annotated[UploadFile, File(description="Route image file (JPEG or PNG)")],
    request: Request,
) -> UploadResponse:
    """Upload a bouldering route image.

    This endpoint accepts JPEG or PNG images up to the configured size limit,
    validates them, and stores them in Supabase Storage.

    Args:
        file: The image file to upload.
        request: FastAPI request object (used to access app settings).

    Returns:
        UploadResponse containing the public URL and metadata.

    Raises:
        HTTPException: 400 for validation errors, 500 for upload failures.

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/routes/upload" \\
             -H "Content-Type: multipart/form-data" \\
             -F "file=@route.jpg"
        ```
    """
    settings = request.app.state.settings

    # Validate the uploaded file
    validate_image_file(file, request)

    try:
        # Read file content
        file_content = await file.read()

        # Validate file signature (magic bytes) matches content type
        validate_file_signature(file_content, file.content_type or "image/jpeg")

        # Validate actual file size after reading
        max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
        if len(file_content) > max_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"File size ({format_bytes(len(file_content))}) exceeds maximum "
                    f"allowed size ({format_bytes(max_size_bytes)})"
                ),
            )

        # Generate unique file path
        file_id, file_path = generate_file_path(file.content_type or "image/jpeg")

        # Upload to Supabase Storage (run in thread to avoid blocking event loop)
        public_url = await asyncio.to_thread(
            upload_to_storage,
            bucket=settings.storage_bucket,
            file_path=file_path,
            file_data=file_content,
            content_type=file.content_type,
        )

        # Return success response
        return UploadResponse(
            file_id=file_id,
            public_url=public_url,
            file_size=len(file_content),
            content_type=file.content_type or "image/jpeg",
            uploaded_at=datetime.now(timezone.utc).isoformat() + "Z",
        )

    except SupabaseClientError as e:
        # Log full error for debugging
        logger.error(
            "Storage upload failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "file_id": file_id if "file_id" in locals() else None,
                "storage_path": file_path if "file_path" in locals() else None,
            },
        )

        # Return categorized, user-friendly error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=categorize_storage_error(e),
        ) from e
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log full error for debugging
        logger.exception(
            "Unexpected error during upload",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )

        # Return safe generic message (hide details in production)
        # Use settings from function scope (already retrieved)
        if settings.debug or settings.testing:
            # Show details in debug/testing mode
            detail = f"Unexpected error during upload: {e!s}"
        else:
            # Generic message for production
            detail = (
                "An unexpected error occurred during upload. Please try again later."
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        ) from e
