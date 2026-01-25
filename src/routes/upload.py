"""Route image upload endpoint.

This module provides endpoints for uploading bouldering route images
to Supabase Storage with validation.
"""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from src.config import get_settings
from src.database.supabase_client import SupabaseClientError, upload_to_storage

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


def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file.

    Args:
        file: Uploaded file from FastAPI.

    Raises:
        HTTPException: If validation fails (400 Bad Request).

    Example:
        >>> validate_image_file(uploaded_file)
    """
    settings = get_settings()

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
                    f"File size ({file.size} bytes) exceeds maximum "
                    f"allowed size ({settings.max_upload_size_mb} MB)"
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
    now = datetime.utcnow()
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
) -> UploadResponse:
    """Upload a bouldering route image.

    This endpoint accepts JPEG or PNG images up to the configured size limit,
    validates them, and stores them in Supabase Storage.

    Args:
        file: The image file to upload.

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
    settings = get_settings()

    # Validate the uploaded file
    validate_image_file(file)

    try:
        # Read file content
        file_content = await file.read()

        # Validate actual file size after reading
        max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
        if len(file_content) > max_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"File size ({len(file_content)} bytes) exceeds maximum "
                    f"allowed size ({settings.max_upload_size_mb} MB)"
                ),
            )

        # Generate unique file path
        file_id, file_path = generate_file_path(file.content_type or "image/jpeg")

        # Upload to Supabase Storage
        public_url = upload_to_storage(
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
            uploaded_at=datetime.utcnow().isoformat() + "Z",
        )

    except SupabaseClientError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload image to storage: {e!s}",
        ) from e
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during upload: {e!s}",
        ) from e
