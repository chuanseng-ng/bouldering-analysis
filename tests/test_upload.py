"""Tests for the upload route endpoint.

This module tests image upload functionality including validation,
storage integration, and error handling.
"""

import io
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from PIL import Image

from src.app import create_app
from src.config import Settings


@pytest.fixture
def app_with_upload_settings() -> Any:
    """Create test application with upload-specific settings."""
    test_config = {
        "testing": True,
        "debug": True,
        "max_upload_size_mb": 5,
        "storage_bucket": "test-bucket",
        "allowed_image_types": ["image/jpeg", "image/png"],
    }
    return create_app(test_config)


@pytest.fixture
def client_with_upload(app_with_upload_settings: Any) -> TestClient:
    """Create test client with upload settings."""
    return TestClient(app_with_upload_settings)


@pytest.fixture
def valid_jpeg_image() -> bytes:
    """Create a valid JPEG image for testing.

    Returns:
        Bytes of a small JPEG image.
    """
    # Create a simple RGB image
    img = Image.new("RGB", (100, 100), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    return img_bytes.getvalue()


@pytest.fixture
def valid_png_image() -> bytes:
    """Create a valid PNG image for testing.

    Returns:
        Bytes of a small PNG image.
    """
    # Create a simple RGB image
    img = Image.new("RGB", (100, 100), color="blue")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    return img_bytes.getvalue()


@pytest.fixture
def large_image() -> bytes:
    """Create a large image exceeding size limit.

    Returns:
        Bytes of a large JPEG image (>5MB).
    """
    import numpy as np

    # Create a large image with random noise (won't compress well)
    # This ensures it exceeds 5MB
    random_array = np.random.randint(0, 256, (3000, 3000, 3), dtype=np.uint8)
    img = Image.fromarray(random_array, mode="RGB")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG", quality=100)

    data = img_bytes.getvalue()
    # If still not large enough, pad with extra data
    if len(data) < 5 * 1024 * 1024:
        # Create a valid JPEG header and pad the rest
        padding = b"\x00" * (6 * 1024 * 1024 - len(data))
        # Note: This makes it invalid but it's OK for size validation testing
        data = data + padding

    return data


class TestUploadEndpoint:
    """Tests for POST /api/v1/routes/upload endpoint."""

    @patch("src.routes.upload.upload_to_storage")
    def test_upload_valid_jpeg_image(
        self,
        mock_upload: MagicMock,
        client_with_upload: TestClient,
        valid_jpeg_image: bytes,
    ) -> None:
        """Valid JPEG image should upload successfully."""
        # Mock the storage upload to return a public URL
        mock_upload.return_value = "https://example.com/uploads/test.jpg"

        response = client_with_upload.post(
            "/api/v1/routes/upload",
            files={"file": ("route.jpg", valid_jpeg_image, "image/jpeg")},
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "file_id" in data
        assert data["public_url"] == "https://example.com/uploads/test.jpg"
        assert data["content_type"] == "image/jpeg"
        assert data["file_size"] == len(valid_jpeg_image)
        assert "uploaded_at" in data

        # Verify storage upload was called
        mock_upload.assert_called_once()

    @patch("src.routes.upload.upload_to_storage")
    def test_upload_valid_png_image(
        self,
        mock_upload: MagicMock,
        client_with_upload: TestClient,
        valid_png_image: bytes,
    ) -> None:
        """Valid PNG image should upload successfully."""
        mock_upload.return_value = "https://example.com/uploads/test.png"

        response = client_with_upload.post(
            "/api/v1/routes/upload",
            files={"file": ("route.png", valid_png_image, "image/png")},
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["content_type"] == "image/png"
        assert data["public_url"] == "https://example.com/uploads/test.png"

    def test_upload_without_file(self, client_with_upload: TestClient) -> None:
        """Upload without file should return 422 validation error."""
        response = client_with_upload.post("/api/v1/routes/upload")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_upload_invalid_content_type(
        self, client_with_upload: TestClient, valid_jpeg_image: bytes
    ) -> None:
        """Upload with invalid content type should be rejected."""
        response = client_with_upload.post(
            "/api/v1/routes/upload",
            files={"file": ("route.gif", valid_jpeg_image, "image/gif")},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "Invalid file type" in data["detail"]
        assert "image/gif" in data["detail"]

    def test_upload_file_too_large(
        self, client_with_upload: TestClient, large_image: bytes
    ) -> None:
        """Upload of file exceeding size limit should be rejected."""
        response = client_with_upload.post(
            "/api/v1/routes/upload",
            files={"file": ("large.jpg", large_image, "image/jpeg")},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "exceeds maximum allowed size" in data["detail"]

    @patch("src.routes.upload.upload_to_storage")
    def test_upload_storage_error(
        self,
        mock_upload: MagicMock,
        client_with_upload: TestClient,
        valid_jpeg_image: bytes,
    ) -> None:
        """Storage upload failure should return 500 error."""
        from src.database.supabase_client import SupabaseClientError

        mock_upload.side_effect = SupabaseClientError("Storage connection failed")

        response = client_with_upload.post(
            "/api/v1/routes/upload",
            files={"file": ("route.jpg", valid_jpeg_image, "image/jpeg")},
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Failed to upload image to storage" in data["detail"]

    @patch("src.routes.upload.upload_to_storage")
    def test_upload_file_path_generation(
        self,
        mock_upload: MagicMock,
        client_with_upload: TestClient,
        valid_jpeg_image: bytes,
    ) -> None:
        """File path should follow year/month/uuid.ext pattern."""
        mock_upload.return_value = "https://example.com/uploads/test.jpg"

        response = client_with_upload.post(
            "/api/v1/routes/upload",
            files={"file": ("route.jpg", valid_jpeg_image, "image/jpeg")},
        )

        assert response.status_code == status.HTTP_201_CREATED

        # Check that upload_to_storage was called with correct structure
        call_args = mock_upload.call_args
        assert call_args is not None
        file_path = call_args.kwargs["file_path"]

        # Path should be in format: YYYY/MM/uuid.ext
        parts = file_path.split("/")
        assert len(parts) == 3
        assert parts[0].isdigit()  # Year
        assert parts[1].isdigit()  # Month
        assert parts[2].endswith(".jpg")  # UUID.ext

    @patch("src.routes.upload.upload_to_storage")
    def test_upload_response_has_unique_file_id(
        self,
        mock_upload: MagicMock,
        client_with_upload: TestClient,
        valid_jpeg_image: bytes,
    ) -> None:
        """Each upload should generate a unique file ID."""
        mock_upload.return_value = "https://example.com/uploads/test.jpg"

        # Upload twice
        response1 = client_with_upload.post(
            "/api/v1/routes/upload",
            files={"file": ("route1.jpg", valid_jpeg_image, "image/jpeg")},
        )
        response2 = client_with_upload.post(
            "/api/v1/routes/upload",
            files={"file": ("route2.jpg", valid_jpeg_image, "image/jpeg")},
        )

        data1 = response1.json()
        data2 = response2.json()

        assert data1["file_id"] != data2["file_id"]


class TestUploadValidation:
    """Tests for upload validation functions."""

    def test_validate_image_file_missing_filename(
        self, client_with_upload: TestClient
    ) -> None:
        """File without filename should be rejected."""
        # Create a file upload with no filename
        response = client_with_upload.post(
            "/api/v1/routes/upload",
            files={"file": ("", b"fake data", "image/jpeg")},
        )

        # Should return 400 for validation error
        # or 422 if FastAPI rejects it first
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]


class TestFilePathGeneration:
    """Tests for file path generation logic."""

    def test_generate_file_path_creates_valid_structure(self) -> None:
        """File path should have correct structure."""
        from src.routes.upload import generate_file_path

        file_id, file_path = generate_file_path("image/jpeg")

        # Check file_id is a valid UUID
        assert len(file_id) == 36  # UUID4 format
        assert file_id.count("-") == 4

        # Check path structure: YYYY/MM/uuid.ext
        parts = file_path.split("/")
        assert len(parts) == 3
        assert parts[2].endswith(".jpg")

    def test_generate_file_path_different_content_types(self) -> None:
        """Different content types should have correct extensions."""
        from src.routes.upload import generate_file_path

        _, jpeg_path = generate_file_path("image/jpeg")
        _, png_path = generate_file_path("image/png")

        assert jpeg_path.endswith(".jpg")
        assert png_path.endswith(".png")

    def test_generate_file_path_unique_ids(self) -> None:
        """Each call should generate unique file IDs."""
        from src.routes.upload import generate_file_path

        id1, _ = generate_file_path("image/jpeg")
        id2, _ = generate_file_path("image/jpeg")

        assert id1 != id2


class TestConfigSettings:
    """Tests for upload-related configuration settings."""

    def test_default_upload_settings(self) -> None:
        """Config should have sensible default upload settings."""
        from src.config import Settings

        settings = Settings(_env_file=None)

        assert settings.max_upload_size_mb == 10
        assert settings.storage_bucket == "route-images"
        assert "image/jpeg" in settings.allowed_image_types
        assert "image/png" in settings.allowed_image_types

    def test_upload_settings_override(self, app_with_upload_settings: Any) -> None:
        """Upload settings should be configurable."""
        settings: Settings = app_with_upload_settings.state.settings

        assert settings.max_upload_size_mb == 5
        assert settings.storage_bucket == "test-bucket"
