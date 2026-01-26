# Future Enhancements & Technical Debt

**Version**: 1.0
**Created**: 2026-01-26
**Last Updated**: 2026-01-26

This document tracks future enhancements, technical debt, and features deferred from PRs for future implementation.

---

## Table of Contents

1. [Security Enhancements](#security-enhancements)
2. [Upload Improvements](#upload-improvements)
3. [Observability & Monitoring](#observability--monitoring)
4. [Performance Optimizations](#performance-optimizations)
5. [Developer Experience](#developer-experience)

---

## Security Enhancements

### File Upload Security (Phase: Post-MVP)

**Priority**: High
**Related PR**: PR-2.1
**Effort**: Medium

**Enhancements**:

1. **File Content Validation**
   - Validate actual file signature/magic bytes, not just MIME type
   - Prevent malicious files disguised with valid MIME types
   - Libraries: `python-magic` or `filetype`

   ```python
   import magic

   def validate_file_signature(file_content: bytes, expected_type: str) -> bool:
       """Validate file matches its declared type."""
       mime = magic.from_buffer(file_content, mime=True)
       return mime in ALLOWED_SIGNATURES[expected_type]
   ```

2. **Virus/Malware Scanning**
   - Integrate with ClamAV or cloud antivirus service
   - Scan uploads before storing
   - Quarantine suspicious files

   ```python
   async def scan_file(file_data: bytes) -> ScanResult:
       """Scan file for malware using ClamAV."""
       # Integration with clamd
       pass
   ```

3. **Rate Limiting**
   - Implement per-IP rate limits on upload endpoint
   - Prevent DoS attacks
   - Use FastAPI middleware or Redis-based limiter

   ```python
   from slowapi import Limiter

   @limiter.limit("10/minute")
   async def upload_route_image(...):
       pass
   ```

4. **Authentication & Authorization**
   - Add user authentication (JWT/OAuth)
   - Implement upload quotas per user
   - Track upload ownership

   **Dependencies**: User management system (future milestone)

---

## Upload Improvements

### Enhanced Error Reporting (Phase: M2)

**Priority**: Medium
**Related PR**: PR-2.1
**Effort**: Small

**Current State**: Error messages are clear but could be more developer-friendly.

**Enhancements**:

1. **Machine-Readable Error Codes**

   Update `ErrorResponse` model:
   ```python
   class ErrorResponse(BaseModel):
       detail: str
       error_code: str  # NEW: e.g., "FILE_TOO_LARGE", "INVALID_TYPE"
       request_id: str | None = None  # NEW: For tracing
       metadata: dict[str, Any] | None = None  # NEW: Additional context
   ```

   Error code mapping:
   ```python
   class UploadErrorCode(str, Enum):
       NO_FILE = "NO_FILE_PROVIDED"
       INVALID_TYPE = "INVALID_FILE_TYPE"
       FILE_TOO_LARGE = "FILE_SIZE_EXCEEDED"
       STORAGE_ERROR = "STORAGE_UPLOAD_FAILED"
       UNKNOWN_ERROR = "UNEXPECTED_ERROR"
   ```

2. **Human-Readable File Sizes**

   ```python
   def format_bytes(size_bytes: int) -> str:
       """Convert bytes to human-readable format."""
       for unit in ['B', 'KB', 'MB', 'GB']:
           if size_bytes < 1024.0:
               return f"{size_bytes:.2f} {unit}"
           size_bytes /= 1024.0
       return f"{size_bytes:.2f} TB"

   # Error message becomes:
   # "File size (15.3 MB) exceeds maximum allowed size (10.0 MB)"
   ```

3. **Include Request ID in Errors**

   ```python
   @router.post("/routes/upload")
   async def upload_route_image(
       file: UploadFile,
       request: Request,  # NEW: Access request context
   ):
       request_id = request.state.request_id

       raise HTTPException(
           status_code=400,
           detail=ErrorResponse(
               detail="File too large",
               error_code="FILE_TOO_LARGE",
               request_id=request_id,
           ).model_dump()
       )
   ```

4. **Sanitize Filenames in Error Messages**

   ```python
   def sanitize_filename(filename: str, max_length: int = 50) -> str:
       """Sanitize filename for safe display in errors."""
       # Remove path traversal attempts
       safe_name = os.path.basename(filename)
       # Truncate long names
       if len(safe_name) > max_length:
           safe_name = safe_name[:max_length] + "..."
       return safe_name

   # Usage:
   detail = f"Invalid file '{sanitize_filename(file.filename)}'"
   ```

5. **Categorized Storage Errors**

   ```python
   def categorize_storage_error(error: SupabaseClientError) -> tuple[str, str]:
       """Categorize storage errors for better debugging."""
       error_str = str(error).lower()

       if "permission" in error_str or "unauthorized" in error_str:
           return "STORAGE_PERMISSION_DENIED", "Insufficient permissions"
       elif "quota" in error_str or "limit" in error_str:
           return "STORAGE_QUOTA_EXCEEDED", "Storage quota exceeded"
       elif "network" in error_str or "timeout" in error_str:
           return "STORAGE_NETWORK_ERROR", "Network connection failed"
       else:
           return "STORAGE_UNKNOWN_ERROR", f"Storage error: {error}"
   ```

**Testing Requirements**:
- Update existing error tests to check error_code field
- Add tests for each error category
- Test request ID propagation

---

### Image Processing (Phase: M3)

**Priority**: Medium
**Effort**: Medium

**Enhancements**:

1. **Image Optimization**
   - Compress images before storage (reduce costs)
   - Generate multiple sizes (thumbnail, medium, full)
   - Strip EXIF data for privacy

   ```python
   from PIL import Image, ImageOps

   async def optimize_image(
       image_data: bytes,
       max_size: tuple[int, int] = (2048, 2048),
       quality: int = 85,
   ) -> bytes:
       """Optimize image for storage."""
       img = Image.open(io.BytesIO(image_data))

       # Strip EXIF data
       img = ImageOps.exif_transpose(img)

       # Resize if too large
       img.thumbnail(max_size, Image.Lanczos)

       # Compress
       output = io.BytesIO()
       img.save(output, format='JPEG', quality=quality, optimize=True)
       return output.getvalue()
   ```

2. **Thumbnail Generation**
   - Create thumbnails for faster loading in UI
   - Store alongside original

   ```python
   async def generate_thumbnail(
       image_data: bytes,
       size: tuple[int, int] = (200, 200),
   ) -> bytes:
       """Generate thumbnail from image."""
       img = Image.open(io.BytesIO(image_data))
       img.thumbnail(size, Image.Lanczos)
       output = io.BytesIO()
       img.save(output, format='JPEG', quality=80)
       return output.getvalue()
   ```

3. **Image Metadata Extraction**
   - Extract dimensions, DPI, color space
   - Store in database for filtering/search

   ```python
   def extract_image_metadata(image_data: bytes) -> ImageMetadata:
       """Extract metadata from image."""
       img = Image.open(io.BytesIO(image_data))
       return ImageMetadata(
           width=img.width,
           height=img.height,
           format=img.format,
           mode=img.mode,
           has_transparency=img.mode in ('RGBA', 'LA', 'P'),
       )
   ```

---

## Observability & Monitoring

### Upload Metrics (Phase: M9)

**Priority**: Medium
**Effort**: Small

**Enhancements**:

1. **Metrics Collection**
   - Track upload success/failure rates
   - Monitor upload latency
   - Track file size distribution
   - Alert on storage quota

   ```python
   from prometheus_client import Counter, Histogram

   upload_requests = Counter(
       'upload_requests_total',
       'Total upload requests',
       ['status', 'file_type']
   )

   upload_duration = Histogram(
       'upload_duration_seconds',
       'Upload duration',
       buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
   )

   upload_file_size = Histogram(
       'upload_file_size_bytes',
       'Upload file size',
       buckets=[100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
   )
   ```

2. **Structured Logging**
   - Log upload events with context
   - Include file_id, size, type in logs
   - Enable log aggregation and search

   ```python
   logger.info(
       "Image uploaded successfully",
       extra={
           "file_id": file_id,
           "file_size": len(file_content),
           "content_type": file.content_type,
           "upload_duration_ms": duration_ms,
           "storage_path": file_path,
       }
   )
   ```

3. **Distributed Tracing**
   - Add OpenTelemetry spans
   - Trace upload flow: validation → storage → response
   - Correlate with other service calls

   ```python
   from opentelemetry import trace

   tracer = trace.get_tracer(__name__)

   @router.post("/routes/upload")
   async def upload_route_image(...):
       with tracer.start_as_current_span("upload_route_image") as span:
           span.set_attribute("file.type", file.content_type)
           span.set_attribute("file.size", len(file_content))
           # ... upload logic
   ```

---

## Performance Optimizations

### Upload Performance (Phase: M10)

**Priority**: Low
**Effort**: Medium

**Enhancements**:

1. **Async Image Processing**
   - Offload image optimization to background task
   - Return response immediately, process async
   - Use Celery or FastAPI BackgroundTasks

   ```python
   from fastapi import BackgroundTasks

   @router.post("/routes/upload")
   async def upload_route_image(
       file: UploadFile,
       background_tasks: BackgroundTasks,
   ):
       # Quick upload
       public_url = await quick_upload(file)

       # Process in background
       background_tasks.add_task(
           optimize_and_generate_variants,
           file_id,
           file_content,
       )

       return UploadResponse(...)
   ```

2. **Chunked Upload Support**
   - Support large file uploads (>100MB)
   - Implement resumable uploads
   - Use multipart upload to Supabase

   ```python
   @router.post("/routes/upload/chunk")
   async def upload_chunk(
       chunk_index: int,
       total_chunks: int,
       upload_id: str,
       chunk: UploadFile,
   ):
       """Upload file in chunks for large files."""
       pass
   ```

3. **CDN Integration**
   - Configure Supabase with CDN
   - Return CDN URLs instead of origin URLs
   - Cache frequently accessed images

   ```python
   def get_cdn_url(storage_url: str) -> str:
       """Convert storage URL to CDN URL."""
       return storage_url.replace(
           "supabase.co/storage",
           "cdn.yourdomain.com"
       )
   ```

---

## Developer Experience

### Testing Improvements (Phase: M2)

**Priority**: Low
**Effort**: Small

**Enhancements**:

1. **Remove NumPy Dependency from Tests**
   - Replace NumPy in `large_image` fixture
   - Use simpler approach for generating large files

   ```python
   @pytest.fixture
   def large_image() -> bytes:
       """Create large image without NumPy."""
       # Create base image
       img = Image.new('RGB', (100, 100), color='red')
       img_bytes = io.BytesIO()
       img.save(img_bytes, format='JPEG')

       # Pad to exceed 5MB (simple but effective)
       base_data = img_bytes.getvalue()
       padding_size = (6 * 1024 * 1024) - len(base_data)
       return base_data + (b'\x00' * padding_size)
   ```

2. **Mock Supabase Storage for Faster Tests**
   - Create in-memory storage mock
   - Speed up test execution

   ```python
   class MockSupabaseStorage:
       """In-memory storage for testing."""
       def __init__(self):
           self.files = {}

       def upload(self, bucket, path, data):
           self.files[f"{bucket}/{path}"] = data
           return f"https://mock.storage/{bucket}/{path}"
   ```

---

## Implementation Priority

| Enhancement | Phase | Priority | Effort | Dependencies |
|-------------|-------|----------|--------|--------------|
| Machine-readable error codes | M2 | Medium | Small | None |
| File content validation | Post-MVP | High | Medium | None |
| Rate limiting | Post-MVP | High | Small | None |
| Image optimization | M3 | Medium | Medium | PIL/Pillow |
| Upload metrics | M9 | Medium | Small | Prometheus |
| Virus scanning | Post-MVP | High | Medium | ClamAV/Cloud service |
| Authentication | Post-MVP | High | Large | User system |
| Async processing | M10 | Low | Medium | Celery/Background tasks |
| CDN integration | M10 | Low | Small | CDN service |
| Chunked uploads | M10 | Low | Medium | None |

---

## Notes

- Items marked "Post-MVP" should be implemented before production launch
- Items in Milestones (M2, M3, etc.) can be added to respective PRs
- Security enhancements should be prioritized based on threat modeling
- Performance optimizations should be data-driven (measure before optimizing)

---

## Changelog

- **2026-01-26**: Initial document created from PR-2.1 review feedback
