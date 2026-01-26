# PR-2.1 Error Message Analysis

**PR**: PR-2.1 Upload Route Image
**Date**: 2026-01-26
**Reviewer Request**: Check if error messages can be more detailed/specific for debugging while remaining safe for production

---

## Current Error Messages Review

### ✅ What's Good

1. **Clear and Specific**
   - Messages explain what went wrong
   - Include actual vs. expected values
   - Use plain language

2. **Safe for Production**
   - No sensitive data exposed (API keys, internal paths, etc.)
   - No stack traces in user-facing errors
   - Proper status codes (400 for client errors, 500 for server errors)

3. **Debugging-Friendly**
   - Include specific values (file size, content type)
   - Show allowed values for validation errors
   - Include underlying error messages in 500s

---

## Error Messages Breakdown

### 1. Missing File (Line 68)

**Current**:
```python
detail="No file provided"
```

**Analysis**: ✅ GOOD
- Clear message
- No ambiguity
- Safe for production

**Recommendation**: No change needed, but could add error code in future:
```python
detail={
    "message": "No file provided",
    "error_code": "NO_FILE_PROVIDED"
}
```

---

### 2. Invalid File Type (Lines 75-78)

**Current**:
```python
detail=(
    f"Invalid file type '{file.content_type}'. "
    f"Allowed types: {', '.join(settings.allowed_image_types)}"
)
```

**Example Output**:
```
Invalid file type 'image/gif'. Allowed types: image/jpeg, image/png
```

**Analysis**: ✅ EXCELLENT
- Shows what was provided
- Shows what's allowed
- Helps user fix the issue

**Potential Enhancement** (safe):
```python
detail=(
    f"Invalid file type '{file.content_type}' for file '{file.filename}'. "
    f"Allowed types: {', '.join(settings.allowed_image_types)}"
)
```

**Security Note**: Including `file.filename` is generally safe since:
- It's user-provided (not system information)
- Helps identify which file in multi-upload scenarios
- Could sanitize if concerned: `os.path.basename(file.filename)[:100]`

**Recommendation**: ⚠️ OPTIONAL - Add filename, but sanitize first

---

### 3. File Too Large - Pre-read Check (Lines 87-90)

**Current**:
```python
detail=(
    f"File size ({file.size} bytes) exceeds maximum "
    f"allowed size ({settings.max_upload_size_mb} MB)"
)
```

**Example Output**:
```
File size (15728640 bytes) exceeds maximum allowed size (10 MB)
```

**Analysis**: ✅ GOOD, but could be BETTER
- Shows exact size vs limit
- Mixes units (bytes vs MB) - harder to compare

**Enhancement** (safe and more user-friendly):
```python
def format_bytes(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['bytes', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

max_size_mb = settings.max_upload_size_mb
max_size_bytes = max_size_mb * 1024 * 1024

detail=(
    f"File size ({format_bytes(file.size)}) exceeds maximum "
    f"allowed size ({format_bytes(max_size_bytes)}) "
    f"[{max_size_mb} MB limit]"
)
```

**Example Output**:
```
File size (15.00 MB) exceeds maximum allowed size (10.00 MB) [10 MB limit]
```

**Recommendation**: ✅ SAFE - Implement human-readable formatting

---

### 4. File Too Large - Post-read Check (Lines 181-183)

**Current**:
```python
detail=(
    f"File size ({len(file_content)} bytes) exceeds maximum "
    f"allowed size ({settings.max_upload_size_mb} MB)"
)
```

**Analysis**: Same as #3

**Recommendation**: Apply same formatting as #3

---

### 5. Storage Upload Failed (Line 209)

**Current**:
```python
detail=f"Failed to upload image to storage: {e!s}"
```

**Example Output**:
```
Failed to upload image to storage: Connection timeout to storage.supabase.co
```

**Analysis**: ⚠️ MIXED
- ✅ Includes underlying error for debugging
- ⚠️ May expose internal details (hostnames, etc.)
- ⚠️ Generic message doesn't help user know what to do

**Security Concerns**:
- Internal hostnames could be exposed
- Error messages from Supabase might contain sensitive info
- Stack traces could leak in some error types

**Enhanced Version** (safer for production):
```python
except SupabaseClientError as e:
    error_msg = str(e).lower()

    # Categorize error for better user messaging
    if "permission" in error_msg or "unauthorized" in error_msg:
        detail = "Storage upload failed: Insufficient permissions"
        error_code = "STORAGE_PERMISSION_ERROR"
    elif "quota" in error_msg or "limit" in error_msg:
        detail = "Storage upload failed: Storage quota exceeded"
        error_code = "STORAGE_QUOTA_ERROR"
    elif "network" in error_msg or "timeout" in error_msg:
        detail = "Storage upload failed: Network connection error"
        error_code = "STORAGE_NETWORK_ERROR"
    else:
        detail = "Storage upload failed: Please try again later"
        error_code = "STORAGE_ERROR"

    # Log full error for debugging (not user-facing)
    logger.error(
        "Storage upload failed",
        extra={
            "error": str(e),
            "error_type": type(e).__name__,
            "file_id": file_id if 'file_id' in locals() else None,
        }
    )

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=detail,
    ) from e
```

**Recommendation**: ✅ IMPROVE - Categorize errors, log details server-side

---

### 6. Unexpected Error (Line 217)

**Current**:
```python
detail=f"Unexpected error during upload: {e!s}"
```

**Analysis**: ⚠️ RISKY for production
- Could expose Python internals
- Could show file paths, variable names
- Not helpful for user

**Enhanced Version** (production-safe):
```python
except Exception as e:
    # Log full error for debugging
    logger.exception(
        "Unexpected error during upload",
        extra={
            "error": str(e),
            "error_type": type(e).__name__,
            "request_id": request.state.request_id if hasattr(request.state, 'request_id') else None,
        }
    )

    # Return safe message to user
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An unexpected error occurred. Please try again later.",
    ) from e
```

**In Debug Mode** (development only):
```python
# Only in debug/testing mode
if settings.debug or settings.testing:
    detail = f"Unexpected error during upload: {e!s}"
else:
    detail = "An unexpected error occurred. Please try again later."
```

**Recommendation**: ✅ CRITICAL - Hide details in production, log server-side

---

## Recommendations Summary

### Immediate Changes (Safe for Production)

1. **Add Human-Readable File Sizes** ✅ DO IT
   - Makes error messages clearer
   - Helps users understand limits
   - No security concerns

2. **Categorize Storage Errors** ✅ DO IT
   - Safer than exposing raw error messages
   - More helpful to users
   - Log full details server-side

3. **Hide Unexpected Error Details in Production** ✅ CRITICAL
   - Generic message for users
   - Full details in server logs
   - Prevent information leakage

### Future Enhancements (See FUTURE_ENHANCEMENTS.md)

4. **Add Machine-Readable Error Codes**
   - Update ErrorResponse model
   - Return structured error objects
   - Enable programmatic error handling

5. **Include Request ID in Errors**
   - Help correlate logs with user reports
   - Enable better debugging
   - Requires middleware update

6. **Add Filename to Validation Errors** (Optional)
   - Sanitize first: `os.path.basename(file.filename)[:100]`
   - Only for 400 errors (not 500s)
   - Helps in multi-file scenarios

---

## Implementation Plan

### Option A: Minimal (Safest for Now)
Just fix the critical issue (#6) to hide unexpected errors in production.

**Changes Required**: 1 function update
**Risk**: Minimal
**Benefit**: Production-safe error handling

### Option B: Enhanced (Recommended)
Implement changes #1, #2, and #3 from immediate changes.

**Changes Required**: 3 functions updated
**Risk**: Low
**Benefit**: Better UX, safer production, better debugging

### Option C: Full Enhancement
All immediate changes + error codes + request IDs.

**Changes Required**: Multiple files (upload.py, models, middleware)
**Risk**: Low-Medium (more code changes)
**Benefit**: Professional-grade error handling

---

## Code Safety Checklist

When modifying error messages, ensure:

- [ ] No API keys, tokens, or credentials in messages
- [ ] No internal file paths or system information
- [ ] No database query details or schema information
- [ ] No stack traces in user-facing responses (only in logs)
- [ ] No version numbers of internal libraries
- [ ] User-provided data is sanitized if included
- [ ] Error categorization doesn't enable enumeration attacks
- [ ] Different errors for same issue don't leak information (timing attacks)

---

## Decision

**Recommended Approach**: Option B (Enhanced)

**Rationale**:
1. Improves user experience significantly
2. Makes debugging easier without exposing internals
3. Low risk, high value
4. Production-safe by design
5. Aligns with best practices

**Next Steps**:
1. Implement changes in a new commit
2. Update tests for new error formats
3. Test with various error scenarios
4. Document error codes in API documentation

---

## Example: Before & After

### Before
```json
{
  "detail": "File size (15728640 bytes) exceeds maximum allowed size (10 MB)"
}
```

### After (Option B)
```json
{
  "detail": "File size (15.00 MB) exceeds maximum allowed size (10.00 MB) [10 MB limit]"
}
```

### After (Option C - Full Enhancement)
```json
{
  "detail": "File size (15.00 MB) exceeds maximum allowed size (10.00 MB) [10 MB limit]",
  "error_code": "FILE_SIZE_EXCEEDED",
  "request_id": "a1b2c3d4-e5f6-7890",
  "metadata": {
    "file_size_bytes": 15728640,
    "max_size_bytes": 10485760,
    "filename": "route_photo.jpg"
  }
}
```

---

## Conclusion

Current error messages are **already quite good** for a first iteration. They are:
- Clear and specific
- Include useful debugging information
- Generally safe for production

**Key improvements needed**:
1. ✅ Human-readable file sizes (UX improvement)
2. ✅ Categorized storage errors (security + UX)
3. ✅ Hide unexpected error details in production (CRITICAL security)

All other enhancements can be deferred to future phases without significant risk.
