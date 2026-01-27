# PR-2.2: Create Route Record — Detailed Specification

**Status**: DRAFT (Pending Review)
**Milestone**: 2 — Image Upload & Persistence
**Dependencies**: PR-2.1 (Upload Route Image) ✅ Completed
**Estimated Effort**: Small

---

## 1. Objective

Create the foundational route record persistence layer that:

- [x] Stores route metadata in Supabase `routes` table
- [x] Links uploaded images to route records
- [x] Provides API endpoints for creating and retrieving routes
- [x] Supports optional wall angle parameter for future grade estimation

---

## 2. Scope

### In Scope

1. **Database Table**: Create `routes` table in Supabase
2. **Database Functions**: Add table operations to `supabase_client.py`
3. **API Endpoint**: `POST /api/v1/routes` to create route records
4. **API Endpoint**: `GET /api/v1/routes/{route_id}` to retrieve route records
5. **Validation**: Input validation for image_url and wall_angle
6. **Testing**: Comprehensive test coverage (≥85%)

### Out of Scope (Future PRs)

- Listing routes (`GET /api/v1/routes`)
- Updating routes (`PUT /api/v1/routes/{id}`)
- Deleting routes (`DELETE /api/v1/routes/{id}`)
- Route analysis endpoints (PR-5.x)
- Holds, features, predictions tables (PR-9.x)

---

## 3. Database Schema

### Table: routes

```sql
CREATE TABLE routes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_url TEXT NOT NULL,
    wall_angle FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security (RLS)
ALTER TABLE routes ENABLE ROW LEVEL SECURITY;

-- Create policy for public read access (adjust based on auth requirements)
CREATE POLICY "Allow public read access" ON routes FOR SELECT USING (true);

-- Create policy for service role write access
CREATE POLICY "Allow service write access" ON routes FOR INSERT
    WITH CHECK (true);

-- Index for faster lookups by creation date
CREATE INDEX idx_routes_created_at ON routes (created_at DESC);
```

### Migration Instructions

The table should be created manually in Supabase SQL Editor, as this project does not use automated migrations. Document in `docs/SUPABASE_SETUP.md`.

---

## 4. File Structure

### New Files

```text
src/
└── routes/
    └── routes.py              # Route record endpoints (~150 lines)

tests/
└── test_routes.py             # Route record tests (~400 lines)
```

### Modified Files

```text
src/
├── database/
│   └── supabase_client.py     # Add table operations (+60 lines)
└── routes/
    └── __init__.py            # Export routes_router

docs/
└── SUPABASE_SETUP.md          # Add routes table setup instructions
```

---

## 5. API Specification

### 5.1 Create Route

**Endpoint**: `POST /api/v1/routes`

**Request Body**:

```json
{
    "image_url": "https://your-project.supabase.co/storage/v1/object/public/route-images/2026/01/uuid.jpg",
    "wall_angle": 15.0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_url` | string | Yes | Public URL of the uploaded route image |
| `wall_angle` | float | No | Wall angle in degrees (-90 to 90). Null if unknown. Negative = overhang, Positive = slab |

**Response** (201 Created):

```json
{
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "image_url": "https://your-project.supabase.co/storage/v1/object/public/route-images/2026/01/uuid.jpg",
    "wall_angle": 15.0,
    "created_at": "2026-01-27T12:00:00Z",
    "updated_at": "2026-01-27T12:00:00Z"
}
```

**Error Responses**:

| Status | Condition | Response |
|--------|-----------|----------|
| 400 | Invalid image_url format | `{"detail": "Invalid image URL format"}` |
| 400 | Invalid wall_angle range | `{"detail": "Wall angle must be between -90 and 90 degrees"}` |
| 422 | Missing required fields | FastAPI validation error |
| 500 | Database error | `{"detail": "Failed to create route record"}` |

### 5.2 Get Route

**Endpoint**: `GET /api/v1/routes/{route_id}`

**Path Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `route_id` | UUID | Route identifier |

**Response** (200 OK):

```json
{
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "image_url": "https://your-project.supabase.co/storage/v1/object/public/route-images/2026/01/uuid.jpg",
    "wall_angle": 15.0,
    "created_at": "2026-01-27T12:00:00Z",
    "updated_at": "2026-01-27T12:00:00Z"
}
```

**Error Responses**:

| Status | Condition | Response |
|--------|-----------|----------|
| 404 | Route not found | `{"detail": "Route not found"}` |
| 422 | Invalid UUID format | FastAPI validation error |
| 500 | Database error | `{"detail": "Failed to retrieve route"}` |

---

## 6. Function Contracts

### 6.1 Database Operations (supabase_client.py)

```python
def insert_record(table: str, data: dict[str, Any]) -> dict[str, Any]:
    """Insert a record into a Supabase table.

    Args:
        table: Name of the table (e.g., "routes").
        data: Dictionary of column names to values.

    Returns:
        The inserted record with server-generated fields (id, created_at, etc.).

    Raises:
        SupabaseClientError: If insert fails.

    Example:
        >>> record = insert_record("routes", {"image_url": "https://..."})
        >>> print(record["id"])
    """


def select_record_by_id(table: str, record_id: str) -> dict[str, Any] | None:
    """Select a single record by ID.

    Args:
        table: Name of the table.
        record_id: UUID of the record.

    Returns:
        The record as a dictionary, or None if not found.

    Raises:
        SupabaseClientError: If query fails.

    Example:
        >>> route = select_record_by_id("routes", "uuid-here")
        >>> if route:
        ...     print(route["image_url"])
    """
```

### 6.2 Route Endpoint Functions (routes.py)

```python
class RouteCreate(BaseModel):
    """Request model for creating a route."""
    image_url: str
    wall_angle: float | None = None


class RouteResponse(BaseModel):
    """Response model for route data."""
    id: str
    image_url: str
    wall_angle: float | None
    created_at: str
    updated_at: str


@router.post("/routes", response_model=RouteResponse, status_code=201)
async def create_route(route_data: RouteCreate, request: Request) -> RouteResponse:
    """Create a new route record.

    Args:
        route_data: Route creation data with image_url and optional wall_angle.
        request: FastAPI request object.

    Returns:
        Created route record with generated ID and timestamps.

    Raises:
        HTTPException: 400 for validation errors, 500 for database errors.
    """


@router.get("/routes/{route_id}", response_model=RouteResponse)
async def get_route(route_id: str) -> RouteResponse:
    """Retrieve a route by ID.

    Args:
        route_id: UUID of the route to retrieve.

    Returns:
        Route record with all fields.

    Raises:
        HTTPException: 404 if not found, 500 for database errors.
    """
```

---

## 7. Validation Rules

### 7.1 image_url

- Must be a valid URL (https:// scheme)
- Must not be empty
- Maximum length: 2048 characters
- Should match expected Supabase Storage URL pattern (warning, not error)

### 7.2 wall_angle

- Optional (nullable)
- If provided, must be between -90 and 90 degrees
- Negative values = overhanging wall
- Zero = vertical wall
- Positive values = slab (less than vertical)
- Precision: 1 decimal place stored

### 7.3 route_id (for GET endpoint)

- Must be a valid UUID v4 format
- Pydantic will handle basic format validation

---

## 8. Error Handling

### 8.1 Error Categories

| Category | HTTP Status | User Message Pattern |
|----------|-------------|---------------------|
| Validation | 400 | Specific validation message |
| Not Found | 404 | "Route not found" |
| Database | 500 | "Failed to [operation] route" |
| Unexpected | 500 | Debug mode: details; Prod: generic |

### 8.2 Logging

All errors should be logged with:
- Error type and message
- Route ID (if applicable)
- Request metadata (in structured JSON format)

---

## 9. Testing Plan

### 9.1 Test Classes

| Test Class | Purpose | Est. Tests |
|------------|---------|------------|
| `TestCreateRouteEndpoint` | Create route happy path & errors | 10 |
| `TestGetRouteEndpoint` | Get route happy path & errors | 6 |
| `TestRouteValidation` | Input validation edge cases | 8 |
| `TestDatabaseOperations` | Database function unit tests | 8 |
| `TestRouteModels` | Pydantic model validation | 4 |

**Total Estimated Tests**: ~36

### 9.2 Test Scenarios

**Create Route Tests**:
1. Create route with valid image_url only
2. Create route with image_url and wall_angle
3. Create route with null wall_angle
4. Create route with boundary wall_angle values (-90, 0, 90)
5. Reject invalid image_url format
6. Reject wall_angle out of range (< -90, > 90)
7. Reject empty image_url
8. Handle database insert error gracefully
9. Response contains all expected fields
10. UUID is valid format

**Get Route Tests**:
1. Get existing route returns correct data
2. Get non-existent route returns 404
3. Get with invalid UUID format returns 422
4. Handle database query error gracefully
5. Response timestamps are ISO 8601 format
6. Response matches RouteResponse schema

**Validation Tests**:
1. image_url max length validation
2. image_url scheme validation (require https)
3. wall_angle precision handling
4. wall_angle exactly at boundaries
5. wall_angle just outside boundaries
6. Empty request body
7. Extra fields ignored
8. Partial request (missing required)

**Database Operation Tests**:
1. insert_record returns complete record
2. insert_record handles duplicate key
3. insert_record handles connection error
4. select_record_by_id returns record
5. select_record_by_id returns None for missing
6. select_record_by_id handles connection error
7. select_record_by_id handles invalid table
8. Mocked Supabase client patterns

---

## 10. Implementation Checklist

### Phase 1: Database Layer

- [ ] Add `insert_record()` function to `supabase_client.py`
- [ ] Add `select_record_by_id()` function to `supabase_client.py`
- [ ] Write tests for new database functions
- [ ] Document table creation in `docs/SUPABASE_SETUP.md`

### Phase 2: API Endpoints

- [ ] Create `src/routes/routes.py` module
- [ ] Implement `RouteCreate` request model
- [ ] Implement `RouteResponse` response model
- [ ] Implement `create_route()` endpoint
- [ ] Implement `get_route()` endpoint
- [ ] Add input validation with clear error messages
- [ ] Export router in `src/routes/__init__.py`
- [ ] Register router in `src/app.py`

### Phase 3: Testing

- [ ] Create `tests/test_routes.py`
- [ ] Write all test cases from Section 9.2
- [ ] Achieve ≥85% coverage for new code
- [ ] Verify all tests pass

### Phase 4: Documentation & QA

- [ ] Update `CLAUDE.md` with new endpoint documentation
- [ ] Run full QA suite: mypy, ruff, pylint, pytest
- [ ] Verify pylint score ≥8.5/10
- [ ] Manual verification with curl commands

---

## 11. Quality Gates

### Pre-Merge Requirements

- [ ] `mypy src/ tests/` passes with no errors
- [ ] `ruff check src/ tests/ --ignore E501` passes
- [ ] `ruff format --check src/ tests/` passes
- [ ] `pytest tests/ --cov=src --cov-fail-under=85` passes
- [ ] `pylint src/ --ignore=archive` score ≥ 8.5/10
- [ ] All new functions have Google-style docstrings
- [ ] All functions have complete type annotations

### Manual Verification

```bash
# Start server
uvicorn src.app:application --reload

# Create a route
curl -X POST http://localhost:8000/api/v1/routes \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg", "wall_angle": 15.0}'

# Get a route
curl http://localhost:8000/api/v1/routes/{route_id}
```

---

## 12. Design Decisions

### 12.1 Separate Upload and Route Creation

**Decision**: Keep image upload and route creation as separate API calls.

**Rationale**:
- Flexibility: Same image can be used for multiple routes
- Simplicity: Each endpoint has a single responsibility
- Consistency: Matches the migration plan structure
- Future-proofing: Allows for image processing before route creation

**Workflow**:
1. Client uploads image → receives `public_url`
2. Client creates route with `image_url` = `public_url`

### 12.2 Wall Angle as Optional

**Decision**: Make `wall_angle` optional (nullable).

**Rationale**:
- User may not know the wall angle initially
- Can be updated later when available
- Allows route creation to proceed without blocking on wall angle

### 12.3 No Image URL Existence Check

**Decision**: Do not verify that `image_url` points to an existing image.

**Rationale**:
- Checking external URLs adds latency
- URLs may become invalid later anyway
- Client is responsible for providing valid URL from upload step
- Keeps create operation fast

### 12.4 UTC Timestamps

**Decision**: All timestamps in UTC with ISO 8601 format.

**Rationale**:
- Consistency across all responses
- Timezone-agnostic storage
- Easy client-side conversion
- Matches existing health endpoint pattern

---

## 13. Integration Points

### 13.1 Depends On

| Component | Usage |
|-----------|-------|
| PR-2.1 Upload | Provides `public_url` for route creation |
| PR-1.2 Supabase Client | Database connection and operations |
| PR-1.1 FastAPI Bootstrap | Application factory, middleware, logging |

### 13.2 Required By

| Future PR | Dependency |
|-----------|------------|
| PR-3.4 Detection Inference | Route ID to associate detected holds |
| PR-5.1 Graph Builder | Route ID to build movement graph |
| PR-9.x Database Schema | Holds, features, predictions reference routes |

---

## 14. Open Questions

1. **URL Validation Strictness**: Should we validate that `image_url` matches Supabase Storage URL pattern, or accept any HTTPS URL?
   - **Recommendation**: Accept any HTTPS URL (more flexible for testing and edge cases)

2. **Route Deletion**: If a route is deleted, should we cascade delete the image from storage?
   - **Recommendation**: Defer to future PR; routes table doesn't need ON DELETE CASCADE initially

3. **Duplicate Image URLs**: Should we allow multiple routes with the same `image_url`?
   - **Recommendation**: Yes, allow duplicates (same wall photo, different routes)

---

## 15. References

- [plans/MIGRATION_PLAN.md](../MIGRATION_PLAN.md) — Migration roadmap
- [docs/DESIGN.md](../../docs/DESIGN.md) — Domain model specification
- [plans/specs/PR-1.1-fastapi-bootstrap.md](PR-1.1-fastapi-bootstrap.md) — Bootstrap spec (pattern reference)
- [Supabase Python Client](https://supabase.com/docs/reference/python/introduction)
- [FastAPI Path Operations](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/)

---

## Changelog

- **2026-01-27**: Initial specification draft created
