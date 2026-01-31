# Production MVP Specification
**Version**: 1.0
**Created**: 2026-01-31
**Target Budget**: $100/month
**Target Users**: 100 beta testers
**Beta Duration**: 6 months
**Timeline**: 3-4 months to production-ready

---

## 1. Executive Summary

This specification defines the production-ready MVP (Minimum Viable Product) for the bouldering route analysis platform, designed to support 100 beta users within a $100/month infrastructure budget.

### 1.1 MVP Scope

**Frontend Interfaces (2):**
- Web application (React/Next.js on Vercel)
- Telegram bot (Python on Railway)

**Authentication:**
- Hybrid model: Anonymous usage + optional account creation
- Session-based tracking for anonymous users
- JWT authentication for registered users

**Core Features (4):**
1. Manual hold annotation (start/finish selection)
2. Route history/gallery
3. Feedback submission
4. Social sharing

**Infrastructure:**
- Total cost: $80/month
- Platform: Railway (backend), Vercel (web), Telegram API
- Storage: Cloudflare R2 (free tier)
- Database: PostgreSQL on Railway
- Queue: Redis + Celery for async ML processing

---

## 2. Architecture Overview

### 2.1 System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                  FRONTEND LAYER                            │
│                                                            │
│  ┌─────────────────────┐    ┌────────────────────┐       │
│  │  Web Frontend       │    │  Telegram Bot       │       │
│  │  (Next.js/React)    │    │  (Python)           │       │
│  │  Vercel (Free)      │    │  Railway ($10/mo)   │       │
│  └──────────┬──────────┘    └─────────┬──────────┘       │
│             │                          │                   │
└─────────────┼──────────────────────────┼───────────────────┘
              │                          │
              └────────────┬─────────────┘
                           │ HTTPS/REST API
                           ↓
┌───────────────────────────────────────────────────────────┐
│                  BACKEND LAYER                             │
│              Railway Platform ($80/month)                  │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   FastAPI    │  │   Celery     │  │  Telegram    │   │
│  │   Backend    │  │   Worker     │  │     Bot      │   │
│  │  $20/month   │  │  $30/month   │  │  $10/month   │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘   │
│         │                  │                              │
│  ┌──────┴──────────────────┴───────┐  ┌────────────┐    │
│  │        Redis Queue               │  │ PostgreSQL │    │
│  │        $10/month                 │  │ $10/month  │    │
│  └──────────────────────────────────┘  └────────────┘    │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ↓
┌───────────────────────────────────────────────────────────┐
│                  STORAGE LAYER                             │
│                                                            │
│  Cloudflare R2 (Image Storage)                            │
│  Free Tier: 10GB storage, unlimited bandwidth             │
└───────────────────────────────────────────────────────────┘
```

### 2.2 Request Flow

**Typical User Journey:**

```
1. User uploads image (Web or Telegram)
   ↓
2. FastAPI validates & uploads to R2 (200ms)
   ↓
3. Task queued in Celery/Redis (50ms)
   ↓
4. User receives task_id immediately
   ↓
5. Celery worker processes in background:
   - YOLOv8 detection (3-5s)
   - ResNet classification (1-2s)
   - Graph construction (0.5s)
   - Feature extraction (0.5s)
   - Grade prediction (0.5s)
   - Total: ~5-10s
   ↓
6. Result cached in Redis (7-day TTL)
   ↓
7. User polls /tasks/{id} or receives notification
   ↓
8. User views prediction, holds, explanation
```

---

## 3. Infrastructure Specification

### 3.1 Cost Breakdown

| Service | Specs | Monthly Cost |
|---------|-------|--------------|
| **Railway - FastAPI Backend** | 512MB RAM, 1 vCPU | $20 |
| **Railway - Celery Worker** | 1GB RAM, 2 vCPU | $30 |
| **Railway - Telegram Bot** | 256MB RAM, 0.5 vCPU | $10 |
| **Railway - Redis** | 256MB RAM | $10 |
| **Railway - PostgreSQL** | 1GB storage | $10 |
| **Vercel - Web Frontend** | Hobby tier | $0 (free) |
| **Cloudflare R2 - Image Storage** | 10GB storage | $0 (free) |
| **Sentry - Error Tracking** | 5K events/month | $0 (free) |
| **TOTAL** | | **$80/month** |

**Budget Headroom:** $20/month for:
- Traffic spikes
- Storage overages (if >10GB images)
- Future auth service (Supabase Auth, if needed)

### 3.2 Performance Expectations

**For 100 Beta Users:**

| Metric | Value |
|--------|-------|
| **Daily Route Uploads** | ~40 average, ~100 peak |
| **Worker Throughput** | 4-8 routes/minute |
| **Daily Capacity** | 5,000+ routes/day |
| **Headroom** | 50-100x |
| **Analysis Latency** | 5-15 seconds |
| **Upload Latency** | 200-500ms |
| **Database Size (3 months)** | ~100MB |
| **Image Storage (3 months)** | ~1.6GB (compressed) |
| **Uptime Target** | 99% during beta |
| **Error Rate Target** | <1% of requests |

### 3.3 Scaling Path (Post-Beta)

**If successful → 1000 users:**

**Option 1: Upgrade Railway** ($140/month)
- FastAPI: $20 → $30
- Celery: $30 → $50
- Add 2nd Celery worker: +$50

**Option 2: Move to AWS** ($470/month but 10x faster)
- ECS Fargate: $100/month
- GPU worker (g4dn.xlarge): $250/month
- RDS PostgreSQL: $50/month
- ElastiCache Redis: $50/month
- S3 + CloudFront: $20/month

---

## 4. Technical Specifications

### 4.1 Backend API Endpoints

**Authentication:**
```
POST /api/v1/auth/signup         - Create account
POST /api/v1/auth/login          - Login
POST /api/v1/auth/claim-routes   - Migrate anonymous routes to account
```

**Routes:**
```
POST   /api/v1/routes/upload            - Upload route image
POST   /api/v1/routes                   - Create route record
GET    /api/v1/routes                   - List user's routes (paginated)
GET    /api/v1/routes/{id}              - Get route details
DELETE /api/v1/routes/{id}              - Delete route
PUT    /api/v1/routes/{id}/constraints  - Set start/finish holds
POST   /api/v1/routes/{id}/analyze      - Trigger analysis
```

**Analysis:**
```
GET    /api/v1/routes/{id}/holds       - Get detected holds
GET    /api/v1/routes/{id}/prediction  - Get grade prediction
POST   /api/v1/routes/{id}/feedback    - Submit feedback
```

**Tasks:**
```
GET /api/v1/tasks/{task_id}   - Get task status (polling)
GET /api/v1/stream/{task_id}  - Stream task progress (SSE)
```

**Task Access Control:**
- **Authentication Required**: Both endpoints require authenticated caller (JWT token or session cookie)
- **Authorization**: Task ownership verification
  - Match `task.session_id` to caller's session ID (anonymous users)
  - OR match `task.user_id` to caller's user ID (authenticated users)
  - Admin/service roles (role='admin' or role='service') bypass ownership checks
- **Error Responses**:
  - `401 Unauthorized` - No valid JWT token or session cookie
  - `403 Forbidden` - Task exists but caller is not the owner (ownership mismatch)
  - `404 Not Found` - Task does not exist OR caller is not owner (prevents task ID enumeration)

**Implementation Notes:**
```python
@router.get("/api/v1/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    request: Request,
    user: User | None = Depends(get_current_user)  # JWT or session
):
    """Get task status with ownership verification."""
    task = await get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    # Verify ownership (unless admin/service role)
    if user and user.role in ['admin', 'service']:
        # Admins and services can view any task
        pass
    elif user and task.user_id == user.id:
        # Authenticated user owns this task
        pass
    elif not user and task.session_id == request.state.session_id:
        # Anonymous session owns this task
        pass
    else:
        # Ownership mismatch - return 404 to prevent enumeration
        raise HTTPException(404, "Task not found")

    return task.to_dict()

@router.get("/api/v1/stream/{task_id}")
async def stream_task_progress(
    task_id: str,
    request: Request,
    user: User | None = Depends(get_current_user)
):
    """Stream task progress via SSE with ownership verification."""
    # Same ownership verification as get_task_status
    task = await get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    # Verify ownership (same logic as above)
    if not await verify_task_ownership(task, user, request.state.session_id):
        raise HTTPException(404, "Task not found")

    # Start SSE stream
    async def event_generator():
        while True:
            task = await get_task(task_id)
            yield f"data: {task.to_json()}\n\n"
            if task.status in ['completed', 'failed']:
                break
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())
```

**Sharing:**
```
GET /api/v1/routes/{id}/share  - Generate shareable link
GET /r/{short_id}              - View shared route (public)
```

**Health:**
```
GET /health         - Basic health check
GET /health/ready   - Readiness check (DB, Redis, workers)
GET /metrics        - Prometheus metrics
```

### 4.2 Database Schema

**users table:**
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL
        CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    password_hash VARCHAR(255) NOT NULL
        CHECK (LENGTH(password_hash) >= 60),  -- bcrypt hashes are 60 chars
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true,
    quota_hourly INT DEFAULT 10,    -- Authenticated users: 10 uploads/hour
    quota_daily INT DEFAULT 50,     -- Authenticated users: 50 uploads/day
    quota_monthly INT DEFAULT 200   -- Authenticated users: 200 uploads/month
);

-- Column comments documenting password complexity requirements
COMMENT ON COLUMN users.email IS 'User email address. Must match basic email format regex.';
COMMENT ON COLUMN users.password_hash IS 'Bcrypt password hash. Must be at least 60 characters (bcrypt standard). Password requirements: minimum 10-12 characters (configurable, default 10), at least one uppercase, one lowercase, one number, one special character (!@#$%^&*()_+-=[]{}|;:,.<>?). See authentication spec (section 6.3) for full complexity rules.';
COMMENT ON COLUMN users.last_login IS 'Timestamp of user''s most recent login. NULL if user has never logged in.';
COMMENT ON COLUMN users.is_active IS 'Account active status. Set to false to disable account without deletion.';
```

**routes table:**
```sql
CREATE TABLE routes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_url TEXT NOT NULL
        CHECK (image_url ~* '^https?://'),  -- Must start with http:// or https://
    wall_angle FLOAT
        CHECK (wall_angle BETWEEN -15 AND 90),  -- Overhang (-15°) to slab (90°)
    owner_id VARCHAR(64) NOT NULL,  -- User UUID (36 chars) or session ID (64 chars max)
    owner_type VARCHAR(20) DEFAULT 'anonymous'
        CHECK (owner_type IN ('anonymous', 'authenticated')),
    start_hold_ids INT[],  -- Array of start hold IDs
    finish_hold_id INT,    -- Single finish hold ID
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_routes_owner ON routes(owner_id, owner_type);
CREATE INDEX idx_routes_created_at ON routes(created_at DESC);

-- Column comments
COMMENT ON COLUMN routes.image_url IS 'URL to route image in storage. Must start with http:// or https://. Additional URL validation performed at application layer.';
COMMENT ON COLUMN routes.wall_angle IS 'Wall angle in degrees. Range: -15° (overhang) to 90° (vertical slab). NULL for unknown/unspecified.';
COMMENT ON COLUMN routes.owner_id IS 'Owner identifier: UUID for authenticated users (36 chars) or session ID for anonymous users (up to 64 chars).';
```

**holds table:**
```sql
CREATE TABLE holds (
    id SERIAL PRIMARY KEY,
    route_id UUID REFERENCES routes(id) ON DELETE CASCADE,
    x FLOAT NOT NULL,  -- Normalized [0, 1]
    y FLOAT NOT NULL,
    size FLOAT,
    type VARCHAR(20),  -- 'jug', 'crimp', 'sloper', 'pinch', 'volume', 'unknown'
    confidence FLOAT
);

CREATE INDEX idx_holds_route_id ON holds(route_id);
```

**features table:**
```sql
CREATE TABLE features (
    id SERIAL PRIMARY KEY,
    route_id UUID REFERENCES routes(id) ON DELETE CASCADE UNIQUE,
    feature_vector JSONB NOT NULL,
    extracted_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_features_vector ON features USING gin(feature_vector);
```

**predictions table:**
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    route_id UUID REFERENCES routes(id) ON DELETE CASCADE,
    grade VARCHAR(10) NOT NULL,  -- 'V0' to 'V17'
    confidence FLOAT,
    uncertainty FLOAT,
    explanation TEXT,
    model_version VARCHAR(50),
    predicted_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_predictions_route_id ON predictions(route_id);
CREATE INDEX idx_predictions_grade ON predictions(grade);
```

**feedback table:**
```sql
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    route_id UUID REFERENCES routes(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    predicted_grade VARCHAR(10),
    user_grade VARCHAR(10),
    is_accurate BOOLEAN,
    comments TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_feedback_route_id ON feedback(route_id);
```

### 4.3 ML Model Specifications

**Detection Model:**
- Architecture: YOLOv8m (ONNX optimized)
- Input: 640x640 RGB image
- Output: Bounding boxes, classes (hold/volume), confidence
- Inference time: 3-5 seconds (CPU)
- Location: `models/detection/yolov8m.onnx`

**Classification Model:**
- Architecture: ResNet-18 (ONNX optimized)
- Input: 224x224 RGB crops
- Output: Hold type probabilities
- Classes: jug, crimp, sloper, pinch, volume, unknown
- Inference time: 0.3-0.5 seconds per hold set
- Location: `models/classification/resnet18.onnx`

**Grading Model:**
- Architecture: XGBoost (future) or heuristic (MVP)
- Input: Feature vector (geometry + hold composition)
- Output: Grade (V0-V17), confidence, uncertainty
- Location: `models/grading/xgboost.json` (future)

### 4.4 Caching Strategy

**Redis Cache Keys:**
```
# Prediction results (7-day TTL)
prediction:{route_id} → Prediction JSON

# Image deduplication (permanent)
image_hash:{sha256} → route_id

# Rate limiting (1-hour sliding window)
ratelimit:{ip}:{endpoint} → request count

# Task results (1-hour TTL)
celery-task-meta-{task_id} → task result

# User session (30-day TTL)
session:{session_id} → session data
```

### 4.5 Rate Limiting

**Anonymous Users (IP-based):**
```
POST /api/v1/routes/upload:
  - 10 requests/hour
  - 50 requests/day
  - 200 requests/month

POST /api/v1/routes/{id}/analyze:
  - 5 requests/hour
  - 30 requests/day
```

**Authenticated Users (user-based):**
```
POST /api/v1/routes/upload:
  - 50 requests/hour
  - 200 requests/day
  - 1000 requests/month

POST /api/v1/routes/{id}/analyze:
  - 30 requests/hour
  - 200 requests/day
```

---

## 5. Feature Specifications

### 5.1 Manual Hold Annotation

**Purpose:** Allow users to manually specify start holds and finish hold for accurate grade prediction.

**UI Flow:**
1. User uploads image → receives detected holds overlay
2. User clicks "Mark Start Holds" → enters selection mode
3. User clicks 1-3 holds to mark as start (green highlights)
4. User clicks "Mark Finish Hold" → enters selection mode
5. User clicks 1 hold to mark as finish (red highlight)
6. User clicks "Analyze Route" → triggers grade prediction
7. System re-runs analysis with constraints

**Backend:**
- Endpoint: `PUT /api/v1/routes/{id}/constraints`
- Validation: Ensure selected hold IDs exist for route
- Effect: Invalidate cached prediction, re-queue analysis

**Database:**
- Store `start_hold_ids` (array) and `finish_hold_id` in routes table

### 5.2 Route History/Gallery

**Purpose:** Allow users to view their previously analyzed routes.

**UI Flow:**
1. User navigates to "My Routes" page
2. System displays grid of route thumbnails
3. Each card shows: image, grade, confidence, date
4. User clicks card → navigates to route detail page
5. Pagination: 20 routes per page

**Backend:**
- Endpoint: `GET /api/v1/routes?page=1&limit=20`
- Sort by: created_at DESC (default)
- Filter by: owner_id + owner_type
- Join: routes + predictions for grade display

**Access Control:**
- Anonymous: View routes from session only
- Authenticated: View all owned routes
- Migration: Claim anonymous routes when creating account

### 5.3 Feedback Submission

**Purpose:** Collect user feedback to improve model accuracy.

**UI Flow:**
1. User views prediction on route detail page
2. User clicks "Submit Feedback"
3. Modal appears with form:
   - "What's the actual grade?" (dropdown V0-V17)
   - "Was our prediction accurate?" (yes/no toggle)
   - "Additional comments" (optional textarea)
4. User submits → receives confirmation
5. Optional: Award points/badges for feedback

**Backend:**
- Endpoint: `POST /api/v1/routes/{id}/feedback`
- Store: predicted_grade, user_grade, is_accurate, comments
- Effect: Queue route for model retraining (future)

**Analytics:**
- Admin dashboard showing:
  - Accuracy rate (% is_accurate = true)
  - Grade distribution (predicted vs actual)
  - Common mispredictions

### 5.4 Social Sharing

**Purpose:** Allow users to share route predictions with friends.

**UI Flow:**
1. User views route detail page
2. User clicks "Share" button
3. System generates short URL (e.g., yourdomain.com/r/abc123)
4. Options:
   - Copy link to clipboard
   - Native share (mobile)
   - Share to social media (future)
5. Recipients can view route without authentication

**Backend:**
- Endpoint: `GET /api/v1/routes/{id}/share`
- Generate: Short URL using hashids
- Public endpoint: `GET /r/{short_id}` (no auth required)
- Metadata: Open Graph tags for social previews

**Security:**
- Public routes are read-only
- No PII exposed in shared routes
- Optional: User can make route private (future)

---

## 6. Authentication Specification

### 6.1 Hybrid Authentication Model

**Anonymous Users:**
- Automatic session creation on first visit
- Session stored in HTTP-only cookie (30-day expiry)
- Session ID used as owner_id in database
- **Rate limits (anonymous):**
  - 10 uploads per hour
  - 50 uploads per day
  - 200 uploads per month

**Authenticated Users:**
- Email + password signup/login
- JWT token issued (access: 15-60 min, refresh: 30 days)
- **Token storage (production recommendation):**
  - **PREFERRED**: HTTP-only cookie with SameSite=Lax attribute
    - SameSite=Lax balances security and usability: provides CSRF protection while allowing cookies on top-level navigation (e.g., clicking email links or shared routes)
    - Automatic CSRF protection for state-changing requests (POST, PUT, DELETE)
    - Immune to XSS token theft
    - No JavaScript access to tokens
  - **ALTERNATIVE**: localStorage (requires explicit security mitigations)
    - ⚠️ Only use if HTTP-only cookies are not feasible (e.g., cross-origin API)
    - **Required mitigations if using localStorage:**
      - Content Security Policy (CSP) with strict directives (`script-src 'self'`)
      - Strict XSS input sanitization and output encoding
      - Short-lived access tokens (15 min max, configurable via `JWT_ACCESS_TOKEN_TTL`)
      - Automatic refresh token rotation on each use
      - Token revocation endpoint (`POST /api/v1/auth/revoke`)
      - Regular security audits for XSS vulnerabilities
- **Rate limits (authenticated):**
  - 10 uploads per hour (same as anonymous)
  - 50 uploads per day
  - 200 uploads per month
- Configurable per-user quotas via database (quota_hourly, quota_daily, quota_monthly)

**Migration Path:**
- When anonymous user creates account:
  - All routes with session_id are transferred
  - Endpoint: `POST /api/v1/auth/claim-routes`
  - Session is preserved for backward compatibility

### 6.2 Authentication Endpoints

**Signup:**
```http
POST /api/v1/auth/signup
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword123"
}

Response:
{
  "token": "<JWT_ACCESS_TOKEN>",
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "created_at": "2026-01-31T12:00:00Z"
  }
}
```

**Login:**
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword123"
}

Response:
{
  "token": "<JWT_ACCESS_TOKEN>",
  "user": { ... }
}
```

**Claim Routes:**
```http
POST /api/v1/auth/claim-routes
Authorization: Bearer {token}
Content-Type: application/json

{
  "session_id": "previous-session-uuid"
}

Response:
{
  "message": "Routes claimed successfully",
  "routes_claimed": 12
}
```

### 6.3 Security Requirements

**Password Security:**
- Minimum 10-12 characters (configurable, default: 10)
- Complexity requirements:
  - At least one uppercase letter
  - At least one lowercase letter
  - At least one number
  - At least one special character (!@#$%^&*()_+-=[]{}|;:,.<>?)
- Hashed with bcrypt (cost factor 12, configurable via `BCRYPT_COST`)
- Never logged or stored in plaintext
- Password history: prevent reuse of last 3 passwords (optional, configurable)

**JWT Security:**
- Algorithm: Configurable (RS256 recommended for production, HS256 for simpler deployments)
  - **RS256 (Recommended)**: Use asymmetric key pairs with key rotation support
    - Private key: `JWT_PRIVATE_KEY` environment variable (PEM format)
    - Public key: `JWT_PUBLIC_KEY` environment variable (PEM format)
    - Key rotation: Support multiple public keys for validation during rotation
  - **HS256 (Alternative)**: Symmetric secret key from environment variable
    - Secret key: `JWT_SECRET` environment variable (minimum 32 characters)
- Configurable via `JWT_ALGORITHM` environment variable (default: RS256)

**Token Strategy:**
- **Access tokens**: Short-lived (15-60 minutes, configurable via `JWT_ACCESS_TOKEN_TTL`)
  - Used for API authentication
  - Payload: user_id, exp, type: 'access'
- **Refresh tokens**: Long-lived (up to 30 days, configurable via `JWT_REFRESH_TOKEN_TTL`)
  - Used to obtain new access tokens
  - Payload: user_id, exp, type: 'refresh', jti (unique token ID)
  - Stored in database for revocation
  - Automatic rotation: Issue new refresh token on each use
- **Revocation endpoint**: `POST /api/v1/auth/revoke` to invalidate refresh tokens
  - Revoke single token by jti
  - Revoke all tokens for a user

**Rate Limiting:**
- Login: 5 attempts/15 minutes per IP
- Signup: 3 signups/hour per IP
- Password reset: 3 attempts/hour per email
- Prevent brute force attacks

**Account Lockout:**
- Lock account after N failed login attempts per account (configurable, default: 10)
- Lockout duration: Exponential backoff (1 min, 5 min, 15 min, 1 hour)
- Unlock methods:
  - Automatic unlock after lockout duration expires
  - Administrative unlock via admin API
  - Email-based unlock link (future enhancement)
- Rate limiting escalation: Increase lockout duration with repeated failures

**HTTPS:**
- All endpoints require HTTPS in production
- HTTP-only cookies for session/token storage
- SameSite=Lax cookie attribute (balances CSRF protection with cross-site usability for email links and shared routes)
- Secure flag on cookies in production

---

## 7. Telegram Bot Specification

### 7.1 Bot Features

**Commands:**
```
/start  - Welcome message, usage instructions
/help   - Detailed help and tips
/history - View recent analyses (future)
```

**Photo Upload:**
1. User sends photo to bot
2. Bot replies: "🔍 Analyzing your route..."
3. Bot uploads image to backend API
4. Bot creates route record
5. Bot triggers analysis (async)
6. Bot polls for completion
7. Bot replies with prediction:
   ```
   ✅ Analysis Complete!

   📊 Grade: V5
   🎯 Confidence: 75%

   📝 Explanation:
   This route is graded V5 because:
   • High crimp count (5 crimps)
   • Long reach moves (max 1.8m)
   • Steep wall angle (45°)

   Was this accurate?
   [👍 Yes] [👎 No]
   ```

### 7.2 Bot Implementation

**Technology:**
- Library: `python-telegram-bot` v20.7
- Deployment: Railway ($10/month)
- Mode: Polling (development), Webhook (production)

**Configuration:**
```python
# Environment variables
TELEGRAM_BOT_TOKEN=xxx  # From @BotFather
TELEGRAM_API_URL=https://api.yourdomain.com
TELEGRAM_LOG_DETAILED_PII=false  # Privacy: hash user IDs in logs (MUST be false in production)
```

**Privacy:**
- User IDs hashed (SHA-256) in logs by default
- **CRITICAL: `TELEGRAM_LOG_DETAILED_PII` MUST be false in production**
  - Read-only in production runtime (hard-coded override if environment attempts to set to true)
  - Deployment checklist validates this before release
  - See deployment guide for production configuration validation
- Detailed PII logging permitted ONLY in local development environments
- Never store Telegram usernames in database
- Comply with GDPR, CCPA

**Message Retention Policy:**
- Telegram messages: Not stored (ephemeral processing only)
- Uploaded photos: Stored in R2/Supabase Storage with route records
  - Retention: Indefinite for authenticated users (user-controlled deletion)
  - Retention: 90 days for anonymous users, then auto-deleted
- Analysis results: Stored in database with route records
  - Same retention as photos

**GDPR Compliance - Right to Be Forgotten:**
- User data deletion endpoint: `DELETE /api/v1/users/{user_id}`
  - Deletes user account record
  - Deletes all associated routes, predictions, feedback
  - Deletes all associated images from storage (R2/Supabase)
  - Cascades to all related tables via ON DELETE CASCADE
- Anonymous user cleanup: Automatic deletion of routes older than 90 days for anonymous users
- Deletion confirmation: Returns deleted resource counts
- Audit log: Record deletion requests (timestamp, user_id, deleted resource counts) for compliance

**Deployment Checklist - Privacy Enforcement:**
- [ ] Verify `TELEGRAM_LOG_DETAILED_PII=false` in production environment variables
- [ ] Test production startup to confirm PII logging is disabled
- [ ] Validate automatic anonymous route cleanup job is scheduled (cron or Railway scheduler)
- [ ] Test GDPR deletion endpoint with test user data

**Rate Limiting:**
- Anonymous bot users: 5 uploads/hour
- Future: Link Telegram account to web account for higher quotas

---

## 8. Monitoring & Observability

### 8.1 Metrics to Track

**Application Metrics:**
```
# Uploads
upload_requests_total{status="success|error", file_type="jpg|png"}
upload_duration_seconds{bucket="0.1,0.5,1,2,5"}
upload_file_size_bytes{bucket="100k,500k,1M,5M,10M"}

# Analysis
analysis_requests_total{status="success|error"}
analysis_duration_seconds{bucket="5,10,15,20,30"}
analysis_queue_depth

# Cache
cache_hits_total{key_type="prediction|image_hash"}
cache_misses_total{key_type="prediction|image_hash"}

# Rate Limiting
ratelimit_exceeded_total{endpoint="/upload|/analyze"}

# Health
celery_workers_active
celery_tasks_queued
database_connections_active
redis_memory_usage_bytes
```

**Business Metrics:**
```
# User Engagement
routes_uploaded_total
predictions_generated_total
feedback_submitted_total
routes_shared_total

# User Growth
signups_total
active_users_daily
active_users_monthly

# Accuracy
prediction_accuracy_rate  # From feedback
feedback_response_rate
```

### 8.2 Monitoring Tools

**Railway Built-in:**
- CPU/memory usage graphs
- Request count
- Response time (P50, P95, P99)
- Error rate
- Deployment history

**Sentry (Free Tier):**
- Error tracking (5K events/month)
- Performance monitoring
- Release tracking
- User feedback

**Custom Dashboards:**
- Grafana (future, if needed)
- Prometheus endpoint: `/metrics`

### 8.3 Alerting

**Critical Alerts:**
- Uptime < 99% (weekly)
- Error rate > 1% (hourly)
- Celery queue depth > 100 (5 minutes)
- Database connections > 80% (immediate)
- **Disk usage > 80% (hourly)** - Changed from daily to hourly checks for faster response
- **Redis memory > 80% (immediate)** - Redis allocation: 256MB, alert at ~205MB usage
- **R2 storage > 8GB (daily)** - Cloudflare R2 free tier: 10GB, alert at 80% (8GB) before quota exhaustion
- **Sentry quota > 4,000 events (daily)** - Sentry free tier: 5K events/month, alert at 80% (4,000 events/month)

**Warning Alerts:**
- Analysis latency P95 > 20s
- Cache hit rate < 20%
- Worker utilization > 80%
- Redis memory > 60% (warning threshold before critical)
- R2 storage > 6GB (60% warning before 80% critical)
- Sentry quota > 3,000 events (60% warning)

**Channels:**
- Email (Railway notifications)
- Slack webhook (future)

---

## 9. Testing Strategy

### 9.1 Unit Tests

**Coverage Target:** ≥85% (current stage), ≥90% (when complete)

**Tools:**
- pytest
- pytest-cov
- pytest-asyncio
- pytest-mock

**Test Categories:**
- API endpoint tests
- Model inference tests
- Database operations
- Authentication/authorization
- Rate limiting
- Caching logic

### 9.2 Integration Tests

**Test Scenarios:**
- Full upload → analyze → prediction flow
- Anonymous → authenticated migration
- Celery task execution
- Redis caching
- Database transactions

**Tools:**
- TestClient (FastAPI)
- Mock Supabase/R2 (in-memory)
- Docker Compose for local integration

### 9.3 End-to-End Tests

**Test Journeys:**
1. New user uploads route → receives prediction
2. User marks start/finish → re-analyzes
3. User submits feedback → stored
4. User shares route → friend views
5. Anonymous user creates account → routes claimed

**Tools:**
- Playwright (web frontend)
- Manual testing (Telegram bot)

### 9.4 Load Testing

**Tool:** Locust

**Scenarios:**
- 100 concurrent users
- 10 uploads/minute
- 40 routes/day sustained

**Success Criteria:**
- P95 latency < 20s for analysis
- Error rate < 1%
- Queue depth < 50

---

## 10. Deployment Specification

### 10.1 Railway Configuration

**File: `railway.toml`**
```toml
[build]
builder = "nixpacks"
buildCommand = "pip install -r requirements.txt"

[deploy]
startCommand = "uvicorn src.app:application --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health/ready"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

[[services]]
name = "fastapi-backend"
port = 8000

[[services]]
name = "celery-worker"
startCommand = "celery -A src.celery_app worker --loglevel=info --concurrency=2"

[[services]]
name = "telegram-bot"
startCommand = "python src/telegram_bot/bot.py"
```

### 10.2 Environment Variables

**Backend:**
```bash
# Application
BA_APP_NAME=bouldering-analysis
BA_DEBUG=false
BA_LOG_LEVEL=INFO

# Database (PostgreSQL)
# DATABASE_URL points to Railway PostgreSQL for application database (routes, users, predictions, etc.)
DATABASE_URL=${POSTGRES_URL}  # Railway provides this automatically

# Supabase (Storage Only)
# BA_SUPABASE_URL and BA_SUPABASE_KEY are ONLY for Supabase Storage (image uploads)
# NOT used for database - Railway PostgreSQL is the primary database
BA_SUPABASE_URL=https://xxx.supabase.co
BA_SUPABASE_KEY=xxx  # Supabase anon/service key for storage access

# Redis
REDIS_URL=${REDIS_URL}  # Railway provides this automatically

# Storage (Cloudflare R2 for images - alternative to Supabase Storage)
R2_ACCESS_KEY_ID=xxx
R2_SECRET_ACCESS_KEY=xxx
R2_BUCKET_NAME=route-images
R2_ENDPOINT=https://xxx.r2.cloudflarestorage.com

# Authentication
# CRITICAL: JWT_SECRET must be cryptographically secure
# Generate: openssl rand -hex 32  (for HS256)
# OR: openssl genrsa -out private.pem 2048 && openssl rsa -in private.pem -pubout -out public.pem  (for RS256)
# DO NOT use placeholder values in production
JWT_SECRET=<GENERATE_SECURE_SECRET>  # Min 32 chars for HS256, store in Railway secrets
JWT_ALGORITHM=RS256  # RS256 (recommended) or HS256

# Token Lifetime Configuration (Precedence: specific TTL settings override deprecated JWT_EXPIRATION_DAYS)
JWT_ACCESS_TOKEN_TTL=60   # Access token lifetime in MINUTES (default: 60 min = 1 hour)
JWT_REFRESH_TOKEN_TTL=30  # Refresh token lifetime in DAYS (default: 30 days)
# For RS256 (asymmetric key pairs):
# JWT_PRIVATE_KEY=<PEM_PRIVATE_KEY>  # Store in Railway secrets, used for signing
# JWT_PUBLIC_KEY=<PEM_PUBLIC_KEY>    # Can be public, used for verification

# DEPRECATED: JWT_EXPIRATION_DAYS - DO NOT USE
# JWT_EXPIRATION_DAYS=30  # REMOVED - Use JWT_REFRESH_TOKEN_TTL instead
# This variable is deprecated and ignored if JWT_ACCESS_TOKEN_TTL or JWT_REFRESH_TOKEN_TTL are set.
# Migration: Replace JWT_EXPIRATION_DAYS with:
#   - JWT_ACCESS_TOKEN_TTL (in minutes) for access token lifetime
#   - JWT_REFRESH_TOKEN_TTL (in days) for refresh token lifetime

# Rate Limiting
# Canonical quota policy (applied to both anonymous and authenticated by default):
# - 10 uploads per hour
# - 50 uploads per day
# - 200 uploads per month
# Authenticated users can have custom quotas via database (quota_hourly, quota_daily, quota_monthly)
BA_MAX_UPLOAD_SIZE_MB=10
BA_UPLOADS_PER_HOUR=10     # Anonymous & default authenticated: 10/hour
BA_UPLOADS_PER_DAY=50      # Anonymous & default authenticated: 50/day
BA_UPLOADS_PER_MONTH=200   # Anonymous & default authenticated: 200/month

# Monitoring
SENTRY_DSN=https://xxx@sentry.io/xxx
```

**Telegram Bot:**
```bash
TELEGRAM_BOT_TOKEN=xxx  # From @BotFather
TELEGRAM_API_URL=https://api.yourdomain.com
TELEGRAM_LOG_DETAILED_PII=false
```

**Web Frontend:**
```bash
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
```

### 10.3 CI/CD Pipeline

**GitHub Actions:**
```yaml
name: Deploy to Railway

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/ --cov=src --cov-fail-under=85

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Railway
        uses: railway/action@v1
        with:
          railway-token: ${{ secrets.RAILWAY_TOKEN }}
```

---

## 11. Success Criteria

### 11.1 Technical Metrics

**Performance:**
- [ ] P95 analysis latency < 20 seconds
- [ ] P95 upload latency < 1 second
- [ ] Uptime > 99% during beta

**Reliability:**
- [ ] Error rate < 1% of requests
- [ ] Zero data loss incidents
- [ ] < 5 critical bugs per month

**Scalability:**
- [ ] Handle 100 concurrent users
- [ ] Process 100 routes/day peak load
- [ ] Database < 80% capacity

**Quality:**
- [ ] Test coverage ≥ 85%
- [ ] Pylint score ≥ 8.5/10
- [ ] Zero security vulnerabilities (critical/high)

### 11.2 Business Metrics

**User Engagement:**
- [ ] 100 beta signups
- [ ] 50% weekly active users
- [ ] 500+ routes analyzed (total)
- [ ] 30% feedback submission rate

**User Satisfaction:**
- [ ] < 5 support tickets/week
- [ ] Feedback accuracy rate > 60%
- [ ] 80% user retention (monthly)

**Cost Efficiency:**
- [ ] Infrastructure cost < $90/month
- [ ] Cost per route < $0.05
- [ ] Cost per user < $1/month

### 11.3 Go-Live Checklist

**Pre-Launch - Security:**
- [ ] **Generate secure JWT_SECRET** using cryptographically secure method:
  - For HS256: `openssl rand -hex 32` (minimum 32 characters)
  - For RS256: `openssl genrsa -out private.pem 2048 && openssl rsa -in private.pem -pubout -out public.pem`
- [ ] **Store JWT_SECRET in Railway secrets** (or chosen secrets manager)
  - Never commit secrets to Git
  - Verify secrets are not exposed in logs or error messages
- [ ] **Verify JWT_SECRET is set and valid** before production deployment
  - Test JWT generation and validation with production secret
  - Confirm JWT_ALGORITHM matches key type (RS256 vs HS256)
- [ ] **Validate TELEGRAM_LOG_DETAILED_PII=false** in production environment
  - Verify production bootstrap code overrides any attempt to set to true
  - Test that user IDs are hashed in production logs

**Pre-Launch - Testing:**
- [ ] All endpoints tested end-to-end
- [ ] Load testing passed (100 concurrent users)
- [ ] Security audit completed
- [ ] Monitoring and alerting configured
- [ ] Documentation complete (user guide, API docs)
- [ ] Backup and recovery tested
- [ ] Incident response plan documented
- [ ] GDPR deletion endpoint tested with test data

**Launch Day:**
- [ ] Deploy to production
- [ ] Verify health checks passing
- [ ] Verify environment variables set correctly (JWT_SECRET, TELEGRAM_LOG_DETAILED_PII, etc.)
- [ ] Invite 10-20 alpha users
- [ ] Monitor metrics closely (hourly)
- [ ] Be ready for hotfixes

**Week 1:**
- [ ] Daily metrics review
- [ ] Respond to user feedback
- [ ] Fix critical bugs within 24 hours
- [ ] Scale infrastructure if needed

---

## 12. Risk Mitigation

### 12.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CPU inference too slow | Medium | High | ONNX optimization, batching, add 2nd worker |
| Storage costs exceed budget | Low | Medium | Image compression, Cloudflare R2 free tier |
| Database performance | Low | Medium | Indexes, connection pooling, read replicas |
| Uptime < 99% | Low | High | Health checks, auto-restart, monitoring |
| Model accuracy < 60% | Medium | High | User feedback loop, model retraining |

### 12.2 Business Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| < 50 beta signups | Medium | Medium | Marketing, referral program, Telegram bot |
| Low user engagement | Medium | High | Gamification, feedback rewards, social features |
| Abuse/spam | Low | High | Rate limiting, CAPTCHA, manual review |
| High support burden | Medium | Medium | Self-service docs, FAQ, Telegram community |
| Competitor launches first | Low | Medium | Focus on quality, unique features (explainability) |

---

## 13. Post-Beta Roadmap

### 13.1 Public Launch (Month 11+)

**Requirements:**
- [ ] 70%+ prediction accuracy (±1 grade)
- [ ] 80%+ uptime over 6-month beta
- [ ] < 1% critical bug rate
- [ ] 100+ beta testimonials
- [ ] Cost-effective at scale (proven)

**Launch Activities:**
- Public announcement
- Press release
- Social media campaign
- Product Hunt launch
- Gym partnerships

### 13.2 Monetization Strategy

**Freemium Model:**
- Free tier: 10 routes/month
- Pro tier: $5/month for 100 routes
- Gym tier: $50/month for unlimited (B2B)

**Revenue Target:**
- 1000 users × 10% conversion × $5/month = $500/month
- Covers infrastructure ($140/month at 1000 users)
- 72% profit margin (($500 - $140) / $500)
- 3.57x revenue-to-cost ratio ($500 / $140)

### 13.3 Feature Roadmap

**Short-term (3-6 months):**
- GPU workers for faster inference
- Mobile app (React Native)
- Gym partnerships (API access)
- Advanced analytics (user insights)

**Long-term (6-12 months):**
- Graph Neural Networks (GNN) for grading
- Beta path prediction
- Video analysis (future research)
- Setter tooling mode

---

## 14. References

- [DESIGN.md](../docs/DESIGN.md) - Overall system architecture
- [MIGRATION_PLAN.md](MIGRATION_PLAN.md) - Flask → FastAPI migration
- [FRONTEND_WORKFLOW.md](../docs/FRONTEND_WORKFLOW.md) - Web frontend guide
- [TELEGRAM_BOT.md](../docs/TELEGRAM_BOT.md) - Telegram bot guide
- [VERCEL_SETUP.md](../docs/VERCEL_SETUP.md) - Vercel deployment
- [MODEL_PRETRAIN.md](../docs/MODEL_PRETRAIN.md) - ML model specs

---

**Version History:**
- v1.0 (2026-01-31): Initial specification based on user requirements
