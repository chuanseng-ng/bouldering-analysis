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
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    quota_daily INT DEFAULT 50,
    quota_monthly INT DEFAULT 200
);
```

**routes table:**
```sql
CREATE TABLE routes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_url TEXT NOT NULL,
    wall_angle FLOAT,
    owner_id VARCHAR(255) NOT NULL,  -- User ID or session ID
    owner_type VARCHAR(20) DEFAULT 'anonymous',  -- 'anonymous' | 'authenticated'
    start_hold_ids INT[],  -- Array of start hold IDs
    finish_hold_id INT,    -- Single finish hold ID
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_routes_owner ON routes(owner_id, owner_type);
CREATE INDEX idx_routes_created_at ON routes(created_at DESC);
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
    route_id UUID REFERENCES routes(id) UNIQUE ON DELETE CASCADE,
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
- Rate limits apply (lower quotas)

**Authenticated Users:**
- Email + password signup/login
- JWT token issued (30-day expiry)
- Token stored in localStorage or HTTP-only cookie
- Higher rate limits and quotas

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
  "token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
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
  "token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
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
- Minimum 8 characters
- Hashed with bcrypt (cost factor 12)
- Never logged or stored in plaintext

**JWT Security:**
- HS256 algorithm
- 30-day expiration
- Secret key from environment variable
- Payload: user_id, exp

**Rate Limiting:**
- Login: 5 attempts/15 minutes per IP
- Signup: 3 signups/hour per IP
- Prevent brute force attacks

**HTTPS:**
- All endpoints require HTTPS in production
- HTTP-only cookies for session/token storage
- SameSite=Lax cookie attribute

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
TELEGRAM_LOG_DETAILED_PII=false  # Privacy: hash user IDs in logs
```

**Privacy:**
- User IDs hashed (SHA-256) in logs by default
- Detailed PII logging opt-in only (for debugging)
- Never store Telegram usernames
- Comply with GDPR, CCPA

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
- Disk usage > 80% (daily)

**Warning Alerts:**
- Analysis latency P95 > 20s
- Cache hit rate < 20%
- Worker utilization > 80%

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

# Database
DATABASE_URL=${POSTGRES_URL}  # Railway provides
BA_SUPABASE_URL=https://xxx.supabase.co
BA_SUPABASE_KEY=xxx

# Redis
REDIS_URL=${REDIS_URL}  # Railway provides

# Storage
R2_ACCESS_KEY_ID=xxx
R2_SECRET_ACCESS_KEY=xxx
R2_BUCKET_NAME=route-images
R2_ENDPOINT=https://xxx.r2.cloudflarestorage.com

# Authentication
JWT_SECRET=random-secret-key-change-in-production
JWT_EXPIRATION_DAYS=30

# Rate Limiting
BA_MAX_UPLOAD_SIZE_MB=10
BA_UPLOADS_PER_MINUTE=10
BA_UPLOADS_PER_DAY=200

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

**Pre-Launch:**
- [ ] All endpoints tested end-to-end
- [ ] Load testing passed (100 concurrent users)
- [ ] Security audit completed
- [ ] Monitoring and alerting configured
- [ ] Documentation complete (user guide, API docs)
- [ ] Backup and recovery tested
- [ ] Incident response plan documented

**Launch Day:**
- [ ] Deploy to production
- [ ] Verify health checks passing
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
- 3.5x profit margin

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
