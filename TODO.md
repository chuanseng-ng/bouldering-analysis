# TODO — Deferred Items

Items here were intentionally deferred from their originating PR. Each entry
records **what**, **why it was deferred**, and **what triggers re-evaluation**.

A dedicated review PR (**PR-10.10**) is scheduled after all PR-10 sub-items
complete to decide which of these should be actioned before the project is
considered production-ready.

---

## Security

### S1 — Rate limiting on `/analyze`, `/constraints`, `/feedback`

**Deferred from**: PR-10.3–10.5 security review
**Severity**: HIGH

`POST /routes/{id}/analyze` runs a 30-second ML pipeline and is currently
unprotected. An attacker can queue many concurrent requests to exhaust the
thread pool. `/constraints` is similarly compute-heavy. `/feedback` accepts
anonymous writes and can be flooded.

**Why deferred**: The right rate-limit values (requests/min/IP) depend on
observed production traffic patterns and the deployment topology
(single-process vs. multi-worker vs. serverless). Picking the wrong numbers
before launch risks blocking legitimate users. A Redis-backed solution
(e.g. `slowapi`) requires an infrastructure decision that should happen
alongside deployment planning (PR-10.8 — Vercel/server setup).

**What to do when revisited**:
- For a single-process deployment: reuse the existing `_UploadRateLimiter`
  pattern from `src/app.py` (`BA_RATE_LIMIT_ANALYZE`, `BA_RATE_LIMIT_FEEDBACK`
  config vars; 0 = disabled).
- For multi-process/serverless: add `slowapi` + Redis or use an edge-layer
  rate limiter (Cloudflare, Vercel middleware).
- Suggested limits: `/analyze` ≤ 5/min/IP, `/constraints` ≤ 10/min/IP,
  `/feedback` ≤ 20/min/IP.

---

### S2 — CORS wildcard default (`BA_CORS_ORIGINS = ["*"]`)

**Deferred from**: PR-10.3–10.5 security review
**Severity**: MEDIUM

The default allows non-credentialed cross-origin requests from any website.
`allow_credentials=True` is also set, though browsers block it with a wildcard
origin. In production, both should be scoped to the actual frontend domain.

**Why deferred**: The frontend deployment URL (Vercel domain or custom domain)
is not yet known — PRs 10.7–10.8 handle deployment. Hardcoding a placeholder
origin now would silently break the frontend.

**What to do when revisited**:
1. Set `BA_CORS_ORIGINS=["https://your-app.vercel.app"]` in the production
   environment once the Vercel URL is known (PR-10.8).
2. Remove `allow_credentials=True` from the CORS middleware unless cookies or
   HTTP auth are introduced (current auth uses `X-API-Key` header, which does
   not require `credentials: include` on the client).

---

### S3 — Internal exception messages exposed in 422 responses

**Deferred from**: PR-10.3–10.5 security review
**Severity**: LOW

`set_constraints` passes `str(e)` directly from `RouteGraphError`,
`FeatureExtractionError`, and `GradeEstimationError` into the HTTP 422 detail.
Example leakage: `"Start hold IDs [0, 5] not found in graph (12 nodes)"`,
`"feature vector has 33 keys, expected 34"`.

**Why deferred**: At the current stage the API is developer-facing only. These
detailed messages are useful for debugging during integration. Sanitising them
before a public user base exists adds noise without security benefit.

**What to do when revisited**:
Replace `detail=str(e)` with a fixed user-facing message per exception type,
and log the original `str(e)` at WARNING level server-side only:

```python
_PIPELINE_ERROR_MESSAGES: dict[type[Exception], str] = {
    RouteGraphError: "Could not build route graph. Check that start and finish hold IDs are valid.",
    FeatureExtractionError: "Feature extraction failed. The route graph may be malformed.",
    GradeEstimationError: "Grade estimation failed. Please try again.",
}
# ...
except (RouteGraphError, FeatureExtractionError, GradeEstimationError) as e:
    logger.warning("Pipeline error", extra={"error": str(e), "route_id": route_id_str})
    raise HTTPException(422, detail=_PIPELINE_ERROR_MESSAGES[type(e)]) from e
```

---

### S4 — Race condition on `pending → processing` status transition

**Deferred from**: PR-10.3–10.5 security review
**Severity**: MEDIUM

The status read and the `UPDATE status='processing'` are not atomic. Two
concurrent `/analyze` requests on the same route can both pass the
`status == 'pending'` check and both proceed to run the pipeline. The second
bulk-insert of holds will fail with a `UNIQUE (route_id, hold_id)` constraint
violation, producing a 500 to the second caller — data integrity is preserved
but compute is wasted.

**Why deferred**: A proper atomic fix requires a conditional Supabase update
(`UPDATE routes SET status='processing' WHERE id=$1 AND status='pending'`)
with a check on rows-affected. This needs either a new `supabase_client`
function or raw SQL execution, which is out of scope for the current PR series.
The UNIQUE constraint ensures no duplicate rows are ever written.

**What to do when revisited**:
Add `update_record_if_status(table, record_id, new_status, expected_status)`
to `supabase_client.py`. This would chain
`.update({"status": new_status}).eq("id", record_id).eq("status", expected_status).execute()`
and return `False` (not updated) instead of raising if `result.data` is empty.
Use it in `analyze_route` to replace the read-then-write pattern.

---

## Changelog

- **2026-03-21**: Created. Recorded S1–S4 deferred from PR-10 security review.
