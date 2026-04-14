"""Analysis, constraints, prediction, and feedback endpoints.

This module provides all endpoints related to running the ML pipeline on an
uploaded route image, setting start/finish hold constraints, retrieving grade
predictions, and submitting feedback.

Endpoints:
    POST /api/v1/routes/{route_id}/analyze  — Run hold detection + classification
    GET  /api/v1/routes/{route_id}/holds    — List classified holds for a route
    PUT  /api/v1/routes/{route_id}/constraints — Build graph, estimate grade, store result
    GET  /api/v1/routes/{route_id}/prediction  — Retrieve latest prediction
    POST /api/v1/routes/{route_id}/feedback    — Submit user feedback
"""

import asyncio
import io
import urllib.parse
import uuid
from typing import Annotated, Any

import httpx
import PIL.Image as PILImage
from fastapi import APIRouter, HTTPException, Path, status
from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import get_settings
from src.database.supabase_client import (
    SupabaseClientError,
    delete_records,
    insert_record,
    insert_records_bulk,
    select_record_by_id,
    select_records,
    update_record,
)
from src.explanation import ExplanationError, ExplanationResult, generate_explanation
from src.features import FeatureExtractionError, RouteFeatures, assemble_features
from src.graph import (
    ClassifiedHold,
    RouteGraphError,
    apply_route_constraints,
    build_route_graph,
)
from src.grading import (
    GradeEstimationError,
    HeuristicGradeResult,
    MLGradeResult,
    estimate_grade_heuristic,
    estimate_grade_ml,
)
from src.inference.classification import classify_holds
from src.inference.crop_extractor import extract_hold_crops
from src.inference.detection import detect_holds
from src.logging_config import get_logger
from src.routes.routes import RouteStatus
from src.routes.shared import ErrorResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analysis"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROUTES_TABLE = "routes"
_HOLDS_TABLE = "holds"
_FEATURES_TABLE = "features"
_PREDICTIONS_TABLE = "predictions"
_FEEDBACK_TABLE = "feedback"

_GRADE_PATTERN = r"^V(1[0-7]|[0-9])$"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AnalyzeResponse(BaseModel):
    """Response model for the analyze endpoint.

    Attributes:
        route_id: UUID of the analyzed route.
        hold_count: Number of holds detected and classified.
        status: Updated route processing status (``"done"`` on success).
    """

    route_id: str
    hold_count: int
    status: RouteStatus


class HoldResponse(BaseModel):
    """Response model for a single classified hold.

    Attributes:
        hold_id: Sequential hold index (0-based) within the route.
        x_center: Horizontal centre of the bounding box (0–1).
        y_center: Vertical centre of the bounding box (0–1).
        width: Bounding box width as a fraction of image width (0–1).
        height: Bounding box height as a fraction of image height (0–1).
        detection_class: Raw detection class (``"hold"`` or ``"volume"``).
        detection_confidence: Detection confidence score (0–1).
        hold_type: Classified hold type.
        type_confidence: Classification confidence score (0–1).
        type_probabilities: Full probability distribution over all hold types.
    """

    hold_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    detection_class: str
    detection_confidence: float
    hold_type: str
    type_confidence: float
    type_probabilities: dict[str, float]


class HoldsListResponse(BaseModel):
    """Response model for the list holds endpoint.

    Attributes:
        route_id: UUID of the route.
        holds: List of classified holds ordered by hold_id.
        count: Total number of holds.
    """

    route_id: str
    holds: list[HoldResponse]
    count: int


class RouteConstraints(BaseModel):
    """Request model for setting route constraints.

    Attributes:
        start_hold_ids: Non-empty list of hold IDs marking the start position(s).
            Multiple start holds are supported (e.g. two-handed start).
        finish_hold_ids: Exactly one hold ID marking the finish position.
            Limited to a single element (``min_length=1, max_length=1``) because
            ``apply_route_constraints`` accepts a single ``finish_id: int``.
            A list is accepted in the request for API consistency, but only
            the first (and only) element is used when building the route graph.
    """

    start_hold_ids: Annotated[list[int], Field(min_length=1)]
    finish_hold_ids: Annotated[list[int], Field(min_length=1, max_length=1)]

    @field_validator("start_hold_ids", "finish_hold_ids")
    @classmethod
    def validate_non_negative(cls, v: list[int]) -> list[int]:
        """Validate that all hold IDs are non-negative integers.

        Args:
            v: List of hold IDs to validate.

        Returns:
            Validated list of hold IDs.

        Raises:
            ValueError: If any hold ID is negative.
        """
        if any(h < 0 for h in v):
            raise ValueError("all hold IDs must be non-negative")
        return v


class PredictionResponse(BaseModel):
    """Response model for a grade prediction.

    Attributes:
        route_id: UUID of the route.
        estimator_type: Which estimator produced the result (``"heuristic"`` or
            ``"ml"``).
        grade: Predicted V-scale grade label (e.g., ``"V5"``).
        grade_index: Ordinal grade index (0–17).
        confidence: Confidence score (0–1).
        difficulty_score: Normalised difficulty score (0–1).
        explanation: Serialised explanation dict, or None.
        model_version: ML model version string, or None for heuristic.
    """

    route_id: str
    estimator_type: str
    grade: str
    grade_index: int
    confidence: float
    difficulty_score: float
    explanation: dict[str, Any] | None = None
    model_version: str | None = None


class FeedbackCreate(BaseModel):
    """Request model for submitting route feedback.

    At least one of ``user_grade``, ``is_accurate``, or ``comments`` must be
    provided.

    Attributes:
        user_grade: User's own V-grade estimate (V0–V17), or None.
        is_accurate: Whether the system grade was accurate, or None.
        comments: Free-text comments, or None.
    """

    user_grade: str | None = Field(
        default=None,
        pattern=_GRADE_PATTERN,
        description="User's own V-grade estimate (V0–V17)",
        examples=["V5", "V10"],
    )
    is_accurate: bool | None = None
    comments: str | None = Field(default=None, max_length=10000)

    @model_validator(mode="after")
    def validate_at_least_one_field(self) -> "FeedbackCreate":
        """Validate that at least one feedback field is non-null.

        Returns:
            Self after validation.

        Raises:
            ValueError: If all fields are None.
        """
        if (
            self.user_grade is None
            and self.is_accurate is None
            and self.comments is None
        ):
            raise ValueError(
                "at least one of 'user_grade', 'is_accurate', or 'comments' must be provided"
            )
        return self


class FeedbackResponse(BaseModel):
    """Response model for submitted feedback.

    Attributes:
        id: UUID of the created feedback record.
        route_id: UUID of the route.
        created_at: ISO 8601 timestamp of submission.
    """

    id: str
    route_id: str
    created_at: str


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _db_row_to_hold_response(row: dict[str, Any]) -> HoldResponse:
    """Convert a holds table row to a HoldResponse.

    Args:
        row: Database row dictionary from the holds table.

    Returns:
        HoldResponse model instance.
    """
    probs: dict[str, float] = {}
    for key in (
        "jug",
        "crimp",
        "sloper",
        "pinch",
        "pocket",
        "foothold",
        "unknown",
    ):
        probs[key] = float(row.get(f"prob_{key}", 0.0))

    return HoldResponse(
        hold_id=int(row["hold_id"]),
        x_center=float(row["x_center"]),
        y_center=float(row["y_center"]),
        width=float(row["width"]),
        height=float(row["height"]),
        detection_class=str(row["detection_class"]),
        detection_confidence=float(row["detection_confidence"]),
        hold_type=str(row["hold_type"]),
        type_confidence=float(row["type_confidence"]),
        type_probabilities=probs,
    )


def _db_rows_to_classified_holds(rows: list[dict[str, Any]]) -> list[ClassifiedHold]:
    """Reconstruct ClassifiedHold objects from database rows.

    Args:
        rows: List of row dictionaries from the holds table, ordered by hold_id.

    Returns:
        List of ClassifiedHold instances.
    """
    holds: list[ClassifiedHold] = []
    for row in rows:
        probs: dict[str, float] = {}
        for key in (
            "jug",
            "crimp",
            "sloper",
            "pinch",
            "pocket",
            "foothold",
            "unknown",
        ):
            probs[key] = float(row.get(f"prob_{key}", 0.0))

        # Normalise probs to sum to 1.0 (guards against floating-point DB round-trips)
        total = sum(probs.values())
        if total > 0.0 and abs(total - 1.0) > 0.01:
            probs = {k: v / total for k, v in probs.items()}

        holds.append(
            ClassifiedHold(
                hold_id=int(row["hold_id"]),
                x_center=float(row["x_center"]),
                y_center=float(row["y_center"]),
                width=float(row["width"]),
                height=float(row["height"]),
                detection_class=str(row["detection_class"]),  # type: ignore[arg-type]
                detection_confidence=float(row["detection_confidence"]),
                hold_type=str(row["hold_type"]),
                type_confidence=float(row["type_confidence"]),
                type_probabilities=probs,
            )
        )
    return holds


def _build_prediction_row(
    route_id: str,
    prediction: HeuristicGradeResult | MLGradeResult,
    explanation: ExplanationResult | None,
) -> dict[str, Any]:
    """Build a predictions table row dictionary from a prediction result.

    Args:
        route_id: UUID of the route.
        prediction: Grade prediction result (heuristic or ML).
        explanation: Optional explanation result.

    Returns:
        Dictionary ready for insertion into the predictions table.
    """
    estimator_type = "ml" if isinstance(prediction, MLGradeResult) else "heuristic"
    model_version: str | None = None
    if isinstance(prediction, MLGradeResult):
        model_version = (
            None  # set by train_grade_estimator via metadata; not stored here
        )

    explanation_dict: dict[str, Any] | None = None
    if explanation is not None:
        explanation_dict = explanation.model_dump()

    return {
        "route_id": route_id,
        "estimator_type": estimator_type,
        "grade": prediction.grade,
        "grade_index": prediction.grade_index,
        "confidence": prediction.confidence,
        "difficulty_score": prediction.difficulty_score,
        "explanation": explanation_dict,
        "model_version": model_version,
    }


async def _load_route_or_404(route_id: str) -> dict[str, Any]:
    """Load a route record by ID or raise 404.

    Args:
        route_id: UUID string of the route.

    Returns:
        Route record dictionary.

    Raises:
        HTTPException: 404 if not found, 500 on database error.
    """
    settings = get_settings()
    try:
        record = await asyncio.wait_for(
            asyncio.to_thread(
                select_record_by_id,
                table=_ROUTES_TABLE,
                record_id=route_id,
            ),
            timeout=settings.supabase_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Route load timed out",
            extra={"route_id": route_id},
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out. Please try again.",
        ) from None
    except SupabaseClientError as e:
        logger.error(
            "Failed to load route record",
            extra={"route_id": route_id, "error": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load route record",
        ) from e

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Route not found",
        )

    return record  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# POST /routes/{route_id}/analyze
# ---------------------------------------------------------------------------


@router.post(
    "/routes/{route_id}/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Route not found"},
        409: {"model": ErrorResponse, "description": "Route not in pending state"},
        503: {"model": ErrorResponse, "description": "Model paths not configured"},
        504: {"model": ErrorResponse, "description": "Inference timed out"},
        500: {"model": ErrorResponse, "description": "Analysis pipeline failed"},
    },
)
async def analyze_route(
    route_id: Annotated[
        uuid.UUID,
        Path(description="UUID of the route to analyze"),
    ],
) -> AnalyzeResponse:
    """Run hold detection and classification on a route image.

    Downloads the route image, runs YOLOv8 detection, classifies each hold,
    and stores the results in the holds table.  The route status transitions
    from ``"pending"`` → ``"processing"`` → ``"done"`` (or ``"failed"``).

    This endpoint is synchronous — the full pipeline runs inline, guarded by
    ``settings.inference_timeout_seconds``.  Only routes with
    ``status="pending"`` can be analyzed (returns 409 otherwise).

    Args:
        route_id: UUID of the route to analyze.

    Returns:
        :class:`AnalyzeResponse` with route_id, hold_count, and final status.

    Raises:
        HTTPException: 404 if route not found, 409 if not pending,
            503 if model paths are not configured, 504 on timeout, 500 on error.
    """
    route_id_str = str(route_id)
    settings = get_settings()

    # Guard: both model paths must be configured
    if not settings.detection_model_path or not settings.classification_model_path:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model paths not configured. "
                "Set BA_DETECTION_MODEL_PATH and BA_CLASSIFICATION_MODEL_PATH."
            ),
        )

    # Load route and validate status
    record = await _load_route_or_404(route_id_str)
    current_status = record.get("status") or RouteStatus.PENDING
    if RouteStatus(current_status) != RouteStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Route is not in pending state (current: {current_status})",
        )

    # Mark as processing.
    # Note: status check above and this update are not atomic; a concurrent request
    # on the same route could also pass the check.  The UNIQUE (route_id, hold_id)
    # constraint in the holds table prevents duplicate hold rows if that occurs.
    try:
        await asyncio.to_thread(
            update_record,
            table=_ROUTES_TABLE,
            record_id=route_id_str,
            data={"status": RouteStatus.PROCESSING},
        )
    except SupabaseClientError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update route status",
        ) from e

    # Run pipeline with timeout
    try:
        hold_count = await asyncio.wait_for(
            _run_analysis_pipeline(
                route_id=route_id_str,
                image_url=str(record["image_url"]),
                detection_path=settings.detection_model_path,
                classification_path=settings.classification_model_path,
            ),
            timeout=settings.inference_timeout_seconds,
        )
    except asyncio.TimeoutError:
        await asyncio.to_thread(
            update_record,
            table=_ROUTES_TABLE,
            record_id=route_id_str,
            data={"status": RouteStatus.FAILED},
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Inference timed out. Please try again.",
        ) from None
    except Exception as e:
        logger.error(
            "Analysis pipeline failed",
            extra={"route_id": route_id_str, "error": str(e)},
        )
        try:
            await asyncio.to_thread(
                update_record,
                table=_ROUTES_TABLE,
                record_id=route_id_str,
                data={"status": RouteStatus.FAILED},
            )
        except SupabaseClientError:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis pipeline failed",
        ) from e

    logger.info(
        "Analysis complete",
        extra={"route_id": route_id_str, "hold_count": hold_count},
    )
    return AnalyzeResponse(
        route_id=route_id_str,
        hold_count=hold_count,
        status=RouteStatus.DONE,
    )


def _validate_image_url(image_url: str, supabase_url: str) -> None:
    """Validate that image_url belongs to the configured Supabase storage domain.

    Guards against SSRF by ensuring the download target is the known Supabase
    project host, not an arbitrary internal or external endpoint.

    Args:
        image_url: URL of the route image to download.
        supabase_url: Configured Supabase project URL (e.g.
            ``"https://<project>.supabase.co"``).

    Raises:
        ValueError: If ``image_url`` does not share the host of
            ``supabase_url``, or if either URL is malformed.
    """
    parsed_image = urllib.parse.urlparse(image_url)
    parsed_supabase = urllib.parse.urlparse(supabase_url)

    if not parsed_image.scheme or not parsed_image.netloc:
        raise ValueError(f"Malformed image URL: {image_url!r}")

    if parsed_supabase.netloc and parsed_image.netloc != parsed_supabase.netloc:
        raise ValueError(
            f"Image URL host {parsed_image.netloc!r} does not match "
            f"Supabase host {parsed_supabase.netloc!r}"
        )


async def _run_analysis_pipeline(
    route_id: str,
    image_url: str,
    detection_path: str,
    classification_path: str,
) -> int:
    """Execute the full detection + classification pipeline.

    Downloads the image, runs detection, classification, bulk-inserts holds,
    and marks the route as done.

    Args:
        route_id: UUID string of the route.
        image_url: Public URL of the route image.
        detection_path: Path to the YOLOv8 weights file.
        classification_path: Path to the classifier weights file.

    Returns:
        Number of holds detected and stored.

    Raises:
        Exception: Any error from the pipeline propagates to the caller.
    """

    def _pipeline() -> int:
        """Run blocking pipeline steps in a thread."""
        settings = get_settings()

        # SSRF guard: restrict download to the configured Supabase host
        try:
            _validate_image_url(image_url, settings.supabase_url)
        except ValueError as exc:
            raise RuntimeError(f"Image URL validation failed: {exc}") from exc

        # Download image with size cap (no redirect following to prevent redirect-based SSRF)
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        buf = io.BytesIO()
        with httpx.stream(
            "GET", image_url, timeout=30.0, follow_redirects=False
        ) as stream:
            stream.raise_for_status()
            for chunk in stream.iter_bytes(chunk_size=65536):
                buf.write(chunk)
                if buf.tell() > max_bytes:
                    raise RuntimeError(
                        f"Image exceeds maximum allowed size of {settings.max_upload_size_mb} MB"
                    )
        buf.seek(0)
        image = PILImage.open(buf).convert("RGB")

        # Detection
        detected_holds = detect_holds(image, detection_path)

        if not detected_holds:
            # No holds detected — still mark as done with 0 holds
            update_record(_ROUTES_TABLE, route_id, {"status": RouteStatus.DONE})
            return 0

        # Crop extraction
        crops = extract_hold_crops(image, detected_holds)

        # Classification
        results = classify_holds(crops, classification_path)

        # Build rows for bulk insert
        rows: list[dict[str, Any]] = []
        for hold_id, (detection, classification) in enumerate(
            zip(detected_holds, results, strict=True)
        ):
            probs = classification.probabilities
            rows.append(
                {
                    "route_id": route_id,
                    "hold_id": hold_id,
                    "x_center": detection.x_center,
                    "y_center": detection.y_center,
                    "width": detection.width,
                    "height": detection.height,
                    "detection_class": detection.class_name,
                    "detection_confidence": detection.confidence,
                    "hold_type": classification.predicted_class,
                    "type_confidence": classification.confidence,
                    "prob_jug": probs.get("jug", 0.0),
                    "prob_crimp": probs.get("crimp", 0.0),
                    "prob_sloper": probs.get("sloper", 0.0),
                    "prob_pinch": probs.get("pinch", 0.0),
                    "prob_pocket": probs.get("pocket", 0.0),
                    "prob_foothold": probs.get("foothold", 0.0),
                    "prob_unknown": probs.get("unknown", 0.0),
                }
            )

        insert_records_bulk(_HOLDS_TABLE, rows)
        update_record(_ROUTES_TABLE, route_id, {"status": RouteStatus.DONE})
        return len(rows)

    return await asyncio.to_thread(_pipeline)


# ---------------------------------------------------------------------------
# GET /routes/{route_id}/holds
# ---------------------------------------------------------------------------


@router.get(
    "/routes/{route_id}/holds",
    response_model=HoldsListResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Route not found"},
        500: {"model": ErrorResponse, "description": "Database error"},
    },
)
async def list_holds(
    route_id: Annotated[
        uuid.UUID,
        Path(description="UUID of the route"),
    ],
) -> HoldsListResponse:
    """List all classified holds for a route, ordered by hold_id.

    Args:
        route_id: UUID of the route.

    Returns:
        :class:`HoldsListResponse` with the list of holds and count.

    Raises:
        HTTPException: 404 if route not found, 500 on database error.
    """
    route_id_str = str(route_id)

    # Verify route exists
    await _load_route_or_404(route_id_str)

    settings = get_settings()
    try:
        rows = await asyncio.wait_for(
            asyncio.to_thread(
                select_records,
                _HOLDS_TABLE,
                {"route_id": route_id_str},
                "*",
                "hold_id.asc",
            ),
            timeout=settings.supabase_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out. Please try again.",
        ) from None
    except SupabaseClientError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve holds",
        ) from e

    holds = [_db_row_to_hold_response(row) for row in rows]
    return HoldsListResponse(
        route_id=route_id_str,
        holds=holds,
        count=len(holds),
    )


# ---------------------------------------------------------------------------
# PUT /routes/{route_id}/constraints
# ---------------------------------------------------------------------------


@router.put(
    "/routes/{route_id}/constraints",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Route not found"},
        409: {"model": ErrorResponse, "description": "Route not in done state"},
        422: {"model": ErrorResponse, "description": "Invalid hold IDs"},
        500: {"model": ErrorResponse, "description": "Pipeline failed"},
    },
)
async def set_constraints(
    route_id: Annotated[
        uuid.UUID,
        Path(description="UUID of the route"),
    ],
    constraints: RouteConstraints,
) -> PredictionResponse:
    """Set start/finish hold constraints and run the grading pipeline.

    Loads the route's classified holds, builds the route graph with the
    provided start and finish hold IDs, extracts features, estimates the
    grade, generates an explanation, and persists all results.

    Only routes with ``status="done"`` (i.e., holds already detected) can
    be processed (returns 409 otherwise).

    Args:
        route_id: UUID of the route.
        constraints: Start and finish hold IDs.

    Returns:
        :class:`PredictionResponse` with the grade and explanation.

    Raises:
        HTTPException: 404 if not found, 409 if not done, 422 for bad IDs,
            500 on pipeline failure.
    """
    route_id_str = str(route_id)
    settings = get_settings()

    # Load and validate route status
    record = await _load_route_or_404(route_id_str)
    current_status = record.get("status") or RouteStatus.PENDING
    if RouteStatus(current_status) != RouteStatus.DONE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Route must be in 'done' state to set constraints (current: {current_status})",
        )

    # Load holds from DB
    try:
        rows = await asyncio.wait_for(
            asyncio.to_thread(
                select_records,
                _HOLDS_TABLE,
                {"route_id": route_id_str},
                "*",
                "hold_id.asc",
            ),
            timeout=settings.supabase_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out loading holds.",
        ) from None
    except SupabaseClientError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load holds",
        ) from e

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No holds found for this route. Run /analyze first.",
        )

    # Reconstruct ClassifiedHold list
    classified_holds = _db_rows_to_classified_holds(rows)
    hold_ids_in_db = {h.hold_id for h in classified_holds}

    # Validate start/finish IDs exist
    unknown_starts = set(constraints.start_hold_ids) - hold_ids_in_db
    unknown_finishes = set(constraints.finish_hold_ids) - hold_ids_in_db
    if unknown_starts or unknown_finishes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Hold IDs not found in route: "
                f"start={sorted(unknown_starts)}, finish={sorted(unknown_finishes)}"
            ),
        )

    # Run grading pipeline in thread
    try:
        prediction, explanation = await asyncio.wait_for(
            asyncio.to_thread(
                _run_grading_pipeline,
                route_id=route_id_str,
                classified_holds=classified_holds,
                wall_angle=record.get("wall_angle"),
                start_hold_ids=constraints.start_hold_ids,
                finish_hold_ids=constraints.finish_hold_ids,
                ml_grade_model_path=settings.ml_grade_model_path,
            ),
            timeout=settings.inference_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Grading pipeline timed out.",
        ) from None
    except (RouteGraphError, FeatureExtractionError, GradeEstimationError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(
            "Grading pipeline failed",
            extra={"route_id": route_id_str, "error": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Grading pipeline failed",
        ) from e

    # Build and return response
    estimator_type = "ml" if isinstance(prediction, MLGradeResult) else "heuristic"
    explanation_dict = explanation.model_dump() if explanation is not None else None

    return PredictionResponse(
        route_id=route_id_str,
        estimator_type=estimator_type,
        grade=prediction.grade,
        grade_index=prediction.grade_index,
        confidence=prediction.confidence,
        difficulty_score=prediction.difficulty_score,
        explanation=explanation_dict,
        model_version=None,
    )


def _run_grading_pipeline(
    route_id: str,
    classified_holds: list[ClassifiedHold],
    wall_angle: float | None,
    start_hold_ids: list[int],
    finish_hold_ids: list[int],
    ml_grade_model_path: str,
) -> tuple[HeuristicGradeResult | MLGradeResult, ExplanationResult | None]:
    """Execute the graph → features → grade → explain pipeline (blocking).

    Persists features, prediction, and updated route constraints to the DB.

    Args:
        route_id: UUID string of the route.
        classified_holds: Holds loaded from the DB.
        wall_angle: Optional wall angle in degrees.
        start_hold_ids: Hold IDs for start positions.
        finish_hold_ids: Hold IDs for finish positions.
        ml_grade_model_path: Path to the ML grade model directory (empty =
            heuristic only).

    Returns:
        Tuple of (prediction, explanation).  Explanation may be None if
        generation fails non-fatally.

    Raises:
        RouteGraphError: If graph construction or constraint application fails.
        FeatureExtractionError: If feature extraction fails.
        GradeEstimationError: If grade estimation fails.
    """
    # Build and constrain graph
    # wall_angle defaults to 0.0 (vertical) when not set
    graph = build_route_graph(
        classified_holds, wall_angle if wall_angle is not None else 0.0
    )
    # apply_route_constraints takes a single finish_id; use the first provided
    finish_id = finish_hold_ids[0]
    graph = apply_route_constraints(graph, start_hold_ids, finish_id)

    # Extract features
    features: RouteFeatures = assemble_features(graph)

    # Grade estimation
    heuristic_result = estimate_grade_heuristic(features)
    prediction: HeuristicGradeResult | MLGradeResult = heuristic_result
    if ml_grade_model_path:
        try:
            prediction = estimate_grade_ml(features, ml_grade_model_path)
        except GradeEstimationError:
            logger.warning(
                "ML grading failed, falling back to heuristic",
                extra={"route_id": route_id},
            )
            prediction = heuristic_result

    # Explanation
    explanation: ExplanationResult | None = None
    try:
        explanation = generate_explanation(features, prediction)
    except ExplanationError as e:
        logger.warning(
            "Explanation generation failed",
            extra={"route_id": route_id, "error": str(e)},
        )

    # Persist features (delete old if present, insert new).
    # delete_records returns 0 when no rows match — no exception is raised for
    # the "no prior features" case, so any SupabaseClientError here is a real
    # failure (connection error, permission denied) and must propagate.
    delete_records(_FEATURES_TABLE, {"route_id": route_id})
    insert_record(
        _FEATURES_TABLE,
        {
            "route_id": route_id,
            "feature_vector": features.to_vector(),
        },
    )

    # Persist prediction
    prediction_row = _build_prediction_row(route_id, prediction, explanation)
    insert_record(_PREDICTIONS_TABLE, prediction_row)

    # Persist constraints on route record
    update_record(
        _ROUTES_TABLE,
        route_id,
        {
            "start_hold_ids": start_hold_ids,
            "finish_hold_ids": finish_hold_ids,
        },
    )

    return prediction, explanation


# ---------------------------------------------------------------------------
# GET /routes/{route_id}/prediction
# ---------------------------------------------------------------------------


@router.get(
    "/routes/{route_id}/prediction",
    response_model=PredictionResponse | None,
    responses={
        404: {"model": ErrorResponse, "description": "Route not found"},
        500: {"model": ErrorResponse, "description": "Database error"},
    },
)
async def get_prediction(
    route_id: Annotated[
        uuid.UUID,
        Path(description="UUID of the route"),
    ],
) -> PredictionResponse | None:
    """Retrieve the latest prediction for a route.

    Args:
        route_id: UUID of the route.

    Returns:
        :class:`PredictionResponse` for the latest prediction, or ``None``
        if no predictions exist yet.

    Raises:
        HTTPException: 404 if route not found, 500 on database error.
    """
    route_id_str = str(route_id)

    # Verify route exists
    await _load_route_or_404(route_id_str)

    settings = get_settings()
    try:
        rows = await asyncio.wait_for(
            asyncio.to_thread(
                select_records,
                _PREDICTIONS_TABLE,
                {"route_id": route_id_str},
                "*",
                "predicted_at.desc",
                1,  # limit
            ),
            timeout=settings.supabase_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out. Please try again.",
        ) from None
    except SupabaseClientError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve prediction",
        ) from e

    if not rows:
        return None

    row = rows[0]
    return PredictionResponse(
        route_id=route_id_str,
        estimator_type=str(row["estimator_type"]),
        grade=str(row["grade"]),
        grade_index=int(row["grade_index"]),
        confidence=float(row["confidence"]),
        difficulty_score=float(row["difficulty_score"]),
        explanation=row.get("explanation"),
        model_version=row.get("model_version"),
    )


# ---------------------------------------------------------------------------
# POST /routes/{route_id}/feedback
# ---------------------------------------------------------------------------


@router.post(
    "/routes/{route_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ErrorResponse, "description": "Route not found"},
        422: {"model": ErrorResponse, "description": "Invalid feedback data"},
        500: {"model": ErrorResponse, "description": "Database error"},
    },
)
async def submit_feedback(
    route_id: Annotated[
        uuid.UUID,
        Path(description="UUID of the route"),
    ],
    feedback: FeedbackCreate,
) -> FeedbackResponse:
    """Submit user feedback for a route's grade prediction.

    Stores the feedback anonymously (no authentication required).  At least
    one of ``user_grade``, ``is_accurate``, or ``comments`` must be provided.

    Args:
        route_id: UUID of the route.
        feedback: Feedback data.

    Returns:
        :class:`FeedbackResponse` with the created record ID.

    Raises:
        HTTPException: 404 if route not found, 500 on database error.
    """
    route_id_str = str(route_id)

    # Verify route exists
    await _load_route_or_404(route_id_str)

    row: dict[str, Any] = {
        "route_id": route_id_str,
        "user_grade": feedback.user_grade,
        "is_accurate": feedback.is_accurate,
        "comments": feedback.comments,
    }

    settings = get_settings()
    try:
        record = await asyncio.wait_for(
            asyncio.to_thread(insert_record, _FEEDBACK_TABLE, row),
            timeout=settings.supabase_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out. Please try again.",
        ) from None
    except SupabaseClientError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save feedback",
        ) from e

    return FeedbackResponse(
        id=str(record["id"]),
        route_id=route_id_str,
        created_at=str(record["created_at"]),
    )
