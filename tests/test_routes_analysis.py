"""Tests for analysis, constraints, prediction, and feedback endpoints (PR-10.3–10.5).

All inference calls are patched with MagicMock so no ML models are required.
All Supabase calls are patched at the function level.
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app import create_app
from src.config import get_settings_override
from src.database.supabase_client import SupabaseClientError
from src.routes.analysis import (
    _build_prediction_row,
    _db_row_to_hold_response,
    _db_rows_to_classified_holds,
    _load_route_or_404,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ROUTE_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_TEST_SETTINGS = {
    "testing": True,
    "debug": True,
    "rate_limit_upload": 1000,
    "detection_model_path": "/models/detect/best.pt",
    "classification_model_path": "/models/classify/best.pt",
    "ml_grade_model_path": "",
}

_MOCK_SETTINGS = get_settings_override(
    {
        "testing": True,
        "supabase_timeout_seconds": 10,
        "inference_timeout_seconds": 30,
        "detection_model_path": "/models/detect/best.pt",
        "classification_model_path": "/models/classify/best.pt",
        "ml_grade_model_path": "",
    }
)


@pytest.fixture
def app() -> FastAPI:
    """Create test app with model paths configured."""
    return create_app(_TEST_SETTINGS)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client."""
    with TestClient(app) as c:
        return c


def _pending_route(route_id: str = _ROUTE_ID) -> dict[str, Any]:
    return {
        "id": route_id,
        "image_url": "https://example.supabase.co/route.jpg",
        "wall_angle": 15.0,
        "status": "pending",
        "created_at": "2026-01-01T12:00:00Z",
        "updated_at": "2026-01-01T12:00:00Z",
        "start_hold_ids": None,
        "finish_hold_ids": None,
    }


def _done_route(route_id: str = _ROUTE_ID) -> dict[str, Any]:
    route = _pending_route(route_id)
    route["status"] = "done"
    return route


def _make_hold_rows(n: int = 3) -> list[dict[str, Any]]:
    """Return n fake holds table rows."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"hold-uuid-{i}",
                "route_id": _ROUTE_ID,
                "hold_id": i,
                "x_center": 0.2 + i * 0.2,
                "y_center": 0.5,
                "width": 0.05,
                "height": 0.05,
                "detection_class": "Jug",
                "detection_confidence": 0.9,
                "hold_type": "jug",
                "type_confidence": 0.85,
                "prob_jug": 0.85,
                "prob_crimp": 0.03,
                "prob_sloper": 0.02,
                "prob_pinch": 0.02,
                "prob_pocket": 0.02,
                "prob_edges": 0.02,
                "prob_foothold": 0.02,
                "prob_unknown": 0.02,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# TestAnalyze
# ---------------------------------------------------------------------------


class TestAnalyze:
    """Tests for POST /api/v1/routes/{route_id}/analyze."""

    def test_analyze_success(self, client: TestClient) -> None:
        """Successful analysis returns 200 with hold_count and status=done."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_pending_route(),
            ),
            patch("src.routes.analysis.update_record"),
            patch("src.routes.analysis.insert_records_bulk"),
            patch("src.routes.analysis._run_analysis_pipeline", return_value=5),
        ):
            resp = client.post(f"/api/v1/routes/{_ROUTE_ID}/analyze")

        assert resp.status_code == 200
        body = resp.json()
        assert body["route_id"] == _ROUTE_ID
        assert body["hold_count"] == 5
        assert body["status"] == "done"

    def test_analyze_route_not_found_returns_404(self, client: TestClient) -> None:
        """Returns 404 when route does not exist."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=None,
            ),
        ):
            resp = client.post(f"/api/v1/routes/{_ROUTE_ID}/analyze")

        assert resp.status_code == 404

    def test_analyze_not_pending_returns_409(self, client: TestClient) -> None:
        """Returns 409 when route is not in pending state."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
        ):
            resp = client.post(f"/api/v1/routes/{_ROUTE_ID}/analyze")

        assert resp.status_code == 409
        assert "not in pending state" in resp.json()["detail"]

    def test_analyze_missing_model_paths_returns_503(self, client: TestClient) -> None:
        """Returns 503 when model paths are not configured."""
        no_models = get_settings_override(
            {
                "testing": True,
                "detection_model_path": "",
                "classification_model_path": "",
            }
        )
        with patch("src.routes.analysis.get_settings", return_value=no_models):
            resp = client.post(f"/api/v1/routes/{_ROUTE_ID}/analyze")

        assert resp.status_code == 503

    def test_analyze_pipeline_failure_marks_failed_and_returns_500(
        self, client: TestClient
    ) -> None:
        """Pipeline error marks route as failed and returns 500."""
        update_mock = MagicMock()
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_pending_route(),
            ),
            patch("src.routes.analysis.update_record", update_mock),
            patch(
                "src.routes.analysis._run_analysis_pipeline",
                side_effect=RuntimeError("detect failed"),
            ),
        ):
            resp = client.post(f"/api/v1/routes/{_ROUTE_ID}/analyze")

        assert resp.status_code == 500
        # Route should have been marked as failed
        calls = [str(c) for c in update_mock.call_args_list]
        assert any("failed" in c for c in calls)

    def test_analyze_invalid_uuid_returns_422(self, client: TestClient) -> None:
        """Invalid UUID returns 422."""
        resp = client.post("/api/v1/routes/not-a-uuid/analyze")
        assert resp.status_code == 422

    def test_analyze_db_error_on_status_update_returns_500(
        self, client: TestClient
    ) -> None:
        """DB error when updating to 'processing' returns 500."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_pending_route(),
            ),
            patch(
                "src.routes.analysis.update_record",
                side_effect=SupabaseClientError("db error"),
            ),
        ):
            resp = client.post(f"/api/v1/routes/{_ROUTE_ID}/analyze")

        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# TestListHolds
# ---------------------------------------------------------------------------


class TestListHolds:
    """Tests for GET /api/v1/routes/{route_id}/holds."""

    def test_list_holds_success(self, client: TestClient) -> None:
        """Returns holds list with correct count."""
        hold_rows = _make_hold_rows(3)
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch(
                "src.routes.analysis.select_records",
                return_value=hold_rows,
            ),
        ):
            resp = client.get(f"/api/v1/routes/{_ROUTE_ID}/holds")

        assert resp.status_code == 200
        body = resp.json()
        assert body["route_id"] == _ROUTE_ID
        assert body["count"] == 3
        assert len(body["holds"]) == 3

    def test_list_holds_empty_returns_zero_count(self, client: TestClient) -> None:
        """Returns empty list and count=0 when no holds."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch("src.routes.analysis.select_records", return_value=[]),
        ):
            resp = client.get(f"/api/v1/routes/{_ROUTE_ID}/holds")

        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 0
        assert body["holds"] == []

    def test_list_holds_route_not_found_returns_404(self, client: TestClient) -> None:
        """Returns 404 when route does not exist."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=None,
            ),
        ):
            resp = client.get(f"/api/v1/routes/{_ROUTE_ID}/holds")

        assert resp.status_code == 404

    def test_list_holds_db_error_returns_500(self, client: TestClient) -> None:
        """Returns 500 on DB error for hold selection."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch(
                "src.routes.analysis.select_records",
                side_effect=SupabaseClientError("db error"),
            ),
        ):
            resp = client.get(f"/api/v1/routes/{_ROUTE_ID}/holds")

        assert resp.status_code == 500

    def test_list_holds_invalid_uuid_returns_422(self, client: TestClient) -> None:
        """Invalid UUID returns 422."""
        resp = client.get("/api/v1/routes/bad-uuid/holds")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# TestConstraints
# ---------------------------------------------------------------------------


class TestConstraints:
    """Tests for PUT /api/v1/routes/{route_id}/constraints."""

    def _make_heuristic_result(self) -> MagicMock:
        result = MagicMock()
        result.grade = "V3"
        result.grade_index = 3
        result.confidence = 0.75
        result.difficulty_score = 0.3
        return result

    def test_constraints_success(self, client: TestClient) -> None:
        """Returns PredictionResponse on success."""
        hold_rows = _make_hold_rows(4)
        heuristic_result = self._make_heuristic_result()
        explanation = MagicMock()
        explanation.model_dump.return_value = {"summary": "test"}

        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch("src.routes.analysis.select_records", return_value=hold_rows),
            patch(
                "src.routes.analysis._run_grading_pipeline",
                return_value=(heuristic_result, explanation),
            ),
        ):
            resp = client.put(
                f"/api/v1/routes/{_ROUTE_ID}/constraints",
                json={"start_hold_ids": [0], "finish_hold_ids": [3]},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["route_id"] == _ROUTE_ID
        assert body["grade"] == "V3"
        assert body["estimator_type"] == "heuristic"

    def test_constraints_route_not_found_returns_404(self, client: TestClient) -> None:
        """Returns 404 when route does not exist."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=None,
            ),
        ):
            resp = client.put(
                f"/api/v1/routes/{_ROUTE_ID}/constraints",
                json={"start_hold_ids": [0], "finish_hold_ids": [3]},
            )
        assert resp.status_code == 404

    def test_constraints_not_done_returns_409(self, client: TestClient) -> None:
        """Returns 409 when route is not in done state."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_pending_route(),
            ),
        ):
            resp = client.put(
                f"/api/v1/routes/{_ROUTE_ID}/constraints",
                json={"start_hold_ids": [0], "finish_hold_ids": [3]},
            )
        assert resp.status_code == 409

    def test_constraints_empty_start_ids_returns_422(self, client: TestClient) -> None:
        """Returns 422 when start_hold_ids is empty."""
        resp = client.put(
            f"/api/v1/routes/{_ROUTE_ID}/constraints",
            json={"start_hold_ids": [], "finish_hold_ids": [3]},
        )
        assert resp.status_code == 422

    def test_constraints_empty_finish_ids_returns_422(self, client: TestClient) -> None:
        """Returns 422 when finish_hold_ids is empty."""
        resp = client.put(
            f"/api/v1/routes/{_ROUTE_ID}/constraints",
            json={"start_hold_ids": [0], "finish_hold_ids": []},
        )
        assert resp.status_code == 422

    def test_constraints_negative_hold_ids_returns_422(
        self, client: TestClient
    ) -> None:
        """Returns 422 when hold IDs are negative."""
        resp = client.put(
            f"/api/v1/routes/{_ROUTE_ID}/constraints",
            json={"start_hold_ids": [-1], "finish_hold_ids": [3]},
        )
        assert resp.status_code == 422

    def test_constraints_hold_id_not_in_db_returns_422(
        self, client: TestClient
    ) -> None:
        """Returns 422 when a hold ID does not exist in the DB."""
        hold_rows = _make_hold_rows(3)  # IDs 0, 1, 2
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch("src.routes.analysis.select_records", return_value=hold_rows),
        ):
            resp = client.put(
                f"/api/v1/routes/{_ROUTE_ID}/constraints",
                json={
                    "start_hold_ids": [0],
                    "finish_hold_ids": [99],
                },  # 99 doesn't exist
            )
        assert resp.status_code == 422

    def test_constraints_no_holds_found_returns_422(self, client: TestClient) -> None:
        """Returns 422 when no holds found (run analyze first)."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch("src.routes.analysis.select_records", return_value=[]),
        ):
            resp = client.put(
                f"/api/v1/routes/{_ROUTE_ID}/constraints",
                json={"start_hold_ids": [0], "finish_hold_ids": [1]},
            )
        assert resp.status_code == 422
        assert "No holds found" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# TestGetPrediction
# ---------------------------------------------------------------------------


class TestGetPrediction:
    """Tests for GET /api/v1/routes/{route_id}/prediction."""

    def test_get_prediction_success(self, client: TestClient) -> None:
        """Returns latest prediction when available."""
        pred_row: dict[str, Any] = {
            "id": "pred-uuid",
            "route_id": _ROUTE_ID,
            "estimator_type": "heuristic",
            "grade": "V4",
            "grade_index": 4,
            "confidence": 0.8,
            "difficulty_score": 0.4,
            "explanation": {"summary": "test"},
            "model_version": None,
            "predicted_at": "2026-01-01T12:00:00Z",
        }
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch("src.routes.analysis.select_records", return_value=[pred_row]),
        ):
            resp = client.get(f"/api/v1/routes/{_ROUTE_ID}/prediction")

        assert resp.status_code == 200
        body = resp.json()
        assert body["grade"] == "V4"
        assert body["estimator_type"] == "heuristic"

    def test_get_prediction_no_predictions_returns_none(
        self, client: TestClient
    ) -> None:
        """Returns null when no predictions exist yet."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch("src.routes.analysis.select_records", return_value=[]),
        ):
            resp = client.get(f"/api/v1/routes/{_ROUTE_ID}/prediction")

        assert resp.status_code == 200
        assert resp.json() is None

    def test_get_prediction_route_not_found_returns_404(
        self, client: TestClient
    ) -> None:
        """Returns 404 when route does not exist."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=None,
            ),
        ):
            resp = client.get(f"/api/v1/routes/{_ROUTE_ID}/prediction")
        assert resp.status_code == 404

    def test_get_prediction_db_error_returns_500(self, client: TestClient) -> None:
        """Returns 500 on DB error for prediction selection."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch(
                "src.routes.analysis.select_records",
                side_effect=SupabaseClientError("db error"),
            ),
        ):
            resp = client.get(f"/api/v1/routes/{_ROUTE_ID}/prediction")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# TestFeedback
# ---------------------------------------------------------------------------


class TestFeedback:
    """Tests for POST /api/v1/routes/{route_id}/feedback."""

    def test_submit_feedback_with_grade(self, client: TestClient) -> None:
        """Returns 201 with created record ID for valid feedback with grade."""
        feedback_row: dict[str, Any] = {
            "id": "feedback-uuid",
            "route_id": _ROUTE_ID,
            "created_at": "2026-01-01T12:00:00Z",
        }
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch("src.routes.analysis.insert_record", return_value=feedback_row),
        ):
            resp = client.post(
                f"/api/v1/routes/{_ROUTE_ID}/feedback",
                json={"user_grade": "V5"},
            )

        assert resp.status_code == 201
        body = resp.json()
        assert body["id"] == "feedback-uuid"
        assert body["route_id"] == _ROUTE_ID

    def test_submit_feedback_with_is_accurate(self, client: TestClient) -> None:
        """Returns 201 for feedback with is_accurate field."""
        feedback_row: dict[str, Any] = {
            "id": "feedback-uuid",
            "route_id": _ROUTE_ID,
            "created_at": "2026-01-01T12:00:00Z",
        }
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch("src.routes.analysis.insert_record", return_value=feedback_row),
        ):
            resp = client.post(
                f"/api/v1/routes/{_ROUTE_ID}/feedback",
                json={"is_accurate": True},
            )

        assert resp.status_code == 201

    def test_submit_feedback_with_comments(self, client: TestClient) -> None:
        """Returns 201 for feedback with comments only."""
        feedback_row: dict[str, Any] = {
            "id": "feedback-uuid",
            "route_id": _ROUTE_ID,
            "created_at": "2026-01-01T12:00:00Z",
        }
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch("src.routes.analysis.insert_record", return_value=feedback_row),
        ):
            resp = client.post(
                f"/api/v1/routes/{_ROUTE_ID}/feedback",
                json={"comments": "Felt hard for V3"},
            )

        assert resp.status_code == 201

    def test_submit_feedback_no_fields_returns_422(self, client: TestClient) -> None:
        """Returns 422 when all fields are None."""
        resp = client.post(
            f"/api/v1/routes/{_ROUTE_ID}/feedback",
            json={},
        )
        assert resp.status_code == 422

    def test_submit_feedback_invalid_grade_format_returns_422(
        self, client: TestClient
    ) -> None:
        """Returns 422 when user_grade format is invalid."""
        resp = client.post(
            f"/api/v1/routes/{_ROUTE_ID}/feedback",
            json={"user_grade": "5.10"},
        )
        assert resp.status_code == 422

    def test_submit_feedback_route_not_found_returns_404(
        self, client: TestClient
    ) -> None:
        """Returns 404 when route does not exist."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=None,
            ),
        ):
            resp = client.post(
                f"/api/v1/routes/{_ROUTE_ID}/feedback",
                json={"user_grade": "V5"},
            )
        assert resp.status_code == 404

    def test_submit_feedback_db_error_returns_500(self, client: TestClient) -> None:
        """Returns 500 on DB error during feedback insertion."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch(
                "src.routes.analysis.insert_record",
                side_effect=SupabaseClientError("db error"),
            ),
        ):
            resp = client.post(
                f"/api/v1/routes/{_ROUTE_ID}/feedback",
                json={"user_grade": "V5"},
            )
        assert resp.status_code == 500

    def test_submit_feedback_all_fields_valid(self, client: TestClient) -> None:
        """Returns 201 with all feedback fields provided."""
        feedback_row: dict[str, Any] = {
            "id": "feedback-uuid",
            "route_id": _ROUTE_ID,
            "created_at": "2026-01-01T12:00:00Z",
        }
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch("src.routes.analysis.insert_record", return_value=feedback_row),
        ):
            resp = client.post(
                f"/api/v1/routes/{_ROUTE_ID}/feedback",
                json={
                    "user_grade": "V7",
                    "is_accurate": False,
                    "comments": "Way harder than V5",
                },
            )
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# TestHelperFunctions -- unit tests for private helpers
# ---------------------------------------------------------------------------


class TestDbRowToHoldResponse:
    """Unit tests for _db_row_to_hold_response."""

    def _make_row(self) -> dict[str, Any]:
        return {
            "hold_id": 2,
            "x_center": 0.5,
            "y_center": 0.4,
            "width": 0.06,
            "height": 0.07,
            "detection_class": "Crimp",
            "detection_confidence": 0.92,
            "hold_type": "crimp",
            "type_confidence": 0.80,
            "prob_jug": 0.05,
            "prob_crimp": 0.80,
            "prob_sloper": 0.03,
            "prob_pinch": 0.03,
            "prob_pocket": 0.03,
            "prob_edges": 0.03,
            "prob_foothold": 0.03,
            "prob_unknown": 0.03,
        }

    def test_maps_all_fields_correctly(self) -> None:
        row = self._make_row()
        result = _db_row_to_hold_response(row)
        assert result.hold_id == 2
        assert result.x_center == 0.5
        assert result.hold_type == "crimp"
        assert result.type_confidence == 0.80
        assert result.type_probabilities["crimp"] == 0.80

    def test_probabilities_dict_has_eight_keys(self) -> None:
        result = _db_row_to_hold_response(self._make_row())
        assert set(result.type_probabilities.keys()) == {
            "jug",
            "crimp",
            "sloper",
            "pinch",
            "pocket",
            "edges",
            "foothold",
            "unknown",
        }


class TestDbRowsToClassifiedHolds:
    """Unit tests for _db_rows_to_classified_holds."""

    def _make_row(self, hold_id: int = 0, scale: float = 1.0) -> dict[str, Any]:
        return {
            "hold_id": hold_id,
            "x_center": 0.3,
            "y_center": 0.5,
            "width": 0.05,
            "height": 0.05,
            "detection_class": "Jug",
            "detection_confidence": 0.9,
            "hold_type": "jug",
            "type_confidence": 0.85,
            "prob_jug": 0.85 * scale,
            "prob_crimp": 0.02 * scale,
            "prob_sloper": 0.02 * scale,
            "prob_pinch": 0.02 * scale,
            "prob_pocket": 0.02 * scale,
            "prob_edges": 0.02 * scale,
            "prob_foothold": 0.02 * scale,
            "prob_unknown": 0.03 * scale,
        }

    def test_returns_classified_holds_list(self) -> None:
        rows = [self._make_row(0), self._make_row(1)]
        holds = _db_rows_to_classified_holds(rows)
        assert len(holds) == 2
        assert holds[0].hold_id == 0
        assert holds[1].hold_id == 1

    def test_prob_normalization_applied_when_off_by_more_than_threshold(self) -> None:
        """Probs summing to 1.1 (>0.01 off) are renormalised to 1.0."""
        row = self._make_row(0, scale=1.1)
        holds = _db_rows_to_classified_holds([row])
        total = sum(holds[0].type_probabilities.values())
        assert abs(total - 1.0) < 1e-9

    def test_prob_normalization_skipped_when_within_threshold(self) -> None:
        """Probs summing to 1.0 exactly are not changed."""
        row = self._make_row(0, scale=1.0)
        original_jug = row["prob_jug"]
        holds = _db_rows_to_classified_holds([row])
        assert holds[0].type_probabilities["jug"] == pytest.approx(original_jug)

    def test_empty_rows_returns_empty_list(self) -> None:
        assert _db_rows_to_classified_holds([]) == []


class TestBuildPredictionRow:
    """Unit tests for _build_prediction_row."""

    def test_heuristic_estimator_type(self) -> None:
        from src.grading import HeuristicGradeResult

        result = HeuristicGradeResult(
            grade="V3", grade_index=3, confidence=0.75, difficulty_score=0.3
        )
        row = _build_prediction_row(_ROUTE_ID, result, None)
        assert row["estimator_type"] == "heuristic"
        assert row["model_version"] is None
        assert row["explanation"] is None

    def test_ml_estimator_type(self) -> None:
        from src.grading import MLGradeResult

        probs = {f"V{i}": 0.0 for i in range(18)}
        probs["V3"] = 1.0
        result = MLGradeResult(
            grade="V3",
            grade_index=3,
            confidence=0.9,
            difficulty_score=0.3,
            grade_probabilities=probs,
        )
        row = _build_prediction_row(_ROUTE_ID, result, None)
        assert row["estimator_type"] == "ml"

    def test_explanation_serialised_when_provided(self) -> None:
        from src.grading import HeuristicGradeResult

        result = HeuristicGradeResult(
            grade="V3", grade_index=3, confidence=0.75, difficulty_score=0.3
        )
        explanation = MagicMock()
        explanation.model_dump.return_value = {"summary": "test summary"}
        row = _build_prediction_row(_ROUTE_ID, result, explanation)
        assert row["explanation"] == {"summary": "test summary"}

    def test_route_id_is_included(self) -> None:
        from src.grading import HeuristicGradeResult

        result = HeuristicGradeResult(
            grade="V0", grade_index=0, confidence=0.5, difficulty_score=0.0
        )
        row = _build_prediction_row(_ROUTE_ID, result, None)
        assert row["route_id"] == _ROUTE_ID


class TestLoadRouteOr404:
    """Unit tests for _load_route_or_404 async helper."""

    def test_returns_record_when_found(self) -> None:
        record = _done_route()
        with (
            patch("src.routes.analysis.select_record_by_id", return_value=record),
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
        ):
            result = asyncio.run(_load_route_or_404(_ROUTE_ID))
        assert result["id"] == _ROUTE_ID

    def test_raises_404_when_not_found(self) -> None:
        from fastapi import HTTPException

        with (
            patch("src.routes.analysis.select_record_by_id", return_value=None),
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(_load_route_or_404(_ROUTE_ID))
        assert exc_info.value.status_code == 404

    def test_raises_500_on_db_error(self) -> None:
        from fastapi import HTTPException

        with (
            patch(
                "src.routes.analysis.select_record_by_id",
                side_effect=SupabaseClientError("db error"),
            ),
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(_load_route_or_404(_ROUTE_ID))
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# TestAnalyzeAdditional
# ---------------------------------------------------------------------------


class TestAnalyzeAdditional:
    """Additional coverage for analyze endpoint."""

    def test_analyze_timeout_marks_failed_and_returns_504(
        self, client: TestClient
    ) -> None:
        """TimeoutError during pipeline sets status=failed and returns 504."""
        update_mock = MagicMock()
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_pending_route(),
            ),
            patch("src.routes.analysis.update_record", update_mock),
            patch(
                "src.routes.analysis._run_analysis_pipeline",
                side_effect=asyncio.TimeoutError(),
            ),
        ):
            resp = client.post(f"/api/v1/routes/{_ROUTE_ID}/analyze")
        assert resp.status_code == 504
        calls = [str(c) for c in update_mock.call_args_list]
        assert any("failed" in c for c in calls)

    def test_analyze_pipeline_failure_inner_db_error_suppressed(
        self, client: TestClient
    ) -> None:
        """SupabaseClientError while marking failed is suppressed; 500 returned."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_pending_route(),
            ),
            patch(
                "src.routes.analysis.update_record",
                side_effect=[None, SupabaseClientError("db fail")],
            ),
            patch(
                "src.routes.analysis._run_analysis_pipeline",
                side_effect=RuntimeError("boom"),
            ),
        ):
            resp = client.post(f"/api/v1/routes/{_ROUTE_ID}/analyze")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# TestListHoldsAdditional
# ---------------------------------------------------------------------------


class TestListHoldsAdditional:
    """Additional coverage for list holds endpoint."""

    def test_list_holds_timeout_returns_504(self, client: TestClient) -> None:
        """TimeoutError returns 504."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch(
                "src.routes.analysis.select_records",
                side_effect=asyncio.TimeoutError(),
            ),
        ):
            resp = client.get(f"/api/v1/routes/{_ROUTE_ID}/holds")
        assert resp.status_code == 504


# ---------------------------------------------------------------------------
# TestConstraintsAdditional
# ---------------------------------------------------------------------------


class TestConstraintsAdditional:
    """Additional coverage for constraints endpoint."""

    def test_constraints_timeout_loading_holds_returns_504(
        self, client: TestClient
    ) -> None:
        """TimeoutError while loading holds returns 504."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch(
                "src.routes.analysis.select_records",
                side_effect=asyncio.TimeoutError(),
            ),
        ):
            resp = client.put(
                f"/api/v1/routes/{_ROUTE_ID}/constraints",
                json={"start_hold_ids": [0], "finish_hold_ids": [2]},
            )
        assert resp.status_code == 504

    def test_constraints_db_error_loading_holds_returns_500(
        self, client: TestClient
    ) -> None:
        """SupabaseClientError while loading holds returns 500."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch(
                "src.routes.analysis.select_records",
                side_effect=SupabaseClientError("db fail"),
            ),
        ):
            resp = client.put(
                f"/api/v1/routes/{_ROUTE_ID}/constraints",
                json={"start_hold_ids": [0], "finish_hold_ids": [2]},
            )
        assert resp.status_code == 500

    def test_constraints_grading_timeout_returns_504(self, client: TestClient) -> None:
        """TimeoutError from grading pipeline returns 504."""
        hold_rows = _make_hold_rows(3)
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch("src.routes.analysis.select_records", return_value=hold_rows),
            patch(
                "src.routes.analysis._run_grading_pipeline",
                side_effect=asyncio.TimeoutError(),
            ),
        ):
            resp = client.put(
                f"/api/v1/routes/{_ROUTE_ID}/constraints",
                json={"start_hold_ids": [0], "finish_hold_ids": [2]},
            )
        assert resp.status_code == 504

    def test_constraints_graph_error_returns_422(self, client: TestClient) -> None:
        """RouteGraphError from pipeline returns 422."""
        from src.graph import RouteGraphError

        hold_rows = _make_hold_rows(3)
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch("src.routes.analysis.select_records", return_value=hold_rows),
            patch(
                "src.routes.analysis._run_grading_pipeline",
                side_effect=RouteGraphError("graph failed"),
            ),
        ):
            resp = client.put(
                f"/api/v1/routes/{_ROUTE_ID}/constraints",
                json={"start_hold_ids": [0], "finish_hold_ids": [2]},
            )
        assert resp.status_code == 422

    def test_constraints_pipeline_generic_error_returns_500(
        self, client: TestClient
    ) -> None:
        """Unexpected error from grading pipeline returns 500."""
        hold_rows = _make_hold_rows(3)
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch("src.routes.analysis.select_records", return_value=hold_rows),
            patch(
                "src.routes.analysis._run_grading_pipeline",
                side_effect=RuntimeError("unexpected"),
            ),
        ):
            resp = client.put(
                f"/api/v1/routes/{_ROUTE_ID}/constraints",
                json={"start_hold_ids": [0], "finish_hold_ids": [2]},
            )
        assert resp.status_code == 500

    def test_constraints_finish_hold_ids_max_length_one(
        self, client: TestClient
    ) -> None:
        """finish_hold_ids with more than 1 element returns 422."""
        resp = client.put(
            f"/api/v1/routes/{_ROUTE_ID}/constraints",
            json={"start_hold_ids": [0], "finish_hold_ids": [2, 3]},
        )
        assert resp.status_code == 422

    def test_constraints_ml_estimator_type_in_response(
        self, client: TestClient
    ) -> None:
        """When grading returns MLGradeResult, estimator_type is ml."""
        from src.grading import MLGradeResult

        probs = {f"V{i}": 0.0 for i in range(18)}
        probs["V4"] = 1.0
        ml_result = MLGradeResult(
            grade="V4",
            grade_index=4,
            confidence=0.9,
            difficulty_score=0.4,
            grade_probabilities=probs,
        )
        explanation = MagicMock()
        explanation.model_dump.return_value = {"summary": "ml summary"}
        hold_rows = _make_hold_rows(4)
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch("src.routes.analysis.select_records", return_value=hold_rows),
            patch(
                "src.routes.analysis._run_grading_pipeline",
                return_value=(ml_result, explanation),
            ),
        ):
            resp = client.put(
                f"/api/v1/routes/{_ROUTE_ID}/constraints",
                json={"start_hold_ids": [0], "finish_hold_ids": [3]},
            )
        assert resp.status_code == 200
        assert resp.json()["estimator_type"] == "ml"

    def test_constraints_null_explanation_in_response(self, client: TestClient) -> None:
        """When explanation is None, response explanation field is None."""
        from src.grading import HeuristicGradeResult

        result = HeuristicGradeResult(
            grade="V2", grade_index=2, confidence=0.6, difficulty_score=0.2
        )
        hold_rows = _make_hold_rows(3)
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch("src.routes.analysis.select_records", return_value=hold_rows),
            patch(
                "src.routes.analysis._run_grading_pipeline",
                return_value=(result, None),
            ),
        ):
            resp = client.put(
                f"/api/v1/routes/{_ROUTE_ID}/constraints",
                json={"start_hold_ids": [0], "finish_hold_ids": [2]},
            )
        assert resp.status_code == 200
        assert resp.json()["explanation"] is None


# ---------------------------------------------------------------------------
# TestGetPredictionAdditional
# ---------------------------------------------------------------------------


class TestGetPredictionAdditional:
    """Additional coverage for get_prediction endpoint."""

    def test_get_prediction_timeout_returns_504(self, client: TestClient) -> None:
        """TimeoutError returns 504."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch(
                "src.routes.analysis.select_records",
                side_effect=asyncio.TimeoutError(),
            ),
        ):
            resp = client.get(f"/api/v1/routes/{_ROUTE_ID}/prediction")
        assert resp.status_code == 504


# ---------------------------------------------------------------------------
# TestFeedbackAdditional
# ---------------------------------------------------------------------------


class TestFeedbackAdditional:
    """Additional coverage for submit_feedback endpoint."""

    def test_submit_feedback_timeout_returns_504(self, client: TestClient) -> None:
        """TimeoutError returns 504."""
        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.select_record_by_id",
                return_value=_done_route(),
            ),
            patch(
                "src.routes.analysis.insert_record",
                side_effect=asyncio.TimeoutError(),
            ),
        ):
            resp = client.post(
                f"/api/v1/routes/{_ROUTE_ID}/feedback",
                json={"user_grade": "V5"},
            )
        assert resp.status_code == 504


# ---------------------------------------------------------------------------
# TestRunGradingPipeline -- direct unit tests for _run_grading_pipeline
# ---------------------------------------------------------------------------


class TestRunGradingPipeline:
    """Direct unit tests for the synchronous _run_grading_pipeline helper."""

    def _make_holds(self, n: int = 3) -> list:
        from src.routes.analysis import _db_rows_to_classified_holds

        return _db_rows_to_classified_holds(_make_hold_rows(n))

    def _mock_heuristic(self) -> MagicMock:
        m = MagicMock()
        m.grade = "V3"
        m.grade_index = 3
        m.confidence = 0.75
        m.difficulty_score = 0.3
        return m

    def test_heuristic_only_pipeline_succeeds(self) -> None:
        """Full heuristic pipeline runs and returns (prediction, explanation)."""
        from src.routes.analysis import _run_grading_pipeline

        heuristic = self._mock_heuristic()
        explanation = MagicMock()
        holds = self._make_holds(3)

        with (
            patch("src.routes.analysis.build_route_graph") as mock_graph,
            patch("src.routes.analysis.apply_route_constraints") as mock_constrain,
            patch("src.routes.analysis.assemble_features") as mock_features,
            patch(
                "src.routes.analysis.estimate_grade_heuristic", return_value=heuristic
            ),
            patch("src.routes.analysis.generate_explanation", return_value=explanation),
            patch("src.routes.analysis.delete_records"),
            patch("src.routes.analysis.insert_record"),
            patch("src.routes.analysis.update_record"),
        ):
            mock_graph.return_value = MagicMock()
            mock_constrain.return_value = MagicMock()
            mock_features.return_value = MagicMock()

            prediction, expl = _run_grading_pipeline(
                route_id=_ROUTE_ID,
                classified_holds=holds,
                wall_angle=15.0,
                start_hold_ids=[0],
                finish_hold_ids=[2],
                ml_grade_model_path="",
            )

        assert prediction is heuristic
        assert expl is explanation

    def test_ml_path_used_when_model_path_set(self) -> None:
        """ML estimator is used when ml_grade_model_path is non-empty."""
        from src.routes.analysis import _run_grading_pipeline
        from src.grading import MLGradeResult

        heuristic = self._mock_heuristic()
        probs = {f"V{i}": 0.0 for i in range(18)}
        probs["V5"] = 1.0
        ml_result = MLGradeResult(
            grade="V5",
            grade_index=5,
            confidence=0.9,
            difficulty_score=0.5,
            grade_probabilities=probs,
        )
        holds = self._make_holds(3)

        with (
            patch("src.routes.analysis.build_route_graph") as mock_graph,
            patch("src.routes.analysis.apply_route_constraints") as mock_constrain,
            patch("src.routes.analysis.assemble_features") as mock_features,
            patch(
                "src.routes.analysis.estimate_grade_heuristic", return_value=heuristic
            ),
            patch("src.routes.analysis.estimate_grade_ml", return_value=ml_result),
            patch("src.routes.analysis.generate_explanation", return_value=MagicMock()),
            patch("src.routes.analysis.delete_records"),
            patch("src.routes.analysis.insert_record"),
            patch("src.routes.analysis.update_record"),
        ):
            mock_graph.return_value = MagicMock()
            mock_constrain.return_value = MagicMock()
            mock_features.return_value = MagicMock()

            prediction, _ = _run_grading_pipeline(
                route_id=_ROUTE_ID,
                classified_holds=holds,
                wall_angle=None,
                start_hold_ids=[0],
                finish_hold_ids=[2],
                ml_grade_model_path="/models/grade/v1",
            )

        assert prediction is ml_result

    def test_ml_failure_falls_back_to_heuristic(self) -> None:
        """When ML grading raises GradeEstimationError, heuristic result is used."""
        from src.routes.analysis import _run_grading_pipeline
        from src.grading import GradeEstimationError

        heuristic = self._mock_heuristic()
        holds = self._make_holds(3)

        with (
            patch("src.routes.analysis.build_route_graph") as mock_graph,
            patch("src.routes.analysis.apply_route_constraints") as mock_constrain,
            patch("src.routes.analysis.assemble_features") as mock_features,
            patch(
                "src.routes.analysis.estimate_grade_heuristic", return_value=heuristic
            ),
            patch(
                "src.routes.analysis.estimate_grade_ml",
                side_effect=GradeEstimationError("ml fail"),
            ),
            patch("src.routes.analysis.generate_explanation", return_value=MagicMock()),
            patch("src.routes.analysis.delete_records"),
            patch("src.routes.analysis.insert_record"),
            patch("src.routes.analysis.update_record"),
        ):
            mock_graph.return_value = MagicMock()
            mock_constrain.return_value = MagicMock()
            mock_features.return_value = MagicMock()

            prediction, _ = _run_grading_pipeline(
                route_id=_ROUTE_ID,
                classified_holds=holds,
                wall_angle=0.0,
                start_hold_ids=[0],
                finish_hold_ids=[2],
                ml_grade_model_path="/models/grade/v1",
            )

        assert prediction is heuristic

    def test_explanation_failure_is_non_fatal(self) -> None:
        """ExplanationError during explanation is logged but not raised."""
        from src.routes.analysis import _run_grading_pipeline
        from src.explanation import ExplanationError

        heuristic = self._mock_heuristic()
        holds = self._make_holds(3)

        with (
            patch("src.routes.analysis.build_route_graph") as mock_graph,
            patch("src.routes.analysis.apply_route_constraints") as mock_constrain,
            patch("src.routes.analysis.assemble_features") as mock_features,
            patch(
                "src.routes.analysis.estimate_grade_heuristic", return_value=heuristic
            ),
            patch(
                "src.routes.analysis.generate_explanation",
                side_effect=ExplanationError("expl fail"),
            ),
            patch("src.routes.analysis.delete_records"),
            patch("src.routes.analysis.insert_record"),
            patch("src.routes.analysis.update_record"),
        ):
            mock_graph.return_value = MagicMock()
            mock_constrain.return_value = MagicMock()
            mock_features.return_value = MagicMock()

            prediction, explanation = _run_grading_pipeline(
                route_id=_ROUTE_ID,
                classified_holds=holds,
                wall_angle=0.0,
                start_hold_ids=[0],
                finish_hold_ids=[2],
                ml_grade_model_path="",
            )

        assert prediction is heuristic
        assert explanation is None

    def test_features_delete_error_propagates(self) -> None:
        """SupabaseClientError from delete_records propagates (real failure, not suppressed)."""
        from src.routes.analysis import _run_grading_pipeline

        heuristic = self._mock_heuristic()
        holds = self._make_holds(3)

        with (
            patch("src.routes.analysis.build_route_graph") as mock_graph,
            patch("src.routes.analysis.apply_route_constraints") as mock_constrain,
            patch("src.routes.analysis.assemble_features") as mock_features,
            patch(
                "src.routes.analysis.estimate_grade_heuristic", return_value=heuristic
            ),
            patch("src.routes.analysis.generate_explanation", return_value=MagicMock()),
            patch(
                "src.routes.analysis.delete_records",
                side_effect=SupabaseClientError("connection refused"),
            ),
            patch("src.routes.analysis.insert_record"),
            patch("src.routes.analysis.update_record"),
        ):
            mock_graph.return_value = MagicMock()
            mock_constrain.return_value = MagicMock()
            mock_features.return_value = MagicMock()

            with pytest.raises(SupabaseClientError):
                _run_grading_pipeline(
                    route_id=_ROUTE_ID,
                    classified_holds=holds,
                    wall_angle=0.0,
                    start_hold_ids=[0],
                    finish_hold_ids=[2],
                    ml_grade_model_path="",
                )


# ---------------------------------------------------------------------------
# TestLoadRouteOr404Timeout
# ---------------------------------------------------------------------------


class TestLoadRouteOr404Timeout:
    """Test the timeout path in _load_route_or_404."""

    def test_load_route_timeout_raises_504(self, client: TestClient) -> None:
        """asyncio.TimeoutError from wait_for in _load_route_or_404 returns 504."""
        import asyncio as _asyncio

        with (
            patch("src.routes.analysis.get_settings", return_value=_MOCK_SETTINGS),
            patch(
                "src.routes.analysis.asyncio.wait_for",
                side_effect=_asyncio.TimeoutError(),
            ),
        ):
            resp = client.post(f"/api/v1/routes/{_ROUTE_ID}/analyze")

        assert resp.status_code == 504
