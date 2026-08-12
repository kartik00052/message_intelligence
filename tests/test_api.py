"""API and dashboard tests for the FastAPI application.

These tests run against the real generated artifacts in ``outputs/`` so they
verify exactly what the dashboard will render during the demonstration. Every
assertion that touches message content works on sanitized output only - the
tests assert that raw sensitive values can never appear in a response.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models.classification import Category
from app.models.task_event import ItemType
from app.services.mandatory_demo import load_mandatory_ids

CATEGORY_VALUES = {category.value for category in Category}
ITEM_TYPE_VALUES = {item_type.value for item_type in ItemType}

# Sensitivity types that are masked with an all-asterisk placeholder.
_STAR_MASKED_TYPES = {
    "one_time_password",
    "password",
    "pin",
    "account_recovery_code",
    "payment_card_number",
    "bank_account_number",
    "upi_payment_identifier",
}


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


# ------------------------------------------------------------------- health


class TestHealth:
    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ------------------------------------------------------------------- stats


class TestStats:
    def test_stats_expected_totals(self, client: TestClient) -> None:
        data = client.get("/api/stats").json()
        assert data["total_messages"] == 900
        assert data["classified_messages"] == 900
        assert data["sensitive_messages"] == 100
        assert data["task_event_count"] == 360
        assert data["rule_based_count"] == 900
        assert data["llm_fallback_count"] == 0
        assert data["validation_status"] == "PASS"


# ---------------------------------------------------------------- messages


class TestMessages:
    def test_list_returns_all_messages(self, client: TestClient) -> None:
        data = client.get("/api/messages?limit=900").json()
        assert data["total"] == 900
        assert len(data["items"]) == 900

    def test_list_default_pagination(self, client: TestClient) -> None:
        data = client.get("/api/messages").json()
        assert data["total"] == 900
        assert data["offset"] == 0
        assert data["limit"] == 100
        assert len(data["items"]) == 100

    def test_list_offset_pagination(self, client: TestClient) -> None:
        first = client.get("/api/messages?limit=100&offset=0").json()
        second = client.get("/api/messages?limit=100&offset=100").json()
        first_ids = [item["message_id"] for item in first["items"]]
        second_ids = [item["message_id"] for item in second["items"]]
        assert not set(first_ids) & set(second_ids)

    def test_item_shape(self, client: TestClient) -> None:
        item = client.get("/api/messages?limit=1").json()["items"][0]
        assert set(item) == {
            "message_id",
            "timestamp",
            "sender",
            "category",
            "confidence",
            "method",
            "has_sensitive",
        }
        assert item["category"] in CATEGORY_VALUES
        assert 0.0 <= item["confidence"] <= 1.0

    def test_search_matches_id_sender_and_content(self, client: TestClient) -> None:
        first = client.get("/api/messages?limit=1").json()["items"][0]
        message_id = first["message_id"]
        by_id = client.get(f"/api/messages?search={message_id}&limit=900").json()
        assert by_id["total"] >= 1
        assert message_id in {item["message_id"] for item in by_id["items"]}

        by_sender = client.get(
            f"/api/messages?search={first['sender']}&limit=900"
        ).json()
        assert by_sender["total"] >= 1

    def test_category_filter(self, client: TestClient) -> None:
        for category in CATEGORY_VALUES:
            data = client.get(f"/api/messages?category={category}&limit=900").json()
            assert data["total"] > 0
            assert {item["category"] for item in data["items"]} == {category}

    def test_sensitive_filter(self, client: TestClient) -> None:
        only_sensitive = client.get("/api/messages?sensitive=true&limit=900").json()
        non_sensitive = client.get("/api/messages?sensitive=false&limit=900").json()
        assert only_sensitive["total"] == 100
        assert non_sensitive["total"] == 800
        assert all(item["has_sensitive"] for item in only_sensitive["items"])
        assert all(not item["has_sensitive"] for item in non_sensitive["items"])

    def test_detail_shape(self, client: TestClient) -> None:
        message_id = client.get("/api/messages?limit=1").json()["items"][0]["message_id"]
        detail = client.get(f"/api/messages/{message_id}").json()
        assert set(detail) == {
            "message_id",
            "timestamp",
            "sender",
            "safe_message",
            "classification",
            "security",
            "extracted_items",
        }
        assert detail["message_id"] == message_id
        assert detail["safe_message"]
        assert detail["classification"]["category"] in CATEGORY_VALUES
        assert 0.0 <= detail["classification"]["confidence"] <= 1.0
        assert detail["classification"]["reason"]
        assert detail["classification"]["method"] in {"rule_based", "llm_fallback"}

    def test_detail_not_found(self, client: TestClient) -> None:
        response = client.get("/api/messages/MSG_9999")
        assert response.status_code == 404
        assert "No processed result" in response.json()["detail"]


# ------------------------------------------------------------------- tasks


class TestTasks:
    def test_tasks_list(self, client: TestClient) -> None:
        data = client.get("/api/tasks?limit=900").json()
        assert data["total"] == 360
        assert len(data["items"]) == 360

    def test_tasks_item_shape(self, client: TestClient) -> None:
        item = client.get("/api/tasks?limit=1").json()["items"][0]
        assert set(item) == {
            "item_id",
            "type",
            "title",
            "description",
            "date",
            "deadline",
            "time",
            "person",
            "priority",
            "source_message_id",
        }
        assert item["type"] in ITEM_TYPE_VALUES
        assert item["priority"] in {"low", "medium", "high", "unknown"}
        assert item["title"]
        assert item["source_message_id"]

    def test_tasks_type_filter(self, client: TestClient) -> None:
        for item_type in ITEM_TYPE_VALUES:
            data = client.get(f"/api/tasks?type={item_type}&limit=900").json()
            assert data["total"] > 0
            assert {item["type"] for item in data["items"]} == {item_type}


# --------------------------------------------------------------- sensitive


class TestSensitive:
    def test_sensitive_list_total(self, client: TestClient) -> None:
        data = client.get("/api/sensitive?limit=900").json()
        assert data["total"] == 100
        assert len(data["items"]) == 100

    def test_detections_are_masked_only(self, client: TestClient) -> None:
        data = client.get("/api/sensitive?limit=900").json()
        raw_payload = json.dumps(data)
        assert "matched_value" not in raw_payload
        for result in data["items"]:
            assert result["has_detection"] is True
            for detection in result["detections"]:
                assert detection["detected"] is True
                assert detection["risk"] in {"low", "medium", "high"}
                assert detection["recommended_action"]
                text = detection["masked_text"]
                if detection["sensitivity_type"] in _STAR_MASKED_TYPES:
                    assert text and set(text) == {"*"}
                else:
                    assert text.startswith("[REDACTED")


# ----------------------------------------------------------- mandatory demo


class TestMandatoryDemo:
    def test_all_fifteen_loaded_from_file(self, client: TestClient) -> None:
        data = client.get("/api/demo/mandatory").json()
        assert data["found"] == 15
        assert data["processed"] == 15
        assert data["missing"] == []
        assert len(data["results"]) == 15

        file_ids = load_mandatory_ids(Settings().mandatory_demo_ids_path)
        assert tuple(data["requested_ids"]) == file_ids
        assert {result["message_id"] for result in data["results"]} == set(file_ids)

    def test_mandatory_results_are_sanitized(self, client: TestClient) -> None:
        data = client.get("/api/demo/mandatory").json()
        for result in data["results"]:
            assert result["safe_message"]
            assert result["classification"]["category"] in CATEGORY_VALUES
            assert 0.0 <= result["classification"]["confidence"] <= 1.0
            assert result["classification"]["reason"]
            assert result["classification"]["method"] in {"rule_based", "llm_fallback"}
            for item in result["extracted_items"]:
                assert item["type"] in ITEM_TYPE_VALUES
                assert item["title"]
                assert item["source_message_id"] == result["message_id"]

    def test_every_mandatory_message_loads_through_detail_endpoint(
        self, client: TestClient
    ) -> None:
        """Walk all 15 mandatory messages exactly as the UI does on card click."""
        data = client.get("/api/demo/mandatory").json()
        assert len(data["results"]) == 15
        for demo_message in data["results"]:
            message_id = demo_message["message_id"]
            detail = client.get(f"/api/messages/{message_id}")
            assert detail.status_code == 200, f"detail failed for {message_id}"
            body = detail.json()
            assert body["message_id"] == message_id
            assert body["safe_message"]
            assert body["classification"]["category"] in CATEGORY_VALUES
            assert 0.0 <= body["classification"]["confidence"] <= 1.0
            assert body["classification"]["reason"]
            assert body["security"]["has_detection"] is bool(
                demo_message["security"]["has_detection"]
            )
            if body["security"]["has_detection"]:
                assert body["security"]["detections"]
                for detection in body["security"]["detections"]:
                    text = detection["masked_text"]
                    if detection["sensitivity_type"] in _STAR_MASKED_TYPES:
                        assert text and set(text) == {"*"}
                    else:
                        assert text.startswith("[REDACTED")

    def test_mandatory_ids_are_not_hardcoded_in_frontend(self) -> None:
        frontend = (
            Path(__file__).resolve().parent.parent / "app" / "static" / "app.js"
        ).read_text(encoding="utf-8")
        assert "MSG_0001" not in frontend
        assert "MSG_0015" not in frontend
        assert '"/api/demo/mandatory"' in frontend


# -------------------------------------------------------------- validation


class TestValidation:
    def test_validation_report(self, client: TestClient) -> None:
        data = client.get("/api/validation").json()
        assert data["validation_status"] == "PASS"
        assert data["leak_scan"]["ok"] is True
        assert data["leak_scan"]["findings"] == []
        assert data["summary"]["total_messages"] == 900
        assert data["summary"]["classified_messages"] == 900
        assert data["report"]["missing_message_ids"] == 0
        assert data["report"]["duplicate_message_ids"] == 0
        assert data["report"]["sensitive_value_leak_check"] == "PASS"
        assert data["report"]["mandatory_messages_found"] == 15
        assert data["report"]["mandatory_messages_processed"] == 15


# ---------------------------------------------------------------- dashboard


class TestDashboard:
    def test_dashboard_html(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        html = response.text
        assert "Message Intelligence" in html
        assert "Mandatory Demo" in html
        assert "Pipeline Status" in html
        assert "/static/styles.css" in html
        assert "/static/app.js" in html

    def test_dashboard_has_no_raw_sensitive_values(self, client: TestClient) -> None:
        html = client.get("/").text
        assert "[REDACTED_ADDRESS]" not in html
        assert "*" * 12 not in html

    def test_static_assets_served(self, client: TestClient) -> None:
        css = client.get("/static/styles.css")
        assert css.status_code == 200
        assert "text/css" in css.headers["content-type"]
        js = client.get("/static/app.js")
        assert js.status_code == 200
        assert "javascript" in js.headers["content-type"]
