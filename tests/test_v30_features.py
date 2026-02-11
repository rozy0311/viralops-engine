"""
Tests for v3.0.0 — Auto-Pilot + 16-Platform Compose + Bulk Actions
"""
import pytest
import json
from fastapi.testclient import TestClient

# ── Import app ──
import sys, pathlib
_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
from web.app import app

client = TestClient(app)


# ════════════════════════════════════════════════════════════════
# 1. AUTO-PILOT PAGE ROUTE
# ════════════════════════════════════════════════════════════════

class TestAutopilotPageRoute:
    """Test /autopilot HTML page route."""

    def test_autopilot_page_returns_200(self):
        r = client.get("/autopilot")
        assert r.status_code == 200

    def test_autopilot_page_is_html(self):
        r = client.get("/autopilot")
        assert "text/html" in r.headers.get("content-type", "")

    def test_autopilot_page_has_section(self):
        r = client.get("/autopilot")
        assert 'id="page-autopilot"' in r.text

    def test_autopilot_page_has_niche_selector(self):
        r = client.get("/autopilot")
        assert 'id="ap-niche"' in r.text

    def test_autopilot_page_has_platform_list(self):
        r = client.get("/autopilot")
        assert 'id="ap-platforms"' in r.text

    def test_autopilot_page_has_launch_button(self):
        r = client.get("/autopilot")
        assert "runAutoPilot" in r.text

    def test_autopilot_page_has_progress_section(self):
        r = client.get("/autopilot")
        assert 'id="ap-progress"' in r.text

    def test_autopilot_page_has_result_section(self):
        r = client.get("/autopilot")
        assert 'id="ap-result"' in r.text


# ════════════════════════════════════════════════════════════════
# 2. AUTOPILOT GENERATE API
# ════════════════════════════════════════════════════════════════

class TestAutopilotGenerateAPI:
    """Test /api/autopilot/generate endpoint."""

    def test_generate_returns_200(self):
        r = client.post("/api/autopilot/generate", json={
            "niche": "fitness", "platforms": ["twitter"]
        })
        assert r.status_code == 200

    def test_generate_returns_title(self):
        r = client.post("/api/autopilot/generate", json={
            "niche": "fitness", "platforms": ["twitter"]
        })
        data = r.json()
        assert "title" in data
        assert len(data["title"]) > 0

    def test_generate_returns_body(self):
        r = client.post("/api/autopilot/generate", json={
            "niche": "fitness", "platforms": ["twitter"]
        })
        data = r.json()
        assert "body" in data
        assert len(data["body"]) > 0

    def test_generate_returns_hashtags(self):
        r = client.post("/api/autopilot/generate", json={
            "niche": "fitness", "platforms": ["instagram"]
        })
        data = r.json()
        assert "hashtags" in data
        assert isinstance(data["hashtags"], list)

    def test_generate_with_topic(self):
        r = client.post("/api/autopilot/generate", json={
            "niche": "tech", "topic": "Python tips", "platforms": ["linkedin"]
        })
        data = r.json()
        assert r.status_code == 200
        assert "title" in data

    def test_generate_with_empty_platforms(self):
        r = client.post("/api/autopilot/generate", json={
            "niche": "food", "platforms": []
        })
        assert r.status_code == 200

    def test_generate_default_niche(self):
        r = client.post("/api/autopilot/generate", json={"platforms": ["twitter"]})
        data = r.json()
        assert r.status_code == 200
        assert "title" in data


# ════════════════════════════════════════════════════════════════
# 3. PUT /api/posts/{id} — UPDATE ENDPOINT
# ════════════════════════════════════════════════════════════════

class TestUpdatePostAPI:
    """Test PUT /api/posts/{id} endpoint."""

    def _create_post(self):
        r = client.post("/api/posts", json={
            "title": "Test", "body": "Body", "category": "general",
            "platforms": ["twitter"], "status": "draft"
        })
        return r.json()["post_id"]

    def test_update_status(self):
        pid = self._create_post()
        r = client.put(f"/api/posts/{pid}", json={"status": "scheduled"})
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_update_title(self):
        pid = self._create_post()
        r = client.put(f"/api/posts/{pid}", json={"title": "Updated Title"})
        assert r.json()["success"] is True

    def test_update_body(self):
        pid = self._create_post()
        r = client.put(f"/api/posts/{pid}", json={"body": "New body content"})
        assert r.json()["success"] is True

    def test_update_platforms_list(self):
        pid = self._create_post()
        r = client.put(f"/api/posts/{pid}", json={"platforms": ["instagram", "linkedin"]})
        assert r.json()["success"] is True

    def test_update_scheduled_at(self):
        pid = self._create_post()
        r = client.put(f"/api/posts/{pid}", json={
            "status": "scheduled", "scheduled_at": "2025-12-01T10:00:00"
        })
        assert r.json()["success"] is True

    def test_update_no_fields(self):
        pid = self._create_post()
        r = client.put(f"/api/posts/{pid}", json={})
        data = r.json()
        assert data["success"] is False

    def test_update_extra_fields(self):
        pid = self._create_post()
        r = client.put(f"/api/posts/{pid}", json={
            "extra_fields": {"hashtags": ["#test"]}
        })
        assert r.json()["success"] is True


# ════════════════════════════════════════════════════════════════
# 4. COMPOSE PAGE — DYNAMIC 16-PLATFORM CHIPS
# ════════════════════════════════════════════════════════════════

class TestComposeDynamicPlatforms:
    """Test Compose page has dynamic platform chip infrastructure."""

    def test_compose_has_chip_container(self):
        r = client.get("/compose")
        assert 'id="platform-chips"' in r.text

    def test_compose_has_select_all_button(self):
        r = client.get("/compose")
        assert "selectAllPlatforms" in r.text

    def test_compose_has_clear_all_button(self):
        r = client.get("/compose")
        assert "clearAllPlatforms" in r.text

    def test_compose_has_platform_count(self):
        r = client.get("/compose")
        assert 'id="compose-plat-count"' in r.text

    def test_compose_has_extra_fields_container(self):
        r = client.get("/compose")
        assert 'id="extra-fields"' in r.text

    def test_compose_has_char_limits_container(self):
        r = client.get("/compose")
        assert 'id="compose-limits"' in r.text

    def test_compose_has_loadComposeChips_function(self):
        r = client.get("/compose")
        assert "loadComposeChips" in r.text

    def test_compose_has_renderExtraFields_function(self):
        r = client.get("/compose")
        assert "renderExtraFields" in r.text

    def test_compose_has_plat_chip_colors(self):
        r = client.get("/compose")
        assert "PLAT_CHIP_COLORS" in r.text

    def test_compose_has_plat_extra_fields_config(self):
        r = client.get("/compose")
        assert "PLAT_EXTRA_FIELDS" in r.text

    def test_compose_has_16_platforms_in_config(self):
        """Ensure all 16 platforms are defined in PLAT_EXTRA_FIELDS."""
        r = client.get("/compose")
        platforms = ["twitter", "instagram", "facebook", "youtube", "linkedin",
                     "tiktok", "pinterest", "reddit", "medium", "tumblr",
                     "threads", "bluesky", "mastodon", "quora", "shopify_blog", "lemon8"]
        for p in platforms:
            assert p in r.text, f"Platform {p} missing from compose config"


# ════════════════════════════════════════════════════════════════
# 5. CONTENT LIBRARY — BULK ACTIONS
# ════════════════════════════════════════════════════════════════

class TestContentLibraryBulkActions:
    """Test Content Library has bulk action infrastructure."""

    def test_content_has_bulk_bar(self):
        r = client.get("/content")
        assert 'id="bulk-bar"' in r.text

    def test_content_has_bulk_publish_button(self):
        r = client.get("/content")
        assert "bulkPublish" in r.text

    def test_content_has_bulk_delete_button(self):
        r = client.get("/content")
        assert "bulkDelete" in r.text

    def test_content_has_bulk_schedule_button(self):
        r = client.get("/content")
        assert "bulkSchedule" in r.text

    def test_content_has_select_all_checkbox(self):
        r = client.get("/content")
        assert "toggleSelectAll" in r.text

    def test_content_has_clear_bulk_button(self):
        r = client.get("/content")
        assert "clearBulk" in r.text

    def test_content_has_bulk_count_display(self):
        r = client.get("/content")
        assert 'id="bulk-count"' in r.text


# ════════════════════════════════════════════════════════════════
# 6. SIDEBAR NAV — AUTO-PILOT LINK
# ════════════════════════════════════════════════════════════════

class TestSidebarAutoPilotLink:
    """Test sidebar has Auto-Pilot navigation link."""

    def test_sidebar_has_autopilot_link(self):
        r = client.get("/")
        assert "/autopilot" in r.text

    def test_autopilot_link_has_icon(self):
        r = client.get("/")
        assert "🤖" in r.text

    def test_autopilot_link_text(self):
        r = client.get("/")
        assert "Auto-Pilot" in r.text


# ════════════════════════════════════════════════════════════════
# 7. AUTO-PILOT JS FUNCTIONS
# ════════════════════════════════════════════════════════════════

class TestAutoPilotJSFunctions:
    """Verify Auto-Pilot JS function definitions exist."""

    def test_has_loadAutoPilotPlatforms(self):
        r = client.get("/autopilot")
        assert "loadAutoPilotPlatforms" in r.text

    def test_has_runAutoPilot(self):
        r = client.get("/autopilot")
        assert "runAutoPilot" in r.text

    def test_has_apConfirmPublish(self):
        r = client.get("/autopilot")
        assert "apConfirmPublish" in r.text

    def test_has_apEditAndPublish(self):
        r = client.get("/autopilot")
        assert "apEditAndPublish" in r.text

    def test_has_apRegenerate(self):
        r = client.get("/autopilot")
        assert "apRegenerate" in r.text

    def test_has_apSelectAll(self):
        r = client.get("/autopilot")
        assert "apSelectAll" in r.text

    def test_has_apClearAll(self):
        r = client.get("/autopilot")
        assert "apClearAll" in r.text

    def test_has_apSetStep(self):
        r = client.get("/autopilot")
        assert "apSetStep" in r.text


# ════════════════════════════════════════════════════════════════
# 8. AUTO-PILOT PAGE ELEMENTS (detailed)
# ════════════════════════════════════════════════════════════════

class TestAutoPilotPageElements:
    """Test detailed Auto-Pilot page UI elements."""

    def test_has_niche_options(self):
        r = client.get("/autopilot")
        assert "fitness" in r.text.lower()
        assert "tech" in r.text.lower()

    def test_has_topic_input(self):
        r = client.get("/autopilot")
        assert 'id="ap-topic"' in r.text

    def test_has_posts_per_platform(self):
        r = client.get("/autopilot")
        assert 'id="ap-posts-per"' in r.text

    def test_has_utc_offset(self):
        r = client.get("/autopilot")
        assert 'id="ap-utc"' in r.text

    def test_has_auto_hashtags_toggle(self):
        r = client.get("/autopilot")
        assert 'id="ap-auto-hashtags"' in r.text

    def test_has_best_times_toggle(self):
        r = client.get("/autopilot")
        assert 'id="ap-best-times"' in r.text

    def test_has_generated_title_area(self):
        r = client.get("/autopilot")
        assert 'id="ap-gen-title"' in r.text

    def test_has_generated_body_area(self):
        r = client.get("/autopilot")
        assert 'id="ap-gen-body"' in r.text

    def test_has_generated_tags_area(self):
        r = client.get("/autopilot")
        assert 'id="ap-gen-tags"' in r.text

    def test_has_schedule_list(self):
        r = client.get("/autopilot")
        assert 'id="ap-schedule-list"' in r.text

    def test_has_confirm_button(self):
        r = client.get("/autopilot")
        assert "apConfirmPublish" in r.text

    def test_has_edit_button(self):
        r = client.get("/autopilot")
        assert "apEditAndPublish" in r.text

    def test_has_regenerate_button(self):
        r = client.get("/autopilot")
        assert "apRegenerate" in r.text

    def test_has_empty_state(self):
        r = client.get("/autopilot")
        assert 'id="ap-empty"' in r.text
