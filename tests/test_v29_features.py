"""
Tests for v2.9.0 — Live Preview + Analytics Charts + Enhanced UI
"""
import pytest
from fastapi.testclient import TestClient

# ── Import app ──
import sys, pathlib
_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
from web.app import app

client = TestClient(app)


# ════════════════════════════════════════════════════════════════
# 1. NEW PAGE ROUTES
# ════════════════════════════════════════════════════════════════

class TestPreviewPageRoute:
    """Test the /preview HTML page route."""

    def test_preview_page_returns_200(self):
        r = client.get("/preview")
        assert r.status_code == 200

    def test_preview_page_is_html(self):
        r = client.get("/preview")
        assert "text/html" in r.headers.get("content-type", "")

    def test_preview_page_has_preview_section(self):
        r = client.get("/preview")
        assert 'id="page-preview"' in r.text

    def test_preview_page_has_twitter_preview(self):
        r = client.get("/preview")
        assert "pv-tw-text" in r.text

    def test_preview_page_has_instagram_preview(self):
        r = client.get("/preview")
        assert "pv-ig-caption" in r.text

    def test_preview_page_has_linkedin_preview(self):
        r = client.get("/preview")
        assert "pv-li-text" in r.text

    def test_preview_page_has_facebook_preview(self):
        r = client.get("/preview")
        assert "pv-fb-text" in r.text

    def test_preview_page_has_tiktok_preview(self):
        r = client.get("/preview")
        assert "pv-tt-text" in r.text

    def test_preview_page_has_reddit_preview(self):
        r = client.get("/preview")
        assert "pv-rd-title" in r.text

    def test_preview_page_has_char_stats(self):
        r = client.get("/preview")
        assert "pv-char-stats" in r.text

    def test_preview_page_has_image_input(self):
        r = client.get("/preview")
        assert "pv-image" in r.text


# ════════════════════════════════════════════════════════════════
# 2. EXISTING PAGE ROUTES STILL WORK
# ════════════════════════════════════════════════════════════════

class TestExistingPageRoutes:
    """Ensure all existing page routes still work after changes."""

    @pytest.mark.parametrize("path,page_id", [
        ("/", "page-dashboard"),
        ("/compose", "page-compose"),
        ("/content", "page-content"),
        ("/calendar", "page-calendar"),
        ("/rss", "page-rss"),
        ("/hashtags", "page-hashtags"),
        ("/analytics", "page-analytics"),
        ("/settings", "page-settings"),
        ("/platforms", "page-platforms"),
        ("/engagement", "page-engagement"),
        ("/time-slots", "page-timeslots"),
        ("/preview", "page-preview"),
    ])
    def test_page_returns_200_and_has_section(self, path, page_id):
        r = client.get(path)
        assert r.status_code == 200
        assert page_id in r.text


# ════════════════════════════════════════════════════════════════
# 3. ANALYTICS CHART ELEMENTS
# ════════════════════════════════════════════════════════════════

class TestAnalyticsCharts:
    """Test analytics page has Chart.js canvas elements."""

    def test_chart_js_loaded(self):
        r = client.get("/analytics")
        assert "chart.js" in r.text.lower() or "chart.umd" in r.text.lower()

    def test_trend_chart_canvas(self):
        r = client.get("/analytics")
        assert 'id="chart-trend"' in r.text

    def test_platform_chart_canvas(self):
        r = client.get("/analytics")
        assert 'id="chart-platforms"' in r.text

    def test_hours_chart_canvas(self):
        r = client.get("/analytics")
        assert 'id="chart-hours"' in r.text

    def test_top_posts_section(self):
        r = client.get("/analytics")
        assert 'id="top-posts"' in r.text

    def test_period_buttons(self):
        r = client.get("/analytics")
        assert 'setAnalyticsPeriod(7)' in r.text
        assert 'setAnalyticsPeriod(30)' in r.text
        assert 'setAnalyticsPeriod(90)' in r.text


# ════════════════════════════════════════════════════════════════
# 4. SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════════

class TestSidebarNav:
    """Test sidebar has all navigation links."""

    def test_sidebar_has_preview_link(self):
        r = client.get("/")
        assert 'href="/preview"' in r.text

    def test_sidebar_has_engagement_link(self):
        r = client.get("/")
        assert 'href="/engagement"' in r.text

    def test_sidebar_has_timeslots_link(self):
        r = client.get("/")
        assert 'href="/time-slots"' in r.text

    def test_sidebar_has_platforms_link(self):
        r = client.get("/")
        assert 'href="/platforms"' in r.text

    def test_sidebar_has_dynamic_channels(self):
        r = client.get("/")
        assert 'id="sidebar-channels"' in r.text


# ════════════════════════════════════════════════════════════════
# 5. LIVE PREVIEW JS FUNCTIONS
# ════════════════════════════════════════════════════════════════

class TestPreviewJSFunctions:
    """Test that preview JS functions exist in the page."""

    def test_update_previews_function(self):
        r = client.get("/preview")
        assert "function updatePreviews()" in r.text

    def test_platform_limits_const(self):
        r = client.get("/preview")
        assert "PLATFORM_LIMITS" in r.text

    def test_twitter_limit_280(self):
        r = client.get("/preview")
        assert "twitter: 280" in r.text or "twitter:280" in r.text

    def test_instagram_limit_2200(self):
        r = client.get("/preview")
        assert "instagram: 2200" in r.text or "instagram:2200" in r.text


# ════════════════════════════════════════════════════════════════
# 6. ANALYTICS CHART JS FUNCTIONS
# ════════════════════════════════════════════════════════════════

class TestAnalyticsChartFunctions:
    """Test analytics chart JS functions exist."""

    def test_build_trend_chart_function(self):
        r = client.get("/analytics")
        assert "function buildTrendChart" in r.text

    def test_build_platform_chart_function(self):
        r = client.get("/analytics")
        assert "function buildPlatformChart" in r.text

    def test_build_hours_chart_function(self):
        r = client.get("/analytics")
        assert "function buildHoursChart" in r.text

    def test_render_top_posts_function(self):
        r = client.get("/analytics")
        assert "function renderTopPosts" in r.text

    def test_set_analytics_period_function(self):
        r = client.get("/analytics")
        assert "function setAnalyticsPeriod" in r.text


# ════════════════════════════════════════════════════════════════
# 7. API ENDPOINTS STILL WORK
# ════════════════════════════════════════════════════════════════

class TestAPIsIntact:
    """Verify critical APIs still return valid responses."""

    def test_health_endpoint(self):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"

    def test_stats_endpoint(self):
        r = client.get("/api/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total" in data

    def test_analytics_endpoint(self):
        r = client.get("/api/analytics")
        assert r.status_code == 200

    def test_platforms_setup_status(self):
        r = client.get("/api/platforms/setup-status")
        assert r.status_code == 200
        data = r.json()
        assert "platforms" in data

    def test_engagement_summary(self):
        r = client.get("/api/engagement/summary")
        assert r.status_code == 200

    def test_time_slots_best_hours(self):
        r = client.get("/api/time-slots/best-hours")
        assert r.status_code == 200
