"""
Tests for Telegram Alert Bot — notification templates + rate limiting.

Covers:
  - Configuration check
  - Message sending (success / failure)
  - Rate limiting
  - All alert templates
  - Bot info
"""

import os
import sys
import time
from collections import deque
from unittest.mock import patch, MagicMock

import pytest

# ── Add project root to path ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import integrations.telegram_bot as tb
from integrations.telegram_bot import (
    is_configured,
    send_message,
    alert_post_created,
    alert_tick_summary,
    alert_error,
    alert_poster_state_change,
    alert_daily_summary,
    send_custom,
    get_bot_info,
)


# ── Fixtures ──

@pytest.fixture(autouse=True)
def reset_rate_limit():
    """Clear rate-limit deque between tests."""
    tb._message_timestamps = deque()
    yield


@pytest.fixture
def telegram_configured(monkeypatch):
    """Patch the module-level constants (set at import time) to valid values."""
    monkeypatch.setattr(tb, "TELEGRAM_BOT_TOKEN", "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")
    monkeypatch.setattr(tb, "TELEGRAM_CHAT_ID", "-1001234567890")


@pytest.fixture
def telegram_not_configured(monkeypatch):
    """Ensure module-level constants are empty."""
    monkeypatch.setattr(tb, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(tb, "TELEGRAM_CHAT_ID", "")


@pytest.fixture
def mock_httpx_client_success():
    """Mock httpx.Client context manager so client.post returns ok=True."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": True, "result": {"message_id": 42}}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post = MagicMock(return_value=mock_response)
    mock_client.get = MagicMock(return_value=mock_response)

    with patch("integrations.telegram_bot.httpx.Client", return_value=mock_client) as _:
        yield mock_client


# ════════════════════════════════════════════════
# Configuration Tests
# ════════════════════════════════════════════════

class TestConfiguration:
    def test_not_configured_by_default(self, telegram_not_configured):
        """Without env vars, not configured."""
        assert is_configured() is False

    def test_configured_with_both(self, telegram_configured):
        """With both vars set, configured."""
        assert is_configured() is True

    def test_missing_token(self, monkeypatch):
        """Only chat_id → not configured."""
        monkeypatch.setattr(tb, "TELEGRAM_BOT_TOKEN", "")
        monkeypatch.setattr(tb, "TELEGRAM_CHAT_ID", "-1001234567890")
        assert is_configured() is False

    def test_missing_chat_id(self, monkeypatch):
        """Only token → not configured."""
        monkeypatch.setattr(tb, "TELEGRAM_BOT_TOKEN", "123456:ABC")
        monkeypatch.setattr(tb, "TELEGRAM_CHAT_ID", "")
        assert is_configured() is False


# ════════════════════════════════════════════════
# Send Message Tests
# ════════════════════════════════════════════════

class TestSendMessage:
    def test_send_success(self, telegram_configured, mock_httpx_client_success):
        """Successful send returns message_id."""
        result = send_message("Hello World")
        assert result["success"] is True
        assert result["message_id"] == 42

        # Verify the client.post was called
        mock_httpx_client_success.post.assert_called_once()
        call_args = mock_httpx_client_success.post.call_args
        assert "sendMessage" in call_args[0][0]

    def test_send_not_configured(self, telegram_not_configured):
        """Without credentials, send fails gracefully."""
        result = send_message("Hello")
        assert result["success"] is False
        assert "not configured" in result["error"].lower()

    def test_send_network_error(self, telegram_configured):
        """Network error returns error dict."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = MagicMock(side_effect=Exception("Timeout"))

        with patch("integrations.telegram_bot.httpx.Client", return_value=mock_client):
            result = send_message("Hello")
            assert result["success"] is False
            assert "Timeout" in result["error"]

    def test_send_custom_chat_id(self, telegram_configured, mock_httpx_client_success):
        """Can override chat_id per call."""
        result = send_message("Hello", chat_id="-999")
        assert result["success"] is True

    def test_send_markdown_parse_mode(self, telegram_configured, mock_httpx_client_success):
        """Default parse mode is Markdown."""
        send_message("*bold* text")
        call_args = mock_httpx_client_success.post.call_args
        payload = call_args[1].get("json", {})
        assert payload.get("parse_mode") == "Markdown"


# ════════════════════════════════════════════════
# Rate Limiting Tests
# ════════════════════════════════════════════════

class TestRateLimiting:
    def test_under_limit(self, telegram_configured, mock_httpx_client_success):
        """Under 30/min should succeed."""
        for i in range(5):
            result = send_message(f"Message {i}")
            assert result["success"] is True

    def test_at_limit(self, telegram_configured, mock_httpx_client_success):
        """At exactly 30 messages → next one should be rate-limited."""
        now = time.time()
        tb._message_timestamps = deque([now] * 30)

        result = send_message("One more")
        assert result["success"] is False
        assert "rate" in result["error"].lower()


# ════════════════════════════════════════════════
# Alert Template Tests
# ════════════════════════════════════════════════

class TestAlertTemplates:
    def test_alert_post_created(self, telegram_configured, mock_httpx_client_success):
        """Post creation alert includes poster name + platforms."""
        result = alert_post_created(
            poster_name="Tech News",
            title="New AI Breakthrough",
            platforms=["tiktok", "instagram"],
            media_type="video",
            has_music=True,
            post_count=5,
        )
        assert result["success"] is True
        # Verify message content
        call_args = mock_httpx_client_success.post.call_args
        payload = call_args[1].get("json", {})
        text = payload.get("text", "")
        assert "Tech News" in text
        assert "AI Breakthrough" in text

    def test_alert_tick_summary_with_posts(self, telegram_configured, mock_httpx_client_success):
        """Tick summary with actual posts → sends message."""
        results = [
            {"name": "Tech", "posted": 3},
            {"name": "Health", "posted": 1},
        ]
        result = alert_tick_summary(
            posters_checked=5,
            total_posted=4,
            results=results,
        )
        assert result["success"] is True

    def test_alert_tick_summary_nothing_posted(self, telegram_configured):
        """Tick summary with zero posts → skipped."""
        result = alert_tick_summary(
            posters_checked=5,
            total_posted=0,
        )
        assert result["success"] is True
        assert result.get("skipped") is True

    def test_alert_error(self, telegram_configured, mock_httpx_client_success):
        """Error alert includes source + error message."""
        result = alert_error(
            source="RSS Feed Reader",
            error="Connection timeout after 30s",
            context={"feed_url": "https://example.com/rss"},
        )
        assert result["success"] is True

    def test_alert_poster_state_change(self, telegram_configured, mock_httpx_client_success):
        """State change alert includes poster info."""
        result = alert_poster_state_change(
            poster_name="Health Blog",
            poster_id="poster_123",
            new_state="paused",
        )
        assert result["success"] is True

    def test_alert_daily_summary(self, telegram_configured, mock_httpx_client_success):
        """Daily summary includes stats."""
        result = alert_daily_summary(
            total_posters=10,
            active_posters=7,
            total_posted_today=42,
            top_poster="Tech News",
            errors_today=2,
        )
        assert result["success"] is True

    def test_send_custom(self, telegram_configured, mock_httpx_client_success):
        """Custom message gets sent."""
        result = send_custom("Testing 1 2 3")
        assert result["success"] is True


# ════════════════════════════════════════════════
# Bot Info Tests
# ════════════════════════════════════════════════

class TestBotInfo:
    def test_get_bot_info_success(self, telegram_configured):
        """Bot info returns username + bot_name."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "ok": True,
            "result": {
                "id": 123456,
                "is_bot": True,
                "first_name": "ViralOps Bot",
                "username": "viralops_bot",
            },
        }

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("integrations.telegram_bot.httpx.Client", return_value=mock_client):
            result = get_bot_info()
            assert result["configured"] is True
            assert result["bot_username"] == "viralops_bot"
            assert result["bot_name"] == "ViralOps Bot"

    def test_get_bot_info_not_configured(self, telegram_not_configured):
        """Without token, returns configured=False."""
        result = get_bot_info()
        assert result["configured"] is False

    def test_get_bot_info_api_error(self, telegram_configured):
        """API error returns error key."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(side_effect=Exception("API down"))

        with patch("integrations.telegram_bot.httpx.Client", return_value=mock_client):
            result = get_bot_info()
            assert result["configured"] is True
            assert "API down" in result["error"]
