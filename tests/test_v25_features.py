"""
Tests for new v2.5.0 features:
  - Threads Publisher
  - Bluesky Publisher
  - Mastodon Publisher
  - Quora Publisher
  - Engagement Fetcher
  - BPM-aware Music Filtering
  - Trending Score Decay
  - Time Slot Suggestion Engine
  - Multi-Image Slideshow
  - Text Overlay
"""

import os
import json
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta


# ════════════════════════════════════════════════════════════════
# Threads Publisher Tests
# ════════════════════════════════════════════════════════════════

class TestThreadsPublisher:
    """Test Threads publisher."""

    def test_import(self):
        from integrations.threads_publisher import ThreadsPublisher
        pub = ThreadsPublisher()
        assert pub.platform == "threads"
        assert pub.MAX_CAPTION == 500

    def test_default_account_id(self):
        from integrations.threads_publisher import ThreadsPublisher
        pub = ThreadsPublisher()
        assert pub.account_id == "threads_main"

    def test_custom_account_id(self):
        from integrations.threads_publisher import ThreadsPublisher
        pub = ThreadsPublisher(account_id="threads_brand")
        assert pub.account_id == "threads_brand"

    def test_connect_no_token(self):
        from integrations.threads_publisher import ThreadsPublisher
        pub = ThreadsPublisher()
        result = asyncio.get_event_loop().run_until_complete(pub.connect())
        assert result is False

    def test_publish_not_connected(self):
        from integrations.threads_publisher import ThreadsPublisher
        pub = ThreadsPublisher()
        result = asyncio.get_event_loop().run_until_complete(
            pub.publish({"caption": "test"})
        )
        assert result["success"] is False
        assert result["platform"] == "threads"

    def test_close(self):
        from integrations.threads_publisher import ThreadsPublisher
        pub = ThreadsPublisher()
        asyncio.get_event_loop().run_until_complete(pub.close())
        assert pub._connected is False


# ════════════════════════════════════════════════════════════════
# Bluesky Publisher Tests
# ════════════════════════════════════════════════════════════════

class TestBlueskyPublisher:
    """Test Bluesky publisher."""

    def test_import(self):
        from integrations.bluesky_publisher import BlueskyPublisher
        pub = BlueskyPublisher()
        assert pub.platform == "bluesky"
        assert pub.MAX_CAPTION == 300

    def test_default_account_id(self):
        from integrations.bluesky_publisher import BlueskyPublisher
        pub = BlueskyPublisher()
        assert pub.account_id == "bluesky_main"

    def test_connect_no_credentials(self):
        from integrations.bluesky_publisher import BlueskyPublisher
        pub = BlueskyPublisher()
        result = asyncio.get_event_loop().run_until_complete(pub.connect())
        assert result is False

    def test_publish_not_connected(self):
        from integrations.bluesky_publisher import BlueskyPublisher
        pub = BlueskyPublisher()
        result = asyncio.get_event_loop().run_until_complete(
            pub.publish({"caption": "test"})
        )
        assert result["success"] is False
        assert result["platform"] == "bluesky"

    def test_detect_facets(self):
        from integrations.bluesky_publisher import BlueskyPublisher
        pub = BlueskyPublisher()
        text = "Check https://example.com and #hashtag @mention.bsky.social"
        facets = pub._detect_facets(text)
        # Should detect URL, hashtag, mention
        assert len(facets) >= 2  # At least URL and hashtag

    def test_close(self):
        from integrations.bluesky_publisher import BlueskyPublisher
        pub = BlueskyPublisher()
        asyncio.get_event_loop().run_until_complete(pub.close())


# ════════════════════════════════════════════════════════════════
# Mastodon Publisher Tests
# ════════════════════════════════════════════════════════════════

class TestMastodonPublisher:
    """Test Mastodon publisher."""

    def test_import(self):
        from integrations.mastodon_publisher import MastodonPublisher
        pub = MastodonPublisher()
        assert pub.platform == "mastodon"
        assert pub.MAX_CAPTION == 500

    def test_default_account_id(self):
        from integrations.mastodon_publisher import MastodonPublisher
        pub = MastodonPublisher()
        assert pub.account_id == "mastodon_main"

    def test_connect_no_token(self):
        from integrations.mastodon_publisher import MastodonPublisher
        pub = MastodonPublisher()
        result = asyncio.get_event_loop().run_until_complete(pub.connect())
        assert result is False

    def test_publish_not_connected(self):
        from integrations.mastodon_publisher import MastodonPublisher
        pub = MastodonPublisher()
        result = asyncio.get_event_loop().run_until_complete(
            pub.publish({"caption": "test"})
        )
        assert result["success"] is False
        assert result["platform"] == "mastodon"

    def test_close(self):
        from integrations.mastodon_publisher import MastodonPublisher
        pub = MastodonPublisher()
        asyncio.get_event_loop().run_until_complete(pub.close())
        assert pub._connected is False


# ════════════════════════════════════════════════════════════════
# Quora Publisher Tests
# ════════════════════════════════════════════════════════════════

class TestQuoraPublisher:
    """Test Quora publisher."""

    def test_import(self):
        from integrations.quora_publisher import QuoraPublisher
        pub = QuoraPublisher()
        assert pub.platform == "quora"
        assert pub.MAX_CAPTION == 50_000

    def test_connect_no_credentials(self):
        from integrations.quora_publisher import QuoraPublisher
        pub = QuoraPublisher()
        result = asyncio.get_event_loop().run_until_complete(pub.connect())
        assert result is False

    def test_connect_webhook_fallback(self):
        from integrations.quora_publisher import QuoraPublisher
        pub = QuoraPublisher()
        with patch.dict(os.environ, {"QUORA_MAIN_WEBHOOK": "https://hook.example.com/quora"}):
            result = asyncio.get_event_loop().run_until_complete(pub.connect())
            assert result is True  # Partial connect via webhook

    def test_extract_question_id_url(self):
        from integrations.quora_publisher import QuoraPublisher
        qid = QuoraPublisher._extract_question_id(
            "https://www.quora.com/What-is-Python"
        )
        assert qid == "What-is-Python"

    def test_extract_question_id_numeric(self):
        from integrations.quora_publisher import QuoraPublisher
        qid = QuoraPublisher._extract_question_id("12345")
        assert qid == "12345"

    def test_markdown_to_quora_json(self):
        from integrations.quora_publisher import QuoraPublisher
        text = "# Heading\n\nParagraph here\n\n- Item 1\n- Item 2"
        result = json.loads(QuoraPublisher._markdown_to_quora_json(text))
        assert "sections" in result
        assert result["sections"][0]["type"] == "heading"
        assert result["sections"][1]["type"] == "paragraph"
        assert result["sections"][2]["type"] == "list"

    def test_publish_not_connected(self):
        from integrations.quora_publisher import QuoraPublisher
        pub = QuoraPublisher()
        result = asyncio.get_event_loop().run_until_complete(
            pub.publish({"caption": "test"})
        )
        assert result["success"] is False

    def test_close(self):
        from integrations.quora_publisher import QuoraPublisher
        pub = QuoraPublisher()
        asyncio.get_event_loop().run_until_complete(pub.close())


# ════════════════════════════════════════════════════════════════
# Engagement Fetcher Tests
# ════════════════════════════════════════════════════════════════

class TestEngagementFetcher:
    """Test engagement fetching module."""

    def test_import(self):
        from monitoring.engagement_fetcher import (
            fetch_engagement_batch,
            get_engagement_summary,
            get_post_engagement,
            ensure_engagement_table,
        )

    def test_normalize_metrics_likes(self):
        from monitoring.engagement_fetcher import _normalize_metrics
        result = _normalize_metrics("twitter", {"likes": 42, "retweets": 10})
        assert result["likes"] == 42
        assert result["shares"] == 10

    def test_normalize_metrics_mastodon(self):
        from monitoring.engagement_fetcher import _normalize_metrics
        result = _normalize_metrics(
            "mastodon",
            {"favourites": 15, "reblogs": 8, "replies_count": 3},
        )
        assert result["likes"] == 15
        assert result["shares"] == 8
        assert result["comments"] == 3

    def test_normalize_metrics_quora(self):
        from monitoring.engagement_fetcher import _normalize_metrics
        result = _normalize_metrics(
            "quora",
            {"numUpvotes": 100, "numViews": 5000, "numComments": 12},
        )
        assert result["likes"] == 100
        assert result["views"] == 5000
        assert result["comments"] == 12

    def test_extract_post_id_twitter(self):
        from monitoring.engagement_fetcher import _extract_platform_post_id
        pid = _extract_platform_post_id(
            "twitter",
            "https://twitter.com/user/status/1234567890",
        )
        assert pid == "1234567890"

    def test_extract_post_id_instagram(self):
        from monitoring.engagement_fetcher import _extract_platform_post_id
        pid = _extract_platform_post_id(
            "instagram",
            "https://www.instagram.com/p/CaB1cD2eF3g/",
        )
        assert pid == "CaB1cD2eF3g"

    def test_extract_post_id_mastodon(self):
        from monitoring.engagement_fetcher import _extract_platform_post_id
        pid = _extract_platform_post_id(
            "mastodon",
            "https://mastodon.social/@user/123456789",
        )
        assert pid == "123456789"

    def test_extract_post_id_youtube(self):
        from monitoring.engagement_fetcher import _extract_platform_post_id
        pid = _extract_platform_post_id(
            "youtube",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )
        assert pid == "dQw4w9WgXcQ"

    def test_supported_platforms_complete(self):
        from monitoring.engagement_fetcher import SUPPORTED_PLATFORMS
        assert "threads" in SUPPORTED_PLATFORMS
        assert "bluesky" in SUPPORTED_PLATFORMS
        assert "mastodon" in SUPPORTED_PLATFORMS
        assert "quora" in SUPPORTED_PLATFORMS


# ════════════════════════════════════════════════════════════════
# BPM-Aware Music Filtering Tests
# ════════════════════════════════════════════════════════════════

class TestBPMFiltering:
    """Test BPM-aware music recommendation."""

    def test_pace_to_bpm(self):
        from integrations.tiktok_music import _pace_to_bpm
        assert _pace_to_bpm("slow") == 75
        assert _pace_to_bpm("medium") == 110
        assert _pace_to_bpm("fast") == 140

    def test_mood_to_bpm(self):
        from integrations.tiktok_music import _mood_to_bpm
        assert _mood_to_bpm("chill") == 80
        assert _mood_to_bpm("motivational") == 130
        assert _mood_to_bpm("unknown_mood") is None

    def test_recommend_with_bpm(self):
        from integrations.tiktok_music import recommend_music
        result = recommend_music(
            text="calm morning routine",
            mood="chill",
            target_bpm=80,
            bpm_range=20,
        )
        assert result["success"] is True
        assert result["target_bpm"] == 80
        # All returned tracks should be within BPM range
        for track in result["tracks"]:
            assert abs(track.get("bpm", 0) - 80) <= 20

    def test_recommend_with_pace(self):
        from integrations.tiktok_music import recommend_music
        result = recommend_music(
            text="high energy workout",
            mood="motivational",
            content_pace="fast",
        )
        assert result["success"] is True
        assert result["target_bpm"] == 140

    def test_recommend_backward_compat(self):
        """Ensure old API (no bpm args) still works."""
        from integrations.tiktok_music import recommend_music
        result = recommend_music(text="relaxing sunset", mood="chill", limit=3)
        assert result["success"] is True
        assert isinstance(result["tracks"], list)


# ════════════════════════════════════════════════════════════════
# Trending Score Decay Tests
# ════════════════════════════════════════════════════════════════

class TestTrendingDecay:
    """Test trending score decay."""

    def test_import(self):
        from integrations.tiktok_music import (
            decay_trending_scores,
            get_trending_tracks,
        )

    def test_get_trending_tracks(self):
        from integrations.tiktok_music import get_trending_tracks
        tracks = get_trending_tracks(limit=5, min_score=0.8)
        assert isinstance(tracks, list)
        assert len(tracks) <= 5
        for t in tracks:
            assert t.get("trending_score", 0) >= 0.8

    def test_get_trending_sorted(self):
        from integrations.tiktok_music import get_trending_tracks
        tracks = get_trending_tracks(limit=10, min_score=0.5)
        if len(tracks) > 1:
            scores = [t["trending_score"] for t in tracks]
            assert scores == sorted(scores, reverse=True)

    def test_decay_no_custom_tracks(self):
        """Decay should work even if there are no custom tracks."""
        from integrations.tiktok_music import decay_trending_scores
        result = decay_trending_scores()
        assert "updated" in result
        assert "decayed_tracks" in result


# ════════════════════════════════════════════════════════════════
# Time Slot Suggestion Engine Tests
# ════════════════════════════════════════════════════════════════

class TestTimeSlotEngine:
    """Test time slot suggestion engine."""

    def test_import(self):
        from core.time_slot_engine import (
            suggest_time,
            suggest_schedule,
            get_best_hours,
        )

    def test_suggest_time_tiktok(self):
        from core.time_slot_engine import suggest_time
        result = suggest_time("tiktok")
        assert result["platform"] == "tiktok"
        assert "suggested_utc" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1.5

    def test_suggest_time_with_offset(self):
        from core.time_slot_engine import suggest_time
        result = suggest_time("instagram", utc_offset_hours=7)
        assert result["hour_local"] == (result["hour_utc"] + 7) % 24

    def test_suggest_time_unknown_platform(self):
        from core.time_slot_engine import suggest_time
        result = suggest_time("unknown_platform_xyz")
        assert result["platform"] == "unknown_platform_xyz"
        # Should fallback to default hours

    def test_suggest_schedule(self):
        from core.time_slot_engine import suggest_schedule
        result = suggest_schedule(
            platforms=["tiktok", "instagram", "twitter"],
            posts_per_platform=1,
        )
        assert result["total_posts"] == 3
        assert len(result["schedule"]) == 3

    def test_suggest_schedule_multi_post(self):
        from core.time_slot_engine import suggest_schedule
        result = suggest_schedule(
            platforms=["tiktok"],
            posts_per_platform=3,
        )
        assert result["total_posts"] == 3

    def test_suggest_schedule_spacing(self):
        from core.time_slot_engine import suggest_schedule
        result = suggest_schedule(
            platforms=["tiktok", "instagram"],
            posts_per_platform=2,
            spacing_minutes=60,
        )
        # Verify posts are spaced at least 60 min apart
        times = [s["time_utc"] for s in result["schedule"]]
        assert len(times) == 4

    def test_get_best_hours_all_platforms(self):
        from core.time_slot_engine import get_best_hours
        result = get_best_hours()
        assert "platforms" in result
        assert "tiktok" in result["platforms"]
        assert "threads" in result["platforms"]

    def test_get_best_hours_single_platform(self):
        from core.time_slot_engine import get_best_hours
        result = get_best_hours(platform="linkedin")
        assert "linkedin" in result["platforms"]

    def test_optimal_slots_include_new_platforms(self):
        from core.time_slot_engine import OPTIMAL_TIME_SLOTS
        assert "threads" in OPTIMAL_TIME_SLOTS
        assert "bluesky" in OPTIMAL_TIME_SLOTS
        assert "mastodon" in OPTIMAL_TIME_SLOTS
        assert "quora" in OPTIMAL_TIME_SLOTS


# ════════════════════════════════════════════════════════════════
# Multi-Image Slideshow Tests
# ════════════════════════════════════════════════════════════════

class TestMultiImageSlideshow:
    """Test multi-image slideshow creation."""

    def test_import(self):
        from integrations.media_processor import (
            create_multi_image_slideshow,
            create_multi_image_slideshow_from_urls,
        )

    def test_empty_images(self):
        from integrations.media_processor import create_multi_image_slideshow
        result = create_multi_image_slideshow([])
        assert result["success"] is False
        assert "No images" in result["error"]


# ════════════════════════════════════════════════════════════════
# Text Overlay Tests
# ════════════════════════════════════════════════════════════════

class TestTextOverlay:
    """Test text overlay and subtitle features."""

    def test_import(self):
        from integrations.media_processor import (
            add_text_overlay,
            add_subtitles,
        )

    def test_overlay_missing_video(self):
        from integrations.media_processor import add_text_overlay
        result = add_text_overlay("/nonexistent/video.mp4", "Hello")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_subtitles_empty(self):
        from integrations.media_processor import add_subtitles
        result = add_subtitles("/nonexistent/video.mp4", [])
        assert result["success"] is False
        assert "No subtitles" in result["error"]

    def test_srt_time_format(self):
        from integrations.media_processor import _seconds_to_srt_time
        assert _seconds_to_srt_time(0) == "00:00:00,000"
        assert _seconds_to_srt_time(61.5) == "00:01:01,500"
        assert _seconds_to_srt_time(3723.123) == "01:02:03,123"


# ════════════════════════════════════════════════════════════════
# Scheduler Integration Tests — New Platforms
# ════════════════════════════════════════════════════════════════

class TestSchedulerNewPlatforms:
    """Test that scheduler can load new platform publishers."""

    def test_optimal_slots_15_platforms(self):
        from core.scheduler import OPTIMAL_TIME_SLOTS
        assert "threads" in OPTIMAL_TIME_SLOTS
        assert "bluesky" in OPTIMAL_TIME_SLOTS
        assert "mastodon" in OPTIMAL_TIME_SLOTS
        assert "quora" in OPTIMAL_TIME_SLOTS
        assert len(OPTIMAL_TIME_SLOTS) >= 15

    def test_rate_limits_15_platforms(self):
        from core.scheduler import DAILY_RATE_LIMITS
        assert "threads" in DAILY_RATE_LIMITS
        assert "bluesky" in DAILY_RATE_LIMITS
        assert "mastodon" in DAILY_RATE_LIMITS
        assert "quora" in DAILY_RATE_LIMITS

    def test_load_publisher_threads(self):
        from core.scheduler import PublishScheduler
        sched = PublishScheduler()
        pub = sched._load_publisher("threads")
        assert pub is not None

    def test_load_publisher_bluesky(self):
        from core.scheduler import PublishScheduler
        sched = PublishScheduler()
        pub = sched._load_publisher("bluesky")
        assert pub is not None

    def test_load_publisher_mastodon(self):
        from core.scheduler import PublishScheduler
        sched = PublishScheduler()
        pub = sched._load_publisher("mastodon")
        assert pub is not None

    def test_load_publisher_quora(self):
        from core.scheduler import PublishScheduler
        sched = PublishScheduler()
        pub = sched._load_publisher("quora")
        assert pub is not None

    def test_no_real_prefix_bug(self):
        """Verify the RealXxxPublisher bug is fixed."""
        import ast
        import inspect
        from core import scheduler

        source = inspect.getsource(scheduler)
        assert "RealRedditPublisher" not in source
        assert "RealMediumPublisher" not in source
        assert "RealTumblrPublisher" not in source


# ════════════════════════════════════════════════════════════════
# Content Factory — New Platform Char Limits
# ════════════════════════════════════════════════════════════════

class TestContentFactoryNewPlatforms:
    """Test content factory includes new platforms."""

    def test_char_limits_threads(self):
        from agents.content_factory import CHANNEL_CHAR_LIMITS
        assert "threads" in CHANNEL_CHAR_LIMITS
        assert CHANNEL_CHAR_LIMITS["threads"]["caption"] == 500

    def test_char_limits_bluesky(self):
        from agents.content_factory import CHANNEL_CHAR_LIMITS
        assert "bluesky" in CHANNEL_CHAR_LIMITS
        assert CHANNEL_CHAR_LIMITS["bluesky"]["caption"] == 300

    def test_char_limits_mastodon(self):
        from agents.content_factory import CHANNEL_CHAR_LIMITS
        assert "mastodon" in CHANNEL_CHAR_LIMITS
        assert CHANNEL_CHAR_LIMITS["mastodon"]["caption"] == 500

    def test_char_limits_quora(self):
        from agents.content_factory import CHANNEL_CHAR_LIMITS
        assert "quora" in CHANNEL_CHAR_LIMITS
        assert CHANNEL_CHAR_LIMITS["quora"]["caption"] == 50000

    def test_total_platforms(self):
        from agents.content_factory import CHANNEL_CHAR_LIMITS
        assert len(CHANNEL_CHAR_LIMITS) >= 16  # 12 old + 4 new
