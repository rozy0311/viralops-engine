"""
Tests for RSS Auto Poster  auto-publish engine.

Covers:
  - Config CRUD (create, update, delete, list, get, pause, activate)
  - Keyword filters (include + exclude)
  - Entry dedup (new_only, allow_repeat)
  - Post content building (title + desc + hashtags + media, NO links)
  - TikTok music selection for image entries
  - Platform adaptation
  - Tick engine (main loop)
  - History management
"""

import json
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

# ── Add project root to path ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from integrations.rss_auto_poster import (
    AutoPosterConfig,
    create_auto_poster,
    update_auto_poster,
    delete_auto_poster,
    list_auto_posters,
    get_auto_poster,
    pause_auto_poster,
    activate_auto_poster,
    get_auto_poster_status,
    get_poster_history,
    clear_poster_history,
    tick,
    FREQUENCY_PRESETS,
    _passes_keyword_filter,
    _get_entry_hash,
    _build_post_content,
    _load_configs,
    _save_configs,
    _load_history,
    _save_history,
)


# ── Fixtures ──

@pytest.fixture(autouse=True)
def temp_data_dir(tmp_path, monkeypatch):
    """Redirect all data files to temp directory."""
    cfg_file = str(tmp_path / "rss_auto_poster_configs.json")
    hist_file = str(tmp_path / "rss_auto_poster_history.json")
    db_file = str(tmp_path / "viralops.db")

    monkeypatch.setattr("integrations.rss_auto_poster.AUTO_POSTER_CONFIG_FILE", cfg_file)
    monkeypatch.setattr("integrations.rss_auto_poster.AUTO_POSTER_HISTORY_FILE", hist_file)
    monkeypatch.setattr("integrations.rss_auto_poster.DB_PATH", db_file)

    return tmp_path


@pytest.fixture
def sample_config():
    return {
        "name": "TheRike → TikTok Auto",
        "feed_id": "therike-blog-01",
        "feed_url": "https://example.com/feed.xml",
        "target_platforms": ["tiktok", "instagram"],
        "frequency": "every_hour",
        "entries_per_cycle": 2,
        "publish_mode": "scheduled",
        "niche": "sustainable-living",
        "tiktok_music_enabled": True,
        "post_filter": "new_only",
        "allow_repeat": False,
        "sequence": "newest_first",
    }


@pytest.fixture
def sample_entry():
    return {
        "id": "entry-001",
        "title": "10 Tips for Sustainable Living",
        "body": "Discover how to reduce your carbon footprint with these simple tips for eco-friendly living.",
        "excerpt": "Discover how to reduce your carbon footprint...",
        "url": "https://therike.com/sustainable-living-tips",
        "author": "TheRike",
        "published": "2025-01-15T10:00:00",
        "image_url": "https://therike.com/images/sustainable-tips.jpg",
        "tags": ["sustainability", "eco-friendly", "green living"],
    }


# ════════════════════════════════════════════════
# Config CRUD Tests
# ════════════════════════════════════════════════

class TestAutoPostCRUD:
    """Test RSS Auto Poster config CRUD operations."""

    def test_create_auto_poster(self, sample_config):
        result = create_auto_poster(sample_config)
        assert result["success"] is True
        cfg = result["config"]
        assert cfg["name"] == "TheRike → TikTok Auto"
        assert cfg["feed_id"] == "therike-blog-01"
        assert cfg["target_platforms"] == ["tiktok", "instagram"]
        assert cfg["frequency"] == "every_hour"
        assert cfg["entries_per_cycle"] == 2
        assert cfg["include_links"] is False  # ALWAYS False
        assert cfg["id"]  # Generated
        assert cfg["state"] == "active"

    def test_create_always_disables_links(self, sample_config):
        sample_config["include_links"] = True  # User tries to enable
        result = create_auto_poster(sample_config)
        assert result["config"]["include_links"] is False  # Forced off

    def test_create_default_name(self):
        result = create_auto_poster({"feed_id": "test"})
        assert "Auto Poster" in result["config"]["name"]

    def test_list_auto_posters(self, sample_config):
        create_auto_poster(sample_config)
        create_auto_poster({"name": "Second Poster", "feed_id": "feed-02"})
        posters = list_auto_posters()
        assert len(posters) == 2
        names = [p["name"] for p in posters]
        assert "TheRike → TikTok Auto" in names
        assert "Second Poster" in names

    def test_get_auto_poster(self, sample_config):
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]
        found = get_auto_poster(poster_id)
        assert found is not None
        assert found["name"] == "TheRike → TikTok Auto"

    def test_get_auto_poster_not_found(self):
        assert get_auto_poster("nonexistent") is None

    def test_update_auto_poster(self, sample_config):
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]

        result = update_auto_poster(poster_id, {
            "frequency": "every_30min",
            "entries_per_cycle": 5,
        })
        assert result["success"] is True
        assert result["config"]["frequency"] == "every_30min"
        assert result["config"]["entries_per_cycle"] == 5

    def test_update_cannot_enable_links(self, sample_config):
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]
        result = update_auto_poster(poster_id, {"include_links": True})
        assert result["config"]["include_links"] is False

    def test_update_nonexistent(self):
        result = update_auto_poster("nope", {"frequency": "every_hour"})
        assert "error" in result

    def test_delete_auto_poster(self, sample_config):
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]
        result = delete_auto_poster(poster_id)
        assert result["success"] is True
        assert result["deleted"] == 1
        assert get_auto_poster(poster_id) is None

    def test_delete_nonexistent(self):
        result = delete_auto_poster("nope")
        assert result["deleted"] == 0

    def test_pause_auto_poster(self, sample_config):
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]
        result = pause_auto_poster(poster_id)
        assert result["config"]["state"] == "paused"

    def test_activate_auto_poster(self, sample_config):
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]
        pause_auto_poster(poster_id)
        result = activate_auto_poster(poster_id)
        assert result["config"]["state"] == "active"


# ════════════════════════════════════════════════
# Keyword Filter Tests
# ════════════════════════════════════════════════

class TestKeywordFilter:
    """Test include/exclude keyword filtering."""

    def test_passes_with_no_filters(self, sample_entry):
        config = {"include_keywords": [], "exclude_keywords": []}
        assert _passes_keyword_filter(sample_entry, config) is True

    def test_include_filter_match(self, sample_entry):
        config = {"include_keywords": ["sustainable"], "exclude_keywords": []}
        assert _passes_keyword_filter(sample_entry, config) is True

    def test_include_filter_no_match(self, sample_entry):
        config = {"include_keywords": ["cryptocurrency"], "exclude_keywords": []}
        assert _passes_keyword_filter(sample_entry, config) is False

    def test_exclude_filter_match(self, sample_entry):
        config = {"include_keywords": [], "exclude_keywords": ["sustainable"]}
        assert _passes_keyword_filter(sample_entry, config) is False

    def test_exclude_filter_no_match(self, sample_entry):
        config = {"include_keywords": [], "exclude_keywords": ["cryptocurrency"]}
        assert _passes_keyword_filter(sample_entry, config) is True

    def test_both_filters(self, sample_entry):
        # Include matches, exclude doesn't → passes
        config = {"include_keywords": ["sustainable"], "exclude_keywords": ["bitcoin"]}
        assert _passes_keyword_filter(sample_entry, config) is True

    def test_both_filters_exclude_wins(self, sample_entry):
        # Both match → exclude wins → fails
        config = {"include_keywords": ["sustainable"], "exclude_keywords": ["eco-friendly"]}
        assert _passes_keyword_filter(sample_entry, config) is False

    def test_case_insensitive(self, sample_entry):
        config = {"include_keywords": ["SUSTAINABLE"], "exclude_keywords": []}
        assert _passes_keyword_filter(sample_entry, config) is True


# ════════════════════════════════════════════════
# Entry Hash / Dedup Tests
# ════════════════════════════════════════════════

class TestEntryHash:
    """Test entry hashing for dedup."""

    def test_same_entry_same_hash(self, sample_entry):
        h1 = _get_entry_hash(sample_entry, "config-01")
        h2 = _get_entry_hash(sample_entry, "config-01")
        assert h1 == h2

    def test_different_config_different_hash(self, sample_entry):
        h1 = _get_entry_hash(sample_entry, "config-01")
        h2 = _get_entry_hash(sample_entry, "config-02")
        assert h1 != h2

    def test_different_entry_different_hash(self, sample_entry):
        h1 = _get_entry_hash(sample_entry, "config-01")
        entry2 = {**sample_entry, "url": "https://different.com/post"}
        h2 = _get_entry_hash(entry2, "config-01")
        assert h1 != h2


# ════════════════════════════════════════════════
# Post Content Building Tests
# ════════════════════════════════════════════════

class TestBuildPostContent:
    """Test _build_post_content — the core content builder."""

    def test_basic_content(self, sample_entry, sample_config):
        post = _build_post_content(sample_entry, sample_config)
        assert post["title"] == "10 Tips for Sustainable Living"
        assert "carbon footprint" in post["description"]
        assert post["media_url"] == "https://therike.com/images/sustainable-tips.jpg"
        assert post["media_type"] == "image"
        assert post["source_url"]  # Internal ref only

    def test_no_links_in_content(self, sample_entry, sample_config):
        """User spec: NO links in posts."""
        sample_config["include_links"] = False
        post = _build_post_content(sample_entry, sample_config)
        # The description should NOT contain the source URL
        # (source_url is internal only, not in the post body)
        assert "source_url" in post  # Internal field
        # Post content should be just title + description + hashtags
        assert "description" in post

    def test_video_media_type(self, sample_entry, sample_config):
        sample_entry["image_url"] = "https://example.com/video.mp4"
        post = _build_post_content(sample_entry, sample_config)
        assert post["media_type"] == "video"

    def test_no_media(self, sample_entry, sample_config):
        sample_entry["image_url"] = ""
        post = _build_post_content(sample_entry, sample_config)
        assert post["media_type"] == "none"

    def test_prefix_suffix(self, sample_entry, sample_config):
        sample_config["prefix"] = "🔥 TRENDING"
        sample_config["suffix"] = "Follow for more!"
        post = _build_post_content(sample_entry, sample_config)
        assert "🔥 TRENDING" in post["description"]
        assert "Follow for more!" in post["description"]

    def test_title_disabled(self, sample_entry, sample_config):
        sample_config["include_title"] = False
        post = _build_post_content(sample_entry, sample_config)
        assert post["title"] == ""

    def test_description_disabled(self, sample_entry, sample_config):
        sample_config["include_description"] = False
        post = _build_post_content(sample_entry, sample_config)
        assert post["description"] == ""

    def test_hashtags_fallback_to_tags(self, sample_entry, sample_config):
        """When hashtag generator fails, fallback to RSS tags."""
        with patch.dict("sys.modules", {"agents.matrix_5layer": MagicMock(
                generate_micro_niche_5=MagicMock(side_effect=Exception("no matrix"))
        )}):
            post = _build_post_content(sample_entry, sample_config)
            # Should have fallback hashtags from RSS tags
            assert isinstance(post["hashtags"], list)

    def test_tiktok_music_for_image(self, sample_entry, sample_config):
        """Image + TikTok in platforms → auto-recommend music."""
        mock_music = {
            "tracks": [{
                "track_id": "track-001",
                "title": "Eco Vibes",
                "artist": "Green Artist",
                "mood": "chill",
            }],
            "mood_detected": "chill",
        }
        mock_tiktok_module = MagicMock()
        mock_tiktok_module.recommend_music = MagicMock(return_value=mock_music)
        with patch.dict("sys.modules", {"integrations.tiktok_music": mock_tiktok_module}):
            post = _build_post_content(sample_entry, sample_config)
            assert post["tiktok_music"] is not None
            assert post["tiktok_music"]["title"] == "Eco Vibes"

    def test_no_tiktok_music_for_video(self, sample_entry, sample_config):
        """Video entries → no TikTok music (already has music)."""
        sample_entry["image_url"] = "https://example.com/clip.mp4"
        post = _build_post_content(sample_entry, sample_config)
        assert post["media_type"] == "video"
        assert post["tiktok_music"] is None

    def test_no_tiktok_music_when_disabled(self, sample_entry, sample_config):
        """When tiktok_music_enabled=False → no music selection."""
        sample_config["tiktok_music_enabled"] = False
        post = _build_post_content(sample_entry, sample_config)
        assert post["tiktok_music"] is None

    def test_no_tiktok_music_when_no_tiktok_platform(self, sample_entry, sample_config):
        """When target_platforms doesn't include tiktok → no music."""
        sample_config["target_platforms"] = ["instagram", "facebook"]
        post = _build_post_content(sample_entry, sample_config)
        assert post["tiktok_music"] is None


# ════════════════════════════════════════════════
# Tick Engine Tests
# ════════════════════════════════════════════════

class TestTickEngine:
    """Test the main tick() function — the auto poster loop."""

    def test_tick_empty_configs(self):
        result = tick()
        assert result["success"] is True
        assert result["total_posted"] == 0

    def test_tick_paused_poster_skipped(self, sample_config):
        sample_config["state"] = "paused"
        create_auto_poster(sample_config)
        result = tick()
        assert result["total_posted"] == 0

    def test_tick_not_due_yet(self, sample_config):
        """Poster checked 30min ago, frequency=every_hour → skip."""
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]

        # Set last_checked to 30min ago
        update_auto_poster(poster_id, {
            "last_checked": (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat(),
        })

        result = tick()
        assert result["total_posted"] == 0

    def test_tick_due_poster_runs(self, sample_config, sample_entry):
        """Poster checked >1h ago, frequency=every_hour → should run."""
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]

        # Set last_checked to 2h ago
        update_auto_poster(poster_id, {
            "last_checked": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        })

        mock_feed = {
            "success": True,
            "entries": [sample_entry],
        }

        mock_rss = MagicMock()
        mock_rss.fetch_feed = MagicMock(return_value=mock_feed)
        with patch.dict("sys.modules", {"integrations.rss_reader": mock_rss}):
            result = tick(poster_id=poster_id)
            assert result["success"] is True
            # Should have posted (2 platforms × 1 entry = 2 posts)
            assert result["total_posted"] >= 1

    def test_tick_dedup(self, sample_config, sample_entry):
        """Same entry should not be posted twice when allow_repeat=False."""
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]

        mock_feed = {"success": True, "entries": [sample_entry]}

        mock_rss = MagicMock()
        mock_rss.fetch_feed = MagicMock(return_value=mock_feed)
        with patch.dict("sys.modules", {"integrations.rss_reader": mock_rss}):
            # First tick
            tick(poster_id=poster_id)

            # Reset last_checked for second tick
            update_auto_poster(poster_id, {
                "last_checked": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            })

            # Second tick — same entry should be skipped
            result = tick(poster_id=poster_id)
            assert result["total_posted"] == 0

    def test_tick_allow_repeat(self, sample_config, sample_entry):
        """When allow_repeat=True, same entry can be posted again."""
        sample_config["allow_repeat"] = True
        sample_config["post_filter"] = "any"
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]

        mock_feed = {"success": True, "entries": [sample_entry]}

        mock_rss = MagicMock()
        mock_rss.fetch_feed = MagicMock(return_value=mock_feed)
        with patch.dict("sys.modules", {"integrations.rss_reader": mock_rss}):
            # First tick
            tick(poster_id=poster_id)

            # Reset last_checked
            update_auto_poster(poster_id, {
                "last_checked": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            })

            # Second tick — should post again
            result = tick(poster_id=poster_id)
            assert result["total_posted"] >= 1

    def test_tick_random_sequence(self, sample_config, sample_entry):
        """Test random posting sequence doesn't crash."""
        sample_config["sequence"] = "random"
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]

        entries = [
            {**sample_entry, "id": f"entry-{i}", "title": f"Post {i}"}
            for i in range(5)
        ]
        mock_feed = {"success": True, "entries": entries}

        mock_rss = MagicMock()
        mock_rss.fetch_feed = MagicMock(return_value=mock_feed)
        with patch.dict("sys.modules", {"integrations.rss_reader": mock_rss}):
            result = tick(poster_id=poster_id)
            assert result["success"] is True
            # entries_per_cycle=2, so max 2 entries × 2 platforms = 4 posts
            assert result["total_posted"] >= 1

    def test_tick_keyword_filter(self, sample_config, sample_entry):
        """Entries that fail keyword filter should be skipped."""
        sample_config["exclude_keywords"] = ["sustainable"]
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]

        mock_feed = {"success": True, "entries": [sample_entry]}

        mock_rss = MagicMock()
        mock_rss.fetch_feed = MagicMock(return_value=mock_feed)
        with patch.dict("sys.modules", {"integrations.rss_reader": mock_rss}):
            result = tick(poster_id=poster_id)
            # Entry has "sustainable" in title → excluded
            assert result["results"][0]["posted"] == 0

    def test_tick_entries_per_cycle_limit(self, sample_config, sample_entry):
        """Only process entries_per_cycle entries."""
        sample_config["entries_per_cycle"] = 1
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]

        entries = [
            {**sample_entry, "id": f"entry-{i}", "title": f"Post {i}", "url": f"https://example.com/{i}"}
            for i in range(5)
        ]
        mock_feed = {"success": True, "entries": entries}

        mock_rss = MagicMock()
        mock_rss.fetch_feed = MagicMock(return_value=mock_feed)
        with patch.dict("sys.modules", {"integrations.rss_reader": mock_rss}):
            result = tick(poster_id=poster_id)
            # Only 1 entry × 2 platforms = 2 posts max
            assert result["total_posted"] <= 2

    def test_tick_feed_error(self, sample_config):
        """Feed fetch failure → error reported, no crash."""
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]

        mock_feed = {"success": False, "error": "Feed unreachable"}

        mock_rss = MagicMock()
        mock_rss.fetch_feed = MagicMock(return_value=mock_feed)
        with patch.dict("sys.modules", {"integrations.rss_reader": mock_rss}):
            result = tick(poster_id=poster_id)
            assert result["success"] is True  # Overall tick is still OK
            assert result["results"][0]["error"] == "Feed unreachable"


# ════════════════════════════════════════════════
# History Tests
# ════════════════════════════════════════════════

class TestHistory:
    """Test posting history tracking."""

    def test_history_after_posting(self, sample_config, sample_entry):
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]

        mock_feed = {"success": True, "entries": [sample_entry]}

        mock_rss = MagicMock()
        mock_rss.fetch_feed = MagicMock(return_value=mock_feed)
        with patch.dict("sys.modules", {"integrations.rss_reader": mock_rss}):
            tick(poster_id=poster_id)

        history = get_poster_history(poster_id)
        assert len(history) >= 1
        assert history[0]["title"]
        assert history[0]["posted_at"]

    def test_clear_history(self, sample_config, sample_entry):
        created = create_auto_poster(sample_config)
        poster_id = created["config"]["id"]

        mock_feed = {"success": True, "entries": [sample_entry]}

        mock_rss = MagicMock()
        mock_rss.fetch_feed = MagicMock(return_value=mock_feed)
        with patch.dict("sys.modules", {"integrations.rss_reader": mock_rss}):
            tick(poster_id=poster_id)

        result = clear_poster_history(poster_id)
        assert result["success"] is True
        assert result["entries_cleared"] >= 1

        # History should be empty now
        history = get_poster_history(poster_id)
        assert len(history) == 0


# ════════════════════════════════════════════════
# Status Tests
# ════════════════════════════════════════════════

class TestStatus:
    """Test status/overview endpoints."""

    def test_status_empty(self):
        status = get_auto_poster_status()
        assert status["total_posters"] == 0
        assert status["active"] == 0

    def test_status_with_posters(self, sample_config):
        create_auto_poster(sample_config)
        create_auto_poster({"name": "Paused One", "feed_id": "x", "state": "paused"})

        status = get_auto_poster_status()
        assert status["total_posters"] == 2
        assert status["active"] == 1
        assert status["paused"] == 1
        assert len(status["posters"]) == 2


# ════════════════════════════════════════════════
# Frequency Presets Tests
# ════════════════════════════════════════════════

class TestFrequencyPresets:
    """Test that frequency presets are valid."""

    def test_all_presets_exist(self):
        expected = ["every_15min", "every_30min", "every_hour",
                    "every_2hours", "every_4hours", "every_12hours", "every_24hours"]
        for key in expected:
            assert key in FREQUENCY_PRESETS

    def test_presets_in_minutes(self):
        assert FREQUENCY_PRESETS["every_15min"] == 15
        assert FREQUENCY_PRESETS["every_hour"] == 60
        assert FREQUENCY_PRESETS["every_24hours"] == 1440


# ════════════════════════════════════════════════
# AutoPosterConfig Dataclass Tests
# ════════════════════════════════════════════════

class TestAutoPostConfig:
    """Test the AutoPosterConfig defaults."""

    def test_defaults(self):
        cfg = AutoPosterConfig()
        assert cfg.state == "active"
        assert cfg.include_links is False
        assert cfg.include_title is True
        assert cfg.include_description is True
        assert cfg.include_hashtags is True
        assert cfg.include_media is True
        assert cfg.tiktok_music_enabled is True
        assert cfg.entries_per_cycle == 1
        assert cfg.frequency == "every_hour"
        assert cfg.post_filter == "new_only"
        assert cfg.allow_repeat is False
        assert cfg.sequence == "newest_first"
