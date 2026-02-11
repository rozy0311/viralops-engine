"""Tests for Content Factory agent."""

import pytest
from agents.content_factory import (
    generate_content_pack,
    smart_truncate,
    adapt_for_platform,
)


class TestContentFactory:

    def test_content_factory_generates_pack(self):
        """generate_content_pack returns state with content_pack dict (fallback mode, no API key)."""
        state = {
            "niche_config": {"id": "raw-almonds", "display_name": "Raw Almonds"},
            "niche_key": "raw-almonds",
            "target_platform": "tiktok",
        }
        result = generate_content_pack(state)

        assert "content_pack" in result
        pack = result["content_pack"]
        assert isinstance(pack, dict)
        # Fallback always sets title + body
        assert "title" in pack or "_generated_by" in pack

    def test_content_factory_fallback_status(self):
        """Without OpenAI key, generate_content_pack falls back gracefully."""
        state = {
            "niche_config": {"id": "chickpeas", "name": "Chickpeas"},
            "niche_key": "chickpeas",
            "target_platform": "instagram",
        }
        result = generate_content_pack(state)
        assert "content_factory_status" in result
        assert "fallback" in result["content_factory_status"]

    def test_smart_truncate_short_text(self):
        assert smart_truncate("Hello world", 100) == "Hello world"

    def test_smart_truncate_at_sentence(self):
        text = "First sentence. Second sentence. Third sentence is very long."
        result = smart_truncate(text, 35)
        assert result.endswith(".")

    def test_smart_truncate_at_word(self):
        text = "No periods here just words that keep going on and on"
        result = smart_truncate(text, 30)
        assert len(result) <= 33

    def test_platform_adaptation_tiktok(self):
        pack = {
            "title": "Test Title For TikTok Content",
            "body": "Long content here for testing purposes.",
            "hook": "Short caption for testing.",
            "hashtags": ["#test", "#niche", "#micro", "#content", "#tips"],
        }
        result = adapt_for_platform(pack, "tiktok")
        assert result["platform"] == "tiktok"
        assert result["hashtag_count"] <= 5

    def test_platform_adaptation_twitter(self):
        pack = {
            "title": "Test",
            "body": "A" * 300,
            "hook": "A" * 300,
            "hashtags": ["#a", "#b", "#c"],
        }
        result = adapt_for_platform(pack, "twitter")
        assert result["hashtag_count"] <= 3
        assert result["char_count"] <= 280
