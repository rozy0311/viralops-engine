"""Tests for Content Factory agent."""

import pytest
from agents.content_factory import (
    content_factory_node,
    _smart_truncate,
    _adapt_for_platform,
)
from core.models import ContentPack, ContentType, HashtagMatrix


class TestContentFactory:

    def test_content_factory_generates_pack(self):
        state = {
            "niche_id": "raw-almonds",
            "niche_name": "Raw Almonds",
            "tone": "conversational",
            "audience": "health-conscious millennials",
            "brand_tags": [],
            "target_platforms": ["tiktok", "instagram_reels"],
            "past_performance": {},
        }
        result = content_factory_node(state)

        assert "content_pack" in result
        pack = result["content_pack"]
        assert isinstance(pack, ContentPack)
        assert pack.niche_id == "raw-almonds"
        assert pack.content_hash
        assert pack.transform_score == 1.0
        assert len(pack.platform_contents) == 2

    def test_content_factory_no_platforms(self):
        state = {
            "niche_id": "chickpeas",
            "tone": "casual",
            "audience": "students",
            "target_platforms": [],
        }
        result = content_factory_node(state)
        pack = result["content_pack"]
        assert len(pack.platform_contents) == 0

    def test_smart_truncate_short_text(self):
        assert _smart_truncate("Hello world", 100) == "Hello world"

    def test_smart_truncate_at_sentence(self):
        text = "First sentence. Second sentence. Third sentence is very long."
        result = _smart_truncate(text, 35)
        assert result.endswith(".")

    def test_smart_truncate_at_word(self):
        text = "No periods here just words that keep going on and on"
        result = _smart_truncate(text, 30)
        assert result.endswith("…") or result.endswith(" ")

    def test_platform_adaptation_tiktok(self):
        pack = ContentPack(
            id="test-1",
            niche_id="test",
            title="Test Title For TikTok Content",
            long_content="Long content here...",
            universal_caption="Short caption for testing.",
            hashtag_matrix=HashtagMatrix(
                niche=["#test"], trending=["#trending"],
                community=["#community"], location_seasonal=["#usa"],
                viral_hook=["#viral"],
            ),
            content_type=ContentType.ORIGINAL,
            content_hash="abc123",
        )
        pc = _adapt_for_platform(pack, "tiktok")
        assert pc.platform == "tiktok"
        assert len(pc.hashtags) <= 5
        assert pc.video_ratio == "9:16"

    def test_platform_adaptation_twitter(self):
        pack = ContentPack(
            id="test-2",
            niche_id="test",
            title="Test",
            long_content="...",
            universal_caption="A" * 300,  # Over Twitter limit
            hashtag_matrix=HashtagMatrix(
                niche=["#a"], trending=["#b"],
                community=["#c"], location_seasonal=["#d"],
                viral_hook=["#e"],
            ),
            content_type=ContentType.ORIGINAL,
            content_hash="def456",
        )
        pc = _adapt_for_platform(pack, "twitter_x")
        assert len(pc.hashtags) <= 2
        assert pc.is_truncated
