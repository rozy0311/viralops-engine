"""Tests for Content Factory agent."""

import pytest
from agents.content_factory import (
    generate_content_pack,
    smart_truncate,
    adapt_for_platform,
    extract_relevant_answer,
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


class TestExtractRelevantAnswer:
    """Test extract_relevant_answer handles both str and list inputs gracefully."""

    def test_string_passthrough(self):
        assert extract_relevant_answer("Hello world") == "Hello world"

    def test_empty_string(self):
        assert extract_relevant_answer("") == ""

    def test_none_passthrough(self):
        assert extract_relevant_answer(None) is None

    def test_strips_preamble(self):
        result = extract_relevant_answer("Sure! Here's the answer.\n\nActual content here.")
        assert "Sure" not in result
        assert "Actual content here" in result


class TestAdaptForPlatformLLMContent:
    """Test that adapt_for_platform uses LLM-generated fields cleanly (no template garbage)."""

    @pytest.fixture
    def llm_content_pack(self):
        """Simulates a real Gemini-generated content_pack."""
        return {
            "title": "Eco-Home Transformation: Your Ultimate Sustainable Guide",
            "body": "Full article body with many paragraphs...",
            "hook": "Your home is silently draining your wallet AND the planet.",
            "pain_point": "Many feel overwhelmed by sustainable living, thinking it's expensive.",
            "solution_steps": [
                "Conduct a home eco-audit",
                "Implement energy-saving hacks",
                "Master waste reduction",
            ],
            "step_1": "Conduct a home eco-audit",
            "step_2": "Implement energy-saving hacks",
            "result": "30% reduction in utility bills",
            "cta": "Save this guide and start your eco-journey today!",
            "micro_keywords": "sustainable living • eco-friendly home • green tips",
            "hashtags": ["#EcoHome", "#SustainableLiving", "#GreenLiving", "#ZeroWaste", "#EcoFriendly"],
            "_generated_by": "gemini/gemini-2.5-flash",
        }

    def test_tiktok_no_hardcoded_location(self, llm_content_pack):
        """TikTok caption must NOT contain [Chicago], [Winter], or other hardcoded template text."""
        result = adapt_for_platform(llm_content_pack, "tiktok")
        caption = result["caption"]
        assert "[Chicago]" not in caption
        assert "[Winter]" not in caption
        assert "Busy people?" not in caption
        assert "Beginners?" not in caption
        assert "Health seekers?" not in caption

    def test_tiktok_uses_hook(self, llm_content_pack):
        """TikTok caption should start with the LLM-generated hook."""
        result = adapt_for_platform(llm_content_pack, "tiktok")
        assert result["caption"].startswith("Your home is silently")

    def test_tiktok_includes_pain_and_cta(self, llm_content_pack):
        """TikTok caption should include pain_point and CTA."""
        result = adapt_for_platform(llm_content_pack, "tiktok")
        caption = result["caption"]
        assert "overwhelmed" in caption
        assert "eco-journey" in caption or "Save" in caption

    def test_tiktok_includes_hashtags(self, llm_content_pack):
        """TikTok caption should include hashtags."""
        result = adapt_for_platform(llm_content_pack, "tiktok")
        assert "#EcoHome" in result["caption"]

    def test_tiktok_handles_list_solution_steps(self, llm_content_pack):
        """TikTok should handle solution_steps as a list (Gemini returns arrays)."""
        result = adapt_for_platform(llm_content_pack, "tiktok")
        caption = result["caption"]
        assert "eco-audit" in caption or "energy-saving" in caption

    def test_instagram_same_as_tiktok(self, llm_content_pack):
        """Instagram should use the same branch as TikTok (explicit, not fallback)."""
        result = adapt_for_platform(llm_content_pack, "instagram")
        caption = result["caption"]
        assert "[Chicago]" not in caption
        assert "Your home is silently" in caption

    def test_pinterest_uses_solution_steps(self, llm_content_pack):
        """Pinterest should use solution_steps, not empty step_1/step_2."""
        result = adapt_for_platform(llm_content_pack, "pinterest")
        caption = result["caption"]
        assert "eco-audit" in caption
        assert result["char_count"] > 50

    def test_pinterest_within_limit(self, llm_content_pack):
        """Pinterest caption must stay within 500 char limit."""
        result = adapt_for_platform(llm_content_pack, "pinterest")
        assert result["within_limit"]

    def test_pinterest_fallback_to_step_fields(self):
        """Pinterest should use step_1/step_2 when solution_steps is missing."""
        pack = {
            "title": "Test",
            "hook": "Test hook",
            "step_1": "Do thing one",
            "step_2": "Do thing two",
            "hashtags": ["#test"],
        }
        result = adapt_for_platform(pack, "pinterest")
        caption = result["caption"]
        assert "thing one" in caption or "thing two" in caption

    def test_facebook_no_template_garbage(self, llm_content_pack):
        """Facebook should use same clean branch as TikTok."""
        result = adapt_for_platform(llm_content_pack, "facebook")
        assert "[Chicago]" not in result["caption"]
        assert "Your home is silently" in result["caption"]

    def test_threads_no_template_garbage(self, llm_content_pack):
        """Threads should use same clean branch as TikTok."""
        result = adapt_for_platform(llm_content_pack, "threads")
        assert "[Chicago]" not in result["caption"]

    def test_default_fallback_no_template(self):
        """Unknown platforms should also use clean composition, not UNIVERSAL_CAPTION_TEMPLATE."""
        pack = {
            "title": "Test",
            "hook": "Attention grabbing hook",
            "pain_point": "The problem is real",
            "solution_steps": "Step 1, step 2, step 3",
            "cta": "Follow for more",
            "hashtags": ["#test"],
        }
        result = adapt_for_platform(pack, "unknown_platform")
        caption = result["caption"]
        assert "[Chicago]" not in caption
        assert "Busy people?" not in caption
        assert "Attention grabbing hook" in caption


class TestHashtagLLMFallback:
    """Test that LLM-generated hashtags are used when niche DB has no curated data."""

    def test_llm_hashtags_used_for_unknown_niche(self):
        """For niches not in DB, generate_content_pack should use LLM hashtags."""
        state = {
            "niche_config": {"id": "sustainable_living", "display_name": "Sustainable Living"},
            "niche_key": "sustainable_living",
            "target_platform": "tiktok",
            "budget_remaining_pct": 0,  # Force fallback (no LLM cost)
        }
        result = generate_content_pack(state)
        pack = result["content_pack"]
        # Even in fallback mode, hashtags should exist
        assert "hashtags" in pack
        assert "hashtag_strategy" in pack

    def test_db_hashtags_used_for_known_niche(self):
        """For niches in DB (like plant_based_raw), curated hashtags should be preferred."""
        state = {
            "niche_config": {"id": "plant_based_raw", "display_name": "Plant-Based Raw Foods"},
            "niche_key": "plant_based_raw",
            "target_platform": "instagram",
            "budget_remaining_pct": 0,  # Force fallback
        }
        result = generate_content_pack(state)
        pack = result["content_pack"]
        assert pack.get("hashtag_strategy") == "micro_niche_5"
        # Should have curated tags, not generic keyword extraction
        assert len(pack.get("hashtags", [])) >= 3


class TestGeminiRetryLogic:
    """Test Gemini retry on JSON truncation (mocked)."""

    def test_call_gemini_returns_none_without_key(self):
        """Without GEMINI_API_KEY, _call_gemini should return (None, '')."""
        import os
        from agents.content_factory import _call_gemini
        original = os.environ.get("GEMINI_API_KEY")
        try:
            os.environ.pop("GEMINI_API_KEY", None)
            result, name = _call_gemini("system", "user", 0.8)
            assert result is None
            assert name == ""
        finally:
            if original:
                os.environ["GEMINI_API_KEY"] = original
