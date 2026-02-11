"""
Tests for platform_compliance.py v2.1 upgrades:
  - optimal_hashtags validation
  - GenAI filler detection
  - Mid-sentence cut detection
  - Title length validation
  - YouTube Short support
  - Suggestions field in compliance result
"""

import pytest
from agents.platform_compliance import (
    check_compliance,
    PLATFORM_RULES,
    _count_hashtags,
    _has_links,
    _has_self_promotion,
    _is_unprofessional,
    _has_genai_filler,
    _is_mid_sentence_cut,
)


# ════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════

class TestHelpers:

    def test_count_hashtags(self):
        assert _count_hashtags("#hello #world") == 2
        assert _count_hashtags("no hashtags") == 0
        assert _count_hashtags("#one") == 1

    def test_has_links(self):
        assert _has_links("Visit https://example.com") is True
        assert _has_links("No links here") is False
        assert _has_links("Check www.google.com") is True

    def test_has_self_promotion(self):
        assert _has_self_promotion("Buy now with code SAVE20") is True
        assert _has_self_promotion("Great tip for your garden") is False

    def test_is_unprofessional_single_element_ok(self):
        # One casual element is fine
        assert _is_unprofessional("lol that's funny") is False

    def test_is_unprofessional_multiple_flags(self):
        # Two+ casual elements = unprofessional
        assert _is_unprofessional("OMG LOL bruh!!! that's CRAZY") is True

    def test_has_genai_filler_preamble(self):
        assert _has_genai_filler("Sure! Here's your answer.") is True
        assert _has_genai_filler("Of course, happy to help!") is True

    def test_has_genai_filler_conclusion(self):
        assert _has_genai_filler("Some content.\n\nLet me know if you need more!") is True

    def test_has_genai_filler_clean_content(self):
        assert _has_genai_filler("Herbs regrow from cuttings. Plant in spring.") is False

    def test_has_genai_filler_as_ai(self):
        assert _has_genai_filler("As an AI language model, I can explain.") is True

    def test_mid_sentence_cut_detected(self):
        # Must be > 50 chars for detection to trigger
        assert _is_mid_sentence_cut("This text ends abruptly without any punctuation and it keeps going on and") is True

    def test_mid_sentence_cut_clean_ending(self):
        assert _is_mid_sentence_cut("This text ends properly.") is False
        assert _is_mid_sentence_cut("Wow, amazing!") is False
        assert _is_mid_sentence_cut("Is this good?") is False

    def test_mid_sentence_cut_empty(self):
        assert _is_mid_sentence_cut("") is False

    def test_mid_sentence_cut_short_text(self):
        # Short text (< 50 chars) should not flag
        assert _is_mid_sentence_cut("Short text") is False


# ════════════════════════════════════════════════
# PLATFORM_RULES coverage
# ════════════════════════════════════════════════

class TestPlatformRules:

    def test_all_platforms_present(self):
        expected = [
            "tiktok", "instagram", "facebook", "youtube", "youtube_short",
            "pinterest", "linkedin", "twitter", "reddit", "medium",
            "tumblr", "shopify_blog",
        ]
        for p in expected:
            assert p in PLATFORM_RULES, f"Missing platform: {p}"

    def test_all_have_optimal_hashtags(self):
        for p, rules in PLATFORM_RULES.items():
            assert "optimal_hashtags" in rules, f"{p} missing optimal_hashtags"
            assert "max_hashtags" in rules, f"{p} missing max_hashtags"

    def test_youtube_short_exists(self):
        assert "youtube_short" in PLATFORM_RULES
        assert PLATFORM_RULES["youtube_short"]["max_chars"] == 100
        assert PLATFORM_RULES["youtube_short"]["optimal_hashtags"] == 3

    def test_reddit_zero_hashtags(self):
        assert PLATFORM_RULES["reddit"]["optimal_hashtags"] == 0
        assert PLATFORM_RULES["reddit"]["max_hashtags"] == 0

    def test_twitter_limits(self):
        assert PLATFORM_RULES["twitter"]["max_chars"] == 280
        assert PLATFORM_RULES["twitter"]["max_hashtags"] == 3


# ════════════════════════════════════════════════
# check_compliance() — main LangGraph node
# ════════════════════════════════════════════════

class TestCheckCompliance:

    def _make_state(self, platforms, body="Good content here.", title="Test Title",
                    hashtags=None, caption=None, image_prompt="", subreddit=None):
        state = {
            "platforms": platforms,
            "content_pack": {
                "body": body,
                "title": title,
                "caption": caption or body,
                "hashtags": hashtags or [],
                "image_prompt": image_prompt,
            },
        }
        if subreddit:
            state["content_pack"]["subreddit"] = subreddit
        return state

    def test_passes_clean_content(self):
        state = self._make_state(
            platforms=["facebook"],
            body="Clean content for Facebook post.",
            hashtags=["#herb", "#garden", "#tips", "#organic", "#spring"],
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert cr["passed"] is True
        assert len(cr["issues"]) == 0

    def test_fails_over_char_limit(self):
        state = self._make_state(
            platforms=["twitter"],
            body="A" * 300,  # Over 280
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert cr["passed"] is False
        assert any("exceeds" in i for i in cr["issues"])

    def test_fails_under_min_chars(self):
        state = self._make_state(
            platforms=["medium"],
            body="Short.",  # Under 200 min
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert cr["passed"] is False
        assert any("below minimum" in i for i in cr["issues"])

    def test_warns_on_links_in_caption(self):
        state = self._make_state(
            platforms=["tiktok"],
            body="Check https://example.com for tips.",
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert any("Links detected" in w for w in cr["warnings"])

    def test_requires_title(self):
        state = self._make_state(
            platforms=["youtube"],
            title="",
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert cr["passed"] is False
        assert any("Title required" in i for i in cr["issues"])

    def test_requires_image_pinterest(self):
        state = self._make_state(
            platforms=["pinterest"],
            image_prompt="",
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert cr["passed"] is False
        assert any("Image required" in i for i in cr["issues"])

    def test_requires_subreddit(self):
        state = self._make_state(
            platforms=["reddit"],
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert cr["passed"] is False
        assert any("Subreddit" in i for i in cr["issues"])

    def test_subreddit_provided_passes(self):
        state = self._make_state(
            platforms=["reddit"],
            subreddit="r/gardening",
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert cr["passed"] is True

    def test_tiktok_requires_hashtags(self):
        state = self._make_state(
            platforms=["tiktok"],
            body="Good content without any hashtags.",
            hashtags=[],
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert cr["passed"] is False
        assert any("Hashtags required" in i for i in cr["issues"])

    def test_optimal_hashtag_suggestion(self):
        """Non-optimal hashtag count should generate a suggestion, not error."""
        state = self._make_state(
            platforms=["instagram"],
            hashtags=["#one", "#two"],  # Only 2, optimal is 5
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        # Suggestion, NOT an issue — post is still valid
        assert "suggestions" in cr
        assert any("optimal" in s for s in cr["suggestions"])

    def test_warns_on_too_many_hashtags(self):
        state = self._make_state(
            platforms=["twitter"],
            body="Post content. #one #two #three #four #five",
            hashtags=[],
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert any("exceeds limit" in w for w in cr["warnings"])

    def test_warns_on_genai_filler(self):
        state = self._make_state(
            platforms=["instagram"],
            body="Sure! Here's the answer. Herbs regrow easily.",
            hashtags=["#a", "#b", "#c", "#d", "#e"],
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert any("GenAI filler" in w for w in cr["warnings"])

    def test_warns_on_mid_sentence_cut(self):
        state = self._make_state(
            platforms=["facebook"],
            body="This content was truncated at a really awkward place and",
            hashtags=["#a", "#b", "#c", "#d", "#e"],
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert any("mid-sentence" in w for w in cr["warnings"])

    def test_warns_on_title_too_long(self):
        state = self._make_state(
            platforms=["youtube"],
            title="A" * 150,  # Over 100 char limit
            hashtags=["#a", "#b", "#c", "#d", "#e"],
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert any("Title exceeds" in w for w in cr["warnings"])

    def test_professional_tone_linkedin(self):
        state = self._make_state(
            platforms=["linkedin"],
            body="OMG LOL this is CRAZY bruh!!! fr fr ngl fam",
            hashtags=["#a", "#b", "#c", "#d", "#e"],
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert any("Unprofessional" in w for w in cr["warnings"])

    def test_self_promotion_reddit(self):
        state = self._make_state(
            platforms=["reddit"],
            body="Buy now with code SAVE20 for 50% discount!",
            subreddit="r/deals",
            hashtags=[],
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert any("Self-promotional" in w for w in cr["warnings"])

    def test_compliance_status_set(self):
        state = self._make_state(platforms=["facebook"])
        result = check_compliance(state)
        assert result["compliance_status"] == "completed"

    def test_multi_platform_check(self):
        state = self._make_state(
            platforms=["tiktok", "instagram", "twitter"],
            hashtags=["#one", "#two", "#three", "#four", "#five"],
        )
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert set(cr["platforms_checked"]) == {"tiktok", "instagram", "twitter"}

    def test_result_has_suggestions_field(self):
        state = self._make_state(platforms=["facebook"])
        result = check_compliance(state)
        cr = result["compliance_result"]
        assert "suggestions" in cr
        assert isinstance(cr["suggestions"], list)
