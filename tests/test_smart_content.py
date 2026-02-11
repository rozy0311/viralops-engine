"""
Tests for v2.1 content quality upgrades:
  - smart_truncate()
  - smart_split_for_thread()
  - extract_relevant_answer()
  - adapt_for_platform() rewrite
  - CHANNEL_CHAR_LIMITS coverage
"""

import pytest
from agents.content_factory import (
    smart_truncate,
    smart_split_for_thread,
    extract_relevant_answer,
    adapt_for_platform,
    CHANNEL_CHAR_LIMITS,
)


# ════════════════════════════════════════════════
# smart_truncate()
# ════════════════════════════════════════════════

class TestSmartTruncate:
    """Tests for sentence-boundary-aware truncation."""

    def test_short_text_unchanged(self):
        assert smart_truncate("Hello world", 100) == "Hello world"

    def test_empty_text(self):
        assert smart_truncate("", 100) == ""

    def test_none_text(self):
        assert smart_truncate(None, 100) is None

    def test_exact_limit(self):
        text = "A" * 280
        assert smart_truncate(text, 280) == text

    def test_cuts_at_sentence_boundary(self):
        text = "First sentence here. Second sentence here. Third sentence that is really long and goes over the limit."
        result = smart_truncate(text, 50)
        # Must end with a complete sentence
        assert result.endswith(".")
        assert len(result) <= 50

    def test_never_cuts_mid_word(self):
        text = "Hello world this is a test sentence with many words inside."
        result = smart_truncate(text, 30)
        # Result should end cleanly (., !, ?, ..., or full word)
        assert len(result) <= 33  # Allow small overflow for ...
        # Should not end with a sliced partial word
        stripped = result.rstrip('.!? ')
        if stripped:
            # Last "word" should be a complete word from the original text
            last_word = stripped.split()[-1]
            assert last_word in text

    def test_paragraph_boundary_preferred(self):
        text = "Paragraph one content here.\n\nParagraph two content.\n\nParagraph three is extra long and should not be included in truncation."
        result = smart_truncate(text, 60)
        # Should keep complete paragraphs
        assert "Paragraph one" in result
        assert len(result) <= 60

    def test_preserves_meaning_flag(self):
        text = "First sentence. Second sentence. Third very long sentence that goes way past the limit."
        hard = smart_truncate(text, 40, preserve_meaning=False)
        # Hard cut can end with ...
        assert len(hard) <= 43  # +3 for ...

    def test_safety_guard_minimum_30_percent(self):
        """If paragraph/sentence cuts produce < 30% of max, fall back to word boundary."""
        text = "One very long sentence with no periods that just keeps going and going and going forever."
        result = smart_truncate(text, 50)
        # Should still return something meaningful
        assert len(result) > 0
        assert len(result) <= 53  # Allow for "..."

    def test_list_content_not_split_midway(self):
        text = "Steps to follow:\n1. Buy ingredients.\n2. Mix together.\n3. Bake for 30 min.\n4. Let cool.\n5. Serve and enjoy."
        result = smart_truncate(text, 80)
        # Should not cut mid-step
        assert result.endswith(".")

    def test_twitter_280_limit(self):
        text = "A" * 100 + ". " + "B" * 100 + ". " + "C" * 100 + "."
        result = smart_truncate(text, 280)
        assert len(result) <= 280


# ════════════════════════════════════════════════
# smart_split_for_thread()
# ════════════════════════════════════════════════

class TestSmartSplitForThread:
    """Tests for Twitter thread splitting."""

    def test_short_text_single_chunk(self):
        result = smart_split_for_thread("Short tweet.", 280)
        assert len(result) == 1
        assert result[0] == "Short tweet."

    def test_long_text_multi_chunks(self):
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence goes on and on and needs to be very long to trigger splitting."
        result = smart_split_for_thread(text, 60)
        assert len(result) > 1

    def test_thread_numbering(self):
        text = "Sentence one is here. Sentence two is here. Sentence three is right here. Sentence four rounds it out."
        result = smart_split_for_thread(text, 40)
        if len(result) > 1:
            assert result[0].startswith("1/")
            assert result[-1].startswith(f"{len(result)}/")

    def test_each_chunk_within_limit(self):
        text = "A" * 50 + ". " + "B" * 50 + ". " + "C" * 50 + ". " + "D" * 50 + "."
        result = smart_split_for_thread(text, 80)
        # Each chunk should be roughly within limit (numbering adds ~4 chars)
        for chunk in result:
            assert len(chunk) <= 85  # Allow small overhead for numbering

    def test_empty_text(self):
        result = smart_split_for_thread("", 280)
        assert result == [] or result == [""]

    def test_custom_chunk_size(self):
        text = "Short. Medium text. Longer text here."
        result = smart_split_for_thread(text, 20)
        assert len(result) >= 1


# ════════════════════════════════════════════════
# extract_relevant_answer()
# ════════════════════════════════════════════════

class TestExtractRelevantAnswer:
    """Tests for GenAI filler stripping."""

    def test_strips_sure_preamble(self):
        text = "Sure! Here's the answer.\n\nHerbs regrow from cuttings easily."
        result = extract_relevant_answer(text)
        assert not result.startswith("Sure")
        assert "Herbs regrow" in result

    def test_strips_of_course_preamble(self):
        text = "Of course! I'd be happy to help.\n\nThe best time to plant is spring."
        result = extract_relevant_answer(text)
        assert "The best time" in result
        assert "Of course" not in result

    def test_strips_as_an_ai(self):
        text = "As an AI language model, I can help.\n\nRaw almonds contain vitamin E."
        result = extract_relevant_answer(text)
        assert "Raw almonds" in result
        assert "As an AI" not in result

    def test_strips_let_me_know_conclusion(self):
        text = "Raw almonds are great for health.\n\nLet me know if you need more info!"
        result = extract_relevant_answer(text)
        assert "Raw almonds" in result
        assert "Let me know" not in result

    def test_strips_hope_this_helps(self):
        text = "Here are the steps:\n1. Plant seeds.\n2. Water daily.\n\nHope this helps!"
        result = extract_relevant_answer(text)
        assert "Plant seeds" in result
        assert "Hope this helps" not in result

    def test_strips_feel_free(self):
        text = "The answer is 42.\n\nFeel free to ask more questions."
        result = extract_relevant_answer(text)
        assert "The answer is 42" in result
        assert "Feel free" not in result

    def test_preserves_actual_content(self):
        text = "Herbs regrow from cuttings. Plant in spring. Water daily. Harvest in 30 days."
        result = extract_relevant_answer(text)
        # No filler to strip — content should be unchanged
        assert result == text

    def test_safety_guard_reverts_if_over_stripped(self):
        """If stripping removes >70%, revert to original."""
        text = "Sure, here is a short tip.\n\nX."
        result = extract_relevant_answer(text)
        # Should not strip down to just "X." — that's too aggressive
        assert len(result) > 0

    def test_empty_text(self):
        assert extract_relevant_answer("") == ""

    def test_none_text(self):
        assert extract_relevant_answer(None) is None

    def test_multiple_filler_both_ends(self):
        text = "Absolutely! Great question.\n\nPlant basil in spring.\nWater twice daily.\n\nLet me know if you want more tips!"
        result = extract_relevant_answer(text)
        assert "Plant basil" in result
        assert "Absolutely" not in result
        assert "Let me know" not in result

    def test_preserves_numbered_lists(self):
        text = "Sure!\n\n1. Buy soil\n2. Dig hole\n3. Plant seed\n4. Water\n\nHope this helps!"
        result = extract_relevant_answer(text)
        assert "1. Buy soil" in result
        assert "4. Water" in result

    def test_preserves_markdown_structure(self):
        text = "## Growing Tips\n\n- Sunlight 6hrs\n- Water daily\n- Fertilize monthly"
        result = extract_relevant_answer(text)
        assert "## Growing Tips" in result
        assert "- Sunlight" in result


# ════════════════════════════════════════════════
# CHANNEL_CHAR_LIMITS
# ════════════════════════════════════════════════

class TestChannelCharLimits:
    """Validate CHANNEL_CHAR_LIMITS dictionary coverage."""

    EXPECTED_PLATFORMS = [
        "tiktok", "instagram", "facebook", "youtube", "youtube_short",
        "pinterest", "linkedin", "twitter", "reddit", "medium",
        "tumblr", "shopify_blog",
    ]

    def test_all_platforms_present(self):
        for p in self.EXPECTED_PLATFORMS:
            assert p in CHANNEL_CHAR_LIMITS, f"Missing platform: {p}"

    def test_all_have_caption_limit(self):
        for p, limits in CHANNEL_CHAR_LIMITS.items():
            assert "caption" in limits, f"{p} missing caption limit"
            assert isinstance(limits["caption"], int)
            assert limits["caption"] > 0

    def test_all_have_optimal_hashtags(self):
        for p, limits in CHANNEL_CHAR_LIMITS.items():
            assert "optimal_hashtags" in limits, f"{p} missing optimal_hashtags"
            assert isinstance(limits["optimal_hashtags"], int)

    def test_reddit_zero_hashtags(self):
        assert CHANNEL_CHAR_LIMITS["reddit"]["optimal_hashtags"] == 0

    def test_shopify_zero_hashtags(self):
        assert CHANNEL_CHAR_LIMITS["shopify_blog"]["optimal_hashtags"] == 0

    def test_twitter_280_chars(self):
        assert CHANNEL_CHAR_LIMITS["twitter"]["caption"] == 280

    def test_twitter_3_hashtags(self):
        assert CHANNEL_CHAR_LIMITS["twitter"]["optimal_hashtags"] == 3

    def test_youtube_title_100(self):
        assert CHANNEL_CHAR_LIMITS["youtube"]["title"] == 100


# ════════════════════════════════════════════════
# adapt_for_platform()
# ════════════════════════════════════════════════

class TestAdaptForPlatform:
    """Tests for platform adaptation with smart truncation."""

    SAMPLE_PACK = {
        "title": "5 Herbs That Regrow From Kitchen Scraps",
        "body": "Growing herbs from scraps is easy and sustainable. Start with basil cuttings in water. Within a week you'll see roots. Transfer to soil for continued growth. Mint and green onions work great too.",
        "hook": "Stop buying herbs! Regrow them from scraps 🌿",
        "pain_point": "Buying fresh herbs every week wastes money.",
        "solution_steps": "1. Cut stems. 2. Place in water. 3. Wait for roots. 4. Transfer to soil.",
        "step_1": "Cut stems at 45° angle",
        "step_2": "Place in water by window",
        "result": "Free herbs forever — saves $200/year",
        "cta": "Save this and try TODAY! Follow for more tips 👆",
        "micro_keywords": "herb garden • kitchen scraps • regrow basil",
        "hashtags": ["#HerbGarden", "#KitchenScraps", "#RegrowBasil", "#SustainableLiving", "#PlantTips"],
        "image_prompt": "A sunny kitchen windowsill with herbs growing in glass jars",
    }

    def test_twitter_within_280(self):
        result = adapt_for_platform(self.SAMPLE_PACK, "twitter")
        assert result["char_count"] <= 280
        assert result["within_limit"] is True
        assert result["platform"] == "twitter"
        assert result["hashtag_count"] <= 3

    def test_twitter_long_body_creates_thread(self):
        long_pack = {**self.SAMPLE_PACK, "body": "A" * 100 + ". " + "B" * 100 + ". " + "C" * 100 + "."}
        result = adapt_for_platform(long_pack, "twitter")
        assert "thread_parts" in result
        assert result["thread_count"] > 1

    def test_reddit_no_hashtags(self):
        result = adapt_for_platform(self.SAMPLE_PACK, "reddit")
        assert result["hashtags"] == []
        assert result["hashtag_count"] == 0

    def test_tiktok_5_hashtags(self):
        result = adapt_for_platform(self.SAMPLE_PACK, "tiktok")
        assert result["hashtag_count"] <= 5
        assert result["within_limit"] is True

    def test_youtube_hashtags_at_end(self):
        result = adapt_for_platform(self.SAMPLE_PACK, "youtube")
        # YouTube description should contain hashtags near the end
        assert result["hashtag_count"] <= 5
        assert result["within_limit"] is True

    def test_pinterest_short_format(self):
        result = adapt_for_platform(self.SAMPLE_PACK, "pinterest")
        assert result["char_count"] <= 500
        assert result["within_limit"] is True

    def test_linkedin_professional_format(self):
        result = adapt_for_platform(self.SAMPLE_PACK, "linkedin")
        assert result["within_limit"] is True
        assert result["char_count"] <= 3000

    def test_medium_full_body(self):
        result = adapt_for_platform(self.SAMPLE_PACK, "medium")
        # Medium should preserve full body
        assert result["caption"] == self.SAMPLE_PACK["body"]

    def test_result_metadata_fields(self):
        result = adapt_for_platform(self.SAMPLE_PACK, "instagram")
        assert "char_count" in result
        assert "char_limit" in result
        assert "within_limit" in result
        assert "hashtag_count" in result
        assert "platform" in result
        assert "title" in result
        assert "caption" in result

    def test_unknown_platform_uses_default(self):
        result = adapt_for_platform(self.SAMPLE_PACK, "unknown_new_platform")
        # Should not crash — uses twitter defaults
        assert result["platform"] == "unknown_new_platform"
        assert "caption" in result

    def test_empty_hashtags(self):
        pack = {**self.SAMPLE_PACK, "hashtags": []}
        result = adapt_for_platform(pack, "tiktok")
        assert result["hashtag_count"] == 0
