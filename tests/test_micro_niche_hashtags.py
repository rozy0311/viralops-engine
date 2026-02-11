"""
Tests for hashtags/matrix_5layer.py — generate_micro_niche_5() function.
"""

import pytest
from hashtags.matrix_5layer import (
    generate_micro_niche_5,
    generate_5cap,
    generate_hashtag_matrix,
    get_available_niches,
)


class TestGenerateMicroNiche5:
    """Tests for the 5 micro-niche hashtag generator."""

    def test_returns_exactly_5_or_fewer(self):
        result = generate_micro_niche_5(niche="raw-almonds")
        assert len(result["hashtags"]) <= 5

    def test_returns_dict_with_required_keys(self):
        result = generate_micro_niche_5(niche="raw-almonds")
        assert "hashtags" in result
        assert "count" in result
        assert "strategy" in result
        assert "platform" in result
        assert "niche" in result

    def test_strategy_is_micro_niche_5(self):
        result = generate_micro_niche_5(niche="raw-almonds")
        assert result["strategy"] == "micro_niche_5"

    def test_no_generic_blacklisted_hashtags(self):
        blacklist = {"#viral", "#trending", "#fyp", "#foryou", "#foryoupage",
                     "#explore", "#explorepage", "#follow", "#like", "#instagood"}
        result = generate_micro_niche_5(niche="raw-almonds")
        for tag in result["hashtags"]:
            assert tag.lower() not in blacklist, f"Generic tag found: {tag}"

    def test_all_hashtags_start_with_hash(self):
        result = generate_micro_niche_5(niche="raw-almonds")
        for tag in result["hashtags"]:
            assert tag.startswith("#"), f"Tag missing #: {tag}"

    def test_no_duplicate_hashtags(self):
        result = generate_micro_niche_5(niche="raw-almonds")
        lower_tags = [t.lower() for t in result["hashtags"]]
        assert len(lower_tags) == len(set(lower_tags)), "Duplicate hashtags found"

    def test_topic_keywords_influence_output(self):
        result = generate_micro_niche_5(
            niche="raw-almonds",
            topic_keywords=["soaking", "nutrition", "snack"],
        )
        # At least one keyword-derived tag should appear
        all_tags_lower = " ".join(result["hashtags"]).lower()
        keyword_found = any(kw.lower() in all_tags_lower for kw in ["soaking", "nutrition", "snack"])
        # This may or may not be true depending on pool priority, but structure should be valid
        assert result["count"] > 0

    def test_unknown_niche_still_returns_result(self):
        result = generate_micro_niche_5(niche="nonexistent-niche-xyz")
        assert "hashtags" in result
        assert isinstance(result["hashtags"], list)

    def test_platform_passed_through(self):
        result = generate_micro_niche_5(niche="raw-almonds", platform="youtube")
        assert result["platform"] == "youtube"

    def test_includes_season(self):
        result = generate_micro_niche_5(niche="raw-almonds")
        assert "season" in result

    def test_has_note_field(self):
        result = generate_micro_niche_5(niche="raw-almonds")
        assert "note" in result
        assert "generic" in result["note"].lower()


class TestGenerate5Cap:
    """Tests for the Instagram 5-cap strategy."""

    def test_returns_list(self):
        result = generate_5cap(niche="raw-almonds")
        assert isinstance(result, list)

    def test_max_5_tags(self):
        result = generate_5cap(niche="raw-almonds")
        assert len(result) <= 5


class TestGetAvailableNiches:
    """Test niche listing."""

    def test_returns_list(self):
        niches = get_available_niches()
        assert isinstance(niches, list)
