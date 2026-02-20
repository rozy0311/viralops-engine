import json
from unittest.mock import patch


def _mock_llm_review_json(*, avg: float = 10.0, rubric10: float = 10.0) -> str:
    return json.dumps(
        {
            "scores": {
                "answer_quality": avg,
                "content_depth": avg,
                "tone": avg,
                "hook": avg,
                "specificity": avg,
                "actionability": avg,
                "formatting": avg,
            },
            "avg": avg,
            "rubric_scores": {
                "hyper_specific_niche": rubric10,
                "personal_story_regret": rubric10,
                "numbered_list_variations": rubric10,
                "practical_steps_measurable": rubric10,
                "quantifiable_value": rubric10,
                "low_cost_no_gear": rubric10,
                "visual_engaging_format": rubric10,
                "seo_longtail_keywords": rubric10,
                "cta_expansion_ladder": rubric10,
                "length_scannability": rubric10,
            },
            "rubric_total_100": int(rubric10 * 10),
            "pass": True,
            "feedback": "",
            "improved_title": "Stop wasting $5 — this costs $0.30 at home",
        },
        ensure_ascii=False,
    )


def test_review_hard_fails_when_missing_required_elements():
    from llm_content import ProviderResult, _review_quality_content

    pack = {
        "title": "$2 hack for sprouts",
        "_topic": "sprouted black beans houston countertop",
        # Missing: regret line, numbered list, ladder, low-cost+zero-alt, enough numbers.
        "content_formatted": "\n".join(
            [
                "🌿 Sprouted black beans are nutritious.",
                "🫙 Rinse and wait.",
                "✅ That's it.",
            ]
        ),
        "hashtags": ["sproutedblackbeans", "budgetvegan", "houston"],
        "steps": ["Step 1: Rinse", "Step 2: Wait"],
    }

    fake = ProviderResult(text=_mock_llm_review_json(), provider="github_models", model="x", success=True)
    with patch("llm_content.call_llm", return_value=fake):
        review = _review_quality_content(pack)

    assert review is not None
    assert review["pass"] is False
    assert "Deterministic gate failed" in (review.get("feedback") or "")
    # Should surface at least one deterministic reason.
    det = review.get("_deterministic") or {}
    assert det.get("hard_fail_reasons"), "Expected deterministic hard-fail reasons"


def test_review_passes_when_content_meets_deterministic_gate():
    from llm_content import ProviderResult, _review_quality_content

    numbered = "\n".join([f"{i}. Variation {i} — $1 vs $5, 2-3 days, 70°F" for i in range(1, 26)])

    # Add enough short lines to satisfy scannability + reach 3200+ chars.
    extra_tips = "\n".join(
        [
            "✅ Tip 1: Keep airflow — 10 seconds, twice daily.",
            "✅ Tip 2: If your apartment is humid, add a paper towel for 6 hours.",
            "✅ Tip 3: Aim for 70-74°F; below 65°F adds 1-2 days.",
            "✅ Tip 4: Don’t crowd: 1/2 cup beans per 16oz jar.",
            "✅ Tip 5: Salt-water rinse (1 tsp/2 cups) once if you smell sour.",
            "✅ Tip 6: Stop at 1/2-inch tails for best crunch.",
            "✅ Tip 7: $0 hack: old coffee filter works as a mesh lid.",
            "✅ Tip 8: If you forget one rinse, do 2 rinses 30 minutes apart.",
            "✅ Tip 9: Store 3 days max — dryness beats slime.",
            "✅ Tip 10: Label day 1-3 so you don’t ‘mystery sprout’ your fridge.",
            "✅ Tip 11: For smell: 1 tbsp vinegar in 2 cups water, 30 seconds.",
            "✅ Tip 12: Don’t soak longer than 12-14 hours — mush risk.",
            "✅ Tip 13: Cheap rack: $3 cooling rack = instant drip stand.",
            "✅ Tip 14: Rinse water should run clear by rinse #2.",
            "✅ Tip 15: If you see bubbles, do 3 rinses in 2 hours.",
            "✅ Tip 16: Use 2 jars to alternate: jar A day 1, jar B day 2.",
            "✅ Tip 17: Light isn’t needed — keep it dark for 48 hours.",
            "✅ Tip 18: If you want shorter tails, harvest at 36-48h.",
            "✅ Tip 19: If you want longer tails, go 60-72h max.",
            "✅ Tip 20: Rinse temp: cool water in summer, lukewarm in winter.",
            "✅ Tip 21: 1 pinch salt = less slime (don’t overdo).",
            "✅ Tip 22: $0 strainer: poke 8-10 holes in a yogurt lid.",
            "✅ Tip 23: If you travel 24h, refrigerate after rinse.",
            "✅ Tip 24: Best texture window: 1/4 to 1/2 inch tails.",
            "✅ Tip 25: Keep a timer: 8am/4pm rinses = zero guesswork.",
        ]
    )

    batch_math = "\n".join(
        [
            "🫘 Batch math:",
            "- 1/2 cup dry = ~2 cups sprouts (2-3 days).",
            "- $2/lb beans ≈ $0.25 per jar batch.",
            "- Store packs: $5 for ~8oz = $10/lb.",
            "- Your rinse time: 2 minutes × 2/day = 4 minutes.",
            "- Mold risk spikes after 24-48h without rinsing.",
            "- If your counter is 70-74°F, expect day 2 harvest.",
            "- If your counter is 65°F, expect day 3 harvest.",
        ]
    )

    content = "\n".join(
        [
            "I ruined 3 batches before I fixed this — wish I knew the 2-minute rinse rule sooner.",
                "🌿 Sprouted black beans Houston countertop method (apartment-friendly) — $2/lb, 2-3 days, 70°F.",
            "🫙 Quick Method:",
            "1. Soak 12 hours.",
            "2. Rinse every 8 hours.",
            "3. Stop at 1/2-inch tails.",
            "❌ Mistakes:",
            "- Skip rinsing = mold in 24-48h.",
            "✅ Low-cost gear:",
            "- $0 option: reuse a pickle jar + a free mesh lid.",
                extra_tips,
                batch_math,
            "Variations / Uses:",
            numbered,
            "CTA: Save this + try 1 jar tonight.",
            "Expansion ladder: Start tiny → weekly → monthly.",
            "Sprouted black beans Houston countertop — sprouted black beans Houston countertop.",
        ]
    )

    pack = {
        "title": "$2/lb sprouts beat $5 store packs — my 3-batch fix",
        "_topic": "sprouted black beans houston countertop",
        "content_formatted": content,
        "hashtags": ["sproutedblackbeans", "budgetvegan", "houston"],
        "steps": ["Step 1: Soak 12 hours", "Step 2: Rinse every 8 hours"],
    }

    fake = ProviderResult(text=_mock_llm_review_json(), provider="github_models", model="x", success=True)
    with patch("llm_content.call_llm", return_value=fake):
        review = _review_quality_content(pack)

    assert review is not None
    assert review["pass"] is True
    assert float(review.get("rubric_total_100", 0)) >= 92
