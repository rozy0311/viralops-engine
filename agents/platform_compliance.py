"""
Platform Compliance Agent — COO Agent (EMADS-PR v1.0)
Checks content against platform-specific rules.

v2.1 upgrades:
  - Precise per-channel char limits (algo-optimal, not just max)
  - Smart hashtag count: exactly 5 micro-niche per channel
  - YouTube: 3-5 hashtags (algo won't push if >15)
  - Twitter: 280 char + 3 hashtags max
  - Content split quality validation (no mid-sentence cuts)
  - GenAI filler detection (warn if preamble/conclusion not stripped)
"""
import re
import structlog

logger = structlog.get_logger()

# ── Platform content rules — ALL enforced ──
# Now with "optimal" fields (what the algo WANTS, not just max allowed)
PLATFORM_RULES = {
    "tiktok":       {"max_chars": 2200, "no_links": True, "requires_hashtags": True,
                     "optimal_hashtags": 5, "max_hashtags": 8},
    "instagram":    {"max_chars": 2200, "max_hashtags": 30, "no_links_in_caption": True,
                     "optimal_hashtags": 5},
    "facebook":     {"max_chars": 5000, "optimal_hashtags": 5, "max_hashtags": 10},
    "youtube":      {"max_chars": 5000, "requires_title": True,
                     "optimal_hashtags": 5, "max_hashtags": 15,
                     "title_max_chars": 100},
    "youtube_short":{"max_chars": 100, "requires_title": True,
                     "optimal_hashtags": 3, "max_hashtags": 5,
                     "title_max_chars": 100},
    "pinterest":    {"max_chars": 500, "requires_image": True,
                     "optimal_hashtags": 5, "max_hashtags": 20,
                     "title_max_chars": 100},
    "linkedin":     {"max_chars": 3000, "professional_tone": True,
                     "optimal_hashtags": 5, "max_hashtags": 5},
    "twitter":      {"max_chars": 280, "optimal_hashtags": 3, "max_hashtags": 3},
    "reddit":       {"max_chars": 40000, "no_self_promotion": True, "requires_subreddit": True,
                     "optimal_hashtags": 0, "max_hashtags": 0,
                     "title_max_chars": 300},
    "medium":       {"min_chars": 200, "supports_html": True, "requires_title": True,
                     "optimal_hashtags": 5, "max_hashtags": 5,
                     "title_max_chars": 200},
    "tumblr":       {"max_chars": 50000, "supports_html": True,
                     "optimal_hashtags": 5, "max_hashtags": 30},
    "shopify_blog": {"min_chars": 100, "requires_title": True, "supports_html": True,
                     "optimal_hashtags": 0, "max_hashtags": 0,
                     "title_max_chars": 200},
}

# Patterns
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_PROMO_PATTERNS = [
    re.compile(r"\b(buy now|shop now|order now|use code|discount|promo)\b", re.IGNORECASE),
]
_UNPROFESSIONAL_PATTERNS = [
    re.compile(r"[A-Z]{5,}"),  # ALL CAPS words
    re.compile(r"[!]{3,}"),     # Multiple exclamation marks
    re.compile(r"\b(lol|omg|wtf|bruh|fam|ngl|fr fr)\b", re.IGNORECASE),
]

# GenAI filler detection — warn if content still has preamble
_GENAI_FILLER_PATTERNS = [
    re.compile(r'^(Sure[!.,]|Of course|Absolutely|Great question|Here\'s)', re.IGNORECASE),
    re.compile(r'(Let me know|Feel free|Hope this helps|If you have any)', re.IGNORECASE),
    re.compile(r'^(As an AI|As a language model)', re.IGNORECASE),
]

# Mid-sentence cut detection
_MID_SENTENCE_CUT = re.compile(r'[a-zA-Z]{2,}\.\.\.$|[a-zA-Z,]\s*$')


def _count_hashtags(text: str) -> int:
    """Count hashtags in text."""
    return len(re.findall(r"#\w+", text))


def _has_links(text: str) -> bool:
    """Check if text contains URLs."""
    return bool(_URL_PATTERN.search(text))


def _has_self_promotion(text: str) -> bool:
    """Check for self-promotional language."""
    return any(p.search(text) for p in _PROMO_PATTERNS)


def _is_unprofessional(text: str) -> bool:
    """Check for unprofessional language patterns."""
    matches = sum(1 for p in _UNPROFESSIONAL_PATTERNS if p.search(text))
    return matches >= 2  # Allow one casual element, flag if 2+


def _has_genai_filler(text: str) -> bool:
    """Detect if GenAI filler/preamble is still present in content."""
    return any(p.search(text) for p in _GENAI_FILLER_PATTERNS)


def _is_mid_sentence_cut(text: str) -> bool:
    """Detect if text was cut mid-sentence (bad truncation)."""
    if not text:
        return False
    # Good endings: . ! ? ) ] " \n
    last_char = text.rstrip()[-1] if text.rstrip() else ""
    return last_char not in ".!?)]\"\n'" and len(text) > 50


def check_compliance(state: dict) -> dict:
    """
    LangGraph node: Check content against ALL platform rules.

    v2.1: Now also checks:
      - Content split quality (no mid-sentence cuts)
      - GenAI filler detection
      - Optimal vs max hashtag count
      - Title length per platform
    """
    content_pack = state.get("content_pack", {})
    platforms = state.get("platforms", [])
    body = content_pack.get("body", "")
    title = content_pack.get("title", "")
    caption = content_pack.get("caption", body)
    hashtags = content_pack.get("hashtags", [])
    image_prompt = content_pack.get("image_prompt", "")

    issues = []
    warnings = []
    suggestions = []
    passed = True

    for platform in platforms:
        rules = PLATFORM_RULES.get(platform, {})
        check_text = caption if platform in ("tiktok", "instagram", "twitter", "linkedin", "pinterest") else body

        # ── Character limits ──
        max_chars = rules.get("max_chars", 50000)
        if len(check_text) > max_chars:
            issues.append(f"{platform}: Content exceeds {max_chars} chars ({len(check_text)}) — will be truncated by platform")
            passed = False

        min_chars = rules.get("min_chars", 0)
        if len(check_text) < min_chars:
            issues.append(f"{platform}: Content below minimum {min_chars} chars ({len(check_text)})")
            passed = False

        # ── Title length ──
        title_max = rules.get("title_max_chars", 0)
        if title_max > 0 and len(title) > title_max:
            warnings.append(f"{platform}: Title exceeds {title_max} chars ({len(title)}) — may be truncated")

        # ── Title required ──
        if rules.get("requires_title") and not title.strip():
            issues.append(f"{platform}: Title required but missing")
            passed = False

        # ── Hashtag requirements ──
        if rules.get("requires_hashtags"):
            hashtag_count = _count_hashtags(check_text) + len(hashtags)
            if hashtag_count == 0:
                issues.append(f"{platform}: Hashtags required but none found")
                passed = False

        # ── Optimal hashtag count (algo recommendation) ──
        optimal = rules.get("optimal_hashtags", 5)
        actual_count = len(hashtags)
        if optimal > 0 and actual_count != optimal:
            suggestions.append(f"{platform}: {actual_count} hashtags (optimal: {optimal} micro-niche for best algo reach)")

        # ── Max hashtags ──
        max_hashtags = rules.get("max_hashtags")
        if max_hashtags is not None:
            hashtag_count = _count_hashtags(check_text) + len(hashtags)
            if hashtag_count > max_hashtags:
                warnings.append(f"{platform}: {hashtag_count} hashtags exceeds limit of {max_hashtags}")

        # ── No links in caption ──
        if rules.get("no_links_in_caption") or rules.get("no_links"):
            if _has_links(check_text):
                warnings.append(f"{platform}: Links detected in caption (may be suppressed by platform)")

        # ── Image required ──
        if rules.get("requires_image") and not image_prompt.strip():
            issues.append(f"{platform}: Image required but no image_prompt provided")
            passed = False

        # ── Subreddit required ──
        if rules.get("requires_subreddit"):
            subreddit = state.get("subreddit") or content_pack.get("subreddit")
            if not subreddit:
                issues.append(f"{platform}: Subreddit must be specified for Reddit posts")
                passed = False

        # ── No self-promotion ──
        if rules.get("no_self_promotion"):
            if _has_self_promotion(check_text):
                warnings.append(f"{platform}: Self-promotional language detected (may violate community rules)")

        # ── Professional tone ──
        if rules.get("professional_tone"):
            if _is_unprofessional(check_text):
                warnings.append(f"{platform}: Unprofessional tone detected (review before posting)")

        # ── Content split quality check ──
        if _is_mid_sentence_cut(check_text):
            warnings.append(f"{platform}: Content appears to be cut mid-sentence — consider smart_truncate")

        # ── GenAI filler check ──
        if _has_genai_filler(check_text):
            warnings.append(f"{platform}: GenAI filler detected (preamble/conclusion not stripped) — run extract_relevant_answer()")

    state["compliance_result"] = {
        "passed": passed,
        "issues": issues,
        "warnings": warnings,
        "suggestions": suggestions,
        "platforms_checked": platforms,
    }
    state["compliance_status"] = "completed"
    logger.info("compliance.done", passed=passed, issues=len(issues),
                warnings=len(warnings), suggestions=len(suggestions))
    return state
