"""
Platform Compliance Agent — COO Agent (EMADS-PR v1.0)
Checks content against platform-specific rules.

Fixed in v2.0:
- ALL defined rules now actually enforced
- Hashtag count validation
- Link detection in captions
- Image requirement check
- Subreddit requirement check
- Professional tone basic check
- compliance_status set on state
"""
import re
import structlog

logger = structlog.get_logger()

# Platform content rules — ALL enforced
PLATFORM_RULES = {
    "tiktok":       {"max_chars": 2200, "no_links": True, "requires_hashtags": True},
    "instagram":    {"max_chars": 2200, "max_hashtags": 30, "no_links_in_caption": True},
    "facebook":     {"max_chars": 5000},
    "youtube":      {"max_chars": 5000, "requires_title": True},
    "pinterest":    {"max_chars": 500, "requires_image": True},
    "linkedin":     {"max_chars": 3000, "professional_tone": True, "max_hashtags": 5},
    "twitter":      {"max_chars": 280, "max_hashtags": 3},
    "reddit":       {"max_chars": 40000, "no_self_promotion": True, "requires_subreddit": True},
    "medium":       {"min_chars": 200, "supports_html": True, "requires_title": True},
    "tumblr":       {"max_chars": 50000, "supports_html": True},
    "shopify_blog": {"min_chars": 100, "requires_title": True, "supports_html": True},
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


def check_compliance(state: dict) -> dict:
    """LangGraph node: Check content against ALL platform rules."""
    content_pack = state.get("content_pack", {})
    platforms = state.get("platforms", [])
    body = content_pack.get("body", "")
    title = content_pack.get("title", "")
    caption = content_pack.get("caption", body)
    hashtags = content_pack.get("hashtags", [])
    image_prompt = content_pack.get("image_prompt", "")

    issues = []
    warnings = []
    passed = True

    for platform in platforms:
        rules = PLATFORM_RULES.get(platform, {})
        check_text = caption if platform in ("tiktok", "instagram", "twitter", "linkedin") else body

        # ── Character limits ──
        max_chars = rules.get("max_chars", 50000)
        if len(check_text) > max_chars:
            issues.append(f"{platform}: Content exceeds {max_chars} chars ({len(check_text)})")
            passed = False

        min_chars = rules.get("min_chars", 0)
        if len(check_text) < min_chars:
            issues.append(f"{platform}: Content below minimum {min_chars} chars ({len(check_text)})")
            passed = False

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

    state["compliance_result"] = {
        "passed": passed,
        "issues": issues,
        "warnings": warnings,
        "platforms_checked": platforms,
    }
    state["compliance_status"] = "completed"
    logger.info("compliance.done", passed=passed, issues=len(issues), warnings=len(warnings))
    return state
