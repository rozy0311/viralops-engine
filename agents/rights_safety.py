"""
Rights & Safety Agent — Legal Agent (EMADS-PR v1.0)
Content safety and rights verification.

Fixed in v2.0:
- Unsafe patterns now explicitly set passed=False
- Niche-aware allowlist to reduce false positives
- Actual text similarity comparison for originality
- rights_status set on state
- Configurable sensitivity
"""
import re
import structlog

logger = structlog.get_logger()

# Blocked patterns — health/financial claims that violate FTC guidelines
UNSAFE_PATTERNS = [
    (r"\b(guaranteed results?|100% cure|miracle cure)\b", "health/results claim"),
    (r"\b(get rich quick|make \$\d+ per day|easy money)\b", "financial claim"),
    (r"\b(act now|limited time only|expires today|hurry)\b", "urgency manipulation"),
    (r"\b(FDA approved|clinically proven|doctor recommended)\b", "medical authority claim"),
]

# Patterns that are OK in e-commerce/marketing niches
NICHE_ALLOWLIST = {
    "plant_based_raw": ["buy now"],  # OK in product context
    "nano_real_life": ["buy now"],
    "indoor_gardening": ["buy now"],
}

TRANSFORM_MIN_PERCENT = 70  # Content must be 70%+ original


def _text_similarity(text1: str, text2: str) -> float:
    """Simple word-overlap similarity ratio (0.0 to 1.0)."""
    if not text1 or not text2:
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def check_rights(state: dict) -> dict:
    """LangGraph node: Check content for safety and rights issues."""
    content_pack = state.get("content_pack", {})
    body = content_pack.get("body", "")
    source_url = content_pack.get("_source_url", "")
    original_body = state.get("rss_content", {}).get("body", "")
    niche_key = state.get("niche_key", "")

    issues = []
    warnings = []
    passed = True

    # ── Get niche-specific allowlist ──
    allowlist = NICHE_ALLOWLIST.get(niche_key, [])

    # ── Check for unsafe patterns ──
    for pattern, description in UNSAFE_PATTERNS:
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            matched_text = match.group(0).lower()
            # Check if this match is in the allowlist for this niche
            if any(allowed in matched_text for allowed in allowlist):
                warnings.append(f"Allowed pattern in {niche_key}: {description} ('{matched_text}')")
            else:
                issues.append(f"Unsafe pattern: {description} ('{matched_text}')")
                passed = False

    # ── Check originality (if from RSS) ──
    if source_url:
        if content_pack.get("_generated_by") == "rss_passthrough":
            issues.append("RSS content not transformed — needs rewriting (70% minimum originality)")
            passed = False
        elif original_body:
            # Actual similarity check
            similarity = _text_similarity(body, original_body)
            originality = (1.0 - similarity) * 100
            if originality < TRANSFORM_MIN_PERCENT:
                issues.append(f"Originality too low: {originality:.0f}% (minimum {TRANSFORM_MIN_PERCENT}%)")
                passed = False
            else:
                logger.info("rights_safety.originality_ok", pct=f"{originality:.0f}%")

    # ── Check for empty/too-short content ──
    if len(body.strip()) < 50:
        issues.append("Content too short (minimum 50 characters)")
        passed = False

    # ── Check for missing attribution on RSS content ──
    if source_url and source_url not in body and "source" not in body.lower():
        warnings.append(f"RSS source not credited in body: {source_url}")

    state["rights_result"] = {
        "passed": passed,
        "issues": issues,
        "warnings": warnings,
        "source_url": source_url,
        "transform_required": bool(source_url),
    }
    state["rights_status"] = "completed"
    logger.info("rights_safety.done", passed=passed, issues=len(issues), warnings=len(warnings))
    return state
