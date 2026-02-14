"""
Hashtag Matrix 7-Layer Generator — EMADS-PR v1.0
From Micro Niche Blogs spec: Universal 7-layer hashtag strategy.

Layers (from spec):
1. Broad      — Main topic (#Gardening, #PlantBased)
2. Local      — Location-based (#ChicagoWinter, #NYCSpring)
3. Micro1     — Audience segment 1 (#BusyPeople, #FitnessMoms)
4. Micro2     — Audience segment 2 (#ApartmentLiving, #BudgetShoppers)
5. Micro3     — Audience segment 3 (#BeginnerGardeners, #SnackBeginners)
6. Creator    — Brand/UGC tag (#ChicagoUGC, #PlantCreator)
7. Trend      — Season/trend 2026 (#WinterRoutine2026, #PlantBased2026)

Also supports "5 highest search hashtags" strategy for Instagram's 5-cap.
"""
import os
import json
import random
from typing import Optional

import structlog
import yaml

logger = structlog.get_logger()

# ── Paths ──
HASHTAG_DB_PATH = os.path.join(os.path.dirname(__file__), "niche_hashtags.json")
NICHES_YAML_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "niches.yaml")

# ── Platform hashtag limits ──
PLATFORM_LIMITS = {
    "tiktok": 8,
    "instagram": 30,
    "instagram_5cap": 5,  # Instagram "5 highest search" strategy
    "facebook": 10,
    "youtube": 15,
    "pinterest": 20,
    "linkedin": 5,
    "twitter": 3,
    "reddit": 0,
    "medium": 5,
    "tumblr": 30,
    "shopify_blog": 0,
}

# ── Fallback pools (used when niche not found in DB) ──
FALLBACK_POOLS = {
    "broad": ["#viral", "#trending", "#fyp", "#explore", "#lifestyle"],
    "local": ["#USALife"],
    "micro1": ["#BusyPeople"],
    "micro2": ["#ModernLiving"],
    "micro3": ["#Beginners"],
    "creator": ["#UGCCreator"],
    "trend": ["#Trending2026"],
    "seasonal": {
        "spring": ["#Spring", "#SpringTime", "#NewBeginnings"],
        "summer": ["#Summer", "#SummerVibes", "#Sunshine"],
        "fall": ["#Fall", "#Autumn", "#CozyVibes"],
        "winter": ["#Winter", "#WinterVibes", "#Holidays"],
    },
}


def _load_niche_db() -> dict:
    """Load 7-layer niche hashtag database from JSON."""
    try:
        if os.path.exists(HASHTAG_DB_PATH):
            with open(HASHTAG_DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Remove schema metadata
                return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception as e:
        logger.warning("hashtag_db.load_error", error=str(e))
    return {}


def _load_niches_yaml() -> dict:
    """Load niches.yaml for sub-niche specific hashtag data."""
    try:
        if os.path.exists(NICHES_YAML_PATH):
            with open(NICHES_YAML_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("niches_yaml.load_error", error=str(e))
    return {}


def _find_sub_niche_hashtags(niche: str) -> dict:
    """Find hashtags_layer for a specific sub-niche from niches.yaml."""
    yaml_data = _load_niches_yaml()
    niches = yaml_data.get("niches", {})

    for _category, category_data in niches.items():
        for sub in category_data.get("sub_niches", []):
            sub_id = sub.get("id", "")
            sub_name = sub.get("name", "").lower()
            # Match by ID or partial name
            if sub_id == niche or niche.replace("_", "-") == sub_id or niche.lower() in sub_name:
                return sub.get("hashtags_layer", {})
    return {}


def _get_season() -> str:
    """Get current season based on month."""
    from datetime import datetime
    month = datetime.now().month
    if month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    elif month in (9, 10, 11):
        return "fall"
    else:
        return "winter"


def generate_hashtag_matrix(
    niche: str,
    platform: str = "instagram",
    location: Optional[str] = None,
    sub_niche_id: Optional[str] = None,
    custom_tags: list = None,
    use_5cap: bool = False,
) -> dict:
    """
    Generate a 7-layer hashtag matrix for a given niche and platform.

    Args:
        niche: Niche key (e.g. "raw_almonds", "chickpeas")
        platform: Target platform (e.g. "instagram", "tiktok")
        location: Optional location override (e.g. "Chicago", "NYC")
        sub_niche_id: Optional specific sub-niche ID from niches.yaml
        custom_tags: Optional additional tags to prepend
        use_5cap: Use Instagram "5 highest search" strategy

    Returns:
        dict with 7 layers, combined list, highest_search, and metadata
    """
    db = _load_niche_db()
    niche_data = db.get(niche, {})

    # If not found in JSON, try niches.yaml
    yaml_hashtags = {}
    if sub_niche_id:
        yaml_hashtags = _find_sub_niche_hashtags(sub_niche_id)
    elif not niche_data:
        yaml_hashtags = _find_sub_niche_hashtags(niche)

    # Determine effective platform limit
    if use_5cap or platform == "instagram_5cap":
        limit = 5
        effective_platform = "instagram_5cap"
    else:
        limit = PLATFORM_LIMITS.get(platform, 10)
        effective_platform = platform

    if limit == 0:
        return {
            "layers": {},
            "combined": [],
            "highest_search": [],
            "platform": platform,
            "total": 0,
            "limit": 0,
            "note": f"{platform} doesn't use hashtags",
        }

    # ── Build 7 layers ──
    season = _get_season()

    # Layer 1: Broad (main topic)
    layer_broad = (
        niche_data.get("broad")
        or yaml_hashtags.get("niche", [])
        or FALLBACK_POOLS["broad"]
    )

    # Layer 2: Local (location-based)
    layer_local = niche_data.get("local") or yaml_hashtags.get("location", []) or []
    if location:
        location_tag = f"#{location.strip().replace(' ', '')}"
        if location_tag not in layer_local:
            layer_local = [location_tag] + layer_local
    # Add seasonal local if empty
    if not layer_local:
        layer_local = FALLBACK_POOLS["seasonal"].get(season, FALLBACK_POOLS["local"])

    # Layer 3: Micro1 (Audience 1)
    layer_micro1 = (
        niche_data.get("micro1")
        or FALLBACK_POOLS["micro1"]
    )

    # Layer 4: Micro2 (Audience 2)
    layer_micro2 = (
        niche_data.get("micro2")
        or FALLBACK_POOLS["micro2"]
    )

    # Layer 5: Micro3 (Audience 3)
    layer_micro3 = (
        niche_data.get("micro3")
        or FALLBACK_POOLS["micro3"]
    )

    # Layer 6: Creator/UGC
    layer_creator = (
        niche_data.get("creator")
        or FALLBACK_POOLS["creator"]
    )

    # Layer 7: Trend (season/year)
    layer_trend = (
        niche_data.get("trend")
        or [f"#{season.title()}2026"]
    )

    # Merge YAML trending + community into layers if available
    if yaml_hashtags.get("trending"):
        layer_broad = list(dict.fromkeys(layer_broad + yaml_hashtags["trending"]))
    if yaml_hashtags.get("community"):
        layer_micro1 = list(dict.fromkeys(layer_micro1 + yaml_hashtags["community"]))
    if yaml_hashtags.get("viral"):
        layer_trend = list(dict.fromkeys(layer_trend + yaml_hashtags["viral"]))

    # Add custom tags to broad layer
    if custom_tags:
        layer_broad = custom_tags + layer_broad

    # ── Combine and deduplicate ──
    all_layers = [layer_broad, layer_local, layer_micro1, layer_micro2, layer_micro3, layer_creator, layer_trend]
    all_tags = []
    seen = set()
    for layer in all_layers:
        for tag in (layer or []):
            tag_clean = tag if tag.startswith("#") else f"#{tag}"
            if tag_clean.lower() not in seen:
                seen.add(tag_clean.lower())
                all_tags.append(tag_clean)

    # Trim to platform limit
    combined = all_tags[:limit]

    # ── Highest search (5-cap strategy) ──
    highest_search = niche_data.get("highest_search", [])
    if not highest_search:
        # Build from broad + trending
        highest_search = (layer_broad[:3] + layer_local[:1] + layer_micro1[:1])[:5]

    result = {
        "layers": {
            "broad": layer_broad,
            "local": layer_local,
            "micro1": layer_micro1,
            "micro2": layer_micro2,
            "micro3": layer_micro3,
            "creator": layer_creator,
            "trend": layer_trend,
        },
        "combined": combined,
        "highest_search": highest_search[:5],
        "total": len(combined),
        "platform": effective_platform,
        "limit": limit,
        "niche": niche,
        "season": season,
    }

    logger.info(
        "hashtag.generated",
        niche=niche,
        platform=effective_platform,
        count=len(combined),
        layers=7,
    )
    return result


def generate_5cap(niche: str, location: Optional[str] = None) -> list[str]:
    """
    Instagram "5 highest search hashtags" strategy.
    Returns exactly 5 hashtags optimized for search volume.
    """
    result = generate_hashtag_matrix(niche, platform="instagram_5cap", location=location, use_5cap=True)
    return result.get("highest_search", result.get("combined", []))[:5]


def generate_micro_niche_5(
    niche: str,
    platform: str = "instagram",
    location: Optional[str] = None,
    topic_keywords: list = None,
) -> dict:
    """
    Generate exactly 5 micro-niche hashtags — NOT broad generic tags.

    Strategy (from Gumloop training docs):
      - YouTube chuẩn algo: 3-5 hashtags (first 3 show on title)
      - Instagram hướng cap 5 per post
      - TikTok: 3-5 micro-niche + #fyp optional

    How it picks 5:
      1. Start with highest_search from DB (pre-curated high-volume)
      2. Fill from micro1/micro2/micro3 layers (audience-specific)
      3. Add 1 trending/seasonal tag if room
      4. NEVER use generic broad (#viral, #trending, #fyp)
      5. ALWAYS specific to the niche topic

    Returns:
        dict with 'hashtags' (list of 5 str), 'strategy', 'platform'
    """
    db = _load_niche_db()
    niche_data = db.get(niche, {})
    yaml_hashtags = _find_sub_niche_hashtags(niche) if not niche_data else {}
    season = _get_season()

    # Candidate pools — micro-niche specific, NOT broad
    pool = []
    seen = set()
    curated_count = 0  # Track how many from DB vs keyword extraction

    def _add(tags, is_curated=False):
        nonlocal curated_count
        for t in (tags or []):
            tag = t if t.startswith("#") else f"#{t}"
            if tag.lower() not in seen:
                seen.add(tag.lower())
                pool.append(tag)
                if is_curated:
                    curated_count += 1

    # Priority 1: highest_search (curated high-volume niche tags)
    _add(niche_data.get("highest_search", []), is_curated=True)

    # Priority 2: micro audience layers (audience-specific)
    _add(niche_data.get("micro1", []), is_curated=True)
    _add(niche_data.get("micro2", []), is_curated=True)
    _add(niche_data.get("micro3", []), is_curated=True)

    # Priority 3: YAML niche-specific tags
    _add(yaml_hashtags.get("niche", []), is_curated=True)
    _add(yaml_hashtags.get("trending", []), is_curated=True)
    _add(yaml_hashtags.get("community", []), is_curated=True)

    # Priority 4: Topic-specific dynamic tags from keywords
    if topic_keywords:
        for kw in topic_keywords[:3]:
            tag = "#" + kw.strip().replace(" ", "").replace("-", "").title()
            if tag.lower() not in seen:
                seen.add(tag.lower())
                pool.append(tag)

    # Priority 5: 1 seasonal trend tag
    trend_tag = f"#{season.title()}2026"
    if trend_tag.lower() not in seen:
        pool.append(trend_tag)

    # Filter out generic/broad tags that don't help algo
    GENERIC_BLACKLIST = {
        "#viral", "#trending", "#fyp", "#foryou", "#foryoupage",
        "#explore", "#explorepage", "#follow", "#like", "#instagood",
        "#love", "#photooftheday", "#reels", "#tiktok",
    }
    pool = [t for t in pool if t.lower() not in GENERIC_BLACKLIST]

    # Take exactly 5
    selected = pool[:5]

    # If we still don't have 5, pad with niche broad (less ideal but niche-specific)
    if len(selected) < 5:
        broad = niche_data.get("broad", [])
        broad = [b if b.startswith("#") else f"#{b}" for b in broad]
        broad = [b for b in broad if b.lower() not in seen and b.lower() not in GENERIC_BLACKLIST]
        selected.extend(broad[:5 - len(selected)])

    logger.info("hashtag.micro_niche_5",
                niche=niche, platform=platform,
                count=len(selected), tags=selected)

    return {
        "hashtags": selected[:5],
        "count": len(selected[:5]),
        "curated_count": curated_count,
        "strategy": "micro_niche_5",
        "platform": platform,
        "niche": niche,
        "season": season,
        "note": "5 micro-niche hashtags only — no generic broad tags",
    }


def get_available_niches() -> list[str]:
    """List all available niches in the database."""
    db = _load_niche_db()
    return list(db.keys())


def get_niche_info(niche: str) -> dict:
    """Get full hashtag data for a niche."""
    db = _load_niche_db()
    return db.get(niche, {})


# ── Backward compatibility ──
# Keep old function name working for existing code
def generate_5layer_matrix(niche: str, platform: str = "instagram", **kwargs) -> dict:
    """Backward-compatible wrapper. Calls 7-layer internally."""
    return generate_hashtag_matrix(niche, platform, **kwargs)
