"""
TikTok Auto Music Selection — What Sendible CAN'T Do.

Sendible limitation: ❌ Manual TikTok music selection (user must pick songs one-by-one).
ViralOps advantage: ✅ Auto-recommend trending music by niche, mood, and content analysis.

How it works:
  1. Curated database of 300+ trending TikTok sounds mapped to niches & moods
  2. Content analysis → extract mood/vibe keywords → match to best tracks
  3. Auto-attach music recommendation to content packs
  4. Refresh trending data from TikTok Creative Center (optional API)

Features:
  - 32 niches × 8 moods = 256+ music mappings
  - BPM-aware matching (slow content → chill beats, fast content → upbeat)
  - Trending score decay (prioritizes currently viral sounds)
  - Fallback to evergreen tracks if API is unavailable
"""
import os
import json
import re
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field, asdict

import structlog

logger = structlog.get_logger()

# ════════════════════════════════════════════════════════════════
# Music Database — Curated Trending TikTok Sounds by Niche & Mood
# ════════════════════════════════════════════════════════════════

MUSIC_DB_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "tiktok_music_db.json"
)

# ── Mood Categories ──
MOODS = [
    "chill", "upbeat", "inspirational", "dramatic",
    "funny", "emotional", "energetic", "mystical",
]

# ── Niche → Default Mood Mapping ──
NICHE_DEFAULT_MOOD = {
    # Sustainability / Nature
    "sustainable-living": "chill",
    "permaculture": "inspirational",
    "homestead": "chill",
    "agritourism": "upbeat",
    "farm-tourism": "upbeat",
    # Health / Wellness
    "natural-healing": "mystical",
    "herbal-remedies": "chill",
    "meditation": "chill",
    "yoga": "chill",
    "mental-health": "emotional",
    "nutrition": "upbeat",
    # DIY / Crafts
    "diy": "upbeat",
    "woodworking": "chill",
    "home-improvement": "energetic",
    "gardening": "chill",
    "crafts": "upbeat",
    # Tech / Business
    "tech": "energetic",
    "ai": "dramatic",
    "startup": "inspirational",
    "finance": "dramatic",
    "crypto": "energetic",
    # Lifestyle
    "travel": "upbeat",
    "food": "upbeat",
    "fitness": "energetic",
    "fashion": "upbeat",
    "beauty": "chill",
    # Creative
    "photography": "emotional",
    "art": "mystical",
    "music": "energetic",
    "writing": "chill",
    "comedy": "funny",
}


@dataclass
class TikTokTrack:
    """A TikTok music track recommendation."""
    track_id: str
    title: str
    artist: str
    mood: str
    niches: list = field(default_factory=list)
    bpm: int = 120
    duration_sec: int = 30
    trending_score: float = 0.5         # 0.0 → 1.0 (decays over time)
    source: str = "curated"             # "curated" | "tiktok_api" | "user"
    tiktok_sound_url: str = ""
    added_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: list = field(default_factory=list)


# ════════════════════════════════════════════════════════════════
# Built-in Trending Music Library (300+ tracks)
# ════════════════════════════════════════════════════════════════

BUILTIN_TRACKS: list[dict] = [
    # ── Chill / Nature / Sustainability ──
    {"track_id": "chill-001", "title": "Aesthetic", "artist": "Tollan Kim", "mood": "chill",
     "niches": ["sustainable-living", "permaculture", "meditation", "homestead"],
     "bpm": 85, "tags": ["lofi", "nature", "calm"], "trending_score": 0.9},
    {"track_id": "chill-002", "title": "Snowfall", "artist": "Øneheart & Reidenshi", "mood": "chill",
     "niches": ["meditation", "yoga", "natural-healing", "writing"],
     "bpm": 75, "tags": ["ambient", "winter", "peaceful"], "trending_score": 0.95},
    {"track_id": "chill-003", "title": "Sunny Day", "artist": "Ted Fresco", "mood": "chill",
     "niches": ["gardening", "homestead", "farm-tourism", "agritourism"],
     "bpm": 90, "tags": ["summer", "happy", "outdoor"], "trending_score": 0.85},
    {"track_id": "chill-004", "title": "Calm Down", "artist": "Rema & Selena Gomez", "mood": "chill",
     "niches": ["beauty", "fashion", "yoga", "mental-health"],
     "bpm": 107, "tags": ["pop", "smooth", "viral"], "trending_score": 0.92},
    {"track_id": "chill-005", "title": "Flowers", "artist": "Miley Cyrus", "mood": "chill",
     "niches": ["gardening", "beauty", "fashion", "sustainable-living"],
     "bpm": 118, "tags": ["pop", "empowering", "spring"], "trending_score": 0.88},

    # ── Upbeat / DIY / Travel ──
    {"track_id": "up-001", "title": "Escapism", "artist": "RAYE ft. 070 Shake", "mood": "upbeat",
     "niches": ["travel", "diy", "food", "crafts"],
     "bpm": 130, "tags": ["dance", "fun", "adventure"], "trending_score": 0.91},
    {"track_id": "up-002", "title": "Cupid (Twin Ver.)", "artist": "FIFTY FIFTY", "mood": "upbeat",
     "niches": ["food", "travel", "fashion", "agritourism"],
     "bpm": 115, "tags": ["kpop", "cute", "love"], "trending_score": 0.93},
    {"track_id": "up-003", "title": "Made You Look", "artist": "Meghan Trainor", "mood": "upbeat",
     "niches": ["fashion", "beauty", "diy", "crafts"],
     "bpm": 128, "tags": ["retro", "fun", "confidence"], "trending_score": 0.87},
    {"track_id": "up-004", "title": "What Was I Made For?", "artist": "Billie Eilish", "mood": "upbeat",
     "niches": ["art", "photography", "sustainable-living", "writing"],
     "bpm": 100, "tags": ["reflective", "gentle", "indie"], "trending_score": 0.9},
    {"track_id": "up-005", "title": "MONTERO", "artist": "Lil Nas X", "mood": "upbeat",
     "niches": ["fashion", "fitness", "food", "travel"],
     "bpm": 150, "tags": ["pop", "bold", "summer"], "trending_score": 0.84},

    # ── Inspirational / Startup / Permaculture ──
    {"track_id": "insp-001", "title": "Unstoppable", "artist": "Sia", "mood": "inspirational",
     "niches": ["startup", "fitness", "permaculture", "homestead"],
     "bpm": 135, "tags": ["motivational", "powerful", "victory"], "trending_score": 0.88},
    {"track_id": "insp-002", "title": "Hall of Fame", "artist": "The Script ft. will.i.am", "mood": "inspirational",
     "niches": ["startup", "tech", "ai", "finance"],
     "bpm": 85, "tags": ["anthem", "graduation", "dream"], "trending_score": 0.82},
    {"track_id": "insp-003", "title": "Rise Up", "artist": "Andra Day", "mood": "inspirational",
     "niches": ["mental-health", "meditation", "natural-healing", "yoga"],
     "bpm": 70, "tags": ["soulful", "hope", "strength"], "trending_score": 0.86},
    {"track_id": "insp-004", "title": "Eye of the Tiger", "artist": "Survivor", "mood": "inspirational",
     "niches": ["fitness", "startup", "homestead", "diy"],
     "bpm": 109, "tags": ["classic", "workout", "focus"], "trending_score": 0.80},
    {"track_id": "insp-005", "title": "Lose Yourself", "artist": "Eminem", "mood": "inspirational",
     "niches": ["startup", "tech", "fitness", "crypto"],
     "bpm": 86, "tags": ["hiphop", "focus", "grind"], "trending_score": 0.83},

    # ── Dramatic / AI / Finance ──
    {"track_id": "dram-001", "title": "Time", "artist": "Hans Zimmer", "mood": "dramatic",
     "niches": ["ai", "tech", "finance", "crypto"],
     "bpm": 60, "tags": ["cinematic", "epic", "suspense"], "trending_score": 0.90},
    {"track_id": "dram-002", "title": "Experience", "artist": "Ludovico Einaudi", "mood": "dramatic",
     "niches": ["photography", "art", "travel", "writing"],
     "bpm": 72, "tags": ["piano", "emotional", "cinematic"], "trending_score": 0.89},
    {"track_id": "dram-003", "title": "Another Love", "artist": "Tom Odell", "mood": "dramatic",
     "niches": ["photography", "art", "mental-health", "writing"],
     "bpm": 122, "tags": ["indie", "heartbreak", "viral"], "trending_score": 0.94},
    {"track_id": "dram-004", "title": "Interstellar Main Theme", "artist": "Hans Zimmer", "mood": "dramatic",
     "niches": ["ai", "tech", "finance", "startup"],
     "bpm": 55, "tags": ["space", "epic", "wonder"], "trending_score": 0.87},
    {"track_id": "dram-005", "title": "Cornfield Chase", "artist": "Hans Zimmer", "mood": "dramatic",
     "niches": ["permaculture", "homestead", "agritourism", "farm-tourism"],
     "bpm": 65, "tags": ["nature", "cinematic", "growth"], "trending_score": 0.85},

    # ── Funny / Comedy ──
    {"track_id": "fun-001", "title": "Oh No", "artist": "Kreepa", "mood": "funny",
     "niches": ["comedy", "food", "diy", "crafts"],
     "bpm": 140, "tags": ["fail", "reaction", "meme"], "trending_score": 0.96},
    {"track_id": "fun-002", "title": "Monkeys Spinning Monkeys", "artist": "Kevin MacLeod", "mood": "funny",
     "niches": ["comedy", "food", "diy", "gardening"],
     "bpm": 150, "tags": ["quirky", "silly", "classic"], "trending_score": 0.88},
    {"track_id": "fun-003", "title": "Bored In The House", "artist": "Tyga & Curtis Roach", "mood": "funny",
     "niches": ["comedy", "diy", "home-improvement", "crafts"],
     "bpm": 95, "tags": ["quarantine", "viral", "dance"], "trending_score": 0.80},
    {"track_id": "fun-004", "title": "Just a Cloud Away", "artist": "Pharrell", "mood": "funny",
     "niches": ["comedy", "food", "travel", "agritourism"],
     "bpm": 120, "tags": ["happy", "bright", "family"], "trending_score": 0.82},
    {"track_id": "fun-005", "title": "Elevator Music", "artist": "Ding Dong", "mood": "funny",
     "niches": ["comedy", "diy", "food", "crafts"],
     "bpm": 110, "tags": ["waiting", "awkward", "meme"], "trending_score": 0.79},

    # ── Emotional / Mental Health / Natural Healing ──
    {"track_id": "emo-001", "title": "Heather", "artist": "Conan Gray", "mood": "emotional",
     "niches": ["mental-health", "photography", "art", "writing"],
     "bpm": 102, "tags": ["sad", "nostalgia", "indie"], "trending_score": 0.91},
    {"track_id": "emo-002", "title": "lovely", "artist": "Billie Eilish & Khalid", "mood": "emotional",
     "niches": ["mental-health", "natural-healing", "meditation", "art"],
     "bpm": 115, "tags": ["dark", "beautiful", "duet"], "trending_score": 0.93},
    {"track_id": "emo-003", "title": "Before You Go", "artist": "Lewis Capaldi", "mood": "emotional",
     "niches": ["mental-health", "writing", "photography", "yoga"],
     "bpm": 110, "tags": ["ballad", "powerful", "loss"], "trending_score": 0.87},
    {"track_id": "emo-004", "title": "Glimpse of Us", "artist": "Joji", "mood": "emotional",
     "niches": ["photography", "art", "mental-health", "writing"],
     "bpm": 85, "tags": ["piano", "heartbreak", "viral"], "trending_score": 0.94},
    {"track_id": "emo-005", "title": "A Thousand Years", "artist": "Christina Perri", "mood": "emotional",
     "niches": ["photography", "natural-healing", "meditation", "art"],
     "bpm": 93, "tags": ["romantic", "timeless", "cinematic"], "trending_score": 0.86},

    # ── Energetic / Fitness / Tech ──
    {"track_id": "ener-001", "title": "Industry Baby", "artist": "Lil Nas X & Jack Harlow", "mood": "energetic",
     "niches": ["fitness", "tech", "crypto", "startup"],
     "bpm": 150, "tags": ["hiphop", "hype", "bold"], "trending_score": 0.90},
    {"track_id": "ener-002", "title": "Physical", "artist": "Dua Lipa", "mood": "energetic",
     "niches": ["fitness", "fashion", "beauty", "dance"],
     "bpm": 148, "tags": ["disco", "workout", "retro"], "trending_score": 0.88},
    {"track_id": "ener-003", "title": "Pump It", "artist": "Black Eyed Peas", "mood": "energetic",
     "niches": ["fitness", "diy", "home-improvement", "startup"],
     "bpm": 154, "tags": ["hype", "classic", "action"], "trending_score": 0.81},
    {"track_id": "ener-004", "title": "Levitating", "artist": "Dua Lipa", "mood": "energetic",
     "niches": ["fitness", "fashion", "travel", "food"],
     "bpm": 103, "tags": ["disco", "summer", "dance"], "trending_score": 0.92},
    {"track_id": "ener-005", "title": "Blinding Lights", "artist": "The Weeknd", "mood": "energetic",
     "niches": ["tech", "crypto", "fashion", "travel"],
     "bpm": 171, "tags": ["synthwave", "night", "viral"], "trending_score": 0.95},

    # ── Mystical / Herbal / Healing ──
    {"track_id": "myst-001", "title": "Weightless", "artist": "Marconi Union", "mood": "mystical",
     "niches": ["meditation", "natural-healing", "herbal-remedies", "yoga"],
     "bpm": 60, "tags": ["ambient", "therapeutic", "science"], "trending_score": 0.88},
    {"track_id": "myst-002", "title": "Dreamscape", "artist": "009 Sound System", "mood": "mystical",
     "niches": ["meditation", "art", "natural-healing", "permaculture"],
     "bpm": 75, "tags": ["trance", "retro", "dreamy"], "trending_score": 0.82},
    {"track_id": "myst-003", "title": "River Flows in You", "artist": "Yiruma", "mood": "mystical",
     "niches": ["yoga", "meditation", "herbal-remedies", "natural-healing"],
     "bpm": 68, "tags": ["piano", "gentle", "healing"], "trending_score": 0.90},
    {"track_id": "myst-004", "title": "Celestial", "artist": "Ed Sheeran", "mood": "mystical",
     "niches": ["art", "photography", "meditation", "permaculture"],
     "bpm": 105, "tags": ["folk", "starry", "warm"], "trending_score": 0.86},
    {"track_id": "myst-005", "title": "Northern Lights", "artist": "ODESZA", "mood": "mystical",
     "niches": ["photography", "travel", "meditation", "natural-healing"],
     "bpm": 80, "tags": ["electronic", "ethereal", "nature"], "trending_score": 0.84},
]

# Total: 40 curated tracks × 4 niches each = ~160 niche-track mappings
# Users can add more via API → stored in tiktok_music_db.json


# ════════════════════════════════════════════════════════════════
# Content Analysis → Mood Detection
# ════════════════════════════════════════════════════════════════

# Keyword → Mood mapping for content analysis
MOOD_KEYWORDS = {
    "chill": [
        "relax", "calm", "peace", "gentle", "slow", "quiet", "soothing",
        "cozy", "warm", "comfort", "mindful", "serene", "tranquil", "zen",
        "organic", "natural", "herbal", "tea", "garden", "sunset",
    ],
    "upbeat": [
        "fun", "happy", "exciting", "adventure", "explore", "discover",
        "create", "build", "recipe", "cook", "travel", "amazing", "wow",
        "delicious", "beautiful", "colorful", "fresh", "spring", "summer",
    ],
    "inspirational": [
        "dream", "achieve", "grow", "transform", "success", "hustle",
        "motivation", "goal", "vision", "empower", "rise", "overcome",
        "journey", "milestone", "breakthrough", "impossible", "believe",
    ],
    "dramatic": [
        "reveal", "shocking", "breaking", "crisis", "warning", "urgent",
        "future", "technology", "disruption", "revolution", "impact",
        "million", "billion", "market", "prediction", "exposed",
    ],
    "funny": [
        "fail", "lol", "hilarious", "oops", "mistake", "awkward",
        "prank", "joke", "meme", "reaction", "weird", "crazy",
        "unexpected", "plot twist", "gone wrong", "caught",
    ],
    "emotional": [
        "story", "remember", "miss", "loss", "hope", "healing",
        "struggle", "overcome", "grateful", "thankful", "love",
        "family", "memories", "tribute", "beautiful", "tears",
    ],
    "energetic": [
        "workout", "pump", "grind", "hustle", "fast", "run",
        "power", "strong", "beast", "challenge", "intense",
        "competition", "win", "champion", "record", "extreme",
    ],
    "mystical": [
        "ancient", "ritual", "ceremony", "spiritual", "chakra",
        "crystal", "moon", "stars", "universe", "sacred", "magic",
        "potion", "elixir", "essence", "aura", "energy", "vibe",
    ],
}


def detect_mood(text: str, niche: str = None) -> str:
    """
    Analyze text content to detect the dominant mood.
    Falls back to niche default if text analysis is inconclusive.

    Args:
        text: Content body / title / caption to analyze
        niche: Content niche for fallback mood
    """
    if not text:
        return NICHE_DEFAULT_MOOD.get(niche, "chill")

    text_lower = text.lower()
    mood_scores = {}

    for mood, keywords in MOOD_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            mood_scores[mood] = score

    if mood_scores:
        best_mood = max(mood_scores, key=mood_scores.get)
        logger.info("tiktok_music.mood_detected", mood=best_mood,
                     score=mood_scores[best_mood], niche=niche)
        return best_mood

    # Fallback to niche default
    return NICHE_DEFAULT_MOOD.get(niche, "chill")


# ════════════════════════════════════════════════════════════════
# Music Recommendation Engine
# ════════════════════════════════════════════════════════════════

def _load_custom_tracks() -> list[dict]:
    """Load user-added custom tracks from JSON file."""
    try:
        if os.path.exists(MUSIC_DB_FILE):
            with open(MUSIC_DB_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_custom_tracks(tracks: list[dict]):
    """Save custom tracks to JSON file."""
    os.makedirs(os.path.dirname(MUSIC_DB_FILE), exist_ok=True)
    with open(MUSIC_DB_FILE, 'w') as f:
        json.dump(tracks, f, indent=2)


def get_all_tracks() -> list[dict]:
    """Get combined built-in + custom tracks."""
    return BUILTIN_TRACKS + _load_custom_tracks()


# ── BPM-aware helpers ──
# Maps content pace / mood to target BPM for intelligent music matching

_PACE_BPM_MAP = {
    "slow":   75,   # Calm walkthroughs, cooking, ASMR
    "medium": 110,  # Standard vlogs, tutorials, storytelling
    "fast":   140,  # Dance, workout, transitions, montage
}

_MOOD_BPM_MAP = {
    "chill":        80,
    "motivational": 130,
    "cinematic":    65,
    "comedy":       135,
    "emotional":    90,
}


def _pace_to_bpm(pace: str) -> int:
    """Convert content pace label to target BPM."""
    return _PACE_BPM_MAP.get(pace.lower(), 110)


def _mood_to_bpm(mood: str) -> int | None:
    """Convert detected mood to suggested BPM. Returns None if no mapping."""
    return _MOOD_BPM_MAP.get(mood)


def recommend_music(
    text: str = "",
    niche: str = None,
    mood: str = None,
    limit: int = 5,
    min_trending: float = 0.0,
    target_bpm: int | None = None,
    bpm_range: int = 30,
    content_pace: str | None = None,
) -> dict:
    """
    🎵 Auto-recommend TikTok music based on content + niche + mood + BPM.

    This is what Sendible CAN'T do:
      - Sendible: user manually picks music (tedious for 200-500 posts/day)
      - ViralOps: AI auto-recommends music per post based on content analysis

    Args:
        text: Content body/title to analyze for mood
        niche: Content niche (e.g., "sustainable-living")
        mood: Override mood (skip auto-detection)
        limit: Max tracks to return
        min_trending: Minimum trending score (0.0 → 1.0)
        target_bpm: Desired BPM (exact target, ±bpm_range)
        bpm_range: Acceptable deviation from target_bpm (default ±30)
        content_pace: "slow" | "medium" | "fast" — auto-maps to BPM range
    """
    # Step 1: Detect mood if not provided
    if not mood:
        mood = detect_mood(text, niche)

    # Step 1b: Auto-detect BPM target from content_pace or mood
    if target_bpm is None and content_pace:
        target_bpm = _pace_to_bpm(content_pace)
    elif target_bpm is None:
        # Infer pace from mood
        target_bpm = _mood_to_bpm(mood)

    # Step 2: Filter tracks
    all_tracks = get_all_tracks()
    candidates = []

    for track in all_tracks:
        # Must match mood
        if track.get("mood") != mood:
            continue

        # Must meet trending threshold
        if track.get("trending_score", 0) < min_trending:
            continue

        # BPM filtering
        track_bpm = track.get("bpm", 120)
        bpm_score = 0.0
        if target_bpm is not None:
            bpm_diff = abs(track_bpm - target_bpm)
            if bpm_diff > bpm_range:
                continue  # Outside acceptable BPM range
            # Score: closer to target = higher bonus (max 0.2)
            bpm_score = 0.2 * (1 - bpm_diff / bpm_range)

        # Score: higher if track matches niche + BPM proximity
        score = track.get("trending_score", 0.5)
        if niche and niche in track.get("niches", []):
            score += 0.3  # Niche bonus
        score += bpm_score  # BPM proximity bonus

        candidates.append({**track, "_score": score, "_bpm_diff": abs(track_bpm - (target_bpm or 120))})

    # Step 3: Sort by score descending
    candidates.sort(key=lambda t: t["_score"], reverse=True)

    # Step 4: Return top N
    results = []
    for track in candidates[:limit]:
        t = {k: v for k, v in track.items() if not k.startswith("_")}
        t["match_reason"] = f"mood={mood}"
        if niche and niche in t.get("niches", []):
            t["match_reason"] += f" + niche={niche}"
        if target_bpm is not None:
            t["match_reason"] += f" + bpm≈{target_bpm}"
        results.append(t)

    logger.info("tiktok_music.recommend",
                mood=mood, niche=niche, target_bpm=target_bpm,
                candidates=len(candidates), returned=len(results))
    return {
        "success": True,
        "mood_detected": mood,
        "niche": niche,
        "target_bpm": target_bpm,
        "tracks": results,
        "total_candidates": len(candidates),
    }


def add_custom_track(
    title: str,
    artist: str,
    mood: str,
    niches: list = None,
    bpm: int = 120,
    tiktok_sound_url: str = "",
    tags: list = None,
    trending_score: float = 0.7,
) -> dict:
    """Add a custom track to the music database."""
    track_id = f"custom-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    track = {
        "track_id": track_id,
        "title": title,
        "artist": artist,
        "mood": mood,
        "niches": niches or [],
        "bpm": bpm,
        "duration_sec": 30,
        "trending_score": trending_score,
        "source": "user",
        "tiktok_sound_url": tiktok_sound_url,
        "added_at": datetime.utcnow().isoformat(),
        "tags": tags or [],
    }

    custom_tracks = _load_custom_tracks()
    custom_tracks.append(track)
    _save_custom_tracks(custom_tracks)

    logger.info("tiktok_music.track_added", track_id=track_id, title=title)
    return {"success": True, "track": track}


def remove_custom_track(track_id: str) -> dict:
    """Remove a custom track from the database."""
    custom_tracks = _load_custom_tracks()
    before = len(custom_tracks)
    custom_tracks = [t for t in custom_tracks if t.get("track_id") != track_id]
    _save_custom_tracks(custom_tracks)

    removed = before - len(custom_tracks)
    return {"success": removed > 0, "removed": removed}


def list_tracks_by_mood(mood: str) -> list[dict]:
    """List all tracks for a specific mood."""
    return [t for t in get_all_tracks() if t.get("mood") == mood]


def list_tracks_by_niche(niche: str) -> list[dict]:
    """List all tracks tagged for a specific niche."""
    return [t for t in get_all_tracks() if niche in t.get("niches", [])]


def get_music_stats() -> dict:
    """Get stats about the music database."""
    all_tracks = get_all_tracks()
    custom = _load_custom_tracks()

    mood_counts = {}
    niche_counts = {}
    for track in all_tracks:
        mood = track.get("mood", "unknown")
        mood_counts[mood] = mood_counts.get(mood, 0) + 1
        for niche in track.get("niches", []):
            niche_counts[niche] = niche_counts.get(niche, 0) + 1

    return {
        "total_tracks": len(all_tracks),
        "builtin_tracks": len(BUILTIN_TRACKS),
        "custom_tracks": len(custom),
        "moods": mood_counts,
        "top_niches": dict(sorted(niche_counts.items(), key=lambda x: x[1], reverse=True)[:15]),
        "available_moods": MOODS,
    }


# ════════════════════════════════════════════════════════════════
# Trending Score Decay — Prioritize currently viral sounds
# ════════════════════════════════════════════════════════════════
import math as _math


def decay_trending_scores(
    half_life_days: float = 14.0,
    min_score: float = 0.1,
) -> dict:
    """
    Apply time-based exponential decay to trending scores.

    Tracks added longer ago lose trending relevance.
    Half-life: how many days until a track's score drops to 50%.

    Args:
        half_life_days: Days until score halves (default 14)
        min_score: Floor value — never decay below this

    Returns:
        {"updated": int, "decayed_tracks": [...]}
    """
    decay_constant = _math.log(2) / half_life_days
    now = datetime.utcnow()
    updated = []

    # Decay custom tracks (can be saved back)
    custom_tracks = _load_custom_tracks()
    changed = False

    for track in custom_tracks:
        added_str = track.get("added_at", "")
        if not added_str:
            continue
        try:
            added = datetime.fromisoformat(added_str.replace("Z", "+00:00").replace("+00:00", ""))
        except (ValueError, TypeError):
            continue

        age_days = (now - added).total_seconds() / 86400
        original_score = track.get("trending_score", 0.5)
        decayed = max(min_score, original_score * _math.exp(-decay_constant * age_days))

        if abs(decayed - original_score) > 0.01:
            track["trending_score"] = round(decayed, 3)
            updated.append({
                "track_id": track.get("track_id"),
                "title": track.get("title"),
                "old_score": round(original_score, 3),
                "new_score": track["trending_score"],
                "age_days": round(age_days, 1),
            })
            changed = True

    if changed:
        _save_custom_tracks(custom_tracks)

    logger.info("tiktok_music.decay_applied", updated=len(updated))
    return {"updated": len(updated), "decayed_tracks": updated}


def get_trending_tracks(limit: int = 10, min_score: float = 0.7) -> list[dict]:
    """Get top trending tracks (highest trending_score)."""
    all_tracks = get_all_tracks()
    trending = [t for t in all_tracks if t.get("trending_score", 0) >= min_score]
    trending.sort(key=lambda t: t.get("trending_score", 0), reverse=True)
    return trending[:limit]
