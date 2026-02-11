"""
RSS Auto Poster — Sendible-style auto-publish from RSS feeds.

Inspired by Sendible's RSS Auto Poster (app.sendible.com) but:
  ✅ Posts: title + description + hashtags + image/video ONLY (NO links)
  ✅ Image content → auto-overlay TikTok music → post as video
  ✅ Video content → post as-is (already has music)
  ✅ AI-generated hashtags per post (5 micro-niche)
  ✅ Content rewrite via content factory (GenAI filler stripped)
  ✅ Per-platform adaptation (char limits, thread splits, etc.)

Sendible settings we replicate:
  - Selected feed → any RSS feed registered in rss_reader
  - Post to → target social accounts (multi-platform)
  - Update frequency → check interval (15min/30min/1h/2h/4h/12h/24h)
  - Publish as → scheduled / draft / queued
  - Number of entries to publish → 1-5 per check cycle
  - Alert when posts published → webhook / log
  - Current state → active / paused
  - Types of posts → any / new_only
  - Don't Repeat / Posts Can Repeat
  - Posting sequence → newest_first / random
  - Prefix / Suffix per post
  - Filters → include/exclude keywords

What we DON'T do (by user spec):
  ❌ No links pasted into posts
  ❌ No "shorten links" (we don't include links at all)
"""

import os
import json
import hashlib
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field, asdict

import structlog

logger = structlog.get_logger()

# ── Persistence ──
AUTO_POSTER_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "rss_auto_poster_configs.json"
)
AUTO_POSTER_HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "rss_auto_poster_history.json"
)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "viralops.db")

# ── Frequency presets (in minutes) ──
FREQUENCY_PRESETS = {
    "every_15min": 15,
    "every_30min": 30,
    "every_hour": 60,
    "every_2hours": 120,
    "every_4hours": 240,
    "every_12hours": 720,
    "every_24hours": 1440,
}


# ════════════════════════════════════════════════
# Auto Poster Config (mirrors Sendible's RSS Auto Poster dialog)
# ════════════════════════════════════════════════

@dataclass
class AutoPosterConfig:
    """
    One RSS Auto Poster rule — equivalent to one Sendible RSS Auto Poster config.

    Maps 1:1 with Sendible's dialog fields:
      - selected_feed → feed_id (references rss_reader.py feeds)
      - post_to → target_platforms + target_accounts
      - update_frequency → frequency ("every_hour", etc.)
      - publish_as → publish_mode ("scheduled" / "draft" / "queued")
      - number_of_entries → entries_per_cycle (1-5)
      - alert_when_published → alert_enabled
      - current_state → state ("active" / "paused")
      - types_of_posts → post_filter ("any" / "new_only")
      - repeat_posts → allow_repeat
      - posting_sequence → sequence ("newest_first" / "random")
      - prefix_each_post → prefix
      - suffix_each_post → suffix
      - filters → include_keywords / exclude_keywords
    """
    id: str = ""
    name: str = ""

    # Feed source
    feed_id: str = ""                       # References rss_reader feed ID
    feed_url: str = ""                      # Direct URL (alternative to feed_id)

    # Where to post
    target_platforms: list = field(default_factory=lambda: ["tiktok"])
    target_accounts: dict = field(default_factory=dict)  # platform → account_id

    # Timing
    frequency: str = "every_hour"           # Key from FREQUENCY_PRESETS
    publish_mode: str = "scheduled"         # scheduled / draft / queued
    entries_per_cycle: int = 1              # 1-5 entries per check

    # Behavior
    state: str = "active"                   # active / paused
    post_filter: str = "new_only"           # any / new_only
    allow_repeat: bool = False              # Don't Repeat / Posts Can Repeat
    sequence: str = "newest_first"          # newest_first / random

    # Content customization
    prefix: str = ""                        # Prepend to each post
    suffix: str = ""                        # Append to each post
    include_keywords: list = field(default_factory=list)   # Only post if contains
    exclude_keywords: list = field(default_factory=list)   # Skip if contains

    # Content settings (user spec: title + description + hashtags + image/video, NO links)
    include_title: bool = True
    include_description: bool = True
    include_hashtags: bool = True
    include_media: bool = True              # image or video
    include_links: bool = False             # ALWAYS False per user spec
    tiktok_music_enabled: bool = True       # Auto-add TikTok music to images

    # Niche (for hashtag generation + TikTok music matching)
    niche: str = "sustainable-living"

    # Alert
    alert_enabled: bool = True

    # Tracking
    last_checked: str = ""
    last_posted: str = ""
    total_posted: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ════════════════════════════════════════════════
# Config CRUD
# ════════════════════════════════════════════════

def _load_configs() -> list[dict]:
    """Load all auto poster configs."""
    try:
        if os.path.exists(AUTO_POSTER_CONFIG_FILE):
            with open(AUTO_POSTER_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_configs(configs: list[dict]):
    """Save auto poster configs."""
    os.makedirs(os.path.dirname(AUTO_POSTER_CONFIG_FILE), exist_ok=True)
    with open(AUTO_POSTER_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)


def _load_history() -> dict:
    """Load posted entry history (for dedup / repeat tracking)."""
    try:
        if os.path.exists(AUTO_POSTER_HISTORY_FILE):
            with open(AUTO_POSTER_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_history(history: dict):
    """Save posted entry history."""
    os.makedirs(os.path.dirname(AUTO_POSTER_HISTORY_FILE), exist_ok=True)
    with open(AUTO_POSTER_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def create_auto_poster(config: dict) -> dict:
    """
    Create a new RSS Auto Poster config.

    Example:
        create_auto_poster({
            "name": "TheRike → TikTok Auto",
            "feed_id": "abc123",
            "target_platforms": ["tiktok", "instagram"],
            "frequency": "every_hour",
            "entries_per_cycle": 1,
            "publish_mode": "scheduled",
            "niche": "sustainable-living",
        })
    """
    configs = _load_configs()

    poster_id = hashlib.md5(
        (config.get("feed_id", "") + str(config.get("target_platforms", [])) + datetime.utcnow().isoformat()).encode()
    ).hexdigest()[:12]

    new_config = asdict(AutoPosterConfig())
    new_config.update(config)
    new_config["id"] = poster_id
    new_config["include_links"] = False  # ALWAYS False per user spec
    new_config["created_at"] = datetime.utcnow().isoformat()

    if not new_config.get("name"):
        new_config["name"] = f"Auto Poster {poster_id[:6]}"

    configs.append(new_config)
    _save_configs(configs)

    logger.info("rss_auto_poster.created", id=poster_id, name=new_config["name"],
                feed_id=new_config.get("feed_id"), platforms=new_config.get("target_platforms"))
    return {"success": True, "config": new_config}


def update_auto_poster(poster_id: str, updates: dict) -> dict:
    """Update an existing auto poster config."""
    configs = _load_configs()
    for cfg in configs:
        if cfg.get("id") == poster_id:
            # Never allow links
            updates.pop("include_links", None)
            cfg.update(updates)
            _save_configs(configs)
            logger.info("rss_auto_poster.updated", id=poster_id)
            return {"success": True, "config": cfg}
    return {"error": "Auto poster not found"}


def delete_auto_poster(poster_id: str) -> dict:
    """Delete an auto poster config."""
    configs = _load_configs()
    before = len(configs)
    configs = [c for c in configs if c.get("id") != poster_id]
    _save_configs(configs)
    return {"success": before > len(configs), "deleted": before - len(configs)}


def list_auto_posters() -> list[dict]:
    """List all auto poster configs."""
    return _load_configs()


def get_auto_poster(poster_id: str) -> Optional[dict]:
    """Get a single auto poster config."""
    for cfg in _load_configs():
        if cfg.get("id") == poster_id:
            return cfg
    return None


def pause_auto_poster(poster_id: str) -> dict:
    """Pause an auto poster."""
    return update_auto_poster(poster_id, {"state": "paused"})


def activate_auto_poster(poster_id: str) -> dict:
    """Activate a paused auto poster."""
    return update_auto_poster(poster_id, {"state": "active"})


# ════════════════════════════════════════════════
# Core Engine — Fetch → Filter → Build → Publish
# ════════════════════════════════════════════════

def _passes_keyword_filter(entry: dict, config: dict) -> bool:
    """Check if an RSS entry passes the include/exclude keyword filters."""
    title = (entry.get("title", "") or "").lower()
    body = (entry.get("body", "") or "").lower()
    text = f"{title} {body}"

    # Include filter: entry MUST contain at least one keyword
    include_kw = config.get("include_keywords", [])
    if include_kw:
        if not any(kw.lower() in text for kw in include_kw):
            return False

    # Exclude filter: entry MUST NOT contain any keyword
    exclude_kw = config.get("exclude_keywords", [])
    if exclude_kw:
        if any(kw.lower() in text for kw in exclude_kw):
            return False

    return True


def _get_entry_hash(entry: dict, config_id: str) -> str:
    """Generate unique hash for an entry+config combo (for dedup)."""
    key = f"{config_id}:{entry.get('url', '')}:{entry.get('title', '')}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _build_post_content(entry: dict, config: dict) -> dict:
    """
    Build the final post content from an RSS entry.

    Per user spec:
      ✅ title + description + hashtags + image/video
      ❌ NO links
      ✅ Image on TikTok → add music overlay
      ✅ Video → post as-is
    """
    title = entry.get("title", "")
    body = entry.get("body", entry.get("excerpt", ""))
    image_url = entry.get("image_url", "")
    tags = entry.get("tags", [])

    # Apply prefix/suffix
    prefix = config.get("prefix", "")
    suffix = config.get("suffix", "")

    # Build description (NO links)
    description = body
    if prefix:
        description = f"{prefix}\n\n{description}"
    if suffix:
        description = f"{description}\n\n{suffix}"

    # Generate 5 micro-niche hashtags
    hashtags = []
    if config.get("include_hashtags", True):
        try:
            from agents.matrix_5layer import generate_micro_niche_5
            niche = config.get("niche", "sustainable-living")
            topic_keywords = [w for w in title.lower().replace("-", " ").split() if len(w) > 3]
            hashtag_result = generate_micro_niche_5(
                niche_key=niche,
                platform=config.get("target_platforms", ["tiktok"])[0],
                topic_keywords=topic_keywords,
            )
            hashtags = hashtag_result.get("hashtags", [])
        except Exception as e:
            logger.warning("rss_auto_poster.hashtag_error", error=str(e))
            # Fallback: convert RSS tags to hashtags
            hashtags = [f"#{t.replace(' ', '').lower()}" for t in tags[:5] if t]

    # Determine media type
    media_type = "none"
    media_url = ""
    if config.get("include_media", True) and image_url:
        # Check if URL is video or image
        video_extensions = ('.mp4', '.mov', '.avi', '.webm', '.mkv')
        if any(image_url.lower().endswith(ext) for ext in video_extensions):
            media_type = "video"
        else:
            media_type = "image"
        media_url = image_url

    # TikTok music for image content
    tiktok_music = None
    if (media_type == "image"
            and config.get("tiktok_music_enabled", True)
            and "tiktok" in config.get("target_platforms", [])):
        try:
            from integrations.tiktok_music import recommend_music
            niche = config.get("niche", "sustainable-living")
            music_result = recommend_music(text=f"{title} {body[:200]}", niche=niche, limit=1)
            if music_result.get("tracks"):
                tiktok_music = music_result["tracks"][0]
                logger.info("rss_auto_poster.tiktok_music_selected",
                            track=tiktok_music.get("title"), mood=music_result.get("mood_detected"))
        except Exception as e:
            logger.warning("rss_auto_poster.tiktok_music_error", error=str(e))

    return {
        "title": title if config.get("include_title", True) else "",
        "description": description if config.get("include_description", True) else "",
        "hashtags": hashtags,
        "media_type": media_type,      # "image" / "video" / "none"
        "media_url": media_url,
        "tiktok_music": tiktok_music,   # Track dict for TikTok image posts
        "source_url": entry.get("url", ""),  # Internal ref only, NOT included in post
        "source_entry_id": entry.get("id", ""),
    }


def _adapt_post_for_platform(post_content: dict, platform: str) -> dict:
    """
    Adapt the post content for a specific platform using content_factory.adapt_for_platform.

    Returns platform-ready content dict.
    """
    try:
        from agents.content_factory import adapt_for_platform, smart_truncate

        # Build a content_pack dict that adapt_for_platform expects
        pack = {
            "title": post_content.get("title", ""),
            "body": post_content.get("description", ""),
            "hook": post_content.get("title", ""),
            "hashtags": post_content.get("hashtags", []),
            "image_prompt": "",
            "content_type": "curated",
        }

        adapted = adapt_for_platform(pack, platform)

        # Merge with media info
        adapted["media_type"] = post_content.get("media_type", "none")
        adapted["media_url"] = post_content.get("media_url", "")

        # TikTok-specific: attach music for image posts
        if platform == "tiktok" and post_content.get("tiktok_music"):
            adapted["tiktok_music"] = post_content["tiktok_music"]
            # If image, we'll create a slideshow video with music
            if post_content.get("media_type") == "image":
                adapted["media_type"] = "image_with_music"
                adapted["music_track_id"] = post_content["tiktok_music"].get("track_id", "")
                adapted["music_title"] = post_content["tiktok_music"].get("title", "")

        return adapted

    except Exception as e:
        logger.error("rss_auto_poster.adapt_error", platform=platform, error=str(e))
        # Fallback: return raw content
        return {
            "platform": platform,
            "title": post_content.get("title", "")[:200],
            "caption": post_content.get("description", "")[:2200],
            "hashtags": post_content.get("hashtags", []),
            "media_type": post_content.get("media_type", "none"),
            "media_url": post_content.get("media_url", ""),
        }


def _save_post_to_db(post_content: dict, adapted: dict, config: dict) -> Optional[int]:
    """Save a post to the SQLite database (as scheduled/draft/queued)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT, body TEXT, category TEXT,
                platforms TEXT, status TEXT DEFAULT 'draft',
                scheduled_at TEXT, published_at TEXT,
                extra_fields TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        platforms = json.dumps(config.get("target_platforms", []))
        status = config.get("publish_mode", "draft")

        # Schedule time: next optimal slot if scheduled
        scheduled_at = None
        if status == "scheduled":
            scheduled_at = (datetime.utcnow() + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M")

        extra = {
            "source": "rss_auto_poster",
            "auto_poster_id": config.get("id", ""),
            "media_type": post_content.get("media_type", "none"),
            "media_url": post_content.get("media_url", ""),
            "hashtags": post_content.get("hashtags", []),
            "source_entry_id": post_content.get("source_entry_id", ""),
        }

        # Add TikTok music info if present
        if post_content.get("tiktok_music"):
            extra["tiktok_music"] = {
                "track_id": post_content["tiktok_music"].get("track_id", ""),
                "title": post_content["tiktok_music"].get("title", ""),
                "artist": post_content["tiktok_music"].get("artist", ""),
                "mood": post_content["tiktok_music"].get("mood", ""),
            }

        # Per-platform adapted content
        extra["adapted_content"] = {
            adapted.get("platform", "default"): {
                "caption": adapted.get("caption", ""),
                "title": adapted.get("title", ""),
                "hashtag_count": adapted.get("hashtag_count", 0),
            }
        }

        # Build post body: title + description + hashtags (NO links)
        hashtag_str = " ".join(post_content.get("hashtags", []))
        body_parts = []
        if post_content.get("title"):
            body_parts.append(post_content["title"])
        if post_content.get("description"):
            body_parts.append(post_content["description"])
        if hashtag_str:
            body_parts.append(hashtag_str)
        body = "\n\n".join(body_parts)

        cursor = conn.execute(
            "INSERT INTO posts (title, body, category, platforms, status, scheduled_at, extra_fields) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (post_content.get("title", ""), body, "rss_auto",
             platforms, status, scheduled_at, json.dumps(extra, ensure_ascii=False))
        )
        post_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info("rss_auto_poster.post_saved", post_id=post_id, status=status,
                     title=post_content.get("title", "")[:50])
        return post_id

    except Exception as e:
        logger.error("rss_auto_poster.db_error", error=str(e))
        return None


# ════════════════════════════════════════════════
# Main Tick — Called by Scheduler
# ════════════════════════════════════════════════

def tick(poster_id: str = None) -> dict:
    """
    Execute one cycle of the RSS Auto Poster.

    If poster_id is given, run only that poster.
    Otherwise, run ALL active posters that are due.

    Called by the scheduler every minute — each poster checks its own frequency.

    Returns summary of actions taken.
    """
    configs = _load_configs()
    history = _load_history()
    now = datetime.utcnow()
    results = []

    posters_to_run = configs
    if poster_id:
        posters_to_run = [c for c in configs if c.get("id") == poster_id]

    for config in posters_to_run:
        cfg_id = config.get("id", "")

        # Skip paused
        if config.get("state") != "active":
            continue

        # Check frequency — is it time to run?
        freq_key = config.get("frequency", "every_hour")
        freq_minutes = FREQUENCY_PRESETS.get(freq_key, 60)
        last_checked = config.get("last_checked", "")

        if last_checked:
            try:
                last_dt = datetime.fromisoformat(last_checked)
                if (now - last_dt).total_seconds() < freq_minutes * 60:
                    continue  # Not time yet
            except (ValueError, TypeError):
                pass  # Invalid date, run anyway

        logger.info("rss_auto_poster.tick", id=cfg_id, name=config.get("name"))

        # Fetch RSS entries
        try:
            from integrations.rss_reader import fetch_feed
            feed_result = fetch_feed(
                feed_id=config.get("feed_id"),
                feed_url=config.get("feed_url"),
                max_entries=20,  # Fetch more, we'll filter down
            )
        except Exception as e:
            results.append({"poster_id": cfg_id, "error": str(e), "posted": 0})
            continue

        if not feed_result.get("success"):
            results.append({"poster_id": cfg_id, "error": feed_result.get("error"), "posted": 0})
            continue

        entries = feed_result.get("entries", [])

        # Filter: new_only → skip already-posted entries
        poster_history = history.get(cfg_id, {})
        if config.get("post_filter") == "new_only" or not config.get("allow_repeat", False):
            entries = [
                e for e in entries
                if _get_entry_hash(e, cfg_id) not in poster_history
            ]

        # Filter: keyword include/exclude
        entries = [e for e in entries if _passes_keyword_filter(e, config)]

        # Sequence: newest_first or random
        if config.get("sequence") == "random":
            random.shuffle(entries)
        # else: newest_first — feedparser already returns newest first

        # Limit to entries_per_cycle
        max_entries = min(config.get("entries_per_cycle", 1), 5)
        entries = entries[:max_entries]

        if not entries:
            # Update last_checked even if nothing to post
            _update_config_field(cfg_id, "last_checked", now.isoformat())
            results.append({"poster_id": cfg_id, "posted": 0, "reason": "no_new_entries"})
            continue

        # Process each entry
        posted_count = 0
        for entry in entries:
            # Build post content (title + description + hashtags + media, NO links)
            post_content = _build_post_content(entry, config)

            # ── Media Processing: Image → Video + TikTok Music ──
            if (post_content.get("media_type") == "image"
                    and config.get("tiktok_music_enabled", True)
                    and "tiktok" in config.get("target_platforms", [])):
                try:
                    from integrations.media_processor import process_image_to_video
                    video_result = process_image_to_video(
                        image_url=post_content.get("media_url"),
                        music_track=post_content.get("tiktok_music"),
                        niche=config.get("niche", "sustainable-living"),
                        content_text=f"{post_content.get('title', '')} {post_content.get('description', '')[:200]}",
                    )
                    if video_result.get("success") and video_result.get("video_path"):
                        post_content["media_type"] = "video"
                        post_content["media_url"] = video_result["video_path"]
                        post_content["video_generated"] = True
                        logger.info("rss_auto_poster.video_created",
                                    video=video_result["video_path"],
                                    has_music=bool(video_result.get("music")))
                except Exception as e:
                    logger.warning("rss_auto_poster.media_process_error", error=str(e))

            # Adapt for each target platform
            platforms = config.get("target_platforms", ["tiktok"])
            for platform in platforms:
                adapted = _adapt_post_for_platform(post_content, platform)
                post_id = _save_post_to_db(post_content, adapted, config)

                if post_id:
                    posted_count += 1

            # Record in history (for dedup)
            entry_hash = _get_entry_hash(entry, cfg_id)
            if cfg_id not in history:
                history[cfg_id] = {}
            history[cfg_id][entry_hash] = {
                "title": entry.get("title", "")[:100],
                "posted_at": now.isoformat(),
                "platforms": platforms,
            }

        # Update config tracking
        _update_config_field(cfg_id, "last_checked", now.isoformat())
        _update_config_field(cfg_id, "last_posted", now.isoformat())
        _update_config_field(cfg_id, "total_posted",
                             config.get("total_posted", 0) + posted_count)

        results.append({
            "poster_id": cfg_id,
            "name": config.get("name", ""),
            "posted": posted_count,
            "entries_checked": len(feed_result.get("entries", [])),
        })

        if config.get("alert_enabled", True) and posted_count > 0:
            logger.info("rss_auto_poster.alert",
                        poster=config.get("name"), count=posted_count,
                        platforms=platforms)
            # Telegram alert
            try:
                from integrations.telegram_bot import alert_post_created
                alert_post_created(
                    poster_name=config.get("name", cfg_id),
                    title=entries[0].get("title", "")[:80] if entries else "",
                    platforms=platforms,
                    media_type=post_content.get("media_type", "none") if entries else "none",
                    has_music=bool(post_content.get("tiktok_music")) if entries else False,
                    post_count=posted_count,
                )
            except Exception as e:
                logger.warning("rss_auto_poster.telegram_error", error=str(e))

    # Save history
    _save_history(history)

    total_posted = sum(r.get("posted", 0) for r in results)
    logger.info("rss_auto_poster.tick_done", posters_run=len(results), total_posted=total_posted)

    # Telegram tick summary
    if total_posted > 0:
        try:
            from integrations.telegram_bot import alert_tick_summary
            alert_tick_summary(
                posters_checked=len(results),
                total_posted=total_posted,
                results=results,
            )
        except Exception as e:
            logger.warning("rss_auto_poster.telegram_summary_error", error=str(e))

    return {
        "success": True,
        "posters_checked": len(results),
        "total_posted": total_posted,
        "results": results,
    }


def _update_config_field(poster_id: str, field_name: str, value):
    """Update a single field in a config (without reloading all)."""
    configs = _load_configs()
    for cfg in configs:
        if cfg.get("id") == poster_id:
            cfg[field_name] = value
            break
    _save_configs(configs)


# ════════════════════════════════════════════════
# Status & Stats
# ════════════════════════════════════════════════

def get_auto_poster_status() -> dict:
    """Get overview status of all auto posters."""
    configs = _load_configs()
    active = sum(1 for c in configs if c.get("state") == "active")
    paused = sum(1 for c in configs if c.get("state") == "paused")
    total_posted = sum(c.get("total_posted", 0) for c in configs)

    return {
        "total_posters": len(configs),
        "active": active,
        "paused": paused,
        "total_posts_created": total_posted,
        "posters": [
            {
                "id": c.get("id"),
                "name": c.get("name"),
                "state": c.get("state"),
                "frequency": c.get("frequency"),
                "platforms": c.get("target_platforms"),
                "total_posted": c.get("total_posted", 0),
                "last_checked": c.get("last_checked"),
                "last_posted": c.get("last_posted"),
            }
            for c in configs
        ],
    }


def get_poster_history(poster_id: str, limit: int = 50) -> list[dict]:
    """Get posting history for a specific auto poster."""
    history = _load_history()
    poster_history = history.get(poster_id, {})

    items = []
    for entry_hash, data in poster_history.items():
        items.append({
            "entry_hash": entry_hash,
            "title": data.get("title", ""),
            "posted_at": data.get("posted_at", ""),
            "platforms": data.get("platforms", []),
        })

    # Sort by posted_at descending
    items.sort(key=lambda x: x.get("posted_at", ""), reverse=True)
    return items[:limit]


def clear_poster_history(poster_id: str) -> dict:
    """Clear posting history for a poster (allows re-posting)."""
    history = _load_history()
    removed = len(history.get(poster_id, {}))
    history.pop(poster_id, None)
    _save_history(history)
    return {"success": True, "entries_cleared": removed}
