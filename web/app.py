"""
ViralOps Engine — Web Dashboard (SocialBee-style)
FastAPI + Tailwind CSS + SQLite + Real Publishers

Features:
- Dashboard with stats & channel status
- Content composer with platform-specific fields
- Content library with filter/search
- Calendar view
- Analytics with publish logs
- Settings with channel connections
- RSS Feed Manager (NEW)
- Hashtag Collections (NEW)
- Background Scheduler (NEW)
"""
import os
import sys
import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import structlog

# ── Add project root to path ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.models import PublishResult

logger = structlog.get_logger()

# ── Database ──
DB_PATH = os.path.join(os.path.dirname(__file__), "viralops.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL DEFAULT '',
            body TEXT NOT NULL DEFAULT '',
            category TEXT DEFAULT 'general',
            platforms TEXT DEFAULT '[]',
            status TEXT DEFAULT 'draft',
            scheduled_at TEXT,
            published_at TEXT,
            extra_fields TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS publish_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER,
            platform TEXT,
            success INTEGER,
            post_url TEXT DEFAULT '',
            error TEXT DEFAULT '',
            published_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            color TEXT DEFAULT '#6366f1',
            icon TEXT DEFAULT '📁'
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS hashtag_collections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            hashtags TEXT DEFAULT '[]',
            platforms TEXT DEFAULT '[]',
            niche TEXT DEFAULT 'general',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
    """)

    # Seed categories
    for name, color, icon in [
        ("general", "#6366f1", "📝"), ("blog", "#3b82f6", "📰"),
        ("promotion", "#ef4444", "🔥"), ("evergreen", "#22c55e", "🌲"),
        ("curated", "#f59e0b", "⭐"), ("rss", "#8b5cf6", "📡"),
    ]:
        conn.execute("INSERT OR IGNORE INTO categories (name, color, icon) VALUES (?, ?, ?)", (name, color, icon))

    conn.commit()
    conn.close()


# ── Scheduler background task ──
_scheduler_task = None


async def scheduler_loop():
    """Background scheduler that checks for due posts."""
    from core.scheduler import get_scheduler
    scheduler = get_scheduler()
    while True:
        try:
            results = scheduler.check_and_publish()
            if results:
                logger.info("scheduler.auto_published", count=len(results))
        except Exception as e:
            logger.error("scheduler.error", error=str(e))
        await asyncio.sleep(60)  # Check every minute


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan — init DB and start scheduler."""
    init_db()
    global _scheduler_task
    _scheduler_task = asyncio.create_task(scheduler_loop())
    logger.info("app.started", msg="Dashboard ready at http://localhost:8000")
    yield
    if _scheduler_task:
        _scheduler_task.cancel()


# ── FastAPI App ──
app = FastAPI(title="ViralOps Engine", lifespan=lifespan)

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)


# Custom Jinja2 filter
def fromjson(value):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return value

templates.env.filters["fromjson"] = fromjson


# ════════════════════════════════════════════════════════════════
# HTML PAGES
# ════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("app.html", {"request": request, "page": "dashboard"})

@app.get("/compose", response_class=HTMLResponse)
async def compose(request: Request):
    conn = get_db()
    cats = conn.execute("SELECT * FROM categories").fetchall()
    conn.close()
    return templates.TemplateResponse("app.html", {"request": request, "page": "compose", "categories": cats})

@app.get("/content", response_class=HTMLResponse)
async def content_library(request: Request):
    return templates.TemplateResponse("app.html", {"request": request, "page": "content"})

@app.get("/calendar", response_class=HTMLResponse)
async def calendar(request: Request):
    return templates.TemplateResponse("app.html", {"request": request, "page": "calendar"})

@app.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request):
    return templates.TemplateResponse("app.html", {"request": request, "page": "analytics"})

@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    return templates.TemplateResponse("app.html", {"request": request, "page": "settings"})

@app.get("/rss", response_class=HTMLResponse)
async def rss_page(request: Request):
    return templates.TemplateResponse("app.html", {"request": request, "page": "rss"})

@app.get("/hashtags", response_class=HTMLResponse)
async def hashtags_page(request: Request):
    return templates.TemplateResponse("app.html", {"request": request, "page": "hashtags"})


# ════════════════════════════════════════════════════════════════
# API — Posts
# ════════════════════════════════════════════════════════════════

@app.get("/api/posts")
async def api_list_posts(status: Optional[str] = None):
    conn = get_db()
    if status and status != "all":
        rows = conn.execute("SELECT * FROM posts WHERE status = ? ORDER BY id DESC", (status,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM posts ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/api/posts")
async def api_create_post(request: Request):
    data = await request.json()
    conn = get_db()
    conn.execute(
        "INSERT INTO posts (title, body, category, platforms, status, scheduled_at, extra_fields) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (data.get("title", ""), data.get("body", ""), data.get("category", "general"),
         json.dumps(data.get("platforms", [])), data.get("status", "draft"),
         data.get("scheduled_at", ""), json.dumps(data.get("extra_fields", {})))
    )
    conn.commit()
    post_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return {"success": True, "post_id": post_id}

@app.delete("/api/posts/{post_id}")
async def api_delete_post(post_id: int):
    conn = get_db()
    conn.execute("DELETE FROM posts WHERE id = ?", (post_id,))
    conn.commit()
    conn.close()
    return {"success": True}


# ════════════════════════════════════════════════════════════════
# API — Publish
# ════════════════════════════════════════════════════════════════

def _get_publisher(platform: str):
    """Lazy-load publisher for a platform."""
    if platform == "reddit":
        from integrations.reddit_publisher import RealRedditPublisher
        return RealRedditPublisher()
    elif platform == "medium":
        from integrations.medium_publisher import RealMediumPublisher
        return RealMediumPublisher()
    elif platform == "tumblr":
        from integrations.tumblr_publisher import RealTumblrPublisher
        return RealTumblrPublisher()
    elif platform == "shopify_blog":
        from integrations.shopify_blog_publisher import ShopifyBlogPublisher
        return ShopifyBlogPublisher()
    else:
        raise ValueError(f"Unknown platform: {platform}")


class _SimpleQueueItem:
    """Duck-typed queue item for publisher compatibility."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@app.post("/api/publish/{post_id}")
async def api_publish_post(post_id: int):
    conn = get_db()
    post = conn.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
    if not post:
        conn.close()
        raise HTTPException(404, "Post not found")

    platforms = json.loads(post["platforms"] or "[]")
    extra = json.loads(post["extra_fields"] or "{}")
    results = []

    for platform in platforms:
        try:
            pub = _get_publisher(platform)
            item = _SimpleQueueItem(
                id=f"web-{post_id}-{platform}",
                platform=platform,
                platform_content={
                    "title": post["title"],
                    "body": post["body"],
                    "caption": post["body"][:2000],
                    **extra.get(platform, {}),
                    **{k: v for k, v in extra.items() if not isinstance(v, dict)},
                },
            )
            result = await pub.publish(item)
            success = result.success
            post_url = result.post_url
            error = result.error
        except Exception as e:
            success = False
            post_url = ""
            error = str(e)

        conn.execute(
            "INSERT INTO publish_log (post_id, platform, success, post_url, error) VALUES (?, ?, ?, ?, ?)",
            (post_id, platform, success, post_url, error)
        )
        results.append({"platform": platform, "success": success, "post_url": post_url, "error": error})

    # Update status
    all_ok = all(r["success"] for r in results)
    conn.execute(
        "UPDATE posts SET status = ?, published_at = ? WHERE id = ?",
        ("published" if all_ok else "failed", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"), post_id)
    )
    conn.commit()
    conn.close()
    return {"success": all_ok, "results": results}


@app.post("/api/test-connection/{platform}")
async def api_test_connection(platform: str):
    try:
        pub = _get_publisher(platform)
        if hasattr(pub, "test_connection"):
            ok = await pub.test_connection()
            return {"platform": platform, "connected": ok}
        return {"platform": platform, "connected": True, "note": "No test method"}
    except Exception as e:
        return {"platform": platform, "connected": False, "error": str(e)}


# ════════════════════════════════════════════════════════════════
# API — Dashboard Stats
# ════════════════════════════════════════════════════════════════

@app.get("/api/stats")
async def api_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    published = conn.execute("SELECT COUNT(*) FROM posts WHERE status='published'").fetchone()[0]
    scheduled = conn.execute("SELECT COUNT(*) FROM posts WHERE status='scheduled'").fetchone()[0]
    drafts = conn.execute("SELECT COUNT(*) FROM posts WHERE status='draft'").fetchone()[0]
    failed = conn.execute("SELECT COUNT(*) FROM posts WHERE status='failed'").fetchone()[0]
    conn.close()
    return {"total": total, "published": published, "scheduled": scheduled, "drafts": drafts, "failed": failed}


@app.get("/api/calendar-events")
async def api_calendar_events(month: Optional[str] = None):
    conn = get_db()
    rows = conn.execute("SELECT id, title, platforms, status, scheduled_at, published_at FROM posts WHERE scheduled_at IS NOT NULL AND scheduled_at != ''").fetchall()
    conn.close()
    events = []
    for r in rows:
        dt = r["scheduled_at"] or r["published_at"] or ""
        if dt:
            events.append({
                "id": r["id"], "title": r["title"], "date": dt[:10],
                "platforms": json.loads(r["platforms"] or "[]"), "status": r["status"]
            })
    return events


@app.get("/api/analytics")
async def api_analytics():
    conn = get_db()
    logs = conn.execute("SELECT * FROM publish_log ORDER BY published_at DESC LIMIT 100").fetchall()
    # Platform stats
    platform_stats = {}
    for log in logs:
        p = log["platform"]
        if p not in platform_stats:
            platform_stats[p] = {"total": 0, "success": 0, "failed": 0}
        platform_stats[p]["total"] += 1
        if log["success"]:
            platform_stats[p]["success"] += 1
        else:
            platform_stats[p]["failed"] += 1
    conn.close()
    return {"logs": [dict(l) for l in logs], "platform_stats": platform_stats}


# ════════════════════════════════════════════════════════════════
# API — RSS Feeds
# ════════════════════════════════════════════════════════════════

@app.get("/api/rss/feeds")
async def api_rss_list_feeds():
    from integrations.rss_reader import list_feeds
    return list_feeds()

@app.post("/api/rss/feeds")
async def api_rss_add_feed(request: Request):
    data = await request.json()
    from integrations.rss_reader import add_feed
    return add_feed(
        url=data.get("url", ""),
        name=data.get("name", ""),
        category=data.get("category", "general"),
        import_mode=data.get("import_mode", "full"),
        target_platforms=data.get("target_platforms", []),
    )

@app.delete("/api/rss/feeds/{feed_id}")
async def api_rss_remove_feed(feed_id: str):
    from integrations.rss_reader import remove_feed
    return remove_feed(feed_id)

@app.put("/api/rss/feeds/{feed_id}")
async def api_rss_update_feed(feed_id: str, request: Request):
    data = await request.json()
    from integrations.rss_reader import update_feed
    return update_feed(feed_id, **data)

@app.post("/api/rss/fetch/{feed_id}")
async def api_rss_fetch(feed_id: str):
    from integrations.rss_reader import fetch_feed
    return fetch_feed(feed_id=feed_id)

@app.post("/api/rss/preview")
async def api_rss_preview(request: Request):
    data = await request.json()
    from integrations.rss_reader import fetch_feed
    return fetch_feed(feed_url=data.get("url", ""), max_entries=5)

@app.post("/api/rss/import")
async def api_rss_import_entry(request: Request):
    """Import an RSS entry as a draft post."""
    data = await request.json()
    from integrations.rss_reader import import_entry_as_draft
    draft = import_entry_as_draft(data.get("entry", {}), data.get("platforms", []))
    conn = get_db()
    conn.execute(
        "INSERT INTO posts (title, body, category, platforms, status, extra_fields) VALUES (?, ?, ?, ?, ?, ?)",
        (draft["title"], draft["body"], draft["category"], draft["platforms"], draft["status"], draft["extra_fields"])
    )
    conn.commit()
    post_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return {"success": True, "post_id": post_id}


# ── Bulk Import (Sendible-killer: 200-500 posts / call) ──

@app.post("/api/rss/bulk-import")
async def api_rss_bulk_import(request: Request):
    """
    Bulk import from ALL configured RSS feeds into SQLite drafts.
    Like Sendible's 200-500 posts/day but in ONE API call.
    Skips already-imported entries (dedup).
    """
    data = await request.json() if (await request.body()) else {}
    from integrations.rss_reader import (
        list_feeds, bulk_fetch_feed, bulk_import_as_drafts
    )
    platforms = data.get("platforms", ["medium", "shopify_blog"])
    max_per_feed = data.get("max_per_feed", 50)

    feeds = list_feeds()
    total_imported = 0
    total_skipped = 0
    feed_results = []

    for feed in feeds:
        fetch = bulk_fetch_feed(feed_id=feed.get("id"), max_entries=max_per_feed)
        if not fetch.get("success"):
            feed_results.append({"feed": feed.get("name"), "error": fetch.get("error")})
            continue
        entries = fetch.get("entries", [])
        if entries:
            result = bulk_import_as_drafts(entries, platforms=platforms)
            total_imported += result.get("imported", 0)
            total_skipped += result.get("skipped_duplicates", 0)
            feed_results.append({
                "feed": feed.get("name"),
                "imported": result.get("imported", 0),
                "skipped": result.get("skipped_duplicates", 0),
            })
        else:
            feed_results.append({
                "feed": feed.get("name"),
                "imported": 0,
                "skipped": fetch.get("skipped_duplicates", 0),
            })

    return {
        "success": True,
        "total_imported": total_imported,
        "total_skipped": total_skipped,
        "feeds_processed": len(feeds),
        "details": feed_results,
    }


@app.post("/api/rss/bulk-fetch/{feed_id}")
async def api_rss_bulk_fetch(feed_id: str, request: Request):
    """Bulk fetch up to 500 entries from a single feed (with dedup)."""
    data = await request.json() if (await request.body()) else {}
    from integrations.rss_reader import bulk_fetch_feed
    return bulk_fetch_feed(
        feed_id=feed_id,
        max_entries=data.get("max_entries", 500),
        skip_imported=data.get("skip_imported", True),
    )


# ── Railway RSS Server (TheRike full-text feeds) ──

@app.get("/api/rss/railway/blogs")
async def api_rss_railway_blogs():
    """List all available TheRike blogs with their Railway RSS URLs."""
    from integrations.rss_reader import list_therike_blogs
    return list_therike_blogs()

@app.post("/api/rss/railway/setup")
async def api_rss_railway_setup(request: Request):
    """
    One-click: register ALL 10 TheRike blogs as RSS feeds.
    Uses the Railway self-hosted full-text RSS server.
    """
    data = await request.json() if (await request.body()) else {}
    from integrations.rss_reader import setup_therike_feeds
    return setup_therike_feeds(
        target_platforms=data.get("platforms", ["medium", "shopify_blog", "reddit"]),
    )

@app.post("/api/rss/railway/bulk-import")
async def api_rss_railway_bulk_import(request: Request):
    """
    One-click bulk import from ALL TheRike blogs via Railway RSS.
    Fetches full-text content + imports as drafts. Dedup included.
    """
    data = await request.json() if (await request.body()) else {}
    from integrations.rss_reader import bulk_import_all_therike
    return bulk_import_all_therike(
        platforms=data.get("platforms", ["medium", "shopify_blog"]),
        max_per_blog=data.get("max_per_blog", 50),
    )


# ════════════════════════════════════════════════════════════════
# API — Hashtag Collections
# ════════════════════════════════════════════════════════════════

@app.get("/api/hashtags")
async def api_list_hashtags():
    conn = get_db()
    rows = conn.execute("SELECT * FROM hashtag_collections ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/api/hashtags")
async def api_create_hashtag_collection(request: Request):
    data = await request.json()
    conn = get_db()
    conn.execute(
        "INSERT INTO hashtag_collections (name, description, hashtags, platforms, niche) VALUES (?, ?, ?, ?, ?)",
        (data.get("name", ""), data.get("description", ""),
         json.dumps(data.get("hashtags", [])), json.dumps(data.get("platforms", [])),
         data.get("niche", "general"))
    )
    conn.commit()
    cid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return {"success": True, "id": cid}

@app.put("/api/hashtags/{collection_id}")
async def api_update_hashtag_collection(collection_id: int, request: Request):
    data = await request.json()
    conn = get_db()
    conn.execute(
        "UPDATE hashtag_collections SET name=?, description=?, hashtags=?, platforms=?, niche=?, updated_at=datetime('now') WHERE id=?",
        (data.get("name"), data.get("description"), json.dumps(data.get("hashtags", [])),
         json.dumps(data.get("platforms", [])), data.get("niche"), collection_id)
    )
    conn.commit()
    conn.close()
    return {"success": True}

@app.delete("/api/hashtags/{collection_id}")
async def api_delete_hashtag_collection(collection_id: int):
    conn = get_db()
    conn.execute("DELETE FROM hashtag_collections WHERE id = ?", (collection_id,))
    conn.commit()
    conn.close()
    return {"success": True}

@app.post("/api/hashtags/generate")
async def api_generate_hashtags(request: Request):
    """Generate hashtags using the 5-layer matrix."""
    data = await request.json()
    niche = data.get("niche", "general")
    platform = data.get("platform", "instagram")
    try:
        from hashtags.matrix_5layer import generate_hashtag_matrix
        result = generate_hashtag_matrix(niche, platform)
        return {"success": True, "hashtags": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ════════════════════════════════════════════════════════════════
# API — Budget / Cost
# ════════════════════════════════════════════════════════════════

@app.get("/api/budget")
async def api_budget():
    try:
        from agents.cost_agent import get_budget_status
        return get_budget_status()
    except Exception as e:
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════════
# API — Scheduler
# ════════════════════════════════════════════════════════════════

@app.get("/api/scheduler/status")
async def api_scheduler_status():
    from core.scheduler import get_scheduler
    s = get_scheduler()
    return {"running": s.running, "check_interval": s.check_interval}

@app.post("/api/scheduler/run-now")
async def api_scheduler_run_now():
    """Manually trigger scheduler check."""
    from core.scheduler import get_scheduler
    s = get_scheduler()
    results = s.check_and_publish()
    return {"triggered": True, "results": results}


# ════════════════════════════════════════════════════════════════
# Health
# ════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "2.2.0",
        "engine": "ViralOps Engine — EMADS-PR v1.0",
        "features": [
            "5 Micro-Niche Hashtags (smart, no generic)",
            "Smart Content Split (sentence-boundary, never mid-word)",
            "GenAI Answer Extraction (strips filler/preamble)",
            "7 Social Connectors (TW/IG/FB/YT/LI/TT/PIN)",
            "Bulk RSS Import (500/call) + Railway RSS",
            "TikTok Auto Music Selection (AI-powered)",
            "Analytics + Hashtag Performance Tracking",
            "Background Scheduler (rate-limited, 11 platforms)",
            "Docker + Railway deployment ready",
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }


# ════════════════════════════════════════════════════════════════
# API — TikTok Auto Music (Sendible CAN'T do this)
# ════════════════════════════════════════════════════════════════

@app.post("/api/tiktok/music/recommend")
async def api_tiktok_music_recommend(request: Request):
    """
    🎵 Auto-recommend TikTok music based on content + niche + mood.

    What Sendible CAN'T do:
      - Sendible: user manually picks music (tedious for 200-500 posts/day)
      - ViralOps: AI auto-recommends music per post based on content analysis
    """
    data = await request.json()
    from integrations.tiktok_music import recommend_music
    return recommend_music(
        text=data.get("text", ""),
        niche=data.get("niche"),
        mood=data.get("mood"),
        limit=data.get("limit", 5),
        min_trending=data.get("min_trending", 0.0),
    )

@app.get("/api/tiktok/music/stats")
async def api_tiktok_music_stats():
    """Get music database stats — track counts by mood and niche."""
    from integrations.tiktok_music import get_music_stats
    return get_music_stats()

@app.get("/api/tiktok/music/moods")
async def api_tiktok_music_moods():
    """List all available mood categories."""
    from integrations.tiktok_music import MOODS, NICHE_DEFAULT_MOOD
    return {"moods": MOODS, "niche_defaults": NICHE_DEFAULT_MOOD}

@app.get("/api/tiktok/music/by-mood/{mood}")
async def api_tiktok_music_by_mood(mood: str):
    """List all tracks for a specific mood."""
    from integrations.tiktok_music import list_tracks_by_mood
    tracks = list_tracks_by_mood(mood)
    return {"mood": mood, "count": len(tracks), "tracks": tracks}

@app.get("/api/tiktok/music/by-niche/{niche}")
async def api_tiktok_music_by_niche(niche: str):
    """List all tracks tagged for a specific niche."""
    from integrations.tiktok_music import list_tracks_by_niche
    tracks = list_tracks_by_niche(niche)
    return {"niche": niche, "count": len(tracks), "tracks": tracks}

@app.post("/api/tiktok/music/tracks")
async def api_tiktok_music_add_track(request: Request):
    """Add a custom track to the music database."""
    data = await request.json()
    from integrations.tiktok_music import add_custom_track
    return add_custom_track(
        title=data.get("title", ""),
        artist=data.get("artist", ""),
        mood=data.get("mood", "chill"),
        niches=data.get("niches", []),
        bpm=data.get("bpm", 120),
        tiktok_sound_url=data.get("tiktok_sound_url", ""),
        tags=data.get("tags", []),
        trending_score=data.get("trending_score", 0.7),
    )

@app.delete("/api/tiktok/music/tracks/{track_id}")
async def api_tiktok_music_remove_track(track_id: str):
    """Remove a custom track from the database."""
    from integrations.tiktok_music import remove_custom_track
    return remove_custom_track(track_id)

@app.get("/api/tiktok/music/all")
async def api_tiktok_music_all():
    """Get all tracks (built-in + custom)."""
    from integrations.tiktok_music import get_all_tracks
    tracks = get_all_tracks()
    return {"count": len(tracks), "tracks": tracks}


# ════════════════════════════════════════════════════════════════
# Analytics API
# ════════════════════════════════════════════════════════════════

@app.get("/api/analytics/dashboard")
async def api_analytics_dashboard(days: int = 30):
    """Full analytics dashboard — publish stats, hashtag performance, best times."""
    from monitoring.analytics import get_analytics_dashboard
    return get_analytics_dashboard(days)

@app.get("/api/analytics/publish-stats")
async def api_analytics_publish_stats(days: int = 30):
    """Publishing success rates per platform."""
    from monitoring.analytics import get_publish_stats
    return get_publish_stats(days)

@app.get("/api/analytics/best-times")
async def api_analytics_best_times(platform: str = None, days: int = 30):
    """Best posting times based on historical data."""
    from monitoring.analytics import get_best_posting_times
    return get_best_posting_times(platform, days)

@app.get("/api/analytics/hashtags")
async def api_analytics_hashtags(platform: str = None, limit: int = 20):
    """Top-performing hashtags by engagement."""
    from monitoring.analytics import get_top_hashtags
    return get_top_hashtags(platform, limit)

@app.get("/api/analytics/hashtag-report")
async def api_analytics_hashtag_report(days: int = 30):
    """Comprehensive hashtag performance report."""
    from monitoring.analytics import get_hashtag_report
    return get_hashtag_report(days)


# ════════════════════════════════════════════════════════════════
# Social Connector Status API
# ════════════════════════════════════════════════════════════════

@app.get("/api/connectors/status")
async def api_connectors_status():
    """Check which social platform connectors are configured."""
    from integrations.social_connectors import get_all_configured_publishers
    configured = get_all_configured_publishers()
    return {
        "configured": list(configured.keys()),
        "total": len(configured),
        "platforms": {name: {"type": type(pub).__name__} for name, pub in configured.items()},
    }

@app.post("/api/connectors/test")
async def api_connectors_test():
    """Test all configured social platform connections."""
    from integrations.social_connectors import test_all_connections
    results = await test_all_connections()
    return {"results": results, "all_ok": all(results.values()) if results else False}

@app.get("/api/connectors/rate-limits")
async def api_connectors_rate_limits():
    """Get current daily rate limit status."""
    from core.scheduler import get_scheduler, DAILY_RATE_LIMITS
    scheduler = get_scheduler()
    scheduler._reset_daily_counts_if_needed()
    return {
        "limits": DAILY_RATE_LIMITS,
        "current_usage": dict(scheduler._daily_counts),
        "remaining": {
            platform: max(0, limit - scheduler._daily_counts.get(platform, 0))
            for platform, limit in DAILY_RATE_LIMITS.items()
        },
    }


# ════════════════════════════════════════════════════════════════
# Run
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
