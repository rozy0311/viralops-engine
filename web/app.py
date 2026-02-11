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

@app.get("/platforms", response_class=HTMLResponse)
async def platforms_page(request: Request):
    return templates.TemplateResponse("app.html", {"request": request, "page": "platforms"})

@app.get("/engagement", response_class=HTMLResponse)
async def engagement_page(request: Request):
    return templates.TemplateResponse("app.html", {"request": request, "page": "engagement"})

@app.get("/time-slots", response_class=HTMLResponse)
async def timeslots_page(request: Request):
    return templates.TemplateResponse("app.html", {"request": request, "page": "timeslots"})

@app.get("/preview", response_class=HTMLResponse)
async def preview_page(request: Request):
    return templates.TemplateResponse("app.html", {"request": request, "page": "preview"})

@app.get("/autopilot", response_class=HTMLResponse)
async def autopilot_page(request: Request):
    return templates.TemplateResponse("app.html", {"request": request, "page": "autopilot"})


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

@app.put("/api/posts/{post_id}")
async def api_update_post(post_id: int, request: Request):
    data = await request.json()
    conn = get_db()
    fields, values = [], []
    for col in ("title", "body", "category", "platforms", "status", "scheduled_at", "extra_fields"):
        if col in data:
            val = data[col]
            if col in ("platforms", "extra_fields") and isinstance(val, (list, dict)):
                val = json.dumps(val)
            fields.append(f"{col} = ?")
            values.append(val)
    if not fields:
        return {"success": False, "error": "No fields to update"}
    values.append(post_id)
    conn.execute(f"UPDATE posts SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()
    conn.close()
    return {"success": True}


# ════════════════════════════════════════════════════════════════
# API — Auto-Pilot (Content Generation)
# ════════════════════════════════════════════════════════════════

@app.post("/api/autopilot/generate")
async def api_autopilot_generate(request: Request):
    """Generate content pack for Auto-Pilot using content_factory or fallback."""
    data = await request.json()
    niche = data.get("niche", "general")
    topic = data.get("topic") or None
    platforms = data.get("platforms", [])

    # Try to use content_factory agent
    try:
        from agents.content_factory import generate_content_pack
        state = {
            "niche_config": {"niche": niche, "sub_niche": niche},
            "topic": topic,
            "platforms": platforms,
        }
        result = generate_content_pack(state)
        pack = result.get("content_pack") or result
        return {
            "title": pack.get("title", f"{niche.replace('_', ' ').title()} — Auto-Pilot"),
            "body": pack.get("body", ""),
            "content": pack.get("body", ""),
            "hook": pack.get("hook", ""),
            "cta": pack.get("cta", ""),
            "hashtags": pack.get("hashtags", []),
        }
    except Exception as e:
        logger.warning("content_factory_fallback", error=str(e))
        # Fallback: generate simple content
        topic_str = topic or niche.replace("_", " ").title()
        return {
            "title": f"🚀 {topic_str} — Expert Tips",
            "body": (
                f"Here are proven strategies for {topic_str.lower()}:\n\n"
                f"1️⃣ Start with a solid foundation\n"
                f"2️⃣ Focus on consistency over perfection\n"
                f"3️⃣ Engage with your community daily\n"
                f"4️⃣ Measure what matters, optimize weekly\n"
                f"5️⃣ Stay authentic to your brand voice\n\n"
                f"Which tip resonates with you most? Drop a comment! 👇"
            ),
            "content": f"Expert tips for {topic_str.lower()}",
            "hook": f"Want to level up your {topic_str.lower()} game?",
            "cta": "Follow for more tips!",
            "hashtags": [f"#{niche}", f"#{niche}tips", "#socialmedia", "#growthhacking", "#contentcreator"],
        }


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
# API — RSS Auto Poster (Sendible-style)
# ════════════════════════════════════════════════════════════════

@app.get("/api/rss-auto-poster")
async def api_list_auto_posters():
    """List all RSS Auto Poster configs."""
    from integrations.rss_auto_poster import list_auto_posters
    return list_auto_posters()

@app.get("/api/rss-auto-poster/status")
async def api_auto_poster_status():
    """Get overview status of all auto posters."""
    from integrations.rss_auto_poster import get_auto_poster_status
    return get_auto_poster_status()

@app.get("/api/rss-auto-poster/{poster_id}")
async def api_get_auto_poster(poster_id: str):
    """Get a single RSS Auto Poster config."""
    from integrations.rss_auto_poster import get_auto_poster
    cfg = get_auto_poster(poster_id)
    if not cfg:
        raise HTTPException(status_code=404, detail="Auto poster not found")
    return cfg

@app.post("/api/rss-auto-poster")
async def api_create_auto_poster(request: Request):
    """
    Create a new RSS Auto Poster (Sendible-style).

    Body example:
    {
        "name": "TheRike → TikTok Auto",
        "feed_id": "abc123",
        "target_platforms": ["tiktok", "instagram"],
        "frequency": "every_hour",
        "entries_per_cycle": 1,
        "publish_mode": "scheduled",
        "niche": "sustainable-living",
        "tiktok_music_enabled": true
    }
    """
    data = await request.json()
    from integrations.rss_auto_poster import create_auto_poster
    return create_auto_poster(data)

@app.put("/api/rss-auto-poster/{poster_id}")
async def api_update_auto_poster(poster_id: str, request: Request):
    """Update an existing RSS Auto Poster config."""
    data = await request.json()
    from integrations.rss_auto_poster import update_auto_poster
    result = update_auto_poster(poster_id, data)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.delete("/api/rss-auto-poster/{poster_id}")
async def api_delete_auto_poster(poster_id: str):
    """Delete an RSS Auto Poster config."""
    from integrations.rss_auto_poster import delete_auto_poster
    return delete_auto_poster(poster_id)

@app.post("/api/rss-auto-poster/{poster_id}/pause")
async def api_pause_auto_poster(poster_id: str):
    """Pause an active auto poster."""
    from integrations.rss_auto_poster import pause_auto_poster
    return pause_auto_poster(poster_id)

@app.post("/api/rss-auto-poster/{poster_id}/activate")
async def api_activate_auto_poster(poster_id: str):
    """Activate a paused auto poster."""
    from integrations.rss_auto_poster import activate_auto_poster
    return activate_auto_poster(poster_id)

@app.post("/api/rss-auto-poster/tick")
async def api_auto_poster_tick():
    """Manually trigger RSS Auto Poster tick (check all active posters)."""
    from integrations.rss_auto_poster import tick
    return tick()

@app.post("/api/rss-auto-poster/{poster_id}/tick")
async def api_auto_poster_tick_single(poster_id: str):
    """Manually trigger tick for a specific auto poster."""
    from integrations.rss_auto_poster import tick
    return tick(poster_id=poster_id)

@app.get("/api/rss-auto-poster/{poster_id}/history")
async def api_auto_poster_history(poster_id: str, limit: int = 50):
    """Get posting history for a specific auto poster."""
    from integrations.rss_auto_poster import get_poster_history
    return get_poster_history(poster_id, limit)

@app.delete("/api/rss-auto-poster/{poster_id}/history")
async def api_clear_auto_poster_history(poster_id: str):
    """Clear posting history (allows re-posting of entries)."""
    from integrations.rss_auto_poster import clear_poster_history
    return clear_poster_history(poster_id)

@app.get("/api/rss-auto-poster/frequencies")
async def api_auto_poster_frequencies():
    """List available frequency presets."""
    from integrations.rss_auto_poster import FREQUENCY_PRESETS
    return {
        "frequencies": {k: f"{v} minutes" for k, v in FREQUENCY_PRESETS.items()},
    }


# ════════════════════════════════════════════════════════════════
# API — Media Processor (Image → Video + TikTok Music)
# ════════════════════════════════════════════════════════════════

@app.post("/api/media/process")
async def api_media_process(request: Request):
    """
    Full pipeline: Image → Slideshow Video → TikTok Music overlay → .mp4

    Body: {"image_url": "...", "niche": "...", "content_text": "...", "duration": 12}
    """
    data = await request.json()
    from integrations.media_processor import process_image_to_video
    return process_image_to_video(
        image_url=data.get("image_url"),
        image_path=data.get("image_path"),
        music_track=data.get("music_track"),
        niche=data.get("niche", "sustainable-living"),
        content_text=data.get("content_text", ""),
        duration=data.get("duration", 12),
    )

@app.post("/api/media/download")
async def api_media_download(request: Request):
    """Download an image from URL."""
    data = await request.json()
    from integrations.media_processor import download_image
    return download_image(data.get("url", ""))

@app.post("/api/media/slideshow")
async def api_media_slideshow(request: Request):
    """Create slideshow video from a local image file."""
    data = await request.json()
    from integrations.media_processor import create_slideshow_video
    return create_slideshow_video(
        image_path=data.get("image_path"),
        duration=data.get("duration", 12),
    )

@app.post("/api/media/add-music")
async def api_media_add_music(request: Request):
    """Add music overlay to an existing video file."""
    data = await request.json()
    from integrations.media_processor import add_music_to_video
    return add_music_to_video(
        video_path=data.get("video_path"),
        music_url=data.get("music_url"),
        music_file=data.get("music_file"),
    )

@app.get("/api/media/stats")
async def api_media_stats():
    """Get media output directory stats."""
    from integrations.media_processor import get_media_stats
    return get_media_stats()

@app.post("/api/media/cleanup")
async def api_media_cleanup(request: Request):
    """Remove old processed media files."""
    data = await request.json() if (await request.body()) else {}
    from integrations.media_processor import cleanup_old_media
    return cleanup_old_media(max_age_hours=data.get("max_age_hours", 72))


# ════════════════════════════════════════════════════════════════
# API — Telegram Alert Bot
# ════════════════════════════════════════════════════════════════

@app.get("/api/telegram/status")
async def api_telegram_status():
    """Check Telegram bot configuration and connection."""
    from integrations.telegram_bot import is_configured, get_bot_info
    if not is_configured():
        return {"configured": False, "message": "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"}
    return get_bot_info()

@app.post("/api/telegram/test")
async def api_telegram_test():
    """Send a test message to verify Telegram setup."""
    from integrations.telegram_bot import send_custom
    return send_custom("🧪 *ViralOps Engine* — Telegram test successful!\n_Your bot is configured correctly._")

@app.post("/api/telegram/send")
async def api_telegram_send(request: Request):
    """Send a custom message to Telegram."""
    data = await request.json()
    from integrations.telegram_bot import send_message
    return send_message(
        text=data.get("text", ""),
        parse_mode=data.get("parse_mode", "Markdown"),
    )


# ════════════════════════════════════════════════════════════════
# Health
# ════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "2.4.0",
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
            "RSS Auto Poster — Sendible-style",
            "Media Processor — Image→Video + TikTok Music (NEW)",
            "Telegram Alert Bot — publish notifications (NEW)",
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
# API — Engagement Fetcher (v2.5)
# ════════════════════════════════════════════════════════════════

@app.post("/api/engagement/fetch")
async def api_engagement_fetch(request: Request):
    """Pull real engagement metrics from platform APIs for recent posts."""
    data = await request.json() if await request.body() else {}
    from monitoring.engagement_fetcher import fetch_engagement_batch
    result = await fetch_engagement_batch(limit=data.get("limit", 50))
    return result

@app.get("/api/engagement/summary")
async def api_engagement_summary(platform: str = None, days: int = 7):
    """Get engagement summary across platforms."""
    from monitoring.engagement_fetcher import get_engagement_summary
    return get_engagement_summary(platform=platform, days=days)

@app.get("/api/engagement/post/{post_id}")
async def api_engagement_post(post_id: int):
    """Get engagement data for a specific post."""
    from monitoring.engagement_fetcher import get_post_engagement
    return {"post_id": post_id, "data": get_post_engagement(post_id)}


# ════════════════════════════════════════════════════════════════
# API — Time Slot Engine (v2.5)
# ════════════════════════════════════════════════════════════════

@app.get("/api/time-slots/suggest/{platform}")
async def api_time_slot_suggest(platform: str, utc_offset: int = 0):
    """Suggest optimal next posting time for a platform."""
    from core.time_slot_engine import suggest_time
    return suggest_time(platform, utc_offset_hours=utc_offset)

@app.post("/api/time-slots/schedule")
async def api_time_slot_schedule(request: Request):
    """Generate a full daily posting schedule across platforms."""
    data = await request.json()
    from core.time_slot_engine import suggest_schedule
    return suggest_schedule(
        platforms=data.get("platforms", ["tiktok", "instagram", "twitter"]),
        posts_per_platform=data.get("posts_per_platform", 1),
        utc_offset_hours=data.get("utc_offset", 0),
        spacing_minutes=data.get("spacing_minutes", 30),
    )

@app.get("/api/time-slots/best-hours")
async def api_time_slot_best_hours(platform: str = None, days: int = 30):
    """Get best posting hours (analytics-backed)."""
    from core.time_slot_engine import get_best_hours
    return get_best_hours(platform=platform, days=days)


# ════════════════════════════════════════════════════════════════
# API — Trending Decay + BPM Music (v2.5)
# ════════════════════════════════════════════════════════════════

@app.post("/api/tiktok/music/decay")
async def api_tiktok_music_decay(request: Request):
    """Apply trending score decay to music library."""
    data = await request.json() if await request.body() else {}
    from integrations.tiktok_music import decay_trending_scores
    return decay_trending_scores(
        half_life_days=data.get("half_life_days", 14.0),
        min_score=data.get("min_score", 0.1),
    )

@app.get("/api/tiktok/music/trending")
async def api_tiktok_music_trending(limit: int = 10, min_score: float = 0.7):
    """Get top trending tracks."""
    from integrations.tiktok_music import get_trending_tracks
    return {"tracks": get_trending_tracks(limit=limit, min_score=min_score)}


# ════════════════════════════════════════════════════════════════
# API — Multi-Image Slideshow + Text Overlay (v2.5)
# ════════════════════════════════════════════════════════════════

@app.post("/api/media/multi-slideshow")
async def api_media_multi_slideshow(request: Request):
    """Create video slideshow from multiple image URLs."""
    data = await request.json()
    from integrations.media_processor import create_multi_image_slideshow_from_urls
    try:
        return create_multi_image_slideshow_from_urls(
            image_urls=data.get("image_urls", []),
            duration_per_image=data.get("duration_per_image", 4),
            width=data.get("width", 1080),
            height=data.get("height", 1920),
        )
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/api/media/text-overlay")
async def api_media_text_overlay(request: Request):
    """Add text caption overlay to a video."""
    data = await request.json()
    from integrations.media_processor import add_text_overlay
    try:
        return add_text_overlay(
            video_path=data.get("video_path", ""),
            text=data.get("text", ""),
            position=data.get("position", "bottom"),
            font_size=data.get("font_size", 48),
            font_color=data.get("font_color", "white"),
        )
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/api/media/subtitles")
async def api_media_subtitles(request: Request):
    """Add timed subtitles to a video (SRT-style)."""
    data = await request.json()
    from integrations.media_processor import add_subtitles
    try:
        return add_subtitles(
            video_path=data.get("video_path", ""),
            subtitles=data.get("subtitles", []),
            font_size=data.get("font_size", 36),
        )
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ════════════════════════════════════════════════════════════════
# API — Platform Setup Status (v2.5)
# ════════════════════════════════════════════════════════════════

@app.get("/api/platforms/setup-status")
async def api_platforms_setup_status():
    """Check which platforms have API keys configured."""
    import os
    platforms = {
        "twitter":      {"env_keys": ["TWITTER_MAIN_API_KEY"], "difficulty": "medium", "docs": "https://developer.twitter.com"},
        "instagram":    {"env_keys": ["INSTAGRAM_MAIN_ACCESS_TOKEN"], "difficulty": "medium", "docs": "https://developers.facebook.com"},
        "facebook":     {"env_keys": ["FACEBOOK_MAIN_ACCESS_TOKEN"], "difficulty": "medium", "docs": "https://developers.facebook.com"},
        "youtube":      {"env_keys": ["YOUTUBE_MAIN_API_KEY"], "difficulty": "medium", "docs": "https://console.cloud.google.com"},
        "linkedin":     {"env_keys": ["LINKEDIN_MAIN_ACCESS_TOKEN"], "difficulty": "medium", "docs": "https://developer.linkedin.com"},
        "tiktok":       {"env_keys": ["TIKTOK_MAIN_ACCESS_TOKEN"], "difficulty": "hard", "docs": "https://developers.tiktok.com"},
        "pinterest":    {"env_keys": ["PINTEREST_MAIN_ACCESS_TOKEN"], "difficulty": "medium", "docs": "https://developers.pinterest.com"},
        "reddit":       {"env_keys": ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"], "difficulty": "easy", "docs": "https://www.reddit.com/prefs/apps"},
        "medium":       {"env_keys": ["MEDIUM_ACCESS_TOKEN"], "difficulty": "easy", "docs": "https://medium.com/me/settings"},
        "tumblr":       {"env_keys": ["TUMBLR_CONSUMER_KEY"], "difficulty": "easy", "docs": "https://www.tumblr.com/oauth/apps"},
        "threads":      {"env_keys": ["THREADS_MAIN_ACCESS_TOKEN"], "difficulty": "hard", "docs": "https://developers.facebook.com/docs/threads"},
        "bluesky":      {"env_keys": ["BLUESKY_MAIN_HANDLE", "BLUESKY_MAIN_APP_PASSWORD"], "difficulty": "easy", "docs": "https://bsky.app/settings/app-passwords"},
        "mastodon":     {"env_keys": ["MASTODON_MAIN_ACCESS_TOKEN"], "difficulty": "easy", "docs": "https://docs.joinmastodon.org/client/intro/"},
        "quora":        {"env_keys": ["QUORA_MAIN_SESSION_COOKIE"], "difficulty": "hard", "docs": ""},
        "shopify_blog": {"env_keys": ["SHOPIFY_SHOP_URL", "SHOPIFY_ACCESS_TOKEN"], "difficulty": "easy", "docs": "https://shopify.dev/docs/api"},
        "lemon8":       {"env_keys": ["LEMON8_SESSION_TOKEN"], "difficulty": "hard", "docs": ""},
    }

    result = {}
    for platform, info in platforms.items():
        configured = all(os.environ.get(k) for k in info["env_keys"])
        result[platform] = {
            "configured": configured,
            "difficulty": info["difficulty"],
            "docs_url": info["docs"],
            "env_keys_needed": info["env_keys"],
        }

    configured_count = sum(1 for p in result.values() if p["configured"])
    return {
        "platforms": result,
        "total": len(result),
        "configured": configured_count,
        "not_configured": len(result) - configured_count,
    }


# ════════════════════════════════════════════════════════════════
# Run
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
