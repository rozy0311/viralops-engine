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
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from typing import Optional

import secrets as _secrets
from contextlib import contextmanager

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

# ── Add project root to path ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.models import PublishResult
from monitoring.alerting import AlertManager, AlertLevel

logger = structlog.get_logger()

# ── Global AlertManager singleton ──
_alert_manager = AlertManager()

# ── Database ──
DB_PATH = os.path.join(os.path.dirname(__file__), "viralops.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db_safe():
    """Context-managed DB connection — guarantees close even on exceptions."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Initialize database tables (with schema migration for old TEXT-id tables)."""
    with get_db_safe() as conn:

        # ── Schema migration: fix legacy TEXT PRIMARY KEY → INTEGER AUTOINCREMENT ──
        try:
            info = conn.execute("PRAGMA table_info(posts)").fetchall()
            if info:
                id_col = next((c for c in info if c["name"] == "id"), None)
                if id_col and id_col["type"].upper() == "TEXT":
                    logger.info("init_db.migrating", msg="Fixing posts table: TEXT id → INTEGER AUTOINCREMENT")
                    conn.executescript("""
                        DROP TABLE IF EXISTS posts;
                        DROP TABLE IF EXISTS publish_log;
                    """)
        except Exception:
            pass  # Table may not exist yet

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


# ── Scheduler background task ──
_scheduler_task = None
_rss_tick_task = None
_blog_share_tick_task = None
_autopilot_task = None
_autopilot_last_run = None
_autopilot_posts_today = 0
_autopilot_posts_today_date = None


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


async def rss_tick_loop():
    """Background loop that triggers RSS Auto Poster tick every 5 minutes."""
    await asyncio.sleep(30)  # Initial delay to let app start
    while True:
        try:
            from integrations.rss_auto_poster import tick
            result = await asyncio.to_thread(tick)  # tick() is sync — run in thread
            posted = sum(r.get("posted", 0) for r in result.get("results", []))
            if posted > 0:
                logger.info("rss_tick.auto_posted", count=posted)
        except Exception as e:
            logger.error("rss_tick.error", error=str(e))
        await asyncio.sleep(300)  # Every 5 minutes


async def blog_share_tick_loop():
    """Background loop that triggers Shopify Blog Auto-Share tick."""
    await asyncio.sleep(60)  # Initial delay
    while True:
        try:
            from integrations.shopify_auto_share import auto_share_tick
            result = await auto_share_tick()
            shared = result.get("shared", 0)
            if shared > 0:
                logger.info("blog_share_tick.auto_shared", count=shared)
        except Exception as e:
            logger.error("blog_share_tick.error", error=str(e))
        # Respect configured interval (default 30 min = 1800 sec)
        try:
            from integrations.shopify_auto_share import get_auto_share
            instance = await get_auto_share()
            interval = instance._config.get("interval_min", 30) * 60
        except Exception:
            interval = 1800
        await asyncio.sleep(interval)


# ── Autopilot defaults ──
_AUTOPILOT_DEFAULTS = {
    "enabled": False,
    "interval_hours": 4,
    "niches": ["general"],
    "platforms": ["tiktok", "pinterest"],
    "max_posts_per_day": 6,
}


def _get_autopilot_config() -> dict:
    """Read autopilot config from settings table, merging with defaults."""
    with get_db_safe() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = 'autopilot_config'"
        ).fetchone()
    if row:
        saved = json.loads(row["value"])
        merged = {**_AUTOPILOT_DEFAULTS, **saved}
        return merged
    return dict(_AUTOPILOT_DEFAULTS)


def _save_autopilot_config(cfg: dict):
    """Persist autopilot config to settings table."""
    with get_db_safe() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES ('autopilot_config', ?)",
            (json.dumps(cfg),),
        )
        conn.commit()


async def autopilot_loop():
    """Background loop — generates content via full EMADS-PR pipeline.

    Picks niche from rotation, runs pipeline, auto-publishes if risk < 4,
    otherwise saves as draft. Respects max_posts_per_day limit.
    """
    global _autopilot_last_run, _autopilot_posts_today, _autopilot_posts_today_date
    await asyncio.sleep(120)  # Initial delay — let other services start
    _niche_index = 0

    while True:
        try:
            cfg = _get_autopilot_config()
            if not cfg.get("enabled"):
                await asyncio.sleep(60)
                continue

            # Reset daily counter
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if _autopilot_posts_today_date != today:
                _autopilot_posts_today = 0
                _autopilot_posts_today_date = today

            # Check daily limit
            if _autopilot_posts_today >= cfg.get("max_posts_per_day", 6):
                logger.info("autopilot.daily_limit_reached",
                            count=_autopilot_posts_today)
                await asyncio.sleep(600)  # Check again in 10 min
                continue

            # Pick niche from rotation
            niches = cfg.get("niches", ["general"])
            niche = niches[_niche_index % len(niches)]
            _niche_index += 1
            platforms = cfg.get("platforms", ["tiktok", "pinterest"])

            logger.info("autopilot.generating", niche=niche, platforms=platforms)

            from graph import get_compiled_graph
            import uuid

            graph = get_compiled_graph()
            thread_id = f"autopilot-{uuid.uuid4().hex[:8]}"

            initial_state = {
                "niche_config": {"niche": niche, "sub_niche": niche},
                "topic": None,  # Let the pipeline generate a topic
                "platforms": platforms,
                "publish_mode": "draft",
            }

            result = await asyncio.to_thread(
                graph.invoke,
                initial_state,
                {"configurable": {"thread_id": thread_id}},
            )

            content_pack = result.get("content_pack", {})
            risk = result.get("risk_result", {})
            risk_score = risk.get("risk_score", 0)

            # Auto-publish if risk score < 4, otherwise save as draft
            publish_mode = "immediate" if risk_score < 4 else "draft"
            status = publish_mode

            publish_results = []
            if publish_mode == "immediate" and content_pack.get("title"):
                # Attempt to publish
                for plat in platforms:
                    try:
                        pub = _get_publisher(plat)
                        if pub:
                            pub_result = await asyncio.to_thread(
                                pub.publish if not asyncio.iscoroutinefunction(pub.publish) else None,
                                content_pack,
                            ) if not asyncio.iscoroutinefunction(pub.publish) else await pub.publish(content_pack)
                            publish_results.append({"platform": plat, **pub_result})
                    except Exception as pub_err:
                        publish_results.append({
                            "platform": plat,
                            "success": False,
                            "error": str(pub_err),
                        })
                status = "published" if publish_results else "draft"

            # Save to SQLite — posts + publish_log
            if content_pack.get("title"):
                with get_db_safe() as conn:
                    cur = conn.execute(
                        "INSERT INTO posts (title, body, platforms, status, extra_fields) VALUES (?, ?, ?, ?, ?)",
                        (
                            content_pack.get("title", ""),
                            content_pack.get("body", ""),
                            json.dumps(platforms),
                            status,
                            json.dumps({
                                "pipeline_thread": thread_id,
                                "source": "autopilot",
                                "niche": niche,
                                "risk_score": risk_score,
                                "hashtags": content_pack.get("hashtags", []),
                                "hook": content_pack.get("hook", ""),
                                "cta": content_pack.get("cta", ""),
                            }),
                        ),
                    )
                    new_post_id = cur.lastrowid
                    # Write publish_log entries for each platform result
                    for pr in publish_results:
                        conn.execute(
                            "INSERT INTO publish_log (post_id, platform, success, post_url, error) VALUES (?, ?, ?, ?, ?)",
                            (
                                new_post_id,
                                pr.get("platform", ""),
                                pr.get("success", False),
                                pr.get("post_url", ""),
                                pr.get("error", ""),
                            ),
                        )
                    conn.commit()
                _autopilot_posts_today += 1

            _autopilot_last_run = datetime.now(timezone.utc).isoformat()
            logger.info("autopilot.completed",
                        niche=niche,
                        title=content_pack.get("title", ""),
                        risk_score=risk_score,
                        status=status,
                        posts_today=_autopilot_posts_today)

        except Exception as e:
            logger.error("autopilot.error", error=str(e))
            try:
                await _alert_manager.send(
                    f"Autopilot error: {e}",
                    level=AlertLevel.CRITICAL,
                )
            except Exception:
                pass

        # Sleep for configured interval
        cfg = _get_autopilot_config()
        interval_sec = cfg.get("interval_hours", 4) * 3600
        await asyncio.sleep(interval_sec)


async def _watchdog():
    """Monitor background tasks and restart any that crash unexpectedly."""
    global _scheduler_task, _rss_tick_task, _blog_share_tick_task, _autopilot_task
    _task_factories = {
        "_scheduler_task": scheduler_loop,
        "_rss_tick_task": rss_tick_loop,
        "_blog_share_tick_task": blog_share_tick_loop,
        "_autopilot_task": autopilot_loop,
    }
    while True:
        for ref_name, factory in _task_factories.items():
            task = globals().get(ref_name)
            if task and task.done():
                exc = None
                if not task.cancelled():
                    try:
                        exc = task.exception()
                    except Exception:
                        pass
                logger.error("watchdog.restarting", task=ref_name, error=str(exc))
                try:
                    await _alert_manager.send(
                        f"Watchdog restarted crashed task: {ref_name}\nError: {exc}",
                        level=AlertLevel.WARNING,
                    )
                except Exception:
                    pass
                globals()[ref_name] = asyncio.create_task(factory())
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan — init DB and start scheduler + auto-tick loops + watchdog."""
    init_db()
    global _scheduler_task, _rss_tick_task, _blog_share_tick_task, _autopilot_task
    _scheduler_task = asyncio.create_task(scheduler_loop())
    _rss_tick_task = asyncio.create_task(rss_tick_loop())
    _blog_share_tick_task = asyncio.create_task(blog_share_tick_loop())
    _autopilot_task = asyncio.create_task(autopilot_loop())
    _watchdog_task = asyncio.create_task(_watchdog())
    logger.info("app.started", msg="Dashboard ready at http://localhost:8000")
    yield
    for task in (_scheduler_task, _rss_tick_task, _blog_share_tick_task, _autopilot_task, _watchdog_task):
        if task:
            task.cancel()


# ── FastAPI App ──
app = FastAPI(title="ViralOps Engine", lifespan=lifespan)


# ── API Key Authentication Middleware ──
_AUTH_EXEMPT = {"/api/health", "/docs", "/openapi.json", "/redoc"}


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Require VIRALOPS_API_KEY on all /api/ routes (except health).

    If VIRALOPS_API_KEY env var is not set, auth is disabled (dev mode).
    In production, set this to a secure random string.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth for non-API routes (HTML pages, static) and exempt paths
        if not path.startswith("/api/") or path in _AUTH_EXEMPT:
            return await call_next(request)

        expected_key = os.environ.get("VIRALOPS_API_KEY")
        if not expected_key:
            # Dev mode — no key = no auth
            return await call_next(request)

        # Check Authorization header (Bearer) or X-API-Key header
        auth_header = request.headers.get("authorization", "")
        api_key_header = request.headers.get("x-api-key", "")

        token = ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:]
        elif api_key_header:
            token = api_key_header

        if not token or not _secrets.compare_digest(token, expected_key):
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "error": "Unauthorized — provide VIRALOPS_API_KEY via Bearer token or X-API-Key header",
                },
            )

        return await call_next(request)


app.add_middleware(ApiKeyMiddleware)


# ── Global Exception Handler ──
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all: prevents raw tracebacks from leaking to clients."""
    logger.error("unhandled_error",
                 path=str(request.url.path),
                 method=request.method,
                 error=str(exc),
                 error_type=type(exc).__name__)
    is_dev = os.environ.get("VIRALOPS_ENV", "development") != "production"
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if is_dev else None,
        },
    )


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
    return templates.TemplateResponse(request, "app.html", {"page": "dashboard"})

@app.get("/compose", response_class=HTMLResponse)
async def compose(request: Request):
    with get_db_safe() as conn:
        cats = conn.execute("SELECT * FROM categories").fetchall()
    return templates.TemplateResponse(request, "app.html", {"page": "compose", "categories": cats})

@app.get("/content", response_class=HTMLResponse)
async def content_library(request: Request):
    return templates.TemplateResponse(request, "app.html", {"page": "content"})

@app.get("/calendar", response_class=HTMLResponse)
async def calendar(request: Request):
    return templates.TemplateResponse(request, "app.html", {"page": "calendar"})

@app.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request):
    return templates.TemplateResponse(request, "app.html", {"page": "analytics"})

@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    return templates.TemplateResponse(request, "app.html", {"page": "settings"})

@app.get("/rss", response_class=HTMLResponse)
async def rss_page(request: Request):
    return templates.TemplateResponse(request, "app.html", {"page": "rss"})

@app.get("/hashtags", response_class=HTMLResponse)
async def hashtags_page(request: Request):
    return templates.TemplateResponse(request, "app.html", {"page": "hashtags"})

@app.get("/platforms", response_class=HTMLResponse)
async def platforms_page(request: Request):
    return templates.TemplateResponse(request, "app.html", {"page": "platforms"})

@app.get("/engagement", response_class=HTMLResponse)
async def engagement_page(request: Request):
    return templates.TemplateResponse(request, "app.html", {"page": "engagement"})

@app.get("/time-slots", response_class=HTMLResponse)
async def timeslots_page(request: Request):
    return templates.TemplateResponse(request, "app.html", {"page": "timeslots"})

@app.get("/preview", response_class=HTMLResponse)
async def preview_page(request: Request):
    return templates.TemplateResponse(request, "app.html", {"page": "preview"})

@app.get("/autopilot", response_class=HTMLResponse)
async def autopilot_page(request: Request):
    return templates.TemplateResponse(request, "app.html", {"page": "autopilot"})

@app.get("/publer", response_class=HTMLResponse)
async def publer_page(request: Request):
    return templates.TemplateResponse(request, "app.html", {"page": "publer"})

@app.get("/blog-share", response_class=HTMLResponse)
async def blog_share_page(request: Request):
    return templates.TemplateResponse(request, "app.html", {"page": "blog-share"})

@app.get("/studio", response_class=HTMLResponse)
async def studio_page(request: Request):
    return templates.TemplateResponse(request, "app.html", {"page": "studio"})


# ════════════════════════════════════════════════════════════════
# API — Content Studio
# ════════════════════════════════════════════════════════════════

@app.get("/api/studio/options")
async def api_studio_options():
    """Return all available content creation methods, niches, and content packs."""
    from publish_microniche import (
        MICRO_NICHES, NANO_NICHES, REAL_LIFE_NICHES,
        PRE_WRITTEN_PACKS, NICHE_HUNTER_PACKS,
    )

    methods = [
        {"id": "auto", "label": "Auto (Weighted)", "icon": "🎲", "desc": "35% AI + 25% Hunter + 20% AI-Niche + 15% Pre-written + 5% Gemini"},
        {"id": "ai_generate", "label": "AI Generate", "icon": "🤖", "desc": "LLM Cascade (GitHub Models → Perplexity) + self-review"},
        {"id": "hunter_prewritten", "label": "Hunter Pre-written", "icon": "🏆", "desc": "Top-scored packs from Niche Hunter DB"},
        {"id": "ai_niche", "label": "AI + Niche DB", "icon": "🔬", "desc": "Top niche scores from DB + LLM content generation"},
        {"id": "prewritten", "label": "Pre-written Packs", "icon": "✍️", "desc": "24 curated content packs (8 manual + 16 scored)"},
        {"id": "gemini", "label": "Gemini Direct", "icon": "✨", "desc": "Gemini 2.5 Flash direct generation"},
        {"id": "niche_hunter", "label": "Niche Hunter", "icon": "🎯", "desc": "Scored questions from niche_hunter.db + Gemini"},
        {"id": "pain_point", "label": "Pain Point", "icon": "😤", "desc": "Urgent pain points with emotional triggers"},
    ]

    niches = {
        "micro": [{"id": n, "label": n.title()} for n in MICRO_NICHES],
        "nano": [{"id": n, "label": n.title()} for n in NANO_NICHES],
        "real_life": [{"id": n, "label": n.replace("_", " ").title()} for n in REAL_LIFE_NICHES],
    }

    # Niche scores from DB
    niche_scores = []
    try:
        import sqlite3
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "niche_hunter.db")
        if os.path.exists(db_path):
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT topic, niche, final_score, risk_level, hook, source FROM niche_scores ORDER BY final_score DESC LIMIT 30"
                ).fetchall()
                niche_scores = [dict(r) for r in rows]
    except Exception:
        pass

    packs = []
    for p in PRE_WRITTEN_PACKS:
        packs.append({
            "title": p.get("title", ""),
            "hook": p.get("hook", ""),
            "niche": p.get("_niche_key", ""),
            "score": None,
            "source": "pre_written",
        })
    for p in NICHE_HUNTER_PACKS:
        packs.append({
            "title": p.get("title", ""),
            "hook": p.get("hook", ""),
            "niche": p.get("_niche_key", ""),
            "score": p.get("_niche_score"),
            "source": "niche_hunter",
        })

    return {
        "methods": methods,
        "niches": niches,
        "niche_scores": niche_scores,
        "packs": packs,
    }


@app.post("/api/studio/generate")
async def api_studio_generate(request: Request):
    """
    Generate content from Content Studio.
    Body: {method, niche?, topic?, custom_idea?, pack_title?}
    """
    data = await request.json()
    method = data.get("method", "auto")
    niche = data.get("niche")
    topic = data.get("topic")
    custom_idea = data.get("custom_idea", "").strip()
    pack_title = data.get("pack_title")

    try:
        from publish_microniche import get_content_pack, ALL_PACKS

        # If a specific pack was selected, use it directly
        if pack_title:
            for p in ALL_PACKS:
                if p.get("title") == pack_title:
                    pack = dict(p)
                    pack["_source"] = "studio_pack_select"
                    return _studio_pack_response(pack)

        # If custom idea provided, use AI to generate from it
        if custom_idea:
            try:
                from llm_content import generate_content_pack as llm_gen
                pack = llm_gen(custom_idea, score=7.5)
                if pack:
                    pack["_source"] = "studio_custom_idea"
                    return _studio_pack_response(pack)
            except Exception as e:
                logger.warning("studio_custom_idea_fallback", error=str(e))

            # Fallback: use AI-generate with custom_idea as topic override
            import publish_microniche as pm
            old_niches = pm.MICRO_NICHES
            pm.MICRO_NICHES = [custom_idea]
            try:
                pack = get_content_pack("ai_generate")
                return _studio_pack_response(pack)
            finally:
                pm.MICRO_NICHES = old_niches

        # If niche selected, inject it as topic for the method
        if niche and method in ("ai_generate", "ai_niche", "gemini", "niche_hunter", "pain_point"):
            import publish_microniche as pm
            old_niches = pm.MICRO_NICHES
            pm.MICRO_NICHES = [niche]
            try:
                pack = get_content_pack(method)
                return _studio_pack_response(pack)
            finally:
                pm.MICRO_NICHES = old_niches

        # Default: use the selected method as-is
        pack = get_content_pack(method)
        return _studio_pack_response(pack)

    except Exception as e:
        logger.error("studio_generate_error", error=str(e))
        return {"error": str(e)}


def _studio_pack_response(pack: dict) -> dict:
    """Normalize a content pack into a consistent API response."""
    return {
        "title": pack.get("title", "Untitled"),
        "hook": pack.get("hook", ""),
        "body": pack.get("body", ""),
        "key_fact": pack.get("key_fact", ""),
        "three_steps": pack.get("three_steps", []),
        "result": pack.get("result", ""),
        "cta": pack.get("cta", ""),
        "hashtags": pack.get("hashtags", []),
        "source": pack.get("_source", "unknown"),
        "niche": pack.get("_niche_key", ""),
        "score": pack.get("_niche_score") or pack.get("_review_score"),
        "location": pack.get("_location", ""),
        "season": pack.get("_season", ""),
    }


# ════════════════════════════════════════════════════════════════
# API — Posts
# ════════════════════════════════════════════════════════════════

@app.get("/api/posts")
async def api_list_posts(status: Optional[str] = None):
    with get_db_safe() as conn:
        if status and status != "all":
            rows = conn.execute("SELECT * FROM posts WHERE status = ? ORDER BY id DESC", (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM posts ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]

@app.post("/api/posts")
async def api_create_post(request: Request):
    data = await request.json()
    with get_db_safe() as conn:
        conn.execute(
            "INSERT INTO posts (title, body, category, platforms, status, scheduled_at, extra_fields) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (data.get("title", ""), data.get("body", ""), data.get("category", "general"),
             json.dumps(data.get("platforms", [])), data.get("status", "draft"),
             data.get("scheduled_at", ""), json.dumps(data.get("extra_fields", {})))
        )
        conn.commit()
        post_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return {"success": True, "post_id": post_id}

@app.delete("/api/posts/{post_id}")
async def api_delete_post(post_id: int):
    with get_db_safe() as conn:
        conn.execute("DELETE FROM posts WHERE id = ?", (post_id,))
        conn.commit()
    return {"success": True}

@app.put("/api/posts/{post_id}")
async def api_update_post(post_id: int, request: Request):
    data = await request.json()
    with get_db_safe() as conn:
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
# API — Full LangGraph Pipeline (EMADS-PR)
# ════════════════════════════════════════════════════════════════

@app.post("/api/pipeline/run")
async def api_pipeline_run(request: Request):
    """
    Run the full EMADS-PR LangGraph pipeline:
    Orchestrator → ContentFactory → [Compliance+Risk+Rights+Cost] PARALLEL
    → ReconcileGPT → Human Review → Publish → Monitor

    Body: {niche, topic?, platforms?, publish_mode?}
    publish_mode: "draft" | "scheduled" | "immediate"
    """
    data = await request.json()
    niche = data.get("niche", "general")
    topic = data.get("topic")
    platforms = data.get("platforms", ["tiktok", "pinterest"])
    publish_mode = data.get("publish_mode", "draft")

    try:
        from graph import get_compiled_graph
        import uuid

        graph = get_compiled_graph()
        thread_id = f"web-{uuid.uuid4().hex[:8]}"

        initial_state = {
            "niche_config": {"niche": niche, "sub_niche": niche},
            "topic": topic,
            "platforms": platforms,
            "publish_mode": publish_mode,
        }

        # Run the full pipeline in a thread (LangGraph is sync)
        result = await asyncio.to_thread(
            graph.invoke,
            initial_state,
            {"configurable": {"thread_id": thread_id}},
        )

        # Extract key results
        content_pack = result.get("content_pack", {})
        reconcile = result.get("reconcile_result", {})
        risk = result.get("risk_result", {})
        cost = result.get("cost_result", {})
        publish_results = result.get("publish_results", [])

        # Save to SQLite if content was generated — posts + publish_log
        if content_pack.get("title"):
            with get_db_safe() as conn:
                cur = conn.execute(
                    "INSERT INTO posts (title, body, platforms, status, extra_fields) VALUES (?, ?, ?, ?, ?)",
                    (
                        content_pack.get("title", ""),
                        content_pack.get("body", ""),
                        json.dumps(platforms),
                        "published" if publish_results else publish_mode,
                        json.dumps({
                            "pipeline_thread": thread_id,
                            "reconcile_summary": reconcile.get("summary", ""),
                            "risk_score": risk.get("risk_score", 0),
                            "hashtags": content_pack.get("hashtags", []),
                            "hook": content_pack.get("hook", ""),
                            "cta": content_pack.get("cta", ""),
                        }),
                    ),
                )
                new_post_id = cur.lastrowid
                # Write publish_log entries for each platform result from the pipeline
                for pr in publish_results:
                    conn.execute(
                        "INSERT INTO publish_log (post_id, platform, success, post_url, error) VALUES (?, ?, ?, ?, ?)",
                        (
                            new_post_id,
                            pr.get("platform", ""),
                            pr.get("status") == "published",
                            pr.get("post_url", ""),
                            pr.get("detail", "") if pr.get("status") == "failed" else "",
                        ),
                    )
                conn.commit()

        return {
            "success": True,
            "thread_id": thread_id,
            "content_pack": {
                "title": content_pack.get("title", ""),
                "body": content_pack.get("body", ""),
                "hook": content_pack.get("hook", ""),
                "cta": content_pack.get("cta", ""),
                "hashtags": content_pack.get("hashtags", []),
            },
            "reconcile": {
                "summary": reconcile.get("summary", ""),
                "risk_level": reconcile.get("risk_level", "low"),
            },
            "risk_score": risk.get("risk_score", 0),
            "cost": cost.get("estimated_cost_usd", 0),
            "publish_mode": publish_mode,
            "publish_results": publish_results,
            "errors": result.get("errors", []),
        }

    except Exception as e:
        logger.error("pipeline.run_error", error=str(e))
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


# ════════════════════════════════════════════════════════════════
# API — Autopilot Config
# ════════════════════════════════════════════════════════════════


@app.get("/api/autopilot/config")
async def api_autopilot_config_get():
    """Return current autopilot configuration."""
    return _get_autopilot_config()


@app.put("/api/autopilot/config")
async def api_autopilot_config_put(request: Request):
    """Update autopilot configuration.

    Body: {enabled?, interval_hours?, niches?, platforms?, max_posts_per_day?}
    """
    data = await request.json()
    current = _get_autopilot_config()
    # Merge incoming fields over current config
    for key in ("enabled", "interval_hours", "niches", "platforms", "max_posts_per_day"):
        if key in data:
            current[key] = data[key]
    _save_autopilot_config(current)
    return current


@app.get("/api/autopilot/status")
async def api_autopilot_status():
    """Return autopilot runtime status."""
    cfg = _get_autopilot_config()
    return {
        "running": cfg.get("enabled", False) and _autopilot_task is not None
                   and not _autopilot_task.done(),
        "last_run": _autopilot_last_run,
        "posts_today": _autopilot_posts_today,
        "posts_today_date": _autopilot_posts_today_date,
        "config": cfg,
    }


# ════════════════════════════════════════════════════════════════
# API — Alerts
# ════════════════════════════════════════════════════════════════


@app.get("/api/alerts/history")
async def api_alerts_history(limit: int = 50):
    """Return recent alert history from the in-memory AlertManager."""
    history = list(_alert_manager._history)[-limit:]
    return {"alerts": list(reversed(history)), "total": len(_alert_manager._history)}


# ════════════════════════════════════════════════════════════════
# API — Publish
# ════════════════════════════════════════════════════════════════

# Platforms that Publer can publish to (use Publer as unified publisher)
_PUBLER_PLATFORMS = {
    "tiktok", "instagram", "facebook", "twitter", "linkedin",
    "youtube", "pinterest", "threads", "bluesky", "mastodon",
    "telegram", "google_business",
}


def _get_publisher(platform: str):
    """Lazy-load publisher for a platform. Publer handles 12+ social platforms."""
    # Priority 1: Use Publer for all supported social platforms
    if platform in _PUBLER_PLATFORMS:
        from integrations.publer_publisher import PublerPublisher
        pub = PublerPublisher()
        if pub.is_configured:
            return pub
        # Fall through to direct publishers if Publer not configured

    # Priority 2: Direct API publishers
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
    elif platform == "lemon8":
        from integrations.lemon8_publisher import Lemon8Publisher
        return Lemon8Publisher()
    elif platform == "quora":
        from integrations.quora_publisher import QuoraPublisher
        return QuoraPublisher()
    else:
        raise ValueError(f"Unknown platform: {platform}")


class _SimpleQueueItem:
    """Duck-typed queue item for publisher compatibility."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@app.post("/api/publish/{post_id}")
async def api_publish_post(post_id: int, request: Request):
    """Publish a post with EMADS-PR safety gate.

    Runs risk assessment before publishing. If risk_score >= 4,
    returns 403 unless ?force=true is passed.
    """
    # Read post from DB (short-lived connection for reads)
    with get_db_safe() as conn:
        post = conn.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
    if not post:
        raise HTTPException(404, "Post not found")

    platforms = json.loads(post["platforms"] or "[]")
    extra = json.loads(post["extra_fields"] or "{}")

    # ── EMADS-PR Safety Gate: risk assessment before publishing ──
    try:
        from agents.risk_health import assess_risk
        safety_state = {
            "platforms": platforms,
            "content_pack": {
                "title": post["title"],
                "body": post["body"],
                **{k: v for k, v in extra.items() if not isinstance(v, dict)},
            },
            "replan_count": 0,
        }
        safety_state = assess_risk(safety_state)
        risk_result = safety_state.get("risk_result", {})
        risk_score = risk_result.get("risk_score", 0)
    except Exception as e:
        logger.warning("publish.risk_check_failed", error=str(e))
        risk_score = 0
        risk_result = {"risk_score": 0, "error": str(e)}

    # Block if risk >= 4 unless force=true
    force = request.query_params.get("force", "").lower() == "true"
    if risk_score >= 4 and not force:
        await _alert_manager.publish_failed(
            "all", f"Publish blocked: risk_score={risk_score} for post {post_id}"
        )
        return JSONResponse(
            status_code=403,
            content={
                "success": False,
                "error": "Risk score too high — human review required",
                "risk_score": risk_score,
                "risk_factors": risk_result.get("risk_factors", []),
                "requires_human_review": True,
                "hint": "Add ?force=true to override after review",
            },
        )

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
            # Alert on publish failure
            await _alert_manager.publish_failed(platform, error)

        results.append({"platform": platform, "success": success, "post_url": post_url, "error": error})

    # Write publish_log + update status (short-lived connection for writes)
    all_ok = all(r["success"] for r in results)
    with get_db_safe() as conn:
        for r in results:
            conn.execute(
                "INSERT INTO publish_log (post_id, platform, success, post_url, error) VALUES (?, ?, ?, ?, ?)",
                (post_id, r["platform"], r["success"], r["post_url"], r["error"])
            )
        conn.execute(
            "UPDATE posts SET status = ?, published_at = ? WHERE id = ?",
            ("published" if all_ok else "failed", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), post_id)
        )
        conn.commit()
    return {"success": all_ok, "risk_score": risk_score, "results": results}


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
    with get_db_safe() as conn:
        total = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        published = conn.execute("SELECT COUNT(*) FROM posts WHERE status='published'").fetchone()[0]
        scheduled = conn.execute("SELECT COUNT(*) FROM posts WHERE status='scheduled'").fetchone()[0]
        drafts = conn.execute("SELECT COUNT(*) FROM posts WHERE status='draft'").fetchone()[0]
        failed = conn.execute("SELECT COUNT(*) FROM posts WHERE status='failed'").fetchone()[0]
    return {"total": total, "published": published, "scheduled": scheduled, "drafts": drafts, "failed": failed}


@app.get("/api/calendar-events")
async def api_calendar_events(month: Optional[str] = None):
    with get_db_safe() as conn:
        rows = conn.execute("SELECT id, title, platforms, status, scheduled_at, published_at FROM posts WHERE scheduled_at IS NOT NULL AND scheduled_at != ''").fetchall()
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
    with get_db_safe() as conn:
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
    with get_db_safe() as conn:
        conn.execute(
            "INSERT INTO posts (title, body, category, platforms, status, extra_fields) VALUES (?, ?, ?, ?, ?, ?)",
            (draft["title"], draft["body"], draft["category"], draft["platforms"], draft["status"], draft["extra_fields"])
        )
        conn.commit()
        post_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return {"success": True, "post_id": post_id}


# ── Bulk Import (200-500 posts / call) ──

@app.post("/api/rss/bulk-import")
async def api_rss_bulk_import(request: Request):
    """
    Bulk import from ALL configured RSS feeds into SQLite drafts.
    200-500 posts/day in ONE API call.
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
    with get_db_safe() as conn:
        rows = conn.execute("SELECT * FROM hashtag_collections ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]

@app.post("/api/hashtags")
async def api_create_hashtag_collection(request: Request):
    data = await request.json()
    with get_db_safe() as conn:
        conn.execute(
            "INSERT INTO hashtag_collections (name, description, hashtags, platforms, niche) VALUES (?, ?, ?, ?, ?)",
            (data.get("name", ""), data.get("description", ""),
             json.dumps(data.get("hashtags", [])), json.dumps(data.get("platforms", [])),
             data.get("niche", "general"))
        )
        conn.commit()
        cid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return {"success": True, "id": cid}

@app.put("/api/hashtags/{collection_id}")
async def api_update_hashtag_collection(collection_id: int, request: Request):
    data = await request.json()
    with get_db_safe() as conn:
        conn.execute(
            "UPDATE hashtag_collections SET name=?, description=?, hashtags=?, platforms=?, niche=?, updated_at=datetime('now') WHERE id=?",
            (data.get("name"), data.get("description"), json.dumps(data.get("hashtags", [])),
             json.dumps(data.get("platforms", [])), data.get("niche"), collection_id)
        )
        conn.commit()
    return {"success": True}

@app.delete("/api/hashtags/{collection_id}")
async def api_delete_hashtag_collection(collection_id: int):
    with get_db_safe() as conn:
        conn.execute("DELETE FROM hashtag_collections WHERE id = ?", (collection_id,))
        conn.commit()
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
# API — RSS Auto Poster
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
    Create a new RSS Auto Poster.

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
    """Real liveness/readiness probe — checks background tasks + DB."""
    checks = {}

    # Check each background task
    for name, task in [
        ("scheduler", _scheduler_task),
        ("rss_tick", _rss_tick_task),
        ("blog_share", _blog_share_tick_task),
        ("autopilot", _autopilot_task),
    ]:
        if task is None:
            checks[name] = "not_started"
        elif task.done():
            exc = None
            if not task.cancelled():
                try:
                    exc = task.exception()
                except Exception:
                    pass
            checks[name] = f"DEAD: {exc}" if exc else "cancelled"
        else:
            checks[name] = "running"

    # DB connectivity
    try:
        with get_db_safe() as conn:
            conn.execute("SELECT 1").fetchone()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"FAIL: {e}"

    all_healthy = all(
        v in ("running", "ok") for v in checks.values()
    )
    status_code = 200 if all_healthy else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if all_healthy else "degraded",
            "checks": checks,
            "version": "3.5.0",
            "engine": "ViralOps Engine — EMADS-PR v1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ════════════════════════════════════════════════════════════════
# API — TikTok Auto Music
# ════════════════════════════════════════════════════════════════

@app.post("/api/tiktok/music/recommend")
async def api_tiktok_music_recommend(request: Request):
    """
    🎵 Auto-recommend TikTok music based on content + niche + mood.

    ViralOps: AI auto-recommends music per post based on content analysis.
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
# API — Shopify Blog Auto-Share (→ TikTok Multi-Account + Pinterest)
# ════════════════════════════════════════════════════════════════

@app.get("/api/blog-share/status")
async def api_blog_share_status():
    """Get auto-share status + stats."""
    from integrations.shopify_auto_share import auto_share_status
    return await auto_share_status()

@app.get("/api/blog-share/config")
async def api_blog_share_config():
    """Get current auto-share configuration."""
    from integrations.shopify_auto_share import auto_share_config
    return await auto_share_config()

@app.put("/api/blog-share/config")
async def api_blog_share_update_config(request: Request):
    """Update auto-share configuration."""
    data = await request.json()
    from integrations.shopify_auto_share import auto_share_update_config
    return await auto_share_update_config(data)

@app.post("/api/blog-share/tick")
async def api_blog_share_tick():
    """Manually trigger one auto-share cycle."""
    from integrations.shopify_auto_share import auto_share_tick
    return await auto_share_tick()

@app.post("/api/blog-share/pause")
async def api_blog_share_pause():
    """Pause auto-sharing."""
    from integrations.shopify_auto_share import auto_share_pause
    return await auto_share_pause()

@app.post("/api/blog-share/resume")
async def api_blog_share_resume():
    """Resume auto-sharing."""
    from integrations.shopify_auto_share import auto_share_resume
    return await auto_share_resume()

@app.get("/api/blog-share/history")
async def api_blog_share_history(limit: int = 50):
    """Get share history."""
    from integrations.shopify_auto_share import auto_share_history
    return await auto_share_history(limit)

@app.delete("/api/blog-share/history")
async def api_blog_share_clear_history():
    """Clear share history (allows re-sharing)."""
    from integrations.shopify_auto_share import get_auto_share
    instance = await get_auto_share()
    return instance.clear_history()

@app.post("/api/blog-share/manual")
async def api_blog_share_manual(request: Request):
    """Manually share a specific article by URL."""
    data = await request.json()
    from integrations.shopify_auto_share import auto_share_manual
    return await auto_share_manual(
        data.get("url", ""),
        force=data.get("force", False),
    )

@app.post("/api/blog-share/latest")
async def api_blog_share_latest(request: Request):
    """Share the latest article(s) from a specific blog."""
    data = await request.json()
    from integrations.shopify_auto_share import auto_share_latest
    return await auto_share_latest(
        blog_handle=data.get("blog_handle", "sustainable-living"),
        count=data.get("count", 1),
        force=data.get("force", False),
    )

@app.get("/api/blog-share/tiktok/accounts")
async def api_blog_share_tiktok_accounts():
    """List all configured TikTok accounts."""
    from integrations.multi_tiktok_publisher import MultiTikTokPublisher
    pub = MultiTikTokPublisher()
    return pub.status()

@app.post("/api/blog-share/tiktok/test")
async def api_blog_share_tiktok_test():
    """Test all TikTok account connections."""
    from integrations.multi_tiktok_publisher import MultiTikTokPublisher
    pub = MultiTikTokPublisher()
    results = await pub.test_all_accounts()
    return {"results": results}

@app.get("/api/blog-share/watcher/blogs")
async def api_blog_share_watcher_blogs():
    """List all blogs in the Shopify store."""
    from integrations.shopify_blog_watcher import ShopifyBlogWatcher
    watcher = ShopifyBlogWatcher()
    connected = await watcher.connect()
    if not connected:
        return {"error": "Failed to connect to Shopify"}
    blogs = await watcher.list_all_blogs()
    await watcher.close()
    return {"blogs": blogs}

@app.post("/api/blog-share/watcher/reset")
async def api_blog_share_watcher_reset(request: Request):
    """Reset watcher state (re-fetch all articles on next check)."""
    data = await request.json() if await request.body() else {}
    from integrations.shopify_auto_share import get_auto_share
    instance = await get_auto_share()
    if instance._watcher:
        return instance._watcher.reset_state(data.get("blog_handle"))
    return {"error": "Watcher not initialized"}


# ════════════════════════════════════════════════════════════════
# API — TikTok Multi-Account Management
# Round-robin posting: 3 posts/account/day, N accounts
# ════════════════════════════════════════════════════════════════

@app.get("/api/tiktok/accounts")
async def api_tiktok_accounts():
    """List all TikTok accounts with daily stats."""
    try:
        from core.tiktok_accounts import get_account_manager
        mgr = get_account_manager()
        return {"accounts": mgr.get_all(), "stats": mgr.get_stats()}
    except Exception as e:
        return {"accounts": [], "error": str(e)}


@app.post("/api/tiktok/accounts")
async def api_tiktok_accounts_add(request: Request):
    """Add a new TikTok account for round-robin posting."""
    try:
        body = await request.json()
        account_id = body.get("id", "").strip()
        if not account_id:
            return {"success": False, "error": "Account ID is required"}
        from core.tiktok_accounts import get_account_manager
        mgr = get_account_manager()
        result = mgr.add_account(
            account_id=account_id,
            label=body.get("label", ""),
            max_daily=int(body.get("max_daily", 3)),
            niche_filter=body.get("niche_filter", ""),
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.put("/api/tiktok/accounts/{account_id}")
async def api_tiktok_accounts_update(account_id: str, request: Request):
    """Update a TikTok account (label, enabled, max_daily, niche_filter)."""
    try:
        body = await request.json()
        from core.tiktok_accounts import get_account_manager
        mgr = get_account_manager()
        return mgr.update_account(account_id, body)
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.delete("/api/tiktok/accounts/{account_id}")
async def api_tiktok_accounts_delete(account_id: str):
    """Remove a TikTok account from round-robin."""
    try:
        from core.tiktok_accounts import get_account_manager
        mgr = get_account_manager()
        return mgr.remove_account(account_id)
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/tiktok/accounts/stats")
async def api_tiktok_accounts_stats():
    """Get multi-account posting stats."""
    try:
        from core.tiktok_accounts import get_account_manager
        mgr = get_account_manager()
        return mgr.get_stats()
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/tiktok/accounts/reorder")
async def api_tiktok_accounts_reorder(request: Request):
    """Reorder accounts (affects round-robin sequence)."""
    try:
        body = await request.json()
        ordered_ids = body.get("ordered_ids", [])
        if not ordered_ids:
            return {"success": False, "error": "ordered_ids array required"}
        from core.tiktok_accounts import get_account_manager
        mgr = get_account_manager()
        return mgr.reorder_accounts(ordered_ids)
    except Exception as e:
        return {"success": False, "error": str(e)}


# ════════════════════════════════════════════════════════════════
# API — Publer Bridge (REST API → TikTok/IG/FB/Pinterest/etc.)
# ~$10/mo per social account (Business plan)
# ════════════════════════════════════════════════════════════════

@app.get("/api/publer/status")
async def api_publer_status():
    """Check Publer REST API status (API key configured)."""
    import os
    api_key = bool(os.environ.get("PUBLER_API_KEY"))
    workspace_id = bool(os.environ.get("PUBLER_WORKSPACE_ID"))
    return {
        "configured": api_key,
        "auth_mode": "rest_api" if api_key else "none",
        "workspace_id_set": workspace_id,
        "method": "publer_rest_api",
    }


@app.post("/api/publer/test")
async def api_publer_test():
    """Test Publer REST API connection."""
    try:
        from integrations.publer_publisher import PublerPublisher
        pub = PublerPublisher()
        result = await pub.test_connection()
        await pub.close()
        return result
    except Exception as e:
        return {"connected": False, "error": str(e)}


@app.get("/api/publer/accounts")
async def api_publer_accounts():
    """List connected social accounts via Publer API."""
    try:
        from integrations.publer_publisher import PublerPublisher
        pub = PublerPublisher()
        connected = await pub.connect()
        if not connected:
            return {"accounts": [], "error": "Not connected — check PUBLER_API_KEY"}
        accounts = await pub.get_accounts(force=True)
        await pub.close()
        return {"accounts": accounts, "count": len(accounts)}
    except Exception as e:
        return {"accounts": [], "error": str(e)}


@app.post("/api/publer/publish")
async def api_publer_publish(request: Request):
    """Publish content via Publer REST API."""
    try:
        body = await request.json()
        from integrations.publer_publisher import PublerPublisher
        pub = PublerPublisher()
        result = await pub.publish(body)
        await pub.close()
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/publer/workspaces")
async def api_publer_workspaces():
    """List Publer workspaces."""
    try:
        from integrations.publer_publisher import PublerPublisher
        pub = PublerPublisher()
        await pub.connect()
        workspaces = await pub.get_workspaces(force=True)
        await pub.close()
        return {"workspaces": workspaces, "count": len(workspaces)}
    except Exception as e:
        return {"workspaces": [], "error": str(e)}


# ════════════════════════════════════════════════════════════════
# Run
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
