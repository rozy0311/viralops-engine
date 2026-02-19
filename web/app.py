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

import httpx
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
from monitoring.account_health import AccountHealthMonitor

logger = structlog.get_logger()

# ── Global singletons ──
_alert_manager = AlertManager()
_account_health = AccountHealthMonitor()

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
                    logger.warning("init_db.migrating",
                                   msg="Legacy TEXT-id schema detected — backing up before migration")
                    # Backup existing data before destructive migration
                    backup_path = DB_PATH.replace(".db", f"_backup_migration.db") if isinstance(DB_PATH, str) else str(DB_PATH) + ".backup"
                    try:
                        import shutil
                        shutil.copy2(DB_PATH, backup_path)
                        logger.info("init_db.backup_created", path=backup_path)
                    except Exception as backup_err:
                        logger.warning("init_db.backup_failed", error=str(backup_err))
                    conn.executescript("""
                        DROP TABLE IF EXISTS posts;
                        DROP TABLE IF EXISTS publish_log;
                    """)
                    logger.info("init_db.migration_complete",
                                msg="Legacy tables dropped, will recreate with INTEGER AUTOINCREMENT")
        except Exception as e:
            logger.warning("init_db.migration_check", error=str(e))

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
            CREATE TABLE IF NOT EXISTS content_packs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                hook TEXT DEFAULT '',
                pain_point TEXT DEFAULT '',
                key_fact TEXT DEFAULT '',
                audiences TEXT DEFAULT '[]',
                steps TEXT DEFAULT '[]',
                result TEXT DEFAULT '',
                cta TEXT DEFAULT '',
                hashtags TEXT DEFAULT '[]',
                image_title TEXT DEFAULT '',
                image_subtitle TEXT DEFAULT '',
                image_steps TEXT DEFAULT '',
                colors TEXT DEFAULT '',
                niche_key TEXT DEFAULT '',
                niche_score REAL,
                source TEXT DEFAULT 'manual',
                enabled INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS autopilot_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT DEFAULT (datetime('now')),
                finished_at TEXT,
                niche TEXT DEFAULT '',
                platforms TEXT DEFAULT '[]',
                thread_id TEXT DEFAULT '',
                risk_score REAL DEFAULT 0,
                action TEXT DEFAULT 'skip',
                post_id INTEGER,
                title TEXT DEFAULT '',
                error TEXT DEFAULT '',
                duration_ms INTEGER DEFAULT 0,
                posts_today INTEGER DEFAULT 0
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

        # ── Auto-seed content_packs from hardcoded packs (once) ──
        pack_count = conn.execute("SELECT COUNT(*) FROM content_packs").fetchone()[0]
        if pack_count == 0:
            _seed_content_packs(conn)


def _seed_content_packs(conn):
    """Migrate hardcoded packs from publish_microniche.py into content_packs table."""
    try:
        from publish_microniche import PRE_WRITTEN_PACKS, NICHE_HUNTER_PACKS
        all_packs = [
            (p, "pre_written") for p in PRE_WRITTEN_PACKS
        ] + [
            (p, "niche_hunter") for p in NICHE_HUNTER_PACKS
        ]
        for pack, source in all_packs:
            conn.execute(
                """INSERT INTO content_packs
                   (title, hook, pain_point, key_fact, audiences, steps, result, cta,
                    hashtags, image_title, image_subtitle, image_steps, colors,
                    niche_key, niche_score, source)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    pack.get("title", ""),
                    pack.get("hook", ""),
                    pack.get("pain_point", ""),
                    pack.get("key_fact", ""),
                    json.dumps(pack.get("audiences", [])),
                    json.dumps(pack.get("steps", pack.get("three_steps", []))),
                    pack.get("result", ""),
                    pack.get("cta", ""),
                    json.dumps(pack.get("hashtags", [])),
                    pack.get("image_title", ""),
                    pack.get("image_subtitle", ""),
                    pack.get("image_steps", ""),
                    json.dumps(pack.get("colors", "")),
                    pack.get("_niche_key", ""),
                    pack.get("_niche_score"),
                    source,
                ),
            )
        conn.commit()
        logger.info("seed_content_packs", count=len(all_packs))
    except Exception as e:
        logger.error("seed_content_packs_error", error=str(e))


def _pack_row_to_dict(row) -> dict:
    """Convert a content_packs DB row to a dict suitable for API responses."""
    d = dict(row)
    for field in ("audiences", "steps", "hashtags"):
        if isinstance(d.get(field), str):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                d[field] = []
    if isinstance(d.get("colors"), str):
        try:
            d["colors"] = json.loads(d["colors"]) if d["colors"] else ""
        except (json.JSONDecodeError, TypeError):
            d["colors"] = ""
    return d


def _pack_row_to_publish_format(row) -> dict:
    """Convert a content_packs DB row to the format publish_microniche.py expects."""
    d = _pack_row_to_dict(row)
    pack = {
        "title": d["title"],
        "hook": d.get("hook", ""),
        "pain_point": d.get("pain_point", ""),
        "key_fact": d.get("key_fact", ""),
        "audiences": d.get("audiences", []),
        "steps": d.get("steps", []),
        "three_steps": d.get("steps", []),
        "result": d.get("result", ""),
        "cta": d.get("cta", ""),
        "hashtags": d.get("hashtags", []),
        "image_title": d.get("image_title", ""),
        "image_subtitle": d.get("image_subtitle", ""),
        "image_steps": d.get("image_steps", ""),
        "colors": d.get("colors", ""),
        "_niche_key": d.get("niche_key", ""),
        "_niche_score": d.get("niche_score"),
        "_source": d.get("source", "db"),
        "_db_id": d.get("id"),
    }
    return pack


def get_all_db_packs(source_filter: str = None, enabled_only: bool = True) -> list[dict]:
    """Read content packs from DB, returning publish_microniche-compatible dicts."""
    with get_db_safe() as conn:
        if source_filter:
            rows = conn.execute(
                "SELECT * FROM content_packs WHERE source = ?" + (" AND enabled = 1" if enabled_only else "") + " ORDER BY niche_score DESC NULLS LAST, id ASC",
                (source_filter,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM content_packs" + (" WHERE enabled = 1" if enabled_only else "") + " ORDER BY niche_score DESC NULLS LAST, id ASC"
            ).fetchall()
    return [_pack_row_to_publish_format(r) for r in rows]


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
    "platforms": ["tiktok", "facebook", "pinterest"],
    "max_posts_per_day": 15,
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
        import time as _time
        _run_start = _time.monotonic()
        _run_action = "skip"
        _run_error = ""
        _run_post_id = None
        _run_title = ""
        _run_risk_score = 0
        _run_niche = ""
        _run_platforms: list = []
        _run_thread_id = ""

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
            if _autopilot_posts_today >= cfg.get("max_posts_per_day", 15):
                logger.info("autopilot.daily_limit_reached",
                            count=_autopilot_posts_today)
                await asyncio.sleep(600)  # Check again in 10 min
                continue

            # Pick niche from rotation
            niches = cfg.get("niches") or ["general"]
            niche = niches[_niche_index % len(niches)] if niches else "general"
            _niche_index += 1
            platforms = cfg.get("platforms", ["tiktok", "facebook", "pinterest"])

            logger.info("autopilot.generating", niche=niche, platforms=platforms)

            _run_niche = niche
            _run_platforms = platforms

            # ── Account health safety gate — skip unsafe platforms ──
            safe_platforms = []
            for plat in platforms:
                safe, reason = _account_health.is_safe_to_post(plat)
                if safe:
                    safe_platforms.append(plat)
                else:
                    logger.warning("autopilot.health_blocked",
                                   platform=plat, reason=reason)
            if not safe_platforms:
                _run_action = "health_blocked"
                _run_error = "All platforms blocked by health check"
                logger.warning("autopilot.all_platforms_blocked",
                               platforms=platforms)
                # Record blocked run, then sleep and continue
                _run_duration = int((_time.monotonic() - _run_start) * 1000)
                try:
                    with get_db_safe() as conn:
                        conn.execute(
                            """INSERT INTO autopilot_runs
                               (finished_at, niche, platforms, thread_id, risk_score,
                                action, post_id, title, error, duration_ms, posts_today)
                               VALUES (datetime('now'), ?, ?, ?, 0, 'health_blocked', NULL, '', ?, ?, ?)""",
                            (_run_niche, json.dumps(_run_platforms), '',
                             _run_error, _run_duration, _autopilot_posts_today),
                        )
                        conn.commit()
                except Exception:
                    pass
                # Fall through to sleep
                try:
                    cfg = _get_autopilot_config()
                    interval_sec = max(1800, cfg.get("interval_hours", 4) * 3600)
                    await asyncio.sleep(interval_sec)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    await asyncio.sleep(14400)
                continue
            platforms = safe_platforms  # Use only safe platforms

            from graph import get_compiled_graph
            import uuid

            graph = get_compiled_graph()
            thread_id = f"autopilot-{uuid.uuid4().hex[:8]}"
            _run_thread_id = thread_id

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
                # Attempt to publish via platform-specific publisher
                import inspect

                for plat in platforms:
                    try:
                        pub = _get_publisher(plat)
                        if not pub:
                            continue

                        # ── TikTok: QUALITY UPGRADE + adapt content + pick account ──
                        if plat == "tiktok":
                            try:
                                # If content_pack lacks quality content (content_formatted),
                                # upgrade via generate_quality_post() — full 3500-4000 char
                                # Perplexity-style content with review loop
                                if not content_pack.get("content_formatted") or \
                                   len(content_pack.get("content_formatted", "")) < 500:
                                    topic = content_pack.get("title", "") or \
                                            result.get("topic", "") or niche
                                    logger.info("tiktok.quality_upgrade",
                                                topic=topic[:60],
                                                reason="content_formatted missing or short")
                                    try:
                                        from llm_content import generate_quality_post
                                        quality_pack = await asyncio.to_thread(
                                            generate_quality_post,
                                            topic=topic,
                                            score=content_pack.get("_niche_score", 8.0),
                                        )
                                        if quality_pack and quality_pack.get("content_formatted") and \
                                           len(quality_pack["content_formatted"]) >= 2000:
                                            # Merge quality fields into content_pack
                                            content_pack["content_formatted"] = quality_pack["content_formatted"]
                                            content_pack["universal_caption_block"] = quality_pack.get("universal_caption_block", "")
                                            if quality_pack.get("title"):
                                                content_pack["title"] = quality_pack["title"]
                                            if quality_pack.get("hashtags"):
                                                content_pack["hashtags"] = quality_pack["hashtags"]
                                            if quality_pack.get("image_prompt"):
                                                content_pack["image_prompt"] = quality_pack.get("image_prompt", "")
                                            # Carry over review metadata
                                            for k in ("_review_score", "_review_pass",
                                                       "_review_feedback", "_content_chars",
                                                       "_location", "_season",
                                                       "_rubric_total_100", "_rubric_pass", "_rubric_scores"):
                                                if k in quality_pack:
                                                    content_pack[k] = quality_pack[k]
                                            logger.info("tiktok.quality_upgrade_ok",
                                                        chars=len(content_pack["content_formatted"]),
                                                        review_score=quality_pack.get("_review_score", 0))
                                        else:
                                            logger.warning("tiktok.quality_upgrade_weak",
                                                           chars=len(quality_pack.get("content_formatted", "")) if quality_pack else 0)
                                    except Exception as qe:
                                        logger.warning("tiktok.quality_upgrade_error",
                                                       error=str(qe)[:200])

                                # ── QUALITY GATE — block low-quality content ──
                                # ReconcileGPT Phase 2: review metadata MUST pass
                                # before we publish to TikTok.  If content fails
                                # these checks it is saved as draft, NOT published.
                                _qg_score = content_pack.get("_review_score", 0)
                                _qg_pass = content_pack.get("_review_pass", False)
                                _qg_chars = len(content_pack.get("content_formatted", ""))
                                _QG_MIN_SCORE = 8.0   # absolute floor
                                _QG_MIN_CHARS = 1500   # minimum content length

                                if not _qg_pass or _qg_score < _QG_MIN_SCORE or _qg_chars < _QG_MIN_CHARS:
                                    _qg_reason = (
                                        f"score={_qg_score:.1f} (need {_QG_MIN_SCORE}+), "
                                        f"chars={_qg_chars} (need {_QG_MIN_CHARS}+), "
                                        f"review_pass={_qg_pass}"
                                    )
                                    logger.warning("tiktok.quality_gate_BLOCKED",
                                                   reason=_qg_reason,
                                                   title=content_pack.get("title", "")[:80])
                                    publish_results.append({
                                        "platform": plat,
                                        "success": False,
                                        "error": f"Quality gate blocked: {_qg_reason}",
                                    })
                                    # Don't publish — mark run as draft so content
                                    # is saved but NOT sent to TikTok
                                    status = "draft"
                                    continue
                                else:
                                    logger.info("tiktok.quality_gate_PASSED",
                                                score=_qg_score,
                                                chars=_qg_chars,
                                                review_pass=_qg_pass)

                                publish_content = await _prepare_tiktok_content(
                                    content_pack, plat
                                )
                            except Exception as tk_err:
                                publish_results.append({
                                    "platform": plat,
                                    "success": False,
                                    "error": f"TikTok prep failed: {tk_err}",
                                })
                                continue

                        elif plat == "facebook":
                            # ── Facebook: rich text with full formatting ──
                            try:
                                publish_content = await _prepare_facebook_content(
                                    content_pack
                                )
                                if not publish_content:
                                    logger.warning("facebook.prepare_empty",
                                                   title=content_pack.get("title", "")[:60])
                                    continue
                                logger.info("facebook.content_ready",
                                            chars=len(publish_content.get("caption", "")))
                            except Exception as fb_err:
                                publish_results.append({
                                    "platform": plat,
                                    "success": False,
                                    "error": f"Facebook prep failed: {fb_err}",
                                })
                                continue

                        elif plat == "pinterest":
                            # ── Pinterest: pin with image + short description ──
                            try:
                                publish_content = await _prepare_pinterest_content(
                                    content_pack
                                )
                                if not publish_content:
                                    logger.warning("pinterest.prepare_empty",
                                                   title=content_pack.get("title", "")[:60],
                                                   reason="no image available for pin")
                                    publish_results.append({
                                        "platform": plat,
                                        "success": False,
                                        "error": "Pinterest requires an image — skipped",
                                    })
                                    continue
                                logger.info("pinterest.content_ready",
                                            title=publish_content.get("title", "")[:60])
                            except Exception as pin_err:
                                publish_results.append({
                                    "platform": plat,
                                    "success": False,
                                    "error": f"Pinterest prep failed: {pin_err}",
                                })
                                continue

                        else:
                            publish_content = content_pack

                        # Detect publisher signature — Publer uses publish(content: dict),
                        # direct publishers use publish(item: QueueItem, content: dict)
                        sig = inspect.signature(pub.publish)
                        param_count = len([
                            p for p in sig.parameters.values()
                            if p.default is inspect.Parameter.empty
                            and p.name != "self"
                        ])

                        if param_count >= 2:
                            # Direct publisher: needs (item, content)
                            item = _SimpleQueueItem(
                                id=f"autopilot-{plat}",
                                platform=plat,
                                platform_content=publish_content,
                            )
                            pub_result = await pub.publish(item, publish_content)
                            publish_results.append({
                                "platform": plat,
                                "success": getattr(pub_result, "success", False),
                                "post_url": getattr(pub_result, "post_url", ""),
                                "error": getattr(pub_result, "error", ""),
                            })
                        else:
                            # Publer-style: publish(content: dict)
                            pub_result = await pub.publish(publish_content)
                            if isinstance(pub_result, dict):
                                publish_results.append({"platform": plat, **pub_result})
                            else:
                                publish_results.append({
                                    "platform": plat,
                                    "success": getattr(pub_result, "success", False),
                                    "post_url": getattr(pub_result, "post_url", ""),
                                    "error": getattr(pub_result, "error", ""),
                                })

                        # ── TikTok: record post for daily-limit tracking ──
                        if plat == "tiktok":
                            tk_success = publish_results[-1].get("success", False)
                            tk_label = publish_content.get("_account_label", "")
                            await _record_tiktok_post(tk_label, tk_success)

                    except Exception as pub_err:
                        publish_results.append({
                            "platform": plat,
                            "success": False,
                            "error": str(pub_err),
                        })

                # ── Feed publish results into account health ──
                for pr in publish_results:
                    plat_name = pr.get("platform", "")
                    # TikTok: per-account health key (e.g. "tiktok:brand_main")
                    health_key = plat_name
                    if plat_name == "tiktok" and pr.get("_account_label"):
                        health_key = f"tiktok:{pr['_account_label']}"
                    health = _account_health.get_health(health_key)
                    cur_warnings = health.warning_count if health else 0
                    if pr.get("success"):
                        # Success: reset warnings (recovery)
                        _account_health.update(
                            health_key,
                            warning_count=max(0, cur_warnings - 1),
                        )
                    else:
                        # Failure: increment warnings
                        _account_health.update(
                            health_key,
                            warning_count=cur_warnings + 1,
                        )

                status = "published" if publish_results else "draft"

            _run_risk_score = risk_score
            _run_title = content_pack.get("title", "")

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
                    _run_post_id = new_post_id
                    _run_action = status  # "published" or "draft"
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
            else:
                _run_action = "skip"

            _autopilot_last_run = datetime.now(timezone.utc).isoformat()
            logger.info("autopilot.completed",
                        niche=niche,
                        title=content_pack.get("title", ""),
                        risk_score=risk_score,
                        status=status,
                        posts_today=_autopilot_posts_today)

            # ── Record autopilot run ──
            _run_duration = int((_time.monotonic() - _run_start) * 1000)
            try:
                with get_db_safe() as conn:
                    conn.execute(
                        """INSERT INTO autopilot_runs
                           (finished_at, niche, platforms, thread_id, risk_score,
                            action, post_id, title, error, duration_ms, posts_today)
                           VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            niche, json.dumps(platforms), thread_id,
                            _run_risk_score, _run_action, _run_post_id,
                            _run_title, _run_error, _run_duration,
                            _autopilot_posts_today,
                        ),
                    )
                    conn.commit()
            except Exception as rec_err:
                logger.warning("autopilot.record_run_failed", error=str(rec_err))

        except Exception as e:
            logger.error("autopilot.error", error=str(e))
            # Record failed run
            try:
                _run_duration = int((_time.monotonic() - _run_start) * 1000)
                with get_db_safe() as conn:
                    conn.execute(
                        """INSERT INTO autopilot_runs
                           (finished_at, niche, platforms, thread_id, risk_score,
                            action, post_id, title, error, duration_ms, posts_today)
                           VALUES (datetime('now'), ?, ?, ?, ?, 'error', NULL, ?, ?, ?, ?)""",
                        (
                            _run_niche,
                            json.dumps(_run_platforms),
                            _run_thread_id,
                            0, str(e), str(e),
                            _run_duration,
                            _autopilot_posts_today,
                        ),
                    )
                    conn.commit()
            except Exception:
                pass
            try:
                await _alert_manager.send(
                    f"Autopilot error: {e}",
                    level=AlertLevel.CRITICAL,
                )
            except Exception:
                pass

        # Sleep for configured interval (inside its own try/except to prevent loop crash)
        try:
            cfg = _get_autopilot_config()
            interval_sec = max(1800, cfg.get("interval_hours", 4) * 3600)  # min 30 min
            await asyncio.sleep(interval_sec)
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(14400)  # fallback: 4 hours


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
    """Return all available content creation methods, niches, and content packs (from DB)."""
    from publish_microniche import (
        MICRO_NICHES, NANO_NICHES, REAL_LIFE_NICHES,
    )

    methods = [
        {"id": "auto", "label": "Auto (Weighted)", "icon": "🎲", "desc": "35% AI + 25% Hunter + 20% AI-Niche + 15% Pre-written + 5% Gemini"},
        {"id": "ai_generate", "label": "AI Generate", "icon": "🤖", "desc": "LLM Cascade (GitHub Models → Perplexity) + self-review"},
        {"id": "hunter_prewritten", "label": "Hunter Pre-written", "icon": "🏆", "desc": "Top-scored packs from Niche Hunter DB"},
        {"id": "ai_niche", "label": "AI + Niche DB", "icon": "🔬", "desc": "Top niche scores from DB + LLM content generation"},
        {"id": "prewritten", "label": "Pre-written Packs", "icon": "✍️", "desc": "Content packs from database"},
        {"id": "gemini", "label": "Gemini Direct", "icon": "✨", "desc": "Gemini 2.5 Flash direct generation"},
        {"id": "niche_hunter", "label": "Niche Hunter", "icon": "🎯", "desc": "Scored questions from niche_hunter.db + Gemini"},
        {"id": "pain_point", "label": "Pain Point", "icon": "😤", "desc": "Urgent pain points with emotional triggers"},
    ]

    niches = {
        "micro": [{"id": n, "label": n.title()} for n in MICRO_NICHES],
        "nano": [{"id": n, "label": n.title()} for n in NANO_NICHES],
        "real_life": [{"id": n, "label": n.replace("_", " ").title()} for n in REAL_LIFE_NICHES],
    }

    # Niche scores from niche_hunter.db
    niche_scores = []
    try:
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "niche_hunter.db")
        if os.path.exists(db_path):
            with sqlite3.connect(db_path) as nh_conn:
                nh_conn.row_factory = sqlite3.Row
                rows = nh_conn.execute(
                    "SELECT topic, niche, final_score, risk_level, hook, source FROM niche_scores ORDER BY final_score DESC LIMIT 30"
                ).fetchall()
                niche_scores = [dict(r) for r in rows]
    except Exception:
        pass

    # Content packs from viralops.db
    packs = []
    with get_db_safe() as conn:
        rows = conn.execute(
            "SELECT id, title, hook, pain_point, niche_key, niche_score, source, enabled "
            "FROM content_packs ORDER BY niche_score DESC NULLS LAST, id ASC"
        ).fetchall()
        for r in rows:
            packs.append({
                "id": r["id"],
                "title": r["title"],
                "hook": r["hook"] or r["pain_point"] or "",
                "niche": r["niche_key"] or "",
                "score": r["niche_score"],
                "source": r["source"],
                "enabled": bool(r["enabled"]),
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
    Body: {method, niche?, topic?, custom_idea?, pack_title?, pack_id?}
    """
    data = await request.json()
    method = data.get("method", "auto")
    niche = data.get("niche")
    topic = data.get("topic")
    custom_idea = data.get("custom_idea", "").strip()
    pack_title = data.get("pack_title")
    pack_id = data.get("pack_id")

    try:
        from publish_microniche import get_content_pack

        # If a specific pack was selected by ID (DB), use it directly
        if pack_id:
            with get_db_safe() as conn:
                row = conn.execute("SELECT * FROM content_packs WHERE id = ?", (pack_id,)).fetchone()
                if row:
                    pack = _pack_row_to_publish_format(row)
                    pack["_source"] = "studio_pack_select"
                    return _studio_pack_response(pack)

        # Fallback: find by title in DB
        if pack_title:
            with get_db_safe() as conn:
                row = conn.execute("SELECT * FROM content_packs WHERE title = ?", (pack_title,)).fetchone()
                if row:
                    pack = _pack_row_to_publish_format(row)
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
# API — Content Packs CRUD
# ════════════════════════════════════════════════════════════════

@app.get("/api/studio/packs")
async def api_list_packs(source: Optional[str] = None, include_disabled: bool = False):
    """List all content packs. Optionally filter by source. Returns full pack data."""
    with get_db_safe() as conn:
        clauses, params = [], []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if not include_disabled:
            clauses.append("enabled = 1")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM content_packs{where} ORDER BY niche_score DESC NULLS LAST, id ASC",
            params,
        ).fetchall()
    return [_pack_row_to_dict(r) for r in rows]


@app.get("/api/studio/packs/{pack_id}")
async def api_get_pack(pack_id: int):
    """Get a single content pack by ID."""
    with get_db_safe() as conn:
        row = conn.execute("SELECT * FROM content_packs WHERE id = ?", (pack_id,)).fetchone()
    if not row:
        return JSONResponse(status_code=404, content={"error": "Pack not found"})
    return _pack_row_to_dict(row)


@app.post("/api/studio/packs")
async def api_create_pack(request: Request):
    """Create a new content pack."""
    data = await request.json()
    with get_db_safe() as conn:
        conn.execute(
            """INSERT INTO content_packs
               (title, hook, pain_point, key_fact, audiences, steps, result, cta,
                hashtags, image_title, image_subtitle, image_steps, colors,
                niche_key, niche_score, source, enabled)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                data.get("title", ""),
                data.get("hook", ""),
                data.get("pain_point", ""),
                data.get("key_fact", ""),
                json.dumps(data.get("audiences", [])),
                json.dumps(data.get("steps", [])),
                data.get("result", ""),
                data.get("cta", ""),
                json.dumps(data.get("hashtags", [])),
                data.get("image_title", ""),
                data.get("image_subtitle", ""),
                data.get("image_steps", ""),
                json.dumps(data.get("colors", "")),
                data.get("niche_key", ""),
                data.get("niche_score"),
                data.get("source", "manual"),
                1 if data.get("enabled", True) else 0,
            ),
        )
        conn.commit()
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return {"success": True, "id": new_id}


@app.put("/api/studio/packs/{pack_id}")
async def api_update_pack(pack_id: int, request: Request):
    """Update an existing content pack."""
    data = await request.json()
    with get_db_safe() as conn:
        existing = conn.execute("SELECT id FROM content_packs WHERE id = ?", (pack_id,)).fetchone()
        if not existing:
            return JSONResponse(status_code=404, content={"error": "Pack not found"})
        fields, values = [], []
        col_map = {
            "title": "title", "hook": "hook", "pain_point": "pain_point",
            "key_fact": "key_fact", "result": "result", "cta": "cta",
            "image_title": "image_title", "image_subtitle": "image_subtitle",
            "image_steps": "image_steps", "niche_key": "niche_key",
            "niche_score": "niche_score", "source": "source",
        }
        json_cols = {"audiences", "steps", "hashtags", "colors"}
        for key, col in col_map.items():
            if key in data:
                fields.append(f"{col} = ?")
                values.append(data[key])
        for jc in json_cols:
            if jc in data:
                fields.append(f"{jc} = ?")
                values.append(json.dumps(data[jc]))
        if "enabled" in data:
            fields.append("enabled = ?")
            values.append(1 if data["enabled"] else 0)
        if fields:
            fields.append("updated_at = datetime('now')")
            values.append(pack_id)
            conn.execute(
                f"UPDATE content_packs SET {', '.join(fields)} WHERE id = ?",
                values,
            )
            conn.commit()
    return {"success": True}


@app.delete("/api/studio/packs/{pack_id}")
async def api_delete_pack(pack_id: int):
    """Delete a content pack."""
    with get_db_safe() as conn:
        conn.execute("DELETE FROM content_packs WHERE id = ?", (pack_id,))
        conn.commit()
    return {"success": True}


@app.post("/api/studio/packs/{pack_id}/toggle")
async def api_toggle_pack(pack_id: int):
    """Toggle a pack's enabled state."""
    with get_db_safe() as conn:
        row = conn.execute("SELECT enabled FROM content_packs WHERE id = ?", (pack_id,)).fetchone()
        if not row:
            return JSONResponse(status_code=404, content={"error": "Pack not found"})
        new_state = 0 if row["enabled"] else 1
        conn.execute(
            "UPDATE content_packs SET enabled = ?, updated_at = datetime('now') WHERE id = ?",
            (new_state, pack_id),
        )
        conn.commit()
    return {"success": True, "enabled": bool(new_state)}


@app.post("/api/studio/packs/reseed")
async def api_reseed_packs():
    """Re-seed content packs from hardcoded data. Only adds packs whose titles don't already exist."""
    try:
        from publish_microniche import PRE_WRITTEN_PACKS, NICHE_HUNTER_PACKS
        added = 0
        with get_db_safe() as conn:
            existing_titles = {
                r[0] for r in conn.execute("SELECT title FROM content_packs").fetchall()
            }
            all_packs = [
                (p, "pre_written") for p in PRE_WRITTEN_PACKS
            ] + [
                (p, "niche_hunter") for p in NICHE_HUNTER_PACKS
            ]
            for pack, source in all_packs:
                if pack.get("title", "") in existing_titles:
                    continue
                conn.execute(
                    """INSERT INTO content_packs
                       (title, hook, pain_point, key_fact, audiences, steps, result, cta,
                        hashtags, image_title, image_subtitle, image_steps, colors,
                        niche_key, niche_score, source)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        pack.get("title", ""),
                        pack.get("hook", ""),
                        pack.get("pain_point", ""),
                        pack.get("key_fact", ""),
                        json.dumps(pack.get("audiences", [])),
                        json.dumps(pack.get("steps", pack.get("three_steps", []))),
                        pack.get("result", ""),
                        pack.get("cta", ""),
                        json.dumps(pack.get("hashtags", [])),
                        pack.get("image_title", ""),
                        pack.get("image_subtitle", ""),
                        pack.get("image_steps", ""),
                        json.dumps(pack.get("colors", "")),
                        pack.get("_niche_key", ""),
                        pack.get("_niche_score"),
                        source,
                    ),
                )
                added += 1
            conn.commit()
        return {"success": True, "added": added}
    except Exception as e:
        return {"error": str(e)}


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
    platforms = data.get("platforms", ["tiktok", "facebook", "pinterest"])
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
    # Merge incoming fields over current config — with bounds validation
    if "enabled" in data:
        current["enabled"] = bool(data["enabled"])
    if "interval_hours" in data:
        try:
            val = float(data["interval_hours"])
            current["interval_hours"] = max(0.5, min(val, 168))  # 30 min – 7 days
        except (ValueError, TypeError):
            pass
    if "niches" in data:
        niches = data["niches"]
        current["niches"] = niches if isinstance(niches, list) and niches else ["general"]
    if "platforms" in data:
        platforms = data["platforms"]
        if isinstance(platforms, list) and platforms:
            current["platforms"] = platforms
    if "max_posts_per_day" in data:
        try:
            val = int(data["max_posts_per_day"])
            current["max_posts_per_day"] = max(1, min(val, 500))
        except (ValueError, TypeError):
            pass
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


@app.get("/api/autopilot/runs")
async def api_autopilot_runs(limit: int = 50, offset: int = 0, action: str = ""):
    """Return paginated autopilot run history (newest first)."""
    with get_db_safe() as conn:
        where = ""
        params: list = []
        if action:
            where = "WHERE action = ?"
            params.append(action)
        total = conn.execute(
            f"SELECT COUNT(*) FROM autopilot_runs {where}", params
        ).fetchone()[0]
        rows = conn.execute(
            f"""SELECT id, started_at, finished_at, niche, platforms, thread_id,
                       risk_score, action, post_id, title, error, duration_ms, posts_today
                FROM autopilot_runs {where}
                ORDER BY id DESC LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()
        runs = []
        for r in rows:
            runs.append({
                "id": r["id"],
                "started_at": r["started_at"],
                "finished_at": r["finished_at"],
                "niche": r["niche"],
                "platforms": json.loads(r["platforms"] or "[]"),
                "thread_id": r["thread_id"],
                "risk_score": r["risk_score"],
                "action": r["action"],
                "post_id": r["post_id"],
                "title": r["title"],
                "error": r["error"],
                "duration_ms": r["duration_ms"],
                "posts_today": r["posts_today"],
            })
    return {"runs": runs, "total": total, "limit": limit, "offset": offset}


@app.get("/api/autopilot/stats")
async def api_autopilot_stats(days: int = 7):
    """Return aggregated autopilot statistics for the last N days."""
    with get_db_safe() as conn:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        row = conn.execute(
            """SELECT COUNT(*) as total_runs,
                      SUM(CASE WHEN action = 'published' THEN 1 ELSE 0 END) as published,
                      SUM(CASE WHEN action = 'draft' THEN 1 ELSE 0 END) as drafts,
                      SUM(CASE WHEN action = 'error' THEN 1 ELSE 0 END) as errors,
                      SUM(CASE WHEN action = 'skip' THEN 1 ELSE 0 END) as skipped,
                      SUM(CASE WHEN action = 'health_blocked' THEN 1 ELSE 0 END) as health_blocked,
                      AVG(duration_ms) as avg_duration_ms,
                      AVG(risk_score) as avg_risk_score,
                      MAX(duration_ms) as max_duration_ms
               FROM autopilot_runs WHERE started_at >= ?""",
            (cutoff,),
        ).fetchone()

        # Per-niche breakdown
        niche_rows = conn.execute(
            """SELECT niche, COUNT(*) as runs,
                      SUM(CASE WHEN action = 'published' THEN 1 ELSE 0 END) as published,
                      SUM(CASE WHEN action = 'error' THEN 1 ELSE 0 END) as errors,
                      AVG(risk_score) as avg_risk
               FROM autopilot_runs WHERE started_at >= ?
               GROUP BY niche ORDER BY runs DESC""",
            (cutoff,),
        ).fetchall()

        # Daily trend (last N days)
        daily_rows = conn.execute(
            """SELECT DATE(started_at) as day, COUNT(*) as runs,
                      SUM(CASE WHEN action = 'published' THEN 1 ELSE 0 END) as published,
                      SUM(CASE WHEN action = 'error' THEN 1 ELSE 0 END) as errors
               FROM autopilot_runs WHERE started_at >= ?
               GROUP BY DATE(started_at) ORDER BY day DESC""",
            (cutoff,),
        ).fetchall()

    return {
        "days": days,
        "summary": {
            "total_runs": row["total_runs"] or 0,
            "published": row["published"] or 0,
            "drafts": row["drafts"] or 0,
            "errors": row["errors"] or 0,
            "skipped": row["skipped"] or 0,
            "health_blocked": row["health_blocked"] or 0,
            "avg_duration_ms": int(row["avg_duration_ms"] or 0),
            "avg_risk_score": round(row["avg_risk_score"] or 0, 2),
            "max_duration_ms": row["max_duration_ms"] or 0,
            "success_rate": round(
                ((row["published"] or 0) + (row["drafts"] or 0))
                / max(row["total_runs"] or 1, 1) * 100, 1
            ),
        },
        "by_niche": [
            {
                "niche": nr["niche"],
                "runs": nr["runs"],
                "published": nr["published"],
                "errors": nr["errors"],
                "avg_risk": round(nr["avg_risk"] or 0, 2),
            }
            for nr in niche_rows
        ],
        "daily_trend": [
            {
                "day": dr["day"],
                "runs": dr["runs"],
                "published": dr["published"],
                "errors": dr["errors"],
            }
            for dr in daily_rows
        ],
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
# API — Account Health
# ════════════════════════════════════════════════════════════════


def _health_to_dict(h) -> dict:
    """Serialize an AccountHealth dataclass to a JSON-safe dict."""
    return {
        "platform": h.platform,
        "account_id": h.account_id,
        "followers": h.followers,
        "engagement_rate": h.engagement_rate,
        "warning_count": h.warning_count,
        "is_flagged": h.is_flagged,
        "is_shadow_banned": h.is_shadow_banned,
        "is_restricted": h.is_restricted,
        "restriction_type": h.restriction_type,
        "error_rate_24h": h.error_rate_24h,
        "last_checked": h.last_checked.isoformat() if h.last_checked else None,
        "safe": not h.is_flagged and not h.is_shadow_banned and h.warning_count < 3,
    }


@app.get("/api/account-health")
async def api_account_health_all():
    """Return health status for all tracked accounts."""
    all_health = _account_health.get_all()
    return {
        "accounts": {k: _health_to_dict(v) for k, v in all_health.items()},
        "total": len(all_health),
    }


@app.get("/api/account-health/{platform}")
async def api_account_health_platform(platform: str):
    """Return health for all accounts on a specific platform."""
    platform_health = _account_health.get_all_for_platform(platform)
    return {
        "platform": platform,
        "accounts": {k: _health_to_dict(v) for k, v in platform_health.items()},
    }


@app.post("/api/account-health/update")
async def api_account_health_update(request: Request):
    """Manually update account health.

    Body: {platform, account_id?, followers?, engagement_rate?,
           warning_count?, is_flagged?, is_shadow_banned?}
    """
    data = await request.json()
    platform = data.get("platform", "")
    if not platform:
        return JSONResponse(status_code=400, content={"error": "platform required"})
    health = _account_health.update(
        platform=platform,
        account_id=data.get("account_id"),
        followers=data.get("followers", 0),
        engagement_rate=data.get("engagement_rate", 0.0),
        warning_count=data.get("warning_count", 0),
        is_flagged=data.get("is_flagged", False),
        is_shadow_banned=data.get("is_shadow_banned", False),
    )
    return {"updated": _health_to_dict(health)}


@app.get("/api/account-health/safe/{platform}")
async def api_account_health_safe(platform: str, account_id: str = ""):
    """Check if it's safe to post on a platform/account."""
    safe, reason = _account_health.is_safe_to_post(
        platform, account_id or None
    )
    return {"safe": safe, "reason": reason, "platform": platform}


# ════════════════════════════════════════════════════════════════
# TikTok content preparation — adapts content_pack for publishing
# ════════════════════════════════════════════════════════════════

# ── TikTok caption format strategy (v4 — clump-proof) ──
# TikTok API strips ALL newlines (\n, \n\n) when posting via third-party
# schedulers like Publer.  No invisible-char workaround survives.
#
# STRATEGY: Keep 3500-4000 chars (full content!) but REFORMAT so the text
# is readable EVEN WHEN every newline is stripped into one big block.
#   • Emoji act as VISUAL SEPARATORS — the ONLY way to mark section breaks
#   • Each idea is a self-contained sentence preceded by its emoji
#   • No invisible chars, no Braille blanks, no tricks
#   • Reads like a continuous flow of emoji-delimited ideas
TIKTOK_MAX_CAPTION = 4000


async def _reformat_for_tiktok(title: str, long_content: str, hashtags: list[str]) -> str:
    """Reformat blog-style content into clump-proof TikTok caption (3500-4000 chars).

    TikTok API strips ALL line breaks when posting via scheduler tools.
    The caption MUST be readable as a single continuous block.

    Emoji act as the ONLY visual separators between ideas.
    """
    tag_str = " ".join(f"#{t.lstrip('#')}" for t in hashtags[:5] if t.strip())
    tag_budget = len(tag_str) + 2
    char_budget = TIKTOK_MAX_CAPTION - tag_budget - 50  # aim 3500-3950

    reformat_prompt = f"""Reformat this blog content into a TikTok-friendly caption.

TITLE: {title}
ORIGINAL CONTENT:
{long_content[:4200]}

CRITICAL CONTEXT: TikTok strips ALL line breaks when posted via API.
Every newline becomes a space. The entire caption will render as ONE continuous
block of text. You MUST design it to be READABLE in that format.

RULES — FOLLOW EXACTLY:
1. OUTPUT ONLY the reformatted caption text. No JSON, no code blocks, no wrapping quotes.
2. ⚠️ HARD LENGTH REQUIREMENT: Your output MUST be {char_budget} characters (3500-3900 range).
   This is NON-NEGOTIABLE. Do NOT summarize or shorten the content.
   You are REFORMATTING, not SUMMARIZING. Keep EVERY detail, EVERY tip, EVERY example.
   If the original is 3800 chars, your output should also be ~3800 chars.
   Count your characters — if below 3400, you MUST add more detail from the original.
3. Start with the provided TITLE as the first line (keep it recognizable), then "—" dash, then 2-3 sentence direct answer.
4. Use emoji (🌿 🫙 ❌ ✅ 🪴 💡 🔥 👉) as VISUAL SECTION MARKERS.
   Each new idea MUST start with an emoji so readers can see where sections begin
   even when everything is one giant block.
5. Within each section, use "•" bullet with text. Example:
   "🌿 Basic Setup • Get a 5-gallon bucket for $5 from any hardware store. • Add shredded cardboard as carbon — tear into 1-2 inch pieces. • Collect fruit peels, coffee grounds, veggie trimmings (no meat, no dairy)."
6. For step-by-step, use numbered emoji: "🫙 Quick Method — 1️⃣ Gather scraps daily. 2️⃣ Add a handful of shredded paper. 3️⃣ Mix every few days. 4️⃣ Keep lid on, check weekly. 5️⃣ Harvest compost in 8-12 weeks."
7. For warnings: "❌ Common Mistakes — • Overloading food scraps makes it stink. • Forgetting to aerate kills the process."
8. For tips: "✅ Pro Tips — • Start small with just a bucket. • Keep it damp but not soggy."
9. PLAIN TEXT ONLY — no ** bold, no ### headings, no Markdown formatting.
10. Keep ALL specific numbers from original ($5, 8-12 weeks, 130°F, 1:1 ratio, etc).
11. Witty personality — dry humor, real talk, zero corporate tone.
12. End with a punchy 1-2 sentence closer.
13. Do NOT use invisible characters, Braille blanks, or zero-width spaces.
14. Do NOT rely on line breaks for formatting — they will ALL be stripped.
15. FINAL CHECK: Re-read your output. Is it 3500-3900 characters? If NOT, expand it NOW."""

    MIN_REFORMAT_LEN = 3200  # minimum acceptable reformat length

    def _clean_llm_caption(raw: str) -> str:
        """Strip wrapping artifacts from LLM output."""
        import re as _re
        c = raw.strip()
        if c.startswith('"') and c.endswith('"'):
            c = c[1:-1]
        if c.startswith("```"):
            c = c.split("```", 2)[-1] if c.count("```") >= 2 else c[3:]
            c = c.rsplit("```", 1)[0] if "```" in c else c
            c = c.strip()
        c = _re.sub(r'\*\*(.+?)\*\*', r'\1', c)
        c = _re.sub(r'###?\s*', '', c)
        return c.strip()

    try:
        from llm_content import call_llm

        def _norm(s: str) -> str:
            import re as _re
            s = (s or "").lower().strip()
            s = _re.sub(r"\s+", " ", s)
            return s

        # ── Attempt 1: normal reformat ──
        result = call_llm(
            reformat_prompt,
            system="You reformat blog content into TikTok captions that read well as one continuous block. Emoji are the ONLY visual separators. Output ONLY the caption text. OUTPUT MUST BE 3500-3900 CHARACTERS — same length as the original.",
            max_tokens=5000,
            temperature=0.5,
        )
        caption = ""
        if result.success and result.text and len(result.text.strip()) > 500:
            caption = _clean_llm_caption(result.text)

        # ── Attempt 2: if too short, retry with stricter prompt ──
        if len(caption) < MIN_REFORMAT_LEN:
            logger.warning("tiktok.reformat_too_short",
                           len=len(caption), min=MIN_REFORMAT_LEN)
            expand_prompt = f"""Your previous output was only {len(caption)} characters. That is TOO SHORT.

I need EXACTLY 3500-3900 characters. The original content was {len(long_content)} characters.

Here is the original content again:
{long_content[:4200]}

EXPAND your previous output to 3500-3900 characters by:
- Adding back ALL details you removed
- Including ALL tips, examples, numbers, and real-talk commentary
- Each section should have 3-5 bullet points with "•"
- Add extra context, comparisons, or "why this matters" explanations

Previous short output to expand:
{caption[:2000]}

OUTPUT ONLY the full 3500-3900 character caption. No JSON, no code blocks."""

            result2 = call_llm(
                expand_prompt,
                system="You MUST output 3500-3900 characters. This is a HARD requirement. Expand the content — do NOT summarize.",
                max_tokens=5000,
                temperature=0.6,
            )
            if result2.success and result2.text and len(result2.text.strip()) > len(caption):
                caption2 = _clean_llm_caption(result2.text)
                if len(caption2) > len(caption):
                    caption = caption2
                    logger.info("tiktok.reformat_expanded",
                                len=len(caption), attempt=2)

        # ── Attempt 3: if STILL too short, pad with original content ──
        if caption and len(caption) < MIN_REFORMAT_LEN:
            import re as _re
            # Flatten original content and append missing parts
            flat_original = _re.sub(r'\s*\n\s*', ' ', long_content)
            flat_original = _re.sub(r' {2,}', ' ', flat_original).strip()
            # Find content not yet in caption (simple: append tail of original)
            needed = MIN_REFORMAT_LEN - len(caption) + 200
            # Take the last N chars of original that aren't the title/hook
            tail = flat_original[-(needed + 200):]
            caption = f"{caption} ✅ More Details — {tail}"
            caption = caption[:TIKTOK_MAX_CAPTION - len(tag_str) - 10]
            logger.info("tiktok.reformat_padded",
                        len=len(caption), padding=needed)

        if caption:
            # Append hashtags if not already present
            if tag_str and tag_str not in caption:
                caption = f"{caption} {tag_str}"

            # Ensure title is visible in TikTok caption.
            # TikTok does not have a separate 'title' field — only caption text.
            # Some LLM reformats may drop the provided TITLE; prefix it back in.
            if title:
                norm_title = _norm(title)
                head = _norm(caption[:220])
                if norm_title and norm_title not in head:
                    # Leave room for hashtags already appended
                    max_len = TIKTOK_MAX_CAPTION
                    prefix = f"{title} — "
                    caption = (prefix + caption).strip()
                    if len(caption) > max_len:
                        caption = caption[:max_len].rsplit(" ", 1)[0]

            logger.info("tiktok.reformat_success",
                        original_len=len(long_content),
                        new_len=len(caption),
                        provider=getattr(result, 'provider', 'unknown'))
            return caption
    except Exception as e:
        logger.warning("tiktok.reformat_llm_error", error=str(e)[:200])

    # ── FALLBACK: programmatic clump-proof reformat (no LLM available) ──
    return _fallback_tiktok_clumpproof(title, long_content, hashtags)


def _fallback_tiktok_clumpproof(title: str, content: str, hashtags: list[str]) -> str:
    """Programmatic fallback: make existing content clump-proof without LLM.

    Strategy: collapse all newlines → spaces, clean up spacing,
    ensure emoji section headers are followed by "—" dash separators.
    """
    import re as _re
    tag_str = " ".join(f"#{t.lstrip('#')}" for t in hashtags[:5] if t.strip())

    # Start with title
    caption = f"{title} — " if title else ""

    # Collapse all newlines → single spaces
    flat = _re.sub(r'\s*\n\s*', ' ', content)
    # Collapse multiple spaces
    flat = _re.sub(r' {2,}', ' ', flat).strip()

    # Ensure emoji section headers have " — " before them for visual separation
    # Match common section-header emoji
    flat = _re.sub(
        r'\s*([\U0001F33F\U0001FAD8\U0001F372\u274C\u2705\U0001FAB4\U0001F4A1\U0001F525\U0001F449'
        r'\U0001F331\U0001F33B\U0001F33C\U0001F33D\U0001F33E])',
        r' \1', flat
    )

    caption += flat

    # Trim to budget, leaving room for hashtags
    max_content = TIKTOK_MAX_CAPTION - len(tag_str) - 10
    if len(caption) > max_content:
        caption = caption[:max_content].rsplit(" ", 1)[0]

    if tag_str:
        caption = f"{caption} {tag_str}"

    return caption[:TIKTOK_MAX_CAPTION]


def _strip_duplicate_title_and_hashtags(*, title: str, text: str) -> str:
    """Remove duplicate title line at the start and hashtag-only lines at the end.

    This prevents patterns like:
      TITLE\n\nTITLE\n\n...
    and avoids double-hashtag tails when universal_caption_block already includes them.
    """
    import re as _re

    t = str(title or "").strip()
    s = str(text or "")
    if not s.strip() or not t:
        return s.strip()

    # Strip trailing hashtag-only blocks (one or more lines mostly made of #tags).
    lines = s.splitlines()
    while lines:
        tail = lines[-1].strip()
        if not tail:
            lines.pop()
            continue
        # A line that contains at least 2 hashtags and little else.
        if len(_re.findall(r"#[A-Za-z0-9_]+", tail)) >= 2 and _re.sub(r"#[A-Za-z0-9_]+", "", tail).strip() == "":
            lines.pop()
            # Also remove any blank line before it
            while lines and not lines[-1].strip():
                lines.pop()
            continue
        break
    s = "\n".join(lines)

    # Strip leading duplicate title line(s).
    def _norm(x: str) -> str:
        x = (x or "").strip().lower()
        x = x.lstrip("🌿✅❌🔥👉🫙🫘🌱🍃 ")
        x = x.replace("—", "-").replace("–", "-")
        x = _re.sub(r"\s+", " ", x)
        return x

    t_norm = _norm(t)
    lines = s.splitlines()
    # drop leading blanks
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines:
        first = _norm(lines[0])
        if first == t_norm or first.startswith(t_norm + " -") or first.startswith(t_norm + " —"):
            lines.pop(0)
            # drop one blank line after title
            while lines and not lines[0].strip():
                lines.pop(0)
    return "\n".join(lines).strip()


async def _prepare_tiktok_content(content_pack: dict, platform: str = "tiktok") -> dict:
    """
    Adapt a raw content_pack into a Publer/TikTok-ready publish dict.

    Handles:
      1. Reformat content into clump-proof TikTok caption (3500-4000 chars)
         — TikTok API strips ALL newlines, so emoji are the only separators
         — Uses LLM to reformat, with programmatic fallback
      2. Set platforms filter so Publer only targets TikTok
      3. Pick next TikTok account via TikTokAccountManager (round-robin)
      4. Generate image from image_prompt if no media_url exists
      5. Return Publer-compatible content dict

    TikTok caption limit: 4000 chars.
    """
    import tempfile
    import uuid as _uuid

    # ── 1. Build caption — reformat for TikTok (clump-proof) ──
    title = content_pack.get("title", "")
    content_formatted = content_pack.get("content_formatted", "")
    universal_caption = content_pack.get("universal_caption_block", "")
    hashtags = content_pack.get("hashtags", [])

    # Ensure all hashtags have # prefix
    hashtags = [f"#{t.lstrip('#')}" for t in hashtags if t.strip()]

    # Get the best available long content
    long_content = ""
    if universal_caption and len(universal_caption) > 500:
        long_content = universal_caption
    elif content_formatted and len(content_formatted) > 500:
        long_content = content_formatted
    else:
        body = content_pack.get("body", "")
        hook = content_pack.get("hook", "")
        cta = content_pack.get("cta", "")
        parts = []
        if hook:
            parts.append(hook)
        if body:
            parts.append(body)
        elif title:
            parts.append(title)
        if cta:
            parts.append(cta)
        long_content = " ".join(parts)

    # Remove duplicate title/hashtags that may already be present in long_content.
    long_content = _strip_duplicate_title_and_hashtags(title=title, text=long_content)

    # ── Reformat: make content clump-proof for TikTok ──
    # Always reformat — even "short" content needs emoji separators
    if long_content and len(long_content) > 200:
        caption = await _reformat_for_tiktok(title, long_content, hashtags)
    else:
        caption = long_content.strip()
        tag_str = " ".join(hashtags[:5])
        if tag_str and tag_str not in caption:
            caption = f"{caption} {tag_str}"

    # ── Strip ALL invisible Unicode artifacts ──
    import re as _re
    caption = _re.sub(r'[\u2800\u200B\u200C\u200D\uFEFF]', '', caption)

    # ── Enforce TikTok 4000 char limit ──
    if len(caption) > TIKTOK_MAX_CAPTION:
        tag_str = " ".join(hashtags[:5])
        tag_space = len(tag_str) + 2
        max_content = TIKTOK_MAX_CAPTION - tag_space - 10
        caption = caption[:max_content].rsplit(" ", 1)[0] + f" {tag_str}"

    # ── 2. Pick TikTok account ─────────────────────────────────────
    # If caller provided explicit Publer account IDs, respect them.
    # This is REQUIRED for multi-account publishing where each account
    # must receive a distinct caption variant.
    account_ids = []
    account_label = None
    explicit_ids = content_pack.get("account_ids", [])
    if isinstance(explicit_ids, list) and any(str(x).strip() for x in explicit_ids):
        account_ids = [str(x).strip() for x in explicit_ids if str(x).strip()]
        account_label = str(content_pack.get("_account_label", "") or "")
        logger.info(
            "tiktok.account_override",
            label=account_label,
            account_ids=account_ids,
        )
    else:
        # Default: round-robin via TikTokAccountManager
        try:
            from core.tiktok_accounts import get_account_manager

            mgr = get_account_manager()
            acct = mgr.next_account()
            if acct:
                # TikTokAccount is a dataclass — use attribute access, not dict .get()
                account_ids = [getattr(acct, "id", "")]
                account_label = getattr(acct, "label", "")
                logger.info("tiktok.account_picked", label=account_label, account_id=account_ids[0])
        except Exception as e:
            logger.warning("tiktok.account_manager_error", error=str(e))

    # ── 3. Generate image if needed ──
    media_url = (content_pack.get("media_url", "")
                 or content_pack.get("image_url", "")
                 or content_pack.get("_ai_image_path", ""))
    # Validate local file still exists
    if media_url and not media_url.startswith(("http://", "https://")) and not os.path.isfile(media_url):
        logger.warning("tiktok.stale_image_path", path=media_url)
        media_url = ""
    if not media_url:
        image_prompt = content_pack.get("image_prompt", "")
        if image_prompt:
            # PRIMARY: AI image generation (Pollinations → Gemini/Imagen cascade)
            try:
                output_dir = tempfile.mkdtemp(prefix="viralops_tiktok_")
                output_path = os.path.join(output_dir, f"tiktok_{_uuid.uuid4().hex[:8]}.png")
                from llm_content import generate_ai_image
                img_path = await asyncio.to_thread(
                    generate_ai_image, image_prompt, output_path
                )
                if img_path and os.path.isfile(img_path):
                    media_url = img_path  # Local file path — Publer will upload
                    logger.info("tiktok.image_generated", path=img_path)
                else:
                    logger.warning("tiktok.image_generation_failed",
                                   prompt=image_prompt[:80])
            except Exception as e:
                logger.warning("tiktok.image_gen_error", error=str(e))

            # FALLBACK: Pexels stock photos (free API, high-quality)
            if not media_url:
                pexels_key = os.environ.get("PEXELS_API_KEY", "")
                if pexels_key:
                    try:
                        # Extract search keywords from image prompt
                        search_query = image_prompt[:80].replace('"', "").strip()
                        async with httpx.AsyncClient(timeout=15.0) as pex_client:
                            pex_resp = await pex_client.get(
                                "https://api.pexels.com/v1/search",
                                params={
                                    "query": search_query,
                                    "orientation": "portrait",
                                    "per_page": 3,
                                },
                                headers={"Authorization": pexels_key},
                            )
                            if pex_resp.status_code == 200:
                                photos = pex_resp.json().get("photos", [])
                                if photos:
                                    import random as _random
                                    photo = _random.choice(photos)
                                    photo_url = photo.get("src", {}).get(
                                        "large2x",
                                        photo.get("src", {}).get("large", ""),
                                    )
                                    if photo_url:
                                        # Download to local file for reliable upload
                                        dl_dir = tempfile.mkdtemp(prefix="viralops_pexels_")
                                        dl_path = os.path.join(
                                            dl_dir, f"pexels_{_uuid.uuid4().hex[:8]}.jpg"
                                        )
                                        dl_resp = await pex_client.get(photo_url)
                                        if dl_resp.status_code == 200:
                                            with open(dl_path, "wb") as df:
                                                df.write(dl_resp.content)
                                            media_url = dl_path
                                            logger.info(
                                                "tiktok.pexels_fallback_image",
                                                path=dl_path,
                                                photo_id=photo.get("id"),
                                            )
                    except Exception as pex_err:
                        logger.warning("tiktok.pexels_fallback_error", error=str(pex_err))

    # ── 4. Build Publer-compatible content dict ──
    # NOTE: Do NOT pass "hashtags" here — they are already embedded in caption.
    # PublerPublisher.publish() appends hashtags to caption, so passing them
    # would cause DUPLICATE hashtags (the exact same tags appearing twice).
    result = {
        "caption": caption,
        "title": title,
        "platforms": [platform],
        "hashtags": [],  # empty — already in caption (prevents publisher double-adding)
        "content_type": "photo" if media_url else "status",
    }
    if account_ids:
        result["account_ids"] = account_ids
    if media_url:
        if media_url.startswith(("http://", "https://")):
            result["media_url"] = media_url
        else:
            # Local file — convert to data for upload
            result["media_local_path"] = media_url
    # Carry over extra metadata
    result["_account_label"] = account_label or ""
    result["_original_content_pack"] = True

    return result


async def _prepare_facebook_content(content_pack: dict) -> dict:
    """
    Adapt a raw content_pack into a Publer/Facebook-ready publish dict.

    Facebook supports rich text with newlines, links, emojis, etc.
    No need for clump-proof reformatting — use the full formatted content
    as-is with proper paragraph breaks.

    Facebook post limits:
      - Status/text: 63,206 chars
      - Photo caption: 63,206 chars
      - No daily post limit enforced by ViralOps
    """
    import tempfile
    import uuid as _uuid

    title = content_pack.get("title", "")
    content_formatted = content_pack.get("content_formatted", "")
    universal_caption = content_pack.get("universal_caption_block", "")
    hashtags = content_pack.get("hashtags", [])
    hashtags = [f"#{t.lstrip('#')}" for t in hashtags if t.strip()]

    # Use the best available content — FB supports full rich text
    long_content = ""
    if universal_caption and len(universal_caption) > 500:
        long_content = universal_caption
    elif content_formatted and len(content_formatted) > 500:
        long_content = content_formatted
    else:
        body = content_pack.get("body", "")
        hook = content_pack.get("hook", "")
        cta = content_pack.get("cta", "")
        parts = []
        if hook:
            parts.append(hook)
        if body:
            parts.append(body)
        elif title:
            parts.append(title)
        if cta:
            parts.append(cta)
        long_content = "\n\n".join(parts)

    # Avoid repeating the title and hashtag block when universal_caption already contains them.
    long_content = _strip_duplicate_title_and_hashtags(title=title, text=long_content)

    # Build caption: title + content + hashtags
    caption_parts = []
    if title:
        caption_parts.append(f"🌿 {title}")
    if long_content:
        caption_parts.append(long_content)
    tag_str = " ".join(hashtags[:10])
    if tag_str:
        caption_parts.append(tag_str)
    caption = "\n\n".join(caption_parts)

    # ── Generate image if needed ──
    media_url = (content_pack.get("media_url", "")
                 or content_pack.get("image_url", "")
                 or content_pack.get("_ai_image_path", ""))
    if media_url and not media_url.startswith(("http://", "https://")) and not os.path.isfile(media_url):
        media_url = ""
    if not media_url:
        image_prompt = content_pack.get("image_prompt", "")
        if image_prompt:
            try:
                output_dir = tempfile.mkdtemp(prefix="viralops_fb_")
                output_path = os.path.join(output_dir, f"fb_{_uuid.uuid4().hex[:8]}.png")
                from llm_content import generate_ai_image
                img_path = await asyncio.to_thread(
                    generate_ai_image, image_prompt, output_path
                )
                if img_path and os.path.isfile(img_path):
                    media_url = img_path
                    logger.info("facebook.image_generated", path=img_path)
            except Exception as e:
                logger.warning("facebook.image_gen_error", error=str(e))

    result = {
        "caption": caption,
        "title": title,
        "platforms": ["facebook"],
        "hashtags": [],  # already in caption
        "content_type": "photo" if media_url else "status",
    }
    if media_url:
        if media_url.startswith(("http://", "https://")):
            result["media_url"] = media_url
        else:
            result["media_local_path"] = media_url

    logger.info("facebook.content_prepared",
                chars=len(caption), has_image=bool(media_url))
    return result


async def _prepare_pinterest_content(content_pack: dict) -> dict:
    """
    Adapt a raw content_pack into a Publer/Pinterest-ready publish dict.

    Pinterest Pin format:
      - Title: max 100 chars
      - Description: max 500 chars
      - Image REQUIRED for pins
      - Link optional but recommended
      - No daily post limit enforced by ViralOps
    """
    import tempfile
    import uuid as _uuid

    title = content_pack.get("title", "")
    content_formatted = content_pack.get("content_formatted", "")
    universal_caption = content_pack.get("universal_caption_block", "")
    hashtags = content_pack.get("hashtags", [])
    hashtags = [f"#{t.lstrip('#')}" for t in hashtags if t.strip()]

    # Pinterest title: max 100 chars
    pin_title = title[:97] + "..." if len(title) > 100 else title

    # Pinterest description: max 500 chars — use compact version
    long_content = ""
    if universal_caption and len(universal_caption) > 200:
        long_content = universal_caption
    elif content_formatted and len(content_formatted) > 200:
        long_content = content_formatted
    else:
        body = content_pack.get("body", "")
        hook = content_pack.get("hook", "")
        long_content = hook or body or title

    # Prevent duplicate title at the top and duplicate hashtag tails.
    long_content = _strip_duplicate_title_and_hashtags(title=title, text=long_content)

    # Build description: truncate to 500 chars with hashtags
    tag_str = " ".join(hashtags[:5])
    max_desc = 500 - len(tag_str) - 2 if tag_str else 500
    desc = long_content[:max_desc].rsplit(" ", 1)[0] if len(long_content) > max_desc else long_content
    if tag_str:
        desc = f"{desc} {tag_str}"

    # ── Image is REQUIRED for Pinterest pins ──
    media_url = (content_pack.get("media_url", "")
                 or content_pack.get("image_url", "")
                 or content_pack.get("_ai_image_path", ""))
    if media_url and not media_url.startswith(("http://", "https://")) and not os.path.isfile(media_url):
        media_url = ""
    if not media_url:
        image_prompt = content_pack.get("image_prompt", "")
        if image_prompt:
            try:
                output_dir = tempfile.mkdtemp(prefix="viralops_pin_")
                output_path = os.path.join(output_dir, f"pin_{_uuid.uuid4().hex[:8]}.png")
                from llm_content import generate_ai_image
                img_path = await asyncio.to_thread(
                    generate_ai_image, image_prompt, output_path
                )
                if img_path and os.path.isfile(img_path):
                    media_url = img_path
                    logger.info("pinterest.image_generated", path=img_path)
            except Exception as e:
                logger.warning("pinterest.image_gen_error", error=str(e))

    if not media_url:
        logger.warning("pinterest.no_image_skipping",
                        title=title[:60],
                        reason="Pinterest requires an image for pins")
        return {}  # Empty dict signals skip

    result = {
        "caption": desc,
        "title": pin_title,
        "platforms": ["pinterest"],
        "hashtags": [],  # already in description
        "content_type": "pin",
    }
    if media_url.startswith(("http://", "https://")):
        result["media_url"] = media_url
    else:
        result["media_local_path"] = media_url

    logger.info("pinterest.content_prepared",
                title_len=len(pin_title), desc_len=len(desc),
                has_image=True)
    return result


async def _record_tiktok_post(account_label: str, success: bool):
    """Record a TikTok post in the account manager for daily limit tracking."""
    try:
        from core.tiktok_accounts import get_account_manager
        mgr = get_account_manager()
        if success:
            mgr.record_post(account_label)
        else:
            mgr.record_draft(account_label)
    except Exception as e:
        logger.warning("tiktok.record_post_error", error=str(e))


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
    elif platform == "tiktok":
        # Fallback: Direct TikTok API (MultiTikTokPublisher → TikTokPublisher)
        try:
            from integrations.multi_tiktok_publisher import MultiTikTokPublisher
            mtp = MultiTikTokPublisher()
            if mtp.accounts:
                return mtp
        except Exception:
            pass
        try:
            from integrations.social_connectors import get_social_publisher
            return get_social_publisher("tiktok")
        except Exception:
            pass
        raise ValueError("TikTok: no publisher configured (need Publer or TIKTOK_ACCESS_TOKEN)")
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
    import inspect

    for platform in platforms:
        try:
            pub = _get_publisher(platform)
            content_dict = {
                "title": post["title"],
                "body": post["body"],
                "caption": post["body"][:2000],
                **extra.get(platform, {}),
                **{k: v for k, v in extra.items() if not isinstance(v, dict)},
            }

            # Detect publisher signature — Publer uses publish(content: dict),
            # direct publishers use publish(item: QueueItem, content: dict)
            sig = inspect.signature(pub.publish)
            param_count = len([
                p for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.name != "self"
            ])

            if param_count >= 2:
                # Direct publisher: needs (item, content)
                item = _SimpleQueueItem(
                    id=f"web-{post_id}-{platform}",
                    platform=platform,
                    platform_content=content_dict,
                )
                result = await pub.publish(item, content_dict)
            else:
                # Publer-style: publish(content: dict)
                result = await pub.publish(content_dict)

            if isinstance(result, dict):
                success = result.get("success", False)
                post_url = result.get("post_url", "")
                error = result.get("error", "")
            else:
                success = getattr(result, "success", False)
                post_url = getattr(result, "post_url", "")
                error = getattr(result, "error", "")
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
        v in ("running", "ok", "not_started") for v in checks.values()
    )
    status_code = 200 if all_healthy else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if all_healthy else "degraded",
            "checks": checks,
            "version": "2.17.0",
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
