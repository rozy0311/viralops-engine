"""
Analytics & Feedback Loop — ViralOps Engine
Track post performance, hashtag conversion rates, best posting times.

Data sources:
  - publish_log table (SQLite)
  - External APIs (future: platform insights APIs)

Provides:
  - Per-platform success rate
  - Per-hashtag engagement correlation
  - Best time-of-day per platform
  - Weekly/monthly trend reports
  - Feedback to content_factory for learning
"""
import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional
from collections import Counter, defaultdict

import structlog

logger = structlog.get_logger()

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "viralops.db")


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_analytics_tables():
    """Create analytics tables if they don't exist."""
    conn = _get_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS post_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER,
                platform TEXT,
                post_url TEXT,
                likes INTEGER DEFAULT 0,
                comments INTEGER DEFAULT 0,
                shares INTEGER DEFAULT 0,
                views INTEGER DEFAULT 0,
                clicks INTEGER DEFAULT 0,
                saves INTEGER DEFAULT 0,
                engagement_rate REAL DEFAULT 0.0,
                hashtags TEXT DEFAULT '[]',
                posted_at TEXT,
                fetched_at TEXT,
                UNIQUE(post_id, platform)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hashtag_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hashtag TEXT,
                platform TEXT,
                total_uses INTEGER DEFAULT 0,
                total_engagement INTEGER DEFAULT 0,
                avg_engagement REAL DEFAULT 0.0,
                last_used TEXT,
                UNIQUE(hashtag, platform)
            )
        """)
        conn.commit()
    except Exception as e:
        logger.error("analytics.table_error", error=str(e))
    finally:
        conn.close()


# ── Initialize on import ──
_ensure_analytics_tables()


# ════════════════════════════════════════════════
# Publish Performance
# ════════════════════════════════════════════════

def get_publish_stats(days: int = 30) -> dict:
    """Get publishing statistics for the last N days."""
    conn = _get_db()
    try:
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")

        # Total posts
        total = conn.execute(
            "SELECT COUNT(*) as c FROM publish_log WHERE published_at >= ?", (since,)
        ).fetchone()["c"]

        # Success rate
        success = conn.execute(
            "SELECT COUNT(*) as c FROM publish_log WHERE published_at >= ? AND success = 1", (since,)
        ).fetchone()["c"]

        # Per platform
        platform_stats = {}
        rows = conn.execute(
            "SELECT platform, COUNT(*) as total, SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as ok "
            "FROM publish_log WHERE published_at >= ? GROUP BY platform", (since,)
        ).fetchall()
        for row in rows:
            platform_stats[row["platform"]] = {
                "total": row["total"],
                "success": row["ok"],
                "rate": round(row["ok"] / row["total"] * 100, 1) if row["total"] > 0 else 0,
            }

        # Per hour distribution
        hour_dist = Counter()
        time_rows = conn.execute(
            "SELECT published_at FROM publish_log WHERE published_at >= ? AND success = 1", (since,)
        ).fetchall()
        for row in time_rows:
            try:
                dt = datetime.fromisoformat(row["published_at"])
                hour_dist[dt.hour] += 1
            except (ValueError, TypeError):
                pass

        # Errors summary
        errors = conn.execute(
            "SELECT platform, error, COUNT(*) as c FROM publish_log "
            "WHERE published_at >= ? AND success = 0 AND error != '' "
            "GROUP BY platform, error ORDER BY c DESC LIMIT 10", (since,)
        ).fetchall()

        return {
            "period_days": days,
            "total_attempts": total,
            "successful": success,
            "failed": total - success,
            "success_rate": round(success / total * 100, 1) if total > 0 else 0,
            "per_platform": platform_stats,
            "best_hours": dict(hour_dist.most_common(5)),
            "top_errors": [{"platform": e["platform"], "error": e["error"], "count": e["c"]} for e in errors],
        }
    finally:
        conn.close()


def get_best_posting_times(platform: str = None, days: int = 30) -> dict:
    """Analyze which posting times produce best results."""
    conn = _get_db()
    try:
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")

        query = "SELECT published_at, platform FROM publish_log WHERE published_at >= ? AND success = 1"
        params = [since]
        if platform:
            query += " AND platform = ?"
            params.append(platform)

        rows = conn.execute(query, params).fetchall()

        # Analyze by hour and day of week
        hour_counts = defaultdict(int)
        day_counts = defaultdict(int)
        platform_hours = defaultdict(lambda: defaultdict(int))

        for row in rows:
            try:
                dt = datetime.fromisoformat(row["published_at"])
                hour_counts[dt.hour] += 1
                day_counts[dt.strftime("%A")] += 1
                platform_hours[row["platform"]][dt.hour] += 1
            except (ValueError, TypeError):
                pass

        # Find optimal slots
        best_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        best_days = sorted(day_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        per_platform_best = {}
        for plat, hours in platform_hours.items():
            sorted_hours = sorted(hours.items(), key=lambda x: x[1], reverse=True)[:3]
            per_platform_best[plat] = [{"hour": h, "count": c} for h, c in sorted_hours]

        return {
            "best_hours_overall": [{"hour": h, "posts": c} for h, c in best_hours],
            "best_days": [{"day": d, "posts": c} for d, c in best_days],
            "per_platform": per_platform_best,
            "recommendation": f"Post at {best_hours[0][0]}:00 UTC on {best_days[0][0]}" if best_hours and best_days else "Not enough data",
        }
    finally:
        conn.close()


# ════════════════════════════════════════════════
# Hashtag Performance
# ════════════════════════════════════════════════

def track_hashtag_usage(hashtags: list[str], platform: str):
    """Record hashtag usage for performance tracking."""
    conn = _get_db()
    try:
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        for tag in hashtags:
            conn.execute("""
                INSERT INTO hashtag_performance (hashtag, platform, total_uses, last_used)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(hashtag, platform) DO UPDATE SET
                    total_uses = total_uses + 1,
                    last_used = ?
            """, (tag.lower(), platform, now, now))
        conn.commit()
    except Exception as e:
        logger.error("analytics.hashtag_track_error", error=str(e))
    finally:
        conn.close()


def update_hashtag_engagement(hashtag: str, platform: str, engagement: int):
    """Update engagement data for a hashtag."""
    conn = _get_db()
    try:
        conn.execute("""
            UPDATE hashtag_performance SET
                total_engagement = total_engagement + ?,
                avg_engagement = CAST((total_engagement + ?) AS REAL) / MAX(total_uses, 1)
            WHERE hashtag = ? AND platform = ?
        """, (engagement, engagement, hashtag.lower(), platform))
        conn.commit()
    except Exception as e:
        logger.error("analytics.hashtag_engagement_error", error=str(e))
    finally:
        conn.close()


def get_top_hashtags(platform: str = None, limit: int = 20) -> list[dict]:
    """Get top-performing hashtags by engagement."""
    conn = _get_db()
    try:
        if platform:
            rows = conn.execute(
                "SELECT * FROM hashtag_performance WHERE platform = ? ORDER BY avg_engagement DESC LIMIT ?",
                (platform, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM hashtag_performance ORDER BY avg_engagement DESC LIMIT ?",
                (limit,)
            ).fetchall()

        return [
            {
                "hashtag": r["hashtag"],
                "platform": r["platform"],
                "uses": r["total_uses"],
                "total_engagement": r["total_engagement"],
                "avg_engagement": round(r["avg_engagement"], 2),
                "last_used": r["last_used"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_hashtag_report(days: int = 30) -> dict:
    """Comprehensive hashtag performance report."""
    conn = _get_db()
    try:
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")

        # Most used
        most_used = conn.execute(
            "SELECT hashtag, SUM(total_uses) as uses FROM hashtag_performance "
            "WHERE last_used >= ? GROUP BY hashtag ORDER BY uses DESC LIMIT 10",
            (since,)
        ).fetchall()

        # Best engagement
        best_engagement = conn.execute(
            "SELECT hashtag, platform, avg_engagement FROM hashtag_performance "
            "WHERE last_used >= ? AND total_uses >= 3 ORDER BY avg_engagement DESC LIMIT 10",
            (since,)
        ).fetchall()

        # Worst performers (high usage, low engagement)
        worst = conn.execute(
            "SELECT hashtag, platform, total_uses, avg_engagement FROM hashtag_performance "
            "WHERE last_used >= ? AND total_uses >= 5 AND avg_engagement < 1.0 "
            "ORDER BY avg_engagement ASC LIMIT 5",
            (since,)
        ).fetchall()

        return {
            "period_days": days,
            "most_used": [{"hashtag": r["hashtag"], "uses": r["uses"]} for r in most_used],
            "best_engagement": [
                {"hashtag": r["hashtag"], "platform": r["platform"], "avg_engagement": round(r["avg_engagement"], 2)}
                for r in best_engagement
            ],
            "underperforming": [
                {"hashtag": r["hashtag"], "platform": r["platform"],
                 "uses": r["total_uses"], "avg_engagement": round(r["avg_engagement"], 2)}
                for r in worst
            ],
            "recommendation": "Replace underperforming hashtags with higher-engagement alternatives",
        }
    finally:
        conn.close()


# ════════════════════════════════════════════════
# Dashboard Summary
# ════════════════════════════════════════════════

def get_analytics_dashboard(days: int = 30) -> dict:
    """Full analytics dashboard data."""
    return {
        "publish_stats": get_publish_stats(days),
        "best_times": get_best_posting_times(days=days),
        "top_hashtags": get_top_hashtags(limit=10),
        "hashtag_report": get_hashtag_report(days),
        "generated_at": datetime.utcnow().isoformat(),
    }
