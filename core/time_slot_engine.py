"""
ViralOps Engine — Time Slot Suggestion Engine

Recommends optimal posting times per platform based on:
  1. Static engagement data (OPTIMAL_TIME_SLOTS from scheduler.py)
  2. Historical publish_log performance (actual success/engagement data)
  3. Platform-specific rules (LinkedIn = work hours, TikTok = evening, etc.)
  4. Timezone awareness (convert UTC → user's local timezone)

Features:
  - suggest_time(platform) → best next posting time
  - suggest_schedule(platforms, posts_per_day) → full daily schedule
  - get_best_hours(platform, days) → analytics-backed best hours
"""

from __future__ import annotations

import os
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any
from collections import defaultdict

import structlog

logger = structlog.get_logger()

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "web", "viralops.db"
)

# Import optimal slots from scheduler
OPTIMAL_TIME_SLOTS = {
    "tiktok":    [7, 10, 19, 21],
    "instagram": [8, 11, 14, 17],
    "facebook":  [9, 13, 16],
    "youtube":   [14, 17, 20],
    "twitter":   [8, 12, 17, 21],
    "linkedin":  [7, 10, 12],
    "pinterest": [14, 20, 21],
    "reddit":    [6, 8, 12],
    "medium":    [10, 14],
    "tumblr":    [19, 22],
    "shopify_blog": [10],
    "threads":   [8, 12, 17, 20],
    "bluesky":   [9, 12, 17, 21],
    "mastodon":  [8, 11, 17, 20],
    "quora":     [9, 14],
    "lemon8":    [10, 14, 19],
}

# Day-of-week multipliers (Mon=0, Sun=6)
# Based on general social media engagement patterns
DAY_OF_WEEK_WEIGHTS = {
    "tiktok":    {0: 0.8, 1: 0.9, 2: 1.0, 3: 1.0, 4: 0.9, 5: 1.1, 6: 1.1},
    "instagram": {0: 0.8, 1: 0.9, 2: 1.0, 3: 1.0, 4: 0.9, 5: 1.1, 6: 1.0},
    "linkedin":  {0: 1.0, 1: 1.1, 2: 1.1, 3: 1.0, 4: 0.8, 5: 0.4, 6: 0.3},
    "twitter":   {0: 0.9, 1: 0.9, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.8, 6: 0.7},
}


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def suggest_time(
    platform: str,
    utc_offset_hours: int = 0,
    avoid_hours: list[int] | None = None,
) -> dict:
    """
    Suggest the next best posting time for a platform.

    Args:
        platform: Target platform name
        utc_offset_hours: User's timezone offset from UTC
        avoid_hours: Hours (UTC) to avoid posting

    Returns:
        {
            "platform": str,
            "suggested_utc": str (ISO format),
            "suggested_local": str,
            "hour_utc": int,
            "reason": str,
            "confidence": float (0-1),
        }
    """
    now = datetime.utcnow()
    current_hour = now.hour
    avoid = set(avoid_hours or [])

    # Get optimal hours for this platform
    optimal_hours = OPTIMAL_TIME_SLOTS.get(platform, [10, 14, 18])

    # Check if we have analytics data for smarter suggestions
    analytics_hours = _get_analytics_best_hours(platform, days=30)

    # Merge: prefer analytics data if enough samples, else use static
    if analytics_hours and len(analytics_hours) >= 3:
        best_hours = [h["hour"] for h in analytics_hours[:6]]
        source = "analytics"
        confidence = 0.85
    else:
        best_hours = optimal_hours
        source = "engagement_research"
        confidence = 0.65

    # Filter out avoided hours
    best_hours = [h for h in best_hours if h not in avoid]
    if not best_hours:
        best_hours = optimal_hours  # Fallback

    # Find next available slot
    suggested_hour = None
    for h in sorted(best_hours):
        if h > current_hour:
            suggested_hour = h
            break

    # If all today's slots passed, suggest first slot tomorrow
    tomorrow = False
    if suggested_hour is None:
        suggested_hour = best_hours[0]
        tomorrow = True

    # Build suggested datetime
    suggested_dt = now.replace(
        hour=suggested_hour, minute=0, second=0, microsecond=0
    )
    if tomorrow:
        suggested_dt += timedelta(days=1)

    # Apply day-of-week weighting
    dow = suggested_dt.weekday()
    dow_weights = DAY_OF_WEEK_WEIGHTS.get(platform, {})
    dow_weight = dow_weights.get(dow, 1.0)
    confidence *= dow_weight

    # Convert to local time
    local_dt = suggested_dt + timedelta(hours=utc_offset_hours)

    return {
        "platform": platform,
        "suggested_utc": suggested_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "suggested_local": local_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "hour_utc": suggested_hour,
        "hour_local": (suggested_hour + utc_offset_hours) % 24,
        "tomorrow": tomorrow,
        "reason": f"Based on {source}",
        "confidence": round(min(1.0, confidence), 2),
        "day_of_week": [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ][dow],
    }


def suggest_schedule(
    platforms: list[str],
    posts_per_platform: int = 1,
    utc_offset_hours: int = 0,
    spacing_minutes: int = 30,
) -> dict:
    """
    Generate a full daily posting schedule across multiple platforms.

    Spaces out posts to avoid flooding and maximize reach.

    Args:
        platforms: List of platform names
        posts_per_platform: How many posts per platform per day
        utc_offset_hours: User's timezone offset from UTC
        spacing_minutes: Minimum minutes between any two posts

    Returns:
        {
            "schedule": [
                {"platform": str, "time_utc": str, "time_local": str, "slot": int},
                ...
            ],
            "total_posts": int,
        }
    """
    schedule = []
    used_times: set[int] = set()  # Track used time slots (minutes from midnight)

    for platform in platforms:
        optimal_hours = OPTIMAL_TIME_SLOTS.get(platform, [10, 14, 18])

        for i in range(posts_per_platform):
            if i < len(optimal_hours):
                target_hour = optimal_hours[i]
            else:
                # Spread additional posts evenly
                target_hour = optimal_hours[i % len(optimal_hours)]

            target_minutes = target_hour * 60

            # Avoid conflicts with existing schedule
            while target_minutes in used_times or any(
                abs(target_minutes - t) < spacing_minutes
                for t in used_times
            ):
                target_minutes += spacing_minutes
                if target_minutes >= 1440:
                    target_minutes = target_minutes % 1440

            used_times.add(target_minutes)

            slot_hour = target_minutes // 60
            slot_min = target_minutes % 60
            local_hour = (slot_hour + utc_offset_hours) % 24

            schedule.append(
                {
                    "platform": platform,
                    "time_utc": f"{slot_hour:02d}:{slot_min:02d}",
                    "time_local": f"{local_hour:02d}:{slot_min:02d}",
                    "slot": i + 1,
                }
            )

    # Sort by UTC time
    schedule.sort(key=lambda s: s["time_utc"])

    return {
        "schedule": schedule,
        "total_posts": len(schedule),
        "spacing_minutes": spacing_minutes,
    }


def _get_analytics_best_hours(
    platform: str, days: int = 30
) -> list[dict]:
    """Get best posting hours from actual publish_log data."""
    try:
        conn = _get_db()
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

        cursor = conn.execute(
            """
            SELECT
                CAST(substr(published_at, 12, 2) AS INTEGER) as hour,
                COUNT(*) as total,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
            FROM publish_log
            WHERE platform = ? AND published_at >= ?
            GROUP BY hour
            ORDER BY successes DESC, total DESC
            """,
            (platform, cutoff),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "hour": r["hour"],
                "total": r["total"],
                "successes": r["successes"],
                "success_rate": round(
                    r["successes"] / r["total"] if r["total"] > 0 else 0, 2
                ),
            }
            for r in rows
        ]
    except Exception:
        return []


def get_best_hours(platform: str | None = None, days: int = 30) -> dict:
    """
    Public API: Get best posting hours (analytics + static combined).

    Returns ranked hours with confidence scores.
    """
    result: dict[str, Any] = {"period_days": days, "platforms": {}}

    platforms = [platform] if platform else list(OPTIMAL_TIME_SLOTS.keys())

    for plat in platforms:
        analytics = _get_analytics_best_hours(plat, days)
        static = OPTIMAL_TIME_SLOTS.get(plat, [])

        if analytics and len(analytics) >= 3:
            hours = [
                {
                    "hour": h["hour"],
                    "source": "analytics",
                    "success_rate": h["success_rate"],
                    "sample_size": h["total"],
                }
                for h in analytics[:6]
            ]
        else:
            hours = [
                {"hour": h, "source": "research", "success_rate": None, "sample_size": 0}
                for h in static
            ]

        result["platforms"][plat] = {
            "best_hours": hours,
            "data_source": "analytics" if analytics else "research",
        }

    return result
