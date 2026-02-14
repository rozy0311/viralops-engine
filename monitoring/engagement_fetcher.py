"""
ViralOps Engine — Engagement Fetcher

Pulls REAL engagement metrics from each platform's API for published posts,
then updates the local database for analytics and feedback loops.

Flow:
  1. Query publish_log for recent successful posts
  2. For each post, load the appropriate publisher
  3. Call publisher.get_metrics(post_id) → {likes, shares, views, ...}
  4. Upsert into engagement_data table
  5. Update hashtag_performance with real engagement numbers

Scheduler integration:
  - Called from scheduler run_loop every cycle (or on demand)
  - Respects rate limits (max 100 metric fetches per cycle)
"""

from __future__ import annotations

import json
import os
import sqlite3
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

logger = structlog.get_logger()

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "web", "viralops.db"
)

# Max metric fetches per cycle to avoid hitting API rate limits
MAX_FETCH_PER_CYCLE = 100

# How far back to fetch engagement (days)
ENGAGEMENT_LOOKBACK_DAYS = 7

# Platforms that support get_metrics()
SUPPORTED_PLATFORMS = {
    "twitter", "instagram", "facebook", "youtube", "linkedin",
    "tiktok", "pinterest", "reddit", "medium", "tumblr",
    "shopify_blog", "lemon8", "threads", "bluesky", "mastodon", "quora",
}


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_engagement_table():
    """Create engagement_data table if it doesn't exist."""
    conn = _get_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS engagement_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER NOT NULL,
                platform TEXT NOT NULL,
                platform_post_id TEXT NOT NULL,
                likes INTEGER DEFAULT 0,
                shares INTEGER DEFAULT 0,
                comments INTEGER DEFAULT 0,
                views INTEGER DEFAULT 0,
                saves INTEGER DEFAULT 0,
                clicks INTEGER DEFAULT 0,
                extra_metrics TEXT DEFAULT '{}',
                fetched_at TEXT NOT NULL,
                UNIQUE(post_id, platform, fetched_at)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_engagement_platform
            ON engagement_data(platform, fetched_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_engagement_post
            ON engagement_data(post_id, platform)
        """)
        conn.commit()
    finally:
        conn.close()


def _get_recent_published_posts(days: int = ENGAGEMENT_LOOKBACK_DAYS) -> list[dict]:
    """Get recently published posts that have platform post IDs."""
    conn = _get_db()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        cursor = conn.execute(
            """
            SELECT pl.post_id, pl.platform, pl.post_url, pl.published_at
            FROM publish_log pl
            WHERE pl.success = 1 AND pl.published_at >= ?
            ORDER BY pl.published_at DESC
            LIMIT ?
            """,
            (cutoff, MAX_FETCH_PER_CYCLE * 2),
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _extract_platform_post_id(platform: str, post_url: str) -> str | None:
    """Extract the platform-specific post ID from the post URL."""
    if not post_url:
        return None

    import re

    extractors = {
        "twitter": r"/status/(\d+)",
        "instagram": r"/p/([A-Za-z0-9_-]+)",
        "facebook": r"/posts?/(\d+)",
        "youtube": r"(?:v=|youtu\.be/)([A-Za-z0-9_-]+)",
        "linkedin": r"activity-(\d+)",
        "tiktok": r"/video/(\d+)",
        "pinterest": r"/pin/(\d+)",
        "reddit": r"/comments/([a-z0-9]+)",
        "medium": r"([a-f0-9]+)$",
        "tumblr": r"/post/(\d+)",
        "threads": r"/post/([A-Za-z0-9_-]+)",
        "bluesky": r"/post/([a-z0-9]+)",
        "mastodon": r"/(\d+)$",
        "quora": r"/answer/(\d+)",
    }

    pattern = extractors.get(platform)
    if pattern:
        match = re.search(pattern, post_url)
        if match:
            return match.group(1)

    # Fallback: use last path segment
    parts = post_url.rstrip("/").split("/")
    return parts[-1] if parts else None


async def _get_publisher_for_metrics(platform: str):
    """Load publisher instance for fetching metrics."""
    try:
        # Try social_connectors first
        from integrations.social_connectors import get_social_publisher

        pub = get_social_publisher(platform)
        if pub:
            return pub

        # Try direct publishers
        publishers_map = {
            "reddit": ("integrations.reddit_publisher", "RedditPublisher"),
            "medium": ("integrations.medium_publisher", "MediumPublisher"),
            "tumblr": ("integrations.tumblr_publisher", "TumblrPublisher"),
            "shopify_blog": (
                "integrations.shopify_blog_publisher",
                "ShopifyBlogPublisher",
            ),
            "lemon8": ("integrations.lemon8_publisher", "Lemon8Publisher"),
            "threads": ("integrations.threads_publisher", "ThreadsPublisher"),
            "bluesky": ("integrations.bluesky_publisher", "BlueskyPublisher"),
            "mastodon": (
                "integrations.mastodon_publisher",
                "MastodonPublisher",
            ),
            "quora": ("integrations.quora_publisher", "QuoraPublisher"),
        }

        if platform in publishers_map:
            module_path, class_name = publishers_map[platform]
            import importlib

            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            pub = cls()
            # Try to connect
            if hasattr(pub, "connect"):
                await pub.connect()
            return pub

        return None
    except Exception as e:
        logger.warning(
            "engagement.publisher_load_failed",
            platform=platform,
            error=str(e),
        )
        return None


def _normalize_metrics(platform: str, raw: dict) -> dict:
    """Normalize platform-specific metric names to standard fields."""
    normalized = {
        "likes": 0,
        "shares": 0,
        "comments": 0,
        "views": 0,
        "saves": 0,
        "clicks": 0,
    }

    # Map platform-specific keys to standard keys
    like_keys = [
        "likes",
        "like_count",
        "favourites",
        "upvotes",
        "numUpvotes",
        "claps",
    ]
    share_keys = [
        "shares",
        "share_count",
        "reblogs",
        "reposts",
        "repost_count",
        "numShares",
        "retweets",
    ]
    comment_keys = [
        "comments",
        "comment_count",
        "replies",
        "replies_count",
        "numComments",
    ]
    view_keys = ["views", "view_count", "impressions", "numViews", "plays"]
    save_keys = ["saves", "save_count", "bookmarks", "quotes"]
    click_keys = ["clicks", "click_count", "url_clicks"]

    for key, val in raw.items():
        if not isinstance(val, (int, float)):
            continue
        val = int(val)
        if key in like_keys:
            normalized["likes"] += val
        elif key in share_keys:
            normalized["shares"] += val
        elif key in comment_keys:
            normalized["comments"] += val
        elif key in view_keys:
            normalized["views"] += val
        elif key in save_keys:
            normalized["saves"] += val
        elif key in click_keys:
            normalized["clicks"] += val

    return normalized


def _store_engagement(
    post_id: int,
    platform: str,
    platform_post_id: str,
    metrics: dict,
) -> None:
    """Store engagement data in database."""
    conn = _get_db()
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        # Standard metrics
        std = _normalize_metrics(platform, metrics)

        # Extra non-standard metrics
        standard_keys = {"likes", "shares", "comments", "views", "saves", "clicks"}
        extra = {k: v for k, v in metrics.items() if k not in standard_keys and isinstance(v, (int, float, str))}

        conn.execute(
            """
            INSERT OR REPLACE INTO engagement_data
            (post_id, platform, platform_post_id, likes, shares, comments,
             views, saves, clicks, extra_metrics, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                post_id,
                platform,
                platform_post_id,
                std["likes"],
                std["shares"],
                std["comments"],
                std["views"],
                std["saves"],
                std["clicks"],
                json.dumps(extra),
                now,
            ),
        )
        conn.commit()
        logger.info(
            "engagement.stored",
            post_id=post_id,
            platform=platform,
            likes=std["likes"],
            views=std["views"],
        )
    finally:
        conn.close()


def _update_hashtag_engagement(
    post_id: int, platform: str, total_engagement: int
) -> None:
    """Update hashtag performance with real engagement data."""
    conn = _get_db()
    try:
        # Get hashtags from the post
        cursor = conn.execute(
            "SELECT extra_fields FROM posts WHERE id = ?", (post_id,)
        )
        row = cursor.fetchone()
        if not row:
            return
        extra = json.loads(row["extra_fields"] or "{}")
        hashtags = extra.get("hashtags", [])
        if isinstance(hashtags, str):
            hashtags = hashtags.split()

        # Update each hashtag's engagement
        if hashtags:
            try:
                from monitoring.analytics import update_hashtag_engagement

                for tag in hashtags[:10]:  # Max 10 hashtags
                    update_hashtag_engagement(tag, platform, total_engagement)
            except ImportError:
                pass
    finally:
        conn.close()


async def fetch_engagement_batch(
    limit: int = MAX_FETCH_PER_CYCLE,
) -> dict:
    """
    Main entry point: Fetch engagement for recent posts.

    Returns:
        {
            "fetched": int,
            "succeeded": int,
            "failed": int,
            "platforms": {platform: count},
            "errors": [str],
        }
    """
    ensure_engagement_table()

    posts = _get_recent_published_posts()
    if not posts:
        return {"fetched": 0, "succeeded": 0, "failed": 0, "platforms": {}, "errors": []}

    # Deduplicate (keep latest per post_id+platform)
    seen: set[tuple[int, str]] = set()
    unique_posts: list[dict] = []
    for p in posts:
        key = (p["post_id"], p["platform"])
        if key not in seen and len(unique_posts) < limit:
            seen.add(key)
            unique_posts.append(p)

    result = {
        "fetched": len(unique_posts),
        "succeeded": 0,
        "failed": 0,
        "platforms": {},
        "errors": [],
    }

    # Group by platform for efficient publisher loading
    from collections import defaultdict

    by_platform: dict[str, list[dict]] = defaultdict(list)
    for p in unique_posts:
        by_platform[p["platform"]].append(p)

    for platform, platform_posts in by_platform.items():
        if platform not in SUPPORTED_PLATFORMS:
            continue

        publisher = await _get_publisher_for_metrics(platform)
        if not publisher:
            result["errors"].append(f"{platform}: Publisher unavailable")
            result["failed"] += len(platform_posts)
            continue

        count = 0
        for post_data in platform_posts:
            post_id = post_data["post_id"]
            post_url = post_data.get("post_url", "")
            platform_post_id = _extract_platform_post_id(platform, post_url)

            if not platform_post_id:
                result["failed"] += 1
                continue

            try:
                if hasattr(publisher, "get_metrics"):
                    metrics = await publisher.get_metrics(platform_post_id)
                else:
                    metrics = None

                if metrics:
                    _store_engagement(
                        post_id, platform, platform_post_id, metrics
                    )

                    # Update hashtag performance
                    std = _normalize_metrics(platform, metrics)
                    total = (
                        std["likes"]
                        + std["shares"]
                        + std["comments"]
                        + std["views"]
                    )
                    if total > 0:
                        _update_hashtag_engagement(post_id, platform, total)

                    result["succeeded"] += 1
                    count += 1
                else:
                    result["failed"] += 1

            except Exception as e:
                result["errors"].append(
                    f"{platform}/{post_id}: {str(e)[:100]}"
                )
                result["failed"] += 1

        result["platforms"][platform] = count

        # Cleanup publisher
        if hasattr(publisher, "close"):
            try:
                await publisher.close()
            except Exception:
                pass

    logger.info(
        "engagement.batch_done",
        succeeded=result["succeeded"],
        failed=result["failed"],
        platforms=result["platforms"],
    )
    return result


def get_engagement_summary(
    platform: str | None = None, days: int = 7
) -> dict:
    """Get engagement summary from stored data."""
    ensure_engagement_table()
    conn = _get_db()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        query = """
            SELECT platform,
                   COUNT(*) as total_posts,
                   SUM(likes) as total_likes,
                   SUM(shares) as total_shares,
                   SUM(comments) as total_comments,
                   SUM(views) as total_views,
                   AVG(likes) as avg_likes,
                   AVG(views) as avg_views
            FROM engagement_data
            WHERE fetched_at >= ?
        """
        params: list[Any] = [cutoff]

        if platform:
            query += " AND platform = ?"
            params.append(platform)

        query += " GROUP BY platform ORDER BY total_views DESC"

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

        return {
            "period_days": days,
            "platforms": [
                {
                    "platform": r["platform"],
                    "total_posts": r["total_posts"],
                    "total_likes": r["total_likes"] or 0,
                    "total_shares": r["total_shares"] or 0,
                    "total_comments": r["total_comments"] or 0,
                    "total_views": r["total_views"] or 0,
                    "avg_likes": round(r["avg_likes"] or 0, 1),
                    "avg_views": round(r["avg_views"] or 0, 1),
                }
                for r in rows
            ],
        }
    finally:
        conn.close()


def get_post_engagement(post_id: int) -> list[dict]:
    """Get all engagement data for a specific post."""
    ensure_engagement_table()
    conn = _get_db()
    try:
        cursor = conn.execute(
            """
            SELECT * FROM engagement_data
            WHERE post_id = ?
            ORDER BY fetched_at DESC
            """,
            (post_id,),
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()
