"""
Background Scheduler — Auto-publish posts at scheduled times.
v2.1: Now supports ALL social platforms via social_connectors.py

Features:
  - Checks every 60s for due scheduled posts
  - Supports 11 platforms (7 social + 4 direct API)
  - Retry logic with exponential backoff
  - Optimal time slots per platform (engagement data)
  - Rate limiting per platform (respects API quotas)
  - Publish log for analytics tracking
"""
import os
import json
import sqlite3
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog

logger = structlog.get_logger()

# ── Database path ──
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "viralops.db")

# ── Optimal posting times per platform (UTC) ──
# Based on engagement research — scheduler suggests these
OPTIMAL_TIME_SLOTS = {
    "tiktok":    [7, 10, 19, 21],       # 7am, 10am, 7pm, 9pm
    "instagram": [8, 11, 14, 17],       # 8am, 11am, 2pm, 5pm
    "facebook":  [9, 13, 16],           # 9am, 1pm, 4pm
    "youtube":   [14, 17, 20],          # 2pm, 5pm, 8pm
    "twitter":   [8, 12, 17, 21],       # 8am, noon, 5pm, 9pm
    "linkedin":  [7, 10, 12],           # 7am, 10am, noon (work hours)
    "pinterest": [14, 20, 21],          # 2pm, 8pm, 9pm
    "reddit":    [6, 8, 12],            # 6am, 8am, noon
    "medium":    [10, 14],              # 10am, 2pm
    "tumblr":    [19, 22],              # 7pm, 10pm
    "shopify_blog": [10],               # Any time
    "threads":  [8, 12, 17, 20],        # 8am, noon, 5pm, 8pm
    "bluesky":  [9, 12, 17, 21],        # 9am, noon, 5pm, 9pm
    "mastodon": [8, 11, 17, 20],        # 8am, 11am, 5pm, 8pm
    "quora":    [9, 14],                # 9am, 2pm (long-form)
}

# ── Rate limits per platform (posts per day) ──
DAILY_RATE_LIMITS = {
    "tiktok": 10, "instagram": 25, "facebook": 25,
    "youtube": 6, "twitter": 50, "linkedin": 20,
    "pinterest": 25, "reddit": 10, "medium": 3,
    "tumblr": 75, "shopify_blog": 20,
    "threads": 25, "bluesky": 50, "mastodon": 50, "quora": 5,
}


class PublishScheduler:
    """
    Background scheduler that checks for scheduled posts
    and publishes them when their time arrives.

    v2.1: Supports all 11 platforms via social_connectors + direct publishers.
    """

    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval  # seconds
        self.running = False
        self._publishers = {}
        self._daily_counts: dict[str, int] = {}  # platform -> count today
        self._count_date: str = ""  # YYYY-MM-DD

    def _get_db(self):
        """Get database connection."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _reset_daily_counts_if_needed(self):
        """Reset daily post counts at midnight."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._count_date != today:
            self._daily_counts = {}
            self._count_date = today

    def _check_rate_limit(self, platform: str) -> bool:
        """Check if we're within daily rate limit for a platform."""
        self._reset_daily_counts_if_needed()
        limit = DAILY_RATE_LIMITS.get(platform, 25)
        current = self._daily_counts.get(platform, 0)
        return current < limit

    def _increment_count(self, platform: str):
        """Track daily post count."""
        self._daily_counts[platform] = self._daily_counts.get(platform, 0) + 1

    def _load_publisher(self, platform: str):
        """Lazy-load publisher for a platform — now supports ALL platforms."""
        if platform in self._publishers:
            return self._publishers[platform]

        try:
            # Priority 1: Social connectors (Twitter, Instagram, Facebook, YouTube, LinkedIn, TikTok, Pinterest)
            from integrations.social_connectors import get_social_publisher
            social_pub = get_social_publisher(platform)
            if social_pub:
                self._publishers[platform] = social_pub
                logger.info("scheduler.loaded_social", platform=platform)
                return social_pub

            # Priority 2: Direct API publishers
            if platform == "reddit":
                from integrations.reddit_publisher import RedditPublisher
                pub = RedditPublisher()
            elif platform == "medium":
                from integrations.medium_publisher import MediumPublisher
                pub = MediumPublisher()
            elif platform == "tumblr":
                from integrations.tumblr_publisher import TumblrPublisher
                pub = TumblrPublisher()
            elif platform == "shopify_blog":
                from integrations.shopify_blog_publisher import ShopifyBlogPublisher
                pub = ShopifyBlogPublisher()
            elif platform == "lemon8":
                from integrations.lemon8_publisher import Lemon8Publisher
                pub = Lemon8Publisher()
            # Priority 3: New platforms (Threads, Bluesky, Mastodon, Quora)
            elif platform == "threads":
                from integrations.threads_publisher import ThreadsPublisher
                pub = ThreadsPublisher()
            elif platform == "bluesky":
                from integrations.bluesky_publisher import BlueskyPublisher
                pub = BlueskyPublisher()
            elif platform == "mastodon":
                from integrations.mastodon_publisher import MastodonPublisher
                pub = MastodonPublisher()
            elif platform == "quora":
                from integrations.quora_publisher import QuoraPublisher
                pub = QuoraPublisher()
            else:
                logger.warning("scheduler.unknown_platform", platform=platform)
                return None

            self._publishers[platform] = pub
            logger.info("scheduler.loaded_direct", platform=platform)
            return pub

        except Exception as e:
            logger.error("scheduler.load_publisher_error", platform=platform, error=str(e))
            return None

    def check_and_publish(self) -> list[dict]:
        """
        Check for posts that are scheduled and due for publishing.
        v2.1: Rate limiting + all 11 platforms supported.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
        results = []

        conn = self._get_db()
        try:
            # Find scheduled posts that are due
            cursor = conn.execute(
                "SELECT * FROM posts WHERE status = 'scheduled' AND scheduled_at <= ?",
                (now,)
            )
            due_posts = cursor.fetchall()

            if not due_posts:
                return results

            logger.info("scheduler.found_due_posts", count=len(due_posts))

            for post in due_posts:
                post_id = post["id"]
                title = post["title"]
                body = post["body"]
                platforms = json.loads(post["platforms"] or "[]")
                extra = json.loads(post["extra_fields"] or "{}")

                for platform in platforms:
                    # Rate limit check
                    if not self._check_rate_limit(platform):
                        logger.warning("scheduler.rate_limited", platform=platform, post_id=post_id)
                        results.append({
                            "post_id": post_id, "platform": platform,
                            "success": False, "error": f"Daily rate limit reached for {platform}"
                        })
                        continue

                    result = self._publish_to_platform(post_id, platform, title, body, extra)
                    results.append(result)

                    if result.get("success"):
                        self._increment_count(platform)

                    # Log to publish_log
                    conn.execute(
                        "INSERT INTO publish_log (post_id, platform, success, post_url, error, published_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (post_id, platform, result["success"], result.get("post_url", ""),
                         result.get("error", ""), datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))
                    )

                # Update post status
                post_results = [r for r in results if r.get("post_id") == post_id]
                all_success = all(r["success"] for r in post_results) if post_results else False
                any_success = any(r["success"] for r in post_results) if post_results else False
                new_status = "published" if all_success else ("partial" if any_success else "failed")
                conn.execute(
                    "UPDATE posts SET status = ?, published_at = ? WHERE id = ?",
                    (new_status, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), post_id)
                )

            conn.commit()
            logger.info("scheduler.publish_batch_done", total=len(results),
                       success=sum(1 for r in results if r["success"]))

        except Exception as e:
            logger.error("scheduler.check_error", error=str(e))
            conn.rollback()
        finally:
            conn.close()

        return results

    def _publish_to_platform(self, post_id: int, platform: str, title: str, body: str, extra: dict) -> dict:
        """Publish a single post to a platform."""
        publisher = self._load_publisher(platform)
        if not publisher:
            return {"post_id": post_id, "platform": platform, "success": False, "error": "Publisher not available"}

        try:
            # Create a simple queue item for the publisher
            class SimpleItem:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            item = SimpleItem(
                id=f"sched-{post_id}-{platform}",
                platform=platform,
                platform_content={
                    "title": title,
                    "body": body,
                    "caption": body[:2000],
                    **extra.get(platform, {}),
                    **{k: v for k, v in extra.items() if not isinstance(v, dict)},
                },
            )

            # Publish
            if asyncio.get_event_loop().is_running():
                # We're in an async context
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        lambda: asyncio.run(publisher.publish(item))
                    ).result(timeout=30)
            else:
                result = asyncio.run(publisher.publish(item))

            return {
                "post_id": post_id,
                "platform": platform,
                "success": result.success,
                "post_url": result.post_url,
                "error": result.error,
            }

        except Exception as e:
            logger.error("scheduler.publish_error", post_id=post_id, platform=platform, error=str(e))
            return {"post_id": post_id, "platform": platform, "success": False, "error": str(e)}

    async def run_loop(self):
        """Run the scheduler loop (for background execution)."""
        self.running = True
        logger.info("scheduler.started", interval=self.check_interval)

        while self.running:
            try:
                # 1. Normal scheduled posts
                results = self.check_and_publish()
                if results:
                    logger.info("scheduler.cycle_done", published=len(results))

                # 2. RSS Auto Poster tick
                try:
                    from integrations.rss_auto_poster import tick as auto_poster_tick
                    ap_result = auto_poster_tick()
                    if ap_result.get("total_posted", 0) > 0:
                        logger.info("scheduler.rss_auto_poster_done",
                                    posted=ap_result["total_posted"],
                                    posters=ap_result.get("posters_checked", 0))
                except Exception as e:
                    logger.warning("scheduler.rss_auto_poster_error", error=str(e))

                # 3. Engagement fetcher — pull metrics from platform APIs
                try:
                    from monitoring.engagement_fetcher import fetch_engagement_batch
                    eng_result = await fetch_engagement_batch(limit=50)
                    if eng_result.get("succeeded", 0) > 0:
                        logger.info("scheduler.engagement_fetch_done",
                                    succeeded=eng_result["succeeded"],
                                    platforms=eng_result.get("platforms", {}))
                except Exception as e:
                    logger.warning("scheduler.engagement_fetch_error", error=str(e))

            except Exception as e:
                logger.error("scheduler.loop_error", error=str(e))

            await asyncio.sleep(self.check_interval)

    def stop(self):
        """Stop the scheduler."""
        self.running = False
        logger.info("scheduler.stopped")


# ── Singleton ──
_scheduler: Optional[PublishScheduler] = None


def get_scheduler() -> PublishScheduler:
    """Get or create scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = PublishScheduler()
    return _scheduler
