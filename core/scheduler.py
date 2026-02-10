"""
Background Scheduler — Auto-publish posts at scheduled times.
Uses APScheduler for reliable background job execution.
"""
import os
import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
from typing import Optional

import structlog

logger = structlog.get_logger()

# ── Database path ──
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "viralops.db")


class PublishScheduler:
    """
    Background scheduler that checks for scheduled posts
    and publishes them when their time arrives.
    """

    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval  # seconds
        self.running = False
        self._publishers = {}

    def _get_db(self):
        """Get database connection."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_publisher(self, platform: str):
        """Lazy-load publisher for a platform."""
        if platform in self._publishers:
            return self._publishers[platform]

        try:
            if platform == "reddit":
                from integrations.reddit_publisher import RealRedditPublisher
                pub = RealRedditPublisher()
            elif platform == "medium":
                from integrations.medium_publisher import RealMediumPublisher
                pub = RealMediumPublisher()
            elif platform == "tumblr":
                from integrations.tumblr_publisher import RealTumblrPublisher
                pub = RealTumblrPublisher()
            elif platform == "shopify_blog":
                from integrations.shopify_blog_publisher import ShopifyBlogPublisher
                pub = ShopifyBlogPublisher()
            else:
                logger.warning("scheduler.unknown_platform", platform=platform)
                return None

            self._publishers[platform] = pub
            return pub
        except Exception as e:
            logger.error("scheduler.load_publisher_error", platform=platform, error=str(e))
            return None

    def check_and_publish(self) -> list[dict]:
        """
        Check for posts that are scheduled and due for publishing.
        Returns list of publish results.
        """
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M")
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
                    result = self._publish_to_platform(post_id, platform, title, body, extra)
                    results.append(result)

                    # Log to publish_log
                    conn.execute(
                        "INSERT INTO publish_log (post_id, platform, success, post_url, error, published_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (post_id, platform, result["success"], result.get("post_url", ""),
                         result.get("error", ""), datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"))
                    )

                # Update post status
                all_success = all(r["success"] for r in results if r["post_id"] == post_id)
                new_status = "published" if all_success else "failed"
                conn.execute(
                    "UPDATE posts SET status = ?, published_at = ? WHERE id = ?",
                    (new_status, datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"), post_id)
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
                results = self.check_and_publish()
                if results:
                    logger.info("scheduler.cycle_done", published=len(results))
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
