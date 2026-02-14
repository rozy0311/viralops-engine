"""
Shopify Blog Auto-Share — Automatically share new blog posts to TikTok + Pinterest.

Pipeline:
  1. ShopifyBlogWatcher polls for new articles from watched blogs
  2. Content transformer extracts image, title, excerpt, hashtags
  3. Multi-TikTok publisher broadcasts/round-robins to N accounts (photo posts)
  4. Pinterest publisher creates pins (image + title + link back to blog)
  5. History tracking prevents duplicate shares
  6. Telegram alert on success

Watched blogs (default):
  - https://therike.com/blogs/sustainable-living
  - https://therike.com/blogs/brand-partnerships

Config via .env:
  SHOPIFY_WATCH_BLOGS=sustainable-living,brand-partnerships
  SHOPIFY_AUTOSHARE_TIKTOK_MODE=broadcast|round_robin|specific
  SHOPIFY_AUTOSHARE_PINTEREST_ENABLED=true
  SHOPIFY_AUTOSHARE_INTERVAL_MIN=30
  SHOPIFY_AUTOSHARE_MAX_PER_CYCLE=5
"""

from __future__ import annotations

import json
import logging
import os
import hashlib
import httpx
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("viralops.shopify_auto_share")

# ── Persistence ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SHARE_HISTORY_FILE = os.path.join(DATA_DIR, "shopify_auto_share_history.json")
SHARE_CONFIG_FILE = os.path.join(DATA_DIR, "shopify_auto_share_config.json")

# ── Media cache for uploads ──────────────────────────────────
MEDIA_CACHE_DIR = os.path.join(DATA_DIR, "media_cache")

# ── Defaults ─────────────────────────────────────────────────
DEFAULT_INTERVAL_MIN = 30
DEFAULT_MAX_PER_CYCLE = 5
DEFAULT_TIKTOK_MODE = "broadcast"  # broadcast | round_robin | specific
DEFAULT_TIKTOK_VIA = "publer"      # publer | api
DEFAULT_PINTEREST_VIA = "publer"    # publer | api


class ShopifyAutoShare:
    """
    Orchestrator: Shopify Blog → TikTok (multi-account) + Pinterest.

    Usage:
        auto_share = ShopifyAutoShare()
        await auto_share.initialize()
        results = await auto_share.tick()  # Called by scheduler
    """

    def __init__(self):
        self._watcher = None
        self._publer = None          # PublerPublisher (REST API bridge)
        self._tiktok = None          # MultiTikTokPublisher (API fallback)
        self._pinterest = None       # PinterestPublisher (API fallback)
        self._initialized = False
        self._config = self._load_config()
        self._history = self._load_history()

    @property
    def _tiktok_via(self) -> str:
        return os.environ.get(
            "SHOPIFY_AUTOSHARE_TIKTOK_VIA",
            self._config.get("tiktok_via", DEFAULT_TIKTOK_VIA),
        ).lower()

    @property
    def _pinterest_via(self) -> str:
        return os.environ.get(
            "SHOPIFY_AUTOSHARE_PINTEREST_VIA",
            self._config.get("pinterest_via", DEFAULT_PINTEREST_VIA),
        ).lower()

    # ════════════════════════════════════════════
    # Initialization
    # ════════════════════════════════════════════

    async def initialize(self) -> dict:
        """Initialize watcher + publishers. Call once before tick()."""
        errors = []

        # 1. Shopify Blog Watcher
        try:
            from integrations.shopify_blog_watcher import ShopifyBlogWatcher
            self._watcher = ShopifyBlogWatcher()
            connected = await self._watcher.connect()
            if not connected:
                errors.append("ShopifyBlogWatcher: Failed to connect")
        except Exception as e:
            errors.append(f"ShopifyBlogWatcher: {e}")

        # 2. Publer REST API Bridge (~$10/mo per account)
        try:
            from integrations.publer_publisher import PublerPublisher
            self._publer = PublerPublisher()
            if self._publer.is_configured:
                publer_ok = await self._publer.connect()
                if not publer_ok:
                    errors.append("Publer: Connection failed")
                    self._publer = None
                else:
                    logger.info("ShopifyAutoShare: Publer connected ✅")
            else:
                logger.warning(
                    "ShopifyAutoShare: Publer not configured "
                    "(PUBLER_API_KEY empty) — will try direct API"
                )
                self._publer = None
        except Exception as e:
            errors.append(f"Publer: {e}")
            self._publer = None

        # 3. Multi-TikTok Publisher (API fallback)
        if self._tiktok_via == "api" or not self._publer:
            try:
                from integrations.multi_tiktok_publisher import MultiTikTokPublisher
                self._tiktok = MultiTikTokPublisher()
                if not self._tiktok.accounts:
                    if not self._publer:
                        errors.append(
                            "TikTok: No API accounts and Publer not available"
                        )
            except Exception as e:
                errors.append(f"MultiTikTok: {e}")

        # 4. Pinterest Publisher (API fallback)
        pinterest_enabled = os.environ.get(
            "SHOPIFY_AUTOSHARE_PINTEREST_ENABLED", "true"
        ).lower() == "true"
        if pinterest_enabled and (self._pinterest_via == "api" or not self._publer):
            try:
                from integrations.social_connectors import PinterestPublisher
                self._pinterest = PinterestPublisher()
                if not self._pinterest.access_token:
                    if not self._publer:
                        logger.warning(
                            "Pinterest: No API token and Publer not available"
                        )
                    self._pinterest = None
            except Exception as e:
                errors.append(f"Pinterest: {e}")

        self._initialized = bool(self._watcher and self._watcher._connected)

        # Determine effective publishers
        tiktok_method = "publer" if self._publer and self._tiktok_via == "publer" else (
            "api" if self._tiktok and self._tiktok.accounts else "none"
        )
        pinterest_method = "publer" if self._publer and self._pinterest_via == "publer" else (
            "api" if self._pinterest else "none"
        )

        result = {
            "initialized": self._initialized,
            "watcher_connected": bool(self._watcher and self._watcher._connected),
            "publer_connected": self._publer is not None,
            "tiktok_via": tiktok_method,
            "tiktok_accounts": len(self._tiktok.accounts) if self._tiktok else 0,
            "pinterest_via": pinterest_method,
            "pinterest_enabled": pinterest_enabled,
            "errors": errors,
        }
        logger.info("ShopifyAutoShare: Initialized — %s", result)
        return result

    # ════════════════════════════════════════════
    # Main Tick — Called by Scheduler
    # ════════════════════════════════════════════

    async def tick(self) -> dict:
        """
        One cycle of the auto-share pipeline:
        1. Check for new blog articles
        2. Transform to social content
        3. Post to TikTok + Pinterest
        4. Track history
        5. Alert via Telegram
        """
        if not self._initialized:
            init_result = await self.initialize()
            if not self._initialized:
                return {
                    "success": False,
                    "error": "Failed to initialize",
                    "details": init_result,
                }

        # Check interval
        last_tick = self._config.get("last_tick", "")
        interval_min = int(os.environ.get(
            "SHOPIFY_AUTOSHARE_INTERVAL_MIN",
            self._config.get("interval_min", DEFAULT_INTERVAL_MIN),
        ))
        if last_tick:
            try:
                last_dt = datetime.fromisoformat(last_tick)
                now = datetime.now(timezone.utc)
                elapsed_min = (now - last_dt).total_seconds() / 60
                if elapsed_min < interval_min:
                    return {
                        "success": True,
                        "skipped": True,
                        "reason": f"Next check in {interval_min - elapsed_min:.0f} min",
                    }
            except (ValueError, TypeError):
                pass

        # Check paused
        if self._config.get("paused", False):
            return {"success": True, "skipped": True, "reason": "Auto-share is paused"}

        max_per_cycle = int(os.environ.get(
            "SHOPIFY_AUTOSHARE_MAX_PER_CYCLE",
            self._config.get("max_per_cycle", DEFAULT_MAX_PER_CYCLE),
        ))

        # ── Step 1: Fetch new articles ──
        new_articles = await self._watcher.check_new_articles(
            max_per_blog=max_per_cycle
        )

        # Filter already-shared
        unseen = []
        for article in new_articles:
            article_hash = self._article_hash(article)
            if article_hash not in self._history:
                unseen.append(article)

        if not unseen:
            self._update_config("last_tick", datetime.now(timezone.utc).isoformat())
            return {
                "success": True,
                "articles_checked": len(new_articles),
                "articles_new": 0,
                "reason": "No new articles to share",
            }

        # ── Step 2-4: Transform + Publish each article ──
        results = []
        for article in unseen[:max_per_cycle]:
            share_result = await self._share_article(article)
            results.append(share_result)

            # Track in history
            article_hash = self._article_hash(article)
            self._history[article_hash] = {
                "title": article.get("title", "")[:100],
                "blog": article.get("blog_handle", ""),
                "url": article.get("url", ""),
                "shared_at": datetime.now(timezone.utc).isoformat(),
                "tiktok_results": share_result.get("tiktok", []),
                "pinterest_result": share_result.get("pinterest"),
            }

        self._save_history(self._history)
        self._update_config("last_tick", datetime.now(timezone.utc).isoformat())
        self._update_config(
            "total_shared",
            self._config.get("total_shared", 0) + len(results),
        )

        # ── Step 5: Telegram alert ──
        success_count = sum(
            1 for r in results
            if any(t.get("success") for t in r.get("tiktok", []))
            or (r.get("pinterest", {}) or {}).get("success")
        )
        if success_count > 0:
            self._send_telegram_alert(results, unseen[:max_per_cycle])

        tick_result = {
            "success": True,
            "articles_checked": len(new_articles),
            "articles_new": len(unseen),
            "articles_shared": len(results),
            "success_count": success_count,
            "results": results,
        }
        logger.info("ShopifyAutoShare: tick done — %s", tick_result)
        return tick_result

    # ════════════════════════════════════════════
    # Share Single Article
    # ════════════════════════════════════════════

    async def _share_article(self, article: dict) -> dict:
        """Transform and share a single blog article to all platforms."""
        title = article.get("title", "")
        excerpt = article.get("excerpt", "")
        image = article.get("featured_image", "")
        url = article.get("url", "")
        tags = article.get("tags", [])
        blog = article.get("blog_handle", "")

        # Build hashtags from tags + blog handle
        hashtags = self._build_hashtags(tags, blog)

        result = {
            "article_id": article.get("article_id"),
            "title": title,
            "blog": blog,
            "tiktok": [],
            "pinterest": None,
        }

        # Download image to temp file (needed for Publer media upload)
        local_image = ""
        if image:
            local_image = await self._download_image(image)

        # ── TikTok ──
        tiktok_caption = self._build_tiktok_caption(title, excerpt, url)

        if self._publer and self._tiktok_via == "publer" and (image or local_image):
            # ── Route via Publer REST API bridge ──
            publer_content = {
                "caption": tiktok_caption,
                "hashtags": hashtags,
                "platforms": ["tiktok"],
            }
            if local_image:
                publer_content["media_path"] = local_image
            elif image:
                publer_content["media_url"] = image

            try:
                tr = await self._publer.publish(publer_content)
                tr["account"] = "publer"
                result["tiktok"] = [tr]
                if tr.get("success"):
                    logger.info(
                        "ShopifyAutoShare: TikTok (Publer) ✅ '%s'",
                        title[:50],
                    )
                else:
                    logger.warning(
                        "ShopifyAutoShare: TikTok (Publer) ❌ '%s': %s",
                        title[:50], tr.get("error"),
                    )
            except Exception as e:
                result["tiktok"] = [{"success": False, "error": str(e), "account": "publer"}]

        elif self._tiktok and self._tiktok.get_active_accounts() and image:
            # ── Fallback: direct TikTok API ──
            tiktok_content = {
                "image_urls": [image],
                "caption": tiktok_caption,
                "hashtags": hashtags,
            }

            tiktok_mode = os.environ.get(
                "SHOPIFY_AUTOSHARE_TIKTOK_MODE",
                self._config.get("tiktok_mode", DEFAULT_TIKTOK_MODE),
            )

            if tiktok_mode == "broadcast":
                tiktok_results = await self._tiktok.publish_to_all(
                    tiktok_content, mode="photo"
                )
            elif tiktok_mode == "round_robin":
                single = await self._tiktok.publish_round_robin(
                    tiktok_content, mode="photo"
                )
                tiktok_results = [single]
            else:
                target = self._config.get("tiktok_target_account", "1")
                single = await self._tiktok.publish_to_account(
                    target, tiktok_content, mode="photo"
                )
                tiktok_results = [single]

            result["tiktok"] = tiktok_results
            for tr in tiktok_results:
                if tr.get("success"):
                    logger.info(
                        "ShopifyAutoShare: TikTok (API) ✅ '%s' → %s",
                        title[:50], tr.get("account"),
                    )
                else:
                    logger.warning(
                        "ShopifyAutoShare: TikTok (API) ❌ '%s' → %s: %s",
                        title[:50], tr.get("account"), tr.get("error"),
                    )

        # ── Pinterest ──
        pinterest_enabled = os.environ.get(
            "SHOPIFY_AUTOSHARE_PINTEREST_ENABLED", "true"
        ).lower() == "true"

        if pinterest_enabled and self._publer and self._pinterest_via == "publer" and (image or local_image):
            # ── Route via Publer REST API ──
            pin_content = {
                "caption": self._build_pinterest_description(title, excerpt, tags),
                "hashtags": hashtags,
                "platforms": ["pinterest"],
            }
            if local_image:
                pin_content["media_path"] = local_image
            elif image:
                pin_content["media_url"] = image

            try:
                pin_result = await self._publer.publish(pin_content)
                result["pinterest"] = pin_result
                if pin_result.get("success"):
                    logger.info(
                        "ShopifyAutoShare: Pinterest (Publer) ✅ '%s'",
                        title[:50],
                    )
                else:
                    logger.warning(
                        "ShopifyAutoShare: Pinterest (Publer) ❌ '%s': %s",
                        title[:50], pin_result.get("error"),
                    )
            except Exception as e:
                result["pinterest"] = {"success": False, "error": str(e)}

        elif pinterest_enabled and self._pinterest and image:
            # ── Fallback: direct Pinterest API ──
            pinterest_content = {
                "title": title[:100],
                "caption": self._build_pinterest_description(title, excerpt, tags),
                "image_url": image,
                "link": url,
            }
            try:
                pin_result = await self._pinterest.publish(pinterest_content)
                result["pinterest"] = pin_result
                if pin_result.get("success"):
                    logger.info(
                        "ShopifyAutoShare: Pinterest (API) ✅ '%s' → pin %s",
                        title[:50], pin_result.get("post_id"),
                    )
                else:
                    logger.warning(
                        "ShopifyAutoShare: Pinterest (API) ❌ '%s': %s",
                        title[:50], pin_result.get("error"),
                    )
            except Exception as e:
                result["pinterest"] = {"success": False, "error": str(e)}

        # ── Cleanup temp image ──
        if local_image:
            try:
                os.unlink(local_image)
            except OSError:
                pass

        return result

    # ════════════════════════════════════════════
    # Manual Triggers
    # ════════════════════════════════════════════

    async def share_specific_article(
        self, article_url: str, force: bool = False
    ) -> dict:
        """Manually share a specific article by URL or handle.

        Args:
            article_url: Full URL or path containing /blogs/{handle}/{slug}.
            force: If True, share even if already shared before.
        """
        if not self._initialized:
            await self.initialize()
        if not self._watcher or not self._watcher._connected:
            return {"error": "Watcher not connected"}

        # Parse blog handle and article handle from URL
        # URL format: https://therike.com/blogs/{blog_handle}/{article_handle}
        import re
        match = re.search(r"/blogs/([^/]+)/([^/]+)", article_url)
        if not match:
            return {"error": f"Cannot parse blog URL: {article_url}"}

        blog_handle = match.group(1)
        articles = await self._watcher.get_recent_articles(blog_handle, limit=50)

        article_handle = match.group(2)
        target = None
        for a in articles:
            if a.get("handle") == article_handle:
                target = a
                break

        if not target:
            return {"error": f"Article not found: {article_handle}"}

        # ── Dedup check ──
        article_hash = self._article_hash(target)
        if article_hash in self._history and not force:
            prev = self._history[article_hash]
            return {
                "skipped": True,
                "reason": "already_shared",
                "title": prev.get("title", ""),
                "shared_at": prev.get("shared_at", ""),
                "hint": "Use force=True to re-share",
            }

        result = await self._share_article(target)
        # Record in history
        self._history[article_hash] = {
            "title": target.get("title", "")[:100],
            "blog": blog_handle,
            "url": target.get("url", ""),
            "shared_at": datetime.now(timezone.utc).isoformat(),
            "manual": True,
            "tiktok_results": result.get("tiktok", []),
            "pinterest_result": result.get("pinterest"),
        }
        self._save_history(self._history)
        return result

    async def share_latest(
        self, blog_handle: str, count: int = 1, force: bool = False
    ) -> list[dict]:
        """Share the N latest articles from a specific blog.

        Args:
            blog_handle: Blog handle (e.g. 'sustainable-living').
            count: Number of latest articles to share.
            force: If True, share even if already shared before.
        """
        if not self._initialized:
            await self.initialize()
        if not self._watcher:
            return [{"error": "Watcher not connected"}]

        articles = await self._watcher.get_recent_articles(blog_handle, limit=count)
        results = []
        for article in articles:
            article_hash = self._article_hash(article)

            # ── Dedup check ──
            if article_hash in self._history and not force:
                prev = self._history[article_hash]
                results.append({
                    "skipped": True,
                    "reason": "already_shared",
                    "title": prev.get("title", ""),
                    "shared_at": prev.get("shared_at", ""),
                })
                continue

            result = await self._share_article(article)
            # Record in history
            self._history[article_hash] = {
                "title": article.get("title", "")[:100],
                "blog": blog_handle,
                "url": article.get("url", ""),
                "shared_at": datetime.now(timezone.utc).isoformat(),
                "manual": True,
                "tiktok_results": result.get("tiktok", []),
                "pinterest_result": result.get("pinterest"),
            }
            results.append(result)

        if results:
            self._save_history(self._history)
        return results

    # ════════════════════════════════════════════
    # Image Download Helper
    # ════════════════════════════════════════════

    @staticmethod
    async def _download_image(url: str) -> str:
        """Download image from URL to temp file. Returns local path or '' on failure."""
        try:
            os.makedirs(MEDIA_CACHE_DIR, exist_ok=True)
            # Derive extension from URL
            ext = ".jpg"
            for candidate in (".png", ".webp", ".gif", ".jpeg"):
                if candidate in url.lower():
                    ext = candidate
                    break

            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            tmp = tempfile.NamedTemporaryFile(
                dir=MEDIA_CACHE_DIR, suffix=ext, delete=False,
            )
            tmp.write(resp.content)
            tmp.close()
            logger.debug("Downloaded image %s → %s (%d bytes)", url[:80], tmp.name, len(resp.content))
            return tmp.name
        except Exception as e:
            logger.warning("Failed to download image %s: %s", url[:80], e)
            return ""

    # ════════════════════════════════════════════
    # Content Transformation
    # ════════════════════════════════════════════

    @staticmethod
    def _build_tiktok_caption(
        title: str, excerpt: str, url: str
    ) -> str:
        """Build TikTok caption from blog article. No links (TikTok doesn't support clickable links)."""
        caption_parts = []
        if title:
            caption_parts.append(f"📖 {title}")
        if excerpt:
            # Truncate excerpt to fit TikTok's 2200 char limit
            max_excerpt = 800
            if len(excerpt) > max_excerpt:
                excerpt = excerpt[:max_excerpt].rsplit(" ", 1)[0] + "..."
            caption_parts.append(excerpt)
        caption_parts.append("🔗 Link in bio")
        return "\n\n".join(caption_parts)

    @staticmethod
    def _build_pinterest_description(
        title: str, excerpt: str, tags: list[str]
    ) -> str:
        """Build Pinterest pin description (max 500 chars)."""
        parts = []
        if excerpt:
            parts.append(excerpt[:300])
        if tags:
            tag_str = " ".join(f"#{t.replace(' ', '')}" for t in tags[:5])
            parts.append(tag_str)
        desc = "\n".join(parts)
        return desc[:500]

    @staticmethod
    def _build_hashtags(tags: list[str], blog_handle: str) -> list[str]:
        """Build hashtags from article tags + blog handle."""
        hashtags = []

        # From article tags
        for tag in tags[:5]:
            clean = tag.strip().replace(" ", "").replace("-", "")
            if clean:
                hashtags.append(f"#{clean}")

        # From blog handle
        handle_tag = blog_handle.replace("-", "")
        if handle_tag:
            hashtags.append(f"#{handle_tag}")

        # Always add brand tag
        hashtags.append("#TheRike")

        return hashtags[:10]

    # ════════════════════════════════════════════
    # Config Management
    # ════════════════════════════════════════════

    def get_config(self) -> dict:
        """Return current auto-share configuration."""
        return {
            "paused": self._config.get("paused", False),
            "interval_min": int(os.environ.get(
                "SHOPIFY_AUTOSHARE_INTERVAL_MIN",
                self._config.get("interval_min", DEFAULT_INTERVAL_MIN),
            )),
            "max_per_cycle": int(os.environ.get(
                "SHOPIFY_AUTOSHARE_MAX_PER_CYCLE",
                self._config.get("max_per_cycle", DEFAULT_MAX_PER_CYCLE),
            )),
            "tiktok_via": self._tiktok_via,
            "tiktok_mode": os.environ.get(
                "SHOPIFY_AUTOSHARE_TIKTOK_MODE",
                self._config.get("tiktok_mode", DEFAULT_TIKTOK_MODE),
            ),
            "pinterest_via": self._pinterest_via,
            "pinterest_enabled": os.environ.get(
                "SHOPIFY_AUTOSHARE_PINTEREST_ENABLED", "true"
            ).lower() == "true",
            "publer_connected": self._publer is not None,
            "total_shared": self._config.get("total_shared", 0),
            "last_tick": self._config.get("last_tick", ""),
            "watched_blogs": self._watcher.get_watched_blogs() if self._watcher else {},
            "tiktok_status": self._tiktok.status() if self._tiktok else {},
        }

    def update_config(self, updates: dict) -> dict:
        """Update auto-share configuration."""
        allowed = {
            "paused", "interval_min", "max_per_cycle",
            "tiktok_via", "tiktok_mode", "tiktok_target_account",
            "pinterest_via",
        }
        for key, value in updates.items():
            if key in allowed:
                self._config[key] = value
        self._save_config(self._config)
        return {"success": True, "config": self.get_config()}

    def pause(self) -> dict:
        return self.update_config({"paused": True})

    def resume(self) -> dict:
        return self.update_config({"paused": False})

    # ════════════════════════════════════════════
    # History
    # ════════════════════════════════════════════

    def get_history(self, limit: int = 50) -> list[dict]:
        """Return share history, newest first."""
        items = []
        for hash_key, data in self._history.items():
            items.append({"hash": hash_key, **data})
        items.sort(key=lambda x: x.get("shared_at", ""), reverse=True)
        return items[:limit]

    def clear_history(self) -> dict:
        """Clear all share history (allows re-sharing)."""
        count = len(self._history)
        self._history = {}
        self._save_history(self._history)
        return {"cleared": count}

    def get_stats(self) -> dict:
        """Return sharing statistics."""
        total = len(self._history)
        by_blog = {}
        tiktok_success = 0
        pinterest_success = 0
        for data in self._history.values():
            blog = data.get("blog", "unknown")
            by_blog[blog] = by_blog.get(blog, 0) + 1
            for tr in data.get("tiktok_results", []):
                if isinstance(tr, dict) and tr.get("success"):
                    tiktok_success += 1
            pr = data.get("pinterest_result")
            if isinstance(pr, dict) and pr.get("success"):
                pinterest_success += 1

        return {
            "total_articles_shared": total,
            "by_blog": by_blog,
            "tiktok_success": tiktok_success,
            "pinterest_success": pinterest_success,
            "config": self.get_config(),
        }

    # ════════════════════════════════════════════
    # Telegram Alert
    # ════════════════════════════════════════════

    def _send_telegram_alert(
        self, results: list[dict], articles: list[dict]
    ):
        """Send Telegram notification about shared articles."""
        try:
            from integrations.telegram_bot import alert_blog_shared

            alert_blog_shared(results, articles)
        except Exception as e:
            logger.warning("ShopifyAutoShare: Telegram alert failed: %s", e)

    # ════════════════════════════════════════════
    # Persistence
    # ════════════════════════════════════════════

    @staticmethod
    def _article_hash(article: dict) -> str:
        """Generate unique hash for an article to track sharing."""
        key = f"{article.get('blog_handle', '')}:{article.get('article_id', '')}:{article.get('title', '')}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    @staticmethod
    def _load_history() -> dict:
        try:
            if os.path.exists(SHARE_HISTORY_FILE):
                with open(SHARE_HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    @staticmethod
    def _save_history(history: dict):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(SHARE_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _load_config() -> dict:
        try:
            if os.path.exists(SHARE_CONFIG_FILE):
                with open(SHARE_CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    @staticmethod
    def _save_config(config: dict):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(SHARE_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _update_config(self, key: str, value: Any):
        self._config[key] = value
        self._save_config(self._config)

    # ════════════════════════════════════════════
    # Cleanup
    # ════════════════════════════════════════════

    async def close(self):
        """Close connections."""
        if self._watcher:
            await self._watcher.close()
        if self._publer:
            try:
                await self._publer.close()
            except Exception:
                pass


# ════════════════════════════════════════════════════════════════
# Module-level convenience functions (for web/app.py endpoints)
# ════════════════════════════════════════════════════════════════

_instance: Optional[ShopifyAutoShare] = None


async def get_auto_share() -> ShopifyAutoShare:
    """Get or create singleton instance."""
    global _instance
    if _instance is None:
        _instance = ShopifyAutoShare()
        await _instance.initialize()
    return _instance


async def auto_share_tick() -> dict:
    """Run one auto-share cycle."""
    instance = await get_auto_share()
    return await instance.tick()


async def auto_share_status() -> dict:
    """Get auto-share status + stats."""
    instance = await get_auto_share()
    return instance.get_stats()


async def auto_share_history(limit: int = 50) -> list[dict]:
    """Get share history."""
    instance = await get_auto_share()
    return instance.get_history(limit)


async def auto_share_config() -> dict:
    """Get current config."""
    instance = await get_auto_share()
    return instance.get_config()


async def auto_share_update_config(updates: dict) -> dict:
    """Update config."""
    instance = await get_auto_share()
    return instance.update_config(updates)


async def auto_share_pause() -> dict:
    """Pause auto-sharing."""
    instance = await get_auto_share()
    return instance.pause()


async def auto_share_resume() -> dict:
    """Resume auto-sharing."""
    instance = await get_auto_share()
    return instance.resume()


async def auto_share_manual(article_url: str, force: bool = False) -> dict:
    """Manually share a specific article."""
    instance = await get_auto_share()
    return await instance.share_specific_article(article_url, force=force)


async def auto_share_latest(
    blog_handle: str, count: int = 1, force: bool = False
) -> list[dict]:
    """Share latest articles from a blog."""
    instance = await get_auto_share()
    return await instance.share_latest(blog_handle, count, force=force)
