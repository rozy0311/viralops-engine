"""
Platform Publisher Registry — EMADS-PR v1.0
Clean version: Only real publishers, no stubs.

Real publishers:
- Reddit (OAuth2) → reddit_publisher.py
- Medium (Bearer) → medium_publisher.py
- Tumblr (OAuth2 NPF) → tumblr_publisher.py
- Shopify Blog (Admin REST) → shopify_blog_publisher.py

SocialBee handles: TikTok, Instagram, Facebook, YouTube, Pinterest,
LinkedIn, Twitter/X, Threads, Bluesky, Google Business
"""
import structlog
from typing import Optional
from core.models import PublishResult

logger = structlog.get_logger()


class PublisherRegistry:
    """
    Registry of all available publishers.
    Only contains REAL, working publishers.
    """

    def __init__(self):
        self._publishers = {}
        self._initialized = False

    def register_all(self):
        """Register all real publishers."""
        if self._initialized:
            return
        self._initialized = True

        # ── Real publishers (direct API integration) ──
        try:
            from integrations.reddit_publisher import RealRedditPublisher
            self._publishers["reddit"] = RealRedditPublisher()
            logger.info("registry.loaded", platform="reddit")
        except Exception as e:
            logger.warning("registry.skip", platform="reddit", error=str(e))

        try:
            from integrations.medium_publisher import RealMediumPublisher
            self._publishers["medium"] = RealMediumPublisher()
            logger.info("registry.loaded", platform="medium")
        except Exception as e:
            logger.warning("registry.skip", platform="medium", error=str(e))

        try:
            from integrations.tumblr_publisher import RealTumblrPublisher
            self._publishers["tumblr"] = RealTumblrPublisher()
            logger.info("registry.loaded", platform="tumblr")
        except Exception as e:
            logger.warning("registry.skip", platform="tumblr", error=str(e))

        try:
            from integrations.shopify_blog_publisher import ShopifyBlogPublisher
            self._publishers["shopify_blog"] = ShopifyBlogPublisher()
            logger.info("registry.loaded", platform="shopify_blog")
        except Exception as e:
            logger.warning("registry.skip", platform="shopify_blog", error=str(e))

        # ── SocialBee-managed platforms (no direct publisher needed) ──
        socialbee_platforms = [
            "tiktok", "instagram", "facebook", "youtube", "pinterest",
            "linkedin", "twitter", "threads", "bluesky", "google_business"
        ]
        for p in socialbee_platforms:
            self._publishers[p] = _SocialBeeProxy(p)

        logger.info("registry.complete",
                    real=len([p for p in self._publishers.values() if not isinstance(p, _SocialBeeProxy)]),
                    socialbee=len(socialbee_platforms))

    def get(self, platform: str):
        """Get publisher for a platform."""
        if not self._initialized:
            self.register_all()
        return self._publishers.get(platform)

    def list_platforms(self) -> dict:
        """List all available platforms and their type."""
        if not self._initialized:
            self.register_all()
        return {
            name: "socialbee" if isinstance(pub, _SocialBeeProxy) else "direct"
            for name, pub in self._publishers.items()
        }

    async def publish(self, platform: str, item) -> PublishResult:
        """Publish to a specific platform."""
        pub = self.get(platform)
        if not pub:
            return PublishResult(
                queue_item_id=getattr(item, 'id', ''),
                platform=platform,
                success=False,
                error=f"No publisher for {platform}"
            )
        return await pub.publish(item)


class _SocialBeeProxy:
    """
    Proxy for platforms managed by SocialBee.
    Returns info message — actual posting happens through SocialBee UI/API.
    """

    def __init__(self, platform: str):
        self.platform = platform

    async def publish(self, item) -> PublishResult:
        return PublishResult(
            queue_item_id=getattr(item, 'id', ''),
            platform=self.platform,
            success=False,
            error=f"{self.platform} is managed by SocialBee — use SocialBee dashboard to publish"
        )

    async def test_connection(self) -> bool:
        return False  # Managed externally


# ── Singleton ──
_registry: Optional[PublisherRegistry] = None


def get_registry() -> PublisherRegistry:
    global _registry
    if _registry is None:
        _registry = PublisherRegistry()
        _registry.register_all()
    return _registry
