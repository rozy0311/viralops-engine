"""
Platform Publisher Registry — EMADS-PR v1.0
v2.1: Added social_connectors for 7 major platforms.

Real publishers (direct API):
- Reddit (OAuth2) → reddit_publisher.py
- Medium (Bearer) → medium_publisher.py
- Tumblr (OAuth2 NPF) → tumblr_publisher.py
- Shopify Blog (Admin REST) → shopify_blog_publisher.py
- Lemon8 → lemon8_publisher.py

Social connectors (via social_connectors.py):
- Twitter/X (v2 API)
- Instagram (Graph API)
- Facebook Page (Graph API)
- YouTube (Data API v3)
- LinkedIn (v2 API)
- TikTok (Content Posting API)
- Pinterest (v5 API)
"""
import structlog
from typing import Optional
from core.models import PublishResult

logger = structlog.get_logger()


class PublisherRegistry:
    """
    Registry of all available publishers.
    v2.1: Includes social_connectors for TikTok/IG/FB/YT/LI/TW/PIN.
    """

    def __init__(self):
        self._publishers = {}
        self._initialized = False

    def register_all(self):
        """Register all real publishers."""
        if self._initialized:
            return
        self._initialized = True

        # ── Social connectors (7 major platforms) ──
        try:
            from integrations.social_connectors import PUBLISHERS as SOCIAL_PUBLISHERS
            for platform_name, cls in SOCIAL_PUBLISHERS.items():
                try:
                    self._publishers[platform_name] = cls()
                    logger.info("registry.loaded_social", platform=platform_name)
                except Exception as e:
                    logger.warning("registry.skip_social", platform=platform_name, error=str(e))
        except ImportError:
            logger.warning("registry.social_connectors_not_available")

        # ── Direct API publishers ──
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

        try:
            from integrations.lemon8_publisher import Lemon8Publisher
            self._publishers["lemon8"] = Lemon8Publisher()
            logger.info("registry.loaded", platform="lemon8")
        except Exception as e:
            logger.warning("registry.skip", platform="lemon8", error=str(e))

        logger.info("registry.complete", total=len(self._publishers))

    def get(self, platform: str):
        """Get publisher for a platform."""
        if not self._initialized:
            self.register_all()
        return self._publishers.get(platform)

    def list_platforms(self) -> dict:
        """List all available platforms and their type."""
        if not self._initialized:
            self.register_all()
        result = {}
        for name, pub in self._publishers.items():
            module = type(pub).__module__
            if "social_connectors" in module:
                result[name] = "social_connector"
            else:
                result[name] = "direct_api"
        return result

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


# ── Singleton ──
_registry: Optional[PublisherRegistry] = None


def get_registry() -> PublisherRegistry:
    global _registry
    if _registry is None:
        _registry = PublisherRegistry()
        _registry.register_all()
    return _registry
