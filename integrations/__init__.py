"""
ViralOps Engine — Integrations Package

REAL publishers (actual HTTP API calls):
  - RedditPublisher      → Reddit OAuth2 API
  - MediumPublisher      → Medium REST API v1
  - TumblrPublisher      → Tumblr API v2 (NPF)
  - ShopifyBlogPublisher → Shopify Admin REST API
  - Lemon8Publisher      → Draft + webhook (mobile-only, no official API)

SocialBee manages: TikTok, Instagram, Facebook, YouTube, Pinterest,
                   LinkedIn, Twitter/X, Threads, Bluesky, Google Business

RSS: feedparser-based content import (rss_reader module)
Publisher Registry: Central routing for all platforms
"""

# Real publishers
from integrations.reddit_publisher import RedditPublisher
from integrations.medium_publisher import MediumPublisher
from integrations.tumblr_publisher import TumblrPublisher
from integrations.shopify_blog_publisher import ShopifyBlogPublisher
from integrations.lemon8_publisher import Lemon8Publisher

# Registry (routes to real publishers + SocialBee proxy)
from integrations.platform_publisher import PublisherRegistry, get_registry

# RSS reader (module-level functions, no class)
from integrations.rss_reader import (
    RSSFeed, RSSEntry, add_feed, remove_feed,
    list_feeds, fetch_feed, import_entry_as_draft,
)

# Research
from integrations.trend_researcher import TrendResearcher

__all__ = [
    "RedditPublisher",
    "MediumPublisher",
    "TumblrPublisher",
    "ShopifyBlogPublisher",
    "Lemon8Publisher",
    "PublisherRegistry",
    "get_registry",
    "RSSFeed",
    "RSSEntry",
    "add_feed",
    "remove_feed",
    "list_feeds",
    "fetch_feed",
    "import_entry_as_draft",
    "TrendResearcher",
]
