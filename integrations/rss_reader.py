"""
RSS Feed Reader — Import blog content for repurposing.
Supports full content + excerpt modes (like SocialBee).
Uses feedparser library.
"""
import os
import json
import hashlib
import re
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

import structlog

logger = structlog.get_logger()

# ── Persistent storage for RSS feeds ──
RSS_DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rss_feeds.json")


@dataclass
class RSSFeed:
    """RSS feed configuration."""
    id: str
    url: str
    name: str
    category: str = "general"
    import_mode: str = "full"  # "full" or "excerpt"
    auto_publish: bool = False
    target_platforms: list = field(default_factory=list)
    last_fetched: Optional[str] = None
    last_entry_id: Optional[str] = None
    active: bool = True


@dataclass
class RSSEntry:
    """Parsed RSS entry."""
    id: str
    feed_id: str
    title: str
    body: str
    excerpt: str
    url: str
    author: str
    published: str
    image_url: str = ""
    tags: list = field(default_factory=list)
    imported_at: str = ""


def _load_feeds() -> list[dict]:
    """Load saved feeds."""
    try:
        if os.path.exists(RSS_DATA_FILE):
            with open(RSS_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_feeds(feeds: list[dict]):
    """Save feeds to file."""
    os.makedirs(os.path.dirname(RSS_DATA_FILE), exist_ok=True)
    with open(RSS_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(feeds, f, indent=2, ensure_ascii=False)


def add_feed(url: str, name: str = "", category: str = "general",
             import_mode: str = "full", target_platforms: list = None) -> dict:
    """Add a new RSS feed."""
    feed_id = hashlib.md5(url.encode()).hexdigest()[:12]
    feeds = _load_feeds()

    # Check duplicate
    for f in feeds:
        if f.get("url") == url:
            return {"error": "Feed already exists", "feed": f}

    feed = {
        "id": feed_id,
        "url": url,
        "name": name or url,
        "category": category,
        "import_mode": import_mode,
        "auto_publish": False,
        "target_platforms": target_platforms or [],
        "last_fetched": None,
        "last_entry_id": None,
        "active": True,
    }
    feeds.append(feed)
    _save_feeds(feeds)
    logger.info("rss.feed_added", url=url, id=feed_id)
    return {"success": True, "feed": feed}


def remove_feed(feed_id: str) -> dict:
    """Remove a feed."""
    feeds = _load_feeds()
    feeds = [f for f in feeds if f.get("id") != feed_id]
    _save_feeds(feeds)
    return {"success": True}


def list_feeds() -> list[dict]:
    """List all configured feeds."""
    return _load_feeds()


def update_feed(feed_id: str, **kwargs) -> dict:
    """Update feed settings."""
    feeds = _load_feeds()
    for f in feeds:
        if f.get("id") == feed_id:
            f.update(kwargs)
            _save_feeds(feeds)
            return {"success": True, "feed": f}
    return {"error": "Feed not found"}


def _strip_html(html: str) -> str:
    """Strip HTML tags, keeping text content."""
    # Remove script/style blocks
    html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Remove tags
    text = re.sub(r'<[^>]+>', ' ', html)
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _make_excerpt(text: str, max_length: int = 300) -> str:
    """Create an excerpt from full text."""
    if len(text) <= max_length:
        return text
    # Cut at last word boundary
    excerpt = text[:max_length]
    last_space = excerpt.rfind(' ')
    if last_space > max_length * 0.7:
        excerpt = excerpt[:last_space]
    return excerpt.rstrip('.') + '...'


def fetch_feed(feed_id: str = None, feed_url: str = None, max_entries: int = 10) -> dict:
    """
    Fetch and parse an RSS feed.
    Returns entries with both full content and excerpt.
    """
    try:
        import feedparser
    except ImportError:
        return {"error": "feedparser not installed. Run: pip install feedparser"}

    feeds = _load_feeds()

    # Find feed config
    feed_config = None
    if feed_id:
        for f in feeds:
            if f.get("id") == feed_id:
                feed_config = f
                break
    elif feed_url:
        feed_config = {"url": feed_url, "id": "preview", "import_mode": "full"}

    if not feed_config:
        return {"error": "Feed not found"}

    url = feed_config["url"]
    import_mode = feed_config.get("import_mode", "full")

    logger.info("rss.fetch_start", url=url, mode=import_mode)

    try:
        parsed = feedparser.parse(url)

        if parsed.bozo and not parsed.entries:
            return {"error": f"Failed to parse feed: {parsed.bozo_exception}"}

        entries = []
        for entry in parsed.entries[:max_entries]:
            # Get full content
            content_html = ""
            if hasattr(entry, 'content') and entry.content:
                content_html = entry.content[0].get('value', '')
            elif hasattr(entry, 'summary'):
                content_html = entry.summary or ""

            # Strip HTML for plain text
            full_text = _strip_html(content_html)
            excerpt = _make_excerpt(full_text)

            # Get image
            image_url = ""
            if hasattr(entry, 'media_content') and entry.media_content:
                image_url = entry.media_content[0].get('url', '')
            elif hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
                image_url = entry.media_thumbnail[0].get('url', '')

            # Get published date
            published = ""
            if hasattr(entry, 'published'):
                published = entry.published
            elif hasattr(entry, 'updated'):
                published = entry.updated

            # Tags
            tags = []
            if hasattr(entry, 'tags') and entry.tags:
                tags = [t.get('term', '') for t in entry.tags if t.get('term')]

            entry_id = hashlib.md5((entry.get('link', '') + entry.get('title', '')).encode()).hexdigest()[:12]

            entries.append({
                "id": entry_id,
                "feed_id": feed_config["id"],
                "title": entry.get("title", "Untitled"),
                "body": full_text if import_mode == "full" else excerpt,
                "body_html": content_html if import_mode == "full" else "",
                "excerpt": excerpt,
                "url": entry.get("link", ""),
                "author": entry.get("author", ""),
                "published": published,
                "image_url": image_url,
                "tags": tags,
                "imported_at": datetime.utcnow().isoformat(),
            })

        # Update last fetched
        if feed_id and entries:
            for f in feeds:
                if f.get("id") == feed_id:
                    f["last_fetched"] = datetime.utcnow().isoformat()
                    f["last_entry_id"] = entries[0]["id"]
                    break
            _save_feeds(feeds)

        result = {
            "success": True,
            "feed_title": parsed.feed.get("title", ""),
            "feed_description": parsed.feed.get("description", ""),
            "entry_count": len(entries),
            "entries": entries,
        }
        logger.info("rss.fetch_done", url=url, entries=len(entries))
        return result

    except Exception as e:
        logger.error("rss.fetch_error", url=url, error=str(e))
        return {"error": str(e)}


def import_entry_as_draft(entry: dict, platforms: list = None) -> dict:
    """
    Convert an RSS entry into a draft post ready for the content pipeline.
    Returns a dict compatible with the web app's post creation.
    """
    return {
        "title": entry.get("title", ""),
        "body": entry.get("body", ""),
        "category": "curated",
        "platforms": json.dumps(platforms or ["medium", "shopify_blog"]),
        "status": "draft",
        "extra_fields": json.dumps({
            "source_url": entry.get("url", ""),
            "source_author": entry.get("author", ""),
            "original_tags": entry.get("tags", []),
            "image_url": entry.get("image_url", ""),
            "import_mode": "rss",
        }),
    }
