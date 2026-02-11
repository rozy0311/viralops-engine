"""
RSS Feed Reader — Import blog content for repurposing.
Supports full content + excerpt modes (like SocialBee/Sendible).
Uses feedparser library.

Features (v2.0 — Sendible-killer):
  - Bulk import up to 500 entries per fetch
  - Pre-configured Railway RSS server for TheRike
  - Batch import → SQLite drafts in one call
  - Smart dedup (skip already-imported entries)
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
IMPORTED_IDS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rss_imported_ids.json")

# ── Railway RSS Server (self-hosted full-text feed for TheRike) ──
RAILWAY_RSS_BASE = os.environ.get(
    "RAILWAY_RSS_URL",
    "https://therike-rss-feed-production.up.railway.app"
)

# Pre-configured TheRike blog handles
THERIKE_BLOG_HANDLES = [
    "sustainable-living",
    "home-stead",
    "natural-healing-herbal-remedy-insights-and-solutions",
    "how-to-diy",
    "the-art-of-healing",
    "agritourism-adventures-exploring-farm-based-tourism",
    "permaculture",
    "meditation",
    "farm-destinations-the-beauty-of-rural-escapes",
    "brand-partnerships",
]

# ── Bulk import settings ──
MAX_BULK_ENTRIES = 500  # Sendible does 200-500/day, we match 500


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


def fetch_feed(feed_id: str = None, feed_url: str = None, max_entries: int = 50) -> dict:
    """
    Fetch and parse an RSS feed.
    Returns entries with both full content and excerpt.
    Supports up to MAX_BULK_ENTRIES (500) for bulk operations.
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


# ════════════════════════════════════════════════════════════════
# Bulk Import — Sendible-killer (200-500 posts/day)
# ════════════════════════════════════════════════════════════════

def _load_imported_ids() -> set:
    """Load set of already-imported entry IDs for dedup."""
    try:
        if os.path.exists(IMPORTED_IDS_FILE):
            with open(IMPORTED_IDS_FILE, 'r') as f:
                return set(json.load(f))
    except Exception:
        pass
    return set()


def _save_imported_ids(ids: set):
    """Persist imported entry IDs."""
    os.makedirs(os.path.dirname(IMPORTED_IDS_FILE), exist_ok=True)
    with open(IMPORTED_IDS_FILE, 'w') as f:
        json.dump(list(ids), f)


def bulk_fetch_feed(feed_id: str = None, feed_url: str = None,
                    max_entries: int = MAX_BULK_ENTRIES,
                    skip_imported: bool = True) -> dict:
    """
    Bulk fetch up to 500 entries from an RSS feed.
    Skips already-imported entries (dedup).

    Like Sendible's bulk import but with:
      - Smart dedup (won't re-import same articles)
      - Full-text content (not just excerpts)
      - 7-layer hashtag extraction from tags
    """
    result = fetch_feed(feed_id=feed_id, feed_url=feed_url, max_entries=max_entries)

    if not result.get("success"):
        return result

    entries = result.get("entries", [])
    imported_ids = _load_imported_ids() if skip_imported else set()

    # Filter out already-imported
    new_entries = []
    skipped = 0
    for entry in entries:
        entry_key = entry.get("url", "") or entry.get("id", "")
        entry_hash = hashlib.md5(entry_key.encode()).hexdigest()[:16]
        if entry_hash in imported_ids:
            skipped += 1
        else:
            new_entries.append(entry)

    result["entries"] = new_entries
    result["entry_count"] = len(new_entries)
    result["skipped_duplicates"] = skipped
    result["total_available"] = len(entries)
    result["bulk_mode"] = True

    logger.info("rss.bulk_fetch",
                total=len(entries), new=len(new_entries),
                skipped=skipped, max=max_entries)
    return result


def bulk_import_as_drafts(entries: list[dict],
                          platforms: list = None,
                          db_path: str = None) -> dict:
    """
    Bulk import RSS entries as draft posts into SQLite.

    This is the Sendible-killer feature:
      - Import 200-500 entries in one call
      - All saved as drafts (no auto-publish without review)
      - Dedup tracking prevents re-import
      - Returns count of new vs skipped

    Args:
        entries: List of RSS entry dicts from fetch_feed/bulk_fetch_feed
        platforms: Target platforms for all entries
        db_path: Path to SQLite database (auto-detected if None)
    """
    import sqlite3

    if not entries:
        return {"success": True, "imported": 0, "message": "No entries to import"}

    if db_path is None:
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "viralops.db"
        )

    imported_ids = _load_imported_ids()
    new_count = 0
    skipped = 0

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Ensure table exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT, body TEXT, category TEXT,
                platforms TEXT, status TEXT DEFAULT 'draft',
                extra_fields TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        for entry in entries:
            entry_key = entry.get("url", "") or entry.get("id", "")
            entry_hash = hashlib.md5(entry_key.encode()).hexdigest()[:16]

            if entry_hash in imported_ids:
                skipped += 1
                continue

            draft = import_entry_as_draft(entry, platforms)
            conn.execute(
                "INSERT INTO posts (title, body, category, platforms, status, extra_fields) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (draft["title"], draft["body"], draft["category"],
                 draft["platforms"], draft["status"], draft["extra_fields"])
            )
            imported_ids.add(entry_hash)
            new_count += 1

        conn.commit()
        conn.close()

        # Save dedup state
        _save_imported_ids(imported_ids)

        logger.info("rss.bulk_import_done",
                     imported=new_count, skipped=skipped, total=len(entries))
        return {
            "success": True,
            "imported": new_count,
            "skipped_duplicates": skipped,
            "total_processed": len(entries),
        }

    except Exception as e:
        logger.error("rss.bulk_import_error", error=str(e))
        return {"error": str(e)}


# ════════════════════════════════════════════════════════════════
# Railway RSS Server Integration
# ════════════════════════════════════════════════════════════════

def get_railway_rss_url(blog_handle: str) -> str:
    """Get the full RSS URL for a TheRike blog handle via Railway server."""
    return f"{RAILWAY_RSS_BASE}/rss/{blog_handle}"


def list_therike_blogs() -> list[dict]:
    """List all available TheRike blog handles with their Railway RSS URLs."""
    return [
        {
            "handle": handle,
            "name": handle.replace("-", " ").title(),
            "rss_url": get_railway_rss_url(handle),
        }
        for handle in THERIKE_BLOG_HANDLES
    ]


def setup_therike_feeds(target_platforms: list = None) -> dict:
    """
    One-click setup: register ALL TheRike blogs as RSS feeds.
    Uses the Railway self-hosted full-text RSS server.

    This is equivalent to Sendible → Content → RSS feeds → Add Railway RSS.
    """
    results = []
    for handle in THERIKE_BLOG_HANDLES:
        url = get_railway_rss_url(handle)
        name = f"TheRike — {handle.replace('-', ' ').title()}"
        result = add_feed(
            url=url,
            name=name,
            category="therike",
            import_mode="full",
            target_platforms=target_platforms or ["medium", "shopify_blog", "reddit"],
        )
        results.append({
            "handle": handle,
            "url": url,
            "status": "added" if result.get("success") else "exists",
        })

    added = sum(1 for r in results if r["status"] == "added")
    logger.info("rss.therike_setup", added=added, total=len(results))
    return {
        "success": True,
        "feeds_added": added,
        "feeds_existing": len(results) - added,
        "feeds": results,
    }


def bulk_import_all_therike(platforms: list = None,
                             max_per_blog: int = 50) -> dict:
    """
    Bulk import ALL TheRike blogs in one call.
    Fetches from Railway RSS server, imports as drafts.

    Sendible does ~200-500/day. This can do 500 in one call.
    """
    total_imported = 0
    total_skipped = 0
    blog_results = []

    for handle in THERIKE_BLOG_HANDLES:
        url = get_railway_rss_url(handle)
        fetch_result = bulk_fetch_feed(
            feed_url=url,
            max_entries=max_per_blog,
            skip_imported=True,
        )

        if not fetch_result.get("success"):
            blog_results.append({
                "handle": handle,
                "error": fetch_result.get("error", "Unknown error"),
                "imported": 0,
            })
            continue

        entries = fetch_result.get("entries", [])
        if entries:
            import_result = bulk_import_as_drafts(entries, platforms=platforms)
            imported = import_result.get("imported", 0)
            skipped = import_result.get("skipped_duplicates", 0)
        else:
            imported = 0
            skipped = fetch_result.get("skipped_duplicates", 0)

        total_imported += imported
        total_skipped += skipped

        blog_results.append({
            "handle": handle,
            "imported": imported,
            "skipped": skipped,
        })

    logger.info("rss.therike_bulk_done",
                imported=total_imported, skipped=total_skipped,
                blogs=len(THERIKE_BLOG_HANDLES))
    return {
        "success": True,
        "total_imported": total_imported,
        "total_skipped": total_skipped,
        "blogs_processed": len(THERIKE_BLOG_HANDLES),
        "details": blog_results,
    }
