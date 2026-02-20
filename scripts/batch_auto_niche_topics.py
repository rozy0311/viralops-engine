"""ViralOps — Auto Niche Topics Batch Publisher

Goal
----
Publish a few posts using auto-picked topics from niche_hunter.db (unused, high score)
so we don't reuse a hardcoded TOPICS list.

By default this runs in DRY-RUN mode and prints the selected topics.

Usage
-----
  # Preview 3 topics (no publish)
  python scripts/batch_auto_niche_topics.py

  # Publish 3 topics × 4 accounts = 12 posts
  python scripts/batch_auto_niche_topics.py --publish

  # Publish 5 topics
  python scripts/batch_auto_niche_topics.py --publish --limit 5

  # Refresh niche_hunter.db first (local scoring only, no Gemini)
  python scripts/batch_auto_niche_topics.py --refresh-db local

  # Refresh niche_hunter.db using Gemini expansion (if keys available)
  python scripts/batch_auto_niche_topics.py --refresh-db full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import uuid
import sqlite3
import sys
import time
from datetime import datetime, timezone
from datetime import timedelta
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(override=True)

TIKTOK_1 = "698c95e5b1ab790def1352c1"   # The Rike Root Stories
TIKTOK_2 = "69951ea30c4677f27c12d98c"   # The Rike Stories
# Legacy IDs kept for backwards compatibility; prefer Publer discovery.
FACEBOOK = "699522978f7449fba2dceebe"    # The Rike (Facebook)
PINTEREST = "69951f098f7449fba2dceadd"   # therike (Pinterest)


async def _get_tiktok_accounts() -> list[tuple[str, str]]:
    """Return list of (account_id, label) for TikTok.

    Priority:
      1) Env `VIRALOPS_TIKTOK_ACCOUNT_IDS` (comma-separated Publer account IDs)
      2) Publer discovery: all connected TikTok accounts in workspace
      3) Fallback to the two historical hardcoded IDs if present
    """
    env_ids = os.environ.get("VIRALOPS_TIKTOK_ACCOUNT_IDS", "").strip()
    if env_ids:
        ids = [x.strip() for x in env_ids.split(",") if x.strip()]
        return [(aid, f"tiktok_{i+1}") for i, aid in enumerate(ids)]

    # Discover from Publer (best for scaling: auto-updates when you connect more accounts)
    try:
        from integrations.publer_publisher import PublerPublisher

        pub = PublerPublisher()
        await pub.connect()
        accounts = await pub.get_accounts()
        out: list[tuple[str, str]] = []
        for acc in accounts or []:
            acc_type = str(acc.get("type", acc.get("provider", "")) or "").lower()
            if "tiktok" not in acc_type:
                continue
            acc_id = str(acc.get("id", acc.get("_id", "")) or "").strip()
            if not acc_id:
                continue
            name = str(acc.get("name", "") or "").strip()
            label = name or acc_id[-6:]
            out.append((acc_id, label))
        if out:
            return out
    except Exception:
        pass

    # Final fallback: keep legacy behavior
    legacy: list[tuple[str, str]] = []
    if TIKTOK_1:
        legacy.append((TIKTOK_1, "RikeRoot"))
    if TIKTOK_2:
        legacy.append((TIKTOK_2, "RikeStories"))
    return legacy


def _log_published_to_viralops_db(
    pack: dict[str, Any],
    platforms: list[str],
    extra: dict[str, Any] | None = None,
    *,
    status: str = "published",
) -> None:
    """Persist a post attempt to web/viralops.db so future runs can dedup/debug."""
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(proj_root, "web", "viralops.db")
    if not os.path.exists(db_path):
        return

    title = str(pack.get("title", "") or "").strip()
    body = str(pack.get("universal_caption_block", "") or pack.get("content_formatted", "") or "").strip()
    if not title:
        return

    payload = dict(extra or {})
    payload.update({
        "_source": pack.get("_source", ""),
        "_niche_score": pack.get("_niche_score", None),
        "_review_score": pack.get("_review_score", None),
        "_gen_provider": pack.get("_gen_provider", ""),
        "_gen_model": pack.get("_gen_model", ""),
    })

    status = (status or payload.get("status") or "published").strip().lower()

    published_at = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO posts (title, body, platforms, status, published_at, extra_fields) VALUES (?, ?, ?, ?, ?, ?)",
            (
                title,
                body,
                json.dumps(platforms),
                status,
                published_at,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _ensure_topic_history_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS topic_history (
            topic TEXT PRIMARY KEY,
            source TEXT,
            created_at TEXT
        )
        """
    )


def _topic_already_used(topic: str) -> bool:
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(proj_root, "web", "viralops.db")
    if not os.path.exists(db_path):
        return False
    t = (topic or "").strip()
    if not t:
        return True
    conn = sqlite3.connect(db_path)
    try:
        _ensure_topic_history_table(conn)
        row = conn.execute("SELECT 1 FROM topic_history WHERE topic = ?", (t,)).fetchone()
        return bool(row)
    finally:
        conn.close()


def _mark_topic_used(topic: str, source: str) -> None:
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(proj_root, "web", "viralops.db")
    if not os.path.exists(db_path):
        return
    t = (topic or "").strip()
    if not t:
        return
    conn = sqlite3.connect(db_path)
    try:
        _ensure_topic_history_table(conn)
        conn.execute(
            "INSERT OR IGNORE INTO topic_history (topic, source, created_at) VALUES (?, ?, ?)",
            (t, source or "", datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def _refresh_db(mode: str) -> None:
    """Refresh niche_hunter.db by running niche_hunter workflow."""
    from niche_hunter import run_niche_hunter

    if mode == "local":
        run_niche_hunter(use_gemini=False, top_n=20, save_db=True)
        return

    if mode == "full":
        run_niche_hunter(use_gemini=True, top_n=20, save_db=True)
        return

    raise ValueError(f"Unknown refresh-db mode: {mode}")


def _pick_topics(limit: int, min_score: float) -> list[tuple[str, float, str, str]]:
    """Pick unused topics from niche_hunter.db via llm_content helper."""
    from llm_content import get_unused_topics

    # get_unused_topics already sorts by final_score DESC and filters published titles
    candidates = get_unused_topics(top_n=max(20, limit * 5))

    picked: list[tuple[str, float, str, str]] = []
    for topic, score, niche, hook in candidates:
        if score < min_score:
            continue
        picked.append((topic, float(score), niche or "", hook or ""))
        if len(picked) >= limit:
            break

    return picked


def _strip_md_links(text: str) -> str:
    # Remove markdown links: [label](url) -> label
    text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1", text)
    # Remove bare URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove trailing citation artifacts like ".nanalyze" / ".pbpc" that occur
    # when the source was like ".[nanalyze](...)".
    text = re.sub(r"\.[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*\s*$", "", text)
    return text


def _pick_topics_from_ideas_file(file_path: str, limit: int) -> list[tuple[str, float, str, str]]:
    """Extract micro-niche idea bullet points from a Markdown file and treat them as topics.

    Expected structure (from the winner-post file): a section with bullet lists under headings like
    "Mycelium-Based Materials", "Fiber Plant Materials", "Other Plant Composites".
    """
    if not file_path or not os.path.isfile(file_path):
        raise FileNotFoundError(f"ideas file not found: {file_path}")

    raw = open(file_path, "r", encoding="utf-8", errors="replace").read()
    lines = raw.splitlines()

    # Start scanning from the micro-niche ideas area if present.
    start_idx = 0
    for i, ln in enumerate(lines):
        if "micro niche idea" in ln.lower() and "plant" in ln.lower():
            start_idx = i
            break
        if "mycelium-based materials" in ln.lower():
            start_idx = i
            break

    candidates: list[str] = []
    for ln in lines[start_idx:]:
        # Stop once we leave the ideas block (comparison table / checklist etc.)
        if ln.strip().lower().startswith("## comparison table"):
            break
        if ln.strip().startswith("|"):
            continue  # skip tables
        if not ln.lstrip().startswith("-"):
            continue

        idea = ln.lstrip().lstrip("-").strip()
        if not idea:
            continue

        idea = _strip_md_links(idea)
        idea = re.sub(r"\s+", " ", idea).strip()
        idea = idea.rstrip(".")

        # Filter out ultra-short bullets
        if len(idea) < 18:
            continue

        candidates.append(idea)

    # De-dupe while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for c in candidates:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
        if len(uniq) >= limit:
            break

    # Return (topic, score, niche, hook)
    return [(t, 9.2, "ideas_file", "") for t in uniq]


def _seed_keywords_from_ideas(ideas: list[str]) -> set[str]:
    seeds: set[str] = set()
    for idea in ideas:
        seeds |= _norm_tokens(idea)
    return seeds


def _rank_db_candidates_by_seed(
    db_candidates: list[tuple[str, float, str, str]],
    seed_keywords: set[str],
) -> list[tuple[str, float, str, str]]:
    if not seed_keywords:
        return db_candidates

    def score_item(item: tuple[str, float, str, str]) -> float:
        topic = item[0]
        tset = _norm_tokens(topic)
        if not tset:
            return 0.0
        return len(tset & seed_keywords) / max(1, len(tset))

    # Prefer higher keyword overlap; stable tie-break by original order
    return sorted(db_candidates, key=score_item, reverse=True)


def _norm_tokens(text: str) -> set[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    toks = [t for t in re.split(r"\s+", text) if t]
    # Drop very short tokens (noise)
    return {t for t in toks if len(t) >= 4}


def _overlap_ratio(topic: str, hay: str) -> float:
    """How much of topic's keyword set appears in hay (0-1)."""
    tset = _norm_tokens(topic)
    if not tset:
        return 0.0
    hset = _norm_tokens(hay)
    return len(tset & hset) / max(1, len(tset))


def _jaccard(a: str, b: str) -> float:
    sa = _norm_tokens(a)
    sb = _norm_tokens(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _shared_count(a: str, b: str) -> int:
    return len(_norm_tokens(a) & _norm_tokens(b))


async def _fetch_recent_publer_text(pages: int, limit: int, status: str = "") -> list[str]:
    """Fetch recent Publer posts text across N pages."""
    from integrations.publer_publisher import PublerPublisher

    pub = PublerPublisher()
    await pub.connect()
    texts: list[str] = []
    for page in range(1, pages + 1):
        posts = await pub.get_posts(status=status, limit=limit, page=page)
        for p in posts:
            title = str(p.get("title", "") or "")
            body = str(p.get("text", p.get("body", "") or "") or "")
            # Keep it bounded for speed
            texts.append((title + "\n" + body[:800]).strip())
        await asyncio.sleep(0.2)
    return [t for t in texts if t]


async def _dedup_against_publer(
    candidates: list[tuple[str, float, str, str]],
    limit: int,
    pages: int,
    per_page: int,
    min_overlap: float,
    min_shared: int,
    min_jaccard: float,
) -> list[tuple[str, float, str, str]]:
    """Filter out candidates that look too similar to recent Publer posts."""
    recent = await _fetch_recent_publer_text(pages=pages, limit=per_page)
    if not recent:
        return candidates[:limit]

    kept: list[tuple[str, float, str, str]] = []
    for topic, score, niche, hook in candidates:
        is_dup = False
        for txt in recent:
            if _overlap_ratio(topic, txt) >= min_overlap:
                is_dup = True
                break
            if _shared_count(topic, txt) >= min_shared and _jaccard(topic, txt) >= min_jaccard:
                is_dup = True
                break
        if is_dup:
            continue
        kept.append((topic, score, niche, hook))
        if len(kept) >= limit:
            break

    return kept


async def _generate_content(topic: str, score: float) -> dict[str, Any]:
    from llm_content import generate_quality_post

    # Season is computed by month inside generate_quality_post().
    # Optional override: set VIRALOPS_SEASON=Winter|Spring|Summer|Fall
    season_override = os.environ.get("VIRALOPS_SEASON", "").strip()
    pack = await asyncio.to_thread(
        generate_quality_post,
        topic=topic,
        score=score,
        season=season_override,
    )
    if not pack or not pack.get("content_formatted"):
        raise ValueError(f"Failed generate: {topic[:80]}")
    return pack


async def _publish_tiktok(cp: dict[str, Any], account_id: str, label: str) -> dict:
    from web.app import _prepare_tiktok_content
    from integrations.publer_publisher import PublerPublisher

    pc = await _prepare_tiktok_content(cp, "tiktok")
    pc["account_ids"] = [account_id]
    pc["_account_label"] = label
    pub = PublerPublisher()
    r = await pub.publish(pc)
    if not isinstance(r, dict):
        return {"success": False, "error": str(r)}

    # TikTok OpenAPI spam guard: if too many posts in last 24h, auto-schedule.
    err = str(r.get("error", "") or "")
    err_l = err.lower()
    is_tiktok_limit = (
        ("tiktok" in err_l and "openapi" in err_l)
        or ("too many posts" in err_l and "24 hours" in err_l)
        or ("blocked" in err_l and "minimize spam" in err_l)
    )
    if (not r.get("success")) and is_tiktok_limit:
        hours = float(os.environ.get("VIRALOPS_TIKTOK_SCHEDULE_FALLBACK_HOURS", "25") or "25")
        base = datetime.now(timezone.utc) + timedelta(hours=hours)
        # Stagger schedules per account to avoid same-minute collisions.
        try:
            offset_min = int(os.environ.get("VIRALOPS_TIKTOK_SCHEDULE_STAGGER_MIN", "2") or "2")
        except Exception:
            offset_min = 2
        # Deterministic offset from account_id
        acct_bump = sum(ord(c) for c in str(account_id)) % max(1, offset_min)

        last: dict | None = None
        for attempt in range(4):
            schedule_at = (base + timedelta(minutes=acct_bump + attempt * 2)).replace(microsecond=0).isoformat()
            pc2 = dict(pc)
            pc2["schedule_at"] = schedule_at
            r2 = await pub.publish(pc2)
            if isinstance(r2, dict) and r2.get("success"):
                r2["_fallback"] = "scheduled_due_to_tiktok_openapi_limit"
                r2["_scheduled_at"] = schedule_at
                return r2
            last = r2 if isinstance(r2, dict) else {"success": False, "error": str(r2)}
            err2 = str((last or {}).get("error", "") or "").lower()
            if "one minute gap" not in err2 and "another post at this time" not in err2:
                break
            await asyncio.sleep(0.5)

        return last or {"success": False, "error": "schedule fallback failed"}

    return r


async def _publish_facebook(cp: dict[str, Any]) -> dict:
    from web.app import _prepare_facebook_content
    from integrations.publer_publisher import PublerPublisher

    pc = await _prepare_facebook_content(cp)
    if not pc:
        return {"success": False, "error": "FB empty"}
    pc["platforms"] = ["facebook"]

    pub = PublerPublisher()
    await pub.connect()

    # 1) Explicit override
    fb_override = os.environ.get("VIRALOPS_FACEBOOK_ACCOUNT_ID", "").strip()
    if fb_override:
        pc["account_ids"] = [fb_override]
    else:
        # 2) Discover currently-connected FB accounts in Publer workspace
        fb_ids = await pub.get_account_ids("facebook")
        if fb_ids:
            pc["account_ids"] = [fb_ids[0]]
        elif FACEBOOK:
            # 3) Last resort legacy ID
            pc["account_ids"] = [FACEBOOK]

    r = await pub.publish(pc)
    if isinstance(r, dict) and (not r.get("success")):
        err = str(r.get("error", "") or "")
        if "composer is in a bad state" in err.lower():
            r["error"] = (
                err
                + " | FIX: In Publer dashboard → Accounts, reselect/refresh the Facebook Page connection (disconnect/reconnect if needed)."
            )
    return r if isinstance(r, dict) else {"success": False, "error": str(r)}


async def _publish_pinterest(cp: dict[str, Any]) -> dict:
    from web.app import _prepare_pinterest_content
    from integrations.publer_publisher import PublerPublisher

    # Publer Pinterest API doesn't expose a dedicated alt_text field.
    # Best-effort: include a short ALT line at the top of the description.
    alt = str(
        cp.get("image_alt")
        or cp.get("image_alt_text")
        or cp.get("title")
        or cp.get("_topic")
        or ""
    ).strip()
    alt = re.sub(r"\s+", " ", alt)[:150] if alt else ""
    if alt:
        cp = dict(cp)
        cp["_pinterest_alt"] = alt

    pc = await _prepare_pinterest_content(cp)
    if not pc:
        return {"success": False, "error": "Pinterest no image"}
    pc["platforms"] = ["pinterest"]

    if alt and pc.get("caption"):
        cap = str(pc.get("caption") or "")
        if cap and "alt:" not in cap.lower():
            pc["caption"] = (f"ALT: {alt}\n\n" + cap)[:500]

    pub = PublerPublisher()
    await pub.connect()

    # 1) Explicit override
    pin_override = os.environ.get("VIRALOPS_PINTEREST_ACCOUNT_ID", "").strip()
    if pin_override:
        pc["account_ids"] = [pin_override]
    else:
        # 2) Discover currently-connected Pinterest accounts in Publer workspace
        pin_ids = await pub.get_account_ids("pinterest")
        if pin_ids:
            pc["account_ids"] = [pin_ids[0]]
        elif PINTEREST:
            # 3) Last resort legacy ID
            pc["account_ids"] = [PINTEREST]

    # Pinterest requires selecting an album/board.
    album_override = os.environ.get("VIRALOPS_PINTEREST_ALBUM_ID", "").strip()
    if album_override:
        pc["album_id"] = album_override
    else:
        try:
            accounts = await pub.get_accounts()
            target_id = (pc.get("account_ids") or [""])[0]
            acc = next((a for a in (accounts or []) if str(a.get("id", a.get("_id", ""))) == str(target_id)), None)
            albums = (acc or {}).get("albums") or []
            if albums and isinstance(albums, list) and isinstance(albums[0], dict) and albums[0].get("id"):
                pc["album_id"] = str(albums[0].get("id"))
        except Exception:
            pass

    r = await pub.publish(pc)
    # Best-effort: fetch the created post objects so we can print the Pinterest pin link.
    # Publer can be eventually consistent; poll briefly.
    try:
        if isinstance(r, dict) and r.get("success") and r.get("created_post_ids"):
            created_ids = [str(x) for x in (r.get("created_post_ids") or []) if str(x).strip()]
            if created_ids:
                created_posts: list[dict[str, Any]] = []

                for _attempt in range(12):
                    if created_posts:
                        break
                    # Scan a few pages of recent posts.
                    found: list[dict[str, Any]] = []
                    for page in (1, 2, 3, 4, 5):
                        recent = await pub.get_posts(limit=50, page=page)
                        for cid in created_ids[:5]:
                            p = next((pp for pp in (recent or []) if str(pp.get("id")) == cid), None)
                            if not isinstance(p, dict):
                                continue
                            found.append(
                                {
                                    "id": p.get("id"),
                                    "url": p.get("url"),
                                    "post_link": p.get("post_link"),
                                    "state": p.get("state"),
                                }
                            )
                    # De-dupe by id
                    seen_ids: set[str] = set()
                    for it in found:
                        iid = str(it.get("id") or "")
                        if not iid or iid in seen_ids:
                            continue
                        seen_ids.add(iid)
                        created_posts.append(it)

                    if not created_posts:
                        await asyncio.sleep(1.0)

                if created_posts:
                    r["created_posts"] = created_posts
                    # Convenience: expose first post_link and url
                    first = created_posts[0] if created_posts else {}
                    first_link = str((first or {}).get("post_link") or "").strip()
                    first_url = str((first or {}).get("url") or "").strip()
                    if first_link:
                        r["post_link"] = first_link
                    if first_url:
                        r["url"] = first_url
    except Exception:
        pass

    # Fallback: match the created Pinterest post by destination URL.
    # This covers cases where Publer doesn't return created_post_ids or the post isn't
    # immediately visible for id-based lookup.
    try:
        if isinstance(r, dict) and r.get("success"):
            if not str(r.get("post_link") or "").strip():
                desired_url = str(pc.get("url") or "").strip()
                if desired_url:
                    for _attempt in range(18):
                        match = None
                        for page in (1, 2, 3, 4, 5):
                            recent = await pub.get_posts(limit=50, page=page)
                            match = next(
                                (
                                    p
                                    for p in (recent or [])
                                    if isinstance(p, dict)
                                    and str(p.get("url") or "").strip() == desired_url
                                ),
                                None,
                            )
                            if isinstance(match, dict):
                                break
                        if isinstance(match, dict):
                            r["url"] = match.get("url")
                            r["matched_post_id"] = match.get("id")
                            link = str(match.get("post_link") or "").strip()
                            if link:
                                r["post_link"] = link
                            break
                        await asyncio.sleep(1.0)
    except Exception:
        pass
    if isinstance(r, dict) and (not r.get("success")):
        err = str(r.get("error", "") or "")
        if "composer is in a bad state" in err.lower():
            r["error"] = (
                err
                + " | FIX: In Publer dashboard → Accounts, reselect/refresh the Pinterest Business connection (disconnect/reconnect if needed)."
            )
        # Some Publer setups reject content_type="pin" for Pinterest; retry with alternatives.
        if ("post type is not valid" in err.lower()) or ("album can't be blank" in err.lower()):
            for ct in ("photo", "status", "video"):
                pc2 = dict(pc)
                pc2["content_type"] = ct
                r2 = await pub.publish(pc2)
                if isinstance(r2, dict) and r2.get("success"):
                    r2["_fallback"] = f"pinterest_content_type_{ct}"
                    return r2
    return r if isinstance(r, dict) else {"success": False, "error": str(r)}


def _build_shopify_article_from_pack(cp: dict[str, Any], *, draft: bool) -> dict[str, Any]:
    """Best-effort map a ViralOps content pack to Shopify Article fields.

    Shopify Admin API requires: title + body_html.
    We also try to fill common Shopify UI fields: excerpt/summary, tags, author, SEO.
    """
    title = str(cp.get("title") or "").strip()
    if not title:
        title = str(cp.get("_topic") or "").strip()

    # Prefer an explicit HTML field if present.
    body_html = (
        cp.get("body_html")
        or cp.get("article_html")
        or cp.get("blog_html")
        or ""
    )

    if not body_html:
        # Convert markdown-ish text to simple HTML paragraphs.
        src = str(cp.get("content_formatted") or cp.get("universal_caption_block") or cp.get("body") or "").strip()
        # Strip bare URLs from body; we want Pinterest to carry the destination link.
        src = re.sub(r"https?://\S+", "", src)
        parts = [ln.strip() for ln in src.splitlines() if ln.strip()]
        body_html = "\n".join(f"<p>{ln}</p>" for ln in parts)

    def _clean_meta(s: str) -> str:
        s = re.sub(r"<[^>]+>", " ", s or "")
        s = re.sub(r"https?://\S+", "", s)
        # Remove most emoji/symbol codepoints for SERP stability
        s = re.sub(r"[\U00010000-\U0010ffff]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = s.replace('"', "").replace("'", "")
        return s

    def _make_meta_description() -> str:
        # Prefer explicit seo_description if it's already good.
        candidate = _clean_meta(str(cp.get("seo_description") or "").strip())
        if 80 <= len(candidate) <= 160:
            return candidate[:155]

        hook = _clean_meta(str(cp.get("hook") or "").strip())
        step_1 = _clean_meta(str(cp.get("step_1") or "").strip())
        step_2 = _clean_meta(str(cp.get("step_2") or "").strip())
        result = _clean_meta(str(cp.get("result") or "").strip())
        micro = _clean_meta(str(cp.get("micro_keywords") or "").strip())

        kws: list[str] = []
        if micro:
            for t in re.split(r"[•,\n]+", micro):
                tt = t.strip()
                if tt and len(tt) >= 3:
                    kws.append(tt)
                if len(kws) >= 2:
                    break

        base = hook or _clean_meta(title)

        parts: list[str] = []
        if base:
            parts.append(base.rstrip(".!?") + ".")
        steps = ", ".join([p for p in [step_1, step_2] if p])
        if steps:
            parts.append(f"Step-by-step: {steps}.")
        if result:
            parts.append(result.rstrip(".!?") + ".")
        if kws:
            parts.append("Keywords: " + ", ".join(kws) + ".")

        meta = _clean_meta(" ".join(parts).strip())
        if len(meta) > 155:
            meta = meta[:155].rsplit(" ", 1)[0]
        if len(meta) < 70:
            meta = _clean_meta(title)[:70] + ". " + meta
            meta = meta[:155].rsplit(" ", 1)[0] if len(meta) > 155 else meta
        return meta

    seo_description = _make_meta_description()

    summary_html = str(cp.get("summary_html") or "").strip()
    if not summary_html and seo_description:
        summary_html = f"<p>{seo_description}</p>"

    # Tags
    tags: list[str] = []
    micro = str(cp.get("micro_keywords") or "").strip()
    if micro:
        for t in re.split(r"[•,\n]+", micro):
            tt = t.strip()
            if tt:
                tags.append(tt)
    niche = str(cp.get("_niche") or cp.get("niche") or "").strip()
    if niche:
        tags.append(niche)
    tags = [t[:60] for t in tags if t][:20]

    author = str(
        os.environ.get("SHOPIFY_AUTHOR", "")
        or os.environ.get("VIRALOPS_SHOPIFY_AUTHOR", "")
        or os.environ.get("BLOG_AUTHOR", "")
        or os.environ.get("AUTHOR", "")
    ).strip()

    # Image handling
    # - Shopify featured image needs a PUBLIC URL; if we only have a local file, we upload it after article creation.
    image_url = ""
    image_local_path = ""
    candidate = str(cp.get("image_url") or cp.get("media_url") or "").strip()
    if candidate.startswith(("http://", "https://")):
        image_url = candidate
    else:
        local = str(cp.get("_ai_image_path") or cp.get("media_local_path") or "").strip()
        if local and os.path.isfile(local):
            image_local_path = local

    image_alt = _clean_meta(str(cp.get("image_alt") or title).strip())
    if not image_alt:
        image_alt = _clean_meta(title)

    out: dict[str, Any] = {
        "title": title,
        "body_html": body_html,
        "summary_html": summary_html,
        "tags": ", ".join(tags) if tags else "",
        "author": author,
        "seo_title": title[:70],
        "seo_description": seo_description,
        "published": (not draft),
        "image_alt": image_alt,
    }
    if image_url:
        out["image_url"] = image_url
    if image_local_path:
        out["image_local_path"] = image_local_path
    return out


async def _publish_shopify_blog(cp: dict[str, Any], *, draft: bool) -> dict:
    """Create a Shopify blog article and return its public URL so Pinterest can match 1-1."""
    from integrations.shopify_blog_publisher import ShopifyBlogPublisher
    from core.models import QueueItem

    shopify_account_id = os.environ.get("VIRALOPS_SHOPIFY_ACCOUNT_ID", "shopify_viralops").strip() or "shopify_viralops"
    publisher = ShopifyBlogPublisher(account_id=shopify_account_id)
    ok = await publisher.connect()
    if not ok:
        return {"success": False, "error": "Shopify connect failed (check SHOPIFY_SHOP/SHOPIFY_ACCESS_TOKEN/SHOPIFY_BLOG_ID)"}

    content = _build_shopify_article_from_pack(cp, draft=draft)
    qi = QueueItem(
        id=f"shopify_{uuid.uuid4().hex[:10]}",
        content_pack_id=str(cp.get("id") or cp.get("content_pack_id") or "batch"),
        platform="shopify_blog",
    )

    try:
        res = await publisher.publish(qi, content)
    finally:
        try:
            await publisher.close()
        except Exception:
            pass

    if not getattr(res, "success", False):
        return {"success": False, "error": getattr(res, "error", "Shopify publish failed")}
    return {
        "success": True,
        "post_url": getattr(res, "post_url", ""),
        "post_id": getattr(res, "post_id", ""),
        "admin_url": (getattr(res, "metadata", {}) or {}).get("admin_url", ""),
        "handle": (getattr(res, "metadata", {}) or {}).get("handle", ""),
        "published": (getattr(res, "metadata", {}) or {}).get("published", True),
    }


async def _publish_all(
    cp: dict[str, Any],
    *,
    platforms: list[str],
    publish_shopify: bool,
    shopify_draft: bool,
) -> dict[str, dict]:
    results: dict[str, dict] = {}

    if "tiktok" in platforms:
        # TikTok multi-account: publish DISTINCT variants per account to avoid
        # duplicate/copyright detection.
        from llm_content import make_tiktok_account_variant

        base_topic = str(cp.get("_topic") or cp.get("title") or "").strip() or "(unknown topic)"
        tiktok_accounts = await _get_tiktok_accounts()

        for idx, (account_id, label) in enumerate(tiktok_accounts, 1):
            variant_id = f"tiktok{idx}"
            cp_variant = make_tiktok_account_variant(
                cp,
                topic=base_topic,
                account_label=label,
                variant_id=variant_id,
            )
            # IMPORTANT: pass the explicit Publer account ID into the pack so `_prepare_tiktok_content()`
            # does not round-robin-pick a different account (and logs match the actual target).
            variant_pack = dict(cp_variant)
            variant_pack["account_ids"] = [account_id]
            variant_pack["_account_label"] = f"tiktok#{idx}"
            results[f"tiktok{idx}"] = await _publish_tiktok(variant_pack, account_id, label)
            await asyncio.sleep(2)

    if "facebook" in platforms:
        results["facebook"] = await _publish_facebook(cp)
        await asyncio.sleep(2)

    # Optional: publish Shopify first, then attach the article URL to Pinterest.
    pin_pack = cp
    if publish_shopify and ("shopify_blog" in platforms):
        shop = await _publish_shopify_blog(cp, draft=shopify_draft)
        results["shopify_blog"] = shop
        if ("pinterest" in platforms):
            if shopify_draft:
                results["pinterest"] = {
                    "success": False,
                    "error": "Skipped Pinterest: Shopify article was created as draft (not public), so the destination link would 404.",
                }
                return results
            if shop.get("success") and shop.get("post_url"):
                pin_pack = dict(cp)
                pin_pack["destination_url"] = shop["post_url"]
                pin_pack["url"] = shop["post_url"]
            else:
                # Enforce 1-1 match: if Shopify failed, do NOT publish Pinterest with a wrong/missing link.
                results["pinterest"] = {
                    "success": False,
                    "error": "Skipped Pinterest: Shopify blog publish failed so destination link would not match.",
                }
                return results

    if "pinterest" in platforms:
        results["pinterest"] = await _publish_pinterest(pin_pack)

    return results


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true", help="Actually publish. Default is dry-run.")
    ap.add_argument(
        "--shopify",
        action="store_true",
        help="Also publish a Shopify blog article first, then set Pinterest destination link to the Shopify article URL.",
    )
    ap.add_argument(
        "--shopify-draft",
        action="store_true",
        help="Create Shopify article as draft (unpublished). Still uses the computed URL for Pinterest if available.",
    )
    ap.add_argument(
        "--platforms",
        type=str,
        default="tiktok,facebook,pinterest",
        help="Comma-separated platforms to publish. Example: shopify_blog,pinterest",
    )
    ap.add_argument(
        "--ideas-file",
        type=str,
        default="",
        help="Optional: path to a Markdown file to source micro-niche idea bullet topics from (overrides niche_hunter.db picking).",
    )
    ap.add_argument("--limit", type=int, default=3, help="How many topics to process.")
    ap.add_argument("--min-score", type=float, default=8.5, help="Min niche score to accept.")
    ap.add_argument("--refresh-db", choices=["local", "full"], default=None, help="Refresh niche_hunter.db before picking topics.")
    ap.add_argument("--no-publer-dedup", action="store_true", help="Disable Publer-level dedup against recent posts.")
    ap.add_argument("--publer-pages", type=int, default=5, help="How many Publer pages to scan for dedup.")
    ap.add_argument("--publer-per-page", type=int, default=50, help="How many posts per page to fetch for dedup.")
    ap.add_argument(
        "--publer-min-overlap",
        type=float,
        default=0.65,
        help="Dedup threshold: fraction of topic keywords seen in a recent Publer post.",
    )
    ap.add_argument(
        "--publer-min-shared",
        type=int,
        default=3,
        help="Dedup threshold: minimum shared keywords with a recent Publer post.",
    )
    ap.add_argument(
        "--publer-min-jaccard",
        type=float,
        default=0.22,
        help="Dedup threshold: minimum Jaccard similarity (on keyword tokens) when shared-keywords threshold is met.",
    )
    args = ap.parse_args()

    raw_platforms = [p.strip().lower() for p in str(args.platforms or "").split(",") if p.strip()]
    # Allow a couple of friendly aliases.
    normalized: list[str] = []
    for p in raw_platforms:
        if p in ("shopify", "shopifyblog", "shopify_blog"):
            normalized.append("shopify_blog")
            continue
        if p in ("fb", "facebook"):
            normalized.append("facebook")
            continue
        if p in ("pin", "pinterest"):
            normalized.append("pinterest")
            continue
        if p in ("tt", "tiktok"):
            normalized.append("tiktok")
            continue
        normalized.append(p)
    # De-dupe while keeping order.
    platforms: list[str] = []
    for p in normalized:
        if p not in platforms:
            platforms.append(p)

    # If --shopify is enabled, ensure shopify_blog is included and that it runs before pinterest (for link matching).
    if args.shopify and ("shopify_blog" not in platforms):
        platforms = ["shopify_blog"] + platforms
    if "shopify_blog" in platforms and "pinterest" in platforms:
        # Ensure order: shopify_blog then pinterest.
        platforms = [p for p in platforms if p not in ("shopify_blog", "pinterest")] \
            + ["shopify_blog", "pinterest"]

    if args.refresh_db:
        print(f"[AUTO] Refreshing niche_hunter.db via mode={args.refresh_db}...")
        _refresh_db(args.refresh_db)

    # Pick extra candidates first, then filter by Publer recent history
    ideas_candidates: list[tuple[str, float, str, str]] = []
    seed_keywords: set[str] = set()

    if args.ideas_file:
        ideas_candidates = _pick_topics_from_ideas_file(args.ideas_file, limit=max(20, args.limit * 8))
        seed_keywords = _seed_keywords_from_ideas([t for (t, *_rest) in ideas_candidates])

    db_candidates = _pick_topics(limit=max(20, args.limit * 10), min_score=args.min_score)
    if seed_keywords:
        db_candidates = _rank_db_candidates_by_seed(db_candidates, seed_keywords)

    # Combine: ideas first, then DB fallback.
    candidates = ideas_candidates + db_candidates

    # Filter out already-used topics (persisted history)
    candidates = [c for c in candidates if not _topic_already_used(c[0])]
    if not args.no_publer_dedup:
        topics = await _dedup_against_publer(
            candidates,
            limit=args.limit,
            pages=args.publer_pages,
            per_page=args.publer_per_page,
            min_overlap=args.publer_min_overlap,
            min_shared=args.publer_min_shared,
            min_jaccard=args.publer_min_jaccard,
        )
    else:
        topics = candidates[: args.limit]

    print("=" * 70)
    print("  AUTO NICHE TOPICS — Batch")
    print(f"  Picked {len(topics)}/{args.limit} topics (min_score={args.min_score})")
    if not args.no_publer_dedup:
        print(
            f"  Publer dedup: pages={args.publer_pages}, per_page={args.publer_per_page}, min_overlap={args.publer_min_overlap}, min_shared={args.publer_min_shared}, min_jaccard={args.publer_min_jaccard}"
        )
    print(f"  Mode: {'PUBLISH' if args.publish else 'DRY-RUN'}")
    print(f"  Platforms: {', '.join(platforms) if platforms else '(none)'}")
    print("=" * 70)

    if not topics:
        print("No topics found (maybe exhausted). Try --refresh-db full")
        return 2

    for idx, (topic, score, niche, hook) in enumerate(topics, 1):
        print(f"\n[{idx}/{len(topics)}] score={score:.2f} niche={niche}")
        if hook:
            print(f"  hook: {hook[:140]}")
        print(f"  topic: {topic}")

        if not args.publish:
            continue

        try:
            cp = await _generate_content(topic, score=9.0)
            title = cp.get("title", "?")
            chars = len(cp.get("content_formatted", ""))
            review_score = float(cp.get("_review_score", 0.0) or 0.0)
            rubric_total = float(cp.get("_rubric_total_100", 0.0) or 0.0)
            review_pass = bool(cp.get("_review_pass"))
            print(f"  ✓ Generated: {title[:90]} ({chars} chars, review={review_score:.1f}/10)")

            # Hard gate: never publish if quality review didn't pass.
            if not review_pass:
                fb = str(cp.get("_review_feedback", "") or "").strip()
                print(
                    f"  ⚠ SKIP PUBLISH: quality gate failed (review_pass={review_pass}, "
                    f"tiktok_avg={review_score:.1f}/10, rubric={rubric_total:.0f}/100)."
                )
                if fb:
                    print(f"  Feedback: {fb[:220]}")

                _log_published_to_viralops_db(
                    {
                        "title": cp.get("title", ""),
                        "content_formatted": cp.get("content_formatted", ""),
                        "universal_caption_block": cp.get("universal_caption_block", ""),
                    },
                    platforms=platforms,
                    extra={
                        "topic": topic,
                        "picked_score": score,
                        "picked_niche": niche,
                        "picked_hook": hook,
                        "review_pass": review_pass,
                        "tiktok_avg": review_score,
                        "rubric_total_100": rubric_total,
                        "feedback": fb,
                    },
                    status="failed",
                )
                continue

            results = await _publish_all(
                cp,
                platforms=platforms,
                publish_shopify=bool(args.shopify),
                shopify_draft=bool(args.shopify_draft),
            )
            for pf, r in results.items():
                ok = bool(r.get("success"))
                if ok and r.get("scheduled"):
                    when = str(r.get("_scheduled_at") or r.get("schedule_at") or "").strip()
                    msg = f"SCHEDULED {when}".strip()
                else:
                    msg = "OK" if ok else f"FAIL: {str(r.get('error', '?'))[:140]}"
                print(f"    {'✓' if ok else '✗'} {pf}: {msg}")

            # Print match summary (Shopify URL ↔ Pinterest pin link)
            shop = results.get("shopify_blog") or {}
            pin = results.get("pinterest") or {}
            shop_url = str(shop.get("post_url") or "").strip()
            shop_admin = str(shop.get("admin_url") or "").strip()
            pin_link = str(pin.get("post_link") or "").strip()
            pin_dest = str(pin.get("url") or "").strip()
            pin_post_id = ""
            if isinstance(pin, dict):
                # Prefer matched_post_id, then created_post_ids, then job_id/post_id.
                pin_post_id = str(pin.get("matched_post_id") or "").strip()
                if not pin_post_id:
                    cids = pin.get("created_post_ids") or []
                    if isinstance(cids, list) and cids:
                        pin_post_id = str(cids[0] or "").strip()
                if not pin_post_id:
                    pin_post_id = str(pin.get("post_id") or "").strip()
            if (not pin_link) or (not pin_dest):
                created_posts = pin.get("created_posts") or []
                if isinstance(created_posts, list) and created_posts and isinstance(created_posts[0], dict):
                    if not pin_link:
                        pin_link = str(created_posts[0].get("post_link") or "").strip()
                    if not pin_dest:
                        pin_dest = str(created_posts[0].get("url") or "").strip()
            if shop_url:
                print(f"      shopify_url: {shop_url}")
            if shop_admin:
                print(f"      shopify_admin: {shop_admin}")
            if pin_link:
                print(f"      pinterest_pin: {pin_link}")
            if pin_dest:
                print(f"      pinterest_dest: {pin_dest}")
            if (not pin_link) and pin_post_id:
                print(f"      pinterest_post_id: {pin_post_id}")

            # Persist only if ALL platform publishes succeeded.
            all_ok = all(bool(r.get("success")) for r in results.values())
            if all_ok:
                _log_published_to_viralops_db(
                    cp,
                    platforms=platforms,
                    extra={
                        "topic": topic,
                        "picked_score": score,
                        "picked_niche": niche,
                        "picked_hook": hook,
                        "shopify_post_url": (results.get("shopify_blog") or {}).get("post_url", ""),
                        "pinterest_dest_url": (results.get("shopify_blog") or {}).get("post_url", "") or cp.get("destination_url") or cp.get("url") or "",
                    },
                )
                _mark_topic_used(topic, source=niche or "")
            else:
                # Record a failure entry (for debugging) but do NOT mark topic used.
                _log_published_to_viralops_db(
                    {"title": cp.get("title", ""), "content_formatted": cp.get("content_formatted", ""), "universal_caption_block": cp.get("universal_caption_block", "")},
                    platforms=platforms,
                    extra={
                        "topic": topic,
                        "picked_score": score,
                        "picked_niche": niche,
                        "picked_hook": hook,
                        "results": results,
                    },
                    status="failed",
                )

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:160]}")

        time.sleep(5)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
