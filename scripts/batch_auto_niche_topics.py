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
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Publer defaults (kept consistent with existing batch scripts)
os.environ.setdefault("PUBLER_API_KEY", "9540295ccce6cb94f26a4559e20f8a98e3ee02c63c7324f0")
os.environ.setdefault("PUBLER_WORKSPACE_ID", "698782eb57bbdca107c404f5")

TIKTOK_1 = "698c95e5b1ab790def1352c1"   # The Rike Root Stories
TIKTOK_2 = "69951ea30c4677f27c12d98c"   # The Rike Stories
FACEBOOK = "699522978f7449fba2dceebe"    # The Rike (Facebook)
PINTEREST = "69951f098f7449fba2dceadd"   # therike (Pinterest)


def _log_published_to_viralops_db(pack: dict[str, Any], platforms: list[str], extra: dict[str, Any] | None = None) -> None:
    """Persist a published post to web/viralops.db so future runs can dedup."""
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

    published_at = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO posts (title, body, platforms, status, published_at, extra_fields) VALUES (?, ?, ?, ?, ?, ?)",
            (
                title,
                body,
                json.dumps(platforms),
                "published",
                published_at,
                json.dumps(payload, ensure_ascii=False),
            ),
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
    r = await PublerPublisher().publish(pc)
    return r if isinstance(r, dict) else {"success": False, "error": str(r)}


async def _publish_facebook(cp: dict[str, Any]) -> dict:
    from web.app import _prepare_facebook_content
    from integrations.publer_publisher import PublerPublisher

    pc = await _prepare_facebook_content(cp)
    if not pc:
        return {"success": False, "error": "FB empty"}
    pc["account_ids"] = [FACEBOOK]
    r = await PublerPublisher().publish(pc)
    return r if isinstance(r, dict) else {"success": False, "error": str(r)}


async def _publish_pinterest(cp: dict[str, Any]) -> dict:
    from web.app import _prepare_pinterest_content
    from integrations.publer_publisher import PublerPublisher

    pc = await _prepare_pinterest_content(cp)
    if not pc:
        return {"success": False, "error": "Pinterest no image"}
    pc["account_ids"] = [PINTEREST]
    r = await PublerPublisher().publish(pc)
    return r if isinstance(r, dict) else {"success": False, "error": str(r)}


async def _publish_all(cp: dict[str, Any]) -> dict[str, dict]:
    results: dict[str, dict] = {}

    results["tiktok1"] = await _publish_tiktok(cp, TIKTOK_1, "RikeRoot")
    await asyncio.sleep(2)

    results["tiktok2"] = await _publish_tiktok(cp, TIKTOK_2, "RikeStories")
    await asyncio.sleep(2)

    results["facebook"] = await _publish_facebook(cp)
    await asyncio.sleep(2)

    results["pinterest"] = await _publish_pinterest(cp)

    return results


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true", help="Actually publish. Default is dry-run.")
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

    if args.refresh_db:
        print(f"[AUTO] Refreshing niche_hunter.db via mode={args.refresh_db}...")
        _refresh_db(args.refresh_db)

    # Pick extra candidates first, then filter by Publer recent history
    candidates = _pick_topics(limit=max(10, args.limit * 6), min_score=args.min_score)
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
            print(f"  ✓ Generated: {title[:90]} ({chars} chars, review={review_score:.1f}/10)")

            results = await _publish_all(cp)
            for pf, r in results.items():
                ok = bool(r.get("success"))
                msg = "OK" if ok else f"FAIL: {str(r.get('error', '?'))[:80]}"
                print(f"    {'✓' if ok else '✗'} {pf}: {msg}")

            # Persist to viralops.db if we had at least one successful publish
            if any(bool(r.get("success")) for r in results.values()):
                _log_published_to_viralops_db(
                    cp,
                    platforms=["tiktok", "facebook", "pinterest"],
                    extra={"topic": topic, "picked_score": score, "picked_niche": niche, "picked_hook": hook},
                )

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:160]}")

        time.sleep(5)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
