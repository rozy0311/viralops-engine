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
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Publer defaults (kept consistent with existing batch scripts)
os.environ.setdefault("PUBLER_API_KEY", "9540295ccce6cb94f26a4559e20f8a98e3ee02c63c7324f0")
os.environ.setdefault("PUBLER_WORKSPACE_ID", "698782eb57bbdca107c404f5")

TIKTOK_1 = "698c95e5b1ab790def1352c1"   # The Rike Root Stories
TIKTOK_2 = "69951ea30c4677f27c12d98c"   # The Rike Stories
FACEBOOK = "699522978f7449fba2dceebe"    # The Rike (Facebook)
PINTEREST = "69951f098f7449fba2dceadd"   # therike (Pinterest)


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


async def _generate_content(topic: str, score: float) -> dict[str, Any]:
    from llm_content import generate_quality_post

    pack = await asyncio.to_thread(generate_quality_post, topic=topic, score=score)
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
    args = ap.parse_args()

    if args.refresh_db:
        print(f"[AUTO] Refreshing niche_hunter.db via mode={args.refresh_db}...")
        _refresh_db(args.refresh_db)

    topics = _pick_topics(limit=args.limit, min_score=args.min_score)

    print("=" * 70)
    print("  AUTO NICHE TOPICS — Batch")
    print(f"  Picked {len(topics)}/{args.limit} topics (min_score={args.min_score})")
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

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:160]}")

        time.sleep(5)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
