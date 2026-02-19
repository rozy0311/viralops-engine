"""
ViralOps — Batch Multi-Platform Publisher
==========================================
Publishes content to ALL 4 accounts simultaneously:
  - TikTok x2 (The Rike Root Stories + The Rike Stories) — 15/day each
  - Facebook (The Rike) — unlimited
  - Pinterest (therike) — unlimited

Each topic generates ONE content piece → published to ALL platforms with
platform-specific formatting:
  - TikTok: clump-proof emoji separators, 3500-4000 chars
  - Facebook: full rich text with newlines, unlimited chars
  - Pinterest: pin with title (100 chars) + description (500 chars) + image
"""
import asyncio
import os
import sys
import time
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(override=True)


# Account IDs
TIKTOK_1 = "698c95e5b1ab790def1352c1"  # The Rike Root Stories
TIKTOK_2 = "69951ea30c4677f27c12d98c"  # The Rike Stories
FACEBOOK = "699522978f7449fba2dceebe"  # The Rike (FB page)
PINTEREST = "69951f098f7449fba2dceadd"  # therike (Pinterest business)

# Topics to publish
TOPICS = [
    "How to build a DIY vertical herb garden with recycled materials — complete step-by-step guide for apartment balconies",
    "The ultimate beginner guide to making natural homemade soap — cold process method with essential oils and herbs",
    "10 genius ways to repurpose glass jars into beautiful home decor — sustainable upcycling projects anyone can do",
    "Complete guide to growing avocado from seed to fruit — indoor and outdoor methods with expert tips",
]


async def generate_content(topic: str) -> dict:
    """Generate quality content using the ViralOps pipeline."""
    from llm_content import generate_quality_post
    pack = await asyncio.to_thread(
        generate_quality_post,
        topic=topic,
        score=9.0,
    )
    if not pack or not pack.get("content_formatted"):
        raise ValueError(f"Failed to generate content for: {topic[:60]}")
    return pack


async def publish_to_tiktok(content_pack: dict, account_id: str, account_label: str) -> dict:
    """Publish to a specific TikTok account with clump-proof formatting."""
    # Import from web.app the TikTok prepare function
    from web.app import _prepare_tiktok_content
    from integrations.publer_publisher import PublerPublisher

    publish_content = await _prepare_tiktok_content(content_pack, "tiktok")
    # Override account_ids to target specific account
    publish_content["account_ids"] = [account_id]
    publish_content["_account_label"] = account_label

    pub = PublerPublisher()
    result = await pub.publish(publish_content)
    if isinstance(result, dict):
        result["_account_label"] = account_label
    return result


async def publish_to_facebook(content_pack: dict) -> dict:
    """Publish to Facebook with full rich text formatting."""
    from web.app import _prepare_facebook_content
    from integrations.publer_publisher import PublerPublisher

    publish_content = await _prepare_facebook_content(content_pack)
    if not publish_content:
        return {"success": False, "error": "Facebook prepare returned empty"}

    # Ensure Facebook account ID is set
    publish_content["account_ids"] = [FACEBOOK]

    pub = PublerPublisher()
    result = await pub.publish(publish_content)
    return result if isinstance(result, dict) else {"success": False, "error": str(result)}


async def publish_to_pinterest(content_pack: dict) -> dict:
    """Publish to Pinterest as a pin with image + description."""
    from web.app import _prepare_pinterest_content
    from integrations.publer_publisher import PublerPublisher

    publish_content = await _prepare_pinterest_content(content_pack)
    if not publish_content:
        return {"success": False, "error": "Pinterest requires image — no image available"}

    # Ensure Pinterest account ID is set
    publish_content["account_ids"] = [PINTEREST]

    pub = PublerPublisher()
    result = await pub.publish(publish_content)
    return result if isinstance(result, dict) else {"success": False, "error": str(result)}


async def publish_all_platforms(content_pack: dict, topic_idx: int) -> dict:
    """Publish one content piece to ALL 4 accounts."""
    title = content_pack.get("title", "")[:60]
    results = {}

    # TikTok 1 — The Rike Root Stories
    print(f"  [{topic_idx}] TikTok 1 (Root Stories)...", end=" ", flush=True)
    try:
        r = await publish_to_tiktok(content_pack, TIKTOK_1, "The Rike Root Stories")
        ok = r.get("success", False)
        results["tiktok_1"] = r
        print(f"{'OK' if ok else 'FAIL: ' + r.get('error', '')[:60]}")
    except Exception as e:
        results["tiktok_1"] = {"success": False, "error": str(e)[:100]}
        print(f"ERROR: {e!s:.60}")

    await asyncio.sleep(2)  # Brief pause between API calls

    # TikTok 2 — The Rike Stories
    print(f"  [{topic_idx}] TikTok 2 (Stories)...", end=" ", flush=True)
    try:
        r = await publish_to_tiktok(content_pack, TIKTOK_2, "The Rike Stories")
        ok = r.get("success", False)
        results["tiktok_2"] = r
        print(f"{'OK' if ok else 'FAIL: ' + r.get('error', '')[:60]}")
    except Exception as e:
        results["tiktok_2"] = {"success": False, "error": str(e)[:100]}
        print(f"ERROR: {e!s:.60}")

    await asyncio.sleep(2)

    # Facebook — The Rike
    print(f"  [{topic_idx}] Facebook (The Rike)...", end=" ", flush=True)
    try:
        r = await publish_to_facebook(content_pack)
        ok = r.get("success", False)
        results["facebook"] = r
        print(f"{'OK' if ok else 'FAIL: ' + r.get('error', '')[:60]}")
    except Exception as e:
        results["facebook"] = {"success": False, "error": str(e)[:100]}
        print(f"ERROR: {e!s:.60}")

    await asyncio.sleep(2)

    # Pinterest — therike
    print(f"  [{topic_idx}] Pinterest (therike)...", end=" ", flush=True)
    try:
        r = await publish_to_pinterest(content_pack)
        ok = r.get("success", False)
        results["pinterest"] = r
        print(f"{'OK' if ok else 'FAIL: ' + r.get('error', '')[:60]}")
    except Exception as e:
        results["pinterest"] = {"success": False, "error": str(e)[:100]}
        print(f"ERROR: {e!s:.60}")

    return results


async def main():
    print("=" * 70)
    print("ViralOps — Multi-Platform Batch Publisher")
    print(f"Targets: TikTok x2 + Facebook + Pinterest = 4 accounts")
    print(f"Topics: {len(TOPICS)}")
    print("=" * 70)

    total_ok = 0
    total_fail = 0

    for i, topic in enumerate(TOPICS, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(TOPICS)}] {topic[:70]}...")
        print(f"{'─' * 60}")

        # Generate content
        print(f"  Generating quality content...", end=" ", flush=True)
        t0 = time.time()
        try:
            content_pack = await generate_content(topic)
            chars = len(content_pack.get("content_formatted", ""))
            score = content_pack.get("_review_score", 0)
            print(f"OK ({chars} chars, score={score:.1f}, {time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            total_fail += 4
            continue

        # Publish to all 4 accounts
        results = await publish_all_platforms(content_pack, i)

        for plat, r in results.items():
            if r.get("success"):
                total_ok += 1
            else:
                total_fail += 1

        # Pause between topics to avoid rate limits
        if i < len(TOPICS):
            print(f"\n  Waiting 10s before next topic...")
            await asyncio.sleep(10)

    print(f"\n{'=' * 70}")
    print(f"DONE: {total_ok} OK / {total_fail} FAIL out of {len(TOPICS) * 4} total")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
