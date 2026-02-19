"""
ViralOps — Winner Blog Series Batch
====================================
Topics extracted from winning blog posts (Winter Garden Layouts, Micro Niche).
Each blog sub-topic becomes a standalone TikTok post.

Method: "Winner Blog → TikTok Series" (Method #49 in TOÀN BỘ PHƯƠNG PHÁP)

Flow:
  1. Take proven winning blog sub-topics (hyper-localized, Zone 6a, clay soil)
  2. Generate 3500-4000 char TikTok content per sub-topic (Q&A expert style)
  3. Publish to 4 accounts: 2 TikTok + 1 Facebook + 1 Pinterest
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PUBLER_API_KEY", "9540295ccce6cb94f26a4559e20f8a98e3ee02c63c7324f0")
os.environ.setdefault("PUBLER_WORKSPACE_ID", "698782eb57bbdca107c404f5")

TIKTOK_1 = "698c95e5b1ab790def1352c1"   # The Rike Root Stories
TIKTOK_2 = "69951ea30c4677f27c12d98c"   # The Rike Stories
FACEBOOK = "699522978f7449fba2dceebe"    # The Rike (Facebook)
PINTEREST = "69951f098f7449fba2dceadd"   # therike (Pinterest)

# ═══════════════════════════════════════════════════════════════
# WINNER BLOG TOPICS — extracted from proven viral blog content
# Source: "Winter Garden Layouts blog" + "Micro Niche - Nano Niche"
# Each sub-topic is a standalone TikTok post (3500-4000 chars)
# ═══════════════════════════════════════════════════════════════
TOPICS = [
    # ─── From Winter Garden Layouts (Zone 6a Staunton Farms) ───
    "Do you know 25 raised bed layouts exist for Zone 6a winter gardens — here are the best 10 for heavy clay soil that floods in spring and freezes solid in winter",
    "Soil prep tips for Zone 6a spring gardens — how to fix heavy clay soil without making it worse and when to actually start planting",
    "Hydrozoning layouts for Zone 6a sloped yards with heavy clay — grouping plants by water needs so the top stays dry and the bottom stays happy",
    "Dwarf currant and gooseberry bushes as raised bed edges — how to grow fruit in a 6x3 bed on heavy clay without creating a thorny jungle",
    "Keyhole garden compost hubs for small lots with spring flooding — the African permaculture design that turns kitchen scraps into a self-feeding raised bed",

    # ─── From Micro/Nano Niche Discovery ───
    "Companion planting maps for raised beds in clay soil — which vegetables actually help each other grow and which ones start fights underground",
    "How to build a 4x8 raised bed for under 40 dollars — cedar vs pine vs concrete block and what actually lasts 10 years in Zone 6a weather",
    "Pizza garden in a raised bed — grow tomatoes basil oregano and peppers in one beautiful circular bed that feeds your Friday night habit",
]


async def generate_content(topic: str) -> dict:
    from llm_content import generate_quality_post
    pack = await asyncio.to_thread(generate_quality_post, topic=topic, score=9.0)
    if not pack or not pack.get("content_formatted"):
        raise ValueError(f"Failed: {topic[:60]}")
    return pack


async def publish_tiktok(cp, account_id, label):
    from web.app import _prepare_tiktok_content
    from integrations.publer_publisher import PublerPublisher
    pc = await _prepare_tiktok_content(cp, "tiktok")
    pc["account_ids"] = [account_id]
    pc["_account_label"] = label
    r = await PublerPublisher().publish(pc)
    return r if isinstance(r, dict) else {"success": False, "error": str(r)}


async def publish_facebook(cp):
    from web.app import _prepare_facebook_content
    from integrations.publer_publisher import PublerPublisher
    pc = await _prepare_facebook_content(cp)
    if not pc:
        return {"success": False, "error": "FB empty"}
    pc["account_ids"] = [FACEBOOK]
    r = await PublerPublisher().publish(pc)
    return r if isinstance(r, dict) else {"success": False, "error": str(r)}


async def publish_pinterest(cp):
    from web.app import _prepare_pinterest_content
    from integrations.publer_publisher import PublerPublisher
    pc = await _prepare_pinterest_content(cp)
    if not pc:
        return {"success": False, "error": "Pinterest no image"}
    pc["account_ids"] = [PINTEREST]
    r = await PublerPublisher().publish(pc)
    return r if isinstance(r, dict) else {"success": False, "error": str(r)}


async def publish_all(cp, idx):
    results = {}
    t1 = await publish_tiktok(cp, TIKTOK_1, "RikeRoot")
    results["tiktok1"] = "OK" if t1.get("success") else f"FAIL: {t1.get('error', '?')[:60]}"
    time.sleep(2)

    t2 = await publish_tiktok(cp, TIKTOK_2, "RikeStories")
    results["tiktok2"] = "OK" if t2.get("success") else f"FAIL: {t2.get('error', '?')[:60]}"
    time.sleep(2)

    fb = await publish_facebook(cp)
    results["facebook"] = "OK" if fb.get("success") else f"FAIL: {fb.get('error', '?')[:60]}"
    time.sleep(2)

    pin = await publish_pinterest(cp)
    results["pinterest"] = "OK" if pin.get("success") else f"FAIL: {pin.get('error', '?')[:60]}"

    return results


async def main():
    # Allow limiting topics via CLI: python batch_winner_blog.py 3
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(TOPICS)
    topics = TOPICS[:limit]
    
    print("=" * 70)
    print("  WINNER BLOG SERIES — TikTok Batch Publisher")
    print(f"  {len(topics)} topics × 4 accounts = {len(topics) * 4} posts")
    print("=" * 70)

    ok = 0
    fail = 0

    for i, topic in enumerate(topics, 1):
        print(f"\n{'─'*60}")
        print(f"  [{i}/{len(topics)}] {topic[:70]}...")
        print(f"{'─'*60}")

        try:
            cp = await generate_content(topic)
            title = cp.get("title", "?")
            chars = len(cp.get("content_formatted", ""))
            score = cp.get("_review_score", 0)
            print(f"  ✓ Generated: {title[:60]} ({chars} chars, {score:.1f}/10)")

            results = await publish_all(cp, i)

            for platform, status in results.items():
                icon = "✓" if status == "OK" else "✗"
                print(f"    {icon} {platform}: {status}")
                if status == "OK":
                    ok += 1
                else:
                    fail += 1

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:120]}")
            fail += 4  # all 4 platforms failed

        time.sleep(5)  # Rate limit between topics

    print(f"\n{'='*70}")
    print(f"  DONE: {ok} OK / {fail} FAIL / {ok + fail} total")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
