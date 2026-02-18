"""
ViralOps — Batch Multi-Platform Publisher (Round 2)
Fill up to 15 posts/account for TikTok, unlimited for FB/Pinterest.
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PUBLER_API_KEY", "9540295ccce6cb94f26a4559e20f8a98e3ee02c63c7324f0")
os.environ.setdefault("PUBLER_WORKSPACE_ID", "698782eb57bbdca107c404f5")

TIKTOK_1 = "698c95e5b1ab790def1352c1"  # The Rike Root Stories
TIKTOK_2 = "69951ea30c4677f27c12d98c"  # The Rike Stories
FACEBOOK = "699522978f7449fba2dceebe"  # The Rike (FB page)
PINTEREST = "69951f098f7449fba2dceadd"  # therike (Pinterest business)

TOPICS = [
    "How to make your own natural beeswax candles at home — step-by-step guide with essential oil blends for relaxation and focus",
    "The complete beginner guide to no-dig gardening — build rich healthy soil without ever turning the earth",
    "How to create a self-watering planter from recycled plastic bottles — perfect for busy people and apartment living",
    "5 powerful homemade fertilizers from kitchen scraps — banana peels eggshells coffee grounds and more for explosive plant growth",
    "The ultimate guide to seed saving — how to collect store and grow your own seeds year after year for a sustainable garden",
    "How to build a worm composting bin for under 20 dollars — vermicomposting tutorial that turns food waste into garden gold",
    "Complete guide to growing mushrooms at home — from spore to harvest using coffee grounds and simple containers",
    "How to naturally purify indoor air with houseplants — the best NASA-approved plants for removing toxins from your home",
    "DIY natural lip balm and body butter recipes — simple homemade skincare with shea butter coconut oil and essential oils",
    "The beginner guide to fermenting kimchi at home — traditional Korean recipe with modern tips for gut health and immunity",
    "How to build a simple rain barrel system for free garden water — collect hundreds of gallons from your roof naturally",
]


async def generate_content(topic: str) -> dict:
    from llm_content import generate_quality_post
    pack = await asyncio.to_thread(generate_quality_post, topic=topic, score=9.0)
    if not pack or not pack.get("content_formatted"):
        raise ValueError(f"Failed to generate content for: {topic[:60]}")
    return pack


async def publish_tiktok(content_pack: dict, account_id: str, label: str) -> dict:
    from web.app import _prepare_tiktok_content
    from integrations.publer_publisher import PublerPublisher
    pc = await _prepare_tiktok_content(content_pack, "tiktok")
    pc["account_ids"] = [account_id]
    pc["_account_label"] = label
    pub = PublerPublisher()
    r = await pub.publish(pc)
    return r if isinstance(r, dict) else {"success": False, "error": str(r)}


async def publish_facebook(content_pack: dict) -> dict:
    from web.app import _prepare_facebook_content
    from integrations.publer_publisher import PublerPublisher
    pc = await _prepare_facebook_content(content_pack)
    if not pc:
        return {"success": False, "error": "FB prepare empty"}
    pc["account_ids"] = [FACEBOOK]
    pub = PublerPublisher()
    r = await pub.publish(pc)
    return r if isinstance(r, dict) else {"success": False, "error": str(r)}


async def publish_pinterest(content_pack: dict) -> dict:
    from web.app import _prepare_pinterest_content
    from integrations.publer_publisher import PublerPublisher
    pc = await _prepare_pinterest_content(content_pack)
    if not pc:
        return {"success": False, "error": "Pinterest no image"}
    pc["account_ids"] = [PINTEREST]
    pub = PublerPublisher()
    r = await pub.publish(pc)
    return r if isinstance(r, dict) else {"success": False, "error": str(r)}


async def publish_all(content_pack: dict, idx: int) -> dict:
    results = {}
    for name, fn in [
        ("tiktok_1", lambda cp: publish_tiktok(cp, TIKTOK_1, "The Rike Root Stories")),
        ("tiktok_2", lambda cp: publish_tiktok(cp, TIKTOK_2, "The Rike Stories")),
        ("facebook", publish_facebook),
        ("pinterest", publish_pinterest),
    ]:
        print(f"  [{idx}] {name}...", end=" ", flush=True)
        try:
            r = await fn(content_pack)
            ok = r.get("success", False)
            results[name] = r
            print(f"{'OK' if ok else 'FAIL: ' + r.get('error', '')[:60]}")
        except Exception as e:
            results[name] = {"success": False, "error": str(e)[:100]}
            print(f"ERROR: {str(e)[:60]}")
        await asyncio.sleep(2)
    return results


async def main():
    print("=" * 70)
    print(f"ViralOps — Multi-Platform Batch Round 2 ({len(TOPICS)} topics)")
    print("=" * 70)
    total_ok = 0
    total_fail = 0

    for i, topic in enumerate(TOPICS, 1):
        print(f"\n{'─'*60}")
        print(f"[{i}/{len(TOPICS)}] {topic[:75]}...")
        print(f"{'─'*60}")
        print(f"  Generating...", end=" ", flush=True)
        t0 = time.time()
        try:
            cp = await generate_content(topic)
            chars = len(cp.get("content_formatted", ""))
            score = cp.get("_review_score", 0)
            print(f"OK ({chars}ch, {score:.1f}/10, {time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"FAIL: {e}")
            total_fail += 4
            continue

        results = await publish_all(cp, i)
        for _, r in results.items():
            if r.get("success"):
                total_ok += 1
            else:
                total_fail += 1

        if i < len(TOPICS):
            print(f"  Waiting 8s...")
            await asyncio.sleep(8)

    print(f"\n{'='*70}")
    print(f"DONE: {total_ok} OK / {total_fail} FAIL out of {len(TOPICS)*4} total")
    print(f"{'='*70}")

if __name__ == "__main__":
    asyncio.run(main())
