"""Batch publish 4 more TikTok posts with clump-proof format."""
import sys, os, asyncio
from dotenv import load_dotenv
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv(override=True)
TIKTOK_ACCOUNT = "698c95e5b1ab790def1352c1"

TOPICS = [
    "How do you make your own natural cleaning products with vinegar and baking soda?",
    "What is the cheapest way to start composting in a small apartment balcony?",
    "How do you grow microgreens on your kitchen counter with almost no effort?",
    "What are the best ways to preserve fresh herbs so they last for months?",
]

async def publish_one(topic, idx, total):
    print(f"\n{'='*60}")
    print(f"  POST {idx+1}/{total}: {topic[:60]}...")
    print(f"{'='*60}")

    from llm_content import generate_quality_post
    pack = generate_quality_post(topic=topic, score=9.0)
    if not pack:
        print(f"  FAIL: generation failed"); return False

    orig = pack.get("content_formatted", "")
    score = pack.get("_review_score", 0)
    print(f"  Content: {len(orig)} chars, {score}/10")

    from web.app import _prepare_tiktok_content
    pub = await _prepare_tiktok_content(pack)
    caption = pub.get("caption", "")
    print(f"  Reformatted: {len(caption)} chars")

    if not pub.get("account_ids"):
        pub["account_ids"] = [TIKTOK_ACCOUNT]

    from integrations.publer_publisher import PublerPublisher
    r = await PublerPublisher().publish(pub)
    ok = r.get("success", False)
    print(f"  {'OK' if ok else 'FAIL'}: {r.get('job_id', r.get('error', '?'))}")
    return ok

async def main():
    results = []
    for i, topic in enumerate(TOPICS):
        if i > 0:
            print(f"\n  Waiting 65s...")
            await asyncio.sleep(65)
        ok = await publish_one(topic, i, len(TOPICS))
        results.append((topic[:50], ok))

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for t, ok in results:
        print(f"  {'OK' if ok else 'FAIL'} | {t}")
    print(f"  Total: {sum(1 for _,ok in results if ok)}/{len(results)}")

if __name__ == "__main__":
    asyncio.run(main())
