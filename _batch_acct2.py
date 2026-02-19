"""Post 4 bai to SECOND TikTok account: The Rike Stories (69951ea30c4677f27c12d98c)."""
import sys, os, asyncio
from dotenv import load_dotenv
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv(override=True)

TIKTOK_ACCOUNT_2 = "69951ea30c4677f27c12d98c"  # The Rike Stories

TOPICS = [
    "How do you make sourdough bread starter from scratch with just flour and water?",
    "What are the easiest vegetables to grow in winter even if you have never gardened before?",
    "How do you make your own beeswax food wraps to replace plastic wrap?",
    "What is the simplest way to collect and use rainwater for your garden?",
]

async def publish_one(topic, idx):
    print(f"\n{'='*60}")
    print(f"  POST {idx+1}/{len(TOPICS)} -> The Rike Stories")
    print(f"  Topic: {topic[:60]}...")
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

    # Force to second account
    pub["account_ids"] = [TIKTOK_ACCOUNT_2]

    from integrations.publer_publisher import PublerPublisher
    r = await PublerPublisher().publish(pub)
    ok = r.get("success", False)
    print(f"  {'OK' if ok else 'FAIL'}: {r.get('job_id', r.get('error', '?'))}")
    return ok

async def main():
    results = []
    for i, topic in enumerate(TOPICS):
        if i > 0:
            print(f"\n  Waiting 65s between posts...")
            await asyncio.sleep(65)
        ok = await publish_one(topic, i)
        results.append((topic[:50], ok))

    print(f"\n{'='*60}")
    print(f"  SUMMARY — The Rike Stories")
    print(f"{'='*60}")
    for t, ok in results:
        print(f"  {'OK' if ok else 'FAIL'} | {t}")
    print(f"\n  Total: {sum(1 for _,ok in results if ok)}/{len(results)} published")

if __name__ == "__main__":
    asyncio.run(main())
