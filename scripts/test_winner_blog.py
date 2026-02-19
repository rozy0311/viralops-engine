"""Quick test: generate 1 winning blog style post."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(override=True)

from llm_content import generate_quality_post

topic = "Soil prep tips for Zone 6a spring gardens — how to fix heavy clay soil without making it worse and when to actually start planting"
pack = generate_quality_post(topic=topic, score=9.0)
if pack:
    title = pack.get("title", "?")
    chars = len(pack.get("content_formatted", ""))
    score = pack.get("_review_score", 0)
    tags = pack.get("hashtags", [])
    provider = pack.get("_gen_provider", "?")
    model = pack.get("_gen_model", "?")
    print(f"\n{'='*60}")
    print(f"  RESULT")
    print(f"{'='*60}")
    print(f"  Title: {title}")
    print(f"  Chars: {chars}")
    print(f"  Score: {score:.1f}/10")
    print(f"  Hashtags: {tags}")
    print(f"  Provider: {provider}/{model}")
    print(f"\n--- CONTENT PREVIEW (first 1000 chars) ---")
    print(pack.get("content_formatted", "")[:1000])
    print(f"\n--- END PREVIEW ---")
else:
    print("FAILED to generate")
