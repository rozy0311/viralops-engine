"""
ViralOps Engine — Smart LLM Content Pipeline
=============================================
Multi-provider cascade with self-review (EMADS-PR pattern).

Providers (cost-aware order):
  1. Gemini 2.5 Flash (free tier, 15 RPM)
  2. GitHub Models / gpt-4o-mini (free via Copilot)
  3. Perplexity / sonar (has web search — great for trending content)
  4. OpenAI / gpt-4o-mini (paid fallback)

Following Training Multi-Agent principles:
  - Cost-Aware Planning (doc 07)
  - Security Defense (doc 04) — never hardcode keys
  - ReconcileGPT pattern — self-review before publishing
"""

import os
import json
import time
import httpx
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=True)

# ═══════════════════════════════════════════════════════════════
# PROVIDER REGISTRY
# ═══════════════════════════════════════════════════════════════

@dataclass
class ProviderResult:
    """Result from an LLM call."""
    text: str
    provider: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0
    cost_usd: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    api_key_env: str
    base_url: str
    model: str
    cost_per_1k_input: float = 0.0    # USD per 1K input tokens
    cost_per_1k_output: float = 0.0   # USD per 1K output tokens
    max_tokens: int = 4000
    is_openai_compatible: bool = True  # Uses OpenAI chat completions API
    is_gemini: bool = False            # Uses Google genai SDK


# Provider cascade — cheapest working first
PROVIDERS = [
    ProviderConfig(
        name="gemini",
        api_key_env="GEMINI_API_KEY",
        base_url="",  # Uses SDK
        model="gemini-2.5-flash",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        is_openai_compatible=False,
        is_gemini=True,
    ),
    ProviderConfig(
        name="github_models",
        api_key_env="GH_MODELS_API_KEY",
        base_url="https://models.github.ai/inference/chat/completions",
        model="openai/gpt-4o-mini",
        cost_per_1k_input=0.0,  # Free via Copilot
        cost_per_1k_output=0.0,
    ),
    ProviderConfig(
        name="perplexity",
        api_key_env="PPLX_API_KEY",
        base_url="https://api.perplexity.ai/chat/completions",
        model="sonar",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.001,
    ),
    ProviderConfig(
        name="openai",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1/chat/completions",
        model="gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
    ),
]


# ═══════════════════════════════════════════════════════════════
# CORE LLM CASCADE
# ═══════════════════════════════════════════════════════════════

def call_llm(
    prompt: str,
    system: str = "",
    max_tokens: int = 4000,
    temperature: float = 0.7,
    providers: Optional[List[str]] = None,
) -> ProviderResult:
    """
    Call LLM using cascade — tries each provider until one works.
    
    Args:
        prompt: User message
        system: System prompt
        max_tokens: Max output tokens
        temperature: Creativity (0-1)
        providers: Optional list of provider names to try (default: all)
    
    Returns:
        ProviderResult with text and metadata
    """
    provider_order = os.environ.get("LLM_PROVIDER_ORDER", "gemini,github_models,perplexity,openai")
    allowed = providers or provider_order.split(",")
    
    errors = []
    
    for pconfig in PROVIDERS:
        if pconfig.name not in allowed:
            continue
            
        api_key = os.environ.get(pconfig.api_key_env, "")
        if not api_key:
            continue
        
        # Check if provider is disabled
        if pconfig.name == "openai" and os.environ.get("DISABLE_OPENAI", "").lower() == "true":
            continue
            
        start_time = time.time()
        
        try:
            if pconfig.is_gemini:
                result = _call_gemini(pconfig, api_key, prompt, system, max_tokens, temperature)
            else:
                result = _call_openai_compatible(pconfig, api_key, prompt, system, max_tokens, temperature)
            
            result.latency_ms = (time.time() - start_time) * 1000
            
            if result.success:
                print(f"  [LLM] {pconfig.name}/{pconfig.model} — OK ({result.latency_ms:.0f}ms)")
                return result
            else:
                errors.append(f"{pconfig.name}: {result.error}")
                print(f"  [LLM] {pconfig.name} — FAIL: {result.error[:100]}")
                
        except Exception as e:
            errors.append(f"{pconfig.name}: {str(e)[:200]}")
            print(f"  [LLM] {pconfig.name} — ERROR: {str(e)[:100]}")
    
    # All providers failed
    return ProviderResult(
        text="",
        provider="none",
        model="none",
        success=False,
        error=f"All providers failed: {'; '.join(errors)}",
    )


def _call_gemini(
    config: ProviderConfig,
    api_key: str,
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
) -> ProviderResult:
    """Call Google Gemini via genai SDK."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return ProviderResult(text="", provider=config.name, model=config.model,
                             success=False, error="google-genai not installed")
    
    client = genai.Client(api_key=api_key)
    
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    
    resp = client.models.generate_content(
        model=config.model,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    
    text = resp.text.strip() if resp.text else ""
    
    # Estimate tokens (Gemini doesn't always return usage)
    est_tokens = len(text.split()) * 1.3
    
    return ProviderResult(
        text=text,
        provider=config.name,
        model=config.model,
        tokens_used=int(est_tokens),
        success=bool(text),
        error="" if text else "Empty response",
    )


def _call_openai_compatible(
    config: ProviderConfig,
    api_key: str,
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
) -> ProviderResult:
    """Call any OpenAI-compatible API (GitHub Models, Perplexity, OpenAI)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": config.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    r = httpx.post(config.base_url, headers=headers, json=payload, timeout=60)
    
    if r.status_code != 200:
        return ProviderResult(
            text="", provider=config.name, model=config.model,
            success=False, error=f"HTTP {r.status_code}: {r.text[:200]}",
        )
    
    data = r.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    usage = data.get("usage", {})
    tokens = usage.get("total_tokens", 0) or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
    
    # Calculate cost
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    cost = (input_tokens / 1000 * config.cost_per_1k_input) + (output_tokens / 1000 * config.cost_per_1k_output)
    
    return ProviderResult(
        text=text,
        provider=config.name,
        model=config.model,
        tokens_used=tokens,
        cost_usd=cost,
        success=bool(text),
        error="" if text else "Empty response",
    )


# ═══════════════════════════════════════════════════════════════
# CONTENT GENERATION PIPELINE (EMADS-PR PATTERN)
# ═══════════════════════════════════════════════════════════════

# System prompt for content generation (CTO Agent role)
CONTENT_SYSTEM = """You are a TikTok content specialist for plant-based, homesteading, and urban farming micro-niches.
Target audience: US-based 18-45, apartment/small-space dwellers, budget-conscious.
Channels: @therikerootstories (plant-based), @agrinomadsvietnam (farming), @therikecom (AI/tech).

RULES:
- Content MUST be educational + actionable (steps people can follow TODAY)
- Include specific numbers (costs, timeframes, quantities)
- Use conversational, witty tone — NOT corporate
- Focus on micro-niche topics that are underserved on TikTok
- NEVER generic fluff — every post must teach something specific"""

# System prompt for self-review (ReconcileGPT role)
REVIEW_SYSTEM = """You are ReconcileGPT — a content quality review agent.
Your job: analyze content for quality, accuracy, and TikTok fitness.

Score 1-10 on each criteria:
1. UNIQUENESS — Is this teaching something most TikTok creators WON'T cover?
2. ACTIONABILITY — Can viewer do this TODAY with items they already have?
3. ACCURACY — Are the facts/numbers correct?
4. HOOK — Would the first line stop someone from scrolling?
5. MICRO-NICHE FIT — Does this fit plant-based/homesteading/urban farming?

Output JSON: {"scores": {"uniqueness": N, "actionability": N, "accuracy": N, "hook": N, "niche_fit": N}, "avg": N.N, "pass": true/false, "feedback": "...", "improved_title": "..."}
Pass threshold: avg >= 7.0"""


def generate_content_pack(topic: str, score: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Generate a full content pack using LLM cascade.
    
    This follows EMADS-PR:
      Step 1 (CTO): Generate content
      Step 2 (ReconcileGPT): Self-review and improve
      Step 3 (COO): Format for publishing
    
    Args:
        topic: The micro-niche topic/question
        score: Niche Hunter score (0-10)
    
    Returns:
        Content pack dict ready for publish_microniche.py
    """
    print(f"\n  [PIPELINE] Generating content for: {topic[:60]}...")
    print(f"  [PIPELINE] Niche score: {score}")
    
    # ── Step 1: CTO Agent — Generate content ──
    gen_prompt = f"""Create a TikTok photo post content pack for this topic:

TOPIC: {topic}
NICHE SCORE: {score}/10

Generate a complete content pack in this EXACT JSON format:
{{
    "title": "Catchy keyword-rich title — with dash separator for clarity",
    "pain_point": "The specific problem/question this solves (1 sentence)",
    "audiences": ["Audience 1", "Audience 2", "Audience 3"],
    "steps": [
        "Step 1: Specific actionable instruction with numbers/costs",
        "Step 2: Follow-up action with expected result"
    ],
    "result": "What they'll achieve — specific outcome with numbers",
    "hashtags": ["#hashtag1", "#hashtag2", "#hashtag3", "#hashtag4", "#hashtag5"],
    "image_title": "Short Title (max 4 words)",
    "image_subtitle": "Subtitle (max 5 words)",
    "image_steps": "Step1 • Step2 • Step3",
    "colors": [[60, 80, 40], [120, 160, 80]]
}}

IMPORTANT:
- Title MUST be specific, not generic (include numbers, methods, or surprise elements)
- Steps MUST include specific costs ($), timeframes (days/weeks), or quantities
- Result MUST be measurable
- Hashtags: 2 broad + 2 niche-specific + 1 trending
- Colors: dark bg color + lighter accent, as RGB arrays
- Output ONLY valid JSON, no markdown, no explanation"""

    result = call_llm(gen_prompt, system=CONTENT_SYSTEM, max_tokens=1500, temperature=0.8)
    
    if not result.success:
        print(f"  [PIPELINE] Generation FAILED: {result.error[:100]}")
        return None
    
    # Parse JSON
    pack = _extract_json(result.text)
    if not pack:
        print(f"  [PIPELINE] JSON parse FAILED from {result.provider}")
        print(f"  [PIPELINE] Raw: {result.text[:300]}")
        return None
    
    pack["_source"] = f"ai_{result.provider}"
    pack["_niche_score"] = score
    pack["_gen_provider"] = result.provider
    pack["_gen_model"] = result.model
    pack["_gen_tokens"] = result.tokens_used
    pack["_gen_cost"] = result.cost_usd
    
    print(f"  [PIPELINE] Generated: {pack.get('title', '?')[:60]}")
    print(f"  [PIPELINE] Provider: {result.provider}/{result.model}")
    
    # ── Step 2: ReconcileGPT — Self-review ──
    review = review_content(pack)
    if review:
        pack["_review_score"] = review.get("avg", 0)
        pack["_review_pass"] = review.get("pass", False)
        pack["_review_feedback"] = review.get("feedback", "")
        pack["_review_provider"] = review.get("_provider", "")
        
        # Use improved title if review suggested one
        if review.get("improved_title") and review.get("avg", 0) < 8:
            pack["title"] = review["improved_title"]
            print(f"  [PIPELINE] Title improved: {pack['title'][:60]}")
        
        if not review.get("pass"):
            print(f"  [PIPELINE] Review FAILED (score={review.get('avg', 0):.1f})")
            print(f"  [PIPELINE] Feedback: {review.get('feedback', '')[:100]}")
            # Still return the pack but marked as needs-improvement
            pack["_needs_improvement"] = True
    
    # ── Step 3: COO Agent — Format for publishing ──
    # Convert colors from list to tuple if needed
    if "colors" in pack:
        colors = pack["colors"]
        if isinstance(colors, list) and len(colors) == 2:
            pack["colors"] = (tuple(colors[0]), tuple(colors[1]))
        else:
            pack["colors"] = ((60, 80, 40), (120, 160, 80))
    else:
        pack["colors"] = ((60, 80, 40), (120, 160, 80))
    
    return pack


def review_content(pack: Dict[str, Any]) -> Optional[Dict]:
    """
    ReconcileGPT — Review content quality before publishing.
    Uses a DIFFERENT provider than generation for unbiased review.
    """
    print(f"  [REVIEW] Reviewing: {pack.get('title', '?')[:50]}...")
    
    review_prompt = f"""Review this TikTok content pack for quality:

TITLE: {pack.get('title', '')}
PAIN POINT: {pack.get('pain_point', '')}
AUDIENCES: {pack.get('audiences', [])}
STEPS: {json.dumps(pack.get('steps', []))}
RESULT: {pack.get('result', '')}
HASHTAGS: {pack.get('hashtags', [])}

Score each criteria 1-10 and output ONLY valid JSON:
{{"scores": {{"uniqueness": N, "actionability": N, "accuracy": N, "hook": N, "niche_fit": N}}, "avg": N.N, "pass": true/false, "feedback": "1-2 sentences of improvement suggestions", "improved_title": "better title if score < 8, else same title"}}"""
    
    # Try to use a different provider for review (avoid self-bias)
    gen_provider = pack.get("_gen_provider", "")
    review_providers = ["github_models", "perplexity", "gemini", "openai"]
    # Move gen provider to end
    if gen_provider in review_providers:
        review_providers.remove(gen_provider)
        review_providers.append(gen_provider)
    
    result = call_llm(review_prompt, system=REVIEW_SYSTEM, max_tokens=500, temperature=0.3, providers=review_providers)
    
    if not result.success:
        print(f"  [REVIEW] Review FAILED: {result.error[:100]}")
        return None
    
    review = _extract_json(result.text)
    if review:
        review["_provider"] = result.provider
        avg = review.get("avg", 0)
        passed = review.get("pass", avg >= 7.0)
        review["pass"] = passed
        print(f"  [REVIEW] Score: {avg:.1f}/10 — {'PASS' if passed else 'FAIL'} (by {result.provider})")
    
    return review


def generate_from_niche_hunter(top_n: int = 5) -> Optional[Dict[str, Any]]:
    """
    Pull top-scored topic from niche_hunter.db and generate AI content.
    
    Combines:
      - Niche Hunter scoring (data-driven topic selection)
      - LLM cascade (multi-provider content generation)
      - ReconcileGPT (self-review)
    """
    import sqlite3
    
    db_path = os.path.join(os.path.dirname(__file__), "niche_hunter.db")
    if not os.path.exists(db_path):
        print(f"  [NICHE] No niche_hunter.db found")
        return None
    
    # Get already-published titles for dedup
    published_titles = set()
    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from web.app import get_db_safe, init_db
        init_db()
        with get_db_safe() as conn:
            rows = conn.execute("SELECT title FROM posts WHERE status = 'published'").fetchall()
            published_titles = {r[0] for r in rows}
    except Exception:
        pass
    
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT topic, final_score, niche, source FROM niche_scores "
        "ORDER BY final_score DESC LIMIT ?",
        (top_n * 3,),  # Get extra for dedup
    ).fetchall()
    conn.close()
    
    if not rows:
        print(f"  [NICHE] No scored topics in DB")
        return None
    
    # Filter out already-published (rough match on topic substring)
    for topic, score, niche, source in rows:
        # Simple dedup: check if topic keywords appear in any published title
        topic_words = set(topic.lower().split()[:5])
        duplicate = False
        for title in published_titles:
            title_words = set(title.lower().split()[:5])
            if len(topic_words & title_words) >= 3:
                duplicate = True
                break
        
        if not duplicate:
            print(f"  [NICHE] Selected: {topic[:60]} (score={score:.2f})")
            return generate_content_pack(topic, score)
    
    # All top topics already published — use first anyway
    topic, score, _, _ = rows[0]
    print(f"  [NICHE] All top topics used, reusing: {topic[:60]}")
    return generate_content_pack(topic, score)


# ═══════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════

def _extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response (handles markdown code blocks)."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code block
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    
    # Try finding JSON object pattern
    import re
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    return None


def test_providers():
    """Test all providers and print status."""
    print("=" * 60)
    print("LLM PROVIDER CASCADE TEST")
    print("=" * 60)
    
    for pconfig in PROVIDERS:
        api_key = os.environ.get(pconfig.api_key_env, "")
        disabled = pconfig.name == "openai" and os.environ.get("DISABLE_OPENAI", "").lower() == "true"
        
        status = "NO KEY" if not api_key else "DISABLED" if disabled else "TESTING..."
        print(f"\n  [{pconfig.name}] {pconfig.model} — {status}")
        
        if api_key and not disabled:
            result = call_llm(
                "Say hello in exactly 5 words.",
                providers=[pconfig.name],
                max_tokens=50,
                temperature=0,
            )
            if result.success:
                print(f"    OK: {result.text[:80]}")
            else:
                print(f"    FAIL: {result.error[:100]}")
    
    print("\n" + "=" * 60)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    
    if cmd == "test":
        test_providers()
    
    elif cmd == "generate":
        topic = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "How to grow mushrooms in apartment with no sunlight?"
        pack = generate_content_pack(topic, score=8.0)
        if pack:
            print(f"\n{'=' * 60}")
            print(f"CONTENT PACK GENERATED")
            print(f"{'=' * 60}")
            print(json.dumps(pack, indent=2, default=str))
    
    elif cmd == "niche":
        pack = generate_from_niche_hunter(top_n=10)
        if pack:
            print(f"\n{'=' * 60}")
            print(f"NICHE CONTENT PACK")
            print(f"{'=' * 60}")
            print(json.dumps(pack, indent=2, default=str))
    
    else:
        print("Usage: python llm_content.py [test|generate|niche] [topic...]")
