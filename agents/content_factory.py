"""
Content Factory Agent — CTO Agent (EMADS-PR v1.0)
Multi-provider LLM: Gemini (primary) → GitHub Models (fallback) → OpenAI (last resort).
Generates content packs: title + body + universal caption + hashtags + image prompt.

v3.0 — Multi-LLM Provider:
- Gemini 2.0 Flash (free/cheap, fast, primary)
- GitHub Models GPT-4.1 (free via Copilot, fallback)
- OpenAI GPT-4.1 (paid, last resort, can be disabled)
- Respects LLM_PROVIDER_ORDER and DISABLE_OPENAI from .env
- Template-based fallback when ALL LLMs unavailable

Fixed in v2.0:
- Extracts hooks/personas from sub_niches[] (correct nesting)
- Integrates 7-layer hashtag matrix auto-generation
- Loads caption_templates.json for fallback generation
- Universal Caption Template matches spec formula
- Loads content_transforms.json for platform adaptation
"""
import os
import json
import asyncio
from typing import Any, Optional
from datetime import datetime

import structlog
import yaml

logger = structlog.get_logger()

# ── Cost tracking ──
_token_usage = {"prompt": 0, "completion": 0, "total_cost": 0.0}

# ── Model pricing (per 1K tokens) ──
MODEL_PRICING = {
    "gpt-4.1": {"input": 0.002, "output": 0.008},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gemini-2.0-flash": {"input": 0.0, "output": 0.0},  # Free tier
    "gemini-2.5-flash": {"input": 0.00015, "output": 0.0006},
    "github/gpt-4.1": {"input": 0.0, "output": 0.0},  # Free via Copilot
}

# ── Paths ──
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
CAPTION_TEMPLATES_PATH = os.path.join(TEMPLATES_DIR, "caption_templates.json")
CONTENT_TRANSFORMS_PATH = os.path.join(TEMPLATES_DIR, "content_transforms.json")
NICHES_YAML_PATH = os.path.join(CONFIG_DIR, "niches.yaml")


# ════════════════════════════════════════════════════════════════
# Multi-Provider LLM Client — Gemini → GitHub Models → OpenAI
# ════════════════════════════════════════════════════════════════

def _get_provider_order() -> list[str]:
    """Get LLM provider priority from env. Default: gemini first."""
    order = os.getenv("LLM_PROVIDER_ORDER", "gemini,github_models,openai")
    return [p.strip() for p in order.split(",") if p.strip()]


def _get_gemini_client():
    """Lazy-load Gemini client (google.genai SDK — new unified API)."""
    try:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return None
        client = genai.Client(api_key=api_key)
        return client
    except ImportError:
        logger.debug("content_factory.no_gemini_sdk", msg="google-genai not installed")
        return None


def _get_github_models_client():
    """Lazy-load GitHub Models client (OpenAI-compatible API)."""
    try:
        from openai import OpenAI
        api_key = os.getenv("GH_MODELS_API_KEY", "")
        api_base = os.getenv("GH_MODELS_API_BASE", "https://models.github.ai/inference")
        if not api_key:
            return None
        return OpenAI(api_key=api_key, base_url=api_base)
    except ImportError:
        logger.debug("content_factory.no_openai_sdk", msg="openai package not installed for GitHub Models")
        return None


def _get_openai_client():
    """Lazy-load OpenAI client (disabled by default via DISABLE_OPENAI)."""
    if os.getenv("DISABLE_OPENAI", "false").lower() in ("true", "1", "yes"):
        return None
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return None
        return OpenAI(api_key=api_key)
    except ImportError:
        return None


def _llm_generate_json(system_prompt: str, user_prompt: str, temperature: float = 0.8) -> tuple[Optional[dict], str]:
    """
    Call LLM with provider cascade: Gemini → GitHub Models → OpenAI.
    Returns (parsed_json_dict, provider_name) or (None, "none").
    """
    providers = _get_provider_order()

    for provider in providers:
        try:
            if provider == "gemini":
                result, name = _call_gemini(system_prompt, user_prompt, temperature)
                if result is not None:
                    return result, name

            elif provider == "github_models":
                result, name = _call_github_models(system_prompt, user_prompt, temperature)
                if result is not None:
                    return result, name

            elif provider == "openai":
                result, name = _call_openai(system_prompt, user_prompt, temperature)
                if result is not None:
                    return result, name

        except Exception as e:
            logger.warning("content_factory.provider_failed", provider=provider, error=str(e))
            continue

    return None, "none"


def _call_gemini(system_prompt: str, user_prompt: str, temperature: float) -> tuple[Optional[dict], str]:
    """Call Gemini API via google.genai SDK. Returns (parsed_json, model_name) or (None, '')."""
    client = _get_gemini_client()
    if not client:
        return None, ""

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    try:
        from google.genai import types

        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=2000,
                response_mime_type="application/json",
            ),
        )

        # Track usage
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            _track_cost(model_name, {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
            })

        text = response.text.strip()
        # Gemini sometimes wraps JSON in ```json ... ```
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(text)
        logger.info("content_factory.gemini_success", model=model_name)
        return parsed, f"gemini/{model_name}"

    except Exception as e:
        logger.warning("content_factory.gemini_error", model=model_name, error=str(e))
        return None, ""


def _call_github_models(system_prompt: str, user_prompt: str, temperature: float) -> tuple[Optional[dict], str]:
    """Call GitHub Models (OpenAI-compatible). Returns (parsed_json, model_name) or (None, '')."""
    client = _get_github_models_client()
    if not client:
        return None, ""

    model_name = os.getenv("GH_MODELS_MODEL", "openai/gpt-4.1")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )

        if response.usage:
            _track_cost(model_name, {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            })

        parsed = json.loads(response.choices[0].message.content)
        logger.info("content_factory.github_models_success", model=model_name)
        return parsed, f"github_models/{model_name}"

    except Exception as e:
        logger.warning("content_factory.github_models_error", model=model_name, error=str(e))
        return None, ""


def _call_openai(system_prompt: str, user_prompt: str, temperature: float) -> tuple[Optional[dict], str]:
    """Call OpenAI API (last resort). Returns (parsed_json, model_name) or (None, '')."""
    client = _get_openai_client()
    if not client:
        return None, ""

    model_name = "gpt-4.1-mini"  # Cost-aware default
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )

        if response.usage:
            _track_cost(model_name, {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            })

        parsed = json.loads(response.choices[0].message.content)
        logger.info("content_factory.openai_success", model=model_name)
        return parsed, f"openai/{model_name}"

    except Exception as e:
        logger.warning("content_factory.openai_error", model=model_name, error=str(e))
        return None, ""


def _select_model(budget_remaining_pct: float = 100.0) -> str:
    """Cost-aware model selection. With Gemini free tier, budget is less of a concern."""
    if budget_remaining_pct > 5:
        return "auto"  # Use provider cascade
    else:
        return "fallback"  # No LLM, use templates


def _track_cost(model: str, usage: dict):
    """Track token usage and cost."""
    pricing = MODEL_PRICING.get(model, {"input": 0.001, "output": 0.004})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cost = (prompt_tokens / 1000 * pricing["input"]) + (completion_tokens / 1000 * pricing["output"])
    _token_usage["prompt"] += prompt_tokens
    _token_usage["completion"] += completion_tokens
    _token_usage["total_cost"] += cost
    logger.info("content_factory.cost", model=model, tokens=prompt_tokens + completion_tokens, cost=f"${cost:.4f}", total=f"${_token_usage['total_cost']:.4f}")


def get_token_usage() -> dict:
    """Return current token usage stats."""
    return dict(_token_usage)


# ── Platform specs ──
PLATFORM_SPECS = {
    "tiktok":    {"max_caption": 2200, "max_hashtags": 8,  "style": "casual, viral, hook-first"},
    "instagram": {"max_caption": 2200, "max_hashtags": 30, "style": "visual, lifestyle, aspirational"},
    "facebook":  {"max_caption": 5000, "max_hashtags": 10, "style": "conversational, community"},
    "youtube":   {"max_caption": 5000, "max_hashtags": 15, "style": "informative, SEO-optimized"},
    "pinterest": {"max_caption": 500,  "max_hashtags": 20, "style": "inspirational, how-to, searchable"},
    "linkedin":  {"max_caption": 3000, "max_hashtags": 5,  "style": "professional, thought-leadership"},
    "twitter":   {"max_caption": 280,  "max_hashtags": 3,  "style": "punchy, news-like, threaded"},
    "reddit":    {"max_caption": 40000,"max_hashtags": 0,  "style": "authentic, community-first, no-promotion"},
    "medium":    {"max_caption": 50000,"max_hashtags": 5,  "style": "long-form, editorial, structured"},
    "tumblr":    {"max_caption": 50000,"max_hashtags": 30, "style": "creative, personal, aesthetic"},
    "shopify_blog": {"max_caption": 50000, "max_hashtags": 0, "style": "SEO, product-focused, educational"},
}

# ── Universal Caption Template (from Micro Niche Blogs spec) ──
# Formula: [LOCATION] [SEASON] [PAIN POINT] → [AUDIENCE 1-3]? → [PRODUCT/SOLUTION] 3 steps → CTA → Micro keywords → Hashtags
UNIVERSAL_CAPTION_TEMPLATE = """[{location}] [{season}] {pain_point} {season_emoji}
{audience_1}? {audience_2}? {audience_3}?

{product_solution} in 3 minutes:
• Step 1: {step_1}
• Step 2: {step_2}
• Result: {result}

Full tutorial pinned on my profile! 👇

Micro keywords: {micro_keywords}

{hashtags}""".strip()

# Simpler fallback template when not all fields available
SIMPLE_CAPTION_TEMPLATE = """{hook}

{pain_point}

{solution_steps}

{cta}

{hashtags}""".strip()


def _load_caption_templates() -> dict:
    """Load caption_templates.json for template-based generation."""
    try:
        if os.path.exists(CAPTION_TEMPLATES_PATH):
            with open(CAPTION_TEMPLATES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("caption_templates.load_error", error=str(e))
    return {}


def _load_content_transforms() -> dict:
    """Load content_transforms.json for platform adaptation rules."""
    try:
        if os.path.exists(CONTENT_TRANSFORMS_PATH):
            with open(CONTENT_TRANSFORMS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("content_transforms.load_error", error=str(e))
    return {}


def _load_niches_yaml() -> dict:
    """Load niches.yaml for rich sub-niche data."""
    try:
        if os.path.exists(NICHES_YAML_PATH):
            with open(NICHES_YAML_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("niches_yaml.load_error", error=str(e))
    return {}


def _extract_niche_data(niche_config: dict) -> dict:
    """
    Extract hooks, personas, pains, desires from niche_config.
    Handles BOTH flat format AND nested sub_niches[] format.
    
    Returns:
        dict with flattened: hooks, personas, pains, desires, sub_niche_names
    """
    result = {
        "hooks": [],
        "personas": [],
        "pains": [],
        "desires": [],
        "sub_niche_names": [],
        "display_name": niche_config.get("display_name", niche_config.get("name", "general")),
    }

    # ── Flat format (direct hooks/personas keys) ──
    if "hooks" in niche_config and isinstance(niche_config["hooks"], list):
        result["hooks"] = niche_config["hooks"]
    if "personas" in niche_config and isinstance(niche_config["personas"], list):
        result["personas"] = niche_config["personas"]
    if "persona" in niche_config and isinstance(niche_config["persona"], str):
        result["personas"] = [niche_config["persona"]]

    # ── Nested sub_niches[] format (from niches.yaml) ──
    sub_niches = niche_config.get("sub_niches", [])
    for sub in sub_niches:
        # Hooks — array per sub-niche
        sub_hooks = sub.get("hooks", [])
        result["hooks"].extend(sub_hooks)

        # Persona — singular string per sub-niche
        persona = sub.get("persona", "")
        if persona and persona not in result["personas"]:
            result["personas"].append(persona)

        # Pain & Desire
        pain = sub.get("pain", "")
        desire = sub.get("desire", "")
        if pain:
            result["pains"].append(pain)
        if desire:
            result["desires"].append(desire)

        # Sub-niche name
        name = sub.get("name", sub.get("id", ""))
        if name:
            result["sub_niche_names"].append(name)

    # Deduplicate
    result["hooks"] = list(dict.fromkeys(result["hooks"]))
    result["personas"] = list(dict.fromkeys(result["personas"]))

    return result


# ════════════════════════════════════════════════════════════════
# Per-channel character limits (precise, algo-optimal)
# ════════════════════════════════════════════════════════════════
CHANNEL_CHAR_LIMITS = {
    "tiktok":       {"caption": 2200,  "title": 0,     "hashtags_in_caption": True,  "optimal_hashtags": 5},
    "instagram":    {"caption": 2200,  "title": 0,     "hashtags_in_caption": True,  "optimal_hashtags": 5},
    "facebook":     {"caption": 5000,  "title": 0,     "hashtags_in_caption": True,  "optimal_hashtags": 5},
    "youtube":      {"caption": 5000,  "title": 100,   "hashtags_in_caption": False, "optimal_hashtags": 5},
    "youtube_short":{"caption": 100,   "title": 100,   "hashtags_in_caption": True,  "optimal_hashtags": 3},
    "pinterest":    {"caption": 500,   "title": 100,   "hashtags_in_caption": True,  "optimal_hashtags": 5},
    "linkedin":     {"caption": 3000,  "title": 0,     "hashtags_in_caption": True,  "optimal_hashtags": 5},
    "twitter":      {"caption": 280,   "title": 0,     "hashtags_in_caption": True,  "optimal_hashtags": 3},
    "reddit":       {"caption": 40000, "title": 300,   "hashtags_in_caption": False, "optimal_hashtags": 0},
    "medium":       {"caption": 50000, "title": 200,   "hashtags_in_caption": False, "optimal_hashtags": 5},
    "tumblr":       {"caption": 50000, "title": 200,   "hashtags_in_caption": True,  "optimal_hashtags": 5},
    "shopify_blog": {"caption": 50000, "title": 200,   "hashtags_in_caption": False, "optimal_hashtags": 0},
    "threads":     {"caption": 500,   "title": 0,     "hashtags_in_caption": True,  "optimal_hashtags": 5},
    "bluesky":     {"caption": 300,   "title": 0,     "hashtags_in_caption": True,  "optimal_hashtags": 3},
    "mastodon":    {"caption": 500,   "title": 0,     "hashtags_in_caption": True,  "optimal_hashtags": 5},
    "quora":       {"caption": 50000, "title": 250,   "hashtags_in_caption": False, "optimal_hashtags": 0},
}


# ════════════════════════════════════════════════════════════════
# Smart Content Splitting — Cut at sentence boundary, never lose meaning
# ════════════════════════════════════════════════════════════════
import re as _re

_SENTENCE_ENDS = _re.compile(r'(?<=[.!?\n])\s+')
_PARAGRAPH_ENDS = _re.compile(r'\n\n+')


def smart_truncate(text: str, max_chars: int, preserve_meaning: bool = True) -> str:
    """
    Truncate text to max_chars WITHOUT losing meaning.

    Rules (from Gumloop training docs):
      1. NEVER cut mid-word
      2. NEVER cut mid-sentence if possible
      3. Prefer cutting at paragraph break > sentence end > word boundary
      4. If content is a list/steps, keep complete items (don't cut "Step 2" in half)
      5. Always end with a complete thought

    Args:
        text: Content to truncate
        max_chars: Maximum characters allowed by the channel
        preserve_meaning: If True, cuts at sentence boundary (may be shorter than max)
    """
    if not text or len(text) <= max_chars:
        return text

    if not preserve_meaning:
        # Hard cut at word boundary
        cut = text[:max_chars]
        last_space = cut.rfind(' ')
        return cut[:last_space] + '...' if last_space > max_chars * 0.5 else cut

    # Strategy 1: Try cutting at paragraph boundary
    paragraphs = _PARAGRAPH_ENDS.split(text)
    result = ""
    for para in paragraphs:
        candidate = (result + "\n\n" + para).strip() if result else para
        if len(candidate) <= max_chars:
            result = candidate
        else:
            break

    if result and len(result) >= max_chars * 0.3:
        return result

    # Strategy 2: Cut at sentence boundary
    sentences = _SENTENCE_ENDS.split(text)
    result = ""
    for sent in sentences:
        candidate = (result + " " + sent).strip() if result else sent
        if len(candidate) <= max_chars:
            result = candidate
        else:
            break

    if result and len(result) >= max_chars * 0.3:
        return result

    # Strategy 3: Last resort — word boundary
    cut = text[:max_chars]
    last_space = cut.rfind(' ')
    if last_space > max_chars * 0.5:
        return cut[:last_space] + '...'
    return cut


def smart_split_for_thread(text: str, chunk_size: int = 280) -> list[str]:
    """
    Split long content into thread-sized chunks (e.g., Twitter thread).
    Each chunk is a complete thought ending at sentence boundary.
    """
    sentences = _SENTENCE_ENDS.split(text)
    chunks = []
    current = ""

    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = sent if len(sent) <= chunk_size else sent[:chunk_size-3] + '...'

    if current.strip():
        chunks.append(current.strip())

    # Add thread numbering
    if len(chunks) > 1:
        total = len(chunks)
        chunks = [f"{i+1}/{total} {c}" for i, c in enumerate(chunks)]

    return chunks


# ════════════════════════════════════════════════════════════════
# GenAI Answer Extraction — Strip irrelevant filler from LLM output
# ════════════════════════════════════════════════════════════════

# Patterns that indicate irrelevant preamble/filler to strip
_FILLER_PATTERNS = [
    _re.compile(r'^(Sure[!.,]?|Of course[!.,]?|Absolutely[!.,]?|Great question[!.,]?|Here\'s?|Here is|Let me|I\'ll|I will|I\'d|Okay[!.,]?)\s*', _re.IGNORECASE | _re.MULTILINE),
    _re.compile(r'^(As an AI|As a language model|I\'m here to|Happy to help)[^.\n]*[.!]?\s*', _re.IGNORECASE | _re.MULTILINE),
]

# Patterns for irrelevant conclusions to strip
_IRRELEVANT_CONCLUSION_PATTERNS = [
    _re.compile(r'\n+(Let me know|Feel free|Hope (this|that)|If you (have|need|want)|Would you like|Do you want|Shall I|Want me to|Is there anything|Happy to|I hope|Don\'t hesitate)[^\n]*$', _re.IGNORECASE),
    _re.compile(r'\n+(In (summary|conclusion),?\s*)?this (is|was) (just )?a (brief|quick|simple) (overview|summary|introduction)[^\n]*$', _re.IGNORECASE),
    _re.compile(r'\n+---+\s*$'),
]


def extract_relevant_answer(text: str) -> str:
    """
    Smart extraction: strip GenAI preamble + irrelevant conclusions.

    The agent MUST be smart enough to:
      1. Detect and remove "Sure, here's..." type preambles
      2. Detect and remove "Let me know if..." type conclusions
      3. Keep ONLY the actual answer content
      4. Preserve all meaning, structure, lists, steps
      5. Handle copy-paste from GenAI output intelligently

    Example:
      Input:  "Sure! Here's the answer. \n\n Herbs regrow from... \n\n Let me know if you need more!"
      Output: "Herbs regrow from..."
    """
    if not text:
        return text

    result = text.strip()

    # Step 1: Strip preamble filler
    for pattern in _FILLER_PATTERNS:
        result = pattern.sub('', result, count=1).strip()

    # Step 2: Strip irrelevant conclusions
    for pattern in _IRRELEVANT_CONCLUSION_PATTERNS:
        result = pattern.sub('', result).strip()

    # Step 3: If the result is substantially shorter, it means we stripped
    # only filler. If we accidentally stripped too much (>70%), revert.
    if len(result) < len(text) * 0.3 and len(text) > 50:
        logger.warning("extract_answer.too_aggressive", original_len=len(text), result_len=len(result))
        return text.strip()  # Revert — we over-stripped

    return result


def _generate_hashtags_for_content(niche_key: str, platform: str, location: str = None,
                                    topic_keywords: list = None) -> dict:
    """
    Generate 5 micro-niche hashtags (NOT broad generic tags).
    Uses the micro_niche_5 strategy from Gumloop training docs.
    """
    try:
        from hashtags.matrix_5layer import generate_micro_niche_5

        result = generate_micro_niche_5(
            niche=niche_key,
            platform=platform,
            location=location,
            topic_keywords=topic_keywords,
        )
        return result
    except ImportError:
        logger.warning("content_factory.no_hashtag_module")
        return {"hashtags": [], "count": 0, "strategy": "fallback"}
    except Exception as e:
        logger.warning("content_factory.hashtag_error", error=str(e))
        return {"hashtags": [], "count": 0, "strategy": "error"}


def _build_system_prompt(niche_config: dict) -> str:
    """Build the system prompt for content generation using properly extracted niche data."""
    extracted = _extract_niche_data(niche_config)
    niche_name = extracted["display_name"]
    hooks = extracted["hooks"]
    personas = extracted["personas"]
    pains = extracted["pains"]
    desires = extracted["desires"]
    sub_names = extracted["sub_niche_names"]

    return f"""You are a viral content factory for the "{niche_name}" micro-niche.

ROLE: CTO Agent in EMADS-PR v1.0 — you generate content packs for multi-platform publishing.

NICHE CONTEXT:
- Sub-niches: {json.dumps(sub_names[:5])}
- Target personas: {json.dumps(personas[:5])}
- Pain points: {json.dumps(pains[:5])}
- Desires: {json.dumps(desires[:5])}
- Proven hooks: {json.dumps(hooks[:5])}

UNIVERSAL CAPTION FORMULA (from spec):
[LOCATION] [SEASON] [PAIN POINT] emoji
[AUDIENCE 1]? [AUDIENCE 2]? [AUDIENCE 3]?
[PRODUCT/SOLUTION] in 3 minutes:
• Step 1: [ACTION 1]
• Step 2: [ACTION 2]
• Result: [METRICS]
Full tutorial pinned on profile! 👇
Micro keywords: [keyword1 • keyword2 • keyword3]

OUTPUT FORMAT (JSON):
{{
  "title": "Compelling title (60 chars max for SEO)",
  "body": "Full article/post body (500-1500 words, markdown)",
  "hook": "Scroll-stopping first line (under 100 chars)",
  "pain_point": "The problem your audience faces (1-2 sentences)",
  "solution_steps": "3-5 actionable steps (numbered list)",
  "step_1": "First action step (short)",
  "step_2": "Second action step (short)",
  "result": "Result with metrics (short)",
  "micro_keywords": "3 SEO micro keywords separated by ' • '",
  "cta": "Call-to-action (save, share, follow, link)",
  "seo_description": "Meta description (155 chars max)",
  "image_prompt": "DALL-E prompt for hero image (detailed, photographic style, 9:16 format)",
  "content_type": "article|tip|listicle|how-to|story",
  "estimated_engagement": "low|medium|high|viral"
}}

RULES:
1. Hook MUST grab attention in first 3 seconds
2. Use proven viral frameworks: Problem-Agitate-Solution, Before-After-Bridge
3. Content must be 70%+ original (no copy-paste from sources)
4. Include specific data/numbers when possible
5. Make it emotionally resonant for the target persona
6. Use the Universal Caption Formula for captions
"""


def _fallback_generate(niche_config: dict, topic: Optional[str] = None) -> dict:
    """Fallback content generation WITHOUT LLM — uses caption_templates.json + niche data."""
    extracted = _extract_niche_data(niche_config)
    niche_name = extracted["display_name"]
    hooks = extracted["hooks"]
    personas = extracted["personas"]
    pains = extracted["pains"]
    desires = extracted["desires"]

    # Try loading caption templates
    templates = _load_caption_templates()
    universal = templates.get("universal", {}).get("plant_based", {})
    template_hooks = universal.get("hooks", [])
    template_ctas = universal.get("cta", [])
    template_closings = universal.get("closings", [])

    # Build from real data
    hook = hooks[0] if hooks else (template_hooks[0].format(title=topic or niche_name, key_fact="", hook="") if template_hooks else "You won't believe this...")
    persona = personas[0] if personas else "health-conscious consumer"
    pain = pains[0] if pains else f"struggling with {topic or niche_name}"
    desire = desires[0] if desires else f"better results with {topic or niche_name}"
    topic_str = topic or niche_name
    cta = template_ctas[0] if template_ctas else f"💾 Save this for later! Follow for more {topic_str} tips."
    closing = template_closings[0] if template_closings else ""

    return {
        "title": f"The Ultimate Guide to {topic_str.title()}",
        "body": f"# {topic_str.title()}\n\n{hook}\n\nIf you're a {persona}, this is for you.\n\n## The Problem\n\n{pain} — but what if you could achieve {desire}?\n\n## 3 Steps to Get Started\n\n1. Research the basics of {topic_str}\n2. Start with small, consistent actions\n3. Share your journey with the community\n\n## Final Thoughts\n\nThe key to success with {topic_str} is consistency. Start today!\n\n{closing}",
        "hook": hook,
        "pain_point": f"Most people struggle with {pain}.",
        "solution_steps": f"1. Learn the fundamentals of {topic_str}\n2. Start with one simple change today\n3. Track your progress weekly\n4. Join a community for accountability\n5. Share your results to inspire others",
        "step_1": f"Learn the basics of {topic_str}",
        "step_2": f"Start with one change today",
        "result": f"Better {desire} in 7-14 days",
        "micro_keywords": f"{topic_str} • {persona.split()[0].lower()} tips • beginner guide",
        "cta": cta,
        "seo_description": f"Discover everything about {topic_str}. Expert tips, actionable steps, and proven strategies for {persona}.",
        "image_prompt": f"Professional photo of {topic_str}, clean white background, natural lighting, lifestyle photography, high quality, 4K, 9:16 format",
        "content_type": "how-to",
        "estimated_engagement": "medium",
        "_generated_by": "fallback_template",
    }


def generate_content_pack(state: dict) -> dict:
    """
    LangGraph node: Content Factory Agent.
    Generates a content pack using OpenAI GPT or fallback templates.
    ALSO generates 7-layer hashtag matrix and injects into content pack.
    """
    niche_config = state.get("niche_config", {})
    topic = state.get("topic", None)
    budget_pct = state.get("budget_remaining_pct", 100.0)
    rss_content = state.get("rss_content", None)
    niche_key = state.get("niche_key", niche_config.get("id", "plant_based_raw"))
    platform = state.get("target_platform", "instagram")
    location = state.get("location", "Chicago")

    model = _select_model(budget_pct)
    logger.info("content_factory.start", niche=niche_config.get("display_name", niche_config.get("name")), model=model, budget_pct=budget_pct)

    # ── Extract topic keywords for micro-niche hashtag matching ──
    topic_keywords = []
    if topic:
        topic_keywords = [w.strip() for w in topic.replace("-", " ").replace("_", " ").split() if len(w) > 3]

    # ── If RSS content provided, rewrite it ──
    if rss_content:
        result = _rewrite_rss_content(rss_content, niche_config, model)
        # Generate 5 micro-niche hashtags
        hashtag_result = _generate_hashtags_for_content(niche_key, platform, location, topic_keywords)
        if "content_pack" in result:
            result["content_pack"]["hashtags"] = hashtag_result.get("hashtags", [])
            result["content_pack"]["hashtag_strategy"] = hashtag_result.get("strategy", "micro_niche_5")
            # Clean GenAI filler from RSS rewrite
            for field in ("body", "hook", "pain_point", "solution_steps", "cta"):
                if field in result["content_pack"] and result["content_pack"][field]:
                    result["content_pack"][field] = extract_relevant_answer(result["content_pack"][field])
        state.update(result)
        return state

    # ── Fallback mode ──
    if model == "fallback":
        logger.warning("content_factory.fallback", reason="budget_empty")
        content_pack = _fallback_generate(niche_config, topic)
    else:
        # ── Real LLM generation via provider cascade ──
        try:
            system_prompt = _build_system_prompt(niche_config)
            user_prompt = f"Generate a viral content pack about: {topic or niche_config.get('display_name', niche_config.get('name', 'trending topic'))}"
            if topic:
                user_prompt += f"\n\nSpecific angle: {topic}"

            content_pack, provider_name = _llm_generate_json(
                system_prompt, user_prompt, temperature=0.8,
            )

            if content_pack is not None:
                content_pack["_generated_by"] = provider_name
                content_pack["_timestamp"] = datetime.utcnow().isoformat()
                state["content_factory_status"] = "completed_llm"
                logger.info("content_factory.success", provider=provider_name, title=content_pack.get("title", "")[:50])

                # ── GenAI Answer Extraction — strip filler/irrelevant conclusions ──
                for field in ("body", "hook", "pain_point", "solution_steps", "cta"):
                    if field in content_pack and content_pack[field]:
                        content_pack[field] = extract_relevant_answer(content_pack[field])
            else:
                logger.warning("content_factory.all_providers_failed", msg="All LLM providers failed — using template")
                content_pack = _fallback_generate(niche_config, topic)
                state["content_factory_status"] = "completed_fallback"

        except Exception as e:
            logger.error("content_factory.error", error=str(e))
            content_pack = _fallback_generate(niche_config, topic)
            content_pack["_error"] = str(e)
            state["content_factory_status"] = "completed_fallback_after_error"

    # ── Generate 5 micro-niche hashtags (NOT broad generic) ──
    hashtag_result = _generate_hashtags_for_content(niche_key, platform, location, topic_keywords)
    content_pack["hashtags"] = hashtag_result.get("hashtags", [])
    content_pack["hashtag_strategy"] = hashtag_result.get("strategy", "micro_niche_5")

    state["content_pack"] = content_pack
    if "content_factory_status" not in state:
        state["content_factory_status"] = "completed_fallback"
    return state


def _rewrite_rss_content(rss_content: dict, niche_config: dict, model: str) -> dict:
    """Rewrite RSS feed content for multi-platform publishing.

    Uses multi-provider LLM cascade: Gemini → GitHub Models → OpenAI.
    Falls back to passthrough if all providers fail or model == 'fallback'.
    """
    original_title = rss_content.get("title", "")
    original_body = rss_content.get("body", "")[:3000]
    source_url = rss_content.get("url", "")

    passthrough_pack = {
        "title": original_title,
        "body": original_body,
        "hook": original_title,
        "pain_point": "",
        "solution_steps": "",
        "cta": f"Read more: {source_url}",
        "seo_description": original_title[:155],
        "image_prompt": f"Blog post illustration for: {original_title}",
        "content_type": "article",
        "estimated_engagement": "medium",
        "_generated_by": "rss_passthrough",
        "_source_url": source_url,
    }

    if model == "fallback":
        return {
            "content_pack": passthrough_pack,
            "content_factory_status": "completed_rss_passthrough",
        }

    try:
        system_prompt = f"""You are a content repurposing engine.
Rewrite the following blog post into a viral content pack.
Keep the core information but make it engaging for social media.
Output JSON with: title, body, hook, pain_point, solution_steps, step_1, step_2, result, micro_keywords, cta, seo_description, image_prompt, content_type, estimated_engagement.
Original source: {source_url} — ALWAYS credit the source."""
        user_prompt = f"Title: {original_title}\n\nBody:\n{original_body}"

        content_pack, provider_name = _llm_generate_json(
            system_prompt, user_prompt, temperature=0.7,
        )

        if content_pack is not None:
            content_pack["_generated_by"] = f"{provider_name}_rss_rewrite"
            content_pack["_source_url"] = source_url
            return {
                "content_pack": content_pack,
                "content_factory_status": "completed_rss_rewrite",
            }
        else:
            logger.warning("content_factory.rss_all_providers_failed")
            return {
                "content_pack": passthrough_pack,
                "content_factory_status": "completed_rss_passthrough",
            }

    except Exception as e:
        logger.error("content_factory.rss_rewrite_error", error=str(e))
        passthrough_pack["_error"] = str(e)
        passthrough_pack["_generated_by"] = "rss_error_fallback"
        return {
            "content_pack": passthrough_pack,
            "content_factory_status": "completed_rss_error_fallback",
        }


def adapt_for_platform(content_pack: dict, platform: str) -> dict:
    """
    Adapt content pack for a specific platform.

    v2.1 upgrades (from Gumloop training docs):
      - Exactly 5 micro-niche hashtags per channel (NOT broad)
      - Per-channel char limits with smart_truncate (sentence-boundary cut)
      - Twitter thread auto-split
      - GenAI answer extraction already applied in content_pack
      - Content split NEVER loses meaning
    """
    channel = CHANNEL_CHAR_LIMITS.get(platform, CHANNEL_CHAR_LIMITS.get("twitter"))
    max_caption = channel["caption"]
    max_title = channel.get("title", 0)
    optimal_hashtags = channel.get("optimal_hashtags", 5)

    title = content_pack.get("title", "")
    body = content_pack.get("body", "")
    hook = content_pack.get("hook", title)
    cta = content_pack.get("cta", "")
    hashtags = content_pack.get("hashtags", [])

    # ── 5 micro-niche hashtags only (from Gumloop spec) ──
    if isinstance(hashtags, list):
        hashtags = hashtags[:optimal_hashtags]
        hashtag_str = " ".join(hashtags)
    else:
        hashtag_str = str(hashtags)

    # Load transform rules
    transforms = _load_content_transforms()
    transform_rules = transforms.get("transforms", {})

    # ── Platform-specific adaptation with smart_truncate ──
    thread_parts = None  # For Twitter thread

    if platform == "twitter":
        # Twitter: hook only, hashtags in remaining space
        if len(body) > 280:
            # Auto-thread if content is long
            thread_parts = smart_split_for_thread(body, chunk_size=280)
        caption = smart_truncate(hook, 240)
        if hashtag_str:
            remaining = 280 - len(caption) - 2
            if remaining > 10:
                caption += "\n" + hashtag_str[:remaining]
        caption = caption[:280]

    elif platform == "reddit":
        # Reddit: full body, NO hashtags (community hates them)
        caption = smart_truncate(body, max_caption)
        hashtag_str = ""
        hashtags = []

    elif platform in ("medium", "shopify_blog"):
        # Long-form: full body, hashtags as tags (not in content)
        caption = body

    elif platform == "linkedin":
        # LinkedIn: professional structure
        ln_rules = transform_rules.get("caption_to_linkedin", {}).get("rules", [])
        raw = f"{hook}\n\n{content_pack.get('pain_point', '')}\n\n{content_pack.get('solution_steps', '')}\n\n{cta}"
        if hashtag_str:
            raw += f"\n\n{hashtag_str}"
        if any("What do you think" in r for r in ln_rules) and "What do you think" not in raw:
            raw += "\n\nWhat do you think?"
        caption = smart_truncate(raw, max_caption)

    elif platform == "pinterest":
        # Pinterest: short + searchable
        raw = f"{hook}\n\n{content_pack.get('step_1', '')}. {content_pack.get('step_2', '')}.\n\n{hashtag_str}"
        caption = smart_truncate(raw, max_caption)

    elif platform == "youtube":
        # YouTube: description + 5 hashtags at end (first 3 show on title)
        raw = f"{content_pack.get('pain_point', '')}\n\n{content_pack.get('solution_steps', '')}\n\n{cta}\n\n{hashtag_str}"
        caption = smart_truncate(raw, max_caption)

    else:
        # Default: Universal Caption Template from spec
        try:
            caption = UNIVERSAL_CAPTION_TEMPLATE.format(
                location=content_pack.get("location", "Chicago"),
                season=content_pack.get("season", _get_current_season()),
                pain_point=content_pack.get("pain_point", ""),
                season_emoji=_season_emoji(),
                audience_1=content_pack.get("audience_1", "Busy people"),
                audience_2=content_pack.get("audience_2", "Beginners"),
                audience_3=content_pack.get("audience_3", "Health seekers"),
                product_solution=content_pack.get("title", "This"),
                step_1=content_pack.get("step_1", "Start here"),
                step_2=content_pack.get("step_2", "Follow through"),
                result=content_pack.get("result", "Amazing results"),
                micro_keywords=content_pack.get("micro_keywords", ""),
                hashtags=hashtag_str,
            )
        except (KeyError, IndexError):
            caption = SIMPLE_CAPTION_TEMPLATE.format(
                hook=hook,
                pain_point=content_pack.get("pain_point", ""),
                solution_steps=content_pack.get("solution_steps", ""),
                cta=cta,
                hashtags=hashtag_str,
            )
        caption = smart_truncate(caption, max_caption)

    # ── Title truncation (if channel has title limit) ──
    if max_title > 0:
        title = smart_truncate(title, max_title, preserve_meaning=False)

    result = {
        "platform": platform,
        "title": title[:200],
        "caption": caption,
        "hashtags": hashtags,
        "hashtag_count": len(hashtags),
        "char_count": len(caption),
        "char_limit": max_caption,
        "within_limit": len(caption) <= max_caption,
        "image_prompt": content_pack.get("image_prompt", ""),
        "content_type": content_pack.get("content_type", "article"),
    }

    # Add thread parts for Twitter
    if thread_parts:
        result["thread_parts"] = thread_parts
        result["thread_count"] = len(thread_parts)

    return result


def _get_current_season() -> str:
    """Get current season name."""
    month = datetime.now().month
    if month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    elif month in (9, 10, 11):
        return "Fall"
    else:
        return "Winter"


def _season_emoji() -> str:
    """Get emoji for current season."""
    month = datetime.now().month
    if month in (3, 4, 5):
        return "🌱"
    elif month in (6, 7, 8):
        return "☀️"
    elif month in (9, 10, 11):
        return "🍂"
    else:
        return "❄️"
