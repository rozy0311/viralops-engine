"""
Content Factory Agent — CTO Agent (EMADS-PR v1.0)
REAL implementation with OpenAI GPT integration.
Generates content packs: title + body + universal caption + hashtags + image prompt.

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
}

# ── Paths ──
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
CAPTION_TEMPLATES_PATH = os.path.join(TEMPLATES_DIR, "caption_templates.json")
CONTENT_TRANSFORMS_PATH = os.path.join(TEMPLATES_DIR, "content_transforms.json")
NICHES_YAML_PATH = os.path.join(CONFIG_DIR, "niches.yaml")


def _get_openai_client():
    """Lazy-load OpenAI client."""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("content_factory.no_api_key", msg="OPENAI_API_KEY not set — using fallback mode")
            return None
        return OpenAI(api_key=api_key)
    except ImportError:
        logger.warning("content_factory.no_openai", msg="openai package not installed — using fallback mode")
        return None


def _select_model(budget_remaining_pct: float = 100.0) -> str:
    """Cost-aware model selection (Training 07-Cost-Aware-Planning)."""
    if budget_remaining_pct > 50:
        return "gpt-4.1"
    elif budget_remaining_pct > 20:
        return "gpt-4.1-mini"
    elif budget_remaining_pct > 5:
        return "gpt-4o-mini"
    else:
        return "fallback"


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


def _generate_hashtags_for_content(niche_key: str, platform: str, location: str = None) -> dict:
    """
    Generate hashtags using the 7-layer matrix.
    Returns the matrix result dict.
    """
    try:
        from hashtags.matrix_5layer import generate_hashtag_matrix, generate_5cap
        
        matrix = generate_hashtag_matrix(
            niche=niche_key,
            platform=platform,
            location=location,
        )
        return matrix
    except ImportError:
        logger.warning("content_factory.no_hashtag_module")
        return {"combined": [], "highest_search": [], "layers": {}}
    except Exception as e:
        logger.warning("content_factory.hashtag_error", error=str(e))
        return {"combined": [], "highest_search": [], "layers": {}}


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

    # ── If RSS content provided, rewrite it ──
    if rss_content:
        result = _rewrite_rss_content(rss_content, niche_config, model)
        # Also generate hashtags for RSS content
        hashtag_matrix = _generate_hashtags_for_content(niche_key, platform, location)
        if "content_pack" in result:
            result["content_pack"]["hashtags"] = hashtag_matrix.get("combined", [])
            result["content_pack"]["hashtag_matrix"] = hashtag_matrix
            result["content_pack"]["highest_search_hashtags"] = hashtag_matrix.get("highest_search", [])
        state.update(result)
        return state

    # ── Fallback mode ──
    if model == "fallback":
        logger.warning("content_factory.fallback", reason="budget_empty")
        content_pack = _fallback_generate(niche_config, topic)
    elif not (client := _get_openai_client()):
        content_pack = _fallback_generate(niche_config, topic)
        state["content_factory_status"] = "completed_fallback"
    else:
        # ── Real LLM generation ──
        try:
            system_prompt = _build_system_prompt(niche_config)
            user_prompt = f"Generate a viral content pack about: {topic or niche_config.get('display_name', niche_config.get('name', 'trending topic'))}"
            if topic:
                user_prompt += f"\n\nSpecific angle: {topic}"

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )

            if response.usage:
                _track_cost(model, {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                })

            content_pack = json.loads(response.choices[0].message.content)
            content_pack["_generated_by"] = model
            content_pack["_timestamp"] = datetime.utcnow().isoformat()
            state["content_factory_status"] = "completed_llm"
            logger.info("content_factory.success", model=model, title=content_pack.get("title", "")[:50])

        except Exception as e:
            logger.error("content_factory.error", error=str(e), model=model)
            content_pack = _fallback_generate(niche_config, topic)
            content_pack["_error"] = str(e)
            state["content_factory_status"] = "completed_fallback_after_error"

    # ── Generate 7-layer hashtag matrix ──
    hashtag_matrix = _generate_hashtags_for_content(niche_key, platform, location)
    content_pack["hashtags"] = hashtag_matrix.get("combined", [])
    content_pack["hashtag_matrix"] = hashtag_matrix
    content_pack["highest_search_hashtags"] = hashtag_matrix.get("highest_search", [])

    state["content_pack"] = content_pack
    if "content_factory_status" not in state:
        state["content_factory_status"] = "completed_fallback"
    return state


def _rewrite_rss_content(rss_content: dict, niche_config: dict, model: str) -> dict:
    """Rewrite RSS feed content for multi-platform publishing."""
    client = _get_openai_client()
    original_title = rss_content.get("title", "")
    original_body = rss_content.get("body", "")[:3000]
    source_url = rss_content.get("url", "")

    if not client or model == "fallback":
        return {
            "content_pack": {
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
            },
            "content_factory_status": "completed_rss_passthrough",
        }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"""You are a content repurposing engine.
Rewrite the following blog post into a viral content pack.
Keep the core information but make it engaging for social media.
Output JSON with: title, body, hook, pain_point, solution_steps, step_1, step_2, result, micro_keywords, cta, seo_description, image_prompt, content_type, estimated_engagement.
Original source: {source_url} — ALWAYS credit the source."""},
                {"role": "user", "content": f"Title: {original_title}\n\nBody:\n{original_body}"},
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )

        if response.usage:
            _track_cost(model, {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            })

        content_pack = json.loads(response.choices[0].message.content)
        content_pack["_generated_by"] = f"{model}_rss_rewrite"
        content_pack["_source_url"] = source_url
        return {
            "content_pack": content_pack,
            "content_factory_status": "completed_rss_rewrite",
        }
    except Exception as e:
        logger.error("content_factory.rss_rewrite_error", error=str(e))
        return {
            "content_pack": {
                "title": original_title,
                "body": original_body,
                "hook": original_title,
                "_generated_by": "rss_error_fallback",
                "_source_url": source_url,
                "_error": str(e),
            },
            "content_factory_status": "completed_rss_error_fallback",
        }


def adapt_for_platform(content_pack: dict, platform: str) -> dict:
    """
    Adapt content pack for a specific platform.
    Uses Universal Caption Template from spec + content_transforms.json rules.
    """
    specs = PLATFORM_SPECS.get(platform, PLATFORM_SPECS["twitter"])
    max_caption = specs["max_caption"]
    max_hashtags = specs["max_hashtags"]

    title = content_pack.get("title", "")
    body = content_pack.get("body", "")
    hook = content_pack.get("hook", title)
    cta = content_pack.get("cta", "")
    hashtags = content_pack.get("hashtags", [])

    # Truncate hashtags to platform limit
    if isinstance(hashtags, list):
        hashtags = hashtags[:max_hashtags]
        hashtag_str = " ".join(hashtags)
    else:
        hashtag_str = str(hashtags)

    # Load transform rules
    transforms = _load_content_transforms()
    transform_rules = transforms.get("transforms", {})

    # Platform-specific adaptation
    if platform == "twitter":
        # Apply caption_to_thread rules if available
        caption = hook[:240]
        if hashtag_str:
            remaining = 280 - len(caption) - 2
            if remaining > 10:
                caption += "\n" + hashtag_str[:remaining]

    elif platform == "reddit":
        caption = body[:max_caption]

    elif platform in ("medium", "shopify_blog"):
        caption = body

    elif platform == "linkedin":
        # Apply caption_to_linkedin transform rules
        ln_rules = transform_rules.get("caption_to_linkedin", {}).get("rules", [])
        caption = f"{hook}\n\n{content_pack.get('pain_point', '')}\n\n{content_pack.get('solution_steps', '')}\n\n{cta}"
        if hashtag_str:
            caption += f"\n\n{hashtag_str}"
        # Apply rule: Add "What do you think?" if not in CTA
        if any("What do you think" in r for r in ln_rules) and "What do you think" not in caption:
            caption += "\n\nWhat do you think?"
        caption = caption[:max_caption]

    else:
        # Default: Use Universal Caption Template from spec
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
            # Fallback to simple template
            caption = SIMPLE_CAPTION_TEMPLATE.format(
                hook=hook,
                pain_point=content_pack.get("pain_point", ""),
                solution_steps=content_pack.get("solution_steps", ""),
                cta=cta,
                hashtags=hashtag_str,
            )
        caption = caption[:max_caption]

    return {
        "platform": platform,
        "title": title[:200],
        "caption": caption,
        "hashtags": hashtags,
        "image_prompt": content_pack.get("image_prompt", ""),
        "content_type": content_pack.get("content_type", "article"),
    }


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
