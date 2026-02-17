"""
ViralOps Engine — Smart LLM Content Pipeline
=============================================
Multi-provider cascade with self-review (EMADS-PR pattern).

Providers (cost-aware order):
  1. Gemini cascade (free tier — auto-fallback through 7 models)
     2.5 Flash → 2.5 Pro → 2.0 Flash → 3.0 Pro → 2.5 Flash Lite → 2.0 Flash Lite → 2.0 Pro Exp
  2. GitHub Models / gpt-4o-mini (free via Copilot)
  3. Perplexity / sonar (has web search — great for trending content)
  4. OpenAI / gpt-4o-mini (paid fallback)

Following Training Multi-Agent principles:
  - Cost-Aware Planning (doc 07)
  - Security Defense (doc 04) — never hardcode keys
  - ReconcileGPT pattern — self-review before publishing
"""

import os
import re
import json
import time
import httpx
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)

# ═══════════════════════════════════════════════════════════════
# DATA ROOT — Short path to avoid Windows MAX_PATH (260) issues
# The workspace path is ~210 chars, so we use a short absolute path.
# ═══════════════════════════════════════════════════════════════
DATA_ROOT = os.environ.get("VIRALOPS_DATA_ROOT", r"D:\vops-data")
POSTS_DIR = os.path.join(DATA_ROOT, "posts")
IMG_DIR = os.path.join(DATA_ROOT, "img")

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


# ── Gemini model fallback chain (all free tier, same API key) ──
# When one model hits RPD/RPM quota, auto-cascade to the next.
# Order matches user preference: smart models first, budget models last.
GEMINI_TEXT_MODELS = [
    "gemini-2.5-flash",       # Primary — 5 RPM, 250K TPM, 20 RPD
    "gemini-2.5-pro",         # 15 RPM, Unlimited TPM, 1.5K RPD
    "gemini-2.0-flash",       # 15 RPM, Unlimited TPM, 1.5K RPD
    "gemini-2.5-flash-lite",  # 10 RPM, 250K TPM, 20 RPD
    "gemini-2.0-flash-lite",  # 15 RPM, Unlimited TPM, 1.5K RPD
    "gemini-2.0-pro-exp",     # 15 RPM, Unlimited TPM, 1.5K RPD
]

# Track which Gemini models are quota-exhausted with timestamps.
# Entries expire after GEMINI_QUOTA_RESET_SECS so recovered quotas are retried.
_gemini_exhausted: dict = {}  # {model_name: exhausted_timestamp}
GEMINI_QUOTA_RESET_SECS = 3600  # 1 hour

def _is_gemini_exhausted(model_name: str) -> bool:
    """Check if a Gemini model is quota-exhausted (with 1-hour expiry)."""
    ts = _gemini_exhausted.get(model_name)
    if ts is None:
        return False
    if time.time() - ts > GEMINI_QUOTA_RESET_SECS:
        _gemini_exhausted.pop(model_name, None)
        return False
    return True

def _mark_gemini_exhausted(model_name: str) -> None:
    """Mark a Gemini model as quota-exhausted (auto-expires after 1 hour)."""
    _gemini_exhausted[model_name] = time.time()

# Singleton Gemini client — reuse across calls to avoid connection overhead
_gemini_client = None
_gemini_client_key = None

def _get_gemini_client(api_key: str):
    """Get or create a cached Gemini SDK client."""
    global _gemini_client, _gemini_client_key
    if _gemini_client is not None and _gemini_client_key == api_key:
        return _gemini_client
    from google import genai
    _gemini_client = genai.Client(api_key=api_key)
    _gemini_client_key = api_key
    return _gemini_client


# Provider cascade — cheapest working first
PROVIDERS = [
    ProviderConfig(
        name="gemini",
        api_key_env="GEMINI_API_KEY",
        base_url="",  # Uses SDK
        model="gemini-2.5-flash",  # Initial model (overridden by cascade)
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
    """
    Call Google Gemini via genai SDK with automatic model fallback.

    Cascades through GEMINI_TEXT_MODELS when a model hits 429/RESOURCE_EXHAUSTED.
    Tracks exhausted models per session so subsequent calls skip them instantly.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return ProviderResult(text="", provider=config.name, model=config.model,
                             success=False, error="google-genai not installed")

    try:
        client = _get_gemini_client(api_key)
    except (ValueError, Exception) as e:
        return ProviderResult(text="", provider=config.name, model=config.model,
                             success=False, error=f"Gemini client init failed: {e}")
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    last_error = ""
    for model_name in GEMINI_TEXT_MODELS:
        # Skip models we already know are exhausted (with 1-hour expiry)
        if _is_gemini_exhausted(model_name):
            continue

        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    http_options={"timeout": 60_000},  # 60s timeout (ms)
                ),
            )

            text = resp.text.strip() if resp.text else ""
            if text:
                # Use actual token count from SDK when available, else estimate
                est_tokens = len(text.split()) * 1.3
                try:
                    if hasattr(resp, 'usage_metadata') and resp.usage_metadata:
                        actual = getattr(resp.usage_metadata, 'candidates_token_count', 0)
                        if actual:
                            est_tokens = actual
                except Exception:
                    pass
                return ProviderResult(
                    text=text,
                    provider=config.name,
                    model=model_name,
                    tokens_used=int(est_tokens),
                    success=True,
                )
            else:
                last_error = f"{model_name}: Empty response"
                print(f"  [LLM] Gemini/{model_name} — empty response, trying next…")

        except Exception as e:
            err_str = str(e).lower()
            is_quota = (
                "resource_exhausted" in err_str
                or "429" in err_str
                or "quota" in err_str
                or "rate limit" in err_str
                or "rate_limit" in err_str
            )
            if is_quota:
                _mark_gemini_exhausted(model_name)
                print(f"  [LLM] Gemini/{model_name} — quota exhausted, cascading…")
                last_error = f"{model_name}: quota exhausted"
                continue
            else:
                # Non-quota error (model not found, bad request, etc.)
                print(f"  [LLM] Gemini/{model_name} — error: {str(e)[:120]}")
                last_error = f"{model_name}: {str(e)[:200]}"
                continue

    # All Gemini models exhausted
    return ProviderResult(
        text="",
        provider=config.name,
        model="gemini-all-exhausted",
        success=False,
        error=f"All {len(GEMINI_TEXT_MODELS)} Gemini models exhausted: {last_error}",
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
    
    # Retry once on 429 with exponential backoff
    for _attempt in range(2):
        r = httpx.post(config.base_url, headers=headers, json=payload, timeout=60)
        if r.status_code == 429:
            try:
                retry_after = int(float(r.headers.get("retry-after", "10")))
            except (ValueError, TypeError):
                retry_after = 10
            wait = min(retry_after, 30)
            print(f"  [LLM] {config.name} rate-limited (429) — waiting {wait}s...")
            time.sleep(wait)
            continue
        break
    
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
# AI IMAGE GENERATION — Realistic photos via Gemini (9:16 format)
# ═══════════════════════════════════════════════════════════════
# User's workflow: "create realistic image for the answer (image size 9:16)"
# We replicate this using Gemini's image generation models.

# Cascade of image-generation models (try in order, skip quota-exhausted)
# Gemini models share one quota pool; Imagen models have a separate quota pool.
# 3 Gemini models × 2 keys = 6 attempts; 2 Imagen models × 2 keys = 4 attempts → 10 total cloud
IMAGE_MODELS_GEMINI = [
    "gemini-2.0-flash-exp-image-generation",
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",   # Gemini 3 Pro image generation
]
IMAGE_MODELS_IMAGEN = [
    "imagen-4-fast-generate",     # Imagen 4 Fast (25 RPD free)
    "imagen-4-generate",          # Imagen 4 Standard (25 RPD free)
]
# Combined for iteration — but quota tracking is per-pool
IMAGE_MODELS = IMAGE_MODELS_GEMINI + IMAGE_MODELS_IMAGEN


def build_image_prompt(pack: Dict[str, Any]) -> str:
    """
    Build a realistic image prompt from a content pack.
    Follows user's style: "create realistic image for the answer (image size 9:16)"
    """
    title = pack.get("title", "")
    topic = title.split("—")[0].split(":")[0].strip() if title else ""
    pain_point = pack.get("pain_point", "")
    image_title = pack.get("image_title", topic)
    
    # Determine visual subject from content
    content = pack.get("content_formatted", "")
    
    # Build hyper-detailed cinematic prompt for maximum realism
    prompt = (
        f"Ultra-realistic photograph, shot on Sony A7IV with 85mm f/1.4 lens. "
        f"Subject: {image_title}. "
        f"Scene context: {pain_point[:120]}. "
        f"Style: Editorial food/lifestyle photography for a premium magazine. "
        f"Warm golden-hour natural lighting streaming through a window, casting soft shadows. "
        f"Shallow depth of field with creamy bokeh background. "
        f"Rich vibrant colors, visible textures and fine details on every surface. "
        f"Wooden cutting board, fresh herbs, rustic kitchen counter, linen napkin — organic props. "
        f"Vertical 9:16 portrait composition with rule-of-thirds framing. "
        f"8K resolution quality, hyper-detailed, photorealistic, no CGI look. "
        f"ABSOLUTELY NO TEXT, NO LETTERS, NO WORDS, NO NUMBERS, NO WATERMARKS, NO LOGOS in the image. "
        f"No human faces, no hands. Raw authentic aesthetic."
    )
    
    return prompt


def generate_ai_image(
    prompt: str,
    output_path: str,
    max_retries: int = 2,
) -> Optional[str]:
    """
    Generate a realistic 9:16 image.
    
    Cascade order (cost-optimised):
      1. Pollinations Flux (FREE — primary)
      2. Gemini image models (quota-limited)
      3. Imagen models (quota-limited)
      ↳ None → caller falls back to PIL gradient
    
    Returns:
        Path to saved image file, or None if all attempts fail.
    """
    import base64
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # ── PRIMARY: Pollinations API (Flux model, free) ──────────────────────
    result = _try_pollinations(prompt, output_path, max_retries=max_retries)
    if result:
        return result
    
    # ── FALLBACK 1: Gemini / Imagen cascade ───────────────────────────────
    # Dual API key rotation: primary → fallback
    api_keys = []
    pk = os.environ.get("GEMINI_API_KEY", "")
    fk = os.environ.get("FALLBACK_GEMINI_API_KEY", "") or os.environ.get("FALLBACK_GOOGLE_AI_STUDIO_API_KEY", "")
    if pk:
        api_keys.append(("primary", pk))
    if fk and fk != pk:
        api_keys.append(("fallback", fk))
    
    if not api_keys:
        print("  [IMAGE] No GEMINI_API_KEY — skipping Gemini/Imagen fallback")
        print("  [IMAGE] All AI image providers failed — will use PIL gradient fallback")
        return None
    
    for key_label, api_key in api_keys:
        # Track quota exhaustion per pool (Gemini vs Imagen) — reset per key
        gemini_img_exhausted = False
        imagen_exhausted = False
        print(f"  [IMAGE] Trying Gemini/Imagen with {key_label} API key...")
        for model in IMAGE_MODELS:
            # Check per-pool quota skip
            is_gemini_model = model in IMAGE_MODELS_GEMINI
            is_imagen_model = model in IMAGE_MODELS_IMAGEN
            if is_gemini_model and gemini_img_exhausted:
                print(f"  [IMAGE] Skipping {model} — Gemini image quota exhausted")
                continue
            if is_imagen_model and imagen_exhausted:
                print(f"  [IMAGE] Skipping {model} — Imagen quota exhausted")
                continue
            for attempt in range(max_retries):
                try:
                    url = (
                        f"https://generativelanguage.googleapis.com/v1beta/"
                        f"models/{model}:generateContent"
                    )
                    img_headers = {
                        "x-goog-api-key": api_key,
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "responseModalities": ["TEXT", "IMAGE"],
                        },
                    }
                    
                    print(f"  [IMAGE] Calling {model} [{key_label}] (attempt {attempt + 1})...")
                    t0 = time.time()
                    r = httpx.post(url, headers=img_headers, json=payload, timeout=90)
                    elapsed = time.time() - t0
                    
                    if r.status_code == 429:
                        # Mark the correct pool as exhausted
                        if "RESOURCE_EXHAUSTED" in r.text or attempt == 0:
                            if is_gemini_model:
                                print(f"  [IMAGE] Gemini image quota exhausted — trying Imagen models…")
                                gemini_img_exhausted = True
                            else:
                                print(f"  [IMAGE] Imagen quota exhausted")
                                imagen_exhausted = True
                            break
                        wait = min(30 * (attempt + 1), 90)
                        print(f"  [IMAGE] Rate limited — waiting {wait}s...")
                        time.sleep(wait)
                        continue
                    
                    if r.status_code != 200:
                        print(f"  [IMAGE] {model} error {r.status_code}: {r.text[:200]}")
                        break  # Try next model
                    
                    data = r.json()
                    for candidate in data.get("candidates", []):
                        for part in candidate.get("content", {}).get("parts", []):
                            if "inlineData" in part:
                                mime = part["inlineData"].get("mimeType", "image/png")
                                img_b64 = part["inlineData"].get("data", "")
                                if img_b64:
                                    # Determine extension
                                    ext = "png" if "png" in mime else "jpg"
                                    if not output_path.endswith(f".{ext}"):
                                        output_path = output_path.rsplit(".", 1)[0] + f".{ext}"
                                    
                                    img_bytes = base64.b64decode(img_b64)
                                    with open(output_path, "wb") as f:
                                        f.write(img_bytes)
                                    
                                    fsize = os.path.getsize(output_path)
                                    print(f"  [IMAGE] ✓ Generated via {model} [{key_label}]! {fsize:,} bytes in {elapsed:.1f}s")
                                    print(f"  [IMAGE] Saved: {output_path}")
                                    return output_path
                    
                    print(f"  [IMAGE] {model}: No image data in response")
                    break  # Try next model
                    
                except Exception as e:
                    print(f"  [IMAGE] {model} error: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
        
        # If both pools exhausted for this key, try next key
        if gemini_img_exhausted and imagen_exhausted:
            print(f"  [IMAGE] All models exhausted for {key_label} key — trying next key...")
            continue
    
    print("  [IMAGE] All AI image providers failed — will use PIL gradient fallback")
    return None


def _try_pollinations(
    prompt: str,
    output_path: str,
    max_retries: int = 2,
) -> Optional[str]:
    """
    Try generating an image via Pollinations API (Flux model).
    
    Primary provider — FREE unlimited with API key.
    Endpoints:
      - GET  https://image.pollinations.ai/prompt/{encoded}  (legacy)
      - GET  https://gen.pollinations.ai/image/{encoded}      (v2)
    """
    import urllib.parse
    
    pollinations_url = os.environ.get(
        "GET_POLLINATIONS_URL",
        "https://image.pollinations.ai/prompt/",
    )
    pollinations_key = os.environ.get("POLLINATIONS_API_KEY", "")
    model_name = os.environ.get("POLLINATIONS_MODEL", "flux")
    
    if not pollinations_url:
        return None
    
    for attempt in range(max_retries):
        try:
            encoded = urllib.parse.quote(prompt[:1500])  # URL limit safety
            img_url = (
                f"{pollinations_url.rstrip('/')}/{encoded}"
                f"?width=768&height=1365&model={model_name}&nologo=true&enhance=true"
            )
            headers = {}
            if pollinations_key:
                headers["Authorization"] = f"Bearer {pollinations_key}"
            
            print(f"  [IMAGE] Trying Pollinations ({model_name}, attempt {attempt + 1})...")
            t0 = time.time()
            r = httpx.get(img_url, headers=headers, timeout=120, follow_redirects=True)
            elapsed = time.time() - t0
            
            if r.status_code == 200 and len(r.content) > 5000:
                # Determine format from content-type
                ct = r.headers.get("content-type", "")
                ext = "jpg" if "jpeg" in ct else "png"
                if not output_path.endswith(f".{ext}"):
                    output_path = output_path.rsplit(".", 1)[0] + f".{ext}"
                
                with open(output_path, "wb") as f:
                    f.write(r.content)
                
                fsize = os.path.getsize(output_path)
                print(f"  [IMAGE] ✓ Pollinations generated! {fsize:,} bytes in {elapsed:.1f}s")
                print(f"  [IMAGE] Saved: {output_path}")
                return output_path
            else:
                print(f"  [IMAGE] Pollinations: status={r.status_code}, size={len(r.content)}")
                if attempt < max_retries - 1:
                    time.sleep(3)
        except Exception as e:
            print(f"  [IMAGE] Pollinations error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
    
    print("  [IMAGE] Pollinations failed — trying Gemini/Imagen fallback...")
    return None


def generate_image_for_pack(
    pack: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Generate an AI image for a content pack.
    Builds prompt from pack content and saves to output_dir.
    
    Returns path to generated image, or None.
    """
    if not output_dir:
        output_dir = IMG_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    prompt = build_image_prompt(pack)
    
    # Build output filename — keep short to stay under MAX_PATH on Windows
    import re
    safe_title = re.sub(r'[^\w-]', '', pack.get('title', 'img').split('—')[0].split(':')[0])[:20].strip('_')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{timestamp}_{safe_title}.png")
    
    print(f"\n  🖼️  AI Image Generation")
    print(f"  Prompt: {prompt[:120]}...")
    
    result = generate_ai_image(prompt, output_path)
    
    if result:
        pack["_ai_image_path"] = result
        pack["_image_prompt"] = prompt
    
    return result


# ═══════════════════════════════════════════════════════════════
# CONTENT GENERATION PIPELINE (EMADS-PR PATTERN)
# ═══════════════════════════════════════════════════════════════

# System prompt for content generation (CTO Agent role)
CONTENT_SYSTEM = f"""You are a TikTok content specialist for plant-based, homesteading, and urban farming micro-niches.
Target audience: US-based 18-45, apartment/small-space dwellers, budget-conscious.
Channels: {os.environ.get('TIKTOK_CHANNELS', '@therikerootstories (plant-based), @agrinomadsvietnam (farming), @therikecom (AI/tech)')}.

RULES:
- Content MUST be educational + actionable (steps people can follow TODAY)
- Include specific numbers (costs, timeframes, quantities)
- Use conversational, witty tone — NOT corporate
- Focus on micro-niche topics that are underserved on TikTok
- NEVER generic fluff — every post must teach something specific"""

# ═══════════════════════════════════════════════════════════════
# QUALITY CONTENT GENERATION (SPEC-COMPLIANT — 3500-4000 chars)
# ═══════════════════════════════════════════════════════════════

# Enhanced system prompt for FULL content generation matching the spec format
QUALITY_CONTENT_SYSTEM = """You are a Perplexity-style expert answering ultra-specific questions about plant-based living, homesteading, urban farming, and DIY sustainability.

YOUR ROLE: ANSWER the question like you're a witty, knowledgeable friend who has actually done this. Give the REAL answer — specific, practical, no BS.

WRITING STYLE (match this EXACTLY):
- Casual, witty, personality-driven: "basically unkillable", "mint spreads like it's trying to take over your life", "Rosemary is slow and stubborn. It's not you. It's rosemary."
- Include EXACT numbers always: costs ($2/lb, $4-$6/jar), timeframes (5-10 days, 2-6 weeks), quantities (4-6 inch stems, 2-tablespoon serving), temperatures (350°F for 10-15 minutes)
- Dry humor + real talk: "your kitchen starts smelling like a swamp experiment", "tall and tragic", "they regrow like they're personally offended you ever threw them away"
- NO filler: ZERO "let's dive in", "great question", "in conclusion", "without further ado"
- Every sentence TEACHES something new — if it doesn't, cut it
- Write like Perplexity AI answers: thorough, fact-dense, organized, but with personality

FORMAT RULES — PLAIN TEXT ONLY (TikTok does NOT render Markdown):
- ABSOLUTELY NEVER use **bold** markers — TikTok shows them as literal ** characters
- ABSOLUTELY NEVER use ### or ## headings — TikTok shows them as literal # characters
- ABSOLUTELY NEVER use Markdown formatting of any kind — no *, no #, no backticks, no [links]()
- Use emoji as section headers on their own line: "🌿 Section Name"
- Use ALL CAPS sparingly for key emphasis: "costs about $2/LB", "soak for 8-12 HOURS"
- Use simple numbered lists: 1. 2. 3. (plain text)
- Use dashes for bullet points: - item
- Separate sections with a blank line for readability
- Required emoji headers: 🌿 main topic, 🫙 quick method, ❌ mistakes, ✅ tips
- End with a punchy one-liner, NOT a motivation speech

CHARACTER TARGET: 3500-4000 characters. This is for TikTok caption.
Count carefully — too short = thin answer, too long = gets cut off (4000 char limit).
If your answer is under 3200 characters, you have NOT answered thoroughly enough.

BRAND NAMES: NEVER mention specific brand names (Walmart, Goya, Just Egg, Trader Joe's, Whole Foods, etc.).
Use generic terms instead: "grocery store", "dried black beans", "liquid egg substitute", "budget store".
This keeps content evergreen and avoids looking like an ad."""

# Broad hashtag pool organized by topic group (from spec)
BROAD_HASHTAG_POOL = {
    "food": ["#foodtok", "#foodhacks", "#cookingtips", "#homecooking"],
    "smoothies": ["#smoothietok", "#smoothierecipes", "#blendersmoothie"],
    "meal_prep": ["#mealprep", "#mealprepideas", "#lunchideas", "#mealprepping"],
    "fitness": ["#fitnesstok", "#gymlife", "#healthylifestyle", "#gymmotivation"],
    "vegan": ["#vegantok", "#plantbasedrecipes", "#veganfood", "#veganeating"],
    "skincare": ["#skincaretok", "#acnetreatment", "#naturalskincare", "#glowup"],
    "home": ["#hometok", "#apartmentliving", "#lifehacks", "#adulting"],
    "plants": ["#planttok", "#indoorgarden", "#plantparent", "#urbangardening"],
    "zero_waste": ["#zerowaste", "#sustainability", "#ecofriendly", "#gogreen"],
    "budget": ["#budgettips", "#savemoney", "#budgetliving", "#frugalliving"],
    "garden": ["#gardentok", "#growyourown", "#urbanfarming", "#gardening"],
    "homestead": ["#homesteadtok", "#selfsufficientliving", "#offgrid", "#homesteading"],
}

# Keyword-to-broad-group routing
_KEYWORD_TO_GROUP = {
    "smoothie": "smoothies", "blend": "smoothies", "shake": "smoothies",
    "spinach": "smoothies", "green drink": "smoothies",
    "meal prep": "meal_prep", "lunch": "meal_prep", "batch cook": "meal_prep",
    "gym": "fitness", "workout": "fitness", "protein": "fitness",
    "vegan": "vegan", "plant-based": "vegan", "plant based": "vegan",
    "dairy-free": "vegan", "nut-free": "vegan",
    "skin": "skincare", "acne": "skincare", "mask": "skincare", "glow": "skincare",
    "apartment": "home", "office": "home", "desk": "home", "small space": "home",
    "plant": "plants", "houseplant": "plants", "indoor": "plants", "grow": "plants",
    "waste": "zero_waste", "zero waste": "zero_waste", "compost": "zero_waste",
    "budget": "budget", "cheap": "budget", "save": "budget", "free": "budget", "$": "budget",
    "garden": "garden", "soil": "garden", "seed": "garden", "microgreen": "garden",
    "mushroom": "garden", "potato": "garden", "herb": "garden",
    "homestead": "homestead", "ferment": "homestead", "preserve": "homestead",
    "sprout": "homestead", "sauerkraut": "homestead",
    "recipe": "food", "cook": "food", "kitchen": "food", "dressing": "food",
    "vinegar": "food", "butter": "food", "bread": "food",
    "iron": "vegan", "protein": "vegan", "nutrition": "vegan", "nutrient": "vegan",
    "vitamin": "vegan", "mineral": "vegan", "calcium": "vegan", "b12": "vegan",
    "tired": "fitness", "energy": "fitness", "fatigue": "fitness",
    "allergy": "food", "allergic": "food", "nut-free": "food", "lunch": "meal_prep",
    "school": "meal_prep", "kid": "food",
}


def _strip_markdown(text: str) -> str:
    """Post-processing safety net: strip Markdown that TikTok renders as literal text.

    Even with explicit anti-Markdown instructions, LLMs sometimes slip in **bold**
    or ### headings.  This function catches them BEFORE the caption reaches Publer.
    """
    if not text:
        return text
    # Remove **bold** / __bold__  →  keep inner text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    # Remove *italic* / _italic_  →  keep inner text (single markers)
    text = re.sub(r'(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)', r'\1', text)
    # Remove ### / ## / #  heading markers at start of line
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Remove leading > blockquote markers
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)
    # Remove `inline code` backticks
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Collapse 3+ consecutive blank lines → 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _ensure_section_breaks(text: str) -> str:
    """Ensure proper double-newline spacing before emoji section headers.

    TikTok captions need explicit blank lines between sections for readability.
    Without this, sections appear clumped/stuck-together on mobile.
    """
    if not text:
        return text
    lines = text.split('\n')
    result = []
    # Emoji ranges commonly used as section headers
    _EMOJI_PATTERN = re.compile(
        r'^[\U0001F300-\U0001FAD6\U00002600-\U000027BF\U0000FE00-\U0000FEFF'
        r'\U0001F900-\U0001F9FF\U00002702-\U000027B0\u2705\u274C\u2611\u26A0]'
    )
    for i, line in enumerate(lines):
        stripped = line.strip()
        # If this line starts with an emoji (section header), ensure blank line before
        if i > 0 and stripped and _EMOJI_PATTERN.match(stripped):
            # Check if previous non-empty content had a blank line before this
            if result and result[-1].strip() != '':
                result.append('')  # Insert blank line
        result.append(line)
    text = '\n'.join(result)
    # Collapse 3+ consecutive blank lines → 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _auto_pick_broad_hashtags(topic: str, existing_tags: List[str], n: int = 2) -> List[str]:
    """Auto-pick 2 broad TikTok hashtags based on topic keywords (from spec)."""
    topic_lower = topic.lower()
    matched_groups = set()
    for keyword, group in _KEYWORD_TO_GROUP.items():
        if keyword in topic_lower:
            matched_groups.add(group)
    
    if not matched_groups:
        matched_groups = {"food", "home"}  # default fallback
    
    existing_lower = {t.lower() for t in existing_tags}
    candidates = []
    for group in matched_groups:
        for tag in BROAD_HASHTAG_POOL.get(group, []):
            if tag.lower() not in existing_lower:
                candidates.append(tag)
    
    if len(candidates) < n:
        # Add from other groups
        for group in BROAD_HASHTAG_POOL:
            if group not in matched_groups:
                for tag in BROAD_HASHTAG_POOL[group]:
                    if tag.lower() not in existing_lower and tag not in candidates:
                        candidates.append(tag)
    
    import random as _rng
    _rng.shuffle(candidates)
    return candidates[:n]


def generate_quality_post(
    topic: str,
    score: float = 8.0,
    location: str = "",
    season: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Generate a FULL quality content pack matching the spec format.
    
    This produces:
      - content_formatted: 3500-4000 chars with the exact spec structure
      - universal_caption_block: [LOCATION] [SEASON] format
      - hashtags: exactly 5 (3 micro + 2 broad auto-matched)
      - All metadata for image generation
    
    Uses 2-phase EMADS-PR:
      Phase 1 (CTO): Generate full content
      Phase 2 (ReconcileGPT): Review + improve
    """
    import random as _rng
    
    if not location:
        locations = ["Chicago", "NYC", "LA", "Houston", "Phoenix", "Denver",
                     "Portland", "Seattle", "Austin", "Miami", "Atlanta", "Boston"]
        location = _rng.choice(locations)
    if not season:
        import datetime
        month = datetime.datetime.now().month
        season = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Fall", 10: "Fall", 11: "Fall"}[month]
    
    print(f"\n{'='*60}")
    print(f"  QUALITY CONTENT GENERATOR — Q&A Natural Style")
    print(f"{'='*60}")
    print(f"  Topic: {topic}")
    print(f"  Score: {score}/10 | Location: {location} | Season: {season}")
    print(f"{'='*60}")
    
    # ══ RETRY LOOP — generate → review → regenerate with feedback until 9.0+ ══
    MAX_ATTEMPTS = 3
    MIN_SCORE = 9.0
    GOOD_ENOUGH = 8.5  # accept after all attempts if best >= this
    best_pack = None
    best_score = 0.0
    prev_feedback = ""
    
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n  ── Attempt {attempt}/{MAX_ATTEMPTS} ──")
        
        # ── Phase 1: CTO Agent — Generate content as Q&A answer ──
        feedback_block = ""
        if prev_feedback:
            feedback_block = f"""
PREVIOUS ATTEMPT FEEDBACK (fix these issues):
{prev_feedback}

Regenerate the content fixing ALL issues above. Score MUST be higher than {best_score:.1f}.
"""
        
        quality_prompt = f"""Answer this nano-niche topic like a Perplexity AI expert:

TOPIC: {topic}
NICHE SCORE: {score}/10
LOCATION: {location}
SEASON: {season}
{feedback_block}
You are answering someone who asked "{topic}". Give them the REAL, COMPLETE answer with personality.

EXAMPLE OF THE EXACT FORMAT TO MATCH (real published post — notice NO Markdown):

Do you know banana peels make natural shoe polish that repels water too?

Yes, banana peels can serve as a natural DIY shoe polish for leather shoes due to their potassium content and oils, which mimic some commercial polishes for shine.

🌿 How to Use Banana Peels as Polish

Rub the soft inside of a fresh (slightly green) banana peel directly onto clean leather shoes in circular motions, covering all surfaces evenly. Let it sit for 10-15 minutes to allow oils to absorb, then buff with a soft, dry cloth to remove residue and reveal shine. Works on bags or belts too — eat the banana first for zero waste.

🫘 Effectiveness and Limitations

It provides a quick, temporary sheen from natural fats and potassium, outperforming nothing at all but not matching wax-based polishes for durability. No strong evidence supports robust water repellency — test on non-valuable items first, and pair with conditioner for better protection.

(notice: NO ** bold, NO ### headings, just clean plain text with emoji section headers)

FORMAT YOUR CONTENT EXACTLY LIKE THAT EXAMPLE. PLAIN TEXT ONLY.

Output as JSON:

{{
  "title": "Do you know [surprising specific claim from the topic with a number or fact]?",
  "content_formatted": "PLAIN TEXT answer. 3500-4000 chars. NO ** markers. NO ### headings. NO Markdown. Emoji section headers. Exact numbers. Witty tone.",
  "pain_point": "Ultra-specific problem (1 sentence with $ or timeframe)",
  "audiences": ["Specific persona 1", "Specific persona 2", "Specific persona 3"],
  "steps": [
    "Step 1: Specific action with exact number",
    "Step 2: Specific result with timeframe"
  ],
  "result": "Specific measurable outcome with numbers",
  "hashtags": ["NanoNiche1", "NanoNiche2", "NanoNiche3"],
  "image_title": "Short Title (max 4 words)",
  "image_subtitle": "Subtitle (max 5 words)",
  "image_steps": "Word1 • Word2 • Word3",
  "colors": [[60, 80, 40], [120, 160, 80]]
}}

CRITICAL:
1. content_formatted = PLAIN TEXT answer, 3500-4000 characters. NO Markdown. COUNT CAREFULLY.
2. Title MUST be a natural question: "Do you know [specific claim]?" — NOT generic like "Unlock the Power of..."
3. ABSOLUTELY NO ** bold markers — TikTok shows them as ugly literal ** characters.
4. ABSOLUTELY NO ### or ## headings — TikTok shows them as literal # characters.
5. Use emoji section headers: 🌿 🫘 🫙 ❌ ✅ on their own lines.
6. Use ALL CAPS for occasional emphasis instead of bold.
7. MUST include at least 15 EXACT numbers: $2/lb, 7-14 days, 4-6 inches, 350°F.
8. MUST have witty personality — dry humor, real talk, zero corporate tone.
9. Hashtags: exactly 3 NANO-NICHE tags (ultra-specific, high-search).
10. NO generic intro, NO "great question", NO motivation speeches, NO brand names.
11. Include sections: 🌿 main (2-3 subsections), 🫙 quick method (6 steps), ❌ mistakes (3-4), ✅ tips (4-5), punchy ending.
12. Output ONLY valid JSON."""

        result = call_llm(quality_prompt, system=QUALITY_CONTENT_SYSTEM, max_tokens=6000, temperature=0.6)
        
        if not result.success:
            print(f"  [QUALITY] Generation FAILED: {result.error[:100]}")
            continue
        
        pack = _extract_json(result.text)
        if not pack:
            print(f"  [QUALITY] JSON parse FAILED from {result.provider}")
            raw_len = len(result.text) if result.text else 0
            print(f"  [QUALITY] Raw text length: {raw_len}")
            print(f"  [QUALITY] Raw text (first 300): {repr(result.text[:300])}")
            print(f"  [QUALITY] Raw text (last 200): {repr(result.text[-200:])}")
            continue
        
        # ── Strip leftover Markdown + ensure section breaks (safety net) ──
        content = _strip_markdown(pack.get("content_formatted", ""))
        content = _ensure_section_breaks(content)
        pack["content_formatted"] = content
        content_len = len(content)
        print(f"  [QUALITY] Content length: {content_len} chars (target: 3500-4000)")
        
        # ── Iterative expansion (up to 2 passes) ──
        for _exp_pass in range(2):
            if content_len >= 3200:
                break
            print(f"  [QUALITY] Content short ({content_len} < 3200). Expansion pass {_exp_pass + 1}...")
            
            needed = 3700 - content_len  # aim for 3700 center of 3500-4000 range
            expand_prompt = f"""Your answer is only {content_len} characters. It MUST be 3500-4000 characters (you need ~{needed} more chars).

FORMATTING RULES (CRITICAL — TikTok shows raw text, NOT rendered Markdown):
- ABSOLUTELY NEVER use **bold** markers
- ABSOLUTELY NEVER use ### or ## headings
- Use emoji as section headers: 🌿 Section Name
- Use ALL CAPS sparingly for emphasis instead of bold
- Use plain numbered lists: 1. 2. 3.
- Use plain dashes: - for bullets

Expand this answer by ADDING concrete sections (don't rewrite what's already good):
- Add an emoji-headed section with 3-4 specific examples/variations with EXACT prices ($), timeframes, quantities
- Add ❌ What usually doesn't work section (3-4 specific mistakes with WHY they fail)
- Add ✅ Survival tips section (4-5 tips with exact numbers)
- Add a 🫙 Quick Method numbered list (6 steps with specific measurements)
- Add more witty personality — dry humor one-liners between sections
- Add real-world context: "In Miami during winter..." or "If you live in an apartment..."
- EVERY new sentence must teach something specific — zero filler

Current answer (EXPAND this, keep everything that's good):
{content}

Return the FULL expanded answer as PLAIN TEXT (3500-4000 chars). Count carefully. No JSON wrapper. No markdown."""
            
            expand_result = call_llm(expand_prompt, system=QUALITY_CONTENT_SYSTEM, max_tokens=6000, temperature=0.5)
            if expand_result.success and len(expand_result.text) > content_len:
                pack["content_formatted"] = _ensure_section_breaks(_strip_markdown(expand_result.text.strip()))
                content = pack["content_formatted"]
                content_len = len(content)
                print(f"  [QUALITY] Expanded to: {content_len} chars")
            else:
                break  # expansion failed, don't retry
        
        if content_len > 4200:
            # Too long — ask AI to trim to 3500-4000 without losing meaning
            print(f"  [QUALITY] Content long ({content_len} > 4200). Trimming...")
            trim_prompt = f"""This answer is {content_len} characters but MUST be 3500-4000 characters.

TRIM it to 3500-4000 chars by:
- Removing redundant sentences and filler words
- Cutting less important examples (keep the best ones)
- Making sentences more concise
- DO NOT remove key facts, specific numbers, or practical steps
- Keep the MEANING and FLOW intact
- NEVER add **bold** or ### headings — this is for TikTok (plain text only)

Content to trim:
{content}

Return the TRIMMED answer as PLAIN TEXT (3500-4000 chars). No JSON wrapper. No markdown."""
            
            trim_result = call_llm(trim_prompt, system=QUALITY_CONTENT_SYSTEM, max_tokens=5000, temperature=0.3)
            if trim_result.success and 3200 < len(trim_result.text) < 4200:
                pack["content_formatted"] = _ensure_section_breaks(_strip_markdown(trim_result.text.strip()))
                content_len = len(pack["content_formatted"])
                print(f"  [QUALITY] Trimmed to: {content_len} chars")
        
        # ── Hashtags: 3 micro + 2 broad = exactly 5 ──
        raw_micro = pack.get("hashtags", [])[:3]
        micro_tags = [t.lstrip('#').strip() for t in raw_micro if t.strip()]
        broad_tags = _auto_pick_broad_hashtags(topic, micro_tags, n=2)
        broad_tags = [t.lstrip('#').strip() for t in broad_tags]
        pack["hashtags"] = micro_tags + broad_tags
        _enforce_5_hashtags(pack, topic)
        
        # ── Metadata ──
        pack["_source"] = f"quality_ai_{result.provider}"
        pack["_niche_score"] = score
        pack["_gen_provider"] = result.provider
        pack["_gen_model"] = result.model
        pack["_gen_tokens"] = result.tokens_used
        pack["_gen_cost"] = result.cost_usd
        pack["_location"] = location
        pack["_season"] = season
        pack["_content_chars"] = len(pack.get("content_formatted", ""))
        
        # Convert colors
        _normalize_colors(pack)
        
        print(f"  [QUALITY] Title: {pack.get('title', '?')}")
        print(f"  [QUALITY] Provider: {result.provider}/{result.model}")
        print(f"  [QUALITY] Hashtags: {' '.join(pack['hashtags'])}")
        
        # ── Phase 2: ReconcileGPT — Quality Review ──
        review = _review_quality_content(pack)
        review_score = 0.0
        if review:
            review_score = review.get("avg", 0)
            pack["_review_score"] = review_score
            pack["_review_pass"] = review_score >= MIN_SCORE
            pack["_review_feedback"] = review.get("feedback", "")
            pack["_review_provider"] = review.get("_provider", "")
            
            if review.get("improved_title") and review_score < 9.5:
                pack["title"] = review["improved_title"]
            
            print(f"  [QUALITY] Review: {review_score:.1f}/10 — {'PASS ✓' if review_score >= MIN_SCORE else 'RETRYING...'}")
            if review.get("feedback"):
                print(f"  [QUALITY] Feedback: {review['feedback'][:150]}")
        
        # ── Track best ──
        if review_score > best_score:
            best_score = review_score
            best_pack = pack.copy()
        
        # ── Check if we reached target ──
        if review_score >= MIN_SCORE:
            print(f"  [QUALITY] ✓ Target {MIN_SCORE}+ reached on attempt {attempt}!")
            break
        
        # ── Prepare feedback for next attempt ──
        if attempt < MAX_ATTEMPTS:
            prev_feedback = review.get("feedback", "") if review else "Content quality insufficient."
            prev_feedback += f"\nPrevious score: {review_score:.1f}/10. Need {MIN_SCORE}+."
            print(f"  [QUALITY] Score {review_score:.1f} < {MIN_SCORE} — regenerating with feedback...")
    
    # Use best pack from all attempts
    pack = best_pack
    if not pack:
        print(f"  [QUALITY] All {MAX_ATTEMPTS} attempts failed.")
        return None
    
    # If best score >= GOOD_ENOUGH after all retries, mark as pass
    if best_score >= GOOD_ENOUGH:
        pack["_review_pass"] = True
    
    print(f"\n  [QUALITY] BEST SCORE: {best_score:.1f}/10 (from {MAX_ATTEMPTS} attempt(s))"
          f"{' ✓ ACCEPTED' if best_score >= GOOD_ENOUGH else ' ⚠ BELOW THRESHOLD'}")
    
    # ── Build Universal Caption Block (Title + Content + Hashtags for TikTok) ──
    tag_str = " ".join("#" + t.lstrip("#") for t in pack["hashtags"] if t.strip())
    title = pack.get("title", topic)
    content = pack.get("content_formatted", "")
    
    caption_lines = [title, "", content, "", tag_str]
    
    pack["universal_caption_block"] = "\n".join(caption_lines)
    
    # ── Phase 3: AI Image Generation ──
    print(f"\n  [QUALITY] Phase 3: Generating AI realistic image...")
    image_path = generate_image_for_pack(pack)
    if image_path:
        print(f"  [QUALITY] ✓ AI image generated: {image_path}")
    else:
        print(f"  [QUALITY] ℹ AI image skipped — will use PIL gradient at publish time")
    
    return pack


def _review_quality_content(pack: Dict[str, Any]) -> Optional[Dict]:
    """Enhanced ReconcileGPT review — strict scoring, target 9.0+."""
    content = pack.get("content_formatted", "")
    content_len = len(content)
    
    review_prompt = f"""You are a STRICT content quality reviewer. Score honestly — 9 or 10 means EXCELLENT.

TITLE: {pack.get('title', '')}
PAIN POINT: {pack.get('pain_point', '')}
CONTENT LENGTH: {content_len} characters (target: 3200-4000)

FULL CONTENT:
{content[:3500]}

HASHTAGS: {pack.get('hashtags', [])}
STEPS: {json.dumps(pack.get('steps', []))}

Score each 1-10 (be STRICT — 10 = professional-grade, 7 = mediocre):
1. ANSWER_QUALITY — Does the content ACTUALLY answer the topic like a Perplexity AI expert? Specific facts, not fluff?
2. CONTENT_DEPTH — 3200-4000 chars of REAL value? Every sentence teaches something? (3000+ is acceptable if dense)
3. TONE — Casual, witty, personality-driven? Dry humor? NOT corporate, NOT generic blog-speak?
4. HOOK — Does the title start with "Do you know" or a natural question? Or is it generic clickbait like "Unlock the Power of..."?
5. SPECIFICITY — Concrete numbers ($prices, timeframes, quantities, temperatures)? At least 15 specific numbers? Or vague advice?
6. ACTIONABILITY — Reader can do this TODAY with what they have?
7. FORMATTING — PLAIN TEXT ONLY. NO ** bold? NO ### headings? NO Markdown at all? Uses emoji section headers (🌿 🫙 ❌ ✅)?
   DEDUCT 2 points if you find ANY ** or ### or ## markers — these show as literal ugly characters on TikTok.

For EACH criterion below 9, explain SPECIFICALLY what's wrong and how to fix it.

Output ONLY valid JSON:
{{"scores": {{"answer_quality": N, "content_depth": N, "tone": N, "hook": N, "specificity": N, "actionability": N, "formatting": N}}, "avg": N.N, "pass": true/false, "feedback": "Specific issues to fix (2-3 sentences, be actionable)", "improved_title": "Do you know [specific claim]? — rewrite if title is generic clickbait, else keep same"}}"""
    
    gen_provider = pack.get("_gen_provider", "")
    review_providers = ["github_models", "perplexity", "gemini", "openai"]
    if gen_provider in review_providers:
        review_providers.remove(gen_provider)
        review_providers.append(gen_provider)
    
    result = call_llm(review_prompt, system=QUALITY_REVIEW_SYSTEM, max_tokens=600, temperature=0.2, providers=review_providers)
    
    if not result.success:
        return None
    
    review = _extract_json(result.text)
    if review:
        review["_provider"] = result.provider
        scores = review.get("scores", {})
        if scores:
            avg = sum(scores.values()) / len(scores)
            review["avg"] = round(avg, 1)
            review["pass"] = avg >= 9.0
    
    return review


def get_unused_topics(top_n: int = 10) -> List[Tuple[str, float, str, str]]:
    """Get top niche_hunter topics that haven't been published yet.
    
    Returns list of (topic, score, niche, hook) tuples.
    """
    import sqlite3
    
    db_path = os.path.join(os.path.dirname(__file__), "niche_hunter.db")
    if not os.path.exists(db_path):
        return []
    
    # Get published titles (direct sqlite3 — avoids circular import from web.app)
    published_titles = set()
    viralops_db = os.path.join(os.path.dirname(__file__), "web", "viralops.db")
    if os.path.exists(viralops_db):
        try:
            _vconn = sqlite3.connect(viralops_db)
            rows = _vconn.execute("SELECT title FROM posts WHERE status = 'published'").fetchall()
            published_titles = {r[0].lower() for r in rows}
            _vconn.close()
        except Exception as e:
            print(f"  [DEDUP] Warning: could not load published titles: {e}")
    
    conn = sqlite3.connect(db_path)
    all_topics = conn.execute(
        "SELECT topic, final_score, niche, COALESCE(hook, '') FROM niche_scores ORDER BY final_score DESC LIMIT ?",
        (top_n * 5,),
    ).fetchall()
    conn.close()
    
    # Filter out already-published (keyword overlap check)
    unused = []
    for topic, score, niche, hook in all_topics:
        topic_words = set(topic.lower().split()[:6])
        is_used = False
        for title in published_titles:
            title_words = set(title.lower().split()[:6])
            if len(topic_words & title_words) >= 3:
                is_used = True
                break
        if not is_used:
            unused.append((topic, score, niche, hook))
        if len(unused) >= top_n:
            break
    
    return unused


def _normalize_colors(pack: dict) -> None:
    """Normalize color data from JSON list to tuple format."""
    if "colors" in pack:
        colors = pack["colors"]
        if isinstance(colors, list) and len(colors) == 2:
            try:
                pack["colors"] = (tuple(colors[0]), tuple(colors[1]))
            except (TypeError, IndexError):
                pack["colors"] = ((60, 80, 40), (120, 160, 80))
        else:
            pack["colors"] = ((60, 80, 40), (120, 160, 80))
    else:
        pack["colors"] = ((60, 80, 40), (120, 160, 80))


# System prompt for basic review (5 criteria, threshold 7.0)
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

# System prompt for quality review (7 criteria, threshold 9.0) — used by _review_quality_content()
QUALITY_REVIEW_SYSTEM = """You are ReconcileGPT — a STRICT content quality review agent for premium TikTok posts.
Your job: analyze content for depth, specificity, tone, and publishing readiness.
Score HONESTLY — 9 or 10 means EXCELLENT professional-grade content.

Score 1-10 on each criteria:
1. ANSWER_QUALITY — Does the content actually answer the topic like a Perplexity AI expert? Specific facts, not fluff?
2. CONTENT_DEPTH — 3200-4000 chars of REAL value? Every sentence teaches something?
3. TONE — Casual, witty, personality-driven? Dry humor? NOT corporate, NOT generic blog-speak?
4. HOOK — Does the title start with "Do you know" or a natural question? Would the first 2 sentences stop someone scrolling?
5. SPECIFICITY — Concrete numbers ($prices, timeframes, quantities, temperatures)? At least 15 specific numbers? Or vague advice?
6. ACTIONABILITY — Reader can do this TODAY with what they have?
7. FORMATTING — PLAIN TEXT ONLY for TikTok. Uses emoji section headers (🌿 🫙 ❌ ✅)? NO ** bold markers? NO ### headings? NO Markdown at all? Clean and readable?
   DEDUCT 2 POINTS if content contains ** or ### or ## — these show as ugly literal characters on TikTok.

Output ONLY valid JSON:
{"scores": {"answer_quality": N, "content_depth": N, "tone": N, "hook": N, "specificity": N, "actionability": N, "formatting": N}, "avg": N.N, "pass": true/false, "feedback": "Specific issues to fix", "improved_title": "Do you know [specific claim]? — rewrite if title is generic clickbait"}
Pass threshold: avg >= 9.0"""


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
    
    # Enforce exactly 5 hashtags
    _enforce_5_hashtags(pack, topic)
    
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
    _normalize_colors(pack)
    
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
    
    # Get already-published titles for dedup (direct sqlite3 — avoids circular import from web.app)
    published_titles = set()
    viralops_db = os.path.join(os.path.dirname(__file__), "web", "viralops.db")
    if os.path.exists(viralops_db):
        try:
            _vconn = sqlite3.connect(viralops_db)
            rows = _vconn.execute("SELECT title FROM posts WHERE status = 'published'").fetchall()
            published_titles = {r[0] for r in rows}
            _vconn.close()
        except Exception as e:
            print(f"  [DEDUP] Warning: could not load published titles: {e}")
    
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

def _repair_json_text(text: str) -> str:
    """Fix common LLM JSON issues: raw newlines/tabs/control chars in strings, smart quotes, etc."""
    # Replace smart quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    # Replace em-dash variants that can confuse parsers
    text = text.replace('\u2014', '-').replace('\u2013', '-')
    
    # Fix raw control characters inside JSON string values
    result = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\\' and in_string and i + 1 < len(text):
            next_ch = text[i + 1]
            # Valid JSON escapes: " \\ / b f n r t u
            if next_ch in '"\\/bfnrtu':
                result.append(ch)
                result.append(next_ch)
                i += 2
                continue
            else:
                # Invalid escape like \x, \a etc → double-escape it
                result.append('\\\\')
                i += 1
                continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue
        if in_string:
            if ch == '\n':
                result.append('\\n')
                i += 1
                continue
            if ch == '\r':
                result.append('\\r')
                i += 1
                continue
            if ch == '\t':
                result.append('\\t')
                i += 1
                continue
            # Strip other control chars (0x00-0x1F except the ones handled above)
            if ord(ch) < 0x20:
                result.append(' ')
                i += 1
                continue
        result.append(ch)
        i += 1
    return ''.join(result)


def _extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response (handles markdown code blocks, nested JSON, raw newlines).
    
    Uses a multi-strategy approach with progressively more aggressive cleaning:
    1. Strip markdown fences (handles ```json, ```JSON, leading spaces, multiple blocks)
    2. Direct parse
    3. Repair (fix control chars, smart quotes, invalid escapes) then parse
    4. Brace-matching extraction + repair
    5. Aggressive cleaning (strip all non-JSON prose) + parse
    """
    import re
    
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # ----- Strategy 0: Strip markdown code block wrapper(s) -----
    # Handle: ```json, ```JSON, ``` json, leading whitespace before ```, multiple blocks
    # Also handle case where response has text before/after the code block
    code_block_match = re.search(r'```(?:json|JSON)?\s*\n(.*?)```', text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()
    elif text.startswith("```"):
        # Fallback: simple strip
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    
    # ----- Strategy 1: Direct parse -----
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # ----- Strategy 2: Repair common LLM JSON issues then parse -----
    try:
        return json.loads(_repair_json_text(text))
    except (json.JSONDecodeError, ValueError, Exception):
        pass
    
    # ----- Strategy 3: Brace-matching extraction -----
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape_next = False
        end = start
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
        if depth == 0 and end > start:
            candidate = text[start:end]
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                pass
            try:
                return json.loads(_repair_json_text(candidate))
            except (json.JSONDecodeError, ValueError, Exception):
                pass
    
    # ----- Strategy 4: Aggressive cleaning -----
    # Sometimes LLM wraps JSON in extra text. Try to find the outermost { ... }
    # and aggressively clean the content.
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        aggressive = text[first_brace:last_brace + 1]
        # Replace literal \n (two chars) that should be \\n in JSON strings
        # This handles cases where the LLM outputs actual backslash-n instead of newline
        try:
            return json.loads(_repair_json_text(aggressive))
        except (json.JSONDecodeError, ValueError, Exception):
            pass
        # Even more aggressive: strip control chars entirely
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', aggressive)
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            return json.loads(_repair_json_text(cleaned))
        except (json.JSONDecodeError, ValueError, Exception):
            pass
    
    # ----- Strategy 5: Last resort — find ANY valid JSON object -----
    # Try increasingly simple regex patterns
    for pattern in [
        r'\{.*\}',  # Greedy — largest possible match
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Non-nested or single-nested
    ]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass
            try:
                return json.loads(_repair_json_text(match.group()))
            except (json.JSONDecodeError, ValueError, Exception):
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

def _display_quality_pack(pack: Dict[str, Any]) -> None:
    """Pretty-print a quality content pack for human review."""
    print(f"\n{'═'*70}")
    print(f"  📝 QUALITY CONTENT PACK — Review Before Publishing")
    print(f"{'═'*70}")
    
    print(f"\n  📌 Title: {pack.get('title', '?')}")
    print(f"  🎯 Pain Point: {pack.get('pain_point', '?')}")
    print(f"  👥 Audiences: {', '.join(pack.get('audiences', []))}")
    print(f"  📍 Location: {pack.get('_location', '?')} | Season: {pack.get('_season', '?')}")
    print(f"  🏆 Niche Score: {pack.get('_niche_score', 'N/A')}/10")
    print(f"  🤖 Provider: {pack.get('_gen_provider', '?')}/{pack.get('_gen_model', '?')}")
    
    # Review scores
    review_score = pack.get('_review_score', 0)
    review_pass = pack.get('_review_pass', False)
    if review_score:
        print(f"  ⭐ Review Score: {review_score:.1f}/10 — {'PASS ✓' if review_pass else 'NEEDS WORK ✗'}")
        if pack.get('_review_feedback'):
            print(f"  💬 Feedback: {pack['_review_feedback'][:120]}")
    
    # Content
    content = pack.get("content_formatted", "")
    print(f"\n{'─'*70}")
    print(f"  CONTENT ({len(content)} chars — target: 3500-4000)")
    print(f"{'─'*70}")
    # Print first 2000 chars for preview
    if len(content) > 2000:
        print(content[:2000])
        print(f"\n  ... [{len(content) - 2000} more chars] ...")
    else:
        print(content)
    
    # Universal Caption
    caption = pack.get("universal_caption_block", "")
    print(f"\n{'─'*70}")
    print(f"  UNIVERSAL CAPTION BLOCK")
    print(f"{'─'*70}")
    print(caption)
    
    # Steps & Result
    print(f"\n{'─'*70}")
    print(f"  STEPS + RESULT")
    print(f"{'─'*70}")
    for i, step in enumerate(pack.get("steps", []), 1):
        import re as _re
        clean = _re.sub(r'^Step\s*\d+\s*:\s*', '', step)
        print(f"  Step {i}: {clean}")
    print(f"  Result: {pack.get('result', '?')}")
    
    # Hashtags — normalize (strip leading # to avoid double ##)
    tags = [t.lstrip('#') for t in pack.get('hashtags', []) if t.strip()]
    print(f"\n  #️⃣  Hashtags: {' '.join('#' + t for t in tags)}")
    
    # Image metadata
    print(f"\n  🖼️  Image: {pack.get('image_title', '?')} | {pack.get('image_subtitle', '?')}")
    print(f"  📐 Image Steps: {pack.get('image_steps', '?')}")
    
    # AI image status
    ai_image = pack.get("_ai_image_path", "")
    if ai_image:
        print(f"  🎨 AI Image: ✓ {ai_image}")
        print(f"  📸 Prompt: {pack.get('_image_prompt', '')[:100]}...")
    else:
        print(f"  🎨 AI Image: ✗ (will use PIL gradient at publish time)")
    
    print(f"\n{'═'*70}")


def cli_single_mode():
    """Interactive single-post generation — pick topic → generate → review → export."""
    print(f"\n{'═'*70}")
    print(f"  🎯 VIRALOPS — Quality Single Post Generator")
    print(f"  Creates ONE spec-compliant post (3500-4000 chars content)")
    print(f"{'═'*70}")
    
    # Show available unused topics
    unused = get_unused_topics(top_n=10)
    
    if unused:
        print(f"\n  📊 Top unused niche_hunter topics:")
        for i, (topic, score, niche, _hook) in enumerate(unused, 1):
            print(f"    {i}) [{score:.2f}] {topic[:65]}")
        print(f"\n  Enter number (1-{len(unused)}) to pick, or type custom topic:")
    else:
        print(f"\n  No niche_hunter topics available. Enter a custom topic:")
    
    try:
        choice = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return
    
    if not choice:
        print("  No input — using first available topic.")
        choice = "1"
    
    # Determine topic
    topic = ""
    score = 8.0
    db_hook = ""
    if choice.isdigit() and unused:
        idx = int(choice) - 1
        if 0 <= idx < len(unused):
            topic, score, niche, db_hook = unused[idx]
        else:
            topic, score, niche, db_hook = unused[0]
    else:
        topic = choice
        score = 8.0
    
    # Generate
    pack = generate_quality_post(topic, score)
    if pack and db_hook:
        pack["_db_hook"] = db_hook
    
    if not pack:
        print(f"\n  ❌ Generation failed. Try again or check API keys.")
        return
    
    # Display for review
    _display_quality_pack(pack)
    
    # Save to file for review
    output_dir = POSTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    import re
    safe_title = re.sub(r'[^\w-]', '', pack.get('title', 'post').split('\u2014')[0].split(':')[0])[:20].strip('_')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{timestamp}_{safe_title}.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n  💾 Saved to: {output_file}")
    print(f"\n  Next: review the content, then publish with:")
    print(f"    python llm_content.py publish {output_file}")


def cli_publish_mode(filepath: str):
    """Publish a previously generated quality post."""
    if not os.path.exists(filepath):
        print(f"  ❌ File not found: {filepath}")
        return
    
    with open(filepath, "r", encoding="utf-8") as f:
        pack = json.load(f)
    
    _display_quality_pack(pack)
    
    print(f"\n  🚀 Ready to publish to TikTok via Publer.")
    try:
        confirm = input("  Publish? (yes/no): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return
    
    if confirm not in ("yes", "y"):
        print("  Skipped.")
        return
    
    # Convert colors back to tuple if needed
    if isinstance(pack.get("colors"), list):
        pack["colors"] = (tuple(pack["colors"][0]), tuple(pack["colors"][1]))
    
    # Use publish_microniche.main() flow
    try:
        import publish_microniche as pm
        # Set the pack on the module level and call main with the pack
        result = pm.main(content_pack_override=pack)
        if result:
            print(f"\n  ✅ Published successfully!")
        else:
            print(f"\n  ❌ Publishing failed — check logs above.")
    except Exception as e:
        print(f"\n  ❌ Publish error: {e}")


# ═══════════════════════════════════════════════════════════════
# HELPER — enforce exactly 5 hashtags
# ═══════════════════════════════════════════════════════════════

def _enforce_5_hashtags(pack: Dict[str, Any], topic: str = "") -> Dict[str, Any]:
    """Ensure pack has EXACTLY 5 hashtags — trim excess, pad with broad if short."""
    raw = pack.get("hashtags", [])
    # Normalize: strip '#' prefix, deduplicate, remove blanks
    seen = set()
    tags: List[str] = []
    for t in raw:
        clean = t.lstrip('#').strip()
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            tags.append(clean)
    
    if len(tags) > 5:
        tags = tags[:5]
    
    if len(tags) < 5:
        # Pad with broad hashtags
        extra = _auto_pick_broad_hashtags(topic or pack.get("title", ""), tags, n=5 - len(tags))
        for e in extra:
            clean = e.lstrip('#').strip()
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                tags.append(clean)
                if len(tags) >= 5:
                    break
    
    # Final hard cap
    pack["hashtags"] = tags[:5]
    return pack


# ═══════════════════════════════════════════════════════════════
# HELPER — save pack to JSON
# ═══════════════════════════════════════════════════════════════

def _save_pack(pack: Dict[str, Any], label: str = "") -> str:
    """Save a content pack to POSTS_DIR and return the filepath."""
    _enforce_5_hashtags(pack)  # final gate — always exactly 5
    import re as _re
    os.makedirs(POSTS_DIR, exist_ok=True)
    safe = _re.sub(r'[^\w-]', '', pack.get('title', 'post').split('\u2014')[0].split(':')[0])[:20].strip('_')
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(POSTS_DIR, f"{ts}_{safe}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  💾 Saved: {out}")
    return out


def _pick_post_file() -> Optional[str]:
    """Let user pick a saved JSON post. Returns filepath or None."""
    if not os.path.exists(POSTS_DIR):
        print("  No saved posts yet.")
        return None
    files = sorted([f for f in os.listdir(POSTS_DIR) if f.endswith(".json")])
    if not files:
        print("  No saved posts yet.")
        return None
    print(f"\n  📂 Saved posts ({POSTS_DIR}):")
    for i, fn in enumerate(files, 1):
        print(f"    {i}) {fn}")
    try:
        c = input(f"  Pick (1-{len(files)}): ").strip()
        if c.isdigit() and 1 <= int(c) <= len(files):
            return os.path.join(POSTS_DIR, files[int(c) - 1])
    except (EOFError, KeyboardInterrupt):
        pass
    return None


# ═══════════════════════════════════════════════════════════════
# METHOD HANDLERS — each generates/acts then returns
# ═══════════════════════════════════════════════════════════════

def _menu_quality_post():
    """[1] Quality Post — 3500-4000 chars, review, AI image (spec-compliant)."""
    cli_single_mode()


def _menu_ai_niche():
    """[2] AI Niche Post — pick top niche_hunter topic → LLM cascade → content."""
    pack = generate_from_niche_hunter(top_n=10)
    if pack:
        _display_quality_pack(pack) if pack.get("content_formatted") else print(json.dumps(pack, indent=2, default=str))
        _save_pack(pack)
    else:
        print("  ❌ AI Niche generation failed.")


def _menu_prewritten():
    """[3] Pre-written Packs — 8 curated packs (zero API cost)."""
    try:
        import publish_microniche as pm
        pack = pm.get_content_pack("hunter_prewritten")
        if pack:
            title = pack.get("title", "?")
            print(f"\n  📌 Title: {title}")
            print(f"  🎯 Pain Point: {pack.get('pain_point', '?')}")
            print(f"  👥 Audiences: {', '.join(pack.get('audiences', []))}")
            steps = pack.get("steps", [])
            for i, s in enumerate(steps, 1):
                print(f"  Step {i}: {s}")
            print(f"  Result: {pack.get('result', '?')}")
            tags = pack.get("hashtags", [])
            print(f"  #️⃣  {' '.join('#' + t.lstrip('#') for t in tags)}")
            _save_pack(pack, "prewritten")
    except Exception as e:
        print(f"  ❌ Error: {e}")


def _menu_gemini_microniche():
    """[4] Gemini Micro-Niche — 20 micro + 10 nano + 10 real-life niches → JSON."""
    import random as _rnd
    try:
        import publish_microniche as pm
        pack = pm.get_content_pack("gemini")
        if pack:
            print(f"\n  📌 Title: {pack.get('title', '?')}")
            _save_pack(pack, "gemini_niche")
    except Exception as e:
        print(f"  ❌ Gemini error: {e}")
        print("  (Gemini quota may be exhausted — try option 1 or 2 instead)")


def _menu_viral_framework():
    """[5] Viral Framework — Problem-Agitate-Solution / Before-After-Bridge."""
    try:
        from agents.content_factory import generate_content_pack as cf_generate
        # Build a state dict for content_factory
        unused = get_unused_topics(top_n=5)
        if unused:
            topic, score, niche, _hook = unused[0]
            print(f"  Topic: {topic}")
        else:
            topic = input("  Enter topic: ").strip() or "How to grow mushrooms indoors?"
            niche = "homesteading"
        
        state = {
            "niche": niche,
            "topic": topic,
            "platform": "tiktok",
            "budget_remaining_pct": 80.0,
        }
        result = cf_generate(state)
        if result and result.get("content_pack"):
            pack = result["content_pack"]
            print(f"\n  📌 Title: {pack.get('title', '?')}")
            print(f"  🤖 Provider: {result.get('provider_used', '?')}")
            content = pack.get("content", pack.get("content_formatted", ""))
            print(f"  📏 Content: {len(content)} chars")
            print(f"\n{content[:1500]}...")
            _save_pack(pack, "viral_framework")
        else:
            print("  ❌ Content factory generation failed.")
    except Exception as e:
        print(f"  ❌ Error: {e}")


def _menu_rss_rewrite():
    """[6] RSS Rewrite — Fetch blog RSS → AI transform 70%+ → publish."""
    try:
        from integrations.rss_auto_poster import RSSAutoPoster
        poster = RSSAutoPoster()
        print("  Fetching RSS feeds...")
        feeds = poster.get_feeds()
        if feeds:
            print(f"  Found {len(feeds)} feeds")
            for i, f in enumerate(feeds[:5], 1):
                print(f"    {i}) {f.get('title', '?')[:60]}")
            poster.tick()  # Process one item
        else:
            print("  No RSS feeds configured. Add feeds in config or .env")
    except Exception as e:
        print(f"  ❌ RSS error: {e}")


def _menu_template_fallback():
    """[7] Template Fallback — caption_templates.json (zero API cost)."""
    try:
        from agents.content_factory import _fallback_generate, _load_niches_yaml, _extract_niche_data
        niches_yaml = _load_niches_yaml()
        niche_keys = list(niches_yaml.get("niches", {}).keys())
        if niche_keys:
            import random as _rnd
            nk = _rnd.choice(niche_keys)
            niche_config = _extract_niche_data(niches_yaml["niches"][nk])
            niche_config["key"] = nk
            pack = _fallback_generate(niche_config)
            print(f"\n  📌 Title: {pack.get('title', '?')}")
            print(f"  📏 Content: {len(pack.get('content', ''))} chars")
            print(f"  Method: Template fallback (no API call)")
            content = pack.get("content", "")
            print(f"\n{content[:1000]}...")
            _save_pack(pack, "template")
        else:
            print("  ⚠️ niches.yaml empty or not found")
    except Exception as e:
        print(f"  ❌ Error: {e}")


def _menu_platform_adapt():
    """[8] Platform Adaptation — convert saved post → other platform format."""
    fp = _pick_post_file()
    if not fp:
        return
    with open(fp, "r", encoding="utf-8") as f:
        pack = json.load(f)
    
    platforms = ["tiktok", "instagram", "facebook", "youtube", "linkedin",
                 "twitter", "pinterest", "reddit", "medium", "threads",
                 "bluesky", "mastodon", "quora", "tumblr", "shopify_blog", "lemon8"]
    print(f"\n  📱 Available platforms:")
    for i, p in enumerate(platforms, 1):
        print(f"    {i:2d}) {p}")
    try:
        c = input(f"  Pick platform (1-{len(platforms)}): ").strip()
        if c.isdigit() and 1 <= int(c) <= len(platforms):
            platform = platforms[int(c) - 1]
            from agents.content_factory import adapt_for_platform
            adapted = adapt_for_platform(pack, platform)
            print(f"\n  ✅ Adapted for {platform}:")
            print(json.dumps(adapted, indent=2, ensure_ascii=False, default=str)[:2000])
    except Exception as e:
        print(f"  ❌ Error: {e}")


def _menu_gen_image():
    """[9] Generate AI Image — for existing or new post."""
    fp = _pick_post_file()
    if not fp:
        return
    with open(fp, "r", encoding="utf-8") as f:
        pack = json.load(f)
    print(f"\n  Generating AI image for: {pack.get('title', '?')}")
    result = generate_image_for_pack(pack)
    if result:
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(pack, f, indent=2, ensure_ascii=False, default=str)
        print(f"  ✅ Image generated and JSON updated!")
    else:
        print(f"  ❌ Image generation failed.")


def _menu_hashtag_matrix():
    """[10] Hashtag Generator — 7-layer matrix OR 5 micro-niche strategy."""
    # Show available niches first
    try:
        from hashtags.matrix_5layer import (
            generate_hashtag_matrix, generate_5cap, generate_micro_niche_5,
            get_available_niches,
        )
        niches = get_available_niches()
        if niches:
            print(f"\n  📂 Available niches in DB: {', '.join(niches[:15])}")
    except ImportError:
        print("  ❌ hashtags/matrix_5layer.py not found")
        return

    print(f"\n  Hashtag strategies:")
    print(f"    [a] 5 Micro-Niche Hashtags (highest search, NO broad/generic)")
    print(f"    [b] 7-Layer Full Matrix (broad→local→micro→nano→trend→creator→seasonal)")
    print(f"    [c] 5-cap Instagram (5 highest search for IG)")
    try:
        strategy = input("  Pick (a/b/c): ").strip().lower() or "a"
    except (EOFError, KeyboardInterrupt):
        return

    topic = input("  Enter niche keyword (e.g. plant_based, homesteading): ").strip()
    if not topic:
        topic = "plant_based"

    location = input("  Location (optional, e.g. NYC): ").strip() or None

    try:
        if strategy == "a":
            # 5 Micro-Niche — the money strategy
            kw_input = input("  Extra topic keywords (comma-sep, optional): ").strip()
            kws = [k.strip() for k in kw_input.split(",") if k.strip()] if kw_input else None
            result = generate_micro_niche_5(topic, platform="tiktok", location=location, topic_keywords=kws)
            tags = result.get("hashtags", [])
            print(f"\n  🏷️  5 Micro-Niche Hashtags for '{topic}':")
            for i, t in enumerate(tags, 1):
                print(f"    {i}) {t}")
            print(f"\n  Strategy: {result.get('strategy')}")
            print(f"  Curated: {result.get('curated_count')}/{len(tags)} from DB")
            print(f"  Season: {result.get('season')}")
            print(f"  Copy-paste: {' '.join(tags)}")

        elif strategy == "b":
            # Full 7-layer matrix
            matrix = generate_hashtag_matrix(topic, location=location)
            print(f"\n  🏷️  7-Layer Hashtag Matrix for '{topic}':")
            for layer, tags in matrix.items():
                if isinstance(tags, list) and tags:
                    print(f"    {layer:16s}: {' '.join(tags[:6])}")
            combined = matrix.get("combined", [])
            if combined:
                print(f"\n  Combined ({len(combined)}): {' '.join(combined[:10])}")

        elif strategy == "c":
            # Instagram 5-cap
            tags = generate_5cap(topic, location=location)
            print(f"\n  🏷️  Instagram 5-Cap for '{topic}':")
            for i, t in enumerate(tags, 1):
                print(f"    {i}) {t}")
            print(f"\n  Copy-paste: {' '.join(tags)}")

        else:
            print(f"  Unknown strategy: {strategy}")

    except Exception as e:
        print(f"  ❌ Error: {e}")


def _menu_publish():
    """[11] Publish — send saved post to TikTok via Publer."""
    fp = _pick_post_file()
    if fp:
        cli_publish_mode(fp)


def _menu_batch_publish():
    """[12] Batch Publish — publish multiple posts with auto-gap."""
    try:
        import publish_microniche as pm
        try:
            count = int(input("  How many posts? (1-5): ").strip() or "1")
            gap = int(input("  Gap in minutes? (default 3): ").strip() or "3")
        except ValueError:
            count, gap = 1, 3
        count = max(1, min(count, 5))
        pm.batch_publish(count=count, gap_minutes=gap)
    except Exception as e:
        print(f"  ❌ Error: {e}")


def _menu_topics():
    """[13] Browse Topics — list unused niche_hunter topics."""
    unused = get_unused_topics(top_n=20)
    print(f"\n{'='*60}")
    print(f"  UNUSED NICHE HUNTER TOPICS")
    print(f"{'='*60}")
    for i, (topic, score, niche, _hook) in enumerate(unused, 1):
        print(f"  {i:2d}) [{score:.2f}] [{niche:20s}] {topic}")
    print(f"\n  Total: {len(unused)} unused topics available")


def _menu_test_providers():
    """[14] Test Providers — check which LLM APIs are working."""
    test_providers()


def _menu_custom_topic():
    """[15] Custom Topic — type any topic → quality post generation."""
    topic = input("  Enter your topic: ").strip()
    if not topic:
        print("  No topic entered.")
        return
    pack = generate_quality_post(topic, score=8.0)
    if pack:
        _display_quality_pack(pack)
        _save_pack(pack)
    else:
        print("  ❌ Generation failed.")


# ═══════════════════════════════════════════════════════════════
# INTERACTIVE MENU
# ═══════════════════════════════════════════════════════════════

MENU_OPTIONS = [
    # (key, label, handler, category)
    ("1",  "Quality Post (3500-4000 chars + AI image)",  _menu_quality_post,      "A"),
    ("2",  "AI Niche Post (niche_hunter DB → LLM)",      _menu_ai_niche,          "A"),
    ("3",  "Pre-written Packs (zero API cost)",           _menu_prewritten,        "A"),
    ("4",  "Gemini Micro-Niche Generator",                _menu_gemini_microniche, "A"),
    ("5",  "Viral Framework (PAS / BAB prompts)",         _menu_viral_framework,   "A"),
    ("6",  "RSS Rewrite (blog → AI transform 70%+)",      _menu_rss_rewrite,       "A"),
    ("7",  "Template Fallback (no API needed)",           _menu_template_fallback, "A"),
    ("8",  "Platform Adaptation (→ IG/FB/YT/LinkedIn…)",  _menu_platform_adapt,    "B"),
    ("9",  "Generate AI Image (Pollinations/Gemini)",     _menu_gen_image,         "C"),
    ("10", "Hashtags (5 micro-niche / 7-layer / 5-cap IG)",    _menu_hashtag_matrix,    "D"),
    ("11", "Publish to TikTok (Publer)",                  _menu_publish,           "E"),
    ("12", "Batch Publish (multi-post + auto-gap)",       _menu_batch_publish,     "E"),
    ("13", "Browse Unused Topics",                        _menu_topics,            "F"),
    ("14", "Test LLM Providers",                          _menu_test_providers,    "F"),
    ("15", "Custom Topic → Quality Post",                 _menu_custom_topic,      "A"),
]

CATEGORY_NAMES = {
    "A": "Content Generation",
    "B": "Content Transforms",
    "C": "Media Generation",
    "D": "Hashtags",
    "E": "Publishing",
    "F": "Tools",
}


def interactive_menu():
    """Main interactive menu — user picks a number, we execute that method."""
    while True:
        print(f"\n{'═'*70}")
        print(f"  🚀 VIRALOPS ENGINE — Content Factory Menu")
        print(f"  {time.strftime('%Y-%m-%d %H:%M')} | Data: {DATA_ROOT}")
        print(f"{'═'*70}")
        
        current_cat = None
        for key, label, _, cat in MENU_OPTIONS:
            if cat != current_cat:
                current_cat = cat
                print(f"\n  ── {CATEGORY_NAMES[cat]} ──")
            print(f"    [{key:>2s}] {label}")
        
        print(f"\n    [ 0] Exit")
        print(f"{'─'*70}")
        
        try:
            choice = input("  Pick a number ▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye! 👋")
            break
        
        if choice in ("0", "q", "quit", "exit"):
            print("  Bye! 👋")
            break
        
        # Find matching handler
        handler = None
        for key, label, fn, _ in MENU_OPTIONS:
            if choice == key:
                handler = fn
                print(f"\n{'─'*70}")
                print(f"  ▶ {label}")
                print(f"{'─'*70}")
                break
        
        if handler:
            try:
                handler()
            except KeyboardInterrupt:
                print("\n  ⏹ Interrupted.")
            except Exception as e:
                print(f"\n  ❌ Error: {e}")
        else:
            print(f"  ⚠️ Unknown option: {choice}")
        
        print()  # Blank line before re-showing menu


if __name__ == "__main__":
    import sys
    
    # If CLI arg given, use direct command mode (backward compatible)
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "test":
            test_providers()
        elif cmd == "single":
            cli_single_mode()
        elif cmd == "quality":
            topic = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
            if not topic:
                unused = get_unused_topics(top_n=5)
                if unused:
                    topic, score, niche, _hook = unused[0]
                    print(f"  Auto-picked: {topic} (score={score:.2f})")
                else:
                    topic = "How to grow mushrooms in apartment with no sunlight?"
                    score = 8.0
            else:
                score = 8.0
            pack = generate_quality_post(topic, score)
            if pack:
                _display_quality_pack(pack)
                _save_pack(pack)
        elif cmd == "publish":
            fp = sys.argv[2] if len(sys.argv) > 2 else ""
            if not fp:
                fp = _pick_post_file()
            if fp:
                cli_publish_mode(fp)
        elif cmd == "topics":
            _menu_topics()
        elif cmd == "genimage":
            fp = sys.argv[2] if len(sys.argv) > 2 else ""
            if not fp:
                fp = _pick_post_file()
            if fp and os.path.exists(fp):
                with open(fp, "r", encoding="utf-8") as f:
                    pack = json.load(f)
                print(f"\n  Generating AI image for: {pack.get('title', '?')}")
                result = generate_image_for_pack(pack)
                if result:
                    with open(fp, "w", encoding="utf-8") as f:
                        json.dump(pack, f, indent=2, ensure_ascii=False, default=str)
                    print(f"\n  ✅ Image generated and JSON updated!")
                else:
                    print(f"\n  ❌ Image generation failed.")
        elif cmd == "menu":
            interactive_menu()
        else:
            print("Usage: python llm_content.py [command]")
            print()
            print("Commands:")
            print("  (no args)         Interactive menu — pick from 15 methods")
            print("  menu              Same as above")
            print("  single            Quality post generator (interactive)")
            print("  quality [topic]   Quality post (auto-pick topic if none)")
            print("  topics            List unused niche_hunter topics")
            print("  publish [file]    Publish saved post to TikTok")
            print("  genimage [file]   Generate AI image for post")
            print("  test              Test all LLM providers")
    else:
        # No args → show interactive menu
        interactive_menu()
