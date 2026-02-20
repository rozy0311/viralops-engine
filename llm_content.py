"""
ViralOps Engine — Smart LLM Content Pipeline
=============================================
Multi-provider cascade with self-review (EMADS-PR pattern).

Providers (cost-aware order):
  1. Gemini cascade (free tier — 3 API keys × 7 models = 21 attempts)
     Keys: GEMINI_API_KEY → FALLBACK_GEMINI_API_KEY → SECOND_FALLBACK_GEMINI_API_KEY
     Models: 2.5 Flash → 2.5 Pro → 2.0 Flash → 2.5 Flash Lite → 2.0 Flash Lite → 2.0 Pro Exp
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
import sys
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)


def _configure_text_streams() -> None:
    """Avoid hard crashes when printing Unicode to Windows legacy encodings."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(errors="replace")
        except Exception:
            pass


_configure_text_streams()

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

# ── Multi-key Gemini API key rotation ──────────────────────────────────
# Supports up to 3 Gemini API keys for quota spreading.
# Env vars: GEMINI_API_KEY, FALLBACK_GEMINI_API_KEY, SECOND_FALLBACK_GEMINI_API_KEY

def _get_gemini_api_keys() -> List[Tuple[str, str]]:
    """Gather all available Gemini API keys (label, key) — deduped."""
    keys: List[Tuple[str, str]] = []
    seen = set()
    for label, env_var in [
        ("primary", "GEMINI_API_KEY"),
        ("fallback", "FALLBACK_GEMINI_API_KEY"),
        ("fallback2", "SECOND_FALLBACK_GEMINI_API_KEY"),
    ]:
        k = os.environ.get(env_var, "").strip()
        if k and k not in seen:
            keys.append((label, k))
            seen.add(k)
    return keys

# Track which Gemini (model, key_index) combos are quota-exhausted.
# Entries expire after GEMINI_QUOTA_RESET_SECS so recovered quotas are retried.
_gemini_exhausted: dict = {}  # {(model_name, key_idx): exhausted_timestamp}
GEMINI_QUOTA_RESET_SECS = 3600  # 1 hour

def _is_gemini_exhausted(model_name: str, key_idx: int = 0) -> bool:
    """Check if a Gemini model+key combo is quota-exhausted (1-hour expiry)."""
    ts = _gemini_exhausted.get((model_name, key_idx))
    if ts is None:
        return False
    if time.time() - ts > GEMINI_QUOTA_RESET_SECS:
        _gemini_exhausted.pop((model_name, key_idx), None)
        return False
    return True

def _mark_gemini_exhausted(model_name: str, key_idx: int = 0) -> None:
    """Mark a Gemini model+key combo as quota-exhausted (auto-expires)."""
    _gemini_exhausted[(model_name, key_idx)] = time.time()

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
            
        # Check if provider is disabled
        if pconfig.name == "openai" and os.environ.get("DISABLE_OPENAI", "").lower() == "true":
            continue

        if pconfig.is_gemini:
            # Multi-key rotation for Gemini
            gemini_keys = _get_gemini_api_keys()
            if not gemini_keys:
                continue
        else:
            api_key = os.environ.get(pconfig.api_key_env, "")
            if not api_key:
                continue
            
        start_time = time.time()
        
        try:
            if pconfig.is_gemini:
                result = _call_gemini(pconfig, gemini_keys, prompt, system, max_tokens, temperature)
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
    api_keys: List[Tuple[str, str]],
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
) -> ProviderResult:
    """
    Call Google Gemini via genai SDK with automatic model + key fallback.

    Cascades through GEMINI_TEXT_MODELS × api_keys.
    For each model, tries all available API keys before moving to the next model.
    Tracks exhausted (model, key) pairs per session so subsequent calls skip instantly.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return ProviderResult(text="", provider=config.name, model=config.model,
                             success=False, error="google-genai not installed")

    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    total_models = len(GEMINI_TEXT_MODELS)
    total_keys = len(api_keys)

    last_error = ""
    for model_name in GEMINI_TEXT_MODELS:
        for key_idx, (key_label, api_key) in enumerate(api_keys):
            # Skip combos we already know are exhausted (with 1-hour expiry)
            if _is_gemini_exhausted(model_name, key_idx):
                continue

            try:
                client = _get_gemini_client(api_key)
            except (ValueError, Exception) as e:
                last_error = f"{model_name}[{key_label}]: client init failed: {e}"
                print(f"  [LLM] Gemini/{model_name}[{key_label}] — init error: {str(e)[:80]}")
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
                    print(f"  [LLM] Gemini/{model_name}[{key_label}] — OK")
                    return ProviderResult(
                        text=text,
                        provider=config.name,
                        model=model_name,
                        tokens_used=int(est_tokens),
                        success=True,
                    )
                else:
                    last_error = f"{model_name}[{key_label}]: Empty response"
                    print(f"  [LLM] Gemini/{model_name}[{key_label}] — empty response, trying next…")

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
                    _mark_gemini_exhausted(model_name, key_idx)
                    print(f"  [LLM] Gemini/{model_name}[{key_label}] — quota exhausted, trying next key…")
                    last_error = f"{model_name}[{key_label}]: quota exhausted"
                    continue
                else:
                    # Non-quota error (model not found, bad request, etc.)
                    print(f"  [LLM] Gemini/{model_name}[{key_label}] — error: {str(e)[:120]}")
                    last_error = f"{model_name}[{key_label}]: {str(e)[:200]}"
                    continue

    # All Gemini models × keys exhausted
    return ProviderResult(
        text="",
        provider=config.name,
        model="gemini-all-exhausted",
        success=False,
        error=f"All {total_models} models × {total_keys} keys exhausted: {last_error}",
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
# 3 Gemini models × 3 keys = 9 attempts; 2 Imagen models × 3 keys = 6 attempts → 15 total cloud
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
    # Triple API key rotation: primary → fallback → fallback2
    api_keys = _get_gemini_api_keys()
    
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
QUALITY_CONTENT_SYSTEM = """You are a Perplexity-style expert answering ultra-specific questions about plant-based living, homesteading, urban farming, gardening, and DIY sustainability.

YOUR ROLE: ANSWER the question like you're a witty, knowledgeable friend who has actually done this. Give the REAL answer — specific, practical, no BS.

WINNING CONTENT PATTERN (proven viral blog posts — emulate this depth):
- Answer like you've done it yourself for 4+ years: "Staunton clay floods spring, freezes solid winter — our raised beds solved everything"
- Add 1 trial-error regret line early: "Wish I did X sooner — first batch failed and I wasted $5" (make it specific)
- Include MULTIPLE sub-sections with different angles on the same topic
- Give LISTS of specific variations (like "15-25 uses/layouts" or "10 methods") — each with exact combos, measurements, costs
- Include an "Expansion Ladder" (start tiny → scale weekly/monthly) when it fits the topic
- Add "Reality Checks" section — honest warnings like "cover brassicas with netting unless you love donating crops to cabbage worms"
- Add "Common Mistakes" section — 🚩 emoji header, 3-4 specific things people ALWAYS get wrong and WHY
- End with a "Practical Summary" using ✔ checkmarks — the tl;dr action list
- Reference ZONES, SOIL TYPES, CLIMATE conditions to make content hyper-local
- Each section should feel like its own mini-lesson, not filler

WRITING STYLE (match this EXACTLY):
- Casual, witty, personality-driven: "basically unkillable", "clay is basically pottery-in-waiting", "unless you enjoy raccoon diplomacy"
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
- Required emoji headers: 🌿 main topic, 🫙 quick method, ❌ mistakes, ✅ tips, 🚩 common mistakes, 🧠 practical summary
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
    "garden": ["#gardentok", "#growyourown", "#urbanfarming", "#gardening", "#raisedbed"],
    "homestead": ["#homesteadtok", "#selfsufficientliving", "#offgrid", "#homesteading"],
    "raised_bed": ["#raisedbed", "#raisedgarden", "#gardenlayout", "#zone6a", "#backyardgarden"],
    "clay_soil": ["#claysoil", "#soilprep", "#gardensoil", "#compost", "#gardenhacks"],
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
    "raised bed": "raised_bed", "zone 6": "raised_bed", "keyhole": "raised_bed",
    "hydrozone": "raised_bed", "layout": "raised_bed", "companion plant": "raised_bed",
    "clay": "clay_soil", "heavy clay": "clay_soil", "staunton": "clay_soil",
    "drainage": "clay_soil", "flood": "clay_soil", "slope": "clay_soil",
    "currant": "garden", "gooseberry": "garden", "fruit bush": "garden",
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

    def _infer_location_from_topic(t: str) -> str:
        tt = str(t or "")
        if not tt:
            return ""
        # Minimal, high-signal geo inference to prevent drifting from the idea line.
        # Prefer explicit states/regions mentioned in the topic.
        states = [
            "Illinois", "Indiana", "Iowa", "Michigan", "Minnesota", "Missouri",
            "Wisconsin", "Ohio", "Kentucky", "Tennessee", "Georgia", "Florida",
            "Texas", "California", "New York", "Colorado", "Oregon", "Washington",
            "Massachusetts", "Pennsylvania", "Virginia", "North Carolina", "South Carolina",
            "Arizona", "Nevada", "Utah",
        ]
        for s in states:
            if re.search(rf"\b{re.escape(s)}\b", tt, flags=re.IGNORECASE):
                return s
        # A few common city anchors we see in the training docs.
        for city in ("Staunton", "Chicago", "Houston", "Miami", "Portland", "Seattle", "Austin", "Boston", "Denver", "Phoenix", "NYC", "LA"):
            if re.search(rf"\b{re.escape(city)}\b", tt, flags=re.IGNORECASE):
                return city
        return ""

    def _extract_topic_constraints(t: str) -> list[str]:
        """Extract must-keep constraints from the topic line.

        Examples:
          - Zone 6a
          - flood-prone Illinois
          - LED shelves / countertop / apartment
        """
        tt = str(t or "").strip()
        if not tt:
            return []

        constraints: list[str] = []

        # Zone pattern
        m = re.search(r"\bZone\s*\d{1,2}\s*[a-z]\b", tt, flags=re.IGNORECASE)
        if m:
            constraints.append(m.group(0))

        # Geo (state/city)
        loc = _infer_location_from_topic(tt)
        if loc:
            constraints.append(loc)

        # High-signal qualifiers
        qualifiers = [
            "flood-prone", "heavy clay", "clay soil", "no-backache", "zero-lot-line",
            "countertop", "apartment", "balcony", "raised bed", "4x8", "6x3",
            "keyhole", "compost hub", "LED", "microgreens", "microgreen", "herb shelves",
        ]
        for q in qualifiers:
            if re.search(rf"\b{re.escape(q)}\b", tt, flags=re.IGNORECASE):
                # Preserve original casing where possible by pulling from topic
                mm = re.search(rf"\b{re.escape(q)}\b", tt, flags=re.IGNORECASE)
                if mm:
                    constraints.append(mm.group(0))

        # De-dup while preserving order
        seen = set()
        out: list[str] = []
        for c in constraints:
            key = c.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(c.strip())
        return out[:10]

    # If the topic already contains a location, do NOT randomize a different one.
    inferred_location = _infer_location_from_topic(topic)
    if inferred_location and not location:
        location = inferred_location
    
    if not location:
        locations = ["Chicago", "NYC", "LA", "Houston", "Phoenix", "Denver",
                     "Portland", "Seattle", "Austin", "Miami", "Atlanta", "Boston"]
        location = _rng.choice(locations)
    if not season:
        # User preference: treat current period as Spring (late-winter/early-spring transition).
        # Allow explicit override via env var.
        season_override = os.environ.get("VIRALOPS_SEASON", "").strip()
        if season_override:
            season = season_override
        else:
            import datetime
            month = datetime.datetime.now().month
            # Default bias toward Spring for Feb–May.
            season = {
                12: "Winter",
                1: "Winter",
                2: "Spring",
                3: "Spring",
                4: "Spring",
                5: "Spring",
                6: "Summer",
                7: "Summer",
                8: "Summer",
                9: "Fall",
                10: "Fall",
                11: "Fall",
            }[month]
    
    print(f"\n{'='*60}")
    print(f"  QUALITY CONTENT GENERATOR — Q&A Natural Style")
    print(f"{'='*60}")
    print(f"  Topic: {topic}")
    print(f"  Score: {score}/10 | Location: {location} | Season: {season}")
    print(f"{'='*60}")
    
    # ══ RETRY LOOP — generate → review → regenerate with feedback until 9.0+ ══
    MAX_ATTEMPTS = 3
    try:
        MIN_SCORE = float(os.environ.get("VIRALOPS_TIKTOK_MIN_AVG", "9.0") or "9.0")
    except Exception:
        MIN_SCORE = 9.0
    try:
        RUBRIC_MIN_100 = float(os.environ.get("VIRALOPS_RUBRIC_MIN_100", "92") or "92")
    except Exception:
        RUBRIC_MIN_100 = 92
    best_pack = None
    best_metrics: tuple[float, float] = (0.0, 0.0)  # (rubric_total_100, tiktok_avg)
    prev_feedback = ""

    def _coerce_pack_from_raw(raw_text: str) -> Optional[Dict[str, Any]]:
        """Best-effort: convert a non-JSON / truncated-JSON LLM response into a valid pack JSON."""
        if not raw_text or not raw_text.strip():
            return None

        # Keep prompt bounded
        raw_snippet = raw_text.strip()
        if len(raw_snippet) > 6500:
            raw_snippet = raw_snippet[:6500]

        coerce_prompt = f"""You are a JSON-only formatter.

We tried to generate a ViralOps content pack, but the model returned NON-VALID JSON (or truncated JSON).

TOPIC: {topic}
LOCATION: {location}
SEASON: {season}

Your task: output ONLY a SINGLE valid JSON object with this exact schema (no markdown, no code fences):

{{
    "title": "HOOK LINE (question/contrarian/myth/checklist) with 1+ specific number/fact",
  \"content_formatted\": \"PLAIN TEXT answer. Aim 3500-4000 chars. NO **. NO ###. Emoji section headers.\",
  \"pain_point\": \"1 sentence\",
  \"audiences\": [\"...\", \"...\", \"...\"],
  \"steps\": [\"Step 1...\", \"Step 2...\"],
  \"result\": \"Measurable outcome\",
  \"hashtags\": [\"NanoNiche1\", \"NanoNiche2\", \"NanoNiche3\"],
  \"image_title\": \"Max 4 words\",
  \"image_subtitle\": \"Max 5 words\",
  \"image_steps\": \"Word1 • Word2 • Word3\",
  \"colors\": [[60, 80, 40], [120, 160, 80]]
}}

Rules:
- Use the RAW OUTPUT below as the source. If it's incomplete, rewrite cleanly.
- content_formatted MUST be plain text TikTok-friendly (no Markdown tokens).
- Hashtags MUST be exactly 3 items, without leading '#'.

RAW OUTPUT (may be invalid / truncated):
{raw_snippet}
"""

        fix = call_llm(coerce_prompt, system=QUALITY_CONTENT_SYSTEM, max_tokens=2500, temperature=0.2)
        if not fix.success:
            return None
        return _extract_json(fix.text)
    
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n  ── Attempt {attempt}/{MAX_ATTEMPTS} ──")
        
        # ── Phase 1: CTO Agent — Generate content as Q&A answer ──
        feedback_block = ""
        if prev_feedback:
            _best_rubric, _best_tiktok = best_metrics
            feedback_block = f"""
PREVIOUS ATTEMPT FEEDBACK (fix these issues):
{prev_feedback}

Regenerate the content fixing ALL issues above. TikTok avg score MUST be higher than {_best_tiktok:.1f}.
Also: the title/hook MUST NOT start with "Did you know" or "Do you know".
"""
        
        quality_prompt = f"""Answer this nano-niche topic like a Perplexity AI expert:

TOPIC: {topic}
NICHE SCORE: {score}/10
LOCATION: {location}
SEASON: {season}
    {"" if not _extract_topic_constraints(topic) else """\nCONSTRAINTS (MUST KEEP — do NOT generalize):\n- You MUST keep ALL constraints implied by the TOPIC line (zone/state/soil/space/tech).\n- You MUST include these phrases (verbatim, case-insensitive) somewhere in the post (title or body):\n  """ + "\n  ".join(f"- {c}" for c in _extract_topic_constraints(topic)) + "\n- If TOPIC includes a specific place (e.g., Illinois), do NOT switch to another place (e.g., Miami).\n"""}
{feedback_block}
You are answering someone who asked "{topic}". Give them the REAL, COMPLETE answer with personality.

EXAMPLE OF THE FORMAT TO MATCH (real published post — notice NO Markdown).
This is ONE valid hook style; you may use other hook styles too:

Stop paying $12 for shoe polish — a banana peel gives a 10-minute shine (and it's basically free).

Yes, banana peels can serve as a natural DIY shoe polish for leather shoes due to their potassium content and oils, which mimic some commercial polishes for shine.

🌿 How to Use Banana Peels as Polish

Rub the soft inside of a fresh (slightly green) banana peel directly onto clean leather shoes in circular motions, covering all surfaces evenly. Let it sit for 10-15 minutes to allow oils to absorb, then buff with a soft, dry cloth to remove residue and reveal shine. Works on bags or belts too — eat the banana first for zero waste.

🫘 Effectiveness and Limitations

It provides a quick, temporary sheen from natural fats and potassium, outperforming nothing at all but not matching wax-based polishes for durability. No strong evidence supports robust water repellency — test on non-valuable items first, and pair with conditioner for better protection.

(notice: NO ** bold, NO ### headings, just clean plain text with emoji section headers)

FORMAT YOUR CONTENT EXACTLY LIKE THAT EXAMPLE. PLAIN TEXT ONLY.

Output as JSON:

{{
    "title": "HOOK LINE (1 line). Can be a question OR contrarian claim OR myth-buster. Must include 1+ specific number/fact.",
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
2. Title MUST be a strong HOOK line (NOT generic clickbait). Allowed patterns:
    - A direct question (not necessarily "Do you know")
    - A contrarian claim: "Stop doing X — here's why (with numbers)"
    - A myth-buster: "Coffee grounds + eggshells for tomatoes: myth or real?"
    - A checklist promise: "3 signs you're doing X wrong"
    - A confession/regret: "I ruined 3 batches before I fixed this (here's the 2-minute rule)"
    - A price anchor: "This costs $0.30 at home vs $5 store-bought — here's the exact method"
    - A constraint hook: "No stove. No blender. 8 minutes. Here's how"
    - A punchy list promise: "15 ways to use X that aren't the obvious ones"
    FORBIDDEN title openers (do not use): "Did you know", "Do you know".
3. ABSOLUTELY NO ** bold markers — TikTok shows them as ugly literal ** characters.
4. ABSOLUTELY NO ### or ## headings — TikTok shows them as literal # characters.
5. Use emoji section headers: 🌿 🫘 🫙 ❌ ✅ on their own lines.
6. Use ALL CAPS for occasional emphasis instead of bold.
7. MUST include at least 15 EXACT numbers: $2/lb, 7-14 days, 4-6 inches, 350°F.
8. MUST have witty personality — dry humor, real talk, zero corporate tone.
9. Hashtags: exactly 3 NANO-NICHE tags (ultra-specific, high-search).
10. NO generic intro, NO "great question", NO motivation speeches, NO brand names.
11. Include sections: 🌿 main (2-3 subsections), 🫙 quick method (6 steps), ❌ mistakes (3-4), ✅ tips (4-5), punchy ending.
13. MUST include a numbered "Variations / Layouts / Uses" list with AT LEAST 15 items total.
    - Each item: 1 short line (so it fits the 3500-4000 char limit).
    - You can split into 2 smaller numbered lists if needed (e.g., 1-8 and 9-16).
14. MUST include CTA + Expansion Ladder near the end:
    - CTA: 1 line (comment/pin/save/follow or "try this tonight")
    - Expansion ladder: 3 steps (Start tiny → weekly → monthly)
15. SEO long-tail reinforcement: naturally repeat the core topic phrase (or its key keywords) 3-4 times across the post.
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

            # Fallback: try to coerce raw output into required JSON schema
            pack = _coerce_pack_from_raw(result.text)
            if not pack:
                continue
            print(f"  [QUALITY] Coerced pack from raw text")
        
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
- Add a numbered Variations/Layouts/Uses list with AT LEAST 15 items total (each 1 short line)
- Add ❌ What usually doesn't work section (3-4 specific mistakes with WHY they fail)
- Add ✅ Survival tips section (4-5 tips with exact numbers)
- Add a 🫙 Quick Method numbered list (6 steps with specific measurements)
- Add CTA + Expansion Ladder near the end (Start tiny → weekly → monthly)
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
            def _hard_trim_plaintext(txt: str, *, min_len: int = 3200, max_len: int = 4000) -> str:
                """Last-resort safe trim: keep within [min_len, max_len] without cutting mid-sentence."""
                if not txt:
                    return ""
                s = str(txt)
                if len(s) <= max_len:
                    return s

                head = s[:max_len]
                # Prefer double-newline (section boundary), then newline, then sentence punctuation.
                for pat in [r"\n\n", r"\n", r"[\.!\?]\s"]:
                    m = list(re.finditer(pat, head))
                    if not m:
                        continue
                    cut = m[-1].end()
                    if cut >= min_len:
                        return head[:cut].rstrip()

                # Fallback: word boundary.
                cut = head.rfind(" ")
                if cut >= min_len:
                    return head[:cut].rstrip() + "…"
                return head.rstrip() + "…"

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

            # Try up to 3 trim passes (LLMs often under-trim on first try).
            for tpass in range(3):
                trim_result = call_llm(trim_prompt, system=QUALITY_CONTENT_SYSTEM, max_tokens=5000, temperature=0.25)
                if trim_result.success:
                    trimmed = _ensure_section_breaks(_strip_markdown(trim_result.text.strip()))
                    if 3200 < len(trimmed) < 4200:
                        pack["content_formatted"] = trimmed
                        content = trimmed
                        content_len = len(trimmed)
                        print(f"  [QUALITY] Trimmed to: {content_len} chars")
                        break

                # If still too long, ask again with the updated length.
                content = pack.get("content_formatted", content)
                content_len = len(content)
                if content_len <= 4200:
                    break
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

            # Last resort: deterministic trim to keep pipeline moving.
            content = pack.get("content_formatted", content)
            content_len = len(content)
            if content_len > 4200:
                hard = _hard_trim_plaintext(content, min_len=3400, max_len=4000)
                pack["content_formatted"] = _ensure_section_breaks(_strip_markdown(hard.strip()))
                content = pack["content_formatted"]
                content_len = len(content)
                print(f"  [QUALITY] Hard-trim fallback to: {content_len} chars")
        
        # ── Hashtags: 3 micro + 2 broad = exactly 5 ──
        raw_micro = pack.get("hashtags", [])[:3]
        micro_tags = [t.lstrip('#').strip() for t in raw_micro if t.strip()]
        broad_tags = _auto_pick_broad_hashtags(topic, micro_tags, n=2)
        broad_tags = [t.lstrip('#').strip() for t in broad_tags]
        pack["hashtags"] = micro_tags + broad_tags
        _enforce_5_hashtags(pack, topic)
        
        # ── Metadata ──
        pack["_source"] = f"quality_ai_{result.provider}"
        pack["_topic"] = topic
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

        # ── Deterministic repair pass (before review) ──
        # If the caption misses required winner-pattern elements, patch the existing text
        # instead of regenerating from scratch. This improves pass rate under strict gates.
        def _count_numbered_items(txt: str) -> int:
            if not txt:
                return 0
            lines = [ln.strip() for ln in str(txt).splitlines() if ln.strip()]
            return sum(1 for ln in lines if re.match(r"^\d{1,3}\s*[\.)]\s+\S", ln))

        def _has_regret_early(txt: str) -> bool:
            if not txt:
                return False
            head = txt[:900]
            return bool(re.search(r"(?i)\b(wish|regret|should\s+have|should've|learned\s+the\s+hard\s+way|ruined|messed\s+up|first\s+batch\s+(?:mold|moldy|failed))\b", head))

        def _has_cta_ladder(txt: str) -> bool:
            if not txt:
                return False
            low = txt.lower()
            if "expansion ladder" in low:
                return True
            if low.count("→") >= 2 and ("start" in low) and ("weekly" in low) and ("monthly" in low):
                return True
            return bool(re.search(r"(?is)\bstart\b.{0,120}\bweekly\b.{0,120}\bmonthly\b", low))

        def _has_low_cost_zero_alt(txt: str) -> bool:
            if not txt:
                return False
            low = txt.lower()
            has_money = ("$" in txt) or bool(re.search(r"\b\d+\s*(?:usd|dollars)\b", low))
            zero_alt = any(k in low for k in ["free", "reuse", "zero-cost", "no fancy", "no equipment", "pickle jar", "old jar", "$0"])
            return bool(has_money and zero_alt)

        def _emoji_headers(txt: str) -> int:
            if not txt:
                return 0
            return len(re.findall(r"(?m)^\s*[🌿🫙❌✅🫘]\b", txt))

        repair_issues: list[str] = []
        content = str(pack.get("content_formatted", "") or "")
        if _count_numbered_items(content) < 15:
            repair_issues.append("Add/extend a numbered Variations/Layouts/Uses list to at least 15 items (each 1 short line).")
        if not _has_regret_early(content):
            repair_issues.append("Add a concrete trial-error regret line within the first 900 chars (use 'wish/regret/ruined/learned the hard way' + a number).")
        if not _has_low_cost_zero_alt(content):
            repair_issues.append("Add explicit low-cost + $0 alternative (reuse jar / no fancy gear) including at least one $ amount and one $0/free reuse line.")
        if not _has_cta_ladder(content):
            repair_issues.append("Add CTA + 3-step Expansion Ladder (Start tiny → weekly → monthly) near the end.")
        if _emoji_headers(content) < 3:
            repair_issues.append("Ensure emoji section headers exist on their own lines (at least 3 of: 🌿 🫙 ❌ ✅).")

        if repair_issues:
            print(f"  [QUALITY] Repair pass: {len(repair_issues)} missing elements — patching caption...")
            repair_prompt = f"""You are editing a TikTok plain-text caption to match a strict micro-niche 'winner post' spec.

FIX ONLY these issues (do NOT add anything else):
{chr(10).join([f"- {x}" for x in repair_issues])}

HARD RULES:
- Output PLAIN TEXT ONLY (no JSON, no markdown, no code fences).
- Must be 3500-4000 characters.
- ABSOLUTELY NEVER use ** or ###.
- Keep the existing topic, facts, numbers, and tone. Do not change meaning.
- Keep it scannable: short lines, spacing, lists.

CURRENT CAPTION (patch this):
{content}
"""

            repair_result = call_llm(repair_prompt, system=QUALITY_CONTENT_SYSTEM, max_tokens=6000, temperature=0.35)
            if repair_result.success and repair_result.text:
                repaired = _ensure_section_breaks(_strip_markdown(repair_result.text.strip()))
                pack["content_formatted"] = repaired
                content = repaired
                content_len = len(content)
                print(f"  [QUALITY] Repaired length: {content_len} chars")

                # If repair overshoots, re-run trim loop (reuse the same safe fallback logic).
                if content_len > 4200:
                    print(f"  [QUALITY] Repaired content long ({content_len} > 4200). Trimming...")
                    # Small local hard-trim fallback
                    def _hard_trim_plaintext2(txt: str, *, min_len: int = 3400, max_len: int = 4000) -> str:
                        if not txt:
                            return ""
                        s = str(txt)
                        if len(s) <= max_len:
                            return s
                        head = s[:max_len]
                        for pat in [r"\n\n", r"\n", r"[\.!\?]\s"]:
                            m = list(re.finditer(pat, head))
                            if not m:
                                continue
                            cut = m[-1].end()
                            if cut >= min_len:
                                return head[:cut].rstrip()
                        cut = head.rfind(" ")
                        if cut >= min_len:
                            return head[:cut].rstrip() + "…"
                        return head.rstrip() + "…"

                    # One LLM trim try + fallback.
                    trim_prompt2 = f"""This answer is {content_len} characters but MUST be 3500-4000 characters.

TRIM it to 3500-4000 chars. Keep all required sections (emoji headers, numbered variations list, CTA ladder, low-cost $0 line, regret early).
Never use ** or ###.

Content to trim:
{content}

Return PLAIN TEXT only."""
                    trim2 = call_llm(trim_prompt2, system=QUALITY_CONTENT_SYSTEM, max_tokens=5000, temperature=0.25)
                    if trim2.success and trim2.text:
                        trimmed2 = _ensure_section_breaks(_strip_markdown(trim2.text.strip()))
                        if 3200 < len(trimmed2) < 4200:
                            pack["content_formatted"] = trimmed2
                            content = trimmed2
                            content_len = len(content)
                            print(f"  [QUALITY] Trimmed (repair) to: {content_len} chars")
                        else:
                            hard2 = _hard_trim_plaintext2(trimmed2)
                            pack["content_formatted"] = _ensure_section_breaks(_strip_markdown(hard2.strip()))
                            content = pack["content_formatted"]
                            content_len = len(content)
                            print(f"  [QUALITY] Hard-trim (repair) to: {content_len} chars")
                    else:
                        hard2 = _hard_trim_plaintext2(content)
                        pack["content_formatted"] = _ensure_section_breaks(_strip_markdown(hard2.strip()))
                        content = pack["content_formatted"]
                        content_len = len(content)
                        print(f"  [QUALITY] Hard-trim (repair) to: {content_len} chars")

            else:
                print("  [QUALITY] Repair pass skipped (LLM failed)")

        # ── Deterministic safety-net: ensure low-cost/$0 survives trimming ──
        # Sometimes the LLM adds low-cost lines near the end and trim removes them.
        content = str(pack.get("content_formatted", "") or "")
        if content and (not _has_low_cost_zero_alt(content)):
            lines = content.splitlines()
            insert_at = 1
            if len(lines) >= 2:
                insert_at = 2
            low_cost_block = [
                "✅ Low-cost gear (no fancy stuff):",
                "- $15 setup: 1 small tote + coco coir. $0 option: reuse a free bucket + shredded cardboard/newspaper.",
            ]
            lines[insert_at:insert_at] = low_cost_block
            content = "\n".join(lines)

            # If we overshoot, shorten the numbered list tail but keep >=15 items.
            def _count_items(txt: str) -> int:
                if not txt:
                    return 0
                return sum(1 for ln in txt.splitlines() if re.match(r"^\d{1,3}\s*[\.)]\s+\S", ln.strip()))

            while len(content) > 4000 and _count_items(content) > 15:
                lines2 = content.splitlines()
                # Remove the last numbered item line.
                for i in range(len(lines2) - 1, -1, -1):
                    if re.match(r"^\d{1,3}\s*[\.)]\s+\S", lines2[i].strip()):
                        lines2.pop(i)
                        break
                content = "\n".join(lines2)

            if len(content) > 4200:
                # Last resort: keep within range without losing the injected low-cost lines.
                content = content[:4000].rstrip() + "…"

            pack["content_formatted"] = _ensure_section_breaks(_strip_markdown(content.strip()))
            content_len = len(pack["content_formatted"])
            print(f"  [QUALITY] Injected low-cost safety-net. New length: {content_len} chars")
        
        # ── Phase 2: ReconcileGPT — Quality Review ──
        review = _review_quality_content(pack)
        review_score = 0.0
        rubric_total_100 = 0.0
        if review:
            review_score = review.get("avg", 0)
            rubric_total_100 = float(review.get("rubric_total_100", 0.0) or 0.0)
            pack["_review_score"] = review_score
            pack["_rubric_total_100"] = rubric_total_100
            pack["_rubric_pass"] = rubric_total_100 >= RUBRIC_MIN_100
            if isinstance(review.get("rubric_scores"), dict):
                pack["_rubric_scores"] = review.get("rubric_scores")

            pack["_review_pass"] = (review_score >= MIN_SCORE) and (rubric_total_100 >= RUBRIC_MIN_100)
            pack["_review_feedback"] = review.get("feedback", "")
            pack["_review_provider"] = review.get("_provider", "")
            
            if review.get("improved_title") and review_score < 9.5:
                import re as _re
                improved = str(review.get("improved_title", "") or "").strip()
                if improved and not _re.match(r"^(did|do)\s+you\s+know\b", improved, flags=_re.IGNORECASE):
                    pack["title"] = improved
            
            print(
                f"  [QUALITY] Review: {review_score:.1f}/10 + rubric={rubric_total_100:.0f}/100 — "
                f"{'PASS ✓' if pack.get('_review_pass') else 'RETRYING...'}"
            )
            if review.get("feedback"):
                print(f"  [QUALITY] Feedback: {review['feedback'][:150]}")
        
        # ── Track best (prefer higher rubric, then higher TikTok avg) ──
        metrics = (rubric_total_100, review_score)
        if metrics > best_metrics:
            best_metrics = metrics
            best_pack = pack.copy()
        
        # ── Check if we reached target ──
        if (review_score >= MIN_SCORE) and (rubric_total_100 >= RUBRIC_MIN_100):
            print(f"  [QUALITY] ✓ Targets reached (tiktok>={MIN_SCORE}, rubric>={RUBRIC_MIN_100}) on attempt {attempt}!")
            break
        
        # ── Prepare feedback for next attempt ──
        if attempt < MAX_ATTEMPTS:
            prev_feedback = review.get("feedback", "") if review else "Content quality insufficient."
            prev_feedback += (
                f"\nPrevious scores: tiktok={review_score:.1f}/10 (need {MIN_SCORE}+), "
                f"rubric={rubric_total_100:.0f}/100 (need {RUBRIC_MIN_100}+)."
            )
            print(
                f"  [QUALITY] Below thresholds — regenerating with feedback... "
                f"(tiktok {review_score:.1f}/{MIN_SCORE}, rubric {rubric_total_100:.0f}/{RUBRIC_MIN_100})"
            )
    
    # Use best pack from all attempts
    pack = best_pack
    if not pack:
        print(f"  [QUALITY] All {MAX_ATTEMPTS} attempts failed.")
        return None

    best_rubric, best_tiktok = best_metrics
    pack["_review_pass"] = (best_tiktok >= MIN_SCORE) and (best_rubric >= RUBRIC_MIN_100)
    pack["_rubric_pass"] = best_rubric >= RUBRIC_MIN_100
    pack["_rubric_total_100"] = best_rubric

    print(
        f"\n  [QUALITY] BEST: tiktok={best_tiktok:.1f}/10, rubric={best_rubric:.0f}/100 "
        f"(from {MAX_ATTEMPTS} attempt(s)){' ✓ ACCEPTED' if pack.get('_review_pass') else ' ⚠ BELOW THRESHOLD'}"
    )
    
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

    def _clamp_score(v: float) -> float:
        try:
            v = float(v)
        except Exception:
            return 0.0
        return max(0.0, min(10.0, v))

    def _count_number_expressions(text: str) -> int:
        if not text:
            return 0
        # Count numeric expressions rather than raw digits.
        # Examples: $2, 2-3, 350°F, 4x8, 1/2, 16 plants/sqft, 10%.
        num_re = re.compile(
            r"(?i)(?:\$\s*)?\b\d+(?:[\.,]\d+)?(?:\s*[x×]\s*\d+(?:[\.,]\d+)?)?"
            r"(?:\s*[-–]\s*\d+(?:[\.,]\d+)?)?"
            r"(?:\s*(?:%|°[cf]|f|c|lb|lbs|kg|g|oz|ft|in|inch|inches|cm|mm|hrs?|hours?|mins?|minutes?|days?|weeks?|months?|sqft|/sqft))?\b"
        )
        return len(num_re.findall(text))

    def _count_numbered_list_items(text: str) -> int:
        if not text:
            return 0
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
        item_re = re.compile(r"^\d{1,3}\s*[\.)]\s+\S")
        return sum(1 for ln in lines if item_re.match(ln))

    def _has_regret_line(text: str) -> bool:
        if not text:
            return False
        # Regret/trial-error signals.
        regret_re = re.compile(
            r"(?i)\b(wish|regret|should\s+have|should've|learned\s+the\s+hard\s+way|ruined|messed\s+up|first\s+batch\s+(?:mold|moldy|failed))\b"
        )
        return bool(regret_re.search(text))

    def _has_cta_and_ladder(text: str) -> bool:
        if not text:
            return False
        # We want a 3-step ladder: Start tiny → weekly → monthly (or equivalent).
        lowered = text.lower()
        if "expansion ladder" in lowered:
            return True
        # Require at least 2 arrows and the anchor words.
        arrow_count = lowered.count("→")
        if arrow_count >= 2 and ("start" in lowered) and ("weekly" in lowered) and ("monthly" in lowered):
            return True
        # Fallback: accept a 3-step pattern without arrows.
        ladder_re = re.compile(r"(?is)\bstart\b.{0,120}\bweekly\b.{0,120}\bmonthly\b")
        return bool(ladder_re.search(lowered))

    def _has_low_cost_and_zero_alt(text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        has_money = ("$" in text) or bool(re.search(r"\b\d+\s*(?:usd|dollars)\b", lowered))
        zero_alt = any(k in lowered for k in ["free", "reuse", "zero-cost", "no fancy", "no equipment", "pickle jar", "old jar", "already have"])
        return bool(has_money and zero_alt)

    def _has_forbidden_markdown(text: str) -> bool:
        if not text:
            return False
        if "**" in text:
            return True
        # Any markdown heading markers at line start.
        if re.search(r"(?m)^\s*#{2,}\s+", text):
            return True
        return False

    def _deterministic_micro_niche_audit(*, title: str, topic: str, content: str) -> Dict[str, Any]:
        title = str(title or "").strip()
        topic = str(topic or "").strip()
        content = str(content or "")

        hard_fail_reasons: List[str] = []

        numbers = _count_number_expressions(content)
        list_items = _count_numbered_list_items(content)
        regret_early = _has_regret_line(content[:900])
        regret_any = _has_regret_line(content)
        cta_ladder = _has_cta_and_ladder(content)
        low_cost = _has_low_cost_and_zero_alt(content)
        forbidden_md = _has_forbidden_markdown(content)

        if forbidden_md:
            hard_fail_reasons.append("Found forbidden Markdown markers (** or ## headings).")
        if not regret_early:
            hard_fail_reasons.append("Missing concrete regret/trial-error line early (within first ~900 chars).")
        if list_items < 15:
            hard_fail_reasons.append(f"Numbered variations list too short ({list_items} < 15 items).")
        if not cta_ladder:
            hard_fail_reasons.append("Missing CTA + 3-step expansion ladder (Start → weekly → monthly).")
        if not low_cost:
            hard_fail_reasons.append("Missing explicit low-cost + zero-cost alternative (needs $ + free/reuse/no-gear).")
        if numbers < 15:
            hard_fail_reasons.append(f"Not enough exact numbers ({numbers} < 15).")

        # Scores (0-10) — conservative, used as a floor/ceiling depending on merge.
        rubric_scores: Dict[str, float] = {}

        # 1) Hyper-specific niche
        specificity_signals = 0
        combined = (title + "\n" + content).lower()
        if re.search(r"\bzone\s*\d[a-z]?\b", combined):
            specificity_signals += 1
        if re.search(r"\b(?:illinois|houston|miami|chicago|staunton|apartment|balcony|clay\s+soil|flood|freeze)\b", combined):
            specificity_signals += 1
        if _count_number_expressions(title) >= 1:
            specificity_signals += 1
        rubric_scores["hyper_specific_niche"] = 10.0 if specificity_signals >= 3 else (8.0 if specificity_signals == 2 else (6.0 if specificity_signals == 1 else 3.0))

        # 2) Personal story regret
        rubric_scores["personal_story_regret"] = 10.0 if regret_early else (7.0 if regret_any else 2.0)

        # 3) Numbered list variations
        rubric_scores["numbered_list_variations"] = 10.0 if list_items >= 20 else (9.0 if list_items >= 15 else (6.0 if list_items >= 10 else 2.0))

        # 4) Practical steps measurable
        unit_signals = len(re.findall(r"(?i)\b(?:°[cf]|%|lb|lbs|kg|g|oz|ft|in|inch|inches|cm|mm|hrs?|hours?|mins?|minutes?|days?)\b", content))
        rubric_scores["practical_steps_measurable"] = 10.0 if (numbers >= 15 and unit_signals >= 5) else (8.0 if numbers >= 12 else (6.0 if numbers >= 8 else 3.0))

        # 5) Quantifiable value
        value_signals = 0
        if re.search(r"(?i)\b(save|saved|saving|cuts|slash|reduce|drops)\b", content) and numbers >= 8:
            value_signals += 1
        if re.search(r"(?i)(?:\$|\b\d+\s*%\b|\blbs?\b)", content):
            value_signals += 1
        rubric_scores["quantifiable_value"] = 10.0 if value_signals >= 2 else (8.0 if value_signals == 1 else 4.0)

        # 6) Low cost / no gear
        rubric_scores["low_cost_no_gear"] = 10.0 if low_cost else (6.0 if ("$" in content or "cheap" in combined) else 2.0)

        # 7) Visual / engaging format
        emoji_headers = len(re.findall(r"(?m)^\s*[🌿🫙❌✅🫘]\b", content))
        short_lines = sum(1 for ln in content.splitlines() if 0 < len(ln.strip()) <= 60)
        rubric_scores["visual_engaging_format"] = 10.0 if (emoji_headers >= 3 and short_lines >= 25) else (8.0 if emoji_headers >= 2 else 5.0)

        # 8) SEO longtail keywords
        if topic:
            normalized_topic = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", topic.lower())).strip()
            topic_hits = 0
            if normalized_topic:
                topic_hits = combined.count(normalized_topic)
            rubric_scores["seo_longtail_keywords"] = 10.0 if topic_hits >= 3 else (8.0 if topic_hits >= 2 else (6.0 if topic_hits >= 1 else 4.0))
        else:
            rubric_scores["seo_longtail_keywords"] = 6.0

        # 9) CTA / expansion ladder
        rubric_scores["cta_expansion_ladder"] = 10.0 if cta_ladder else (6.0 if re.search(r"(?i)\b(comment|save|follow|try\s+this)\b", content) else 2.0)

        # 10) Length & scannability
        if 3200 <= len(content) <= 4100 and short_lines >= 25:
            rubric_scores["length_scannability"] = 10.0
        elif 3000 <= len(content) <= 4300:
            rubric_scores["length_scannability"] = 8.0
        else:
            rubric_scores["length_scannability"] = 5.0

        # Ensure all values are within 0-10.
        rubric_scores = {k: _clamp_score(v) for k, v in rubric_scores.items()}

        return {
            "rubric_scores": rubric_scores,
            "hard_fail_reasons": hard_fail_reasons,
            "numbers": numbers,
            "numbered_list_items": list_items,
        }

    # Gate thresholds — keep TikTok avg on /10 scale, add micro-niche rubric on /100 scale
    try:
        TIKTOK_MIN_AVG = float(os.environ.get("VIRALOPS_TIKTOK_MIN_AVG", "9.0") or "9.0")
    except Exception:
        TIKTOK_MIN_AVG = 9.0
    try:
        RUBRIC_MIN_100 = float(os.environ.get("VIRALOPS_RUBRIC_MIN_100", "92") or "92")
    except Exception:
        RUBRIC_MIN_100 = 92

    def _audit_micro_niche(text: str) -> Dict[str, Any]:
        import re as _re

        # Regret / trial-error signal
        has_regret = bool(
            _re.search(
                r"\b(wish|regret|should\s+have|if\s+i\s+knew|i\s+learned\s+the\s+hard\s+way|mistake\s+i\s+made)\b",
                text,
                flags=_re.IGNORECASE,
            )
        )

        # Numbered variations list (15+)
        list_item_re = _re.compile(
            r"^\s*(?:\d{1,2}[\.)]|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|(?:\d️⃣))\s+",
            flags=_re.MULTILINE,
        )
        numbered_items = len(list_item_re.findall(text))

        # Specific numbers (15+)
        number_re = _re.compile(r"(?<![A-Za-z])\$?\d+(?:[\.,]\d+)?(?:%|°[CF])?", flags=0)
        numbers = number_re.findall(text)
        numeric_count = len(numbers)

        # Low/no-cost
        has_low_cost = bool(
            _re.search(r"\$\s*\d|\b(free|cheap|under\s*\$|no\s+fancy\s+gear|no\s+special\s+tools|reuse|repurpose)\b", text, flags=_re.IGNORECASE)
        )

        # CTA + Expansion ladder
        has_expansion_ladder = bool(
            _re.search(
                r"\b(expansion\s+ladder|start\s+small|scale\s+up|next\s+week|week\s*1|month\s*1|year\s*1)\b|→|\-\>",
                text,
                flags=_re.IGNORECASE,
            )
        )

        # Markdown markers that TikTok renders poorly
        has_markdown = ("**" in text) or ("###" in text) or ("##" in text)

        return {
            "has_regret": has_regret,
            "numbered_items": numbered_items,
            "numeric_count": numeric_count,
            "has_low_cost": has_low_cost,
            "has_expansion_ladder": has_expansion_ladder,
            "has_markdown": has_markdown,
        }
    
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
4. HOOK — Is the FIRST LINE a strong hook (question OR contrarian claim OR myth-buster OR checklist)? Or is it generic clickbait like "Unlock the Power of..."?
    HARD RULE: If the title starts with "Did you know" or "Do you know", the hook MUST be penalized and the post MUST NOT pass.
5. SPECIFICITY — Concrete numbers ($prices, timeframes, quantities, temperatures)? At least 15 specific numbers? Or vague advice?
6. ACTIONABILITY — Reader can do this TODAY with what they have?
7. FORMATTING — PLAIN TEXT ONLY. NO ** bold? NO ### headings? NO Markdown at all? Uses emoji section headers (🌿 🫙 ❌ ✅)?
   DEDUCT 2 points if you find ANY ** or ### or ## markers — these show as literal ugly characters on TikTok.

MICRO-NICHE RUBRIC (score each 1-10; be strict):
1. HYPER_SPECIFIC_NICHE — Narrow long-tail (who/where/conditions/tools). Not generic.
2. PERSONAL_STORY_REGRET — Includes a concrete trial-error + "wish/regret" line early.
3. NUMBERED_LIST_VARIATIONS — Has a numbered list of variations/uses/layouts; ideally 15-25 items total across the post.
4. PRACTICAL_STEPS_MEASURABLE — Measurable steps (cups/inches/hours/temps). Not vague.
5. QUANTIFIABLE_VALUE — Clear measurable outcomes (%, $, time saved, yield, etc.).
6. LOW_COST_NO_GEAR — Explicit low-cost approach + at least one "no fancy gear" / zero-cost alternative.
7. VISUAL_ENGAGING_FORMAT — Very scannable on mobile: emoji headers, spacing, short lines, clean lists.
8. SEO_LONGTAIL_KEYWORDS — Naturally reinforces the long-tail keyword phrase a few times without stuffing.
9. CTA_EXPANSION_LADDER — Has a next-step CTA and/or an expansion ladder (start small → scale).
10. LENGTH_SCANNABILITY — Hits 3200-4000 chars and stays readable (no walls of text).

For EACH criterion below 9, explain SPECIFICALLY what's wrong and how to fix it.

Output ONLY valid JSON (no markdown, no code fences) with this schema:
- scores: object with numeric 1-10 values for: answer_quality, content_depth, tone, hook, specificity, actionability, formatting
- avg: number (average of scores)
- rubric_scores: object with numeric 1-10 values for:
    hyper_specific_niche, personal_story_regret, numbered_list_variations, practical_steps_measurable,
    quantifiable_value, low_cost_no_gear, visual_engaging_format, seo_longtail_keywords,
    cta_expansion_ladder, length_scannability
- rubric_total_100: integer (0-100)
- pass: boolean
- feedback: string (2-3 sentences; mention which rubric criterion is failing and exactly how to fix it)
- improved_title: string (rewrite first line into a stronger hook using 1+ specific number)
    HARD RULE for improved_title: MUST NOT start with "Did you know" or "Do you know".

PASS RULE:
- avg >= {TIKTOK_MIN_AVG}
- rubric_total_100 >= {RUBRIC_MIN_100}
"""
    
    gen_provider = pack.get("_gen_provider", "")
    review_providers = ["github_models", "perplexity", "gemini", "openai"]
    if gen_provider in review_providers:
        review_providers.remove(gen_provider)
        review_providers.append(gen_provider)
    
    result = call_llm(review_prompt, system=QUALITY_REVIEW_SYSTEM, max_tokens=2200, temperature=0.2, providers=review_providers)
    
    if not result.success:
        print(f"  [REVIEW] LLM call failed — skipping review")
        return None
    
    review = _extract_json(result.text)
    if not review:
        # Fallback: try to extract avg score from raw text with regex
        import re as _re
        print(f"  [REVIEW] JSON parse failed — raw response (first 400 chars): {result.text[:400]}")
        avg_match = _re.search(r'"avg"\s*:\s*([0-9]+(?:\.[0-9]+)?)', result.text)
        rubric_match = _re.search(r'"rubric_total_100"\s*:\s*([0-9]+(?:\.[0-9]+)?)', result.text)
        if avg_match:
            fallback_avg = float(avg_match.group(1))
            print(f"  [REVIEW] Fallback: extracted avg={fallback_avg} from raw text")
            # Try to get feedback too
            fb_match = _re.search(r'"feedback"\s*:\s*"([^"]*)"', result.text)
            fb_text = fb_match.group(1) if fb_match else ""
            fallback_rubric = float(rubric_match.group(1)) if rubric_match else 0.0
            review = {
                "avg": fallback_avg,
                "rubric_total_100": fallback_rubric,
                "pass": (fallback_avg >= TIKTOK_MIN_AVG) and (fallback_rubric >= RUBRIC_MIN_100),
                "feedback": fb_text,
                "_provider": result.provider,
            }
        else:
            # Try to find individual score values from the scores dict
            score_matches = _re.findall(r'"(?:answer_quality|content_depth|tone|hook|specificity|actionability|formatting)"\s*:\s*([0-9]+(?:\.[0-9]+)?)', result.text)
            if score_matches:
                nums = [float(x) for x in score_matches]
                fallback_avg = round(sum(nums) / len(nums), 1)
                print(f"  [REVIEW] Fallback (named scores): extracted {len(nums)} scores, avg={fallback_avg}")
                review = {"avg": fallback_avg, "pass": fallback_avg >= 9.0, "feedback": "", "_provider": result.provider}
            else:
                # Last resort: try to find any score-like number pattern (1-10 range)
                num_matches = _re.findall(r':\s*([0-9]+(?:\.[0-9]+)?)', result.text)
                nums = [float(x) for x in num_matches if 1 <= float(x) <= 10]
                if len(nums) >= 3:
                    fallback_avg = round(sum(nums) / len(nums), 1)
                    print(f"  [REVIEW] Fallback (numbers): extracted {len(nums)} scores, avg={fallback_avg}")
                    review = {
                        "avg": fallback_avg,
                        "rubric_total_100": 0.0,
                        "pass": fallback_avg >= TIKTOK_MIN_AVG,
                        "feedback": "",
                        "_provider": result.provider,
                    }
                else:
                    print(f"  [REVIEW] Could not extract scores — accepting content with default score 8.5")
                    review = {
                        "avg": 8.5,
                        "rubric_total_100": 0.0,
                        "pass": False,
                        "feedback": "Review parsing failed — blocked by quality gate",
                        "_provider": result.provider,
                    }
    
    if review:
        review["_provider"] = review.get("_provider", result.provider)
        scores = review.get("scores", {})
        if scores:
            avg = sum(scores.values()) / len(scores)
            review["avg"] = round(avg, 1)

        # Deterministic micro-niche audit — acts as a hard guardrail.
        det = _deterministic_micro_niche_audit(
            title=str(pack.get("title", "") or ""),
            topic=str(pack.get("_topic", "") or ""),
            content=str(content or ""),
        )
        review["_deterministic"] = {
            "numbers": det.get("numbers", 0),
            "numbered_list_items": det.get("numbered_list_items", 0),
            "hard_fail_reasons": det.get("hard_fail_reasons", []),
        }

        rubric_scores = review.get("rubric_scores", {})
        det_scores = det.get("rubric_scores", {}) if isinstance(det.get("rubric_scores"), dict) else {}

        # Merge rubric: take the minimum of (LLM score, deterministic score) per criterion.
        rubric_keys = [
            "hyper_specific_niche",
            "personal_story_regret",
            "numbered_list_variations",
            "practical_steps_measurable",
            "quantifiable_value",
            "low_cost_no_gear",
            "visual_engaging_format",
            "seo_longtail_keywords",
            "cta_expansion_ladder",
            "length_scannability",
        ]
        merged_rubric: Dict[str, float] = {}
        for k in rubric_keys:
            llm_v = rubric_scores.get(k, None) if isinstance(rubric_scores, dict) else None
            det_v = det_scores.get(k, None)
            if llm_v is None and det_v is None:
                merged_rubric[k] = 0.0
            elif llm_v is None:
                merged_rubric[k] = _clamp_score(det_v)
            elif det_v is None:
                merged_rubric[k] = _clamp_score(llm_v)
            else:
                merged_rubric[k] = min(_clamp_score(llm_v), _clamp_score(det_v))

        review["rubric_scores"] = merged_rubric
        try:
            rubric_avg = sum(float(v) for v in merged_rubric.values()) / len(merged_rubric)
            review["rubric_total_100"] = int(round(rubric_avg * 10))
        except Exception:
            review["rubric_total_100"] = float(review.get("rubric_total_100", 0.0) or 0.0)

        review["pass"] = (float(review.get("avg", 0.0) or 0.0) >= TIKTOK_MIN_AVG) and (
            float(review.get("rubric_total_100", 0.0) or 0.0) >= RUBRIC_MIN_100
        )

        # Hard-fail if deterministic requirements are missing.
        det_fail = det.get("hard_fail_reasons", []) if isinstance(det, dict) else []
        if det_fail:
            review["pass"] = False
            fb_existing = str(review.get("feedback", "") or "").strip()
            fb_det = "Deterministic gate failed: " + "; ".join(det_fail)
            review["feedback"] = (fb_det + (" " + fb_existing if fb_existing else "")).strip()

        # Deterministic micro-niche audit — prevents "LLM said pass" when key winner-pattern parts are missing.
        audit = _audit_micro_niche(content)
        review["_deterministic_audit"] = audit

        rubric_scores = review.get("rubric_scores")
        if not isinstance(rubric_scores, dict):
            rubric_scores = {}

        hard_fail_reasons: List[str] = []

        if not audit["has_regret"]:
            rubric_scores["personal_story_regret"] = float(min(float(rubric_scores.get("personal_story_regret", 10) or 10), 6.0))
            hard_fail_reasons.append("missing regret/trial-error line (use 'wish/regret/mistake I made' early)")
        if int(audit["numbered_items"] or 0) < 15:
            rubric_scores["numbered_list_variations"] = float(min(float(rubric_scores.get("numbered_list_variations", 10) or 10), 6.0))
            hard_fail_reasons.append("numbered Variations/Layouts/Uses list < 15 items")
        if not audit["has_low_cost"]:
            rubric_scores["low_cost_no_gear"] = float(min(float(rubric_scores.get("low_cost_no_gear", 10) or 10), 6.0))
            hard_fail_reasons.append("missing low/no-cost + no-fancy-gear alternative")
        if not audit["has_expansion_ladder"]:
            rubric_scores["cta_expansion_ladder"] = float(min(float(rubric_scores.get("cta_expansion_ladder", 10) or 10), 6.0))
            hard_fail_reasons.append("missing CTA/Expansion Ladder (start small → scale)")
        if int(audit["numeric_count"] or 0) < 15:
            # Keep this as a soft fail on TikTok specificity + micro-niche quantifiable value.
            if isinstance(scores, dict):
                scores["specificity"] = float(min(float(scores.get("specificity", 10) or 10), 8.0))
                review["scores"] = scores
                avg = sum(scores.values()) / len(scores)
                review["avg"] = round(avg, 1)
            rubric_scores["quantifiable_value"] = float(min(float(rubric_scores.get("quantifiable_value", 10) or 10), 7.0))
            hard_fail_reasons.append("not enough specific numbers (need 15+)")
        if audit["has_markdown"]:
            if isinstance(scores, dict):
                scores["formatting"] = float(min(float(scores.get("formatting", 10) or 10), 7.0))
                review["scores"] = scores
                avg = sum(scores.values()) / len(scores)
                review["avg"] = round(avg, 1)
            hard_fail_reasons.append("contains Markdown markers (**/##/###)")

        # Recompute rubric_total_100 from (possibly adjusted) rubric_scores.
        if rubric_scores:
            try:
                rubric_avg = sum(float(v) for v in rubric_scores.values()) / len(rubric_scores)
                review["rubric_total_100"] = int(round(rubric_avg * 10))
            except Exception:
                pass
        review["rubric_scores"] = rubric_scores

        # Apply hard fail if any winner-pattern requirement is missing.
        if hard_fail_reasons:
            review["pass"] = False
            fb_existing = str(review.get("feedback", "") or "").strip()
            fb_prefix = "Deterministic micro-niche audit failed: " + "; ".join(hard_fail_reasons) + "."
            review["feedback"] = (fb_prefix + (" " + fb_existing if fb_existing else "")).strip()

        # Deterministic hard-fail: forbidden hook openers.
        import re as _re
        title = str(pack.get("title", "") or "").strip()
        if _re.match(r"^(did|do)\s+you\s+know\b", title, flags=_re.IGNORECASE):
            review["avg"] = float(min(float(review.get("avg", 0.0) or 0.0), 8.0))
            review["pass"] = False
            fb_existing = str(review.get("feedback", "") or "").strip()
            fb_ban = "Forbidden hook opener: title starts with 'Did/Do you know'. Rewrite the first line using contrarian/myth-buster/checklist/price-anchor." 
            review["feedback"] = (fb_ban + (" " + fb_existing if fb_existing else "")).strip()
    
    return review


def make_tiktok_account_variant(
    base_pack: Dict[str, Any],
    *,
    topic: str,
    account_label: str,
    variant_id: str,
) -> Dict[str, Any]:
    """Create a per-account TikTok variant to reduce duplicate/copyright detection.

    This rewrites the long caption while preserving factual meaning, numbers,
    structure, and micro-niche depth.

    Returns a NEW pack dict (does not mutate base_pack).
    """
    # Keep it safe: if anything goes wrong, fall back to base_pack.
    try:
        content = str(base_pack.get("content_formatted", "") or "").strip()
        title = str(base_pack.get("title", "") or "").strip()
        if not content or len(content) < 1200:
            return dict(base_pack)

        # Bound the content to keep the rewrite prompt stable.
        snippet = content
        if len(snippet) > 3900:
            snippet = snippet[:3900]

        variant_prompt = f"""You are rewriting a TikTok caption into a DISTINCT variant for a different TikTok account.

GOAL:
- Make the writing clearly different (different hook, different phrasing, reorder lists/examples),
  but keep the SAME practical meaning and keep specific numbers/facts.
- This is to avoid duplicate-content detection across multiple TikTok accounts.

ACCOUNT LABEL: {account_label}
VARIANT ID: {variant_id}

TOPIC (keep aligned to this long-tail): {topic}

ORIGINAL TITLE:
{title}

ORIGINAL CAPTION (source):
{snippet}

HARD RULES:
- Output ONLY valid JSON (no markdown, no code fences).
- Title must be 1 line hook with 1+ specific number.
- content_formatted MUST be PLAIN TEXT (TikTok shows raw text): NO **, NO ###, NO markdown.
- Keep 3500-4000 characters.
- Must include: trial-error regret line, reality checks, common mistakes, practical summary.
- Must include a numbered Variations/Layouts/Uses list with AT LEAST 15 items total (each 1 short line).
- Must include CTA + Expansion Ladder (Start tiny → weekly → monthly).
- Preserve at least 15 specific numbers total.
- Rewrite synonyms and sentence structure; reorder at least 30% of the content.

Return JSON schema:
{
  "title": "...",
  "content_formatted": "...",
  "hashtags": ["NanoNiche1", "NanoNiche2", "NanoNiche3"]
}
"""

        # Prefer a provider different from the generator when possible.
        gen_provider = str(base_pack.get("_gen_provider", "") or "").strip()
        providers = ["github_models", "perplexity", "gemini", "openai"]
        if gen_provider in providers:
            providers.remove(gen_provider)
            providers.append(gen_provider)

        result = call_llm(
            variant_prompt,
            system=QUALITY_CONTENT_SYSTEM,
            max_tokens=4500,
            temperature=0.55,
            providers=providers,
        )
        if not result.success:
            return dict(base_pack)

        parsed = _extract_json(result.text)
        if not isinstance(parsed, dict):
            return dict(base_pack)

        variant_pack = dict(base_pack)
        if parsed.get("title"):
            variant_pack["title"] = str(parsed["title"]).strip()
        if parsed.get("content_formatted"):
            vcontent = _ensure_section_breaks(_strip_markdown(str(parsed["content_formatted"]).strip()))
            variant_pack["content_formatted"] = vcontent

        # Hashtags: keep exactly 5 total (3 micro from variant + 2 broad auto)
        raw_micro = (parsed.get("hashtags") or [])
        if isinstance(raw_micro, list):
            micro_tags = [str(t).lstrip('#').strip() for t in raw_micro if str(t).strip()][:3]
        else:
            micro_tags = []

        broad_tags = _auto_pick_broad_hashtags(topic, micro_tags, n=2)
        broad_tags = [t.lstrip('#').strip() for t in broad_tags]
        variant_pack["hashtags"] = micro_tags + broad_tags
        _enforce_5_hashtags(variant_pack, topic)

        variant_pack["_variant_of"] = str(base_pack.get("_source", "") or "")
        variant_pack["_variant_for_account"] = account_label
        variant_pack["_variant_id"] = variant_id
        variant_pack["_variant_provider"] = result.provider
        variant_pack["_variant_model"] = result.model

        # Re-run review on the variant so quality gate is consistent
        review = _review_quality_content(variant_pack)
        if review:
            tiktok_avg = float(review.get("avg", 0.0) or 0.0)
            rubric_total = float(review.get("rubric_total_100", 0.0) or 0.0)
            variant_pack["_review_score"] = tiktok_avg
            variant_pack["_rubric_total_100"] = rubric_total
            variant_pack["_rubric_pass"] = rubric_total >= 92
            variant_pack["_review_pass"] = (tiktok_avg >= 9.0) and (rubric_total >= 92)
            variant_pack["_review_feedback"] = review.get("feedback", "")
            variant_pack["_review_provider"] = review.get("_provider", "")
            if isinstance(review.get("rubric_scores"), dict):
                variant_pack["_rubric_scores"] = review.get("rubric_scores")

        return variant_pack

    except Exception:
        return dict(base_pack)


def get_unused_topics(top_n: int = 10) -> List[Tuple[str, float, str, str]]:
    """Get top niche_hunter topics that haven't been published yet.
    
    Returns list of (topic, score, niche, hook) tuples.
    """
    import sqlite3
    
    db_path = os.path.join(os.path.dirname(__file__), "niche_hunter.db")
    if not os.path.exists(db_path):
        return []
    
    def _norm_tokens(text: str) -> set:
        import re
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        toks = [t for t in re.split(r"\s+", text) if t]
        # keep informative tokens only
        stop = {
            "do", "you", "know", "that", "can", "make", "your", "with", "just",
            "how", "to", "in", "under", "for", "and", "the", "a", "an",
        }
        return {t for t in toks if len(t) >= 4 and t not in stop}

    # Get published titles/topics (direct sqlite3 — avoids circular import from web.app)
    published_texts = set()
    viralops_db = os.path.join(os.path.dirname(__file__), "web", "viralops.db")
    if os.path.exists(viralops_db):
        try:
            _vconn = sqlite3.connect(viralops_db)
            rows = _vconn.execute("SELECT title, extra_fields FROM posts WHERE status = 'published'").fetchall()
            for title, extra_fields in rows:
                title = (title or "").strip()
                if title:
                    published_texts.add(title)
                # Prefer original picked topic if stored
                if extra_fields:
                    try:
                        obj = json.loads(extra_fields)
                        if isinstance(obj, dict):
                            t = (obj.get("topic") or "").strip()
                            if t:
                                published_texts.add(t)
                    except Exception:
                        pass
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
        topic_words = _norm_tokens(topic)
        is_used = False
        for pub_text in published_texts:
            pub_words = _norm_tokens(pub_text)
            if not pub_words:
                continue
            shared = len(topic_words & pub_words)
            # Strong match: 3+ shared informative tokens
            if shared >= 3:
                is_used = True
                break
            # Smaller topic strings: shared/len(topic) ratio
            if topic_words and (shared / max(1, len(topic_words))) >= 0.6 and shared >= 2:
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
4. HOOK — Is the first line a strong hook (question/contrarian/myth/checklist) with a specific number/fact? Would the first 2 sentences stop someone scrolling?
5. SPECIFICITY — Concrete numbers ($prices, timeframes, quantities, temperatures)? At least 15 specific numbers? Or vague advice?
6. ACTIONABILITY — Reader can do this TODAY with what they have?
7. FORMATTING — PLAIN TEXT ONLY for TikTok. Uses emoji section headers (🌿 🫙 ❌ ✅)? NO ** bold markers? NO ### headings? NO Markdown at all? Clean and readable?
   DEDUCT 2 POINTS if content contains ** or ### or ## — these show as ugly literal characters on TikTok.

Output ONLY valid JSON:
{"scores": {"answer_quality": N, "content_depth": N, "tone": N, "hook": N, "specificity": N, "actionability": N, "formatting": N}, "avg": N.N, "pass": true/false, "feedback": "Specific issues to fix", "improved_title": "Rewrite the first line into a stronger hook using 1+ specific number/fact (question/contrarian/myth/checklist)"}
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
