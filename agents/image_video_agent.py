"""
Image/Video Agent — CTO Sub-Agent (EMADS-PR v1.0)
Generates image prompts for content. 
Future: DALL-E / Stable Diffusion integration.
"""
import structlog

logger = structlog.get_logger()


def generate_image_prompt(state: dict) -> dict:
    """Generate image prompt from content pack."""
    content_pack = state.get("content_pack", {})
    title = content_pack.get("title", "")
    
    # If content_factory already generated an image prompt, use it
    if content_pack.get("image_prompt"):
        state["image_prompt"] = content_pack["image_prompt"]
    else:
        state["image_prompt"] = f"Professional lifestyle photo related to: {title}. Clean composition, natural lighting, modern aesthetic, 16:9 aspect ratio."

    state["image_status"] = "prompt_generated"
    logger.info("image_video.prompt_generated", prompt=state["image_prompt"][:80])
    return state
