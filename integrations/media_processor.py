"""
Media Processor — Image→Video slideshow with TikTok music overlay.

Pipeline:
  1. Download image from URL (or read local file)
  2. Create slideshow video (zoom/pan Ken Burns effect, 10-15 sec)
  3. Overlay TikTok music track (from tiktok_music.py recommendation)
  4. Export as .mp4 (H.264 + AAC) ready for TikTok/IG/FB upload

Dependencies:
  - moviepy >= 2.0 (video compositing + audio overlay)
  - Pillow >= 10.0 (image processing)

If moviepy is not installed, falls back to ffmpeg CLI.
If neither is available, returns metadata-only (no actual video created).
"""

import os
import hashlib
import shutil
import tempfile
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

import httpx
import structlog

logger = structlog.get_logger()

# ── Output directory ──
MEDIA_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "media_output"
)

# ── Default video settings ──
DEFAULT_VIDEO_DURATION = 12       # seconds per image slide
DEFAULT_VIDEO_WIDTH = 1080        # TikTok portrait: 1080x1920
DEFAULT_VIDEO_HEIGHT = 1920
DEFAULT_FPS = 30
DEFAULT_ZOOM_FACTOR = 1.15        # Ken Burns zoom factor (15% zoom)


@dataclass
class MediaJob:
    """Represents a media processing job."""
    id: str = ""
    input_type: str = "image_url"     # image_url / image_file / video_url / video_file
    input_path: str = ""              # URL or file path
    output_path: str = ""             # Generated .mp4 path
    music_track_id: str = ""          # TikTok music track ID
    music_url: str = ""               # TikTok sound URL
    music_title: str = ""
    duration: int = DEFAULT_VIDEO_DURATION
    width: int = DEFAULT_VIDEO_WIDTH
    height: int = DEFAULT_VIDEO_HEIGHT
    status: str = "pending"           # pending / processing / done / error
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str = ""


# ════════════════════════════════════════════════
# Image Download
# ════════════════════════════════════════════════

def download_image(url: str, output_dir: str = None) -> dict:
    """
    Download an image from URL to local file.

    Returns: {"success": bool, "path": str, "size": int, "content_type": str}
    """
    if not output_dir:
        output_dir = os.path.join(MEDIA_OUTPUT_DIR, "downloads")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Generate filename from URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        ext = _guess_extension(url)

        filename = f"{url_hash}{ext}"
        filepath = os.path.join(output_dir, filename)

        # Skip if already downloaded
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            logger.info("media.download_cached", path=filepath, size=size)
            return {"success": True, "path": filepath, "size": size, "cached": True}

        # Download
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        with open(filepath, "wb") as f:
            f.write(response.content)

        size = len(response.content)
        logger.info("media.downloaded", url=url[:80], path=filepath,
                     size=size, content_type=content_type)

        return {
            "success": True,
            "path": filepath,
            "size": size,
            "content_type": content_type,
            "cached": False,
        }

    except Exception as e:
        logger.error("media.download_error", url=url[:80], error=str(e))
        return {"success": False, "error": str(e)}


def _guess_extension(url: str) -> str:
    """Guess file extension from URL."""
    url_lower = url.lower().split("?")[0]
    for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]:
        if url_lower.endswith(ext):
            return ext
    return ".jpg"


# ════════════════════════════════════════════════
# Image → Video Slideshow (Ken Burns effect)
# ════════════════════════════════════════════════

def create_slideshow_video(
    image_path: str,
    output_path: str = None,
    duration: int = DEFAULT_VIDEO_DURATION,
    width: int = DEFAULT_VIDEO_WIDTH,
    height: int = DEFAULT_VIDEO_HEIGHT,
    fps: int = DEFAULT_FPS,
    zoom_factor: float = DEFAULT_ZOOM_FACTOR,
) -> dict:
    """
    Create a slideshow video from a single image with Ken Burns zoom effect.

    The image slowly zooms in over the duration, creating motion for TikTok.

    Returns: {"success": bool, "path": str, "duration": int}
    """
    if not output_path:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(MEDIA_OUTPUT_DIR, f"{base}_slideshow.mp4")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try moviepy first
    try:
        return _create_slideshow_moviepy(image_path, output_path, duration,
                                          width, height, fps, zoom_factor)
    except ImportError:
        logger.info("media.moviepy_not_available, trying ffmpeg")

    # Try ffmpeg CLI
    try:
        return _create_slideshow_ffmpeg(image_path, output_path, duration,
                                         width, height, fps, zoom_factor)
    except FileNotFoundError:
        logger.warning("media.ffmpeg_not_available")

    # Fallback: metadata-only
    logger.warning("media.no_video_processor_available",
                    msg="Install moviepy or ffmpeg for video generation")
    return {
        "success": False,
        "fallback": "metadata_only",
        "image_path": image_path,
        "planned_output": output_path,
        "duration": duration,
        "error": "No video processor available. Install: pip install moviepy Pillow",
    }


def _create_slideshow_moviepy(
    image_path: str, output_path: str, duration: int,
    width: int, height: int, fps: int, zoom_factor: float
) -> dict:
    """Create slideshow using moviepy (Ken Burns zoom)."""
    from moviepy import ImageClip, vfx

    # Load image
    clip = ImageClip(image_path).with_duration(duration)

    # Resize to cover the target dimensions (crop to fill)
    clip = clip.resized(height=height)
    if clip.w < width:
        clip = clip.resized(width=width)

    # Ken Burns: zoom from 1.0 to zoom_factor over duration
    def zoom_effect(get_frame, t):
        """Apply gradual zoom over time."""
        import numpy as np
        from PIL import Image

        frame = get_frame(t)
        progress = t / duration
        current_zoom = 1.0 + (zoom_factor - 1.0) * progress

        h, w = frame.shape[:2]
        new_h = int(h / current_zoom)
        new_w = int(w / current_zoom)
        y_start = (h - new_h) // 2
        x_start = (w - new_w) // 2

        cropped = frame[y_start:y_start + new_h, x_start:x_start + new_w]

        # Resize back to original dimensions
        img = Image.fromarray(cropped)
        img = img.resize((w, h), Image.LANCZOS)
        return np.array(img)

    clip = clip.transform(zoom_effect)

    # Final crop to exact target dimensions
    if clip.w > width or clip.h > height:
        x_center = clip.w // 2
        y_center = clip.h // 2
        clip = clip.cropped(
            x1=x_center - width // 2,
            y1=y_center - height // 2,
            x2=x_center + width // 2,
            y2=y_center + height // 2,
        )

    # Export
    clip.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio=False,
        preset="medium",
        logger=None,
    )
    clip.close()

    file_size = os.path.getsize(output_path)
    logger.info("media.slideshow_created", path=output_path,
                duration=duration, size=file_size, method="moviepy")

    return {
        "success": True,
        "path": output_path,
        "duration": duration,
        "size": file_size,
        "method": "moviepy",
    }


def _create_slideshow_ffmpeg(
    image_path: str, output_path: str, duration: int,
    width: int, height: int, fps: int, zoom_factor: float
) -> dict:
    """Create slideshow using ffmpeg CLI (Ken Burns zoom)."""
    import subprocess

    # ffmpeg zoompan filter for Ken Burns effect
    zoom_increment = (zoom_factor - 1.0) / (duration * fps)
    total_frames = duration * fps

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-vf", (
            f"scale={width * 2}:{height * 2},"
            f"zoompan=z='min(zoom+{zoom_increment:.6f},{zoom_factor})':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={width}x{height}:fps={fps}"
        ),
        "-t", str(duration),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr[:500]}")

    file_size = os.path.getsize(output_path)
    logger.info("media.slideshow_created", path=output_path,
                duration=duration, size=file_size, method="ffmpeg")

    return {
        "success": True,
        "path": output_path,
        "duration": duration,
        "size": file_size,
        "method": "ffmpeg",
    }


# ════════════════════════════════════════════════
# Audio Overlay — Add TikTok music to video
# ════════════════════════════════════════════════

def add_music_to_video(
    video_path: str,
    music_url: str = None,
    music_file: str = None,
    output_path: str = None,
    fade_in: float = 1.0,
    fade_out: float = 2.0,
) -> dict:
    """
    Overlay a TikTok music track onto a video file.

    Either music_url or music_file must be provided.
    Music is trimmed to match video duration, with fade in/out.

    Returns: {"success": bool, "path": str}
    """
    if not output_path:
        base = os.path.splitext(video_path)[0]
        output_path = f"{base}_with_music.mp4"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Download music if URL provided
    music_path = music_file
    if music_url and not music_file:
        dl_result = _download_audio(music_url)
        if not dl_result.get("success"):
            return {"success": False, "error": f"Music download failed: {dl_result.get('error')}"}
        music_path = dl_result["path"]

    if not music_path or not os.path.exists(music_path):
        # No music file available — return video as-is
        logger.warning("media.no_music_file", msg="No music file available, returning video without music")
        return {
            "success": True,
            "path": video_path,
            "has_music": False,
            "note": "No music file available. TikTok music URL is metadata-only (requires TikTok SDK for actual audio).",
        }

    # Try moviepy
    try:
        return _add_music_moviepy(video_path, music_path, output_path, fade_in, fade_out)
    except ImportError:
        pass

    # Try ffmpeg
    try:
        return _add_music_ffmpeg(video_path, music_path, output_path, fade_in, fade_out)
    except FileNotFoundError:
        pass

    return {
        "success": False,
        "error": "No audio processor available. Install moviepy or ffmpeg.",
    }


def _download_audio(url: str) -> dict:
    """Download audio file from URL."""
    download_dir = os.path.join(MEDIA_OUTPUT_DIR, "audio")
    os.makedirs(download_dir, exist_ok=True)

    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    filepath = os.path.join(download_dir, f"{url_hash}.mp3")

    if os.path.exists(filepath):
        return {"success": True, "path": filepath, "cached": True}

    try:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)

        return {"success": True, "path": filepath, "size": len(response.content)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _add_music_moviepy(
    video_path: str, music_path: str, output_path: str,
    fade_in: float, fade_out: float
) -> dict:
    """Add music overlay using moviepy."""
    from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip, afx

    video = VideoFileClip(video_path)
    audio = AudioFileClip(music_path)

    # Trim audio to video duration
    if audio.duration > video.duration:
        audio = audio.subclipped(0, video.duration)

    # Fade in/out
    if fade_in > 0:
        audio = audio.with_effects([afx.AudioFadeIn(fade_in)])
    if fade_out > 0 and audio.duration > fade_out:
        audio = audio.with_effects([afx.AudioFadeOut(fade_out)])

    # Combine
    final = video.with_audio(audio)

    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        logger=None,
    )

    video.close()
    audio.close()
    final.close()

    file_size = os.path.getsize(output_path)
    logger.info("media.music_added", path=output_path, size=file_size, method="moviepy")

    return {
        "success": True,
        "path": output_path,
        "size": file_size,
        "has_music": True,
        "method": "moviepy",
    }


def _add_music_ffmpeg(
    video_path: str, music_path: str, output_path: str,
    fade_in: float, fade_out: float
) -> dict:
    """Add music overlay using ffmpeg CLI."""
    import subprocess

    # Get video duration
    probe_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
    video_duration = float(probe.stdout.strip()) if probe.stdout.strip() else 12

    # Build ffmpeg command
    audio_filter = f"atrim=0:{video_duration}"
    if fade_in > 0:
        audio_filter += f",afade=t=in:st=0:d={fade_in}"
    if fade_out > 0:
        fade_start = max(0, video_duration - fade_out)
        audio_filter += f",afade=t=out:st={fade_start}:d={fade_out}"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", music_path,
        "-filter_complex", f"[1:a]{audio_filter}[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr[:500]}")

    file_size = os.path.getsize(output_path)
    logger.info("media.music_added", path=output_path, size=file_size, method="ffmpeg")

    return {
        "success": True,
        "path": output_path,
        "size": file_size,
        "has_music": True,
        "method": "ffmpeg",
    }


# ════════════════════════════════════════════════
# Full Pipeline — Image + TikTok Music → Video
# ════════════════════════════════════════════════

def process_image_to_video(
    image_url: str = None,
    image_path: str = None,
    music_track: dict = None,
    niche: str = "sustainable-living",
    content_text: str = "",
    duration: int = DEFAULT_VIDEO_DURATION,
    width: int = DEFAULT_VIDEO_WIDTH,
    height: int = DEFAULT_VIDEO_HEIGHT,
) -> dict:
    """
    Full pipeline: Image → Slideshow Video → Add TikTok Music → .mp4

    This is the main function called by RSS Auto Poster when
    media_type == "image" and tiktok_music_enabled == True.

    Args:
        image_url: URL of the image to convert
        image_path: Local path (alternative to URL)
        music_track: TikTok music track dict from recommend_music()
        niche: Content niche (for auto music selection if no track given)
        content_text: Post text (for mood-based music selection)
        duration: Video duration in seconds
        width/height: Output video dimensions

    Returns: {"success": bool, "video_path": str, "music": dict, ...}
    """
    result = {
        "success": False,
        "steps": [],
    }

    # Step 1: Get image file
    if image_url and not image_path:
        dl = download_image(image_url)
        if not dl.get("success"):
            result["error"] = f"Image download failed: {dl.get('error')}"
            return result
        image_path = dl["path"]
        result["steps"].append({"step": "download_image", "success": True, "path": image_path})
    elif image_path:
        if not os.path.exists(image_path):
            result["error"] = f"Image file not found: {image_path}"
            return result
        result["steps"].append({"step": "local_image", "path": image_path})
    else:
        result["error"] = "No image_url or image_path provided"
        return result

    # Step 2: Create slideshow video
    slideshow = create_slideshow_video(
        image_path=image_path,
        duration=duration,
        width=width,
        height=height,
    )
    result["steps"].append({"step": "create_slideshow", **slideshow})

    if not slideshow.get("success"):
        result["error"] = slideshow.get("error", "Slideshow creation failed")
        result["fallback"] = slideshow.get("fallback")
        # Even if video creation failed, return metadata
        result["image_path"] = image_path
        return result

    video_path = slideshow["path"]

    # Step 3: Get music track if not provided
    if not music_track:
        try:
            from integrations.tiktok_music import recommend_music
            music_result = recommend_music(
                text=content_text, niche=niche, limit=1
            )
            if music_result.get("tracks"):
                music_track = music_result["tracks"][0]
                result["steps"].append({
                    "step": "auto_music_selected",
                    "track": music_track.get("title"),
                    "mood": music_result.get("mood_detected"),
                })
        except Exception as e:
            logger.warning("media.music_selection_failed", error=str(e))
            result["steps"].append({"step": "music_selection", "error": str(e)})

    # Step 4: Add music overlay
    if music_track:
        music_url = music_track.get("tiktok_sound_url", "")
        music_result = add_music_to_video(
            video_path=video_path,
            music_url=music_url if music_url else None,
        )
        result["steps"].append({"step": "add_music", **music_result})

        if music_result.get("success") and music_result.get("has_music"):
            video_path = music_result["path"]

        result["music"] = {
            "track_id": music_track.get("track_id", ""),
            "title": music_track.get("title", ""),
            "artist": music_track.get("artist", ""),
            "mood": music_track.get("mood", ""),
        }
    else:
        result["music"] = None
        result["steps"].append({"step": "no_music", "reason": "No track available"})

    result["success"] = True
    result["video_path"] = video_path
    result["duration"] = duration
    result["dimensions"] = f"{width}x{height}"

    logger.info("media.pipeline_done", video_path=video_path,
                has_music=bool(music_track), duration=duration)

    return result


# ════════════════════════════════════════════════
# Cleanup
# ════════════════════════════════════════════════

def cleanup_old_media(max_age_hours: int = 72) -> dict:
    """Remove processed media files older than max_age_hours."""
    if not os.path.exists(MEDIA_OUTPUT_DIR):
        return {"success": True, "removed": 0}

    removed = 0
    now = datetime.utcnow().timestamp()
    cutoff = now - (max_age_hours * 3600)

    for root, dirs, files in os.walk(MEDIA_OUTPUT_DIR):
        for f in files:
            filepath = os.path.join(root, f)
            try:
                if os.path.getmtime(filepath) < cutoff:
                    os.remove(filepath)
                    removed += 1
            except OSError:
                pass

    logger.info("media.cleanup", removed=removed, max_age_hours=max_age_hours)
    return {"success": True, "removed": removed}


def get_media_stats() -> dict:
    """Get media output directory stats."""
    if not os.path.exists(MEDIA_OUTPUT_DIR):
        return {"total_files": 0, "total_size_mb": 0}

    total_files = 0
    total_size = 0

    for root, dirs, files in os.walk(MEDIA_OUTPUT_DIR):
        for f in files:
            filepath = os.path.join(root, f)
            total_files += 1
            total_size += os.path.getsize(filepath)

    return {
        "total_files": total_files,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "output_dir": MEDIA_OUTPUT_DIR,
    }


# ════════════════════════════════════════════════════════════════
# Multi-Image Slideshow — Combine multiple images into one video
# ════════════════════════════════════════════════════════════════

def create_multi_image_slideshow(
    image_paths: list[str],
    output_path: str | None = None,
    duration_per_image: int = 4,
    transition_duration: float = 0.5,
    width: int = DEFAULT_VIDEO_WIDTH,
    height: int = DEFAULT_VIDEO_HEIGHT,
    fps: int = DEFAULT_FPS,
    zoom_factor: float = 1.08,
) -> dict:
    """
    Create a video slideshow from multiple images with Ken Burns + crossfade.

    Args:
        image_paths: List of local image file paths
        output_path: Output video path (.mp4)
        duration_per_image: Seconds to display each image
        transition_duration: Crossfade transition duration (seconds)
        width: Output video width
        height: Output video height
        fps: Frames per second
        zoom_factor: Ken Burns zoom factor per slide

    Returns:
        {"success": bool, "path": str, "duration": int, "slides": int}
    """
    if not image_paths:
        return {"success": False, "error": "No images provided"}

    if not output_path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(MEDIA_OUTPUT_DIR, f"slideshow_{ts}.mp4")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_duration = len(image_paths) * duration_per_image

    # Try moviepy
    try:
        return _multi_slideshow_moviepy(
            image_paths, output_path, duration_per_image,
            transition_duration, width, height, fps, zoom_factor,
        )
    except ImportError:
        logger.info("media.moviepy_not_available_for_multi_slideshow")

    # Try ffmpeg
    try:
        return _multi_slideshow_ffmpeg(
            image_paths, output_path, duration_per_image,
            width, height, fps, zoom_factor,
        )
    except FileNotFoundError:
        pass

    return {
        "success": False,
        "error": "No video processor available. Install: pip install moviepy Pillow",
        "slides": len(image_paths),
        "planned_output": output_path,
    }


def _multi_slideshow_moviepy(
    image_paths: list[str], output_path: str,
    duration_per_image: int, transition_duration: float,
    width: int, height: int, fps: int, zoom_factor: float,
) -> dict:
    """Create multi-image slideshow using moviepy with crossfade transitions."""
    from moviepy import ImageClip, CompositeVideoClip, concatenate_videoclips
    import numpy as np
    from PIL import Image

    clips = []
    for img_path in image_paths:
        # Load and resize image to cover target dimensions
        pil_img = Image.open(img_path).convert("RGB")

        # Resize to cover (may crop)
        img_ratio = pil_img.width / pil_img.height
        target_ratio = width / height
        if img_ratio > target_ratio:
            new_h = height
            new_w = int(height * img_ratio)
        else:
            new_w = width
            new_h = int(width / img_ratio)

        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        # Center crop to exact dimensions
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        pil_img = pil_img.crop((left, top, left + width, top + height))

        img_array = np.array(pil_img)

        clip = ImageClip(img_array).with_duration(duration_per_image)
        clips.append(clip)

    # Concatenate with crossfade
    if transition_duration > 0 and len(clips) > 1:
        final = concatenate_videoclips(
            clips, method="compose",
            padding=-transition_duration,
        )
    else:
        final = concatenate_videoclips(clips)

    final = final.with_fps(fps)
    final.write_videofile(
        output_path, codec="libx264", audio=False,
        preset="fast", logger=None,
    )
    final.close()
    for c in clips:
        c.close()

    total_duration = int(final.duration) if hasattr(final, 'duration') else len(image_paths) * duration_per_image
    logger.info("media.multi_slideshow_created",
                slides=len(image_paths), duration=total_duration)
    return {
        "success": True,
        "path": output_path,
        "duration": total_duration,
        "slides": len(image_paths),
    }


def _multi_slideshow_ffmpeg(
    image_paths: list[str], output_path: str,
    duration_per_image: int, width: int, height: int,
    fps: int, zoom_factor: float,
) -> dict:
    """Create multi-image slideshow using ffmpeg CLI."""
    import subprocess

    # Create a file list for ffmpeg
    list_path = os.path.join(MEDIA_OUTPUT_DIR, "ffmpeg_input.txt")
    with open(list_path, "w") as f:
        for img in image_paths:
            f.write(f"file '{img}'\n")
            f.write(f"duration {duration_per_image}\n")
        # Repeat last image to avoid ffmpeg truncation
        f.write(f"file '{image_paths[-1]}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", str(fps), "-preset", "fast",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    # Cleanup
    try:
        os.remove(list_path)
    except OSError:
        pass

    if result.returncode == 0:
        total_duration = len(image_paths) * duration_per_image
        return {
            "success": True,
            "path": output_path,
            "duration": total_duration,
            "slides": len(image_paths),
        }
    else:
        return {
            "success": False,
            "error": f"ffmpeg error: {result.stderr[:300]}",
            "slides": len(image_paths),
        }


def create_multi_image_slideshow_from_urls(
    image_urls: list[str],
    output_path: str | None = None,
    duration_per_image: int = 4,
    **kwargs,
) -> dict:
    """
    Download multiple images from URLs and create a slideshow.

    Convenience wrapper around create_multi_image_slideshow().
    """
    downloaded_paths = []
    temp_dir = tempfile.mkdtemp(prefix="viralops_slides_")

    try:
        for i, url in enumerate(image_urls):
            result = download_image(url, output_dir=temp_dir)
            if result.get("success"):
                downloaded_paths.append(result["path"])
            else:
                logger.warning("media.slide_download_failed",
                               url=url, error=result.get("error"))

        if not downloaded_paths:
            return {"success": False, "error": "No images downloaded successfully"}

        return create_multi_image_slideshow(
            downloaded_paths, output_path=output_path,
            duration_per_image=duration_per_image, **kwargs,
        )
    finally:
        # Cleanup temp files
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════
# Text Overlay — Add captions / subtitles to video
# ════════════════════════════════════════════════════════════════

def add_text_overlay(
    video_path: str,
    text: str,
    output_path: str | None = None,
    position: str = "bottom",
    font_size: int = 48,
    font_color: str = "white",
    bg_color: str = "black@0.6",
    margin: int = 40,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> dict:
    """
    Add text caption/subtitle overlay to a video.

    Args:
        video_path: Input video file path
        text: Text to overlay
        output_path: Output video path (default: adds _captioned suffix)
        position: "top", "center", "bottom"
        font_size: Text font size
        font_color: Text color (CSS name or hex)
        bg_color: Background color with opacity (ffmpeg format)
        margin: Pixels from edge
        start_time: When to show text (seconds)
        end_time: When to hide text (None = full video)

    Returns:
        {"success": bool, "path": str}
    """
    if not os.path.exists(video_path):
        return {"success": False, "error": f"Video not found: {video_path}"}

    if not output_path:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_captioned{ext}"

    # Try moviepy
    try:
        return _text_overlay_moviepy(
            video_path, text, output_path, position,
            font_size, font_color, margin, start_time, end_time,
        )
    except ImportError:
        logger.info("media.moviepy_not_available_for_text_overlay")

    # Try ffmpeg
    try:
        return _text_overlay_ffmpeg(
            video_path, text, output_path, position,
            font_size, font_color, bg_color, margin,
            start_time, end_time,
        )
    except FileNotFoundError:
        pass

    return {
        "success": False,
        "error": "No video processor available for text overlay",
    }


def _text_overlay_moviepy(
    video_path: str, text: str, output_path: str,
    position: str, font_size: int, font_color: str,
    margin: int, start_time: float, end_time: float | None,
) -> dict:
    """Add text overlay using moviepy."""
    from moviepy import VideoFileClip, TextClip, CompositeVideoClip

    video = VideoFileClip(video_path)

    if end_time is None:
        end_time = video.duration

    # Position mapping
    pos_map = {
        "top": ("center", margin),
        "center": ("center", "center"),
        "bottom": ("center", video.h - margin - font_size),
    }
    pos = pos_map.get(position, pos_map["bottom"])

    # Create text clip
    txt_clip = (
        TextClip(
            text=text,
            font_size=font_size,
            color=font_color,
            method="caption",
            size=(video.w - 2 * margin, None),
        )
        .with_position(pos)
        .with_start(start_time)
        .with_duration(end_time - start_time)
    )

    final = CompositeVideoClip([video, txt_clip])
    final.write_videofile(
        output_path, codec="libx264", audio_codec="aac",
        preset="fast", logger=None,
    )

    final.close()
    video.close()

    logger.info("media.text_overlay_added", path=output_path, text=text[:50])
    return {"success": True, "path": output_path}


def _text_overlay_ffmpeg(
    video_path: str, text: str, output_path: str,
    position: str, font_size: int, font_color: str,
    bg_color: str, margin: int,
    start_time: float, end_time: float | None,
) -> dict:
    """Add text overlay using ffmpeg drawtext filter."""
    import subprocess

    # Escape text for ffmpeg
    escaped_text = text.replace("'", "'\\''").replace(":", "\\:")

    # Position
    if position == "top":
        y_expr = f"{margin}"
    elif position == "center":
        y_expr = "(h-text_h)/2"
    else:  # bottom
        y_expr = f"h-text_h-{margin}"

    # Time filter
    time_filter = ""
    if start_time > 0 or end_time is not None:
        enable_parts = []
        if start_time > 0:
            enable_parts.append(f"gte(t\\,{start_time})")
        if end_time is not None:
            enable_parts.append(f"lte(t\\,{end_time})")
        time_filter = f":enable='{'+'.join(enable_parts)}'"

    drawtext = (
        f"drawtext=text='{escaped_text}':"
        f"fontsize={font_size}:fontcolor={font_color}:"
        f"x=(w-text_w)/2:y={y_expr}:"
        f"box=1:boxcolor={bg_color}:boxborderw=10"
        f"{time_filter}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", drawtext,
        "-c:v", "libx264", "-c:a", "copy",
        "-preset", "fast",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode == 0:
        logger.info("media.text_overlay_ffmpeg", path=output_path)
        return {"success": True, "path": output_path}
    else:
        return {
            "success": False,
            "error": f"ffmpeg error: {result.stderr[:300]}",
        }


def add_subtitles(
    video_path: str,
    subtitles: list[dict],
    output_path: str | None = None,
    font_size: int = 36,
    font_color: str = "white",
) -> dict:
    """
    Add timed subtitles to a video (SRT-style).

    Args:
        video_path: Input video file path
        subtitles: List of {"text": str, "start": float, "end": float}
        output_path: Output video path
        font_size: Subtitle font size
        font_color: Subtitle color

    Returns:
        {"success": bool, "path": str, "subtitle_count": int}
    """
    if not subtitles:
        return {"success": False, "error": "No subtitles provided"}

    if not output_path:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_subtitled{ext}"

    # Generate SRT file
    srt_path = os.path.splitext(output_path)[0] + ".srt"
    _generate_srt(subtitles, srt_path)

    # Try ffmpeg subtitles filter
    try:
        import subprocess

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"subtitles={srt_path}:force_style="
                   f"'FontSize={font_size},PrimaryColour=&H00FFFFFF'",
            "-c:v", "libx264", "-c:a", "copy",
            "-preset", "fast",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            logger.info("media.subtitles_added",
                        count=len(subtitles), path=output_path)
            return {
                "success": True,
                "path": output_path,
                "subtitle_count": len(subtitles),
                "srt_path": srt_path,
            }
        else:
            return {"success": False, "error": result.stderr[:300]}

    except FileNotFoundError:
        return {
            "success": False,
            "error": "ffmpeg not found. Install ffmpeg for subtitle overlay.",
            "srt_path": srt_path,
        }


def _generate_srt(subtitles: list[dict], srt_path: str) -> None:
    """Generate an SRT subtitle file."""
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, sub in enumerate(subtitles, 1):
            start = _seconds_to_srt_time(sub["start"])
            end = _seconds_to_srt_time(sub["end"])
            text = sub["text"]
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
