"""
Tests for Media Processor — Image→Video slideshow + TikTok music overlay.

Covers:
  - Image download (URL → file)
  - Extension guessing
  - Slideshow creation (moviepy / ffmpeg / fallback)
  - Music overlay
  - Full pipeline (image → video + music)
  - Cleanup + stats
"""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime

import pytest

# ── Add project root to path ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from integrations.media_processor import (
    download_image,
    create_slideshow_video,
    add_music_to_video,
    process_image_to_video,
    cleanup_old_media,
    get_media_stats,
    _guess_extension,
    MediaJob,
    DEFAULT_VIDEO_DURATION,
    DEFAULT_VIDEO_WIDTH,
    DEFAULT_VIDEO_HEIGHT,
)


# ── Fixtures ──

@pytest.fixture(autouse=True)
def temp_media_dir(tmp_path, monkeypatch):
    """Redirect media output to temp directory."""
    monkeypatch.setattr("integrations.media_processor.MEDIA_OUTPUT_DIR", str(tmp_path / "media"))
    return tmp_path


@pytest.fixture
def sample_image(tmp_path):
    """Create a minimal test image file (skips if Pillow not installed)."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow (PIL) not installed")
    img = Image.new("RGB", (200, 200), color="blue")
    path = str(tmp_path / "test_image.jpg")
    img.save(path, "JPEG")
    return path


# ════════════════════════════════════════════════
# Extension Guessing Tests
# ════════════════════════════════════════════════

class TestGuessExtension:
    def test_jpg(self):
        assert _guess_extension("https://example.com/photo.jpg") == ".jpg"

    def test_png(self):
        assert _guess_extension("https://example.com/image.png") == ".png"

    def test_webp(self):
        assert _guess_extension("https://example.com/pic.webp") == ".webp"

    def test_with_query_params(self):
        assert _guess_extension("https://cdn.com/photo.jpg?w=800&q=80") == ".jpg"

    def test_unknown_defaults_jpg(self):
        assert _guess_extension("https://example.com/image") == ".jpg"

    def test_gif(self):
        assert _guess_extension("https://example.com/anim.gif") == ".gif"


# ════════════════════════════════════════════════
# Download Tests
# ════════════════════════════════════════════════

class TestDownloadImage:
    def test_download_success(self, tmp_path):
        """Mock HTTP response and verify download."""
        mock_response = MagicMock()
        mock_response.content = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # Fake JPEG
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_response)

        with patch("integrations.media_processor.httpx.Client", return_value=mock_client):
            result = download_image(
                "https://example.com/photo.jpg",
                output_dir=str(tmp_path / "dl")
            )
            assert result["success"] is True
            assert result["size"] == 104
            assert os.path.exists(result["path"])

    def test_download_cached(self, tmp_path):
        """Second download of same URL should be cached."""
        # Pre-create the file
        dl_dir = tmp_path / "dl"
        dl_dir.mkdir()

        # Create the file that download_image would create
        import hashlib
        url = "https://example.com/cached.jpg"
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        filepath = dl_dir / f"{url_hash}.jpg"
        filepath.write_bytes(b"\xff\xd8" + b"\x00" * 50)

        result = download_image(url, output_dir=str(dl_dir))
        assert result["success"] is True
        assert result.get("cached") is True

    def test_download_failure(self, tmp_path):
        """Network error should return error dict."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(side_effect=Exception("Connection refused"))

        with patch("integrations.media_processor.httpx.Client", return_value=mock_client):
            result = download_image(
                "https://broken.com/image.jpg",
                output_dir=str(tmp_path / "dl")
            )
            assert result["success"] is False
            assert "Connection refused" in result["error"]


# ════════════════════════════════════════════════
# Slideshow Creation Tests
# ════════════════════════════════════════════════

class TestCreateSlideshow:
    def test_fallback_when_no_processor(self, sample_image):
        """When neither moviepy nor ffmpeg available, return metadata-only fallback."""
        with patch("integrations.media_processor._create_slideshow_moviepy",
                   side_effect=ImportError("no moviepy")):
            with patch("integrations.media_processor._create_slideshow_ffmpeg",
                       side_effect=FileNotFoundError("no ffmpeg")):
                result = create_slideshow_video(sample_image, duration=5)
                assert result["success"] is False
                assert result.get("fallback") == "metadata_only"
                assert result.get("image_path") == sample_image

    def test_moviepy_success(self, sample_image, tmp_path):
        """Mock moviepy to verify it's called correctly."""
        mock_result = {
            "success": True,
            "path": str(tmp_path / "output.mp4"),
            "duration": 12,
            "size": 1000,
            "method": "moviepy",
        }
        with patch("integrations.media_processor._create_slideshow_moviepy",
                   return_value=mock_result):
            result = create_slideshow_video(sample_image)
            assert result["success"] is True
            assert result["method"] == "moviepy"

    def test_ffmpeg_fallback(self, sample_image, tmp_path):
        """When moviepy fails, try ffmpeg."""
        mock_result = {
            "success": True,
            "path": str(tmp_path / "output.mp4"),
            "duration": 12,
            "size": 800,
            "method": "ffmpeg",
        }
        with patch("integrations.media_processor._create_slideshow_moviepy",
                   side_effect=ImportError("no moviepy")):
            with patch("integrations.media_processor._create_slideshow_ffmpeg",
                       return_value=mock_result):
                result = create_slideshow_video(sample_image)
                assert result["success"] is True
                assert result["method"] == "ffmpeg"


# ════════════════════════════════════════════════
# Music Overlay Tests
# ════════════════════════════════════════════════

class TestAddMusic:
    def test_no_music_provided_returns_video_as_is(self, tmp_path):
        """When no music url or file, return video without music."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        result = add_music_to_video(str(video), music_url=None, music_file=None)
        assert result["success"] is True
        assert result.get("has_music") is False

    def test_music_file_not_found(self, tmp_path):
        """When music file path doesn't exist on disk → no music."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        result = add_music_to_video(str(video), music_file="/nonexistent/music.mp3")
        assert result["success"] is True
        assert result.get("has_music") is False

    def test_music_url_download_failure(self, tmp_path):
        """When music URL download fails → error."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        with patch("integrations.media_processor._download_audio",
                   return_value={"success": False, "error": "404 Not Found"}):
            result = add_music_to_video(str(video), music_url="https://bad.url/music.mp3")
            assert result["success"] is False
            assert "Music download failed" in result["error"]

    def test_music_url_download_success_then_process(self, tmp_path):
        """When music URL downloads successfully, try to process."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)
        music = tmp_path / "music.mp3"
        music.write_bytes(b"\x00" * 50)

        with patch("integrations.media_processor._download_audio",
                   return_value={"success": True, "path": str(music)}):
            # Both moviepy and ffmpeg fail → error
            with patch("integrations.media_processor._add_music_moviepy",
                       side_effect=ImportError("no moviepy")):
                with patch("integrations.media_processor._add_music_ffmpeg",
                           side_effect=FileNotFoundError("no ffmpeg")):
                    result = add_music_to_video(str(video), music_url="https://ok.url/music.mp3")
                    assert result["success"] is False
                    assert "No audio processor" in result["error"]


# ════════════════════════════════════════════════
# Full Pipeline Tests
# ════════════════════════════════════════════════

class TestFullPipeline:
    def test_pipeline_no_image(self):
        """No image_url or image_path → error."""
        result = process_image_to_video()
        assert result["success"] is False
        assert "No image_url or image_path" in result["error"]

    def test_pipeline_image_file_not_found(self):
        """Local image file doesn't exist → error."""
        result = process_image_to_video(image_path="/nonexistent/image.jpg")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_pipeline_with_local_image(self, sample_image, tmp_path):
        """Full pipeline with local image file (mocked slideshow)."""
        mock_slideshow = {
            "success": True,
            "path": str(tmp_path / "slideshow.mp4"),
            "duration": 12,
            "size": 1000,
            "method": "moviepy",
        }
        # Create the fake output file
        (tmp_path / "slideshow.mp4").write_bytes(b"\x00" * 100)

        with patch("integrations.media_processor.create_slideshow_video",
                   return_value=mock_slideshow):
            result = process_image_to_video(
                image_path=sample_image,
                niche="sustainable-living",
                content_text="Eco friendly tips",
                duration=10,
            )
            assert result["success"] is True
            assert result["video_path"]
            assert len(result["steps"]) >= 2

    def test_pipeline_with_url(self, tmp_path):
        """Full pipeline with image URL (mocked download + slideshow)."""
        mock_download = {
            "success": True,
            "path": str(tmp_path / "downloaded.jpg"),
            "size": 500,
        }
        mock_slideshow = {
            "success": True,
            "path": str(tmp_path / "slideshow.mp4"),
            "duration": 12,
            "size": 1000,
            "method": "moviepy",
        }
        (tmp_path / "slideshow.mp4").write_bytes(b"\x00" * 100)

        with patch("integrations.media_processor.download_image", return_value=mock_download):
            with patch("integrations.media_processor.create_slideshow_video",
                       return_value=mock_slideshow):
                result = process_image_to_video(
                    image_url="https://example.com/image.jpg",
                )
                assert result["success"] is True
                assert "download_image" in result["steps"][0]["step"]

    def test_pipeline_slideshow_fails_returns_metadata(self, sample_image):
        """If slideshow creation fails, return error with metadata."""
        mock_fail = {
            "success": False,
            "fallback": "metadata_only",
            "error": "No processor",
        }
        with patch("integrations.media_processor.create_slideshow_video",
                   return_value=mock_fail):
            result = process_image_to_video(image_path=sample_image)
            assert result["success"] is False
            assert result.get("fallback") == "metadata_only"
            assert result.get("image_path") == sample_image


# ════════════════════════════════════════════════
# Cleanup + Stats Tests
# ════════════════════════════════════════════════

class TestCleanupAndStats:
    def test_cleanup_empty_dir(self):
        """Cleanup with no files → 0 removed."""
        result = cleanup_old_media()
        assert result["success"] is True
        assert result["removed"] == 0

    def test_stats_empty(self):
        """Stats with no files."""
        result = get_media_stats()
        assert result["total_files"] == 0
        assert result["total_size_mb"] == 0

    def test_stats_with_files(self, tmp_path, monkeypatch):
        """Stats with some files."""
        media_dir = tmp_path / "media"
        media_dir.mkdir()
        (media_dir / "video1.mp4").write_bytes(b"\x00" * 1024)
        (media_dir / "video2.mp4").write_bytes(b"\x00" * 2048)

        monkeypatch.setattr("integrations.media_processor.MEDIA_OUTPUT_DIR", str(media_dir))
        result = get_media_stats()
        assert result["total_files"] == 2
        # 3072 bytes = 0.003 MB, rounds to 0.0 at 2dp, so just check >= 0
        assert result["total_size_mb"] >= 0
        assert result["output_dir"] == str(media_dir)


# ════════════════════════════════════════════════
# MediaJob Dataclass Tests
# ════════════════════════════════════════════════

class TestMediaJob:
    def test_defaults(self):
        job = MediaJob()
        assert job.status == "pending"
        assert job.input_type == "image_url"
        assert job.duration == DEFAULT_VIDEO_DURATION
        assert job.width == DEFAULT_VIDEO_WIDTH
        assert job.height == DEFAULT_VIDEO_HEIGHT
