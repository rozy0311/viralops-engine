"""
Social Platform API Connectors — ViralOps Engine
Real posting APIs for TikTok, Instagram, YouTube, Twitter/X, LinkedIn, Facebook, Pinterest.

Architecture:
  - Each connector uses official API / OAuth2
  - All connectors share BaseSocialPublisher interface
  - Credentials from .env (never hardcoded)
  - Rate-limit aware with retry-after logic
  - Returns PublishResult for every operation

Credential setup (add to .env):
  TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
  INSTAGRAM_ACCESS_TOKEN, INSTAGRAM_BUSINESS_ACCOUNT_ID
  FACEBOOK_PAGE_TOKEN, FACEBOOK_PAGE_ID
  YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN
  LINKEDIN_ACCESS_TOKEN, LINKEDIN_ORG_ID
  TIKTOK_ACCESS_TOKEN, TIKTOK_OPEN_ID
  PINTEREST_ACCESS_TOKEN, PINTEREST_BOARD_ID
"""
import os
import json
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger()


# ════════════════════════════════════════════════
# Base Publisher Interface
# ════════════════════════════════════════════════

class BaseSocialPublisher(ABC):
    """Base class for all social platform publishers."""

    PLATFORM: str = ""
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @abstractmethod
    async def publish(self, content: dict) -> dict:
        """Publish content. Returns dict with success, post_url, post_id, error."""
        ...

    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if credentials are valid."""
        ...

    async def _retry_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """HTTP request with retry-after logic for 429s."""
        client = await self._get_client()
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = await getattr(client, method)(url, **kwargs)

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", self.RETRY_DELAY * (attempt + 1)))
                    logger.warning(f"{self.PLATFORM}.rate_limited", retry_after=retry_after, attempt=attempt)
                    await asyncio.sleep(retry_after)
                    continue

                return resp

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"{self.PLATFORM}.timeout", attempt=attempt)
                await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
            except Exception as e:
                last_error = e
                logger.error(f"{self.PLATFORM}.request_error", error=str(e), attempt=attempt)
                await asyncio.sleep(self.RETRY_DELAY)

        raise last_error or Exception(f"{self.PLATFORM}: Max retries exceeded")


# ════════════════════════════════════════════════
# Twitter/X Publisher (v2 API)
# ════════════════════════════════════════════════

class TwitterPublisher(BaseSocialPublisher):
    """Post tweets via Twitter API v2 (OAuth 1.0a User Context)."""

    PLATFORM = "twitter"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("TWITTER_API_KEY", "")
        self.api_secret = os.getenv("TWITTER_API_SECRET", "")
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN", "")
        self.access_secret = os.getenv("TWITTER_ACCESS_SECRET", "")
        self.base_url = "https://api.twitter.com/2"

    def _oauth_headers(self) -> dict:
        """Generate OAuth 1.0a headers (simplified — use authlib in production)."""
        return {"Authorization": f"Bearer {self.access_token}"}

    async def publish(self, content: dict) -> dict:
        """
        Post a tweet.
        content: {caption, hashtags, thread_parts?, media_url?}
        """
        if not self.access_token:
            return {"success": False, "error": "TWITTER_ACCESS_TOKEN not configured", "platform": self.PLATFORM}

        caption = content.get("caption", "")
        thread_parts = content.get("thread_parts", [])

        try:
            if thread_parts and len(thread_parts) > 1:
                return await self._post_thread(thread_parts)

            resp = await self._retry_request("post", f"{self.base_url}/tweets",
                json={"text": caption[:280]},
                headers=self._oauth_headers())

            if resp.status_code in (200, 201):
                data = resp.json().get("data", {})
                tweet_id = data.get("id", "")
                return {
                    "success": True,
                    "post_id": tweet_id,
                    "post_url": f"https://twitter.com/i/status/{tweet_id}",
                    "platform": self.PLATFORM,
                }
            else:
                return {"success": False, "error": resp.text, "platform": self.PLATFORM}

        except Exception as e:
            return {"success": False, "error": str(e), "platform": self.PLATFORM}

    async def _post_thread(self, parts: list) -> dict:
        """Post a Twitter thread (reply chain)."""
        previous_id = None
        thread_urls = []

        for i, part in enumerate(parts):
            payload = {"text": part[:280]}
            if previous_id:
                payload["reply"] = {"in_reply_to_tweet_id": previous_id}

            resp = await self._retry_request("post", f"{self.base_url}/tweets",
                json=payload, headers=self._oauth_headers())

            if resp.status_code in (200, 201):
                data = resp.json().get("data", {})
                previous_id = data.get("id", "")
                thread_urls.append(f"https://twitter.com/i/status/{previous_id}")
            else:
                return {"success": False, "error": f"Thread failed at part {i+1}: {resp.text}",
                        "platform": self.PLATFORM, "partial_urls": thread_urls}

        return {
            "success": True,
            "post_id": previous_id,
            "post_url": thread_urls[0] if thread_urls else "",
            "thread_urls": thread_urls,
            "platform": self.PLATFORM,
        }

    async def test_connection(self) -> bool:
        try:
            resp = await self._retry_request("get", f"{self.base_url}/users/me",
                headers=self._oauth_headers())
            return resp.status_code == 200
        except Exception:
            return False


# ════════════════════════════════════════════════
# Instagram Publisher (Graph API via Facebook)
# ════════════════════════════════════════════════

class InstagramPublisher(BaseSocialPublisher):
    """Post to Instagram via Facebook Graph API (Business accounts only)."""

    PLATFORM = "instagram"

    def __init__(self):
        super().__init__()
        self.access_token = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
        self.account_id = os.getenv("INSTAGRAM_BUSINESS_ACCOUNT_ID", "")
        self.base_url = "https://graph.facebook.com/v19.0"

    async def publish(self, content: dict) -> dict:
        """
        Post to Instagram.
        content: {caption, hashtags, image_url?, video_url?}
        For images: creates media container → publishes
        For reels: creates video container → publishes
        """
        if not self.access_token or not self.account_id:
            return {"success": False, "error": "Instagram credentials not configured", "platform": self.PLATFORM}

        caption = content.get("caption", "")
        hashtags = content.get("hashtags", [])
        image_url = content.get("image_url", "")
        video_url = content.get("video_url", "")

        # Append hashtags to caption
        if hashtags:
            tag_str = " ".join(hashtags[:5])
            if len(caption) + len(tag_str) + 2 <= 2200:
                caption = f"{caption}\n\n{tag_str}"

        try:
            # Step 1: Create media container
            if video_url:
                container_data = {
                    "media_type": "REELS",
                    "video_url": video_url,
                    "caption": caption[:2200],
                    "access_token": self.access_token,
                }
            elif image_url:
                container_data = {
                    "image_url": image_url,
                    "caption": caption[:2200],
                    "access_token": self.access_token,
                }
            else:
                return {"success": False, "error": "Instagram requires image_url or video_url", "platform": self.PLATFORM}

            resp = await self._retry_request("post",
                f"{self.base_url}/{self.account_id}/media",
                data=container_data)

            if resp.status_code != 200:
                return {"success": False, "error": f"Container creation failed: {resp.text}", "platform": self.PLATFORM}

            container_id = resp.json().get("id")

            # Step 2: Wait for processing (video) then publish
            if video_url:
                await asyncio.sleep(10)  # Wait for video processing

            resp2 = await self._retry_request("post",
                f"{self.base_url}/{self.account_id}/media_publish",
                data={"creation_id": container_id, "access_token": self.access_token})

            if resp2.status_code == 200:
                media_id = resp2.json().get("id", "")
                return {
                    "success": True,
                    "post_id": media_id,
                    "post_url": f"https://www.instagram.com/p/{media_id}/",
                    "platform": self.PLATFORM,
                }
            else:
                return {"success": False, "error": f"Publish failed: {resp2.text}", "platform": self.PLATFORM}

        except Exception as e:
            return {"success": False, "error": str(e), "platform": self.PLATFORM}

    async def test_connection(self) -> bool:
        try:
            resp = await self._retry_request("get",
                f"{self.base_url}/{self.account_id}",
                params={"access_token": self.access_token, "fields": "id,username"})
            return resp.status_code == 200
        except Exception:
            return False


# ════════════════════════════════════════════════
# Facebook Page Publisher (Graph API)
# ════════════════════════════════════════════════

class FacebookPublisher(BaseSocialPublisher):
    """Post to Facebook Page via Graph API."""

    PLATFORM = "facebook"

    def __init__(self):
        super().__init__()
        self.page_token = os.getenv("FACEBOOK_PAGE_TOKEN", "")
        self.page_id = os.getenv("FACEBOOK_PAGE_ID", "")
        self.base_url = "https://graph.facebook.com/v19.0"

    async def publish(self, content: dict) -> dict:
        if not self.page_token or not self.page_id:
            return {"success": False, "error": "Facebook credentials not configured", "platform": self.PLATFORM}

        caption = content.get("caption", "")
        hashtags = content.get("hashtags", [])
        image_url = content.get("image_url", "")
        link = content.get("link", "")

        if hashtags:
            tag_str = " ".join(hashtags[:5])
            caption = f"{caption}\n\n{tag_str}"

        try:
            payload = {"message": caption[:5000], "access_token": self.page_token}
            if image_url:
                payload["url"] = image_url
                endpoint = f"{self.base_url}/{self.page_id}/photos"
            elif link:
                payload["link"] = link
                endpoint = f"{self.base_url}/{self.page_id}/feed"
            else:
                endpoint = f"{self.base_url}/{self.page_id}/feed"

            resp = await self._retry_request("post", endpoint, data=payload)

            if resp.status_code == 200:
                post_id = resp.json().get("id", "")
                return {
                    "success": True,
                    "post_id": post_id,
                    "post_url": f"https://www.facebook.com/{post_id}",
                    "platform": self.PLATFORM,
                }
            else:
                return {"success": False, "error": resp.text, "platform": self.PLATFORM}

        except Exception as e:
            return {"success": False, "error": str(e), "platform": self.PLATFORM}

    async def test_connection(self) -> bool:
        try:
            resp = await self._retry_request("get",
                f"{self.base_url}/{self.page_id}",
                params={"access_token": self.page_token, "fields": "id,name"})
            return resp.status_code == 200
        except Exception:
            return False


# ════════════════════════════════════════════════
# YouTube Publisher (Data API v3)
# ════════════════════════════════════════════════

class YouTubePublisher(BaseSocialPublisher):
    """Upload videos to YouTube via Data API v3 (OAuth2)."""

    PLATFORM = "youtube"

    def __init__(self):
        super().__init__()
        self.client_id = os.getenv("YOUTUBE_CLIENT_ID", "")
        self.client_secret = os.getenv("YOUTUBE_CLIENT_SECRET", "")
        self.refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN", "")
        self._access_token = ""
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.upload_url = "https://www.googleapis.com/upload/youtube/v3/videos"

    async def _refresh_access_token(self):
        """Refresh OAuth2 access token."""
        client = await self._get_client()
        resp = await client.post("https://oauth2.googleapis.com/token", data={
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
        })
        if resp.status_code == 200:
            self._access_token = resp.json()["access_token"]
        else:
            raise Exception(f"YouTube token refresh failed: {resp.text}")

    async def publish(self, content: dict) -> dict:
        """
        Upload video or create post on YouTube.
        content: {title, caption (description), hashtags, video_path?, video_url?,
                  privacy_status?, category_id?}
        """
        if not self.refresh_token:
            return {"success": False, "error": "YouTube credentials not configured", "platform": self.PLATFORM}

        try:
            await self._refresh_access_token()

            title = content.get("title", "")[:100]
            description = content.get("caption", "")
            hashtags = content.get("hashtags", [])
            video_path = content.get("video_path", "")
            privacy = content.get("privacy_status", "private")  # Default private for safety
            category_id = content.get("category_id", "22")  # 22 = People & Blogs

            # Append hashtags at end of description (YouTube shows first 3 above title)
            if hashtags:
                tag_str = " ".join(hashtags[:5])
                description = f"{description}\n\n{tag_str}"

            if not video_path:
                return {"success": False, "error": "video_path required for YouTube upload", "platform": self.PLATFORM}

            # Resumable upload
            metadata = {
                "snippet": {
                    "title": title,
                    "description": description[:5000],
                    "categoryId": category_id,
                    "tags": [t.replace("#", "") for t in hashtags[:15]],
                },
                "status": {
                    "privacyStatus": privacy,
                    "selfDeclaredMadeForKids": False,
                }
            }

            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
            }

            # Step 1: Initiate resumable upload
            resp = await self._retry_request("post",
                f"{self.upload_url}?uploadType=resumable&part=snippet,status",
                json=metadata, headers=headers)

            if resp.status_code == 200:
                upload_url = resp.headers.get("Location", "")
                if not upload_url:
                    return {"success": False, "error": "No upload URL returned", "platform": self.PLATFORM}

                # Step 2: Upload video file
                with open(video_path, "rb") as f:
                    video_data = f.read()

                client = await self._get_client()
                resp2 = await client.put(upload_url,
                    content=video_data,
                    headers={"Content-Type": "video/*"})

                if resp2.status_code in (200, 201):
                    video_id = resp2.json().get("id", "")
                    return {
                        "success": True,
                        "post_id": video_id,
                        "post_url": f"https://www.youtube.com/watch?v={video_id}",
                        "platform": self.PLATFORM,
                    }

            return {"success": False, "error": resp.text, "platform": self.PLATFORM}

        except Exception as e:
            return {"success": False, "error": str(e), "platform": self.PLATFORM}

    async def test_connection(self) -> bool:
        try:
            await self._refresh_access_token()
            resp = await self._retry_request("get",
                f"{self.base_url}/channels?part=snippet&mine=true",
                headers={"Authorization": f"Bearer {self._access_token}"})
            return resp.status_code == 200
        except Exception:
            return False


# ════════════════════════════════════════════════
# LinkedIn Publisher (v2 API)
# ════════════════════════════════════════════════

class LinkedInPublisher(BaseSocialPublisher):
    """Post to LinkedIn via v2 API (ugcPosts / shares)."""

    PLATFORM = "linkedin"

    def __init__(self):
        super().__init__()
        self.access_token = os.getenv("LINKEDIN_ACCESS_TOKEN", "")
        self.org_id = os.getenv("LINKEDIN_ORG_ID", "")  # Optional: org page
        self.base_url = "https://api.linkedin.com/v2"

    async def _get_person_urn(self) -> str:
        """Get the authenticated user's URN."""
        resp = await self._retry_request("get", f"{self.base_url}/userinfo",
            headers={"Authorization": f"Bearer {self.access_token}"})
        if resp.status_code == 200:
            sub = resp.json().get("sub", "")
            return f"urn:li:person:{sub}"
        raise Exception(f"LinkedIn: cannot get user info: {resp.text}")

    async def publish(self, content: dict) -> dict:
        if not self.access_token:
            return {"success": False, "error": "LinkedIn credentials not configured", "platform": self.PLATFORM}

        caption = content.get("caption", "")
        hashtags = content.get("hashtags", [])
        link = content.get("link", "")

        if hashtags:
            tag_str = " ".join(hashtags[:5])
            caption = f"{caption}\n\n{tag_str}"

        try:
            author = f"urn:li:organization:{self.org_id}" if self.org_id else await self._get_person_urn()

            payload = {
                "author": author,
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {"text": caption[:3000]},
                        "shareMediaCategory": "NONE",
                    }
                },
                "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
            }

            if link:
                payload["specificContent"]["com.linkedin.ugc.ShareContent"]["shareMediaCategory"] = "ARTICLE"
                payload["specificContent"]["com.linkedin.ugc.ShareContent"]["media"] = [{
                    "status": "READY",
                    "originalUrl": link,
                }]

            resp = await self._retry_request("post", f"{self.base_url}/ugcPosts",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "X-Restli-Protocol-Version": "2.0.0",
                    "Content-Type": "application/json",
                })

            if resp.status_code in (200, 201):
                post_id = resp.json().get("id", "")
                return {
                    "success": True,
                    "post_id": post_id,
                    "post_url": f"https://www.linkedin.com/feed/update/{post_id}",
                    "platform": self.PLATFORM,
                }
            else:
                return {"success": False, "error": resp.text, "platform": self.PLATFORM}

        except Exception as e:
            return {"success": False, "error": str(e), "platform": self.PLATFORM}

    async def test_connection(self) -> bool:
        try:
            resp = await self._retry_request("get", f"{self.base_url}/userinfo",
                headers={"Authorization": f"Bearer {self.access_token}"})
            return resp.status_code == 200
        except Exception:
            return False


# ════════════════════════════════════════════════
# TikTok Publisher (Content Posting API)
# ════════════════════════════════════════════════

class TikTokPublisher(BaseSocialPublisher):
    """Post to TikTok via Content Posting API v2."""

    PLATFORM = "tiktok"

    def __init__(self):
        super().__init__()
        self.access_token = os.getenv("TIKTOK_ACCESS_TOKEN", "")
        self.open_id = os.getenv("TIKTOK_OPEN_ID", "")
        self.base_url = "https://open.tiktokapis.com/v2"

    async def publish(self, content: dict) -> dict:
        """
        Post video to TikTok.
        content: {caption, hashtags, video_url or video_path, music_id?}
        """
        if not self.access_token:
            return {"success": False, "error": "TikTok credentials not configured", "platform": self.PLATFORM}

        caption = content.get("caption", "")
        hashtags = content.get("hashtags", [])
        video_url = content.get("video_url", "")

        if hashtags:
            tag_str = " ".join(hashtags[:5])
            caption = f"{caption} {tag_str}"

        try:
            # Direct post (video URL method)
            payload = {
                "post_info": {
                    "title": caption[:2200],
                    "privacy_level": "PUBLIC_TO_EVERYONE",
                    "disable_duet": False,
                    "disable_comment": False,
                    "disable_stitch": False,
                },
                "source_info": {
                    "source": "PULL_FROM_URL",
                    "video_url": video_url,
                },
            }

            resp = await self._retry_request("post",
                f"{self.base_url}/post/publish/video/init/",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json; charset=UTF-8",
                })

            if resp.status_code == 200:
                data = resp.json().get("data", {})
                publish_id = data.get("publish_id", "")
                return {
                    "success": True,
                    "post_id": publish_id,
                    "post_url": "",  # TikTok doesn't return URL immediately
                    "platform": self.PLATFORM,
                    "note": "Video is processing — URL available after processing completes",
                }
            else:
                return {"success": False, "error": resp.text, "platform": self.PLATFORM}

        except Exception as e:
            return {"success": False, "error": str(e), "platform": self.PLATFORM}

    async def test_connection(self) -> bool:
        try:
            resp = await self._retry_request("get",
                f"{self.base_url}/user/info/",
                headers={"Authorization": f"Bearer {self.access_token}"},
                params={"fields": "open_id,display_name"})
            return resp.status_code == 200
        except Exception:
            return False


# ════════════════════════════════════════════════
# Pinterest Publisher (v5 API)
# ════════════════════════════════════════════════

class PinterestPublisher(BaseSocialPublisher):
    """Create pins via Pinterest API v5."""

    PLATFORM = "pinterest"

    def __init__(self):
        super().__init__()
        self.access_token = os.getenv("PINTEREST_ACCESS_TOKEN", "")
        self.board_id = os.getenv("PINTEREST_BOARD_ID", "")
        self.base_url = "https://api.pinterest.com/v5"

    async def publish(self, content: dict) -> dict:
        if not self.access_token or not self.board_id:
            return {"success": False, "error": "Pinterest credentials not configured", "platform": self.PLATFORM}

        title = content.get("title", "")[:100]
        caption = content.get("caption", "")[:500]
        image_url = content.get("image_url", "")
        link = content.get("link", "")

        if not image_url:
            return {"success": False, "error": "Pinterest requires image_url", "platform": self.PLATFORM}

        try:
            payload = {
                "board_id": self.board_id,
                "title": title,
                "description": caption,
                "media_source": {"source_type": "image_url", "url": image_url},
            }
            if link:
                payload["link"] = link

            resp = await self._retry_request("post", f"{self.base_url}/pins",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json",
                })

            if resp.status_code in (200, 201):
                pin_id = resp.json().get("id", "")
                return {
                    "success": True,
                    "post_id": pin_id,
                    "post_url": f"https://www.pinterest.com/pin/{pin_id}/",
                    "platform": self.PLATFORM,
                }
            else:
                return {"success": False, "error": resp.text, "platform": self.PLATFORM}

        except Exception as e:
            return {"success": False, "error": str(e), "platform": self.PLATFORM}

    async def test_connection(self) -> bool:
        try:
            resp = await self._retry_request("get", f"{self.base_url}/user_account",
                headers={"Authorization": f"Bearer {self.access_token}"})
            return resp.status_code == 200
        except Exception:
            return False


# ════════════════════════════════════════════════
# Publisher Factory
# ════════════════════════════════════════════════

PUBLISHERS = {
    "twitter": TwitterPublisher,
    "instagram": InstagramPublisher,
    "facebook": FacebookPublisher,
    "youtube": YouTubePublisher,
    "linkedin": LinkedInPublisher,
    "tiktok": TikTokPublisher,
    "pinterest": PinterestPublisher,
}


def get_social_publisher(platform: str) -> Optional[BaseSocialPublisher]:
    """Factory: get publisher instance for a platform."""
    cls = PUBLISHERS.get(platform)
    if cls:
        return cls()
    return None


def get_all_configured_publishers() -> dict[str, BaseSocialPublisher]:
    """Return publishers that have credentials configured."""
    configured = {}
    for name, cls in PUBLISHERS.items():
        pub = cls()
        # Check if any credential is set
        has_creds = any(
            getattr(pub, attr, "")
            for attr in vars(pub)
            if "token" in attr.lower() or "key" in attr.lower() or "secret" in attr.lower()
        )
        if has_creds:
            configured[name] = pub
    return configured


async def test_all_connections() -> dict[str, bool]:
    """Test connections for all configured publishers."""
    results = {}
    for name, pub in get_all_configured_publishers().items():
        try:
            results[name] = await pub.test_connection()
        except Exception:
            results[name] = False
    return results
