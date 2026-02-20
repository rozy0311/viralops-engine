"""
ViralOps Engine -- Shopify Blog Publisher (REAL Implementation)

Shopify Admin REST API -- blog article CRUD.
Syncs content as published/draft blog articles to a Shopify store.

API Docs: https://shopify.dev/docs/api/admin-rest/2025-01/resources/article
Auth: Custom App Admin API access token (X-Shopify-Access-Token header)
Rate limit: 2 requests/second (bucket leak, 40 burst capacity)

SETUP:
1. Go to Shopify Admin -> Settings -> Apps and sales channels -> Develop apps
2. Create a Custom App with "read_content" + "write_content" scopes
3. Install the app and copy the Admin API access token
4. Set env vars: SHOPIFY_SHOP, SHOPIFY_ACCESS_TOKEN, SHOPIFY_BLOG_ID
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import base64
from datetime import datetime, timezone
from typing import Any

import httpx

from core.models import PublishResult, QueueItem

logger = logging.getLogger("viralops.publisher.shopify_blog")


class ShopifyBlogPublisher:
    """Real Shopify Admin REST API publisher for blog articles."""

    platform = "shopify_blog"
    API_VERSION = "2025-01"

    def __init__(self, account_id: str = "shopify_main"):
        self.account_id = account_id
        self._shop: str | None = None
        self._token: str | None = None
        self._blog_id: str | None = None
        self._blog_handle: str | None = None
        self._public_domain: str | None = None
        self._base_url: str | None = None
        self._connected = False
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0

    async def connect(self) -> bool:
        """Set up Shopify Admin API connection and verify blog exists."""
        prefix = self.account_id.upper().replace("-", "_")
        self._shop = os.environ.get(
            f"{prefix}_SHOP", os.environ.get("SHOPIFY_SHOP")
        )
        self._token = os.environ.get(
            f"{prefix}_ACCESS_TOKEN",
            os.environ.get("SHOPIFY_ACCESS_TOKEN"),
        )
        self._blog_id = os.environ.get(
            f"{prefix}_BLOG_ID", os.environ.get("SHOPIFY_BLOG_ID")
        )

        public_domain = os.environ.get(
            f"{prefix}_PUBLIC_DOMAIN",
            os.environ.get(
                "SHOPIFY_PUBLIC_DOMAIN",
                os.environ.get("SHOPIFY_CUSTOM_DOMAIN", ""),
            ),
        )
        self._public_domain = (public_domain or "").strip() or None

        if not all([self._shop, self._token, self._blog_id]):
            logger.error(
                "Shopify [%s]: Missing SHOPIFY_SHOP, SHOPIFY_ACCESS_TOKEN, "
                "or SHOPIFY_BLOG_ID",
                self.account_id,
            )
            return False

        # Normalize shop domain
        shop = self._shop.strip()
        if not shop.endswith(".myshopify.com"):
            shop = f"{shop}.myshopify.com"
        self._shop = shop
        self._base_url = (
            f"https://{self._shop}/admin/api/{self.API_VERSION}"
        )

        self._client = httpx.AsyncClient(
            headers={
                "X-Shopify-Access-Token": self._token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

        try:
            # Verify blog exists
            resp = await self._rate_limited_get(
                f"{self._base_url}/blogs/{self._blog_id}.json"
            )
            resp.raise_for_status()
            blog_data = resp.json().get("blog", {})
            blog_title = blog_data.get("title", "Unknown")
            self._blog_handle = (blog_data.get("handle") or "").strip() or None

            self._connected = True
            logger.info(
                "Shopify [%s]: Connected to store '%s', blog '%s' (id=%s)",
                self.account_id,
                self._shop,
                blog_title,
                self._blog_id,
            )
            return True

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error(
                    "Shopify [%s]: Invalid access token", self.account_id
                )
            elif e.response.status_code == 404:
                logger.error(
                    "Shopify [%s]: Blog ID %s not found",
                    self.account_id,
                    self._blog_id,
                )
            else:
                logger.error(
                    "Shopify [%s]: Connection failed: HTTP %s",
                    self.account_id,
                    e.response.status_code,
                )
            return False
        except httpx.HTTPError as e:
            logger.error(
                "Shopify [%s]: Connection error: %s", self.account_id, e
            )
            return False

    async def publish(self, item: QueueItem, content: dict) -> PublishResult:
        """
        Create a blog article in the Shopify store.

        content keys:
            title (str): Article title (required)
            body_html (str): Article body in HTML (required)
            body (str): Fallback for body_html (plain text or HTML)
            summary_html (str): Article excerpt/summary HTML
            tags (str | list): Comma-separated tags or list
            handle (str): URL-friendly slug (auto-generated if empty)
            published (bool): True=published, False=draft (default True)
            image_url (str): Featured image URL
            image_alt (str): Featured image alt text
            author (str): Author name
            seo_title (str): Meta title for SEO
            seo_description (str): Meta description for SEO
            template_suffix (str): Custom Liquid template suffix
        """
        if not self._connected or not self._client:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error="Shopify not connected",
            )

        title = content.get("title", content.get("caption", ""))
        body_html = content.get(
            "body_html", content.get("body", content.get("text", ""))
        )

        if not title:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error="Article title is required",
            )

        # Build article payload
        article: dict[str, Any] = {
            "title": title,
            "body_html": body_html,
            "published": content.get("published", True),
        }

        # Handle
        handle = content.get("handle", "")
        if not handle:
            handle = self._slugify(title)
        article["handle"] = handle

        # Tags
        tags = content.get("tags", "")
        if isinstance(tags, list):
            tags = ", ".join(tags)
        if tags:
            article["tags"] = tags

        # Summary/excerpt
        if content.get("summary_html"):
            article["summary_html"] = content["summary_html"]

        # Author
        if content.get("author"):
            article["author"] = content["author"]

        # Featured image
        # Shopify accepts either a public `src` or a base64 `attachment`.
        if content.get("image_url"):
            article["image"] = {
                "src": content["image_url"],
                "alt": content.get("image_alt", title),
            }
        else:
            image_local_path = str(content.get("image_local_path", "") or "").strip()
            if image_local_path and os.path.isfile(image_local_path):
                try:
                    encoded = base64.b64encode(open(image_local_path, "rb").read()).decode("utf-8")
                    article["image"] = {
                        "attachment": encoded,
                        "filename": os.path.basename(image_local_path),
                        "alt": str(content.get("image_alt", title) or title)[:255],
                    }
                except Exception as e:
                    logger.warning(
                        "Shopify [%s]: Failed to read/encode local image: %s",
                        self.account_id,
                        str(e)[:200],
                    )

        # SEO metafields
        if content.get("seo_title"):
            article["metafields_global_title_tag"] = content["seo_title"]
        if content.get("seo_description"):
            article["metafields_global_description_tag"] = content[
                "seo_description"
            ]

        # Template suffix
        if content.get("template_suffix"):
            article["template_suffix"] = content["template_suffix"]

        payload = {"article": article}
        sent_image = bool(article.get("image"))

        def _inject_inline_image(html: str, *, src: str, alt: str) -> str:
            """Inject a single inline figure+img after the first paragraph."""
            if not html or not src:
                return html
            safe_alt = (alt or "").replace('"', "&quot;")
            block = (
                "<figure>"
                f"<img src=\"{src}\" alt=\"{safe_alt}\" loading=\"lazy\" />"
                f"<figcaption>{safe_alt}</figcaption>"
                "</figure>"
            )
            # If already has an image, don't duplicate.
            if re.search(r"<img\b", html, re.IGNORECASE):
                return html
            m = re.search(r"</p>", html, re.IGNORECASE)
            if m:
                return html[: m.end()] + "\n" + block + html[m.end() :]
            return block + "\n" + html

        def _build_public_url(article_handle: str) -> str:
            domain = (self._public_domain or self._shop or "").strip()
            if domain.startswith("http://") or domain.startswith("https://"):
                domain = domain.split("//", 1)[-1]
            blog_handle = (self._blog_handle or "").strip()
            if domain and blog_handle and article_handle:
                return f"https://{domain}/blogs/{blog_handle}/{article_handle}"
            # Fallback (admin domain always exists; public path may still resolve)
            if self._shop and article_handle:
                return f"https://{self._shop}/blogs/{self._blog_id}/{article_handle}"
            return ""

        async def _upsert_article_metafield(
            *,
            article_id: str,
            namespace: str,
            key: str,
            value: str,
            value_type: str = "single_line_text_field",
        ) -> bool:
            """Upsert an article metafield (REST). Used for SEO fields like global.description_tag."""
            if not self._connected or not self._client or not self._base_url:
                return False
            aid = str(article_id or "").strip()
            if not aid:
                return False
            ns = str(namespace or "").strip() or "global"
            k = str(key or "").strip()
            v = str(value or "").strip()
            if not k or not v:
                return False

            # 1) List current metafields on the article
            resp = await self._rate_limited_get(
                f"{self._base_url}/articles/{aid}/metafields.json"
            )
            resp.raise_for_status()
            metafields = (resp.json() or {}).get("metafields", [])
            existing = None
            if isinstance(metafields, list):
                for mf in metafields:
                    if not isinstance(mf, dict):
                        continue
                    if str(mf.get("namespace") or "").strip() == ns and str(mf.get("key") or "").strip() == k:
                        existing = mf
                        break

            payload = {"metafield": {"namespace": ns, "key": k, "value": v, "type": value_type}}

            # 2) Update or create
            if isinstance(existing, dict) and existing.get("id"):
                mf_id = str(existing.get("id"))
                resp2 = await self._rate_limited_put(
                    f"{self._base_url}/metafields/{mf_id}.json",
                    json={"metafield": {"id": int(mf_id), "value": v, "type": value_type}},
                )
                resp2.raise_for_status()
                return True

            resp3 = await self._rate_limited_post(
                f"{self._base_url}/articles/{aid}/metafields.json",
                json=payload,
            )
            resp3.raise_for_status()
            return True

        try:
            url = f"{self._base_url}/blogs/{self._blog_id}/articles.json"

            try:
                resp = await self._rate_limited_post(url, json=payload)
                resp.raise_for_status()
                data = resp.json().get("article", {})
            except httpx.HTTPStatusError as e:
                # Handle collision: retry once with a unique handle suffix.
                status = e.response.status_code
                detail = (e.response.text or "")[:500]
                is_handle_taken = (
                    status == 422
                    and ("handle" in detail.lower())
                    and ("taken" in detail.lower() or "already" in detail.lower())
                )
                if not is_handle_taken:
                    raise

                unique_handle = f"{article['handle']}-{int(time.time())}"
                payload2 = {"article": dict(article, handle=unique_handle)}
                resp = await self._rate_limited_post(url, json=payload2)
                resp.raise_for_status()
                data = resp.json().get("article", {})

            article_id = str(data.get("id", ""))
            article_handle = data.get("handle", handle)

            # If an image was provided (src or attachment), Shopify may return a CDN src.
            uploaded_src = ""
            try:
                img = data.get("image") if isinstance(data, dict) else None
                if isinstance(img, dict):
                    uploaded_src = str(img.get("src") or img.get("url") or "").strip()
            except Exception:
                uploaded_src = ""

            # Some Shopify API variants (or async CDN processing) don't return
            # `image.src` immediately in the create response, even if the image
            # was accepted. Poll the article a few times to fetch the CDN URL.
            if (not uploaded_src) and sent_image and article_id:
                for _ in range(5):
                    try:
                        await asyncio.sleep(1.0)
                        art_resp = await self._rate_limited_get(
                            f"{self._base_url}/articles/{article_id}.json"
                        )
                        if art_resp.status_code != 200:
                            continue
                        art = (art_resp.json() or {}).get("article", {})
                        img2 = art.get("image") if isinstance(art, dict) else None
                        if isinstance(img2, dict):
                            uploaded_src = str(
                                img2.get("src") or img2.get("url") or ""
                            ).strip()
                        if uploaded_src:
                            break
                    except Exception:
                        continue

            # If we have a CDN src, update article: inject inline <img alt=...>
            if uploaded_src:
                try:
                    new_html = _inject_inline_image(
                        body_html,
                        src=uploaded_src,
                        alt=str(content.get("image_alt", title) or title),
                    )
                    update_article: dict[str, Any] = {
                        "id": int(article_id),
                        "body_html": new_html,
                    }
                    # Best-effort: set SEO fields on update too (some Shopify API variants
                    # don't persist them reliably on create).
                    if content.get("seo_title"):
                        update_article["metafields_global_title_tag"] = content.get("seo_title")
                    if content.get("seo_description"):
                        update_article["metafields_global_description_tag"] = content.get("seo_description")
                    await self._rate_limited_put(
                        f"{self._base_url}/articles/{article_id}.json",
                        json={
                            "article": {
                                **update_article,
                            }
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Shopify [%s]: Post-create update failed: %s",
                        self.account_id,
                        str(e)[:200],
                    )
            else:
                # If no uploaded_src was returned (no inline injection), still set SEO via update.
                try:
                    if content.get("seo_title") or content.get("seo_description"):
                        update_article: dict[str, Any] = {"id": int(article_id)}
                        if content.get("seo_title"):
                            update_article["metafields_global_title_tag"] = content.get("seo_title")
                        if content.get("seo_description"):
                            update_article["metafields_global_description_tag"] = content.get("seo_description")
                        await self._rate_limited_put(
                            f"{self._base_url}/articles/{article_id}.json",
                            json={"article": update_article},
                        )
                except Exception as e:
                    logger.warning(
                        "Shopify [%s]: SEO update skipped: %s",
                        self.account_id,
                        str(e)[:200],
                    )

            # Shopify SEO for Articles is stored as metafields in namespace `global`.
            # Ensure these are set so the Admin "Search engine listing" shows a good snippet.
            try:
                seo_title = str(content.get("seo_title") or "").strip()
                seo_desc = str(content.get("seo_description") or "").strip()
                if seo_title:
                    await _upsert_article_metafield(
                        article_id=article_id,
                        namespace="global",
                        key="title_tag",
                        value=seo_title[:255],
                    )
                if seo_desc:
                    await _upsert_article_metafield(
                        article_id=article_id,
                        namespace="global",
                        key="description_tag",
                        value=seo_desc[:1000],
                    )
            except Exception as e:
                logger.warning(
                    "Shopify [%s]: SEO metafields update skipped: %s",
                    self.account_id,
                    str(e)[:200],
                )

            post_url = _build_public_url(article_handle)
            admin_url = (
                f"https://{self._shop}/admin/articles/{article_id}"
            )

            logger.info(
                "Shopify [%s]: Article '%s' created (id=%s, handle=%s)",
                self.account_id,
                title[:50],
                article_id,
                article_handle,
            )
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=True,
                published_at=datetime.now(timezone.utc),
                post_url=post_url,
                post_id=article_id,
                metadata={
                    "admin_url": admin_url,
                    "handle": article_handle,
                    "blog_handle": self._blog_handle,
                    "public_domain": self._public_domain,
                    "published": content.get("published", True),
                },
            )

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            detail = e.response.text[:300]
            if status == 429:
                retry_after = e.response.headers.get("Retry-After", "2")
                logger.warning(
                    "Shopify [%s]: Rate limited, retry after %ss",
                    self.account_id,
                    retry_after,
                )
            elif status == 422:
                logger.error(
                    "Shopify [%s]: Validation error: %s",
                    self.account_id,
                    detail,
                )
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error=f"HTTP {status}: {detail}",
            )
        except Exception as e:
            return PublishResult(
                queue_item_id=item.id,
                platform=self.platform,
                success=False,
                error=str(e),
            )

    async def update_article(
        self, article_id: str, content: dict
    ) -> PublishResult:
        """Update an existing blog article."""
        if not self._connected or not self._client:
            return PublishResult(
                queue_item_id="",
                platform=self.platform,
                success=False,
                error="Shopify not connected",
            )

        article: dict[str, Any] = {"id": int(article_id)}

        if "title" in content:
            article["title"] = content["title"]
        if "body_html" in content or "body" in content:
            article["body_html"] = content.get("body_html", content.get("body"))
        if "tags" in content:
            tags = content["tags"]
            if isinstance(tags, list):
                tags = ", ".join(tags)
            article["tags"] = tags
        if "published" in content:
            article["published"] = content["published"]
        if "image_url" in content:
            article["image"] = {
                "src": content["image_url"],
                "alt": content.get("image_alt", ""),
            }
        if "seo_title" in content:
            article["metafields_global_title_tag"] = content["seo_title"]
        if "seo_description" in content:
            article["metafields_global_description_tag"] = content["seo_description"]

        try:
            resp = await self._rate_limited_put(
                f"{self._base_url}/articles/{article_id}.json",
                json={"article": article},
            )
            resp.raise_for_status()
            data = resp.json().get("article", {})
            logger.info(
                "Shopify [%s]: Article %s updated", self.account_id, article_id
            )
            return PublishResult(
                queue_item_id="",
                platform=self.platform,
                success=True,
                post_id=article_id,
                post_url=f"https://{self._shop}/admin/articles/{article_id}",
            )
        except Exception as e:
            return PublishResult(
                queue_item_id="",
                platform=self.platform,
                success=False,
                error=str(e),
            )

    async def list_articles(
        self,
        limit: int = 50,
        published_status: str = "any",
        since_id: str | None = None,
    ) -> list[dict]:
        """List blog articles with pagination support."""
        if not self._connected or not self._client:
            return []

        params: dict[str, Any] = {
            "limit": min(limit, 250),
            "published_status": published_status,
        }
        if since_id:
            params["since_id"] = since_id

        try:
            resp = await self._rate_limited_get(
                f"{self._base_url}/blogs/{self._blog_id}/articles.json",
                params=params,
            )
            resp.raise_for_status()
            return resp.json().get("articles", [])
        except Exception as e:
            logger.error(
                "Shopify [%s]: Failed to list articles: %s",
                self.account_id, e,
            )
            return []

    async def get_article(self, article_id: str) -> dict | None:
        """Get a single article by ID."""
        if not self._connected or not self._client:
            return None
        try:
            resp = await self._rate_limited_get(
                f"{self._base_url}/articles/{article_id}.json"
            )
            resp.raise_for_status()
            return resp.json().get("article")
        except Exception as e:
            logger.error(
                "Shopify [%s]: Failed to get article %s: %s",
                self.account_id, article_id, e,
            )
            return None

    async def delete_article(self, article_id: str) -> bool:
        """Delete a blog article."""
        if not self._connected or not self._client:
            return False
        try:
            resp = await self._rate_limited_request(
                "DELETE", f"{self._base_url}/articles/{article_id}.json"
            )
            resp.raise_for_status()
            logger.info(
                "Shopify [%s]: Article %s deleted", self.account_id, article_id
            )
            return True
        except Exception as e:
            logger.error(
                "Shopify [%s]: Failed to delete article %s: %s",
                self.account_id, article_id, e,
            )
            return False

    # --- Rate limiting helpers ---

    async def _rate_limited_get(
        self, url: str, **kwargs: Any
    ) -> httpx.Response:
        """GET request with 2 req/sec rate limit."""
        return await self._rate_limited_request("GET", url, **kwargs)

    async def _rate_limited_post(
        self, url: str, **kwargs: Any
    ) -> httpx.Response:
        """POST request with 2 req/sec rate limit."""
        return await self._rate_limited_request("POST", url, **kwargs)

    async def _rate_limited_put(
        self, url: str, **kwargs: Any
    ) -> httpx.Response:
        """PUT request with 2 req/sec rate limit."""
        return await self._rate_limited_request("PUT", url, **kwargs)

    async def _rate_limited_request(
        self, method: str, url: str, **kwargs: Any
    ) -> httpx.Response:
        """Execute HTTP request with Shopify 2 req/sec rate limiting."""
        import asyncio

        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < 0.5:  # 2 req/sec = 0.5s between requests
            await asyncio.sleep(0.5 - elapsed)

        self._last_request_time = time.monotonic()
        return await self._client.request(method, url, **kwargs)

    # --- Utility ---

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to URL-friendly handle."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_]+", "-", text)
        text = re.sub(r"-+", "-", text)
        return text[:200]

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False