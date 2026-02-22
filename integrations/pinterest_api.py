from __future__ import annotations

import os
import re
from typing import Any

import httpx


class PinterestApiError(RuntimeError):
    pass


class PinterestClient:
    """Minimal Pinterest v5 client for boards + pin creation.

    Uses env var:
      - PINTEREST_ACCESS_TOKEN

    API base defaults to https://api.pinterest.com/v5
    """

    def __init__(self, access_token: str | None = None, *, api_base: str | None = None):
        token = (access_token or os.environ.get("PINTEREST_ACCESS_TOKEN") or "").strip()
        if not token:
            raise PinterestApiError("Missing PINTEREST_ACCESS_TOKEN")
        self._token = token
        self._api_base = (api_base or os.environ.get("PINTEREST_API_BASE") or "https://api.pinterest.com/v5").rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def list_boards(self, *, page_size: int = 100) -> list[dict[str, Any]]:
        url = f"{self._api_base}/boards"
        params = {"page_size": int(page_size)}
        async with httpx.AsyncClient(timeout=30.0, headers=self._headers()) as client:
            r = await client.get(url, params=params)
        if r.status_code != 200:
            raise PinterestApiError(f"List boards failed: HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        items = data.get("items") if isinstance(data, dict) else None
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]
        # Some variants might return a raw list.
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []

    async def pick_board_id_by_name(self, name: str) -> str:
        want = str(name or "").strip().lower()
        if not want:
            raise PinterestApiError("Board name is required")

        boards = await self.list_boards(page_size=250)
        # Prefer exact name match.
        for b in boards:
            bn = str(b.get("name") or "").strip().lower()
            if bn == want:
                bid = str(b.get("id") or "").strip()
                if bid:
                    return bid
        # Fallback: match by slug-ish URL fragment
        for b in boards:
            url = str(b.get("url") or "").strip().lower()
            if want and want in url:
                bid = str(b.get("id") or "").strip()
                if bid:
                    return bid

        available = ", ".join(sorted({str(b.get("name") or "").strip() for b in boards if str(b.get("name") or "").strip()}))
        raise PinterestApiError(f"Board '{name}' not found. Available boards: {available[:300]}")

    async def create_pin(
        self,
        *,
        board_id: str,
        title: str,
        description: str,
        link: str,
        image_url: str,
        alt_text: str,
    ) -> dict[str, Any]:
        """Create a Pin.

        Pinterest v5 create pin payload (best-effort):
          POST /pins
          {
            board_id, title, description, link, alt_text,
            media_source: { source_type: "image_url", url }
          }
        """
        bid = str(board_id or "").strip()
        if not bid:
            raise PinterestApiError("board_id is required")

        t = re.sub(r"\s+", " ", str(title or "").strip())[:100]
        d = str(description or "").strip()[:500]
        lnk = str(link or "").strip()
        img = str(image_url or "").strip()
        alt = re.sub(r"\s+", " ", str(alt_text or "").strip())[:500]

        if not img.startswith(("http://", "https://")):
            raise PinterestApiError("image_url must be a public http(s) URL")
        if not lnk.startswith(("http://", "https://")):
            raise PinterestApiError("link must be a public http(s) URL")

        payload: dict[str, Any] = {
            "board_id": bid,
            "title": t,
            "description": d,
            "link": lnk,
            "alt_text": alt,
            "media_source": {
                "source_type": "image_url",
                "url": img,
            },
        }

        url = f"{self._api_base}/pins"
        async with httpx.AsyncClient(timeout=45.0, headers=self._headers()) as client:
            r = await client.post(url, json=payload)

        if r.status_code not in (200, 201):
            raise PinterestApiError(f"Create pin failed: HTTP {r.status_code}: {r.text[:400]}")

        data = r.json()
        if isinstance(data, dict):
            return data
        return {"raw": data}
