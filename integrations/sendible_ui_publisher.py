"""
⚠️ DEPRECATED — Use publer_publisher.py instead.

Sendible ($199/mo) has been replaced by Publer (~$10/mo per account).
This file is kept for reference only. It will be removed in v5.0.

Browser automation is NO LONGER NEEDED — Publer provides a proper REST API
at $10/mo per account vs Sendible's $199/mo.

Migration: python setup_publer.py

---

ViralOps Engine — Sendible UI Automation Publisher (LEGACY — DEPRECATED)

Browser automation via Playwright with FULL anti-detect:
  - playwright-stealth: hide webdriver flags, fake navigator properties
  - Real Chrome user-agent (rotated)
  - Random delays 2-5s between actions
  - Natural mouse movement curves (Bezier)
  - Residential proxy rotation support
  - Non-headless mode (real browser window)
  - Cookie persistence across sessions
  - Human-like typing speed variation

Auth: Sendible email + password → browser login → session cookies.
NO Developer API needed. NO Application ID needed.

SETUP (.env):
  SENDIBLE_EMAIL=your-sendible-login-email
  SENDIBLE_PASSWORD=your-sendible-password
  SENDIBLE_PROXY=socks5://user:pass@proxy:port  (optional)
  SENDIBLE_HEADLESS=false                        (default: false)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("viralops.publisher.sendible_ui")

# ── User-Agent rotation pool (real Chrome on Windows) ──
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.6613.120 Safari/537.36",
]

# ── Cookie/session storage ──
COOKIES_DIR = Path(__file__).parent.parent / ".sendible_session"


def _random_delay(min_s: float = 2.0, max_s: float = 5.0) -> float:
    """Generate random delay with slight normal distribution bias."""
    mean = (min_s + max_s) / 2
    std = (max_s - min_s) / 4
    delay = random.gauss(mean, std)
    return max(min_s, min(max_s, delay))


def _bezier_points(
    start: tuple[float, float],
    end: tuple[float, float],
    steps: int = 20,
) -> list[tuple[float, float]]:
    """
    Generate natural mouse movement curve using cubic Bezier.
    Two random control points create human-like movement.
    """
    x0, y0 = start
    x3, y3 = end

    # Random control points (offset from straight line)
    dx = x3 - x0
    dy = y3 - y0
    dist = math.sqrt(dx * dx + dy * dy)
    offset = max(30, dist * 0.3)

    x1 = x0 + dx * random.uniform(0.2, 0.4) + random.uniform(-offset, offset)
    y1 = y0 + dy * random.uniform(0.2, 0.4) + random.uniform(-offset, offset)
    x2 = x0 + dx * random.uniform(0.6, 0.8) + random.uniform(-offset, offset)
    y2 = y0 + dy * random.uniform(0.6, 0.8) + random.uniform(-offset, offset)

    points = []
    for i in range(steps + 1):
        t = i / steps
        mt = 1 - t
        x = mt**3 * x0 + 3 * mt**2 * t * x1 + 3 * mt * t**2 * x2 + t**3 * x3
        y = mt**3 * y0 + 3 * mt**2 * t * y1 + 3 * mt * t**2 * y2 + t**3 * y3
        points.append((x, y))

    return points


async def _human_move_to(page, selector: str):
    """Move mouse to element using natural Bezier curve."""
    try:
        element = await page.query_selector(selector)
        if not element:
            return
        box = await element.bounding_box()
        if not box:
            return

        # Target: random point within element
        target_x = box["x"] + random.uniform(box["width"] * 0.2, box["width"] * 0.8)
        target_y = box["y"] + random.uniform(box["height"] * 0.2, box["height"] * 0.8)

        # Current mouse position (approximate from viewport center)
        vp = page.viewport_size or {"width": 1280, "height": 720}
        current_x = random.uniform(vp["width"] * 0.3, vp["width"] * 0.7)
        current_y = random.uniform(vp["height"] * 0.3, vp["height"] * 0.7)

        points = _bezier_points((current_x, current_y), (target_x, target_y))

        for px, py in points:
            await page.mouse.move(px, py)
            await asyncio.sleep(random.uniform(0.005, 0.02))

    except Exception as e:
        logger.debug("Mouse move failed (non-critical): %s", e)


async def _human_type(page, selector: str, text: str):
    """Type text with human-like speed variation."""
    try:
        await page.click(selector)
        await asyncio.sleep(random.uniform(0.3, 0.8))

        for char in text:
            await page.keyboard.type(char, delay=random.uniform(30, 120))
            # Occasional pause (like thinking)
            if random.random() < 0.05:
                await asyncio.sleep(random.uniform(0.3, 1.0))
    except Exception as e:
        # Fallback: direct fill
        logger.debug("Human type failed, using fill: %s", e)
        await page.fill(selector, text)


class SendibleUIPublisher:
    """
    Sendible browser UI automation with full anti-detect stealth.

    Uses Playwright + stealth plugin to automate Sendible's web UI:
    - Login with email/password
    - Compose and publish posts
    - Select social profiles/services
    - Schedule posts
    - Upload media

    Anti-detect features:
    - playwright-stealth hides webdriver flags
    - Real Chrome user-agent (rotated)
    - Random delays 2-5s between actions
    - Natural mouse curves (Bezier)
    - Residential proxy support
    - Non-headless (real browser window)
    - Session persistence (cookies saved)
    """

    platform = "sendible"
    BASE_URL = "https://app.sendible.com"

    def __init__(self, account_id: str = "sendible_main"):
        self.account_id = account_id
        self.email: str = os.environ.get("SENDIBLE_EMAIL", "")
        self.password: str = os.environ.get("SENDIBLE_PASSWORD", "")
        self.proxy: str = os.environ.get("SENDIBLE_PROXY", "")
        self.headless: bool = os.environ.get("SENDIBLE_HEADLESS", "false").lower() == "true"

        self._browser = None
        self._context = None
        self._page = None
        self._logged_in = False
        self._services: list[dict] = []
        self._playwright = None

    @property
    def is_configured(self) -> bool:
        """Check if login credentials are set."""
        return bool(self.email and self.password)

    async def _launch_browser(self):
        """Launch stealth browser with anti-detect measures."""
        if self._browser:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "playwright required. Install:\n"
                "  pip install playwright playwright-stealth\n"
                "  playwright install chromium"
            )

        self._playwright = await async_playwright().__aenter__()

        # ── Browser launch args (anti-detect) ──
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
            "--window-size=1366,768",
        ]

        # ── Proxy config ──
        proxy_config = None
        if self.proxy:
            proxy_parts = self.proxy.replace("socks5://", "").replace("http://", "").replace("https://", "")
            proxy_config = {"server": self.proxy}
            if "@" in proxy_parts:
                user_pass, host = proxy_parts.rsplit("@", 1)
                if ":" in user_pass:
                    proxy_config["username"] = user_pass.split(":")[0]
                    proxy_config["password"] = user_pass.split(":", 1)[1]

        # ── Launch browser ──
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=launch_args,
            slow_mo=random.randint(50, 150),  # Natural slowdown
        )

        # ── User agent ──
        user_agent = random.choice(USER_AGENTS)

        # ── Persistent context with cookies ──
        COOKIES_DIR.mkdir(parents=True, exist_ok=True)
        cookie_file = COOKIES_DIR / f"{self.account_id}_cookies.json"

        context_opts = {
            "viewport": {"width": 1366, "height": 768},
            "user_agent": user_agent,
            "locale": "en-US",
            "timezone_id": "America/New_York",
            "color_scheme": "light",
            "java_script_enabled": True,
            "ignore_https_errors": True,
        }

        if proxy_config:
            context_opts["proxy"] = proxy_config

        self._context = await self._browser.new_context(**context_opts)

        # ── Apply stealth ──
        # NOTE: playwright-stealth's stealth_async() injects scripts with
        # broken references ("utils is not defined", "opts is not defined")
        # that prevent SPA frameworks (like Sendible's) from loading.
        # We use manual anti-detect scripts below instead — they handle the
        # critical detection vectors without breaking page JS.
        use_stealth_plugin = os.environ.get(
            "SENDIBLE_USE_STEALTH_PLUGIN", "false"
        ).lower() == "true"
        if use_stealth_plugin:
            try:
                from playwright_stealth import stealth_async
                await stealth_async(self._context)
                logger.info("Sendible UI: Stealth plugin applied")
            except ImportError:
                logger.warning(
                    "playwright-stealth not installed — using manual anti-detect"
                )
        else:
            logger.info(
                "Sendible UI: Using manual anti-detect (stealth plugin disabled)"
            )

        # ── Restore cookies ──
        if cookie_file.exists():
            try:
                cookies = json.loads(cookie_file.read_text(encoding="utf-8"))
                await self._context.add_cookies(cookies)
                logger.info("Sendible UI: Restored %d session cookies", len(cookies))
            except Exception as e:
                logger.debug("Cookie restore failed: %s", e)

        self._page = await self._context.new_page()

        # ── Enhanced anti-detect JS (replaces broken stealth plugin) ──
        await self._page.add_init_script("""
            // 1. Navigator.webdriver
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            try { delete navigator.__proto__.webdriver; } catch(e) {}

            // 2. Navigator.plugins (realistic)
            Object.defineProperty(navigator, 'plugins', {
                get: () => {
                    const p = [
                        {name:'Chrome PDF Plugin', filename:'internal-pdf-viewer',
                         description:'Portable Document Format'},
                        {name:'Chrome PDF Viewer',
                         filename:'mhjfbmdgcfjbbpaeojofohoefgiehjai', description:''},
                        {name:'Native Client', filename:'internal-nacl-plugin',
                         description:''},
                    ];
                    p.length = 3;
                    return p;
                }
            });

            // 3. Navigator.languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });

            // 4. Chrome runtime + loadTimes + csi
            if (!window.chrome) window.chrome = {};
            window.chrome.runtime = {
                connect: function(){}, sendMessage: function(){},
                onMessage: {addListener: function(){}}
            };
            window.chrome.loadTimes = function() {
                var n = Date.now()/1000;
                return {requestTime:n-3, startLoadTime:n-2, commitLoadTime:n-1,
                        finishDocumentLoadTime:n-0.5, finishLoadTime:n,
                        firstPaintTime:n-1.5, firstPaintAfterLoadTime:0,
                        navigationType:'Other', wasFetchedViaSpdy:false,
                        wasNpnNegotiated:true, npnNegotiatedProtocol:'h2',
                        wasAlternateProtocolAvailable:false, connectionInfo:'h2'};
            };
            window.chrome.csi = function() {
                return {startE:Date.now(), onloadT:Date.now(),
                        pageT:Math.random()*1000+500, tran:15};
            };

            // 5. Permissions query
            var origQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = function(params) {
                if (params.name === 'notifications')
                    return Promise.resolve({state: Notification.permission});
                return origQuery(params);
            };

            // 6. WebGL vendor/renderer
            var origGetParam = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(p) {
                if (p === 37445) return 'Intel Inc.';
                if (p === 37446) return 'Intel Iris OpenGL Engine';
                return origGetParam.call(this, p);
            };

            // 7. Canvas noise
            var origToDataURL = HTMLCanvasElement.prototype.toDataURL;
            HTMLCanvasElement.prototype.toDataURL = function(type) {
                var c = this.getContext('2d');
                if (c) {
                    var s = c.fillStyle;
                    c.fillStyle = 'rgba(0,0,0,0.01)';
                    c.fillRect(0, 0, 1, 1);
                    c.fillStyle = s;
                }
                return origToDataURL.apply(this, arguments);
            };
        """)

        logger.info(
            "Sendible UI: Browser launched (headless=%s, proxy=%s, ua=%s...)",
            self.headless,
            bool(self.proxy),
            user_agent[:40],
        )

    async def _save_cookies(self):
        """Save session cookies for next time."""
        if not self._context:
            return
        try:
            cookies = await self._context.cookies()
            COOKIES_DIR.mkdir(parents=True, exist_ok=True)
            cookie_file = COOKIES_DIR / f"{self.account_id}_cookies.json"
            cookie_file.write_text(json.dumps(cookies, indent=2), encoding="utf-8")
            logger.debug("Saved %d cookies", len(cookies))
        except Exception as e:
            logger.debug("Cookie save failed: %s", e)

    async def _random_wait(self, min_s: float = 2.0, max_s: float = 5.0):
        """Wait random time to appear human."""
        delay = _random_delay(min_s, max_s)
        await asyncio.sleep(delay)

    # ──────────────────────────────────────────────
    # Login
    # ──────────────────────────────────────────────

    async def login(self) -> bool:
        """Login to Sendible with anti-detect measures."""
        if not self.is_configured:
            logger.error("Sendible UI: SENDIBLE_EMAIL and SENDIBLE_PASSWORD not set!")
            return False

        await self._launch_browser()

        try:
            page = self._page

            # Check if already logged in (try dashboard)
            await page.goto(
                f"{self.BASE_URL}/dashboard",
                wait_until="networkidle",
                timeout=30000,
            )
            await self._random_wait(3, 5)

            # Sendible SPA uses hash routing — URL might stay as /#login
            # even on the dashboard.  Detect login state by page content.
            if await self._is_dashboard_visible(page):
                logger.info("Sendible UI: Already logged in (session cookies)")
                self._logged_in = True
                await self._dismiss_popups(page)
                await self._save_cookies()
                return True

            # Navigate to login
            logger.info("Sendible UI: Navigating to login page...")
            await page.goto(
                f"{self.BASE_URL}/login",
                wait_until="networkidle",
                timeout=30000,
            )
            await self._random_wait(2, 4)

            # Wait for login form — use Sendible-specific selectors first
            await page.wait_for_selector(
                '#login-inputEmail, input[name="email"], input[type="email"]',
                timeout=15000,
            )

            # ── Human-like login ──
            email_sel = '#login-inputEmail, input[name="email"], input[type="email"]'
            await _human_move_to(page, email_sel)
            await self._random_wait(0.5, 1.5)
            await _human_type(page, email_sel, self.email)
            await self._random_wait(1, 3)

            pass_sel = '#login-inputPassword, input[type="password"], input[name="password"]'
            await _human_move_to(page, pass_sel)
            await self._random_wait(0.5, 1.5)
            await _human_type(page, pass_sel, self.password)
            await self._random_wait(1, 3)

            # Click login button
            login_btn = (
                '#login-submit, input[type="submit"], button[type="submit"], '
                '.login-button, button:has-text("Log in"), '
                'button:has-text("Sign in")'
            )
            await _human_move_to(page, login_btn)
            await self._random_wait(0.5, 1.0)
            await page.click(login_btn)

            # Wait for dashboard content to appear
            logger.info("Sendible UI: Waiting for dashboard content...")
            await self._random_wait(3, 6)

            # ── Verify login by checking page content ──
            # (Sendible SPA may not change the URL hash on login)
            for attempt in range(6):  # retry up to ~15s
                if await self._is_dashboard_visible(page):
                    self._logged_in = True
                    logger.info("Sendible UI: Login successful!")
                    await self._dismiss_popups(page)
                    await self._save_cookies()
                    return True
                await asyncio.sleep(2.5)

            # Check for error message
            try:
                error_el = await page.query_selector(
                    ".error-message, .alert-danger, .login-error, "
                    ".text-danger, .form-error"
                )
                if error_el:
                    error_text = await error_el.text_content()
                    logger.error("Sendible UI: Login error: %s", error_text)
                    return False
            except Exception:
                pass

            logger.error("Sendible UI: Login failed — ended at %s", page.url)
            return False

        except Exception as e:
            logger.error("Sendible UI: Login exception: %s", e)
            return False

    async def _is_dashboard_visible(self, page) -> bool:
        """Detect if the Sendible dashboard is visible (content-based)."""
        try:
            return await page.evaluate("""
                () => {
                    const text = (document.body && document.body.innerText) || '';
                    // Dashboard nav items that only appear when logged in
                    const markers = ['Compose', 'Calendar', 'Campaigns',
                                     'Publish', 'Activity', 'Reports'];
                    let hits = 0;
                    for (const m of markers) {
                        if (text.includes(m)) hits++;
                    }
                    return hits >= 3;
                }
            """)
        except Exception:
            return False

    async def _dismiss_popups(self, page):
        """Dismiss trial banners, modals, or upgrade prompts."""
        try:
            # Common close/dismiss selectors
            dismiss_sels = [
                'button[aria-label="Close"], .modal .close, .dismiss-btn',
                'button:has-text("Maybe later"), button:has-text("Not now")',
                'button:has-text("Dismiss"), button:has-text("Skip")',
                '.trial-banner .close, .upgrade-banner .close',
            ]
            for sel in dismiss_sels:
                el = await page.query_selector(sel)
                if el and await el.is_visible():
                    await el.click()
                    await asyncio.sleep(0.5)
                    logger.info("Sendible UI: Dismissed popup: %s", sel[:40])
        except Exception:
            pass

    # ──────────────────────────────────────────────
    # Connection / Services
    # ──────────────────────────────────────────────

    async def connect(self) -> bool:
        """Connect = login + verify session."""
        if self._logged_in:
            return True
        return await self.login()

    async def get_services(self, force: bool = False) -> list[dict]:
        """
        Get connected social services from Sendible UI.
        Navigates to settings/services page and scrapes.
        """
        if self._services and not force:
            return self._services

        if not self._logged_in:
            await self.connect()
        if not self._logged_in:
            return []

        try:
            page = self._page
            await page.goto(
                f"{self.BASE_URL}/settings/services",
                wait_until="domcontentloaded",
            )
            await self._random_wait(3, 5)

            # Scrape service cards from the services page
            services = await page.evaluate("""
                () => {
                    const items = [];
                    // Try various selectors for service items
                    const cards = document.querySelectorAll(
                        '.service-item, .profile-card, .connected-service, ' +
                        '[data-service], .social-profile, tr[data-id]'
                    );
                    cards.forEach((card, i) => {
                        const name = card.querySelector(
                            '.service-name, .profile-name, .name, h4, h5, .title'
                        );
                        const type = card.querySelector(
                            '.service-type, .type, .network, .badge'
                        );
                        const id = card.getAttribute('data-id') ||
                                   card.getAttribute('data-service-id') ||
                                   String(i);
                        items.push({
                            id: id,
                            name: name ? name.textContent.trim() : 'Service ' + i,
                            service_type: type ? type.textContent.trim() : 'unknown',
                        });
                    });
                    return items;
                }
            """)

            self._services = services or []
            logger.info("Sendible UI: Found %d connected services", len(self._services))
            return self._services

        except Exception as e:
            logger.error("Sendible UI: get_services error: %s", e)
            return []

    # ──────────────────────────────────────────────
    # Publish
    # ──────────────────────────────────────────────

    async def publish(self, content: dict) -> dict:
        """
        Publish content via Sendible's compose UI modal.

        Proven workflow (tested against Sendible SPA):
        1. Ensure on dashboard
        2. Close any leftover compose modal
        3. Click #btnCompose to open fresh compose modal
        4. Select service/profile via Select2 dropdown
        5. Type message via TinyMCE JS focus + keyboard
        6. Upload media (optional)
        7. Force-enable Send button and click
        8. Monitor API response for success

        content dict keys:
          caption (str): Post text
          title (str, optional): Title for blog platforms
          media_path (str, optional): Local file path for image/video
          media_url (str, optional): URL to download and attach
          platforms (list[str], optional): e.g. ["tiktok", "pinterest"]
          schedule_at (str, optional): Schedule datetime
          hashtags (list[str], optional): Append hashtags
        """
        if not self._logged_in:
            ok = await self.connect()
            if not ok:
                return {
                    "success": False,
                    "error": "Not logged in — check SENDIBLE_EMAIL/PASSWORD",
                    "platform": self.platform,
                }

        try:
            page = self._page

            # ── 1. Ensure on dashboard ──
            if not await self._is_dashboard_visible(page):
                await page.goto(
                    f"{self.BASE_URL}/dashboard",
                    wait_until="networkidle",
                    timeout=30000,
                )
                await self._random_wait(3, 5)
                await self._dismiss_popups(page)

            # ── 2. Close any leftover compose modal ──
            await self._close_compose_modal(page)
            await self._random_wait(0.5, 1)

            # ── 3. Open compose modal ──
            logger.info("Sendible UI: Opening compose box...")
            compose_btn = await page.query_selector("#btnCompose")
            if not compose_btn:
                compose_btn = await page.query_selector("a:has-text('Compose')")
            if not compose_btn:
                return {
                    "success": False,
                    "error": "Compose button not found on dashboard",
                    "platform": self.platform,
                }

            await compose_btn.click()
            await self._random_wait(2, 3)

            # Wait for compose modal + TinyMCE iframe
            try:
                await page.wait_for_selector(
                    "#compose-message_ifr",
                    state="attached",
                    timeout=10000,
                )
            except Exception:
                logger.warning("Sendible UI: Compose modal did not appear")
                return {
                    "success": False,
                    "error": "Compose modal did not open",
                    "platform": self.platform,
                }

            await self._random_wait(1, 2)

            # ── 4. Select service/profile via Select2 ──
            platforms = content.get("platforms", [])
            platform_name = platforms[0] if platforms else ""
            if platform_name:
                selected = await self._select_service(page, platform_name)
                if not selected:
                    logger.warning(
                        "Sendible UI: Could not select '%s' profile", platform_name
                    )
                    return {
                        "success": False,
                        "error": f"No '{platform_name}' profile found in Sendible",
                        "platform": self.platform,
                    }
                await self._random_wait(1, 2)

            # ── 5. Build message text ──
            caption = content.get("caption", "")
            hashtags = content.get("hashtags", [])
            if hashtags:
                tag_str = " ".join(
                    f"#{t}" if not t.startswith("#") else t for t in hashtags
                )
                caption = f"{caption}\n\n{tag_str}".strip()

            # ── 6. Type in TinyMCE editor ──
            logger.info("Sendible UI: Typing message in TinyMCE editor...")
            typed = await self._type_in_tinymce(page, caption)
            if not typed:
                return {
                    "success": False,
                    "error": "Failed to type in compose editor",
                    "platform": self.platform,
                }
            await self._random_wait(1, 2)

            # ── 7. Upload media (optional) ──
            media_path = content.get("media_path", "")
            if media_path and Path(media_path).exists():
                logger.info("Sendible UI: Uploading media: %s", media_path)
                await self._upload_media(page, media_path)
                await self._random_wait(3, 5)

            # ── 8. Force-enable Send + click (Backbone validation
            #       doesn't sync with TinyMCE keyboard input, but
            #       the API accepts the message — verified working) ──
            logger.info("Sendible UI: Sending post...")
            send_btn = await page.query_selector("#compose-send")
            if not send_btn:
                return {
                    "success": False,
                    "error": "Send button not found",
                    "platform": self.platform,
                }

            # Force-enable (Sendible's Backbone model validation
            # doesn't detect keyboard-typed TinyMCE content, but
            # the API endpoint accepts the post correctly)
            await page.evaluate("""() => {
                const btn = document.querySelector('#compose-send');
                if (btn) {
                    btn.disabled = false;
                    btn.removeAttribute('disabled');
                }
            }""")
            await self._random_wait(0.3, 0.6)

            # Set up API response monitoring
            api_result = {"status": None, "message_id": None, "error": None}

            async def on_api_response(response):
                url = response.url
                if "api.sendible.com" in url and "/api/message" in url:
                    try:
                        if response.status == 200:
                            data = await response.json()
                            api_result["status"] = "success"
                            api_result["message_id"] = data.get("message_id")
                            api_result["response"] = str(data.get("status", ""))
                        else:
                            body_text = await response.text()
                            api_result["status"] = "error"
                            api_result["error"] = f"HTTP {response.status}: {body_text[:200]}"
                    except Exception:
                        pass

            page.on("response", on_api_response)

            try:
                await send_btn.click()
                await self._random_wait(4, 7)
            finally:
                page.remove_listener("response", on_api_response)

            # ── 9. Check result ──
            # Prefer API response monitoring (most reliable)
            if api_result["status"] == "success":
                msg_id = api_result["message_id"] or f"ui_{int(time.time())}"
                logger.info(
                    "Sendible UI: Published! message_id=%s status=%s",
                    msg_id,
                    api_result.get("response", ""),
                )
                await self._save_cookies()
                return {
                    "success": True,
                    "post_id": str(msg_id),
                    "post_url": "",
                    "platform": self.platform,
                    "method": "ui_automation",
                    "message": f"Queued (id={msg_id})",
                }

            if api_result["status"] == "error":
                logger.warning("Sendible UI: API error: %s", api_result["error"])
                return {
                    "success": False,
                    "error": api_result["error"],
                    "platform": self.platform,
                }

            # Fallback: check DOM for success/error indicators
            try:
                success_el = await page.query_selector(
                    '.alert-success, .toast-success, .noty_body'
                )
                if success_el:
                    success_text = await success_el.text_content() or "Published"
                    logger.info("Sendible UI: Published! %s", success_text.strip())
                    await self._save_cookies()
                    return {
                        "success": True,
                        "post_id": f"ui_{int(time.time())}",
                        "post_url": "",
                        "platform": self.platform,
                        "method": "ui_automation",
                        "message": success_text.strip(),
                    }
            except Exception:
                pass

            try:
                error_el = await page.query_selector(
                    '.alert-danger, .noty_type__error .noty_body'
                )
                if error_el:
                    error_text = await error_el.text_content() or "Unknown error"
                    logger.warning("Sendible UI: Post error: %s", error_text.strip())
                    return {
                        "success": False,
                        "error": error_text.strip(),
                        "platform": self.platform,
                    }
            except Exception:
                pass

            # No explicit signal — assume sent
            logger.info("Sendible UI: Post sent (no explicit confirmation)")
            await self._save_cookies()
            return {
                "success": True,
                "post_id": f"ui_{int(time.time())}",
                "post_url": "",
                "platform": self.platform,
                "method": "ui_automation",
            }

        except Exception as e:
            logger.error("Sendible UI: Publish error: %s", e)
            return {
                "success": False,
                "error": str(e),
                "platform": self.platform,
            }

    # ── Compose helpers ──────────────────────────

    async def _select_service(self, page, platform: str) -> bool:
        """Select a social profile in Sendible's compose Select2 dropdown.

        Opens the Select2 dropdown for #compose-service-selector, finds the
        item matching *platform* (e.g. 'tiktok', 'facebook', 'pinterest'),
        and clicks it.  Returns True if a profile was successfully selected.
        """
        try:
            # Open Select2 dropdown by clicking the search input
            s2_input = page.locator("#s2id_compose-service-selector input").first
            await s2_input.click()
            await asyncio.sleep(1.5)

            # Use has_text locator to find matching platform in active dropdown
            # (more reliable than iterating with .nth() which can time out)
            item = page.locator(
                ".select2-drop-active .select2-results li",
                has_text=re.compile(platform, re.IGNORECASE),
            ).first
            try:
                await item.wait_for(state="visible", timeout=5000)
                item_text = await item.inner_text()
                await item.click()
                await asyncio.sleep(1)
                logger.info(
                    "Sendible UI: Selected service '%s'",
                    item_text.strip().replace("\n", " "),
                )
                return True
            except Exception:
                pass

            # Close dropdown if nothing matched
            await page.keyboard.press("Escape")
            logger.warning(
                "Sendible UI: No '%s' profile found in Select2 dropdown",
                platform,
            )
            return False
        except Exception as e:
            logger.warning("Sendible UI: Service selection error: %s", e)
            return False

    async def _close_compose_modal(self, page):
        """Close the compose modal if it's currently open."""
        try:
            compose_box = await page.query_selector("#compose-box")
            if not compose_box:
                return
            # Check if visible (has 'in' class and aria-hidden=false)
            aria = await compose_box.get_attribute("aria-hidden")
            classes = await compose_box.get_attribute("class") or ""
            if aria == "false" or "in" in classes.split():
                # Try close button or keyboard escape
                close_btn = await page.query_selector(
                    "#compose-box .close, #compose-box [data-dismiss='modal']"
                )
                if close_btn and await close_btn.is_visible():
                    await close_btn.click()
                    await asyncio.sleep(1)
                else:
                    await page.keyboard.press("Escape")
                    await asyncio.sleep(1)
                # Clear Select2 selections for fresh state
                await page.evaluate("""() => {
                    try {
                        const $sel = $('#compose-service-selector');
                        if ($sel.length) $sel.select2('val', '');
                    } catch(e) {}
                }""")
                logger.debug("Sendible UI: Closed leftover compose modal")

            # Also remove any leftover modal backdrop that blocks clicks
            await page.evaluate("""() => {
                document.querySelectorAll('.modal-backdrop').forEach(el => el.remove());
            }""")
        except Exception:
            pass

    async def _type_in_tinymce(self, page, text: str) -> bool:
        """Type text into Sendible's TinyMCE compose editor.

        Uses TinyMCE JS API to focus the editor and remove the placeholder,
        then types via Playwright keyboard events so that Backbone picks up
        the input.
        """
        try:
            # Step 1: Focus editor via TinyMCE API and remove placeholder
            focused = await page.evaluate("""() => {
                try {
                    const editor = tinymce.get('compose-message');
                    if (!editor) return false;
                    editor.focus();
                    // Remove placeholder that intercepts clicks
                    const body = editor.getBody();
                    const placeholder = body.querySelector('.placeholder-text');
                    if (placeholder) placeholder.remove();
                    // Place cursor at start
                    editor.selection.setCursorLocation(body, 0);
                    return true;
                } catch(e) { return false; }
            }""")

            if focused:
                # Step 2: Type via keyboard (triggers proper DOM events)
                await asyncio.sleep(0.3)
                for char in text:
                    if char == "\n":
                        await page.keyboard.press("Enter")
                    else:
                        await page.keyboard.type(
                            char, delay=random.uniform(8, 25)
                        )
                logger.info("Sendible UI: Typed via TinyMCE focus + keyboard")
                return True
        except Exception as e:
            logger.debug("TinyMCE focus+keyboard failed: %s", e)

        try:
            # Fallback: Set content via TinyMCE API directly
            typed = await page.evaluate("""
                (text) => {
                    if (typeof tinymce !== 'undefined' && tinymce.editors.length > 0) {
                        const editor = tinymce.get('compose-message') || tinymce.editors[0];
                        editor.setContent('<p>' + text.replace(/\\n/g, '</p><p>') + '</p>');
                        editor.fire('change');
                        return true;
                    }
                    return false;
                }
            """, text)
            if typed:
                logger.info("Sendible UI: Typed via TinyMCE API (fallback)")
                return True
        except Exception as e:
            logger.debug("TinyMCE API fallback failed: %s", e)

        try:
            # Last resort: set hidden textarea directly
            textarea = await page.query_selector("#compose-message")
            if textarea:
                await textarea.evaluate("(el, t) => el.value = t", text)
                logger.info("Sendible UI: Set via textarea value (last resort)")
                return True
        except Exception as e:
            logger.debug("Textarea fallback failed: %s", e)

        return False

    async def _upload_media(self, page, media_path: str):
        """Upload media file in Sendible compose box."""
        try:
            # Try clicking attachment button first
            attach_btn = await page.query_selector(
                "#compose-attachment, button.media-library-attach"
            )
            if attach_btn and await attach_btn.is_visible():
                await attach_btn.click()
                await self._random_wait(1, 2)

            # Find file input and upload
            file_input = await page.query_selector('input[type="file"]')
            if file_input:
                await file_input.set_input_files(media_path)
                logger.info("Sendible UI: Media file set: %s", media_path)
                await self._random_wait(3, 5)
            else:
                logger.warning("Sendible UI: No file input found for upload")
        except Exception as e:
            logger.warning("Sendible UI: Media upload failed: %s", e)

    async def schedule(self, content: dict) -> dict:
        """Schedule content for future posting."""
        if "schedule_at" not in content:
            return {
                "success": False,
                "error": "schedule_at is required",
                "platform": self.platform,
            }
        return await self.publish(content)

    # ──────────────────────────────────────────────
    # Messages (read from sent tab)
    # ──────────────────────────────────────────────

    async def get_messages(
        self, status: str = "", per_page: int = 20, page: int = 1
    ) -> list[dict]:
        """Get messages from Sendible outbox/sent UI."""
        if not self._logged_in:
            await self.connect()
        if not self._logged_in:
            return []

        try:
            pg = self._page
            await pg.goto(
                f"{self.BASE_URL}/outbox",
                wait_until="domcontentloaded",
            )
            await self._random_wait(3, 5)

            messages = await pg.evaluate("""
                () => {
                    const items = [];
                    const rows = document.querySelectorAll(
                        '.message-item, .outbox-item, .post-item, ' +
                        'tr.message-row, .scheduled-post'
                    );
                    rows.forEach((row, i) => {
                        const text = row.querySelector(
                            '.message-text, .post-text, .content, p'
                        );
                        const date = row.querySelector(
                            '.date, .timestamp, time, .scheduled-date'
                        );
                        const status = row.querySelector(
                            '.status, .badge, .state'
                        );
                        items.push({
                            id: row.getAttribute('data-id') || String(i),
                            message_text: text ? text.textContent.trim() : '',
                            date: date ? date.textContent.trim() : '',
                            status: status ? status.textContent.trim() : 'unknown',
                        });
                    });
                    return items;
                }
            """)

            return messages or []

        except Exception as e:
            logger.error("Sendible UI: get_messages error: %s", e)
            return []

    # ──────────────────────────────────────────────
    # Account info
    # ──────────────────────────────────────────────

    async def get_account_details(self) -> dict:
        """Get account info from settings page."""
        if not self._logged_in:
            await self.connect()
        if not self._logged_in:
            return {}

        return {
            "email": self.email,
            "method": "ui_automation",
            "logged_in": self._logged_in,
            "services_count": len(self._services),
        }

    # ──────────────────────────────────────────────
    # Test
    # ──────────────────────────────────────────────

    async def test_connection(self) -> dict:
        """Test if Sendible login works."""
        try:
            connected = await self.connect()
            if connected:
                services = await self.get_services()
                return {
                    "connected": True,
                    "method": "ui_automation",
                    "services_count": len(services),
                    "services": services[:10],
                    "account": self.email,
                }
            return {"connected": False, "error": "Login failed"}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    # ──────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────

    async def close(self):
        """Close browser and save cookies."""
        if self._page:
            await self._save_cookies()
        if self._context:
            try:
                await self._context.close()
            except Exception:
                pass
            self._context = None
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright:
            try:
                await self._playwright.__aexit__(None, None, None)
            except Exception:
                pass
            self._playwright = None
        self._page = None
        self._logged_in = False
