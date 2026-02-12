"""
ViralOps Engine — Sendible UI Automation Publisher (Stealth Mode)

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
        try:
            from playwright_stealth import stealth_async
            await stealth_async(self._context)
            logger.info("Sendible UI: Stealth plugin applied")
        except ImportError:
            logger.warning(
                "playwright-stealth not installed. Running without stealth. "
                "Install: pip install playwright-stealth"
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

        # ── Extra anti-detect JS ──
        await self._page.add_init_script("""
            // Override navigator.webdriver
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});

            // Override navigator.plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });

            // Override navigator.languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });

            // Override chrome runtime
            window.chrome = { runtime: {} };

            // Override permissions query
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) =>
                parameters.name === 'notifications'
                    ? Promise.resolve({ state: Notification.permission })
                    : originalQuery(parameters);
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
            await page.goto(f"{self.BASE_URL}/dashboard", wait_until="domcontentloaded")
            await self._random_wait(2, 4)

            # If we're on dashboard, already logged in
            if "/dashboard" in page.url or "/compose" in page.url:
                logger.info("Sendible UI: Already logged in (session cookies)")
                self._logged_in = True
                await self._save_cookies()
                return True

            # Navigate to login
            logger.info("Sendible UI: Navigating to login page...")
            await page.goto(f"{self.BASE_URL}/login", wait_until="domcontentloaded")
            await self._random_wait(2, 4)

            # Wait for login form
            await page.wait_for_selector(
                'input[type="email"], input[name="email"], #email, input[type="text"]',
                timeout=15000,
            )

            # ── Human-like login ──
            # Move to email field
            email_sel = 'input[type="email"], input[name="email"], #email, input[type="text"]'
            await _human_move_to(page, email_sel)
            await self._random_wait(0.5, 1.5)

            # Type email
            await _human_type(page, email_sel, self.email)
            await self._random_wait(1, 3)

            # Move to password field
            pass_sel = 'input[type="password"], input[name="password"], #password'
            await _human_move_to(page, pass_sel)
            await self._random_wait(0.5, 1.5)

            # Type password
            await _human_type(page, pass_sel, self.password)
            await self._random_wait(1, 3)

            # Click login button
            login_btn = 'button[type="submit"], input[type="submit"], .login-button, button:has-text("Log in"), button:has-text("Sign in")'
            await _human_move_to(page, login_btn)
            await self._random_wait(0.5, 1.0)
            await page.click(login_btn)

            # Wait for navigation
            logger.info("Sendible UI: Waiting for login redirect...")
            await self._random_wait(3, 6)

            # Check if login successful
            try:
                await page.wait_for_url("**/dashboard**", timeout=20000)
                self._logged_in = True
                logger.info("Sendible UI: Login successful!")
                await self._save_cookies()
                return True
            except Exception:
                pass

            # Fallback: check if we're on any authenticated page
            current_url = page.url
            if any(x in current_url for x in ["/dashboard", "/compose", "/calendar", "/settings"]):
                self._logged_in = True
                logger.info("Sendible UI: Login successful (detected auth page)!")
                await self._save_cookies()
                return True

            # Check for error message
            try:
                error_el = await page.query_selector(".error-message, .alert-danger, .login-error")
                if error_el:
                    error_text = await error_el.text_content()
                    logger.error("Sendible UI: Login error: %s", error_text)
                    return False
            except Exception:
                pass

            logger.error("Sendible UI: Login failed — ended at %s", current_url)
            return False

        except Exception as e:
            logger.error("Sendible UI: Login exception: %s", e)
            return False

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
        Publish content via Sendible's compose UI.

        content dict keys:
          caption (str): Post text
          title (str, optional): Title for blog platforms
          media_path (str, optional): Local file path for image/video
          media_url (str, optional): URL to download and attach
          platforms (list[str], optional): Filter platforms in compose box
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

            # Navigate to compose
            logger.info("Sendible UI: Navigating to compose box...")
            await page.goto(
                f"{self.BASE_URL}/compose",
                wait_until="domcontentloaded",
            )
            await self._random_wait(3, 5)

            # ── Build message text ──
            caption = content.get("caption", "")
            hashtags = content.get("hashtags", [])
            if hashtags:
                tag_str = " ".join(
                    f"#{t}" if not t.startswith("#") else t for t in hashtags
                )
                caption = f"{caption}\n\n{tag_str}".strip()

            # ── Type message ──
            compose_sel = (
                'textarea[name="message"], .compose-textarea, '
                '#message-text, .ql-editor, [contenteditable="true"], '
                'textarea.form-control, div[role="textbox"]'
            )
            await page.wait_for_selector(compose_sel, timeout=15000)
            await _human_move_to(page, compose_sel)
            await self._random_wait(0.5, 1.5)

            # Clear existing text
            await page.click(compose_sel)
            await page.keyboard.press("Control+a")
            await asyncio.sleep(0.2)
            await page.keyboard.press("Delete")
            await asyncio.sleep(0.3)

            # Type with human-like speed
            await _human_type(page, compose_sel, caption)
            await self._random_wait(1, 3)

            # ── Upload media if provided ──
            media_path = content.get("media_path", "")
            if media_path and Path(media_path).exists():
                try:
                    file_input = await page.query_selector(
                        'input[type="file"], .file-upload-input, #media-upload'
                    )
                    if file_input:
                        await file_input.set_input_files(media_path)
                        logger.info("Sendible UI: Media uploaded: %s", media_path)
                        await self._random_wait(3, 6)  # Wait for upload
                except Exception as e:
                    logger.warning("Sendible UI: Media upload failed: %s", e)

            # ── Schedule if requested ──
            schedule_at = content.get("schedule_at", "")
            if schedule_at:
                try:
                    schedule_btn = (
                        'button:has-text("Schedule"), .schedule-button, '
                        '#schedule-btn, [data-action="schedule"]'
                    )
                    sched = await page.query_selector(schedule_btn)
                    if sched:
                        await _human_move_to(page, schedule_btn)
                        await page.click(schedule_btn)
                        await self._random_wait(1, 2)
                        # Try to fill date/time input
                        date_input = await page.query_selector(
                            'input[type="datetime-local"], input[name="schedule_date"], '
                            '.schedule-date-input'
                        )
                        if date_input:
                            await date_input.fill(schedule_at)
                            await self._random_wait(1, 2)
                except Exception as e:
                    logger.warning("Sendible UI: Schedule set failed: %s", e)

            # ── Click Send / Publish ──
            send_sel = (
                'button:has-text("Send"), button:has-text("Share"), '
                'button:has-text("Publish"), button:has-text("Post"), '
                '#send-button, .send-btn, .publish-btn, '
                'button[type="submit"].btn-primary'
            )
            await self._random_wait(1, 2)
            await _human_move_to(page, send_sel)
            await self._random_wait(0.5, 1.0)
            await page.click(send_sel)

            # Wait for confirmation
            await self._random_wait(3, 6)

            # Check for success
            try:
                success_el = await page.query_selector(
                    '.success-message, .alert-success, .toast-success, '
                    '.notification-success, [class*="success"]'
                )
                if success_el:
                    success_text = await success_el.text_content() or "Published"
                    logger.info("Sendible UI: Published! %s", success_text)
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

            # No explicit success, but also no error → assume success
            logger.info("Sendible UI: Post sent (no explicit confirmation detected)")
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
