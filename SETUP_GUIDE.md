# 🔑 ViralOps Engine — API Keys Setup Guide

> Step-by-step guide to connect all 16 social media platforms.
> Ordered from **easiest → hardest**. Only set up platforms you need.

---

## 📋 Prerequisites

1. **Python 3.13+** installed
2. **Copy environment template**: `cp .env.template .env`
3. **OpenAI API Key** (required for content generation):
   - Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Create new secret key
   - Add to `.env`: `OPENAI_API_KEY=sk-...`

---

## 🟢 EASY — Under 5 Minutes

### 1. Bluesky

| Field | Value |
|-------|-------|
| Time | ~2 minutes |
| Cost | Free |
| Approval | None |

**Steps:**
1. Go to [bsky.app/settings/app-passwords](https://bsky.app/settings/app-passwords)
2. Click **"Add App Password"**
3. Name it `viralops` → click **Create**
4. Copy the generated password

```env
BLUESKY_MAIN_HANDLE=yourname.bsky.social
BLUESKY_MAIN_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
```

---

### 2. Mastodon

| Field | Value |
|-------|-------|
| Time | ~3 minutes |
| Cost | Free |
| Approval | None |

**Steps:**
1. Log in to your Mastodon instance (e.g., mastodon.social)
2. Go to **Settings → Development → New Application**
3. Name: `ViralOps Engine`
4. Scopes: ✅ `read` ✅ `write` ✅ `push`
5. Click **Submit** → copy **Your access token**

```env
MASTODON_MAIN_ACCESS_TOKEN=your-token-here
MASTODON_MAIN_INSTANCE_URL=https://mastodon.social
```

---

### 3. Medium

| Field | Value |
|-------|-------|
| Time | ~2 minutes |
| Cost | Free |
| Approval | None |

**Steps:**
1. Go to [medium.com/me/settings](https://medium.com/me/settings)
2. Scroll to **"Integration tokens"**
3. Enter description `viralops` → click **Get token**
4. Copy the token

```env
MEDIUM_ACCESS_TOKEN=your-integration-token
```

---

### 4. Reddit

| Field | Value |
|-------|-------|
| Time | ~5 minutes |
| Cost | Free |
| Approval | None |

**Steps:**
1. Go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Click **"create another app..."**
3. Fill in:
   - Name: `ViralOps Engine`
   - Type: **script**
   - Redirect URI: `http://localhost:8000`
4. Click **Create app**
5. Note: **client ID** is under the app name, **secret** is labeled

```env
REDDIT_CLIENT_ID=your-client-id
REDDIT_CLIENT_SECRET=your-client-secret
REDDIT_USERNAME=your-reddit-username
REDDIT_PASSWORD=your-reddit-password
REDDIT_SUBREDDIT=your_subreddit
```

---

### 5. Tumblr

| Field | Value |
|-------|-------|
| Time | ~5 minutes |
| Cost | Free |
| Approval | None |

**Steps:**
1. Go to [tumblr.com/oauth/apps](https://www.tumblr.com/oauth/apps)
2. Click **"Register application"**
3. Fill in app details → callback URL: `http://localhost:8000`
4. Copy **OAuth Consumer Key** and **Secret**
5. Use [Tumblr API Console](https://api.tumblr.com/console/calls/user/info) to get OAuth tokens

```env
TUMBLR_CONSUMER_KEY=your-consumer-key
TUMBLR_CONSUMER_SECRET=your-consumer-secret
TUMBLR_OAUTH_TOKEN=your-oauth-token
TUMBLR_OAUTH_SECRET=your-oauth-secret
TUMBLR_BLOG_NAME=your-blog-name
```

---

### 6. Shopify Blog

| Field | Value |
|-------|-------|
| Time | ~5 minutes |
| Cost | Free (Shopify plan required) |
| Approval | None |

**Steps:**
1. In Shopify Admin → **Settings → Apps and sales channels → Develop apps**
2. Click **Create an app** → name it `ViralOps`
3. Click **Configure Admin API scopes**
4. Enable: ✅ `write_content` ✅ `read_content`
5. Click **Install app** → copy **Admin API access token**

```env
SHOPIFY_SHOP_URL=your-store.myshopify.com
SHOPIFY_ACCESS_TOKEN=shpat_xxxxxxxxxxxx
```

---

## 🟡 MEDIUM — 15-30 Minutes

### 7. Twitter/X

| Field | Value |
|-------|-------|
| Time | ~15 minutes |
| Cost | Free (Basic plan) or $100/mo (Pro) |
| Approval | Developer account |

**Steps:**
1. Go to [developer.twitter.com](https://developer.twitter.com)
2. Sign up for **Free** or **Basic** access
3. Create a **Project** → create an **App**
4. Go to **Keys and Tokens** tab
5. Generate:
   - **API Key and Secret** (Consumer Keys)
   - **Access Token and Secret** (Authentication Tokens)
6. Set app permissions to **Read and Write**

```env
TWITTER_MAIN_API_KEY=your-api-key
TWITTER_MAIN_API_SECRET=your-api-secret
TWITTER_MAIN_ACCESS_TOKEN=your-access-token
TWITTER_MAIN_ACCESS_SECRET=your-access-secret
```

> ⚠️ **Free plan**: 1,500 tweets/month. **Basic ($100/mo)**: 3,000 tweets/month.

---

### 8. LinkedIn

| Field | Value |
|-------|-------|
| Time | ~20 minutes |
| Cost | Free |
| Approval | App review (auto-approved for personal) |

**Steps:**
1. Go to [linkedin.com/developers/apps](https://www.linkedin.com/developers/apps)
2. Click **Create app** → fill in details
3. Under **Products** → request access to **Share on LinkedIn** and **Sign In with LinkedIn using OpenID Connect**
4. Go to **Auth** tab → copy **Client ID** and **Client Secret**
5. Use OAuth 2.0 flow to get an access token (scopes: `w_member_social`, `r_liteprofile`)
6. Or use [LinkedIn Token Generator](https://www.linkedin.com/developers/tools/oauth/token-generator)

```env
LINKEDIN_MAIN_ACCESS_TOKEN=your-access-token
```

---

### 9. Pinterest

| Field | Value |
|-------|-------|
| Time | ~15 minutes |
| Cost | Free |
| Approval | App review |

**Steps:**
1. Go to [developers.pinterest.com/apps](https://developers.pinterest.com/apps/)
2. Click **Create app** → fill in details
3. Request access to **Pins** and **Boards**
4. Go to **OAuth** → generate access token with scopes: `boards:read`, `pins:read`, `pins:write`

```env
PINTEREST_MAIN_ACCESS_TOKEN=your-access-token
```

---

### 10. YouTube

| Field | Value |
|-------|-------|
| Time | ~20 minutes |
| Cost | Free |
| Approval | Google Cloud project |

**Steps:**
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project (or select existing)
3. Enable **YouTube Data API v3**
4. Go to **Credentials** → Create **OAuth 2.0 Client ID**
   - Type: **Web application**
   - Redirect URI: `http://localhost:8000/oauth/youtube/callback`
5. Download the credentials JSON
6. Get a refresh token via OAuth 2.0 flow

```env
YOUTUBE_MAIN_API_KEY=your-api-key
YOUTUBE_MAIN_CLIENT_ID=your-client-id.apps.googleusercontent.com
YOUTUBE_MAIN_CLIENT_SECRET=your-client-secret
YOUTUBE_MAIN_REFRESH_TOKEN=your-refresh-token
```

---

### 11. Instagram

| Field | Value |
|-------|-------|
| Time | ~30 minutes |
| Cost | Free |
| Approval | Facebook/Meta Business account |

**Steps:**
1. You need a **Facebook Page** connected to your **Instagram Business/Creator** account
2. Go to [developers.facebook.com](https://developers.facebook.com) → create an app
3. Add **Instagram Graph API** product
4. Generate a **long-lived access token** (60-day)
5. Get your **Instagram Business Account ID** via the Graph API Explorer

```env
INSTAGRAM_MAIN_ACCESS_TOKEN=your-long-lived-token
INSTAGRAM_MAIN_USER_ID=your-ig-business-id
```

> 💡 Use [Graph API Explorer](https://developers.facebook.com/tools/explorer/) to get your user ID.

---

### 12. Facebook Page

| Field | Value |
|-------|-------|
| Time | ~20 minutes |
| Cost | Free |
| Approval | Facebook/Meta Business account |

**Steps:**
1. Go to [developers.facebook.com](https://developers.facebook.com) → create an app (if not done)
2. Add **Pages** product
3. In Graph API Explorer, select your Page → generate **Page Access Token**
4. Extend to long-lived token

```env
FACEBOOK_MAIN_ACCESS_TOKEN=your-page-access-token
FACEBOOK_MAIN_PAGE_ID=your-page-id
```

---

## 🔴 HARD — Requires Verification or Workarounds

### 13. TikTok

| Field | Value |
|-------|-------|
| Time | Days to weeks (app review) |
| Cost | Free |
| Approval | **App review required** |

**Steps:**
1. Go to [developers.tiktok.com](https://developers.tiktok.com)
2. Create a developer account (requires TikTok account)
3. Create an app → select **Content Posting API**
4. Submit for **app review** (can take days/weeks)
5. Once approved, get access token via OAuth 2.0

```env
TIKTOK_MAIN_ACCESS_TOKEN=your-access-token
TIKTOK_MAIN_OPEN_ID=your-open-id
```

> ⚠️ TikTok's app review process is the longest of all platforms. Start this early.

---

### 14. Threads

| Field | Value |
|-------|-------|
| Time | ~30 minutes (if you have Instagram Business) |
| Cost | Free |
| Approval | Meta Business account + Instagram |

**Steps:**
1. **Prerequisite**: Set up Instagram (step 11) first
2. Your Instagram token also works for Threads API
3. Go to [developers.facebook.com/docs/threads](https://developers.facebook.com/docs/threads)
4. Use the same access token from Instagram
5. Get Threads User ID via Graph API

```env
THREADS_MAIN_ACCESS_TOKEN=your-ig-access-token
THREADS_MAIN_USER_ID=your-threads-user-id
```

---

### 15. Quora

| Field | Value |
|-------|-------|
| Time | ~10 minutes |
| Cost | Free |
| Approval | None (but uses session cookies — unofficial) |

**Steps:**
1. Log in to [quora.com](https://quora.com)
2. Open **DevTools** (F12) → **Application** tab → **Cookies**
3. Copy the `m-b` cookie value → this is your session cookie
4. In the **Network** tab, find any GraphQL request → copy the `formkey` from headers
5. (Optional) Set up a webhook URL for fallback posting

```env
QUORA_MAIN_SESSION_COOKIE=your-m-b-cookie-value
QUORA_MAIN_FORMKEY=your-formkey
QUORA_MAIN_WEBHOOK=https://your-webhook-url (optional)
```

> ⚠️ Session cookies expire. You'll need to refresh them periodically.

---

### 16. Lemon8

| Field | Value |
|-------|-------|
| Time | ~10 minutes |
| Cost | Free |
| Approval | None (unofficial API) |

**Steps:**
1. Log in to Lemon8 app
2. Use a proxy tool (mitmproxy/Charles) to capture the session token
3. Extract the session token from request headers

```env
LEMON8_SESSION_TOKEN=your-session-token
```

> ⚠️ No official API. Session tokens expire frequently.

---

## 🔔 Telegram Notifications (Optional)

| Field | Value |
|-------|-------|
| Time | ~5 minutes |
| Cost | Free |
| Approval | None |

**Steps:**
1. Open Telegram → search for **@BotFather**
2. Send `/newbot` → follow prompts → get **bot token**
3. Send a message to your bot
4. Go to `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
5. Find your **chat_id** in the response

```env
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
TELEGRAM_CHAT_ID=123456789
```

---

## ✅ Verify Your Setup

After adding API keys, verify which platforms are configured:

```bash
# Start the server
python -m uvicorn web.app:app --port 8000

# Check setup status
curl http://localhost:8000/api/platforms/setup-status
```

This returns a JSON showing which platforms are ✅ configured and which are ❌ missing.

---

## 🏗️ Quick Start (Docker)

```bash
# 1. Copy environment template
cp .env.template .env

# 2. Fill in your API keys (at minimum: OPENAI_API_KEY + 1 platform)
nano .env

# 3. Start everything
docker compose up -d

# 4. Open dashboard
open http://localhost:8000
```

---

---

## 🌉 Publer — Bridge Publisher (Optional)

| Field | Value |
|-------|-------|
| Time | ~10 minutes |
| Cost | ~$10/mo per social account |
| Approval | None |

Publer is a **REST API bridge** that can post to any connected social account.
Useful for platforms where direct OAuth setup requires app review (TikTok) or business verification (Meta).

> **Note**: Direct OAuth APIs are always preferred (free). Use Publer only as fallback.

### Automated Setup (Recommended)

```bash
python setup_publer.py
```

The wizard will guide you through all 6 steps.

### Manual Setup

**Steps:**
1. Sign up at [publer.com](https://publer.com)
2. Upgrade to **Business plan** ($10/mo per account)
3. Go to **Settings → API** → Generate API Key
4. Connect your social accounts (TikTok, Pinterest, LinkedIn, etc.)
5. Note your **Workspace ID** from the URL: `app.publer.com/workspaces/{WORKSPACE_ID}`
6. Test connection:

```bash
curl -H "Authorization: Bearer-API YOUR_KEY" \
     -H "Publer-Workspace-Id: YOUR_WORKSPACE_ID" \
     https://app.publer.com/api/v1/me
```

```env
PUBLER_API_KEY=your-api-key-here
PUBLER_WORKSPACE_ID=your-workspace-id-here
```

### Supported Platforms via Publer

| Platform | Publer Network Key |
|----------|--------------------|
| TikTok | `tiktok` |
| Instagram | `instagram` |
| Facebook | `facebook` |
| Pinterest | `pinterest` |
| LinkedIn | `linkedin` |
| YouTube | `youtube` |
| Twitter/X | `twitter` |
| Threads | `threads` |
| Mastodon | `mastodon` |
| Bluesky | `bluesky` |
| Telegram | `telegram` |
| Google Business | `google` |

---

## 💡 Tips

- **Start small**: Just set up OpenAI + 1-2 easy platforms (Bluesky, Mastodon)
- **Add more later**: The system works with any combination of platforms
- **Token refresh**: Meta/Google tokens expire — set up token refresh cron jobs
- **Rate limits**: Each platform has daily limits built into the scheduler
- **Monitoring**: Enable Telegram alerts to get notified of publish results
- **Publer bridge**: Use `python setup_publer.py` for quick multi-platform access without individual OAuth setup
