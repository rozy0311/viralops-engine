# 🚀 ViralOps Engine — Multi-Agent Content Factory

> **1 niche → AI generates micro-topics + content + 7-Layer Hashtags + media → auto-schedules + posts to 16 platforms → tracks engagement → optimizes → repeat 24/7.**

Built on **EMADS-PR v1.0** architecture with **LangGraph StateGraph**, real **OpenAI GPT** integration, and a **SocialBee-style web dashboard**.

[![CI — Tests](https://github.com/rozy0311/viralops-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/rozy0311/viralops-engine/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-342%20passing-brightgreen.svg)](#-testing)
[![Platforms](https://img.shields.io/badge/platforms-16-orange.svg)](#-supported-platforms)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](#-docker-deployment)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-red.svg)](https://fastapi.tiangolo.com)

---

## ⚡ What Does It Do?

```
You pick a niche (e.g. "skincare", "crypto", "fitness")
    ↓
AI generates 32 micro-topics automatically
    ↓
AI writes platform-optimized content for each topic
    ↓
Scheduler posts at optimal times across 16 platforms
    ↓
Engine tracks engagement (likes, views, shares)
    ↓
Loops 24/7 — you do nothing
```

**You provide**: 1 niche + API keys for your social accounts.
**Agent handles**: Everything else — content, hashtags, scheduling, posting, analytics.

---

## 🌐 Supported Platforms (16)

| Difficulty | Platform | Auth | Status |
|:----------:|----------|------|:------:|
| 🟢 Easy | **Bluesky** | App Password | ✅ |
| 🟢 Easy | **Mastodon** | OAuth2 Token | ✅ |
| 🟢 Easy | **Medium** | Integration Token | ✅ |
| 🟢 Easy | **Reddit** | OAuth2 Script App | ✅ |
| 🟢 Easy | **Tumblr** | OAuth1 | ✅ |
| 🟢 Easy | **Shopify Blog** | Admin API | ✅ |
| 🟡 Medium | **Twitter/X** | API Key + Token | ✅ |
| 🟡 Medium | **LinkedIn** | OAuth2 | ✅ |
| 🟡 Medium | **Pinterest** | OAuth2 | ✅ |
| 🟡 Medium | **YouTube** | Google OAuth2 | ✅ |
| 🟡 Medium | **Instagram** | Meta Graph API | ✅ |
| 🟡 Medium | **Facebook** | Page Access Token | ✅ |
| 🔴 Hard | **TikTok** | App Review Required | ✅ |
| 🔴 Hard | **Threads** | Meta Business | ✅ |
| 🔴 Hard | **Quora** | Session Cookie | ✅ |
| 🔴 Hard | **Lemon8** | Session Token | ✅ |

> See [SETUP_GUIDE.md](SETUP_GUIDE.md) for step-by-step API key setup for each platform.

---

## 🚀 Quick Start

### Option A — Local (2 minutes)

```bash
git clone https://github.com/rozy0311/viralops-engine.git
cd viralops-engine
pip install -r requirements.txt

# Set up API keys (at minimum: OPENAI_API_KEY + 1 platform)
cp .env.template .env
nano .env

# Start dashboard
python -m uvicorn web.app:app --port 8000
# Open http://localhost:8000
```

### Option B — Docker (1 minute)

```bash
git clone https://github.com/rozy0311/viralops-engine.git
cd viralops-engine
cp .env.template .env
nano .env    # add your API keys

docker compose up -d
# Dashboard: http://localhost:8000
# Scheduler runs automatically in background
```

### Option C — CLI

```bash
# Draft mode (no posting)
python main.py --niche "plant_based_raw" --platform reddit --mode draft

# Full pipeline with human review
python main.py --niche "nano_real_life" --platform all --mode review
```

---

## 🏗️ Architecture (EMADS-PR v1.0)

```
User Input (niche + topic + platforms)
    ↓
┌──────────────────────────────────────┐
│ Orchestrator                         │ Initialize, set defaults
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Content Factory (CTO)                │ GPT → title + body + caption
│                                      │ + 7-layer hashtags + platform adapt
└──────────────────────────────────────┘
    │ (fan-out — parallel)
    ├── Platform Compliance (COO)       → char limits, hashtags, format
    ├── Rights & Safety (Legal)         → originality, NSFW, attribution
    ├── Risk & Health (Risk)            → rate limits, peak hours, health
    └── Cost Agent (Cost)               → budget tracking, model selection
    │ (fan-in)
    ↓
┌──────────────────────────────────────┐
│ ReconcileGPT (TOOL — no decisions)   │ Score + trade-off analysis
│                                      │ → AUTO_APPROVE / HUMAN_REVIEW / BLOCK
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Human Review Gate                    │ Required for risk ≥ 4
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Publish → Schedule → Post            │ Real APIs, 16 platforms
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Monitor + Engagement Fetch           │ Track metrics, optimize
└──────────────────────────────────────┘
    ↓
  ✅ END  or  🔄 Replan (max 3 loops)
```

---

## 📁 Project Structure

```
viralops-engine/
├── graph.py                        # LangGraph StateGraph (11 nodes)
├── main.py                         # CLI entry point
├── Dockerfile                      # Production container
├── docker-compose.yml              # Web + scheduler services
├── .env.template                   # All 16 platform env vars
├── SETUP_GUIDE.md                  # Step-by-step API key guide
│
├── agents/                         # EMADS-PR Agent Suite
│   ├── content_factory.py          # CTO — GPT content gen (16 platforms)
│   ├── platform_compliance.py      # COO — platform rule enforcement
│   ├── rights_safety.py            # Legal — originality + safety
│   ├── risk_health.py              # Risk — rate limits + health
│   ├── cost_agent.py               # Cost — budget + model selection
│   ├── reconcile_gpt.py            # ReconcileGPT — scoring
│   └── orchestrator.py             # Coordinator
│
├── core/                           # Engine Core
│   ├── scheduler.py                # SQLite scheduler + 16 publishers
│   ├── time_slot_engine.py         # Optimal posting times (analytics-backed)
│   ├── rate_limiter.py             # Per-platform throttling
│   ├── kill_switch.py              # Circuit breaker
│   ├── queue_adapter.py            # Retry + DLQ
│   └── dedup.py                    # Content deduplication
│
├── integrations/                   # Platform Publishers (16)
│   ├── reddit_publisher.py         # PRAW OAuth2
│   ├── medium_publisher.py         # REST API
│   ├── tumblr_publisher.py         # PyTumblr OAuth
│   ├── shopify_blog_publisher.py   # Admin REST API
│   ├── threads_publisher.py        # Meta Threads API
│   ├── bluesky_publisher.py        # AT Protocol
│   ├── mastodon_publisher.py       # REST + OAuth2
│   ├── quora_publisher.py          # GraphQL + webhook
│   ├── social_connectors.py        # Twitter/IG/FB/YT/LI/Pin/TikTok
│   ├── media_processor.py          # Image→video, slideshow, text overlay
│   ├── tiktok_music.py             # BPM-aware music + trending decay
│   ├── rss_auto_poster.py          # RSS→content→schedule
│   ├── rss_reader.py               # RSS feed CRUD
│   └── telegram_bot.py             # Telegram alerts
│
├── monitoring/                     # Observability
│   ├── engagement_fetcher.py       # Pull metrics from all 16 platforms
│   ├── analytics.py                # Analytics engine
│   ├── engagement_tracker.py       # Views, retention, CTR
│   └── alerting.py                 # Alert rules
│
├── hashtags/                       # 7-Layer Hashtag System
│   ├── matrix_5layer.py            # Generator: 7 layers + 5-cap
│   └── niche_hashtags.json         # 26 niches, pre-built pools
│
├── web/                            # Dashboard
│   ├── app.py                      # FastAPI — 75+ API endpoints
│   └── templates/                  # SocialBee-style 8-page SPA
│
├── tests/                          # 342 tests, all passing
│   ├── test_v26_features.py        # API endpoints + Docker (41)
│   ├── test_v25_features.py        # Publishers + engines (71)
│   └── ...                         # Core + integration tests (230)
│
└── .github/workflows/
    ├── ci.yml                      # Auto-test on push (342 tests)
    └── copilot-agent.yml           # Copilot coding agent
```

---

## 🔌 API Endpoints (75+)

### Content & Publishing
```
POST /api/posts                     → Create content
POST /api/publish/{id}              → Publish to platform
GET  /api/posts                     → List all posts
GET  /api/stats                     → Post counts by status
GET  /api/calendar-events           → Calendar view
```

### Engagement & Analytics
```
POST /api/engagement/fetch          → Pull real metrics from 16 platforms
GET  /api/engagement/summary        → Engagement summary (filter by platform/days)
GET  /api/engagement/post/{id}      → Per-post engagement data
GET  /api/analytics                 → Analytics dashboard data
```

### Smart Scheduling
```
GET  /api/time-slots/suggest/{plat} → Optimal next posting time
POST /api/time-slots/schedule       → Full daily schedule across platforms
GET  /api/time-slots/best-hours     → Analytics-backed best hours
GET  /api/scheduler/status          → Scheduler running state
POST /api/scheduler/run-now         → Trigger scheduler manually
```

### TikTok Music (BPM-Aware)
```
POST /api/tiktok/music/recommend    → Music by mood + content_pace + target_bpm
POST /api/tiktok/music/decay        → Apply trending score decay
GET  /api/tiktok/music/trending     → Top trending tracks
```

### Media Processing
```
POST /api/media/process             → Image → Ken Burns video
POST /api/media/multi-slideshow     → Multi-image video slideshow
POST /api/media/text-overlay        → Add text caption to video
POST /api/media/subtitles           → Add timed SRT subtitles
```

### RSS Auto-Poster
```
GET  /api/rss/feeds                 → List feeds
POST /api/rss/feeds                 → Add feed
POST /api/rss/auto-poster/start     → Start auto-poster daemon
GET  /api/rss/auto-poster/status    → Check daemon status
```

### Platform Management
```
GET  /api/platforms/setup-status    → Check all 16 platform API key status
GET  /api/social/status             → Live connection test per platform
GET  /api/budget                    → Budget remaining
GET  /api/health                    → Engine health check
```

---

## 🏷️ 7-Layer Hashtag Matrix

| Layer | Purpose | Example |
|-------|---------|---------|
| Broad | High volume | `#PlantBased`, `#HealthyEating` |
| Local | Geographic | `#ChicagoVegan`, `#NYCWellness` |
| Micro1 | Audience: busy people | `#MealPrepSimple` |
| Micro2 | Audience: apartments | `#SmallSpaceGarden` |
| Micro3 | Audience: beginners | `#BeginnerGardener` |
| Creator | UGC community | `#PlantBasedUGC` |
| Trend | Year-tagged | `#CleanEating2026` |

**5-Cap Strategy**: Instagram 2025-2026 algorithm → 5 highest-search hashtags per post.

---

## 🌿 Niche Database (32 Sub-Niches)

| Category | Count | Examples |
|----------|:-----:|---------|
| plant_based_raw | 15 | raw_almond_milk, chia_seed_puddings, hemp_hearts_salads |
| nano_real_life | 12 | before_after_clean, tiny_win_journaling, 5min_skincare |
| indoor_gardening | 5 | led_grow_lights, hydroponic_herbs, window_sill_garden |

Each sub-niche includes: persona, pain points, desires, hooks, 7-layer hashtags, search volume, competition gap.

---

## 💰 Cost-Aware Model Selection

| Budget Remaining | Model | Approx. Cost |
|:----------------:|-------|:------------:|
| >50% | GPT-4.1 | ~$0.002/1K tok |
| 20-50% | GPT-4.1-mini | ~$0.0004/1K tok |
| <20% | Local fallback | $0 |
| 0% | **STOP** | — |

---

## 🔒 Production Safety Rails

| # | Rail | Description |
|:-:|------|-------------|
| 1 | Dedup | Hash + semantic similarity before post |
| 2 | Rate Limiter | Per-platform throttle with jitter |
| 3 | Circuit Breaker | Error spike → auto-stop |
| 4 | Rights Gate | Source rights + originality check |
| 5 | Kill Switch | Manual or auto halt |
| 6 | DLQ | Dead Letter Queue with retry backoff |
| 7 | Human Review | Mandatory for risk ≥ 4 |
| 8 | Replan Loop | Max 3 retries for failed publishes |
| 9 | Budget Guard | Auto-downgrade model when budget low |

---

## 🐳 Docker Deployment

```bash
# 1. Configure
cp .env.template .env
nano .env   # add OPENAI_API_KEY + platform keys

# 2. Launch (web dashboard + background scheduler)
docker compose up -d

# 3. Check
docker compose logs -f
curl http://localhost:8000/api/platforms/setup-status
```

**Services**:
- `viralops-web` — Dashboard + API on port 8000
- `viralops-scheduler` — Auto-publish + engagement fetch (background)

---

## 🧪 Testing

```bash
# Run full suite (342 tests)
pytest tests/ -v

# Run specific version tests
pytest tests/test_v26_features.py -v   # API endpoints (41)
pytest tests/test_v25_features.py -v   # Publishers + engines (71)
```

**CI/CD**: Tests run automatically on every push via [GitHub Actions](.github/workflows/ci.yml).

---

## 📊 EMADS-PR Score

| Component | Score | Detail |
|-----------|:-----:|--------|
| Data Sources | 4 | RSS, Blog URLs, 32 sub-niches, hashtag DB |
| Logic Complexity | 4 | Multi-agent, 7-layer hashtags, ReconcileGPT |
| Integration Points | 4 | 16 real platform publishers |
| **Automation Score** | **12/12 🔴** | Multi-stakeholder, phased rollout |

---

## 📜 License

Internal use only. PR-only workflow required for all changes.
