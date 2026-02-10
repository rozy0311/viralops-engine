# 🚀 ViralOps Engine — Multi-Agent Content Factory

> **1 micro-niche → content pack (bài + caption + 7-Layer Hashtag Matrix + ảnh) → queue → schedule/auto-post 20+ kênh → production-safe.**

Built on **EMADS-PR v1.0** architecture with **LangGraph StateGraph**, real **OpenAI GPT** integration, and a **SocialBee-style web dashboard**.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-red.svg)](https://fastapi.tiangolo.com)

---

## 📊 EMADS-PR Analysis

| Component | Score | Detail |
|-----------|-------|--------|
| Data Sources | 4 | RSS, Blog URLs, Micro-niche DB (32 sub-niches), niche_hashtags.json |
| Logic Complexity | 4 | Content transform, 7-layer hashtag matrix, multi-platform adapt, ReconcileGPT |
| Integration Points | 4 | Reddit, Medium, Tumblr, Shopify Blog (real publishers) + 16 more platforms |
| **Automation Score** | **12/12 🔴** | Multi-stakeholder, phased rollout required |
| **Risk Level** | **🔴 HIGH** | One-way door (ban/flag = mất kênh) |

---

## 🏗️ Architecture

```
CEO/User Input (micro-niche + topic + platforms)
    ↓
┌──────────────────────────────┐
│   Orchestrator               │  Initialize pipeline, set defaults
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│   Content Factory (CTO)     │  Real OpenAI GPT → title + body + caption
│                              │  + 7-layer hashtag matrix + platform adapt
└──────────────────────────────┘
    │ (fan-out — parallel dispatch)
    ├── Platform Compliance (COO)  → 9 rules: chars, hashtags, title, links, tone
    ├── Rights & Safety (Legal)    → originality check, unsafe patterns, NSFW
    ├── Risk & Health (Risk)       → peak hours, rate limits, account health
    └── Cost Agent (Cost)          → budget tracking, model selection
    │ (fan-in — Annotated reducers)
    ↓
┌──────────────────────────────┐
│   ReconcileGPT               │  Composite scoring, trade-off analysis
│   (TOOL — no decisions)      │  Action: AUTO_APPROVE / HUMAN_REVIEW / BLOCK
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│   Human Review Gate          │  Dev: auto-approve | Prod: block until human
│   (VIRALOPS_ENV check)       │  Required for risk ≥ 4
└──────────────────────────────┘
    ↓ (approved)
┌──────────────────────────────┐
│   Publish Node               │  Real Scheduler → draft/scheduled/immediate
│                              │  Sets published/failed status per platform
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│   Monitor Node               │  Track published/failed counts
└──────────────────────────────┘
    ↓
  ✅ END  or  🔄 Replan Node → Orchestrator (max 3 loops)
```

### Key Design Decisions

- **Annotated reducers** on all state keys — solves LangGraph fan-in `InvalidUpdateError`
- **`replan_node`** — dedicated node for state mutation (never mutate in routing functions)
- **`VIRALOPS_ENV=production`** — blocks auto-approve in human review
- **Cost-aware model selection**: budget >50% → GPT-4.1, 20-50% → GPT-4.1-mini, <20% → fallback

---

## 📁 Project Structure

```
viralops-engine/
├── graph.py                           # LangGraph StateGraph (11 nodes, 2 routing)
├── main.py                            # CLI entry point
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
│
├── agents/                            # EMADS-PR Agent Suite
│   ├── content_factory.py             # CTO — Real GPT + fallback + RSS rewrite
│   ├── platform_compliance.py         # COO — 9 platform rules enforced
│   ├── rights_safety.py               # Legal — originality + safety + attribution
│   ├── risk_health.py                 # Risk — peak hours + rate limits + health
│   ├── cost_agent.py                  # Cost — persistent budget + model pricing
│   ├── reconcile_gpt.py               # ReconcileGPT — composite scoring + GPT
│   ├── orchestrator.py                # Route + coordinate
│   └── image_video_agent.py           # Image/video generation (scaffold)
│
├── config/                            # Configuration Database
│   ├── niches.yaml                    # 3 categories, 32 sub-niches, 7-layer data
│   ├── platforms.yaml                 # 20+ platforms: limits, formats, rules
│   ├── accounts.yaml                  # Multi-account credentials mapping
│   ├── guardrails.yaml                # Safety rules, kill-switch thresholds
│   └── cost_budget.yaml               # Budget allocation per tier
│
├── core/                              # Engine Core
│   ├── scheduler.py                   # SQLite-backed scheduler + lazy publishers
│   ├── models.py                      # Pydantic/dataclass models
│   ├── account_router.py              # Multi-account rotation
│   ├── dedup.py                       # Content deduplication
│   ├── rate_limiter.py                # Per-platform rate limiting
│   ├── kill_switch.py                 # Circuit breaker
│   ├── queue_adapter.py               # Queue + retry + DLQ
│   └── state.py                       # Shared state definitions
│
├── integrations/                      # Platform Publishers
│   ├── platform_publisher.py          # PublisherRegistry (lazy loading)
│   ├── reddit_publisher.py            # ✅ Real — PRAW OAuth2
│   ├── medium_publisher.py            # ✅ Real — REST API
│   ├── tumblr_publisher.py            # ✅ Real — PyTumblr OAuth
│   ├── shopify_blog_publisher.py      # ✅ Real — Admin REST API
│   ├── lemon8_publisher.py            # Draft staging + webhook
│   ├── rss_reader.py                  # RSS feed management (CRUD)
│   └── trend_researcher.py            # Google Trends + research
│
├── hashtags/                          # 7-Layer Hashtag System
│   ├── matrix_5layer.py               # Generator: 7 layers + 5-cap strategy
│   └── niche_hashtags.json            # 26 niches, pre-built hashtag pools
│
├── templates/                         # Content Templates
│   ├── caption_templates.json         # Universal Caption Formula + 14 niche hooks
│   └── content_transforms.json        # Transformation rules
│
├── web/                               # Dashboard (SocialBee-style)
│   ├── app.py                         # FastAPI — 30+ endpoints
│   └── templates/
│       ├── app.html                   # 8-page SPA (Tailwind CSS)
│       └── dashboard.html             # Analytics dashboard
│
├── monitoring/                        # Observability
│   ├── dashboard.py                   # Metrics dashboard
│   ├── account_health.py              # Account health monitor
│   ├── engagement_tracker.py          # Views/hour, retention, CTR
│   └── alerting.py                    # Alert rules + notifications
│
└── tests/                             # Test Suite
    ├── test_content_factory.py
    ├── test_dedup.py
    ├── test_kill_switch.py
    ├── test_queue.py
    └── test_rate_limiter.py
```

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone https://github.com/rozy0311/viralops-engine.git
cd viralops-engine
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env: add OPENAI_API_KEY (required), platform tokens (optional)

# 3. Start the web dashboard
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000

# 4. CLI — Generate content (draft mode, no posting)
python main.py --niche "plant_based_raw" --platform reddit --mode draft

# 5. CLI — Full pipeline with human review
python main.py --niche "nano_real_life" --platform all --mode review
```

### Production Mode

```bash
# Block auto-approve — human must review every post
export VIRALOPS_ENV=production
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```

---

## 🌐 Web Dashboard

8-page SocialBee-style dashboard at `http://localhost:8000`:

| Page | Endpoint | Description |
|------|----------|-------------|
| Dashboard | `/` | Stats overview, recent posts |
| Compose | `/compose` | Create new content with niche/platform picker |
| Content | `/content` | Manage all content packs |
| Calendar | `/calendar` | Visual scheduling calendar |
| Analytics | `/analytics` | Performance metrics |
| RSS | `/rss` | RSS feed management (add/fetch/import) |
| Hashtags | `/hashtags` | 7-layer hashtag generator |
| Settings | `/settings` | Platform connections, budget config |

### API Endpoints

```
GET  /api/stats                    → Post counts by status
GET  /api/health                   → Engine health check (v2.0.0)
GET  /api/budget                   → Budget remaining (daily/monthly)
POST /api/hashtags/generate        → Generate 7-layer hashtag matrix
GET  /api/posts                    → List all posts
POST /api/posts                    → Create new post
POST /api/publish/{id}             → Publish a specific post
GET  /api/calendar-events          → Calendar events
GET  /api/analytics                → Analytics data
GET  /api/rss/feeds                → List RSS feeds
POST /api/rss/feeds                → Add RSS feed
POST /api/rss/fetch/{id}           → Fetch RSS entries
POST /api/rss/import               → Import RSS entry as draft
GET  /api/scheduler/status         → Scheduler running state
POST /api/scheduler/run-now        → Trigger scheduler manually
```

---

## 🏷️ 7-Layer Hashtag Matrix

The engine uses a **7-layer hashtag strategy** based on Instagram's 2025-2026 algorithm:

| Layer | Purpose | Example |
|-------|---------|---------|
| Broad | High volume, category-level | `#PlantBased`, `#HealthyEating` |
| Local | Geographic targeting | `#ChicagoWinter`, `#NYCVegan` |
| Micro1 | Audience 1 (busy people) | `#MealPrepSimple` |
| Micro2 | Audience 2 (apartment living) | `#SmallSpaceGarden` |
| Micro3 | Audience 3 (beginners) | `#BeginnerGardener` |
| Creator | UGC / creator community | `#PlantBasedUGC` |
| Trend | Year-tagged trending | `#CleanEating2026` |

**5-Cap Strategy**: Instagram pushes 5 hashtags max → use `highest_search` tags from `niche_hashtags.json`.

---

## 🌿 Niche Database

`config/niches.yaml` — **32 sub-niches** across 3 categories:

| Category | Count | Examples |
|----------|-------|----------|
| plant_based_raw | 15 | raw_almond_milk, chia_seed_puddings, hemp_hearts_salads |
| nano_real_life | 12 | before_after_clean, tiny_win_journaling, 5min_skincare |
| indoor_gardening | 5 | led_grow_lights, hydroponic_herbs, window_sill_garden |

Each sub-niche includes: persona, pain points, desires, hooks, 7-layer hashtags, search volume, competition gap.

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key for GPT content generation |
| `VIRALOPS_ENV` | ❌ | `production` = block auto-approve in human review |
| `VIRALOPS_MONTHLY_BUDGET_USD` | ❌ | Monthly budget cap (default: $50) |
| `REDDIT_MAIN_*` | ❌ | Reddit OAuth2 credentials (4 vars) |
| `MEDIUM_MAIN_ACCESS_TOKEN` | ❌ | Medium integration token |
| `TUMBLR_MAIN_*` | ❌ | Tumblr OAuth credentials |
| `SHOPIFY_*` | ❌ | Shopify Admin API (shop, token, blog_id) |
| `TELEGRAM_*` | ❌ | Telegram bot for alerts |

See [.env.example](.env.example) for the full list (20+ platform configurations).

---

## 💰 Cost-Aware Model Selection

| Budget Remaining | Model | Approx. Cost |
|------------------|-------|-------------|
| >50% | GPT-4.1 | ~$0.002/1K tok |
| 20-50% | GPT-4.1-mini | ~$0.0004/1K tok |
| <20% | Local fallback | $0 |
| 0% | **STOP** | — |

---

## 🔒 Production Rails

1. **Dedup + Idempotency** — Hash content + semantic similarity before post
2. **Rate-limit per account/platform** — Throttle with jitter
3. **Circuit Breaker** — Error/flag spike → auto-stop
4. **Rights/Policy Gate** — Source rights + originality check before publish
5. **Kill-switch** — Manual or auto trigger to halt all operations
6. **DLQ (Dead Letter Queue)** — Failed posts retry with backoff
7. **Human Review** — Mandatory for risk ≥ 4 (enforced in production)
8. **Replan Loop** — Max 3 retries for failed publishes
9. **Budget Guard** — Auto-downgrade model when budget low

---

## ⚠️ Kill-Switch Thresholds

| Signal | Threshold | Action |
|--------|-----------|--------|
| Account restriction | 1 occurrence | STOP platform immediately |
| Reach drop | >30% over 7 days | Reduce frequency |
| Upload error rate | >3%/day | STOP, investigate |
| Duplicate detection | >5% flagged | STOP, review content |
| Budget burn | >80% monthly | Switch to cheaper model |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# E2E smoke test (3-platform fan-in)
python -c "
from graph import get_compiled_graph
app = get_compiled_graph()
result = app.invoke({
    'niche_config': {'name': 'plant_based_raw'},
    'topic': 'raw almonds benefits',
    'platforms': ['reddit', 'medium', 'instagram'],
    'publish_mode': 'draft',
}, config={'configurable': {'thread_id': 'test-1'}})
pub = result.get('publish_results', [])
print(f'Published: {sum(1 for r in pub if r[\"status\"]==\"published\")}')
print(f'Failed: {sum(1 for r in pub if r[\"status\"]==\"failed\")}')
"
```

---

## 📜 License

Internal use only. PR-only workflow required for all changes.
