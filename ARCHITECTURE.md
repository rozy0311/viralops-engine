# 🏗️ ARCHITECTURE — ViralOps Engine (EMADS-PR v1.0)

## 1. System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        ViralOps Engine                                   │
│                                                                          │
│  Input: micro-niche + tone + location + audience + brand tag             │
│  Output: content pack → queue → auto-post (multi-channel, production)    │
│                                                                          │
│  ┌────────────┐   ┌──────────────┐   ┌─────────────┐   ┌────────────┐  │
│  │ Input      │──▶│ Orchestrator │──▶│ Specialist  │──▶│ Reconcile  │  │
│  │ Wizard     │   │ Agent        │   │ Agents ×6   │   │ GPT        │  │
│  └────────────┘   └──────────────┘   │ (parallel)  │   └─────┬──────┘  │
│                        ▲              └─────────────┘         │         │
│                        │                                      ▼         │
│                   ┌────┴────┐                          ┌─────────────┐  │
│                   │ Memory  │                          │ Human Review│  │
│                   │ Agent   │                          │ Gate        │  │
│                   └─────────┘                          └─────┬───────┘  │
│                                                              │         │
│                                                              ▼         │
│                        ┌──────────────────────────────────────┐         │
│                        │     Queue Adapter (Publisher)         │         │
│                        │  ┌─────┐ ┌──────┐ ┌────────┐        │         │
│                        │  │Dedup│ │Rate  │ │Retry + │        │         │
│                        │  │     │ │Limit │ │DLQ     │        │         │
│                        │  └─────┘ └──────┘ └────────┘        │         │
│                        └──────────────┬───────────────────────┘         │
│                                       │                                 │
│                        ┌──────────────▼───────────────────────┐         │
│                        │     Platform Adapters                 │         │
│                        │  ┌────┐ ┌──┐ ┌──┐ ┌───┐ ┌──┐ ┌──┐  │         │
│                        │  │TT  │ │IG│ │FB│ │Pin│ │LI│ │YT│  │         │
│                        │  └────┘ └──┘ └──┘ └───┘ └──┘ └──┘  │         │
│                        └──────────────────────────────────────┘         │
│                                       │                                 │
│                        ┌──────────────▼───────────────────────┐         │
│                        │     Monitor / Validator               │         │
│                        │  Engagement │ Health │ Kill-switch     │         │
│                        └──────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Agent Roles (EMADS-PR Mapping)

| EMADS Role | ViralOps Agent | Responsibility |
|------------|---------------|----------------|
| **CEO Input** | Input Wizard | micro-niche, tone, audience, brand tags |
| **Orchestrator** | `orchestrator.py` | Route tasks, manage state, coordinate parallel agents |
| **CTO Agent** | `content_factory.py` + `image_video_agent.py` | Generate content pack: title, body, caption, hashtag matrix, images |
| **COO Agent** | `platform_compliance.py` | Format per platform, enforce char limits, optimize for algo |
| **Legal Agent** | `rights_safety.py` | Rights check, brand safety filter, PII guard, no-celeb-voice |
| **Risk Agent** | `risk_health.py` | Account health, ban-risk scoring, duplicate detection |
| **Cost Agent** | `cost_agent.py` | Token/API budget tracking, model tier selection |
| **ReconcileGPT** | `reconcile_gpt.py` | Merge outputs, score trade-offs, flag risky content |
| **Human Review** | Human Review Gate | Approve/Edit/Reject (mandatory for risk ≥ 4) |
| **Executor** | `queue_adapter.py` | Publish via direct Platform OAuth APIs |
| **Monitor** | `monitoring/` | Engagement tracking, health checks, kill-switch |
| **Memory** | State persistence | Past niche performance, failed posts, learned patterns |

---

## 3. Content Factory Pipeline

```
micro-niche input
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 1: Research (Built-in Trend Researcher)│
│   - Get trending hashtags (Google Trends)   │
│   - Analyze niche performance (internal DB) │
│   - Cross-reference platform hashtag APIs   │
└────────────────┬────────────────────────┘
                 │
    ▼
┌─────────────────────────────────────────┐
│ Step 2: Content Generation (GPT-4.1)    │
│   - Title (SEO + platform optimized)    │
│   - Long-form content (4200+ chars)     │
│   - Universal caption (multi-platform)  │
│   - Hashtag Matrix 5 Layer              │
└────────────────┬────────────────────────┘
                 │
    ▼
┌─────────────────────────────────────────┐
│ Step 3: Image/Video Generation          │
│   - Prompt generation (9:16 ratio)      │
│   - Image creation via DALL-E/Midjourney│
│   - Alt-text generation                 │
│   - Micro-niche caption overlay         │
└────────────────┬────────────────────────┘
                 │
    ▼
┌─────────────────────────────────────────┐
│ Step 4: Platform Adaptation             │
│   Per platform:                         │
│   - Trim to char limit                  │
│   - Adjust format (9:16 / 1:1 / 16:9)  │
│   - Platform-specific hashtag count     │
│   - CTA optimization                    │
└────────────────┬────────────────────────┘
                 │
    ▼
┌─────────────────────────────────────────┐
│ Step 5: Safety & Quality Gate           │
│   - Dedup hash check                    │
│   - Semantic similarity scan            │
│   - Rights/source verification          │
│   - Brand safety filter                 │
│   - Transform score (must be ≥70%)      │
└─────────────────────────────────────────┘
```

---

## 4. Hashtag Matrix 5 Layer

```
Layer 1: 🎯 Niche-Specific (1-2 tags)
    #RawAlmonds #VeganProtein

Layer 2: 📊 High-Volume Trending (1-2 tags)
    #PlantBased #HealthySnacks

Layer 3: 🌍 Community/Lifestyle (1 tag)
    #SustainableLiving

Layer 4: 📍 Location/Seasonal (1 tag)
    #CostcoBulk OR #WinterMealPrep

Layer 5: 🔥 Viral/Hook (1 tag)
    #FoodHack OR #MealPrepTikTok

Total: 5 hashtags (YouTube sweet spot) — expandable to 15 for IG/TikTok
```

---

## 5. Platform Specs Matrix

| Platform | Max Caption | Max Hashtags | Video Ratio | Rate Limit | Auth Method |
|----------|-------------|-------------|-------------|------------|-------------|
| TikTok | 2200 chars | 5-8 | 9:16 | 25/24h | OAuth (Content Posting API) |
| Instagram Reels | 2200 chars | 30 | 9:16, 1:1 | 25/24h | OAuth |
| Instagram Feed | 2200 chars | 30 | 1:1, 4:5 | 25/24h | OAuth |
| Facebook Reels | 5000 chars | 30 | 9:16 | 50/24h | OAuth |
| YouTube Shorts | 100 title + 5000 desc | 3-5 | 9:16 | 6/24h | OAuth |
| Pinterest | 500 chars | 20 | 2:3, 1:1 | 100/24h | OAuth |
| LinkedIn | 3000 chars | 5 | 1:1, 16:9 | 50/24h | OAuth |
| Twitter/X | 280 chars | 3-5 | 16:9, 1:1 | 50/24h | OAuth |

---

## 6. Safety & Compliance Stack

### 6.1 Content Policy Rules
```yaml
transform_minimum: 0.70          # ≥70% biến đổi so với nguồn gốc
spacing_hours: 24                 # Cách nhau ≥24h cùng niche
max_same_content_platforms: 3     # Không đăng y chang trên >3 platform
no_celeb_voice: true              # Chỉ dùng licensed voice
watermark_check: true             # Phát hiện watermark nguồn
pii_scan: true                    # Quét PII trước khi đăng
```

### 6.2 Kill-Switch Logic
```python
class KillSwitch:
    triggers = {
        "account_restriction": {"threshold": 1, "action": "STOP_PLATFORM"},
        "reach_drop_7d":       {"threshold": 0.30, "action": "REDUCE_FREQUENCY"},
        "error_rate_daily":    {"threshold": 0.03, "action": "STOP_ALL"},
        "duplicate_rate":      {"threshold": 0.05, "action": "STOP_REVIEW"},
        "budget_burn":         {"threshold": 0.80, "action": "DOWNGRADE_MODEL"},
    }
```

---

## 7. Cost-Aware Model Hierarchy

```
Budget healthy (>50%)   → GPT-4.1        (best content quality)
Budget moderate (20-50%) → GPT-4.1-mini   (cost-balanced)
Budget tight (<20%)     → Llama 4         (open-source, self-hosted)
Budget critical (<5%)   → Qwen3 Flash     (ultra-budget)
Budget empty (0%)       → STOP & report to human
```

---

## 8. Integration Strategy (Self-Contained Pipeline)

**ViralOps Engine IS the pipeline** — no third-party schedulers.
Primary: direct Platform OAuth APIs. Fallback: **Publer REST API bridge** (~$10/mo per account).

### Publer Bridge Publisher (NEW — v4.0)

For platforms where direct OAuth setup is complex (TikTok app review, Meta Business verification),
Publer provides a REST API bridge at `https://app.publer.com/api/v1/`.

| Feature | Detail |
|---------|--------|
| Auth | `Bearer-API {key}` + `Publer-Workspace-Id` header |
| Workflow | Async: submit → job_id → poll `/job_status/{id}` |
| Cost | ~$10/mo per social account (vs Sendible $199/mo) |
| Module | `integrations.publer_publisher.PublerPublisher` |
| Setup | `python setup_publer.py` |

> **Priority**: Direct OAuth APIs are always preferred (free, no middleman).
> Publer bridge is used ONLY when direct API access requires app review or business verification.

### Platform API Clients
| Platform | API | Auth | Module |
|----------|-----|------|--------|
| TikTok | Content Posting API | OAuth2 (registered app) | `platform_publisher.TikTokPublisher` |
| Instagram | Graph API (via Facebook Business) | Long-lived token | `platform_publisher.InstagramPublisher` |
| Facebook | Graph API (Page token) | OAuth2 | `platform_publisher.FacebookPublisher` |
| YouTube | Data API v3 (resumable upload) | OAuth2 | `platform_publisher.YouTubePublisher` |
| Pinterest | API v5 | OAuth2 | `platform_publisher.PinterestPublisher` |
| LinkedIn | Marketing API | OAuth2 | `platform_publisher.LinkedInPublisher` |
| Twitter/X | API v2 | Bearer token / OAuth2 | `platform_publisher.TwitterPublisher` |

### Trend Research (Built-in)
- **Internal niche history** from state/DB
- **Google Trends** via pytrends (optional)
- **Platform hashtag APIs** via official OAuth
- **NO external MCP / scraper services**

### Security Rules
- **KHÔNG BAO GIỊ**: login/password/cookie/RPA automation
- **ONLY**: Official OAuth2 APIs with registered apps
- **Token refresh**: Auto-refresh via OAuth2 refresh flow
- **Rate limits**: Enforced per-platform via `core.rate_limiter`

---

## 9. Data Flow (LangGraph State Machine)

```python
# States
class ViralOpsState(TypedDict):
    # Input
    niche: str
    tone: str
    audience: str
    brand_tags: list[str]
    
    # Content Factory output
    content_pack: Optional[ContentPack]
    
    # Agent outputs (parallel)
    compliance_check: Optional[dict]
    rights_check: Optional[dict]
    risk_score: int
    cost_estimate: Optional[dict]
    
    # ReconcileGPT
    reconcile_decision: Optional[dict]
    publish_mode: str  # draft | review | queue | auto
    
    # Human Review
    human_approved: Optional[bool]
    human_feedback: Optional[str]
    
    # Execution
    queue_status: Optional[dict]
    publish_results: list[dict]
    
    # Monitoring
    engagement_metrics: Optional[dict]
    account_health: Optional[dict]
    
    # Control
    replan_count: int  # max 3
    kill_switch_active: bool
```

---

## 10. RACI Matrix

| Decision | Responsible | Accountable | Consulted | Informed |
|----------|-------------|-------------|-----------|----------|
| Content quality/style | Content Factory | CTO Agent | COO Agent | CEO |
| Platform compliance | Platform Agent | COO Agent | Legal Agent | CTO |
| Publish timing | Queue Adapter | COO Agent | Risk Agent | CEO |
| Rights/safety | Rights Agent | Legal Agent | Risk Agent | All |
| Budget allocation | Cost Agent | CEO | CTO + COO | All |
| Kill-switch trigger | Monitor | Risk Agent | All | CEO |
| ReconcileGPT role | — | Always TOOL | — | — |
