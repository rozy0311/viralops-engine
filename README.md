# 🚀 Agent Multi-Channel Scheduler Content Factory — ViralOps Engine

> **1 micro-niche → content pack chuẩn (bài + caption + Hashtag Matrix 5 lớp + ảnh) → queue → schedule/auto-post đa kênh → production-safe.**

---

## 📊 EMADS-PR Analysis

| Component | Score | Detail |
|-----------|-------|--------|
| Data Sources | 4 | RSS, Blog URLs, TikTok MCP scrape, YouTube feeds, Micro-niche DB |
| Logic Complexity | 4 | Content transform, dedup, multi-platform format, A/B test |
| Integration Points | 4 | TikTok, IG, FB, Pinterest, LinkedIn, YouTube (Direct OAuth) |
| **Automation Score** | **12/12 🔴** | Multi-stakeholder, phased rollout required |
| **Risk Level** | **🔴 HIGH** | One-way door (ban/flag = mất kênh) |

### Decision
- **Option A (Selected)**: Production hóa guardrails trước, pilot 1-2 kênh rủi ro thấp, rồi scale
- **Confidence**: 0.78
- **Human Review**: BẮT BUỘC cho risk score ≥ 4

---

## 🏗️ Architecture (EMADS-PR v1.0)

```
CEO/User Input (micro-niche + tone + audience)
    ↓
┌─────────────────────────────┐
│   Orchestrator Agent        │ ← Memory Agent (niche history, past performance)
│   (Route + Coordinate)      │
└─────────────────────────────┘
    │ (parallel dispatch)
    ├── Content Factory Agent (CTO)     → title + body + caption + hashtag matrix
    ├── Image/Video Agent (CTO)         → prompt 9:16 + generate + alt-text
    ├── Platform Compliance Agent (COO) → character limits, format rules, ToS check
    ├── Rights & Safety Agent (Legal)   → source rights, brand safety, PII guard
    ├── Risk & Health Agent (Risk)      → account health, ban-risk, duplicate score
    └── Cost Agent (Cost)               → API credits, token budget, ROI projection
    │
    ↓ (merge all outputs)
┌─────────────────────────────┐
│   ReconcileGPT              │ → Analyze trade-offs, score content pack
│   (Decision Engine)         │ → Flag risky content, recommend publish mode
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│   Human Review Gate         │ → Approve / Edit / Reject
│   (Governance)              │ → Required for risk ≥ 4
└─────────────────────────────┘
    ↓ (approved)
┌─────────────────────────────┐
│   Queue Adapter             │ → Normalize payload per platform
│   (Publisher)               │ → Rate-limit + Retry + DLQ
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│   Monitor / Validator       │ → Engagement tracker + Account health
│   (Observability)           │ → Kill-switch + Alert + Weekly report
└─────────────────────────────┘
    ↓
  ✅ Complete  OR  🔄 Re-plan (max 3 loops)
```

---

## 📁 Project Structure

```
viralops-engine/
├── README.md                          # This file
├── ARCHITECTURE.md                    # Deep-dive architecture doc
├── config/
│   ├── platforms.yaml                 # Platform specs (char limits, formats, rates)
│   ├── niches.yaml                    # Micro-niche database
│   ├── guardrails.yaml                # Safety rules, kill-switch thresholds
│   └── cost_budget.yaml               # Budget allocation per tier
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py                # Main orchestrator/supervisor
│   ├── content_factory.py             # Content generation agent
│   ├── image_video_agent.py           # Image/video generation
│   ├── platform_compliance.py         # Platform-specific formatting
│   ├── rights_safety.py               # Rights check, brand safety
│   ├── risk_health.py                 # Account health monitoring
│   ├── cost_agent.py                  # Budget tracking
│   └── reconcile_gpt.py              # ReconcileGPT decision engine
├── core/
│   ├── __init__.py
│   ├── state.py                       # LangGraph state definitions
│   ├── models.py                      # Data models (ContentPack, etc.)
│   ├── queue_adapter.py               # Queue + retry + DLQ
│   ├── dedup.py                       # Deduplication engine
│   ├── rate_limiter.py                # Per-platform rate limiting
│   └── kill_switch.py                 # Circuit breaker / kill-switch
├── integrations/
│   ├── __init__.py
│   ├── platform_publisher.py          # Direct OAuth publisher (all 8 platforms)
│   └── trend_researcher.py            # Built-in trend research (replaces MCP)
├── monitoring/
│   ├── __init__.py
│   ├── dashboard.py                   # Metrics dashboard
│   ├── account_health.py              # Account health monitor
│   ├── engagement_tracker.py          # Views/hour, retention, CTR
│   └── alerting.py                    # Alert rules + notifications
├── hashtags/
│   ├── __init__.py
│   ├── matrix_5layer.py               # 5-layer hashtag matrix generator
│   └── niche_hashtags.json            # Pre-built niche hashtag database
├── templates/
│   ├── caption_templates.json         # Caption templates per platform
│   └── content_transforms.json        # Transformation rules
├── tests/
│   ├── test_content_factory.py
│   ├── test_dedup.py
│   ├── test_rate_limiter.py
│   ├── test_queue.py
│   └── test_kill_switch.py
├── graph.py                           # LangGraph workflow definition
├── main.py                            # Entry point
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure platforms & credentials
cp config/platforms.yaml.example config/platforms.yaml
# Edit with your API keys (NEVER plaintext in repo)

# 3. Run pilot (single niche, single platform)
python main.py --niche "raw-almonds-vegan-protein" --platform twitter --mode draft

# 4. Run with human review
python main.py --niche "chia-seed-puddings" --platform all --mode review

# 5. Run production (with kill-switch)
python main.py --niche "hemp-hearts-salads" --platform all --mode queue --kill-switch
```

---

## ⚙️ Publish Modes

| Mode | Description | Risk |
|------|-------------|------|
| `draft` | Generate content, save locally, no posting | 🟢 None |
| `review` | Generate + send to human for approval | 🟢 Low |
| `queue` | Approved content → scheduler queue | 🟡 Medium |
| `auto` | Queue + auto-post (pilot only, kill-switch required) | 🔴 High |

---

## 🔒 Production Rails (MUST-HAVE)

1. **Dedup + Idempotency** — Hash video + semantic similarity before post
2. **Rate-limit per account/platform** — Throttle profile + jitter
3. **Circuit Breaker** — Error/flag spike → auto-stop
4. **Rights/Policy Gate** — Source rights check before publish
5. **Kill-switch** — Manual or auto trigger to halt all operations
6. **DLQ (Dead Letter Queue)** — Failed posts retry with backoff
7. **Audit Log** — Every action logged for compliance

---

## 📅 30/60/90 Plan

### 30 Days — Foundation
- [ ] Integration Matrix (platform-by-platform: auth, limits, errors)
- [ ] Content Policy (transform ≥70%, spacing 24h+)
- [ ] Reliability stack (queue, retry/backoff, dedup, idempotency)
- [ ] Observability (log, metrics, alerts)
- [ ] Pilot: 1-2 low-risk platforms (Twitter/Reddit)

### 60 Days — MVP
- [ ] A/B test caption/hashtag with real KPIs
- [ ] Multi-platform distribution (add Pinterest, LinkedIn)
- [ ] Account health monitor + auto kill-switch
- [ ] Human approval workflow via Dashboard/Slack

### 90 Days — Scale
- [ ] Multi-tenant workspaces (agency mode)
- [ ] Unified analytics dashboard
- [ ] Cost guardrails + billing per client
- [ ] On-call playbook + incident response

---

## ⚠️ Kill-Switch Thresholds

| Signal | Threshold | Action |
|--------|-----------|--------|
| Account restriction/flag | 1 occurrence | STOP platform immediately |
| Reach drop | >30% over 7 days | Reduce frequency, increase transform |
| Upload error rate | >3%/day | STOP, investigate pipeline |
| Duplicate detection | >5% posts flagged | STOP, review content policy |
| Budget burn | >80% monthly budget | Switch to cheaper model tier |

---

## 📜 License

Internal use only. PR-only workflow required for all changes.
