"""ViralOps Engine — LangGraph State Definitions"""

from __future__ import annotations
from typing import Optional, TypedDict

from core.models import (
    ContentPack,
    ComplianceCheck,
    RightsCheck,
    RiskAssessment,
    CostEstimate,
    ReconcileDecision,
    PublishResult,
    EngagementMetrics,
    AccountHealth,
)


class ViralOpsState(TypedDict):
    """
    LangGraph state for the ViralOps Engine workflow.
    
    Flow:
    CEO Input → Orchestrator → [Specialists PARALLEL] → ReconcileGPT 
    → Human Review → Queue Adapter → Monitor → Complete / Re-plan
    """

    # ─── Input (CEO/User) ───────────────────────
    niche_id: str
    niche_name: str
    tone: str                               # e.g., "conversational", "professional"
    audience: str                           # e.g., "Fitness moms 25-40"
    brand_tags: list[str]                   # e.g., ["therike", "sustainable"]
    target_platforms: list[str]             # e.g., ["tiktok", "instagram_reels"]
    publish_mode: str                       # draft | review | queue | auto
    blog_urls: list[str]                    # Optional: blog URLs to repurpose

    # ─── Memory (History/Context) ───────────────
    past_performance: Optional[dict]        # Previous niche performance data
    failed_posts_history: list[dict]        # Recent failures for learning
    niche_run_count: int                    # How many times this niche has been run

    # ─── Content Factory Output (CTO) ──────────
    content_pack: Optional[ContentPack]
    content_generation_log: str

    # ─── Specialist Agent Outputs (PARALLEL) ────
    compliance_checks: dict[str, ComplianceCheck]   # Per-platform compliance
    rights_check: Optional[RightsCheck]
    risk_assessment: Optional[RiskAssessment]
    cost_estimate: Optional[CostEstimate]

    # ─── ReconcileGPT Output ────────────────────
    reconcile_decision: Optional[ReconcileDecision]

    # ─── Human Review ───────────────────────────
    human_approved: Optional[bool]
    human_feedback: Optional[str]
    human_edits: Optional[dict]             # Edits made by human reviewer

    # ─── Queue & Publishing ─────────────────────
    queue_items: list[dict]
    publish_results: list[PublishResult]

    # ─── Monitoring ─────────────────────────────
    engagement_metrics: list[EngagementMetrics]
    account_health: dict[str, AccountHealth]

    # ─── Control Flow ───────────────────────────
    replan_count: int                       # Current re-plan iteration (max 3)
    kill_switch_active: bool                # Emergency stop flag
    current_step: str                       # Current workflow step
    errors: list[str]                       # Error messages collected
    audit_log: list[dict]                   # Audit trail
