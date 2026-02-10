"""
ViralOps Engine — LangGraph Workflow (EMADS-PR v1.0)

Flow:
  CEO Input → Orchestrator → [CTO + COO + Legal + Risk + Cost] PARALLEL
  → ReconcileGPT → Human Review → Execute (Publish) → Monitor
  → Re-plan (max 3 loops)

Training 12-LangGraph-Implementation applied.
"""
import os
import json
from typing import TypedDict, Annotated, Literal, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import structlog

logger = structlog.get_logger()


# ── State Definition ──
class ViralOpsState(TypedDict, total=False):
    """Complete state for the ViralOps pipeline."""
    # Input
    niche_config: dict
    topic: Optional[str]
    platforms: list[str]
    publish_mode: str  # "immediate" | "scheduled" | "draft"

    # RSS content (if repurposing)
    rss_content: Optional[dict]

    # Budget
    budget_remaining_pct: float

    # Agent outputs
    content_pack: dict
    content_factory_status: str

    compliance_result: dict
    rights_result: dict
    risk_result: dict
    cost_result: dict
    cost_status: str

    reconcile_result: dict
    reconcile_status: str

    # Human review
    human_approved: bool
    human_feedback: str

    # Execution
    publish_results: list[dict]
    monitor_data: dict

    # Control
    replan_count: int
    kill_switch: bool
    errors: list[str]
    timestamps: dict


# ── Node Functions ──

def orchestrator_node(state: ViralOpsState) -> ViralOpsState:
    """Route and initialize the pipeline."""
    state.setdefault("replan_count", 0)
    state.setdefault("errors", [])
    state.setdefault("kill_switch", False)
    state.setdefault("budget_remaining_pct", 100.0)
    state.setdefault("timestamps", {})
    state["timestamps"]["orchestrator_start"] = datetime.utcnow().isoformat()

    logger.info("graph.orchestrator", replan=state.get("replan_count", 0))
    return state


def content_factory_node(state: ViralOpsState) -> ViralOpsState:
    """CTO Agent — Generate content."""
    from agents.content_factory import generate_content_pack
    return generate_content_pack(state)


def compliance_node(state: ViralOpsState) -> ViralOpsState:
    """COO Agent — Platform compliance check."""
    from agents.platform_compliance import check_compliance
    return check_compliance(state)


def rights_safety_node(state: ViralOpsState) -> ViralOpsState:
    """Legal Agent — Rights & safety check."""
    from agents.rights_safety import check_rights
    return check_rights(state)


def risk_health_node(state: ViralOpsState) -> ViralOpsState:
    """Risk Agent — Risk assessment."""
    from agents.risk_health import assess_risk
    return assess_risk(state)


def cost_node(state: ViralOpsState) -> ViralOpsState:
    """Cost Agent — Budget check."""
    from agents.cost_agent import cost_check
    return cost_check(state)


def reconcile_node(state: ViralOpsState) -> ViralOpsState:
    """ReconcileGPT — Merge and analyze trade-offs."""
    from agents.reconcile_gpt import reconcile_decision
    return reconcile_decision(state)


def human_review_node(state: ViralOpsState) -> ViralOpsState:
    """Human Review Gate — Wait for approval."""
    reconcile = state.get("reconcile_result", {})
    action = reconcile.get("action", "HUMAN_REVIEW")

    if action == "AUTO_APPROVE":
        state["human_approved"] = True
        state["human_feedback"] = "Auto-approved (risk < 4, no blockers)"
        logger.info("graph.human_review.auto_approved")
    elif action == "BLOCK":
        state["human_approved"] = False
        state["human_feedback"] = f"BLOCKED: {reconcile.get('blockers', [])}"
        logger.warning("graph.human_review.blocked", blockers=reconcile.get("blockers"))
    else:
        # In production, this would pause and wait for human input
        # For now, auto-approve with warning
        state["human_approved"] = True
        state["human_feedback"] = "Pending human review (auto-approved in dev mode)"
        logger.info("graph.human_review.pending")

    return state


def publish_node(state: ViralOpsState) -> ViralOpsState:
    """Execute — Publish to platforms."""
    if not state.get("human_approved", False):
        state["publish_results"] = []
        logger.warning("graph.publish.skipped", reason="not approved")
        return state

    # In web mode, publishing is handled by the web API
    # In CLI mode, use the publisher registry
    content_pack = state.get("content_pack", {})
    platforms = state.get("platforms", [])
    state["publish_results"] = [{
        "platform": p,
        "status": "queued",
        "content_title": content_pack.get("title", ""),
    } for p in platforms]

    state["timestamps"]["publish_end"] = datetime.utcnow().isoformat()
    logger.info("graph.publish.queued", platforms=platforms)
    return state


def monitor_node(state: ViralOpsState) -> ViralOpsState:
    """Monitor — Track results and decide re-plan."""
    state["monitor_data"] = {
        "publish_count": len(state.get("publish_results", [])),
        "replan_count": state.get("replan_count", 0),
        "timestamp": datetime.utcnow().isoformat(),
    }
    logger.info("graph.monitor", data=state["monitor_data"])
    return state


# ── Routing Functions ──

def should_continue_after_review(state: ViralOpsState) -> str:
    """Route after human review."""
    if state.get("kill_switch"):
        return "end"
    if not state.get("human_approved", False):
        replan = state.get("replan_count", 0)
        if replan >= 3:
            logger.warning("graph.max_replan", count=replan)
            return "end"
        return "replan"
    return "publish"


def should_replan(state: ViralOpsState) -> str:
    """Check if re-planning is needed after monitoring."""
    if state.get("kill_switch"):
        return "end"
    # Check if any publish failed
    results = state.get("publish_results", [])
    failed = [r for r in results if r.get("status") == "failed"]
    replan = state.get("replan_count", 0)

    if failed and replan < 3:
        state["replan_count"] = replan + 1
        return "replan"
    return "end"


# ── Build Graph ──

def build_graph():
    """Build the EMADS-PR LangGraph workflow."""
    graph = StateGraph(ViralOpsState)

    # Add nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("content_factory", content_factory_node)
    graph.add_node("compliance", compliance_node)
    graph.add_node("rights_safety", rights_safety_node)
    graph.add_node("risk_health", risk_health_node)
    graph.add_node("cost", cost_node)
    graph.add_node("reconcile", reconcile_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("publish", publish_node)
    graph.add_node("monitor", monitor_node)

    # Set entry point
    graph.set_entry_point("orchestrator")

    # Orchestrator → Content Factory
    graph.add_edge("orchestrator", "content_factory")

    # Content Factory → Parallel specialists
    # (LangGraph handles parallel via fan-out)
    graph.add_edge("content_factory", "compliance")
    graph.add_edge("content_factory", "rights_safety")
    graph.add_edge("content_factory", "risk_health")
    graph.add_edge("content_factory", "cost")

    # Parallel specialists → ReconcileGPT (fan-in)
    graph.add_edge("compliance", "reconcile")
    graph.add_edge("rights_safety", "reconcile")
    graph.add_edge("risk_health", "reconcile")
    graph.add_edge("cost", "reconcile")

    # ReconcileGPT → Human Review
    graph.add_edge("reconcile", "human_review")

    # Human Review → conditional
    graph.add_conditional_edges(
        "human_review",
        should_continue_after_review,
        {
            "publish": "publish",
            "replan": "orchestrator",
            "end": END,
        }
    )

    # Publish → Monitor
    graph.add_edge("publish", "monitor")

    # Monitor → conditional
    graph.add_conditional_edges(
        "monitor",
        should_replan,
        {
            "replan": "orchestrator",
            "end": END,
        }
    )

    return graph


def get_compiled_graph():
    """Get compiled graph with memory checkpoint."""
    graph = build_graph()
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ── Quick test ──
if __name__ == "__main__":
    import yaml

    # Load niche config
    config_path = os.path.join(os.path.dirname(__file__), "config", "niches.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            niches = yaml.safe_load(f)
        niche = niches.get("niches", {}).get("plant_based_raw", {})
    else:
        niche = {"name": "test_niche"}

    app = get_compiled_graph()
    result = app.invoke(
        {
            "niche_config": niche,
            "topic": "raw almonds benefits",
            "platforms": ["reddit", "medium"],
            "publish_mode": "draft",
        },
        config={"configurable": {"thread_id": "test-1"}},
    )
    print(json.dumps(result.get("content_pack", {}), indent=2))
    print(f"\nReconcile: {result.get('reconcile_result', {}).get('summary', 'N/A')}")
