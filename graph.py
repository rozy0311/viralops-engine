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
from typing import TypedDict, Annotated, Literal, Optional, Any
from datetime import datetime, timezone
from operator import __or__

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import structlog

logger = structlog.get_logger()


# ── Reducers for fan-in ──

def _last_value(a, b):
    """Keep the latest non-None value. Used for keys written by one node at a time."""
    return b if b is not None else a


def _merge_dict(a: dict, b: dict) -> dict:
    """Merge two dicts, keeping the latest values."""
    if not a:
        return b or {}
    if not b:
        return a or {}
    merged = {**a, **b}
    return merged


def _merge_list(a: list, b: list) -> list:
    """Merge lists — append new items."""
    if not a:
        return b or []
    if not b:
        return a or []
    return a + b


# ── State Definition ──
class ViralOpsState(TypedDict, total=False):
    """Complete state for the ViralOps pipeline.
    
    Annotated keys use reducers to handle fan-in from parallel nodes.
    Without reducers, LangGraph throws InvalidUpdateError when 4 parallel
    agents all return the same key.
    """
    # Input — written once, read by many parallel nodes
    niche_config: Annotated[dict, _last_value]
    topic: Annotated[Optional[str], _last_value]
    platforms: Annotated[list[str], _last_value]
    publish_mode: Annotated[str, _last_value]

    # RSS content (if repurposing)
    rss_content: Annotated[Optional[dict], _last_value]

    # Budget
    budget_remaining_pct: Annotated[float, _last_value]

    # Agent outputs — each written by ONE specific agent
    content_pack: Annotated[dict, _last_value]
    content_factory_status: Annotated[str, _last_value]

    compliance_result: Annotated[dict, _last_value]
    rights_result: Annotated[dict, _last_value]
    risk_result: Annotated[dict, _last_value]
    cost_result: Annotated[dict, _last_value]
    cost_status: Annotated[str, _last_value]

    reconcile_result: Annotated[dict, _last_value]
    reconcile_status: Annotated[str, _last_value]

    # Human review
    human_approved: Annotated[bool, _last_value]
    human_feedback: Annotated[str, _last_value]

    # Execution
    publish_results: Annotated[list[dict], _last_value]
    monitor_data: Annotated[dict, _last_value]

    # Control
    replan_count: Annotated[int, _last_value]
    kill_switch: Annotated[bool, _last_value]
    errors: Annotated[list[str], _merge_list]
    timestamps: Annotated[dict, _merge_dict]


# ── Node Functions ──

def orchestrator_node(state: ViralOpsState) -> ViralOpsState:
    """Route and initialize the pipeline."""
    replan = state.get("replan_count", 0)
    updates = {
        "timestamps": {"orchestrator_start": datetime.now(timezone.utc).isoformat()},
    }
    # Only set defaults on first run (replan_count == 0)
    if replan == 0:
        updates["replan_count"] = 0
        updates["errors"] = []
        updates["kill_switch"] = False
        if not state.get("budget_remaining_pct"):
            updates["budget_remaining_pct"] = 100.0

    logger.info("graph.orchestrator", replan=replan)
    return updates


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
    is_production = os.environ.get("VIRALOPS_ENV", "development") == "production"

    if action == "AUTO_APPROVE":
        state["human_approved"] = True
        state["human_feedback"] = "Auto-approved (risk < 4, no blockers)"
        logger.info("graph.human_review.auto_approved")
    elif action == "BLOCK":
        state["human_approved"] = False
        state["human_feedback"] = f"BLOCKED: {reconcile.get('blockers', [])}"
        logger.warning("graph.human_review.blocked", blockers=reconcile.get("blockers"))
    else:
        if is_production:
            # In production: BLOCK until real human reviews
            state["human_approved"] = False
            state["human_feedback"] = "PENDING: Requires human review in production mode"
            logger.warning("graph.human_review.waiting_human", env="production")
        else:
            # Dev mode: auto-approve with warning
            state["human_approved"] = True
            state["human_feedback"] = "Pending human review (auto-approved in dev mode)"
            logger.info("graph.human_review.dev_auto_approved")

    return state


def publish_node(state: ViralOpsState) -> ViralOpsState:
    """Execute — Publish to platforms via scheduler or direct publishers."""
    if not state.get("human_approved", False):
        state["publish_results"] = []
        logger.warning("graph.publish.skipped", reason="not approved")
        return state

    content_pack = state.get("content_pack", {})
    platforms = state.get("platforms", [])
    publish_mode = state.get("publish_mode", "draft")
    results = []

    for platform in platforms:
        entry = {
            "platform": platform,
            "content_title": content_pack.get("title", ""),
        }
        try:
            if publish_mode == "draft":
                # Draft mode — mark as published (no actual post)
                entry["status"] = "published"
                entry["detail"] = "Draft mode — saved locally"
            elif publish_mode == "scheduled":
                # Use the scheduler to queue for later
                from core.scheduler import PublishScheduler
                scheduler = PublishScheduler()
                adapted = content_pack.get("adapted", {}).get(platform, {})
                scheduler.schedule(
                    platform=platform,
                    content=adapted or content_pack,
                    mode="scheduled",
                )
                entry["status"] = "published"
                entry["detail"] = "Scheduled via Scheduler"
            else:
                # Immediate — attempt real publishing
                from core.scheduler import PublishScheduler
                scheduler = PublishScheduler()
                adapted = content_pack.get("adapted", {}).get(platform, {})
                result = scheduler.publish_now(
                    platform=platform,
                    content=adapted or content_pack,
                )
                # publish_now returns dict with 'success' key
                if isinstance(result, dict) and result.get("success"):
                    entry["status"] = "published"
                    entry["detail"] = result.get("message", "Published")
                elif isinstance(result, dict) and not result.get("success"):
                    entry["status"] = "failed"
                    entry["detail"] = result.get("error", "Unknown error")
                else:
                    entry["status"] = "published"
                    entry["detail"] = "Published (no detail)"
        except Exception as e:
            entry["status"] = "failed"
            entry["detail"] = str(e)
            logger.error("graph.publish.error", platform=platform, error=str(e))

        results.append(entry)

    state["publish_results"] = results
    state["timestamps"]["publish_end"] = datetime.now(timezone.utc).isoformat()

    published = sum(1 for r in results if r["status"] == "published")
    failed = sum(1 for r in results if r["status"] == "failed")
    logger.info("graph.publish.done", published=published, failed=failed, total=len(results))
    return state


def monitor_node(state: ViralOpsState) -> ViralOpsState:
    """Monitor — Track results and decide re-plan."""
    results = state.get("publish_results", [])
    published = [r for r in results if r.get("status") == "published"]
    failed = [r for r in results if r.get("status") == "failed"]

    state["monitor_data"] = {
        "publish_count": len(results),
        "published_count": len(published),
        "failed_count": len(failed),
        "failed_platforms": [r["platform"] for r in failed],
        "replan_count": state.get("replan_count", 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("graph.monitor", published=len(published), failed=len(failed))
    return state


def replan_node(state: ViralOpsState) -> ViralOpsState:
    """Re-plan node — Increment replan counter and prepare for retry.
    
    State mutation MUST happen in a node, not in a routing function.
    """
    replan = state.get("replan_count", 0) + 1
    state["replan_count"] = replan

    failed_platforms = state.get("monitor_data", {}).get("failed_platforms", [])
    state["errors"].append(
        f"Replan #{replan}: retrying failed platforms {failed_platforms}"
    )
    logger.warning("graph.replan", attempt=replan, failed_platforms=failed_platforms)
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
    """Check if re-planning is needed after monitoring.
    
    PURE routing function — no state mutation here.
    State changes happen in replan_node.
    """
    if state.get("kill_switch"):
        return "end"
    # Check if any publish failed
    monitor = state.get("monitor_data", {})
    failed_count = monitor.get("failed_count", 0)
    replan = state.get("replan_count", 0)

    if failed_count > 0 and replan < 3:
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
    graph.add_node("replan", replan_node)

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
            "replan": "replan",
            "end": END,
        }
    )

    # Publish → Monitor
    graph.add_edge("publish", "monitor")

    # Monitor → conditional (replan goes through replan_node first)
    graph.add_conditional_edges(
        "monitor",
        should_replan,
        {
            "replan": "replan",
            "end": END,
        }
    )

    # Replan node → back to Orchestrator for retry
    graph.add_edge("replan", "orchestrator")

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
