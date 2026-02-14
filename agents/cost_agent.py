"""
Cost Agent — Budget & Cost Management (EMADS-PR v1.0)
Training 07-Cost-Aware-Planning: Budget tracking, model selection, cost alerts.
"""
import os
import json
from typing import Any
from datetime import datetime, date, timezone

import structlog

logger = structlog.get_logger()

# ── Budget tracking (persistent via JSON file) ──
BUDGET_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "budget_tracker.json")

# ── Model pricing (per 1K tokens) ──
MODEL_COSTS = {
    "gpt-4.1":      {"input": 0.002, "output": 0.008},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4o-mini":  {"input": 0.00015, "output": 0.0006},
    "gpt-4o":       {"input": 0.0025, "output": 0.01},
    "dall-e-3":     {"per_image": 0.04},
}


def _load_budget() -> dict:
    """Load budget from persistent storage."""
    try:
        if os.path.exists(BUDGET_FILE):
            with open(BUDGET_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {
        "daily_limit": float(os.getenv("DAILY_BUDGET_USD", "5.0")),
        "monthly_limit": float(os.getenv("MONTHLY_BUDGET_USD", "50.0")),
        "spent_today": 0.0,
        "spent_this_month": 0.0,
        "last_reset_date": str(date.today()),
        "last_reset_month": date.today().strftime("%Y-%m"),
        "history": [],
    }


def _save_budget(budget: dict):
    """Save budget to persistent storage."""
    try:
        os.makedirs(os.path.dirname(BUDGET_FILE), exist_ok=True)
        with open(BUDGET_FILE, 'w') as f:
            json.dump(budget, f, indent=2)
    except Exception as e:
        logger.error("cost_agent.save_error", error=str(e))


def _reset_if_needed(budget: dict) -> dict:
    """Reset daily/monthly counters if date changed."""
    today = str(date.today())
    this_month = date.today().strftime("%Y-%m")

    if budget.get("last_reset_date") != today:
        budget["spent_today"] = 0.0
        budget["last_reset_date"] = today

    if budget.get("last_reset_month") != this_month:
        budget["spent_this_month"] = 0.0
        budget["last_reset_month"] = this_month

    return budget


def get_budget_status() -> dict:
    """Get current budget status."""
    budget = _load_budget()
    budget = _reset_if_needed(budget)

    daily_remaining = budget["daily_limit"] - budget["spent_today"]
    monthly_remaining = budget["monthly_limit"] - budget["spent_this_month"]
    daily_pct = max(0, (daily_remaining / budget["daily_limit"]) * 100) if budget["daily_limit"] > 0 else 0
    monthly_pct = max(0, (monthly_remaining / budget["monthly_limit"]) * 100) if budget["monthly_limit"] > 0 else 0

    return {
        "daily_limit": budget["daily_limit"],
        "monthly_limit": budget["monthly_limit"],
        "spent_today": round(budget["spent_today"], 4),
        "spent_this_month": round(budget["spent_this_month"], 4),
        "daily_remaining": round(daily_remaining, 4),
        "monthly_remaining": round(monthly_remaining, 4),
        "daily_pct": round(daily_pct, 1),
        "monthly_pct": round(monthly_pct, 1),
        "recommended_model": _recommend_model(min(daily_pct, monthly_pct)),
    }


def _recommend_model(budget_pct: float) -> str:
    """
    Cost-aware model recommendation.
    Training 14-CHEAT-SHEET:
      Budget healthy (>50%)   → GPT-4.1
      Budget moderate (20-50%) → GPT-4.1-mini
      Budget tight (<20%)     → GPT-4o-mini
      Budget critical (<5%)   → Local/fallback
      Budget empty (0%)       → STOP
    """
    if budget_pct <= 0:
        return "STOP"
    elif budget_pct < 5:
        return "fallback"
    elif budget_pct < 20:
        return "gpt-4o-mini"
    elif budget_pct < 50:
        return "gpt-4.1-mini"
    else:
        return "gpt-4.1"


def record_spend(amount: float, model: str, operation: str):
    """Record a cost event."""
    budget = _load_budget()
    budget = _reset_if_needed(budget)
    budget["spent_today"] += amount
    budget["spent_this_month"] += amount
    budget["history"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "amount": amount,
        "model": model,
        "operation": operation,
    })
    # Keep only last 1000 entries
    budget["history"] = budget["history"][-1000:]
    _save_budget(budget)
    logger.info("cost_agent.spend", amount=f"${amount:.4f}", model=model, operation=operation)


def cost_check(state: dict) -> dict:
    """
    LangGraph node: Cost Agent.
    Evaluates cost implications and recommends model selection.
    """
    budget = get_budget_status()

    cost_score = 0
    if budget["daily_pct"] < 20:
        cost_score += 4
    elif budget["daily_pct"] < 50:
        cost_score += 2

    if budget["monthly_pct"] < 20:
        cost_score += 4
    elif budget["monthly_pct"] < 50:
        cost_score += 2

    platforms_count = len(state.get("platforms", []))
    if platforms_count > 4:
        cost_score += 2  # More platforms = more API calls

    state["cost_result"] = {
        "cost_score": min(10, cost_score),
        "budget_status": budget,
        "recommended_model": budget["recommended_model"],
        "estimated_cost": platforms_count * 0.01,  # Rough estimate
        "warnings": [] if budget["recommended_model"] != "STOP" else ["BUDGET EMPTY — Cannot proceed"],
    }
    state["cost_status"] = "completed"

    logger.info("cost_agent.done", score=cost_score, model=budget["recommended_model"], daily_pct=budget["daily_pct"])
    return state
