"""
ReconcileGPT — Decision Engine (EMADS-PR v1.0)
TOOL role: Analyzes trade-offs, does NOT make decisions.
Training 01: "ReconcileGPT = TOOL — Phân tích trade-offs, KHÔNG ra quyết định"
"""
import os
import json
from typing import Any
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger()


def _get_openai_client():
    """Lazy-load OpenAI client."""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return None
        return OpenAI(api_key=api_key)
    except ImportError:
        return None


def reconcile_decision(state: dict) -> dict:
    """
    LangGraph node: ReconcileGPT.
    Merges outputs from all specialist agents, analyzes trade-offs,
    and produces a structured recommendation for Human Review.

    DOES NOT DECIDE — only recommends.
    """
    content_pack = state.get("content_pack", {})
    compliance_result = state.get("compliance_result", {})
    rights_result = state.get("rights_result", {})
    risk_result = state.get("risk_result", {})
    cost_result = state.get("cost_result", {})

    logger.info("reconcile_gpt.start", content_title=content_pack.get("title", "")[:50])

    # ── Calculate composite scores ──
    risk_score = risk_result.get("risk_score", 0)
    cost_score = cost_result.get("cost_score", 0)
    compliance_pass = compliance_result.get("passed", True)
    rights_pass = rights_result.get("passed", True)

    # ── Automation Score (Training 03-Rosie: 0-12) ──
    data_sources = min(4, len(state.get("platforms", [])))
    logic_complexity = 2 if content_pack.get("_generated_by", "").startswith("gpt") else 1
    integration_points = min(4, len([p for p in state.get("platforms", []) if p in ("reddit", "medium", "tumblr", "shopify_blog")]))
    automation_score = data_sources + logic_complexity + integration_points

    # ── Decision logic ──
    blockers = []
    warnings = []

    if not compliance_pass:
        blockers.append("Content failed compliance check")
    if not rights_pass:
        blockers.append("Content has rights/safety issues")
    if risk_score >= 8:
        blockers.append(f"Risk score {risk_score}/10 exceeds threshold")

    if risk_score >= 4:
        warnings.append(f"Risk score {risk_score}/10 — Human Review REQUIRED (EMADS-PR rule)")
    if cost_score > 7:
        warnings.append(f"Cost score {cost_score}/10 — consider cheaper model")
    if automation_score >= 8:
        warnings.append(f"Automation score {automation_score}/12 — multi-stakeholder review needed")

    # ── Determine action ──
    if blockers:
        action = "BLOCK"
        confidence = 0.95
    elif risk_score >= 4 or automation_score >= 4:
        action = "HUMAN_REVIEW"
        confidence = 0.7
    else:
        action = "AUTO_APPROVE"
        confidence = 0.85

    # ── Try GPT analysis for complex cases ──
    gpt_analysis = None
    if action == "HUMAN_REVIEW" and risk_score >= 4:
        client = _get_openai_client()
        if client:
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",  # Cost-balanced for analysis
                    messages=[
                        {"role": "system", "content": """You are ReconcileGPT — a neutral decision analysis tool.
Analyze the trade-offs between publishing this content across platforms.
You do NOT make decisions — you only provide structured analysis.
Output JSON: {analysis, trade_offs: [{factor, pro, con}], recommendation, risk_factors, confidence}"""},
                        {"role": "user", "content": json.dumps({
                            "content": content_pack.get("title", ""),
                            "risk_score": risk_score,
                            "cost_score": cost_score,
                            "compliance": compliance_pass,
                            "rights": rights_pass,
                            "platforms": state.get("platforms", []),
                        })},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )
                gpt_analysis = json.loads(response.choices[0].message.content)
            except Exception as e:
                logger.warning("reconcile_gpt.gpt_error", error=str(e))

    # ── Build reconcile result ──
    reconcile_result = {
        "action": action,
        "confidence": confidence,
        "automation_score": automation_score,
        "risk_score": risk_score,
        "cost_score": cost_score,
        "blockers": blockers,
        "warnings": warnings,
        "gpt_analysis": gpt_analysis,
        "human_review_required": action != "AUTO_APPROVE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": f"Action={action} | Risk={risk_score}/10 | AutoScore={automation_score}/12 | Blockers={len(blockers)} | Warnings={len(warnings)}",
    }

    state["reconcile_result"] = reconcile_result
    state["reconcile_status"] = "completed"

    logger.info("reconcile_gpt.done",
                action=action,
                risk=risk_score,
                auto_score=automation_score,
                blockers=len(blockers))

    return state
