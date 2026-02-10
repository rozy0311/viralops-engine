"""
ViralOps Engine — Main Entry Point
CLI + Web server for the multi-agent content scheduler.

Usage:
  python main.py --niche raw-almonds --platforms reddit,medium --mode review
  python main.py --niche chickpeas --platforms all --mode draft
  python main.py --kill-switch status
  python main.py --dashboard
  python main.py --web  # Start the web dashboard
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("viralops.main")

ALL_PLATFORMS = [
    "reddit", "medium", "tumblr", "shopify_blog",
]

SOCIALBEE_PLATFORMS = [
    "tiktok", "instagram_reels", "instagram_feed",
    "facebook_reels", "youtube_shorts",
    "pinterest", "linkedin", "twitter_x",
    "threads", "bluesky", "google_business",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ViralOps Engine — Multi-Channel Content Scheduler (EMADS-PR v1.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--niche", "-n", type=str, help="Niche ID (e.g., plant_based_raw)")
    parser.add_argument("--platforms", "-p", type=str, default="reddit,medium",
                        help="Comma-separated platforms or 'all'")
    parser.add_argument("--mode", "-m", type=str, choices=["draft", "review", "queue", "auto"],
                        default="review", help="Publish mode")
    parser.add_argument("--tone", type=str, default="conversational", help="Content tone")
    parser.add_argument("--audience", type=str, default="health-conscious millennials", help="Target audience")
    parser.add_argument("--kill-switch", type=str, choices=["status", "reset"], help="Kill-switch command")
    parser.add_argument("--dashboard", action="store_true", help="Show CLI monitoring dashboard")
    parser.add_argument("--web", action="store_true", help="Start web dashboard (default: localhost:8000)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    parser.add_argument("--dry-run", action="store_true", help="Generate content without publishing")
    return parser.parse_args()


def resolve_platforms(platforms_str: str) -> list[str]:
    if platforms_str.lower() == "all":
        return ALL_PLATFORMS
    platforms = [p.strip() for p in platforms_str.split(",")]
    invalid = [p for p in platforms if p not in ALL_PLATFORMS + SOCIALBEE_PLATFORMS]
    if invalid:
        logger.warning("Unknown platforms: %s", invalid)
    # Only keep directly-supported platforms
    direct = [p for p in platforms if p in ALL_PLATFORMS]
    socialbee = [p for p in platforms if p in SOCIALBEE_PLATFORMS]
    if socialbee:
        logger.info("SocialBee-managed platforms (external): %s", socialbee)
    return direct


def run_kill_switch_command(command: str) -> None:
    from core.kill_switch import KillSwitch
    ks = KillSwitch()
    if command == "status":
        print(json.dumps(ks.get_status(), indent=2))
    elif command == "reset":
        ks.reset()
        print("Kill-switch reset.")


def run_dashboard() -> None:
    from monitoring.dashboard import Dashboard
    dashboard = Dashboard()
    print(dashboard.print_summary())


def start_web_server(port: int = 8000) -> None:
    """Start the FastAPI web dashboard."""
    import uvicorn
    logger.info("Starting web dashboard at http://localhost:%d", port)
    uvicorn.run("web.app:app", host="0.0.0.0", port=port, reload=True)


async def run_workflow(args: argparse.Namespace) -> None:
    """Run the EMADS-PR LangGraph workflow."""
    if not args.niche:
        logger.error("--niche is required. Example: --niche plant_based_raw")
        sys.exit(1)

    platforms = resolve_platforms(args.platforms)
    if not platforms:
        logger.error("No valid direct platforms. Use --platforms reddit,medium")
        sys.exit(1)

    logger.info("═══ ViralOps Engine Starting ═══")
    logger.info("Niche: %s | Platforms: %s | Mode: %s", args.niche, platforms, args.mode)

    # Load niche config
    niche_config = {"niche_id": args.niche, "tone": args.tone, "audience": args.audience}
    try:
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "config", "niches.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                niches_data = yaml.safe_load(f)
            nc = niches_data.get("niches", {}).get(args.niche, {})
            if nc:
                niche_config.update(nc)
    except Exception as e:
        logger.warning("Could not load niches.yaml: %s", e)

    initial_state = {
        "niche_config": niche_config,
        "topic": None,
        "platforms": platforms,
        "publish_mode": "draft" if args.dry_run else args.mode,
        "budget_remaining_pct": 100.0,
        "replan_count": 0,
        "kill_switch": False,
        "errors": [],
    }

    try:
        from graph import get_compiled_graph
        app = get_compiled_graph()
        config = {"configurable": {"thread_id": f"viralops-{args.niche}-{datetime.now():%Y%m%d%H%M}"}}

        logger.info("Running LangGraph workflow...")
        final_state = None
        async for event in app.astream(initial_state, config=config):
            for node_name, node_output in event.items():
                logger.info("Node [%s] completed", node_name)
                final_state = node_output

        if final_state:
            logger.info("═══ Workflow Complete ═══")
            cp = final_state.get("content_pack", {})
            logger.info("Content: %s", cp.get("title", "N/A"))
            rec = final_state.get("reconcile_result", {})
            logger.info("Reconcile: action=%s, risk=%s, score=%s/12",
                rec.get("action", "N/A"),
                rec.get("risk_score", "N/A"),
                rec.get("automation_score", "N/A"))
            pub = final_state.get("publish_results", [])
            if pub:
                for r in pub:
                    logger.info("Published: %s → %s", r.get("platform"), r.get("status"))

    except ImportError as e:
        logger.warning("LangGraph not available. Running standalone. (%s)", e)
        await run_standalone(initial_state)


async def run_standalone(state: dict) -> None:
    """Fallback: run agents sequentially without LangGraph."""
    from agents.content_factory import generate_content_pack
    from agents.platform_compliance import check_compliance
    from agents.rights_safety import check_rights
    from agents.risk_health import assess_risk
    from agents.cost_agent import cost_check
    from agents.reconcile_gpt import reconcile_decision

    logger.info("Running in standalone mode (no LangGraph)")

    steps = [
        ("Content Factory", generate_content_pack),
        ("Platform Compliance", check_compliance),
        ("Rights & Safety", check_rights),
        ("Risk & Health", assess_risk),
        ("Cost Agent", cost_check),
        ("ReconcileGPT", reconcile_decision),
    ]

    for name, fn in steps:
        try:
            result = fn(state)
            state.update(result)
            logger.info("✅ %s complete", name)
        except Exception as e:
            logger.error("❌ %s failed: %s", name, e)
            state.setdefault("errors", []).append(f"{name}: {e}")

    # Print summary
    rec = state.get("reconcile_result", {})
    if rec:
        print(f"\n{'='*50}")
        print(f"Action: {rec.get('action', 'N/A')}")
        print(f"Risk Score: {rec.get('risk_score', 'N/A')}")
        print(f"Automation Score: {rec.get('automation_score', 'N/A')}/12")
        print(f"Summary: {rec.get('summary', 'N/A')}")
        if rec.get("trade_offs"):
            print(f"\nTrade-offs:")
            for tf in rec["trade_offs"]:
                print(f"  • {tf}")
        print(f"{'='*50}")


def main():
    args = parse_args()

    if args.kill_switch:
        run_kill_switch_command(args.kill_switch)
        return

    if args.dashboard:
        run_dashboard()
        return

    if args.web:
        start_web_server(args.port)
        return

    asyncio.run(run_workflow(args))


if __name__ == "__main__":
    main()
