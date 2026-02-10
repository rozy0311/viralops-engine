"""
Meta-Agent Main Entry Point
============================
Autonomous agent that:
1. Loads all agents (built-in + generated)
2. Optionally spawns new agent children
3. Runs all scheduled jobs
4. Generates reports
5. Commits and pushes changes
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# Add parent dir to path for imports
AGENT_DIR = Path(__file__).parent
sys.path.insert(0, str(AGENT_DIR))

from agent_factory import load_all_agents, get_factory
from task_runner import TaskRunner, run_jobs
from spawn_agent import spawn_child_agent


def git_commit_and_push(message: str) -> bool:
    """Commit and push changes to GitHub."""
    try:
        repo_root = AGENT_DIR.parent.parent

        # Stage allowed paths
        allowed_paths = [
            "artifacts/",
            "blogs/",
            "logs/",
            "REPORT.md",
            ".github/agent/agents/generated/",
            ".github/agent/reports/",
        ]

        for path in allowed_paths:
            subprocess.run(["git", "add", path], cwd=repo_root, capture_output=True)

        # Check if there are changes
        result = subprocess.run(
            ["git", "diff", "--staged", "--quiet"], cwd=repo_root, capture_output=True
        )

        if result.returncode == 0:
            print("[MAIN] No changes to commit")
            return True

        # Commit
        subprocess.run(
            ["git", "commit", "-m", message], cwd=repo_root, capture_output=True
        )

        # Push
        result = subprocess.run(
            ["git", "push", "origin", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("[MAIN] Changes committed and pushed successfully")
            return True
        else:
            print(f"[MAIN] Push failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"[MAIN] Git error: {e}")
        return False


def save_report(report: str) -> Path:
    """Save the run report to file."""
    reports_dir = AGENT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    report_file = reports_dir / f"run-{timestamp}.md"

    report_file.write_text(report, encoding="utf-8")
    print(f"[MAIN] Report saved to {report_file}")

    return report_file


def main():
    """Main entry point for the autonomous agent."""
    print("=" * 60)
    print("META-AGENT - Autonomous GitHub Worker")
    print(f"Started at: {datetime.utcnow().isoformat()}")
    print("=" * 60)

    # Check environment
    dry_run = os.environ.get("DRY_RUN", "").lower() == "true"
    if dry_run:
        print("[MAIN] DRY RUN MODE - No commits will be made")

    # Step 1: Load all agents
    print("\n[MAIN] Step 1: Loading agents...")
    agents = load_all_agents()
    print(f"[MAIN] Loaded {len(agents)} agents: {list(agents.keys())}")

    # Step 2: Try to spawn child agent (if enabled)
    print("\n[MAIN] Step 2: Checking agent spawning...")
    new_agent_path = spawn_child_agent()
    if new_agent_path:
        print(f"[MAIN] Spawned new agent: {new_agent_path}")
        # Reload agents to include new one
        agents = load_all_agents()
    else:
        print("[MAIN] No new agents spawned")

    # Step 3: Run all jobs
    print("\n[MAIN] Step 3: Running jobs...")
    runner = TaskRunner()
    summary = runner.run_all_jobs()

    # Step 4: Generate report
    print("\n[MAIN] Step 4: Generating report...")
    report = runner.generate_report()
    report_path = save_report(report)

    # Step 5: Commit and push (if not dry run)
    if not dry_run:
        print("\n[MAIN] Step 5: Committing changes...")
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        success_count = summary.get("success", 0)
        failure_count = summary.get("failure", 0)

        commit_msg = f"🤖 Auto-run: {timestamp} | {success_count}✓ {failure_count}✗"
        git_commit_and_push(commit_msg)
    else:
        print("\n[MAIN] Step 5: Skipping commit (dry run)")

    # Summary
    print("\n" + "=" * 60)
    print("META-AGENT RUN COMPLETE")
    print(f"  Total jobs: {summary.get('total', 0)}")
    print(f"  Success: {summary.get('success', 0)}")
    print(f"  Failure: {summary.get('failure', 0)}")
    print(f"  Skipped: {summary.get('skipped', 0)}")
    print(f"Finished at: {datetime.utcnow().isoformat()}")
    print("=" * 60)

    # Exit with error if any failures
    if summary.get("failure", 0) > 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
