"""
Task Runner
===========
Loads jobs from config and runs them with appropriate agents.
Includes self-check and self-fix loop.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml

from agent_factory import load_all_agents, get_factory
from llm_client import get_llm_client


# Paths
AGENT_DIR = Path(__file__).parent
JOBS_PATH = AGENT_DIR / "config" / "jobs.yaml"
REPORTS_DIR = AGENT_DIR / "reports"


def load_jobs() -> List[Dict[str, Any]]:
    """Load job configurations from YAML."""
    if not JOBS_PATH.exists():
        return []

    with open(JOBS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data.get("jobs", [])


def run_check(command: str, cwd: str = None) -> Tuple[bool, str]:
    """
    Run a check command and return (success, output).
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=cwd or str(AGENT_DIR.parent.parent),  # repo root
        )

        success = result.returncode == 0
        output = result.stdout + result.stderr

        return success, output

    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def run_checks(checks: List[str]) -> Tuple[bool, List[str]]:
    """
    Run multiple check commands.

    Returns:
        (all_passed, list_of_failed_outputs)
    """
    failed_outputs = []

    for check in checks:
        success, output = run_check(check)
        if not success:
            failed_outputs.append(f"Check '{check}' failed:\n{output}")

    return len(failed_outputs) == 0, failed_outputs


class TaskRunner:
    """Runs jobs with self-check and self-fix capabilities."""

    def __init__(self):
        self.agents = load_all_agents()
        self.jobs = load_jobs()
        self.llm = get_llm_client()
        self.results: List[Dict[str, Any]] = []
        self.max_fix_attempts = 3

    def run_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single job with its assigned agent.

        Includes:
        1. Pre-checks
        2. Agent execution
        3. Post-checks
        4. Self-fix loop if needed
        """
        job_name = job.get("name", "unknown")
        agent_name = job.get("agent", "meta-agent")
        checks = job.get("checks", [])
        on_failure = job.get("on_failure", "skip")

        print(f"\n[RUNNER] Starting job: {job_name}")
        print(f"[RUNNER] Agent: {agent_name}")

        # Get agent
        agent = self.agents.get(agent_name)
        if not agent:
            print(f"[RUNNER] Agent not found: {agent_name}")
            return {
                "job": job_name,
                "status": "failure",
                "message": f"Agent not found: {agent_name}",
            }

        # Build context from job config
        context = job.get("context", {})

        # Run agent
        result = agent.execute(context)

        # Run post-checks
        if checks:
            print(f"[RUNNER] Running {len(checks)} checks...")
            checks_passed, failed_outputs = run_checks(checks)

            if not checks_passed:
                print(f"[RUNNER] Checks failed!")

                # Try self-fix
                if self.llm.is_available():
                    result = self._attempt_fix(job, agent, context, failed_outputs)
                else:
                    result["status"] = "failure"
                    result["check_failures"] = failed_outputs

        # Handle failure
        if result.get("status") == "failure":
            if on_failure == "skip":
                print(f"[RUNNER] Job failed, skipping (on_failure=skip)")
            elif on_failure == "log":
                print(f"[RUNNER] Job failed, logging (on_failure=log)")
                self._log_failure(job_name, result)
            # Could add more handlers: alert, rollback, etc.

        result["job"] = job_name
        self.results.append(result)

        return result

    def _attempt_fix(
        self,
        job: Dict[str, Any],
        agent,
        context: Dict[str, Any],
        failed_outputs: List[str],
    ) -> Dict[str, Any]:
        """
        Attempt to fix failures using LLM.

        Loop: generate fix -> run agent -> check -> repeat if needed
        """
        job_name = job.get("name", "unknown")
        checks = job.get("checks", [])

        for attempt in range(1, self.max_fix_attempts + 1):
            print(f"[RUNNER] Fix attempt {attempt}/{self.max_fix_attempts}")

            # Generate fix suggestion
            fix_prompt = f"""The job "{job_name}" failed with these errors:

{chr(10).join(failed_outputs)}

Context:
{context}

Suggest a fix or adjustment to make the checks pass.
Be specific about what needs to change."""

            suggestion = self.llm.generate_code(fix_prompt)

            if not suggestion:
                print(f"[RUNNER] LLM could not generate fix")
                break

            print(f"[RUNNER] LLM suggestion: {suggestion[:200]}...")

            # Re-run agent with fix context
            context["fix_suggestion"] = suggestion
            context["fix_attempt"] = attempt

            result = agent.execute(context)

            # Re-run checks
            checks_passed, failed_outputs = run_checks(checks)

            if checks_passed:
                print(f"[RUNNER] Fix successful on attempt {attempt}!")
                result["status"] = "success"
                result["message"] = f"Fixed after {attempt} attempts"
                return result

            print(f"[RUNNER] Fix attempt {attempt} failed, checks still failing")

        # All attempts failed
        print(f"[RUNNER] All fix attempts exhausted")
        return {
            "status": "failure",
            "message": f"Failed after {self.max_fix_attempts} fix attempts",
            "check_failures": failed_outputs,
        }

    def _log_failure(self, job_name: str, result: Dict[str, Any]):
        """Log a failure to reports directory."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = REPORTS_DIR / f"failure-{job_name}-{timestamp}.log"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Job: {job_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Status: {result.get('status')}\n")
            f.write(f"Message: {result.get('message')}\n")
            if result.get("check_failures"):
                f.write("\nCheck Failures:\n")
                for failure in result["check_failures"]:
                    f.write(f"  - {failure}\n")

        print(f"[RUNNER] Logged failure to {log_file}")

    def run_all_jobs(self) -> Dict[str, Any]:
        """Run all enabled jobs."""
        print("=" * 50)
        print("TASK RUNNER - Starting all jobs")
        print("=" * 50)

        enabled_jobs = [j for j in self.jobs if j.get("enabled", True)]

        print(f"Found {len(enabled_jobs)} enabled jobs")

        for job in enabled_jobs:
            self.run_job(job)

        # Summary
        success_count = sum(1 for r in self.results if r.get("status") == "success")
        failure_count = sum(1 for r in self.results if r.get("status") == "failure")
        skipped_count = sum(1 for r in self.results if r.get("status") == "skipped")

        summary = {
            "total": len(self.results),
            "success": success_count,
            "failure": failure_count,
            "skipped": skipped_count,
            "results": self.results,
        }

        print("\n" + "=" * 50)
        print(
            f"SUMMARY: {success_count} success, {failure_count} failure, {skipped_count} skipped"
        )
        print("=" * 50)

        return summary

    def generate_report(self) -> str:
        """Generate a markdown report of the run."""
        timestamp = datetime.utcnow().isoformat()

        report = f"""# Agent Run Report

**Timestamp**: {timestamp}

## Summary

| Metric | Count |
|--------|-------|
| Total Jobs | {len(self.results)} |
| Success | {sum(1 for r in self.results if r.get('status') == 'success')} |
| Failure | {sum(1 for r in self.results if r.get('status') == 'failure')} |
| Skipped | {sum(1 for r in self.results if r.get('status') == 'skipped')} |

## Job Results

"""

        for result in self.results:
            status_emoji = {"success": "✅", "failure": "❌", "skipped": "⏭️"}.get(
                result.get("status"), "❓"
            )

            report += f"""### {status_emoji} {result.get('job', 'Unknown')}

- **Status**: {result.get('status')}
- **Message**: {result.get('message', 'N/A')}
- **Agent**: {result.get('agent', 'N/A')}

"""

        return report


def run_jobs() -> Dict[str, Any]:
    """Main entry point to run all jobs."""
    runner = TaskRunner()
    return runner.run_all_jobs()


if __name__ == "__main__":
    run_jobs()
