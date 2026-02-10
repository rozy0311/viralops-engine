"""
Base Agent Class
================
All agents (both built-in and generated) inherit from this class.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional
import json
import os


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.created_at = datetime.utcnow().isoformat()
        self.run_count = 0
        self.last_run = None
        self.last_status = None

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main task.

        Args:
            context: Dictionary containing task context and parameters

        Returns:
            Dictionary with:
                - status: "success" | "failure" | "skipped"
                - message: Human-readable result message
                - data: Any output data (optional)
                - changes: List of files changed (optional)
        """
        pass

    def pre_run(self, context: Dict[str, Any]) -> bool:
        """
        Pre-run checks. Override to add custom validation.

        Returns:
            True if agent should run, False to skip
        """
        return True

    def post_run(self, result: Dict[str, Any]) -> None:
        """Post-run cleanup or logging. Override to customize."""
        self.run_count += 1
        self.last_run = datetime.utcnow().isoformat()
        self.last_status = result.get("status", "unknown")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full execution flow with pre/post hooks.

        This is the main entry point - don't override this.
        Override run() instead.
        """
        try:
            # Pre-run check
            if not self.pre_run(context):
                return {
                    "status": "skipped",
                    "message": f"Agent {self.name} skipped pre-run check",
                    "agent": self.name,
                }

            # Main execution
            result = self.run(context)
            result["agent"] = self.name

            # Post-run
            self.post_run(result)

            return result

        except Exception as e:
            error_result = {
                "status": "failure",
                "message": f"Agent {self.name} failed: {str(e)}",
                "agent": self.name,
                "error": str(e),
            }
            self.post_run(error_result)
            return error_result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "run_count": self.run_count,
            "last_run": self.last_run,
            "last_status": self.last_status,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"


class TaskAgent(BaseAgent):
    """
    Agent that performs a specific task.
    Base class for simple task-oriented agents.
    """

    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.allowed_paths: list = []
        self.forbidden_paths: list = []

    def is_path_allowed(self, path: str) -> bool:
        """Check if a path is allowed for modification."""
        # Check forbidden first
        for forbidden in self.forbidden_paths:
            if path.startswith(forbidden) or path == forbidden:
                return False

        # Check allowed
        for allowed in self.allowed_paths:
            if path.startswith(allowed) or path == allowed.rstrip("/"):
                return True

        return False

    def safe_write(self, path: str, content: str) -> bool:
        """Write to file only if path is allowed."""
        if not self.is_path_allowed(path):
            raise PermissionError(f"Path not allowed: {path}")

        # Create directory if needed
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return True


class OrchestratorAgent(BaseAgent):
    """
    Agent that manages other agents.
    Can spawn, monitor, and coordinate child agents.
    """

    def __init__(self, name: str = "meta-agent"):
        super().__init__(name, "Orchestrator that manages other agents")
        self.child_agents: Dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a child agent."""
        self.child_agents[agent.name] = agent

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get a registered agent by name."""
        return self.child_agents.get(name)

    def list_agents(self) -> list:
        """List all registered agent names."""
        return list(self.child_agents.keys())

    def run_agent(self, name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific child agent."""
        agent = self.get_agent(name)
        if not agent:
            return {"status": "failure", "message": f"Agent not found: {name}"}
        return agent.execute(context)

    def run_all(self, context: Dict[str, Any]) -> Dict[str, list]:
        """Run all registered agents."""
        results = []
        for name, agent in self.child_agents.items():
            result = agent.execute(context)
            results.append(result)
        return {"results": results}
