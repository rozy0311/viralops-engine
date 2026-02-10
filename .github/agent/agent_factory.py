"""
Agent Factory
=============
Loads, registers, and manages agents.
Discovers generated agents and creates new ones.
"""

import os
import sys
import importlib
import importlib.util
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml

from agents.base_agent import BaseAgent, TaskAgent, OrchestratorAgent


# Paths
AGENT_DIR = Path(__file__).parent
CONFIG_PATH = AGENT_DIR / "config" / "agents.yaml"
GENERATED_DIR = AGENT_DIR / "agents" / "generated"


def load_config() -> Dict[str, Any]:
    """Load agent configuration from YAML."""
    if not CONFIG_PATH.exists():
        return {"agents": [], "auto_spawn": {}, "safety": {}}

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_safety_config() -> Dict[str, Any]:
    """Get safety configuration."""
    config = load_config()
    return config.get(
        "safety",
        {
            "max_retries": 3,
            "timeout_seconds": 300,
            "allowed_paths": [],
            "forbidden_paths": [],
        },
    )


def get_auto_spawn_config() -> Dict[str, Any]:
    """Get auto-spawn configuration."""
    config = load_config()
    return config.get(
        "auto_spawn",
        {"enabled": False, "max_agents_per_run": 1, "max_total_agents": 10},
    )


class AgentFactory:
    """Factory for creating and managing agents."""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.config = load_config()
        self.safety = get_safety_config()
        self._ensure_generated_dir()

    def _ensure_generated_dir(self):
        """Ensure the generated agents directory exists."""
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)

        # Create __init__.py if not exists
        init_file = GENERATED_DIR / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Auto-generated agents\n")

    def create_builtin_agents(self) -> List[BaseAgent]:
        """Create built-in agents from config."""
        agents = []

        for agent_config in self.config.get("agents", []):
            if not agent_config.get("enabled", True):
                continue

            name = agent_config["name"]
            agent_type = agent_config.get("type", "task")
            description = agent_config.get("description", "")

            if agent_type == "orchestrator":
                agent = OrchestratorAgent(name)
            else:
                agent = self._create_task_agent(name, description)

            agents.append(agent)
            self.register(agent)

        return agents

    def _create_task_agent(self, name: str, description: str) -> TaskAgent:
        """Create a task agent with safety config."""

        class DynamicTaskAgent(TaskAgent):
            def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
                # Default implementation - override in generated agents
                return {"status": "success", "message": f"Agent {self.name} completed"}

        agent = DynamicTaskAgent(name, description)
        agent.allowed_paths = self.safety.get("allowed_paths", [])
        agent.forbidden_paths = self.safety.get("forbidden_paths", [])

        return agent

    def register(self, agent: BaseAgent) -> None:
        """Register an agent."""
        self.agents[agent.name] = agent

    def get(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self.agents.keys())

    def discover_generated_agents(self) -> List[BaseAgent]:
        """
        Discover and load agents from the generated directory.
        """
        discovered = []

        if not GENERATED_DIR.exists():
            return discovered

        for file in GENERATED_DIR.glob("*.py"):
            if file.name.startswith("_"):
                continue

            try:
                agent = self._load_agent_from_file(file)
                if agent:
                    discovered.append(agent)
                    self.register(agent)
            except Exception as e:
                print(f"Failed to load agent from {file}: {e}")

        return discovered

    def _load_agent_from_file(self, file_path: Path) -> Optional[BaseAgent]:
        """Load an agent class from a Python file."""
        module_name = f"generated.{file_path.stem}"

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Error loading module {file_path}: {e}")
            return None

        # Find BaseAgent subclass in module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseAgent)
                and attr is not BaseAgent
                and attr is not TaskAgent
                and attr is not OrchestratorAgent
            ):

                # Instantiate and return
                return attr()

        return None

    def count_generated_agents(self) -> int:
        """Count the number of generated agent files."""
        if not GENERATED_DIR.exists():
            return 0
        return len(
            [f for f in GENERATED_DIR.glob("*.py") if not f.name.startswith("_")]
        )

    def can_spawn_more(self) -> bool:
        """Check if we can spawn more agents based on limits."""
        spawn_config = get_auto_spawn_config()

        if not spawn_config.get("enabled", False):
            return False

        max_total = spawn_config.get("max_total_agents", 10)
        current_count = self.count_generated_agents()

        return current_count < max_total


# Singleton factory instance
_factory: Optional[AgentFactory] = None


def get_factory() -> AgentFactory:
    """Get the singleton factory instance."""
    global _factory
    if _factory is None:
        _factory = AgentFactory()
    return _factory


def load_all_agents() -> Dict[str, BaseAgent]:
    """Load all agents (built-in and generated)."""
    factory = get_factory()
    factory.create_builtin_agents()
    factory.discover_generated_agents()
    return factory.agents
