"""
Base Tool interface. All concrete tools subclass `BaseTool` and register a JSON schema
so the LLM Agent can decide when to call them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)   # latency, gpu_mem, ...


class BaseTool(ABC):
    """All tools expose: name, description, json_schema, run()."""

    name: str = "base_tool"
    description: str = "abstract base tool"

    @property
    def json_schema(self) -> Dict:
        """OpenAI function-calling schema. Override in subclass."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": {}},
        }

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        ...


# --- Registry --------------------------------------------------------------
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        return self._tools[name]

    def list_schemas(self) -> List[Dict]:
        return [t.json_schema for t in self._tools.values()]

    def names(self) -> List[str]:
        return list(self._tools.keys())
