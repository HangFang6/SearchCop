"""SearchCop tool library.

To add a new tool: subclass `BaseTool`, implement `run`, then register it in
`build_default_registry`.
"""
from .base import BaseTool, ToolResult, ToolRegistry
from .gait_tool import GaitEncodeTool
from .reid_tool import ReidEncodeTool


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(GaitEncodeTool())
    reg.register(ReidEncodeTool())
    # TODO: register VLM caption / spatio-temporal filter / quality scorer / faiss search / ...
    return reg


__all__ = ["BaseTool", "ToolResult", "ToolRegistry",
           "GaitEncodeTool", "ReidEncodeTool", "build_default_registry"]
