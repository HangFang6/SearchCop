"""
Spatio-Temporal Filter Tool.

Given a list of candidates with (camera_id, time_seen) and a camera topology
that records inter-camera transit time distributions, prune candidates that
cannot physically reach the next camera within the requested time window.

Topology JSON:
  {
    "cam_id": {
      "neighbors": ["cam_x", "cam_y"],
      "transit_seconds": {"cam_x": {"p50": 12.3, "p90": 27.8}, ...}
    }
  }
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import BaseTool, ToolResult


def feasible(topo: Dict, cam_a: str, t_a: float, cam_b: str, t_b: float,
             slack_seconds: float = 5.0) -> bool:
    """True if going A@t_a -> B@t_b is plausible under topology p90 transit."""
    if cam_a == cam_b:
        return True
    info = (topo.get(cam_a) or {}).get("transit_seconds", {}).get(cam_b)
    if not info:
        # Unknown edge: be permissive (don't prune) to avoid losing recall
        return True
    dt = t_b - t_a
    if dt < -slack_seconds:                # going backwards in time
        return False
    p90 = info.get("p90", info.get("p50", 60.0))
    return abs(dt) <= p90 + slack_seconds


class SpatioTemporalFilterTool(BaseTool):
    name = "spatio_temporal_filter"
    description = (
        "Drop candidates that cannot plausibly travel from query (cam, t) to "
        "(candidate.cam, candidate.t) under the camera topology. Conservative: "
        "unknown edges are kept."
    )

    @property
    def json_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query_camera":    {"type": "string"},
                    "query_time":      {"type": "number", "description": "epoch seconds"},
                    "candidates":      {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "db_id":        {"type": "integer"},
                                "camera_id":    {"type": "string"},
                                "time_seen":    {"type": "number"},
                                "score":        {"type": "number"},
                            },
                            "required": ["db_id", "camera_id", "time_seen"],
                        },
                    },
                    "slack_seconds":   {"type": "number", "default": 5.0},
                },
                "required": ["query_camera", "query_time", "candidates"],
            },
        }

    def __init__(self, topology: Optional[Dict] = None):
        self.topology = topology or {}

    def set_topology(self, topo: Dict) -> None:
        self.topology = topo or {}

    def run(self,
            query_camera: str,
            query_time: float,
            candidates: List[Dict],
            slack_seconds: float = 5.0) -> ToolResult:
        kept: List[Dict] = []
        dropped: List[Dict] = []
        for c in candidates:
            ok = feasible(self.topology, query_camera, query_time,
                          c.get("camera_id", ""), c.get("time_seen", 0.0),
                          slack_seconds=slack_seconds)
            (kept if ok else dropped).append(c)
        return ToolResult(
            success=True,
            data={"kept": kept, "dropped": dropped},
            meta={"n_kept": len(kept), "n_dropped": len(dropped),
                  "topology_known": bool(self.topology)},
        )
