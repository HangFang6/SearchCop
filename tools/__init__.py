"""SearchCop tool library.

Currently 8 of the 10 tools planned in docs/architecture.md §3:
  ✅ gait_encode
  ✅ reid_encode
  ✅ faiss_search
  ✅ multivector_rerank
  ✅ vlm_caption
  ✅ spatio_temporal_filter
  ✅ quality_score
  ✅ evidence_fuse
  ⏳ sam_silhouette          (Day 5)
  ⏳ track_extract           (Day 5)
"""
from .base import BaseTool, ToolResult, ToolRegistry
from .gait_tool import GaitEncodeTool
from .reid_tool import ReidEncodeTool
from .faiss_tool import FaissSearchTool
from .rerank_tool import MultivectorRerankTool
from .vlm_tool import VLMCaptionTool
from .spatial_tool import SpatioTemporalFilterTool
from .quality_tool import QualityScoreTool
from .fusion_tool import EvidenceFuseTool


def build_default_registry(stub: bool = None, llm=None, topology: dict = None) -> ToolRegistry:
    """Build the default tool registry.

    Args:
        stub:     force stub mode for all tools (for unit tests).
        llm:      shared LLMClient instance for tools that need it (vlm_caption).
        topology: camera-topology dict for spatio_temporal_filter.
    """
    reg = ToolRegistry()
    reg.register(GaitEncodeTool(stub=stub))
    reg.register(ReidEncodeTool(stub=stub))
    reg.register(FaissSearchTool(stub=stub))
    reg.register(MultivectorRerankTool(stub=stub))
    reg.register(VLMCaptionTool(llm=llm, stub=stub if stub is not None else (llm is None)))
    reg.register(SpatioTemporalFilterTool(topology=topology))
    reg.register(QualityScoreTool(stub=stub))
    reg.register(EvidenceFuseTool())
    return reg


__all__ = [
    "BaseTool", "ToolResult", "ToolRegistry",
    "GaitEncodeTool", "ReidEncodeTool", "FaissSearchTool",
    "MultivectorRerankTool", "VLMCaptionTool", "SpatioTemporalFilterTool",
    "QualityScoreTool", "EvidenceFuseTool",
    "build_default_registry",
]
