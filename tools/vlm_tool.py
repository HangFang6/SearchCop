"""
VLM Caption Tool — describe a person crop in natural language.

Implementation: a single Doubao multi-modal chat call (tier=lite).
The LLM receives a system instruction + a user message that mixes one
text block and one image_url block. Output is a short structured caption
the planner can use to filter candidates with non-visual cues
(e.g. "red shirt, backpack").
"""
from __future__ import annotations

import base64
import os
from typing import Optional

from .base import BaseTool, ToolResult


VLM_SYSTEM = (
    "You are a person-attribute describer for a multi-camera search system. "
    "Given an image of a single person, return ONLY a compact JSON object with "
    "fields: gender, top_color, top_type, bottom_color, bottom_type, has_backpack, "
    "has_hat, accessories, height_estimate, age_bucket. Use null for unknown."
)


class VLMCaptionTool(BaseTool):
    name = "vlm_caption"
    description = (
        "Generate a structured attribute caption for a single person crop. "
        "Use when the gait/reid features alone are weak (heavy occlusion, low light) "
        "and clothing or accessory cues would help disambiguate candidates."
    )

    @property
    def json_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url":    {"type": "string", "description": "Public URL of the crop"},
                    "image_path":   {"type": "string", "description": "Local file path (will be base64-encoded)"},
                    "extra_prompt": {"type": "string", "default": ""},
                },
            },
        }

    def __init__(self, llm=None, tier: str = "lite", stub: Optional[bool] = None):
        # llm: optional LLMClient; if None and stub auto-decides
        self.llm = llm
        self.tier = tier
        if stub is None:
            stub = (llm is None) or os.getenv("SEARCHCOP_STUB", "0") == "1"
        self.stub = stub

    def run(self,
            image_url: Optional[str] = None,
            image_path: Optional[str] = None,
            extra_prompt: str = "") -> ToolResult:
        if not image_url and not image_path:
            return ToolResult(success=False, error="image_url or image_path required")

        if self.stub:
            return ToolResult(
                success=True,
                data={
                    "caption": {
                        "gender": "unknown", "top_color": "red", "top_type": "tshirt",
                        "bottom_color": "blue", "bottom_type": "jeans",
                        "has_backpack": True, "has_hat": False,
                        "accessories": ["backpack"], "height_estimate": "medium",
                        "age_bucket": "20-40",
                    },
                    "raw": "[STUB-VLM] mock caption",
                },
                meta={"stub": True},
            )

        if image_path and not image_url:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content_blocks = [
                {"type": "text", "value":
                    "Describe this single person.\n" + (extra_prompt or "")},
                {"type": "image_base64", "value": b64},
            ]
        else:
            content_blocks = [
                {"type": "text", "value":
                    "Describe this single person.\n" + (extra_prompt or "")},
                {"type": "image_url", "value": image_url},
            ]

        msgs = [
            {"role": "system", "content": VLM_SYSTEM},
            {"role": "user",   "content": content_blocks},
        ]
        try:
            text = self.llm.chat(msgs, tier=self.tier, max_tokens=256, use_cache=True)
        except Exception as e:
            return ToolResult(success=False, error=f"vlm call failed: {e}")

        # Parse JSON; tolerate bare object
        from agent.prompts import parse_json_block
        parsed = parse_json_block(text) or {}
        return ToolResult(
            success=True,
            data={"caption": parsed, "raw": text},
            meta={"stub": False, "tier": self.tier},
        )
