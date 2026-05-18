"""Smoke test — runs on local machine, no GPU / no real API call.

Forces tool-level stub via env var so the suite is deterministic & fast.
"""
import os
import json
import numpy as np
import pytest

# Force stub mode for tools (faiss / opengait / clip-reid)
os.environ["SEARCHCOP_STUB"] = "1"

from agent import LLMClient, SearchCopAgent
from agent.llm_client import _sign, _to_doubao_content, _normalize_messages
from agent.prompts import (
    parse_json_block, build_planner_messages, build_critic_messages,
    render_tool_schemas,
)
from tools import build_default_registry


# ============== LLM client (mock) ==============
def test_llm_mock_text():
    cli = LLMClient(mock=True)
    out = cli.chat([{"role": "user", "content": "ping"}], tier="pro")
    assert out.startswith("[MOCK-DOUBAO/pro]")


def test_llm_mock_lite():
    cli = LLMClient(mock=True)
    out = cli.chat([{"role": "user", "content": "ping"}], tier="lite")
    assert out.startswith("[MOCK-DOUBAO/lite]")


def test_llm_mock_image_payload():
    cli = LLMClient(mock=True)
    out = cli.chat([{"role": "user", "content": [
        {"type": "text", "value": "describe"},
        {"type": "image_url", "value": "https://x/y.jpg"},
    ]}])
    assert "describe" in out


# ============== auth signing ==============
def test_sign_format():
    auth, dt = _sign("src", "appid", "appkey")
    assert auth.startswith('hmac id="appid"')
    assert "algorithm=\"hmac-sha1\"" in auth
    assert "GMT" in dt


# ============== content normalization ==============
def test_string_content_to_doubao():
    assert _to_doubao_content("hello") == [{"type": "text", "value": "hello"}]


def test_openai_image_content_to_doubao():
    blocks = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "https://x/y.jpg"}},
    ]
    out = _to_doubao_content(blocks)
    assert out[0] == {"type": "text", "value": "look"}
    assert out[1] == {"type": "image_url", "value": "https://x/y.jpg"}


def test_normalize_messages_roles():
    msgs = _normalize_messages([{"role": "system", "content": "S"},
                                 {"role": "user",   "content": "U"}])
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == [{"type": "text", "value": "U"}]


# ============== prompts: JSON parse ==============
def test_parse_fenced_json():
    txt = "Some thought.\n```json\n{\"a\": 1, \"b\": [1,2]}\n```\nDone."
    assert parse_json_block(txt) == {"a": 1, "b": [1, 2]}


def test_parse_unfenced_json():
    txt = "Output: {\"thought\": \"go\", \"stop\": true}"
    assert parse_json_block(txt)["stop"] is True


def test_parse_trailing_comma():
    txt = "```json\n{\"a\": 1, \"b\": 2,}\n```"
    assert parse_json_block(txt) == {"a": 1, "b": 2}


def test_parse_garbage_returns_none():
    assert parse_json_block("absolutely not json") is None
    assert parse_json_block("") is None


# ============== prompts: builders ==============
def test_render_tool_schemas_lists_names():
    reg = build_default_registry(stub=True)
    rendered = render_tool_schemas(reg.list_schemas())
    assert "gait_encode" in rendered
    assert "reid_encode" in rendered
    assert "faiss_search" in rendered


def test_planner_messages_have_history():
    reg = build_default_registry(stub=True)
    msgs = build_planner_messages(
        query={"image": "q.jpg"},
        tool_schemas=reg.list_schemas(),
        trace_summary="step1: gait_encode -> ok",
    )
    assert msgs[0]["role"] == "system"
    assert "gait_encode" in msgs[0]["content"]
    assert "step1" in msgs[1]["content"]


def test_critic_messages_have_last_step():
    msgs = build_critic_messages("LAST=foo", "FULL=bar")
    assert "LAST=foo" in msgs[1]["content"]


# ============== tools (stub mode) ==============
def test_tools_registered_three():
    reg = build_default_registry(stub=True)
    names = reg.names()
    for n in ("gait_encode", "reid_encode", "faiss_search"):
        assert n in names


def test_gait_tool_stub_shape(tmp_path):
    # create some empty silhouette files
    sil = tmp_path / "sil"
    sil.mkdir()
    for i in range(40):
        (sil / f"silhouette{i:06d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # bytes only, stub doesn't read
    reg = build_default_registry(stub=True)
    out = reg.get("gait_encode").run(silhouette_dir=str(sil))
    assert out.success is True
    assert out.data["mean_feature"].shape == (256,)
    assert out.data["window_features"].ndim == 2
    assert abs(np.linalg.norm(out.data["mean_feature"]) - 1.0) < 1e-5


def test_reid_tool_stub_shape(tmp_path):
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    for i in range(5):
        (crop_dir / f"reid_crop_frame{i}.jpg").write_bytes(b"\xff\xd8\xff")  # not a real jpg, ok in stub
    reg = build_default_registry(stub=True)
    out = reg.get("reid_encode").run(crop_dir=str(crop_dir))
    assert out.success is True
    assert out.data["mean_feature"].shape == (1280,)
    assert out.data["crop_features"].shape == (5, 1280)


def test_faiss_tool_stub_returns_topk():
    reg = build_default_registry(stub=True)
    q = np.random.randn(1280).astype("float32")
    out = reg.get("faiss_search").run(modality="reid", query_vec=q, top_k=10)
    assert out.success is True
    assert len(out.data["candidates"]) == 10
    assert out.data["candidates"][0]["score"] >= out.data["candidates"][-1]["score"]


def test_gait_tool_missing_dir_fails():
    reg = build_default_registry(stub=True)
    out = reg.get("gait_encode").run(silhouette_dir="/no/such/dir")
    assert out.success is False
    assert "not found" in out.error


# ============== programmable mock LLM for agent loop ==============
class ScriptedLLM:
    """A mock LLMClient with a hand-written reply script.

    The agent calls .chat(messages, tier=...) repeatedly. We inspect the
    system prompt to figure out which role is asking, and pop the next
    scripted reply for that role.
    """
    def __init__(self, planner_replies, critic_replies, aggregator_reply):
        self._planner = list(planner_replies)
        self._critic  = list(critic_replies)
        self._agg     = aggregator_reply

    def chat(self, messages, tier="pro", **kw):
        sys_text = messages[0].get("content", "") if messages else ""
        if "SearchCop, an autonomous planner" in sys_text:
            return self._planner.pop(0) if self._planner else \
                   '```json\n{"thought":"done","tool_call":null,"stop":true}\n```'
        if "SearchCop critic" in sys_text:
            return self._critic.pop(0) if self._critic else \
                   '```json\n{"useful":true,"confidence":0.9,"needs_correction":false,"note":""}\n```'
        if "SearchCop aggregator" in sys_text:
            return self._agg
        return '```json\n{"thought":"unknown","stop":true}\n```'


def test_agent_runs_with_mock_loop():
    """End-to-end: planner picks faiss_search, critic returns high confidence, agent stops."""
    planner_replies = [
        # turn 1: ask for a faiss_search on the reid index
        ('```json\n{'
         '"thought":"start with cheap reid coarse search",'
         '"tool_call":{"name":"faiss_search","arguments":{"modality":"reid","top_k":5}},'
         '"stop":false'
         '}\n```'),
    ]
    critic_replies = [
        '```json\n{"useful":true,"confidence":0.92,"needs_correction":false,"note":"strong"}\n```',
    ]
    aggregator_reply = (
        '```json\n{'
        '"top_k":[{"db_id":42,"score":0.91,"why":"reid #1"}],'
        '"reasoning":"Top-1 came from reid faiss with high score."'
        '}\n```'
    )

    # Provide a query_vec via tool args — but our mock script doesn't include it,
    # and faiss_tool will fail. Patch the planner reply to embed the vec? Simpler:
    # we run with a pre-set arg and accept the resulting failure path. Instead
    # use the registry's stub mode where query_vec is required → expect fail
    # then critic still scores → fine. Replace planner to skip vec? Provide a vec.
    import numpy as np
    q = np.random.randn(1280).astype("float32").tolist()
    planner_replies[0] = (
        '```json\n{'
        '"thought":"start with cheap reid coarse search",'
        '"tool_call":{"name":"faiss_search","arguments":{"modality":"reid",'
        f'"query_vec":{q},"top_k":5'
        '}},'
        '"stop":false'
        '}\n```'
    )

    agent = SearchCopAgent(
        llm=ScriptedLLM(planner_replies, critic_replies, aggregator_reply),
        registry=build_default_registry(stub=True),
        max_steps=4,
    )
    trace = agent.search(query={"image": "q.jpg"})
    assert len(trace.steps) == 1
    assert trace.steps[0].tool_name == "faiss_search"
    assert trace.steps[0].critic_verdict["confidence"] >= 0.85
    assert trace.stop_reason == "confidence-threshold"
    assert trace.top_k and trace.top_k[0]["db_id"] == 42
    assert "reid" in trace.reasoning.lower() or trace.reasoning


def test_agent_dedup_breaks_loop():
    """Same tool+args twice in a row -> agent should stop with dedup reason."""
    same_call = ('```json\n{'
                 '"thought":"repeat","tool_call":{"name":"faiss_search",'
                 '"arguments":{"modality":"reid","query_vec":[0.1,0.2],"top_k":3}},'
                 '"stop":false}\n```')
    low_conf = '```json\n{"useful":false,"confidence":0.1,"needs_correction":true,"note":"weak"}\n```'

    agent = SearchCopAgent(
        llm=ScriptedLLM([same_call, same_call, same_call], [low_conf, low_conf, low_conf],
                        '```json\n{"top_k":[],"reasoning":""}\n```'),
        registry=build_default_registry(stub=True),
        max_steps=5,
        dedup_limit=1,
    )
    trace = agent.search(query={"image": "q.jpg"})
    assert "dedup" in trace.stop_reason


def test_agent_bad_tool_name_stops():
    agent = SearchCopAgent(
        llm=ScriptedLLM(
            ['```json\n{"thought":"oops","tool_call":{"name":"no_such_tool","arguments":{}},"stop":false}\n```'],
            [],
            '```json\n{"top_k":[],"reasoning":""}\n```'
        ),
        registry=build_default_registry(stub=True),
        max_steps=3,
    )
    trace = agent.search(query={"image": "q.jpg"})
    assert "bad tool" in trace.stop_reason
