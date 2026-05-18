"""Smoke test — runs on local machine, no GPU / no real API call."""
import numpy as np

from agent import LLMClient, SearchCopAgent
from tools import build_default_registry


def test_llm_mock():
    cli = LLMClient(mock=True)
    out = cli.chat([{"role": "user", "content": "ping"}])
    assert out.startswith("[MOCK-LLM]")


def test_tools_registered():
    reg = build_default_registry()
    assert "gait_encode" in reg.names()
    assert "reid_encode" in reg.names()
    schemas = reg.list_schemas()
    assert all("parameters" in s for s in schemas)


def test_tool_run_returns_feature():
    reg = build_default_registry()
    gait = reg.get("gait_encode").run(silhouette_dir="/fake/sil")
    reid = reg.get("reid_encode").run(crop_paths=["a.jpg", "b.jpg"])
    assert gait.success and gait.data.shape == (256,)
    assert reid.success and reid.data.shape == (1280,)
    assert abs(np.linalg.norm(gait.data) - 1.0) < 1e-5


def test_agent_runs_with_mocks():
    agent = SearchCopAgent(llm=LLMClient(mock=True))
    trace = agent.search(query={"image": "fake.jpg"})
    assert trace.query == {"image": "fake.jpg"}
    # planner stub returns None, so 0 steps is expected at this stage
    assert isinstance(trace.steps, list)
