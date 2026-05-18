"""Tests for Day-3 tools: rerank / vlm / spatial / quality / fusion."""
import os
os.environ["SEARCHCOP_STUB"] = "1"

import numpy as np
import pytest

from tools import build_default_registry
from tools.rerank_tool import rerank_score, _combined, _topk_mean, _max, _set2set
from tools.spatial_tool import feasible
from tools.fusion_tool import fuse_weighted, fuse_borda, fuse_rrf


# ============== multivector_rerank ==============
def test_rerank_combined_monotone():
    """Higher all-to-all sim -> higher combined score."""
    a = np.eye(3, 4).astype("f4")            # query 3x4
    g_good = a + 0.0
    g_bad  = np.zeros((3, 4), dtype="f4"); g_bad[0, 0] = 1.0
    s_good = rerank_score(a, g_good, "combined")
    s_bad  = rerank_score(a, g_bad,  "combined")
    assert s_good > s_bad


def test_rerank_strategies_run():
    sim = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.3]], dtype="f4")
    assert 0.0 <= _topk_mean(sim) <= 1.0
    assert _max(sim) == 0.9
    s = _set2set(sim)
    assert 0.0 <= s <= 1.0
    c = _combined(sim)
    assert 0.0 <= c <= 1.0


def test_rerank_tool_stub():
    reg = build_default_registry(stub=True)
    out = reg.get("multivector_rerank").run(
        candidate_db_ids=[1, 2, 3, 4], strategy="combined", top_k=3,
    )
    assert out.success
    assert len(out.data["ranked"]) == 3
    # sorted desc
    scores = [r["score"] for r in out.data["ranked"]]
    assert scores == sorted(scores, reverse=True)


def test_rerank_tool_real_path():
    reg = build_default_registry(stub=False)   # tool itself isn't stubbed
    rng = np.random.default_rng(0)
    q = rng.normal(size=(2, 8)).astype("f4")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    gallery = {i: rng.normal(size=(3, 8)).astype("f4") for i in [10, 20, 30]}
    for k in gallery:
        gallery[k] /= np.linalg.norm(gallery[k], axis=1, keepdims=True)
    out = reg.get("multivector_rerank").run(
        candidate_db_ids=[10, 20, 30],
        query_features=q, gallery_features=gallery,
        strategy="combined", top_k=3,
    )
    assert out.success
    assert len(out.data["ranked"]) == 3


# ============== vlm_caption ==============
def test_vlm_tool_stub_returns_caption():
    reg = build_default_registry(stub=True)
    out = reg.get("vlm_caption").run(image_url="https://example.com/x.jpg")
    assert out.success
    assert "caption" in out.data
    assert out.data["caption"]["top_color"] == "red"


def test_vlm_tool_no_input_fails():
    reg = build_default_registry(stub=True)
    out = reg.get("vlm_caption").run()
    assert not out.success


# ============== spatio_temporal_filter ==============
def test_feasible_same_camera_always_true():
    assert feasible({}, "cam1", 0, "cam1", 100) is True


def test_feasible_unknown_edge_kept():
    # missing topology -> permissive
    assert feasible({}, "cam1", 0, "cam2", 100) is True


def test_feasible_within_p90():
    topo = {"cam1": {"transit_seconds": {"cam2": {"p50": 30, "p90": 60}}}}
    assert feasible(topo, "cam1", 0.0, "cam2", 50.0)
    assert not feasible(topo, "cam1", 0.0, "cam2", 200.0)


def test_spatial_tool_drops_unreachable():
    reg = build_default_registry(
        stub=True,
        topology={"cam1": {"transit_seconds": {"cam2": {"p50": 30, "p90": 60}}}},
    )
    out = reg.get("spatio_temporal_filter").run(
        query_camera="cam1", query_time=0.0,
        candidates=[
            {"db_id": 1, "camera_id": "cam2", "time_seen": 50.0, "score": 0.9},   # ok
            {"db_id": 2, "camera_id": "cam2", "time_seen": 500.0, "score": 0.8},  # too late
        ],
    )
    assert out.success
    assert len(out.data["kept"]) == 1
    assert out.data["kept"][0]["db_id"] == 1


# ============== quality_score ==============
def test_quality_tool_stub():
    reg = build_default_registry(stub=True)
    out = reg.get("quality_score").run(image_path="/no/such/file.jpg")
    assert out.success
    assert 0.0 <= out.data["score"] <= 1.0


def test_quality_tool_real_missing_file():
    reg = build_default_registry(stub=False)
    out = reg.get("quality_score").run(image_path="/nope/x.jpg")
    assert not out.success


# ============== evidence_fuse ==============
def test_fuse_weighted_basic():
    L1 = [{"db_id": 1, "score": 1.0}, {"db_id": 2, "score": 0.5}]
    L2 = [{"db_id": 2, "score": 1.0}, {"db_id": 3, "score": 0.5}]
    s = fuse_weighted([L1, L2])
    # 2 should win (in both lists with high score)
    assert max(s, key=s.get) == 2


def test_fuse_borda_basic():
    L1 = [{"db_id": 1, "score": 0.9}, {"db_id": 2, "score": 0.8}, {"db_id": 3, "score": 0.7}]
    L2 = [{"db_id": 3, "score": 0.9}, {"db_id": 2, "score": 0.85}, {"db_id": 1, "score": 0.6}]
    s = fuse_borda([L1, L2])
    # 2 ranks (2 + 2) = 4, 1 ranks (3+1)=4, 3 ranks (1+3)=4 -> tie; just check all present
    assert set(s.keys()) == {1, 2, 3}


def test_fuse_rrf_strict_winner():
    L1 = [{"db_id": 1, "score": 0.9}, {"db_id": 2, "score": 0.8}]
    L2 = [{"db_id": 1, "score": 0.7}, {"db_id": 3, "score": 0.5}]
    s = fuse_rrf([L1, L2], k=60)
    # 1 appears at rank 1 in both -> highest
    assert max(s, key=s.get) == 1


def test_fusion_tool_end_to_end():
    reg = build_default_registry(stub=True)
    out = reg.get("evidence_fuse").run(
        ranked_lists=[
            [{"db_id": 1, "score": 0.9}, {"db_id": 2, "score": 0.8}],
            [{"db_id": 1, "score": 0.7}, {"db_id": 3, "score": 0.5}],
        ],
        strategy="rrf", top_k=3,
    )
    assert out.success
    assert out.data["ranked"][0]["db_id"] == 1


def test_fusion_unknown_strategy():
    reg = build_default_registry(stub=True)
    out = reg.get("evidence_fuse").run(
        ranked_lists=[[{"db_id": 1, "score": 1.0}]],
        strategy="not_a_strategy",
    )
    assert not out.success


# ============== registry has all 8 tools ==============
def test_registry_has_eight_tools():
    reg = build_default_registry(stub=True)
    expected = {"gait_encode", "reid_encode", "faiss_search",
                "multivector_rerank", "vlm_caption", "spatio_temporal_filter",
                "quality_score", "evidence_fuse"}
    assert expected.issubset(set(reg.names()))
