"""Tests for data loaders."""
import os
import pytest

from data import build_dataset, MockDataset, MEVIDDataset, QueryRecord, GalleryRecord


def test_factory_mock():
    ds = build_dataset("MOCK")
    assert isinstance(ds, MockDataset)


def test_mock_dataset_contract():
    ds = MockDataset(n_queries=4, n_gallery=20, seed=1)
    qs = ds.queries()
    gs = ds.gallery()
    gt = ds.ground_truth()
    topo = ds.topology()

    assert len(qs) == 4 and all(isinstance(q, QueryRecord) for q in qs)
    assert len(gs) == 20 and all(isinstance(g, GalleryRecord) for g in gs)
    assert set(gt.keys()) == {q.query_id for q in qs}
    # every gt set is non-empty
    for qid, ids in gt.items():
        assert isinstance(ids, set) and len(ids) >= 1
    # topology has 4 cams with neighbors
    assert all("neighbors" in v for v in topo.values())


def test_mock_query_to_agent_format():
    ds = MockDataset(n_queries=2)
    aq = ds.queries()[0].as_agent_query()
    assert "query_id" in aq and "image" in aq and "cam_id" in aq


def test_mevid_missing_root_returns_empty(tmp_path):
    ds = MEVIDDataset(root=str(tmp_path), split="test")
    assert ds.queries() == []
    assert ds.gallery() == []
    assert ds.topology() == {}
    assert ds.ground_truth() == {}


def test_factory_unknown_raises():
    with pytest.raises(ValueError):
        build_dataset("never_heard_of")


def test_factory_mevid_needs_root():
    with pytest.raises(ValueError):
        build_dataset("MEVID")
