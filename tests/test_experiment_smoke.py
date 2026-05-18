"""End-to-end pipeline smoke test (no GPU, no API)."""
import os
os.environ["SEARCHCOP_STUB"] = "1"

import json

from experiments.run_searchcop import make_searchcop_predictor
from experiments.runner import run_experiment


def test_searchcop_pipeline_with_mock_llm(tmp_path):
    cfg_path = "experiments/configs/mock_smoke.yaml"
    assert os.path.isfile(cfg_path), "mock_smoke.yaml must exist"

    # Use mock LLM so no network call
    from experiments.runner import load_config
    cfg = load_config(cfg_path)
    predictor, traces = make_searchcop_predictor(cfg, stub=True, mock_llm=True)

    out = tmp_path / "smoke_out"
    metrics = run_experiment(cfg_path, predictor, str(out), trace_dump=traces)

    # files written
    assert (out / "predictions.jsonl").exists()
    assert (out / "metrics.json").exists()

    # metrics shape
    assert "person_map" in metrics
    assert "cmc_at_k" in metrics
    assert metrics["n_queries"] == 5
    # we collected one trace per query
    assert len(traces) == 5


def test_baseline_pipeline_smoke(tmp_path):
    from experiments.run_baseline import make_baseline_predictor
    cfg_path = "experiments/configs/mock_smoke.yaml"
    pred = make_baseline_predictor(stub=True)
    out = tmp_path / "baseline_out"
    metrics = run_experiment(cfg_path, pred, str(out))
    assert (out / "predictions.jsonl").exists()
    assert metrics["n_queries"] == 5
