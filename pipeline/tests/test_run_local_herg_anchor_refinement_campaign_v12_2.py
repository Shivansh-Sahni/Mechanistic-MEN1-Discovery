from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts/run_local_herg_anchor_refinement_campaign_v12_2.py"
SPEC = importlib.util.spec_from_file_location("herg_v12_2_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_projection_enforces_ic10_ic30_ic50_order() -> None:
    projected = MODULE._project_nonincreasing(np.array([4.0, 5.0, 4.5]))
    assert projected[0] >= projected[1] >= projected[2]


def test_regression_metrics_are_exact() -> None:
    metrics = MODULE._metrics(np.array([1.0, 2.0]), np.array([1.5, 1.5]))
    assert metrics["n"] == 2
    assert metrics["mae"] == 0.5
    assert metrics["rmse"] == 0.5


def test_scaffold_bootstrap_uses_paired_delta() -> None:
    frame = pd.DataFrame({"scaffold_group_id": ["a", "b", "c"]})
    observed = np.array([1.0, 2.0, 3.0])
    baseline = np.array([2.0, 3.0, 4.0])
    candidate = np.array([1.0, 2.0, 3.0])
    result = MODULE._bootstrap_delta(frame, baseline, candidate, observed, 100)
    assert result["delta_mae_candidate_minus_baseline"] == -1.0
    assert result["probability_candidate_better"] == 1.0
