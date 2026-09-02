from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_local_herg_receptor_hybrid_validation_v13_6.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("herg_v13_6_hybrid_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame(lgbm: np.ndarray, receptor: np.ndarray, baseline: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"baseline_prediction": baseline})
    for index, column in enumerate(MODULE.LGBM_COLUMNS):
        frame[column] = lgbm[:, index]
    for index, column in enumerate(MODULE.RECEPTOR_COLUMNS):
        frame[column] = receptor[:, index]
    return frame


def test_hybrid_policy_uses_only_frozen_tail_overrides() -> None:
    lgbm = np.array([[0.1, 0.8, 0.1]] * 5)
    receptor = np.array(
        [
            [0.60, 0.30, 0.10],
            [0.60, 0.30, 0.10],
            [0.20, 0.30, 0.50],
            [0.20, 0.35, 0.45],
            [0.20, 0.60, 0.20],
        ]
    )
    frame = _frame(lgbm, receptor, np.array([4.7, 4.9, 5.0, 5.0, 5.0]))
    assert MODULE._apply_hybrid_policy(frame).tolist() == [0, 1, 2, 2, 1]
    assert MODULE.SAFE_PROBABILITY_MINIMUM == 0.34
    assert MODULE.POTENT_PROBABILITY_MINIMUM == 0.40


def test_score_confirms_uniform_safe_improvement_over_lgbm() -> None:
    y = np.tile(np.arange(3), 5)
    panel = pd.DataFrame(
        {
            "observed_target": np.where(y == 0, 4.0, np.where(y == 1, 5.0, 7.0)),
            "outer_fold": np.repeat(np.arange(5), 3),
        }
    )
    lgbm = np.zeros((len(y), 3))
    lgbm[:, 1] = 1.0
    predictions = _frame(lgbm, np.eye(3)[y], np.full(len(y), 5.0))
    predictions["hybrid_prediction"] = y
    report = MODULE._score(panel, predictions, 100)
    assert report["validation_decision"]["confirmed"]
    assert report["validation_decision"]["safety_passed"]
    assert report["validation_decision"]["moderate_retention_passed"]
    assert report["frozen_hybrid"]["folds_better"] == 5


def test_parser_defaults_match_frozen_hybrid_protocol() -> None:
    args = MODULE._parser().parse_args([])
    assert args.exhaustiveness == 8
    assert args.modes == 9
    assert args.bootstrap_replicates == 10_000


def test_moderate_retention_accepts_exact_declared_boundary() -> None:
    assert MODULE._moderate_retention_passed(66 / 90, 84 / 90)
    assert not MODULE._moderate_retention_passed((66 / 90) - 1e-6, 84 / 90)
