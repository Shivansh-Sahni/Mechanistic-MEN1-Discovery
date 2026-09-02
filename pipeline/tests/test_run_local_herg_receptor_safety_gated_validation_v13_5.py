from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_local_herg_receptor_safety_gated_validation_v13_5.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("herg_v13_5_safety_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _prediction_frame(probabilities: np.ndarray, baseline: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"baseline_prediction": baseline})
    for index, name in enumerate(MODULE.v13.CLASS_NAMES):
        frame[f"primary_frozen_f649__probability_{name.lower()}"] = probabilities[:, index]
    return frame


def test_safety_policy_is_frozen_and_overrides_unsafe_safe_calls() -> None:
    probabilities = np.array(
        [
            [0.60, 0.30, 0.10],
            [0.60, 0.30, 0.10],
            [0.335, 0.334, 0.331],
            [0.20, 0.60, 0.20],
        ]
    )
    frame = _prediction_frame(probabilities, np.array([4.7, 4.9, 4.0, 5.2]))
    assert MODULE._apply_safety_policy(frame).tolist() == [0, 1, 1, 1]
    assert MODULE.SAFE_PROBABILITY_MINIMUM == 0.34
    assert MODULE.BASELINE_PIC50_SAFE_GATE == 4.80


def test_score_policy_confirms_safe_uniform_improvement() -> None:
    y = np.tile(np.arange(3), 5)
    panel = pd.DataFrame(
        {
            "observed_target": np.where(y == 0, 4.0, np.where(y == 1, 5.0, 7.0)),
            "outer_fold": np.repeat(np.arange(5), 3),
        }
    )
    predictions = pd.DataFrame(
        {
            "baseline_prediction": np.full(len(y), 5.0),
            "safety_gated_prediction": y,
        }
    )
    report = MODULE._score_policy(panel, predictions, 100)
    assert report["validation_decision"]["confirmed"]
    assert report["validation_decision"]["safety_passed"]
    assert report["safety_gated_frozen_f649"]["folds_better"] == 5


def test_parser_defaults_match_frozen_validation_protocol() -> None:
    args = MODULE._parser().parse_args([])
    assert args.exhaustiveness == 8
    assert args.modes == 9
    assert args.bootstrap_replicates == 10_000
