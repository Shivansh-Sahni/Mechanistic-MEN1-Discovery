from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_local_herg_receptor_classification_confirmation_v13_3.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("herg_v13_3_confirmation_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_primary_feature_is_frozen_single_surface() -> None:
    assert MODULE.STATES == ("8ZYO", "8ZYQ")
    assert MODULE.PRIMARY_FEATURES == ("dock__8ZYO__contact_649_count",)


def test_balanced_bootstrap_detects_uniform_classification_improvement() -> None:
    y = np.tile(np.arange(3), 10)
    baseline = np.zeros((len(y), 3))
    baseline[:, 1] = 1.0
    candidate = np.eye(3)[y]
    result = MODULE._balanced_bootstrap(y, baseline, candidate, 100)
    assert np.isclose(result["delta_balanced_accuracy_candidate_minus_baseline"], 2 / 3)
    assert result["ci95"][0] > 0
    assert result["probability_candidate_better"] == 1.0


def test_confirmation_rule_requires_fold_consistency_and_safety() -> None:
    y = np.tile(np.arange(3), 5)
    confirmation = pd.DataFrame(
        {
            "observed_target": np.where(y == 0, 4.0, np.where(y == 1, 5.0, 7.0)),
            "outer_fold": np.repeat(np.arange(5), 3),
        }
    )
    predictions = pd.DataFrame()
    for class_index, class_name in enumerate(MODULE.v13.CLASS_NAMES):
        predictions[f"baseline_probability_{class_name.lower()}"] = (
            np.ones(len(y)) if class_index == 1 else np.zeros(len(y))
        )
        perfect = np.eye(3)[y]
        for hypothesis in (
            "primary_frozen_f649",
            "secondary_frozen_sensitivity_contacts",
        ):
            predictions[f"{hypothesis}__probability_{class_name.lower()}"] = perfect[
                :, class_index
            ]
    report = MODULE._score(confirmation, predictions, 100)
    assert report["confirmation_decision"]["confirmed"]
    assert report["hypotheses"]["primary_frozen_f649"]["folds_better"] == 5


def test_parser_defaults_match_frozen_confirmation_protocol() -> None:
    args = MODULE._parser().parse_args([])
    assert args.exhaustiveness == 8
    assert args.modes == 9
    assert args.bootstrap_replicates == 10_000
