from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_local_herg_receptor_classification_replication_v13_4.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("herg_v13_4_replication_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _perfect_example() -> tuple[pd.DataFrame, pd.DataFrame]:
    y = np.tile(np.arange(3), 5)
    panel = pd.DataFrame(
        {
            "observed_target": np.where(y == 0, 4.0, np.where(y == 1, 5.0, 7.0)),
            "outer_fold": np.repeat(np.arange(5), 3),
        }
    )
    predictions = pd.DataFrame()
    perfect = np.eye(3)[y]
    for class_index, class_name in enumerate(MODULE.v13.CLASS_NAMES):
        predictions[f"baseline_probability_{class_name.lower()}"] = (
            np.ones(len(y)) if class_index == 1 else np.zeros(len(y))
        )
        predictions[f"primary_frozen_f649__probability_{class_name.lower()}"] = perfect[
            :, class_index
        ]
    return panel, predictions


def test_replication_contract_is_single_frozen_surface() -> None:
    assert MODULE.STATE == "8ZYO"
    assert MODULE.PRIMARY_FEATURE == "dock__8ZYO__contact_649_count"
    assert MODULE.PER_FOLD_TIER == 18


def test_score_panel_confirms_uniform_foldwise_improvement() -> None:
    panel, predictions = _perfect_example()
    report = MODULE._score_panel(panel, predictions, 100)
    assert report["confirmation_decision"]["confirmed"]
    assert report["primary_frozen_f649"]["folds_better"] == 5
    assert report["primary_frozen_f649"]["vs_baseline_bootstrap"]["ci95"][0] > 0


def test_parser_defaults_match_discovery_protocol() -> None:
    args = MODULE._parser().parse_args([])
    assert args.exhaustiveness == 8
    assert args.modes == 9
    assert args.bootstrap_replicates == 10_000
