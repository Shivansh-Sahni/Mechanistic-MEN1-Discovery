from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/analyze_local_herg_endpoint_order_projection_v12_3.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("herg_v12_3_projection_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_order_violation_counter_detects_any_adjacent_failure() -> None:
    values = np.array([[6.0, 5.0, 4.0], [5.0, 5.5, 4.0], [6.0, 4.0, 4.5]])
    assert MODULE._order_violations(values) == 2


def test_analysis_projection_removes_violations_without_using_labels() -> None:
    frame = pd.DataFrame(
        {
            "standard_inchi_key": ["a", "b", "c"],
            "scaffold_group_id": ["a", "b", "c"],
            "outer_fold": [0, 1, 2],
            "observed_pic10": [6.0, 7.0, 8.0],
            "observed_pic30": [5.0, 6.0, 7.0],
            "observed_pic50": [4.0, 5.0, 6.0],
            "historical_predicted_pic10": [5.0, 6.0, 7.0],
            "historical_predicted_pic30": [6.0, 7.0, 8.0],
            "historical_predicted_pic50": [4.0, 5.0, 6.0],
        }
    )
    predictions, report = MODULE._analyze(frame, 100)
    assert report["baseline_order_violations"] == 3
    assert report["projected_order_violations"] == 0
    assert np.all(
        predictions.projected_historical_pic10 >= predictions.projected_historical_pic30
    )
    assert np.all(
        predictions.projected_historical_pic30 >= predictions.projected_historical_pic50
    )


def test_parser_defaults_to_full_scaffold_bootstrap() -> None:
    args = MODULE._parser().parse_args([])
    assert args.bootstrap_replicates == 10_000
