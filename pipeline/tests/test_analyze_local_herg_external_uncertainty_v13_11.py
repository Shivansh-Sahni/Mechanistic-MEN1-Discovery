from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/analyze_local_herg_external_uncertainty_v13_11.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("herg_v13_11_uncertainty_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parser_defaults_point_to_both_external_campaigns() -> None:
    args = MODULE._parser().parse_args([])
    assert "v13_7" in str(args.v13_7_root)
    assert "v13_8" in str(args.v13_8_root)


def test_similarity_bins_have_fixed_left_closed_boundaries() -> None:
    result = MODULE._similarity_bin([0.29, 0.30, 0.50, 0.70, 1.0])
    assert result.tolist() == [
        "lt_0p3",
        "0p3_to_lt_0p5",
        "0p5_to_lt_0p7",
        "ge_0p7",
        "ge_0p7",
    ]


def test_finite_sample_radius_uses_conservative_higher_order_statistic() -> None:
    values = np.arange(1, 11, dtype=float)
    assert MODULE._finite_sample_radius(values, 0.80) == 9.0
    assert MODULE._finite_sample_radius(values, 0.90) == 10.0


def test_wilson_interval_contains_observed_rate() -> None:
    interval = MODULE._wilson(9, 10)
    assert interval[0] < 0.9 < interval[1]


def test_calibration_has_global_and_every_mondrian_radius() -> None:
    frame = pd.DataFrame(
        {
            "observed_pic50": np.linspace(4.0, 7.0, 40),
            "pred__honest_stack": np.linspace(4.1, 6.9, 40),
            "maximum_train_tanimoto": np.tile([0.2, 0.4, 0.6, 0.8], 10),
            "scaffold_group_id": [f"s{index}" for index in range(40)],
        }
    )
    calibration = MODULE._fit_calibration(frame)
    assert calibration["calibration_rows"] == 40
    for level in ("0.80", "0.90", "0.95"):
        assert calibration["coverage_levels"][level]["global_radius_pic50"] > 0
        assert set(
            calibration["coverage_levels"][level][
                "similarity_mondrian_radius_pic50"
            ]
        ) == set(MODULE.SIMILARITY_NAMES)
