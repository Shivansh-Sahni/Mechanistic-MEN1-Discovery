from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts/run_local_herg_external_validation_v13_7.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("herg_v13_7_external_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_external_identifiers_are_stable_and_namespace_separated() -> None:
    first = MODULE._external_structure_id("AAAA-BBBB-C", "CC")
    second = MODULE._external_structure_id("AAAA-BBBB-C", "unseen-fallback")
    scaffold = MODULE._external_scaffold_id("c1ccccc1")
    assert first == second
    assert first.startswith("EXT-")
    assert scaffold.startswith("EXTSCF-")
    assert first != scaffold
    assert MODULE._connectivity_key("AAAA-BBBB-C") == "AAAA"


def test_regression_metrics_report_exact_errors_and_bias() -> None:
    report = MODULE._regression_metrics(
        np.array([4.0, 5.0, 6.0]),
        np.array([4.0, 5.5, 5.5]),
    )
    assert report["n"] == 3
    assert np.isclose(report["mae"], 1 / 3)
    assert np.isclose(report["rmse"], np.sqrt(1 / 6))
    assert np.isclose(report["mean_error_prediction_minus_observed"], 0.0)


def test_classification_metrics_preserve_potent_to_safe_safety_readout() -> None:
    y = np.array([0, 1, 2, 2])
    prediction = np.array([0, 1, 0, 2])
    report = MODULE._classification_metrics(y, prediction, [0, 1, 2])
    assert report["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [1, 0, 1]]
    assert report["potent_predicted_safe_rate"] == 0.5
    assert report["potent_predicted_safe_count"] == 1
    assert report["observed_potent_count"] == 2
    assert report["potent_predicted_safe_wilson_ci95"][0] < 0.5
    assert report["potent_predicted_safe_wilson_ci95"][1] > 0.5


def test_parser_defaults_match_frozen_v13_protocol() -> None:
    args = MODULE._parser().parse_args([])
    assert args.stage == "all"
    assert args.exhaustiveness == 8
    assert args.modes == 9
    assert args.cpu == 6
    assert args.bootstrap_replicates == 10_000


def test_tier_boundaries_match_v13() -> None:
    values = np.array([MODULE.v13.SAFE_PIC50 - 1e-6, MODULE.v13.SAFE_PIC50, 6.0, 6.000001])
    assert MODULE._tier(values).tolist() == [0, 1, 1, 2]


def test_probability_metrics_are_zero_for_perfect_probabilities() -> None:
    y = np.array([0, 1, 2])
    report = MODULE._probability_metrics(y, np.eye(3))
    assert report["multiclass_brier"] == 0.0
    assert report["top_class_ece_10_fixed_bins"] == 0.0


def test_scaffold_cluster_bootstrap_detects_uniform_improvement() -> None:
    y = np.tile(np.arange(3), 6)
    scaffolds = np.repeat([f"s{index}" for index in range(6)], 3)
    baseline = np.ones(len(y), dtype=int)
    candidate = y.copy()
    report = MODULE._scaffold_bootstrap_classification(
        scaffolds, y, baseline, candidate, 100
    )
    assert report["ci95"][0] > 0
    assert report["macro_f1_ci95"][0] > 0
    assert report["unique_scaffold_clusters"] == 6
