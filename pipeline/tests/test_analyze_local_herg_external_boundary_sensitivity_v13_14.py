from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_local_herg_external_boundary_sensitivity_v13_14 as module  # noqa: E402

CAMPAIGN = Path("research/local_runs/herg_external_validation_v13_7")
UNCERTAINTY = Path("research/local_runs/herg_external_uncertainty_v13_11")
WORKBOOK = Path("research/external_validation_sources/tx5c00065_si_002.xlsx")


def test_metrics_report_prediction_minus_observed_bias() -> None:
    result = module._metrics(np.array([1.0, 2.0]), np.array([2.0, 4.0]))  # noqa: SLF001
    assert result["mae"] == 1.5
    assert result["bias_prediction_minus_observed"] == 1.5


def test_paired_cluster_bootstrap_detects_uniform_v9_advantage() -> None:
    frame = pd.DataFrame(
        {
            "scaffold_group_id": ["A", "B", "C", "D"],
            "true_pic50_m": [4.0, 5.0, 6.0, 7.0],
            "v9_mixed_ic50_predicted_pic50": [4.1, 5.1, 6.1, 7.1],
            "source_model_prediction_pic50_m": [5.0, 6.0, 7.0, 8.0],
        }
    )
    result = module._paired_cluster_bootstrap(frame, 100)  # noqa: SLF001
    assert result["delta_mae_v9_minus_source"] < -0.8
    assert result["ci95"][1] < 0


def test_default_analysis_traces_four_publisher_exact_extremes(tmp_path: Path) -> None:
    result = module.analyze(CAMPAIGN, UNCERTAINTY, WORKBOOK, tmp_path, bootstrap=100)
    assert result["source_integrity"]["extreme_labels_traced_exactly"] is True
    assert result["source_integrity"]["parsing_error_found"] is False
    assert result["diagnostic_rule"]["exact_values"] == [2.0, 2.0, 3.0, 9.0]
    assert result["metrics"]["primary_all_rows"]["n"] == 110
    assert result["metrics"]["posthoc_non_extreme_sensitivity"]["n"] == 106
    assert result["decision"]["primary_v13_7_metrics_unchanged"] is True
    assert result["decision"]["v9_external_advantage_robust"] is True
    assert result["decision"]["receptor_classification_gain_not_created_by_extremes"] is True
    sensitivity = result["classification_sensitivity"]["posthoc_non_extreme_sensitivity"]
    assert (
        sensitivity["paired_within_class_bootstrap"]["delta_balanced_accuracy_candidate_minus_baseline"] > 0
    )
    trace = pd.read_csv(tmp_path / "publisher_boundary_trace.csv")
    assert len(trace) == 4
    assert np.allclose(trace.publisher_y_true_pic50, trace.true_pic50_m)
