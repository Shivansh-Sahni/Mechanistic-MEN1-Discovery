from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_herg_mw_pair_delta_attenuation as module  # noqa: E402


def _synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic rows used only to test joins and metric arithmetic."""

    pairs = pd.DataFrame(
        {
            "pair_id": ["synthetic-1", "synthetic-2", "synthetic-3"],
            "structure_id_a": ["A", "C", "E"],
            "structure_id_b": ["B", "D", "F"],
            "pic50_median_a": [5.0, 5.0, 5.0],
            "pic50_median_b": [6.0, 4.0, 5.5],
            "delta_pic50_b_minus_a": [1.0, -1.0, 0.5],
            "absolute_delta_pic50": [1.0, 1.0, 0.5],
        }
    )
    oof = pd.DataFrame(
        {
            "structure_id": ["A", "B", "C", "D", "E", "F"],
            "observed_pic50": [5.0, 6.0, 5.0, 4.0, 5.0, 5.5],
            "prediction": [5.0, 5.5, 5.0, 4.5, 5.0, 5.1],
            "rdkit2d__MolWt": [300.0, 320.0, 520.0, 540.0, 720.0, 740.0],
            "outer_fold": [0, 0, 1, 1, 2, 2],
        }
    )
    return pairs, oof


def test_prepare_pair_frame_constructs_oof_deltas_mw_and_components() -> None:
    pairs, oof = _synthetic_frames()
    rows, summary = module.prepare_pair_frame(pairs, oof, prediction_column="prediction")

    assert summary["retained_pairs"] == 3
    assert rows["predicted_delta_pic50_b_minus_a"].tolist() == pytest.approx([0.5, -0.5, 0.1])
    assert rows["pair_mean_mw_da"].tolist() == pytest.approx([310.0, 530.0, 730.0])
    assert rows["direction_correct"].tolist() == [True, True, True]
    assert rows["mmp_component_id"].nunique() == 3


def test_target_mismatch_is_excluded_and_counted() -> None:
    pairs, oof = _synthetic_frames()
    oof.loc[oof.structure_id.eq("B"), "observed_pic50"] = 5.8
    rows, summary = module.prepare_pair_frame(pairs, oof, prediction_column="prediction")

    assert rows.pair_id.tolist() == ["synthetic-2", "synthetic-3"]
    assert summary["target_mismatch_pairs"] == 1
    assert summary["retained_pairs"] == 2


def test_same_fold_filter_is_explicit() -> None:
    pairs, oof = _synthetic_frames()
    oof.loc[oof.structure_id.eq("B"), "outer_fold"] = 4
    rows, summary = module.prepare_pair_frame(
        pairs,
        oof,
        prediction_column="prediction",
        same_outer_fold_only=True,
    )

    assert rows.pair_id.tolist() == ["synthetic-2", "synthetic-3"]
    assert summary["different_outer_fold_pairs"] == 1


def test_audit_reports_transparent_magnitude_and_l1_metrics() -> None:
    pairs, oof = _synthetic_frames()
    audit = module.audit_mw_pair_delta_attenuation(
        pairs,
        oof,
        prediction_column="prediction",
        primary_minimum_delta_pic50=0.1,
        mw_boundaries=(400.0, 600.0, 700.0),
        bootstrap_replicates=40,
        seed=7,
    )

    overall = audit.report["overall"]
    assert overall["n_pairs"] == 3
    assert overall["magnitude_capture_fraction"] == pytest.approx(1.1 / 2.5)
    assert overall["signed_calibration_slope_through_origin"] == pytest.approx(1.05 / 2.25)
    assert overall["directional_accuracy"] == 1.0
    assert overall["magnitude_capture_fraction_ci95_lower"] is not None
    assert audit.report["reference_training_structure_median_mw_da"] == pytest.approx(530.0)
    high = audit.band_metrics.loc[audit.band_metrics.mw_band.eq(">=700")].iloc[0]
    assert bool(high.sparse_support)
    assert set(audit.cliff_band_metrics.mw_band) == {"<400", "400-600", "600-700", ">=700"}


def test_invalid_measured_delta_contract_is_rejected() -> None:
    pairs, oof = _synthetic_frames()
    pairs.loc[0, "delta_pic50_b_minus_a"] = -1.0
    with pytest.raises(ValueError, match="must equal"):
        module.prepare_pair_frame(pairs, oof, prediction_column="prediction")


def test_stress_test_is_off_by_default_and_primary_is_unchanged() -> None:
    result = module.mw_proportional_delta_stress_test(5.0, 4.0, 600.0, 620.0)

    assert result["enabled"] is False
    assert result["default_enabled"] is False
    assert result["applied"] is False
    assert result["primary_prediction_unchanged"]["candidate_pic50"] == 4.0
    assert result["stress_scenario"]["candidate_pic50"] == 4.0
    assert result["stress_scenario"]["applied_multiplier"] == 1.0


@pytest.mark.parametrize(("candidate", "expected_sign"), [(4.0, -1), (6.0, 1)])
def test_enabled_stress_test_preserves_direction(candidate: float, expected_sign: int) -> None:
    result = module.mw_proportional_delta_stress_test(
        5.0,
        candidate,
        600.0,
        700.0,
        enabled=True,
        reference_median_mw_da=500.0,
        baseline_multiplier=1.5,
        provenance={"scope": "synthetic unit test only"},
    )

    scenario = result["stress_scenario"]
    assert scenario["applied_multiplier"] == pytest.approx(1.95)
    assert scenario["candidate_minus_parent_delta_pic50"] == pytest.approx((candidate - 5.0) * 1.95)
    assert (scenario["candidate_minus_parent_delta_pic50"] > 0) is (expected_sign > 0)
    assert scenario["direction_preserved"] is True
    assert result["primary_prediction_unchanged"]["candidate_pic50"] == candidate


def test_stress_test_optional_cap_is_auditable() -> None:
    result = module.mw_proportional_delta_stress_test(
        5.0,
        6.0,
        800.0,
        800.0,
        enabled=True,
        reference_median_mw_da=400.0,
        baseline_multiplier=2.0,
        maximum_multiplier=2.5,
    )
    assert result["stress_scenario"]["raw_multiplier"] == 4.0
    assert result["stress_scenario"]["applied_multiplier"] == 2.5
    assert "capped" in result["formula"]


def test_enabled_stress_test_requires_empirical_reference() -> None:
    with pytest.raises(ValueError, match="requires a finite, positive reference"):
        module.mw_proportional_delta_stress_test(5.0, 6.0, 500.0, 510.0, enabled=True)
