from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_herg_paired_edits as module  # noqa: E402


def _synthetic_pairs() -> pd.DataFrame:
    """Synthetic values used only to unit-test metric arithmetic."""

    return pd.DataFrame(
        {
            "pair_id": ["synthetic-1", "synthetic-2", "synthetic-3", "synthetic-4"],
            "parent_smiles": ["CC", "CCC", "CCO", "CCN"],
            "candidate_smiles": ["CO", "CCCO", "CCCO", "CCCN"],
            "parent_measured_ic50_um": [8.0, 1.0, 5.0, 10.0],
            "candidate_measured_ic50_um": [2.0, 10.0, 5.0, 1.0],
            "parent_predicted_ic50_um": [8.0, 2.0, 4.0, 10.0],
            "candidate_predicted_ic50_um": [6.0, 6.0, 4.0, 20.0],
        }
    )


def test_precomputed_predictions_report_direction_and_magnitude() -> None:
    evaluation = module.evaluate_paired_edits(_synthetic_pairs(), tie_threshold_log10=0.05)
    metrics = evaluation.summary["prediction_metrics"]

    assert evaluation.summary["measured_direction_counts"] == {
        "improved": 1,
        "worsened": 2,
        "negligible": 1,
    }
    assert metrics["n_direction_evaluable"] == 3
    assert metrics["n_direction_correct"] == 2
    assert metrics["directional_accuracy"] == pytest.approx(2 / 3)
    first = evaluation.rows.iloc[0]
    assert first["measured_delta_log10_ic50"] == pytest.approx(np.log10(2 / 8))
    assert first["predicted_delta_log10_ic50"] == pytest.approx(np.log10(6 / 8))
    assert first["cliff_capture_ratio"] == pytest.approx(abs(np.log10(6 / 8) / np.log10(2 / 8)))
    assert pd.isna(evaluation.rows.loc[2, "direction_correct"])


def test_predictor_callback_is_called_once_in_documented_order() -> None:
    pairs = _synthetic_pairs().drop(columns=list(module.PREDICTION_COLUMNS)).iloc[:2].copy()
    calls: list[list[str]] = []

    def predictor(smiles: list[str]) -> list[float]:
        calls.append(list(smiles))
        return [8.0, 1.0, 4.0, 9.0]

    evaluation = module.evaluate_paired_edits(pairs, predictor=predictor)

    assert calls == [[*pairs.parent_smiles, *pairs.candidate_smiles]]
    assert evaluation.summary["predictions_available"] is True
    assert evaluation.rows[module.PREDICTION_COLUMNS[0]].tolist() == [8.0, 1.0]
    assert evaluation.rows[module.PREDICTION_COLUMNS[1]].tolist() == [4.0, 9.0]


def test_aligned_prediction_arrays_do_not_require_model_loading() -> None:
    pairs = _synthetic_pairs().drop(columns=list(module.PREDICTION_COLUMNS)).iloc[:2].copy()
    evaluation = module.evaluate_paired_edits(
        pairs,
        parent_predictions_ic50_um=[8.0, 1.0],
        candidate_predictions_ic50_um=[4.0, 9.0],
    )
    assert evaluation.summary["prediction_metrics"]["directional_accuracy"] == 1.0


def test_measured_only_input_reports_no_fabricated_prediction_metrics() -> None:
    pairs = _synthetic_pairs().drop(columns=list(module.PREDICTION_COLUMNS))
    evaluation = module.evaluate_paired_edits(pairs)
    assert evaluation.summary["predictions_available"] is False
    assert evaluation.summary["prediction_metrics"] is None
    assert "predicted_direction" not in evaluation.rows


def test_all_measured_ties_have_safe_null_metrics() -> None:
    pairs = _synthetic_pairs().iloc[:1].copy()
    pairs["candidate_measured_ic50_um"] = 8.0
    evaluation = module.evaluate_paired_edits(pairs, tie_threshold_log10=0.1)
    metrics = evaluation.summary["prediction_metrics"]
    assert metrics["n_direction_evaluable"] == 0
    assert metrics["directional_accuracy"] is None
    assert metrics["cliff_capture_ratio"] is None
    assert np.isnan(evaluation.rows.loc[0, "cliff_capture_ratio"])


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.assign(parent_measured_ic50_um=0.0), "finite and positive"),
        (lambda frame: pd.concat([frame, frame], ignore_index=True), "pair_id values must be unique"),
        (
            lambda frame: frame.drop(columns=["candidate_predicted_ic50_um"]),
            "Precomputed predictions require both",
        ),
    ],
)
def test_invalid_inputs_are_rejected(mutator, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        module.evaluate_paired_edits(mutator(_synthetic_pairs().iloc[:1].copy()))


def test_reversed_duplicate_structure_pair_is_rejected() -> None:
    first = _synthetic_pairs().iloc[:1].copy()
    second = first.copy()
    second["pair_id"] = "synthetic-reversed"
    second[["parent_smiles", "candidate_smiles"]] = second[["candidate_smiles", "parent_smiles"]].to_numpy()
    with pytest.raises(ValueError, match="Duplicate or reversed"):
        module.evaluate_paired_edits(pd.concat([first, second], ignore_index=True))


def test_configurable_mw_and_similarity_strata() -> None:
    pairs = _synthetic_pairs().copy()
    pairs["parent_mw"] = [400.0, 600.0, 800.0, 500.0]
    pairs["candidate_mw"] = [410.0, 620.0, 810.0, 510.0]
    pairs["similarity"] = [0.95, 0.85, 0.75, 0.65]
    pairs = module._validate_pairs(pairs)  # noqa: SLF001
    pairs = module.add_numeric_band(
        pairs,
        source_column="pair_max_mw",
        output_column="mw_band",
        cutoffs=[650.0, 750.0],
        unit="Da",
    )
    pairs = module.add_numeric_band(
        pairs,
        source_column="similarity",
        output_column="similarity_band",
        cutoffs=[0.8, 0.9],
    )
    evaluation = module.evaluate_paired_edits(
        pairs, tie_threshold_log10=0.05, stratify_by=["mw_band", "similarity_band"]
    )
    assert set(evaluation.strata.stratifier) == {"mw_band", "similarity_band"}
    assert evaluation.strata.groupby("stratifier", observed=True).n_pairs.sum().to_dict() == {
        "mw_band": 4,
        "similarity_band": 4,
    }


def test_rdkit_pair_features_are_real_calculations() -> None:
    pairs = _synthetic_pairs().drop(columns=list(module.PREDICTION_COLUMNS)).iloc[:1].copy()
    result = module.add_rdkit_pair_features(module._validate_pairs(pairs))  # noqa: SLF001
    assert result.loc[0, "parent_mw"] > 0
    assert result.loc[0, "candidate_mw"] > 0
    assert 0 <= result.loc[0, "similarity"] <= 1
