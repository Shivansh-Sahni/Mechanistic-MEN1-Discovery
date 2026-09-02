from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_local_herg_menin_receptor_prioritization_v13_10.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("herg_v13_10_prioritization_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parser_defaults_define_24_compound_challenge_panel() -> None:
    args = MODULE._parser().parse_args([])
    assert args.stage == "all"
    assert args.panel_size == 24
    assert args.cpu == 6


def test_jensen_shannon_is_zero_for_equal_and_one_for_disjoint() -> None:
    equal = MODULE._jensen_shannon(
        np.array([[0.2, 0.3, 0.5]]), np.array([[0.2, 0.3, 0.5]])
    )
    disjoint = MODULE._jensen_shannon(
        np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]])
    )
    assert np.isclose(equal[0], 0.0)
    assert np.isclose(disjoint[0], 1.0)


def test_challenge_strata_keep_risk_direction_explicit() -> None:
    assert MODULE._challenge_stratum(1, 2) == "hybrid_escalates_to_potent"
    assert MODULE._challenge_stratum(0, 1) == "hybrid_other_risk_escalation"
    assert MODULE._challenge_stratum(2, 1) == "hybrid_risk_deescalation"
    assert MODULE._challenge_stratum(2, 2) == "consensus_potent"
    assert MODULE._challenge_stratum(1, None) == "ligand_only"


def test_prepare_rejects_any_observed_herg_row(tmp_path: Path) -> None:
    source = tmp_path / "plan.csv"
    pd.DataFrame(
        {
            "selection_order": [1],
            "selection_category": ["test"],
            "structure_id": ["STR-1"],
            "standardized_smiles": ["CC"],
            "p_activity_median": [8.0],
            "series_id": ["SER-1"],
            "herg_evidence_status": ["observed_blocker"],
        }
    ).to_csv(source, index=False)
    with pytest.raises(MODULE.CampaignError, match="observed hERG"):
        MODULE._prepare(source, tmp_path / "output", 1)


def test_panel_preserves_categories_and_prefers_unique_series() -> None:
    ranked = pd.DataFrame(
        {
            "selection_category": ["a", "a", "b", "b", "c", "c"],
            "series_id": ["s1", "s1", "s2", "s3", "s4", "s4"],
            "model_challenge_rank": np.arange(1, 7),
        }
    )
    panel = MODULE._select_panel(ranked, 4)
    assert len(panel) == 4
    assert panel.selection_category.nunique() == 3
    assert panel.series_id.nunique() == 4
    assert panel.assay_wave_order.tolist() == [1, 2, 3, 4]


def test_outcome_template_is_blinded_and_contains_no_predictions(tmp_path: Path) -> None:
    panel = pd.DataFrame(
        {
            "structure_id": ["STR-1", "STR-2"],
            "selection_category": ["a", "b"],
            "model_challenge_rank": [1, 2],
            "hybrid_prediction": [2, 0],
        }
    )
    report = MODULE._write_blinded_outcome_template(tmp_path, panel)
    template = pd.read_csv(tmp_path / "analysis/blinded_herg_outcome_release.csv")
    assert report["predictions_or_ranks_present"] is False
    assert "hybrid_prediction" not in template
    assert "model_challenge_rank" not in template
    assert template.blinded_sample_id.str.startswith("H13-").all()
