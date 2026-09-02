from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts/run_local_herg_comprehensive_optimization_v11.py"
SPEC = importlib.util.spec_from_file_location("herg_v11_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_candidate_plan_is_material_and_diverse() -> None:
    candidates = MODULE._candidate_plan()
    assert len(candidates) >= 35
    assert len({row.candidate_id for row in candidates}) == len(candidates)
    assert {row.engine for row in candidates} == {"xgboost", "lightgbm", "extratrees"}
    assert {row.surface for row in candidates} >= {
        "rdkit2d",
        "morgan",
        "anchor",
        "anchor_qc_physics",
        "full_ligand",
    }
    assert {row.training_scope for row in candidates} >= {
        "all_exact",
        "high_quality",
        "patch_or_functional",
    }


def test_smoke_promotion_cannot_escape_smoke_candidate_set() -> None:
    candidates = MODULE._candidate_plan(smoke=True)
    units = [
        {
            "unit_spec": {"candidate": {"candidate_id": row.candidate_id}},
            "metrics": {"mae": float(index + 1)},
        }
        for index, row in enumerate(candidates)
    ]
    promoted = MODULE._promotion_ids(units, candidates, maximum=4)
    assert set(promoted) <= {row.candidate_id for row in candidates}


def test_small_quality_surface_is_exploratory_only() -> None:
    assert MODULE._minimum_training_structures("patch_or_functional") == 250
    assert MODULE._minimum_training_structures("all_exact") == 500
    regular = MODULE.Candidate("regular", "xgboost", "anchor", {})
    exploratory = MODULE.Candidate(
        "exploratory",
        "xgboost",
        "anchor",
        {},
        training_scope="patch_or_functional",
    )
    units = [
        {
            "unit_spec": {"candidate": {"candidate_id": "exploratory"}},
            "metrics": {"mae": 0.1},
        },
        {
            "unit_spec": {"candidate": {"candidate_id": "regular"}},
            "metrics": {"mae": 0.5},
        },
    ]
    assert MODULE._candidate_selection_eligible(exploratory) is False
    assert MODULE._promotion_ids(units, [regular, exploratory], maximum=1) == ["regular"]


def test_training_weights_target_special_populations() -> None:
    joined = pd.DataFrame(
        {
            "structure_id": ["a", "b", "c"],
            "training_target": [3.5, 5.5, 7.0],
            "reliability_weight": [1.0, 1.0, 1.0],
            "quality_weight": [1.0, 1.0, 1.0],
            "rdkit2d__MolWt": [300.0, 550.0, 350.0],
            "rdkit2d__NumRotatableBonds": [2.0, 9.0, 3.0],
        }
    )
    mmp = pd.DataFrame(
        {
            "structure_id_a": ["a"],
            "structure_id_b": ["c"],
            "activity_cliff_ge_1_pic50": [True],
        }
    )
    tail = MODULE.Candidate("tail", "xgboost", "anchor", {}, weight_mode="potency_tail")
    heavy = MODULE.Candidate("heavy", "xgboost", "anchor", {}, weight_mode="heavy_flexible")
    cliff = MODULE.Candidate("cliff", "xgboost", "anchor", {}, weight_mode="cliff_risk")
    tail_weights = MODULE._training_weights(joined, mmp, {"a", "b", "c"}, tail)
    heavy_weights = MODULE._training_weights(joined, mmp, {"a", "b", "c"}, heavy)
    cliff_weights = MODULE._training_weights(joined, mmp, {"a", "b", "c"}, cliff)
    assert tail_weights[0] > tail_weights[1] and tail_weights[2] > tail_weights[1]
    assert heavy_weights[1] > heavy_weights[0]
    assert cliff_weights[0] > cliff_weights[1] and cliff_weights[2] > cliff_weights[1]


def test_merge_predictions_concatenates_folds_before_candidates(tmp_path: Path) -> None:
    units = []
    for candidate_index, candidate in enumerate(("a", "b")):
        for fold in range(3):
            directory = tmp_path / f"{candidate}_{fold}"
            directory.mkdir()
            path = directory / "predictions.parquet"
            frame = pd.DataFrame(
                {
                    "structure_id": [f"s{fold}"],
                    "scaffold_group_id": [f"g{fold}"],
                    "observed_pic50": [5.0 + fold],
                    "inner_fold": [fold],
                    "predicted_pic50": [5.0 + fold + 0.1 * candidate_index],
                }
            )
            frame.to_parquet(path, index=False)
            units.append(
                {
                    "unit_spec": {"candidate": {"candidate_id": candidate}},
                    "artifacts": [MODULE._binding(path, "predictions")],
                }
            )
    merged = MODULE._merge_predictions(units)
    assert len(merged) == 3
    assert set(merged) >= {"pred__a", "pred__b"}
    assert np.allclose(merged.pred__b - merged.pred__a, 0.1)


def test_parser_defaults_to_full_five_outer_folds() -> None:
    args = MODULE._parser().parse_args(["run"])
    assert args.workers == 6
    assert args.outer_limit == 5
    assert args.bootstrap_replicates == 10_000
