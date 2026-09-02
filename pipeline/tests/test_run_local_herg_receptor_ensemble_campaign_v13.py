from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts/run_local_herg_receptor_ensemble_campaign_v13.py"
SPEC = importlib.util.spec_from_file_location("herg_v13_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_receptor_state_contract_is_fixed() -> None:
    assert MODULE.RECEPTOR_IDS == ("8ZYN", "8ZYP", "9CHP", "9CHQ", "8ZYO", "8ZYQ")
    assert MODULE.CORE_IDS == ("8ZYN", "8ZYP", "9CHP", "9CHQ")
    assert set(MODULE.HOLO_SPECS) == {"8ZYO", "8ZYP", "8ZYQ"}


def test_endpoint_projection_enforces_physical_order() -> None:
    raw = np.array([4.0, 5.0, 4.5])
    projected = MODULE._project_nonincreasing(raw)
    assert projected[0] >= projected[1] >= projected[2]
    assert np.isclose(projected[0], projected[1])


def test_tier_boundaries_match_30_and_1_micromolar() -> None:
    values = np.array(
        [
            MODULE.SAFE_PIC50 - 0.1,
            MODULE.SAFE_PIC50,
            MODULE.POTENT_PIC50,
            MODULE.POTENT_PIC50 + 0.1,
        ]
    )
    assert MODULE._tier_index(values).tolist() == [0, 1, 1, 2]


def test_scaffold_diverse_selection_prefers_unique_scaffolds() -> None:
    frame = pd.DataFrame(
        {
            "structure_id": ["a", "b", "c", "d"],
            "scaffold_group_id": ["same", "same", "different", "third"],
        }
    )
    selected = MODULE._select_diverse(frame, 3, "structure_id", "test")
    assert len(selected) == 3
    assert selected.scaffold_group_id.nunique() == 3


def test_docking_applicability_excludes_runtime_outliers() -> None:
    ordinary = MODULE._dockability("CCN(CC)CCCC(C)Nc1ncc2ccccc2n1")
    oversized = MODULE._dockability("C" * 60)
    unsupported = MODULE._dockability("C[Si](C)(C)C")
    salt = MODULE._dockability("CC[NH+](CC)CC.[Cl-]")
    assert ordinary["docking_eligible"]
    assert not oversized["docking_eligible"]
    assert "heavy_atoms_gt_55" in oversized["docking_exclusion_reason"]
    assert not unsupported["docking_eligible"]
    assert "unsupported_elements_Si" in unsupported["docking_exclusion_reason"]
    assert salt["docking_eligible"]
    assert "." not in salt["docking_parent_smiles"]
    assert salt["docking_removed_fragment_count"] == 1


def test_endpoint_fold_reconstruction_keeps_scaffolds_together() -> None:
    rows = []
    for endpoint in (10, 30, 50):
        for index in range(15):
            rows.append(
                {
                    "endpoint": f"IC{endpoint}",
                    "standard_inchi_key": f"key-{index}",
                    "scaffold_group_id": f"scaffold-{index // 2}",
                }
            )
    folds = MODULE._endpoint_fold_maps(pd.DataFrame(rows))
    assert len(folds) == 15
    assert not folds.isna().any().any()
    assert all(folds[f"endpoint_outer_fold_pic{endpoint}"].nunique() == 5 for endpoint in (10, 30, 50))


def test_class_probability_assignment_preserves_matrix_shape() -> None:
    target = np.full((5, 3), np.nan)
    mask = np.array([False, True, False, True, False])
    probabilities = np.array([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]])
    MODULE._assign_class_probabilities(target, mask, np.array([0, 1, 2]), probabilities)
    assert np.allclose(target[mask], probabilities)
    assert np.isnan(target[~mask]).all()


def test_docking_aggregation_preserves_state_contrasts() -> None:
    rows = []
    scores = {"8ZYN": -7.0, "8ZYP": -8.0, "9CHP": -6.0, "9CHQ": -6.5}
    for state, score in scores.items():
        row = {
            "ligand_id": "ligand",
            "pdb_id": state,
            "best_affinity_kcal_mol": score,
            "microstate_index": 0,
            "ligand_efficiency_kcal_mol_per_heavy_atom": score / 20,
            "pose_count_within_1kcal": 3,
            "affinity_range_kcal_mol": 2.0,
            "formal_charge": 1,
            "minimum_protein_distance_A": 2.0,
        }
        for residue in MODULE.CONTACT_RESIDUES:
            row[f"contact_{residue}_minimum_A"] = 3.0
            row[f"contact_{residue}_ligand_atom_count"] = 2
        rows.append(row)
    aggregated = MODULE._aggregate_docking(pd.DataFrame(rows), MODULE.CORE_IDS)
    assert len(aggregated) == 1
    assert np.isclose(aggregated.iloc[0].dock__e4031_conditioned_minus_apo, -1.0)
    assert np.isclose(aggregated.iloc[0].dock__lowK_minus_highK, -0.5)


def test_parser_defaults_to_current_vina_campaign() -> None:
    args = MODULE._parser().parse_args([])
    assert args.states == ",".join(MODULE.CORE_IDS)
    assert args.redock_exhaustiveness == 32
    assert args.pilot_exhaustiveness == 8
    assert args.bootstrap_replicates == 10_000
