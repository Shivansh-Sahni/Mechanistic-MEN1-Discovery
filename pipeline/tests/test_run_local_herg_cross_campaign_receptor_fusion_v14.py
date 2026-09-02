from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts/run_local_herg_cross_campaign_receptor_fusion_v14.py"
REPO = Path(__file__).parents[2]
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("herg_v14_cross_campaign_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_tier_boundaries_are_strict_and_match_frozen_contract() -> None:
    values = np.asarray(
        [MODULE.SAFE_PIC50 - 0.01, MODULE.SAFE_PIC50, MODULE.POTENT_PIC50, 6.01]
    )
    assert MODULE._tier(values).tolist() == [0, 1, 1, 2]


def test_prepared_receptor_identifies_true_aromatic_cage() -> None:
    receptor = MODULE._parse_receptor_atoms(
        REPO
        / "research/local_runs/herg_receptor_ensemble_campaign_v13/receptors/8ZYO/8ZYO_prepared.pdb"
    )
    assert set(receptor["ring_centroids"]) == {652, 656}
    assert 649 in receptor["by_residue"]
    assert receptor["ring_centroids"][652].shape == (4, 3)
    assert receptor["ring_centroids"][656].shape == (4, 3)


def test_pose_parser_removes_meeko_macrocycle_pseudoatoms() -> None:
    root = REPO / "research/local_runs/herg_receptor_classification_confirmation_v13_3"
    ligand_id = "confirm__HSTR-07FB558E61DC2FCEFB91E73E"
    pose_path = root / f"docking/tasks/{ligand_id}__m0__8ZYO/poses.pdbqt"
    sdf_path = root / f"ligands/{ligand_id}/microstate_0.sdf"
    poses = MODULE._parse_pose_file(pose_path)
    assert len(poses) == 7
    assert all(len(pose["coordinates"]) == 52 for pose in poses)
    assert all(not atom_type.startswith("G") for atom_type in poses[0]["atom_types"])
    roles = MODULE._ligand_roles(sdf_path, poses[0])
    assert len(roles["aromatic"]) == 52
    assert roles["formal_positive"].sum() == 1


def test_campaign_and_class_weights_equalize_both_levels() -> None:
    frame = pd.DataFrame(
        {
            "campaign": ["a"] * 6 + ["b"] * 12,
            "target_class": [0, 1, 1, 2, 2, 2] + [0] * 2 + [1] * 4 + [2] * 6,
        }
    )
    weights = MODULE._classification_weights(frame)
    weighted = frame.assign(weight=weights).groupby(["campaign", "target_class"]).weight.sum()
    assert np.allclose(weighted.to_numpy(), weighted.iloc[0])
    regression = MODULE._regression_weights(frame)
    campaign_weight = frame.assign(weight=regression).groupby("campaign").weight.sum()
    assert np.allclose(campaign_weight.to_numpy(), campaign_weight.iloc[0])


def test_classifier_metrics_include_safety_and_calibration() -> None:
    y = np.tile(np.arange(3), 4)
    probability = np.eye(3)[y]
    metrics = MODULE._classification_metrics(y, probability)
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["potent_predicted_safe_rate"] == 0.0
    assert metrics["multiclass_brier"] < 1e-12


def test_parser_defaults_to_full_evidence_run() -> None:
    args = MODULE._parser().parse_args([])
    assert args.stage == "all"
    assert args.bootstrap_replicates == 5_000
