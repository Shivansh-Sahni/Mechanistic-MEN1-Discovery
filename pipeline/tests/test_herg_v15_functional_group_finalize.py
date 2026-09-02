from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parents[2]
SCRIPT_DIR = Path(__file__).parents[1] / "scripts"
FEATURE_SCRIPT = SCRIPT_DIR / "herg_v15_functional_group_features.py"
STUDY_SCRIPT = SCRIPT_DIR / "run_local_herg_functional_group_finalize_v15.py"
MATRIX = (
    REPO_ROOT
    / "research/local_runs/herg_cross_campaign_receptor_fusion_v14/data/harmonized_campaigns.parquet"
)


def _load(name: str, path: Path) -> ModuleType:
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FG = _load("herg_v15_functional_group_features_test", FEATURE_SCRIPT)
STUDY = _load("herg_v15_functional_group_finalize_test", STUDY_SCRIPT)


def test_registry_is_versioned_integrity_checked_and_governed() -> None:
    registry = FG.load_registry()
    assert registry["schema_version"] == "herg-functional-group-registry/1.0"
    assert registry["registry_version"] == "v1.0.0"
    keys = {rule["key"] for rule in registry["rules"]}
    assert {
        "amide",
        "primary_aliphatic_amine",
        "tertiary_aliphatic_amine",
        "ether",
        "pyridine_like_aromatic_n",
        "carboxylic_acid",
        "guanidine",
        "aryl_halogen",
        "formal_cation_atom",
        "aryl_one_atom_amine_link",
    } <= keys
    assert "causal hERG mechanisms" in registry["scientific_boundary"]


def test_registry_rejects_content_changed_without_new_hash(tmp_path: Path) -> None:
    value = json.loads(FG.DEFAULT_REGISTRY.read_text())
    value["rules"][0]["label"] = "Changed after governance"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(value))
    with pytest.raises(FG.FunctionalGroupFeatureError, match="hash mismatch"):
        FG.load_registry(path)


@pytest.mark.parametrize(
    ("smiles", "present", "absent"),
    [
        ("CC(=O)Nc1ccccc1", ("amide",), ("aniline_like_nitrogen",)),
        ("CCN(CC)CC", ("tertiary_aliphatic_amine",), ("amide",)),
        ("COc1ccccc1", ("ether", "aromatic_ether_link"), ("amide",)),
        ("n1ccccc1", ("pyridine_like_aromatic_n",), ("pyrrole_like_aromatic_nh",)),
        ("O=C(O)c1ccccc1", ("carboxylic_acid",), ("ester",)),
        ("C[N+](C)(C)C", ("quaternary_ammonium", "formal_cation_atom"), ("formal_anion_atom",)),
        ("Clc1ccccc1", ("chlorine", "aryl_halogen"), ("bromine",)),
    ],
)
def test_governed_smarts_detect_expected_motifs(
    smiles: str,
    present: tuple[str, ...],
    absent: tuple[str, ...],
) -> None:
    row = FG.build_feature_row(smiles)
    for key in present:
        assert row[f"fg_present__{key}"] == 1
        assert row[f"fg_count__{key}"] >= 1
    for key in absent:
        assert row[f"fg_present__{key}"] == 0
        assert row[f"fg_count__{key}"] == 0


def test_charge_context_and_controlled_interactions_are_calculated_not_rules() -> None:
    neutral_base = FG.build_feature_row("CCN(CC)CC")
    assert neutral_base["formal_charge"] == 0
    assert neutral_base["positive_ionizable_centers"] >= 1
    assert neutral_base["clogp_x_positive_ionizable"] == pytest.approx(
        neutral_base["clogp"] * neutral_base["positive_ionizable_centers"]
    )
    cation = FG.build_feature_row("C[N+](C)(C)C")
    assert cation["formal_charge"] == 1
    assert cation["permanent_positive_atoms"] == 1
    assert cation["mw_x_abs_formal_charge"] == pytest.approx(cation["molecular_weight"])
    numeric = np.asarray(
        [cation[column] for column in FG.feature_columns(FG.load_registry())], dtype=float
    )
    assert np.isfinite(numeric).all()


def test_basic_center_on_aromatic_atom_has_zero_graph_distance() -> None:
    aromatic_base = FG.build_feature_row("c1ncc[nH]1")
    assert aromatic_base["positive_ionizable_centers"] >= 1
    assert aromatic_base["aromatic_ring_count"] == 1
    assert aromatic_base["basic_aromatic_min_topological_distance"] == 0


@pytest.mark.parametrize("smiles", ["", "not-a-smiles", "C1CC"])
def test_feature_builder_rejects_empty_and_invalid_smiles(smiles: str) -> None:
    with pytest.raises(FG.FunctionalGroupFeatureError):
        FG.build_feature_row(smiles)


def test_feature_frame_is_deterministic_and_has_exact_governed_columns() -> None:
    registry = FG.load_registry()
    smiles = ["CCN(CC)CC", "COc1ccccc1"]
    left = FG.build_feature_frame(smiles, registry)
    right = FG.build_feature_frame(smiles, registry)
    pd.testing.assert_frame_equal(left, right)
    assert list(left) == ["canonical_smiles", *FG.feature_columns(registry)]
    assert not left.isna().any().any()


def test_surface_ablation_adds_fg_then_only_prespecified_interactions() -> None:
    registry = FG.load_registry()
    surfaces = STUDY._surface_contracts(registry)
    physchem = set(surfaces["physchem_absolute_ridge"]["columns"])
    functional = set(surfaces["fg_absolute_ridge"]["columns"])
    interaction = set(surfaces[STUDY.PRIMARY_CANDIDATE]["columns"])
    assert set(FG.CHARGE_CONTEXT_FEATURES) <= physchem
    assert all(not column.startswith("fg_") for column in physchem)
    assert any(column.startswith("fg_count__") for column in functional)
    assert set(FG.INTERACTION_FEATURES).isdisjoint(functional)
    assert interaction - functional == set(FG.INTERACTION_FEATURES)


def test_frozen_campaign_matrix_has_zero_outer_exact_and_scaffold_overlap() -> None:
    matrix = pd.read_parquet(MATRIX)
    audit = STUDY._audit_matrix(matrix)
    assert audit["rows"] == 1_224
    assert audit["campaigns"] == 6
    assert audit["cross_campaign_exact_overlap"] == 0
    assert audit["cross_campaign_scaffold_overlap"] == 0

    altered = matrix.copy()
    first_campaign, second_campaign = sorted(altered.campaign.unique())[:2]
    donor = altered.loc[altered.campaign.eq(first_campaign)].iloc[0]
    receiver_index = altered.index[altered.campaign.eq(second_campaign)][0]
    altered.loc[receiver_index, "cross_campaign_connectivity_key"] = (
        donor.cross_campaign_connectivity_key
    )
    with pytest.raises(STUDY.StudyError, match="overlap"):
        STUDY._audit_matrix(altered)


def test_pair_builder_is_same_campaign_scaffold_deterministic_and_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(STUDY, "PAIR_MINIMUM_TANIMOTO", 0.0)
    frame = pd.DataFrame(
        {
            "sample_id": [f"molecule-{index}" for index in range(6)],
            "campaign": ["held-campaign"] * 6,
            "cross_campaign_scaffold_key": ["one-series-proxy"] * 6,
            "standardized_smiles": [
                "CCOc1ccccc1",
                "CCOc1ccc(F)cc1",
                "CCOc1ccc(Cl)cc1",
                "CCOc1ccc(Br)cc1",
                "CCOc1ccc(C)cc1",
                "CCOc1ccc(OC)cc1",
            ],
            "target_pic50": np.linspace(4.5, 6.0, 6),
        }
    )
    first = STUDY._build_pairs(frame)
    second = STUDY._build_pairs(frame)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == STUDY.PAIR_MAXIMUM_PER_SCAFFOLD
    assert first.campaign.eq("held-campaign").all()
    assert first.scaffold.eq("one-series-proxy").all()
    assert (first.left_sample_id != first.right_sample_id).all()


def test_tier_boundaries_preserve_frozen_contract() -> None:
    values = np.asarray(
        [STUDY.SAFE_PIC50 - 0.01, STUDY.SAFE_PIC50, STUDY.POTENT_PIC50, 6.01]
    )
    assert STUDY._tier_index(values).tolist() == [0, 1, 1, 2]
