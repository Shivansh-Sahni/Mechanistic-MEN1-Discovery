from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/build_local_herg_receptor_inference_bundle_v13_9.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("herg_v13_9_bundle_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parser_defaults_match_frozen_external_protocol() -> None:
    args = MODULE._parser().parse_args([])
    assert args.stage == "all"
    assert args.cpu == 6
    assert args.smiles == []
    assert "v13_11" in str(args.v13_11_root)


def test_input_registry_standardizes_and_deduplicates_connectivity() -> None:
    registry = MODULE._input_registry(["CCO", "OCC"])
    assert len(registry) == 1
    assert registry.external_structure_id.str.startswith("EXT-").all()
    assert registry.ligand_id.str.startswith("inference__EXT-").all()
    assert registry.docking_eligible.dtype == bool


def test_input_registry_rejects_invalid_smiles() -> None:
    with pytest.raises(MODULE.CampaignError, match="failed standardization"):
        MODULE._input_registry(["not-a-smiles"])


def test_hybrid_frame_applies_frozen_potent_override() -> None:
    base = pd.DataFrame(
        {
            "external_structure_id": ["EXT-1"],
            "structure_id": ["EXT-1"],
            "ligand_id": ["inference__EXT-1"],
            "scaffold_group_id": ["EXTSCF-1"],
            "baseline_prediction": [5.2],
            MODULE.v136.LGBM_COLUMNS[0]: [0.2],
            MODULE.v136.LGBM_COLUMNS[1]: [0.7],
            MODULE.v136.LGBM_COLUMNS[2]: [0.1],
        }
    )
    result = MODULE._hybrid_frame(base, np.array([[0.1, 0.2, 0.7]]))
    assert result.hybrid_prediction.tolist() == [2]
    assert np.isclose(
        result[list(MODULE.v136.RECEPTOR_COLUMNS)].sum(axis=1).iloc[0], 1.0
    )


def test_schema_and_bundle_name_are_versioned() -> None:
    assert "v13.9" in MODULE.SCHEMA_VERSION
    assert MODULE.BUNDLE_NAME.endswith(".joblib")


def test_finite_sample_radius_uses_conservative_order_statistic() -> None:
    values = np.arange(1, 11, dtype=float)
    assert MODULE._finite_sample_radius(values, 0.8) == 9.0
    assert MODULE._finite_sample_radius(values, 0.9) == 10.0


def test_model_card_keeps_vina_and_validation_claims_bounded() -> None:
    card = MODULE._render_model_card(
        {
            "bundle_sha256": "a" * 64,
            "receptor_models": 5,
            "receptor_training_rows": 90,
            "validated_vina_binary_sha256": "b" * 64,
        }
    )
    assert "not experimental affinities" in card
    assert "not prospectively validated" in card
    assert "IC10 <= IC30 <= IC50" in card


def test_inference_artifact_integrity_checks_vina_and_receptor_hashes(
    tmp_path: Path,
) -> None:
    vina = tmp_path / "vina"
    vina.write_bytes(b"validated-vina")
    receptor_root = tmp_path / "receptors" / MODULE.STATE
    receptor_root.mkdir(parents=True)
    hashes = {}
    for name in MODULE.RECEPTOR_FILES:
        path = receptor_root / name
        path.write_text(name)
        hashes[name] = MODULE._sha(path)
    bundle = {
        "vina_protocol": {"binary_sha256": MODULE._sha(vina)},
        "receptor_file_sha256": hashes,
    }
    MODULE._validate_inference_artifacts(
        bundle, tmp_path, SimpleNamespace(vina=vina)
    )
    (receptor_root / MODULE.RECEPTOR_FILES[0]).write_text("changed")
    with pytest.raises(MODULE.CampaignError, match="integrity check"):
        MODULE._validate_inference_artifacts(
            bundle, tmp_path, SimpleNamespace(vina=vina)
        )


def test_stale_verification_is_not_accepted_after_bundle_rebuild(tmp_path: Path) -> None:
    bundle_path = tmp_path / "models" / MODULE.BUNDLE_NAME
    bundle_path.parent.mkdir(parents=True)
    bundle_path.write_bytes(b"first bundle")
    MODULE._json(
        tmp_path / "verification_report.json",
        {
            "bundle_sha256": MODULE._sha(bundle_path),
            "all_passed": True,
        },
        "report_sha256",
    )
    assert MODULE._verification_is_current(tmp_path)
    bundle_path.write_bytes(b"rebuilt bundle")
    assert not MODULE._verification_is_current(tmp_path)
