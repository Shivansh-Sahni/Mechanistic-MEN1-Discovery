from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_local_herg_nonlinear_receptor_stress_test_v14_1.py"
)
V14_ROOT = Path(__file__).parents[2] / "research/local_runs/herg_cross_campaign_receptor_fusion_v14"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("herg_v14_1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _matrix(module: ModuleType) -> pd.DataFrame:
    frame = pd.read_parquet(V14_ROOT / "data/harmonized_campaigns.parquet")
    pose = pd.read_parquet(V14_ROOT / "data/pose_ensemble_features.parquet")
    return module.v14._build_model_matrix(frame, pose)


def test_tree_surface_contract_preserves_matched_ligand_comparator() -> None:
    module = _module()
    surfaces = module._tree_surfaces(_matrix(module))
    assert set(surfaces) == {
        "extratrees_ligand_core",
        "extratrees_ligand_physchem",
        "extratrees_receptor_frozen",
        "extratrees_receptor_mechanistic",
        "extratrees_receptor_combined",
        "extratrees_receptor_all_pose",
    }
    matched = set(surfaces["extratrees_ligand_physchem"])
    assert matched < set(surfaces["extratrees_receptor_frozen"])
    assert matched < set(surfaces["extratrees_receptor_mechanistic"])
    for features in surfaces.values():
        assert "target_class" not in features
        assert "target_pic50" not in features
        assert len(features) == len(set(features))


def test_tree_configurations_are_bounded_and_distinct() -> None:
    module = _module()
    names = [config["name"] for config in module.TREE_CONFIGS]
    assert len(names) == len(set(names)) == 3
    assert [config["min_samples_leaf"] for config in module.TREE_CONFIGS] == [6, 12, 24]


def test_tree_models_fit_weighted_three_class_and_residual_contracts() -> None:
    module = _module()
    rng = np.random.default_rng(7)
    rows = []
    for campaign in ("a", "b"):
        for class_index in range(3):
            for replicate in range(4):
                rows.append(
                    {
                        "campaign": campaign,
                        "target_class": class_index,
                        "target_pic50": 4.0 + class_index + 0.01 * replicate,
                        "baseline_pic50": 4.1 + class_index,
                        "x1": class_index + rng.normal(0, 0.1),
                        "x2": rng.normal(),
                    }
                )
    frame = pd.DataFrame(rows)
    features = ("x1", "x2")
    classifier = module._fit_classifier(frame, features, module.TREE_CONFIGS[0], 64)
    probabilities = classifier.predict_proba(frame[list(features)])
    assert probabilities.shape == (len(frame), 3)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(len(frame)))
    regressor = module._fit_regressor(frame, features, module.TREE_CONFIGS[0], 64)
    residual = regressor.predict(frame[list(features)])
    assert residual.shape == (len(frame),)
    assert np.isfinite(residual).all()


def test_v14_input_report_self_hash_is_enforced() -> None:
    module = _module()
    report = module._read_json(V14_ROOT / "analysis_report.json", "report_sha256")
    assert report["dataset"]["campaigns"] == 6
    assert report["dataset"]["rows"] == 1_224
