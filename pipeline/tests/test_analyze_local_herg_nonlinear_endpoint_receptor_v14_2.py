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
    / "scripts/analyze_local_herg_nonlinear_endpoint_receptor_v14_2.py"
)
V132_ROOT = Path(__file__).parents[2] / "research/local_runs/herg_receptor_strict_analysis_v13_2"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("herg_v14_2", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _direct() -> pd.DataFrame:
    frame = pd.read_parquet(V132_ROOT / "six_state_analysis_matrix.parquet")
    return frame.loc[frame.cohort.eq("direct_curve")].reset_index(drop=True)


def test_feature_surfaces_have_matched_ligand_base_and_no_targets() -> None:
    module = _module()
    surfaces = module._feature_surfaces(_direct())
    ligand = set(surfaces["extratrees_ligand_physchem"])
    assert len(surfaces) == 6
    for name, features in surfaces.items():
        assert len(features) == len(set(features))
        assert not any("observed" in feature or "baseline_predicted" in feature for feature in features)
        if name != "extratrees_ligand_physchem":
            assert ligand < set(features)


def test_strict_baseline_is_unique_across_v13_2_feature_sets() -> None:
    module = _module()
    predictions = pd.read_parquet(V132_ROOT / "strict_nested_predictions.parquet")
    table = module._strict_baseline_table(predictions)
    assert table.shape == (90, 3)
    assert list(table.columns) == ["IC10", "IC30", "IC50"]
    assert np.isfinite(table.to_numpy(float)).all()


def test_scaffold_bootstrap_delta_has_expected_sign_and_fold_count() -> None:
    module = _module()
    frame = pd.DataFrame(
        {
            "scaffold_group_id": ["a", "b", "c", "d"],
            "outer_fold": [0, 1, 2, 3],
            "observed": [1.0, 2.0, 3.0, 4.0],
            "candidate": [1.0, 2.0, 3.0, 4.0],
            "comparator": [1.5, 2.5, 3.5, 4.5],
        }
    )
    result = module._bootstrap_delta(frame, "candidate", "comparator", 500, "test")
    assert result["delta_mae_candidate_minus_comparator"] == pytest.approx(-0.5)
    assert result["ci95"][1] < 0
    assert result["folds_better"] == 4


def test_tree_configuration_search_is_bounded() -> None:
    module = _module()
    assert [item["min_samples_leaf"] for item in module.TREE_CONFIGS] == [4, 4, 8, 16]
    assert len({item["name"] for item in module.TREE_CONFIGS}) == 4
