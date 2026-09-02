from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts/run_local_herg_temporal_confirmation_v13_8.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("herg_v13_8_temporal_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parser_defaults_match_frozen_receptor_protocol() -> None:
    args = MODULE._parser().parse_args([])
    assert args.stage == "all"
    assert args.exhaustiveness == 8
    assert args.modes == 9
    assert args.cpu == 6
    assert args.bootstrap_replicates == 10_000


def test_scaffold_bootstrap_interval_is_finite_and_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "scaffold_group_id": ["a", "a", "b", "c"],
            "true_pic50_m": [4.0, 4.5, 5.0, 6.0],
            "prediction": [4.1, 4.4, 5.2, 5.7],
        }
    )
    first = MODULE._cluster_mae_interval(frame, "prediction", 100)
    second = MODULE._cluster_mae_interval(frame, "prediction", 100)
    assert first == second
    assert np.isclose(first["observed_mae"], 0.175)
    assert first["ci95"][0] <= first["observed_mae"] <= first["ci95"][1]


def test_source_constants_identify_official_jcim_supplement() -> None:
    assert MODULE.SOURCE_DOI == "10.1021/acs.jcim.6c00163"
    assert MODULE.SUPPLEMENT_DOI.endswith(".s001")
    assert len(MODULE.EXPECTED_WORKBOOK_SHA256) == 64


def test_receptor_score_join_preserves_the_evaluation_scaffold_column() -> None:
    frame = pd.DataFrame(
        {
            "external_structure_id": ["EXT-1"],
            "scaffold_group_id": ["evaluation-scaffold"],
        }
    )
    receptor = pd.DataFrame(
        {
            "external_structure_id": ["EXT-1"],
            "scaffold_group_id": ["duplicate-sealed-metadata"],
            "hybrid_prediction": [1],
            MODULE.v136.RECEPTOR_COLUMNS[0]: [0.2],
            MODULE.v136.RECEPTOR_COLUMNS[1]: [0.7],
            MODULE.v136.RECEPTOR_COLUMNS[2]: [0.1],
            MODULE.v137.PRIMARY_FEATURE: [7.0],
            "dock__8ZYO__affinity": [-10.0],
            "dock__8ZYO__ligand_efficiency": [-0.3],
        }
    )
    selected = MODULE._receptor_score_frame(frame, receptor)
    assert selected.scaffold_group_id.tolist() == ["evaluation-scaffold"]
    assert not any(column.endswith("_x") or column.endswith("_y") for column in selected)
