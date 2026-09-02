from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).parents[1] / "scripts/run_local_herg_endpoint_xgb_ensemble_v12_4.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("herg_v12_4_ensemble_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_confirmation_collapse_excludes_every_strict_structure() -> None:
    labels = pd.DataFrame(
        {
            "standard_inchi_key": ["strict", "strict", "holdout", "holdout"],
            "standardized_smiles": ["CC", "CC", "CCC", "CCC"],
            "scaffold_group_id": ["a", "a", "b", "b"],
            "strict_curve_qc": [True, False, False, False],
            "empirical_ic10_um": [1.0, 2.0, 10.0, 100.0],
        }
    )
    strict = MODULE._collapsed_endpoint(labels, 10, strict=True)
    confirmation = MODULE._collapsed_endpoint(
        labels,
        10,
        strict=False,
        excluded_keys=set(strict.standard_inchi_key),
    )
    assert strict.standard_inchi_key.tolist() == ["strict"]
    assert confirmation.standard_inchi_key.tolist() == ["holdout"]
    expected = np.median(6.0 - np.log10(np.array([10.0, 100.0])))
    assert np.isclose(confirmation.iloc[0].target, expected)


def test_historical_surface_contract_matches_v10_1_winners() -> None:
    assert MODULE.HISTORICAL_WINNER_SURFACE == {
        10: "rdkit2d_morgan",
        30: "rdkit2d_morgan",
        50: "rdkit2d",
    }


def test_parser_defaults_to_full_scaffold_bootstrap() -> None:
    args = MODULE._parser().parse_args([])
    assert args.workers == 6
    assert args.bootstrap_replicates == 10_000
