from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_local_herg_v10_tiered_platform.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("herg_v10", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tier_boundaries() -> None:
    module = _module()
    values = np.array([module.SAFE_PIC50 - 0.01, module.SAFE_PIC50, 6.0, 6.01])
    assert module.tier_index(values).tolist() == [0, 1, 1, 2]


def test_pic50_and_hill_conversions() -> None:
    module = _module()
    assert math.isclose(module.ic50_um_from_pic50(6.0), 1.0)
    assert math.isclose(module.hill_icx_um(9.0, 0.1), 1.0)
    assert math.isclose(module.hill_icx_um(7.0, 0.3), 3.0)


def test_smiles_feature_extraction_matches_frozen_names() -> None:
    module = _module()
    columns = ["rdkit2d__MolWt", "rdkit2d__MolLogP", "morgan__0000", "morgan__2047"]
    frame, molecule = module._feature_frame("CCO", columns)
    assert molecule.GetNumAtoms() == 3
    assert frame.columns.tolist() == columns
    assert np.isfinite(frame["rdkit2d__MolWt"]).all()
    assert set(frame[["morgan__0000", "morgan__2047"]].to_numpy().ravel()) <= {0, 1}


def test_ui_states_endpoint_and_scope_limits() -> None:
    module = _module()
    html = module._html()
    assert "Derived, not trained" in html
    assert "Repository validation and test labels remain sealed" in html
    assert "broad fixed-dose surface is not yet a valid three-class dataset" in html
