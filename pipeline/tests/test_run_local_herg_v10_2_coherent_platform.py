from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts/run_local_herg_v10_2_coherent_platform.py"
SPEC = importlib.util.spec_from_file_location("herg_v102", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_tiers_come_from_same_ic50_thresholds() -> None:
    assert MODULE._tier(-math.log10(31e-6)) == "Safe"
    assert MODULE._tier(-math.log10(30e-6)) == "Moderate"
    assert MODULE._tier(-math.log10(1e-6)) == "Moderate"
    assert MODULE._tier(-math.log10(0.9e-6)) == "Potent"


def test_ic50_round_trip() -> None:
    for value in (0.1, 1.0, 30.0, 100.0):
        pic50 = -math.log10(value * 1e-6)
        assert math.isclose(MODULE._ic_um(pic50), value, rel_tol=1e-12)


def test_direct_curve_projection_is_ordered() -> None:
    projected, adjusted = MODULE.v101._coherent_threshold_concentrations([8.0, 4.0, 12.0])
    assert adjusted
    assert projected[0] <= projected[1] <= projected[2]


def test_safe_numeric_masks_extreme_and_nonfinite() -> None:
    import pandas as pd

    frame = pd.DataFrame({"a": [1.0], "b": [np.inf], "c": [1e31]})
    result = MODULE._safe_numeric(frame, ["a", "b", "c"])
    assert result.loc[0, "a"] == 1.0
    assert np.isnan(result.loc[0, "b"])
    assert np.isnan(result.loc[0, "c"])


def test_tanimoto_does_not_overflow_uint8() -> None:
    bits = np.zeros((1, MODULE.MORGAN_BITS), dtype=np.uint8)
    bits[0, :300] = 1
    packed = np.packbits(bits, axis=1)
    similarities = MODULE._tanimoto_similarities(packed, np.asarray([300], dtype=np.int32), bits[0])
    assert similarities.tolist() == [1.0]


def test_app_has_no_fixed_dose_prediction_panel(tmp_path: Path) -> None:
    path = tmp_path / "app.html"
    MODULE._write_app(path)
    text = path.read_text()
    assert "Primary hERG Liability" in text
    assert "Direct Functional Curve Module" in text
    assert "Evidence And Applicability" in text
    assert "46 µM score" in text
    assert "probability_screen_active_at_46um" not in text
