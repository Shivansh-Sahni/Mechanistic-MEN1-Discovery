from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts/run_local_herg_v10_3_decision_platform.py"
SPEC = importlib.util.spec_from_file_location("herg_v103", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_cross_assay_consistency_thresholds() -> None:
    assert MODULE._cross_assay_consistency(2.0, 5.0)["label"] == "Cross-assay agreement"
    assert MODULE._cross_assay_consistency(2.0, 8.0)["label"] == "Assay-sensitive estimate"
    assert MODULE._cross_assay_consistency(2.0, 30.0)["label"] == "Large cross-assay divergence"


def test_decision_confidence_discloses_training_overlap() -> None:
    value = MODULE._decision_confidence(
        maximum_similarity=1.0,
        exact_overlap=True,
        interval_lower_pic50=5.0,
        interval_upper_pic50=5.5,
    )
    assert value["label"] == "Training overlap"
    assert "not independent evidence" in value["explanation"]


def test_decision_confidence_requires_domain_and_threshold_stability() -> None:
    low_domain = MODULE._decision_confidence(
        maximum_similarity=0.4,
        exact_overlap=False,
        interval_lower_pic50=5.0,
        interval_upper_pic50=5.5,
    )
    unstable = MODULE._decision_confidence(
        maximum_similarity=0.8,
        exact_overlap=False,
        interval_lower_pic50=4.4,
        interval_upper_pic50=6.2,
    )
    stable = MODULE._decision_confidence(
        maximum_similarity=0.8,
        exact_overlap=False,
        interval_lower_pic50=5.0,
        interval_upper_pic50=5.5,
    )
    assert low_domain["label"] == "Low"
    assert unstable["label"] == "Low"
    assert stable["label"] == "High"


def test_calibration_audit_rejects_unhelpful_calibration() -> None:
    # Each held-out fold follows the same already-calibrated relationship.
    frame = pd.DataFrame(
        {
            "observed_pic50": [4.0, 4.8, 5.6, 6.4, 7.2] * 5,
            "pred__honest_stack": [4.0, 4.8, 5.6, 6.4, 7.2] * 5,
            "outer_fold": [0, 1, 2, 3, 4] * 5,
        }
    )
    result = MODULE._calibration_audit(frame)
    assert result["raw_mae"] == 0.0
    assert not result["selected_for_deployment"]


def test_app_emphasizes_reliability_and_removes_fixed_dose_score(tmp_path: Path) -> None:
    target = tmp_path / "app.html"
    MODULE._write_app(target)
    text = target.read_text()
    assert "Decision Reliability" in text
    assert "Exact training overlap" in text
    assert "Separate Direct Functional Curve" in text
    assert "cross-assay" in text.lower()
    assert "46 µM" not in text
    assert "liability score" not in text.lower()
