from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_local_herg_menin_blinded_evaluation_v13_13 as module  # noqa: E402

PANEL = Path(
    "research/local_runs/herg_menin_receptor_prioritization_v13_10/analysis/"
    "blinded_assay_challenge_panel.parquet"
)
BLANK = Path(
    "research/local_runs/herg_menin_receptor_prioritization_v13_10/analysis/blinded_herg_outcome_release.csv"
)


def _completed(path: Path) -> Path:
    frame = pd.read_csv(BLANK)
    endpoint_rows = [
        (10.0, 30.0, 100.0),
        (100.0, 1000.0, 10_000.0),
        (10_000.0, 30_000.0, 100_000.0),
    ]
    for index in frame.index:
        values = endpoint_rows[index % 3]
        frame.loc[index, list(module.OUTCOME_COLUMNS)] = values
    frame["assay_temperature_c"] = 37.0
    frame["cell_system"] = "CHO-hERG"
    frame["voltage_protocol_id"] = "PROTO-1"
    frame["biological_replicate_count"] = 3
    frame["qc_pass"] = True
    frame["assay_notes"] = "synthetic unit-test outcome"
    frame.to_csv(path, index=False)
    return path


def test_protocol_locks_only_while_default_template_is_blank(tmp_path: Path) -> None:
    result = module._protocol(PANEL, BLANK, tmp_path)  # noqa: SLF001
    assert result["status"] == "preanalysis_locked_before_outcomes"
    assert result["panel"]["router_hybrid_disagreements"] == 13
    assert result["implementation"]["sha256"] == module._sha(Path(module.__file__))  # noqa: SLF001
    design = result["primary_receptor_hypothesis"]["design_operating_characteristics"]
    assert design["critical_corrected_count_if_all_are_informative"] == 10
    assert 0.5 < design["exact_power_by_true_corrected_probability"]["0.75"] < 0.7
    assert result["outcomes_and_thresholds"]["ternary_ic50"]["matches_frozen_platform_tiers"] is True
    assert (tmp_path / "REPORT.md").is_file()
    body = dict(result)
    digest = body.pop("protocol_sha256")
    assert hashlib.sha256(module._canonical(body)).hexdigest() == digest  # noqa: SLF001


def test_seal_refuses_blank_template(tmp_path: Path) -> None:
    module._protocol(PANEL, BLANK, tmp_path)  # noqa: SLF001
    with pytest.raises(module.EvaluationError, match="still-blank"):
        module._seal(PANEL, BLANK, tmp_path)  # noqa: SLF001


def test_tier_threshold_matches_frozen_30_micromolar_safe_boundary() -> None:
    values = module._tier(  # noqa: SLF001
        pd.Series([module.SAFE_PIC50 - 0.001, module.SAFE_PIC50, 6.0, 6.001]).to_numpy()
    )
    assert values.tolist() == [0, 1, 1, 2]


def test_seal_refuses_physical_endpoint_order_violation(tmp_path: Path) -> None:
    module._protocol(PANEL, BLANK, tmp_path)  # noqa: SLF001
    completed = _completed(tmp_path / "completed.csv")
    frame = pd.read_csv(completed)
    frame.loc[0, "measured_ic10_nm"] = frame.loc[0, "measured_ic50_nm"] * 2
    frame.to_csv(completed, index=False)
    with pytest.raises(module.EvaluationError, match="physical endpoint order"):
        module._seal(PANEL, completed, tmp_path)  # noqa: SLF001


def test_seal_refuses_changed_blinded_mapping(tmp_path: Path) -> None:
    module._protocol(PANEL, BLANK, tmp_path)  # noqa: SLF001
    completed = _completed(tmp_path / "completed.csv")
    frame = pd.read_csv(completed)
    frame.loc[0, "blinded_sample_id"] = "TAMPERED"
    frame.to_csv(completed, index=False)
    with pytest.raises(module.EvaluationError, match="mapping changed"):
        module._seal(PANEL, completed, tmp_path)  # noqa: SLF001


def test_synthetic_completed_workflow_scores_all_predeclared_endpoints(tmp_path: Path) -> None:
    module._protocol(PANEL, BLANK, tmp_path)  # noqa: SLF001
    completed = _completed(tmp_path / "completed.csv")
    seal = module._seal(PANEL, completed, tmp_path)  # noqa: SLF001
    result = module._score(PANEL, tmp_path)  # noqa: SLF001
    assert seal["rows"] == 24
    assert result["eligible_rows"] == 24
    assert result["primary_receptor_hypothesis"]["preselected_disagreement_rows_eligible"] == 13
    assert set(result["regression"]) == {"IC10", "IC30", "IC50"}
    assert set(result["regression"]["IC50"]["v9_mixed_intervals"]) == {"80", "90", "95"}
    assert result["decision"]["general_promotion_authorized"] is False
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "complete"
