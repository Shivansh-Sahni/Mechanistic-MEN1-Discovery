from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_local_herg_receptor_heterogeneity_v13_12 as module  # noqa: E402


def _frame(router: list[int], hybrid: list[int]) -> pd.DataFrame:
    y = np.tile(np.arange(3), len(router) // 3)
    return pd.DataFrame(
        {
            "external_structure_id": [f"X{index}" for index in range(len(router))],
            "scaffold_group_id": [f"S{index}" for index in range(len(router))],
            "true_class": y,
            "router_prediction": router,
            "hybrid_prediction": hybrid,
        }
    )


def test_bh_adjust_is_monotone_in_p_value_order() -> None:
    raw = [0.04, 0.001, 0.02, 0.8]
    adjusted = module._bh_adjust(raw)  # noqa: SLF001
    ordered = [adjusted[index] for index in np.argsort(raw)]
    assert ordered == sorted(ordered)
    assert all(value >= raw[index] for index, value in enumerate(adjusted))


def test_campaign_summary_counts_corrected_and_degraded_decisions() -> None:
    frame = _frame(
        router=[1, 1, 2, 0, 2, 2],
        hybrid=[0, 2, 2, 0, 1, 0],
    )
    result = module._campaign_summary(frame)  # noqa: SLF001
    assert result["n"] == 6
    assert result["decision_changes"]["changed_count"] == 4
    assert result["decision_changes"]["corrected_count"] == 2
    assert result["decision_changes"]["degraded_count"] == 2
    assert result["decision_changes"]["net_correct_count"] == 0
    assert result["decision_changes"]["wrong_to_different_wrong_count"] == 0
    assert sum(row["n"] for row in result["decision_changes"]["transition_outcomes"]) == 6


def test_decision_change_yield_builds_correct_two_by_two_table() -> None:
    first = {
        "decision_changes": {
            "corrected_count": 9,
            "changed_count": 14,
            "corrected_among_changed_rate": 9 / 14,
        }
    }
    second = {
        "decision_changes": {
            "corrected_count": 2,
            "changed_count": 28,
            "corrected_among_changed_rate": 2 / 28,
        }
    }
    result = module._decision_change_yield(first, second)  # noqa: SLF001
    assert result["table"] == [[9, 5], [2, 26]]
    assert result["fisher_exact_odds_ratio"] > 1


def test_interaction_bootstrap_detects_large_directional_difference() -> None:
    y = np.tile(np.arange(3), 20)
    first = pd.DataFrame(
        {
            "true_class": y,
            "router_prediction": np.roll(y, 1),
            "hybrid_prediction": y,
            "scaffold_group_id": [f"A{index}" for index in range(len(y))],
        }
    )
    second = pd.DataFrame(
        {
            "true_class": y,
            "router_prediction": y,
            "hybrid_prediction": np.roll(y, 1),
            "scaffold_group_id": [f"B{index}" for index in range(len(y))],
        }
    )
    result = module._interaction_bootstrap(  # noqa: SLF001
        first, second, replicates=200, mode="within_class"
    )
    assert result["balanced_accuracy"]["observed_interaction"] > 1.5
    assert result["balanced_accuracy"]["ci95"][0] > 1.0


def test_default_sealed_campaigns_load_without_duplicate_outcomes() -> None:
    first, _ = module._load_campaign(  # noqa: SLF001
        Path("research/local_runs/herg_external_validation_v13_7"), "first"
    )
    second, _ = module._load_campaign(  # noqa: SLF001
        Path("research/local_runs/herg_temporal_confirmation_v13_8"), "second"
    )
    assert len(first) == 110
    assert len(second) == 214
    assert first.external_structure_id.is_unique
    assert second.external_structure_id.is_unique
    assert np.argmax(first[list(module.ROUTER_PROBABILITY_COLUMNS)], axis=1).tolist() == (
        first.router_prediction.tolist()
    )


def test_end_to_end_report_is_self_hashed_and_binds_sources(tmp_path: Path) -> None:
    result = module.analyze(
        Path("research/local_runs/herg_external_validation_v13_7"),
        Path("research/local_runs/herg_temporal_confirmation_v13_8"),
        tmp_path,
        bootstrap=100,
    )
    report = json.loads((tmp_path / "analysis_report.json").read_text())
    digest = report.pop("report_sha256")
    assert hashlib.sha256(module._canonical(report)).hexdigest() == digest  # noqa: SLF001
    assert result["report"]["campaigns"]["V13.7_2025"]["n"] == 110
    assert result["report"]["campaigns"]["V13.8_2026"]["n"] == 214
    assert result["manifest"]["source_report_sha256"].keys() == {
        "V13.7_2025",
        "V13.8_2026",
    }
