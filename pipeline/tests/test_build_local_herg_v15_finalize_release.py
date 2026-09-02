from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/build_local_herg_v15_finalize_release.py"
RELEASE = Path(__file__).parents[2] / "research/local_runs/herg_v15_finalize/release"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("herg_v15_release", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module() -> ModuleType:
    return _module()


def test_absolute_evaluation_binds_frozen_v9_decision_without_copying_data(
    module: ModuleType,
) -> None:
    payload = module._absolute_evaluation_payload()  # noqa: SLF001
    assert payload["structure_count"] == 18_801
    assert payload["fixed_split_rows"] == 94_005
    assert payload["retained_v9_metrics"]["mae"] == pytest.approx(0.432760753275872)
    assert payload["rejected_v11_metrics"]["mae"] == pytest.approx(0.4395394378585102)
    bootstrap = payload["paired_scaffold_bootstrap"]
    assert bootstrap["point_estimate"] == pytest.approx(0.006778684582638204)
    assert bootstrap["ci95_lower"] == pytest.approx(0.004865598762111426)
    assert bootstrap["ci95_upper"] == pytest.approx(0.008706514716992382)
    assert payload["decision"] == {
        "retained_default": "V9 anchor",
        "v11_promoted": False,
        "reason": (
            "V11 MAE is higher; V11-minus-V9 scaffold-bootstrap MAE delta is positive "
            "with its 95% interval excluding zero"
        ),
    }
    assert payload["data_handling"]["data_files_copied_into_release"] is False
    assert {item["role"] for item in payload["artifacts"]} == {
        "v11_nested_oof_predictions",
        "fixed_nested_scaffold_splits",
        "v9_vs_v11_scaffold_bootstrap",
        "analysis_validation",
        "campaign_validation",
    }


def test_locked_pair_contract_reports_v9_and_prespecified_v11_separately(
    module: ModuleType,
) -> None:
    bindings, contract = module._paired_evidence()  # noqa: SLF001
    assert "baseline_scored_pairs_without_structures" in {row["role"] for row in bindings}
    v9 = contract["deployed_v9_locked_test"]
    v11 = contract["prespecified_v11_locked_test"]
    assert v9["n_pairs"] == v11["n_pairs"] == 517
    assert v9["directional_accuracy"] == pytest.approx(0.6533864541832669)
    assert v9["directional_accuracy_ci95"] == pytest.approx(
        [0.5115897345548509, 0.7718828587278106]
    )
    assert v9["delta_mae_pic50"] == pytest.approx(0.26668139865749113)
    assert v9["magnitude_capture_ratio"] == pytest.approx(0.44290614324398536)
    assert v9["predicted_negligible_rate_on_measured_non_ties"] == pytest.approx(
        0.3904382470119522
    )
    assert v11["directional_accuracy"] == pytest.approx(0.6812749003984063)
    assert v11["delta_mae_pic50"] == pytest.approx(0.2614825794166134)
    assert v11["magnitude_capture_ratio"] == pytest.approx(0.41337356012667575)
    assert contract["pairwise_challenger_promoted"] is False
    assert "does not override" in contract["primary_designation_scope"]


def test_requested_mw_strata_are_post_lock_and_high_bands_are_sparse(
    module: ModuleType,
) -> None:
    frame = module._requested_mw_metrics_frame()  # noqa: SLF001
    assert len(frame) == 18
    assert set(frame.model_id) == {"v11_nested", "v9_anchor", "xgb_depth10"}
    assert set(frame.benchmark_split) == {"development", "locked_test"}
    assert set(frame.requested_mean_mw_stratum) == {"<650", "650-750", ">750"}
    per_model = frame.loc[frame.model_id.eq("v9_anchor")].set_index(
        ["benchmark_split", "requested_mean_mw_stratum"]
    )
    assert per_model.loc[("development", "<650"), "n_pairs"] == 2060
    assert per_model.loc[("development", "650-750"), "n_pairs"] == 4
    assert per_model.loc[("development", ">750"), "n_pairs"] == 0
    assert per_model.loc[("locked_test", "<650"), "n_pairs"] == 515
    assert per_model.loc[("locked_test", "650-750"), "n_pairs"] == 1
    assert per_model.loc[("locked_test", ">750"), "n_pairs"] == 1
    sparse = frame.loc[frame.requested_mean_mw_stratum.ne("<650")]
    assert set(sparse.support_flag) == {"sparse_descriptive_only"}
    assert set(frame.analysis_status) == {
        "supplemental_user_requested_post_lock_reporting_only"
    }


def test_public_examples_are_exact_train_overlaps_and_errors_are_consistent(
    module: ModuleType,
) -> None:
    audit = module._validate_public_examples(RELEASE / "public_example_evaluation.csv")  # noqa: SLF001
    assert audit["rows"] == 2
    rows = (RELEASE / "public_example_evaluation.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    assert "selected demonstrations; not an unbiased benchmark" == audit["selection_scope"]
    assert "60196404" in rows[2]
    assert "counterion removed" in rows[2]
    assert all("private" not in row.lower() for row in rows)


def test_functional_group_and_historical_integrity_boundaries(module: ModuleType) -> None:
    bindings, contract = module._functional_group_evidence()  # noqa: SLF001
    assert "nested_loco_predictions" in {row["role"] for row in bindings}
    assert contract["promotion_supported"] is False
    assert contract["primary_vs_frozen"]["observed"] == pytest.approx(
        0.033040426784024426
    )
    historical = module._historical_v14_3_audit()  # noqa: SLF001
    assert historical["status"] == "detected_historical_drift_not_resealed"
    assert historical["drift_count"] == 9
    assert historical["historical_manifest_rewritten"] is False
    assert historical["historical_manifest_resealed"] is False


def test_release_build_is_deterministic_and_verifies(module: ModuleType) -> None:
    first = module.build(RELEASE)
    first_bytes = (RELEASE / "release_manifest.json").read_bytes()
    second = module.build(RELEASE)
    second_bytes = (RELEASE / "release_manifest.json").read_bytes()
    assert first == second
    assert first_bytes == second_bytes
    verified = module.verify(RELEASE)
    assert verified["status"] == "verified"
    assert verified["release_sha256"] == first["release_sha256"]
    manifest = json.loads(first_bytes)
    assert manifest["claim_boundary"]["public_release_or_deployment"] is False
    assert manifest["claim_boundary"]["prospective_or_independent_validation"] is False
    artifact_ids = [row["artifact_id"] for row in manifest["artifacts"]]
    assert len(artifact_ids) == len(set(artifact_ids))
    assert all(not Path(row["path"]).is_absolute() for row in manifest["artifacts"])
    assert "paired:baseline_scored_pairs_without_structures" in artifact_ids
    assert "functional_group:nested_loco_predictions" in artifact_ids
    assert "supplemental_mw:requested_mw_strata_metrics_json" in artifact_ids
    uncertainty = manifest["evidence_contracts"]["research_diagnostic_nested_manifests"]
    assert "v13_11_external_uncertainty" in {row["evidence_id"] for row in uncertainty}
    assert int(first["release_sha256"], 16) > 0
