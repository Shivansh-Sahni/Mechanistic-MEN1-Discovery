#!/usr/bin/env python3
"""Build and verify the deterministic local hERG V15 finalization release.

The builder never rewrites scientific inputs, model artifacts, the website, or
the historical V14.3 manifest. It verifies and hash-binds the already-frozen
evidence, records historical drift explicitly, and writes one canonical release
manifest whose bytes are deterministic for an unchanged repository state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import build_herg_v15_paired_benchmark as paired_benchmark
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCHEMA_VERSION = "platform-local-herg-v15-finalize-release/1.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE = REPO_ROOT / "research/local_runs/herg_v15_finalize/release"
PAIRED_ROOT = REPO_ROOT / "research/local_runs/herg_v15_finalize/paired"
FG_ROOT = REPO_ROOT / "research/local_runs/herg_functional_group_finalize_v15"
HISTORICAL_V14_RELEASE = REPO_ROOT / "research/local_runs/herg_v14_release"
V11_ROOT = REPO_ROOT / "research/local_runs/herg_comprehensive_optimization_v11_1"

RELEASE_DOCUMENTS: tuple[tuple[str, str], ...] = (
    ("README.md", "release_index"),
    ("MODEL_CARD.md", "composite_model_card"),
    ("METHODS.md", "release_methods"),
    ("SCIENTIFIC_LIMITATIONS.md", "scientific_limitations"),
    ("DEMO.md", "local_demo_guide"),
    ("public_example_evaluation.csv", "selected_public_demo_evaluation"),
    ("release_decision_table.csv", "release_decision_table"),
    ("CALIBRATION_UNCERTAINTY_APPLICABILITY_REPORT.md", "calibration_report"),
    ("HIGH_MW_AND_EDIT_MAGNITUDE_REPORT.md", "high_mw_edit_report"),
    ("RECEPTOR_STATE_INCREMENTAL_VALUE_REPORT.md", "receptor_state_report"),
)


class ReleaseError(RuntimeError):
    """Raised when release evidence or integrity policy fails."""


@dataclass(frozen=True)
class NestedManifest:
    evidence_id: str
    path: Path
    hash_field: str
    collections: tuple[str, ...]
    artifact_collections: tuple[str, ...]
    status: str


FROZEN_DEFAULT_MANIFESTS = (
    NestedManifest(
        "v9_quantitative_ic50",
        REPO_ROOT / "research/local_runs/herg_domain_mixture_campaign_v9/manifest.json",
        "manifest_sha256",
        ("artifacts", "unit_documents"),
        ("artifacts", "unit_documents"),
        "passed",
    ),
    NestedManifest(
        "v10_1_classification_and_direct_endpoints",
        REPO_ROOT / "research/local_runs/herg_v10_1_expanded_platform/manifest.json",
        "manifest_sha256",
        ("inputs", "artifacts"),
        ("artifacts",),
        "passed",
    ),
    NestedManifest(
        "v10_2_coherent_inference_bundle",
        REPO_ROOT / "research/local_runs/herg_v10_2_coherent_platform/manifest.json",
        "manifest_sha256",
        ("input_bindings", "artifact_bindings"),
        ("artifact_bindings",),
        "passed",
    ),
    NestedManifest(
        "v10_3_decision_reliability",
        REPO_ROOT / "research/local_runs/herg_v10_3_decision_platform/manifest.json",
        "manifest_sha256",
        ("input_bindings", "artifact_bindings"),
        ("artifact_bindings",),
        "passed",
    ),
    NestedManifest(
        "v12_3_endpoint_order_projection",
        REPO_ROOT / "research/local_runs/herg_endpoint_order_projection_v12_3/manifest.json",
        "manifest_sha256",
        ("inputs", "artifacts"),
        ("artifacts",),
        "complete",
    ),
)

RESEARCH_DIAGNOSTIC_MANIFESTS = (
    NestedManifest(
        "v13_9_receptor_inference_bundle",
        REPO_ROOT / "research/local_runs/herg_receptor_inference_bundle_v13_9/manifest.json",
        "manifest_sha256",
        ("inputs", "artifacts"),
        ("artifacts",),
        "complete_verified",
    ),
    NestedManifest(
        "v14_cross_campaign_receptor_fusion",
        REPO_ROOT / "research/local_runs/herg_cross_campaign_receptor_fusion_v14/manifest.json",
        "manifest_sha256",
        ("artifacts",),
        ("artifacts",),
        "complete",
    ),
    NestedManifest(
        "v14_1_nonlinear_receptor_stress",
        REPO_ROOT / "research/local_runs/herg_nonlinear_receptor_stress_v14_1/manifest.json",
        "manifest_sha256",
        ("inputs", "artifacts"),
        ("artifacts",),
        "complete",
    ),
    NestedManifest(
        "v14_2_endpoint_receptor_stress",
        REPO_ROOT / "research/local_runs/herg_nonlinear_endpoint_receptor_v14_2/manifest.json",
        "manifest_sha256",
        ("inputs", "artifacts"),
        ("artifacts",),
        "complete",
    ),
    NestedManifest(
        "v13_11_external_uncertainty",
        REPO_ROOT / "research/local_runs/herg_external_uncertainty_v13_11/manifest.json",
        "manifest_sha256",
        ("inputs", "artifacts"),
        ("artifacts",),
        "complete",
    ),
)


def _canonical(value: Any, *, newline: bool = True) -> bytes:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (payload + ("\n" if newline else "")).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, repo_root: Path = REPO_ROOT) -> str:
    path = path.resolve()
    repo_root = repo_root.resolve()
    if not path.is_relative_to(repo_root):
        raise ReleaseError(f"release path escapes repository: {path}")
    return str(path.relative_to(repo_root))


def _binding(
    path: Path,
    role: str,
    group: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ReleaseError(f"missing {group}/{role}: {path}")
    result: dict[str, Any] = {
        "artifact_id": f"{group}:{role}",
        "group": group,
        "role": role,
        "path": _relative(path, repo_root),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        result["rows"] = pq.read_metadata(path).num_rows
        result["arrow_schema_sha256"] = hashlib.sha256(
            pq.read_schema(path).serialize().to_pybytes()
        ).hexdigest()
    return result


def _verify_binding(binding: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> None:
    path = (repo_root / str(binding["path"])).resolve()
    if not path.is_relative_to(repo_root.resolve()):
        raise ReleaseError(f"bound path escapes repository: {path}")
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
        raise ReleaseError(f"release artifact missing or changed size: {path}")
    if _sha256(path) != binding["sha256"]:
        raise ReleaseError(f"release artifact hash changed: {path}")
    if path.suffix == ".parquet":
        if pq.read_metadata(path).num_rows != int(binding["rows"]):
            raise ReleaseError(f"Parquet row count changed: {path}")
        schema_hash = hashlib.sha256(pq.read_schema(path).serialize().to_pybytes()).hexdigest()
        if schema_hash != binding["arrow_schema_sha256"]:
            raise ReleaseError(f"Parquet schema changed: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReleaseError(f"expected JSON object: {path}")
    return value


def _read_self_hashed(path: Path, hash_field: str) -> tuple[dict[str, Any], str]:
    value = _read_json(path)
    expected = value.get(hash_field)
    unsigned = dict(value)
    unsigned.pop(hash_field, None)
    modes = {
        "canonical_json": hashlib.sha256(_canonical(unsigned, newline=False)).hexdigest(),
        "canonical_json_newline": hashlib.sha256(_canonical(unsigned, newline=True)).hexdigest(),
    }
    matches = [mode for mode, digest in modes.items() if digest == expected]
    if len(matches) != 1:
        raise ReleaseError(f"self-hash mismatch: {path}")
    return value, matches[0]


def _write_self_hashed(
    path: Path,
    payload: dict[str, Any],
    hash_field: str,
) -> dict[str, Any]:
    body = dict(payload)
    body.pop(hash_field, None)
    body[hash_field] = hashlib.sha256(_canonical(body)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical(body))
    os.replace(temporary, path)
    return body


def _write_release_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return _write_self_hashed(path, payload, "release_sha256")


def _resolve_nested_path(record: dict[str, Any], manifest_path: Path) -> Path:
    if "relative_path" in record:
        return (manifest_path.parent / str(record["relative_path"])).resolve()
    raw = Path(str(record["path"]))
    if raw.is_absolute():
        return raw.resolve()
    repository_candidate = (REPO_ROOT / raw).resolve()
    if repository_candidate.exists():
        return repository_candidate
    return (manifest_path.parent / raw).resolve()


def _audit_expected_binding(record: dict[str, Any], path: Path) -> dict[str, Any] | None:
    actual_bytes = path.stat().st_size if path.is_file() else None
    actual_sha = _sha256(path) if path.is_file() else None
    size_matches = (
        actual_bytes == int(record["bytes"])
        if actual_bytes is not None and "bytes" in record
        else actual_bytes is not None
    )
    hash_matches = actual_sha == record["sha256"]
    if size_matches and hash_matches:
        if path.suffix == ".parquet" and "rows" in record:
            if pq.read_metadata(path).num_rows != int(record["rows"]):
                hash_matches = False
        if path.suffix == ".parquet" and "arrow_schema_sha256" in record:
            schema_hash = hashlib.sha256(pq.read_schema(path).serialize().to_pybytes()).hexdigest()
            if schema_hash != record["arrow_schema_sha256"]:
                hash_matches = False
    if size_matches and hash_matches:
        return None
    return {
        "role": record.get("role"),
        "path": _relative(path) if path.is_relative_to(REPO_ROOT) else str(path),
        "expected_bytes": int(record["bytes"]) if "bytes" in record else None,
        "actual_bytes": actual_bytes,
        "expected_sha256": record["sha256"],
        "actual_sha256": actual_sha,
    }


def _audit_nested_manifest(spec: NestedManifest) -> dict[str, Any]:
    value, hash_mode = _read_self_hashed(spec.path, spec.hash_field)
    if value.get("status") != spec.status:
        raise ReleaseError(
            f"{spec.evidence_id} status changed: {value.get('status')!r} != {spec.status!r}"
        )
    drifts = []
    counts: dict[str, int] = {}
    for collection in spec.collections:
        records = value.get(collection)
        if not isinstance(records, list):
            raise ReleaseError(f"{spec.evidence_id} lacks binding collection {collection}")
        counts[collection] = len(records)
        for index, record in enumerate(records):
            drift = _audit_expected_binding(record, _resolve_nested_path(record, spec.path))
            if drift is not None:
                drift["collection"] = collection
                drift["index"] = index
                drifts.append(drift)
    artifact_drifts = [row for row in drifts if row["collection"] in spec.artifact_collections]
    if artifact_drifts:
        raise ReleaseError(
            f"{spec.evidence_id} has {len(artifact_drifts)} drifted frozen artifact bindings"
        )
    return {
        "evidence_id": spec.evidence_id,
        "manifest": _binding(spec.path, f"{spec.evidence_id}_manifest", "evidence_manifest"),
        "manifest_self_hash_field": spec.hash_field,
        "manifest_self_hash": value[spec.hash_field],
        "manifest_self_hash_mode": hash_mode,
        "verified_binding_counts": counts,
        "artifact_drift_count": 0,
        "historical_input_drift": drifts,
        "historical_input_drift_count": len(drifts),
    }


def _paired_evidence() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation_path = PAIRED_ROOT / "validation.json"
    validation, hash_mode = _read_self_hashed(validation_path, "validation_sha256")
    if validation.get("status") != "passed":
        raise ReleaseError("paired benchmark validation is not passed")
    checks = validation.get("checks", {})
    required_checks = {
        "canonical_pair_duplicates": 0,
        "prediction_frozen_before_label_open": True,
        "repository_validation_or_test_labels_opened": False,
        "scored_model_pair_duplicates": 0,
    }
    if any(checks.get(key) != expected for key, expected in required_checks.items()):
        raise ReleaseError("paired benchmark leakage or lock checks changed")
    expected_roles = {
        "lock_contract",
        "locked_pair_manifest",
        "baseline_oof_predictions_before_score",
        "baseline_prediction_seal",
        "sealed_train_only_pair_labels",
        "train_only_label_seal",
        "baseline_scored_pairs_without_structures",
        "baseline_clustered_metrics",
        "baseline_clustered_metrics_csv",
        "baseline_evaluation",
        "human_readable_report",
    }
    records = validation.get("artifacts", [])
    if {record.get("role") for record in records} != expected_roles:
        raise ReleaseError("paired benchmark artifact census changed")
    bindings = [_binding(validation_path, "validation", "paired")]
    for record in records:
        path = Path(str(record["path"])).resolve()
        drift = _audit_expected_binding(record, path)
        if drift is not None:
            raise ReleaseError(f"paired benchmark artifact drift: {path}")
        bindings.append(_binding(path, str(record["role"]), "paired"))

    contract, _ = _read_self_hashed(PAIRED_ROOT / "locked/lock_contract.json", "contract_sha256")
    prediction_seal, _ = _read_self_hashed(
        PAIRED_ROOT / "locked/baseline_prediction_seal.json", "prediction_seal_sha256"
    )
    label_seal, _ = _read_self_hashed(
        PAIRED_ROOT / "sealed/train_only_label_seal.json", "label_seal_sha256"
    )
    evaluation, _ = _read_self_hashed(
        PAIRED_ROOT / "analysis/baseline_evaluation.json", "evaluation_sha256"
    )
    if validation["contract_sha256"] != contract["contract_sha256"]:
        raise ReleaseError("paired validation and lock contract disagree")
    if validation["evaluation_sha256"] != evaluation["evaluation_sha256"]:
        raise ReleaseError("paired validation and evaluation disagree")
    if label_seal["lock_contract_sha256"] != contract["contract_sha256"]:
        raise ReleaseError("paired label seal and lock contract disagree")
    if evaluation["prediction_seal_sha256"] != prediction_seal["prediction_seal_sha256"]:
        raise ReleaseError("paired evaluation and prediction seal disagree")
    metric_frame = pq.read_table(PAIRED_ROOT / "analysis/baseline_metrics.parquet").to_pandas()

    def locked_overall(model_id: str) -> dict[str, Any]:
        selected = metric_frame.loc[
            metric_frame.model_id.eq(model_id)
            & metric_frame.benchmark_split.eq("locked_test")
            & metric_frame.stratifier.eq("overall")
            & metric_frame.stratum.eq("all")
        ]
        if len(selected) != 1:
            raise ReleaseError(f"paired locked-test overall row changed for {model_id}")
        row = selected.iloc[0]
        return {
            "n_pairs": int(row.n_pairs),
            "n_leakage_groups": int(row.n_leakage_groups),
            "n_direction_evaluable": int(row.n_direction_evaluable),
            "directional_accuracy": float(row.directional_accuracy),
            "directional_accuracy_ci95": [
                float(row.directional_accuracy_ci95_lower),
                float(row.directional_accuracy_ci95_upper),
            ],
            "delta_mae_pic50": float(row.delta_mae_pic50),
            "delta_mae_pic50_ci95": [
                float(row.delta_mae_pic50_ci95_lower),
                float(row.delta_mae_pic50_ci95_upper),
            ],
            "magnitude_capture_ratio": float(row.magnitude_capture_ratio),
            "magnitude_capture_ratio_ci95": [
                float(row.magnitude_capture_ratio_ci95_lower),
                float(row.magnitude_capture_ratio_ci95_upper),
            ],
            "predicted_negligible_rate_on_measured_non_ties": float(
                row.predicted_negligible_rate_on_measured_non_ties
            ),
        }

    v9_locked = locked_overall("v9_anchor")
    v11_locked = locked_overall("v11_nested")
    if v9_locked["n_pairs"] != 517 or v11_locked["n_pairs"] != 517:
        raise ReleaseError("paired locked-test pair census changed")
    return bindings, {
        "validation_sha256": validation["validation_sha256"],
        "validation_self_hash_mode": hash_mode,
        "contract_sha256": contract["contract_sha256"],
        "prediction_seal_sha256": prediction_seal["prediction_seal_sha256"],
        "label_seal_sha256": label_seal["label_seal_sha256"],
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "deployed_v9_locked_test": v9_locked,
        "prespecified_v11_locked_test": v11_locked,
        "primary_designation_scope": (
            "V11 was prespecified as the paired-benchmark primary only; this does not override "
            "the absolute-evaluation decision retaining deployed V9"
        ),
        "pairwise_challenger_promoted": False,
        "interpretation_boundary": evaluation["interpretation_boundary"],
    }


def _functional_group_evidence() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = FG_ROOT / "manifest.json"
    manifest, hash_mode = _read_self_hashed(manifest_path, "report_sha256")
    if manifest.get("status") != "complete_retrospective_functional_group_study":
        raise ReleaseError("functional-group study status changed")
    for collection in ("inputs", "artifacts", "code_dependencies"):
        for record in manifest.get(collection, []):
            path = Path(str(record["path"])).resolve()
            if _audit_expected_binding(record, path) is not None:
                raise ReleaseError(f"functional-group {collection} binding drift: {path}")
    script = manifest.get("script")
    if not isinstance(script, dict) or _audit_expected_binding(script, Path(script["path"])):
        raise ReleaseError("functional-group runner source binding drift")
    report, _ = _read_self_hashed(FG_ROOT / "analysis_report.json", "report_sha256")
    decision = report["promotion_decision"]
    if decision["production_or_default_promotion_supported"] is not False:
        raise ReleaseError("functional-group production promotion policy changed")
    if decision["research_signal_gate_passed"] is not False:
        raise ReleaseError("functional-group retrospective research gate changed")
    if report["dataset"]["cross_campaign_exact_overlap"] != 0:
        raise ReleaseError("functional-group exact-overlap contract changed")
    if report["dataset"]["cross_campaign_scaffold_overlap"] != 0:
        raise ReleaseError("functional-group scaffold-overlap contract changed")
    bindings = [_binding(manifest_path, "manifest", "functional_group")]
    for record in manifest["artifacts"]:
        path = Path(str(record["path"])).resolve()
        relative_name = path.relative_to(FG_ROOT).as_posix()
        role = relative_name.replace("/", "__").replace(".", "_")
        if relative_name == "predictions/nested_loco_predictions.parquet":
            role = "nested_loco_predictions"
        bindings.append(_binding(path, role, "functional_group"))
    primary = decision["prespecified_primary_candidate"]
    return bindings, {
        "manifest_self_hash": manifest["report_sha256"],
        "manifest_self_hash_mode": hash_mode,
        "analysis_report_sha256": report["report_sha256"],
        "rows": report["dataset"]["rows"],
        "campaigns": report["dataset"]["campaigns"],
        "primary_candidate": primary,
        "primary_macro_campaign_mae": report["metrics"][primary]["regression"][
            "macro_campaign_mae"
        ],
        "frozen_comparator": decision["frozen_comparator"],
        "frozen_comparator_macro_campaign_mae": report["metrics"][
            decision["frozen_comparator"]
        ]["regression"]["macro_campaign_mae"],
        "primary_vs_frozen": report["bootstrap"]["primary_vs_frozen"],
        "promotion_supported": False,
    }


def _absolute_evaluation_payload() -> dict[str, Any]:
    paths = {
        "v11_nested_oof_predictions": V11_ROOT / "analysis/nested_oof_predictions.parquet",
        "fixed_nested_scaffold_splits": V11_ROOT
        / "prepared/fixed_nested_scaffold_splits.parquet",
        "v9_vs_v11_scaffold_bootstrap": V11_ROOT
        / "analysis/v11_vs_v9_scaffold_bootstrap.json",
        "analysis_validation": V11_ROOT / "analysis/validation.json",
        "campaign_validation": V11_ROOT / "validation.json",
    }
    analysis_validation, analysis_hash_mode = _read_self_hashed(
        paths["analysis_validation"], "validation_sha256"
    )
    campaign_validation, campaign_hash_mode = _read_self_hashed(
        paths["campaign_validation"], "validation_sha256"
    )
    comparison, comparison_hash_mode = _read_self_hashed(
        paths["v9_vs_v11_scaffold_bootstrap"], "comparison_sha256"
    )
    for validation in (analysis_validation, campaign_validation):
        if validation.get("status") != "passed":
            raise ReleaseError("V11.1 absolute validation status changed")
        if validation.get("exact_nested_oof_rows") != 18_801:
            raise ReleaseError("V11.1 absolute validation row census changed")
        if validation.get("repository_validation_labels_opened") is not False:
            raise ReleaseError("V11.1 repository validation labels were opened")
        if validation.get("repository_test_labels_opened") is not False:
            raise ReleaseError("V11.1 repository test labels were opened")
    if comparison.get("status") != "passed":
        raise ReleaseError("V9-versus-V11 comparison status changed")
    if pq.read_metadata(paths["v11_nested_oof_predictions"]).num_rows != 18_801:
        raise ReleaseError("V11.1 nested OOF row count changed")
    if pq.read_metadata(paths["fixed_nested_scaffold_splits"]).num_rows != 94_005:
        raise ReleaseError("V11.1 fixed split row count changed")
    metrics = analysis_validation["metrics"]
    if metrics["v9_anchor"]["n"] != 18_801 or metrics["v11_nested"]["n"] != 18_801:
        raise ReleaseError("V9/V11 comparable structure census changed")
    if metrics["v9_anchor"]["mae"] >= metrics["v11_nested"]["mae"]:
        raise ReleaseError("V9-retention evidence changed")
    bindings = [
        _binding(path, role, "absolute_evaluation_source")
        for role, path in paths.items()
    ]
    return {
        "schema_version": "platform-local-herg-v15-absolute-evaluation/1.0",
        "status": "complete_hash_bound_reference_without_data_duplication",
        "structure_count": 18_801,
        "outer_folds": 5,
        "fixed_split_rows": 94_005,
        "artifacts": sorted(bindings, key=lambda row: row["artifact_id"]),
        "validation_self_hashes": {
            "analysis_validation": analysis_validation["validation_sha256"],
            "analysis_validation_hash_mode": analysis_hash_mode,
            "campaign_validation": campaign_validation["validation_sha256"],
            "campaign_validation_hash_mode": campaign_hash_mode,
            "v9_vs_v11_comparison": comparison["comparison_sha256"],
            "v9_vs_v11_comparison_hash_mode": comparison_hash_mode,
        },
        "retained_v9_metrics": metrics["v9_anchor"],
        "rejected_v11_metrics": metrics["v11_nested"],
        "paired_scaffold_bootstrap": metrics["paired_scaffold_bootstrap"],
        "decision": {
            "retained_default": "V9 anchor",
            "v11_promoted": False,
            "reason": (
                "V11 MAE is higher; V11-minus-V9 scaffold-bootstrap MAE delta is positive "
                "with its 95% interval excluding zero"
            ),
        },
        "data_handling": {
            "data_files_copied_into_release": False,
            "existing_artifacts_referenced_by_repository_relative_path_and_sha256": True,
            "repository_validation_or_test_labels_opened_for_release": False,
        },
    }


def _write_absolute_evaluation_manifest(output: Path) -> dict[str, Any]:
    return _write_self_hashed(
        output / "absolute_evaluation_manifest.json",
        _absolute_evaluation_payload(),
        "absolute_evaluation_sha256",
    )


def _verify_absolute_evaluation_manifest(output: Path) -> dict[str, Any]:
    manifest, _ = _read_self_hashed(
        output / "absolute_evaluation_manifest.json", "absolute_evaluation_sha256"
    )
    expected = _absolute_evaluation_payload()
    unsigned = dict(manifest)
    unsigned.pop("absolute_evaluation_sha256", None)
    if unsigned != expected:
        raise ReleaseError("absolute evaluation manifest content drift")
    for binding in manifest["artifacts"]:
        _verify_binding(binding)
    return manifest


def _validate_public_examples(path: Path) -> dict[str, Any]:
    expected = {
        "A": {
            "pubchem_cid": "16760153",
            "standardized_canonical_smiles": (
                "C#CCC1=C(C)[C@H](OC(=O)[C@H]2[C@H](C=C(C)C)C2(C)C)CC1=O"
            ),
            "measured_pic50": 4.362980844,
            "measured_ic50_um": 43.353,
            "frozen_ligand_predicted_ic50_um": 37.7368405887,
        },
        "B": {
            "pubchem_cid": "60196404",
            "standardized_canonical_smiles": (
                "CN(C)C(=O)C1(N2CCCCC2)CCN(CCC2(c3ccc(Cl)c(Cl)c3)"
                "CN(C(=O)c3ccccc3)CCO2)CC1"
            ),
            "measured_pic50": 4.939559036,
            "measured_ic50_um": 11.493,
            "frozen_ligand_predicted_ic50_um": 12.343282366,
        },
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if {row.get("demo_id") for row in rows} != set(expected) or len(rows) != 2:
        raise ReleaseError("public example census changed")
    for row in rows:
        reference = expected[row["demo_id"]]
        for field in ("pubchem_cid", "standardized_canonical_smiles"):
            if row.get(field) != reference[field]:
                raise ReleaseError(f"public example {row['demo_id']} {field} changed")
        for field in ("measured_pic50", "measured_ic50_um", "frozen_ligand_predicted_ic50_um"):
            if not math.isclose(float(row[field]), float(reference[field]), abs_tol=1e-12):
                raise ReleaseError(f"public example {row['demo_id']} {field} changed")
        predicted_pic50 = 6.0 - math.log10(reference["frozen_ligand_predicted_ic50_um"])
        signed_error = predicted_pic50 - reference["measured_pic50"]
        if not math.isclose(float(row["frozen_ligand_predicted_pic50"]), predicted_pic50, abs_tol=1e-12):
            raise ReleaseError(f"public example {row['demo_id']} predicted pIC50 is inconsistent")
        if not math.isclose(
            float(row["signed_pic50_error_predicted_minus_measured"]),
            signed_error,
            abs_tol=1e-12,
        ):
            raise ReleaseError(f"public example {row['demo_id']} signed error is inconsistent")
        if not math.isclose(float(row["absolute_pic50_error"]), abs(signed_error), abs_tol=1e-12):
            raise ReleaseError(f"public example {row['demo_id']} absolute error is inconsistent")
        if row.get("source_partition") != "Train" or row.get("exact_train_overlap") != "true":
            raise ReleaseError("public examples must remain labeled exact Train overlaps")
        if row.get("evidence_role") != "selected_demo_not_unbiased_benchmark":
            raise ReleaseError("public examples must not be presented as an unbiased benchmark")
    return {
        "rows": 2,
        "source": "Zenodo 8359714 data_herg_dev.csv quantitative compilation",
        "reported_source": "PubChem",
        "selection_scope": "selected demonstrations; not an unbiased benchmark",
    }


def _requested_mw_metrics_frame() -> pd.DataFrame:
    scored_path = PAIRED_ROOT / "analysis/baseline_scored_pairs.parquet"
    scored = pd.read_parquet(scored_path)
    if any("smiles" in column.lower() or "structure" in column.lower() for column in scored):
        raise ReleaseError("supplemental MW source unexpectedly contains structures or SMILES")
    models = sorted(scored.model_id.unique())
    splits = sorted(scored.benchmark_split.unique())
    if models != ["v11_nested", "v9_anchor", "xgb_depth10"]:
        raise ReleaseError("supplemental MW model census changed")
    if splits != ["development", "locked_test"]:
        raise ReleaseError("supplemental MW split census changed")
    scored = scored.copy()
    scored["requested_mean_mw_stratum"] = np.select(
        [scored.mean_endpoint_mw.lt(650.0), scored.mean_endpoint_mw.le(750.0)],
        ["<650", "650-750"],
        default=">750",
    )
    strata = ("<650", "650-750", ">750")
    metric_names: list[str] | None = None
    rows = []
    for model_id in models:
        for split in splits:
            model_split = scored.loc[
                scored.model_id.eq(model_id) & scored.benchmark_split.eq(split)
            ]
            for stratum in strata:
                group = model_split.loc[model_split.requested_mean_mw_stratum.eq(stratum)]
                local_seed = 20260848 + int(
                    hashlib.sha256(
                        f"{model_id}|{split}|requested_mean_mw_stratum|{stratum}".encode()
                    ).hexdigest()[:8],
                    16,
                )
                if len(group):
                    summary = paired_benchmark.clustered_metrics(
                        group,
                        bootstrap_replicates=1_000,
                        seed=local_seed,
                    )
                    metric_names = list(summary["metrics"])
                else:
                    if metric_names is None:
                        raise ReleaseError("cannot initialize supplemental MW metric schema")
                    summary = {
                        "n_pairs": 0,
                        "n_leakage_groups": 0,
                        "n_direction_evaluable": 0,
                        "n_activity_cliffs": 0,
                        "metrics": dict.fromkeys(metric_names),
                        "ci95": {
                            metric: {"lower": None, "upper": None}
                            for metric in metric_names
                        },
                    }
                row: dict[str, Any] = {
                    "model_id": model_id,
                    "benchmark_split": split,
                    "requested_mean_mw_stratum": stratum,
                    "n_pairs": summary["n_pairs"],
                    "n_leakage_groups": summary["n_leakage_groups"],
                    "n_direction_evaluable": summary["n_direction_evaluable"],
                    "n_activity_cliffs": summary["n_activity_cliffs"],
                    "support_flag": (
                        "adequate_internal_support"
                        if summary["n_pairs"] >= 30 and summary["n_leakage_groups"] >= 20
                        else "sparse_descriptive_only"
                    ),
                    "analysis_status": "supplemental_user_requested_post_lock_reporting_only",
                    "ci_method": "95% percentile bootstrap; leakage-group clusters; 1000 replicates",
                    "bootstrap_seed": local_seed,
                }
                for metric, value in summary["metrics"].items():
                    row[metric] = value
                    row[f"{metric}_ci95_lower"] = summary["ci95"][metric]["lower"]
                    row[f"{metric}_ci95_upper"] = summary["ci95"][metric]["upper"]
                rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != 18:
        raise ReleaseError("supplemental requested-MW output must contain 18 rows")
    return frame


def _requested_mw_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, float_format="%.15g", na_rep="", lineterminator="\n")
    return buffer.getvalue().encode()


def _requested_mw_payload(frame: pd.DataFrame, csv_path: Path) -> dict[str, Any]:
    contract, _ = _read_self_hashed(PAIRED_ROOT / "locked/lock_contract.json", "contract_sha256")
    validation, _ = _read_self_hashed(PAIRED_ROOT / "validation.json", "validation_sha256")
    return {
        "schema_version": "platform-local-herg-v15-requested-mw-strata/1.0",
        "status": "supplemental_user_requested_post_lock_reporting_only",
        "source_artifact": _binding(
            PAIRED_ROOT / "analysis/baseline_scored_pairs.parquet",
            "baseline_scored_pairs_without_structures",
            "supplemental_mw_source",
        ),
        "csv_artifact": _binding(
            csv_path,
            "requested_mw_strata_metrics_csv",
            "supplemental_mw",
        ),
        "lock_contract_sha256": contract["contract_sha256"],
        "paired_validation_sha256": validation["validation_sha256"],
        "stratifier": "mean_endpoint_mw",
        "requested_strata": [
            {"label": "<650", "definition": "mean endpoint MW < 650 Da"},
            {
                "label": "650-750",
                "definition": "650 Da <= mean endpoint MW <= 750 Da",
            },
            {"label": ">750", "definition": "mean endpoint MW > 750 Da"},
        ],
        "models": ["v11_nested", "v9_anchor", "xgb_depth10"],
        "splits": ["development", "locked_test"],
        "bootstrap": {
            "unit": "leakage_group_id",
            "replicates": 1_000,
            "confidence_level": 0.95,
            "seed_contract": (
                "locked base seed 20260848 plus deterministic SHA-256 offset for model/split/stratum"
            ),
        },
        "metrics": json.loads(frame.to_json(orient="records", double_precision=15)),
        "scientific_boundary": {
            "paired_lock_modified_or_resealed": False,
            "strata_prespecified_before_lock": False,
            "used_for_model_selection_or_tuning": False,
            "high_mw_model_or_correction_promoted": False,
            "interpretation": (
                "post-lock supplemental reporting requested by the user; sparse strata remain "
                "descriptive and cannot establish high-MW performance"
            ),
        },
    }


def _write_requested_mw_metrics(output: Path) -> dict[str, Any]:
    frame = _requested_mw_metrics_frame()
    csv_path = output / "requested_mw_strata_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_requested_mw_csv_bytes(frame))
    os.replace(temporary, csv_path)
    return _write_self_hashed(
        output / "requested_mw_strata_metrics.json",
        _requested_mw_payload(frame, csv_path),
        "requested_mw_strata_sha256",
    )


def _verify_requested_mw_metrics(output: Path) -> dict[str, Any]:
    frame = _requested_mw_metrics_frame()
    csv_path = output / "requested_mw_strata_metrics.csv"
    if not csv_path.is_file() or csv_path.read_bytes() != _requested_mw_csv_bytes(frame):
        raise ReleaseError("supplemental requested-MW CSV content drift")
    manifest, _ = _read_self_hashed(
        output / "requested_mw_strata_metrics.json", "requested_mw_strata_sha256"
    )
    expected = _requested_mw_payload(frame, csv_path)
    unsigned = dict(manifest)
    unsigned.pop("requested_mw_strata_sha256", None)
    if unsigned != expected:
        raise ReleaseError("supplemental requested-MW JSON content drift")
    _verify_binding(manifest["source_artifact"])
    _verify_binding(manifest["csv_artifact"])
    return manifest


def _historical_v14_3_audit() -> dict[str, Any]:
    manifest_path = HISTORICAL_V14_RELEASE / "release_manifest.json"
    value, hash_mode = _read_self_hashed(manifest_path, "release_sha256")
    drift = []
    for index, record in enumerate(value.get("artifacts", [])):
        path = _resolve_nested_path(record, manifest_path)
        changed = _audit_expected_binding(record, path)
        if changed is not None:
            changed["index"] = index
            drift.append(changed)
    return {
        "status": "detected_historical_drift_not_resealed" if drift else "historical_snapshot_intact",
        "manifest": _binding(manifest_path, "v14_3_release_manifest", "historical_reference"),
        "manifest_self_hash": value["release_sha256"],
        "manifest_self_hash_mode": hash_mode,
        "declared_artifacts": len(value.get("artifacts", [])),
        "unchanged_artifacts": len(value.get("artifacts", [])) - len(drift),
        "drift_count": len(drift),
        "drift": drift,
        "historical_manifest_rewritten": False,
        "historical_manifest_resealed": False,
    }


def _scientific_policy() -> dict[str, Any]:
    v14, _ = _read_self_hashed(
        REPO_ROOT / "research/local_runs/herg_cross_campaign_receptor_fusion_v14/analysis_report.json",
        "report_sha256",
    )
    v141, _ = _read_self_hashed(
        REPO_ROOT / "research/local_runs/herg_nonlinear_receptor_stress_v14_1/analysis_report.json",
        "report_sha256",
    )
    v142, _ = _read_self_hashed(
        REPO_ROOT / "research/local_runs/herg_nonlinear_endpoint_receptor_v14_2/analysis_report.json",
        "report_sha256",
    )
    v123, _ = _read_self_hashed(
        REPO_ROOT / "research/local_runs/herg_endpoint_order_projection_v12_3/analysis_report.json",
        "report_sha256",
    )
    if not v14["promotion_decision"]["ligand_recalibration_research_preview_supported"]:
        raise ReleaseError("V14 ligand preview policy changed")
    if v14["promotion_decision"]["general_receptor_prediction_promotion_supported"]:
        raise ReleaseError("V14 receptor policy changed")
    if v141["decision"]["general_receptor_prediction_promotion_supported"]:
        raise ReleaseError("V14.1 receptor policy changed")
    if v142["decision"]["website_integration_supported"]:
        raise ReleaseError("V14.2 endpoint receptor policy changed")
    if not v123["promotion_decision"]["promote_for_complete_paired_curves"]:
        raise ReleaseError("V12.3 deterministic order-projection policy changed")
    return {
        "production_default": "frozen ligand-only V9/V10.1/V10.2/V10.3 with V12.3 complete-triplet order projection",
        "production_default_changed": False,
        "v14_classification_preview_default": False,
        "v14_1_regression_preview_default": False,
        "receptor_prediction_promoted": False,
        "multi_receptor_docking_role": "diagnostic_only; no score averaging or state-specific ML",
        "functional_group_prediction_promoted": False,
        "matched_pair_challenger_promoted": False,
        "high_mw_correction_promoted": False,
        "next_decisive_evidence": (
            "a preregistered external/prospective analogue-series panel with frozen models, "
            "explicit series/campaign IDs, high-MW coverage, and assay outcomes unopened"
        ),
    }


def _release_payload(output: Path) -> dict[str, Any]:
    absolute_manifest = _verify_absolute_evaluation_manifest(output)
    requested_mw = _verify_requested_mw_metrics(output)
    public_examples = _validate_public_examples(output / "public_example_evaluation.csv")
    paired_bindings, paired_contract = _paired_evidence()
    fg_bindings, fg_contract = _functional_group_evidence()
    default_audits = [_audit_nested_manifest(spec) for spec in FROZEN_DEFAULT_MANIFESTS]
    diagnostic_audits = [_audit_nested_manifest(spec) for spec in RESEARCH_DIAGNOSTIC_MANIFESTS]
    historical_audit = _historical_v14_3_audit()
    if historical_audit["drift_count"] == 0:
        raise ReleaseError("expected stale V14.3 historical manifest drift was not detected")
    documents = [
        _binding(output / filename, role, "release_document")
        for filename, role in RELEASE_DOCUMENTS
    ]
    evidence_manifests = [
        audit["manifest"] for audit in [*default_audits, *diagnostic_audits]
    ]
    artifacts = [
        *documents,
        *paired_bindings,
        *fg_bindings,
        *evidence_manifests,
        _binding(
            output / "absolute_evaluation_manifest.json",
            "absolute_evaluation_manifest",
            "absolute_evaluation",
        ),
        _binding(
            output / "requested_mw_strata_metrics.csv",
            "requested_mw_strata_metrics_csv",
            "supplemental_mw",
        ),
        _binding(
            output / "requested_mw_strata_metrics.json",
            "requested_mw_strata_metrics_json",
            "supplemental_mw",
        ),
        historical_audit["manifest"],
        _binding(Path(__file__), "manifest_builder", "release_tooling"),
        _binding(
            REPO_ROOT / "pipeline/tests/test_build_local_herg_v15_finalize_release.py",
            "manifest_builder_tests",
            "release_tooling",
        ),
        _binding(
            REPO_ROOT / "pipeline/environments/requirements.lock",
            "python_environment_lock",
            "release_environment",
        ),
    ]
    artifacts = sorted(artifacts, key=lambda row: row["artifact_id"])
    artifact_ids = [row["artifact_id"] for row in artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ReleaseError("release artifact identifiers are duplicated")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_local_research_release",
        "release_label": "hERG V15 finalization",
        "determinism_contract": {
            "canonical_json": "UTF-8, sorted keys, compact separators, one trailing newline",
            "wall_clock_timestamp_embedded": False,
            "unchanged_inputs_produce_identical_manifest_bytes": True,
            "writes_outside_release_manifest": False,
        },
        "artifacts": artifacts,
        "evidence_contracts": {
            "absolute_evaluation": {
                "absolute_evaluation_sha256": absolute_manifest[
                    "absolute_evaluation_sha256"
                ],
                "structure_count": absolute_manifest["structure_count"],
                "retained_v9_metrics": absolute_manifest["retained_v9_metrics"],
                "paired_scaffold_bootstrap": absolute_manifest[
                    "paired_scaffold_bootstrap"
                ],
            },
            "selected_public_examples": public_examples,
            "requested_mw_strata": {
                "requested_mw_strata_sha256": requested_mw[
                    "requested_mw_strata_sha256"
                ],
                "status": requested_mw["status"],
                "stratifier": requested_mw["stratifier"],
                "requested_strata": requested_mw["requested_strata"],
                "models": requested_mw["models"],
                "splits": requested_mw["splits"],
                "scientific_boundary": requested_mw["scientific_boundary"],
            },
            "locked_paired": paired_contract,
            "functional_group": fg_contract,
            "frozen_default_nested_manifests": default_audits,
            "research_diagnostic_nested_manifests": diagnostic_audits,
        },
        "historical_v14_3_manifest_audit": historical_audit,
        "scientific_policy": _scientific_policy(),
        "claim_boundary": {
            "research_use_only": True,
            "public_release_or_deployment": False,
            "clinical_or_regulatory_use": False,
            "prospective_or_independent_validation": False,
            "classifier_probability_is_ic50": False,
            "vina_score_is_binding_free_energy": False,
            "functional_group_association_is_causal": False,
            "high_mw_correction_validated": False,
        },
    }


def build(output: Path = DEFAULT_RELEASE) -> dict[str, Any]:
    output = output.resolve()
    if not output.is_relative_to(REPO_ROOT):
        raise ReleaseError("release output must remain inside the repository")
    _write_absolute_evaluation_manifest(output)
    _write_requested_mw_metrics(output)
    return _write_release_manifest(output / "release_manifest.json", _release_payload(output))


def verify(output: Path = DEFAULT_RELEASE) -> dict[str, Any]:
    output = output.resolve()
    _verify_absolute_evaluation_manifest(output)
    _verify_requested_mw_metrics(output)
    manifest, _ = _read_self_hashed(output / "release_manifest.json", "release_sha256")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseError("release schema changed")
    for binding in manifest.get("artifacts", []):
        _verify_binding(binding)
    expected = _release_payload(output)
    for key in (
        "schema_version",
        "status",
        "release_label",
        "determinism_contract",
        "artifacts",
        "evidence_contracts",
        "historical_v14_3_manifest_audit",
        "scientific_policy",
        "claim_boundary",
    ):
        if manifest.get(key) != expected[key]:
            raise ReleaseError(f"release manifest content drift: {key}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "release_sha256": manifest["release_sha256"],
        "artifacts_verified": len(manifest["artifacts"]),
        "historical_v14_3_drift_count": manifest["historical_v14_3_manifest_audit"][
            "drift_count"
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build", "verify", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RELEASE)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result: dict[str, Any] = {}
        if args.stage in {"build", "all"}:
            result = build(args.output_root)
        if args.stage in {"verify", "all"}:
            result = verify(args.output_root)
    except ReleaseError as exc:
        print(f"V15 RELEASE ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
