#!/usr/bin/env python3
"""Audit exact extreme V13.7 labels without changing the primary external result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_local_herg_external_validation_v13_7 as v137  # noqa: E402
import run_local_herg_receptor_ensemble_campaign_v13 as v13  # noqa: E402

SCHEMA_VERSION = "platform-local-herg-external-boundary-sensitivity-v13.14/1.0"
EXPECTED_WORKBOOK_SHA256 = "28a0c203691eef1d4e9cfbc0da022573ad107e213514eb48de529eafd3444170"
SEED = 20260821


class AuditError(RuntimeError):
    """Raised when a source or prediction integrity invariant fails."""


def _canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any], field: str) -> dict[str, Any]:
    body = dict(payload)
    body.pop(field, None)
    body[field] = hashlib.sha256(_canonical(body)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(body))
    temporary.replace(path)
    return body


def _read_json(path: Path, field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    expected = payload.get(field)
    body = dict(payload)
    body.pop(field, None)
    if hashlib.sha256(_canonical(body)).hexdigest() != expected:
        raise AuditError(f"self-hash mismatch: {path}")
    return payload


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - y
    return {
        "n": len(y),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
        "median_absolute_error": float(np.median(np.abs(error))),
        "bias_prediction_minus_observed": float(np.mean(error)),
        "spearman": float(spearmanr(y, prediction).statistic) if len(y) >= 3 else None,
    }


def _paired_cluster_bootstrap(frame: pd.DataFrame, replicates: int) -> dict[str, Any]:
    if replicates < 100:
        raise AuditError("at least 100 bootstrap replicates are required")
    groups = [group.index.to_numpy(int) for _, group in frame.groupby("scaffold_group_id")]
    rng = np.random.default_rng(SEED)
    values = np.empty(replicates, dtype=float)
    y = frame.true_pic50_m.to_numpy(float)
    v9 = frame.v9_mixed_ic50_predicted_pic50.to_numpy(float)
    source = frame.source_model_prediction_pic50_m.to_numpy(float)
    for iteration in range(replicates):
        selected = rng.choice(len(groups), len(groups), replace=True)
        rows = np.concatenate([groups[index] for index in selected])
        values[iteration] = np.mean(np.abs(v9[rows] - y[rows]) - np.abs(source[rows] - y[rows]))
    observed = float(np.mean(np.abs(v9 - y) - np.abs(source - y)))
    return {
        "delta_mae_v9_minus_source": observed,
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "probability_v9_lower_mae": float(np.mean(values < 0)),
        "replicates": replicates,
        "unique_scaffolds": len(groups),
        "resampling": "paired scaffold-cluster bootstrap",
    }


def _publisher_trace(workbook_path: Path, registry: pd.DataFrame, boundary_ids: set[str]) -> pd.DataFrame:
    selected = registry.loc[
        registry.external_structure_id.isin(boundary_ids),
        ["external_structure_id", "original_smiles"],
    ].copy()
    by_smiles = dict(zip(selected.original_smiles, selected.external_structure_id, strict=True))
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    sheet = workbook["Table S8"]
    rows = []
    for excel_row, values in enumerate(sheet.iter_rows(min_row=3, values_only=True), start=3):
        smiles = values[0]
        if smiles not in by_smiles:
            continue
        rows.append(
            {
                "external_structure_id": by_smiles[smiles],
                "publisher_sheet": "Table S8",
                "publisher_excel_row": excel_row,
                "publisher_smiles": smiles,
                "publisher_y_true_pic50": float(values[1]),
                "publisher_c_true": int(values[2]),
                "publisher_present_model_pic50": float(values[3]),
                "publisher_present_model_class": int(values[4]),
                "publisher_previous_model_pic50": float(values[5]),
                "publisher_previous_model_class": int(values[6]),
            }
        )
    workbook.close()
    trace = pd.DataFrame(rows)
    if len(trace) != len(boundary_ids) or set(trace.external_structure_id) != boundary_ids:
        raise AuditError("could not trace every extreme label to publisher Table S8")
    return trace.sort_values("external_structure_id").reset_index(drop=True)


def _load_frame(campaign: Path) -> pd.DataFrame:
    report = _read_json(campaign / "analysis_report.json", "report_sha256")
    if report["status"] != "complete_with_receptor_external_validation":
        raise AuditError("V13.7 campaign is not complete")
    seal = _read_json(campaign / "predictions/receptor_prediction_seal.json", "seal_sha256")
    receptor_path = campaign / "predictions/receptor_predictions_before_score.parquet"
    if _sha(receptor_path) != seal["prediction_sha256"]:
        raise AuditError("V13.7 receptor prediction seal mismatch")
    receptor_ids = pd.read_parquet(receptor_path)[
        [
            "external_structure_id",
            "hybrid_prediction",
            "lgbm_rdkit2d_morgan__probability_safe",
            "lgbm_rdkit2d_morgan__probability_moderate",
            "lgbm_rdkit2d_morgan__probability_potent",
        ]
    ]
    labels = pd.read_parquet(campaign / "sealed/labels.parquet")
    labels = labels.loc[labels.set_name.eq("ev2_exact_quantitative")]
    labels = labels.groupby("external_structure_id", as_index=False).agg(
        true_pic50_m=("true_pic50_m", "median"),
        source_model_prediction_pic50_m=("source_model_prediction_pic50_m", "mean"),
        sealed_source_row=("source_row", "first"),
    )
    baseline = pd.read_parquet(campaign / "predictions/baseline_predictions_before_score.parquet")
    registry = pd.read_parquet(campaign / "prepared/structure_registry.parquet")
    return (
        receptor_ids.merge(labels, on="external_structure_id", validate="one_to_one")
        .merge(
            baseline[["external_structure_id", "v9_mixed_ic50_predicted_pic50"]],
            on="external_structure_id",
            validate="one_to_one",
        )
        .merge(
            registry[["external_structure_id", "scaffold_group_id", "original_smiles"]],
            on="external_structure_id",
            validate="one_to_one",
        )
    )


def analyze(
    campaign: Path,
    uncertainty: Path,
    workbook_path: Path,
    output: Path,
    bootstrap: int,
) -> dict[str, Any]:
    if _sha(workbook_path) != EXPECTED_WORKBOOK_SHA256:
        raise AuditError("publisher workbook hash mismatch")
    frame = _load_frame(campaign)
    frame["exact_extreme_boundary_diagnostic"] = frame.true_pic50_m.le(3.0) | frame.true_pic50_m.ge(9.0)
    frame["v9_absolute_error"] = (frame.v9_mixed_ic50_predicted_pic50 - frame.true_pic50_m).abs()
    frame["source_absolute_error"] = (frame.source_model_prediction_pic50_m - frame.true_pic50_m).abs()
    frame["true_tier"] = v13._tier_index(frame.true_pic50_m.to_numpy(float))  # noqa: SLF001
    frame["router_prediction"] = np.argmax(
        frame[
            [
                "lgbm_rdkit2d_morgan__probability_safe",
                "lgbm_rdkit2d_morgan__probability_moderate",
                "lgbm_rdkit2d_morgan__probability_potent",
            ]
        ].to_numpy(float),
        axis=1,
    )
    boundary = frame.exact_extreme_boundary_diagnostic
    boundary_ids = set(frame.loc[boundary, "external_structure_id"])
    trace = _publisher_trace(
        workbook_path,
        pd.read_parquet(campaign / "prepared/structure_registry.parquet"),
        boundary_ids,
    )
    trace = trace.merge(
        frame[
            [
                "external_structure_id",
                "sealed_source_row",
                "true_pic50_m",
                "v9_mixed_ic50_predicted_pic50",
                "source_model_prediction_pic50_m",
            ]
        ],
        on="external_structure_id",
        validate="one_to_one",
    )
    if not np.allclose(trace.publisher_y_true_pic50, trace.true_pic50_m):
        raise AuditError("sealed labels do not reproduce publisher Table S8")
    intervals = pd.read_parquet(uncertainty / "external_interval_predictions.parquet")
    intervals = intervals.loc[
        intervals.campaign.eq("v13.7_ev2_project_exact_and_v9_scaffold_novel")
        & intervals.method.eq("global")
        & intervals.nominal_coverage.eq(0.9),
        ["external_structure_id", "covered"],
    ]
    frame = frame.merge(intervals, on="external_structure_id", validate="one_to_one")
    strata = {
        "primary_all_rows": np.ones(len(frame), dtype=bool),
        "posthoc_non_extreme_sensitivity": ~boundary.to_numpy(bool),
        "exact_extreme_rows": boundary.to_numpy(bool),
    }
    metrics = {}
    for name, keep in strata.items():
        part = frame.loc[keep]
        metrics[name] = {
            "n": len(part),
            "unique_scaffolds": int(part.scaffold_group_id.nunique()),
            "v9_mixed_ic50": _metrics(
                part.true_pic50_m.to_numpy(float),
                part.v9_mixed_ic50_predicted_pic50.to_numpy(float),
            ),
            "source_2025_model": _metrics(
                part.true_pic50_m.to_numpy(float),
                part.source_model_prediction_pic50_m.to_numpy(float),
            ),
            "v9_minus_source_mae": float(part.v9_absolute_error.mean() - part.source_absolute_error.mean()),
            "global_interval90_coverage": float(part.covered.mean()),
        }
    metrics["posthoc_non_extreme_sensitivity"]["paired_scaffold_bootstrap"] = _paired_cluster_bootstrap(
        frame.loc[~boundary].reset_index(drop=True), bootstrap
    )
    classification = {}
    for name, keep in {
        "primary_all_rows": np.ones(len(frame), dtype=bool),
        "posthoc_non_extreme_sensitivity": ~boundary.to_numpy(bool),
    }.items():
        part = frame.loc[keep].reset_index(drop=True)
        y = part.true_tier.to_numpy(int)
        router = part.router_prediction.to_numpy(int)
        hybrid = part.hybrid_prediction.to_numpy(int)
        classification[name] = {
            "ligand_router": v137._classification_metrics(y, router, [0, 1, 2]),  # noqa: SLF001
            "frozen_hybrid": v137._classification_metrics(y, hybrid, [0, 1, 2]),  # noqa: SLF001
            "paired_within_class_bootstrap": v137._balanced_bootstrap_classification(  # noqa: SLF001
                y, router, hybrid, bootstrap
            ),
            "paired_scaffold_cluster_bootstrap": v137._scaffold_bootstrap_classification(  # noqa: SLF001
                part.scaffold_group_id.to_numpy(object), y, router, hybrid, bootstrap
            ),
        }
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "row_influence_audit.parquet"
    trace_path = output / "publisher_boundary_trace.csv"
    frame.sort_values("v9_absolute_error", ascending=False).to_parquet(
        audit_path, index=False, compression="zstd"
    )
    trace.to_csv(trace_path, index=False)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete_primary_unchanged_posthoc_sensitivity",
        "source_integrity": {
            "publisher_workbook_sha256": _sha(workbook_path),
            "publisher_sheet": "Table S8",
            "extreme_labels_traced_exactly": True,
            "parsing_error_found": False,
            "qualifier_or_censor_status_documented_in_table_s8": False,
        },
        "diagnostic_rule": {
            "rule": "exact pIC50 <= 3.0 or >= 9.0",
            "chosen_posthoc_after_visual_influence_review": True,
            "used_to_change_primary_metric_or_decision": False,
            "extreme_rows": int(boundary.sum()),
            "exact_values": sorted(frame.loc[boundary, "true_pic50_m"].tolist()),
        },
        "metrics": metrics,
        "classification_sensitivity": classification,
        "decision": {
            "primary_v13_7_metrics_unchanged": True,
            "v9_external_advantage_robust": bool(
                metrics["posthoc_non_extreme_sensitivity"]["paired_scaffold_bootstrap"]["ci95"][1] < 0
            ),
            "receptor_classification_gain_not_created_by_extremes": bool(
                classification["posthoc_non_extreme_sensitivity"]["paired_within_class_bootstrap"]["ci95"][0]
                > 0
                and classification["posthoc_non_extreme_sensitivity"]["paired_scaffold_cluster_bootstrap"][
                    "ci95"
                ][0]
                > 0
            ),
            "interpretation": (
                "publisher-exact extreme labels dominate four errors and all miss the frozen 90% "
                "interval, but excluding them only as a transparent post-hoc sensitivity makes "
                "V9's advantage over the source model larger; no label is removed from primary results"
            ),
        },
        "artifacts": {
            "row_influence_audit_sha256": _sha(audit_path),
            "publisher_boundary_trace_sha256": _sha(trace_path),
        },
    }
    result = _write_json(output / "analysis_report.json", report, "report_sha256")
    sensitivity = result["metrics"]["posthoc_non_extreme_sensitivity"]
    bootstrap_result = sensitivity["paired_scaffold_bootstrap"]
    class_sensitivity = result["classification_sensitivity"]["posthoc_non_extreme_sensitivity"]
    markdown = f"""# hERG V13.14 external boundary-label sensitivity

The four exact extreme V13.7 labels (2, 2, 3, and 9 pIC50) were verified directly in the
publisher's Table S8. They are not parsing errors. Table S8 does not carry qualifier/censor
metadata for these rows, so all four remain in every primary result.

| Population | N | V9 MAE | Source-model MAE | V9-source delta | 90% interval coverage |
|---|---:|---:|---:|---:|---:|
| Primary, all strict rows | 110 | {metrics["primary_all_rows"]["v9_mixed_ic50"]["mae"]:.4f} | {metrics["primary_all_rows"]["source_2025_model"]["mae"]:.4f} | {metrics["primary_all_rows"]["v9_minus_source_mae"]:+.4f} | {metrics["primary_all_rows"]["global_interval90_coverage"]:.1%} |
| Post-hoc non-extreme sensitivity | 106 | {sensitivity["v9_mixed_ic50"]["mae"]:.4f} | {sensitivity["source_2025_model"]["mae"]:.4f} | {sensitivity["v9_minus_source_mae"]:+.4f} | {sensitivity["global_interval90_coverage"]:.1%} |
| Four exact extremes | 4 | {metrics["exact_extreme_rows"]["v9_mixed_ic50"]["mae"]:.4f} | {metrics["exact_extreme_rows"]["source_2025_model"]["mae"]:.4f} | {metrics["exact_extreme_rows"]["v9_minus_source_mae"]:+.4f} | {metrics["exact_extreme_rows"]["global_interval90_coverage"]:.1%} |

On the 106-row sensitivity set, the paired scaffold-bootstrap V9-source MAE delta is
{bootstrap_result["delta_mae_v9_minus_source"]:+.4f}, 95% CI
[{bootstrap_result["ci95"][0]:+.4f}, {bootstrap_result["ci95"][1]:+.4f}]. The external V9
advantage is therefore not created by the four extremes; it becomes larger when they are
set aside diagnostically. This sensitivity is post-hoc and does not replace the primary result.

The V13.7 receptor balanced-accuracy delta is also stable: on the 106 non-extreme rows it is
{class_sensitivity["paired_within_class_bootstrap"]["delta_balanced_accuracy_candidate_minus_baseline"]:+.4f},
with within-class CI [{class_sensitivity["paired_within_class_bootstrap"]["ci95"][0]:+.4f},
{class_sensitivity["paired_within_class_bootstrap"]["ci95"][1]:+.4f}] and scaffold-cluster CI
[{class_sensitivity["paired_scaffold_cluster_bootstrap"]["ci95"][0]:+.4f},
{class_sensitivity["paired_scaffold_cluster_bootstrap"]["ci95"][1]:+.4f}]. The four exact
extremes do not create the first-campaign classification gain either; V13.8 nonconfirmation
and both external safety failures still control the platform decision.
"""
    (output / "REPORT.md").write_text(markdown)
    _write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "analysis_report_sha256": _sha(output / "analysis_report.json"),
            "report_markdown_sha256": _sha(output / "REPORT.md"),
            "row_influence_audit_sha256": _sha(audit_path),
            "publisher_boundary_trace_sha256": _sha(trace_path),
            "publisher_workbook_sha256": _sha(workbook_path),
        },
        "manifest_sha256",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("research/local_runs/herg_external_validation_v13_7"),
    )
    parser.add_argument(
        "--uncertainty",
        type=Path,
        default=Path("research/local_runs/herg_external_uncertainty_v13_11"),
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path("research/external_validation_sources/tx5c00065_si_002.xlsx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/local_runs/herg_external_boundary_sensitivity_v13_14"),
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    result = analyze(args.campaign, args.uncertainty, args.workbook, args.output, args.bootstrap)
    print(json.dumps({"status": result["status"], "report_sha256": result["report_sha256"]}))


if __name__ == "__main__":
    main()
