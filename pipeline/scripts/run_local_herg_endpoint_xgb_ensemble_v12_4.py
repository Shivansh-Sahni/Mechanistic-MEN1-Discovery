#!/usr/bin/env python3
"""Confirm a frozen two-surface XGBoost endpoint ensemble on unused curves.

Exploration on the strict V10.1 OOF rows selected the unweighted average of
XGBoost RDKit2D and XGBoost RDKit2D+Morgan predictions. This script records
those strict results as post-hoc development evidence, then trains both models
on all strict structures and evaluates their frozen average on disjoint
structures having empirical crossings but failing the original strict curve-QC
gate. The confirmation labels are lower quality; they are not external data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import analyze_local_herg_receptor_ensemble_v13_2 as v132
import joblib
import numpy as np
import pandas as pd
import run_local_herg_v10_1_expanded_platform as v101
from sklearn.model_selection import GroupKFold

SCHEMA_VERSION = "platform-local-herg-endpoint-xgb-ensemble-v12.4/1.0"
ENDPOINTS = (10, 30, 50)
HISTORICAL_WINNER_SURFACE = {10: "rdkit2d_morgan", 30: "rdkit2d_morgan", 50: "rdkit2d"}


class CampaignError(RuntimeError):
    """Endpoint ensemble integrity or scientific-contract failure."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, payload: dict[str, Any], hash_field: str) -> dict[str, Any]:
    body = dict(payload)
    body.pop(hash_field, None)
    body[hash_field] = hashlib.sha256(_canonical(body)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(body))
    temporary.replace(path)
    return body


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _collapsed_endpoint(
    labels: pd.DataFrame,
    endpoint: int,
    strict: bool,
    excluded_keys: set[str] | None = None,
) -> pd.DataFrame:
    column = f"empirical_ic{endpoint}_um"
    keep = labels[column].notna() & labels[column].gt(0)
    keep &= labels.strict_curve_qc if strict else ~labels.strict_curve_qc
    frame = labels.loc[keep].copy()
    if excluded_keys:
        frame = frame.loc[~frame.standard_inchi_key.astype(str).isin(excluded_keys)]
    frame["target"] = 6.0 - np.log10(frame[column].to_numpy(float))
    return (
        frame.groupby("standard_inchi_key", as_index=False)
        .agg(
            standardized_smiles=("standardized_smiles", "first"),
            scaffold_group_id=("scaffold_group_id", "first"),
            target=("target", "median"),
            replicate_count=("target", "size"),
        )
        .reset_index(drop=True)
    )


def _feature_matrix(
    repo: Path,
    output: Path,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    schema = v101._read_json(  # noqa: SLF001
        repo
        / "research/local_runs/herg_domain_mixture_campaign_v9/final_model/feature_preprocessing_schema.json"
    )
    all_features = [str(value) for value in schema["feature_columns"]]
    rdkit = [name for name in all_features if name.startswith("rdkit2d__")]
    cache = output / "prepared/empirical_feature_matrix.parquet"
    if cache.is_file():
        matrix = pd.read_parquet(cache)
        if set(["standard_inchi_key", *all_features]) - set(matrix):
            raise CampaignError("cached empirical feature matrix has an incompatible schema")
        return matrix, rdkit, all_features
    endpoint_columns = [f"empirical_ic{endpoint}_um" for endpoint in ENDPOINTS]
    eligible = labels[endpoint_columns].notna().any(axis=1)
    union = (
        labels.loc[eligible]
        .groupby("standard_inchi_key", as_index=False)
        .agg(
            standardized_smiles=("standardized_smiles", "first"),
            scaffold_group_id=("scaffold_group_id", "first"),
        )
    )
    features = v101._empirical_feature_frame(union, all_features).reset_index(drop=True)  # noqa: SLF001
    features.insert(0, "standard_inchi_key", union.standard_inchi_key.to_numpy())
    _parquet(cache, features)
    return features, rdkit, all_features


def _fit_endpoint(
    output: Path,
    endpoint: int,
    strict: pd.DataFrame,
    confirmation: pd.DataFrame,
    features: pd.DataFrame,
    rdkit: list[str],
    all_features: list[str],
    workers: int,
    bootstrap: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    strict_x = strict[["standard_inchi_key"]].merge(
        features, on="standard_inchi_key", validate="one_to_one"
    )
    confirmation_x = confirmation[["standard_inchi_key"]].merge(
        features, on="standard_inchi_key", validate="one_to_one"
    )
    y = strict.target.to_numpy(float)
    confirmation_y = confirmation.target.to_numpy(float)
    folds = list(GroupKFold(5).split(strict_x, y, strict.scaffold_group_id.to_numpy(str)))
    surfaces = {"rdkit2d": rdkit, "rdkit2d_morgan": all_features}
    strict_predictions: dict[str, np.ndarray] = {}
    confirmation_predictions: dict[str, np.ndarray] = {}
    models = {}
    for surface, columns in surfaces.items():
        oof = np.full(len(strict), np.nan)
        for fold, (fit, evaluate) in enumerate(folds):
            model = v101._empirical_candidate("xgboost", workers, v101.SEED + fold)  # noqa: SLF001
            model.fit(v101._safe_numeric(strict_x.iloc[fit], columns), y[fit])  # noqa: SLF001
            oof[evaluate] = model.predict(  # noqa: SLF001
                v101._safe_numeric(strict_x.iloc[evaluate], columns)  # noqa: SLF001
            )
        strict_predictions[surface] = oof
        final = v101._empirical_candidate("xgboost", workers, v101.SEED + 99)  # noqa: SLF001
        final.fit(v101._safe_numeric(strict_x, columns), y)  # noqa: SLF001
        confirmation_predictions[surface] = final.predict(  # noqa: SLF001
            v101._safe_numeric(confirmation_x, columns)  # noqa: SLF001
        )
        models[surface] = {"model": final, "feature_columns": columns}
    strict_ensemble = np.mean(np.column_stack(list(strict_predictions.values())), axis=1)
    confirmation_ensemble = np.mean(
        np.column_stack(list(confirmation_predictions.values())), axis=1
    )
    winner_surface = HISTORICAL_WINNER_SURFACE[endpoint]
    strict_baseline = strict_predictions[winner_surface]
    confirmation_baseline = confirmation_predictions[winner_surface]
    strict_bootstrap = v132._bootstrap_delta(  # noqa: SLF001
        strict,
        strict_baseline,
        strict_ensemble,
        y,
        bootstrap,
    )
    confirmation_bootstrap = v132._bootstrap_delta(  # noqa: SLF001
        confirmation,
        confirmation_baseline,
        confirmation_ensemble,
        confirmation_y,
        bootstrap,
    )
    rows = []
    for split, frame, observed, baseline, ensemble, by_surface in (
        ("strict_oof_development", strict, y, strict_baseline, strict_ensemble, strict_predictions),
        (
            "lower_qc_structure_disjoint_confirmation",
            confirmation,
            confirmation_y,
            confirmation_baseline,
            confirmation_ensemble,
            confirmation_predictions,
        ),
    ):
        for index in range(len(frame)):
            rows.append(
                {
                    "endpoint": f"IC{endpoint}",
                    "split": split,
                    "standard_inchi_key": frame.iloc[index].standard_inchi_key,
                    "scaffold_group_id": frame.iloc[index].scaffold_group_id,
                    "observed_picx": float(observed[index]),
                    "historical_surface_prediction": float(baseline[index]),
                    "xgb_rdkit2d_prediction": float(by_surface["rdkit2d"][index]),
                    "xgb_rdkit2d_morgan_prediction": float(
                        by_surface["rdkit2d_morgan"][index]
                    ),
                    "frozen_xgb_equal_ensemble_prediction": float(ensemble[index]),
                }
            )
    model_path = output / f"models/ic{endpoint}_xgb_equal_ensemble.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "endpoint": f"IC{endpoint}",
            "ensemble": "unweighted mean",
            "members": models,
            "development_was_posthoc": True,
            "confirmation_label_quality": "empirical crossings failing original strict curve QC",
        },
        model_path,
        compress=3,
    )
    report = {
        "historical_winner_surface": winner_surface,
        "strict_oof_development": {
            "posthoc": True,
            "n": len(strict),
            "historical": v101._regression_metrics(y, strict_baseline),  # noqa: SLF001
            "frozen_ensemble": v101._regression_metrics(y, strict_ensemble),  # noqa: SLF001
            "bootstrap": strict_bootstrap,
        },
        "lower_qc_structure_disjoint_confirmation": {
            "n": len(confirmation),
            "unique_scaffolds": int(confirmation.scaffold_group_id.nunique()),
            "historical_surface_model": v101._regression_metrics(  # noqa: SLF001
                confirmation_y, confirmation_baseline
            ),
            "frozen_ensemble": v101._regression_metrics(  # noqa: SLF001
                confirmation_y, confirmation_ensemble
            ),
            "bootstrap": confirmation_bootstrap,
        },
        "model_path": str(model_path.resolve()),
    }
    return pd.DataFrame(rows), report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/local_runs/herg_endpoint_xgb_ensemble_v12_4"),
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser


def _main(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    output = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels_path = (
        repo
        / "research/local_runs/herg_v10_1_expanded_platform/evidence/empirical_ic10_ic30_ic50_labels.parquet"
    )
    labels = pd.read_parquet(labels_path)
    features, rdkit, all_features = _feature_matrix(repo, output, labels)
    endpoint_reports = {}
    evidence = []
    for endpoint in ENDPOINTS:
        strict = _collapsed_endpoint(labels, endpoint, strict=True)
        confirmation = _collapsed_endpoint(
            labels,
            endpoint,
            strict=False,
            excluded_keys=set(strict.standard_inchi_key.astype(str)),
        )
        if set(strict.standard_inchi_key.astype(str)) & set(
            confirmation.standard_inchi_key.astype(str)
        ):
            raise CampaignError(f"IC{endpoint} confirmation overlaps a strict training structure")
        rows, endpoint_report = _fit_endpoint(
            output,
            endpoint,
            strict,
            confirmation,
            features,
            rdkit,
            all_features,
            args.workers,
            args.bootstrap_replicates,
        )
        evidence.append(rows)
        endpoint_reports[f"IC{endpoint}"] = endpoint_report
        print(json.dumps({"stage": "endpoint_complete", "endpoint": endpoint}), flush=True)
    evidence_path = output / "endpoint_ensemble_predictions.parquet"
    _parquet(evidence_path, pd.concat(evidence, ignore_index=True))
    confirmation_deltas = {
        endpoint: details["lower_qc_structure_disjoint_confirmation"]["bootstrap"]
        for endpoint, details in endpoint_reports.items()
    }
    confirmed = bool(
        all(result["delta_mae_candidate_minus_baseline"] <= 0 for result in confirmation_deltas.values())
        and sum(result["ci95"][1] < 0 for result in confirmation_deltas.values()) >= 2
    )
    report = _json(
        output / "analysis_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete_sequential_lower_quality_confirmation",
            "endpoint_reports": endpoint_reports,
            "confirmation_decision": {
                "confirmed": confirmed,
                "rule": (
                    "no endpoint has positive observed confirmation MAE delta and at least two "
                    "endpoint scaffold-bootstrap upper 95% bounds are <0"
                ),
                "promotion_scope": (
                    "internal lower-QC empirical curves only; external or strict-QC confirmation "
                    "still required"
                ),
            },
            "scientific_scope": {
                "strict_oof_ensemble_selection_was_posthoc": True,
                "confirmation_structures_disjoint_from_strict_training": True,
                "confirmation_curves_failed_original_strict_qc": True,
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
                "external_or_prospective_validation": False,
            },
        },
        "report_sha256",
    )
    inputs = [
        repo / "pipeline/scripts/run_local_herg_endpoint_xgb_ensemble_v12_4.py",
        labels_path,
        repo / "research/local_runs/herg_v10_1_expanded_platform/manifest.json",
        repo
        / "research/local_runs/herg_domain_mixture_campaign_v9/final_model/feature_preprocessing_schema.json",
    ]
    artifacts = [
        output / "prepared/empirical_feature_matrix.parquet",
        evidence_path,
        *(output / f"models/ic{endpoint}_xgb_equal_ensemble.joblib" for endpoint in ENDPOINTS),
        output / "analysis_report.json",
    ]
    manifest = _json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete",
            "inputs": [
                {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}
                for path in inputs
            ],
            "artifacts": [
                {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}
                for path in artifacts
            ],
        },
        "manifest_sha256",
    )
    return _json(
        output / "final_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "finished_utc": _utc(),
            "status": "complete",
            "lower_quality_confirmation_passed": confirmed,
            "report_sha256": report["report_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "summary_sha256",
    )


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _main(args)
    except CampaignError as exc:
        print(f"V12.4 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
