#!/usr/bin/env python3
"""Audit V11 and run fold-aligned, end-to-end V12 endpoint refinements.

This campaign never opens the repository validation or test partitions. V11 is
treated as a completed negative experiment. V12.2 uses the existing paired
strict-qHTS cohort and its fixed scaffold folds to compare three predeclared
endpoint formulations: independent XGBoost regressors, projected independent
predictions, and an end-to-end coherent IC50-plus-gap model. A shared-tree
multi-output control tests whether common partitions help without pretending
that post-hoc OOF ranking is an unbiased model-selection estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from xgboost import XGBRegressor

SCHEMA_VERSION = "platform-local-herg-anchor-refinement-v12.2/1.0"
SEED = 20260821
ENDPOINTS = (10, 30, 50)
CLASS_MODELS = ("xgb_rdkit2d", "xgb_rdkit2d_morgan", "lgbm_rdkit2d", "lgbm_rdkit2d_morgan")


class CampaignError(RuntimeError):
    """Integrity or scientific-contract failure."""


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


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = np.asarray(y, dtype=float) - np.asarray(prediction, dtype=float)
    rho = spearmanr(y, prediction).statistic
    return {
        "n": len(y),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": float(rho) if np.isfinite(rho) else None,
        "within_0p5": float(np.mean(np.abs(error) <= 0.5)),
        "within_1p0": float(np.mean(np.abs(error) <= 1.0)),
    }


def _classification_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    prediction = np.argmax(probabilities, axis=1)
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "potent_predicted_safe_rate": float(np.mean(prediction[y == 2] == 0)),
    }


def _project_nonincreasing(values: np.ndarray) -> np.ndarray:
    blocks: list[tuple[float, int]] = []
    for value in -np.asarray(values, dtype=float):
        blocks.append((float(value), 1))
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right, right_n = blocks.pop()
            left, left_n = blocks.pop()
            total = left_n + right_n
            blocks.append(((left * left_n + right * right_n) / total, total))
    return -np.concatenate([np.repeat(value, count) for value, count in blocks])


def _xgb(seed: int, workers: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.035,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.4,
        reg_lambda=6.0,
        max_bin=128,
        tree_method="hist",
        objective="reg:squarederror",
        n_jobs=workers,
        random_state=seed,
        verbosity=0,
    )


def _safe_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame[columns].replace([np.inf, -np.inf], np.nan)
    return result.mask(result.abs() > 1e30).fillna(0.0).astype(np.float32)


def _bootstrap_delta(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    observed: np.ndarray,
    replicates: int,
) -> dict[str, Any]:
    groups = frame.scaffold_group_id.astype(str).to_numpy()
    codes, unique = pd.factorize(groups, sort=True)
    counts = np.bincount(codes, minlength=len(unique)).astype(float)
    error_difference = np.abs(observed - candidate) - np.abs(observed - baseline)
    group_error_difference = np.bincount(codes, weights=error_difference, minlength=len(unique))
    rng = np.random.default_rng(SEED)
    deltas = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 256):
        stop = min(replicates, start + 256)
        sampled = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        deltas[start:stop] = np.sum(group_error_difference[sampled], axis=1) / np.sum(
            counts[sampled], axis=1
        )
    return {
        "delta_mae_candidate_minus_baseline": float(
            np.mean(np.abs(observed - candidate)) - np.mean(np.abs(observed - baseline))
        ),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "probability_candidate_better": float(np.mean(deltas < 0)),
        "replicates": replicates,
        "resampling_unit": "scaffold_group_id",
    }


def _v11_audit(repo: Path, bootstrap: int) -> tuple[dict[str, Any], pd.DataFrame]:
    path = repo / "research/local_runs/herg_comprehensive_optimization_v11_1/analysis/nested_oof_predictions.parquet"
    frame = pd.read_parquet(path)
    candidates = [column for column in frame if column.startswith("pred__")]
    rows = []
    for column in ["v9_predicted_pic50", *candidates]:
        keep = frame[column].notna().to_numpy()
        values = frame.loc[keep]
        metric = _metrics(values.observed_pic50.to_numpy(float), values[column].to_numpy(float))
        rows.append({"candidate": column, **metric, "outer_folds_covered": int(values.outer_fold.nunique())})
    landscape = pd.DataFrame(rows).sort_values(["outer_folds_covered", "mae"], ascending=[False, True])
    y = frame.observed_pic50.to_numpy(float)
    v9 = frame.v9_predicted_pic50.to_numpy(float)
    v11 = frame.pred__v11_nested.to_numpy(float)
    fold_deltas = []
    for fold, group in frame.groupby("outer_fold"):
        fold_deltas.append(
            {
                "outer_fold": int(fold),
                "v9_mae": _metrics(group.observed_pic50.to_numpy(), group.v9_predicted_pic50.to_numpy())["mae"],
                "v11_mae": _metrics(group.observed_pic50.to_numpy(), group.pred__v11_nested.to_numpy())["mae"],
            }
        )
    for row in fold_deltas:
        row["delta_v11_minus_v9"] = row["v11_mae"] - row["v9_mae"]
    report = {
        "v9": _metrics(y, v9),
        "v11_nested": _metrics(y, v11),
        "paired_scaffold_bootstrap": _bootstrap_delta(frame, v9, v11, y, bootstrap),
        "outer_fold_deltas": fold_deltas,
        "decision": "retain_v9_anchor",
        "reason": (
            "V11 is worse overall and its fully covered candidates do not beat V9. Partial-fold candidates "
            "remain exploratory and are not promoted by subset performance."
        ),
    }
    return report, landscape


def _class_audit(repo: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    path = repo / "research/local_runs/herg_v10_1_expanded_platform/evidence/exact_router_oof.parquet"
    frame = pd.read_parquet(path)
    y = frame.observed_tier.to_numpy(int)
    predictions: dict[str, np.ndarray] = {}
    for model in CLASS_MODELS:
        predictions[model] = frame[
            [
                f"{model}__probability_safe",
                f"{model}__probability_moderate",
                f"{model}__probability_potent",
            ]
        ].to_numpy(float)
    predictions["equal_weight_four_model_ensemble"] = np.mean(list(predictions.values()), axis=0)
    report = {name: _classification_metrics(y, value) for name, value in predictions.items()}
    report["decision"] = "retain_lgbm_rdkit2d_morgan"
    report["selection_boundary"] = (
        "equal-weight ensemble is a fixed diagnostic; no OOF-tuned class thresholds or weights are reported"
    )
    evidence = frame[["structure_id", "scaffold_group_id", "outer_fold", "observed_tier"]].copy()
    for name, values in predictions.items():
        evidence[f"{name}__prediction"] = np.argmax(values, axis=1)
    return report, evidence


def _endpoint_inputs(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    v12 = repo / "research/local_runs/herg_endpoint_receptor_campaign_v12_1"
    labels = pd.read_parquet(v12 / "endpoints/coherent_gap_oof.parquet")
    features = pd.read_parquet(v12 / "prepared/direct_paired_features.parquet")
    frame = labels.merge(features, on="standard_inchi_key", validate="one_to_one")
    schema = json.loads(
        (repo / "research/local_runs/herg_domain_mixture_campaign_v9/final_model/feature_preprocessing_schema.json").read_text()
    )
    all_features = [str(value) for value in schema["feature_columns"]]
    rdkit = [column for column in all_features if column.startswith("rdkit2d__")]
    raw = pd.read_parquet(
        repo / "research/local_runs/herg_v10_1_expanded_platform/evidence/empirical_ic10_ic30_ic50_oof.parquet"
    )
    pivot = raw.pivot_table(index="standard_inchi_key", columns="endpoint", values="predicted_picx")
    pivot.columns = [f"historical_predicted_pic{str(endpoint)[2:]}" for endpoint in pivot.columns]
    frame = frame.merge(pivot.reset_index(), on="standard_inchi_key", validate="one_to_one")
    if set(frame.outer_fold.astype(int)) != set(range(5)):
        raise CampaignError("V12 common-fold contract is incomplete")
    return frame, features, all_features, rdkit


def _endpoint_campaign(
    repo: Path,
    output: Path,
    workers: int,
    bootstrap: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    frame, _, all_features, rdkit = _endpoint_inputs(repo)
    prediction_path = output / "endpoints/common_fold_oof_predictions.parquet"
    if prediction_path.is_file():
        evidence = pd.read_parquet(prediction_path)
        required = {
            *(f"independent_pic{endpoint}" for endpoint in ENDPOINTS),
            *(f"shared_tree_pic{endpoint}" for endpoint in ENDPOINTS),
        }
        if len(evidence) != len(frame) or not required.issubset(evidence):
            raise CampaignError("cached V12.2 endpoint evidence is incompatible")
    else:
        evidence = frame[
            [
                "standard_inchi_key",
                "scaffold_group_id",
                "outer_fold",
                "observed_pic10",
                "observed_pic30",
                "observed_pic50",
                "historical_predicted_pic10",
                "historical_predicted_pic30",
                "historical_predicted_pic50",
                "predicted_gap_pic10_minus_pic30",
                "predicted_gap_pic30_minus_pic50",
            ]
        ].copy()
        for endpoint in ENDPOINTS:
            evidence[f"independent_pic{endpoint}"] = np.nan
        for endpoint in ENDPOINTS:
            evidence[f"shared_tree_pic{endpoint}"] = np.nan
        for fold in range(5):
            fit = frame.outer_fold.ne(fold).to_numpy()
            evaluate = ~fit
            for endpoint in ENDPOINTS:
                columns = rdkit if endpoint == 50 else all_features
                model = _xgb(SEED + 100 * endpoint + fold, workers)
                model.fit(
                    _safe_numeric(frame.loc[fit], columns),
                    frame.loc[fit, f"observed_pic{endpoint}"].to_numpy(float),
                )
                evidence.loc[evaluate, f"independent_pic{endpoint}"] = model.predict(
                    _safe_numeric(frame.loc[evaluate], columns)
                )
            shared = ExtraTreesRegressor(
                n_estimators=500,
                max_features=0.5,
                min_samples_leaf=2,
                n_jobs=workers,
                random_state=SEED + fold,
            )
            shared.fit(
                _safe_numeric(frame.loc[fit], all_features),
                frame.loc[fit, [f"observed_pic{endpoint}" for endpoint in ENDPOINTS]],
            )
            shared_prediction = shared.predict(_safe_numeric(frame.loc[evaluate], all_features))
            for column, endpoint in enumerate(ENDPOINTS):
                evidence.loc[evaluate, f"shared_tree_pic{endpoint}"] = shared_prediction[:, column]
            print(json.dumps({"stage": "v12.2_common_fold", "completed_fold": fold}), flush=True)
        _parquet(prediction_path, evidence)

    independent = evidence[[f"independent_pic{endpoint}" for endpoint in ENDPOINTS]].to_numpy(float)
    shared = evidence[[f"shared_tree_pic{endpoint}" for endpoint in ENDPOINTS]].to_numpy(float)
    projected_independent = np.vstack([_project_nonincreasing(row) for row in independent])
    projected_shared = np.vstack([_project_nonincreasing(row) for row in shared])
    coherent_gap = np.column_stack(
        [
            evidence.independent_pic50
            + evidence.predicted_gap_pic30_minus_pic50
            + evidence.predicted_gap_pic10_minus_pic30,
            evidence.independent_pic50 + evidence.predicted_gap_pic30_minus_pic50,
            evidence.independent_pic50,
        ]
    )
    for column, endpoint in enumerate(ENDPOINTS):
        evidence[f"projected_independent_pic{endpoint}"] = projected_independent[:, column]
        evidence[f"projected_shared_tree_pic{endpoint}"] = projected_shared[:, column]
        evidence[f"coherent_gap_end_to_end_pic{endpoint}"] = coherent_gap[:, column]
    _parquet(prediction_path, evidence)

    surfaces = {
        "historical_v10_1_separate_oof": "historical_predicted",
        "independent_common_fold": "independent",
        "projected_independent_common_fold": "projected_independent",
        "shared_tree_common_fold": "shared_tree",
        "projected_shared_tree_common_fold": "projected_shared_tree",
        "coherent_gap_end_to_end": "coherent_gap_end_to_end",
    }
    report: dict[str, Any] = {"surfaces": {}, "order_violations": {}}
    for surface, prefix in surfaces.items():
        report["surfaces"][surface] = {}
        values = evidence[[f"{prefix}_pic{endpoint}" for endpoint in ENDPOINTS]].to_numpy(float)
        report["order_violations"][surface] = int(
            np.sum((values[:, 0] < values[:, 1]) | (values[:, 1] < values[:, 2]))
        )
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            observed = evidence[f"observed_pic{endpoint}"].to_numpy(float)
            prediction = values[:, endpoint_index]
            baseline = evidence[f"historical_predicted_pic{endpoint}"].to_numpy(float)
            report["surfaces"][surface][f"IC{endpoint}"] = {
                **_metrics(observed, prediction),
                "vs_historical_scaffold_bootstrap": _bootstrap_delta(
                    evidence, baseline, prediction, observed, bootstrap
                ),
            }
    report["selection_boundary"] = (
        "all fixed candidates are reported; relative OOF ranking is exploratory and no winner is claimed "
        "without a future nested or external confirmation"
    )
    report["isolation_boundary"] = (
        "coherent_gap_end_to_end replaces V12's observed IC50 anchor with a common-fold predicted IC50"
    )

    final_models: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "feature_columns": {}}
    for endpoint in ENDPOINTS:
        columns = rdkit if endpoint == 50 else all_features
        model = _xgb(SEED + 10_000 + endpoint, workers)
        model.fit(_safe_numeric(frame, columns), frame[f"observed_pic{endpoint}"].to_numpy(float))
        final_models[f"independent_ic{endpoint}"] = model
        final_models["feature_columns"][f"independent_ic{endpoint}"] = columns
    shared_final = ExtraTreesRegressor(
        n_estimators=500,
        max_features=0.5,
        min_samples_leaf=2,
        n_jobs=workers,
        random_state=SEED + 20_000,
    )
    shared_final.fit(
        _safe_numeric(frame, all_features),
        frame[[f"observed_pic{endpoint}" for endpoint in ENDPOINTS]],
    )
    final_models["shared_tree"] = shared_final
    final_models["feature_columns"]["shared_tree"] = all_features
    return report, evidence, final_models


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/local_runs/herg_anchor_refinement_campaign_v12_2"),
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser


def _main(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    output = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    v11, v11_landscape = _v11_audit(repo, args.bootstrap_replicates)
    classification, classification_evidence = _class_audit(repo)
    endpoints, endpoint_evidence, final_models = _endpoint_campaign(
        repo, output, args.workers, args.bootstrap_replicates
    )
    _parquet(output / "v11/candidate_audit.parquet", v11_landscape)
    _parquet(output / "classification/ensemble_diagnostic.parquet", classification_evidence)
    model_path = output / "models/v12_2_endpoint_models.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_suffix(".joblib.tmp")
    joblib.dump(final_models, temporary_model, compress=3)
    temporary_model.replace(model_path)
    report = _json(
        output / "analysis_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete_internal_anchor_refinement",
            "v11": v11,
            "classification": classification,
            "direct_endpoints": endpoints,
            "scientific_scope": {
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
                "fixed_scaffold_folds": True,
                "external_or_prospective_validation": False,
                "posthoc_oof_candidate_ranking_is_confirmatory": False,
            },
        },
        "report_sha256",
    )
    inputs = [
        repo / "pipeline/scripts/run_local_herg_anchor_refinement_campaign_v12_2.py",
        repo / "research/local_runs/herg_comprehensive_optimization_v11_1/analysis/nested_oof_predictions.parquet",
        repo / "research/local_runs/herg_v10_1_expanded_platform/evidence/exact_router_oof.parquet",
        repo / "research/local_runs/herg_v10_1_expanded_platform/evidence/empirical_ic10_ic30_ic50_oof.parquet",
        repo / "research/local_runs/herg_endpoint_receptor_campaign_v12_1/endpoints/coherent_gap_oof.parquet",
        repo / "research/local_runs/herg_endpoint_receptor_campaign_v12_1/prepared/direct_paired_features.parquet",
    ]
    artifacts = [
        output / "analysis_report.json",
        output / "v11/candidate_audit.parquet",
        output / "classification/ensemble_diagnostic.parquet",
        output / "endpoints/common_fold_oof_predictions.parquet",
        model_path,
    ]
    manifest = _json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "inputs": [
                {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}
                for path in inputs
            ],
            "artifacts": [
                {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}
                for path in artifacts
            ],
            "versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
            },
        },
        "manifest_sha256",
    )
    summary = _json(
        output / "final_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "finished_utc": _utc(),
            "status": "complete",
            "v11_decision": report["v11"]["decision"],
            "classification_decision": report["classification"]["decision"],
            "paired_direct_structures": len(endpoint_evidence),
            "manifest_sha256": manifest["manifest_sha256"],
            "report_sha256": report["report_sha256"],
        },
        "summary_sha256",
    )
    return summary


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _main(args)
    except CampaignError as exc:
        print(f"V12.2 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
