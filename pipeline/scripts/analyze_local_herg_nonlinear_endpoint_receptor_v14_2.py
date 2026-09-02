#!/usr/bin/env python3
"""Challenge direct IC10/IC30/IC50 models with nonlinear receptor features.

V14.2 uses the frozen 90-compound six-state receptor panel from V13.2.  The
endpoint anchor is regenerated inside each outer scaffold fold exactly as in
the strict V13.2 stack.  ExtraTrees regularization is then selected only from
the four outer-training folds, and the fifth fold stays untouched.  Results
are compared with both the strict common-fold anchor and the stronger
historical V10.1 OOF reference.  The panel was already opened, so the entire
exercise is an exploratory stress test and cannot promote a website model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import run_local_herg_receptor_ensemble_campaign_v13 as v13
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

SCHEMA_VERSION = "platform-local-herg-nonlinear-endpoint-receptor-v14.2/1.0"
SEED = 20260831
ENDPOINTS = (10, 30, 50)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V132 = REPO_ROOT / "research/local_runs/herg_receptor_strict_analysis_v13_2"
DEFAULT_OUTPUT = REPO_ROOT / "research/local_runs/herg_nonlinear_endpoint_receptor_v14_2"
N_JOBS = 4

TREE_CONFIGS: tuple[dict[str, Any], ...] = (
    {"name": "leaf4_sqrt", "min_samples_leaf": 4, "max_features": "sqrt"},
    {"name": "leaf4_fraction70", "min_samples_leaf": 4, "max_features": 0.70},
    {"name": "leaf8_fraction70", "min_samples_leaf": 8, "max_features": 0.70},
    {"name": "leaf16_all", "min_samples_leaf": 16, "max_features": 1.0},
)


class CampaignError(RuntimeError):
    """Raised when an endpoint stress-test contract is violated."""


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any], hash_field: str) -> dict[str, Any]:
    body = dict(payload)
    body.pop(hash_field, None)
    body[hash_field] = hashlib.sha256(_canonical(body)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(body))
    temporary.replace(path)
    return body


def _read_json(path: Path, hash_field: str) -> dict[str, Any]:
    body = json.loads(path.read_text())
    expected = body.get(hash_field)
    candidate = dict(body)
    candidate.pop(hash_field, None)
    if expected != hashlib.sha256(_canonical(candidate)).hexdigest():
        raise CampaignError(f"self-hash mismatch: {path}")
    return body


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _deduplicate(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _feature_surfaces(frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    ligand = [
        "docking_molecular_weight",
        "docking_heavy_atom_count",
        "docking_rotatable_bond_count",
        "dock__8ZYO__formal_charge",
    ]
    scores = [
        column
        for column in frame
        if column.startswith("dock__")
        and (
            "__affinity" in column
            or "__ligand_efficiency" in column
            or column in {"dock__ensemble_sd_affinity", "dock__ensemble_range_affinity"}
        )
    ]
    contrasts = [
        column
        for column in frame
        if column.startswith("dock__")
        and (
            "_minus_" in column
            or column in {"dock__core_sd_affinity", "dock__core_range_affinity"}
        )
    ]
    contacts = [
        column
        for column in frame
        if column.startswith(("dock__8ZYO__contact_", "dock__8ZYQ__contact_"))
    ]
    all_receptor = [
        column
        for column in frame
        if column.startswith("dock__") and pd.api.types.is_numeric_dtype(frame[column])
    ]
    surfaces = {
        "extratrees_ligand_physchem": _deduplicate(ligand),
        "extratrees_plus_six_state_scores": _deduplicate([*ligand, *scores]),
        "extratrees_plus_state_contrasts": _deduplicate([*ligand, *contrasts]),
        "extratrees_plus_sensitivity_contacts": _deduplicate([*ligand, *contacts]),
        "extratrees_plus_combined_receptor": _deduplicate(
            [*ligand, *scores, *contrasts, *contacts]
        ),
        "extratrees_plus_all_receptor": _deduplicate([*ligand, *all_receptor]),
    }
    for name, features in surfaces.items():
        missing = set(features) - set(frame)
        if not features or missing:
            raise CampaignError(f"feature contract failed for {name}: {sorted(missing)}")
    return surfaces


def _model(config: dict[str, Any], n_estimators: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    criterion="squared_error",
                    min_samples_leaf=int(config["min_samples_leaf"]),
                    max_features=config["max_features"],
                    bootstrap=False,
                    n_jobs=N_JOBS,
                    random_state=SEED,
                ),
            ),
        ]
    )


def _strict_baseline_table(predictions: pd.DataFrame) -> pd.DataFrame:
    direct = predictions.loc[predictions.cohort.eq("direct_curve")].copy()
    counts = direct.groupby(["ligand_id", "endpoint"]).baseline.nunique()
    if counts.ne(1).any():
        raise CampaignError("strict V13.2 baselines differ across feature sets")
    return (
        direct[["ligand_id", "endpoint", "baseline"]]
        .drop_duplicates()
        .pivot(index="ligand_id", columns="endpoint", values="baseline")
    )


def _context_anchors(
    frame: pd.DataFrame,
    inner: pd.DataFrame,
    outer_fold: int,
    endpoint: int,
) -> np.ndarray:
    part = inner.loc[inner.context_outer_fold.eq(outer_fold)].copy()
    column = f"inner_baseline_pic{endpoint}"
    consistency = part.groupby("standard_inchi_key")[column].nunique(dropna=False)
    if consistency.gt(1).any():
        raise CampaignError(f"inner IC{endpoint} anchors are inconsistent in outer fold {outer_fold}")
    lookup = part.drop_duplicates("standard_inchi_key").set_index("standard_inchi_key")[column]
    anchors = frame.standard_inchi_key.map(lookup).to_numpy(float)
    return anchors


def _tune_config(
    training: pd.DataFrame,
    anchors: np.ndarray,
    features: tuple[str, ...],
    endpoint: int,
    n_estimators: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    observed = training[f"observed_pic{endpoint}"].to_numpy(float)
    if not np.isfinite(anchors).all():
        raise CampaignError(f"non-finite inner IC{endpoint} anchors")
    for config in TREE_CONFIGS:
        fold_mae = []
        fold_delta = []
        for inner_fold in sorted(training.outer_fold.astype(int).unique()):
            fit = training.outer_fold.ne(inner_fold).to_numpy()
            evaluate = ~fit
            model = _model(config, n_estimators)
            model.fit(training.loc[fit, list(features)], observed[fit] - anchors[fit])
            prediction = anchors[evaluate] + model.predict(training.loc[evaluate, list(features)])
            error = float(np.mean(np.abs(observed[evaluate] - prediction)))
            baseline_error = float(np.mean(np.abs(observed[evaluate] - anchors[evaluate])))
            fold_mae.append(error)
            fold_delta.append(error - baseline_error)
        rows.append(
            {
                "config": config["name"],
                "min_samples_leaf": config["min_samples_leaf"],
                "max_features": str(config["max_features"]),
                "macro_inner_fold_mae": float(np.mean(fold_mae)),
                "macro_inner_fold_delta_vs_anchor": float(np.mean(fold_delta)),
            }
        )
    evidence = pd.DataFrame(rows)
    chosen_name = str(
        evidence.sort_values(
            ["macro_inner_fold_mae", "macro_inner_fold_delta_vs_anchor", "min_samples_leaf"],
            ascending=[True, True, False],
        ).iloc[0].config
    )
    return next(config for config in TREE_CONFIGS if config["name"] == chosen_name), evidence


def _nested_predictions(
    frame: pd.DataFrame,
    inner: pd.DataFrame,
    strict: pd.DataFrame,
    surfaces: dict[str, tuple[str, ...]],
    n_estimators: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    tuning_rows: list[pd.DataFrame] = []
    for outer_fold in sorted(frame.outer_fold.astype(int).unique()):
        fit_mask = frame.outer_fold.ne(outer_fold).to_numpy()
        evaluation_mask = ~fit_mask
        training = frame.loc[fit_mask].reset_index(drop=True)
        evaluation = frame.loc[evaluation_mask].copy()
        for endpoint in ENDPOINTS:
            all_context = _context_anchors(frame, inner, outer_fold, endpoint)
            training_anchors = all_context[fit_mask]
            if not np.isfinite(training_anchors).all():
                raise CampaignError(f"missing inner IC{endpoint} anchor in outer fold {outer_fold}")
            outer_anchor = evaluation.ligand_id.map(strict[f"IC{endpoint}"]).to_numpy(float)
            if not np.isfinite(outer_anchor).all():
                raise CampaignError(f"missing strict IC{endpoint} outer anchor")
            for surface, features in surfaces.items():
                config, evidence = _tune_config(
                    training,
                    training_anchors,
                    features,
                    endpoint,
                    n_estimators,
                )
                evidence.insert(0, "surface", surface)
                evidence.insert(0, "endpoint", f"IC{endpoint}")
                evidence.insert(0, "outer_fold", outer_fold)
                tuning_rows.append(evidence)
                model = _model(config, n_estimators)
                observed_training = training[f"observed_pic{endpoint}"].to_numpy(float)
                model.fit(
                    training[list(features)],
                    observed_training - training_anchors,
                )
                prediction = outer_anchor + model.predict(evaluation[list(features)])
                for position, (_, row) in enumerate(evaluation.iterrows()):
                    prediction_rows.append(
                        {
                            "ligand_id": row.ligand_id,
                            "standard_inchi_key": row.standard_inchi_key,
                            "scaffold_group_id": row.scaffold_group_id,
                            "outer_fold": int(outer_fold),
                            "endpoint": f"IC{endpoint}",
                            "surface": surface,
                            "selected_config": config["name"],
                            "observed": float(row[f"observed_pic{endpoint}"]),
                            "strict_baseline": float(outer_anchor[position]),
                            "historical_baseline": float(row[f"baseline_predicted_pic{endpoint}"]),
                            "prediction": float(prediction[position]),
                        }
                    )
                print(
                    json.dumps(
                        {
                            "stage": "endpoint_nested_outer",
                            "outer_fold": int(outer_fold),
                            "endpoint": f"IC{endpoint}",
                            "surface": surface,
                            "selected_config": config["name"],
                        }
                    ),
                    flush=True,
                )
    predictions = pd.DataFrame(prediction_rows)
    expected = len(frame) * len(ENDPOINTS) * len(surfaces)
    if len(predictions) != expected:
        raise CampaignError(f"created {len(predictions)} endpoint predictions, expected {expected}")
    return predictions, pd.concat(tuning_rows, ignore_index=True)


def _metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - observed
    rho = float(spearmanr(observed, prediction).statistic)
    return {
        "n": len(observed),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": rho if np.isfinite(rho) else None,
        "within_0p5": float(np.mean(np.abs(error) <= 0.5)),
        "mean_error_prediction_minus_observed": float(np.mean(error)),
    }


def _folds_better(frame: pd.DataFrame, candidate: str, comparator: str) -> int:
    values = []
    for _fold, group in frame.groupby("outer_fold", sort=True):
        values.append(
            float(np.mean(np.abs(group.observed - group[candidate])))
            - float(np.mean(np.abs(group.observed - group[comparator])))
        )
    return int(np.sum(np.asarray(values) < 0))


def _bootstrap_delta(
    frame: pd.DataFrame,
    candidate: str,
    comparator: str,
    replicates: int,
    seed_label: str,
) -> dict[str, Any]:
    difference = np.abs(frame.observed - frame[candidate]) - np.abs(
        frame.observed - frame[comparator]
    )
    codes, groups = pd.factorize(frame.scaffold_group_id.astype(str), sort=True)
    group_sum = np.bincount(codes, weights=difference, minlength=len(groups))
    group_count = np.bincount(codes, minlength=len(groups)).astype(float)
    seed = SEED + int(hashlib.sha256(seed_label.encode()).hexdigest()[:7], 16)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 512):
        stop = min(replicates, start + 512)
        sampled = rng.integers(0, len(groups), size=(stop - start, len(groups)))
        values[start:stop] = np.sum(group_sum[sampled], axis=1) / np.sum(
            group_count[sampled], axis=1
        )
    observed = float(np.mean(difference))
    return {
        "delta_mae_candidate_minus_comparator": observed,
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "bootstrap_probability_candidate_better": float(np.mean(values < 0)),
        "folds_better": _folds_better(frame, candidate, comparator),
        "replicates": replicates,
        "resampling_unit": "scaffold_group_id",
    }


def _score_and_compare(
    predictions: pd.DataFrame,
    surfaces: dict[str, tuple[str, ...]],
    bootstrap_replicates: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    for (endpoint, surface), group in predictions.groupby(["endpoint", "surface"], sort=True):
        metric_rows.append(
            {
                "endpoint": endpoint,
                "surface": surface,
                "features": len(surfaces[surface]),
                **_metrics(group.observed, group.prediction),
                "strict_baseline_mae": _metrics(group.observed, group.strict_baseline)["mae"],
                "historical_baseline_mae": _metrics(group.observed, group.historical_baseline)["mae"],
            }
        )
        comparisons.setdefault(endpoint, {})[surface] = {
            "vs_strict_common_fold_anchor": _bootstrap_delta(
                group,
                "prediction",
                "strict_baseline",
                bootstrap_replicates,
                f"{endpoint}:{surface}:strict",
            ),
            "vs_historical_v10_1_oof": _bootstrap_delta(
                group,
                "prediction",
                "historical_baseline",
                bootstrap_replicates,
                f"{endpoint}:{surface}:historical",
            ),
        }
    metrics = pd.DataFrame(metric_rows)
    decision: dict[str, Any] = {"endpoints": {}}
    ligand_surface = "extratrees_ligand_physchem"
    receptor_surfaces = [
        surface for surface in predictions.surface.unique() if surface != ligand_surface
    ]
    for endpoint in [f"IC{value}" for value in ENDPOINTS]:
        endpoint_metrics = metrics.loc[metrics.endpoint.eq(endpoint)].set_index("surface")
        best_receptor = min(receptor_surfaces, key=lambda name: endpoint_metrics.loc[name, "mae"])
        group = predictions.loc[
            predictions.endpoint.eq(endpoint) & predictions.surface.eq(best_receptor)
        ].copy()
        ligand = predictions.loc[
            predictions.endpoint.eq(endpoint) & predictions.surface.eq(ligand_surface),
            ["ligand_id", "prediction"],
        ].rename(columns={"prediction": "ligand_prediction"})
        group = group.merge(ligand, on="ligand_id", validate="one_to_one")
        incremental = _bootstrap_delta(
            group,
            "prediction",
            "ligand_prediction",
            bootstrap_replicates,
            f"{endpoint}:{best_receptor}:ligand",
        )
        comparisons[endpoint][best_receptor]["vs_equal_capacity_ligand"] = incremental
        historical = comparisons[endpoint][best_receptor]["vs_historical_v10_1_oof"]
        receptor_gate = bool(incremental["ci95"][1] < 0 and incremental["folds_better"] >= 4)
        endpoint_gate = bool(historical["ci95"][1] < 0 and historical["folds_better"] >= 4)
        decision["endpoints"][endpoint] = {
            "best_receptor_surface_posthoc": best_receptor,
            "best_receptor_mae": float(endpoint_metrics.loc[best_receptor, "mae"]),
            "ligand_tree_mae": float(endpoint_metrics.loc[ligand_surface, "mae"]),
            "historical_v10_1_mae": float(
                endpoint_metrics.loc[best_receptor, "historical_baseline_mae"]
            ),
            "receptor_incremental_gate_passed": receptor_gate,
            "endpoint_improvement_gate_passed": endpoint_gate,
        }
    decision["any_receptor_incremental_gate_passed"] = any(
        item["receptor_incremental_gate_passed"] for item in decision["endpoints"].values()
    )
    decision["any_endpoint_improvement_gate_passed"] = any(
        item["endpoint_improvement_gate_passed"] for item in decision["endpoints"].values()
    )
    decision["website_integration_supported"] = False
    decision["production_promotion_supported"] = False
    decision["claim_boundary"] = (
        "surface and tree family were evaluated after the 90 outcomes were open; numeric gates are "
        "diagnostic and cannot replace prospective confirmation"
    )
    return metrics, comparisons, decision


def _coherence(predictions: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for surface in sorted(predictions.surface.unique()):
        selected = predictions.loc[predictions.surface.eq(surface)]
        wide = selected.pivot(index="ligand_id", columns="endpoint", values="prediction")
        observed = selected.pivot(index="ligand_id", columns="endpoint", values="observed")
        wide = wide[["IC10", "IC30", "IC50"]]
        observed = observed.loc[wide.index, ["IC10", "IC30", "IC50"]]
        raw = wide.to_numpy(float)
        projected = np.vstack([v13._project_nonincreasing(row) for row in raw])  # noqa: SLF001
        result[surface] = {
            "raw_order_violations": int(
                np.sum((raw[:, 0] < raw[:, 1]) | (raw[:, 1] < raw[:, 2]))
            ),
            "projected_order_violations": int(
                np.sum(
                    (projected[:, 0] < projected[:, 1])
                    | (projected[:, 1] < projected[:, 2])
                )
            ),
            "raw_metrics": {
                endpoint: _metrics(observed[endpoint], raw[:, index])
                for index, endpoint in enumerate(("IC10", "IC30", "IC50"))
            },
            "projected_metrics": {
                endpoint: _metrics(observed[endpoint], projected[:, index])
                for index, endpoint in enumerate(("IC10", "IC30", "IC50"))
            },
        }
    return result


def _model_card(
    output: Path,
    metrics: pd.DataFrame,
    comparisons: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    rows = []
    for row in metrics.sort_values(["endpoint", "mae"]).itertuples():
        rows.append(
            f"| {row.endpoint} | `{row.surface}` | {row.mae:.4f} | "
            f"{row.strict_baseline_mae:.4f} | {row.historical_baseline_mae:.4f} |"
        )
    endpoint_lines = []
    for endpoint, item in decision["endpoints"].items():
        surface = item["best_receptor_surface_posthoc"]
        receptor = comparisons[endpoint][surface]["vs_equal_capacity_ligand"]
        historical = comparisons[endpoint][surface]["vs_historical_v10_1_oof"]
        endpoint_lines.append(
            f"- **{endpoint}:** best receptor `{surface}`; versus ligand tree "
            f"{receptor['delta_mae_candidate_minus_comparator']:+.4f} "
            f"(95% CI [{receptor['ci95'][0]:+.4f}, {receptor['ci95'][1]:+.4f}]); "
            f"versus historical V10.1 {historical['delta_mae_candidate_minus_comparator']:+.4f} "
            f"(95% CI [{historical['ci95'][0]:+.4f}, {historical['ci95'][1]:+.4f}])."
        )
    content = f"""# hERG V14.2 nonlinear direct-endpoint receptor stress test

## Status

Exploratory 90-compound analysis. Website integration: **false**. Production promotion: **false**.

## Contract

- Strict outer scaffold folds and inner-OOF IC10/IC30/IC50 anchors are inherited from V13.2.
- ExtraTrees regularization is selected only within each outer-training partition.
- Every receptor surface includes an equal-capacity ligand-physicochemical comparator.
- Comparisons use 5,000 scaffold-cluster bootstrap replicates and also report the stronger historical V10.1 OOF reference.
- Historical columns named F649 refer to deposited **S649**; names remain frozen only for reproducibility.
- The 90 labels were open before V14.2, so no positive result would be prospective confirmation.

## Nested outer-fold metrics

| Endpoint | Surface | Candidate MAE | Strict anchor MAE | Historical V10.1 MAE |
|---|---|---:|---:|---:|
{chr(10).join(rows)}

## Best receptor surfaces

{chr(10).join(endpoint_lines)}

## Decision

No receptor result can be integrated unless it beats the equal-capacity ligand tree and the historical endpoint model with an upper 95% MAE-delta bound below zero and at least four of five folds better. The result remains research-only regardless because model family and surfaces were examined after labels opened.
"""
    (output / "MODEL_CARD.md").write_text(content)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    v132_root = args.v132_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    v132_report = _read_json(v132_root / "analysis_report.json", "report_sha256")
    matrix = pd.read_parquet(v132_root / "six_state_analysis_matrix.parquet")
    direct = matrix.loc[matrix.cohort.eq("direct_curve")].reset_index(drop=True)
    if len(direct) != 90 or direct.outer_fold.nunique() != 5:
        raise CampaignError("the frozen direct-endpoint panel changed")
    inner = pd.read_parquet(v132_root / "baselines/direct_common_fold_inner_oof.parquet")
    strict_predictions = pd.read_parquet(v132_root / "strict_nested_predictions.parquet")
    strict = _strict_baseline_table(strict_predictions)
    surfaces = _feature_surfaces(direct)
    predictions, tuning = _nested_predictions(
        direct,
        inner,
        strict,
        surfaces,
        args.n_estimators,
    )
    metrics, comparisons, decision = _score_and_compare(
        predictions,
        surfaces,
        args.bootstrap_replicates,
    )
    coherence = _coherence(predictions)
    _model_card(output, metrics, comparisons, decision)
    prediction_path = output / "predictions/nested_endpoint_predictions.parquet"
    tuning_path = output / "evidence/nested_tuning_evidence.parquet"
    metrics_path = output / "evidence/endpoint_metrics.parquet"
    _parquet(prediction_path, predictions)
    _parquet(tuning_path, tuning)
    _parquet(metrics_path, metrics)
    bootstrap_report = _write_json(
        output / "evidence/paired_scaffold_bootstrap.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "replicates": args.bootstrap_replicates,
            "comparisons": comparisons,
        },
        "report_sha256",
    )
    coherence_report = _write_json(
        output / "evidence/coherence_analysis.json",
        {"schema_version": SCHEMA_VERSION, "created_utc": _utc(), "surfaces": coherence},
        "report_sha256",
    )
    report = _write_json(
        output / "analysis_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete_exploratory_nested_endpoint_receptor_stress_test",
            "dataset": {
                "rows": len(direct),
                "outer_scaffold_folds": int(direct.outer_fold.nunique()),
                "endpoints": [f"IC{value}" for value in ENDPOINTS],
                "receptor_states": 6,
            },
            "evaluation_contract": {
                "outer_split": "frozen five-fold scaffold split",
                "inner_selection": "remaining outer folds with V13.2 inner-OOF endpoint anchors",
                "model_family": "regularized ExtraTrees residual regression",
                "historical_comparator": "V10.1 endpoint-specific OOF",
                "all_labels_open_before_analysis": True,
            },
            "decision": decision,
            "feature_surfaces": {name: list(features) for name, features in surfaces.items()},
            "bootstrap_report_sha256": bootstrap_report["report_sha256"],
            "coherence_report_sha256": coherence_report["report_sha256"],
            "v13_2_input_report_sha256": v132_report["report_sha256"],
            "scientific_scope": {
                "receptor_aware": True,
                "new_docking_performed": False,
                "prospective_confirmation": False,
                "website_default_changed": False,
                "clinical_or_regulatory_use": False,
            },
        },
        "report_sha256",
    )
    artifacts = [
        prediction_path,
        tuning_path,
        metrics_path,
        output / "evidence/paired_scaffold_bootstrap.json",
        output / "evidence/coherence_analysis.json",
        output / "MODEL_CARD.md",
        output / "analysis_report.json",
    ]
    inputs = [
        Path(__file__).resolve(),
        Path(v13.__file__).resolve(),
        v132_root / "analysis_report.json",
        v132_root / "six_state_analysis_matrix.parquet",
        v132_root / "baselines/direct_common_fold_inner_oof.parquet",
        v132_root / "strict_nested_predictions.parquet",
    ]
    manifest = _write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete",
            "inputs": [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}
                for path in inputs
            ],
            "artifacts": [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}
                for path in artifacts
            ],
        },
        "manifest_sha256",
    )
    return _write_json(
        output / "final_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "finished_utc": _utc(),
            "status": "complete",
            "website_integration_supported": decision["website_integration_supported"],
            "production_promotion_supported": decision["production_promotion_supported"],
            "report_sha256": report["report_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "summary_sha256",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v132-root", type=Path, default=DEFAULT_V132)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-estimators", type=int, default=192)
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.n_estimators < 64 or args.bootstrap_replicates < 100:
        print("V14.2 ERROR: require at least 64 trees and 100 bootstrap replicates", file=sys.stderr)
        return 2
    try:
        result = _run(args)
    except CampaignError as exc:
        print(f"V14.2 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
