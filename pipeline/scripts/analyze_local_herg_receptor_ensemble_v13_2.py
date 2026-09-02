#!/usr/bin/env python3
"""Run strict nested six-state analysis for the V13 receptor-aware pilot.

The original V13 residual analysis is retained as a diagnostic, but its base
OOF predictions for meta-training rows are not nested inside each evaluation
fold. V13.2 closes that subtle stacking loophole. Exact IC50 uses V9's stored
inner-OOF honest stacks; direct IC10/IC30/IC50 regenerates inner-OOF endpoint
baselines inside each fixed V12 common scaffold fold. The six receptor states
are then evaluated with small, fixed mechanistic feature sets and a clearly
marked post-hoc F649 diagnostic.
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
import run_local_herg_anchor_refinement_campaign_v12_2 as v122
import run_local_herg_receptor_ensemble_campaign_v13 as v13
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCHEMA_VERSION = "platform-local-herg-receptor-ensemble-v13.2/1.0"
SEED = 20260821
ENDPOINTS = (10, 30, 50)


class CampaignError(RuntimeError):
    """Strict-analysis integrity or execution failure."""


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
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
        "potent_predicted_safe_rate": float(np.mean(prediction[y == 2] == 0)),
    }


def _bootstrap_delta(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    observed: np.ndarray,
    replicates: int,
) -> dict[str, Any]:
    codes, unique = pd.factorize(frame.scaffold_group_id.astype(str), sort=True)
    counts = np.bincount(codes, minlength=len(unique)).astype(float)
    difference = np.abs(observed - candidate) - np.abs(observed - baseline)
    group_difference = np.bincount(codes, weights=difference, minlength=len(unique))
    rng = np.random.default_rng(SEED)
    deltas = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 256):
        stop = min(replicates, start + 256)
        sampled = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        deltas[start:stop] = np.sum(group_difference[sampled], axis=1) / np.sum(
            counts[sampled], axis=1
        )
    return {
        "delta_mae_candidate_minus_baseline": float(np.mean(difference)),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "probability_candidate_better": float(np.mean(deltas < 0)),
        "replicates": replicates,
        "resampling_unit": "scaffold_group_id",
    }


def _feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    states = list(v13.RECEPTOR_IDS)
    sets = {
        "core_state_contrasts": [
            "dock__lowK_minus_highK",
            "dock__e4031_conditioned_minus_apo",
            "dock__core_sd_affinity",
            "dock__core_range_affinity",
        ],
        "core_ensemble_scores": [
            *(f"dock__{state}__affinity" for state in v13.CORE_IDS),
            *(f"dock__{state}__ligand_efficiency" for state in v13.CORE_IDS),
        ],
        "core_pose_stability": [
            column
            for column in frame
            if any(column.startswith(f"dock__{state}__") for state in v13.CORE_IDS)
            and ("pose_count_within_1kcal" in column or "affinity_range" in column)
        ],
        "e4031_state_contacts": [
            column for column in frame if column.startswith("dock__8ZYP__contact_")
        ],
        "sensitivity_affinity": ["dock__8ZYO__affinity", "dock__8ZYQ__affinity"],
        "sensitivity_state_contrasts": [
            "dock__astemizole_minus_apo",
            "dock__pimozide_minus_apo",
            "dock__astemizole_minus_e4031",
            "dock__pimozide_minus_e4031",
            "dock__pimozide_minus_astemizole",
        ],
        "sensitivity_contacts": [
            column
            for column in frame
            if column.startswith(("dock__8ZYO__contact_", "dock__8ZYQ__contact_"))
        ],
        "sensitivity_y652_f656": [
            column
            for column in frame
            if column.startswith(
                (
                    "dock__8ZYO__contact_652",
                    "dock__8ZYO__contact_656",
                    "dock__8ZYQ__contact_652",
                    "dock__8ZYQ__contact_656",
                )
            )
        ],
        "six_state_scores": [
            *(f"dock__{state}__affinity" for state in states),
            *(f"dock__{state}__ligand_efficiency" for state in states),
            "dock__ensemble_sd_affinity",
            "dock__ensemble_range_affinity",
        ],
        "combined_state_contrasts": [
            "dock__lowK_minus_highK",
            "dock__e4031_conditioned_minus_apo",
            "dock__astemizole_minus_apo",
            "dock__pimozide_minus_apo",
            "dock__pimozide_minus_astemizole",
        ],
        "posthoc_astemizole_f649_count": ["dock__8ZYO__contact_649_count"],
    }
    for name, columns in sets.items():
        missing = set(columns) - set(frame)
        if not columns or missing:
            raise CampaignError(f"feature-set contract failed for {name}: {sorted(missing)}")
    return sets


def _six_state_matrix(primary: Path, sensitivity: Path) -> pd.DataFrame:
    selected = pd.read_parquet(primary / "pilot/pilot_selection.parquet")
    raw = pd.concat(
        [
            pd.read_parquet(primary / "docking/docking_results.parquet"),
            pd.read_parquet(sensitivity / "docking_results.parquet"),
        ],
        ignore_index=True,
    )
    if raw.groupby(["ligand_id", "pdb_id"]).size().ne(1).any():
        raise CampaignError("six-state docking rows are not one-to-one")
    features = v13._aggregate_docking(raw, v13.RECEPTOR_IDS)  # noqa: SLF001
    features["dock__astemizole_minus_apo"] = (
        features.dock__8ZYO__affinity - features.dock__8ZYN__affinity
    )
    features["dock__pimozide_minus_apo"] = (
        features.dock__8ZYQ__affinity - features.dock__8ZYN__affinity
    )
    features["dock__astemizole_minus_e4031"] = (
        features.dock__8ZYO__affinity - features.dock__8ZYP__affinity
    )
    features["dock__pimozide_minus_e4031"] = (
        features.dock__8ZYQ__affinity - features.dock__8ZYP__affinity
    )
    features["dock__pimozide_minus_astemizole"] = (
        features.dock__8ZYQ__affinity - features.dock__8ZYO__affinity
    )
    core_scores = features[[f"dock__{state}__affinity" for state in v13.CORE_IDS]].to_numpy(float)
    features["dock__core_sd_affinity"] = np.std(core_scores, axis=1)
    features["dock__core_range_affinity"] = np.max(core_scores, axis=1) - np.min(
        core_scores, axis=1
    )
    matrix = selected.merge(features, on="ligand_id", validate="one_to_one")
    if len(matrix) != len(selected):
        raise CampaignError("six-state feature coverage is incomplete")
    return matrix


def _v9_inner_baselines(repo: Path, exact: pd.DataFrame) -> pd.DataFrame:
    root = repo / "research/local_runs/herg_domain_mixture_campaign_v9/units"
    rows = []
    for outer in range(5):
        bundle = joblib.load(root / f"outer_o{outer}/outer_models.joblib")
        merged: pd.DataFrame | None = None
        for column in bundle["columns"]:
            candidate = column.removeprefix("pred__")
            prediction = pd.read_parquet(
                root / f"inner_o{outer}__{candidate}/inner_oof_predictions.parquet"
            )[["structure_id", "predicted_pic50"]].rename(columns={"predicted_pic50": column})
            merged = (
                prediction
                if merged is None
                else merged.merge(prediction, on="structure_id", validate="one_to_one")
            )
        if merged is None:
            raise CampaignError(f"V9 inner predictions missing for outer fold {outer}")
        merged["inner_baseline"] = bundle["stack"].predict(merged[bundle["columns"]])
        required = exact.loc[exact.outer_fold.ne(outer), "structure_id"]
        subset = merged.loc[merged.structure_id.isin(required), ["structure_id", "inner_baseline"]].copy()
        if len(subset) != len(required):
            raise CampaignError(f"V9 inner baseline coverage failed for outer fold {outer}")
        subset["context_outer_fold"] = outer
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)


def _direct_inner_baselines(
    repo: Path,
    output: Path,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = output / "baselines/direct_common_fold_inner_oof.parquet"
    v122_oof = pd.read_parquet(
        repo
        / "research/local_runs/herg_anchor_refinement_campaign_v12_2/endpoints/common_fold_oof_predictions.parquet"
    )
    labels = pd.read_parquet(
        repo / "research/local_runs/herg_endpoint_receptor_campaign_v12_1/endpoints/coherent_gap_oof.parquet"
    )
    features = pd.read_parquet(
        repo
        / "research/local_runs/herg_endpoint_receptor_campaign_v12_1/prepared/direct_paired_features.parquet"
    )
    frame = labels.merge(features, on="standard_inchi_key", validate="one_to_one")
    schema = json.loads(
        (
            repo
            / "research/local_runs/herg_domain_mixture_campaign_v9/final_model/feature_preprocessing_schema.json"
        ).read_text()
    )
    all_features = [str(value) for value in schema["feature_columns"]]
    rdkit = [column for column in all_features if column.startswith("rdkit2d__")]
    if cache.is_file():
        inner = pd.read_parquet(cache)
        expected = len(frame) * 4
        if len(inner) != expected:
            raise CampaignError("cached direct inner-baseline row count is incompatible")
        return inner, v122_oof
    rows = []
    for outer in range(5):
        outer_fit = frame.loc[frame.outer_fold.ne(outer)].copy().reset_index(drop=True)
        groups = outer_fit.scaffold_group_id.astype(str).to_numpy()
        endpoint_predictions = {endpoint: np.full(len(outer_fit), np.nan) for endpoint in ENDPOINTS}
        for inner_fold, (fit, evaluate) in enumerate(
            GroupKFold(4).split(outer_fit, groups=groups)
        ):
            if set(groups[fit]) & set(groups[evaluate]):
                raise CampaignError("direct inner scaffold leakage")
            for endpoint in ENDPOINTS:
                columns = rdkit if endpoint == 50 else all_features
                model = v122._xgb(SEED + outer * 1_000 + endpoint * 10 + inner_fold, workers)  # noqa: SLF001
                model.fit(
                    v122._safe_numeric(outer_fit.iloc[fit], columns),  # noqa: SLF001
                    outer_fit.iloc[fit][f"observed_pic{endpoint}"].to_numpy(float),
                )
                endpoint_predictions[endpoint][evaluate] = model.predict(
                    v122._safe_numeric(outer_fit.iloc[evaluate], columns)  # noqa: SLF001
                )
        part = outer_fit[["standard_inchi_key"]].copy()
        part["context_outer_fold"] = outer
        for endpoint in ENDPOINTS:
            if not np.isfinite(endpoint_predictions[endpoint]).all():
                raise CampaignError(f"incomplete direct IC{endpoint} inner predictions")
            part[f"inner_baseline_pic{endpoint}"] = endpoint_predictions[endpoint]
        rows.append(part)
        print(json.dumps({"stage": "direct_inner_baseline", "completed_outer": outer}), flush=True)
    inner = pd.concat(rows, ignore_index=True)
    _parquet(cache, inner)
    return inner, v122_oof


def _ridge() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=100.0)),
        ]
    )


def _strict_exact_predictions(
    exact: pd.DataFrame,
    inner: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    regression = {name: np.full(len(exact), np.nan) for name in feature_sets}
    classification = {name: np.full((len(exact), 3), np.nan) for name in feature_sets}
    y_class = v13._tier_index(exact.observed_target.to_numpy(float))  # noqa: SLF001
    for outer in range(5):
        fit = exact.outer_fold.ne(outer).to_numpy()
        evaluate = ~fit
        baseline_map = inner.loc[inner.context_outer_fold.eq(outer)].set_index("structure_id").inner_baseline
        baseline_fit = exact.loc[fit, "structure_id"].map(baseline_map).to_numpy(float)
        if not np.isfinite(baseline_fit).all():
            raise CampaignError(f"exact inner baseline missing in outer fold {outer}")
        for name, columns in feature_sets.items():
            model = _ridge()
            model.fit(
                exact.loc[fit, columns],
                exact.loc[fit, "observed_target"].to_numpy(float) - baseline_fit,
            )
            regression[name][evaluate] = exact.loc[evaluate, "baseline_prediction"].to_numpy(float) + model.predict(
                exact.loc[evaluate, columns]
            )
            class_columns = ["strict_inner_baseline", *columns]
            training = exact.loc[fit, columns].copy()
            training.insert(0, "strict_inner_baseline", baseline_fit)
            evaluation = exact.loc[evaluate, columns].copy()
            evaluation.insert(
                0,
                "strict_inner_baseline",
                exact.loc[evaluate, "baseline_prediction"].to_numpy(float),
            )
            classifier = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.03,
                            class_weight="balanced",
                            max_iter=2_000,
                            random_state=SEED + outer,
                        ),
                    ),
                ]
            )
            classifier.fit(training[class_columns], y_class[fit])
            probabilities = classifier.predict_proba(evaluation[class_columns])
            rows = np.flatnonzero(evaluate)
            classes = classifier.named_steps["model"].classes_.astype(int)
            classification[name][np.ix_(rows, classes)] = probabilities
    for collection in (regression, classification):
        for name, values in collection.items():
            if not np.isfinite(values).all():
                raise CampaignError(f"incomplete strict exact predictions: {name}")
    return regression, classification


def _strict_direct_predictions(
    direct: pd.DataFrame,
    inner: pd.DataFrame,
    common_oof: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[dict[str, dict[int, np.ndarray]], dict[int, np.ndarray]]:
    common = common_oof.set_index("standard_inchi_key")
    baselines = {
        endpoint: direct.standard_inchi_key.map(common[f"independent_pic{endpoint}"]).to_numpy(float)
        for endpoint in ENDPOINTS
    }
    predictions = {
        name: {endpoint: np.full(len(direct), np.nan) for endpoint in ENDPOINTS}
        for name in feature_sets
    }
    for outer in range(5):
        fit = direct.outer_fold.ne(outer).to_numpy()
        evaluate = ~fit
        inner_part = inner.loc[inner.context_outer_fold.eq(outer)].set_index("standard_inchi_key")
        for endpoint in ENDPOINTS:
            baseline_fit = direct.loc[fit, "standard_inchi_key"].map(
                inner_part[f"inner_baseline_pic{endpoint}"]
            ).to_numpy(float)
            if not np.isfinite(baseline_fit).all():
                raise CampaignError(f"direct inner IC{endpoint} baseline missing in outer fold {outer}")
            for name, columns in feature_sets.items():
                model = _ridge()
                model.fit(
                    direct.loc[fit, columns],
                    direct.loc[fit, f"observed_pic{endpoint}"].to_numpy(float) - baseline_fit,
                )
                predictions[name][endpoint][evaluate] = baselines[endpoint][evaluate] + model.predict(
                    direct.loc[evaluate, columns]
                )
    for name in feature_sets:
        for endpoint in ENDPOINTS:
            if not np.isfinite(predictions[name][endpoint]).all():
                raise CampaignError(f"incomplete strict direct predictions: {name} IC{endpoint}")
    return predictions, baselines


def _fold_deltas(
    frame: pd.DataFrame,
    observed: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for fold in sorted(frame.outer_fold.astype(int).unique()):
        keep = frame.outer_fold.astype(int).eq(fold).to_numpy()
        delta = float(
            np.mean(np.abs(observed[keep] - candidate[keep]))
            - np.mean(np.abs(observed[keep] - baseline[keep]))
        )
        rows.append({"outer_fold": int(fold), "delta_mae_candidate_minus_baseline": delta})
    return rows


def _associations(frame: pd.DataFrame, residual: np.ndarray, endpoint: str) -> pd.DataFrame:
    rows = []
    for column in [value for value in frame if value.startswith("dock__")]:
        result = spearmanr(frame[column], residual, nan_policy="omit")
        rows.append(
            {
                "cohort": frame.cohort.iloc[0],
                "endpoint": endpoint,
                "feature": column,
                "spearman_residual": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    count = len(result)
    adjusted = np.minimum.accumulate(
        (result.p_value.to_numpy(float) * count / np.arange(1, count + 1))[::-1]
    )[::-1]
    result["bh_q_value"] = np.minimum(1.0, adjusted)
    return result


def _analyze(
    repo: Path,
    output: Path,
    matrix: pd.DataFrame,
    workers: int,
    bootstrap: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    feature_sets = _feature_sets(matrix)
    exact = matrix.loc[matrix.cohort.eq("exact_ic50")].reset_index(drop=True)
    direct = matrix.loc[matrix.cohort.eq("direct_curve")].reset_index(drop=True)
    exact_inner = _v9_inner_baselines(repo, exact)
    _parquet(output / "baselines/v9_exact_inner_oof.parquet", exact_inner)
    direct_inner, common_oof = _direct_inner_baselines(repo, output, workers)
    exact_predictions, class_predictions = _strict_exact_predictions(exact, exact_inner, feature_sets)
    direct_predictions, direct_baselines = _strict_direct_predictions(
        direct, direct_inner, common_oof, feature_sets
    )
    prediction_rows = []
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "feature_sets": {name: columns for name, columns in feature_sets.items()},
        "exact_ic50": {},
        "classification": {},
        "direct_endpoints": {},
    }
    exact_y = exact.observed_target.to_numpy(float)
    exact_baseline = exact.baseline_prediction.to_numpy(float)
    report["exact_ic50"]["v9_baseline"] = _metrics(exact_y, exact_baseline)
    y_class = v13._tier_index(exact_y)  # noqa: SLF001
    baseline_class = v13._tier_index(exact_baseline)  # noqa: SLF001
    report["classification"]["v9_threshold_baseline"] = _classification_metrics(
        y_class, np.eye(3)[baseline_class]
    )
    for name, prediction in exact_predictions.items():
        bootstrap_result = _bootstrap_delta(exact, exact_baseline, prediction, exact_y, bootstrap)
        fold_deltas = _fold_deltas(exact, exact_y, exact_baseline, prediction)
        report["exact_ic50"][name] = {
            **_metrics(exact_y, prediction),
            "vs_v9_scaffold_bootstrap": bootstrap_result,
            "fold_deltas": fold_deltas,
            "folds_better": int(sum(row["delta_mae_candidate_minus_baseline"] < 0 for row in fold_deltas)),
            "posthoc": name.startswith("posthoc_"),
        }
        report["classification"][name] = {
            **_classification_metrics(y_class, class_predictions[name]),
            "posthoc": name.startswith("posthoc_"),
        }
        for index, row in exact.iterrows():
            prediction_rows.append(
                {
                    "ligand_id": row.ligand_id,
                    "cohort": "exact_ic50",
                    "endpoint": "IC50",
                    "feature_set": name,
                    "outer_fold": int(row.outer_fold),
                    "observed": float(exact_y[index]),
                    "baseline": float(exact_baseline[index]),
                    "prediction": float(prediction[index]),
                }
            )

    direct_associations = []
    for endpoint in ENDPOINTS:
        observed = direct[f"observed_pic{endpoint}"].to_numpy(float)
        baseline = direct_baselines[endpoint]
        historical = direct[f"baseline_predicted_pic{endpoint}"].to_numpy(float)
        report["direct_endpoints"][f"IC{endpoint}"] = {
            "strict_common_fold_baseline": _metrics(observed, baseline),
            "historical_v10_1_oof_reference": _metrics(observed, historical),
            "feature_sets": {},
        }
        direct_associations.append(
            _associations(direct, observed - historical, f"IC{endpoint}_historical_residual")
        )
        for name in feature_sets:
            prediction = direct_predictions[name][endpoint]
            bootstrap_result = _bootstrap_delta(direct, baseline, prediction, observed, bootstrap)
            fold_deltas = _fold_deltas(direct, observed, baseline, prediction)
            report["direct_endpoints"][f"IC{endpoint}"]["feature_sets"][name] = {
                **_metrics(observed, prediction),
                "vs_strict_baseline_scaffold_bootstrap": bootstrap_result,
                "fold_deltas": fold_deltas,
                "folds_better": int(
                    sum(row["delta_mae_candidate_minus_baseline"] < 0 for row in fold_deltas)
                ),
                "posthoc": name.startswith("posthoc_"),
            }
            for index, row in direct.iterrows():
                prediction_rows.append(
                    {
                        "ligand_id": row.ligand_id,
                        "cohort": "direct_curve",
                        "endpoint": f"IC{endpoint}",
                        "feature_set": name,
                        "outer_fold": int(row.outer_fold),
                        "observed": float(observed[index]),
                        "baseline": float(baseline[index]),
                        "prediction": float(prediction[index]),
                    }
                )
    for name in feature_sets:
        raw = np.column_stack([direct_predictions[name][endpoint] for endpoint in ENDPOINTS])
        coherent = np.vstack([v13._project_nonincreasing(row) for row in raw])  # noqa: SLF001
        report["direct_endpoints"].setdefault("coherence", {})[name] = {
            "raw_order_violations": int(
                np.sum((raw[:, 0] < raw[:, 1]) | (raw[:, 1] < raw[:, 2]))
            ),
            "coherent_order_violations": int(
                np.sum((coherent[:, 0] < coherent[:, 1]) | (coherent[:, 1] < coherent[:, 2]))
            ),
            "coherent_metrics": {
                f"IC{endpoint}": _metrics(
                    direct[f"observed_pic{endpoint}"].to_numpy(float), coherent[:, index]
                )
                for index, endpoint in enumerate(ENDPOINTS)
            },
        }

    associations = pd.concat(
        [
            _associations(exact, exact_y - exact_baseline, "IC50_v9_residual"),
            *direct_associations,
        ],
        ignore_index=True,
    )
    promotion_rows = []
    for cohort_key, surfaces in (
        ("exact_ic50", report["exact_ic50"]),
        *(
            (
                f"direct_IC{endpoint}",
                report["direct_endpoints"][f"IC{endpoint}"]["feature_sets"],
            )
            for endpoint in ENDPOINTS
        ),
    ):
        for name, result in surfaces.items():
            if name.endswith("baseline") or name == "v9_baseline" or not isinstance(result, dict):
                continue
            bootstrap_key = (
                "vs_v9_scaffold_bootstrap"
                if "vs_v9_scaffold_bootstrap" in result
                else "vs_strict_baseline_scaffold_bootstrap"
            )
            if bootstrap_key not in result:
                continue
            bound = result[bootstrap_key]
            promoted = bool(
                not result.get("posthoc", False)
                and bound["ci95"][1] < 0
                and result.get("folds_better", 0) >= 4
            )
            promotion_rows.append(
                {
                    "cohort_endpoint": cohort_key,
                    "feature_set": name,
                    "delta_mae": bound["delta_mae_candidate_minus_baseline"],
                    "ci95_low": bound["ci95"][0],
                    "ci95_high": bound["ci95"][1],
                    "folds_better": result.get("folds_better"),
                    "posthoc": result.get("posthoc", False),
                    "promoted": promoted,
                }
            )
    promotion = pd.DataFrame(promotion_rows)
    report["decision"] = {
        "promoted_surfaces": int(promotion.promoted.sum()),
        "expand_same_vina_features": bool(promotion.promoted.any()),
        "rule": "non-posthoc; upper 95% scaffold-bootstrap delta <0; improvement in >=4/5 folds",
        "interpretation": (
            "No receptor-aware surface is promoted; retain ligand-only anchors and treat receptor observables "
            "as mechanistic diagnostics unless new dynamic/receptor physics or external data are introduced."
            if not promotion.promoted.any()
            else "At least one receptor surface meets the internal pilot promotion rule."
        ),
    }
    report["scientific_scope"] = {
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "exact_base_meta_training_strictly_inner_oof": True,
        "direct_base_meta_training_strictly_inner_oof": True,
        "six_rigid_receptor_states": True,
        "external_or_prospective_validation": False,
        "posthoc_feature_is_confirmatory": False,
        "vina_scores_are_binding_free_energies": False,
    }
    return report, pd.DataFrame(prediction_rows), pd.concat([associations, promotion], axis=0, ignore_index=True, sort=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--primary-root",
        type=Path,
        default=Path("research/local_runs/herg_receptor_ensemble_campaign_v13"),
    )
    parser.add_argument(
        "--sensitivity-root",
        type=Path,
        default=Path("research/local_runs/herg_receptor_sensitivity_campaign_v13_1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/local_runs/herg_receptor_strict_analysis_v13_2"),
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser


def _main(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    primary = args.primary_root if args.primary_root.is_absolute() else repo / args.primary_root
    sensitivity = (
        args.sensitivity_root if args.sensitivity_root.is_absolute() else repo / args.sensitivity_root
    )
    output = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    primary, sensitivity, output = primary.resolve(), sensitivity.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    matrix = _six_state_matrix(primary, sensitivity)
    _parquet(output / "six_state_analysis_matrix.parquet", matrix)
    report, predictions, evidence = _analyze(
        repo, output, matrix, args.workers, args.bootstrap_replicates
    )
    report = _json(output / "analysis_report.json", report, "report_sha256")
    _parquet(output / "strict_nested_predictions.parquet", predictions)
    _parquet(output / "association_and_promotion_evidence.parquet", evidence)
    inputs = [
        repo / "pipeline/scripts/analyze_local_herg_receptor_ensemble_v13_2.py",
        primary / "manifest.json",
        sensitivity / "manifest.json",
        repo / "research/local_runs/herg_anchor_refinement_campaign_v12_2/manifest.json",
        repo / "research/local_runs/herg_domain_mixture_campaign_v9/manifest.json",
        repo / "research/local_runs/herg_endpoint_receptor_campaign_v12_1/manifest.json",
    ]
    artifacts = [
        output / "six_state_analysis_matrix.parquet",
        output / "baselines/v9_exact_inner_oof.parquet",
        output / "baselines/direct_common_fold_inner_oof.parquet",
        output / "strict_nested_predictions.parquet",
        output / "association_and_promotion_evidence.parquet",
        output / "analysis_report.json",
    ]
    manifest = _json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete_strict_nested_six_state_analysis",
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
            "selected_ligands": len(matrix),
            "receptor_states": 6,
            "promoted_surfaces": report["decision"]["promoted_surfaces"],
            "expand_same_vina_features": report["decision"]["expand_same_vina_features"],
            "report_sha256": report["report_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "summary_sha256",
    )


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _main(args)
    except (CampaignError, v13.CampaignError, v122.CampaignError) as exc:
        print(f"V13.2 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
