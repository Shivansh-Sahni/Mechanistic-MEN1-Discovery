#!/usr/bin/env python3
"""Run the comprehensive train-only hERG optimization campaign V11.

V11 is a resumable decision campaign, not a single hyperparameter sweep. It
tests model family, representation, measurement target, evidence quality, and
error-specialist hypotheses under fixed nested scaffold splits. All candidate
selection, stacking, residual correction, and uncertainty calibration happen
inside the repository TRAIN partition. Repository validation and test labels
remain sealed.

The primary output is a five-fold nested scaffold OOF estimate over exactly
18,801 structures, paired against the strongest V9 internal anchor. A final
full-TRAIN model bundle is emitted for subsequent prospective evaluation, but
its performance claim is the nested OOF result, never its in-sample fit.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

SCHEMA_VERSION = "platform-local-herg-comprehensive-optimization-v11/1.0"
EXACT_ROWS = 18_801
OUTERS = tuple(range(5))
INNERS = tuple(range(3))
SEED = 20260819
V9_ANCHOR_MAE = 0.4327607533


class CampaignError(RuntimeError):
    """Scientific, leakage, resource, or integrity contract failure."""


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    engine: str
    surface: str
    params: dict[str, Any]
    objective: str = "squared"
    target_mode: str = "canonical"
    weight_mode: str = "uniform"
    training_scope: str = "all_exact"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha(path: Path) -> str:
    return hashlib.sha256(pq.read_schema(path).serialize().to_pybytes()).hexdigest()


def _binding(path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise CampaignError(f"missing {role}: {path}")
    result: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }
    if path.suffix == ".parquet":
        result.update(rows=pq.read_metadata(path).num_rows, arrow_schema_sha256=_schema_sha(path))
    return result


def _verify_binding(binding: dict[str, Any], root: Path | None = None) -> None:
    path = Path(str(binding["path"])).resolve()
    if root is not None and not path.is_relative_to(root.resolve()):
        raise CampaignError(f"artifact escapes output root: {path}")
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
        raise CampaignError(f"artifact missing or size changed: {path}")
    if _sha(path) != binding["sha256"]:
        raise CampaignError(f"artifact hash changed: {path}")
    if path.suffix == ".parquet":
        if pq.read_metadata(path).num_rows != int(binding["rows"]):
            raise CampaignError(f"row count changed: {path}")
        if _schema_sha(path) != binding["arrow_schema_sha256"]:
            raise CampaignError(f"schema changed: {path}")


def _self_hash(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop(key, None)
    result[key] = _digest(result)
    return result


def _atomic_json(path: Path, value: dict[str, Any], key: str) -> dict[str, Any]:
    document = _self_hash(value, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(json.dumps(document, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
    os.replace(temporary, path)
    return document


def _read_json(path: Path, key: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError(f"expected JSON object: {path}")
    if key and value.get(key) != _self_hash(value, key)[key]:
        raise CampaignError(f"self-hash mismatch: {path}")
    return value


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


class _Lock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> _Lock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignError(f"campaign already running: {self.path}") from error
        return self

    def __exit__(self, *_: object) -> None:
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def _load_v9_implementation(repo: Path) -> Any:
    path = repo / "pipeline/scripts/run_local_herg_domain_mixture_campaign_v9.py"
    spec = importlib.util.spec_from_file_location("_herg_v9_bound", path)
    if spec is None or spec.loader is None:
        raise CampaignError(f"cannot load V9 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _candidate_plan(smoke: bool = False) -> list[Candidate]:
    small = 40 if smoke else 700
    anchor_trees = 50 if smoke else 1200
    anchor = {
        "n_estimators": anchor_trees,
        "max_depth": 8,
        "learning_rate": 0.02,
        "min_child_weight": 8.0,
        "subsample": 0.75,
        "colsample_bytree": 0.60,
        "reg_alpha": 0.50,
        "reg_lambda": 5.0,
        "max_bin": 96,
    }

    def xgb(identifier: str, surface: str = "anchor", **changes: Any) -> Candidate:
        params = dict(anchor)
        params.update(changes)
        return Candidate(identifier, "xgboost", surface, params)

    result = [
        xgb("xgb_v9_anchor"),
        xgb("xgb_depth5", max_depth=5),
        xgb("xgb_depth6", max_depth=6),
        xgb("xgb_depth10", max_depth=10, min_child_weight=12.0),
        xgb("xgb_lr012", learning_rate=0.012, n_estimators=80 if smoke else 1800),
        xgb("xgb_lr035", learning_rate=0.035, n_estimators=35 if smoke else 800),
        xgb("xgb_child3", min_child_weight=3.0),
        xgb("xgb_child16", min_child_weight=16.0),
        xgb("xgb_rows09", subsample=0.90),
        xgb("xgb_cols045", colsample_bytree=0.45),
        xgb("xgb_cols08", colsample_bytree=0.80),
        xgb("xgb_reg_strong", max_depth=6, reg_alpha=1.5, reg_lambda=14.0),
        xgb("xgb_reg_weak", reg_alpha=0.05, reg_lambda=1.0),
        xgb("xgb_bin192", max_bin=192, max_depth=6),
        Candidate("xgb_absolute", "xgboost", "anchor", dict(anchor), objective="absolute"),
        xgb("xgb_rdkit2d", surface="rdkit2d", max_depth=6, colsample_bytree=0.80),
        xgb("xgb_morgan", surface="morgan", max_depth=6, colsample_bytree=0.55),
        xgb("xgb_anchor_maccs", surface="anchor_maccs", max_depth=7),
        xgb("xgb_qc_physics", surface="anchor_qc_physics", max_depth=6),
        xgb("xgb_full_ligand", surface="full_ligand", max_depth=5, max_bin=128),
        Candidate(
            "xgb_reliability",
            "xgboost",
            "anchor",
            dict(anchor),
            weight_mode="reliability",
        ),
        Candidate(
            "xgb_quality_weighted",
            "xgboost",
            "anchor",
            dict(anchor),
            target_mode="quality_weighted",
            weight_mode="quality",
        ),
        Candidate(
            "xgb_hierarchical",
            "xgboost",
            "anchor",
            dict(anchor),
            target_mode="hierarchical",
            weight_mode="reliability",
        ),
        Candidate(
            "xgb_high_quality_only",
            "xgboost",
            "anchor",
            dict(anchor),
            target_mode="quality_weighted",
            weight_mode="quality",
            training_scope="high_quality",
        ),
        Candidate(
            "xgb_patch_functional_only",
            "xgboost",
            "anchor",
            dict(anchor),
            target_mode="quality_weighted",
            weight_mode="quality",
            training_scope="patch_or_functional",
        ),
        Candidate("xgb_tail", "xgboost", "anchor", dict(anchor), weight_mode="potency_tail"),
        Candidate("xgb_cliff", "xgboost", "anchor", dict(anchor), weight_mode="cliff_risk"),
        Candidate(
            "xgb_heavy_flexible",
            "xgboost",
            "anchor_qc_physics",
            dict(anchor),
            weight_mode="heavy_flexible",
        ),
        Candidate(
            "xgb_combined_weight",
            "xgboost",
            "anchor",
            dict(anchor),
            target_mode="quality_weighted",
            weight_mode="combined",
        ),
    ]

    def lgb(identifier: str, objective: str = "squared", **changes: Any) -> Candidate:
        params = {
            "n_estimators": small,
            "num_leaves": 31,
            "learning_rate": 0.025,
            "min_child_samples": 30,
            "subsample": 0.82,
            "colsample_bytree": 0.70,
            "reg_alpha": 0.50,
            "reg_lambda": 5.0,
            "max_bin": 127,
        }
        params.update(changes)
        return Candidate(identifier, "lightgbm", "anchor", params, objective=objective)

    result.extend(
        [
            lgb("lgb_l2"),
            lgb("lgb_l1", objective="absolute"),
            lgb("lgb_huber", objective="huber", alpha=0.85),
            lgb("lgb_leaves15", num_leaves=15, min_child_samples=20),
            lgb("lgb_leaves63", num_leaves=63, min_child_samples=45),
            lgb("lgb_regularized", num_leaves=23, reg_alpha=2.0, reg_lambda=14.0),
            Candidate(
                "lgb_quality_weighted",
                "lightgbm",
                "anchor",
                lgb("temporary").params,
                target_mode="quality_weighted",
                weight_mode="quality",
            ),
        ]
    )
    result.extend(
        [
            Candidate(
                "et_rdkit2d",
                "extratrees",
                "rdkit2d",
                {
                    "n_estimators": 60 if smoke else 700,
                    "max_features": 0.75,
                    "min_samples_leaf": 2,
                    "max_depth": None,
                },
            ),
            Candidate(
                "et_rdkit2d_leaf4",
                "extratrees",
                "rdkit2d",
                {
                    "n_estimators": 60 if smoke else 700,
                    "max_features": 0.90,
                    "min_samples_leaf": 4,
                    "max_depth": None,
                },
            ),
        ]
    )
    if smoke:
        required = {"xgb_v9_anchor", "lgb_l2", "et_rdkit2d", "xgb_quality_weighted"}
        return [row for row in result if row.candidate_id in required]
    identifiers = [row.candidate_id for row in result]
    if len(identifiers) != len(set(identifiers)):
        raise CampaignError("candidate identifiers are not unique")
    if len({_digest(asdict(row)) for row in result}) != len(result):
        raise CampaignError("candidate specifications are not materially unique")
    return result


def _surface_columns(blocks: dict[str, list[str]]) -> dict[str, list[str]]:
    rdkit = blocks["rdkit2d"]
    morgan = blocks["morgan"]
    anchor = rdkit + morgan
    qc_physics = blocks["polarity_charge_internal_contacts"] + blocks["energy_flexibility"]
    qc_physics += blocks["selected_interactions"] + blocks["new3d_stable_misc"]
    full = qc_physics + blocks["shape"] + blocks["autocorr3d"] + blocks["whim"]
    surfaces = {
        "rdkit2d": rdkit,
        "morgan": morgan,
        "anchor": anchor,
        "anchor_maccs": anchor + blocks["maccs"],
        "anchor_qc_physics": anchor + qc_physics,
        "full_ligand": anchor + full,
    }
    return {key: list(dict.fromkeys(value)) for key, value in surfaces.items()}


def _safe_metrics(observed: Iterable[float], predicted: Iterable[float]) -> dict[str, Any]:
    y = np.asarray(observed, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if len(y) == 0 or y.shape != p.shape or not np.isfinite(p).all():
        raise CampaignError("invalid metric arrays")
    absolute = np.abs(y - p)
    correlation = spearmanr(y, p).statistic if len(y) > 1 else math.nan
    tail = absolute[(y < 4.5) | (y >= 6.5)]
    return {
        "n": int(len(y)),
        "mae": float(np.mean(absolute)),
        "rmse": float(math.sqrt(mean_squared_error(y, p))),
        "median_absolute_error": float(np.median(absolute)),
        "spearman": None if not math.isfinite(float(correlation)) else float(correlation),
        "fraction_within_0p5": float(np.mean(absolute <= 0.5)),
        "fraction_within_1p0": float(np.mean(absolute <= 1.0)),
        "tail_mae": None if len(tail) == 0 else float(np.mean(tail)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
    }


def _quality_score(frame: pd.DataFrame) -> np.ndarray:
    modality = frame.measurement_modality.astype("string").fillna("unresolved")
    base = (
        modality.map(
            {
                "patch_clamp": 1.60,
                "functional_electrophysiology": 1.50,
                "ion_flux": 1.30,
                "thallium_flux": 1.10,
                "functional_unspecified": 1.00,
                "radioligand": 0.85,
                "binding_unspecified": 0.75,
                "unresolved": 0.60,
            }
        )
        .fillna(0.65)
        .to_numpy(float)
    )
    protocol = pd.to_numeric(frame.protocol_completeness_score, errors="coerce").fillna(0).to_numpy(float)
    automation = frame.automation_class.astype("string").fillna("unresolved")
    auto_factor = automation.map({"manual": 1.12, "automated": 1.00, "unresolved": 0.92}).fillna(0.92)
    result = base * (1.0 + 0.07 * np.clip(protocol, 0, 6)) * auto_factor.to_numpy(float)
    if "v1_5_conflict_review_structure" in frame:
        result *= np.where(frame.v1_5_conflict_review_structure.to_numpy(bool), 0.55, 1.0)
    if "evaluation_or_lineage_leakage_caution" in frame:
        result *= np.where(frame.evaluation_or_lineage_leakage_caution.to_numpy(bool), 0.70, 1.0)
    return np.clip(result, 0.15, 2.5)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    threshold = 0.5 * float(weights.sum())
    return float(values[np.searchsorted(np.cumsum(weights), threshold, side="left")])


def _scope_mask(frame: pd.DataFrame, scope: str) -> np.ndarray:
    modality = frame.measurement_modality.astype("string").fillna("unresolved")
    if scope == "all_exact":
        return np.ones(len(frame), dtype=bool)
    if scope == "high_quality":
        allowed = {"patch_clamp", "functional_electrophysiology", "ion_flux", "thallium_flux"}
        return modality.isin(allowed).to_numpy() | (
            pd.to_numeric(frame.protocol_completeness_score, errors="coerce").fillna(0).to_numpy(float) >= 2
        )
    if scope == "patch_or_functional":
        return modality.isin(
            {"patch_clamp", "functional_electrophysiology", "ion_flux", "thallium_flux"}
        ).to_numpy()
    raise CampaignError(f"unsupported training scope: {scope}")


def _minimum_training_structures(training_scope: str) -> int:
    """Return the prespecified support floor for a training scope."""
    return 250 if training_scope == "patch_or_functional" else 500


def _candidate_selection_eligible(candidate: Candidate) -> bool:
    """Keep the small quality-only surface exploratory, never headline-selectable."""
    return candidate.training_scope != "patch_or_functional"


def _measurement_targets(
    v9: Any,
    observations: pd.DataFrame,
    fit_ids: set[str],
    candidate: Candidate,
) -> pd.DataFrame:
    part = observations.loc[observations.structure_id.astype(str).isin(fit_ids)].copy()
    part = part.loc[_scope_mask(part, candidate.training_scope)].copy()
    if part.empty:
        raise CampaignError(f"no observations for scope {candidate.training_scope}")
    if candidate.target_mode == "hierarchical":
        return v9._measurement_targets(part, set(part.structure_id.astype(str)), "hierarchical")
    part["quality_score"] = _quality_score(part)
    rows: list[dict[str, Any]] = []
    for sid, group in part.groupby("structure_id", observed=True, sort=False):
        values = group.potency_pic50_point.to_numpy(float)
        quality = group.quality_score.to_numpy(float)
        target = (
            _weighted_median(values, quality)
            if candidate.target_mode == "quality_weighted"
            else float(np.median(values))
        )
        spread = float(np.ptp(values)) if len(values) > 1 else 0.0
        reliability = math.sqrt(len(values)) / (1.0 + spread)
        rows.append(
            {
                "structure_id": str(sid),
                "training_target": target,
                "replicate_range": spread,
                "observation_count": int(len(values)),
                "reliability_weight": float(np.clip(reliability, 0.15, 3.0)),
                "quality_weight": float(np.clip(np.mean(quality), 0.15, 2.5)),
            }
        )
    result = pd.DataFrame(rows)
    minimum = _minimum_training_structures(candidate.training_scope)
    if len(result) < minimum:
        raise CampaignError(
            f"training scope {candidate.training_scope} has only {len(result)} structures; "
            f"minimum is {minimum}"
        )
    return result


def _qc_clean_matrix(v9: Any, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = v9._clean_matrix(frame, columns)
    failed_name = "new3d__feature_failed_indicator"
    excluded_name = "new3d__physics_qc_excluded_indicator"
    mask = np.zeros(len(frame), dtype=bool)
    for name in (failed_name, excluded_name):
        if name in frame:
            mask |= pd.to_numeric(frame[name], errors="coerce").fillna(1).to_numpy(float) > 0.5
    physics_columns = [
        name
        for name in columns
        if name.startswith("new3d__")
        or name.startswith("interaction__")
        or name.startswith("v5interaction__")
    ]
    if physics_columns and mask.any():
        result.loc[mask, physics_columns] = np.nan
    for name in columns:
        if "energy_range" in name and name in result:
            values = result[name].to_numpy(float, copy=True)
            values[(values < 0) | (values > 5_000)] = np.nan
            result[name] = values
    return result


def _training_weights(
    joined: pd.DataFrame,
    mmp: pd.DataFrame,
    fit_ids: set[str],
    candidate: Candidate,
) -> np.ndarray:
    """Apply the same prespecified structure weighting during CV and final refits."""
    y = joined.training_target.to_numpy(float)
    weights = np.ones(len(joined), dtype=float)
    if candidate.weight_mode in {"reliability", "combined"}:
        weights *= joined.reliability_weight.to_numpy(float)
    if candidate.weight_mode in {"quality", "combined"}:
        weights *= joined.quality_weight.to_numpy(float)
    if candidate.weight_mode in {"potency_tail", "combined"}:
        weights *= 1.0 + 1.25 * ((y < 4.5) | (y >= 6.5))
    if candidate.weight_mode == "heavy_flexible":
        mw = joined["rdkit2d__MolWt"].fillna(0).to_numpy(float)
        rotors = joined["rdkit2d__NumRotatableBonds"].fillna(0).to_numpy(float)
        weights *= 1.0 + 0.75 * ((mw >= 500) | (rotors >= 8))
    if candidate.weight_mode == "cliff_risk":
        contained = mmp.loc[
            mmp.structure_id_a.astype(str).isin(fit_ids)
            & mmp.structure_id_b.astype(str).isin(fit_ids)
            & mmp.activity_cliff_ge_1_pic50
        ]
        cliff_ids = set(contained.structure_id_a.astype(str)) | set(contained.structure_id_b.astype(str))
        weights *= 1.0 + joined.structure_id.astype(str).isin(cliff_ids).to_numpy(float)
    return np.clip(weights / np.mean(weights), 0.15, 6.0)


def _fit_predict(
    v9: Any,
    matrix: pd.DataFrame,
    observations: pd.DataFrame,
    mmp: pd.DataFrame,
    columns: list[str],
    fit_ids: set[str],
    eval_ids: set[str],
    candidate: Candidate,
    workers: int,
    seed: int,
) -> tuple[Any, pd.DataFrame, float, dict[str, Any]]:
    fit = matrix.loc[matrix.structure_id.astype(str).isin(fit_ids)].copy()
    evaluation = matrix.loc[matrix.structure_id.astype(str).isin(eval_ids)].copy()
    if set(fit.scaffold_group_id.astype(str)) & set(evaluation.scaffold_group_id.astype(str)):
        raise CampaignError("scaffold leakage in model fit")
    targets = _measurement_targets(v9, observations, fit_ids, candidate)
    joined = fit.merge(targets, on="structure_id", validate="one_to_one")
    y = joined.training_target.to_numpy(float)
    weights = _training_weights(joined, mmp, fit_ids, candidate)
    model = v9._new_model(candidate, workers, seed)
    started = time.monotonic()
    fit_kwargs = (
        {"model__sample_weight": weights} if isinstance(model, Pipeline) else {"sample_weight": weights}
    )
    model.fit(_qc_clean_matrix(v9, joined, columns), y, **fit_kwargs)
    prediction = np.asarray(model.predict(_qc_clean_matrix(v9, evaluation, columns)), dtype=float)
    elapsed = time.monotonic() - started
    result = evaluation[["structure_id", "scaffold_group_id", "target_pic50"]].copy()
    result = result.rename(columns={"target_pic50": "observed_pic50"})
    result["predicted_pic50"] = prediction
    training_summary = {
        "fit_structures": int(len(joined)),
        "available_fit_structures": int(len(fit)),
        "training_scope": candidate.training_scope,
        "target_mode": candidate.target_mode,
        "weight_mode": candidate.weight_mode,
        "mean_weight": float(np.mean(weights)),
    }
    return model, result, elapsed, training_summary


def _unit_document(
    directory: Path,
    unit_id: str,
    spec: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return _atomic_json(
        directory / "unit.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "unit_id": unit_id,
            "unit_spec": spec,
            "unit_spec_sha256": _digest(spec),
            "metrics": metrics,
            "artifacts": artifacts,
            "scientific_scope": {
                "source_partition": "train",
                "fixed_nested_scaffold_folds": True,
                "selection_inside_outer_training_only": True,
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
                "external_or_prospective_validation": False,
            },
        },
        "unit_json_sha256",
    )


def _existing(directory: Path, spec: dict[str, Any], output: Path) -> dict[str, Any] | None:
    path = directory / "unit.json"
    if not path.is_file():
        return None
    try:
        value = _read_json(path, "unit_json_sha256")
        if value.get("status") != "passed" or value.get("unit_spec") != spec:
            return None
        for binding in value["artifacts"]:
            _verify_binding(binding, output)
        return value
    except Exception:
        return None


def _artifact(unit: dict[str, Any], role: str) -> Path:
    matches = [Path(row["path"]) for row in unit["artifacts"] if row["role"] == role]
    if len(matches) != 1:
        raise CampaignError(f"unit {unit['unit_id']} lacks unique role {role}")
    return matches[0]


def _prepare(repo: Path, source: Path, output: Path) -> dict[str, Any]:
    target = output / "prepared/validation.json"
    if target.is_file():
        value = _read_json(target, "validation_sha256")
        for row in value["inputs"]:
            _verify_binding(row)
        for row in value["artifacts"]:
            _verify_binding(row, output)
        return value
    source_validation = _read_json(source / "prepared/validation.json", "validation_sha256")
    if source_validation.get("status") != "passed":
        raise CampaignError("V9 prepared validation did not pass")
    required = {
        "training_matrix": source / "prepared/training_matrix.parquet",
        "splits": source / "prepared/fixed_nested_scaffold_splits.parquet",
        "observations": source / "prepared/exact_train_observations.parquet",
        "mmp": source / "prepared/training_mmp_effects.parquet",
        "ad": source / "prepared/outer_applicability_domain.parquet",
        "blocks": source / "prepared/feature_blocks.json",
        "v9_oof": source / "analysis/nested_oof_predictions.parquet",
    }
    for path in required.values():
        if not path.is_file():
            raise CampaignError(f"missing V9 source artifact: {path}")
    matrix = pd.read_parquet(required["training_matrix"])
    splits = pd.read_parquet(required["splits"])
    observations = pd.read_parquet(required["observations"])
    v9_oof = pd.read_parquet(required["v9_oof"])
    ids = set(matrix.structure_id.astype(str))
    if len(matrix) != EXACT_ROWS or matrix.structure_id.duplicated().any():
        raise CampaignError("matrix is not 18,801 unique structures")
    if set(observations.structure_id.astype(str)) != ids:
        raise CampaignError("observation identities do not match matrix")
    if len(splits) != EXACT_ROWS * 5:
        raise CampaignError("nested split registry has wrong size")
    for outer in OUTERS:
        part = splits.loc[splits.outer_fold.eq(outer)]
        if set(part.structure_id.astype(str)) != ids or part.structure_id.duplicated().any():
            raise CampaignError(f"outer {outer} split identity coverage failed")
        fit_scaffolds = set(part.loc[part.outer_role.eq("fit"), "scaffold_group_id"].astype(str))
        held_scaffolds = set(part.loc[part.outer_role.eq("heldout"), "scaffold_group_id"].astype(str))
        if fit_scaffolds & held_scaffolds:
            raise CampaignError(f"outer {outer} scaffold leakage")
    if len(v9_oof) != EXACT_ROWS or set(v9_oof.structure_id.astype(str)) != ids:
        raise CampaignError("V9 anchor OOF coverage failed")
    anchor_columns = [name for name in ("pred__honest_stack", "predicted_pic50") if name in v9_oof]
    if not anchor_columns:
        raise CampaignError("V9 anchor prediction column absent")
    anchor = v9_oof[["structure_id", anchor_columns[0]]].rename(
        columns={anchor_columns[0]: "v9_predicted_pic50"}
    )
    prepared = output / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_matrix": prepared / "training_matrix.parquet",
        "splits": prepared / "fixed_nested_scaffold_splits.parquet",
        "observations": prepared / "exact_train_observations.parquet",
        "mmp": prepared / "training_mmp_effects.parquet",
        "ad": prepared / "outer_applicability_domain.parquet",
        "v9_anchor": prepared / "v9_anchor_oof.parquet",
        "blocks": prepared / "feature_blocks.json",
    }
    for role in ("training_matrix", "splits", "observations", "mmp", "ad"):
        _atomic_parquet(paths[role], pd.read_parquet(required[role]))
    _atomic_parquet(paths["v9_anchor"], anchor)
    blocks = _read_json(required["blocks"], "feature_blocks_sha256")
    _atomic_json(
        paths["blocks"],
        {"schema_version": SCHEMA_VERSION, "blocks": blocks["blocks"]},
        "feature_blocks_sha256",
    )
    inputs = [_binding(path, f"source_{role}") for role, path in required.items()]
    inputs += [
        _binding(Path(__file__).resolve(), "v11_implementation"),
        _binding(
            repo / "pipeline/scripts/run_local_herg_domain_mixture_campaign_v9.py", "v9_model_implementation"
        ),
    ]
    artifacts = [_binding(path, role) for role, path in paths.items()]
    return _atomic_json(
        target,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "exact_train_structures": EXACT_ROWS,
            "outer_folds": 5,
            "inner_folds": 3,
            "v9_anchor_mae": V9_ANCHOR_MAE,
            "source_partition": "train",
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "inputs": inputs,
            "artifacts": artifacts,
        },
        "validation_sha256",
    )


def _load(
    output: Path,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]
]:
    validation = _read_json(output / "prepared/validation.json", "validation_sha256")
    for row in validation["artifacts"]:
        _verify_binding(row, output)
    blocks = _read_json(output / "prepared/feature_blocks.json", "feature_blocks_sha256")["blocks"]
    return (
        pd.read_parquet(output / "prepared/training_matrix.parquet"),
        pd.read_parquet(output / "prepared/fixed_nested_scaffold_splits.parquet"),
        pd.read_parquet(output / "prepared/exact_train_observations.parquet"),
        pd.read_parquet(output / "prepared/training_mmp_effects.parquet"),
        pd.read_parquet(output / "prepared/outer_applicability_domain.parquet"),
        pd.read_parquet(output / "prepared/v9_anchor_oof.parquet"),
        blocks,
    )


def _fit_unit(
    v9: Any,
    output: Path,
    matrix: pd.DataFrame,
    observations: pd.DataFrame,
    mmp: pd.DataFrame,
    surfaces: dict[str, list[str]],
    candidate: Candidate,
    fit_ids: set[str],
    eval_ids: set[str],
    unit_id: str,
    operation: str,
    outer: int,
    inner: int,
    workers: int,
) -> dict[str, Any]:
    directory = output / "units" / unit_id
    spec = {
        "operation": operation,
        "outer_fold": outer,
        "inner_fold": inner,
        "candidate": asdict(candidate),
        "fit_identity_sha256": _digest(sorted(fit_ids)),
        "evaluation_identity_sha256": _digest(sorted(eval_ids)),
        "feature_count": len(surfaces[candidate.surface]),
        "selection_eligible": _candidate_selection_eligible(candidate),
        "analysis_role": (
            "candidate_model_selection"
            if _candidate_selection_eligible(candidate)
            else "exploratory_quality_sensitivity"
        ),
    }
    existing = _existing(directory, spec, output)
    if existing is not None:
        return existing
    _model, predictions, elapsed, training_summary = _fit_predict(
        v9,
        matrix,
        observations,
        mmp,
        surfaces[candidate.surface],
        fit_ids,
        eval_ids,
        candidate,
        workers,
        SEED + outer * 10_000 + inner * 100 + int(_digest(asdict(candidate))[:4], 16),
    )
    predictions["outer_fold"] = outer
    predictions["inner_fold"] = inner
    path = directory / "predictions.parquet"
    _atomic_parquet(path, predictions)
    metrics = _safe_metrics(predictions.observed_pic50, predictions.predicted_pic50)
    metrics.update(fit_elapsed_seconds=elapsed, training_summary=training_summary)
    return _unit_document(directory, unit_id, spec, metrics, [_binding(path, "predictions")])


def _promotion_ids(units: list[dict[str, Any]], candidates: list[Candidate], maximum: int) -> list[str]:
    lookup = {row.candidate_id: row for row in candidates if _candidate_selection_eligible(row)}
    ranked = sorted(
        (row for row in units if row["unit_spec"]["candidate"]["candidate_id"] in lookup),
        key=lambda row: float(row["metrics"]["mae"]),
    )
    selected = [row["unit_spec"]["candidate"]["candidate_id"] for row in ranked[:maximum]]
    eligible_candidates = list(lookup.values())
    for attribute in ("engine", "surface", "target_mode", "weight_mode", "training_scope"):
        values = sorted({getattr(row, attribute) for row in eligible_candidates})
        for value in values:
            eligible = [
                row
                for row in ranked
                if getattr(lookup[row["unit_spec"]["candidate"]["candidate_id"]], attribute) == value
            ]
            if eligible:
                selected.append(eligible[0]["unit_spec"]["candidate"]["candidate_id"])
    anchor = ["xgb_v9_anchor", "xgb_quality_weighted", "xgb_qc_physics", "lgb_l2", "et_rdkit2d"]
    selected.extend(candidate_id for candidate_id in anchor if candidate_id in lookup)
    return list(dict.fromkeys(selected))[: maximum + 8]


def _merge_predictions(units: list[dict[str, Any]]) -> pd.DataFrame:
    grouped: dict[str, list[pd.DataFrame]] = {}
    for unit in units:
        cid = unit["unit_spec"]["candidate"]["candidate_id"]
        grouped.setdefault(cid, []).append(pd.read_parquet(_artifact(unit, "predictions")))
    result: pd.DataFrame | None = None
    for cid, frames in grouped.items():
        frame = pd.concat(frames, ignore_index=True)
        if frame.structure_id.duplicated().any() or set(frame.inner_fold) != set(INNERS):
            raise CampaignError(f"candidate {cid} lacks disjoint three-fold inner OOF predictions")
        frame = frame[
            ["structure_id", "scaffold_group_id", "observed_pic50", "inner_fold", "predicted_pic50"]
        ]
        frame = frame.rename(columns={"predicted_pic50": f"pred__{cid}"})
        keys = ["structure_id", "scaffold_group_id", "observed_pic50", "inner_fold"]
        result = frame if result is None else result.merge(frame, on=keys, validate="one_to_one")
    if result is None:
        raise CampaignError("no predictions to merge")
    return result


def _gate_features(matrix: pd.DataFrame, ids: pd.Series, prediction_frame: pd.DataFrame) -> pd.DataFrame:
    names = [
        "rdkit2d__MolWt",
        "rdkit2d__MolLogP",
        "rdkit2d__TPSA",
        "rdkit2d__NumRotatableBonds",
        "rdkit2d__HeavyAtomCount",
        "rdkit2d__RingCount",
        "new3d__formal_charge",
        "new3d__feature_failed_indicator",
        "new3d__physics_qc_excluded_indicator",
    ]
    available = [name for name in names if name in matrix]
    indexed = matrix.set_index("structure_id")
    result = indexed.loc[ids.astype(str), available].reset_index(drop=True).astype(float)
    predictions = prediction_frame.filter(regex=r"^pred__").reset_index(drop=True)
    result["prediction_mean"] = predictions.mean(axis=1)
    result["prediction_sd"] = predictions.std(axis=1)
    result["prediction_min"] = predictions.min(axis=1)
    result["prediction_max"] = predictions.max(axis=1)
    return result


def _crossfit_meta(
    matrix: pd.DataFrame,
    inner: pd.DataFrame,
    workers: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [name for name in inner if name.startswith("pred__")]
    if len(columns) < 2:
        raise CampaignError("meta learning requires at least two base candidates")
    result = inner[["structure_id", "scaffold_group_id", "observed_pic50", "inner_fold"]].copy()
    for name in ("mean", "ridge", "positive"):
        result[f"meta__{name}"] = np.nan
    result["meta__mean"] = inner[columns].mean(axis=1)
    for fold in INNERS:
        train_mask = inner.inner_fold.ne(fold)
        eval_mask = inner.inner_fold.eq(fold)
        x_train = inner.loc[train_mask, columns]
        y_train = inner.loc[train_mask, "observed_pic50"]
        x_eval = inner.loc[eval_mask, columns]
        ridge = RidgeCV(alphas=np.logspace(-5, 3, 32)).fit(x_train, y_train)
        positive = LinearRegression(positive=True).fit(x_train, y_train)
        result.loc[eval_mask, "meta__ridge"] = ridge.predict(x_eval)
        result.loc[eval_mask, "meta__positive"] = positive.predict(x_eval)
    base_metrics = {
        name.removeprefix("meta__"): _safe_metrics(result.observed_pic50, result[name])
        for name in ("meta__mean", "meta__ridge", "meta__positive")
    }
    base_name = min(base_metrics, key=lambda name: float(base_metrics[name]["mae"]))
    result["meta__selected_base"] = result[f"meta__{base_name}"]
    result["meta__residual_corrected"] = np.nan
    gate = _gate_features(matrix, result.structure_id, inner)
    residual = result.observed_pic50 - result.meta__selected_base
    for fold in INNERS:
        train_mask = result.inner_fold.ne(fold)
        eval_mask = result.inner_fold.eq(fold)
        correction = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=250,
                        min_samples_leaf=25,
                        max_features=0.75,
                        n_jobs=workers,
                        random_state=SEED + fold,
                    ),
                ),
            ]
        )
        correction.fit(gate.loc[train_mask], residual.loc[train_mask])
        delta = np.clip(correction.predict(gate.loc[eval_mask]), -0.75, 0.75)
        result.loc[eval_mask, "meta__residual_corrected"] = (
            result.loc[eval_mask, "meta__selected_base"].to_numpy(float) + delta
        )
    corrected = _safe_metrics(result.observed_pic50, result.meta__residual_corrected)
    selected_name = (
        "residual_corrected" if corrected["mae"] + 0.001 < base_metrics[base_name]["mae"] else base_name
    )
    result["meta__selected"] = (
        result.meta__residual_corrected
        if selected_name == "residual_corrected"
        else result[f"meta__{base_name}"]
    )
    metrics = {
        "base_methods": base_metrics,
        "residual_corrected": corrected,
        "selected_method": selected_name,
        "selected_base_method": base_name,
        "selected_metrics": _safe_metrics(result.observed_pic50, result.meta__selected),
        "candidate_columns": columns,
    }
    return result, metrics


def _fit_meta_for_outer(
    matrix: pd.DataFrame,
    inner: pd.DataFrame,
    outer: pd.DataFrame,
    meta_metrics: dict[str, Any],
    workers: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    columns = list(meta_metrics["candidate_columns"])
    base_name = str(meta_metrics["selected_base_method"])
    if base_name == "mean":
        base_prediction = outer[columns].mean(axis=1).to_numpy(float)
        base_model: Any = {"method": "mean"}
    else:
        model: Any = (
            RidgeCV(alphas=np.logspace(-5, 3, 32))
            if base_name == "ridge"
            else LinearRegression(positive=True)
        )
        model.fit(inner[columns], inner.observed_pic50)
        base_prediction = np.asarray(model.predict(outer[columns]), dtype=float)
        base_model = model
    bundle: dict[str, Any] = {"base_method": base_name, "base_model": base_model, "columns": columns}
    if meta_metrics["selected_method"] != "residual_corrected":
        return base_prediction, bundle
    inner_crossfit, _ = _crossfit_meta(matrix, inner, workers)
    residual = inner_crossfit.observed_pic50 - inner_crossfit.meta__selected_base
    gate_inner = _gate_features(matrix, inner.structure_id, inner)
    gate_outer = _gate_features(matrix, outer.structure_id, outer)
    correction = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=400,
                    min_samples_leaf=25,
                    max_features=0.75,
                    n_jobs=workers,
                    random_state=SEED + 77,
                ),
            ),
        ]
    )
    correction.fit(gate_inner, residual)
    delta = np.clip(correction.predict(gate_outer), -0.75, 0.75)
    bundle["residual_model"] = correction
    return base_prediction + delta, bundle


def _run_outer(
    v9: Any,
    output: Path,
    matrix: pd.DataFrame,
    splits: pd.DataFrame,
    observations: pd.DataFrame,
    mmp: pd.DataFrame,
    ad: pd.DataFrame,
    surfaces: dict[str, list[str]],
    candidates: list[Candidate],
    outer: int,
    workers: int,
    screen_maximum: int,
    finalist_maximum: int,
) -> dict[str, Any]:
    registry = splits.loc[splits.outer_fold.eq(outer)].copy()
    fit_registry = registry.loc[registry.outer_role.eq("fit")]
    screen_units: list[dict[str, Any]] = []
    screen_inner = 0
    screen_fit = set(fit_registry.loc[fit_registry.inner_fold.ne(screen_inner), "structure_id"].astype(str))
    screen_eval = set(fit_registry.loc[fit_registry.inner_fold.eq(screen_inner), "structure_id"].astype(str))
    for index, candidate in enumerate(candidates):
        unit = _fit_unit(
            v9,
            output,
            matrix,
            observations,
            mmp,
            surfaces,
            candidate,
            screen_fit,
            screen_eval,
            f"screen_o{outer}__{candidate.candidate_id}",
            "successive_halving_screen",
            outer,
            screen_inner,
            workers,
        )
        screen_units.append(unit)
        print(
            json.dumps(
                {
                    "stage": "screen",
                    "outer": outer,
                    "completed": index + 1,
                    "total": len(candidates),
                    "candidate": candidate.candidate_id,
                    "mae": unit["metrics"]["mae"],
                }
            ),
            flush=True,
        )
    promoted_ids = _promotion_ids(screen_units, candidates, screen_maximum)
    lookup = {row.candidate_id: row for row in candidates}
    confirm_units: dict[str, list[dict[str, Any]]] = {}
    for cid in promoted_ids:
        candidate = lookup[cid]
        units = [next(row for row in screen_units if row["unit_spec"]["candidate"]["candidate_id"] == cid)]
        for inner in (1, 2):
            fit_ids = set(fit_registry.loc[fit_registry.inner_fold.ne(inner), "structure_id"].astype(str))
            eval_ids = set(fit_registry.loc[fit_registry.inner_fold.eq(inner), "structure_id"].astype(str))
            units.append(
                _fit_unit(
                    v9,
                    output,
                    matrix,
                    observations,
                    mmp,
                    surfaces,
                    candidate,
                    fit_ids,
                    eval_ids,
                    f"confirm_o{outer}_i{inner}__{cid}",
                    "successive_halving_confirmation",
                    outer,
                    inner,
                    workers,
                )
            )
        confirm_units[cid] = units
    ranked = sorted(
        promoted_ids,
        key=lambda cid: float(np.mean([row["metrics"]["mae"] for row in confirm_units[cid]])),
    )
    finalists = ranked[:finalist_maximum]
    for engine in sorted({lookup[cid].engine for cid in promoted_ids}):
        eligible = [cid for cid in ranked if lookup[cid].engine == engine]
        if eligible:
            finalists.append(eligible[0])
    for concept in ("quality", "physics", "tail", "cliff"):
        eligible = [cid for cid in ranked if concept in cid]
        if eligible:
            finalists.append(eligible[0])
    finalists = list(dict.fromkeys(finalists))[: finalist_maximum + 5]
    inner_units = [unit for cid in finalists for unit in confirm_units[cid]]
    inner = _merge_predictions(inner_units)
    if len(inner) != len(fit_registry) or inner.structure_id.duplicated().any():
        raise CampaignError(f"outer {outer} inner OOF coverage failed")
    inner_meta, meta_metrics = _crossfit_meta(matrix, inner, workers)
    held_ids = set(registry.loc[registry.outer_role.eq("heldout"), "structure_id"].astype(str))
    fit_ids = set(fit_registry.structure_id.astype(str))
    outer_frames: list[pd.DataFrame] = []
    models: dict[str, Any] = {}
    for index, cid in enumerate(finalists):
        candidate = lookup[cid]
        model, frame, elapsed, summary = _fit_predict(
            v9,
            matrix,
            observations,
            mmp,
            surfaces[candidate.surface],
            fit_ids,
            held_ids,
            candidate,
            workers,
            SEED + outer * 1000 + 500 + index,
        )
        models[cid] = {"model": model, "candidate": asdict(candidate), "columns": surfaces[candidate.surface]}
        frame = frame.rename(columns={"predicted_pic50": f"pred__{cid}"})
        frame["fit_elapsed_seconds"] = elapsed
        frame["fit_structure_count"] = summary["fit_structures"]
        outer_frames.append(frame)
    outer_frame = outer_frames[0]
    for frame in outer_frames[1:]:
        outer_frame = outer_frame.merge(
            frame.drop(columns=["fit_elapsed_seconds", "fit_structure_count"]),
            on=["structure_id", "scaffold_group_id", "observed_pic50"],
            validate="one_to_one",
        )
    prediction_columns = [f"pred__{cid}" for cid in finalists]
    outer_frame["pred__v11_nested"], meta_bundle = _fit_meta_for_outer(
        matrix, inner, outer_frame, meta_metrics, workers
    )
    residuals = np.abs(inner_meta.observed_pic50 - inner_meta.meta__selected)
    q50, q80, q90, q95 = [
        float(np.quantile(residuals, level, method="higher")) for level in (0.50, 0.80, 0.90, 0.95)
    ]
    outer_frame["interval50_half_width"] = q50
    outer_frame["interval80_half_width"] = q80
    outer_frame["interval90_half_width"] = q90
    outer_frame["interval95_half_width"] = q95
    domain = ad.loc[ad.outer_fold.eq(outer)].copy()
    outer_frame = outer_frame.merge(domain, on="structure_id", validate="one_to_one")
    outer_frame["outer_fold"] = outer
    outer_frame["prediction_spread"] = outer_frame[prediction_columns].std(axis=1)
    outer_frame["extrapolation_flag"] = outer_frame.maximum_train_tanimoto.lt(0.50)
    unit_id = f"outer_o{outer}"
    directory = output / "units" / unit_id
    spec = {
        "operation": "nested_outer_optimization",
        "outer_fold": outer,
        "finalists": finalists,
        "selected_meta_method": meta_metrics["selected_method"],
        "selected_base_meta_method": meta_metrics["selected_base_method"],
        "selection_unit_hashes": [unit["unit_json_sha256"] for unit in inner_units],
        "outer_labels_used_for_selection": False,
    }
    existing = _existing(directory, spec, output)
    if existing is not None:
        return existing
    prediction_path = directory / "outer_predictions.parquet"
    inner_path = directory / "inner_meta_predictions.parquet"
    model_path = directory / "outer_models.joblib"
    _atomic_parquet(prediction_path, outer_frame)
    _atomic_parquet(inner_path, inner_meta)
    joblib.dump(
        {"base_models": models, "meta": meta_bundle, "meta_metrics": meta_metrics}, model_path, compress=3
    )
    metrics = _safe_metrics(outer_frame.observed_pic50, outer_frame.pred__v11_nested)
    metrics.update(
        candidate_count=len(finalists),
        finalists=finalists,
        inner_meta=meta_metrics,
        interval_half_widths={"50": q50, "80": q80, "90": q90, "95": q95},
    )
    return _unit_document(
        directory,
        unit_id,
        spec,
        metrics,
        [
            _binding(prediction_path, "outer_predictions"),
            _binding(inner_path, "inner_meta_predictions"),
            _binding(model_path, "outer_models"),
        ],
    )


def _bootstrap_scaffold_delta(frame: pd.DataFrame, replicates: int, seed: int) -> dict[str, Any]:
    scaffold_rows = []
    for scaffold, group in frame.groupby("scaffold_group_id", observed=True):
        new_error = np.abs(group.observed_pic50 - group.pred__v11_nested).to_numpy(float)
        old_error = np.abs(group.observed_pic50 - group.v9_predicted_pic50).to_numpy(float)
        scaffold_rows.append((str(scaffold), float(new_error.sum() - old_error.sum()), int(len(group))))
    deltas = np.array([row[1] for row in scaffold_rows], dtype=float)
    counts = np.array([row[2] for row in scaffold_rows], dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = rng.integers(0, len(scaffold_rows), len(scaffold_rows))
        estimates[index] = float(deltas[sample].sum() / counts[sample].sum())
    point = float(deltas.sum() / counts.sum())
    return {
        "definition": "V11 absolute error minus V9 absolute error; negative favors V11",
        "scaffolds": len(scaffold_rows),
        "replicates": replicates,
        "point_estimate": point,
        "ci95_lower": float(np.quantile(estimates, 0.025)),
        "ci95_upper": float(np.quantile(estimates, 0.975)),
        "probability_v11_better": float(np.mean(estimates < 0)),
    }


def _subgroup_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    masks = {
        "all": np.ones(len(frame), dtype=bool),
        "low_potency_tail": frame.observed_pic50.lt(4.5).to_numpy(),
        "high_potency_tail": frame.observed_pic50.ge(6.5).to_numpy(),
        "molecular_weight_ge_500": frame["rdkit2d__MolWt"].ge(500).to_numpy(),
        "molecular_weight_lt_500": frame["rdkit2d__MolWt"].lt(500).to_numpy(),
        "low_similarity_lt_0p5": frame.maximum_train_tanimoto.lt(0.5).to_numpy(),
        "high_similarity_ge_0p7": frame.maximum_train_tanimoto.ge(0.7).to_numpy(),
        "flexible_rotors_ge_8": frame["rdkit2d__NumRotatableBonds"].ge(8).to_numpy(),
    }
    for name, mask in masks.items():
        part = frame.loc[mask]
        if len(part) < 20:
            continue
        for model, column in (("v11", "pred__v11_nested"), ("v9", "v9_predicted_pic50")):
            rows.append(
                {"subgroup": name, "model": model, **_safe_metrics(part.observed_pic50, part[column])}
            )
    for outer, part in frame.groupby("outer_fold", observed=True):
        for model, column in (("v11", "pred__v11_nested"), ("v9", "v9_predicted_pic50")):
            rows.append(
                {
                    "subgroup": f"outer_fold_{outer}",
                    "model": model,
                    **_safe_metrics(part.observed_pic50, part[column]),
                }
            )
    return pd.DataFrame(rows)


def _aggregate(
    output: Path,
    matrix: pd.DataFrame,
    observations: pd.DataFrame,
    v9_anchor: pd.DataFrame,
    outer_units: list[dict[str, Any]],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    analysis = output / "analysis"
    validation_path = analysis / "validation.json"
    if validation_path.is_file():
        value = _read_json(validation_path, "validation_sha256")
        for binding in value["artifacts"]:
            _verify_binding(binding, output)
        return value
    frames = [pd.read_parquet(_artifact(unit, "outer_predictions")) for unit in outer_units]
    oof = pd.concat(frames, ignore_index=True)
    if len(oof) != EXACT_ROWS or oof.structure_id.duplicated().any():
        raise CampaignError("nested OOF is not exactly 18,801 unique structures")
    if set(oof.outer_fold.astype(int)) != set(OUTERS):
        raise CampaignError("nested OOF lacks five outer folds")
    oof = oof.merge(v9_anchor, on="structure_id", validate="one_to_one")
    context = matrix[
        [
            "structure_id",
            "rdkit2d__MolWt",
            "rdkit2d__MolLogP",
            "rdkit2d__TPSA",
            "rdkit2d__NumRotatableBonds",
            "rdkit2d__HeavyAtomCount",
        ]
    ]
    oof = oof.merge(context, on="structure_id", validate="one_to_one")
    bootstrap = _bootstrap_scaffold_delta(oof, bootstrap_replicates, SEED)
    subgroup = _subgroup_metrics(oof)
    oof_path = analysis / "nested_oof_predictions.parquet"
    subgroup_path = analysis / "subgroup_metrics.parquet"
    comparison_path = analysis / "v11_vs_v9_scaffold_bootstrap.json"
    _atomic_parquet(oof_path, oof)
    _atomic_parquet(subgroup_path, subgroup)
    comparison = _atomic_json(
        comparison_path,
        {"schema_version": SCHEMA_VERSION, "status": "passed", **bootstrap},
        "comparison_sha256",
    )
    metrics = {
        "v11_nested": _safe_metrics(oof.observed_pic50, oof.pred__v11_nested),
        "v9_anchor": _safe_metrics(oof.observed_pic50, oof.v9_predicted_pic50),
        "paired_scaffold_bootstrap": comparison,
    }
    unit_rows: list[dict[str, Any]] = []
    for path in sorted((output / "units").glob("*/unit.json")):
        unit = _read_json(path, "unit_json_sha256")
        if unit["unit_spec"]["operation"] not in {
            "successive_halving_screen",
            "successive_halving_confirmation",
        }:
            continue
        candidate = unit["unit_spec"]["candidate"]
        unit_rows.append(
            {
                "unit_id": unit["unit_id"],
                "operation": unit["unit_spec"]["operation"],
                "outer_fold": unit["unit_spec"]["outer_fold"],
                "inner_fold": unit["unit_spec"]["inner_fold"],
                "candidate_id": candidate["candidate_id"],
                "engine": candidate["engine"],
                "surface": candidate["surface"],
                "target_mode": candidate["target_mode"],
                "weight_mode": candidate["weight_mode"],
                "training_scope": candidate["training_scope"],
                "mae": unit["metrics"]["mae"],
                "rmse": unit["metrics"]["rmse"],
                "fit_structures": unit["metrics"]["training_summary"]["fit_structures"],
            }
        )
    landscape = pd.DataFrame(unit_rows)
    landscape_path = analysis / "candidate_landscape.parquet"
    _atomic_parquet(landscape_path, landscape)
    modality_rows: list[dict[str, Any]] = []
    joined = observations.merge(
        oof[["structure_id", "observed_pic50", "pred__v11_nested", "v9_predicted_pic50"]],
        on="structure_id",
        validate="many_to_one",
    )
    for modality, group in joined.groupby("measurement_modality", observed=True):
        if group.structure_id.nunique() < 50:
            continue
        structures = group.drop_duplicates("structure_id")
        modality_rows.append(
            {
                "measurement_modality": str(modality),
                "structures": int(len(structures)),
                "observations": int(len(group)),
                "v11_mae": _safe_metrics(structures.observed_pic50, structures.pred__v11_nested)["mae"],
                "v9_mae": _safe_metrics(structures.observed_pic50, structures.v9_predicted_pic50)["mae"],
                "interpretation": "descriptive and chemistry-confounded; not a causal assay effect",
            }
        )
    modality_path = analysis / "measurement_quality_strata.parquet"
    _atomic_parquet(modality_path, pd.DataFrame(modality_rows))
    report = [
        "# hERG Comprehensive Optimization V11",
        "",
        "## Primary Result",
        "",
        f"- V11 nested scaffold MAE: {metrics['v11_nested']['mae']:.6f}",
        f"- V9 anchor MAE on identical identities: {metrics['v9_anchor']['mae']:.6f}",
        f"- Paired MAE delta (V11 minus V9): {bootstrap['point_estimate']:+.6f}",
        f"- 95% scaffold-bootstrap interval: [{bootstrap['ci95_lower']:+.6f}, {bootstrap['ci95_upper']:+.6f}]",
        "",
        "## What Was Tested",
        "",
        "Model families, independent hyperparameters, feature surfaces, measurement-quality targets, high-quality-only training, reliability weighting, potency-tail weighting, activity-cliff weighting, heavy/flexible weighting, inner-only stacking, and residual correction.",
        "",
        "## Scientific Boundary",
        "",
        "All results are internal TRAIN-partition nested scaffold evidence. Repository validation and test labels remained sealed. No external, prospective, clinical, or causal superiority is claimed.",
    ]
    report_path = analysis / "ANALYSIS.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    artifacts = [
        _binding(oof_path, "nested_oof_predictions"),
        _binding(subgroup_path, "subgroup_metrics"),
        _binding(comparison_path, "v11_vs_v9_bootstrap"),
        _binding(landscape_path, "candidate_landscape"),
        _binding(modality_path, "measurement_quality_strata"),
        _binding(report_path, "analysis_report"),
    ]
    return _atomic_json(
        validation_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "exact_nested_oof_rows": EXACT_ROWS,
            "metrics": metrics,
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "artifacts": artifacts,
        },
        "validation_sha256",
    )


def _refit_final(
    v9: Any,
    output: Path,
    matrix: pd.DataFrame,
    observations: pd.DataFrame,
    mmp: pd.DataFrame,
    surfaces: dict[str, list[str]],
    candidates: list[Candidate],
    outer_units: list[dict[str, Any]],
    workers: int,
) -> dict[str, Any]:
    target = output / "final_model/validation.json"
    if target.is_file():
        value = _read_json(target, "validation_sha256")
        for row in value["artifacts"]:
            _verify_binding(row, output)
        return value
    counts = Counter(cid for unit in outer_units for cid in unit["unit_spec"]["finalists"])
    landscape = pd.read_parquet(output / "analysis/candidate_landscape.parquet")
    confirmed = landscape.loc[landscape.operation.eq("successive_halving_confirmation")]
    ranking = confirmed.groupby("candidate_id", observed=True).mae.mean().sort_values()
    ordered = sorted(counts, key=lambda cid: (-counts[cid], float(ranking.get(cid, 99.0))))
    selected_ids = ordered[:3]
    lookup = {row.candidate_id: row for row in candidates}
    ids = set(matrix.structure_id.astype(str))
    models: list[dict[str, Any]] = []
    training_counts: dict[str, int] = {}
    smoke_rows: list[dict[str, Any]] = []
    for index, cid in enumerate(selected_ids):
        candidate = lookup[cid]
        targets = _measurement_targets(v9, observations, ids, candidate)
        joined = matrix.merge(targets, on="structure_id", validate="one_to_one")
        training_counts[cid] = int(len(joined))
        y = joined.training_target.to_numpy(float)
        weights = _training_weights(joined, mmp, ids, candidate)
        model = v9._new_model(candidate, workers, SEED + 90_000 + index)
        kwargs = (
            {"model__sample_weight": weights} if isinstance(model, Pipeline) else {"sample_weight": weights}
        )
        columns = surfaces[candidate.surface]
        model.fit(_qc_clean_matrix(v9, joined, columns), y, **kwargs)
        sample = matrix.iloc[:5]
        prediction = np.asarray(model.predict(_qc_clean_matrix(v9, sample, columns)), dtype=float)
        models.append({"candidate": asdict(candidate), "columns": columns, "model": model})
        smoke_rows.extend(
            {"candidate_id": cid, "structure_id": str(sid), "predicted_pic50": float(value)}
            for sid, value in zip(sample.structure_id, prediction, strict=True)
        )
    final = output / "final_model"
    final.mkdir(parents=True, exist_ok=True)
    model_path = final / "model_bundle.joblib"
    schema_path = final / "feature_schema.json"
    smoke_path = final / "inference_smoke.parquet"
    card_path = final / "MODEL_CARD.md"
    joblib.dump(
        {"schema_version": SCHEMA_VERSION, "models": models, "aggregation": "mean"}, model_path, compress=3
    )
    _atomic_json(
        schema_path,
        {
            "schema_version": SCHEMA_VERSION,
            "selected_candidates": [asdict(lookup[cid]) for cid in selected_ids],
            "model_feature_columns": {cid: surfaces[lookup[cid].surface] for cid in selected_ids},
            "input_structure_standardization": "must match bound V9 matrix lineage",
            "prediction_output": "pIC50; ensemble mean across selected full-TRAIN models",
        },
        "feature_schema_sha256",
    )
    _atomic_parquet(smoke_path, pd.DataFrame(smoke_rows))
    card_path.write_text(
        "# V11 Full-TRAIN hERG Model Bundle\n\n"
        "This artifact refits the train-only selected recipes on all authorized exact TRAIN structures. "
        "Its performance estimate is the five-fold nested scaffold OOF analysis, not an in-sample score. "
        "Repository validation and test labels remain sealed. The endpoint is wild-type-or-unspecified "
        "hERG pIC50 and is not confirmed-WT clinical risk.\n",
        encoding="utf-8",
    )
    artifacts = [
        _binding(model_path, "model_bundle"),
        _binding(schema_path, "feature_schema"),
        _binding(smoke_path, "inference_smoke"),
        _binding(card_path, "model_card"),
    ]
    return _atomic_json(
        target,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "selected_candidate_ids": selected_ids,
            "authorized_full_train_structures": EXACT_ROWS,
            "actual_training_structures_by_candidate": training_counts,
            "performance_evidence_role": "analysis/nested_oof_predictions.parquet",
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "artifacts": artifacts,
        },
        "validation_sha256",
    )


def _finalize(
    output: Path,
    prepared: dict[str, Any],
    analysis: dict[str, Any],
    model: dict[str, Any],
    outer_units: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    validation_path = output / "validation.json"
    summary_path = output / "final_summary.json"
    unit_paths = sorted((output / "units").glob("*/unit.json"))
    unit_bindings = [_binding(path, f"unit_{path.parent.name}") for path in unit_paths]
    artifact_bindings = [
        *analysis["artifacts"],
        *model["artifacts"],
        _binding(output / "prepared/validation.json", "prepared_validation"),
        *unit_bindings,
    ]
    manifest = _atomic_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "prepared_validation_sha256": prepared["validation_sha256"],
            "analysis_validation_sha256": analysis["validation_sha256"],
            "model_validation_sha256": model["validation_sha256"],
            "outer_unit_sha256": [unit["unit_json_sha256"] for unit in outer_units],
            "all_unit_count": len(unit_bindings),
            "artifacts": artifact_bindings,
            "scientific_scope": {
                "source_partition": "train",
                "nested_scaffold_internal_evidence": True,
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
                "external_or_prospective_validation": False,
            },
        },
        "manifest_sha256",
    )
    validation = _atomic_json(
        validation_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": _utc(),
            "manifest_sha256_bound": manifest["manifest_sha256"],
            "exact_nested_oof_rows": EXACT_ROWS,
            "outer_folds": 5,
            "metrics": analysis["metrics"],
            "final_model_candidates": model["selected_candidate_ids"],
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
        },
        "validation_sha256",
    )
    return _atomic_json(
        summary_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "finished_utc": _utc(),
            "manifest_sha256": manifest["manifest_sha256"],
            "validation_sha256": validation["validation_sha256"],
            "v11_nested_mae": analysis["metrics"]["v11_nested"]["mae"],
            "v9_anchor_mae": analysis["metrics"]["v9_anchor"]["mae"],
            "paired_delta": analysis["metrics"]["paired_scaffold_bootstrap"]["point_estimate"],
            "interpretation": "negative paired delta favors V11; internal train-only nested scaffold evidence",
        },
        "summary_sha256",
    )


def _validate(output: Path) -> dict[str, Any]:
    summary = _read_json(output / "final_summary.json", "summary_sha256")
    manifest = _read_json(output / "manifest.json", "manifest_sha256")
    validation = _read_json(output / "validation.json", "validation_sha256")
    if summary.get("status") != "complete" or validation.get("status") != "passed":
        raise CampaignError("campaign is not complete")
    if summary["manifest_sha256"] != manifest["manifest_sha256"]:
        raise CampaignError("summary/manifest binding mismatch")
    for binding in manifest["artifacts"]:
        _verify_binding(binding, output)
    oof = pd.read_parquet(output / "analysis/nested_oof_predictions.parquet")
    if len(oof) != EXACT_ROWS or oof.structure_id.duplicated().any() or set(oof.outer_fold) != set(OUTERS):
        raise CampaignError("final nested OOF contract failed")
    return {
        "status": "passed",
        "summary_sha256": summary["summary_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "validation_sha256": validation["validation_sha256"],
        "exact_nested_oof_rows": EXACT_ROWS,
        "v11_nested_mae": summary["v11_nested_mae"],
        "v9_anchor_mae": summary["v9_anchor_mae"],
        "paired_delta": summary["paired_delta"],
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
    }


def _status(output: Path) -> dict[str, Any]:
    unit_paths = sorted((output / "units").glob("*/unit.json")) if (output / "units").is_dir() else []
    operations: Counter[str] = Counter()
    for path in unit_paths:
        try:
            operations[_read_json(path, "unit_json_sha256")["unit_spec"]["operation"]] += 1
        except Exception:
            operations["invalid"] += 1
    result: dict[str, Any] = {
        "status": "complete" if (output / "final_summary.json").is_file() else "incomplete",
        "output_root": str(output),
        "completed_units": len(unit_paths),
        "completed_by_operation": dict(sorted(operations.items())),
        "prepared": (output / "prepared/validation.json").is_file(),
        "analysis": (output / "analysis/validation.json").is_file(),
        "final_model": (output / "final_model/validation.json").is_file(),
        "resume": "rerun the identical command",
    }
    if result["status"] == "complete":
        result["summary"] = _read_json(output / "final_summary.json", "summary_sha256")
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.workers < 1 or args.workers > 6:
        raise CampaignError("workers must be in [1, 6]")
    if (output / "final_summary.json").is_file():
        return _validate(output)
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()
    with _Lock(output / ".campaign.lock"):
        prepared = _prepare(repo, source, output)
        matrix, splits, observations, mmp, ad, v9_anchor, blocks = _load(output)
        v9 = _load_v9_implementation(repo)
        surfaces = _surface_columns(blocks)
        candidates = _candidate_plan(args.smoke)
        if args.plan_only:
            return {
                "status": "plan_only",
                "candidate_count": len(candidates),
                "candidates": [asdict(row) for row in candidates],
                "surfaces": {key: len(value) for key, value in surfaces.items()},
            }
        outer_units: list[dict[str, Any]] = []
        for outer in OUTERS[: args.outer_limit]:
            if stop_requested:
                return {"status": "paused", **_status(output)}
            if args.max_active_hours and time.monotonic() - started >= args.max_active_hours * 3600:
                return {"status": "time_budget_reached", **_status(output)}
            outer_units.append(
                _run_outer(
                    v9,
                    output,
                    matrix,
                    splits,
                    observations,
                    mmp,
                    ad,
                    surfaces,
                    candidates,
                    outer,
                    args.workers,
                    4 if args.smoke else args.screen_maximum,
                    3 if args.smoke else args.finalist_maximum,
                )
            )
            print(
                json.dumps(
                    {"stage": "outer_complete", "outer": outer, "metrics": outer_units[-1]["metrics"]}
                ),
                flush=True,
            )
        if args.outer_limit < len(OUTERS):
            return {
                "status": "bounded_smoke_complete",
                "completed_outer_folds": args.outer_limit,
                **_status(output),
            }
        analysis = _aggregate(
            output,
            matrix,
            observations,
            v9_anchor,
            outer_units,
            200 if args.smoke else args.bootstrap_replicates,
        )
        model = _refit_final(
            v9, output, matrix, observations, mmp, surfaces, candidates, outer_units, args.workers
        )
        summary = _finalize(output, prepared, analysis, model, outer_units)
        return {"status": "complete", "summary": summary, "validation": _validate(output)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "status", "validate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--repo-root", type=Path, default=Path.cwd())
        sub.add_argument(
            "--source-root",
            type=Path,
            default=Path("research/local_runs/herg_domain_mixture_campaign_v9"),
        )
        sub.add_argument(
            "--output-root",
            type=Path,
            default=Path("research/local_runs/herg_comprehensive_optimization_v11"),
        )
        sub.add_argument("--workers", type=int, default=6)
    run = subparsers.choices["run"]
    run.add_argument("--screen-maximum", type=int, default=12)
    run.add_argument("--finalist-maximum", type=int, default=6)
    run.add_argument("--bootstrap-replicates", type=int, default=10_000)
    run.add_argument("--max-active-hours", type=float, default=0.0)
    run.add_argument("--outer-limit", type=int, choices=range(1, 6), default=5)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--plan-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.repo_root = args.repo_root.resolve()
    if not args.source_root.is_absolute():
        args.source_root = args.repo_root / args.source_root
    if not args.output_root.is_absolute():
        args.output_root = args.repo_root / args.output_root
    if args.command == "run":
        result = _run(args)
    elif args.command == "status":
        result = _status(args.output_root)
    else:
        result = _validate(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
