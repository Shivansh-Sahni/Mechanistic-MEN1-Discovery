#!/usr/bin/env python3
"""Build, validate, predict with, and serve the ligand-only hERG V10 prototype.

V10 is deliberately tiered:

1. A three-class scaffold-validated router predicts Safe (>30 uM),
   Moderate (1-30 uM), or Potent (<1 uM) from RDKit2D + Morgan features.
2. The frozen V9 quantitative model predicts pIC50/IC50. The UI emphasizes
   this estimate for the Moderate class, where an exact value matters most.

The canonical corpus has no direct IC10 or IC30 measurements. Those values
are therefore shown only as Hill-equation estimates derived from predicted
IC50, with slope sensitivity. They are never described as trained endpoints.

This is a local research prototype. Repository validation and test labels are
not opened by the build.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

SCHEMA_VERSION = "platform-local-herg-v10-tiered-prototype/1.0"
SEED = 20260819
CLASS_NAMES = ["Safe", "Moderate", "Potent"]
SAFE_PIC50 = -math.log10(30e-6)
POTENT_PIC50 = -math.log10(1e-6)
MORGAN_BITS = 2048
DEFAULT_V9 = Path("research/local_runs/herg_domain_mixture_campaign_v9")
DEFAULT_SURFACE = Path(
    "research/data/platform/processed/herg_hierarchy/"
    "v1_6_training_surfaces/herg_training_observations.parquet"
)
DEFAULT_BROAD = Path(
    "research/data/platform/processed/herg_hierarchy/"
    "v1_6_training_surfaces/confirmed_wt_fixed_dose_structure_labels.parquet"
)
DEFAULT_OUTPUT = Path("research/local_runs/herg_v10_tiered_platform")

RDLogger.DisableLog("rdApp.warning")


class V10Error(RuntimeError):
    """Raised when the V10 scientific or artifact contract is violated."""


@dataclass(frozen=True)
class Paths:
    repo: Path
    v9: Path
    surface: Path
    broad: Path
    output: Path


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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


def _binding(path: Path, role: str, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise V10Error(f"missing {role}: {resolved}")
    if root is not None and not resolved.is_relative_to(root.resolve()):
        raise V10Error(f"artifact escapes output root: {resolved}")
    result: dict[str, Any] = {
        "role": role,
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha(resolved),
    }
    if resolved.suffix == ".parquet":
        result["rows"] = pq.read_metadata(resolved).num_rows
        result["arrow_schema_sha256"] = _schema_sha(resolved)
    return result


def _verify_binding(binding: dict[str, Any], *, root: Path | None = None) -> None:
    path = Path(str(binding["path"])).resolve()
    if root is not None and not path.is_relative_to(root.resolve()):
        raise V10Error(f"artifact escapes output root: {path}")
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
        raise V10Error(f"artifact missing or size changed: {path}")
    if _sha(path) != str(binding["sha256"]):
        raise V10Error(f"artifact hash changed: {path}")
    if path.suffix == ".parquet":
        if pq.read_metadata(path).num_rows != int(binding["rows"]):
            raise V10Error(f"artifact row count changed: {path}")
        if _schema_sha(path) != str(binding["arrow_schema_sha256"]):
            raise V10Error(f"artifact schema changed: {path}")


def _atomic_json(path: Path, value: dict[str, Any], self_key: str | None = None) -> None:
    document = copy.deepcopy(value)
    if self_key:
        document.pop(self_key, None)
        document[self_key] = _digest(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path, self_key: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise V10Error(f"expected JSON object: {path}")
    if self_key:
        expected = value.get(self_key)
        candidate = copy.deepcopy(value)
        candidate.pop(self_key, None)
        if expected != _digest(candidate):
            raise V10Error(f"self-hash mismatch: {path}")
    return value


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _clean(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    values = frame[columns].to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    values[np.abs(values) > 1e30] = np.nan
    return pd.DataFrame(values, columns=columns, index=frame.index)


def tier_index(pic50: np.ndarray | list[float] | float) -> np.ndarray:
    """Return 0 Safe, 1 Moderate, 2 Potent for pIC50 values."""
    values = np.asarray(pic50, dtype=float)
    return np.where(values < SAFE_PIC50, 0, np.where(values <= POTENT_PIC50, 1, 2)).astype(int)


def ic50_um_from_pic50(pic50: float) -> float:
    return float(10 ** (6.0 - float(pic50)))


def hill_icx_um(ic50_um: float, fraction: float, hill_slope: float = 1.0) -> float:
    """Derive ICx from IC50 under a Hill inhibition curve."""
    if not 0 < fraction < 1 or hill_slope <= 0 or ic50_um <= 0:
        raise ValueError("ICx requires positive IC50/slope and 0 < fraction < 1")
    return float(ic50_um * (fraction / (1.0 - fraction)) ** (1.0 / hill_slope))


def _descriptor_registry() -> dict[str, Any]:
    registry: dict[str, Any] = {}
    for name, function in Descriptors._descList:  # noqa: SLF001
        if name in registry:
            raise V10Error(f"duplicate RDKit descriptor: {name}")
        registry[str(name)] = function
    return registry


def _feature_frame(smiles: str, columns: list[str]) -> tuple[pd.DataFrame, Chem.Mol]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise V10Error("RDKit could not parse the submitted SMILES")
    values: dict[str, float] = {}
    registry = _descriptor_registry()
    for column in columns:
        if column.startswith("rdkit2d__"):
            name = column.removeprefix("rdkit2d__")
            function = registry.get(name)
            if function is None:
                values[column] = math.nan
                continue
            try:
                value = float(function(molecule))
                values[column] = value if math.isfinite(value) and abs(value) <= 1e30 else math.nan
            except Exception:
                values[column] = math.nan
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_BITS, includeChirality=True)
    array = np.zeros(MORGAN_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), array)
    for index in range(MORGAN_BITS):
        key = f"morgan__{index:04d}"
        if key in columns:
            values[key] = int(array[index])
    return pd.DataFrame([{name: values.get(name, math.nan) for name in columns}]), molecule


def _class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=3).astype(float)
    weights = len(labels) / (3.0 * np.maximum(counts, 1.0))
    return weights[labels]


def _new_classifier(workers: int, seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=650,
        max_depth=6,
        learning_rate=0.035,
        min_child_weight=6.0,
        subsample=0.8,
        colsample_bytree=0.65,
        reg_alpha=0.5,
        reg_lambda=6.0,
        max_bin=128,
        n_jobs=workers,
        random_state=seed,
        verbosity=0,
    )


def _classification_metrics(observed: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    probability_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probability_sums).all() or np.any(probability_sums <= 0):
        raise V10Error("classification probabilities are invalid")
    probabilities = probabilities / probability_sums
    predicted = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(observed, predicted, labels=[0, 1, 2])
    potent_truth = observed == 2
    safe_truth = observed == 0
    review_count = max(1, math.ceil(0.01 * len(observed)))
    top = np.argsort(probabilities[:, 2])[::-1][:review_count]
    potent_binary = (observed == 2).astype(int)
    return {
        "n": int(len(observed)),
        "accuracy": float(accuracy_score(observed, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(observed, predicted)),
        "macro_f1": float(f1_score(observed, predicted, average="macro")),
        "multiclass_log_loss": float(log_loss(observed, probabilities, labels=[0, 1, 2])),
        "confusion_matrix_rows_actual_columns_predicted": matrix.tolist(),
        "per_class_precision": {
            CLASS_NAMES[i]: float(precision_score(observed == i, predicted == i, zero_division=0))
            for i in range(3)
        },
        "per_class_recall": {
            CLASS_NAMES[i]: float(recall_score(observed == i, predicted == i, zero_division=0))
            for i in range(3)
        },
        "per_class_average_precision": {
            CLASS_NAMES[i]: float(average_precision_score(observed == i, probabilities[:, i]))
            for i in range(3)
        },
        "dangerous_potent_predicted_safe_rate": float(
            np.mean(predicted[potent_truth] == 0) if potent_truth.any() else 0.0
        ),
        "safe_predicted_potent_rate": float(np.mean(predicted[safe_truth] == 2) if safe_truth.any() else 0.0),
        "severe_safe_potent_crossing_rate": float(np.mean(np.abs(predicted - observed) == 2)),
        "potent_precision_at_top_1_percent": float(np.mean(potent_binary[top])),
        "potent_recall_at_top_1_percent": float(np.sum(potent_binary[top]) / max(1, potent_binary.sum())),
    }


def _regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(observed - predicted)
    return {
        "n": int(len(observed)),
        "mae": float(np.mean(absolute)),
        "median_absolute_error": float(np.median(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(observed - predicted)))),
        "within_0p5": float(np.mean(absolute <= 0.5)),
        "within_1p0": float(np.mean(absolute <= 1.0)),
    }


def _train_router(paths: Paths, workers: int) -> tuple[dict[str, Any], pd.DataFrame]:
    schema = _read_json(paths.v9 / "final_model/feature_preprocessing_schema.json")
    features = [str(value) for value in schema["feature_columns"]]
    matrix = pd.read_parquet(paths.v9 / "prepared/training_matrix.parquet")
    required = {"structure_id", "scaffold_group_id", "target_pic50", *features}
    if not required.issubset(matrix.columns):
        raise V10Error("V9 training matrix does not satisfy the frozen feature contract")
    if len(matrix) != 18_801 or matrix.structure_id.nunique() != 18_801:
        raise V10Error("expected exactly 18,801 unique V9 train structures")
    split = pd.read_parquet(paths.v9 / "prepared/fixed_nested_scaffold_splits.parquet")
    heldout = split.loc[split.outer_role.eq("heldout"), ["structure_id", "outer_fold"]]
    if len(heldout) != 18_801 or heldout.structure_id.nunique() != 18_801:
        raise V10Error("frozen heldout fold map is not exactly-once")
    matrix = matrix.merge(heldout, on="structure_id", validate="one_to_one")
    labels = tier_index(matrix.target_pic50.to_numpy(float))
    raw_oof = np.full((len(matrix), 3), np.nan, dtype=float)
    for outer_fold in range(5):
        fit = matrix.outer_fold.ne(outer_fold).to_numpy()
        evaluate = ~fit
        model = _new_classifier(workers, SEED + outer_fold)
        model.fit(
            _clean(matrix.loc[fit], features),
            labels[fit],
            sample_weight=_class_weights(labels[fit]),
        )
        raw_oof[evaluate] = model.predict_proba(_clean(matrix.loc[evaluate], features))
    if not np.isfinite(raw_oof).all():
        raise V10Error("router OOF predictions are incomplete")
    calibrator = LogisticRegression(C=3.0, max_iter=2_000, random_state=SEED)
    calibrator.fit(np.log(np.clip(raw_oof, 1e-7, 1.0)), labels)
    calibrated = calibrator.predict_proba(np.log(np.clip(raw_oof, 1e-7, 1.0)))
    final_model = _new_classifier(workers, SEED + 99)
    final_model.fit(_clean(matrix, features), labels, sample_weight=_class_weights(labels))
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "model": final_model,
        "calibrator": calibrator,
        "feature_columns": features,
        "class_names": CLASS_NAMES,
        "safe_pic50_boundary": SAFE_PIC50,
        "potent_pic50_boundary": POTENT_PIC50,
        "training_partition": "train",
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
    }
    joblib.dump(bundle, paths.output / "models/router_classifier.joblib", compress=3)
    oof = pd.DataFrame(
        {
            "structure_id": matrix.structure_id.astype(str),
            "scaffold_group_id": matrix.scaffold_group_id.astype(str),
            "outer_fold": matrix.outer_fold.astype(int),
            "observed_pic50": matrix.target_pic50.astype(float),
            "observed_tier_index": labels,
            "observed_tier": [CLASS_NAMES[i] for i in labels],
            "probability_safe": calibrated[:, 0],
            "probability_moderate": calibrated[:, 1],
            "probability_potent": calibrated[:, 2],
            "predicted_tier_index": np.argmax(calibrated, axis=1),
        }
    )
    oof["predicted_tier"] = [CLASS_NAMES[i] for i in oof.predicted_tier_index]
    metrics = {
        "direct_router_raw": _classification_metrics(labels, raw_oof),
        "direct_router_calibrated": _classification_metrics(labels, calibrated),
        "class_counts": {CLASS_NAMES[i]: int(np.sum(labels == i)) for i in range(3)},
    }
    return metrics, oof


def _regression_evidence(paths: Paths) -> tuple[dict[str, Any], pd.DataFrame]:
    source = paths.v9 / "analysis/nested_oof_predictions.parquet"
    frame = pd.read_parquet(source)
    column = "pred__xgb_depth10"
    if len(frame) != 18_801 or column not in frame:
        raise V10Error("V9 nested regression evidence is missing")
    observed = frame.observed_pic50.to_numpy(float)
    predicted = frame[column].to_numpy(float)
    actual_tier = tier_index(observed)
    derived = np.eye(3)[tier_index(predicted)]
    moderate = actual_tier == 1
    residual = np.abs(observed - predicted)
    evidence = frame[["structure_id", "scaffold_group_id", "outer_fold", "observed_pic50"]].copy()
    evidence["predicted_pic50"] = predicted
    evidence["absolute_error"] = residual
    metrics = {
        "all_exact_train_oof": _regression_metrics(observed, predicted),
        "moderate_band_oof": _regression_metrics(observed[moderate], predicted[moderate]),
        "regression_threshold_router": _classification_metrics(actual_tier, derived),
        "absolute_residual_quantiles": {
            "q80": float(np.quantile(residual, 0.8)),
            "q90": float(np.quantile(residual, 0.9)),
            "q95": float(np.quantile(residual, 0.95)),
        },
    }
    return metrics, evidence


def _dataset_census(paths: Paths, class_counts: dict[str, int]) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = [
        "observation_id",
        "structure_id",
        "source_family",
        "assay_family",
        "measurement_modality",
        "native_endpoint",
        "native_relation",
        "native_unit",
        "endpoint_standardization_status",
        "standardized_pic50_primary",
        "confirmed_wt_fixed_dose_primary",
        "model_split",
    ]
    observations = pd.read_parquet(paths.surface, columns=columns)
    broad = pd.read_parquet(paths.broad, columns=["structure_id", "model_split"])
    queue = observations.loc[observations.confirmed_wt_fixed_dose_primary].copy()
    queue_summary = (
        queue.groupby(
            [
                "source_family",
                "assay_family",
                "measurement_modality",
                "native_endpoint",
                "native_relation",
                "native_unit",
            ],
            dropna=False,
        )
        .agg(observations=("observation_id", "size"), structures=("structure_id", "nunique"))
        .reset_index()
        .sort_values(["observations", "structures"], ascending=False)
    )
    direct_endpoints = observations.native_endpoint.fillna("").str.upper()
    census = {
        "reported_observations": int(len(observations)),
        "reported_unique_structures": int(observations.structure_id.nunique()),
        "confirmed_wt_fixed_dose_structures": int(broad.structure_id.nunique()),
        "confirmed_wt_fixed_dose_train_structures": int(
            broad.loc[broad.model_split.eq("train"), "structure_id"].nunique()
        ),
        "exact_standardized_observations": int(
            observations.endpoint_standardization_status.eq("exact_standardized").sum()
        ),
        "censored_standardized_observations": int(
            observations.endpoint_standardization_status.eq("censored_standardized").sum()
        ),
        "v10_direct_router_train_structures": 18_801,
        "v10_direct_router_class_counts": class_counts,
        "native_ic50_observations": int(direct_endpoints.eq("IC50").sum()),
        "native_ic10_observations": int(direct_endpoints.eq("IC10").sum()),
        "native_ic30_observations": int(direct_endpoints.eq("IC30").sum()),
        "broad_surface_ternary_status": (
            "assay-standardization queue; fixed-dose labels are not forced into "
            "1 uM and 30 uM tiers without assay-specific threshold adjudication"
        ),
    }
    return census, queue_summary


def _reference_artifacts(paths: Paths, feature_columns: list[str]) -> None:
    matrix = pd.read_parquet(
        paths.v9 / "prepared/training_matrix.parquet",
        columns=["structure_id", "scaffold_group_id", "target_pic50", *feature_columns],
    )
    bit_columns = [name for name in feature_columns if name.startswith("morgan__")]
    bits = matrix[bit_columns].fillna(0).to_numpy(dtype=np.uint8)
    packed = np.packbits(bits, axis=1, bitorder="big")
    observations = pd.read_parquet(paths.surface, columns=["structure_id", "standardized_smiles"])
    smiles = observations.drop_duplicates("structure_id")
    reference = matrix[["structure_id", "scaffold_group_id", "target_pic50"]].merge(
        smiles, on="structure_id", how="left", validate="one_to_one"
    )
    _atomic_parquet(paths.output / "reference/training_reference.parquet", reference)
    target = paths.output / "reference/training_morgan_bits.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, packed=packed)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _model_info(paths: Paths, metrics: dict[str, Any], census: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "local_ligand_only_research_prototype",
        "tiers": [
            {"name": "Safe", "ic50_rule": "> 30 uM", "pic50_rule": f"< {SAFE_PIC50:.4f}"},
            {
                "name": "Moderate",
                "ic50_rule": "1-30 uM",
                "pic50_rule": f"{SAFE_PIC50:.4f}-6.0000",
            },
            {"name": "Potent", "ic50_rule": "< 1 uM", "pic50_rule": "> 6.0000"},
        ],
        "directly_trained_outputs": ["three-class IC50 tier", "pIC50", "IC50"],
        "derived_outputs": {
            "IC10": "Hill-equation estimate from predicted IC50; no direct IC10 records",
            "IC30": "Hill-equation estimate from predicted IC50; no direct IC30 records",
            "hill_slope_sensitivity": [0.7, 1.3],
        },
        "metrics": metrics,
        "dataset_census": census,
        "scientific_scope": {
            "target": "wild-type-or-unspecified hERG exact pIC50 for quantitative training",
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "external_or_prospective_validation": False,
            "receptor_aware_features": False,
        },
    }


def _write_report(paths: Paths, info: dict[str, Any]) -> None:
    router = info["metrics"]["router"]["direct_router_calibrated"]
    regression = info["metrics"]["regression"]["all_exact_train_oof"]
    moderate = info["metrics"]["regression"]["moderate_band_oof"]
    census = info["dataset_census"]
    report = f"""# hERG V10 Tiered Prototype

V10 converts the ligand-only V9 work into a two-stage decision tool.

## Stage 1: IC50 Safety Routing

The router assigns **Safe (>30 uM)**, **Moderate (1-30 uM)**, or **Potent (<1 uM)**.
It was evaluated once per structure across five scaffold-held-out train folds.

- Structures: {router["n"]:,}
- Balanced accuracy: {router["balanced_accuracy"]:.3f}
- Macro F1: {router["macro_f1"]:.3f}
- Potent predicted Safe: {router["dangerous_potent_predicted_safe_rate"]:.2%}
- Severe Safe/Potent crossings: {router["severe_safe_potent_crossing_rate"]:.2%}

## Stage 2: Quantitative IC50

The frozen V9 XGBoost recipe supplies the quantitative estimate.

- Overall nested OOF MAE: {regression["mae"]:.4f} pIC50
- Overall within 1 log: {regression["within_1p0"]:.1%}
- Moderate-band nested OOF MAE: {moderate["mae"]:.4f} pIC50

## Endpoint Honesty

The canonical surface has {census["native_ic50_observations"]:,} native IC50 observations,
but **zero direct IC10 and zero direct IC30 observations**. The UI therefore presents IC10
and IC30 only as Hill-equation estimates derived from IC50 and shows sensitivity to Hill
slope 0.7-1.3. These are not trained endpoint predictions.

## The 400k-Scale Surface

The corpus contains {census["reported_observations"]:,} reported observations and
{census["confirmed_wt_fixed_dose_structures"]:,} confirmed-WT fixed-dose structure labels.
Those fixed-dose labels come from assays with heterogeneous concentrations and inactivity
rules. They remain an assay-standardization queue; V10 does not pretend that every negative
means IC50 >30 uM or that every positive means IC50 <1 uM.

## Scope

This build is ligand-only and train-partition-only. It does not open repository validation
or test labels, does not perform prospective validation, and does not yet use receptor
structures. It is a research prototype, not a clinical or synthesis decision system.
"""
    (paths.output / "REPORT.md").write_text(report)


def _html() -> str:
    return r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>hERG V10 Safety Router</title>
<style>
:root{--ink:#101828;--muted:#667085;--line:#e4e7ec;--blue:#155eef;--safe:#087443;--mid:#b54708;--risk:#b42318;--bg:#f7f9fc}
*{box-sizing:border-box}body{margin:0;background:var(--bg);font:15px/1.5 Inter,ui-sans-serif,system-ui;color:var(--ink)}
.wrap{max-width:1180px;margin:auto;padding:38px 24px 80px}.top{display:flex;justify-content:space-between;gap:24px;align-items:flex-end}
h1{font-size:42px;line-height:1.05;margin:0}.subtitle{color:var(--muted);max-width:700px}.badge{padding:7px 11px;border-radius:999px;background:#fff;border:1px solid var(--line);font-weight:700}
.panel{background:#fff;border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 8px 30px #1018280a;margin-top:22px}
label{font-weight:750;display:block;margin-bottom:8px}textarea{width:100%;min-height:92px;border:1px solid #b9c0cb;border-radius:12px;padding:14px;font:14px ui-monospace,monospace}
.actions{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}button{border:0;border-radius:10px;background:var(--blue);color:white;padding:11px 18px;font-weight:750;cursor:pointer}.example{background:#eef4ff;color:#1849a9}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.metric{border:1px solid var(--line);border-radius:14px;padding:18px}.metric .k{font-size:13px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.04em}.metric .v{font-size:30px;font-weight:800;margin-top:4px}
.result{display:none}.route{display:flex;align-items:center;gap:16px}.tier{font-size:34px;font-weight:850}.safe{color:var(--safe)}.moderate{color:var(--mid)}.potent{color:var(--risk)}
.bar{height:10px;background:#eef0f3;border-radius:10px;overflow:hidden;margin-top:6px}.fill{height:100%}.p-row{display:grid;grid-template-columns:82px 1fr 52px;gap:10px;align-items:center;margin:9px 0}
.callout{border-left:4px solid var(--blue);padding:10px 14px;background:#f5f8ff;border-radius:0 10px 10px 0}.warn{border-left-color:var(--mid);background:#fff8f1}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px 8px;border-bottom:1px solid var(--line)}th{font-size:12px;text-transform:uppercase;color:var(--muted)}
.small{font-size:13px;color:var(--muted)}.section-title{font-size:21px;margin:0 0 14px}.error{color:var(--risk);font-weight:700}.footer{color:var(--muted);margin-top:25px;font-size:13px}
@media(max-width:800px){.grid{grid-template-columns:1fr}.top{display:block}h1{font-size:34px}}
</style></head><body><main class="wrap">
<div class="top"><div><h1>hERG V10 Safety Router</h1><p class="subtitle">Three-class liability routing followed by quantitative IC50 estimation. Ligand-only, scaffold-validated internal evidence, with explicit applicability and endpoint limits.</p></div><div class="badge">Research Prototype</div></div>
<section class="panel"><label for="smiles">Compound SMILES</label><textarea id="smiles" spellcheck="false" placeholder="Paste one SMILES string"></textarea><div class="actions"><button id="predict">Run V10</button><button class="example" data-s="CN1CCC[C@H]1COc1ccc2[nH]c(=O)ccc2c1">Example A</button><button class="example" data-s="CC(C)NCC(O)COc1cccc2ccccc12">Example B</button><span id="status" class="small"></span></div></section>
<section id="result" class="result">
<div class="panel"><div class="route"><div><div class="small">STAGE 1 ROUTE</div><div id="tier" class="tier"></div></div><div id="routeText"></div></div><div id="probabilities"></div></div>
<div class="panel"><h2 class="section-title">Stage 2 Quantitative Estimate</h2><div class="grid"><div class="metric"><div class="k">Predicted pIC50</div><div class="v" id="pic50"></div></div><div class="metric"><div class="k">Predicted IC50</div><div class="v" id="ic50"></div><div class="small" id="interval"></div></div><div class="metric"><div class="k">Applicability</div><div class="v" id="domain"></div><div class="small" id="similarity"></div></div></div></div>
<div class="panel"><h2 class="section-title">IC10 And IC30 Hypothesis Layer</h2><div class="grid"><div class="metric"><div class="k">IC10, Hill Slope 1</div><div class="v" id="ic10"></div><div class="small" id="ic10range"></div></div><div class="metric"><div class="k">IC30, Hill Slope 1</div><div class="v" id="ic30"></div><div class="small" id="ic30range"></div></div><div class="callout warn"><b>Derived, not trained.</b><br>The canonical corpus contains no direct IC10 or IC30 records. These values are Hill-equation transformations of predicted IC50, with slope sensitivity shown.</div></div></div>
<div class="panel"><h2 class="section-title">Chemical Context And Nearest Training Analogs</h2><div id="descriptors" class="grid"></div><table><thead><tr><th>Training Structure</th><th>SMILES</th><th>Observed pIC50</th><th>Tanimoto</th></tr></thead><tbody id="analogs"></tbody></table></div>
<div class="panel"><h2 class="section-title">How To Read This Result</h2><div id="explanation"></div></div>
</section>
<section class="panel"><h2 class="section-title">Data Contract</h2><div id="census" class="grid"></div><p class="callout warn"><b>The broad fixed-dose surface is not yet a valid three-class dataset.</b> Assay concentrations and inactivity thresholds must be adjudicated before 339,373 fixed-dose structures can be mapped to the 1 uM and 30 uM boundaries.</p></section>
<p class="footer">Not for clinical use. V10 is an internal, ligand-only research prototype. Repository validation and test labels remain sealed; receptor-aware modeling and prospective validation are future work.</p>
</main><script>
const $=id=>document.getElementById(id); const fmt=x=>Number(x).toFixed(x<10?2:1);
async function loadInfo(){const r=await fetch('/api/model-info');const d=await r.json();const c=d.dataset_census; $('census').innerHTML=[['Reported Observations',c.reported_observations.toLocaleString()],['Broad Fixed-Dose Structures',c.confirmed_wt_fixed_dose_structures.toLocaleString()],['Direct Tier Train Structures',c.v10_direct_router_train_structures.toLocaleString()],['Direct IC10 / IC30 Records',`${c.native_ic10_observations} / ${c.native_ic30_observations}`]].map(x=>`<div class="metric"><div class="k">${x[0]}</div><div class="v">${x[1]}</div></div>`).join('')}
function probRow(name,value,color){return `<div class="p-row"><b>${name}</b><div class="bar"><div class="fill" style="width:${100*value}%;background:${color}"></div></div><span>${(100*value).toFixed(1)}%</span></div>`}
async function predict(){const smiles=$('smiles').value.trim();if(!smiles)return;$('status').textContent='Calculating descriptors, fingerprints, routing, and applicability...';try{const r=await fetch('/api/predict',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({smiles})});const d=await r.json();if(!r.ok)throw new Error(d.error||'Prediction failed');$('result').style.display='block';$('tier').textContent=d.route.tier;$('tier').className='tier '+d.route.tier.toLowerCase();$('routeText').textContent=d.route.interpretation;$('probabilities').innerHTML=probRow('Safe',d.route.probabilities.Safe,'#12b76a')+probRow('Moderate',d.route.probabilities.Moderate,'#f79009')+probRow('Potent',d.route.probabilities.Potent,'#f04438');$('pic50').textContent=d.quantitative.predicted_pic50.toFixed(3);$('ic50').textContent=fmt(d.quantitative.predicted_ic50_um)+' µM';$('interval').textContent=`90% residual interval: ${fmt(d.quantitative.ic50_interval90_um[0])}-${fmt(d.quantitative.ic50_interval90_um[1])} µM`;$('domain').textContent=d.applicability.label;$('similarity').textContent=`Maximum train Tanimoto ${d.applicability.maximum_train_tanimoto.toFixed(3)}`;$('ic10').textContent=fmt(d.derived_icx.ic10_um)+' µM';$('ic30').textContent=fmt(d.derived_icx.ic30_um)+' µM';$('ic10range').textContent=`Hill slope 0.7-1.3: ${fmt(d.derived_icx.ic10_sensitivity_um[0])}-${fmt(d.derived_icx.ic10_sensitivity_um[1])} µM`;$('ic30range').textContent=`Hill slope 0.7-1.3: ${fmt(d.derived_icx.ic30_sensitivity_um[0])}-${fmt(d.derived_icx.ic30_sensitivity_um[1])} µM`;$('descriptors').innerHTML=Object.entries(d.chemical_context).map(([k,v])=>`<div class="metric"><div class="k">${k.replaceAll('_',' ')}</div><div class="v">${typeof v==='number'?fmt(v):v}</div></div>`).join('');$('analogs').innerHTML=d.applicability.nearest_analogs.map(a=>`<tr><td>${a.structure_id}</td><td class="small">${a.standardized_smiles||'unavailable'}</td><td>${a.target_pic50.toFixed(3)}</td><td>${a.tanimoto.toFixed(3)}</td></tr>`).join('');$('explanation').innerHTML=d.explanation.map(x=>`<p class="callout">${x}</p>`).join('');$('status').textContent='Complete';$('result').scrollIntoView({behavior:'smooth'})}catch(e){$('status').innerHTML=`<span class="error">${e.message}</span>`}}
$('predict').onclick=predict;document.querySelectorAll('.example').forEach(b=>b.onclick=()=>{$('smiles').value=b.dataset.s;predict()});loadInfo();
</script></body></html>"""


def _build(paths: Paths, workers: int, *, force: bool = False) -> dict[str, Any]:
    if (paths.output / "manifest.json").exists() and not force:
        return _validate(paths.output)
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "models").mkdir(exist_ok=True)
    router_metrics, router_oof = _train_router(paths, workers)
    regression_metrics, regression_oof = _regression_evidence(paths)
    router_from_regression = np.eye(3)[tier_index(regression_oof.predicted_pic50.to_numpy(float))]
    router_metrics["v9_regression_threshold_comparator"] = _classification_metrics(
        tier_index(regression_oof.observed_pic50.to_numpy(float)), router_from_regression
    )
    census, queue = _dataset_census(paths, router_metrics["class_counts"])
    _atomic_parquet(paths.output / "evidence/router_nested_oof.parquet", router_oof)
    _atomic_parquet(paths.output / "evidence/regression_nested_oof.parquet", regression_oof)
    _atomic_parquet(paths.output / "evidence/assay_standardization_queue_summary.parquet", queue)
    shutil.copy2(
        paths.v9 / "final_model/molecular_model.joblib",
        paths.output / "models/ic50_regressor.joblib",
    )
    shutil.copy2(
        paths.v9 / "final_model/feature_preprocessing_schema.json",
        paths.output / "models/ic50_feature_schema.json",
    )
    router_bundle = joblib.load(paths.output / "models/router_classifier.joblib")
    _reference_artifacts(paths, router_bundle["feature_columns"])
    metrics = {"router": router_metrics, "regression": regression_metrics}
    info = _model_info(paths, metrics, census)
    _atomic_json(paths.output / "model_info.json", info, "model_info_sha256")
    (paths.output / "app.html").write_text(_html())
    _write_report(paths, info)
    artifacts = [
        _binding(path, role, root=paths.output)
        for path, role in [
            (paths.output / "models/router_classifier.joblib", "router_classifier"),
            (paths.output / "models/ic50_regressor.joblib", "ic50_regressor"),
            (paths.output / "models/ic50_feature_schema.json", "ic50_feature_schema"),
            (paths.output / "reference/training_reference.parquet", "training_reference"),
            (paths.output / "reference/training_morgan_bits.npz", "training_morgan_bits"),
            (paths.output / "evidence/router_nested_oof.parquet", "router_nested_oof"),
            (paths.output / "evidence/regression_nested_oof.parquet", "regression_nested_oof"),
            (
                paths.output / "evidence/assay_standardization_queue_summary.parquet",
                "assay_standardization_queue_summary",
            ),
            (paths.output / "model_info.json", "model_info"),
            (paths.output / "app.html", "local_ui"),
            (paths.output / "REPORT.md", "report"),
        ]
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "artifacts": artifacts,
        "inputs": [
            _binding(paths.v9 / "prepared/training_matrix.parquet", "v9_training_matrix"),
            _binding(paths.v9 / "prepared/fixed_nested_scaffold_splits.parquet", "v9_folds"),
            _binding(paths.v9 / "analysis/nested_oof_predictions.parquet", "v9_nested_oof"),
            _binding(paths.surface, "herg_training_surface"),
            _binding(paths.broad, "confirmed_wt_fixed_dose_surface"),
            _binding(Path(__file__), "implementation"),
        ],
        "scientific_contract": info["scientific_scope"],
    }
    _atomic_json(paths.output / "manifest.json", manifest, "manifest_sha256")
    return _validate(paths.output)


def _validate(output: Path) -> dict[str, Any]:
    output = output.resolve()
    manifest = _read_json(output / "manifest.json", "manifest_sha256")
    for artifact in manifest["artifacts"]:
        _verify_binding(artifact, root=output)
    for source in manifest["inputs"]:
        _verify_binding(source)
    if manifest["scientific_contract"]["repository_validation_labels_opened"]:
        raise V10Error("repository validation labels must remain sealed")
    if manifest["scientific_contract"]["repository_test_labels_opened"]:
        raise V10Error("repository test labels must remain sealed")
    router = pd.read_parquet(output / "evidence/router_nested_oof.parquet")
    regression = pd.read_parquet(output / "evidence/regression_nested_oof.parquet")
    if len(router) != 18_801 or router.structure_id.nunique() != 18_801:
        raise V10Error("router evidence is not exactly 18,801 unique structures")
    if len(regression) != 18_801 or regression.structure_id.nunique() != 18_801:
        raise V10Error("regression evidence is not exactly 18,801 unique structures")
    if set(router.outer_fold.unique()) != set(range(5)):
        raise V10Error("router evidence is not five-fold")
    info = _read_json(output / "model_info.json", "model_info_sha256")
    if info["dataset_census"]["native_ic10_observations"] != 0:
        raise V10Error("IC10 contract changed; direct endpoint handling must be reviewed")
    if info["dataset_census"]["native_ic30_observations"] != 0:
        raise V10Error("IC30 contract changed; direct endpoint handling must be reviewed")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "artifacts_verified": len(manifest["artifacts"]),
        "router_oof_rows": len(router),
        "regression_oof_rows": len(regression),
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "output_root": str(output),
    }


class Predictor:
    def __init__(self, root: Path):
        self.root = root.resolve()
        _validate(self.root)
        self.router = joblib.load(self.root / "models/router_classifier.joblib")
        self.regressor = joblib.load(self.root / "models/ic50_regressor.joblib")
        self.info = _read_json(self.root / "model_info.json", "model_info_sha256")
        self.reference = pd.read_parquet(self.root / "reference/training_reference.parquet")
        packed = np.load(self.root / "reference/training_morgan_bits.npz")["packed"]
        self.reference_bits = np.unpackbits(packed, axis=1, bitorder="big")[:, :MORGAN_BITS].astype(bool)
        self.reference_counts = self.reference_bits.sum(axis=1)
        residual = pd.read_parquet(self.root / "evidence/regression_nested_oof.parquet")
        self.q90 = float(np.quantile(residual.absolute_error, 0.9))

    def predict(self, smiles: str) -> dict[str, Any]:
        frame, molecule = _feature_frame(smiles, self.router["feature_columns"])
        raw = self.router["model"].predict_proba(_clean(frame, self.router["feature_columns"]))
        probabilities = self.router["calibrator"].predict_proba(np.log(np.clip(raw, 1e-7, 1.0)))[0]
        tier = int(np.argmax(probabilities))
        regression_frame, _ = _feature_frame(smiles, self.regressor["feature_columns"])
        pic50 = float(
            self.regressor["model"].predict(_clean(regression_frame, self.regressor["feature_columns"]))[0]
        )
        ic50 = ic50_um_from_pic50(pic50)
        pic50_low = pic50 - self.q90
        pic50_high = pic50 + self.q90
        interval = sorted([ic50_um_from_pic50(pic50_high), ic50_um_from_pic50(pic50_low)])
        bits = frame[[f"morgan__{i:04d}" for i in range(MORGAN_BITS)]].to_numpy(bool)[0]
        intersection = np.logical_and(self.reference_bits, bits).sum(axis=1)
        union = self.reference_counts + int(bits.sum()) - intersection
        similarity = np.divide(
            intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0
        )
        top = np.argsort(similarity)[-3:][::-1]
        maximum = float(similarity[top[0]])
        domain = "Strong" if maximum >= 0.7 else "Moderate" if maximum >= 0.5 else "Extrapolative"
        ic10_values = [hill_icx_um(ic50, 0.1, slope) for slope in (0.7, 1.3)]
        ic30_values = [hill_icx_um(ic50, 0.3, slope) for slope in (0.7, 1.3)]
        context = {
            "molecular_weight": float(Descriptors.MolWt(molecule)),
            "logp": float(Crippen.MolLogP(molecule)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
            "h_bond_donors": int(Lipinski.NumHDonors(molecule)),
            "h_bond_acceptors": int(Lipinski.NumHAcceptors(molecule)),
            "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
            "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
            "formal_charge": int(Chem.GetFormalCharge(molecule)),
        }
        explanation = [
            (
                f"The direct router assigns {CLASS_NAMES[tier]} with "
                f"{probabilities[tier]:.1%} calibrated internal confidence."
            ),
            (
                f"The closest training analog has Morgan Tanimoto {maximum:.3f}; "
                f"this is labeled {domain.lower()} applicability."
            ),
            (
                "RDKit properties and Morgan substructures are displayed as chemical context. "
                "They are associations used by the model, not causal receptor mechanisms."
            ),
        ]
        if tier != 1:
            explanation.append(
                "The exact IC50 is secondary outside the Moderate route; the class decision is "
                "the intended Stage 1 output."
            )
        return {
            "smiles": smiles,
            "route": {
                "tier": CLASS_NAMES[tier],
                "probabilities": {CLASS_NAMES[i]: float(probabilities[i]) for i in range(3)},
                "interpretation": [
                    "Likely clean at 30 uM"
                    if tier == 0
                    else "Quantitative IC50 is decision-relevant"
                    if tier == 1
                    else "Likely sub-micromolar hERG liability"
                ][0],
            },
            "quantitative": {
                "predicted_pic50": pic50,
                "predicted_ic50_um": ic50,
                "pic50_residual_interval90": [pic50_low, pic50_high],
                "ic50_interval90_um": interval,
                "directly_trained_endpoint": "IC50/pIC50",
            },
            "derived_icx": {
                "ic10_um": hill_icx_um(ic50, 0.1, 1.0),
                "ic30_um": hill_icx_um(ic50, 0.3, 1.0),
                "ic10_sensitivity_um": sorted(ic10_values),
                "ic30_sensitivity_um": sorted(ic30_values),
                "hill_slope_range": [0.7, 1.3],
                "status": "derived_from_predicted_ic50_not_directly_trained",
            },
            "applicability": {
                "label": domain,
                "maximum_train_tanimoto": maximum,
                "nearest_analogs": [
                    {
                        "structure_id": str(self.reference.iloc[index].structure_id),
                        "standardized_smiles": self.reference.iloc[index].standardized_smiles,
                        "target_pic50": float(self.reference.iloc[index].target_pic50),
                        "tanimoto": float(similarity[index]),
                    }
                    for index in top
                ],
            },
            "chemical_context": context,
            "explanation": explanation,
            "scope": self.info["scientific_scope"],
        }


def _handler(predictor: Predictor) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status.value)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                body = (predictor.root / "app.html").read_bytes()
                self.send_response(HTTPStatus.OK.value)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/model-info":
                self._json(predictor.info)
            elif self.path == "/api/health":
                self._json({"status": "ok", "schema_version": SCHEMA_VERSION})
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/predict":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                if length <= 0 or length > 100_000:
                    raise V10Error("invalid request size")
                data = json.loads(self.rfile.read(length))
                smiles = str(data.get("smiles", "")).strip()
                if not smiles:
                    raise V10Error("SMILES is required")
                self._json(predictor.predict(smiles))
            except (V10Error, ValueError, TypeError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("[V10 UI] " + format % args + "\n")

    return Handler


def _paths(args: argparse.Namespace) -> Paths:
    repo = args.repo_root.resolve()
    return Paths(
        repo=repo,
        v9=(repo / args.v9_root).resolve() if not args.v9_root.is_absolute() else args.v9_root.resolve(),
        surface=(repo / args.surface).resolve() if not args.surface.is_absolute() else args.surface.resolve(),
        broad=(repo / args.broad).resolve() if not args.broad.is_absolute() else args.broad.resolve(),
        output=(repo / args.output_root).resolve()
        if not args.output_root.is_absolute()
        else args.output_root.resolve(),
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--v9-root", type=Path, default=DEFAULT_V9)
    parser.add_argument("--surface", type=Path, default=DEFAULT_SURFACE)
    parser.add_argument("--broad", type=Path, default=DEFAULT_BROAD)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    _add_common(build)
    build.add_argument("--workers", type=int, default=6, choices=range(1, 7))
    build.add_argument("--force", action="store_true")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-root", type=Path, required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--model-root", type=Path, required=True)
    predict.add_argument("--smiles", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--model-root", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    launch = subparsers.add_parser("launch")
    _add_common(launch)
    launch.add_argument("--workers", type=int, default=6, choices=range(1, 7))
    launch.add_argument("--host", default="127.0.0.1")
    launch.add_argument("--port", type=int, default=8787)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate":
        result = _validate(args.output_root)
    elif args.command == "predict":
        result = Predictor(args.model_root).predict(args.smiles)
    elif args.command == "build":
        result = _build(_paths(args), args.workers, force=args.force)
    elif args.command in {"serve", "launch"}:
        if args.command == "launch":
            paths = _paths(args)
            print(json.dumps(_build(paths, args.workers), indent=2), flush=True)
            root = paths.output
        else:
            root = args.model_root.resolve()
        predictor = Predictor(root)
        server = ThreadingHTTPServer((args.host, args.port), _handler(predictor))
        print(f"hERG V10 UI: http://{args.host}:{args.port}", flush=True)
        print("Press Ctrl-C to stop the UI server.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
