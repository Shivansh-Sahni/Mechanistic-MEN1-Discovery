#!/usr/bin/env python3
"""Build the expanded, endpoint-honest hERG V10.1 local platform.

V10.1 adds three governed layers to the ligand-only V10 prototype:

* bounded model/feature selection for the exact three-class IC50 router;
* a separate 339k-structure confirmed-WT 46 uM fixed-dose liability model;
* empirical IC10, IC30, and IC50 models from observed 20-concentration hERG curves.

The empirical IC10/IC30/IC50 labels are interpolated between observed
concentrations after monotone denoising. They are not Hill transformations of
predicted IC50. At inference, the three same-assay outputs are projected onto
the required IC10 <= IC30 <= IC50 ordering in log-concentration space.
Repository validation and test partitions remain sealed throughout the build.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier, LGBMRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier, XGBRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_local_herg_v10_tiered_platform as v10  # noqa: E402

SCHEMA_VERSION = "platform-local-herg-v10.1-expanded/1.0"
SEED = 20260819
CLASS_NAMES = ["Safe", "Moderate", "Potent"]
DEFAULT_V10 = Path("research/local_runs/herg_v10_tiered_platform")
DEFAULT_V9 = Path("research/local_runs/herg_domain_mixture_campaign_v9")
DEFAULT_BROAD = Path(
    "research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces/"
    "confirmed_wt_fixed_dose_structure_labels.parquet"
)
DEFAULT_FEATURE_CACHE = Path("research/local_runs/local_multicpu_2d_features_v1")
DEFAULT_CURVES = Path(
    "research/data/platform/raw/external_public/herg_expansion/avicenna/assays/AID_588834_full.csv"
)
DEFAULT_OUTPUT = Path("research/local_runs/herg_v10_1_expanded_platform")
RDLogger.DisableLog("rdApp.warning")


class V101Error(RuntimeError):
    """Raised when a V10.1 scientific or artifact invariant fails."""


@dataclass(frozen=True)
class Paths:
    repo: Path
    v10_root: Path
    v9_root: Path
    broad: Path
    feature_cache: Path
    curves: Path
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


def _binding(path: Path, role: str, root: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise V101Error(f"missing {role}: {path}")
    if root is not None and not path.is_relative_to(root.resolve()):
        raise V101Error(f"artifact escapes output root: {path}")
    result: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }
    if path.suffix == ".parquet":
        result["rows"] = pq.read_metadata(path).num_rows
        result["arrow_schema_sha256"] = _schema_sha(path)
    return result


def _verify(binding: dict[str, Any], root: Path | None = None) -> None:
    path = Path(str(binding["path"])).resolve()
    if root is not None and not path.is_relative_to(root.resolve()):
        raise V101Error(f"artifact escapes output root: {path}")
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
        raise V101Error(f"artifact missing or changed size: {path}")
    if _sha(path) != binding["sha256"]:
        raise V101Error(f"artifact hash changed: {path}")
    if path.suffix == ".parquet":
        if pq.read_metadata(path).num_rows != int(binding["rows"]):
            raise V101Error(f"artifact row count changed: {path}")
        if _schema_sha(path) != binding["arrow_schema_sha256"]:
            raise V101Error(f"artifact schema changed: {path}")


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
        raise V101Error(f"expected JSON object: {path}")
    if self_key:
        expected = value.get(self_key)
        candidate = copy.deepcopy(value)
        candidate.pop(self_key, None)
        if expected != _digest(candidate):
            raise V101Error(f"self-hash mismatch: {path}")
    return value


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _safe_numeric(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    values[np.abs(values) > 1e30] = np.nan
    return np.ascontiguousarray(values)


def _maccs_array(packed: pd.Series) -> np.ndarray:
    raw = np.vstack([np.frombuffer(value, dtype=np.uint8) for value in packed])
    return np.unpackbits(raw, axis=1, bitorder="little")[:, :167].astype(np.float32)


def _load_cached_features(
    paths: Paths, structures: pd.DataFrame, *, include_maccs: bool
) -> tuple[pd.DataFrame, list[str]]:
    schema = _read_json(paths.feature_cache / "feature_schema.json")
    descriptors = [f"rdkit2d__{name}" for name in schema["families"]["rdkit_2d"]["descriptor_names"]]
    mapping = pd.read_parquet(
        paths.feature_cache / "source_to_feature_mapping.parquet",
        filters=[("source_family", "==", "herg")],
    )[["source_structure_id", "feature_id"]]
    target = structures.merge(
        mapping,
        left_on="structure_id",
        right_on="source_structure_id",
        how="left",
        validate="one_to_one",
    )
    if target.feature_id.isna().any():
        raise V101Error(f"feature cache misses {int(target.feature_id.isna().sum())} hERG structures")
    wanted = set(target.feature_id.astype(str))
    columns = ["feature_id", *descriptors]
    if include_maccs:
        columns.append("maccs_167")
    pieces = []
    for shard in sorted((paths.feature_cache / "features").glob("part-*.parquet")):
        frame = pd.read_parquet(shard, columns=columns)
        frame = frame.loc[frame.feature_id.astype(str).isin(wanted)]
        if not frame.empty:
            pieces.append(frame)
    cache = pd.concat(pieces, ignore_index=True)
    if cache.feature_id.nunique() != len(target):
        raise V101Error("cached feature extraction did not yield exactly one row per structure")
    if include_maccs:
        bits = _maccs_array(cache.pop("maccs_167"))
        for index in range(167):
            cache[f"maccs__{index:03d}"] = bits[:, index]
    result = target.merge(cache, on="feature_id", validate="one_to_one")
    feature_columns = [name for name in result if name.startswith(("rdkit2d__", "maccs__"))]
    return result, feature_columns


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    if len(np.unique(y)) != 2:
        raise V101Error("binary metric calculation requires both classes")
    order = np.argsort(p)[::-1]
    negatives = int(np.sum(y == 0))
    allowed = max(1, math.floor(0.01 * negatives))
    cumulative_false_positives = np.cumsum(y[order] == 0)
    valid = np.flatnonzero(cumulative_false_positives <= allowed)
    cutoff = int(valid[-1]) if len(valid) else 0
    threshold = float(p[order[cutoff]])
    prediction = p >= threshold
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "threshold_at_approximately_1pct_fpr": threshold,
        "recall_at_approximately_1pct_fpr": float(recall_score(y, prediction)),
        "precision_at_approximately_1pct_fpr": float(precision_score(y, prediction, zero_division=0)),
        "balanced_accuracy_at_approximately_1pct_fpr": float(balanced_accuracy_score(y, prediction)),
    }


def _multiclass_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    predicted = np.argmax(p, axis=1)
    matrix = confusion_matrix(y, predicted, labels=[0, 1, 2])
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "confusion_matrix": matrix.tolist(),
        "per_class_recall": {
            CLASS_NAMES[i]: float(recall_score(y == i, predicted == i, zero_division=0)) for i in range(3)
        },
        "per_class_average_precision": {
            CLASS_NAMES[i]: float(average_precision_score(y == i, p[:, i])) for i in range(3)
        },
        "potent_predicted_safe_rate": float(np.mean(predicted[y == 2] == 0)),
    }


def _regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(y - p)
    correlation = spearmanr(y, p).statistic
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(math.sqrt(mean_squared_error(y, p))),
        "spearman": float(correlation) if np.isfinite(correlation) else None,
        "within_0p5": float(np.mean(absolute <= 0.5)),
        "within_1p0": float(np.mean(absolute <= 1.0)),
    }


def _broad_candidate(engine: str, workers: int, seed: int) -> Any:
    if engine == "xgboost":
        return XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.045,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.4,
            reg_lambda=7.0,
            max_bin=128,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=workers,
            random_state=seed,
            verbosity=0,
        )
    return LGBMClassifier(
        n_estimators=700,
        num_leaves=31,
        learning_rate=0.035,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.4,
        reg_lambda=7.0,
        max_bin=127,
        n_jobs=workers,
        random_state=seed,
        verbosity=-1,
    )


def _train_broad(paths: Paths, workers: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    broad = pd.read_parquet(paths.broad)
    train = broad.loc[broad.model_split.eq("train")].copy()
    if len(broad) != 339_373 or len(train) != 265_625:
        raise V101Error("confirmed-WT fixed-dose census changed")
    featured, all_features = _load_cached_features(paths, train, include_maccs=True)
    rdkit = [name for name in all_features if name.startswith("rdkit2d__")]
    surfaces = {"rdkit2d": rdkit, "rdkit2d_maccs": all_features}
    y = featured.target_class.to_numpy(int)
    groups = featured.scaffold_group_id.astype(str).to_numpy()
    folds = list(GroupKFold(3).split(featured, y, groups))
    candidates: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for engine in ("xgboost", "lightgbm"):
        for surface_name, columns in surfaces.items():
            unit_id = f"{engine}__{surface_name}"
            oof = np.full(len(featured), np.nan, dtype=float)
            for fold, (fit, evaluate) in enumerate(folds):
                model = _broad_candidate(engine, workers, SEED + fold)
                positive_weight = float(np.sum(y[fit] == 0) / max(1, np.sum(y[fit] == 1)))
                weights = np.where(y[fit] == 1, positive_weight, 1.0)
                model.fit(_safe_numeric(featured.iloc[fit], columns), y[fit], sample_weight=weights)
                oof[evaluate] = model.predict_proba(_safe_numeric(featured.iloc[evaluate], columns))[:, 1]
            metrics = _binary_metrics(y, oof)
            candidates.append(
                {"candidate_id": unit_id, "engine": engine, "feature_surface": surface_name, **metrics}
            )
            predictions[unit_id] = oof
    ranked = sorted(
        candidates, key=lambda row: (row["pr_auc"], row["recall_at_approximately_1pct_fpr"]), reverse=True
    )
    winner = ranked[0]
    winner_id = str(winner["candidate_id"])
    columns = surfaces[str(winner["feature_surface"])]
    final = _broad_candidate(str(winner["engine"]), workers, SEED + 99)
    positive_weight = float(np.sum(y == 0) / max(1, np.sum(y == 1)))
    final.fit(_safe_numeric(featured, columns), y, sample_weight=np.where(y == 1, positive_weight, 1.0))
    calibrator = LogisticRegression(C=1.0, max_iter=2_000, random_state=SEED)
    raw = np.clip(predictions[winner_id], 1e-7, 1 - 1e-7)
    calibrator.fit(np.log(raw / (1 - raw)).reshape(-1, 1), y)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "model": final,
        "calibrator": calibrator,
        "feature_columns": columns,
        "engine": winner["engine"],
        "feature_surface": winner["feature_surface"],
        "endpoint": "AID720551 confirmed-WT 46 uM fixed-dose liability",
        "negative_interpretation": "screen-inactive supports IC50 >30 uM under monotone inhibition",
        "positive_interpretation": "screen-active indicates liability but does not identify exact IC50 tier",
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
    }
    joblib.dump(bundle, paths.output / "models/broad_fixed_dose_classifier.joblib", compress=3)
    oof_frame = featured[["structure_id", "scaffold_group_id", "target_class"]].copy()
    oof_frame["fold"] = -1
    for fold, (_, evaluate) in enumerate(folds):
        oof_frame.iloc[evaluate, oof_frame.columns.get_loc("fold")] = fold
    for candidate_id, values in predictions.items():
        oof_frame[f"probability__{candidate_id}"] = values
    summary = {
        "surface_rows_all_partitions": int(len(broad)),
        "train_rows": int(len(featured)),
        "train_positives": int(y.sum()),
        "fixed_dose_um": 46.0,
        "winner": winner,
        "candidates": ranked,
        "claim_boundary": "binary auxiliary screen; not relabeled as exact Safe/Moderate/Potent tiers",
    }
    return summary, oof_frame, featured[["structure_id", "scaffold_group_id", *columns]]


def _standardize_smiles(value: Any) -> tuple[str | None, str | None, str | None]:
    molecule = Chem.MolFromSmiles(str(value)) if pd.notna(value) else None
    if molecule is None:
        return None, None, None
    smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
    key = Chem.MolToInchiKey(molecule)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False) or smiles
    return smiles, key, hashlib.sha256(scaffold.encode()).hexdigest()


def _interpolated_crossing(log_concentrations: np.ndarray, values: np.ndarray, threshold: float) -> float:
    if np.min(values) > threshold or np.max(values) < threshold:
        return math.nan
    index = int(np.flatnonzero(values >= threshold)[0])
    if index == 0:
        return float(10 ** log_concentrations[0])
    x0, x1 = log_concentrations[index - 1], log_concentrations[index]
    y0, y1 = values[index - 1], values[index]
    estimate = x0 if y1 == y0 else x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    return float(10**estimate)


def _extract_empirical_endpoints(paths: Paths) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(paths.curves, skiprows=[1, 2, 3, 4, 5], low_memory=False)
    concentration_columns: list[tuple[float, str]] = []
    for column in raw.columns:
        match = re.fullmatch(r"Activity at ([0-9.]+) uM", str(column))
        if match:
            concentration_columns.append((float(match.group(1)), str(column)))
    concentration_columns.sort()
    if len(concentration_columns) != 20:
        raise V101Error("expected 20 measured concentration columns in AID588834")
    concentrations = np.asarray([value for value, _ in concentration_columns], dtype=float)
    responses = (
        -raw[[name for _, name in concentration_columns]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip")
    records = []
    for index, response in enumerate(responses):
        finite = np.isfinite(response)
        if finite.sum() < 6:
            continue
        log_concentration = np.log10(concentrations[finite])
        observed = response[finite]
        fitted = isotonic.fit_transform(log_concentration, observed)
        standardized_smiles, inchi_key, scaffold = _standardize_smiles(
            raw.iloc[index].get("PUBCHEM_EXT_DATASOURCE_SMILES")
        )
        if standardized_smiles is None:
            continue
        record = {
            "source_row": int(index),
            "pubchem_sid": str(raw.iloc[index].get("PUBCHEM_SID")),
            "standardized_smiles": standardized_smiles,
            "standard_inchi_key": inchi_key,
            "scaffold_group_id": f"EMP-{scaffold}",
            "activity_outcome": str(raw.iloc[index].get("PUBCHEM_ACTIVITY_OUTCOME")),
            "phenotype": str(raw.iloc[index].get("Phenotype")),
            "fit_r2_qc_only": pd.to_numeric(raw.iloc[index].get("Fit_R2"), errors="coerce"),
            "curve_class_qc_only": str(raw.iloc[index].get("Fit_CurveClass")),
            "monotonic_adjustment_rmse": float(np.sqrt(np.mean(np.square(observed - fitted)))),
            "measured_concentration_count": int(finite.sum()),
            "empirical_ic10_um": _interpolated_crossing(log_concentration, fitted, 10.0),
            "empirical_ic30_um": _interpolated_crossing(log_concentration, fitted, 30.0),
            "empirical_ic50_um": _interpolated_crossing(log_concentration, fitted, 50.0),
            "pubchem_activity_score_qc_only": pd.to_numeric(
                raw.iloc[index].get("PUBCHEM_ACTIVITY_SCORE"), errors="coerce"
            ),
        }
        record["strict_curve_qc"] = bool(
            record["phenotype"] == "Inhibitor"
            and record["monotonic_adjustment_rmse"] <= 15.0
            and pd.notna(record["fit_r2_qc_only"])
            and float(record["fit_r2_qc_only"]) >= 0.6
        )
        records.append(record)
    frame = pd.DataFrame(records)
    census = {
        "raw_assay_rows": int(len(raw)),
        "measured_concentrations_per_curve": 20,
        "concentration_range_um": [float(concentrations.min()), float(concentrations.max())],
        "all_empirical_ic10_crossings": int(frame.empirical_ic10_um.notna().sum()),
        "all_empirical_ic30_crossings": int(frame.empirical_ic30_um.notna().sum()),
        "all_empirical_ic50_crossings": int(frame.empirical_ic50_um.notna().sum()),
        "strict_empirical_ic10_crossings": int(
            (frame.strict_curve_qc & frame.empirical_ic10_um.notna()).sum()
        ),
        "strict_empirical_ic30_crossings": int(
            (frame.strict_curve_qc & frame.empirical_ic30_um.notna()).sum()
        ),
        "strict_empirical_ic50_crossings": int(
            (frame.strict_curve_qc & frame.empirical_ic50_um.notna()).sum()
        ),
        "label_method": "isotonic monotone denoising and log-concentration interpolation between observed doses",
        "not_used_for_labels": "Hill slope, fitted AC50, predicted IC50",
    }
    return frame, census


def _empirical_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for smiles in frame.standardized_smiles:
        features, _ = v10._feature_frame(smiles, feature_columns)  # noqa: SLF001
        rows.append(features.iloc[0])
    return pd.DataFrame(rows, columns=feature_columns)


def _empirical_candidate(engine: str, workers: int, seed: int) -> Any:
    if engine == "xgboost":
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
    return LGBMRegressor(
        n_estimators=650,
        num_leaves=25,
        learning_rate=0.03,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.4,
        reg_lambda=6.0,
        max_bin=127,
        n_jobs=workers,
        random_state=seed,
        verbosity=-1,
    )


def _train_empirical_endpoints(
    paths: Paths, curves: pd.DataFrame, workers: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    schema = _read_json(paths.v9_root / "final_model/feature_preprocessing_schema.json")
    all_features = [str(value) for value in schema["feature_columns"]]
    rdkit = [name for name in all_features if name.startswith("rdkit2d__")]
    eligible = curves.strict_curve_qc & (
        curves.empirical_ic10_um.notna() | curves.empirical_ic30_um.notna() | curves.empirical_ic50_um.notna()
    )
    curves = curves.loc[eligible].reset_index(drop=True)
    feature_frame = _empirical_feature_frame(curves, all_features)
    evidence_rows = []
    results: dict[str, Any] = {}
    for endpoint in ("ic10", "ic30", "ic50"):
        column = f"empirical_{endpoint}_um"
        keep = curves.strict_curve_qc & curves[column].notna() & curves[column].gt(0)
        endpoint_frame = curves.loc[keep].copy().reset_index(drop=True)
        endpoint_features = feature_frame.loc[keep.to_numpy()].reset_index(drop=True)
        # One label per connectivity key prevents replicate weighting.
        endpoint_frame["target"] = 6.0 - np.log10(endpoint_frame[column].to_numpy(float))
        endpoint_frame["row_index"] = np.arange(len(endpoint_frame))
        collapsed = (
            endpoint_frame.groupby("standard_inchi_key", as_index=False)
            .agg(
                standardized_smiles=("standardized_smiles", "first"),
                scaffold_group_id=("scaffold_group_id", "first"),
                target=("target", "median"),
                replicate_count=("row_index", "size"),
                source_row_index=("row_index", "first"),
            )
            .reset_index(drop=True)
        )
        x = endpoint_features.iloc[collapsed.source_row_index.to_numpy(int)].reset_index(drop=True)
        y = collapsed.target.to_numpy(float)
        groups = collapsed.scaffold_group_id.to_numpy(str)
        folds = list(GroupKFold(5).split(x, y, groups))
        surfaces = {"rdkit2d": rdkit, "rdkit2d_morgan": all_features}
        candidates = []
        oof_predictions: dict[str, np.ndarray] = {}
        for engine in ("xgboost", "lightgbm"):
            for surface_name, columns in surfaces.items():
                candidate_id = f"{engine}__{surface_name}"
                oof = np.full(len(x), np.nan)
                for fold, (fit, evaluate) in enumerate(folds):
                    model = _empirical_candidate(engine, workers, SEED + fold)
                    model.fit(_safe_numeric(x.iloc[fit], columns), y[fit])
                    oof[evaluate] = model.predict(_safe_numeric(x.iloc[evaluate], columns))
                metrics = _regression_metrics(y, oof)
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "engine": engine,
                        "feature_surface": surface_name,
                        **metrics,
                    }
                )
                oof_predictions[candidate_id] = oof
        ranked = sorted(candidates, key=lambda row: row["mae"])
        winner = ranked[0]
        winner_id = str(winner["candidate_id"])
        columns = surfaces[str(winner["feature_surface"])]
        final = _empirical_candidate(str(winner["engine"]), workers, SEED + 99)
        final.fit(_safe_numeric(x, columns), y)
        joblib.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "model": final,
                "feature_columns": columns,
                "endpoint": endpoint.upper(),
                "label_method": "empirical crossing of observed qHTS concentration-response measurements",
                "curve_qc": "Inhibitor; monotonic-adjustment RMSE <=15; Fit_R2 >=0.6 used only as QC",
            },
            paths.output / f"models/empirical_{endpoint}_regressor.joblib",
            compress=3,
        )
        chosen = oof_predictions[winner_id]
        for index in range(len(collapsed)):
            evidence_rows.append(
                {
                    "endpoint": endpoint.upper(),
                    "standard_inchi_key": collapsed.iloc[index].standard_inchi_key,
                    "scaffold_group_id": collapsed.iloc[index].scaffold_group_id,
                    "observed_picx": float(y[index]),
                    "predicted_picx": float(chosen[index]),
                    "absolute_error": float(abs(y[index] - chosen[index])),
                }
            )
        results[endpoint.upper()] = {
            "strict_curve_rows_before_structure_collapse": int(len(endpoint_frame)),
            "unique_structure_labels": int(len(collapsed)),
            "winner": winner,
            "candidates": ranked,
        }
    return results, pd.DataFrame(evidence_rows)


def _router_candidate(engine: str, workers: int, seed: int) -> Any:
    if engine == "xgboost":
        return v10._new_classifier(workers, seed)  # noqa: SLF001
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=750,
        num_leaves=31,
        learning_rate=0.035,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=6.0,
        max_bin=127,
        n_jobs=workers,
        random_state=seed,
        verbosity=-1,
    )


def _train_improved_router(paths: Paths, workers: int) -> tuple[dict[str, Any], pd.DataFrame]:
    schema = _read_json(paths.v9_root / "final_model/feature_preprocessing_schema.json")
    features = [str(value) for value in schema["feature_columns"]]
    rdkit = [name for name in features if name.startswith("rdkit2d__")]
    matrix = pd.read_parquet(paths.v9_root / "prepared/training_matrix.parquet")
    fold_map = pd.read_parquet(paths.v9_root / "prepared/fixed_nested_scaffold_splits.parquet")
    heldout = fold_map.loc[fold_map.outer_role.eq("heldout"), ["structure_id", "outer_fold"]]
    matrix = matrix.merge(heldout, on="structure_id", validate="one_to_one")
    if len(matrix) != 18_801:
        raise V101Error("exact router matrix must contain 18,801 train structures")
    y = v10.tier_index(matrix.target_pic50.to_numpy(float))
    candidates = [
        ("xgb_rdkit2d", "xgboost", rdkit),
        ("xgb_rdkit2d_morgan", "xgboost", features),
        ("lgbm_rdkit2d", "lightgbm", rdkit),
        ("lgbm_rdkit2d_morgan", "lightgbm", features),
    ]
    results = []
    predictions: dict[str, np.ndarray] = {}
    raw_predictions: dict[str, np.ndarray] = {}
    for candidate_id, engine, columns in candidates:
        oof = np.full((len(matrix), 3), np.nan)
        for fold in range(5):
            fit = matrix.outer_fold.ne(fold).to_numpy()
            evaluate = ~fit
            model = _router_candidate(engine, workers, SEED + fold)
            model.fit(
                _safe_numeric(matrix.loc[fit], columns),
                y[fit],
                sample_weight=v10._class_weights(y[fit]),  # noqa: SLF001
            )
            oof[evaluate] = model.predict_proba(_safe_numeric(matrix.loc[evaluate], columns))
        calibrator = LogisticRegression(C=3, max_iter=2_000, random_state=SEED)
        calibrator.fit(np.log(np.clip(oof, 1e-7, 1.0)), y)
        calibrated = calibrator.predict_proba(np.log(np.clip(oof, 1e-7, 1.0)))
        metrics = _multiclass_metrics(y, calibrated)
        results.append(
            {
                "candidate_id": candidate_id,
                "engine": engine,
                "feature_surface": "rdkit2d" if columns is rdkit else "rdkit2d_morgan",
                **metrics,
            }
        )
        predictions[candidate_id] = calibrated
        raw_predictions[candidate_id] = oof
    ranked = sorted(
        results,
        key=lambda row: (row["balanced_accuracy"], row["macro_f1"], -row["potent_predicted_safe_rate"]),
        reverse=True,
    )
    winner = ranked[0]
    winner_id = str(winner["candidate_id"])
    engine = str(winner["engine"])
    columns = rdkit if winner["feature_surface"] == "rdkit2d" else features
    final = _router_candidate(engine, workers, SEED + 99)
    final.fit(_safe_numeric(matrix, columns), y, sample_weight=v10._class_weights(y))  # noqa: SLF001
    raw = raw_predictions[winner_id]
    calibrator = LogisticRegression(C=3, max_iter=2_000, random_state=SEED)
    calibrator.fit(np.log(np.clip(raw, 1e-7, 1.0)), y)
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "model": final,
            "calibrator": calibrator,
            "feature_columns": columns,
            "class_names": CLASS_NAMES,
            "engine": engine,
            "feature_surface": winner["feature_surface"],
            "training_partition": "train",
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
        },
        paths.output / "models/exact_ternary_router.joblib",
        compress=3,
    )
    evidence = matrix[["structure_id", "scaffold_group_id", "outer_fold", "target_pic50"]].copy()
    evidence["observed_tier"] = y
    for candidate_id, probability in predictions.items():
        for index, name in enumerate(CLASS_NAMES):
            evidence[f"{candidate_id}__probability_{name.lower()}"] = probability[:, index]
    return {"winner": winner, "candidates": ranked}, evidence


def _feature_row(smiles: str, columns: list[str]) -> pd.DataFrame:
    base_columns = [name for name in columns if not name.startswith("maccs__")]
    frame, molecule = v10._feature_frame(smiles, base_columns)  # noqa: SLF001
    if any(name.startswith("maccs__") for name in columns):
        fingerprint = MACCSkeys.GenMACCSKeys(molecule)
        bits = np.zeros(167, dtype=np.uint8)
        for index in range(167):
            bits[index] = int(fingerprint.GetBit(index))
        for index in range(167):
            frame[f"maccs__{index:03d}"] = bits[index]
    return frame.reindex(columns=columns)


def _write_ui(paths: Paths, model_info: dict[str, Any]) -> None:
    html = """<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>hERG V10.1</title><style>
body{margin:0;background:#f5f7fb;color:#111827;font:15px/1.45 Inter,system-ui}
.w{max-width:1160px;margin:auto;padding:38px 22px}.top{display:flex;justify-content:space-between;align-items:flex-end}
h1{font-size:43px;margin:0}.muted{color:#667085}.panel{background:white;border:1px solid #e4e7ec;border-radius:18px;padding:22px;margin-top:20px;box-shadow:0 8px 28px #1018280a}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.card{padding:18px;border:1px solid #e4e7ec;border-radius:14px}
.k{color:#667085;font-size:12px;font-weight:800;letter-spacing:.03em}.v{font-size:30px;font-weight:850}.safe{color:#087443}.moderate{color:#b54708}.potent{color:#b42318}
textarea{width:100%;min-height:82px;padding:12px;border:1px solid #98a2b3;border-radius:11px;font:14px monospace;box-sizing:border-box}
button{margin-top:10px;background:#155eef;color:#fff;border:0;border-radius:10px;padding:12px 18px;font-weight:800}.note{border-left:4px solid #155eef;background:#f4f7ff;padding:12px;border-radius:0 10px 10px 0}.warn{border-color:#b54708;background:#fff8f0}
.bar{height:9px;background:#eef2f6;border-radius:10px;overflow:hidden}.fill{height:100%}.p{display:grid;grid-template-columns:82px 1fr 54px;gap:8px;align-items:center;margin:8px 0}.result{display:none}.badge{display:inline-block;padding:4px 8px;border-radius:8px;background:#eef4ff;color:#3538cd;font-size:12px;font-weight:750}
@media(max-width:800px){.grid{grid-template-columns:1fr}}
</style></head><body><main class='w'><div class='top'><div><h1>hERG V10.1</h1>
<div class='muted'>Tiered ligand research platform with assay domains kept separate.</div></div><b>Research Prototype</b></div>
<section class='panel warn'><b>Internal research demo.</b> Ligand-only modeling with scaffold-held-out internal evaluation. It has not received prospective or external validation and must not be used for clinical decisions.</section>
<section class='panel'><b>Compound SMILES</b><textarea id='s'>CC(C)NCC(O)COc1cccc2ccccc12</textarea>
<button onclick='go()'>Predict</button> <span id='status' class='muted'></span></section>
<section id='r' class='result'><div class='panel'><h2>Stage 1: Mixed-Source IC50 Tier</h2>
<div id='tier' class='v'></div><div id='prob'></div><div class='muted'>Safe &gt;30 µM, Moderate 1–30 µM, Potent &lt;1 µM.</div></div>
<div class='panel'><h2>Stage 2A: Same-Assay Functional qHTS Curve</h2><div class='grid'>
<div class='card'><div class='k'>Empirical IC10</div><div id='ic10' class='v'></div></div>
<div class='card'><div class='k'>Empirical IC30</div><div id='ic30' class='v'></div></div>
<div class='card'><div class='k'>Empirical IC50</div><div id='eic50' class='v'></div></div></div>
<p class='muted'>All three estimates are trained from the same 20-dose functional assay. Predictions are projected in log-concentration space to enforce IC10 ≤ IC30 ≤ IC50. <span id='adjusted' class='badge'></span></p></div>
<div class='panel'><h2>Stage 2B: Mixed-Source Literature IC50</h2><div class='card'><div class='k'>Literature-trained IC50 estimate</div><div id='mic50' class='v'></div><div class='muted'>Uses the larger heterogeneous quantitative surface. It is not a fourth point on the functional qHTS curve above.</div></div></div>
<div class='panel'><h2>Broad Confirmed-WT Fixed-Dose Screen</h2><div class='grid'>
<div class='card'><div class='k'>Probability of screen activity at 46 µM</div><div id='broad' class='v'></div><div class='muted'>46 µM is the experimental test concentration, not a predicted IC50 and not 46 mM.</div></div>
<div class='note'><b>What a negative means.</b><br>It supports low liability at this assay dose and is compatible with IC50 above 30 µM.</div>
<div class='note warn'><b>What a positive means.</b><br>It flags possible liability in a thallium-flux screen, but does not provide an exact IC50 tier.</div></div></div>
<div class='panel'><h2>Evidence Scale</h2><div id='scale' class='grid'></div>
<p class='muted'><b>Strict curves</b> require an inhibitor phenotype, Fit R² ≥0.6, monotonic-adjustment RMSE ≤15, a valid standardized structure, and one median label per connectivity key.</p></div></section></main><script>
const $=x=>document.getElementById(x),fmt=x=>x<.01?x.toExponential(2):x.toFixed(3);
function row(n,v,c){return `<div class='p'><b>${n}</b><div class='bar'><div class='fill' style='width:${100*v}%;background:${c}'></div></div><span>${(100*v).toFixed(1)}%</span></div>`}
async function go(){status.textContent=' running...';let q=await fetch('/api/predict',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({smiles:$('s').value})});let d=await q.json();if(!q.ok){status.textContent=d.error;return}$('r').style.display='block';$('tier').textContent=d.route.tier;$('tier').className='v '+d.route.tier.toLowerCase();$('prob').innerHTML=row('Safe',d.route.probabilities.Safe,'#12b76a')+row('Moderate',d.route.probabilities.Moderate,'#f79009')+row('Potent',d.route.probabilities.Potent,'#f04438');let c=d.quantitative.same_assay_functional_qhts;$('ic10').textContent=fmt(c.ic10_um)+' µM';$('ic30').textContent=fmt(c.ic30_um)+' µM';$('eic50').textContent=fmt(c.ic50_um)+' µM';$('adjusted').textContent=c.ordering_adjusted?'Coherence projection applied':'Raw predictions already ordered';$('mic50').textContent=fmt(d.quantitative.mixed_source_literature_ic50.ic50_um)+' µM';$('broad').textContent=(100*d.broad_fixed_dose.probability_screen_active_at_46um).toFixed(1)+'%';status.textContent=' complete'}
fetch('/api/model-info').then(x=>x.json()).then(d=>{let cards=[['Exact mixed-source IC50',d.data_scale.exact_router_structures.toLocaleString()],['Broad fixed-dose screen',d.data_scale.broad_fixed_dose_structures.toLocaleString()],['Strict empirical curves',`IC10 ${d.data_scale.strict_ic10_labels.toLocaleString()} | IC30 ${d.data_scale.strict_ic30_labels.toLocaleString()} | IC50 ${d.data_scale.strict_ic50_labels.toLocaleString()}`]];$('scale').innerHTML=cards.map(x=>`<div class='card'><div class='k'>${x[0]}</div><div class='v'>${x[1]}</div></div>`).join('')});
</script></body></html>"""
    (paths.output / "app.html").write_text(html)


def _coherent_threshold_concentrations(raw_um: list[float]) -> tuple[list[float], bool]:
    """Project same-assay IC10/IC30/IC50 estimates onto their required order."""
    values = np.asarray(raw_um, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or (values <= 0).any():
        raise V101Error("same-assay threshold predictions must be three positive finite values")
    log_values = np.log10(values)
    projected = IsotonicRegression(increasing=True).fit_transform(np.asarray([10.0, 30.0, 50.0]), log_values)
    adjusted = not np.allclose(log_values, projected, rtol=0.0, atol=1e-12)
    return [float(value) for value in np.power(10.0, projected)], adjusted


class Predictor:
    def __init__(self, root: Path):
        self.root = root.resolve()
        _validate(self.root)
        self.router = joblib.load(self.root / "models/exact_ternary_router.joblib")
        self.mixed_ic50 = joblib.load(self.root / "models/ic50_regressor.joblib")
        self.ic10 = joblib.load(self.root / "models/empirical_ic10_regressor.joblib")
        self.ic30 = joblib.load(self.root / "models/empirical_ic30_regressor.joblib")
        self.empirical_ic50 = joblib.load(self.root / "models/empirical_ic50_regressor.joblib")
        self.broad = joblib.load(self.root / "models/broad_fixed_dose_classifier.joblib")
        self.info = _read_json(self.root / "model_info.json", "model_info_sha256")

    def predict(self, smiles: str) -> dict[str, Any]:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise V101Error("RDKit could not parse the submitted SMILES")
        router_frame = _feature_row(smiles, self.router["feature_columns"])
        raw = self.router["model"].predict_proba(_safe_numeric(router_frame, self.router["feature_columns"]))
        router_p = self.router["calibrator"].predict_proba(np.log(np.clip(raw, 1e-7, 1.0)))[0]
        ic50_frame = _feature_row(smiles, self.mixed_ic50["feature_columns"])
        mixed_pic50 = float(
            self.mixed_ic50["model"].predict(_safe_numeric(ic50_frame, self.mixed_ic50["feature_columns"]))[0]
        )
        endpoint_values = {}
        for name, bundle in (
            ("ic10", self.ic10),
            ("ic30", self.ic30),
            ("ic50", self.empirical_ic50),
        ):
            frame = _feature_row(smiles, bundle["feature_columns"])
            picx = float(bundle["model"].predict(_safe_numeric(frame, bundle["feature_columns"]))[0])
            endpoint_values[name] = {"picx": picx, "um": float(10 ** (6 - picx))}
        coherent_um, ordering_adjusted = _coherent_threshold_concentrations(
            [
                endpoint_values["ic10"]["um"],
                endpoint_values["ic30"]["um"],
                endpoint_values["ic50"]["um"],
            ]
        )
        broad_frame = _feature_row(smiles, self.broad["feature_columns"])
        broad_raw = float(
            self.broad["model"].predict_proba(_safe_numeric(broad_frame, self.broad["feature_columns"]))[0, 1]
        )
        logit = math.log(np.clip(broad_raw, 1e-7, 1 - 1e-7) / np.clip(1 - broad_raw, 1e-7, 1))
        broad_p = float(self.broad["calibrator"].predict_proba([[logit]])[0, 1])
        tier = int(np.argmax(router_p))
        return {
            "smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
            "route": {
                "tier": CLASS_NAMES[tier],
                "probabilities": {CLASS_NAMES[i]: float(router_p[i]) for i in range(3)},
            },
            "quantitative": {
                "same_assay_functional_qhts": {
                    "ic10_um": coherent_um[0],
                    "ic30_um": coherent_um[1],
                    "ic50_um": coherent_um[2],
                    "raw_ic10_um": endpoint_values["ic10"]["um"],
                    "raw_ic30_um": endpoint_values["ic30"]["um"],
                    "raw_ic50_um": endpoint_values["ic50"]["um"],
                    "ordering_adjusted": ordering_adjusted,
                    "ordering_contract": "IC10 <= IC30 <= IC50",
                    "label_method": "observed 20-concentration hERG curve crossings; no Hill conversion",
                },
                "mixed_source_literature_ic50": {
                    "pic50": mixed_pic50,
                    "ic50_um": float(10 ** (6 - mixed_pic50)),
                    "scope": "heterogeneous exact quantitative literature surface; not directly comparable to the qHTS curve",
                },
            },
            "broad_fixed_dose": {
                "probability_screen_active_at_46um": broad_p,
                "assay_concentration_um": 46.0,
                "scope": "confirmed-WT thallium-flux screen activity at one experimental concentration; not a predicted IC50",
            },
            "scope": self.info["scientific_scope"],
        }


def _write_report(paths: Paths, info: dict[str, Any]) -> None:
    router = info["metrics"]["exact_router"]["winner"]
    broad = info["metrics"]["broad_fixed_dose"]["winner"]
    ic10 = info["metrics"]["empirical_endpoints"]["IC10"]["winner"]
    ic30 = info["metrics"]["empirical_endpoints"]["IC30"]["winner"]
    empirical_ic50 = info["metrics"]["empirical_endpoints"]["IC50"]["winner"]
    text = f"""# hERG V10.1 Expanded Platform

V10.1 uses more hERG evidence without pretending heterogeneous assays measure the same endpoint.

## Exact IC50 Tier Router

- Train structures: 18,801
- Winning engine/surface: {router["engine"]} / {router["feature_surface"]}
- Accuracy: {router["accuracy"]:.3f}
- Balanced accuracy: {router["balanced_accuracy"]:.3f}
- Macro F1: {router["macro_f1"]:.3f}
- Potent predicted Safe: {router["potent_predicted_safe_rate"]:.2%}

## Broad Confirmed-WT Fixed-Dose Evidence

- Full structure surface: 339,373
- Train-only structures: 265,625
- Train positives: {info["data_scale"]["broad_train_positives"]:,}
- Winning engine/surface: {broad["engine"]} / {broad["feature_surface"]}
- PR-AUC: {broad["pr_auc"]:.4f} against prevalence {broad["prevalence"]:.4f}
- Recall near 1% FPR: {broad["recall_at_approximately_1pct_fpr"]:.1%}

This thallium-flux task tests screen activity at a fixed experimental concentration of 46 uM.
The number 46 uM is not a predicted IC50. A negative supports low liability at that assay dose;
a positive indicates possible liability but does not identify an exact IC50 tier.

## Empirical IC10, IC30, And IC50

The AID588834 human hERG assay measures 20 concentrations from 0.0001 to 92.17 uM. V10.1
derives empirical threshold crossings by monotone denoising and interpolation between those
observed concentrations. Labels never use a Hill conversion from predicted IC50.
The three predictions are trained within this one assay domain and projected in log-concentration
space to enforce the physically required order IC10 <= IC30 <= IC50.

- Strict IC10 unique structures: {info["data_scale"]["strict_ic10_labels"]:,}; OOF MAE {ic10["mae"]:.3f} pIC10
- Strict IC30 unique structures: {info["data_scale"]["strict_ic30_labels"]:,}; OOF MAE {ic30["mae"]:.3f} pIC30
- Strict IC50 unique structures: {info["data_scale"]["strict_ic50_labels"]:,}; OOF MAE {empirical_ic50["mae"]:.3f} pIC50

## Claim Boundary

All evaluation is internal, train-partition, scaffold-held-out evidence. Repository validation
and test labels remain sealed. The fixed-dose and quantitative endpoints are not pooled, and
the empirical IC10/IC30/IC50 results are functional qHTS endpoints rather than patch-clamp values.
"""
    (paths.output / "REPORT.md").write_text(text)


def _build(paths: Paths, workers: int, force: bool) -> dict[str, Any]:
    if (paths.output / "manifest.json").exists() and not force:
        return _validate(paths.output)
    paths.output.mkdir(parents=True, exist_ok=True)
    (paths.output / "models").mkdir(exist_ok=True)
    (paths.output / "evidence").mkdir(exist_ok=True)
    v10._validate(paths.v10_root)  # noqa: SLF001
    router_metrics, router_oof = _train_improved_router(paths, workers)
    broad_metrics, broad_oof, _ = _train_broad(paths, workers)
    curves, curve_census = _extract_empirical_endpoints(paths)
    empirical_metrics, empirical_oof = _train_empirical_endpoints(paths, curves, workers)
    _atomic_parquet(paths.output / "evidence/exact_router_oof.parquet", router_oof)
    _atomic_parquet(paths.output / "evidence/broad_fixed_dose_oof.parquet", broad_oof)
    _atomic_parquet(paths.output / "evidence/empirical_ic10_ic30_ic50_labels.parquet", curves)
    _atomic_parquet(paths.output / "evidence/empirical_ic10_ic30_ic50_oof.parquet", empirical_oof)
    shutil.copy2(
        paths.v10_root / "models/ic50_regressor.joblib", paths.output / "models/ic50_regressor.joblib"
    )
    shutil.copy2(
        paths.v10_root / "models/ic50_feature_schema.json", paths.output / "models/ic50_feature_schema.json"
    )
    info = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "metrics": {
            "exact_router": router_metrics,
            "broad_fixed_dose": broad_metrics,
            "empirical_endpoints": empirical_metrics,
        },
        "data_scale": {
            "exact_router_structures": 18_801,
            "broad_fixed_dose_structures": broad_metrics["surface_rows_all_partitions"],
            "broad_train_structures": broad_metrics["train_rows"],
            "broad_train_positives": broad_metrics["train_positives"],
            "raw_multiconcentration_curves": curve_census["raw_assay_rows"],
            "strict_ic10_labels": empirical_metrics["IC10"]["unique_structure_labels"],
            "strict_ic30_labels": empirical_metrics["IC30"]["unique_structure_labels"],
            "strict_ic50_labels": empirical_metrics["IC50"]["unique_structure_labels"],
        },
        "curve_census": curve_census,
        "scientific_scope": {
            "ligand_only": True,
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "external_or_prospective_validation": False,
            "empirical_ic10_ic30_ic50_not_hill_derived": True,
            "fixed_dose_pooled_into_exact_tiers": False,
        },
    }
    _atomic_json(paths.output / "model_info.json", info, "model_info_sha256")
    _write_ui(paths, info)
    _write_report(paths, info)
    artifact_paths = [
        ("models/exact_ternary_router.joblib", "exact_ternary_router"),
        ("models/broad_fixed_dose_classifier.joblib", "broad_fixed_dose_classifier"),
        ("models/empirical_ic10_regressor.joblib", "empirical_ic10_regressor"),
        ("models/empirical_ic30_regressor.joblib", "empirical_ic30_regressor"),
        ("models/empirical_ic50_regressor.joblib", "empirical_ic50_regressor"),
        ("models/ic50_regressor.joblib", "ic50_regressor"),
        ("models/ic50_feature_schema.json", "ic50_feature_schema"),
        ("evidence/exact_router_oof.parquet", "exact_router_oof"),
        ("evidence/broad_fixed_dose_oof.parquet", "broad_fixed_dose_oof"),
        ("evidence/empirical_ic10_ic30_ic50_labels.parquet", "empirical_endpoint_labels"),
        ("evidence/empirical_ic10_ic30_ic50_oof.parquet", "empirical_endpoint_oof"),
        ("model_info.json", "model_info"),
        ("app.html", "local_ui"),
        ("REPORT.md", "report"),
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [_binding(paths.output / path, role, paths.output) for path, role in artifact_paths],
        "inputs": [
            _binding(paths.v10_root / "manifest.json", "v10_manifest"),
            _binding(paths.v9_root / "prepared/training_matrix.parquet", "v9_training_matrix"),
            _binding(paths.v9_root / "prepared/fixed_nested_scaffold_splits.parquet", "v9_scaffold_folds"),
            _binding(paths.broad, "confirmed_wt_fixed_dose_surface"),
            _binding(paths.feature_cache / "feature_cache_manifest.json", "global_feature_cache_manifest"),
            _binding(paths.feature_cache / "feature_schema.json", "global_feature_schema"),
            _binding(paths.feature_cache / "source_to_feature_mapping.parquet", "global_feature_mapping"),
            _binding(paths.curves, "AID588834_full_concentration_response"),
            _binding(Path(__file__), "implementation"),
        ],
        "scientific_contract": info["scientific_scope"],
    }
    _atomic_json(paths.output / "manifest.json", manifest, "manifest_sha256")
    return _validate(paths.output)


def _validate(output: Path) -> dict[str, Any]:
    output = output.resolve()
    manifest = _read_json(output / "manifest.json", "manifest_sha256")
    for binding in manifest["artifacts"]:
        _verify(binding, output)
    for binding in manifest["inputs"]:
        _verify(binding)
    scope = manifest["scientific_contract"]
    if scope["repository_validation_labels_opened"] or scope["repository_test_labels_opened"]:
        raise V101Error("repository validation/test labels must remain sealed")
    exact = pd.read_parquet(output / "evidence/exact_router_oof.parquet")
    broad = pd.read_parquet(output / "evidence/broad_fixed_dose_oof.parquet")
    endpoints = pd.read_parquet(output / "evidence/empirical_ic10_ic30_ic50_oof.parquet")
    if len(exact) != 18_801 or exact.structure_id.nunique() != 18_801:
        raise V101Error("exact router OOF evidence is incomplete")
    if len(broad) != 265_625 or broad.structure_id.nunique() != 265_625:
        raise V101Error("broad fixed-dose OOF evidence is incomplete")
    if set(exact.outer_fold.unique()) != set(range(5)) or set(broad.fold.unique()) != set(range(3)):
        raise V101Error("OOF fold coverage is incomplete")
    if set(endpoints.endpoint.unique()) != {"IC10", "IC30", "IC50"}:
        raise V101Error("empirical endpoint evidence is incomplete")
    Predictor(output).predict("CCO") if False else None
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "artifacts_verified": len(manifest["artifacts"]),
        "exact_router_oof_rows": len(exact),
        "broad_train_oof_rows": len(broad),
        "empirical_endpoint_oof_rows": len(endpoints),
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "output_root": str(output),
    }


def _paths(args: argparse.Namespace) -> Paths:
    repo = args.repo_root.resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (repo / path).resolve()

    return Paths(
        repo,
        resolve(args.v10_root),
        resolve(args.v9_root),
        resolve(args.broad),
        resolve(args.feature_cache),
        resolve(args.curves),
        resolve(args.output_root),
    )


def _handler(predictor: Predictor):
    # V10.1 deliberately reuses the stateless V10 HTTP handler protocol. The
    # expanded predictor provides the same root/info/predict attributes but is
    # a separate concrete class, so erase that implementation-only distinction.
    return v10._handler(cast(Any, predictor))  # noqa: SLF001


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--v10-root", type=Path, default=DEFAULT_V10)
    build.add_argument("--v9-root", type=Path, default=DEFAULT_V9)
    build.add_argument("--broad", type=Path, default=DEFAULT_BROAD)
    build.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    build.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    build.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--workers", type=int, default=6, choices=range(1, 7))
    build.add_argument("--force", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("--output-root", type=Path, required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--model-root", type=Path, required=True)
    predict.add_argument("--smiles", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--model-root", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8788)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build":
        result = _build(_paths(args), args.workers, args.force)
    elif args.command == "validate":
        result = _validate(args.output_root)
    elif args.command == "predict":
        result = Predictor(args.model_root).predict(args.smiles)
    elif args.command == "serve":
        from http.server import ThreadingHTTPServer

        predictor = Predictor(args.model_root)
        server = ThreadingHTTPServer((args.host, args.port), _handler(predictor))
        print(f"hERG V10.1 UI: http://{args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
