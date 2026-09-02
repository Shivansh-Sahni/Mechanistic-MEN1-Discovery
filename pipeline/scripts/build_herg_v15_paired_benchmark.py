#!/usr/bin/env python3
"""Freeze and score a leakage-controlled, train-only hERG matched-pair benchmark.

``freeze`` is deliberately label-blind.  It decodes only pair identifiers,
structure identifiers, scaffold/fold metadata, pre-existing OOF predictions,
and prediction-side applicability descriptors.  It writes the immutable pair
allocation and baseline predictions before any MMP outcome column is opened.

``score-baseline`` verifies that lock, then opens only the exact TRAIN-partition
MMP effects.  Repository validation/test, prospective, showcase, and final-test
measurements are neither required nor read by this script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCHEMA_VERSION = "herg-v15-locked-paired-benchmark/1.0"
DEFAULT_SEED = 20260831
DEFAULT_TEST_FRACTION = 0.20
DEFAULT_BOOTSTRAP_REPLICATES = 1_000
DEFAULT_TIE_THRESHOLD_PIC50 = 0.10
DEFAULT_CLIFF_THRESHOLD_PIC50 = 1.00
DEFAULT_MIN_PAIRS = 30
DEFAULT_MIN_GROUPS = 20

BASELINE_MODELS: tuple[tuple[str, str, str], ...] = (
    ("v11_nested", "pred__v11_nested", "primary_prespecified_v11_nested_oof"),
    ("v9_anchor", "v9_predicted_pic50", "historical_prespecified_oof_reference"),
    ("xgb_depth10", "pred__xgb_depth10", "prespecified_v11_component_oof_reference"),
)

REGISTRY_COLUMNS = (
    "pair_id",
    "model_split",
    "structure_id_a",
    "structure_id_b",
)
OOF_FREEZE_COLUMNS = (
    "structure_id",
    "scaffold_group_id",
    "outer_fold",
    "maximum_train_tanimoto",
    "extrapolation_flag",
    "rdkit2d__MolWt",
    *(column for _, column, _ in BASELINE_MODELS),
)
EFFECT_COLUMNS = (
    "pair_id",
    "structure_id_a",
    "structure_id_b",
    "pic50_median_a",
    "pic50_median_b",
    "pic50_observations_a",
    "pic50_observations_b",
    "delta_pic50_b_minus_a",
    "activity_cliff_ge_1_pic50",
    "exploratory_training_only",
)


class BenchmarkError(RuntimeError):
    """Raised when the paired-benchmark evidence contract is violated."""


@dataclass(frozen=True)
class BenchmarkInputs:
    """Bound local inputs for the frozen benchmark."""

    pair_registry: Path
    training_effects: Path
    mmp_manifest: Path
    nested_oof: Path
    exact_observations: Path


def default_inputs(repo: Path) -> BenchmarkInputs:
    """Return the authoritative local paths without opening any data."""

    mmp_root = repo / "research/data/platform/processed/herg_hierarchy/v1_5_mmp_analysis"
    v11_root = repo / "research/local_runs/herg_comprehensive_optimization_v11_1"
    return BenchmarkInputs(
        pair_registry=mmp_root / "mmp_pair_registry.parquet",
        training_effects=mmp_root / "training_mmp_effects.parquet",
        mmp_manifest=mmp_root / "mmp_analysis_manifest.json",
        nested_oof=v11_root / "analysis/nested_oof_predictions.parquet",
        exact_observations=v11_root / "prepared/exact_train_observations.parquet",
    )


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha256(path: Path) -> str:
    return hashlib.sha256(pq.read_schema(path).serialize().to_pybytes()).hexdigest()


def _binding(path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise BenchmarkError(f"missing {role}: {path}")
    result: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        result.update(
            rows=pq.read_metadata(path).num_rows,
            arrow_schema_sha256=_schema_sha256(path),
        )
    return result


def _verify_binding(binding: dict[str, Any]) -> None:
    path = Path(str(binding["path"])).resolve()
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
        raise BenchmarkError(f"artifact missing or size changed: {path}")
    if _sha256(path) != binding["sha256"]:
        raise BenchmarkError(f"artifact hash changed: {path}")
    if path.suffix == ".parquet":
        if pq.read_metadata(path).num_rows != int(binding["rows"]):
            raise BenchmarkError(f"Parquet row count changed: {path}")
        if _schema_sha256(path) != binding["arrow_schema_sha256"]:
            raise BenchmarkError(f"Parquet schema changed: {path}")


def _self_hash(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop(key, None)
    result[key] = _digest(result)
    return result


def _atomic_json(path: Path, value: dict[str, Any], hash_key: str) -> dict[str, Any]:
    document = _self_hash(value, hash_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(json.dumps(document, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
    os.replace(temporary, path)
    return document


def _read_json(path: Path, hash_key: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkError(f"expected JSON object: {path}")
    if hash_key is not None and value.get(hash_key) != _self_hash(value, hash_key)[hash_key]:
        raise BenchmarkError(f"self-hash mismatch: {path}")
    return value


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _read_columns(path: Path, columns: tuple[str, ...] | list[str]) -> pd.DataFrame:
    available = set(pq.read_schema(path).names)
    missing = sorted(set(columns) - available)
    if missing:
        raise BenchmarkError(f"missing required columns in {path}: {missing}")
    return pq.read_table(path, columns=list(columns)).to_pandas()


def _series_campaign_identifier_columns(path: Path) -> list[str]:
    return sorted(
        column
        for column in pq.read_schema(path).names
        if "series" in column.lower() or "campaign" in column.lower()
    )


def _source_binding(contract: dict[str, Any], role: str) -> dict[str, Any]:
    matches = [binding for binding in contract["source_bindings"] if binding["role"] == role]
    if len(matches) != 1:
        raise BenchmarkError(f"lock does not contain exactly one source binding for {role}")
    return matches[0]


def _validate_mmp_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected != actual:
        raise BenchmarkError("source MMP manifest self-hash mismatch")
    contract = manifest.get("scientific_contract", {})
    if contract.get("effect_estimation_partition") != "train_only":
        raise BenchmarkError("MMP effects are not contractually train-only")
    if contract.get("validation_test_pairs_are_definition_only") is not True:
        raise BenchmarkError("validation/test pair definitions are not contractually label-blind")
    return manifest


def _assert_registry_binding(manifest: dict[str, Any], registry_path: Path) -> None:
    expected = manifest.get("artifacts", {}).get("mmp_pair_registry.parquet")
    if not expected:
        raise BenchmarkError("MMP manifest does not bind mmp_pair_registry.parquet")
    observed = _binding(registry_path, "source_train_pair_registry")
    for key in ("bytes", "sha256", "rows", "arrow_schema_sha256"):
        if observed[key] != expected[key]:
            raise BenchmarkError(f"pair registry disagrees with MMP manifest: {key}")


def canonicalize_registry(registry: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Canonicalize unordered pairs and collapse exact/reversed duplicates."""

    missing = sorted(set(REGISTRY_COLUMNS) - set(registry.columns))
    if missing:
        raise BenchmarkError(f"registry columns absent: {missing}")
    frame = registry.loc[:, REGISTRY_COLUMNS].copy()
    for column in REGISTRY_COLUMNS:
        if frame[column].isna().any():
            raise BenchmarkError(f"registry column contains null values: {column}")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise BenchmarkError(f"registry column contains blank values: {column}")
    if frame["pair_id"].duplicated().any():
        raise BenchmarkError("source pair_id is not unique")
    if frame["structure_id_a"].eq(frame["structure_id_b"]).any():
        raise BenchmarkError("self-pairs are not admissible")

    source_a = frame["structure_id_a"].copy()
    source_b = frame["structure_id_b"].copy()
    swapped = source_a.gt(source_b)
    frame["canonical_structure_id_a"] = np.where(swapped, source_b, source_a)
    frame["canonical_structure_id_b"] = np.where(swapped, source_a, source_b)
    frame["source_orientation_sign"] = np.where(swapped, -1, 1).astype(np.int8)
    frame["source_pair_id"] = frame.pop("pair_id")
    frame["canonical_pair_id"] = [
        "HPAIR-" + hashlib.sha256(f"{left}|{right}".encode()).hexdigest()[:24].upper()
        for left, right in zip(
            frame["canonical_structure_id_a"], frame["canonical_structure_id_b"], strict=True
        )
    ]
    reverse_rows = int(swapped.sum())
    before = len(frame)
    frame = frame.sort_values(
        ["canonical_structure_id_a", "canonical_structure_id_b", "source_pair_id"]
    ).drop_duplicates(["canonical_structure_id_a", "canonical_structure_id_b"], keep="first")
    frame = frame.sort_values("canonical_pair_id").reset_index(drop=True)
    if frame["canonical_pair_id"].duplicated().any():
        raise BenchmarkError("canonical pair hash collision")
    return frame, {
        "source_rows": int(before),
        "canonical_rows": int(len(frame)),
        "reversed_source_rows_normalized": reverse_rows,
        "duplicate_or_reversed_rows_collapsed": int(before - len(frame)),
    }


class _UnionFind:
    def __init__(self, values: set[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def assign_leakage_groups(pairs: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    """Union all MMP-connected structures and all shared-scaffold structures."""

    structure_ids = set(pairs["canonical_structure_id_a"]) | set(
        pairs["canonical_structure_id_b"]
    )
    if oof["structure_id"].duplicated().any():
        raise BenchmarkError("OOF structure_id is not unique")
    metadata = oof[oof["structure_id"].isin(structure_ids)].copy()
    missing = structure_ids - set(metadata["structure_id"])
    if missing:
        raise BenchmarkError(f"OOF metadata misses {len(missing)} registry structures")
    union = _UnionFind(structure_ids)
    for row in pairs.itertuples(index=False):
        union.union(row.canonical_structure_id_a, row.canonical_structure_id_b)
    for _, group in metadata.groupby("scaffold_group_id", observed=True, sort=True):
        members = sorted(group["structure_id"].astype(str).unique())
        for member in members[1:]:
            union.union(members[0], member)

    components: dict[str, list[str]] = defaultdict(list)
    for structure_id in sorted(structure_ids):
        components[union.find(structure_id)].append(structure_id)
    structure_to_group: dict[str, str] = {}
    for members in components.values():
        group_id = "HLG-" + hashlib.sha256("|".join(members).encode()).hexdigest()[:24].upper()
        structure_to_group.update(dict.fromkeys(members, group_id))

    result = pairs.copy()
    result["leakage_group_id"] = result["canonical_structure_id_a"].map(structure_to_group)
    if result["leakage_group_id"].isna().any():
        raise BenchmarkError("failed to assign every pair to a leakage group")
    return result


def allocate_locked_split(
    pairs: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> pd.DataFrame:
    """Assign whole leakage groups using only group IDs and eligible pair counts."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie strictly between zero and one")
    counts = (
        pairs[pairs["baseline_pair_eligible"]]
        .groupby("leakage_group_id", observed=True)
        .size()
        .to_dict()
    )
    groups = sorted(pairs["leakage_group_id"].unique())
    ordering = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}|{group}".encode()).hexdigest())
    target = float(sum(counts.values())) * test_fraction
    test_count = 0
    assignments: dict[str, str] = {}
    for group in ordering:
        weight = int(counts.get(group, 0))
        add_distance = abs((test_count + weight) - target)
        keep_distance = abs(test_count - target)
        if weight > 0 and (test_count == 0 or add_distance <= keep_distance):
            assignments[group] = "locked_test"
            test_count += weight
        elif weight == 0:
            score = int(hashlib.sha256(f"zero|{seed}|{group}".encode()).hexdigest()[:16], 16)
            assignments[group] = "locked_test" if score / 16**16 < test_fraction else "development"
        else:
            assignments[group] = "development"
    result = pairs.copy()
    result["benchmark_split"] = result["leakage_group_id"].map(assignments)
    if result["benchmark_split"].isna().any():
        raise BenchmarkError("split allocation failed")
    return result


def _attach_freeze_metadata(pairs: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    meta = oof.loc[:, ["structure_id", "scaffold_group_id", "outer_fold"]]
    left = meta.rename(
        columns={
            "structure_id": "canonical_structure_id_a",
            "scaffold_group_id": "scaffold_group_id_a",
            "outer_fold": "outer_fold_a",
        }
    )
    right = meta.rename(
        columns={
            "structure_id": "canonical_structure_id_b",
            "scaffold_group_id": "scaffold_group_id_b",
            "outer_fold": "outer_fold_b",
        }
    )
    result = pairs.merge(left, on="canonical_structure_id_a", validate="many_to_one").merge(
        right, on="canonical_structure_id_b", validate="many_to_one"
    )
    result["same_outer_fold_pair"] = result["outer_fold_a"].eq(result["outer_fold_b"])
    return result


def apply_group_fold_eligibility(pairs: pd.DataFrame) -> pd.DataFrame:
    """Require the complete MMP/scaffold leakage group to be held out together.

    A pair whose own endpoints share an outer fold is not sufficient when a
    third analogue in the same MMP connected component occurs in another fold:
    that analogue was available to the pair's OOF model.  The strict primary
    benchmark therefore retains only leakage groups contained in one fold.
    """

    folds_by_group: dict[str, set[int]] = defaultdict(set)
    for row in pairs.itertuples(index=False):
        folds_by_group[row.leakage_group_id].update(
            (int(row.outer_fold_a), int(row.outer_fold_b))
        )
    result = pairs.copy()
    result["leakage_group_outer_fold_count"] = result["leakage_group_id"].map(
        {group: len(folds) for group, folds in folds_by_group.items()}
    )
    result["series_proxy_outer_fold_exclusive"] = result[
        "leakage_group_outer_fold_count"
    ].eq(1)
    result["baseline_pair_eligible"] = result["same_outer_fold_pair"] & result[
        "series_proxy_outer_fold_exclusive"
    ]
    result["eligibility_reason"] = np.select(
        [
            ~result["same_outer_fold_pair"],
            ~result["series_proxy_outer_fold_exclusive"],
        ],
        [
            "excluded_endpoints_have_different_outer_oof_models",
            "excluded_series_proxy_has_analogue_in_oof_training_folds",
        ],
        default="strict_series_and_scaffold_exclusive_oof_pair",
    )
    return result


def _assert_no_split_leakage(manifest: pd.DataFrame) -> dict[str, int]:
    checks: dict[str, int] = {}
    if manifest["canonical_pair_id"].duplicated().any():
        raise BenchmarkError("canonical pair duplicates remain")
    reverse_key = manifest.apply(
        lambda row: "|".join(
            sorted((row["canonical_structure_id_a"], row["canonical_structure_id_b"]))
        ),
        axis=1,
    )
    if reverse_key.duplicated().any():
        raise BenchmarkError("duplicate/reversed pair definitions remain")
    checks["canonical_or_reversed_pair_overlap_count"] = 0

    structure_splits: dict[str, set[str]] = defaultdict(set)
    scaffold_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest.itertuples(index=False):
        structure_splits[row.canonical_structure_id_a].add(row.benchmark_split)
        structure_splits[row.canonical_structure_id_b].add(row.benchmark_split)
        scaffold_splits[row.scaffold_group_id_a].add(row.benchmark_split)
        scaffold_splits[row.scaffold_group_id_b].add(row.benchmark_split)
    structure_overlap = sum(len(value) > 1 for value in structure_splits.values())
    scaffold_overlap = sum(len(value) > 1 for value in scaffold_splits.values())
    group_overlap = int(
        (manifest.groupby("leakage_group_id")["benchmark_split"].nunique() > 1).sum()
    )
    checks.update(
        structure_split_overlap_count=structure_overlap,
        scaffold_split_overlap_count=scaffold_overlap,
        leakage_group_split_overlap_count=group_overlap,
    )
    if any(checks.values()):
        raise BenchmarkError(f"benchmark split leakage detected: {checks}")
    return checks


def _prediction_frame(manifest: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    eligible = manifest[manifest["baseline_pair_eligible"]].copy()
    feature_columns = [
        "structure_id",
        "maximum_train_tanimoto",
        "extrapolation_flag",
        "rdkit2d__MolWt",
        *(column for _, column, _ in BASELINE_MODELS),
    ]
    endpoint = oof.loc[:, feature_columns]
    left = endpoint.rename(
        columns={column: f"{column}__a" for column in feature_columns if column != "structure_id"}
    ).rename(columns={"structure_id": "canonical_structure_id_a"})
    right = endpoint.rename(
        columns={column: f"{column}__b" for column in feature_columns if column != "structure_id"}
    ).rename(columns={"structure_id": "canonical_structure_id_b"})
    result = eligible.merge(left, on="canonical_structure_id_a", validate="many_to_one").merge(
        right, on="canonical_structure_id_b", validate="many_to_one"
    )
    output = result[
        [
            "canonical_pair_id",
            "benchmark_split",
            "leakage_group_id",
            "outer_fold_a",
        ]
    ].rename(columns={"outer_fold_a": "outer_fold"})
    for model_id, column, _ in BASELINE_MODELS:
        a = pd.to_numeric(result[f"{column}__a"], errors="coerce")
        b = pd.to_numeric(result[f"{column}__b"], errors="coerce")
        if (~np.isfinite(a) | ~np.isfinite(b)).any():
            raise BenchmarkError(f"non-finite frozen baseline prediction: {model_id}")
        output[f"{model_id}_pic50_a"] = a
        output[f"{model_id}_pic50_b"] = b
        output[f"{model_id}_delta_pic50_b_minus_a"] = b - a
    mw_a = pd.to_numeric(result["rdkit2d__MolWt__a"], errors="coerce")
    mw_b = pd.to_numeric(result["rdkit2d__MolWt__b"], errors="coerce")
    if (~np.isfinite(mw_a) | ~np.isfinite(mw_b) | mw_a.le(0) | mw_b.le(0)).any():
        raise BenchmarkError("invalid MolWt in frozen OOF descriptors")
    output["mean_endpoint_mw"] = (mw_a + mw_b) / 2.0
    output["max_endpoint_mw"] = np.maximum(mw_a, mw_b)
    support_a = pd.to_numeric(result["maximum_train_tanimoto__a"], errors="coerce")
    support_b = pd.to_numeric(result["maximum_train_tanimoto__b"], errors="coerce")
    if (~np.isfinite(support_a) | ~np.isfinite(support_b)).any():
        raise BenchmarkError("invalid maximum_train_tanimoto in OOF descriptors")
    output["minimum_endpoint_train_tanimoto"] = np.minimum(support_a, support_b)
    output["any_endpoint_extrapolation"] = (
        result["extrapolation_flag__a"].astype(bool)
        | result["extrapolation_flag__b"].astype(bool)
    )
    if output["canonical_pair_id"].duplicated().any():
        raise BenchmarkError("baseline prediction pair IDs are not unique")
    return output.sort_values("canonical_pair_id").reset_index(drop=True)


def _output_paths(root: Path) -> dict[str, Path]:
    return {
        "manifest": root / "locked/locked_pair_manifest.parquet",
        "predictions": root / "locked/baseline_oof_predictions_before_score.parquet",
        "prediction_seal": root / "locked/baseline_prediction_seal.json",
        "contract": root / "locked/lock_contract.json",
        "labels": root / "sealed/train_only_pair_labels.parquet",
        "label_seal": root / "sealed/train_only_label_seal.json",
        "scores": root / "analysis/baseline_scored_pairs.parquet",
        "metrics": root / "analysis/baseline_metrics.parquet",
        "metrics_csv": root / "analysis/baseline_metrics.csv",
        "evaluation": root / "analysis/baseline_evaluation.json",
        "report": root / "analysis/REPORT.md",
        "validation": root / "validation.json",
    }


def _verify_existing_lock(root: Path) -> dict[str, Any]:
    paths = _output_paths(root)
    contract = _read_json(paths["contract"], "contract_sha256")
    for binding in contract["frozen_artifacts"]:
        _verify_binding(binding)
    for binding in contract["source_bindings"]:
        _verify_binding(binding)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise BenchmarkError("unsupported lock schema")
    return contract


def freeze_benchmark(
    *,
    inputs: BenchmarkInputs,
    output_root: Path,
    seed: int = DEFAULT_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    tie_threshold_pic50: float = DEFAULT_TIE_THRESHOLD_PIC50,
    cliff_threshold_pic50: float = DEFAULT_CLIFF_THRESHOLD_PIC50,
) -> dict[str, Any]:
    """Freeze label-blind allocation and prespecified OOF baseline predictions."""

    paths = _output_paths(output_root)
    frozen = (paths["manifest"], paths["predictions"], paths["prediction_seal"], paths["contract"])
    existing = [path.exists() for path in frozen]
    if any(existing):
        if not all(existing):
            raise BenchmarkError("partial frozen benchmark exists; refusing overwrite")
        return _verify_existing_lock(output_root)
    if any(path.exists() for key, path in paths.items() if key not in {"manifest", "predictions", "prediction_seal", "contract"}):
        raise BenchmarkError("scored outputs predate the lock; refusing retrospective overwrite")
    if bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    if tie_threshold_pic50 < 0 or cliff_threshold_pic50 <= 0:
        raise ValueError("direction and cliff thresholds are invalid")

    source_manifest = _validate_mmp_manifest(inputs.mmp_manifest)
    _assert_registry_binding(source_manifest, inputs.pair_registry)
    registry = _read_columns(inputs.pair_registry, list(REGISTRY_COLUMNS))
    registry = registry[registry["model_split"].eq("train")].reset_index(drop=True)
    canonical, deduplication = canonicalize_registry(registry)
    oof = _read_columns(inputs.nested_oof, list(OOF_FREEZE_COLUMNS))
    canonical = _attach_freeze_metadata(canonical, oof)
    canonical = assign_leakage_groups(canonical, oof)
    canonical = apply_group_fold_eligibility(canonical)
    canonical = allocate_locked_split(canonical, seed=seed, test_fraction=test_fraction)
    leakage_checks = _assert_no_split_leakage(canonical)
    predictions = _prediction_frame(canonical, oof)

    manifest_columns = [
        "canonical_pair_id",
        "source_pair_id",
        "canonical_structure_id_a",
        "canonical_structure_id_b",
        "source_orientation_sign",
        "scaffold_group_id_a",
        "scaffold_group_id_b",
        "outer_fold_a",
        "outer_fold_b",
        "leakage_group_id",
        "leakage_group_outer_fold_count",
        "benchmark_split",
        "same_outer_fold_pair",
        "series_proxy_outer_fold_exclusive",
        "baseline_pair_eligible",
        "eligibility_reason",
    ]
    locked_manifest = canonical.loc[:, manifest_columns].sort_values("canonical_pair_id").reset_index(drop=True)
    _atomic_parquet(paths["manifest"], locked_manifest)
    _atomic_parquet(paths["predictions"], predictions)
    prediction_seal = _atomic_json(
        paths["prediction_seal"],
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "frozen_before_mmp_outcomes_opened",
            "baseline_models": [
                {"model_id": model_id, "source_column": column, "role": role}
                for model_id, column, role in BASELINE_MODELS
            ],
            "label_columns_decoded_during_freeze": [],
            "outcome_source_opened_during_freeze": False,
            "artifact": _binding(paths["predictions"], "baseline_oof_predictions_before_score"),
        },
        "prediction_seal_sha256",
    )

    eligible_counts = (
        locked_manifest[locked_manifest["baseline_pair_eligible"]]["benchmark_split"]
        .value_counts()
        .to_dict()
    )
    source_bindings = [
        _binding(inputs.pair_registry, "label_blind_pair_registry"),
        _binding(inputs.nested_oof, "pre_existing_nested_oof_predictions"),
        _binding(inputs.mmp_manifest, "mmp_source_manifest"),
    ]
    contract = _atomic_json(
        paths["contract"],
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "locked_before_any_v15_challenger",
            "source_bindings": source_bindings,
            "source_training_effects_expected_binding": source_manifest["artifacts"][
                "training_mmp_effects.parquet"
            ],
            "frozen_artifacts": [
                _binding(paths["manifest"], "locked_pair_manifest"),
                _binding(paths["predictions"], "baseline_oof_predictions_before_score"),
                _binding(paths["prediction_seal"], "baseline_prediction_seal"),
            ],
            "prediction_seal_sha256": prediction_seal["prediction_seal_sha256"],
            "prespecified_analysis": {
                "primary_baseline_model": "v11_nested",
                "reference_models": ["v9_anchor", "xgb_depth10"],
                "direction_delta": "candidate/B pIC50 minus parent/A pIC50",
                "positive_direction_label": "increased_hERG_liability",
                "negative_direction_label": "decreased_hERG_liability",
                "tie_threshold_pic50": float(tie_threshold_pic50),
                "primary_direction_rule": "exclude measured negligible changes, then score the sign of every nonzero predicted delta",
                "secondary_thresholded_direction_rule": "apply the same negligible threshold to measured and predicted deltas; predicted negligible calls count as incorrect",
                "activity_cliff_threshold_absolute_pic50": float(cliff_threshold_pic50),
                "cluster_bootstrap_unit": "leakage_group_id",
                "cluster_bootstrap_replicates": int(bootstrap_replicates),
                "cluster_bootstrap_seed": int(seed + 17),
                "confidence_level": 0.95,
                "mw_band_edges_da": [400.0, 500.0, 600.0, 700.0],
                "endpoint_oof_support_band_edges_tanimoto": [0.3, 0.5, 0.7],
                "minimum_support_pairs": DEFAULT_MIN_PAIRS,
                "minimum_support_leakage_groups": DEFAULT_MIN_GROUPS,
                "target_consistency_tolerance_pic50": 1e-6,
            },
            "split_definition": {
                "seed": int(seed),
                "target_locked_test_fraction": float(test_fraction),
                "assignment_inputs": ["leakage_group_id", "eligible_pair_count"],
                "measurement_values_used_for_assignment": False,
                "eligible_pair_counts": {key: int(value) for key, value in eligible_counts.items()},
                "all_pair_rows": int(len(locked_manifest)),
                "same_outer_fold_pair_rows": int(locked_manifest["same_outer_fold_pair"].sum()),
                "eligible_strict_series_scaffold_oof_pairs": int(
                    locked_manifest["baseline_pair_eligible"].sum()
                ),
                "excluded_cross_outer_fold_pair_rows": int(
                    (~locked_manifest["same_outer_fold_pair"]).sum()
                ),
                "excluded_series_proxy_cross_fold_pair_rows": int(
                    (
                        locked_manifest["same_outer_fold_pair"]
                        & ~locked_manifest["series_proxy_outer_fold_exclusive"]
                    ).sum()
                ),
                "leakage_groups": int(locked_manifest["leakage_group_id"].nunique()),
            },
            "canonicalization": deduplication,
            "series_campaign_identifier_schema_audit": {
                "pair_registry": _series_campaign_identifier_columns(inputs.pair_registry),
                "training_effects": _series_campaign_identifier_columns(inputs.training_effects),
                "nested_oof": _series_campaign_identifier_columns(inputs.nested_oof),
                "exact_observations": _series_campaign_identifier_columns(inputs.exact_observations),
                "interpretation": "no explicit identifier available" if not any(
                    _series_campaign_identifier_columns(path)
                    for path in (
                        inputs.pair_registry,
                        inputs.training_effects,
                        inputs.nested_oof,
                        inputs.exact_observations,
                    )
                ) else "identifier column exists and requires explicit review",
            },
            "leakage_controls": {
                **leakage_checks,
                "reversed_pairs": "canonicalized_and_deduplicated_before_allocation",
                "replicates": "one_structure_level_target_per_endpoint; every structure restricted to one benchmark split",
                "series": "explicit_series_identifier_unavailable; full TRAIN MMP connected component used as prespecified proxy and only single-outer-fold proxy groups are scoreable",
                "scaffold": "all shared scaffold_group_id structures unioned into one leakage group",
                "campaign": "unavailable_no_campaign_identifier_in_authoritative_inputs",
                "outer_oof_comparability": "both endpoints and their complete MMP/scaffold leakage group must be exclusive to one outer fold",
            },
            "censoring_contract": {
                "policy": "exact_point_only",
                "source_projection": "Q1 target_pic50_point and model_split=train",
                "censored_values_coerced_to_points": False,
                "censored_pairs_in_benchmark": False,
                "limitation": "directional performance for censored measurements is not estimated",
            },
            "label_access_contract": {
                "training_mmp_effect_columns_opened_during_freeze": False,
                "oof_observed_pic50_column_opened_during_freeze": False,
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
                "prospective_labels_opened": False,
                "showcase_or_final_test_measurements_opened": False,
                "whole_file_hashing_means_physical_page_level_nonaccess_not_claimed": True,
            },
            "limitations": [
                "This is a retrospective internal TRAIN-partition benchmark, not external, blinded, prospective, or independent evidence.",
                "Explicit chemical-series identifiers are absent; the MMP connected component is only a series proxy.",
                "Campaign identifiers are absent, so campaign-level leakage cannot be tested or controlled.",
                "MMP pairs are overlapping and non-independent; intervals therefore resample leakage groups, not individual pairs.",
                "Pairs whose endpoints require different outer-fold models, or whose MMP/scaffold group has an analogue in another outer fold, are excluded.",
                "Outcome consistency with the V11 exact target is checked only after predictions and allocation are frozen.",
                "No molecule structures, SMILES, MMP cores, or transformations are emitted by this benchmark.",
            ],
        },
        "contract_sha256",
    )
    return contract


def _verify_effect_binding(contract: dict[str, Any], effects_path: Path) -> None:
    expected = contract["source_training_effects_expected_binding"]
    observed = _binding(effects_path, "train_only_mmp_effects_opened_after_lock")
    for key in ("bytes", "sha256", "rows", "arrow_schema_sha256"):
        if observed[key] != expected[key]:
            raise BenchmarkError(f"training effects disagree with frozen source manifest: {key}")


def _aggregate_observation_context(observations: pd.DataFrame) -> pd.DataFrame:
    def unique_or_mixed(values: pd.Series) -> str:
        unique = sorted(set(values.dropna().astype(str)))
        return unique[0] if len(unique) == 1 else "mixed"

    return (
        observations.groupby("structure_id", observed=True)
        .agg(
            assay_family=("assay_family", unique_or_mixed),
            source_family=("source_family", unique_or_mixed),
        )
        .reset_index()
    )


def _pair_context(left: pd.Series, right: pd.Series) -> pd.Series:
    same = left.eq(right) & left.ne("mixed")
    return pd.Series(np.where(same, "same:" + left.astype(str), "mixed_or_cross"), index=left.index)


def _prepare_locked_labels(
    manifest: pd.DataFrame,
    effects: pd.DataFrame,
    oof_observed: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    target_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    eligible = manifest[manifest["baseline_pair_eligible"]].copy()
    if effects["pair_id"].duplicated().any():
        raise BenchmarkError("training effects pair_id is not unique")
    joined = eligible.merge(
        effects,
        left_on="source_pair_id",
        right_on="pair_id",
        how="left",
        validate="one_to_one",
    )
    if joined["pic50_median_a"].isna().any():
        raise BenchmarkError("frozen train pair is absent from training effects")
    source_ids_match = joined["structure_id_a"].eq(
        np.where(
            joined["source_orientation_sign"].eq(1),
            joined["canonical_structure_id_a"],
            joined["canonical_structure_id_b"],
        )
    ) & joined["structure_id_b"].eq(
        np.where(
            joined["source_orientation_sign"].eq(1),
            joined["canonical_structure_id_b"],
            joined["canonical_structure_id_a"],
        )
    )
    if not source_ids_match.all():
        raise BenchmarkError("effect endpoints disagree with frozen source orientation")
    for column in ("pic50_median_a", "pic50_median_b", "delta_pic50_b_minus_a"):
        numeric = pd.to_numeric(joined[column], errors="coerce")
        if (~np.isfinite(numeric)).any():
            raise BenchmarkError(f"non-finite exact MMP target: {column}")
        joined[column] = numeric
    source_delta = joined["pic50_median_b"] - joined["pic50_median_a"]
    if not np.allclose(source_delta, joined["delta_pic50_b_minus_a"], atol=1e-10, rtol=0):
        raise BenchmarkError("source MMP delta is internally inconsistent")

    normal = joined["source_orientation_sign"].eq(1)
    joined["measured_pic50_a"] = np.where(normal, joined["pic50_median_a"], joined["pic50_median_b"])
    joined["measured_pic50_b"] = np.where(normal, joined["pic50_median_b"], joined["pic50_median_a"])
    joined["observations_a"] = np.where(
        normal, joined["pic50_observations_a"], joined["pic50_observations_b"]
    ).astype(int)
    joined["observations_b"] = np.where(
        normal, joined["pic50_observations_b"], joined["pic50_observations_a"]
    ).astype(int)
    joined["measured_delta_pic50_b_minus_a"] = (
        joined["measured_pic50_b"] - joined["measured_pic50_a"]
    )

    observed_left = oof_observed.rename(
        columns={"structure_id": "canonical_structure_id_a", "observed_pic50": "oof_observed_a"}
    )
    observed_right = oof_observed.rename(
        columns={"structure_id": "canonical_structure_id_b", "observed_pic50": "oof_observed_b"}
    )
    joined = joined.merge(
        observed_left, on="canonical_structure_id_a", validate="many_to_one"
    ).merge(observed_right, on="canonical_structure_id_b", validate="many_to_one")
    joined["target_consistent_with_v11_oof"] = (
        (joined["measured_pic50_a"] - joined["oof_observed_a"]).abs().le(target_tolerance)
        & (joined["measured_pic50_b"] - joined["oof_observed_b"]).abs().le(target_tolerance)
    )

    context = _aggregate_observation_context(observations)
    left_context = context.rename(
        columns={
            "structure_id": "canonical_structure_id_a",
            "assay_family": "assay_family_a",
            "source_family": "source_family_a",
        }
    )
    right_context = context.rename(
        columns={
            "structure_id": "canonical_structure_id_b",
            "assay_family": "assay_family_b",
            "source_family": "source_family_b",
        }
    )
    joined = joined.merge(left_context, on="canonical_structure_id_a", how="left", validate="many_to_one")
    joined = joined.merge(right_context, on="canonical_structure_id_b", how="left", validate="many_to_one")
    for column in ("assay_family_a", "assay_family_b", "source_family_a", "source_family_b"):
        joined[column] = joined[column].fillna("unavailable")
    joined["assay_context"] = _pair_context(joined["assay_family_a"], joined["assay_family_b"])
    joined["source_context"] = _pair_context(joined["source_family_a"], joined["source_family_b"])
    joined["replicate_support"] = np.select(
        [
            joined[["observations_a", "observations_b"]].min(axis=1).ge(2),
            joined[["observations_a", "observations_b"]].max(axis=1).ge(2),
        ],
        ["both_endpoints_replicated", "one_endpoint_replicated"],
        default="single_observation_both",
    )
    output_columns = [
        "canonical_pair_id",
        "benchmark_split",
        "leakage_group_id",
        "outer_fold_a",
        "measured_pic50_a",
        "measured_pic50_b",
        "measured_delta_pic50_b_minus_a",
        "observations_a",
        "observations_b",
        "replicate_support",
        "assay_context",
        "source_context",
        "target_consistent_with_v11_oof",
    ]
    labels = joined.loc[:, output_columns].rename(columns={"outer_fold_a": "outer_fold"})
    labels["label_semantics"] = "exact_train_partition_q1_point_target"
    labels["censoring_status"] = "exact_uncensored_by_source_contract"
    audit = {
        "same_outer_fold_pairs_with_exact_train_effects": int(len(labels)),
        "target_consistent_pairs": int(labels["target_consistent_with_v11_oof"].sum()),
        "target_inconsistent_pairs_excluded": int((~labels["target_consistent_with_v11_oof"]).sum()),
    }
    return labels.sort_values("canonical_pair_id").reset_index(drop=True), audit


def _direction(values: pd.Series, threshold: float) -> pd.Series:
    return pd.Series(
        np.select(
            [values.gt(threshold), values.lt(-threshold)],
            ["increased_hERG_liability", "decreased_hERG_liability"],
            default="negligible",
        ),
        index=values.index,
        dtype="string",
    )


def _band(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(values, [-np.inf, *edges, np.inf], labels=labels, right=False).astype("string")


def _scored_rows(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    tie_threshold: float,
    cliff_threshold: float,
) -> pd.DataFrame:
    labels = labels[labels["target_consistent_with_v11_oof"]].copy()
    shared = predictions.merge(
        labels.drop(columns=["benchmark_split", "leakage_group_id", "outer_fold"]),
        on="canonical_pair_id",
        validate="one_to_one",
    )
    rows: list[pd.DataFrame] = []
    for model_id, _, role in BASELINE_MODELS:
        model = shared[
            [
                "canonical_pair_id",
                "benchmark_split",
                "leakage_group_id",
                "outer_fold",
                "mean_endpoint_mw",
                "max_endpoint_mw",
                "minimum_endpoint_train_tanimoto",
                "any_endpoint_extrapolation",
                "measured_delta_pic50_b_minus_a",
                "observations_a",
                "observations_b",
                "replicate_support",
                "assay_context",
                "source_context",
                f"{model_id}_delta_pic50_b_minus_a",
            ]
        ].copy()
        model = model.rename(
            columns={f"{model_id}_delta_pic50_b_minus_a": "predicted_delta_pic50_b_minus_a"}
        )
        model["model_id"] = model_id
        model["model_role"] = role
        rows.append(model)
    scored = pd.concat(rows, ignore_index=True)
    scored["measured_direction"] = _direction(
        scored["measured_delta_pic50_b_minus_a"], tie_threshold
    )
    scored["predicted_direction"] = _direction(
        scored["predicted_delta_pic50_b_minus_a"], 0.0
    )
    scored["predicted_thresholded_direction"] = _direction(
        scored["predicted_delta_pic50_b_minus_a"], tie_threshold
    )
    evaluable = scored["measured_direction"].ne("negligible")
    scored["direction_correct"] = pd.array([pd.NA] * len(scored), dtype="boolean")
    scored.loc[evaluable, "direction_correct"] = scored.loc[
        evaluable, "measured_direction"
    ].eq(scored.loc[evaluable, "predicted_direction"])
    scored["thresholded_direction_correct"] = pd.array([pd.NA] * len(scored), dtype="boolean")
    scored.loc[evaluable, "thresholded_direction_correct"] = scored.loc[
        evaluable, "measured_direction"
    ].eq(scored.loc[evaluable, "predicted_thresholded_direction"])
    scored["delta_error_pic50"] = (
        scored["predicted_delta_pic50_b_minus_a"]
        - scored["measured_delta_pic50_b_minus_a"]
    )
    scored["absolute_delta_error_pic50"] = scored["delta_error_pic50"].abs()
    scored["activity_cliff"] = scored["measured_delta_pic50_b_minus_a"].abs().ge(
        cliff_threshold
    )
    scored["mean_mw_band"] = _band(
        scored["mean_endpoint_mw"],
        [400.0, 500.0, 600.0, 700.0],
        ["<400", "400-<500", "500-<600", "600-<700", ">=700"],
    )
    scored["measured_magnitude_band"] = _band(
        scored["measured_delta_pic50_b_minus_a"].abs(),
        [0.10, 0.30, 1.00],
        ["<0.10", "0.10-<0.30", "0.30-<1.00", ">=1.00"],
    )
    scored["endpoint_oof_support_band"] = _band(
        scored["minimum_endpoint_train_tanimoto"],
        [0.3, 0.5, 0.7],
        ["<0.3", "0.3-<0.5", "0.5-<0.7", ">=0.7"],
    )
    scored["extrapolation_status"] = np.where(
        scored["any_endpoint_extrapolation"], "any_endpoint_extrapolation", "both_in_domain"
    )
    return scored.sort_values(["model_id", "canonical_pair_id"]).reset_index(drop=True)


_SUM_COLUMNS = (
    "n_pairs",
    "n_direction_evaluable",
    "n_direction_correct",
    "n_thresholded_direction_correct",
    "n_predicted_negligible_on_direction_evaluable",
    "absolute_error_sum",
    "squared_error_sum",
    "error_sum",
    "measured_absolute_evaluable_sum",
    "predicted_absolute_evaluable_sum",
    "measured_squared_evaluable_sum",
    "measured_predicted_evaluable_sum",
    "n_activity_cliffs",
    "n_activity_cliff_direction_correct",
    "activity_cliff_absolute_error_sum",
    "measured_sum",
    "predicted_sum",
    "measured_squared_sum",
    "predicted_squared_sum",
    "measured_predicted_sum",
)


def _cluster_contributions(frame: pd.DataFrame) -> pd.DataFrame:
    y = frame["measured_delta_pic50_b_minus_a"].to_numpy(float)
    p = frame["predicted_delta_pic50_b_minus_a"].to_numpy(float)
    error = p - y
    evaluable = frame["measured_direction"].ne("negligible").to_numpy(bool)
    correct = frame["direction_correct"].fillna(False).to_numpy(bool)
    thresholded_correct = frame["thresholded_direction_correct"].fillna(False).to_numpy(bool)
    predicted_negligible = frame["predicted_thresholded_direction"].eq("negligible").to_numpy(bool)
    cliffs = frame["activity_cliff"].to_numpy(bool)
    values = pd.DataFrame(
        {
            "leakage_group_id": frame["leakage_group_id"].astype(str),
            "n_pairs": np.ones(len(frame)),
            "n_direction_evaluable": evaluable.astype(float),
            "n_direction_correct": (evaluable & correct).astype(float),
            "n_thresholded_direction_correct": (evaluable & thresholded_correct).astype(float),
            "n_predicted_negligible_on_direction_evaluable": (
                evaluable & predicted_negligible
            ).astype(float),
            "absolute_error_sum": np.abs(error),
            "squared_error_sum": error**2,
            "error_sum": error,
            "measured_absolute_evaluable_sum": np.where(evaluable, np.abs(y), 0.0),
            "predicted_absolute_evaluable_sum": np.where(evaluable, np.abs(p), 0.0),
            "measured_squared_evaluable_sum": np.where(evaluable, y**2, 0.0),
            "measured_predicted_evaluable_sum": np.where(evaluable, y * p, 0.0),
            "n_activity_cliffs": cliffs.astype(float),
            "n_activity_cliff_direction_correct": (cliffs & correct).astype(float),
            "activity_cliff_absolute_error_sum": np.where(cliffs, np.abs(error), 0.0),
            "measured_sum": y,
            "predicted_sum": p,
            "measured_squared_sum": y**2,
            "predicted_squared_sum": p**2,
            "measured_predicted_sum": y * p,
        }
    )
    return values.groupby("leakage_group_id", observed=True, sort=True)[list(_SUM_COLUMNS)].sum()


def _metrics_from_sums(values: np.ndarray) -> dict[str, float]:
    item = dict(zip(_SUM_COLUMNS, values, strict=True))
    n = item["n_pairs"]
    n_eval = item["n_direction_evaluable"]
    n_cliffs = item["n_activity_cliffs"]
    measured_var = item["measured_squared_sum"] - item["measured_sum"] ** 2 / n if n else math.nan
    predicted_var = item["predicted_squared_sum"] - item["predicted_sum"] ** 2 / n if n else math.nan
    covariance = item["measured_predicted_sum"] - item["measured_sum"] * item["predicted_sum"] / n if n else math.nan
    return {
        "directional_accuracy": item["n_direction_correct"] / n_eval if n_eval else math.nan,
        "thresholded_directional_accuracy": (
            item["n_thresholded_direction_correct"] / n_eval if n_eval else math.nan
        ),
        "predicted_negligible_rate_on_measured_non_ties": (
            item["n_predicted_negligible_on_direction_evaluable"] / n_eval
            if n_eval
            else math.nan
        ),
        "delta_mae_pic50": item["absolute_error_sum"] / n if n else math.nan,
        "delta_rmse_pic50": math.sqrt(item["squared_error_sum"] / n) if n else math.nan,
        "delta_bias_pic50": item["error_sum"] / n if n else math.nan,
        "magnitude_capture_ratio": (
            item["predicted_absolute_evaluable_sum"] / item["measured_absolute_evaluable_sum"]
            if item["measured_absolute_evaluable_sum"]
            else math.nan
        ),
        "signed_delta_slope_through_origin": (
            item["measured_predicted_evaluable_sum"] / item["measured_squared_evaluable_sum"]
            if item["measured_squared_evaluable_sum"]
            else math.nan
        ),
        "delta_pearson_r": (
            covariance / math.sqrt(measured_var * predicted_var)
            if measured_var > 0 and predicted_var > 0
            else math.nan
        ),
        "activity_cliff_directional_accuracy": (
            item["n_activity_cliff_direction_correct"] / n_cliffs if n_cliffs else math.nan
        ),
        "activity_cliff_delta_mae_pic50": (
            item["activity_cliff_absolute_error_sum"] / n_cliffs if n_cliffs else math.nan
        ),
    }


def clustered_metrics(
    frame: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Return point metrics and leakage-group clustered percentile intervals."""

    clusters = _cluster_contributions(frame)
    matrix = clusters.to_numpy(float)
    point = _metrics_from_sums(matrix.sum(axis=0))
    ci: dict[str, dict[str, float | None]] = {
        metric: {"lower": None, "upper": None} for metric in point
    }
    if len(clusters) >= 2 and bootstrap_replicates > 0:
        rng = np.random.default_rng(seed)
        samples: dict[str, list[float]] = defaultdict(list)
        for _ in range(bootstrap_replicates):
            indices = rng.integers(0, len(clusters), size=len(clusters))
            metrics = _metrics_from_sums(matrix[indices].sum(axis=0))
            for metric, value in metrics.items():
                if math.isfinite(value):
                    samples[metric].append(value)
        for metric, values in samples.items():
            if values:
                ci[metric] = {
                    "lower": float(np.quantile(values, 0.025)),
                    "upper": float(np.quantile(values, 0.975)),
                }
    return {
        "n_pairs": int(len(frame)),
        "n_leakage_groups": int(frame["leakage_group_id"].nunique()),
        "n_direction_evaluable": int(frame["measured_direction"].ne("negligible").sum()),
        "n_activity_cliffs": int(frame["activity_cliff"].sum()),
        "metrics": {key: (float(value) if math.isfinite(value) else None) for key, value in point.items()},
        "ci95": ci,
    }


def summarize_metrics(
    scored: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    minimum_pairs: int,
    minimum_groups: int,
) -> pd.DataFrame:
    """Calculate prespecified overall and stratified clustered metrics."""

    stratifiers = (
        "mean_mw_band",
        "measured_magnitude_band",
        "replicate_support",
        "endpoint_oof_support_band",
        "extrapolation_status",
        "assay_context",
        "source_context",
        "outer_fold",
    )
    records: list[dict[str, Any]] = []
    for (model_id, split), model_split in scored.groupby(
        ["model_id", "benchmark_split"], observed=True, sort=True
    ):
        groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", model_split)]
        for stratifier in stratifiers:
            groups.extend(
                (stratifier, str(stratum), group)
                for stratum, group in model_split.groupby(stratifier, observed=True, dropna=False, sort=True)
            )
        for stratifier, stratum, group in groups:
            local_seed = bootstrap_seed + int(
                hashlib.sha256(f"{model_id}|{split}|{stratifier}|{stratum}".encode()).hexdigest()[:8],
                16,
            )
            summary = clustered_metrics(
                group,
                bootstrap_replicates=bootstrap_replicates,
                seed=local_seed,
            )
            support_flag = (
                "adequate_internal_support"
                if summary["n_pairs"] >= minimum_pairs
                and summary["n_leakage_groups"] >= minimum_groups
                else "sparse_descriptive_only"
            )
            record: dict[str, Any] = {
                "model_id": str(model_id),
                "benchmark_split": str(split),
                "stratifier": stratifier,
                "stratum": stratum,
                "n_pairs": summary["n_pairs"],
                "n_leakage_groups": summary["n_leakage_groups"],
                "n_direction_evaluable": summary["n_direction_evaluable"],
                "n_activity_cliffs": summary["n_activity_cliffs"],
                "support_flag": support_flag,
                "ci_method": f"95% percentile bootstrap; leakage-group clusters; {bootstrap_replicates} replicates",
            }
            for metric, value in summary["metrics"].items():
                record[metric] = value
                record[f"{metric}_ci95_lower"] = summary["ci95"][metric]["lower"]
                record[f"{metric}_ci95_upper"] = summary["ci95"][metric]["upper"]
            records.append(record)
    return pd.DataFrame.from_records(records)


def _metric_for_report(metrics: pd.DataFrame, model: str, split: str) -> pd.Series:
    match = metrics[
        metrics["model_id"].eq(model)
        & metrics["benchmark_split"].eq(split)
        & metrics["stratifier"].eq("overall")
    ]
    if len(match) != 1:
        raise BenchmarkError(f"missing unique overall metric row for {model}/{split}")
    return match.iloc[0]


def _format_ci(row: pd.Series, metric: str) -> str:
    value = row[metric]
    lower = row[f"{metric}_ci95_lower"]
    upper = row[f"{metric}_ci95_upper"]
    if pd.isna(value):
        return "not estimable"
    if pd.isna(lower) or pd.isna(upper):
        return f"{float(value):.3f} (CI not estimable)"
    return f"{float(value):.3f} (clustered 95% CI {float(lower):.3f}–{float(upper):.3f})"


def score_baseline(*, inputs: BenchmarkInputs, output_root: Path) -> dict[str, Any]:
    """Score the already-frozen baseline against exact TRAIN MMP outcomes."""

    paths = _output_paths(output_root)
    contract = _verify_existing_lock(output_root)
    if any(path.exists() for path in (paths["evaluation"], paths["validation"])):
        validation = _read_json(paths["validation"], "validation_sha256")
        for binding in validation["artifacts"]:
            _verify_binding(binding)
        return validation
    if any(
        path.exists()
        for path in (
            paths["labels"],
            paths["label_seal"],
            paths["scores"],
            paths["metrics"],
            paths["metrics_csv"],
            paths["report"],
        )
    ):
        raise BenchmarkError("partial baseline score exists; refusing overwrite")

    bound_oof = _source_binding(contract, "pre_existing_nested_oof_predictions")
    bound_manifest = _source_binding(contract, "mmp_source_manifest")
    if _sha256(inputs.nested_oof) != bound_oof["sha256"]:
        raise BenchmarkError("score-baseline OOF input differs from the frozen prediction source")
    if _sha256(inputs.mmp_manifest) != bound_manifest["sha256"]:
        raise BenchmarkError("score-baseline MMP manifest differs from the frozen source")
    identifier_audit = contract["series_campaign_identifier_schema_audit"]
    if identifier_audit["interpretation"] != "no explicit identifier available":
        raise BenchmarkError("explicit series/campaign fields require a new prespecified leakage lock")
    _verify_effect_binding(contract, inputs.training_effects)
    source_manifest = _validate_mmp_manifest(inputs.mmp_manifest)
    if source_manifest["manifest_sha256"] != _read_json(inputs.mmp_manifest)["manifest_sha256"]:
        raise BenchmarkError("MMP manifest changed after freeze")
    manifest = pd.read_parquet(paths["manifest"])
    predictions = pd.read_parquet(paths["predictions"])
    effects = _read_columns(inputs.training_effects, list(EFFECT_COLUMNS))
    if not effects["exploratory_training_only"].astype(bool).all():
        raise BenchmarkError("MMP effects include a non-training row")
    oof_observed = _read_columns(inputs.nested_oof, ["structure_id", "observed_pic50"])
    observations = _read_columns(
        inputs.exact_observations,
        ["structure_id", "source_family", "assay_family"],
    )
    plan = contract["prespecified_analysis"]
    labels, label_audit = _prepare_locked_labels(
        manifest,
        effects,
        oof_observed,
        observations,
        target_tolerance=float(plan["target_consistency_tolerance_pic50"]),
    )
    _atomic_parquet(paths["labels"], labels)
    label_seal = _atomic_json(
        paths["label_seal"],
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "train_only_exact_labels_opened_after_prediction_lock",
            "lock_contract_sha256": contract["contract_sha256"],
            "source_binding": _binding(inputs.training_effects, "train_only_mmp_effects"),
            "oof_observed_binding": _binding(inputs.nested_oof, "same_frozen_oof_with_observed_train_target"),
            "observation_context_binding": _binding(
                inputs.exact_observations, "train_only_assay_source_context"
            ),
            "artifact": _binding(paths["labels"], "sealed_train_only_pair_labels"),
            "audit": label_audit,
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "prospective_showcase_or_final_test_measurements_opened": False,
            "censoring": "exact uncensored Q1 point targets only; no censored value coerced",
        },
        "label_seal_sha256",
    )
    scored = _scored_rows(
        predictions,
        labels,
        tie_threshold=float(plan["tie_threshold_pic50"]),
        cliff_threshold=float(plan["activity_cliff_threshold_absolute_pic50"]),
    )
    if any("structure" in column.lower() or "smiles" in column.lower() for column in scored.columns):
        raise BenchmarkError("scored output would expose structure-level fields")
    _atomic_parquet(paths["scores"], scored)
    metrics = summarize_metrics(
        scored,
        bootstrap_replicates=int(plan["cluster_bootstrap_replicates"]),
        bootstrap_seed=int(plan["cluster_bootstrap_seed"]),
        minimum_pairs=int(plan["minimum_support_pairs"]),
        minimum_groups=int(plan["minimum_support_leakage_groups"]),
    )
    _atomic_parquet(paths["metrics"], metrics)
    metrics.to_csv(paths["metrics_csv"], index=False)

    primary_test = _metric_for_report(metrics, "v11_nested", "locked_test")
    primary_development = _metric_for_report(metrics, "v11_nested", "development")
    headline = {
        "locked_test": {
            key: (None if pd.isna(primary_test[key]) else float(primary_test[key]))
            for key in (
                "n_pairs",
                "n_leakage_groups",
                "n_direction_evaluable",
                "n_activity_cliffs",
                "directional_accuracy",
                "directional_accuracy_ci95_lower",
                "directional_accuracy_ci95_upper",
                "delta_mae_pic50",
                "delta_mae_pic50_ci95_lower",
                "delta_mae_pic50_ci95_upper",
                "magnitude_capture_ratio",
                "magnitude_capture_ratio_ci95_lower",
                "magnitude_capture_ratio_ci95_upper",
                "activity_cliff_directional_accuracy",
                "activity_cliff_delta_mae_pic50",
            )
        },
        "development": {
            key: (None if pd.isna(primary_development[key]) else float(primary_development[key]))
            for key in (
                "n_pairs",
                "n_leakage_groups",
                "directional_accuracy",
                "delta_mae_pic50",
                "magnitude_capture_ratio",
            )
        },
    }
    evaluation = _atomic_json(
        paths["evaluation"],
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "baseline_scored_against_locked_train_only_exact_pairs",
            "lock_contract_sha256": contract["contract_sha256"],
            "prediction_seal_sha256": contract["prediction_seal_sha256"],
            "label_seal_sha256": label_seal["label_seal_sha256"],
            "primary_baseline_model": "v11_nested",
            "headline_metrics": headline,
            "label_audit": label_audit,
            "direction_definition": (
                "delta pIC50 = candidate/B minus parent/A; positive means stronger hERG inhibition "
                "and increased predicted liability"
            ),
            "interpretation_boundary": (
                "Internal train-only retrospective OOF evidence; not an external, prospective, "
                "blinded, clinical, or independent validation"
            ),
            "repository_validation_labels_opened": False,
            "repository_test_labels_opened": False,
            "prospective_showcase_or_final_test_measurements_opened": False,
            "artifacts": [
                _binding(paths["scores"], "baseline_scored_pairs_without_structures"),
                _binding(paths["metrics"], "baseline_clustered_metrics"),
                _binding(paths["metrics_csv"], "baseline_clustered_metrics_csv"),
            ],
        },
        "evaluation_sha256",
    )
    report = "\n".join(
        [
            "# Locked hERG matched-pair baseline",
            "",
            "## Prespecified primary baseline",
            "",
            f"- Locked-test pairs: {int(primary_test['n_pairs']):,} across {int(primary_test['n_leakage_groups']):,} leakage groups.",
            f"- Directional accuracy: {_format_ci(primary_test, 'directional_accuracy')}.",
            f"- Delta-pIC50 MAE: {_format_ci(primary_test, 'delta_mae_pic50')}.",
            f"- Magnitude capture ratio: {_format_ci(primary_test, 'magnitude_capture_ratio')}.",
            f"- Activity-cliff direction accuracy: {_format_ci(primary_test, 'activity_cliff_directional_accuracy')}.",
            "",
            "Positive delta pIC50 means stronger hERG inhibition and therefore increased liability. "
            f"Measured changes with |delta| <= {float(plan['tie_threshold_pic50']):.2f} are excluded from primary directional accuracy, which scores the sign of each nonzero prediction. "
            f"Activity cliffs are prespecified as |delta| >= {float(plan['activity_cliff_threshold_absolute_pic50']):.2f}.",
            "",
            "## Evidence boundary",
            "",
            "This is internal TRAIN-partition retrospective evidence using pre-existing nested scaffold OOF predictions. "
            "It is not external, blinded, prospective, clinical, or independent validation. Repository validation/test "
            "labels and showcase/final-test measurements were not opened.",
            "",
            "The lock canonicalizes reversed pairs, keeps every structure and scaffold in one benchmark split, and "
            "uses the full MMP connected component as a series proxy. Explicit series and campaign identifiers are "
            "absent; campaign leakage cannot be assessed. Clustered intervals resample these leakage groups. Censored "
            "measurements are not coerced: this benchmark contains only exact Q1 point targets.",
            "",
            "No SMILES, molecular structures, MMP cores, or transformations are emitted.",
        ]
    )
    paths["report"].write_text(report + "\n", encoding="utf-8")
    validation = _atomic_json(
        paths["validation"],
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "passed",
            "contract_sha256": contract["contract_sha256"],
            "evaluation_sha256": evaluation["evaluation_sha256"],
            "checks": {
                "prediction_frozen_before_label_open": True,
                "target_consistency_filter_applied_after_lock": True,
                "canonical_pair_duplicates": int(manifest["canonical_pair_id"].duplicated().sum()),
                "scored_model_pair_duplicates": int(
                    scored.duplicated(["model_id", "canonical_pair_id"]).sum()
                ),
                "scored_structure_or_smiles_columns": [],
                "repository_validation_or_test_labels_opened": False,
            },
            "artifacts": [
                _binding(paths["contract"], "lock_contract"),
                _binding(paths["manifest"], "locked_pair_manifest"),
                _binding(paths["predictions"], "baseline_oof_predictions_before_score"),
                _binding(paths["prediction_seal"], "baseline_prediction_seal"),
                _binding(paths["labels"], "sealed_train_only_pair_labels"),
                _binding(paths["label_seal"], "train_only_label_seal"),
                _binding(paths["scores"], "baseline_scored_pairs_without_structures"),
                _binding(paths["metrics"], "baseline_clustered_metrics"),
                _binding(paths["metrics_csv"], "baseline_clustered_metrics_csv"),
                _binding(paths["evaluation"], "baseline_evaluation"),
                _binding(paths["report"], "human_readable_report"),
            ],
        },
        "validation_sha256",
    )
    return validation


def _parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[2]
    defaults = default_inputs(repo)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "score-baseline", "validate"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo / "research/local_runs/herg_v15_finalize/paired",
    )
    parser.add_argument("--pair-registry", type=Path, default=defaults.pair_registry)
    parser.add_argument("--training-effects", type=Path, default=defaults.training_effects)
    parser.add_argument("--mmp-manifest", type=Path, default=defaults.mmp_manifest)
    parser.add_argument("--nested-oof", type=Path, default=defaults.nested_oof)
    parser.add_argument("--exact-observations", type=Path, default=defaults.exact_observations)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    return parser


def main() -> None:
    args = _parser().parse_args()
    inputs = BenchmarkInputs(
        pair_registry=args.pair_registry.resolve(),
        training_effects=args.training_effects.resolve(),
        mmp_manifest=args.mmp_manifest.resolve(),
        nested_oof=args.nested_oof.resolve(),
        exact_observations=args.exact_observations.resolve(),
    )
    output_root = args.output_root.resolve()
    if args.command == "freeze":
        result = freeze_benchmark(
            inputs=inputs,
            output_root=output_root,
            seed=args.seed,
            test_fraction=args.test_fraction,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    elif args.command == "score-baseline":
        result = score_baseline(inputs=inputs, output_root=output_root)
    else:
        result = _verify_existing_lock(output_root)
        validation_path = _output_paths(output_root)["validation"]
        if validation_path.exists():
            validation = _read_json(validation_path, "validation_sha256")
            for binding in validation["artifacts"]:
                _verify_binding(binding)
            result = validation
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
