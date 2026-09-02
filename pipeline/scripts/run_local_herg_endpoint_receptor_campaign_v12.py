#!/usr/bin/env python3
"""Build the coherent endpoint and receptor-state hERG V12 evidence layer.

V12 deliberately separates three scientific objects:

1. V11 is the quantitative IC50 anchor trained on the 18,801-structure exact
   surface under nested scaffold evaluation.
2. Direct PubChem qHTS curve crossings provide IC10/IC30 spacing evidence.
   V12 never calls fixed-dose screens or Hill-derived values direct IC10/IC30.
3. Six deposited hERG coordinate states define a receptor-state contract.
   Coordinate comparisons are receptor aware, but are not represented as
   docking, binding energies, or molecular dynamics.

The endpoint parameterization makes IC10 <= IC30 <= IC50 true by construction.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import shutil
import sys
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

SCHEMA_VERSION = "platform-local-herg-endpoint-receptor-v12/1.0"
SEED = 20260819
RECEPTOR_IDS = ("8ZYN", "8ZYP", "9CHP", "9CHQ", "8ZYO", "8ZYQ")
CORE_IDS = ("8ZYN", "8ZYP", "9CHP", "9CHQ")


class CampaignError(RuntimeError):
    """Integrity, scientific-contract, or evidence-boundary failure."""


class _Lock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> _Lock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError("V12 is already running") from exc
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


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


def _read_json(path: Path, hash_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    expected = payload.pop(hash_field)
    actual = hashlib.sha256(_canonical(payload)).hexdigest()
    payload[hash_field] = expected
    if actual != expected:
        raise CampaignError(f"self-hash mismatch: {path}")
    return payload


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _binding(path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise CampaignError(f"missing {role}: {path}")
    row: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }
    if path.suffix == ".parquet":
        row["rows"] = pq.read_metadata(path).num_rows
    return row


def _verify_binding(row: dict[str, Any]) -> None:
    path = Path(row["path"])
    if not path.is_file() or path.stat().st_size != int(row["bytes"]) or _sha(path) != row["sha256"]:
        raise CampaignError(f"bound artifact changed: {path}")
    if path.suffix == ".parquet" and pq.read_metadata(path).num_rows != int(row["rows"]):
        raise CampaignError(f"bound row count changed: {path}")


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CampaignError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    rho = spearmanr(observed, predicted).statistic
    return {
        "n": int(len(observed)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "spearman": float(rho) if np.isfinite(rho) else 0.0,
        "within_0_5": float(np.mean(np.abs(observed - predicted) <= 0.5)),
        "within_1_0": float(np.mean(np.abs(observed - predicted) <= 1.0)),
    }


def _project_nonincreasing(values: np.ndarray) -> np.ndarray:
    """Least-squares PAVA projection onto x0 >= x1 >= ... >= xn."""
    blocks: list[tuple[float, int]] = []
    for value in -np.asarray(values, dtype=float):
        blocks.append((float(value), 1))
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right_value, right_n = blocks.pop()
            left_value, left_n = blocks.pop()
            total = left_n + right_n
            blocks.append(((left_value * left_n + right_value * right_n) / total, total))
    projected = np.concatenate([np.repeat(value, count) for value, count in blocks])
    return -projected


def _prepare_direct(labels_path: Path) -> pd.DataFrame:
    labels = pd.read_parquet(labels_path)
    required = [
        "standard_inchi_key",
        "standardized_smiles",
        "scaffold_group_id",
        "empirical_ic10_um",
        "empirical_ic30_um",
        "empirical_ic50_um",
        "strict_curve_qc",
    ]
    missing = set(required) - set(labels.columns)
    if missing:
        raise CampaignError(f"direct endpoint labels missing {sorted(missing)}")
    strict = labels.loc[labels.strict_curve_qc.astype(bool)].dropna(
        subset=["empirical_ic10_um", "empirical_ic30_um", "empirical_ic50_um"]
    )
    strict = strict.loc[
        (strict.empirical_ic10_um > 0)
        & (strict.empirical_ic30_um > 0)
        & (strict.empirical_ic50_um > 0)
    ].copy()
    if (strict.empirical_ic10_um > strict.empirical_ic30_um + 1e-12).any() or (
        strict.empirical_ic30_um > strict.empirical_ic50_um + 1e-12
    ).any():
        raise CampaignError("direct empirical endpoint order is physically incoherent")
    aggregated = (
        strict.groupby("standard_inchi_key", as_index=False)
        .agg(
            standardized_smiles=("standardized_smiles", "first"),
            scaffold_group_id=("scaffold_group_id", "first"),
            empirical_ic10_um=("empirical_ic10_um", "median"),
            empirical_ic30_um=("empirical_ic30_um", "median"),
            empirical_ic50_um=("empirical_ic50_um", "median"),
            source_curve_rows=("standard_inchi_key", "size"),
        )
        .sort_values("standard_inchi_key")
        .reset_index(drop=True)
    )
    if len(aggregated) < 500:
        raise CampaignError("insufficient strict paired direct IC10/IC30/IC50 evidence")
    for endpoint in (10, 30, 50):
        aggregated[f"observed_pic{endpoint}"] = 6.0 - np.log10(
            aggregated[f"empirical_ic{endpoint}_um"].to_numpy(float)
        )
    aggregated["gap_pic10_minus_pic30"] = aggregated.observed_pic10 - aggregated.observed_pic30
    aggregated["gap_pic30_minus_pic50"] = aggregated.observed_pic30 - aggregated.observed_pic50
    if (aggregated[["gap_pic10_minus_pic30", "gap_pic30_minus_pic50"]] < -1e-10).any().any():
        raise CampaignError("negative direct endpoint gap after aggregation")
    return aggregated


def _feature_cache(repo: Path, output: Path, direct: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    path = output / "prepared/direct_paired_features.parquet"
    if path.is_file():
        cached = pd.read_parquet(path)
        if set(cached.standard_inchi_key) == set(direct.standard_inchi_key):
            return cached
    v101 = _load_module("herg_v101_for_v12", repo / "pipeline/scripts/run_local_herg_v10_1_expanded_platform.py")
    rows: list[pd.DataFrame] = []
    total = len(direct)
    for index, row in direct.iterrows():
        frame = v101._feature_row(str(row.standardized_smiles), columns)  # noqa: SLF001
        frame.insert(0, "standard_inchi_key", str(row.standard_inchi_key))
        rows.append(frame)
        if (index + 1) % 100 == 0:
            print(f"V12 direct features {index + 1:,}/{total:,}", flush=True)
    features = pd.concat(rows, ignore_index=True)
    numeric = features[columns].replace([np.inf, -np.inf], np.nan)
    numeric = numeric.mask(numeric.abs() > 1e30)
    features.loc[:, columns] = numeric
    _parquet(path, features)
    return features


def _gap_model(workers: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=400,
                    max_features=0.5,
                    min_samples_leaf=3,
                    n_jobs=workers,
                    random_state=seed,
                ),
            ),
        ]
    )


def _endpoint_campaign(
    repo: Path, output: Path, v101: Path, workers: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels_path = v101 / "evidence/empirical_ic10_ic30_ic50_labels.parquet"
    direct_oof_path = v101 / "evidence/empirical_ic10_ic30_ic50_oof.parquet"
    bundle10 = joblib.load(v101 / "models/empirical_ic10_regressor.joblib")
    columns = list(bundle10["feature_columns"])
    direct = _prepare_direct(labels_path)
    features = _feature_cache(repo, output, direct, columns)
    joined = direct.merge(features, on="standard_inchi_key", validate="one_to_one")
    x = joined[columns]
    groups = joined.scaffold_group_id.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    prediction10 = np.full(len(joined), np.nan)
    prediction30 = np.full(len(joined), np.nan)
    folds = np.full(len(joined), -1, dtype=int)
    y10 = joined.gap_pic10_minus_pic30.to_numpy(float)
    y30 = joined.gap_pic30_minus_pic50.to_numpy(float)
    for fold, (fit, evaluation) in enumerate(splitter.split(x, groups=groups)):
        if set(groups[fit]) & set(groups[evaluation]):
            raise CampaignError("scaffold leakage in direct endpoint gap model")
        model10 = _gap_model(workers, SEED + fold)
        model30 = _gap_model(workers, SEED + 100 + fold)
        model10.fit(x.iloc[fit], y10[fit])
        model30.fit(x.iloc[fit], y30[fit])
        prediction10[evaluation] = np.maximum(0.0, model10.predict(x.iloc[evaluation]))
        prediction30[evaluation] = np.maximum(0.0, model30.predict(x.iloc[evaluation]))
        folds[evaluation] = fold
    if not np.isfinite(prediction10).all() or not np.isfinite(prediction30).all() or (folds < 0).any():
        raise CampaignError("incomplete direct endpoint OOF")
    isolated = joined[
        [
            "standard_inchi_key",
            "scaffold_group_id",
            "observed_pic10",
            "observed_pic30",
            "observed_pic50",
        ]
    ].copy()
    isolated["outer_fold"] = folds
    isolated["predicted_gap_pic10_minus_pic30"] = prediction10
    isolated["predicted_gap_pic30_minus_pic50"] = prediction30
    isolated["predicted_pic50"] = isolated.observed_pic50
    isolated["predicted_pic30"] = isolated.predicted_pic50 + prediction30
    isolated["predicted_pic10"] = isolated.predicted_pic30 + prediction10
    isolated["evaluation_scope"] = "gap-model isolation using observed direct IC50 anchor"
    if (isolated.predicted_pic10 + 1e-12 < isolated.predicted_pic30).any() or (
        isolated.predicted_pic30 + 1e-12 < isolated.predicted_pic50
    ).any():
        raise CampaignError("coherent gap parameterization failed")
    isolated_path = output / "endpoints/coherent_gap_oof.parquet"
    _parquet(isolated_path, isolated)

    raw = pd.read_parquet(direct_oof_path)
    pivot = raw.pivot_table(
        index=["standard_inchi_key", "scaffold_group_id"],
        columns="endpoint",
        values=["observed_picx", "predicted_picx"],
        aggfunc="mean",
    )
    pivot.columns = [f"{left}_{right}" for left, right in pivot.columns]
    pivot = pivot.reset_index().dropna(
        subset=[
            "observed_picx_IC10",
            "observed_picx_IC30",
            "observed_picx_IC50",
            "predicted_picx_IC10",
            "predicted_picx_IC30",
            "predicted_picx_IC50",
        ]
    )
    projected_rows: list[dict[str, Any]] = []
    raw_violations = 0
    for row in pivot.itertuples(index=False):
        raw_values = np.array(
            [row.predicted_picx_IC10, row.predicted_picx_IC30, row.predicted_picx_IC50], dtype=float
        )
        raw_violations += int(raw_values[0] < raw_values[1] or raw_values[1] < raw_values[2])
        coherent = _project_nonincreasing(raw_values)
        projected_rows.append(
            {
                "standard_inchi_key": row.standard_inchi_key,
                "scaffold_group_id": row.scaffold_group_id,
                "observed_pic10": row.observed_picx_IC10,
                "observed_pic30": row.observed_picx_IC30,
                "observed_pic50": row.observed_picx_IC50,
                "raw_predicted_pic10": raw_values[0],
                "raw_predicted_pic30": raw_values[1],
                "raw_predicted_pic50": raw_values[2],
                "coherent_predicted_pic10": coherent[0],
                "coherent_predicted_pic30": coherent[1],
                "coherent_predicted_pic50": coherent[2],
            }
        )
    projected = pd.DataFrame(projected_rows)
    projected_path = output / "endpoints/separate_model_coherence_diagnostic.parquet"
    _parquet(projected_path, projected)
    coherent_violations = int(
        (
            (projected.coherent_predicted_pic10 < projected.coherent_predicted_pic30 - 1e-12)
            | (projected.coherent_predicted_pic30 < projected.coherent_predicted_pic50 - 1e-12)
        ).sum()
    )
    full10 = _gap_model(workers, SEED + 1000).fit(x, y10)
    full30 = _gap_model(workers, SEED + 2000).fit(x, y30)
    model_path = output / "models/coherent_endpoint_gap_models.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "feature_columns": columns,
            "pic10_minus_pic30_model": full10,
            "pic30_minus_pic50_model": full30,
            "ic50_anchor": "V11 final model bundle",
            "endpoint_order": "IC10 <= IC30 <= IC50; pIC10 >= pIC30 >= pIC50",
            "direct_label_method": "strict observed qHTS curve crossings; not Hill-derived",
            "paired_direct_structures": len(joined),
        },
        model_path,
        compress=3,
    )
    result = {
        "paired_direct_structures": int(len(joined)),
        "direct_label_method": "observed qHTS curve crossings with strict QC; not Hill-derived",
        "gap_oof": {
            "pic10": _metrics(isolated.observed_pic10.to_numpy(), isolated.predicted_pic10.to_numpy()),
            "pic30": _metrics(isolated.observed_pic30.to_numpy(), isolated.predicted_pic30.to_numpy()),
            "anchor_scope": "observed direct IC50 supplied to isolate gap-model error",
        },
        "separate_model_complete_case_structures": int(len(projected)),
        "separate_model_order_violations_before_projection": int(raw_violations),
        "order_violations_after_projection": coherent_violations,
        "deployment_rule": "V11 pIC50 + nonnegative predicted pIC30-pIC50 + nonnegative predicted pIC10-pIC30",
    }
    report_path = output / "endpoints/endpoint_report.json"
    report = _json(
        report_path,
        {"schema_version": SCHEMA_VERSION, "status": "passed", **result},
        "report_sha256",
    )
    artifacts = [
        _binding(isolated_path, "coherent_gap_oof"),
        _binding(projected_path, "separate_model_coherence_diagnostic"),
        _binding(model_path, "coherent_endpoint_gap_models"),
        _binding(report_path, "endpoint_report"),
    ]
    return report, artifacts


def _receptor_campaign(repo: Path, output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit = repo / "research/simulations/pk_herg/local_m3_pilot/receptor_state_analysis"
    selection_path = audit / "receptor_state_selection.csv"
    pocket_path = audit / "pocket_symmetry_metrics.csv"
    contrast_path = audit / "pairwise_coordinate_contrasts.csv"
    contacts_path = audit / "deposited_ligand_contacts.csv"
    gate_path = (
        repo
        / "research/simulations/pk_herg/local_m3_pilot/site_specific_extension/core_receptor_preparation_gate.json"
    )
    selection = pd.read_csv(selection_path)
    if tuple(selection.pdb_id.astype(str)) != RECEPTOR_IDS:
        raise CampaignError("unexpected receptor-state ordering or membership")
    if set(selection.loc[selection.production_tier == "core", "pdb_id"]) != set(CORE_IDS):
        raise CampaignError("four-core receptor contract changed")
    pocket = pd.read_csv(pocket_path)
    contrasts = pd.read_csv(contrast_path)
    pd.read_csv(contacts_path)
    gate = json.loads(gate_path.read_text())
    state = selection.copy()
    pocket_wide = pocket.pivot(index="pdb_id", columns="residue", values=[
        "ring_centroid_pair_distance_mean_angstrom",
        "ring_radial_distance_mean_angstrom",
        "ring_radial_distance_sd_angstrom",
    ])
    pocket_wide.columns = [f"{left}__{right}" for left, right in pocket_wide.columns]
    state = state.merge(pocket_wide.reset_index(), on="pdb_id", validate="one_to_one")
    state["mean_pairwise_filter_rmsd_angstrom"] = state.pdb_id.map(
        contrasts.loc[contrasts.region == "selectivity_filter"].groupby("mobile_pdb_id")
        .post_scaffold_alignment_rmsd_angstrom.mean()
    ).fillna(0.0)
    state["mean_pairwise_cavity_rmsd_angstrom"] = state.pdb_id.map(
        contrasts.loc[contrasts.region == "cavity_s6"].groupby("mobile_pdb_id")
        .post_scaffold_alignment_rmsd_angstrom.mean()
    ).fillna(0.0)
    state_path = output / "receptors/receptor_state_features.parquet"
    _parquet(state_path, state)
    cif_paths = {
        "8ZYN": repo / "research/literature/herg/structural_biology/2024_miyashita_inhibitor_bound/8ZYN.cif",
        "8ZYO": repo / "research/literature/herg/structural_biology/2024_miyashita_inhibitor_bound/8ZYO.cif",
        "8ZYP": repo / "research/literature/herg/structural_biology/2024_miyashita_inhibitor_bound/8ZYP.cif",
        "8ZYQ": repo / "research/literature/herg/structural_biology/2024_miyashita_inhibitor_bound/8ZYQ.cif",
        "9CHP": repo / "research/literature/herg/structural_biology/2024_lau_potassium_states/9CHP.cif",
        "9CHQ": repo / "research/literature/herg/structural_biology/2024_lau_potassium_states/9CHQ.cif",
    }
    cif_bindings = [_binding(cif_paths[pdb], f"receptor_coordinate_{pdb}") for pdb in RECEPTOR_IDS]
    vina = shutil.which("vina")
    meeko = shutil.which("mk_prepare_receptor.py") or shutil.which("mk_prepare_receptor")
    try:
        import openmm  # type: ignore[import-not-found]  # noqa: F401

        openmm_available = True
    except Exception:
        openmm_available = False
    true_docking_ready = bool(
        vina
        and meeko
        and gate.get("simulation_preparation_gate") == "passed"
    )
    if true_docking_ready:
        # V12 requires a separately reviewed docking protocol before execution.
        raise CampaignError("docking tools appeared, but no reviewed V12 docking protocol is bound")
    result = {
        "receptor_state_count": 6,
        "core_state_ids": list(CORE_IDS),
        "sensitivity_state_ids": ["8ZYO", "8ZYQ"],
        "core_strategy": "four core states evaluated first; sensitivity states remain separate",
        "coordinate_state_aware": True,
        "query_ligand_pose_aware": False,
        "true_docking_executed": False,
        "molecular_dynamics_executed": False,
        "vina_available": bool(vina),
        "meeko_available": bool(meeko),
        "openmm_available": openmm_available,
        "preparation_gate": gate.get("simulation_preparation_gate"),
        "truth_boundary": gate.get("truth_boundary"),
        "next_receptor_step": "review construct/protonation/membrane policy, prepare all four core systems, then preregister docking and score aggregation before ligand scoring",
    }
    report_path = output / "receptors/receptor_contract.json"
    report = _json(
        report_path,
        {"schema_version": SCHEMA_VERSION, "status": "passed_coordinate_contract", **result},
        "report_sha256",
    )
    artifacts = [
        _binding(state_path, "receptor_state_features"),
        _binding(report_path, "receptor_contract"),
        _binding(selection_path, "receptor_state_selection"),
        _binding(pocket_path, "pocket_symmetry_metrics"),
        _binding(contrast_path, "pairwise_coordinate_contrasts"),
        _binding(contacts_path, "deposited_ligand_contacts"),
        _binding(gate_path, "receptor_preparation_gate"),
        *cif_bindings,
    ]
    return report, artifacts


def _validate(output: Path) -> dict[str, Any]:
    summary = _read_json(output / "final_summary.json", "summary_sha256")
    manifest = _read_json(output / "manifest.json", "manifest_sha256")
    if summary.get("status") != "complete":
        raise CampaignError("V12 is incomplete")
    for binding in manifest["inputs"] + manifest["artifacts"]:
        _verify_binding(binding)
    endpoints = _read_json(output / "endpoints/endpoint_report.json", "report_sha256")
    receptor = _read_json(output / "receptors/receptor_contract.json", "report_sha256")
    if endpoints["order_violations_after_projection"] != 0:
        raise CampaignError("endpoint coherence validation failed")
    if endpoints["paired_direct_structures"] < 500:
        raise CampaignError("direct endpoint support gate failed")
    if receptor["receptor_state_count"] != 6 or receptor["true_docking_executed"]:
        raise CampaignError("receptor truth boundary failed")
    return {
        "status": "passed",
        "paired_direct_structures": endpoints["paired_direct_structures"],
        "endpoint_order_violations": 0,
        "receptor_states_bound": 6,
        "coordinate_state_aware": True,
        "true_docking_executed": False,
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "summary_sha256": summary["summary_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    output = args.output_root.resolve()
    v11 = args.v11_root.resolve()
    v101 = args.v101_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.workers < 1 or args.workers > 6:
        raise CampaignError("workers must be in [1, 6]")
    if (output / "final_summary.json").is_file():
        return _validate(output)
    with _Lock(output / ".campaign.lock"):
        v11_module = _load_module("herg_v11_for_v12", repo / "pipeline/scripts/run_local_herg_comprehensive_optimization_v11.py")
        v11_validation = v11_module._validate(v11)  # noqa: SLF001
        inputs = [
            _binding(repo / "pipeline/scripts/run_local_herg_endpoint_receptor_campaign_v12.py", "v12_implementation"),
            _binding(repo / "pipeline/scripts/run_local_herg_comprehensive_optimization_v11.py", "v11_implementation"),
            _binding(v11 / "final_summary.json", "v11_final_summary"),
            _binding(v11 / "analysis/nested_oof_predictions.parquet", "v11_nested_oof"),
            _binding(v11 / "final_model/model_bundle.joblib", "v11_ic50_model_bundle"),
            _binding(v101 / "evidence/empirical_ic10_ic30_ic50_labels.parquet", "direct_endpoint_labels"),
            _binding(v101 / "evidence/empirical_ic10_ic30_ic50_oof.parquet", "direct_endpoint_oof"),
        ]
        endpoint_report, endpoint_artifacts = _endpoint_campaign(repo, output, v101, args.workers)
        receptor_report, receptor_artifacts = _receptor_campaign(repo, output)
        manifest = _json(
            output / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "passed",
                "created_utc": _utc(),
                "inputs": inputs,
                "artifacts": endpoint_artifacts + receptor_artifacts,
                "scientific_scope": {
                    "v11_ic50_anchor": True,
                    "direct_ic10_ic30_only": True,
                    "endpoint_order_enforced": True,
                    "coordinate_state_aware": True,
                    "true_docking_claimed": False,
                    "repository_validation_labels_opened": False,
                    "repository_test_labels_opened": False,
                },
            },
            "manifest_sha256",
        )
        summary = _json(
            output / "final_summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "finished_utc": _utc(),
                "v11_nested_mae": v11_validation["v11_nested_mae"],
                "paired_direct_structures": endpoint_report["paired_direct_structures"],
                "endpoint_order_violations": 0,
                "receptor_states_bound": receptor_report["receptor_state_count"],
                "coordinate_state_aware": True,
                "true_docking_executed": False,
                "manifest_sha256": manifest["manifest_sha256"],
            },
            "summary_sha256",
        )
        return _validate(output) | {"final_summary": summary}


def _status(output: Path) -> dict[str, Any]:
    if (output / "final_summary.json").is_file():
        return _validate(output)
    return {
        "status": "incomplete",
        "direct_feature_cache": (output / "prepared/direct_paired_features.parquet").is_file(),
        "endpoint_report": (output / "endpoints/endpoint_report.json").is_file(),
        "receptor_contract": (output / "receptors/receptor_contract.json").is_file(),
        "resume": "rerun the identical launcher",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "status", "validate"):
        child = subparsers.add_parser(command)
        child.add_argument("--repo-root", type=Path, required=True)
        child.add_argument("--v11-root", type=Path, required=True)
        child.add_argument("--v101-root", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument("--workers", type=int, default=6)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "run":
            result = _run(args)
        elif args.command == "status":
            result = _status(args.output_root.resolve())
        else:
            result = _validate(args.output_root.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (CampaignError, ValueError, KeyError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
