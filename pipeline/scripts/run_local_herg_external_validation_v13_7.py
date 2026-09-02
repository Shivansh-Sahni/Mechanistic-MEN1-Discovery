#!/usr/bin/env python3
"""Run blinded publisher-declared external validation for the hERG platform.

V13.7 evaluates the frozen V9/V10.1 ligand models and the frozen V13.6
8ZYO/F649 hybrid on external structures from the official supplement to
Zhang et al. (2025), DOI 10.1021/acs.chemrestox.5c00065.  The workflow is
deliberately staged:

1. ``prepare`` standardizes structures and audits exact/scaffold overlap while
   writing outcomes to a separate sealed label artifact;
2. ``baseline`` predicts with the deployed V9/V10.1 bundles using only the
   label-free structure registry and seals those predictions;
3. ``dock`` docks only V9-scaffold-novel, project-exact-novel structures to
   the frozen 8ZYO receptor with the validated V13 AutoDock Vina protocol;
4. ``receptor`` averages the five frozen discovery-fold receptor classifiers,
   applies the unchanged V13.6 policy, and seals decisions; and
5. ``score`` opens labels, reports overlap-stratified metrics, and never tunes
   a model, threshold, feature, or policy on these outcomes.

AutoDock Vina scores are scoring-function observables, not binding free
energies or proof of a unique binding pose.  The external studies remain
retrospective literature data, not prospective experimental validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
for candidate in (SCRIPT_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import analyze_local_herg_receptor_ensemble_v13_2 as v132  # noqa: E402
import run_local_herg_receptor_classification_confirmation_v13_3 as v133  # noqa: E402
import run_local_herg_receptor_classification_replication_v13_4 as v134  # noqa: E402
import run_local_herg_receptor_ensemble_campaign_v13 as v13  # noqa: E402
import run_local_herg_receptor_hybrid_validation_v13_6 as v136  # noqa: E402
import run_local_herg_v10_1_expanded_platform as v101  # noqa: E402
from menin_discovery.chemistry import standardize_smiles  # noqa: E402
from menin_discovery.features import scaffold_key  # noqa: E402

SCHEMA_VERSION = "platform-local-herg-external-validation-v13.7/1.0"
SEED = 20260821
STATE = "8ZYO"
PRIMARY_FEATURE = "dock__8ZYO__contact_649_count"
DEFAULT_SOURCE = Path("research/external_validation_sources/acs_2025_external_sets.json")
DEFAULT_SUPPLEMENT = Path("research/external_validation_sources/tx5c00065_si_002.xlsx")
DEFAULT_OUTPUT = Path("research/local_runs/herg_external_validation_v13_7")
DEFAULT_V101 = Path("research/local_runs/herg_v10_1_expanded_platform")
DEFAULT_V9 = Path("research/local_runs/herg_domain_mixture_campaign_v9")
DEFAULT_V13 = Path("research/local_runs/herg_receptor_ensemble_campaign_v13")
DEFAULT_V132 = Path("research/local_runs/herg_receptor_strict_analysis_v13_2")
EXPECTED_SUPPLEMENT_SHA256 = "28a0c203691eef1d4e9cfbc0da022573ad107e213514eb48de529eafd3444170"
QUANTITATIVE_SETS = (
    "ev2_exact_quantitative",
    "prior_study_external_exact_quantitative",
)
BINARY_SET = "ev3_binary_classification"


class CampaignError(RuntimeError):
    """Raised when an external-validation integrity invariant is violated."""


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


def _read_json(path: Path, hash_field: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise CampaignError(f"expected JSON object: {path}")
    if hash_field:
        expected = payload.get(hash_field)
        body = dict(payload)
        body.pop(hash_field, None)
        actual = hashlib.sha256(_canonical(body)).hexdigest()
        if expected != actual:
            raise CampaignError(f"self-hash mismatch: {path}")
    return payload


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _resolve(repo: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _connectivity_key(value: object) -> str:
    text = str(value or "").strip()
    return text.split("-")[0] if text else ""


def _external_structure_id(inchi_key: str, standardized_smiles: str) -> str:
    identity = inchi_key or standardized_smiles
    return "EXT-" + hashlib.sha256(identity.encode()).hexdigest()[:24].upper()


def _external_scaffold_id(key: str) -> str:
    return "EXTSCF-" + hashlib.sha256(key.encode()).hexdigest().upper()


def _record_id(set_name: str, source_row: int) -> str:
    return f"{set_name}__row_{int(source_row):06d}"


def _load_training_surfaces(repo: Path, v101_root: Path, v9_root: Path) -> dict[str, Any]:
    master_path = repo / "research/data/platform/processed/herg_hierarchy/v1_3_master/structure_master.parquet"
    master = pd.read_parquet(
        master_path,
        columns=[
            "structure_id",
            "standardized_smiles",
            "standard_inchi_key",
            "model_split",
            "scaffold_group_id",
        ],
    )
    v9_ids = set(
        pd.read_parquet(v9_root / "prepared/training_matrix.parquet", columns=["structure_id"])
        .structure_id.astype(str)
    )
    v9 = master.loc[master.structure_id.astype(str).isin(v9_ids)].copy()
    if len(v9) != len(v9_ids):
        raise CampaignError(f"resolved {len(v9)} of {len(v9_ids)} V9 training structures")
    v9_scaffolds = [scaffold_key(value)[0] for value in v9.standardized_smiles]

    qhts_path = v101_root / "evidence/empirical_ic10_ic30_ic50_labels.parquet"
    qhts = pd.read_parquet(
        qhts_path,
        columns=["standard_inchi_key", "standardized_smiles", "scaffold_group_id"],
    ).drop_duplicates("standard_inchi_key")
    qhts_scaffolds = [scaffold_key(value)[0] for value in qhts.standardized_smiles]
    return {
        "master_path": master_path,
        "master": master,
        "v9": v9,
        "v9_scaffolds": v9_scaffolds,
        "qhts_path": qhts_path,
        "qhts": qhts,
        "qhts_scaffolds": qhts_scaffolds,
    }


def _nearest_v9_similarity(registry: pd.DataFrame, v9: pd.DataFrame) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    references = []
    reference_ids = []
    for row in v9.itertuples(index=False):
        molecule = Chem.MolFromSmiles(str(row.standardized_smiles))
        if molecule is not None:
            references.append(generator.GetFingerprint(molecule))
            reference_ids.append(str(row.structure_id))
    if not references:
        raise CampaignError("V9 reference fingerprints are empty")
    similarities = []
    neighbors = []
    for value in registry.standardized_smiles:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            similarities.append(math.nan)
            neighbors.append("")
            continue
        scores = DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(molecule), references)
        index = int(np.argmax(scores))
        similarities.append(float(scores[index]))
        neighbors.append(reference_ids[index])
    return pd.DataFrame(
        {
            "nearest_v9_morgan_tanimoto": similarities,
            "nearest_v9_structure_id": neighbors,
        }
    )


def _label_columns(set_name: str) -> tuple[str, ...]:
    if set_name in QUANTITATIVE_SETS:
        return (
            "true_pic50_m",
            "source_model_prediction_pic50_m",
            "previous_model_prediction_pic50_m",
        )
    return (
        "true_blocker_at_pic50_5",
        "source_model_prediction_pic50_m",
        "source_model_prediction_class",
    )


def _prepare(repo: Path, source: Path, supplement: Path, output: Path, v101_root: Path, v9_root: Path) -> dict[str, Any]:
    if not source.is_file() or not supplement.is_file():
        raise CampaignError("publisher supplement or extracted external-set source is missing")
    supplement_sha = _sha(supplement)
    if supplement_sha != EXPECTED_SUPPLEMENT_SHA256:
        raise CampaignError(f"unexpected ACS supplement checksum: {supplement_sha}")
    payload = _read_json(source)
    external_sets = payload.get("external_sets", {})
    if set(external_sets) != {*QUANTITATIVE_SETS, BINARY_SET}:
        raise CampaignError("external-set extraction does not have the expected three sets")

    surfaces = _load_training_surfaces(repo, v101_root, v9_root)
    master = surfaces["master"]
    v9 = surfaces["v9"]
    qhts = surfaces["qhts"]
    master_inchi = set(master.standard_inchi_key.fillna("").astype(str))
    master_connectivity = {_connectivity_key(value) for value in master_inchi if value}
    v9_inchi = set(v9.standard_inchi_key.fillna("").astype(str))
    v9_connectivity = {_connectivity_key(value) for value in v9_inchi if value}
    v9_scaffold = set(surfaces["v9_scaffolds"])
    qhts_inchi = set(qhts.standard_inchi_key.fillna("").astype(str))
    qhts_connectivity = {_connectivity_key(value) for value in qhts_inchi if value}
    qhts_scaffold = set(surfaces["qhts_scaffolds"])

    structure_rows: dict[str, dict[str, Any]] = {}
    label_rows: list[dict[str, Any]] = []
    invalid_rows = []
    for set_name, records in external_sets.items():
        for item in records:
            standardized = standardize_smiles(
                item.get("smiles", ""),
                strip_salts=True,
                canonicalize_tautomer=False,
                require_rdkit=True,
            )
            record_id = _record_id(set_name, int(item["source_row"]))
            if not standardized.structure_valid:
                invalid_rows.append(
                    {
                        "record_id": record_id,
                        "set_name": set_name,
                        "source_row": int(item["source_row"]),
                        "structure_status": standardized.structure_standardization_status,
                        "structure_error": standardized.structure_error,
                    }
                )
                continue
            key = standardized.standard_inchi_key
            external_id = _external_structure_id(key, standardized.standardized_smiles)
            scaffold, method = scaffold_key(standardized.standardized_smiles)
            if external_id not in structure_rows:
                structure_rows[external_id] = {
                    "external_structure_id": external_id,
                    "structure_id": external_id,
                    "ligand_id": "external__" + external_id,
                    "original_smiles": standardized.original_smiles,
                    "standardized_smiles": standardized.standardized_smiles,
                    "standard_inchi_key": key,
                    "connectivity_key": _connectivity_key(key),
                    "raw_scaffold_key": scaffold,
                    "scaffold_group_id": _external_scaffold_id(scaffold),
                    "scaffold_method": method,
                    "structure_standardization_version": standardized.structure_standardization_version,
                    "rdkit_version": standardized.rdkit_version,
                    "source_memberships": set(),
                    "source_record_count": 0,
                }
            structure_rows[external_id]["source_memberships"].add(set_name)
            structure_rows[external_id]["source_record_count"] += 1
            label_row: dict[str, Any] = {
                "record_id": record_id,
                "set_name": set_name,
                "source_row": int(item["source_row"]),
                "external_structure_id": external_id,
            }
            for column in _label_columns(set_name):
                label_row[column] = item.get(column)
            label_rows.append(label_row)

    registry = pd.DataFrame(structure_rows.values()).sort_values("external_structure_id").reset_index(drop=True)
    registry["source_memberships"] = registry.source_memberships.map(lambda values: ";".join(sorted(values)))
    registry["master_full_inchi_overlap"] = registry.standard_inchi_key.isin(master_inchi)
    registry["master_connectivity_overlap"] = registry.connectivity_key.isin(master_connectivity)
    registry["v9_full_inchi_overlap"] = registry.standard_inchi_key.isin(v9_inchi)
    registry["v9_connectivity_overlap"] = registry.connectivity_key.isin(v9_connectivity)
    registry["v9_scaffold_overlap"] = registry.raw_scaffold_key.isin(v9_scaffold)
    registry["qhts_full_inchi_overlap"] = registry.standard_inchi_key.isin(qhts_inchi)
    registry["qhts_connectivity_overlap"] = registry.connectivity_key.isin(qhts_connectivity)
    registry["qhts_scaffold_overlap"] = registry.raw_scaffold_key.isin(qhts_scaffold)
    registry = pd.concat(
        [registry, _nearest_v9_similarity(registry, v9)], axis=1
    )
    registry = v13._add_dockability(registry)  # noqa: SLF001
    registry["v9_exact_novel"] = ~registry.v9_connectivity_overlap
    registry["v9_scaffold_novel"] = ~registry.v9_scaffold_overlap
    registry["project_exact_novel"] = ~registry.master_connectivity_overlap
    registry["receptor_evaluation_eligible"] = (
        registry.v9_scaffold_novel
        & registry.project_exact_novel
        & registry.docking_eligible
    )
    registry["outer_fold"] = -1

    labels = pd.DataFrame(label_rows).sort_values(["set_name", "source_row"]).reset_index(drop=True)
    labels_path = output / "sealed/labels.parquet"
    registry_path = output / "prepared/structure_registry.parquet"
    invalid_path = output / "prepared/invalid_structures.parquet"
    _parquet(labels_path, labels)
    _parquet(registry_path, registry)
    _parquet(invalid_path, pd.DataFrame(invalid_rows))
    label_seal = _json(
        output / "sealed/label_seal.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "labels_path": str(labels_path.resolve()),
            "labels_sha256": _sha(labels_path),
            "rows": len(labels),
            "source_sha256": _sha(source),
            "supplement_sha256": supplement_sha,
            "opened_by_prepare_only_to_separate_outcomes_from_structures": True,
        },
        "seal_sha256",
    )

    membership = labels[["set_name", "external_structure_id"]].drop_duplicates()
    audit_rows = []
    for set_name in (*QUANTITATIVE_SETS, BINARY_SET, "combined_unique"):
        selected = registry if set_name == "combined_unique" else registry.loc[
            registry.external_structure_id.isin(
                membership.loc[membership.set_name.eq(set_name), "external_structure_id"]
            )
        ]
        audit_rows.append(
            {
                "set_name": set_name,
                "unique_structures": len(selected),
                "unique_scaffolds": int(selected.raw_scaffold_key.nunique()),
                "v9_connectivity_overlap": int(selected.v9_connectivity_overlap.sum()),
                "v9_scaffold_overlap": int(selected.v9_scaffold_overlap.sum()),
                "qhts_connectivity_overlap": int(selected.qhts_connectivity_overlap.sum()),
                "qhts_scaffold_overlap": int(selected.qhts_scaffold_overlap.sum()),
                "master_connectivity_overlap": int(selected.master_connectivity_overlap.sum()),
                "project_exact_and_v9_scaffold_novel": int(
                    (selected.project_exact_novel & selected.v9_scaffold_novel).sum()
                ),
                "receptor_evaluation_eligible": int(selected.receptor_evaluation_eligible.sum()),
                "median_nearest_v9_morgan_tanimoto": float(
                    selected.nearest_v9_morgan_tanimoto.median()
                ),
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit_path = output / "prepared/overlap_audit.parquet"
    _parquet(audit_path, audit)
    report = _json(
        output / "prepared/preparation_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "source": payload.get("source", {}),
            "source_sha256": _sha(source),
            "supplement_sha256": supplement_sha,
            "valid_source_records": len(labels),
            "invalid_source_records": len(invalid_rows),
            "unique_standardized_structures": len(registry),
            "duplicate_source_records_after_standardization": len(labels) - len(registry),
            "overlap_audit": audit.to_dict("records"),
            "v9_training_structures": len(v9),
            "qhts_training_structures": len(qhts),
            "master_structure_lake": len(master),
            "label_seal_sha256": label_seal["seal_sha256"],
            "structure_registry_contains_outcomes": False,
            "receptor_selection_rule": (
                "V9 scaffold novel AND project structure-lake connectivity novel AND V13 docking eligible"
            ),
        },
        "report_sha256",
    )
    return report


def _feature_matrix(smiles: Iterable[str], columns: list[str]) -> np.ndarray:
    rows = [v101._feature_row(str(value), columns) for value in smiles]  # noqa: SLF001
    frame = pd.concat(rows, ignore_index=True)
    return v101._safe_numeric(frame, columns)  # noqa: SLF001


def _predict_baseline(output: Path, v101_root: Path) -> dict[str, Any]:
    registry_path = output / "prepared/structure_registry.parquet"
    registry = pd.read_parquet(registry_path)
    forbidden = {"true_pic50_m", "true_blocker_at_pic50_5", "observed_target", "target_pic50"}
    if forbidden & set(registry):
        raise CampaignError("outcome column crossed into the label-free structure registry")
    router = joblib.load(v101_root / "models/exact_ternary_router.joblib")
    mixed = joblib.load(v101_root / "models/ic50_regressor.joblib")
    empirical = joblib.load(v101_root / "models/empirical_ic50_regressor.joblib")

    feature_cache: dict[tuple[str, ...], np.ndarray] = {}
    for bundle in (router, mixed, empirical):
        key = tuple(bundle["feature_columns"])
        if key not in feature_cache:
            feature_cache[key] = _feature_matrix(registry.standardized_smiles, list(key))
    router_raw = router["model"].predict_proba(feature_cache[tuple(router["feature_columns"])])
    router_probability = router["calibrator"].predict_proba(
        np.log(np.clip(router_raw, 1e-7, 1.0))
    )
    mixed_pic50 = mixed["model"].predict(feature_cache[tuple(mixed["feature_columns"])])
    empirical_pic50 = empirical["model"].predict(feature_cache[tuple(empirical["feature_columns"])])
    predictions = registry[
        [
            "external_structure_id",
            "structure_id",
            "ligand_id",
            "scaffold_group_id",
            "v9_connectivity_overlap",
            "v9_scaffold_overlap",
            "master_connectivity_overlap",
            "qhts_connectivity_overlap",
            "qhts_scaffold_overlap",
            "v9_exact_novel",
            "v9_scaffold_novel",
            "project_exact_novel",
            "receptor_evaluation_eligible",
            "nearest_v9_morgan_tanimoto",
        ]
    ].copy()
    predictions["baseline_prediction"] = np.asarray(mixed_pic50, dtype=float)
    predictions["v9_mixed_ic50_predicted_pic50"] = np.asarray(mixed_pic50, dtype=float)
    predictions["v10_1_empirical_qhts_ic50_predicted_pic50"] = np.asarray(
        empirical_pic50, dtype=float
    )
    for index, class_name in enumerate(v13.CLASS_NAMES):
        predictions[f"lgbm_rdkit2d_morgan__probability_{class_name.lower()}"] = (
            router_probability[:, index]
        )
    predictions["lgbm_rdkit2d_morgan__prediction"] = np.argmax(router_probability, axis=1)
    prediction_path = output / "predictions/baseline_predictions_before_score.parquet"
    _parquet(prediction_path, predictions)
    seal = _json(
        output / "predictions/baseline_prediction_seal.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": _sha(prediction_path),
            "rows": len(predictions),
            "labels_present_in_prediction_artifact": False,
            "models": {
                "router_sha256": _sha(v101_root / "models/exact_ternary_router.joblib"),
                "v9_mixed_ic50_sha256": _sha(v101_root / "models/ic50_regressor.joblib"),
                "empirical_qhts_ic50_sha256": _sha(
                    v101_root / "models/empirical_ic50_regressor.joblib"
                ),
            },
        },
        "seal_sha256",
    )
    return seal


def _dock(repo: Path, output: Path, primary: Path, vina: Path | None, cpu: int, exhaustiveness: int, modes: int) -> dict[str, Any]:
    registry = pd.read_parquet(output / "prepared/structure_registry.parquet")
    selected = registry.loc[registry.receptor_evaluation_eligible.astype(bool)].copy()
    selected = selected.sort_values("external_structure_id").reset_index(drop=True)
    if selected.empty:
        raise CampaignError("no label-independent external structures are receptor-evaluation eligible")
    selection_columns = [
        "ligand_id",
        "structure_id",
        "external_structure_id",
        "standardized_smiles",
        "scaffold_group_id",
        "outer_fold",
        "docking_parent_smiles",
        "docking_removed_fragment_count",
        "docking_molecular_weight",
        "docking_heavy_atom_count",
        "docking_rotatable_bond_count",
        "nearest_v9_morgan_tanimoto",
    ]
    selected = selected[selection_columns]
    selection_path = output / "selection/receptor_external_panel.parquet"
    _parquet(selection_path, selected)
    selection_report = _json(
        output / "selection/selection_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "selected_structures": len(selected),
            "unique_scaffolds": int(selected.scaffold_group_id.nunique()),
            "selection_uses_labels": False,
            "selection_rule": (
                "V9 scaffold novel AND project structure-lake connectivity novel AND V13 docking eligible"
            ),
            "state": STATE,
            "feature_and_policy_frozen_before_external_source_discovery": True,
        },
        "report_sha256",
    )
    tools = v13._resolve_toolchain(repo, vina)  # noqa: SLF001
    v136._prepare_panel(selected, output, tools)  # noqa: SLF001
    docking = v134._dock(  # noqa: SLF001
        primary,
        output,
        tools,
        selected,
        exhaustiveness,
        modes,
        cpu,
    )
    feature = v13._aggregate_docking(docking, (STATE,))  # noqa: SLF001
    feature_path = output / "docking/external_receptor_features.parquet"
    _parquet(feature_path, feature)
    return _json(
        output / "docking/docking_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete",
            "state": STATE,
            "structures": len(selected),
            "docking_rows": len(docking),
            "feature_rows": len(feature),
            "exhaustiveness": exhaustiveness,
            "modes": modes,
            "cpu_per_vina_task": cpu,
            "selection_report_sha256": selection_report["report_sha256"],
            "docking_sha256": _sha(output / "docking/docking_results.parquet"),
            "features_sha256": _sha(feature_path),
            "vina_scores_are_binding_free_energies": False,
        },
        "report_sha256",
    )


def _external_receptor_probabilities(
    repo: Path,
    discovery: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    discovery = discovery.loc[discovery.cohort.eq("exact_ic50")].reset_index(drop=True)
    inner = v132._v9_inner_baselines(repo, discovery)  # noqa: SLF001
    fold_probabilities = []
    for outer in range(5):
        fit = discovery.outer_fold.ne(outer).to_numpy()
        baseline_map = inner.loc[inner.context_outer_fold.eq(outer)].set_index(
            "structure_id"
        ).inner_baseline
        training = discovery.loc[fit, [PRIMARY_FEATURE]].copy()
        training.insert(
            0,
            "strict_inner_baseline",
            discovery.loc[fit, "structure_id"].map(baseline_map).to_numpy(float),
        )
        validation = external[[PRIMARY_FEATURE]].copy()
        validation.insert(
            0,
            "strict_inner_baseline",
            external.baseline_prediction.to_numpy(float),
        )
        model = v133._classifier()  # noqa: SLF001
        model.fit(
            training[["strict_inner_baseline", PRIMARY_FEATURE]],
            v13._tier_index(discovery.loc[fit, "observed_target"].to_numpy(float)),  # noqa: SLF001
        )
        raw = model.predict_proba(validation[["strict_inner_baseline", PRIMARY_FEATURE]])
        aligned = np.zeros((len(external), 3), dtype=float)
        classes = model.named_steps["model"].classes_.astype(int)
        aligned[:, classes] = raw
        fold_probabilities.append(aligned)
    stack = np.stack(fold_probabilities, axis=0)
    return stack.mean(axis=0), stack.std(axis=0, ddof=0)


def _predict_receptor(repo: Path, output: Path, discovery_root: Path) -> dict[str, Any]:
    baseline_path = output / "predictions/baseline_predictions_before_score.parquet"
    baseline = pd.read_parquet(baseline_path)
    features = pd.read_parquet(output / "docking/external_receptor_features.parquet")
    external = baseline.loc[baseline.receptor_evaluation_eligible.astype(bool)].merge(
        features, on="ligand_id", validate="one_to_one"
    )
    discovery_path = discovery_root / "six_state_analysis_matrix.parquet"
    discovery = pd.read_parquet(discovery_path)
    probability, probability_sd = _external_receptor_probabilities(repo, discovery, external)
    predictions = external[
        [
            "external_structure_id",
            "structure_id",
            "ligand_id",
            "scaffold_group_id",
            "baseline_prediction",
            "nearest_v9_morgan_tanimoto",
            PRIMARY_FEATURE,
            "dock__8ZYO__affinity",
            "dock__8ZYO__ligand_efficiency",
        ]
    ].copy()
    for index, class_name in enumerate(v13.CLASS_NAMES):
        lower = class_name.lower()
        predictions[f"primary_frozen_f649__probability_{lower}"] = probability[:, index]
        predictions[f"primary_frozen_f649__fold_sd_{lower}"] = probability_sd[:, index]
    for column in v136.LGBM_COLUMNS:
        predictions[column] = external[column].to_numpy(float)
    predictions["hybrid_prediction"] = v136._apply_hybrid_policy(predictions)  # noqa: SLF001
    prediction_path = output / "predictions/receptor_predictions_before_score.parquet"
    _parquet(prediction_path, predictions)
    return _json(
        output / "predictions/receptor_prediction_seal.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": _sha(prediction_path),
            "rows": len(predictions),
            "labels_present_in_prediction_artifact": False,
            "external_fold_aggregation": "arithmetic mean of five frozen discovery-fold probabilities",
            "external_fold_aggregation_tuned_on_external_labels": False,
            "primary_feature": PRIMARY_FEATURE,
            "state": STATE,
            "policy": v136._policy_contract(),  # noqa: SLF001
            "discovery_matrix_sha256": _sha(discovery_path),
            "baseline_prediction_sha256": _sha(baseline_path),
        },
        "seal_sha256",
    )


def _tier(values: np.ndarray) -> np.ndarray:
    return v13._tier_index(np.asarray(values, dtype=float))  # noqa: SLF001


def _regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    keep = np.isfinite(y) & np.isfinite(prediction)
    y = y[keep]
    prediction = prediction[keep]
    if not len(y):
        return {"n": 0}
    spearman = spearmanr(y, prediction).statistic if len(y) >= 3 else math.nan
    pearson = pearsonr(y, prediction).statistic if len(y) >= 3 else math.nan
    return {
        "n": len(y),
        "mae": float(mean_absolute_error(y, prediction)),
        "median_absolute_error": float(np.median(np.abs(y - prediction))),
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)) if len(y) >= 2 else math.nan,
        "spearman": float(spearman),
        "pearson": float(pearson),
        "mean_error_prediction_minus_observed": float(np.mean(prediction - y)),
    }


def _classification_metrics(y: np.ndarray, prediction: np.ndarray, labels: list[int]) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    observed = sorted(set(y.tolist()))
    recalls = {
        str(label): float(np.mean(prediction[y == label] == label))
        for label in labels
        if np.any(y == label)
    }
    payload = {
        "n": len(y),
        "observed_classes": observed,
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(y, prediction)),
        "recall": recalls,
        "confusion_matrix": confusion_matrix(y, prediction, labels=labels).tolist(),
    }
    if labels == [0, 1, 2] and np.any(y == 2):
        potent_n = int(np.sum(y == 2))
        potent_safe_n = int(np.sum(prediction[y == 2] == 0))
        rate = potent_safe_n / potent_n
        z = 1.959963984540054
        denominator = 1.0 + z**2 / potent_n
        center = (rate + z**2 / (2 * potent_n)) / denominator
        half_width = (
            z
            * math.sqrt(rate * (1 - rate) / potent_n + z**2 / (4 * potent_n**2))
            / denominator
        )
        payload["potent_predicted_safe_rate"] = float(rate)
        payload["potent_predicted_safe_count"] = potent_safe_n
        payload["observed_potent_count"] = potent_n
        payload["potent_predicted_safe_wilson_ci95"] = [
            float(max(0.0, center - half_width)),
            float(min(1.0, center + half_width)),
        ]
    return payload


def _probability_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(y), 3):
        raise CampaignError("ternary probability matrix has an unexpected shape")
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise CampaignError("ternary probabilities are not finite normalized rows")
    one_hot = np.eye(3)[y]
    confidence = probabilities.max(axis=1)
    correct = np.argmax(probabilities, axis=1) == y
    edges = np.linspace(0.0, 1.0, 11)
    bins = []
    ece = 0.0
    for index in range(10):
        keep = (confidence >= edges[index]) & (
            confidence < edges[index + 1] if index < 9 else confidence <= edges[index + 1]
        )
        if not keep.any():
            continue
        accuracy = float(correct[keep].mean())
        mean_confidence = float(confidence[keep].mean())
        ece += float(keep.mean()) * abs(accuracy - mean_confidence)
        bins.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "n": int(keep.sum()),
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    return {
        "n": len(y),
        "multiclass_log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "classwise_brier": {
            v13.CLASS_NAMES[index]: float(np.mean((probabilities[:, index] - one_hot[:, index]) ** 2))
            for index in range(3)
        },
        "top_class_ece_10_fixed_bins": float(ece),
        "top_class_calibration_bins": bins,
        "calibration_bins_frozen_before_external_scoring": True,
    }


def _cluster_bootstrap_regression(
    frame: pd.DataFrame,
    candidate: str,
    comparator: str,
    replicates: int,
) -> dict[str, Any]:
    groups = [group.index.to_numpy() for _, group in frame.groupby("scaffold_group_id")]
    if len(groups) < 2:
        return {"replicates": 0, "reason": "fewer than two scaffold clusters"}
    rng = np.random.default_rng(SEED)
    deltas = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selected = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[value] for value in selected])
        y = frame.loc[rows, "true_pic50_m"].to_numpy(float)
        candidate_mae = mean_absolute_error(y, frame.loc[rows, candidate].to_numpy(float))
        comparator_mae = mean_absolute_error(y, frame.loc[rows, comparator].to_numpy(float))
        deltas[index] = candidate_mae - comparator_mae
    observed = mean_absolute_error(frame.true_pic50_m, frame[candidate]) - mean_absolute_error(
        frame.true_pic50_m, frame[comparator]
    )
    return {
        "delta_mae_candidate_minus_comparator": float(observed),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "probability_candidate_lower_mae": float(np.mean(deltas < 0)),
        "replicates": replicates,
        "resampling": "paired scaffold-cluster bootstrap",
    }


def _balanced_bootstrap_classification(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    replicates: int,
) -> dict[str, Any]:
    present = np.unique(y)
    by_class = [np.flatnonzero(y == value) for value in present]
    rng = np.random.default_rng(SEED)
    delta = np.empty(replicates, dtype=float)
    macro_delta = np.empty(replicates, dtype=float)
    for index in range(replicates):
        rows = np.concatenate([rng.choice(group, len(group), replace=True) for group in by_class])
        delta[index] = balanced_accuracy_score(y[rows], candidate[rows]) - balanced_accuracy_score(
            y[rows], baseline[rows]
        )
        macro_delta[index] = f1_score(
            y[rows], candidate[rows], labels=[0, 1, 2], average="macro", zero_division=0
        ) - f1_score(
            y[rows], baseline[rows], labels=[0, 1, 2], average="macro", zero_division=0
        )
    observed = balanced_accuracy_score(y, candidate) - balanced_accuracy_score(y, baseline)
    observed_macro = f1_score(
        y, candidate, labels=[0, 1, 2], average="macro", zero_division=0
    ) - f1_score(y, baseline, labels=[0, 1, 2], average="macro", zero_division=0)
    return {
        "delta_balanced_accuracy_candidate_minus_baseline": float(observed),
        "ci95": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
        "probability_candidate_better": float(np.mean(delta > 0)),
        "delta_macro_f1_candidate_minus_baseline": float(observed_macro),
        "macro_f1_ci95": [
            float(np.quantile(macro_delta, 0.025)),
            float(np.quantile(macro_delta, 0.975)),
        ],
        "probability_candidate_higher_macro_f1": float(np.mean(macro_delta > 0)),
        "replicates": replicates,
        "resampling": "paired within-observed-class bootstrap",
    }


def _scaffold_bootstrap_classification(
    scaffold_group_id: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    replicates: int,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "scaffold_group_id": np.asarray(scaffold_group_id, dtype=object),
            "y": np.asarray(y, dtype=int),
            "baseline": np.asarray(baseline, dtype=int),
            "candidate": np.asarray(candidate, dtype=int),
        }
    )
    groups = [group.index.to_numpy() for _, group in frame.groupby("scaffold_group_id")]
    rng = np.random.default_rng(SEED + 1)
    deltas = []
    macro_deltas = []
    attempts = 0
    maximum_attempts = max(replicates * 5, replicates + 100)
    while len(deltas) < replicates and attempts < maximum_attempts:
        attempts += 1
        rows = np.concatenate(
            [groups[index] for index in rng.integers(0, len(groups), len(groups))]
        )
        if len(np.unique(y[rows])) != len(np.unique(y)):
            continue
        deltas.append(
            balanced_accuracy_score(y[rows], candidate[rows])
            - balanced_accuracy_score(y[rows], baseline[rows])
        )
        macro_deltas.append(
            f1_score(
                y[rows],
                candidate[rows],
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
            - f1_score(
                y[rows],
                baseline[rows],
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        )
    if len(deltas) != replicates:
        raise CampaignError("could not complete scaffold-cluster classification bootstrap")
    values = np.asarray(deltas, dtype=float)
    macro_values = np.asarray(macro_deltas, dtype=float)
    observed = balanced_accuracy_score(y, candidate) - balanced_accuracy_score(y, baseline)
    observed_macro = f1_score(
        y, candidate, labels=[0, 1, 2], average="macro", zero_division=0
    ) - f1_score(y, baseline, labels=[0, 1, 2], average="macro", zero_division=0)
    return {
        "delta_balanced_accuracy_candidate_minus_baseline": float(observed),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "probability_candidate_better": float(np.mean(values > 0)),
        "delta_macro_f1_candidate_minus_baseline": float(observed_macro),
        "macro_f1_ci95": [
            float(np.quantile(macro_values, 0.025)),
            float(np.quantile(macro_values, 0.975)),
        ],
        "probability_candidate_higher_macro_f1": float(np.mean(macro_values > 0)),
        "replicates": replicates,
        "unique_scaffold_clusters": len(groups),
        "resampling": "paired scaffold-cluster bootstrap; replicates missing an observed class rejected",
    }


def _quantitative_rows(labels: pd.DataFrame, registry: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows = labels.loc[labels.set_name.isin(QUANTITATIVE_SETS)].copy()
    numeric = [
        "true_pic50_m",
        "source_model_prediction_pic50_m",
        "previous_model_prediction_pic50_m",
    ]
    for column in numeric:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    aggregation = {"true_pic50_m": "median", "source_row": "count"}
    for column in numeric[1:]:
        aggregation[column] = "mean"
    rows = rows.groupby(["set_name", "external_structure_id"], as_index=False).agg(aggregation)
    rows = rows.rename(columns={"source_row": "source_measurement_count"})
    overlap_columns = [
        "external_structure_id",
        "scaffold_group_id",
        "v9_connectivity_overlap",
        "v9_scaffold_overlap",
        "master_connectivity_overlap",
        "qhts_connectivity_overlap",
        "qhts_scaffold_overlap",
        "nearest_v9_morgan_tanimoto",
        "receptor_evaluation_eligible",
    ]
    return rows.merge(registry[overlap_columns], on="external_structure_id", validate="many_to_one").merge(
        baseline,
        on="external_structure_id",
        validate="many_to_one",
        suffixes=("", "__prediction"),
    )


def _score_quantitative(frame: pd.DataFrame, bootstrap: int) -> dict[str, Any]:
    models = {
        "v9_mixed_ic50": "v9_mixed_ic50_predicted_pic50",
        "v10_1_empirical_qhts_ic50": "v10_1_empirical_qhts_ic50_predicted_pic50",
        "source_2025_model": "source_model_prediction_pic50_m",
        "source_previous_model": "previous_model_prediction_pic50_m",
    }
    strata = {
        "all": np.ones(len(frame), dtype=bool),
        "v9_connectivity_novel": ~frame.v9_connectivity_overlap.to_numpy(bool),
        "v9_scaffold_novel": ~frame.v9_scaffold_overlap.to_numpy(bool),
        "project_connectivity_and_v9_scaffold_novel": (
            ~frame.master_connectivity_overlap.to_numpy(bool)
            & ~frame.v9_scaffold_overlap.to_numpy(bool)
        ),
        "qhts_scaffold_novel": ~frame.qhts_scaffold_overlap.to_numpy(bool),
    }
    result: dict[str, Any] = {}
    for set_name in QUANTITATIVE_SETS:
        source = frame.loc[frame.set_name.eq(set_name)].reset_index(drop=True)
        set_result: dict[str, Any] = {}
        for stratum_name, global_keep in strata.items():
            keep = pd.Series(global_keep, index=frame.index).loc[frame.set_name.eq(set_name)].to_numpy()
            selected = source.loc[keep].reset_index(drop=True)
            if selected.empty:
                set_result[stratum_name] = {"n": 0}
                continue
            metrics = {
                name: _regression_metrics(
                    selected.true_pic50_m.to_numpy(float), selected[column].to_numpy(float)
                )
                for name, column in models.items()
            }
            comparisons = {}
            if selected.source_model_prediction_pic50_m.notna().all() and len(selected) >= 2:
                comparisons["v9_vs_source_2025"] = _cluster_bootstrap_regression(
                    selected,
                    "v9_mixed_ic50_predicted_pic50",
                    "source_model_prediction_pic50_m",
                    bootstrap,
                )
            set_result[stratum_name] = {
                "n": len(selected),
                "unique_scaffolds": int(selected.scaffold_group_id.nunique()),
                "models": metrics,
                "paired_comparisons": comparisons,
            }
        result[set_name] = set_result
    return result


def _score_ternary(frame: pd.DataFrame, receptor: pd.DataFrame | None, bootstrap: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    strata = {
        "all": lambda part: np.ones(len(part), dtype=bool),
        "v9_connectivity_novel": lambda part: ~part.v9_connectivity_overlap.to_numpy(bool),
        "v9_scaffold_novel": lambda part: ~part.v9_scaffold_overlap.to_numpy(bool),
        "project_connectivity_and_v9_scaffold_novel": lambda part: (
            ~part.master_connectivity_overlap.to_numpy(bool)
            & ~part.v9_scaffold_overlap.to_numpy(bool)
        ),
    }
    for set_name in QUANTITATIVE_SETS:
        part = frame.loc[frame.set_name.eq(set_name)].reset_index(drop=True)
        y = _tier(part.true_pic50_m.to_numpy(float))
        router = part["lgbm_rdkit2d_morgan__prediction"].to_numpy(int)
        v9_tier = _tier(part.v9_mixed_ic50_predicted_pic50.to_numpy(float))
        set_result = {}
        for stratum_name, selector in strata.items():
            keep = selector(part)
            if not keep.any():
                set_result[stratum_name] = {"n": 0}
                continue
            set_result[stratum_name] = {
                "router": _classification_metrics(y[keep], router[keep], [0, 1, 2]),
                "router_probability_quality": _probability_metrics(
                    y[keep], part.loc[keep, list(v136.LGBM_COLUMNS)].to_numpy(float)
                ),
                "v9_regression_derived_tier": _classification_metrics(
                    y[keep], v9_tier[keep], [0, 1, 2]
                ),
            }
        result[set_name] = set_result
    if receptor is not None and not receptor.empty:
        selected = frame.merge(
            receptor[
                ["external_structure_id", "hybrid_prediction", *v136.RECEPTOR_COLUMNS]
            ],
            on="external_structure_id",
            validate="many_to_one",
        )
        external_y = _tier(selected.true_pic50_m.to_numpy(float))
        external_router = selected["lgbm_rdkit2d_morgan__prediction"].to_numpy(int)
        hybrid = selected.hybrid_prediction.to_numpy(int)
        router_metrics = _classification_metrics(external_y, external_router, [0, 1, 2])
        hybrid_metrics = _classification_metrics(external_y, hybrid, [0, 1, 2])
        paired = _balanced_bootstrap_classification(
            external_y, external_router, hybrid, bootstrap
        )
        scaffold_paired = _scaffold_bootstrap_classification(
            selected.scaffold_group_id.to_numpy(object),
            external_y,
            external_router,
            hybrid,
            bootstrap,
        )
        safety_passed = bool(
            hybrid_metrics["potent_predicted_safe_rate"]
            <= v136.MAX_ABSOLUTE_POTENT_TO_SAFE_RATE
            and hybrid_metrics["potent_predicted_safe_rate"]
            <= router_metrics["potent_predicted_safe_rate"]
            + v136.MAX_POTENT_TO_SAFE_INCREASE
        )
        moderate_retention_passed = v136._moderate_retention_passed(  # noqa: SLF001
            hybrid_metrics["recall"]["1"], router_metrics["recall"]["1"]
        )
        incremental_value_confirmed = bool(
            paired["ci95"][0] > 0
            and scaffold_paired["ci95"][0] > 0
            and hybrid_metrics["macro_f1"] > router_metrics["macro_f1"]
        )
        result["receptor_project_exact_and_v9_scaffold_novel"] = {
            "router": router_metrics,
            "frozen_v13_6_hybrid": hybrid_metrics,
            "paired_bootstrap": paired,
            "paired_scaffold_cluster_bootstrap": scaffold_paired,
            "receptor_probability_quality": _probability_metrics(
                external_y, selected[list(v136.RECEPTOR_COLUMNS)].to_numpy(float)
            ),
            "external_decision": {
                "incremental_value_confirmed": incremental_value_confirmed,
                "safety_gate_passed": safety_passed,
                "moderate_retention_passed": moderate_retention_passed,
                "full_policy_external_gate_passed": bool(
                    incremental_value_confirmed
                    and safety_passed
                    and moderate_retention_passed
                ),
                "potent_predicted_safe_count": int(
                    np.sum(hybrid[external_y == 2] == 0)
                ),
                "observed_potent_count": int(np.sum(external_y == 2)),
                "interpretation": (
                    "incremental class-balanced value and safety are separate gates; "
                    "a favorable BA result cannot waive the frozen Potent-to-Safe limit"
                ),
            },
        }
    return result


def _score_binary(
    labels: pd.DataFrame,
    registry: pd.DataFrame,
    baseline: pd.DataFrame,
    quantitative: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for set_name in QUANTITATIVE_SETS:
        frame = quantitative.loc[quantitative.set_name.eq(set_name)].copy()
        y = (frame.true_pic50_m.to_numpy(float) >= 5.0).astype(int)
        set_result: dict[str, Any] = {
            "threshold": "pIC50 >= 5.0",
            "v9_mixed_ic50": _classification_metrics(
                y, (frame.v9_mixed_ic50_predicted_pic50.to_numpy(float) >= 5.0).astype(int), [0, 1]
            ),
            "v10_1_empirical_qhts_ic50": _classification_metrics(
                y,
                (frame.v10_1_empirical_qhts_ic50_predicted_pic50.to_numpy(float) >= 5.0).astype(int),
                [0, 1],
            ),
        }
        if frame.source_model_prediction_pic50_m.notna().all():
            set_result["source_2025_model"] = _classification_metrics(
                y, (frame.source_model_prediction_pic50_m.to_numpy(float) >= 5.0).astype(int), [0, 1]
            )
        result[set_name] = set_result
    ev3 = labels.loc[labels.set_name.eq(BINARY_SET)].copy()
    ev3["true_blocker_at_pic50_5"] = pd.to_numeric(
        ev3.true_blocker_at_pic50_5, errors="raise"
    ).astype(int)
    ev3 = ev3.groupby("external_structure_id", as_index=False).agg(
        true_blocker_at_pic50_5=("true_blocker_at_pic50_5", "median"),
        source_model_prediction_pic50_m=("source_model_prediction_pic50_m", "mean"),
    )
    ev3["true_blocker_at_pic50_5"] = (ev3.true_blocker_at_pic50_5 >= 0.5).astype(int)
    overlap = registry[
        ["external_structure_id", "v9_connectivity_overlap", "v9_scaffold_overlap"]
    ]
    ev3 = ev3.merge(overlap, on="external_structure_id", validate="one_to_one").merge(
        baseline, on="external_structure_id", validate="one_to_one", suffixes=("", "__prediction")
    )
    ev3_y = ev3.true_blocker_at_pic50_5.to_numpy(int)
    ev3_models = {
        "v9_mixed_ic50": (ev3.v9_mixed_ic50_predicted_pic50.to_numpy(float) >= 5.0).astype(int),
        "v10_1_empirical_qhts_ic50": (
            ev3.v10_1_empirical_qhts_ic50_predicted_pic50.to_numpy(float) >= 5.0
        ).astype(int),
        "source_2025_model": (ev3.source_model_prediction_pic50_m.to_numpy(float) >= 5.0).astype(int),
    }
    result[BINARY_SET] = {
        "threshold": "source binary label; continuous predictions thresholded at pIC50 >= 5.0",
        "all": {
            name: _classification_metrics(ev3_y, prediction, [0, 1])
            for name, prediction in ev3_models.items()
        },
        "overlap": {
            "v9_connectivity_overlap": int(ev3.v9_connectivity_overlap.sum()),
            "v9_scaffold_overlap": int(ev3.v9_scaffold_overlap.sum()),
        },
    }
    return result


def _receptor_findings(quantitative: pd.DataFrame, receptor: pd.DataFrame | None) -> dict[str, Any]:
    if receptor is None or receptor.empty:
        return {"available": False}
    frame = quantitative.merge(receptor, on="external_structure_id", validate="many_to_one")
    if frame.empty:
        return {"available": False, "reason": "no quantitative receptor rows"}
    y = frame.true_pic50_m.to_numpy(float)
    findings = {}
    for column in (
        PRIMARY_FEATURE,
        "dock__8ZYO__affinity",
        "dock__8ZYO__ligand_efficiency",
    ):
        values = frame[column].to_numpy(float)
        rho = spearmanr(y, values)
        findings[column] = {
            "n": len(frame),
            "spearman": float(rho.statistic),
            "two_sided_p_value_descriptive_not_multiplicity_adjusted": float(rho.pvalue),
        }
    frame = frame.assign(observed_tier=_tier(y))
    summaries = []
    for tier_index, group in frame.groupby("observed_tier"):
        summaries.append(
            {
                "tier_index": int(tier_index),
                "tier": v13.CLASS_NAMES[int(tier_index)],
                "n": len(group),
                "median_f649_contact_count": float(group[PRIMARY_FEATURE].median()),
                "median_affinity_kcal_mol": float(group["dock__8ZYO__affinity"].median()),
            }
        )
    return {
        "available": True,
        "univariate_descriptive_associations": findings,
        "observed_tier_summaries": summaries,
        "claim_boundary": (
            "post-score descriptive associations only; no external-label feature or threshold tuning"
        ),
    }


def _applicability_findings(quantitative: pd.DataFrame) -> dict[str, Any]:
    frame = quantitative.loc[
        quantitative.set_name.eq("ev2_exact_quantitative")
        & ~quantitative.master_connectivity_overlap.astype(bool)
        & ~quantitative.v9_scaffold_overlap.astype(bool)
    ].copy()
    if frame.empty:
        return {"available": False}
    frame["v9_absolute_error"] = (
        frame.v9_mixed_ic50_predicted_pic50 - frame.true_pic50_m
    ).abs()
    frame["source_2025_absolute_error"] = (
        frame.source_model_prediction_pic50_m - frame.true_pic50_m
    ).abs()
    edges = [-math.inf, 0.3, 0.5, 0.7, math.inf]
    names = ["lt_0p3", "0p3_to_lt_0p5", "0p5_to_lt_0p7", "ge_0p7"]
    frame["fixed_similarity_bin"] = pd.cut(
        frame.nearest_v9_morgan_tanimoto,
        bins=edges,
        labels=names,
        right=False,
    )
    bins = []
    for name in names:
        group = frame.loc[frame.fixed_similarity_bin.eq(name)]
        bins.append(
            {
                "bin": name,
                "n": len(group),
                "unique_scaffolds": int(group.scaffold_group_id.nunique()),
                "v9_mae": float(group.v9_absolute_error.mean()) if len(group) else None,
                "source_2025_mae": (
                    float(group.source_2025_absolute_error.mean()) if len(group) else None
                ),
            }
        )
    v9_association = spearmanr(
        frame.nearest_v9_morgan_tanimoto, frame.v9_absolute_error
    )
    source_association = spearmanr(
        frame.nearest_v9_morgan_tanimoto, frame.source_2025_absolute_error
    )
    return {
        "available": True,
        "population": "EV-2 project-connectivity-novel and V9-scaffold-novel",
        "n": len(frame),
        "nearest_v9_morgan_tanimoto": {
            "minimum": float(frame.nearest_v9_morgan_tanimoto.min()),
            "median": float(frame.nearest_v9_morgan_tanimoto.median()),
            "maximum": float(frame.nearest_v9_morgan_tanimoto.max()),
        },
        "fixed_similarity_bins": bins,
        "similarity_vs_absolute_error": {
            "v9_spearman": float(v9_association.statistic),
            "v9_two_sided_p_value_descriptive": float(v9_association.pvalue),
            "source_2025_spearman": float(source_association.statistic),
            "source_2025_two_sided_p_value_descriptive": float(source_association.pvalue),
        },
        "claim_boundary": (
            "fixed, label-independent similarity bins; descriptive external applicability analysis, "
            "not a fitted acceptance threshold"
        ),
    }


def _render_report(report: dict[str, Any]) -> str:
    audit = report["overlap_audit"]
    lines = [
        "# hERG V13.7 Publisher-Declared External Validation",
        "",
        f"Status: **{report['status']}**  ",
        f"Created: {report['created_utc']}",
        "",
        "## Integrity boundary",
        "",
        "Structures were standardized and overlap-audited before outcomes were joined. Frozen V9/V10.1 predictions and, when available, V13.6 receptor decisions were hash-sealed before scoring. No model, feature, threshold, or policy was tuned on these external outcomes.",
        "",
        "## Overlap audit",
        "",
        "| Set | Unique structures | V9 connectivity overlap | V9 scaffold overlap | Project-exact + V9-scaffold novel | Dock eligible |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in audit:
        lines.append(
            f"| {row['set_name']} | {row['unique_structures']} | {row['v9_connectivity_overlap']} | "
            f"{row['v9_scaffold_overlap']} | {row['project_exact_and_v9_scaffold_novel']} | "
            f"{row['receptor_evaluation_eligible']} |"
        )
    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "The machine-readable analysis report contains all overlap-stratified regression, ternary, and binary metrics with paired uncertainty. Vina observables are not experimental affinities or binding free energies. These are retrospective external literature sets, not prospective laboratory validation.",
            "",
        ]
    )
    return "\n".join(lines)


def _score(output: Path, bootstrap: int) -> dict[str, Any]:
    label_seal = _read_json(output / "sealed/label_seal.json", "seal_sha256")
    baseline_seal = _read_json(
        output / "predictions/baseline_prediction_seal.json", "seal_sha256"
    )
    labels_path = output / "sealed/labels.parquet"
    baseline_path = output / "predictions/baseline_predictions_before_score.parquet"
    if _sha(labels_path) != label_seal["labels_sha256"]:
        raise CampaignError("sealed label artifact changed before scoring")
    if _sha(baseline_path) != baseline_seal["prediction_sha256"]:
        raise CampaignError("sealed baseline prediction artifact changed before scoring")
    labels = pd.read_parquet(labels_path)
    registry = pd.read_parquet(output / "prepared/structure_registry.parquet")
    baseline = pd.read_parquet(baseline_path)
    receptor_path = output / "predictions/receptor_predictions_before_score.parquet"
    receptor = None
    receptor_seal_sha = None
    if receptor_path.is_file():
        receptor_seal = _read_json(
            output / "predictions/receptor_prediction_seal.json", "seal_sha256"
        )
        if _sha(receptor_path) != receptor_seal["prediction_sha256"]:
            raise CampaignError("sealed receptor prediction artifact changed before scoring")
        receptor = pd.read_parquet(receptor_path)
        receptor_seal_sha = receptor_seal["seal_sha256"]
    quantitative = _quantitative_rows(labels, registry, baseline)
    overlap = pd.read_parquet(output / "prepared/overlap_audit.parquet")
    report = _json(
        output / "analysis_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": (
                "complete_with_receptor_external_validation"
                if receptor is not None
                else "complete_ligand_only_external_validation"
            ),
            "source": {
                "article_doi": "10.1021/acs.chemrestox.5c00065",
                "supplement_doi": "10.1021/acs.chemrestox.5c00065.s002",
                "publisher_supplement_sha256": EXPECTED_SUPPLEMENT_SHA256,
            },
            "overlap_audit": overlap.to_dict("records"),
            "quantitative_regression": _score_quantitative(quantitative, bootstrap),
            "ternary_classification": _score_ternary(quantitative, receptor, bootstrap),
            "binary_classification": _score_binary(labels, registry, baseline, quantitative),
            "applicability_findings": _applicability_findings(quantitative),
            "receptor_findings": _receptor_findings(quantitative, receptor),
            "prediction_seals": {
                "baseline": baseline_seal["seal_sha256"],
                "receptor": receptor_seal_sha,
            },
            "scientific_scope": {
                "external_labels_used_for_model_feature_threshold_or_policy_tuning": False,
                "models_and_v13_6_policy_frozen_before_external_source_discovery": True,
                "overlap_stratified_reporting": True,
                "retrospective_external_literature_validation": True,
                "prospective_validation": False,
                "vina_scores_are_binding_free_energies": False,
                "direct_ic10_and_ic30_externally_validated_here": False,
                "reason_ic10_ic30_not_externally_scored": (
                    "the publisher-declared external sets expose IC50 or binary labels, not measured IC10/IC30"
                ),
            },
        },
        "report_sha256",
    )
    report_path = output / "REPORT.md"
    report_path.write_text(_render_report(report))
    return report


def _manifest(repo: Path, source: Path, supplement: Path, output: Path) -> dict[str, Any]:
    inputs = [
        repo / "pipeline/scripts/run_local_herg_external_validation_v13_7.py",
        source,
        supplement,
        repo / "research/local_runs/herg_v10_1_expanded_platform/manifest.json",
        repo / "research/local_runs/herg_domain_mixture_campaign_v9/manifest.json",
        repo / "research/local_runs/herg_receptor_ensemble_campaign_v13/manifest.json",
        repo / "research/local_runs/herg_receptor_strict_analysis_v13_2/manifest.json",
        repo / "research/local_runs/herg_receptor_hybrid_validation_v13_6/manifest.json",
    ]
    artifacts = [
        path
        for path in (
            output / "prepared/structure_registry.parquet",
            output / "prepared/overlap_audit.parquet",
            output / "prepared/preparation_report.json",
            output / "prepared/invalid_structures.parquet",
            output / "sealed/labels.parquet",
            output / "sealed/label_seal.json",
            output / "predictions/baseline_predictions_before_score.parquet",
            output / "predictions/baseline_prediction_seal.json",
            output / "selection/receptor_external_panel.parquet",
            output / "docking/docking_results.parquet",
            output / "docking/external_receptor_features.parquet",
            output / "docking/docking_report.json",
            output / "predictions/receptor_predictions_before_score.parquet",
            output / "predictions/receptor_prediction_seal.json",
            output / "analysis_report.json",
            output / "REPORT.md",
        )
        if path.is_file()
    ]
    return _json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete" if (output / "analysis_report.json").is_file() else "in_progress",
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "baseline", "dock", "receptor", "score", "all"),
        default="all",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--supplement", type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--v10-1-root", type=Path, default=DEFAULT_V101)
    parser.add_argument("--v9-root", type=Path, default=DEFAULT_V9)
    parser.add_argument("--v13-root", type=Path, default=DEFAULT_V13)
    parser.add_argument("--v13-2-root", type=Path, default=DEFAULT_V132)
    parser.add_argument("--vina", type=Path)
    parser.add_argument("--cpu", type=int, default=6)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--modes", type=int, default=9)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser


def _main(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    source = _resolve(repo, args.source)
    supplement = _resolve(repo, args.supplement)
    output = _resolve(repo, args.output_root)
    v101_root = _resolve(repo, args.v10_1_root)
    v9_root = _resolve(repo, args.v9_root)
    primary = _resolve(repo, args.v13_root)
    discovery_root = _resolve(repo, args.v13_2_root)
    vina = args.vina.resolve() if args.vina else None
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "stage": args.stage}
    if args.stage in ("prepare", "all"):
        result["prepare"] = _prepare(repo, source, supplement, output, v101_root, v9_root)
    if args.stage in ("baseline", "all"):
        result["baseline"] = _predict_baseline(output, v101_root)
    if args.stage in ("dock", "all"):
        result["dock"] = _dock(
            repo,
            output,
            primary,
            vina,
            args.cpu,
            args.exhaustiveness,
            args.modes,
        )
    if args.stage in ("receptor", "all"):
        result["receptor"] = _predict_receptor(repo, output, discovery_root)
    if args.stage in ("score", "all"):
        result["score"] = _score(output, args.bootstrap_replicates)
    result["manifest"] = _manifest(repo, source, supplement, output)
    return result


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _main(args)
    except (
        CampaignError,
        v13.CampaignError,
        v132.CampaignError,
        v133.CampaignError,
        v134.CampaignError,
        v136.CampaignError,
    ) as exc:
        print(f"V13.7 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
