#!/usr/bin/env python3
"""Confirm frozen hERG models on the 2026 Sun-Wang-Shen validation set.

The official JCIM supplement contains 1,133 post-2021 ChEMBL hERG IC50
records.  Because the current project snapshot is newer than the source
study, this campaign does not accept the paper's "external" designation at
face value.  It standardizes structures against the actual V9 training set,
separates and seals labels, reports exact- and scaffold-overlap strata, and
uses only V9-scaffold-novel compounds for V13 receptor confirmation.

The model, 8ZYO state, F649 feature, five-fold probability averaging rule, and
V13.6 hybrid policy are frozen before outcomes are scored.  Vina outputs are
scoring-function observables, not experimental affinities or binding free
energies.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
for candidate in (SCRIPT_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_local_herg_external_validation_v13_7 as v137  # noqa: E402
import run_local_herg_receptor_classification_confirmation_v13_3 as v133  # noqa: E402
import run_local_herg_receptor_classification_replication_v13_4 as v134  # noqa: E402
import run_local_herg_receptor_ensemble_campaign_v13 as v13  # noqa: E402
import run_local_herg_receptor_hybrid_validation_v13_6 as v136  # noqa: E402
from menin_discovery.chemistry import standardize_smiles  # noqa: E402
from menin_discovery.features import scaffold_key  # noqa: E402

SCHEMA_VERSION = "platform-local-herg-temporal-confirmation-v13.8/1.0"
SOURCE_DOI = "10.1021/acs.jcim.6c00163"
SUPPLEMENT_DOI = "10.1021/acs.jcim.6c00163.s001"
EXPECTED_WORKBOOK_SHA256 = "acf8e8082339fad0a7668a9fdaafdada0333f00c65c904f5a468b276ba3c7845"
DEFAULT_SOURCE = Path("research/external_validation_sources/jcim_2026_external_validation.json")
DEFAULT_WORKBOOK = Path("research/external_validation_sources/ci6c00163_si_001.xlsx")
DEFAULT_OUTPUT = Path("research/local_runs/herg_temporal_confirmation_v13_8")
DEFAULT_V101 = Path("research/local_runs/herg_v10_1_expanded_platform")
DEFAULT_V9 = Path("research/local_runs/herg_domain_mixture_campaign_v9")
DEFAULT_V13 = Path("research/local_runs/herg_receptor_ensemble_campaign_v13")
DEFAULT_V132 = Path("research/local_runs/herg_receptor_strict_analysis_v13_2")


class CampaignError(RuntimeError):
    """Raised when a V13.8 scientific or artifact invariant fails."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return v137._sha(path)  # noqa: SLF001


def _json(path: Path, payload: dict[str, Any], field: str) -> dict[str, Any]:
    return v137._json(path, payload, field)  # noqa: SLF001


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    v137._parquet(path, frame)  # noqa: SLF001


def _read_json(path: Path, field: str | None = None) -> dict[str, Any]:
    return v137._read_json(path, field)  # noqa: SLF001


def _resolve(repo: Path, path: Path) -> Path:
    return v137._resolve(repo, path)  # noqa: SLF001


def _prepare(
    repo: Path,
    source: Path,
    workbook: Path,
    output: Path,
    v101_root: Path,
    v9_root: Path,
) -> dict[str, Any]:
    if _sha(workbook) != EXPECTED_WORKBOOK_SHA256:
        raise CampaignError("unexpected 2026 JCIM workbook checksum")
    payload = _read_json(source)
    records = payload.get("external_validation", [])
    if len(records) != 1_133:
        raise CampaignError(f"expected 1,133 source rows, found {len(records)}")
    surfaces = v137._load_training_surfaces(repo, v101_root, v9_root)  # noqa: SLF001
    master = surfaces["master"]
    v9 = surfaces["v9"]
    qhts = surfaces["qhts"]
    master_inchi = set(master.standard_inchi_key.fillna("").astype(str))
    master_connectivity = {
        v137._connectivity_key(value) for value in master_inchi if value  # noqa: SLF001
    }
    v9_inchi = set(v9.standard_inchi_key.fillna("").astype(str))
    v9_connectivity = {
        v137._connectivity_key(value) for value in v9_inchi if value  # noqa: SLF001
    }
    v9_scaffolds = set(surfaces["v9_scaffolds"])
    qhts_inchi = set(qhts.standard_inchi_key.fillna("").astype(str))
    qhts_connectivity = {
        v137._connectivity_key(value) for value in qhts_inchi if value  # noqa: SLF001
    }
    qhts_scaffolds = set(surfaces["qhts_scaffolds"])

    structures: dict[str, dict[str, Any]] = {}
    labels = []
    invalid = []
    for item in records:
        result = standardize_smiles(
            item.get("smiles", ""),
            strip_salts=True,
            canonicalize_tautomer=False,
            require_rdkit=True,
        )
        source_row = int(item["source_row"])
        if not result.structure_valid:
            invalid.append(
                {
                    "source_row": source_row,
                    "structure_status": result.structure_standardization_status,
                    "structure_error": result.structure_error,
                }
            )
            continue
        inchi = result.standard_inchi_key
        external_id = v137._external_structure_id(  # noqa: SLF001
            inchi, result.standardized_smiles
        )
        raw_scaffold, method = scaffold_key(result.standardized_smiles)
        structures.setdefault(
            external_id,
            {
                "external_structure_id": external_id,
                "structure_id": external_id,
                "ligand_id": "temporal__" + external_id,
                "original_smiles": result.original_smiles,
                "standardized_smiles": result.standardized_smiles,
                "standard_inchi_key": inchi,
                "connectivity_key": v137._connectivity_key(inchi),  # noqa: SLF001
                "raw_scaffold_key": raw_scaffold,
                "scaffold_group_id": v137._external_scaffold_id(raw_scaffold),  # noqa: SLF001
                "scaffold_method": method,
                "structure_standardization_version": result.structure_standardization_version,
                "rdkit_version": result.rdkit_version,
                "source_memberships": "jcim_2026_post_2021_validation",
                "source_record_count": 0,
            },
        )
        structures[external_id]["source_record_count"] += 1
        labels.append(
            {
                "record_id": f"jcim_2026__row_{source_row:06d}",
                "source_row": source_row,
                "external_structure_id": external_id,
                "true_ic50_nm": float(item["true_ic50_nm"]),
                "true_pic50_m": float(9.0 - math.log10(float(item["true_ic50_nm"]))),
            }
        )
    registry = pd.DataFrame(structures.values()).sort_values(
        "external_structure_id"
    ).reset_index(drop=True)
    registry["master_full_inchi_overlap"] = registry.standard_inchi_key.isin(master_inchi)
    registry["master_connectivity_overlap"] = registry.connectivity_key.isin(master_connectivity)
    registry["v9_full_inchi_overlap"] = registry.standard_inchi_key.isin(v9_inchi)
    registry["v9_connectivity_overlap"] = registry.connectivity_key.isin(v9_connectivity)
    registry["v9_scaffold_overlap"] = registry.raw_scaffold_key.isin(v9_scaffolds)
    registry["qhts_full_inchi_overlap"] = registry.standard_inchi_key.isin(qhts_inchi)
    registry["qhts_connectivity_overlap"] = registry.connectivity_key.isin(qhts_connectivity)
    registry["qhts_scaffold_overlap"] = registry.raw_scaffold_key.isin(qhts_scaffolds)
    registry = pd.concat(
        [registry, v137._nearest_v9_similarity(registry, v9)], axis=1  # noqa: SLF001
    )
    registry = v13._add_dockability(registry)  # noqa: SLF001
    registry["v9_exact_novel"] = ~registry.v9_connectivity_overlap
    registry["v9_scaffold_novel"] = ~registry.v9_scaffold_overlap
    registry["project_exact_novel"] = ~registry.master_connectivity_overlap
    registry["receptor_evaluation_eligible"] = (
        registry.v9_scaffold_novel & registry.docking_eligible
    )
    registry["outer_fold"] = -1

    label_frame = pd.DataFrame(labels).sort_values("source_row").reset_index(drop=True)
    registry_path = output / "prepared/structure_registry.parquet"
    labels_path = output / "sealed/labels.parquet"
    _parquet(registry_path, registry)
    _parquet(labels_path, label_frame)
    _parquet(output / "prepared/invalid_structures.parquet", pd.DataFrame(invalid))
    label_seal = _json(
        output / "sealed/label_seal.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "labels_path": str(labels_path.resolve()),
            "labels_sha256": _sha(labels_path),
            "rows": len(label_frame),
            "source_sha256": _sha(source),
            "workbook_sha256": _sha(workbook),
        },
        "seal_sha256",
    )
    audit = {
        "source_records": len(label_frame),
        "invalid_records": len(invalid),
        "unique_standardized_structures": len(registry),
        "unique_connectivity_keys": int(registry.connectivity_key.nunique()),
        "unique_scaffolds": int(registry.raw_scaffold_key.nunique()),
        "master_connectivity_overlap": int(registry.master_connectivity_overlap.sum()),
        "v9_connectivity_overlap": int(registry.v9_connectivity_overlap.sum()),
        "v9_scaffold_overlap": int(registry.v9_scaffold_overlap.sum()),
        "v9_connectivity_novel": int((~registry.v9_connectivity_overlap).sum()),
        "v9_scaffold_novel": int((~registry.v9_scaffold_overlap).sum()),
        "receptor_evaluation_eligible": int(registry.receptor_evaluation_eligible.sum()),
    }
    return _json(
        output / "prepared/preparation_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "source": payload.get("source", {}),
            "audit": audit,
            "label_seal_sha256": label_seal["seal_sha256"],
            "structure_registry_contains_outcomes": False,
            "external_designation_accepted_without_overlap_audit": False,
            "receptor_selection_rule": "V9 scaffold novel AND V13 docking eligible",
            "entire_source_present_in_current_project_structure_lake": bool(
                registry.master_connectivity_overlap.all()
            ),
        },
        "report_sha256",
    )


def _baseline(output: Path, v101_root: Path) -> dict[str, Any]:
    seal = v137._predict_baseline(output, v101_root)  # noqa: SLF001
    seal["schema_version"] = SCHEMA_VERSION
    return _json(
        output / "predictions/baseline_prediction_seal.json", seal, "seal_sha256"
    )


def _dock(
    repo: Path,
    output: Path,
    primary: Path,
    vina: Path | None,
    cpu: int,
    exhaustiveness: int,
    modes: int,
) -> dict[str, Any]:
    registry = pd.read_parquet(output / "prepared/structure_registry.parquet")
    selected = registry.loc[registry.receptor_evaluation_eligible.astype(bool)].copy()
    selected = selected.sort_values("external_structure_id").reset_index(drop=True)
    columns = [
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
    selected = selected[columns]
    selection_path = output / "selection/receptor_temporal_panel.parquet"
    _parquet(selection_path, selected)
    selection_report = _json(
        output / "selection/selection_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "selected_structures": len(selected),
            "unique_scaffolds": int(selected.scaffold_group_id.nunique()),
            "selection_uses_labels": False,
            "selection_rule": "V9 scaffold novel AND V13 docking eligible",
            "all_selected_structures_present_in_broader_project_lake": True,
        },
        "report_sha256",
    )
    tools = v13._resolve_toolchain(repo, vina)  # noqa: SLF001
    v136._prepare_panel(selected, output, tools)  # noqa: SLF001
    docking = v134._dock(  # noqa: SLF001
        primary, output, tools, selected, exhaustiveness, modes, cpu
    )
    features = v13._aggregate_docking(docking, (v137.STATE,))  # noqa: SLF001
    feature_path = output / "docking/external_receptor_features.parquet"
    _parquet(feature_path, features)
    return _json(
        output / "docking/docking_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete",
            "state": v137.STATE,
            "structures": len(selected),
            "unique_scaffolds": int(selected.scaffold_group_id.nunique()),
            "exhaustiveness": exhaustiveness,
            "modes": modes,
            "selection_report_sha256": selection_report["report_sha256"],
            "docking_sha256": _sha(output / "docking/docking_results.parquet"),
            "features_sha256": _sha(feature_path),
            "vina_scores_are_binding_free_energies": False,
        },
        "report_sha256",
    )


def _receptor(repo: Path, output: Path, discovery_root: Path) -> dict[str, Any]:
    baseline_path = output / "predictions/baseline_predictions_before_score.parquet"
    baseline = pd.read_parquet(baseline_path)
    features = pd.read_parquet(output / "docking/external_receptor_features.parquet")
    external = baseline.loc[baseline.receptor_evaluation_eligible.astype(bool)].merge(
        features, on="ligand_id", validate="one_to_one"
    )
    discovery_path = discovery_root / "six_state_analysis_matrix.parquet"
    discovery = pd.read_parquet(discovery_path)
    probability, probability_sd = v137._external_receptor_probabilities(  # noqa: SLF001
        repo, discovery, external
    )
    predictions = external[
        [
            "external_structure_id",
            "structure_id",
            "ligand_id",
            "scaffold_group_id",
            "baseline_prediction",
            "nearest_v9_morgan_tanimoto",
            v137.PRIMARY_FEATURE,
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
            "external_fold_aggregation": "mean of five frozen discovery-fold probabilities",
            "policy": v136._policy_contract(),  # noqa: SLF001
            "discovery_matrix_sha256": _sha(discovery_path),
        },
        "seal_sha256",
    )


def _cluster_mae_interval(frame: pd.DataFrame, prediction: str, replicates: int) -> dict[str, Any]:
    groups = [group.index.to_numpy() for _, group in frame.groupby("scaffold_group_id")]
    if len(groups) < 2:
        return {"replicates": 0}
    rng = np.random.default_rng(v137.SEED)
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        rows = np.concatenate([groups[value] for value in rng.integers(0, len(groups), len(groups))])
        values[index] = mean_absolute_error(
            frame.loc[rows, "true_pic50_m"], frame.loc[rows, prediction]
        )
    return {
        "observed_mae": float(mean_absolute_error(frame.true_pic50_m, frame[prediction])),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "replicates": replicates,
        "resampling": "scaffold-cluster bootstrap",
    }


def _receptor_score_frame(frame: pd.DataFrame, receptor: pd.DataFrame) -> pd.DataFrame:
    receptor_columns = [
        "external_structure_id",
        "hybrid_prediction",
        *v136.RECEPTOR_COLUMNS,
        v137.PRIMARY_FEATURE,
        "dock__8ZYO__affinity",
        "dock__8ZYO__ligand_efficiency",
    ]
    missing = set(receptor_columns) - set(receptor)
    if missing:
        raise CampaignError(f"sealed receptor predictions lack score columns: {sorted(missing)}")
    return frame.merge(
        receptor[receptor_columns],
        on="external_structure_id",
        validate="one_to_one",
    )


def _score(output: Path, bootstrap: int) -> dict[str, Any]:
    label_seal = _read_json(output / "sealed/label_seal.json", "seal_sha256")
    baseline_seal = _read_json(
        output / "predictions/baseline_prediction_seal.json", "seal_sha256"
    )
    labels_path = output / "sealed/labels.parquet"
    baseline_path = output / "predictions/baseline_predictions_before_score.parquet"
    if _sha(labels_path) != label_seal["labels_sha256"]:
        raise CampaignError("label seal mismatch")
    if _sha(baseline_path) != baseline_seal["prediction_sha256"]:
        raise CampaignError("baseline prediction seal mismatch")
    labels = pd.read_parquet(labels_path)
    registry = pd.read_parquet(output / "prepared/structure_registry.parquet")
    baseline = pd.read_parquet(baseline_path)
    labels = labels.groupby("external_structure_id", as_index=False).agg(
        true_pic50_m=("true_pic50_m", "median"),
        true_ic50_nm=("true_ic50_nm", "median"),
        source_measurement_count=("source_row", "count"),
    )
    columns = [
        "external_structure_id",
        "scaffold_group_id",
        "v9_connectivity_overlap",
        "v9_scaffold_overlap",
        "master_connectivity_overlap",
        "nearest_v9_morgan_tanimoto",
        "receptor_evaluation_eligible",
    ]
    frame = labels.merge(registry[columns], on="external_structure_id", validate="one_to_one").merge(
        baseline, on="external_structure_id", validate="one_to_one", suffixes=("", "__prediction")
    )
    strata = {
        "all": np.ones(len(frame), dtype=bool),
        "v9_connectivity_novel": ~frame.v9_connectivity_overlap.to_numpy(bool),
        "v9_scaffold_novel": ~frame.v9_scaffold_overlap.to_numpy(bool),
    }
    regression = {}
    classification = {}
    for name, keep in strata.items():
        part = frame.loc[keep].reset_index(drop=True)
        models = {}
        for model, column in {
            "v9_mixed_ic50": "v9_mixed_ic50_predicted_pic50",
            "v10_1_empirical_qhts_ic50": "v10_1_empirical_qhts_ic50_predicted_pic50",
        }.items():
            metrics = v137._regression_metrics(  # noqa: SLF001
                part.true_pic50_m.to_numpy(float), part[column].to_numpy(float)
            )
            metrics["scaffold_bootstrap_mae"] = _cluster_mae_interval(
                part, column, bootstrap
            )
            models[model] = metrics
        regression[name] = {
            "n": len(part),
            "unique_scaffolds": int(part.scaffold_group_id.nunique()),
            "models": models,
        }
        y = v137._tier(part.true_pic50_m.to_numpy(float))  # noqa: SLF001
        router = part.lgbm_rdkit2d_morgan__prediction.to_numpy(int)
        classification[name] = {
            "router": v137._classification_metrics(y, router, [0, 1, 2]),  # noqa: SLF001
            "router_probability_quality": v137._probability_metrics(  # noqa: SLF001
                y, part[list(v136.LGBM_COLUMNS)].to_numpy(float)
            ),
            "v9_regression_derived_tier": v137._classification_metrics(  # noqa: SLF001
                y,
                v137._tier(part.v9_mixed_ic50_predicted_pic50.to_numpy(float)),  # noqa: SLF001
                [0, 1, 2],
            ),
        }
    receptor_path = output / "predictions/receptor_predictions_before_score.parquet"
    receptor_result = {"available": False}
    receptor_seal_sha = None
    if receptor_path.is_file():
        receptor_seal = _read_json(
            output / "predictions/receptor_prediction_seal.json", "seal_sha256"
        )
        if _sha(receptor_path) != receptor_seal["prediction_sha256"]:
            raise CampaignError("receptor prediction seal mismatch")
        receptor = pd.read_parquet(receptor_path)
        selected = _receptor_score_frame(frame, receptor)
        y = v137._tier(selected.true_pic50_m.to_numpy(float))  # noqa: SLF001
        router = selected.lgbm_rdkit2d_morgan__prediction.to_numpy(int)
        hybrid = selected.hybrid_prediction.to_numpy(int)
        contact = spearmanr(y, selected[v137.PRIMARY_FEATURE].to_numpy(float))
        contact_pic50 = spearmanr(
            selected.true_pic50_m.to_numpy(float),
            selected[v137.PRIMARY_FEATURE].to_numpy(float),
        )
        affinity_pic50 = spearmanr(
            selected.true_pic50_m.to_numpy(float),
            selected["dock__8ZYO__affinity"].to_numpy(float),
        )
        router_metrics = v137._classification_metrics(y, router, [0, 1, 2])  # noqa: SLF001
        hybrid_metrics = v137._classification_metrics(y, hybrid, [0, 1, 2])  # noqa: SLF001
        paired = v137._balanced_bootstrap_classification(  # noqa: SLF001
            y, router, hybrid, bootstrap
        )
        scaffold_paired = v137._scaffold_bootstrap_classification(  # noqa: SLF001
            selected.scaffold_group_id.to_numpy(object),
            y,
            router,
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
        receptor_result = {
            "available": True,
            "n": len(selected),
            "router": router_metrics,
            "frozen_v13_6_hybrid": hybrid_metrics,
            "paired_bootstrap": paired,
            "paired_scaffold_cluster_bootstrap": scaffold_paired,
            "receptor_probability_quality": v137._probability_metrics(  # noqa: SLF001
                y, selected[list(v136.RECEPTOR_COLUMNS)].to_numpy(float)
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
                "potent_predicted_safe_count": int(np.sum(hybrid[y == 2] == 0)),
                "observed_potent_count": int(np.sum(y == 2)),
                "general_promotion_supported": False,
                "interpretation": (
                    "this second external set does not confirm incremental value and fails "
                    "the frozen safety gate; V13.7's gain is not generalizable evidence"
                ),
            },
            "f649_contact_vs_observed_tier_spearman": float(contact.statistic),
            "f649_contact_two_sided_p_value_descriptive": float(contact.pvalue),
            "receptor_observable_associations": {
                "f649_contact_vs_pic50_spearman": float(contact_pic50.statistic),
                "f649_contact_vs_pic50_two_sided_p_value_descriptive": float(
                    contact_pic50.pvalue
                ),
                "vina_affinity_vs_pic50_spearman": float(affinity_pic50.statistic),
                "vina_affinity_vs_pic50_two_sided_p_value_descriptive": float(
                    affinity_pic50.pvalue
                ),
                "interpretation": (
                    "descriptive univariate checks; the frozen receptor classifier is a "
                    "conditional model and these tests do not establish mechanism"
                ),
            },
            "external_labels_used_for_feature_or_policy_tuning": False,
        }
        receptor_seal_sha = receptor_seal["seal_sha256"]
    preparation = _read_json(output / "prepared/preparation_report.json", "report_sha256")
    return _json(
        output / "analysis_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": (
                (
                    "complete_with_receptor_confirmation"
                    if receptor_result.get("external_decision", {}).get(
                        "full_policy_external_gate_passed", False
                    )
                    else "complete_with_receptor_nonconfirmation"
                )
                if receptor_path.is_file()
                else "complete_ligand_only_confirmation"
            ),
            "source": {
                "article_doi": SOURCE_DOI,
                "supplement_doi": SUPPLEMENT_DOI,
                "paper_reported_corrected_external_aae": 0.50,
                "paper_predictions_available_per_structure": False,
            },
            "overlap_audit": preparation["audit"],
            "regression": regression,
            "ternary_classification": classification,
            "receptor_confirmation": receptor_result,
            "prediction_seals": {
                "baseline": baseline_seal["seal_sha256"],
                "receptor": receptor_seal_sha,
            },
            "scientific_scope": {
                "source_authors_external_relative_to_their_model": True,
                "entire_source_present_in_current_project_structure_lake": True,
                "v9_exact_and_scaffold_overlap_reported": True,
                "primary_confirmation_stratum": "v9_scaffold_novel",
                "external_labels_used_for_tuning": False,
                "prospective_validation": False,
                "vina_scores_are_binding_free_energies": False,
            },
        },
        "report_sha256",
    )


def _manifest(repo: Path, source: Path, workbook: Path, output: Path) -> dict[str, Any]:
    inputs = [
        repo / "pipeline/scripts/run_local_herg_temporal_confirmation_v13_8.py",
        repo / "pipeline/scripts/run_local_herg_external_validation_v13_7.py",
        source,
        workbook,
        repo / "research/local_runs/herg_v10_1_expanded_platform/manifest.json",
        repo / "research/local_runs/herg_domain_mixture_campaign_v9/manifest.json",
        repo / "research/local_runs/herg_receptor_hybrid_validation_v13_6/manifest.json",
    ]
    artifacts = [
        path
        for path in (
            output / "prepared/structure_registry.parquet",
            output / "prepared/preparation_report.json",
            output / "sealed/labels.parquet",
            output / "sealed/label_seal.json",
            output / "predictions/baseline_predictions_before_score.parquet",
            output / "predictions/baseline_prediction_seal.json",
            output / "selection/receptor_temporal_panel.parquet",
            output / "docking/docking_results.parquet",
            output / "docking/external_receptor_features.parquet",
            output / "docking/docking_report.json",
            output / "predictions/receptor_predictions_before_score.parquet",
            output / "predictions/receptor_prediction_seal.json",
            output / "analysis_report.json",
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
        "--stage", choices=("prepare", "baseline", "dock", "receptor", "score", "all"), default="all"
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
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
    workbook = _resolve(repo, args.workbook)
    output = _resolve(repo, args.output_root)
    v101_root = _resolve(repo, args.v10_1_root)
    v9_root = _resolve(repo, args.v9_root)
    primary = _resolve(repo, args.v13_root)
    discovery_root = _resolve(repo, args.v13_2_root)
    vina = args.vina.resolve() if args.vina else None
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "stage": args.stage}
    if args.stage in ("prepare", "all"):
        result["prepare"] = _prepare(
            repo, source, workbook, output, v101_root, v9_root
        )
    if args.stage in ("baseline", "all"):
        result["baseline"] = _baseline(output, v101_root)
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
        result["receptor"] = _receptor(repo, output, discovery_root)
    if args.stage in ("score", "all"):
        result["score"] = _score(output, args.bootstrap_replicates)
    result["manifest"] = _manifest(repo, source, workbook, output)
    return result


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _main(args)
    except (
        CampaignError,
        v137.CampaignError,
        v13.CampaignError,
        v133.CampaignError,
        v134.CampaignError,
        v136.CampaignError,
    ) as exc:
        print(f"V13.8 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
