#!/usr/bin/env python3
"""Validate a frozen LGBM/receptor hybrid on a fourth 270-scaffold panel.

Development used the 630 previously scored V13.3/V13.4/V13.5 compounds. The
frozen policy begins with the historical V10.1 LGBM RDKit2D+Morgan OOF class,
then permits only two receptor-driven overrides:

* Safe when receptor argmax is Safe, P(Safe) >= 0.34, and V9 pIC50 <= 4.80;
* Potent when receptor argmax is Potent and P(Potent) >= 0.40.

All other calls retain LGBM. The policy, receptor model, feature, docking state,
and Vina protocol are fixed before phase-four selection. Predictions and hybrid
decisions are hash-sealed before phase-four labels are scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import analyze_local_herg_receptor_ensemble_v13_2 as v132
import numpy as np
import pandas as pd
import run_local_herg_receptor_classification_confirmation_v13_3 as v133
import run_local_herg_receptor_classification_replication_v13_4 as v134
import run_local_herg_receptor_ensemble_campaign_v13 as v13
from rdkit import Chem
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

SCHEMA_VERSION = "platform-local-herg-receptor-hybrid-validation-v13.6/1.0"
STATE = "8ZYO"
PRIMARY_FEATURE = "dock__8ZYO__contact_649_count"
PER_FOLD_TIER = 18
SAFE_PROBABILITY_MINIMUM = 0.34
BASELINE_PIC50_SAFE_GATE = 4.80
POTENT_PROBABILITY_MINIMUM = 0.40
MAX_ABSOLUTE_POTENT_TO_SAFE_RATE = 0.05
MAX_POTENT_TO_SAFE_INCREASE = 0.03
MINIMUM_MODERATE_RECALL = 0.70
MAXIMUM_MODERATE_RECALL_LOSS = 0.20
LGBM_COLUMNS = tuple(
    f"lgbm_rdkit2d_morgan__probability_{name}" for name in ("safe", "moderate", "potent")
)
RECEPTOR_COLUMNS = tuple(
    f"primary_frozen_f649__probability_{name}" for name in ("safe", "moderate", "potent")
)


class CampaignError(RuntimeError):
    """Hybrid validation integrity, execution, or scientific-contract failure."""


def _select_validation(
    repo: Path,
    primary: Path,
    phase1: Path,
    phase2: Path,
    phase3: Path,
    output: Path,
) -> pd.DataFrame:
    discovery = pd.read_parquet(primary / "pilot/pilot_selection.parquet")
    discovery = discovery.loc[discovery.cohort.eq("exact_ic50")]
    previous = [
        pd.read_parquet(phase1 / "selection/confirmation_panel.parquet"),
        pd.read_parquet(phase2 / "selection/replication_panel.parquet"),
        pd.read_parquet(phase3 / "selection/validation_panel.parquet"),
    ]
    reference = pd.read_parquet(
        repo / "research/local_runs/herg_v10_tiered_platform/reference/training_reference.parquet"
    )
    v9 = pd.read_parquet(
        repo / "research/local_runs/herg_domain_mixture_campaign_v9/analysis/nested_oof_predictions.parquet"
    )
    frame = reference.merge(
        v9[["structure_id", "outer_fold", "observed_pic50", "pred__honest_stack"]],
        on="structure_id",
        validate="one_to_one",
    )
    frame = v13._add_dockability(frame)  # noqa: SLF001
    frame = frame.loc[frame.docking_eligible].reset_index(drop=True)
    frame["tier_index"] = v13._tier_index(frame.observed_pic50.to_numpy(float))  # noqa: SLF001
    frame["stratum"] = [v13.CLASS_NAMES[value] for value in frame.tier_index]
    forbidden = set(discovery.scaffold_group_id.astype(str))
    for prior in previous:
        forbidden.update(prior.scaffold_group_id.astype(str))
    frame = frame.loc[~frame.scaffold_group_id.astype(str).isin(forbidden)].copy()
    rows = []
    used_scaffolds: set[str] = set()
    for (fold, stratum), group in frame.groupby(["outer_fold", "stratum"], observed=True):
        available = group.loc[
            ~group.scaffold_group_id.astype(str).isin(used_scaffolds)
        ]
        chosen = v13._select_diverse(  # noqa: SLF001
            available,
            PER_FOLD_TIER,
            "structure_id",
            f"v13.6-hybrid-validation-{fold}-{stratum}",
        )
        rows.append(chosen)
        used_scaffolds.update(chosen.scaffold_group_id.astype(str))
    selected = pd.concat(rows, ignore_index=True)
    selected["cohort"] = "exact_ic50_hybrid_validation"
    selected["ligand_id"] = "hybrid__" + selected.structure_id.astype(str)
    selected["baseline_prediction"] = selected.pred__honest_stack.astype(float)
    selected["observed_target"] = selected.observed_pic50.astype(float)
    expected = PER_FOLD_TIER * 5 * 3
    if len(selected) != expected:
        raise CampaignError(f"hybrid panel has {len(selected)} rows, expected {expected}")
    if selected.groupby(["outer_fold", "stratum"]).size().ne(PER_FOLD_TIER).any():
        raise CampaignError("hybrid panel is not exactly balanced by fold and tier")
    if selected.scaffold_group_id.duplicated().any():
        raise CampaignError("hybrid panel is not scaffold-unique")
    if set(selected.scaffold_group_id.astype(str)) & forbidden:
        raise CampaignError("hybrid panel overlaps a prior scaffold")
    columns = [
        "ligand_id",
        "cohort",
        "structure_id",
        "standardized_smiles",
        "scaffold_group_id",
        "outer_fold",
        "stratum",
        "tier_index",
        "baseline_prediction",
        "observed_target",
        "docking_parent_smiles",
        "docking_removed_fragment_count",
        "docking_molecular_weight",
        "docking_heavy_atom_count",
        "docking_rotatable_bond_count",
    ]
    selected = selected[columns].sort_values(
        ["outer_fold", "stratum", "ligand_id"]
    ).reset_index(drop=True)
    v133._parquet(output / "selection/hybrid_validation_panel.parquet", selected)  # noqa: SLF001
    v133._json(  # noqa: SLF001
        output / "selection/selection_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": v133._utc(),  # noqa: SLF001
            "selected_ligands": len(selected),
            "unique_scaffolds": int(selected.scaffold_group_id.nunique()),
            "prior_scaffold_overlap": 0,
            "prior_scaffolds_excluded": len(forbidden),
            "per_outer_fold_tier": PER_FOLD_TIER,
            "selection_uses_observed_tier_for_balance": True,
            "selection_labels_used_for_model_or_policy_tuning": False,
            "frozen_policy": _policy_contract(),
        },
        "report_sha256",
    )
    return selected


def _policy_contract() -> dict[str, Any]:
    return {
        "starting_classifier": "V10.1 LGBM RDKit2D+Morgan OOF argmax",
        "safe_probability_minimum": SAFE_PROBABILITY_MINIMUM,
        "baseline_pic50_safe_gate": BASELINE_PIC50_SAFE_GATE,
        "potent_probability_minimum": POTENT_PROBABILITY_MINIMUM,
        "development_n": 630,
        "development_balanced_accuracy": 2 / 3,
        "development_macro_f1": 0.673774,
        "development_potent_to_safe_rate": 0.028571,
        "phase4_labels_used_for_policy_tuning": False,
    }


def _apply_hybrid_policy(predictions: pd.DataFrame) -> np.ndarray:
    lgbm = predictions[list(LGBM_COLUMNS)].to_numpy(float)
    receptor = predictions[list(RECEPTOR_COLUMNS)].to_numpy(float)
    hybrid = np.argmax(lgbm, axis=1)
    receptor_class = np.argmax(receptor, axis=1)
    safe_override = (
        (receptor_class == 0)
        & (receptor[:, 0] >= SAFE_PROBABILITY_MINIMUM)
        & (predictions.baseline_prediction.to_numpy(float) <= BASELINE_PIC50_SAFE_GATE)
    )
    potent_override = (receptor_class == 2) & (
        receptor[:, 2] >= POTENT_PROBABILITY_MINIMUM
    )
    hybrid[safe_override] = 0
    hybrid[potent_override] = 2
    return hybrid


def _prepare_with_charge_fallback(
    row: Any,
    output: Path,
    tools: v13.Toolchain,
) -> dict[str, Any]:
    try:
        return v134._prepare_microstate(row, output, tools)  # noqa: SLF001
    except v13.CampaignError as exc:
        fallback_reason = f"{type(exc).__name__}: {exc}"
    directory = output / "ligands" / v13._safe_component(str(row.ligand_id))  # noqa: SLF001
    sdf = directory / "microstate_0.sdf"
    molecules = [molecule for molecule in Chem.SDMolSupplier(str(sdf), removeHs=False) if molecule]
    if len(molecules) != 1:
        raise CampaignError(f"zero-charge fallback lacks one valid SDF for {row.ligand_id}")
    molecule = molecules[0]
    pdbqt = directory / "microstate_0.pdbqt"
    v13._run(  # noqa: SLF001
        [
            tools.meeko_ligand,
            "-i",
            sdf,
            "-o",
            pdbqt,
            "--add_index_map",
            "--charge_model",
            "zero",
        ],
        directory / "meeko_0_zero_charge.log",
    )
    microstate = {
        "microstate_index": 0,
        "microstate_smiles": Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True),
        "formal_charge": int(Chem.GetFormalCharge(molecule)),
        "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
        "sdf_path": str(sdf.resolve()),
        "pdbqt_path": str(pdbqt.resolve()),
        "pdbqt_sha256": v133._sha(pdbqt),  # noqa: SLF001
    }
    v13._self_hashed_json(  # noqa: SLF001
        directory / "microstates.json",
        {
            "schema_version": SCHEMA_VERSION,
            "ligand_id": str(row.ligand_id),
            "input_smiles": str(row.standardized_smiles),
            "docking_parent_smiles": str(row.docking_parent_smiles),
            "removed_fragment_count": int(row.docking_removed_fragment_count),
            "ph": 7.4,
            "tautomer_enumeration": False,
            "generated_microstate_count": 1,
            "retained_microstate_count": 1,
            "microstates": [microstate],
            "preparation_fallback": "Meeko zero-charge for Gasteiger-unsupported graph",
            "charge_model": "zero",
            "fallback_reason": fallback_reason,
            "truth_boundary": (
                "Gasteiger produced non-finite charges; zero partial charges retained the fixed "
                "panel member but remove ligand electrostatics for this docking"
            ),
        },
        "report_sha256",
    )
    return microstate


def _prepare_panel(
    selected: pd.DataFrame,
    output: Path,
    tools: v13.Toolchain,
) -> None:
    total = len(selected)
    for completed, row in enumerate(selected.itertuples(index=False), start=1):
        _prepare_with_charge_fallback(row, output, tools)
        if completed % 25 == 0 or completed == total:
            print(
                json.dumps(
                    {"stage": "hybrid_ligand_preflight", "completed": completed, "total": total}
                ),
                flush=True,
            )


def _class_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    recalls = [float(np.mean(prediction[y == index] == index)) for index in range(3)]
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "per_class_recall": dict(zip(v13.CLASS_NAMES, recalls, strict=True)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
        "potent_predicted_safe_rate": float(np.mean(prediction[y == 2] == 0)),
    }


def _moderate_retention_passed(candidate_recall: float, baseline_recall: float) -> bool:
    tolerance = 1e-12
    return bool(
        candidate_recall + tolerance >= MINIMUM_MODERATE_RECALL
        and candidate_recall + tolerance
        >= baseline_recall - MAXIMUM_MODERATE_RECALL_LOSS
    )


def _score(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    bootstrap: int,
) -> dict[str, Any]:
    y = v13._tier_index(panel.observed_target.to_numpy(float))  # noqa: SLF001
    lgbm = np.argmax(predictions[list(LGBM_COLUMNS)].to_numpy(float), axis=1)
    hybrid = predictions.hybrid_prediction.to_numpy(int)
    baseline_metrics = _class_metrics(y, lgbm)
    hybrid_metrics = _class_metrics(y, hybrid)
    hybrid_metrics["vs_lgbm_bootstrap"] = v133._balanced_bootstrap(  # noqa: SLF001
        y, np.eye(3)[lgbm], np.eye(3)[hybrid], bootstrap
    )
    fold_metrics = []
    for fold in range(5):
        keep = panel.outer_fold.eq(fold).to_numpy()
        fold_metrics.append(
            {
                "outer_fold": fold,
                "lgbm_balanced_accuracy": float(
                    balanced_accuracy_score(y[keep], lgbm[keep])
                ),
                "hybrid_balanced_accuracy": float(
                    balanced_accuracy_score(y[keep], hybrid[keep])
                ),
            }
        )
    hybrid_metrics["fold_metrics"] = fold_metrics
    hybrid_metrics["folds_better"] = int(
        sum(row["hybrid_balanced_accuracy"] > row["lgbm_balanced_accuracy"] for row in fold_metrics)
    )
    safety_passed = bool(
        hybrid_metrics["potent_predicted_safe_rate"] <= MAX_ABSOLUTE_POTENT_TO_SAFE_RATE
        and hybrid_metrics["potent_predicted_safe_rate"]
        <= baseline_metrics["potent_predicted_safe_rate"] + MAX_POTENT_TO_SAFE_INCREASE
    )
    moderate_recall = hybrid_metrics["per_class_recall"]["Moderate"]
    baseline_moderate_recall = baseline_metrics["per_class_recall"]["Moderate"]
    moderate_retention_passed = _moderate_retention_passed(
        moderate_recall, baseline_moderate_recall
    )
    confirmed = bool(
        hybrid_metrics["vs_lgbm_bootstrap"]["ci95"][0] > 0
        and hybrid_metrics["folds_better"] >= 4
        and hybrid_metrics["macro_f1"] > baseline_metrics["macro_f1"]
        and safety_passed
        and moderate_retention_passed
    )
    return {
        "lgbm_rdkit2d_morgan": baseline_metrics,
        "frozen_hybrid": hybrid_metrics,
        "validation_decision": {
            "confirmed": confirmed,
            "safety_passed": safety_passed,
            "moderate_retention_passed": moderate_retention_passed,
            "rule": (
                "lower 95% BA delta vs LGBM >0; better in >=4/5 folds; macro-F1 improves; "
                "potent-to-safe <=0.05 and <=LGBM+0.03; Moderate recall >=0.70 and "
                ">=LGBM-0.20"
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    defaults = {
        "primary_root": "research/local_runs/herg_receptor_ensemble_campaign_v13",
        "discovery_root": "research/local_runs/herg_receptor_strict_analysis_v13_2",
        "phase1_root": "research/local_runs/herg_receptor_classification_confirmation_v13_3",
        "phase2_root": "research/local_runs/herg_receptor_classification_replication_v13_4",
        "phase3_root": "research/local_runs/herg_receptor_safety_gated_validation_v13_5",
        "output_root": "research/local_runs/herg_receptor_hybrid_validation_v13_6",
    }
    for name, value in defaults.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=Path(value))
    parser.add_argument("--vina", type=Path)
    parser.add_argument("--cpu", type=int, default=6)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--modes", type=int, default=9)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser


def _resolve(repo: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _main(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    primary = _resolve(repo, args.primary_root)
    discovery_root = _resolve(repo, args.discovery_root)
    phase1 = _resolve(repo, args.phase1_root)
    phase2 = _resolve(repo, args.phase2_root)
    phase3 = _resolve(repo, args.phase3_root)
    output = _resolve(repo, args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    selected = _select_validation(repo, primary, phase1, phase2, phase3, output)
    tools = v13._resolve_toolchain(repo, args.vina.resolve() if args.vina else None)  # noqa: SLF001
    _prepare_panel(selected, output, tools)
    docking = v134._dock(  # noqa: SLF001
        primary,
        output,
        tools,
        selected,
        args.exhaustiveness,
        args.modes,
        args.cpu,
    )
    features = v13._aggregate_docking(docking, (STATE,))  # noqa: SLF001
    validation = selected.merge(features, on="ligand_id", validate="one_to_one")
    discovery = pd.read_parquet(discovery_root / "six_state_analysis_matrix.parquet")
    predictions = v134._predict_before_score(repo, discovery, validation)  # noqa: SLF001
    router_path = repo / "research/local_runs/herg_v10_1_expanded_platform/evidence/exact_router_oof.parquet"
    router = pd.read_parquet(router_path)[
        ["structure_id", "scaffold_group_id", "outer_fold", *LGBM_COLUMNS]
    ]
    predictions = predictions.merge(
        router,
        on=["structure_id", "scaffold_group_id", "outer_fold"],
        validate="one_to_one",
    )
    predictions["hybrid_prediction"] = _apply_hybrid_policy(predictions)
    prediction_path = output / "predictions/predictions_before_score.parquet"
    v133._parquet(prediction_path, predictions)  # noqa: SLF001
    prediction_seal = v133._json(  # noqa: SLF001
        output / "predictions/prediction_seal.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": v133._utc(),  # noqa: SLF001
            "prediction_path": str(prediction_path.resolve()),
            "prediction_bytes": prediction_path.stat().st_size,
            "prediction_sha256": v133._sha(prediction_path),  # noqa: SLF001
            "rows": len(predictions),
            "policy": _policy_contract(),
            "labels_present_in_prediction_artifact": False,
        },
        "seal_sha256",
    )
    score = _score(validation, predictions, args.bootstrap_replicates)
    fallbacks = []
    for path in sorted((output / "ligands").glob("*/microstates.json")):
        payload = json.loads(path.read_text())
        if payload.get("preparation_fallback"):
            fallbacks.append(
                {
                    "ligand_id": payload["ligand_id"],
                    "preparation_fallback": payload["preparation_fallback"],
                    "stereochemical_embedding_constraints_relaxed": payload.get(
                        "stereochemical_embedding_constraints_relaxed", False
                    ),
                }
            )
    fallback_ids = {item["ligand_id"] for item in fallbacks}
    fallback_exclusion_sensitivity = None
    if fallback_ids:
        if not validation.ligand_id.equals(predictions.ligand_id):
            raise CampaignError("validation and prediction rows are not aligned for sensitivity analysis")
        keep = ~validation.ligand_id.isin(fallback_ids).to_numpy()
        fallback_exclusion_sensitivity = _score(
            validation.loc[keep].reset_index(drop=True),
            predictions.loc[keep].reset_index(drop=True),
            args.bootstrap_replicates,
        )
    report = v133._json(  # noqa: SLF001
        output / "analysis_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": v133._utc(),  # noqa: SLF001
            "status": "complete_scaffold_disjoint_hybrid_validation",
            "selection": {
                "n": len(validation),
                "unique_scaffolds": int(validation.scaffold_group_id.nunique()),
                "prior_scaffold_overlap": 0,
                "balanced_by_observed_tier": True,
            },
            "frozen_policy": _policy_contract(),
            "prediction_seal_sha256": prediction_seal["seal_sha256"],
            "preparation_fallbacks": fallbacks,
            "fallback_exclusion_sensitivity": fallback_exclusion_sensitivity,
            "score": score,
            "scientific_scope": {
                "model_feature_and_policy_frozen_before_phase4_docking": True,
                "phase4_labels_used_for_model_or_policy_tuning": False,
                "phase4_labels_used_for_balanced_sampling": True,
                "lgbm_comparator_is_historical_oof_on_identical_rows": True,
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
                "external_or_prospective_validation": False,
                "vina_scores_are_binding_free_energies": False,
            },
        },
        "report_sha256",
    )
    inputs = [
        repo / "pipeline/scripts/run_local_herg_receptor_hybrid_validation_v13_6.py",
        repo / "pipeline/scripts/run_local_herg_receptor_classification_replication_v13_4.py",
        primary / "manifest.json",
        discovery_root / "manifest.json",
        phase1 / "manifest.json",
        phase2 / "manifest.json",
        phase3 / "manifest.json",
        router_path,
        repo / "research/local_runs/herg_domain_mixture_campaign_v9/manifest.json",
    ]
    artifacts = [
        output / "selection/hybrid_validation_panel.parquet",
        output / "selection/selection_report.json",
        output / "docking/docking_results.parquet",
        prediction_path,
        output / "predictions/prediction_seal.json",
        output / "analysis_report.json",
    ]
    manifest = v133._json(  # noqa: SLF001
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": v133._utc(),  # noqa: SLF001
            "status": "complete",
            "inputs": [
                {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": v133._sha(path)}  # noqa: SLF001
                for path in inputs
            ],
            "artifacts": [
                {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": v133._sha(path)}  # noqa: SLF001
                for path in artifacts
            ],
        },
        "manifest_sha256",
    )
    return v133._json(  # noqa: SLF001
        output / "final_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "finished_utc": v133._utc(),  # noqa: SLF001
            "status": "complete",
            "validation_n": len(validation),
            "hybrid_confirmed": score["validation_decision"]["confirmed"],
            "safety_passed": score["validation_decision"]["safety_passed"],
            "moderate_retention_passed": score["validation_decision"][
                "moderate_retention_passed"
            ],
            "report_sha256": report["report_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "summary_sha256",
    )


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
    ) as exc:
        print(f"V13.6 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
