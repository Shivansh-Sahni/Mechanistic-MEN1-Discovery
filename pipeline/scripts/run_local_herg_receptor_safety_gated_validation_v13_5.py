#!/usr/bin/env python3
"""Validate a frozen safety-gated 8ZYO/F649 classifier on 270 new scaffolds.

The raw frozen receptor classifier improved balanced accuracy in V13.3/V13.4
but produced too many potent-to-safe errors. On the now-open 360-ligand
development pool, one policy family was searched under a safety constraint.
This script freezes the selected rule before phase-three selection/docking:

* begin with the frozen V13.3 receptor-model argmax prediction;
* permit a Safe call only when P(Safe) >= 0.34 and V9 pIC50 <= 4.80;
* otherwise choose the larger of the Moderate and Potent probabilities.

The model, feature, receptor state, Vina protocol, and policy are not tuned on
phase-three labels. Predictions and policy decisions are hash-sealed before
the 270 new scaffold-unique labels are scored.
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

SCHEMA_VERSION = "platform-local-herg-receptor-safety-gated-validation-v13.5/1.0"
STATE = "8ZYO"
PRIMARY_FEATURE = "dock__8ZYO__contact_649_count"
PER_FOLD_TIER = 18
SAFE_PROBABILITY_MINIMUM = 0.34
BASELINE_PIC50_SAFE_GATE = 4.80
MAX_ABSOLUTE_POTENT_TO_SAFE_RATE = 0.05
MAX_POTENT_TO_SAFE_INCREASE = 0.03


class CampaignError(RuntimeError):
    """Validation integrity, execution, or scientific-contract failure."""


def _select_validation(
    repo: Path,
    primary: Path,
    phase1: Path,
    phase2: Path,
    output: Path,
) -> pd.DataFrame:
    discovery = pd.read_parquet(primary / "pilot/pilot_selection.parquet")
    discovery = discovery.loc[discovery.cohort.eq("exact_ic50")]
    confirmation = pd.read_parquet(phase1 / "selection/confirmation_panel.parquet")
    replication = pd.read_parquet(phase2 / "selection/replication_panel.parquet")
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
    forbidden = (
        set(discovery.scaffold_group_id.astype(str))
        | set(confirmation.scaffold_group_id.astype(str))
        | set(replication.scaffold_group_id.astype(str))
    )
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
            f"v13.5-safety-validation-{fold}-{stratum}",
        )
        rows.append(chosen)
        used_scaffolds.update(chosen.scaffold_group_id.astype(str))
    selected = pd.concat(rows, ignore_index=True)
    selected["cohort"] = "exact_ic50_safety_validation"
    selected["ligand_id"] = "validate__" + selected.structure_id.astype(str)
    selected["baseline_prediction"] = selected.pred__honest_stack.astype(float)
    selected["observed_target"] = selected.observed_pic50.astype(float)
    expected = PER_FOLD_TIER * 5 * 3
    if len(selected) != expected:
        raise CampaignError(f"validation panel has {len(selected)} rows, expected {expected}")
    if selected.groupby(["outer_fold", "stratum"]).size().ne(PER_FOLD_TIER).any():
        raise CampaignError("validation panel is not exactly balanced by fold and tier")
    if selected.scaffold_group_id.duplicated().any():
        raise CampaignError("validation panel is not scaffold-unique")
    if set(selected.scaffold_group_id.astype(str)) & forbidden:
        raise CampaignError("validation panel overlaps a prior scaffold")
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
    v133._parquet(output / "selection/validation_panel.parquet", selected)  # noqa: SLF001
    v133._json(  # noqa: SLF001
        output / "selection/selection_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": v133._utc(),  # noqa: SLF001
            "selected_ligands": len(selected),
            "unique_scaffolds": int(selected.scaffold_group_id.nunique()),
            "prior_scaffold_overlap": 0,
            "per_outer_fold_tier": PER_FOLD_TIER,
            "selection_uses_observed_tier_for_balance": True,
            "selection_labels_used_for_model_or_policy_tuning": False,
            "frozen_primary_feature": PRIMARY_FEATURE,
            "frozen_safe_probability_minimum": SAFE_PROBABILITY_MINIMUM,
            "frozen_baseline_pic50_safe_gate": BASELINE_PIC50_SAFE_GATE,
        },
        "report_sha256",
    )
    return selected


def _apply_safety_policy(predictions: pd.DataFrame) -> np.ndarray:
    probabilities = predictions[
        [f"primary_frozen_f649__probability_{name.lower()}" for name in v13.CLASS_NAMES]
    ].to_numpy(float)
    classes = np.argmax(probabilities, axis=1)
    safe_allowed = (
        (probabilities[:, 0] >= SAFE_PROBABILITY_MINIMUM)
        & (predictions.baseline_prediction.to_numpy(float) <= BASELINE_PIC50_SAFE_GATE)
    )
    override = (classes == 0) & ~safe_allowed
    classes[override] = 1 + np.argmax(probabilities[override, 1:], axis=1)
    return classes


def _class_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
        "potent_predicted_safe_rate": float(np.mean(prediction[y == 2] == 0)),
    }


def _score_policy(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    bootstrap: int,
) -> dict[str, Any]:
    y = v13._tier_index(panel.observed_target.to_numpy(float))  # noqa: SLF001
    baseline = v13._tier_index(predictions.baseline_prediction.to_numpy(float))  # noqa: SLF001
    gated = predictions.safety_gated_prediction.to_numpy(int)
    baseline_metrics = _class_metrics(y, baseline)
    gated_metrics = _class_metrics(y, gated)
    baseline_onehot = np.eye(3)[baseline]
    gated_onehot = np.eye(3)[gated]
    gated_metrics["vs_baseline_bootstrap"] = v133._balanced_bootstrap(  # noqa: SLF001
        y, baseline_onehot, gated_onehot, bootstrap
    )
    fold_metrics = []
    for fold in range(5):
        keep = panel.outer_fold.eq(fold).to_numpy()
        fold_metrics.append(
            {
                "outer_fold": fold,
                "baseline_balanced_accuracy": float(
                    balanced_accuracy_score(y[keep], baseline[keep])
                ),
                "candidate_balanced_accuracy": float(
                    balanced_accuracy_score(y[keep], gated[keep])
                ),
            }
        )
    gated_metrics["fold_metrics"] = fold_metrics
    gated_metrics["folds_better"] = int(
        sum(
            row["candidate_balanced_accuracy"] > row["baseline_balanced_accuracy"]
            for row in fold_metrics
        )
    )
    safety_passed = bool(
        gated_metrics["potent_predicted_safe_rate"] <= MAX_ABSOLUTE_POTENT_TO_SAFE_RATE
        and gated_metrics["potent_predicted_safe_rate"]
        <= baseline_metrics["potent_predicted_safe_rate"] + MAX_POTENT_TO_SAFE_INCREASE
    )
    confirmed = bool(
        gated_metrics["vs_baseline_bootstrap"]["ci95"][0] > 0
        and gated_metrics["folds_better"] >= 4
        and gated_metrics["macro_f1"] > baseline_metrics["macro_f1"]
        and safety_passed
    )
    return {
        "baseline_v9_threshold": baseline_metrics,
        "safety_gated_frozen_f649": gated_metrics,
        "validation_decision": {
            "confirmed": confirmed,
            "safety_passed": safety_passed,
            "rule": (
                "lower 95% balanced-accuracy delta >0; better in >=4/5 folds; macro-F1 "
                "improves; potent-to-safe <=0.05 absolute and <=baseline+0.03"
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--primary-root",
        type=Path,
        default=Path("research/local_runs/herg_receptor_ensemble_campaign_v13"),
    )
    parser.add_argument(
        "--discovery-root",
        type=Path,
        default=Path("research/local_runs/herg_receptor_strict_analysis_v13_2"),
    )
    parser.add_argument(
        "--phase1-root",
        type=Path,
        default=Path("research/local_runs/herg_receptor_classification_confirmation_v13_3"),
    )
    parser.add_argument(
        "--phase2-root",
        type=Path,
        default=Path("research/local_runs/herg_receptor_classification_replication_v13_4"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/local_runs/herg_receptor_safety_gated_validation_v13_5"),
    )
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
    output = _resolve(repo, args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    selected = _select_validation(repo, primary, phase1, phase2, output)
    tools = v13._resolve_toolchain(repo, args.vina.resolve() if args.vina else None)  # noqa: SLF001
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
    predictions["safety_gated_prediction"] = _apply_safety_policy(predictions)
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
            "primary_feature": PRIMARY_FEATURE,
            "model_C": 0.03,
            "safe_probability_minimum": SAFE_PROBABILITY_MINIMUM,
            "baseline_pic50_safe_gate": BASELINE_PIC50_SAFE_GATE,
            "labels_present_in_prediction_artifact": False,
        },
        "seal_sha256",
    )
    score = _score_policy(validation, predictions, args.bootstrap_replicates)
    fallback_manifests = []
    for path in sorted((output / "ligands").glob("*/microstates.json")):
        payload = json.loads(path.read_text())
        if payload.get("preparation_fallback"):
            fallback_manifests.append(
                {
                    "ligand_id": payload["ligand_id"],
                    "preparation_fallback": payload["preparation_fallback"],
                    "stereochemical_embedding_constraints_relaxed": payload.get(
                        "stereochemical_embedding_constraints_relaxed", False
                    ),
                }
            )
    report = v133._json(  # noqa: SLF001
        output / "analysis_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": v133._utc(),  # noqa: SLF001
            "status": "complete_scaffold_disjoint_safety_policy_validation",
            "selection": {
                "n": len(validation),
                "unique_scaffolds": int(validation.scaffold_group_id.nunique()),
                "prior_scaffold_overlap": 0,
                "balanced_by_observed_tier": True,
            },
            "frozen_policy": {
                "safe_probability_minimum": SAFE_PROBABILITY_MINIMUM,
                "baseline_pic50_safe_gate": BASELINE_PIC50_SAFE_GATE,
                "developed_on_phase1_plus_phase2_n": 360,
                "phase3_labels_used_for_policy_tuning": False,
            },
            "prediction_seal_sha256": prediction_seal["seal_sha256"],
            "preparation_fallbacks": fallback_manifests,
            "score": score,
            "scientific_scope": {
                "primary_feature_model_and_policy_frozen_before_phase3_docking": True,
                "phase3_labels_used_for_model_or_policy_tuning": False,
                "phase3_labels_used_for_balanced_sampling": True,
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
                "external_or_prospective_validation": False,
                "vina_scores_are_binding_free_energies": False,
            },
        },
        "report_sha256",
    )
    inputs = [
        repo / "pipeline/scripts/run_local_herg_receptor_safety_gated_validation_v13_5.py",
        repo / "pipeline/scripts/run_local_herg_receptor_classification_replication_v13_4.py",
        primary / "manifest.json",
        discovery_root / "manifest.json",
        phase1 / "manifest.json",
        phase2 / "manifest.json",
        repo / "research/local_runs/herg_domain_mixture_campaign_v9/manifest.json",
    ]
    artifacts = [
        output / "selection/validation_panel.parquet",
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
            "safety_gated_classifier_confirmed": score["validation_decision"]["confirmed"],
            "safety_gate_passed": score["validation_decision"]["safety_passed"],
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
        print(f"V13.5 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
