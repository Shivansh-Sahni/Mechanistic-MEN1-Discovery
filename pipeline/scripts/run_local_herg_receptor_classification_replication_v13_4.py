#!/usr/bin/env python3
"""Run a powered independent replication of the frozen V13 8ZYO/F649 signal.

V13.3 was directionally positive but did not pass its predeclared confirmation
gate. This second batch selects 270 new ligands (18 per outer-fold/tier),
excludes every V13 discovery and V13.3 confirmation scaffold, and docks only
the primary 8ZYO state. The discovery-trained model, C value, feature, class
boundaries, and decision rule are unchanged. Predictions are written and
self-hashed before batch-two labels are scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import analyze_local_herg_receptor_ensemble_v13_2 as v132
import numpy as np
import pandas as pd
import run_local_herg_receptor_classification_confirmation_v13_3 as v133
import run_local_herg_receptor_ensemble_campaign_v13 as v13
from rdkit import Chem
from rdkit.Chem import AllChem

SCHEMA_VERSION = "platform-local-herg-receptor-classification-replication-v13.4/1.0"
STATE = "8ZYO"
PRIMARY_FEATURE = "dock__8ZYO__contact_649_count"
PER_FOLD_TIER = 18


class CampaignError(RuntimeError):
    """Replication integrity, execution, or scientific-contract failure."""


def _select_replication(
    repo: Path,
    primary: Path,
    phase1: Path,
    output: Path,
) -> pd.DataFrame:
    discovery = pd.read_parquet(primary / "pilot/pilot_selection.parquet")
    discovery = discovery.loc[discovery.cohort.eq("exact_ic50")]
    confirmation = pd.read_parquet(phase1 / "selection/confirmation_panel.parquet")
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
    forbidden = set(discovery.scaffold_group_id.astype(str)) | set(
        confirmation.scaffold_group_id.astype(str)
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
            f"v13.4-replication-{fold}-{stratum}",
        )
        rows.append(chosen)
        used_scaffolds.update(chosen.scaffold_group_id.astype(str))
    selected = pd.concat(rows, ignore_index=True)
    selected["cohort"] = "exact_ic50_replication"
    selected["ligand_id"] = "replicate__" + selected.structure_id.astype(str)
    selected["baseline_prediction"] = selected.pred__honest_stack.astype(float)
    selected["observed_target"] = selected.observed_pic50.astype(float)
    expected = PER_FOLD_TIER * 5 * 3
    if len(selected) != expected:
        raise CampaignError(f"replication panel has {len(selected)} rows, expected {expected}")
    if selected.groupby(["outer_fold", "stratum"]).size().ne(PER_FOLD_TIER).any():
        raise CampaignError("replication panel is not exactly balanced by fold and tier")
    if selected.scaffold_group_id.duplicated().any():
        raise CampaignError("replication panel is not scaffold-unique")
    if set(selected.scaffold_group_id.astype(str)) & forbidden:
        raise CampaignError("replication panel overlaps a prior scaffold")
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
    v133._parquet(output / "selection/replication_panel.parquet", selected)  # noqa: SLF001
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
            "selection_labels_used_for_model_tuning": False,
            "primary_feature_frozen_before_selection": PRIMARY_FEATURE,
        },
        "report_sha256",
    )
    return selected


def _dock(
    primary: Path,
    output: Path,
    tools: v13.Toolchain,
    selected: pd.DataFrame,
    exhaustiveness: int,
    modes: int,
    cpu: int,
) -> pd.DataFrame:
    receptor_atoms = v13._receptor_contact_atoms(  # noqa: SLF001
        primary / f"receptors/{STATE}/{STATE}_prepared.pdb"
    )
    receptor = primary / f"receptors/{STATE}/{STATE}_prepared.pdbqt"
    config = primary / f"receptors/{STATE}/{STATE}_prepared.box.txt"
    rows: list[dict[str, Any]] = []
    total = len(selected)
    for completed, ligand in enumerate(selected.itertuples(index=False), start=1):
        microstate = _prepare_microstate(ligand, output, tools)
        task_id = (
            f"{v13._safe_component(ligand.ligand_id)}"  # noqa: SLF001
            f"__m{microstate['microstate_index']}__{STATE}"
        )
        directory = output / "docking/tasks" / task_id
        directory.mkdir(parents=True, exist_ok=True)
        docked = directory / "poses.pdbqt"
        seed = v13.SEED + int(hashlib.sha256(task_id.encode()).hexdigest()[:7], 16)
        if not docked.is_file():
            v13._run(  # noqa: SLF001
                [
                    tools.vina,
                    "--receptor",
                    receptor,
                    "--ligand",
                    microstate["pdbqt_path"],
                    "--config",
                    config,
                    "--exhaustiveness",
                    str(exhaustiveness),
                    "--num_modes",
                    str(modes),
                    "--seed",
                    str(seed),
                    "--cpu",
                    str(cpu),
                    "--out",
                    docked,
                ],
                directory / "vina.log",
            )
        models = v13._pdbqt_models(docked)  # noqa: SLF001
        affinities = np.asarray([model[0] for model in models], dtype=float)
        best_index = int(np.argmin(affinities))
        features = v13._contact_features(models[best_index][1], receptor_atoms)  # noqa: SLF001
        rows.append(
            {
                "ligand_id": ligand.ligand_id,
                "structure_id": ligand.structure_id,
                "scaffold_group_id": ligand.scaffold_group_id,
                "outer_fold": int(ligand.outer_fold),
                "pdb_id": STATE,
                "microstate_index": microstate["microstate_index"],
                "formal_charge": microstate["formal_charge"],
                "heavy_atom_count": microstate["heavy_atom_count"],
                "seed": seed,
                "pose_count": len(models),
                "best_affinity_kcal_mol": float(np.min(affinities)),
                "median_affinity_kcal_mol": float(np.median(affinities)),
                "affinity_range_kcal_mol": float(np.max(affinities) - np.min(affinities)),
                "pose_count_within_1kcal": int(
                    np.sum(affinities <= np.min(affinities) + 1.0)
                ),
                "ligand_efficiency_kcal_mol_per_heavy_atom": float(
                    np.min(affinities) / max(1, microstate["heavy_atom_count"])
                ),
                **features,
            }
        )
        if completed % 10 == 0 or completed == total:
            print(
                json.dumps({"stage": "replication_dock", "completed": completed, "total": total}),
                flush=True,
            )
        v133._parquet(  # noqa: SLF001
            output / "docking/docking_results.partial.parquet", pd.DataFrame(rows)
        )
    frame = pd.DataFrame(rows)
    v133._parquet(output / "docking/docking_results.parquet", frame)  # noqa: SLF001
    partial = output / "docking/docking_results.partial.parquet"
    if partial.is_file():
        partial.unlink()
    if len(frame) != len(selected) or frame.ligand_id.nunique() != len(selected):
        raise CampaignError("replication docking coverage is incomplete")
    return frame


def _prepare_microstate(row: Any, output: Path, tools: v13.Toolchain) -> dict[str, Any]:
    """Use the V13 pH workflow, retaining a fixed ligand via an RDKit fallback."""
    try:
        return v13._prepare_ligand_microstates(row, output, tools, 1)[0]  # noqa: SLF001
    except (OSError, v13.CampaignError) as exc:
        fallback_reason = f"{type(exc).__name__}: {exc}"
    directory = output / "ligands" / v13._safe_component(str(row.ligand_id))  # noqa: SLF001
    parent = Chem.MolFromSmiles(str(row.docking_parent_smiles))
    if parent is None:
        raise CampaignError(f"RDKit fallback could not parse {row.ligand_id}")
    molecule = Chem.AddHs(parent)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = v13.SEED
    status = AllChem.EmbedMolecule(molecule, parameters)
    stereo_relaxed = False
    if status != 0:
        parent = Chem.Mol(parent)
        Chem.RemoveStereochemistry(parent)
        molecule = Chem.AddHs(parent)
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = v13.SEED
        parameters.enforceChirality = False
        parameters.ignoreSmoothingFailures = True
        status = AllChem.EmbedMolecule(molecule, parameters)
        stereo_relaxed = status == 0
    if status != 0:
        raise CampaignError(f"RDKit fallback could not embed {row.ligand_id}")
    if AllChem.MMFFHasAllMoleculeParams(molecule):
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=1_000)
        optimizer = "MMFF94"
    else:
        AllChem.UFFOptimizeMolecule(molecule, maxIters=1_000)
        optimizer = "UFF"
    sdf = directory / "microstate_0.sdf"
    pdbqt = directory / "microstate_0.pdbqt"
    writer = Chem.SDWriter(str(sdf))
    writer.write(molecule)
    writer.close()
    v13._run(  # noqa: SLF001
        [
            tools.meeko_ligand,
            "-i",
            sdf,
            "-o",
            pdbqt,
            "--add_index_map",
            "--charge_model",
            "gasteiger",
        ],
        directory / "meeko_0.log",
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
            "preparation_fallback": "RDKit ETKDGv3 preserving input protonation/charge",
            "fallback_reason": fallback_reason,
            "stereochemical_embedding_constraints_relaxed": stereo_relaxed,
            "optimizer": optimizer,
            "truth_boundary": (
                "molscrub failed; deterministic input-graph/charge conformer retained to avoid "
                "outcome-dependent panel exclusion; stereochemical constraints are explicitly "
                "flagged if distance geometry required relaxation"
            ),
        },
        "report_sha256",
    )
    return microstate


def _predict_before_score(
    repo: Path,
    discovery_matrix: pd.DataFrame,
    replication: pd.DataFrame,
) -> pd.DataFrame:
    discovery = discovery_matrix.loc[discovery_matrix.cohort.eq("exact_ic50")].reset_index(
        drop=True
    )
    inner = v132._v9_inner_baselines(repo, discovery)  # noqa: SLF001
    predictions = replication[
        ["ligand_id", "structure_id", "scaffold_group_id", "outer_fold", "baseline_prediction"]
    ].copy()
    baseline_tier = v13._tier_index(predictions.baseline_prediction.to_numpy(float))  # noqa: SLF001
    for class_index, class_name in enumerate(v13.CLASS_NAMES):
        predictions[f"baseline_probability_{class_name.lower()}"] = (
            baseline_tier == class_index
        ).astype(float)
    probabilities = np.full((len(replication), 3), np.nan)
    for outer in range(5):
        fit = discovery.outer_fold.ne(outer).to_numpy()
        evaluate = replication.outer_fold.eq(outer).to_numpy()
        baseline_map = inner.loc[inner.context_outer_fold.eq(outer)].set_index(
            "structure_id"
        ).inner_baseline
        training = discovery.loc[fit, [PRIMARY_FEATURE]].copy()
        training.insert(
            0,
            "strict_inner_baseline",
            discovery.loc[fit, "structure_id"].map(baseline_map).to_numpy(float),
        )
        validation = replication.loc[evaluate, [PRIMARY_FEATURE]].copy()
        validation.insert(
            0,
            "strict_inner_baseline",
            replication.loc[evaluate, "baseline_prediction"].to_numpy(float),
        )
        feature_columns = ["strict_inner_baseline", PRIMARY_FEATURE]
        model = v133._classifier()  # noqa: SLF001
        model.fit(
            training[feature_columns],
            v13._tier_index(discovery.loc[fit, "observed_target"].to_numpy(float)),  # noqa: SLF001
        )
        fold_probabilities = model.predict_proba(validation[feature_columns])
        rows = np.flatnonzero(evaluate)
        classes = model.named_steps["model"].classes_.astype(int)
        probabilities[np.ix_(rows, classes)] = fold_probabilities
    if not np.isfinite(probabilities).all():
        raise CampaignError("incomplete pre-score replication predictions")
    for class_index, class_name in enumerate(v13.CLASS_NAMES):
        predictions[f"primary_frozen_f649__probability_{class_name.lower()}"] = probabilities[
            :, class_index
        ]
    return predictions


def _score_panel(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    bootstrap: int,
) -> dict[str, Any]:
    y = v13._tier_index(panel.observed_target.to_numpy(float))  # noqa: SLF001
    baseline = predictions[
        [f"baseline_probability_{name.lower()}" for name in v13.CLASS_NAMES]
    ].to_numpy(float)
    candidate = predictions[
        [f"primary_frozen_f649__probability_{name.lower()}" for name in v13.CLASS_NAMES]
    ].to_numpy(float)
    baseline_metrics = v133._metrics(y, baseline)  # noqa: SLF001
    candidate_metrics = v133._metrics(y, candidate)  # noqa: SLF001
    fold_metrics = []
    for fold in range(5):
        keep = panel.outer_fold.eq(fold).to_numpy()
        fold_metrics.append(
            {
                "outer_fold": fold,
                "baseline_balanced_accuracy": v133._metrics(y[keep], baseline[keep])[  # noqa: SLF001
                    "balanced_accuracy"
                ],
                "candidate_balanced_accuracy": v133._metrics(y[keep], candidate[keep])[  # noqa: SLF001
                    "balanced_accuracy"
                ],
            }
        )
    candidate_metrics["vs_baseline_bootstrap"] = v133._balanced_bootstrap(  # noqa: SLF001
        y, baseline, candidate, bootstrap
    )
    candidate_metrics["fold_metrics"] = fold_metrics
    candidate_metrics["folds_better"] = int(
        sum(
            row["candidate_balanced_accuracy"] > row["baseline_balanced_accuracy"]
            for row in fold_metrics
        )
    )
    confirmed = bool(
        candidate_metrics["vs_baseline_bootstrap"]["ci95"][0] > 0
        and candidate_metrics["folds_better"] >= 4
        and candidate_metrics["macro_f1"] > baseline_metrics["macro_f1"]
        and candidate_metrics["potent_predicted_safe_rate"]
        <= baseline_metrics["potent_predicted_safe_rate"] + 0.05
    )
    return {
        "baseline_v9_threshold": baseline_metrics,
        "primary_frozen_f649": candidate_metrics,
        "confirmation_decision": {
            "confirmed": confirmed,
            "rule": (
                "lower 95% balanced-accuracy delta >0; better in >=4/5 folds; "
                "macro-F1 improves; potent-to-safe rate increases by <=0.05"
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
        "--output-root",
        type=Path,
        default=Path("research/local_runs/herg_receptor_classification_replication_v13_4"),
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
    output = _resolve(repo, args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    selected = _select_replication(repo, primary, phase1, output)
    tools = v13._resolve_toolchain(repo, args.vina.resolve() if args.vina else None)  # noqa: SLF001
    docking = _dock(
        primary,
        output,
        tools,
        selected,
        args.exhaustiveness,
        args.modes,
        args.cpu,
    )
    features = v13._aggregate_docking(docking, (STATE,))  # noqa: SLF001
    replication = selected.merge(features, on="ligand_id", validate="one_to_one")
    discovery = pd.read_parquet(discovery_root / "six_state_analysis_matrix.parquet")
    predictions = _predict_before_score(repo, discovery, replication)
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
            "decision_rule": "argmax",
            "labels_present_in_prediction_artifact": False,
        },
        "seal_sha256",
    )
    phase2_score = _score_panel(replication, predictions, args.bootstrap_replicates)
    phase1_panel = pd.read_parquet(phase1 / "selection/confirmation_panel.parquet")
    phase1_predictions = pd.read_parquet(
        phase1 / "predictions/predictions_before_score.parquet"
    )
    pooled_panel = pd.concat([phase1_panel, selected], ignore_index=True)
    pooled_predictions = pd.concat(
        [phase1_predictions[predictions.columns], predictions], ignore_index=True
    )
    if pooled_panel.scaffold_group_id.duplicated().any():
        raise CampaignError("pooled confirmation panels overlap by scaffold")
    pooled_score = _score_panel(pooled_panel, pooled_predictions, args.bootstrap_replicates)
    phase1_delta = json.loads((phase1 / "analysis_report.json").read_text())["score"][
        "hypotheses"
    ]["primary_frozen_f649"]["vs_baseline_bootstrap"][
        "delta_balanced_accuracy_candidate_minus_baseline"
    ]
    phase2_delta = phase2_score["primary_frozen_f649"]["vs_baseline_bootstrap"][
        "delta_balanced_accuracy_candidate_minus_baseline"
    ]
    pooled_support = bool(
        phase1_delta > 0
        and phase2_delta > 0
        and pooled_score["confirmation_decision"]["confirmed"]
    )
    report = v133._json(  # noqa: SLF001
        output / "analysis_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": v133._utc(),  # noqa: SLF001
            "status": "complete_powered_scaffold_disjoint_internal_replication",
            "selection": {
                "n": len(replication),
                "unique_scaffolds": int(replication.scaffold_group_id.nunique()),
                "prior_scaffold_overlap": 0,
                "balanced_by_observed_tier": True,
            },
            "prediction_seal_sha256": prediction_seal["seal_sha256"],
            "phase2_independent": phase2_score,
            "sequential_pooled_phase1_plus_phase2": {
                **pooled_score,
                "phase1_delta_balanced_accuracy": phase1_delta,
                "phase2_delta_balanced_accuracy": phase2_delta,
                "pooled_support_with_both_batches_directionally_positive": pooled_support,
                "interpretation": (
                    "secondary sequential evidence declared after phase 1; independent phase-2 "
                    "decision remains the primary replication claim"
                ),
            },
            "scientific_scope": {
                "primary_feature_and_model_frozen_before_phase2_docking": True,
                "phase2_labels_used_for_model_tuning": False,
                "phase2_labels_used_for_balanced_sampling": True,
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
                "external_or_prospective_validation": False,
                "vina_scores_are_binding_free_energies": False,
            },
        },
        "report_sha256",
    )
    inputs = [
        repo / "pipeline/scripts/run_local_herg_receptor_classification_replication_v13_4.py",
        repo / "pipeline/scripts/run_local_herg_receptor_classification_confirmation_v13_3.py",
        primary / "manifest.json",
        discovery_root / "manifest.json",
        phase1 / "manifest.json",
        repo / "research/local_runs/herg_domain_mixture_campaign_v9/manifest.json",
    ]
    artifacts = [
        output / "selection/replication_panel.parquet",
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
            "replication_n": len(replication),
            "independently_confirmed": phase2_score["confirmation_decision"]["confirmed"],
            "sequential_pooled_support": pooled_support,
            "report_sha256": report["report_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "summary_sha256",
    )


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _main(args)
    except (CampaignError, v13.CampaignError, v132.CampaignError, v133.CampaignError) as exc:
        print(f"V13.4 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
