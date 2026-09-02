#!/usr/bin/env python3
"""Confirm the post-hoc V13.2 astemizole-state F649 classification signal.

Discovery used 90 balanced exact-IC50 ligands. This campaign selects 90 new
ligands (six per outer-fold/tier), excluding every discovery scaffold, docks
only the two sensitivity states needed by the frozen hypotheses, writes and
self-hashes predictions before scoring, then compares against the V9 threshold
baseline. The primary hypothesis is the single 8ZYO F649 ligand-contact count;
the secondary hypothesis is the full 8ZYO/8ZYQ contact surface. No validation
labels are used to tune the model, C value, feature set, or decision rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import analyze_local_herg_receptor_ensemble_v13_2 as v132
import numpy as np
import pandas as pd
import run_local_herg_receptor_ensemble_campaign_v13 as v13
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCHEMA_VERSION = "platform-local-herg-receptor-classification-confirmation-v13.3/1.0"
SEED = 20260821
STATES = ("8ZYO", "8ZYQ")
PRIMARY_FEATURES = ("dock__8ZYO__contact_649_count",)


class CampaignError(RuntimeError):
    """Confirmation integrity, execution, or scientific-contract failure."""


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


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    prediction = np.argmax(probabilities, axis=1)
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
        "potent_predicted_safe_rate": float(np.mean(prediction[y == 2] == 0)),
    }


def _balanced_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    replicates: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    baseline_class = np.argmax(baseline, axis=1)
    candidate_class = np.argmax(candidate, axis=1)
    by_class = [np.flatnonzero(y == class_index) for class_index in range(3)]
    deltas = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        rows = np.concatenate(
            [rng.choice(indices, len(indices), replace=True) for indices in by_class]
        )
        deltas[replicate] = balanced_accuracy_score(y[rows], candidate_class[rows]) - balanced_accuracy_score(
            y[rows], baseline_class[rows]
        )
    observed_delta = balanced_accuracy_score(y, candidate_class) - balanced_accuracy_score(
        y, baseline_class
    )
    return {
        "delta_balanced_accuracy_candidate_minus_baseline": float(observed_delta),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "probability_candidate_better": float(np.mean(deltas > 0)),
        "replicates": replicates,
        "resampling": "within observed tier; paired predictions",
    }


def _select_confirmation(repo: Path, primary: Path, output: Path) -> pd.DataFrame:
    discovery = pd.read_parquet(primary / "pilot/pilot_selection.parquet")
    discovery = discovery.loc[discovery.cohort.eq("exact_ic50")].copy()
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
    forbidden_scaffolds = set(discovery.scaffold_group_id.astype(str))
    frame = frame.loc[~frame.scaffold_group_id.astype(str).isin(forbidden_scaffolds)].copy()
    rows = []
    for (fold, stratum), group in frame.groupby(["outer_fold", "stratum"], observed=True):
        chosen = v13._select_diverse(  # noqa: SLF001
            group,
            6,
            "structure_id",
            f"v13.3-confirm-{fold}-{stratum}",
        )
        rows.append(chosen)
    selected = pd.concat(rows, ignore_index=True)
    selected["cohort"] = "exact_ic50_confirmation"
    selected["ligand_id"] = "confirm__" + selected.structure_id.astype(str)
    selected["baseline_prediction"] = selected.pred__honest_stack.astype(float)
    selected["observed_target"] = selected.observed_pic50.astype(float)
    if len(selected) != 90 or selected.groupby(["outer_fold", "stratum"]).size().ne(6).any():
        raise CampaignError("confirmation panel is not exactly balanced 6 x 5 folds x 3 tiers")
    if set(selected.scaffold_group_id.astype(str)) & forbidden_scaffolds:
        raise CampaignError("confirmation panel overlaps a discovery scaffold")
    if selected.scaffold_group_id.duplicated().any():
        raise CampaignError("confirmation panel is not scaffold-unique")
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
    selected = selected[columns].sort_values(["outer_fold", "stratum", "ligand_id"]).reset_index(drop=True)
    _parquet(output / "selection/confirmation_panel.parquet", selected)
    _json(
        output / "selection/selection_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "selected_ligands": len(selected),
            "unique_scaffolds": int(selected.scaffold_group_id.nunique()),
            "discovery_scaffold_overlap": 0,
            "per_outer_fold_tier": 6,
            "selection_uses_observed_tier_for_balance": True,
            "selection_labels_used_for_model_tuning": False,
            "primary_feature_frozen_before_selection": list(PRIMARY_FEATURES),
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
    receptor_atoms = {
        state: v13._receptor_contact_atoms(  # noqa: SLF001
            primary / f"receptors/{state}/{state}_prepared.pdb"
        )
        for state in STATES
    }
    rows: list[dict[str, Any]] = []
    total = len(selected) * len(STATES)
    completed = 0
    for ligand in selected.itertuples(index=False):
        microstate = v13._prepare_ligand_microstates(ligand, output, tools, 1)[0]  # noqa: SLF001
        for state in STATES:
            receptor = primary / f"receptors/{state}/{state}_prepared.pdbqt"
            config = primary / f"receptors/{state}/{state}_prepared.box.txt"
            task_id = (
                f"{v13._safe_component(ligand.ligand_id)}"  # noqa: SLF001
                f"__m{microstate['microstate_index']}__{state}"
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
            features = v13._contact_features(  # noqa: SLF001
                models[best_index][1], receptor_atoms[state]
            )
            rows.append(
                {
                    "ligand_id": ligand.ligand_id,
                    "structure_id": ligand.structure_id,
                    "scaffold_group_id": ligand.scaffold_group_id,
                    "outer_fold": int(ligand.outer_fold),
                    "pdb_id": state,
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
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(
                    json.dumps({"stage": "confirmation_dock", "completed": completed, "total": total}),
                    flush=True,
                )
            _parquet(output / "docking/docking_results.partial.parquet", pd.DataFrame(rows))
    frame = pd.DataFrame(rows)
    _parquet(output / "docking/docking_results.parquet", frame)
    partial = output / "docking/docking_results.partial.parquet"
    if partial.is_file():
        partial.unlink()
    if len(frame) != 180 or frame.groupby("ligand_id").pdb_id.nunique().ne(2).any():
        raise CampaignError("confirmation docking coverage is incomplete")
    return frame


def _classifier() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.03,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def _predict_before_score(
    repo: Path,
    discovery_matrix: pd.DataFrame,
    confirmation: pd.DataFrame,
) -> pd.DataFrame:
    discovery = discovery_matrix.loc[discovery_matrix.cohort.eq("exact_ic50")].reset_index(drop=True)
    inner = v132._v9_inner_baselines(repo, discovery)  # noqa: SLF001
    hypotheses = {
        "primary_frozen_f649": list(PRIMARY_FEATURES),
        "secondary_frozen_sensitivity_contacts": [
            column
            for column in discovery
            if column.startswith(("dock__8ZYO__contact_", "dock__8ZYQ__contact_"))
        ],
    }
    predictions = confirmation[
        ["ligand_id", "structure_id", "scaffold_group_id", "outer_fold", "baseline_prediction"]
    ].copy()
    baseline_tier = v13._tier_index(predictions.baseline_prediction.to_numpy(float))  # noqa: SLF001
    for class_index, class_name in enumerate(v13.CLASS_NAMES):
        predictions[f"baseline_probability_{class_name.lower()}"] = (baseline_tier == class_index).astype(float)
    for name, columns in hypotheses.items():
        probabilities = np.full((len(confirmation), 3), np.nan)
        for outer in range(5):
            fit = discovery.outer_fold.ne(outer).to_numpy()
            evaluate = confirmation.outer_fold.eq(outer).to_numpy()
            baseline_map = inner.loc[inner.context_outer_fold.eq(outer)].set_index("structure_id").inner_baseline
            training = discovery.loc[fit, columns].copy()
            training.insert(
                0,
                "strict_inner_baseline",
                discovery.loc[fit, "structure_id"].map(baseline_map).to_numpy(float),
            )
            validation = confirmation.loc[evaluate, columns].copy()
            validation.insert(
                0,
                "strict_inner_baseline",
                confirmation.loc[evaluate, "baseline_prediction"].to_numpy(float),
            )
            feature_columns = ["strict_inner_baseline", *columns]
            model = _classifier()
            model.fit(
                training[feature_columns],
                v13._tier_index(discovery.loc[fit, "observed_target"].to_numpy(float)),  # noqa: SLF001
            )
            fold_probabilities = model.predict_proba(validation[feature_columns])
            rows = np.flatnonzero(evaluate)
            classes = model.named_steps["model"].classes_.astype(int)
            probabilities[np.ix_(rows, classes)] = fold_probabilities
        if not np.isfinite(probabilities).all():
            raise CampaignError(f"incomplete pre-score predictions for {name}")
        for class_index, class_name in enumerate(v13.CLASS_NAMES):
            predictions[f"{name}__probability_{class_name.lower()}"] = probabilities[:, class_index]
    return predictions


def _score(
    confirmation: pd.DataFrame,
    predictions: pd.DataFrame,
    bootstrap: int,
) -> dict[str, Any]:
    y = v13._tier_index(confirmation.observed_target.to_numpy(float))  # noqa: SLF001
    baseline = predictions[
        [f"baseline_probability_{name.lower()}" for name in v13.CLASS_NAMES]
    ].to_numpy(float)
    report: dict[str, Any] = {
        "baseline_v9_threshold": _metrics(y, baseline),
        "hypotheses": {},
    }
    for name in ("primary_frozen_f649", "secondary_frozen_sensitivity_contacts"):
        probabilities = predictions[
            [f"{name}__probability_{class_name.lower()}" for class_name in v13.CLASS_NAMES]
        ].to_numpy(float)
        fold_metrics = []
        for fold in range(5):
            keep = confirmation.outer_fold.eq(fold).to_numpy()
            fold_metrics.append(
                {
                    "outer_fold": fold,
                    "baseline_balanced_accuracy": _metrics(y[keep], baseline[keep])["balanced_accuracy"],
                    "candidate_balanced_accuracy": _metrics(y[keep], probabilities[keep])["balanced_accuracy"],
                }
            )
        report["hypotheses"][name] = {
            **_metrics(y, probabilities),
            "vs_baseline_bootstrap": _balanced_bootstrap(y, baseline, probabilities, bootstrap),
            "fold_metrics": fold_metrics,
            "folds_better": int(
                sum(
                    row["candidate_balanced_accuracy"] > row["baseline_balanced_accuracy"]
                    for row in fold_metrics
                )
            ),
        }
    primary = report["hypotheses"]["primary_frozen_f649"]
    baseline_metrics = report["baseline_v9_threshold"]
    report["confirmation_decision"] = {
        "confirmed": bool(
            primary["vs_baseline_bootstrap"]["ci95"][0] > 0
            and primary["folds_better"] >= 4
            and primary["macro_f1"] > baseline_metrics["macro_f1"]
            and primary["potent_predicted_safe_rate"] <= baseline_metrics["potent_predicted_safe_rate"] + 0.05
        ),
        "rule": (
            "lower 95% balanced-accuracy delta >0; better in >=4/5 folds; macro-F1 improves; "
            "potent-to-safe rate increases by <=0.05"
        ),
    }
    return report


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
        "--output-root",
        type=Path,
        default=Path("research/local_runs/herg_receptor_classification_confirmation_v13_3"),
    )
    parser.add_argument("--vina", type=Path)
    parser.add_argument("--cpu", type=int, default=6)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--modes", type=int, default=9)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser


def _main(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    primary = args.primary_root if args.primary_root.is_absolute() else repo / args.primary_root
    discovery_root = (
        args.discovery_root if args.discovery_root.is_absolute() else repo / args.discovery_root
    )
    output = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    primary, discovery_root, output = primary.resolve(), discovery_root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = _select_confirmation(repo, primary, output)
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
    features = v13._aggregate_docking(docking, STATES)  # noqa: SLF001
    confirmation = selected.merge(features, on="ligand_id", validate="one_to_one")
    discovery = pd.read_parquet(discovery_root / "six_state_analysis_matrix.parquet")
    predictions = _predict_before_score(repo, discovery, confirmation)
    prediction_path = output / "predictions/predictions_before_score.parquet"
    _parquet(prediction_path, predictions)
    prediction_seal = _json(
        output / "predictions/prediction_seal.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "prediction_path": str(prediction_path.resolve()),
            "prediction_bytes": prediction_path.stat().st_size,
            "prediction_sha256": _sha(prediction_path),
            "rows": len(predictions),
            "primary_features": list(PRIMARY_FEATURES),
            "model_C": 0.03,
            "decision_rule": "argmax",
            "labels_present_in_prediction_artifact": False,
        },
        "seal_sha256",
    )
    score = _score(confirmation, predictions, args.bootstrap_replicates)
    report = _json(
        output / "analysis_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete_scaffold_disjoint_internal_confirmation",
            "selection": {
                "n": len(confirmation),
                "unique_scaffolds": int(confirmation.scaffold_group_id.nunique()),
                "discovery_scaffold_overlap": 0,
                "balanced_by_observed_tier": True,
            },
            "prediction_seal_sha256": prediction_seal["seal_sha256"],
            "score": score,
            "scientific_scope": {
                "primary_feature_frozen_before_confirmation_docking": True,
                "confirmation_labels_used_for_model_tuning": False,
                "confirmation_labels_used_for_balanced_sampling": True,
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
                "external_or_prospective_validation": False,
                "vina_scores_are_binding_free_energies": False,
            },
        },
        "report_sha256",
    )
    inputs = [
        repo / "pipeline/scripts/run_local_herg_receptor_classification_confirmation_v13_3.py",
        primary / "manifest.json",
        discovery_root / "manifest.json",
        repo / "research/local_runs/herg_domain_mixture_campaign_v9/manifest.json",
    ]
    artifacts = [
        output / "selection/confirmation_panel.parquet",
        output / "selection/selection_report.json",
        output / "docking/docking_results.parquet",
        prediction_path,
        output / "predictions/prediction_seal.json",
        output / "analysis_report.json",
    ]
    manifest = _json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete",
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
    return _json(
        output / "final_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "finished_utc": _utc(),
            "status": "complete",
            "confirmation_n": len(confirmation),
            "confirmed": report["score"]["confirmation_decision"]["confirmed"],
            "report_sha256": report["report_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "summary_sha256",
    )


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _main(args)
    except (CampaignError, v13.CampaignError, v132.CampaignError) as exc:
        print(f"V13.3 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
