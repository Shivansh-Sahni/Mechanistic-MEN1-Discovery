#!/usr/bin/env python3
"""Stress-test nonlinear receptor fusion across the six frozen V13 campaigns.

This exploratory V14.1 campaign reuses the structure-disjoint V14 matrix and
the exact same whole-campaign validation contract.  It asks whether shallow,
regularized ExtraTrees interactions can recover receptor value that the V14
linear fusion may have missed.  Every outer campaign remains untouched while
tree regularization is selected by leave-one-campaign-out validation among the
other five campaigns.  All labels were already open before V14.1, so even a
positive result is research evidence rather than prospective confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import run_local_herg_cross_campaign_receptor_fusion_v14 as v14
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

SCHEMA_VERSION = "platform-local-herg-nonlinear-receptor-stress-v14.1/1.0"
SEED = 20260831
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V14 = REPO_ROOT / "research/local_runs/herg_cross_campaign_receptor_fusion_v14"
DEFAULT_OUTPUT = REPO_ROOT / "research/local_runs/herg_nonlinear_receptor_stress_v14_1"
N_JOBS = 4

TREE_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "leaf6_sqrt",
        "min_samples_leaf": 6,
        "max_features": "sqrt",
        "max_depth": None,
    },
    {
        "name": "leaf12_fraction70",
        "min_samples_leaf": 12,
        "max_features": 0.70,
        "max_depth": None,
    },
    {
        "name": "leaf24_all",
        "min_samples_leaf": 24,
        "max_features": 1.0,
        "max_depth": None,
    },
)


class CampaignError(RuntimeError):
    """Raised when a V14.1 scientific or artifact contract is violated."""


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any], hash_field: str) -> dict[str, Any]:
    body = dict(payload)
    body.pop(hash_field, None)
    body[hash_field] = hashlib.sha256(_canonical(body)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(body))
    temporary.replace(path)
    return body


def _read_json(path: Path, hash_field: str) -> dict[str, Any]:
    body = json.loads(path.read_text())
    if not isinstance(body, dict):
        raise CampaignError(f"expected JSON object: {path}")
    expected = body.get(hash_field)
    candidate = dict(body)
    candidate.pop(hash_field, None)
    actual = hashlib.sha256(_canonical(candidate)).hexdigest()
    if expected != actual:
        raise CampaignError(f"self-hash mismatch: {path}")
    return body


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _tree_surfaces(matrix: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    linear = v14._feature_surfaces(matrix)  # noqa: SLF001
    return {
        "extratrees_ligand_core": linear["ligand_calibrated"],
        "extratrees_ligand_physchem": linear["ligand_calibrated_with_physchem"],
        "extratrees_receptor_frozen": linear["ligand_plus_frozen_receptor"],
        "extratrees_receptor_mechanistic": linear["ligand_plus_mechanistic_pose"],
        "extratrees_receptor_combined": linear["ligand_plus_frozen_and_mechanistic"],
        "extratrees_receptor_all_pose": linear["ligand_plus_all_pose_features"],
    }


def _classifier(config: dict[str, Any], n_estimators: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    criterion="gini",
                    min_samples_leaf=int(config["min_samples_leaf"]),
                    max_features=config["max_features"],
                    max_depth=config["max_depth"],
                    bootstrap=False,
                    n_jobs=N_JOBS,
                    random_state=SEED,
                ),
            ),
        ]
    )


def _regressor(config: dict[str, Any], n_estimators: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    criterion="squared_error",
                    min_samples_leaf=int(config["min_samples_leaf"]),
                    max_features=config["max_features"],
                    max_depth=config["max_depth"],
                    bootstrap=False,
                    n_jobs=N_JOBS,
                    random_state=SEED,
                ),
            ),
        ]
    )


def _fit_classifier(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    config: dict[str, Any],
    n_estimators: int,
) -> Pipeline:
    model = _classifier(config, n_estimators)
    model.fit(
        frame[list(features)],
        frame.target_class,
        model__sample_weight=v14._classification_weights(frame.reset_index(drop=True)),  # noqa: SLF001
    )
    return model


def _fit_regressor(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    config: dict[str, Any],
    n_estimators: int,
) -> Pipeline:
    model = _regressor(config, n_estimators)
    model.fit(
        frame[list(features)],
        frame.target_pic50.to_numpy(float) - frame.baseline_pic50.to_numpy(float),
        model__sample_weight=v14._regression_weights(frame.reset_index(drop=True)),  # noqa: SLF001
    )
    return model


def _inner_predictions(
    training: pd.DataFrame,
    features: tuple[str, ...],
    config: dict[str, Any],
    n_estimators: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for campaign in sorted(training.campaign.unique()):
        fit = training.loc[training.campaign.ne(campaign)].reset_index(drop=True)
        evaluation = training.loc[training.campaign.eq(campaign)].copy()
        classifier = _fit_classifier(fit, features, config, n_estimators)
        probabilities = classifier.predict_proba(evaluation[list(features)])
        if list(classifier.named_steps["model"].classes_.astype(int)) != [0, 1, 2]:
            raise CampaignError("an inner ExtraTrees classifier lost an observed class")
        regressor = _fit_regressor(fit, features, config, n_estimators)
        evaluation["predicted_pic50"] = evaluation.baseline_pic50.to_numpy(float) + regressor.predict(
            evaluation[list(features)]
        )
        for index, name in enumerate(v14.CLASS_NAMES):
            evaluation[f"probability_{name.lower()}"] = probabilities[:, index]
        rows.append(evaluation)
    return pd.concat(rows, ignore_index=True)


def _tune(
    training: pd.DataFrame,
    features: tuple[str, ...],
    n_estimators: int,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    probability_columns = [f"probability_{name.lower()}" for name in v14.CLASS_NAMES]
    rows: list[dict[str, Any]] = []
    for config in TREE_CONFIGS:
        predictions = _inner_predictions(training, features, config, n_estimators)
        class_metrics: list[dict[str, Any]] = []
        regression_metrics: list[dict[str, Any]] = []
        for _campaign, group in predictions.groupby("campaign", sort=True):
            class_metrics.append(
                v14._classification_metrics(  # noqa: SLF001
                    group.target_class.to_numpy(int),
                    group[probability_columns].to_numpy(float),
                )
            )
            regression_metrics.append(
                v14._regression_metrics(group.target_pic50, group.predicted_pic50)  # noqa: SLF001
            )
        rows.append(
            {
                "config": config["name"],
                "min_samples_leaf": config["min_samples_leaf"],
                "max_features": str(config["max_features"]),
                "macro_campaign_balanced_accuracy": float(
                    np.mean([item["balanced_accuracy"] for item in class_metrics])
                ),
                "macro_campaign_macro_f1": float(np.mean([item["macro_f1"] for item in class_metrics])),
                "macro_campaign_log_loss": float(np.mean([item["log_loss"] for item in class_metrics])),
                "macro_campaign_potent_predicted_safe_rate": float(
                    np.mean([item["potent_predicted_safe_rate"] for item in class_metrics])
                ),
                "macro_campaign_mae": float(np.mean([item["mae"] for item in regression_metrics])),
                "macro_campaign_rmse": float(np.mean([item["rmse"] for item in regression_metrics])),
                "macro_campaign_spearman": float(
                    np.mean([item["spearman"] for item in regression_metrics])
                ),
            }
        )
    evidence = pd.DataFrame(rows)
    evidence["classification_safety_gate"] = evidence.macro_campaign_potent_predicted_safe_rate.le(
        0.06
    )
    class_pool = evidence.loc[evidence.classification_safety_gate]
    if not len(class_pool):
        class_pool = evidence
    class_name = str(
        class_pool.sort_values(
            [
                "macro_campaign_balanced_accuracy",
                "macro_campaign_macro_f1",
                "macro_campaign_log_loss",
                "min_samples_leaf",
            ],
            ascending=[False, False, True, False],
        ).iloc[0].config
    )
    regression_name = str(
        evidence.sort_values(
            ["macro_campaign_mae", "macro_campaign_rmse", "macro_campaign_spearman", "min_samples_leaf"],
            ascending=[True, True, False, False],
        ).iloc[0].config
    )
    by_name = {str(config["name"]): config for config in TREE_CONFIGS}
    return by_name[class_name], by_name[regression_name], evidence


def _nested_loco(
    matrix: pd.DataFrame,
    surfaces: dict[str, tuple[str, ...]],
    n_estimators: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[pd.DataFrame] = []
    tuning_rows: list[pd.DataFrame] = []
    campaigns = sorted(matrix.campaign.unique())
    for outer_index, outer_campaign in enumerate(campaigns, start=1):
        training = matrix.loc[matrix.campaign.ne(outer_campaign)].reset_index(drop=True)
        evaluation = matrix.loc[matrix.campaign.eq(outer_campaign)].copy()
        for surface, features in surfaces.items():
            missing = set(features) - set(matrix)
            if missing:
                raise CampaignError(f"{surface} lacks features: {sorted(missing)}")
            class_config, regression_config, evidence = _tune(training, features, n_estimators)
            evidence.insert(0, "surface", surface)
            evidence.insert(0, "outer_campaign", outer_campaign)
            tuning_rows.append(evidence)
            classifier = _fit_classifier(training, features, class_config, n_estimators)
            probabilities = classifier.predict_proba(evaluation[list(features)])
            regressor = _fit_regressor(training, features, regression_config, n_estimators)
            predicted_pic50 = evaluation.baseline_pic50.to_numpy(float) + regressor.predict(
                evaluation[list(features)]
            )
            output = evaluation[
                [
                    "sample_id",
                    "campaign",
                    "campaign_kind",
                    "ligand_id",
                    "cross_campaign_scaffold_key",
                    "target_class",
                    "target_pic50",
                    "baseline_pic50",
                    *v14.ROUTER_COLUMNS,
                ]
            ].copy()
            output["surface"] = surface
            output["selected_class_config"] = class_config["name"]
            output["selected_regression_config"] = regression_config["name"]
            output["predicted_pic50"] = predicted_pic50
            for index, name in enumerate(v14.CLASS_NAMES):
                output[f"probability_{name.lower()}"] = probabilities[:, index]
            prediction_rows.append(output)
            print(
                json.dumps(
                    {
                        "stage": "nested_nonlinear_loco",
                        "completed_outer_campaigns": outer_index,
                        "total_outer_campaigns": len(campaigns),
                        "outer_campaign": outer_campaign,
                        "surface": surface,
                        "classification_config": class_config["name"],
                        "regression_config": regression_config["name"],
                    }
                ),
                flush=True,
            )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    expected = len(matrix) * len(surfaces)
    if len(predictions) != expected:
        raise CampaignError(f"nested nonlinear LOCO produced {len(predictions)} rows, expected {expected}")
    return predictions, pd.concat(tuning_rows, ignore_index=True)


def _selection_and_bootstrap(
    predictions: pd.DataFrame,
    aggregate: dict[str, Any],
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ligand_surfaces = [
        "ligand_calibrated",
        "ligand_calibrated_with_physchem",
        "extratrees_ligand_core",
        "extratrees_ligand_physchem",
    ]
    receptor_surfaces = [
        surface for surface in aggregate if surface.startswith("extratrees_receptor_")
    ]
    best_ligand_class = max(
        ligand_surfaces,
        key=lambda surface: aggregate[surface]["classification"]["macro_by_campaign"][
            "balanced_accuracy"
        ],
    )
    best_ligand_regression = min(
        ligand_surfaces,
        key=lambda surface: aggregate[surface]["regression"]["macro_by_campaign"]["mae"],
    )
    best_receptor_class = max(
        receptor_surfaces,
        key=lambda surface: aggregate[surface]["classification"]["macro_by_campaign"][
            "balanced_accuracy"
        ],
    )
    best_receptor_regression = min(
        receptor_surfaces,
        key=lambda surface: aggregate[surface]["regression"]["macro_by_campaign"]["mae"],
    )
    comparisons: dict[str, Any] = {}
    for surface in [*ligand_surfaces, *receptor_surfaces]:
        if surface.startswith("extratrees_receptor_"):
            comparisons[surface] = {
                "vs_matched_tree_ligand": v14._paired_bootstrap(  # noqa: SLF001
                    predictions,
                    surface,
                    "extratrees_ligand_physchem",
                    bootstrap_replicates,
                ),
                "vs_best_ligand_classification": v14._paired_bootstrap(  # noqa: SLF001
                    predictions,
                    surface,
                    best_ligand_class,
                    bootstrap_replicates,
                ),
                "vs_best_ligand_regression": v14._paired_bootstrap(  # noqa: SLF001
                    predictions,
                    surface,
                    best_ligand_regression,
                    bootstrap_replicates,
                ),
            }
        elif surface.startswith("extratrees_ligand_"):
            comparisons[surface] = {
                "vs_frozen_router": v14._paired_bootstrap(  # noqa: SLF001
                    predictions,
                    surface,
                    "frozen_ligand_router",
                    bootstrap_replicates,
                ),
                "vs_v14_linear_ligand": v14._paired_bootstrap(  # noqa: SLF001
                    predictions,
                    surface,
                    "ligand_calibrated",
                    bootstrap_replicates,
                ),
            }
    class_evidence = comparisons[best_receptor_class]["vs_best_ligand_classification"][
        "classification_delta_balanced_accuracy"
    ]
    regression_evidence = comparisons[best_receptor_regression]["vs_best_ligand_regression"][
        "regression_delta_mae"
    ]
    class_metrics = aggregate[best_receptor_class]["classification"]["macro_by_campaign"]
    ligand_class_metrics = aggregate[best_ligand_class]["classification"]["macro_by_campaign"]
    class_gate = bool(
        class_evidence["ci95"][0] > 0
        and class_evidence["campaigns_better"] >= 4
        and class_metrics["macro_f1"] > ligand_class_metrics["macro_f1"]
        and class_metrics["potent_predicted_safe_rate"] <= 0.05
        and class_metrics["potent_predicted_safe_rate"]
        <= ligand_class_metrics["potent_predicted_safe_rate"] + 0.02
    )
    regression_gate = bool(
        regression_evidence["ci95"][1] < 0 and regression_evidence["campaigns_better"] >= 4
    )
    decision = {
        "best_ligand_classification_surface_posthoc": best_ligand_class,
        "best_ligand_regression_surface_posthoc": best_ligand_regression,
        "best_receptor_classification_surface_posthoc": best_receptor_class,
        "best_receptor_regression_surface_posthoc": best_receptor_regression,
        "receptor_classification_gate_passed": class_gate,
        "receptor_regression_gate_passed": regression_gate,
        "general_receptor_prediction_promotion_supported": bool(class_gate and regression_gate),
        "production_promotion_supported": False,
        "selection_is_posthoc": True,
        "claim_boundary": (
            "all six outcomes were already open and tree families were added after V14; nested outer "
            "predictions measure campaign transport but do not provide prospective confirmation"
        ),
        "gate": (
            "classification requires scaffold-bootstrap lower 95% BA delta >0 versus the best ligand "
            "surface, >=4/6 campaigns better, higher macro-F1, Potent-to-Safe <=0.05 and no more than "
            "0.02 above ligand; regression requires bootstrap upper 95% MAE delta <0 and >=4/6 "
            "campaigns better"
        ),
    }
    return decision, comparisons


def _fit_full_models(
    matrix: pd.DataFrame,
    surfaces: dict[str, tuple[str, ...]],
    output: Path,
    n_estimators: int,
) -> dict[str, Any]:
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    for surface, features in surfaces.items():
        class_config, regression_config, _evidence = _tune(matrix, features, n_estimators)
        classifier = _fit_classifier(matrix, features, class_config, n_estimators)
        regressor = _fit_regressor(matrix, features, regression_config, n_estimators)
        path = model_dir / f"{surface}.joblib"
        joblib.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "surface": surface,
                "features": list(features),
                "class_names": list(v14.CLASS_NAMES),
                "classifier_config": class_config,
                "regression_config": regression_config,
                "classifier": classifier,
                "residual_regressor": regressor,
                "regression_anchor": "baseline_pic50",
                "research_only": True,
            },
            path,
            compress=3,
        )
        result[surface] = {
            "path": str(path.resolve()),
            "sha256": _sha(path),
            "bytes": path.stat().st_size,
            "features": list(features),
            "classifier_config": class_config["name"],
            "regression_config": regression_config["name"],
        }
    return result


def _model_card(
    output: Path,
    aggregate: dict[str, Any],
    decision: dict[str, Any],
    comparisons: dict[str, Any],
) -> None:
    rows = []
    display_order = [
        "frozen_ligand_router",
        "ligand_calibrated",
        "ligand_calibrated_with_physchem",
        "extratrees_ligand_core",
        "extratrees_ligand_physchem",
        "extratrees_receptor_frozen",
        "extratrees_receptor_mechanistic",
        "extratrees_receptor_combined",
        "extratrees_receptor_all_pose",
    ]
    for surface in display_order:
        class_metrics = aggregate[surface]["classification"]["macro_by_campaign"]
        regression_metrics = aggregate[surface]["regression"]["macro_by_campaign"]
        rows.append(
            f"| `{surface}` | {class_metrics['balanced_accuracy']:.4f} | "
            f"{class_metrics['macro_f1']:.4f} | {class_metrics['potent_predicted_safe_rate']:.4f} | "
            f"{regression_metrics['mae']:.4f} |"
        )
    best_class = decision["best_receptor_classification_surface_posthoc"]
    best_regression = decision["best_receptor_regression_surface_posthoc"]
    class_delta = comparisons[best_class]["vs_best_ligand_classification"][
        "classification_delta_balanced_accuracy"
    ]
    regression_delta = comparisons[best_regression]["vs_best_ligand_regression"][
        "regression_delta_mae"
    ]
    content = f"""# hERG V14.1 nonlinear receptor stress-test model card

## Status

Exploratory research stress test. General receptor-aware prediction promotion: **{str(decision['general_receptor_prediction_promotion_supported']).lower()}**. Production promotion: **false**.

## Evaluation contract

- 1,224 labeled and docked structures from six structure-disjoint V13 campaigns.
- Each outer test set is an entire campaign. Tree regularization is selected only by leave-one-campaign-out validation within the other five campaigns.
- Training weights equalize campaigns and observed classes for classification and equalize campaigns for regression.
- The same frozen 8ZYO pose ensembles and corrected T623/S624/S649/Y652/F656 interpretation from V14 are used.
- All outcomes were open before this nonlinear follow-up. This is a robustness stress test, not prospective confirmation.

## Macro-by-campaign benchmark

| Surface | Balanced accuracy | Macro-F1 | Potent→Safe | pIC50 MAE |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Incremental receptor result

- Best receptor classification surface: `{best_class}`.
- Balanced-accuracy delta versus best ligand classifier: **{class_delta['observed']:+.4f}**, scaffold-bootstrap 95% CI **[{class_delta['ci95'][0]:+.4f}, {class_delta['ci95'][1]:+.4f}]**, better in **{class_delta['campaigns_better']}/6** campaigns.
- Best receptor regression surface: `{best_regression}`.
- MAE delta versus best ligand regressor: **{regression_delta['observed']:+.4f} pIC50**, scaffold-bootstrap 95% CI **[{regression_delta['ci95'][0]:+.4f}, {regression_delta['ci95'][1]:+.4f}]**, better in **{regression_delta['campaigns_better']}/6** campaigns.

## Decision boundary

The nonlinear family tests whether receptor/ligand interactions were missed by V14 ridge models. A tree-family gain is not automatically a receptor gain: receptor surfaces must beat both an equal-capacity ligand-only ExtraTrees comparator and the strongest ligand-only surface overall. Post-hoc surface selection and already-open labels keep every V14.1 model research-only. The locked 24-compound assay panel remains the proper prospective discriminator.
"""
    (output / "MODEL_CARD.md").write_text(content)


def _release_artifacts(output: Path, models: dict[str, Any]) -> list[Path]:
    return [
        output / "predictions/nested_loco_predictions.parquet",
        output / "evidence/nested_tuning_evidence.parquet",
        output / "evidence/classification_metrics_by_campaign.parquet",
        output / "evidence/regression_metrics_by_campaign.parquet",
        output / "evidence/aggregate_metrics.json",
        output / "evidence/paired_scaffold_bootstrap.json",
        output / "MODEL_CARD.md",
        output / "analysis_report.json",
        *[Path(item["path"]) for item in models.values()],
    ]


def _write_release_seal(
    output: Path,
    v14_root: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    models = report["models"]
    decision = report["decision"]
    artifacts = _release_artifacts(output, models)
    inputs = [
        Path(v14.__file__).resolve(),
        v14_root / "analysis_report.json",
        v14_root / "predictions/nested_loco_predictions.parquet",
        v14_root / "data/harmonized_campaigns.parquet",
        v14_root / "data/pose_ensemble_features.parquet",
    ]
    for path in [*artifacts, *inputs]:
        if not path.is_file():
            raise CampaignError(f"release dependency is missing: {path}")
    manifest = _write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete",
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha(Path(__file__).resolve()),
            },
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
    return _write_json(
        output / "final_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "finished_utc": _utc(),
            "status": "complete",
            "best_ligand_classification_surface_posthoc": decision[
                "best_ligand_classification_surface_posthoc"
            ],
            "best_ligand_regression_surface_posthoc": decision[
                "best_ligand_regression_surface_posthoc"
            ],
            "best_receptor_classification_surface_posthoc": decision[
                "best_receptor_classification_surface_posthoc"
            ],
            "best_receptor_regression_surface_posthoc": decision[
                "best_receptor_regression_surface_posthoc"
            ],
            "general_receptor_prediction_promotion_supported": decision[
                "general_receptor_prediction_promotion_supported"
            ],
            "report_sha256": report["report_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "summary_sha256",
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    v14_root = args.v14_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    v14_report = _read_json(v14_root / "analysis_report.json", "report_sha256")
    if v14_report["dataset"]["campaigns"] != 6 or v14_report["dataset"]["rows"] != 1_224:
        raise CampaignError("the frozen V14 campaign census changed")
    if v14_report["promotion_decision"]["general_receptor_prediction_promotion_supported"]:
        raise CampaignError("V14 receptor policy changed; review V14.1 assumptions")
    frame = pd.read_parquet(v14_root / "data/harmonized_campaigns.parquet")
    pose = pd.read_parquet(v14_root / "data/pose_ensemble_features.parquet")
    matrix = v14._build_model_matrix(frame, pose)  # noqa: SLF001
    surfaces = _tree_surfaces(matrix)
    tree_predictions, tuning = _nested_loco(matrix, surfaces, args.n_estimators)
    v14_predictions = pd.read_parquet(v14_root / "predictions/nested_loco_predictions.parquet")
    combined = pd.concat([v14_predictions, tree_predictions], ignore_index=True, sort=False)
    class_metrics, regression_metrics, aggregate = v14._score_predictions(combined)  # noqa: SLF001
    decision, comparisons = _selection_and_bootstrap(
        combined,
        aggregate,
        args.bootstrap_replicates,
    )
    models = _fit_full_models(matrix, surfaces, output, args.n_estimators)
    _model_card(output, aggregate, decision, comparisons)
    paths = {
        "predictions": output / "predictions/nested_loco_predictions.parquet",
        "tuning": output / "evidence/nested_tuning_evidence.parquet",
        "classification_metrics": output / "evidence/classification_metrics_by_campaign.parquet",
        "regression_metrics": output / "evidence/regression_metrics_by_campaign.parquet",
    }
    _parquet(paths["predictions"], tree_predictions)
    _parquet(paths["tuning"], tuning)
    _parquet(paths["classification_metrics"], class_metrics)
    _parquet(paths["regression_metrics"], regression_metrics)
    aggregate_report = _write_json(
        output / "evidence/aggregate_metrics.json",
        {"schema_version": SCHEMA_VERSION, "created_utc": _utc(), "metrics": aggregate},
        "report_sha256",
    )
    bootstrap_report = _write_json(
        output / "evidence/paired_scaffold_bootstrap.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "replicates": args.bootstrap_replicates,
            "comparisons": comparisons,
        },
        "report_sha256",
    )
    report = _write_json(
        output / "analysis_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete_exploratory_nonlinear_campaign_stress_test",
            "dataset": v14_report["dataset"],
            "evaluation_contract": {
                "outer_split": "leave one entire campaign out",
                "inner_selection": "leave one campaign out within the five outer-training campaigns",
                "cross_campaign_exact_overlap": 0,
                "cross_campaign_scaffold_overlap": 0,
                "tree_family": "ExtraTrees with three prespecified regularization configurations",
                "inner_estimators_per_fit": args.n_estimators,
                "all_labels_open_before_analysis": True,
            },
            "decision": decision,
            "models": models,
            "aggregate_metrics_report_sha256": aggregate_report["report_sha256"],
            "bootstrap_report_sha256": bootstrap_report["report_sha256"],
            "v14_input_report_sha256": v14_report["report_sha256"],
            "scientific_scope": {
                "receptor_aware": True,
                "new_docking_performed": False,
                "prospective_confirmation": False,
                "website_default_changed": False,
                "clinical_or_regulatory_use": False,
            },
        },
        "report_sha256",
    )
    return _write_release_seal(output, v14_root, report)


def _repackage(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.resolve()
    v14_root = args.v14_root.resolve()
    report = _read_json(output / "analysis_report.json", "report_sha256")
    aggregate = _read_json(output / "evidence/aggregate_metrics.json", "report_sha256")
    bootstrap = _read_json(output / "evidence/paired_scaffold_bootstrap.json", "report_sha256")
    if report["aggregate_metrics_report_sha256"] != aggregate["report_sha256"]:
        raise CampaignError("aggregate metrics do not match the analysis report")
    if report["bootstrap_report_sha256"] != bootstrap["report_sha256"]:
        raise CampaignError("bootstrap evidence does not match the analysis report")
    for item in report["models"].values():
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(output) or _sha(path) != item["sha256"]:
            raise CampaignError(f"model bundle failed integrity validation: {path}")
    return _write_release_seal(output, v14_root, report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v14-root", type=Path, default=DEFAULT_V14)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-estimators", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--stage", choices=("benchmark", "repackage"), default="benchmark")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.n_estimators < 64:
        print("V14.1 ERROR: at least 64 trees are required", file=sys.stderr)
        return 2
    if args.bootstrap_replicates < 100:
        print("V14.1 ERROR: at least 100 bootstrap replicates are required", file=sys.stderr)
        return 2
    try:
        result = _repackage(args) if args.stage == "repackage" else _run(args)
    except (CampaignError, v14.CampaignError) as exc:
        print(f"V14.1 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
