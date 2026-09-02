#!/usr/bin/env python3
"""Run the governed retrospective hERG V15 functional-group feature study.

The study uses six already-open V13/V14 campaigns. Every outer test is an
entire campaign, while hyperparameters are selected by leave-one-campaign-out
validation inside the other five campaigns. Cross-campaign exact and scaffold
overlap must remain zero. No prospective, blinded, or final-test claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import herg_v15_functional_group_features as fg
import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

SCHEMA_VERSION = "platform-local-herg-functional-group-finalize-v15/1.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = (
    REPO_ROOT
    / "research/local_runs/herg_cross_campaign_receptor_fusion_v14/data/harmonized_campaigns.parquet"
)
DEFAULT_V141_PREDICTIONS = (
    REPO_ROOT
    / "research/local_runs/herg_nonlinear_receptor_stress_v14_1/predictions/nested_loco_predictions.parquet"
)
DEFAULT_OUTPUT = REPO_ROOT / "research/local_runs/herg_functional_group_finalize_v15"
SEED = 20260831
RIDGE_ALPHAS = (1.0, 10.0, 100.0)
BOOTSTRAP_REPLICATES = 2_000
PAIR_MINIMUM_TANIMOTO = 0.70
PAIR_MAXIMUM_PER_SCAFFOLD = 10
PAIR_OBSERVED_DIRECTION_MINIMUM_PIC50 = 0.15
PAIR_PREDICTED_TIE_PIC50 = 0.05
SAFE_PIC50 = 6.0 - math.log10(30.0)
POTENT_PIC50 = 6.0
PRIMARY_CANDIDATE = "fg_interaction_absolute_ridge"
FROZEN_COMPARATOR = "frozen_v14_1_ligand_core"


class StudyError(RuntimeError):
    """Raised when the retrospective study contract is violated."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _self_hashed_json(path: Path, value: dict[str, Any], key: str = "report_sha256") -> dict[str, Any]:
    payload = dict(value)
    payload.pop(key, None)
    canonical = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    payload[key] = hashlib.sha256(canonical).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return payload


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _tier_index(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values < SAFE_PIC50, 0, np.where(values <= POTENT_PIC50, 1, 2)).astype(int)


def _campaign_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.campaign.value_counts().to_dict()
    weights = frame.campaign.map({key: 1.0 / value for key, value in counts.items()}).to_numpy(float)
    return weights * (len(weights) / weights.sum())


def _fit_predict_ridge(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: list[str],
    alpha: float,
    *,
    residual: bool,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(training[columns].to_numpy(float))
    evaluation_x = scaler.transform(evaluation[columns].to_numpy(float))
    target = training.target_pic50.to_numpy(float)
    if residual:
        target = target - training.baseline_pic50.to_numpy(float)
    model = Ridge(alpha=float(alpha), random_state=SEED)
    model.fit(train_x, target, sample_weight=_campaign_weights(training))
    prediction = model.predict(evaluation_x)
    if residual:
        prediction = evaluation.baseline_pic50.to_numpy(float) + prediction
    return np.asarray(prediction, dtype=float), np.asarray(model.coef_, dtype=float)


def _surface_contracts(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    physicochemical = [
        *fg.PHYSICOCHEMICAL_FEATURES,
        *fg.CHARGE_CONTEXT_FEATURES,
        *fg.RING_LINKER_FEATURES,
    ]
    functional = fg.feature_columns(
        registry, include_functional_groups=True, include_interactions=False
    )
    full = fg.feature_columns(registry, include_functional_groups=True, include_interactions=True)
    return {
        "physchem_absolute_ridge": {
            "columns": ["baseline_pic50", *physicochemical],
            "target": "absolute_pic50",
            "role": "descriptor-only ablation control",
        },
        "fg_absolute_ridge": {
            "columns": ["baseline_pic50", *functional],
            "target": "absolute_pic50",
            "role": "functional-group count/presence and charge-context candidate",
        },
        "fg_interaction_absolute_ridge": {
            "columns": ["baseline_pic50", *full],
            "target": "absolute_pic50",
            "role": "prespecified primary functional-group candidate with controlled interactions",
        },
        "fg_global_residual_ridge": {
            "columns": ["baseline_pic50", *full],
            "target": "global_residual_from_frozen_baseline",
            "role": "global residual comparator; not a local or same-series correction",
        },
    }


def _audit_matrix(matrix: pd.DataFrame) -> dict[str, Any]:
    required = {
        "sample_id",
        "campaign",
        "campaign_kind",
        "standardized_smiles",
        "target_pic50",
        "baseline_pic50",
        "cross_campaign_connectivity_key",
        "cross_campaign_scaffold_key",
    }
    missing = required - set(matrix)
    if missing:
        raise StudyError("matrix is missing required columns: " + ", ".join(sorted(missing)))
    if len(matrix) != 1_224 or matrix.campaign.nunique() != 6:
        raise StudyError("the frozen V14 six-campaign census changed")
    if matrix.sample_id.duplicated().any():
        raise StudyError("sample IDs are duplicated")
    exact_overlap = 0
    scaffold_overlap = 0
    campaigns = sorted(matrix.campaign.unique())
    for left_index, left_name in enumerate(campaigns):
        left = matrix.loc[matrix.campaign.eq(left_name)]
        for right_name in campaigns[left_index + 1 :]:
            right = matrix.loc[matrix.campaign.eq(right_name)]
            exact_overlap += len(
                set(left.cross_campaign_connectivity_key)
                & set(right.cross_campaign_connectivity_key)
            )
            scaffold_overlap += len(
                set(left.cross_campaign_scaffold_key) & set(right.cross_campaign_scaffold_key)
            )
    if exact_overlap or scaffold_overlap:
        raise StudyError("cross-campaign exact or scaffold overlap violates the outer-test contract")
    return {
        "rows": int(len(matrix)),
        "campaigns": int(len(campaigns)),
        "unique_connectivity_keys": int(matrix.cross_campaign_connectivity_key.nunique()),
        "unique_scaffolds": int(matrix.cross_campaign_scaffold_key.nunique()),
        "cross_campaign_exact_overlap": exact_overlap,
        "cross_campaign_scaffold_overlap": scaffold_overlap,
    }


def _load_matrix(matrix_path: Path, v141_predictions: Path) -> pd.DataFrame:
    matrix = pd.read_parquet(matrix_path).copy()
    audit = _audit_matrix(matrix)
    previous = pd.read_parquet(v141_predictions)
    previous = previous.loc[previous.surface.eq("extratrees_ligand_core"), ["sample_id", "predicted_pic50"]]
    if previous.sample_id.duplicated().any() or len(previous) != len(matrix):
        raise StudyError("the frozen V14.1 ligand comparator has an unexpected census")
    previous = previous.rename(columns={"predicted_pic50": "frozen_v14_1_pic50"})
    matrix = matrix.merge(previous, on="sample_id", validate="one_to_one")
    registry = fg.load_registry()
    feature_frame = fg.build_feature_frame(matrix.standardized_smiles, registry)
    feature_frame.insert(0, "sample_id", matrix.sample_id.to_numpy())
    matrix = matrix.merge(feature_frame, on="sample_id", validate="one_to_one")
    if len(matrix) != audit["rows"]:
        raise StudyError("functional-group feature merge changed the row census")
    return matrix


def _inner_select_alpha(
    training: pd.DataFrame,
    columns: list[str],
    *,
    residual: bool,
) -> tuple[float, list[dict[str, Any]]]:
    evidence = []
    for alpha in RIDGE_ALPHAS:
        campaign_mae = []
        for campaign in sorted(training.campaign.unique()):
            fit = training.loc[training.campaign.ne(campaign)].reset_index(drop=True)
            evaluate = training.loc[training.campaign.eq(campaign)].reset_index(drop=True)
            prediction, _ = _fit_predict_ridge(
                fit, evaluate, columns, alpha, residual=residual
            )
            campaign_mae.append(
                float(np.mean(np.abs(evaluate.target_pic50.to_numpy(float) - prediction)))
            )
        evidence.append(
            {
                "alpha": float(alpha),
                "inner_macro_campaign_mae": float(np.mean(campaign_mae)),
            }
        )
    selected = min(evidence, key=lambda row: (row["inner_macro_campaign_mae"], row["alpha"]))
    return float(selected["alpha"]), evidence


def _nested_predictions(
    matrix: pd.DataFrame,
    contracts: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    selection_rows = []
    coefficient_rows = []
    identity = [
        "sample_id",
        "campaign",
        "campaign_kind",
        "cross_campaign_scaffold_key",
        "target_pic50",
        "baseline_pic50",
    ]
    for outer_campaign in sorted(matrix.campaign.unique()):
        training = matrix.loc[matrix.campaign.ne(outer_campaign)].reset_index(drop=True)
        evaluation = matrix.loc[matrix.campaign.eq(outer_campaign)].reset_index(drop=True)
        if set(training.cross_campaign_scaffold_key) & set(evaluation.cross_campaign_scaffold_key):
            raise StudyError("an outer campaign shares a scaffold with model training")
        for surface, contract in contracts.items():
            residual = contract["target"] == "global_residual_from_frozen_baseline"
            alpha, evidence = _inner_select_alpha(
                training, contract["columns"], residual=residual
            )
            for row in evidence:
                selection_rows.append(
                    {
                        "outer_campaign": outer_campaign,
                        "surface": surface,
                        **row,
                        "selected": bool(row["alpha"] == alpha),
                    }
                )
            prediction, coefficients = _fit_predict_ridge(
                training,
                evaluation,
                contract["columns"],
                alpha,
                residual=residual,
            )
            surface_frame = evaluation[identity].copy()
            surface_frame["surface"] = surface
            surface_frame["selected_alpha"] = alpha
            surface_frame["predicted_pic50"] = prediction
            predictions.append(surface_frame)
            for feature, coefficient in zip(contract["columns"], coefficients, strict=True):
                coefficient_rows.append(
                    {
                        "outer_campaign": outer_campaign,
                        "surface": surface,
                        "feature": feature,
                        "standardized_coefficient": float(coefficient),
                    }
                )
    frozen_rows = []
    for surface, column in (
        ("frozen_v9_v10_baseline", "baseline_pic50"),
        (FROZEN_COMPARATOR, "frozen_v14_1_pic50"),
    ):
        frame = matrix[identity].copy()
        frame["surface"] = surface
        frame["selected_alpha"] = np.nan
        frame["predicted_pic50"] = matrix[column].to_numpy(float)
        frozen_rows.append(frame)
    final = pd.concat([*frozen_rows, *predictions], ignore_index=True)
    if final.groupby("surface").size().nunique() != 1:
        raise StudyError("nested surfaces do not have identical prediction coverage")
    return final, pd.DataFrame(selection_rows), pd.DataFrame(coefficient_rows)


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = np.asarray(target, dtype=float) - np.asarray(prediction, dtype=float)
    rho = spearmanr(target, prediction).statistic
    return {
        "n": int(len(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": float(rho) if np.isfinite(rho) else None,
        "within_0p5": float(np.mean(np.abs(error) <= 0.5)),
    }


def _classification_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    observed = _tier_index(target)
    predicted = _tier_index(prediction)
    potent = observed == 2
    return {
        "n": int(len(observed)),
        "balanced_accuracy": float(balanced_accuracy_score(observed, predicted)),
        "macro_f1": float(f1_score(observed, predicted, average="macro")),
        "potent_predicted_safe_rate": float(np.mean(predicted[potent] == 0)) if potent.any() else None,
    }


def _metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    aggregate: dict[str, Any] = {}
    for surface, surface_frame in predictions.groupby("surface", sort=True):
        campaign_rows = []
        for campaign, group in surface_frame.groupby("campaign", sort=True):
            regression = _regression_metrics(group.target_pic50, group.predicted_pic50)
            classification = _classification_metrics(group.target_pic50, group.predicted_pic50)
            row = {"surface": surface, "campaign": campaign, **regression}
            row.update({f"classification_{key}": value for key, value in classification.items() if key != "n"})
            rows.append(row)
            campaign_rows.append(row)
        aggregate[surface] = {
            "regression": {
                "macro_campaign_mae": float(np.mean([row["mae"] for row in campaign_rows])),
                "macro_campaign_rmse": float(np.mean([row["rmse"] for row in campaign_rows])),
                "macro_campaign_spearman": float(
                    np.mean([row["spearman"] for row in campaign_rows if row["spearman"] is not None])
                ),
                "overall": _regression_metrics(
                    surface_frame.target_pic50, surface_frame.predicted_pic50
                ),
            },
            "classification": {
                "macro_campaign_balanced_accuracy": float(
                    np.mean([row["classification_balanced_accuracy"] for row in campaign_rows])
                ),
                "macro_campaign_macro_f1": float(
                    np.mean([row["classification_macro_f1"] for row in campaign_rows])
                ),
                "macro_campaign_potent_predicted_safe_rate": float(
                    np.mean(
                        [
                            row["classification_potent_predicted_safe_rate"]
                            for row in campaign_rows
                            if row["classification_potent_predicted_safe_rate"] is not None
                        ]
                    )
                ),
            },
        }
    return pd.DataFrame(rows), aggregate


def _cluster_bootstrap_delta(
    predictions: pd.DataFrame,
    candidate: str,
    comparator: str,
) -> dict[str, Any]:
    wide = predictions.pivot(
        index=["sample_id", "campaign", "cross_campaign_scaffold_key", "target_pic50"],
        columns="surface",
        values="predicted_pic50",
    ).reset_index()
    wide["absolute_error_delta"] = np.abs(wide.target_pic50 - wide[candidate]) - np.abs(
        wide.target_pic50 - wide[comparator]
    )
    rng = np.random.default_rng(SEED)
    campaign_clusters = []
    observed_by_campaign = []
    for _, campaign in wide.groupby("campaign", sort=True):
        clusters = (
            campaign.groupby("cross_campaign_scaffold_key", sort=True)
            .absolute_error_delta.agg(["sum", "size"])
            .reset_index(drop=True)
        )
        campaign_clusters.append(
            (clusters["sum"].to_numpy(float), clusters["size"].to_numpy(float))
        )
        observed_by_campaign.append(float(campaign.absolute_error_delta.mean()))
    observed = float(np.mean(observed_by_campaign))
    campaigns_better = int(np.sum(np.asarray(observed_by_campaign) < 0))
    replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate_index in range(BOOTSTRAP_REPLICATES):
        campaign_deltas = []
        for sums, sizes in campaign_clusters:
            selection = rng.integers(0, len(sums), size=len(sums))
            campaign_deltas.append(float(sums[selection].sum() / sizes[selection].sum()))
        replicates[replicate_index] = float(np.mean(campaign_deltas))
    return {
        "candidate": candidate,
        "comparator": comparator,
        "metric": "macro-campaign MAE delta, candidate minus comparator",
        "observed": observed,
        "ci95": [float(np.quantile(replicates, 0.025)), float(np.quantile(replicates, 0.975))],
        "campaigns_better": int(campaigns_better),
        "campaigns_total": int(wide.campaign.nunique()),
        "replicates": BOOTSTRAP_REPLICATES,
        "resampling_unit": "cross-campaign scaffold within campaign",
    }


def _pair_bootstrap_comparison(
    pair_predictions: pd.DataFrame,
    candidate: str,
    comparator: str,
) -> dict[str, Any]:
    if pair_predictions.empty:
        return {"status": "unsupported_no_pairs"}
    keys = [
        "campaign",
        "scaffold",
        "left_sample_id",
        "right_sample_id",
        "observed_direction_evaluable",
    ]
    values = ["absolute_delta_error_pic50", "direction_correct"]
    left = pair_predictions.loc[pair_predictions.surface.eq(candidate), [*keys, *values]]
    right = pair_predictions.loc[pair_predictions.surface.eq(comparator), [*keys, *values]]
    wide = left.merge(
        right,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("__candidate", "__comparator"),
    )
    if len(wide) != len(left) or len(wide) != len(right):
        raise StudyError("pairwise candidate and comparator coverage differs")
    wide["delta_error_difference"] = (
        wide.absolute_delta_error_pic50__candidate
        - wide.absolute_delta_error_pic50__comparator
    )
    wide["direction_correct_difference"] = (
        wide.direction_correct__candidate.astype(int)
        - wide.direction_correct__comparator.astype(int)
    )
    rng = np.random.default_rng(SEED + 1)
    observed_delta_mae = []
    observed_direction = []
    campaign_clusters = []
    for _, campaign in wide.groupby("campaign", sort=True):
        observed_delta_mae.append(float(campaign.delta_error_difference.mean()))
        evaluable = campaign.loc[campaign.observed_direction_evaluable]
        if len(evaluable):
            observed_direction.append(float(evaluable.direction_correct_difference.mean()))
        cluster_rows = []
        for _, cluster in campaign.groupby("scaffold", sort=True):
            cluster_evaluable = cluster.loc[cluster.observed_direction_evaluable]
            cluster_rows.append(
                (
                    float(cluster.delta_error_difference.sum()),
                    int(len(cluster)),
                    float(cluster_evaluable.direction_correct_difference.sum()),
                    int(len(cluster_evaluable)),
                )
            )
        campaign_clusters.append(np.asarray(cluster_rows, dtype=float))
    delta_replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    direction_replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate_index in range(BOOTSTRAP_REPLICATES):
        delta_by_campaign = []
        direction_by_campaign = []
        for clusters in campaign_clusters:
            selection = rng.integers(0, len(clusters), size=len(clusters))
            selected = clusters[selection]
            delta_by_campaign.append(float(selected[:, 0].sum() / selected[:, 1].sum()))
            evaluable_n = selected[:, 3].sum()
            if evaluable_n:
                direction_by_campaign.append(float(selected[:, 2].sum() / evaluable_n))
        delta_replicates[replicate_index] = float(np.mean(delta_by_campaign))
        direction_replicates[replicate_index] = (
            float(np.mean(direction_by_campaign)) if direction_by_campaign else np.nan
        )
    direction_comparison = None
    finite_direction = direction_replicates[np.isfinite(direction_replicates)]
    if observed_direction and len(finite_direction):
        direction_comparison = {
            "observed_macro_campaign": float(np.mean(observed_direction)),
            "ci95": [
                float(np.quantile(finite_direction, 0.025)),
                float(np.quantile(finite_direction, 0.975)),
            ],
        }
    return {
        "status": "limited_descriptive_held_campaign_pair_comparison",
        "candidate": candidate,
        "comparator": comparator,
        "campaigns": int(wide.campaign.nunique()),
        "scaffolds": int(wide[["campaign", "scaffold"]].drop_duplicates().shape[0]),
        "pairs": int(len(wide)),
        "delta_mae_candidate_minus_comparator": {
            "observed_macro_campaign": float(np.mean(observed_delta_mae)),
            "ci95": [
                float(np.quantile(delta_replicates, 0.025)),
                float(np.quantile(delta_replicates, 0.975)),
            ],
        },
        "direction_accuracy_candidate_minus_comparator_including_ties": direction_comparison,
        "replicates": BOOTSTRAP_REPLICATES,
        "resampling_unit": "same-scaffold cluster within each represented campaign",
        "limitation": (
            "pairs occur in only two external campaigns; this evidence cannot promote a pairwise model"
        ),
    }


def _build_pairs(matrix: pd.DataFrame) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2_048, includeChirality=True
    )
    rows = []
    for (campaign, scaffold), group in matrix.groupby(
        ["campaign", "cross_campaign_scaffold_key"], sort=True
    ):
        if len(group) < 2:
            continue
        group = group.sort_values("sample_id").reset_index(drop=True)
        molecules = [Chem.MolFromSmiles(value) for value in group.standardized_smiles]
        fingerprints = [generator.GetFingerprint(molecule) for molecule in molecules]
        candidates = []
        for left, right in itertools.combinations(range(len(group)), 2):
            similarity = float(
                DataStructs.TanimotoSimilarity(fingerprints[left], fingerprints[right])
            )
            if similarity < PAIR_MINIMUM_TANIMOTO:
                continue
            left_row = group.iloc[left]
            right_row = group.iloc[right]
            candidates.append(
                {
                    "campaign": campaign,
                    "scaffold": scaffold,
                    "left_sample_id": left_row.sample_id,
                    "right_sample_id": right_row.sample_id,
                    "morgan_tanimoto": similarity,
                    "observed_delta_pic50": float(
                        right_row.target_pic50 - left_row.target_pic50
                    ),
                }
            )
        candidates.sort(
            key=lambda row: (
                -row["morgan_tanimoto"],
                row["left_sample_id"],
                row["right_sample_id"],
            )
        )
        rows.extend(candidates[:PAIR_MAXIMUM_PER_SCAFFOLD])
    return pd.DataFrame(rows)


def _pair_evidence(
    matrix: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pairs = _build_pairs(matrix)
    if pairs.empty:
        return pairs, {"status": "unsupported_no_pairs"}
    prediction_map = predictions.set_index(["surface", "sample_id"]).predicted_pic50
    rows = []
    for pair in pairs.itertuples(index=False):
        for surface in sorted(predictions.surface.unique()):
            left = float(prediction_map.loc[(surface, pair.left_sample_id)])
            right = float(prediction_map.loc[(surface, pair.right_sample_id)])
            predicted_delta = right - left
            observed_delta = float(pair.observed_delta_pic50)
            observed_evaluable = abs(observed_delta) >= PAIR_OBSERVED_DIRECTION_MINIMUM_PIC50
            predicted_tie = abs(predicted_delta) < PAIR_PREDICTED_TIE_PIC50
            direction_correct = bool(
                observed_evaluable
                and not predicted_tie
                and np.sign(observed_delta) == np.sign(predicted_delta)
            )
            rows.append(
                {
                    **pair._asdict(),
                    "surface": surface,
                    "predicted_delta_pic50": predicted_delta,
                    "absolute_delta_error_pic50": abs(predicted_delta - observed_delta),
                    "observed_direction_evaluable": observed_evaluable,
                    "predicted_tie": predicted_tie,
                    "direction_correct": direction_correct,
                }
            )
    frame = pd.DataFrame(rows)
    aggregate: dict[str, Any] = {
        "status": "held_campaign_pair_evaluation_available",
        "pair_definition": {
            "same_cross_campaign_scaffold": True,
            "minimum_morgan_tanimoto": PAIR_MINIMUM_TANIMOTO,
            "maximum_pairs_per_scaffold": PAIR_MAXIMUM_PER_SCAFFOLD,
            "observed_direction_minimum_pic50": PAIR_OBSERVED_DIRECTION_MINIMUM_PIC50,
            "predicted_tie_pic50": PAIR_PREDICTED_TIE_PIC50,
        },
        "campaigns_with_pairs": sorted(frame.campaign.unique()),
        "campaigns_without_pairs": sorted(set(matrix.campaign.unique()) - set(frame.campaign.unique())),
        "surfaces": {},
    }
    for surface, group in frame.groupby("surface", sort=True):
        evaluable = group.loc[group.observed_direction_evaluable]
        non_tie = evaluable.loc[~evaluable.predicted_tie]
        observed = group.observed_delta_pic50.to_numpy(float)
        predicted = group.predicted_delta_pic50.to_numpy(float)
        denominator = float(np.sum(observed**2))
        delta_rho = spearmanr(observed, predicted).statistic
        aggregate["surfaces"][surface] = {
            "pairs": int(len(group)),
            "direction_evaluable_pairs": int(len(evaluable)),
            "observed_increased_pic50_pairs": int((evaluable.observed_delta_pic50 > 0).sum()),
            "observed_decreased_pic50_pairs": int((evaluable.observed_delta_pic50 < 0).sum()),
            "observed_negligible_pairs": int((~group.observed_direction_evaluable).sum()),
            "direction_accuracy_including_predicted_ties_as_incorrect": (
                float(evaluable.direction_correct.mean()) if len(evaluable) else None
            ),
            "direction_accuracy_excluding_predicted_ties": (
                float(non_tie.direction_correct.mean()) if len(non_tie) else None
            ),
            "predicted_ties": int(evaluable.predicted_tie.sum()),
            "delta_mae_pic50": float(group.absolute_delta_error_pic50.mean()),
            "delta_spearman": float(delta_rho) if np.isfinite(delta_rho) else None,
            "delta_slope_through_origin": float(np.sum(observed * predicted) / denominator)
            if denominator > 0
            else None,
        }
    return frame, aggregate


def _feature_associations(matrix: pd.DataFrame, registry: dict[str, Any]) -> pd.DataFrame:
    rows = []
    target = matrix.target_pic50.to_numpy(float)
    for rule in registry["rules"]:
        column = f"fg_present__{rule['key']}"
        present = matrix[column].to_numpy(bool)
        rows.append(
            {
                "feature": column,
                "label": rule["label"],
                "category": rule["category"],
                "present_n": int(present.sum()),
                "absent_n": int((~present).sum()),
                "prevalence": float(present.mean()),
                "median_pic50_present": float(np.median(target[present])) if present.any() else np.nan,
                "median_pic50_absent": float(np.median(target[~present])) if (~present).any() else np.nan,
                "median_difference_pic50_present_minus_absent": float(
                    np.median(target[present]) - np.median(target[~present])
                )
                if present.any() and (~present).any()
                else np.nan,
                "interpretation_scope": "unadjusted retrospective association; not causal effect",
            }
        )
    return pd.DataFrame(rows)


def _coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (surface, feature), group in coefficients.groupby(["surface", "feature"], sort=True):
        values = group.standardized_coefficient.to_numpy(float)
        nonzero = values[np.abs(values) > 1e-12]
        sign_consistency = (
            max(np.mean(nonzero > 0), np.mean(nonzero < 0)) if len(nonzero) else 1.0
        )
        rows.append(
            {
                "surface": surface,
                "feature": feature,
                "mean_standardized_coefficient": float(np.mean(values)),
                "sd_standardized_coefficient": float(np.std(values)),
                "sign_consistency": float(sign_consistency),
                "outer_models": int(len(values)),
                "interpretation_scope": "nested-model coefficient stability; not causal attribution",
            }
        )
    return pd.DataFrame(rows)


def _model_card(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    decision = report["promotion_decision"]
    pair_metrics = report["pairwise_evaluation"]
    pair_comparison = report["pairwise_primary_vs_frozen"]
    lines = [
        "# hERG V15 functional-group finalization study",
        "",
        "## Decision",
        "",
        "**No model promotion is supported.** The prespecified functional-group candidate failed its retrospective improvement gate, and every label in this study was already open.",
        "",
        "## Scope",
        "",
        "Retrospective, label-open feature study only. No blinded, prospective, final-test, clinical, regulatory, or deployment claim is supported.",
        "",
        "## Evaluation contract",
        "",
        "- Six entire campaigns are outer holds.",
        "- Ridge regularization is selected by leave-one-campaign-out validation inside the other five campaigns.",
        "- Cross-campaign exact and scaffold overlap are both zero.",
        "- The primary candidate was fixed before execution: `fg_interaction_absolute_ridge`.",
        "",
        "## Macro campaign MAE",
        "",
    ]
    for surface, values in sorted(metrics.items()):
        lines.append(
            f"- `{surface}`: {values['regression']['macro_campaign_mae']:.4f} pIC50"
        )
    primary_comparison = report["bootstrap"]["primary_vs_frozen"]
    lines.extend(
        [
            "",
            "## Prespecified primary comparison",
            "",
            (
                f"`{PRIMARY_CANDIDATE}` minus `{FROZEN_COMPARATOR}` macro-campaign MAE: "
                f"{primary_comparison['observed']:+.4f} pIC50 (scaffold-cluster bootstrap "
                f"95% CI {primary_comparison['ci95'][0]:+.4f} to "
                f"{primary_comparison['ci95'][1]:+.4f}); candidate better in "
                f"{primary_comparison['campaigns_better']}/{primary_comparison['campaigns_total']} campaigns."
            ),
            "",
            "## Held-campaign analogue pairs",
            "",
        ]
    )
    if pair_metrics.get("status") == "held_campaign_pair_evaluation_available":
        primary_pair = pair_metrics["surfaces"][PRIMARY_CANDIDATE]
        lines.extend(
            [
                (
                    f"There are {primary_pair['pairs']} same-scaffold pairs in only "
                    f"{len(pair_metrics['campaigns_with_pairs'])} external campaigns. The primary "
                    f"surface has direction accuracy "
                    f"{primary_pair['direction_accuracy_including_predicted_ties_as_incorrect']:.3f} "
                    f"when ties count as incorrect, delta MAE {primary_pair['delta_mae_pic50']:.3f} "
                    f"pIC50, and delta slope {primary_pair['delta_slope_through_origin']:.3f}."
                ),
                (
                    "The scaffold-cluster pair comparison versus the frozen V14.1 ligand model is "
                    f"descriptive only: delta-MAE difference "
                    f"{pair_comparison['delta_mae_candidate_minus_comparator']['observed_macro_campaign']:+.3f} "
                    "pIC50 and direction-accuracy difference "
                    f"{pair_comparison['direction_accuracy_candidate_minus_comparator_including_ties']['observed_macro_campaign']:+.3f}."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Unsupported comparisons",
            "",
        ]
    )
    for name, reason in report["unsupported_comparisons"].items():
        lines.append(f"- `{name}`: {reason}")
    lines.extend(
        [
            "",
            "## Scientific boundary",
            "",
            "Functional-group presence, coefficients, and property interactions are associations and encodings, not anecdotal direction rules or causal hERG mechanisms. Same-series local residual fitting and a dedicated pairwise delta model were not supported by the leakage-controlled campaign matrix.",
            "",
            "## Failed promotion reasons",
            "",
        ]
    )
    for reason in decision["failed_promotion_reasons"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            ".venv/bin/python pipeline/scripts/run_local_herg_functional_group_finalize_v15.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    matrix_path: Path,
    v141_predictions: Path,
    registry_path: Path,
    output: Path,
) -> dict[str, Any]:
    registry = fg.load_registry(registry_path)
    matrix = _load_matrix(matrix_path, v141_predictions)
    audit = _audit_matrix(matrix)
    contracts = _surface_contracts(registry)
    predictions, selection, coefficients = _nested_predictions(matrix, contracts)
    metrics_by_campaign, aggregate = _metric_tables(predictions)
    pair_predictions, pair_metrics = _pair_evidence(matrix, predictions)
    associations = _feature_associations(matrix, registry)
    coefficient_stability = _coefficient_stability(coefficients)

    bootstrap_comparisons = {
        candidate: {
            "vs_frozen_v14_1_ligand_core": _cluster_bootstrap_delta(
                predictions, candidate, FROZEN_COMPARATOR
            ),
            "vs_frozen_v9_v10_baseline": _cluster_bootstrap_delta(
                predictions, candidate, "frozen_v9_v10_baseline"
            ),
        }
        for candidate in contracts
    }
    primary_vs_frozen = bootstrap_comparisons[PRIMARY_CANDIDATE][
        "vs_frozen_v14_1_ligand_core"
    ]
    primary_vs_deployed = bootstrap_comparisons[PRIMARY_CANDIDATE][
        "vs_frozen_v9_v10_baseline"
    ]
    pairwise_primary_vs_frozen = _pair_bootstrap_comparison(
        pair_predictions, PRIMARY_CANDIDATE, FROZEN_COMPARATOR
    )
    research_signal_gate = bool(
        primary_vs_frozen["ci95"][1] < 0
        and primary_vs_frozen["campaigns_better"] >= 4
        and aggregate[PRIMARY_CANDIDATE]["regression"]["macro_campaign_mae"]
        < aggregate[FROZEN_COMPARATOR]["regression"]["macro_campaign_mae"]
    )
    promotion = {
        "prespecified_primary_candidate": PRIMARY_CANDIDATE,
        "frozen_comparator": FROZEN_COMPARATOR,
        "research_signal_gate_passed": research_signal_gate,
        "production_or_default_promotion_supported": False,
        "failed_promotion_reasons": [
            "all six campaign labels were open before this V15 feature study",
            "no blinded or prospective confirmation was reserved",
            "functional-group registry and model surfaces were evaluated post hoc on existing campaigns",
            "same-series local residual and dedicated pairwise-delta models lack leakage-controlled multi-campaign support",
            *([] if research_signal_gate else ["the prespecified primary candidate failed its retrospective scaffold-bootstrap improvement gate"]),
        ],
    }
    report_value = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "status": "complete_retrospective_functional_group_study",
        "dataset": audit,
        "registry": {
            "path": str(registry_path.resolve()),
            "version": registry["registry_version"],
            "registry_sha256": registry["registry_sha256"],
            "rules": len(registry["rules"]),
            "feature_columns": len(fg.feature_columns(registry)),
        },
        "software_and_calculation_methods": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "scikit_learn": sklearn.__version__,
            "molecular_weight": "RDKit Descriptors.MolWt",
            "tpsa": "RDKit rdMolDescriptors.CalcTPSA",
            "clogp": "RDKit Crippen.MolLogP",
            "hbd_hba_rotatable_bonds": "RDKit Lipinski descriptors",
            "ionizable_centers": "RDKit BaseFeatures PosIonizable/NegIonizable feature families; not pKa",
            "pair_similarity": (
                "RDKit Morgan bit vector radius=2, size=2048, includeChirality=True; Tanimoto"
            ),
        },
        "evaluation_contract": {
            "outer_split": "leave one entire campaign out",
            "inner_selection": "leave one campaign out among the five outer-training campaigns",
            "outer_scaffold_overlap": 0,
            "outer_exact_overlap": 0,
            "ridge_alphas": list(RIDGE_ALPHAS),
            "campaign_balanced_training_weights": True,
            "random_seed": SEED,
            "all_labels_open_before_study": True,
            "blinded_or_prospective_test": False,
        },
        "surfaces": {
            "frozen_v9_v10_baseline": {"role": "unchanged deployed frozen regression comparator"},
            FROZEN_COMPARATOR: {"role": "unchanged prior V14.1 nested-LOCO ligand comparator"},
            **contracts,
        },
        "metrics": aggregate,
        "bootstrap": {
            "primary_vs_frozen": primary_vs_frozen,
            "primary_vs_deployed": primary_vs_deployed,
            "all_ablation_comparisons": bootstrap_comparisons,
        },
        "pairwise_evaluation": pair_metrics,
        "pairwise_primary_vs_frozen": pairwise_primary_vs_frozen,
        "unsupported_comparisons": {
            "same_series_local_residual": (
                "unsupported: cross-campaign scaffold overlap is intentionally zero, so no labeled "
                "same-series reference is available to an outer held campaign without leakage"
            ),
            "dedicated_pairwise_delta_model": (
                "unsupported: repeated-scaffold analogue pairs occur in only two of six campaigns, "
                "which cannot support nested campaign selection for every outer test"
            ),
            "pairwise_absolute_optimization": (
                "unsupported: held-pair evaluation measures induced ranking/delta only and does not "
                "supply a prospective optimization guarantee"
            ),
        },
        "promotion_decision": promotion,
        "scientific_scope": {
            "public_output": False,
            "website_or_deployment_changed": False,
            "causal_functional_group_effects_claimed": False,
            "pka_predicted": False,
            "anecdotal_direction_rules_hard_coded": False,
            "research_use_only": True,
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    feature_output = matrix[["sample_id", "campaign", "canonical_smiles", *fg.feature_columns(registry)]].copy()
    _parquet(output / "data/functional_group_features.parquet", feature_output)
    _parquet(output / "predictions/nested_loco_predictions.parquet", predictions)
    _parquet(output / "evidence/inner_selection.parquet", selection)
    _parquet(output / "evidence/nested_coefficients.parquet", coefficients)
    _parquet(output / "evidence/coefficient_stability.parquet", coefficient_stability)
    _parquet(output / "evidence/metrics_by_campaign.parquet", metrics_by_campaign)
    _parquet(output / "evidence/pair_predictions.parquet", pair_predictions)
    _parquet(output / "evidence/functional_group_associations.parquet", associations)
    metrics_report = _self_hashed_json(
        output / "evidence/aggregate_metrics.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": report_value["created_utc"],
            "metrics": aggregate,
            "bootstrap": report_value["bootstrap"],
            "pairwise_evaluation": pair_metrics,
            "pairwise_primary_vs_frozen": pairwise_primary_vs_frozen,
        },
    )
    registry_snapshot = output / "registry/herg_v15_functional_group_registry_v1.json"
    registry_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(registry_path, registry_snapshot)
    report_value["aggregate_metrics_report_sha256"] = metrics_report["report_sha256"]
    report = _self_hashed_json(output / "analysis_report.json", report_value)
    (output / "MODEL_CARD.md").write_text(_model_card(report))

    artifact_paths = [
        output / "registry/herg_v15_functional_group_registry_v1.json",
        output / "data/functional_group_features.parquet",
        output / "predictions/nested_loco_predictions.parquet",
        output / "evidence/inner_selection.parquet",
        output / "evidence/nested_coefficients.parquet",
        output / "evidence/coefficient_stability.parquet",
        output / "evidence/metrics_by_campaign.parquet",
        output / "evidence/pair_predictions.parquet",
        output / "evidence/functional_group_associations.parquet",
        output / "evidence/aggregate_metrics.json",
        output / "analysis_report.json",
        output / "MODEL_CARD.md",
    ]
    _self_hashed_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": report["created_utc"],
            "status": report["status"],
            "script": {"path": str(Path(__file__).resolve()), "sha256": _sha(Path(__file__))},
            "code_dependencies": [
                {
                    "path": str(Path(fg.__file__).resolve()),
                    "sha256": _sha(Path(fg.__file__).resolve()),
                }
            ],
            "inputs": [
                {"path": str(matrix_path.resolve()), "sha256": _sha(matrix_path)},
                {"path": str(v141_predictions.resolve()), "sha256": _sha(v141_predictions)},
                {"path": str(registry_path.resolve()), "sha256": _sha(registry_path)},
            ],
            "artifacts": [
                {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}
                for path in artifact_paths
            ],
            "claim_boundary": (
                "retrospective label-open functional-group study; no deployment or prospective promotion"
            ),
        },
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--v141-predictions", type=Path, default=DEFAULT_V141_PREDICTIONS)
    parser.add_argument("--registry", type=Path, default=fg.DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run(
        args.matrix.resolve(),
        args.v141_predictions.resolve(),
        args.registry.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(report["promotion_decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
