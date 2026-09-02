#!/usr/bin/env python3
"""Audit molecular-weight attenuation of hERG matched-pair prediction deltas.

This module is deliberately separate from prediction serving.  It joins an
existing measured matched-molecular-pair (MMP) registry to structure-level OOF
predictions and molecular weights, then reports transparent retrospective
diagnostics.  It never refits or mutates the primary hERG model.

The optional :func:`mw_proportional_delta_stress_test` is also isolated and
off by default.  It is a scenario calculation, not a calibrated prediction.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "herg-mw-pair-delta-attenuation/1.0"
DEFAULT_MW_BOUNDARIES = (400.0, 500.0, 600.0, 700.0)
DEFAULT_SENSITIVITY_THRESHOLDS = (0.1, 0.3, 0.5, 1.0)
PAIR_COLUMNS = (
    "pair_id",
    "structure_id_a",
    "structure_id_b",
    "pic50_median_a",
    "pic50_median_b",
    "delta_pic50_b_minus_a",
    "absolute_delta_pic50",
)


@dataclass(frozen=True)
class MWPairDeltaAudit:
    """Prepared pair rows, MW-band diagnostics, sensitivity results, and report."""

    rows: pd.DataFrame
    band_metrics: pd.DataFrame
    cliff_band_metrics: pd.DataFrame
    sensitivity_metrics: pd.DataFrame
    report: dict[str, Any]


def _finite_numeric(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{label} column {column!r} must contain only finite numeric values")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _component_ids(frame: pd.DataFrame) -> pd.Series:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        parent[larger] = smaller

    for left, right in frame[["structure_id_a", "structure_id_b"]].itertuples(index=False, name=None):
        union(str(left), str(right))
    return frame["structure_id_a"].astype(str).map(find).astype("string")


def prepare_pair_frame(
    pairs: pd.DataFrame,
    oof: pd.DataFrame,
    *,
    prediction_column: str,
    molecular_weight_column: str = "rdkit2d__MolWt",
    observed_column: str = "observed_pic50",
    outer_fold_column: str = "outer_fold",
    require_target_consistency: bool = True,
    target_tolerance_pic50: float = 1e-6,
    same_outer_fold_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join measured MMPs to OOF values and enforce a defensible comparison contract.

    ``require_target_consistency`` removes pairs whose MMP endpoint aggregate is
    not the same quantitative target represented by the OOF row.  This prevents
    silently comparing a model delta with a differently aggregated measurement.
    """

    _require_columns(pairs, PAIR_COLUMNS, "pair frame")
    oof_columns = (
        "structure_id",
        observed_column,
        prediction_column,
        molecular_weight_column,
        outer_fold_column,
    )
    _require_columns(oof, oof_columns, "OOF frame")
    if pairs.empty:
        raise ValueError("pair frame contains no rows")
    if oof.empty:
        raise ValueError("OOF frame contains no rows")
    if pairs["pair_id"].isna().any() or pairs["pair_id"].astype(str).str.strip().eq("").any():
        raise ValueError("pair_id must not contain missing or blank values")
    if pairs["pair_id"].duplicated().any():
        raise ValueError("pair_id values must be unique")
    if pairs["structure_id_a"].astype(str).eq(pairs["structure_id_b"].astype(str)).any():
        raise ValueError("matched pairs must contain two different structure IDs")
    if oof["structure_id"].isna().any() or oof["structure_id"].duplicated().any():
        raise ValueError("OOF structure_id values must be unique and non-missing")
    if target_tolerance_pic50 < 0 or not math.isfinite(target_tolerance_pic50):
        raise ValueError("target_tolerance_pic50 must be finite and nonnegative")

    result = pairs.copy().reset_index(drop=True)
    _finite_numeric(
        result,
        (
            "pic50_median_a",
            "pic50_median_b",
            "delta_pic50_b_minus_a",
            "absolute_delta_pic50",
        ),
        "pair frame",
    )
    calculated_delta = result["pic50_median_b"] - result["pic50_median_a"]
    if not np.allclose(
        calculated_delta,
        result["delta_pic50_b_minus_a"],
        rtol=0.0,
        atol=max(target_tolerance_pic50, 1e-12),
    ):
        raise ValueError("delta_pic50_b_minus_a must equal pic50_median_b - pic50_median_a")
    if not np.allclose(
        result["absolute_delta_pic50"],
        result["delta_pic50_b_minus_a"].abs(),
        rtol=0.0,
        atol=max(target_tolerance_pic50, 1e-12),
    ):
        raise ValueError("absolute_delta_pic50 is inconsistent with the signed measured delta")

    selected_oof = oof[list(oof_columns)].copy()
    _finite_numeric(
        selected_oof,
        (observed_column, prediction_column, molecular_weight_column, outer_fold_column),
        "OOF frame",
    )
    if selected_oof[molecular_weight_column].le(0).any():
        raise ValueError("OOF molecular weights must be positive")

    for side in ("a", "b"):
        renamed = selected_oof.rename(
            columns={
                "structure_id": f"structure_id_{side}",
                observed_column: f"oof_observed_pic50_{side}",
                prediction_column: f"oof_predicted_pic50_{side}",
                molecular_weight_column: f"molecular_weight_{side}_da",
                outer_fold_column: f"outer_fold_{side}",
            }
        )
        before = len(result)
        result = result.merge(renamed, on=f"structure_id_{side}", how="left", validate="many_to_one")
        if len(result) != before:
            raise AssertionError("pair-to-OOF merge changed row count")

    joined_columns = [
        "oof_observed_pic50_a",
        "oof_observed_pic50_b",
        "oof_predicted_pic50_a",
        "oof_predicted_pic50_b",
        "molecular_weight_a_da",
        "molecular_weight_b_da",
        "outer_fold_a",
        "outer_fold_b",
    ]
    missing_join = result[joined_columns].isna().any(axis=1)
    if missing_join.any():
        missing_ids = sorted(
            set(result.loc[missing_join, "structure_id_a"].astype(str))
            | set(result.loc[missing_join, "structure_id_b"].astype(str))
        )
        raise ValueError(f"OOF frame lacks required values for {len(missing_ids)} pair structures")

    result["target_consistent"] = (
        (result["pic50_median_a"] - result["oof_observed_pic50_a"]).abs() <= target_tolerance_pic50
    ) & ((result["pic50_median_b"] - result["oof_observed_pic50_b"]).abs() <= target_tolerance_pic50)
    result["same_outer_fold"] = result["outer_fold_a"].eq(result["outer_fold_b"])
    source_pairs = len(result)
    target_mismatch_pairs = int((~result["target_consistent"]).sum())
    different_fold_pairs = int((~result["same_outer_fold"]).sum())
    if require_target_consistency:
        result = result.loc[result["target_consistent"]].copy()
    if same_outer_fold_only:
        result = result.loc[result["same_outer_fold"]].copy()
    if result.empty:
        raise ValueError("no pairs remain after target/fold consistency filters")

    result["predicted_delta_pic50_b_minus_a"] = (
        result["oof_predicted_pic50_b"] - result["oof_predicted_pic50_a"]
    )
    result["absolute_predicted_delta_pic50"] = result["predicted_delta_pic50_b_minus_a"].abs()
    result["pair_mean_mw_da"] = (result["molecular_weight_a_da"] + result["molecular_weight_b_da"]) / 2.0
    result["pair_max_mw_da"] = result[["molecular_weight_a_da", "molecular_weight_b_da"]].max(axis=1)
    result["direction_correct"] = np.sign(result["predicted_delta_pic50_b_minus_a"]).eq(
        np.sign(result["delta_pic50_b_minus_a"])
    )
    result["mmp_component_id"] = _component_ids(result)
    return result.reset_index(drop=True), {
        "source_pairs": int(source_pairs),
        "retained_pairs": int(len(result)),
        "target_mismatch_pairs": target_mismatch_pairs,
        "different_outer_fold_pairs": different_fold_pairs,
        "require_target_consistency": bool(require_target_consistency),
        "same_outer_fold_only": bool(same_outer_fold_only),
        "target_tolerance_pic50": float(target_tolerance_pic50),
    }


def add_mw_bands(
    frame: pd.DataFrame,
    boundaries: tuple[float, ...] = DEFAULT_MW_BOUNDARIES,
    *,
    source_column: str = "pair_mean_mw_da",
) -> pd.DataFrame:
    """Add deterministic left-closed MW bands to a prepared pair frame."""

    if source_column not in frame:
        raise ValueError(f"missing MW source column: {source_column}")
    values = np.asarray(boundaries, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError("MW boundaries must be finite and strictly increasing")
    labels = [f"<{values[0]:g}"]
    labels.extend(f"{left:g}-{right:g}" for left, right in zip(values[:-1], values[1:], strict=True))
    labels.append(f">={values[-1]:g}")
    result = frame.copy()
    result["mw_band"] = pd.cut(
        result[source_column],
        [-np.inf, *values, np.inf],
        labels=labels,
        right=False,
        ordered=True,
    )
    return result


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0 or len(values) != len(weights):
        return math.nan
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    total = float(ordered_weights.sum())
    if not math.isfinite(total) or total <= 0:
        return math.nan
    index = int(np.searchsorted(np.cumsum(ordered_weights), total / 2.0, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _l1_optimal_nonnegative_scale(measured: np.ndarray, predicted: np.ndarray) -> float:
    """Return argmin_s>=0 sum |measured - s*predicted|.

    The solution is the |predicted|-weighted median of measured/predicted.
    It is a hindsight calibration diagnostic, not an ideal or deployable ratio.
    """

    nonzero = np.abs(predicted) > 1e-12
    if not nonzero.any():
        return math.nan
    value = _weighted_median(measured[nonzero] / predicted[nonzero], np.abs(predicted[nonzero]))
    return max(0.0, value) if math.isfinite(value) else math.nan


def _point_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    measured = frame["delta_pic50_b_minus_a"].to_numpy(dtype=float)
    predicted = frame["predicted_delta_pic50_b_minus_a"].to_numpy(dtype=float)
    measured_abs_sum = float(np.abs(measured).sum())
    magnitude_capture = (
        float(np.abs(predicted).sum() / measured_abs_sum) if measured_abs_sum > 0 else math.nan
    )
    denominator = float(np.dot(measured, measured))
    signed_slope = float(np.dot(measured, predicted) / denominator) if denominator > 0 else math.nan
    l1_scale = _l1_optimal_nonnegative_scale(measured, predicted)
    unscaled_mae = float(np.mean(np.abs(predicted - measured)))
    scaled_mae = (
        float(np.mean(np.abs(l1_scale * predicted - measured))) if math.isfinite(l1_scale) else math.nan
    )
    return {
        "n_pairs": int(len(frame)),
        "n_components": int(frame["mmp_component_id"].nunique()),
        "mean_pair_mw_da": float(frame["pair_mean_mw_da"].mean()),
        "median_pair_mw_da": float(frame["pair_mean_mw_da"].median()),
        "median_absolute_measured_delta_pic50": float(frame["absolute_delta_pic50"].median()),
        "magnitude_capture_fraction": magnitude_capture,
        "signed_calibration_slope_through_origin": signed_slope,
        "directional_accuracy": float(frame["direction_correct"].mean()),
        "delta_mae_pic50": unscaled_mae,
        "retrospective_l1_optimal_nonnegative_scale": (l1_scale if math.isfinite(l1_scale) else None),
        "delta_mae_after_retrospective_l1_scale_pic50": (scaled_mae if math.isfinite(scaled_mae) else None),
    }


def _cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, tuple[float | None, float | None]]:
    """Percentile intervals from a connected-component cluster bootstrap."""

    if replicates < 0:
        raise ValueError("bootstrap replicates must be nonnegative")
    components = pd.Index(frame["mmp_component_id"].drop_duplicates())
    if replicates == 0 or len(components) < 2:
        return {
            name: (None, None)
            for name in (
                "magnitude_capture_fraction",
                "signed_calibration_slope_through_origin",
                "directional_accuracy",
                "retrospective_l1_optimal_nonnegative_scale",
            )
        }

    component_index = pd.Categorical(frame["mmp_component_id"], categories=components).codes
    measured = frame["delta_pic50_b_minus_a"].to_numpy(dtype=float)
    predicted = frame["predicted_delta_pic50_b_minus_a"].to_numpy(dtype=float)
    correct = frame["direction_correct"].to_numpy(dtype=float)
    n_components = len(components)
    component_stats = np.zeros((n_components, 6), dtype=float)
    np.add.at(component_stats[:, 0], component_index, np.abs(predicted))
    np.add.at(component_stats[:, 1], component_index, np.abs(measured))
    np.add.at(component_stats[:, 2], component_index, measured * predicted)
    np.add.at(component_stats[:, 3], component_index, measured * measured)
    np.add.at(component_stats[:, 4], component_index, correct)
    np.add.at(component_stats[:, 5], component_index, 1.0)

    nonzero = np.abs(predicted) > 1e-12
    ratios = measured[nonzero] / predicted[nonzero]
    ratio_weights = np.abs(predicted[nonzero])
    ratio_components = component_index[nonzero]
    ratio_order = np.argsort(ratios, kind="mergesort")
    ratios = ratios[ratio_order]
    ratio_weights = ratio_weights[ratio_order]
    ratio_components = ratio_components[ratio_order]

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {
        "magnitude_capture_fraction": [],
        "signed_calibration_slope_through_origin": [],
        "directional_accuracy": [],
        "retrospective_l1_optimal_nonnegative_scale": [],
    }
    for _ in range(replicates):
        sampled = rng.integers(0, n_components, size=n_components)
        counts = np.bincount(sampled, minlength=n_components).astype(float)
        totals = counts @ component_stats
        if totals[1] > 0:
            draws["magnitude_capture_fraction"].append(float(totals[0] / totals[1]))
        if totals[3] > 0:
            draws["signed_calibration_slope_through_origin"].append(float(totals[2] / totals[3]))
        if totals[5] > 0:
            draws["directional_accuracy"].append(float(totals[4] / totals[5]))
        weights = ratio_weights * counts[ratio_components]
        value = _weighted_median(ratios, weights)
        if math.isfinite(value):
            draws["retrospective_l1_optimal_nonnegative_scale"].append(max(0.0, value))

    intervals: dict[str, tuple[float | None, float | None]] = {}
    for name, values in draws.items():
        if not values:
            intervals[name] = (None, None)
            continue
        lower, upper = np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])
        intervals[name] = (float(lower), float(upper))
    return intervals


def summarize_pairs(
    frame: pd.DataFrame,
    *,
    minimum_absolute_measured_delta_pic50: float,
    bootstrap_replicates: int,
    seed: int,
    minimum_pairs_for_nonsparse_label: int = 30,
    minimum_components_for_nonsparse_label: int = 20,
) -> dict[str, Any]:
    """Summarize one pair subset with transparent attenuation diagnostics."""

    threshold = float(minimum_absolute_measured_delta_pic50)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("minimum measured delta must be finite and nonnegative")
    selected = frame.loc[frame["absolute_delta_pic50"].ge(threshold)].copy()
    if selected.empty:
        return {
            "n_pairs": 0,
            "n_components": 0,
            "minimum_absolute_measured_delta_pic50": threshold,
            "sparse_support": True,
        }
    point = _point_metrics(selected)
    intervals = _cluster_bootstrap(selected, replicates=bootstrap_replicates, seed=seed)
    point["minimum_absolute_measured_delta_pic50"] = threshold
    point["bootstrap_replicates"] = int(bootstrap_replicates)
    point["bootstrap_unit"] = "connected MMP component"
    point["sparse_support"] = bool(
        int(point["n_pairs"]) < minimum_pairs_for_nonsparse_label
        or int(point["n_components"]) < minimum_components_for_nonsparse_label
    )
    point["sparse_support_rule"] = {
        "minimum_pairs": int(minimum_pairs_for_nonsparse_label),
        "minimum_components": int(minimum_components_for_nonsparse_label),
        "meaning": "operational reporting flag, not a universal statistical threshold",
    }
    for metric, (lower, upper) in intervals.items():
        point[f"{metric}_ci95_lower"] = lower
        point[f"{metric}_ci95_upper"] = upper
    return point


def audit_mw_pair_delta_attenuation(
    pairs: pd.DataFrame,
    oof: pd.DataFrame,
    *,
    prediction_column: str,
    molecular_weight_column: str = "rdkit2d__MolWt",
    primary_minimum_delta_pic50: float = 0.1,
    sensitivity_thresholds: tuple[float, ...] = DEFAULT_SENSITIVITY_THRESHOLDS,
    mw_boundaries: tuple[float, ...] = DEFAULT_MW_BOUNDARIES,
    bootstrap_replicates: int = 500,
    seed: int = 20260831,
    require_target_consistency: bool = True,
    same_outer_fold_only: bool = False,
) -> MWPairDeltaAudit:
    """Run the complete retrospective MW/pair-delta audit from in-memory frames."""

    prepared, preparation = prepare_pair_frame(
        pairs,
        oof,
        prediction_column=prediction_column,
        molecular_weight_column=molecular_weight_column,
        require_target_consistency=require_target_consistency,
        same_outer_fold_only=same_outer_fold_only,
    )
    prepared = add_mw_bands(prepared, mw_boundaries)
    overall = summarize_pairs(
        prepared,
        minimum_absolute_measured_delta_pic50=primary_minimum_delta_pic50,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    band_rows = []
    cliff_band_rows = []
    for index, (band, group) in enumerate(prepared.groupby("mw_band", observed=False, sort=True)):
        metrics = summarize_pairs(
            group,
            minimum_absolute_measured_delta_pic50=primary_minimum_delta_pic50,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + index + 1,
        )
        band_rows.append({"mw_band": str(band), **metrics})
        cliff_metrics = summarize_pairs(
            group,
            minimum_absolute_measured_delta_pic50=1.0,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 1_000 + index,
        )
        cliff_band_rows.append({"mw_band": str(band), **cliff_metrics})
    sensitivity_rows = []
    for threshold in sensitivity_thresholds:
        metrics = summarize_pairs(
            prepared,
            minimum_absolute_measured_delta_pic50=float(threshold),
            bootstrap_replicates=0,
            seed=seed,
        )
        sensitivity_rows.append(metrics)

    reference_median_mw = float(oof.drop_duplicates("structure_id")[molecular_weight_column].median())
    cliff = summarize_pairs(
        prepared,
        minimum_absolute_measured_delta_pic50=1.0,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 10_000,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "result_kind": "retrospective_oof_mmp_delta_attenuation_audit",
        "prediction_column": prediction_column,
        "molecular_weight_column": molecular_weight_column,
        "preparation": preparation,
        "pair_mw_definition": "arithmetic mean of endpoint molecular weights in Da",
        "measured_delta_definition": "pIC50(B) - pIC50(A); positive means stronger hERG inhibition",
        "predicted_delta_definition": "OOF predicted pIC50(B) - OOF predicted pIC50(A)",
        "overall": overall,
        "activity_cliff_subset_absolute_delta_pic50_ge_1": cliff,
        "reference_training_structure_median_mw_da": reference_median_mw,
        "metric_definitions": {
            "magnitude_capture_fraction": (
                "sum(abs(predicted delta)) / sum(abs(measured delta)); ignores direction and is not "
                "an ideal target ratio"
            ),
            "signed_calibration_slope_through_origin": (
                "argmin least-squares slope in predicted delta = slope * measured delta; sign errors "
                "reduce the slope"
            ),
            "retrospective_l1_optimal_nonnegative_scale": (
                "nonnegative hindsight multiplier minimizing sum(abs(measured delta - multiplier * "
                "predicted delta)); fitted on the evaluated data and not an independent correction"
            ),
        },
        "stress_test_parameter_candidate": {
            "reference_median_mw_da": reference_median_mw,
            "retrospective_cliff_l1_baseline_multiplier": cliff.get(
                "retrospective_l1_optimal_nonnegative_scale"
            ),
            "formula": ("max(1, baseline_multiplier * pair_mean_mw_da / reference_median_mw_da)"),
            "status": "off_by_default_retrospective_scenario_only",
        },
        "claim_boundary": (
            "Internal train-partition MMPs and OOF predictions are retrospective and chemically dependent. "
            "They support an attenuation diagnostic, not a production recalibration, causal MW effect, "
            "or prospective accuracy claim. Sparse high-MW bands must remain descriptive."
        ),
    }
    return MWPairDeltaAudit(
        rows=prepared,
        band_metrics=pd.DataFrame.from_records(band_rows),
        cliff_band_metrics=pd.DataFrame.from_records(cliff_band_rows),
        sensitivity_metrics=pd.DataFrame.from_records(sensitivity_rows),
        report=report,
    )


def mw_proportional_delta_stress_test(
    parent_pic50: float,
    candidate_pic50: float,
    parent_mw_da: float,
    candidate_mw_da: float,
    *,
    enabled: bool = False,
    reference_median_mw_da: float | None = None,
    baseline_multiplier: float = 1.0,
    minimum_multiplier: float = 1.0,
    maximum_multiplier: float | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an isolated MW-proportional ΔpIC50 stress scenario.

    The primary candidate prediction is always returned unchanged.  Only the
    separately labeled scenario delta is scaled, so direction is preserved.
    The caller must explicitly set ``enabled=True`` and provide the empirical
    reference median.  No default production correction is hidden here.
    """

    inputs = np.asarray([parent_pic50, candidate_pic50, parent_mw_da, candidate_mw_da], dtype=float)
    if not np.isfinite(inputs).all():
        raise ValueError("pIC50 and MW inputs must be finite")
    if parent_mw_da <= 0 or candidate_mw_da <= 0:
        raise ValueError("molecular weights must be positive")
    if not math.isfinite(baseline_multiplier) or baseline_multiplier <= 0:
        raise ValueError("baseline_multiplier must be finite and positive")
    if not math.isfinite(minimum_multiplier) or minimum_multiplier < 0:
        raise ValueError("minimum_multiplier must be finite and nonnegative")
    if maximum_multiplier is not None:
        if not math.isfinite(maximum_multiplier) or maximum_multiplier < minimum_multiplier:
            raise ValueError("maximum_multiplier must be finite and >= minimum_multiplier")
    if enabled and (
        reference_median_mw_da is None
        or not math.isfinite(reference_median_mw_da)
        or reference_median_mw_da <= 0
    ):
        raise ValueError("enabled stress testing requires a finite, positive reference median MW")

    primary_delta = float(candidate_pic50 - parent_pic50)
    pair_mean_mw = float((parent_mw_da + candidate_mw_da) / 2.0)
    if enabled:
        assert reference_median_mw_da is not None
        raw_multiplier = float(baseline_multiplier * pair_mean_mw / reference_median_mw_da)
        applied_multiplier = max(float(minimum_multiplier), raw_multiplier)
        if maximum_multiplier is not None:
            applied_multiplier = min(applied_multiplier, float(maximum_multiplier))
    else:
        raw_multiplier = 1.0
        applied_multiplier = 1.0
    stressed_delta = float(primary_delta * applied_multiplier)
    stressed_candidate = float(parent_pic50 + stressed_delta)
    direction_preserved = bool(primary_delta == 0.0 or np.sign(primary_delta) == np.sign(stressed_delta))
    return {
        "schema_version": SCHEMA_VERSION,
        "result_kind": "experimental_mw_proportional_delta_stress_test",
        "enabled": bool(enabled),
        "default_enabled": False,
        "applied": bool(enabled),
        "primary_prediction_unchanged": {
            "parent_pic50": float(parent_pic50),
            "candidate_pic50": float(candidate_pic50),
            "candidate_minus_parent_delta_pic50": primary_delta,
        },
        "stress_scenario": {
            "candidate_pic50": stressed_candidate,
            "candidate_minus_parent_delta_pic50": stressed_delta,
            "pair_mean_mw_da": pair_mean_mw,
            "raw_multiplier": raw_multiplier,
            "applied_multiplier": applied_multiplier,
            "direction_preserved": direction_preserved,
        },
        "inputs": {
            "parent_mw_da": float(parent_mw_da),
            "candidate_mw_da": float(candidate_mw_da),
            "reference_median_mw_da": (
                float(reference_median_mw_da) if reference_median_mw_da is not None else None
            ),
            "baseline_multiplier": float(baseline_multiplier),
            "minimum_multiplier": float(minimum_multiplier),
            "maximum_multiplier": (float(maximum_multiplier) if maximum_multiplier is not None else None),
        },
        "formula": (
            "pair_mean_mw = (parent_mw + candidate_mw) / 2; "
            "multiplier = clamp_min(baseline_multiplier * pair_mean_mw / "
            "reference_median_mw, minimum_multiplier); stressed_candidate_pIC50 = "
            "parent_pIC50 + (candidate_pIC50 - parent_pIC50) * multiplier"
            + (
                "; multiplier is additionally capped at maximum_multiplier"
                if maximum_multiplier is not None
                else ""
            )
        ),
        "provenance": provenance or {},
        "limitations": [
            "This is an opt-in retrospective stress scenario, not the primary prediction.",
            "It assumes MW-proportional amplification and cannot repair an incorrect predicted direction.",
            "Any fitted baseline multiplier is non-independent if estimated from the audit pairs.",
            "Support above 700 Da is sparse in the current local measured MMP evidence.",
            "The calculation does not model assay noise, ionization, receptor state, or edit mechanism.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[2]
    default_root = repo / "research/local_runs/herg_comprehensive_optimization_v11_1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=default_root / "prepared/training_mmp_effects.parquet",
    )
    parser.add_argument(
        "--oof",
        type=Path,
        default=default_root / "analysis/nested_oof_predictions.parquet",
    )
    parser.add_argument("--prediction-column", default="v9_predicted_pic50")
    parser.add_argument("--molecular-weight-column", default="rdkit2d__MolWt")
    parser.add_argument("--primary-minimum-delta-pic50", type=float, default=0.1)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--same-outer-fold-only", action="store_true")
    parser.add_argument("--allow-target-mismatch", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-bands-csv", type=Path)
    parser.add_argument("--output-sensitivity-csv", type=Path)
    return parser


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars and missing float values to strict JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    pairs = pd.read_parquet(args.pairs)
    oof = pd.read_parquet(args.oof)
    audit = audit_mw_pair_delta_attenuation(
        pairs,
        oof,
        prediction_column=args.prediction_column,
        molecular_weight_column=args.molecular_weight_column,
        primary_minimum_delta_pic50=args.primary_minimum_delta_pic50,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        require_target_consistency=not args.allow_target_mismatch,
        same_outer_fold_only=args.same_outer_fold_only,
    )
    payload = {
        **audit.report,
        "mw_band_metrics": audit.band_metrics.to_dict(orient="records"),
        "activity_cliff_mw_band_metrics": audit.cliff_band_metrics.to_dict(orient="records"),
        "minimum_delta_sensitivity": audit.sensitivity_metrics.to_dict(orient="records"),
    }
    rendered = json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    if args.output_bands_csv:
        args.output_bands_csv.parent.mkdir(parents=True, exist_ok=True)
        audit.band_metrics.to_csv(args.output_bands_csv, index=False)
    if args.output_sensitivity_csv:
        args.output_sensitivity_csv.parent.mkdir(parents=True, exist_ok=True)
        audit.sensitivity_metrics.to_csv(args.output_sensitivity_csv, index=False)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
