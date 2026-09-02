#!/usr/bin/env python3
"""Evaluate whether hERG predictions capture measured parent-to-candidate edits.

The module deliberately has no model import. Callers may provide frozen IC50
predictions in the input frame, pass aligned arrays, or supply a lightweight
callback that maps an ordered SMILES sequence to predicted IC50 values in uM.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "pair_id",
    "parent_smiles",
    "candidate_smiles",
    "parent_measured_ic50_um",
    "candidate_measured_ic50_um",
)
PREDICTION_COLUMNS = (
    "parent_predicted_ic50_um",
    "candidate_predicted_ic50_um",
)
MW_COLUMNS = ("parent_mw", "candidate_mw")
Predictor = Callable[[Sequence[str]], Sequence[float]]


@dataclass(frozen=True)
class PairedEvaluation:
    """Detailed pair rows, overall metrics, and optional stratified metrics."""

    rows: pd.DataFrame
    summary: dict[str, Any]
    strata: pd.DataFrame


def _finite_positive(values: pd.Series, column: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(values, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must contain only numeric values") from exc
    invalid = ~np.isfinite(numeric.to_numpy()) | numeric.le(0).to_numpy()
    if invalid.any():
        rows = (np.flatnonzero(invalid) + 2).tolist()
        raise ValueError(f"{column} must be finite and positive; invalid CSV rows: {rows}")
    return numeric


def _validate_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Input contains no paired compounds")

    result = frame.copy().reset_index(drop=True)
    for column in ("pair_id", "parent_smiles", "candidate_smiles"):
        if result[column].isna().any():
            raise ValueError(f"{column} must not contain missing values")
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError(f"{column} must not contain blank values")

    if result["pair_id"].duplicated().any():
        duplicate_ids = sorted(result.loc[result["pair_id"].duplicated(False), "pair_id"].unique())
        raise ValueError(f"pair_id values must be unique; duplicates: {duplicate_ids}")
    identical = result["parent_smiles"].eq(result["candidate_smiles"])
    if identical.any():
        ids = result.loc[identical, "pair_id"].tolist()
        raise ValueError(f"Parent and candidate SMILES must differ; identical pairs: {ids}")

    pair_keys = result.apply(
        lambda row: tuple(sorted((row["parent_smiles"], row["candidate_smiles"]))), axis=1
    )
    if pair_keys.duplicated().any():
        ids = result.loc[pair_keys.duplicated(False), "pair_id"].tolist()
        raise ValueError(f"Duplicate or reversed structure pairs are not allowed: {ids}")

    for column in REQUIRED_COLUMNS[3:]:
        result[column] = _finite_positive(result[column], column)

    prediction_presence = [column in result.columns for column in PREDICTION_COLUMNS]
    if any(prediction_presence) and not all(prediction_presence):
        raise ValueError(
            "Precomputed predictions require both parent_predicted_ic50_um and candidate_predicted_ic50_um"
        )
    if all(prediction_presence):
        for column in PREDICTION_COLUMNS:
            result[column] = _finite_positive(result[column], column)

    mw_presence = [column in result.columns for column in MW_COLUMNS]
    if any(mw_presence) and not all(mw_presence):
        raise ValueError("MW stratification requires both parent_mw and candidate_mw")
    if all(mw_presence):
        for column in MW_COLUMNS:
            result[column] = _finite_positive(result[column], column)
        result["pair_max_mw"] = result[list(MW_COLUMNS)].max(axis=1)

    if "similarity" in result.columns:
        try:
            similarity = pd.to_numeric(result["similarity"], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("similarity must contain only numeric values") from exc
        invalid_similarity = ~np.isfinite(similarity.to_numpy()) | ~similarity.between(0.0, 1.0).to_numpy()
        if invalid_similarity.any():
            rows = (np.flatnonzero(invalid_similarity) + 2).tolist()
            raise ValueError(f"similarity must be finite and within [0, 1]; invalid CSV rows: {rows}")
        result["similarity"] = similarity
    return result


def _prediction_array(values: Sequence[float], expected: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric IC50 values") from exc
    if array.ndim != 1 or len(array) != expected:
        raise ValueError(f"{label} must be a one-dimensional sequence with {expected} values")
    if (~np.isfinite(array) | (array <= 0)).any():
        raise ValueError(f"{label} must contain only finite, positive IC50 values in uM")
    return array


def _attach_predictions(
    frame: pd.DataFrame,
    parent_predictions_ic50_um: Sequence[float] | None,
    candidate_predictions_ic50_um: Sequence[float] | None,
    predictor: Predictor | None,
) -> pd.DataFrame:
    result = frame.copy()
    has_precomputed = all(column in result.columns for column in PREDICTION_COLUMNS)
    has_arrays = parent_predictions_ic50_um is not None or candidate_predictions_ic50_um is not None
    supplied_sources = int(has_precomputed) + int(has_arrays) + int(predictor is not None)
    if supplied_sources > 1:
        raise ValueError("Provide predictions through exactly one source: CSV columns, arrays, or predictor")
    if has_arrays and (parent_predictions_ic50_um is None or candidate_predictions_ic50_um is None):
        raise ValueError("Both parent and candidate prediction arrays are required")

    size = len(result)
    if has_arrays:
        result[PREDICTION_COLUMNS[0]] = _prediction_array(
            parent_predictions_ic50_um,
            size,
            "parent_predictions_ic50_um",  # type: ignore[arg-type]
        )
        result[PREDICTION_COLUMNS[1]] = _prediction_array(
            candidate_predictions_ic50_um,
            size,
            "candidate_predictions_ic50_um",  # type: ignore[arg-type]
        )
    elif predictor is not None:
        ordered_smiles = [*result["parent_smiles"].tolist(), *result["candidate_smiles"].tolist()]
        predicted = _prediction_array(predictor(ordered_smiles), 2 * size, "predictor output")
        result[PREDICTION_COLUMNS[0]] = predicted[:size]
        result[PREDICTION_COLUMNS[1]] = predicted[size:]
    return result


def _direction(delta: pd.Series, threshold: float) -> pd.Series:
    return pd.Series(
        np.select(
            [delta.gt(threshold), delta.lt(-threshold)],
            ["improved", "worsened"],
            default="negligible",
        ),
        index=delta.index,
        dtype="string",
    )


def _optional_float(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _summarize(frame: pd.DataFrame, tie_threshold_log10: float) -> dict[str, Any]:
    measured_counts = frame["measured_direction"].value_counts()
    summary: dict[str, Any] = {
        "n_pairs": int(len(frame)),
        "tie_threshold_log10_ic50": float(tie_threshold_log10),
        "measured_direction_counts": {
            label: int(measured_counts.get(label, 0)) for label in ("improved", "worsened", "negligible")
        },
        "predictions_available": "predicted_direction" in frame.columns,
        "direction_definition": (
            "log10(candidate IC50 uM) - log10(parent IC50 uM); positive is improved "
            "(lower predicted hERG liability)"
        ),
    }
    if "predicted_direction" not in frame.columns:
        summary["prediction_metrics"] = None
        return summary

    predicted_counts = frame["predicted_direction"].value_counts()
    evaluable = frame["measured_direction"].ne("negligible")
    correct = frame.loc[evaluable, "predicted_direction"].eq(frame.loc[evaluable, "measured_direction"])
    measured_abs = frame.loc[evaluable, "measured_delta_log10_ic50"].abs()
    predicted_abs = frame.loc[evaluable, "predicted_delta_log10_ic50"].abs()
    denominator = float(measured_abs.sum())
    capture_ratio = float(predicted_abs.sum() / denominator) if denominator > 0 else math.nan
    per_pair_capture = frame.loc[evaluable, "cliff_capture_ratio"].dropna()
    delta_error = frame["predicted_delta_log10_ic50"] - frame["measured_delta_log10_ic50"]
    summary["prediction_metrics"] = {
        "predicted_direction_counts": {
            label: int(predicted_counts.get(label, 0)) for label in ("improved", "worsened", "negligible")
        },
        "n_direction_evaluable": int(evaluable.sum()),
        "n_direction_correct": int(correct.sum()),
        "directional_accuracy": _optional_float(float(correct.mean()) if len(correct) else math.nan),
        "delta_log10_ic50_mae": float(delta_error.abs().mean()),
        "delta_log10_ic50_bias": float(delta_error.mean()),
        "cliff_capture_ratio": _optional_float(capture_ratio),
        "median_pair_cliff_capture_ratio": _optional_float(
            float(per_pair_capture.median()) if len(per_pair_capture) else math.nan
        ),
    }
    return summary


def stratified_metrics(
    rows: pd.DataFrame,
    *,
    stratify_by: Sequence[str],
    tie_threshold_log10: float,
) -> pd.DataFrame:
    """Summarize any caller-provided categorical strata, including MW/similarity bands."""

    records: list[dict[str, Any]] = []
    for column in stratify_by:
        if column not in rows.columns:
            raise ValueError(f"Requested stratification column is absent: {column}")
        for level, group in rows.groupby(column, observed=True, dropna=False, sort=True):
            summary = _summarize(group, tie_threshold_log10)
            prediction = summary["prediction_metrics"] or {}
            records.append(
                {
                    "stratifier": column,
                    "stratum": "missing" if pd.isna(level) else str(level),
                    "n_pairs": summary["n_pairs"],
                    **{
                        f"measured_{label}": summary["measured_direction_counts"][label]
                        for label in ("improved", "worsened", "negligible")
                    },
                    "n_direction_evaluable": prediction.get("n_direction_evaluable"),
                    "directional_accuracy": prediction.get("directional_accuracy"),
                    "delta_log10_ic50_mae": prediction.get("delta_log10_ic50_mae"),
                    "cliff_capture_ratio": prediction.get("cliff_capture_ratio"),
                }
            )
    return pd.DataFrame.from_records(records)


def add_numeric_band(
    frame: pd.DataFrame,
    *,
    source_column: str,
    output_column: str,
    cutoffs: Sequence[float],
    unit: str = "",
) -> pd.DataFrame:
    """Add explicit caller-selected numeric bands without inventing domain thresholds."""

    if source_column not in frame.columns:
        raise ValueError(f"Cannot band absent column: {source_column}")
    edges = [float(value) for value in cutoffs]
    if not edges or any(not math.isfinite(value) for value in edges):
        raise ValueError("Band cutoffs must contain at least one finite value")
    if edges != sorted(set(edges)):
        raise ValueError("Band cutoffs must be unique and strictly increasing")
    suffix = f" {unit}" if unit else ""
    labels = [f"<={edges[0]:g}{suffix}"]
    labels.extend(
        f"({lower:g}, {upper:g}]{suffix}" for lower, upper in zip(edges[:-1], edges[1:], strict=True)
    )
    labels.append(f">{edges[-1]:g}{suffix}")
    result = frame.copy()
    result[output_column] = pd.cut(
        result[source_column], bins=[-np.inf, *edges, np.inf], labels=labels, include_lowest=True
    )
    return result


def add_rdkit_pair_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Lazily calculate RDKit MW and Morgan/Tanimoto similarity for stratification."""

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import Descriptors, rdFingerprintGenerator
    except ImportError as exc:  # pragma: no cover - project runtime includes RDKit
        raise RuntimeError("RDKit is required only when RDKit pair features are requested") from exc

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    parent_mw: list[float] = []
    candidate_mw: list[float] = []
    similarities: list[float] = []
    invalid: list[str] = []
    for row in frame.itertuples(index=False):
        parent = Chem.MolFromSmiles(row.parent_smiles)
        candidate = Chem.MolFromSmiles(row.candidate_smiles)
        if parent is None or candidate is None:
            invalid.append(str(row.pair_id))
            parent_mw.append(math.nan)
            candidate_mw.append(math.nan)
            similarities.append(math.nan)
            continue
        parent_mw.append(float(Descriptors.MolWt(parent)))
        candidate_mw.append(float(Descriptors.MolWt(candidate)))
        similarities.append(
            float(
                DataStructs.TanimotoSimilarity(
                    generator.GetFingerprint(parent), generator.GetFingerprint(candidate)
                )
            )
        )
    if invalid:
        raise ValueError(f"RDKit could not parse SMILES for pair_id values: {invalid}")
    result = frame.copy()
    result["parent_mw"] = parent_mw
    result["candidate_mw"] = candidate_mw
    result["pair_max_mw"] = np.maximum(parent_mw, candidate_mw)
    result["similarity"] = similarities
    return result


def evaluate_paired_edits(
    pairs: pd.DataFrame,
    *,
    parent_predictions_ic50_um: Sequence[float] | None = None,
    candidate_predictions_ic50_um: Sequence[float] | None = None,
    predictor: Predictor | None = None,
    tie_threshold_log10: float = 0.0,
    stratify_by: Sequence[str] = (),
) -> PairedEvaluation:
    """Evaluate paired IC50 directions without importing or fitting a prediction model.

    The optional predictor is called once with all parent SMILES followed by all
    candidate SMILES. It must return finite, positive IC50 predictions in uM in
    the same order. Arrays are interpreted in current DataFrame row order.
    """

    if not math.isfinite(tie_threshold_log10) or tie_threshold_log10 < 0:
        raise ValueError("tie_threshold_log10 must be finite and non-negative")
    rows = _validate_pairs(pairs)
    rows = _attach_predictions(
        rows,
        parent_predictions_ic50_um,
        candidate_predictions_ic50_um,
        predictor,
    )
    rows["measured_delta_log10_ic50"] = np.log10(rows["candidate_measured_ic50_um"]) - np.log10(
        rows["parent_measured_ic50_um"]
    )
    rows["measured_direction"] = _direction(rows["measured_delta_log10_ic50"], tie_threshold_log10)

    if all(column in rows.columns for column in PREDICTION_COLUMNS):
        rows["predicted_delta_log10_ic50"] = np.log10(rows[PREDICTION_COLUMNS[1]]) - np.log10(
            rows[PREDICTION_COLUMNS[0]]
        )
        rows["predicted_direction"] = _direction(rows["predicted_delta_log10_ic50"], tie_threshold_log10)
        evaluable = rows["measured_direction"].ne("negligible")
        rows["direction_evaluable"] = evaluable
        direction_correct = rows["predicted_direction"].eq(rows["measured_direction"])
        rows["direction_correct"] = direction_correct.where(evaluable).astype("boolean")
        measured_abs = rows["measured_delta_log10_ic50"].abs().to_numpy()
        predicted_abs = rows["predicted_delta_log10_ic50"].abs().to_numpy()
        rows["cliff_capture_ratio"] = np.divide(
            predicted_abs,
            measured_abs,
            out=np.full(len(rows), np.nan),
            where=measured_abs > 0,
        )

    summary = _summarize(rows, tie_threshold_log10)
    strata = stratified_metrics(rows, stratify_by=stratify_by, tie_threshold_log10=tie_threshold_log10)
    return PairedEvaluation(rows=rows, summary=summary, strata=strata)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tie-threshold-log10",
        type=float,
        default=0.0,
        help="Absolute log10(IC50) delta treated as negligible (default: exact ties only)",
    )
    parser.add_argument(
        "--compute-rdkit-features",
        action="store_true",
        help="Calculate RDKit MolWt and Morgan/Tanimoto similarity for each pair",
    )
    parser.add_argument("--mw-cutoffs", type=float, nargs="+", help="Explicit pair-max-MW band cutoffs")
    parser.add_argument("--similarity-cutoffs", type=float, nargs="+", help="Explicit Tanimoto band cutoffs")
    parser.add_argument(
        "--stratify-by",
        action="append",
        default=[],
        help="Categorical column to summarize; repeat for multiple columns",
    )
    return parser.parse_args()


def _write_outputs(evaluation: PairedEvaluation, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation.rows.to_csv(output_dir / "paired_edit_results.csv", index=False)
    (output_dir / "paired_edit_summary.json").write_text(
        json.dumps(evaluation.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not evaluation.strata.empty:
        evaluation.strata.to_csv(output_dir / "paired_edit_stratified_metrics.csv", index=False)


def main() -> None:
    args = _parse_args()
    frame = pd.read_csv(args.input_csv)
    frame = _validate_pairs(frame)
    if args.compute_rdkit_features:
        frame = add_rdkit_pair_features(frame)

    strata = list(args.stratify_by)
    if args.mw_cutoffs:
        if "pair_max_mw" not in frame.columns:
            raise ValueError(
                "MW cutoffs require parent_mw/candidate_mw input columns or --compute-rdkit-features"
            )
        frame = add_numeric_band(
            frame,
            source_column="pair_max_mw",
            output_column="mw_band",
            cutoffs=args.mw_cutoffs,
            unit="Da",
        )
        strata.append("mw_band")
    if args.similarity_cutoffs:
        if "similarity" not in frame.columns:
            raise ValueError(
                "Similarity cutoffs require a similarity input column or --compute-rdkit-features"
            )
        frame = add_numeric_band(
            frame,
            source_column="similarity",
            output_column="similarity_band",
            cutoffs=args.similarity_cutoffs,
        )
        strata.append("similarity_band")

    evaluation = evaluate_paired_edits(
        frame,
        tie_threshold_log10=args.tie_threshold_log10,
        stratify_by=tuple(dict.fromkeys(strata)),
    )
    _write_outputs(evaluation, args.output_dir)
    print(
        json.dumps(
            {
                "n_pairs": evaluation.summary["n_pairs"],
                "predictions_available": evaluation.summary["predictions_available"],
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
