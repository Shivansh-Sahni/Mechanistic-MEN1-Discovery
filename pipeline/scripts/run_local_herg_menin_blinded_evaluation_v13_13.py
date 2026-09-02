#!/usr/bin/env python3
"""Predeclare, seal, and score the blinded V13.10 Menin hERG panel.

The ``protocol`` stage must be run while the outcome-release CSV is blank.  It
hash-locks the panel, template, metrics, exclusions, and decision rules.  A
future ``seal`` stage validates a completed outcome file and copies outcomes
into a prediction-free sealed artifact.  Only ``score`` joins that artifact to
the frozen predictions.  General model promotion is never authorized by this
small, deliberately enriched challenge panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

SCHEMA_VERSION = "platform-local-herg-menin-blinded-evaluation-v13.13/1.0"
OUTCOME_COLUMNS = ("measured_ic10_nm", "measured_ic30_nm", "measured_ic50_nm")
ASSAY_COLUMNS = (
    "assay_temperature_c",
    "cell_system",
    "voltage_protocol_id",
    "biological_replicate_count",
    "qc_pass",
    "assay_notes",
)
STATIC_COLUMNS = ("blinded_sample_id", "structure_id", "selection_category")
CLASS_NAMES = ("Safe", "Moderate", "Potent")
SAFE_PIC50 = 6.0 - math.log10(30.0)  # IC50 > 30 uM
POTENT_PIC50 = 6.0  # IC50 < 1 uM


class EvaluationError(RuntimeError):
    """Raised when the blinded workflow contract is violated."""


def _canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any], field: str) -> dict[str, Any]:
    body = dict(payload)
    body.pop(field, None)
    body[field] = hashlib.sha256(_canonical(body)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(body))
    temporary.replace(path)
    return body


def _read_json(path: Path, field: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise EvaluationError(f"expected JSON object: {path}")
    if field:
        expected = payload.get(field)
        body = dict(payload)
        body.pop(field, None)
        if hashlib.sha256(_canonical(body)).hexdigest() != expected:
            raise EvaluationError(f"self-hash mismatch: {path}")
    return payload


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = set(columns) - set(frame)
    if missing:
        raise EvaluationError(f"{name} lacks required columns: {sorted(missing)}")


def _all_outcomes_blank(frame: pd.DataFrame) -> bool:
    numeric = frame[list(OUTCOME_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return bool(numeric.isna().all().all())


def _static_mapping_sha(frame: pd.DataFrame) -> str:
    records = (
        frame[list(STATIC_COLUMNS)].fillna("").astype(str).sort_values("blinded_sample_id").to_dict("records")
    )
    return hashlib.sha256(_canonical(records)).hexdigest()


def _exact_design(n: int, alpha: float) -> dict[str, Any]:
    critical = next(
        value for value in range(n + 1) if binomtest(value, n, 0.5, alternative="greater").pvalue < alpha
    )
    alternatives = (0.6, 0.7, 0.75, 0.8, 0.9)
    return {
        "maximum_disagreement_rows": n,
        "critical_corrected_count_if_all_are_informative": critical,
        "critical_corrected_fraction": critical / n,
        "exact_power_by_true_corrected_probability": {
            str(value): float(binom.sf(critical - 1, n, value)) for value in alternatives
        },
        "interpretation": (
            "with 13 informative disagreements the primary test needs at least 10 corrected; "
            "power is only 0.584 when the true corrected probability is 0.75"
        ),
    }


def _protocol(panel_path: Path, outcome_path: Path, output: Path) -> dict[str, Any]:
    panel = pd.read_parquet(panel_path)
    outcomes = pd.read_csv(outcome_path, dtype=str)
    _require_columns(outcomes, (*STATIC_COLUMNS, *OUTCOME_COLUMNS, *ASSAY_COLUMNS), "outcomes")
    if len(panel) != 24 or len(outcomes) != 24:
        raise EvaluationError("the frozen challenge panel and blank template must each have 24 rows")
    if not _all_outcomes_blank(outcomes):
        raise EvaluationError("protocol must be locked before any measured outcome is entered")
    if outcomes.blinded_sample_id.duplicated().any() or outcomes.structure_id.duplicated().any():
        raise EvaluationError("blank outcome identifiers must be unique")
    if set(outcomes.structure_id) != set(panel.structure_id):
        raise EvaluationError("blank template does not match the frozen panel structures")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "preanalysis_locked_before_outcomes",
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha(Path(__file__).resolve()),
        },
        "panel": {
            "path": str(panel_path),
            "sha256": _sha(panel_path),
            "rows": len(panel),
            "unique_series": int(panel.series_id.nunique()),
            "router_hybrid_disagreements": int(panel.router_hybrid_disagreement.sum()),
        },
        "blank_outcome_template": {
            "path": str(outcome_path),
            "sha256": _sha(outcome_path),
            "rows": len(outcomes),
            "outcomes_present_when_locked": False,
            "static_blinded_mapping_sha256": _static_mapping_sha(outcomes),
        },
        "eligibility": {
            "primary_analysis": "rows with qc_pass=true and all three positive finite endpoints",
            "physical_order_required": "IC10 <= IC30 <= IC50 concentrations",
            "protocol_deviations_reported_not_silently_removed": [
                "assay temperature outside 35-38 C",
                "biological replicate count below 3",
                "missing cell system or voltage protocol identifier",
            ],
            "only_exclusion_after_unblinding": "pre-entered assay qc_pass=false",
        },
        "outcomes_and_thresholds": {
            "conversion": "pICx = 9 - log10(ICx in nM)",
            "ternary_ic50": {
                "Safe": f"pIC50 < {SAFE_PIC50:.12f} (IC50 > 30 uM)",
                "Moderate": f"{SAFE_PIC50:.12f} <= pIC50 <= 6.0 (1-30 uM)",
                "Potent": "pIC50 > 6.0 (IC50 < 1 uM)",
                "matches_frozen_platform_tiers": True,
            },
        },
        "primary_receptor_hypothesis": {
            "population": "the 13 preselected ligand-router/hybrid disagreements with qc_pass=true",
            "estimand": "hybrid-corrected count versus hybrid-degraded count; wrong-to-other-wrong excluded from the exact sign test",
            "test": "one-sided exact binomial test, H1 corrected probability > 0.5",
            "alpha": 0.05,
            "design_operating_characteristics": _exact_design(13, 0.05),
        },
        "key_secondary_classification": {
            "population": "all qc_pass=true panel rows",
            "metrics": [
                "accuracy",
                "balanced accuracy when all three observed classes are present",
                "macro-F1 over Safe/Moderate/Potent",
                "per-class recall",
                "Potent-to-Safe count and rate",
            ],
            "frozen_safety_gate": "hybrid Potent-to-Safe <= 0.05 absolute and <= router + 0.03",
        },
        "regression": {
            "endpoints": ["IC10", "IC30", "IC50"],
            "metrics": ["MAE in pIC units", "RMSE", "median absolute error", "bias", "Spearman"],
            "comparisons": [
                "projected versus raw direct IC10/IC30/IC50 predictions",
                "V9 mixed-IC50 versus direct empirical IC50",
                "frozen 80/90/95% V9 IC50 interval coverage",
                "frozen 90% direct-endpoint interval coverage",
            ],
            "no_model_feature_threshold_or_interval_tuning": True,
        },
        "model_challenge_success_rule": {
            "all_required": [
                "one-sided disagreement exact-binomial p < 0.05 and corrected_count > degraded_count",
                "all-panel hybrid balanced accuracy > router balanced accuracy",
                "all-panel hybrid macro-F1 >= router macro-F1",
                "frozen Potent-to-Safe safety gate passes",
            ],
            "general_promotion_authorized_even_if_passed": False,
            "interpretation_if_passed": "supports another independent naturally imbalanced prospective study only",
        },
        "multiplicity_and_scope": {
            "single_primary_test": True,
            "all_other_tests": "descriptive with exact denominators and uncertainty where applicable",
            "panel_is_enriched": True,
            "clinical_or_regulatory_claim_authorized": False,
        },
    }
    result = _write_json(output / "preanalysis_protocol.json", contract, "protocol_sha256")
    markdown = f"""# hERG V13.13 blinded Menin preanalysis protocol

Status: **{result["status"]}**  
Protocol SHA-256: `{result["protocol_sha256"]}`

The V13.10 panel contains 24 compounds, 24 series, and 13 frozen ligand-router/hybrid
disagreements. Every measured outcome field was blank when this protocol was locked.

## Primary hypothesis

On QC-passing disagreement rows, count hybrid-corrected versus hybrid-degraded decisions.
Use a one-sided exact binomial test at alpha 0.05; wrong-to-other-wrong changes are reported
but excluded from that sign test.

## Required gates

Model-challenge success additionally requires higher all-panel balanced accuracy, non-worse
macro-F1, and the frozen Potent-to-Safe safety gate. General promotion remains unauthorized
even if every challenge criterion passes.

## Future sequence

1. Complete all IC10/IC30/IC50 and assay-QC fields in the blinded release CSV.
2. Run `--stage seal` before joining outcomes to predictions.
3. Run `--stage score` only from the sealed artifact.

The enriched panel is a discrimination experiment, not a natural-prevalence safety validation
or a clinical/regulatory dataset.
"""
    (output / "REPORT.md").write_text(markdown)
    return result


def _parse_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise EvaluationError(f"qc_pass is not a recognized boolean: {value!r}")


def _seal(panel_path: Path, outcome_path: Path, output: Path) -> dict[str, Any]:
    protocol = _read_json(output / "preanalysis_protocol.json", "protocol_sha256")
    if _sha(Path(__file__).resolve()) != protocol["implementation"]["sha256"]:
        raise EvaluationError("evaluation implementation changed after protocol lock")
    if _sha(panel_path) != protocol["panel"]["sha256"]:
        raise EvaluationError("frozen panel changed after protocol lock")
    panel = pd.read_parquet(panel_path)
    outcomes = pd.read_csv(outcome_path, dtype=str)
    _require_columns(outcomes, (*STATIC_COLUMNS, *OUTCOME_COLUMNS, *ASSAY_COLUMNS), "outcomes")
    if len(outcomes) != protocol["panel"]["rows"]:
        raise EvaluationError("completed outcome row count differs from the locked panel")
    if outcomes.blinded_sample_id.duplicated().any() or outcomes.structure_id.duplicated().any():
        raise EvaluationError("completed outcome identifiers must be unique")
    if set(outcomes.structure_id) != set(panel.structure_id):
        raise EvaluationError("completed outcomes do not match the frozen panel")
    if _static_mapping_sha(outcomes) != protocol["blank_outcome_template"]["static_blinded_mapping_sha256"]:
        raise EvaluationError("blinded ID/structure/category mapping changed after protocol lock")
    if _all_outcomes_blank(outcomes):
        raise EvaluationError("cannot seal the still-blank outcome template")
    forbidden = [
        column
        for column in outcomes
        if any(token in column.lower() for token in ("predict", "probability", "rank", "hybrid"))
    ]
    if forbidden:
        raise EvaluationError(f"outcome file contains forbidden model fields: {forbidden}")
    numeric_columns = (*OUTCOME_COLUMNS, "assay_temperature_c", "biological_replicate_count")
    for column in numeric_columns:
        outcomes[column] = pd.to_numeric(outcomes[column], errors="coerce")
    if outcomes[list(OUTCOME_COLUMNS)].isna().any().any():
        raise EvaluationError("all three measured endpoints are required for every panel row")
    if (outcomes[list(OUTCOME_COLUMNS)] <= 0).any().any():
        raise EvaluationError("measured concentrations must be positive")
    order = outcomes.measured_ic10_nm.le(outcomes.measured_ic30_nm) & outcomes.measured_ic30_nm.le(
        outcomes.measured_ic50_nm
    )
    if not order.all():
        bad = outcomes.loc[~order, "blinded_sample_id"].tolist()
        raise EvaluationError(f"physical endpoint order failed for blinded samples: {bad}")
    outcomes["qc_pass"] = outcomes.qc_pass.map(_parse_bool)
    deviations = pd.DataFrame(
        {
            "blinded_sample_id": outcomes.blinded_sample_id,
            "temperature_outside_35_38c": ~outcomes.assay_temperature_c.between(35, 38),
            "biological_replicates_below_3": outcomes.biological_replicate_count.lt(3),
            "missing_cell_system": outcomes.cell_system.fillna("").str.strip().eq(""),
            "missing_voltage_protocol_id": outcomes.voltage_protocol_id.fillna("").str.strip().eq(""),
            "qc_pass": outcomes.qc_pass,
        }
    )
    sealed_path = output / "sealed/outcomes.parquet"
    sealed_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_parquet(sealed_path, index=False, compression="zstd")
    deviation_path = output / "sealed/protocol_deviations.parquet"
    deviations.to_parquet(deviation_path, index=False, compression="zstd")
    seal = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "outcomes_sealed_before_prediction_join",
        "protocol_sha256": protocol["protocol_sha256"],
        "completed_outcome_file_sha256": _sha(outcome_path),
        "sealed_outcomes_sha256": _sha(sealed_path),
        "protocol_deviations_sha256": _sha(deviation_path),
        "rows": len(outcomes),
        "qc_pass_rows": int(outcomes.qc_pass.sum()),
        "predictions_or_ranks_present": False,
    }
    return _write_json(output / "sealed/outcome_seal.json", seal, "seal_sha256")


def _pic(values_nm: pd.Series) -> np.ndarray:
    return 9.0 - np.log10(values_nm.to_numpy(float))


def _tier(pic50: np.ndarray) -> np.ndarray:
    return np.where(pic50 < SAFE_PIC50, 0, np.where(pic50 > POTENT_PIC50, 2, 1)).astype(int)


def _regression(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    residual = prediction - y
    correlation = spearmanr(y, prediction).statistic if len(y) >= 3 else math.nan
    return {
        "n": len(y),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "bias_prediction_minus_observed": float(np.mean(residual)),
        "spearman": float(correlation),
    }


def _classification(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    recalls = {
        CLASS_NAMES[index]: float(np.mean(prediction[y == index] == index))
        for index in (0, 1, 2)
        if np.any(y == index)
    }
    result = {
        "n": len(y),
        "observed_classes": sorted(set(y.tolist())),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
        "recall": recalls,
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
    }
    result["balanced_accuracy"] = (
        float(balanced_accuracy_score(y, prediction)) if set(y) == {0, 1, 2} else None
    )
    potent = y == 2
    if potent.any():
        result["potent_predicted_safe_count"] = int(np.sum(prediction[potent] == 0))
        result["observed_potent_count"] = int(potent.sum())
        result["potent_predicted_safe_rate"] = float(np.mean(prediction[potent] == 0))
    return result


def _coverage(frame: pd.DataFrame, observed: np.ndarray, prefix: str, levels: list[int]) -> dict[str, Any]:
    result = {}
    for level in levels:
        lower_column = f"{prefix}_interval{level}_lower_pic50"
        upper_column = f"{prefix}_interval{level}_upper_pic50"
        if lower_column not in frame:
            lower_column = f"{prefix}_interval{level}_lower_picx"
            upper_column = f"{prefix}_interval{level}_upper_picx"
        lower = frame[lower_column].to_numpy(float)
        upper = frame[upper_column].to_numpy(float)
        result[str(level)] = {
            "n": len(frame),
            "covered": int(np.sum((observed >= lower) & (observed <= upper))),
            "coverage": float(np.mean((observed >= lower) & (observed <= upper))),
        }
    return result


def _score(panel_path: Path, output: Path) -> dict[str, Any]:
    protocol = _read_json(output / "preanalysis_protocol.json", "protocol_sha256")
    if _sha(Path(__file__).resolve()) != protocol["implementation"]["sha256"]:
        raise EvaluationError("evaluation implementation changed after protocol lock")
    seal = _read_json(output / "sealed/outcome_seal.json", "seal_sha256")
    sealed_path = output / "sealed/outcomes.parquet"
    if _sha(sealed_path) != seal["sealed_outcomes_sha256"]:
        raise EvaluationError("sealed outcomes changed before scoring")
    if _sha(panel_path) != protocol["panel"]["sha256"]:
        raise EvaluationError("frozen panel changed before scoring")
    panel = pd.read_parquet(panel_path)
    outcomes = pd.read_parquet(sealed_path)
    frame = panel.merge(outcomes, on="structure_id", validate="one_to_one")
    frame = frame.loc[frame.qc_pass.astype(bool)].reset_index(drop=True)
    if frame.empty:
        raise EvaluationError("no qc_pass=true rows remain for scoring")
    observed = {
        endpoint: _pic(frame[f"measured_{endpoint.lower()}_nm"]) for endpoint in ("IC10", "IC30", "IC50")
    }
    y = _tier(observed["IC50"])
    router = frame.ligand_router_prediction.to_numpy(int)
    hybrid = frame.hybrid_prediction.to_numpy(int)
    router_metrics = _classification(y, router)
    hybrid_metrics = _classification(y, hybrid)
    disagreement = frame.router_hybrid_disagreement.to_numpy(bool)
    corrected = disagreement & (hybrid == y) & (router != y)
    degraded = disagreement & (hybrid != y) & (router == y)
    wrong_to_wrong = disagreement & (hybrid != y) & (router != y)
    informative = int(corrected.sum() + degraded.sum())
    exact_p = (
        float(binomtest(int(corrected.sum()), informative, 0.5, alternative="greater").pvalue)
        if informative
        else 1.0
    )
    safety_passed = bool(
        hybrid_metrics.get("potent_predicted_safe_rate", 0.0) <= 0.05
        and hybrid_metrics.get("potent_predicted_safe_rate", 0.0)
        <= router_metrics.get("potent_predicted_safe_rate", 0.0) + 0.03
    )
    ba_improved = bool(
        hybrid_metrics["balanced_accuracy"] is not None
        and router_metrics["balanced_accuracy"] is not None
        and hybrid_metrics["balanced_accuracy"] > router_metrics["balanced_accuracy"]
    )
    challenge_success = bool(
        corrected.sum() > degraded.sum()
        and exact_p < protocol["primary_receptor_hypothesis"]["alpha"]
        and ba_improved
        and hybrid_metrics["macro_f1"] >= router_metrics["macro_f1"]
        and safety_passed
    )
    regression = {}
    for endpoint in ("IC10", "IC30", "IC50"):
        lower = endpoint.lower()
        regression[endpoint] = {
            "projected_direct": _regression(
                observed[endpoint], frame[f"empirical_{lower}_predicted_picx"].to_numpy(float)
            ),
            "raw_direct": _regression(
                observed[endpoint], frame[f"empirical_{lower}_raw_predicted_picx"].to_numpy(float)
            ),
            "direct_interval90": _coverage(frame, observed[endpoint], f"empirical_{lower}", [90])["90"],
        }
    regression["IC50"]["v9_mixed"] = _regression(
        observed["IC50"], frame.v9_mixed_ic50_predicted_pic50.to_numpy(float)
    )
    regression["IC50"]["v9_mixed_intervals"] = _coverage(
        frame, observed["IC50"], "v9_mixed_ic50", [80, 90, 95]
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete_blinded_model_challenge_score",
        "protocol_sha256": protocol["protocol_sha256"],
        "outcome_seal_sha256": seal["seal_sha256"],
        "eligible_rows": len(frame),
        "excluded_qc_fail_rows": int(len(outcomes) - len(frame)),
        "primary_receptor_hypothesis": {
            "preselected_disagreement_rows_eligible": int(disagreement.sum()),
            "corrected_count": int(corrected.sum()),
            "degraded_count": int(degraded.sum()),
            "wrong_to_different_wrong_count": int(wrong_to_wrong.sum()),
            "informative_corrected_or_degraded": informative,
            "one_sided_exact_binomial_p_value": exact_p,
        },
        "classification": {"ligand_router": router_metrics, "frozen_hybrid": hybrid_metrics},
        "regression": regression,
        "decision": {
            "model_challenge_success": challenge_success,
            "safety_gate_passed": safety_passed,
            "general_promotion_authorized": False,
            "interpretation": (
                "a passing enriched challenge supports another independent naturally imbalanced "
                "prospective study; it cannot erase V13.8 nonconfirmation or authorize deployment"
            ),
        },
        "external_outcome_used_for_model_feature_threshold_or_interval_tuning": False,
    }
    result = _write_json(output / "analysis_report.json", report, "report_sha256")
    scored_path = output / "scored/blinded_joined_results.parquet"
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(scored_path, index=False, compression="zstd")
    _write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "complete",
            "protocol_sha256": protocol["protocol_sha256"],
            "outcome_seal_sha256": seal["seal_sha256"],
            "analysis_report_sha256": _sha(output / "analysis_report.json"),
            "scored_results_sha256": _sha(scored_path),
        },
        "manifest_sha256",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("protocol", "seal", "score"), required=True)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path(
            "research/local_runs/herg_menin_receptor_prioritization_v13_10/analysis/"
            "blinded_assay_challenge_panel.parquet"
        ),
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=Path(
            "research/local_runs/herg_menin_receptor_prioritization_v13_10/analysis/"
            "blinded_herg_outcome_release.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/local_runs/herg_menin_blinded_evaluation_v13_13"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.stage == "protocol":
        result = _protocol(args.panel, args.outcomes, args.output)
    elif args.stage == "seal":
        result = _seal(args.panel, args.outcomes, args.output)
    else:
        result = _score(args.panel, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
