#!/usr/bin/env python3
"""Post-hoc heterogeneity audit for the two frozen V13 receptor evaluations.

This analysis never fits a model, changes a threshold, or modifies the V13.6
policy.  It quantifies the campaign interaction after the sealed V13.7 and
V13.8 evaluations were complete so that a favorable first external result is
not mistaken for a generally replicated result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import balanced_accuracy_score, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_local_herg_external_validation_v13_7 as v137  # noqa: E402
import run_local_herg_receptor_ensemble_campaign_v13 as v13  # noqa: E402

SCHEMA_VERSION = "platform-local-herg-receptor-heterogeneity-v13.12/1.0"
SEED = 20260821
CLASS_NAMES = tuple(v13.CLASS_NAMES)
ROUTER_PROBABILITY_COLUMNS = tuple(f"lgbm_rdkit2d_morgan__probability_{name.lower()}" for name in CLASS_NAMES)
RECEPTOR_PROBABILITY_COLUMNS = tuple(
    f"primary_frozen_f649__probability_{name.lower()}" for name in CLASS_NAMES
)
CONTINUOUS_COLUMNS = {
    "nearest_v9_morgan_tanimoto": "nearest V9 Morgan similarity",
    "dock__8ZYO__contact_649_count": "F649 contact count",
    "dock__8ZYO__affinity": "Vina affinity (kcal/mol)",
    "dock__8ZYO__ligand_efficiency": "Vina ligand efficiency",
    "primary_frozen_f649__probability_potent": "receptor-model Potent probability",
    "lgbm_rdkit2d_morgan__probability_potent": "ligand-router Potent probability",
}


class AuditError(RuntimeError):
    """Raised when a sealed source artifact is incomplete or inconsistent."""


def _canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_self_hashed(path: Path, field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    expected = payload.get(field)
    body = dict(payload)
    body.pop(field, None)
    if hashlib.sha256(_canonical(body)).hexdigest() != expected:
        raise AuditError(f"self-hash mismatch: {path}")
    return payload


def _write_self_hashed(path: Path, payload: dict[str, Any], field: str) -> dict[str, Any]:
    body = dict(payload)
    body.pop(field, None)
    body[field] = hashlib.sha256(_canonical(body)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(body))
    temporary.replace(path)
    return body


def _verify_prediction_seal(root: Path) -> dict[str, Any]:
    seal = _read_self_hashed(root / "predictions/receptor_prediction_seal.json", "seal_sha256")
    prediction_path = root / "predictions/receptor_predictions_before_score.parquet"
    if _sha(prediction_path) != seal["prediction_sha256"]:
        raise AuditError(f"receptor prediction seal mismatch: {root}")
    _read_self_hashed(root / "analysis_report.json", "report_sha256")
    return seal


def _load_campaign(root: Path, name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    seal = _verify_prediction_seal(root)
    receptor = pd.read_parquet(root / "predictions/receptor_predictions_before_score.parquet")
    labels = pd.read_parquet(root / "sealed/labels.parquet")
    if "set_name" in labels:
        labels = labels.loc[labels.set_name.isin(v137.QUANTITATIVE_SETS)]
    labels = labels.groupby("external_structure_id", as_index=False).agg(
        true_pic50_m=("true_pic50_m", "median")
    )
    required = {
        "external_structure_id",
        "scaffold_group_id",
        "hybrid_prediction",
        *ROUTER_PROBABILITY_COLUMNS,
        *RECEPTOR_PROBABILITY_COLUMNS,
        *CONTINUOUS_COLUMNS,
    }
    missing = required - set(receptor)
    if missing:
        raise AuditError(f"{name} receptor predictions lack: {sorted(missing)}")
    frame = receptor.merge(labels, on="external_structure_id", validate="one_to_one")
    if len(frame) != len(receptor):
        raise AuditError(f"{name} is missing sealed outcomes")
    router_probability = frame[list(ROUTER_PROBABILITY_COLUMNS)].to_numpy(float)
    receptor_probability = frame[list(RECEPTOR_PROBABILITY_COLUMNS)].to_numpy(float)
    if not np.allclose(router_probability.sum(axis=1), 1.0, atol=1e-6):
        raise AuditError(f"{name} router probabilities are not normalized")
    if not np.allclose(receptor_probability.sum(axis=1), 1.0, atol=1e-6):
        raise AuditError(f"{name} receptor probabilities are not normalized")
    frame = frame.assign(
        campaign=name,
        true_class=v137._tier(frame.true_pic50_m.to_numpy(float)),  # noqa: SLF001
        router_prediction=np.argmax(router_probability, axis=1).astype(int),
        hybrid_prediction=frame.hybrid_prediction.astype(int),
    )
    if set(frame.true_class) != {0, 1, 2}:
        raise AuditError(f"{name} does not contain all three observed classes")
    return frame, {
        "root": str(root),
        "analysis_report_sha256": _sha(root / "analysis_report.json"),
        "receptor_prediction_sha256": seal["prediction_sha256"],
        "receptor_prediction_seal_sha256": seal["seal_sha256"],
    }


def _metric_delta(y: np.ndarray, router: np.ndarray, hybrid: np.ndarray) -> tuple[float, float]:
    ba = balanced_accuracy_score(y, hybrid) - balanced_accuracy_score(y, router)
    macro = f1_score(y, hybrid, labels=[0, 1, 2], average="macro", zero_division=0) - f1_score(
        y, router, labels=[0, 1, 2], average="macro", zero_division=0
    )
    return float(ba), float(macro)


def _campaign_summary(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame.true_class.to_numpy(int)
    router = frame.router_prediction.to_numpy(int)
    hybrid = frame.hybrid_prediction.to_numpy(int)
    ba_delta, macro_delta = _metric_delta(y, router, hybrid)
    class_rows: dict[str, Any] = {}
    for index, name in enumerate(CLASS_NAMES):
        keep = y == index
        router_recall = float(np.mean(router[keep] == index))
        hybrid_recall = float(np.mean(hybrid[keep] == index))
        class_rows[name] = {
            "n": int(keep.sum()),
            "prevalence": float(keep.mean()),
            "router_recall": router_recall,
            "hybrid_recall": hybrid_recall,
            "delta_recall": hybrid_recall - router_recall,
        }
    corrected = (hybrid == y) & (router != y)
    degraded = (hybrid != y) & (router == y)
    changed = hybrid != router
    transitions = pd.crosstab(
        pd.Categorical(router, categories=[0, 1, 2]),
        pd.Categorical(hybrid, categories=[0, 1, 2]),
        dropna=False,
    ).to_numpy(int)
    transition_outcomes = []
    for router_class in (0, 1, 2):
        for hybrid_class in (0, 1, 2):
            keep = (router == router_class) & (hybrid == hybrid_class)
            if not keep.any():
                continue
            true_counts = np.bincount(y[keep], minlength=3)
            transition_outcomes.append(
                {
                    "router_class": CLASS_NAMES[router_class],
                    "hybrid_class": CLASS_NAMES[hybrid_class],
                    "n": int(keep.sum()),
                    "true_class_counts": {CLASS_NAMES[index]: int(true_counts[index]) for index in (0, 1, 2)},
                    "corrected_count": int(np.sum(corrected & keep)),
                    "degraded_count": int(np.sum(degraded & keep)),
                    "wrong_to_different_wrong_count": int(np.sum((router != y) & (hybrid != y) & keep)),
                }
            )
    return {
        "n": len(frame),
        "unique_scaffolds": int(frame.scaffold_group_id.nunique()),
        "class_composition": class_rows,
        "delta_balanced_accuracy_hybrid_minus_router": ba_delta,
        "delta_macro_f1_hybrid_minus_router": macro_delta,
        "decision_changes": {
            "changed_count": int(changed.sum()),
            "changed_rate": float(changed.mean()),
            "corrected_count": int(corrected.sum()),
            "degraded_count": int(degraded.sum()),
            "wrong_to_different_wrong_count": int(np.sum(changed & (router != y) & (hybrid != y))),
            "net_correct_count": int(corrected.sum() - degraded.sum()),
            "corrected_among_changed_rate": float(corrected.sum() / changed.sum()),
            "router_to_hybrid_matrix": transitions.tolist(),
            "matrix_order": list(CLASS_NAMES),
            "transition_outcomes": transition_outcomes,
        },
    }


def _decision_change_yield(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    left = first["decision_changes"]
    right = second["decision_changes"]
    table = np.asarray(
        [
            [left["corrected_count"], left["changed_count"] - left["corrected_count"]],
            [right["corrected_count"], right["changed_count"] - right["corrected_count"]],
        ],
        dtype=int,
    )
    test = fisher_exact(table, alternative="two-sided")
    return {
        "table_rows": ["V13.7_2025", "V13.8_2026"],
        "table_columns": ["corrected", "not_corrected"],
        "table": table.tolist(),
        "corrected_among_changed_rate_difference_v13_7_minus_v13_8": float(
            left["corrected_among_changed_rate"] - right["corrected_among_changed_rate"]
        ),
        "fisher_exact_odds_ratio": float(test.statistic),
        "fisher_exact_two_sided_p_value_descriptive": float(test.pvalue),
        "scope": (
            "post-hoc row-level descriptor; the scaffold-cluster interaction bootstrap is the "
            "preferred dependence-aware uncertainty analysis"
        ),
    }


def _sample_within_class(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    y = frame.true_class.to_numpy(int)
    return np.concatenate(
        [
            rng.choice(rows, size=len(rows), replace=True)
            for value in (0, 1, 2)
            for rows in [np.flatnonzero(y == value)]
        ]
    )


def _sample_scaffolds(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    groups = [group.index.to_numpy(int) for _, group in frame.groupby("scaffold_group_id")]
    for _ in range(1000):
        chosen = rng.choice(len(groups), size=len(groups), replace=True)
        rows = np.concatenate([groups[index] for index in chosen])
        if set(frame.true_class.to_numpy(int)[rows]) == {0, 1, 2}:
            return rows
    raise AuditError("could not produce a three-class scaffold bootstrap replicate")


def _interaction_bootstrap(
    first: pd.DataFrame,
    second: pd.DataFrame,
    replicates: int,
    mode: str,
) -> dict[str, Any]:
    if replicates < 100:
        raise AuditError("at least 100 interaction bootstrap replicates are required")
    rng = np.random.default_rng(SEED + (0 if mode == "within_class" else 1))
    values = np.empty((replicates, 2), dtype=float)
    sampler = _sample_within_class if mode == "within_class" else _sample_scaffolds
    for iteration in range(replicates):
        deltas = []
        for frame in (first, second):
            rows = sampler(frame, rng)
            deltas.append(
                _metric_delta(
                    frame.true_class.to_numpy(int)[rows],
                    frame.router_prediction.to_numpy(int)[rows],
                    frame.hybrid_prediction.to_numpy(int)[rows],
                )
            )
        values[iteration] = np.subtract(deltas[0], deltas[1])
    observed_first = _metric_delta(first.true_class, first.router_prediction, first.hybrid_prediction)
    observed_second = _metric_delta(second.true_class, second.router_prediction, second.hybrid_prediction)
    observed = np.subtract(observed_first, observed_second)
    names = ("balanced_accuracy", "macro_f1")
    result: dict[str, Any] = {
        "replicates": replicates,
        "resampling": (
            "paired within-observed-class bootstrap independently within each campaign"
            if mode == "within_class"
            else "paired scaffold-cluster bootstrap independently within each campaign"
        ),
        "contrast": "(V13.7 hybrid-router delta) minus (V13.8 hybrid-router delta)",
    }
    for column, name in enumerate(names):
        result[name] = {
            "observed_interaction": float(observed[column]),
            "ci95": [
                float(np.quantile(values[:, column], 0.025)),
                float(np.quantile(values[:, column], 0.975)),
            ],
            "bootstrap_probability_interaction_positive": float(np.mean(values[:, column] > 0)),
        }
    return result


def _bh_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for rank_index in range(len(values) - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, values[original_index] * len(values) / rank)
        adjusted[original_index] = running
    return adjusted.tolist()


def _distribution_audit(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, Any]:
    rows = []
    p_values = []
    for column, label in CONTINUOUS_COLUMNS.items():
        left = first[column].to_numpy(float)
        right = second[column].to_numpy(float)
        test = mannwhitneyu(left, right, alternative="two-sided", method="asymptotic")
        p_values.append(float(test.pvalue))
        rows.append(
            {
                "feature": column,
                "label": label,
                "v13_7_median": float(np.median(left)),
                "v13_7_iqr": [float(np.quantile(left, 0.25)), float(np.quantile(left, 0.75))],
                "v13_8_median": float(np.median(right)),
                "v13_8_iqr": [float(np.quantile(right, 0.25)), float(np.quantile(right, 0.75))],
                "mann_whitney_two_sided_p_value": float(test.pvalue),
            }
        )
    adjusted = _bh_adjust(p_values)
    for row, value in zip(rows, adjusted, strict=True):
        row["benjamini_hochberg_q_value"] = value
    return {
        "comparisons": rows,
        "multiplicity": "Benjamini-Hochberg across the six descriptive feature comparisons",
        "inferential_scope": (
            "post-hoc distribution-shift descriptors; they do not identify a causal failure mode "
            "and were not used to tune the model or policy"
        ),
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    first = report["campaigns"]["V13.7_2025"]
    second = report["campaigns"]["V13.8_2026"]
    interaction = report["interaction_bootstrap"]["within_class"]["balanced_accuracy"]
    lines = [
        "# hERG receptor campaign heterogeneity — V13.12",
        "",
        "## Decision",
        "",
        "The frozen V13.6 hybrid is **not promoted**. Its class-balanced gain in V13.7 did not "
        "replicate in V13.8, and both campaigns failed the pre-existing Potent-to-Safe safety gate. "
        "This audit is explanatory and post-hoc; no model, feature, threshold, or policy was changed.",
        "",
        "## External campaign contrast",
        "",
        "| Campaign | n | scaffolds | Δ balanced accuracy | Δ macro-F1 | changed | corrected | degraded |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in (("V13.7 (2025)", first), ("V13.8 (2026)", second)):
        changes = row["decision_changes"]
        lines.append(
            f"| {label} | {row['n']} | {row['unique_scaffolds']} | "
            f"{row['delta_balanced_accuracy_hybrid_minus_router']:+.4f} | "
            f"{row['delta_macro_f1_hybrid_minus_router']:+.4f} | "
            f"{changes['changed_count']} | {changes['corrected_count']} | {changes['degraded_count']} |"
        )
    lines.extend(
        [
            "",
            "The cross-campaign interaction in balanced-accuracy gain was "
            f"{interaction['observed_interaction']:+.4f}; the within-class bootstrap 95% interval "
            f"was [{interaction['ci95'][0]:+.4f}, {interaction['ci95'][1]:+.4f}].",
            "",
            "## Interpretation limits",
            "",
            "Feature shifts and decision transitions are descriptive. Different assay sources, class "
            "mixtures, and chemical domains can all contribute. Vina scores are docking observables, "
            "not experimental binding free energies. The only defensible production decision is to "
            "retain the ligand-only router and reserve the receptor hybrid for blinded model-challenge "
            "experiments.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def analyze(first_root: Path, second_root: Path, output: Path, bootstrap: int) -> dict[str, Any]:
    first, first_source = _load_campaign(first_root, "V13.7_2025")
    second, second_source = _load_campaign(second_root, "V13.8_2026")
    combined_columns = [
        "campaign",
        "external_structure_id",
        "scaffold_group_id",
        "true_pic50_m",
        "true_class",
        "router_prediction",
        "hybrid_prediction",
        *CONTINUOUS_COLUMNS,
    ]
    output.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([first[combined_columns], second[combined_columns]], ignore_index=True)
    combined_path = output / "campaign_decision_audit.parquet"
    combined.to_parquet(combined_path, index=False, compression="zstd")
    first_summary = _campaign_summary(first)
    second_summary = _campaign_summary(second)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete_posthoc_nonpromotion_audit",
        "scientific_scope": {
            "model_feature_threshold_or_policy_tuned": False,
            "external_outcomes_used_for_model_selection": False,
            "analysis_timing": "post-hoc after sealed V13.7 and V13.8 scoring",
            "production_decision": "retain ligand-only router; do not generally promote V13.6 hybrid",
        },
        "sources": {"V13.7_2025": first_source, "V13.8_2026": second_source},
        "campaigns": {
            "V13.7_2025": first_summary,
            "V13.8_2026": second_summary,
        },
        "decision_change_yield_comparison": _decision_change_yield(first_summary, second_summary),
        "interaction_bootstrap": {
            "within_class": _interaction_bootstrap(first, second, bootstrap, "within_class"),
            "scaffold_cluster": _interaction_bootstrap(first, second, bootstrap, "scaffold"),
        },
        "feature_distribution_audit": _distribution_audit(first, second),
        "artifacts": {
            "campaign_decision_audit": str(combined_path),
            "campaign_decision_audit_sha256": _sha(combined_path),
        },
    }
    report = _write_self_hashed(output / "analysis_report.json", payload, "report_sha256")
    _write_report(output / "REPORT.md", report)
    manifest = _write_self_hashed(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "analysis_report_sha256": _sha(output / "analysis_report.json"),
            "report_markdown_sha256": _sha(output / "REPORT.md"),
            "campaign_decision_audit_sha256": _sha(combined_path),
            "source_report_sha256": {
                key: value["analysis_report_sha256"] for key, value in payload["sources"].items()
            },
        },
        "manifest_sha256",
    )
    return {"report": report, "manifest": manifest}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v13-7",
        type=Path,
        default=Path("research/local_runs/herg_external_validation_v13_7"),
    )
    parser.add_argument(
        "--v13-8",
        type=Path,
        default=Path("research/local_runs/herg_temporal_confirmation_v13_8"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/local_runs/herg_receptor_heterogeneity_v13_12"),
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = analyze(args.v13_7, args.v13_8, args.output, args.bootstrap)
    summary = {
        "status": result["report"]["status"],
        "report_sha256": result["report"]["report_sha256"],
        "manifest_sha256": result["manifest"]["manifest_sha256"],
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
