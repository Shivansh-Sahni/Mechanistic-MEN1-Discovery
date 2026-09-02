#!/usr/bin/env python3
"""Audit frozen V9 residual intervals on the two V13 external campaigns.

Absolute nested-OOF residuals and fixed Morgan-similarity bins are the only
calibration inputs. External outcomes are used once for coverage evaluation,
never to choose interval width, coverage level, or bin boundaries.  The result
is an uncertainty audit, not an exchangeability guarantee under domain shift.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_local_herg_external_validation_v13_7 as v137  # noqa: E402

SCHEMA_VERSION = "platform-local-herg-external-uncertainty-v13.11/1.0"
DEFAULT_V9 = Path("research/local_runs/herg_domain_mixture_campaign_v9")
DEFAULT_V137 = Path("research/local_runs/herg_external_validation_v13_7")
DEFAULT_V138 = Path("research/local_runs/herg_temporal_confirmation_v13_8")
DEFAULT_OUTPUT = Path("research/local_runs/herg_external_uncertainty_v13_11")
COVERAGE_LEVELS = (0.80, 0.90, 0.95)
SIMILARITY_EDGES = (-math.inf, 0.30, 0.50, 0.70, math.inf)
SIMILARITY_NAMES = ("lt_0p3", "0p3_to_lt_0p5", "0p5_to_lt_0p7", "ge_0p7")


class CampaignError(RuntimeError):
    """Raised when uncertainty calibration or evaluation integrity fails."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(repo: Path, path: Path) -> Path:
    return v137._resolve(repo, path)  # noqa: SLF001


def _sha(path: Path) -> str:
    return v137._sha(path)  # noqa: SLF001


def _json(path: Path, payload: dict[str, Any], field: str) -> dict[str, Any]:
    return v137._json(path, payload, field)  # noqa: SLF001


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    v137._parquet(path, frame)  # noqa: SLF001


def _similarity_bin(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(
        pd.cut(
            np.asarray(values, dtype=float),
            SIMILARITY_EDGES,
            labels=SIMILARITY_NAMES,
            right=False,
        ),
        dtype="string",
    )


def _finite_sample_radius(values: np.ndarray, coverage: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise CampaignError("uncertainty calibration received no finite residuals")
    rank = min(len(values), math.ceil((len(values) + 1) * coverage))
    return float(np.partition(values, rank - 1)[rank - 1])


def _fit_calibration(frame: pd.DataFrame) -> dict[str, Any]:
    error = np.abs(
        frame.observed_pic50.to_numpy(float) - frame.pred__honest_stack.to_numpy(float)
    )
    bins = _similarity_bin(frame.maximum_train_tanimoto)
    levels: dict[str, Any] = {}
    for coverage in COVERAGE_LEVELS:
        levels[f"{coverage:.2f}"] = {
            "global_radius_pic50": _finite_sample_radius(error, coverage),
            "similarity_mondrian_radius_pic50": {
                name: _finite_sample_radius(error[bins.eq(name).to_numpy()], coverage)
                for name in SIMILARITY_NAMES
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "calibration_rows": len(frame),
        "unique_scaffolds": int(frame.scaffold_group_id.nunique()),
        "residual_source": "V9 nested scaffold OOF observed_pic50 - pred__honest_stack",
        "similarity_source": "V9 outer-fold maximum_train_tanimoto",
        "similarity_bins": [
            {
                "name": name,
                "lower": None if not np.isfinite(SIMILARITY_EDGES[index]) else SIMILARITY_EDGES[index],
                "upper": (
                    None
                    if not np.isfinite(SIMILARITY_EDGES[index + 1])
                    else SIMILARITY_EDGES[index + 1]
                ),
                "rows": int(bins.eq(name).sum()),
            }
            for index, name in enumerate(SIMILARITY_NAMES)
        ],
        "coverage_levels": levels,
        "external_outcomes_used_for_calibration": False,
    }


def _wilson(successes: int, total: int) -> list[float]:
    if total < 1:
        return [math.nan, math.nan]
    rate = successes / total
    z = 1.959963984540054
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return [float(max(0, center - half)), float(min(1, center + half))]


def _cross_fold_internal(frame: pd.DataFrame) -> list[dict[str, Any]]:
    residual = np.abs(frame.observed_pic50 - frame.pred__honest_stack).to_numpy(float)
    bins = _similarity_bin(frame.maximum_train_tanimoto)
    rows = []
    for coverage in COVERAGE_LEVELS:
        for method in ("global", "similarity_mondrian"):
            radii = np.full(len(frame), np.nan)
            for fold in sorted(frame.outer_fold.unique()):
                fit = frame.outer_fold.ne(fold).to_numpy()
                evaluate = ~fit
                if method == "global":
                    radii[evaluate] = _finite_sample_radius(residual[fit], coverage)
                else:
                    for name in SIMILARITY_NAMES:
                        calibration_rows = fit & bins.eq(name).to_numpy()
                        evaluation_rows = evaluate & bins.eq(name).to_numpy()
                        radii[evaluation_rows] = _finite_sample_radius(
                            residual[calibration_rows], coverage
                        )
            if not np.isfinite(radii).all():
                raise CampaignError("cross-fold interval coverage is incomplete")
            covered = residual <= radii
            rows.append(
                {
                    "nominal_coverage": coverage,
                    "method": method,
                    "n": len(frame),
                    "empirical_coverage": float(covered.mean()),
                    "covered": int(covered.sum()),
                    "wilson_ci95": _wilson(int(covered.sum()), len(frame)),
                    "mean_half_width_pic50": float(radii.mean()),
                    "calibration": "other outer folds only for every evaluated row",
                }
            )
    return rows


def _v137_rows(root: Path) -> pd.DataFrame:
    labels = pd.read_parquet(root / "sealed/labels.parquet")
    labels = (
        labels.loc[labels.set_name.eq("ev2_exact_quantitative")]
        .groupby("external_structure_id", as_index=False)
        .true_pic50_m.median()
    )
    registry = pd.read_parquet(
        root / "prepared/structure_registry.parquet",
        columns=[
            "external_structure_id",
            "scaffold_group_id",
            "master_connectivity_overlap",
            "v9_scaffold_overlap",
            "nearest_v9_morgan_tanimoto",
        ],
    )
    prediction = pd.read_parquet(
        root / "predictions/baseline_predictions_before_score.parquet",
        columns=["external_structure_id", "v9_mixed_ic50_predicted_pic50"],
    )
    frame = labels.merge(registry, on="external_structure_id", validate="one_to_one").merge(
        prediction, on="external_structure_id", validate="one_to_one"
    )
    frame = frame.loc[
        ~frame.master_connectivity_overlap.astype(bool)
        & ~frame.v9_scaffold_overlap.astype(bool)
    ].copy()
    frame["campaign"] = "v13.7_ev2_project_exact_and_v9_scaffold_novel"
    return frame


def _v138_rows(root: Path) -> pd.DataFrame:
    labels = (
        pd.read_parquet(root / "sealed/labels.parquet")
        .groupby("external_structure_id", as_index=False)
        .true_pic50_m.median()
    )
    registry = pd.read_parquet(
        root / "prepared/structure_registry.parquet",
        columns=[
            "external_structure_id",
            "scaffold_group_id",
            "v9_scaffold_overlap",
            "nearest_v9_morgan_tanimoto",
        ],
    )
    prediction = pd.read_parquet(
        root / "predictions/baseline_predictions_before_score.parquet",
        columns=["external_structure_id", "v9_mixed_ic50_predicted_pic50"],
    )
    frame = labels.merge(registry, on="external_structure_id", validate="one_to_one").merge(
        prediction, on="external_structure_id", validate="one_to_one"
    )
    frame = frame.loc[~frame.v9_scaffold_overlap.astype(bool)].copy()
    frame["campaign"] = "v13.8_post2021_v9_scaffold_novel"
    return frame


def _evaluate_external(
    campaigns: list[pd.DataFrame], calibration: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    prediction_rows = []
    summaries = []
    for frame in campaigns:
        frame = frame.reset_index(drop=True)
        bins = _similarity_bin(frame.nearest_v9_morgan_tanimoto)
        observed = frame.true_pic50_m.to_numpy(float)
        prediction = frame.v9_mixed_ic50_predicted_pic50.to_numpy(float)
        for coverage in COVERAGE_LEVELS:
            level = calibration["coverage_levels"][f"{coverage:.2f}"]
            for method in ("global", "similarity_mondrian"):
                if method == "global":
                    radii = np.repeat(float(level["global_radius_pic50"]), len(frame))
                else:
                    mapping = level["similarity_mondrian_radius_pic50"]
                    radii = np.asarray([float(mapping[str(name)]) for name in bins], dtype=float)
                covered = np.abs(observed - prediction) <= radii
                campaign = str(frame.campaign.iloc[0])
                summaries.append(
                    {
                        "campaign": campaign,
                        "nominal_coverage": coverage,
                        "method": method,
                        "n": len(frame),
                        "unique_scaffolds": int(frame.scaffold_group_id.nunique()),
                        "covered": int(covered.sum()),
                        "empirical_coverage": float(covered.mean()),
                        "coverage_minus_nominal": float(covered.mean() - coverage),
                        "wilson_ci95": _wilson(int(covered.sum()), len(frame)),
                        "mean_half_width_pic50": float(radii.mean()),
                        "mean_full_width_pic50": float(2 * radii.mean()),
                    }
                )
                for index, source in frame.iterrows():
                    prediction_rows.append(
                        {
                            "campaign": campaign,
                            "external_structure_id": source.external_structure_id,
                            "scaffold_group_id": source.scaffold_group_id,
                            "nearest_v9_morgan_tanimoto": float(
                                source.nearest_v9_morgan_tanimoto
                            ),
                            "similarity_bin": str(bins.iloc[index]),
                            "nominal_coverage": coverage,
                            "method": method,
                            "observed_pic50": float(observed[index]),
                            "predicted_pic50": float(prediction[index]),
                            "half_width_pic50": float(radii[index]),
                            "lower_pic50": float(prediction[index] - radii[index]),
                            "upper_pic50": float(prediction[index] + radii[index]),
                            "covered": bool(covered[index]),
                        }
                    )
    return pd.DataFrame(prediction_rows), summaries


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# hERG V13.11 external uncertainty audit",
        "",
        "Frozen V9 nested-OOF residual widths were evaluated without external recalibration.",
        "",
        "| Campaign | Nominal | Method | N | Coverage (95% Wilson CI) | Mean half-width |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in report["external_coverage"]:
        ci = row["wilson_ci95"]
        lines.append(
            f"| {row['campaign']} | {row['nominal_coverage']:.0%} | {row['method']} | "
            f"{row['n']} | {row['empirical_coverage']:.1%} "
            f"[{ci[0]:.1%}, {ci[1]:.1%}] | {row['mean_half_width_pic50']:.3f} pIC50 |"
        )
    lines.extend(
        [
            "",
            "These are retrospective coverage measurements under domain shift. The intervals are operational uncertainty bands, not clinical guarantees; external outcomes were never used to set their widths.",
            "",
        ]
    )
    return "\n".join(lines)


def _run(v9: Path, v137_root: Path, v138_root: Path, output: Path) -> dict[str, Any]:
    oof_path = v9 / "analysis/nested_oof_predictions.parquet"
    oof = pd.read_parquet(oof_path)
    required = {
        "structure_id",
        "scaffold_group_id",
        "outer_fold",
        "observed_pic50",
        "pred__honest_stack",
        "maximum_train_tanimoto",
    }
    if len(oof) != 18_801 or not required.issubset(oof):
        raise CampaignError("V9 nested OOF uncertainty source is incomplete")
    calibration = _fit_calibration(oof)
    calibration["source_sha256"] = _sha(oof_path)
    calibration_report = _json(
        output / "uncertainty_calibration.json", calibration, "calibration_sha256"
    )
    prediction_rows, external_summary = _evaluate_external(
        [_v137_rows(v137_root), _v138_rows(v138_root)], calibration_report
    )
    prediction_path = output / "external_interval_predictions.parquet"
    _parquet(prediction_path, prediction_rows)
    report = _json(
        output / "analysis_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete_external_uncertainty_audit",
            "calibration_sha256": calibration_report["calibration_sha256"],
            "internal_cross_fold_coverage": _cross_fold_internal(oof),
            "external_coverage": external_summary,
            "external_interval_predictions_sha256": _sha(prediction_path),
            "external_outcomes_used_for_interval_calibration": False,
            "claim_boundary": (
                "cross-validated residual intervals; retrospective coverage under domain shift, "
                "not a finite-sample clinical or prospective guarantee"
            ),
        },
        "report_sha256",
    )
    (output / "REPORT.md").write_text(_render_report(report))
    return report


def _manifest(
    repo: Path, v9: Path, v137_root: Path, v138_root: Path, output: Path
) -> dict[str, Any]:
    inputs = [
        repo / "pipeline/scripts/analyze_local_herg_external_uncertainty_v13_11.py",
        v9 / "analysis/nested_oof_predictions.parquet",
        v137_root / "manifest.json",
        v138_root / "manifest.json",
    ]
    artifacts = [
        output / "uncertainty_calibration.json",
        output / "external_interval_predictions.parquet",
        output / "analysis_report.json",
        output / "REPORT.md",
    ]
    return _json(
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--v9-root", type=Path, default=DEFAULT_V9)
    parser.add_argument("--v13-7-root", type=Path, default=DEFAULT_V137)
    parser.add_argument("--v13-8-root", type=Path, default=DEFAULT_V138)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _main(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    v9 = _resolve(repo, args.v9_root)
    v137_root = _resolve(repo, args.v13_7_root)
    v138_root = _resolve(repo, args.v13_8_root)
    output = _resolve(repo, args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    report = _run(v9, v137_root, v138_root, output)
    manifest = _manifest(repo, v9, v137_root, v138_root, output)
    return {"report": report, "manifest": manifest}


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _main(args)
    except (CampaignError, v137.CampaignError) as exc:
        print(f"V13.11 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
