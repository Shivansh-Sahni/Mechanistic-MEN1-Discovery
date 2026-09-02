#!/usr/bin/env python3
"""Evaluate deterministic physical-order projection of V10.1 endpoint OOF predictions.

The paired direct-curve subset contains IC10, IC30, and IC50 observations for
the same 764 structures. V12.3 applies a label-free Euclidean isotonic
projection to the strongest historical endpoint-specific OOF predictions so
that pIC10 >= pIC30 >= pIC50. No parameter is fitted and no outcome is used by
the projection. Scaffold bootstrap quantifies the change in MAE.
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

SCHEMA_VERSION = "platform-local-herg-endpoint-order-projection-v12.3/1.0"
ENDPOINTS = (10, 30, 50)


class CampaignError(RuntimeError):
    """Projection integrity or scientific-contract failure."""


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


def _order_violations(values: np.ndarray) -> int:
    return int(np.sum((values[:, 0] < values[:, 1]) | (values[:, 1] < values[:, 2])))


def _analyze(frame: pd.DataFrame, bootstrap: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_columns = [f"historical_predicted_pic{endpoint}" for endpoint in ENDPOINTS]
    observed_columns = [f"observed_pic{endpoint}" for endpoint in ENDPOINTS]
    raw = frame[raw_columns].to_numpy(float)
    observed = frame[observed_columns].to_numpy(float)
    if not np.isfinite(raw).all() or not np.isfinite(observed).all():
        raise CampaignError("paired endpoint matrix contains non-finite predictions or labels")
    projected = np.vstack([v13._project_nonincreasing(row) for row in raw])  # noqa: SLF001
    result = frame[
        ["standard_inchi_key", "scaffold_group_id", "outer_fold", *observed_columns, *raw_columns]
    ].copy()
    for index, endpoint in enumerate(ENDPOINTS):
        result[f"projected_historical_pic{endpoint}"] = projected[:, index]
    endpoint_reports = {}
    for index, endpoint in enumerate(ENDPOINTS):
        baseline_metrics = v13._regression_metrics(observed[:, index], raw[:, index])  # noqa: SLF001
        projected_metrics = v13._regression_metrics(  # noqa: SLF001
            observed[:, index], projected[:, index]
        )
        endpoint_reports[f"IC{endpoint}"] = {
            "baseline_historical": baseline_metrics,
            "projected_historical": projected_metrics,
            "bootstrap": v132._bootstrap_delta(  # noqa: SLF001
                frame,
                raw[:, index],
                projected[:, index],
                observed[:, index],
                bootstrap,
            ),
        }
    pooled_frame = pd.DataFrame(
        {"scaffold_group_id": np.repeat(frame.scaffold_group_id.astype(str).to_numpy(), 3)}
    )
    pooled_bootstrap = v132._bootstrap_delta(  # noqa: SLF001
        pooled_frame,
        raw.ravel(),
        projected.ravel(),
        observed.ravel(),
        bootstrap,
    )
    report = {
        "n_paired_structures": len(frame),
        "unique_scaffolds": int(frame.scaffold_group_id.nunique()),
        "baseline_order_violations": _order_violations(raw),
        "projected_order_violations": _order_violations(projected),
        "endpoint_reports": endpoint_reports,
        "pooled": {
            "baseline_mae": float(np.mean(np.abs(observed - raw))),
            "projected_mae": float(np.mean(np.abs(observed - projected))),
            "bootstrap": pooled_bootstrap,
        },
    }
    report["promotion_decision"] = {
        "promote_for_complete_paired_curves": bool(
            report["projected_order_violations"] == 0
            and pooled_bootstrap["ci95"][1] < 0
            and all(
                details["bootstrap"]["delta_mae_candidate_minus_baseline"] <= 0
                for details in endpoint_reports.values()
            )
        ),
        "rule": (
            "zero physical-order violations; upper 95% pooled scaffold-bootstrap delta <0; "
            "no endpoint has a positive observed MAE delta"
        ),
    }
    return result, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("research/local_runs/herg_anchor_refinement_campaign_v12_2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/local_runs/herg_endpoint_order_projection_v12_3"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser


def _main(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    input_root = args.input_root if args.input_root.is_absolute() else repo / args.input_root
    output = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    input_root, output = input_root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = input_root / "endpoints/common_fold_oof_predictions.parquet"
    frame = pd.read_parquet(source)
    predictions, analysis = _analyze(frame, args.bootstrap_replicates)
    prediction_path = output / "projected_endpoint_oof_predictions.parquet"
    _parquet(prediction_path, predictions)
    report = _json(
        output / "analysis_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete_label_free_order_projection_analysis",
            **analysis,
            "scientific_scope": {
                "projection_uses_outcomes": False,
                "projection_has_fitted_parameters": False,
                "evaluation_predictions_are_historical_endpoint_specific_oof": True,
                "applies_only_when_all_three_endpoint_predictions_are_available": True,
                "external_or_prospective_validation": False,
            },
        },
        "report_sha256",
    )
    inputs = [
        repo / "pipeline/scripts/analyze_local_herg_endpoint_order_projection_v12_3.py",
        source,
        input_root / "manifest.json",
    ]
    artifacts = [prediction_path, output / "analysis_report.json"]
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
            "n_paired_structures": len(frame),
            "promote_for_complete_paired_curves": report["promotion_decision"][
                "promote_for_complete_paired_curves"
            ],
            "report_sha256": report["report_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "summary_sha256",
    )


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _main(args)
    except CampaignError as exc:
        print(f"V12.3 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
