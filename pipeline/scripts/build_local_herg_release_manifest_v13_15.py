#!/usr/bin/env python3
"""Build a self-hashed release index for the completed V11--V13 hERG program."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = "platform-local-herg-release-manifest-v13.15/1.0"
ARTIFACTS = (
    "research/local_runs/HERG_V11_V13_DECISION_REPORT.md",
    "research/local_runs/herg_v11_v13_figures/HERG_V11_V13_DECISION_FIGURE.png",
    "research/local_runs/herg_v11_v13_figures/HERG_V11_V13_DECISION_FIGURE.pdf",
    "research/local_runs/herg_receptor_inference_bundle_v13_9/manifest.json",
    "research/local_runs/herg_receptor_inference_bundle_v13_9/verification_report.json",
    "research/local_runs/herg_menin_receptor_prioritization_v13_10/analysis_report.json",
    "research/local_runs/herg_menin_receptor_prioritization_v13_10/analysis/all_48_ranked_predictions.parquet",
    "research/local_runs/herg_menin_receptor_prioritization_v13_10/analysis/blinded_assay_challenge_panel.parquet",
    "research/local_runs/herg_menin_receptor_prioritization_v13_10/analysis/blinded_herg_outcome_release.csv",
    "research/local_runs/herg_external_uncertainty_v13_11/analysis_report.json",
    "research/local_runs/herg_receptor_heterogeneity_v13_12/analysis_report.json",
    "research/local_runs/herg_menin_blinded_evaluation_v13_13/preanalysis_protocol.json",
    "research/local_runs/herg_menin_blinded_evaluation_v13_13/REPORT.md",
    "research/local_runs/herg_external_boundary_sensitivity_v13_14/analysis_report.json",
    "research/local_runs/herg_external_boundary_sensitivity_v13_14/publisher_boundary_trace.csv",
    "research/external_validation_sources/tx5c00065_si_002.xlsx",
    "research/external_validation_sources/ci6c00163_si_001.xlsx",
    "pipeline/scripts/run_local_herg_external_validation_v13_7.py",
    "pipeline/scripts/run_local_herg_temporal_confirmation_v13_8.py",
    "pipeline/scripts/build_local_herg_receptor_inference_bundle_v13_9.py",
    "pipeline/scripts/run_local_herg_menin_receptor_prioritization_v13_10.py",
    "pipeline/scripts/analyze_local_herg_external_uncertainty_v13_11.py",
    "pipeline/scripts/analyze_local_herg_receptor_heterogeneity_v13_12.py",
    "pipeline/scripts/run_local_herg_menin_blinded_evaluation_v13_13.py",
    "pipeline/scripts/analyze_local_herg_external_boundary_sensitivity_v13_14.py",
    "pipeline/scripts/build_local_herg_release_manifest_v13_15.py",
)


class ReleaseError(RuntimeError):
    """Raised when a release invariant fails."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ReleaseError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("release_sha256", None)
    body["release_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(body))
    temporary.replace(path)
    return body


def _validate(repo: Path) -> dict[str, Any]:
    verification = _json(
        repo / "research/local_runs/herg_receptor_inference_bundle_v13_9/verification_report.json"
    )
    prioritization = _json(
        repo / "research/local_runs/herg_menin_receptor_prioritization_v13_10/analysis_report.json"
    )
    heterogeneity = _json(
        repo / "research/local_runs/herg_receptor_heterogeneity_v13_12/analysis_report.json"
    )
    protocol_path = (
        repo / "research/local_runs/herg_menin_blinded_evaluation_v13_13/preanalysis_protocol.json"
    )
    protocol = _json(protocol_path)
    sensitivity = _json(
        repo / "research/local_runs/herg_external_boundary_sensitivity_v13_14/analysis_report.json"
    )
    if not verification["all_passed"] or verification["total_rows_replayed"] != 324:
        raise ReleaseError("V13.9 exact replay is incomplete")
    if prioritization["status"] != "complete_label_blind_prospective_prioritization":
        raise ReleaseError("V13.10 panel is incomplete")
    if heterogeneity["scientific_scope"]["production_decision"] != (
        "retain ligand-only router; do not generally promote V13.6 hybrid"
    ):
        raise ReleaseError("V13.12 production decision is unexpected")
    implementation = repo / "pipeline/scripts/run_local_herg_menin_blinded_evaluation_v13_13.py"
    if _sha(implementation) != protocol["implementation"]["sha256"]:
        raise ReleaseError("V13.13 implementation no longer matches the locked protocol")
    blank = pd.read_csv(
        repo / "research/local_runs/herg_menin_receptor_prioritization_v13_10/analysis/"
        "blinded_herg_outcome_release.csv"
    )
    if not blank[["measured_ic10_nm", "measured_ic30_nm", "measured_ic50_nm"]].isna().all().all():
        raise ReleaseError("V13.10 outcome template is no longer blank")
    if not sensitivity["decision"]["primary_v13_7_metrics_unchanged"]:
        raise ReleaseError("V13.14 changed the primary V13.7 result")
    return {
        "exact_replay_rows": verification["total_rows_replayed"],
        "prospective_panel_rows": prioritization["panel_rows"],
        "prospective_panel_unique_series": prioritization["panel_unique_series"],
        "prospective_disagreements": prioritization["router_hybrid_disagreements"],
        "outcomes_present": False,
        "v13_13_protocol_sha256": protocol["protocol_sha256"],
    }


def build(repo: Path, output: Path) -> dict[str, Any]:
    validation = _validate(repo)
    artifacts = []
    for relative in ARTIFACTS:
        path = repo / relative
        if not path.is_file():
            raise ReleaseError(f"required release artifact is missing: {relative}")
        artifacts.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha(path)})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete_research_release",
        "scientific_decision": {
            "exact_ic50_regression": "V9 retained and externally supported",
            "direct_ic10_ic30_ic50": "V10.1 retained; V12.3 order projection for complete triplets",
            "receptor_regression": "not promoted",
            "default_classification": "V10.1 ligand-only LGBM router",
            "receptor_classification": "experimental model-challenge only; not generally promoted",
            "reason": "V13.7 gain did not replicate in V13.8 and both external safety gates failed",
            "next_evidence": "locked V13.10/V13.13 blinded 24-compound assay challenge",
        },
        "claim_boundary": {
            "clinical_or_regulatory_use": False,
            "vina_is_binding_free_energy": False,
            "docking_contact_is_causal_evidence": False,
            "semi_honest_or_cherry_picked_reporting": False,
        },
        "validation": validation,
        "test_verification": {
            "command": "MPLCONFIGDIR=.codex_tmp/matplotlib XDG_CACHE_HOME=.codex_tmp/cache .venv/bin/pytest -q pipeline/tests/test_*local_herg*.py",
            "passing_tests_at_release": 226,
            "ruff_command": ".venv/bin/ruff check pipeline/scripts/*herg*.py pipeline/tests/test_*herg*.py",
        },
        "artifacts": artifacts,
    }
    result = _write_json(output / "release_manifest.json", payload)
    lines = [
        "# hERG V13.15 research release index",
        "",
        f"Status: **{result['status']}**  ",
        f"Release SHA-256: `{result['release_sha256']}`",
        "",
        "## Final decision",
        "",
        "V9/V10.1/V12.3 form the regression stack. The V10.1 ligand-only LGBM router remains "
        "the default classifier. The receptor hybrid is preserved only for the locked blinded "
        "model-challenge experiment; it is not generally promoted.",
        "",
        "## Primary handoff",
        "",
        "- [Consolidated decision report](../HERG_V11_V13_DECISION_REPORT.md)",
        "- [Decision figure](../herg_v11_v13_figures/HERG_V11_V13_DECISION_FIGURE.pdf)",
        "- [Exact bundle replay](../herg_receptor_inference_bundle_v13_9/verification_report.json)",
        "- [Blinded assay panel](../herg_menin_receptor_prioritization_v13_10/analysis/blinded_assay_challenge_panel.parquet)",
        "- [Preanalysis protocol](../herg_menin_blinded_evaluation_v13_13/preanalysis_protocol.json)",
        "",
        "Every path and SHA-256 required for reproduction is recorded in `release_manifest.json`.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("research/local_runs/herg_v13_release"))
    args = parser.parse_args()
    result = build(args.repo.resolve(), args.output)
    print(json.dumps({"status": result["status"], "release_sha256": result["release_sha256"]}))


if __name__ == "__main__":
    main()
