#!/usr/bin/env python3
"""Build and verify the consolidated hERG V14.3 research release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "platform-local-herg-release-manifest-v14.3/1.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "research/local_runs/herg_v14_release"
V14_ROOT = REPO_ROOT / "research/local_runs/herg_cross_campaign_receptor_fusion_v14"
V141_ROOT = REPO_ROOT / "research/local_runs/herg_nonlinear_receptor_stress_v14_1"
V142_ROOT = REPO_ROOT / "research/local_runs/herg_nonlinear_endpoint_receptor_v14_2"


class ReleaseError(RuntimeError):
    """Raised when a release artifact or scientific policy contract fails."""


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


def _read_json(path: Path, hash_field: str) -> dict[str, Any]:
    body = json.loads(path.read_text())
    if not isinstance(body, dict):
        raise ReleaseError(f"expected JSON object: {path}")
    expected = body.get(hash_field)
    candidate = dict(body)
    candidate.pop(hash_field, None)
    if expected != hashlib.sha256(_canonical(candidate)).hexdigest():
        raise ReleaseError(f"self-hash mismatch: {path}")
    return body


def _write_json(path: Path, payload: dict[str, Any], hash_field: str) -> dict[str, Any]:
    body = dict(payload)
    body.pop(hash_field, None)
    body[hash_field] = hashlib.sha256(_canonical(body)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(body))
    temporary.replace(path)
    return body


def _relative(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ReleaseError(f"release path escapes repository: {resolved}")
    return str(resolved.relative_to(REPO_ROOT))


def _binding(path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ReleaseError(f"missing {role}: {path}")
    return {
        "role": role,
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _verify_binding(binding: dict[str, Any]) -> None:
    path = (REPO_ROOT / binding["path"]).resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ReleaseError(f"manifest path escapes repository: {path}")
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
        raise ReleaseError(f"release artifact is missing or changed size: {path}")
    if _sha(path) != binding["sha256"]:
        raise ReleaseError(f"release artifact hash changed: {path}")


def _verify_nested_manifest(path: Path, hash_field: str = "manifest_sha256") -> dict[str, Any]:
    manifest = _read_json(path, hash_field)
    for collection in ("inputs", "artifacts"):
        for binding in manifest.get(collection, []):
            nested_path = Path(binding["path"]).resolve()
            if not nested_path.is_file():
                raise ReleaseError(f"nested manifest dependency is missing: {nested_path}")
            if "bytes" in binding and nested_path.stat().st_size != int(binding["bytes"]):
                raise ReleaseError(f"nested dependency changed size: {nested_path}")
            if _sha(nested_path) != binding["sha256"]:
                raise ReleaseError(f"nested dependency hash changed: {nested_path}")
    script = manifest.get("script")
    if script:
        script_path = Path(script["path"]).resolve()
        if _sha(script_path) != script["sha256"]:
            raise ReleaseError(f"nested release script changed: {script_path}")
    return manifest


def _scientific_contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    v14 = _read_json(V14_ROOT / "analysis_report.json", "report_sha256")
    v141 = _read_json(V141_ROOT / "analysis_report.json", "report_sha256")
    v142 = _read_json(V142_ROOT / "analysis_report.json", "report_sha256")
    if not v14["promotion_decision"]["ligand_recalibration_research_preview_supported"]:
        raise ReleaseError("V14 no longer supports the ligand research preview")
    if v14["promotion_decision"]["general_receptor_prediction_promotion_supported"]:
        raise ReleaseError("V14 receptor policy changed")
    if v141["decision"]["general_receptor_prediction_promotion_supported"]:
        raise ReleaseError("V14.1 receptor policy changed")
    if v141["decision"]["best_ligand_classification_surface_posthoc"] != "ligand_calibrated":
        raise ReleaseError("V14.1 best ligand classifier changed")
    if v141["decision"]["best_ligand_regression_surface_posthoc"] != "extratrees_ligand_core":
        raise ReleaseError("V14.1 best ligand regressor changed")
    if v142["decision"]["website_integration_supported"]:
        raise ReleaseError("V14.2 website policy changed")
    if v142["decision"]["any_receptor_incremental_gate_passed"]:
        raise ReleaseError("V14.2 receptor endpoint decision changed")
    return v14, v141, v142


def _artifacts(output: Path, v14: dict[str, Any], v141: dict[str, Any]) -> list[dict[str, Any]]:
    v14_model = Path(v14["full_research_models"]["ligand_calibrated"]["path"])
    v141_model = Path(v141["models"]["extratrees_ligand_core"]["path"])
    paths_and_roles = [
        (output / "README.md", "release_index"),
        (output / "DISCUSSION_BRIEF.md", "discussion_brief"),
        (V14_ROOT / "MODEL_CARD.md", "v14_model_card"),
        (V14_ROOT / "analysis_report.json", "v14_analysis"),
        (V14_ROOT / "manifest.json", "v14_manifest"),
        (V14_ROOT / "final_summary.json", "v14_summary"),
        (v14_model, "v14_ligand_classification_preview_model"),
        (V141_ROOT / "MODEL_CARD.md", "v14_1_model_card"),
        (V141_ROOT / "analysis_report.json", "v14_1_analysis"),
        (V141_ROOT / "manifest.json", "v14_1_manifest"),
        (V141_ROOT / "final_summary.json", "v14_1_summary"),
        (v141_model, "v14_1_ligand_regression_preview_model"),
        (V142_ROOT / "MODEL_CARD.md", "v14_2_model_card"),
        (V142_ROOT / "analysis_report.json", "v14_2_analysis"),
        (V142_ROOT / "manifest.json", "v14_2_manifest"),
        (V142_ROOT / "final_summary.json", "v14_2_summary"),
        (
            REPO_ROOT
            / "research/local_runs/herg_menin_blinded_evaluation_v13_13/preanalysis_protocol.json",
            "locked_prospective_protocol",
        ),
        (
            REPO_ROOT
            / "research/local_runs/herg_menin_receptor_prioritization_v13_10/analysis/blinded_assay_challenge_panel.parquet",
            "locked_prospective_panel",
        ),
        (REPO_ROOT / "pipeline/scripts/run_herg_prediction_website.py", "website_backend"),
        (REPO_ROOT / "pipeline/web/herg/index.html", "website_html"),
        (REPO_ROOT / "pipeline/web/herg/app.js", "website_javascript"),
        (REPO_ROOT / "pipeline/web/herg/styles.css", "website_styles"),
        (
            REPO_ROOT / "pipeline/scripts/launch_herg_prediction_website.sh",
            "private_launch_script",
        ),
        (
            REPO_ROOT / "pipeline/scripts/stop_herg_prediction_website.sh",
            "private_stop_script",
        ),
        (REPO_ROOT / "docs/herg_website_launch.md", "website_launch_guide"),
        (
            REPO_ROOT / "pipeline/scripts/run_local_herg_cross_campaign_receptor_fusion_v14.py",
            "v14_source",
        ),
        (
            REPO_ROOT / "pipeline/scripts/run_local_herg_nonlinear_receptor_stress_test_v14_1.py",
            "v14_1_source",
        ),
        (
            REPO_ROOT / "pipeline/scripts/analyze_local_herg_nonlinear_endpoint_receptor_v14_2.py",
            "v14_2_source",
        ),
        (Path(__file__).resolve(), "v14_3_release_builder"),
    ]
    return [_binding(path, role) for path, role in paths_and_roles]


def _build(output: Path, passing_tests: int) -> dict[str, Any]:
    v14, v141, v142 = _scientific_contract()
    _verify_nested_manifest(V14_ROOT / "manifest.json")
    _verify_nested_manifest(V141_ROOT / "manifest.json")
    _verify_nested_manifest(V142_ROOT / "manifest.json")
    artifacts = _artifacts(output, v14, v141)
    manifest = _write_json(
        output / "release_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete_research_release",
            "artifacts": artifacts,
            "scientific_decision": {
                "production_default": "V9/V10.1/V10.3/V12.3 ligand-only stack",
                "research_preview_classification": "V14 ligand_calibrated",
                "research_preview_regression": "V14.1 extratrees_ligand_core",
                "research_preview_default": False,
                "receptor_classification_promoted": False,
                "receptor_regression_promoted": False,
                "direct_ic10_ic30_ic50_receptor_promoted": False,
                "next_decisive_evidence": "locked 24-compound blinded assay challenge",
            },
            "validation": {
                "cross_campaign_rows": 1_224,
                "cross_campaigns": 6,
                "cross_campaign_exact_overlap": 0,
                "cross_campaign_scaffold_overlap": 0,
                "v14_report_sha256": v14["report_sha256"],
                "v14_1_report_sha256": v141["report_sha256"],
                "v14_2_report_sha256": v142["report_sha256"],
            },
            "nomenclature_correction": {
                "historical_residue_649_label": "F649",
                "deposited_structure_residue": "S649",
                "true_aromatic_cage_residues": ["Y652", "F656"],
            },
            "claim_boundary": {
                "research_use_only": True,
                "prospectively_validated": False,
                "clinical_or_regulatory_use": False,
                "vina_is_binding_free_energy": False,
                "pose_is_experimentally_confirmed": False,
                "outcome_or_subset_cherry_picking": False,
            },
            "test_verification": {
                "passing_herg_tests": passing_tests,
                "command": (
                    "MPLCONFIGDIR=.codex_tmp/matplotlib XDG_CACHE_HOME=.codex_tmp/cache "
                    ".venv/bin/pytest -q pipeline/tests/*herg*.py"
                ),
                "website_validation": (
                    ".venv/bin/python pipeline/scripts/run_herg_prediction_website.py validate"
                ),
            },
        },
        "release_sha256",
    )
    return _write_json(
        output / "final_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "finished_utc": _utc(),
            "status": "complete_research_release",
            "release_sha256": manifest["release_sha256"],
            "artifacts": len(artifacts),
            "production_default_changed": False,
            "ligand_research_preview_integrated": True,
            "receptor_prediction_promoted": False,
        },
        "summary_sha256",
    )


def _verify(output: Path) -> dict[str, Any]:
    manifest = _read_json(output / "release_manifest.json", "release_sha256")
    for binding in manifest["artifacts"]:
        _verify_binding(binding)
    summary = _read_json(output / "final_summary.json", "summary_sha256")
    if summary["release_sha256"] != manifest["release_sha256"]:
        raise ReleaseError("release summary does not match the manifest")
    _scientific_contract()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "release_sha256": manifest["release_sha256"],
        "artifacts_verified": len(manifest["artifacts"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build", "verify", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--passing-tests", type=int, default=356)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.resolve()
    try:
        result: dict[str, Any] = {}
        if args.stage in {"build", "all"}:
            result = _build(output, args.passing_tests)
        if args.stage in {"verify", "all"}:
            result = _verify(output)
    except ReleaseError as exc:
        print(f"V14.3 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
