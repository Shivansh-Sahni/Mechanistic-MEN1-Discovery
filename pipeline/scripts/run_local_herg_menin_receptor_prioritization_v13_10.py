#!/usr/bin/env python3
"""Apply the frozen V13.9 receptor bundle to the prospective Menin assay plan.

This campaign is deliberately label-blind: it accepts only prospective-plan
rows without observed hERG evidence, runs the frozen ligand/Vina/receptor path,
and creates a diverse model-challenge panel.  Ranking favours hybrid-vs-router
disagreement and risk escalation; it is an experimental prioritization, not a
replacement for a concentration-response hERG assay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_local_herg_receptor_inference_bundle_v13_9 as v139  # noqa: E402

SCHEMA_VERSION = "platform-local-herg-menin-receptor-prioritization-v13.10/1.0"
DEFAULT_SOURCE = Path("research/analysis/prospective_selection_plan.csv")
DEFAULT_BUNDLE = Path("research/local_runs/herg_receptor_inference_bundle_v13_9")
DEFAULT_OUTPUT = Path("research/local_runs/herg_menin_receptor_prioritization_v13_10")
DEFAULT_PANEL_SIZE = 24
REQUIRED_COLUMNS = {
    "selection_order",
    "selection_category",
    "structure_id",
    "standardized_smiles",
    "p_activity_median",
    "series_id",
    "herg_evidence_status",
}
STRATUM_PRIORITY = {
    "hybrid_escalates_to_potent": 0,
    "hybrid_other_risk_escalation": 1,
    "hybrid_risk_deescalation": 2,
    "consensus_potent": 3,
    "consensus_moderate": 4,
    "consensus_safe": 5,
    "ligand_only": 6,
}


class CampaignError(RuntimeError):
    """Raised when prospective input, inference, or ranking integrity fails."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(repo: Path, path: Path) -> Path:
    return v139._resolve(repo, path)  # noqa: SLF001


def _sha(path: Path) -> str:
    return v139._sha(path)  # noqa: SLF001


def _json(path: Path, payload: dict[str, Any], field: str) -> dict[str, Any]:
    return v139._json(path, payload, field)  # noqa: SLF001


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    v139._parquet(path, frame)  # noqa: SLF001


def _prepare(source: Path, output: Path, panel_size: int) -> dict[str, Any]:
    frame = pd.read_csv(source)
    missing = REQUIRED_COLUMNS - set(frame)
    if missing:
        raise CampaignError(f"prospective plan is missing columns: {sorted(missing)}")
    if frame.structure_id.duplicated().any() or frame.standardized_smiles.duplicated().any():
        raise CampaignError("prospective plan must contain unique structures and SMILES")
    observed = frame.herg_evidence_status.fillna("").str.lower().str.startswith("observed")
    if observed.any():
        raise CampaignError("observed hERG evidence is forbidden from the prospective panel")
    if panel_size < 1 or panel_size > len(frame):
        raise CampaignError("panel size must be between one and the prospective row count")
    registry = frame.sort_values("selection_order").reset_index(drop=True)
    registry_path = output / "prepared/prospective_registry.parquet"
    _parquet(registry_path, registry)
    return _json(
        output / "prepared/selection_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "source_path": str(source.resolve()),
            "source_sha256": _sha(source),
            "registry_sha256": _sha(registry_path),
            "rows": len(registry),
            "unique_series": int(registry.series_id.nunique()),
            "observed_herg_rows": int(observed.sum()),
            "panel_size": panel_size,
            "ranking_policy": (
                "one representative per source selection category, then distinct series, "
                "then remaining rows; lexicographic hybrid-risk stratum, ligand/receptor "
                "Jensen-Shannon divergence, receptor Potent probability, Menin pActivity"
            ),
            "outcomes_used_for_ranking": False,
        },
        "contract_sha256",
    )


def _predict(
    repo: Path,
    bundle_root: Path,
    output: Path,
    vina: Path | None,
    cpu: int,
) -> dict[str, Any]:
    registry = pd.read_parquet(output / "prepared/prospective_registry.parquet")
    return v139._predict(  # noqa: SLF001
        repo,
        bundle_root,
        output / "inference",
        registry.standardized_smiles.astype(str).tolist(),
        vina,
        cpu,
    )


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.clip(np.asarray(left, dtype=float), 1e-12, 1.0)
    right = np.clip(np.asarray(right, dtype=float), 1e-12, 1.0)
    left = left / left.sum(axis=1, keepdims=True)
    right = right / right.sum(axis=1, keepdims=True)
    midpoint = 0.5 * (left + right)
    divergence = 0.5 * np.sum(left * np.log2(left / midpoint), axis=1)
    divergence += 0.5 * np.sum(right * np.log2(right / midpoint), axis=1)
    return divergence


def _challenge_stratum(ligand: int, hybrid: int | None) -> str:
    if hybrid is None:
        return "ligand_only"
    if hybrid == 2 and ligand != 2:
        return "hybrid_escalates_to_potent"
    if hybrid > ligand:
        return "hybrid_other_risk_escalation"
    if hybrid < ligand:
        return "hybrid_risk_deescalation"
    return ("consensus_safe", "consensus_moderate", "consensus_potent")[hybrid]


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["router_hybrid_disagreement"] = (
        result.receptor_prediction_available.astype(bool)
        & result.hybrid_prediction.ne(result.ligand_router_prediction)
    )
    ligand_probability = result[list(v139.v136.LGBM_COLUMNS)].to_numpy(float)
    receptor_columns = [
        "receptor_probability_safe",
        "receptor_probability_moderate",
        "receptor_probability_potent",
    ]
    available = result.receptor_prediction_available.to_numpy(bool)
    divergence = np.full(len(result), np.nan)
    divergence[available] = _jensen_shannon(
        ligand_probability[available], result.loc[available, receptor_columns].to_numpy(float)
    )
    result["ligand_receptor_js_divergence_bits"] = divergence
    strata = []
    for row in result.itertuples(index=False):
        hybrid = int(row.hybrid_prediction) if pd.notna(row.hybrid_prediction) else None
        strata.append(_challenge_stratum(int(row.ligand_router_prediction), hybrid))
    result["model_challenge_stratum"] = strata
    result["model_challenge_priority"] = result.model_challenge_stratum.map(STRATUM_PRIORITY)
    result = result.sort_values(
        [
            "model_challenge_priority",
            "ligand_receptor_js_divergence_bits",
            "receptor_probability_potent",
            "p_activity_median",
            "selection_order",
        ],
        ascending=[True, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    result["model_challenge_rank"] = np.arange(1, len(result) + 1)
    return result


def _select_panel(ranked: pd.DataFrame, panel_size: int) -> pd.DataFrame:
    chosen: list[int] = []
    chosen_set: set[int] = set()
    used_series: set[str] = set()

    def take(index: int) -> None:
        if index not in chosen_set and len(chosen) < panel_size:
            chosen.append(index)
            chosen_set.add(index)
            used_series.add(str(ranked.loc[index, "series_id"]))

    # Preserve the original project's six experimental design categories.
    for category in ranked.selection_category.drop_duplicates():
        candidates = ranked.index[ranked.selection_category.eq(category)]
        if len(candidates):
            take(int(candidates[0]))
    # Maximise series diversity before allowing a second analogue.
    for index, row in ranked.iterrows():
        if str(row.series_id) not in used_series:
            take(int(index))
    for index in ranked.index:
        take(int(index))
    panel = ranked.loc[chosen].sort_values("model_challenge_rank").reset_index(drop=True)
    panel.insert(0, "assay_wave_order", np.arange(1, len(panel) + 1))
    return panel


def _write_blinded_outcome_template(output: Path, panel: pd.DataFrame) -> dict[str, Any]:
    template = panel[["structure_id", "selection_category"]].copy()
    template.insert(
        0,
        "blinded_sample_id",
        [
            "H13-" + hashlib.sha256(f"v13.10|{value}".encode()).hexdigest()[:10].upper()
            for value in template.structure_id.astype(str)
        ],
    )
    for column in (
        "measured_ic10_nm",
        "measured_ic30_nm",
        "measured_ic50_nm",
        "assay_temperature_c",
        "cell_system",
        "voltage_protocol_id",
        "biological_replicate_count",
        "qc_pass",
        "assay_notes",
    ):
        template[column] = ""
    template = template.sort_values("blinded_sample_id").reset_index(drop=True)
    path = output / "analysis/blinded_herg_outcome_release.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, index=False)
    return _json(
        output / "analysis/blinded_outcome_release_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "rows": len(template),
            "template_sha256": _sha(path),
            "predictions_or_ranks_present": False,
            "required_endpoint_order": "IC10 <= IC30 <= IC50 concentrations",
            "unblinding_rule": (
                "lock the completed outcome file and its SHA-256 before joining to predictions"
            ),
        },
        "contract_sha256",
    )


def _render_report(report: dict[str, Any]) -> str:
    counts = report["challenge_strata"]
    lines = [
        "# hERG V13.10 Menin receptor-aware prospective prioritization",
        "",
        f"Status: **{report['status']}**  ",
        f"Created: {report['created_utc']}",
        "",
        "## Integrity",
        "",
        f"All {report['input_rows']} compounds had no observed hERG outcome in the supplied plan. "
        "The frozen V13.9 ligand/Vina/receptor bundle generated every ranking field.",
        "",
        "## Model-challenge composition",
        "",
        "| Stratum | Full plan | Assay panel |",
        "|---|---:|---:|",
    ]
    for name in STRATUM_PRIORITY:
        lines.append(
            f"| {name} | {counts.get(name, 0)} | "
            f"{report['panel_challenge_strata'].get(name, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Use",
            "",
            "The selected panel is designed to test the model where receptor and ligand-only evidence agree or disagree while retaining source-category and chemical-series diversity. Run blinded experimental hERG concentration-response measurements and release IC10/IC30/IC50 together. Follow the [ICH E14/S7B Step 4 in-vitro best-practice considerations](https://database.ich.org/sites/default/files/E14-S7B_QAs_Step4_2022_0221.pdf), including near-physiological recording temperature, a physiologically relevant voltage protocol, appropriate controls, and documented concentration/QC procedures. Predictions are research hypotheses, not experimental measurements, clinical conclusions, or proof of a docking pose.",
            "",
        ]
    )
    return "\n".join(lines)


def _analyze(output: Path, panel_size: int) -> dict[str, Any]:
    contract = v139.v137._read_json(  # noqa: SLF001
        output / "prepared/selection_contract.json", "contract_sha256"
    )
    registry_path = output / "prepared/prospective_registry.parquet"
    if _sha(registry_path) != contract["registry_sha256"]:
        raise CampaignError("prospective registry changed after the selection contract was sealed")
    registry = pd.read_parquet(registry_path)
    predictions = pd.read_parquet(output / "inference/predictions.parquet")
    frame = registry.merge(
        predictions,
        left_on="standardized_smiles",
        right_on="input_smiles",
        how="left",
        validate="one_to_one",
        suffixes=("", "__inference"),
    )
    if len(frame) != len(registry) or frame.ligand_router_prediction.isna().any():
        raise CampaignError("inference coverage is incomplete")
    ranked = _rank(frame)
    panel = _select_panel(ranked, panel_size)
    ranked_path = output / "analysis/all_48_ranked_predictions.parquet"
    panel_path = output / "analysis/blinded_assay_challenge_panel.parquet"
    _parquet(ranked_path, ranked)
    _parquet(panel_path, panel)
    outcome_contract = _write_blinded_outcome_template(output, panel)
    report = _json(
        output / "analysis_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete_label_blind_prospective_prioritization",
            "input_rows": len(ranked),
            "receptor_predictions": int(ranked.receptor_prediction_available.sum()),
            "router_hybrid_disagreements": int(ranked.router_hybrid_disagreement.sum()),
            "challenge_strata": ranked.model_challenge_stratum.value_counts().to_dict(),
            "panel_rows": len(panel),
            "panel_unique_series": int(panel.series_id.nunique()),
            "panel_source_categories": int(panel.selection_category.nunique()),
            "panel_challenge_strata": panel.model_challenge_stratum.value_counts().to_dict(),
            "ranked_predictions_sha256": _sha(ranked_path),
            "challenge_panel_sha256": _sha(panel_path),
            "selection_contract_sha256": contract["contract_sha256"],
            "blinded_outcome_release_contract_sha256": outcome_contract[
                "contract_sha256"
            ],
            "observed_herg_outcomes_used": False,
            "wet_lab_measurements_performed": False,
            "claim_boundary": (
                "label-blind experimental prioritization; predictions are not measurements, "
                "clinical conclusions, or mechanistic proof"
            ),
        },
        "report_sha256",
    )
    (output / "REPORT.md").write_text(_render_report(report))
    return report


def _manifest(repo: Path, source: Path, bundle_root: Path, output: Path) -> dict[str, Any]:
    inputs = [
        repo / "pipeline/scripts/run_local_herg_menin_receptor_prioritization_v13_10.py",
        repo / "pipeline/scripts/build_local_herg_receptor_inference_bundle_v13_9.py",
        source,
        bundle_root / "manifest.json",
        bundle_root / "verification_report.json",
    ]
    artifacts = [
        path
        for path in (
            output / "prepared/prospective_registry.parquet",
            output / "prepared/selection_contract.json",
            output / "inference/predictions.parquet",
            output / "inference/prediction_report.json",
            output / "analysis/all_48_ranked_predictions.parquet",
            output / "analysis/blinded_assay_challenge_panel.parquet",
            output / "analysis/blinded_herg_outcome_release.csv",
            output / "analysis/blinded_outcome_release_contract.json",
            output / "analysis_report.json",
            output / "REPORT.md",
        )
        if path.is_file()
    ]
    return _json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete" if (output / "analysis_report.json").is_file() else "in_progress",
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
    parser.add_argument("--stage", choices=("prepare", "predict", "analyze", "all"), default="all")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vina", type=Path)
    parser.add_argument("--cpu", type=int, default=6)
    parser.add_argument("--panel-size", type=int, default=DEFAULT_PANEL_SIZE)
    return parser


def _main(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    source = _resolve(repo, args.source)
    bundle_root = _resolve(repo, args.bundle_root)
    output = _resolve(repo, args.output_root)
    vina = args.vina.resolve() if args.vina else None
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "stage": args.stage}
    if args.stage in ("prepare", "all"):
        result["prepare"] = _prepare(source, output, args.panel_size)
    if args.stage in ("predict", "all"):
        result["predict"] = _predict(repo, bundle_root, output, vina, args.cpu)
    if args.stage in ("analyze", "all"):
        result["analyze"] = _analyze(output, args.panel_size)
    result["manifest"] = _manifest(repo, source, bundle_root, output)
    return result


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _main(args)
    except (CampaignError, v139.CampaignError, v139.v13.CampaignError) as exc:
        print(f"V13.10 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
