#!/usr/bin/env python3
"""Dock the fixed V13 pilot into the two withheld holo sensitivity states.

The primary V13 four-state artifact is immutable. This extension reuses its
bound receptor preparation, ligand-parent policy, and one retained pH-7.4
microstate, but writes every sensitivity pose and table under a separate root.
It tests whether astemizole- and pimozide-conditioned states contain a signal
that the apo/E-4031/low-K/high-K core ensemble missed.
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
import run_local_herg_receptor_ensemble_campaign_v13 as v13

SCHEMA_VERSION = "platform-local-herg-receptor-sensitivity-v13.1/1.0"
SENSITIVITY_STATES = ("8ZYO", "8ZYQ")


class CampaignError(RuntimeError):
    """Sensitivity-docking integrity or execution failure."""


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


def _dock(
    primary: Path,
    output: Path,
    tools: v13.Toolchain,
    exhaustiveness: int,
    modes: int,
    cpu: int,
) -> pd.DataFrame:
    redocking = v13._read_self_hashed_json(  # noqa: SLF001
        primary / "redocking/redocking_report.json", "report_sha256"
    )
    if not redocking["primary_gate_passed"]:
        raise CampaignError("primary native-redocking gate is not passed")
    selected = pd.read_parquet(primary / "pilot/pilot_selection.parquet")
    receptor_atoms = {
        state: v13._receptor_contact_atoms(  # noqa: SLF001
            primary / f"receptors/{state}/{state}_prepared.pdb"
        )
        for state in SENSITIVITY_STATES
    }
    rows: list[dict[str, Any]] = []
    total = len(selected) * len(SENSITIVITY_STATES)
    completed = 0
    for ligand in selected.itertuples(index=False):
        microstates = v13._prepare_ligand_microstates(ligand, primary, tools, 1)  # noqa: SLF001
        if len(microstates) != 1:
            raise CampaignError(f"primary one-microstate contract changed for {ligand.ligand_id}")
        microstate = microstates[0]
        for state in SENSITIVITY_STATES:
            receptor = primary / f"receptors/{state}/{state}_prepared.pdbqt"
            config = primary / f"receptors/{state}/{state}_prepared.box.txt"
            task_id = (
                f"{v13._safe_component(ligand.ligand_id)}"  # noqa: SLF001
                f"__m{microstate['microstate_index']}__{state}"
            )
            directory = output / "tasks" / task_id
            directory.mkdir(parents=True, exist_ok=True)
            docked = directory / "poses.pdbqt"
            seed = v13.SEED + int(hashlib.sha256(task_id.encode()).hexdigest()[:7], 16)
            if not docked.is_file():
                v13._run(  # noqa: SLF001
                    [
                        tools.vina,
                        "--receptor",
                        receptor,
                        "--ligand",
                        microstate["pdbqt_path"],
                        "--config",
                        config,
                        "--exhaustiveness",
                        str(exhaustiveness),
                        "--num_modes",
                        str(modes),
                        "--seed",
                        str(seed),
                        "--cpu",
                        str(cpu),
                        "--out",
                        docked,
                    ],
                    directory / "vina.log",
                )
            models = v13._pdbqt_models(docked)  # noqa: SLF001
            affinities = np.asarray([model[0] for model in models], dtype=float)
            best_index = int(np.argmin(affinities))
            feature = v13._contact_features(  # noqa: SLF001
                models[best_index][1], receptor_atoms[state]
            )
            rows.append(
                {
                    "ligand_id": ligand.ligand_id,
                    "cohort": ligand.cohort,
                    "outer_fold": int(ligand.outer_fold),
                    "scaffold_group_id": ligand.scaffold_group_id,
                    "pdb_id": state,
                    "microstate_index": microstate["microstate_index"],
                    "microstate_smiles": microstate["microstate_smiles"],
                    "formal_charge": microstate["formal_charge"],
                    "heavy_atom_count": microstate["heavy_atom_count"],
                    "seed": seed,
                    "pose_count": len(models),
                    "best_affinity_kcal_mol": float(np.min(affinities)),
                    "median_affinity_kcal_mol": float(np.median(affinities)),
                    "affinity_range_kcal_mol": float(np.max(affinities) - np.min(affinities)),
                    "pose_count_within_1kcal": int(
                        np.sum(affinities <= np.min(affinities) + 1.0)
                    ),
                    "ligand_efficiency_kcal_mol_per_heavy_atom": float(
                        np.min(affinities) / max(1, microstate["heavy_atom_count"])
                    ),
                    **feature,
                }
            )
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(
                    json.dumps({"stage": "sensitivity_dock", "completed": completed, "total": total}),
                    flush=True,
                )
            _parquet(output / "docking_results.partial.parquet", pd.DataFrame(rows))
    frame = pd.DataFrame(rows)
    _parquet(output / "docking_results.parquet", frame)
    partial = output / "docking_results.partial.parquet"
    if partial.is_file():
        partial.unlink()
    return frame


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--primary-root",
        type=Path,
        default=Path("research/local_runs/herg_receptor_ensemble_campaign_v13"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/local_runs/herg_receptor_sensitivity_campaign_v13_1"),
    )
    parser.add_argument("--vina", type=Path)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--modes", type=int, default=9)
    parser.add_argument("--cpu", type=int, default=6)
    return parser


def _main(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    primary = args.primary_root if args.primary_root.is_absolute() else repo / args.primary_root
    output = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    primary = primary.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tools = v13._resolve_toolchain(repo, args.vina.resolve() if args.vina else None)  # noqa: SLF001
    if "1.2." not in v13._tool_version(tools.vina, "--version"):  # noqa: SLF001
        raise CampaignError("V13.1 requires AutoDock Vina 1.2.x")
    frame = _dock(primary, output, tools, args.exhaustiveness, args.modes, args.cpu)
    expected = len(pd.read_parquet(primary / "pilot/pilot_selection.parquet")) * 2
    if len(frame) != expected or frame.ligand_id.nunique() * 2 != expected:
        raise CampaignError("sensitivity receptor coverage is incomplete")
    inputs = [
        repo / "pipeline/scripts/run_local_herg_receptor_sensitivity_docking_v13_1.py",
        primary / "manifest.json",
        primary / "pilot/pilot_selection.parquet",
        *(primary / f"receptors/{state}/{state}_prepared.pdbqt" for state in SENSITIVITY_STATES),
    ]
    artifact = output / "docking_results.parquet"
    manifest = _json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete_withheld_holo_sensitivity_docking",
            "states": list(SENSITIVITY_STATES),
            "exhaustiveness": args.exhaustiveness,
            "modes": args.modes,
            "rows": len(frame),
            "ligands": int(frame.ligand_id.nunique()),
            "vina": str(tools.vina),
            "vina_sha256": _sha(tools.vina),
            "inputs": [
                {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}
                for path in inputs
            ],
            "artifacts": [
                {
                    "path": str(artifact.resolve()),
                    "bytes": artifact.stat().st_size,
                    "sha256": _sha(artifact),
                }
            ],
            "truth_boundary": (
                "withheld-state internal sensitivity analysis; Vina scores are observables, not affinities"
            ),
        },
        "manifest_sha256",
    )
    return _json(
        output / "final_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "finished_utc": _utc(),
            "status": "complete",
            "states": list(SENSITIVITY_STATES),
            "rows": len(frame),
            "ligands": int(frame.ligand_id.nunique()),
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "summary_sha256",
    )


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _main(args)
    except (CampaignError, v13.CampaignError) as exc:
        print(f"V13.1 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
