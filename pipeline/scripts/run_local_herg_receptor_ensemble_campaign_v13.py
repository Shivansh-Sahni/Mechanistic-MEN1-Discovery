#!/usr/bin/env python3
"""Run the hERG receptor-ensemble docking and incremental-value V13 campaign.

V13 is deliberately a receptor-aware *pilot*, not an affinity claim.  It:

1. prepares a common-coordinate intersection of six deposited hERG states;
2. protonates every receptor with one PDB2PQR/PROPKA policy at pH 7.4;
3. requires native-ligand redocking controls for the three holo structures;
4. docks scaffold-diverse, fold-balanced IC50 and direct IC10/IC30/IC50 panels;
5. measures whether receptor features add value to already out-of-fold ligand
   predictions under the original held-out scaffold folds; and
6. includes shuffled-feature negative controls and explicit claim boundaries.

AutoDock Vina scores are treated as scoring-function observables.  They are not
binding free energies, experimental affinities, or evidence of a unique pose.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, MMCIFParser, Select
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolAlign, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCHEMA_VERSION = "platform-local-herg-receptor-ensemble-v13/1.0"
SEED = 20260820
RECEPTOR_IDS = ("8ZYN", "8ZYP", "9CHP", "9CHQ", "8ZYO", "8ZYQ")
CORE_IDS = ("8ZYN", "8ZYP", "9CHP", "9CHQ")
SENSITIVITY_IDS = ("8ZYO", "8ZYQ")
HOLO_SPECS = {
    "8ZYO": ("XB7", "astemizole"),
    "8ZYP": ("A1L2J", "E-4031"),
    "8ZYQ": ("1II", "pimozide"),
}
POCKET_RESIDUES = (621, 622, 623, 624, 645, 648, 649, 652, 656)
CONTACT_RESIDUES = (623, 624, 649, 652, 656)
DOCKABLE_ATOMIC_NUMBERS = frozenset((1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53))
CLASS_NAMES = ("Safe", "Moderate", "Potent")
SAFE_PIC50 = 6.0 - math.log10(30.0)
POTENT_PIC50 = 6.0
VINA_RESULT = re.compile(r"^REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)")


class CampaignError(RuntimeError):
    """A scientific-contract, integrity, or execution failure."""


@dataclass(frozen=True)
class Toolchain:
    vina: Path
    pdb2pqr: Path
    meeko_receptor: Path
    meeko_ligand: Path
    meeko_export: Path
    molscrub: Path
    obabel: Path


class _Lock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> _Lock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError(f"V13 campaign already running: {self.path}") from exc
        return self

    def __exit__(self, *_args: object) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


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


def _self_hashed_json(path: Path, payload: dict[str, Any], field: str) -> dict[str, Any]:
    body = dict(payload)
    body.pop(field, None)
    body[field] = hashlib.sha256(_canonical(body)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(body))
    temporary.replace(path)
    return body


def _read_self_hashed_json(path: Path, field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    expected = str(payload.pop(field))
    actual = hashlib.sha256(_canonical(payload)).hexdigest()
    payload[field] = expected
    if expected != actual:
        raise CampaignError(f"self-hash mismatch: {path}")
    return payload


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _run(command: Sequence[str | Path], log: Path | None = None) -> str:
    args = [str(value) for value in command]
    completed = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = completed.stdout or ""
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        temporary = log.with_suffix(log.suffix + ".tmp")
        temporary.write_text(output)
        temporary.replace(log)
    if completed.returncode:
        tail = "\n".join(output.splitlines()[-80:])
        raise CampaignError(f"command failed ({completed.returncode}): {' '.join(args)}\n{tail}")
    return output


def _tool_version(path: Path, *arguments: str) -> str:
    try:
        output = _run([path, *arguments])
    except CampaignError:
        return "unavailable"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[0] if lines else "unknown"


def _resolve_toolchain(repo: Path, vina: Path | None) -> Toolchain:
    virtual = repo / ".venv/bin"

    def required(name: str, explicit: Path | None = None) -> Path:
        candidate = explicit or (virtual / name if (virtual / name).is_file() else None)
        if candidate is None:
            found = shutil.which(name)
            candidate = Path(found) if found else None
        if candidate is None or not candidate.is_file():
            raise CampaignError(f"required executable unavailable: {name}")
        return candidate.resolve()

    preferred_vina = repo / ".codex_tmp/tools/vina_1.2.7_mac_aarch64"
    resolved_vina = vina or (preferred_vina if preferred_vina.is_file() else None)
    return Toolchain(
        vina=required("vina", resolved_vina),
        pdb2pqr=required("pdb2pqr30"),
        meeko_receptor=required("mk_prepare_receptor.py"),
        meeko_ligand=required("mk_prepare_ligand.py"),
        meeko_export=required("mk_export.py"),
        molscrub=required("scrub.py"),
        obabel=required("obabel"),
    )


def _coordinate_paths(repo: Path) -> dict[str, Path]:
    miyashita = repo / "research/literature/herg/structural_biology/2024_miyashita_inhibitor_bound"
    lau = repo / "research/literature/herg/structural_biology/2024_lau_potassium_states"
    paths = {identifier: miyashita / f"{identifier}.cif" for identifier in ("8ZYN", "8ZYO", "8ZYP", "8ZYQ")}
    paths.update({identifier: lau / f"{identifier}.cif" for identifier in ("9CHP", "9CHQ")})
    for identifier, path in paths.items():
        if not path.is_file():
            raise CampaignError(f"missing {identifier} coordinate: {path}")
    return paths


def _common_residue_ids(paths: dict[str, Path]) -> tuple[int, ...]:
    parser = MMCIFParser(QUIET=True)
    residue_sets: list[set[int]] = []
    for identifier in RECEPTOR_IDS:
        model = next(parser.get_structure(identifier, paths[identifier]).get_models())
        chains = {chain.id: chain for chain in model if chain.id in "ABCD"}
        if set(chains) != set("ABCD"):
            raise CampaignError(f"{identifier} does not contain the required A/B/C/D tetramer")
        for chain in chains.values():
            residue_sets.append({residue.id[1] for residue in chain if residue.id[0] == " "})
    common = tuple(sorted(set.intersection(*residue_sets)))
    if len(common) < 150 or not set(POCKET_RESIDUES).issubset(common):
        raise CampaignError("common receptor intersection lost required cavity coverage")
    return common


class _ReceptorSelect(Select):
    def __init__(self, residue_ids: set[int]) -> None:
        self.residue_ids = residue_ids

    def accept_chain(self, chain: Any) -> bool:
        return chain.id in "ABCD"

    def accept_residue(self, residue: Any) -> bool:
        return residue.id[0] == " " and residue.id[1] in self.residue_ids

    def accept_atom(self, atom: Any) -> bool:
        return atom.element != "H" and atom.altloc in (" ", "A")


class _LigandSelect(Select):
    def accept_residue(self, residue: Any) -> bool:
        return residue.resname == "LIG"

    def accept_atom(self, atom: Any) -> bool:
        return atom.element != "H" and atom.altloc in (" ", "A")


def _pocket_center(model: Any) -> tuple[float, float, float]:
    coordinates = []
    for chain in model:
        if chain.id not in "ABCD":
            continue
        for residue in chain:
            if residue.id[0] == " " and residue.id[1] in POCKET_RESIDUES:
                coordinates.extend(atom.coord for atom in residue if atom.element != "H")
    if len(coordinates) < 100:
        raise CampaignError("too few key-pocket atoms to define docking box")
    center = np.mean(np.asarray(coordinates, dtype=float), axis=0)
    return tuple(float(value) for value in center)


def _write_raw_receptor(cif_path: Path, output: Path, common: set[int]) -> tuple[float, float, float]:
    structure = MMCIFParser(QUIET=True).get_structure(cif_path.stem, cif_path)
    model = next(structure.get_models())
    center = _pocket_center(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(model)
    io.save(str(output), _ReceptorSelect(common))
    return center


def _prepare_receptors(
    repo: Path,
    output: Path,
    tools: Toolchain,
    box_size: tuple[float, float, float],
) -> dict[str, Any]:
    paths = _coordinate_paths(repo)
    common = _common_residue_ids(paths)
    rows: list[dict[str, Any]] = []
    for identifier in RECEPTOR_IDS:
        directory = output / "receptors" / identifier
        directory.mkdir(parents=True, exist_ok=True)
        raw_pdb = directory / f"{identifier}_common.pdb"
        pqr = directory / f"{identifier}_ph7p4.pqr"
        basename = directory / f"{identifier}_prepared"
        pdbqt = basename.with_suffix(".pdbqt")
        prepared_pdb = directory / f"{identifier}_prepared.pdb"
        if not raw_pdb.is_file():
            center = _write_raw_receptor(paths[identifier], raw_pdb, set(common))
        else:
            model = next(MMCIFParser(QUIET=True).get_structure(identifier, paths[identifier]).get_models())
            center = _pocket_center(model)
        if not pqr.is_file():
            _run(
                [
                    tools.pdb2pqr,
                    "--ff=AMBER",
                    "--with-ph=7.4",
                    "--keep-chain",
                    "--drop-water",
                    "--titration-state-method=propka",
                    raw_pdb,
                    pqr,
                ],
                directory / "pdb2pqr.log",
            )
        if not pdbqt.is_file():
            _run(
                [
                    tools.meeko_receptor,
                    "--read_pqr",
                    pqr,
                    "-o",
                    basename,
                    "--write_pdbqt",
                    "--write_json",
                    "--write_pdb",
                    prepared_pdb,
                    "-a",
                    "--charge_model",
                    "read",
                    "--box_center",
                    *[f"{value:.6f}" for value in center],
                    "--box_size",
                    *[str(value) for value in box_size],
                    "--write_vina_box",
                ],
                directory / "meeko_receptor.log",
            )
        required = [pdbqt, basename.with_suffix(".box.txt"), prepared_pdb]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise CampaignError(f"incomplete prepared receptor for {identifier}")
        rows.append(
            {
                "pdb_id": identifier,
                "coordinate_sha256": _sha(paths[identifier]),
                "common_pdb_sha256": _sha(raw_pdb),
                "pqr_sha256": _sha(pqr),
                "pdbqt_sha256": _sha(pdbqt),
                "box_center_x": center[0],
                "box_center_y": center[1],
                "box_center_z": center[2],
                "box_size_x": box_size[0],
                "box_size_y": box_size[1],
                "box_size_z": box_size[2],
                "production_tier": "core" if identifier in CORE_IDS else "sensitivity",
            }
        )
    frame = pd.DataFrame(rows)
    _parquet(output / "receptors/receptor_preparation.parquet", frame)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "status": "prepared_common_coordinate_intersection",
        "common_residue_count_per_chain": len(common),
        "common_residue_ids": list(common),
        "receptor_count": len(frame),
        "ph": 7.4,
        "force_field": "AMBER through PDB2PQR; PROPKA titration state assignment",
        "waters": "no resolved waters in deposits; PDB2PQR drop-water policy fixed",
        "membrane": "not represented in rigid docking",
        "tool_versions": {
            "vina": _tool_version(tools.vina, "--version"),
            "pdb2pqr": _tool_version(tools.pdb2pqr, "--version"),
            "obabel": _tool_version(tools.obabel, "-V"),
        },
        "truth_boundary": (
            "common-coordinate rigid receptors remove gross construct-coverage differences but do not "
            "supply membrane equilibration, water networks, induced fit, map-level uncertainty, or kinetics"
        ),
    }
    return _self_hashed_json(output / "receptors/receptor_preparation.json", payload, "report_sha256")


def _extract_native_ligand(cif_path: Path, resname: str, pdb_path: Path) -> None:
    structure = MMCIFParser(QUIET=True).get_structure(cif_path.stem, cif_path)
    model = next(structure.get_models())
    found = 0
    for chain in model:
        for residue in chain:
            if residue.resname == resname:
                residue.resname = "LIG"
                found += 1
    if found != 1:
        raise CampaignError(f"expected one native {resname} residue in {cif_path}, found {found}")
    io = PDBIO()
    io.set_structure(model)
    io.save(str(pdb_path), _LigandSelect())


def _pdbqt_models(path: Path) -> list[tuple[float, np.ndarray]]:
    models: list[tuple[float, np.ndarray]] = []
    score: float | None = None
    coordinates: list[list[float]] = []
    for line in path.read_text().splitlines():
        match = VINA_RESULT.match(line)
        if match:
            if score is not None and coordinates:
                models.append((score, np.asarray(coordinates, dtype=float)))
            score = float(match.group(1))
            coordinates = []
        elif line.startswith(("ATOM  ", "HETATM")):
            atom_type = line[77:].strip().split()[0] if line[77:].strip() else ""
            if not atom_type.startswith("H"):
                coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if score is not None and coordinates:
        models.append((score, np.asarray(coordinates, dtype=float)))
    if not models:
        raise CampaignError(f"no Vina models parsed: {path}")
    return models


def _native_rmsds(reference_sdf: Path, docked_sdf: Path) -> list[float]:
    reference = next((mol for mol in Chem.SDMolSupplier(str(reference_sdf), removeHs=True) if mol), None)
    poses = [mol for mol in Chem.SDMolSupplier(str(docked_sdf), removeHs=True) if mol]
    if reference is None or not poses:
        raise CampaignError("native redocking SDF conversion failed")
    rmsds = []
    for pose in poses:
        try:
            rmsds.append(float(rdMolAlign.GetBestRMS(reference, pose)))
        except (RuntimeError, ValueError) as exc:
            raise CampaignError("native and docked ligand graphs are incompatible") from exc
    return rmsds


def _redock(
    repo: Path,
    output: Path,
    tools: Toolchain,
    exhaustiveness: int,
    seeds: int,
    cpu: int,
) -> dict[str, Any]:
    coordinates = _coordinate_paths(repo)
    rows: list[dict[str, Any]] = []
    for pdb_id, (resname, ligand_name) in HOLO_SPECS.items():
        directory = output / "redocking" / pdb_id
        directory.mkdir(parents=True, exist_ok=True)
        native_pdb = directory / "native_ligand.pdb"
        native_sdf = directory / "native_ligand.sdf"
        native_pdbqt = directory / "native_ligand.pdbqt"
        if not native_pdb.is_file():
            _extract_native_ligand(coordinates[pdb_id], resname, native_pdb)
        if not native_sdf.is_file():
            _run([tools.obabel, native_pdb, "-O", native_sdf, "-h", "--partialcharge", "gasteiger"], directory / "obabel.log")
        if not native_pdbqt.is_file():
            _run(
                [tools.meeko_ligand, "-i", native_sdf, "-o", native_pdbqt, "--add_index_map", "--charge_model", "gasteiger"],
                directory / "meeko_ligand.log",
            )
        receptor = output / f"receptors/{pdb_id}/{pdb_id}_prepared.pdbqt"
        config = output / f"receptors/{pdb_id}/{pdb_id}_prepared.box.txt"
        for replicate in range(seeds):
            seed = SEED + 1_000 * list(HOLO_SPECS).index(pdb_id) + replicate
            docked = directory / f"redock_seed{seed}.pdbqt"
            vina_log = directory / f"redock_seed{seed}.log"
            docked_sdf = directory / f"redock_seed{seed}.sdf"
            if not docked.is_file():
                _run(
                    [
                        tools.vina,
                        "--receptor",
                        receptor,
                        "--ligand",
                        native_pdbqt,
                        "--config",
                        config,
                        "--exhaustiveness",
                        str(exhaustiveness),
                        "--num_modes",
                        "20",
                        "--seed",
                        str(seed),
                        "--cpu",
                        str(cpu),
                        "--out",
                        docked,
                    ],
                    vina_log,
                )
            if not docked_sdf.is_file():
                _run([tools.meeko_export, docked, "-s", docked_sdf], directory / f"export_seed{seed}.log")
            models = _pdbqt_models(docked)
            rmsds = _native_rmsds(native_sdf, docked_sdf)
            if len(models) != len(rmsds):
                raise CampaignError("redocking pose-count mismatch")
            best_score_index = int(np.argmin([row[0] for row in models]))
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "ligand_name": ligand_name,
                    "seed": seed,
                    "pose_count": len(models),
                    "best_affinity_kcal_mol": float(models[best_score_index][0]),
                    "top_scored_pose_rmsd_angstrom": rmsds[best_score_index],
                    "best_sampled_rmsd_angstrom": float(min(rmsds)),
                    "any_pose_le_2p5": bool(min(rmsds) <= 2.5),
                    "top_pose_le_3p0": bool(rmsds[best_score_index] <= 3.0),
                }
            )
            print(
                json.dumps(
                    {
                        "stage": "redock",
                        "pdb_id": pdb_id,
                        "seed": seed,
                        "top_rmsd": rmsds[best_score_index],
                        "best_rmsd": min(rmsds),
                    }
                ),
                flush=True,
            )
    frame = pd.DataFrame(rows)
    _parquet(output / "redocking/redocking_results.parquet", frame)
    summaries = []
    for pdb_id, group in frame.groupby("pdb_id"):
        summaries.append(
            {
                "pdb_id": pdb_id,
                "replicates": len(group),
                "successful_any_pose_replicates": int(group.any_pose_le_2p5.sum()),
                "successful_top_pose_replicates": int(group.top_pose_le_3p0.sum()),
                "median_top_pose_rmsd_angstrom": float(group.top_scored_pose_rmsd_angstrom.median()),
                "median_best_sampled_rmsd_angstrom": float(group.best_sampled_rmsd_angstrom.median()),
                "redocking_gate_passed": bool(group.any_pose_le_2p5.mean() >= 2 / 3),
            }
        )
    summary_frame = pd.DataFrame(summaries)
    _parquet(output / "redocking/redocking_summary.parquet", summary_frame)
    passed = int(summary_frame.redocking_gate_passed.sum())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "holo_receptors": len(summary_frame),
        "holo_receptors_passing": passed,
        "primary_gate_passed": passed >= 2,
        "success_rule": "at least two of three holo receptors have native-pose RMSD <=2.5 A in >=2/3 seeds",
        "exhaustiveness": exhaustiveness,
        "truth_boundary": (
            "redocking checks search/preparation compatibility for three known binders; it does not validate "
            "affinity ranking, novel-chemotype pose accuracy, or physiological channel-state occupancy"
        ),
    }
    return _self_hashed_json(output / "redocking/redocking_report.json", payload, "report_sha256")


def _tier_index(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values < SAFE_PIC50, 0, np.where(values > POTENT_PIC50, 2, 1)).astype(int)


def _stable_order(values: Iterable[str], salt: str) -> np.ndarray:
    hashes = [hashlib.sha256(f"{SEED}|{salt}|{value}".encode()).hexdigest() for value in values]
    return np.argsort(hashes)


def _select_diverse(group: pd.DataFrame, n: int, id_column: str, salt: str) -> pd.DataFrame:
    ordered = group.iloc[_stable_order(group[id_column].astype(str), salt)].copy()
    diverse = ordered.drop_duplicates("scaffold_group_id", keep="first")
    if len(diverse) >= n:
        return diverse.head(n)
    remainder = ordered.loc[~ordered[id_column].isin(diverse[id_column])]
    selected = pd.concat([diverse, remainder.head(n - len(diverse))], ignore_index=True)
    if len(selected) != n:
        raise CampaignError(f"insufficient eligible molecules for {salt}: requested {n}, found {len(selected)}")
    return selected


def _dockability(smiles: str) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return {
            "docking_eligible": False,
            "docking_exclusion_reason": "rdkit_parse_failure",
            "docking_parent_smiles": "",
            "docking_removed_fragment_count": math.nan,
            "docking_molecular_weight": math.nan,
            "docking_heavy_atom_count": math.nan,
            "docking_rotatable_bond_count": math.nan,
        }
    fragment_count = len(Chem.GetMolFrags(molecule))
    parent = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(molecule)
    parent_smiles = Chem.MolToSmiles(parent, isomericSmiles=True)
    molecular_weight = float(Descriptors.MolWt(parent))
    heavy_atoms = int(parent.GetNumHeavyAtoms())
    rotatable_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(parent))
    unsupported = sorted(
        {atom.GetSymbol() for atom in parent.GetAtoms() if atom.GetAtomicNum() not in DOCKABLE_ATOMIC_NUMBERS}
    )
    reasons = []
    if molecular_weight > 750.0:
        reasons.append("molecular_weight_gt_750")
    if heavy_atoms > 55:
        reasons.append("heavy_atoms_gt_55")
    if rotatable_bonds > 15:
        reasons.append("rotatable_bonds_gt_15")
    if unsupported:
        reasons.append("unsupported_elements_" + "-".join(unsupported))
    return {
        "docking_eligible": not reasons,
        "docking_exclusion_reason": ";".join(reasons) if reasons else "eligible",
        "docking_parent_smiles": parent_smiles,
        "docking_removed_fragment_count": fragment_count - 1,
        "docking_molecular_weight": molecular_weight,
        "docking_heavy_atom_count": heavy_atoms,
        "docking_rotatable_bond_count": rotatable_bonds,
    }


def _add_dockability(frame: pd.DataFrame) -> pd.DataFrame:
    annotations = pd.DataFrame([_dockability(value) for value in frame.standardized_smiles])
    return pd.concat([frame.reset_index(drop=True), annotations], axis=1)


def _endpoint_fold_maps(oof: pd.DataFrame) -> pd.DataFrame:
    maps = []
    for endpoint in (10, 30, 50):
        part = oof.loc[oof.endpoint.eq(f"IC{endpoint}")].reset_index(drop=True)
        groups = part.scaffold_group_id.astype(str).to_numpy()
        folds = np.full(len(part), -1, dtype=int)
        for fold, (fit, evaluate) in enumerate(GroupKFold(5).split(part, groups=groups)):
            if set(groups[fit]) & set(groups[evaluate]):
                raise CampaignError(f"scaffold leakage while reconstructing IC{endpoint} folds")
            folds[evaluate] = fold
        if (folds < 0).any():
            raise CampaignError(f"incomplete reconstructed IC{endpoint} folds")
        maps.append(
            part[["standard_inchi_key"]].assign(**{f"endpoint_outer_fold_pic{endpoint}": folds})
        )
    result = maps[0]
    for mapping in maps[1:]:
        result = result.merge(mapping, on="standard_inchi_key", how="outer", validate="one_to_one")
    return result


def _direct_paired(repo: Path) -> pd.DataFrame:
    root = repo / "research/local_runs/herg_v10_1_expanded_platform/evidence"
    labels = pd.read_parquet(root / "empirical_ic10_ic30_ic50_labels.parquet")
    strict = labels.loc[labels.strict_curve_qc.astype(bool)].dropna(
        subset=["empirical_ic10_um", "empirical_ic30_um", "empirical_ic50_um"]
    )
    strict = strict.loc[
        strict.empirical_ic10_um.gt(0) & strict.empirical_ic30_um.gt(0) & strict.empirical_ic50_um.gt(0)
    ]
    direct = (
        strict.groupby("standard_inchi_key", as_index=False)
        .agg(
            standardized_smiles=("standardized_smiles", "first"),
            scaffold_group_id=("scaffold_group_id", "first"),
            empirical_ic10_um=("empirical_ic10_um", "median"),
            empirical_ic30_um=("empirical_ic30_um", "median"),
            empirical_ic50_um=("empirical_ic50_um", "median"),
        )
        .sort_values("standard_inchi_key")
        .reset_index(drop=True)
    )
    oof = pd.read_parquet(root / "empirical_ic10_ic30_ic50_oof.parquet")
    observed = oof.pivot_table(index="standard_inchi_key", columns="endpoint", values="observed_picx")
    predicted = oof.pivot_table(index="standard_inchi_key", columns="endpoint", values="predicted_picx")
    observed.columns = [f"observed_pic{str(value)[2:]}" for value in observed.columns]
    predicted.columns = [f"baseline_predicted_pic{str(value)[2:]}" for value in predicted.columns]
    direct = direct.merge(observed.join(predicted).reset_index(), on="standard_inchi_key", validate="one_to_one")
    direct = direct.merge(_endpoint_fold_maps(oof), on="standard_inchi_key", validate="one_to_one")
    groups = direct.scaffold_group_id.astype(str).to_numpy()
    folds = np.full(len(direct), -1, dtype=int)
    for fold, (_, evaluate) in enumerate(GroupKFold(5).split(direct, groups=groups)):
        folds[evaluate] = fold
    direct["outer_fold"] = folds
    direct["potency_stratum"] = pd.qcut(direct.observed_pic50, 3, labels=["low", "middle", "high"])
    return direct


def _select_pilot(repo: Path, output: Path, exact_n: int, direct_n: int) -> pd.DataFrame:
    reference = pd.read_parquet(repo / "research/local_runs/herg_v10_tiered_platform/reference/training_reference.parquet")
    v9 = pd.read_parquet(repo / "research/local_runs/herg_domain_mixture_campaign_v9/analysis/nested_oof_predictions.parquet")
    exact = reference.merge(
        v9[["structure_id", "outer_fold", "observed_pic50", "pred__honest_stack"]],
        on="structure_id",
        validate="one_to_one",
    )
    exact = _add_dockability(exact)
    exact_census = {
        "all": len(exact),
        "eligible": int(exact.docking_eligible.sum()),
        "excluded": int((~exact.docking_eligible).sum()),
        "exclusion_reasons": exact.loc[~exact.docking_eligible, "docking_exclusion_reason"]
        .value_counts()
        .to_dict(),
    }
    exact = exact.loc[exact.docking_eligible].reset_index(drop=True)
    exact["tier_index"] = _tier_index(exact.observed_pic50.to_numpy(float))
    exact["stratum"] = [CLASS_NAMES[value] for value in exact.tier_index]
    population_counts = exact.stratum.value_counts().to_dict()
    exact_rows = []
    for (fold, stratum), group in exact.groupby(["outer_fold", "stratum"], observed=True):
        chosen = _select_diverse(group, exact_n, "structure_id", f"exact-{fold}-{stratum}")
        chosen["population_stratum_count"] = int(population_counts[stratum])
        exact_rows.append(chosen)
    exact_selected = pd.concat(exact_rows, ignore_index=True)
    sample_counts = exact_selected.stratum.value_counts().to_dict()
    exact_selected["population_weight"] = exact_selected.stratum.map(
        {key: population_counts[key] / sample_counts[key] for key in population_counts}
    )
    exact_selected["cohort"] = "exact_ic50"
    exact_selected["ligand_id"] = "exact__" + exact_selected.structure_id.astype(str)
    exact_selected["baseline_prediction"] = exact_selected.pred__honest_stack.astype(float)
    exact_selected["observed_target"] = exact_selected.observed_pic50.astype(float)

    direct = _add_dockability(_direct_paired(repo))
    direct_census = {
        "all": len(direct),
        "eligible": int(direct.docking_eligible.sum()),
        "excluded": int((~direct.docking_eligible).sum()),
        "exclusion_reasons": direct.loc[~direct.docking_eligible, "docking_exclusion_reason"]
        .value_counts()
        .to_dict(),
    }
    direct = direct.loc[direct.docking_eligible].reset_index(drop=True)
    direct_rows = []
    for (fold, stratum), group in direct.groupby(["outer_fold", "potency_stratum"], observed=True):
        direct_rows.append(
            _select_diverse(group, direct_n, "standard_inchi_key", f"direct-{fold}-{stratum}")
        )
    direct_selected = pd.concat(direct_rows, ignore_index=True)
    direct_selected["cohort"] = "direct_curve"
    direct_selected["ligand_id"] = "direct__" + direct_selected.standard_inchi_key.astype(str)
    direct_selected["stratum"] = direct_selected.potency_stratum.astype(str)
    direct_selected["population_weight"] = 1.0
    direct_selected["observed_target"] = direct_selected.observed_pic50.astype(float)
    direct_selected["baseline_prediction"] = direct_selected.baseline_predicted_pic50.astype(float)

    columns = [
        "ligand_id",
        "cohort",
        "standardized_smiles",
        "scaffold_group_id",
        "outer_fold",
        "stratum",
        "population_weight",
        "observed_target",
        "baseline_prediction",
        "structure_id",
        "standard_inchi_key",
        "observed_pic10",
        "observed_pic30",
        "observed_pic50",
        "baseline_predicted_pic10",
        "baseline_predicted_pic30",
        "baseline_predicted_pic50",
        "endpoint_outer_fold_pic10",
        "endpoint_outer_fold_pic30",
        "endpoint_outer_fold_pic50",
        "docking_molecular_weight",
        "docking_heavy_atom_count",
        "docking_rotatable_bond_count",
        "docking_parent_smiles",
        "docking_removed_fragment_count",
    ]
    for frame in (exact_selected, direct_selected):
        for column in columns:
            if column not in frame:
                frame[column] = np.nan
    selected = pd.concat([exact_selected[columns], direct_selected[columns]], ignore_index=True)
    if selected.ligand_id.duplicated().any():
        raise CampaignError("pilot selection produced duplicate ligand IDs")
    _parquet(output / "pilot/pilot_selection.parquet", selected)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "selection_seed": SEED,
        "exact_per_fold_tier": exact_n,
        "direct_per_fold_tertile": direct_n,
        "selected_ligands": len(selected),
        "exact_ligands": int(selected.cohort.eq("exact_ic50").sum()),
        "direct_ligands": int(selected.cohort.eq("direct_curve").sum()),
        "selection_rule": "stable-hash scaffold-diverse sampling within fixed outer fold and potency stratum",
        "docking_applicability": {
            "criteria": "MW <=750 Da; heavy atoms <=55; rotatable bonds <=15; supported organic elements",
            "exact_census": exact_census,
            "direct_census": direct_census,
        },
        "truth_boundary": (
            "balanced pilot uses population weights for the docking-eligible exact-corpus strata; "
            "excluded structures are explicitly out of the Vina pilot applicability domain"
        ),
    }
    _self_hashed_json(output / "pilot/pilot_selection.json", payload, "report_sha256")
    return selected


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _prepare_ligand_microstates(
    row: Any,
    output: Path,
    tools: Toolchain,
    maximum: int,
) -> list[dict[str, Any]]:
    directory = output / "ligands" / _safe_component(str(row.ligand_id))
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "microstates.json"
    if manifest_path.is_file():
        payload = _read_self_hashed_json(manifest_path, "report_sha256")
        cache_matches = payload.get("docking_parent_smiles") == str(row.docking_parent_smiles)
        files_match = all(
            Path(item["pdbqt_path"]).is_file()
            and _sha(Path(item["pdbqt_path"])) == item["pdbqt_sha256"]
            for item in payload["microstates"]
        )
        if cache_matches and files_match:
            return list(payload["microstates"])
    all_sdf = directory / "molscrub_all.sdf"
    _run(
        [
            tools.molscrub,
            str(row.docking_parent_smiles),
            "-o",
            all_sdf,
            "--ph",
            "7.4",
            "--skip_tautomers",
            "--cpu",
            "1",
        ],
        directory / "molscrub.log",
    )
    molecules = [molecule for molecule in Chem.SDMolSupplier(str(all_sdf), removeHs=False) if molecule]
    unique: dict[str, Chem.Mol] = {}
    for molecule in molecules:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True)
        unique.setdefault(smiles, molecule)
    ranked = sorted(unique.items(), key=lambda item: (abs(Chem.GetFormalCharge(item[1])), item[0]))[:maximum]
    if not ranked:
        raise CampaignError(f"no pH 7.4 microstate generated for {row.ligand_id}")
    microstates = []
    for index, (smiles, molecule) in enumerate(ranked):
        sdf = directory / f"microstate_{index}.sdf"
        pdbqt = directory / f"microstate_{index}.pdbqt"
        writer = Chem.SDWriter(str(sdf))
        writer.write(molecule)
        writer.close()
        _run(
            [tools.meeko_ligand, "-i", sdf, "-o", pdbqt, "--add_index_map", "--charge_model", "gasteiger"],
            directory / f"meeko_{index}.log",
        )
        microstates.append(
            {
                "microstate_index": index,
                "microstate_smiles": smiles,
                "formal_charge": int(Chem.GetFormalCharge(molecule)),
                "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
                "sdf_path": str(sdf.resolve()),
                "pdbqt_path": str(pdbqt.resolve()),
                "pdbqt_sha256": _sha(pdbqt),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ligand_id": str(row.ligand_id),
        "input_smiles": str(row.standardized_smiles),
        "docking_parent_smiles": str(row.docking_parent_smiles),
        "removed_fragment_count": int(row.docking_removed_fragment_count),
        "ph": 7.4,
        "tautomer_enumeration": False,
        "generated_microstate_count": len(unique),
        "retained_microstate_count": len(microstates),
        "microstates": microstates,
        "truth_boundary": "rule-based acid/base enumeration; no microstate population free energies",
    }
    _self_hashed_json(manifest_path, payload, "report_sha256")
    return microstates


def _receptor_contact_atoms(prepared_pdb: Path) -> dict[int, np.ndarray]:
    structure = next(ChemistryPDB(prepared_pdb).models())
    result: dict[int, list[np.ndarray]] = {residue: [] for residue in CONTACT_RESIDUES}
    for chain in structure:
        if chain.id not in "ABCD":
            continue
        for residue in chain:
            if residue.id[1] not in result:
                continue
            result[residue.id[1]].extend(atom.coord for atom in residue if atom.element != "H")
    return {key: np.asarray(value, dtype=float) for key, value in result.items()}


class ChemistryPDB:
    """Tiny indirection that keeps Bio.PDB construction out of hot loops."""

    def __init__(self, path: Path) -> None:
        from Bio.PDB import PDBParser

        self.structure = PDBParser(QUIET=True).get_structure(path.stem, path)

    def models(self) -> Any:
        return self.structure.get_models()


def _contact_features(pose: np.ndarray, receptor_atoms: dict[int, np.ndarray]) -> dict[str, float]:
    features: dict[str, float] = {}
    global_minimum = math.inf
    for residue, atoms in receptor_atoms.items():
        if atoms.size == 0:
            minimum = math.nan
            contacts = 0
        else:
            distances = np.linalg.norm(pose[:, None, :] - atoms[None, :, :], axis=2)
            minimum = float(np.min(distances))
            contacts = int(np.sum(np.min(distances, axis=1) <= 4.5))
            global_minimum = min(global_minimum, minimum)
        features[f"contact_{residue}_minimum_A"] = minimum
        features[f"contact_{residue}_ligand_atom_count"] = float(contacts)
    features["minimum_protein_distance_A"] = float(global_minimum)
    features["severe_clash_indicator"] = float(global_minimum < 1.5)
    return features


def _dock_pilot(
    output: Path,
    tools: Toolchain,
    states: tuple[str, ...],
    exhaustiveness: int,
    modes: int,
    cpu: int,
    microstate_maximum: int,
) -> pd.DataFrame:
    redocking = _read_self_hashed_json(output / "redocking/redocking_report.json", "report_sha256")
    if not redocking["primary_gate_passed"]:
        raise CampaignError("native-redocking gate failed; pilot docking remains blocked")
    selected = pd.read_parquet(output / "pilot/pilot_selection.parquet")
    receptor_atoms = {
        state: _receptor_contact_atoms(output / f"receptors/{state}/{state}_prepared.pdb") for state in states
    }
    rows: list[dict[str, Any]] = []
    total = len(selected) * len(states)
    completed = 0
    for ligand in selected.itertuples(index=False):
        microstates = _prepare_ligand_microstates(ligand, output, tools, microstate_maximum)
        for state in states:
            receptor = output / f"receptors/{state}/{state}_prepared.pdbqt"
            config = output / f"receptors/{state}/{state}_prepared.box.txt"
            for microstate in microstates:
                task_id = f"{_safe_component(ligand.ligand_id)}__m{microstate['microstate_index']}__{state}"
                directory = output / "docking" / task_id
                directory.mkdir(parents=True, exist_ok=True)
                docked = directory / "poses.pdbqt"
                vina_log = directory / "vina.log"
                seed = SEED + int(hashlib.sha256(task_id.encode()).hexdigest()[:7], 16)
                if not docked.is_file():
                    _run(
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
                        vina_log,
                    )
                models = _pdbqt_models(docked)
                affinities = np.asarray([model[0] for model in models], dtype=float)
                best_index = int(np.argmin(affinities))
                feature = _contact_features(models[best_index][1], receptor_atoms[state])
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
                        "pose_count_within_1kcal": int(np.sum(affinities <= np.min(affinities) + 1.0)),
                        "ligand_efficiency_kcal_mol_per_heavy_atom": float(
                            np.min(affinities) / max(1, microstate["heavy_atom_count"])
                        ),
                        **feature,
                    }
                )
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(json.dumps({"stage": "dock", "completed": completed, "total": total}), flush=True)
            _parquet(output / "docking/docking_results.partial.parquet", pd.DataFrame(rows))
    frame = pd.DataFrame(rows)
    _parquet(output / "docking/docking_results.parquet", frame)
    partial = output / "docking/docking_results.partial.parquet"
    if partial.is_file():
        partial.unlink()
    return frame


def _aggregate_docking(frame: pd.DataFrame, states: tuple[str, ...]) -> pd.DataFrame:
    sort_columns = ["ligand_id", "pdb_id", "best_affinity_kcal_mol", "microstate_index"]
    best = frame.sort_values(sort_columns).drop_duplicates(["ligand_id", "pdb_id"], keep="first")
    rows = []
    for ligand_id, group in best.groupby("ligand_id"):
        by_state = group.set_index("pdb_id")
        if set(states) - set(by_state.index):
            raise CampaignError(f"incomplete receptor panel for {ligand_id}")
        scores = by_state.loc[list(states), "best_affinity_kcal_mol"].to_numpy(float)
        row: dict[str, Any] = {"ligand_id": ligand_id}
        for state in states:
            state_row = by_state.loc[state]
            row[f"dock__{state}__affinity"] = float(state_row.best_affinity_kcal_mol)
            row[f"dock__{state}__ligand_efficiency"] = float(
                state_row.ligand_efficiency_kcal_mol_per_heavy_atom
            )
            row[f"dock__{state}__pose_count_within_1kcal"] = float(state_row.pose_count_within_1kcal)
            row[f"dock__{state}__affinity_range"] = float(state_row.affinity_range_kcal_mol)
            row[f"dock__{state}__formal_charge"] = float(state_row.formal_charge)
            row[f"dock__{state}__minimum_protein_distance"] = float(state_row.minimum_protein_distance_A)
            for residue in CONTACT_RESIDUES:
                row[f"dock__{state}__contact_{residue}_distance"] = float(
                    state_row[f"contact_{residue}_minimum_A"]
                )
                row[f"dock__{state}__contact_{residue}_count"] = float(
                    state_row[f"contact_{residue}_ligand_atom_count"]
                )
        row["dock__ensemble_min_affinity"] = float(np.min(scores))
        row["dock__ensemble_mean_affinity"] = float(np.mean(scores))
        row["dock__ensemble_sd_affinity"] = float(np.std(scores))
        row["dock__ensemble_range_affinity"] = float(np.max(scores) - np.min(scores))
        if {"8ZYN", "8ZYP"}.issubset(states):
            row["dock__e4031_conditioned_minus_apo"] = float(
                by_state.loc["8ZYP", "best_affinity_kcal_mol"]
                - by_state.loc["8ZYN", "best_affinity_kcal_mol"]
            )
        if {"9CHP", "9CHQ"}.issubset(states):
            row["dock__lowK_minus_highK"] = float(
                by_state.loc["9CHQ", "best_affinity_kcal_mol"]
                - by_state.loc["9CHP", "best_affinity_kcal_mol"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _regression_metrics(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray | None = None) -> dict[str, Any]:
    error = y - prediction
    rho = spearmanr(y, prediction).statistic
    if weights is None:
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
        within = float(np.mean(np.abs(error) <= 0.5))
    else:
        weights = weights / np.sum(weights)
        mae = float(np.sum(weights * np.abs(error)))
        rmse = float(np.sqrt(np.sum(weights * error**2)))
        within = float(np.sum(weights * (np.abs(error) <= 0.5)))
    return {
        "n": len(y),
        "mae": mae,
        "rmse": rmse,
        "spearman": float(rho) if np.isfinite(rho) else None,
        "within_0p5": within,
    }


def _classification_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predicted = np.argmax(probabilities, axis=1)
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
        "potent_predicted_safe_rate": float(np.mean(predicted[y == 2] == 0)) if np.any(y == 2) else None,
    }


def _assign_class_probabilities(
    target: np.ndarray,
    evaluation_mask: np.ndarray,
    classes: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    rows = np.flatnonzero(evaluation_mask)
    columns = np.asarray(classes, dtype=int)
    if probabilities.shape != (len(rows), len(columns)):
        raise CampaignError("classifier probability shape does not match evaluation rows/classes")
    target[np.ix_(rows, columns)] = probabilities


def _ridge() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )


def _crossfit_residual(
    frame: pd.DataFrame,
    features: list[str],
    observed: str,
    baseline: str,
    fold_column: str = "outer_fold",
) -> np.ndarray:
    prediction = np.full(len(frame), np.nan)
    folds = pd.to_numeric(frame[fold_column], errors="raise").astype(int)
    for fold in sorted(folds.unique()):
        fit = folds.ne(fold).to_numpy()
        evaluate = ~fit
        model = _ridge()
        residual = frame.loc[fit, observed].to_numpy(float) - frame.loc[fit, baseline].to_numpy(float)
        model.fit(frame.loc[fit, features], residual)
        prediction[evaluate] = frame.loc[evaluate, baseline].to_numpy(float) + model.predict(
            frame.loc[evaluate, features]
        )
    if not np.isfinite(prediction).all():
        raise CampaignError("incomplete cross-fitted regression predictions")
    return prediction


def _shuffled_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    shuffled = frame.copy()
    order = _stable_order(shuffled.ligand_id.astype(str), "shuffled-negative-control")
    shuffled.loc[:, features] = shuffled.iloc[order][features].to_numpy()
    return shuffled


def _bootstrap_delta(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    observed: np.ndarray,
    replicates: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    groups = frame.scaffold_group_id.astype(str).to_numpy()
    codes, unique = pd.factorize(groups, sort=True)
    counts = np.bincount(codes, minlength=len(unique)).astype(float)
    error_difference = np.abs(observed - candidate) - np.abs(observed - baseline)
    group_error_difference = np.bincount(codes, weights=error_difference, minlength=len(unique))
    deltas = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 256):
        stop = min(replicates, start + 256)
        sampled = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        deltas[start:stop] = np.sum(group_error_difference[sampled], axis=1) / np.sum(
            counts[sampled], axis=1
        )
    return {
        "delta_mae_candidate_minus_baseline": float(
            np.mean(np.abs(observed - candidate)) - np.mean(np.abs(observed - baseline))
        ),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "probability_candidate_better": float(np.mean(deltas < 0)),
        "replicates": replicates,
        "resampling_unit": "scaffold_group_id",
    }


def _project_nonincreasing(values: np.ndarray) -> np.ndarray:
    blocks: list[tuple[float, int]] = []
    for value in -np.asarray(values, dtype=float):
        blocks.append((float(value), 1))
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right, right_n = blocks.pop()
            left, left_n = blocks.pop()
            total = left_n + right_n
            blocks.append(((left * left_n + right * right_n) / total, total))
    return -np.concatenate([np.repeat(value, count) for value, count in blocks])


def _analyze(output: Path, states: tuple[str, ...], bootstrap: int) -> dict[str, Any]:
    selected = pd.read_parquet(output / "pilot/pilot_selection.parquet")
    docking = pd.read_parquet(output / "docking/docking_results.parquet")
    features = _aggregate_docking(docking, states)
    analysis = selected.merge(features, on="ligand_id", validate="one_to_one")
    dock_columns = [column for column in analysis if column.startswith("dock__")]
    score_columns = [
        column
        for column in dock_columns
        if any(token in column for token in ("affinity", "ligand_efficiency", "pose_count"))
    ]
    if len(analysis) != len(selected) or not dock_columns:
        raise CampaignError("docking aggregation did not cover the full pilot")
    prediction_rows = []
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "states": list(states),
        "feature_count": len(dock_columns),
        "score_feature_count": len(score_columns),
        "exact_ic50": {},
        "direct_endpoints": {},
    }

    exact = analysis.loc[analysis.cohort.eq("exact_ic50")].reset_index(drop=True)
    observed = exact.observed_target.to_numpy(float)
    baseline = exact.baseline_prediction.to_numpy(float)
    weights = exact.population_weight.to_numpy(float)
    for surface, columns in (("score_only", score_columns), ("full_interaction", dock_columns)):
        candidate = _crossfit_residual(exact, columns, "observed_target", "baseline_prediction")
        shuffled_frame = _shuffled_features(exact, columns)
        shuffled = _crossfit_residual(shuffled_frame, columns, "observed_target", "baseline_prediction")
        report["exact_ic50"][surface] = {
            "unweighted": _regression_metrics(observed, candidate),
            "population_weighted": _regression_metrics(observed, candidate, weights),
            "shuffled_negative_control": _regression_metrics(observed, shuffled),
            "bootstrap": _bootstrap_delta(exact, baseline, candidate, observed, bootstrap),
        }
        for index, row in exact.iterrows():
            prediction_rows.append(
                {
                    "ligand_id": row.ligand_id,
                    "cohort": "exact_ic50",
                    "endpoint": "IC50",
                    "surface": surface,
                    "outer_fold": int(row.outer_fold),
                    "observed": float(observed[index]),
                    "baseline": float(baseline[index]),
                    "receptor_augmented": float(candidate[index]),
                    "shuffled_control": float(shuffled[index]),
                }
            )
    report["exact_ic50"]["baseline"] = {
        "unweighted": _regression_metrics(observed, baseline),
        "population_weighted": _regression_metrics(observed, baseline, weights),
    }

    # A leakage-safe classification comparison uses the out-of-fold V9 pIC50
    # prediction as the baseline covariate; historical V10 calibration fitted
    # all OOF labels and is intentionally not used as a V13 meta-feature.
    y_class = _tier_index(observed)
    baseline_class = _tier_index(baseline)
    baseline_prob = np.eye(3)[baseline_class]
    class_prob = np.full((len(exact), 3), np.nan)
    class_features = ["baseline_prediction", *dock_columns]
    for fold in sorted(exact.outer_fold.unique()):
        fit = exact.outer_fold.ne(fold).to_numpy()
        evaluate = ~fit
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(C=0.1, class_weight="balanced", max_iter=2_000, random_state=SEED),
                ),
            ]
        )
        model.fit(exact.loc[fit, class_features], y_class[fit])
        probabilities = model.predict_proba(exact.loc[evaluate, class_features])
        model_classes = model.named_steps["model"].classes_.astype(int)
        _assign_class_probabilities(class_prob, evaluate, model_classes, probabilities)
    if not np.isfinite(class_prob).all():
        raise CampaignError("incomplete V13 classification probabilities")
    report["classification"] = {
        "baseline_thresholded_v9_oof": _classification_metrics(y_class, baseline_prob),
        "v13_receptor_augmented": _classification_metrics(y_class, class_prob),
        "claim_boundary": (
            "balanced pilot classification; not the full-corpus V10 router and not external validation"
        ),
    }

    direct = analysis.loc[analysis.cohort.eq("direct_curve")].reset_index(drop=True)
    endpoint_predictions: dict[str, np.ndarray] = {}
    for endpoint in (10, 30, 50):
        observed_column = f"observed_pic{endpoint}"
        baseline_column = f"baseline_predicted_pic{endpoint}"
        direct["observed_target_endpoint"] = direct[observed_column]
        direct["baseline_prediction_endpoint"] = direct[baseline_column]
        observed_endpoint = direct[observed_column].to_numpy(float)
        baseline_endpoint = direct[baseline_column].to_numpy(float)
        candidate = _crossfit_residual(
            direct,
            dock_columns,
            "observed_target_endpoint",
            "baseline_prediction_endpoint",
            f"endpoint_outer_fold_pic{endpoint}",
        )
        endpoint_predictions[str(endpoint)] = candidate
        report["direct_endpoints"][f"IC{endpoint}"] = {
            "baseline": _regression_metrics(observed_endpoint, baseline_endpoint),
            "receptor_augmented_raw": _regression_metrics(observed_endpoint, candidate),
            "bootstrap": _bootstrap_delta(
                direct, baseline_endpoint, candidate, observed_endpoint, bootstrap
            ),
        }
        for index, row in direct.iterrows():
            prediction_rows.append(
                {
                    "ligand_id": row.ligand_id,
                    "cohort": "direct_curve",
                    "endpoint": f"IC{endpoint}",
                    "surface": "full_interaction",
                    "outer_fold": int(row[f"endpoint_outer_fold_pic{endpoint}"]),
                    "observed": float(observed_endpoint[index]),
                    "baseline": float(baseline_endpoint[index]),
                    "receptor_augmented": float(candidate[index]),
                    "shuffled_control": math.nan,
                }
            )
    coherent = np.vstack(
        [
            _project_nonincreasing(values)
            for values in np.column_stack(
                [endpoint_predictions["10"], endpoint_predictions["30"], endpoint_predictions["50"]]
            )
        ]
    )
    for column_index, endpoint in enumerate((10, 30, 50)):
        observed_endpoint = direct[f"observed_pic{endpoint}"].to_numpy(float)
        report["direct_endpoints"][f"IC{endpoint}"]["receptor_augmented_coherent"] = _regression_metrics(
            observed_endpoint, coherent[:, column_index]
        )
    report["direct_endpoints"]["coherence"] = {
        "raw_order_violations": int(
            np.sum(
                (endpoint_predictions["10"] < endpoint_predictions["30"])
                | (endpoint_predictions["30"] < endpoint_predictions["50"])
            )
        ),
        "coherent_order_violations": int(np.sum((coherent[:, 0] < coherent[:, 1]) | (coherent[:, 1] < coherent[:, 2]))),
    }

    _parquet(output / "analysis/receptor_features.parquet", features)
    _parquet(output / "analysis/pilot_analysis_matrix.parquet", analysis)
    _parquet(output / "analysis/crossfit_predictions.parquet", pd.DataFrame(prediction_rows))
    report["scientific_scope"] = {
        "repository_validation_labels_opened": False,
        "repository_test_labels_opened": False,
        "receptor_aware": True,
        "rigid_receptor": True,
        "native_redocking_controlled": True,
        "external_or_prospective_validation": False,
        "docking_scores_are_binding_free_energies": False,
        "pilot_selection_is_full_corpus_evaluation": False,
    }
    return _self_hashed_json(output / "analysis/analysis_report.json", report, "report_sha256")


def _manifest(repo: Path, output: Path, tools: Toolchain, states: tuple[str, ...]) -> dict[str, Any]:
    inputs = [
        repo / "pipeline/scripts/run_local_herg_receptor_ensemble_campaign_v13.py",
        repo / "research/local_runs/herg_domain_mixture_campaign_v9/analysis/nested_oof_predictions.parquet",
        repo / "research/local_runs/herg_v10_tiered_platform/reference/training_reference.parquet",
        repo / "research/local_runs/herg_v10_1_expanded_platform/evidence/empirical_ic10_ic30_ic50_labels.parquet",
        repo / "research/local_runs/herg_v10_1_expanded_platform/evidence/empirical_ic10_ic30_ic50_oof.parquet",
        *_coordinate_paths(repo).values(),
    ]
    artifacts = [
        output / "receptors/receptor_preparation.json",
        output / "redocking/redocking_report.json",
        output / "pilot/pilot_selection.json",
        output / "docking/docking_results.parquet",
        output / "analysis/receptor_features.parquet",
        output / "analysis/crossfit_predictions.parquet",
        output / "analysis/analysis_report.json",
    ]
    for path in [*inputs, *artifacts]:
        if not path.is_file():
            raise CampaignError(f"manifest path missing: {path}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "status": "complete_receptor_ensemble_pilot",
        "states": list(states),
        "toolchain": {**{key: str(value) for key, value in asdict(tools).items()}, "vina_sha256": _sha(tools.vina)},
        "inputs": [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)} for path in inputs],
        "artifacts": [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)} for path in artifacts],
        "claim_boundary": (
            "internal scaffold-fold receptor-aware pilot; Vina observables are hypotheses, not experimental affinities"
        ),
    }
    return _self_hashed_json(output / "manifest.json", payload, "manifest_sha256")


def _validate(output: Path) -> dict[str, Any]:
    preparation = _read_self_hashed_json(output / "receptors/receptor_preparation.json", "report_sha256")
    redocking = _read_self_hashed_json(output / "redocking/redocking_report.json", "report_sha256")
    selection = _read_self_hashed_json(output / "pilot/pilot_selection.json", "report_sha256")
    analysis = _read_self_hashed_json(output / "analysis/analysis_report.json", "report_sha256")
    manifest = _read_self_hashed_json(output / "manifest.json", "manifest_sha256")
    for collection in (manifest["inputs"], manifest["artifacts"]):
        for row in collection:
            path = Path(row["path"])
            if not path.is_file() or path.stat().st_size != row["bytes"] or _sha(path) != row["sha256"]:
                raise CampaignError(f"bound artifact changed: {path}")
    docking = pd.read_parquet(output / "docking/docking_results.parquet")
    selected = pd.read_parquet(output / "pilot/pilot_selection.parquet")
    features = pd.read_parquet(output / "analysis/receptor_features.parquet")
    if features.ligand_id.nunique() != len(selected):
        raise CampaignError("receptor feature coverage is incomplete")
    if docking.ligand_id.nunique() != len(selected):
        raise CampaignError("docking result coverage is incomplete")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "validated_utc": _utc(),
        "status": "passed",
        "common_residue_count_per_chain": preparation["common_residue_count_per_chain"],
        "redocking_primary_gate_passed": redocking["primary_gate_passed"],
        "selected_ligands": selection["selected_ligands"],
        "docking_rows": len(docking),
        "receptor_feature_rows": len(features),
        "repository_validation_labels_opened": analysis["scientific_scope"]["repository_validation_labels_opened"],
        "repository_test_labels_opened": analysis["scientific_scope"]["repository_test_labels_opened"],
        "manifest_sha256": manifest["manifest_sha256"],
    }
    return _self_hashed_json(output / "validation.json", payload, "validation_sha256")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "redock", "select", "dock", "analyze", "run", "validate"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/local_runs/herg_receptor_ensemble_campaign_v13"),
    )
    parser.add_argument("--vina", type=Path)
    parser.add_argument("--states", default=",".join(CORE_IDS))
    parser.add_argument("--cpu", type=int, default=6)
    parser.add_argument("--box-size", type=float, nargs=3, default=(28.0, 28.0, 30.0))
    parser.add_argument("--redock-exhaustiveness", type=int, default=32)
    parser.add_argument("--redock-seeds", type=int, default=3)
    parser.add_argument("--exact-per-fold-tier", type=int, default=12)
    parser.add_argument("--direct-per-fold-tertile", type=int, default=12)
    parser.add_argument("--pilot-exhaustiveness", type=int, default=8)
    parser.add_argument("--pilot-modes", type=int, default=9)
    parser.add_argument("--microstate-maximum", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _main(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    output = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    output = output.resolve()
    states = tuple(value.strip() for value in args.states.split(",") if value.strip())
    if not states or set(states) - set(RECEPTOR_IDS):
        raise CampaignError(f"invalid receptor states: {states}")
    if args.cpu < 1 or args.microstate_maximum < 1:
        raise CampaignError("CPU and microstate limits must be positive")
    tools = _resolve_toolchain(repo, args.vina.resolve() if args.vina else None)
    if "1.2." not in _tool_version(tools.vina, "--version"):
        raise CampaignError("V13 requires AutoDock Vina 1.2.x")
    if args.smoke:
        args.redock_exhaustiveness = min(args.redock_exhaustiveness, 4)
        args.redock_seeds = 1
        args.exact_per_fold_tier = 1
        args.direct_per_fold_tertile = 1
        args.pilot_exhaustiveness = min(args.pilot_exhaustiveness, 2)
        args.pilot_modes = min(args.pilot_modes, 3)
        args.microstate_maximum = 1
        args.bootstrap_replicates = min(args.bootstrap_replicates, 100)
    output.mkdir(parents=True, exist_ok=True)

    with _Lock(output / ".campaign.lock"):
        if args.command in ("prepare", "run"):
            result = _prepare_receptors(repo, output, tools, tuple(args.box_size))
            if args.command == "prepare":
                return result
        if args.command in ("redock", "run"):
            result = _redock(
                repo,
                output,
                tools,
                args.redock_exhaustiveness,
                args.redock_seeds,
                args.cpu,
            )
            if args.command == "redock":
                return result
        if args.command in ("select", "run"):
            selected = _select_pilot(
                repo,
                output,
                args.exact_per_fold_tier,
                args.direct_per_fold_tertile,
            )
            if args.command == "select":
                return {"status": "selected", "ligands": len(selected)}
        if args.command in ("dock", "run"):
            docking = _dock_pilot(
                output,
                tools,
                states,
                args.pilot_exhaustiveness,
                args.pilot_modes,
                args.cpu,
                args.microstate_maximum,
            )
            if args.command == "dock":
                return {"status": "docked", "rows": len(docking)}
        if args.command in ("analyze", "run"):
            result = _analyze(output, states, args.bootstrap_replicates)
            if args.command == "analyze":
                return result
        if args.command == "run":
            manifest = _manifest(repo, output, tools, states)
            validation = _validate(output)
            summary = {
                "schema_version": SCHEMA_VERSION,
                "finished_utc": _utc(),
                "status": "complete",
                "selected_ligands": validation["selected_ligands"],
                "docking_rows": validation["docking_rows"],
                "redocking_primary_gate_passed": validation["redocking_primary_gate_passed"],
                "manifest_sha256": manifest["manifest_sha256"],
                "validation_sha256": validation["validation_sha256"],
            }
            return _self_hashed_json(output / "final_summary.json", summary, "summary_sha256")
        if args.command == "validate":
            return _validate(output)
    raise CampaignError(f"unhandled command: {args.command}")


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _main(args)
    except CampaignError as exc:
        print(f"V13 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
