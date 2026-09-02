#!/usr/bin/env python3
"""Build and benchmark the V14 cross-campaign hERG receptor fusion surface.

V14 is an exploratory, leakage-controlled re-analysis of six completed V13
campaigns.  It does not relabel those campaigns as prospective confirmation.
The workflow harmonizes the common frozen ligand outputs and 8ZYO AutoDock
Vina poses, extracts pose-ensemble interaction features from all returned
poses, and evaluates ligand-only and ligand+receptor models with nested
leave-one-campaign-out validation.

AutoDock Vina scores and pose contacts are scoring-function observables.  They
are not binding free energies, experimental affinities, or evidence for a
unique physiological binding pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from menin_discovery.features import scaffold_key  # noqa: E402

SCHEMA_VERSION = "platform-local-herg-cross-campaign-receptor-fusion-v14/1.0"
SEED = 20260830
STATE = "8ZYO"
CLASS_NAMES = ("Safe", "Moderate", "Potent")
SAFE_PIC50 = 6.0 - math.log10(30.0)
POTENT_PIC50 = 6.0
CONTACT_RESIDUES = (623, 624, 649, 652, 656)
POCKET_RESIDUES = (621, 622, 623, 624, 645, 648, 649, 652, 656)
ROUTER_COLUMNS = tuple(
    f"lgbm_rdkit2d_morgan__probability_{name.lower()}" for name in CLASS_NAMES
)
FROZEN_RECEPTOR_COLUMNS = tuple(
    f"primary_frozen_f649__probability_{name.lower()}" for name in CLASS_NAMES
)
DEFAULT_OUTPUT = Path("research/local_runs/herg_cross_campaign_receptor_fusion_v14")
DEFAULT_ROUTER = Path(
    "research/local_runs/herg_v10_1_expanded_platform/evidence/exact_router_oof.parquet"
)
DEFAULT_RECEPTOR = Path(
    "research/local_runs/herg_receptor_ensemble_campaign_v13/receptors/8ZYO/8ZYO_prepared.pdb"
)


class CampaignError(RuntimeError):
    """Raised when an input or scientific-contract invariant is violated."""


@dataclass(frozen=True)
class CampaignSpec:
    name: str
    root: Path
    selection: Path
    kind: str


CAMPAIGNS = (
    CampaignSpec(
        "v13_3_internal_confirmation",
        Path("research/local_runs/herg_receptor_classification_confirmation_v13_3"),
        Path("selection/confirmation_panel.parquet"),
        "internal",
    ),
    CampaignSpec(
        "v13_4_internal_replication",
        Path("research/local_runs/herg_receptor_classification_replication_v13_4"),
        Path("selection/replication_panel.parquet"),
        "internal",
    ),
    CampaignSpec(
        "v13_5_internal_safety",
        Path("research/local_runs/herg_receptor_safety_gated_validation_v13_5"),
        Path("selection/validation_panel.parquet"),
        "internal",
    ),
    CampaignSpec(
        "v13_6_internal_hybrid",
        Path("research/local_runs/herg_receptor_hybrid_validation_v13_6"),
        Path("selection/hybrid_validation_panel.parquet"),
        "internal",
    ),
    CampaignSpec(
        "v13_7_external_2025",
        Path("research/local_runs/herg_external_validation_v13_7"),
        Path("selection/receptor_external_panel.parquet"),
        "external_2025",
    ),
    CampaignSpec(
        "v13_8_external_2026",
        Path("research/local_runs/herg_temporal_confirmation_v13_8"),
        Path("selection/receptor_temporal_panel.parquet"),
        "external_2026",
    ),
)


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


def _write_json(path: Path, payload: dict[str, Any], hash_field: str) -> dict[str, Any]:
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


def _resolve(repo: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _tier(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values < SAFE_PIC50, 0, np.where(values > POTENT_PIC50, 2, 1)).astype(int)


def _normalize_probabilities(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    values = frame[list(columns)].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise CampaignError(f"{label} probabilities are invalid")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise CampaignError(f"{label} probabilities are not normalized")


def _connectivity_key(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise CampaignError(f"RDKit could not parse harmonized SMILES: {smiles}")
    return Chem.MolToInchiKey(molecule).split("-")[0]


def _raw_scaffold_key(smiles: str) -> str:
    key, _method = scaffold_key(str(smiles))
    return str(key)


def _aggregate_best_pose(docking: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ligand_id",
        "pdb_id",
        "microstate_index",
        "best_affinity_kcal_mol",
        "median_affinity_kcal_mol",
        "affinity_range_kcal_mol",
        "pose_count",
        "pose_count_within_1kcal",
        "ligand_efficiency_kcal_mol_per_heavy_atom",
        "formal_charge",
        "heavy_atom_count",
        "minimum_protein_distance_A",
        "severe_clash_indicator",
    }
    for residue in CONTACT_RESIDUES:
        required.update(
            {
                f"contact_{residue}_minimum_A",
                f"contact_{residue}_ligand_atom_count",
            }
        )
    missing = required - set(docking)
    if missing:
        raise CampaignError(f"docking artifact lacks columns: {sorted(missing)}")
    state = docking.loc[docking.pdb_id.astype(str).eq(STATE)].copy()
    state = state.sort_values(
        ["ligand_id", "best_affinity_kcal_mol", "microstate_index"]
    ).drop_duplicates("ligand_id", keep="first")
    rows: dict[str, Any] = {
        "ligand_id": state.ligand_id.astype(str).to_numpy(),
        "dock_affinity": state.best_affinity_kcal_mol.to_numpy(float),
        "dock_median_affinity": state.median_affinity_kcal_mol.to_numpy(float),
        "dock_affinity_range": state.affinity_range_kcal_mol.to_numpy(float),
        "dock_pose_count": state.pose_count.to_numpy(float),
        "dock_pose_count_within_1kcal": state.pose_count_within_1kcal.to_numpy(float),
        "dock_ligand_efficiency": state.ligand_efficiency_kcal_mol_per_heavy_atom.to_numpy(float),
        "dock_formal_charge": state.formal_charge.to_numpy(float),
        "dock_heavy_atom_count": state.heavy_atom_count.to_numpy(float),
        "dock_minimum_protein_distance": state.minimum_protein_distance_A.to_numpy(float),
        "dock_severe_clash": state.severe_clash_indicator.astype(float).to_numpy(),
    }
    for residue in CONTACT_RESIDUES:
        rows[f"dock_contact_{residue}_distance"] = state[
            f"contact_{residue}_minimum_A"
        ].to_numpy(float)
        rows[f"dock_contact_{residue}_count"] = state[
            f"contact_{residue}_ligand_atom_count"
        ].to_numpy(float)
    return pd.DataFrame(rows)


def _load_internal(
    repo: Path,
    spec: CampaignSpec,
    router: pd.DataFrame,
) -> pd.DataFrame:
    root = _resolve(repo, spec.root)
    selection = pd.read_parquet(root / spec.selection)
    docking = _aggregate_best_pose(pd.read_parquet(root / "docking/docking_results.parquet"))
    predictions = pd.read_parquet(root / "predictions/predictions_before_score.parquet")
    required_selection = {
        "ligand_id",
        "structure_id",
        "standardized_smiles",
        "scaffold_group_id",
        "baseline_prediction",
        "observed_target",
    }
    if required_selection - set(selection):
        raise CampaignError(f"{spec.name} selection lacks required columns")
    frame = selection.merge(docking, on="ligand_id", validate="one_to_one")
    frame = frame.merge(
        router[["structure_id", *ROUTER_COLUMNS]],
        on="structure_id",
        validate="one_to_one",
    )
    frame = frame.merge(
        predictions[["ligand_id", *FROZEN_RECEPTOR_COLUMNS]],
        on="ligand_id",
        validate="one_to_one",
    )
    frame = frame.rename(
        columns={"baseline_prediction": "baseline_pic50", "observed_target": "target_pic50"}
    )
    frame["external_structure_id"] = ""
    frame["nearest_v9_morgan_tanimoto"] = 1.0
    frame["campaign"] = spec.name
    frame["campaign_kind"] = spec.kind
    return frame


def _load_external(repo: Path, spec: CampaignSpec) -> pd.DataFrame:
    root = _resolve(repo, spec.root)
    selection = pd.read_parquet(root / spec.selection)
    docking = _aggregate_best_pose(pd.read_parquet(root / "docking/docking_results.parquet"))
    predictions = pd.read_parquet(root / "predictions/receptor_predictions_before_score.parquet")
    labels = pd.read_parquet(root / "sealed/labels.parquet")
    if "set_name" in labels:
        quantitative = {
            "ev2_exact_quantitative",
            "prior_study_external_exact_quantitative",
        }
        labels = labels.loc[labels.set_name.astype(str).isin(quantitative)]
    labels = labels.groupby("external_structure_id", as_index=False).agg(
        target_pic50=("true_pic50_m", "median")
    )
    frame = selection.merge(
        predictions[
            [
                "external_structure_id",
                "baseline_prediction",
                *ROUTER_COLUMNS,
                *FROZEN_RECEPTOR_COLUMNS,
            ]
        ],
        on="external_structure_id",
        validate="one_to_one",
    )
    frame = frame.merge(labels, on="external_structure_id", validate="one_to_one")
    frame = frame.merge(docking, on="ligand_id", validate="one_to_one")
    frame = frame.rename(columns={"baseline_prediction": "baseline_pic50"})
    frame["campaign"] = spec.name
    frame["campaign_kind"] = spec.kind
    return frame


def _harmonize(repo: Path, router_path: Path) -> pd.DataFrame:
    router = pd.read_parquet(router_path)
    required_router = {"structure_id", *ROUTER_COLUMNS}
    if required_router - set(router):
        raise CampaignError("V10.1 exact-router OOF artifact lacks required columns")
    if router.structure_id.astype(str).duplicated().any():
        raise CampaignError("V10.1 exact-router OOF structure IDs are not unique")
    frames = []
    for spec in CAMPAIGNS:
        frame = (
            _load_internal(repo, spec, router)
            if spec.kind == "internal"
            else _load_external(repo, spec)
        )
        if frame.ligand_id.astype(str).duplicated().any():
            raise CampaignError(f"{spec.name} ligand IDs are not unique")
        _normalize_probabilities(frame, ROUTER_COLUMNS, f"{spec.name} router")
        _normalize_probabilities(frame, FROZEN_RECEPTOR_COLUMNS, f"{spec.name} receptor")
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["target_class"] = _tier(frame.target_pic50.to_numpy(float))
    frame["target_label"] = [CLASS_NAMES[value] for value in frame.target_class]
    frame["cross_campaign_connectivity_key"] = [
        _connectivity_key(value) for value in frame.standardized_smiles
    ]
    frame["cross_campaign_scaffold_key"] = [
        _raw_scaffold_key(value) for value in frame.standardized_smiles
    ]
    frame["sample_id"] = frame.campaign.astype(str) + "::" + frame.ligand_id.astype(str)
    frame["router_prediction"] = np.argmax(frame[list(ROUTER_COLUMNS)].to_numpy(float), axis=1)
    frame["frozen_receptor_prediction"] = np.argmax(
        frame[list(FROZEN_RECEPTOR_COLUMNS)].to_numpy(float), axis=1
    )
    if frame.sample_id.duplicated().any():
        raise CampaignError("harmonized sample IDs are not unique")
    expected = {
        "v13_3_internal_confirmation": 90,
        "v13_4_internal_replication": 270,
        "v13_5_internal_safety": 270,
        "v13_6_internal_hybrid": 270,
        "v13_7_external_2025": 110,
        "v13_8_external_2026": 214,
    }
    observed = frame.campaign.value_counts().to_dict()
    if observed != expected:
        raise CampaignError(f"unexpected harmonized campaign census: {observed}")
    return frame.sort_values(["campaign", "sample_id"]).reset_index(drop=True)


def _overlap_audit(frame: pd.DataFrame) -> pd.DataFrame:
    campaigns = sorted(frame.campaign.unique())
    rows = []
    for left_index, left_name in enumerate(campaigns):
        left = frame.loc[frame.campaign.eq(left_name)]
        for right_name in campaigns[left_index + 1 :]:
            right = frame.loc[frame.campaign.eq(right_name)]
            exact = set(left.cross_campaign_connectivity_key) & set(
                right.cross_campaign_connectivity_key
            )
            scaffolds = set(left.cross_campaign_scaffold_key) & set(
                right.cross_campaign_scaffold_key
            )
            rows.append(
                {
                    "campaign_left": left_name,
                    "campaign_right": right_name,
                    "exact_connectivity_overlap": len(exact),
                    "scaffold_overlap": len(scaffolds),
                }
            )
    return pd.DataFrame(rows)


def _campaign_census(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for campaign, group in frame.groupby("campaign", sort=True):
        class_counts = np.bincount(group.target_class.to_numpy(int), minlength=3)
        rows.append(
            {
                "campaign": campaign,
                "campaign_kind": group.campaign_kind.iloc[0],
                "n": len(group),
                "unique_connectivity_keys": group.cross_campaign_connectivity_key.nunique(),
                "unique_scaffolds": group.cross_campaign_scaffold_key.nunique(),
                **{
                    f"observed_{name.lower()}_n": int(class_counts[index])
                    for index, name in enumerate(CLASS_NAMES)
                },
                "median_target_pic50": float(group.target_pic50.median()),
                "median_nearest_v9_morgan_tanimoto": float(
                    group.nearest_v9_morgan_tanimoto.median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _parse_receptor_atoms(path: Path) -> dict[str, Any]:
    by_residue: dict[int, list[list[float]]] = {residue: [] for residue in POCKET_RESIDUES}
    by_residue_element: dict[int, list[str]] = {residue: [] for residue in POCKET_RESIDUES}
    ring_atoms: dict[tuple[int, str], list[list[float]]] = {}
    aromatic_ring_names = {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}
    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                residue = int(line[22:26])
                chain = line[21:22]
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip().upper()
                element = (line[76:78].strip() or line[12:16].strip()[0]).upper()
                coordinate = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except (ValueError, IndexError) as exc:
                raise CampaignError(f"could not parse receptor atom line in {path}") from exc
            if residue in by_residue and element != "H":
                by_residue[residue].append(coordinate)
                by_residue_element[residue].append(element)
                if (
                    residue in {652, 656}
                    and residue_name in {"PHE", "TYR"}
                    and atom_name in aromatic_ring_names
                ):
                    ring_atoms.setdefault((residue, chain), []).append(coordinate)
    if any(not atoms for atoms in by_residue.values()):
        missing = [residue for residue, atoms in by_residue.items() if not atoms]
        raise CampaignError(f"prepared receptor lacks pocket residues: {missing}")
    arrays = {residue: np.asarray(atoms, dtype=float) for residue, atoms in by_residue.items()}
    all_atoms = np.concatenate(list(arrays.values()), axis=0)
    center_atoms = np.concatenate([arrays[residue] for residue in CONTACT_RESIDUES], axis=0)
    ring_centroids: dict[int, np.ndarray] = {}
    for residue in (652, 656):
        centroids = [
            np.mean(np.asarray(atoms, dtype=float), axis=0)
            for (number, _chain), atoms in ring_atoms.items()
            if number == residue and len(atoms) >= 6
        ]
        if not centroids:
            raise CampaignError(f"prepared receptor lacks aromatic ring atoms for residue {residue}")
        ring_centroids[residue] = np.asarray(centroids, dtype=float)
    return {
        "by_residue": arrays,
        "by_residue_element": by_residue_element,
        "all_atoms": all_atoms,
        "center": np.mean(center_atoms, axis=0),
        "ring_centroids": ring_centroids,
    }


def _parse_pose_file(path: Path) -> list[dict[str, Any]]:
    poses: list[dict[str, Any]] = []
    score: float | None = None
    coordinates: list[list[float]] = []
    atom_types: list[str] = []
    charges: list[float] = []
    serials: list[int] = []
    index_map: dict[int, int] = {}

    def finish() -> None:
        nonlocal score, coordinates, atom_types, charges, serials
        if score is None and not coordinates:
            return
        if score is None or not coordinates:
            raise CampaignError(f"incomplete Vina pose in {path}")
        poses.append(
            {
                "score": score,
                "coordinates": np.asarray(coordinates, dtype=float),
                "atom_types": tuple(atom_types),
                "charges": np.asarray(charges, dtype=float),
                "serials": tuple(serials),
            }
        )
        score = None
        coordinates = []
        atom_types = []
        charges = []
        serials = []

    with path.open() as handle:
        for line in handle:
            if line.startswith("MODEL"):
                finish()
            elif line.startswith("REMARK VINA RESULT:"):
                try:
                    score = float(line.split()[3])
                except (ValueError, IndexError) as exc:
                    raise CampaignError(f"invalid Vina result line in {path}") from exc
            elif line.startswith("REMARK INDEX MAP"):
                try:
                    values = [int(value) for value in line.split()[3:]]
                except ValueError as exc:
                    raise CampaignError(f"invalid Vina index-map line in {path}") from exc
                if len(values) % 2:
                    raise CampaignError(f"odd Vina index-map value count in {path}")
                for original, pdbqt in zip(values[::2], values[1::2], strict=True):
                    if pdbqt in index_map and index_map[pdbqt] != original:
                        raise CampaignError(f"inconsistent Vina index map in {path}")
                    index_map[pdbqt] = original
            elif line.startswith(("ATOM  ", "HETATM")):
                tokens = line.split()
                try:
                    atom_type = tokens[-1].upper()
                    charge = float(tokens[-2])
                    serial = int(line[6:11])
                    coordinate = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                except (ValueError, IndexError) as exc:
                    raise CampaignError(f"invalid ligand atom line in {path}") from exc
                # Meeko inserts G0/G1 macrocycle-closure pseudoatoms.  They are
                # search constraints, not ligand atoms, and have no SDF index.
                if not atom_type.startswith(("H", "G")):
                    coordinates.append(coordinate)
                    atom_types.append(atom_type)
                    charges.append(charge)
                    serials.append(serial)
            elif line.startswith("ENDMDL"):
                finish()
    finish()
    if not poses:
        raise CampaignError(f"no Vina poses parsed from {path}")
    atom_counts = {len(pose["coordinates"]) for pose in poses}
    if len(atom_counts) != 1:
        raise CampaignError(f"Vina poses have inconsistent heavy-atom counts in {path}")
    if not index_map:
        raise CampaignError(f"Vina pose file lacks an atom index map: {path}")
    for pose in poses:
        pose["index_map"] = index_map
    return poses


def _ligand_roles(sdf_path: Path, pose: dict[str, Any]) -> dict[str, np.ndarray]:
    molecules = [molecule for molecule in Chem.SDMolSupplier(str(sdf_path), removeHs=False) if molecule]
    if len(molecules) != 1:
        raise CampaignError(f"expected one prepared ligand in {sdf_path}")
    molecule = molecules[0]
    atoms = []
    for serial in pose["serials"]:
        original = pose["index_map"].get(int(serial))
        if original is None or not 1 <= original <= molecule.GetNumAtoms():
            raise CampaignError(f"Vina-to-SDF atom mapping is incomplete for {sdf_path}")
        atom = molecule.GetAtomWithIdx(original - 1)
        if atom.GetAtomicNum() == 1:
            raise CampaignError(f"heavy Vina atom maps to hydrogen in {sdf_path}")
        atoms.append(atom)
    return {
        "formal_positive": np.asarray([atom.GetFormalCharge() > 0 for atom in atoms], dtype=bool),
        "formal_negative": np.asarray([atom.GetFormalCharge() < 0 for atom in atoms], dtype=bool),
        "aromatic": np.asarray([atom.GetIsAromatic() for atom in atoms], dtype=bool),
        "polar": np.asarray(
            [atom.GetAtomicNum() in {7, 8, 15, 16} for atom in atoms], dtype=bool
        ),
        "halogen": np.asarray(
            [atom.GetAtomicNum() in {9, 17, 35, 53} for atom in atoms], dtype=bool
        ),
    }


def _distance_matrix(ligand: np.ndarray, receptor: np.ndarray) -> np.ndarray:
    delta = ligand[:, None, :] - receptor[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights))


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def _pose_ensemble_features(
    path: Path,
    receptor: dict[str, Any],
    sdf_path: Path,
) -> dict[str, float]:
    poses = _parse_pose_file(path)
    roles = _ligand_roles(sdf_path, poses[0])
    scores = np.asarray([pose["score"] for pose in poses], dtype=float)
    relative = scores - np.min(scores)
    weights = np.exp(-relative)
    weights /= np.sum(weights)
    heavy_atoms = len(poses[0]["coordinates"])
    centroids = np.asarray([np.mean(pose["coordinates"], axis=0) for pose in poses])
    center_distance = np.linalg.norm(centroids - receptor["center"][None, :], axis=1)
    centroid_spread = float(
        np.sqrt(np.mean(np.sum((centroids - np.mean(centroids, axis=0)) ** 2, axis=1)))
    )
    coordinate_stack = np.asarray([pose["coordinates"] for pose in poses])
    ligand_types = np.asarray(poses[0]["atom_types"], dtype=object)
    ligand_charges = poses[0]["charges"]
    ligand_hetero = np.asarray(
        [not (value.startswith("C") or value == "A") for value in ligand_types], dtype=bool
    )
    ligand_hydrophobic = ~ligand_hetero
    ligand_positive = ligand_charges >= 0.20
    ligand_negative = ligand_charges <= -0.20
    ligand_hetero_count = int(np.sum(ligand_hetero))
    ligand_hydrophobic_count = int(np.sum(ligand_hydrophobic))
    ligand_positive_count = int(np.sum(ligand_positive))
    ligand_negative_count = int(np.sum(ligand_negative))
    ligand_formal_positive_count = int(np.sum(roles["formal_positive"]))
    ligand_formal_negative_count = int(np.sum(roles["formal_negative"]))
    ligand_aromatic_count = int(np.sum(roles["aromatic"]))
    ligand_polar_count = int(np.sum(roles["polar"]))
    ligand_halogen_count = int(np.sum(roles["halogen"]))
    pose_spread = float(
        np.sqrt(np.mean(np.sum((coordinate_stack - coordinate_stack[0:1]) ** 2, axis=2)))
    )
    result: dict[str, float] = {
        "pose_ensemble_count": float(len(poses)),
        "pose_score_best": float(np.min(scores)),
        "pose_score_mean": float(np.mean(scores)),
        "pose_score_sd": float(np.std(scores)),
        "pose_score_range": float(np.ptp(scores)),
        "pose_effective_count": float(1.0 / np.sum(weights**2)),
        "pose_centroid_spread": centroid_spread,
        "pose_coordinate_spread_from_best": pose_spread,
        "pose_center_distance_best": float(center_distance[0]),
        "pose_center_distance_weighted": _weighted_mean(center_distance, weights),
        "pose_heavy_atom_count": float(heavy_atoms),
        "pose_ligand_hetero_atom_count": float(ligand_hetero_count),
        "pose_ligand_hydrophobic_atom_count": float(ligand_hydrophobic_count),
        "pose_ligand_positive_atom_count": float(ligand_positive_count),
        "pose_ligand_negative_atom_count": float(ligand_negative_count),
        "pose_ligand_formal_positive_atom_count": float(ligand_formal_positive_count),
        "pose_ligand_formal_negative_atom_count": float(ligand_formal_negative_count),
        "pose_ligand_aromatic_atom_count": float(ligand_aromatic_count),
        "pose_ligand_polar_atom_count": float(ligand_polar_count),
        "pose_ligand_halogen_atom_count": float(ligand_halogen_count),
    }
    all_contact_counts = []
    all_hetero_counts = []
    all_hydrophobic_counts = []
    all_positive_counts = []
    all_negative_counts = []
    all_minimum_distances = []
    for residue in CONTACT_RESIDUES:
        minimum_distances = []
        contact_counts = []
        hetero_counts = []
        hydrophobic_counts = []
        positive_counts = []
        negative_counts = []
        aromatic_counts = []
        polar_counts = []
        halogen_counts = []
        cation_ring_distances = []
        cation_ring_engagement = []
        receptor_atoms = receptor["by_residue"][residue]
        for pose in poses:
            matrix = _distance_matrix(pose["coordinates"], receptor_atoms)
            per_atom = np.min(matrix, axis=1)
            contact = per_atom <= 4.0
            types = np.asarray(pose["atom_types"], dtype=object)
            charges = pose["charges"]
            hetero = np.asarray(
                [not (value.startswith("C") or value == "A") for value in types], dtype=bool
            )
            hydrophobic = np.asarray(
                [value.startswith("C") or value == "A" for value in types], dtype=bool
            )
            minimum_distances.append(float(np.min(matrix)))
            contact_counts.append(float(np.sum(contact)))
            hetero_counts.append(float(np.sum(contact & hetero)))
            hydrophobic_counts.append(float(np.sum(contact & hydrophobic)))
            positive_counts.append(float(np.sum(contact & (charges >= 0.20))))
            negative_counts.append(float(np.sum(contact & (charges <= -0.20))))
            aromatic_counts.append(float(np.sum(contact & roles["aromatic"])))
            polar_counts.append(float(np.sum(contact & roles["polar"])))
            halogen_counts.append(float(np.sum(contact & roles["halogen"])))
            if residue in receptor["ring_centroids"] and roles["formal_positive"].any():
                cation_coordinates = pose["coordinates"][roles["formal_positive"]]
                ring_distance = float(
                    np.min(
                        _distance_matrix(
                            cation_coordinates,
                            receptor["ring_centroids"][residue],
                        )
                    )
                )
            else:
                ring_distance = 12.0
            cation_ring_distances.append(ring_distance)
            cation_ring_engagement.append(float(ring_distance <= 6.0))
        minimum_distances_array = np.asarray(minimum_distances)
        contact_counts_array = np.asarray(contact_counts)
        hetero_counts_array = np.asarray(hetero_counts)
        hydrophobic_counts_array = np.asarray(hydrophobic_counts)
        positive_counts_array = np.asarray(positive_counts)
        negative_counts_array = np.asarray(negative_counts)
        aromatic_counts_array = np.asarray(aromatic_counts)
        polar_counts_array = np.asarray(polar_counts)
        halogen_counts_array = np.asarray(halogen_counts)
        cation_ring_distances_array = np.asarray(cation_ring_distances)
        cation_ring_engagement_array = np.asarray(cation_ring_engagement)
        all_contact_counts.append(contact_counts_array)
        all_hetero_counts.append(hetero_counts_array)
        all_hydrophobic_counts.append(hydrophobic_counts_array)
        all_positive_counts.append(positive_counts_array)
        all_negative_counts.append(negative_counts_array)
        all_minimum_distances.append(minimum_distances_array)
        prefix = f"pose_residue_{residue}"
        result.update(
            {
                f"{prefix}_minimum_distance_best": float(minimum_distances_array[0]),
                f"{prefix}_minimum_distance_weighted": _weighted_mean(
                    minimum_distances_array, weights
                ),
                f"{prefix}_minimum_distance_sd": float(np.std(minimum_distances_array)),
                f"{prefix}_contact_count_best": float(contact_counts_array[0]),
                f"{prefix}_contact_count_weighted": _weighted_mean(
                    contact_counts_array, weights
                ),
                f"{prefix}_contact_count_sd": float(np.std(contact_counts_array)),
                f"{prefix}_contact_occupancy": float(np.mean(contact_counts_array > 0)),
                f"{prefix}_hetero_count_weighted": _weighted_mean(
                    hetero_counts_array, weights
                ),
                f"{prefix}_hydrophobic_count_weighted": _weighted_mean(
                    hydrophobic_counts_array, weights
                ),
                f"{prefix}_positive_count_weighted": _weighted_mean(
                    positive_counts_array, weights
                ),
                f"{prefix}_negative_count_weighted": _weighted_mean(
                    negative_counts_array, weights
                ),
                f"{prefix}_aromatic_count_weighted": _weighted_mean(
                    aromatic_counts_array, weights
                ),
                f"{prefix}_polar_count_weighted": _weighted_mean(
                    polar_counts_array, weights
                ),
                f"{prefix}_halogen_count_weighted": _weighted_mean(
                    halogen_counts_array, weights
                ),
                f"{prefix}_aromatic_contact_fraction": _weighted_mean(
                    aromatic_counts_array, weights
                )
                / max(1, ligand_aromatic_count),
                f"{prefix}_polar_contact_fraction": _weighted_mean(
                    polar_counts_array, weights
                )
                / max(1, ligand_polar_count),
                f"{prefix}_halogen_contact_fraction": _weighted_mean(
                    halogen_counts_array, weights
                )
                / max(1, ligand_halogen_count),
                f"{prefix}_cation_ring_distance_weighted": _weighted_mean(
                    cation_ring_distances_array, weights
                ),
                f"{prefix}_cation_ring_engagement": _weighted_mean(
                    cation_ring_engagement_array, weights
                ),
                f"{prefix}_hetero_contact_fraction": _weighted_mean(
                    hetero_counts_array, weights
                )
                / max(1, ligand_hetero_count),
                f"{prefix}_hydrophobic_contact_fraction": _weighted_mean(
                    hydrophobic_counts_array, weights
                )
                / max(1, ligand_hydrophobic_count),
                f"{prefix}_positive_contact_fraction": _weighted_mean(
                    positive_counts_array, weights
                )
                / max(1, ligand_positive_count),
                f"{prefix}_negative_contact_fraction": _weighted_mean(
                    negative_counts_array, weights
                )
                / max(1, ligand_negative_count),
                f"{prefix}_score_contact_correlation": _safe_correlation(
                    scores, contact_counts_array
                ),
            }
        )
    total_contacts = np.sum(np.asarray(all_contact_counts), axis=0)
    total_hetero = np.sum(np.asarray(all_hetero_counts), axis=0)
    total_hydrophobic = np.sum(np.asarray(all_hydrophobic_counts), axis=0)
    total_positive = np.sum(np.asarray(all_positive_counts), axis=0)
    total_negative = np.sum(np.asarray(all_negative_counts), axis=0)
    minimum_pocket_distance = np.min(np.asarray(all_minimum_distances), axis=0)
    cage_indices = [CONTACT_RESIDUES.index(value) for value in (652, 656)]
    cage_contacts = np.sum(np.asarray(all_contact_counts)[cage_indices], axis=0)
    cage_cation_distances = []
    cage_cation_engagement = []
    for pose in poses:
        if roles["formal_positive"].any():
            all_ring_centroids = np.concatenate(
                [receptor["ring_centroids"][residue] for residue in (652, 656)],
                axis=0,
            )
            value = float(
                np.min(
                    _distance_matrix(
                        pose["coordinates"][roles["formal_positive"]],
                        all_ring_centroids,
                    )
                )
            )
        else:
            value = 12.0
        cage_cation_distances.append(value)
        cage_cation_engagement.append(float(value <= 6.0))
    cage_cation_distances_array = np.asarray(cage_cation_distances)
    cage_cation_engagement_array = np.asarray(cage_cation_engagement)
    result.update(
        {
            "pose_total_contact_count_best": float(total_contacts[0]),
            "pose_total_contact_count_weighted": _weighted_mean(total_contacts, weights),
            "pose_total_contact_count_sd": float(np.std(total_contacts)),
            "pose_total_contact_density_weighted": _weighted_mean(total_contacts, weights)
            / heavy_atoms,
            "pose_total_hetero_count_weighted": _weighted_mean(total_hetero, weights),
            "pose_total_hydrophobic_count_weighted": _weighted_mean(total_hydrophobic, weights),
            "pose_total_positive_count_weighted": _weighted_mean(total_positive, weights),
            "pose_total_negative_count_weighted": _weighted_mean(total_negative, weights),
            "pose_total_hetero_contact_fraction": _weighted_mean(total_hetero, weights)
            / max(1, ligand_hetero_count),
            "pose_total_hydrophobic_contact_fraction": _weighted_mean(
                total_hydrophobic, weights
            )
            / max(1, ligand_hydrophobic_count),
            "pose_total_positive_contact_fraction": _weighted_mean(total_positive, weights)
            / max(1, ligand_positive_count),
            "pose_total_negative_contact_fraction": _weighted_mean(total_negative, weights)
            / max(1, ligand_negative_count),
            "pose_cage_contact_count_best": float(cage_contacts[0]),
            "pose_cage_contact_count_weighted": _weighted_mean(cage_contacts, weights),
            "pose_cage_contact_occupancy": float(np.mean(cage_contacts > 0)),
            "pose_cage_cation_distance_best": float(cage_cation_distances_array[0]),
            "pose_cage_cation_distance_weighted": _weighted_mean(
                cage_cation_distances_array, weights
            ),
            "pose_cage_cation_engagement": _weighted_mean(
                cage_cation_engagement_array, weights
            ),
            "pose_minimum_pocket_distance_best": float(minimum_pocket_distance[0]),
            "pose_minimum_pocket_distance_weighted": _weighted_mean(
                minimum_pocket_distance, weights
            ),
            "pose_score_total_contact_correlation": _safe_correlation(scores, total_contacts),
            "pose_score_cage_contact_correlation": _safe_correlation(scores, cage_contacts),
        }
    )
    return result


def _pose_path(root: Path, ligand_id: str) -> Path:
    path = root / "docking/tasks" / f"{ligand_id}__m0__{STATE}" / "poses.pdbqt"
    if not path.is_file():
        raise CampaignError(f"missing frozen Vina pose file: {path}")
    return path


def _ligand_sdf_path(root: Path, ligand_id: str) -> Path:
    path = root / "ligands" / ligand_id / "microstate_0.sdf"
    if not path.is_file():
        raise CampaignError(f"missing frozen prepared ligand SDF: {path}")
    return path


def _extract_pose_features(
    repo: Path,
    frame: pd.DataFrame,
    receptor_path: Path,
) -> pd.DataFrame:
    receptor = _parse_receptor_atoms(receptor_path)
    roots = {spec.name: _resolve(repo, spec.root) for spec in CAMPAIGNS}
    rows = []
    total = len(frame)
    for completed, row in enumerate(frame.itertuples(index=False), start=1):
        path = _pose_path(roots[str(row.campaign)], str(row.ligand_id))
        sdf_path = _ligand_sdf_path(roots[str(row.campaign)], str(row.ligand_id))
        features = _pose_ensemble_features(path, receptor, sdf_path)
        rows.append(
            {
                "sample_id": str(row.sample_id),
                "campaign": str(row.campaign),
                "ligand_id": str(row.ligand_id),
                "pose_path": str(path.resolve()),
                "pose_sha256": _sha(path),
                **features,
            }
        )
        if completed % 100 == 0 or completed == total:
            print(
                json.dumps(
                    {"stage": "pose_features", "completed": completed, "total": total}
                ),
                flush=True,
            )
    result = pd.DataFrame(rows)
    if len(result) != len(frame) or result.sample_id.duplicated().any():
        raise CampaignError("pose feature extraction did not preserve one row per sample")
    feature_columns = [column for column in result if column.startswith("pose_")]
    numeric = result[[column for column in feature_columns if column not in {"pose_path", "pose_sha256"}]]
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise CampaignError("pose feature extraction produced non-finite values")
    return result


def _prepare(
    repo: Path,
    output: Path,
    router_path: Path,
    receptor_path: Path,
) -> dict[str, Any]:
    frame = _harmonize(repo, router_path)
    overlap = _overlap_audit(frame)
    if overlap[["exact_connectivity_overlap", "scaffold_overlap"]].to_numpy(int).any():
        raise CampaignError("cross-campaign exact or scaffold overlap was detected")
    census = _campaign_census(frame)
    pose_features = _extract_pose_features(repo, frame, receptor_path)
    data_path = output / "data/harmonized_campaigns.parquet"
    pose_path = output / "data/pose_ensemble_features.parquet"
    overlap_path = output / "evidence/cross_campaign_overlap_audit.parquet"
    census_path = output / "evidence/campaign_census.parquet"
    _parquet(data_path, frame)
    _parquet(pose_path, pose_features)
    _parquet(overlap_path, overlap)
    _parquet(census_path, census)
    return _write_json(
        output / "prepare_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete",
            "campaigns": len(CAMPAIGNS),
            "rows": len(frame),
            "unique_connectivity_keys": int(frame.cross_campaign_connectivity_key.nunique()),
            "unique_scaffolds": int(frame.cross_campaign_scaffold_key.nunique()),
            "pose_feature_columns": int(
                sum(
                    column.startswith("pose_")
                    and column not in {"pose_path", "pose_sha256"}
                    for column in pose_features
                )
            ),
            "pairwise_exact_overlap": int(overlap.exact_connectivity_overlap.sum()),
            "pairwise_scaffold_overlap": int(overlap.scaffold_overlap.sum()),
            "inputs": {
                "router_oof": {
                    "path": str(router_path),
                    "sha256": _sha(router_path),
                },
                "prepared_receptor": {
                    "path": str(receptor_path),
                    "sha256": _sha(receptor_path),
                },
            },
            "scientific_scope": {
                "analysis_is_exploratory_after_all_six_campaign_labels_opened": True,
                "outer_validation_unit": "entire campaign",
                "cross_campaign_exact_and_scaffold_overlap": False,
                "pose_features_use_all_frozen_vina_modes": True,
                "new_docking_performed": False,
                "vina_scores_are_binding_free_energies": False,
            },
        },
        "report_sha256",
    )


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _log_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    left = np.clip(numerator.to_numpy(float), 1e-6, 1.0)
    right = np.clip(denominator.to_numpy(float), 1e-6, 1.0)
    return np.log(left / right)


def _build_model_matrix(frame: pd.DataFrame, pose: pd.DataFrame) -> pd.DataFrame:
    matrix = frame.merge(
        pose.drop(columns=["pose_path", "pose_sha256"]),
        on=["sample_id", "campaign", "ligand_id"],
        validate="one_to_one",
    )
    matrix["ligand_log_safe_vs_moderate"] = _log_ratio(
        matrix[ROUTER_COLUMNS[0]], matrix[ROUTER_COLUMNS[1]]
    )
    matrix["ligand_log_potent_vs_moderate"] = _log_ratio(
        matrix[ROUTER_COLUMNS[2]], matrix[ROUTER_COLUMNS[1]]
    )
    matrix["ligand_router_entropy"] = _entropy(matrix[list(ROUTER_COLUMNS)].to_numpy(float))
    matrix["receptor_log_safe_vs_moderate"] = _log_ratio(
        matrix[FROZEN_RECEPTOR_COLUMNS[0]], matrix[FROZEN_RECEPTOR_COLUMNS[1]]
    )
    matrix["receptor_log_potent_vs_moderate"] = _log_ratio(
        matrix[FROZEN_RECEPTOR_COLUMNS[2]], matrix[FROZEN_RECEPTOR_COLUMNS[1]]
    )
    matrix["frozen_receptor_entropy"] = _entropy(
        matrix[list(FROZEN_RECEPTOR_COLUMNS)].to_numpy(float)
    )
    cationic = matrix.pose_ligand_formal_positive_atom_count.gt(0).to_numpy(float)
    for column in (
        "pose_residue_652_cation_ring_distance_weighted",
        "pose_residue_656_cation_ring_distance_weighted",
        "pose_cage_cation_distance_best",
        "pose_cage_cation_distance_weighted",
    ):
        matrix[f"{column}_conditional"] = cationic * (matrix[column].to_numpy(float) - 6.0)
    return matrix


LIGAND_CORE_FEATURES = (
    "ligand_log_safe_vs_moderate",
    "ligand_log_potent_vs_moderate",
    "baseline_pic50",
    "ligand_router_entropy",
)
LIGAND_CHEMISTRY_FEATURES = (
    "docking_molecular_weight",
    "docking_heavy_atom_count",
    "docking_rotatable_bond_count",
    "dock_formal_charge",
    "pose_ligand_hetero_atom_count",
    "pose_ligand_hydrophobic_atom_count",
    "pose_ligand_positive_atom_count",
    "pose_ligand_negative_atom_count",
    "pose_ligand_formal_positive_atom_count",
    "pose_ligand_formal_negative_atom_count",
    "pose_ligand_aromatic_atom_count",
    "pose_ligand_polar_atom_count",
    "pose_ligand_halogen_atom_count",
)
FROZEN_RECEPTOR_FEATURES = (
    "receptor_log_safe_vs_moderate",
    "receptor_log_potent_vs_moderate",
    "frozen_receptor_entropy",
)
MECHANISTIC_RECEPTOR_FEATURES = (
    "pose_score_sd",
    "pose_effective_count",
    "pose_centroid_spread",
    "pose_coordinate_spread_from_best",
    "pose_center_distance_weighted",
    "pose_residue_623_polar_contact_fraction",
    "pose_residue_624_polar_contact_fraction",
    "pose_residue_649_aromatic_contact_fraction",
    "pose_residue_649_halogen_contact_fraction",
    "pose_residue_652_aromatic_contact_fraction",
    "pose_residue_652_polar_contact_fraction",
    "pose_residue_652_halogen_contact_fraction",
    "pose_residue_652_cation_ring_engagement",
    "pose_residue_652_cation_ring_distance_weighted_conditional",
    "pose_residue_656_aromatic_contact_fraction",
    "pose_residue_656_polar_contact_fraction",
    "pose_residue_656_halogen_contact_fraction",
    "pose_residue_656_cation_ring_engagement",
    "pose_residue_656_cation_ring_distance_weighted_conditional",
    "pose_cage_cation_distance_best_conditional",
    "pose_cage_cation_distance_weighted_conditional",
    "pose_cage_cation_engagement",
    "pose_cage_contact_occupancy",
    "pose_total_contact_density_weighted",
    "pose_total_hetero_contact_fraction",
    "pose_total_hydrophobic_contact_fraction",
    "pose_total_positive_contact_fraction",
    "pose_total_negative_contact_fraction",
    "pose_minimum_pocket_distance_weighted",
)


def _feature_surfaces(matrix: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    excluded_pose = {
        "pose_ensemble_count",
        "pose_heavy_atom_count",
        "pose_score_best",
        "pose_score_mean",
        *(
            column
            for column in matrix
            if column.startswith("pose_ligand_")
        ),
    }
    all_pose_receptor = tuple(
        sorted(
            column
            for column in matrix
            if column.startswith("pose_")
            and column not in excluded_pose
            and not column.endswith("_conditional")
        )
    )
    matched = (*LIGAND_CORE_FEATURES, *LIGAND_CHEMISTRY_FEATURES)
    return {
        "ligand_calibrated": LIGAND_CORE_FEATURES,
        "ligand_calibrated_with_physchem": matched,
        "ligand_plus_frozen_receptor": (*matched, *FROZEN_RECEPTOR_FEATURES),
        "ligand_plus_mechanistic_pose": (*matched, *MECHANISTIC_RECEPTOR_FEATURES),
        "ligand_plus_frozen_and_mechanistic": (
            *matched,
            *FROZEN_RECEPTOR_FEATURES,
            *MECHANISTIC_RECEPTOR_FEATURES,
        ),
        "ligand_plus_all_pose_features": (*matched, *all_pose_receptor),
    }


def _classification_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=float)
    for positions in frame.groupby("campaign", sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        campaign_y = frame.iloc[positions].target_class.to_numpy(int)
        for class_index in range(3):
            selected = positions[campaign_y == class_index]
            if not len(selected):
                raise CampaignError("a training campaign lacks an observed class")
            weights[selected] = 1.0 / (3.0 * len(selected))
    return weights / np.mean(weights)


def _regression_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=float)
    for positions in frame.groupby("campaign", sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        weights[positions] = 1.0 / len(positions)
    return weights / np.mean(weights)


def _classifier(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    max_iter=5_000,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )


def _regressor(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _fit_classifier(frame: pd.DataFrame, features: tuple[str, ...], c_value: float) -> Pipeline:
    model = _classifier(c_value)
    model.fit(
        frame[list(features)],
        frame.target_class,
        model__sample_weight=_classification_weights(frame.reset_index(drop=True)),
    )
    return model


def _fit_regressor(frame: pd.DataFrame, features: tuple[str, ...], alpha: float) -> Pipeline:
    model = _regressor(alpha)
    model.fit(
        frame[list(features)],
        frame.target_pic50.to_numpy(float) - frame.baseline_pic50.to_numpy(float),
        model__sample_weight=_regression_weights(frame.reset_index(drop=True)),
    )
    return model


def _classification_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 1e-9, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    prediction = np.argmax(probabilities, axis=1)
    recalls = {
        name: float(np.mean(prediction[y == index] == index))
        for index, name in enumerate(CLASS_NAMES)
    }
    confidence = np.max(probabilities, axis=1)
    correctness = prediction == y
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        keep = (confidence >= lower) & (confidence < lower + 0.1 + 1e-12)
        if keep.any():
            ece += float(keep.mean() * abs(np.mean(correctness[keep]) - np.mean(confidence[keep])))
    one_hot = np.eye(3)[y]
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "top_class_ece_10_fixed_bins": ece,
        "per_class_recall": recalls,
        "potent_predicted_safe_rate": float(np.mean(prediction[y == 2] == 0)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
    }


def _regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    rho = float(spearmanr(y, prediction).statistic)
    return {
        "n": len(y),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "spearman": rho if np.isfinite(rho) else None,
        "within_0p5": float(np.mean(np.abs(y - prediction) <= 0.5)),
        "mean_error_prediction_minus_observed": float(np.mean(prediction - y)),
    }


def _inner_classification_predictions(
    training: pd.DataFrame,
    features: tuple[str, ...],
    c_value: float,
) -> pd.DataFrame:
    rows = []
    for campaign in sorted(training.campaign.unique()):
        fit = training.loc[training.campaign.ne(campaign)].reset_index(drop=True)
        evaluate = training.loc[training.campaign.eq(campaign)].copy()
        model = _fit_classifier(fit, features, c_value)
        probability = model.predict_proba(evaluate[list(features)])
        for index, name in enumerate(CLASS_NAMES):
            evaluate[f"probability_{name.lower()}"] = probability[:, index]
        rows.append(evaluate)
    return pd.concat(rows, ignore_index=True)


def _select_classifier_hyperparameter(
    training: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[float, pd.DataFrame]:
    candidates = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
    rows = []
    probability_columns = [f"probability_{name.lower()}" for name in CLASS_NAMES]
    for c_value in candidates:
        predictions = _inner_classification_predictions(training, features, c_value)
        campaign_metrics = []
        for campaign, group in predictions.groupby("campaign", sort=True):
            metrics = _classification_metrics(
                group.target_class.to_numpy(int), group[probability_columns].to_numpy(float)
            )
            campaign_metrics.append({"campaign": campaign, **metrics})
        metrics_frame = pd.DataFrame(campaign_metrics)
        rows.append(
            {
                "C": c_value,
                "macro_campaign_balanced_accuracy": float(metrics_frame.balanced_accuracy.mean()),
                "macro_campaign_macro_f1": float(metrics_frame.macro_f1.mean()),
                "macro_campaign_log_loss": float(metrics_frame.log_loss.mean()),
                "macro_campaign_potent_predicted_safe_rate": float(
                    metrics_frame.potent_predicted_safe_rate.mean()
                ),
            }
        )
    evidence = pd.DataFrame(rows)
    evidence["safety_gate"] = evidence.macro_campaign_potent_predicted_safe_rate.le(0.06)
    eligible = evidence.loc[evidence.safety_gate]
    pool = eligible if len(eligible) else evidence
    chosen = pool.sort_values(
        [
            "macro_campaign_balanced_accuracy",
            "macro_campaign_macro_f1",
            "macro_campaign_log_loss",
            "C",
        ],
        ascending=[False, False, True, True],
    ).iloc[0]
    return float(chosen.C), evidence


def _inner_regression_predictions(
    training: pd.DataFrame,
    features: tuple[str, ...],
    alpha: float,
) -> pd.DataFrame:
    rows = []
    for campaign in sorted(training.campaign.unique()):
        fit = training.loc[training.campaign.ne(campaign)].reset_index(drop=True)
        evaluate = training.loc[training.campaign.eq(campaign)].copy()
        model = _fit_regressor(fit, features, alpha)
        evaluate["prediction"] = evaluate.baseline_pic50.to_numpy(float) + model.predict(
            evaluate[list(features)]
        )
        rows.append(evaluate)
    return pd.concat(rows, ignore_index=True)


def _select_regressor_hyperparameter(
    training: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[float, pd.DataFrame]:
    candidates = (1.0, 10.0, 100.0, 1_000.0, 10_000.0)
    rows = []
    for alpha in candidates:
        predictions = _inner_regression_predictions(training, features, alpha)
        metrics = []
        for campaign, group in predictions.groupby("campaign", sort=True):
            metrics.append(
                {
                    "campaign": campaign,
                    **_regression_metrics(group.target_pic50, group.prediction),
                }
            )
        metrics_frame = pd.DataFrame(metrics)
        rows.append(
            {
                "alpha": alpha,
                "macro_campaign_mae": float(metrics_frame.mae.mean()),
                "macro_campaign_rmse": float(metrics_frame.rmse.mean()),
                "macro_campaign_spearman": float(metrics_frame.spearman.mean()),
            }
        )
    evidence = pd.DataFrame(rows)
    chosen = evidence.sort_values(
        ["macro_campaign_mae", "macro_campaign_rmse", "macro_campaign_spearman", "alpha"],
        ascending=[True, True, False, False],
    ).iloc[0]
    return float(chosen.alpha), evidence


def _nested_loco(
    matrix: pd.DataFrame,
    surfaces: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    tuning_rows = []
    campaigns = sorted(matrix.campaign.unique())
    for outer_index, outer_campaign in enumerate(campaigns, start=1):
        training = matrix.loc[matrix.campaign.ne(outer_campaign)].reset_index(drop=True)
        evaluation = matrix.loc[matrix.campaign.eq(outer_campaign)].copy()
        frozen = evaluation[
            [
                "sample_id",
                "campaign",
                "campaign_kind",
                "ligand_id",
                "cross_campaign_scaffold_key",
                "target_class",
                "target_pic50",
                "baseline_pic50",
                *ROUTER_COLUMNS,
            ]
        ].copy()
        frozen["surface"] = "frozen_ligand_router"
        frozen["selected_hyperparameter"] = math.nan
        frozen["predicted_pic50"] = frozen.baseline_pic50
        for index, name in enumerate(CLASS_NAMES):
            frozen[f"probability_{name.lower()}"] = frozen[ROUTER_COLUMNS[index]].to_numpy(float)
        prediction_rows.append(frozen)
        for surface, features in surfaces.items():
            missing = set(features) - set(matrix)
            if missing:
                raise CampaignError(f"{surface} lacks model features: {sorted(missing)}")
            c_value, class_tuning = _select_classifier_hyperparameter(training, features)
            class_model = _fit_classifier(training, features, c_value)
            probabilities = class_model.predict_proba(evaluation[list(features)])
            alpha, regression_tuning = _select_regressor_hyperparameter(training, features)
            regression_model = _fit_regressor(training, features, alpha)
            predicted_pic50 = evaluation.baseline_pic50.to_numpy(float) + regression_model.predict(
                evaluation[list(features)]
            )
            output = evaluation[
                [
                    "sample_id",
                    "campaign",
                    "campaign_kind",
                    "ligand_id",
                    "cross_campaign_scaffold_key",
                    "target_class",
                    "target_pic50",
                    "baseline_pic50",
                    *ROUTER_COLUMNS,
                ]
            ].copy()
            output["surface"] = surface
            output["selected_hyperparameter"] = c_value
            output["selected_regression_alpha"] = alpha
            output["predicted_pic50"] = predicted_pic50
            for index, name in enumerate(CLASS_NAMES):
                output[f"probability_{name.lower()}"] = probabilities[:, index]
            prediction_rows.append(output)
            class_tuning.insert(0, "task", "classification")
            class_tuning.insert(0, "surface", surface)
            class_tuning.insert(0, "outer_campaign", outer_campaign)
            tuning_rows.append(class_tuning)
            regression_tuning.insert(0, "task", "regression")
            regression_tuning.insert(0, "surface", surface)
            regression_tuning.insert(0, "outer_campaign", outer_campaign)
            tuning_rows.append(regression_tuning)
        print(
            json.dumps(
                {
                    "stage": "nested_loco",
                    "completed_outer_campaigns": outer_index,
                    "total_outer_campaigns": len(campaigns),
                    "outer_campaign": outer_campaign,
                }
            ),
            flush=True,
        )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    tuning = pd.concat(tuning_rows, ignore_index=True, sort=False)
    expected = len(matrix) * (len(surfaces) + 1)
    if len(predictions) != expected:
        raise CampaignError(f"nested LOCO produced {len(predictions)} rows, expected {expected}")
    return predictions, tuning


def _score_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    probability_columns = [f"probability_{name.lower()}" for name in CLASS_NAMES]
    class_rows = []
    regression_rows = []
    aggregate: dict[str, Any] = {}
    for surface, surface_frame in predictions.groupby("surface", sort=True):
        per_campaign_class = []
        per_campaign_regression = []
        for campaign, group in surface_frame.groupby("campaign", sort=True):
            classification = _classification_metrics(
                group.target_class.to_numpy(int), group[probability_columns].to_numpy(float)
            )
            regression = _regression_metrics(group.target_pic50, group.predicted_pic50)
            class_rows.append({"surface": surface, "campaign": campaign, **classification})
            regression_rows.append({"surface": surface, "campaign": campaign, **regression})
            per_campaign_class.append(classification)
            per_campaign_regression.append(regression)
        pooled_class = _classification_metrics(
            surface_frame.target_class.to_numpy(int),
            surface_frame[probability_columns].to_numpy(float),
        )
        pooled_regression = _regression_metrics(
            surface_frame.target_pic50, surface_frame.predicted_pic50
        )
        aggregate[surface] = {
            "classification": {
                "macro_by_campaign": {
                    key: float(np.mean([row[key] for row in per_campaign_class]))
                    for key in (
                        "accuracy",
                        "balanced_accuracy",
                        "macro_f1",
                        "log_loss",
                        "multiclass_brier",
                        "top_class_ece_10_fixed_bins",
                        "potent_predicted_safe_rate",
                    )
                },
                "pooled": pooled_class,
            },
            "regression": {
                "macro_by_campaign": {
                    key: float(np.mean([row[key] for row in per_campaign_regression]))
                    for key in ("mae", "rmse", "spearman", "within_0p5")
                },
                "pooled": pooled_regression,
            },
        }
    return pd.DataFrame(class_rows), pd.DataFrame(regression_rows), aggregate


def _bootstrap_indices(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    groups = [group.index.to_numpy(int) for _, group in frame.groupby("cross_campaign_scaffold_key")]
    for _attempt in range(100):
        chosen = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[index] for index in chosen])
        if set(frame.target_class.to_numpy(int)[rows]) == {0, 1, 2}:
            return rows
    raise CampaignError("could not create a three-class scaffold bootstrap sample")


def _paired_bootstrap(
    predictions: pd.DataFrame,
    candidate: str,
    comparator: str,
    replicates: int,
) -> dict[str, Any]:
    if replicates < 100:
        raise CampaignError("at least 100 bootstrap replicates are required")
    keys = ["sample_id", "campaign", "cross_campaign_scaffold_key", "target_class", "target_pic50"]
    probability_columns = [f"probability_{name.lower()}" for name in CLASS_NAMES]
    left = predictions.loc[predictions.surface.eq(candidate), [*keys, "predicted_pic50", *probability_columns]]
    right = predictions.loc[predictions.surface.eq(comparator), [*keys, "predicted_pic50", *probability_columns]]
    merged = left.merge(right, on=keys, suffixes=("_candidate", "_comparator"), validate="one_to_one")
    campaigns = sorted(merged.campaign.unique())
    observed_ba = []
    observed_mae = []
    for campaign in campaigns:
        group = merged.loc[merged.campaign.eq(campaign)]
        y = group.target_class.to_numpy(int)
        candidate_class = np.argmax(
            group[[f"{column}_candidate" for column in probability_columns]].to_numpy(float), axis=1
        )
        comparator_class = np.argmax(
            group[[f"{column}_comparator" for column in probability_columns]].to_numpy(float), axis=1
        )
        observed_ba.append(
            balanced_accuracy_score(y, candidate_class)
            - balanced_accuracy_score(y, comparator_class)
        )
        observed_mae.append(
            mean_absolute_error(group.target_pic50, group.predicted_pic50_candidate)
            - mean_absolute_error(group.target_pic50, group.predicted_pic50_comparator)
        )
    rng = np.random.default_rng(SEED + int(hashlib.sha256(candidate.encode()).hexdigest()[:6], 16))
    values = np.empty((replicates, 2), dtype=float)
    by_campaign = {
        campaign: merged.loc[merged.campaign.eq(campaign)].reset_index(drop=True)
        for campaign in campaigns
    }
    for iteration in range(replicates):
        ba_deltas = []
        mae_deltas = []
        for campaign in campaigns:
            group = by_campaign[campaign]
            rows = _bootstrap_indices(group, rng)
            sampled = group.iloc[rows]
            y = sampled.target_class.to_numpy(int)
            candidate_class = np.argmax(
                sampled[[f"{column}_candidate" for column in probability_columns]].to_numpy(float),
                axis=1,
            )
            comparator_class = np.argmax(
                sampled[[f"{column}_comparator" for column in probability_columns]].to_numpy(float),
                axis=1,
            )
            ba_deltas.append(
                balanced_accuracy_score(y, candidate_class)
                - balanced_accuracy_score(y, comparator_class)
            )
            mae_deltas.append(
                mean_absolute_error(sampled.target_pic50, sampled.predicted_pic50_candidate)
                - mean_absolute_error(sampled.target_pic50, sampled.predicted_pic50_comparator)
            )
        values[iteration] = [np.mean(ba_deltas), np.mean(mae_deltas)]
    observed_ba_value = float(np.mean(observed_ba))
    observed_mae_value = float(np.mean(observed_mae))
    return {
        "candidate": candidate,
        "comparator": comparator,
        "replicates": replicates,
        "resampling": "scaffold-cluster bootstrap within campaign; macro-average across six campaigns",
        "classification_delta_balanced_accuracy": {
            "observed": observed_ba_value,
            "ci95": [float(np.quantile(values[:, 0], 0.025)), float(np.quantile(values[:, 0], 0.975))],
            "bootstrap_probability_candidate_better": float(np.mean(values[:, 0] > 0)),
            "campaigns_better": int(np.sum(np.asarray(observed_ba) > 0)),
            "campaign_deltas": dict(zip(campaigns, map(float, observed_ba), strict=True)),
        },
        "regression_delta_mae": {
            "observed": observed_mae_value,
            "ci95": [float(np.quantile(values[:, 1], 0.025)), float(np.quantile(values[:, 1], 0.975))],
            "bootstrap_probability_candidate_better": float(np.mean(values[:, 1] < 0)),
            "campaigns_better": int(np.sum(np.asarray(observed_mae) < 0)),
            "campaign_deltas": dict(zip(campaigns, map(float, observed_mae), strict=True)),
        },
    }


def _association_audit(matrix: pd.DataFrame) -> pd.DataFrame:
    features = {
        "legacy_contact_649_actually_S649": "dock_contact_649_count",
        "vina_affinity": "dock_affinity",
        "pose_cage_cation_engagement": "pose_cage_cation_engagement",
        "pose_Y652_cation_engagement": "pose_residue_652_cation_ring_engagement",
        "pose_F656_aromatic_contact_fraction": "pose_residue_656_aromatic_contact_fraction",
        "pose_S624_polar_contact_fraction": "pose_residue_624_polar_contact_fraction",
    }
    rows = []
    for label, feature in features.items():
        for campaign, group in matrix.groupby("campaign", sort=True):
            result = spearmanr(group[feature], group.target_pic50)
            rows.append(
                {
                    "feature_label": label,
                    "feature_column": feature,
                    "campaign": campaign,
                    "n": len(group),
                    "spearman": float(result.statistic) if np.isfinite(result.statistic) else 0.0,
                    "two_sided_p_value_descriptive": float(result.pvalue)
                    if np.isfinite(result.pvalue)
                    else 1.0,
                }
            )
    return pd.DataFrame(rows)


def _fit_full_models(
    matrix: pd.DataFrame,
    surfaces: dict[str, tuple[str, ...]],
    output: Path,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for surface, features in surfaces.items():
        c_value, _class_evidence = _select_classifier_hyperparameter(matrix, features)
        alpha, _regression_evidence = _select_regressor_hyperparameter(matrix, features)
        classifier = _fit_classifier(matrix.reset_index(drop=True), features, c_value)
        regressor = _fit_regressor(matrix.reset_index(drop=True), features, alpha)
        path = model_dir / f"{surface}.joblib"
        joblib.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "surface": surface,
                "features": list(features),
                "class_names": list(CLASS_NAMES),
                "safe_pic50_threshold": SAFE_PIC50,
                "potent_pic50_threshold": POTENT_PIC50,
                "classifier_C": c_value,
                "regression_alpha": alpha,
                "classifier": classifier,
                "residual_regressor": regressor,
                "regression_anchor": "baseline_pic50",
            },
            path,
            compress=3,
        )
        models[surface] = {
            "path": str(path.resolve()),
            "sha256": _sha(path),
            "bytes": path.stat().st_size,
            "features": list(features),
            "classifier_C": c_value,
            "regression_alpha": alpha,
        }
    return models


def _promotion_decision(
    aggregate: dict[str, Any],
    bootstraps: dict[str, Any],
) -> dict[str, Any]:
    matched_comparator = "ligand_calibrated_with_physchem"
    best_ligand_comparator = "ligand_calibrated"
    receptor_surfaces = [
        surface
        for surface in aggregate
        if surface.startswith("ligand_plus_")
    ]
    best_class = max(
        receptor_surfaces,
        key=lambda surface: aggregate[surface]["classification"]["macro_by_campaign"][
            "balanced_accuracy"
        ],
    )
    best_regression = min(
        receptor_surfaces,
        key=lambda surface: aggregate[surface]["regression"]["macro_by_campaign"]["mae"],
    )
    class_bootstrap = bootstraps[best_class]["vs_best_ligand"]
    regression_bootstrap = bootstraps[best_regression]["vs_best_ligand"]
    comparator_safety = aggregate[best_ligand_comparator]["classification"]["macro_by_campaign"][
        "potent_predicted_safe_rate"
    ]
    candidate_safety = aggregate[best_class]["classification"]["macro_by_campaign"][
        "potent_predicted_safe_rate"
    ]
    class_gate = bool(
        class_bootstrap["classification_delta_balanced_accuracy"]["ci95"][0] > 0
        and class_bootstrap["classification_delta_balanced_accuracy"]["campaigns_better"] >= 4
        and aggregate[best_class]["classification"]["macro_by_campaign"]["macro_f1"]
        > aggregate[best_ligand_comparator]["classification"]["macro_by_campaign"]["macro_f1"]
        and candidate_safety <= 0.05
        and candidate_safety <= comparator_safety + 0.02
    )
    regression_gate = bool(
        regression_bootstrap["regression_delta_mae"]["ci95"][1] < 0
        and regression_bootstrap["regression_delta_mae"]["campaigns_better"] >= 4
    )
    ligand_bootstrap = bootstraps[best_ligand_comparator]
    ligand_class = aggregate[best_ligand_comparator]["classification"]["macro_by_campaign"]
    frozen_class = aggregate["frozen_ligand_router"]["classification"]["macro_by_campaign"]
    ligand_recalibration_gate = bool(
        ligand_bootstrap["classification_delta_balanced_accuracy"]["ci95"][0] > 0
        and ligand_bootstrap["classification_delta_balanced_accuracy"]["campaigns_better"] == 6
        and ligand_class["macro_f1"] > frozen_class["macro_f1"]
        and ligand_class["potent_predicted_safe_rate"] <= 0.05
        and ligand_class["potent_predicted_safe_rate"]
        <= frozen_class["potent_predicted_safe_rate"] + 0.02
        and ligand_bootstrap["regression_delta_mae"]["ci95"][1] < 0
        and ligand_bootstrap["regression_delta_mae"]["campaigns_better"] >= 5
    )
    return {
        "matched_ligand_comparator": matched_comparator,
        "best_ligand_comparator": best_ligand_comparator,
        "best_receptor_classification_surface_posthoc": best_class,
        "best_receptor_regression_surface_posthoc": best_regression,
        "ligand_recalibration_cross_campaign_gate_passed": ligand_recalibration_gate,
        "ligand_recalibration_research_preview_supported": ligand_recalibration_gate,
        "ligand_recalibration_production_promotion_supported": False,
        "receptor_classification_promotion_gate_passed": class_gate,
        "receptor_regression_promotion_gate_passed": regression_gate,
        "general_receptor_prediction_promotion_supported": bool(class_gate and regression_gate),
        "rule": (
            "receptor classification: beat the best ligand-only surface with scaffold-bootstrap lower "
            "95% macro-campaign BA delta >0, better in >=4/6 campaigns, macro-F1 improves, "
            "Potent-to-Safe <=0.05 and <=best ligand+0.02; receptor regression: beat the best ligand-only "
            "surface with scaffold-bootstrap upper 95% macro-campaign MAE delta <0 and better in >=4/6 "
            "campaigns; ligand recalibration preview: beat frozen router in all six BA campaigns and >=5/6 "
            "MAE campaigns with both bootstrap intervals excluding zero and safety constraints retained"
        ),
        "claim_boundary": (
            "the best surfaces are selected after all six labels were open; gates control website promotion "
            "but do not convert this exploratory nested benchmark into prospective confirmation"
        ),
    }


def _model_card(
    output: Path,
    aggregate: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    comparator = decision["best_ligand_comparator"]
    class_surface = decision["best_receptor_classification_surface_posthoc"]
    regression_surface = decision["best_receptor_regression_surface_posthoc"]
    comparator_class = aggregate[comparator]["classification"]["macro_by_campaign"]
    candidate_class = aggregate[class_surface]["classification"]["macro_by_campaign"]
    comparator_reg = aggregate[comparator]["regression"]["macro_by_campaign"]
    candidate_reg = aggregate[regression_surface]["regression"]["macro_by_campaign"]
    content = f"""# hERG V14 cross-campaign receptor fusion model card

## Status

Exploratory research bundle. General receptor-aware prediction promotion: **{str(decision['general_receptor_prediction_promotion_supported']).lower()}**.

## Evaluation contract

- 1,224 labeled and docked compounds from six V13 campaigns.
- Entire-campaign outer holdout; model and regularization selected only by leave-one-campaign-out validation inside the remaining five campaigns.
- Zero pairwise exact-connectivity and zero pairwise scaffold overlap between campaigns after structure-derived re-audit.
- All six outcome sets were already open before V14, so this is a retrospective exploratory robustness benchmark, not prospective confirmation.

## Receptor representation

The bundle uses the deposited 8ZYO receptor, frozen AutoDock Vina poses, all nine pose modes, pose-weighted contacts, Y652/F656 cation/aromatic-cage geometry, T623/S624/S649 polar or ligand-specific contacts, and the frozen V13 receptor probability surface. Vina values are scoring-function observables, not binding free energies.

Important nomenclature correction: the historical column `dock__8ZYO__contact_649_count` and model name `frozen_f649` refer to **S649** in the deposited structure, not F649. Frozen names remain unchanged only for artifact compatibility.

## Macro-by-campaign results

| Surface | Balanced accuracy | Macro-F1 | Potent→Safe | pIC50 MAE |
|---|---:|---:|---:|---:|
| Matched ligand comparator | {comparator_class['balanced_accuracy']:.4f} | {comparator_class['macro_f1']:.4f} | {comparator_class['potent_predicted_safe_rate']:.4f} | {comparator_reg['mae']:.4f} |
| Best receptor classification ({class_surface}) | {candidate_class['balanced_accuracy']:.4f} | {candidate_class['macro_f1']:.4f} | {candidate_class['potent_predicted_safe_rate']:.4f} | — |
| Best receptor regression ({regression_surface}) | — | — | — | {candidate_reg['mae']:.4f} |

## Integration tiers

1. **Website-integrable now:** ligand-only frozen production remains authoritative; V14 ligand recalibration may be exposed only as an explicitly labeled research preview until a new locked panel confirms it. Cross-campaign preview gate passed: **{str(decision['ligand_recalibration_cross_campaign_gate_passed']).lower()}**.
2. **Conducted research:** exact V14 receptor pipelines and bundles are runnable and versioned here. They may power explanations, pose diagnostics, and internal research comparisons.
3. **Not promoted:** receptor-driven potency or class overrides unless both prespecified V14 gates pass and a new locked prospective panel confirms the chosen surface.
4. **Theoretical next step:** experimentally anchored active/inactive-state ensembles, membrane-aware refinement, and prospective assay data; more rigid-receptor Vina features alone are not assumed to solve the observed campaign interaction.
"""
    path = output / "MODEL_CARD.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _benchmark(output: Path, bootstrap_replicates: int) -> dict[str, Any]:
    frame = pd.read_parquet(output / "data/harmonized_campaigns.parquet")
    pose = pd.read_parquet(output / "data/pose_ensemble_features.parquet")
    matrix = _build_model_matrix(frame, pose)
    surfaces = _feature_surfaces(matrix)
    predictions, tuning = _nested_loco(matrix, surfaces)
    class_metrics, regression_metrics, aggregate = _score_predictions(predictions)
    prediction_path = output / "predictions/nested_loco_predictions.parquet"
    tuning_path = output / "evidence/nested_tuning_evidence.parquet"
    class_path = output / "evidence/classification_metrics_by_campaign.parquet"
    regression_path = output / "evidence/regression_metrics_by_campaign.parquet"
    association_path = output / "evidence/receptor_association_audit.parquet"
    _parquet(prediction_path, predictions)
    _parquet(tuning_path, tuning)
    _parquet(class_path, class_metrics)
    _parquet(regression_path, regression_metrics)
    _parquet(association_path, _association_audit(matrix))
    matched_comparator = "ligand_calibrated_with_physchem"
    best_ligand_comparator = "ligand_calibrated"
    comparisons = [
        "ligand_calibrated",
        *[surface for surface in surfaces if surface.startswith("ligand_plus_")],
    ]
    bootstraps: dict[str, Any] = {}
    for surface in comparisons:
        if surface == best_ligand_comparator:
            bootstraps[surface] = _paired_bootstrap(
                predictions,
                surface,
                "frozen_ligand_router",
                bootstrap_replicates,
            )
        else:
            bootstraps[surface] = {
                "vs_matched_physchem": _paired_bootstrap(
                    predictions,
                    surface,
                    matched_comparator,
                    bootstrap_replicates,
                ),
                "vs_best_ligand": _paired_bootstrap(
                    predictions,
                    surface,
                    best_ligand_comparator,
                    bootstrap_replicates,
                ),
            }
    bootstrap_report = _write_json(
        output / "evidence/paired_scaffold_bootstrap.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "comparisons": bootstraps,
        },
        "report_sha256",
    )
    decision = _promotion_decision(aggregate, bootstraps)
    models = _fit_full_models(matrix, surfaces, output)
    aggregate_report = _write_json(
        output / "evidence/aggregate_metrics.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "metrics": aggregate,
        },
        "report_sha256",
    )
    _model_card(output, aggregate, decision)
    report = _write_json(
        output / "analysis_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete_exploratory_nested_cross_campaign_benchmark",
            "dataset": {
                "campaigns": int(matrix.campaign.nunique()),
                "rows": len(matrix),
                "unique_connectivity_keys": int(matrix.cross_campaign_connectivity_key.nunique()),
                "unique_scaffolds": int(matrix.cross_campaign_scaffold_key.nunique()),
                "external_rows": int(matrix.campaign_kind.str.startswith("external").sum()),
            },
            "evaluation_contract": {
                "outer_split": "leave one entire campaign out",
                "inner_model_selection": "leave one campaign out within the five outer-training campaigns",
                "training_weights": "equal campaign and observed class weight for classification; equal campaign weight for regression",
                "cross_campaign_exact_overlap": 0,
                "cross_campaign_scaffold_overlap": 0,
                "hyperparameters_selected_without_outer_campaign_labels": True,
                "best_surface_selected_after_outer_scoring": True,
            },
            "aggregate_metrics_report_sha256": aggregate_report["report_sha256"],
            "bootstrap_report_sha256": bootstrap_report["report_sha256"],
            "promotion_decision": decision,
            "full_research_models": models,
            "nomenclature_correction": {
                "historical_name": "F649 / frozen_f649 / dock__8ZYO__contact_649_count",
                "deposited_structure_residue": "S649",
                "true_aromatic_cage_features_in_v14": ["Y652", "F656"],
                "frozen_column_names_changed": False,
                "reason_names_retained": "artifact compatibility and exact V13 reproducibility",
            },
            "scientific_scope": {
                "analysis_is_exploratory_after_labels_opened": True,
                "external_or_prospective_confirmation": False,
                "receptor_aware": True,
                "all_nine_vina_poses_used": True,
                "new_docking_performed": False,
                "vina_scores_are_binding_free_energies": False,
                "website_default_changed": False,
            },
        },
        "report_sha256",
    )
    artifacts = [
        output / "prepare_report.json",
        output / "data/harmonized_campaigns.parquet",
        output / "data/pose_ensemble_features.parquet",
        output / "evidence/cross_campaign_overlap_audit.parquet",
        output / "evidence/campaign_census.parquet",
        prediction_path,
        tuning_path,
        class_path,
        regression_path,
        association_path,
        output / "evidence/paired_scaffold_bootstrap.json",
        output / "evidence/aggregate_metrics.json",
        output / "MODEL_CARD.md",
        output / "analysis_report.json",
        *[Path(item["path"]) for item in models.values()],
    ]
    manifest = _write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": "complete",
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha(Path(__file__).resolve()),
            },
            "artifacts": [
                {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}
                for path in artifacts
            ],
        },
        "manifest_sha256",
    )
    return _write_json(
        output / "final_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "finished_utc": _utc(),
            "status": "complete",
            "campaigns": int(matrix.campaign.nunique()),
            "rows": len(matrix),
            "general_receptor_prediction_promotion_supported": decision[
                "general_receptor_prediction_promotion_supported"
            ],
            "best_receptor_classification_surface_posthoc": decision[
                "best_receptor_classification_surface_posthoc"
            ],
            "best_receptor_regression_surface_posthoc": decision[
                "best_receptor_regression_surface_posthoc"
            ],
            "report_sha256": report["report_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "summary_sha256",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--router-oof", type=Path, default=DEFAULT_ROUTER)
    parser.add_argument("--prepared-receptor", type=Path, default=DEFAULT_RECEPTOR)
    parser.add_argument("--stage", choices=("prepare", "benchmark", "all"), default="all")
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    return parser


def _main(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    output = _resolve(repo, args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    router_path = _resolve(repo, args.router_oof)
    receptor_path = _resolve(repo, args.prepared_receptor)
    if args.stage in {"prepare", "all"}:
        prepare = _prepare(repo, output, router_path, receptor_path)
    else:
        prepare = json.loads((output / "prepare_report.json").read_text())
    if args.stage in {"benchmark", "all"}:
        return _benchmark(output, args.bootstrap_replicates)
    return prepare


def main() -> int:
    try:
        result = _main(_parser().parse_args())
    except CampaignError as exc:
        print(f"V14 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
