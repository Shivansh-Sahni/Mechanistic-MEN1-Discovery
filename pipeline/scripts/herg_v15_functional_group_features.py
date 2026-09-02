#!/usr/bin/env python3
"""Governed functional-group features for the retrospective hERG V15 study."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, Crippen, Descriptors, Lipinski, rdMolDescriptors

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "pipeline/config/herg_v15_functional_group_registry_v1.json"

PHYSICOCHEMICAL_FEATURES = (
    "molecular_weight",
    "tpsa",
    "clogp",
    "hbd",
    "hba",
    "formal_charge",
    "rotatable_bonds",
    "heavy_atom_count",
)
CHARGE_CONTEXT_FEATURES = (
    "positive_ionizable_centers",
    "negative_ionizable_centers",
    "permanent_positive_atoms",
    "permanent_negative_atoms",
    "zwitterion_indicator",
    "basic_aromatic_min_topological_distance",
)
RING_LINKER_FEATURES = (
    "ring_count",
    "aromatic_ring_count",
    "aliphatic_ring_count",
    "ring_system_count",
    "ring_exocyclic_attachment_bonds",
    "minimum_inter_ring_linker_atoms",
)
INTERACTION_FEATURES = (
    "clogp_x_positive_ionizable",
    "tpsa_x_positive_ionizable",
    "mw_x_abs_formal_charge",
    "clogp_x_formal_charge",
    "clogp_x_aromatic_ring_count",
    "tpsa_x_hba",
)
_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
)


class FunctionalGroupFeatureError(ValueError):
    """Raised when the governed registry or a structure is invalid."""


def _canonical_hash(value: dict[str, Any], self_key: str) -> str:
    candidate = dict(value)
    candidate.pop(self_key, None)
    raw = (
        json.dumps(candidate, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Load, integrity-check, and compile-check the versioned SMARTS registry."""
    value = json.loads(Path(path).read_text())
    expected = value.get("registry_sha256")
    if not isinstance(expected, str) or expected != _canonical_hash(value, "registry_sha256"):
        raise FunctionalGroupFeatureError("functional-group registry hash mismatch")
    rules = value.get("rules")
    if not isinstance(rules, list) or not rules:
        raise FunctionalGroupFeatureError("functional-group registry has no rules")
    keys = [str(rule.get("key", "")) for rule in rules]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise FunctionalGroupFeatureError("functional-group registry keys are empty or duplicated")
    for rule in rules:
        if Chem.MolFromSmarts(str(rule.get("smarts", ""))) is None:
            raise FunctionalGroupFeatureError(f"invalid SMARTS for {rule['key']}")
    return value


def _ring_systems(molecule: Chem.Mol) -> list[set[int]]:
    systems: list[set[int]] = []
    for ring in molecule.GetRingInfo().AtomRings():
        current = set(ring)
        merged = []
        for index, system in enumerate(systems):
            if current & system:
                current |= system
                merged.append(index)
        for index in reversed(merged):
            systems.pop(index)
        systems.append(current)
    return systems


def _minimum_inter_ring_linker_atoms(
    molecule: Chem.Mol,
    systems: list[set[int]],
) -> int:
    if len(systems) < 2:
        return 0
    ring_atoms = set().union(*systems)
    minimum: int | None = None
    for left_index, left in enumerate(systems):
        for right in systems[left_index + 1 :]:
            for left_atom in left:
                for right_atom in right:
                    path = Chem.GetShortestPath(molecule, left_atom, right_atom)
                    linker_atoms = sum(atom not in ring_atoms for atom in path[1:-1])
                    if minimum is None or linker_atoms < minimum:
                        minimum = linker_atoms
    return int(minimum or 0)


def _basic_aromatic_distance(
    molecule: Chem.Mol,
    positive_features: list[Any],
) -> int:
    basic_atoms = sorted({int(atom) for feature in positive_features for atom in feature.GetAtomIds()})
    aromatic_atoms = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetIsAromatic()]
    # Thirteen is an explicit capped-missing sentinel; the companion center and
    # aromatic-ring counts distinguish absence from a long graph distance.
    if not basic_atoms or not aromatic_atoms:
        return 13
    distances = []
    for basic_atom in basic_atoms:
        for aromatic_atom in aromatic_atoms:
            if basic_atom == aromatic_atom:
                distances.append(0)
                continue
            path = Chem.GetShortestPath(molecule, basic_atom, aromatic_atom)
            # Disconnected counterions have no graph path and should not create
            # an artificial within-molecule basic/aromatic proximity.
            if path:
                distances.append(len(path) - 1)
    return int(min(min(distances), 12)) if distances else 13


def feature_columns(
    registry: dict[str, Any],
    *,
    include_functional_groups: bool = True,
    include_interactions: bool = True,
) -> list[str]:
    columns = [*PHYSICOCHEMICAL_FEATURES, *CHARGE_CONTEXT_FEATURES, *RING_LINKER_FEATURES]
    if include_functional_groups:
        for rule in registry["rules"]:
            columns.extend((f"fg_count__{rule['key']}", f"fg_present__{rule['key']}"))
    if include_interactions:
        columns.extend(INTERACTION_FEATURES)
    return columns


def build_feature_row(
    smiles: str,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic descriptor, functional-group, and interaction features."""
    registry = registry or load_registry()
    normalized_smiles = str(smiles).strip()
    if not normalized_smiles:
        raise FunctionalGroupFeatureError("SMILES is empty")
    molecule = Chem.MolFromSmiles(normalized_smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise FunctionalGroupFeatureError("SMILES could not be parsed")
    canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
    chemical_features = _FEATURE_FACTORY.GetFeaturesForMol(molecule)
    positive = [feature for feature in chemical_features if feature.GetFamily() == "PosIonizable"]
    negative = [feature for feature in chemical_features if feature.GetFamily() == "NegIonizable"]
    permanent_positive = sum(atom.GetFormalCharge() > 0 for atom in molecule.GetAtoms())
    permanent_negative = sum(atom.GetFormalCharge() < 0 for atom in molecule.GetAtoms())
    formal_charge = int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()))
    systems = _ring_systems(molecule)
    ring_atoms = set().union(*systems) if systems else set()

    result: dict[str, Any] = {
        "canonical_smiles": canonical,
        "molecular_weight": float(Descriptors.MolWt(molecule)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
        "clogp": float(Crippen.MolLogP(molecule)),
        "hbd": int(Lipinski.NumHDonors(molecule)),
        "hba": int(Lipinski.NumHAcceptors(molecule)),
        "formal_charge": formal_charge,
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
        "heavy_atom_count": int(molecule.GetNumHeavyAtoms()),
        "positive_ionizable_centers": int(len(positive)),
        "negative_ionizable_centers": int(len(negative)),
        "permanent_positive_atoms": int(permanent_positive),
        "permanent_negative_atoms": int(permanent_negative),
        "zwitterion_indicator": int(permanent_positive > 0 and permanent_negative > 0),
        "basic_aromatic_min_topological_distance": _basic_aromatic_distance(molecule, positive),
        "ring_count": int(rdMolDescriptors.CalcNumRings(molecule)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "aliphatic_ring_count": int(rdMolDescriptors.CalcNumAliphaticRings(molecule)),
        "ring_system_count": int(len(systems)),
        "ring_exocyclic_attachment_bonds": int(
            sum(
                (bond.GetBeginAtomIdx() in ring_atoms)
                != (bond.GetEndAtomIdx() in ring_atoms)
                for bond in molecule.GetBonds()
            )
        ),
        "minimum_inter_ring_linker_atoms": _minimum_inter_ring_linker_atoms(molecule, systems),
    }
    for rule in registry["rules"]:
        pattern = Chem.MolFromSmarts(rule["smarts"])
        matches = molecule.GetSubstructMatches(pattern, uniquify=True)
        count = int(len(matches))
        result[f"fg_count__{rule['key']}"] = count
        result[f"fg_present__{rule['key']}"] = int(count > 0)

    result.update(
        {
            "clogp_x_positive_ionizable": result["clogp"]
            * result["positive_ionizable_centers"],
            "tpsa_x_positive_ionizable": result["tpsa"]
            * result["positive_ionizable_centers"],
            "mw_x_abs_formal_charge": result["molecular_weight"] * abs(formal_charge),
            "clogp_x_formal_charge": result["clogp"] * formal_charge,
            "clogp_x_aromatic_ring_count": result["clogp"] * result["aromatic_ring_count"],
            "tpsa_x_hba": result["tpsa"] * result["hba"],
        }
    )
    numeric = np.asarray([result[column] for column in feature_columns(registry)], dtype=float)
    if not np.isfinite(numeric).all():
        raise FunctionalGroupFeatureError("functional-group feature row contains non-finite values")
    return result


def build_feature_frame(
    smiles_values: Iterable[str],
    registry: dict[str, Any] | None = None,
) -> pd.DataFrame:
    registry = registry or load_registry()
    rows = [build_feature_row(smiles, registry) for smiles in smiles_values]
    frame = pd.DataFrame(rows)
    expected = ["canonical_smiles", *feature_columns(registry)]
    return frame[expected]
