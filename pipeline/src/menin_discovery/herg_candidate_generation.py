"""Bounded, traceable candidate enumeration for hERG structure-edit research.

This module deliberately does not predict hERG, synthesis feasibility, ADMET, or
biological activity.  It applies a small governed registry of one-step RDKit
reaction SMARTS, records exact lineage and failures, and offers a transparent
ranking contract for predictions produced elsewhere.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

try:  # pragma: no cover - project environments normally provide RDKit.
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import Descriptors, rdChemReactions, rdFingerprintGenerator

    _RDKIT_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    DataStructs = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    rdFingerprintGenerator = None  # type: ignore[assignment]
    rdChemReactions = None  # type: ignore[assignment]
    rdBase = None  # type: ignore[assignment]
    _RDKIT_IMPORT_ERROR = str(exc)


REGISTRY_VERSION = "herg-bounded-medicinal-edits-v1"
REGISTRY_PROVENANCE = (
    "Manually governed project SMARTS registry for bounded analogue enumeration; "
    "not a reaction database, retrosynthesis method, or synthesis-feasibility claim."
)


@dataclass(frozen=True)
class TransformationSpec:
    """One governed, single-reactant/single-product enumerative transformation."""

    key: str
    name: str
    reaction_smarts: str
    category: str
    provenance: str = REGISTRY_PROVENANCE


# The registry intentionally favors local substitutions and common functional-group
# interconversions. Products still pass sanitization and explicit small-edit bounds.
DEFAULT_TRANSFORMATIONS: tuple[TransformationSpec, ...] = (
    TransformationSpec("aryl_h_to_f", "Aromatic H → F", "[cH:1]>>[c:1]F", "aryl_substitution"),
    TransformationSpec("aryl_h_to_cl", "Aromatic H → Cl", "[cH:1]>>[c:1]Cl", "aryl_substitution"),
    TransformationSpec("aryl_h_to_br", "Aromatic H → Br", "[cH:1]>>[c:1]Br", "aryl_substitution"),
    TransformationSpec("aryl_h_to_me", "Aromatic H → methyl", "[cH:1]>>[c:1]C", "aryl_substitution"),
    TransformationSpec("aryl_h_to_et", "Aromatic H → ethyl", "[cH:1]>>[c:1]CC", "aryl_substitution"),
    TransformationSpec("aryl_h_to_oh", "Aromatic H → hydroxyl", "[cH:1]>>[c:1]O", "aryl_substitution"),
    TransformationSpec("aryl_h_to_ome", "Aromatic H → methoxy", "[cH:1]>>[c:1]OC", "aryl_substitution"),
    TransformationSpec("aryl_h_to_oet", "Aromatic H → ethoxy", "[cH:1]>>[c:1]OCC", "aryl_substitution"),
    TransformationSpec("aryl_h_to_nh2", "Aromatic H → amino", "[cH:1]>>[c:1]N", "aryl_substitution"),
    TransformationSpec(
        "aryl_h_to_nme2",
        "Aromatic H → dimethylamino",
        "[cH:1]>>[c:1]N(C)C",
        "aryl_substitution",
    ),
    TransformationSpec("aryl_h_to_cn", "Aromatic H → nitrile", "[cH:1]>>[c:1]C#N", "aryl_substitution"),
    TransformationSpec(
        "aryl_h_to_conh2",
        "Aromatic H → primary amide",
        "[cH:1]>>[c:1]C(=O)N",
        "aryl_substitution",
    ),
    TransformationSpec(
        "aryl_h_to_conhme",
        "Aromatic H → N-methyl amide",
        "[cH:1]>>[c:1]C(=O)NC",
        "aryl_substitution",
    ),
    TransformationSpec(
        "aryl_h_to_co2h",
        "Aromatic H → carboxylic acid",
        "[cH:1]>>[c:1]C(=O)O",
        "aryl_substitution",
    ),
    TransformationSpec("aryl_h_to_cf3", "Aromatic H → CF3", "[cH:1]>>[c:1]C(F)(F)F", "aryl_substitution"),
    TransformationSpec("aryl_h_to_chf2", "Aromatic H → CHF2", "[cH:1]>>[c:1]C(F)F", "aryl_substitution"),
    TransformationSpec(
        "aryl_h_to_ch2oh",
        "Aromatic H → hydroxymethyl",
        "[cH:1]>>[c:1]CO",
        "aryl_substitution",
    ),
    TransformationSpec(
        "aryl_h_to_nhcome",
        "Aromatic H → acetamide",
        "[cH:1]>>[c:1]NC(=O)C",
        "aryl_substitution",
    ),
    TransformationSpec(
        "aryl_h_to_so2me",
        "Aromatic H → methyl sulfone",
        "[cH:1]>>[c:1]S(=O)(=O)C",
        "aryl_substitution",
    ),
    TransformationSpec(
        "aryl_h_to_morpholine",
        "Aromatic H → morpholinyl",
        "[cH:1]>>[c:1]N1CCOCC1",
        "aryl_substitution",
    ),
    TransformationSpec(
        "aryl_h_to_so2nh2",
        "Aromatic H → sulfonamide",
        "[cH:1]>>[c:1]S(=O)(=O)N",
        "aryl_substitution",
    ),
    TransformationSpec("aryl_ch_to_n", "Aromatic CH → N", "[cH:1]>>[n:1]", "heteroatom_swap"),
    TransformationSpec("aryl_n_to_ch", "Aromatic N → CH", "[n;+0:1]>>[cH:1]", "heteroatom_swap"),
    TransformationSpec(
        "aliphatic_ch_to_f",
        "Aliphatic C–H → C–F",
        "[C;H1,H2,H3;+0:1]>>[C:1]F",
        "aliphatic_substitution",
    ),
    TransformationSpec(
        "aliphatic_ch_to_oh",
        "Aliphatic C–H → C–OH",
        "[C;H1,H2,H3;+0:1]>>[C:1]O",
        "aliphatic_substitution",
    ),
    TransformationSpec(
        "aliphatic_ch_to_me",
        "Aliphatic C–H → C–methyl",
        "[C;H1,H2,H3;+0:1]>>[C:1]C",
        "aliphatic_substitution",
    ),
    TransformationSpec(
        "amine_n_methylation",
        "Non-amide N–H → N-methyl",
        "[N;H1,H2;+0;!$(N-C=O):1]>>[N:1]C",
        "nitrogen_edit",
    ),
    TransformationSpec(
        "amine_n_hydroxyethylation",
        "Non-amide N–H → N-hydroxyethyl",
        "[N;H1,H2;+0;!$(N-C=O):1]>>[N:1]CCO",
        "nitrogen_edit",
    ),
    TransformationSpec(
        "amine_n_acetylation",
        "Amine N–H → acetamide",
        "[N;H1,H2;+0;!$(N-C=O):1]>>[N:1]C(=O)C",
        "basicity_edit",
    ),
    TransformationSpec(
        "amide_n_methylation",
        "Amide N–H → N-methyl amide",
        "[N;H1;+0;$(N-C=O):1]>>[N:1]C",
        "nitrogen_edit",
    ),
    TransformationSpec("oh_to_ome", "O–H → O-methyl", "[O;H1;+0:1]>>[O:1]C", "oxygen_edit"),
    TransformationSpec("oh_to_oet", "O–H → O-ethyl", "[O;H1;+0:1]>>[O:1]CC", "oxygen_edit"),
    TransformationSpec(
        "alcohol_to_carbonyl",
        "Alcohol → carbonyl",
        "[C;H1,H2:1][O;H1:2]>>[C:1]=[O:2]",
        "oxygen_edit",
    ),
    TransformationSpec(
        "aryl_cl_to_f",
        "Aryl chloride → fluoride",
        "[c:1]Cl>>[c:1]F",
        "halogen_edit",
    ),
    TransformationSpec(
        "aryl_br_to_f",
        "Aryl bromide → fluoride",
        "[c:1]Br>>[c:1]F",
        "halogen_edit",
    ),
    TransformationSpec(
        "aryl_halogen_to_me",
        "Aryl halogen → methyl",
        "[c:1][F,Cl,Br,I]>>[c:1]C",
        "halogen_edit",
    ),
    TransformationSpec(
        "aryl_halogen_to_cn",
        "Aryl halogen → nitrile",
        "[c:1][F,Cl,Br,I]>>[c:1]C#N",
        "halogen_edit",
    ),
    TransformationSpec(
        "aryl_halogen_to_ome",
        "Aryl halogen → methoxy",
        "[c:1][F,Cl,Br,I]>>[c:1]OC",
        "halogen_edit",
    ),
    TransformationSpec(
        "acid_to_methyl_ester",
        "Carboxylic acid → methyl ester",
        "[C:1](=[O:2])[O;H1:3]>>[C:1](=[O:2])[O:3]C",
        "carbonyl_edit",
    ),
    TransformationSpec(
        "acid_to_primary_amide",
        "Carboxylic acid → primary amide",
        "[C:1](=[O:2])[O;H1]>>[C:1](=[O:2])N",
        "carbonyl_edit",
    ),
    TransformationSpec(
        "ester_to_primary_amide",
        "Ester → primary amide",
        "[C:1](=[O:2])[O][C]>>[C:1](=[O:2])N",
        "carbonyl_edit",
    ),
    TransformationSpec(
        "nitrile_to_primary_amide",
        "Nitrile → primary amide",
        "[C:1]#[N:2]>>[C:1](=O)[N:2]",
        "carbonyl_edit",
    ),
    TransformationSpec("carbonyl_o_to_s", "Carbonyl O → S", "[C:1]=O>>[C:1]=S", "heteroatom_swap"),
    TransformationSpec("thiocarbonyl_s_to_o", "Thiocarbonyl S → O", "[C:1]=S>>[C:1]=O", "heteroatom_swap"),
    TransformationSpec(
        "secondary_amine_to_ether",
        "Secondary amine linker → ether",
        "[C:1][NH;+0:2][C:3]>>[C:1][O:2][C:3]",
        "basicity_edit",
    ),
    TransformationSpec(
        "ether_to_secondary_amine",
        "Ether linker → secondary amine",
        "[C:1][O;+0:2][C:3]>>[C:1][NH:2][C:3]",
        "basicity_edit",
    ),
    TransformationSpec(
        "n_demethylation",
        "Tertiary amine N-demethylation",
        "[N;X3;+0:1][CH3]>>[N:1]",
        "basicity_edit",
    ),
)


@dataclass(frozen=True)
class GenerationBounds:
    """Hard limits for a bounded generation run."""

    target_count: int = 100
    seed: int = 20260831
    max_products_per_transformation: int = 256
    max_total_raw_products: int = 8_000
    max_failure_records: int = 1_000
    max_parent_heavy_atoms: int = 180
    max_abs_heavy_atom_delta: int = 8
    max_abs_molecular_weight_delta_da: float = 120.0
    max_abs_formal_charge_delta: int = 1
    minimum_parent_tanimoto: float = 0.45

    def __post_init__(self) -> None:
        if not 1 <= self.target_count <= 1_000:
            raise ValueError("target_count must be between 1 and 1,000")
        if self.max_products_per_transformation < 1 or self.max_total_raw_products < 1:
            raise ValueError("product bounds must be positive")
        if not 0.0 <= self.minimum_parent_tanimoto <= 1.0:
            raise ValueError("minimum_parent_tanimoto must be in [0, 1]")


@dataclass(frozen=True)
class TransformationLineage:
    transformation_key: str
    transformation_name: str
    reaction_smarts: str
    category: str
    provenance: str
    registry_version: str
    reaction_product_index: int


@dataclass(frozen=True)
class GeneratedCandidate:
    candidate_id: str
    canonical_smiles: str
    parent_canonical_smiles: str
    lineages: tuple[TransformationLineage, ...]
    parent_morgan_tanimoto: float
    molecular_weight_da: float
    molecular_weight_delta_da: float
    heavy_atom_count: int
    heavy_atom_delta: int
    formal_charge: int
    formal_charge_delta: int
    interpretation_boundary: str = (
        "Enumerated structure only; not a synthesis-feasibility, hERG, activity, or ADMET claim."
    )


@dataclass(frozen=True)
class GenerationFailure:
    transformation_key: str
    transformation_name: str
    stage: str
    reason: str
    detail: str = ""
    reaction_product_index: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    status: str
    parent_input_smiles: str
    parent_canonical_smiles: str
    registry_version: str
    rdkit_version: str
    bounds: GenerationBounds
    candidates: tuple[GeneratedCandidate, ...]
    failures: tuple[GenerationFailure, ...]
    raw_product_count: int
    valid_unique_before_selection: int
    suppressed_failure_count: int
    truncated_to_target: bool
    claim_boundary: str = (
        "Candidates are deterministic, bounded RDKit enumerations. No synthesis feasibility, "
        "biological activity, hERG improvement, or developability is asserted."
    )


def rdkit_generation_available() -> bool:
    return Chem is not None and rdChemReactions is not None


@lru_cache(maxsize=256)
def _compiled_reaction(smarts: str) -> Any:
    if rdChemReactions is None:  # pragma: no cover
        raise RuntimeError(f"RDKit reaction support is unavailable: {_RDKIT_IMPORT_ERROR}")
    reaction = rdChemReactions.ReactionFromSmarts(smarts)
    if reaction is None:
        raise ValueError(f"RDKit could not parse reaction SMARTS: {smarts}")
    reaction.Initialize()
    if reaction.GetNumReactantTemplates() != 1 or reaction.GetNumProductTemplates() != 1:
        raise ValueError("Candidate transformations must have one reactant and one product template")
    return reaction


def validate_transformation_registry(
    registry: Sequence[TransformationSpec] = DEFAULT_TRANSFORMATIONS,
) -> tuple[str, ...]:
    """Compile the complete registry and return any validation errors."""

    errors: list[str] = []
    keys: set[str] = set()
    for spec in registry:
        if spec.key in keys:
            errors.append(f"duplicate transformation key: {spec.key}")
        keys.add(spec.key)
        try:
            _compiled_reaction(spec.reaction_smarts)
        except Exception as exc:  # pragma: no cover - explicit error contract.
            errors.append(f"{spec.key}: {type(exc).__name__}: {exc}")
    return tuple(errors)


def _canonical_product(product: Any) -> tuple[Any, str]:
    molecule = Chem.Mol(product)
    Chem.SanitizeMol(molecule)
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("disconnected product")
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    reparsed = Chem.MolFromSmiles(smiles)
    if reparsed is None:
        raise ValueError("canonical product could not be reparsed")
    return reparsed, Chem.MolToSmiles(reparsed, canonical=True, isomericSmiles=True)


def _formal_charge(molecule: Any) -> int:
    return int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()))


def _candidate_id(smiles: str) -> str:
    digest = hashlib.sha256(f"{REGISTRY_VERSION}\0{smiles}".encode()).hexdigest()
    return f"HERG-EDIT-{digest[:16].upper()}"


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _round_robin_select(
    candidates: Sequence[GeneratedCandidate],
    target_count: int,
    seed: int,
) -> tuple[GeneratedCandidate, ...]:
    groups: dict[str, list[GeneratedCandidate]] = {}
    for candidate in candidates:
        primary = candidate.lineages[0].transformation_key
        groups.setdefault(primary, []).append(candidate)
    for _key, values in groups.items():
        values.sort(
            key=lambda candidate: (
                abs(candidate.heavy_atom_delta),
                abs(candidate.molecular_weight_delta_da),
                -candidate.parent_morgan_tanimoto,
                _stable_key(seed, candidate.canonical_smiles),
            )
        )
    group_order = sorted(groups, key=lambda key: _stable_key(seed, key))
    selected: list[GeneratedCandidate] = []
    offset = 0
    while len(selected) < target_count:
        added = False
        for key in group_order:
            values = groups[key]
            if offset < len(values):
                selected.append(values[offset])
                added = True
                if len(selected) == target_count:
                    break
        if not added:
            break
        offset += 1
    return tuple(selected)


def generate_herg_edit_candidates(
    parent_smiles: str,
    *,
    bounds: GenerationBounds | None = None,
    registry: Sequence[TransformationSpec] = DEFAULT_TRANSFORMATIONS,
) -> GenerationResult:
    """Enumerate bounded one-step analogues and retain complete transformation lineage."""

    config = bounds or GenerationBounds()
    parent_input = str(parent_smiles or "").strip()
    failures: list[GenerationFailure] = []
    suppressed_failures = 0

    def record_failure(failure: GenerationFailure) -> None:
        nonlocal suppressed_failures
        if len(failures) < config.max_failure_records:
            failures.append(failure)
        else:
            suppressed_failures += 1

    if not rdkit_generation_available():  # pragma: no cover
        record_failure(
            GenerationFailure("registry", "RDKit availability", "setup", "rdkit_unavailable", _RDKIT_IMPORT_ERROR)
        )
        return GenerationResult(
            "rdkit_unavailable", parent_input, "", REGISTRY_VERSION, "", config, (),
            tuple(failures), 0, 0, suppressed_failures, False,
        )
    parent = Chem.MolFromSmiles(parent_input)
    if parent is None:
        record_failure(
            GenerationFailure("parent", "Parent validation", "parent", "invalid_parent_smiles")
        )
        return GenerationResult(
            "invalid_parent", parent_input, "", REGISTRY_VERSION, str(rdBase.rdkitVersion), config,
            (), tuple(failures), 0, 0, suppressed_failures, False,
        )
    Chem.SanitizeMol(parent)
    parent_canonical = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    parent = Chem.MolFromSmiles(parent_canonical)
    parent_heavy = int(parent.GetNumHeavyAtoms())
    if parent_heavy > config.max_parent_heavy_atoms:
        record_failure(
            GenerationFailure(
                "parent",
                "Parent generation bound",
                "parent",
                "parent_exceeds_heavy_atom_bound",
                f"{parent_heavy} > {config.max_parent_heavy_atoms}",
            )
        )
        return GenerationResult(
            "parent_outside_generation_bounds", parent_input, parent_canonical, REGISTRY_VERSION,
            str(rdBase.rdkitVersion), config, (), tuple(failures), 0, 0,
            suppressed_failures, False,
        )

    registry_errors = validate_transformation_registry(registry)
    if registry_errors:
        for detail in registry_errors:
            record_failure(
                GenerationFailure("registry", "Registry validation", "registry", "invalid_registry", detail)
            )
        return GenerationResult(
            "invalid_registry", parent_input, parent_canonical, REGISTRY_VERSION,
            str(rdBase.rdkitVersion), config, (), tuple(failures), 0, 0,
            suppressed_failures, False,
        )

    parent_mw = float(Descriptors.MolWt(parent))
    parent_charge = _formal_charge(parent)
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    parent_fp = fingerprint.GetFingerprint(parent)
    records: dict[str, dict[str, Any]] = {}
    raw_product_count = 0
    global_bound_reached = False

    for spec in registry:
        reaction = _compiled_reaction(spec.reaction_smarts)
        try:
            product_sets = reaction.RunReactants(
                (parent,), maxProducts=config.max_products_per_transformation
            )
        except Exception as exc:
            record_failure(
                GenerationFailure(
                    spec.key, spec.name, "reaction", "reaction_execution_failed",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        if not product_sets:
            record_failure(GenerationFailure(spec.key, spec.name, "reaction", "no_match"))
            continue
        for product_index, product_tuple in enumerate(product_sets):
            if raw_product_count >= config.max_total_raw_products:
                global_bound_reached = True
                break
            raw_product_count += 1
            if len(product_tuple) != 1:
                record_failure(
                    GenerationFailure(
                        spec.key, spec.name, "product", "unexpected_product_count",
                        str(len(product_tuple)), product_index,
                    )
                )
                continue
            try:
                candidate_mol, candidate_smiles = _canonical_product(product_tuple[0])
            except Exception as exc:
                record_failure(
                    GenerationFailure(
                        spec.key, spec.name, "sanitization", "invalid_product",
                        f"{type(exc).__name__}: {exc}", product_index,
                    )
                )
                continue
            if candidate_smiles == parent_canonical:
                record_failure(
                    GenerationFailure(
                        spec.key, spec.name, "filter", "parent_unchanged", "", product_index
                    )
                )
                continue
            candidate_heavy = int(candidate_mol.GetNumHeavyAtoms())
            heavy_delta = candidate_heavy - parent_heavy
            candidate_mw = float(Descriptors.MolWt(candidate_mol))
            mw_delta = candidate_mw - parent_mw
            candidate_charge = _formal_charge(candidate_mol)
            charge_delta = candidate_charge - parent_charge
            candidate_fp = fingerprint.GetFingerprint(candidate_mol)
            similarity = float(DataStructs.TanimotoSimilarity(parent_fp, candidate_fp))
            rejection = ""
            detail = ""
            if abs(heavy_delta) > config.max_abs_heavy_atom_delta:
                rejection, detail = "heavy_atom_delta_exceeded", str(heavy_delta)
            elif abs(mw_delta) > config.max_abs_molecular_weight_delta_da:
                rejection, detail = "molecular_weight_delta_exceeded", f"{mw_delta:.6g} Da"
            elif abs(charge_delta) > config.max_abs_formal_charge_delta:
                rejection, detail = "formal_charge_delta_exceeded", str(charge_delta)
            elif similarity < config.minimum_parent_tanimoto:
                rejection, detail = "minimum_parent_similarity_not_met", f"{similarity:.6g}"
            if rejection:
                record_failure(
                    GenerationFailure(spec.key, spec.name, "filter", rejection, detail, product_index)
                )
                continue
            lineage = TransformationLineage(
                spec.key,
                spec.name,
                spec.reaction_smarts,
                spec.category,
                spec.provenance,
                REGISTRY_VERSION,
                product_index,
            )
            if candidate_smiles in records:
                records[candidate_smiles]["lineages"].append(lineage)
                continue
            records[candidate_smiles] = {
                "molecule": candidate_mol,
                "lineages": [lineage],
                "similarity": similarity,
                "molecular_weight": candidate_mw,
                "molecular_weight_delta": mw_delta,
                "heavy_atoms": candidate_heavy,
                "heavy_atom_delta": heavy_delta,
                "formal_charge": candidate_charge,
                "formal_charge_delta": charge_delta,
            }
        if global_bound_reached:
            record_failure(
                GenerationFailure(
                    spec.key,
                    spec.name,
                    "generation",
                    "global_raw_product_bound_reached",
                    str(config.max_total_raw_products),
                )
            )
            break

    all_candidates = tuple(
        GeneratedCandidate(
            candidate_id=_candidate_id(smiles),
            canonical_smiles=smiles,
            parent_canonical_smiles=parent_canonical,
            lineages=tuple(record["lineages"]),
            parent_morgan_tanimoto=float(record["similarity"]),
            molecular_weight_da=float(record["molecular_weight"]),
            molecular_weight_delta_da=float(record["molecular_weight_delta"]),
            heavy_atom_count=int(record["heavy_atoms"]),
            heavy_atom_delta=int(record["heavy_atom_delta"]),
            formal_charge=int(record["formal_charge"]),
            formal_charge_delta=int(record["formal_charge_delta"]),
        )
        for smiles, record in records.items()
    )
    selected = _round_robin_select(all_candidates, config.target_count, config.seed)
    if not selected:
        status = "no_candidates"
    elif len(selected) < config.target_count:
        status = "partial"
        record_failure(
            GenerationFailure(
                "generation",
                "Target count",
                "selection",
                "target_count_not_reached",
                f"generated {len(selected)} of requested {config.target_count}",
            )
        )
    else:
        status = "completed"
    return GenerationResult(
        status=status,
        parent_input_smiles=parent_input,
        parent_canonical_smiles=parent_canonical,
        registry_version=REGISTRY_VERSION,
        rdkit_version=str(rdBase.rdkitVersion),
        bounds=config,
        candidates=selected,
        failures=tuple(failures),
        raw_product_count=raw_product_count,
        valid_unique_before_selection=len(all_candidates),
        suppressed_failure_count=suppressed_failures,
        truncated_to_target=len(all_candidates) > len(selected),
    )


@dataclass(frozen=True)
class CandidatePredictionEvidence:
    """Externally computed evidence consumed by the ranking utility."""

    candidate_id: str
    canonical_smiles: str
    predicted_ic50_um: float
    interval90_lower_um: float | None
    interval90_upper_um: float | None
    applicability_label: str
    maximum_train_tanimoto: float | None
    parent_morgan_tanimoto: float | None
    property_deltas: Mapping[str, float] = field(default_factory=dict)
    model_provenance: str = ""


@dataclass(frozen=True)
class RankingConfig:
    """Transparent prioritization weights; these are not learned confidence values."""

    improvement_weight: float = 1.0
    interval_width_weight: float = 0.35
    applicability_weight: float = 0.80
    parent_similarity_weight: float = 0.40
    property_change_weight: float = 0.15
    minimum_train_tanimoto: float = 0.35
    preferred_parent_tanimoto: float = 0.55
    missing_uncertainty_penalty: float = 1.0
    missing_applicability_penalty: float = 0.75
    missing_parent_similarity_penalty: float = 0.50
    missing_property_delta_penalty: float = 0.25
    property_delta_scales: Mapping[str, float] = field(
        default_factory=lambda: {
            "molecular_weight": 60.0,
            "tpsa": 30.0,
            "clogp": 1.0,
            "formal_charge": 1.0,
            "basic_centers": 1.0,
            "rotatable_bonds": 3.0,
        }
    )

    def __post_init__(self) -> None:
        weights = (
            self.improvement_weight,
            self.interval_width_weight,
            self.applicability_weight,
            self.parent_similarity_weight,
            self.property_change_weight,
        )
        if any(value < 0 for value in weights):
            raise ValueError("ranking weights must be non-negative")
        if not 0 < self.minimum_train_tanimoto <= 1:
            raise ValueError("minimum_train_tanimoto must be in (0, 1]")
        if not 0 < self.preferred_parent_tanimoto <= 1:
            raise ValueError("preferred_parent_tanimoto must be in (0, 1]")


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    candidate_id: str
    canonical_smiles: str
    prioritization_score: float
    predicted_log10_ic50_gain: float
    interval_width_penalty: float
    applicability_penalty: float
    parent_similarity_penalty: float
    property_change_penalty: float
    evidence_notes: tuple[str, ...]
    score_boundary: str = (
        "Deterministic prioritization score, not a probability, confidence, activity claim, "
        "or synthesis recommendation."
    )


def _applicability_label_penalty(label: str) -> float:
    normalized = str(label).strip().lower()
    if "extrapolat" in normalized or "out of domain" in normalized or "outside domain" in normalized:
        return 1.0
    if "low" in normalized or "weak" in normalized:
        return 0.6
    if "moderate" in normalized or "borderline" in normalized:
        return 0.3
    return 0.0


def rank_candidate_evidence(
    parent_predicted_ic50_um: float,
    candidates: Sequence[CandidatePredictionEvidence],
    *,
    config: RankingConfig | None = None,
) -> tuple[RankedCandidate, ...]:
    """Rank supplied predictions while penalizing uncertainty and out-of-domain evidence.

    The utility never computes model confidence. Every score component is returned
    so downstream interfaces can disclose exactly why ordering changed.
    """

    settings = config or RankingConfig()
    parent_ic50 = float(parent_predicted_ic50_um)
    if not math.isfinite(parent_ic50) or parent_ic50 <= 0:
        raise ValueError("parent_predicted_ic50_um must be a positive finite value")
    provisional: list[RankedCandidate] = []
    for evidence in candidates:
        point = float(evidence.predicted_ic50_um)
        if not math.isfinite(point) or point <= 0:
            raise ValueError(f"candidate {evidence.candidate_id} has invalid predicted_ic50_um")
        gain = math.log10(point / parent_ic50)
        notes: list[str] = []

        lower = evidence.interval90_lower_um
        upper = evidence.interval90_upper_um
        if (
            lower is None
            or upper is None
            or not math.isfinite(float(lower))
            or not math.isfinite(float(upper))
            or float(lower) <= 0
            or float(upper) < float(lower)
        ):
            interval_penalty = settings.missing_uncertainty_penalty
            notes.append("90% interval missing or invalid; declared missing-uncertainty penalty applied")
        else:
            interval_penalty = math.log10(float(upper) / float(lower))
            notes.append(f"90% interval log10 width={interval_penalty:.4f}")

        train_similarity = evidence.maximum_train_tanimoto
        if train_similarity is None or not math.isfinite(float(train_similarity)):
            applicability_penalty = settings.missing_applicability_penalty
            notes.append("training similarity missing; declared missing-applicability penalty applied")
        else:
            similarity_shortfall = max(
                0.0,
                (settings.minimum_train_tanimoto - float(train_similarity))
                / settings.minimum_train_tanimoto,
            )
            applicability_penalty = max(
                similarity_shortfall,
                _applicability_label_penalty(evidence.applicability_label),
            )
            notes.append(
                f"applicability label={evidence.applicability_label or 'unlabeled'}; "
                f"maximum train Tanimoto={float(train_similarity):.4f}"
            )

        parent_similarity = evidence.parent_morgan_tanimoto
        if parent_similarity is None or not math.isfinite(float(parent_similarity)):
            similarity_penalty = settings.missing_parent_similarity_penalty
            notes.append("parent similarity missing; declared missing-similarity penalty applied")
        else:
            similarity_penalty = max(
                0.0,
                (settings.preferred_parent_tanimoto - float(parent_similarity))
                / settings.preferred_parent_tanimoto,
            )
            notes.append(f"parent Morgan Tanimoto={float(parent_similarity):.4f}")

        scaled_property_changes = []
        for key, scale in settings.property_delta_scales.items():
            if key not in evidence.property_deltas or scale <= 0:
                continue
            value = float(evidence.property_deltas[key])
            if math.isfinite(value):
                scaled_property_changes.append(min(abs(value) / float(scale), 3.0))
        if scaled_property_changes:
            property_penalty = sum(scaled_property_changes) / len(scaled_property_changes)
            notes.append(f"mean governed property-change scale={property_penalty:.4f}")
        else:
            property_penalty = settings.missing_property_delta_penalty
            notes.append("property deltas missing; declared missing-property penalty applied")

        score = (
            settings.improvement_weight * gain
            - settings.interval_width_weight * interval_penalty
            - settings.applicability_weight * applicability_penalty
            - settings.parent_similarity_weight * similarity_penalty
            - settings.property_change_weight * property_penalty
        )
        provisional.append(
            RankedCandidate(
                rank=0,
                candidate_id=evidence.candidate_id,
                canonical_smiles=evidence.canonical_smiles,
                prioritization_score=score,
                predicted_log10_ic50_gain=gain,
                interval_width_penalty=interval_penalty,
                applicability_penalty=applicability_penalty,
                parent_similarity_penalty=similarity_penalty,
                property_change_penalty=property_penalty,
                evidence_notes=tuple(notes),
            )
        )
    ordered = sorted(provisional, key=lambda row: (-row.prioritization_score, row.candidate_id))
    return tuple(
        RankedCandidate(
            rank=index,
            candidate_id=row.candidate_id,
            canonical_smiles=row.canonical_smiles,
            prioritization_score=row.prioritization_score,
            predicted_log10_ic50_gain=row.predicted_log10_ic50_gain,
            interval_width_penalty=row.interval_width_penalty,
            applicability_penalty=row.applicability_penalty,
            parent_similarity_penalty=row.parent_similarity_penalty,
            property_change_penalty=row.property_change_penalty,
            evidence_notes=row.evidence_notes,
        )
        for index, row in enumerate(ordered, start=1)
    )
