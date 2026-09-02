import math

import pytest
from menin_discovery.herg_candidate_generation import (
    CandidatePredictionEvidence,
    GenerationBounds,
    RankingConfig,
    TransformationSpec,
    generate_herg_edit_candidates,
    rank_candidate_evidence,
    rdkit_generation_available,
    validate_transformation_registry,
)
from rdkit import Chem

pytestmark = pytest.mark.skipif(
    not rdkit_generation_available(),
    reason="candidate enumeration tests require RDKit",
)


RICH_PARENT = "CCN(CC)CCOc1ccc(C(=O)NCCc2ccc(F)cc2)cc1"


def test_governed_registry_compiles_and_default_run_reaches_bounded_target():
    assert validate_transformation_registry() == ()

    result = generate_herg_edit_candidates(RICH_PARENT)

    assert result.status == "completed"
    assert len(result.candidates) == 100
    assert result.valid_unique_before_selection >= len(result.candidates)
    assert result.truncated_to_target is True
    assert result.registry_version
    assert "no synthesis feasibility" in result.claim_boundary.lower()


def test_generation_is_deterministic_valid_unique_and_excludes_parent():
    bounds = GenerationBounds(target_count=40, seed=17)
    first = generate_herg_edit_candidates(RICH_PARENT, bounds=bounds)
    second = generate_herg_edit_candidates(RICH_PARENT, bounds=bounds)
    first_smiles = [candidate.canonical_smiles for candidate in first.candidates]
    second_smiles = [candidate.canonical_smiles for candidate in second.candidates]

    assert first_smiles == second_smiles
    assert len(first_smiles) == len(set(first_smiles)) == 40
    assert first.parent_canonical_smiles not in first_smiles
    assert all(Chem.MolFromSmiles(smiles) is not None for smiles in first_smiles)
    assert all(candidate.lineages for candidate in first.candidates)
    assert all(candidate.lineages[0].reaction_smarts for candidate in first.candidates)
    assert all(candidate.lineages[0].provenance for candidate in first.candidates)
    assert all(candidate.parent_morgan_tanimoto >= bounds.minimum_parent_tanimoto for candidate in first.candidates)


def test_duplicate_products_merge_lineage_instead_of_duplicating_candidate():
    registry = (
        TransformationSpec("fluoro_a", "Aryl fluorination A", "[cH:1]>>[c:1]F", "test"),
        TransformationSpec("fluoro_b", "Aryl fluorination B", "[cH:1]>>[c:1]F", "test"),
    )
    result = generate_herg_edit_candidates(
        "c1ccccc1",
        bounds=GenerationBounds(target_count=10, minimum_parent_tanimoto=0.0),
        registry=registry,
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.canonical_smiles == "Fc1ccccc1"
    assert {lineage.transformation_key for lineage in candidate.lineages} == {"fluoro_a", "fluoro_b"}
    assert len(candidate.lineages) == 12  # six symmetric matches from each governed transform


def test_charge_and_edit_bounds_are_enforced_for_charged_parent():
    bounds = GenerationBounds(target_count=35, max_abs_formal_charge_delta=0)
    result = generate_herg_edit_candidates("C[N+](C)(C)Cc1ccccc1", bounds=bounds)

    assert result.candidates
    assert all(candidate.formal_charge_delta == 0 for candidate in result.candidates)
    assert all(abs(candidate.heavy_atom_delta) <= bounds.max_abs_heavy_atom_delta for candidate in result.candidates)
    assert all(
        abs(candidate.molecular_weight_delta_da) <= bounds.max_abs_molecular_weight_delta_da
        for candidate in result.candidates
    )


def test_invalid_and_very_large_parents_fail_gracefully_with_records():
    invalid = generate_herg_edit_candidates("not-a-smiles")
    assert invalid.status == "invalid_parent"
    assert invalid.candidates == ()
    assert invalid.failures[0].reason == "invalid_parent_smiles"

    large = generate_herg_edit_candidates(
        "C" * 181,
        bounds=GenerationBounds(max_parent_heavy_atoms=180),
    )
    assert large.status == "parent_outside_generation_bounds"
    assert large.candidates == ()
    assert large.failures[0].reason == "parent_exceeds_heavy_atom_bound"


def test_no_match_and_target_shortfall_are_recorded_without_claiming_success():
    result = generate_herg_edit_candidates(
        "CCO",
        bounds=GenerationBounds(target_count=100),
    )
    reasons = {failure.reason for failure in result.failures}

    assert result.status in {"partial", "no_candidates"}
    assert "no_match" in reasons
    if result.candidates:
        assert "target_count_not_reached" in reasons


def test_ranking_penalizes_uncertainty_and_ood_instead_of_using_point_ic50_alone():
    candidates = (
        CandidatePredictionEvidence(
            candidate_id="large-point-weak-evidence",
            canonical_smiles="CC",
            predicted_ic50_um=10.0,
            interval90_lower_um=0.001,
            interval90_upper_um=1_000.0,
            applicability_label="Extrapolative chemistry",
            maximum_train_tanimoto=0.10,
            parent_morgan_tanimoto=0.40,
            property_deltas={"molecular_weight": 140.0, "clogp": 2.0, "tpsa": 50.0},
            model_provenance="test-model",
        ),
        CandidatePredictionEvidence(
            candidate_id="moderate-point-supported",
            canonical_smiles="CCC",
            predicted_ic50_um=4.0,
            interval90_lower_um=3.0,
            interval90_upper_um=5.0,
            applicability_label="High support",
            maximum_train_tanimoto=0.90,
            parent_morgan_tanimoto=0.90,
            property_deltas={"molecular_weight": 14.0, "clogp": 0.1, "tpsa": 2.0},
            model_provenance="test-model",
        ),
    )

    ranked = rank_candidate_evidence(1.0, candidates)

    assert ranked[0].candidate_id == "moderate-point-supported"
    assert ranked[0].predicted_log10_ic50_gain < ranked[1].predicted_log10_ic50_gain
    assert ranked[1].interval_width_penalty > ranked[0].interval_width_penalty
    assert ranked[1].applicability_penalty > ranked[0].applicability_penalty
    assert "not a probability" in ranked[0].score_boundary.lower()


def test_ranking_missing_evidence_penalties_and_tie_order_are_deterministic():
    evidence = tuple(
        CandidatePredictionEvidence(
            candidate_id=candidate_id,
            canonical_smiles="CCO",
            predicted_ic50_um=2.0,
            interval90_lower_um=None,
            interval90_upper_um=None,
            applicability_label="",
            maximum_train_tanimoto=None,
            parent_morgan_tanimoto=None,
            property_deltas={},
        )
        for candidate_id in ("b", "a")
    )

    ranked = rank_candidate_evidence(1.0, evidence, config=RankingConfig())

    assert [row.candidate_id for row in ranked] == ["a", "b"]
    assert all(math.isfinite(row.prioritization_score) for row in ranked)
    assert all(row.interval_width_penalty > 0 for row in ranked)
    assert all(any("missing" in note for note in row.evidence_notes) for row in ranked)
