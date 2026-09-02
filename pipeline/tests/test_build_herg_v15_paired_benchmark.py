from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_herg_v15_paired_benchmark as module  # noqa: E402


def test_canonicalization_collapses_reversed_pairs_without_using_labels() -> None:
    registry = pd.DataFrame(
        {
            "pair_id": ["pair-b", "pair-a", "pair-c"],
            "model_split": ["train", "train", "train"],
            "structure_id_a": ["S2", "S1", "S3"],
            "structure_id_b": ["S1", "S2", "S4"],
        }
    )
    canonical, audit = module.canonicalize_registry(registry)

    assert len(canonical) == 2
    assert audit == {
        "source_rows": 3,
        "canonical_rows": 2,
        "reversed_source_rows_normalized": 1,
        "duplicate_or_reversed_rows_collapsed": 1,
    }
    first = canonical[
        canonical["canonical_structure_id_a"].eq("S1")
        & canonical["canonical_structure_id_b"].eq("S2")
    ].iloc[0]
    assert first["source_pair_id"] == "pair-a"
    assert first["source_orientation_sign"] == 1


def test_leakage_groups_union_mmp_series_proxy_and_shared_scaffolds() -> None:
    registry = pd.DataFrame(
        {
            "pair_id": ["p1", "p2", "p3"],
            "model_split": ["train"] * 3,
            "structure_id_a": ["S1", "S3", "S5"],
            "structure_id_b": ["S2", "S4", "S6"],
        }
    )
    pairs, _ = module.canonicalize_registry(registry)
    oof = pd.DataFrame(
        {
            "structure_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
            # S2 and S3 share a scaffold, joining pair p1 to pair p2.
            "scaffold_group_id": ["A", "B", "B", "C", "D", "E"],
            "outer_fold": [0, 0, 0, 0, 1, 1],
        }
    )
    pairs = module._attach_freeze_metadata(pairs, oof)  # noqa: SLF001
    grouped = module.assign_leakage_groups(pairs, oof)
    grouped = module.apply_group_fold_eligibility(grouped)
    grouped = module.allocate_locked_split(grouped, seed=17, test_fraction=0.34)

    groups = grouped.set_index("source_pair_id")["leakage_group_id"].to_dict()
    assert groups["p1"] == groups["p2"]
    assert groups["p1"] != groups["p3"]
    assert module._assert_no_split_leakage(grouped) == {  # noqa: SLF001
        "canonical_or_reversed_pair_overlap_count": 0,
        "structure_split_overlap_count": 0,
        "scaffold_split_overlap_count": 0,
        "leakage_group_split_overlap_count": 0,
    }


def test_pic50_direction_and_clustered_metrics_have_correct_scientific_sign() -> None:
    # Synthetic values exist only to test arithmetic. Positive delta pIC50 is
    # stronger hERG inhibition and thus increased liability.
    frame = pd.DataFrame(
        {
            "leakage_group_id": ["g1", "g1", "g2", "g3"],
            "measured_delta_pic50_b_minus_a": [1.0, -1.0, 0.05, 2.0],
            "predicted_delta_pic50_b_minus_a": [0.8, -0.4, 0.2, -0.5],
        }
    )
    frame["measured_direction"] = module._direction(  # noqa: SLF001
        frame["measured_delta_pic50_b_minus_a"], 0.1
    )
    frame["predicted_direction"] = module._direction(  # noqa: SLF001
        frame["predicted_delta_pic50_b_minus_a"], 0.0
    )
    frame["predicted_thresholded_direction"] = module._direction(  # noqa: SLF001
        frame["predicted_delta_pic50_b_minus_a"], 0.1
    )
    evaluable = frame["measured_direction"].ne("negligible")
    frame["direction_correct"] = pd.array([pd.NA] * len(frame), dtype="boolean")
    frame.loc[evaluable, "direction_correct"] = frame.loc[evaluable, "measured_direction"].eq(
        frame.loc[evaluable, "predicted_direction"]
    )
    frame["thresholded_direction_correct"] = pd.array([pd.NA] * len(frame), dtype="boolean")
    frame.loc[evaluable, "thresholded_direction_correct"] = frame.loc[
        evaluable, "measured_direction"
    ].eq(frame.loc[evaluable, "predicted_thresholded_direction"])
    frame["activity_cliff"] = frame["measured_delta_pic50_b_minus_a"].abs().ge(1.0)
    result = module.clustered_metrics(frame, bootstrap_replicates=200, seed=11)

    assert frame.loc[0, "measured_direction"] == "increased_hERG_liability"
    assert result["n_direction_evaluable"] == 3
    assert result["metrics"]["directional_accuracy"] == pytest.approx(2 / 3)
    assert result["metrics"]["delta_mae_pic50"] == pytest.approx(
        np.mean([0.2, 0.6, 0.15, 2.5])
    )
    assert result["ci95"]["directional_accuracy"]["lower"] is not None


def _source_manifest(registry: Path, effects: Path, destination: Path) -> None:
    registry_binding = module._binding(registry, "registry")  # noqa: SLF001
    effects_binding = module._binding(effects, "effects")  # noqa: SLF001
    manifest = {
        "schema_version": "synthetic-unit-test-only",
        "artifacts": {
            "mmp_pair_registry.parquet": {
                key: registry_binding[key]
                for key in ("bytes", "sha256", "rows", "arrow_schema_sha256")
            },
            "training_mmp_effects.parquet": {
                key: effects_binding[key]
                for key in ("bytes", "sha256", "rows", "arrow_schema_sha256")
            },
        },
        "scientific_contract": {
            "effect_estimation_partition": "train_only",
            "validation_test_pairs_are_definition_only": True,
        },
    }
    unsigned = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(unsigned).hexdigest()
    destination.write_text(json.dumps(manifest), encoding="utf-8")


def _synthetic_inputs(root: Path) -> module.BenchmarkInputs:
    structure_ids = [f"S{index:02d}" for index in range(12)]
    registry = pd.DataFrame(
        {
            "pair_id": [f"P{index:02d}" for index in range(6)],
            "model_split": ["train"] * 6,
            "structure_id_a": structure_ids[::2],
            "structure_id_b": structure_ids[1::2],
        }
    )
    observed = {structure_id: 4.0 + index * 0.2 for index, structure_id in enumerate(structure_ids)}
    effects = pd.DataFrame(
        {
            "pair_id": registry["pair_id"],
            "structure_id_a": registry["structure_id_a"],
            "structure_id_b": registry["structure_id_b"],
            "pic50_median_a": registry["structure_id_a"].map(observed),
            "pic50_median_b": registry["structure_id_b"].map(observed),
            "pic50_observations_a": [1, 2, 1, 2, 1, 2],
            "pic50_observations_b": [1, 2, 2, 1, 1, 3],
            "delta_pic50_b_minus_a": [0.2] * 6,
            "activity_cliff_ge_1_pic50": [False] * 6,
            "exploratory_training_only": [True] * 6,
        }
    )
    oof = pd.DataFrame(
        {
            "structure_id": structure_ids,
            "scaffold_group_id": [f"SCF{index:02d}" for index in range(12)],
            "outer_fold": np.repeat([0, 1, 2], 4),
            "maximum_train_tanimoto": np.linspace(0.2, 0.9, 12),
            "extrapolation_flag": [False] * 10 + [True, True],
            "rdkit2d__MolWt": np.linspace(300, 800, 12),
            "pred__v11_nested": [observed[value] - 0.05 for value in structure_ids],
            "v9_predicted_pic50": [observed[value] - 0.10 for value in structure_ids],
            "pred__xgb_depth10": [observed[value] + 0.05 for value in structure_ids],
            "observed_pic50": [observed[value] for value in structure_ids],
        }
    )
    observations = pd.DataFrame(
        {
            "structure_id": structure_ids,
            "source_family": ["synthetic_source"] * 12,
            "assay_family": ["synthetic_assay"] * 12,
        }
    )
    paths = module.BenchmarkInputs(
        pair_registry=root / "registry.parquet",
        training_effects=root / "effects.parquet",
        mmp_manifest=root / "manifest.json",
        nested_oof=root / "oof.parquet",
        exact_observations=root / "observations.parquet",
    )
    registry.to_parquet(paths.pair_registry, index=False)
    effects.to_parquet(paths.training_effects, index=False)
    oof.to_parquet(paths.nested_oof, index=False)
    observations.to_parquet(paths.exact_observations, index=False)
    _source_manifest(paths.pair_registry, paths.training_effects, paths.mmp_manifest)
    return paths


def test_freeze_then_score_preserves_auditable_label_boundary(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    output = tmp_path / "paired"
    contract = module.freeze_benchmark(
        inputs=inputs,
        output_root=output,
        seed=9,
        test_fraction=0.34,
        bootstrap_replicates=100,
    )

    predictions = pd.read_parquet(
        output / "locked/baseline_oof_predictions_before_score.parquet"
    )
    locked_manifest = pd.read_parquet(output / "locked/locked_pair_manifest.parquet")
    assert contract["label_access_contract"]["training_mmp_effect_columns_opened_during_freeze"] is False
    assert contract["leakage_controls"]["campaign"].startswith("unavailable")
    assert contract["series_campaign_identifier_schema_audit"]["interpretation"] == (
        "no explicit identifier available"
    )
    assert locked_manifest.loc[
        locked_manifest["baseline_pair_eligible"], "leakage_group_outer_fold_count"
    ].eq(1).all()
    assert not any("observed" in column or "measured" in column for column in predictions.columns)
    assert not (output / "sealed/train_only_pair_labels.parquet").exists()

    validation = module.score_baseline(inputs=inputs, output_root=output)
    scores = pd.read_parquet(output / "analysis/baseline_scored_pairs.parquet")
    metrics = pd.read_parquet(output / "analysis/baseline_metrics.parquet")
    assert validation["status"] == "passed"
    assert not any("structure" in column.lower() or "smiles" in column.lower() for column in scores.columns)
    assert set(scores["model_id"]) == {"v11_nested", "v9_anchor", "xgb_depth10"}
    assert set(metrics["benchmark_split"]) == {"development", "locked_test"}
    assert module.score_baseline(inputs=inputs, output_root=output)["validation_sha256"] == validation[
        "validation_sha256"
    ]


def test_lock_is_idempotent_and_refuses_partial_state(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    output = tmp_path / "paired"
    first = module.freeze_benchmark(
        inputs=inputs,
        output_root=output,
        bootstrap_replicates=100,
    )
    second = module.freeze_benchmark(
        inputs=inputs,
        output_root=output,
        bootstrap_replicates=100,
    )
    assert first["contract_sha256"] == second["contract_sha256"]

    partial = tmp_path / "partial"
    (partial / "locked").mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(partial / "locked/locked_pair_manifest.parquet")
    with pytest.raises(module.BenchmarkError, match="partial frozen benchmark"):
        module.freeze_benchmark(
            inputs=inputs,
            output_root=partial,
            bootstrap_replicates=100,
        )
