#!/usr/bin/env python3
"""Build and verify a deployable research-only hERG V13.9 inference bundle.

The bundle packages the deployed V9/V10.1 ligand models, five frozen V13
discovery-fold receptor classifiers, the exact prepared 8ZYO receptor and box,
and the unchanged V13.6 hybrid policy.  The production receptor probability is
the arithmetic mean across the five fold models, matching the label-blind rule
used for V13.7 and V13.8 external prediction sealing.

``build`` creates the bundle and verifies probability/decision replay against
every available sealed external prediction artifact. ``predict`` accepts one
or more SMILES, runs the complete ligand + AutoDock Vina path, and writes
research predictions. Vina scores remain scoring-function observables rather
than binding free energies or proof of a unique pose.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
for candidate in (SCRIPT_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import analyze_local_herg_receptor_ensemble_v13_2 as v132  # noqa: E402
import run_local_herg_external_validation_v13_7 as v137  # noqa: E402
import run_local_herg_receptor_classification_confirmation_v13_3 as v133  # noqa: E402
import run_local_herg_receptor_classification_replication_v13_4 as v134  # noqa: E402
import run_local_herg_receptor_ensemble_campaign_v13 as v13  # noqa: E402
import run_local_herg_receptor_hybrid_validation_v13_6 as v136  # noqa: E402
import run_local_herg_v10_1_expanded_platform as v101  # noqa: E402
from menin_discovery.chemistry import standardize_smiles  # noqa: E402
from menin_discovery.features import scaffold_key  # noqa: E402

SCHEMA_VERSION = "platform-local-herg-receptor-inference-bundle-v13.9/1.0"
DEFAULT_OUTPUT = Path("research/local_runs/herg_receptor_inference_bundle_v13_9")
DEFAULT_V101 = Path("research/local_runs/herg_v10_1_expanded_platform")
DEFAULT_V13 = Path("research/local_runs/herg_receptor_ensemble_campaign_v13")
DEFAULT_V132 = Path("research/local_runs/herg_receptor_strict_analysis_v13_2")
DEFAULT_V1311 = Path("research/local_runs/herg_external_uncertainty_v13_11")
STATE = "8ZYO"
BUNDLE_NAME = "herg_receptor_hybrid_bundle.joblib"
MODEL_NAMES = (
    "exact_ternary_router.joblib",
    "ic50_regressor.joblib",
    "empirical_ic10_regressor.joblib",
    "empirical_ic30_regressor.joblib",
    "empirical_ic50_regressor.joblib",
    "broad_fixed_dose_classifier.joblib",
)
RECEPTOR_FILES = (
    "8ZYO_prepared.pdb",
    "8ZYO_prepared.pdbqt",
    "8ZYO_prepared.box.txt",
    "8ZYO_prepared.json",
)
VALIDATED_VINA_SHA256 = "823c2bbacf26d72183861322345f0a89736aca66c8e81054c66f93af5ad623f1"


class CampaignError(RuntimeError):
    """Raised when V13.9 build, replay, or inference integrity fails."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return v137._sha(path)  # noqa: SLF001


def _json(path: Path, payload: dict[str, Any], field: str) -> dict[str, Any]:
    return v137._json(path, payload, field)  # noqa: SLF001


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    v137._parquet(path, frame)  # noqa: SLF001


def _resolve(repo: Path, path: Path) -> Path:
    return v137._resolve(repo, path)  # noqa: SLF001


def _fit_receptor_models(repo: Path, discovery: pd.DataFrame) -> list[Any]:
    discovery = discovery.loc[discovery.cohort.eq("exact_ic50")].reset_index(drop=True)
    inner = v132._v9_inner_baselines(repo, discovery)  # noqa: SLF001
    models = []
    for outer in range(5):
        fit = discovery.outer_fold.ne(outer).to_numpy()
        baseline_map = inner.loc[inner.context_outer_fold.eq(outer)].set_index(
            "structure_id"
        ).inner_baseline
        training = discovery.loc[fit, [v137.PRIMARY_FEATURE]].copy()
        training.insert(
            0,
            "strict_inner_baseline",
            discovery.loc[fit, "structure_id"].map(baseline_map).to_numpy(float),
        )
        model = v133._classifier()  # noqa: SLF001
        model.fit(
            training[["strict_inner_baseline", v137.PRIMARY_FEATURE]],
            v13._tier_index(discovery.loc[fit, "observed_target"].to_numpy(float)),  # noqa: SLF001
        )
        models.append(model)
    return models


def _bundle_receptor_probabilities(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    validation = frame[[v137.PRIMARY_FEATURE]].copy()
    validation.insert(
        0,
        "strict_inner_baseline",
        frame.baseline_prediction.to_numpy(float),
    )
    fold_predictions = []
    for model in bundle["receptor_models"]:
        raw = model.predict_proba(
            validation[["strict_inner_baseline", v137.PRIMARY_FEATURE]]
        )
        aligned = np.zeros((len(frame), 3), dtype=float)
        classes = model.named_steps["model"].classes_.astype(int)
        aligned[:, classes] = raw
        fold_predictions.append(aligned)
    return np.mean(np.stack(fold_predictions, axis=0), axis=0)


def _finite_sample_radius(values: np.ndarray, coverage: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise CampaignError("uncertainty calibration received no finite residuals")
    rank = min(len(values), math.ceil((len(values) + 1) * coverage))
    return float(np.partition(values, rank - 1)[rank - 1])


def _load_uncertainty_calibration(
    v101_root: Path, uncertainty_root: Path
) -> dict[str, Any]:
    external = v137._read_json(  # noqa: SLF001
        uncertainty_root / "uncertainty_calibration.json", "calibration_sha256"
    )
    endpoint_path = v101_root / "evidence/empirical_ic10_ic30_ic50_oof.parquet"
    endpoints = pd.read_parquet(endpoint_path)
    direct = {}
    for name in ("IC10", "IC30", "IC50"):
        selected = endpoints.loc[endpoints.endpoint.eq(name)]
        residual = np.abs(
            selected.observed_picx.to_numpy(float)
            - selected.predicted_picx.to_numpy(float)
        )
        direct[name.lower()] = {
            "coverage": 0.90,
            "radius_picx": _finite_sample_radius(residual, 0.90),
            "calibration_rows": len(selected),
            "source": "V10.1 scaffold-held-out endpoint OOF absolute residuals",
        }
    return {
        "v9_mixed_ic50": {
            "coverage_levels": {
                key: {
                    "global_radius_pic50": float(value["global_radius_pic50"])
                }
                for key, value in external["coverage_levels"].items()
            },
            "external_audit": {
                "campaigns": ("V13.7", "V13.8"),
                "outcomes_used_for_width_calibration": False,
                "report_sha256": v137._read_json(  # noqa: SLF001
                    uncertainty_root / "analysis_report.json", "report_sha256"
                )["report_sha256"],
            },
            "source": external["residual_source"],
        },
        "direct_endpoints": direct,
        "claim_boundary": (
            "cross-validated residual bands; operational research uncertainty, "
            "not prospective, clinical, or finite-sample domain-shift guarantees"
        ),
        "source_hashes": {
            "external_uncertainty_calibration": _sha(
                uncertainty_root / "uncertainty_calibration.json"
            ),
            "direct_endpoint_oof": _sha(endpoint_path),
        },
    }


def _hybrid_frame(
    base: pd.DataFrame,
    receptor_probability: np.ndarray,
) -> pd.DataFrame:
    predictions = base[
        [
            "external_structure_id",
            "structure_id",
            "ligand_id",
            "scaffold_group_id",
            "baseline_prediction",
            *v136.LGBM_COLUMNS,
        ]
    ].copy()
    for index, class_name in enumerate(v13.CLASS_NAMES):
        predictions[f"primary_frozen_f649__probability_{class_name.lower()}"] = (
            receptor_probability[:, index]
        )
    predictions["hybrid_prediction"] = v136._apply_hybrid_policy(predictions)  # noqa: SLF001
    return predictions


def _build(
    repo: Path,
    output: Path,
    v101_root: Path,
    primary: Path,
    discovery_root: Path,
    uncertainty_root: Path,
) -> dict[str, Any]:
    discovery_path = discovery_root / "six_state_analysis_matrix.parquet"
    discovery = pd.read_parquet(discovery_path)
    receptor_models = _fit_receptor_models(repo, discovery)
    ligand_models = {
        name.removesuffix(".joblib"): joblib.load(v101_root / "models" / name)
        for name in MODEL_NAMES
    }
    uncertainty = _load_uncertainty_calibration(v101_root, uncertainty_root)
    source_receptor = primary / "receptors" / STATE
    receptor_hashes = {}
    for name in RECEPTOR_FILES:
        source = source_receptor / name
        if not source.is_file():
            raise CampaignError(f"missing frozen receptor file: {source}")
        receptor_hashes[name] = _sha(source)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc(),
        "class_names": tuple(v13.CLASS_NAMES),
        "state": STATE,
        "primary_feature": v137.PRIMARY_FEATURE,
        "receptor_feature_columns": ("strict_inner_baseline", v137.PRIMARY_FEATURE),
        "receptor_probability_aggregation": "arithmetic mean across five frozen fold models",
        "receptor_models": receptor_models,
        "ligand_models": ligand_models,
        "uncertainty_calibration": uncertainty,
        "policy": v136._policy_contract(),  # noqa: SLF001
        "vina_protocol": {
            "version": "1.2.7",
            "binary_sha256": VALIDATED_VINA_SHA256,
            "exhaustiveness": 8,
            "modes": 9,
            "state": STATE,
            "box_file": "receptors/8ZYO/8ZYO_prepared.box.txt",
        },
        "receptor_file_sha256": receptor_hashes,
        "claim_scope": {
            "research_only": True,
            "clinical_or_regulatory_use": False,
            "vina_scores_are_binding_free_energies": False,
            "prospective_validation": False,
        },
        "source_hashes": {
            "discovery_matrix": _sha(discovery_path),
            **{
                name: _sha(v101_root / "models" / name)
                for name in MODEL_NAMES
            },
            **uncertainty["source_hashes"],
        },
    }
    bundle_path = output / "models" / BUNDLE_NAME
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(bundle_path)

    target_receptor = output / "receptors" / STATE
    target_receptor.mkdir(parents=True, exist_ok=True)
    for name in RECEPTOR_FILES:
        source = source_receptor / name
        shutil.copy2(source, target_receptor / name)
    report = _json(
        output / "build_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "bundle_path": str(bundle_path.resolve()),
            "bundle_bytes": bundle_path.stat().st_size,
            "bundle_sha256": _sha(bundle_path),
            "receptor_models": len(receptor_models),
            "receptor_training_rows": int(discovery.cohort.eq("exact_ic50").sum()),
            "ligand_model_bundles": sorted(ligand_models),
            "receptor_files": [
                {
                    "path": str((target_receptor / name).resolve()),
                    "sha256": _sha(target_receptor / name),
                }
                for name in RECEPTOR_FILES
            ],
            "validated_vina_binary_sha256": VALIDATED_VINA_SHA256,
            "uncertainty": {
                "v9_mixed_ic50_external_audit_available": True,
                "direct_endpoint_interval_coverage": 0.90,
                "claim_boundary": uncertainty["claim_boundary"],
            },
            "external_labels_used_for_bundle_fitting_or_policy_tuning": False,
        },
        "report_sha256",
    )
    (output / "MODEL_CARD.md").write_text(_render_model_card(report))
    return report


def _render_model_card(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# hERG V13.9 receptor-aware inference bundle",
            "",
            "## Intended use",
            "",
            "Research-stage prioritization of chemically standardized, docking-eligible small molecules. The bundle emits ligand-only mixed IC50, direct empirical IC10/IC30/IC50 estimates, ternary router probabilities, one-state 8ZYO receptor probabilities, and the frozen V13.6 hybrid tier.",
            "",
            "## Frozen implementation",
            "",
            f"- Bundle SHA-256: `{report['bundle_sha256']}`",
            f"- Receptor models: {report['receptor_models']} fold models trained without external labels",
            f"- Receptor discovery rows: {report['receptor_training_rows']}",
            f"- Vina binary SHA-256: `{report['validated_vina_binary_sha256']}`",
            "- Receptor state: 8ZYO; exhaustiveness 8; nine requested modes.",
            "- Receptor probability: arithmetic mean of the five frozen fold-model probabilities.",
            "- Hybrid routing: the unchanged V13.6 Safe/Potent tail-override policy.",
            "- Uncertainty: frozen nested-OOF residual bands; the mixed-IC50 90% band was audited unchanged on V13.7 and V13.8.",
            "",
            "## Output meaning",
            "",
            "pIC values are -log10(molar concentration); concentration outputs are micromolar. The deterministic endpoint projection enforces IC10 <= IC30 <= IC50 in concentration units when all three predictions are generated. Tier labels are Safe, Moderate, and Potent under the repository's fixed pIC50 cutoffs.",
            "",
            "## Validation and limitations",
            "",
            "The verification report must show exact replay of sealed external probabilities and decisions before use. V13.7 showed an incremental class-balanced gain but narrowly failed its absolute Potent-to-Safe gate; V13.8 did not replicate the gain and also failed safety. The hybrid is therefore an experimental model-challenge hypothesis, not a generally promoted classifier. Vina scores are scoring-function observables, not experimental affinities, binding free energies, unique poses, or causal residue evidence. This bundle is not prospectively validated and must not be used for clinical or regulatory decisions.",
            "",
        ]
    )


def _validate_inference_artifacts(
    bundle: dict[str, Any], output: Path, tools: v13.Toolchain
) -> None:
    expected_vina = str(bundle["vina_protocol"]["binary_sha256"])
    observed_vina = _sha(tools.vina)
    if observed_vina != expected_vina:
        raise CampaignError(
            "Vina binary does not match the externally validated V13.7/V13.8 binary "
            f"({observed_vina} != {expected_vina})"
        )
    for name, expected in bundle["receptor_file_sha256"].items():
        path = output / "receptors" / STATE / name
        if not path.is_file() or _sha(path) != expected:
            raise CampaignError(f"frozen receptor artifact failed integrity check: {path}")


def _verify_one(bundle: dict[str, Any], root: Path) -> dict[str, Any]:
    baseline_path = root / "predictions/baseline_predictions_before_score.parquet"
    feature_path = root / "docking/external_receptor_features.parquet"
    sealed_path = root / "predictions/receptor_predictions_before_score.parquet"
    seal_path = root / "predictions/receptor_prediction_seal.json"
    for path in (baseline_path, feature_path, sealed_path, seal_path):
        if not path.is_file():
            raise CampaignError(f"verification artifact missing: {path}")
    seal = v137._read_json(seal_path, "seal_sha256")  # noqa: SLF001
    if _sha(sealed_path) != seal["prediction_sha256"]:
        raise CampaignError(f"sealed prediction checksum mismatch: {root}")
    baseline = pd.read_parquet(baseline_path)
    feature = pd.read_parquet(feature_path)
    selected = baseline.loc[baseline.receptor_evaluation_eligible.astype(bool)].merge(
        feature, on="ligand_id", validate="one_to_one"
    )
    expected = pd.read_parquet(sealed_path).sort_values("external_structure_id").reset_index(
        drop=True
    )
    selected = selected.sort_values("external_structure_id").reset_index(drop=True)
    probability = _bundle_receptor_probabilities(bundle, selected)
    actual = _hybrid_frame(selected, probability)
    expected_probability = expected[list(v136.RECEPTOR_COLUMNS)].to_numpy(float)
    max_difference = float(np.max(np.abs(probability - expected_probability)))
    decisions_equal = bool(
        np.array_equal(
            actual.hybrid_prediction.to_numpy(int),
            expected.hybrid_prediction.to_numpy(int),
        )
    )
    ids_equal = bool(
        actual.external_structure_id.astype(str).equals(
            expected.external_structure_id.astype(str)
        )
    )
    passed = bool(max_difference <= 1e-12 and decisions_equal and ids_equal)
    if not passed:
        raise CampaignError(f"bundle replay failed for {root.name}")
    return {
        "campaign": root.name,
        "rows": len(actual),
        "maximum_absolute_probability_difference": max_difference,
        "hybrid_decisions_identical": decisions_equal,
        "structure_order_identical": ids_equal,
        "sealed_prediction_sha256": seal["prediction_sha256"],
        "passed": passed,
    }


def _verify(repo: Path, output: Path) -> dict[str, Any]:
    bundle_path = output / "models" / BUNDLE_NAME
    bundle = joblib.load(bundle_path)
    candidates = [
        repo / "research/local_runs/herg_external_validation_v13_7",
        repo / "research/local_runs/herg_temporal_confirmation_v13_8",
    ]
    available = [
        root
        for root in candidates
        if (root / "predictions/receptor_predictions_before_score.parquet").is_file()
    ]
    if not available:
        raise CampaignError("no sealed external receptor predictions are available for replay")
    replays = [_verify_one(bundle, root) for root in available]
    return _json(
        output / "verification_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "bundle_sha256": _sha(bundle_path),
            "sealed_external_campaigns_replayed": len(replays),
            "total_rows_replayed": sum(row["rows"] for row in replays),
            "all_passed": all(row["passed"] for row in replays),
            "replays": replays,
        },
        "report_sha256",
    )


def _input_registry(smiles: Iterable[str]) -> pd.DataFrame:
    rows = []
    seen = set()
    for position, value in enumerate(smiles):
        standardized = standardize_smiles(
            value,
            strip_salts=True,
            canonicalize_tautomer=False,
            require_rdkit=True,
        )
        if not standardized.structure_valid:
            raise CampaignError(
                f"SMILES {position + 1} failed standardization: {standardized.structure_error}"
            )
        external_id = v137._external_structure_id(  # noqa: SLF001
            standardized.standard_inchi_key, standardized.standardized_smiles
        )
        if external_id in seen:
            continue
        seen.add(external_id)
        raw_scaffold = scaffold_key(standardized.standardized_smiles)[0]
        rows.append(
            {
                "external_structure_id": external_id,
                "structure_id": external_id,
                "ligand_id": "inference__" + external_id,
                "input_smiles": str(value),
                "standardized_smiles": standardized.standardized_smiles,
                "standard_inchi_key": standardized.standard_inchi_key,
                "scaffold_group_id": v137._external_scaffold_id(raw_scaffold),  # noqa: SLF001
                "outer_fold": -1,
            }
        )
    if not rows:
        raise CampaignError("no unique valid SMILES were supplied")
    return v13._add_dockability(pd.DataFrame(rows))  # noqa: SLF001


def _ligand_predictions(bundle: dict[str, Any], registry: pd.DataFrame) -> pd.DataFrame:
    models = bundle["ligand_models"]
    router = models["exact_ternary_router"]
    mixed = models["ic50_regressor"]
    endpoint_models = {
        "ic10": models["empirical_ic10_regressor"],
        "ic30": models["empirical_ic30_regressor"],
        "ic50": models["empirical_ic50_regressor"],
    }
    broad = models["broad_fixed_dose_classifier"]
    feature_cache: dict[tuple[str, ...], np.ndarray] = {}
    for model in (router, mixed, *endpoint_models.values(), broad):
        key = tuple(model["feature_columns"])
        if key not in feature_cache:
            feature_cache[key] = v137._feature_matrix(  # noqa: SLF001
                registry.standardized_smiles, list(key)
            )
    router_raw = router["model"].predict_proba(feature_cache[tuple(router["feature_columns"])])
    router_probability = router["calibrator"].predict_proba(
        np.log(np.clip(router_raw, 1e-7, 1.0))
    )
    mixed_pic50 = mixed["model"].predict(feature_cache[tuple(mixed["feature_columns"])])
    raw_endpoints = {
        name: model["model"].predict(feature_cache[tuple(model["feature_columns"])])
        for name, model in endpoint_models.items()
    }
    broad_raw = broad["model"].predict_proba(
        feature_cache[tuple(broad["feature_columns"])]
    )[:, 1]
    broad_logit = np.log(
        np.clip(broad_raw, 1e-7, 1 - 1e-7) / np.clip(1 - broad_raw, 1e-7, 1)
    )
    broad_probability = broad["calibrator"].predict_proba(broad_logit[:, None])[:, 1]

    result = registry[
        [
            "external_structure_id",
            "structure_id",
            "ligand_id",
            "input_smiles",
            "standardized_smiles",
            "standard_inchi_key",
            "scaffold_group_id",
            "docking_eligible",
            "docking_exclusion_reason",
        ]
    ].copy()
    result["baseline_prediction"] = np.asarray(mixed_pic50, dtype=float)
    result["v9_mixed_ic50_predicted_pic50"] = np.asarray(mixed_pic50, dtype=float)
    mixed_calibration = bundle["uncertainty_calibration"]["v9_mixed_ic50"]
    for level, calibration in mixed_calibration["coverage_levels"].items():
        percent = int(round(float(level) * 100))
        radius = float(calibration["global_radius_pic50"])
        lower = result.v9_mixed_ic50_predicted_pic50 - radius
        upper = result.v9_mixed_ic50_predicted_pic50 + radius
        result[f"v9_mixed_ic50_interval{percent}_lower_pic50"] = lower
        result[f"v9_mixed_ic50_interval{percent}_upper_pic50"] = upper
        result[f"v9_mixed_ic50_interval{percent}_lower_um"] = 10 ** (6.0 - upper)
        result[f"v9_mixed_ic50_interval{percent}_upper_um"] = 10 ** (6.0 - lower)
    for index, name in enumerate(v13.CLASS_NAMES):
        result[f"lgbm_rdkit2d_morgan__probability_{name.lower()}"] = router_probability[
            :, index
        ]
    result["ligand_router_prediction"] = np.argmax(router_probability, axis=1)
    result["broad_fixed_dose_probability_active_at_46um"] = broad_probability
    projected = []
    for row in range(len(registry)):
        raw_um = [
            float(10 ** (6 - raw_endpoints[name][row]))
            for name in ("ic10", "ic30", "ic50")
        ]
        coherent_um, adjusted = v101._coherent_threshold_concentrations(raw_um)  # noqa: SLF001
        projected.append((*coherent_um, adjusted))
    for index, name in enumerate(("ic10", "ic30", "ic50")):
        result[f"empirical_{name}_predicted_um"] = [row[index] for row in projected]
        result[f"empirical_{name}_predicted_picx"] = [
            6.0 - math.log10(row[index]) for row in projected
        ]
        result[f"empirical_{name}_raw_predicted_picx"] = raw_endpoints[name]
        radius = float(
            bundle["uncertainty_calibration"]["direct_endpoints"][name]["radius_picx"]
        )
        lower = result[f"empirical_{name}_predicted_picx"] - radius
        upper = result[f"empirical_{name}_predicted_picx"] + radius
        result[f"empirical_{name}_interval90_lower_picx"] = lower
        result[f"empirical_{name}_interval90_upper_picx"] = upper
        result[f"empirical_{name}_interval90_lower_um"] = 10 ** (6.0 - upper)
        result[f"empirical_{name}_interval90_upper_um"] = 10 ** (6.0 - lower)
    result["endpoint_order_projection_applied"] = [row[3] for row in projected]
    return result


def _predict(
    repo: Path,
    output: Path,
    prediction_root: Path,
    smiles: list[str],
    vina: Path | None,
    cpu: int,
) -> dict[str, Any]:
    bundle_path = output / "models" / BUNDLE_NAME
    bundle = joblib.load(bundle_path)
    registry = _input_registry(smiles)
    ligand = _ligand_predictions(bundle, registry)
    ligand_path = prediction_root / "ligand_predictions.parquet"
    _parquet(ligand_path, ligand)
    eligible = registry.loc[registry.docking_eligible.astype(bool)].copy()
    final = ligand.copy()
    final["receptor_prediction_available"] = False
    final["hybrid_prediction"] = pd.Series([pd.NA] * len(final), dtype="Int64")
    if not eligible.empty:
        tools = v13._resolve_toolchain(repo, vina)  # noqa: SLF001
        _validate_inference_artifacts(bundle, output, tools)
        v136._prepare_panel(eligible, prediction_root, tools)  # noqa: SLF001
        docking = v134._dock(  # noqa: SLF001
            output,
            prediction_root,
            tools,
            eligible,
            int(bundle["vina_protocol"]["exhaustiveness"]),
            int(bundle["vina_protocol"]["modes"]),
            cpu,
        )
        features = v13._aggregate_docking(docking, (STATE,))  # noqa: SLF001
        receptor_input = ligand.merge(features, on="ligand_id", validate="one_to_one")
        probability = _bundle_receptor_probabilities(bundle, receptor_input)
        hybrid = _hybrid_frame(receptor_input, probability)
        receptor = hybrid[
            ["external_structure_id", "ligand_id", "hybrid_prediction"]
        ].copy()
        for index, name in enumerate(v13.CLASS_NAMES):
            receptor[f"receptor_probability_{name.lower()}"] = probability[:, index]
        receptor = receptor.merge(
            features[
                [
                    "ligand_id",
                    v137.PRIMARY_FEATURE,
                    "dock__8ZYO__affinity",
                    "dock__8ZYO__ligand_efficiency",
                ]
            ],
            on="ligand_id",
            how="left",
            validate="one_to_one",
        )
        receptor_path = prediction_root / "receptor_predictions.parquet"
        _parquet(receptor_path, receptor)
        final = final.drop(columns="hybrid_prediction").merge(
            receptor.drop(columns="ligand_id"),
            on="external_structure_id",
            how="left",
            validate="one_to_one",
        )
        final["receptor_prediction_available"] = final.hybrid_prediction.notna()
    final["ligand_router_tier"] = final.ligand_router_prediction.map(
        dict(enumerate(v13.CLASS_NAMES))
    )
    final["hybrid_tier"] = final.hybrid_prediction.map(dict(enumerate(v13.CLASS_NAMES)))
    final_path = prediction_root / "predictions.parquet"
    _parquet(final_path, final)
    return _json(
        prediction_root / "prediction_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "bundle_sha256": _sha(bundle_path),
            "input_rows": len(smiles),
            "unique_standardized_structures": len(registry),
            "receptor_predictions": int(final.receptor_prediction_available.sum()),
            "predictions_path": str(final_path.resolve()),
            "predictions_sha256": _sha(final_path),
            "research_only": True,
            "prospective_or_clinical_validation": False,
        },
        "report_sha256",
    )


def _verification_is_current(output: Path) -> bool:
    bundle_path = output / "models" / BUNDLE_NAME
    verification_path = output / "verification_report.json"
    if not bundle_path.is_file() or not verification_path.is_file():
        return False
    try:
        verification = v137._read_json(verification_path, "report_sha256")  # noqa: SLF001
        return bool(
            verification.get("all_passed")
            and verification.get("bundle_sha256") == _sha(bundle_path)
        )
    except (OSError, KeyError, ValueError, v137.CampaignError):
        return False


def _manifest(repo: Path, output: Path) -> dict[str, Any]:
    inputs = [
        repo / "pipeline/scripts/build_local_herg_receptor_inference_bundle_v13_9.py",
        repo / "pipeline/scripts/run_local_herg_external_validation_v13_7.py",
        repo / "research/local_runs/herg_receptor_strict_analysis_v13_2/manifest.json",
        repo / "research/local_runs/herg_receptor_hybrid_validation_v13_6/manifest.json",
        repo / "research/local_runs/herg_v10_1_expanded_platform/manifest.json",
        repo / "research/local_runs/herg_external_uncertainty_v13_11/manifest.json",
    ]
    bundle_path = output / "models" / BUNDLE_NAME
    verification_path = output / "verification_report.json"
    verification_current = _verification_is_current(output)
    artifacts = [
        path
        for path in (
            bundle_path,
            output / "build_report.json",
            output / "MODEL_CARD.md",
            *([verification_path] if verification_current else []),
            *[output / "receptors" / STATE / name for name in RECEPTOR_FILES],
        )
        if path.is_file()
    ]
    return _json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc(),
            "status": (
                "complete_verified"
                if verification_current
                else "in_progress"
            ),
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
    parser.add_argument("--stage", choices=("build", "verify", "predict", "all"), default="all")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--v10-1-root", type=Path, default=DEFAULT_V101)
    parser.add_argument("--v13-root", type=Path, default=DEFAULT_V13)
    parser.add_argument("--v13-2-root", type=Path, default=DEFAULT_V132)
    parser.add_argument("--v13-11-root", type=Path, default=DEFAULT_V1311)
    parser.add_argument("--prediction-root", type=Path, default=Path("research/local_runs/herg_v13_9_prediction"))
    parser.add_argument("--smiles", action="append", default=[])
    parser.add_argument("--smiles-file", type=Path)
    parser.add_argument("--vina", type=Path)
    parser.add_argument("--cpu", type=int, default=6)
    return parser


def _read_smiles(args: argparse.Namespace, repo: Path) -> list[str]:
    values = [str(value).strip() for value in args.smiles if str(value).strip()]
    if args.smiles_file:
        path = _resolve(repo, args.smiles_file)
        values.extend(
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return values


def _main(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    output = _resolve(repo, args.output_root)
    v101_root = _resolve(repo, args.v10_1_root)
    primary = _resolve(repo, args.v13_root)
    discovery_root = _resolve(repo, args.v13_2_root)
    uncertainty_root = _resolve(repo, args.v13_11_root)
    prediction_root = _resolve(repo, args.prediction_root)
    vina = args.vina.resolve() if args.vina else None
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "stage": args.stage}
    if args.stage in ("build", "all"):
        result["build"] = _build(
            repo, output, v101_root, primary, discovery_root, uncertainty_root
        )
    if args.stage in ("verify", "all"):
        result["verify"] = _verify(repo, output)
    if args.stage in ("predict", "all"):
        smiles = _read_smiles(args, repo)
        if smiles:
            prediction_root.mkdir(parents=True, exist_ok=True)
            result["predict"] = _predict(
                repo, output, prediction_root, smiles, vina, args.cpu
            )
        elif args.stage == "predict":
            raise CampaignError("predict stage requires --smiles or --smiles-file")
    result["manifest"] = _manifest(repo, output)
    return result


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _main(args)
    except (
        CampaignError,
        v137.CampaignError,
        v13.CampaignError,
        v133.CampaignError,
        v134.CampaignError,
        v136.CampaignError,
    ) as exc:
        print(f"V13.9 ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
