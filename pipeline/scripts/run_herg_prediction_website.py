#!/usr/bin/env python3
"""Serve the launch-ready hERG prediction website and frozen research models."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import hmac
import json
import math
import os
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import analyze_herg_mw_pair_delta_attenuation as mw_delta
import build_local_herg_receptor_inference_bundle_v13_9 as v139
import herg_v15_functional_group_features as v15fg
import joblib
import numpy as np
import pandas as pd
import run_local_herg_cross_campaign_receptor_fusion_v14 as v14
import run_local_herg_v10_1_expanded_platform as v101
import run_local_herg_v10_3_decision_platform as v103
from menin_discovery.herg_candidate_generation import (
    CandidatePredictionEvidence,
    GenerationBounds,
    generate_herg_edit_candidates,
    rank_candidate_evidence,
)
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import (
    QED,
    ChemicalFeatures,
    Crippen,
    Descriptors,
    Fragments,
    Lipinski,
    rdMolDescriptors,
)
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

WEBSITE_SCHEMA = "herg-prediction-website/1.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = REPO_ROOT / "pipeline/web/herg"
DEFAULT_V101 = REPO_ROOT / "research/local_runs/herg_v10_1_expanded_platform"
DEFAULT_V102 = REPO_ROOT / "research/local_runs/herg_v10_2_coherent_platform"
DEFAULT_V103 = REPO_ROOT / "research/local_runs/herg_v10_3_decision_platform"
DEFAULT_V13_PANEL = REPO_ROOT / "research/local_runs/herg_receptor_ensemble_campaign_v13"
DEFAULT_V139 = REPO_ROOT / "research/local_runs/herg_receptor_inference_bundle_v13_9"
DEFAULT_V14 = REPO_ROOT / "research/local_runs/herg_cross_campaign_receptor_fusion_v14"
DEFAULT_V141 = REPO_ROOT / "research/local_runs/herg_nonlinear_receptor_stress_v14_1"
DEFAULT_V111_OOF = (
    REPO_ROOT
    / "research/local_runs/herg_comprehensive_optimization_v11_1/analysis/nested_oof_predictions.parquet"
)
DEFAULT_V15_ROOT = REPO_ROOT / "research/local_runs/herg_functional_group_finalize_v15"
DEFAULT_V15_FEATURES = DEFAULT_V15_ROOT / "data/functional_group_features.parquet"
DEFAULT_V15_MATRIX = (
    REPO_ROOT / "research/local_runs/herg_cross_campaign_receptor_fusion_v14/data/harmonized_campaigns.parquet"
)
DEFAULT_INTERNAL_EXAMPLES = REPO_ROOT / "research/data/internal/exports/ten_internal_menin_herg_examples.xlsx"
MAX_BODY_BYTES = 16_384
MAX_SMILES_LENGTH = 4_096
CLASS_NAMES = ("Safe", "Moderate", "Potent")
PREDICTION_MODES = ("ligand", "ligand_receptor")
FEATURE_MODES = ("atomwise", "functional_group")
RECEPTOR_CACHE_SIZE = 128
COMPARISON_DIRECTION_EPSILON_PIC50 = 0.05
HIGH_SIMILARITY_TANIMOTO = 0.80
LOW_SENSITIVITY_DELTA_PIC50 = 0.15
HIGH_MW_REFERENCE_PERCENTILE = 0.95
MAX_RECEPTOR_SELECTION = 6
MAX_GENERATED_CANDIDATES = 100
MW_STRESS_REFERENCE_MEDIAN_MW_DA = 432.4859924316406
MW_STRESS_BASELINE_MULTIPLIER = 1.5468564492450243
MW_STRESS_AUDIT_PROVENANCE = {
    "source": "V11.1 measured matched-molecular-pair registry with scaffold-held-out V9 OOF predictions",
    "activity_cliff_definition": "absolute measured delta pIC50 >= 1.0",
    "activity_cliff_pairs": 5_484,
    "connected_mmp_components": 536,
    "directional_accuracy": 0.711889132020423,
    "magnitude_capture_fraction": 0.3040557331365394,
    "retrospective_l1_optimal_nonnegative_scale": MW_STRESS_BASELINE_MULTIPLIER,
    "reference_training_structures": 18_801,
    "support_note": "support above 700 Da is sparse and descriptive only",
}
RECEPTOR_STATES = {
    "8ZYN": {
        "structure_context": "apo hERG",
        "asset_tier": "V13 core ensemble",
        "model_role": "docking_only_diagnostic",
    },
    "8ZYO": {
        "structure_context": "astemizole-bound hERG",
        "asset_tier": "V13 sensitivity state and frozen V13.9 inference receptor",
        "model_role": "model_driving_and_docking_diagnostic",
    },
    "8ZYP": {
        "structure_context": "E-4031-bound hERG",
        "asset_tier": "V13 core ensemble",
        "model_role": "docking_only_diagnostic",
    },
    "8ZYQ": {
        "structure_context": "pimozide-bound hERG",
        "asset_tier": "V13 sensitivity state",
        "model_role": "docking_only_diagnostic",
    },
    "9CHP": {
        "structure_context": "high-potassium C4 hERG",
        "asset_tier": "V13 core ensemble",
        "model_role": "docking_only_diagnostic",
    },
    "9CHQ": {
        "structure_context": "low-potassium C4 hERG",
        "asset_tier": "V13 core ensemble",
        "model_role": "docking_only_diagnostic",
    },
}

_FUNCTIONAL_GROUP_SURFACE_CACHE: dict[str, Any] | None = None
_FUNCTIONAL_GROUP_SURFACE_LOCK = threading.Lock()


def _quiet_smiles_molecule(smiles: str) -> Chem.Mol | None:
    """Parse user input without echoing a submitted structure into RDKit stderr logs."""
    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(smiles)


def _docking_execution_contract(registry: pd.DataFrame) -> dict[str, Any]:
    """Separate technical Vina execution from the frozen receptor-model domain."""
    row = registry.iloc[0]
    model_reasons = {
        reason
        for reason in str(row.docking_exclusion_reason).split(";")
        if reason and reason != "eligible"
    }
    blocking_reasons = {
        reason
        for reason in model_reasons
        if reason == "rdkit_parse_failure" or reason.startswith("unsupported_elements_")
    }
    parent_smiles = str(getattr(row, "docking_parent_smiles", "") or "")
    if not parent_smiles:
        blocking_reasons.add("missing_docking_parent_smiles")
    execution_eligible = not blocking_reasons
    advisory_reasons = sorted(model_reasons - blocking_reasons)
    return {
        "docking_execution_eligible": execution_eligible,
        "docking_execution_reason": (
            "eligible"
            if execution_eligible and not advisory_reasons
            else "eligible_with_model_domain_warnings:" + ";".join(advisory_reasons)
            if execution_eligible
            else ";".join(sorted(blocking_reasons))
        ),
        "receptor_feature_computation_eligible": execution_eligible,
        "receptor_model_applicable": bool(row.docking_eligible),
        "receptor_model_applicability_reason": str(row.docking_exclusion_reason),
        "final_ligand_prediction_eligible": True,
        "final_receptor_prediction_eligible": bool(row.docking_eligible),
        "advisory_model_domain_reasons": advisory_reasons,
        "blocking_execution_reasons": sorted(blocking_reasons),
    }


def _docking_execution_registry(registry: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row registry accepted by preparation when execution is technically allowed."""
    contract = _docking_execution_contract(registry)
    if not contract["docking_execution_eligible"]:
        return registry.iloc[0:0].copy()
    execution = registry.copy()
    execution.loc[:, "docking_eligible"] = True
    execution.loc[:, "docking_exclusion_reason"] = "eligible_for_diagnostic_execution"
    return execution

STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8", "no-store"),
    "/index.html": ("index.html", "text/html; charset=utf-8", "no-store"),
    "/assets/styles.css": ("styles.css", "text/css; charset=utf-8", "no-store"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8", "no-store"),
}


class WebsiteError(ValueError):
    """A safe, user-facing website validation error."""


def _structure_depiction(molecule: Chem.Mol) -> dict[str, Any]:
    """Return a browser-safe, self-contained RDKit 2D depiction."""
    drawer = rdMolDraw2D.MolDraw2DSVG(420, 260)
    options = drawer.drawOptions()
    options.clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().encode("utf-8")
    return {
        "result_kind": "calculated_depiction",
        "data_uri": "data:image/svg+xml;base64," + base64.b64encode(svg).decode("ascii"),
        "media_type": "image/svg+xml",
        "method": "RDKit 2D coordinate generation and SVG depiction",
    }


def _basic_authorization(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {token}"


def _finite_probability(value: Any) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise WebsiteError("The classification model returned an invalid probability")
    return number


def _prediction_mode(value: Any) -> str:
    mode = str(value or "ligand").strip().lower()
    if mode not in PREDICTION_MODES:
        raise WebsiteError("Prediction mode must be 'ligand' or 'ligand_receptor'")
    return mode


def _feature_mode(value: Any) -> str:
    """Validate the two deliberately narrow ligand feature choices."""
    mode = str(value or "atomwise").strip().lower()
    if mode not in FEATURE_MODES:
        raise WebsiteError("Feature mode must be 'atomwise' or 'functional_group'")
    return mode


def _boolean_flag(value: Any, label: str) -> bool:
    """Accept only an actual JSON boolean for an opt-in scientific diagnostic."""
    if not isinstance(value, bool):
        raise WebsiteError(f"{label} must be true or false")
    return value


def _receptor_selection(value: Any) -> tuple[str, ...]:
    """Validate an ordered, explicit receptor panel from the API request."""
    if value is None:
        return ("8ZYO",)
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise WebsiteError("Receptors must be a JSON array of 1 to 6 receptor IDs")
    if not 1 <= len(value) <= MAX_RECEPTOR_SELECTION:
        raise WebsiteError("Select between 1 and 6 receptor structures")
    receptors = tuple(str(item).strip().upper() for item in value)
    if any(not receptor for receptor in receptors):
        raise WebsiteError("Receptor IDs cannot be empty")
    if len(set(receptors)) != len(receptors):
        raise WebsiteError("Each receptor structure may be selected only once")
    unsupported = sorted(set(receptors) - set(RECEPTOR_STATES))
    if unsupported:
        raise WebsiteError(
            "Unsupported receptor structure(s): "
            + ", ".join(unsupported)
            + ". Available structures: "
            + ", ".join(RECEPTOR_STATES)
        )
    return receptors


def _load_internal_examples(path: Path = DEFAULT_INTERNAL_EXAMPLES) -> dict[str, Any]:
    """Load private examples only when the localhost server explicitly enables them."""
    resolved = path.resolve()
    if not resolved.is_relative_to((REPO_ROOT / "research/data/internal").resolve()):
        raise WebsiteError("The internal-example workbook must remain under research/data/internal")
    if not resolved.is_file():
        raise WebsiteError("The internal-example workbook is unavailable")
    frame = pd.read_excel(resolved, sheet_name="Examples", header=2)
    required = {
        "Example ID",
        "Canonical SMILES",
        "IC50 Relation",
        "hERG IC50 (µM)",
        "Reported Result",
        "Measurement Class",
        "MW (g/mol)",
        "Source Locator",
    }
    missing = required - set(frame)
    if missing:
        raise WebsiteError("The internal-example workbook is missing required columns")
    examples = []
    for row in frame.to_dict(orient="records"):
        relation = str(row["IC50 Relation"]).strip()
        if relation not in {"=", "<", ">"}:
            raise WebsiteError("An internal example has an unsupported measurement relation")
        smiles = str(row["Canonical SMILES"]).strip()
        if _quiet_smiles_molecule(smiles) is None:
            raise WebsiteError("An internal example contains an invalid canonical SMILES")
        value = float(row["hERG IC50 (µM)"])
        molecular_weight = float(row["MW (g/mol)"])
        if not np.isfinite(value) or value <= 0 or not np.isfinite(molecular_weight):
            raise WebsiteError("An internal example contains a non-finite measurement")
        examples.append(
            {
                "id": str(row["Example ID"]).strip(),
                "smiles": smiles,
                "measurement": {
                    "endpoint": "hERG IC50",
                    "relation": relation,
                    "value_um": value,
                    "reported_result": str(row["Reported Result"]).strip(),
                    "measurement_class": str(row["Measurement Class"]).strip(),
                    "unit": "µM",
                },
                "molecular_weight_da": molecular_weight,
                "source_locator": str(row["Source Locator"]).strip(),
            }
        )
    if len(examples) != 10:
        raise WebsiteError("The internal-example workbook must contain exactly ten examples")
    return {
        "schema_version": WEBSITE_SCHEMA,
        "result_kind": "local_private_measured_examples",
        "examples": examples,
        "selection_note": (
            "Illustrative local subset with seven exact and three censored measurements; "
            "not an unbiased benchmark and not an assertion of training status."
        ),
        "privacy": "Local private research data. Do not expose through a public tunnel or static asset.",
    }


def _load_functional_group_surface() -> dict[str, Any]:
    """Fit the deterministic V15 research residual surface from its frozen open-label matrix."""
    global _FUNCTIONAL_GROUP_SURFACE_CACHE  # noqa: PLW0603
    with _FUNCTIONAL_GROUP_SURFACE_LOCK:
        if _FUNCTIONAL_GROUP_SURFACE_CACHE is not None:
            return _FUNCTIONAL_GROUP_SURFACE_CACHE
        registry = v15fg.load_registry()
        features = pd.read_parquet(DEFAULT_V15_FEATURES)
        matrix = pd.read_parquet(
            DEFAULT_V15_MATRIX,
            columns=["sample_id", "campaign", "baseline_pic50", "target_pic50"],
        )
        if len(features) != 1_224 or len(matrix) != 1_224:
            raise WebsiteError("The V15 functional-group research census changed")
        if features.sample_id.duplicated().any() or matrix.sample_id.duplicated().any():
            raise WebsiteError("The V15 functional-group research matrix contains duplicate rows")
        feature_columns = v15fg.feature_columns(
            registry,
            include_functional_groups=True,
            include_interactions=True,
        )
        required = {"sample_id", *feature_columns}
        if required - set(features):
            raise WebsiteError("The V15 functional-group feature contract is incomplete")
        training = matrix.merge(
            features[["sample_id", *feature_columns]],
            on="sample_id",
            validate="one_to_one",
        )
        columns = ["baseline_pic50", *feature_columns]
        values = training[columns].to_numpy(dtype=float)
        target = (
            training.target_pic50.to_numpy(dtype=float)
            - training.baseline_pic50.to_numpy(dtype=float)
        )
        if not np.isfinite(values).all() or not np.isfinite(target).all():
            raise WebsiteError("The V15 functional-group research matrix contains non-finite values")
        campaign_counts = training.campaign.value_counts().to_dict()
        weights = training.campaign.map(
            {key: 1.0 / count for key, count in campaign_counts.items()}
        ).to_numpy(dtype=float, copy=True)
        weights *= len(weights) / weights.sum()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(values)
        model = Ridge(alpha=100.0, random_state=20260831)
        model.fit(scaled, target, sample_weight=weights)
        metrics = _read_v14_json(
            DEFAULT_V15_ROOT / "evidence/aggregate_metrics.json",
            "report_sha256",
        )
        _FUNCTIONAL_GROUP_SURFACE_CACHE = {
            "registry": registry,
            "columns": columns,
            "scaler": scaler,
            "model": model,
            "metrics": metrics,
            "training_rows": int(len(training)),
            "campaigns": int(training.campaign.nunique()),
            "alpha": 100.0,
            "feature_source_sha256": v101._sha(DEFAULT_V15_FEATURES),  # noqa: SLF001
            "matrix_source_sha256": v101._sha(DEFAULT_V15_MATRIX),  # noqa: SLF001
        }
        return _FUNCTIONAL_GROUP_SURFACE_CACHE


def _read_v14_json(path: Path, self_key: str) -> dict[str, Any]:
    """Read a V14 JSON artifact using V14's newline-terminated canonical hash."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise WebsiteError(f"Expected a JSON object: {path.name}")
    expected = value.get(self_key)
    candidate = dict(value)
    candidate.pop(self_key, None)
    canonical = (
        json.dumps(candidate, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    if not isinstance(expected, str) or not hmac.compare_digest(
        expected, hashlib.sha256(canonical).hexdigest()
    ):
        raise WebsiteError(f"The V14 artifact failed integrity validation: {path.name}")
    return value


class WebsitePredictor:
    """Serve fast ligand inference plus an explicit ligand+receptor research path."""

    def __init__(
        self,
        v101_root: Path,
        v102_root: Path,
        v103_root: Path,
        v14_root: Path,
        v141_root: Path,
        v139_root: Path = DEFAULT_V139,
        receptor_cpu: int = 6,
    ):
        self.v101_root = v101_root.resolve()
        self.v102_root = v102_root.resolve()
        self.v103_root = v103_root.resolve()
        self.v139_root = v139_root.resolve()
        self.v14_root = v14_root.resolve()
        self.v141_root = v141_root.resolve()
        self.receptor_cpu = max(1, min(int(receptor_cpu), 6))
        # Validate the immutable releases without loading the slow V9 on-demand
        # 3D conformer ensemble. The fast deployment path below uses the exact
        # V13.9-packaged V10.1 RDKit2D+Morgan models.
        v103._validate(self.v103_root)  # noqa: SLF001
        v103.v102._validate(self.v102_root)  # noqa: SLF001
        v103.v102._validate_v101_source(self.v101_root)  # noqa: SLF001
        self.decision_reference = joblib.load(self.v103_root / "models/decision_reference.joblib")
        property_columns = [
            "structure_id",
            "observed_pic50",
            "rdkit2d__MolWt",
            "rdkit2d__MolLogP",
            "rdkit2d__TPSA",
            "rdkit2d__NumRotatableBonds",
            "rdkit2d__HeavyAtomCount",
        ]
        self.property_reference = pd.read_parquet(DEFAULT_V111_OOF, columns=property_columns)
        if (
            len(self.property_reference) != len(self.decision_reference["structure_ids"])
            or not np.isfinite(self.property_reference.drop(columns="structure_id").to_numpy()).all()
        ):
            raise WebsiteError("The compound-property reference distribution is incomplete")
        self.chemical_feature_factory = ChemicalFeatures.BuildFeatureFactory(
            str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
        )
        self.functional_group_surface = _load_functional_group_surface()
        self.router_info = v101._read_json(  # noqa: SLF001
            self.v101_root / "model_info.json", "model_info_sha256"
        )

        receptor_build = v139.v137._read_json(  # noqa: SLF001
            self.v139_root / "build_report.json", "report_sha256"
        )
        receptor_verification = v139.v137._read_json(  # noqa: SLF001
            self.v139_root / "verification_report.json", "report_sha256"
        )
        receptor_bundle_path = self.v139_root / "models" / v139.BUNDLE_NAME
        if (
            not receptor_bundle_path.is_file()
            or v101._sha(receptor_bundle_path) != receptor_build["bundle_sha256"]  # noqa: SLF001
            or receptor_verification.get("bundle_sha256") != receptor_build["bundle_sha256"]
            or not receptor_verification.get("all_passed")
        ):
            raise WebsiteError("The exact receptor-aware inference bundle failed validation")
        self.receptor_bundle = joblib.load(receptor_bundle_path)
        self.receptor_tools = v139.v13._resolve_toolchain(REPO_ROOT, None)  # noqa: SLF001
        v139._validate_inference_artifacts(  # noqa: SLF001
            self.receptor_bundle, self.v139_root, self.receptor_tools
        )
        self.v13_panel_root = DEFAULT_V13_PANEL.resolve()
        self.receptor_assets = self._validated_receptor_assets()
        self.receptor_atoms = v14._parse_receptor_atoms(  # noqa: SLF001
            self.v139_root / "receptors/8ZYO/8ZYO_prepared.pdb"
        )
        self.v14_report = _read_v14_json(self.v14_root / "analysis_report.json", "report_sha256")
        self.v14_metrics = _read_v14_json(self.v14_root / "evidence/aggregate_metrics.json", "report_sha256")
        self.v14_bootstrap = _read_v14_json(
            self.v14_root / "evidence/paired_scaffold_bootstrap.json", "report_sha256"
        )
        if not hmac.compare_digest(
            self.v14_report["aggregate_metrics_report_sha256"],
            self.v14_metrics["report_sha256"],
        ):
            raise WebsiteError("The V14 report and aggregate metrics do not match")
        if not hmac.compare_digest(
            self.v14_report["bootstrap_report_sha256"],
            self.v14_bootstrap["report_sha256"],
        ):
            raise WebsiteError("The V14 report and bootstrap evidence do not match")
        decision = self.v14_report["promotion_decision"]
        if not decision["ligand_recalibration_research_preview_supported"]:
            raise WebsiteError("The V14 ligand research preview did not pass its release gate")
        if decision["general_receptor_prediction_promotion_supported"]:
            raise WebsiteError("The website policy must be reviewed before receptor promotion")
        v14_model_info = self.v14_report["full_research_models"]["ligand_calibrated"]
        v14_model_path = Path(v14_model_info["path"]).resolve()
        if not v14_model_path.is_relative_to(self.v14_root):
            raise WebsiteError("The V14 ligand research-preview bundle escapes its release root")
        if not v14_model_path.is_file() or v101._sha(v14_model_path) != v14_model_info["sha256"]:  # noqa: SLF001
            raise WebsiteError("The V14 ligand research-preview bundle failed integrity validation")
        self.v14_ligand_preview = joblib.load(v14_model_path)
        self.v141_report = _read_v14_json(self.v141_root / "analysis_report.json", "report_sha256")
        self.v141_metrics = _read_v14_json(
            self.v141_root / "evidence/aggregate_metrics.json", "report_sha256"
        )
        self.v141_bootstrap = _read_v14_json(
            self.v141_root / "evidence/paired_scaffold_bootstrap.json", "report_sha256"
        )
        if not hmac.compare_digest(
            self.v141_report["aggregate_metrics_report_sha256"],
            self.v141_metrics["report_sha256"],
        ) or not hmac.compare_digest(
            self.v141_report["bootstrap_report_sha256"],
            self.v141_bootstrap["report_sha256"],
        ):
            raise WebsiteError("The V14.1 report does not match its evaluation evidence")
        v141_decision = self.v141_report["decision"]
        if v141_decision["general_receptor_prediction_promotion_supported"]:
            raise WebsiteError("The website policy must be reviewed before receptor promotion")
        if v141_decision["best_ligand_regression_surface_posthoc"] != "extratrees_ligand_core":
            raise WebsiteError("The expected V14.1 ligand regression surface changed")
        if v141_decision["best_receptor_classification_surface_posthoc"] != "extratrees_receptor_all_pose":
            raise WebsiteError("The expected V14.1 receptor classification surface changed")
        if v141_decision["best_receptor_regression_surface_posthoc"] != "extratrees_receptor_frozen":
            raise WebsiteError("The expected V14.1 receptor regression surface changed")

        def load_v141_model(name: str) -> dict[str, Any]:
            model_info = self.v141_report["models"][name]
            model_path = Path(model_info["path"]).resolve()
            if not model_path.is_relative_to(self.v141_root):
                raise WebsiteError(f"The V14.1 {name} bundle escapes its release root")
            if not model_path.is_file() or v101._sha(model_path) != model_info["sha256"]:  # noqa: SLF001
                raise WebsiteError(f"The V14.1 {name} bundle failed integrity validation")
            return joblib.load(model_path)

        self.v141_ligand_preview = load_v141_model("extratrees_ligand_core")
        self.v141_receptor_classifier = load_v141_model("extratrees_receptor_all_pose")
        self.v141_receptor_regressor = load_v141_model("extratrees_receptor_frozen")
        self._ligand_lock = threading.Lock()
        self._receptor_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._receptor_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        router_metrics = self.router_info["metrics"]["exact_router"]["winner"]
        ligand_bootstrap = self.v14_bootstrap["comparisons"]["ligand_calibrated"]
        v141_ligand_bootstrap = self.v141_bootstrap["comparisons"]["extratrees_ligand_core"]
        receptor_class_bootstrap = self.v141_bootstrap["comparisons"]["extratrees_receptor_all_pose"][
            "vs_best_ligand_classification"
        ]["classification_delta_balanced_accuracy"]
        receptor_reg_bootstrap = self.v141_bootstrap["comparisons"]["extratrees_receptor_frozen"][
            "vs_best_ligand_regression"
        ]["regression_delta_mae"]
        self.info = {
            "schema_version": WEBSITE_SCHEMA,
            "release": "August 2026 research release",
            "status": "ready",
            "prediction_modes": {
                "ligand": {
                    "label": "Ligand-only",
                    "features": "RDKit2D and Morgan ligand features",
                    "estimated_time": "usually under 1 second",
                    "default": True,
                },
                "ligand_receptor": {
                    "label": "Ligand + receptor",
                    "features": "ligand features plus 8ZYO AutoDock Vina all-pose and residue-contact features",
                    "estimated_time": (
                        "typically 5-45 seconds fresh; large flexible molecules commonly "
                        "20-45 seconds; under 1 second when cached"
                    ),
                    "default": False,
                    "research_only": True,
                },
            },
            "feature_modes": {
                "atomwise": {
                    "label": "Validated atom-wise baseline",
                    "features": "Frozen RDKit2D descriptors plus Morgan atom-environment fingerprints",
                    "default": True,
                    "promotion_status": "live_default",
                },
                "functional_group": {
                    "label": "Functional-group augmented research",
                    "features": (
                        "Governed SMARTS counts/presence, RDKit cLogP, TPSA, MW, HBD/HBA, charge, "
                        "basic-center, ring/linker, and controlled interaction terms layered on the frozen baseline"
                    ),
                    "default": False,
                    "promotion_status": "research_only_failed_promotion",
                },
            },
            "default_policy": {
                "classification": "ligand-only LGBM RDKit2D+Morgan router",
                "literature_ic50": "fast frozen V9/V10 RDKit2D+Morgan regression",
                "direct_curve": "endpoint-specific IC10/IC30/IC50 with deterministic order projection",
                "receptor_features": (
                    "optional research mode uses ligand plus 8ZYO Vina all-pose/receptor-contact features"
                ),
                "research_preview": (
                    "V14 classification plus V14.1 ligand-only regression recalibration; visibly non-default"
                ),
            },
            "task_identity_contract": {
                "literature_ic50_regression": {
                    "task": "quantitative literature IC50/pIC50 regression",
                    "default": True,
                    "substitutable_with_classifier": False,
                },
                "tier_classification": {
                    "task": "Safe/Moderate/Potent categorical classification",
                    "default": True,
                    "substitutable_with_regression": False,
                },
                "direct_functional_curve": {
                    "task": "endpoint-specific IC10/IC30/IC50 regression",
                    "default": True,
                    "order_projection": "IC10 <= IC30 <= IC50 for complete triplets",
                    "substitutable_with_literature_ic50": False,
                },
                "receptor_docking": {
                    "task": "AutoDock Vina scoring, pose, and contact diagnostics",
                    "default": False,
                    "substitutable_with_ic50_or_binding_free_energy": False,
                },
            },
            "feature_role_contract": {
                "prediction_active": (
                    "Frozen RDKit2D+Morgan features used by the default ligand models. The RDKit2D "
                    "contract already includes some fragment-count descriptors."
                ),
                "calculated_property": "Method-labeled RDKit descriptors not asserted as model attribution.",
                "interpretation_only": (
                    "Named functional-group and basic-center detections shown for chemistry context. "
                    "The governed V15 SMARTS augmentation failed held-out promotion and is not active."
                ),
                "docking_diagnostic": (
                    "Per-state Vina scores, poses, and contacts; only sealed 8ZYO features can feed a "
                    "non-default receptor model when its applicability contract passes."
                ),
            },
            "classification_validation": {
                "structures": int(router_metrics["n"]),
                "accuracy": float(router_metrics["accuracy"]),
                "balanced_accuracy": float(router_metrics["balanced_accuracy"]),
                "macro_f1": float(router_metrics["macro_f1"]),
                "potent_to_safe_rate": float(router_metrics["potent_predicted_safe_rate"]),
                "scope": "internal scaffold-held-out broad corpus",
            },
            "v14_research_preview_validation": {
                "version": "V14 classification + V14.1 regression",
                "campaigns": 6,
                "structures": 1_224,
                **self.v14_metrics["metrics"]["ligand_calibrated"]["classification"]["macro_by_campaign"],
                "regression": self.v141_metrics["metrics"]["extratrees_ligand_core"]["regression"][
                    "macro_by_campaign"
                ],
                "delta_vs_frozen": {
                    "classification_balanced_accuracy": ligand_bootstrap[
                        "classification_delta_balanced_accuracy"
                    ],
                    "regression_mae_pic50": v141_ligand_bootstrap["vs_frozen_router"]["regression_delta_mae"],
                },
                "regression_delta_vs_v14_linear": v141_ligand_bootstrap["vs_v14_linear_ligand"][
                    "regression_delta_mae"
                ],
                "model_selection": {
                    "classification": "V14 ligand_calibrated",
                    "regression": "V14.1 extratrees_ligand_core",
                },
                "scope": "retrospective nested leave-one-campaign-out research preview",
                "production_default": False,
            },
            "receptor_research_validation": {
                "version": "V13.9 docking + V14.1 task-specific receptor surfaces",
                "campaigns": 6,
                "structures": 1_224,
                "classification": self.v141_metrics["metrics"]["extratrees_receptor_all_pose"][
                    "classification"
                ]["macro_by_campaign"],
                "regression": self.v141_metrics["metrics"]["extratrees_receptor_frozen"]["regression"][
                    "macro_by_campaign"
                ],
                "delta_vs_best_ligand": {
                    "classification_balanced_accuracy": receptor_class_bootstrap,
                    "regression_mae_pic50": receptor_reg_bootstrap,
                },
                "model_selection": {
                    "classification": "V14.1 extratrees_receptor_all_pose",
                    "regression": "V14.1 extratrees_receptor_frozen",
                },
                "promotion_supported": False,
                "scope": "retrospective nested leave-one-campaign-out research comparison",
            },
            "regression_validation": {
                "literature_ic50_internal": {"n": 18_801, "mae_pic50": 0.43697},
                "literature_ic50_external_2025": {"n": 110, "mae_pic50": 0.48222},
                "literature_ic50_external_2026": {"n": 215, "mae_pic50": 0.37016},
                "direct_curve_internal": {
                    "ic10": {"n": 1_067, "mae_picx": 0.55632},
                    "ic30": {"n": 1_028, "mae_picx": 0.40483},
                    "ic50": {"n": 769, "mae_picx": 0.39519},
                },
            },
            "claim_boundary": {
                "research_use_only": True,
                "clinical_or_regulatory_use": False,
                "prospectively_validated": False,
                "receptor_aware_default": False,
                "receptor_aware_research_mode_available": True,
                "v14_ligand_preview_default": False,
            },
            "capabilities": {
                "functional_group_prediction": {
                    "available": True,
                    "request_field": "feature_mode",
                    "choices": ["atomwise", "functional_group"],
                    "default": "atomwise",
                    "research_surface": "V15 fg_global_residual_ridge full-open-label fit",
                    "training_rows": self.functional_group_surface["training_rows"],
                    "campaigns": self.functional_group_surface["campaigns"],
                    "held_out_macro_campaign_mae_pic50": self.functional_group_surface["metrics"][
                        "metrics"
                    ]["fg_global_residual_ridge"]["regression"]["macro_campaign_mae"],
                    "delta_mae_vs_deployed": self.functional_group_surface["metrics"]["bootstrap"][
                        "all_ablation_comparisons"
                    ]["fg_global_residual_ridge"]["vs_frozen_v9_v10_baseline"],
                    "promotion_supported": False,
                    "claim_boundary": (
                        "The functional-group surface is a real model input option, but it did not show "
                        "a stable held-out improvement and remains a label-open retrospective comparison."
                    ),
                },
                "multi_receptor_diagnostics": {
                    "available": True,
                    "request_field": "receptor_ids",
                    "accepted_legacy_aliases": ["receptors"],
                    "minimum_selected": 1,
                    "maximum_selected": MAX_RECEPTOR_SELECTION,
                    "default_selection": ["8ZYO"],
                    "available_receptors": [
                        {
                            "id": receptor_id,
                            **RECEPTOR_STATES[receptor_id],
                            "prepared_assets_verified": True,
                        }
                        for receptor_id in RECEPTOR_STATES
                    ],
                    "model_driving_receptors": ["8ZYO"],
                    "selection_semantics": (
                        "Every selected receptor is docked with AutoDock Vina. Only 8ZYO feeds the "
                        "current frozen receptor-aware ML models; every other state is a docking-only diagnostic."
                    ),
                    "runtime_note": "Fresh runtime scales approximately with the number of selected receptors.",
                    "claim_boundary": (
                        "Cross-state Vina scores are scoring-function observables, not binding free energies, "
                        "state-population estimates, or independently validated state-specific hERG predictions."
                    ),
                },
                "experimental_mw_sensitivity": {
                    "available": True,
                    "request_field": "mw_delta_stress_test",
                    "workflow": "manual parent-candidate comparison",
                    "default_enabled": False,
                    "primary_prediction_unchanged": True,
                    "formula": (
                        "stress multiplier = max(1, retrospective activity-cliff multiplier × "
                        "pair mean MW / training-reference median MW); applied only to candidate-minus-parent pIC50"
                    ),
                    "reference_median_mw_da": MW_STRESS_REFERENCE_MEDIAN_MW_DA,
                    "retrospective_baseline_multiplier": MW_STRESS_BASELINE_MULTIPLIER,
                    "provenance": dict(MW_STRESS_AUDIT_PROVENANCE),
                    "claim_boundary": (
                        "Opt-in retrospective scenario only; not a recalibrated prediction. It preserves "
                        "the model's direction and can amplify a wrong direction. Evidence above 700 Da is sparse."
                    ),
                },
                "experimental_candidate_generation": {
                    "available": True,
                    "endpoint": "/api/generate",
                    "default_candidate_count": MAX_GENERATED_CANDIDATES,
                    "maximum_candidate_count": MAX_GENERATED_CANDIDATES,
                    "prediction_mode": "ligand-only frozen default stack",
                    "ranking_inputs": [
                        "predicted IC50 gain",
                        "global 90% interval width",
                        "training-neighbor applicability",
                        "parent similarity",
                        "bounded property changes",
                    ],
                    "claim_boundary": (
                        "Generated structures are deterministic bounded enumerations, not synthesis "
                        "recommendations or activity claims. Ranking is prioritization, not confidence."
                    ),
                },
            },
        }

    def _validated_receptor_assets(self) -> dict[str, dict[str, Any]]:
        """Resolve prepared receptor files and verify them against the frozen V13 inventory."""
        report = v139.v13._read_self_hashed_json(  # noqa: SLF001
            self.v13_panel_root / "receptors/receptor_preparation.json", "report_sha256"
        )
        if int(report.get("receptor_count", 0)) != len(RECEPTOR_STATES):
            raise WebsiteError("The prepared multi-receptor inventory is incomplete")
        inventory = pd.read_parquet(self.v13_panel_root / "receptors/receptor_preparation.parquet").set_index(
            "pdb_id"
        )
        if set(RECEPTOR_STATES) - set(inventory.index.astype(str)):
            raise WebsiteError("The prepared receptor inventory is missing an advertised state")

        assets: dict[str, dict[str, Any]] = {}
        for receptor_id, metadata in RECEPTOR_STATES.items():
            root = (
                self.v139_root / "receptors" / receptor_id
                if receptor_id == "8ZYO"
                else self.v13_panel_root / "receptors" / receptor_id
            )
            paths = {
                "pdb": root / f"{receptor_id}_prepared.pdb",
                "pdbqt": root / f"{receptor_id}_prepared.pdbqt",
                "box": root / f"{receptor_id}_prepared.box.txt",
                "preparation": root / f"{receptor_id}_prepared.json",
            }
            if any(not path.is_file() or path.stat().st_size == 0 for path in paths.values()):
                raise WebsiteError(f"Prepared receptor assets are incomplete for {receptor_id}")
            expected_pdbqt_sha = str(inventory.loc[receptor_id, "pdbqt_sha256"])
            actual_pdbqt_sha = v101._sha(paths["pdbqt"])  # noqa: SLF001
            if not hmac.compare_digest(expected_pdbqt_sha, actual_pdbqt_sha):
                raise WebsiteError(f"Prepared receptor integrity validation failed for {receptor_id}")
            assets[receptor_id] = {
                **metadata,
                **paths,
                "pdbqt_sha256": actual_pdbqt_sha,
                "preparation_ph": float(report["ph"]),
                "preparation_force_field": str(report["force_field"]),
            }
        return assets

    def properties(self, smiles: str) -> dict[str, Any]:
        """Calculate structure properties without running prediction or docking models."""
        smiles = str(smiles).strip()
        if not smiles:
            raise WebsiteError("Enter a SMILES string")
        if len(smiles) > MAX_SMILES_LENGTH:
            raise WebsiteError(f"SMILES must be {MAX_SMILES_LENGTH:,} characters or fewer")
        if _quiet_smiles_molecule(smiles) is None:
            raise WebsiteError("The SMILES string could not be parsed")
        started = time.perf_counter()
        registry = v139._input_registry([smiles])  # noqa: SLF001
        canonical = str(registry.iloc[0].standardized_smiles)
        molecule = Chem.MolFromSmiles(canonical)
        if molecule is None:
            raise WebsiteError("The standardized structure could not be parsed")

        heavy_atoms = int(molecule.GetNumHeavyAtoms())
        aromatic_atoms = sum(int(atom.GetIsAromatic()) for atom in molecule.GetAtoms())
        formal_charge = int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()))
        ring_sizes = [len(ring) for ring in molecule.GetRingInfo().AtomRings()]
        descriptors = {
            "molecular_weight": {
                "label": "Molecular weight",
                "value": float(Descriptors.MolWt(molecule)),
                "unit": "Da",
                "method": "RDKit average molecular weight",
            },
            "exact_mass": {
                "label": "Exact mass",
                "value": float(Descriptors.ExactMolWt(molecule)),
                "unit": "Da",
                "method": "RDKit monoisotopic exact mass",
            },
            "clogp": {
                "label": "cLogP",
                "value": float(Crippen.MolLogP(molecule)),
                "unit": "",
                "method": "RDKit Wildman–Crippen MolLogP",
            },
            "tpsa": {
                "label": "Topological PSA",
                "value": float(rdMolDescriptors.CalcTPSA(molecule)),
                "unit": "Å²",
                "method": "RDKit Ertl-style TPSA",
            },
            "hbd": {
                "label": "H-bond donors",
                "value": int(Lipinski.NumHDonors(molecule)),
                "unit": "",
                "method": "RDKit Lipinski definition",
            },
            "hba": {
                "label": "H-bond acceptors",
                "value": int(Lipinski.NumHAcceptors(molecule)),
                "unit": "",
                "method": "RDKit Lipinski definition",
            },
            "rotatable_bonds": {
                "label": "Rotatable bonds",
                "value": int(Lipinski.NumRotatableBonds(molecule)),
                "unit": "",
                "method": "RDKit Lipinski definition",
            },
            "heavy_atoms": {
                "label": "Heavy atoms",
                "value": heavy_atoms,
                "unit": "",
                "method": "RDKit atom count excluding hydrogen",
            },
            "hetero_atoms": {
                "label": "Hetero atoms",
                "value": int(Lipinski.NumHeteroatoms(molecule)),
                "unit": "",
                "method": "RDKit Lipinski definition",
            },
            "ring_count": {
                "label": "Rings",
                "value": int(rdMolDescriptors.CalcNumRings(molecule)),
                "unit": "",
                "method": "RDKit ring count",
            },
            "aromatic_rings": {
                "label": "Aromatic rings",
                "value": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
                "unit": "",
                "method": "RDKit aromatic ring count",
            },
            "largest_ring": {
                "label": "Largest ring",
                "value": int(max(ring_sizes, default=0)),
                "unit": "atoms",
                "method": "Largest RDKit ring-info atom cycle",
            },
            "fraction_csp3": {
                "label": "Fraction Csp3",
                "value": float(rdMolDescriptors.CalcFractionCSP3(molecule)),
                "unit": "",
                "method": "RDKit fraction of sp3 carbon atoms",
            },
            "aromatic_heavy_fraction": {
                "label": "Aromatic heavy-atom fraction",
                "value": float(aromatic_atoms / heavy_atoms) if heavy_atoms else 0.0,
                "unit": "",
                "method": "Aromatic atoms divided by heavy atoms",
            },
            "formal_charge": {
                "label": "Formal charge",
                "value": formal_charge,
                "unit": "",
                "method": "Sum of RDKit atomic formal charges",
            },
            "halogen_atoms": {
                "label": "Halogen atoms",
                "value": int(sum(atom.GetAtomicNum() in {9, 17, 35, 53} for atom in molecule.GetAtoms())),
                "unit": "",
                "method": "F, Cl, Br, and I atom count",
            },
            "stereocenters": {
                "label": "Specified or possible stereocenters",
                "value": int(len(Chem.FindMolChiralCenters(molecule, includeUnassigned=True))),
                "unit": "",
                "method": "RDKit chiral-center detection",
            },
            "qed": {
                "label": "QED",
                "value": float(QED.qed(molecule)),
                "unit": "",
                "method": "RDKit quantitative estimate of drug-likeness",
            },
        }
        for descriptor in descriptors.values():
            descriptor["feature_role"] = "calculated_property"

        fragment_definitions = {
            "aliphatic_alcohol": ("Aliphatic alcohol", Fragments.fr_Al_OH),
            "phenol": ("Phenol", Fragments.fr_Ar_OH),
            "ether": ("Ether", Fragments.fr_ether),
            "ester": ("Ester", Fragments.fr_ester),
            "amide": ("Amide", Fragments.fr_amide),
            "urea": ("Urea", Fragments.fr_urea),
            "carboxylic_acid": ("Carboxylic acid", Fragments.fr_COO),
            "nitrile": ("Nitrile", Fragments.fr_nitrile),
            "sulfone": ("Sulfone", Fragments.fr_sulfone),
            "sulfonamide": ("Sulfonamide", Fragments.fr_sulfonamd),
            "alkyl_halide": ("Alkyl halide", Fragments.fr_alkyl_halide),
            "aromatic_nitrogen": ("Aromatic nitrogen", Fragments.fr_Ar_N),
            "aromatic_nh": ("Aromatic N–H", Fragments.fr_Ar_NH),
            "quaternary_nitrogen": ("Quaternary nitrogen", Fragments.fr_quatN),
            "guanidine": ("Guanidine", Fragments.fr_guanido),
        }
        functional_groups = [
            {
                "key": key,
                "label": label,
                "count": int(function(molecule)),
                "feature_role": "interpretation_only",
            }
            for key, (label, function) in fragment_definitions.items()
            if int(function(molecule)) > 0
        ]

        ionization_features = self.chemical_feature_factory.GetFeaturesForMol(molecule)
        positive_ionizable = [
            feature for feature in ionization_features if feature.GetFamily() == "PosIonizable"
        ]
        negative_ionizable = [
            feature for feature in ionization_features if feature.GetFamily() == "NegIonizable"
        ]
        permanent_positive_atoms = sum(int(atom.GetFormalCharge() > 0) for atom in molecule.GetAtoms())
        permanent_negative_atoms = sum(int(atom.GetFormalCharge() < 0) for atom in molecule.GetAtoms())
        aromatic_nitrogens = sum(
            int(atom.GetAtomicNum() == 7 and atom.GetIsAromatic()) for atom in molecule.GetAtoms()
        )
        if permanent_positive_atoms:
            basicity_class = "Permanently or explicitly cationic"
        elif positive_ionizable:
            basicity_class = "Likely protonatable basic center present"
        elif aromatic_nitrogens:
            basicity_class = "Heteroaromatic nitrogen present; basicity is context-dependent"
        else:
            basicity_class = "No positive-ionizable center detected by the structural feature rules"

        clogp = float(descriptors["clogp"]["value"])
        aromatic_rings = int(descriptors["aromatic_rings"]["value"])
        tpsa = float(descriptors["tpsa"]["value"])
        herg_relationships = []
        if positive_ionizable or permanent_positive_atoms:
            herg_relationships.append(
                "A positive-ionizable or cationic center can support interactions in the hERG aromatic "
                "cage; the effect depends on protonation, accessibility, and geometry."
            )
        if clogp >= 3.0:
            herg_relationships.append(
                "Elevated cLogP can increase membrane partitioning and is frequently associated with hERG "
                "liability, but it is not sufficient by itself."
            )
        if aromatic_rings >= 2:
            herg_relationships.append(
                "Multiple aromatic rings can provide hydrophobic and aromatic interaction surfaces relevant "
                "to Y652/F656 engagement."
            )
        if tpsa >= 120.0:
            herg_relationships.append(
                "High TPSA may reduce passive membrane access, although intramolecular shielding and active "
                "transport can make this relationship non-monotonic for large molecules."
            )
        if permanent_positive_atoms:
            herg_relationships.append(
                "Permanent charge may reduce passive permeability while still permitting strong pore "
                "interactions once the compound reaches the intracellular binding region."
            )
        if not herg_relationships:
            herg_relationships.append(
                "No strong basicity/lipophilicity/aromaticity pattern was triggered by the current structural rules."
            )
        basicity = {
            "classification": basicity_class,
            "positive_ionizable_centers": int(len(positive_ionizable)),
            "negative_ionizable_centers": int(len(negative_ionizable)),
            "permanent_positive_atoms": int(permanent_positive_atoms),
            "permanent_negative_atoms": int(permanent_negative_atoms),
            "aromatic_nitrogens": int(aromatic_nitrogens),
            "cationic_lipophilic_aromatic_pattern": bool(
                (positive_ionizable or permanent_positive_atoms) and clogp >= 3.0 and aromatic_rings >= 2
            ),
            "herg_relationship_context": herg_relationships,
            "method": "RDKit BaseFeatures ionizable-group rules plus charge and aromaticity counts",
            "feature_role": "interpretation_only",
            "pka_calculated": False,
            "boundary": (
                "This is a structural basicity/ionization proxy, not a predicted or experimental pKa. "
                "Protonation depends on microstate, solvent, and pH."
            ),
        }

        reference_map = {
            "molecular_weight": "rdkit2d__MolWt",
            "clogp": "rdkit2d__MolLogP",
            "tpsa": "rdkit2d__TPSA",
            "rotatable_bonds": "rdkit2d__NumRotatableBonds",
            "heavy_atoms": "rdkit2d__HeavyAtomCount",
        }
        global_pic50 = float(self.property_reference.observed_pic50.median())
        associations = []
        for key, column in reference_map.items():
            value = float(descriptors[key]["value"])
            reference_values = self.property_reference[column].to_numpy(dtype=float)
            percentile = float(np.mean(reference_values <= value))
            lower = float(np.quantile(reference_values, max(0.0, percentile - 0.05)))
            upper = float(np.quantile(reference_values, min(1.0, percentile + 0.05)))
            mask = (reference_values >= lower) & (reference_values <= upper)
            local_pic50 = float(self.property_reference.loc[mask, "observed_pic50"].median())
            delta = local_pic50 - global_pic50
            if delta >= 0.20:
                direction = (
                    "higher observed hERG inhibitory potency (greater liability) in this property band"
                )
            elif delta <= -0.20:
                direction = "lower observed hERG inhibitory potency (lower liability) in this property band"
            else:
                direction = "observed hERG inhibitory potency near the broad-corpus median"
            associations.append(
                {
                    "key": key,
                    "label": descriptors[key]["label"],
                    "percentile": percentile,
                    "nearby_band_median_pic50": local_pic50,
                    "delta_from_global_median_pic50": delta,
                    "interpretation": direction,
                }
            )

        return {
            "schema_version": WEBSITE_SCHEMA,
            "analysis_mode": "properties_only",
            "submitted_smiles": smiles,
            "smiles": canonical,
            "standardization": {
                "result_kind": "calculated_structure_preparation",
                "model_used_smiles": canonical,
                "method": "Project standardization using RDKit parent-fragment selection and Uncharger",
                "disclosure": (
                    "Descriptors and structural rules describe the standardized parent used by the model. "
                    "RDKit cleanup, salt/fragment-parent selection, and Uncharger processing can change "
                    "the submitted representation; formal charges may be neutralized. Tautomers are not "
                    "canonicalized in this contract. "
                    "No pH-specific protonation enumeration is performed for ligand-only inference. "
                    "Stereochemistry present in the standardized SMILES is retained, while unspecified "
                    "stereocenters remain unspecified. Aromaticity and implicit hydrogens follow RDKit's "
                    "sanitization model, so other chemistry tools may display or calculate different values."
                ),
                "effects": {
                    "salts_and_fragments": "RDKit FragmentParent selects the parent representation",
                    "neutralization": "RDKit Uncharger may alter formal charges where its rules apply",
                    "protonation": "no pH-specific ligand-only microstate enumeration",
                    "tautomers": "not canonicalized",
                    "stereochemistry": "specified centers retained; unspecified centers remain unspecified",
                    "aromaticity": "RDKit aromaticity perception",
                    "implicit_hydrogens": "RDKit valence and implicit-hydrogen model",
                },
            },
            "structure_depiction": _structure_depiction(molecule),
            "descriptors": descriptors,
            "functional_groups": functional_groups,
            "basicity": basicity,
            "training_associations": associations,
            "docking_eligibility": {
                # Legacy keys remain for backward-compatible clients. They describe the
                # frozen receptor-model domain, not whether Vina can technically run.
                "eligible_under_current_receptor_model": bool(registry.iloc[0].docking_eligible),
                "reason": str(registry.iloc[0].docking_exclusion_reason),
                **_docking_execution_contract(registry),
                "contract_note": (
                    "A structure may be docked as a diagnostic even when the frozen receptor-aware "
                    "ML surface is out of domain. Diagnostic docking never changes the ligand-only default."
                ),
            },
            "runtime_seconds": time.perf_counter() - started,
            "claim_boundary": (
                "Training-property associations are univariate retrospective context, not causal effects "
                "and not model feature attributions. No hERG prediction or docking was run. Values describe "
                "the standardized parent, not necessarily the submitted salt or protonation microstate."
            ),
        }

    @staticmethod
    def _classification(ligand: pd.DataFrame) -> dict[str, Any]:
        row = ligand.iloc[0]
        probabilities = {
            name: _finite_probability(row[f"lgbm_rdkit2d_morgan__probability_{name.lower()}"])
            for name in CLASS_NAMES
        }
        tier = max(probabilities, key=probabilities.__getitem__)
        return {
            "tier": tier,
            "probabilities": probabilities,
            "model": "ligand-only LGBM RDKit2D+Morgan",
            "decision_role": "default classification model",
            "tier_definition": {
                "Safe": "IC50 >30 µM",
                "Moderate": "1 ≤ IC50 ≤ 30 µM",
                "Potent": "IC50 <1 µM",
            },
        }

    def _applicability(self, canonical_smiles: str) -> dict[str, Any]:
        columns = [f"morgan__{index:04d}" for index in range(v103.v102.MORGAN_BITS)]
        frame, molecule = v103.v102.v10._feature_frame(canonical_smiles, columns)  # noqa: SLF001
        query = frame[columns].fillna(0).to_numpy(dtype=np.uint8)[0]
        packed_query = np.packbits(query)
        lookup = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)
        intersections = lookup[np.bitwise_and(self.decision_reference["packed_morgan"], packed_query)].sum(
            axis=1
        )
        union = (
            np.asarray(self.decision_reference["bit_counts"], dtype=np.int32)
            + int(query.sum())
            - intersections
        )
        similarities = np.divide(
            intersections,
            union,
            out=np.zeros_like(intersections, dtype=float),
            where=union > 0,
        )
        order = np.argsort(similarities)[::-1]
        nearest = order[:5]
        local = order[: min(v103.LOCAL_COHORT_SIZE, len(order))]
        residuals = np.asarray(self.decision_reference["signed_oof_residuals"], dtype=float)
        canonical = v103.Chem.MolToSmiles(molecule, isomericSmiles=True)
        exact_candidates = np.flatnonzero(np.isclose(similarities, 1.0, atol=1e-12))
        exact_overlap = any(
            str(self.decision_reference["smiles"][index]) == canonical for index in exact_candidates
        )
        maximum = float(similarities[nearest[0]])
        if maximum >= 0.70:
            label = "High analogue support"
            interpretation = "A close training analogue exists; local interpolation is more plausible."
        elif maximum >= 0.50:
            label = "Moderate analogue support"
            interpretation = "Related chemistry exists, but this is still a scaffold-sensitive prediction."
        else:
            label = "Extrapolative chemistry"
            interpretation = "No training compound reaches Tanimoto 0.50; treat the estimate as high risk."
        return {
            "maximum_train_tanimoto": maximum,
            "analog_count_ge_0p5": int(np.sum(similarities >= 0.5)),
            "exact_training_overlap": bool(exact_overlap),
            "domain_label": label,
            "interpretation": interpretation,
            "local_oof_error_diagnostic": {
                "cohort_size": int(len(local)),
                "selection": f"{len(local)} most similar exact-training structures",
                "median_tanimoto": float(np.median(similarities[local])),
                "minimum_tanimoto": float(np.min(similarities[local])),
                "retrospective_mae_pic50": float(np.mean(np.abs(residuals[local]))),
                "retrospective_median_absolute_error_pic50": float(np.median(np.abs(residuals[local]))),
                "retrospective_q90_absolute_error_pic50": float(np.quantile(np.abs(residuals[local]), 0.90)),
                "scope": (
                    "descriptive analogue-neighborhood OOF performance; not a calibrated per-query interval"
                ),
            },
            "nearest_analog_records_disclosed": False,
            "analogue_detail_disclosure": (
                "Training-record identifiers, structures, measurements, and per-record errors are "
                "withheld. Aggregate similarity and local held-out error diagnostics remain available."
            ),
        }

    def _ligand_prediction(self, smiles: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
        registry = v139._input_registry([smiles])  # noqa: SLF001
        ligand = v139._ligand_predictions(self.receptor_bundle, registry)  # noqa: SLF001
        row = ligand.iloc[0]
        canonical = str(row.standardized_smiles)
        pic50 = float(row.v9_mixed_ic50_predicted_pic50)
        interval_lower = float(row.v9_mixed_ic50_interval90_lower_pic50)
        interval_upper = float(row.v9_mixed_ic50_interval90_upper_pic50)
        residuals = np.asarray(self.decision_reference["signed_oof_residuals"], dtype=float)
        tier_probabilities = {
            "Safe": float(np.mean(pic50 + residuals < v103.v102.SAFE_PIC50)),
            "Moderate": float(
                np.mean(
                    (pic50 + residuals >= v103.v102.SAFE_PIC50)
                    & (pic50 + residuals <= v103.v102.POTENT_PIC50)
                )
            ),
            "Potent": float(np.mean(pic50 + residuals > v103.v102.POTENT_PIC50)),
        }
        classification = self._classification(ligand)
        applicability = self._applicability(canonical)
        confidence = v103._decision_confidence(  # noqa: SLF001
            maximum_similarity=float(applicability["maximum_train_tanimoto"]),
            exact_overlap=bool(applicability["exact_training_overlap"]),
            interval_lower_pic50=interval_lower,
            interval_upper_pic50=interval_upper,
        )
        direct = {
            "ic10_um": float(row.empirical_ic10_predicted_um),
            "ic30_um": float(row.empirical_ic30_predicted_um),
            "ic50_um": float(row.empirical_ic50_predicted_um),
            "raw_predictions_um": {
                name: float(10.0 ** (6.0 - row[f"empirical_{name}_raw_predicted_picx"]))
                for name in ("ic10", "ic30", "ic50")
            },
            "interval90_um": {
                name: {
                    "lower": float(row[f"empirical_{name}_interval90_lower_um"]),
                    "upper": float(row[f"empirical_{name}_interval90_upper_um"]),
                }
                for name in ("ic10", "ic30", "ic50")
            },
            "ordering_adjusted": bool(row.endpoint_order_projection_applied),
            "ordering_contract": "IC10 <= IC30 <= IC50",
            "scope": "one direct 20-concentration human hERG functional qHTS assay",
        }
        direct["cross_assay_consistency"] = v103._cross_assay_consistency(  # noqa: SLF001
            float(10.0 ** (6.0 - pic50)), direct["ic50_um"]
        )
        direct["internal_endpoint_metrics"] = {
            "IC10": {"labels": 1_067, "mae": 0.5563188034620088},
            "IC30": {"labels": 1_028, "mae": 0.40483151933562134},
            "IC50": {"labels": 769, "mae": 0.3951873453940611},
        }
        molecule = v103.Chem.MolFromSmiles(canonical)
        if molecule is None:
            raise WebsiteError("The standardized structure could not be parsed")
        result = {
            "smiles": canonical,
            "structure_depiction": _structure_depiction(molecule),
            "main_ic50": {
                "pic50": pic50,
                "ic50_um": float(10.0 ** (6.0 - pic50)),
                "tier": v103.v102._tier(pic50),  # noqa: SLF001
                "tier_definition": {
                    "Safe": "IC50 >30 µM",
                    "Moderate": "1 ≤ IC50 ≤ 30 µM",
                    "Potent": "IC50 <1 µM",
                },
                "tier_probabilities": tier_probabilities,
                "interval90_pic50": {"lower": interval_lower, "upper": interval_upper},
                "interval90_um": {
                    "lower": float(row.v9_mixed_ic50_interval90_lower_um),
                    "upper": float(row.v9_mixed_ic50_interval90_upper_um),
                },
                "model_source": "fast frozen V9/V10 RDKit2D+Morgan regression",
                "decision_confidence": confidence,
            },
            "classification": classification,
            "direct_functional_curve": direct,
            "broad_fixed_dose": {
                "probability_screen_active_at_46um": float(row.broad_fixed_dose_probability_active_at_46um),
                "scope": "auxiliary 46 µM fixed-dose screen; not an IC50",
            },
            "applicability": applicability,
            "chemistry": {
                "molecular_weight": float(v103.v102.Descriptors.MolWt(molecule)),
                "clogp": float(v103.v102.Crippen.MolLogP(molecule)),
                "tpsa": float(v103.v102.rdMolDescriptors.CalcTPSA(molecule)),
                "hbd": int(v103.v102.Lipinski.NumHDonors(molecule)),
                "hba": int(v103.v102.Lipinski.NumHAcceptors(molecule)),
                "rotatable_bonds": int(v103.v102.Lipinski.NumRotatableBonds(molecule)),
            },
            "decision_reliability": {
                "confidence": confidence,
                "exact_training_overlap": bool(applicability["exact_training_overlap"]),
                "cross_assay_consistency": direct["cross_assay_consistency"],
            },
        }
        regression_point_tier = str(result["main_ic50"]["tier"])
        regression_probability_tier = max(tier_probabilities, key=tier_probabilities.__getitem__)
        classifier_tier = str(classification["tier"])
        result["classification_regression_consistency"] = {
            "result_kind": "model_output_diagnostic",
            "standalone_classifier": {
                "top_tier": classifier_tier,
                "probabilities": dict(classification["probabilities"]),
                "model": str(classification["model"]),
            },
            "regression_point_estimate": {
                "tier": regression_point_tier,
                "predicted_ic50_um": float(result["main_ic50"]["ic50_um"]),
                "model": str(result["main_ic50"]["model_source"]),
            },
            "regression_residual_distribution": {
                "top_tier": regression_probability_tier,
                "tier_probabilities": dict(tier_probabilities),
                "method": "frozen regression estimate plus the signed OOF residual distribution",
            },
            "classifier_vs_regression_point_tier_disagreement": bool(
                classifier_tier != regression_point_tier
            ),
            "classifier_vs_regression_distribution_top_tier_disagreement": bool(
                classifier_tier != regression_probability_tier
            ),
            "interpretation": (
                "The standalone classifier and IC50 regression are separately trained statistical "
                "outputs. Their probabilities and point estimate need not map one-to-one; any tier "
                "disagreement is exposed here rather than cosmetically reconciled."
            ),
        }
        result["research_preview"] = self._v14_research_preview(classification, pic50)
        return result, registry, ligand

    def _functional_group_prediction(
        self,
        canonical_smiles: str,
        baseline_pic50: float,
    ) -> dict[str, Any]:
        """Run the real, non-promoted V15 functional-group residual surface."""
        surface = self.functional_group_surface
        row = v15fg.build_feature_row(canonical_smiles, surface["registry"])
        feature_values = {"baseline_pic50": float(baseline_pic50), **row}
        values = np.asarray(
            [[feature_values[column] for column in surface["columns"]]],
            dtype=float,
        )
        scaled = surface["scaler"].transform(values)
        residual = float(surface["model"].predict(scaled)[0])
        predicted_pic50 = float(baseline_pic50 + residual)
        predicted_ic50 = float(10.0 ** (6.0 - predicted_pic50))
        coefficients = np.asarray(surface["model"].coef_, dtype=float)
        contributions = scaled[0] * coefficients
        top_indices = np.argsort(np.abs(contributions))[::-1][:10]

        rule_labels = {
            str(rule["key"]): str(rule["label"]) for rule in surface["registry"]["rules"]
        }

        def display_name(name: str) -> str:
            if name == "baseline_pic50":
                return "Frozen atom-wise baseline pIC50"
            if name.startswith("fg_count__"):
                return rule_labels.get(name.removeprefix("fg_count__"), name)
            if name.startswith("fg_present__"):
                return rule_labels.get(name.removeprefix("fg_present__"), name) + " present"
            return name.replace("_x_", " × ").replace("_", " ")

        active_groups = []
        for rule in surface["registry"]["rules"]:
            count = int(row[f"fg_count__{rule['key']}"])
            if count:
                active_groups.append(
                    {
                        "key": str(rule["key"]),
                        "label": str(rule["label"]),
                        "category": str(rule["category"]),
                        "count": count,
                        "feature_role": "prediction_active_in_selected_research_surface",
                    }
                )
        metrics = surface["metrics"]
        held_out = metrics["metrics"]["fg_global_residual_ridge"]
        comparison = metrics["bootstrap"]["all_ablation_comparisons"][
            "fg_global_residual_ridge"
        ]["vs_frozen_v9_v10_baseline"]
        return {
            "status": "research_only_failed_promotion",
            "selected_surface": "V15 fg_global_residual_ridge",
            "regression": {
                "pic50": predicted_pic50,
                "ic50_um": predicted_ic50,
                "tier": v103.v102._tier(predicted_pic50),  # noqa: SLF001
                "tier_definition": {
                    "Safe": "IC50 >30 µM",
                    "Moderate": "1 ≤ IC50 ≤ 30 µM",
                    "Potent": "IC50 <1 µM",
                },
                "delta_vs_atomwise_pic50": residual,
                "fold_vs_atomwise_ic50": float(
                    predicted_ic50 / (10.0 ** (6.0 - float(baseline_pic50)))
                ),
                "interval90_um": None,
                "decision_confidence": {
                    "label": "Research-only surface",
                    "explanation": (
                        "No query-calibrated uncertainty interval was validated for this functional-group surface."
                    ),
                },
                "model_source": "V15 governed SMARTS + physicochemical global residual Ridge",
            },
            "feature_contract": {
                "registry_version": surface["registry"]["registry_version"],
                "feature_count": len(surface["columns"]),
                "features": (
                    "Governed SMARTS functional-group counts/presence; RDKit MW, cLogP, TPSA, HBD/HBA, "
                    "formal charge, rotors, heavy atoms, ionizable-center proxies, ring/linker descriptors, "
                    "and controlled interaction terms."
                ),
                "active_functional_groups": active_groups,
                "top_linear_contributions_pic50": [
                    {
                        "feature": surface["columns"][int(index)],
                        "label": display_name(surface["columns"][int(index)]),
                        "contribution_pic50": float(contributions[int(index)]),
                        "feature_role": "prediction_active_in_selected_research_surface",
                    }
                    for index in top_indices
                ],
                "intercept_pic50": float(surface["model"].intercept_),
                "count_contract": surface["registry"]["count_contract"],
            },
            "validation": {
                "rows": surface["training_rows"],
                "campaigns": surface["campaigns"],
                "nested_leave_one_campaign_out_macro_mae_pic50": held_out["regression"][
                    "macro_campaign_mae"
                ],
                "nested_leave_one_campaign_out_overall_mae_pic50": held_out["regression"][
                    "overall"
                ]["mae"],
                "delta_mae_vs_deployed": comparison,
                "production_or_default_promotion_supported": False,
            },
            "provenance": {
                "feature_source_sha256": surface["feature_source_sha256"],
                "matrix_source_sha256": surface["matrix_source_sha256"],
                "full_open_label_fit_alpha": surface["alpha"],
            },
            "claim_boundary": (
                "This is an actual functional-group-augmented prediction, but the governed V15 study "
                "did not demonstrate stable held-out improvement. It is label-open, retrospective, "
                "non-default, and has no independently calibrated per-query interval."
            ),
        }

    def _v14_research_preview(
        self,
        default_classification: dict[str, Any],
        baseline_pic50: float,
    ) -> dict[str, Any]:
        probabilities = default_classification["probabilities"]
        values = np.asarray([probabilities[name] for name in CLASS_NAMES], dtype=float)
        clipped = np.clip(values, 1e-9, 1.0)
        features = pd.DataFrame(
            [
                {
                    "ligand_log_safe_vs_moderate": float(
                        np.log(np.clip(values[0], 1e-6, 1.0) / np.clip(values[1], 1e-6, 1.0))
                    ),
                    "ligand_log_potent_vs_moderate": float(
                        np.log(np.clip(values[2], 1e-6, 1.0) / np.clip(values[1], 1e-6, 1.0))
                    ),
                    "baseline_pic50": float(baseline_pic50),
                    "ligand_router_entropy": float(-np.sum(clipped * np.log(clipped))),
                }
            ]
        )
        expected_classifier = list(self.v14_ligand_preview["features"])
        expected_regressor = list(self.v141_ligand_preview["features"])
        if list(features) != expected_classifier or list(features) != expected_regressor:
            raise WebsiteError("The V14 research-preview feature contract does not match the website")
        raw = self.v14_ligand_preview["classifier"].predict_proba(features)[0]
        classes = [
            int(value) for value in self.v14_ligand_preview["classifier"].named_steps["model"].classes_
        ]
        by_index = {
            class_index: _finite_probability(raw[position]) for position, class_index in enumerate(classes)
        }
        preview_probabilities = {CLASS_NAMES[index]: by_index[index] for index in range(3)}
        preview_tier = max(preview_probabilities, key=preview_probabilities.__getitem__)
        residual = float(self.v141_ligand_preview["residual_regressor"].predict(features)[0])
        preview_pic50 = float(baseline_pic50 + residual)
        return {
            "status": "research_preview_non_default",
            "classification": {
                "tier": preview_tier,
                "probabilities": preview_probabilities,
                "model": "V14 ligand-only cross-campaign recalibration",
            },
            "literature_ic50": {
                "pic50": preview_pic50,
                "ic50_um": float(10.0 ** (6.0 - preview_pic50)),
                "model": "V14.1 ligand-only ExtraTrees residual recalibration of frozen V9 IC50",
            },
            "validation": self.info["v14_research_preview_validation"],
            "receptor_model": {
                "evaluated": True,
                "promoted": False,
                "reason": "neither V14 linear nor V14.1 nonlinear receptor fusion beat the best ligand-only surface across campaigns",
            },
            "claim_boundary": (
                "V14/V14.1 retrospective nested leave-one-campaign-out preview; not prospective confirmation, "
                "not the production default, and not for clinical or regulatory use"
            ),
        }

    def _receptor_cache_get(self, key: str) -> dict[str, Any] | None:
        with self._cache_lock:
            value = self._receptor_cache.get(key)
            if value is None:
                return None
            self._receptor_cache.move_to_end(key)
            return copy.deepcopy(value)

    def _receptor_cache_put(self, key: str, value: dict[str, Any]) -> None:
        with self._cache_lock:
            self._receptor_cache[key] = copy.deepcopy(value)
            self._receptor_cache.move_to_end(key)
            while len(self._receptor_cache) > RECEPTOR_CACHE_SIZE:
                self._receptor_cache.popitem(last=False)

    def _dock_diagnostic_receptors(
        self,
        registry: pd.DataFrame,
        receptor_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        """Run real Vina docking for prepared states that have no deployed ML surface."""
        if not receptor_ids:
            return {"status": "not_requested", "per_receptor": [], "runtime_seconds": 0.0}
        execution_contract = _docking_execution_contract(registry)
        if not execution_contract["docking_execution_eligible"]:
            reason = str(execution_contract["docking_execution_reason"])
            return {
                "status": "unavailable",
                "per_receptor": [
                    {
                        "receptor_id": receptor_id,
                        "status": "unavailable",
                        "model_role": RECEPTOR_STATES[receptor_id]["model_role"],
                        "model_prediction_generated": False,
                        "reason": reason,
                        "execution_eligibility": execution_contract,
                    }
                    for receptor_id in receptor_ids
                ],
                "runtime_seconds": 0.0,
            }

        started = time.perf_counter()
        eligible = _docking_execution_registry(registry)
        rows: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="herg_receptor_panel_") as temporary:
            prediction_root = Path(temporary)
            v139.v136._prepare_panel(eligible, prediction_root, self.receptor_tools)  # noqa: SLF001
            ligand_row = next(eligible.itertuples(index=False))
            microstate = v139.v134._prepare_microstate(  # noqa: SLF001
                ligand_row, prediction_root, self.receptor_tools
            )
            safe_ligand_id = v139.v13._safe_component(str(ligand_row.ligand_id))  # noqa: SLF001
            for receptor_id in receptor_ids:
                state_started = time.perf_counter()
                metadata = self.receptor_assets[receptor_id]
                task_id = f"{safe_ligand_id}__m{microstate['microstate_index']}__{receptor_id}"
                directory = prediction_root / "docking/tasks" / task_id
                directory.mkdir(parents=True, exist_ok=True)
                docked = directory / "poses.pdbqt"
                seed = (
                    v139.v13.SEED
                    + int(  # noqa: SLF001
                        hashlib.sha256(task_id.encode()).hexdigest()[:7], 16
                    )
                )
                try:
                    v139.v13._run(  # noqa: SLF001
                        [
                            self.receptor_tools.vina,
                            "--receptor",
                            metadata["pdbqt"],
                            "--ligand",
                            microstate["pdbqt_path"],
                            "--config",
                            metadata["box"],
                            "--exhaustiveness",
                            str(self.receptor_bundle["vina_protocol"]["exhaustiveness"]),
                            "--num_modes",
                            str(self.receptor_bundle["vina_protocol"]["modes"]),
                            "--seed",
                            str(seed),
                            "--cpu",
                            str(self.receptor_cpu),
                            "--out",
                            docked,
                        ],
                        directory / "vina.log",
                    )
                    models = v139.v13._pdbqt_models(docked)  # noqa: SLF001
                    affinities = np.asarray([model[0] for model in models], dtype=float)
                    best_index = int(np.argmin(affinities))
                    receptor_atoms = v139.v13._receptor_contact_atoms(metadata["pdb"])  # noqa: SLF001
                    contacts = v139.v13._contact_features(  # noqa: SLF001
                        models[best_index][1], receptor_atoms
                    )
                    residue_contacts = {
                        str(residue): {
                            "minimum_distance_A": float(contacts[f"contact_{residue}_minimum_A"]),
                            "ligand_atom_count_within_4p5A": int(
                                contacts[f"contact_{residue}_ligand_atom_count"]
                            ),
                        }
                        for residue in v139.v13.CONTACT_RESIDUES
                    }
                    rows.append(
                        {
                            "receptor_id": receptor_id,
                            "status": "completed",
                            "structure_context": metadata["structure_context"],
                            "asset_tier": metadata["asset_tier"],
                            "model_role": RECEPTOR_STATES[receptor_id]["model_role"],
                            "model_prediction_generated": False,
                            "execution_eligibility": execution_contract,
                            "best_vina_affinity_kcal_mol": float(np.min(affinities)),
                            "median_vina_affinity_kcal_mol": float(np.median(affinities)),
                            "vina_affinity_range_kcal_mol": float(np.max(affinities) - np.min(affinities)),
                            "returned_pose_count": int(len(models)),
                            "pose_count_within_1kcal": int(np.sum(affinities <= np.min(affinities) + 1.0)),
                            "ligand_efficiency_kcal_mol_per_heavy_atom": float(
                                np.min(affinities) / max(1, int(microstate["heavy_atom_count"]))
                            ),
                            "docked_microstate_formal_charge": int(microstate["formal_charge"]),
                            "best_pose_contacts": residue_contacts,
                            "minimum_protein_distance_A": float(contacts["minimum_protein_distance_A"]),
                            "severe_clash_indicator": bool(contacts["severe_clash_indicator"]),
                            "runtime_seconds": time.perf_counter() - state_started,
                            "claim_boundary": (
                                "Real rigid-receptor AutoDock Vina diagnostic only. No state-specific "
                                "hERG classifier or IC50 model is applied to this receptor."
                            ),
                        }
                    )
                except (OSError, v139.v13.CampaignError) as error:
                    print(
                        f"[receptor diagnostic error] {receptor_id} {type(error).__name__}: {error}",
                        flush=True,
                    )
                    rows.append(
                        {
                            "receptor_id": receptor_id,
                            "status": "failed",
                            "structure_context": metadata["structure_context"],
                            "asset_tier": metadata["asset_tier"],
                            "model_role": RECEPTOR_STATES[receptor_id]["model_role"],
                            "model_prediction_generated": False,
                            "reason": "AutoDock Vina docking did not complete for this receptor state.",
                            "execution_eligibility": execution_contract,
                            "runtime_seconds": time.perf_counter() - state_started,
                        }
                    )
        completed = sum(row["status"] == "completed" for row in rows)
        return {
            "status": ("completed" if completed == len(rows) else "partial" if completed else "failed"),
            "per_receptor": rows,
            "runtime_seconds": time.perf_counter() - started,
            "execution_eligibility": execution_contract,
        }

    @staticmethod
    def _primary_receptor_diagnostic(receptor_result: dict[str, Any]) -> dict[str, Any]:
        docking = receptor_result.get("docking")
        if not isinstance(docking, dict):
            return {
                "receptor_id": "8ZYO",
                "status": str(receptor_result.get("status", "unavailable")),
                "structure_context": RECEPTOR_STATES["8ZYO"]["structure_context"],
                "asset_tier": RECEPTOR_STATES["8ZYO"]["asset_tier"],
                "model_role": "model_driving_and_docking_diagnostic",
                "model_prediction_generated": False,
                "reason": str(
                    receptor_result.get("reason", "The model-driving docking result was unavailable.")
                ),
            }
        return {
            "receptor_id": "8ZYO",
            "status": "completed",
            "structure_context": RECEPTOR_STATES["8ZYO"]["structure_context"],
            "asset_tier": RECEPTOR_STATES["8ZYO"]["asset_tier"],
            "model_role": "model_driving_and_docking_diagnostic",
            "model_prediction_generated": True,
            "best_vina_affinity_kcal_mol": float(docking["best_affinity_kcal_mol"]),
            "median_vina_affinity_kcal_mol": float(docking["median_affinity_kcal_mol"]),
            "returned_pose_count": int(docking["returned_pose_count"]),
            "ligand_efficiency_kcal_mol_per_heavy_atom": float(
                docking["ligand_efficiency_kcal_mol_per_heavy_atom"]
            ),
            "best_pose_contacts": dict(docking["best_pose_contacts"]),
            "claim_boundary": (
                "8ZYO alone supplies features to the current frozen receptor-aware ML surfaces; "
                "those surfaces remain non-default retrospective research models."
            ),
        }

    def _receptor_panel_prediction(
        self,
        result: dict[str, Any],
        registry: pd.DataFrame,
        ligand: pd.DataFrame,
        receptor_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        """Combine the one supported model-driving state with optional docking-only states."""
        execution_contract = _docking_execution_contract(registry)
        model_applicable = bool(execution_contract["receptor_model_applicable"])
        if "8ZYO" in receptor_ids and model_applicable:
            model_receptor = self._receptor_prediction(result, registry, ligand)
        elif "8ZYO" in receptor_ids:
            model_receptor = {
                "status": "unavailable_out_of_model_domain",
                "reason": str(execution_contract["receptor_model_applicability_reason"]),
                "docking_execution_eligible": bool(
                    execution_contract["docking_execution_eligible"]
                ),
                "research_only": True,
                "claim_boundary": (
                    "The selected structure is outside the sealed 8ZYO receptor-model contract. "
                    "Vina may still run as a docking diagnostic, but no receptor-aware IC50 or "
                    "classifier output is generated."
                ),
            }
        else:
            model_receptor = {
                "status": "docking_only_no_model",
                "reason": (
                    "8ZYO was not selected. The other prepared receptor states provide docking "
                    "diagnostics only and do not drive a current hERG ML prediction."
                ),
                "research_only": True,
            }
        diagnostic_ids = tuple(
            receptor
            for receptor in receptor_ids
            if receptor != "8ZYO" or not model_applicable
        )
        diagnostic_run = self._dock_diagnostic_receptors(registry, diagnostic_ids)
        by_id = {row["receptor_id"]: row for row in diagnostic_run.get("per_receptor", [])}
        if "8ZYO" in receptor_ids and model_applicable:
            by_id["8ZYO"] = self._primary_receptor_diagnostic(model_receptor)
        per_receptor = [by_id[receptor_id] for receptor_id in receptor_ids]
        completed_rows = [row for row in per_receptor if row.get("status") == "completed"]
        score_rows = [row for row in completed_rows if "best_vina_affinity_kcal_mol" in row]
        cross_state: dict[str, Any] = {
            "completed_receptors": len(completed_rows),
            "requested_receptors": len(receptor_ids),
        }
        if score_rows:
            lowest = min(score_rows, key=lambda row: row["best_vina_affinity_kcal_mol"])
            scores = [float(row["best_vina_affinity_kcal_mol"]) for row in score_rows]
            cross_state.update(
                {
                    "lowest_vina_score_receptor_id": lowest["receptor_id"],
                    "vina_score_range_kcal_mol": max(scores) - min(scores),
                    "interpretation": (
                        "Lowest Vina score is reported as a scoring-function diagnostic only; it "
                        "does not establish the preferred biological state or a binding free energy."
                    ),
                }
            )
        panel_status = (
            "completed"
            if len(completed_rows) == len(per_receptor)
            else "partial"
            if completed_rows
            else "unavailable"
        )
        return {
            "receptor_aware": model_receptor,
            "multi_receptor_diagnostics": {
                "result_kind": "calculated_docking_diagnostic",
                "status": panel_status,
                "requested_receptors": list(receptor_ids),
                "model_driving_receptors": [
                    receptor
                    for receptor in receptor_ids
                    if receptor == "8ZYO" and model_applicable
                ],
                "docking_only_receptors": [
                    receptor
                    for receptor in receptor_ids
                    if receptor != "8ZYO" or not model_applicable
                ],
                "per_receptor": per_receptor,
                "cross_state": cross_state,
                "eligibility": execution_contract,
                "protocol": {
                    "tool": "AutoDock Vina 1.2.7",
                    "exhaustiveness": int(self.receptor_bundle["vina_protocol"]["exhaustiveness"]),
                    "requested_modes": int(self.receptor_bundle["vina_protocol"]["modes"]),
                    "prepared_receptors": (
                        "common-coordinate rigid receptors prepared at pH 7.4 with PDB2PQR/AMBER; "
                        "no membrane, explicit water network, induced fit, or state populations"
                    ),
                },
                "runtime_seconds": float(diagnostic_run.get("runtime_seconds", 0.0))
                + float(model_receptor.get("runtime", {}).get("total_seconds", 0.0)),
                "cache_hit": False,
                "claim_boundary": (
                    "Selected states are independently docked. Only 8ZYO can influence the current "
                    "receptor-aware ML output. Other-state scores and contacts are diagnostics, not "
                    "state-specific IC50 predictions and not binding free energies."
                ),
            },
        }

    def _receptor_prediction(
        self,
        result: dict[str, Any],
        registry: pd.DataFrame,
        ligand: pd.DataFrame,
    ) -> dict[str, Any]:
        if not bool(registry.iloc[0].docking_eligible):
            return {
                "status": "unavailable",
                "reason": str(registry.iloc[0].docking_exclusion_reason),
                "research_only": True,
            }
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="herg_receptor_") as temporary:
            prediction_root = Path(temporary)
            preparation_started = time.perf_counter()
            v139.v136._prepare_panel(  # noqa: SLF001
                registry.loc[registry.docking_eligible.astype(bool)].copy(),
                prediction_root,
                self.receptor_tools,
            )
            preparation_seconds = time.perf_counter() - preparation_started
            docking_started = time.perf_counter()
            docking = v139.v134._dock(  # noqa: SLF001
                self.v139_root,
                prediction_root,
                self.receptor_tools,
                registry.loc[registry.docking_eligible.astype(bool)].copy(),
                int(self.receptor_bundle["vina_protocol"]["exhaustiveness"]),
                int(self.receptor_bundle["vina_protocol"]["modes"]),
                self.receptor_cpu,
            )
            docking_seconds = time.perf_counter() - docking_started
            aggregate = v139.v13._aggregate_docking(docking, (v139.STATE,))  # noqa: SLF001
            receptor_input = ligand.merge(aggregate, on="ligand_id", validate="one_to_one")
            receptor_probability = v139._bundle_receptor_probabilities(  # noqa: SLF001
                self.receptor_bundle, receptor_input
            )
            hybrid = v139._hybrid_frame(receptor_input, receptor_probability)  # noqa: SLF001
            dock_row = docking.iloc[0]
            microstate = int(dock_row.microstate_index)
            safe_ligand_id = v139.v13._safe_component(str(dock_row.ligand_id))  # noqa: SLF001
            pose_path = (
                prediction_root
                / "docking/tasks"
                / f"{safe_ligand_id}__m{microstate}__{v139.STATE}"
                / "poses.pdbqt"
            )
            sdf_path = prediction_root / "ligands" / safe_ligand_id / f"microstate_{microstate}.sdf"
            feature_started = time.perf_counter()
            pose_features = v14._pose_ensemble_features(  # noqa: SLF001
                pose_path, self.receptor_atoms, sdf_path
            )
            router_values = np.asarray(
                [result["classification"]["probabilities"][name] for name in CLASS_NAMES],
                dtype=float,
            )
            frozen_values = receptor_probability[0].astype(float)
            feature_values: dict[str, float] = {
                "ligand_log_safe_vs_moderate": float(
                    math.log(np.clip(router_values[0], 1e-6, 1.0) / np.clip(router_values[1], 1e-6, 1.0))
                ),
                "ligand_log_potent_vs_moderate": float(
                    math.log(np.clip(router_values[2], 1e-6, 1.0) / np.clip(router_values[1], 1e-6, 1.0))
                ),
                "baseline_pic50": float(result["main_ic50"]["pic50"]),
                "ligand_router_entropy": float(
                    -np.sum(np.clip(router_values, 1e-9, 1.0) * np.log(np.clip(router_values, 1e-9, 1.0)))
                ),
                "docking_molecular_weight": float(registry.iloc[0].docking_molecular_weight),
                "docking_heavy_atom_count": float(registry.iloc[0].docking_heavy_atom_count),
                "docking_rotatable_bond_count": float(registry.iloc[0].docking_rotatable_bond_count),
                "dock_formal_charge": float(dock_row.formal_charge),
                "receptor_log_safe_vs_moderate": float(
                    math.log(np.clip(frozen_values[0], 1e-6, 1.0) / np.clip(frozen_values[1], 1e-6, 1.0))
                ),
                "receptor_log_potent_vs_moderate": float(
                    math.log(np.clip(frozen_values[2], 1e-6, 1.0) / np.clip(frozen_values[1], 1e-6, 1.0))
                ),
                "frozen_receptor_entropy": float(
                    -np.sum(np.clip(frozen_values, 1e-9, 1.0) * np.log(np.clip(frozen_values, 1e-9, 1.0)))
                ),
                **pose_features,
            }
            cationic = float(feature_values["pose_ligand_formal_positive_atom_count"] > 0)
            for column in (
                "pose_residue_652_cation_ring_distance_weighted",
                "pose_residue_656_cation_ring_distance_weighted",
                "pose_cage_cation_distance_best",
                "pose_cage_cation_distance_weighted",
            ):
                feature_values[f"{column}_conditional"] = cationic * (feature_values[column] - 6.0)
            model_frame = pd.DataFrame([feature_values])
            classifier_features = list(self.v141_receptor_classifier["features"])
            regressor_features = list(self.v141_receptor_regressor["features"])
            missing = (set(classifier_features) | set(regressor_features)) - set(model_frame)
            if missing:
                raise WebsiteError(
                    "The receptor feature contract is incomplete: " + ", ".join(sorted(missing))
                )
            raw = self.v141_receptor_classifier["classifier"].predict_proba(model_frame[classifier_features])[
                0
            ]
            classes = [
                int(value)
                for value in self.v141_receptor_classifier["classifier"].named_steps["model"].classes_
            ]
            by_index = {
                class_index: _finite_probability(raw[position])
                for position, class_index in enumerate(classes)
            }
            probabilities = {CLASS_NAMES[index]: by_index[index] for index in range(3)}
            residual = float(
                self.v141_receptor_regressor["residual_regressor"].predict(model_frame[regressor_features])[0]
            )
            receptor_pic50 = float(result["main_ic50"]["pic50"] + residual)
            feature_seconds = time.perf_counter() - feature_started
            aggregate_row = aggregate.iloc[0]
        frozen_probabilities = {
            name: _finite_probability(frozen_values[index]) for index, name in enumerate(CLASS_NAMES)
        }
        return {
            "status": "research_only_non_default",
            "definition": (
                "ligand RDKit2D+Morgan features plus AutoDock Vina 8ZYO all-pose, "
                "residue-contact, and frozen receptor-classifier features"
            ),
            "classification": {
                "tier": max(probabilities, key=probabilities.__getitem__),
                "probabilities": probabilities,
                "model": "V14.1 ExtraTrees ligand + all-pose receptor features",
            },
            "literature_ic50": {
                "pic50": receptor_pic50,
                "ic50_um": float(10.0 ** (6.0 - receptor_pic50)),
                "delta_from_ligand_pic50": residual,
                "model": "V14.1 ExtraTrees ligand + frozen-receptor residual model",
            },
            "frozen_receptor_classifier": {
                "tier": CLASS_NAMES[int(np.argmax(frozen_values))],
                "probabilities": frozen_probabilities,
                "hybrid_tier": CLASS_NAMES[int(hybrid.iloc[0].hybrid_prediction)],
                "model": "V13.9 five-fold ligand baseline + 8ZYO S649-contact ensemble",
            },
            "docking": {
                "tool": "AutoDock Vina 1.2.7",
                "receptor_state": "8ZYO",
                "exhaustiveness": int(self.receptor_bundle["vina_protocol"]["exhaustiveness"]),
                "requested_modes": int(self.receptor_bundle["vina_protocol"]["modes"]),
                "returned_pose_count": int(dock_row.pose_count),
                "best_affinity_kcal_mol": float(dock_row.best_affinity_kcal_mol),
                "median_affinity_kcal_mol": float(dock_row.median_affinity_kcal_mol),
                "ligand_efficiency_kcal_mol_per_heavy_atom": float(
                    dock_row.ligand_efficiency_kcal_mol_per_heavy_atom
                ),
                "best_pose_contacts": {
                    "T623": int(aggregate_row["dock__8ZYO__contact_623_count"]),
                    "S624": int(aggregate_row["dock__8ZYO__contact_624_count"]),
                    "S649": int(aggregate_row["dock__8ZYO__contact_649_count"]),
                    "Y652": int(aggregate_row["dock__8ZYO__contact_652_count"]),
                    "F656": int(aggregate_row["dock__8ZYO__contact_656_count"]),
                },
                "cage_contact_occupancy": float(feature_values["pose_cage_contact_occupancy"]),
                "historical_residue_note": (
                    "Legacy columns called position 649 F649; the deposited 8ZYO residue is S649."
                ),
            },
            "validation": self.info["receptor_research_validation"],
            "runtime": {
                "preparation_seconds": preparation_seconds,
                "docking_seconds": docking_seconds,
                "feature_and_model_seconds": feature_seconds,
                "total_seconds": time.perf_counter() - started,
                "cache_hit": False,
            },
            "claim_boundary": (
                "Retrospective research comparison only. The receptor surfaces did not beat the "
                "strongest ligand-only comparators with statistically conclusive gains and are not "
                "promoted for clinical, regulatory, or production-default use. Vina scores are not "
                "binding free energies."
            ),
        }

    def predict(
        self,
        smiles: str,
        mode: str = "ligand",
        receptor_ids: Any = None,
        feature_mode: str = "atomwise",
        heavy_molecule_analysis: bool = False,
    ) -> dict[str, Any]:
        smiles = str(smiles).strip()
        if not smiles:
            raise WebsiteError("Enter a SMILES string")
        if len(smiles) > MAX_SMILES_LENGTH:
            raise WebsiteError(f"SMILES must be {MAX_SMILES_LENGTH:,} characters or fewer")
        if _quiet_smiles_molecule(smiles) is None:
            raise WebsiteError("The SMILES string could not be parsed")
        selected_mode = _prediction_mode(mode)
        selected_feature_mode = _feature_mode(feature_mode)
        run_heavy_analysis = _boolean_flag(
            heavy_molecule_analysis,
            "Heavy-molecule analysis",
        )
        if selected_mode == "ligand_receptor":
            selected_receptors = _receptor_selection(receptor_ids)
        else:
            if receptor_ids is not None:
                raise WebsiteError(
                    "Receptor selection is available only in 'ligand_receptor' prediction mode"
                )
            selected_receptors = ()
        started = time.perf_counter()
        with self._ligand_lock:
            result, registry, ligand = self._ligand_prediction(smiles)
        ligand_seconds = time.perf_counter() - started
        result["property_profile"] = self.properties(smiles)
        result["submitted_smiles"] = smiles
        result["standardization"] = dict(result["property_profile"]["standardization"])
        result["heavy_molecule_diagnostic"] = self._heavy_molecule_diagnostic(
            result["property_profile"], result
        )
        functional_group_prediction = None
        if selected_feature_mode == "functional_group":
            functional_group_prediction = self._functional_group_prediction(
                result["smiles"],
                float(result["main_ic50"]["pic50"]),
            )
            selected_quantitative_result = {
                **functional_group_prediction["regression"],
                "surface": "functional_group_research",
                "promotion_status": "research_only_failed_promotion",
            }
        else:
            selected_quantitative_result = {
                **copy.deepcopy(result["main_ic50"]),
                "surface": "atomwise_default",
                "promotion_status": "live_default",
            }
        result["feature_mode"] = {
            "selected": selected_feature_mode,
            "label": self.info["feature_modes"][selected_feature_mode]["label"],
            "validated_default_preserved": True,
            "default_surface": "atomwise",
        }
        result["selected_quantitative_result"] = selected_quantitative_result
        result["functional_group_research"] = functional_group_prediction
        result["advanced_heavy_analysis"] = {
            "enabled": run_heavy_analysis,
            "primary_prediction_unchanged": True,
            "molecular_weight_da": result["property_profile"]["descriptors"]["molecular_weight"][
                "value"
            ],
            "heavy_atom_count": result["property_profile"]["descriptors"]["heavy_atoms"]["value"],
            "rotatable_bonds": result["property_profile"]["descriptors"]["rotatable_bonds"]["value"],
            "clogp": result["property_profile"]["descriptors"]["clogp"]["value"],
            "tpsa": result["property_profile"]["descriptors"]["tpsa"]["value"],
            "diagnostic": copy.deepcopy(result["heavy_molecule_diagnostic"]),
            "docking_contract": copy.deepcopy(result["property_profile"]["docking_eligibility"]),
            "interpretation": (
                "This opt-in view expands size, flexibility, applicability, and docking-execution checks. "
                "It does not rescale or replace the validated IC50 prediction because high-MW correction "
                "has not been independently validated."
            ),
        }
        receptor_cache_hit = False
        if selected_mode == "ligand_receptor":
            cache_material = json.dumps(
                {"smiles": result["smiles"], "receptor_ids": selected_receptors},
                separators=(",", ":"),
            ).encode()
            cache_key = hashlib.sha256(cache_material).hexdigest()
            panel = self._receptor_cache_get(cache_key)
            if panel is None:
                with self._receptor_lock:
                    panel = self._receptor_cache_get(cache_key)
                    if panel is None:
                        panel = self._receptor_panel_prediction(result, registry, ligand, selected_receptors)
                        receptor_status = panel["receptor_aware"].get("status")
                        diagnostic_status = panel["multi_receptor_diagnostics"].get("status")
                        if receptor_status == "research_only_non_default" or diagnostic_status == "completed":
                            self._receptor_cache_put(cache_key, panel)
                    else:
                        receptor_cache_hit = True
            else:
                receptor_cache_hit = True
            receptor = panel["receptor_aware"]
            multi_receptor = panel["multi_receptor_diagnostics"]
            if receptor.get("runtime") and receptor_cache_hit:
                receptor["runtime"]["cache_hit"] = True
                receptor["runtime"]["served_from_cache_seconds"] = (
                    time.perf_counter() - started - ligand_seconds
                )
            if receptor_cache_hit:
                multi_receptor["cache_hit"] = True
                multi_receptor["served_from_cache_seconds"] = time.perf_counter() - started - ligand_seconds
            result["receptor_aware"] = receptor
            result["multi_receptor_diagnostics"] = multi_receptor
        else:
            result["receptor_aware"] = {
                "status": "not_requested",
                "how_to_run": "Choose Ligand + receptor to run AutoDock Vina and both-feature models.",
            }
            result["multi_receptor_diagnostics"] = {
                "result_kind": "calculated_docking_diagnostic",
                "status": "not_requested",
                "requested_receptors": [],
            }
        result["schema_version"] = WEBSITE_SCHEMA
        result["prediction_mode"] = selected_mode
        result["selected_receptor_ids"] = list(selected_receptors)
        result["runtime"] = {
            "ligand_seconds": ligand_seconds,
            "total_seconds": time.perf_counter() - started,
            "receptor_cache_hit": receptor_cache_hit,
        }
        result["release"] = {
            "name": self.info["release"],
            "default_policy": self.info["default_policy"],
        }
        receptor_features_used = result["receptor_aware"].get("status") == "research_only_non_default"
        docking_diagnostics_used = result["multi_receptor_diagnostics"].get("status") in {
            "completed",
            "partial",
        }
        result["scientific_scope"] = {
            "research_use_only": True,
            "clinical_or_regulatory_use": False,
            "prospectively_validated": False,
            "ligand_only_default": True,
            "v14_ligand_research_preview_available": True,
            "selected_feature_mode": selected_feature_mode,
            "functional_group_prediction_used": selected_feature_mode == "functional_group",
            "functional_group_prediction_promoted": False,
            "heavy_molecule_analysis_requested": run_heavy_analysis,
            "receptor_research_mode_available": True,
            "receptor_mode_requested": selected_mode == "ligand_receptor",
            "receptor_features_used": receptor_features_used,
            "receptor_docking_diagnostics_used": docking_diagnostics_used,
            "selected_receptor_ids": list(selected_receptors),
            "receptor_prediction_promoted": False,
            "external_regression_evaluation": True,
        }
        return result

    @staticmethod
    def _comparison_smiles(value: Any, role: str) -> str:
        """Validate a comparison structure before invoking the frozen model stack."""
        if not isinstance(value, str) or not value.strip():
            raise WebsiteError(f"Enter a {role} SMILES string")
        smiles = value.strip()
        if len(smiles) > MAX_SMILES_LENGTH:
            raise WebsiteError(
                f"{role.capitalize()} SMILES must be {MAX_SMILES_LENGTH:,} characters or fewer"
            )
        if _quiet_smiles_molecule(smiles) is None:
            raise WebsiteError(f"The {role} SMILES string could not be parsed")
        return smiles

    @staticmethod
    def _known_parent_ic50(value: Any) -> float | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise WebsiteError("Known parent IC50 must be a positive number in µM")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise WebsiteError("Known parent IC50 must be a positive number in µM") from error
        if not math.isfinite(number) or number <= 0.0:
            raise WebsiteError("Known parent IC50 must be a positive number in µM")
        return number

    @staticmethod
    def _comparison_prediction_output(prediction: dict[str, Any]) -> dict[str, Any]:
        """Keep regression and classifier quantities visibly distinct in comparison output."""
        main = prediction["main_ic50"]
        classifier = prediction["classification"]
        return {
            "result_kind": "model_prediction",
            "smiles": prediction["smiles"],
            "structure_depiction": dict(prediction["structure_depiction"]),
            "regression": {
                "predicted_ic50_um": float(main["ic50_um"]),
                "predicted_pic50": float(main["pic50"]),
                "threshold_tier": str(main["tier"]),
                "residual_distribution_tier_probabilities": dict(main["tier_probabilities"]),
                "interval90_um": dict(main["interval90_um"]),
                "model": str(main["model_source"]),
            },
            "classifier": {
                "predicted_tier": str(classifier["tier"]),
                "probabilities": dict(classifier["probabilities"]),
                "model": str(classifier["model"]),
            },
            "classification_regression_consistency": dict(
                prediction["classification_regression_consistency"]
            ),
            "applicability": dict(prediction["applicability"]),
            "probability_note": (
                "Classifier probabilities and regression-derived tier probabilities are separate "
                "statistical outputs; neither is a conversion of the other."
            ),
        }

    @staticmethod
    def _property_delta(
        parent: dict[str, Any],
        candidate: dict[str, Any],
        key: str,
    ) -> dict[str, Any]:
        parent_descriptor = parent["descriptors"][key]
        candidate_descriptor = candidate["descriptors"][key]
        parent_value = float(parent_descriptor["value"])
        candidate_value = float(candidate_descriptor["value"])
        return {
            "label": str(parent_descriptor["label"]),
            "parent": parent_value,
            "candidate": candidate_value,
            "candidate_minus_parent": candidate_value - parent_value,
            "unit": str(parent_descriptor["unit"]),
            "method": str(parent_descriptor["method"]),
        }

    @staticmethod
    def _basic_center_count(properties: dict[str, Any]) -> int:
        # Permanently charged atoms are reported separately via formal charge. Counting
        # only PosIonizable features avoids describing a quaternary center as a pKa.
        return int(properties["basicity"]["positive_ionizable_centers"])

    @staticmethod
    def _functional_group_changes(
        parent: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        parent_groups = {
            str(row["key"]): (str(row["label"]), int(row["count"])) for row in parent["functional_groups"]
        }
        candidate_groups = {
            str(row["key"]): (str(row["label"]), int(row["count"])) for row in candidate["functional_groups"]
        }
        added: list[dict[str, Any]] = []
        removed: list[dict[str, Any]] = []
        for key in sorted(set(parent_groups) | set(candidate_groups)):
            label = candidate_groups[key][0] if key in candidate_groups else parent_groups[key][0]
            parent_count = parent_groups.get(key, (label, 0))[1]
            candidate_count = candidate_groups.get(key, (label, 0))[1]
            delta = candidate_count - parent_count
            if not delta:
                continue
            record = {
                "key": key,
                "label": label,
                "parent_count": parent_count,
                "candidate_count": candidate_count,
                "count_change": delta,
            }
            (added if delta > 0 else removed).append(record)

        parent_basic = WebsitePredictor._basic_center_count(parent)
        candidate_basic = WebsitePredictor._basic_center_count(candidate)
        basic_delta = candidate_basic - parent_basic
        parent_charge = int(parent["descriptors"]["formal_charge"]["value"])
        candidate_charge = int(candidate["descriptors"]["formal_charge"]["value"])
        charge_delta = candidate_charge - parent_charge
        descriptions = [
            *(f"added {row['count_change']} × {row['label']}" for row in added),
            *(f"removed {abs(row['count_change'])} × {row['label']}" for row in removed),
        ]
        if basic_delta > 0:
            descriptions.append(f"added {basic_delta} RDKit positive-ionizable center(s)")
        elif basic_delta < 0:
            descriptions.append(f"removed {abs(basic_delta)} RDKit positive-ionizable center(s)")
        if charge_delta:
            descriptions.append(f"formal charge changed by {charge_delta:+d}")
        if not descriptions:
            descriptions.append(
                "No change detected by the current RDKit fragment, formal-charge, or "
                "positive-ionizable-center rules; the edit may be positional, stereochemical, "
                "or outside these named patterns."
            )
        return {
            "added": added,
            "removed": removed,
            "basic_centers": {
                "parent": parent_basic,
                "candidate": candidate_basic,
                "candidate_minus_parent": basic_delta,
                "method": "RDKit BaseFeatures PosIonizable feature count; not a pKa calculation",
            },
            "formal_charge": {
                "parent": parent_charge,
                "candidate": candidate_charge,
                "candidate_minus_parent": charge_delta,
            },
            "description": "; ".join(descriptions),
        }

    @staticmethod
    def _morgan_tanimoto(parent_smiles: str, candidate_smiles: str) -> float:
        columns = [f"morgan__{index:04d}" for index in range(v103.v102.MORGAN_BITS)]
        parent_frame, _ = v103.v102.v10._feature_frame(parent_smiles, columns)  # noqa: SLF001
        candidate_frame, _ = v103.v102.v10._feature_frame(candidate_smiles, columns)  # noqa: SLF001
        parent_bits = parent_frame[columns].fillna(0).to_numpy(dtype=bool)[0]
        candidate_bits = candidate_frame[columns].fillna(0).to_numpy(dtype=bool)[0]
        intersection = int(np.logical_and(parent_bits, candidate_bits).sum())
        union = int(np.logical_or(parent_bits, candidate_bits).sum())
        return float(intersection / union) if union else 1.0

    def _heavy_molecule_diagnostic(
        self,
        properties: dict[str, Any],
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        mw_association = next(
            row for row in properties["training_associations"] if row["key"] == "molecular_weight"
        )
        percentile = float(mw_association["percentile"])
        molecular_weight = float(properties["descriptors"]["molecular_weight"]["value"])
        warnings = []
        if percentile >= HIGH_MW_REFERENCE_PERCENTILE:
            warnings.append(
                "Molecular weight is at or above the empirical 95th percentile of the model's "
                "property-reference corpus; small-edit sensitivity may be less reliable in this region."
            )
        if molecular_weight > 750.0:
            warnings.append(
                "Molecular weight exceeds 750 Da. The ligand-only prediction completed and Vina "
                "may be attempted as a diagnostic, but the frozen receptor-aware ML surface is "
                "outside its documented applicability contract."
            )
        docking = properties["docking_eligibility"]
        if not docking["eligible_under_current_receptor_model"]:
            warnings.append(
                "The current receptor-aware model marks this structure out of domain: "
                + str(docking["reason"])
            )
        applicability = prediction["applicability"]
        if applicability["domain_label"] == "Extrapolative chemistry":
            warnings.append(str(applicability["interpretation"]))
        return {
            "molecular_weight_da": molecular_weight,
            "training_reference_percentile": percentile,
            "training_reference_structures": int(len(self.property_reference)),
            "at_or_above_empirical_95th_percentile": bool(percentile >= HIGH_MW_REFERENCE_PERCENTILE),
            "above_current_receptor_750_da_limit": bool(molecular_weight > 750.0),
            "ligand_model_applicability": {
                "domain_label": str(applicability["domain_label"]),
                "maximum_train_tanimoto": float(applicability["maximum_train_tanimoto"]),
                "exact_training_overlap": bool(applicability["exact_training_overlap"]),
            },
            "current_receptor_model_eligibility": dict(docking),
            "docking_execution_eligibility": {
                "eligible": bool(docking.get("docking_execution_eligible", False)),
                "reason": str(docking.get("docking_execution_reason", "unavailable")),
            },
            "warnings": warnings,
        }

    @staticmethod
    def _generated_candidate_count(value: Any) -> int:
        if value is None:
            return MAX_GENERATED_CANDIDATES
        if isinstance(value, bool) or not isinstance(value, int):
            raise WebsiteError("Generated candidate count must be a whole number")
        if not 1 <= value <= MAX_GENERATED_CANDIDATES:
            raise WebsiteError(
                f"Generated candidate count must be between 1 and {MAX_GENERATED_CANDIDATES}"
            )
        return value

    def generate_candidates(
        self,
        parent_smiles: Any,
        max_candidates: Any = MAX_GENERATED_CANDIDATES,
    ) -> dict[str, Any]:
        """Enumerate and evidence-rank bounded edits without making synthesis claims."""
        parent_input = self._comparison_smiles(parent_smiles, "parent")
        requested_count = self._generated_candidate_count(max_candidates)
        started = time.perf_counter()
        generated = generate_herg_edit_candidates(
            parent_input,
            bounds=GenerationBounds(target_count=requested_count),
        )
        parent_prediction = self.predict(parent_input, "ligand")
        parent_properties = parent_prediction["property_profile"]
        parent_ic50 = float(parent_prediction["main_ic50"]["ic50_um"])
        parent_pic50 = float(parent_prediction["main_ic50"]["pic50"])

        evidence: list[CandidatePredictionEvidence] = []
        evaluated: dict[str, dict[str, Any]] = {}
        prediction_failures: list[dict[str, str]] = []
        property_keys = (
            "molecular_weight",
            "tpsa",
            "clogp",
            "hbd",
            "hba",
            "formal_charge",
            "rotatable_bonds",
        )
        for generated_candidate in generated.candidates:
            try:
                prediction = self.predict(generated_candidate.canonical_smiles, "ligand")
                properties = prediction["property_profile"]
                property_deltas = {
                    key: self._property_delta(parent_properties, properties, key)
                    for key in property_keys
                }
                parent_basic = self._basic_center_count(parent_properties)
                candidate_basic = self._basic_center_count(properties)
                property_deltas["basic_centers"] = {
                    "label": "Positive-ionizable centers",
                    "parent": parent_basic,
                    "candidate": candidate_basic,
                    "candidate_minus_parent": candidate_basic - parent_basic,
                    "unit": "",
                    "method": "RDKit BaseFeatures PosIonizable feature count; not a pKa calculation",
                }
                main = prediction["main_ic50"]
                applicability = prediction["applicability"]
                evidence.append(
                    CandidatePredictionEvidence(
                        candidate_id=generated_candidate.candidate_id,
                        canonical_smiles=generated_candidate.canonical_smiles,
                        predicted_ic50_um=float(main["ic50_um"]),
                        interval90_lower_um=float(main["interval90_um"]["lower"]),
                        interval90_upper_um=float(main["interval90_um"]["upper"]),
                        applicability_label=str(applicability["domain_label"]),
                        maximum_train_tanimoto=float(applicability["maximum_train_tanimoto"]),
                        parent_morgan_tanimoto=float(
                            generated_candidate.parent_morgan_tanimoto
                        ),
                        property_deltas={
                            key: float(record["candidate_minus_parent"])
                            for key, record in property_deltas.items()
                        },
                        model_provenance=str(main["model_source"]),
                    )
                )
                delta_pic50 = float(main["pic50"]) - parent_pic50
                if delta_pic50 < -COMPARISON_DIRECTION_EPSILON_PIC50:
                    direction = "improved"
                elif delta_pic50 > COMPARISON_DIRECTION_EPSILON_PIC50:
                    direction = "worsened"
                else:
                    direction = "essentially_unchanged"
                functional_changes = self._functional_group_changes(
                    parent_properties,
                    properties,
                )
                evaluated[generated_candidate.candidate_id] = {
                    "candidate_id": generated_candidate.candidate_id,
                    "canonical_smiles": generated_candidate.canonical_smiles,
                    "generation": {
                        "lineages": [asdict(row) for row in generated_candidate.lineages],
                        "lineage_count": len(generated_candidate.lineages),
                        "parent_morgan_tanimoto": float(
                            generated_candidate.parent_morgan_tanimoto
                        ),
                        "molecular_weight_delta_da": float(
                            generated_candidate.molecular_weight_delta_da
                        ),
                        "heavy_atom_delta": int(generated_candidate.heavy_atom_delta),
                        "formal_charge_delta": int(generated_candidate.formal_charge_delta),
                        "interpretation_boundary": generated_candidate.interpretation_boundary,
                    },
                    "model_output": self._comparison_prediction_output(prediction),
                    "comparison": {
                        "direction": direction,
                        "candidate_minus_parent_pic50": delta_pic50,
                        "candidate_minus_parent_ic50_um": float(main["ic50_um"]) - parent_ic50,
                        "candidate_over_parent_ic50_fold": float(main["ic50_um"]) / parent_ic50,
                        "functional_group_changes": functional_changes,
                        "low_predicted_sensitivity": bool(
                            generated_candidate.parent_morgan_tanimoto
                            >= HIGH_SIMILARITY_TANIMOTO
                            and abs(delta_pic50) < LOW_SENSITIVITY_DELTA_PIC50
                        ),
                    },
                    "calculated_property_deltas": property_deltas,
                }
            except WebsiteError as error:
                prediction_failures.append(
                    {
                        "candidate_id": generated_candidate.candidate_id,
                        "canonical_smiles": generated_candidate.canonical_smiles,
                        "reason": str(error),
                    }
                )

        ranked = rank_candidate_evidence(parent_ic50, evidence) if evidence else ()
        ranked_candidates = []
        for index, ranked_row in enumerate(ranked):
            candidate = evaluated[ranked_row.candidate_id]
            candidate["ranking"] = asdict(ranked_row)
            if index >= 12:
                candidate["model_output"].pop("structure_depiction", None)
            ranked_candidates.append(candidate)

        failure_counts: dict[str, int] = {}
        for failure in generated.failures:
            failure_counts[failure.reason] = failure_counts.get(failure.reason, 0) + 1
        return {
            "schema_version": WEBSITE_SCHEMA,
            "analysis_mode": "experimental_bounded_candidate_generation",
            "target": {"key": "herg", "label": "hERG potassium channel"},
            "parent": {
                "canonical_smiles": generated.parent_canonical_smiles,
                "model_output": self._comparison_prediction_output(parent_prediction),
                "calculated_properties": parent_properties,
            },
            "generation": {
                "status": generated.status,
                "registry_version": generated.registry_version,
                "rdkit_version": generated.rdkit_version,
                "requested_candidate_count": requested_count,
                "generated_candidate_count": len(generated.candidates),
                "successfully_evaluated_count": len(ranked_candidates),
                "valid_unique_before_selection": generated.valid_unique_before_selection,
                "raw_product_count": generated.raw_product_count,
                "truncated_to_target": generated.truncated_to_target,
                "bounds": asdict(generated.bounds),
                "failure_counts": failure_counts,
                "suppressed_failure_count": generated.suppressed_failure_count,
                "claim_boundary": generated.claim_boundary,
            },
            "ranked_candidates": ranked_candidates,
            "prediction_failures": prediction_failures,
            "runtime_seconds": time.perf_counter() - started,
            "claim_boundary": (
                "Experimental local research workflow. Candidates are deterministic one-step RDKit "
                "enumerations. Rankings combine frozen ligand-model outputs with disclosed uncertainty, "
                "applicability, similarity, and property-change penalties. They are not probabilities, "
                "synthesis recommendations, measured SAR, or prospective activity claims. Generated "
                "candidates are never added to training data."
            ),
        }

    def compare(
        self,
        parent_smiles: Any,
        candidate_smiles: Any,
        known_parent_ic50_um: Any = None,
        mw_delta_stress_test: Any = False,
    ) -> dict[str, Any]:
        """Compare two supplied structures without generating or docking a candidate."""
        parent_input = self._comparison_smiles(parent_smiles, "parent")
        candidate_input = self._comparison_smiles(candidate_smiles, "candidate")
        measured_parent_ic50 = self._known_parent_ic50(known_parent_ic50_um)
        run_mw_stress_test = _boolean_flag(mw_delta_stress_test, "MW delta stress-test flag")
        started = time.perf_counter()

        # These public methods remain the single sources of model and property behavior.
        parent_prediction = self.predict(parent_input, "ligand")
        candidate_prediction = self.predict(candidate_input, "ligand")
        parent_properties = parent_prediction["property_profile"]
        candidate_properties = candidate_prediction["property_profile"]

        parent_ic50 = float(parent_prediction["main_ic50"]["ic50_um"])
        candidate_ic50 = float(candidate_prediction["main_ic50"]["ic50_um"])
        parent_pic50 = float(parent_prediction["main_ic50"]["pic50"])
        candidate_pic50 = float(candidate_prediction["main_ic50"]["pic50"])
        delta_ic50 = candidate_ic50 - parent_ic50
        delta_pic50 = candidate_pic50 - parent_pic50
        if delta_pic50 < -COMPARISON_DIRECTION_EPSILON_PIC50:
            direction = "improved"
        elif delta_pic50 > COMPARISON_DIRECTION_EPSILON_PIC50:
            direction = "worsened"
        else:
            direction = "essentially_unchanged"
        fold_ratio = candidate_ic50 / parent_ic50
        absolute_fold = max(fold_ratio, 1.0 / fold_ratio)
        similarity = self._morgan_tanimoto(parent_properties["smiles"], candidate_properties["smiles"])
        low_sensitivity = bool(
            similarity >= HIGH_SIMILARITY_TANIMOTO and abs(delta_pic50) < LOW_SENSITIVITY_DELTA_PIC50
        )

        property_deltas = {
            key: self._property_delta(parent_properties, candidate_properties, key)
            for key in (
                "molecular_weight",
                "tpsa",
                "clogp",
                "hbd",
                "hba",
                "formal_charge",
                "rotatable_bonds",
            )
        }
        parent_basic = self._basic_center_count(parent_properties)
        candidate_basic = self._basic_center_count(candidate_properties)
        property_deltas["basic_centers"] = {
            "label": "Positive-ionizable centers",
            "parent": parent_basic,
            "candidate": candidate_basic,
            "candidate_minus_parent": candidate_basic - parent_basic,
            "unit": "",
            "method": "RDKit BaseFeatures PosIonizable feature count; not a pKa calculation",
        }
        functional_changes = self._functional_group_changes(parent_properties, candidate_properties)
        parent_heavy = self._heavy_molecule_diagnostic(parent_properties, parent_prediction)
        candidate_heavy = self._heavy_molecule_diagnostic(candidate_properties, candidate_prediction)

        warnings = [*parent_heavy["warnings"], *candidate_heavy["warnings"]]
        if low_sensitivity:
            warnings.append(
                "High structural similarity; the model predicts only a small hERG difference. "
                "This is a sensitivity diagnostic, not evidence that the compounds are experimentally equivalent."
            )
        statements = [
            (
                f"The ligand-only regression predicts the candidate is {direction.replace('_', ' ')} "
                f"relative to the parent using a ±{COMPARISON_DIRECTION_EPSILON_PIC50:.2f} pIC50 "
                "negligible-change band."
            ),
            functional_changes["description"],
        ]
        changed_context = []
        if property_deltas["clogp"]["candidate_minus_parent"] < 0:
            changed_context.append("lower calculated cLogP")
        elif property_deltas["clogp"]["candidate_minus_parent"] > 0:
            changed_context.append("higher calculated cLogP")
        if property_deltas["tpsa"]["candidate_minus_parent"] > 0:
            changed_context.append("higher TPSA")
        elif property_deltas["tpsa"]["candidate_minus_parent"] < 0:
            changed_context.append("lower TPSA")
        basic_delta = property_deltas["basic_centers"]["candidate_minus_parent"]
        if basic_delta < 0:
            changed_context.append("fewer detected positive-ionizable centers")
        elif basic_delta > 0:
            changed_context.append("more detected positive-ionizable centers")
        if changed_context:
            statements.append(
                "The prediction change coincides with "
                + ", ".join(changed_context)
                + "; these are calculated associations, not a causal model attribution."
            )

        measured_values = None
        if measured_parent_ic50 is not None:
            measured_values = {
                "result_kind": "user_supplied_measured_value",
                "parent_ic50_um": measured_parent_ic50,
                "parent_pic50": float(6.0 - math.log10(measured_parent_ic50)),
                "source": "user_supplied_known_or_experimental_value",
                "unit": "µM",
                "claim_boundary": "The website does not independently verify this user-supplied value.",
            }

        experimental_mw_sensitivity = None
        if run_mw_stress_test:
            parent_mw = float(parent_properties["descriptors"]["molecular_weight"]["value"])
            candidate_mw = float(candidate_properties["descriptors"]["molecular_weight"]["value"])
            experimental_mw_sensitivity = mw_delta.mw_proportional_delta_stress_test(
                parent_pic50,
                candidate_pic50,
                parent_mw,
                candidate_mw,
                enabled=True,
                reference_median_mw_da=MW_STRESS_REFERENCE_MEDIAN_MW_DA,
                baseline_multiplier=MW_STRESS_BASELINE_MULTIPLIER,
                minimum_multiplier=1.0,
                provenance=dict(MW_STRESS_AUDIT_PROVENANCE),
            )
            scenario = experimental_mw_sensitivity["stress_scenario"]
            stressed_candidate_pic50 = float(scenario["candidate_pic50"])
            stressed_candidate_ic50 = float(10.0 ** (6.0 - stressed_candidate_pic50))
            applied_multiplier = float(scenario["applied_multiplier"])
            pair_mean_mw = float(scenario["pair_mean_mw_da"])
            experimental_mw_sensitivity["status"] = "applied_opt_in_scenario"
            experimental_mw_sensitivity["metrics"] = {
                "parent_molecular_weight_da": parent_mw,
                "candidate_molecular_weight_da": candidate_mw,
                "pair_mean_molecular_weight_da": pair_mean_mw,
                "raw_delta_pic50": delta_pic50,
                "mw_scaled_delta_pic50": float(scenario["candidate_minus_parent_delta_pic50"]),
                "scaling_factor": applied_multiplier,
                "primary_candidate_ic50_um": candidate_ic50,
                "stress_candidate_ic50_um": stressed_candidate_ic50,
                "stress_candidate_minus_parent_ic50_um": (stressed_candidate_ic50 - parent_ic50),
            }
            experimental_mw_sensitivity["interpretation"] = [
                (
                    f"The opt-in scenario multiplies the model's candidate-minus-parent pIC50 "
                    f"change by {applied_multiplier:.2f}× at a pair mean MW of {pair_mean_mw:.1f} Da."
                ),
                (
                    "The primary parent and candidate predictions above are unchanged. This view "
                    "only asks how the result would look if the predicted edit magnitude were amplified."
                ),
                (
                    "Matched-pair support above 700 Da is sparse, so high-MW scenario values are "
                    "especially uncertain. Direction errors are not repaired by this calculation."
                    if pair_mean_mw >= 700.0
                    else "This is retrospective sensitivity context, not an independently validated correction."
                ),
            ]
            experimental_mw_sensitivity["claim_boundary"] = (
                "Experimental MW-proportional pIC50-delta stress test only. It does not alter the "
                "displayed primary model output, probabilities, tiers, or intervals. It is not a "
                "calibrated IC50 prediction."
            )

        return {
            "schema_version": WEBSITE_SCHEMA,
            "analysis_mode": "manual_parent_candidate_comparison",
            "target": {"key": "herg", "label": "hERG potassium channel"},
            "prediction_mode": "ligand",
            "measured_values": measured_values,
            "model_outputs": {
                "result_kind": "model_prediction",
                "parent": self._comparison_prediction_output(parent_prediction),
                "candidate": self._comparison_prediction_output(candidate_prediction),
            },
            "calculated_properties": {
                "result_kind": "calculated_descriptors",
                "parent": parent_properties,
                "candidate": candidate_properties,
                "deltas": property_deltas,
            },
            "experimental_mw_sensitivity": experimental_mw_sensitivity,
            "comparison": {
                "result_kind": "calculated_comparison_diagnostic",
                "direction": direction,
                "direction_definition": (
                    "Improved means higher predicted IC50 and lower predicted hERG liability; "
                    f"|ΔpIC50| ≤ {COMPARISON_DIRECTION_EPSILON_PIC50:.2f} is treated as essentially unchanged."
                ),
                "delta_ic50_um": delta_ic50,
                "delta_pic50": delta_pic50,
                "improvement_delta_pic50": -delta_pic50,
                "absolute_ic50_change_um": abs(delta_ic50),
                "candidate_over_parent_ic50_fold": fold_ratio,
                "absolute_fold_change": absolute_fold,
                "similarity": {
                    "morgan_tanimoto": similarity,
                    "method": (
                        "RDKit Morgan radius 2, 2,048 bits, chirality enabled; identical to the "
                        "deployed ligand-model fingerprint configuration"
                    ),
                },
                "low_predicted_sensitivity": low_sensitivity,
                "low_sensitivity_rule": {
                    "minimum_tanimoto": HIGH_SIMILARITY_TANIMOTO,
                    "maximum_absolute_delta_pic50": LOW_SENSITIVITY_DELTA_PIC50,
                    "scope": "transparent product diagnostic; not a confidence probability",
                },
                "functional_group_changes": functional_changes,
                "heavy_molecule_diagnostic": {
                    "parent": parent_heavy,
                    "candidate": candidate_heavy,
                },
                "warnings": list(dict.fromkeys(warnings)),
            },
            "heuristic_interpretation": {
                "result_kind": "heuristic_interpretation",
                "statements": statements,
                "claim_boundary": (
                    "Interpretation summarizes model output and calculated descriptor changes. "
                    "It is not SHAP attribution, causal evidence, a measured SAR result, or a pKa prediction."
                ),
            },
            "runtime_seconds": time.perf_counter() - started,
            "claim_boundary": (
                "Manual research comparison only. No candidate was generated and no docking was run. "
                "Small-edit direction and magnitude require prospective or paired experimental validation. "
                "Any MW-scaled result is a separate opt-in stress scenario, not a model recalibration."
            ),
        }


def _validate_static_files(web_root: Path) -> None:
    for filename, _, _ in STATIC_ROUTES.values():
        path = (web_root / filename).resolve()
        if not path.is_relative_to(web_root.resolve()) or not path.is_file():
            raise WebsiteError(f"Missing website asset: {filename}")


def _handler(
    predictor: WebsitePredictor,
    web_root: Path,
    username: str,
    password: str | None,
    internal_examples: dict[str, Any] | None = None,
) -> type[BaseHTTPRequestHandler]:
    expected = _basic_authorization(username, password) if password else None

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "hERGResearch/1.0"

        def _security_headers(self, cache_control: str = "no-store") -> None:
            self.send_header("Cache-Control", cache_control)
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'self'; "
                "frame-ancestors 'none'; object-src 'none'",
            )
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            if self.close_connection:
                self.send_header("Connection", "close")

        def _send_bytes(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str,
            cache_control: str = "no-store",
            *,
            include_body: bool = True,
        ) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self._security_headers(cache_control)
            self.end_headers()
            if include_body:
                self.wfile.write(body)

        def _json(
            self,
            status: HTTPStatus,
            value: dict[str, Any],
            *,
            include_body: bool = True,
        ) -> None:
            body = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
            self._send_bytes(
                status,
                body,
                "application/json; charset=utf-8",
                include_body=include_body,
            )

        def _authorized(self, path: str) -> bool:
            if expected is None or path == "/api/health":
                return True
            supplied = self.headers.get("Authorization", "")
            return hmac.compare_digest(supplied, expected)

        def _challenge(self, *, include_body: bool = True) -> None:
            body = b"Authentication required"
            self.send_response(HTTPStatus.UNAUTHORIZED.value)
            self.send_header("WWW-Authenticate", 'Basic realm="hERG research website", charset="UTF-8"')
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._security_headers()
            self.end_headers()
            if include_body:
                self.wfile.write(body)

        def _get(self, *, include_body: bool) -> None:
            path = urlsplit(self.path).path
            if not self._authorized(path):
                self._challenge(include_body=include_body)
                return
            if path == "/api/health":
                self._json(
                    HTTPStatus.OK,
                    {"status": "ok", "schema_version": WEBSITE_SCHEMA},
                    include_body=include_body,
                )
                return
            if path == "/api/info":
                self._json(HTTPStatus.OK, predictor.info, include_body=include_body)
                return
            if path == "/api/internal-examples":
                if internal_examples is None:
                    self._json(
                        HTTPStatus.NOT_FOUND,
                        {"error": "Local internal examples are not enabled"},
                        include_body=include_body,
                    )
                else:
                    self._json(HTTPStatus.OK, internal_examples, include_body=include_body)
                return
            if path == "/robots.txt":
                self._send_bytes(
                    HTTPStatus.OK,
                    b"User-agent: *\nDisallow: /\n",
                    "text/plain; charset=utf-8",
                    include_body=include_body,
                )
                return
            if path == "/favicon.ico":
                self._send_bytes(
                    HTTPStatus.NO_CONTENT,
                    b"",
                    "image/x-icon",
                    include_body=include_body,
                )
                return
            asset = STATIC_ROUTES.get(path)
            if asset is None:
                self._json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "Page not found"},
                    include_body=include_body,
                )
                return
            filename, content_type, cache_control = asset
            body = (web_root / filename).read_bytes()
            self._send_bytes(
                HTTPStatus.OK,
                body,
                content_type,
                cache_control,
                include_body=include_body,
            )

        def do_GET(self) -> None:  # noqa: N802
            self._get(include_body=True)

        def do_HEAD(self) -> None:  # noqa: N802
            self._get(include_body=False)

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if not self._authorized(path):
                self.close_connection = True
                self._challenge()
                return
            if path not in {"/api/predict", "/api/properties", "/api/compare", "/api/generate"}:
                self.close_connection = True
                self._json(HTTPStatus.NOT_FOUND, {"error": "Endpoint not found"})
                return
            try:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except (TypeError, ValueError) as error:
                    self.close_connection = True
                    raise WebsiteError("Content-Length must be a valid integer") from error
                if length <= 0:
                    self.close_connection = True
                    raise WebsiteError("Prediction request is empty")
                if length > MAX_BODY_BYTES:
                    self.close_connection = True
                    self._json(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Prediction request is too large"}
                    )
                    return
                body = self.rfile.read(length)
                content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if content_type != "application/json":
                    raise WebsiteError("Send the prediction request as JSON")
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise WebsiteError("Prediction request must be a JSON object")
                if path == "/api/properties":
                    result = predictor.properties(payload.get("smiles", ""))
                elif path == "/api/generate":
                    result = predictor.generate_candidates(
                        payload.get("parent_smiles", ""),
                        payload.get("max_candidates", MAX_GENERATED_CANDIDATES),
                    )
                elif path == "/api/compare":
                    result = predictor.compare(
                        payload.get("parent_smiles", ""),
                        payload.get("candidate_smiles", ""),
                        payload.get("known_parent_ic50_um"),
                        payload.get("mw_delta_stress_test", False),
                    )
                else:
                    receptor_ids = payload.get("receptor_ids")
                    if receptor_ids is None and "receptors" in payload:
                        receptor_ids = payload.get("receptors")
                    result = predictor.predict(
                        payload.get("smiles", ""),
                        payload.get("mode", "ligand"),
                        receptor_ids,
                        payload.get("feature_mode", "atomwise"),
                        payload.get("heavy_molecule_analysis", False),
                    )
                self._json(HTTPStatus.OK, result)
            except json.JSONDecodeError:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "Prediction request contains invalid JSON"})
            except WebsiteError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:  # noqa: BLE001
                message = str(error).strip()
                if "parse" in message.lower() or "smiles" in message.lower():
                    safe_message = "The SMILES string could not be parsed. Check the structure and try again."
                else:
                    safe_message = "The prediction could not be completed. Please try again."
                print(f"[prediction error] {type(error).__name__}: {error}", flush=True)
                self._json(HTTPStatus.BAD_REQUEST, {"error": safe_message})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            print("[hERG website] " + format % args, flush=True)

    return Handler


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "serve"):
        child = subparsers.add_parser(command)
        child.add_argument("--web-root", default=str(WEB_ROOT))
        child.add_argument("--v101-root", default=str(DEFAULT_V101))
        child.add_argument("--v102-root", default=str(DEFAULT_V102))
        child.add_argument("--v103-root", default=str(DEFAULT_V103))
        child.add_argument("--v139-root", default=str(DEFAULT_V139))
        child.add_argument("--v14-root", default=str(DEFAULT_V14))
        child.add_argument("--v141-root", default=str(DEFAULT_V141))
        if command == "serve":
            child.add_argument("--host", default="127.0.0.1")
            child.add_argument("--port", type=int, default=8794)
            child.add_argument("--username", default=os.environ.get("HERG_WEBSITE_USERNAME", "research"))
            child.add_argument("--password-env", default="HERG_WEBSITE_PASSWORD")
            child.add_argument("--receptor-cpu", type=int, default=6, choices=range(1, 7))
            child.add_argument("--enable-internal-examples", action="store_true")
            child.add_argument(
                "--internal-examples-workbook",
                default=str(DEFAULT_INTERNAL_EXAMPLES),
            )
    return parser


def main() -> int:
    args = _parser().parse_args()
    web_root = Path(args.web_root).resolve()
    _validate_static_files(web_root)
    predictor = WebsitePredictor(
        Path(args.v101_root),
        Path(args.v102_root),
        Path(args.v103_root),
        Path(args.v14_root),
        Path(args.v141_root),
        Path(args.v139_root),
        args.receptor_cpu if args.command == "serve" else 6,
    )
    if args.command == "validate":
        print(json.dumps({"status": "ok", "schema_version": WEBSITE_SCHEMA}, sort_keys=True))
        return 0
    password = os.environ.get(args.password_env) or None
    if args.host not in {"127.0.0.1", "localhost", "::1"} and password is None:
        raise SystemExit(f"Set {args.password_env} before binding outside localhost")
    internal_examples = None
    if args.enable_internal_examples:
        if args.host not in {"127.0.0.1", "localhost", "::1"}:
            raise SystemExit("Internal examples may be enabled only on localhost")
        internal_examples = _load_internal_examples(Path(args.internal_examples_workbook))
        predictor.info["capabilities"]["internal_examples"] = {
            "available": True,
            "endpoint": "/api/internal-examples",
            "count": len(internal_examples["examples"]),
            "scope": "localhost_private_only",
        }
    else:
        predictor.info["capabilities"]["internal_examples"] = {
            "available": False,
            "scope": "disabled_to_prevent_private_structure_disclosure",
        }
    server = ThreadingHTTPServer(
        (args.host, args.port),
        _handler(predictor, web_root, args.username, password, internal_examples),
    )
    auth_label = "password protected" if password else "local preview"
    print(f"hERG Prediction Platform ({auth_label}): http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
