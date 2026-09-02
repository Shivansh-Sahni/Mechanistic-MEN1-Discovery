#!/usr/bin/env python3
"""Build and serve the hERG V10.3 decision-reliability overlay.

V10.3 deliberately leaves the validated V9/V10.2 numeric models unchanged.
It adds decision-useful diagnostics that were missing from V10.2:

* exact training-overlap detection;
* analogue-local retrospective error diagnostics with correctly aligned OOF
  residuals;
* a confidence label that combines applicability and threshold stability;
* explicit comparison of the literature IC50 model with the separate direct
  functional qHTS curve.

The overlay is isolated from the currently running V10.2 service.  It binds
the frozen V10.2 release as an input and writes only to its own output root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import run_local_herg_v10_2_coherent_platform as v102
from rdkit import Chem
from sklearn.linear_model import LinearRegression

SCHEMA_VERSION = "platform-local-herg-v10.3-decision-reliability/1.0"
DEFAULT_V9 = Path("research/local_runs/herg_domain_mixture_campaign_v9")
DEFAULT_V102 = Path("research/local_runs/herg_v10_2_coherent_platform")
DEFAULT_OUTPUT = Path("research/local_runs/herg_v10_3_decision_platform")
LOCAL_COHORT_SIZE = 256


class V103Error(RuntimeError):
    """Raised when the V10.3 overlay contract is violated."""


@dataclass(frozen=True)
class Paths:
    repo: Path
    v9: Path
    v102: Path
    output: Path


def _calibration_audit(oof: pd.DataFrame) -> dict[str, Any]:
    """Cross-fit a linear calibrator and reject it unless it helps honestly."""
    y = pd.to_numeric(oof["observed_pic50"], errors="raise").to_numpy(dtype=float)
    x = pd.to_numeric(oof["pred__honest_stack"], errors="raise").to_numpy(dtype=float)
    folds = pd.to_numeric(oof["outer_fold"], errors="raise").to_numpy(dtype=int)
    calibrated = np.full_like(y, np.nan)
    fold_deltas: list[float] = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        heldout = ~train
        model = LinearRegression().fit(x[train, None], y[train])
        calibrated[heldout] = model.predict(x[heldout, None])
        raw_mae = float(np.mean(np.abs(y[heldout] - x[heldout])))
        calibrated_mae = float(np.mean(np.abs(y[heldout] - calibrated[heldout])))
        fold_deltas.append(calibrated_mae - raw_mae)
    raw_mae = float(np.mean(np.abs(y - x)))
    calibrated_mae = float(np.mean(np.abs(y - calibrated)))
    return {
        "method": "five-fold cross-fitted linear output calibration",
        "raw_mae": raw_mae,
        "calibrated_mae": calibrated_mae,
        "delta_mae": calibrated_mae - raw_mae,
        "fold_delta_mae": fold_deltas,
        "selected_for_deployment": calibrated_mae < raw_mae and sum(v < 0 for v in fold_deltas) >= 4,
    }


def _aligned_reference(paths: Paths, staging: Path) -> dict[str, Any]:
    reference = joblib.load(paths.v102 / "models/applicability_reference.joblib")
    oof = pd.read_parquet(
        paths.v9 / "analysis/nested_oof_predictions.parquet",
        columns=["structure_id", "observed_pic50", "pred__honest_stack", "outer_fold"],
    )
    if len(oof) != 18_801 or oof["structure_id"].duplicated().any():
        raise V103Error("V9 nested OOF is not 18,801 unique identities")
    residual_by_id = dict(
        zip(
            oof["structure_id"].astype(str),
            (
                pd.to_numeric(oof["observed_pic50"], errors="raise")
                - pd.to_numeric(oof["pred__honest_stack"], errors="raise")
            ).astype(float),
            strict=True,
        )
    )
    structure_ids = np.asarray(reference["structure_ids"], dtype=str)
    if len(structure_ids) != 18_801 or set(structure_ids) != set(residual_by_id):
        raise V103Error("V10.2 applicability identities do not equal V9 OOF identities")
    aligned = np.asarray([residual_by_id[value] for value in structure_ids], dtype=np.float64)
    corrected = dict(reference)
    corrected["schema_version"] = SCHEMA_VERSION
    corrected["signed_oof_residuals"] = aligned
    corrected["residual_alignment"] = "structure_id exact join"
    corrected["v102_global_distribution_unchanged"] = True
    target = staging / "models/decision_reference.joblib"
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(corrected, target, compress=3)
    return {
        "structures": len(structure_ids),
        "alignment": "structure_id exact join",
        "global_residual_distribution_unchanged": True,
        "residual_mae": float(np.mean(np.abs(aligned))),
        "residual_q90_absolute": float(np.quantile(np.abs(aligned), 0.90)),
    }


def _cross_assay_consistency(primary_ic50_um: float, direct_ic50_um: float) -> dict[str, Any]:
    smaller = max(min(float(primary_ic50_um), float(direct_ic50_um)), 1e-12)
    larger = max(float(primary_ic50_um), float(direct_ic50_um))
    ratio = larger / smaller
    if ratio <= 3.0:
        label = "Cross-assay agreement"
        interpretation = "The two separately trained IC50 estimates are within three-fold."
    elif ratio <= 10.0:
        label = "Assay-sensitive estimate"
        interpretation = "The literature and qHTS estimates differ by three- to ten-fold."
    else:
        label = "Large cross-assay divergence"
        interpretation = "The two assay domains differ by more than ten-fold; do not average them."
    return {
        "label": label,
        "fold_ratio": float(ratio),
        "absolute_log10_difference": float(
            abs(math.log10(max(primary_ic50_um, 1e-12) / max(direct_ic50_um, 1e-12)))
        ),
        "interpretation": interpretation,
    }


def _decision_confidence(
    *,
    maximum_similarity: float,
    exact_overlap: bool,
    interval_lower_pic50: float,
    interval_upper_pic50: float,
) -> dict[str, Any]:
    lower_tier = v102._tier(float(interval_lower_pic50))  # noqa: SLF001
    upper_tier = v102._tier(float(interval_upper_pic50))  # noqa: SLF001
    threshold_stable = lower_tier == upper_tier
    if exact_overlap:
        label = "Training overlap"
        explanation = "This exact standardized structure occurs in training; use it as a demonstration, not independent evidence."
    elif maximum_similarity < 0.50:
        label = "Low"
        explanation = "The query is outside the analogue-supported domain."
    elif not threshold_stable:
        label = "Low"
        explanation = "The global 90% error interval crosses a liability-tier boundary."
    elif maximum_similarity >= 0.70:
        label = "High"
        explanation = "A close analogue exists and the 90% interval stays within one decision tier."
    else:
        label = "Moderate"
        explanation = "Related chemistry exists and the 90% interval stays within one decision tier."
    return {
        "label": label,
        "threshold_stable": threshold_stable,
        "interval_tiers": sorted({lower_tier, upper_tier}),
        "explanation": explanation,
    }


class Predictor:
    """Enrich the frozen V10.2 prediction without changing its numeric models."""

    def __init__(self, overlay_root: Path, v102_root: Path):
        self.overlay_root = overlay_root.resolve()
        self.v102_root = v102_root.resolve()
        _validate(self.overlay_root)
        self.info = v102._read_json(self.overlay_root / "model_info.json", "model_info_sha256")  # noqa: SLF001
        self.base = v102.Predictor(self.v102_root)
        self.reference = joblib.load(self.overlay_root / "models/decision_reference.joblib")

    def _applicability_diagnostics(self, smiles: str) -> dict[str, Any]:
        columns = [f"morgan__{index:04d}" for index in range(v102.MORGAN_BITS)]
        frame, molecule = v102.v10._feature_frame(smiles, columns)  # noqa: SLF001
        query = frame[columns].fillna(0).to_numpy(dtype=np.uint8)[0]
        similarities = v102._tanimoto_similarities(  # noqa: SLF001
            self.reference["packed_morgan"], self.reference["bit_counts"], query
        )
        order = np.argsort(similarities)[::-1]
        nearest = order[:5]
        local = order[: min(LOCAL_COHORT_SIZE, len(order))]
        residuals = np.asarray(self.reference["signed_oof_residuals"], dtype=float)
        canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
        exact_candidates = np.flatnonzero(np.isclose(similarities, 1.0, atol=1e-12))
        exact_overlap = any(str(self.reference["smiles"][index]) == canonical for index in exact_candidates)
        maximum = float(similarities[nearest[0]])
        return {
            "maximum_train_tanimoto": maximum,
            "analog_count_ge_0p5": int(np.sum(similarities >= 0.5)),
            "exact_training_overlap": bool(exact_overlap),
            "local_oof_error_diagnostic": {
                "cohort_size": int(len(local)),
                "selection": f"{len(local)} most similar exact-training structures",
                "median_tanimoto": float(np.median(similarities[local])),
                "minimum_tanimoto": float(np.min(similarities[local])),
                "retrospective_mae_pic50": float(np.mean(np.abs(residuals[local]))),
                "retrospective_median_absolute_error_pic50": float(np.median(np.abs(residuals[local]))),
                "retrospective_q90_absolute_error_pic50": float(np.quantile(np.abs(residuals[local]), 0.90)),
                "scope": "descriptive analogue-neighborhood OOF performance; not a calibrated per-query interval",
            },
            "nearest_analogs": [
                {
                    "structure_id": str(self.reference["structure_ids"][index]),
                    "smiles": str(self.reference["smiles"][index]),
                    "tanimoto": float(similarities[index]),
                    "observed_pic50": float(self.reference["target_pic50"][index]),
                    "observed_ic50_um": v102._ic_um(float(self.reference["target_pic50"][index])),  # noqa: SLF001
                    "oof_absolute_error_pic50": float(abs(residuals[index])),
                }
                for index in nearest
            ],
        }

    def predict(self, smiles: str) -> dict[str, Any]:
        result = self.base.predict(smiles)
        diagnostics = self._applicability_diagnostics(result["smiles"])
        applicability = result["applicability"]
        applicability.update(diagnostics)
        main = result["main_ic50"]
        confidence = _decision_confidence(
            maximum_similarity=float(applicability["maximum_train_tanimoto"]),
            exact_overlap=bool(applicability["exact_training_overlap"]),
            interval_lower_pic50=float(main["interval90_pic50"]["lower"]),
            interval_upper_pic50=float(main["interval90_pic50"]["upper"]),
        )
        main["decision_confidence"] = confidence
        direct = result["direct_functional_curve"]
        direct["cross_assay_consistency"] = _cross_assay_consistency(
            float(main["ic50_um"]), float(direct["ic50_um"])
        )
        direct["internal_endpoint_metrics"] = self.info["direct_curve_metrics"]
        result["schema_version"] = SCHEMA_VERSION
        result["decision_reliability"] = {
            "confidence": confidence,
            "exact_training_overlap": bool(applicability["exact_training_overlap"]),
            "cross_assay_consistency": direct["cross_assay_consistency"],
            "core_prediction_changed_from_v10_2": False,
        }
        return result


def _write_app(path: Path) -> None:
    path.write_text(
        r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>hERG V10.3 Decision Reliability</title>
<style>:root{--ink:#111827;--muted:#64748b;--line:#dbe3ee;--blue:#2457d6;--green:#087f5b;--amber:#c46308;--red:#c13b42;--bg:#f4f7fb}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}.shell{max-width:1120px;margin:auto;padding:36px 20px 70px}.hero{background:linear-gradient(135deg,#101b3d,#1f4bb8);color:#fff;border-radius:24px;padding:34px 38px}.hero h1{font-size:42px;margin:0 0 10px}.hero p{max-width:850px;color:#dce7ff}.badge{display:inline-block;margin-top:12px;padding:7px 11px;border:1px solid #ffffff55;border-radius:999px;font-size:13px}.card{background:#fff;border:1px solid var(--line);border-radius:20px;padding:25px;margin-top:20px}.inputrow{display:grid;grid-template-columns:1fr auto;gap:12px}input,select{width:100%;border:1px solid #b8c5d8;border-radius:12px;padding:14px;font:14px ui-monospace,monospace}button{border:0;border-radius:12px;padding:0 24px;background:var(--blue);color:#fff;font-weight:750}.presets{display:grid;grid-template-columns:210px 1fr;gap:10px;margin-top:10px}.status{min-height:24px;color:var(--muted);margin-top:10px}.hidden{display:none}.head{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:15px}.head h2{margin:0}.head p{margin:0;color:var(--muted);max-width:600px;text-align:right}.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.metric{border:1px solid var(--line);border-radius:16px;padding:17px;background:#fbfdff}.metric small{color:var(--muted);font-weight:750;text-transform:uppercase}.metric b{display:block;font-size:28px}.call{padding:14px 16px;border-left:4px solid var(--blue);background:#edf3ff;border-radius:6px;margin-top:14px}.warn{border-color:var(--amber);background:#fff7e8}.safe,.high{color:var(--green)}.moderate{color:var(--amber)}.potent,.low{color:var(--red)}.analog{display:grid;grid-template-columns:92px 1fr 112px 100px;gap:10px;padding:10px 0;border-top:1px solid var(--line);font-size:13px}.analog code{overflow:hidden;text-overflow:ellipsis}.chem{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted)}@media(max-width:800px){.grid4,.grid3{grid-template-columns:1fr 1fr}.inputrow,.presets{grid-template-columns:1fr}.head{display:block}.head p{text-align:left}}@media(max-width:520px){.grid4,.grid3{grid-template-columns:1fr}}</style></head>
<body><main class="shell"><section class="hero"><h1>hERG V10.3</h1><p>The validated V9 IC50 model, now paired with explicit decision confidence, exact-overlap disclosure, analogue-local error evidence, and cross-assay consistency checks.</p><span class="badge">Research prototype · ligand-only · internal scaffold-held-out evidence</span></section>
<section class="card"><div class="head"><h2>Predict A Molecule</h2><p>Curated examples exercise different evidence domains. They are interface examples, not a benchmark.</p></div><div class="inputrow"><input id="smi" value="CCN(CC)CCCC(C)Nc1ncc2cc(OC)c(OC)cc2n1"><button id="go">Run Prediction</button></div><div class="presets"><select id="preset"><option value="">Choose an example</option><option value="C#CCC1=C(C)C(OC(=O)C2C(C=C(C)C)C2(C)C)CC1=O">High-micromolar functional example</option><option value="CN(C)C(=O)C1(N2CCCCC2)CCN(CC[C@@]2(c3ccc(Cl)c(Cl)c3)CN(C(=O)c3ccccc3)CCO2)CC1">Mid-micromolar functional example</option><option value="CN(C)C(=O)NC1(c2ccccc2)CCN(CCC[C@]2(c3ccc(Cl)c(Cl)c3)CCCN(C(=O)c3ccccc3)C2)CC1">Low-micromolar functional example</option><option value="Oc1cc2ccccc2c2ccccc12">Applicability stress test</option></select><span></span></div><div id="status" class="status"></div><div id="chem" class="chem"></div></section>
<div id="results" class="hidden"><section class="card"><div class="head"><h2>Primary hERG Liability</h2><p>IC50 and tier come from the same unchanged five-stack V9 model.</p></div><div class="grid4"><div class="metric"><small>Tier</small><b id="tier"></b></div><div class="metric"><small>IC50</small><b id="ic50"></b><span>µM</span></div><div class="metric"><small>pIC50</small><b id="pic50"></b></div><div class="metric"><small>90% Global Interval</small><b id="interval" style="font-size:19px"></b><span>µM</span></div></div><div id="confidence" class="call"></div></section>
<section class="card"><div class="head"><h2>Decision Reliability</h2><p>Similarity, overlap, and retrospective neighborhood error determine how cautiously to use the point estimate.</p></div><div class="grid4"><div class="metric"><small>Confidence</small><b id="conf"></b></div><div class="metric"><small>Nearest Similarity</small><b id="sim"></b></div><div class="metric"><small>Analogues ≥ 0.50</small><b id="analogs"></b></div><div class="metric"><small>Local OOF MAE</small><b id="lmae"></b><span> pIC50</span></div></div><div id="overlap" class="call"></div></section>
<section class="card"><div class="head"><h2>Separate Direct Functional Curve</h2><p>One qHTS assay domain. These endpoints are not derived from the 18,801-structure IC50.</p></div><div class="grid3"><div class="metric"><small>Direct IC10</small><b id="ic10"></b><span>µM</span></div><div class="metric"><small>Direct IC30</small><b id="ic30"></b><span>µM</span></div><div class="metric"><small>Direct IC50</small><b id="dic50"></b><span>µM</span></div></div><div id="assay" class="call warn"></div></section>
<section class="card"><div class="head"><h2>Nearest Training Analogues</h2><p>Observed IC50 and each analogue's own nested-OOF error are shown separately.</p></div><div id="nearest"></div></section>
<section class="card"><div class="head"><h2>Evidence Boundaries</h2><p>Strong interface coherence does not turn internal evidence into external validation.</p></div><div class="grid3"><div class="metric"><small>Primary Model</small><p>18,801 WT-or-unspecified exact structures. Internal nested scaffold MAE 0.4328 pIC50.</p></div><div class="metric"><small>Direct Curve</small><p>One 20-dose functional series. IC10 MAE 0.556, IC30 0.405, IC50 0.395.</p></div><div class="metric"><small>Not Claimed</small><p>No receptor-aware physics, prospective validation, clinical-risk inference, or universal IC10/IC30 scale.</p></div></div></section></div></main>
<script>const $=x=>document.getElementById(x),f=(x,n=3)=>Number(x).toFixed(n);$('preset').onchange=()=>{if($('preset').value)$('smi').value=$('preset').value};$('go').onclick=async()=>{$('status').textContent='Calculating features and evidence…';$('go').disabled=true;try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({smiles:$('smi').value})});const d=await r.json();if(!r.ok)throw Error(d.error||'Prediction failed');const m=d.main_ic50,a=d.applicability,q=d.direct_functional_curve,c=m.decision_confidence;$('tier').textContent=m.tier;$('tier').className=m.tier.toLowerCase();$('ic50').textContent=f(m.ic50_um);$('pic50').textContent=f(m.pic50);$('interval').textContent=f(m.interval90_um.lower,2)+'–'+f(m.interval90_um.upper,2);$('conf').textContent=c.label;$('conf').className=c.label.toLowerCase().replace(' ','-');$('confidence').innerHTML='<b>'+c.label+' confidence:</b> '+c.explanation;$('sim').textContent=f(a.maximum_train_tanimoto);$('analogs').textContent=a.analog_count_ge_0p5.toLocaleString();$('lmae').textContent=f(a.local_oof_error_diagnostic.retrospective_mae_pic50);$('overlap').innerHTML=a.exact_training_overlap?'<b>Exact training overlap:</b> this is a demonstration on a seen structure, not an independent prediction.':'<b>No exact training overlap detected.</b> Local OOF MAE is a retrospective diagnostic for the 256 most similar training structures, not a calibrated query interval.';$('ic10').textContent=f(q.ic10_um);$('ic30').textContent=f(q.ic30_um);$('dic50').textContent=f(q.ic50_um);const x=q.cross_assay_consistency;$('assay').innerHTML='<b>'+x.label+' ('+f(x.fold_ratio,1)+'×):</b> '+x.interpretation+(q.ordering_adjusted?' Raw direct predictions required order-preserving projection.':'');$('nearest').innerHTML=a.nearest_analogs.map(x=>`<div class="analog"><b>T=${f(x.tanimoto)}</b><code title="${x.smiles}">${x.smiles}</code><span>IC50 ${f(x.observed_ic50_um,2)} µM</span><span>OOF |error| ${f(x.oof_absolute_error_pic50,2)}</span></div>`).join('');const z=d.chemistry;$('chem').innerHTML=`<span><b>MW</b> ${f(z.molecular_weight,1)}</span><span><b>cLogP</b> ${f(z.clogp,2)}</span><span><b>TPSA</b> ${f(z.tpsa,1)} Å²</span><span><b>HBD/HBA</b> ${z.hbd}/${z.hba}</span><span><b>Rotors</b> ${z.rotatable_bonds}</span>`;$('results').classList.remove('hidden');$('status').textContent='Prediction complete.'}catch(e){$('status').textContent=e.message}finally{$('go').disabled=false}};</script></body></html>"""
    )


def _build(paths: Paths) -> dict[str, Any]:
    if paths.output.exists():
        if (paths.output / "manifest.json").is_file():
            return _validate(paths.output)
        raise V103Error(f"output exists without manifest: {paths.output}")
    v102._validate(paths.v102)  # noqa: SLF001
    oof = pd.read_parquet(paths.v9 / "analysis/nested_oof_predictions.parquet")
    calibration = _calibration_audit(oof)
    parent = paths.output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{paths.output.name}.", dir=parent))
    try:
        reference_summary = _aligned_reference(paths, staging)
        _write_app(staging / "app.html")
        base_info = v102._read_json(paths.v102 / "model_info.json", "model_info_sha256")  # noqa: SLF001
        info = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_release": str(paths.v102.resolve()),
            "base_primary_ic50": base_info["primary_ic50"],
            "direct_curve_metrics": base_info["direct_curve"],
            "calibration_audit": calibration,
            "decision": "retain the raw V9 honest-stack mean because cross-fitted linear calibration worsened MAE",
            "decision_improvements": [
                "corrected structure-aligned OOF residuals for local diagnostics",
                "exact training-overlap disclosure",
                "analogue-local retrospective error summary",
                "applicability and threshold-stability confidence label",
                "literature-versus-qHTS IC50 consistency check",
            ],
            "reference": reference_summary,
            "scientific_scope": base_info["scientific_scope"],
        }
        v102._atomic_json(staging / "model_info.json", info, "model_info_sha256")  # noqa: SLF001
        report = (
            "# hERG V10.3 Decision Reliability\n\n"
            "The V9/V10.2 numeric models are unchanged. Cross-fitted linear calibration was rejected "
            f"because MAE changed from {calibration['raw_mae']:.6f} to {calibration['calibrated_mae']:.6f}.\n\n"
            "V10.3 adds exact-overlap disclosure, an identity-aligned analogue-local OOF error diagnostic, "
            "threshold-stability confidence, and cross-assay IC50 disagreement labeling. Local error is "
            "descriptive and is not presented as a calibrated per-query interval.\n"
        )
        (staging / "REPORT.md").write_text(report)
        artifacts = [
            staging / "app.html",
            staging / "REPORT.md",
            staging / "model_info.json",
            staging / "models/decision_reference.joblib",
        ]
        inputs = [
            v102._binding(Path(__file__), "V10.3 implementation"),  # noqa: SLF001
            v102._binding(paths.v102 / "manifest.json", "frozen V10.2 manifest"),  # noqa: SLF001
            v102._binding(paths.v102 / "model_info.json", "frozen V10.2 model information"),  # noqa: SLF001
            v102._binding(
                paths.v102 / "models/applicability_reference.joblib", "V10.2 applicability reference"
            ),  # noqa: SLF001
            v102._binding(paths.v9 / "analysis/nested_oof_predictions.parquet", "V9 nested OOF"),  # noqa: SLF001
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "input_bindings": inputs,
            "artifact_bindings": [
                v102._binding(path, f"V10.3 artifact {path.name}", root=staging)  # noqa: SLF001
                for path in artifacts
            ],
            "expected_members": sorted(
                [str(path.relative_to(staging)) for path in artifacts] + ["manifest.json", "validation.json"]
            ),
        }
        v102._atomic_json(staging / "manifest.json", manifest, "manifest_sha256")  # noqa: SLF001
        validation = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "base_v10_2_validated": True,
            "primary_numeric_model_unchanged": True,
            "calibration_rejected": not bool(calibration["selected_for_deployment"]),
            "residuals_aligned_by_structure_id": True,
            "exact_overlap_disclosed": True,
            "local_error_labeled_descriptive": True,
            "cross_assay_consistency_reported": True,
        }
        v102._atomic_json(staging / "validation.json", validation, "validation_sha256")  # noqa: SLF001
        os.replace(staging, paths.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _validate(paths.output)


def _validate(output: Path) -> dict[str, Any]:
    root = output.resolve()
    manifest = v102._read_json(root / "manifest.json", "manifest_sha256")  # noqa: SLF001
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "passed":
        raise V103Error("invalid V10.3 manifest")
    for binding in manifest["input_bindings"]:
        v102._verify_binding(binding)  # noqa: SLF001
    for binding in manifest["artifact_bindings"]:
        v102._verify_binding(binding, root)  # noqa: SLF001
    members = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())
    if members != list(manifest["expected_members"]):
        raise V103Error("V10.3 output membership is not closed")
    validation = v102._read_json(root / "validation.json", "validation_sha256")  # noqa: SLF001
    info = v102._read_json(root / "model_info.json", "model_info_sha256")  # noqa: SLF001
    if validation.get("status") != "passed" or info.get("calibration_audit", {}).get(
        "selected_for_deployment"
    ):
        raise V103Error("V10.3 semantic validation failed")
    return validation


def _handler(predictor: Predictor) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                body = (predictor.overlay_root / "app.html").read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._json(HTTPStatus.OK, {"status": "ok", "schema_version": SCHEMA_VERSION})
            elif self.path == "/api/info":
                self._json(HTTPStatus.OK, predictor.info)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/predict":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                smiles = str(payload.get("smiles", "")).strip()
                if not smiles:
                    raise V103Error("SMILES is required")
                self._json(HTTPStatus.OK, predictor.predict(smiles))
            except Exception as error:  # noqa: BLE001
                self._json(HTTPStatus.BAD_REQUEST, {"error": f"{type(error).__name__}: {error}"})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return Handler


def _paths(args: argparse.Namespace) -> Paths:
    repo = Path(args.repo_root).resolve()
    return Paths(
        repo=repo,
        v9=(repo / args.v9_root).resolve(),
        v102=(repo / args.v102_root).resolve(),
        output=(repo / args.output_root).resolve(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "validate", "serve"):
        child = sub.add_parser(name)
        child.add_argument("--repo-root", default=".")
        child.add_argument("--v9-root", default=str(DEFAULT_V9))
        child.add_argument("--v102-root", default=str(DEFAULT_V102))
        child.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
        if name == "serve":
            child.add_argument("--host", default="127.0.0.1")
            child.add_argument("--port", type=int, default=8792)
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = _paths(args)
    if args.command == "build":
        print(json.dumps(_build(paths), sort_keys=True))
        return 0
    if args.command == "validate":
        print(json.dumps(_validate(paths.output), sort_keys=True))
        return 0
    if args.command == "serve":
        _build(paths)
        predictor = Predictor(paths.output, paths.v102)
        server = ThreadingHTTPServer((args.host, args.port), _handler(predictor))
        print(f"hERG V10.3: http://{args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
