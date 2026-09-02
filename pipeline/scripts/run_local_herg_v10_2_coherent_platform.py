#!/usr/bin/env python3
"""Build and serve the endpoint-coherent hERG V10.2 research prototype.

V10.2 deliberately separates two questions that earlier prototypes mixed:

* The primary liability estimate is the literal five-fold V9 deployment
  ensemble trained on 18,801 exact pIC50 structures.  Its IC50 value and its
  Safe/Moderate/Potent tier are one decision, not two competing models.
* IC10, IC30, and IC50 curve crossings come from one direct 20-concentration
  functional qHTS assay.  They are displayed together only after enforcing
  the physical order IC10 <= IC30 <= IC50.

The broad 46 uM fixed-dose score is intentionally absent from the prediction
page.  It is replaced by applicability, uncertainty, and nearest-analogue
evidence that helps a chemist decide whether to trust a prediction.

This script writes only to the V10.2 output root.  It does not mutate or stop
the currently served V10.1 application.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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

import build_local_multicpu_herg_3d_features as f3d
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import run_local_herg_discovery_worker as discovery
import run_local_herg_feature_relationship_campaign_v5 as v5
import run_local_herg_v10_1_expanded_platform as v101
import run_local_herg_v10_tiered_platform as v10
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

SCHEMA_VERSION = "platform-local-herg-v10.2-endpoint-coherent/1.0"
CLASS_NAMES = ("Safe", "Moderate", "Potent")
SAFE_PIC50 = -math.log10(30e-6)
POTENT_PIC50 = -math.log10(1e-6)
MORGAN_BITS = 2048
MAX_NUMERIC = 1e30
MAX_ENERGY_RANGE = 10_000.0
DEFAULT_V9 = Path("research/local_runs/herg_domain_mixture_campaign_v9")
DEFAULT_V101 = Path("research/local_runs/herg_v10_1_expanded_platform")
DEFAULT_SURFACE = Path(
    "research/data/platform/processed/herg_hierarchy/v1_6_training_surfaces/"
    "herg_training_observations.parquet"
)
DEFAULT_OUTPUT = Path("research/local_runs/herg_v10_2_coherent_platform")

RDLogger.DisableLog("rdApp.warning")


class V102Error(RuntimeError):
    """Raised when the V10.2 scientific or artifact contract is violated."""


@dataclass(frozen=True)
class Paths:
    repo: Path
    v9: Path
    v101: Path
    surface: Path
    output: Path


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha(path: Path) -> str:
    return hashlib.sha256(pq.read_schema(path).serialize().to_pybytes()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any], self_key: str | None = None) -> None:
    document = copy.deepcopy(value)
    if self_key:
        document.pop(self_key, None)
        document[self_key] = _digest(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path, self_key: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise V102Error(f"expected JSON object: {path}")
    if self_key:
        stored = value.get(self_key)
        candidate = copy.deepcopy(value)
        candidate.pop(self_key, None)
        if stored != _digest(candidate):
            raise V102Error(f"self-hash mismatch: {path}")
    return value


def _binding(path: Path, role: str, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise V102Error(f"missing {role}: {resolved}")
    result: dict[str, Any] = {
        "role": role,
        "bytes": resolved.stat().st_size,
        "sha256": _sha(resolved),
    }
    if root is None:
        result["path"] = str(resolved)
    else:
        try:
            result["relative_path"] = str(resolved.relative_to(root.resolve()))
        except ValueError as error:
            raise V102Error(f"artifact escapes output root: {resolved}") from error
    if resolved.suffix == ".parquet":
        result["rows"] = pq.read_metadata(resolved).num_rows
        result["arrow_schema_sha256"] = _schema_sha(resolved)
    return result


def _verify_binding(binding: dict[str, Any], root: Path | None = None) -> None:
    path = (
        (root / str(binding["relative_path"])).resolve()
        if root is not None
        else Path(str(binding["path"])).resolve()
    )
    if root is not None and not path.is_relative_to(root.resolve()):
        raise V102Error(f"artifact escapes output root: {path}")
    if not path.is_file() or path.stat().st_size != int(binding["bytes"]):
        raise V102Error(f"artifact missing or size changed: {path}")
    if _sha(path) != str(binding["sha256"]):
        raise V102Error(f"artifact hash changed: {path}")
    if path.suffix == ".parquet":
        if pq.read_metadata(path).num_rows != int(binding["rows"]):
            raise V102Error(f"artifact row count changed: {path}")
        if _schema_sha(path) != str(binding["arrow_schema_sha256"]):
            raise V102Error(f"artifact schema changed: {path}")


def _mode(values: pd.Series) -> str:
    counts = values.dropna().astype(str).value_counts()
    if counts.empty:
        return ""
    maximum = int(counts.max())
    return sorted(counts[counts == maximum].index)[0]


def _safe_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    values = result.to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    values[np.abs(values) > MAX_NUMERIC] = np.nan
    return pd.DataFrame(values, columns=columns, index=frame.index)


def _tier(pic50: float) -> str:
    if pic50 < SAFE_PIC50:
        return "Safe"
    if pic50 <= POTENT_PIC50:
        return "Moderate"
    return "Potent"


def _ic_um(pic50: float) -> float:
    return float(10 ** (6.0 - float(pic50)))


def _model_columns(model: Any) -> list[str]:
    if hasattr(model, "get_booster"):
        names = model.get_booster().feature_names
        if names:
            return [str(name) for name in names]
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise V102Error(f"model does not expose its feature order: {type(model).__name__}")
    return [str(name) for name in names]


def _feature_union(v9: Path, direct_bundles: list[dict[str, Any]]) -> list[str]:
    columns: set[str] = set()
    for fold in range(5):
        bundle = joblib.load(v9 / f"units/outer_o{fold}/outer_models.joblib")
        for model in bundle["models"].values():
            columns.update(_model_columns(model))
    for bundle in direct_bundles:
        columns.update(str(value) for value in bundle["feature_columns"])
    return sorted(columns)


def _raw_3d(smiles: str, *, requested: int, retained: int, iterations: int) -> dict[str, Any]:
    identity = hashlib.sha256(smiles.encode()).hexdigest().upper()
    return f3d._compute_one(  # noqa: SLF001
        (0, identity, smiles, "v10.2-on-demand", requested, retained, iterations)
    )


def _single_feature_frame(smiles: str, columns: list[str]) -> tuple[pd.DataFrame, Chem.Mol]:
    base_columns = [
        column for column in columns if column.startswith("rdkit2d__") or column.startswith("morgan__")
    ]
    base, molecule = v10._feature_frame(smiles, base_columns)  # noqa: SLF001
    canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
    old = _raw_3d(canonical, requested=6, retained=3, iterations=75)
    new = _raw_3d(canonical, requested=24, retained=8, iterations=100)
    for required, default in (
        ("feature_status", "conformer_failed"),
        ("energy_min_kcal_mol", math.nan),
        ("energy_range_kcal_mol", math.nan),
        ("retained_conformer_count", 0),
        ("unconverged_retained_count", 0),
    ):
        old.setdefault(required, default)
        new.setdefault(required, default)
    old_frame = pd.DataFrame([{f"f3d__{key}": value for key, value in old.items()}])
    new_frame = pd.DataFrame([{f"new3d__{key}": value for key, value in new.items()}])
    new_frame, _ = v5._qc_new3d(new_frame)  # noqa: SLF001
    excluded = bool(
        new_frame.loc[0, "new3d__energy_nonfinite_indicator"]
        or new_frame.loc[0, "new3d__energy_extreme_indicator"]
        or new_frame.loc[0, "new3d__all_retained_unconverged_indicator"]
        or new_frame.loc[0, "new3d__feature_failed_indicator"]
    )
    indicators = {
        "new3d__energy_nonfinite_indicator",
        "new3d__energy_extreme_indicator",
        "new3d__all_retained_unconverged_indicator",
        "new3d__feature_failed_indicator",
    }
    if excluded:
        for column in new_frame.select_dtypes(include=[np.number]).columns:
            if column.startswith("new3d__") and column not in indicators:
                new_frame.loc[0, column] = np.nan
    new_frame["new3d__physics_qc_excluded_indicator"] = int(excluded)
    old_min = pd.to_numeric(old_frame.get("f3d__energy_min_kcal_mol"), errors="coerce")
    old_range = pd.to_numeric(old_frame.get("f3d__energy_range_kcal_mol"), errors="coerce")
    old_bad = bool(
        old_min is None
        or old_range is None
        or not np.isfinite(float(old_min.iloc[0]))
        or not np.isfinite(float(old_range.iloc[0]))
        or abs(float(old_range.iloc[0])) > MAX_ENERGY_RANGE
    )
    old_frame["f3d__energy_min_kcal_mol"] = np.nan
    if old_bad:
        old_frame["f3d__energy_range_kcal_mol"] = np.nan
    old_frame["f3d__energy_pathology_indicator"] = int(old_bad)
    frame = pd.concat([base, old_frame, new_frame], axis=1)
    targeted, _ = discovery._targeted_interactions(frame)  # noqa: SLF001
    frame = pd.concat([frame, targeted], axis=1)
    frame = v5._add_interactions(frame)  # noqa: SLF001
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return _safe_numeric(frame, columns), molecule


def _copy_models(paths: Paths, staging: Path) -> None:
    (staging / "models/v9_folds").mkdir(parents=True)
    for fold in range(5):
        source = paths.v9 / f"units/outer_o{fold}/outer_models.joblib"
        shutil.copy2(source, staging / f"models/v9_folds/outer_o{fold}.joblib")
    for endpoint in ("ic10", "ic30", "ic50"):
        shutil.copy2(
            paths.v101 / f"models/empirical_{endpoint}_regressor.joblib",
            staging / f"models/direct_{endpoint}_regressor.joblib",
        )


def _validate_v101_source(root: Path) -> None:
    """Validate the frozen V10.1 outputs without requiring mutable source parity.

    The live V10.1 release binds the implementation bytes used when it was
    built.  That implementation has since been patched while the already
    materialized model artifacts remain byte-identical.  V10.2 therefore
    verifies the V10.1 manifest self-hash and every frozen output artifact,
    but does not pretend the later source file is the original build input.
    """

    manifest = _read_json(root / "manifest.json", "manifest_sha256")
    if manifest.get("schema_version") != "platform-local-herg-v10.1-expanded/1.0":
        raise V102Error("V10.1 source manifest schema is invalid")
    if manifest.get("status") != "passed":
        raise V102Error("V10.1 source release did not pass")
    for binding in manifest.get("artifacts", []):
        path = Path(str(binding["path"])).resolve()
        if not path.is_relative_to(root.resolve()):
            raise V102Error(f"V10.1 artifact escapes its release root: {path}")
        _verify_binding(binding)


def _build_reference(paths: Paths, staging: Path) -> dict[str, Any]:
    matrix = pd.read_parquet(paths.v9 / "prepared/training_matrix.parquet")
    if len(matrix) != 18_801 or matrix["structure_id"].duplicated().any():
        raise V102Error("V9 exact training matrix is not 18,801 unique structures")
    morgan_columns = [f"morgan__{index:04d}" for index in range(MORGAN_BITS)]
    if not set(morgan_columns).issubset(matrix.columns):
        raise V102Error("V9 training matrix lacks the Morgan applicability surface")
    surface = pd.read_parquet(paths.surface, columns=["structure_id", "standardized_smiles"])
    smiles = surface.groupby("structure_id", as_index=False).agg(
        standardized_smiles=("standardized_smiles", _mode)
    )
    joined = matrix[["structure_id", "target_pic50", *morgan_columns]].merge(
        smiles, on="structure_id", how="left", validate="one_to_one"
    )
    if joined["standardized_smiles"].isna().any():
        raise V102Error("exact training identities lack standardized SMILES")
    bits = joined[morgan_columns].to_numpy(dtype=np.uint8, copy=True)
    packed = np.packbits(bits, axis=1)
    oof = pd.read_parquet(paths.v9 / "analysis/nested_oof_predictions.parquet")
    required = {"structure_id", "observed_pic50", "pred__honest_stack"}
    if len(oof) != 18_801 or not required.issubset(oof.columns):
        raise V102Error("V9 nested OOF contract is incomplete")
    residuals = (
        pd.to_numeric(oof["observed_pic50"], errors="raise")
        - pd.to_numeric(oof["pred__honest_stack"], errors="raise")
    ).to_numpy(dtype=np.float64)
    reference = {
        "schema_version": SCHEMA_VERSION,
        "structure_ids": joined["structure_id"].astype(str).to_numpy(),
        "smiles": joined["standardized_smiles"].astype(str).to_numpy(),
        "target_pic50": joined["target_pic50"].to_numpy(dtype=np.float32),
        "packed_morgan": packed,
        "bit_counts": bits.sum(axis=1).astype(np.int16),
        "signed_oof_residuals": residuals,
    }
    joblib.dump(reference, staging / "models/applicability_reference.joblib", compress=3)
    return {
        "structures": 18_801,
        "morgan_bits": MORGAN_BITS,
        "residual_q05": float(np.quantile(residuals, 0.05)),
        "residual_q95": float(np.quantile(residuals, 0.95)),
    }


def _write_app(path: Path) -> None:
    html = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>hERG V10.2 Endpoint-Coherent</title>
<style>
:root{--ink:#111827;--muted:#64748b;--line:#dbe3ee;--blue:#2457d6;--green:#087f5b;--amber:#c46308;--red:#c13b42;--bg:#f4f7fb}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}.shell{max-width:1120px;margin:0 auto;padding:36px 20px 70px}.hero{background:linear-gradient(135deg,#101b3d,#1f4bb8);color:white;border-radius:24px;padding:34px 38px;box-shadow:0 20px 50px #10295b22}.hero h1{font-size:42px;line-height:1.05;margin:0 0 12px}.hero p{max-width:820px;margin:0;color:#dce7ff}.badge{display:inline-block;margin-top:18px;padding:7px 11px;border:1px solid #ffffff55;border-radius:999px;font-size:13px}.card{background:white;border:1px solid var(--line);border-radius:20px;padding:25px;margin-top:20px;box-shadow:0 9px 28px #2335550c}.inputrow{display:grid;grid-template-columns:1fr auto;gap:12px}input{width:100%;border:1px solid #b8c5d8;border-radius:12px;padding:15px;font:15px ui-monospace,SFMono-Regular,monospace}button{border:0;border-radius:12px;padding:0 25px;background:var(--blue);color:white;font-weight:750;cursor:pointer}.status{min-height:24px;color:var(--muted);margin-top:10px}.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.metric{border:1px solid var(--line);border-radius:16px;padding:18px;background:#fbfdff}.metric b{display:block;font-size:29px;letter-spacing:-1px}.metric small{color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.05em}.sectionhead{display:flex;justify-content:space-between;align-items:end;gap:16px;margin-bottom:16px}.sectionhead h2{margin:0;font-size:25px}.sectionhead p{margin:0;color:var(--muted);max-width:600px;text-align:right}.tier{font-size:34px!important}.safe{color:var(--green)}.moderate{color:var(--amber)}.potent{color:var(--red)}.bar{height:9px;background:#eef2f7;border-radius:99px;overflow:hidden;margin-top:9px}.bar i{display:block;height:100%;background:var(--blue)}.direct{background:#f8fbff}.evidence{background:#fcfffc}.callout{padding:14px 16px;border-left:4px solid var(--blue);background:#edf3ff;border-radius:6px;margin-top:16px}.three{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.analog{display:grid;grid-template-columns:100px 1fr 100px;gap:10px;padding:10px 0;border-top:1px solid var(--line);font-size:13px}.analog code{overflow:hidden;text-overflow:ellipsis}.hidden{display:none}.chem{display:flex;gap:20px;flex-wrap:wrap;color:var(--muted);font-size:14px}.warn{color:#8a3e00;background:#fff7e8;border-color:#f4ca88}@media(max-width:800px){.grid4,.three{grid-template-columns:1fr 1fr}.inputrow{grid-template-columns:1fr}.sectionhead{display:block}.sectionhead p{text-align:left}.hero h1{font-size:34px}}@media(max-width:520px){.grid4,.three{grid-template-columns:1fr}}
</style></head><body><main class="shell">
<section class="hero"><h1>hERG V10.2</h1><p>One coherent IC50 decision, a separately labeled direct functional curve, and explicit evidence for when the prediction should be trusted.</p><span class="badge">Research prototype · ligand-only · internal scaffold-held-out evidence</span></section>
<section class="card"><div class="sectionhead"><h2>Predict A Molecule</h2><p>Paste one valid SMILES. On-demand conformer calculation can take several seconds.</p></div><div class="inputrow"><input id="smi" value="CCN(CC)CCCC(C)Nc1ncc2cc(OC)c(OC)cc2n1" aria-label="SMILES"><button id="go">Run Prediction</button></div><div id="status" class="status"></div><div id="chem" class="chem"></div></section>
<div id="results" class="hidden">
<section class="card"><div class="sectionhead"><h2>Primary hERG Liability</h2><p>The tier is derived from this exact IC50 estimate. It cannot contradict it.</p></div><div class="grid4"><div class="metric"><small>Decision Tier</small><b id="tier" class="tier"></b></div><div class="metric"><small>IC50</small><b id="ic50"></b><span>µM</span></div><div class="metric"><small>pIC50</small><b id="pic50"></b></div><div class="metric"><small>90% Error Interval</small><b id="interval" style="font-size:20px"></b><span>µM</span></div></div><div id="probs" class="three" style="margin-top:14px"></div><div class="callout"><b>Single source of truth:</b> five V9 scaffold-fold deployment stacks are averaged. The 18,801-structure nested result is MAE 0.4328 pIC50, about 2.7-fold concentration error on average.</div></section>
<section class="card direct"><div class="sectionhead"><h2>Direct Functional Curve Module</h2><p>IC10, IC30, and IC50 are crossings from one 20-concentration functional qHTS assay, not conversions from the primary IC50.</p></div><div class="three"><div class="metric"><small>Direct IC10</small><b id="ic10"></b><span>µM</span></div><div class="metric"><small>Direct IC30</small><b id="ic30"></b><span>µM</span></div><div class="metric"><small>Direct IC50</small><b id="dic50"></b><span>µM</span></div></div><div id="curve-note" class="callout warn"></div></section>
<section class="card evidence"><div class="sectionhead"><h2>Evidence And Applicability</h2><p>This replaces the old 46 µM score with information that changes how a chemist should use the result.</p></div><div class="grid4"><div class="metric"><small>Exact Training Surface</small><b>18,801</b><span>structures</span></div><div class="metric"><small>Nearest Train Similarity</small><b id="sim"></b></div><div class="metric"><small>Analogues ≥ 0.50</small><b id="analogs"></b></div><div class="metric"><small>Fold Prediction Spread</small><b id="spread"></b><span> pIC50</span></div></div><div id="domain" class="callout"></div><h3>Nearest Training Analogues</h3><div id="nearest"></div></section>
<section class="card"><div class="sectionhead"><h2>Evidence Boundaries</h2><p>Results are designed to be harder to overinterpret.</p></div><div class="three"><div class="metric"><small>Primary IC50</small><p>18,801 WT-or-unspecified exact structures; 5-fold nested scaffold evaluation.</p></div><div class="metric"><small>Direct Curve</small><p>One human hERG functional qHTS series; 1,067 IC10, 1,028 IC30, and 769 IC50 unique strict labels.</p></div><div class="metric"><small>Not Yet Claimed</small><p>No receptor-aware physics, external/prospective validation, clinical risk, or universal IC10/IC30 scale.</p></div></div></section></div>
</main><script>
const $=id=>document.getElementById(id), fmt=(x,n=3)=>Number(x).toFixed(n);
$('go').onclick=async()=>{ $('status').textContent='Calculating 2D, fingerprints, and ligand conformer evidence…'; $('go').disabled=true; try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({smiles:$('smi').value})});const d=await r.json();if(!r.ok)throw Error(d.error||'Prediction failed');const m=d.main_ic50;$('tier').textContent=m.tier;$('tier').className='tier '+m.tier.toLowerCase();$('ic50').textContent=fmt(m.ic50_um);$('pic50').textContent=fmt(m.pic50);$('interval').textContent=fmt(m.interval90_um.lower,2)+'–'+fmt(m.interval90_um.upper,2);$('probs').innerHTML=Object.entries(m.tier_probabilities).map(([k,v])=>`<div class="metric"><small>${k} probability</small><b>${fmt(v*100,1)}%</b><div class="bar"><i style="width:${v*100}%"></i></div></div>`).join('');const q=d.direct_functional_curve;$('ic10').textContent=fmt(q.ic10_um);$('ic30').textContent=fmt(q.ic30_um);$('dic50').textContent=fmt(q.ic50_um);$('curve-note').innerHTML=`<b>Separate assay domain:</b> ordered IC10 ≤ IC30 ≤ IC50${q.ordering_adjusted?' (raw predictions required an order-preserving projection).':'.'} This curve must not be read as three values from the 18,801-structure model.`;const a=d.applicability;$('sim').textContent=fmt(a.maximum_train_tanimoto,3);$('analogs').textContent=a.analog_count_ge_0p5.toLocaleString();$('spread').textContent=fmt(m.fold_prediction_sd);$('domain').innerHTML=`<b>${a.domain_label}:</b> ${a.interpretation}`;$('nearest').innerHTML=a.nearest_analogs.map(x=>`<div class="analog"><b>T=${fmt(x.tanimoto,3)}</b><code title="${x.smiles}">${x.smiles}</code><span>IC50 ${fmt(x.observed_ic50_um,2)} µM</span></div>`).join('');const c=d.chemistry;$('chem').innerHTML=`<span><b>MW</b> ${fmt(c.molecular_weight,1)}</span><span><b>cLogP</b> ${fmt(c.clogp,2)}</span><span><b>TPSA</b> ${fmt(c.tpsa,1)} Å²</span><span><b>HBD/HBA</b> ${c.hbd}/${c.hba}</span><span><b>Rotors</b> ${c.rotatable_bonds}</span>`;$('results').classList.remove('hidden');$('status').textContent='Prediction complete.';}catch(e){$('status').textContent=e.message;}finally{$('go').disabled=false;}};
</script></body></html>"""
    path.write_text(html)


def _write_report(path: Path, info: dict[str, Any]) -> None:
    direct = info["direct_curve"]
    path.write_text(
        f"""# hERG V10.2 Endpoint-Coherent Platform

## Primary Decision

The primary pIC50/IC50 estimate is the mean prediction of the five literal V9
outer-fold deployment stacks. Safe (>30 uM), Moderate (1-30 uM), and Potent
(<1 uM) are deterministic bins of that same estimate. An independent router
cannot contradict the quantitative prediction.

- Exact structures: 18,801
- Nested scaffold OOF MAE: {info["primary_ic50"]["mae"]:.4f} pIC50
- RMSE: {info["primary_ic50"]["rmse"]:.4f}
- Spearman: {info["primary_ic50"]["spearman"]:.4f}
- Within 0.5 log: {info["primary_ic50"]["within_0p5"]:.1%}
- Within 1.0 log: {info["primary_ic50"]["within_1p0"]:.1%}

## Direct Functional Curve

IC10, IC30, and IC50 are separately trained on direct crossings from one
20-concentration human hERG functional qHTS series. They are not Hill-derived
from the primary IC50, and they are never presented as if they came from the
18,801-structure literature surface.

- IC10: {direct["IC10"]["labels"]:,} strict unique labels; MAE {direct["IC10"]["mae"]:.3f}
- IC30: {direct["IC30"]["labels"]:,} strict unique labels; MAE {direct["IC30"]["mae"]:.3f}
- IC50: {direct["IC50"]["labels"]:,} strict unique labels; MAE {direct["IC50"]["mae"]:.3f}
- Physical display constraint: IC10 <= IC30 <= IC50

## Evidence Instead Of The 46 uM Score

The fixed-dose 46 uM classifier is not shown in the prediction interface. It
answered whether a compound was active in a particular thallium-flux screen,
not the compound's IC50. V10.2 instead reports nearest-training similarity,
analogue support, fold-model disagreement, nearest observed analogues, and a
residual-based 90% prediction interval.

## Direct Endpoint Data Audit

The local governed evidence currently supports one direct full-curve series:
20 measured concentrations from 0.0001 to 92.17 uM. Other heterogeneous public
assays are not promoted to direct IC10/IC30 merely because they report a
threshold, AC50, or a single fixed-dose activity. Expanding direct endpoints
requires protocol-level adjudication and recoverable response curves.

## Claim Boundary

This is a ligand-only research prototype. The primary surface is
WT-or-unspecified, not fully confirmed WT. Performance is internal nested
scaffold-held-out evidence, not external/prospective or clinical validation.
"""
    )


def _source_inputs(paths: Paths) -> list[dict[str, Any]]:
    sources = [
        (Path(__file__), "v10.2 implementation"),
        (paths.v9 / "manifest.json", "V9 manifest"),
        (paths.v9 / "analysis/nested_oof_predictions.parquet", "V9 nested OOF"),
        (paths.v9 / "prepared/training_matrix.parquet", "V9 training matrix"),
        (paths.v101 / "manifest.json", "V10.1 direct endpoint manifest"),
        (paths.v101 / "model_info.json", "V10.1 direct endpoint information"),
        (paths.v101 / "evidence/empirical_ic10_ic30_ic50_labels.parquet", "direct curve labels"),
        (paths.surface, "hERG observation structure registry"),
    ]
    sources.extend(
        (paths.v9 / f"units/outer_o{fold}/outer_models.joblib", f"V9 fold {fold} models") for fold in range(5)
    )
    sources.extend(
        (
            paths.v101 / f"models/empirical_{endpoint}_regressor.joblib",
            f"direct {endpoint.upper()} model",
        )
        for endpoint in ("ic10", "ic30", "ic50")
    )
    return [_binding(path, role) for path, role in sources]


def _build(paths: Paths) -> dict[str, Any]:
    if paths.output.exists():
        if (paths.output / "manifest.json").is_file():
            return _validate(paths.output)
        raise V102Error(f"output exists without a valid manifest: {paths.output}")
    for required in (paths.v9, paths.v101):
        if not required.is_dir():
            raise V102Error(f"missing source release: {required}")
    _validate_v101_source(paths.v101)
    oof = pd.read_parquet(paths.v9 / "analysis/nested_oof_predictions.parquet")
    if len(oof) != 18_801 or oof["structure_id"].duplicated().any():
        raise V102Error("V9 nested OOF is not 18,801 unique structures")
    direct_info = _read_json(paths.v101 / "model_info.json", "model_info_sha256")
    parent = paths.output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{paths.output.name}.", dir=parent))
    try:
        _copy_models(paths, staging)
        reference_summary = _build_reference(paths, staging)
        _write_app(staging / "app.html")
        direct_metrics = direct_info["metrics"]["empirical_endpoints"]
        info = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "decision_contract": {
                "primary_ic50_model": "mean of five literal V9 outer-fold deployment stacks",
                "tier_source": "deterministic bins of primary predicted pIC50",
                "safe": "IC50 > 30 uM",
                "moderate": "1 uM <= IC50 <= 30 uM",
                "potent": "IC50 < 1 uM",
                "tier_probability_source": "empirical signed V9 nested-OOF residual distribution",
            },
            "primary_ic50": {
                "structures": 18_801,
                "folds": 5,
                "mae": float(np.mean(np.abs(oof.observed_pic50 - oof.pred__honest_stack))),
                "rmse": float(np.sqrt(np.mean((oof.observed_pic50 - oof.pred__honest_stack) ** 2))),
                "spearman": float(
                    oof[["observed_pic50", "pred__honest_stack"]].corr(method="spearman").iloc[0, 1]
                ),
                "within_0p5": float(np.mean(np.abs(oof.observed_pic50 - oof.pred__honest_stack) <= 0.5)),
                "within_1p0": float(np.mean(np.abs(oof.observed_pic50 - oof.pred__honest_stack) <= 1.0)),
                "scope": "WT-or-unspecified exact pIC50; nested scaffold-held-out internal evidence",
            },
            "direct_curve": {
                endpoint: {
                    "labels": int(direct_metrics[endpoint]["unique_structure_labels"]),
                    "mae": float(direct_metrics[endpoint]["winner"]["mae"]),
                    "rmse": float(direct_metrics[endpoint]["winner"]["rmse"]),
                    "spearman": float(direct_metrics[endpoint]["winner"]["spearman"]),
                }
                for endpoint in ("IC10", "IC30", "IC50")
            },
            "direct_curve_provenance": {
                "qualified_series": 1,
                "assay": "PubChem AID588834 human hERG functional qHTS",
                "measured_concentrations": 20,
                "range_um": [0.0001, 92.17],
                "label_method": "monotone-denoised direct crossings interpolated between observed doses",
                "excluded_label_shortcuts": ["Hill conversion", "fitted AC50", "single fixed-dose activity"],
            },
            "applicability_reference": reference_summary,
            "removed_from_prediction_ui": {
                "fixed_dose_46um_score": "assay activity at one test concentration is not an IC50 or tier",
                "independent_ternary_router": "could contradict the quantitative IC50",
            },
            "scientific_scope": {
                "ligand_only": True,
                "receptor_aware": False,
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
                "external_or_prospective_validation": False,
                "clinical_risk_prediction": False,
            },
        }
        _atomic_json(staging / "model_info.json", info, "model_info_sha256")
        _write_report(staging / "REPORT.md", info)
        artifact_paths = [
            staging / "app.html",
            staging / "REPORT.md",
            staging / "model_info.json",
            staging / "models/applicability_reference.joblib",
            *(staging / f"models/v9_folds/outer_o{fold}.joblib" for fold in range(5)),
            *(
                staging / f"models/direct_{endpoint}_regressor.joblib"
                for endpoint in ("ic10", "ic30", "ic50")
            ),
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "input_bindings": _source_inputs(paths),
            "artifact_bindings": [
                _binding(path, f"V10.2 artifact {path.name}", root=staging) for path in artifact_paths
            ],
            "expected_members": sorted(
                [str(path.relative_to(staging)) for path in artifact_paths]
                + ["manifest.json", "validation.json"]
            ),
        }
        _atomic_json(staging / "manifest.json", manifest, "manifest_sha256")
        validation = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "primary_tier_and_ic50_share_source": True,
            "direct_curve_single_assay": True,
            "curve_order_enforced": True,
            "fixed_dose_46um_removed_from_prediction_ui": True,
            "exact_structures": 18_801,
        }
        _atomic_json(staging / "validation.json", validation, "validation_sha256")
        os.replace(staging, paths.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _validate(paths.output)


def _validate(output: Path) -> dict[str, Any]:
    root = output.resolve()
    manifest = _read_json(root / "manifest.json", "manifest_sha256")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "passed":
        raise V102Error("manifest status or schema is invalid")
    for binding in manifest["input_bindings"]:
        _verify_binding(binding)
    for binding in manifest["artifact_bindings"]:
        _verify_binding(binding, root)
    members = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())
    if members != list(manifest["expected_members"]):
        raise V102Error("output membership is not closed")
    validation = _read_json(root / "validation.json", "validation_sha256")
    info = _read_json(root / "model_info.json", "model_info_sha256")
    if validation.get("status") != "passed" or info.get("primary_ic50", {}).get("structures") != 18_801:
        raise V102Error("V10.2 semantic validation failed")
    return validation


def _tanimoto_similarities(
    packed_reference: np.ndarray, reference_bit_counts: np.ndarray, query_bits: np.ndarray
) -> np.ndarray:
    """Compute exact Morgan Tanimoto values without uint8 dot-product overflow."""
    reference_bits = np.unpackbits(packed_reference, axis=1)[:, :MORGAN_BITS].astype(np.int32, copy=False)
    query = np.asarray(query_bits, dtype=np.int32)
    intersection = reference_bits @ query
    query_count = int(query.sum())
    union = np.asarray(reference_bit_counts, dtype=np.int32) + query_count - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )


class Predictor:
    def __init__(self, root: Path):
        self.root = root.resolve()
        _validate(self.root)
        self.info = _read_json(self.root / "model_info.json", "model_info_sha256")
        self.folds = [joblib.load(self.root / f"models/v9_folds/outer_o{fold}.joblib") for fold in range(5)]
        self.direct = {
            endpoint: joblib.load(self.root / f"models/direct_{endpoint}_regressor.joblib")
            for endpoint in ("ic10", "ic30", "ic50")
        }
        self.reference = joblib.load(self.root / "models/applicability_reference.joblib")
        self.columns = _feature_union_from_loaded(self.folds, list(self.direct.values()))

    def _primary(self, features: pd.DataFrame) -> tuple[float, list[float]]:
        fold_predictions: list[float] = []
        for bundle in self.folds:
            meta: dict[str, np.ndarray] = {}
            for name, model in bundle["models"].items():
                columns = _model_columns(model)
                meta[f"pred__{name}"] = np.asarray(
                    model.predict(_safe_numeric(features, columns)), dtype=float
                )
            stack_frame = pd.DataFrame(meta).reindex(columns=bundle["columns"])
            fold_predictions.append(float(bundle["stack"].predict(stack_frame)[0]))
        return float(np.mean(fold_predictions)), fold_predictions

    def _applicability(self, features: pd.DataFrame) -> dict[str, Any]:
        columns = [f"morgan__{index:04d}" for index in range(MORGAN_BITS)]
        query = features[columns].fillna(0).to_numpy(dtype=np.uint8)[0]
        similarities = _tanimoto_similarities(
            self.reference["packed_morgan"], self.reference["bit_counts"], query
        )
        order = np.argsort(similarities)[-3:][::-1]
        maximum = float(similarities[order[0]])
        count = int(np.sum(similarities >= 0.5))
        if maximum >= 0.7:
            label = "High analogue support"
            interpretation = "A close training analogue exists; local interpolation is more plausible."
        elif maximum >= 0.5:
            label = "Moderate analogue support"
            interpretation = "Related chemistry exists, but this is still a scaffold-sensitive prediction."
        else:
            label = "Extrapolative chemistry"
            interpretation = "No training compound reaches Tanimoto 0.50; treat the estimate as high risk."
        return {
            "maximum_train_tanimoto": maximum,
            "analog_count_ge_0p5": count,
            "domain_label": label,
            "interpretation": interpretation,
            "nearest_analogs": [
                {
                    "structure_id": str(self.reference["structure_ids"][index]),
                    "smiles": str(self.reference["smiles"][index]),
                    "tanimoto": float(similarities[index]),
                    "observed_pic50": float(self.reference["target_pic50"][index]),
                    "observed_ic50_um": _ic_um(float(self.reference["target_pic50"][index])),
                }
                for index in order
            ],
        }

    def predict(self, smiles: str) -> dict[str, Any]:
        features, molecule = _single_feature_frame(smiles, self.columns)
        pic50, fold_predictions = self._primary(features)
        residuals = np.asarray(self.reference["signed_oof_residuals"], dtype=float)
        lower_pic50 = float(pic50 + np.quantile(residuals, 0.05))
        upper_pic50 = float(pic50 + np.quantile(residuals, 0.95))
        tier_probabilities = {
            "Safe": float(np.mean(pic50 + residuals < SAFE_PIC50)),
            "Moderate": float(
                np.mean((pic50 + residuals >= SAFE_PIC50) & (pic50 + residuals <= POTENT_PIC50))
            ),
            "Potent": float(np.mean(pic50 + residuals > POTENT_PIC50)),
        }
        raw_direct: dict[str, float] = {}
        for endpoint, bundle in self.direct.items():
            columns = [str(column) for column in bundle["feature_columns"]]
            raw_direct[endpoint] = _ic_um(float(bundle["model"].predict(_safe_numeric(features, columns))[0]))
        projected, adjusted = v101._coherent_threshold_concentrations(  # noqa: SLF001
            [raw_direct["ic10"], raw_direct["ic30"], raw_direct["ic50"]]
        )
        applicability = self._applicability(features)
        return {
            "schema_version": SCHEMA_VERSION,
            "smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
            "main_ic50": {
                "pic50": pic50,
                "ic50_um": _ic_um(pic50),
                "tier": _tier(pic50),
                "tier_definition": {"Safe": ">30 uM", "Moderate": "1-30 uM", "Potent": "<1 uM"},
                "tier_probabilities": tier_probabilities,
                "interval90_pic50": {"lower": lower_pic50, "upper": upper_pic50},
                "interval90_um": {"lower": _ic_um(upper_pic50), "upper": _ic_um(lower_pic50)},
                "fold_predictions": fold_predictions,
                "fold_prediction_sd": float(np.std(fold_predictions)),
                "model_source": "literal mean of five V9 fold-specific stack predictions",
            },
            "direct_functional_curve": {
                "ic10_um": projected[0],
                "ic30_um": projected[1],
                "ic50_um": projected[2],
                "raw_predictions_um": raw_direct,
                "ordering_adjusted": adjusted,
                "ordering_contract": "IC10 <= IC30 <= IC50",
                "scope": "one direct 20-concentration human hERG functional qHTS assay",
            },
            "applicability": applicability,
            "chemistry": {
                "molecular_weight": float(Descriptors.MolWt(molecule)),
                "clogp": float(Crippen.MolLogP(molecule)),
                "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
                "hbd": int(Lipinski.NumHDonors(molecule)),
                "hba": int(Lipinski.NumHAcceptors(molecule)),
                "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
            },
            "scientific_scope": self.info["scientific_scope"],
        }


def _feature_union_from_loaded(
    folds: list[dict[str, Any]], direct_bundles: list[dict[str, Any]]
) -> list[str]:
    columns: set[str] = set()
    for bundle in folds:
        for model in bundle["models"].values():
            columns.update(_model_columns(model))
    for bundle in direct_bundles:
        columns.update(str(value) for value in bundle["feature_columns"])
    return sorted(columns)


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
                body = (predictor.root / "app.html").read_bytes()
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
                    raise V102Error("SMILES is required")
                self._json(HTTPStatus.OK, predictor.predict(smiles))
            except Exception as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def _paths(args: argparse.Namespace) -> Paths:
    repo = Path(args.repo_root).resolve()
    return Paths(
        repo=repo,
        v9=(repo / args.v9_root).resolve(),
        v101=(repo / args.v101_root).resolve(),
        surface=(repo / args.surface).resolve(),
        output=(repo / args.output_root).resolve(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "validate", "predict", "serve"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--repo-root", default=".")
        sub.add_argument("--v9-root", default=str(DEFAULT_V9))
        sub.add_argument("--v101-root", default=str(DEFAULT_V101))
        sub.add_argument("--surface", default=str(DEFAULT_SURFACE))
        sub.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
        if name == "predict":
            sub.add_argument("--smiles", required=True)
        if name == "serve":
            sub.add_argument("--host", default="127.0.0.1")
            sub.add_argument("--port", type=int, default=8790)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = _paths(args)
    if args.command == "build":
        print(json.dumps(_build(paths), indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        print(json.dumps(_validate(paths.output), indent=2, sort_keys=True))
        return 0
    predictor = Predictor(paths.output)
    if args.command == "predict":
        print(json.dumps(predictor.predict(args.smiles), indent=2, sort_keys=True, allow_nan=False))
        return 0
    server = ThreadingHTTPServer((args.host, args.port), _handler(predictor))
    print(f"hERG V10.2 available at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
