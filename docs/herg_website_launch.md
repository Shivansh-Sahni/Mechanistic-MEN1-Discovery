# hERG prediction website launch

## Readiness contract

The website has three researcher workflows:

- **Predict** preserves the frozen single-structure hERG prediction stack and now separates the
  independent classifier, continuous IC50 regression, pooled OOF-residual tier frequencies,
  calculated properties, applicability, and heuristic interpretation.
- **Optimize** retains the investigator-supplied parent/candidate comparison and adds a separate,
  visibly experimental bounded-generation mode. Generation applies a versioned one-step RDKit
  transformation registry, validates and deduplicates up to 100 candidates, runs the frozen
  ligand-only stack, and ranks with disclosed interval-width, applicability, parent-similarity,
  and property-change penalties. The score is prioritization—not confidence, synthesis feasibility,
  or an activity claim—and generated candidates are never used for retraining. The manual mode's
  opt-in MW-scaled scenario remains separate and never changes either primary prediction.
- **Properties** calculates a method-labeled RDKit descriptor, functional-group, and structural
  ionization profile without running hERG prediction or docking. It discloses salt removal and
  charge neutralization because results describe the standardized model parent.

Predict has two explicit prediction modes.

The default fast ligand-only mode uses the exact frozen models packaged in the V13.9
inference bundle:

- V10.1 LGBM RDKit2D+Morgan classification router;
- V9/V10 literature IC50 regression and V10.3 applicability diagnostics;
- V10.1 endpoint-specific IC10, IC30, and IC50 regressions;
- deterministic V12.3 ordering so complete direct curves satisfy IC10 ≤ IC30 ≤ IC50.

This path avoids on-demand 3D conformer generation. On the launch machine it normally
finishes in roughly 0.1–0.2 seconds, so the website conservatively labels it “usually under
1 second.”

The optional ligand + receptor mode supports an investigator-selected panel of one to six
prepared receptor structures. Six states are available: 8ZYN (apo), 8ZYO
(astemizole-bound), 8ZYP (E-4031-bound), 8ZYQ (pimozide-bound), 9CHP (high-K+ C4), and
9CHQ (low-K+ C4). Every selected state runs a real AutoDock Vina calculation and retains its
own score and residue-contact diagnostics; scores are not averaged into a fabricated consensus.

Only 8ZYO has a sealed feature contract and frozen receptor-aware prediction model. When 8ZYO
is selected, the combined-feature workflow:

- it first computes the same ligand RDKit2D+Morgan features and predictions;
- it prepares the submitted ligand and docks it into the 8ZYO hERG structure with AutoDock
  Vina 1.2.7 (exhaustiveness 8, up to 9 poses);
- it extracts Vina affinity, all-pose geometry, and T623/S624/S649/Y652/F656 receptor-contact
  features;
- it runs the V13.9 frozen receptor ensemble and the selected V14.1 ligand+receptor
  classification and IC50-regression surfaces.

The other five states are explicitly docking-only diagnostics. Selecting them does not create a
state-specific IC50 prediction and does not alter the ligand-only default. A panel that omits 8ZYO
therefore reports receptor-state docking diagnostics but correctly reports that no receptor-aware
ML prediction was generated.

Docking execution is separate from receptor-model applicability. A compound above 750 Da may be
attempted by Vina when RDKit preparation succeeds and its elements are supported, even though the
sealed receptor-aware ML surface remains out of domain and no receptor IC50/classifier output is
generated. Heavy-atom, flexibility, and MW warnings remain visible; failed preparation or docking
is returned per receptor without suppressing the ligand-only prediction.

Fresh receptor calculations are strongly structure-dependent. Exact public comparators took
about 6.5 and 17.0 seconds on the launch machine. Ten larger, flexible internal-series compounds
took 20.8–42.4 seconds (31.4-second mean). The interface therefore estimates 5–45 seconds for a
fresh molecule and notes that large flexible structures commonly take 20–45 seconds per state.
Successful receptor results are cached in process memory by canonical SMILES plus the ordered
receptor selection; an exact repeat of the same panel usually takes under 1 second. Fresh runtime
scales approximately with the number of selected states. Cached structures are not written to a
request log and the cache disappears when the service stops.

Predict also exposes exactly two ligand feature choices under **Advanced calculation options**:

- **Atom-wise** is the validated default RDKit2D+Morgan surface.
- **Functional-group augmented** runs the actual V15 governed SMARTS/physicochemical residual
  surface. It uses functional-group counts and presence, cLogP, TPSA, MW, HBD/HBA, charge and
  ionizable-center proxies, ring/linker descriptors, and controlled interactions. It remains a
  non-default retrospective comparison because its nested campaign-held-out MAE did not improve
  stably over the deployed baseline. The API retains the validated atom-wise result alongside it,
  and no calibrated query-specific interval is claimed for the V15 output.

The optional heavy-molecule view expands size, flexibility, applicability, and docking-execution
diagnostics. It does not rescale the single-structure IC50 estimate or claim improved high-MW
accuracy. Compounds above 750 Da can still attempt Vina when preparation is technically feasible.

## MW edit-magnitude stress test

Optimize exposes an off-by-default **MW-scaled edit-magnitude stress test**. It is a separate
what-if calculation, not a replacement prediction. The primary parent/candidate IC50 values,
classifier probabilities, tiers, and intervals remain unchanged.

The scenario scales only `candidate pIC50 - parent pIC50`, preserving the model's direction:

```text
pair mean MW = (parent MW + candidate MW) / 2
multiplier = max(1, 1.546856 × pair mean MW / 432.485992 Da)
stress candidate pIC50 = parent pIC50 + raw delta pIC50 × multiplier
```

The 432.486 Da reference is the empirical median RDKit molecular weight of the 18,801-structure
OOF corpus. The 1.546856 baseline is the retrospective L1-optimal nonnegative scalar among 5,484
measured activity-cliff MMPs. It was fitted on the same internal evidence and is therefore
non-independent. The interface shows the factor, unscaled and scaled delta, and what-if IC50, and
states that the scenario cannot repair a wrong direction. Evidence above 700 Da is sparse. Full
methods and MW-band results are in `docs/herg_mw_pair_delta_attenuation.md`.

## Evaluation examples

The static dropdown includes two exact measured public comparators, both explicitly labeled as
training-overlap demonstrations rather than independent validation. Private internal-series
structures, measurements, and the disclosed hard failure were removed from all static website
assets and remain only in access-controlled local evaluation artifacts. The localhost launcher
enables a private API-backed dropdown of ten internal examples; their structures are loaded from
the internal workbook at runtime and are never embedded in HTML or JavaScript. The public launcher
does not enable this endpoint. The displayed public table is a
selected release check, not an unbiased benchmark. Receptor-aware values are representative audit
runs: Vina uses a stable task seed, but fresh 3D preparation may still yield small run-to-run changes.

It also exposes one visibly separate, non-default research preview:

- V14 ligand-only cross-campaign classification recalibration;
- V14.1 ligand-only ExtraTrees residual recalibration for literature IC50.

Both ligand preview components and the task-specific receptor surfaces were evaluated with
nested whole-campaign holdouts across 1,224 rows
from six structure-disjoint campaigns. They remain retrospective and are not prospectively
confirmed. Linear and nonlinear receptor-aware surfaces were challenged in V14/V14.1 and did
not beat the strongest task-specific ligand comparators. V14.2 likewise did not improve the
retained IC10, IC30, or IC50 models. The receptor outputs are therefore shown as an optional,
visibly research-only comparison and never alter the website default.

## Localhost launch (default)

Run:

```bash
pipeline/scripts/launch_herg_prediction_website_local.sh
```

This validates the complete artifact stack, starts the service at `127.0.0.1:8795`, writes its PID
under `.codex_tmp/herg_website_launch`, and creates no tunnel. Stop it with:

```bash
pipeline/scripts/stop_herg_prediction_website.sh
```

## Temporary external review (explicit opt-in only)

Do not use a tunnel with unpublished structures. After explicit authorization for an external
review of public/non-confidential examples only, run:

```bash
HERG_ENABLE_PUBLIC_TUNNEL=1 pipeline/scripts/launch_herg_prediction_website.sh
```

The launcher first verifies every required V10–V14.1 runtime bundle, starts the local model
service, creates a temporary HTTPS tunnel through Pinggy, verifies password protection from
the public endpoint, and prints two temporary Pinggy aliases plus the credentials. Use the
alternate alias if a network filter blocks the first. It generates a fresh password when
`HERG_WEBSITE_PASSWORD` is not set. A free Pinggy tunnel expires after about 60 minutes and
may show a one-time tunnel notice in each browser before the website login prompt.
The URL remains available only while this computer, the model service, the tunnel, and the
launch command are running.

Stop the review site with:

```bash
pipeline/scripts/stop_herg_prediction_website.sh
```

Stopping removes the temporary URL credentials and recorded process identifiers. Public tunneling
is disabled unless `HERG_ENABLE_PUBLIC_TUNNEL=1` is supplied for that exact launch.

## Pre-launch checks

```bash
PYTHONPATH=pipeline/scripts .venv/bin/python \
  pipeline/scripts/run_herg_prediction_website.py validate

.venv/bin/pytest -q pipeline/tests/test_run_herg_prediction_website.py
.venv/bin/ruff check pipeline/scripts/run_herg_prediction_website.py \
  pipeline/tests/test_run_herg_prediction_website.py
node --check pipeline/web/herg/app.js
```

The site is a research prototype. It must not be represented as prospectively validated,
clinical, regulatory, or patient-care software.

## Paired structural-edit evaluation

Use [the paired-edit evaluation guide](herg_paired_edit_evaluation.md) to evaluate measured
parent/candidate series. The standalone evaluator reports directional accuracy, delta-log-IC50
error and bias, cliff-capture ratio, ties, and optional investigator-defined MW/similarity strata.
It accepts measured-only input without inventing prediction metrics and rejects censored values
unless a separate interval-aware evaluation is defined.

The finalized label-blind frozen benchmark and its limitations are documented in
[the V15 locked paired-benchmark guide](herg_v15_locked_paired_benchmark.md). Its primary locked
test contains 517 target-consistent pairs across 96 leakage groups; only four pairs are at or above
600 Da, so it does not support a high-MW performance claim.
