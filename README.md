# Menin inhibitor discovery

This repository is a Menin/MEN1 inhibitor research workflow for public-data curation, potency modeling, Menin-focused hERG liability assessment, rat PK analysis, mechanistic feature development, and candidate-design support.

hERG is included only as a safety endpoint for Menin inhibitors. Public hERG measurements may be used to pretrain or contextualize the Menin-specific safety model, but this repository does not claim to be a general-purpose hERG platform. PK work is likewise scoped to the disposition and exposure of the large-molecule Menin series.

The software is an analytical research system, not a validated clinical, toxicology, or compound-selection product. Predictions are hypotheses that require applicability-domain checks, uncertainty reporting, expert review, and experimental validation.

## Project scope

The public workflow supports:

- Menin activity data from ChEMBL, BindingDB, and PubChem BioAssay;
- endpoint- and assay-specific Menin potency models;
- hERG IC50 and blocker-risk prediction for Menin inhibitor candidates;
- rat IV/PO PK endpoints and process-centered PK research;
- mechanistic molecular features for large, flexible, beyond-rule-of-five compounds;
- scaffold-aware validation, uncertainty, applicability-domain analysis, and failure reporting;
- Menin-Edit workflows for proposing and evaluating candidate modifications.

## Repository layout

```text
pipeline/                  Menin data, modeling, hERG-safety, PK, and research code
packages/menin-edit/       Candidate-editing and multi-objective design package
research/data/             Public Menin and supporting safety/PK evidence
research/models/           Public models and released model manifests
research/reports/          Menin-centered analyses, figures, and validation reports
research/literature/       Evidence reviews and mechanistic synthesis
docs/                      Methods, architecture, limitations, and reproducibility
```

Large simulations, caches, raw third-party literature files, and target-agnostic experiments are excluded from Git.

## Menin pipeline

The pipeline:

1. collects target-anchored public Menin and supporting hERG/PK evidence;
2. preserves source fields, censoring, units, assay context, and provenance;
3. standardizes structures with RDKit and assigns stable measurement/structure identifiers;
4. quarantines unresolved units, structures, target mismatches, and conflicting records;
5. trains endpoint-specific Menin potency and Menin-safety models using grouped, scaffold, chemical-cluster, or temporal splits;
6. reports calibration, uncertainty, applicability domain, activity cliffs, matched pairs, and residual failure clusters;
7. maintains a separate mechanistic discovery track for physics-derived PK and hERG features.

The fixed target anchors are Menin `CHEMBL1615381` / UniProt `O00255` and hERG `CHEMBL240` / UniProt `Q12809`. hERG records are supporting safety evidence; downstream interpretation remains Menin-candidate-specific.

## Current hERG output for a Menin candidate

Given a candidate structure, the workflow can return:

- predicted continuous hERG potency (`pIC50` and IC50 in µM);
- blocker probability at the declared decision threshold;
- calibrated uncertainty or prediction interval;
- nearest-neighbor/applicability-domain status;
- similarity to the Menin-relevant training chemistry;
- mechanistic feature contributions and required-data flags.

Predictions outside the supported chemical domain are flagged rather than presented as reliable point estimates.

## Current PK scope

Rat PK is the immediate target. The workflow treats IV clearance or dose-normalized IV exposure, volume of distribution, PO dose-normalized exposure, Cmax, and Tmax as distinct process-linked endpoints. Algebraically derived quantities such as clearance and bioavailability are not treated as independent labels when their parent dose/AUC measurements are used.

Heavy molecular dynamics, membrane permeability free-energy calculations, receptor-state simulations, and converged transition kinetics remain HPC-stage work. Local fast-physics calculations are retained only when they meet explicit convergence and biological-interpretability criteria.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip "setuptools>=83"
python -m pip install -e ".[dev]" -e "./packages/menin-edit[lab,dev]"
```

## Common commands

```bash
# Rebuild the checked-in public Menin snapshot without network access
menin-pipeline --stage all --skip-network --fast

# Validate public artifacts
menin-pipeline --stage verify --skip-network --fast

# Run the mechanistic PK/hERG workflow for Menin candidates locally
menin-research --config pipeline/config/pk_herg_research.yaml --stage all-local

# Run tests
pytest -q pipeline/tests
pytest -q -c packages/menin-edit/pyproject.toml packages/menin-edit/tests
```

## hERG prediction website

Validate the launch-ready local website with:

```bash
PYTHONPATH=pipeline/scripts .venv/bin/python \
  pipeline/scripts/run_herg_prediction_website.py validate
```

For local preview, private HTTPS review, and shutdown instructions, see the
[website launch guide](docs/herg_website_launch.md).

The research interface now provides **Predict**, **Optimize**, and **Properties** workspaces.
Optimize supports a manual parent/candidate comparison plus explicitly experimental generation of
bounded, chemically valid one-step edits. Both use real frozen-model outputs and calculated
property deltas; generated candidates are hypotheses, not synthesis or activity claims. Properties
exposes RDKit descriptors, functional-group and basicity proxies, training-distribution context,
and structure-standardization disclosures without running a prediction. Measured close-analogue
series can be evaluated with the [paired-edit evaluator](docs/herg_paired_edit_evaluation.md).

The default decision remains the fast ligand-only path. A separate V14/V14.1 retrospective
preview is available in the result page, and an optional 8ZYO AutoDock Vina mode runs true
ligand+receptor feature models. Receptor-aware outputs remain visibly non-default after linear,
nonlinear, six-campaign, and direct-endpoint challenge tests failed to show a conclusive gain.

## Data availability

Source datasets that are not publicly distributable are not included. The checked-in examples and public-data workflow remain independently usable.

See [architecture](docs/architecture.md), [methodology](docs/methodology.md), [limitations](docs/limitations.md), and [reproducibility](docs/reproducibility.md).

## Scientific claim boundary

The current models support retrospective research and cautious same-domain prediction. They do not establish broad novel-scaffold generalization, clinical cardiac safety, human PK translation, or causal proof of a proposed physical mechanism. Those claims require prospective Menin-series data, protocol-matched assays, and converged HPC simulations.
