# Paired hERG edit evaluation

`pipeline/scripts/analyze_herg_paired_edits.py` evaluates whether a frozen hERG
model captures the direction and magnitude of measured changes between a parent
compound and a closely related candidate. It does not fit or import a model.

## Input format

The input is a CSV with one row per experimental pair and these required columns:

| Column | Meaning |
|---|---|
| `pair_id` | Unique, nonblank pair identifier |
| `parent_smiles` | Parent structure |
| `candidate_smiles` | Edited structure |
| `parent_measured_ic50_um` | Experimental parent hERG IC50 in µM |
| `candidate_measured_ic50_um` | Experimental candidate hERG IC50 in µM |

Both measured values must be finite and positive. Pair IDs must be unique, and a
structure pair cannot be repeated in either orientation. Censored values such as
`>10` are not accepted as exact measurements; such data require a separately
defined interval-aware evaluation.

Frozen predictions may be included as:

- `parent_predicted_ic50_um`
- `candidate_predicted_ic50_um`

Both prediction columns must be present together, finite, positive, and in µM.
If they are absent, the command produces measured-direction counts without
inventing prediction metrics.

Optional `parent_mw`, `candidate_mw`, and `similarity` columns support externally
defined stratification. Alternatively, `--compute-rdkit-features` calculates
RDKit `MolWt` and radius-2, 2048-bit Morgan/Tanimoto similarity. Any MW or
similarity band cutoffs are supplied explicitly by the investigator; the tool
does not claim universal applicability thresholds.

## Command

```bash
python pipeline/scripts/analyze_herg_paired_edits.py pairs.csv \
  --output-dir paired_edit_evaluation \
  --tie-threshold-log10 0.1 \
  --compute-rdkit-features \
  --mw-cutoffs 650 750 \
  --similarity-cutoffs 0.8 0.9
```

The zero default tie threshold excludes only exact experimental ties. A nonzero
threshold must be selected and reported before evaluating results. Outputs are
`paired_edit_results.csv`, `paired_edit_summary.json`, and, when requested,
`paired_edit_stratified_metrics.csv`.

## Importable prediction interface

`evaluate_paired_edits` also accepts aligned parent/candidate IC50 arrays or a
`predictor` callback. The callback receives all parent SMILES followed by all
candidate SMILES and returns predictions in the same order. This keeps slow or
model-specific loading entirely outside the evaluator.

```python
evaluation = evaluate_paired_edits(
    pairs,
    predictor=frozen_model_predict_ic50_um,
    tie_threshold_log10=predeclared_threshold,
    stratify_by=("mw_band", "similarity_band"),
)
```

## Metric definitions

The measured and predicted deltas are
`log10(candidate IC50 µM) - log10(parent IC50 µM)`. Positive means improved
(higher IC50 and lower hERG liability); negative means worsened.

- **Directional accuracy:** correct direction among experimentally
  non-negligible pairs. A predicted negligible change on a measured change is a
  miss. Experimental negligible pairs are excluded.
- **Delta MAE:** mean absolute error between predicted and measured log10 IC50
  deltas.
- **Cliff capture ratio:** sum of absolute predicted deltas divided by the sum
  of absolute measured deltas among experimentally non-negligible pairs. It is
  null when the measured denominator is zero and can exceed 1 when effects are
  over-amplified. Directional accuracy must be interpreted alongside it.

These metrics describe paired prediction behavior. They do not establish a
causal effect of a functional-group edit, and synthetic values in the unit tests
exist only to verify arithmetic and validation behavior.
