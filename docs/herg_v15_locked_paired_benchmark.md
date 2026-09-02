# hERG V15 locked matched-pair benchmark

`pipeline/scripts/build_herg_v15_paired_benchmark.py` creates an immutable,
train-only retrospective benchmark for testing whether an already-frozen model
captures the direction and magnitude of hERG changes between matched molecular
pairs. It never emits SMILES, structures, MMP cores, or transformations.

## Scientific boundary

- Outcomes come only from the exact Q1 TRAIN-partition point targets in
  `v1_5_mmp_analysis/training_mmp_effects.parquet`.
- Censored values are not converted to point estimates. The authoritative MMP
  source contains exact values only, so performance on censored measurements is
  not estimated.
- Repository validation/test labels, prospective labels, internal showcase
  measurements, and final-test measurements are not opened.
- This is internal retrospective OOF evidence, not external, prospective,
  blinded, clinical, or independent validation.

## Lock and leakage controls

The `freeze` operation runs before any MMP outcome column is decoded. It:

1. Canonicalizes every unordered structure pair and collapses exact/reversed
   duplicates.
2. Unions all structures connected by any TRAIN MMP edge and all structures
   sharing a scaffold group.
3. Uses that complete component as a conservative chemical-series proxy.
4. Retains a pair for primary scoring only when both endpoints and the complete
   MMP/scaffold component occur in one OOF fold. This prevents an analogue in
   the same proxy series from appearing in that pair's OOF training folds.
5. Assigns complete leakage groups, never individual pairs, to development or
   locked-test partitions using a deterministic label-blind allocation.
6. Freezes the prespecified V11 nested OOF baseline and the V9/XGB reference
   surfaces before opening outcomes.

Every structure, replicate-aggregated endpoint, scaffold, and series proxy is
therefore confined to one benchmark partition. The authoritative inputs do not
contain a real series or campaign identifier. The MMP component is explicitly a
proxy, and campaign leakage cannot be tested or controlled; the lock records
this limitation instead of inventing metadata.

## Prespecified metrics

The signed quantity is `candidate/B pIC50 - parent/A pIC50`. Positive values
mean stronger hERG inhibition and increased liability. The primary direction
metric excludes measured changes with absolute delta pIC50 no greater than
0.10, then scores the sign of each nonzero prediction. A secondary thresholded
direction metric applies 0.10 to both measured and predicted deltas, making an
under-sensitive predicted tie an error.

Magnitude metrics include delta-pIC50 MAE/RMSE/bias, Pearson correlation,
slope through the origin, and the ratio of total predicted to measured absolute
delta among measured non-ties. Activity cliffs are prespecified as absolute
delta pIC50 at least 1.00. Confidence intervals resample complete leakage groups
rather than non-independent pair rows.

Prespecified strata cover pair-mean MW, measured change magnitude, endpoint
replicate support, minimum endpoint OOF applicability support, extrapolation
status, assay/source context, and outer fold. Every row includes a support flag;
fewer than 30 pairs or 20 leakage groups is `sparse_descriptive_only`.

## Commands

```bash
.venv/bin/python pipeline/scripts/build_herg_v15_paired_benchmark.py freeze
.venv/bin/python pipeline/scripts/build_herg_v15_paired_benchmark.py score-baseline
.venv/bin/python pipeline/scripts/build_herg_v15_paired_benchmark.py validate
```

The default output is
`research/local_runs/herg_v15_finalize/paired/`. Frozen artifacts are under
`locked/`, decoded exact TRAIN labels under `sealed/`, and aggregate/row-level
baseline evidence under `analysis/`. Re-running is validation-only and refuses
partial or changed state; it does not overwrite a lock.

## Challenger rule

Any future challenger must be declared and frozen before reading
`sealed/train_only_pair_labels.parquet` or the scored outputs. Locked-test
results are for one final comparison, not iterative model or hyperparameter
selection. Development results may guide work only if the locked-test outputs
remain uninspected for that challenger cycle.
