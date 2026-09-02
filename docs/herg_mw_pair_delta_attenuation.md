# hERG matched-pair MW attenuation audit

## Outcome

Local measured matched-pair data support a **retrospective attenuation diagnostic** through roughly
700 Da. They do not support an automatic production correction. Evidence above 700 Da is sparse.

The audit uses:

- `research/local_runs/herg_comprehensive_optimization_v11_1/prepared/training_mmp_effects.parquet`
  (43,824 internal train-partition MMPs), and
- `research/local_runs/herg_comprehensive_optimization_v11_1/analysis/nested_oof_predictions.parquet`
  (`v9_predicted_pic50`, the V9 honest-stack OOF surface carried into V11.1, plus RDKit MolWt).

The default audit excludes 2,235 pairs for which the MMP endpoint aggregate does not match the OOF
target value. This leaves 41,589 target-consistent pairs. The MMP registry marks these pairs
exploratory/training-only; the analysis is neither external nor prospective.

## Evidence

For measured shifts of at least 0.1 pIC50, aggregate magnitude capture is
`sum(|predicted ΔpIC50|) / sum(|measured ΔpIC50|)`. This is a descriptive fraction, not an invented
"ideal" ratio and not a calibration probability.

| Pair mean MW (Da) | Pairs | MMP components | Magnitude capture | Component-bootstrap 95% interval | Direction accuracy |
|---|---:|---:|---:|---:|---:|
| <400 | 7,039 | 604 | 0.496 | 0.446–0.557 | 0.636 |
| 400–500 | 13,071 | 784 | 0.450 | 0.413–0.488 | 0.638 |
| 500–600 | 6,059 | 327 | 0.428 | 0.373–0.493 | 0.648 |
| 600–700 | 288 | 49 | 0.327 | 0.272–0.384 | 0.556 |
| ≥700 | 11 | 7 | 0.280 | 0.181–0.550 | 0.455 |

The same downward magnitude pattern is present for the deployment-candidate
`pred__xgb_depth10` OOF surface (0.444, 0.404, 0.381, 0.305, and 0.236 across the same bands) and the
V11 nested surface (0.466, 0.419, 0.410, 0.315, and 0.264). This makes the observation less likely to
be peculiar to one OOF prediction column, but the three surfaces share data and are not independent
replications.

For measured activity cliffs (`|ΔpIC50| >= 1`), the V9 OOF evidence is:

| Pair mean MW (Da) | Pairs | Components | Magnitude capture | Retrospective L1 scalar |
|---|---:|---:|---:|---:|
| <400 | 1,523 | 204 | 0.343 | 1.355 |
| 400–500 | 2,579 | 295 | 0.288 | 1.630 |
| 500–600 | 1,303 | 136 | 0.295 | 1.678 |
| 600–700 | 74 | 21 | 0.236 | 2.057 |
| ≥700 | 5 | 3 | 0.154 | 4.506 (descriptive only) |

The overall activity-cliff L1 scalar is 1.5469 (component-bootstrap 95% interval 1.4303–1.6745).
It is the nonnegative hindsight multiplier minimizing
`sum(|measured ΔpIC50 - scalar * predicted ΔpIC50|)` on these same pairs. It is therefore
non-independent and is not an "ideal" biological ratio. It cannot fix direction errors.

Important limitations:

- MMP rows are correlated; connected MMP components, not individual rows, are bootstrapped.
- Chemistry series, assay composition, and molecular weight are confounded.
- OOF endpoint predictions can come from different fold models. Use `--same-outer-fold-only` as a
  sensitivity check when inter-fold offsets are a concern.
- `v9_predicted_pic50` is an OOF honest-stack surface, not the literal final deployable candidate.
  Use `--prediction-column pred__xgb_depth10` for the closest deployment-candidate audit.
- The ≥700 Da result has only 11 informative pairs and five activity cliffs. It must not be used as a
  validated high-MW correction.

## Standalone audit

```bash
.venv/bin/python pipeline/scripts/analyze_herg_mw_pair_delta_attenuation.py \
  --bootstrap-replicates 1000 \
  --output-json /tmp/herg_mw_pair_audit.json \
  --output-bands-csv /tmp/herg_mw_pair_bands.csv
```

The default filters require the MMP measured endpoint to match the OOF target. `--allow-target-mismatch`
exists only for an explicit sensitivity analysis. The script loads frozen tables, not model objects.

## Opt-in stress scenario

`mw_proportional_delta_stress_test` is importable from
`pipeline/scripts/analyze_herg_mw_pair_delta_attenuation.py`. It is disabled by default and always
returns the primary parent, candidate, and ΔpIC50 unchanged in a separate field.

```python
scenario = mw_proportional_delta_stress_test(
    parent_pic50,
    candidate_pic50,
    parent_mw_da,
    candidate_mw_da,
    enabled=True,
    reference_median_mw_da=432.4859924316406,
    baseline_multiplier=1.5468564492450243,
    provenance={
        "scope": "target-consistent internal V11.1 MMPs; V9 honest-stack OOF",
        "filter": "absolute measured delta pIC50 >= 1",
    },
)
```

The scenario is:

```text
pair_mean_mw = (parent_mw + candidate_mw) / 2
multiplier = max(1, baseline_multiplier * pair_mean_mw / reference_median_mw)
stressed_candidate_pIC50 = parent_pIC50
                          + (candidate_pIC50 - parent_pIC50) * multiplier
```

The 432.486 Da reference is the empirical median RDKit MolWt of the 18,801-structure OOF corpus. The
1.5469 baseline is the non-independent cliff scalar above. Callers may instead leave the baseline at
1.0 for an MW-only sensitivity test. An optional caller-chosen maximum multiplier is supported and is
returned in the response; no cap is silently imposed.

This scenario preserves the predicted direction, scales only the candidate-minus-parent delta in
pIC50 space, and never replaces the primary prediction. It should be labeled **experimental MW
sensitivity**, not corrected hERG potency.
