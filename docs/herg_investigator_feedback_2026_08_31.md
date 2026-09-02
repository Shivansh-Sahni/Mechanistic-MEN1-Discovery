# hERG investigator feedback — 2026-08-31

These requirements capture live testing feedback and must be considered in the next model and website revisions.

## Analog sensitivity

- Small structural changes are sometimes detected in the correct direction, but the predicted activity difference is too small (for example, predicting an IC50 change from 8 to 6 µM when the observed change is 8 to 2 µM).
- Evaluate close analogs in log-activity space using direction accuracy, pairwise delta error, series-ranking accuracy, and a cliff-capture ratio: `abs(predicted delta pIC50) / abs(observed delta pIC50)`.
- Develop a matched-molecular-pair or local-series residual model rather than relying only on global point-prediction error.
- Identify cases where small changes produce conflicting or unstable predictions.

## Functional-group effects

- **Standing requirement for future major work:** named functional groups, basicity-related motifs, and their parent-to-analog transformations must be evaluated as actual learned model inputs or pairwise training targets—not merely detected for the website or discussed after prediction. Every release must state explicitly which of these features affect the numerical prediction and which are interpretation-only.
- Functional-group changes must be represented and audited explicitly, including charge/protonation changes, heteroatom replacements, methyl and halogen substitutions, linker edits, ring changes, stereochemistry, aromaticity, hydrogen-bond donors/acceptors, and cationic-center placement.
- Report performance by transformation family and molecular-weight band so aggregate metrics cannot hide missed functional-group effects.
- For close analogs, classification probabilities and predicted IC50 should move in a chemically consistent direction and magnitude.

## Heavy molecules

- Current performance is weaker for some high-molecular-weight compounds, particularly when a small functional-group change is diluted by a large shared scaffold.
- Compounds above 750 Da should be allowed to attempt AutoDock Vina rather than being rejected solely by molecular weight.
- Separate docking execution eligibility from receptor-model applicability. Successful docking does not make an out-of-domain receptor prediction validated.
- Build and test a heavy-molecule expert or residual model, initially stratifying at 650 and 750 Da and accounting for heavy-atom count, flexibility, charge, and preparation/pose instability.
- A retrospective matched-pair audit now confirms progressive edit-magnitude attenuation through 600–700 Da, with inadequate support above 700 Da. The website exposes an off-by-default MW-proportional pIC50-delta stress scenario; it preserves the raw prediction and direction and is not a calibrated correction.

## Receptor-aware modeling

- Add an optional multi-receptor hERG workflow using 8ZYO and the existing 8ZYN, 8ZYP, 9CHP, and 9CHQ assets.
- Preserve receptor-specific scores and contacts; do not simply average Vina scores.
- Evaluate learned state gating, cross-state consensus/disagreement, and whether small modifications change the preferred receptor state or Y652/F656 cage engagement.
- Multi-receptor features must be tested specifically on close analog pairs and heavy compounds before affecting the default prediction.
- Website implementation now allows one to four states selected from 8ZYN, 8ZYO, 8ZYP, 8ZYQ, 9CHP, and 9CHQ. Only the sealed 8ZYO feature contract is model-coupled; the other states are real Vina diagnostics and must remain visibly docking-only until separately trained and validated models exist.

## Classification–regression coherence

- Potent/Moderate/Safe probabilities must align with the predicted IC50 and its uncertainty. A Moderate point estimate with a large Potent probability should be explained by a coherent uncertainty distribution or flagged as model disagreement.
- Prefer deriving tier probabilities from a calibrated latent pIC50 distribution using the same thresholds as the displayed IC50 result.
- Until then, display a clear classification–IC50 disagreement flag.

## Calculated properties

- Provide a fast properties-only website mode that calculates a broad, method-labeled descriptor and functional-group profile without running hERG prediction or docking.
- Show where selected properties fall within the hERG training distribution and the observed activity of nearby property bands. Label these as retrospective univariate associations, not causal effects or model attributions.
- Label every calculated property with its implementation and definition, including RDKit Wildman–Crippen cLogP, RDKit Ertl TPSA, average molecular weight, and exact mass where shown.
- Display the standardized parent structure used by the model. Differences from ChemDraw may reflect algorithm, salt handling, protonation, tautomer, aromaticity, implicit hydrogen, or software-version differences.
- Do not substitute ChemDraw values into models trained on RDKit values without retraining and validation.

## Validation boundary

- Maintain separate evaluations for novel-scaffold transfer, within-series interpolation, matched-pair delta prediction, activity cliffs, and heavy-molecule performance.
- Favorable training-overlap demonstrations are demonstrations only and must not be presented as independent validation.
- New internal assay results should be evaluated once against locked predictions before any retraining.

## Product workflow contract

- Keep three explicit researcher workflows: **Predict** for the retained single-compound hERG models, **Optimize** for manual parent-to-candidate comparisons, and **Properties** for model-free RDKit descriptor analysis.
- Optimization results must report the parent and candidate predictions separately, the signed pIC50 and IC50 changes, calculated-property deltas, Morgan/Tanimoto similarity, functional-group changes, basic-center changes, and any applicability warning. A favorable predicted edit is a prioritization hypothesis, not a designed or experimentally validated compound.
- When a high-similarity edit produces a small prediction change, show a low-sensitivity diagnostic instead of implying that the model captured the edit. Preserve the measured-versus-predicted direction and magnitude as separate future evaluation targets.
- Keep model outputs, calculated descriptors, and medicinal-chemistry interpretation visually and semantically distinct. Do not label RDKit structural rules as pKa, feature attribution, mechanism, or confidence.
- Centralize target-specific labels and thresholds so future receptor models can reuse the comparison interface, while keeping hERG the only live target until another model is validated.

## Immediate model experiment

- Build a locked matched-pair benchmark from experimentally measured close analogs before fitting another global model. Report directional accuracy, delta-pIC50 MAE, cliff-capture ratio, ties, and performance stratified by molecular weight, basic-center change, TPSA change, cLogP change, and transformation family.
- Use scaffold- or series-held-out validation for any learned pairwise residual model. Compare it with the frozen point predictor and require improvement in direction and magnitude without materially degrading global calibration.
