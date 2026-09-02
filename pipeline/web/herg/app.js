"use strict";

const TARGET_CONFIG = Object.freeze({
  herg: Object.freeze({
    key: "herg",
    label: "hERG / KCNH2",
    endpoints: Object.freeze({
      info: "/api/info",
      predict: "/api/predict",
      properties: "/api/properties",
      compare: "/api/compare",
      generate: "/api/generate",
      internalExamples: "/api/internal-examples",
    }),
    tiers: Object.freeze({
      Safe: "IC50 >30 µM",
      Moderate: "1 ≤ IC50 ≤ 30 µM",
      Potent: "IC50 <1 µM",
    }),
  }),
});

const ACTIVE_TARGET = TARGET_CONFIG.herg;
const element = (id) => document.getElementById(id);
const form = element("prediction-form");
const smilesInput = element("smiles");
const exampleSelect = element("example");
const exampleNote = element("example-note");
const predictButton = element("predict-button");
const clearButton = element("clear-button");
const status = element("status");
const results = element("results");
const propertyResults = element("property-results");
const propertyForm = element("property-form");
const propertySmilesInput = element("property-smiles-input");
const propertyButton = element("property-button");
const propertyClearButton = element("property-clear");
const propertyStatus = element("property-status");
const optimizeForm = element("optimize-form");
const parentSmilesInput = element("parent-smiles");
const candidateSmilesInput = element("candidate-smiles");
const optimizeButton = element("optimize-button");
const optimizeClearButton = element("optimize-clear");
const optimizeStatus = element("optimize-status");
const optimizeResults = element("optimize-results");
const manualOptimizeResult = element("manual-optimize-result");
const generatedOptimizeResult = element("generated-optimize-result");

let currentResult = null;
let currentPropertyResult = null;
let currentComparison = null;
let activeWorkflow = "predict";
const capabilityState = {
  multiReceptor: false,
  maximumReceptors: 6,
  mwDeltaStressTest: false,
  candidateGeneration: false,
  receptorIds: new Set(),
};
const internalExampleMetadata = new Map();

const selectedMode = () => form.elements.namedItem("mode").value;
const selectedFeatureMode = () => form.elements.namedItem("feature_mode").value;
const selectedOptimizeMode = () => optimizeForm.elements.namedItem("optimize_mode").value;

const formatNumber = (value, digits = 3) => {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (Math.abs(number) > 0 && Math.abs(number) < 0.001) return number.toExponential(2);
  if (Math.abs(number) >= 1000) return number.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return number.toFixed(digits);
};

const formatPercent = (value) => `${formatNumber(Number(value) * 100, 1)}%`;

const formatSigned = (value, digits = 3) => {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}`;
};

const formatDockingReason = (value) => String(value || "unavailable")
  .split(";")
  .map((reason) => {
    const labels = {
      molecular_weight_gt_750: "molecular weight >750 Da",
      heavy_atoms_gt_55: "heavy atoms >55",
      rotatable_bonds_gt_15: "rotatable bonds >15",
      rdkit_parse_failure: "structure preparation failed",
      eligible: "eligible",
    };
    if (labels[reason]) return labels[reason];
    if (reason.startsWith("unsupported_elements_")) {
      const symbols = reason.slice("unsupported_elements_".length).replaceAll("-", ", ");
      return `unsupported element(s): ${symbols}`;
    }
    return reason.replaceAll("_", " ");
  })
  .join("; ");

const formatScientificText = (value) => String(value)
  .replaceAll("molecular_weight_gt_750", "molecular weight >750 Da")
  .replaceAll("heavy_atoms_gt_55", "heavy atoms >55")
  .replaceAll("rotatable_bonds_gt_15", "rotatable bonds >15");

const setText = (id, value) => {
  element(id).textContent = value;
};

const setStatus = (message, kind = "") => {
  status.textContent = message;
  status.className = `status ${kind}`.trim();
};

const setPanelStatus = (node, message, kind = "") => {
  node.textContent = message;
  node.className = `status ${kind}`.trim();
};

const setActionLoading = (button, loading, idleLabel, loadingLabel) => {
  button.disabled = loading;
  button.classList.toggle("is-loading", loading);
  button.querySelector(".button-label").textContent = loading ? loadingLabel : idleLabel;
};

const workflowDescriptions = Object.freeze({
  predict: "Predict hERG liability from one structure.",
  optimize: "Compare one supplied edit or experimentally enumerate bounded candidates.",
  properties: "Calculate descriptors, ionization proxies, and functional-group context.",
});

const setWorkflow = (workflow, options = {}) => {
  if (!Object.hasOwn(workflowDescriptions, workflow)) return;
  activeWorkflow = workflow;
  document.querySelectorAll("[data-workflow-target]").forEach((tab) => {
    const active = tab.dataset.workflowTarget === workflow;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  document.querySelectorAll("[data-workflow-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.workflowPanel !== workflow;
  });
  document.querySelectorAll("[data-workflow-result]").forEach((panel) => {
    const hasResult = panel === results
      ? Boolean(currentResult)
      : (panel === propertyResults ? Boolean(currentPropertyResult) : Boolean(currentComparison));
    panel.hidden = panel.dataset.workflowResult !== workflow || !hasResult;
  });
  setText("workflow-description", workflowDescriptions[workflow]);
  if (options.copyCurrent !== false) {
    if (workflow === "properties" && !propertySmilesInput.value.trim() && smilesInput.value.trim()) {
      propertySmilesInput.value = smilesInput.value.trim();
    }
    if (workflow === "optimize" && !parentSmilesInput.value.trim() && smilesInput.value.trim()) {
      parentSmilesInput.value = smilesInput.value.trim();
    }
  }
  if (options.focus) {
    element(workflow).scrollIntoView({ behavior: "smooth", block: "start" });
  }
};

const RECEPTOR_LABELS = Object.freeze({
  "8ZYN": "apo",
  "8ZYO": "astemizole-bound · model-coupled",
  "8ZYP": "E-4031-bound · docking-only",
  "8ZYQ": "pimozide-bound · docking-only",
  "9CHP": "high-K+ C4 · docking-only",
  "9CHQ": "low-K+ C4 · docking-only",
});

const capabilityEntry = (info, names) => {
  const roots = [info?.capabilities, info?.api_capabilities, info?.features, info];
  for (const root of roots) {
    if (!root || typeof root !== "object") continue;
    for (const name of names) {
      if (root[name] !== undefined) return root[name];
    }
  }
  return null;
};

const capabilityEnabled = (entry) => {
  if (entry === true) return true;
  if (!entry || typeof entry !== "object") return false;
  return entry.available === true || entry.enabled === true || entry.supported === true;
};

const receptorIdsFromCapability = (entry) => {
  if (!entry || typeof entry !== "object") return [];
  const values = entry.receptor_ids || entry.available_receptor_ids
    || entry.available_receptors || entry.receptor_states || entry.states || entry.available_states || [];
  return (Array.isArray(values) ? values : []).map((value) => (
    typeof value === "string" ? value : value.id || value.receptor_id || value.pdb_id
  )).filter(Boolean);
};

const receptorInputs = (context) => [
  ...document.querySelectorAll(`input[data-receptor-context="${context}"]`),
];

const selectedReceptorIds = (context) => receptorInputs(context)
  .filter((input) => input.checked && !input.disabled)
  .map((input) => input.value);

const syncPredictReceptorControls = () => {
  const receptorMode = selectedMode() === "ligand_receptor";
  const selector = element("predict-receptor-selector");
  selector.hidden = !receptorMode;
  selector.disabled = !capabilityState.multiReceptor;
  receptorInputs("predict").forEach((input) => {
    input.disabled = !capabilityState.multiReceptor || !capabilityState.receptorIds.has(input.value);
  });
};

const syncOptimizeMode = () => {
  const generationMode = selectedOptimizeMode() === "generate";
  document.querySelectorAll("[data-optimize-manual]").forEach((node) => {
    node.hidden = generationMode;
    node.querySelectorAll("input, textarea, select").forEach((input) => {
      input.disabled = generationMode;
    });
  });
  element("generation-control").hidden = !generationMode;
  element("generated-candidate-count").disabled = !generationMode;
  candidateSmilesInput.required = !generationMode;
  const label = generationMode ? "Generate and rank edits" : "Compare structures";
  optimizeButton.querySelector(".button-label").textContent = label;
};

const configureCapabilities = (info) => {
  const multi = capabilityEntry(info, [
    "multi_receptor_diagnostics",
    "multi_receptor_docking",
    "multi_receptor",
  ]);
  const stress = capabilityEntry(info, [
    "experimental_mw_sensitivity",
    "mw_delta_stress_test",
    "mw_scaled_delta_stress_test",
  ]);
  const generation = capabilityEntry(info, [
    "experimental_candidate_generation",
    "candidate_generation",
  ]);
  capabilityState.multiReceptor = capabilityEnabled(multi);
  capabilityState.maximumReceptors = Number.isInteger(Number(multi?.maximum_selected))
    ? Number(multi.maximum_selected)
    : 6;
  capabilityState.mwDeltaStressTest = capabilityEnabled(stress);
  capabilityState.candidateGeneration = capabilityEnabled(generation);
  const advertisedIds = receptorIdsFromCapability(multi);
  capabilityState.receptorIds = new Set(
    capabilityState.multiReceptor
      ? (advertisedIds.length ? advertisedIds : Object.keys(RECEPTOR_LABELS))
      : [],
  );

  const receptorMessage = capabilityState.multiReceptor
    ? `Available. Select one to ${capabilityState.maximumReceptors} states; every selected state is docked, and only 8ZYO is model-coupled.`
    : "This server does not advertise multi-receptor diagnostics. Fixed 8ZYO receptor mode remains available.";
  setText("predict-receptor-capability", receptorMessage);
  syncPredictReceptorControls();

  element("mw-stress-control").disabled = !capabilityState.mwDeltaStressTest;
  element("mw-delta-stress-test").disabled = !capabilityState.mwDeltaStressTest;
  setText(
    "mw-stress-capability",
    capabilityState.mwDeltaStressTest
      ? "Available as an opt-in experimental diagnostic."
      : "This server does not advertise the MW-scaled stress test.",
  );
  const generationInput = optimizeForm.querySelector('input[name="optimize_mode"][value="generate"]');
  generationInput.disabled = !capabilityState.candidateGeneration;
  if (!capabilityState.candidateGeneration && generationInput.checked) {
    optimizeForm.querySelector('input[name="optimize_mode"][value="manual"]').checked = true;
  }
  syncOptimizeMode();
};

const enforceReceptorSelection = (event) => {
  const input = event.currentTarget;
  const context = input.dataset.receptorContext;
  const selected = selectedReceptorIds(context);
  const statusId = "predict-receptor-capability";
  if (selected.length > capabilityState.maximumReceptors) {
    input.checked = false;
    setText(statusId, `Select no more than ${capabilityState.maximumReceptors} receptor states.`);
    return;
  }
  if (!selected.length) {
    input.checked = true;
    setText(statusId, "Keep at least one receptor state selected.");
    return;
  }
  setText(
    statusId,
    `${selected.length} state${selected.length === 1 ? "" : "s"} selected. Only 8ZYO is model-coupled.`,
  );
};

const initializeInternalExamples = async () => {
  const group = element("internal-example-options");
  try {
    const response = await fetch(ACTIVE_TARGET.endpoints.internalExamples, {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) return;
    const data = await response.json();
    const examples = Array.isArray(data.examples) ? data.examples : [];
    examples.forEach((example) => {
      const measurement = example.measurement || {};
      const option = document.createElement("option");
      const relation = measurement.relation === "=" ? "" : measurement.relation || "";
      option.value = example.smiles;
      option.textContent = `${example.id} · observed ${relation}${formatNumber(measurement.value_um, 3)} µM · ${formatNumber(example.molecular_weight_da, 1)} Da`;
      group.appendChild(option);
      internalExampleMetadata.set(example.smiles, { ...example, selectionNote: data.selection_note });
    });
    group.hidden = examples.length === 0;
  } catch {
    group.hidden = true;
  }
};

const initializeCapabilities = async () => {
  try {
    const response = await fetch(ACTIVE_TARGET.endpoints.info, { headers: { Accept: "application/json" } });
    const info = response.ok ? await response.json() : null;
    configureCapabilities(info);
  } catch {
    configureCapabilities(null);
  }
};

const setLoading = (loading, mode = selectedMode()) => {
  predictButton.disabled = loading;
  clearButton.disabled = loading;
  predictButton.classList.toggle("is-loading", loading);
  let label = mode === "properties" ? "Calculate properties" : "Run prediction";
  if (loading) {
    label = mode === "ligand_receptor"
      ? "Docking receptor panel"
      : (mode === "properties" ? "Calculating properties" : "Running models");
  }
  predictButton.querySelector(".button-label").textContent = label;
  form.setAttribute("aria-busy", String(loading));
};

const appendProbability = (container, label, probability) => {
  const row = document.createElement("div");
  row.className = "probability-row";

  const name = document.createElement("span");
  name.textContent = `${label} (${ACTIVE_TARGET.tiers[label]})`;

  const track = document.createElement("progress");
  track.className = "probability-track";
  track.max = 1;
  track.value = Math.max(0, Math.min(1, Number(probability)));
  track.setAttribute("aria-label", `${label} probability`);

  const value = document.createElement("span");
  value.textContent = formatPercent(probability);

  row.append(name, track, value);
  container.appendChild(row);
};

const renderClassification = (classification) => {
  const tier = element("classification-tier");
  tier.textContent = classification.tier;
  tier.className = `tier-badge ${classification.tier.toLowerCase()}`;
  setText("classification-tier-definition", ACTIVE_TARGET.tiers[classification.tier] || "");
  const topProbability = Math.max(...Object.values(classification.probabilities));
  setText("classification-confidence", `${formatPercent(topProbability)} top probability`);

  const container = element("classification-bars");
  container.replaceChildren();
  ["Safe", "Moderate", "Potent"].forEach((label) => {
    appendProbability(container, label, classification.probabilities[label]);
  });
};

const renderRegressionTierProbabilities = (main) => {
  const block = element("regression-tier-probability-block");
  const container = element("regression-tier-bars");
  container.replaceChildren();
  const probabilities = main.tier_probabilities;
  if (!probabilities || typeof probabilities !== "object") {
    block.hidden = true;
    return;
  }
  ["Safe", "Moderate", "Potent"].forEach((label) => {
    appendProbability(container, label, probabilities[label]);
  });
  block.hidden = false;
};

const renderModelCoherence = (data) => {
  const notice = element("model-coherence-notice");
  const classifierTier = data.classification.tier;
  const regressionTier = data.main_ic50.tier;
  const supplied = data.classification_regression_consistency;
  const disagreement = typeof supplied?.classifier_vs_regression_point_tier_disagreement === "boolean"
    ? supplied.classifier_vs_regression_point_tier_disagreement
    : classifierTier !== regressionTier;
  const agrees = !disagreement;
  const classifierProbability = data.classification.probabilities?.[classifierTier];
  const directSummary = agrees
    ? `The independent classifier and the regression point estimate both indicate ${regressionTier}. `
      + "They remain separate statistical outputs."
    : `Output disagreement: the independent classifier favors ${classifierTier} `
      + `(${formatPercent(classifierProbability)}), while the regression point estimate falls in the `
      + `${regressionTier} tier. For example, an IC50 near 3 µM and a substantial Potent-class `
      + "probability are not mathematically required to match because the models were trained separately.";
  notice.textContent = supplied?.interpretation
    ? `${directSummary} ${supplied.interpretation}`
    : directSummary;
  notice.className = agrees ? "callout model-coherence" : "callout warning model-coherence";
};

const renderResearchPreview = (preview) => {
  const classification = preview.classification;
  const tier = element("research-classification-tier");
  tier.textContent = classification.tier;
  tier.className = `tier-badge ${classification.tier.toLowerCase()}`;
  const topProbability = Math.max(...Object.values(classification.probabilities));
  setText("research-classification-confidence", `${formatPercent(topProbability)} top probability`);

  const container = element("research-classification-bars");
  container.replaceChildren();
  ["Safe", "Moderate", "Potent"].forEach((label) => {
    appendProbability(container, label, classification.probabilities[label]);
  });

  setText("research-ic50", formatNumber(preview.literature_ic50.ic50_um));
  setText("research-pic50", formatNumber(preview.literature_ic50.pic50));
  setText("research-balanced-accuracy", formatNumber(preview.validation.balanced_accuracy, 3));

  const delta = preview.validation.delta_vs_frozen;
  const ba = delta.classification_balanced_accuracy;
  const mae = delta.regression_mae_pic50;
  const linearRegression = preview.validation.regression_delta_vs_v14_linear;
  setText(
    "research-gain",
    `vs frozen: balanced accuracy ${formatSigned(ba.observed)} `
      + `(95% CI ${formatSigned(ba.ci95[0])} to ${formatSigned(ba.ci95[1])}); `
      + `pIC50 MAE ${formatSigned(mae.observed)} `
      + `(95% CI ${formatSigned(mae.ci95[0])} to ${formatSigned(mae.ci95[1])}); `
      + `tree regression vs V14 linear ${formatSigned(linearRegression.observed)} `
      + `(95% CI ${formatSigned(linearRegression.ci95[0])} to `
      + `${formatSigned(linearRegression.ci95[1])}).`,
  );
  setText(
    "research-receptor-status",
    `Receptor-aware challenge evaluated, not promoted: ${preview.receptor_model.reason}.`,
  );
  setText("research-boundary", preview.claim_boundary);
};

const renderReceptor = (receptor) => {
  const panel = element("receptor-result-panel");
  if (!receptor || receptor.status === "not_requested") {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  if (receptor.status !== "research_only_non_default") {
    setText("receptor-classification-tier", "Unavailable");
    element("receptor-classification-tier").className = "tier-badge";
    setText("receptor-classification-confidence", "");
    element("receptor-classification-bars").replaceChildren();
    ["receptor-ic50", "receptor-pic50", "receptor-delta", "receptor-affinity"].forEach((id) => {
      setText(id, "—");
    });
    setText(
      "receptor-docking-summary",
      receptor.reason ? formatDockingReason(receptor.reason) : "The structure is not docking-eligible.",
    );
    ["receptor-state", "receptor-contacts", "receptor-runtime", "receptor-cache"].forEach((id) => {
      setText(id, "—");
    });
    setText(
      "receptor-validation-status",
      receptor.reason ? formatDockingReason(receptor.reason) : "Receptor analysis unavailable.",
    );
    setText("receptor-boundary", "The ligand-only result remains available above.");
    return;
  }

  const classification = receptor.classification;
  const tier = element("receptor-classification-tier");
  tier.textContent = classification.tier;
  tier.className = `tier-badge ${classification.tier.toLowerCase()}`;
  const topProbability = Math.max(...Object.values(classification.probabilities));
  setText("receptor-classification-confidence", `${formatPercent(topProbability)} top probability`);
  const container = element("receptor-classification-bars");
  container.replaceChildren();
  ["Safe", "Moderate", "Potent"].forEach((label) => {
    appendProbability(container, label, classification.probabilities[label]);
  });

  setText("receptor-ic50", formatNumber(receptor.literature_ic50.ic50_um));
  setText("receptor-pic50", formatNumber(receptor.literature_ic50.pic50));
  setText("receptor-delta", formatSigned(receptor.literature_ic50.delta_from_ligand_pic50));
  setText("receptor-affinity", `${formatNumber(receptor.docking.best_affinity_kcal_mol, 2)} kcal/mol`);
  setText(
    "receptor-docking-summary",
    `${receptor.docking.returned_pose_count} poses · exhaustiveness ${receptor.docking.exhaustiveness}`,
  );
  setText("receptor-state", receptor.docking.receptor_state);
  setText(
    "receptor-contacts",
    Object.entries(receptor.docking.best_pose_contacts).map(([name, count]) => `${name} ${count}`).join(" · "),
  );
  setText("receptor-runtime", `${formatNumber(receptor.runtime.total_seconds, 2)} s`);
  setText("receptor-cache", receptor.runtime.cache_hit ? "Reused in-memory result" : "New docking run");

  const validation = receptor.validation;
  const classDelta = validation.delta_vs_best_ligand.classification_balanced_accuracy;
  const regressionDelta = validation.delta_vs_best_ligand.regression_mae_pic50;
  setText(
    "receptor-validation-status",
    `Six-campaign research comparison: classification BA Δ ${formatSigned(classDelta.observed, 4)} `
      + `(95% CI ${formatSigned(classDelta.ci95[0], 4)} to ${formatSigned(classDelta.ci95[1], 4)}); `
      + `pIC50 MAE Δ ${formatSigned(regressionDelta.observed, 4)} `
      + `(95% CI ${formatSigned(regressionDelta.ci95[0], 4)} to ${formatSigned(regressionDelta.ci95[1], 4)}).`,
  );
  setText("receptor-boundary", receptor.claim_boundary);
};

const renderAnalogueDisclosure = (applicability) => {
  setText(
    "analogue-disclosure",
    applicability.analogue_detail_disclosure
      || "Training-record identifiers, structures, measurements, and per-record errors are not disclosed by this release.",
  );
};

const appendDefinitionRow = (container, label, value) => {
  const row = document.createElement("div");
  const term = document.createElement("dt");
  const description = document.createElement("dd");
  term.textContent = label;
  description.textContent = value;
  row.append(term, description);
  container.appendChild(row);
};

const appendTextList = (container, items, emptyText, className = "") => {
  container.replaceChildren();
  const values = Array.isArray(items) ? items : [];
  if (!values.length) {
    const empty = document.createElement("p");
    empty.className = "empty-note";
    empty.textContent = emptyText;
    container.appendChild(empty);
    return;
  }
  values.forEach((item) => {
    const node = document.createElement(className === "interpretation-list" ? "li" : "span");
    const label = typeof item === "string" ? item : item.label || item.key || "Detected group";
    const rawCount = typeof item === "object" ? (item.count ?? item.count_change) : null;
    const count = Number.isFinite(Number(rawCount)) && Math.abs(Number(rawCount)) > 1
      ? ` ×${Math.abs(Number(rawCount))}`
      : "";
    node.textContent = `${label}${count}`;
    container.appendChild(node);
  });
};

const descriptorValue = (properties, key) => {
  const descriptor = properties?.descriptors?.[key];
  return descriptor && Number.isFinite(Number(descriptor.value)) ? Number(descriptor.value) : null;
};

const safeStructureImage = (id, ...sources) => {
  const image = element(id);
  const source = sources.find((value) => (
    typeof value === "string" && /^data:image\/(?:png|webp|svg\+xml);base64,/i.test(value)
  ));
  if (source) {
    image.src = source;
    image.hidden = false;
  } else {
    image.removeAttribute("src");
    image.hidden = true;
  }
};

const renderProperties = (data) => {
  currentPropertyResult = data;
  setText("property-smiles", data.smiles);
  element("property-smiles").title = data.smiles;
  setText(
    "property-standardization-note",
    data.standardization?.disclosure
      || "Values describe the standardized parent structure used by the project pipeline.",
  );
  setText("property-runtime", `Properties-only mode · ${formatNumber(data.runtime_seconds, 3)} seconds`);
  safeStructureImage(
    "property-structure-image",
    data.structure_depiction?.data_uri,
    data.structure_image_data_uri,
  );

  const propertyTable = element("property-table");
  propertyTable.replaceChildren();
  Object.values(data.descriptors).forEach((descriptor) => {
    const row = document.createElement("tr");
    const label = document.createElement("th");
    const value = document.createElement("td");
    const method = document.createElement("td");
    label.scope = "row";
    label.textContent = descriptor.label;
    const numeric = Number(descriptor.value);
    const digits = Number.isInteger(numeric) ? 0 : 3;
    value.textContent = `${formatNumber(numeric, digits)}${descriptor.unit ? ` ${descriptor.unit}` : ""}`;
    method.textContent = descriptor.method;
    row.append(label, value, method);
    propertyTable.appendChild(row);
  });

  const functionalGroups = element("functional-group-list");
  appendTextList(
    functionalGroups,
    data.functional_groups,
    "No functional groups from the current named-fragment panel were detected.",
  );

  const basicity = data.basicity || {};
  const basicitySummary = element("basicity-summary");
  basicitySummary.replaceChildren();
  appendDefinitionRow(basicitySummary, "Structural classification", basicity.classification || "Not available");
  appendDefinitionRow(
    basicitySummary,
    "Positive-ionizable centers",
    formatNumber(basicity.positive_ionizable_centers, 0),
  );
  appendDefinitionRow(
    basicitySummary,
    "Negative-ionizable centers",
    formatNumber(basicity.negative_ionizable_centers, 0),
  );
  appendDefinitionRow(
    basicitySummary,
    "Explicit positive / negative atoms",
    `${formatNumber(basicity.permanent_positive_atoms, 0)} / ${formatNumber(basicity.permanent_negative_atoms, 0)}`,
  );
  appendDefinitionRow(
    basicitySummary,
    "Cationic + lipophilic + aromatic pattern",
    basicity.cationic_lipophilic_aromatic_pattern ? "Detected" : "Not detected",
  );

  const relationships = element("herg-relationship-list");
  relationships.replaceChildren();
  (basicity.herg_relationship_context || []).forEach((statement) => {
    const paragraph = document.createElement("p");
    paragraph.textContent = statement;
    relationships.appendChild(paragraph);
  });
  setText("basicity-boundary", basicity.boundary || "No numerical pKa is calculated by this release.");

  const docking = data.docking_eligibility || {};
  const dockingStatus = element("property-docking-status");
  dockingStatus.textContent = docking.eligible_under_current_receptor_model
    ? "Inside the sealed receptor-model applicability contract; Vina execution is also eligible."
    : `Outside the sealed receptor-model applicability contract: ${formatDockingReason(docking.reason)}. `
      + (docking.docking_execution_eligible
        ? "Vina may still be attempted as a docking-only diagnostic."
        : `Vina execution is blocked: ${formatDockingReason(docking.docking_execution_reason)}.`);
  dockingStatus.className = docking.eligible_under_current_receptor_model ? "callout" : "callout warning";

  const associationTable = element("property-association-table");
  associationTable.replaceChildren();
  (data.training_associations || []).forEach((association) => {
    const row = document.createElement("tr");
    const label = document.createElement("th");
    const percentile = document.createElement("td");
    const median = document.createElement("td");
    const interpretation = document.createElement("td");
    label.scope = "row";
    label.textContent = association.label;
    percentile.textContent = formatPercent(association.percentile);
    median.textContent = formatNumber(association.nearby_band_median_pic50, 3);
    interpretation.textContent = association.interpretation;
    row.append(label, percentile, median, interpretation);
    associationTable.appendChild(row);
  });
  setText("property-boundary", data.claim_boundary);
  propertyResults.hidden = false;
};

const renderPredictionProperties = (properties) => {
  const groups = element("prediction-functional-groups");
  if (!properties) {
    setText("prediction-formal-charge", "—");
    setText("prediction-basic-centers", "—");
    groups.replaceChildren();
    setText("prediction-property-method", "Detailed property context was unavailable; the prediction itself is unaffected.");
    return;
  }
  setText("prediction-formal-charge", formatSigned(descriptorValue(properties, "formal_charge"), 0));
  setText(
    "prediction-basic-centers",
    `${formatNumber(properties.basicity?.positive_ionizable_centers, 0)} positive-ionizable`,
  );
  appendTextList(groups, properties.functional_groups, "No named functional groups detected by the current panel.");
  setText(
    "prediction-property-method",
    `${properties.basicity?.method || "RDKit structural feature rules"}. `
      + "No numerical pKa is calculated; descriptors may differ from ChemDraw or other software. "
      + (properties.standardization?.disclosure || "Values describe the standardized model input."),
  );
};

const renderFunctionalGroupResearch = (data) => {
  const panel = element("functional-group-result-panel");
  const research = data.functional_group_research;
  if (!research || data.feature_mode?.selected !== "functional_group") {
    panel.hidden = true;
    return;
  }
  const regression = research.regression || {};
  setText("fg-default-ic50", formatNumber(data.main_ic50?.ic50_um));
  setText("fg-research-ic50", formatNumber(regression.ic50_um));
  setText("fg-delta-pic50", formatSigned(regression.delta_vs_atomwise_pic50));
  setText("fg-fold-change", `${formatNumber(regression.fold_vs_atomwise_ic50, 2)}× IC50 vs atom-wise`);

  const groups = element("fg-active-groups");
  groups.replaceChildren();
  const activeGroups = research.feature_contract?.active_functional_groups || [];
  activeGroups.forEach((group) => {
    const chip = document.createElement("span");
    chip.textContent = `${group.label} ×${group.count}`;
    chip.title = "Prediction-active only in the selected non-default V15 research surface";
    groups.appendChild(chip);
  });
  if (!activeGroups.length) {
    const empty = document.createElement("span");
    empty.textContent = "No governed SMARTS groups detected";
    groups.appendChild(empty);
  }

  const body = element("fg-contribution-table");
  body.replaceChildren();
  (research.feature_contract?.top_linear_contributions_pic50 || []).forEach((entry) => {
    const row = document.createElement("tr");
    const label = document.createElement("th");
    const contribution = document.createElement("td");
    label.scope = "row";
    label.textContent = entry.label;
    contribution.textContent = formatSigned(entry.contribution_pic50, 4);
    row.append(label, contribution);
    body.appendChild(row);
  });
  const validation = research.validation || {};
  const delta = validation.delta_mae_vs_deployed || {};
  setText(
    "fg-validation-note",
    `Nested leave-one-campaign-out macro MAE ${formatNumber(validation.nested_leave_one_campaign_out_macro_mae_pic50, 3)} pIC50 across ${formatNumber(validation.campaigns, 0)} campaigns. `
      + `MAE delta vs deployed ${formatSigned(delta.observed, 3)} pIC50 (95% CI ${formatSigned(delta.ci95?.[0], 3)} to ${formatSigned(delta.ci95?.[1], 3)}).`,
  );
  setText("fg-boundary", research.claim_boundary);
  panel.hidden = false;
};

const renderHeavyAnalysis = (data) => {
  const panel = element("heavy-analysis-panel");
  const analysis = data.advanced_heavy_analysis;
  if (!analysis?.enabled) {
    panel.hidden = true;
    return;
  }
  const docking = analysis.docking_contract || {};
  setText("heavy-analysis-mw", `${formatNumber(analysis.molecular_weight_da, 1)} Da`);
  setText("heavy-analysis-atoms", formatNumber(analysis.heavy_atom_count, 0));
  setText("heavy-analysis-rotors", formatNumber(analysis.rotatable_bonds, 0));
  setText(
    "heavy-analysis-properties",
    `${formatNumber(analysis.clogp, 2)} / ${formatNumber(analysis.tpsa, 1)} Å²`,
  );
  setText(
    "heavy-analysis-docking",
    docking.docking_execution_eligible
      ? "Eligible for an attempted diagnostic Vina run"
      : `Blocked: ${formatDockingReason(docking.docking_execution_reason)}`,
  );
  setText(
    "heavy-analysis-domain",
    docking.receptor_model_applicable
      ? "Inside sealed 8ZYO model contract"
      : `Outside: ${formatDockingReason(docking.receptor_model_applicability_reason)}`,
  );
  setText("heavy-analysis-note", analysis.interpretation);
  panel.hidden = false;
};

const renderResult = (data, properties = null) => {
  currentResult = data;
  const main = data.main_ic50;
  const selectedMain = data.selected_quantitative_result || main;
  const direct = data.direct_functional_curve;
  const applicability = data.applicability;
  const confidence = selectedMain.decision_confidence || main.decision_confidence;
  const chemistry = data.chemistry;
  const propertyProfile = properties || data.calculated_properties || data.property_profile || null;
  const receptorRequested = data.scientific_scope?.receptor_mode_requested === true
    || data.prediction_mode === "ligand_receptor";
  const receptorFeaturesUsed = data.scientific_scope?.receptor_features_used === true;
  const modeLabel = receptorFeaturesUsed
    ? "Ligand + receptor mode"
    : (receptorRequested ? "Ligand-only result · receptor unavailable" : "Ligand-only mode");
  const featureLabel = data.feature_mode?.selected === "functional_group"
    ? "functional-group research surface selected"
    : "validated atom-wise surface";

  setText("canonical-smiles", data.smiles);
  element("canonical-smiles").title = data.smiles;
  setText(
    "prediction-runtime",
    `${modeLabel} · ${featureLabel} · `
      + `${formatNumber(data.runtime.total_seconds, 2)} seconds`
      + (data.runtime.receptor_cache_hit ? " · receptor cache hit" : ""),
  );
  renderClassification(data.classification);
  renderRegressionTierProbabilities(selectedMain);
  renderModelCoherence({
    ...data,
    main_ic50: selectedMain,
    classification_regression_consistency: data.feature_mode?.selected === "functional_group"
      ? null
      : data.classification_regression_consistency,
  });
  renderResearchPreview(data.research_preview);
  renderReceptor(data.receptor_aware);
  renderFunctionalGroupResearch(data);
  renderHeavyAnalysis(data);
  renderMultiReceptorDiagnostics(
    data.multi_receptor_diagnostics || data.receptor_panel_diagnostics,
    "predict",
  );

  setText("primary-model-label", data.feature_mode?.label || "Validated atom-wise baseline");
  setText("primary-ic50", formatNumber(selectedMain.ic50_um));
  setText("primary-pic50", formatNumber(selectedMain.pic50));
  setText("regression-tier", selectedMain.tier);
  setText("regression-tier-definition", ACTIVE_TARGET.tiers[selectedMain.tier] || "");
  setText(
    "primary-interval",
    selectedMain.interval90_um
      ? `${formatNumber(selectedMain.interval90_um.lower, 2)}–${formatNumber(selectedMain.interval90_um.upper, 2)} µM`
      : "Not calibrated for this research surface",
  );
  setText(
    "selected-model-note",
    data.feature_mode?.selected === "functional_group"
      ? "Selected V15 functional-group research prediction. It is a real model output, but it failed promotion; the validated atom-wise result remains available below. No calibrated interval is claimed."
      : "The tier is a deterministic bin of the point estimate. The bars apply the pooled signed out-of-fold residual distribution and are empirical frequencies—not query-calibrated class probabilities.",
  );

  setText("decision-confidence", confidence.label);
  setText("decision-explanation", confidence.explanation);
  const overlap = element("overlap-notice");
  if (applicability.exact_training_overlap) {
    overlap.textContent = "Exact training overlap: use this as a model demonstration, not independent evidence.";
    overlap.className = "inline-notice warning";
  } else {
    overlap.textContent = "No exact training overlap was detected.";
    overlap.className = "inline-notice";
  }

  setText("direct-ic10", formatNumber(direct.ic10_um));
  setText("direct-ic30", formatNumber(direct.ic30_um));
  setText("direct-ic50", formatNumber(direct.ic50_um));
  const projection = element("projection-badge");
  projection.textContent = direct.ordering_adjusted ? "Order projection applied" : "Predictions already ordered";

  const consistency = direct.cross_assay_consistency;
  const crossAssay = element("cross-assay-notice");
  crossAssay.textContent = `${consistency.label} (${formatNumber(consistency.fold_ratio, 1)}×): ${consistency.interpretation}`;
  crossAssay.className = consistency.fold_ratio > 3 ? "callout warning" : "callout";

  setText("nearest-similarity", formatNumber(applicability.maximum_train_tanimoto, 3));
  setText("analogue-count", Number(applicability.analog_count_ge_0p5).toLocaleString());
  setText(
    "local-mae",
    `${formatNumber(applicability.local_oof_error_diagnostic.retrospective_mae_pic50, 3)} pIC50`,
  );
  setText("domain-label", applicability.domain_label);
  setText("domain-interpretation", applicability.interpretation);

  setText("molecular-weight", `${formatNumber(chemistry.molecular_weight, 1)} Da`);
  setText("clogp", formatNumber(chemistry.clogp, 2));
  setText("tpsa", `${formatNumber(chemistry.tpsa, 1)} Å²`);
  setText("hbd-hba", `${chemistry.hbd} / ${chemistry.hba}`);
  setText("rotatable-bonds", String(chemistry.rotatable_bonds));
  renderPredictionProperties(propertyProfile);
  const highMwWarning = element("prediction-mw-warning");
  const highMwWarnings = data.heavy_molecule_diagnostic?.warnings || [];
  if (highMwWarnings.length) {
    highMwWarning.textContent = formatScientificText(highMwWarnings.join(" "));
    highMwWarning.hidden = false;
  } else {
    highMwWarning.textContent = "";
    highMwWarning.hidden = true;
  }
  safeStructureImage(
    "prediction-structure-image",
    data.structure_depiction?.data_uri,
    propertyProfile?.structure_depiction?.data_uri,
    propertyProfile?.structure_image_data_uri,
  );
  renderAnalogueDisclosure(applicability);

  results.hidden = false;
};

const modelSummary = (model = {}) => ({
  smiles: model.smiles || model.canonical_smiles || "",
  ic50: model.regression?.predicted_ic50_um
    ?? model.predicted_ic50_um
    ?? model.main_ic50?.ic50_um
    ?? model.ic50_um,
  pic50: model.regression?.predicted_pic50
    ?? model.predicted_pic50
    ?? model.main_ic50?.pic50
    ?? model.pic50,
  tier: model.regression?.threshold_tier || model.tier || model.main_ic50?.tier || "—",
  classifierTier: model.classifier?.predicted_tier || model.classification?.tier || "—",
  classifierProbabilities: model.classifier?.probabilities
    || model.classification?.probabilities
    || model.tier_probabilities
    || {},
});

const topClassifierLabel = (summary) => {
  const probabilities = summary.classifierProbabilities;
  const probability = probabilities?.[summary.classifierTier];
  return summary.classifierTier === "—"
    ? "Not available"
    : `${summary.classifierTier}${Number.isFinite(Number(probability)) ? ` (${formatPercent(probability)})` : ""}`;
};

const comparisonProperty = (properties, key) => {
  if (key === "basic_centers") return Number(properties?.basicity?.positive_ionizable_centers);
  return descriptorValue(properties, key);
};

const renderComparisonPropertyTable = (data) => {
  const properties = data.calculated_properties || {};
  const deltas = properties.deltas || {};
  const body = element("optimize-property-table");
  body.replaceChildren();
  const rows = [
    ["molecular_weight", "Molecular weight", "Da", 1],
    ["tpsa", "TPSA", "Å²", 1],
    ["clogp", "RDKit cLogP", "", 2],
    ["hbd", "H-bond donors", "", 0],
    ["hba", "H-bond acceptors", "", 0],
    ["rotatable_bonds", "Rotatable bonds", "", 0],
    ["formal_charge", "Formal charge", "", 0],
    ["basic_centers", "Positive-ionizable centers", "", 0],
  ];
  rows.forEach(([key, labelText, unit, digits]) => {
    const parentValue = comparisonProperty(properties.parent, key);
    const candidateValue = comparisonProperty(properties.candidate, key);
    const providedDelta = deltas[key]?.candidate_minus_parent ?? deltas[key];
    const calculatedDelta = Number.isFinite(parentValue) && Number.isFinite(candidateValue)
      ? candidateValue - parentValue
      : null;
    const delta = Number.isFinite(Number(providedDelta)) ? Number(providedDelta) : calculatedDelta;
    const row = document.createElement("tr");
    const label = document.createElement("th");
    const parent = document.createElement("td");
    const candidate = document.createElement("td");
    const deltaCell = document.createElement("td");
    label.scope = "row";
    label.textContent = labelText;
    parent.textContent = `${formatNumber(parentValue, digits)}${unit ? ` ${unit}` : ""}`;
    candidate.textContent = `${formatNumber(candidateValue, digits)}${unit ? ` ${unit}` : ""}`;
    deltaCell.textContent = `${formatSigned(delta, digits)}${unit ? ` ${unit}` : ""}`;
    deltaCell.className = Number(delta) === 0 ? "delta-neutral" : (Number(delta) > 0 ? "delta-up" : "delta-down");
    row.append(label, parent, candidate, deltaCell);
    body.appendChild(row);
  });
};

const renderComparisonWarnings = (comparison) => {
  const container = element("optimize-warning-list");
  container.replaceChildren();
  const warnings = Array.isArray(comparison.warnings) ? [...comparison.warnings] : [];
  const heavy = comparison.heavy_molecule_diagnostic || {};
  [heavy.parent, heavy.candidate].forEach((diagnostic) => {
    if (!diagnostic || typeof diagnostic !== "object") return;
    const message = diagnostic.warning || diagnostic.message || diagnostic.caution;
    if (message && !warnings.includes(message)) warnings.push(message);
  });
  warnings.forEach((warning) => {
    const notice = document.createElement("div");
    notice.className = "callout warning";
    notice.textContent = formatScientificText(warning);
    container.appendChild(notice);
  });
};

const diagnosticRows = (diagnostics) => {
  if (!diagnostics || typeof diagnostics !== "object") return [];
  const collection = diagnostics.receptor_states || diagnostics.states
    || diagnostics.receptors || diagnostics.results || diagnostics.per_receptor || [];
  if (Array.isArray(collection)) return collection;
  if (collection && typeof collection === "object") {
    return Object.entries(collection).map(([receptorId, value]) => ({
      receptor_id: receptorId,
      ...(value && typeof value === "object" ? value : { status: value }),
    }));
  }
  return [];
};

const renderMultiReceptorDiagnostics = (diagnostics, context) => {
  const panel = element(`${context}-multi-receptor-panel`);
  const body = element(`${context}-multi-receptor-table`);
  body.replaceChildren();
  if (!diagnostics || typeof diagnostics !== "object") {
    panel.hidden = true;
    return;
  }
  const rows = diagnosticRows(diagnostics);
  if (!rows.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.textContent = diagnostics.message || diagnostics.status || "No per-receptor rows were returned.";
    row.appendChild(cell);
    body.appendChild(row);
  } else {
    rows.forEach((diagnostic) => {
      const receptorId = diagnostic.receptor_id || diagnostic.id || diagnostic.pdb_id || "—";
      let role = diagnostic.role || diagnostic.model_role
        || (receptorId === "8ZYO" ? "Model-coupled" : "Docking-only");
      if (receptorId === "8ZYO" && diagnostic.model_prediction_generated === false) {
        role = "8ZYO contract; docking-only this run";
      }
      const affinity = diagnostic.best_affinity_kcal_mol
        ?? diagnostic.best_vina_affinity_kcal_mol
        ?? diagnostic.best_vina_score_kcal_mol
        ?? diagnostic.docking?.best_affinity_kcal_mol;
      const poses = diagnostic.pose_count
        ?? diagnostic.returned_pose_count
        ?? diagnostic.docking?.returned_pose_count;
      const values = [
        receptorId,
        String(role).replaceAll("_", " "),
        diagnostic.status || diagnostic.docking_status || "Completed",
        Number.isFinite(Number(affinity)) ? `${formatNumber(affinity, 2)} kcal/mol` : "—",
        formatNumber(poses, 0),
      ];
      const row = document.createElement("tr");
      values.forEach((value, index) => {
        const cell = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) cell.scope = "row";
        cell.textContent = String(value);
        row.appendChild(cell);
      });
      body.appendChild(row);
    });
  }
  const crossState = diagnostics.cross_state || {};
  const crossStateSummary = crossState.lowest_vina_score_receptor_id
    ? ` Lowest Vina score: ${crossState.lowest_vina_score_receptor_id}; cross-state score range `
      + `${formatNumber(crossState.vina_score_range_kcal_mol, 2)} kcal/mol. `
      + `${crossState.interpretation || ""}`
    : "";
  setText(
    `${context}-multi-receptor-boundary`,
    `${diagnostics.claim_boundary || diagnostics.interpretation
      || "Non-8ZYO states are exploratory AutoDock Vina diagnostics and do not contribute to a trained potency model."}${crossStateSummary}`,
  );
  panel.hidden = false;
};

const renderMwSensitivity = (diagnostic) => {
  const panel = element("mw-sensitivity-panel");
  const metrics = element("mw-sensitivity-metrics");
  const statements = element("mw-sensitivity-statements");
  metrics.replaceChildren();
  statements.replaceChildren();
  if (!diagnostic || typeof diagnostic !== "object") {
    panel.hidden = true;
    return;
  }
  const source = diagnostic.metrics && typeof diagnostic.metrics === "object"
    ? { ...diagnostic, ...diagnostic.metrics }
    : diagnostic;
  const definitions = [
    ["Status", source.status],
    ["Parent molecular weight", source.parent_molecular_weight_da ?? source.parent_mw_da, "Da", 1],
    ["Candidate molecular weight", source.candidate_molecular_weight_da ?? source.candidate_mw_da, "Da", 1],
    ["Raw predicted ΔpIC50", source.raw_delta_pic50 ?? source.predicted_delta_pic50, "", 3],
    ["MW-scaled stress ΔpIC50", source.mw_scaled_delta_pic50 ?? source.stress_test_delta_pic50, "", 3],
    ["Stress scaling factor", source.scaling_factor ?? source.mw_scaling_factor, "×", 3],
    ["Primary candidate IC50", source.primary_candidate_ic50_um, " µM", 3],
    ["Stress-scenario candidate IC50", source.stress_candidate_ic50_um, " µM", 3],
    ["Magnitude ratio", source.magnitude_ratio ?? source.delta_magnitude_ratio, "×", 3],
  ];
  definitions.forEach(([label, value, unit = "", digits = 3]) => {
    if (value === undefined || value === null || value === "") return;
    const formatted = Number.isFinite(Number(value))
      ? `${formatNumber(value, digits)}${unit}`
      : String(value).replaceAll("_", " ");
    appendDefinitionRow(metrics, label, formatted);
  });
  const messages = [
    ...(Array.isArray(diagnostic.statements) ? diagnostic.statements : []),
    ...(Array.isArray(diagnostic.interpretation) ? diagnostic.interpretation : []),
  ];
  if (typeof diagnostic.interpretation === "string") messages.push(diagnostic.interpretation);
  if (typeof diagnostic.message === "string") messages.push(diagnostic.message);
  messages.forEach((message) => {
    const paragraph = document.createElement("p");
    paragraph.textContent = message;
    statements.appendChild(paragraph);
  });
  if (!metrics.children.length) appendDefinitionRow(metrics, "Diagnostic", "Returned without scalar metrics");
  setText(
    "mw-sensitivity-boundary",
    diagnostic.claim_boundary
      || "Experimental sensitivity stress test only. It does not recalibrate the model or replace the displayed prediction.",
  );
  panel.hidden = false;
};

const renderComparison = (data) => {
  currentComparison = data;
  manualOptimizeResult.hidden = false;
  generatedOptimizeResult.hidden = true;
  setText("optimize-results-kicker", "Manual structural-edit analysis");
  setText("optimize-results-title", "Parent → candidate comparison");
  const parent = modelSummary(data.model_outputs?.parent);
  const candidate = modelSummary(data.model_outputs?.candidate);
  const comparison = data.comparison || {};
  const direction = String(comparison.direction || "essentially unchanged").replaceAll("_", " ");
  const directionBadge = element("optimize-direction");
  directionBadge.textContent = direction;
  directionBadge.className = `direction-badge ${direction.toLowerCase().replaceAll(" ", "-")}`;
  setText("optimize-runtime", `Ligand-only comparison · ${formatNumber(data.runtime_seconds, 2)} seconds`);

  setText("parent-smiles-output", parent.smiles);
  setText("candidate-smiles-output", candidate.smiles);
  element("parent-smiles-output").title = parent.smiles;
  element("candidate-smiles-output").title = candidate.smiles;
  setText("parent-ic50", formatNumber(parent.ic50));
  setText("candidate-ic50", formatNumber(candidate.ic50));
  setText("parent-pic50", formatNumber(parent.pic50));
  setText("candidate-pic50", formatNumber(candidate.pic50));
  setText("parent-tier", parent.tier);
  setText("candidate-tier", candidate.tier);
  setText("parent-classifier", topClassifierLabel(parent));
  setText("candidate-classifier", topClassifierLabel(candidate));

  const parentProperties = data.calculated_properties?.parent || {};
  const candidateProperties = data.calculated_properties?.candidate || {};
  safeStructureImage(
    "parent-structure-image",
    data.model_outputs?.parent?.structure_depiction?.data_uri,
    data.model_outputs?.parent?.structure_image_data_uri,
    parentProperties.structure_image_data_uri,
    parentProperties.structure_svg_data_uri,
  );
  safeStructureImage(
    "candidate-structure-image",
    data.model_outputs?.candidate?.structure_depiction?.data_uri,
    data.model_outputs?.candidate?.structure_image_data_uri,
    candidateProperties.structure_image_data_uri,
    candidateProperties.structure_svg_data_uri,
  );

  const measured = data.measured_values;
  const measuredContext = element("known-parent-context");
  if (measured && Number.isFinite(Number(measured.parent_ic50_um))) {
    measuredContext.textContent = `User-supplied parent IC50: ${formatNumber(measured.parent_ic50_um)} µM `
      + "(context only; not substituted for the model output).";
    measuredContext.hidden = false;
  } else {
    measuredContext.hidden = true;
  }

  setText("optimize-delta-ic50", `${formatSigned(comparison.delta_ic50_um)} µM`);
  setText("optimize-delta-pic50", formatSigned(comparison.delta_pic50));
  setText("optimize-fold", `${formatNumber(comparison.candidate_over_parent_ic50_fold, 2)}×`);
  setText("optimize-similarity", formatNumber(comparison.similarity?.morgan_tanimoto, 3));
  setText(
    "optimize-direction-definition",
    comparison.direction_definition
      || "Improved means higher predicted IC50 and lower predicted pIC50; all deltas are candidate minus parent.",
  );

  const sensitivity = element("low-sensitivity-notice");
  if (comparison.low_predicted_sensitivity) {
    sensitivity.textContent = "High structural similarity; the model predicts only a small hERG difference. "
      + "This is informational and highlights a known small-edit sensitivity limitation—it does not prove the prediction is wrong.";
    sensitivity.className = "callout warning";
    sensitivity.hidden = false;
  } else {
    sensitivity.hidden = true;
  }

  renderComparisonPropertyTable(data);
  const changes = comparison.functional_group_changes || {};
  appendTextList(element("functional-groups-added"), changes.added, "No named functional group added.");
  appendTextList(element("functional-groups-removed"), changes.removed, "No named functional group removed.");
  setText(
    "functional-change-description",
    changes.description || "Differences are based on the current RDKit named-fragment panel.",
  );

  const basicityList = element("optimize-basicity-list");
  basicityList.replaceChildren();
  appendDefinitionRow(
    basicityList,
    "Parent",
    parentProperties.basicity?.classification || "Not available",
  );
  appendDefinitionRow(
    basicityList,
    "Candidate",
    candidateProperties.basicity?.classification || "Not available",
  );
  appendDefinitionRow(
    basicityList,
    "Δ positive-ionizable centers",
    formatSigned(
      changes.basic_centers?.candidate_minus_parent
        ?? changes.basic_center_delta
        ?? data.calculated_properties?.deltas?.basic_centers?.candidate_minus_parent,
      0,
    ),
  );
  appendDefinitionRow(
    basicityList,
    "Δ formal charge",
    formatSigned(
      changes.formal_charge?.candidate_minus_parent
        ?? changes.formal_charge_delta
        ?? data.calculated_properties?.deltas?.formal_charge?.candidate_minus_parent,
      0,
    ),
  );
  appendDefinitionRow(basicityList, "Numerical pKa", "Not calculated");

  const interpretation = data.heuristic_interpretation || {};
  appendTextList(
    element("optimize-interpretation-list"),
    interpretation.statements,
    "No additional heuristic interpretation was returned.",
    "interpretation-list",
  );
  setText("optimize-claim-boundary", interpretation.claim_boundary || data.claim_boundary || "Research use only.");
  renderComparisonWarnings(comparison);
  renderMwSensitivity(
    data.experimental_mw_sensitivity || comparison.experimental_mw_sensitivity,
  );
  optimizeResults.hidden = false;
};

const generatedTransformationLabel = (candidate) => {
  const lineages = candidate.generation?.lineages || [];
  const labels = [...new Set(lineages.map((row) => row.transformation_name).filter(Boolean))];
  return labels.slice(0, 2).join("; ") || "Governed one-step edit";
};

const renderGeneratedCandidates = (data) => {
  currentComparison = data;
  manualOptimizeResult.hidden = true;
  generatedOptimizeResult.hidden = false;
  setText("optimize-results-kicker", "Experimental candidate enumeration");
  setText("optimize-results-title", "Ranked candidate edits");
  const directionBadge = element("optimize-direction");
  directionBadge.textContent = "experimental";
  directionBadge.className = "direction-badge unchanged";
  setText(
    "optimize-runtime",
    `Bounded ligand-only generation and ranking · ${formatNumber(data.runtime_seconds, 2)} seconds`,
  );

  const generation = data.generation || {};
  const summary = element("generation-summary");
  summary.replaceChildren();
  [
    ["Requested", formatNumber(generation.requested_candidate_count, 0)],
    ["Generated", formatNumber(generation.generated_candidate_count, 0)],
    ["Successfully evaluated", formatNumber(generation.successfully_evaluated_count, 0)],
    ["Valid unique before selection", formatNumber(generation.valid_unique_before_selection, 0)],
    ["Registry", generation.registry_version || "—"],
    ["RDKit", generation.rdkit_version || "—"],
    ["Status", String(generation.status || "—").replaceAll("_", " ")],
    ["Prediction mode", "Frozen ligand-only default"],
  ].forEach(([label, value]) => appendDefinitionRow(summary, label, value));
  setText("generation-boundary", generation.claim_boundary || data.claim_boundary);

  const candidates = Array.isArray(data.ranked_candidates) ? data.ranked_candidates : [];
  const table = element("generated-candidate-table");
  table.replaceChildren();
  candidates.forEach((candidate) => {
    const ranking = candidate.ranking || {};
    const model = candidate.model_output || {};
    const regression = model.regression || {};
    const applicability = model.applicability || {};
    const interval = regression.interval90_um || {};
    const values = [
      formatNumber(ranking.rank, 0),
      generatedTransformationLabel(candidate),
      `${formatNumber(regression.predicted_ic50_um)} µM`,
      formatSigned(candidate.comparison?.candidate_minus_parent_pic50),
      `${formatNumber(interval.lower, 2)}–${formatNumber(interval.upper, 2)} µM`,
      formatNumber(candidate.generation?.parent_morgan_tanimoto, 3),
      applicability.domain_label || "—",
      formatNumber(ranking.prioritization_score, 3),
    ];
    const row = document.createElement("tr");
    values.forEach((value, index) => {
      const cell = document.createElement(index === 0 ? "th" : "td");
      if (index === 0) cell.scope = "row";
      cell.textContent = value;
      if (index === 1) {
        const smiles = document.createElement("code");
        smiles.textContent = candidate.canonical_smiles;
        smiles.title = candidate.canonical_smiles;
        cell.appendChild(smiles);
      }
      row.appendChild(cell);
    });
    table.appendChild(row);
  });
  if (!candidates.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 8;
    cell.textContent = "No valid candidates were available for model evaluation.";
    row.appendChild(cell);
    table.appendChild(row);
  }

  const cards = element("generated-candidate-cards");
  cards.replaceChildren();
  candidates.slice(0, 6).forEach((candidate) => {
    const card = document.createElement("article");
    card.className = "generated-candidate-card";
    const heading = document.createElement("h4");
    heading.textContent = `Rank ${formatNumber(candidate.ranking?.rank, 0)} · ${generatedTransformationLabel(candidate)}`;
    const depiction = candidate.model_output?.structure_depiction?.data_uri;
    if (typeof depiction === "string" && /^data:image\/svg\+xml;base64,/i.test(depiction)) {
      const image = document.createElement("img");
      image.className = "structure-image";
      image.alt = `RDKit depiction of generated candidate rank ${candidate.ranking?.rank}`;
      image.src = depiction;
      card.appendChild(heading);
      card.appendChild(image);
    } else {
      card.appendChild(heading);
    }
    const smiles = document.createElement("code");
    smiles.textContent = candidate.canonical_smiles;
    const main = candidate.model_output?.regression || {};
    const metrics = document.createElement("p");
    metrics.textContent = `${formatNumber(main.predicted_ic50_um)} µM IC50 · ΔpIC50 ${formatSigned(candidate.comparison?.candidate_minus_parent_pic50)} · similarity ${formatNumber(candidate.generation?.parent_morgan_tanimoto, 3)}`;
    const applicability = document.createElement("p");
    applicability.textContent = `Applicability: ${candidate.model_output?.applicability?.domain_label || "not available"}. Score ${formatNumber(candidate.ranking?.prioritization_score, 3)} is prioritization, not confidence.`;
    card.append(smiles, metrics, applicability);
    cards.appendChild(card);
  });
  setText("generated-claim-boundary", data.claim_boundary);
  optimizeResults.hidden = false;
};

const csvCell = (value) => {
  let text = value === null || value === undefined ? "" : String(value);
  if (/^[=+\-@]/.test(text) && typeof value === "string") text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
};

const predictionCsv = (data) => {
  const receptor = data.receptor_aware?.status === "research_only_non_default"
    ? data.receptor_aware
    : null;
  const columns = {
    canonical_smiles: data.smiles,
    prediction_mode: data.prediction_mode,
    feature_mode: data.feature_mode?.selected || "atomwise",
    selected_quantitative_surface: data.selected_quantitative_result?.surface || "atomwise_default",
    selected_pic50: data.selected_quantitative_result?.pic50 ?? data.main_ic50.pic50,
    selected_ic50_um: data.selected_quantitative_result?.ic50_um ?? data.main_ic50.ic50_um,
    selected_tier: data.selected_quantitative_result?.tier ?? data.main_ic50.tier,
    functional_group_promotion_status: data.functional_group_research?.status || "not_selected",
    heavy_molecule_analysis_enabled: data.advanced_heavy_analysis?.enabled || false,
    total_runtime_seconds: data.runtime.total_seconds,
    classification_tier: data.classification.tier,
    probability_safe: data.classification.probabilities.Safe,
    probability_moderate: data.classification.probabilities.Moderate,
    probability_potent: data.classification.probabilities.Potent,
    regression_point_tier: data.main_ic50.tier,
    regression_residual_frequency_safe: data.main_ic50.tier_probabilities?.Safe,
    regression_residual_frequency_moderate: data.main_ic50.tier_probabilities?.Moderate,
    regression_residual_frequency_potent: data.main_ic50.tier_probabilities?.Potent,
    literature_pic50: data.main_ic50.pic50,
    literature_ic50_um: data.main_ic50.ic50_um,
    literature_interval90_lower_um: data.main_ic50.interval90_um.lower,
    literature_interval90_upper_um: data.main_ic50.interval90_um.upper,
    direct_ic10_um: data.direct_functional_curve.ic10_um,
    direct_ic30_um: data.direct_functional_curve.ic30_um,
    direct_ic50_um: data.direct_functional_curve.ic50_um,
    decision_confidence: data.main_ic50.decision_confidence.label,
    maximum_train_tanimoto: data.applicability.maximum_train_tanimoto,
    exact_training_overlap: data.applicability.exact_training_overlap,
    v14_preview_status: data.research_preview.status,
    v14_preview_classification_tier: data.research_preview.classification.tier,
    v14_preview_probability_safe: data.research_preview.classification.probabilities.Safe,
    v14_preview_probability_moderate: data.research_preview.classification.probabilities.Moderate,
    v14_preview_probability_potent: data.research_preview.classification.probabilities.Potent,
    v14_preview_pic50: data.research_preview.literature_ic50.pic50,
    v14_preview_ic50_um: data.research_preview.literature_ic50.ic50_um,
    receptor_prediction_promoted: data.research_preview.receptor_model.promoted,
    receptor_status: data.receptor_aware?.status || "not_requested",
    receptor_classification_tier: receptor?.classification?.tier ?? "",
    receptor_probability_safe: receptor?.classification?.probabilities?.Safe ?? "",
    receptor_probability_moderate: receptor?.classification?.probabilities?.Moderate ?? "",
    receptor_probability_potent: receptor?.classification?.probabilities?.Potent ?? "",
    receptor_pic50: receptor?.literature_ic50?.pic50 ?? "",
    receptor_ic50_um: receptor?.literature_ic50?.ic50_um ?? "",
    receptor_best_vina_affinity_kcal_mol: receptor?.docking?.best_affinity_kcal_mol ?? "",
    receptor_pose_count: receptor?.docking?.returned_pose_count ?? "",
    receptor_cache_hit: receptor?.runtime?.cache_hit ?? false,
  };
  return `${Object.keys(columns).map(csvCell).join(",")}\n${Object.values(columns).map(csvCell).join(",")}\n`;
};

const optimizationCsv = (data) => {
  if (data.analysis_mode === "experimental_bounded_candidate_generation") {
    const headers = [
      "rank", "parent_canonical_smiles", "candidate_id", "canonical_smiles", "transformation", "predicted_ic50_um",
      "predicted_pic50", "delta_pic50_candidate_minus_parent", "interval90_lower_um",
      "interval90_upper_um", "predicted_direction", "low_predicted_sensitivity", "regression_tier", "classifier_tier", "parent_tanimoto",
      "maximum_train_tanimoto", "applicability", "prioritization_score",
      "interval_width_penalty", "applicability_penalty", "parent_similarity_penalty",
      "property_change_penalty", "registry_version",
    ];
    const rows = (data.ranked_candidates || []).map((candidate) => {
      const model = candidate.model_output || {};
      const regression = model.regression || {};
      const ranking = candidate.ranking || {};
      const interval = regression.interval90_um || {};
      return [
        ranking.rank,
        data.parent?.canonical_smiles,
        candidate.candidate_id,
        candidate.canonical_smiles,
        generatedTransformationLabel(candidate),
        regression.predicted_ic50_um,
        regression.predicted_pic50,
        candidate.comparison?.candidate_minus_parent_pic50,
        interval.lower,
        interval.upper,
        candidate.comparison?.direction,
        candidate.comparison?.low_predicted_sensitivity,
        regression.threshold_tier,
        model.classifier?.predicted_tier,
        candidate.generation?.parent_morgan_tanimoto,
        model.applicability?.maximum_train_tanimoto,
        model.applicability?.domain_label,
        ranking.prioritization_score,
        ranking.interval_width_penalty,
        ranking.applicability_penalty,
        ranking.parent_similarity_penalty,
        ranking.property_change_penalty,
        data.generation?.registry_version,
      ];
    });
    return [headers, ...rows].map((row) => row.map(csvCell).join(",")).join("\n") + "\n";
  }
  const parent = data.model_outputs?.parent || {};
  const candidate = data.model_outputs?.candidate || {};
  const headers = [
    "parent_smiles", "candidate_smiles", "parent_predicted_ic50_um",
    "candidate_predicted_ic50_um", "delta_ic50_um", "parent_predicted_pic50",
    "candidate_predicted_pic50", "delta_pic50", "predicted_direction",
    "morgan_tanimoto", "known_parent_ic50_um",
  ];
  const row = [
    parent.smiles,
    candidate.smiles,
    parent.regression?.predicted_ic50_um,
    candidate.regression?.predicted_ic50_um,
    data.comparison?.delta_ic50_um,
    parent.regression?.predicted_pic50,
    candidate.regression?.predicted_pic50,
    data.comparison?.delta_pic50,
    data.comparison?.direction,
    data.comparison?.similarity?.morgan_tanimoto,
    data.measured_values?.parent_ic50_um,
  ];
  return `${headers.map(csvCell).join(",")}\n${row.map(csvCell).join(",")}\n`;
};

const download = (content, mimeType, filename) => {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
};

const timestamp = () => new Date().toISOString().replaceAll(":", "-").replace(".", "-");

const postJson = async (url, payload, signal) => {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    throw new Error(data?.error || "The analysis service returned an unexpected response.");
  }
  if (!data) throw new Error("The analysis service returned an unreadable response.");
  return data;
};

const runPrediction = async () => {
  const smiles = smilesInput.value.trim();
  const mode = selectedMode();
  const featureMode = selectedFeatureMode();
  const heavyMoleculeAnalysis = element("heavy-molecule-analysis").checked;
  const receptorIds = mode === "ligand_receptor" && capabilityState.multiReceptor
    ? selectedReceptorIds("predict")
    : [];
  currentResult = null;
  results.hidden = true;
  if (!smiles) {
    setStatus("Enter a SMILES string before running the analysis.", "error");
    smilesInput.focus();
    return;
  }

  const controller = new AbortController();
  const timeout = window.setTimeout(
    () => controller.abort(),
    mode === "ligand_receptor" ? (receptorIds.length > 1 ? 300000 : 180000) : 30000,
  );
  const progressUpdate = window.setTimeout(() => {
    setStatus(
      mode === "ligand_receptor"
        ? `AutoDock Vina is still running across ${receptorIds.length || 1} selected receptor state${(receptorIds.length || 1) === 1 ? "" : "s"}. Fresh runtime scales with receptor count and molecular flexibility.`
        : "Still calculating the ligand prediction.",
    );
  }, mode === "ligand_receptor" ? 7000 : 3000);
  setLoading(true, mode);
  setStatus(
    mode === "ligand_receptor"
      ? `Running fast ligand models and AutoDock Vina across ${receptorIds.length || 1} receptor state${(receptorIds.length || 1) === 1 ? "" : "s"}; only 8ZYO can drive the receptor-aware ML output.`
      : (featureMode === "functional_group"
        ? "Running the validated atom-wise model plus the V15 functional-group research comparison."
        : "Running the fast RDKit2D + Morgan ligand models."),
  );

  try {
    const payload = {
      smiles,
      mode,
      feature_mode: featureMode,
      heavy_molecule_analysis: heavyMoleculeAnalysis,
    };
    if (receptorIds.length) payload.receptor_ids = receptorIds;
    const data = await postJson(
      ACTIVE_TARGET.endpoints.predict,
      payload,
      controller.signal,
    );
    renderResult(data);
    setStatus(`Prediction complete in ${formatNumber(data.runtime.total_seconds, 2)} seconds.`, "success");
    results.focus({ preventScroll: true });
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    const message = error.name === "AbortError"
      ? "The prediction took too long. Please try again with one molecule."
      : error.message || "The prediction could not be completed. Please try again.";
    setStatus(message, "error");
  } finally {
    window.clearTimeout(timeout);
    window.clearTimeout(progressUpdate);
    setLoading(false, mode);
  }
};

const runProperties = async () => {
  const smiles = propertySmilesInput.value.trim();
  currentPropertyResult = null;
  propertyResults.hidden = true;
  if (!smiles) {
    setPanelStatus(propertyStatus, "Enter a SMILES string before calculating properties.", "error");
    propertySmilesInput.focus();
    return;
  }
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 30000);
  setActionLoading(propertyButton, true, "Calculate properties", "Calculating properties");
  propertyClearButton.disabled = true;
  propertyForm.setAttribute("aria-busy", "true");
  setPanelStatus(propertyStatus, "Calculating RDKit descriptors, structural motifs, and corpus context.");
  try {
    const data = await postJson(
      ACTIVE_TARGET.endpoints.properties,
      { smiles, mode: "properties" },
      controller.signal,
    );
    renderProperties(data);
    setPanelStatus(
      propertyStatus,
      `Property calculation complete in ${formatNumber(data.runtime_seconds, 3)} seconds.`,
      "success",
    );
    propertyResults.focus({ preventScroll: true });
    propertyResults.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    const message = error.name === "AbortError"
      ? "The property calculation took too long. Please check the structure and try again."
      : error.message || "The properties could not be calculated.";
    setPanelStatus(propertyStatus, message, "error");
  } finally {
    window.clearTimeout(timeout);
    setActionLoading(propertyButton, false, "Calculate properties", "Calculating properties");
    propertyClearButton.disabled = false;
    propertyForm.setAttribute("aria-busy", "false");
  }
};

const runComparison = async () => {
  const parentSmiles = parentSmilesInput.value.trim();
  const candidateSmiles = candidateSmilesInput.value.trim();
  const knownValueText = element("known-parent-ic50").value.trim();
  currentComparison = null;
  optimizeResults.hidden = true;
  if (!parentSmiles || !candidateSmiles) {
    setPanelStatus(optimizeStatus, "Enter both parent and edited-candidate SMILES.", "error");
    (parentSmiles ? candidateSmilesInput : parentSmilesInput).focus();
    return;
  }
  const knownParentIc50 = knownValueText ? Number(knownValueText) : null;
  if (knownValueText && (!Number.isFinite(knownParentIc50) || knownParentIc50 <= 0)) {
    setPanelStatus(optimizeStatus, "Known parent IC50 must be a positive value in µM.", "error");
    element("known-parent-ic50").focus();
    return;
  }
  const payload = { parent_smiles: parentSmiles, candidate_smiles: candidateSmiles };
  if (knownParentIc50 !== null) payload.known_parent_ic50_um = knownParentIc50;
  if (capabilityState.mwDeltaStressTest && element("mw-delta-stress-test").checked) {
    payload.mw_delta_stress_test = true;
  }
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 90000);
  setActionLoading(optimizeButton, true, "Compare structures", "Comparing structures");
  optimizeClearButton.disabled = true;
  optimizeForm.setAttribute("aria-busy", "true");
  setPanelStatus(
    optimizeStatus,
    "Running the same ligand model and property calculations for both structures.",
  );
  try {
    const data = await postJson(ACTIVE_TARGET.endpoints.compare, payload, controller.signal);
    renderComparison(data);
    setPanelStatus(
      optimizeStatus,
      `Comparison complete in ${formatNumber(data.runtime_seconds, 2)} seconds.`,
      "success",
    );
    optimizeResults.focus({ preventScroll: true });
    optimizeResults.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    const message = error.name === "AbortError"
      ? "The comparison took too long. Try the ligand-only Predict workflow for each structure."
      : error.message || "The structures could not be compared.";
    setPanelStatus(optimizeStatus, message, "error");
  } finally {
    window.clearTimeout(timeout);
    setActionLoading(optimizeButton, false, "Compare structures", "Comparing structures");
    optimizeClearButton.disabled = false;
    optimizeForm.setAttribute("aria-busy", "false");
  }
};

const runCandidateGeneration = async () => {
  const parentSmiles = parentSmilesInput.value.trim();
  const candidateCount = Number(element("generated-candidate-count").value);
  currentComparison = null;
  optimizeResults.hidden = true;
  if (!parentSmiles) {
    setPanelStatus(optimizeStatus, "Enter a parent SMILES before generating candidates.", "error");
    parentSmilesInput.focus();
    return;
  }
  if (!Number.isInteger(candidateCount) || candidateCount < 1 || candidateCount > 100) {
    setPanelStatus(optimizeStatus, "Candidate count must be a whole number from 1 to 100.", "error");
    element("generated-candidate-count").focus();
    return;
  }
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 300000);
  setActionLoading(optimizeButton, true, "Generate and rank edits", "Generating and ranking edits");
  optimizeClearButton.disabled = true;
  optimizeForm.setAttribute("aria-busy", "true");
  setPanelStatus(
    optimizeStatus,
    `Enumerating up to ${candidateCount} valid one-step edits, then running the frozen ligand model and evidence-aware ranking.`,
  );
  try {
    const data = await postJson(
      ACTIVE_TARGET.endpoints.generate,
      { parent_smiles: parentSmiles, max_candidates: candidateCount },
      controller.signal,
    );
    renderGeneratedCandidates(data);
    const evaluatedCount = Number(data.generation?.successfully_evaluated_count || 0);
    setPanelStatus(
      optimizeStatus,
      evaluatedCount
        ? `Evaluated ${evaluatedCount} candidates in ${formatNumber(data.runtime_seconds, 2)} seconds.`
        : `Generation completed in ${formatNumber(data.runtime_seconds, 2)} seconds, but no valid candidates were available for model evaluation.`,
      evaluatedCount ? "success" : "warning",
    );
    optimizeResults.focus({ preventScroll: true });
    optimizeResults.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    const message = error.name === "AbortError"
      ? "Candidate generation and evaluation exceeded five minutes. Try fewer candidates."
      : error.message || "Candidates could not be generated and evaluated.";
    setPanelStatus(optimizeStatus, message, "error");
  } finally {
    window.clearTimeout(timeout);
    setActionLoading(optimizeButton, false, "Generate and rank edits", "Generating and ranking edits");
    optimizeClearButton.disabled = false;
    optimizeForm.setAttribute("aria-busy", "false");
  }
};

const runOptimization = () => (
  selectedOptimizeMode() === "generate" ? runCandidateGeneration() : runComparison()
);

form.addEventListener("submit", (event) => {
  event.preventDefault();
  runPrediction();
});

form.addEventListener("change", (event) => {
  if (event.target.name === "mode") {
    setLoading(false, selectedMode());
    syncPredictReceptorControls();
  }
});

document.querySelectorAll("[data-receptor-context]").forEach((input) => {
  input.addEventListener("change", enforceReceptorSelection);
});

document.querySelectorAll("[data-workflow-target]").forEach((tab) => {
  tab.addEventListener("click", () => setWorkflow(tab.dataset.workflowTarget, { focus: true }));
  tab.addEventListener("keydown", (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const order = ["predict", "optimize", "properties"];
    const offset = event.key === "ArrowRight" ? 1 : -1;
    const index = (order.indexOf(activeWorkflow) + offset + order.length) % order.length;
    const next = order[index];
    setWorkflow(next, { focus: false });
    element(`workflow-tab-${next}`).focus();
  });
});

smilesInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    runPrediction();
  }
});

exampleSelect.addEventListener("change", () => {
  if (exampleSelect.value) {
    smilesInput.value = exampleSelect.value;
    const metadata = internalExampleMetadata.get(exampleSelect.value);
    if (metadata) {
      const measurement = metadata.measurement || {};
      const relation = measurement.relation === "=" ? "" : measurement.relation || "";
      exampleNote.textContent = `${metadata.id}: measured ${relation}${formatNumber(measurement.value_um, 3)} µM (${measurement.measurement_class}); ${formatNumber(metadata.molecular_weight_da, 1)} Da. Local private example—do not share through a public tunnel. ${metadata.selectionNote || ""}`;
      exampleNote.hidden = false;
    } else {
      exampleNote.textContent = "";
      exampleNote.hidden = true;
    }
    smilesInput.focus();
  } else {
    exampleNote.textContent = "";
    exampleNote.hidden = true;
  }
});

clearButton.addEventListener("click", () => {
  smilesInput.value = "";
  exampleSelect.value = "";
  exampleNote.textContent = "";
  exampleNote.hidden = true;
  results.hidden = true;
  element("receptor-result-panel").hidden = true;
  element("predict-multi-receptor-panel").hidden = true;
  currentResult = null;
  element("heavy-molecule-analysis").checked = false;
  form.querySelector('input[name="feature_mode"][value="atomwise"]').checked = true;
  setStatus("");
  smilesInput.focus();
});

propertyForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runProperties();
});

propertySmilesInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    runProperties();
  }
});

propertyClearButton.addEventListener("click", () => {
  propertySmilesInput.value = "";
  currentPropertyResult = null;
  propertyResults.hidden = true;
  setPanelStatus(propertyStatus, "");
  propertySmilesInput.focus();
});

const usePropertyStructureInPredict = () => {
  const smiles = propertySmilesInput.value.trim() || currentPropertyResult?.smiles || "";
  smilesInput.value = smiles;
  setWorkflow("predict", { focus: true, copyCurrent: false });
  smilesInput.focus();
};

element("property-to-predict").addEventListener("click", usePropertyStructureInPredict);
element("properties-result-to-predict").addEventListener("click", usePropertyStructureInPredict);

optimizeForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runOptimization();
});

optimizeForm.addEventListener("change", (event) => {
  if (event.target.name !== "optimize_mode") return;
  currentComparison = null;
  optimizeResults.hidden = true;
  setPanelStatus(optimizeStatus, "");
  syncOptimizeMode();
});

[parentSmilesInput, candidateSmilesInput].forEach((input) => {
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      runOptimization();
    }
  });
});

optimizeClearButton.addEventListener("click", () => {
  parentSmilesInput.value = "";
  candidateSmilesInput.value = "";
  element("known-parent-ic50").value = "";
  element("mw-delta-stress-test").checked = false;
  currentComparison = null;
  optimizeResults.hidden = true;
  manualOptimizeResult.hidden = false;
  generatedOptimizeResult.hidden = true;
  element("mw-sensitivity-panel").hidden = true;
  setPanelStatus(optimizeStatus, "");
  parentSmilesInput.focus();
});

element("copy-smiles").addEventListener("click", async () => {
  if (!currentResult) return;
  try {
    await navigator.clipboard.writeText(currentResult.smiles);
    setStatus("Canonical SMILES copied.", "success");
  } catch {
    setStatus("The browser could not copy automatically. Select the canonical SMILES and copy it manually.", "error");
  }
});

element("download-csv").addEventListener("click", () => {
  if (!currentResult) return;
  download(predictionCsv(currentResult), "text/csv;charset=utf-8", `herg-prediction-${timestamp()}.csv`);
});

element("download-json").addEventListener("click", () => {
  if (!currentResult) return;
  download(JSON.stringify(currentResult, null, 2), "application/json", `herg-prediction-${timestamp()}.json`);
});

element("optimize-download-json").addEventListener("click", () => {
  if (!currentComparison) return;
  const generated = currentComparison.analysis_mode === "experimental_bounded_candidate_generation";
  download(
    JSON.stringify(currentComparison, null, 2),
    "application/json",
    `${generated ? "herg-generated-candidates" : "herg-structural-edit"}-${timestamp()}.json`,
  );
});

element("optimize-download-csv").addEventListener("click", () => {
  if (!currentComparison) return;
  const generated = currentComparison.analysis_mode === "experimental_bounded_candidate_generation";
  download(
    optimizationCsv(currentComparison),
    "text/csv;charset=utf-8",
    `${generated ? "herg-generated-candidates" : "herg-structural-edit"}-${timestamp()}.csv`,
  );
});

setWorkflow("predict", { copyCurrent: false, focus: false });
initializeCapabilities();
initializeInternalExamples();
