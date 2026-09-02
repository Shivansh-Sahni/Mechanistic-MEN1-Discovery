#!/usr/bin/env python3
"""Create the publication-ready V11--V13 hERG decision summary figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"V13.7": "#2F6B8A", "V13.8": "#D07A3A"}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forest_data(root: Path) -> list[dict[str, Any]]:
    v135 = _json(root / "herg_receptor_safety_gated_validation_v13_5/analysis_report.json")
    v136 = _json(root / "herg_receptor_hybrid_validation_v13_6/analysis_report.json")
    v137 = _json(root / "herg_external_validation_v13_7/analysis_report.json")
    v138 = _json(root / "herg_temporal_confirmation_v13_8/analysis_report.json")
    sources = [
        (
            "V13.5 internal",
            v135["score"]["safety_gated_frozen_f649"]["vs_baseline_bootstrap"],
            True,
        ),
        (
            "V13.6 internal",
            v136["score"]["frozen_hybrid"]["vs_lgbm_bootstrap"],
            True,
        ),
        (
            "V13.7 external",
            v137["ternary_classification"]["receptor_project_exact_and_v9_scaffold_novel"][
                "paired_scaffold_cluster_bootstrap"
            ],
            False,
        ),
        (
            "V13.8 external",
            v138["receptor_confirmation"]["paired_scaffold_cluster_bootstrap"],
            False,
        ),
    ]
    return [
        {
            "label": label,
            "delta": value["delta_balanced_accuracy_candidate_minus_baseline"],
            "ci": value["ci95"],
            "internal": internal,
        }
        for label, value, internal in sources
    ]


def _style_axis(axis: plt.Axes, letter: str, title: str) -> None:
    axis.set_title(f"{letter}  {title}", loc="left", fontsize=11, fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)


def make_figure(root: Path, output: Path) -> dict[str, Any]:
    interval = pd.read_parquet(
        root / "herg_external_uncertainty_v13_11/external_interval_predictions.parquet"
    )
    interval = interval.loc[interval.method.eq("global") & interval.nominal_coverage.eq(0.9)].copy()
    uncertainty = _json(root / "herg_external_uncertainty_v13_11/analysis_report.json")
    heterogeneity = _json(root / "herg_receptor_heterogeneity_v13_12/analysis_report.json")
    boundary = _json(root / "herg_external_boundary_sensitivity_v13_14/analysis_report.json")
    campaign_names = {
        "v13.7_ev2_project_exact_and_v9_scaffold_novel": "V13.7",
        "v13.8_post2021_v9_scaffold_novel": "V13.8",
    }

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    scatter = axes[0, 0]
    for campaign, part in interval.groupby("campaign"):
        label = campaign_names[campaign]
        mae = np.mean(np.abs(part.predicted_pic50 - part.observed_pic50))
        scatter.scatter(
            part.observed_pic50,
            part.predicted_pic50,
            s=20,
            alpha=0.67,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.35,
            label=f"{label}: n={len(part)}, MAE={mae:.3f}",
        )
    limits = [
        float(min(interval.observed_pic50.min(), interval.predicted_pic50.min()) - 0.2),
        float(max(interval.observed_pic50.max(), interval.predicted_pic50.max()) + 0.2),
    ]
    scatter.plot(limits, limits, linestyle="--", color="#555555", linewidth=1)
    scatter.set(xlim=limits, ylim=limits, xlabel="Observed pIC50", ylabel="Frozen V9 predicted pIC50")
    scatter.legend(frameon=False, fontsize=8, loc="upper left")
    sensitivity = boundary["metrics"]["posthoc_non_extreme_sensitivity"]
    scatter.text(
        0.02,
        0.02,
        f"V13.7 post-hoc non-extreme sensitivity: n=106, MAE={sensitivity['v9_mixed_ic50']['mae']:.3f}\n"
        "Four publisher-exact extremes remain in the primary metric.",
        transform=scatter.transAxes,
        fontsize=7.5,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )
    _style_axis(scatter, "A", "Strict external regression")

    forest = axes[0, 1]
    entries = _forest_data(root)
    positions = np.arange(len(entries))[::-1]
    for position, entry in zip(positions, entries, strict=True):
        color = "#507D5A" if entry["internal"] else "#7E5686"
        forest.errorbar(
            entry["delta"],
            position,
            xerr=[[entry["delta"] - entry["ci"][0]], [entry["ci"][1] - entry["delta"]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6,
        )
    forest.axvline(0, color="#555555", linewidth=1, linestyle="--")
    forest.set_yticks(positions, [entry["label"] for entry in entries])
    forest.set_xlabel("Hybrid − ligand router balanced accuracy")
    forest.text(
        0.02,
        -0.21,
        "Internal gains did not generalize; V13.7 also failed the safety gate.",
        transform=forest.transAxes,
        fontsize=8,
    )
    _style_axis(forest, "B", "Frozen receptor evidence sequence")

    coverage_axis = axes[1, 0]
    coverage_rows = [row for row in uncertainty["external_coverage"] if row["method"] == "global"]
    x = np.arange(3)
    width = 0.34
    nominal = [0.8, 0.9, 0.95]
    for offset, campaign in zip((-width / 2, width / 2), campaign_names, strict=True):
        rows = sorted(
            [row for row in coverage_rows if row["campaign"] == campaign],
            key=lambda row: row["nominal_coverage"],
        )
        label = campaign_names[campaign]
        values = [row["empirical_coverage"] for row in rows]
        lower = [value - row["wilson_ci95"][0] for value, row in zip(values, rows, strict=True)]
        upper = [row["wilson_ci95"][1] - value for value, row in zip(values, rows, strict=True)]
        coverage_axis.bar(x + offset, values, width, color=COLORS[label], alpha=0.86, label=label)
        coverage_axis.errorbar(
            x + offset,
            values,
            yerr=[lower, upper],
            fmt="none",
            ecolor="#333333",
            capsize=2,
            linewidth=0.8,
        )
    coverage_axis.scatter(x, nominal, marker="_", s=350, linewidth=2, color="black", label="Nominal")
    coverage_axis.set_xticks(x, ["80%", "90%", "95%"])
    coverage_axis.set_ylim(0.65, 1.01)
    coverage_axis.set_ylabel("External empirical coverage")
    coverage_axis.set_xlabel("Frozen interval level")
    coverage_axis.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    _style_axis(coverage_axis, "C", "OOF-calibrated V9 uncertainty")

    yield_axis = axes[1, 1]
    campaign_summaries = heterogeneity["campaigns"]
    labels = ["V13.7", "V13.8"]
    keys = ["V13.7_2025", "V13.8_2026"]
    corrected = [campaign_summaries[key]["decision_changes"]["corrected_count"] for key in keys]
    degraded = [campaign_summaries[key]["decision_changes"]["degraded_count"] for key in keys]
    other = [campaign_summaries[key]["decision_changes"]["wrong_to_different_wrong_count"] for key in keys]
    x = np.arange(2)
    yield_axis.bar(x, corrected, color="#3E8E62", label="Corrected")
    yield_axis.bar(x, degraded, bottom=corrected, color="#B84D4D", label="Degraded")
    yield_axis.bar(
        x,
        other,
        bottom=np.add(corrected, degraded),
        color="#999999",
        label="Wrong→other wrong",
    )
    for index, key in enumerate(keys):
        rate = campaign_summaries[key]["decision_changes"]["corrected_among_changed_rate"]
        total = campaign_summaries[key]["decision_changes"]["changed_count"]
        yield_axis.text(index, total + 0.7, f"{rate:.1%} corrected", ha="center", fontsize=8)
    interaction = heterogeneity["interaction_bootstrap"]["scaffold_cluster"]["balanced_accuracy"]
    yield_axis.text(
        0.02,
        0.95,
        f"BA interaction {interaction['observed_interaction']:+.3f}\n"
        f"95% CI [{interaction['ci95'][0]:+.3f}, {interaction['ci95'][1]:+.3f}]",
        transform=yield_axis.transAxes,
        va="top",
        fontsize=8,
    )
    yield_axis.set_xticks(x, labels)
    yield_axis.set_ylabel("Router decisions changed by hybrid")
    yield_axis.set_ylim(0, max(np.add(np.add(corrected, degraded), other)) + 6)
    yield_axis.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.45, 1.0))
    _style_axis(yield_axis, "D", "Why the external result did not replicate")

    figure.suptitle(
        "hERG V11–V13: ligand regression generalizes; receptor classification remains experimental",
        fontsize=14,
        fontweight="bold",
    )
    output.mkdir(parents=True, exist_ok=True)
    png = output / "HERG_V11_V13_DECISION_FIGURE.png"
    pdf = output / "HERG_V11_V13_DECISION_FIGURE.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    manifest = {
        "png": str(png),
        "png_sha256": _sha(png),
        "pdf": str(pdf),
        "pdf_sha256": _sha(pdf),
        "source_report_sha256": {
            "v13_11": _sha(root / "herg_external_uncertainty_v13_11/analysis_report.json"),
            "v13_12": _sha(root / "herg_receptor_heterogeneity_v13_12/analysis_report.json"),
            "v13_14": _sha(root / "herg_external_boundary_sensitivity_v13_14/analysis_report.json"),
        },
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("research/local_runs"))
    parser.add_argument("--output", type=Path, default=Path("research/local_runs/herg_v11_v13_figures"))
    args = parser.parse_args()
    print(json.dumps(make_figure(args.root, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
