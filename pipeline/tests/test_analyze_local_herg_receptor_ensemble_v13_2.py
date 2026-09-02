from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts/analyze_local_herg_receptor_ensemble_v13_2.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("herg_v13_2_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_grouped_bootstrap_detects_uniform_improvement() -> None:
    frame = pd.DataFrame({"scaffold_group_id": ["a", "a", "b"]})
    observed = np.array([1.0, 2.0, 3.0])
    baseline = observed + 1.0
    candidate = observed.copy()
    result = MODULE._bootstrap_delta(frame, baseline, candidate, observed, 100)
    assert result["delta_mae_candidate_minus_baseline"] == -1.0
    assert result["probability_candidate_better"] == 1.0


def test_feature_contract_separates_posthoc_surface() -> None:
    columns = {
        "dock__lowK_minus_highK",
        "dock__e4031_conditioned_minus_apo",
        "dock__core_sd_affinity",
        "dock__core_range_affinity",
        "dock__ensemble_sd_affinity",
        "dock__ensemble_range_affinity",
        "dock__astemizole_minus_apo",
        "dock__pimozide_minus_apo",
        "dock__astemizole_minus_e4031",
        "dock__pimozide_minus_e4031",
        "dock__pimozide_minus_astemizole",
    }
    for state in MODULE.v13.RECEPTOR_IDS:
        columns.update(
            {
                f"dock__{state}__affinity",
                f"dock__{state}__ligand_efficiency",
                f"dock__{state}__pose_count_within_1kcal",
                f"dock__{state}__affinity_range",
            }
        )
        for residue in MODULE.v13.CONTACT_RESIDUES:
            columns.update(
                {
                    f"dock__{state}__contact_{residue}_distance",
                    f"dock__{state}__contact_{residue}_count",
                }
            )
    frame = pd.DataFrame(columns=sorted(columns))
    feature_sets = MODULE._feature_sets(frame)
    assert feature_sets["posthoc_astemizole_f649_count"] == [
        "dock__8ZYO__contact_649_count"
    ]
    assert not any(name.startswith("posthoc_") for name in feature_sets if name != "posthoc_astemizole_f649_count")


def test_parser_defaults_to_all_strict_replicates() -> None:
    args = MODULE._parser().parse_args([])
    assert args.bootstrap_replicates == 10_000
    assert args.workers == 6
