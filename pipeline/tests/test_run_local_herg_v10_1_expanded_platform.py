from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_local_herg_v10_1_expanded_platform.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("herg_v10_1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_observed_dose_crossing_is_interpolated_in_log_concentration() -> None:
    module = _module()
    log_concentrations = np.log10(np.array([0.1, 1.0, 10.0]))
    crossing = module._interpolated_crossing(
        log_concentrations,
        np.array([0.0, 10.0, 50.0]),
        30.0,
    )
    assert math.isclose(crossing, math.sqrt(10.0))


def test_binary_operating_point_respects_approximately_one_percent_fpr() -> None:
    module = _module()
    negatives = np.zeros(1000, dtype=int)
    positives = np.ones(20, dtype=int)
    y = np.concatenate([negatives, positives])
    probabilities = np.concatenate(
        [np.linspace(0.0, 0.9, len(negatives)), np.linspace(0.7, 1.0, len(positives))]
    )
    metrics = module._binary_metrics(y, probabilities)
    predictions = probabilities >= metrics["threshold_at_approximately_1pct_fpr"]
    assert int(np.sum(predictions & (y == 0))) <= 10
    assert 0.0 <= metrics["precision_at_approximately_1pct_fpr"] <= 1.0


def test_feature_row_contains_frozen_ligand_features() -> None:
    module = _module()
    columns = ["rdkit2d__MolWt", "morgan__0000", "morgan__2047", "maccs__001"]
    frame = module._feature_row("CCO", columns)
    assert frame.columns.tolist() == columns
    assert np.isfinite(frame["rdkit2d__MolWt"]).all()
    assert set(frame[["morgan__0000", "morgan__2047", "maccs__001"]].to_numpy().ravel()) <= {
        0,
        1,
    }


def test_same_assay_threshold_projection_enforces_physical_order() -> None:
    module = _module()
    projected, adjusted = module._coherent_threshold_concentrations([6.0, 20.0, 12.0])
    assert adjusted is True
    assert projected[0] <= projected[1] <= projected[2]

    already_ordered, adjusted = module._coherent_threshold_concentrations([6.0, 12.0, 20.0])
    assert adjusted is False
    assert np.allclose(already_ordered, [6.0, 12.0, 20.0])


def test_parser_requires_explicit_model_or_output_root() -> None:
    module = _module()
    validate = module._parser().parse_args(["validate", "--output-root", "/tmp/v10_1"])
    predict = module._parser().parse_args(["predict", "--model-root", "/tmp/v10_1", "--smiles", "CCO"])
    assert validate.command == "validate"
    assert predict.command == "predict"
