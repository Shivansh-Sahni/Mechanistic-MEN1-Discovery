from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).parents[1] / "scripts/run_local_herg_endpoint_receptor_campaign_v12.py"
    spec = importlib.util.spec_from_file_location("v12_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_projection_enforces_endpoint_order() -> None:
    module = _module()
    raw = np.array([4.2, 5.1, 4.8])
    projected = module._project_nonincreasing(raw)
    assert projected[0] >= projected[1] >= projected[2]
    assert np.isclose(projected[0], projected[1])


def test_projection_preserves_coherent_values() -> None:
    module = _module()
    raw = np.array([6.0, 5.5, 4.5])
    assert np.allclose(module._project_nonincreasing(raw), raw)


def test_receptor_contract_is_fixed() -> None:
    module = _module()
    assert module.RECEPTOR_IDS == ("8ZYN", "8ZYP", "9CHP", "9CHQ", "8ZYO", "8ZYQ")
    assert module.CORE_IDS == ("8ZYN", "8ZYP", "9CHP", "9CHQ")
