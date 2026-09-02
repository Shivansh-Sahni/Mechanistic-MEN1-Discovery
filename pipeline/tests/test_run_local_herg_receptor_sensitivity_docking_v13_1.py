from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/run_local_herg_receptor_sensitivity_docking_v13_1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("herg_v13_1_sensitivity_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_sensitivity_states_are_withheld_holo_structures() -> None:
    assert MODULE.SENSITIVITY_STATES == ("8ZYO", "8ZYQ")
    assert set(MODULE.SENSITIVITY_STATES).issubset(MODULE.v13.HOLO_SPECS)


def test_sensitivity_defaults_match_primary_docking_protocol() -> None:
    args = MODULE._parser().parse_args([])
    assert args.exhaustiveness == 8
    assert args.modes == 9
    assert args.cpu == 6
