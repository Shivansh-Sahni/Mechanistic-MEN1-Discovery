from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import make_local_herg_v11_v13_decision_figure as module  # noqa: E402


def test_forest_data_follows_internal_then_external_evidence() -> None:
    values = module._forest_data(Path("research/local_runs"))  # noqa: SLF001
    assert [row["label"] for row in values] == [
        "V13.5 internal",
        "V13.6 internal",
        "V13.7 external",
        "V13.8 external",
    ]
    assert values[2]["delta"] > values[3]["delta"]


def test_figure_generation_writes_nonempty_png_pdf_and_manifest(tmp_path: Path) -> None:
    result = module.make_figure(Path("research/local_runs"), tmp_path)
    assert Path(result["png"]).stat().st_size > 100_000
    assert Path(result["pdf"]).stat().st_size > 10_000
    assert (tmp_path / "figure_manifest.json").is_file()
