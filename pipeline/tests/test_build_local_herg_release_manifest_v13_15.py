from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_local_herg_release_manifest_v13_15 as module  # noqa: E402


def test_release_validation_binds_blank_panel_protocol_and_exact_replay() -> None:
    result = module._validate(Path.cwd())  # noqa: SLF001
    assert result["exact_replay_rows"] == 324
    assert result["prospective_panel_rows"] == 24
    assert result["prospective_disagreements"] == 13
    assert result["outcomes_present"] is False


def test_build_writes_self_hashed_release_and_readme(tmp_path: Path) -> None:
    result = module.build(Path.cwd(), tmp_path)
    body = dict(result)
    digest = body.pop("release_sha256")
    assert hashlib.sha256(module._canonical(body)).hexdigest() == digest  # noqa: SLF001
    assert result["status"] == "complete_research_release"
    assert len(result["artifacts"]) == len(module.ARTIFACTS)
    assert all(row["bytes"] > 0 for row in result["artifacts"])
    assert (tmp_path / "README.md").is_file()
    parsed = json.loads((tmp_path / "release_manifest.json").read_text())
    assert parsed["release_sha256"] == digest
