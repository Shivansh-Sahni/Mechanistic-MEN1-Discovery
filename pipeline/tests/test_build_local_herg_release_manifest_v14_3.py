from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/build_local_herg_release_manifest_v14_3.py"
RELEASE = Path(__file__).parents[2] / "research/local_runs/herg_v14_release"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("herg_v14_3", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scientific_contract_keeps_receptor_models_nonpromoted() -> None:
    module = _module()
    v14, v141, v142 = module._scientific_contract()
    assert v14["promotion_decision"]["ligand_recalibration_research_preview_supported"] is True
    assert v14["promotion_decision"]["general_receptor_prediction_promotion_supported"] is False
    assert v141["decision"]["general_receptor_prediction_promotion_supported"] is False
    assert v142["decision"]["website_integration_supported"] is False


def test_release_artifact_roles_are_unique_and_repository_relative() -> None:
    module = _module()
    v14, v141, _v142 = module._scientific_contract()
    artifacts = module._artifacts(RELEASE, v14, v141)
    roles = [item["role"] for item in artifacts]
    assert len(roles) == len(set(roles))
    assert all(not Path(item["path"]).is_absolute() for item in artifacts)
    assert {"website_backend", "v14_analysis", "v14_1_analysis", "v14_2_analysis"} <= set(
        roles
    )


def test_historical_release_manifest_explicitly_detects_post_release_drift() -> None:
    module = _module()
    manifest = module._read_json(RELEASE / "release_manifest.json", "release_sha256")
    drifted_roles = {
        binding["role"]
        for binding in manifest["artifacts"]
        if not (module.REPO_ROOT / binding["path"]).is_file()
        or (module.REPO_ROOT / binding["path"]).stat().st_size != int(binding["bytes"])
        or module._sha(module.REPO_ROOT / binding["path"]) != binding["sha256"]
    }

    # V14.3 is an immutable historical seal over website files that have since
    # advanced.  Passing verification now would silently reseal history.
    assert {"website_backend", "website_html", "website_javascript", "website_styles"} <= (
        drifted_roles
    )
    with pytest.raises(module.ReleaseError, match="release artifact .*changed"):
        module._verify(RELEASE)
