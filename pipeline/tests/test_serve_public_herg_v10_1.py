from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).parents[1] / "scripts" / "serve_public_herg_v10_1.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("serve_public_herg_v10_1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_authorization_value_is_basic_auth() -> None:
    module = _module()
    username = "unit-user"
    password = "unit-pass"
    value = module._authorization_value(username, password)
    scheme, encoded = value.split(" ", 1)
    assert scheme == "Basic"
    assert base64.b64decode(encoded).decode() == f"{username}:{password}"


def test_proxy_exposes_only_demo_routes() -> None:
    module = _module()
    assert module.ALLOWED_PATHS == {
        "/",
        "/index.html",
        "/api/model-info",
        "/api/health",
        "/api/predict",
    }
