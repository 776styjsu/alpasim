# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "run_tutorial_alpamayo_cot_analysis.py"
)
SPEC = importlib.util.spec_from_file_location("tutorial_cot_runner", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
API_KEY_ENV = MODULE.API_KEY_ENV
resolve_api_key = MODULE.resolve_api_key
run_single_asl = MODULE.run_single_asl


def test_resolve_api_key_expands_env_alias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.setenv("TEST_PROVIDER_KEY", "provider-secret")
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"{API_KEY_ENV}=${{TEST_PROVIDER_KEY}}\n",
        encoding="utf-8",
    )

    assert resolve_api_key(None, env_file, "uva") == "provider-secret"


def test_resolve_api_key_precedence(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"{API_KEY_ENV}=file-secret\n", encoding="utf-8")
    monkeypatch.setenv(API_KEY_ENV, "process-secret")

    assert resolve_api_key(None, env_file, "uva") == "process-secret"
    assert resolve_api_key("cli-secret", env_file, "uva") == "cli-secret"


def test_run_forwards_key_in_environment_not_command_line(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return None

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    args = Namespace(
        module="cot_analysis",
        every_nth=10,
        provider="uva",
        delay=0.0,
        camera_id="front",
        log_level="INFO",
        model=None,
        retry_errors=False,
        no_images=True,
        api_key="provider-secret",
    )

    run_single_asl(args, tmp_path / "input.asl", tmp_path / "output.json")

    assert "provider-secret" not in captured["command"]
    assert captured["env"]["UVA_RC_GENAI_KEY"] == "provider-secret"
