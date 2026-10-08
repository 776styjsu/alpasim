# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import pytest
from alpasim_wizard.deployment import apptainer


@pytest.mark.parametrize("dry_run", [False, True])
def test_apptainer_starts_services_in_background_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    context = SimpleNamespace(
        cfg=SimpleNamespace(wizard=SimpleNamespace(log_dir=tmp_path, dry_run=dry_run))
    )
    service = SimpleNamespace(command="service")
    runtime = SimpleNamespace(command="runtime")
    containers = SimpleNamespace(sim=[service], runtime=runtime)
    monkeypatch.setattr(
        apptainer, "build_container_set", MagicMock(return_value=containers)
    )
    background = create_autospec(apptainer.dispatch_background)
    blocking = create_autospec(apptainer.dispatch_command)
    monkeypatch.setattr(apptainer, "dispatch_background", background)
    monkeypatch.setattr(apptainer, "dispatch_command", blocking)
    deployment = apptainer.ApptainerDeployment(context)
    monkeypatch.setattr(deployment, "_to_apptainer_exec", lambda c: c.command)
    wait = MagicMock()
    monkeypatch.setattr(deployment, "_wait_for_containers", wait)
    calls = MagicMock()
    calls.attach_mock(background, "background")
    calls.attach_mock(wait, "wait")
    calls.attach_mock(blocking, "blocking")

    deployment.deploy_simulation()

    background.assert_called_once_with("service", log_dir=tmp_path, dry_run=dry_run)
    blocking.assert_called_once_with("runtime", log_dir=tmp_path, dry_run=dry_run)
    if dry_run:
        wait.assert_not_called()
        assert [call[0] for call in calls.mock_calls] == ["background", "blocking"]
    else:
        wait.assert_called_once_with([service])
        assert [call[0] for call in calls.mock_calls] == [
            "background",
            "wait",
            "blocking",
        ]
