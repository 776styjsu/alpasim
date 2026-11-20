# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from alpasim_wizard.__main__ import run_wizard
from alpasim_wizard.schema import RunMethod
from alpasim_wizard.wizard import AlpasimWizard


@pytest.mark.parametrize(
    ("run_method", "selected_constructor"),
    [
        (RunMethod.SLURM, "SlurmDeployment"),
        (RunMethod.SLURM_ENROOT, "SlurmEnrootDeployment"),
        (RunMethod.APPTAINER, "ApptainerDeployment"),
        (RunMethod.DOCKER_COMPOSE, "DockerComposeDeployment"),
        (RunMethod.NONE, "DockerComposeDeployment"),
    ],
)
def test_wizard_selects_deployment_from_run_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_method: RunMethod,
    selected_constructor: str,
) -> None:
    context = SimpleNamespace(
        cfg=SimpleNamespace(
            wizard=SimpleNamespace(
                driver_code_hash=None,
                log_dir=str(tmp_path),
                run_method=run_method,
                slurm_job_id=123,
            )
        )
    )
    constructors = {
        name: MagicMock()
        for name in (
            "DockerComposeDeployment",
            "SlurmDeployment",
            "SlurmEnrootDeployment",
            "ApptainerDeployment",
        )
    }
    config_manager = MagicMock()
    for name, constructor in constructors.items():
        monkeypatch.setattr(f"alpasim_wizard.wizard.{name}", constructor)
    monkeypatch.setattr(
        "alpasim_wizard.wizard.ConfigurationManager",
        MagicMock(return_value=config_manager),
    )

    AlpasimWizard(context).cast()

    selected_deployment = constructors[selected_constructor].return_value
    constructors[selected_constructor].assert_called_once_with(context)
    for name, constructor in constructors.items():
        if name == selected_constructor and run_method != RunMethod.NONE:
            constructor.return_value.deploy_all_services.assert_called_once_with()
        else:
            constructor.return_value.deploy_all_services.assert_not_called()
        if name not in (selected_constructor, "DockerComposeDeployment"):
            constructor.assert_not_called()
    config_manager.generate_all.assert_called_once_with(
        selected_deployment.container_set,
        context,
    )
    config_manager.cleanup_central_file_sd.assert_called_once_with()


def test_wizard_rejects_direct_enroot_without_slurm_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(
        cfg=SimpleNamespace(
            wizard=SimpleNamespace(
                driver_code_hash=None,
                log_dir=str(tmp_path),
                run_method=RunMethod.SLURM_ENROOT,
                slurm_job_id=0,
            )
        )
    )
    monkeypatch.setattr(
        "alpasim_wizard.wizard.DockerComposeDeployment",
        MagicMock(),
    )

    with pytest.raises(ValueError, match="active Slurm allocation"):
        AlpasimWizard(context).cast()


def test_run_wizard_starts_mps_before_scene_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    @contextmanager
    def fake_mps_daemon(wizard_cfg):
        events.append(("mps_start", wizard_cfg.mps_pipe_dir))
        yield
        events.append("mps_stop")

    monkeypatch.setattr("alpasim_wizard.__main__.mps_daemon", fake_mps_daemon)
    monkeypatch.setattr("alpasim_wizard.__main__.validate_config", lambda cfg: None)
    monkeypatch.setattr(
        "alpasim_wizard.__main__.update_scene_config",
        lambda cfg: events.append("scenes"),
    )
    monkeypatch.setattr(
        "alpasim_wizard.__main__.AlpasimWizard.create",
        lambda cfg: SimpleNamespace(cast=lambda: events.append("cast")),
    )
    cfg = SimpleNamespace(
        wizard=SimpleNamespace(
            log_dir=str(tmp_path),
            slurm_job_id=123,
            enable_mps=True,
            mps_pipe_dir=None,
        )
    )

    run_wizard(cfg)

    assert events == [
        ("mps_start", "/tmp/nvidia-mps-123"),
        "scenes",
        "cast",
        "mps_stop",
    ]
