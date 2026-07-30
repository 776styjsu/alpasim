# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import shlex
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import pytest
from alpasim_wizard.context import TelemetryPorts, WizardContext
from alpasim_wizard.deployment.apptainer import ApptainerDeployment
from alpasim_wizard.deployment.dispatcher import OsDispatchError
from alpasim_wizard.schema import DebugFlags, RunMode, WizardApptainerConfig
from alpasim_wizard.services import Address, VolumeMount
from alpasim_wizard.utils import (
    image_to_apptainer_basename,
    resolve_apptainer_image,
)


def _context(
    tmp_path: Path,
    *,
    dry_run: bool = False,
    num_gpus: int = 0,
    apptainer: WizardApptainerConfig | None = None,
) -> WizardContext:
    cfg = SimpleNamespace(
        wizard=SimpleNamespace(
            log_dir=str(tmp_path),
            dry_run=dry_run,
            timeout=1,
            nr_retries=1,
            run_mode=RunMode.ONESHOT,
            slurm_job_id=None,
            apptainer=apptainer or WizardApptainerConfig(),
            debug_flags=DebugFlags(use_localhost=False),
        )
    )
    return WizardContext(
        cfg=cfg,
        port_assigner=iter(()),
        telemetry_ports=TelemetryPorts(
            workers=(),
            prometheus=6100,
            node_exporter=6101,
            process_exporter=6102,
            dcgm_exporter=6103,
        ),
        artifact_list=[],
        num_gpus=num_gpus,
    )


def _deployment(
    tmp_path: Path,
    *,
    dry_run: bool = False,
    num_gpus: int = 0,
    apptainer: WizardApptainerConfig | None = None,
) -> ApptainerDeployment:
    deployment = ApptainerDeployment.__new__(ApptainerDeployment)
    deployment.context = _context(
        tmp_path, dry_run=dry_run, num_gpus=num_gpus, apptainer=apptainer
    )
    deployment._background = []
    return deployment


def _container(
    name: str = "driver",
    *,
    gpu: int | None = None,
    image: str = "alpasim-base:0.1.0",
    external_image: bool = False,
    volumes: Iterable[str] = (),
    environments: Iterable[str] = (),
    workdir: str | None = None,
    command: str = "echo ok",
    ports: Iterable[int] = (),
) -> Any:
    addresses = [Address(host="0.0.0.0", port=port) for port in ports]
    return SimpleNamespace(
        uuid=f"{name}-0",
        name=name,
        service_config=SimpleNamespace(image=image, external_image=external_image),
        gpu=gpu,
        environments=list(environments),
        volumes=[VolumeMount.from_str(volume) for volume in volumes],
        workdir=workdir,
        command=command,
        get_all_addresses=lambda: addresses,
    )


class FakeProcess:
    """Stand-in for a launched container process."""

    def __init__(self, pid: int = 4242, returncode: int = 0) -> None:
        self.pid = pid
        self.returncode = returncode
        self._running = True

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self._running = False
        return self.returncode

    def poll(self) -> int | None:
        return None if self._running else self.returncode


def _bash_payload(command: str) -> str:
    """Extract the script an `apptainer exec ... bash -c <script>` runs."""
    parts = shlex.split(command)
    return parts[parts.index("-c") + 1]


def _pwd_arg(command: str) -> str:
    """Extract the directory `apptainer exec --pwd <dir>` starts in."""
    parts = shlex.split(command)
    return parts[parts.index("--pwd") + 1]


# ----------------------------------------------------------------------
# Command construction
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gpu", "expected"),
    [
        (0, "--env CUDA_VISIBLE_DEVICES=0"),
        (2, "--env CUDA_VISIBLE_DEVICES=2"),
        (None, None),
    ],
)
def test_gpu_containers_request_passthrough(
    tmp_path: Path, gpu: int | None, expected: str | None
) -> None:
    command = _deployment(tmp_path).apptainer_command(_container(gpu=gpu))

    if expected is None:
        assert "CUDA_VISIBLE_DEVICES" not in command
        assert "--nv" not in command
    else:
        assert expected in command
        assert "--nv" in command


def test_telemetry_container_sees_all_gpus(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, num_gpus=2)

    command = deployment.apptainer_command(_container("prometheus"))

    assert "--nv" in command
    assert "CUDA_VISIBLE_DEVICES" not in command


def test_telemetry_container_without_gpus_has_no_nv(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, num_gpus=0)

    assert "--nv" not in deployment.apptainer_command(_container("prometheus"))


def test_volumes_become_binds(tmp_path: Path) -> None:
    container = _container(volumes=["/host/data:/mnt/data", "/host/ro:/mnt/ro:ro"])

    command = _deployment(tmp_path).apptainer_command(container)

    assert "--bind /host/data:/mnt/data" in command
    assert "--bind /host/ro:/mnt/ro:ro" in command


def test_workdir_defaults_to_config_and_service_overrides_it(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path)

    assert "--pwd /repo" in deployment.apptainer_command(_container())
    assert "--pwd /app/runfiles" in deployment.apptainer_command(
        _container(workdir="/app/runfiles")
    )


def test_external_image_runs_from_root_unless_it_sets_a_workdir(tmp_path: Path) -> None:
    """`/repo` exists only in the alpasim images; Apptainer fails to chdir there."""
    deployment = _deployment(tmp_path)

    external = _container("renderer", image="nre-ga:26.02", external_image=True)
    assert _pwd_arg(deployment.apptainer_command(external)) == "/"

    with_workdir = _container(
        "renderer", image="nre-ga:26.02", external_image=True, workdir="/app"
    )
    assert _pwd_arg(deployment.apptainer_command(with_workdir)) == "/app"


def test_environment_is_set_and_passed_through(tmp_path: Path) -> None:
    container = _container(environments=["HF_TOKEN", "HOME=/tmp"])

    command = _deployment(tmp_path).apptainer_command(container)

    # Defaults from the config, then the service's own settings.
    assert "--env UV_PROJECT_ENVIRONMENT=/repo/.venv" in command
    assert "--env PYTHONDONTWRITEBYTECODE=1" in command
    assert "--env HOME=/tmp" in command
    # Bare names take the host value, defaulting to empty so `set -u` in the
    # generated run.sh does not abort on an unset variable.
    assert '--env HF_TOKEN="${HF_TOKEN-}"' in command


def test_multiline_command_with_quotes_survives_quoting(tmp_path: Path) -> None:
    script = 'uv run server &\nPID0=$!\ntrap \'kill "$PID0"\' TERM\nwait "$PID0"'
    container = _container(command=script)

    command = _deployment(tmp_path).apptainer_command(container)

    assert _bash_payload(command) == script


def test_escaped_dollar_is_unescaped_like_slurm(tmp_path: Path) -> None:
    container = _container(command="echo $$HOSTNAME")

    command = _deployment(tmp_path).apptainer_command(container)

    assert _bash_payload(command) == "echo $HOSTNAME"


def test_noop_container_runs_no_command(tmp_path: Path) -> None:
    command = _deployment(tmp_path).apptainer_command(_container(command="noop"))

    assert "bash -c" not in command


def test_binary_and_extra_args_are_configurable(tmp_path: Path) -> None:
    deployment = _deployment(
        tmp_path,
        apptainer=WizardApptainerConfig(
            binary="/opt/apptainer/bin/apptainer",
            extra_exec_args=["--containall"],
        ),
    )

    command = deployment.apptainer_command(_container())

    assert command.startswith("/opt/apptainer/bin/apptainer exec")
    assert "--containall" in command


# ----------------------------------------------------------------------
# Image resolution
# ----------------------------------------------------------------------


def test_image_basename_matches_cache_convention() -> None:
    assert image_to_apptainer_basename("alpasim-base:0.111.0") == (
        "alpasim_base_0.111.0.sif"
    )
    assert image_to_apptainer_basename("nvcr.io/org/my-image:1.2") == (
        "my_image_1.2.sif"
    )


@pytest.mark.parametrize("suffix", [".sif", ".sandbox", ""])
def test_cached_image_is_preferred(tmp_path: Path, suffix: str) -> None:
    cache = tmp_path / "sif-cache"
    cache.mkdir()
    cached = cache / f"alpasim_base_0.1.0{suffix}"
    if suffix == ".sif":
        cached.write_bytes(b"")
    else:
        cached.mkdir()

    resolved = resolve_apptainer_image("alpasim-base:0.1.0", [str(cache)])

    assert resolved == str(cached)


def test_missing_image_falls_back_to_registry(tmp_path: Path) -> None:
    resolved = resolve_apptainer_image("alpasim-base:0.1.0", [str(tmp_path)])

    assert resolved == "docker://alpasim-base:0.1.0"


def test_missing_image_without_fallback_reports_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="alpasim_base_0.1.0.sif"):
        resolve_apptainer_image(
            "alpasim-base:0.1.0", [str(tmp_path)], registry_fallback=False
        )


def test_deployment_uses_resolved_image(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "alpasim_base_0.1.0.sif").write_bytes(b"")
    deployment = _deployment(
        tmp_path, apptainer=WizardApptainerConfig(image_caches=[str(cache)])
    )

    command = deployment.apptainer_command(_container())

    assert str(cache / "alpasim_base_0.1.0.sif") in command
    assert "docker://" not in command


# ----------------------------------------------------------------------
# Writable scratch
# ----------------------------------------------------------------------


def test_writable_tmpfs_is_the_default_scratch(tmp_path: Path) -> None:
    assert "--writable-tmpfs" in _deployment(tmp_path).apptainer_command(_container())


def test_writable_tmpfs_can_be_disabled(tmp_path: Path) -> None:
    deployment = _deployment(
        tmp_path, apptainer=WizardApptainerConfig(writable_tmpfs=False)
    )

    command = deployment.apptainer_command(_container())

    assert "--writable-tmpfs" not in command
    assert "--overlay" not in command


def test_matching_images_get_a_file_backed_overlay(tmp_path: Path) -> None:
    deployment = _deployment(
        tmp_path,
        apptainer=WizardApptainerConfig(
            overlay_image_patterns=["renderer"], overlay_size_mb=2048
        ),
    )
    container = _container("renderer", image="my-renderer:1.0")

    command = deployment.apptainer_command(container)
    overlay = tmp_path / "apptainer-overlays" / "renderer-0.img"

    assert f"overlay create --size 2048 {overlay}" in command
    assert f"--overlay {overlay}" in command
    # tmpfs and a file-backed overlay are alternatives, not cumulative.
    assert "--writable-tmpfs" not in command
    # Created lazily by the command, so re-runs reuse an existing overlay.
    assert f"[ -f {overlay} ] ||" in command
    assert overlay.parent.is_dir()


def test_overlay_is_per_container(tmp_path: Path) -> None:
    """Concurrent containers must not share one ext3 overlay."""
    deployment = _deployment(
        tmp_path, apptainer=WizardApptainerConfig(overlay_image_patterns=["renderer"])
    )
    first = _container("renderer", image="renderer:1")
    second = _container("renderer", image="renderer:1")
    second.uuid = "renderer-1"

    assert "renderer-0.img" in deployment.apptainer_command(first)
    assert "renderer-1.img" in deployment.apptainer_command(second)


# ----------------------------------------------------------------------
# Deployment lifecycle
# ----------------------------------------------------------------------


def _staged_deployment(
    tmp_path: Path,
    *,
    runtime_returncode: int = 0,
    dry_run: bool = False,
) -> tuple[ApptainerDeployment, list[tuple[str, str]]]:
    """Deployment with a fake container set and no real subprocesses."""
    deployment = _deployment(tmp_path, dry_run=dry_run)
    deployment.container_set = SimpleNamespace(
        sim=[
            _container("driver", ports=[6000]),
            _container("trafficsim", command="noop"),
        ],
        prometheus=_container("prometheus", ports=[6100]),
        runtime=_container("runtime", command="uv run simulate"),
    )

    events: list[tuple[str, str]] = []

    def fake_popen(command: str, log_file: Any) -> FakeProcess:
        del log_file
        if "uv run simulate" in command:
            return FakeProcess(returncode=runtime_returncode)
        return FakeProcess()

    deployment._popen = fake_popen  # type: ignore[method-assign]
    deployment._launch_background = _recording(  # type: ignore[method-assign]
        deployment._launch_background, events, "background"
    )
    deployment._run_foreground = _recording(  # type: ignore[method-assign]
        deployment._run_foreground, events, "foreground"
    )
    deployment._signal_group = lambda uuid, process, sig: events.append(  # type: ignore[method-assign]
        ("signal", uuid)
    )
    deployment._wait_for_containers = lambda containers, timeout=None: events.append(  # type: ignore[method-assign]
        ("wait", ",".join(c.uuid for c in containers))
    )
    return deployment, events


def _recording(method: Any, events: list[tuple[str, str]], label: str) -> Any:
    def wrapper(container: Any) -> None:
        events.append((label, container.uuid))
        return method(container)

    return wrapper


def test_services_start_before_runtime_and_stop_after_it(tmp_path: Path) -> None:
    deployment, events = _staged_deployment(tmp_path)

    deployment.deploy_all_services()

    assert events == [
        ("background", "driver-0"),
        ("background", "prometheus-0"),
        ("wait", "driver-0"),
        ("foreground", "runtime-0"),
        ("signal", "driver-0"),
        ("signal", "prometheus-0"),
    ]


def test_services_only_deployment_serves_until_the_services_exit(
    tmp_path: Path,
) -> None:
    deployment, events = _staged_deployment(tmp_path)
    deployment.container_set.runtime = None

    deployment.deploy_all_services()

    assert events == [
        ("background", "driver-0"),
        ("background", "prometheus-0"),
        ("wait", "driver-0"),
        ("signal", "driver-0"),
        ("signal", "prometheus-0"),
    ]


def test_services_are_stopped_when_the_runtime_fails(tmp_path: Path) -> None:
    deployment, events = _staged_deployment(tmp_path, runtime_returncode=1)

    with pytest.raises(OsDispatchError, match="runtime-0"):
        deployment.deploy_all_services()

    assert ("signal", "driver-0") in events
    assert ("signal", "prometheus-0") in events


def test_dry_run_starts_nothing_but_writes_the_script(tmp_path: Path) -> None:
    deployment, events = _staged_deployment(tmp_path, dry_run=True)

    deployment.deploy_all_services()

    assert deployment._background == []
    assert ("wait", "driver-0") not in events
    assert (tmp_path / "run.sh").is_file()


def test_startup_failure_is_reported_immediately(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path)
    # Port 1 is privileged, so nothing on the test host can be listening there.
    container = _container("driver", ports=[1])
    dead = FakeProcess(returncode=3)
    dead.wait()
    deployment._background = [("driver-0", dead)]  # type: ignore[list-item]

    with pytest.raises(OsDispatchError, match="return code 3"):
        deployment._wait_for_containers([container])


def test_wait_times_out_when_a_service_never_listens(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path)
    container = _container("driver", ports=[1])

    with pytest.raises(TimeoutError, match="did not become ready"):
        deployment._wait_for_containers([container], timeout=0)


# ----------------------------------------------------------------------
# Generated run script
# ----------------------------------------------------------------------


def test_run_script_is_executable_and_ordered(tmp_path: Path) -> None:
    deployment, _ = _staged_deployment(tmp_path)

    script_path = deployment.generate_run_script()
    script = script_path.read_text()

    assert script.startswith("#!/bin/bash\n")
    assert script_path.stat().st_mode & stat.S_IXUSR
    assert "set -euo pipefail" in script
    # Services background, the runtime does not.
    assert "SERVICE_PIDS+=($!)" in script
    assert script.rstrip().endswith("bash -c 'uv run simulate'")
    # Services are waited on by address before the runtime starts.
    assert "wait_for_port 0.0.0.0 6000 1" in script
    assert script.index("wait_for_port 0.0.0.0 6000") < script.index("uv run simulate")
    # Telemetry starts but is not waited on, as in deploy_all_services.
    assert "prometheus-0" in script
    assert "wait_for_port 0.0.0.0 6100" not in script
    # Background services are cleaned up when the script exits.
    assert "trap cleanup EXIT" in script
    # The skipped service is not started.
    assert "trafficsim" not in script
