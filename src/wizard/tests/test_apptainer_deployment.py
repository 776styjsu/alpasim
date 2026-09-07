# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from alpasim_wizard.deployment.apptainer import ApptainerDeployment
from alpasim_wizard.deployment.dispatcher import OsDispatchError
from alpasim_wizard.schema import WizardApptainerConfig
from alpasim_wizard.services import Address, VolumeMount
from alpasim_wizard.utils import image_to_apptainer_basename, resolve_apptainer_image


def _container(
    name: str = "driver",
    *,
    gpu: int | None = None,
    image: str = "alpasim-base:1",
    external_image: bool = False,
    workdir: str | None = None,
    environments: tuple[str, ...] = (),
    volumes: tuple[str, ...] = (),
    command: str = "echo ok",
    port: int | None = None,
    overlay_size: int | None = None,
) -> Any:
    return SimpleNamespace(
        name=name,
        uuid=f"{name}-0",
        gpu=gpu,
        service_config=SimpleNamespace(
            image=image,
            external_image=external_image,
            apptainer_overlay_size_mb=overlay_size,
        ),
        environments=list(environments),
        volumes=[VolumeMount.from_str(v) for v in volumes],
        workdir=workdir,
        command=command,
        get_all_addresses=lambda: [Address("127.0.0.1", port)] if port else [],
    )


def _deployment(tmp_path: Path, **settings: Any) -> ApptainerDeployment:
    deployment = ApptainerDeployment.__new__(ApptainerDeployment)
    deployment.context = SimpleNamespace(
        cfg=SimpleNamespace(
            wizard=SimpleNamespace(
                log_dir=str(tmp_path),
                dry_run=False,
                timeout=3,
                apptainer=WizardApptainerConfig(**settings),
            )
        ),
        num_gpus=2,
    )
    return deployment


@pytest.fixture
def binary(tmp_path: Path) -> str:
    """Record the actual shell argv, then execute the service without a container."""
    path = tmp_path / "fake apptainer"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if args[0] == 'overlay':\n"
        "    pathlib.Path(args[-1]).touch()\n"
        "else:\n"
        "    observed = []\n"
        "    for name in ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'TEST_TOKEN', 'HOME']:\n"
        "        key = 'APPTAINERENV_' + name\n"
        "        if key in os.environ:\n"
        "            observed.append(name + '=' + os.environ[key])\n"
        "    print(json.dumps(args + observed), flush=True)\n"
        "    index = args.index('bash')\n"
        "    os.execvp('bash', args[index:])\n"
    )
    path.chmod(0o755)
    return str(path)


def _run_command(deployment: ApptainerDeployment, container: Any) -> list[str]:
    result = subprocess.run(
        ["bash", "-euc", deployment.apptainer_command(container)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.splitlines()[0])


@pytest.mark.parametrize(
    ("visible", "index", "expected"),
    [
        (None, 2, "2"),
        ("3,5", 0, "3"),
        ("3,5", 1, "5"),
        ("GPU-abc,GPU-def", 1, "GPU-def"),
    ],
)
def test_gpu_indices_resolve_at_execution(
    tmp_path: Path,
    binary: str,
    monkeypatch: pytest.MonkeyPatch,
    visible: str | None,
    index: int,
    expected: str,
) -> None:
    deployment = _deployment(tmp_path, binary=binary)
    # The allocation can change after script generation.
    command = deployment.apptainer_command(_container(gpu=index))
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    result = subprocess.run(
        ["bash", "-euc", command], check=True, capture_output=True, text=True
    )
    args = json.loads(result.stdout.splitlines()[0])
    assert f"CUDA_VISIBLE_DEVICES={expected}" in args
    assert "--nv" in args


@pytest.mark.parametrize("visible", ["", "-1", "3"])
def test_gpu_outside_allocation_fails(
    tmp_path: Path,
    binary: str,
    monkeypatch: pytest.MonkeyPatch,
    visible: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    command = _deployment(tmp_path, binary=binary).apptainer_command(_container(gpu=1))
    result = subprocess.run(["bash", "-euc", command], capture_output=True, text=True)
    assert result.returncode != 0
    assert "outside CUDA_VISIBLE_DEVICES" in result.stderr


def test_telemetry_preserves_allocation(
    tmp_path: Path,
    binary: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc,GPU-def")
    args = _run_command(_deployment(tmp_path, binary=binary), _container("prometheus"))
    assert "CUDA_VISIBLE_DEVICES=GPU-abc,GPU-def" in args


def test_command_preserves_arguments_and_service_environment(
    tmp_path: Path,
    binary: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_TOKEN", "literal $value, with spaces")
    deployment = _deployment(
        tmp_path, binary=binary, extra_exec_args=["--hostname", "a b"]
    )
    container = _container(
        volumes=("/host/data dir:/mnt/data:ro",),
        environments=("TEST_TOKEN", "HOME=/tmp"),
        workdir="/app/run files",
        command="printf '%s' '$$literal'",
        external_image=True,
    )
    args = _run_command(deployment, container)
    assert "--cleanenv" in args and "--no-eval" in args
    assert "a b" in args
    assert "/host/data dir:/mnt/data:ro" in args
    assert "/app/run files" in args
    assert "TEST_TOKEN=literal $value, with spaces" in args
    assert "HOME=/tmp" in args
    assert not any(arg.startswith("UV_PROJECT_ENVIRONMENT=") for arg in args)
    assert args[args.index("-c") + 1] == "printf '%s' '$literal'"


@pytest.mark.parametrize(("external", "workdir"), [(False, "/repo"), (True, "/")])
def test_image_workdir(
    tmp_path: Path, binary: str, external: bool, workdir: str
) -> None:
    args = _run_command(
        _deployment(tmp_path, binary=binary), _container(external_image=external)
    )
    assert args[args.index("--pwd") + 1] == workdir


def test_overlay_is_explicit_and_per_container(tmp_path: Path, binary: str) -> None:
    deployment = _deployment(tmp_path, binary=binary)
    first = _container("renderer", overlay_size=1024)
    second = _container("renderer", overlay_size=2048)
    second.uuid = "renderer-1"
    args = _run_command(deployment, first)
    other = _run_command(deployment, second)
    assert "--writable-tmpfs" not in args
    assert args[args.index("--overlay") + 1] != other[other.index("--overlay") + 1]
    assert (tmp_path / "apptainer-overlays/renderer-0.img").is_file()
    assert "--overlay" not in _run_command(deployment, _container())


@pytest.mark.parametrize("suffix", [".sif", ".sandbox"])
def test_direct_local_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    path = tmp_path / f"custom{suffix}"
    path.touch() if suffix == ".sif" else path.mkdir()
    monkeypatch.chdir(tmp_path)
    assert resolve_apptainer_image(path.name, [], False) == str(path)
    assert resolve_apptainer_image(str(path), []) == str(path)


@pytest.mark.parametrize(
    "reference", ["docker://org/image:1", "oras://org/image:1", "library://org/image:1"]
)
def test_native_uri_is_preserved(reference: str) -> None:
    assert resolve_apptainer_image(reference, [], False) == reference


def test_missing_local_image_does_not_pull(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_apptainer_image(str(tmp_path / "missing.sif"), [])


def test_full_reference_cache_identity(tmp_path: Path) -> None:
    references = [
        "registry-a/team/image:1",
        "registry-b/team/image:1",
        "a-b:1",
        "a_b:1",
    ]
    assert len({image_to_apptainer_basename(r) for r in references}) == len(references)
    reference = references[0]
    image = tmp_path / image_to_apptainer_basename(reference)
    image.touch()
    assert resolve_apptainer_image(reference, [str(tmp_path)], False) == str(image)
    assert resolve_apptainer_image(f"docker://{reference}", [str(tmp_path)]) == str(
        image
    )
    assert (
        resolve_apptainer_image(references[1], [str(tmp_path)])
        == f"docker://{references[1]}"
    )
    with pytest.raises(ValueError, match="No cached Apptainer image"):
        resolve_apptainer_image(references[1], [str(tmp_path)], False)


def _python_command(code: str) -> str:
    return shlex.join([sys.executable, "-c", code])


def _wait_for_file(path: Path, process: subprocess.Popen[Any]) -> None:
    deadline = time.monotonic() + 5
    while not path.exists():
        assert process.poll() is None, f"process exited with {process.returncode}"
        assert time.monotonic() < deadline, f"did not create {path}"
        time.sleep(0.02)


def _service(tmp_path: Path, *, ignore_term: bool = False) -> Any:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    code = (
        "import os, pathlib, signal, socket, time; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
        + f"s=socket.socket(); s.bind(('127.0.0.1', {port})); s.listen(); "
        + f"pathlib.Path({str(tmp_path / 'service.pid')!r}).write_text(str(os.getpid())); "
        + "time.sleep(60)"
    )
    return _container(command=_python_command(code), port=port)


def _staged(tmp_path: Path, binary: str, runtime: Any | None) -> ApptainerDeployment:
    deployment = _deployment(tmp_path, binary=binary)
    deployment.container_set = SimpleNamespace(
        sim=[_service(tmp_path), _container("skipped", command="noop")],
        prometheus=_container("prometheus", command="exit 0"),
        runtime=runtime,
    )
    return deployment


def test_wizard_executes_script_and_preserves_failure(
    tmp_path: Path, binary: str
) -> None:
    deployment = _staged(
        tmp_path, binary, _container("runtime", command="echo runtime-output; exit 7")
    )
    with pytest.raises(OsDispatchError, match="return code 7"):
        deployment.deploy_all_services()
    assert "runtime-output" in (tmp_path / "txt-logs/out-runtime-0-log.txt").read_text()
    pid = int((tmp_path / "service.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("runtime_enabled", [False, True])
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_script_waits_and_cleans_up_on_interrupt(
    tmp_path: Path,
    binary: str,
    runtime_enabled: bool,
    sig: signal.Signals,
) -> None:
    runtime_pid = tmp_path / "runtime.pid"
    runtime = (
        _container(
            "runtime",
            command=_python_command(
                "import os, pathlib, time; "
                f"pathlib.Path({str(runtime_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
            ),
        )
        if runtime_enabled
        else None
    )
    deployment = _staged(tmp_path, binary, runtime)
    script = deployment.generate_run_script()
    process = subprocess.Popen(["bash", str(script)], start_new_session=True)
    try:
        _wait_for_file(
            runtime_pid if runtime_enabled else tmp_path / "service.pid", process
        )
        time.sleep(0.15)
        assert process.poll() is None
        process.send_signal(sig)
        assert process.wait(timeout=15) == 128 + sig
        for path in [
            tmp_path / "service.pid",
            *([runtime_pid] if runtime_enabled else []),
        ]:
            with pytest.raises(ProcessLookupError):
                os.kill(int(path.read_text()), 0)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)


def test_startup_failure_and_timeout(tmp_path: Path, binary: str) -> None:
    deployment = _staged(tmp_path, binary, _container("runtime"))
    deployment.container_set.sim[0].command = "exit 3"
    result = subprocess.run(
        ["bash", str(deployment.generate_run_script())],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "exited while starting" in result.stderr
    deployment.container_set.sim[0].command = "exec sleep 60"
    deployment.context.cfg.wizard.timeout = 0
    result = subprocess.run(
        ["bash", str(deployment.generate_run_script())],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "Timed out" in result.stderr


def test_dry_run_does_not_launch_or_create_overlays(tmp_path: Path) -> None:
    deployment = _staged(
        tmp_path, "missing-apptainer", _container("runtime", overlay_size=1024)
    )
    deployment.context.cfg.wizard.dry_run = True
    deployment.deploy_all_services()
    assert (tmp_path / "run.sh").is_file()
    assert not (tmp_path / "apptainer-overlays").exists()
    assert not (tmp_path / "service.pid").exists()


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_wizard_forwards_signals_to_runtime(
    tmp_path: Path,
    binary: str,
    sig: signal.Signals,
) -> None:
    runtime_pid = tmp_path / "runtime.pid"
    runtime_code = (
        "import os, pathlib, time; "
        f"pathlib.Path({str(runtime_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    code = (
        f"import runpy; from pathlib import Path; h=runpy.run_path({__file__!r}); "
        f"runtime=h['_container']('runtime', command=h['_python_command']("
        f"{runtime_code!r})); "
        f"h['_staged'](Path({str(tmp_path)!r}), {binary!r}, runtime).deploy_all_services()"
    )
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        _wait_for_file(runtime_pid, process)
        process.send_signal(sig)
        assert process.wait(timeout=15) != 0
        for path in [tmp_path / "service.pid", runtime_pid]:
            with pytest.raises(ProcessLookupError):
                os.kill(int(path.read_text()), 0)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)


def test_cleanup_kills_a_service_that_ignores_term(tmp_path: Path, binary: str) -> None:
    deployment = _staged(tmp_path, binary, _container("runtime", command="exit 0"))
    deployment.container_set.sim = [_service(tmp_path, ignore_term=True)]
    start = time.monotonic()
    deployment.deploy_all_services()
    assert time.monotonic() - start < 15
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "service.pid").read_text()), 0)
