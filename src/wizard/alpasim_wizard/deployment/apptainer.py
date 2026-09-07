# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Apptainer commands and a single, reproducible shell deployment."""

from __future__ import annotations

import logging
import re
import shlex
import signal
import subprocess
from pathlib import Path
from types import FrameType

from ..context import WizardContext
from ..schema import WizardApptainerConfig
from ..services import ContainerDefinition, build_container_set
from ..utils import resolve_apptainer_image
from .dispatcher import OsDispatchError, terminate_process

logger = logging.getLogger(__name__)


class ApptainerDeployment:
    """Run services on one host, optionally inside a scheduler allocation.

    The wizard and standalone runs execute the same generated script. Image
    environments are preserved; workdirs and service overrides are explicit.
    """

    def __init__(self, context: WizardContext):
        self.context = context
        self.container_set = build_container_set(context, use_address_string="0.0.0.0")

    def deploy_all_services(self) -> None:
        """Execute run.sh and forward interruption to its cleanup traps."""
        script = self.generate_run_script()
        if self.context.cfg.wizard.dry_run:
            logger.info("[DRY-RUN] Prepared %s", script)
            return
        process = subprocess.Popen(
            ["bash", str(script)], start_new_session=True, text=True
        )

        def forward_signal(signum: int, frame: FrameType | None) -> None:
            process.send_signal(signum)

        previous = {
            sig: signal.signal(sig, forward_signal)
            for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            return_code = process.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            if process.poll() is None:
                terminate_process(process, timeout=15)
        if return_code != 0:
            raise OsDispatchError(
                f"Apptainer deployment failed with return code {return_code}; "
                f"see {script.parent / 'txt-logs'}"
            )

    def generate_run_script(self) -> Path:
        """Write the executable deployment, including logs and bounded cleanup."""
        log_dir = Path(self.context.cfg.wizard.log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        lines = [Path(__file__).with_name("apptainer_run.sh").read_text()]
        lines.append(f"mkdir -p {shlex.quote(str(log_dir / 'txt-logs'))}")
        services = [c for c in self.container_set.sim if c.command != "noop"]
        timeout = self.context.cfg.wizard.timeout
        lines.append(
            f"deadline=$((SECONDS + {timeout if timeout is not None else 600}))"
        )
        for container in [*services, self.container_set.prometheus]:
            log = log_dir / "txt-logs" / f"out-{container.uuid}-log.txt"
            lines.append(
                f"launch {shlex.quote(container.uuid)} "
                f"{shlex.quote(self.apptainer_command(container))} {shlex.quote(str(log))}"
            )
        for container in services:
            for address in container.get_all_addresses():
                lines.append(
                    f"wait_for_port {shlex.quote(container.uuid)} "
                    f"{shlex.quote(address.host)} {address.port}"
                )
        runtime = self.container_set.runtime
        if runtime is not None:
            log = log_dir / "txt-logs" / f"out-{runtime.uuid}-log.txt"
            lines += [
                f"launch {shlex.quote(runtime.uuid)} "
                f"{shlex.quote(self.apptainer_command(runtime))} {shlex.quote(str(log))}",
                'wait "$last_pid"',
            ]
        else:
            # A services-only run remains alive until a service exits. Telemetry
            # does not gate startup or decide the simulation's lifetime.
            names = services or [self.container_set.prometheus]
            pids = " ".join(f'"${{pids[{c.uuid}]}}"' for c in names)
            lines.append(f"wait_for_services {pids}")
        script = log_dir / "run.sh"
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script.chmod(0o755)
        logger.info("Generated Apptainer run script: %s", script)
        return script

    @property
    def config(self) -> WizardApptainerConfig:
        """Apptainer-specific settings from the wizard config."""
        apptainer_config: WizardApptainerConfig = self.context.cfg.wizard.apptainer
        return apptainer_config

    def apptainer_command(self, container: ContainerDefinition) -> str:
        """Build the shell command that runs a container under Apptainer.

        Args:
            container: ContainerDefinition instance

        Returns:
            A single shell command, safe to pass to a shell as-is.
        """
        config = self.config
        image = resolve_apptainer_image(
            container.service_config.image,
            list(config.image_caches),
            registry_fallback=config.registry_fallback,
        )

        args: list[str] = [shlex.quote(config.binary), "exec"]

        if config.cleanenv:
            args.append("--cleanenv")
        args.append("--no-eval")
        setup = ""
        # Resolve topology indices inside the allocation at execution time, so
        # scripts prepared on a login node also honor Slurm's device selection.
        if container.gpu is not None:
            if container.gpu < 0:
                raise ValueError("GPU indices must be nonnegative")
            setup = (
                f"gpu={container.gpu}; "
                "if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then "
                'IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"; '
                "if [[ -z ${devices[$gpu]-} || ${devices[$gpu]} == -1 ]]; then "
                'echo "GPU index $gpu is outside CUDA_VISIBLE_DEVICES" >&2; exit 1; fi; '
                "gpu=${devices[$gpu]}; fi; "
            )
            setup += 'export APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu"; '
            args.append("--nv")
        elif container.name == "prometheus" and self.context.num_gpus > 0:
            args += ["--nv"]
            setup = (
                "if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then "
                'export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"; fi; '
            )

        if container.gpu is not None or container.name == "prometheus":
            setup += (
                "if [[ ${CUDA_DEVICE_ORDER+x} ]]; then "
                'export APPTAINERENV_CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER"; fi; '
            )

        args += [
            f"--bind {shlex.quote(volume.to_str())}" for volume in container.volumes
        ]

        # Apptainer ignores the image's WORKDIR and starts in the host CWD. The
        # configured default lives in the alpasim images only, so images from
        # elsewhere run from `/` unless their service config names a directory.
        workdir = container.workdir or (
            "/" if container.service_config.external_image else config.workdir
        )
        args.append(f"--pwd {shlex.quote(workdir)}")

        # Prefix variables avoid --env's comma-separated value parsing. Export
        # them only in the launched shell, never into the wizard's environment.
        for env in list(config.environments) + list(container.environments or []):
            name, separator, value = env.partition("=")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"Invalid environment variable name: {name!r}")
            value = shlex.quote(value) if separator else f'"${{{name}-}}"'
            setup += f"export APPTAINERENV_{name}={value}; "

        args += [shlex.quote(arg) for arg in config.extra_exec_args]

        size = container.service_config.apptainer_overlay_size_mb
        if size is not None:
            if size <= 0:
                raise ValueError("apptainer_overlay_size_mb must be positive")
            overlay = (
                Path(self.context.cfg.wizard.log_dir).resolve()
                / "apptainer-overlays"
                / f"{container.uuid}.img"
            )
            setup += (
                f"mkdir -p {shlex.quote(str(overlay.parent))}; "
                f"if [ ! -f {shlex.quote(str(overlay))} ]; then "
                f"{shlex.quote(config.binary)} overlay create "
                f"--size {size} {shlex.quote(str(overlay))} || exit $?; fi; "
            )
            args.append(f"--overlay {shlex.quote(str(overlay))}")
        elif config.writable_tmpfs:
            args.append("--writable-tmpfs")

        args.append(shlex.quote(image))

        if container.command and container.command != "noop":
            # Same unescaping as the SLURM backend: `$$` marks a `$` that the
            # service, not the wizard's config parser, should interpret.
            command = container.command.replace("$$", "$")
            args.append(f"bash -c {shlex.quote(command)}")

        return setup + "exec " + " ".join(args)
