# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class DebugFlags:
    """Flags or settings purely for developing or debugging.

    All must be entirely optional and cannot be used in production.
    Even the existence of `debug_flags` in the config should be optional!
    """

    # Use `localhost` in `generated-network-config.yaml` and adds
    # `network_mode: host` to the docker-compose.yaml.
    # This allows combining running services and/or the runtime on the host and
    # in containers. Very helpful for debugging.
    use_localhost: bool = False


@dataclass
class AlpasimConfig:
    defines: dict[str, str] = MISSING
    wizard: WizardConfig = MISSING
    scenes: ScenesConfig = MISSING
    services: ServicesConfig = MISSING
    runtime: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    trafficsim: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    eval: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    driver: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    controller: DictConfig = field(default_factory=lambda: OmegaConf.create({}))


@dataclass
class ScenesConfig:
    # Selection method (exactly one must be set)
    scene_ids: list[str] | None = None
    test_suite_id: str | None = None

    # Limit the number of scenes to run (0 or negative means no limit)
    limit_to_first_n: int = 0

    # Paths
    scene_cache: str = MISSING
    scenes_csv: list[str] = MISSING
    suites_csv: list[str] = MISSING

    # Relative path within scene_cache to the sceneset directory for this run.
    # Set automatically by the wizard; used by reeval to locate the correct USDZs.
    sceneset_path: str | None = None

    # Optional: path to a local directory containing *.usdz files.
    # When set, the wizard will scan this directory to generate in-memory
    # sim_scenes/sim_suites data, bypassing the CSV files. A test suite (called "local")
    # is created automatically containing all discovered scenes.
    # If local_usdz_dir is provided and neither scene_ids nor test_suite_id is set,
    # all scenes in the directory will be simulated.
    local_usdz_dir: str | None = None

    # Used to override services.renderer.image for the USDZ database service if NRE is not enabled.
    nre_version_string: str | None = None


class RunMethod(Enum):
    SLURM = "slurm"
    SLURM_ENROOT = "slurm_enroot"
    DOCKER_COMPOSE = "docker_compose"
    APPTAINER = "apptainer"
    NONE = "none"

    @property
    def is_slurm(self) -> bool:
        return self in (RunMethod.SLURM, RunMethod.SLURM_ENROOT)


class RunMode(Enum):
    """Runtime lifecycle mode.

    ONESHOT starts the runtime to execute the generated simulation once and
    then exit. SERVER starts a long-running runtime daemon that serves
    simulation requests over gRPC.
    """

    ONESHOT = "oneshot"
    SERVER = "server"


@dataclass
class WizardPrometheusConfig:
    """Prometheus ownership, identity, and discovery settings."""

    scrape_interval: str = "5s"
    file_sd_dir: str | None = None
    start_prometheus: bool = True
    run_uuid: str | None = None


@dataclass
class WizardApptainerConfig:
    """Settings for the Apptainer deployment (`run_method: APPTAINER`).

    Apptainer runs containers unprivileged as the calling user, which makes it
    the usual choice on HPC clusters where Docker is unavailable. Defaults here
    are deliberately site-neutral: everything that depends on a particular
    cluster (module names, image caches, overlay policy) is opt-in.
    """

    # Optional caches populated by build_apptainer.sh. Filenames include a
    # digest of the full image reference to avoid registry/name collisions.
    image_caches: list[str] = field(default_factory=list)

    # When no cached image matches, run `docker://<image>` so Apptainer pulls and
    # converts it on the fly. Set to false on clusters without outbound network
    # access, so a missing image fails immediately with the paths searched.
    registry_fallback: bool = True

    # Apptainer executable. Set an absolute path when it is not on PATH; on
    # clusters using environment modules, `module load apptainer` in the job
    # script before starting the wizard works too.
    binary: str = "apptainer"

    # Keep module/host variables out; service environments are passed explicitly.
    cleanenv: bool = True

    # Extra arguments added to every `apptainer exec`, e.g. ["--containall"].
    extra_exec_args: list[str] = field(default_factory=list)

    # Explicit additions to the image environment. Bare names pass host values.
    environments: list[str] = field(
        default_factory=lambda: ["PYTHONDONTWRITEBYTECODE=1"]
    )

    # Working directory for services built from an alpasim image that do not set
    # `workdir` themselves. Apptainer ignores the image's WORKDIR, so it has to
    # be passed explicitly. Services with `external_image` run from `/` instead,
    # unless their own service config sets `workdir`.
    workdir: str = "/repo"

    # Give containers a writable in-memory layer so processes can write outside
    # bind mounts. Disable on systems where Apptainer cannot set up overlays.
    writable_tmpfs: bool = True


@dataclass
class WizardConfig:
    # Name of the run, used to identify the run in the databases.
    run_name: str | None = None
    run_method: RunMethod = MISSING
    run_mode: RunMode = MISSING

    # Global log level for all alpasim services (DEBUG, INFO, WARNING, ERROR)
    log_level: str = "INFO"
    prometheus: WizardPrometheusConfig = field(default_factory=WizardPrometheusConfig)
    description: str | None = None  # TODO(mwatson): is this redundant to run_name?
    submitter: str | None = None

    latest_symlink: bool = MISSING
    log_dir: str = "."
    array_job_dir: str | None = None
    dry_run: bool = MISSING
    baseport: int = MISSING
    runtime_server_port: int | None = None
    validate_mount_points: bool = MISSING

    # If set, the wizard will pull the driver code from the specified hash into
    # `${wizard.log_dir}/driver_code`. Can be useful for mounting into the
    # driver container for debugging.
    driver_code_hash: str | None = None

    # Used if `driver_code_hash` is set. Requires configured ssh keys for
    # pulling from gitlab, but can also point towards a local repo!
    driver_code_repo: str | None = None

    helper: str = MISSING
    vscode: str = MISSING

    sqshcaches: list[str] = MISSING

    # Settings for run_method: APPTAINER. Ignored by the other run methods.
    apptainer: WizardApptainerConfig = field(default_factory=WizardApptainerConfig)

    slurm_job_id: int | None = MISSING
    timeout: int = MISSING
    nr_retries: int = 3
    run_sim_services: list[str] | None = MISSING
    debug_flags: DebugFlags = field(default_factory=DebugFlags)

    # When True, add --cpu-bind=none to srun --overlap steps.  Required on
    # SLURM nodes where the batch step binds all CPUs (e.g. non-exclusive
    # allocations on CI nodes), otherwise overlapping steps are killed.
    slurm_cpu_bind_none: bool = False

    # Root directory where the wizard installs direct-Enroot FUSE tools.
    # Required with run_method=SLURM_ENROOT.
    fuse_dir: str | None = None

    # Optional persistent node-local cache for direct-Enroot squash images.
    node_local_sqsh_cache_dir: str | None = None
    node_local_sqsh_cache_max_gib: int = 500

    # Run GPU services through CUDA MPS so kernels from co-located processes
    # execute concurrently instead of time-slicing between CUDA contexts.
    # Starts a per-job MPS control daemon on the node. Slurm run methods only.
    enable_mps: bool = False

    # External service addresses for services running outside the deployment.
    # Maps service name to list of addresses (e.g., {"driver": ["localhost:6789"]}).
    # These addresses are added to generated-network-config.yaml so the runtime
    # can connect to services running externally (e.g., on developer's machine).
    external_services: dict[str, list[str]] | None = None


@dataclass
class ServicesConfig:
    driver: ServiceConfig | None = MISSING
    renderer: ServiceConfig | None = MISSING
    physics: ServiceConfig | None = MISSING
    trafficsim: ServiceConfig | None = MISSING
    controller: ServiceConfig | None = MISSING
    runtime: RuntimeServiceConfig = MISSING
    prometheus: ContainerConfig = MISSING


@dataclass
class ContainerConfig:
    volumes: list[str] = field(default_factory=list)
    image: str = MISSING
    # Images that don't correspond to a service in the repo.
    # No Dockerfile path is added to the docker-compose.yaml.
    external_image: bool = False
    pull_policy: str = "missing"
    environments: list[str] = field(default_factory=list)
    workdir: str | None = None
    remap_root: bool = False
    # Optional per-container ext3 scratch for Apptainer, in MiB.
    apptainer_overlay_size_mb: int | None = None


@dataclass
class ServiceConfig(ContainerConfig):
    command: list[str] = MISSING
    # Number of service replicas to run per container.
    # If gpus is None or empty, creates a single container with this many replicas.
    # If gpus is specified, creates one container per GPU, each with this many replicas.
    replicas_per_container: int = MISSING
    gpus: list[int] | None = MISSING


@dataclass
class RuntimeServiceConfig(ServiceConfig):
    depends_on: list[str] = MISSING
