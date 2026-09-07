# Running AlpaSim with Apptainer

[Apptainer](https://apptainer.org) runs containers as the calling user without a Docker daemon.
The `deploy=local_apptainer` profile runs the usual AlpaSim services on one host, including inside
an existing Slurm allocation. Module names, storage paths, and scheduler options belong in your
site's job script or deployment config.

## Prepare images, then run

The recommended workflow uses published Docker/OCI images. Apptainer downloads and converts
these itself; Docker is not required on the machine doing the conversion or running the simulation.
Prepare images before submitting a job so startup does not depend on registry access.

```bash
# Replace the registry and version with the image you have access to.
apptainer pull alpasim.sif docker://<registry>/alpasim-base:<version>
# Use the renderer version selected by your service configuration.
apptainer pull renderer.sif docker://nvcr.io/nvidia/nre/nre-ga:26.04

uv run alpasim_wizard deploy=local_apptainer topology=1gpu driver=vavam \
  defines.base_image=$PWD/alpasim.sif \
  services.renderer.image=$PWD/renderer.sif \
  wizard.log_dir=$PWD/out
```

`defines.base_image` selects the shared AlpaSim image. Any `services.<name>.image` can instead be
an absolute or relative SIF path, a sandbox directory, or an explicit Apptainer URI such as
`docker://...`, `oras://...`, or `library://...`. Relative paths resolve when the wizard prepares
the run. A missing local path fails immediately.

For registry authentication, use `apptainer registry login`, or set
`APPTAINER_DOCKER_USERNAME` and `APPTAINER_DOCKER_PASSWORD` before pulling.
See [Apptainer's OCI guide](https://apptainer.org/docs/user/latest/docker_and_oci.html).

## Optional image cache and source builds

The build helper can prepare images in `sif-cache/`, which the deployment profile searches:

```bash
./build_apptainer.sh --from-registry docker://<registry>/alpasim-base:<version>
./build_apptainer.sh --from-archive alpasim.tar --tag <registry>/alpasim-base:<version>
```

Use the same complete image reference in the service configuration as in the build command.
Cache filenames contain a readable basename and a hash of the full reference, so registries,
repositories, and tags cannot accidentally share a cache entry. Inspect the filename with:

```bash
./build_apptainer.sh --print-image-name --tag <registry>/alpasim-base:<version>
```

Existing images with older cache names remain usable by setting their paths directly in
`defines.base_image` or `services.<name>.image`. Automatic cache lookup uses only the new names.

If you need to build from source entirely without Docker, the optional native recipe is available:

```bash
./build_apptainer.sh --from-def
# To build a directory instead of a SIF:
./build_apptainer.sh --from-def --format sandbox
```

Definition builds require root or working fakeroot support. `Apptainer.def` builds x86_64 and
installs the same core and recipes dependencies as the Dockerfile. It omits the Docker image's
DCGM exporter; convert a Docker image if you need that exporter or an aarch64 image.
The helper stages the source tree, binds `~/.netrc` read-only for private dependencies, and removes
the copied credentials before finalizing the image. Use `--netrc` to select another credential file.

The helper respects `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR`; `--tmpdir` overrides the former.
Otherwise scratch and cache live under the output directory. Choose storage with enough space
for the unpacked image and build artifacts. `--output`, `--output-dir`, and `--format sandbox`
work for conversions as well as native builds. Run `./build_apptainer.sh --help` for details.

## Configuration

General runtime settings live under `wizard.apptainer`:

| Key | Default | Purpose |
| --- | --- | --- |
| `binary` | `apptainer` | Executable name or absolute path. |
| `image_caches` | `[]`; `sif-cache/` in the profile | Optional directories containing images prepared by the helper. |
| `registry_fallback` | `true` | Allow implicit pulls for bare OCI references absent from the cache. Explicit URIs remain explicit pull requests. |
| `cleanenv` | `true` | Exclude inherited host/module variables; preserve image variables and explicit overrides. |
| `environments` | `[PYTHONDONTWRITEBYTECODE=1]` | Additional variables for every service; `VAR=value` sets a value and bare `VAR` passes the host value. |
| `extra_exec_args` | `[]` | Additional Apptainer arguments, with each argument a separate list element. |
| `workdir` | `/repo` | Default for AlpaSim images; external images use `/`. A service's own `workdir` takes precedence. |
| `writable_tmpfs` | `true` | Temporary writable layer for paths outside bind mounts. Disable where overlays are unavailable. |

Image `ENV` values are preserved. `services.<name>.environments` overrides global additions.
The launcher uses `--no-eval` so environment values are not evaluated as shell expressions inside
the container. With `cleanenv`, explicitly list any host variables a service needs, such as
`HF_TOKEN`. AlpaSim image recipes set `UV_PROJECT_ENVIRONMENT=/repo/.venv` and install managed
Python under `/opt/python`, accessible to the unprivileged caller.

For a service that needs more writable space than tmpfs permits, set a size in MiB:

```bash
uv run alpasim_wizard deploy=local_apptainer topology=1gpu driver=vavam \
  services.renderer.apptainer_overlay_size_mb=4096 wizard.log_dir=$PWD/out
```

This creates one ext3 overlay per container under `<log_dir>/apptainer-overlays/`, reused when
rerunning that deployment. It takes precedence over `writable_tmpfs`. The setting defaults to
`null`; there is no image-name matching. To migrate `overlay_image_patterns` / `overlay_size_mb`,
set `services.<name>.apptainer_overlay_size_mb` for each service that requires an overlay.

## On Slurm

Load the site's module and start the wizard on an allocated compute node:

```bash
#!/bin/bash
#SBATCH --gpus=1 --nodes=1 --time=02:00:00
module load apptainer
uv run alpasim_wizard deploy=local_apptainer topology=1gpu driver=vavam \
  defines.base_image=$PWD/alpasim.sif \
  services.renderer.image=$PWD/renderer.sif \
  wizard.log_dir=$PWD/out-$SLURM_JOB_ID
```

Add the account, partition, CPU, and memory requests required by your site. The backend does not
submit jobs or distribute services across nodes. It needs Bash 4.4 or newer and `setsid` on `PATH`.
The wizard does not load environment modules; alternatively set `wizard.apptainer.binary`.

Topology GPU indices select entries from the execution environment's `CUDA_VISIBLE_DEVICES`.
For example, index `0` selects device `3` when the allocation exposes `3,5`; GPU UUIDs also work.
Without that variable, indices are used directly. Selection happens when `run.sh` executes, so
a script prepared on a login node uses the compute node's allocation. Out-of-range indices fail.

## Lifecycle and logs

The wizard writes `<log_dir>/run.sh` and executes that same script. `wizard.dry_run=true` writes
it without starting services or creating overlays. The script can also run directly, or be
submitted with `sbatch` and your site's resource options.

Services start in separate process groups. Readiness checks fail on startup exit or timeout.
The runtime's exit status determines success. Without a runtime, the script keeps serving until
a simulation service exits or the run is interrupted. Telemetry starts alongside the services,
but its readiness and exit status do not gate the simulation.

Normal completion, SIGINT, and SIGTERM trigger cleanup of all launched groups, including the
runtime. After a ten-second grace period, remaining processes receive SIGKILL. Per-service output
is appended to `<log_dir>/txt-logs/out-<service>-log.txt` for both wizard and standalone runs.

If GPU initialization fails, check the allocation's visible devices and run
`apptainer exec --nv <image> nvidia-smi` on the allocated node. For write failures, check bind
permissions and choose tmpfs or a service overlay large enough for the workload.
