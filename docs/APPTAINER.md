# Running AlpaSim with Apptainer

[Apptainer](https://apptainer.org) (formerly Singularity) runs containers unprivileged, as the
calling user. Use this deployment when Docker is unavailable — typically on HPC clusters, shared
workstations, or anywhere you cannot join the `docker` group.

Everything else works as usual: the wizard composes the same services from the same configs, and
`deploy=local_apptainer` swaps only the container runtime.

```bash
./build_apptainer.sh                                    # build the image into ./sif-cache
uv run alpasim_wizard deploy=local_apptainer topology=1gpu driver=vavam \
  wizard.log_dir=$PWD/out
```

## Getting an image

The wizard looks for images in `wizard.apptainer.image_caches` under a name derived from the image
reference: `:` and `-` become `_`, and the extension is `.sif` (or `.sandbox`). So
`alpasim-base:0.111.0` is looked up as `alpasim_base_0.111.0.sif`. `build_apptainer.sh` writes
exactly that name into `./sif-cache`, which is what `deploy=local_apptainer` searches, so the
default flow needs no configuration. Print the expected name with
`./build_apptainer.sh --print-image-name`.

There are three ways to get an image, in decreasing order of what the cluster must allow:

```bash
# 1. Build from Apptainer.def. Needs root or a working --fakeroot setup.
./build_apptainer.sh

# 2. Convert an image from a registry. Needs no fakeroot, only network access.
./build_apptainer.sh --from-registry docker://<registry>/alpasim-base:<version>

# 3. Convert a `docker save` tarball, for air-gapped clusters. Build the Docker
#    image on a machine that has Docker, then copy the tarball over. Add
#    --tag <image> if it is not alpasim-base at this repo's version.
docker save alpasim-base:<version> -o alpasim.tar
./build_apptainer.sh --from-archive alpasim.tar
```

If SIF creation itself fails (it needs `mksquashfs` and enough scratch space), build a sandbox
directory instead — the wizard runs those too:

```bash
./build_apptainer.sh --format sandbox
```

`Apptainer.def` mirrors the `Dockerfile`, and a test keeps the base image and package list in sync.
Two Docker features have no def-file equivalent and are intentionally absent: multi-arch selection
(the def builds x86_64; on aarch64 convert a Docker image instead) and the `dcgm-exporter` binary
copied from an NVIDIA image, without which telemetry simply reports no GPU metrics.

Private dependencies are read from `~/.netrc`, bind-mounted read-only during the build and deleted
before the image is finalized, so credentials are never stored in the image. Override the path with
`--netrc`.

## Configuration

All settings live under `wizard.apptainer`; the defaults are site-neutral, so anything specific to
a cluster is opt-in.

| Key | Default | Purpose |
| --- | --- | --- |
| `image_caches` | `[]` (`sif-cache` in the deploy profile) | Directories searched for `.sif` files or sandbox directories. |
| `registry_fallback` | `true` | Run `docker://<image>` when no cached image matches. Set `false` on air-gapped clusters to fail fast with the paths searched. |
| `binary` | `apptainer` | Absolute path to the binary when it is not on `PATH`. |
| `extra_exec_args` | `[]` | Extra flags for every `apptainer exec`, e.g. `[--containall]`. |
| `environments` | `UV_PROJECT_ENVIRONMENT`, `PYTHONDONTWRITEBYTECODE` | Environment for every container. `VAR=value` sets a value; a bare `VAR` passes the host's value through. |
| `workdir` | `/repo` | Working directory for services running an alpasim image that do not set `workdir` themselves. Services with `external_image` run from `/` unless they set one. |
| `writable_tmpfs` | `true` | Writable in-memory layer. Disable where Apptainer cannot set up overlays. |
| `overlay_image_patterns` | `[]` | Images matching these substrings get a file-backed ext3 overlay instead of tmpfs. |
| `overlay_size_mb` | `4096` | Size of each of those overlays. |

On clusters using environment modules, `module load apptainer` in your job script before starting
the wizard is enough. Do not rely on the wizard loading it for you: `module` is a shell function
that often does not exist in the non-interactive shell services are dispatched through. Set
`binary` to an absolute path if that is a problem.

## On Slurm

Apptainer needs no special Slurm integration — it runs inside your allocation as an ordinary
process. Load the module (or set `wizard.apptainer.binary`) and run the wizard on the allocated
node:

```bash
#!/bin/bash
#SBATCH --gpus=1 --nodes=1 --time=02:00:00
module load apptainer
uv run alpasim_wizard deploy=local_apptainer topology=1gpu driver=vavam \
  wizard.log_dir=$PWD/out-$SLURM_JOB_ID
```

Use `run_method: SLURM` instead when your cluster provides enroot/pyxis and you want the wizard to
launch each service as its own job step; see the `sqshcaches` settings.

Every run also writes `<log_dir>/run.sh`, a standalone script that starts the services, waits for
their ports, runs the runtime, and cleans up on exit. It is useful for submitting a prepared run
(`wizard.dry_run=true` generates it without executing anything) or for reproducing one by hand.

## Differences from the Docker backends

* **Nothing runs as root.** Files written to bind mounts are owned by you, so the umask workaround
  the Docker Compose backend needs does not apply here.
* **The image's `WORKDIR`, `USER` and `ENV` are ignored** by Apptainer. The wizard passes `--pwd`
  and `--env` explicitly, which is why `workdir` and `environments` exist above. Where Docker falls
  back to the image's `WORKDIR`, Apptainer needs a directory that exists in the image, so a service
  running a third-party image needs its own `workdir` if it cares about the working directory.
* **Services are children of the wizard.** They are started in the background and terminated when
  the runtime exits; there is no daemon that keeps them alive afterwards.
* **Per-service logs** are written to `<log_dir>/txt-logs/out-<service>-log.txt`.
* **Telemetry never blocks the run.** The Prometheus sidecar starts alongside the services, but the
  simulation does not wait for it.

## Troubleshooting

**`Could not find apptainer (or singularity) on PATH`** — `module load apptainer`, or set
`APPTAINER_BIN` for the build script and `wizard.apptainer.binary` for the wizard.

**The def-file build fails with a fakeroot or user-namespace error** — the cluster does not allow
unprivileged def builds. Convert a pre-built image instead (`--from-registry`, `--from-archive`), or
use `apptainer remote build`.

**The build is killed, or fails with `no space left on device`** — `/tmp` is often a small,
memory-backed tmpfs. `build_apptainer.sh` already redirects Apptainer's scratch and cache into the
output directory; use `--tmpdir` to point somewhere with more room.

**A service fails writing to a path that is not bind-mounted** — its writable layer is too small.
Add the image to `overlay_image_patterns` to give it a file-backed overlay, and raise
`overlay_size_mb` if needed. File-backed overlays also work around shared filesystems such as GPFS
and Lustre, where directory overlays fail because overlayfs upper layers need xattr support the
filesystem does not provide.

**`FATAL: while mounting ...: destination is already in use` or a missing mount point** — the
target path does not exist inside the image and Apptainer could not create it. Setting
`writable_tmpfs: true` (the default) usually resolves this.

**GPU code does not see the device** — services get `--nv` only when the topology assigns them a
GPU. Check `services.<name>.gpus` in your topology config, and confirm the host driver is visible
with `apptainer exec --nv <image> nvidia-smi`.

**A registry pull fails to authenticate** — export `APPTAINER_DOCKER_USERNAME` and
`APPTAINER_DOCKER_PASSWORD`, or pull once by hand into the cache directory.
