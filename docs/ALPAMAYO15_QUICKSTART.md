# Running Alpamayo 1.5 from scratch (Apptainer)

A single-scene smoke run of the Alpamayo 1.5 driving model, on a machine without Docker.
Roughly 90 minutes end to end, most of it downloads.

## 1. Prerequisites

- A GPU node with **≥ 48 GB VRAM** (A1.5 standard inference needs ~40 GB).
- `apptainer` (`module load apptainer` on clusters using environment modules).
- `uv` 0.9.17+ and a Rust toolchain (`setup_local_env.sh` can install Rust for you).
- A Hugging Face token, with access requested and approved for three gated repos:
  - `nvidia/Alpamayo-1.5-10B` — the model
  - `nvidia/Cosmos-Reason2-8B` — its VLM backbone
  - `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` — the scenes
- Network access to `nvcr.io`, for the NuRec renderer image. No NGC account needed — that
  image is public.

```bash
export HF_TOKEN=<your token>
```

## 2. Set up the host environment

The wizard runs on the host; only the services run in containers.

```bash
git clone git@github.com:NVlabs/alpasim.git && cd alpasim
source setup_local_env.sh          # compiles protos, installs the workspace
```

## 3. Download the model weights

`HF_TOKEN` from step 1 is all AlpaSim needs — no `hf auth login` required. Use the CLI from the
workspace venv rather than a system-wide one, which on a cluster is easily left pointing at a module
interpreter that a later software rebuild removes. **Run these from the repo root**, after step 2
has populated `.venv` — the path is relative, and an empty `.venv` has no `hf` in it:

```bash
cd /path/to/alpasim
.venv/bin/hf download nvidia/Alpamayo-1.5-10B      # ~21 GB, 5 safetensors shards
.venv/bin/hf download nvidia/Cosmos-Reason2-8B     # configs + tokenizer only; weights
                                                   # ship inside the Alpamayo shards
```

These land in `~/.cache/huggingface`, which the wizard bind-mounts into the driver container. The
driver will otherwise download them on first use, which often times out — fetch them up front.

> **Never run a bare `uv run <cmd>` or `uv sync` in this repo.** The root `pyproject.toml` declares
> no dependencies on purpose, so either command resyncs the workspace to *empty* and deletes every
> installed package. Always pass the extras (`uv run --extra all ...`, `uv sync --extra all`), or
> call binaries directly out of `.venv/bin/`.

## 4. Build the container images

The simulation image, built from `Apptainer.def` with no Docker daemon:

```bash
./build_apptainer.sh               # writes sif-cache/alpasim_base_<version>.sif
```

If your cluster forbids unprivileged builds (`--fakeroot` errors), convert a pre-built image
instead: `./build_apptainer.sh --from-registry docker://<registry>/alpasim-base:<version>`, or
`--from-archive` a `docker save` tarball. See [docs/APPTAINER.md](docs/APPTAINER.md).

The renderer image comes from NGC, and pulls anonymously — no login, no API key:

```bash
./build_apptainer.sh --from-registry docker://nvcr.io/nvidia/nre/nre-ga:26.04
```

This step is optional in principle: `wizard.apptainer.registry_fallback` is on by default, so the
wizard would pull `docker://` on demand. Do it up front anyway — the on-demand pull runs inside the
600 s service-startup timeout and usually exceeds it, and it caches ~12 GB into `~/.apptainer/cache`
unless you set `APPTAINER_CACHEDIR`. If you do skip it, pass `wizard.timeout=3600`.

If you ever need a *gated* `nvcr.io` image, authenticate once with
`apptainer remote login -u '$oauthtoken' docker://nvcr.io` and paste an NGC API key at the prompt;
it caches to `~/.apptainer/docker-config.json`. Do not put the key on the command line — it lands
in your shell history.

Check the version `services.renderer.image` expects in `src/wizard/configs/base_config.yaml` before
pulling; the entrypoint binary changes between NRE releases.

## 5. Run it

```bash
module load apptainer
uv run alpasim_wizard \
  deploy=local_apptainer topology=1gpu driver=alpamayo1_5 \
  wizard.log_dir=$PWD/a15_smoke
```

That runs the one default scene, downloading it if needed. Add `wizard.dry_run=true` to print the
commands without executing them.

The first rollout is slow — a 21 GB checkpoint loads before anything moves. Watch progress with:

```bash
tail -f a15_smoke/txt-logs/out-driver-0-log.txt     # model load, then inference
tail -f a15_smoke/txt-logs/out-runtime-0-log.txt    # rollout progress
```

Results land in `a15_smoke/`: `rollouts/`, `benchmark.json`, `metrics_plot.png`. The services shut
down when the runtime exits; `a15_smoke/run.sh` reproduces the whole deployment by hand.

## 6. Scale up

```bash
uv run alpasim_wizard \
  deploy=local_apptainer topology=1gpu driver=alpamayo1_5 \
  scenes.test_suite_id=public_2604 scenes.limit_to_first_n=20 \
  wizard.log_dir=$PWD/a15_run
```

Use `topology=8gpu_12rollouts` on a multi-GPU node. For a lighter single-camera run,
`driver=alpamayo1_5_1cam` renders one camera instead of four.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Could not find apptainer` | `module load apptainer`, or set `wizard.apptainer.binary=/abs/path`. |
| Build fails on `--fakeroot` | Cluster forbids unprivileged builds; use `--from-registry` / `--from-archive`. |
| Renderer exits instantly, "no such file" | Your NRE image predates the entrypoint rename. Add `defines.renderer_entrypoint=/app/internal/scripts/pycena/runtime/pycena_run`, or pull the version `base_config.yaml` expects. |
| `GatedRepoError` on scene download | Access not approved for the NuRec dataset, or `HF_TOKEN` unset. |
| `hf: bad interpreter: No such file or directory` | A `~/.local/bin` script pinned to a module Python that was removed in a cluster rebuild. Use `.venv/bin/hf`, or reinstall with `python3 -m pip install --user --force-reinstall huggingface_hub`. |
| `.venv/bin/hf: No such file or directory` | Either you are not in the repo root (the path is relative), or `.venv` is empty because step 2 never finished. Check with `ls .venv/lib/python3.12/site-packages \| wc -l` — three entries means empty. Re-run `source setup_local_env.sh`. |
| Imports suddenly fail across the workspace | A bare `uv run`/`uv sync` emptied the venv. Restore with `uv sync --extra all`. |
| CUDA OOM | Lower concurrency: `runtime.endpoints.driver.n_concurrent_rollouts=1`. Do not enable `driver.model.use_classifier_free_guidance_nav` — it needs ~60 GB. |
| Wizard hangs waiting for a service | Read that service's `txt-logs/out-<name>-0-log.txt`; the wizard waits on its port. |

---

### On this machine (`/project/AV/alpasim`)

Steps 1–4 are already done: weights are cached, `sif-cache/alpasim_base_0.111.0.sif` was built on
30 Jul, and 152 catalog scenes are local. The only gap is the renderer — you have `nre_ga_26.02.sif`
while 0.111.0 expects 26.04, so either pull 26.04 as in step 4, or run with the older entrypoint:

```bash
module load apptainer
cd /project/AV/alpasim
uv run alpasim_wizard \
  deploy=local_apptainer topology=1gpu driver=alpamayo1_5 \
  services.renderer.image=nvcr.io/nvidia/nre/nre-ga:26.02 \
  defines.renderer_entrypoint=/app/internal/scripts/pycena/runtime/pycena_run \
  wizard.log_dir=$PWD/a15_smoke
```

The default scene is already cached, so this run downloads nothing. Note the GPU here is a single
48 GB A6000: A1.5 fits, classifier-free-guidance navigation does not.
