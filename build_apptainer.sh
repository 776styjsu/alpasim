#!/usr/bin/env bash
# Build (or convert) an Apptainer image for AlpaSim. No Docker daemon required.
#
#   ./build_apptainer.sh                                   # build from Apptainer.def
#   ./build_apptainer.sh --from-registry docker://IMAGE     # convert a published image
#   ./build_apptainer.sh --from-archive alpasim.tar         # convert `docker save` output
#
# The image is written to ./sif-cache under the name the wizard looks for, so
# `deploy=local_apptainer` finds it without further configuration.
# See docs/APPTAINER.md.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

BUILD_FORMAT="sif"
SOURCE_KIND="def"
SOURCE_REF=""
OUTPUT_PATH=""
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/sif-cache}"
IMAGE_TAG=""
NETRC_PATH="${NETRC_PATH:-$HOME/.netrc}"
USE_FAKEROOT="auto"
TMPDIR_OVERRIDE=""
PRINT_IMAGE_NAME="no"
PRINT_CONTEXT_PATHS="no"

# Repo paths copied into a def build's context, and from there into the image.
# Mirrors the allowlist in .dockerignore; `test_build_context_covers_dockerignore`
# checks that it stays in step. Missing entries are skipped.
CONTEXT_PATHS=(Apptainer.def pyproject.toml uv.lock README.md src plugins e2e_challenge)

usage() {
    # Reprint this file's header comment block, minus the shebang.
    awk 'NR == 1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
    cat <<'EOF'

Options:
  -o, --output PATH       Full output path for the image
      --output-dir DIR    Directory to write the image into (default: ./sif-cache)
      --tag IMAGE         Docker-style image name to derive the filename from
                          (default: alpasim-base:<version in pyproject.toml>)
      --format FORMAT     sif (default) or sandbox (a directory; no mksquashfs
                          and no fakeroot needed for the final packing step)
      --from-def          Build from Apptainer.def (default)
      --from-registry REF Convert an existing image, e.g. docker://ghcr.io/org/img:tag
      --from-archive FILE Convert a tarball produced by `docker save`
      --netrc PATH        .netrc used for private dependencies during a def
                          build (default: ~/.netrc; never stored in the image)
      --fakeroot          Force --fakeroot
      --no-fakeroot       Never pass --fakeroot
      --tmpdir DIR        Scratch directory for the build (default: <output-dir>/tmp)
      --print-image-name  Print the filename the wizard expects, then exit
      --print-context-paths
                          Print the repo paths copied into the image, then exit
  -h, --help              Show this help

Environment:
  APPTAINER_BIN           Path to the apptainer/singularity binary
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -o|--output) OUTPUT_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tag) IMAGE_TAG="$2"; shift 2 ;;
        --format) BUILD_FORMAT="$2"; shift 2 ;;
        --from-def) SOURCE_KIND="def"; shift ;;
        --from-registry) SOURCE_KIND="registry"; SOURCE_REF="$2"; shift 2 ;;
        --from-archive) SOURCE_KIND="archive"; SOURCE_REF="$2"; shift 2 ;;
        --netrc) NETRC_PATH="$2"; shift 2 ;;
        --fakeroot) USE_FAKEROOT="yes"; shift ;;
        --no-fakeroot) USE_FAKEROOT="no"; shift ;;
        --tmpdir) TMPDIR_OVERRIDE="$2"; shift 2 ;;
        --print-image-name) PRINT_IMAGE_NAME="yes"; shift ;;
        --print-context-paths) PRINT_CONTEXT_PATHS="yes"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$BUILD_FORMAT" in
    sif|sandbox) ;;
    *) echo "ERROR: --format must be 'sif' or 'sandbox', got '$BUILD_FORMAT'." >&2; exit 2 ;;
esac

# Size of a directory. Uses apparent size because block accounting lags behind
# fresh writes on GPFS and other delayed-allocation filesystems, which makes a
# just-staged context look far smaller than it is.
dir_size() {
    local out
    out="$(du -sh --apparent-size "$1" 2>/dev/null)" || out="$(du -sh "$1" 2>/dev/null)"
    printf '%s' "${out%%$'\t'*}"
}

repo_version() {
    sed -n 's/^version = "\(.*\)"$/\1/p' "$REPO_ROOT/pyproject.toml" | head -n 1
}

# Canonical image filename for a docker-style image name. Must stay in sync with
# `image_to_apptainer_basename` in src/wizard/alpasim_wizard/utils.py; the test
# `test_apptainer_image_name_matches_build_script` checks that it does.
image_basename() {
    local image="$1" stem
    stem="${image##*/}"
    stem="${stem//:/_}"
    stem="${stem//-/_}"
    if [ "$BUILD_FORMAT" = "sandbox" ]; then
        printf '%s.sandbox\n' "$stem"
    else
        printf '%s.sif\n' "$stem"
    fi
}

if [ -z "$IMAGE_TAG" ]; then
    if [ "$SOURCE_KIND" = "registry" ]; then
        # Name the output after the image being converted, so the wizard finds
        # it under the reference it resolves.
        IMAGE_TAG="${SOURCE_REF#*://}"
    else
        IMAGE_TAG="alpasim-base:$(repo_version)"
    fi
fi

if [ "$PRINT_IMAGE_NAME" = "yes" ]; then
    image_basename "$IMAGE_TAG"
    exit 0
fi

if [ "$PRINT_CONTEXT_PATHS" = "yes" ]; then
    printf '%s\n' "${CONTEXT_PATHS[@]}"
    exit 0
fi

if [ -z "$OUTPUT_PATH" ]; then
    OUTPUT_PATH="$OUTPUT_DIR/$(image_basename "$IMAGE_TAG")"
fi
OUTPUT_DIR="$(dirname -- "$OUTPUT_PATH")"

find_apptainer_bin() {
    if [ -n "${APPTAINER_BIN:-}" ]; then
        printf '%s\n' "$APPTAINER_BIN"
        return 0
    fi
    local candidate
    for candidate in apptainer singularity; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    # Environment modules: load only the unversioned name, so no particular
    # cluster's version is baked in here.
    if type module >/dev/null 2>&1; then
        module load apptainer >/dev/null 2>&1 || true
        if command -v apptainer >/dev/null 2>&1; then
            command -v apptainer
            return 0
        fi
    fi
    return 1
}

if ! APPTAINER_BIN="$(find_apptainer_bin)"; then
    cat >&2 <<'EOF'
ERROR: Could not find apptainer (or singularity) on PATH.
Load it first, or point at it explicitly:
  module load apptainer          # on clusters using environment modules
  export APPTAINER_BIN=/path/to/apptainer
EOF
    exit 127
fi

# Keep scratch and cache off /tmp: it is a memory-backed tmpfs on many clusters,
# where unpacking a multi-gigabyte image gets the build OOM-killed.
export APPTAINER_TMPDIR="${TMPDIR_OVERRIDE:-$OUTPUT_DIR/tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$OUTPUT_DIR/cache}"
mkdir -p "$OUTPUT_DIR" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

BUILD_ARGS=()
if [ "$BUILD_FORMAT" = "sandbox" ]; then
    BUILD_ARGS+=(--sandbox)
fi

# Only a def build runs %post as root and therefore needs fakeroot; converting
# an existing image does not.
if [ "$USE_FAKEROOT" = "yes" ] ||
   { [ "$USE_FAKEROOT" = "auto" ] && [ "$SOURCE_KIND" = "def" ] && [ "$(id -u)" -ne 0 ]; }; then
    BUILD_ARGS+=(--fakeroot)
fi

echo "=== AlpaSim Apptainer build ==="
echo "  apptainer: $APPTAINER_BIN"
echo "  source:    $SOURCE_KIND${SOURCE_REF:+ ($SOURCE_REF)}"
echo "  image:     $OUTPUT_PATH"
echo "  format:    $BUILD_FORMAT"

BUILD_CONTEXT_DIR=""
cleanup() {
    if [ -n "$BUILD_CONTEXT_DIR" ] && [ -d "$BUILD_CONTEXT_DIR" ]; then
        rm -rf "$BUILD_CONTEXT_DIR"
    fi
}
trap cleanup EXIT

case "$SOURCE_KIND" in
    registry)
        BUILD_TARGET="$SOURCE_REF"
        case "$BUILD_TARGET" in
            *://*) ;;
            *) BUILD_TARGET="docker://$BUILD_TARGET" ;;
        esac
        BUILD_CWD="$REPO_ROOT"
        ;;
    archive)
        if [ ! -f "$SOURCE_REF" ]; then
            echo "ERROR: archive not found: $SOURCE_REF" >&2
            exit 2
        fi
        BUILD_TARGET="docker-archive:$(cd -- "$(dirname -- "$SOURCE_REF")" && pwd)/$(basename -- "$SOURCE_REF")"
        BUILD_CWD="$REPO_ROOT"
        ;;
    def)
        if ! command -v rsync >/dev/null 2>&1; then
            echo "ERROR: rsync is required to stage the build context." >&2
            echo "Install rsync, or convert a pre-built image with --from-registry/--from-archive." >&2
            exit 127
        fi
        # Apptainer.def copies the build context into the image, so stage a
        # copy holding only what the image needs. This is an allowlist for the
        # same reason .dockerignore is one: a working repo accumulates
        # datasets, run outputs, virtualenvs and caches next to the sources,
        # and excluding them by name never keeps up.
        BUILD_CONTEXT_DIR="$(mktemp -d "$APPTAINER_TMPDIR/alpasim-context-XXXXXX")"
        echo "Staging build context in $BUILD_CONTEXT_DIR"
        for path in "${CONTEXT_PATHS[@]}"; do
            [ -e "$REPO_ROOT/$path" ] || continue
            # Regenerable build output, matching .dockerignore. `utils_rs/target`
            # is host-architecture Rust output that `uv sync` rebuilds in the
            # image; leaving it in makes up most of the context size.
            rsync -a \
                --exclude '.venv' \
                --exclude '__pycache__' \
                --exclude '.mypy_cache' \
                --exclude '.pytest_cache' \
                --exclude '.ruff_cache' \
                --exclude '*.egg-info' \
                --exclude 'build' \
                --exclude 'dist' \
                --exclude 'outputs' \
                --exclude 'runs' \
                --exclude 'test_output' \
                --exclude 'runtime/build.env' \
                --exclude 'utils_rs/target' \
                --exclude '*.sif' \
                --exclude '*.sqsh' \
                "$REPO_ROOT/$path" "$BUILD_CONTEXT_DIR/"
        done
        echo "Build context staged ($(dir_size "$BUILD_CONTEXT_DIR"))"

        if [ -f "$NETRC_PATH" ]; then
            echo "Binding $NETRC_PATH read-only for private dependencies"
            BUILD_ARGS+=(--bind "${NETRC_PATH}:/run/netrc:ro")
        else
            echo "NOTE: $NETRC_PATH not found; private dependencies will be skipped."
        fi

        BUILD_TARGET="Apptainer.def"
        BUILD_CWD="$BUILD_CONTEXT_DIR"
        ;;
esac

# Build to scratch first, so a failed build leaves any previous image intact.
STAGED_OUTPUT="$APPTAINER_TMPDIR/$(basename -- "$OUTPUT_PATH")"
rm -rf "$STAGED_OUTPUT"

set -x
if ! (cd "$BUILD_CWD" && "$APPTAINER_BIN" build "${BUILD_ARGS[@]}" "$STAGED_OUTPUT" "$BUILD_TARGET"); then
    set +x
    if [ "$SOURCE_KIND" = "def" ]; then
        cat >&2 <<'EOF'

The def-file build failed. Building from a definition file needs root or a
working `--fakeroot` setup (unprivileged user namespaces plus subuid/subgid
mappings), which many clusters disable. Alternatives:
  * --from-registry docker://<image>   convert an already-published image
  * --from-archive <file.tar>          convert `docker save` output copied in
  * --format sandbox                   avoids the SIF packing step
  * apptainer remote build ...         build on a remote builder service
EOF
    fi
    exit 1
fi
set +x

rm -rf "$OUTPUT_PATH"
mv "$STAGED_OUTPUT" "$OUTPUT_PATH"

cat <<EOF

=== Done ===
Image: $OUTPUT_PATH

Smoke test:
  $APPTAINER_BIN exec --nv "$OUTPUT_PATH" uv run python -c 'import torch; print(torch.cuda.is_available())'

Run a simulation:
  uv run alpasim_wizard deploy=local_apptainer topology=1gpu driver=vavam \\
    wizard.log_dir=\$PWD/out
EOF
