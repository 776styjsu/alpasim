# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Guards against drift between Apptainer.def, the Dockerfile and the wizard.

The Apptainer image has to provide the same environment as the Docker image, but
Apptainer definition files cannot reuse the Dockerfile: BuildKit secrets, cache
mounts, multi-arch selection and `COPY --from` have no direct equivalent. These
tests keep the unavoidable duplication honest instead of silently stale.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from alpasim_utils.paths import find_repo_root
from alpasim_wizard.schema import RunMethod
from alpasim_wizard.utils import image_to_apptainer_basename
from hydra import compose, initialize_config_dir

REPO_ROOT = find_repo_root(__file__)
DOCKERFILE = REPO_ROOT / "Dockerfile"
APPTAINER_DEF = REPO_ROOT / "Apptainer.def"
BUILD_SCRIPT = REPO_ROOT / "build_apptainer.sh"
CONFIG_DIR = REPO_ROOT / "src" / "wizard" / "configs"

# Packages the Docker image installs that the Apptainer image deliberately does
# not. Keep the reason with the entry; anything else must exist in both.
DOCKER_ONLY_PACKAGES = {
    # Needed only by dcgm-exporter, whose binary is copied from another Docker
    # image; without that binary the telemetry sidecar skips GPU metrics.
    "datacenter-gpu-manager-4-cuda12",
}


def _apt_packages(text: str) -> set[str]:
    """Collect package names from `apt-get install` blocks in a build file.

    The list ends at the next `&&`, so the shell commands both files chain after
    the install (symlinking, cache cleanup) are not mistaken for packages.
    """
    packages: set[str] = set()
    for match in re.finditer(r"apt-get install\s+(.*?)(?:&&|\n\s*\n|$)", text, re.S):
        for token in match.group(1).replace("\\", " ").split():
            if re.fullmatch(r"[a-z0-9][a-z0-9.+-]*", token):
                packages.add(token)
    return packages


def test_apptainer_definition_matches_dockerfile_base_image() -> None:
    dockerfile = DOCKERFILE.read_text()
    definition = APPTAINER_DEF.read_text()

    docker_base = re.search(r"^FROM (\S+) AS base-amd64$", dockerfile, re.M)
    apptainer_base = re.search(r"^From: (\S+)$", definition, re.M)

    assert docker_base is not None, "Dockerfile has no base-amd64 stage"
    assert apptainer_base is not None, "Apptainer.def has no From: line"
    assert apptainer_base.group(1) == docker_base.group(1), (
        "Apptainer.def builds on a different base image than the Dockerfile; "
        "update Apptainer.def."
    )


def test_apptainer_definition_matches_dockerfile_packages() -> None:
    docker_packages = _apt_packages(DOCKERFILE.read_text())
    apptainer_packages = _apt_packages(APPTAINER_DEF.read_text())

    assert docker_packages, "no apt packages parsed from the Dockerfile"
    missing = docker_packages - apptainer_packages - DOCKER_ONLY_PACKAGES
    assert not missing, (
        f"Apptainer.def is missing packages the Dockerfile installs: {sorted(missing)}. "
        "Add them, or document the omission in DOCKER_ONLY_PACKAGES."
    )


def test_apptainer_definition_keeps_credentials_out_of_the_image() -> None:
    definition = APPTAINER_DEF.read_text()

    assert "/run/netrc" in definition, "no bind-mounted credential path"
    assert "rm -f /root/.netrc" in definition, (
        "Apptainer.def must delete the copied .netrc, or credentials end up in "
        "the distributed image."
    )


@pytest.mark.parametrize(
    ("image", "extra_args"),
    [
        ("alpasim-base:0.1.0", []),
        ("nvcr.io/org/my-image:1.2.3", []),
        ("alpasim-base:0.1.0", ["--format", "sandbox"]),
    ],
)
def test_build_script_names_images_the_way_the_wizard_looks_them_up(
    image: str, extra_args: list[str]
) -> None:
    """The build output must land on the name `image_caches` lookups expect."""
    result = subprocess.run(
        ["bash", str(BUILD_SCRIPT), "--print-image-name", "--tag", image, *extra_args],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )

    expected = image_to_apptainer_basename(image)
    if "sandbox" in extra_args:
        expected = expected.replace(".sif", ".sandbox")
    assert result.stdout.strip() == expected


def test_build_context_covers_dockerignore_allowlist() -> None:
    """A def build must ship everything the Docker build context ships.

    Both are allowlists because a working repo accumulates datasets, outputs and
    caches beside the sources; if .dockerignore starts un-ignoring a new path,
    the Apptainer context has to follow or the image silently loses it.
    """
    dockerignore = (REPO_ROOT / ".dockerignore").read_text()
    allowed_roots = {
        line[1:].strip().split("/")[0]
        for line in dockerignore.splitlines()
        if line.startswith("!")
    }

    staged = set(
        subprocess.run(
            ["bash", str(BUILD_SCRIPT), "--print-context-paths"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.split()
    )

    missing = allowed_roots - staged
    assert not missing, (
        f".dockerignore ships {sorted(missing)}, which build_apptainer.sh does "
        "not stage. Add them to CONTEXT_PATHS."
    )


def test_build_script_is_executable() -> None:
    assert BUILD_SCRIPT.stat().st_mode & 0o111, "build_apptainer.sh is not executable"


def test_default_build_output_is_where_the_deploy_profile_looks() -> None:
    """`./build_apptainer.sh` then `deploy=local_apptainer` must work as-is."""
    import alpasim_wizard.setup_omegaconf  # noqa: F401  (registers resolvers)

    with initialize_config_dir(
        config_dir=str(CONFIG_DIR), version_base="1.3", job_name="apptainer_test"
    ):
        cfg = compose(
            config_name="base_config",
            overrides=[
                "deploy=local_apptainer",
                "topology=1gpu",
                "driver=vavam",
                "wizard.log_dir=/tmp/alpasim-apptainer-test",
            ],
        )

    assert cfg.wizard.run_method == RunMethod.APPTAINER

    built = subprocess.run(
        ["bash", str(BUILD_SCRIPT), "--print-image-name"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    ).stdout.strip()

    cache = Path(cfg.wizard.apptainer.image_caches[0])
    expected = image_to_apptainer_basename(cfg.services.runtime.image)
    assert built == expected, (
        "build_apptainer.sh names the default image differently than the wizard "
        "resolves it"
    )
    # ... and writes it into the directory the profile searches.
    assert cache == REPO_ROOT / "sif-cache"


def test_source_build_includes_docker_workspace_extras() -> None:
    docker_extras = set(re.findall(r"--extra (\w+)", DOCKERFILE.read_text()))
    native_extras = set(re.findall(r"--extra (\w+)", APPTAINER_DEF.read_text()))
    assert native_extras == docker_extras


def test_build_requires_explicit_source() -> None:
    result = subprocess.run(["bash", str(BUILD_SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "--from-registry" in result.stderr


@pytest.mark.parametrize("source", ["registry", "archive"])
def test_conversion_respects_relative_output_and_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    import json
    import sys

    binary = tmp_path / "fake-apptainer"
    record = tmp_path / "build.json"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        f"pathlib.Path({str(record)!r}).write_text(json.dumps({{\n"
        "    'args': sys.argv[1:], 'tmpdir': os.environ['APPTAINER_TMPDIR']\n"
        "}))\n"
        "pathlib.Path(sys.argv[-2]).write_text('converted')\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("APPTAINER_BIN", str(binary))
    monkeypatch.setenv("APPTAINER_TMPDIR", "scratch")
    archive = tmp_path / "image.tar"
    archive.touch()
    reference = "registry.example/team/image:1"
    source_args = (
        ["--from-registry", reference]
        if source == "registry"
        else ["--from-archive", archive.name, "--tag", reference]
    )
    subprocess.run(
        ["bash", str(BUILD_SCRIPT), *source_args, "--output-dir", "images"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(record.read_text())
    assert result["tmpdir"] == str(tmp_path / "scratch")
    assert "--fakeroot" not in result["args"]
    expected_source = (
        f"docker://{reference}" if source == "registry" else f"docker-archive:{archive}"
    )
    assert result["args"][-1] == expected_source
    assert (
        tmp_path / "images" / image_to_apptainer_basename(reference)
    ).read_text() == "converted"
    assert not list((tmp_path / "scratch").glob("alpasim-build-*"))
