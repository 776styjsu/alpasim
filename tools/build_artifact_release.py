#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Build a clean Drive the Thoughts artifact staging directory.

The artifact is intentionally data-first: it packages the canonical benchmark,
released monitor outputs, media referenced by the benchmark, and the code needed
to recompute tables from those JSON outputs. It also normalizes private absolute
paths into artifact-relative paths and validates that the staged files are
internally consistent.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PRIVATE_ROOTS = (
    "/sfs/gpfs/tardis/project/AV/alpasim",
    "/project/AV/alpasim",
)
PRIVATE_MARKERS = (
    "/sfs/",
    "/project/",
    "/home/",
)
STALE_BENCHMARK_MARKERS = (
    "benchmark_expanded_100",
    "benchmark_expanded_50",
)

TOOL_FILES = (
    "tools/check_consistency_accuracy.py",
    "tools/build_consistency_spreadsheet.py",
    "tools/plot_repeated_main_eval.py",
    "tools/plot_cot_reliability_heatmap.py",
    "tools/run_benchmark_llm_matrix.py",
    "tools/run_benchmark_llm_matrix_update.slurm.sh",
    "tools/export_benchmark_frame_folders.py",
    "tools/serve_qwen35_4b_fp8_vllm.sh",
    "tools/serve_qwen3_4b_fp8_vllm.sh",
)

SOURCE_DIRS = (
    "src/tools/trajectory_safety",
    "src/utils/alpasim_utils/cot_consistency",
)

SOURCE_FILES = (
    "src/tools/cot_analysis/__init__.py",
    "src/tools/cot_analysis/__main__.py",
    "src/tools/cot_analysis/consistency_check.py",
    "src/tools/cot_analysis/prompt.py",
    "src/tools/cot_analysis/prompt_center_of_lane_v5.py",
    "src/tools/cot_analysis/trajectory.py",
    "src/utils/alpasim_utils/__init__.py",
    "src/utils/alpasim_utils/consistency.py",
    "src/utils/alpasim_utils/cot_meta_actions.py",
    "src/utils/alpasim_utils/lane_projection.py",
    "src/utils/alpasim_utils/meta_actions_types.py",
    "src/utils/alpasim_utils/trajectory_additional_info.py",
)

TEST_FILES = (
    "src/tools/tests/test_cot_analysis_trajectory.py",
    "src/tools/tests/test_trajectory_safety_outcomes.py",
    "src/utils/tests/test_cot_consistency.py",
    "src/utils/tests/test_lane_projection.py",
    "src/utils/tests/test_trajectory_additional_info.py",
)

SCENE_SIDE_FILES = (
    "metadata.json",
    "additional_info.json",
    "trajectory_plot_geometry.json",
    "trajectory_plot.png",
    "camera_front_wide_120fov.jpg",
)

ARTIFACT_REQUIREMENTS = (
    "numpy",
    "shapely",
    "Pillow",
    "matplotlib",
    "pytest",
)

ARTIFACT_CONDA_REQUIREMENTS = (
    "python>=3.11,<3.13",
    "numpy",
    "shapely",
    "pillow",
    "matplotlib",
    "pytest",
)

ARTIFACT_PYPROJECT = """[project]
name = "drive-the-thoughts-artifact"
version = "0.1.0"
description = "Lightweight reproduction environment for the Drive the Thoughts artifact"
readme = "README.md"
requires-python = ">=3.11,<3.13"
dependencies = [
  "numpy",
  "shapely",
  "Pillow",
  "matplotlib",
  "pytest",
]

[tool.uv]
package = false

[tool.pytest.ini_options]
pythonpath = ["src/utils", "src/tools"]
testpaths = ["src/utils/tests", "src/tools/tests"]
"""

ARTIFACT_CONDA_ENVIRONMENT = (
    "name: drive-the-thoughts-artifact\n"
    "channels:\n"
    "  - conda-forge\n"
    "dependencies:\n"
    + "".join(f"  - {dep}\n" for dep in ARTIFACT_CONDA_REQUIREMENTS)
)

ALPASIM_TOP_LEVEL_FILES = (
    ".dockerignore",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".python-version",
    "Apptainer.def",
    "build_apptainer.sh",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "Dockerfile",
    "LICENSE",
    "mypy.ini",
    "pyproject.toml",
    "README.md",
    "SECURITY.md",
    "setup_local_env.sh",
    "uv.lock",
)

ALPASIM_TREE_DIRS = (
    "docs",
    "plugins",
    "src",
    "tools",
)

ALPASIM_DATA_FILES = (
    "data/README.md",
    "data/auto-init.sh",
    "data/download_vavam_assets.sh",
    "data/scenes/README.md",
    "data/scenes/sim_scenes.csv",
    "data/scenes/sim_suites.csv",
)

ALPASIM_DATA_DIRS = (
    "data/trafficsim",
)

ALPASIM_EXCLUDED_DIR_NAMES = {
    ".git",
    ".hf-cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "logs",
    "outputs",
    "runs",
    "target",
}

ALPASIM_EXCLUDED_REL_PATHS = {
    Path("SafeVLA"),
    Path("archive"),
    Path("benchmark_annotation_bundle_unsafe_reliable"),
    Path("benchmark_annotation_bundle_unsafe_reliable_set2"),
    Path("denied"),
    Path("permission"),
    Path("sif-cache"),
    Path("slurm_logs"),
    Path("src/runtime/alpasim_runtime/consistency_monitor.py"),
    Path("src/runtime/tests/test_consistency_monitor.py"),
    Path("src/tools/rollout_monitoring"),
    Path("src/tools/tests/test_rollout_monitoring.py"),
    Path("tools/build_artifact_release.py"),
}

ALPASIM_EXCLUDED_SUFFIXES = {
    ".ckpt",
    ".err",
    ".ipynb",
    ".log",
    ".mp4",
    ".out",
    ".png",
    ".pt",
    ".pth",
    ".pyc",
    ".sif",
    ".sqsh",
    ".tmp",
    ".zip",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".gitignore",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".slurm",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

TEXT_FILENAMES = {
    ".dockerignore",
    ".flake8",
    ".gitattributes",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".python-version",
    "Dockerfile",
    "LICENSE",
}

PRIVATE_PATH_RE = re.compile(r"/(?:sfs|project|home)/[A-Za-z0-9_.+@=:/-]*")


@dataclass
class BuildReport:
    benchmark_entries: int = 0
    reliable_entries: int = 0
    benchmark_video_paths: int = 0
    copied_videos: int = 0
    copied_scene_dirs: int = 0
    copied_result_files: int = 0
    copied_source_files: int = 0
    copied_alpasim_files: int = 0
    result_file_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Source repository root. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Artifact staging directory to create.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=repo_root / "benchmark.json",
        help="Canonical benchmark JSON. Default: %(default)s",
    )
    parser.add_argument(
        "--llm-matrix-dir",
        type=Path,
        default=repo_root / "runs" / "llm_matrix",
        help="Directory containing LLM matrix result JSON files.",
    )
    parser.add_argument(
        "--rule-result",
        type=Path,
        default=repo_root / "benchmark.rule_consistency.json",
        help="Rule monitor output JSON. Default: %(default)s",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before creating the staged artifact.",
    )
    parser.add_argument(
        "--allow-incomplete-results",
        action="store_true",
        help=(
            "Stage files even when LLM result files do not exactly cover the "
            "current reliable benchmark subset. The validation report will keep "
            "these as warnings instead of errors."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def is_text_path(path: Path) -> bool:
    return path.name in TEXT_FILENAMES or path.suffix.lower() in TEXT_SUFFIXES


def sanitize_private_text(text: str) -> str:
    for root in PRIVATE_ROOTS:
        text = text.replace(root, "<alpasim-source-root>")
    text = PRIVATE_PATH_RE.sub("<redacted-path>", text)
    return text


def copy_artifact_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if is_text_path(src):
        try:
            text = src.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return copy_file(src, dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(sanitize_private_text(text), encoding="utf-8")
        shutil.copystat(src, dst)
        return True
    return copy_file(src, dst)


def copy_tree(src: Path, dst: Path) -> int:
    count = 0
    for path in src.rglob("*"):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(src)
        copy_artifact_file(path, dst / rel)
        count += 1
    return count


def is_or_under(path: Path, maybe_parent: Path) -> bool:
    return path == maybe_parent or maybe_parent in path.parents


def should_skip_alpasim_rel(rel: Path) -> bool:
    if any(part in ALPASIM_EXCLUDED_DIR_NAMES for part in rel.parts):
        return True
    if any(part.endswith(".egg-info") for part in rel.parts):
        return True
    if rel.name.startswith(".qwen") or rel.name == ".env":
        return True
    if rel.suffix.lower() in ALPASIM_EXCLUDED_SUFFIXES:
        return True
    if rel.name.startswith("benchmark_expanded_"):
        return True
    if rel.name.startswith("benchmark_annotation_bundle"):
        return True
    if rel.name in {
        "benchmark.json",
        "benchmark_old.json",
        "benchmark_pre_calibration.json",
        "benchmark.rule_consistency.json",
        "benchmark_unsafe_reliable_labels.json",
        "result.json",
    }:
        return True
    return any(is_or_under(rel, excluded) for excluded in ALPASIM_EXCLUDED_REL_PATHS)


def copy_filtered_tree(src: Path, dst: Path, base_rel: Path) -> int:
    count = 0
    for child in sorted(src.iterdir()):
        rel = base_rel / child.name
        if should_skip_alpasim_rel(rel):
            continue
        if child.is_symlink():
            continue
        if child.is_dir():
            count += copy_filtered_tree(child, dst / child.name, rel)
        elif child.is_file():
            copy_artifact_file(child, dst / child.name)
            count += 1
    return count


def resolve_existing(repo_root: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
        for root in PRIVATE_ROOTS:
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            candidates.append(repo_root / rel)
            candidates.append(repo_root / "tutorial_alpamayo" / rel)
    else:
        candidates.append(repo_root / path)
        candidates.append(repo_root / "tutorial_alpamayo" / path)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def cot_is_reliable(entry: dict[str, Any]) -> bool:
    nested = entry.get("cot_reliability")
    if isinstance(nested, dict) and "reliable" in nested:
        return nested["reliable"] is True
    value = entry.get("cot_reliable")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "reliable", "yes", "1"}
    return False


def release_path(path: Path) -> str:
    return path.as_posix()


def add_path_replacements(
    replacements: dict[str, str],
    raw_value: str | None,
    resolved: Path | None,
    release_rel: Path,
) -> None:
    if raw_value:
        replacements[raw_value] = release_path(release_rel)
    if resolved is not None:
        replacements[str(resolved)] = release_path(release_rel)


def sanitize_string(value: str, replacements: dict[str, str]) -> str:
    if value in replacements:
        return replacements[value]

    text = value
    for old, new in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        if old and old in text:
            text = text.replace(old, new)

    for root in PRIVATE_ROOTS:
        if text.startswith(root + "/"):
            text = text.replace(root + "/", "", 1)
        else:
            text = text.replace(root, ".")

    text = text.replace("/home/buc9hh", "<redacted-home>")
    return text


def sanitize_json(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_json(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item, replacements) for item in value]
    if isinstance(value, str):
        return sanitize_string(value, replacements)
    return value


def normalize_benchmark(
    repo_root: Path,
    output: Path,
    benchmark_path: Path,
    report: BuildReport,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    benchmark = load_json(benchmark_path)
    if not isinstance(benchmark, list):
        raise ValueError(f"{benchmark_path} must contain a JSON list")

    replacements: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []
    video_sources: dict[str, Path] = {}
    scene_sources: dict[str, Path] = {}

    for entry in benchmark:
        if not isinstance(entry, dict):
            continue
        clip_id = str(entry.get("clip_id", "")).strip()
        if not clip_id:
            report.warnings.append("Skipped benchmark entry without clip_id")
            continue

        item = dict(entry)
        video_value = item.get("rollout_mp4")
        if video_value:
            resolved_video = resolve_existing(repo_root, str(video_value))
            if resolved_video is None:
                report.errors.append(f"Missing rollout_mp4 for {clip_id}: {video_value}")
            else:
                video_rel = Path("data/media/videos") / f"{clip_id}.mp4"
                video_sources[clip_id] = resolved_video
                item["rollout_mp4"] = release_path(video_rel)
                add_path_replacements(replacements, str(video_value), resolved_video, video_rel)

        scene_value = item.get("source_scene_file")
        if scene_value:
            resolved_scene = resolve_existing(repo_root, str(scene_value))
            if resolved_scene is None:
                report.errors.append(f"Missing source_scene_file for {clip_id}: {scene_value}")
            else:
                scene_dir_rel = Path("data/media/scenes") / clip_id
                scene_rel = scene_dir_rel / "metadata.json"
                scene_sources[clip_id] = resolved_scene
                item["source_scene_file"] = release_path(scene_rel)
                add_path_replacements(replacements, str(scene_value), resolved_scene, scene_rel)
                add_path_replacements(
                    replacements,
                    str(Path(scene_value).parent),
                    resolved_scene.parent,
                    scene_dir_rel,
                )

        normalized.append(item)

    for clip_id, src in video_sources.items():
        dst = output / "data/media/videos" / f"{clip_id}.mp4"
        copy_file(src, dst)
        report.copied_videos += 1

    for clip_id, metadata_path in scene_sources.items():
        dst_dir = output / "data/media/scenes" / clip_id
        for name in SCENE_SIDE_FILES:
            src = metadata_path.parent / name
            if src.exists():
                dst = dst_dir / name
                if src.suffix == ".json":
                    payload = sanitize_json(load_json(src), replacements)
                    write_json(dst, payload)
                else:
                    copy_file(src, dst)
        report.copied_scene_dirs += 1

    report.benchmark_entries = len(normalized)
    report.reliable_entries = sum(1 for item in normalized if cot_is_reliable(item))
    report.benchmark_video_paths = sum(1 for item in normalized if item.get("rollout_mp4"))
    write_json(output / "data/benchmark/benchmark.json", normalized)

    reliable_subset = [item for item in normalized if cot_is_reliable(item)]
    write_json(output / "data/results/llm_matrix/benchmark.all_reliable.json", reliable_subset)
    return normalized, replacements


def copy_results(
    output: Path,
    llm_matrix_dir: Path,
    rule_result: Path,
    replacements: dict[str, str],
    report: BuildReport,
) -> None:
    result_dir = output / "data/results/llm_matrix"
    result_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for src in sorted(llm_matrix_dir.glob("*.run_*.json")):
        payload = sanitize_json(load_json(src), replacements)
        dst = result_dir / src.name
        write_json(dst, payload)
        report.copied_result_files += 1
        results = payload.get("results", payload if isinstance(payload, list) else [])
        runs.append(
            {
                "name": src.stem,
                "path": release_path(Path("data/results/llm_matrix") / src.name),
                "result_count": len(results) if isinstance(results, list) else None,
            }
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_json": "data/benchmark/benchmark.json",
        "subset_path": "data/results/llm_matrix/benchmark.all_reliable.json",
        "note": (
            "Generated by tools/build_artifact_release.py from the current "
            "canonical benchmark and current llm_matrix result files."
        ),
        "runs": runs,
    }
    write_json(result_dir / "manifest.json", manifest)

    if rule_result.exists():
        payload = sanitize_json(load_json(rule_result), replacements)
        write_json(output / "data/results/rule/benchmark.rule_consistency.json", payload)
        report.copied_result_files += 1
    else:
        report.warnings.append(f"Rule result not found: {rule_result}")


def copy_code(repo_root: Path, output: Path, report: BuildReport) -> None:
    for rel in TOOL_FILES:
        src = repo_root / rel
        if copy_artifact_file(src, output / rel):
            report.copied_source_files += 1
        else:
            report.warnings.append(f"Missing tool file: {rel}")

    for rel in SOURCE_DIRS:
        src = repo_root / rel
        if src.exists():
            report.copied_source_files += copy_tree(src, output / rel)
        else:
            report.warnings.append(f"Missing source directory: {rel}")

    for rel in SOURCE_FILES:
        src = repo_root / rel
        if copy_artifact_file(src, output / rel):
            report.copied_source_files += 1
        else:
            report.warnings.append(f"Missing source file: {rel}")

    for rel in TEST_FILES:
        src = repo_root / rel
        if copy_artifact_file(src, output / rel):
            report.copied_source_files += 1

    for rel in ("LICENSE", "Apptainer.def", "build_apptainer.sh"):
        src = repo_root / rel
        if copy_artifact_file(src, output / rel):
            report.copied_source_files += 1


def write_artifact_requirements(output: Path, report: BuildReport) -> None:
    text = "\n".join(ARTIFACT_REQUIREMENTS) + "\n"
    (output / "requirements-artifact.txt").write_text(text, encoding="utf-8")
    (output / "pyproject.toml").write_text(ARTIFACT_PYPROJECT, encoding="utf-8")
    (output / "environment-artifact.yml").write_text(
        ARTIFACT_CONDA_ENVIRONMENT,
        encoding="utf-8",
    )
    report.copied_source_files += 3


def copy_alpasim_snapshot(repo_root: Path, output: Path, report: BuildReport) -> None:
    dst_root = output / "alpasim"

    for rel in ALPASIM_TOP_LEVEL_FILES:
        src = repo_root / rel
        if copy_artifact_file(src, dst_root / rel):
            report.copied_alpasim_files += 1
        else:
            report.warnings.append(f"Missing AlpaSim top-level file: {rel}")

    for rel in ALPASIM_TREE_DIRS:
        src = repo_root / rel
        if src.exists():
            report.copied_alpasim_files += copy_filtered_tree(src, dst_root / rel, Path(rel))
        else:
            report.warnings.append(f"Missing AlpaSim directory: {rel}")

    for rel in ALPASIM_DATA_FILES:
        src = repo_root / rel
        if copy_artifact_file(src, dst_root / rel):
            report.copied_alpasim_files += 1
        else:
            report.warnings.append(f"Missing AlpaSim data file: {rel}")

    for rel in ALPASIM_DATA_DIRS:
        src = repo_root / rel
        if src.exists():
            report.copied_alpasim_files += copy_filtered_tree(src, dst_root / rel, Path(rel))
        else:
            report.warnings.append(f"Missing AlpaSim data directory: {rel}")


def validate_results(
    output: Path,
    benchmark: list[dict[str, Any]],
    report: BuildReport,
    allow_incomplete_results: bool,
) -> None:
    reliable_ids = {
        item["clip_id"]
        for item in benchmark
        if item.get("clip_id") and cot_is_reliable(item)
    }
    result_dir = output / "data/results/llm_matrix"
    for path in sorted(result_dir.glob("*.run_*.json")):
        payload = load_json(path)
        results = payload.get("results", payload if isinstance(payload, list) else [])
        ids = {
            item.get("clip_id")
            for item in results
            if isinstance(item, dict) and item.get("clip_id")
        }
        missing = sorted(reliable_ids - ids)
        extra = sorted(ids - reliable_ids)
        status = {
            "result_count": len(ids),
            "missing_reliable_count": len(missing),
            "extra_vs_reliable_count": len(extra),
            "missing_reliable_sample": missing[:10],
            "extra_vs_reliable_sample": extra[:10],
        }
        report.result_file_status[path.name] = status
        if missing or extra:
            message = (
                f"{path.name} does not match current reliable benchmark subset: "
                f"missing={len(missing)} extra={len(extra)}"
            )
            if allow_incomplete_results:
                report.warnings.append(message)
            else:
                report.errors.append(message)


def validate_benchmark_media(output: Path, benchmark: list[dict[str, Any]], report: BuildReport) -> None:
    seen_videos: dict[str, str] = {}
    for entry in benchmark:
        clip_id = entry.get("clip_id", "<missing>")
        rollout = entry.get("rollout_mp4")
        if rollout:
            path = output / rollout
            if not path.exists():
                report.errors.append(f"Staged video missing for {clip_id}: {rollout}")
            if rollout in seen_videos:
                report.errors.append(
                    f"Duplicate staged rollout_mp4 path: {rollout} "
                    f"({seen_videos[rollout]}, {clip_id})"
                )
            seen_videos[rollout] = str(clip_id)
        source_scene = entry.get("source_scene_file")
        if source_scene and not (output / source_scene).exists():
            report.errors.append(f"Staged metadata missing for {clip_id}: {source_scene}")


def validate_json_leaks(output: Path, report: BuildReport) -> None:
    for path in sorted((output / "data").rglob("*.json")):
        text = path.read_text(encoding="utf-8")
        for marker in PRIVATE_MARKERS:
            if marker in text:
                report.errors.append(f"Private path marker {marker!r} found in {path}")
        for marker in STALE_BENCHMARK_MARKERS:
            if marker in text:
                report.errors.append(f"Stale benchmark marker {marker!r} found in {path}")


def validate_private_text_leaks(output: Path, report: BuildReport) -> None:
    for path in sorted(output.rglob("*")):
        if not path.is_file() or not is_text_path(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in PRIVATE_MARKERS:
            if marker in text:
                report.errors.append(f"Private path marker {marker!r} found in {path}")


def write_readme(output: Path, report: BuildReport) -> None:
    readme = f"""# Drive the Thoughts Artifact

This directory is a staged artifact for the paper "Drive the Thoughts:
Runtime Monitoring of VLA Reasoning-Trajectory Consistency".

The artifact has two reproduction tiers:

1. A lightweight data-first tier that recomputes checks, tables, and plots from
   the released benchmark plus JSON monitor outputs.
2. A pinned AlpaSim source tier under `alpasim/` for reviewers who want to
   inspect or attempt full simulator reruns.

Apptainer is the tested environment path for AlpaSim. Docker/local Python can
be attempted on a best-effort basis, but is not guaranteed.

## Contents

- `data/benchmark/benchmark.json`: canonical benchmark with artifact-relative
  media paths.
- `data/results/llm_matrix/`: raw LLM judge outputs and the reliable subset
  derived from the current benchmark.
- `data/results/rule/benchmark.rule_consistency.json`: deterministic rule
  monitor output.
- `data/media/videos/`: rollout videos copied from benchmark references.
- `data/media/scenes/`: per-clip scene metadata and supporting files when
  available.
- `tools/` and `src/`: scripts and monitor implementation used for the
  JSON-output reproduction tier.
- `pyproject.toml`: lightweight `uv` environment for the data-first tier.
- `requirements-artifact.txt`: plain `venv`/`pip` fallback dependencies.
- `environment-artifact.yml`: optional Conda environment for the same
  lightweight tier.
- `alpasim/`: pinned AlpaSim workspace snapshot with its own `pyproject.toml`
  and `uv.lock`. Generated rollouts, caches, model/data artifacts, paper source,
  secrets, and obsolete benchmark variants are intentionally not included.

## Current Staging Counts

- Benchmark entries: {report.benchmark_entries}
- Reliable benchmark entries: {report.reliable_entries}
- Benchmark entries with videos: {report.benchmark_video_paths}
- Copied videos: {report.copied_videos}
- Copied scene directories: {report.copied_scene_dirs}
- Copied result JSON files: {report.copied_result_files}
- Copied AlpaSim source-tier files: {report.copied_alpasim_files}

These counts are generated from the current files at staging time. If the paper
tables or benchmark are updated, rerun `tools/build_artifact_release.py` and
check `ARTIFACT_REPORT.json`.

## Lightweight Setup

The recommended lightweight path uses `uv` from the artifact root:

```bash
uv sync --frozen
```

If `uv.lock` is not present because the artifact was rebuilt locally without
locking, use `uv sync` once to resolve the small dependency set.

Plain Python fallback:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-artifact.txt
export PYTHONPATH="$PWD/src/utils:$PWD/src/tools${{PYTHONPATH:+:$PYTHONPATH}}"
```

Optional Conda fallback:

```bash
conda env create -f environment-artifact.yml
conda activate drive-the-thoughts-artifact
export PYTHONPATH="$PWD/src/utils:$PWD/src/tools${{PYTHONPATH:+:$PYTHONPATH}}"
```

## Quick Checks

From the artifact root:

```bash
uv run python tools/check_consistency_accuracy.py \\
  --consistency-file data/results/llm_matrix/gpt.f_llm_map_graph.run_003.json \\
  --benchmark-file data/benchmark/benchmark.json \\
  --reliable-only
```

For plotting:

```bash
uv run python tools/plot_repeated_main_eval.py \\
  --run-dir data/results/llm_matrix \\
  --benchmark data/benchmark/benchmark.json \\
  --rule-result data/results/rule/benchmark.rule_consistency.json \\
  --output repeated_main_eval.pdf
```

Run the packaged unit tests:

```bash
uv run pytest
```

## AlpaSim Setup

The full simulator tier keeps AlpaSim isolated in `alpasim/` so its `uv`
workspace and lockfile do not interfere with the lightweight artifact tools:

```bash
cd alpasim
source setup_local_env.sh
uv sync --frozen
```

Use `uv run ...` from inside `alpasim/` for simulator commands. This source
snapshot is intended to preserve the implementation context and Apptainer setup,
not to bundle the full upstream data/model store. Large generated data and
rollouts are omitted; benchmark media needed for the JSON-output tier lives in
the artifact root under `data/media/`.

## Validation

`ARTIFACT_REPORT.json` records warnings/errors from the staging run. Treat the
artifact as release-ready only when `ok` is `true`.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def write_report(output: Path, report: BuildReport) -> None:
    payload = {
        "ok": report.ok,
        "benchmark_entries": report.benchmark_entries,
        "reliable_entries": report.reliable_entries,
        "benchmark_video_paths": report.benchmark_video_paths,
        "copied_videos": report.copied_videos,
        "copied_scene_dirs": report.copied_scene_dirs,
        "copied_result_files": report.copied_result_files,
        "copied_source_files": report.copied_source_files,
        "copied_alpasim_files": report.copied_alpasim_files,
        "result_file_status": report.result_file_status,
        "warnings": report.warnings,
        "errors": report.errors,
    }
    write_json(output / "ARTIFACT_REPORT.json", payload)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()

    if output.exists() and args.clean:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    report = BuildReport()
    benchmark, replacements = normalize_benchmark(
        repo_root=repo_root,
        output=output,
        benchmark_path=args.benchmark.resolve(),
        report=report,
    )
    copy_results(
        output=output,
        llm_matrix_dir=args.llm_matrix_dir.resolve(),
        rule_result=args.rule_result.resolve(),
        replacements=replacements,
        report=report,
    )
    copy_code(repo_root, output, report)
    write_artifact_requirements(output, report)
    copy_alpasim_snapshot(repo_root, output, report)

    validate_benchmark_media(output, benchmark, report)
    validate_results(output, benchmark, report, args.allow_incomplete_results)
    validate_json_leaks(output, report)
    validate_private_text_leaks(output, report)

    write_readme(output, report)
    write_report(output, report)

    print(f"wrote artifact stage: {output}")
    print(f"benchmark_entries={report.benchmark_entries}")
    print(f"reliable_entries={report.reliable_entries}")
    print(f"copied_videos={report.copied_videos}")
    print(f"copied_result_files={report.copied_result_files}")
    print(f"warnings={len(report.warnings)} errors={len(report.errors)}")
    if report.errors:
        for error in report.errors:
            print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
