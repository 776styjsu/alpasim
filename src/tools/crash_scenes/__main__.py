# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Command line for the crash-scene tools.

    python -m crash_scenes label <run dir or rollout.asl> ... --output labels.jsonl
    python -m crash_scenes replay-monitor <run dir or rollout.asl> ... --output-dir DIR
    python -m crash_scenes mine --output-dir DIR
    python -m crash_scenes plans --mine-dir DIR --output-dir DIR
    python -m crash_scenes scene-ids PLAN.json
    python -m crash_scenes confirm --plan-set DIR <run dir> ...

Run from the repository root with ``PYTHONPATH=src/utils:src/tools``.
See docs/CRASH_SCENES.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from alpasim_utils.artifact import Artifact

from .confirm import confirm_runs
from .label import label_rollout
from .mine import SearchConfig, mine_scene
from .monitor_replay import load_monitor_module, replay_two_second_monitor
from .plans import read_plan_file, select_crashes, write_plan_set
from .scenes import index_scenes, load_scene
from .tracks import read_rollout_tracks

logger = logging.getLogger(__name__)

DEFAULT_USDZ_DIR = "data/nre-artifacts/all-usdzs"
DEFAULT_MONITOR = "/project/AV/monitors/two_second_monitor.py"
MANIFEST = "manifest.jsonl"
SEARCH_CONFIG = "search_config.json"


def find_rollout_logs(paths: list[str]) -> list[Path]:
    """Every ``rollout.asl`` named by, or found beneath, ``paths``."""
    logs: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            logs.append(path)
        elif path.is_dir():
            logs.extend(sorted(path.rglob("rollout.asl")))
        else:
            raise FileNotFoundError(raw)
    if not logs:
        raise FileNotFoundError(f"No rollout.asl under {paths}")
    return logs


def _label(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    crashed = 0
    with output.open("w") as out:
        for asl_path in find_rollout_logs(args.paths):
            row = label_rollout(asyncio.run(read_rollout_tracks(str(asl_path))))
            row["asl_path"] = str(asl_path)
            out.write(json.dumps(row) + "\n")
            crashed += bool(row["crashes"])
            first = row["crashes"][0] if row["crashes"] else None
            logger.info(
                "%s: %s",
                asl_path,
                (
                    "no crash"
                    if first is None
                    else f"hit {first['actor_group']} {first['actor_id']} "
                    f"on ego {first['ego_zone8']} ({first['crash_type']})"
                ),
            )
    logger.info("Wrote %s (%d rollouts with a crash)", output, crashed)


def _replay_monitor(args: argparse.Namespace) -> None:
    monitor_module = load_monitor_module(args.monitor)
    scenes = index_scenes(args.usdz_dir)
    output_root = Path(args.output_dir)
    summary = []
    for asl_path in find_rollout_logs(args.paths):
        tracks = asyncio.run(read_rollout_tracks(str(asl_path)))
        result = asyncio.run(
            replay_two_second_monitor(
                monitor_module,
                str(asl_path),
                str(output_root / tracks.scene_id / tracks.session_uuid),
                vector_map=scenes[tracks.scene_id].map,
            )
        )
        summary.append(result)
        logger.info("%s", result)
    (output_root / "replay_summary.json").write_text(json.dumps(summary, indent=2))


def _mine_one(job: tuple[str, SearchConfig]) -> tuple[str, list[dict], str | None]:
    """Mine one scene file; returns its path, its rows and any error."""
    usdz_path, config = job
    # The scene loader warns for every smoothed track.
    logging.getLogger("alpasim_utils.scenario").setLevel(logging.ERROR)
    try:
        scene = load_scene(
            Artifact(usdz_path),
            config.n_sim_steps,
            config.force_gt_duration_us,
            config.control_timestep_us,
            config.min_traffic_duration_us,
        )
        return usdz_path, mine_scene(scene, config), None
    except Exception as error:  # one unreadable scene must not end the sweep
        return usdz_path, [], f"{type(error).__name__}: {error}"


def _mine(args: argparse.Namespace) -> None:
    config = SearchConfig()
    artifacts = index_scenes(args.usdz_dir)
    paths = sorted(artifact.source for artifact in artifacts.values())
    if args.limit is not None:
        paths = paths[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SEARCH_CONFIG).write_text(json.dumps(asdict(config), indent=2))

    rows: list[dict] = []
    failed: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [(path, config) for path in paths]
        for done, (path, scene_rows, error) in enumerate(pool.map(_mine_one, jobs), 1):
            if error is not None:
                failed[path] = error
            rows.extend(scene_rows)
            logger.info(
                "[%d/%d] %s: %s",
                done,
                len(paths),
                Path(path).name,
                error or f"{len(scene_rows)} crashes",
            )

    with (output_dir / MANIFEST).open("w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")

    def count(key) -> dict[str, int]:
        return dict(sorted(Counter(map(key, rows)).items()))

    summary = {
        "scenes_searched": len(paths),
        "scenes_failed": failed,
        "scenes_with_a_crash": len({row["scene_id"] for row in rows}),
        "crashes": len(rows),
        "crashes_with_near_miss": sum(row["near_miss"] is not None for row in rows),
        "by_actor_group": count(lambda r: r["predicted"]["actor_group"]),
        "by_ego_zone8": count(lambda r: r["predicted"]["ego_zone8"]),
        "by_actor_group_and_ego_zone4": count(
            lambda r: f"{r['predicted']['actor_group']}/{r['predicted']['ego_zone4']}"
        ),
        "by_crash_type": count(lambda r: r["predicted"]["crash_type"]),
        "by_maneuver": count(lambda r: r["maneuver"]["kind"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("%s", json.dumps(summary, indent=2))


def _plans(args: argparse.Namespace) -> None:
    mine_dir = Path(args.mine_dir)
    rows = [json.loads(line) for line in (mine_dir / MANIFEST).open()]
    selected = select_crashes(rows, args.per_cell, args.max_per_scene)
    output_dir = Path(args.output_dir)
    summary = write_plan_set(selected, output_dir)
    shutil.copyfile(mine_dir / SEARCH_CONFIG, output_dir / SEARCH_CONFIG)
    logger.info("Wrote %s: %s", output_dir, summary)


def _scene_ids(args: argparse.Namespace) -> None:
    print(",".join(sorted(read_plan_file(args.plan_file))))


def _confirm(args: argparse.Namespace) -> None:
    plan_set_dir = Path(args.plan_set)
    results = confirm_runs(plan_set_dir, [Path(path) for path in args.run_dirs])
    output = Path(args.output) if args.output else plan_set_dir / "confirmation.jsonl"
    with output.open("w") as out:
        for row in results:
            out.write(json.dumps(row) + "\n")
    outcomes = Counter((row["variant"], row["outcome"]) for row in results)
    for (variant, outcome), number in sorted(outcomes.items()):
        logger.info("%-10s %-12s %d", variant, outcome, number)
    logger.info("Wrote %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(prog="crash_scenes", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    label = commands.add_parser("label", help="Label crashes in rollout logs.")
    label.add_argument("paths", nargs="+", help="rollout.asl files or run directories")
    label.add_argument("--output", required=True, help="JSONL, one row per rollout")
    label.set_defaults(run=_label)

    replay = commands.add_parser(
        "replay-monitor", help="Run the two-second monitor over saved rollout logs."
    )
    replay.add_argument("paths", nargs="+", help="rollout.asl files or run directories")
    replay.add_argument("--output-dir", required=True)
    replay.add_argument("--monitor", default=DEFAULT_MONITOR)
    replay.add_argument("--usdz-dir", default=DEFAULT_USDZ_DIR)
    replay.set_defaults(run=_replay_monitor)

    mine = commands.add_parser(
        "mine", help="Search recorded scenes for maneuvers that end in a crash."
    )
    mine.add_argument("--usdz-dir", default=DEFAULT_USDZ_DIR)
    mine.add_argument("--output-dir", required=True)
    mine.add_argument("--workers", type=int, default=8)
    mine.add_argument("--limit", type=int, help="Only search the first N scenes")
    mine.set_defaults(run=_mine)

    plans = commands.add_parser(
        "plans", help="Choose mined crashes and write plan files for them."
    )
    plans.add_argument("--mine-dir", required=True, help="Output directory of `mine`")
    plans.add_argument("--output-dir", required=True)
    plans.add_argument("--per-cell", type=int, default=3)
    plans.add_argument("--max-per-scene", type=int, default=4)
    plans.set_defaults(run=_plans)

    scene_ids = commands.add_parser(
        "scene-ids", help="Print the scene ids of a plan file, comma separated."
    )
    scene_ids.add_argument("plan_file")
    scene_ids.set_defaults(run=_scene_ids)

    confirm = commands.add_parser(
        "confirm", help="Compare rollouts with the plans they ran."
    )
    confirm.add_argument(
        "--plan-set", required=True, help="Output directory of `plans`"
    )
    confirm.add_argument("run_dirs", nargs="+", help="Wizard run directories")
    confirm.add_argument("--output", help="Default: <plan set>/confirmation.jsonl")
    confirm.set_defaults(run=_confirm)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.run(args)


if __name__ == "__main__":
    main()
