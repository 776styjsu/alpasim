#!/usr/bin/env python3
"""Build a "flat" tutorial directory that aggregates every per-scene run into one
big folder via symlinks, mirroring the layout of tutorial_alpamayo_next100_unique_flat.

Layout produced:
  <dst>/<config files>            -> symlinks to the representative scene (scene_1)
  <dst>/aggregate/metrics_*       -> symlinks to the representative scene's aggregate
  <dst>/aggregate/videos/all/     -> symlinks to every scene's aggregate/videos/all/*
  <dst>/aggregate/videos/violations/<cat>/ -> symlinks gathered from every scene
  <dst>/controller/               -> symlinks to every scene's controller/*
  <dst>/rollouts/                 -> symlinks to every scene's rollouts/* (clip dirs)

extracted_frames/ is NOT created here; it is generated afterwards with
  tools/extract_rollout_frames.sh -r <dst> --resume
"""
import re
import sys
from pathlib import Path

# Entries inside a scene dir that are aggregated specially rather than copied as-is.
GATHER = {"aggregate", "controller", "rollouts"}


def natural_key(p: Path):
    m = re.search(r"scene_(\d+)_", p.name)
    return (int(m.group(1)) if m else 1 << 30, p.name)


def link(target: Path, linkname: Path):
    """Create an absolute symlink (first writer wins; skip if already present)."""
    if linkname.is_symlink() or linkname.exists():
        return
    linkname.parent.mkdir(parents=True, exist_ok=True)
    linkname.symlink_to(str(target))


def main(src, dst, representative):
    scenes = sorted([p for p in src.iterdir() if p.is_dir() and p.name.startswith("scene_")],
                    key=natural_key)
    if not scenes:
        sys.exit(f"No scene_* directories found under {src}")

    rep = src / representative if representative else scenes[0]
    if not rep.is_dir():
        sys.exit(f"Representative scene not found: {rep}")
    print(f"Representative scene: {rep.name}")
    print(f"Scenes to aggregate: {len(scenes)}")

    dst.mkdir(parents=True, exist_ok=True)

    # 1. Top-level representative config files/dirs.
    for entry in sorted(rep.iterdir()):
        if entry.name in GATHER:
            continue
        link(entry, dst / entry.name)

    # 2. aggregate/ metrics from the representative scene.
    for entry in sorted((rep / "aggregate").iterdir()):
        if entry.name == "videos":
            continue
        link(entry, dst / "aggregate" / entry.name)

    # 3. aggregate/videos/all + violations, gathered from every scene.
    #    Pre-create violation category dirs (some may stay empty, like collision_rear).
    for cat in sorted((rep / "aggregate" / "videos" / "violations").iterdir()):
        (dst / "aggregate" / "videos" / "violations" / cat.name).mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        all_dir = scene / "aggregate" / "videos" / "all"
        if all_dir.is_dir():
            for mp4 in sorted(all_dir.iterdir()):
                link(mp4, dst / "aggregate" / "videos" / "all" / mp4.name)
        viol_dir = scene / "aggregate" / "videos" / "violations"
        if viol_dir.is_dir():
            for cat in sorted(viol_dir.iterdir()):
                for vid in sorted(cat.iterdir()):
                    link(vid, dst / "aggregate" / "videos" / "violations" / cat.name / vid.name)

        # 4. controller CSVs.
        ctrl_dir = scene / "controller"
        if ctrl_dir.is_dir():
            for f in sorted(ctrl_dir.iterdir()):
                link(f, dst / "controller" / f.name)

        # 5. rollouts (clip dirs).
        roll_dir = scene / "rollouts"
        if roll_dir.is_dir():
            for clip in sorted(roll_dir.iterdir()):
                link(clip, dst / "rollouts" / clip.name)

    print(f"Done. Flat directory built at {dst}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: build_flat_tutorial.py <src_dir> <dst_dir> [representative_scene_name]")
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
