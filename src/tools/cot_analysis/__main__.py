# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
CoT Consistency Analysis for AlpaSim simulation outputs.

Reads ASL files from AlpaSim rollouts, extracts per-timestep Chain-of-Thought
reasoning and predicted trajectories, then evaluates self-consistency using
Gemini multimodal LLM. Produces structured per-dimension scores.

Usage:
    # Dry run (no API key, verifies trajectory analysis):
    uv run python -m cot_analysis \
        --asl_glob "tutorial_alpamayo/rollouts/**/*.asl" \
        --output /tmp/cot_dry.json

    # Full run:
    uv run python -m cot_analysis \
        --asl_glob "tutorial_alpamayo/rollouts/**/*.asl" \
        --output tutorial_alpamayo/cot_consistency.json \
        --api_key YOUR_KEY
"""

import argparse
import asyncio
import glob
import io
import json
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .prompt import build_prompt
    
except ImportError:
    from prompt import build_prompt
    from trajectory import compute_trajectory_features


logger = logging.getLogger(__name__)

# Dimensions to score
DIMENSIONS = ["cot_coherence", "cot_output_alignment"]


# =============================================================================
# ASL Data Extraction
# =============================================================================


def _extract_debug_info(driver_response) -> dict | None:
    """Extract and unpickle debug dict from a DriveResponse."""
    try:
        dbg_bytes = driver_response.debug_info.unstructured_debug_info
        if not dbg_bytes:
            return None
        extra = pickle.loads(dbg_bytes)
        if isinstance(extra, dict):
            return extra
        return None
    except Exception:
        return None


def _trajectory_poses_to_xy(trajectory) -> np.ndarray | None:
    """Convert a gRPC Trajectory to a (T, 2) rig-frame XY array.

    The DriveResponse trajectory is in the local frame. The poses are
    expressed relative to the first pose (current ego position), so the
    first point is near origin and subsequent points are the predicted
    future path in rig-frame coordinates.
    """
    poses = trajectory.poses
    if len(poses) < 3:
        return None

    # The first pose is the current position (anchor).
    # Subsequent poses are the planned future trajectory in local frame.
    # Convert to rig-frame relative to the anchor.
    anchor = poses[0]
    anchor_x = anchor.pose.vec.x
    anchor_y = anchor.pose.vec.y
    anchor_yaw = np.arctan2(
        2.0 * (anchor.pose.quat.w * anchor.pose.quat.z
               + anchor.pose.quat.x * anchor.pose.quat.y),
        1.0 - 2.0 * (anchor.pose.quat.y ** 2 + anchor.pose.quat.z ** 2),
    )

    cos_yaw = np.cos(-anchor_yaw)
    sin_yaw = np.sin(-anchor_yaw)

    xy = []
    for pose_at_time in poses[1:]:
        p = pose_at_time.pose
        dx = p.vec.x - anchor_x
        dy = p.vec.y - anchor_y
        # Rotate into rig frame (relative to ego heading)
        rx = cos_yaw * dx - sin_yaw * dy
        ry = sin_yaw * dx + cos_yaw * dy
        xy.append([rx, ry])

    return np.array(xy, dtype=float) if xy else None


async def extract_entries_from_asl(
    asl_path: str,
    camera_id: str = "camera_front_wide_120fov",
    every_nth: int = 1,
) -> list[dict]:
    """Parse an ASL file and extract per-timestep entries for analysis.

    Args:
        asl_path: Path to the .asl file.
        camera_id: Camera logical ID for image extraction.
        every_nth: Only keep every Nth timestep (1 = keep all).

    Returns:
        List of dicts, each with keys:
            clip_id, chain_of_thought, trajectory_xy, image (PIL or None),
            timestamp_us
    """
    from alpasim_utils.logs import async_read_pb_log

    # Collect raw data from ASL messages
    scene_id = "unknown"
    pending_request: tuple[int, int] | None = None
    driver_responses: list[tuple[int, int, object]] = []  # (now_us, query_us, resp)
    camera_frames: dict[int, bytes] = {}  # timestamp_us -> image_bytes

    async for message in async_read_pb_log(asl_path):
        msg_type = message.WhichOneof("log_entry")

        if msg_type == "rollout_metadata":
            if message.rollout_metadata.session_metadata is not None:
                scene_id = message.rollout_metadata.session_metadata.scene_id

        elif msg_type == "driver_request":
            pending_request = (
                message.driver_request.time_now_us,
                message.driver_request.time_query_us,
            )

        elif msg_type == "driver_return":
            if pending_request is not None:
                driver_responses.append(
                    (*pending_request, message.driver_return)
                )
                pending_request = None

        elif msg_type == "driver_camera_image":
            cam_img = message.driver_camera_image.camera_image
            if cam_img.logical_id == camera_id:
                camera_frames[cam_img.frame_end_us] = cam_img.image_bytes

    # Build entries from driver responses
    entries = []
    for idx, (now_us, _query_us, resp) in enumerate(driver_responses):
        if every_nth > 1 and idx % every_nth != 0:
            continue

        # Skip empty trajectories (warmup period)
        if len(resp.trajectory.poses) < 3:
            continue

        # Extract CoT
        extra = _extract_debug_info(resp)
        reasoning_text = None
        if extra is not None:
            reasoning_text = extra.get("reasoning_text")
        if reasoning_text is None:
            continue  # no CoT for this timestep

        # Extract trajectory (rig-frame XY)
        trajectory_xy = _trajectory_poses_to_xy(resp.trajectory)
        if trajectory_xy is None:
            continue

        # Find closest camera image
        image = None
        if camera_frames:
            # Find the camera frame with timestamp closest to (but <= ) now_us
            valid_ts = [ts for ts in camera_frames if ts <= now_us]
            if valid_ts:
                closest_ts = max(valid_ts)
                try:
                    image = Image.open(io.BytesIO(camera_frames[closest_ts]))
                except Exception:
                    pass

        entry = {
            "clip_id": f"{scene_id}_t{idx:04d}",
            "chain_of_thought": reasoning_text,
            "trajectory_xy": trajectory_xy,
            "image": image,
            "timestamp_us": now_us,
        }
        entries.append(entry)

    logger.info(
        "Extracted %d entries from %s (scene: %s, total driver responses: %d)",
        len(entries),
        os.path.basename(asl_path),
        scene_id,
        len(driver_responses),
    )
    return entries


# =============================================================================
# Gemini API (google-genai SDK)
# =============================================================================


def _create_gemini_client(api_key: str):
    """Create and return a google-genai Client."""
    from google import genai

    client = genai.Client(api_key=api_key)
    return client


def call_gemini(client, model_name: str, prompt: str, image=None) -> str:
    """Send prompt (+ optional image) to Gemini and return response text."""
    from google.genai import types

    try:
        contents = []
        if image is not None:
            contents.append(image)
        contents.append(prompt)

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                ],
            ),
        )
        return response.text
    except Exception as e:
        return json.dumps({"error": str(e)})


def parse_gemini_response(response_text: str) -> dict:
    """Parse structured JSON response from Gemini."""
    try:
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        parsed = json.loads(text)
        if isinstance(parsed, list):
            if len(parsed) > 0 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            else:
                return {"parse_error": True, "raw_response": response_text}
        if not isinstance(parsed, dict):
            return {"parse_error": True, "raw_response": response_text}
        return parsed
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_response": response_text}


# =============================================================================
# Processing & Aggregation
# =============================================================================


def flatten_cot(cot) -> str:
    """Recursively flatten CoT from nested lists to a single string."""
    if isinstance(cot, str):
        return cot
    if isinstance(cot, list):
        return " ".join(flatten_cot(item) for item in cot)
    return str(cot)


def process_entry(
    client, model_name: str, entry: dict, use_images: bool = True,
) -> dict:
    """Process a single extracted entry through the analysis pipeline."""
    clip_id = entry["clip_id"]
    cot_text = flatten_cot(entry["chain_of_thought"])

    # Compute trajectory features
    traj_features = compute_trajectory_features(entry["trajectory_xy"])

    # Build prompt
    prompt = build_prompt(cot_text, traj_features)

    # Call Gemini
    if client is not None:
        image = entry.get("image") if use_images else None
        raw_response = call_gemini(client, model_name, prompt, image=image)
        parsed = parse_gemini_response(raw_response)
    else:
        parsed = {"dry_run": True}

    return {
        "clip_id": clip_id,
        "chain_of_thought": cot_text,
        "trajectory_features": traj_features,
        "evaluation": parsed,
        "timestamp_us": entry.get("timestamp_us"),
    }


def aggregate_results(results: list) -> dict:
    """Compute summary statistics across all evaluated entries."""
    scores = {d: [] for d in DIMENSIONS}
    for r in results:
        ev = r.get("evaluation", {})
        if ev.get("dry_run") or ev.get("parse_error") or ev.get("error"):
            continue
        for d in DIMENSIONS:
            dim_data = ev.get(d, {})
            if isinstance(dim_data, dict) and "score" in dim_data:
                try:
                    scores[d].append(float(dim_data["score"]))
                except (ValueError, TypeError):
                    pass

    summary = {}
    for d in DIMENSIONS:
        s = scores[d]
        if s:
            summary[d] = {
                "mean": round(np.mean(s), 2),
                "std": round(np.std(s), 2),
                "min": round(min(s), 1),
                "max": round(max(s), 1),
                "count": len(s),
            }
        else:
            summary[d] = {"count": 0}

    # Flag low-scoring entries (any dimension <= 2)
    flagged = []
    for r in results:
        ev = r.get("evaluation", {})
        for d in DIMENSIONS:
            dim_data = ev.get(d, {})
            if isinstance(dim_data, dict) and dim_data.get("score", 5) <= 2:
                flagged.append(
                    {
                        "clip_id": r["clip_id"],
                        "dimension": d,
                        "score": dim_data["score"],
                        "justification": dim_data.get("justification", ""),
                    }
                )

    summary["flagged_entries"] = flagged
    summary["total_evaluated"] = sum(
        1
        for r in results
        if not r.get("evaluation", {}).get("dry_run")
        and not r.get("evaluation", {}).get("error")
    )
    return summary


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="CoT consistency analysis for AlpaSim simulation outputs"
    )
    parser.add_argument(
        "--asl_glob",
        type=str,
        required=True,
        help="Glob pattern to find ASL files (e.g. 'rollouts/**/*.asl')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cot_consistency.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Gemini model name",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls (seconds)",
    )
    parser.add_argument(
        "--every_nth",
        type=int,
        default=1,
        help="Only analyze every Nth timestep (default: 1 = all)",
    )
    parser.add_argument(
        "--camera_id",
        type=str,
        default="camera_front_wide_120fov",
        help="Camera logical ID for image context",
    )
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="Skip sending images to Gemini (text-only mode)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed entries",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # --- Setup Gemini ---
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    gemini_client = None
    model_name = args.model

    if api_key:
        gemini_client = _create_gemini_client(api_key)
        logger.info("Using Gemini model: %s", model_name)
    else:
        logger.info("No API key provided. Running in DRY-RUN mode (trajectory analysis only).")

    # --- Find ASL files ---
    asl_files = sorted(glob.glob(args.asl_glob, recursive=True))
    if not asl_files:
        logger.error("No ASL files found matching: %s", args.asl_glob)
        return

    logger.info("Found %d ASL file(s)", len(asl_files))

    # --- Extract entries from all ASL files ---
    all_entries = []
    for asl_path in asl_files:
        logger.info("Parsing: %s", asl_path)
        entries = asyncio.run(
            extract_entries_from_asl(
                asl_path,
                camera_id=args.camera_id,
                every_nth=args.every_nth,
            )
        )
        all_entries.extend(entries)

    logger.info("Total entries to analyze: %d", len(all_entries))

    if not all_entries:
        logger.warning("No entries extracted. Check ASL files and settings.")
        return

    # --- Resume support ---
    completed_ids = set()
    results = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and output_path.exists():
        try:
            with open(output_path, "r") as f:
                saved = json.load(f)
                results = saved.get("results", saved) if isinstance(saved, dict) else saved
            completed_ids = {r["clip_id"] for r in results}
            logger.info("Resuming: %d entries already completed", len(completed_ids))
        except (json.JSONDecodeError, KeyError):
            logger.warning("Output file invalid. Starting fresh.")

    # --- Process entries ---
    use_images = not args.no_images
    for idx, entry in enumerate(all_entries):
        clip_id = entry["clip_id"]

        if clip_id in completed_ids:
            continue

        cot_text = flatten_cot(entry["chain_of_thought"])
        logger.info(
            "[%d/%d] %s — CoT: %s",
            idx + 1, len(all_entries), clip_id, cot_text[:80],
        )

        try:
            result = process_entry(
                gemini_client,
                model_name,
                entry,
                use_images=use_images,
            )
            results.append(result)

            # Print summary
            stats = result["trajectory_features"].get("summary_stats", {})
            ev = result["evaluation"]

            logger.info(
                "  Final pos: (%.1fm fwd, %.1fm lat) | Speed: %.1f m/s avg",
                stats.get("final_longitudinal_m", 0),
                stats.get("final_lateral_m", 0),
                stats.get("mean_speed_ms", 0),
            )

            if not ev.get("dry_run"):
                for dim in DIMENSIONS:
                    d = ev.get(dim, {})
                    if isinstance(d, dict) and "score" in d:
                        logger.info(
                            "  %s: %s/5 — %s",
                            dim, d["score"], d.get("justification", "")[:60],
                        )

        except Exception as e:
            logger.error("  ERROR: %s", e)
            results.append({"clip_id": clip_id, "error": str(e)})

        # Incremental save
        summary = aggregate_results(results)
        output_data = {"results": results, "summary": summary}
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        if args.delay > 0 and gemini_client is not None:
            time.sleep(args.delay)

    # --- Final summary ---
    summary = aggregate_results(results)
    output_data = {"results": results, "summary": summary}
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n{'=' * 50}")
    print(f"Results saved to: {output_path}")
    print(f"Total evaluated: {summary.get('total_evaluated', 0)}/{len(all_entries)}")
    for dim in DIMENSIONS:
        d = summary.get(dim, {})
        if d.get("count", 0) > 0:
            print(f"  {dim}: {d['mean']:.2f} ± {d['std']:.2f} (n={d['count']})")
    flagged = summary.get("flagged_entries", [])
    if flagged:
        print(f"\n⚠ {len(flagged)} low-scoring flags:")
        for f_item in flagged[:10]:
            print(f"  {f_item['clip_id']} {f_item['dimension']}: {f_item['score']}/5")


if __name__ == "__main__":
    main()
