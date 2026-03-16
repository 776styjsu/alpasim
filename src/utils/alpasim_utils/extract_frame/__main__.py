import argparse
import asyncio
import json
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.patches import Rectangle
from pathlib import Path

mpl.use("Agg")
mplstyle.use("fast")

from alpasim_utils.logs import async_read_pb_log


def render_eval_visualization(sim_result, cfg, time_now_us_target, output_path):
    # This logic matches video.py map rendering logic

    import matplotlib.pyplot as plt
    try:
        from eval.video import get_ego_transform
        from eval.video_data import ShapelyMap
        from eval.schema import MapElements
    except ImportError:
        print("eval package unavailable.")
        return

    if hasattr(sim_result, "actor_polygons"):
        sim_result.actor_polygons.artists = {}
    if hasattr(sim_result, "route"):
        sim_result.route.artists = None
    if hasattr(sim_result, "driver_responses"):
        sim_result.driver_responses.artists = None

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    # Using white background to match standard eval video
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3, color="#dddddd")

    shapely_map = ShapelyMap.from_vec_map(sim_result.vec_map) if sim_result.vec_map else None
    ego_transform = get_ego_transform(sim_result=sim_result, cfg=cfg, time=time_now_us_target)
    
    image_center_xy = sim_result.actor_polygons.set_axis_limits_around_agent(
        ax, "EGO", time_now_us_target, cfg, axis_transform=ego_transform
    )

    if shapely_map:
        shapely_map.render(
            ax, cfg, center=image_center_xy, max_dist=cfg.video.map_video.map_radius_m + 10
        )

    # Note: the exact objects in video.py
    if cfg.video.map_video.map_elements_to_plot is None or MapElements.GT_LINESTRING in cfg.video.map_video.map_elements_to_plot:
        sim_result.ego_recorded_ground_truth_trajectory.set_linestring_plot_style(
            "gt_linestring", linewidth=1, style="g-", alpha=0.7
        ).render_linestring(ax)

    if cfg.video.map_video.map_elements_to_plot is None or MapElements.AGENTS in cfg.video.map_video.map_elements_to_plot:
        sim_result.actor_polygons.render_at_time(
            ax, time_now_us_target, center=image_center_xy, max_dist=cfg.video.map_video.map_radius_m + 10
        )
    else:
        sim_result.actor_polygons.render_at_time(ax, time_now_us_target, only_agents=["EGO"])

    if cfg.video.map_video.map_elements_to_plot is None or MapElements.DRIVER_RESPONSES in cfg.video.map_video.map_elements_to_plot:
        sim_result.driver_responses.render_at_time(ax, time_now_us_target, "now")

    if (cfg.video.map_video.map_elements_to_plot is None or MapElements.ROUTE in cfg.video.map_video.map_elements_to_plot) and hasattr(sim_result, 'routes'):
        sim_result.routes.render_at_time(ax, time_now_us_target)

    if image_center_xy:
        r = cfg.video.map_video.map_radius_m
        ax.set_xlim(image_center_xy.x - r, image_center_xy.x + r)
        ax.set_ylim(image_center_xy.y - r + 0.3*r, image_center_xy.y + r + 0.3*r)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    # Extract literal geometry into a sibling JSON file
    geometry_path = output_path.with_name(f"{output_path.stem}_geometry.json")
    try:
        geometry_data = {
            "timestamp_us": int(time_now_us_target),
            "map_linestrings": [],
            "actors": []
        }
        if shapely_map:
            for ls in shapely_map.renderable_linestrings:
                geometry_data["map_linestrings"].append({
                    "type": str(ls.name),
                    "coords": list(ls.linestring.coords)
                })
        
        polys_at_time = sim_result.actor_polygons.get_polygons_at_time(time_now_us_target)
        for agent_id, poly, yaw in zip(polys_at_time.agent_ids, polys_at_time.bbox_polygons, polys_at_time.yaws):
            geometry_data["actors"].append({
                "id": agent_id,
                "yaw": float(yaw),
                "polygon_coords": list(poly.exterior.coords)
            })

        with open(geometry_path, "w") as f:
            json.dump(geometry_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to extract geometry JSON: {e}")

def save_trajectory_plot(trajectory_xy, output_path: Path):
    """Draws a neat BEV plot of the trajectory based on output visualization styles."""
    fig, ax = plt.subplots(figsize=(4.0, 6.0), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    if len(trajectory_xy) > 0:
        # In rig-frame output:
        # rx is longitudinal (forward) -> plot_y
        # ry is lateral (left) -> plot_x = -ry for standard layout (right is positive in plot)
        px = [-pt["ry"] for pt in trajectory_xy]
        py = [pt["rx"] for pt in trajectory_xy]
        
        ax.plot(
            px,
            py,
            "o-",
            color="#4ecdc4",
            linewidth=2.5,
            markersize=5,
            alpha=0.9,
        )

    # Mark current ego position (t0) as a rectangle
    car_length = 4.5
    car_width = 1.8
    ego_rect = Rectangle(
        (-car_width / 2, -car_length / 2),
        car_width,
        car_length,
        facecolor="#ffd93d",
        edgecolor="#ffffff",
        linewidth=1.5,
        zorder=5,
    )
    ax.add_patch(ego_rect)

    # Set FIXED axis limits and ticks
    ax.set_xlim(-20, 20)
    ax.set_ylim(-10, 80)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3, color="#555555")
    
    ax.tick_params(colors="#ffffff", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#555555")

    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


async def extract_inference(asl_path: str, output_dir: str, target_time_us: int | None = None):
    """
    Programmatically extracts inputs, Chain of Thought (CoT), trajectory outputs,
    and image frames from an ALPASIM .asl rollout file.
    Saves outputs to the specified directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Reading ASL file: {asl_path}")
    print(f"Extracting to directory: {out_path.absolute()}\n")
    # Preload the simulation result if eval package exists
    sim_result = None
    eval_cfg = None
    try:
        from omegaconf import OmegaConf
        from eval.schema import EvalConfig
        from eval.data import SimulationResult
        from eval.accumulator import EvalDataAccumulator
        from alpasim_utils.artifact import Artifact
        print("Discovering map artifacts for full visualization...")
        artifacts = {}
        import glob
        for path in glob.glob("data/**/*.usdz", recursive=True):
            try:
                # To prevent Torch/CUDA from stalling here unnecessarily, we just parse ID
                a = Artifact(path, _smooth_trajectories=True)
                if a.scene_id not in artifacts:
                    artifacts[a.scene_id] = a
            except Exception:
                pass
            
        eval_cfg = OmegaConf.structured(EvalConfig)
        if Path("outputs/wizard-config-loadable.yaml").exists():
            wiz_cfg = OmegaConf.load("outputs/wizard-config-loadable.yaml")
            if "eval" in wiz_cfg:
                eval_cfg = OmegaConf.merge(eval_cfg, wiz_cfg.eval)
                
        acc = EvalDataAccumulator(cfg=eval_cfg)
        preload_scene_id = "unknown"
        async for m in async_read_pb_log(asl_path):
            acc.handle_message(m)
            if m.WhichOneof("log_entry") == "rollout_metadata":
                if m.rollout_metadata.session_metadata is not None:
                    preload_scene_id = m.rollout_metadata.session_metadata.scene_id
                    
        vec_map = artifacts[preload_scene_id].map if preload_scene_id in artifacts else None
        scenario_input = acc.build_scenario_eval_input("extract", "extract", "0", vec_map)
        sim_result = SimulationResult.from_scenario_input(scenario_input, eval_cfg)
        print("Successfully pre-loaded visualization evaluation context.")
    except Exception as e:
        print(f"Could not load full visualization context, will fallback. ({e})")
    
    # Extract scene_id from metadata to build clip_id
    scene_id = "unknown"
    frame_index = 0
    
    # Track the active driver request to associate responses
    pending_request = None
    
    # Store metadata grouped by time_now_us
    # Since driver_camera_image.frame_end_us typically matches driver_request.time_now_us
    frames_metadata = {}

    async for message in async_read_pb_log(asl_path):
        msg_type = message.WhichOneof("log_entry")
        
        if msg_type == "rollout_metadata":
            if message.rollout_metadata.session_metadata is not None:
                scene_id = message.rollout_metadata.session_metadata.scene_id
        
        # 1. Inputs: Related Images (these happen BEFORE driver_request in the log)
        elif msg_type == "driver_camera_image":
            cam_img = message.driver_camera_image.camera_image
            frame_end_us = cam_img.frame_end_us # Matches time_now_us of the inference step
            
            # Skip images not matching our target frame (if specified)
            if target_time_us and frame_end_us != target_time_us:
                continue
                
            camera_name = cam_img.logical_id
            image_bytes = cam_img.image_bytes 
            
            # Store images in metadata dict immediately by their end time
            # Since driver_request hasn't happened yet, we initialize a shell for this frame
            if frame_end_us not in frames_metadata:
                clip_id = f"{scene_id}_t{frame_index:04d}"
                frames_metadata[frame_end_us] = {
                    "clip_id": clip_id,
                    "time_now_us": frame_end_us,
                    "chain_of_thought": None,
                    "trajectory_poses": [],
                    "images_pending_save": []
                }
                frame_index += 1
                
            frames_metadata[frame_end_us]["images_pending_save"].append((camera_name, image_bytes))
            
        # 2. Inputs: Driver Requests
        elif msg_type == "driver_request":
            time_query_us = message.driver_request.time_query_us
            time_now_us = message.driver_request.time_now_us
            
            # If target_time_us is specified, skip if we don't match exactly
            if target_time_us and time_query_us != target_time_us and time_now_us != target_time_us:
                continue
            
            # The images should have created the dict key already via time_now_us
            # We just need to record the time_query_us and link the pending_request
            if time_now_us not in frames_metadata:
                # Fallback if no images fired
                clip_id = f"{scene_id}_t{frame_index:04d}"
                frames_metadata[time_now_us] = {
                    "clip_id": clip_id,
                    "time_now_us": time_now_us,
                    "chain_of_thought": None,
                    "trajectory_poses": [],
                    "images_pending_save": []
                }
                frame_index += 1
                
            frames_metadata[time_now_us]["time_query_us"] = time_query_us
            pending_request = time_query_us
            
            # Keep a reverse mapping so we can find the time_now_us from time_query_us
            # Since we just need the dictionary reference directly, let's keep a lookup pointer
            # We don't overwrite frames_metadata[time_query_us] if it conflicts. 
            pass

        # 3. Outputs & Chain-Of-Thought
        elif msg_type == "driver_return":
            resp = message.driver_return
            
            if pending_request is None:
                continue
            
            # Use the currently tracked pending_request from the previous driver_request parsing
            meta_match = None
            for md in frames_metadata.values():
                if isinstance(md, dict) and md.get("time_query_us") == pending_request:
                    meta_match = md
                    break
            
            if not meta_match:
                continue
            
            meta = meta_match
            
            # 3a. Outputs  (Rig-frame relative XY trajectory like in output videos)
            trajectory_xy = []
            if len(resp.trajectory.poses) >= 1:
                anchor = resp.trajectory.poses[0]
                anchor_x = anchor.pose.vec.x
                anchor_y = anchor.pose.vec.y
                anchor_yaw = np.arctan2(
                    2.0 * (anchor.pose.quat.w * anchor.pose.quat.z + anchor.pose.quat.x * anchor.pose.quat.y),
                    1.0 - 2.0 * (anchor.pose.quat.y ** 2 + anchor.pose.quat.z ** 2),
                )
                cos_yaw = np.cos(-anchor_yaw)
                sin_yaw = np.sin(-anchor_yaw)
                
                # Starting from anchor (t=0, ego rig-frame offset [0, 0])
                for pt in resp.trajectory.poses:
                    dx = pt.pose.vec.x - anchor_x
                    dy = pt.pose.vec.y - anchor_y
                    rx = cos_yaw * dx - sin_yaw * dy
                    ry = sin_yaw * dx + cos_yaw * dy
                    trajectory_xy.append({
                        "timestamp_us": pt.timestamp_us,
                        "rx": rx,
                        "ry": ry
                    })
            
            meta["trajectory_xy_rig_frame"] = trajectory_xy
            
            # Keep keeping raw global positions for completion
            poses = []
            for p in resp.trajectory.poses:
                poses.append({
                    "timestamp_us": p.timestamp_us,
                    "x": p.pose.vec.x,
                    "y": p.pose.vec.y,
                    "z": p.pose.vec.z
                })
            meta["trajectory_poses"] = poses
            
            # 3b. Chain of Thought
            dbg_bytes = resp.debug_info.unstructured_debug_info
            if dbg_bytes:
                try:
                    extra_payload = pickle.loads(dbg_bytes)
                    if isinstance(extra_payload, dict):
                        meta["chain_of_thought"] = extra_payload.get("reasoning_text")
                except Exception:
                    pass
            
            # We are complete! Output the clip structures directly using its clip_id
            clip_id = meta["clip_id"]
            clip_dir = out_path / clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)
            
            # Dump the images
            for cam_name, img_bytes in meta.get("images_pending_save", []):
                image_path = clip_dir / f"{cam_name}.jpg"
                with open(image_path, "wb") as f:
                    f.write(img_bytes)
                print(f"Saved Image: {image_path}")
            
            # Dump the metadata
            meta_path = clip_dir / "metadata.json"
            
            # Strip the temporary cached binary data before stringifying
            meta_to_save = {k: v for k, v in meta.items() if k != "images_pending_save"}
            
            with open(meta_path, "w") as f:
                json.dump(meta_to_save, f, indent=2)
                
            print(f"Saved Metadata: {meta_path}")
            
            # Save the plotted trajectory image
            plot_path = clip_dir / "trajectory_plot.png"
            if sim_result is not None and eval_cfg is not None:
                try:
                    time_uint64 = np.uint64(meta["time_now_us"])
                    render_eval_visualization(sim_result, eval_cfg, time_uint64, plot_path)
                except Exception as e:
                    print(f"Eval visualization failed, fallback to trajectory plot. ({e})")
                    save_trajectory_plot(meta["trajectory_xy_rig_frame"], plot_path)
            else:
                save_trajectory_plot(meta["trajectory_xy_rig_frame"], plot_path)
            
            print(f"Saved Trajectory Plot: {plot_path}")
            
            pending_request = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract inferences, CoT, and images from ASL files to disk.")
    parser.add_argument("asl_path", type=str, help="Path to the rollout.asl file")
    parser.add_argument("--output-dir", type=str, default="extracted_frames", help="Directory where structures will be saved")
    parser.add_argument("--time", type=int, default=None, help="Target specific time_query_us or time_now_us to extract a single frame")
    args = parser.parse_args()

    # Create an asyncio event loop to run the async generator
    asyncio.run(extract_inference(args.asl_path, args.output_dir, target_time_us=args.time))