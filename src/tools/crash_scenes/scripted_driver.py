# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""A driver service that replays a planned ego path for each scene.

It speaks the same gRPC interface as the policy driver, so the runtime, the
controller and the vehicle model treat its output like any policy's. It needs
no cameras and no GPU.

    python -m crash_scenes.scripted_driver --plans plan.json --port 6000

Each session is matched to its plan by the scene id the runtime sends when the
session starts. On every ``drive`` call the service returns the stretch of the
plan that lies ahead of the current time, and asks the runtime to end the
rollout once the plan's end time has passed.
"""

from __future__ import annotations

import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0.common_pb2 import (
    Empty,
    Pose,
    PoseAtTime,
    Quat,
    SessionRequestStatus,
    Trajectory,
    Vec3,
    VersionId,
)
from alpasim_grpc.v0.egodriver_pb2 import (
    DriveRequest,
    DriveResponse,
    DriveSessionCloseRequest,
    DriveSessionRequest,
)
from alpasim_grpc.v0.egodriver_pb2_grpc import (
    EgodriverServiceServicer,
    add_EgodriverServiceServicer_to_server,
)
from alpasim_utils.geometry import yaw_to_quat_components

import grpc

from .plans import Plan, read_plan_file

logger = logging.getLogger(__name__)

# How much of the plan each response carries, and at what spacing. Five seconds
# matches the reference the runtime builds for itself while it replays the
# recording.
LOOKAHEAD_US = 5_000_000
WAYPOINT_STEP_US = 100_000


def plan_ahead(plan: Plan, time_now_us: int) -> Trajectory:
    """The planned rig poses from ``time_now_us`` to the look-ahead horizon."""
    times_us = np.arange(
        time_now_us, time_now_us + LOOKAHEAD_US + 1, WAYPOINT_STEP_US, dtype=np.int64
    )
    times_us = times_us[times_us <= plan.timestamps_us[-1]]
    if times_us.size == 0 or times_us[0] < plan.timestamps_us[0]:
        raise ValueError(
            f"{plan.scene_id}: time {time_now_us} is outside the plan "
            f"{plan.timestamps_us[0]}..{plan.timestamps_us[-1]}"
        )

    xyz = [
        np.interp(times_us, plan.timestamps_us, plan.positions[:, i]) for i in range(3)
    ]
    yaw = np.interp(times_us, plan.timestamps_us, plan.yaw)
    poses = []
    for time_us, x, y, z, heading in zip(times_us, *xyz, yaw, strict=True):
        w, qx, qy, qz = yaw_to_quat_components(float(heading))
        poses.append(
            PoseAtTime(
                timestamp_us=int(time_us),
                pose=Pose(vec=Vec3(x=x, y=y, z=z), quat=Quat(w=w, x=qx, y=qy, z=qz)),
            )
        )
    return Trajectory(poses=poses)


class ScriptedDriver(EgodriverServiceServicer):
    def __init__(self, plans: dict[str, Plan]):
        self._plans = plans
        self._sessions: dict[str, Plan] = {}
        self._lock = threading.Lock()

    def start_session(
        self, request: DriveSessionRequest, context: grpc.ServicerContext
    ) -> SessionRequestStatus:
        scene_id = request.debug_info.scene_id
        plan = self._plans.get(scene_id)
        if plan is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"No plan for scene {scene_id!r}")
        with self._lock:
            self._sessions[request.session_uuid] = plan
        logger.info(
            "Session %s: scene %s, %s plan %s",
            request.session_uuid,
            scene_id,
            plan.variant,
            plan.crash_id,
        )
        return SessionRequestStatus()

    def close_session(
        self, request: DriveSessionCloseRequest, context: grpc.ServicerContext
    ) -> Empty:
        with self._lock:
            del self._sessions[request.session_uuid]
        return Empty()

    def get_version(self, request: Empty, context: grpc.ServicerContext) -> VersionId:
        return VersionId(
            version_id="scripted-crash-driver",
            git_hash="unknown",
            grpc_api_version=API_VERSION_MESSAGE,
        )

    def submit_image_observation(self, request, context) -> Empty:
        return Empty()

    def submit_egomotion_observation(self, request, context) -> Empty:
        return Empty()

    def submit_route(self, request, context) -> Empty:
        return Empty()

    def submit_recording_ground_truth(self, request, context) -> Empty:
        return Empty()

    def drive(
        self, request: DriveRequest, context: grpc.ServicerContext
    ) -> DriveResponse:
        with self._lock:
            plan = self._sessions[request.session_uuid]
        return DriveResponse(
            trajectory=plan_ahead(plan, request.time_now_us),
            terminate_session=request.time_now_us >= plan.end_us,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plans", required=True, help="Plan file written by `plans`")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    plans = read_plan_file(args.plans)
    server = grpc.server(ThreadPoolExecutor())
    add_EgodriverServiceServicer_to_server(ScriptedDriver(plans), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    logger.info(
        "Scripted driver on %s:%d with %d plans", args.host, args.port, len(plans)
    )
    server.wait_for_termination()


if __name__ == "__main__":
    main()
