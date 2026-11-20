# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Label crashes: what the ego hit, and where each of the two was hit.

AlpaSim has no contact physics, so boxes pass through each other. The label is
therefore taken at the first instant of contact, found by scanning between the
logged 0.1 s steps; the first logged overlap can already be metres deep and
point at the wrong side.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import nearest_points

from .boxes import (
    box_corners,
    boxes_gap,
    boxes_overlap,
    boxes_penetration,
    impact_zone,
    to_box_frame,
)
from .tracks import ActorTrack, RolloutTracks

# Scene class labels (``tracks_label_class`` in a scene's sequence_tracks.json)
# grouped by the kind of road user.
ACTOR_GROUPS = {
    "automobile": "car",
    "other_vehicle": "car",
    "heavy_truck": "truck",
    "trailer": "truck",
    "bus": "bus",
    "train_or_tram_car": "bus",
    "rider": "cyclist",
    "person": "pedestrian",
    "stroller": "pedestrian",
    "animal": "animal",
    "protruding_object": "object",
}

UNORIENTED_GROUPS = ("pedestrian", "animal", "object")

SCAN_STEP_US = 10_000
CONTACT_RESOLUTION_US = 1_000
# A party whose own motion closes on the contact point faster than this struck
# the other one.
STRIKING_SPEED_MPS = 0.5
# Headings within this of parallel or anti-parallel count as a glancing blow.
SIDESWIPE_ANGLE_DEG = 30.0


@dataclass(frozen=True)
class CrashLabel:
    actor_id: str
    actor_label: str
    actor_group: str
    actor_is_static: bool
    time_us: int
    # True when the boxes already overlapped at the start of the labelled
    # window, so no moment of first contact exists.
    overlap_at_start: bool
    ego_zone8: str
    ego_zone4: str
    ego_contact_uv: tuple[float, float]
    actor_zone8: str
    actor_zone4: str
    actor_contact_uv: tuple[float, float]
    crash_type: str
    striking: str  # ego | actor | both | neither
    ego_speed_mps: float
    actor_speed_mps: float
    closing_speed_mps: float
    heading_difference_deg: float
    overlap_duration_s: float
    # Deepest the boxes get into each other while the ego drives through; a
    # corner graze stays shallow.
    max_penetration_m: float


@dataclass(frozen=True)
class ClosestApproach:
    actor_id: str
    actor_label: str
    actor_group: str
    gap_m: float
    time_us: int


def _overlap_at(ego: ActorTrack, actor: ActorTrack, times_us: np.ndarray) -> np.ndarray:
    return boxes_overlap(
        ego.poses_at(times_us), ego.size, actor.poses_at(times_us), actor.size
    )


def _first_contact_us(
    ego: ActorTrack, actor: ActorTrack, scan_us: np.ndarray, overlap: np.ndarray
) -> int:
    """Refine the first overlapping scan time to ``CONTACT_RESOLUTION_US``."""
    first = int(np.argmax(overlap))
    if first == 0:
        return int(scan_us[0])
    fine = np.arange(
        scan_us[first - 1], scan_us[first] + 1, CONTACT_RESOLUTION_US, dtype=np.int64
    )
    return int(fine[np.argmax(_overlap_at(ego, actor, fine))])


def _contact_point(
    ego_pose: np.ndarray, ego: ActorTrack, actor_pose: np.ndarray, actor: ActorTrack
) -> np.ndarray:
    """Centre of the overlap region; at first contact that is the touching point."""
    ego_box = Polygon(box_corners(ego_pose, ego.size))
    actor_box = Polygon(box_corners(actor_pose, actor.size))
    overlap = ego_box.intersection(actor_box)
    if overlap.is_empty:
        # Boxes that only just touch can intersect in nothing at float precision.
        on_ego, on_actor = nearest_points(ego_box, actor_box)
        return (np.array(on_ego.coords[0]) + np.array(on_actor.coords[0])) / 2.0
    return np.array(overlap.centroid.coords[0])


def _crash_type(
    actor_group: str, ego_zone4: str, actor_zone4: str, heading_difference_deg: float
) -> str:
    """Name the collision from the two struck faces and the heading difference.

    A pedestrian, animal or object has no front or side worth naming, so those
    are typed by the struck face of the ego alone.
    """
    sides = ("left", "right")
    if actor_group in UNORIENTED_GROUPS:
        return f"{'side' if ego_zone4 in sides else ego_zone4}_impact"
    glancing = min(heading_difference_deg, 180.0 - heading_difference_deg) < (
        SIDESWIPE_ANGLE_DEG
    )
    same_direction = heading_difference_deg < 90.0
    sideswipe = (
        "sideswipe_same_direction" if same_direction else "sideswipe_opposite_direction"
    )

    if ego_zone4 == "front" and actor_zone4 == "rear":
        return "rear_end_ego_striking"
    if ego_zone4 == "rear" and actor_zone4 == "front":
        return "rear_end_ego_struck"
    if ego_zone4 == "front" and actor_zone4 == "front":
        return "head_on" if not same_direction else "angle_front_to_front"
    if ego_zone4 in sides and actor_zone4 in sides:
        return sideswipe
    if ego_zone4 == "front" and actor_zone4 in sides:
        return sideswipe if glancing else "angle_ego_striking"
    if ego_zone4 in sides and actor_zone4 == "front":
        return sideswipe if glancing else "angle_ego_struck"
    return "other"


def _closing_speed(
    velocity: np.ndarray, centre: np.ndarray, contact: np.ndarray
) -> float:
    """Speed at which a box's own motion carries it towards the contact point."""
    towards = contact - centre
    distance = float(np.linalg.norm(towards))
    if distance == 0.0:
        return 0.0
    return float(velocity @ towards) / distance


def _label_contact(
    ego: ActorTrack,
    actor: ActorTrack,
    time_us: int,
    overlap_at_start: bool,
    overlap_duration_s: float,
    max_penetration_m: float,
) -> CrashLabel:
    time = np.array([time_us])
    ego_pose = ego.poses_at(time)[0]
    actor_pose = actor.poses_at(time)[0]
    contact = _contact_point(ego_pose, ego, actor_pose, actor)

    ego_zone = impact_zone(to_box_frame(contact, ego_pose), ego.length_m, ego.width_m)
    actor_zone = impact_zone(
        to_box_frame(contact, actor_pose), actor.length_m, actor.width_m
    )

    ego_velocity = ego.velocity_at(time_us)
    actor_velocity = actor.velocity_at(time_us)
    ego_closing = _closing_speed(ego_velocity, ego_pose[:2], contact)
    actor_closing = _closing_speed(actor_velocity, actor_pose[:2], contact)
    ego_strikes = ego_closing > STRIKING_SPEED_MPS
    actor_strikes = actor_closing > STRIKING_SPEED_MPS
    striking = {
        (True, True): "both",
        (True, False): "ego",
        (False, True): "actor",
        (False, False): "neither",
    }[(ego_strikes, actor_strikes)]

    heading_difference = abs(
        math.degrees(
            (actor_pose[2] - ego_pose[2] + math.pi) % (2.0 * math.pi) - math.pi
        )
    )

    return CrashLabel(
        actor_id=actor.actor_id,
        actor_label=actor.label,
        actor_group=ACTOR_GROUPS[actor.label],
        actor_is_static=actor.is_static,
        time_us=time_us,
        overlap_at_start=overlap_at_start,
        ego_zone8=ego_zone["zone8"],
        ego_zone4=ego_zone["zone4"],
        ego_contact_uv=(ego_zone["u"], ego_zone["v"]),
        actor_zone8=actor_zone["zone8"],
        actor_zone4=actor_zone["zone4"],
        actor_contact_uv=(actor_zone["u"], actor_zone["v"]),
        crash_type=_crash_type(
            ACTOR_GROUPS[actor.label],
            ego_zone["zone4"],
            actor_zone["zone4"],
            heading_difference,
        ),
        striking=striking,
        ego_speed_mps=round(float(np.linalg.norm(ego_velocity)), 2),
        actor_speed_mps=round(float(np.linalg.norm(actor_velocity)), 2),
        closing_speed_mps=round(
            float(np.linalg.norm(ego_velocity - actor_velocity)), 2
        ),
        heading_difference_deg=round(heading_difference, 1),
        overlap_duration_s=round(overlap_duration_s, 2),
        max_penetration_m=round(max_penetration_m, 3),
    )


def _scan(
    ego: ActorTrack, actor: ActorTrack, start_us: int, end_us: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Scan times and both actors' poses where their tracks share the window."""
    first_us = max(start_us, ego.start_us, actor.start_us)
    last_us = min(end_us, ego.end_us, actor.end_us)
    if last_us <= first_us:
        return None
    scan_us = np.arange(first_us, last_us + 1, SCAN_STEP_US, dtype=np.int64)
    return scan_us, ego.poses_at(scan_us), actor.poses_at(scan_us)


def _reach(ego: ActorTrack, actor: ActorTrack) -> float:
    """Centre distance beyond which the two boxes cannot touch."""
    return float(np.hypot(*ego.size) + np.hypot(*actor.size)) / 2.0


def find_crashes(
    ego: ActorTrack, actors: list[ActorTrack], start_us: int, end_us: int
) -> list[CrashLabel]:
    """Every actor the ego touches in ``[start_us, end_us]``, by time of first contact.

    Only the first crash is physically meaningful: afterwards the ego keeps
    driving through whatever it hit.
    """
    crashes: list[CrashLabel] = []
    for actor in actors:
        scan = _scan(ego, actor, start_us, end_us)
        if scan is None:
            continue
        scan_us, ego_poses, actor_poses = scan
        centre_distance = np.linalg.norm(ego_poses[:, :2] - actor_poses[:, :2], axis=1)

        overlap = np.zeros(len(scan_us), dtype=bool)
        near = centre_distance <= _reach(ego, actor)
        if near.any():
            overlap[near] = boxes_overlap(
                ego_poses[near], ego.size, actor_poses[near], actor.size
            )
        if not overlap.any():
            continue

        crashes.append(
            _label_contact(
                ego,
                actor,
                time_us=_first_contact_us(ego, actor, scan_us, overlap),
                overlap_at_start=bool(overlap[0]),
                overlap_duration_s=float(overlap.sum()) * SCAN_STEP_US / 1e6,
                max_penetration_m=float(
                    boxes_penetration(
                        ego_poses[overlap], ego.size, actor_poses[overlap], actor.size
                    ).max()
                ),
            )
        )
    crashes.sort(key=lambda crash: crash.time_us)
    return crashes


def closest_approach(
    ego: ActorTrack, actors: list[ActorTrack], start_us: int, end_us: int
) -> ClosestApproach | None:
    """The actor that comes closest to the ego in ``[start_us, end_us]``."""
    closest: ClosestApproach | None = None
    for actor in actors:
        scan = _scan(ego, actor, start_us, end_us)
        if scan is None:
            continue
        scan_us, ego_poses, actor_poses = scan
        centre_distance = np.linalg.norm(ego_poses[:, :2] - actor_poses[:, :2], axis=1)
        # The gap is at least the centre distance less the reach, so only
        # samples that can still beat the best gap need the exact test.
        bound = math.inf if closest is None else closest.gap_m
        candidates = centre_distance - _reach(ego, actor) < min(
            bound, float(centre_distance.min())
        )
        if not candidates.any():
            continue
        gaps = boxes_gap(
            ego_poses[candidates], ego.size, actor_poses[candidates], actor.size
        )
        index = int(np.argmin(gaps))
        if gaps[index] < bound:
            closest = ClosestApproach(
                actor_id=actor.actor_id,
                actor_label=actor.label,
                actor_group=ACTOR_GROUPS[actor.label],
                gap_m=round(float(gaps[index]), 3),
                time_us=int(scan_us[candidates][index]),
            )
    return closest


def label_tracks(
    ego: ActorTrack,
    actors: list[ActorTrack],
    start_us: int,
    end_us: int,
) -> tuple[list[CrashLabel], ClosestApproach | None]:
    """Crashes in ``[start_us, end_us]``, and the closest approach among the rest."""
    crashes = find_crashes(ego, actors, start_us, end_us)
    hit = {crash.actor_id for crash in crashes}
    untouched = [actor for actor in actors if actor.actor_id not in hit]
    return crashes, closest_approach(ego, untouched, start_us, end_us)


def nearest_pass(
    ego: ActorTrack, actor: ActorTrack, start_us: int, end_us: int
) -> tuple[float, int] | None:
    """Smallest gap between the ego and one actor in the window, and its time.

    The gap is zero if they touch. None if the two are never present together.
    """
    scan = _scan(ego, actor, start_us, end_us)
    if scan is None:
        return None
    scan_us, ego_poses, actor_poses = scan
    gaps = boxes_gap(ego_poses, ego.size, actor_poses, actor.size)
    index = int(np.argmin(gaps))
    return float(gaps[index]), int(scan_us[index])


def gap_to(ego: ActorTrack, actor: ActorTrack, start_us: int, end_us: int) -> float:
    """Smallest gap between the ego and one actor; infinite if they never meet."""
    nearest = nearest_pass(ego, actor, start_us, end_us)
    return math.inf if nearest is None else nearest[0]


def label_rollout(rollout: RolloutTracks) -> dict:
    """Crash labels for one rollout, from the moment the policy takes over."""
    crashes, closest = label_tracks(
        rollout.ego,
        list(rollout.actors.values()),
        start_us=rollout.closed_loop_start_us,
        end_us=rollout.ego.end_us,
    )
    return {
        "scene_id": rollout.scene_id,
        "session_uuid": rollout.session_uuid,
        "closed_loop_start_us": rollout.closed_loop_start_us,
        "end_us": rollout.ego.end_us,
        "crashes": [asdict(crash) for crash in crashes],
        "closest_approach": None if closest is None else asdict(closest),
    }
