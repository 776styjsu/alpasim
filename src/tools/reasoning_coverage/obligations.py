# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Versioned causal-reasoning obligations for Alpamayo traces.

The catalog is deliberately independent of any model output and safety outcome.
Version 1 contains the two critical-object families that can be triggered from
the recorded actor state without inferring map rules or route intent.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable

from .program import ReasoningProgram

CATALOG_VERSION = "alpamayo-cro-v1"
DISTANCE_BANDS = ("near", "mid", "far")
LEAD_ACTOR_GROUPS = ("light_vehicle", "heavy_vehicle")
CROSSING_ACTOR_GROUPS = (*LEAD_ACTOR_GROUPS, "vru")
LEAD_MOTION_STATES = ("stopped", "slower", "not_slower")
REASONING_STATUSES = (
    "exercised_not_recognized",
    "expected_decision_without_recognition",
    "recognized_other_decision",
    "recognized_expected_decision",
)
ALPAMAYO_LONGITUDINAL_DECISIONS = (
    "none",
    "set_speed_tracking",
    "lead_obstacle_following",
    "speed_adaptation_road_events",
    "gap_searching_for_lane_change_merge_zipper",
    "acceleration_for_passing_overtaking",
    "yield_agent_right_of_way",
    "stop_for_static_constraints",
)
ALPAMAYO_LATERAL_DECISIONS = (
    "none",
    "lane_keeping_centering",
    "merge_split_facility_change",
    "out_of_lane_nudge",
    "in_lane_nudge",
    "lane_change",
    "pull_over_curb_approach",
    "turn",
    "lateral_maneuver_abort",
)

CRITICAL_COMPONENT_DOMAINS = (
    ("critical_objects", "independent_partial"),
    ("traffic_lights", "not_observable_in_v1"),
    ("stop_yield_control", "not_observable_in_v1"),
    ("road_events", "not_observable_in_v1"),
    ("lanes_and_lane_lines", "not_observable_in_v1"),
    ("routing_intent", "not_observable_in_v1"),
    ("odd_constraints", "not_observable_in_v1"),
)

_LIGHT_VEHICLE_LABELS = frozenset({"automobile", "other_vehicle"})
_HEAVY_VEHICLE_LABELS = frozenset(
    {"heavy_truck", "bus", "trailer", "train_or_tram_car"}
)
_VRU_LABELS = frozenset({"person", "rider", "animal", "stroller"})


@dataclass(frozen=True, order=True)
class ObligationDefinition:
    """One feasible cell in the frozen v1 denominator."""

    family: str
    actor_group: str
    distance_band: str
    motion_state: str
    expected_longitudinal_decision: str

    @property
    def obligation_id(self) -> str:
        return ":".join(
            (self.family, self.actor_group, self.motion_state, self.distance_band)
        )


@dataclass(frozen=True, order=True)
class ActorState:
    """Minimal actor state used by the outcome-blind applicability oracle."""

    actor_id: str
    actor_label: str
    x_m: float
    y_m: float
    yaw_rad: float
    velocity_x_mps: float
    velocity_y_mps: float
    length_m: float
    width_m: float


@dataclass(frozen=True, order=True)
class ApplicableObligation:
    """A catalog obligation independently triggered for one decision frame."""

    obligation_id: str
    family: str
    actor_group: str
    actor_id: str
    distance_band: str
    motion_state: str
    distance_m: float
    time_to_closest_approach_s: float | None
    distance_at_closest_approach_m: float | None


def catalog() -> tuple[ObligationDefinition, ...]:
    """Return the complete, outcome-blind v1 catalog."""
    definitions = [
        ObligationDefinition(
            family="lead_agent_following",
            actor_group=actor_group,
            distance_band=distance,
            motion_state=motion,
            expected_longitudinal_decision="lead_obstacle_following",
        )
        for actor_group in LEAD_ACTOR_GROUPS
        for motion in LEAD_MOTION_STATES
        for distance in DISTANCE_BANDS
    ]
    definitions.extend(
        ObligationDefinition(
            family="crossing_agent_yield",
            actor_group=actor_group,
            distance_band=distance,
            motion_state="moving",
            expected_longitudinal_decision="yield_agent_right_of_way",
        )
        for actor_group in CROSSING_ACTOR_GROUPS
        for distance in DISTANCE_BANDS
    )
    return tuple(sorted(definitions))


def catalog_sha256() -> str:
    payload = {
        "version": CATALOG_VERSION,
        "definitions": [asdict(item) for item in catalog()],
        "reasoning_statuses": REASONING_STATUSES,
        "critical_component_domains": CRITICAL_COMPONENT_DOMAINS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def decision_schema_sha256() -> str:
    payload = {
        "longitudinal_decisions": ALPAMAYO_LONGITUDINAL_DECISIONS,
        "lateral_decisions": ALPAMAYO_LATERAL_DECISIONS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def actor_group(actor_label: str) -> str | None:
    if actor_label in _LIGHT_VEHICLE_LABELS:
        return "light_vehicle"
    if actor_label in _HEAVY_VEHICLE_LABELS:
        return "heavy_vehicle"
    if actor_label in _VRU_LABELS:
        return "vru"
    return None


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _distance_band(distance_m: float) -> str:
    if distance_m <= 15.0:
        return "near"
    if distance_m <= 35.0:
        return "mid"
    return "far"


def _definition_id(
    family: str, actor_group_name: str, motion_state: str, distance_band: str
) -> str:
    return ":".join((family, actor_group_name, motion_state, distance_band))


def applicable_obligations(
    actors: Iterable[ActorState], *, ego_actor_id: str = "EGO"
) -> tuple[ApplicableObligation, ...]:
    """Trigger high-confidence actor obligations from one timestamped snapshot.

    ``lead_agent_following`` uses the nearest aligned vehicle whose center lies
    in an AABB-width corridor ahead of ego. ``crossing_agent_yield`` requires a
    transverse heading and a constant-velocity closest approach within five
    seconds and the actors' AABB circumscribed radii plus 0.5 m.
    """
    by_id = {actor.actor_id: actor for actor in actors}
    if ego_actor_id not in by_id:
        raise ValueError(f"Snapshot has no {ego_actor_id} actor")
    ego = by_id[ego_actor_id]
    cos_yaw = math.cos(ego.yaw_rad)
    sin_yaw = math.sin(ego.yaw_rad)
    ego_speed = math.hypot(ego.velocity_x_mps, ego.velocity_y_mps)

    lead_candidates: list[tuple[float, ActorState, str]] = []
    crossing: list[ApplicableObligation] = []
    for actor in by_id.values():
        if actor.actor_id == ego_actor_id:
            continue
        group = actor_group(actor.actor_label)
        if group is None:
            continue
        dx = actor.x_m - ego.x_m
        dy = actor.y_m - ego.y_m
        forward = cos_yaw * dx + sin_yaw * dy
        left = -sin_yaw * dx + cos_yaw * dy
        distance = math.hypot(dx, dy)
        heading_difference = abs(_wrap_angle(actor.yaw_rad - ego.yaw_rad))
        corridor_half_width = 0.5 * (ego.width_m + actor.width_m) + 0.75

        if (
            group in LEAD_ACTOR_GROUPS
            and 0.0 < forward <= 60.0
            and abs(left) <= corridor_half_width
            and heading_difference <= math.radians(30.0)
        ):
            lead_candidates.append((forward, actor, group))

        actor_speed = math.hypot(actor.velocity_x_mps, actor.velocity_y_mps)
        if (
            group not in CROSSING_ACTOR_GROUPS
            or distance > 60.0
            or actor_speed < 0.5
            or not math.radians(45.0) <= heading_difference <= math.radians(135.0)
        ):
            continue
        relative_vx = actor.velocity_x_mps - ego.velocity_x_mps
        relative_vy = actor.velocity_y_mps - ego.velocity_y_mps
        relative_speed_sq = relative_vx**2 + relative_vy**2
        if relative_speed_sq < 0.25:
            continue
        tcpa = -(dx * relative_vx + dy * relative_vy) / relative_speed_sq
        if not 0.0 <= tcpa <= 5.0:
            continue
        dcpa = math.hypot(dx + tcpa * relative_vx, dy + tcpa * relative_vy)
        collision_radius = 0.5 * math.hypot(
            ego.length_m, ego.width_m
        ) + 0.5 * math.hypot(actor.length_m, actor.width_m)
        if dcpa > collision_radius + 0.5:
            continue
        band = _distance_band(distance)
        obligation_id = _definition_id("crossing_agent_yield", group, "moving", band)
        crossing.append(
            ApplicableObligation(
                obligation_id=obligation_id,
                family="crossing_agent_yield",
                actor_group=group,
                actor_id=actor.actor_id,
                distance_band=band,
                motion_state="moving",
                distance_m=distance,
                time_to_closest_approach_s=tcpa,
                distance_at_closest_approach_m=dcpa,
            )
        )

    result = crossing
    if lead_candidates:
        forward, actor, group = min(lead_candidates, key=lambda item: item[0])
        actor_forward_speed = (
            cos_yaw * actor.velocity_x_mps + sin_yaw * actor.velocity_y_mps
        )
        if math.hypot(actor.velocity_x_mps, actor.velocity_y_mps) <= 0.5:
            motion = "stopped"
        elif ego_speed - actor_forward_speed >= 1.0:
            motion = "slower"
        else:
            motion = "not_slower"
        band = _distance_band(forward)
        obligation_id = _definition_id("lead_agent_following", group, motion, band)
        result.append(
            ApplicableObligation(
                obligation_id=obligation_id,
                family="lead_agent_following",
                actor_group=group,
                actor_id=actor.actor_id,
                distance_band=band,
                motion_state=motion,
                distance_m=forward,
                time_to_closest_approach_s=None,
                distance_at_closest_approach_m=None,
            )
        )
    known_ids = {definition.obligation_id for definition in catalog()}
    unknown = {item.obligation_id for item in result} - known_ids
    if unknown:
        raise AssertionError(
            f"Applicability oracle emitted unknown obligations: {unknown}"
        )
    return tuple(sorted(set(result)))


def reasoning_status(
    program: ReasoningProgram, obligation: ApplicableObligation
) -> str:
    """Classify how the trace handles one independently triggered obligation."""
    predicates = {fact.predicate for step in program.steps for fact in step.facts}
    decisions = {step.longitudinal_decision for step in program.steps}
    if obligation.family == "lead_agent_following":
        recognized = bool(predicates & {"lead_present", "lead_slow", "lead_stopped"})
        expected_decision = "lead_obstacle_following"
    elif obligation.family == "crossing_agent_yield":
        recognized = "crossing_agent" in predicates
        expected_decision = "yield_agent_right_of_way"
    else:
        raise ValueError(f"Unknown obligation family: {obligation.family}")
    has_expected_decision = expected_decision in decisions
    if recognized and has_expected_decision:
        return "recognized_expected_decision"
    if recognized:
        return "recognized_other_decision"
    if has_expected_decision:
        return "expected_decision_without_recognition"
    return "exercised_not_recognized"


def applicability_items(obligations: Iterable[ApplicableObligation]) -> frozenset[str]:
    """Return hierarchical fixed-catalog coverage items for a frame."""
    items: set[str] = set()
    for obligation in obligations:
        prefix = f"cro:{obligation.family}"
        items.update(
            {
                f"cro:family={obligation.family}",
                f"{prefix}:actor_group={obligation.actor_group}",
                f"{prefix}:motion_state={obligation.motion_state}",
                f"{prefix}:distance_band={obligation.distance_band}",
                f"cro:variant={obligation.obligation_id}",
            }
        )
    return frozenset(items)


def status_items(
    program: ReasoningProgram, obligations: Iterable[ApplicableObligation]
) -> frozenset[str]:
    """Return applicability items plus obligation-discharge status cells."""
    obligations = tuple(obligations)
    items = set(applicability_items(obligations))
    for obligation in obligations:
        status = reasoning_status(program, obligation)
        items.update(
            {
                f"cro-status:family={obligation.family}|{status}",
                f"cro-status:variant={obligation.obligation_id}|{status}",
            }
        )
    return frozenset(items)


def alpamayo_decision_pairs() -> tuple[str, ...]:
    """Return the 71 non-empty pairs in the published high-level schema."""
    return tuple(
        f"{longitudinal}|{lateral}"
        for longitudinal in ALPAMAYO_LONGITUDINAL_DECISIONS
        for lateral in ALPAMAYO_LATERAL_DECISIONS
        if (longitudinal, lateral) != ("none", "none")
    )


def program_decision_pairs(program: ReasoningProgram) -> frozenset[str]:
    """Map compiler decisions into the published Alpamayo pair space."""
    result: set[str] = set()
    for step in program.steps:
        longitudinal = step.longitudinal_decision
        lateral = step.lateral_decision
        if longitudinal == "no_longitudinal_change":
            longitudinal = "set_speed_tracking"
        elif longitudinal == "not_stated":
            longitudinal = "none"
        if lateral == "not_stated":
            lateral = "none"
        if longitudinal not in ALPAMAYO_LONGITUDINAL_DECISIONS:
            continue
        if lateral not in ALPAMAYO_LATERAL_DECISIONS:
            continue
        if (longitudinal, lateral) != ("none", "none"):
            result.add(f"{longitudinal}|{lateral}")
    return frozenset(result)


def decision_items(
    program: ReasoningProgram, obligations: Iterable[ApplicableObligation]
) -> frozenset[str]:
    """Return closed obligation-conditioned decision-pair coverage items."""
    obligations = tuple(obligations)
    items = set(status_items(program, obligations))
    pairs = program_decision_pairs(program)
    for obligation in obligations:
        for pair in pairs:
            items.update(
                {
                    f"cro-decision:family={obligation.family}|{pair}",
                    f"cro-decision:variant={obligation.obligation_id}|{pair}",
                }
            )
    return frozenset(items)


def denominator_items(
    *, include_status: bool, include_decisions: bool = False
) -> frozenset[str]:
    """Return the complete fixed denominator for catalog coverage reporting."""
    fake_items: set[str] = set()
    for definition in catalog():
        prefix = f"cro:{definition.family}"
        fake_items.update(
            {
                f"cro:family={definition.family}",
                f"{prefix}:actor_group={definition.actor_group}",
                f"{prefix}:motion_state={definition.motion_state}",
                f"{prefix}:distance_band={definition.distance_band}",
                f"cro:variant={definition.obligation_id}",
            }
        )
        if include_status:
            for status in REASONING_STATUSES:
                fake_items.update(
                    {
                        f"cro-status:family={definition.family}|{status}",
                        f"cro-status:variant={definition.obligation_id}|{status}",
                    }
                )
        if include_decisions:
            for pair in alpamayo_decision_pairs():
                fake_items.update(
                    {
                        f"cro-decision:family={definition.family}|{pair}",
                        f"cro-decision:variant={definition.obligation_id}|{pair}",
                    }
                )
    return frozenset(fake_items)
