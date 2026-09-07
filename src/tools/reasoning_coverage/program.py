# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Closed, text-only reasoning-program intermediate representation.

The compiler is deliberately deterministic and only labels information stated
in the chain of thought.  It never reads images, trajectories, scene labels, or
outcomes.  This makes it suitable for large blinded sensitivity experiments;
it is not a substitute for a human-validated semantic parser.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Iterable

FACT_PREDICATES = (
    "lead_present",
    "lead_slow",
    "lead_stopped",
    "merging_or_cutin_agent",
    "crossing_agent",
    "oncoming_agent",
    "adjacent_agent",
    "vru_present",
    "static_obstacle",
    "path_clear",
    "gap_available",
    "gap_blocked",
    "lane_blocked_or_narrow",
    "lane_position_or_boundary_constraint",
    "signal_go",
    "signal_caution",
    "signal_stop",
    "stop_control",
    "yield_or_right_of_way_requirement",
    "directional_control",
    "intersection_or_turn",
    "road_curve",
    "merge_split_or_ramp",
    "speed_constraint",
    "route_maneuver_requirement",
    "workzone_or_road_event",
    "other_explicit",
)
OBJECTIVES = (
    "nominal_progress",
    "maintain_following_gap",
    "avoid_collision_or_increase_clearance",
    "obey_signal_or_stop_control",
    "yield_right_of_way",
    "adapt_speed_to_road",
    "acquire_merge_or_lane_change_gap",
    "pass_slow_or_blocking_agent",
    "satisfy_route_or_lane_requirement",
    "remain_in_lane_or_drivable_area",
    "other_explicit",
    "not_stated",
)
LONGITUDINAL_DECISIONS = (
    "set_speed_tracking",
    "lead_obstacle_following",
    "stop_for_static_constraints",
    "gap_searching_for_lane_change_merge_zipper",
    "yield_agent_right_of_way",
    "acceleration_for_passing_overtaking",
    "speed_adaptation_road_events",
    "no_longitudinal_change",
    "not_stated",
    "ambiguous",
)
LATERAL_DECISIONS = (
    "lane_keeping_centering",
    "in_lane_nudge",
    "lane_change",
    "merge_split_facility_change",
    "turn",
    "not_stated",
    "ambiguous",
)
LONGITUDINAL_ACTIONS = (
    "accelerate",
    "decelerate",
    "stop",
    "maintain_speed",
    "reverse",
    "not_stated",
    "ambiguous",
)
LATERAL_ACTIONS = (
    "steer_left",
    "steer_right",
    "go_straight",
    "reverse_left",
    "reverse_right",
    "not_stated",
    "ambiguous",
)
ENTRY_CONDITIONS = (
    "immediate",
    "after_previous_complete",
    "when_gap_available",
    "when_path_clear",
    "when_signal_permits",
    "until_stop_or_yield_satisfied",
    "not_stated",
)


@dataclass(frozen=True, order=True)
class Fact:
    predicate: str
    location: str = "unspecified"
    modality: str = "asserted"


@dataclass(frozen=True)
class ProgramStep:
    facts: tuple[Fact, ...]
    objectives: tuple[str, ...]
    longitudinal_decision: str
    lateral_decision: str
    longitudinal_action: str
    lateral_action: str
    entry_condition: str

    @property
    def decision_pair(self) -> str:
        return f"{self.longitudinal_decision}|{self.lateral_decision}"

    @property
    def action_pair(self) -> str:
        return f"{self.longitudinal_action}|{self.lateral_action}"

    @property
    def has_longitudinal_decision(self) -> bool:
        return self.longitudinal_decision not in {"not_stated", "ambiguous"}

    @property
    def has_lateral_decision(self) -> bool:
        return self.lateral_decision not in {"not_stated", "ambiguous"}

    @property
    def has_longitudinal_action(self) -> bool:
        return self.longitudinal_action not in {"not_stated", "ambiguous"}

    @property
    def has_lateral_action(self) -> bool:
        return self.lateral_action not in {"not_stated", "ambiguous"}


@dataclass(frozen=True)
class ReasoningProgram:
    text: str
    steps: tuple[ProgramStep, ...]
    ambiguity_status: str


def flatten_chain_of_thought(value: object) -> str:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    if not isinstance(value, str):
        return "" if value is None else str(value)
    stripped = value.strip()
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return stripped
    if isinstance(parsed, list):
        return " | ".join(str(item) for item in parsed)
    return str(parsed)


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


def _location(text: str) -> str:
    if _has(text, r"cross[- ]?traffic|cross(?:es|ing)? (?:our|the) (?:path|lane)"):
        return "cross_path"
    if _has(text, r"intersection|crosswalk"):
        return "intersection"
    if _has(text, r"curb|road edge|shoulder|cliff|fence|greenbelt|traffic island"):
        return "road_edge"
    if _has(text, r"target lane|after (?:the )?(?:merge|lane change)"):
        return "target_lane"
    if _has(text, r"ahead.*(?:our|same) lane|ahead in (?:our|the) lane"):
        return "ahead_same_lane"
    if "ahead" in text:
        return "route_ahead"
    has_left = "left" in text
    has_right = "right" in text
    if has_left != has_right:
        return "left" if has_left else "right"
    if "lane" in text:
        return "current_lane"
    return "unspecified"


def _modality(text: str) -> str:
    if _has(text, r"\bif\b|\bonce\b|\bwhen\b"):
        return "conditional"
    if _has(text, r"potential|possible|may |might "):
        return "possible"
    return "asserted"


_FACT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("lead_present", r"lead (?:vehicle|car|truck|motorcycle|bus)|vehicle ahead"),
    ("lead_slow", r"(?:slow|slower|slowing) (?:lead |vehicle)|lead .*slow"),
    ("lead_stopped", r"(?:stopped|stationary) (?:lead |vehicle)|lead .*stopp"),
    (
        "merging_or_cutin_agent",
        r"cut[- ]?in|merg(?:e|es|ing) (?:into|from)|entering our lane",
    ),
    ("crossing_agent", r"cross[- ]?traffic|cross(?:es|ing) (?:our|the) (?:path|lane)"),
    ("oncoming_agent", r"oncoming|opposing traffic|opposing vehicle"),
    ("adjacent_agent", r"adjacent|alongside|side-by-side"),
    ("vru_present", r"pedestrian|cyclist|bicycl|scooter|motorcycl"),
    (
        "static_obstacle",
        r"cone|barrel|barricade|barrier|debris|parked|stopped (?:car|truck|van|vehicle)|sign ?post",
    ),
    (
        "path_clear",
        r"(?:lane|path|intersection|side street|side) (?:is )?clear|once clear|no lead vehicle",
    ),
    ("gap_available", r"safe gap|clear gap|gap (?:forming|available|opens)|usable gap"),
    (
        "gap_blocked",
        r"gap (?:is )?block|blocking an immediate (?:merge|lane change)|wait for (?:a )?gap",
    ),
    (
        "lane_blocked_or_narrow",
        r"block(?:s|ed|ing) (?:our |the |right |left )?(?:lane|side|fork)|lane narrow",
    ),
    (
        "lane_position_or_boundary_constraint",
        r"curb|road edge|lane boundary|shoulder|cliff|fence|greenbelt|traffic island|clearance",
    ),
    (
        "signal_go",
        r"(?:traffic light|signal|turn arrow).*green|green (?:traffic light|signal|turn arrow)",
    ),
    (
        "signal_caution",
        r"(?:traffic light|signal).*yellow|yellow (?:traffic light|signal)",
    ),
    ("signal_stop", r"(?:traffic light|signal).*red|red (?:traffic light|signal)"),
    ("stop_control", r"stop sign|stop line|closed toll barrier"),
    ("yield_or_right_of_way_requirement", r"\byield\b|right[- ]of[- ]way|give way"),
    (
        "directional_control",
        r"turn[- ]only|turn lane|lane marking|double (?:white|yellow)|solid (?:white|yellow)",
    ),
    ("intersection_or_turn", r"intersection|\bturn(?:ing)?\b|crosswalk"),
    ("road_curve", r"\bcurve\b|\bbend(?:s)?\b|corner"),
    ("merge_split_or_ramp", r"\bmerge\b|\bsplit\b|ramp|exit lane|slip lane|zipper"),
    ("speed_constraint", r"speed (?:limit|advisory|bump)|\d+ mph"),
    (
        "route_maneuver_requirement",
        r"prepare for|upcoming .*turn|take the|route|exit lane|off-ramp",
    ),
    (
        "workzone_or_road_event",
        r"construction|work ?zone|flagger|cone|barrel|barricade|speed bump",
    ),
)


def _facts(text: str) -> tuple[Fact, ...]:
    location = _location(text)
    modality = _modality(text)
    facts = {
        Fact(predicate, location, modality)
        for predicate, pattern in _FACT_PATTERNS
        if _has(text, pattern)
    }
    if _has(text, r"no lead vehicle"):
        facts.discard(Fact("lead_present", location, modality))
        facts.add(Fact("lead_present", location, "negated"))
    if not facts and text.strip():
        facts.add(Fact("other_explicit"))
    return tuple(sorted(facts))


def _longitudinal_decision(text: str) -> str:
    if _has(text, r"\byield\b|give way"):
        return "yield_agent_right_of_way"
    if _has(
        text,
        r"^stop\b|\bto stop\b|\bprepare to stop\b|\bthen stop\b|"
        r"\bcome to (?:a )?stop\b",
    ):
        return "stop_for_static_constraints"
    if _has(text, r"create .*gap|search .*gap|wait for .*gap|usable gap"):
        return "gap_searching_for_lane_change_merge_zipper"
    if _has(text, r"\bpass\b|overtak"):
        return "acceleration_for_passing_overtaking"
    if _has(
        text,
        r"keep (?:a )?(?:safe )?distance|maintain .*distance|following gap|follow (?:the|a)",
    ):
        return "lead_obstacle_following"
    if _has(
        text,
        r"adapt speed|adjust speed|slow down|decelerat|speed bump|\bcurve\b|\bbend",
    ):
        return "speed_adaptation_road_events"
    if _has(text, r"accelerat|increase speed|keep speed|maintain speed|match speed"):
        return "set_speed_tracking"
    if _has(text, r"no (?:speed|longitudinal) change"):
        return "no_longitudinal_change"
    return "not_stated"


def _lateral_decision(text: str) -> str:
    if _has(
        text,
        r"lane change|change (?:to |into )?(?:the )?(?:left|right)? ?lane|"
        r"change lanes|merge (?:to (?:the )?)?(?:left|right)",
    ):
        return "lane_change"
    if _has(
        text,
        r"\bsplit\b|ramp|exit lane|slip lane|zipper|lane (?:is )?(?:ending|ends)",
    ):
        return "merge_split_facility_change"
    if _has(text, r"\bnudge\b"):
        return "in_lane_nudge"
    if _has(
        text,
        r"^turn (?:left|right)\b|\b(?:complete|execute|make|take) "
        r"(?:a )?(?:left|right) turn\b|\bthen turn (?:left|right)\b",
    ):
        return "turn"
    if _has(
        text, r"keep (?:the )?lane|maintain (?:the )?lane|go straight|proceed straight"
    ):
        return "lane_keeping_centering"
    return "not_stated"


def _longitudinal_action(text: str) -> str:
    if _has(
        text,
        r"^stop\b|\bprepare to stop\b|\bthen stop\b|\bstopping\b|"
        r"\bcome to (?:a )?stop\b",
    ):
        return "stop"
    if _has(text, r"accelerat|increase speed"):
        return "accelerate"
    if _has(text, r"decelerat|slow down"):
        return "decelerate"
    if _has(text, r"keep speed|maintain speed|match speed"):
        return "maintain_speed"
    if "reverse" in text:
        return "reverse"
    return "not_stated"


def _direction(text: str) -> str | None:
    matches = list(re.finditer(r"\b(left|right)\b", text))
    if not matches:
        return None
    values = {match.group(1) for match in matches}
    return values.pop() if len(values) == 1 else None


def _lateral_action(text: str, lateral_decision: str) -> str:
    direction = _direction(text)
    if "reverse" in text and direction:
        return f"reverse_{direction}"
    if direction and lateral_decision in {
        "lane_change",
        "merge_split_facility_change",
        "in_lane_nudge",
        "turn",
    }:
        return f"steer_{direction}"
    if lateral_decision == "lane_keeping_centering":
        return "go_straight"
    return "not_stated"


def _objectives(
    text: str, facts: tuple[Fact, ...], long_decision: str, lateral_decision: str
) -> tuple[str, ...]:
    predicates = {fact.predicate for fact in facts}
    objectives: set[str] = set()
    if long_decision == "lead_obstacle_following":
        objectives.add("maintain_following_gap")
    if long_decision == "stop_for_static_constraints" and predicates & {
        "signal_stop",
        "stop_control",
    }:
        objectives.add("obey_signal_or_stop_control")
    if long_decision == "yield_agent_right_of_way":
        objectives.add("yield_right_of_way")
    if long_decision == "speed_adaptation_road_events":
        objectives.add("adapt_speed_to_road")
    if long_decision == "gap_searching_for_lane_change_merge_zipper":
        objectives.add("acquire_merge_or_lane_change_gap")
    if long_decision == "acceleration_for_passing_overtaking":
        objectives.add("pass_slow_or_blocking_agent")
    if _has(text, r"avoid|clear(?:ance)?|keep clear|safe distance") or predicates & {
        "static_obstacle",
        "lane_position_or_boundary_constraint",
        "oncoming_agent",
        "crossing_agent",
        "vru_present",
    }:
        objectives.add("avoid_collision_or_increase_clearance")
    if predicates & {"route_maneuver_requirement", "directional_control"}:
        objectives.add("satisfy_route_or_lane_requirement")
    if lateral_decision == "lane_keeping_centering":
        objectives.add("remain_in_lane_or_drivable_area")
    if _has(text, r"proceed|lane is clear|intersection is clear"):
        objectives.add("nominal_progress")
    if not objectives:
        objectives.add("not_stated")
    return tuple(sorted(objectives))


def _entry_condition(text: str, index: int) -> str:
    if _has(text, r"once .*clear|when .*clear"):
        return "when_path_clear"
    if _has(text, r"when .*gap|once .*gap|after .*gap"):
        return "when_gap_available"
    if _has(text, r"when .*signal|once .*green|after .*green"):
        return "when_signal_permits"
    if _has(text, r"until .*stop|after .*yield"):
        return "until_stop_or_yield_satisfied"
    return "after_previous_complete" if index else "immediate"


def _segments(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return [""]
    return [
        segment.strip(" ,;.")
        for segment in re.split(r"(?:[,;]\s*)?\bthen\b", normalized)
        if segment.strip(" ,;.")
    ]


def compile_reasoning(value: object) -> ReasoningProgram:
    text = flatten_chain_of_thought(value)
    steps: list[ProgramStep] = []
    for index, raw_segment in enumerate(_segments(text)):
        segment = raw_segment.lower()
        facts = _facts(segment)
        long_decision = _longitudinal_decision(segment)
        lateral_decision = _lateral_decision(segment)
        steps.append(
            ProgramStep(
                facts=facts,
                objectives=_objectives(segment, facts, long_decision, lateral_decision),
                longitudinal_decision=long_decision,
                lateral_decision=lateral_decision,
                longitudinal_action=_longitudinal_action(segment),
                lateral_action=_lateral_action(segment, lateral_decision),
                entry_condition=_entry_condition(segment, index),
            )
        )
    missing_axes = sum(
        step.longitudinal_decision == "not_stated"
        or step.lateral_decision == "not_stated"
        for step in steps
    )
    ambiguity = "underspecified" if missing_axes else "resolved"
    return ReasoningProgram(text=text, steps=tuple(steps), ambiguity_status=ambiguity)


_FACT_OBJECTIVES = {
    "lead_present": {"maintain_following_gap"},
    "lead_slow": {"maintain_following_gap", "pass_slow_or_blocking_agent"},
    "lead_stopped": {"maintain_following_gap", "avoid_collision_or_increase_clearance"},
    "merging_or_cutin_agent": {"maintain_following_gap", "yield_right_of_way"},
    "crossing_agent": {"yield_right_of_way", "avoid_collision_or_increase_clearance"},
    "oncoming_agent": {"avoid_collision_or_increase_clearance"},
    "vru_present": {"yield_right_of_way", "avoid_collision_or_increase_clearance"},
    "static_obstacle": {"avoid_collision_or_increase_clearance"},
    "signal_go": {"nominal_progress"},
    "signal_stop": {"obey_signal_or_stop_control"},
    "stop_control": {"obey_signal_or_stop_control"},
    "yield_or_right_of_way_requirement": {"yield_right_of_way"},
    "road_curve": {"adapt_speed_to_road"},
    "speed_constraint": {"adapt_speed_to_road"},
    "gap_available": {"acquire_merge_or_lane_change_gap"},
    "gap_blocked": {"acquire_merge_or_lane_change_gap"},
    "directional_control": {"satisfy_route_or_lane_requirement"},
    "route_maneuver_requirement": {"satisfy_route_or_lane_requirement"},
    "lane_position_or_boundary_constraint": {
        "remain_in_lane_or_drivable_area",
        "avoid_collision_or_increase_clearance",
    },
}


def program_node_items(program: ReasoningProgram) -> set[str]:
    items: set[str] = set()
    for step in program.steps:
        for fact in step.facts:
            items.add(f"program:fact={fact.predicate}")
            items.add(
                f"program:fact_detail={fact.predicate}|{fact.location}|{fact.modality}"
            )
        items.update(
            f"program:objective={value}"
            for value in step.objectives
            if value != "not_stated"
        )
        if step.has_longitudinal_decision:
            items.add(f"program:long_decision={step.longitudinal_decision}")
        if step.has_lateral_decision:
            items.add(f"program:lat_decision={step.lateral_decision}")
        if step.has_longitudinal_action:
            items.add(f"program:long_action={step.longitudinal_action}")
        if step.has_lateral_action:
            items.add(f"program:lat_action={step.lateral_action}")
        items.add(f"program:entry={step.entry_condition}")
    return items


def _linked_fact_objectives(step: ProgramStep) -> Iterable[tuple[Fact, str]]:
    for fact in step.facts:
        allowed = _FACT_OBJECTIVES.get(fact.predicate, set())
        for objective in step.objectives:
            if objective in allowed:
                yield fact, objective


def program_edge_items(program: ReasoningProgram) -> set[str]:
    items = program_node_items(program)
    for step in program.steps:
        for fact, objective in _linked_fact_objectives(step):
            items.add(f"program:edge={fact.predicate}->{objective}")
        for objective in step.objectives:
            if objective == "not_stated":
                continue
            if step.has_longitudinal_decision:
                items.add(f"program:edge={objective}->{step.longitudinal_decision}")
            if step.has_lateral_decision:
                items.add(f"program:edge={objective}->{step.lateral_decision}")
        if step.has_longitudinal_decision and step.has_longitudinal_action:
            items.add(
                f"program:edge={step.longitudinal_decision}->{step.longitudinal_action}"
            )
        if step.has_lateral_decision and step.has_lateral_action:
            items.add(f"program:edge={step.lateral_decision}->{step.lateral_action}")
        if step.has_longitudinal_decision and step.has_lateral_decision:
            items.add(f"program:decision_pair={step.decision_pair}")
        if step.has_longitudinal_action and step.has_lateral_action:
            items.add(f"program:action_pair={step.action_pair}")
    return items


def program_path_items(program: ReasoningProgram) -> set[str]:
    items = program_edge_items(program)
    for step in program.steps:
        for fact, objective in _linked_fact_objectives(step):
            if step.has_longitudinal_decision and step.has_longitudinal_action:
                items.add(
                    "program:path_long="
                    f"{fact.predicate}>{objective}>{step.longitudinal_decision}>"
                    f"{step.longitudinal_action}"
                )
            if step.has_lateral_decision and step.has_lateral_action:
                items.add(
                    "program:path_lat="
                    f"{fact.predicate}>{objective}>{step.lateral_decision}>"
                    f"{step.lateral_action}"
                )
            if (
                step.has_longitudinal_decision
                and step.has_lateral_decision
                and step.has_longitudinal_action
                and step.has_lateral_action
            ):
                items.add(
                    "program:path_joint="
                    f"{fact.predicate}>{objective}>{step.decision_pair}>"
                    f"{step.action_pair}"
                )
    for previous, current in zip(program.steps, program.steps[1:]):
        if previous.has_longitudinal_decision and current.has_longitudinal_decision:
            items.add(
                "program:temporal_long="
                f"{previous.longitudinal_decision}/{previous.longitudinal_action}>"
                f"{current.entry_condition}>{current.longitudinal_decision}/"
                f"{current.longitudinal_action}"
            )
        if previous.has_lateral_decision and current.has_lateral_decision:
            items.add(
                "program:temporal_lat="
                f"{previous.lateral_decision}/{previous.lateral_action}>"
                f"{current.entry_condition}>{current.lateral_decision}/"
                f"{current.lateral_action}"
            )
    return items


def cross_trace_transition_items(
    previous: ReasoningProgram, current: ReasoningProgram
) -> set[str]:
    left = previous.steps[-1]
    right = current.steps[0]
    items: set[str] = set()
    if left.has_longitudinal_decision and right.has_longitudinal_decision:
        items.add(
            "program:trace_transition_long="
            f"{left.longitudinal_decision}/{left.longitudinal_action}>"
            f"{right.longitudinal_decision}/{right.longitudinal_action}"
        )
    if left.has_lateral_decision and right.has_lateral_decision:
        items.add(
            "program:trace_transition_lat="
            f"{left.lateral_decision}/{left.lateral_action}>"
            f"{right.lateral_decision}/{right.lateral_action}"
        )
    return items


def lexical_items(program: ReasoningProgram) -> set[str]:
    """Closed-vocabulary-free text baseline using normalized word bigrams."""
    words = re.findall(r"[a-z0-9]+", program.text.lower())
    return {f"text:bigram={left}_{right}" for left, right in zip(words, words[1:])}
