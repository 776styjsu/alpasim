# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Alpamayo-table-native reasoning program and coverage abstractions.

The text program ends at the Table 1 decision. Explicit ``then`` clauses are
retained as ordered continuation stages. Table 5 actions are loaded from the
independently derived trajectory artifact and remain outside the reasoning
program; ``reasoning_action_items`` combines them only in a separate arm.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .alpamayo_tables import (
    TABLE1_LATERAL_DECISIONS,
    TABLE1_LONGITUDINAL_DECISIONS,
    TABLE2_COMPONENT_CATEGORIES,
    TABLE5_LATERAL_ACTIONS,
    TABLE5_LONGITUDINAL_ACTIONS,
)
from .program import Fact, compile_reasoning, flatten_chain_of_thought

CAUSAL_ROLES = (
    "proximate_cause",
    "enabling_condition",
    "blocking_condition",
    "constraint",
    "route_requirement",
    "regulatory_requirement",
)
TEMPORAL_GUARDS = (
    "immediate",
    "after_previous_complete",
    "when_gap_available",
    "when_path_clear",
    "when_signal_permits",
    "until_stop_or_yield_satisfied",
)

_FACT_CATEGORY = {
    "lead_present": "critical_objects",
    "lead_slow": "critical_objects",
    "lead_stopped": "critical_objects",
    "merging_or_cutin_agent": "critical_objects",
    "crossing_agent": "critical_objects",
    "oncoming_agent": "critical_objects",
    "adjacent_agent": "critical_objects",
    "vru_present": "critical_objects",
    "static_obstacle": "critical_objects",
    "signal_go": "traffic_lights",
    "signal_caution": "traffic_lights",
    "signal_stop": "traffic_lights",
    "stop_control": "stop_yield_control",
    "yield_or_right_of_way_requirement": "stop_yield_control",
    "intersection_or_turn": "road_events",
    "road_curve": "road_events",
    "merge_split_or_ramp": "road_events",
    "speed_constraint": "road_events",
    "path_clear": "lane_lanelines",
    "gap_available": "lane_lanelines",
    "gap_blocked": "lane_lanelines",
    "lane_blocked_or_narrow": "lane_lanelines",
    "lane_position_or_boundary_constraint": "lane_lanelines",
    "directional_control": "lane_lanelines",
    "route_maneuver_requirement": "routing_intent",
    "workzone_or_road_event": "odd_constraints",
}

_ROLE_BY_PREDICATE = {
    "path_clear": "enabling_condition",
    "gap_available": "enabling_condition",
    "gap_blocked": "blocking_condition",
    "adjacent_agent": "blocking_condition",
    "lane_blocked_or_narrow": "constraint",
    "lane_position_or_boundary_constraint": "constraint",
    "speed_constraint": "constraint",
    "workzone_or_road_event": "constraint",
    "route_maneuver_requirement": "route_requirement",
    "directional_control": "regulatory_requirement",
    "signal_go": "regulatory_requirement",
    "signal_caution": "regulatory_requirement",
    "signal_stop": "regulatory_requirement",
    "stop_control": "regulatory_requirement",
}


@dataclass(frozen=True, order=True)
class CriticalComponent:
    """One text-claimed, Table 2-aligned critical component."""

    category: str
    predicate: str
    location: str
    modality: str

    @property
    def key(self) -> str:
        return f"{self.category}/{self.predicate}"


@dataclass(frozen=True)
class TableDecision:
    """One closed Table 1 decision pair and optional lateral direction."""

    longitudinal: str
    lateral: str
    lateral_direction: str | None

    @property
    def pair(self) -> str:
        return f"{self.longitudinal}|{self.lateral}"

    @property
    def key(self) -> str:
        direction = self.lateral_direction or "none"
        return f"{self.pair}|direction={direction}"


@dataclass(frozen=True)
class CausalClause:
    """A typed support link from one claimed component to one decision axis."""

    premise: CriticalComponent | None
    role: str
    conclusion_axis: str
    conclusion: str


@dataclass(frozen=True)
class ReasoningStage:
    """One immediate or continuation reasoning stage."""

    index: int
    text: str
    temporal_guard: str
    components: tuple[CriticalComponent, ...]
    clauses: tuple[CausalClause, ...]
    decision: TableDecision


@dataclass(frozen=True)
class TableReasoningProgram:
    """Primary Alpamayo reasoning plus explicitly verbalized continuations."""

    text: str
    stages: tuple[ReasoningStage, ...]

    @property
    def primary(self) -> ReasoningStage:
        return self.stages[0]

    @property
    def continuations(self) -> tuple[ReasoningStage, ...]:
        return self.stages[1:]


@dataclass(frozen=True, order=True)
class AtomicActionRun:
    """A run-length-compressed Table 5 action pair."""

    longitudinal: str
    lateral: str
    segment_count: int

    @property
    def pair(self) -> str:
        return f"{self.longitudinal}|{self.lateral}"


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


def _segments(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return [""]
    return [
        segment.strip(" ,;.")
        for segment in re.split(r"(?:[,;]\s*)?\bthen\b", normalized)
        if segment.strip(" ,;.")
    ]


def _longitudinal_decision(text: str) -> str:
    if _has(text, r"\byield\b|give way|concede (?:priority|right[- ]of[- ]way)"):
        return "yield_agent_right_of_way"
    if _has(text, r"create .*gap|search .*gap|wait for .*gap|usable gap|open .*gap"):
        return "gap_searching_for_lane_change_merge_zipper"
    if _has(text, r"\bpass\b|overtak"):
        return "acceleration_for_passing_overtaking"
    if _has(
        text,
        r"keep (?:a )?(?:safe )?distance|maintain .*distance|following gap|"
        r"follow (?:the|a)|(?:slow|stop|decelerat).*lead|lead.*(?:slow|stop)",
    ):
        return "lead_obstacle_following"
    dynamic_agent = _has(
        text,
        r"pedestrian|cyclist|vru|cross[- ]?traffic|cut[- ]?in|oncoming|"
        r"emergency vehicle",
    )
    if dynamic_agent and _has(text, r"slow|decelerat|\bstop(?:ping)?\b"):
        return "yield_agent_right_of_way"
    static_control = _has(
        text,
        r"red (?:traffic light|signal)|stop (?:sign|line)|school (?:bus|zone)|"
        r"rail(?:road)? crossing|closed toll barrier",
    )
    if static_control and _has(text, r"slow|decelerat|\bstop(?:ping)?\b"):
        return "stop_for_static_constraints"
    road_constraint = _has(
        text,
        r"curve|bend|corner|grade|ramp|roundabout|turn|speed bump|"
        r"speed (?:limit|advisory)|traction|snow|ice|road condition",
    )
    if road_constraint and _has(text, r"adapt speed|adjust speed|slow|decelerat"):
        return "speed_adaptation_road_events"
    if _has(text, r"accelerat|increase speed|keep speed|maintain speed|match speed"):
        return "set_speed_tracking"
    if _has(text, r"proceed|continue (?:straight|forward)|resume"):
        return "set_speed_tracking"
    return "none"


def _lateral_decision(text: str) -> str:
    if _has(text, r"abort|cancel .*?(?:lane change|merge|nudge|pull[- ]?over)"):
        return "lateral_maneuver_abort"
    if _has(text, r"pull[- ]?over|curb approach|move to (?:the )?(?:shoulder|curb)"):
        return "pull_over_curb_approach"
    if _has(text, r"\bnudge\b"):
        if _has(text, r"out[- ]of[- ]lane|cross .*lane line|straddle"):
            return "out_of_lane_nudge"
        return "in_lane_nudge"
    if _has(
        text,
        r"lane change|change (?:to |into )?(?:the )?(?:left|right)? ?lane|"
        r"change lanes|merge (?:to (?:the )?)?(?:left|right) lane",
    ):
        return "lane_change"
    if _has(
        text,
        r"on[- ]?ramp|off[- ]?ramp|mainline|facility|\bsplit\b|zipper|"
        r"exit lane|slip lane|lane (?:is )?(?:ending|ends)",
    ):
        return "merge_split_facility_change"
    if _has(
        text,
        r"^turn (?:left|right)\b|\b(?:complete|execute|make|take|navigate) "
        r"(?:a )?(?:left|right) turn\b|\bu[- ]?turn\b",
    ):
        return "turn"
    if _has(
        text,
        r"keep (?:the )?lane|maintain (?:the )?lane|lane keep|"
        r"go straight|proceed straight|continue straight",
    ):
        return "lane_keeping_centering"
    return "none"


def _lateral_direction(text: str, lateral_decision: str) -> str | None:
    if lateral_decision not in {
        "out_of_lane_nudge",
        "in_lane_nudge",
        "lane_change",
        "pull_over_curb_approach",
        "turn",
        "lateral_maneuver_abort",
    }:
        return None
    action_text = re.split(r"\b(?:because|due to|since)\b", text, maxsplit=1)[0]
    values = set(re.findall(r"\b(left|right)\b", action_text))
    return values.pop() if len(values) == 1 else None


def _temporal_guard(text: str, index: int) -> str:
    if index == 0:
        return "immediate"
    if _has(text, r"once .*clear|when .*clear"):
        return "when_path_clear"
    if _has(text, r"when .*gap|once .*gap|after .*gap"):
        return "when_gap_available"
    if _has(text, r"when .*signal|once .*green|after .*green"):
        return "when_signal_permits"
    if _has(text, r"until .*stop|after .*yield"):
        return "until_stop_or_yield_satisfied"
    return "after_previous_complete"


def _components(facts: tuple[Fact, ...]) -> tuple[CriticalComponent, ...]:
    components = {
        CriticalComponent(
            category=_FACT_CATEGORY[fact.predicate],
            predicate=fact.predicate,
            location=fact.location,
            modality=fact.modality,
        )
        for fact in facts
        if fact.predicate in _FACT_CATEGORY
    }
    return tuple(sorted(components))


def _clauses(
    components: tuple[CriticalComponent, ...], decision: TableDecision
) -> tuple[CausalClause, ...]:
    conclusions = []
    if decision.longitudinal != "none":
        conclusions.append(("longitudinal", decision.longitudinal))
    if decision.lateral != "none":
        conclusions.append(("lateral", decision.lateral))
    if not conclusions:
        return ()
    if not components:
        return tuple(
            CausalClause(None, "unobserved", axis, conclusion)
            for axis, conclusion in conclusions
        )
    return tuple(
        CausalClause(
            component,
            _ROLE_BY_PREDICATE.get(component.predicate, "proximate_cause"),
            axis,
            conclusion,
        )
        for component in components
        for axis, conclusion in conclusions
    )


def compile_table_program(value: object) -> TableReasoningProgram:
    """Compile text into primary and continuation stages without reading A."""
    text = flatten_chain_of_thought(value)
    raw_segments = _segments(text)
    stages: list[ReasoningStage] = []
    for index, raw_segment in enumerate(raw_segments):
        segment = raw_segment.lower()
        fact_program = compile_reasoning(segment)
        facts = fact_program.steps[0].facts
        components = _components(facts)
        lateral = _lateral_decision(segment)
        decision = TableDecision(
            longitudinal=_longitudinal_decision(segment),
            lateral=lateral,
            lateral_direction=_lateral_direction(segment, lateral),
        )
        stages.append(
            ReasoningStage(
                index=index,
                text=raw_segment,
                temporal_guard=_temporal_guard(segment, index),
                components=components,
                clauses=_clauses(components, decision),
                decision=decision,
            )
        )
    return TableReasoningProgram(text=text, stages=tuple(stages))


def _stage_node_items(stage: ReasoningStage) -> set[str]:
    decision = stage.decision
    items = {
        f"table:z2_guard={stage.temporal_guard}",
        f"table:z3_longitudinal={decision.longitudinal}",
        f"table:z3_lateral={decision.lateral}",
    }
    if decision.pair != "none|none":
        items.add(f"table:z3_pair={decision.pair}")
    if decision.lateral_direction is not None:
        items.add(f"table:z3_lateral_direction={decision.lateral_direction}")
    for component in stage.components:
        items.update(
            {
                f"table:z1_category={component.category}",
                f"table:z1_component={component.key}",
                "table:z1_detail="
                f"{component.key}|{component.location}|{component.modality}",
            }
        )
    items.update(f"table:z2_role={clause.role}" for clause in stage.clauses)
    return items


def _stage_path_items(stage: ReasoningStage) -> set[str]:
    items = _stage_node_items(stage)
    for clause in stage.clauses:
        premise = clause.premise.key if clause.premise is not None else "unobserved"
        items.update(
            {
                f"table:edge=z1:{premise}>z2:{clause.role}",
                "table:edge="
                f"z2:{clause.role}>z3:{clause.conclusion_axis}:{clause.conclusion}",
                "table:path="
                f"{premise}>{clause.role}>{clause.conclusion_axis}:"
                f"{clause.conclusion}",
            }
        )
    return items


def primary_reasoning_items(program: TableReasoningProgram) -> set[str]:
    """Return cumulative Z1/Z2/Z3 coverage for the immediate decision only."""
    return _stage_path_items(program.primary)


def atomic_reasoning_path_items(program: TableReasoningProgram) -> set[str]:
    """Return only realized Z1-to-Z2-to-Z3 paths for the immediate decision."""
    return {
        item
        for item in _stage_path_items(program.primary)
        if item.startswith("table:path=")
    }


def sequence_reasoning_items(program: TableReasoningProgram) -> set[str]:
    """Return primary coverage plus explicit continuation stages/transitions."""
    items: set[str] = set()
    for stage in program.stages:
        items.update(_stage_path_items(stage))
    for previous, current in zip(program.stages, program.stages[1:]):
        items.add(f"table:continuation_guard={current.temporal_guard}")
        items.add(
            "table:transition="
            f"{previous.decision.key}>{current.temporal_guard}>{current.decision.key}"
        )
    return items


def z3_pair_items(
    program: TableReasoningProgram, *, include_continuations: bool = False
) -> set[str]:
    """Return pure non-empty Table 1 decision-pair coverage."""
    stages = program.stages if include_continuations else (program.primary,)
    return {
        f"table:z3_pair={stage.decision.pair}"
        for stage in stages
        if stage.decision.pair != "none|none"
    }


def z1_category_x_z3_pair_items(
    program: TableReasoningProgram, *, include_continuations: bool = False
) -> set[str]:
    """Return pure Table 2 category by non-empty Table 1 pair products."""
    stages = program.stages if include_continuations else (program.primary,)
    return {
        f"table:z1_category_x_z3_pair={component.category}|{stage.decision.pair}"
        for stage in stages
        for component in stage.components
        if stage.decision.pair != "none|none"
    }


def compress_action_pairs(
    pairs: tuple[tuple[str, str], ...],
) -> tuple[AtomicActionRun, ...]:
    """Run-length compress validated Table 5 per-segment action pairs."""
    runs: list[AtomicActionRun] = []
    for longitudinal, lateral in pairs:
        if longitudinal not in TABLE5_LONGITUDINAL_ACTIONS:
            raise ValueError(f"Unknown Table 5 longitudinal action: {longitudinal}")
        if lateral not in TABLE5_LATERAL_ACTIONS:
            raise ValueError(f"Unknown Table 5 lateral action: {lateral}")
        if runs and (runs[-1].longitudinal, runs[-1].lateral) == (
            longitudinal,
            lateral,
        ):
            previous = runs[-1]
            runs[-1] = AtomicActionRun(
                longitudinal, lateral, previous.segment_count + 1
            )
        else:
            runs.append(AtomicActionRun(longitudinal, lateral, 1))
    return tuple(runs)


def load_action_runs(metadata_path: Path | None) -> tuple[AtomicActionRun, ...]:
    """Load the full 100 ms Table 5 sequence derived from one trajectory."""
    if metadata_path is None:
        return ()
    info_path = metadata_path.with_name("additional_info.json")
    try:
        with info_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return ()
    segments = payload.get("meta_actions", {}).get("per_segment")
    if not isinstance(segments, list):
        return ()
    ordered = sorted(segments, key=lambda segment: int(segment["index"]))
    expected_indices = list(range(len(ordered)))
    actual_indices = [int(segment["index"]) for segment in ordered]
    if actual_indices != expected_indices:
        raise ValueError(f"Non-contiguous action segments in {info_path}")
    pairs = tuple(
        (str(segment["longitudinal"]), str(segment["lateral"])) for segment in ordered
    )
    return compress_action_pairs(pairs)


def action_items(runs: tuple[AtomicActionRun, ...]) -> set[str]:
    """Return Table 5 action nodes, pairs, and temporal transitions."""
    items: set[str] = set()
    for run in runs:
        items.update(
            {
                f"table:a_longitudinal={run.longitudinal}",
                f"table:a_lateral={run.lateral}",
                f"table:a_pair={run.pair}",
            }
        )
    for previous, current in zip(runs, runs[1:]):
        items.add(f"table:a_transition={previous.pair}>{current.pair}")
    return items


def reasoning_action_items(
    program: TableReasoningProgram, runs: tuple[AtomicActionRun, ...]
) -> set[str]:
    """Combine reasoning and A without treating A as a reasoning node."""
    items = sequence_reasoning_items(program) | action_items(runs)
    action_pairs = {(run.longitudinal, run.lateral) for run in runs}
    for stage in program.stages:
        decision = stage.decision
        lateral_decision = decision.lateral
        if decision.lateral_direction is not None:
            lateral_decision = f"{lateral_decision}:{decision.lateral_direction}"
        for longitudinal, lateral in action_pairs:
            items.update(
                {
                    f"table:z3a_longitudinal={decision.longitudinal}>{longitudinal}",
                    f"table:z3a_lateral={lateral_decision}>{lateral}",
                    f"table:z3a_pair={decision.pair}>{longitudinal}|{lateral}",
                }
            )
    return items


def validate_closed_vocabulary() -> None:
    """Fail fast if local mappings drift outside the published closed tables."""
    if set(_FACT_CATEGORY.values()) - set(TABLE2_COMPONENT_CATEGORIES):
        raise AssertionError("Fact mapping contains an unknown Table 2 category")
    if set(_ROLE_BY_PREDICATE.values()) - set(CAUSAL_ROLES):
        raise AssertionError("Predicate mapping contains an unknown causal role")
    for decision in (
        _longitudinal_decision(""),
        _longitudinal_decision("yield to a pedestrian"),
        _lateral_decision(""),
        _lateral_decision("change lanes left"),
    ):
        if decision not in {
            *TABLE1_LONGITUDINAL_DECISIONS,
            *TABLE1_LATERAL_DECISIONS,
        }:
            raise AssertionError(
                f"Decision parser emitted an unknown value: {decision}"
            )


validate_closed_vocabulary()
