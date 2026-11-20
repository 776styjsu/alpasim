# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json
from dataclasses import asdict

import numpy as np
import pytest
from alpasim_grpc.v0.egodriver_pb2 import DriveRequest
from alpasim_utils.geometry import Pose, Trajectory
from alpasim_utils.scenario import VehicleConfig
from crash_scenes.boxes import (
    boxes_gap,
    boxes_overlap,
    boxes_penetration,
    impact_zone,
)
from crash_scenes.confirm import confirm_rollout
from crash_scenes.label import label_tracks
from crash_scenes.maneuvers import (
    RECORDED,
    Maneuver,
    RecordedDrive,
    is_drivable,
    plan_arrays,
)
from crash_scenes.mine import SearchConfig, mine_scene
from crash_scenes.plans import (
    Plan,
    read_plan_file,
    select_crashes,
    write_plan_file,
)
from crash_scenes.scenes import RolloutWindow, Scene
from crash_scenes.scripted_driver import ScriptedDriver, plan_ahead
from crash_scenes.tracks import ActorTrack, RolloutTracks, make_track

STEP_US = 100_000
EGO_LENGTH, EGO_WIDTH = 5.0, 2.0


def _track(
    actor_id: str,
    start_xy: tuple[float, float],
    velocity_xy: tuple[float, float],
    yaw: float = 0.0,
    label: str = "automobile",
    size: tuple[float, float] = (4.5, 2.0),
    seconds: float = 10.0,
) -> ActorTrack:
    """A box moving at constant velocity, sampled every 0.1 s from t = 0."""
    times = np.arange(0, int(seconds * 1e6) + 1, STEP_US, dtype=np.int64)
    xy = np.array(start_xy) + np.outer(times / 1e6, velocity_xy)
    return make_track(
        actor_id, label, size[0], size[1], times, xy, np.full(len(times), yaw)
    )


def _ego(velocity_xy: tuple[float, float] = (10.0, 0.0)) -> ActorTrack:
    return _track("EGO", (0.0, 0.0), velocity_xy, size=(EGO_LENGTH, EGO_WIDTH))


def _first_crash(ego: ActorTrack, actor: ActorTrack):
    crashes, closest = label_tracks(ego, [actor], 0, 10_000_000)
    assert closest is None
    return crashes[0]


def test_boxes_overlap_and_gap() -> None:
    size = np.array([4.0, 2.0])
    origin = np.array([0.0, 0.0, 0.0])

    assert boxes_overlap(origin, size, np.array([3.9, 0.0, 0.0]), size)
    assert not boxes_overlap(origin, size, np.array([4.1, 0.0, 0.0]), size)
    assert boxes_gap(origin, size, np.array([5.0, 0.0, 0.0]), size) == pytest.approx(
        1.0
    )
    assert boxes_gap(origin, size, np.array([3.0, 0.0, 0.0]), size) == 0.0

    # Turned a quarter, the second box is 2 long and 4 wide along x and y.
    turned = np.array([3.5, 0.0, np.pi / 2])
    assert boxes_gap(origin, size, turned, size) == pytest.approx(0.5)


def test_boxes_penetration_is_the_shallow_direction() -> None:
    size = np.array([4.0, 2.0])
    origin = np.array([0.0, 0.0, 0.0])

    assert boxes_penetration(origin, size, np.array([3.0, 0.0, 0.0]), size) == (
        pytest.approx(1.0)
    )
    assert boxes_penetration(origin, size, np.array([0.5, 1.5, 0.0]), size) == (
        pytest.approx(0.5)
    )
    assert boxes_penetration(origin, size, np.array([5.0, 0.0, 0.0]), size) == 0.0


@pytest.mark.parametrize(
    ("point", "zone8", "zone4"),
    [
        ((2.5, 0.0), "front", "front"),
        ((2.5, 0.9), "front_left", "front"),
        ((2.5, -0.9), "front_right", "front"),
        ((0.0, 1.0), "left", "left"),
        ((-2.0, 1.0), "rear_left", "left"),
        ((-2.5, 0.0), "rear", "rear"),
        ((-2.5, -0.9), "rear_right", "rear"),
        ((0.3, -1.0), "right", "right"),
    ],
)
def test_impact_zone(point: tuple[float, float], zone8: str, zone4: str) -> None:
    zone = impact_zone(np.array(point), EGO_LENGTH, EGO_WIDTH)
    assert (zone["zone8"], zone["zone4"]) == (zone8, zone4)


def test_ego_rear_ends_a_stopped_car() -> None:
    stopped = _track("lead", (30.0, 0.0), (0.0, 0.0))
    crash = _first_crash(_ego(), stopped)

    assert crash.actor_group == "car"
    assert (crash.ego_zone8, crash.actor_zone8) == ("front", "rear")
    assert crash.crash_type == "rear_end_ego_striking"
    assert crash.striking == "ego"
    assert crash.closing_speed_mps == pytest.approx(10.0)
    # Bumpers meet after (30 - 2.5 - 2.25) m at 10 m/s.
    assert crash.time_us == pytest.approx(2_525_000, abs=1_000)


def test_contact_is_found_between_logged_steps() -> None:
    # Closing at 30 m/s the boxes are up to 3 m deep at the first logged overlap.
    oncoming = _track("oncoming", (60.0, 0.0), (-20.0, 0.0), yaw=np.pi)
    crash = _first_crash(_ego(), oncoming)

    assert crash.crash_type == "head_on"
    assert crash.striking == "both"
    assert crash.time_us == pytest.approx((60.0 - 2.5 - 2.25) / 30.0 * 1e6, abs=1_000)
    assert abs(crash.ego_contact_uv[0]) == pytest.approx(1.0, abs=0.02)


def test_ego_is_hit_from_behind() -> None:
    follower = _track("follower", (-20.0, 0.0), (15.0, 0.0))
    crash = _first_crash(_ego(), follower)

    assert (crash.ego_zone8, crash.actor_zone8) == ("rear", "front")
    assert crash.crash_type == "rear_end_ego_struck"
    assert crash.striking == "actor"


def test_ego_drifts_into_a_car_alongside() -> None:
    alongside = _track("neighbour", (0.0, 3.5), (10.0, 0.0))
    crash = _first_crash(_ego(velocity_xy=(10.0, 0.5)), alongside)

    assert (crash.ego_zone4, crash.actor_zone4) == ("left", "right")
    assert crash.crash_type == "sideswipe_same_direction"
    assert crash.striking == "ego"


def test_pedestrian_is_typed_by_the_ego_face() -> None:
    pedestrian = _track(
        "walker",
        (40.0, -4.0),
        (0.0, 1.0),
        yaw=np.pi / 2,
        label="person",
        size=(0.6, 0.6),
    )
    crash = _first_crash(_ego(), pedestrian)

    assert crash.actor_group == "pedestrian"
    assert crash.ego_zone4 == "front"
    assert crash.crash_type == "front_impact"


def test_graze_is_shallow_and_drive_through_is_deep() -> None:
    # 0.1 m of sideways overlap with a parked car, against driving straight
    # through one.
    grazed = _first_crash(_ego(), _track("parked", (30.0, 1.9), (0.0, 0.0)))
    driven_through = _first_crash(_ego(), _track("lead", (30.0, 0.0), (0.0, 0.0)))

    assert grazed.max_penetration_m == pytest.approx(0.1, abs=0.01)
    assert driven_through.max_penetration_m == pytest.approx(2.0, abs=0.01)


def test_closest_approach_without_a_crash() -> None:
    parked = _track("parked", (30.0, 3.0), (0.0, 0.0))
    crashes, closest = label_tracks(_ego(), [parked], 0, 10_000_000)

    assert crashes == []
    assert closest.actor_id == "parked"
    assert closest.gap_m == pytest.approx(1.0, abs=0.01)


def _straight_recording(speed_mps: float = 10.0, seconds: float = 20.0) -> Trajectory:
    times = np.arange(0, int(seconds * 1e6) + 1, STEP_US, dtype=np.uint64)
    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    poses = [
        Pose(np.array([speed_mps * t / 1e6, 0.0, 0.0], dtype=np.float32), identity)
        for t in times
    ]
    return Trajectory.from_poses(times, poses)


def test_recorded_maneuver_reproduces_the_drive() -> None:
    drive = RecordedDrive.from_trajectory(_straight_recording())
    positions, yaw = plan_arrays(drive, RECORDED)

    np.testing.assert_allclose(positions, drive.positions, atol=1e-6)
    np.testing.assert_allclose(yaw, drive.yaw, atol=1e-6)
    # The plan runs past the recording so the controller keeps its look-ahead.
    assert drive.timestamps_us[-1] > drive.recorded_end_us


def test_swerve_shifts_sideways_and_straightens_out() -> None:
    drive = RecordedDrive.from_trajectory(_straight_recording())
    swerve = Maneuver("swerve", start_us=5_000_000, lateral_m=3.0, ramp_s=3.0)
    positions, yaw = plan_arrays(drive, swerve)
    at = {t: i for i, t in enumerate(drive.timestamps_us.tolist())}

    assert positions[at[5_000_000], 1] == pytest.approx(0.0)
    assert positions[at[6_500_000], 1] == pytest.approx(1.5, abs=1e-6)
    assert positions[at[8_000_000], 1] == pytest.approx(3.0)
    assert yaw[at[6_500_000]] > 0.1  # heading turns towards the shift
    assert yaw[at[10_000_000]] == pytest.approx(0.0, abs=1e-6)

    half = plan_arrays(drive, swerve.scaled(0.5))[0]
    assert half[at[8_000_000], 1] == pytest.approx(1.5)


def test_brake_stops_the_ego() -> None:
    drive = RecordedDrive.from_trajectory(_straight_recording())
    brake = Maneuver("brake", start_us=5_000_000, accel_mps2=-5.0)
    positions, _ = plan_arrays(drive, brake)

    # From 10 m/s at 5 m/s^2 the ego stops 10 m further on, at x = 60 m.
    assert positions[-1, 0] == pytest.approx(60.0, abs=1e-3)
    assert positions[:, 0].max() == pytest.approx(60.0, abs=1e-3)


def test_violent_or_crawling_swerves_are_not_drivable() -> None:
    drive = RecordedDrive.from_trajectory(_straight_recording())
    crawl = RecordedDrive.from_trajectory(_straight_recording(speed_mps=1.0))

    assert is_drivable(drive, Maneuver("swerve", 5_000_000, lateral_m=3.0, ramp_s=3.0))
    assert not is_drivable(
        drive, Maneuver("swerve", 5_000_000, lateral_m=4.5, ramp_s=2.0)
    )
    assert not is_drivable(
        crawl, Maneuver("swerve", 5_000_000, lateral_m=1.0, ramp_s=3.0)
    )


def _scene(actors: list[ActorTrack]) -> Scene:
    recording = _straight_recording()
    config = SearchConfig()
    return Scene(
        scene_id="clipgt-test",
        usdz_path="unused.usdz",
        window=RolloutWindow.for_recording(
            recording,
            config.n_sim_steps,
            config.force_gt_duration_us,
            config.control_timestep_us,
        ),
        recording=recording,
        vehicle=VehicleConfig(
            aabb_x_m=EGO_LENGTH, aabb_y_m=EGO_WIDTH, aabb_x_offset_m=-1.0
        ),
        actors=actors,
    )


def test_mine_finds_a_swerve_into_a_parked_car_and_its_near_miss() -> None:
    parked = _track("parked", (120.0, 3.5), (0.0, 0.0), seconds=20.0)
    rows = mine_scene(_scene([parked]), SearchConfig())

    assert rows
    assert {row["maneuver"]["kind"] for row in rows} == {"swerve"}
    assert all(row["maneuver"]["lateral_m"] > 0 for row in rows)
    assert all(row["predicted"]["actor_id"] == "parked" for row in rows)
    assert all(row["predicted"]["striking"] == "ego" for row in rows)

    near_misses = [row["near_miss"] for row in rows if row["near_miss"] is not None]
    assert near_misses
    assert all(0.2 <= miss["target_gap_m"] <= 1.5 for miss in near_misses)
    assert all(0.0 < miss["scale"] < 1.0 for miss in near_misses)
    assert all(miss["end_us"] > miss["time_us"] for miss in near_misses)


def test_mine_finds_braking_in_front_of_a_follower() -> None:
    follower = _track("follower", (-15.0, 0.0), (10.0, 0.0), seconds=20.0)
    rows = mine_scene(_scene([follower]), SearchConfig())

    brakes = [row for row in rows if row["maneuver"]["kind"] == "brake"]
    assert brakes
    assert {row["predicted"]["crash_type"] for row in brakes} == {"rear_end_ego_struck"}
    assert {row["predicted"]["ego_zone8"] for row in brakes} == {"rear"}


def test_mine_ignores_actors_the_recording_already_overlaps() -> None:
    # A box sitting on the recorded path is label noise, not a crash to plan.
    on_path = _track("ghost", (100.0, 0.0), (0.0, 0.0), seconds=20.0)
    assert mine_scene(_scene([on_path]), SearchConfig()) == []


def _driven_exactly(scene: Scene, maneuver: Maneuver, variant: str, row: dict):
    """A rollout in which the ego follows the plan for ``maneuver`` to the letter."""
    drive = RecordedDrive.from_trajectory(scene.recording)
    positions, yaw = plan_arrays(drive, maneuver)
    plan = Plan(
        scene.scene_id,
        variant,
        None if variant == "control" else row["crash_id"],
        row["end_us"],
        drive.timestamps_us,
        positions,
        yaw,
    )
    rollout = RolloutTracks(
        scene_id=scene.scene_id,
        session_uuid="session",
        ego=scene.ego_track(drive.timestamps_us, positions, yaw),
        actors={actor.actor_id: actor for actor in scene.actors},
        closed_loop_start_us=scene.window.closed_loop_start_us,
    )
    return rollout, plan


def test_confirm_agrees_with_the_search_when_the_plan_is_driven_exactly() -> None:
    scene = _scene([_track("parked", (120.0, 3.5), (0.0, 0.0), seconds=20.0)])
    row = next(r for r in mine_scene(scene, SearchConfig()) if r["near_miss"])
    maneuver = Maneuver(**row["maneuver"])

    def outcome(maneuver: Maneuver, variant: str) -> dict:
        rollout, plan = _driven_exactly(scene, maneuver, variant, row)
        selection_row = None if variant == "control" else row
        return confirm_rollout(rollout, plan, selection_row, scene, set())

    crash = outcome(maneuver, "crash")
    assert crash["outcome"] == "confirmed"
    assert crash["same_ego_zone8"] and crash["same_crash_type"]
    assert crash["contact_time_error_s"] == pytest.approx(0.0, abs=0.002)
    assert crash["tracking_error_m"] == pytest.approx(0.0, abs=1e-6)

    near_miss = outcome(maneuver.scaled(row["near_miss"]["scale"]), "near_miss")
    assert near_miss["outcome"] == "near_miss"
    assert near_miss["target_gap_m"] == pytest.approx(
        row["near_miss"]["target_gap_m"], abs=0.01
    )

    assert outcome(RECORDED, "control")["outcome"] == "clean"


def _plan(scene_id: str = "clipgt-test", end_us: int = 3_000_000) -> Plan:
    times = np.arange(0, 10_000_001, STEP_US, dtype=np.int64)
    positions = np.stack(
        [10.0 * times / 1e6, np.zeros(len(times)), np.zeros(len(times))], axis=-1
    )
    return Plan(
        scene_id, "crash", "crash-1", end_us, times, positions, np.zeros(len(times))
    )


def test_plan_file_round_trip_and_one_plan_per_scene(tmp_path) -> None:
    path = tmp_path / "plan.json"
    write_plan_file([_plan("a"), _plan("b")], path)
    plans = read_plan_file(str(path))

    assert set(plans) == {"a", "b"}
    np.testing.assert_allclose(plans["a"].positions, _plan().positions)
    with pytest.raises(ValueError, match="more than one plan"):
        write_plan_file([_plan("a"), _plan("a")], path)


def test_plan_ahead_interpolates_the_plan_from_now() -> None:
    trajectory = plan_ahead(_plan(), time_now_us=1_250_000)

    assert trajectory.poses[0].timestamp_us == 1_250_000
    assert trajectory.poses[0].pose.vec.x == pytest.approx(12.5)
    assert trajectory.poses[-1].timestamp_us == 6_250_000
    assert trajectory.poses[0].pose.quat.w == pytest.approx(1.0)


def test_scripted_driver_ends_the_rollout_at_the_plan_end() -> None:
    driver = ScriptedDriver({"clipgt-test": _plan(end_us=3_000_000)})
    driver._sessions["session"] = driver._plans["clipgt-test"]

    def drive(time_us: int):
        request = DriveRequest(session_uuid="session", time_now_us=time_us)
        return driver.drive(request, context=None)

    assert not drive(2_900_000).terminate_session
    assert drive(3_000_000).terminate_session


def _manifest_row(
    scene: str, name: str, group: str, zone: str, near_miss: bool
) -> dict:
    return {
        "crash_id": f"{scene}|{name}",
        "scene_id": scene,
        "predicted": {
            "actor_group": group,
            "ego_zone4": zone,
            "crash_type": "rear_end_ego_striking",
        },
        "near_miss": {"scale": 0.5} if near_miss else None,
    }


def test_select_crashes_spreads_over_cells_and_scenes() -> None:
    rows = [
        _manifest_row("s1", f"car{i}", "car", "front", near_miss=False)
        for i in range(5)
    ]
    rows.append(_manifest_row("s2", "car", "car", "front", near_miss=True))
    rows.append(_manifest_row("s1", "truck", "truck", "rear", near_miss=False))

    selected = select_crashes(rows, per_cell=2, max_per_scene=2)
    ids = [row["crash_id"] for row in selected]

    # The rare truck cell is filled first; the car cell prefers the crash with
    # a near miss, then takes one more without exceeding two per scene.
    assert ids[0] == "s1|truck"
    assert "s2|car" in ids
    assert len(ids) == 3
    assert sum(row["scene_id"] == "s1" for row in selected) == 2
    assert json.dumps(selected)  # rows stay plain JSON
    assert asdict(RECORDED)["kind"] == "none"
