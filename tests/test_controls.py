"""CPU checks for the keyboard-to-Cosmos action contract."""

import math

import pytest

from cholescope.controls import IDENTITY_ROT6D, KeyboardController, TimedController


def test_rest_pose_has_two_identity_rotations_and_open_jaws():
    row = KeyboardController().action()
    assert len(row) == 28
    assert row[3:9] == list(IDENTITY_ROT6D)
    assert row[17:23] == list(IDENTITY_ROT6D)
    assert row[9] == row[23] == 1
    assert row[10:14] == row[24:28] == [0, 0, 0, 0]


def test_releasing_keys_holds_pose_and_jaw_until_explicit_reset():
    controller = KeyboardController()
    moved = controller.step({"KeyW", "KeyF"}, "right")
    assert moved[16] > 0 and moved[23] == -2
    assert moved[:14] == KeyboardController().action()[:14]
    assert controller.step(set(), "left") == moved
    assert controller.step(set(), "right") == moved
    controller.reset()
    assert controller.action() == KeyboardController().action()


def test_camera_translation_axes_and_opposing_keys():
    controller = KeyboardController()
    row = controller.step({"a", "q", "s"})
    assert all(x < 0 for x in row[:3])
    held = controller.step({"w", "s", "a", "d", "q", "e"})
    assert held == row


def test_diagonal_motion_is_not_faster_and_shift_is_fine():
    straight = KeyboardController().step({"d"})
    diagonal = KeyboardController().step({"d", "w"})
    fine = KeyboardController().step({"KeyD", "ShiftLeft"})
    assert math.sqrt(sum(x*x for x in diagonal[:3])) == pytest.approx(straight[0])
    assert fine[0] == pytest.approx(straight[0] * 0.25)


def test_stall_is_capped_and_repeated_motion_is_bounded():
    controller = KeyboardController(max_translation=0.1, max_rotation=0.2)
    first = controller.step({"d"}, dt=10.0)
    assert first[0] == pytest.approx(0.02)
    for _ in range(500):
        row = controller.step({"d", "arrowup", "f", "space"})
    assert row[0] == 0.1
    assert controller.snapshot()["left"]["rotation"][0] == 0.2
    assert row[10] == row[24] == 0  # No keyboard key enables energy.


def test_rotation_representation_stays_orthonormal_after_many_steps():
    controller = KeyboardController()
    for _ in range(80):
        row = controller.step({"ArrowUp", "ArrowRight", "KeyZ"})
    a, b = row[3:6], row[6:9]
    assert sum(x*x for x in a) == pytest.approx(1)
    assert sum(x*x for x in b) == pytest.approx(1)
    assert sum(x*y for x, y in zip(a, b)) == pytest.approx(0, abs=1e-10)


def test_normalization_matches_headset_jaw_and_energy_bypass():
    controller = KeyboardController(offset=[0.5]*28, scale=[2]*28)
    row = controller.step({"f"}, "right")
    assert row[0] == -0.25
    assert row[3] == 0.25
    assert row[9] == 1 and row[23] == -2
    assert row[10] == row[24] == 0


@pytest.mark.parametrize("kwargs", [
    {"offset": [0]*27, "scale": [1]*28},
    {"scale": [1]*28},
    {"offset": [0]*28, "scale": [0]*28},
    {"offset": [float("nan")]*28, "scale": [1]*28},
    {"translation_speed": float("inf")},
    {"fine_scale": 2},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        KeyboardController(**kwargs)


@pytest.mark.parametrize("dt", [-0.1, float("nan"), float("inf")])
def test_invalid_delta_time_rejected(dt):
    with pytest.raises(ValueError):
        KeyboardController().step([], dt=dt)


def test_invalid_arm_rejected_and_snapshots_do_not_mutate_state():
    controller = KeyboardController()
    with pytest.raises(ValueError):
        controller.step([], "both")
    snapshot = controller.snapshot()
    snapshot["left"]["position"][0] = 999
    assert controller.action()[0] == 0


def test_pointer_translation_combines_with_keyboard_and_holds_after_release():
    controller = KeyboardController()
    row = controller.step({"d", "f"}, "right", pointer_delta=[.03, -.01, .02])
    assert row[14:17] == pytest.approx([.05, -.01, .02])
    assert row[:14] == KeyboardController().action()[:14]
    assert row[23] == -2
    assert row[10] == row[24] == 0
    assert controller.step() == row


def test_pointer_full_distance_is_applied_once_independent_of_packet_size():
    controller = KeyboardController()
    row = controller.step(pointer_delta=[.06, .08, 0])
    assert row[:3] == pytest.approx([.06, .08, 0])
    split = KeyboardController()
    for _ in range(10):
        split.step(pointer_delta=[.006, .008, 0])
    assert split.action() == pytest.approx(row)
    assert controller.step() == row
    # Shift is already accounted for by the pointer client, never twice.
    fine = KeyboardController().step({"shift"}, pointer_delta=[.02, 0, 0])
    assert fine[0] == .02


def test_pointer_commit_changes_only_selected_translation_and_obeys_pose_bounds():
    controller = KeyboardController(max_translation=.05)
    before = controller.step({"f", "arrowup"}, "right")
    controller.apply_pointer_delta([.09, 0, 0], "right")
    after = controller.action()
    assert after[14] == .05
    assert after[:14] == before[:14]
    assert after[15:] == before[15:]
    controller.apply_pointer_delta([0, 0, 0], "right")
    assert controller.action() == after
    # Bounds discard excess beyond the workspace, without a delayed backlog.
    controller.apply_pointer_delta([-.02, 0, 0], "right")
    assert controller.action()[14] == pytest.approx(.03)
    controller.apply_pointer_delta([-1e308, 1e308, 0], "right")
    assert controller.action()[14:17] == [-.05, .05, 0]
    controller.reset()
    assert controller.action() == KeyboardController().action()


@pytest.mark.parametrize("delta", [[], [0, 0], [0, 0, 0, 0], [True, 0, 0],
                                  [".01", 0, 0], [None, 0, 0],
                                  [float("nan"), 0, 0], [0, float("inf"), 0]])
def test_invalid_pointer_delta_does_not_partially_apply_keyboard(delta):
    controller = KeyboardController()
    before = controller.action()
    with pytest.raises(ValueError):
        controller.step({"d", "f"}, pointer_delta=delta)
    assert controller.action() == before
    with pytest.raises(ValueError):
        controller.apply_pointer_delta(delta)
    assert controller.action() == before


def test_timed_short_tap_between_samples_is_preserved_once():
    timed = TimedController()
    timed.begin(10.)
    timed.update({"d"}, now=10.03)
    timed.update(set(), now=10.08)
    # No action was sampled while the key was down, as during GPU inference.
    row = timed.sample(10.3)
    assert row[0] == pytest.approx(.20 * .05)
    assert timed.sample(11.) == row


@pytest.mark.parametrize("sample_times", [[], [.05, .10, .15, .20, .25], [.04, .17, .29]])
def test_timed_hold_distance_is_independent_of_sampling_cadence(sample_times):
    timed = TimedController()
    timed.update({"d", "w", "shift", "arrowup"}, now=0.)
    for now in sample_times:
        timed.sample(now)
    timed.release(.3)
    pose = timed.controller.snapshot()["left"]
    distance = math.hypot(*pose["position"])
    assert distance == pytest.approx(.20 * .3 * .25)
    assert pose["rotation"][0] == pytest.approx(.45 * .3 * .25)
    assert timed.sample(1.) == timed.controller.action()


def test_timed_watchdog_caps_huge_stalls_and_fresh_input_has_no_debt():
    timed = TimedController()
    timed.update({"d"}, now=0.)
    assert timed.sample(1_000_000.)[0] == pytest.approx(.20 * .35)
    assert not timed.keys
    timed.update({"d"}, now=1_000_000.)
    timed.release(1_000_000.05)
    assert timed.sample(1_000_001.)[0] == pytest.approx(.20 * (.35 + .05))


def test_timed_heartbeats_extend_only_current_valid_hold():
    timed = TimedController()
    timed.update({"d"}, now=0.)
    timed.update({"d"}, now=.2)
    timed.update({"d"}, now=.4)
    timed.release(.6)
    assert timed.sample(.6)[0] == pytest.approx(.12)
    assert not timed.keys and timed.last_input is None


def test_timed_arm_and_direction_transitions_integrate_the_old_state():
    timed = TimedController()
    timed.update({"d"}, "left", now=0.)
    timed.update({"a"}, "right", now=.1)
    timed.update({"w"}, "left", now=.2)
    timed.release(.25)
    row = timed.sample(.4)
    assert row[:3] == pytest.approx([.02, 0, .01])
    assert row[14:17] == pytest.approx([-.02, 0, 0])


def test_timed_pointer_follows_old_key_integration_and_is_never_replayed():
    timed = TimedController(KeyboardController(max_translation=.05))
    timed.update({"d"}, now=0.)
    # Old keyboard motion reaches +.02 before the pointer reverses by -.04.
    timed.update({"shift"}, now=.1, pointer_delta=[-.04, 0, 0])
    assert timed.sample(.1)[0] == pytest.approx(-.02)
    timed.update(set(), now=.1, pointer_delta=[.04, 0, 0])
    assert timed.sample(.1)[0] == pytest.approx(.02)
    assert timed.sample(5.)[0] == pytest.approx(.02)


def test_timed_zero_duration_jaw_taps_latch_without_motion():
    timed = TimedController()
    timed.update({"f", "d"}, "right", now=0.)
    timed.update(set(), "right", now=0.)
    row = timed.sample(.5)
    assert row[23] == -2 and row[14:17] == [0, 0, 0]
    timed.update({"v"}, "right", now=.5)
    timed.release(.5)
    assert timed.sample(.5)[23] == 1


def test_timed_pause_resume_and_reset_do_not_replay_old_input():
    timed = TimedController()
    timed.update({"d", "f"}, now=0.)
    timed.release(.1)
    held = timed.controller.action()
    timed.begin(20.)
    assert timed.sample(21.) == held
    timed.update({"d"}, now=21.)
    timed.release(21.05)
    assert timed.sample(22.)[0] == pytest.approx(.03)
    timed.update({"d"}, now=22.)
    timed.reset(22.05)
    assert timed.sample(23.) == KeyboardController().action()
    assert not timed.keys and timed.last_input is None


@pytest.mark.parametrize("event", [
    {"keys": "d"}, {"keys": None}, {"keys": ["d", 1]},
    {"selected_arm": "both"}, {"selected_arm": []}, {"pointer_delta": [True, 0, 0]},
    {"pointer_delta": [float("nan"), 0, 0]}, {"now": float("inf")},
    {"now": float("nan")}, {"now": True}, {"now": -.1},
])
def test_timed_invalid_update_is_atomic_before_elapsed_motion(event):
    timed = TimedController()
    timed.update({"d"}, now=0.)
    before = timed.controller.action()
    with pytest.raises(ValueError):
        timed.update(**{"keys": set(), "now": .2, **event})
    assert timed.controller.action() == before
    assert timed.keys == frozenset({"d"}) and timed.arm == "left" and timed.last_input == 0.
    # Rejection did not consume the elapsed interval or refresh the heartbeat.
    assert timed.sample(.35)[0] == pytest.approx(.07)
    assert not timed.keys


@pytest.mark.parametrize("operation", ["sample", "release", "begin", "reset"])
def test_timed_backward_lifecycle_events_are_rejected_without_pose_changes(operation):
    timed = TimedController()
    timed.update({"d", "f"}, now=1.)
    timed.sample(1.1)
    before = timed.controller.action()
    with pytest.raises(ValueError, match="backwards"):
        getattr(timed, operation)(1.0)
    assert timed.controller.action() == before
    assert timed.keys == frozenset({"d", "f"}) and timed.last_input == 1.
