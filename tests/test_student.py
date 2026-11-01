"""Verify real learning and bounded inference in commanded coordinates only."""

import copy
import hashlib
import json

import numpy as np
import pytest

from cholescope.student import (
    GOAL_TOLERANCE, MAX_DELTA, PARAMETER_COUNT, POSITION_LIMIT, TARGET_LIMIT,
    TinyStudent, analytic_teacher, evaluate,
)


@pytest.fixture(scope="module")
def learned():
    student = TinyStudent()
    initial = student.checkpoint()
    metrics = student.train()
    return student, initial, metrics


def test_actual_gradient_training_improves_held_out_rollouts(learned):
    student, initial, metrics = learned
    assert metrics["updates_this_run"] == metrics["total_updates"] == 800
    assert metrics["history"][0]["updates"] == 1
    assert metrics["history"][-1]["updates"] == 800
    assert metrics["history"][-1]["batch_mse"] < metrics["history"][0]["batch_mse"] / 10
    assert metrics["after"]["goal_reached_fraction"] >= .96
    assert metrics["after"]["goal_reached_fraction"] > metrics["before"]["goal_reached_fraction"]
    assert metrics["after"]["mean_terminal_distance"] < metrics["before"]["mean_terminal_distance"] / 10
    assert metrics["neutral"]["goal_reached_fraction"] == 0
    assert metrics["analytic_teacher"]["goal_reached_fraction"] == 1
    assert initial["weights_sha256"] != student.checkpoint()["weights_sha256"]
    assert metrics["initial_weights_sha256"] == initial["weights_sha256"]
    assert metrics["trained_weights_sha256"] == student.checkpoint()["weights_sha256"]
    for name in ("w1", "w2", "w3"):
        assert not np.array_equal(student.weights[name], initial["weights"][name])


def test_generalizes_to_additional_goals_without_updating_weights(learned):
    student, _, _ = learned
    snapshot = student.checkpoint()
    measured = evaluate(student, seed=103931, goals=100)
    assert measured["goal_reached_fraction"] >= .95
    assert measured["mean_terminal_distance"] < GOAL_TOLERANCE / 10
    assert student.checkpoint() == snapshot


def test_initialization_training_and_checkpoint_are_reproducible(learned):
    student, initial, metrics = learned
    repeated = TinyStudent(seed=17)
    assert repeated.checkpoint() == initial
    assert repeated.train() == metrics
    assert repeated.checkpoint() == student.checkpoint()
    assert TinyStudent(seed=18).checkpoint()["weights_sha256"] != initial["weights_sha256"]


def test_prediction_comes_from_same_mlp_before_and_after_training(learned):
    student, initial, _ = learned
    for policy in (TinyStudent.from_checkpoint(initial), student):
        position, target = np.array([.02, -.01]), np.array([.10, .09])
        weights = policy.weights
        inputs = (target - position) / TARGET_LIMIT
        h1 = np.tanh(inputs @ weights["w1"] + weights["b1"])
        h2 = np.tanh(h1 @ weights["w2"] + weights["b2"])
        expected = MAX_DELTA * np.tanh(h2 @ weights["w3"] + weights["b3"])
        expected /= max(1.0, np.linalg.norm(expected) / MAX_DELTA)
        assert policy.predict(position, target) == pytest.approx(expected, abs=1e-14)


def test_bounded_predictions_at_edges_and_across_random_inputs(learned):
    rng = np.random.default_rng(4326)
    student, initial, _ = learned
    for policy in (student, TinyStudent.from_checkpoint(initial)):
        for _ in range(300):
            position = rng.uniform(-POSITION_LIMIT, POSITION_LIMIT, 2)
            if _ < 4:
                position = np.array([POSITION_LIMIT * (-1 if _ & 1 else 1),
                                     POSITION_LIMIT * (-1 if _ & 2 else 1)])
            target = rng.uniform(-TARGET_LIMIT, TARGET_LIMIT, 2)
            delta = np.array(policy.predict(position, target))
            assert np.all(np.isfinite(delta))
            assert np.linalg.norm(delta) <= MAX_DELTA + 1e-15
            assert np.all(np.abs(position + delta) <= POSITION_LIMIT + 1e-15)


def test_saturated_float32_command_is_accepted_and_clipped_for_prediction():
    student = TinyStudent()
    saturated = np.array([POSITION_LIMIT, -POSITION_LIMIT], dtype=np.float32)
    assert float(saturated[0]) > POSITION_LIMIT
    for policy in (student.predict, analytic_teacher):
        assert policy(saturated, [-.15, .15]) == policy([.20, -.20], [-.15, .15])
        for invalid in ([.200001, 0], [-.200001, 0], [.20000001, 0]):
            with pytest.raises(ValueError):
                policy(invalid, [0, 0])
        # Only observed positions get float32 boundary tolerance; goals stay strict.
        with pytest.raises(ValueError):
            policy([0, 0], np.array([TARGET_LIMIT, 0], dtype=np.float32))


def test_initial_student_rollout_survives_float32_command_saturation():
    student = TinyStudent()
    position = np.zeros(2, dtype=np.float32)
    saturated = False
    for _ in range(64):
        delta = student.predict(position, [-.15, .15])
        position = np.asarray(position.astype(np.float64) + delta, dtype=np.float32)
        saturated |= bool(np.any(np.abs(position.astype(np.float64)) > POSITION_LIMIT))
        assert np.all(np.isfinite(position))
        assert np.all(np.abs(position) <= np.float32(POSITION_LIMIT))
    assert saturated


def test_checkpoint_contains_exact_parameter_count_and_canonical_hashes(learned):
    student, _, _ = learned
    checkpoint = json.loads(json.dumps(student.checkpoint(), allow_nan=False))
    assert sum(np.asarray(w).size for w in checkpoint["weights"].values()) == PARAMETER_COUNT == 1218
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert checkpoint["weights_sha256"] == hashlib.sha256(canonical(checkpoint["weights"])).hexdigest()
    checksum = checkpoint.pop("checkpoint_sha256")
    assert checksum == hashlib.sha256(canonical(checkpoint)).hexdigest()


def test_checkpoint_roundtrip_clone_and_mutation_isolation(learned):
    student, _, _ = learned
    exported = student.checkpoint()
    restored = TinyStudent.from_checkpoint(json.loads(json.dumps(exported)))
    cloned = student.clone()
    assert restored.checkpoint() == cloned.checkpoint() == exported
    assert restored.predict([0, 0], [.1, .1]) == student.predict([0, 0], [.1, .1])
    restored.weights["w1"][0, 0] += 1
    exported["weights"]["w1"][0][0] += 1
    assert student.checkpoint() == cloned.checkpoint()


@pytest.mark.parametrize("modification", ["weights", "updates", "architecture", "missing", "nan"])
def test_modified_checkpoint_is_rejected(learned, modification):
    checkpoint = copy.deepcopy(learned[0].checkpoint())
    if modification == "weights":
        checkpoint["weights"]["w1"][0][0] += .01
    elif modification == "updates":
        checkpoint["updates"] += 1
    elif modification == "architecture":
        checkpoint["architecture"] = [2, 2]
    elif modification == "missing":
        checkpoint.pop("seed")
    else:
        checkpoint["weights"]["b3"][0] = float("nan")
    with pytest.raises(ValueError, match="checkpoint"):
        TinyStudent.from_checkpoint(checkpoint)


@pytest.mark.parametrize("bad", [[float("nan"), 0], [0, float("inf")], [True, False], [False, 0],
                                ["0", "0"], [0], [[0, 0]], [1, 0], None])
def test_nonfinite_malformed_or_unbounded_coordinates_are_rejected(bad):
    student = TinyStudent()
    for policy in (student.predict, analytic_teacher):
        with pytest.raises(ValueError):
            policy(bad, [0, 0])
        with pytest.raises(ValueError):
            policy([0, 0], bad)


@pytest.mark.parametrize("bad", [0, -1, 10001, True, 1.5, None])
def test_invalid_update_counts_are_rejected_without_mutation(bad):
    student = TinyStudent()
    original = student.checkpoint()
    with pytest.raises(ValueError):
        student.train(bad)
    assert student.checkpoint() == original


def test_continued_training_reports_real_total_updates():
    student = TinyStudent()
    student.train(steps=5)
    restored = TinyStudent.from_checkpoint(student.checkpoint())
    result = restored.train(steps=7)
    assert result["updates_this_run"] == 7
    assert result["total_updates"] == restored.updates == 12
    assert result["history"][0]["updates"] == 6
    assert result["history"][-1]["updates"] == 12


def test_evaluation_rejects_invalid_policy_motion():
    with pytest.raises(ValueError, match="vector motion bound"):
        evaluate(lambda *_: [MAX_DELTA, MAX_DELTA])
    with pytest.raises(ValueError):
        evaluate(lambda *_: [float("nan"), 0])
