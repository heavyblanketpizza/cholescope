"""A tiny student learns geometric control, not surgical success.

The supervised labels are analytic corrections in *commanded XY coordinates*.
No images, anatomy labels, clinical outcomes, or Cosmos weights enter training.
The same MLP supplies actions before and after learning; only its weights change.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable, Sequence

import numpy as np

TARGET_LIMIT = 0.15
POSITION_LIMIT = 0.20
MAX_DELTA = 0.012
GOAL_TOLERANCE = 0.015
PARAMETER_COUNT = 1218
EVALUATION_SEED = 80351
TASK = "analytic-teacher imitation in commanded XY coordinates"
_SHAPES = {"w1": (2, 32), "b1": (32,), "w2": (32, 32),
           "b2": (32,), "w3": (32, 2), "b3": (2,)}


def _xy(value: Sequence[float], limit: float, name: str, *,
        allow_float32_boundary: bool = False) -> np.ndarray:
    """Reject invalid coordinates instead of silently training on bad input."""
    try:
        raw = np.asarray(value)
        if raw.shape != (2,) or raw.dtype.kind not in "iuf":
            raise ValueError
        if any(isinstance(item, (bool, np.bool_)) for item in value):
            raise ValueError
        result = raw.astype(np.float64)
        # Actual Cosmos commands are float32. Admit only the exact rounding of
        # the boundary (.20 -> .20000000298...), not a wider movement tolerance.
        accepted_limit = max(limit, float(np.float32(limit))) if allow_float32_boundary else limit
        if not np.all(np.isfinite(result)) or np.any(np.abs(result) > accepted_limit):
            raise ValueError
        if allow_float32_boundary:
            result = np.clip(result, -limit, limit)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be two finite numbers within +/-{limit}") from exc
    return result


def _bounded_delta(position: np.ndarray, delta: np.ndarray) -> list[float]:
    length = float(np.linalg.norm(delta))
    if length > MAX_DELTA:
        delta = delta * (MAX_DELTA / length)
    next_position = np.clip(position + delta, -POSITION_LIMIT, POSITION_LIMIT)
    result = next_position - position
    # Subtraction at a nonzero pose can round a clipped increment above its cap.
    result_length = float(np.linalg.norm(result))
    if result_length > MAX_DELTA:
        result *= (MAX_DELTA / result_length) * (1 - 1e-15)
    return result.tolist()


def analytic_teacher(position: Sequence[float], target: Sequence[float]) -> list[float]:
    """Proportional geometric controller used to label the synthetic examples."""
    current = _xy(position, POSITION_LIMIT, "position", allow_float32_boundary=True)
    goal = _xy(target, TARGET_LIMIT, "target")
    return _bounded_delta(current, 0.25 * (goal - current))


def neutral_policy(position: Sequence[float], target: Sequence[float]) -> list[float]:
    _xy(position, POSITION_LIMIT, "position", allow_float32_boundary=True)
    _xy(target, TARGET_LIMIT, "target")
    return [0.0, 0.0]


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: dict) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


class TinyStudent:
    """A deterministically initialized 1,218-parameter tanh MLP.

    ``predict`` returns a bounded XY *increment*, not an absolute position.
    The caller integrates it into a pose, leaving all other action channels fixed.
    ``train`` performs real gradient updates on CPU and returns measured metrics.
    """

    def __init__(self, seed: int = 17):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        self.seed = seed
        self.updates = 0
        rng = np.random.default_rng(seed)
        self.weights = {
            name: (rng.normal(0, np.sqrt(2 / sum(shape)), shape)
                   if name.startswith("w") else np.zeros(shape, dtype=np.float64))
            for name, shape in _SHAPES.items()
        }

    def _forward(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = self.weights
        h1 = np.tanh(inputs @ w["w1"] + w["b1"])
        h2 = np.tanh(h1 @ w["w2"] + w["b2"])
        output = np.tanh(h2 @ w["w3"] + w["b3"])
        return h1, h2, output

    def predict(self, position: Sequence[float], target: Sequence[float]) -> list[float]:
        current = _xy(position, POSITION_LIMIT, "position", allow_float32_boundary=True)
        goal = _xy(target, TARGET_LIMIT, "target")
        _, _, output = self._forward((goal - current) / TARGET_LIMIT)
        if not np.all(np.isfinite(output)):
            raise ValueError("student produced a non-finite prediction")
        return _bounded_delta(current, output * MAX_DELTA)

    def train(self, steps: int = 800) -> dict:
        """Imitate analytic labels using Adam, batch size 64, learning rate .003.

        Mirrored examples teach the same correction rule in both directions and
        keep the origin stable. Near-goal examples are included in every batch.
        Held-out rollout targets use a separate fixed seed and never train weights.
        """
        if type(steps) is not int or not 1 <= steps <= 10000:
            raise ValueError("steps must be an integer from 1 through 10000")
        before = evaluate(self)
        initial_hash = self.checkpoint()["weights_sha256"]
        start_updates = self.updates
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, start_updates, 491]))
        first_moment = {key: np.zeros_like(value) for key, value in self.weights.items()}
        second_moment = {key: np.zeros_like(value) for key, value in self.weights.items()}
        history = []
        for local_step in range(1, steps + 1):
            targets = rng.uniform(-TARGET_LIMIT, TARGET_LIMIT, (32, 2))
            positions = rng.uniform(-POSITION_LIMIT, POSITION_LIMIT, (32, 2))
            positions[:16] = np.clip(targets[:16] + rng.normal(0, .035, (16, 2)),
                                     -POSITION_LIMIT, POSITION_LIMIT)
            errors = targets - positions
            errors = np.concatenate((errors, -errors), axis=0)
            inputs = errors / TARGET_LIMIT
            labels = .25 * errors / MAX_DELTA
            norms = np.linalg.norm(labels, axis=1, keepdims=True)
            labels = labels / np.maximum(norms, 1.0)
            h1, h2, output = self._forward(inputs)
            residual = output - labels
            loss = float(np.mean(residual ** 2))
            g3 = (2.0 / residual.size) * residual * (1 - output ** 2)
            g2 = (g3 @ self.weights["w3"].T) * (1 - h2 ** 2)
            g1 = (g2 @ self.weights["w2"].T) * (1 - h1 ** 2)
            gradients = {"w3": h2.T @ g3, "b3": g3.sum(axis=0),
                         "w2": h1.T @ g2, "b2": g2.sum(axis=0),
                         "w1": inputs.T @ g1, "b1": g1.sum(axis=0)}
            for key, gradient in gradients.items():
                first_moment[key] = .9 * first_moment[key] + .1 * gradient
                second_moment[key] = .999 * second_moment[key] + .001 * gradient ** 2
                moment = first_moment[key] / (1 - .9 ** local_step)
                variance = second_moment[key] / (1 - .999 ** local_step)
                self.weights[key] -= .003 * moment / (np.sqrt(variance) + 1e-8)
            self.updates += 1
            if local_step == 1 or local_step % 50 == 0 or local_step == steps:
                history.append({"updates": self.updates, "batch_mse": loss})
        after = evaluate(self)
        return {"task": TASK, "parameter_count": PARAMETER_COUNT,
                "algorithm": "supervised imitation / Adam", "batch_size": 64,
                "examples_per_update": 64, "updates_this_run": steps,
                "total_updates": self.updates, "learning_rate": .003,
                "training_data": "synthetic geometric corrections; no medical images",
                "initial_weights_sha256": initial_hash,
                "trained_weights_sha256": self.checkpoint()["weights_sha256"],
                "before": before, "after": after,
                "neutral": evaluate(neutral_policy),
                "analytic_teacher": evaluate(analytic_teacher), "history": history}

    def checkpoint(self) -> dict:
        weights = {key: value.tolist() for key, value in self.weights.items()}
        payload = {"format": "cholescope-tiny-student-v1", "task": TASK,
                   "architecture": [2, 32, 32, 2], "activation": "tanh",
                   "parameter_count": PARAMETER_COUNT, "seed": self.seed,
                   "updates": self.updates,
                   "coordinates": "commanded XY, model units",
                   "target_limit": TARGET_LIMIT, "position_limit": POSITION_LIMIT,
                   "max_delta": MAX_DELTA, "weights": weights,
                   "weights_sha256": _digest(weights)}
        return {**payload, "checkpoint_sha256": _digest(payload)}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "TinyStudent":
        """Load a fully validated JSON checkpoint, including its content hashes."""
        try:
            if not isinstance(checkpoint, dict):
                raise ValueError
            payload = {key: value for key, value in checkpoint.items()
                       if key != "checkpoint_sha256"}
            if _digest(payload) != checkpoint["checkpoint_sha256"]:
                raise ValueError
            obj = cls(checkpoint["seed"])
            reference = obj.checkpoint()
            for key in ("format", "task", "architecture", "activation", "parameter_count",
                        "coordinates", "target_limit", "position_limit", "max_delta"):
                if checkpoint[key] != reference[key]:
                    raise ValueError
            updates = checkpoint["updates"]
            if type(updates) is not int or updates < 0:
                raise ValueError
            weights = checkpoint["weights"]
            if set(weights) != set(_SHAPES) or _digest(weights) != checkpoint["weights_sha256"]:
                raise ValueError
            for key, shape in _SHAPES.items():
                raw = np.asarray(weights[key])
                if raw.shape != shape or raw.dtype.kind not in "iuf":
                    raise ValueError
                value = raw.astype(np.float64)
                if not np.all(np.isfinite(value)):
                    raise ValueError
                obj.weights[key] = value.copy()
            obj.updates = updates
            return obj
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid or modified student checkpoint") from exc

    def clone(self) -> "TinyStudent":
        return self.from_checkpoint(self.checkpoint())


def evaluate(policy: TinyStudent | Callable, *, seed: int = EVALUATION_SEED,
             goals: int = 25, rollout_steps: int = 64) -> dict:
    """Measure unseen goal rollouts in commanded coordinates, never video space."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("evaluation seed must be an integer in [0, 2**32)")
    if type(goals) is not int or not 1 <= goals <= 1000:
        raise ValueError("goals must be an integer from 1 through 1000")
    if type(rollout_steps) is not int or not 1 <= rollout_steps <= 1000:
        raise ValueError("rollout_steps must be an integer from 1 through 1000")
    rng = np.random.default_rng(seed)
    targets = rng.uniform(-TARGET_LIMIT, TARGET_LIMIT, (goals, 2))
    predict = policy.predict if hasattr(policy, "predict") else policy
    distances = []
    reached_steps = []
    for target in targets:
        position = np.zeros(2, dtype=np.float64)
        first_reached = 0 if np.linalg.norm(target) <= GOAL_TOLERANCE else None
        for step in range(1, rollout_steps + 1):
            raw_delta = _xy(predict(position, target), MAX_DELTA, "policy delta")
            if np.linalg.norm(raw_delta) > MAX_DELTA + 1e-12:
                raise ValueError("policy delta exceeds the vector motion bound")
            position += raw_delta
            if np.any(np.abs(position) > POSITION_LIMIT + 1e-12):
                raise ValueError("policy exceeded the position bound")
            position = np.clip(position, -POSITION_LIMIT, POSITION_LIMIT)
            if first_reached is None and np.linalg.norm(target - position) <= GOAL_TOLERANCE:
                first_reached = step
        distances.append(float(np.linalg.norm(target - position)))
        reached_steps.append(first_reached)
    terminal_successes = sum(distance <= GOAL_TOLERANCE for distance in distances)
    return {"coordinate_space": "commanded XY, not measured tool-tip pixels",
            "seed": seed, "goals": goals, "rollout_steps": rollout_steps,
            "goal_tolerance": GOAL_TOLERANCE,
            "mean_terminal_distance": float(np.mean(distances)),
            "median_terminal_distance": float(np.median(distances)),
            "max_terminal_distance": float(np.max(distances)),
            "goal_reached_fraction": terminal_successes / goals,
            "terminal_distances": distances, "first_reached_steps": reached_steps,
            "targets": targets.tolist()}
