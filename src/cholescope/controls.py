"""Keyboard and pointer poses in the Cosmos3-H-Dreams 28D action format.

The pose is relative to the scene's initial reference and persists when keys
are released. Values follow NVIDIA's headset adapter: translation in model
units (10 times controller metres), two rotation-matrix columns, jaw 1/-2.
These inputs are an experimental adaptation, not physical robot commands.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

ACTION_DIM = 28
ACTION_FPS = 10.0
ACTIONS_PER_CHUNK = 8
IDENTITY_ROT6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
ARM_OFFSETS = {"left": 0, "right": 14}


def validate_pointer_delta(delta: Sequence[float]) -> tuple[float, float, float]:
    """Accept only three finite numbers; booleans and numeric strings are invalid."""
    if not isinstance(delta, (list, tuple)) or len(delta) != 3:
        raise ValueError("Pointer delta must contain three finite numbers")
    if any(type(value) not in (int, float) for value in delta):
        raise ValueError("Pointer delta must contain three finite numbers")
    try:
        values = tuple(float(value) for value in delta)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Pointer delta must contain three finite numbers") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Pointer delta must contain three finite numbers")
    return values


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _rot6d(angles: Sequence[float]) -> list[float]:
    """First two columns of Rz(roll) Ry(yaw) Rx(pitch), column-major."""
    pitch, yaw, roll = angles
    sx, sy, sz = math.sin(pitch), math.sin(yaw), math.sin(roll)
    cx, cy, cz = math.cos(pitch), math.cos(yaw), math.cos(roll)
    return [cz * cy, sz * cy, -sy,
            cz * sy * sx - sz * cx, sz * sy * sx + cz * cx, cy * sx]


def _key_name(key: str) -> str:
    key = key.lower()
    if key.startswith("key") and len(key) == 4:
        return key[-1]
    if key in {"shiftleft", "shiftright"}:
        return "shift"
    return key


class KeyboardController:
    """Integrate bounded keyboard and pointer motion into persistent arm poses.

    Call ``step`` at 10 Hz with the *current* key set, and collect eight rows
    for one default model chunk. Pass an empty set after blur, disconnection,
    or the server's input watchdog expires. Empty input holds the last pose
    and jaw; ``reset`` is the only operation that returns to the scene origin.

    The caller owns key-state timing. This class never caches pressed keys.
    A long tick is capped at 100 ms, preventing a stalled client from causing
    a large pose jump. Optional affine normalization matches upstream, with
    jaw and energy bypassing normalization as in its headset adapter.

    ``pointer_delta`` is a one-shot translation in model units (+X right,
    +Y down, +Z forward). Its full distance updates the requested pose, subject
    only to the workspace bounds. It is never cached or multiplied by ``dt``;
    pointer fine scaling belongs to the client, independently of keyboard Shift.
    """

    def __init__(
        self,
        *,
        translation_speed: float = 0.20,
        rotation_speed: float = 0.45,
        max_translation: float = 1.0,
        max_rotation: float = math.pi / 3,
        fine_scale: float = 0.25,
        normalizer: str | Path | None = None,
        offset: Sequence[float] | None = None,
        scale: Sequence[float] | None = None,
    ) -> None:
        self.translation_speed = _finite_positive(translation_speed, "translation_speed")
        self.rotation_speed = _finite_positive(rotation_speed, "rotation_speed")
        self.max_translation = _finite_positive(max_translation, "max_translation")
        self.max_rotation = _finite_positive(max_rotation, "max_rotation")
        self.fine_scale = _finite_positive(fine_scale, "fine_scale")
        if self.fine_scale > 1:
            raise ValueError("fine_scale cannot exceed 1")
        if normalizer is not None:
            if offset is not None or scale is not None:
                raise ValueError("use either a normalizer path or offset/scale")
            import numpy as np

            with np.load(normalizer, allow_pickle=False) as data:
                if "offset" not in data or "scale" not in data:
                    raise ValueError("normalizer requires offset and scale")
                offset, scale = data["offset"].tolist(), data["scale"].tolist()
        if (offset is None) != (scale is None):
            raise ValueError("offset and scale must be provided together")
        self._offset = tuple(float(x) for x in offset) if offset is not None else (0.0,) * ACTION_DIM
        self._scale = tuple(float(x) for x in scale) if scale is not None else (1.0,) * ACTION_DIM
        if len(self._offset) != ACTION_DIM or len(self._scale) != ACTION_DIM:
            raise ValueError("normalizer offset and scale must both be 28D")
        if not all(math.isfinite(x) for x in self._offset):
            raise ValueError("normalizer offset must be finite")
        if not all(math.isfinite(x) and x > 0 for x in self._scale):
            raise ValueError("normalizer scale must be finite and positive")
        self.reset()

    def reset(self) -> None:
        self._position = {arm: [0.0, 0.0, 0.0] for arm in ARM_OFFSETS}
        self._angles = {arm: [0.0, 0.0, 0.0] for arm in ARM_OFFSETS}
        self._jaw = {arm: 1.0 for arm in ARM_OFFSETS}

    def step(
        self, keys: Iterable[str] = (), selected_arm: str = "left", dt: float = 0.1,
        *, pointer_delta: Sequence[float] | None = None,
    ) -> list[float]:
        if selected_arm not in ARM_OFFSETS:
            raise ValueError("selected_arm must be 'left' or 'right'")
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative")
        dt = min(float(dt), 1.0 / ACTION_FPS)
        if isinstance(keys, str):
            raise ValueError("keys must be a collection, not a string")
        pointer = validate_pointer_delta(pointer_delta if pointer_delta is not None else (0.0, 0.0, 0.0))
        pressed = {_key_name(key) for key in keys}
        fine = self.fine_scale if "shift" in pressed else 1.0
        direction = [int("d" in pressed) - int("a" in pressed),
                     int("e" in pressed) - int("q" in pressed),
                     int("w" in pressed) - int("s" in pressed)]
        turn = [int("arrowup" in pressed) - int("arrowdown" in pressed),
                int("arrowright" in pressed) - int("arrowleft" in pressed),
                int("x" in pressed) - int("z" in pressed)]
        for state, delta, speed, bound in (
            (self._position[selected_arm], direction, self.translation_speed, self.max_translation),
            (self._angles[selected_arm], turn, self.rotation_speed, self.max_rotation),
        ):
            magnitude = max(1.0, math.sqrt(sum(x * x for x in delta)))
            for axis in range(3):
                state[axis] = max(-bound, min(bound, state[axis] + delta[axis] / magnitude * speed * dt * fine))
        self.apply_pointer_delta(pointer, selected_arm)
        if ("f" in pressed) != ("v" in pressed):
            self._jaw[selected_arm] = -2.0 if "f" in pressed else 1.0
        return self.action()

    def apply_pointer_delta(self, pointer_delta: Sequence[float], selected_arm: str = "left") -> None:
        """Update the requested pose without advancing keyboard motion.

        Each accepted pointer message contributes its full translation once,
        even while the model is generating. Only the resulting workspace
        position is clamped; action sampling later exports the held pose.
        """
        if selected_arm not in ARM_OFFSETS:
            raise ValueError("selected_arm must be 'left' or 'right'")
        pointer = validate_pointer_delta(pointer_delta)
        position = self._position[selected_arm]
        for axis in range(3):
            position[axis] = max(-self.max_translation, min(self.max_translation, position[axis] + pointer[axis]))

    def action(self) -> list[float]:
        raw = [0.0] * ACTION_DIM
        for arm, start in ARM_OFFSETS.items():
            raw[start:start + 3] = self._position[arm]
            raw[start + 3:start + 9] = _rot6d(self._angles[arm])
            raw[start + 9] = self._jaw[arm]
        normalized = [(x - offset) / scale for x, offset, scale in zip(raw, self._offset, self._scale)]
        for index in (9, 10, 23, 24):
            normalized[index] = raw[index]
        return normalized

    def snapshot(self) -> dict[str, dict[str, object]]:
        return {
            arm: {"position": self._position[arm].copy(),
                  "rotation": self._angles[arm].copy(), "jaw": self._jaw[arm], "energy": 0.0}
            for arm in ARM_OFFSETS
        }


class TimedController:
    """Integrate live input by elapsed time independently of model sampling.

    The wrapped ``KeyboardController`` remains usable by fixed-step callers
    such as the student lab. While this wrapper owns live input, route key,
    pointer and sampling events through it so elapsed motion is applied once.

    ``update`` advances the previous key state before changing keys or arms.
    This preserves a key tap even when no model sample occurs during the tap.
    The new jaw state and one-shot pointer displacement apply immediately.
    Missing heartbeats permit at most ``input_timeout`` seconds of motion;
    later samples hold the pose rather than replaying stale input.

    Call ``release`` when pausing to account for valid motion up to the pause.
    Call ``begin`` when resuming to discard old timing without moving the
    instruments, or ``reset`` to also restore the initial pose. Timestamps
    must be finite and nondecreasing, including across lifecycle operations.
    """

    def __init__(self, controller: KeyboardController | None = None, *, input_timeout: float = .35) -> None:
        self.controller = controller if controller is not None else KeyboardController()
        self.input_timeout = _finite_positive(input_timeout, "input_timeout")
        self.keys: frozenset[str] = frozenset()
        self.arm = "left"
        self.last_input: float | None = None
        self._advanced_at: float | None = None

    def _time(self, now: float) -> float:
        if isinstance(now, bool) or not isinstance(now, (int, float)):
            raise ValueError("now must be a finite monotonic timestamp")
        try:
            now = float(now)
        except (OverflowError, ValueError) as error:
            raise ValueError("now must be a finite monotonic timestamp") from error
        if not math.isfinite(now):
            raise ValueError("now must be a finite monotonic timestamp")
        if self._advanced_at is not None and now < self._advanced_at:
            raise ValueError("now cannot move backwards")
        return now

    def _advance(self, now: float) -> None:
        if self._advanced_at is not None and self.last_input is not None:
            expires_at = self.last_input + self.input_timeout
            remaining = max(0., min(now, expires_at) - self._advanced_at)
            # KeyboardController caps each call at one action tick. Split
            # valid elapsed time so a delayed model call cannot erase a tap.
            while self.keys and remaining > 0:
                dt = min(remaining, 1. / ACTION_FPS)
                self.controller.step(self.keys, self.arm, dt=dt)
                remaining -= dt
            if now >= expires_at:
                self.keys = frozenset()
        self._advanced_at = now

    def begin(self, now: float) -> None:
        """Begin or resume live input without pose changes or old motion debt."""
        now = self._time(now)
        self.keys = frozenset()
        self.last_input = None
        self._advanced_at = now

    def reset(self, now: float) -> None:
        """Restore both arm poses and clear any pending timing or key state."""
        now = self._time(now)
        self.controller.reset()
        self.begin(now)

    def update(
        self, keys: Iterable[str] = (), selected_arm: str = "left", *, now: float,
        pointer_delta: Sequence[float] | None = None,
    ) -> list[float]:
        # Validate the complete event before advancing previous motion, so a
        # rejected command changes neither the pose nor the watchdog timing.
        now = self._time(now)
        if not isinstance(selected_arm, str) or selected_arm not in ARM_OFFSETS:
            raise ValueError("selected_arm must be 'left' or 'right'")
        if isinstance(keys, str):
            raise ValueError("keys must be a collection, not a string")
        try:
            values = tuple(keys)
        except TypeError as error:
            raise ValueError("keys must be a collection of strings") from error
        if any(not isinstance(key, str) for key in values):
            raise ValueError("keys must be a collection of strings")
        pressed = frozenset(_key_name(key) for key in values)
        pointer = validate_pointer_delta(pointer_delta if pointer_delta is not None else (0., 0., 0.))
        self._advance(now)
        self.keys, self.arm, self.last_input = pressed, selected_arm, now
        # dt=0 preserves short jaw taps without advancing translation twice.
        return self.controller.step(pressed, selected_arm, dt=0., pointer_delta=pointer)

    def sample(self, now: float) -> list[float]:
        self._advance(self._time(now))
        return self.controller.action()

    def release(self, now: float) -> None:
        """Account for valid elapsed input, then hold both poses and jaws."""
        now = self._time(now)
        self._advance(now)
        self.keys = frozenset()
        self.last_input = None
