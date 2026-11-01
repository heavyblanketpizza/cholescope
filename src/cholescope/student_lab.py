"""Small-controller learning experiments; Cosmos supplies visualization only.

The observation and score use commanded XY coordinates, never inferred tissue
state or generated pixels. The analytic teacher is a reaching controller, not
a surgical demonstration. Each trial freezes one actual neural checkpoint.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import math

import numpy as np
from PIL import Image

from .recording import Recording
from .student import TinyStudent, GOAL_TOLERANCE, POSITION_LIMIT, TARGET_LIMIT


def validate_target(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Choose a two-coordinate target")
    if any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > TARGET_LIMIT for x in value):
        raise ValueError("Target coordinates must be finite and within ±0.15")
    if math.hypot(*value) < .04:
        raise ValueError("Choose a target at least 0.04 from the starting point")
    return [float(x) for x in value]


def checkpoint_hash(checkpoint):
    return checkpoint["checkpoint_sha256"]


class StudentLab:
    total_steps = 64

    def __init__(self, studio):
        self.studio = studio
        self.before = TinyStudent(seed=17)
        self.after = None
        self.metrics = None
        self.training = False
        self.active = False
        self.state = "idle"
        self.phase = None
        self.target = [.12, .08]
        self.position = [0., 0.]
        self.step = 0
        self.results = {}

    def checkpoint(self, phase):
        if phase not in {"before", "after"}:
            raise ValueError("Unknown student checkpoint")
        model = self.before if phase == "before" else self.after
        if model is None:
            raise ValueError("Train the student before using its learned checkpoint")
        return model.checkpoint()

    def status(self):
        return {"type": "student_state", "model_name": "Tiny reach student", "parameter_count": 1218,
                "scope": "command_coordinates", "trained": self.after is not None,
                "training": self.training, "state": self.state, "phase": self.phase,
                "target": self.target, "position": self.position, "step": self.step,
                "total_steps": self.total_steps, "metrics": self.metrics,
                "train_history": (self.metrics or {}).get("history", []),
                "before_sha256": checkpoint_hash(self.checkpoint("before")),
                "after_sha256": checkpoint_hash(self.checkpoint("after")) if self.after else None,
                "results": self.results}

    async def send(self):
        await self.studio.send(self.status())

    async def train(self):
        if self.after is not None:
            await self.send()
            return
        await self.studio.pause()
        self.training, self.state = True, "training"
        await self.send()
        try:
            # Keep the original random checkpoint immutable for honest A/B.
            candidate = self.before.clone()
            metrics = await asyncio.to_thread(candidate.train, steps=800)
            self.after, self.metrics = candidate, metrics
            directory = self.studio.output_dir / "student-checkpoints"
            directory.mkdir(parents=True, exist_ok=True)
            for phase in ("before", "after"):
                (directory / f"{phase}.json").write_text(json.dumps(self.checkpoint(phase), indent=2, allow_nan=False) + "\n")
            (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        finally:
            self.training, self.state = False, "idle"
            await self.send()

    async def reset_learning(self):
        await self.studio.pause()
        self.after, self.metrics = None, None
        self.phase, self.position, self.step = None, [0., 0.], 0
        self.results = {}
        self.state = "idle"
        await self.send()

    async def start(self, payload):
        phase = payload.get("phase")
        model = TinyStudent.from_checkpoint(self.checkpoint(phase))
        target = validate_target(payload.get("target", [.12, .08]))
        studio = self.studio
        # Reset through the same code as manual A/B, but do not start its loop.
        await studio.reset(payload.get("scene_id", studio.scene_id), payload.get("seed", studio.seed), run=False)
        self.phase, self.target, self.position, self.step = phase, target, [0., 0.], 0
        self.active, self.state = True, "running"
        studio.running = True
        await self.send()
        studio.generation_task = asyncio.create_task(self.run(model))
        await studio.set_state("running", f"Student {phase} training · commanded-coordinate reaching")

    async def run(self, model):
        studio = self.studio
        epoch = studio.epoch
        checkpoint = model.checkpoint()
        digest = checkpoint_hash(checkpoint)
        recorder, recording = None, None
        complete = False
        produced = 0
        positions = []
        initial_distance = math.hypot(*self.target)
        try:
            provenance = await studio.model_call(studio.backend.provenance) if hasattr(studio.backend, "provenance") else {}
            provenance.update({"initial_image_sha256": hashlib.sha256(studio.scenes[studio.scene_id]["path"].read_bytes()).hexdigest(),
                               "controller": "tiny-student-command-xy-v1", "controls": "tiny-student-command-xy-v1",
                               "student_checkpoint_sha256": digest,
                               "student_phase": self.phase, "student_target": self.target,
                               "student_observation": "commanded_xy_and_target; no_video_feedback",
                               "student_teacher": "analytic_geometric_reaching; no_surgical_examples",
                               "student_score": "command_coordinate_distance; not_surgical_success"})
            recorder = Recording(studio.output_dir, branch="a" if self.phase == "before" else "b",
                                 scene_id=studio.scene_id, seed=studio.seed, provenance=provenance)
            position = [0., 0.]
            while produced < self.total_steps and studio.running and epoch == studio.epoch and not studio.closed:
                actions, chunk_positions = [], []
                for _ in range(min(studio.backend.actions_per_chunk, self.total_steps - produced)):
                    delta = model.predict(position, self.target)
                    next_position = [max(-POSITION_LIMIT, min(POSITION_LIMIT, x + d)) for x, d in zip(position, delta)]
                    studio.controller.apply_pointer_delta([next_position[0] - position[0], next_position[1] - position[1], 0.], "left")
                    # The actual exported commands, not a display-only cursor,
                    # are the observations used for the following policy step.
                    action = np.asarray(studio.controller.action(), dtype=np.float32).tolist()
                    position = action[:2]
                    actions.append(action)
                    chunk_positions.append(position.copy())
                result = await studio.model_call(studio.backend.generate, actions)
                if not studio.running or epoch != studio.epoch:
                    break
                if len(result.frames) != len(actions):
                    raise RuntimeError("Student action/frame cadence mismatch")
                await asyncio.to_thread(recorder.append, result.frames, actions)
                positions.extend(chunk_positions)
                produced += len(actions)
                await studio.send({"type": "metrics", "fps": result.metrics.get("generation_fps"), **result.metrics})
                for index, frame in enumerate(result.frames):
                    if not studio.running or epoch != studio.epoch:
                        break
                    self.step = produced - len(actions) + index + 1
                    self.position = chunk_positions[index]
                    buffer = io.BytesIO()
                    Image.fromarray(frame).save(buffer, format="JPEG", quality=85)
                    await studio.send({"type": "frame", "jpeg": base64.b64encode(buffer.getvalue()).decode(),
                                       "frame_index": self.step - 1, "generated": True})
                    await self.send()
                    await asyncio.sleep(.1)
            complete = produced == self.total_steps and studio.running and epoch == studio.epoch
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Student trial failed")
            await studio.send({"type": "error", "message": "Student trial failed. Inspect the server log; no success is inferred."})
        finally:
            self.active = False
            studio.running = False
            studio.release()
            self.state = "finished" if complete else "cancelled"
            if positions:
                self.position, self.step = positions[-1], produced
            if recorder:
                try:
                    recording = await asyncio.to_thread(recorder.finish)
                except (ValueError, RuntimeError, OSError) as error:
                    if recorder.frame_count:
                        await studio.send({"type": "error", "message": str(error)})
            distance = math.dist(self.position, self.target)
            score = {"initial_distance": initial_distance, "final_distance": distance,
                     "reached": complete and distance <= GOAL_TOLERANCE, "complete": complete,
                     "steps": produced, "goal_tolerance": GOAL_TOLERANCE,
                     "scope": "command_coordinates"}
            result = {"type": "student_result", "phase": self.phase, "recording": recording,
                      "target": self.target, "seed": studio.seed, "scene_id": studio.scene_id,
                      "score": score, "checkpoint_sha256": digest}
            self.results[self.phase] = result
            if recorder and recording:
                (recorder.directory / "student-result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            await studio.send(result)
            await self.send()
            if studio.ready:
                await studio.set_state("paused", "Student trial finished" if complete else "Student trial stopped")
