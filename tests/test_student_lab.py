import asyncio
import math

import numpy as np
import pytest
from PIL import Image

from cholescope.model_backend import GenerationResult
from cholescope.server import create_app
from cholescope.student import TinyStudent
from cholescope.student_lab import validate_target


class Backend:
    actions_per_chunk = 8

    def __init__(self):
        self.resets = []

    def load(self):
        pass

    def reset(self, path, seed=0):
        self.resets.append((path, seed))

    def generate(self, actions):
        return GenerationResult(np.zeros((len(actions), 16, 16, 3), dtype=np.uint8), {"generation_fps": 100})

    def close(self):
        pass


async def event(ws, kind):
    async with asyncio.timeout(10):
        while True:
            message = await ws.receive_json()
            if message["type"] == "error":
                raise AssertionError(message["message"])
            if message["type"] == kind:
                return message


def scenes(tmp_path):
    path = tmp_path / "scene.png"
    Image.new("RGB", (16, 16)).save(path)
    return {"test": {"id": "test", "name": "Test", "path": path, "description": "Test"}}


async def test_training_requires_no_gpu_and_exports_real_checkpoints(aiohttp_client, tmp_path):
    app = create_app(output_dir=tmp_path / "runs")
    client = await aiohttp_client(app)
    before = await (await client.get("/api/student/checkpoint/before")).json()
    assert (await client.get("/api/student/checkpoint/after")).status == 404
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "student_train"})
    while not (status := await event(ws, "student_state"))["trained"]:
        pass
    after = await (await client.get("/api/student/checkpoint/after")).json()
    assert before["weights_sha256"] != after["weights_sha256"]
    assert before["updates"] == 0 and after["updates"] == 800
    assert status["after_sha256"] == after["checkpoint_sha256"]
    assert status["before_sha256"] == before["checkpoint_sha256"]
    assert status["scope"] == "command_coordinates"
    assert status["metrics"]["after"]["goal_reached_fraction"] > status["metrics"]["before"]["goal_reached_fraction"]
    await ws.send_json({"type": "student_train"})
    assert (await event(ws, "student_state"))["metrics"]["total_updates"] == 800
    assert not app["studio"].ready
    await ws.send_json({"type": "student_reset"})
    status = await event(ws, "student_state")
    assert not status["trained"] and status["metrics"] is None and status["results"] == {}
    assert (await client.get("/api/student/checkpoint/after")).status == 404
    assert (await (await client.get("/api/student/checkpoint/before")).json()) == before
    await ws.close()


async def test_trials_use_checkpoint_actions_same_initial_condition_and_fixed_channels(aiohttp_client, tmp_path):
    backend = Backend()
    configured = scenes(tmp_path)
    app = create_app(backend=backend, scenes=configured, output_dir=tmp_path / "runs")
    app["studio"].student_lab.total_steps = 8
    client = await aiohttp_client(app)
    await app["studio"].startup_task
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "student_train"})
    while not (await event(ws, "student_state"))["trained"]:
        pass
    recorded = []
    for phase in ("before", "after"):
        await ws.send_json({"type": "student_trial", "phase": phase, "target": [.12, .08], "scene_id": "test", "seed": 42})
        result = await event(ws, "student_result")
        assert result["score"]["complete"] and result["score"]["steps"] == 8
        manifest = await (await client.get(result["recording"]["actions_url"])).json()
        checkpoint = await (await client.get(f"/api/student/checkpoint/{phase}")).json()
        assert result["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
        assert manifest["provenance"]["student_checkpoint_sha256"] == result["checkpoint_sha256"]
        assert manifest["provenance"]["controller"] == "tiny-student-command-xy-v1"
        actions = np.asarray(manifest["actions"])
        assert len(actions) == manifest["frame_count"] == 8
        assert np.all(actions[:, [2, 10, 24]] == 0)
        assert np.all(actions[:, 14:] == actions[0, 14:])
        assert np.all(actions[:, 2:] == actions[0, 2:])
        assert np.all(np.linalg.norm(np.diff(np.vstack(([0, 0], actions[:, :2])), axis=0), axis=1) <= .01200001)
        assert result["score"]["final_distance"] == math.dist(actions[-1, :2], [.12, .08])
        model = TinyStudent.from_checkpoint(checkpoint)
        first_expected = np.asarray(model.predict([0, 0], [.12, .08]), dtype=np.float32)
        np.testing.assert_array_equal(actions[0, :2], first_expected)
        recorded.append(actions)
    assert not np.array_equal(*recorded)
    assert backend.resets == [(configured["test"]["path"], 42)] * 2
    await ws.close()


async def test_pause_cancels_trial_and_manual_input_cannot_change_student(aiohttp_client, tmp_path):
    app = create_app(backend=Backend(), scenes=scenes(tmp_path), output_dir=tmp_path / "runs")
    client = await aiohttp_client(app)
    studio = app["studio"]
    await studio.startup_task
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "student_trial", "phase": "before", "target": [.12, .08], "scene_id": "test", "seed": 42})
    await event(ws, "frame")
    snapshot = studio.controller.snapshot()
    await studio.command({"type": "input", "keys": ["f", "w"], "selected_arm": "right",
                          "pointer_commit": True, "pointer_delta": [.06, 0, 0]})
    assert snapshot == studio.controller.snapshot()
    await ws.send_json({"type": "pause"})
    result = await event(ws, "student_result")
    assert not result["score"]["complete"]
    assert not result["score"]["reached"]
    assert result["score"]["steps"] < 64
    assert not studio.student_lab.active and not studio.running
    await ws.close()


@pytest.mark.parametrize("target", [[True, 0], [float("nan"), 0], [.151, .04], [0, 0], [.1], "bad"])
def test_reject_invalid_goal(target):
    with pytest.raises(ValueError):
        validate_target(target)
