import asyncio
import base64
import io
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from aiohttp import WSServerHandshakeError
from PIL import Image

from cholescope.model_backend import GenerationResult
from cholescope.recording import Recording
from cholescope.server import Studio, create_app, validate_input
from cholescope import model_backend, server


class Backend:
    actions_per_chunk = 1
    fps = 10

    def __init__(self):
        self.calls = []
        self.resets = []

    def track(self):
        self.calls.append(threading.get_ident())

    def load(self):
        self.track()

    def reset(self, path, seed=0):
        self.track()
        self.resets.append((path, seed))

    def generate(self, actions):
        self.track()
        return GenerationResult(np.zeros((len(actions), 16, 16, 3), dtype=np.uint8), {"generation_fps": 100})

    def close(self):
        self.track()


async def event(ws, kind):
    async with asyncio.timeout(5):
        while True:
            message = await ws.receive_json()
            if message["type"] == kind:
                return message


@pytest.fixture
def scenes(tmp_path):
    path = tmp_path / "scene.png"
    Image.new("RGB", (16, 16)).save(path)
    return {"test": {"id": "test", "name": "Test", "path": path, "description": "Test scene"}}


async def ready(client):
    for _ in range(100):
        response = await client.get("/api/status")
        if (await response.json())["backend_ready"]:
            return
        await asyncio.sleep(.01)
    raise AssertionError("Backend never became ready")


@pytest.mark.parametrize(("flags", "decoder", "chunk_frames"), [
    ([], "wan", None),
    (["--decoder", "taehv"], "taehv", None),
    (["--chunk-frames", "4"], "wan", 4),
    (["--decoder", "taehv", "--chunk-frames", "8"], "taehv", 8),
])
def test_cli_passes_generation_options_to_model_backend(monkeypatch, tmp_path, flags, decoder, chunk_frames):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    created = []

    def backend(*args, **kwargs):
        created.append(kwargs)
        return Backend()

    monkeypatch.setattr(model_backend, "CosmosBackend", backend)
    monkeypatch.setattr(server.web, "run_app", lambda app, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["cholescope", "--checkpoint", str(checkpoint),
                                      "--output-dir", str(tmp_path / "out"), *flags])
    server.main()
    assert [kwargs["decoder"] for kwargs in created] == [decoder]
    assert [kwargs["chunk_frames"] for kwargs in created] == [chunk_frames]


def test_cli_rejects_unsupported_chunk_size(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cholescope", "--preview", "--chunk-frames", "2"])
    with pytest.raises(SystemExit) as error:
        server.main()
    assert error.value.code == 2


async def test_preview_never_claims_generation(aiohttp_client, tmp_path, scenes):
    client = await aiohttp_client(create_app(scenes=scenes, output_dir=tmp_path / "outputs"))
    ws = await client.ws_connect("/ws")
    assert (await event(ws, "status"))["state"] == "preview"
    await ws.send_json({"type": "start", "scene_id": "test"})
    assert "GPU model" in (await event(ws, "error"))["message"]
    await ws.send_json({"type": "pause"})
    await ws.close()
    assert (await (await client.get("/api/status")).json())["state"] == "preview"


async def test_retired_visual_view_and_commands_are_unavailable(aiohttp_client, tmp_path, monkeypatch):
    client = await aiohttp_client(create_app(output_dir=tmp_path / "outputs"))
    for path in ("/", "/student"):
        response = await client.get(path)
        assert response.status == 200
        assert 'href="/visual"' not in await response.text()
    for path in ("/visual", "/visual/", "/visual?setup=1"):
        response = await client.get(path, allow_redirects=False)
        assert response.status == 302
        assert response.headers["Location"] == "/"
    for path in ("/api/visual/metrics", "/api/visual/checkpoint/before", "/api/visual/checkpoint/after"):
        assert (await client.get(path)).status == 404

    # Old files can remain on a persistent pod after an in-place source upload.
    for name in ("visual.html", "visual.mjs", "visual.css"):
        (tmp_path / name).write_text("retired content")
    (tmp_path / "app.mjs").write_text("current content")
    monkeypatch.setattr(server, "STATIC", tmp_path)
    for name in ("visual.html", "visual.mjs", "visual.css"):
        assert (await client.get(f"/static/{name}")).status == 404
    assert await (await client.get("/static/app.mjs")).text() == "current content"

    ws = await client.ws_connect("/ws")
    await event(ws, "status")
    for kind in ("visual_status", "visual_train", "visual_reset", "visual_trial"):
        await ws.send_json({"type": kind})
        assert (await event(ws, "error"))["message"] == "Unknown command"
    await ws.close()


async def test_pages_share_one_stylesheet_and_retired_layers_are_not_served(aiohttp_client, tmp_path, monkeypatch):
    # Earlier override stylesheets can also remain on a pod after an in-place upload.
    for name in ("style.css", "clean.css", "student.css"):
        (tmp_path / name).write_text(name)
    monkeypatch.setattr(server, "STATIC", tmp_path)
    client = await aiohttp_client(create_app(output_dir=tmp_path / "outputs"))
    assert await (await client.get("/static/style.css")).text() == "style.css"
    for name in ("clean.css", "student.css"):
        assert (await client.get(f"/static/{name}")).status == 404


async def test_authenticated_routes_and_single_same_origin_operator(aiohttp_client, tmp_path, scenes):
    client = await aiohttp_client(create_app(scenes=scenes, output_dir=tmp_path, access_token="test-secret"))
    assert (await client.get("/api/scenes")).status == 401
    headers = {"Authorization": "Basic " + base64.b64encode(b"demo:test-secret").decode()}
    assert (await client.get("/api/scenes", headers=headers)).status == 200
    with pytest.raises(WSServerHandshakeError) as bad_origin:
        await client.ws_connect("/ws", headers={**headers, "Origin": "https://elsewhere.example"})
    assert bad_origin.value.status == 403
    ws = await client.ws_connect("/ws", headers=headers)
    with pytest.raises(WSServerHandshakeError) as duplicate:
        await client.ws_connect("/ws", headers=headers)
    assert duplicate.value.status == 409
    await ws.close()


async def test_model_calls_have_single_owner_and_disconnect_releases(aiohttp_client, tmp_path, scenes):
    backend = Backend()
    app = create_app(backend=backend, scenes=scenes, output_dir=tmp_path)
    client = await aiohttp_client(app)
    await ready(client)
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "pause"})
    await event(ws, "status")
    await ws.send_json({"type": "resume"})
    assert "Start a scene" in (await event(ws, "error"))["message"]
    await ws.send_json({"type": "start", "scene_id": "test", "seed": 11})
    await ws.send_json({"type": "input", "keys": ["w"], "selected_arm": "right"})
    assert (await event(ws, "frame"))["generated"] is True
    await app["studio"].command(pointer_input([.04, 0, 0]))
    accepted_pose = app["studio"].controller.action()
    assert accepted_pose[0] == pytest.approx(.04)
    await ws.close()
    await asyncio.sleep(.15)
    assert not app["studio"].running
    assert not app["studio"].keys
    assert app["studio"].controller.action() == accepted_pose
    await client.close()
    assert len(set(backend.calls)) == 1
    assert backend.calls[0] != threading.get_ident()


async def test_comparison_branches_reset_same_seed_and_export_actual_actions(aiohttp_client, tmp_path, scenes):
    backend = Backend()
    client = await aiohttp_client(create_app(backend=backend, scenes=scenes, output_dir=tmp_path / "runs"))
    await ready(client)
    ws = await client.ws_connect("/ws")
    for branch in ("a", "b"):
        await ws.send_json({"type": "record_start", "branch": branch})
        assert (await event(ws, "recording_started"))["branch"] == branch
        await event(ws, "frame")
        await ws.send_json({"type": "record_stop", "branch": branch})
        clip = await event(ws, "recording")
        assert clip["branch"] == branch
        response = await client.get(clip["actions_url"])
        manifest = await response.json()
        assert manifest["generated"] is True
        assert manifest["frame_count"] == len(manifest["actions"]) > 0
        assert all(len(row) == 28 for row in manifest["actions"])
        assert (await client.get(clip["url"])).status == 200
    assert backend.resets == [(scenes["test"]["path"], 42)] * 2
    assert (await client.get("/api/runs/bad/video.mp4")).status == 404
    await ws.close()


async def test_reconnect_waits_for_previous_operator_cleanup(aiohttp_client, tmp_path, scenes):
    class SlowBackend(Backend):
        entered = threading.Event()
        proceed = threading.Event()

        def reset(self, path, seed=0):
            self.entered.set()
            self.proceed.wait(timeout=5)
            super().reset(path, seed)

    backend = SlowBackend()
    app = create_app(backend=backend, scenes=scenes, output_dir=tmp_path)
    client = await aiohttp_client(app)
    await ready(client)
    ws = await client.ws_connect("/ws")
    try:
        await ws.send_json({"type": "start", "scene_id": "test"})
        async with asyncio.timeout(5):
            while not backend.entered.is_set():
                await asyncio.sleep(.01)
        await ws.close()
        with pytest.raises(WSServerHandshakeError) as cleaning:
            await client.ws_connect("/ws")
        assert cleaning.value.status == 409
    finally:
        backend.proceed.set()
    async with asyncio.timeout(5):
        while app["studio"].socket is not None:
            await asyncio.sleep(.01)
    replacement = await client.ws_connect("/ws")
    assert (await event(replacement, "status"))["state"] == "paused"
    await replacement.close()


def test_encoder_failure_still_reaps_process(tmp_path):
    class BrokenInput:
        def close(self):
            raise BrokenPipeError("Encoder exited")

    class FailedEncoder:
        stdin = BrokenInput()
        waited = False

        def wait(self, timeout):
            self.waited = True
            return 1

    recording = Recording(tmp_path, branch="a", scene_id="test", seed=1)
    recording.process = FailedEncoder()
    recording.log = io.BytesIO()
    with pytest.raises(RuntimeError, match="encoding failed"):
        recording.finish()
    assert recording.process.waited
    assert recording.log.closed


def test_legacy_keyboard_message_defaults_to_no_pointer_motion():
    assert validate_input({"keys": ["w"], "selected_arm": "right"}) == (
        frozenset({"w"}), "right", False, False, (0, 0, 0))


@pytest.mark.parametrize("change", [
    {"pointer_delta": None}, {"pointer_delta": []}, {"pointer_delta": [0, 0]},
    {"pointer_delta": [0, 0, 0, 0]}, {"pointer_delta": "0,0,0"},
    {"pointer_delta": [True, 0, 0]}, {"pointer_delta": [".01", 0, 0]},
    {"pointer_delta": [None, 0, 0]}, {"pointer_delta": [float("nan"), 0, 0]},
    {"pointer_delta": [float("inf"), 0, 0]}, {"pointer_delta": [0, 0, float("-inf")]},
    {"pointer_delta": [.1, .1, 0]}, {"pointer_delta": [.10001, 0, 0]},
    {"pointer_active": 1}, {"pointer_active": "true"}, {"pointer_commit": 1},
    {"pointer_commit": True}, {"selected_arm": []},
    {"pointer_active": False, "pointer_delta": [.01, 0, 0]},
])
def test_pointer_wire_rejects_malformed_and_out_of_bounds_inputs(change):
    with pytest.raises(ValueError):
        validate_input({"pointer_active": True, **change})


@pytest.fixture
def studio(tmp_path):
    studio = Studio(backend=None, scenes={}, output_dir=tmp_path)
    studio.running = True
    yield studio
    studio.executor.shutdown(wait=True)


def pointer_input(delta=(0, 0, 0), arm="left", active=True, commit=False, keys=()):
    return {"type": "input", "keys": list(keys), "selected_arm": arm,
            "pointer_active": active, "pointer_commit": commit, "pointer_delta": list(delta)}


def fake_clock(monkeypatch):
    """Advance only the server's clock, leaving the real event loop untouched."""
    clock = SimpleNamespace(now=0.0)

    async def sleep(seconds):
        clock.now += seconds

    async def to_thread(function, *args):
        return function(*args)

    monkeypatch.setattr(server, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(server, "asyncio", SimpleNamespace(
        sleep=sleep, to_thread=to_thread, create_task=asyncio.create_task))
    return clock


@pytest.mark.parametrize("chunk_frames", [4, 8])
async def test_full_chunk_dispatches_on_last_sample_and_latency_excludes_recording(studio, monkeypatch, chunk_frames):
    studio.backend = Backend()
    studio.backend.actions_per_chunk = chunk_frames
    samples, dispatches, metrics = [], [], []
    original_sample = studio.sample_action

    def sample(now):
        samples.append(now)
        return original_sample(now)

    async def generate(method, actions):
        dispatches.append(time.monotonic())
        await asyncio.sleep(.05)
        return method(actions)

    def record(frames, actions):
        time.sleep(.2)

    async def send(message):
        if message["type"] == "metrics":
            metrics.append(message)
            studio.running = False

    studio.sample_action = sample
    studio.model_call = generate
    studio.send = send
    studio.recorder = SimpleNamespace(append=record, frame_count=0)
    await studio.generate_loop()
    expected_dispatch = (chunk_frames - 1) * .1
    assert [at - samples[0] for at in samples[:chunk_frames]] == pytest.approx(
        [index * .1 for index in range(chunk_frames)], abs=.04)
    assert dispatches[0] - samples[0] == pytest.approx(expected_dispatch, abs=.04)
    assert metrics[0]["latency_ms"] == pytest.approx((expected_dispatch + .05) * 1000, abs=60)
    assert metrics[0]["latency_basis"] == "first_action_sample_to_generated_chunk_ready"
    assert time.monotonic() - samples[0] >= expected_dispatch + .05 + .2
    # The sampler also runs during recording, without inflating GPU latency.
    assert len(samples) >= chunk_frames + 2
    assert studio.action_stream is None


async def test_pause_between_action_samples_does_not_dispatch_partial_chunk(studio, monkeypatch):
    studio.backend = Backend()
    studio.backend.actions_per_chunk = 4
    sampled = []

    def sample(now):
        sampled.append(now)
        studio.running = False
        return studio.controller.action()

    async def generate(*args):
        pytest.fail("A paused partial chunk must not reach the model")

    studio.sample_action = sample
    studio.model_call = generate
    await studio.generate_loop()
    assert len(sampled) == 1
    assert studio.action_stream is None


@pytest.mark.parametrize("pause_after", [None, 2])
async def test_frame_publication_keeps_ten_hz_order_and_stops_on_pause(studio, monkeypatch, pause_after):
    clock = fake_clock(monkeypatch)
    frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    published = []

    async def send(message):
        published.append((clock.now, message["frame_index"]))
        if len(published) == pause_after:
            studio.running = False
            studio.epoch += 1

    studio.send = send
    await studio.publish_frames(frames, offset=8, epoch=studio.epoch)
    count = pause_after or len(frames)
    assert [timestamp for timestamp, _ in published] == pytest.approx([index * .1 for index in range(count)])
    assert [index for _, index in published] == list(range(8, 8 + count))


async def test_cold_and_slow_encoding_never_compresses_publication_intervals(studio, monkeypatch):
    clock = fake_clock(monkeypatch)
    frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    encode_times = iter([.09, .005, .2, .005])
    original = server.Image.fromarray
    published = []

    def encode(frame):
        clock.now += next(encode_times)
        return original(frame)

    async def send(message):
        published.append(message["published_at_s"])

    monkeypatch.setattr(server.Image, "fromarray", encode)
    studio.send = send
    await studio.publish_frames(frames, 0, studio.epoch)
    assert published == pytest.approx([.09, .19, .39, .49])
    assert clock.now == pytest.approx(.59)


async def test_pointer_packets_preserve_full_distance_without_sampling_or_replay(studio):
    for _ in range(8):
        await studio.command(pointer_input([.09, 0, 0]))
    assert studio.controller.action()[0] == pytest.approx(.72)
    row = studio.sample_action()
    assert row[0] == pytest.approx(.72)
    assert studio.sample_action() == row


async def test_same_distance_in_one_or_many_allowed_packets_requests_same_pose(studio):
    await studio.command(pointer_input([.06, .08, 0]))
    single = studio.controller.action()
    studio.controller.reset()
    for _ in range(10):
        await studio.command(pointer_input([.006, .008, 0]))
        studio.sample_action()
    assert studio.controller.action() == pytest.approx(single)
    await studio.command(pointer_input([-.06, -.08, 0]))
    assert studio.controller.action()[:3] == pytest.approx([0, 0, 0])


async def test_pointer_workspace_bounds_allow_immediate_direction_reversal(studio):
    for _ in range(12):
        await studio.command(pointer_input([.1, 0, 0]))
    assert studio.controller.action()[0] == 1
    await studio.command(pointer_input([-.1, 0, 0]))
    assert studio.sample_action()[0] == pytest.approx(.9)


async def test_pointer_arm_switch_preserves_each_accepted_pose(studio):
    await studio.command(pointer_input([.03, .01, 0]))
    left = studio.sample_action()
    await studio.command(pointer_input([.04, 0, 0]))
    await studio.command(pointer_input([0, -.02, .01], arm="right"))
    both = studio.sample_action()
    assert both[:3] == pytest.approx([.07, .01, 0])
    assert both[3:14] == left[3:14]
    assert both[14:17] == pytest.approx([0, -.02, .01])
    assert both[10] == both[24] == 0
    await studio.command(pointer_input(arm="left"))
    assert studio.sample_action() == both


@pytest.fixture
def input_clock(monkeypatch):
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr(server, "time", SimpleNamespace(monotonic=lambda: clock.now))
    return clock


async def test_keyboard_watchdog_preserves_accepted_pointer_pose(studio, input_clock):
    await studio.command(pointer_input([.04, 0, 0]))
    input_clock.now += 1
    assert studio.sample_action()[:3] == pytest.approx([.04, 0, 0])
    await studio.command(pointer_input(keys=["w"]))
    input_clock.now += .1
    assert studio.sample_action()[:3] == pytest.approx([.04, 0, .02])
    input_clock.now += 1
    assert studio.sample_action()[:3] == pytest.approx([.04, 0, .07])
    assert not studio.keys
    await studio.command(pointer_input([.01, 0, 0]))
    assert studio.sample_action()[:3] == pytest.approx([.05, 0, .07])


async def test_pointer_cancel_and_pause_hold_accepted_pose_and_ignore_paused_inputs(studio, input_clock):
    await studio.command(pointer_input([.04, 0, 0], keys=["w"]))
    accepted_pose = studio.controller.action()
    await studio.command(pointer_input(active=False))
    assert studio.sample_action() == accepted_pose
    assert not studio.keys
    await studio.pause()
    assert studio.sample_action() == accepted_pose
    # Paused messages cannot change either arm or advance a release tail.
    await studio.command(pointer_input([.04, 0, 0]))
    await studio.command(pointer_input([.03, 0, 0], active=False, commit=True))
    assert studio.sample_action() == accepted_pose


async def test_quick_drag_commits_once_without_advancing_keyboard(studio, input_clock):
    await studio.command(pointer_input([.03, .02, 0], active=False, commit=True, keys=["d"]))
    assert studio.controller.action()[:3] == pytest.approx([.03, .02, 0])
    input_clock.now += .1
    assert studio.sample_action()[:3] == pytest.approx([.05, .02, 0])
    await studio.command(pointer_input(active=False))
    held = studio.controller.action()
    assert studio.sample_action() == held
    assert studio.sample_action() == held


async def test_release_commit_adds_full_tail_and_never_replays_accepted_delta(studio):
    await studio.command(pointer_input([.04, 0, 0]))
    await studio.command(pointer_input([.04, 0, 0], active=False, commit=True))
    assert studio.controller.action()[0] == pytest.approx(.08)
    await studio.command(pointer_input([.02, 0, 0]))
    moved = studio.sample_action()
    assert moved[0] == pytest.approx(.10)
    await studio.command(pointer_input(active=False, commit=True))
    assert studio.controller.action() == moved
    assert studio.sample_action() == moved


async def test_release_commit_adds_only_tail_after_watchdog_timeout(studio, input_clock):
    await studio.command(pointer_input([.04, 0, 0]))
    input_clock.now += 1
    studio.sample_action()
    await studio.command(pointer_input([.01, 0, 0], active=False, commit=True))
    assert studio.sample_action()[0] == pytest.approx(.05)


async def test_invalid_pointer_command_preserves_valid_state(studio):
    await studio.command(pointer_input([.02, 0, 0]))
    accepted_pose = studio.controller.action()
    accepted_at = studio.last_input
    with pytest.raises(ValueError):
        await studio.command(pointer_input([True, 0, 0], arm="right", keys=["w"]))
    assert studio.arm == "left" and not studio.keys
    assert studio.last_input == accepted_at
    assert studio.controller.action() == accepted_pose
    assert studio.sample_action()[0] == pytest.approx(.02)


async def test_drag_during_blocked_generation_survives_until_next_model_chunk(tmp_path):
    class BlockingBackend(Backend):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.proceed = threading.Event()
            self.following_chunk = threading.Event()
            self.action_chunks = []

        def generate(self, actions):
            self.action_chunks.append(actions)
            if len(self.action_chunks) == 1:
                self.entered.set()
                self.proceed.wait(timeout=5)
            else:
                self.following_chunk.set()
            return super().generate(actions)

    backend = BlockingBackend()
    studio = Studio(backend=backend, scenes={}, output_dir=tmp_path)
    studio.running = True
    studio.generation_task = asyncio.create_task(studio.generate_loop())
    try:
        async with asyncio.timeout(5):
            while not backend.entered.is_set():
                await asyncio.sleep(.01)
        for _ in range(8):
            await studio.command(pointer_input([.09, 0, 0]))
        assert studio.controller.action()[:3] == pytest.approx([.72, 0, 0])
        # Inference may outlast the watchdog. Requested pointer displacement
        # persists, while held keyboard velocity must stop before sampling.
        await asyncio.sleep(.4)
        backend.proceed.set()
        async with asyncio.timeout(5):
            while not backend.following_chunk.is_set():
                await asyncio.sleep(.01)
        assert backend.action_chunks[0][0][:3] == [0, 0, 0]
        assert backend.action_chunks[1][0][:3] == pytest.approx([.72, 0, 0])
        assert not studio.keys
    finally:
        backend.proceed.set()
        await studio.cleanup(None)


async def test_input_during_reset_cannot_queue_motion_for_new_scene(tmp_path, scenes):
    class SlowBackend(Backend):
        entered = threading.Event()
        proceed = threading.Event()

        def reset(self, path, seed=0):
            self.entered.set()
            self.proceed.wait(timeout=5)
            super().reset(path, seed)

    backend = SlowBackend()
    studio = Studio(backend=backend, scenes=scenes, output_dir=tmp_path)
    studio.ready = studio.running = True
    await studio.command(pointer_input([.04, 0, 0]))
    task = asyncio.create_task(studio.reset("test", 42))
    try:
        async with asyncio.timeout(5):
            while not backend.entered.is_set():
                await asyncio.sleep(.01)
        await studio.command(pointer_input([.04, 0, 0], arm="right", keys=["w"]))
        await studio.command(pointer_input([.03, 0, 0], active=False, commit=True))
        assert not studio.keys
        assert studio.controller.action()[:3] == pytest.approx([.04, 0, 0])
        assert studio.controller.action()[14:17] == [0, 0, 0]
        backend.proceed.set()
        await task
        assert studio.controller.action()[:3] == [0, 0, 0]
        assert studio.controller.action()[14:17] == [0, 0, 0]
    finally:
        backend.proceed.set()
        await task
        await studio.cleanup(None)


async def test_continuous_sampling_preserves_short_tap_during_gpu_stall_and_resume(tmp_path):
    class BlockedBackend(Backend):
        actions_per_chunk = 4

        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.proceed = threading.Event()
            self.chunks = []

        def generate(self, actions):
            self.chunks.append(actions)
            if len(self.chunks) == 1:
                self.entered.set()
                self.proceed.wait(timeout=5)
            return super().generate(actions)

    backend = BlockedBackend()
    studio = Studio(backend=backend, scenes={}, output_dir=tmp_path)
    studio.ready = studio.running = studio.scene_initialized = True
    messages = []
    samples = []
    original = studio.sample_action

    def sample(now):
        samples.append(now)
        return original(now)

    async def send(message):
        messages.append(message)

    async def until(condition):
        async with asyncio.timeout(5):
            while not condition():
                await asyncio.sleep(.005)

    studio.sample_action = sample
    studio.send = send
    studio.generation_task = asyncio.create_task(studio.generate_loop())
    try:
        await until(backend.entered.is_set)
        await studio.command({"type": "input", "keys": ["d"], "input_id": 1})
        await asyncio.sleep(.05)
        await studio.command({"type": "input", "keys": [], "input_id": 2})
        ack = [message for message in messages if message["type"] == "input_ack"]
        expected = .2 * (ack[1]["received_at_s"] - ack[0]["received_at_s"])
        assert expected > .009
        assert ack[1]["requested_action"][0] == pytest.approx(expected)
        # GPU is still blocked; samples keep arriving and only one chunk waits.
        await asyncio.sleep(.65)
        assert not backend.proceed.is_set()
        assert len(samples) >= 10
        stream = studio.action_stream
        assert stream.queue.qsize() == 4
        assert stream.skipped >= 2
        backend.proceed.set()
        await until(lambda: len(backend.chunks) >= 2)
        assert all(row[0] == pytest.approx(expected) for row in backend.chunks[1])
        await studio.pause()
        assert studio.action_stream is None and stream.task.done()
        assert studio.controller.action()[0] == pytest.approx(expected)
        # Paused keys and pointer messages cannot leak into the next run.
        await studio.command(pointer_input([.04, 0, 0], keys=["d"]))
        old_chunks = len(backend.chunks)
        await studio.resume()
        await until(lambda: len(backend.chunks) > old_chunks)
        assert all(row[0] == pytest.approx(expected) for row in backend.chunks[old_chunks])
        await studio.pause()
        frames = [message for message in messages if message["type"] == "frame"]
        assert frames
        for frame in frames:
            assert frame["published_at_s"] >= frame["action_sampled_at_s"]
        for epoch in {frame["epoch"] for frame in frames}:
            indices = [frame["frame_index"] for frame in frames if frame["epoch"] == epoch]
            assert indices == list(range(len(indices)))
    finally:
        backend.proceed.set()
        await studio.cleanup(None)


async def test_pause_waits_for_gpu_but_stops_sampler_and_discards_unpublished_output(tmp_path):
    backend = Backend()
    studio = Studio(backend=backend, scenes={}, output_dir=tmp_path)
    studio.running = True
    entered, proceed = asyncio.Event(), asyncio.Event()
    messages = []

    async def generate(method, *args):
        entered.set()
        await proceed.wait()
        return method(*args)

    async def send(message):
        messages.append(message)

    studio.model_call = generate
    studio.send = send
    studio.generation_task = asyncio.create_task(studio.generate_loop())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        stream = studio.action_stream
        paused = asyncio.create_task(studio.pause())
        await asyncio.sleep(.03)
        assert stream.task.done()
        assert not paused.done()
        proceed.set()
        await asyncio.wait_for(paused, 1)
        assert not any(message["type"] in {"metrics", "frame"} for message in messages)
        assert studio.action_stream is None
    finally:
        proceed.set()
        await studio.cleanup(None)


@pytest.mark.parametrize("failure", ["sample", "generate", "publish"])
async def test_loop_failures_leave_no_sampler_and_allow_backend_cleanup(tmp_path, failure):
    backend = Backend()
    studio = Studio(backend=backend, scenes={}, output_dir=tmp_path)
    studio.running = True

    def broken(*args):
        raise RuntimeError("deliberate failure")

    async def broken_async(*args):
        broken()

    if failure == "sample":
        studio.sample_action = broken
    elif failure == "generate":
        backend.generate = broken
    else:
        studio.publish_frames = broken_async
    studio.generation_task = asyncio.create_task(studio.generate_loop())
    await asyncio.wait_for(studio.generation_task, 2)
    assert studio.state == "error"
    assert not studio.running
    assert studio.action_stream is None
    calls_before_cleanup = len(backend.calls)
    await studio.cleanup(None)
    assert len(backend.calls) == calls_before_cleanup + 1  # close still runs


@pytest.mark.parametrize("input_id", [True, -1, 1.5, "1", 2**53])
async def test_invalid_diagnostic_id_does_not_change_input(studio, input_clock, input_id):
    await studio.command({"type": "input", "keys": ["d"]})
    input_clock.now += .1
    before = studio.controller.action()
    with pytest.raises(ValueError, match="Input ID"):
        await studio.command({"type": "input", "keys": [], "input_id": input_id})
    assert studio.controller.action() == before
    assert studio.keys == frozenset({"d"})
