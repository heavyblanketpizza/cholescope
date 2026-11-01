"""Single-operator surgical model server. Preview mode never synthesizes footage."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import functools
import hashlib
import hmac
import io
import json
import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web
from PIL import Image

from .control_stream import ActionStream
from .controls import KeyboardController, TimedController, validate_pointer_delta
from .recording import Recording

STATIC = Path(__file__).parent / "static"
STATIC_FILES = frozenset({
    "index.html", "student.html", "style.css",
    "app.mjs", "student.mjs", "controls.mjs", "pointer-controls.mjs", "pod-controls.mjs",
    "control-recovery.mjs",
})
LOG = logging.getLogger(__name__)
ALLOWED_KEYS = frozenset("w s a d q e z x f v shift arrowup arrowdown arrowleft arrowright".split())
INPUT_TIMEOUT = 0.35
MAX_POINTER_MESSAGE = 0.10


async def static_file(request):
    # An in-place GPU deployment can retain files from retired views.
    filename = request.match_info["filename"]
    if filename not in STATIC_FILES:
        raise web.HTTPNotFound()
    return web.FileResponse(STATIC / filename)


async def retired_view(request):
    raise web.HTTPFound(location="/")


def load_scenes(scene_dir: Path | None, upstream_root: Path | None):
    scenes = {}
    if scene_dir:
        for path in sorted(scene_dir.iterdir()):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} or not path.is_file():
                continue
            with Image.open(path) as image:
                image.verify()
            scene_id = f"scene-{len(scenes) + 1}"
            scenes[scene_id] = {"id": scene_id, "name": path.stem.replace("_", " "),
                                "path": path.resolve(), "description": "Operator-supplied surgical scene"}
    if not scenes and upstream_root:
        path = upstream_root / "assets/example_data/scenes/example_frame.png"
        if path.is_file():
            scenes["cmr-4148"] = {"id": "cmr-4148", "name": "Gallbladder · CMR 4148", "path": path.resolve(),
                                  "description": "Open-H / CMR · CC BY 4.0 · initial recorded frame"}
    return scenes


def validate_input(payload):
    keys = payload.get("keys", [])
    arm = payload.get("selected_arm", "left")
    if not isinstance(keys, list) or len(keys) > 20 or any(not isinstance(k, str) or k not in ALLOWED_KEYS for k in keys):
        raise ValueError("Invalid keyboard state")
    if not isinstance(arm, str) or arm not in {"left", "right"}:
        raise ValueError("Invalid instrument selection")
    pointer_active = payload.get("pointer_active", False)
    if type(pointer_active) is not bool:
        raise ValueError("Pointer active must be a boolean")
    pointer_commit = payload.get("pointer_commit", False)
    if type(pointer_commit) is not bool:
        raise ValueError("Pointer commit must be a boolean")
    if pointer_commit and pointer_active:
        raise ValueError("Pointer commit requires a released drag")
    delta = payload.get("pointer_delta", [0.0, 0.0, 0.0])
    if not isinstance(delta, list):
        raise ValueError("Pointer delta must be a three-number array")
    delta = validate_pointer_delta(delta)
    if math.hypot(*delta) > MAX_POINTER_MESSAGE:
        raise ValueError("Pointer delta magnitude cannot exceed 0.10")
    if not pointer_active and not pointer_commit and any(delta):
        raise ValueError("Pointer motion requires an active drag or release commit")
    return frozenset(keys), arm, pointer_active, pointer_commit, delta


class Studio:
    def __init__(self, *, backend, scenes, output_dir: Path, max_record_seconds=30):
        self.backend, self.scenes, self.output_dir = backend, scenes, output_dir
        self.max_record_frames = int(max_record_seconds * 10)
        self.controller = KeyboardController()
        self.timed = TimedController(self.controller, input_timeout=INPUT_TIMEOUT)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cholescope-gpu")
        self.lock = asyncio.Lock()
        self.socket = None
        self.ready = False
        self.scene_initialized = False
        self.running = False
        self.state = "loading" if backend else "preview"
        self.message = "Loading pretrained weights" if backend else "Interface preview. Connect a GPU backend to generate video."
        self.scene_id = next(iter(scenes), None)
        self.seed = 42
        self.recorder = None
        self.generation_task = None
        self.action_stream = None
        self.startup_task = None
        self.operator_cleanup_tasks = set()
        self.epoch = 0
        self.closed = False
        from .student_lab import StudentLab
        self.student_lab = StudentLab(self)

    async def model_call(self, method, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(self.executor, functools.partial(method, *args, **kwargs))

    def status(self):
        return {"type": "status", "backend_ready": self.ready, "backend_name": "Cosmos3-H-Dreams" if self.backend else "Preview",
                "state": self.state, "message": self.message, "scene_id": self.scene_id, "seed": self.seed}

    async def send(self, payload):
        if self.socket is not None and not self.socket.closed:
            try:
                await self.socket.send_json(payload)
            except (ConnectionError, RuntimeError):
                self.release()
                self.running = False

    async def set_state(self, state, message):
        self.state, self.message = state, message
        await self.send(self.status())

    async def startup(self, app):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.backend:
            self.startup_task = asyncio.create_task(self.initialize())

    async def initialize(self):
        try:
            await self.model_call(self.backend.load)
            self.ready = True
            await self.set_state("ready", "Model loaded. Choose a scene and start.")
        except Exception:
            LOG.exception("Model startup failed")
            await self.set_state("error", "Model startup failed. Check the server log; no video was generated.")

    @property
    def keys(self):
        return self.timed.keys

    @property
    def arm(self):
        return self.timed.arm

    @property
    def last_input(self):
        return self.timed.last_input

    def release(self):
        self.timed.release(time.monotonic())

    def sample_action(self, now=None):
        """Export the pose integrated up to this real sampling timestamp."""
        now = time.monotonic() if now is None else now
        return self.timed.sample(now)

    async def finish_recording(self):
        recorder, self.recorder = self.recorder, None
        if recorder:
            try:
                result = await asyncio.to_thread(recorder.finish)
                await self.send(result)
            except (ValueError, RuntimeError, OSError) as exc:
                await self.send({"type": "error", "message": str(exc)})

    async def pause(self):
        self.running = False
        self.epoch += 1
        self.release()
        if self.action_stream:
            self.action_stream.stop()
        # An in-flight model call finishes on its owner thread. Its output is discarded.
        if self.generation_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self.generation_task
            self.generation_task = None
        await self.finish_recording()
        if self.ready:
            await self.set_state("paused", "Paused. Instrument inputs released.")

    async def reset(self, scene_id, seed, *, run=True):
        if not self.ready:
            raise ValueError("A loaded GPU model is required")
        if scene_id not in self.scenes:
            raise ValueError("Choose an available scene")
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("Seed must be an integer between 0 and 2147483647")
        await self.pause()
        self.scene_initialized = False
        self.scene_id, self.seed = scene_id, seed
        await self.set_state("loading", "Preparing this scene. First use includes GPU warmup.")
        await self.model_call(self.backend.reset, self.scenes[scene_id]["path"], seed=seed)
        self.timed.reset(time.monotonic())
        self.scene_initialized = True
        if not run:
            return
        self.running = True
        self.generation_task = asyncio.create_task(self.generate_loop())
        await self.set_state("running", "Live generated video · mouse and keyboard controls enabled")

    async def resume(self):
        if not self.ready or self.state != "paused" or not self.scene_initialized:
            raise ValueError("Start a scene before resuming")
        self.timed.begin(time.monotonic())
        self.running = True
        self.generation_task = asyncio.create_task(self.generate_loop())
        await self.set_state("running", "Resumed from the current generated state")

    async def generate_loop(self):
        epoch = self.epoch
        frame_index = 0
        chunk_id = 0
        playback = None
        active = lambda: self.running and epoch == self.epoch and not self.closed
        stream = ActionStream(self.sample_action, active, self.backend.actions_per_chunk,
                              clock=time.monotonic, sleep=asyncio.sleep)
        self.action_stream = stream
        stream.start()
        try:
            while active():
                rows = await stream.chunk(self.backend.actions_per_chunk)
                if rows is None:
                    return
                actions = [row.action for row in rows]
                sampled_at = [row.at for row in rows]
                timing = {"epoch": epoch, "chunk_id": chunk_id,
                          "generation_started_at_s": time.monotonic(),
                          "action_sampled_at_s": sampled_at, "first_frame_index": frame_index}
                await self.send({"type": "generation_started", **timing})
                if not active():
                    return
                result = await self.model_call(self.backend.generate, actions)
                chunk_ready_at = time.monotonic()
                if not active():
                    return
                if playback:
                    await playback
                if not active():
                    return
                if self.recorder:
                    await asyncio.to_thread(self.recorder.append, result.frames, actions)
                    if self.recorder.frame_count >= self.max_record_frames:
                        await self.finish_recording()
                metrics = {"type": "metrics", "fps": result.metrics.get("generation_fps"),
                           **result.metrics, **timing,
                           "latency_ms": (chunk_ready_at - sampled_at[0]) * 1000,
                           "latency_basis": "first_action_sample_to_generated_chunk_ready",
                           "model_fps": 10, "generation_ready_at_s": chunk_ready_at,
                           "first_action": actions[0], "last_action": actions[-1],
                           "skipped_action_samples": stream.skipped}
                await self.send(metrics)
                playback = asyncio.create_task(self.publish_frames(result.frames, frame_index, epoch, sampled_at))
                frame_index += len(result.frames)
                chunk_id += 1
        except Exception:
            LOG.exception("Generation failed")
            self.running = False
            self.release()
            await self.finish_recording()
            await self.set_state("error", "Generation failed. Check the server log and reset the scene.")
        finally:
            await stream.close()
            if self.action_stream is stream:
                self.action_stream = None
            if playback:
                # A failed publisher may already have been handled above.
                # Always retrieve its exception without making pause/cleanup
                # abandon the single-owner GPU executor.
                outcome, = await asyncio.gather(playback, return_exceptions=True)
                if isinstance(outcome, Exception) and self.state != "error":
                    LOG.error("Frame publication failed", exc_info=(type(outcome), outcome, outcome.__traceback__))
                    self.running = False
                    self.release()
                    await self.finish_recording()
                    await self.set_state("error", "Frame publication failed. Reset the scene to continue.")

    async def publish_frames(self, frames, offset, epoch, sampled_at=None):
        deadline = time.monotonic()
        for index, frame in enumerate(frames):
            if not self.running or epoch != self.epoch:
                return
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, format="JPEG", quality=85)
            # Encode ahead of the next publication deadline. A cold JPEG
            # encoder must not consume the first frame's display interval.
            await asyncio.sleep(max(0., deadline - time.monotonic()))
            if not self.running or epoch != self.epoch:
                return
            published_at = time.monotonic()
            await self.send({"type": "frame", "jpeg": base64.b64encode(buffer.getvalue()).decode(),
                             "frame_index": offset + index, "generated": True, "epoch": epoch,
                             "published_at_s": published_at,
                             "action_sampled_at_s": sampled_at[index] if sampled_at else None})
            deadline = published_at + .1
        # Include the last interval so consecutive chunks stay at 10 Hz.
        await asyncio.sleep(max(0., deadline - time.monotonic()))

    async def command(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Expected a command object")
        kind = payload.get("type")
        if kind == "student_status":
            await self.student_lab.send()
            return
        if kind == "input":
            keys, arm, pointer_active, pointer_commit, delta = validate_input(payload)
            input_id = payload.get("input_id")
            if input_id is not None and (type(input_id) is not int or not 0 <= input_id < 2**53):
                raise ValueError("Input ID must be a non-negative safe integer")
            if not self.running or self.student_lab.active:
                self.release()
                return
            now = time.monotonic()
            self.timed.update(keys, arm, now=now,
                              pointer_delta=delta if pointer_active or pointer_commit else None)
            if input_id is not None:
                await self.send({"type": "input_ack", "input_id": input_id, "epoch": self.epoch,
                                 "received_at_s": now, "requested_action": self.controller.action()})
            return
        async with self.lock:
            if kind == "student_train":
                await self.student_lab.train()
            elif kind == "student_reset":
                await self.student_lab.reset_learning()
            elif kind == "student_trial":
                await self.student_lab.start(payload)
            elif kind in {"start", "reset"}:
                await self.reset(payload.get("scene_id", self.scene_id), payload.get("seed", self.seed))
            elif kind in {"pause", "stop"}:
                await self.pause()
            elif kind == "resume":
                await self.resume()
            elif kind == "record_start":
                branch = payload.get("branch")
                if branch not in {"a", "b"}:
                    raise ValueError("Unknown comparison branch")
                # Every branch begins at the same initial condition, not midway through a rollout.
                await self.reset(self.scene_id, self.seed)
                provenance = await self.model_call(self.backend.provenance) if hasattr(self.backend, "provenance") else {}
                scene_hash = hashlib.sha256(self.scenes[self.scene_id]["path"].read_bytes()).hexdigest()
                self.recorder = Recording(self.output_dir, branch=branch, scene_id=self.scene_id, seed=self.seed,
                                          provenance={**provenance, "initial_image_sha256": scene_hash})
                await self.send({"type": "recording_started", "branch": branch})
            elif kind == "record_stop":
                await self.pause()
            else:
                raise ValueError("Unknown command")

    async def cleanup(self, app):
        self.closed = True
        # Request cancellation must not abandon a GPU call or release operator
        # ownership early. Disconnect cleanups outlive their HTTP handlers.
        if self.operator_cleanup_tasks:
            await asyncio.gather(*self.operator_cleanup_tasks, return_exceptions=True)
        await self.pause()
        if self.startup_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self.startup_task
        if self.backend:
            await self.model_call(self.backend.close)
        self.executor.shutdown(wait=True)


def create_app(*, backend=None, scenes=None, output_dir=Path("outputs"), access_token=None):
    studio = Studio(backend=backend, scenes=scenes or {}, output_dir=Path(output_dir))

    @web.middleware
    async def authorization(request, handler):
        if access_token:
            header = request.headers.get("Authorization", "")
            supplied = ""
            try:
                if header.startswith("Basic "):
                    supplied = base64.b64decode(header[6:], validate=True).decode().split(":", 1)[1]
                elif header.startswith("Bearer "):
                    supplied = header[7:]
            except (ValueError, IndexError, UnicodeError):
                pass
            if not hmac.compare_digest(supplied.encode(), access_token.encode()):
                raise web.HTTPUnauthorized(headers={"WWW-Authenticate": 'Basic realm="CholeScope"'})
        response = await handler(request)
        if not isinstance(response, web.WebSocketResponse):
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
        return response

    app = web.Application(middlewares=[authorization], client_max_size=16 * 1024)
    app["studio"] = studio

    async def status(request):
        return web.json_response(studio.status())

    async def list_scenes(request):
        return web.json_response({"scenes": [{k: v for k, v in s.items() if k != "path"} for s in studio.scenes.values()]})

    async def scene_image(request):
        scene = studio.scenes.get(request.match_info["scene"])
        if not scene:
            raise web.HTTPNotFound()
        return web.FileResponse(scene["path"])

    async def run_file(request):
        run_id, filename = request.match_info["run_id"], request.match_info["filename"]
        if not re.fullmatch(r"[a-f0-9]{32}", run_id) or filename not in {"video.mp4", "actions"}:
            raise web.HTTPNotFound()
        target = studio.output_dir / run_id / ("actions.json" if filename == "actions" else filename)
        if not target.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(target)

    async def websocket(request):
        origin = request.headers.get("Origin")
        if origin and urlsplit(origin).netloc != request.host:
            raise web.HTTPForbidden(reason="Cross-origin control is disabled")
        if studio.socket is not None:
            raise web.HTTPConflict(reason="Another operator is connected")
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=8192)
        studio.socket = ws
        pending = set()

        async def handle(payload):
            try:
                await studio.command(payload)
            except (ValueError, RuntimeError) as exc:
                await studio.send({"type": "error", "message": str(exc)})
            except Exception:
                LOG.exception("Operator command failed")
                await studio.send({"type": "error", "message": "Command failed. Inspect the server log."})

        try:
            await ws.prepare(request)
            await studio.send(studio.status())
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(message.data)
                    except json.JSONDecodeError:
                        await studio.send({"type": "error", "message": "Invalid JSON command"})
                        continue
                    if isinstance(payload, dict) and payload.get("type") == "input":
                        await handle(payload)
                    elif len(pending) < 2:
                        task = asyncio.create_task(handle(payload))
                        pending.add(task)
                        task.add_done_callback(pending.discard)
                    else:
                        await studio.send({"type": "error", "message": "Wait for the current command to finish"})
        finally:
            studio.release()
            studio.running = False

            async def finish_operator():
                try:
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    await studio.pause()
                finally:
                    if studio.socket is ws:
                        studio.socket = None

            def cleanup_finished(task):
                studio.operator_cleanup_tasks.discard(task)
                if not task.cancelled() and task.exception() is not None:
                    LOG.error("Operator cleanup failed", exc_info=task.exception())

            # A transport disconnect may cancel this request *during* finally.
            # Shield the complete cleanup, not only the underlying worker call:
            # cancelling gather would otherwise cancel pending command tasks.
            cleanup_task = asyncio.create_task(finish_operator())
            studio.operator_cleanup_tasks.add(cleanup_task)
            cleanup_task.add_done_callback(cleanup_finished)
            await asyncio.shield(cleanup_task)
        return ws

    async def index(request):
        return web.FileResponse(STATIC / "index.html")

    async def student_page(request):
        return web.FileResponse(STATIC / "student.html")

    async def student_checkpoint(request):
        try:
            checkpoint = studio.student_lab.checkpoint(request.match_info["phase"])
        except ValueError as error:
            raise web.HTTPNotFound(reason=str(error))
        return web.json_response(checkpoint, headers={"Content-Disposition": "attachment; filename=student-checkpoint.json"})

    app.router.add_get("/", index)
    app.router.add_get("/student", student_page)
    app.router.add_get("/visual", retired_view)
    app.router.add_get("/visual/", retired_view)
    app.router.add_get("/api/student/checkpoint/{phase}", student_checkpoint)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/scenes", list_scenes)
    app.router.add_get("/api/scene/{scene}/image", scene_image)
    app.router.add_get("/api/runs/{run_id}/{filename}", run_file)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/static/{filename:.*}", static_file)
    app.on_startup.append(studio.startup)
    app.on_cleanup.append(studio.cleanup)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--scene-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--preview", action="store_true", help="Preview the UI without pretending to run a model")
    parser.add_argument("--preset", default="wan-taehv-chunk2-2steps-fademem-compile")
    parser.add_argument("--decoder", choices=("wan", "taehv"), default="wan",
                        help="Full Wan VAE decoder, or the preset's faster but softer TAEHV")
    parser.add_argument("--chunk-frames", type=int, choices=(4, 8),
                        help="Video frames per model call; 4 is an experimental lower-latency opt-in")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    token = os.environ.get("CHOLESCOPE_ACCESS_TOKEN")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not token:
        parser.error("Non-loopback serving requires CHOLESCOPE_ACCESS_TOKEN; an SSH tunnel is preferred")
    backend = None
    if not args.preview:
        if not args.checkpoint or not args.checkpoint.is_file():
            parser.error("Supply an existing --checkpoint, or use --preview")
        from .model_backend import CosmosBackend
        backend = CosmosBackend(args.checkpoint, upstream_root=args.upstream_root, preset=args.preset,
                                decoder=args.decoder, chunk_frames=args.chunk_frames)
    scenes = load_scenes(args.scene_dir, args.upstream_root)
    web.run_app(create_app(backend=backend, scenes=scenes, output_dir=args.output_dir, access_token=token),
                host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
