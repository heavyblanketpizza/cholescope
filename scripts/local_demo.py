#!/usr/bin/env python3
"""Serve the demo locally while a privately configured Runpod pod is stopped."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
from pathlib import Path
import re
import secrets
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web
from cholescope.server import STATIC, load_scenes, retired_view, static_file
from pod_manager import PodError


PROXY_PATH = re.compile(
    r"/api/(?:status|student/checkpoint/(?:before|after)|"
    r"runs/[a-f0-9]{32}/(?:video\.mp4|actions))\Z"
)


def create_app(manager, *, backend_url="http://127.0.0.1:8793", scenes=None, output_dir=None,
               port=8792):
    """All cloud authority stays in manager; browser requests cannot choose a pod."""
    backend = urlsplit(backend_url)
    if backend.scheme != "http" or backend.hostname != "127.0.0.1" or backend.path not in {"", "/"}:
        raise ValueError("The model backend must be a local SSH tunnel")
    if backend.username or backend.password or backend.query or backend.fragment:
        raise ValueError("Invalid model backend URL")
    backend_url = backend_url.rstrip("/")
    scenes = scenes or {}
    output_dir = Path(output_dir).resolve() if output_dir else None

    @web.middleware
    async def local_only(request, handler):
        host = request.host
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"} if port else {host}
        # Port zero is reserved for test servers; production main always uses an
        # explicit nonzero port. Reject DNS rebinding and cross-site control.
        if host not in allowed or urlsplit("http://" + host).hostname not in {"127.0.0.1", "localhost"}:
            raise web.HTTPForbidden(reason="Local access only")
        origin = request.headers.get("Origin")
        if origin and origin != f"http://{host}":
            raise web.HTTPForbidden(reason="Same-origin access required")
        if request.method == "POST" or request.path == "/ws":
            if request.headers.get("Sec-Fetch-Site", "same-origin") not in {"same-origin", "none"}:
                raise web.HTTPForbidden(reason="Same-origin control required")
        if request.method == "POST":
            if request.headers.get("X-CholeScope-Request") != "1" or request.content_type != "application/json":
                raise web.HTTPForbidden(reason="Use the local demo controls")
            try:
                payload = await request.json()
            except (json.JSONDecodeError, UnicodeError):
                raise web.HTTPBadRequest(reason="Expected an empty JSON object")
            if payload != {}:
                raise web.HTTPBadRequest(reason="Pod settings are configured locally")
        response = await handler(request)
        if not isinstance(response, web.WebSocketResponse):
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Cache-Control"] = "no-store"
        return response

    app = web.Application(middlewares=[local_only], client_max_size=4096)
    app["manager"] = manager
    app["sockets"] = set()
    remote_sockets = {}
    handshakes = set()
    control_token = None
    takeover_lock = asyncio.Lock()

    def has_control(request):
        supplied = request.query.get("control_token", "")
        return control_token is None or secrets.compare_digest(control_token.encode(), supplied.encode())

    async def close_operators(message):
        # Close the upstream connection directly as well as the browser socket;
        # an idle or unresponsive browser must not retain the model's operator.
        browsers = tuple(app["sockets"])
        remotes = tuple(remote_sockets.values())
        await asyncio.gather(*(ws.close(code=1001, message=message) for ws in (*browsers, *remotes)),
                             return_exceptions=True)

    async def startup(app):
        app["http"] = ClientSession(timeout=ClientTimeout(total=30, connect=5))

    async def cleanup(app):
        await close_operators(b"Local launcher closing")
        await app["http"].close()
        if hasattr(manager, "close"):
            await manager.close()

    async def page(request):
        name = {"/": "index.html", "/student": "student.html"}[request.path]
        return web.FileResponse(STATIC / name)

    async def pod_status(request):
        return web.json_response(await manager.status())

    async def control_status(request):
        return web.json_response({"supported": True,
                                  "occupied": bool(app["sockets"] or handshakes),
                                  "requires_claim": not has_control(request)})

    async def takeover(request):
        nonlocal control_token
        async with takeover_lock:
            # Invalidating first also fences handshakes currently awaiting the
            # backend, including auto-retries from an older browser tab.
            control_token = secrets.token_urlsafe(32)
            await close_operators(b"Control moved to another tab")
            return web.json_response({"control_token": control_token})

    async def start(request):
        try:
            return web.json_response(await manager.start(), status=202)
        except PodError as exc:
            return web.json_response({"message": str(exc)}, status=409)

    async def stop(request):
        # Close the operator first so the model server can flush its recording.
        await close_operators(b"Pod stopping")
        return web.json_response(await manager.stop(), status=202)

    async def list_scenes(request):
        return web.json_response({"scenes": [{k: v for k, v in s.items() if k != "path"} for s in scenes.values()]})

    async def scene_image(request):
        scene = scenes.get(request.match_info["scene"])
        if not scene or not Path(scene["path"]).is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(scene["path"])

    async def proxy(request):
        if not PROXY_PATH.fullmatch(request.path):
            raise web.HTTPNotFound()
        # Previously downloaded recordings remain viewable without a GPU.
        if output_dir and request.path.startswith("/api/runs/"):
            parts = request.path.split("/")
            target = output_dir / parts[3] / ("actions.json" if parts[4] == "actions" else parts[4])
            if target.is_file() and target.resolve().is_relative_to(output_dir):
                return web.FileResponse(target)
        state = await manager.status()
        if state.get("state") != "ready":
            if request.path == "/api/status":
                return web.json_response({"type": "status", "backend_ready": False,
                    "state": "paused", "message": state.get("message", "Start pod to generate"),
                    "scene_id": next(iter(scenes), None), "seed": 42, "backend_name": "Cosmos3-H-Dreams"})
            raise web.HTTPServiceUnavailable(reason="Start the pod to use this feature")
        try:
            headers = {name: request.headers[name] for name in ("Range", "If-Range") if name in request.headers}
            async with app["http"].get(backend_url + request.path, headers=headers, allow_redirects=False) as remote:
                response = web.StreamResponse(status=remote.status, headers={
                    name: remote.headers[name] for name in
                    ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges", "Content-Disposition")
                    if name in remote.headers})
                await response.prepare(request)
                async for chunk in remote.content.iter_chunked(64 * 1024):
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (ClientError, OSError, asyncio.TimeoutError):
            raise web.HTTPServiceUnavailable(reason="Model connection unavailable") from None

    async def websocket(request):
        if not has_control(request):
            raise web.HTTPConflict(reason="Control belongs to another tab")
        if (await manager.status()).get("state") != "ready":
            raise web.HTTPServiceUnavailable(reason="Start the pod first")
        if not has_control(request):
            raise web.HTTPConflict(reason="Control belongs to another tab")
        handshake = object()
        handshakes.add(handshake)
        try:
            try:
                remote = await app["http"].ws_connect(backend_url + "/ws", origin=backend_url,
                                                      heartbeat=20, max_msg_size=4 * 1024 * 1024)
            except Exception:
                raise web.HTTPServiceUnavailable(reason="Model connection unavailable") from None
            if not has_control(request):
                await remote.close()
                raise web.HTTPConflict(reason="Control belongs to another tab")
        finally:
            handshakes.discard(handshake)
        socket = web.WebSocketResponse(heartbeat=20, max_msg_size=8192)
        app["sockets"].add(socket)
        remote_sockets[socket] = remote

        async def relay(source, destination):
            async for message in source:
                if message.type == WSMsgType.TEXT:
                    await destination.send_str(message.data)
                elif message.type == WSMsgType.BINARY:
                    await destination.send_bytes(message.data)
                elif message.type == WSMsgType.ERROR:
                    break

        tasks = []
        try:
            await socket.prepare(request)
            tasks = [asyncio.create_task(relay(socket, remote)), asyncio.create_task(relay(remote, socket))]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await remote.close()
            await socket.close()
            app["sockets"].discard(socket)
            remote_sockets.pop(socket, None)
        return socket

    for path in ("/", "/student"):
        app.router.add_get(path, page)
    app.router.add_get("/visual", retired_view)
    app.router.add_get("/visual/", retired_view)
    app.router.add_get("/api/pod/status", pod_status)
    app.router.add_get("/api/control/status", control_status)
    app.router.add_post("/api/control/takeover", takeover)
    app.router.add_post("/api/pod/start", start)
    app.router.add_post("/api/pod/stop", stop)
    app.router.add_get("/api/scenes", list_scenes)
    app.router.add_get("/api/scene/{scene}/image", scene_image)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/api/{tail:.*}", proxy)
    app.router.add_get("/static/{filename:.*}", static_file)
    async def response_headers(request, response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
        response.headers["Cache-Control"] = "no-store"

    app.on_response_prepare.append(response_headers)
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Private local configuration outside Git")
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--scene-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a local port between 1024 and 65535")
    from pod_manager import PodManager
    manager = PodManager(args.config)
    web.run_app(create_app(manager, backend_url=manager.backend_url,
                           scenes=load_scenes(args.scene_dir, args.upstream_root),
                           output_dir=args.output_dir, port=args.port),
                host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
