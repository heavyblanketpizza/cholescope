"""The browser never receives cloud credentials or arbitrary cloud authority."""
import asyncio
import sys
from pathlib import Path

import pytest
from aiohttp import WSServerHandshakeError, WSMsgType, web
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from local_demo import create_app


class Manager:
    def __init__(self, state="stopped"):
        self.state = state
        self.starts = 0
        self.stops = 0

    async def status(self):
        return {"available": True, "state": self.state, "can_start": self.state == "stopped",
                "can_stop": self.state == "ready", "message": "Pod paused"}

    async def start(self):
        self.starts += 1
        self.state = "starting"
        return await self.status()

    async def stop(self):
        self.stops += 1
        self.state = "stopping"
        return await self.status()


async def test_stopped_interface_and_scene_are_local(aiohttp_client, tmp_path):
    scene = tmp_path / "source.png"
    Image.new("RGB", (8, 8)).save(scene)
    manager = Manager()
    client = await aiohttp_client(create_app(manager, scenes={"test": {"id": "test", "path": scene}}, port=0))
    for path in ("/", "/student", "/static/pod-controls.mjs", "/static/control-recovery.mjs",
                 "/api/scenes", "/api/scene/test/image"):
        response = await client.get(path)
        assert response.status == 200
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Content-Security-Policy"] == "frame-ancestors 'none'"
    assert (await (await client.get("/api/status")).json())["backend_ready"] is False
    assert (await client.get("/api/student/checkpoint/before")).status == 503
    assert manager.starts == 0


async def test_retired_view_redirects_and_never_reaches_backend(aiohttp_client):
    backend = web.Application()
    requests = []

    async def old_api(request):
        requests.append(request.path)
        return web.json_response({"old_feature": True})

    backend.router.add_get("/{tail:.*}", old_api)
    remote = await aiohttp_client(backend)
    manager = Manager("ready")
    client = await aiohttp_client(create_app(manager, backend_url=str(remote.make_url("/")), port=0))
    for path in ("/visual", "/visual/", "/visual?setup=1"):
        response = await client.get(path, allow_redirects=False)
        assert response.status == 302
        assert response.headers["Location"] == "/"
    for path in ("/api/visual/metrics", "/api/visual/checkpoint/before", "/api/visual/checkpoint/after",
                 "/static/visual.html", "/static/visual.mjs", "/static/visual.css"):
        assert (await client.get(path)).status == 404
    assert requests == []
    assert manager.starts == manager.stops == 0


async def test_only_local_explicit_empty_mutations_are_accepted(aiohttp_client):
    manager = Manager()
    client = await aiohttp_client(create_app(manager, port=0))
    for headers, payload in [({}, {}), ({"X-CholeScope-Request": "1", "Origin": "http://evil.example"}, {}),
                             ({"X-CholeScope-Request": "1", "Sec-Fetch-Site": "cross-site"}, {}),
                             ({"X-CholeScope-Request": "1", "Host": "evil.example"}, {}),
                             ({"X-CholeScope-Request": "1"}, {"pod_id": "different"})]:
        assert (await client.post("/api/pod/start", json=payload, headers=headers)).status in {400, 403}
    assert manager.starts == 0
    assert (await client.post("/api/pod/start", json={}, headers={"X-CholeScope-Request": "1"})).status == 202
    assert manager.starts == 1
    assert (await client.post("/api/pod/stop", json={}, headers={"X-CholeScope-Request": "1"})).status == 202
    assert manager.stops == 1


async def test_fixed_host_blocks_rebinding_even_without_origin(aiohttp_client):
    client = await aiohttp_client(create_app(Manager(), port=8792))
    assert (await client.get("/", headers={"Host": "evil.example:8792"})).status == 403
    assert (await client.get("/", headers={"Host": "127.0.0.1:8792"})).status == 200


async def test_proxy_relays_socket_and_uses_backend_origin(aiohttp_client):
    backend = web.Application()
    origins = []
    async def echo(request):
        origins.append(request.headers.get("Origin"))
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for message in socket:
            if message.type == WSMsgType.TEXT:
                await socket.send_str(message.data)
        return socket
    async def status(request):
        assert "Authorization" not in request.headers
        return web.json_response({"backend_ready": True})
    backend.router.add_get("/ws", echo)
    backend.router.add_get("/api/status", status)
    remote = await aiohttp_client(backend)
    address = str(remote.make_url("/")).rstrip("/")
    client = await aiohttp_client(create_app(Manager("ready"), backend_url=address, port=0))
    assert (await (await client.get("/api/status", headers={"Authorization": "Bearer browser-value"})).json())["backend_ready"]
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "input", "keys": ["w"]})
    assert (await ws.receive_json())["keys"] == ["w"]
    assert origins == [address]
    await ws.close()
    assert (await client.get("/api/private-key")).status == 404


async def exclusive_proxy(aiohttp_client):
    """An idle backend that only releases its operator after socket teardown."""
    backend = web.Application()
    state = {"occupied": False, "attempts": 0}
    disconnected = asyncio.Queue()

    async def operator(request):
        state["attempts"] += 1
        if state["occupied"]:
            raise web.HTTPConflict()
        state["occupied"] = True
        socket = web.WebSocketResponse()
        try:
            await socket.prepare(request)
            await socket.send_json({"backend_ready": True})
            async for _ in socket:
                pass
        finally:
            state["occupied"] = False
            disconnected.put_nowait(True)
        return socket

    backend.router.add_get("/ws", operator)
    remote = await aiohttp_client(backend)
    manager = Manager("ready")
    client = await aiohttp_client(create_app(manager, backend_url=str(remote.make_url("/")), port=0))
    return client, manager, state, disconnected


@pytest.mark.parametrize("abrupt", [False, True])
async def test_browser_disconnect_releases_upstream_operator(aiohttp_client, abrupt):
    client, manager, state, disconnected = await exclusive_proxy(aiohttp_client)
    ws = await client.ws_connect("/ws")
    assert (await ws.receive_json())["backend_ready"]
    if abrupt:
        ws._response.connection.transport.abort()
    await ws.close()
    await asyncio.wait_for(disconnected.get(), timeout=2)
    replacement = await client.ws_connect("/ws")
    assert (await replacement.receive_json())["backend_ready"]
    await replacement.close()
    assert state["attempts"] == 2
    assert manager.starts == manager.stops == 0


async def test_takeover_releases_operator_and_fences_old_tabs(aiohttp_client):
    client, manager, state, disconnected = await exclusive_proxy(aiohttp_client)
    assert await (await client.get("/api/control/status")).json() == {
        "supported": True, "occupied": False, "requires_claim": False}
    old = await client.ws_connect("/ws")
    await old.receive_json()
    assert (await (await client.get("/api/control/status")).json())["occupied"]
    old_closed = asyncio.create_task(old.receive())
    response = await client.post("/api/control/takeover", json={}, headers={"X-CholeScope-Request": "1"})
    assert response.status == 200
    token = (await response.json())["control_token"]
    assert len(token) >= 32
    assert (await asyncio.wait_for(old_closed, 2)).type == WSMsgType.CLOSE
    await asyncio.wait_for(disconnected.get(), 2)
    status = await (await client.get("/api/control/status")).json()
    assert status == {"supported": True, "occupied": False, "requires_claim": True}
    assert token not in str(status)
    for path in ("/ws", "/ws?control_token=obsolete"):
        with pytest.raises(WSServerHandshakeError) as stale:
            await client.ws_connect(path)
        assert stale.value.status == 409
    assert state["attempts"] == 1
    current = await client.ws_connect("/ws", params={"control_token": token})
    await current.receive_json()
    status = await (await client.get("/api/control/status", params={"control_token": token})).json()
    assert status == {"supported": True, "occupied": True, "requires_claim": False}

    current_closed = asyncio.create_task(current.receive())
    response = await client.post("/api/control/takeover", json={}, headers={"X-CholeScope-Request": "1"})
    new_token = (await response.json())["control_token"]
    assert new_token != token
    await asyncio.wait_for(current_closed, 2)
    with pytest.raises(WSServerHandshakeError) as superseded:
        await client.ws_connect("/ws", params={"control_token": token})
    assert superseded.value.status == 409
    assert state["attempts"] == 2
    assert manager.starts == manager.stops == 0


async def test_takeover_invalidates_upstream_handshake_in_flight(aiohttp_client):
    entered = asyncio.Event()
    proceed = asyncio.Event()
    disconnected = asyncio.Event()
    backend = web.Application()

    async def delayed_operator(request):
        entered.set()
        await proceed.wait()
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        try:
            async for _ in socket:
                pass
        finally:
            disconnected.set()
        return socket

    backend.router.add_get("/ws", delayed_operator)
    remote = await aiohttp_client(backend)
    client = await aiohttp_client(create_app(Manager("ready"), backend_url=str(remote.make_url("/")), port=0))
    stale = asyncio.create_task(client.ws_connect("/ws"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert (await (await client.get("/api/control/status")).json())["occupied"]
        response = await client.post("/api/control/takeover", json={}, headers={"X-CholeScope-Request": "1"})
        assert response.status == 200
    finally:
        proceed.set()
    with pytest.raises(WSServerHandshakeError) as invalidated:
        await asyncio.wait_for(stale, 2)
    assert invalidated.value.status == 409
    await asyncio.wait_for(disconnected.wait(), 2)
    assert not (await (await client.get("/api/control/status")).json())["occupied"]


async def test_control_takeover_uses_local_mutation_guards(aiohttp_client):
    client = await aiohttp_client(create_app(Manager(), port=0))
    for headers, payload in [({}, {}), ({"X-CholeScope-Request": "1", "Origin": "http://evil.example"}, {}),
                             ({"X-CholeScope-Request": "1", "Sec-Fetch-Site": "cross-site"}, {}),
                             ({"X-CholeScope-Request": "1", "Host": "evil.example"}, {}),
                             ({"X-CholeScope-Request": "1"}, {"control_token": "chosen"})]:
        response = await client.post("/api/control/takeover", json=payload, headers=headers)
        assert response.status in {400, 403}
    assert (await (await client.get("/api/control/status")).json())["requires_claim"] is False
    assert (await client.get("/api/control/status", headers={"Origin": "http://evil.example"})).status == 403
    assert (await client.post("/api/control/takeover", json={}, headers={"X-CholeScope-Request": "1"})).status == 200


async def test_local_recording_path_is_constrained(aiohttp_client, tmp_path):
    run = tmp_path / ("a" * 32)
    run.mkdir()
    (run / "actions.json").write_text('{"recording":true}')
    client = await aiohttp_client(create_app(Manager(), output_dir=tmp_path, port=0))
    assert (await (await client.get(f"/api/runs/{run.name}/actions")).json())["recording"]
    assert (await client.get(f"/api/runs/{run.name}/secrets")).status == 404


def test_external_backend_is_rejected():
    with pytest.raises(ValueError):
        create_app(Manager(), backend_url="http://evil.example")
