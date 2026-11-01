"""Offline lifecycle and spending tests. No cloud or SSH calls are made."""
import asyncio
import contextlib
import copy
import io
import json
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pod_manager as manager
import runpod_budget as guard

NOW = guard.timestamp("2026-10-23T15:00:00Z")


class FakeAPI:
    def __init__(self):
        self.pod = {"id": "pod-123", "name": "demo-test", "createdAt": "2026-10-23T09:00:00Z",
                    "status": "EXITED", "cost": 2.49, "locked": False, "actions": ["start", "terminate"]}
        self.calls = []
        self.fail_start = False

    def get(self, pod_id):
        self.calls.append(("get", pod_id))
        return copy.deepcopy(self.pod)

    def request(self, method, pod_id, action):
        self.calls.append((method, pod_id, action))
        if action == "start":
            self.pod["status"] = "RUNNING"
            self.pod["actions"] = ["stop", "terminate"]
            if self.fail_start:
                raise RuntimeError("simulated response loss")
        elif action == "stop":
            self.pod["status"] = "EXITED"
            self.pod["actions"] = ["start", "terminate"]
        else:
            raise AssertionError("Forbidden pod action")
        return copy.deepcopy(self.pod)


class OfflineManager(manager.PodManager):
    async def _arm_local(self, lease_path):
        self.arm_count = getattr(self, "arm_count", 0) + 1
        self.armed_lease = json.loads(Path(lease_path).read_text())
        assert not any(c[-1] == "start" for c in self.api.calls if len(c) == 3)

    async def _attach_worker(self):
        self._state, self._message = "ready", "Demo ready."

    async def _flush_app(self):
        self.flushed = True


class PodManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.api = FakeAPI()
        self.now = NOW
        self.config = {
            "pod_id": "pod-123", "pod_name": "demo-test", "created_at": "2026-10-23T09:00:00Z",
            "key_file": str(self.root / "api-key"), "ssh_key": str(self.root / "ssh-key"),
            "known_hosts": str(self.root / "known_hosts"), "state_dir": str(self.root / "state"),
            "initial_spend_usd": 8.2, "initial_spend_epoch": NOW - 3600,
        }
        for name in ("api-key", "ssh-key"):
            path = self.root / name
            path.write_text("not-a-real-key")
            path.chmod(0o600)
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps(self.config))
        self.instance = self.create()

    def create(self, cls=OfflineManager):
        return cls(self.path, api=self.api, clock=lambda: self.now)

    async def asyncTearDown(self):
        await self.instance.close()
        self.directory.cleanup()

    async def finish_start(self):
        await self.instance._start_task

    async def test_status_is_read_only_and_accounts_paused_storage(self):
        result = await self.instance.status()
        self.assertTrue(result["can_start"])
        self.assertEqual(result["state"], "stopped")
        self.assertAlmostEqual(result["conservative_spend_usd"], 8.25)
        self.assertEqual(self.api.calls, [("get", "pod-123")])
        self.assertNotIn("pod_id", result)
        self.assertNotIn("key", json.dumps(result))

    async def test_start_is_idempotent_and_guard_precedes_paid_request(self):
        first, second = await asyncio.gather(self.instance.start(), self.instance.start())
        await self.finish_start()
        calls = [c for c in self.api.calls if len(c) == 3]
        self.assertEqual(calls, [("POST", "pod-123", "start")])
        self.assertEqual(self.instance.arm_count, 1)
        self.assertEqual(self.instance._state, "ready")
        self.assertEqual(self.instance.armed_lease["action"], "stop")
        self.assertEqual(self.instance.armed_lease["deadline_epoch"], NOW + 3480)
        self.assertAlmostEqual(self.instance.ledger["spent_usd"], 11.25)

    async def test_stop_never_deletes_or_refunds_paid_session(self):
        await self.instance.start()
        await self.finish_start()
        await self.instance.stop()
        await self.instance._stop_task
        self.assertEqual(self.instance._state, "stopped")
        self.assertTrue(self.instance.flushed)
        self.assertAlmostEqual(self.instance.ledger["spent_usd"], 11.25)
        self.assertEqual([c[-1] for c in self.api.calls if len(c) == 3], ["start", "stop"])

    async def test_lost_start_response_reconciles_and_stops(self):
        self.api.fail_start = True
        await self.instance.start()
        await self.finish_start()
        self.assertEqual(self.instance._state, "stopped")
        self.assertEqual([c[-1] for c in self.api.calls if len(c) == 3], ["start", "stop"])
        self.assertAlmostEqual(self.instance.ledger["spent_usd"], 11.25)

    async def test_budget_and_identity_block_start(self):
        self.instance.ledger["spent_usd"] = 73
        result = await self.instance.status()
        self.assertFalse(result["can_start"])
        with self.assertRaises(manager.PodError):
            await self.instance.start()
        self.assertFalse(any(len(c) == 3 for c in self.api.calls))
        self.instance.ledger["spent_usd"] = 8.2
        self.api.pod["name"] = "someone-elses-pod"
        self.now += 9
        result = await self.instance.status()
        self.assertEqual(result["state"], "error")
        self.assertFalse(result["can_start"])
        await self.instance.stop()
        await self.instance._stop_task
        self.assertFalse(any(len(c) == 3 for c in self.api.calls))

    async def test_locked_or_expensive_pod_cannot_start(self):
        self.api.pod["locked"] = True
        with self.assertRaises(manager.PodError):
            await self.instance.start()
        self.api.pod["locked"] = False
        self.api.pod["cost"] = 3.01
        self.now += 9
        with self.assertRaises(manager.PodError):
            await self.instance.start()
        self.assertFalse(any(c[-1] == "start" for c in self.api.calls if len(c) == 3))

    async def test_ledger_survives_controller_restart(self):
        await self.instance.start()
        await self.finish_start()
        lease_deadline = self.instance._lease["deadline_epoch"]
        await self.instance.close()
        self.api.calls.clear()
        self.instance = self.create()
        await self.instance.status()
        await self.finish_start()
        self.assertAlmostEqual(self.instance.ledger["spent_usd"], 11.25)
        self.assertEqual(self.instance._lease["deadline_epoch"], lease_deadline)
        self.assertFalse(any(len(c) == 3 for c in self.api.calls))

    async def test_running_without_lease_is_stopped_fail_closed(self):
        self.api.pod["status"] = "RUNNING"
        self.api.pod["actions"] = ["stop", "terminate"]
        result = await self.instance.status()
        self.assertEqual(result["state"], "stopping")
        await self.instance._stop_task
        self.assertEqual(self.instance._state, "stopped")
        self.assertEqual([c[-1] for c in self.api.calls if len(c) == 3], ["stop"])

    async def test_second_controller_cannot_renew_or_reset_ledger(self):
        with self.assertRaisesRegex(ValueError, "already running"):
            self.create()

    async def test_stop_interrupts_model_loading(self):
        gate = asyncio.Event()
        async def loading():
            self.instance._state = "loading"
            await gate.wait()
        self.instance._attach_worker = loading
        await self.instance.start()
        for _ in range(10):
            await asyncio.sleep(.01)
            if self.instance._state == "loading":
                break
        await self.instance.stop()
        await asyncio.wait_for(self.instance._stop_task, 1)
        self.assertEqual(self.instance._state, "stopped")
        self.assertTrue(self.instance._start_task.done())

    async def test_unconfirmed_dispatched_start_cannot_be_rebooked(self):
        await self.instance.status()
        lease = guard.build_session_lease(**self.instance.identity, hourly_rate=3,
                    session_start_epoch=NOW, minutes=60, spent=8.25)
        path = self.root / "state/session-uncertain.json"
        manager.private_json(path, lease)
        self.instance.ledger.update(phase="requested", lease_file=str(path), session_id="uncertain", spent_usd=11.25)
        self.instance._save()
        await self.instance.close()
        self.instance = self.create()
        result = await self.instance.status()
        self.assertEqual(result["state"], "error")
        self.assertFalse(result["can_start"])
        self.assertIn("unconfirmed", result["message"])
        self.assertEqual(self.instance.arm_count, 1)
        with self.assertRaises(manager.PodError):
            await self.instance.start()
        self.assertFalse(any(len(c) == 3 for c in self.api.calls))
        # A queued request becoming active must honor an earlier Stop click.
        await self.instance.stop()
        await self.instance._stop_task
        self.api.pod["status"] = "RUNNING"
        self.api.pod["actions"] = ["stop", "terminate"]
        self.now += 9
        await self.instance.status()
        await self.instance._stop_task
        self.assertEqual(self.instance._state, "stopped")
        self.assertEqual([c[-1] for c in self.api.calls if len(c) == 3], ["stop"])

    async def test_paused_storage_after_autostop_survives_closed_controller(self):
        await self.instance.start()
        await self.finish_start()
        await self.instance.close()
        self.api.pod["status"] = "EXITED"
        self.api.pod["actions"] = ["start", "terminate"]
        self.now += 3 * 3600
        self.instance = self.create()
        result = await self.instance.status()
        self.assertTrue(result["can_start"])
        self.assertAlmostEqual(self.instance.ledger["spent_usd"], 11.35)

    async def test_refresh_cleanup_disables_new_start(self):
        await self.instance.status()
        async with self.instance._refresh_lock:
            self.assertFalse(self.instance._public()["can_start"])
            task = asyncio.create_task(self.instance.start())
            await asyncio.sleep(0)
            self.assertFalse(task.done())
        await task
        await self.finish_start()
        self.assertEqual([c[-1] for c in self.api.calls if len(c) == 3], ["start"])

    async def test_known_provider_rejection_records_status_and_unlocks_after_stop(self):
        def reject(method, pod_id, action):
            self.api.calls.append((method, pod_id, action))
            raise RuntimeError("Runpod HTTP 422")
        self.api.request = reject
        await self.instance.start()
        await self.finish_start()
        self.assertEqual(self.instance._state, "stopped")
        self.assertEqual(self.instance.ledger["phase"], "stopped")
        self.assertIn("HTTP 422", self.instance._message)
        self.assertTrue(self.instance._public()["can_start"])
        receipt = json.loads((self.root / "state/last-start-error.json").read_text())
        self.assertEqual(receipt["http_status"], 422)
        self.assertEqual(receipt["outcome"], "rejected")

    async def test_capacity_notice_persists_until_next_start(self):
        def reject(*args):
            raise guard.RunpodHTTPError(400, guard.GPU_UNAVAILABLE_MESSAGE)
        self.api.request = reject
        await self.instance.start()
        await self.finish_start()
        self.assertEqual(self.instance._public()["last_error"], guard.GPU_UNAVAILABLE_MESSAGE)
        self.now += 9
        status = await self.instance.status()
        self.assertEqual(status["last_error"], guard.GPU_UNAVAILABLE_MESSAGE)
        self.assertTrue(status["can_start"])
        self.api.request = FakeAPI.request.__get__(self.api, FakeAPI)
        self.api.calls.clear()
        await self.instance.start()
        self.assertIsNone(self.instance._public()["last_error"])
        await self.finish_start()
        self.assertIsNone(self.instance._public()["last_error"])

    async def test_action_errors_distinguish_rejected_from_unknown_without_secrets(self):
        for error, status, rejected in [
            ("Runpod HTTP 402", 402, True), ("Runpod HTTP 429", 429, True),
            ("Runpod HTTP 408", 408, False), ("Runpod HTTP 500", 500, False),
            ("socket failed not-a-real-key", None, False),
        ]:
            def fail(*args):
                raise RuntimeError(error)
            self.api.request = fail
            with self.assertRaises(manager.PodActionError) as raised:
                await self.instance._action("start")
            self.assertEqual(raised.exception.status, status)
            self.assertEqual(raised.exception.definitely_rejected, rejected)
            receipt = (self.root / "state/last-start-error.json").read_text()
            self.assertNotIn("not-a-real-key", receipt)
            self.assertNotIn("socket failed", receipt)

    async def test_remote_guard_requires_explicit_opt_in_and_payload_has_no_default_key(self):
        self.instance._lease = guard.build_session_lease(**self.instance.identity, hourly_rate=3,
                    session_start_epoch=NOW, minutes=60, spent=8.25)
        self.instance.ledger["session_id"] = "test-session"
        payloads = []
        async def ssh(command, *, data, **kwargs):
            arguments = shlex.split(command)
            self.assertEqual(arguments[:2], ["python3", "-c"])
            compile(arguments[2], "remote-startup", "exec")
            payloads.append(json.loads(data))
            return b"configured\n"
        self.instance._ssh = ssh
        self.assertFalse(self.instance.remote_guard)
        await self.instance._arm_remote_and_start_app()
        self.assertNotIn("key", payloads[-1])
        self.assertFalse(payloads[-1]["remote_guard"])
        self.assertFalse(self.instance._public()["remote_guard_enabled"])
        self.instance.remote_guard = True
        await self.instance._arm_remote_and_start_app()
        self.assertTrue(payloads[-1]["remote_guard"])
        self.assertEqual(payloads[-1]["key"], "not-a-real-key")
        self.assertTrue(self.instance._public()["remote_guard_enabled"])

    async def pod_application_command(self):
        """Run the remote startup code against a local stand-in for the pod."""
        self.instance._lease = guard.build_session_lease(**self.instance.identity, hourly_rate=3,
                    session_start_epoch=NOW, minutes=60, spent=8.25)
        self.instance.ledger["session_id"] = "test-session"
        sent = {}
        async def ssh(command, *, data, **kwargs):
            sent["code"], sent["payload"] = shlex.split(command)[2], json.loads(data)
            return b"configured\n"
        self.instance._ssh = ssh
        await self.instance._arm_remote_and_start_app()
        assets = self.root / "assets"
        python = assets / "source/cosmos3-h-dreams/.venv/bin/python"
        python.parent.mkdir(parents=True)
        python.touch()
        checkpoint = assets / "model.safetensors"
        checkpoint.touch()
        files = {"checkpoints/exp15/c3-h-d_2-step_model_bf16.safetensors": {"path": str(checkpoint)}}
        (assets / "asset-manifest.json").write_text(json.dumps(
            {"repositories": {"nvidia/Cosmos3-H-Dreams": {"files": files}}}))
        payload = {**sent["payload"], "repo": str(manager.REPO), "assets": str(assets),
                   "private": str(self.root / "private")}
        started = []
        def popen(command, **kwargs):
            started.append(command)
            kwargs["stdout"].close()
            return SimpleNamespace(pid=4242)
        with patch("subprocess.Popen", popen), patch("sys.stdin", io.StringIO(json.dumps(payload))), \
                contextlib.redirect_stdout(io.StringIO()):
            exec(sent["code"], {})
        self.assertEqual(len(started), 1)
        return started[0]

    async def test_pod_application_keeps_the_preset_chunk_by_default(self):
        self.assertNotIn("--chunk-frames", await self.pod_application_command())

    async def test_four_frame_config_reaches_the_pod_application(self):
        await self.instance.close()
        self.config["chunk_frames"] = 4
        self.path.write_text(json.dumps(self.config))
        self.instance = self.create()
        command = await self.pod_application_command()
        self.assertIn("--chunk-frames", command)
        self.assertEqual(command[command.index("--chunk-frames") + 1], "4")

    async def test_unmeasured_chunk_frames_config_is_rejected(self):
        await self.instance.close()
        for value in (6, True, "4"):
            with self.subTest(chunk_frames=value):
                self.path.write_text(json.dumps({**self.config, "chunk_frames": value}))
                with self.assertRaisesRegex(ValueError, "chunk_frames"):
                    self.create()

    async def test_preflight_names_missing_cache_component_and_saves_receipt(self):
        checks = {"repo": {"directory": True}, "assets": {"directory": True},
                  "manifest": {"file": True}, "runtime": {"file": False, "executable": False,
                  "symlink": True, "link_target": "/missing/python"}, "checkpoint": {"file": True},
                  "ffmpeg": {"available": True, "source": "persistent"}}
        async def ssh(*args, **kwargs):
            self.assertEqual(kwargs["stage"], "preflight")
            self.assertNotIn("not-a-real-key", kwargs["data"].decode())
            return json.dumps(checks).encode()
        self.instance._ssh = ssh
        with self.assertRaisesRegex(manager.PodError, "Python runtime"):
            await self.instance._preflight()
        receipt = json.loads((self.root / "state/last-preflight.json").read_text())
        self.assertEqual(receipt["checks"], checks)

    async def test_preflight_reports_missing_encoder_without_loading_model(self):
        checks = {"repo": {"directory": True}, "assets": {"directory": True},
                  "manifest": {"file": True}, "runtime": {"file": True, "executable": True},
                  "checkpoint": {"file": True}, "ffmpeg": {"available": False, "source": None}}
        async def ssh(*args, **kwargs):
            return json.dumps(checks).encode()
        self.instance._ssh = ssh
        with self.assertRaisesRegex(manager.PodError, "missing FFmpeg encoder"):
            await self.instance._preflight()
        receipt = json.loads((self.root / "state/last-preflight.json").read_text())
        self.assertFalse(receipt["checks"]["ffmpeg"]["available"])
        self.assertFalse(any(len(c) == 3 for c in self.api.calls))

    def test_ssh_diagnostics_are_private_bounded_and_redacted(self):
        self.instance._ssh_diagnostic("preflight", 1, b"x" * 40000 + b"not-a-real-key")
        path = self.root / "state/last-ssh-error.json"
        receipt = json.loads(path.read_text())
        self.assertLessEqual(len(receipt["stderr"]), 32768)
        self.assertNotIn("not-a-real-key", path.read_text())
        self.assertIn("[redacted]", receipt["stderr"])
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_ssh_endpoint_cannot_inject_options(self):
        for endpoint in (
            {"host": "-oProxyCommand=bad", "username": "root", "port": 22},
            {"host": "good.host", "username": "root; bad", "port": 22},
            {"host": "good.host", "username": "root", "port": 0},
        ):
            with self.assertRaises(manager.PodError):
                self.instance._validated_endpoint(endpoint)


if __name__ == "__main__":
    unittest.main()
