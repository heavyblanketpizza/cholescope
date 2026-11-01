"""Private local lifecycle controller for a single preconfigured Runpod demo.

The browser receives only public status. Credentials, conservative spending
receipts, and stop-only session leases are stored outside the checkout. The
controller never creates, deletes, or changes the identity of a pod.
"""
from __future__ import annotations

import asyncio
import contextlib
import fcntl
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from urllib.request import urlopen

import runpod_budget as guard
from pod_replacement_controller import AllocationPending, ReplacementController, finish_adoption
from pod_replacement import ReplacementError

REPO = Path(__file__).resolve().parent.parent
ACTIVE = {"RUNNING", "STARTING", "PROVISIONING", "STOPPING", "RESTARTING"}


class PodError(RuntimeError):
    """A concise error safe to expose to the local UI."""


class PodActionError(PodError):
    def __init__(self, message, *, status=None, definitely_rejected=False):
        super().__init__(message)
        self.status = status
        self.definitely_rejected = definitely_rejected


def private_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def bounded_number(value, name, minimum, maximum):
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"Invalid {name}")
    return number


class PodManager:
    def __init__(self, config_path, *, api=None, clock=time.time):
        self.config_path = Path(config_path).expanduser().resolve()
        if self.config_path.is_relative_to(REPO):
            raise ValueError("Keep pod configuration outside the repository")
        self.config = json.loads(self.config_path.read_text())
        c = self.config
        self.identity = {k: c[k] for k in ("pod_id", "pod_name", "created_at")}
        if not guard.POD_ID.fullmatch(c["pod_id"]):
            raise ValueError("Invalid fixed pod ID")
        guard.timestamp(c["created_at"])
        if not c["pod_name"] or not isinstance(c["pod_name"], str):
            raise ValueError("An exact pod name is required")
        # Only the two measured chunk sizes may reach the GPU application.
        self.chunk_frames = c.get("chunk_frames", 8)
        if type(self.chunk_frames) is not int or self.chunk_frames not in (4, 8):
            raise ValueError("chunk_frames must be 4 or 8")
        self.state_dir = Path(c["state_dir"]).expanduser().resolve()
        if self.state_dir.is_relative_to(REPO):
            raise ValueError("Keep private pod state outside the repository")
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.state_dir, 0o700)
        self._lock_file = open(self.state_dir / "manager.lock", "a+")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.close()
            raise ValueError("A local pod controller is already running") from None
        finish_adoption(self.config_path)
        c = self.config = json.loads(self.config_path.read_text())
        self.identity = {k: c[k] for k in ("pod_id", "pod_name", "created_at")}
        self.key_file = Path(c["key_file"]).expanduser().resolve()
        self.ssh_key = Path(c["ssh_key"]).expanduser().resolve()
        self.known_hosts = Path(c["known_hosts"]).expanduser().resolve()
        for secret in (self.key_file, self.ssh_key):
            if secret.is_relative_to(REPO) or not secret.is_file():
                self._lock_file.close()
                raise ValueError("Private key files must exist outside the repository")
            if secret.stat().st_mode & 0o077:
                self._lock_file.close()
                raise ValueError("Private key files must have mode 0600")
        self.clock = clock
        self.api = api or guard.RunpodAPI(self.key_file.read_text().strip())
        self.remote_guard = c.get("remote_guard", False)
        if not isinstance(self.remote_guard, bool):
            raise ValueError("remote_guard must be an explicit boolean")
        self.minutes = int(bounded_number(c.get("session_minutes", 60), "session minutes", 3, 60))
        self.max_rate = bounded_number(c.get("max_rate", 3), "maximum rate", .01, 3)
        self.reserve = bounded_number(c.get("storage_reserve", 5), "storage reserve", 5, 79)
        self.storage_rate = bounded_number(c.get("paused_storage_rate", .05), "storage rate", 0, 1)
        self.port = int(c.get("backend_port", 8793))
        if not 1024 <= self.port <= 65535:
            raise ValueError("Invalid local backend port")
        self.backend_url = f"http://127.0.0.1:{self.port}"
        self.remote_repo = self._remote_path(c.get("remote_repo", "/workspace/cholescope"))
        self.remote_assets = self._remote_path(c.get("remote_assets", "/workspace/cholescope-assets"))
        self.remote_private = self._remote_path(c.get("remote_private", "/workspace/cholescope-private"))
        self.ledger_path = self.state_dir / "budget-ledger.json"
        if self.ledger_path.exists():
            self.ledger = json.loads(self.ledger_path.read_text())
            if self.ledger.get("identity") != self.identity:
                raise ValueError("Saved budget belongs to another pod")
            bounded_number(self.ledger["spent_usd"], "saved spend", 0, 1e6)
            bounded_number(self.ledger["accounted_at"], "saved time", 0, 1e12)
        else:
            self.ledger = {
                "schema_version": 1, "identity": self.identity,
                "spent_usd": bounded_number(c["initial_spend_usd"], "initial spend", 0, 80),
                "accounted_at": bounded_number(c["initial_spend_epoch"], "initial spend time", 0, 1e12),
                "phase": "stopped", "lease_file": None, "session_id": None,
            }
            private_json(self.ledger_path, self.ledger)
        self._state = "stopped"
        self._message = "Pod paused."
        self._rate = float(c.get("display_rate", 2.49))
        self._pod = None
        self._lease = None
        self._endpoint = None
        self._tunnel = None
        self._watchers = []
        self._sleep_inhibitors = []
        self._start_task = None
        self._stop_task = None
        self._cancel_start = asyncio.Event()
        self._mutation_lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        self._last_refresh = -math.inf
        self._initialized = False
        self._closed = False
        self.replacements = ReplacementController(self)

    @staticmethod
    def _remote_path(value):
        if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
            raise ValueError("Remote directories must be absolute paths")
        return value.rstrip("/")

    def _busy(self, task):
        return task is not None and not task.done()

    def _save(self):
        private_json(self.ledger_path, self.ledger)

    def _account_storage(self):
        now = self.clock()
        if self.ledger["phase"] == "stopped":
            elapsed = max(0, now - self.ledger["accounted_at"])
            self.ledger["spent_usd"] += elapsed * self.storage_rate / 3600
            self.ledger["accounted_at"] = max(now, self.ledger["accounted_at"])
            if elapsed:
                self._save()

    def _public(self):
        remaining = max(0.0, 80 - self.reserve - self.ledger["spent_usd"])
        busy = self._busy(self._start_task) or self._busy(self._stop_task)
        pod_stopped = self._pod is not None and self._pod.get("status") == "EXITED"
        allocation_pending = self.replacements.pending()
        can_start = (pod_stopped and not allocation_pending and not busy and not self._closed and not self._refresh_lock.locked()
                     and self._state == "stopped" and remaining > self.max_rate * self.minutes / 60
                     and not self._pod.get("locked") and self._rate <= self.max_rate
                     and "start" in self._pod.get("actions", []))
        display_state, message = self._state, self._message
        if pod_stopped and self._state == "stopped" and not busy:
            if remaining <= self.max_rate * self.minutes / 60:
                display_state, message = "error", "Insufficient demo budget for another session."
            elif self._rate > self.max_rate:
                display_state, message = "error", "GPU price exceeds the demo’s hourly limit."
            elif self._pod.get("locked"):
                display_state, message = "error", "Unlock this pod in RunPod before starting."
            elif "start" not in self._pod.get("actions", []):
                display_state, message = "error", "RunPod cannot start this pod right now."
        result = {
            "available": True, "state": display_state, "message": message,
            "last_error": self.ledger.get("last_error"),
            "remote_guard_enabled": self.remote_guard,
            "rate_usd_per_hour": self._rate, "session_limit_minutes": self.minutes,
            "remaining_budget_usd": round(remaining, 2), "reserved_storage_usd": self.reserve,
            "conservative_spend_usd": round(self.ledger["spent_usd"], 2),
            "can_start": can_start,
            "can_stop": not self._closed and not self._busy(self._stop_task) and (
                allocation_pending or self._busy(self._start_task) or self._state in {"starting", "loading", "ready"}
                or (self._pod is not None and self._pod.get("status") in ACTIVE)),
        }
        if self._lease and self.ledger["phase"] != "stopped":
            result["deadline_epoch"] = self._lease["deadline_epoch"]
        return result

    async def _get_pod(self):
        pod = await asyncio.to_thread(self.api.get, self.identity["pod_id"])
        if pod is None:
            raise PodError("The configured pod no longer exists.")
        try:
            guard.verify_pod(pod, self.identity)
        except (ValueError, KeyError, TypeError):
            raise PodError("Pod identity changed; controls are disabled.") from None
        self._pod = pod
        if self.ledger["phase"] == "requested" and pod.get("status") in ACTIVE:
            # Once an active transition has been observed, a subsequent EXITED
            # conclusively belongs to this session rather than a queued start.
            self.ledger["phase"] = "active"
            self._save()
        rate = float(pod.get("cost") or self._rate)
        if not math.isfinite(rate) or rate < 0:
            raise PodError("RunPod returned an invalid price.")
        self._rate = rate
        return pod

    async def _action(self, action):
        if action not in {"start", "stop"}:
            raise ValueError("Only start and stop actions are permitted")
        receipt = {"at_epoch": self.clock(), "action": action, "outcome": "dispatching",
                   "session_id": self.ledger.get("session_id"), "http_status": None}
        private_json(self.state_dir / "last-action.json", receipt)
        try:
            response = await asyncio.to_thread(self.api.request, "POST", self.identity["pod_id"], action)
            # Support older redacted API helpers as well as the typed client.
            if response is None:
                raise guard.RunpodHTTPError(404)
        except Exception as exc:
            # Prefer the typed helper; strict matching keeps compatibility with
            # older helpers without exposing arbitrary exception text.
            match = re.fullmatch(r"Runpod HTTP ([0-9]{3})", str(exc))
            status = exc.status if isinstance(exc, guard.RunpodHTTPError) else (int(match.group(1)) if match else None)
            public_message = exc.public_message if isinstance(exc, guard.RunpodHTTPError) else None
            rejected = status is not None and 400 <= status < 500 and status != 408
            receipt.update(outcome="rejected" if rejected else "unconfirmed",
                           http_status=status, error_type=type(exc).__name__)
            if public_message:
                receipt["public_message"] = public_message
            private_json(self.state_dir / "last-action.json", receipt)
            private_json(self.state_dir / "last-action-error.json", receipt)
            private_json(self.state_dir / f"last-{action}-error.json", receipt)
            verb = "startup" if action == "start" else "pause"
            message = public_message or (f"RunPod rejected {verb} (HTTP {status})." if rejected
                       else f"RunPod did not confirm {verb}" + (f" (HTTP {status})." if status else "."))
            self.ledger["last_error"] = message
            self._save()
            raise PodActionError(message, status=status, definitely_rejected=rejected) from None
        receipt.update(outcome="accepted", at_epoch=self.clock())
        if isinstance(response, dict) and response.get("status") in ACTIVE | {"EXITED", "TERMINATED", "ERROR"}:
            receipt["pod_status"] = response["status"]
        private_json(self.state_dir / "last-action.json", receipt)
        private_json(self.state_dir / f"last-{action}-action.json", receipt)
        return response

    async def status(self):
        if self._closed:
            return self._public()
        if self.clock() - self._last_refresh < 8 or self._refresh_lock.locked():
            return self._public()
        async with self._refresh_lock:
            self._last_refresh = self.clock()
            try:
                pod = await self._get_pod()
                if self.replacements.pending():
                    self._initialized = True
                    if not self._busy(self._start_task) and not self._busy(self._stop_task):
                        self._start_task = asyncio.create_task(self.replacements.recover_worker())
                elif not self._initialized:
                    self._initialized = True
                    await self._recover(pod)
                elif not self._busy(self._start_task) and not self._busy(self._stop_task):
                    if pod["status"] == "EXITED":
                        if self.ledger["phase"] == "requested":
                            await self._recover(pod)
                        else:
                            await self._mark_stopped()
                    elif pod["status"] in ACTIVE and self._state == "stopped":
                        # An out-of-band start has no new spending authorization.
                        self._state, self._message = "stopping", "Pausing an unguarded session."
                        self._stop_task = asyncio.create_task(self._stop_worker())
                    elif pod["status"] in ACTIVE and (self._state == "error" or (
                            self._state == "ready" and not await self._backend_ready())):
                        # Reconnect only the existing lease; never create paid
                        # time or extend the deadline to repair a dropped tunnel.
                        await self._recover(pod)
                self._account_storage()
            except (RuntimeError, ValueError, KeyError, OSError):
                if not self._busy(self._start_task) and not self._busy(self._stop_task):
                    self._state, self._message = "error", "Cannot check RunPod. Retry shortly."
        return self._public()

    async def _recover(self, pod):
        if pod["status"] == "EXITED":
            if self.ledger["phase"] == "requested":
                path = Path(self.ledger["lease_file"])
                if path.parent.resolve() != self.state_dir:
                    raise PodError("Saved startup needs operator attention.")
                lease = guard.validate_lease(json.loads(path.read_text()))
                guard.verify_pod(pod, lease)
                if lease["schema_version"] != 2 or lease["action"] != "stop":
                    raise PodError("Saved startup needs operator attention.")
                if lease["deadline_epoch"] > self.clock():
                    self._lease = lease
                    if not any(p.poll() is None for p in self._watchers):
                        await self._arm_local(path)
                    self._state, self._message = "error", "Previous start is unconfirmed. Its budget guard remains active."
                    return
            await self._mark_stopped()
            return
        if pod["status"] not in ACTIVE:
            self._state, self._message = "error", "Pod needs attention in RunPod."
            return
        if self.ledger.get("stop_requested"):
            self._state, self._message = "stopping", "Completing the previous pause request."
            self._stop_task = asyncio.create_task(self._stop_worker())
            return
        try:
            path = Path(self.ledger["lease_file"])
            if path.parent.resolve() != self.state_dir:
                raise ValueError("Invalid lease path")
            lease = guard.validate_lease(json.loads(path.read_text()))
            guard.verify_pod(pod, lease)
            if lease["schema_version"] != 2 or lease["action"] != "stop" or lease["deadline_epoch"] <= self.clock():
                raise ValueError("No live stop-only lease")
            if self.ledger["phase"] not in {"armed", "requested", "active"}:
                raise ValueError("No saved paid session")
            self._lease = lease
            await self._arm_local(path)
            self._state, self._message = "loading", "Reconnecting to the current session."
            self._start_task = asyncio.create_task(self._attach_worker())
        except (ValueError, KeyError, TypeError, OSError, RuntimeError):
            self._state, self._message = "stopping", "Pausing a session without a valid guard."
            self._stop_task = asyncio.create_task(self._stop_worker())

    async def start(self):
        await self.status()
        # A concurrent read may still be cleaning up an older stopped guard.
        # Wait for that reconciliation before accepting a new paid session.
        async with self._refresh_lock:
            pass
        if self._busy(self._start_task) or self._busy(self._stop_task):
            return self._public()
        if not self._public()["can_start"]:
            raise PodError("Pod cannot start yet. Check its status and remaining budget.")
        self._cancel_start.clear()
        self.ledger["last_error"] = None
        self._save()
        self._state, self._message = "starting", "Starting GPU…"
        self._start_task = asyncio.create_task(self._start_worker())
        return self._public()

    async def stop(self):
        self.replacements.cancel()
        self.ledger["stop_requested"] = True
        self._save()
        if self._busy(self._stop_task):
            return self._public()
        self._cancel_start.set()
        self._state, self._message = "stopping", "Pausing GPU…"
        self._stop_task = asyncio.create_task(self._stop_worker())
        return self._public()

    def _check_start(self):
        if self._cancel_start.is_set() or self._closed:
            raise asyncio.CancelledError
        if not self._lease or self.clock() >= self._lease["deadline_epoch"]:
            raise PodError("The demo session expired before it became ready.")

    async def _start_worker(self):
        try:
            async with self._mutation_lock:
                pod = await self._get_pod()
                if self._cancel_start.is_set():
                    return
                if pod["status"] != "EXITED" or pod.get("locked") or "start" not in pod.get("actions", []):
                    raise PodError("The pod is no longer available to start.")
                if self._rate > self.max_rate:
                    raise PodError("The current GPU price exceeds this demo’s rate limit.")
                self._account_storage()
                start = max(self.clock(), self.ledger["accounted_at"])
                if start > self.clock() + 5:
                    raise PodError("The system clock changed. Check it before starting.")
                self._lease = guard.build_session_lease(
                    **self.identity, hourly_rate=self.max_rate, session_start_epoch=start,
                    minutes=self.minutes, spent=self.ledger["spent_usd"],
                    reserve=self.reserve, maximum_hourly_rate=self.max_rate)
                session_id = uuid.uuid4().hex
                path = self.state_dir / f"session-{session_id}.json"
                private_json(path, self._lease)
                # Commit the full capped session before any paid mutation. Early
                # stops never refund this conservative ledger or renew a lease.
                self.ledger.update({"phase": "armed", "lease_file": str(path),
                                    "session_id": session_id, "accounted_at": start, "prepaid_until": start + self.minutes * 60,
                                    "stop_requested": False, "allocation_intent": None,
                                    "refund_eligible": True, "reservation_refunded": False,
                                    "session_charge_usd": self.max_rate * self.minutes / 60,
                                    "spent_usd": self.ledger["spent_usd"] + self.max_rate * self.minutes / 60})
                self._save()
                await self._arm_local(path)
                self._check_start()
                # Persist intent before the request; a crashed/lost response is
                # never permission to start again with a fresh budget window.
                self.ledger["phase"] = "requested"
                self.ledger["refund_eligible"] = False
                self._save()
                try:
                    response = await self._action("start")
                except PodActionError as exc:
                    if exc.definitely_rejected:
                        # A provider rejection is different from a lost response.
                        # Reconcile stopped state, then retire this unused guard.
                        self.ledger["phase"] = "armed"
                        self.ledger["refund_eligible"] = True
                        self._save()
                    if (exc.definitely_rejected and str(exc) == guard.GPU_UNAVAILABLE_MESSAGE
                            and self.replacements.enabled):
                        response = await self.replacements.replace()
                        if response is None:
                            return
                    else:
                        raise
                if isinstance(response, dict) and response.get("status") in ACTIVE:
                    guard.verify_pod(response, self.identity)
                    self.ledger["phase"] = "active"
                    self._save()
            await self._attach_worker()
        except asyncio.CancelledError:
            return
        except AllocationPending as exc:
            self._state, self._message = "error", str(exc)
            self.ledger["last_error"] = str(exc)
            self._save()
        except Exception as exc:
            message = str(exc) if isinstance(exc, (PodError, ReplacementError)) else "Could not start the demo."
            await self._failed_start(message)

    async def _attach_worker(self):
        try:
            end = min(self.clock() + 900, self._lease["deadline_epoch"])
            pod = None
            while self.clock() < end:
                self._check_start()
                pod = await self._get_pod()
                if pod.get("status") == "RUNNING" and pod.get("ssh", {}).get("direct"):
                    self._endpoint = self._validated_endpoint(pod["ssh"]["direct"])
                    try:
                        await self._ssh("true", timeout=20)
                        break
                    except PodError:
                        pass
                if pod.get("status") in {"ERROR", "TERMINATED"}:
                    raise PodError("RunPod could not start this GPU.")
                await asyncio.sleep(4)
            else:
                raise PodError("GPU startup took too long; the pod is being paused.")
            self._check_start()
            self._state, self._message = "loading", "Loading the surgical model…"
            await self._deploy()
            self._check_start()
            await self._arm_remote_and_start_app()
            self._check_start()
            await self._open_tunnel()
            while self.clock() < end:
                self._check_start()
                if self._tunnel.poll() is not None:
                    raise PodError("The GPU connection closed.")
                if await self._backend_ready():
                    self._state, self._message = "ready", "Demo ready."
                    return
                await asyncio.sleep(3)
            raise PodError("Model loading took too long; the pod is being paused.")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            message = str(exc) if isinstance(exc, PodError) else "Could not connect to the demo."
            await self._failed_start(message)

    async def _failed_start(self, message):
        if self._busy(self._stop_task) or self._cancel_start.is_set():
            return
        self.ledger["stop_requested"] = True
        self.ledger["last_error"] = message
        self._save()
        self._state, self._message = "stopping", message + " Pausing GPU…"
        await self._stop_worker(final_message=message + " Pod paused.")

    async def _arm_local(self, lease_path, *, identity=None):
        guarded_identity = identity or self.identity
        log_path = self.state_dir / (lease_path.stem + "-guard.log")
        log_offset = log_path.stat().st_size if log_path.exists() else 0
        with open(log_path, "ab", buffering=0) as stream:
            os.chmod(log_path, 0o600)
            process = subprocess.Popen(
                [sys.executable, str(REPO / "scripts/runpod_budget.py"), "watch",
                 "--lease", str(lease_path), "--pod-id", guarded_identity["pod_id"],
                 "--api-key-file", str(self.key_file), "--poll", "10"],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                start_new_session=True, close_fds=True)
        self._watchers.append(process)
        self.ledger.setdefault("guard_pids", []).append(process.pid)
        self._save()
        if sys.platform == "darwin":
            try:
                inhibitor = subprocess.Popen(
                    ["/usr/bin/caffeinate", "-i", "-w", str(process.pid)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
                self._sleep_inhibitors.append(inhibitor)
            except OSError:
                raise PodError("Could not keep the Mac awake for the local budget guard.") from None
        # A detached process and an armed receipt must exist before the start
        # request. Merely creating a lease file is not an active watchdog.
        for _ in range(30):
            if process.poll() is not None:
                raise PodError("The budget guard could not start.")
            if b'"event": "armed"' in log_path.read_bytes()[log_offset:]:
                return
            await asyncio.sleep(.1)
        raise PodError("The budget guard did not confirm it was armed.")

    @staticmethod
    def _validated_endpoint(endpoint):
        host, user, port = endpoint["host"], endpoint["username"], int(endpoint["port"])
        if (not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]{0,253}", host)
                or not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,127}", user)
                or not 1 <= port <= 65535):
            raise PodError("RunPod returned an invalid SSH endpoint.")
        return {"host": host, "username": user, "port": port}

    def _ssh_args(self):
        endpoint = self._endpoint
        if not endpoint:
            raise PodError("No GPU connection is available.")
        return ["ssh", "-i", str(self.ssh_key), "-o", "IdentitiesOnly=yes",
                "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={self.known_hosts}", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                "-p", str(endpoint["port"])]

    def _destination(self):
        return f"{self._endpoint['username']}@{self._endpoint['host']}"

    async def _ssh(self, command, *, data=None, timeout=60, stage="ssh"):
        # Never persist stdin: remote startup passes the API credential there.
        # Only bounded, redacted stderr is retained in this private state folder.
        process = await asyncio.create_subprocess_exec(
            *self._ssh_args(), self._destination(), command,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            output, errors = await asyncio.wait_for(process.communicate(data), timeout)
            if process.returncode:
                self._ssh_diagnostic(stage, process.returncode, errors)
                raise PodError("The GPU setup command failed. Check the private setup log.")
            return output
        except asyncio.TimeoutError:
            self._ssh_diagnostic(stage, None, b"SSH command timed out")
            raise PodError("The GPU connection timed out.") from None
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    def _ssh_diagnostic(self, stage, returncode, errors):
        text = errors[-32768:].decode("utf-8", errors="replace")
        # Defense in depth: an unexpected remote exception must not save the
        # credential even to the private diagnostic file.
        key = self.key_file.read_text().strip()
        if key:
            text = text.replace(key, "[redacted]")
        private_json(self.state_dir / "last-ssh-error.json", {
            "at_epoch": self.clock(), "stage": stage,
            "returncode": returncode, "stderr": text,
        })

    async def _preflight(self):
        code = r'''
import json,os,shutil,sys
from pathlib import Path
p=json.load(sys.stdin)
paths={'repo':Path(p['repo']),'assets':Path(p['assets']),
       'manifest':Path(p['assets'])/'asset-manifest.json',
       'runtime':Path(p['assets'])/'source/cosmos3-h-dreams/.venv/bin/python',
       'ffmpeg':Path(p['assets'])/'bin/ffmpeg'}
checks={}
def inspect(path):
 result={'path':str(path),'exists':path.exists(),'file':path.is_file(),'directory':path.is_dir(),
         'executable':os.access(path,os.X_OK),'symlink':path.is_symlink()}
 if path.is_symlink():result['link_target']=os.readlink(path)
 return result
for name,path in paths.items():checks[name]=inspect(path)
checks['ffmpeg']['available']=checks['ffmpeg']['file'] and checks['ffmpeg']['executable']
checks['ffmpeg']['source']='persistent' if checks['ffmpeg']['available'] else None
if not checks['ffmpeg']['available']:
 fallback=shutil.which('ffmpeg')
 if fallback:
  checks['ffmpeg']['system_fallback']=inspect(Path(fallback))
  checks['ffmpeg']['available']=checks['ffmpeg']['system_fallback']['file'] and checks['ffmpeg']['system_fallback']['executable']
  if checks['ffmpeg']['available']:checks['ffmpeg']['source']='system'
# A persistent mount and the container root can survive different lifecycle
# operations. Record mount evidence without guessing that a directory persists.
mounts=[]
try:
 for line in Path('/proc/self/mountinfo').read_text().splitlines():
  fields=line.split(); divider=fields.index('-')
  mount=fields[4].replace('\\040',' ')
  mounts.append({'mount_point':mount,'filesystem_type':fields[divider+1],
                 'read_only':'ro' in fields[5].split(',')})
 for name in ('repo','assets'):
  path=str(paths[name].resolve())
  candidates=[m for m in mounts if m['mount_point']=='/' or path==m['mount_point'] or path.startswith(m['mount_point'].rstrip('/')+'/')]
  if candidates:checks[name]['mount']=max(candidates,key=lambda m:len(m['mount_point']))
except (OSError,ValueError,IndexError):pass
try:
 manifest=json.loads(paths['manifest'].read_text())
 checkpoint=Path(manifest['repositories']['nvidia/Cosmos3-H-Dreams']['files']['checkpoints/exp15/c3-h-d_2-step_model_bf16.safetensors']['path'])
 checks['checkpoint']=inspect(checkpoint)
except (OSError,ValueError,KeyError,TypeError) as exc:
 checks['checkpoint']={'exists':False,'manifest_error':type(exc).__name__}
print(json.dumps(checks))
'''
        output = await self._ssh("python3 -c " + shlex.quote(code),
            data=json.dumps({"repo": self.remote_repo, "assets": self.remote_assets}).encode(),
            timeout=30, stage="preflight")
        try:
            checks = json.loads(output)
        except (ValueError, TypeError):
            raise PodError("GPU preflight returned an unreadable result.") from None
        private_json(self.state_dir / "last-preflight.json", {"at_epoch": self.clock(), "checks": checks})
        requirements = [("repo", "directory", "application source"),
                        ("assets", "directory", "cached assets"),
                        ("manifest", "file", "asset manifest"),
                        ("runtime", "file", "Python runtime"),
                        ("runtime", "executable", "executable Python runtime"),
                        ("checkpoint", "file", "model checkpoint"),
                        ("ffmpeg", "available", "FFmpeg encoder")]
        missing = [label for name, field, label in requirements if not checks.get(name, {}).get(field)]
        if missing:
            raise PodError("GPU is missing " + ", ".join(missing) + ". Restore the cached setup before restarting.")
        return checks

    async def _deploy(self):
        await self._preflight()
        # Include source only: no .git, credentials, models, caches, or outputs.
        archive = io.BytesIO()
        def source_metadata(info):
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            return info
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            for base in (REPO / "src", REPO / "scripts"):
                for path in sorted(base.rglob("*")):
                    relative = path.relative_to(REPO)
                    if (not path.is_file() or path.is_symlink()
                            or any(part.startswith(".") or part == "__pycache__" for part in relative.parts)
                            or path.suffix in {".pyc", ".pyo"}):
                        continue
                    tar.add(path, arcname=str(relative), recursive=False, filter=source_metadata)
        command = (f"test -d {shlex.quote(self.remote_repo)} && "
                   f"test -f {shlex.quote(self.remote_assets + '/asset-manifest.json')} && "
                   f"test -x {shlex.quote(self.remote_assets + '/source/cosmos3-h-dreams/.venv/bin/python')} && "
                   f"tar --no-same-owner --no-same-permissions -xzf - -C {shlex.quote(self.remote_repo)}")
        await self._ssh(command, data=archive.getvalue(), timeout=90, stage="source-deploy")

    async def _arm_remote_and_start_app(self):
        payload = {"repo": self.remote_repo, "assets": self.remote_assets,
                   "private": self.remote_private, "lease": self._lease,
                   "remote_guard": self.remote_guard, "session_id": self.ledger["session_id"],
                   "chunk_frames": self.chunk_frames}
        # Remote credentials are opt-in. Without this explicit private setting,
        # no API key is read into the payload or written to the GPU host.
        if self.remote_guard:
            payload["key"] = self.key_file.read_text().strip()
        # An opted-in secret crosses authenticated SSH stdin, never a command
        # argument, browser field, log, or application environment.
        code = r'''
import json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin); root=Path(p['private']); root.mkdir(mode=0o700,parents=True,exist_ok=True)
os.chmod(root,0o700)
if p.get('remote_guard') is True:
 lease=root/('session-'+p['session_id']+'.json'); key=root/'demo-api-key'
 for path,data in [(lease,json.dumps(p['lease'])),(key,p['key'])]:
  fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
  with os.fdopen(fd,'w') as f:f.write(data)
  os.chmod(path,0o600)
 log=open(root/('session-'+p['session_id']+'-guard.log'),'ab',buffering=0)
 g=subprocess.Popen([sys.executable,p['repo']+'/scripts/runpod_budget.py','watch','--lease',str(lease),'--pod-id',p['lease']['pod_id'],'--api-key-file',str(key),'--poll','10'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 (root/'session-guard.pid').write_text(str(g.pid))
 time.sleep(.5)
 if g.poll() is not None:raise RuntimeError('guard exited')
# Reattachment preserves an existing application instead of loading a second model.
pidfile=root/'server.pid'; alive=False
if pidfile.exists():
 try:
  pid=int(pidfile.read_text()); cmd=Path('/proc/'+str(pid)+'/cmdline').read_bytes()
  argv=cmd.split(b'\x00'); alive=b'cholescope' in argv and b'-m' in argv and b'--port' in argv and argv[argv.index(b'--port')+1]==b'8790'
 except (OSError,ValueError,IndexError):pass
if not alive:
 assets=Path(p['assets']); upstream=assets/'source/cosmos3-h-dreams'
 sys.path.insert(0,p['repo']+'/scripts')
 from bootstrap_gpu import cache_environment
 manifest=json.loads((assets/'asset-manifest.json').read_text())
 checkpoint=Path(manifest['repositories']['nvidia/Cosmos3-H-Dreams']['files']['checkpoints/exp15/c3-h-d_2-step_model_bf16.safetensors']['path'])
 python=upstream/'.venv/bin/python'
 if not checkpoint.is_file() or not python.is_file():raise RuntimeError('missing cached model or environment')
 env={k:v for k,v in os.environ.items() if k!='RUNPOD_API_KEY'}
 env.update(cache_environment(assets))
 env['PATH']=str(assets/'bin')+os.pathsep+env.get('PATH','')
 env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
 env['PYTHONPATH']=p['repo']+'/src'+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
 out=open(root/'server.log','ab',buffering=0)
 command=[str(python),'-m','cholescope','--checkpoint',str(checkpoint),'--upstream-root',str(upstream),'--output-dir',str(assets/'outputs'),'--host','127.0.0.1','--port','8790']
 if p.get('chunk_frames',8)!=8:command+=['--chunk-frames',str(p['chunk_frames'])]
 app=subprocess.Popen(command,cwd=p['repo'],env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=out,start_new_session=True)
 pidfile.write_text(str(app.pid))
print('configured')
'''
        output = await self._ssh("python3 -c " + shlex.quote(code),
                                 data=json.dumps(payload).encode(), timeout=30)
        if output.strip() != b"configured":
            raise PodError("The GPU application did not confirm setup.")

    async def _open_tunnel(self):
        await self._close_tunnel()
        path = self.state_dir / "tunnel.log"
        with open(path, "ab", buffering=0) as stream:
            os.chmod(path, 0o600)
            self._tunnel = subprocess.Popen(
                self._ssh_args() + ["-o", "ExitOnForwardFailure=yes", "-N", "-L",
                                    f"127.0.0.1:{self.port}:127.0.0.1:8790", self._destination()],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, start_new_session=True)
        await asyncio.sleep(.4)
        if self._tunnel.poll() is not None:
            raise PodError("Could not open the local GPU connection.")

    async def _backend_ready(self):
        def read():
            try:
                with urlopen(self.backend_url + "/api/status", timeout=3) as response:
                    value = json.load(response)
                return bool(value.get("backend_ready"))
            except Exception:
                return False
        return await asyncio.to_thread(read)

    async def _flush_app(self):
        if not self._endpoint:
            return
        code = r'''
import os,signal,time
from pathlib import Path
root=Path(ROOT); path=root/'server.pid'
if path.exists():
 try:
  pid=int(path.read_text()); proc=Path('/proc/'+str(pid)); cmd=(proc/'cmdline').read_bytes()
  argv=cmd.split(b'\x00')
  if b'cholescope' in argv and b'-m' in argv and b'--port' in argv and argv[argv.index(b'--port')+1]==b'8790':
   os.kill(pid,signal.SIGTERM)
   end=time.monotonic()+25
   while proc.exists() and time.monotonic()<end:time.sleep(.25)
 except (OSError,ValueError,IndexError):pass
'''.replace("ROOT", repr(self.remote_private))
        await self._ssh("python3 -c " + shlex.quote(code), timeout=30)

    async def _stop_worker(self, final_message="Pod paused."):
        try:
            # Wait only for the short cloud mutation, never for model loading.
            async with self._mutation_lock:
                task = self._start_task
                if task and task is not asyncio.current_task() and not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                if self.replacements.pending():
                    self.replacements.cancel()
                    try:
                        await self.replacements.reconcile()
                    except AllocationPending as exc:
                        self._state, self._message = "error", str(exc)
                        return
                    if self.replacements.pending():
                        self._state, self._message = "error", "Allocation pause is not confirmed; its guard remains active."
                        return
                pod = await self._get_pod()
                if pod["status"] != "EXITED":
                    with contextlib.suppress(Exception):
                        await self._flush_app()
                    # Refresh identity immediately before the only mutation.
                    pod = await self._get_pod()
                    if pod["status"] != "EXITED":
                        if pod.get("locked"):
                            raise PodError("Unlock this pod in RunPod so it can be paused.")
                        await self._action("stop")
            end = self.clock() + 120
            while self.clock() < end:
                pod = await self._get_pod()
                if pod["status"] == "EXITED":
                    if self.ledger["phase"] == "requested":
                        await self._recover(pod)
                        if self.ledger["phase"] == "requested":
                            return
                    await self._mark_stopped()
                    self._message = final_message
                    return
                await asyncio.sleep(3)
            raise PodError("Pause is not confirmed. The budget guard will keep retrying.")
        except Exception as exc:
            self._state = "error"
            self._message = (str(exc) if isinstance(exc, PodError)
                             else "Pause is not confirmed. The budget guard remains active.")

    async def _mark_stopped(self):
        if self.replacements.pending():
            raise PodError("Allocation pause is not confirmed; its guard remains active.")
        await self._close_tunnel()
        if self.ledger.get("refund_eligible") and not self.ledger.get("reservation_refunded"):
            self.ledger["spent_usd"] = max(0, self.ledger["spent_usd"] - self.ledger.get("session_charge_usd", 0))
            self.ledger["reservation_refunded"] = True
            self._save()
        if self.ledger["phase"] != "stopped":
            # The full paid session was prepaid. Retained storage continues
            # after that window even when this controller was closed or asleep.
            prepaid_end = self.ledger.get("prepaid_until", self.ledger["accounted_at"] + self.minutes * 60)
            self.ledger["spent_usd"] += max(0, self.clock() - prepaid_end) * self.storage_rate / 3600
            self.ledger["phase"] = "stopped"
            self.ledger["accounted_at"] = max(self.clock(), self.ledger["accounted_at"])
            self._save()
        self._state, self._message = "stopped", "Pod paused."
        self._lease = None
        # A confirmed stopped pod makes these detached guards obsolete. Killing
        # only a verified own PID prevents a stale deadline stopping a new session.
        for pid in self.ledger.get("guard_pids", []):
            try:
                command = await asyncio.to_thread(subprocess.check_output,
                    ["ps", "-p", str(int(pid)), "-o", "command="], text=True, stderr=subprocess.DEVNULL)
                if (str(REPO / "scripts/runpod_budget.py") in command and " watch " in command
                        and str(self.state_dir) in command and self.identity["pod_id"] in command):
                    os.kill(int(pid), signal.SIGKILL)
            except (OSError, ValueError, subprocess.CalledProcessError):
                pass
        for process in self._watchers + self._sleep_inhibitors:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=.1)
        if self.ledger.get("guard_pids"):
            self.ledger["guard_pids"] = []
            self._save()
        await self.replacements.retire_guards()

    async def _close_tunnel(self):
        if self._tunnel and self._tunnel.poll() is None:
            self._tunnel.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(self._tunnel.wait), 3)
            except asyncio.TimeoutError:
                self._tunnel.kill()
                await asyncio.to_thread(self._tunnel.wait)
        self._tunnel = None

    async def close(self):
        """Close local transport, leaving detached budget guards in control."""
        self._closed = True
        for task in (self._start_task, self._stop_task):
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        await self._close_tunnel()
        self._lock_file.close()
