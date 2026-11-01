#!/usr/bin/env python3
"""Plan and enforce one explicit Runpod pod lease using REST v2.

The quote command is offline and never provisions resources. Run watch both on
the pod and on an independent controller. A sleeping, killed, or disconnected
watchdog cannot enforce a billing cap. API outages can delay shutdown. Stop
retains billable storage; terminate deletes the pod and its local disks. Neither
operation deletes separate network volumes. Export results before the deadline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler

API = "https://api.runpod.io/v2"
MAX_HOURS = 12.0
MAX_COMPUTE_USD = 75.0
MAX_TOTAL_USD = 80.0
RESERVE_USD = 5.0
SHUTDOWN_MARGIN_SECONDS = 120
MAX_SESSION_MINUTES = 60.0
MAX_SESSION_HOURLY_RATE_USD = 3.0
POD_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def number(value, name, minimum=0.0):
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return result.timestamp()


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def quote(hourly_rate, hours=12, spent=0, reserve=5):
    rate = number(hourly_rate, "hourly rate", 0.000001)
    duration = number(hours, "hours", 0.000001)
    used = number(spent, "previous project spending")
    storage = number(reserve, "storage reserve", RESERVE_USD)
    if duration > MAX_HOURS:
        raise ValueError("Runtime may not exceed 12 hours")
    ceiling = min(MAX_COMPUTE_USD, MAX_TOTAL_USD - storage)
    projected = used + rate * duration
    if projected >= ceiling:
        raise ValueError("Quote exceeds the budget; shorten runtime or choose a cheaper pod")
    if duration * 3600 <= SHUTDOWN_MARGIN_SECONDS:
        raise ValueError("Runtime must exceed the two-minute shutdown margin")
    return {
        "hourly_rate_usd": rate,
        "hours": duration,
        "previous_spend_usd": used,
        "storage_reserve_usd": storage,
        "compute_ceiling_usd": ceiling,
        "projected_compute_usd": round(projected, 6),
        "projected_total_usd": round(projected + storage, 6),
        "shutdown_margin_seconds": SHUTDOWN_MARGIN_SECONDS,
    }


def validate_lease(lease):
    version = lease.get("schema_version")
    if version not in {1, 2}:
        raise ValueError("Unsupported lease schema")
    if not POD_ID.fullmatch(str(lease.get("pod_id", ""))):
        raise ValueError("An explicit valid pod ID is required")
    if not isinstance(lease.get("pod_name"), str) or not lease["pod_name"].strip():
        raise ValueError("An exact pod name is required")
    if lease.get("action") not in {"stop", "terminate"}:
        raise ValueError("Lease action must be stop or terminate")
    if version == 2 and lease["action"] != "stop":
        raise ValueError("Resumable sessions may only stop the pod")
    budget = quote(lease["hourly_rate_usd"], lease["hours"],
                   lease["previous_spend_usd"], lease["storage_reserve_usd"])
    created = timestamp(lease["created_at"])
    start = created
    if version == 2:
        start = number(lease["session_start_epoch"], "session start", created)
        if budget["hours"] * 60 > MAX_SESSION_MINUTES:
            raise ValueError("Demo sessions may not exceed 60 minutes")
        maximum = number(lease["maximum_hourly_rate_usd"], "maximum hourly rate", 0.000001)
        if maximum > MAX_SESSION_HOURLY_RATE_USD:
            raise ValueError("Maximum session hourly rate exceeds the permitted rate")
        if budget["hourly_rate_usd"] > maximum:
            raise ValueError("Quoted hourly rate exceeds the permitted rate")
    expected = start + budget["hours"] * 3600 - SHUTDOWN_MARGIN_SECONDS
    deadline = number(lease["deadline_epoch"], "deadline")
    if deadline > expected + 0.001:
        raise ValueError("Lease deadline exceeds its original budget")
    if lease["compute_ceiling_usd"] != budget["compute_ceiling_usd"]:
        raise ValueError("Lease compute ceiling has been changed")
    return lease


def make_lease(pod_id, pod_name, created_at, hourly_rate, hours=12, spent=0,
               reserve=5, action="stop"):
    budget = quote(hourly_rate, hours, spent, reserve)
    created = timestamp(created_at)
    return validate_lease({
        "schema_version": 1, "pod_id": pod_id, "pod_name": pod_name,
        "created_at": utc(created), "action": action, **budget,
        "deadline_epoch": created + budget["hours"] * 3600 - SHUTDOWN_MARGIN_SECONDS,
    })


def build_session_lease(pod_id, pod_name, created_at, hourly_rate, *,
                        session_start_epoch, minutes=60, spent=0, reserve=5,
                        maximum_hourly_rate=3):
    """Arm a stop-only resume session before requesting a paid start.

    ``created_at`` remains the immutable identity of the existing pod, whereas
    ``session_start_epoch`` is recorded immediately before this resume. The
    caller must durably book earlier spending (including stopped storage), save
    this lease, and launch an independent watcher before its one start request.
    Never retry a start with a fresh lease until the earlier session's stopped
    state and spending have been reconciled. A resumed process must reuse its
    saved lease; a new process is not a new spending allowance.
    """
    budget = quote(hourly_rate, number(minutes, "minutes", 0.000001) / 60,
                   spent, reserve)
    start = number(session_start_epoch, "session start")
    return validate_lease({
        "schema_version": 2, "pod_id": pod_id, "pod_name": pod_name,
        "created_at": utc(timestamp(created_at)), "action": "stop", **budget,
        "session_start_epoch": start,
        "maximum_hourly_rate_usd": number(maximum_hourly_rate, "maximum hourly rate", 0.000001),
        "deadline_epoch": start + budget["hours"] * 3600 - SHUTDOWN_MARGIN_SECONDS,
    })


def verify_pod(pod, lease):
    if pod.get("id") != lease["pod_id"] or pod.get("name") != lease["pod_name"]:
        raise ValueError("Pod identity does not match the lease; refusing mutation")
    if abs(timestamp(pod["createdAt"]) - timestamp(lease["created_at"])) > 1:
        raise ValueError("Pod creation time does not match the lease")


def effective_deadline(lease, observed_rate):
    # Conservatively price the whole lease at the highest rate observed. A rate
    # decrease can never extend the saved deadline.
    rate = max(lease["hourly_rate_usd"], number(observed_rate, "observed rate"))
    remaining = lease["compute_ceiling_usd"] - lease["previous_spend_usd"]
    start = (lease["session_start_epoch"] if lease["schema_version"] == 2
             else timestamp(lease["created_at"]))
    price_deadline = start + remaining / rate * 3600
    return min(lease["deadline_epoch"], price_deadline - SHUTDOWN_MARGIN_SECONDS)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward the bearer token to another origin.


GPU_UNAVAILABLE_MESSAGE = "GPU unavailable on this host. Try again later."


class RunpodHTTPError(RuntimeError):
    """Typed, redacted provider failure; bodies and headers are never exposed."""
    def __init__(self, status, public_message=None):
        self.status = int(status)
        self.public_message = (public_message if public_message == GPU_UNAVAILABLE_MESSAGE else None)
        super().__init__(f"Runpod HTTP {self.status}")


def safe_http_message(status, body):
    if status != 400 or len(body) > 16384:
        return None
    try:
        payload = json.loads(body)
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if (isinstance(detail, str) and " ".join(detail.lower().split()).rstrip(".") ==
                "there are not enough free gpus on the host machine to start this pod"):
            return GPU_UNAVAILABLE_MESSAGE
    except (ValueError, TypeError, UnicodeDecodeError):
        pass
    return None


class RunpodAPI:
    def __init__(self, key, opener=None):
        if not key.strip():
            raise ValueError("Set RUNPOD_API_KEY before running watch")
        self.key = key
        self.opener = opener or build_opener(NoRedirect())

    def request(self, method, pod_id, action=None):
        if not POD_ID.fullmatch(pod_id):
            raise ValueError("Invalid pod ID")
        endpoint = f"{API}/pods/{pod_id}"
        payload = None
        if action is not None:
            endpoint += "/action"
            payload = json.dumps({"action": action}).encode()
        req = Request(endpoint, data=payload, method=method, headers={
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        })
        try:
            with self.opener.open(req, timeout=15) as response:
                body = response.read()
                return json.loads(body) if body else {}
        except HTTPError as exc:
            status = exc.code
            try:
                body = exc.read(16385)
            except Exception:
                body = b""
            finally:
                exc.close()
            if status == 404 and method == "GET":
                return None
            # Only a fixed allowlisted phrase may cross into public status.
            # Arbitrary response bodies and request headers never reach logs.
            raise RunpodHTTPError(status, safe_http_message(status, body)) from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("Runpod API is unreachable") from None

    def get(self, pod_id):
        return self.request("GET", pod_id)

    def shutdown(self, pod_id, action):
        if action == "terminate":
            return self.request("DELETE", pod_id)
        return self.request("POST", pod_id, action="stop")


def terminal(pod, action):
    if pod is None:
        return True
    # A terminate lease must delete even an already stopped pod to release its
    # local storage. A successful mutation alone is not confirmation.
    return action == "stop" and pod.get("status") in {"EXITED", "TERMINATED"}


def watch(lease, api, *, clock=time.time, monotonic=time.monotonic,
          sleep=time.sleep, log=print, stop_requested=lambda: False, poll=15):
    validate_lease(lease)
    poll = min(30.0, number(poll, "poll interval", 1))
    deadline = lease["deadline_epoch"]
    # A controller starts a schema-2 guard BEFORE its paid start request. The
    # initial EXITED state must not disarm that guard. Once active was observed,
    # a verified stop ends this session normally. A start that never happens
    # remains guarded until its original deadline (or a stop signal).
    awaiting_start = lease["schema_version"] == 2
    # Monotonic time prevents a backwards wall-clock correction extending a
    # running watchdog. The absolute timestamp survives a process restart.
    mono_deadline = monotonic() + max(0, deadline - clock())
    log(json.dumps({"event": "armed", "pod_id": lease["pod_id"],
                    "action": lease["action"], "deadline": utc(deadline)}), flush=True)
    while True:
        due = stop_requested() or clock() >= deadline or monotonic() >= mono_deadline
        try:
            pod = api.get(lease["pod_id"])
            if pod is not None:
                verify_pod(pod, lease)
            stopped = terminal(pod, lease["action"])
            if stopped and (pod is None or not awaiting_start or due):
                log(json.dumps({"event": "shutdown_verified", "pod_id": lease["pod_id"]}), flush=True)
                return
            if stopped:
                sleep(min(poll, max(1, deadline - clock())))
                continue
            awaiting_start = False
            if pod.get("locked"):
                log("WARNING: pod is locked; unlock it in Runpod so shutdown can succeed", flush=True)
                due = True
            try:
                observed_rate = number(pod["cost"], "observed hourly rate")
                if (lease["schema_version"] == 2 and
                        observed_rate > lease["maximum_hourly_rate_usd"]):
                    log("WARNING: observed hourly rate exceeds session limit; stopping", flush=True)
                    due = True
                deadline = min(deadline, effective_deadline(lease, observed_rate))
            except (KeyError, TypeError, ValueError):
                # Identity was checked above; an invalid price must not kill
                # the guard and leave an unbounded paid session running.
                log("WARNING: cannot verify hourly rate; stopping", flush=True)
                due = True
            mono_deadline = min(mono_deadline, monotonic() + max(0, deadline - clock()))
            due = due or clock() >= deadline or monotonic() >= mono_deadline
        except (RuntimeError, KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise  # Identity mismatch must never mutate an unintended pod.
            log(f"WARNING: inspection failed ({type(exc).__name__}); shutdown is not confirmed", flush=True)
        if due:
            try:
                api.shutdown(lease["pod_id"], lease["action"])
                log(json.dumps({"event": "shutdown_requested", "pod_id": lease["pod_id"]}), flush=True)
            except RuntimeError:
                log("WARNING: shutdown request failed; retrying; billing may continue", flush=True)
        delay = poll if due else min(poll, max(1, deadline - clock()))
        sleep(delay)


def write_lease(path, lease):
    validate_lease(lease)
    target = Path(path).expanduser().resolve()
    repo = Path(__file__).resolve().parent.parent
    if target.is_relative_to(repo):
        raise ValueError("Store private lease state outside the repository")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(lease, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("quote", "lease"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--hourly-rate", type=float, required=True,
                         help="Actual quoted total pod USD/hour, not per-GPU price")
        cmd.add_argument("--hours", type=float, default=12)
        cmd.add_argument("--spent", type=float, default=0,
                         help="All earlier spending for this project, including failed pods")
        cmd.add_argument("--storage-reserve", type=float, default=5)
        if name == "lease":
            cmd.add_argument("--pod-id", required=True)
            cmd.add_argument("--pod-name", required=True)
            cmd.add_argument("--created-at", required=True, help="Exact createdAt from Runpod")
            cmd.add_argument("--action", choices=("stop", "terminate"), default="stop")
            cmd.add_argument("--output", required=True, help="New private file outside Git")
    watcher = commands.add_parser("watch")
    watcher.add_argument("--lease", required=True)
    watcher.add_argument("--pod-id", required=True, help="Must match the lease exactly")
    watcher.add_argument("--poll", type=float, default=15)
    watcher.add_argument("--api-key-file", help="Private key file; avoids putting secrets in arguments")
    args = parser.parse_args(argv)
    try:
        if args.command == "quote":
            print(json.dumps(quote(args.hourly_rate, args.hours, args.spent,
                                   args.storage_reserve), indent=2))
        elif args.command == "lease":
            lease = make_lease(args.pod_id, args.pod_name, args.created_at,
                               args.hourly_rate, args.hours, args.spent,
                               args.storage_reserve, args.action)
            write_lease(args.output, lease)
            print(json.dumps({"deadline": utc(lease["deadline_epoch"]),
                              "projected_total_usd": lease["projected_total_usd"]}))
        else:
            lease = json.loads(Path(args.lease).read_text())
            validate_lease(lease)
            if args.pod_id != lease["pod_id"]:
                raise ValueError("Explicit pod ID does not match lease")
            stop = [False]
            def request_stop(signum, frame):
                stop[0] = True
            signal.signal(signal.SIGINT, request_stop)
            signal.signal(signal.SIGTERM, request_stop)
            key = (Path(args.api_key_file).expanduser().read_text().strip()
                   if args.api_key_file else os.environ.get("RUNPOD_API_KEY", ""))
            watch(lease, RunpodAPI(key),
                  stop_requested=lambda: stop[0], poll=args.poll)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Budget guard error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
