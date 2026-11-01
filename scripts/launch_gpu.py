#!/usr/bin/env python3
"""Launch CholeScope with caches on an explicit disk and a Runpod lease guard.

On Runpod, start another runpod_budget.py watch process on an independent
controller before this launcher. The local watchdog cannot cover pod boot or
an API outage. Its successful exit confirms shutdown; killing this wrapper
alone is not a billing control.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from bootstrap_gpu import cache_environment, UPSTREAM_REVISION
from runpod_budget import RunpodAPI, timestamp, validate_lease, verify_pod


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upstream-root", default=None)
    parser.add_argument("--lease", help="Private lease file; required inside Runpod")
    parser.add_argument("--scene-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8790, type=int)
    parser.add_argument("--chunk-frames", type=int, choices=(4, 8),
                        help="Video frames per model call; 4 is an experimental lower-latency opt-in")
    parser.add_argument("--offline", action="store_true",
                        help="Use only prefetched dependencies; forbid Hugging Face refreshes")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    upstream = Path(args.upstream_root).expanduser().resolve() if args.upstream_root else repo / ".vendor/cosmos3-h-dreams"
    python = upstream / ".venv/bin/python"
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not python.is_file() or not checkpoint.is_file():
        parser.error("A completed checkpoint and bootstrapped upstream environment are required")
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        parser.error("Upstream does not match the supported pinned revision")
    env = os.environ.copy()
    env.update(cache_environment(args.asset_dir))
    env["PATH"] = str(Path(args.asset_dir).expanduser().resolve() / "bin") + os.pathsep + env.get("PATH", "")
    for value in cache_environment(args.asset_dir).values():
        Path(value).mkdir(parents=True, exist_ok=True)
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    pod_id = env.get("RUNPOD_POD_ID")
    guard = None
    if pod_id or args.lease:
        if not pod_id or not args.lease:
            parser.error("Runpod launch requires RUNPOD_POD_ID and --lease")
        lease = validate_lease(json.loads(Path(args.lease).read_text()))
        if pod_id != lease["pod_id"]:
            parser.error("Runpod environment does not match the explicit lease")
        if time.time() >= lease["deadline_epoch"] or timestamp(lease["created_at"]) > time.time() + 5:
            parser.error("Lease is expired or has a future creation time")
        api = RunpodAPI(env.get("RUNPOD_API_KEY", ""))
        pod = api.get(pod_id)
        if pod is None:
            parser.error("Runpod reports that this pod no longer exists")
        verify_pod(pod, lease)
        if pod.get("locked"):
            parser.error("Unlock the exact pod before launch so the lease guard can stop it")
        guard = subprocess.Popen([
            sys.executable, str(repo / "scripts/runpod_budget.py"), "watch",
            "--lease", str(Path(args.lease).resolve()), "--pod-id", pod_id,
        ], env=env, start_new_session=True)
        # Detect immediate configuration/authentication failures before loading
        # large model weights. The independent controller is still required.
        time.sleep(0.5)
        if guard.poll() is not None:
            parser.error("Pod watchdog exited before launch; inspect its output")
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(args.asset_dir).expanduser().resolve() / "outputs"
    command = [str(python), "-m", "cholescope", "--checkpoint", str(checkpoint),
               "--upstream-root", str(upstream), "--output-dir", str(output),
               "--host", args.host, "--port", str(args.port)]
    if args.scene_dir:
        command += ["--scene-dir", str(Path(args.scene_dir).expanduser().resolve())]
    if args.chunk_frames:
        command += ["--chunk-frames", str(args.chunk_frames)]
    server = None
    stopping = [False]
    def request_stop(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        # Keep Runpod credentials out of the application process.
        app_env = {key: value for key, value in env.items() if key != "RUNPOD_API_KEY"}
        server = subprocess.Popen(command, cwd=repo, env=app_env)
        while server.poll() is None and not stopping[0]:
            if guard and guard.poll() is not None:
                break
            time.sleep(0.5)
    finally:
        if server and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        if guard and guard.poll() is None:
            guard.send_signal(signal.SIGTERM)
            # Do not kill the watchdog while the cloud is still billing. It
            # continues retrying until the cloud reports shutdown.
            print("Waiting for verified Runpod shutdown; API failure can extend billing.", flush=True)
            guard.wait()
    return server.returncode if server and server.returncode and not stopping[0] else 0


if __name__ == "__main__":
    raise SystemExit(main())
