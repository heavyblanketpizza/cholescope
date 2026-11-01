#!/usr/bin/env python3
"""Paired Cosmos responsiveness experiment, run inside the existing GPU runtime.

This script never provisions a GPU or contacts a cloud API. A manifest is either
a list of scenes or {"scenes": [...]}; every scene needs id, path, source_url,
source_recording_id, sha256 and license. Paths are relative to the manifest.
The caller must separately enforce the rental budget and shut the GPU down.

Latency is first action sample to decoded chunk ready. The current application
waits 100 ms between samples and dispatches immediately after the last sample;
--sampling-policy legacy adds the old trailing wait for before/after checks.
Latency excludes network and browser playback. An in-flight model call cannot
be interrupted by the wall guard; it refuses new work conservatively and logs
incomplete cases.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Callable
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from cholescope.controls import ACTION_FPS, KeyboardController
from cholescope.model_backend import CosmosBackend, DEFAULT_CHECKPOINT, DEFAULT_PRESET

MODES = ("neutral", "x+", "x-", "y+", "y-")


@dataclass(frozen=True)
class Scene:
    id: str
    path: Path
    source_url: str
    source_recording_id: str
    sha256: str
    license: str

    def metadata(self) -> dict[str, Any]:
        return {**asdict(self), "path": str(self.path)}


@dataclass(frozen=True)
class Settings:
    output_dir: Path
    checkpoint: str = DEFAULT_CHECKPOINT
    upstream_root: str | None = None
    preset: str = DEFAULT_PRESET
    decoders: tuple[str, ...] = ("wan",)
    chunk_frames: tuple[int, ...] = (8, 4)
    seeds: tuple[int, ...] = (42,)
    modes: tuple[str, ...] = MODES
    frames: int = 40
    displacement: float = .04
    sampling_policy: str = "current"
    max_cases: int | None = None
    max_wall_seconds: float = 1800
    case_reserve_seconds: float = 120
    load_reserve_seconds: float = 300
    chunk_reserve_seconds: float = 10
    video: bool = True


def load_scenes(manifest: Path) -> list[Scene]:
    payload = json.loads(manifest.read_text())
    records = payload.get("scenes") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("manifest must contain a nonempty scene list")
    scenes, ids = [], set()
    required = {"id", "path", "source_url", "source_recording_id", "sha256", "license"}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not required <= record.keys():
            raise ValueError(f"scene {index} requires {', '.join(sorted(required))}")
        if any(not isinstance(record[key], str) or not record[key].strip() for key in required):
            raise ValueError(f"scene {index} metadata must be nonempty strings")
        scene_id = record["id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", scene_id) or scene_id in ids:
            raise ValueError(f"unsafe or duplicate scene id: {scene_id!r}")
        ids.add(scene_id)
        url = urlparse(record["source_url"])
        if url.scheme not in {"https", "http"} or not url.netloc:
            raise ValueError(f"{scene_id}: source_url must be a public HTTP(S) source")
        digest = record["sha256"].lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{scene_id}: invalid SHA-256")
        path = (manifest.resolve().parent / Path(record["path"]).expanduser()).resolve()
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"{scene_id}: image SHA-256 mismatch")
        with Image.open(path) as image:
            image.verify()
        scenes.append(Scene(scene_id, path, record["source_url"],
                            record["source_recording_id"], digest, record["license"]))
    return scenes


def validate(settings: Settings) -> None:
    if not settings.decoders or any(x not in {"wan", "taehv"} for x in settings.decoders):
        raise ValueError("decoders must be wan and/or taehv")
    if not settings.chunk_frames or any(type(x) is not int or x not in {4, 8} for x in settings.chunk_frames):
        raise ValueError("chunk_frames must contain 4 and/or 8")
    if not settings.modes or any(x not in MODES for x in settings.modes):
        raise ValueError("unknown action mode")
    if settings.sampling_policy not in {"current", "legacy"}:
        raise ValueError("sampling_policy must be current or legacy")
    if not settings.seeds or any(type(x) is not int or not 0 <= x < 2**32 for x in settings.seeds):
        raise ValueError("seeds must be integers in [0, 2**32)")
    for values in (settings.decoders, settings.chunk_frames, settings.modes, settings.seeds):
        if len(values) != len(set(values)):
            raise ValueError("duplicate experiment settings would duplicate cases")
    if type(settings.frames) is not int or settings.frames <= 0 or any(settings.frames % x for x in settings.chunk_frames):
        raise ValueError("frames must be positive and divisible by every chunk size")
    if not math.isfinite(settings.displacement) or not 0 < settings.displacement <= .04:
        raise ValueError("displacement must be positive and at most 0.04 model units")
    if settings.max_cases is not None and (type(settings.max_cases) is not int or settings.max_cases <= 0):
        raise ValueError("max_cases must be a positive integer")
    for value in (settings.max_wall_seconds, settings.case_reserve_seconds,
                  settings.load_reserve_seconds, settings.chunk_reserve_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("wall-time limits and reserves must be finite and positive")


def action_rows(mode: str, frames: int, displacement: float) -> np.ndarray:
    """Same float32 10 Hz pose trajectory for every decoder/chunk configuration."""
    rows = np.tile(np.asarray(KeyboardController().action(), dtype=np.float32), (frames, 1))
    if mode != "neutral":
        axis = 0 if mode[0] == "x" else 1
        sign = 1 if mode[1] == "+" else -1
        rows[:, axis] = np.linspace(0, sign * displacement, frames, dtype=np.float32)
    return rows


def cases_for(scenes: list[Scene], settings: Settings) -> list[dict[str, Any]]:
    validate(settings)
    cases = []
    for decoder in settings.decoders:
        for chunk in settings.chunk_frames:
            for scene in scenes:
                for seed in settings.seeds:
                    for mode in settings.modes:
                        pair_id = f"{scene.id}-seed{seed}-{mode.replace('+', 'plus').replace('-', 'minus')}"
                        cases.append({"case_id": f"{decoder}-chunk{chunk}-{pair_id}", "pair_id": pair_id,
                                      "decoder": decoder, "chunk_frames": chunk, "scene": scene.metadata(),
                                      "seed": seed, "mode": mode})
    return cases


def plan(scenes: list[Scene], settings: Settings) -> dict[str, Any]:
    cases = cases_for(scenes, settings)
    selected = cases[:settings.max_cases]
    groups = {(case["decoder"], case["chunk_frames"]) for case in selected}
    # Reserves are intentionally conservative planning allowances, not benchmarks.
    allowance = len(groups) * settings.load_reserve_seconds + len(selected) * settings.case_reserve_seconds
    root = Path(__file__).resolve().parents[1]
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        "scripts/benchmark_responsiveness.py", "src/cholescope/model_backend.py",
        "src/cholescope/controls.py", "src/cholescope/server.py")}
    return {"schema_version": 1, "planned_cases": len(cases), "selected_cases": len(selected),
            "cases_excluded_by_limit": len(cases) - len(selected),
            "scene_count": len(scenes), "source_recording_count": len({s.source_recording_id for s in scenes}),
            "generated_frames_if_complete": len(selected) * settings.frames,
            "sampling_wait_seconds_if_complete": sum(
                (settings.frames if settings.sampling_policy == "legacy"
                 else settings.frames - settings.frames // case["chunk_frames"]) / ACTION_FPS
                for case in selected),
            "planning_allowance_seconds": allowance, "max_wall_seconds": settings.max_wall_seconds,
            "allowance_exceeds_wall_limit": allowance > settings.max_wall_seconds,
            "timing_basis": "first_action_sample_to_decoded_chunk_ready; no network or browser playback",
            "source_sha256": source_hashes,
            "case_order": "decoder, chunk_frames, scene, seed, mode; max_cases may truncate pairing",
            "settings": {**asdict(settings), "output_dir": str(settings.output_dir)},
            "scenes": [scene.metadata() for scene in scenes], "cases": cases}


def encoder_path() -> str:
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError("FFmpeg is required; use the persistent asset runtime or --no-video") from error


class Video:
    def __init__(self, directory: Path, binary: str, shape: tuple[int, ...]):
        self.path = directory / "video.mp4"
        self.log = (directory / "encoder.log").open("wb")
        height, width = shape[1:3]
        try:
            self.process = subprocess.Popen([
                binary, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                "-framerate", str(ACTION_FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(self.path),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        except Exception:
            self.log.close()
            raise

    def append(self, frames: np.ndarray) -> None:
        self.process.stdin.write(np.ascontiguousarray(frames).tobytes())

    def close(self) -> None:
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError("video encoding timed out")
            if code:
                raise RuntimeError("video encoding failed; inspect encoder.log")
        finally:
            self.log.close()


def contact_sheet(path: Path, tiles: list[tuple[str, Image.Image]]) -> None:
    width, height, columns = 320, 204, 3
    canvas = Image.new("RGB", (width * columns, height * math.ceil(len(tiles) / columns)), "#171717")
    draw = ImageDraw.Draw(canvas)
    for index, (label, picture) in enumerate(tiles):
        x, y = index % columns * width, index // columns * height
        canvas.paste(ImageOps.contain(picture, (width, height - 24)), (x, y + 24))
        draw.text((x + 6, y + 5), label, fill="white")
    canvas.save(path)


def summarize_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not chunks:
        return {}
    after_first = chunks[1:]
    return {"chunk_count": len(chunks), "after_first_chunk_count": len(after_first),
            "median_latency_s": statistics.median(c["first_sample_to_ready_s"] for c in chunks),
            "after_first_chunk_median_latency_s": (
                statistics.median(c["first_sample_to_ready_s"] for c in after_first) if after_first else None),
            "median_generate_wall_s": statistics.median(c["generate_wall_s"] for c in chunks),
            "generated_frames": sum(c["frame_count"] for c in chunks)}


def run_benchmark(scenes: list[Scene], settings: Settings, *,
                  backend_factory: Callable[..., Any] = CosmosBackend,
                  clock: Callable[[], float] = time.monotonic,
                  sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    experiment = plan(scenes, settings)
    binary = encoder_path() if settings.video else None
    settings.output_dir.mkdir(parents=True, exist_ok=False)
    (settings.output_dir / "plan.json").write_text(json.dumps(experiment, indent=2) + "\n")
    started = clock()
    deadline = started + settings.max_wall_seconds
    rows = []
    backend, group, load_error = None, None, None
    load_s = 0.0
    observed_case_s = 0.0
    observed_chunk_s = 0.0
    log = (settings.output_dir / "results.jsonl").open("w", buffering=1)

    def record(row):
        rows.append(row)
        log.write(json.dumps(row, allow_nan=False) + "\n")
        log.flush()

    try:
        for index, case in enumerate(experiment["cases"]):
            row = {"schema_version": 1, **case, "requested_frames": settings.frames,
                   "model_fps": ACTION_FPS, "displacement": settings.displacement,
                   "sampling_policy": settings.sampling_policy,
                   "timing_basis": "first_action_sample_to_decoded_chunk_ready", "chunks": []}
            if settings.max_cases is not None and index >= settings.max_cases:
                record({**row, "status": "skipped", "reason": "max_cases"})
                continue
            next_group = (case["decoder"], case["chunk_frames"])
            reserve = max(settings.case_reserve_seconds, observed_case_s * 1.5)
            if next_group != group:
                reserve += settings.load_reserve_seconds
            if deadline - clock() < reserve:
                record({**row, "status": "skipped", "reason": "wall_time_reserve", "remaining_s": max(0, deadline - clock())})
                continue
            if next_group != group:
                if backend is not None:
                    backend.close()
                    backend = None
                group, load_error = next_group, None
                load_started = clock()
                try:
                    backend = backend_factory(settings.checkpoint, upstream_root=settings.upstream_root,
                                              preset=settings.preset, decoder=group[0], chunk_frames=group[1])
                    backend.load()
                except Exception as error:
                    load_error = f"{type(error).__name__}: {error}"
                load_s = clock() - load_started
            row["configuration_load_s"] = load_s
            if load_error:
                record({**row, "status": "failed", "stage": "load", "error": load_error})
                continue
            # Loading can overrun its allowance. Do not start scene preparation
            # (which may itself compile) unless its case reserve still remains.
            reserve = max(settings.case_reserve_seconds, observed_case_s * 1.5)
            if deadline - clock() < reserve:
                record({**row, "status": "skipped", "reason": "wall_time_reserve_after_load",
                        "remaining_s": max(0, deadline - clock())})
                continue
            case_started = clock()
            directory = settings.output_dir / case["case_id"]
            directory.mkdir()
            actions = action_rows(case["mode"], settings.frames, settings.displacement)
            row["action_sha256"] = hashlib.sha256(actions.astype("<f4").tobytes()).hexdigest()
            (directory / "actions.json").write_text(json.dumps({
                "action_format": "normalized_cmr_28d", "dtype": "float32", "model_fps": ACTION_FPS,
                "model_timestamps_s": [i / ACTION_FPS for i in range(settings.frames)],
                "actions": actions.tolist(), "sha256_little_endian_float32": row["action_sha256"],
            }, indent=2) + "\n")
            row["actions_file"] = f"{case['case_id']}/actions.json"
            video, tiles, stage = None, [], "reset"
            try:
                reset_started = clock()
                backend.reset(case["scene"]["path"], seed=case["seed"])
                row["reset_s"] = clock() - reset_started
                row["provenance"] = backend.provenance()
                if backend.actions_per_chunk != case["chunk_frames"] or backend.fps != ACTION_FPS:
                    raise RuntimeError("backend cadence differs from the planned paired experiment")
                with Image.open(case["scene"]["path"]) as initial:
                    tiles.append(("Input image", initial.convert("RGB")))
                samples = set(np.linspace(0, settings.frames - 1, 8, dtype=int).tolist())
                row["status"] = "complete"
                for offset in range(0, settings.frames, case["chunk_frames"]):
                    stage = "generate"
                    needed = case["chunk_frames"] / ACTION_FPS + max(settings.chunk_reserve_seconds, observed_chunk_s * 1.5)
                    if deadline - clock() < needed:
                        row.update(status="incomplete", reason="wall_time_reserve")
                        break
                    block = actions[offset:offset + case["chunk_frames"]]
                    sampled_at = []
                    for sample_index, _ in enumerate(block):
                        if sample_index:
                            sleep(1 / ACTION_FPS)
                        sampled_at.append(clock())
                    if settings.sampling_policy == "legacy":
                        sleep(1 / ACTION_FPS)
                    generation_started = clock()
                    try:
                        result = backend.generate(block)
                    except Exception:
                        row["failed_chunk"] = {"frame_offset": offset,
                                               "sampled_at_run_s": [t - started for t in sampled_at],
                                               "generate_started_at_run_s": generation_started - started,
                                               "failed_at_run_s": clock() - started}
                        raise
                    ready = clock()
                    if result.frames.dtype != np.uint8 or result.frames.ndim != 4 or result.frames.shape[-1] != 3 or len(result.frames) != len(block):
                        raise RuntimeError("backend returned invalid frame geometry or count")
                    observed_chunk_s = max(observed_chunk_s, ready - generation_started)
                    row["chunks"].append({
                        "frame_offset": offset, "frame_count": len(result.frames),
                        "sampled_at_run_s": [t - started for t in sampled_at],
                        "generate_started_at_run_s": generation_started - started,
                        "ready_at_run_s": ready - started,
                        "first_sample_to_ready_s": ready - sampled_at[0],
                        "sample_to_ready_s": [ready - t for t in sampled_at],
                        "generate_wall_s": ready - generation_started,
                        "metrics": result.metrics,
                    })
                    # Persist every successful chunk before any output encoding.
                    with (directory / "chunks.jsonl").open("a") as chunk_log:
                        chunk_log.write(json.dumps(row["chunks"][-1], allow_nan=False) + "\n")
                    stage = "artifacts"
                    if binary:
                        if video is None:
                            video = Video(directory, binary, result.frames.shape)
                        video.append(result.frames)
                    for local, frame in enumerate(result.frames):
                        if offset + local in samples:
                            tiles.append((f"Frame {offset + local + 1} / {(offset + local + 1) / ACTION_FPS:.1f}s model", Image.fromarray(frame)))
            except Exception as error:
                row.update(status="failed", stage=stage, error=f"{type(error).__name__}: {error}")
            finally:
                if video is not None:
                    try:
                        video.close()
                        row["video_file"] = f"{case['case_id']}/video.mp4"
                    except Exception as error:
                        row.update(status="failed", artifact_error=f"{type(error).__name__}: {error}")
                if tiles:
                    try:
                        contact_sheet(directory / "contact-sheet.jpg", tiles)
                        row["contact_sheet_file"] = f"{case['case_id']}/contact-sheet.jpg"
                    except Exception as error:
                        row.update(status="failed", artifact_error=f"{type(error).__name__}: {error}")
            row["case_wall_s"] = clock() - case_started
            observed_case_s = max(observed_case_s, row["case_wall_s"])
            row["summary"] = summarize_chunks(row["chunks"])
            record(row)
            print(json.dumps({key: row[key] for key in ("case_id", "status", "summary")}), flush=True)
    finally:
        try:
            if backend is not None:
                backend.close()
        finally:
            log.close()
    summary = {"schema_version": 1, "wall_seconds": clock() - started,
               "planned_cases": len(rows),
               "status_counts": {status: sum(row["status"] == status for row in rows)
                                 for status in ("complete", "failed", "incomplete", "skipped")},
               "generated_frames": sum(row.get("summary", {}).get("generated_frames", 0) for row in rows),
               "selected_cases_complete": all(row["status"] == "complete" or
                                              row.get("reason") == "max_cases" for row in rows),
               "quality_scoring": "not automated; inspect paired videos/contact sheets before selecting a setting",
               "cost_accounting": "external rental/storage ledger required; no pricing or cloud API used"}
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing results are never overwritten")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--upstream-root")
    parser.add_argument("--asset-dir", type=Path, help="Reuse existing compiler caches and persistent FFmpeg; no downloads")
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--decoders", nargs="+", choices=("wan", "taehv"), default=["wan"])
    parser.add_argument("--chunk-frames", nargs="+", type=int, choices=(4, 8), default=[8, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--displacement", type=float, default=.04)
    parser.add_argument("--sampling-policy", choices=("current", "legacy"), default="current")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--max-wall-seconds", type=float, default=1800)
    parser.add_argument("--case-reserve-seconds", type=float, default=120)
    parser.add_argument("--load-reserve-seconds", type=float, default=300)
    parser.add_argument("--chunk-reserve-seconds", type=float, default=10)
    parser.add_argument("--no-video", action="store_true", help="Still writes contact sheets, actions and timings")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and show plan without loading the GPU or writing output")
    args = parser.parse_args()
    try:
        scenes = load_scenes(args.manifest)
        settings = Settings(output_dir=args.output_dir, checkpoint=args.checkpoint, upstream_root=args.upstream_root,
                            preset=args.preset, decoders=tuple(args.decoders), chunk_frames=tuple(args.chunk_frames),
                            seeds=tuple(args.seeds), modes=tuple(args.modes), frames=args.frames,
                            displacement=args.displacement, max_cases=args.max_cases,
                            sampling_policy=args.sampling_policy,
                            max_wall_seconds=args.max_wall_seconds, case_reserve_seconds=args.case_reserve_seconds,
                            load_reserve_seconds=args.load_reserve_seconds, chunk_reserve_seconds=args.chunk_reserve_seconds,
                            video=not args.no_video)
        experiment = plan(scenes, settings)
        if args.dry_run:
            print(json.dumps(experiment, indent=2))
            return
        if args.asset_dir:
            from bootstrap_gpu import cache_environment
            os.environ.update(cache_environment(args.asset_dir))
            os.environ["PATH"] = str(args.asset_dir.resolve() / "bin") + os.pathsep + os.environ.get("PATH", "")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        summary = run_benchmark(scenes, settings)
        print(json.dumps(summary, indent=2))
        if not summary["selected_cases_complete"]:
            raise SystemExit(1)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
