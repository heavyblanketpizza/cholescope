"""Bounded, losslessly logged action recordings with streamed MP4 encoding."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np


class Recording:
    def __init__(self, output_dir: Path, *, branch: str, scene_id: str, seed: int, fps: int = 10, provenance=None):
        if branch not in {"a", "b"}:
            raise ValueError("Unknown comparison branch")
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg is required to record MP4 comparisons")
        self.id = uuid.uuid4().hex
        self.branch, self.scene_id, self.seed, self.fps = branch, scene_id, seed, fps
        self.provenance = provenance or {}
        self.directory = output_dir / self.id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.process = None
        self.actions: list[list[float]] = []
        self.frame_count = 0
        self.shape = None
        self.closed = False
        self.log = None

    def append(self, frames, actions):
        if self.closed:
            raise RuntimeError("Recording is closed")
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError("Expected uint8 THWC RGB frames")
        if len(frames) != len(actions):
            raise ValueError("Actions and frames must have the same cadence")
        if not self.process:
            height, width = frames.shape[1:3]
            self.shape = (height, width)
            self.log = (self.directory / "encoder.log").open("wb")
            self.process = subprocess.Popen([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                "-pixel_format", "rgb24", "-video_size", f"{width}x{height}", "-framerate", str(self.fps),
                "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.directory / "video.mp4"),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        if tuple(frames.shape[1:3]) != self.shape:
            raise ValueError("Frame geometry changed during recording")
        self.process.stdin.write(np.ascontiguousarray(frames).tobytes())
        # Match the adapter's conversion before normalization/padding, so the
        # export preserves the actual float32 values consumed by the model.
        self.actions.extend(np.asarray(actions, dtype=np.float32).tolist())
        self.frame_count += len(frames)

    def finish(self):
        if self.closed:
            raise RuntimeError("Recording is already closed")
        self.closed = True
        if self.process:
            try:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
                finally:
                    try:
                        code = self.process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
                        raise RuntimeError("MP4 encoding timed out")
            finally:
                self.log.close()
            if code:
                raise RuntimeError("MP4 encoding failed; inspect the local encoder log")
        if not self.frame_count:
            raise ValueError("No generated frames were recorded")
        metadata = {
            "schema_version": 1, "run_id": self.id, "branch": self.branch,
            "scene_id": self.scene_id, "seed": self.seed, "fps": self.fps,
            "frame_count": self.frame_count, "generated": True,
            "model": "nvidia/Cosmos3-H-Dreams", "action_format": "normalized_cmr_28d",
            "action_dtype": "float32", "model_action_dimensions": 64,
            "padding": "zeros_after_dimension_28",
            "provenance": self.provenance,
            "actions": self.actions,
        }
        (self.directory / "actions.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return {"type": "recording", "branch": self.branch, "run_id": self.id,
                "url": f"/api/runs/{self.id}/video.mp4", "actions_url": f"/api/runs/{self.id}/actions",
                "duration_ms": self.frame_count * 1000 / self.fps}
