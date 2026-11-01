# Warmup portions adapted from NVIDIA Cosmos3-H-Dreams.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy adapter to the pinned, real NVIDIA Cosmos3-H-Dreams runtime.

No synthetic video fallback is provided. Importing this module is safe in a
CPU-only UI process; ``load`` requires the upstream GPU environment. All model
operations must run on the same owner thread, normally a single-worker pool.
"""

from __future__ import annotations

import importlib
import hashlib
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

DEFAULT_PRESET = "wan-taehv-chunk2-2steps-fademem-compile"
DEFAULT_CHECKPOINT = "checkpoints/exp15/c3-h-d_2-step_model_bf16.safetensors"
UPSTREAM_COMMIT = "fb18792d8af680a713801dea7688919e46329810"
# Replaces the preset's decoder. The full Wan VAE decoder is the one the
# latents were trained against; the preset's tiny TAEHV approximation is
# faster but visibly softer.
DECODERS = ("wan", "taehv")
# Replaces the preset's video frames per model call. Eight is the chunk of
# NVIDIA's recommended 2-step preset; four is one latent frame, the model
# card's streaming step, which no named upstream preset validates.
CHUNK_FRAMES = (4, 8)


@dataclass(frozen=True)
class GenerationResult:
    frames: Any
    """Contiguous uint8 RGB array shaped [time, height, width, 3]."""
    metrics: dict[str, float]


def _runtime_imports(upstream_root: Path | None) -> dict[str, Any]:
    if upstream_root is not None:
        for package in ("flashdreams", "cosmos3_h_dreams"):
            source = upstream_root / package
            if not source.is_dir():
                raise FileNotFoundError(f"upstream package directory missing: {source}")
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))
    try:
        torch = importlib.import_module("torch")
        np = importlib.import_module("numpy")
        config = importlib.import_module("cosmos3_h_dreams.config")
        inputs = importlib.import_module("cosmos3_h_dreams.inputs")
        model = importlib.import_module("cosmos3_h_dreams.model")
    except ImportError as error:
        raise RuntimeError(
            "Cosmos GPU dependencies are unavailable. Install the pinned upstream "
            "workspace and run CholeScope in its Python environment."
        ) from error
    return {"torch": torch, "np": np, "config": config, "inputs": inputs, "model": model}


class CosmosBackend:
    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        upstream_root: str | Path | None = None,
        preset: str = DEFAULT_PRESET,
        *,
        decoder: str = "wan",
        chunk_frames: int | None = None,
        device: str = "cuda:0",
    ) -> None:
        if decoder not in DECODERS:
            raise ValueError(f"decoder must be one of: {', '.join(DECODERS)}")
        if chunk_frames is not None and (type(chunk_frames) is not int or chunk_frames not in CHUNK_FRAMES):
            raise ValueError(f"chunk_frames must be one of: {', '.join(map(str, CHUNK_FRAMES))}")
        self.checkpoint = str(Path(checkpoint).expanduser())
        self.upstream_root = Path(upstream_root).expanduser().resolve() if upstream_root else None
        self.preset = preset
        self.decoder = decoder
        self.chunk_frames = chunk_frames
        self.device = device
        self.status = "not loaded"
        self.actions_per_chunk = 8
        self.fps = 10.0
        self._owner: int | None = None
        self._runtime: dict[str, Any] | None = None
        self._model: Any = None
        self._scene: Any = None
        self._rollout: Any = None
        self._config: Any = None
        self._options: dict[str, Any] = {}
        self._scene_key: tuple[Path, int, str] | None = None
        self._chunks = 0

    def _check_owner(self) -> None:
        current = threading.get_ident()
        if self._owner is None:
            self._owner = current
        elif self._owner != current:
            raise RuntimeError("CosmosBackend must be used from its single GPU owner thread")

    def load(self) -> None:
        self._check_owner()
        if self._model is not None:
            return
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError(f"model checkpoint missing: {self.checkpoint}")
        self.status = "loading"
        try:
            runtime = _runtime_imports(self.upstream_root)
            torch = runtime["torch"]
            if not self.device.startswith("cuda") or not torch.cuda.is_available():
                raise RuntimeError("Real Cosmos generation requires an NVIDIA CUDA GPU")
            presets = runtime["config"].COSMOS3_H_DREAMS_PRESETS
            if self.preset not in presets:
                raise ValueError(f"unknown upstream preset: {self.preset}")
            self._options = {**presets[self.preset], "decoder": self.decoder, "device": self.device}
            if self.chunk_frames is not None:
                self._options["chunk_size"] = self.chunk_frames // 4
            # Mirror the public application constructor exactly, including
            # dataclass defaults absent from individual presets.
            defaults = {field.name: field.default
                        for field in fields(runtime["config"].Cosmos3HDreamsConfig)}
            options = {**defaults, **self._options}
            if options["compile_denoise"]:
                os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
            model_keys = (
                "device", "height", "width", "seed", "chunk_size", "denoise_steps", "decoder",
                "credential_path", "window_frames", "cache_mode", "fademem_slots", "fademem_beta",
                "fademem_merge_selection", "overlap_cache_update", "compile_denoise",
                "compile_denoise_mode", "cuda_graph_denoise", "cuda_graph_warmup_iters", "profile",
            )
            with torch.inference_mode():
                self._model = runtime["model"].Cosmos3HDreamsModel(
                    checkpoint=self.checkpoint,
                    action_domain_id=options["domain_id"],
                    **{key: options[key] for key in model_keys},
                )
            self._runtime = runtime
            self.actions_per_chunk = int(options["chunk_size"]) * 4
            self.fps = float(options["fps"])
            self.status = "loaded; scene required"
        except Exception:
            self.status = "load failed"
            raise

    def reset(self, image_path: str | Path, seed: int = 0) -> None:
        """Reset image, noise, decoder and history without reloading weights."""
        self._check_owner()
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"scene image missing: {path}")
        self.load()
        assert self._runtime is not None
        runtime = self._runtime
        torch = runtime["torch"]
        self.status = "preparing scene"
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        key = (path, seed, digest)
        try:
            with torch.inference_mode():
                if self._rollout is not None:
                    self._rollout.wait_for_pending_cache_update(block=True)
                if key == self._scene_key and self._rollout is not None:
                    self._model.release_session(self._scene, self._rollout)
                    self._rollout = self._model.new_session(self._scene)
                else:
                    # Drop old KV caches before allocating a different scene.
                    self._rollout = None
                    self._scene = None
                    self._scene_key = None
                    self._model.rollout_seed = seed
                    self._config = runtime["config"].Cosmos3HDreamsConfig(
                        **{**self._options, "checkpoint": self.checkpoint,
                           "initial_frame": path, "seed": seed, "actions": None},
                    )
                    self._scene = runtime["inputs"].prepare_scene(self._config, self._model)
                    self._rollout = self._model.new_session(self._scene)
                    self._warmup()
                    self._scene_key = key
            self._chunks = 0
            self.status = "ready"
        except Exception:
            self.status = "scene failed"
            self._rollout = None
            self._scene_key = None
            raise

    def _warmup(self) -> None:
        if not self._config.compile_denoise:
            return
        self.status = "compiling"
        rollout = self._rollout
        capture_steps = math.ceil((self._config.cuda_graph_warmup_iters + 1) / self._config.denoise_steps)
        for _ in range(capture_steps):
            timings = rollout.prewarm_rollout(rollout._compile_start_step + capture_steps, profile=True)
            if not self._config.cuda_graph_denoise:
                break
            if timings and timings[-1].get("cuda_graph_captured") and (
                self._config.cache_mode != "fademem"
                or rollout._compile_start_step == 0
                or timings[-1].get("cuda_graph_fill_captured")
            ):
                break
        else:
            raise RuntimeError("upstream model warmup did not capture denoising graphs")
        self._runtime["torch"].cuda.synchronize(self.device)

    def generate(self, action_rows: Any) -> GenerationResult:
        """Consume one full normalized 28D chunk and return actual model pixels."""
        self._check_owner()
        if self._runtime is None or self._rollout is None:
            raise RuntimeError("reset a scene before generating")
        runtime = self._runtime
        np, torch = runtime["np"], runtime["torch"]
        actions = np.asarray(action_rows, dtype=np.float32)
        if actions.shape != (self.actions_per_chunk, 28) or not np.isfinite(actions).all():
            raise ValueError(f"actions must be finite with shape [{self.actions_per_chunk}, 28]")
        padded = runtime["inputs"].normalize_and_pad_actions(actions)
        with torch.inference_mode():
            started = time.perf_counter()
            latents = self._rollout.step(padded, profile=True)
            torch.cuda.synchronize(self.device)
            inferred = time.perf_counter()
            pixels = self._model.decode_tensor(torch.cat(latents, dim=1))
            # NVIDIA decode_tensor returns [-1, 1], TCHW. CPU materialization
            # synchronizes decode and makes the measured wall latency honest.
            frames = (pixels.detach().float().clamp(-1, 1).add(1).mul(127.5)
                      .round().to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy())
            if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) != self.actions_per_chunk:
                raise RuntimeError(f"upstream returned unexpected RGB frame shape: {frames.shape}")
            finished = time.perf_counter()
        self._chunks += 1
        metrics = {str(key): float(value) for key, value in self._rollout.last_timing.items()}
        metrics.update(model_s=inferred - started, decode_s=finished - inferred,
                       total_s=finished - started, frames=float(len(frames)),
                       generation_fps=len(frames) / max(finished - started, 1e-9),
                       model_fps=self.fps, chunk_index=float(self._chunks - 1))
        return GenerationResult(frames=frames, metrics=metrics)

    def close(self) -> None:
        self._check_owner()
        if self._rollout is not None:
            self._rollout.wait_for_pending_cache_update(block=True)
        self._rollout = self._scene = self._model = None
        self._scene_key = None
        self.status = "closed"

    def provenance(self) -> dict[str, Any]:
        """Cheap export metadata; excludes machine paths and never hashes weights.

        ``upstream_commit_expected`` identifies the supported source revision,
        not a fresh verification of the deployed checkout. Checkpoint byte size
        identifies truncation but is not a checksum or integrity assertion.
        """
        checkpoint = Path(self.checkpoint)
        return {
            "model": "nvidia/Cosmos3-H-Dreams",
            "checkpoint_name": checkpoint.name,
            "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
            "upstream_commit_expected": UPSTREAM_COMMIT,
            "preset": self.preset,
            "decoder": self.decoder,
            "action_dimensions": 28,
            "actions_per_chunk": self.actions_per_chunk,
            "chunk_frames_override": self.chunk_frames,
            "model_fps": self.fps,
            "domain_id": self._options.get("domain_id", 25),
            "seed": self._scene_key[1] if self._scene_key else None,
            "scene_image_sha256": self._scene_key[2] if self._scene_key else None,
            "height": self._options.get("height", 480),
            "width": self._options.get("width", 832),
            "controls": "pointer-keyboard-relative-pose-v2",
        }
