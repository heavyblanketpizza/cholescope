"""Adapter contract tests with a small runtime double; no CUDA claim is made."""

import contextlib
import dataclasses
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from cholescope.controls import KeyboardController
from cholescope import model_backend as module


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self): return self
    def float(self): return Tensor(self.values.astype(np.float32))
    def clamp(self, low, high): return Tensor(np.clip(self.values, low, high))
    def add(self, x): return Tensor(self.values + x)
    def mul(self, x): return Tensor(self.values * x)
    def round(self): return Tensor(np.round(self.values))
    def to(self, *, dtype): return Tensor(self.values.astype(dtype))
    def permute(self, *axes): return Tensor(self.values.transpose(*axes))
    def contiguous(self): return Tensor(np.ascontiguousarray(self.values))
    def cpu(self): return self
    def numpy(self): return self.values


class Rollout:
    _compile_start_step = 2

    def __init__(self):
        self.calls = []
        self.warmups = []
        self.waits = 0
        self.last_timing = {"denoise_s": 0.01}

    def wait_for_pending_cache_update(self, *, block=False):
        self.waits += 1

    def prewarm_rollout(self, steps, *, profile):
        self.warmups.append(steps)
        return [{"cuda_graph_captured": 1, "cuda_graph_fill_captured": 1}]

    def step(self, actions, *, profile):
        self.calls.append(actions)
        # One latent frame per four action rows, as in the upstream session.
        return [Tensor(np.zeros((1, 1, 1, 1))) for _ in range(len(actions.values) // 4)]


class Model:
    def __init__(self, **kwargs):
        self.arguments = kwargs
        self.rollout_seed = kwargs["seed"]
        self.sessions = []
        self.releases = []
        self.saved = None

    def new_session(self, scene):
        result = self.saved or Rollout()
        self.saved = None
        self.sessions.append((scene, self.rollout_seed, result))
        return result

    def release_session(self, scene, rollout):
        self.releases.append((scene, rollout))
        self.saved = rollout

    def decode_tensor(self, latent):
        # The Wan VAE expands each latent frame into four video frames.
        pixels = np.zeros((4 * latent.values.shape[1], 3, 2, 2), dtype=np.float32)
        pixels[:, 0] = -1
        pixels[:, 2] = 1
        return Tensor(pixels)


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    defaults = dict(
        domain_id=25, prompt="upstream prompt", device="cuda:0", fps=10.0,
        height=480, width=832, seed=0, chunk_size=2, denoise_steps=2,
        decoder="taehv", credential_path=None, window_frames=12,
        cache_mode="fademem", fademem_slots=12, fademem_beta=0.3,
        fademem_merge_selection="cosine", overlap_cache_update=True,
        compile_denoise=True, compile_denoise_mode="default",
        cuda_graph_denoise=True, cuda_graph_warmup_iters=2, profile=False,
        actions=None,
    )
    config_class = dataclasses.make_dataclass(
        "Config", [("checkpoint", str), ("initial_frame", Path)] +
        [(key, object, dataclasses.field(default=value)) for key, value in defaults.items()],
    )
    scenes = []

    def scene(config, model):
        result = SimpleNamespace(config=config)
        scenes.append(result)
        return result

    def pad(actions):
        result = np.zeros((len(actions), 64), dtype=np.float32)
        result[:, :28] = actions
        return Tensor(result)

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, synchronize=lambda *_: None),
        inference_mode=contextlib.nullcontext,
        uint8=np.uint8,
        cat=lambda values, dim: Tensor(np.concatenate([x.values for x in values], axis=dim)),
    )
    value = dict(torch=torch, np=np,
                 config=SimpleNamespace(Cosmos3HDreamsConfig=config_class,
                                        COSMOS3_H_DREAMS_PRESETS={module.DEFAULT_PRESET: defaults}),
                 inputs=SimpleNamespace(prepare_scene=scene, normalize_and_pad_actions=pad),
                 model=SimpleNamespace(Cosmos3HDreamsModel=Model))
    monkeypatch.setattr(module, "_runtime_imports", lambda _root: value)
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    image = tmp_path / "scene.png"
    image.touch()
    return module.CosmosBackend(checkpoint), image, scenes


def test_import_does_not_load_torch_or_upstream():
    env = dict(os.environ, PYTHONPATH=str(Path(module.__file__).parents[1]))
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import cholescope.model_backend; "
         "assert 'torch' not in sys.modules; assert 'cosmos3_h_dreams' not in sys.modules"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_checkpoint_fails_before_loading_dependencies(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        module.CosmosBackend(tmp_path / "missing").load()


def test_unknown_decoder_fails_before_loading(tmp_path):
    with pytest.raises(ValueError, match="decoder"):
        module.CosmosBackend(tmp_path / "missing", decoder="taehv2super")


def test_default_decoder_is_full_wan_although_preset_names_taehv(runtime):
    backend, image, scenes = runtime
    backend.reset(image)
    assert backend._model.arguments["decoder"] == "wan"
    assert scenes[-1].config.decoder == "wan"


def test_taehv_decoder_remains_selectable(runtime):
    backend, image, scenes = runtime
    fast = module.CosmosBackend(backend.checkpoint, decoder="taehv")
    fast.reset(image)
    assert fast._model.arguments["decoder"] == "taehv"
    assert scenes[-1].config.decoder == "taehv"
    assert fast.provenance()["decoder"] == "taehv"


@pytest.mark.parametrize("chunk_frames", [0, 6, 16, True, 4.0])
def test_unmeasured_chunk_frames_fail_before_loading(tmp_path, chunk_frames):
    with pytest.raises(ValueError, match="chunk_frames"):
        module.CosmosBackend(tmp_path / "missing", chunk_frames=chunk_frames)


def test_four_frame_chunks_replace_only_the_preset_chunk_size(runtime):
    backend, image, scenes = runtime
    quick = module.CosmosBackend(backend.checkpoint, chunk_frames=4)
    quick.reset(image)
    assert quick._model.arguments["chunk_size"] == 1
    assert quick._model.arguments["cache_mode"] == "fademem"
    assert scenes[-1].config.chunk_size == 1
    result = quick.generate([KeyboardController().action()] * 4)
    assert result.frames.shape == (4, 2, 2, 3)
    with pytest.raises(ValueError, match="shape"):
        quick.generate([KeyboardController().action()] * 8)


def test_provenance_records_a_chunk_override(runtime):
    backend, image, _ = runtime
    quick = module.CosmosBackend(backend.checkpoint, chunk_frames=4)
    quick.reset(image)
    metadata = quick.provenance()
    assert metadata["chunk_frames_override"] == 4
    assert metadata["actions_per_chunk"] == 4


def test_chunk_is_padded_and_real_decoder_range_becomes_rgb(runtime):
    backend, image, _ = runtime
    backend.reset(image, seed=42)
    rows = [KeyboardController().action()] * 8
    result = backend.generate(rows)
    forwarded = backend._rollout.calls[0].values
    assert forwarded.shape == (8, 64)
    np.testing.assert_allclose(forwarded[:, :28], rows)
    assert not forwarded[:, 28:].any()
    assert result.frames.shape == (8, 2, 2, 3)
    assert result.frames.dtype == np.uint8
    assert result.frames[0, 0, 0].tolist() == [0, 128, 255]
    assert result.metrics["frames"] == 8
    assert result.metrics["chunk_index"] == 0
    assert result.metrics["total_s"] > 0
    assert backend._model.arguments["action_domain_id"] == 25
    assert backend._model.sessions[-1][1] == 42


def test_same_scene_and_seed_reuse_compiled_rollout_and_reset_counter(runtime):
    backend, image, scenes = runtime
    backend.reset(image, seed=7)
    rollout = backend._rollout
    backend.generate([KeyboardController().action()] * 8)
    backend.reset(image, seed=7)
    assert backend._rollout is rollout
    assert len(scenes) == 1
    assert len(rollout.warmups) == 1
    assert len(backend._model.releases) == 1
    assert backend.generate([KeyboardController().action()] * 8).metrics["chunk_index"] == 0


def test_new_seed_allocates_new_rollout(runtime):
    backend, image, scenes = runtime
    backend.reset(image, seed=1)
    old = backend._rollout
    backend.reset(image, seed=2)
    assert backend._rollout is not old
    assert backend._model.sessions[-1][1] == 2
    assert len(scenes) == 2


def test_changed_image_at_same_path_does_not_reuse_old_conditioning(runtime):
    backend, image, scenes = runtime
    backend.reset(image, seed=1)
    first_hash = backend.provenance()["scene_image_sha256"]
    image.write_bytes(b"different image")
    backend.reset(image, seed=1)
    assert len(scenes) == 2
    assert backend.provenance()["scene_image_sha256"] != first_hash


@pytest.mark.parametrize("bad", [np.zeros((7, 28)), np.zeros((8, 64)), np.full((8, 28), np.nan)])
def test_invalid_action_chunks_fail_before_inference(runtime, bad):
    backend, image, _ = runtime
    backend.reset(image)
    with pytest.raises(ValueError, match="shape"):
        backend.generate(bad)
    assert not backend._rollout.calls


def test_gpu_calls_cannot_cross_owner_threads(runtime):
    backend, image, _ = runtime
    backend.reset(image)
    with ThreadPoolExecutor(max_workers=1) as worker:
        with pytest.raises(RuntimeError, match="owner thread"):
            worker.submit(backend.reset, image).result()


def test_generate_requires_scene_and_close_waits_for_cache(runtime):
    backend, image, _ = runtime
    with pytest.raises(RuntimeError, match="reset a scene"):
        backend.generate([])
    backend.reset(image)
    rollout = backend._rollout
    backend.close()
    assert rollout.waits > 0
    assert backend.status == "closed"
    with pytest.raises(RuntimeError, match="reset a scene"):
        backend.generate([])


def test_provenance_omits_local_paths_and_reports_scene_seed(runtime):
    backend, image, _ = runtime
    backend.reset(image, seed=19)
    metadata = backend.provenance()
    assert metadata["checkpoint_name"] == "model.safetensors"
    assert metadata["checkpoint_bytes"] == 0
    assert metadata["seed"] == 19
    assert metadata["upstream_commit_expected"] == module.UPSTREAM_COMMIT
    assert metadata["preset"] == module.DEFAULT_PRESET
    assert metadata["decoder"] == "wan"
    assert metadata["chunk_frames_override"] is None
    assert metadata["actions_per_chunk"] == 8
    assert metadata["controls"] == "pointer-keyboard-relative-pose-v2"
    assert str(image.parent) not in str(metadata)
