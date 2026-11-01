"""Experiment pairing, timing and budget guard tests; no CUDA claims."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

SPEC = importlib.util.spec_from_file_location("responsiveness_benchmark", Path(__file__).parents[1] / "scripts/benchmark_responsiveness.py")
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def manifest(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (16, 16), "#778899").save(image)
    row = dict(id="scene-1", path=image.name, source_url="https://example.org/source",
               source_recording_id="recording-1", sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
               license="CC-BY-4.0")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"scenes": [row]}))
    return path, row


def fake_factory(clock, *, fail=False, duration=.2):
    instances = []

    class Backend:
        fps = 10.0

        def __init__(self, _checkpoint, **kwargs):
            self.actions_per_chunk = kwargs["chunk_frames"]
            self.kwargs = kwargs
            self.received = []
            self.closed = False
            instances.append(self)

        def load(self):
            clock.sleep(.5)

        def reset(self, path, seed):
            self.seed = seed
            clock.sleep(.3)

        def generate(self, rows):
            self.received.append(rows.copy())
            clock.sleep(duration)
            if fail:
                raise RuntimeError("injected model failure")
            return SimpleNamespace(frames=np.zeros((len(rows), 16, 16, 3), dtype=np.uint8),
                                   metrics={"total_s": duration, "generation_fps": len(rows) / duration})

        def provenance(self):
            return {"fake_backend": True, "seed": self.seed}

        def close(self):
            self.closed = True

    return Backend, instances


def settings(tmp_path, **kwargs):
    return replace(bench.Settings(output_dir=tmp_path / "results", modes=("neutral", "x+", "y-"),
                                  frames=16, video=False, max_wall_seconds=100,
                                  load_reserve_seconds=2, case_reserve_seconds=2,
                                  chunk_reserve_seconds=.3), **kwargs)


def results(config):
    return [json.loads(line) for line in (config.output_dir / "results.jsonl").read_text().splitlines()]


def test_manifest_verifies_provenance_and_hash(manifest):
    path, row = manifest
    scene = bench.load_scenes(path)[0]
    assert scene.path == path.parent / row["path"]
    assert scene.source_recording_id == "recording-1"
    (path.parent / row["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        bench.load_scenes(path)


@pytest.mark.parametrize("change", [{"license": ""}, {"source_recording_id": None},
                                     {"id": "../unsafe"}, {"source_url": "file:///secret"}])
def test_manifest_rejects_missing_or_unsafe_provenance(manifest, change):
    path, row = manifest
    path.write_text(json.dumps([{**row, **change}]))
    with pytest.raises(ValueError):
        bench.load_scenes(path)


def test_plan_is_read_only_and_counts_all_cases(manifest, tmp_path):
    scenes = bench.load_scenes(manifest[0])
    config = settings(tmp_path, seeds=(42, 7), decoders=("wan", "taehv"))
    plan = bench.plan(scenes, config)
    assert plan["planned_cases"] == 24
    assert plan["generated_frames_if_complete"] == 384
    assert plan["sampling_wait_seconds_if_complete"] == pytest.approx(31.2)
    assert not config.output_dir.exists()


def test_paired_actions_and_sample_to_ready_timing(manifest, tmp_path):
    clock = Clock()
    factory, instances = fake_factory(clock)
    config = settings(tmp_path)
    summary = bench.run_benchmark(bench.load_scenes(manifest[0]), config,
                                  backend_factory=factory, clock=clock, sleep=clock.sleep)
    rows = results(config)
    assert summary["status_counts"] == {"complete": 6, "failed": 0, "incomplete": 0, "skipped": 0}
    assert summary["generated_frames"] == 96
    assert all(backend.closed for backend in instances)
    for mode in config.modes:
        paired = [row for row in rows if row["mode"] == mode]
        assert len({row["pair_id"] for row in paired}) == 1
        assert len({row["action_sha256"] for row in paired}) == 1
        exported = [json.loads((config.output_dir / row["actions_file"]).read_text()) for row in paired]
        assert exported[0] == exported[1]
        actions = np.asarray(exported[0]["actions"])
        assert np.abs(actions[:, :3]).max() <= .04
        assert np.all(actions[:, 14:17] == 0)
    for row in rows:
        chunk = row["chunk_frames"]
        assert row["summary"]["median_latency_s"] == pytest.approx((chunk - 1) / 10 + .2)
        for block in row["chunks"]:
            assert np.diff(block["sampled_at_run_s"]) == pytest.approx([.1] * (chunk - 1))
            assert block["sample_to_ready_s"][-1] == pytest.approx(.2)
            assert block["generate_wall_s"] == pytest.approx(.2)
        assert (config.output_dir / row["contact_sheet_file"]).is_file()


def test_deadline_refuses_loading_and_accounts_every_skipped_case(manifest, tmp_path):
    clock = Clock()
    factory, instances = fake_factory(clock)
    config = settings(tmp_path, max_wall_seconds=3)
    summary = bench.run_benchmark(bench.load_scenes(manifest[0]), config,
                                  backend_factory=factory, clock=clock, sleep=clock.sleep)
    assert not instances
    assert summary["status_counts"]["skipped"] == 6
    assert all(row["reason"] == "wall_time_reserve" for row in results(config))


def test_slow_load_does_not_start_reset_after_case_reserve_expires(manifest, tmp_path):
    clock = Clock()
    factory, instances = fake_factory(clock)

    def slow_load(self):
        clock.sleep(4)

    def unexpected_reset(self, path, seed):
        pytest.fail("scene reset started after loading consumed the reserve")

    factory.load = slow_load
    factory.reset = unexpected_reset
    config = settings(tmp_path, modes=("neutral",), chunk_frames=(4,), max_wall_seconds=5)
    summary = bench.run_benchmark(bench.load_scenes(manifest[0]), config,
                                  backend_factory=factory, clock=clock, sleep=clock.sleep)
    assert summary["status_counts"]["skipped"] == 1
    assert results(config)[0]["reason"] == "wall_time_reserve_after_load"
    assert instances[0].closed


def test_after_first_summary_does_not_claim_steady_state():
    chunks = [dict(first_sample_to_ready_s=1.1, generate_wall_s=.4, frame_count=8),
              dict(first_sample_to_ready_s=1.2, generate_wall_s=.5, frame_count=8)]
    summary = bench.summarize_chunks(chunks)
    assert summary["median_latency_s"] == pytest.approx(1.15)
    assert summary["after_first_chunk_median_latency_s"] == pytest.approx(1.2)
    assert summary["after_first_chunk_count"] == 1
    assert not any("steady" in key for key in summary)
    assert bench.summarize_chunks(chunks[:1])["after_first_chunk_median_latency_s"] is None


def test_legacy_timing_includes_old_trailing_wait(manifest, tmp_path):
    clock = Clock()
    factory, _ = fake_factory(clock)
    config = settings(tmp_path, modes=("neutral",), chunk_frames=(4,), sampling_policy="legacy")
    bench.run_benchmark(bench.load_scenes(manifest[0]), config,
                        backend_factory=factory, clock=clock, sleep=clock.sleep)
    row = results(config)[0]
    assert row["summary"]["median_latency_s"] == pytest.approx(.6)
    assert row["chunks"][0]["sample_to_ready_s"][-1] == pytest.approx(.3)


def test_case_limit_keeps_unattempted_cases_visible(manifest, tmp_path):
    clock = Clock()
    factory, _ = fake_factory(clock)
    config = settings(tmp_path, max_cases=1)
    summary = bench.run_benchmark(bench.load_scenes(manifest[0]), config,
                                  backend_factory=factory, clock=clock, sleep=clock.sleep)
    assert summary["status_counts"]["complete"] == 1
    assert summary["status_counts"]["skipped"] == 5
    assert all(row["reason"] == "max_cases" for row in results(config)[1:])


def test_model_failure_is_recorded_and_backend_closed(manifest, tmp_path):
    clock = Clock()
    factory, instances = fake_factory(clock, fail=True)
    config = settings(tmp_path, modes=("x+",), chunk_frames=(4,))
    summary = bench.run_benchmark(bench.load_scenes(manifest[0]), config,
                                  backend_factory=factory, clock=clock, sleep=clock.sleep)
    assert summary["status_counts"]["failed"] == 1
    row = results(config)[0]
    assert row["stage"] == "generate"
    assert "injected model failure" in row["error"]
    assert row["summary"] == {}
    assert row["failed_chunk"]["frame_offset"] == 0
    assert len(row["failed_chunk"]["sampled_at_run_s"]) == 4
    assert instances[0].closed
    assert (config.output_dir / row["actions_file"]).is_file()


def test_load_failure_counts_all_configuration_cases_without_retrying(manifest, tmp_path):
    clock = Clock()
    factory, instances = fake_factory(clock)

    def fail_load(self):
        raise RuntimeError("injected load failure")

    factory.load = fail_load
    config = settings(tmp_path)
    summary = bench.run_benchmark(bench.load_scenes(manifest[0]), config,
                                  backend_factory=factory, clock=clock, sleep=clock.sleep)
    assert summary["status_counts"]["failed"] == 6
    assert not summary["selected_cases_complete"]
    assert len(instances) == 2
    assert all(instance.closed for instance in instances)
    assert all(row["stage"] == "load" for row in results(config))


def test_inflight_call_overrun_stops_next_chunk_and_keeps_partial_result(manifest, tmp_path):
    clock = Clock()
    factory, instances = fake_factory(clock, duration=4)
    config = settings(tmp_path, modes=("x+",), chunk_frames=(4,), max_wall_seconds=5)
    summary = bench.run_benchmark(bench.load_scenes(manifest[0]), config,
                                  backend_factory=factory, clock=clock, sleep=clock.sleep)
    assert summary["status_counts"]["incomplete"] == 1
    row = results(config)[0]
    assert row["summary"]["generated_frames"] == 4
    assert len(instances[0].received) == 1


@pytest.mark.parametrize("overrides", [{"frames": 12}, {"displacement": .05}, {"max_wall_seconds": float("nan")},
                                       {"seeds": (True,)}, {"chunk_frames": (8, 8)}])
def test_invalid_experiment_rejected_before_running(tmp_path, overrides):
    with pytest.raises(ValueError):
        bench.validate(settings(tmp_path, **overrides))
