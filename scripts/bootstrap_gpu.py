#!/usr/bin/env python3
"""Install the pinned CUDA runtime; run only on a compatible Linux GPU host."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess

UPSTREAM_URL = "https://github.com/isaac-for-healthcare/cosmos3-h-dreams.git"
UPSTREAM_REVISION = "fb18792d8af680a713801dea7688919e46329810"
IMAGEIO_FFMPEG_VERSION = "0.6.0"


def cache_environment(asset_dir):
    root = Path(asset_dir).expanduser().resolve()
    return {
        "HF_HOME": str(root / "cache/huggingface"),
        "HF_HUB_CACHE": str(root / "cache/huggingface/hub"),
        "HUGGINGFACE_HUB_CACHE": str(root / "cache/huggingface/hub"),
        "TORCH_HOME": str(root / "cache/torch"),
        "UV_CACHE_DIR": str(root / "cache/uv"),
        "UV_PYTHON_INSTALL_DIR": str(root / "cache/python"),
        "XDG_CACHE_HOME": str(root / "cache/xdg"),
        "FLASHDREAMS_CACHE_DIR": str(root / "cache/flashdreams"),
        "TRITON_CACHE_DIR": str(root / "cache/triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(root / "cache/torchinductor"),
        "TMPDIR": str(root / "tmp"),
    }


def install_persistent_ffmpeg(python, asset_dir, env):
    """Keep the video encoder with the mounted Python environment on resume."""
    subprocess.run(["uv", "pip", "install", "--python", str(python), "--no-deps",
                    f"imageio-ffmpeg=={IMAGEIO_FFMPEG_VERSION}"], env=env, check=True)
    # Ignore an inherited override so it cannot select an ephemeral system file.
    encoder = Path(subprocess.check_output([
        str(python), "-c", "import os; os.environ.pop('IMAGEIO_FFMPEG_EXE', None); "
        "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
        env=env, text=True).strip()).resolve()
    environment_root = Path(python).absolute().parent.parent.resolve()
    if (not encoder.is_file() or not os.access(encoder, os.X_OK)
            or not encoder.is_relative_to(environment_root)):
        raise RuntimeError("The FFmpeg wheel must provide its encoder inside the mounted Python environment")
    subprocess.run([str(encoder), "-version"], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    binary_dir = Path(asset_dir).expanduser().resolve() / "bin"
    binary_dir.mkdir(parents=True, exist_ok=True)
    temporary = binary_dir / f".ffmpeg-{os.getpid()}"
    try:
        temporary.symlink_to(encoder)
        temporary.replace(binary_dir / "ffmpeg")
    finally:
        temporary.unlink(missing_ok=True)
    return binary_dir / "ffmpeg"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True,
                        help="Large mounted disk for all downloads and caches")
    parser.add_argument("--upstream-root", default=None)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    upstream = Path(args.upstream_root).expanduser().resolve() if args.upstream_root else repo / ".vendor/cosmos3-h-dreams"
    if platform.system() != "Linux":
        parser.error("GPU installation requires Linux; use the CPU preview on other systems")
    for command in ("git", "uv", "nvidia-smi"):
        if not shutil.which(command):
            parser.error(f"Missing {command}; install the documented GPU prerequisites first")
    env = os.environ.copy()
    env.update(cache_environment(args.asset_dir))
    for path in cache_environment(args.asset_dir).values():
        Path(path).mkdir(parents=True, exist_ok=True)
    if not upstream.exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", UPSTREAM_URL, str(upstream)], check=True, env=env)
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True)
    if revision != UPSTREAM_REVISION:
        if status:
            parser.error("Upstream has local edits; refusing to change its revision")
        subprocess.run(["git", "-C", str(upstream), "fetch", "origin", UPSTREAM_REVISION], check=True, env=env)
        subprocess.run(["git", "-C", str(upstream), "checkout", "--detach", UPSTREAM_REVISION], check=True, env=env)
    elif status:
        parser.error("Pinned upstream has local edits; use a clean checkout")
    subprocess.run(["uv", "sync", "--package", "flashdreams-cosmos3-h-dreams", "--locked"],
                   cwd=upstream, env=env, check=True)
    python = upstream / ".venv/bin/python"
    subprocess.run(["uv", "pip", "install", "--python", str(python), "--no-deps", "-e", str(repo)],
                   env=env, check=True)
    encoder = install_persistent_ffmpeg(python, args.asset_dir, env)
    print(f"Installed CholeScope into {python}")
    print(f"Installed the persistent video encoder at {encoder}")
    print("Use scripts/launch_gpu.py so model and compiler caches stay on the asset disk.")


if __name__ == "__main__":
    main()
