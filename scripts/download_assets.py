#!/usr/bin/env python3
"""Fetch the 2-step model and its small runtime dependencies to one asset disk.

Requires huggingface_hub in the active Python environment. The file allowlists
deliberately exclude the 4-step checkpoint and Qwen/Wan base-model weights.
The saved manifest records exact resolved Hub commits and local cache paths.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bootstrap_gpu import cache_environment

MODEL_ID = "nvidia/Cosmos3-H-Dreams"
MODEL_FILE = "checkpoints/exp15/c3-h-d_2-step_model_bf16.safetensors"
DEPENDENCIES = {
    "Wan-AI/Wan2.2-TI2V-5B": ["Wan2.2_VAE.pth"],
    "lightx2v/Autoencoders": ["taew2_2.pth"],
    "Qwen/Qwen3-VL-8B-Instruct": ["config.json", "tokenizer.json", "tokenizer_config.json",
                                 "vocab.json", "merges.txt", "chat_template.json",
                                 "chat_template.jinja", "added_tokens.json", "special_tokens_map.json"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dependencies-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print allowlists without network or downloads")
    args = parser.parse_args()
    selections = dict(DEPENDENCIES)
    if not args.dependencies_only:
        selections = {MODEL_ID: [MODEL_FILE], **selections}
    if args.dry_run:
        print(json.dumps(selections, indent=2))
        return
    root = Path(args.asset_dir).expanduser().resolve()
    repo = Path(__file__).resolve().parent.parent
    if root.is_relative_to(repo):
        parser.error("Choose an asset disk outside the repository")
    os.environ.update(cache_environment(root))
    for value in cache_environment(root).values():
        Path(value).mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    manifest_path = root / "asset-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema_version": 1, "repositories": {}}
    for repo_id, allowlist in selections.items():
        prior = manifest["repositories"].get(repo_id)
        revision = prior["revision"] if prior else args.model_revision if repo_id == MODEL_ID else "main"
        info = api.model_info(repo_id, revision=revision)
        available = {s.rfilename for s in info.siblings}
        filenames = [name for name in allowlist if name in available]
        required = set(allowlist if repo_id != "Qwen/Qwen3-VL-8B-Instruct" else ["tokenizer.json", "tokenizer_config.json"])
        if not required.issubset(available):
            parser.error(f"Required files missing from {repo_id}: {sorted(required - available)}")
        filenames += sorted(name for name in available if "/" not in name and
                            (name == "README.md" or name.upper().startswith(("LICENSE", "NOTICE"))))
        files = {}
        for filename in filenames:
            # Resolve once, then pin every file. Hugging Face verifies each
            # object's downloaded size and stores blobs by content identity.
            path = hf_hub_download(repo_id, filename, revision=info.sha)
            files[filename] = {"path": path, "bytes": Path(path).stat().st_size}
            print(f"Ready: {repo_id}/{filename}", flush=True)
        # Upstream defaults to main for these dependency URLs. Point the local
        # main cache ref at the revision recorded here so offline startup uses
        # precisely the prefetched files. Online upstream may refresh this ref.
        cache_repo = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + repo_id.replace("/", "--"))
        refs = cache_repo / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(info.sha)
        manifest["repositories"][repo_id] = {"revision": info.sha, "files": files}
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
