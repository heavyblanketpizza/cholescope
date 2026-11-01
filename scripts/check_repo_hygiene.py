#!/usr/bin/env python3
"""Check publishable repository files for common data and privacy mistakes.

Run ``python3 scripts/check_repo_hygiene.py`` to inspect tracked and unignored
worktree files, honoring local deletions. Run with ``--staged`` before a commit
to inspect the complete index, including its staged contents and deletions.
The check reports locations and categories without reproducing matched values.
It is a practical guardrail, not a comprehensive secret or personal-data audit;
it does not inspect Git history or ignored local files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


PUBLIC_MARKDOWN = {"README.md", "THIRD_PARTY.md"}
PUBLIC_MEDIA = {"assets/cholescope-comparison.gif"}
PUBLIC_ARTIFACTS = {
    "src/cholescope/static/index.html", "src/cholescope/static/student.html",
}
ARTIFACT_SUFFIXES = {".json", ".json5", ".jsonc", ".html", ".htm", ".ipynb"}
FONT_SUFFIXES = {".ttf", ".otf", ".woff", ".woff2", ".eot"}
INTERNAL_DIRECTORIES = {
    "research", "research_notes", "notes", "private", ".local", ".agents", ".codex", ".claude", ".idea", ".vscode",
}
LOCAL_DIRECTORIES = {
    "__pycache__", ".venv", "venv", ".vendor", ".runtime", "node_modules", "build", "dist",
    ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox",
    ".ipynb_checkpoints", "htmlcov",
}
CREDENTIAL_DIRECTORIES = {
    ".ssh", ".aws", ".azure", ".kube", ".gnupg", ".runpod",
    "credentials", "secrets", ".credentials", ".secrets",
}
DATA_DIRECTORIES = {
    "data", "data_local", "dataset", "datasets", "outputs", "frames", "archives",
    "models", "checkpoints",
}
DATA_SUFFIXES = {
    ".csv", ".tsv", ".jsonl", ".parquet", ".arrow", ".feather",
    ".npy", ".npz", ".h5", ".hdf5", ".pt", ".pth", ".safetensors", ".gguf",
    ".bin", ".ckpt", ".onnx", ".pkl", ".pickle", ".joblib",
    ".sqlite", ".sqlite3", ".db", ".log",
}
MEDIA_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff", ".bmp",
    ".dcm", ".nii", ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".wav", ".mp3", ".flac", ".pdf", ".docx", ".xlsx", ".pptx",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tgz",
}
CREDENTIAL_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
SSH_KEY_NAMES = {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "id_ecdsa_sk", "id_ed25519_sk"}
CREDENTIAL_NAMES = {
    "credentials", "secrets", ".credentials", ".secrets", ".netrc", ".npmrc", ".pypirc", ".runpod.yaml",
    "runpod-api-key", "known_hosts",
}
EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})")
MACHINE_PATH = re.compile(r"/(?:Users|home|Volumes)/([^/\s\"'`]+)")
GENERIC_PATH_COMPONENTS = {
    "user", "username", "example", "test", "synthetic", "redacted",
    "<user>", "<username>", "<volume>", "<dataset>",
}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{20,}\b"),
    re.compile(r"\brp[as]_[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(r"(?<![A-Za-z0-9_\-])AIza[A-Za-z0-9_\-]{35}(?![A-Za-z0-9_\-])"),
    # Signed compact JWTs: JSON header/payload and a nontrivial signature.
    re.compile(r"\beyJ[A-Za-z0-9_\-]{7,}\.eyJ[A-Za-z0-9_\-]{7,}\.[A-Za-z0-9_\-]{20,}(?![A-Za-z0-9_\-])"),
    # Require explicit user:password URL credentials; ordinary git@host URLs
    # and environment-variable/angle-bracket examples do not match.
    re.compile(r"\b(?:https?|ftp|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|rediss?)://"
               r"[^\s/:@<>\"'`$\\{}]*:[^\s/@<>\"'`$\\{}]+@[^\s/<>\"'`]+", re.IGNORECASE),
)


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def repository_files(root: Path, staged: bool):
    if staged:
        entries = git(root, "ls-files", "--stage", "-z").split(b"\0")
        for entry in entries:
            if not entry:
                continue
            metadata, filename = entry.split(b"\t", 1)
            mode, blob, stage = metadata.split()
            if stage != b"0":
                raise ValueError("Resolve index conflicts before checking staged files.")
            name = os.fsdecode(filename)
            if mode == b"160000":
                raise ValueError(f"Cannot inspect submodule contents: {name}")
            yield name, git(root, "cat-file", "blob", blob.decode("ascii"))
        return

    names = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    for filename in sorted(set(names.split(b"\0")) - {b""}):
        name = os.fsdecode(filename)
        path = root / name
        if path.is_symlink():
            yield name, os.fsencode(os.readlink(path))
        elif path.is_file():
            yield name, path.read_bytes()


def filename_categories(name: str):
    path = PurePosixPath(name)
    directories = {part.lower() for part in path.parts[:-1]}
    basename = path.name.lower()
    suffix = path.suffix.lower()
    if directories & INTERNAL_DIRECTORIES or basename in {"plan.md", "verification.md", "agents.md"}:
        yield "internal-notes"
    elif suffix == ".md" and name not in PUBLIC_MARKDOWN:
        yield "markdown-not-in-public-allowlist"
    if (directories & LOCAL_DIRECTORIES or any(part.endswith(".egg-info") for part in directories)
            or basename in {".ds_store", ".coverage"} or basename.startswith(".coverage.")
            or suffix in {".pyc", ".pyo", ".pyd"}):
        yield "local-development-file"
    if suffix in FONT_SUFFIXES:
        yield "vendored-font"
    if suffix in ARTIFACT_SUFFIXES and name not in PUBLIC_ARTIFACTS:
        yield "artifact-not-in-public-allowlist"
    if directories & DATA_DIRECTORIES or basename in {"source_inventory.json", "source_manifest.json"} or suffix in DATA_SUFFIXES:
        yield "dataset-or-generated-data"
    if suffix in MEDIA_SUFFIXES and name not in PUBLIC_MEDIA:
        yield "source-media-or-archive"
    env_file = basename == ".env" or basename.startswith(".env.")
    credential_variant = basename.startswith(("credentials.", "secrets.", ".credentials.", ".secrets."))
    ssh_key = any(basename == key or basename.startswith(key + ".") for key in SSH_KEY_NAMES)
    if (directories & CREDENTIAL_DIRECTORIES or env_file or credential_variant or ssh_key
            or basename in CREDENTIAL_NAMES or suffix in CREDENTIAL_SUFFIXES):
        yield "credential-file"


def allowed_email_domain(domain: str) -> bool:
    domain = domain.lower()
    allowed = ("example.com", "example.org", "example.invalid", "users.noreply.github.com")
    return any(domain == suffix or domain.endswith("." + suffix) for suffix in allowed)


def findings(name: str, content: bytes):
    for category in filename_categories(name):
        yield 1, category
    binary = b"\0" in content
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        binary = True
        text = content.decode("utf-8", errors="replace")
    if binary and name not in PUBLIC_MEDIA:
        yield 1, "binary-not-in-public-allowlist"
    for number, line in enumerate(text.splitlines(), 1):
        if not binary and any(not allowed_email_domain(match.group(1)) for match in EMAIL.finditer(line)):
            yield number, "personal-email"
        if not binary and any(match.group(1).lower() not in GENERIC_PATH_COMPONENTS for match in MACHINE_PATH.finditer(line)):
            yield number, "machine-specific-path"
        # Binary allowlisting only permits the format. Embedded ASCII token
        # signatures (for example in GIF comments) still require inspection.
        if any(pattern.search(line) for pattern in SECRET_PATTERNS):
            yield number, "secret-or-private-key"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Inspect the complete Git index instead of worktree files.")
    args = parser.parse_args()
    try:
        root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
        count = 0
        for name, content in repository_files(root, args.staged):
            for line, category in findings(name, content):
                # Escape unusual filename characters so each finding stays on one line.
                safe_name = name.encode("unicode_escape").decode("ascii")
                print(f"{safe_name}:{line}: {category}")
                count += 1
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Repository hygiene check could not finish: {error}", file=sys.stderr)
        return 2
    if count:
        print(f"Repository hygiene check failed: {count} finding(s).")
        return 1
    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
