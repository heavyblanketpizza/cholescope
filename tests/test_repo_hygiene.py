"""Publication guard regressions using synthetic values, never live secrets."""
import base64
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("repo_hygiene", ROOT / "scripts/check_repo_hygiene.py")
hygiene = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hygiene)


def token(kind):
    # Assemble fixtures so the repository never contains token-shaped literals.
    body = "Ab3Cd4Ef5Gh6Ij7Kl8Mn9Op0Qr1St2Uv3Wx4Yz5"
    if kind == "huggingface":
        return "hf_" + body
    if kind == "google":
        return "AIza" + body[:35]
    if kind == "runpod-api":
        return "rpa_" + body
    if kind == "runpod-storage":
        return "rps_" + body
    if kind == "url":
        return "https://operator:" + body + "@example.invalid/private"
    if kind == "url-empty-user":
        return "redis://:" + body + "@example.invalid/0"
    if kind == "jwt":
        def encode(value):
            return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
        return ".".join((encode({"alg": "HS256", "typ": "JWT"}),
                         encode({"sub": "synthetic", "exp": 1}), body))
    raise AssertionError("Unknown synthetic token kind")


@pytest.mark.parametrize("kind", ["huggingface", "google", "runpod-api", "runpod-storage", "url", "url-empty-user", "jwt"])
def test_additional_secret_signatures_report_location_only(kind):
    content = ("ordinary first line\nvalue = " + token(kind) + "\n").encode()
    findings = list(hygiene.findings("src/example.py", content))
    assert (2, "secret-or-private-key") in findings
    assert all(token(kind) not in str(finding) for finding in findings)


@pytest.mark.parametrize("prefix", [b"GIF89a\0", b"\xff\xfe"])
def test_binary_media_allowlist_does_not_bypass_embedded_token_scan(prefix):
    content = prefix + b"comment\n" + token("huggingface").encode()
    findings = list(hygiene.findings("assets/cholescope-comparison.gif", content))
    assert (2, "secret-or-private-key") in findings
    assert not any(category == "binary-not-in-public-allowlist" for _, category in findings)
    unlisted = list(hygiene.findings("src/example.dat", content))
    assert (1, "binary-not-in-public-allowlist") in unlisted
    assert (2, "secret-or-private-key") in unlisted


def test_documentation_placeholders_and_scanner_source_do_not_match():
    placeholders = [
        "hf_<your-token>", "rpa_<your-key>", "rps_<your-key>",
        "https://user:${PASSWORD}@example.invalid/resource",
        "https://user:<password>@example.invalid/resource",
        "ssh://git@example.invalid/project.git",
        "eyJ...payload...signature", "AIza...",
    ]
    contents = ["\n".join(placeholders).encode(), Path(hygiene.__file__).read_bytes(), Path(__file__).read_bytes()]
    for content in contents:
        assert not any(category == "secret-or-private-key"
                       for _, category in hygiene.findings("src/example.py", content))


@pytest.mark.parametrize("name", [
    "config/.aws/credentials", "config/.runpod/config.toml", "config/.runpod.yaml",
    ".secrets.toml", ".credentials.json", "id_ecdsa_sk", "id_ed25519_sk.pub",
    "secrets/development.txt", "credentials/development.txt",
    "runpod-api-key", "known_hosts",
])
def test_sensitive_config_names_are_rejected_even_if_force_added(name):
    assert "credential-file" in set(hygiene.filename_categories(name))


def test_cli_never_prints_matched_secret(monkeypatch, capsys):
    value = token("huggingface")
    monkeypatch.setattr(sys, "argv", ["check_repo_hygiene.py"])
    monkeypatch.setattr(hygiene.subprocess, "check_output", lambda *args, **kwargs: str(ROOT))
    monkeypatch.setattr(hygiene, "repository_files", lambda root, staged: [("src/example.py", value.encode())])
    assert hygiene.main() == 1
    output = capsys.readouterr()
    assert value not in output.out + output.err
    assert "src/example.py:1: secret-or-private-key" in output.out
