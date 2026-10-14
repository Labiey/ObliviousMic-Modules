"""Audit the Git index before a local commit. Print categories, never secret values."""

import re
import subprocess
from pathlib import PurePosixPath


def audit() -> list[str]:
    names = subprocess.check_output(["git", "diff", "--cached", "--name-only", "-z"])
    findings = []
    banned_parts = {
        "tmps",
        "obmic-context",
        "recordings",
        "profiles",
        "logs",
        "__pycache__",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
    }
    banned_suffixes = {
        ".wav",
        ".flac",
        ".mp3",
        ".ogg",
        ".m4a",
        ".pcm",
        ".raw",
        ".npy",
        ".npz",
        ".onnx",
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".bin",
        ".log",
        ".tmp",
        ".partial",
        ".pyc",
        ".pem",
        ".key",
        ".zip",
    }
    credentials = re.compile(
        rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"
        rb"|sk-(?:proj-)?[A-Za-z0-9_-]{32,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
    )
    for raw in names.split(b"\0"):
        if not raw:
            continue
        name = raw.decode("utf-8")
        path = PurePosixPath(name)
        if (
            any(part.lower() in banned_parts for part in path.parts)
            or path.suffix.lower() in banned_suffixes
            or path.name.lower() in {"agents.md", ".env"}
            or path.name.startswith((".env.", "config.local."))
        ):
            findings.append(f"{name}: private asset/cache/temporary file")
        result = subprocess.run(["git", "show", f":{name}"], capture_output=True)
        if result.returncode:  # staged deletion
            continue
        if b"\0" in result.stdout or credentials.search(result.stdout):
            findings.append(f"{name}: binary asset or possible credential")
    return findings


if __name__ == "__main__":
    found = audit()
    print("\n".join(found) if found else "PASS: staged source contains no prohibited assets")
    raise SystemExit(bool(found))
