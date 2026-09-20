"""The identity of a run: the commit and the hashes of its inputs."""

import hashlib
import subprocess
from collections.abc import Iterable
from pathlib import Path

UNKNOWN_SHA = "unknown"
_BLOCK_BYTES = 1 << 20


def sha256_text(text: str) -> str:
    """Return the SHA-256 of a text, in hexadecimal."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_files(paths: Iterable[Path]) -> str:
    """Return one SHA-256 for a set of files.

    The files are read in the order of their names, and each name goes into
    the hash, so a renamed file or a moved line changes the result.
    """
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            # An audio file can be large, so it is read in blocks.
            while block := handle.read(_BLOCK_BYTES):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    """Return a seed that is the same in each process, unlike the built-in hash."""
    joined = "\x1f".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(joined.encode("utf-8")).digest()[:8], "big")


def short_token(*parts: object) -> str:
    """Return eight hexadecimal characters that identify the parts."""
    return sha256_text("\x1f".join(str(part) for part in parts))[:8]


def _git(root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_sha(root: Path) -> str:
    """Return the commit of the checkout at `root`, or `unknown`."""
    sha = _git(root, "rev-parse", "HEAD")
    return sha if sha else UNKNOWN_SHA


def git_dirty(root: Path) -> bool:
    """Return True if the checkout has changes that are not committed."""
    status = _git(root, "status", "--porcelain")
    return bool(status)
