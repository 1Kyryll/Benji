"""Fetch corpus repositories at their pinned commits.

Blobless partial clones (``--filter=blob:none``) keep this affordable: full
history metadata, file contents fetched only for the commit we check out. A
successful checkout drops a marker file so repeated scoring runs are offline and
instant.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

MARKER = ".benji-checkout-ok"


@dataclass(frozen=True)
class Repo:
    name: str
    url: str
    commit: str
    kind: str
    notes: str = ""

    @property
    def slug(self) -> str:
        return self.name.replace("/", "__")


class FetchError(RuntimeError):
    pass


def _run(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FetchError(f"{' '.join(args)}\n{result.stderr.strip()}")
    return result.stdout


def checkout_dir(cache_root: Path, repo: Repo) -> Path:
    return cache_root / f"{repo.slug}@{repo.commit[:12]}"


def ensure_checkout(repo: Repo, cache_root: Path) -> Path:
    """Return a directory containing ``repo`` at its pinned commit.

    Idempotent. A partially-fetched directory from an interrupted run is deleted
    and retried rather than scored, because a half-checked-out tree would quietly
    lower the candidate count.
    """
    target = checkout_dir(cache_root, repo)
    if (target / MARKER).exists():
        return target

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--quiet",
            repo.url,
            str(target),
        ]
    )
    _run(["git", "checkout", "--quiet", repo.commit], cwd=target)

    actual = _run(["git", "rev-parse", "HEAD"], cwd=target).strip()
    if actual != repo.commit:
        raise FetchError(f"{repo.name}: expected {repo.commit}, checked out {actual}")

    (target / MARKER).write_text(repo.commit + "\n")
    return target


def python_files(root: Path, exclude_dirs: frozenset[str]) -> list[Path]:
    """Every ``.py`` file under ``root``, minus excluded directory segments."""
    found: list[Path] = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if any(part in exclude_dirs for part in relative.parts[:-1]):
            continue
        found.append(path)
    return sorted(found)
