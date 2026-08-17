"""The fitness function: how many candidate metered call sites can we price?

Run it::

    python -m corpus.score                    # score every pinned repo
    python -m corpus.score --repo aider       # substring match on repo name
    python -m corpus.score --sample 25        # print candidates to go read
    python -m corpus.score --json report.json

Until the analyser exists this reports ``0 resolved``, and that is the point: the
denominator is fixed and reproducible *before* anything has an incentive to make
the numerator look good.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from corpus.detect import Candidate, find_candidates
from corpus.fetch import FetchError, Repo, ensure_checkout, python_files

MANIFEST = Path(__file__).parent / "manifest.toml"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / ".corpus-cache"


# --------------------------------------------------------------------------
# Resolver plug point
#
# Build step 0 has no analyser, so nothing resolves. Steps 1 and 2 supply a real
# resolver and the numerator starts moving. The harness never imports analyser
# internals — only this one narrow interface.
# --------------------------------------------------------------------------


class NullResolver:
    name = "none (no analyser yet — build step 0)"

    def resolve(self, candidate: Candidate, source: str) -> str | None:
        return None


def load_resolver() -> object:
    try:
        from chihuahuabot.resolve import CorpusResolver
    except ImportError:
        return NullResolver()
    return CorpusResolver()


# --------------------------------------------------------------------------


@dataclass
class RepoScore:
    name: str
    commit: str
    kind: str
    files: int = 0
    unparseable: int = 0
    candidates: int = 0
    resolved: int = 0
    by_rule: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    samples: list[dict[str, object]] = field(default_factory=list)

    @property
    def coverage(self) -> float | None:
        if self.candidates == 0:
            return None
        return self.resolved / self.candidates


def load_manifest(path: Path = MANIFEST) -> tuple[list[Repo], frozenset[str]]:
    data = tomllib.loads(path.read_text())
    exclude = frozenset(data.get("defaults", {}).get("exclude_dirs", []))
    repos = [
        Repo(
            name=entry["name"],
            url=entry["url"],
            commit=entry["commit"],
            kind=entry.get("kind", "unknown"),
            notes=entry.get("notes", ""),
        )
        for entry in data.get("repo", [])
    ]
    return repos, exclude


def score_repo(
    repo: Repo,
    cache_root: Path,
    exclude: frozenset[str],
    resolver: object,
    sample_size: int,
) -> RepoScore:
    score = RepoScore(name=repo.name, commit=repo.commit, kind=repo.kind)
    try:
        root = ensure_checkout(repo, cache_root)
    except FetchError as exc:
        score.error = str(exc).splitlines()[-1] if str(exc) else "fetch failed"
        return score

    rules: Counter[str] = Counter()
    pool: list[tuple[Candidate, str]] = []

    for path in python_files(root, exclude):
        score.files += 1
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            candidates = find_candidates(source, str(path.relative_to(root)))
        except (SyntaxError, ValueError, OSError):
            # Python 2 files, generated stubs, unreadable blobs. Counted, not
            # hidden: an unparseable file is a call site we cannot see.
            score.unparseable += 1
            continue

        for candidate in candidates:
            score.candidates += 1
            rules["+".join(candidate.rules)] += 1
            if resolver.resolve(candidate, source) is not None:
                score.resolved += 1
            pool.append((candidate, source))

    score.by_rule = dict(sorted(rules.items()))

    if sample_size and pool:
        for candidate, source in random.sample(pool, min(sample_size, len(pool))):
            lines = source.splitlines()
            snippet = lines[candidate.lineno - 1].strip() if candidate.lineno <= len(lines) else ""
            score.samples.append(
                {
                    "file": candidate.file,
                    "lineno": candidate.lineno,
                    "call": candidate.dotted,
                    "rules": list(candidate.rules),
                    "source": snippet[:160],
                }
            )

    return score


def print_table(scores: list[RepoScore], resolver_name: str) -> None:
    header = f"{'repo':<34}{'files':>7}{'bad':>5}{'cand':>7}{'resolved':>10}{'cov':>7}"
    print(header)
    print("-" * len(header))

    for score in scores:
        if score.error:
            print(f"{score.name:<34}{'FETCH FAILED':>36}")
            print(f"{'':<34}{score.error[:60]}")
            continue
        coverage = score.coverage
        shown = f"{coverage:.0%}" if coverage is not None else "  -"
        print(
            f"{score.name:<34}{score.files:>7}{score.unparseable:>5}"
            f"{score.candidates:>7}{score.resolved:>10}{shown:>7}"
        )

    ok = [s for s in scores if not s.error]
    files = sum(s.files for s in ok)
    bad = sum(s.unparseable for s in ok)
    candidates = sum(s.candidates for s in ok)
    resolved = sum(s.resolved for s in ok)
    total_cov = f"{resolved / candidates:.0%}" if candidates else "  -"

    print("-" * len(header))
    print(f"{'TOTAL':<34}{files:>7}{bad:>5}{candidates:>7}{resolved:>10}{total_cov:>7}")
    print(f"\nresolver: {resolver_name}")

    rules: Counter[str] = Counter()
    for score in ok:
        rules.update(score.by_rule)
    if rules:
        breakdown = "  ".join(f"{rule}={count}" for rule, count in sorted(rules.items()))
        print(f"candidates by rule: {breakdown}")


def print_samples(scores: list[RepoScore]) -> None:
    print("\n--- sampled candidates (go read these) ---")
    for score in scores:
        if not score.samples:
            continue
        print(f"\n{score.name}")
        for sample in score.samples:
            rules = ",".join(sample["rules"])
            print(f"  {sample['file']}:{sample['lineno']}  [{rules}]  {sample['call']}")
            print(f"      {sample['source']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score ChihuahuaBot coverage over the corpus.")
    parser.add_argument("--repo", help="substring match; score only matching repos")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="checkout cache dir")
    parser.add_argument("--sample", type=int, default=0, help="print N sampled candidates per repo")
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    repos, exclude = load_manifest()
    if args.repo:
        needle = args.repo.lower()
        repos = [r for r in repos if needle in r.name.lower()]
        if not repos:
            print(f"no repo matching {args.repo!r}", file=sys.stderr)
            return 2

    resolver = load_resolver()
    scores: list[RepoScore] = []
    for repo in repos:
        print(f"scoring {repo.name} ...", file=sys.stderr)
        scores.append(score_repo(repo, args.cache, exclude, resolver, args.sample))

    print()
    print_table(scores, getattr(resolver, "name", type(resolver).__name__))
    if args.sample:
        print_samples(scores)

    if args.json:
        args.json.write_text(json.dumps([asdict(s) for s in scores], indent=2))
        print(f"\nreport written to {args.json}")

    return 1 if any(s.error for s in scores) else 0


if __name__ == "__main__":
    raise SystemExit(main())
