"""`chihuahuabot diff <base> <head>` — every layer, assembled.

Up to here each stage has been exercised on its own. This is where they meet:
two commits in, one PR comment out.

The shape of the run follows the design's one non-obvious rule — **changed files
are not the same set as affected call sites**. Editing a wrapper changes the cost
of every site that reaches it, in files the diff never touches, so both commits
are indexed in full rather than only where the diff landed. Anything less would
report zero impact on the most expensive pull request of the quarter.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from chihuahuabot.candidates import count_candidates
from chihuahuabot.estimate import (
    Factor,
    Range,
    SiteEstimate,
    aggregate,
    cost_range,
    dominant,
    input_range,
    output_range,
)
from chihuahuabot.extract import CallSite
from chihuahuabot.frequency import Config, ConfigFrequencySource, propagate
from chihuahuabot.identity import match
from chihuahuabot.index import RepoIndex
from chihuahuabot.multiplicity import apply_declared, multiplicities
from chihuahuabot.pricing import PriceTable
from chihuahuabot.render import BlastRadius, Coverage, Report, SiteChange, render
from chihuahuabot.tokens import estimate_input

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "site-packages", "build", "dist"}


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def export_tree(ref: str, destination: Path, repo: Path) -> Path:
    """Materialise a commit's tree without disturbing the working copy.

    `git archive` in one pass rather than `git show` per file: a repository the
    size of onyx has 1,885 Python files, and a subprocess each is a minute of
    nothing happening.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination.with_suffix(".tar")
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", ref], cwd=repo, stdout=handle, check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")
    archive.unlink()
    return destination


def python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if not any(part in SKIP_DIRS for part in path.relative_to(root).parts[:-1])
    )


@dataclass
class Analysis:
    """One commit, fully indexed."""

    index: RepoIndex
    sites: list[CallSite]
    candidates: int

    @classmethod
    def of(cls, root: Path) -> Analysis:
        files = python_files(root)
        index = RepoIndex.build(root, files)
        sites: list[CallSite] = []
        for relative in sorted(index.sources):
            sites.extend(index.call_sites(relative))
        candidates = sum(count_candidates(source, name) for name, source in index.sources.items())
        return cls(index=index, sites=sites, candidates=candidates)


def function_key(index: RepoIndex, site: CallSite) -> str | None:
    for key, info in index.functions.items():
        if info.file == site.file and info.qualname == site.qualname:
            return key
    return None


def estimate_site(
    site: CallSite,
    analysis: Analysis,
    table: PriceTable,
    config: Config,
    source_of_frequency,
) -> tuple[SiteEstimate, tuple[Factor, ...], str | None]:
    """Assemble one call site's three layers. Any of them may be unknown."""
    factors: list[Factor] = []

    price = table.lookup(site.provider, site.model)
    per_call: Range | None = None
    if price is not None:
        tokens = estimate_input(site.content, site.model)
        ceiling = dict(site.literals).get("max_tokens")
        inputs = input_range(tokens.static, tokens.unknown_slots)
        outputs = output_range(ceiling if isinstance(ceiling, int) else None)
        per_call = cost_range(price.input_per_mtok, price.output_per_mtok, inputs, outputs)
        if tokens.unknown_slots:
            name = tokens.note.split(": ", 1)[-1].split(",")[0] if tokens.note else "prompt input"
            factors.append(Factor(name, "cost", inputs, "prior"))
        factors.append(Factor("output length", "cost", outputs, "prior"))

    source = analysis.index.sources.get(site.file, "")
    found = multiplicities(source, site.file).get(site.lineno)
    label: str | None = None
    multiplicity: Range | None = Range.exact(1)
    if found is not None:
        found = apply_declared(found, config.iterables)
        multiplicity = found.range
        if found.unknowns:
            label = found.unknowns[0].name
        for factor in found.factors:
            if factor.range is not None and not factor.range.is_point:
                factors.append(Factor(factor.name, "multiplicity", factor.range, "declared"))

    key = function_key(analysis.index, site)
    traffic = propagate(analysis.index, source_of_frequency, key) if key else None
    frequency = traffic.range if traffic is not None else None
    if frequency is not None and not frequency.is_point:
        factors.append(Factor("declared traffic", "frequency", frequency, "declared"))

    estimate = SiteEstimate(
        site_id=site.id,
        shape=site.shape,
        cost_per_call=per_call,
        multiplicity=multiplicity,
        frequency=frequency,
        factors=tuple(factors),
        confidence=site.confidence,
    )
    return estimate, tuple(factors), label


def blast_radius(head: Analysis, changed_files: set[str]) -> BlastRadius | None:
    """Which metered call sites a change reaches, including untouched files.

    Counts call sites, not callers. A function that reaches the changed code but
    holds no metered call of its own costs nothing extra, and counting it
    inflates the headline: gpt-engineer has twenty-four functions reaching its
    single LLM call, and reporting "affects 24 call sites" in a repository that
    contains exactly one is the confidently wrong number this project exists to
    avoid.
    """
    site_owner: dict[str, str] = {}
    for site in head.sites:
        key = function_key(head.index, site)
        if key is not None:
            site_owner[site.id] = key

    best: BlastRadius | None = None
    for key, info in head.index.functions.items():
        if info.file not in changed_files:
            continue
        reached = head.index.blast_radius(key) | {key}
        affected = [site for site in head.sites if site_owner.get(site.id) in reached]
        if not affected:
            continue
        candidate = BlastRadius(
            function=info.qualname,
            sites=len(affected),
            files=len({site.file for site in affected}),
            callers=len(reached) - 1,
        )
        if best is None or candidate.sites > best.sites:
            best = candidate
    return best


def build_report(repo: Path, base_ref: str, head_ref: str) -> Report:
    table = PriceTable.load()
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        base = Analysis.of(export_tree(base_ref, root / "base", repo))
        head = Analysis.of(export_tree(head_ref, root / "head", repo))
        config = Config.discover(root / "head")
        frequency = ConfigFrequencySource(config)

        diff = match(base.sites, head.sites)
        changed_files = {
            line.strip()
            for line in git("diff", "--name-only", f"{base_ref}..{head_ref}", cwd=repo).splitlines()
            if line.strip().endswith(".py")
        }

        changes: list[SiteChange] = []
        head_estimates: list[SiteEstimate] = []
        base_estimates: list[SiteEstimate] = []
        factors: list[Factor] = []

        for site in head.sites:
            estimate, site_factors, label = estimate_site(site, head, table, config, frequency)
            head_estimates.append(estimate)
            factors.extend(site_factors)
            matched = next((m for m in diff.matched if m.head.id == site.id), None)
            if matched is None:
                changes.append(
                    SiteChange("added", site.id, "", estimate, label, confidence=site.confidence)
                )
            elif matched.edited:
                detail = ""
                if matched.model_changed:
                    detail = f"`{matched.base.model}` → `{matched.head.model}`"
                changes.append(
                    SiteChange(
                        "edited", site.id, detail, estimate, label, confidence=site.confidence
                    )
                )

        for site in base.sites:
            estimate, _, _ = estimate_site(site, base, table, config, frequency)
            base_estimates.append(estimate)
            if not any(m.base.id == site.id for m in diff.matched):
                changes.append(SiteChange("removed", site.id, "", estimate))

        after = aggregate(head_estimates).per_month
        before = aggregate(base_estimates).per_month
        delta: Range | None = None
        if after is not None and before is not None:
            delta = after - before
        elif after is not None:
            delta = after

        spread = 1.0
        if after is not None and after.low > 0:
            spread = after.spread

        return Report(
            delta=delta,
            changes=tuple(changes),
            dominant=dominant(factors),
            total_spread=spread,
            coverage=Coverage(analysed=len(head.sites), candidates=head.candidates),
            blast=blast_radius(head, changed_files),
            prices_as_of=table.as_of,
            prices_stale=table.is_stale(date.today()),
            # Naming a source when nothing was read implies a file was read,
            # directly above a note saying none was found.
            frequency_source=frequency.name if config.declared else "",
            config_present=config.declared,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chihuahuabot")
    sub = parser.add_subparsers(dest="command", required=True)

    diff = sub.add_parser("diff", help="predict the cost impact between two commits")
    diff.add_argument("base")
    diff.add_argument("head")
    diff.add_argument("--repo", type=Path, default=Path.cwd())
    diff.add_argument("--format", choices=("markdown", "json"), default="markdown")
    diff.add_argument("--output", type=Path)
    diff.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="exit non-zero when nothing metered changed (off by default: a cost "
        "bot that breaks the build is a bot people uninstall)",
    )

    args = parser.parse_args(argv)
    report = build_report(args.repo, args.base, args.head)

    if args.format == "json":
        payload = {
            "empty": report.empty,
            "markdown": render(report),
            "delta": None
            if report.delta is None
            else {
                "low": report.delta.low,
                "expected": report.delta.expected,
                "high": report.delta.high,
            },
            "coverage": {
                "analysed": report.coverage.analysed if report.coverage else 0,
                "candidates": report.coverage.candidates if report.coverage else 0,
            },
            "changes": [
                {"kind": c.kind, "site": c.site_id, "detail": c.detail} for c in report.changes
            ],
        }
        text = json.dumps(payload, indent=2)
    else:
        text = render(report)

    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)

    return 1 if (args.fail_on_empty and report.empty) else 0


if __name__ == "__main__":
    raise SystemExit(main())
