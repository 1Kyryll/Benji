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
from chihuahuabot.terminal import render_terminal
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


@dataclass
class Priced:
    """A site's estimate, plus the pieces a paired delta needs to cancel.

    Keeping the token ranges and the unit prices apart from the finished cost is
    what lets a model swap be reported as nearly exact: the same token count sits
    on both sides of the diff, so only the price ratio should survive.
    """

    estimate: SiteEstimate
    factors: tuple[Factor, ...]
    label: str | None = None
    inputs: Range | None = None
    outputs: Range | None = None
    input_price: float | None = None
    output_price: float | None = None


def estimate_site(
    site: CallSite,
    analysis: Analysis,
    table: PriceTable,
    config: Config,
    source_of_frequency,
) -> Priced:
    """Assemble one call site's three layers. Any of them may be unknown."""
    factors: list[Factor] = []

    price = table.lookup(site.provider, site.model)
    per_call: Range | None = None
    inputs = outputs = None
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
    return Priced(
        estimate=estimate,
        factors=tuple(factors),
        label=label,
        inputs=inputs,
        outputs=outputs,
        input_price=price.input_per_mtok if price else None,
        output_price=price.output_per_mtok if price else None,
    )


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


def edited_delta(was, now, estimate) -> Range | None:
    """The monthly change for one edited call site.

    Where the two sides share a factor, it multiplies the difference instead of
    being subtracted twice. A model swap with an untouched prompt is the case
    that matters: the same token count sits on both sides, so only the price
    ratio survives and the answer is nearly exact even if the token estimate is
    badly wrong. Subtracting two finished costs would instead widen the result
    by the full token uncertainty, twice.
    """
    if was is None or now is None:
        return estimate.per_month

    before, after = was.estimate, now.estimate
    shared_factors = (
        before.multiplicity == after.multiplicity and before.frequency == after.frequency
    )
    same_prompt = (
        was.inputs is not None
        and was.inputs == now.inputs
        and was.outputs == now.outputs
        and None not in (was.input_price, now.input_price, was.output_price, now.output_price)
    )

    if shared_factors and after.multiplicity is not None and after.frequency is not None:
        if same_prompt:
            per_call = (
                now.inputs * Range.exact(now.input_price - was.input_price)
                + now.outputs * Range.exact(now.output_price - was.output_price)
            ).scale(1 / 1_000_000)
        elif before.cost_per_call is not None and after.cost_per_call is not None:
            per_call = after.cost_per_call - before.cost_per_call
        else:
            return after.per_month
        return (per_call * after.multiplicity * after.frequency).scale(30.0)

    monthly, previously = after.per_month, before.per_month
    if monthly is not None and previously is not None:
        return monthly - previously
    return monthly


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
        head_priced: dict[str, Priced] = {}
        base_estimates: list[SiteEstimate] = []
        factors: list[Factor] = []

        for site in head.sites:
            priced = estimate_site(site, head, table, config, frequency)
            estimate, label = priced.estimate, priced.label
            head_priced[site.id] = priced
            head_estimates.append(estimate)
            factors.extend(priced.factors)
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

        base_by_id: dict[str, Priced] = {}
        for site in base.sites:
            priced = estimate_site(site, base, table, config, frequency)
            estimate = priced.estimate
            base_estimates.append(estimate)
            matched = next((m for m in diff.matched if m.base.id == site.id), None)
            if matched is not None:
                base_by_id[matched.head.id] = priced
            if not any(m.base.id == site.id for m in diff.matched):
                changes.append(SiteChange("removed", site.id, "", estimate))

        # Deltas are computed per changed site, not as one total minus another.
        #
        # Errors cancel in a delta. Swapping a model without touching the prompt
        # is nearly exact even with a 50% token error, because the same wrong
        # token count sits on both sides and only the price ratio survives.
        # Subtracting two whole-repository totals throws that away: it treats the
        # two sides as independent and reports a config-only commit, which
        # changes nothing at all, as somewhere between saving $8,386 and costing
        # $8,386.
        delta: Range | None = None

        def contribute(piece: Range | None) -> None:
            nonlocal delta
            if piece is None:
                return
            delta = piece if delta is None else delta + piece

        for change in changes:
            estimate = change.estimate
            if estimate is None:
                continue
            if change.kind == "added":
                contribute(estimate.per_month)
            elif change.kind == "removed":
                monthly = estimate.per_month
                contribute(-monthly if monthly is not None else None)
            elif change.kind == "edited":
                was = base_by_id.get(change.site_id)
                now = head_priced.get(change.site_id)
                contribute(edited_delta(was, now, estimate))

        after = aggregate(head_estimates).per_month
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
    diff.add_argument(
        "--format",
        choices=("auto", "terminal", "markdown", "json"),
        default="auto",
        help="auto reads well in a console and emits markdown when piped or "
        "written to a file, which is what a PR comment needs",
    )
    diff.add_argument("--no-color", action="store_true", help="disable ANSI styling")
    diff.add_argument("--output", type=Path)
    diff.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="exit non-zero when nothing metered changed (off by default: a cost "
        "bot that breaks the build is a bot people uninstall)",
    )

    args = parser.parse_args(argv)
    report = build_report(args.repo, args.base, args.head)

    # A human at a terminal wants something readable; a pipe or a file is almost
    # always on its way to a PR comment, which has to be markdown.
    chosen = args.format
    if chosen == "auto":
        chosen = "terminal" if (not args.output and sys.stdout.isatty()) else "markdown"

    if args.format == "json" or chosen == "json":
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
    elif chosen == "terminal":
        text = render_terminal(report, colour=not args.no_color)
    else:
        text = render(report)

    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)

    return 1 if (args.fail_on_empty and report.empty) else 0


if __name__ == "__main__":
    raise SystemExit(main())
