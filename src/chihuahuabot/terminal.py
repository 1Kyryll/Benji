"""Human-readable output for a terminal.

The PR comment is markdown because GitHub renders it. A terminal does not, so
piping the same text to a console produces a wall of pipes and asterisks that
buries the one number anybody wanted.

This is the same report, laid out for reading: the figure first and large, then
what it applies to, then the single question whose answer would sharpen it. Same
data, same honesty rules — only the presentation differs.
"""

from __future__ import annotations

import os
import shutil
import sys

from chihuahuabot.render import Report, SiteChange, count, money, money_range, multiplier

SYMBOLS = {"added": "+", "removed": "-", "edited": "~", "affected": "·"}


class Style:
    """ANSI styling that switches itself off when nobody can see it."""

    CODES = {
        "bold": "1",
        "dim": "2",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
    }

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def __call__(self, text: str, *names: str) -> str:
        if not self.enabled or not names:
            return text
        codes = ";".join(self.CODES[name] for name in names if name in self.CODES)
        return f"\033[{codes}m{text}\033[0m" if codes else text


def use_colour(stream=None) -> bool:
    """Respect NO_COLOR, and never emit escape codes into a pipe."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def split_id(site_id: str) -> tuple[str, str]:
    """`api/tickets.py:TicketService.triage:0` -> path, `TicketService.triage:0`."""
    head, _, rest = site_id.partition(":")
    return (head, rest) if rest else (site_id, "")


def rule(width: int, style: Style) -> str:
    return style("─" * width, "dim")


def headline(report: Report, style: Style) -> list[str]:
    if not report.changes:
        return [style("  no metered call sites changed", "bold")]
    if report.delta is None:
        return [
            style("  cost impact unknown", "bold", "yellow"),
            style("  call sites changed, but a layer could not be resolved", "dim"),
        ]

    span = report.delta
    if span.is_point and span.expected == 0:
        return [style("  no change in predicted cost", "bold")]

    colour = "red" if span.expected > 0 else "green"
    figure = money_range(span, sign=True)
    lines = [style(f"  {figure}", "bold", colour) + style("  per month", "dim")]
    sign = "+" if span.expected > 0 else "−"
    lines.append(style(f"  expected {sign}{money(abs(span.expected))}", colour))
    if span.low < 0 < span.high:
        lines.append(style("  this change could save money or cost far more", "dim"))
    return lines


def change_block(change: SiteChange, style: Style) -> list[str]:
    symbol = SYMBOLS.get(change.kind, "·")
    colour = {"added": "green", "removed": "red", "edited": "yellow"}.get(change.kind, "dim")
    path, qualname = split_id(change.site_id)

    lines = [
        f"  {style(symbol, 'bold', colour)} {style(path, 'cyan')}"
        f"{style('  ' + qualname, 'dim') if qualname else ''}"
    ]

    if change.detail:
        lines.append(f"      {change.detail.replace('`', '')}")

    estimate = change.estimate
    if estimate is None:
        return lines

    if estimate.cost_per_call is None:
        missing = ", ".join(estimate.missing)
        lines.append(f"      {style('not priced', 'yellow')} {style('— ' + missing, 'dim')}")
        return lines

    bits = [
        f"{money(estimate.cost_per_call.expected)}/call",
        multiplier(estimate.multiplicity, change.multiplicity_label),
        f"{count(estimate.frequency.expected)}/day"
        if estimate.frequency is not None
        else style("frequency undeclared", "yellow"),
    ]
    lines.append("      " + style("   ".join(bits), "dim"))

    monthly = estimate.per_month
    if monthly is not None:
        lines.append("      " + style(f"= {money_range(monthly)} / month", "dim"))
    if change.confidence < 1.0:
        lines.append("      " + style(f"confidence {change.confidence:.0%}", "dim"))
    return lines


def unknown_block(report: Report, style: Style, width: int) -> list[str]:
    factor = report.dominant
    spread = report.total_spread
    if factor is None or not (2.0 <= spread < float("inf")):
        return []

    narrowed = spread / factor.range.spread if factor.range.spread > 0 else None
    lines = [
        "",
        style("  BIGGEST UNKNOWN", "bold", "yellow") + f"   {style(factor.name, 'bold')}",
        style(f"  the estimate spans {spread:.0f}× because of this one value", "dim"),
    ]
    if narrowed is not None and narrowed >= 1.0:
        lines.append(style(f"  pin it down and the range tightens to about {narrowed:.1f}×", "dim"))

    section = {"multiplicity": "iterables", "frequency": "frequency"}.get(factor.layer)
    if factor.source == "prior" and section:
        lines += [
            "",
            style("  add to chihuahuabot.toml:", "dim"),
            style(f"      [{section}]", "green"),
            style(f'      "{factor.name}" = ' + "{ low = ?, expected = ?, high = ? }", "green"),
        ]
    elif factor.source == "prior":
        lines.append(
            style("  this is prompt text whose length varies; telemetry would settle it", "dim")
        )
    return lines


def render_terminal(report: Report, colour: bool | None = None, width: int | None = None) -> str:
    style = Style(use_colour() if colour is None else colour)
    width = width or min(shutil.get_terminal_size((80, 24)).columns, 88)

    out: list[str] = ["", style("  💸 ChihuahuaBot", "bold") + style("  cost impact", "dim"), ""]
    out += headline(report, style)

    if report.changes:
        out += ["", rule(width, style), ""]
        plural = "s" if len(report.changes) != 1 else ""
        out.append(style(f"  {len(report.changes)} call site{plural} changed", "bold"))
        out.append("")
        for change in report.changes:
            out += change_block(change, style)
            out.append("")
        out.pop()

    out += unknown_block(report, style, width)

    if report.blast is not None and report.blast.worth_saying:
        blast = report.blast
        out += [
            "",
            style("  REACHES FURTHER THAN THE DIFF", "bold", "yellow"),
            style(
                f"  editing {blast.function} moves the cost of "
                f"{blast.sites} call sites across {blast.files} files",
                "dim",
            ),
        ]

    if not report.config_present:
        out += [
            "",
            style("  NO CONFIG", "bold", "yellow") + style("   chihuahuabot.toml not found", "dim"),
            style("  cost per call and multiplicity are exact; dollars per day need", "dim"),
            style("  you to declare how often your entry points run", "dim"),
        ]

    out += ["", rule(width, style)]
    if report.coverage is not None:
        coverage = report.coverage
        out.append(
            style("  coverage  ", "dim")
            + f"{coverage.analysed} of {coverage.candidates} candidate call sites analysed"
        )
    out.append(style("  prices    ", "dim") + f"as of {report.prices_as_of}")
    if report.prices_stale:
        out.append(style("            this table may be out of date", "yellow"))
    if report.frequency_source:
        out.append(style("  frequency ", "dim") + report.frequency_source)
    out.append("")
    return "\n".join(out) + "\n"
