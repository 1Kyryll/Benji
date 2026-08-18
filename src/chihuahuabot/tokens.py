"""Counting the tokens a call sends, and admitting which ones we cannot see.

The design splits input tokens in two, and the split is the whole point::

    input_tokens = tokenize(static template)   <- exact
                 + interpolated variables      <- unknown, recovered from telemetry

A prompt like ``f"Summarise this ticket: {ticket.body}"`` is part literal and
part mystery. Counting the literal exactly and *reporting the mystery as a
count* is what lets a later stage say "the range is driven by ``ticket.body``"
instead of printing a made-up total. Collapsing the two into one number would
throw away the only thing that makes the estimate explainable.

This module is allowed a dependency; ``pricing.py`` next door is not.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

# Keyword arguments that carry prompt text. Everything else in a call is
# configuration and costs nothing.
PROMPT_KWARGS = frozenset(
    {"messages", "input", "prompt", "system", "contents", "text", "json", "data", "json_data"}
)

# Dict keys whose values are structure, not prose. Counting "user" and
# "assistant" as prompt text would inflate every message by a token or two.
STRUCTURAL_KEYS = frozenset({"role", "type", "name", "model", "id", "tool_call_id"})

# Per-message framing the providers add on the wire (role delimiters and so on).
# Approximate by construction, and small next to the text itself.
TOKENS_PER_MESSAGE = 4

FALLBACK_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class TokenEstimate:
    """What we counted, and what we could not.

    ``unknown_slots`` is the count of interpolated values whose size we cannot
    know statically. It is deliberately a count rather than a guess: a later
    stage attaches a distribution to each one, and the biggest of them usually
    turns out to be the dominant uncertainty in the whole estimate.
    """

    static: int
    unknown_slots: int
    messages: int
    approximate: bool
    encoding: str
    note: str = ""

    @property
    def complete(self) -> bool:
        """True when nothing was interpolated, so ``static`` is the real count."""
        return self.unknown_slots == 0


def _encoder(model: str | None):
    """An encoder plus whether it is the right one for this model.

    Falls back twice: to a general encoding when the model is unknown, and to
    character counting when ``tiktoken`` is missing or cannot fetch its data. A
    tool that crashes offline is worse than one that says its numbers are rough.
    """
    try:
        import tiktoken
    except ImportError:
        return None, True, "characters"

    try:
        if model:
            return tiktoken.encoding_for_model(model), False, "model"
    except KeyError:
        pass

    try:
        encoding = tiktoken.get_encoding("o200k_base")
        return encoding, True, encoding.name
    except Exception:
        return None, True, "characters"


def count(text: str, model: str | None = None) -> tuple[int, bool, str]:
    """Token count for a string, plus whether it is approximate."""
    encoder, approximate, name = _encoder(model)
    if encoder is None:
        return max(1, len(text) // FALLBACK_CHARS_PER_TOKEN) if text else 0, True, name
    try:
        return len(encoder.encode(text)), approximate, getattr(encoder, "name", name)
    except Exception:
        return max(1, len(text) // FALLBACK_CHARS_PER_TOKEN) if text else 0, True, "characters"


def _walk_prompt(node: ast.AST, statics: list[str], slots: list[str], messages: list[int]) -> None:
    """Split a prompt argument into literal text and unknowable holes."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            statics.append(node.value)
        return

    if isinstance(node, ast.JoinedStr):
        # An f-string is the split in miniature: constant parts are exact, every
        # `{...}` is a hole.
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                statics.append(part.value)
            elif isinstance(part, ast.FormattedValue):
                slots.append(ast.unparse(part.value))
        return

    if isinstance(node, ast.Dict):
        messages.append(1)
        for key, value in zip(node.keys, node.values, strict=False):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value in STRUCTURAL_KEYS
            ):
                continue
            _walk_prompt(value, statics, slots, messages)
        return

    if isinstance(node, ast.List | ast.Tuple):
        for item in node.elts:
            _walk_prompt(item, statics, slots, messages)
        return

    if isinstance(node, ast.BinOp):
        _walk_prompt(node.left, statics, slots, messages)
        _walk_prompt(node.right, statics, slots, messages)
        return

    # A name, an attribute, a call, a comprehension: something we cannot read.
    slots.append(ast.unparse(node))


def estimate_input(content: str, model: str | None = None) -> TokenEstimate:
    """Estimate the input tokens of a call from its source.

    ``content`` is the normalised source of the call, which ``CallSite`` already
    carries. Re-parsing it keeps this module independent of the extractor: it
    needs the prompt's shape, not the extractor's internals.
    """
    try:
        parsed = ast.parse(content, mode="eval").body
    except SyntaxError:
        return TokenEstimate(0, 0, 0, True, "characters", note="unparseable call")

    if not isinstance(parsed, ast.Call):
        return TokenEstimate(0, 0, 0, True, "characters", note="not a call")

    statics: list[str] = []
    slots: list[str] = []
    messages: list[int] = []
    for keyword in parsed.keywords:
        if keyword.arg in PROMPT_KWARGS:
            _walk_prompt(keyword.value, statics, slots, messages)

    text = "".join(statics)
    static_tokens, approximate, encoding = count(text, model)
    static_tokens += len(messages) * TOKENS_PER_MESSAGE

    note = ""
    if slots:
        note = f"{len(slots)} interpolated value(s): " + ", ".join(sorted(set(slots))[:3])

    return TokenEstimate(
        static=static_tokens,
        unknown_slots=len(slots),
        messages=len(messages),
        approximate=approximate,
        encoding=encoding,
        note=note,
    )
