"""The price table, and the arithmetic that turns token counts into dollars.

Stdlib only. `tomllib` reads the table, and everything here is multiplication —
the tokenizer lives next door in ``tokens.py`` because it needs a dependency and
this must not.

Two rules shape the whole module. An unknown model returns ``None`` rather than a
guessed price, because a confident wrong dollar figure is the failure mode that
kills the product. And a price matched by prefix rather than exactly says so, so
the layer above can lower its confidence instead of inheriting ours.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

DEFAULT_TABLE = Path(__file__).parent / "prices.toml"

# Providers that are not really providers: the detector knew a call was metered
# but not who bills for it. A model name can still identify it.
UNKNOWN_PROVIDERS = frozenset({"", "framework", "http", "litellm", "ollama"})


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1,000,000 tokens for one model."""

    provider: str
    model: str
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None
    as_of: str
    match: str  # "exact" | "prefix" | "cross-provider"

    @property
    def exact(self) -> bool:
        return self.match == "exact"


def cost(price: ModelPrice, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """Dollars for one call at these token counts.

    ``cached_tokens`` is a subset of ``input_tokens``, priced at the cached rate
    where the provider publishes one. Prompt caching changes the input price by
    up to ten times, so ignoring it silently would be an order-of-magnitude
    error on any repository that uses it.
    """
    uncached = max(0, input_tokens - cached_tokens)
    total = uncached * price.input_per_mtok + output_tokens * price.output_per_mtok
    if cached_tokens:
        rate = (
            price.cached_input_per_mtok
            if price.cached_input_per_mtok is not None
            else price.input_per_mtok
        )
        total += cached_tokens * rate
    return total / 1_000_000


class PriceTable:
    """Loaded price data, plus the model-name resolution rules."""

    def __init__(self, data: dict) -> None:
        self.as_of: str = data.get("as_of", "unknown")
        self.stale_after_days: int = data.get("stale_after_days", 90)
        self._providers: dict[str, dict[str, dict]] = {
            key: value
            for key, value in data.items()
            if isinstance(value, dict) and all(isinstance(v, dict) for v in value.values())
        }

    @classmethod
    def load(cls, path: Path | None = None) -> PriceTable:
        return cls(tomllib.loads((path or DEFAULT_TABLE).read_text()))

    # --- staleness --------------------------------------------------------

    def age_days(self, today: date) -> int | None:
        try:
            return (today - date.fromisoformat(self.as_of)).days
        except ValueError:
            return None

    def is_stale(self, today: date) -> bool:
        """Unparseable dates count as stale. Not knowing the age is not reassuring."""
        age = self.age_days(today)
        return True if age is None else age > self.stale_after_days

    # --- lookup -----------------------------------------------------------

    def _build(self, provider: str, model: str, entry: dict, match: str) -> ModelPrice:
        return ModelPrice(
            provider=provider,
            model=model,
            input_per_mtok=float(entry["input"]),
            output_per_mtok=float(entry.get("output", 0.0)),
            cached_input_per_mtok=(
                float(entry["cached_input"]) if "cached_input" in entry else None
            ),
            as_of=self.as_of,
            match=match,
        )

    def _in_provider(self, provider: str, model: str) -> ModelPrice | None:
        models = self._providers.get(provider)
        if not models:
            return None

        if model in models:
            return self._build(provider, model, models[model], "exact")

        # Real code pins dated snapshots: `gpt-4o-2024-08-06`. Longest prefix
        # wins, so `gpt-4o-mini-2024-07-18` cannot fall through to `gpt-4o` and
        # be billed at sixteen times its real rate. The boundary check stops
        # `gpt-4omega` matching `gpt-4o`.
        best: str | None = None
        for known in models:
            if model.startswith(known) and model[len(known) : len(known) + 1] in ("-", "@", ":"):
                if best is None or len(known) > len(best):
                    best = known
        if best is not None:
            return self._build(provider, best, models[best], "prefix")
        return None

    def lookup(self, provider: str, model: str | None) -> ModelPrice | None:
        """The price for this model, or ``None`` when we do not know it.

        ``None`` is a real answer. `model=config.DEFAULT_MODEL` is unresolvable
        by design, and inventing a price for it would turn an honest gap into a
        confident wrong number.
        """
        if not model:
            return None

        found = self._in_provider(provider, model)
        if found is not None:
            return found

        # The detector may know a call is metered without knowing who bills for
        # it — a LangChain call, or raw HTTP to a gateway. The model name is
        # still evidence, but only when exactly one provider claims it.
        if provider in UNKNOWN_PROVIDERS or provider not in self._providers:
            hits = [
                self._in_provider(name, model)
                for name in self._providers
                if self._in_provider(name, model) is not None
            ]
            if len(hits) == 1 and hits[0] is not None:
                found = hits[0]
                return ModelPrice(
                    provider=found.provider,
                    model=found.model,
                    input_per_mtok=found.input_per_mtok,
                    output_per_mtok=found.output_per_mtok,
                    cached_input_per_mtok=found.cached_input_per_mtok,
                    as_of=found.as_of,
                    match="cross-provider",
                )
        return None

    @property
    def models(self) -> list[tuple[str, str]]:
        return [(p, m) for p, models in self._providers.items() for m in models]
