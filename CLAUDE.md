# CLAUDE.md

**Benji** predicts the LLM/metered-API cost impact of a code diff and comments it on the PR.
*Infracost prices the infrastructure you declare; this prices the code that actually spends.*

Open source, for the community. The project is **Benji**, the package is `benji`, the
repository is `1Kyryll/Benji`. `costbot` and `chihuahuabot` are **deprecated earlier names**
and must not appear in code, CLI, config, or docs.

## How to work here

Specify and talk through the important design decisions with the user. We need a working project
for real teams and apps.

**Stop after each build step and present a short brief** — what was built, the design calls made
inside it, what the tests prove, and anything that came out differently from the plan. The next
step does not start until that brief is reviewed.

## Stack

Python 3.11+ (3.13 in use) · stdlib `ast` · `tiktoken` · FastAPI · Postgres · GitHub Action · Fly.io

Langfuse is the *destination* for frequency data, not an MVP dependency. See Invariants.

No runtime deps in the analysis modules — `ast` only.

## Layout

```
src/benji/
  extract.py        AST → call sites, loose + strict detectors   stdlib only
  identity.py       call-site IDs, matching across versions      stdlib only
  index.py          wrapper index + call graph + reverse graph   stdlib only
  pricing.py        price table + arithmetic                     stdlib only
  tokens.py         tokenisation, static/dynamic split           tiktoken
  multiplicity.py   loops, guards, retries                       stdlib only
  frequency.py      FrequencySource protocol + adapters          deps allowed
  estimate.py       range propagation, dominant uncertainty      stdlib only
  render.py         PR comment markdown
  cli.py            python -m benji.cli diff <base> <head>
corpus/
  manifest.toml     pinned OSS repos — the coverage denominator
  detect.py         loose detector (candidates)
  fetch.py          pinned checkouts
  score.py          the fitness function
tests/
action/             GitHub Action entrypoint (two-workflow, fork-safe)
```

`pricing.py` and `tokens.py` are split deliberately: the no-runtime-deps rule names `pricing`,
but tokenisation needs `tiktoken`. Table and arithmetic stay stdlib; the tokeniser does not.

src layout: the wheel contains `benji` only. `corpus/` is a development harness, kept
importable in tests via `pythonpath` and never shipped.

`corpus/` must not import `benji` except through the resolver plug point in `score.py`.
A measuring instrument that shares helpers with the system under test agrees with its own bugs.

## Commands

```bash
pip install -e ".[dev]"           # uv is not installed on this machine
pytest -q
ruff check . && ruff format .
python -m corpus.score                        # coverage across pinned repos
python -m corpus.score --repo aider --sample 20
python -m benji.cli diff HEAD~1 HEAD
```

## Invariants

Rationale lives in the `cost-pr-bot` skill and in the design doc. Don't silently re-litigate —
say so if one looks wrong. Note the skill still uses the deprecated `costbot` name and describes
a "current state" that predates this repo; its **engineering** decisions are authoritative, its
naming and status are not.

- `call_site_id = file : qualified_function : ordinal`. **Never a line number.** Ordinal counts
  metered calls only, per function; nested functions get their own counter.
- Three layers, never collapsed: `cost/call × multiplicity × frequency`. Tokenizer, AST, telemetry
  respectively.
- **Frequency is adapter-based, not config-based.** `FrequencySource` is an interface;
  `ConfigFrequencySource` (reads `benji.yml`) is implementation #1 and
  `LangfuseFrequencySource` is where this is going. The MVP ships config because install friction
  is the binding constraint for a community tool, not because declared numbers are good enough.
- **Never emit a point estimate.** Carry `(low, expected, high)`; name the dominant uncertainty.
- Deterministic code computes every number. An LLM may write prose or classify ambiguity — never
  produce a figure.
- Unresolvable input (`model=config.DEFAULT_MODEL`, unknown loop bound) returns `None` or a
  flagged range. Correct output, not a gap. Never a guessed default.
- Unmatched call site → new, low confidence. Visible wrong beats silent wrong.
- Changed files ≠ affected call sites. Editing a wrapper changes cost in files the diff never
  touches — hence the reverse call graph, which also carries declared frequency from entry points
  down to individual call sites.
- **Always report coverage**: "analysed 34 of 51 candidate call sites". The denominator comes from
  the loose detector in `corpus/detect.py`, chosen before the analyser existed.
- **Raw HTTP provider calls are in scope and get priced from their request payload.** 26% of the
  corpus reaches providers with `session.post(f"{url}/api/chat", json={...})` and never imports an
  SDK. The model comes from the `model` key of the payload, exactly as it comes from the `model=`
  kwarg of an SDK call. Same three layers, different syntax — not a separate subsystem.

## Build order

0. Corpus + coverage harness — the fitness function
1. Extraction + identity
2. Wrapper index + call graph + reverse call graph
3. Matching across two file versions
4. Pricing + tokens
5. Multiplicity — loops, guards, retries
6. Frequency — `FrequencySource`, config adapter, propagation over the reverse graph
7. Range propagation + dominant uncertainty
8. PR comment renderer
9. GitHub Action + fork-safe distribution

Each depends on the previous. Don't scaffold ahead.

This differs from the skill's build order: the corpus comes first (decisions 4 and 6 are untested
bets about what real wrapper code looks like), and the wrapper index is an explicit step rather
than an unplaced decision.

## Conventions

- `from __future__ import annotations`; `CallSite` frozen and hashable.
- Detectors match the attribute-path **tail** (`[-3:]`), so any receiver name matches.
- Tests before implementation for new behaviour.
- MVP: Python only, OpenAI + Anthropic SDKs, `for` loops and retry decorators. Not: multi-language,
  own OTLP ingest, agent loops, merge gating, dashboard.
- Fork PRs get a read-only `GITHUB_TOKEN`. The Action is two workflows: analyse on `pull_request`
  with no secrets, post from a privileged `workflow_run` job. Never `pull_request_target` with a
  head-ref checkout.
