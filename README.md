# Benji

Predicts the LLM and metered-API cost impact of a code diff, and comments it on the pull request.

> Infracost prices the infrastructure you declare. **This prices the code that actually spends.**

Static analysis knows what a diff changed. Observability knows what you already paid. Neither
knows what a change *about to merge* is going to cost.

```markdown
## 💸 Benji — cost impact

**−$462 to +$8,186 / month** · expected **+$299**

| call site | change |
|---|---|
| `api/tickets.py:handle:0` | `gpt-4o-mini` → `gpt-4o` · $0.00212/call · ×1 · 5,000/day |
| `workers/digest.py:digest:0` | **new** · $0.00022/call · ×3–500 · 1/day |

> **The range is driven almost entirely by `len(orgs)`**, which is undeclared.
> Declare it under `[iterables]` or `[frequency]` in `benji.toml`.

⚠️ This 4-line change to `LLMClient.chat` affects **47 call sites across 12 files**.

<details><summary>Coverage: 34 of 51 candidate call sites analysed</summary>
17 unresolved — 12 dynamic dispatch, 5 depth limit.
</details>
```

## How it thinks

Every number is one of three terms, and they are never collapsed into each other:

```
    cost per call     dollars for one execution      tokenizer + price table
  × multiplicity      firings per invocation         AST: loops, guards, retries
  × frequency         invocations per day            your declaration, or telemetry
  ─────────────────
  = dollars per day
```

They use different machinery, fail in different ways, and change on different timescales. Keeping
them apart is what lets the bot say *why* a number moved — a price change and a traffic change need
opposite responses.

Three rules follow from that, and they are the whole personality of the tool:

- **Never a point estimate.** A confident `+$127.40/mo` that is wrong by ten times is worse than
  saying nothing. Everything is a range, and the bot names the factor driving its width.
- **Never a guess.** An unresolvable model or an unbounded loop returns *nothing*, and the comment
  says which layer was missing. Unknown is a real answer.
- **Always report coverage.** "Analysed 34 of 51" turns a miss from silent into visible.

Deterministic code computes every figure. A language model may write prose or classify an ambiguous
call; it never produces a number.

## Install

```yaml
# .github/workflows/benji.yml
name: benji
on: pull_request
permissions:
  contents: read
jobs:
  analyse:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: 1Kyryll/Benji/action@v1
      - uses: actions/upload-artifact@v4
        with:
          name: benji-report
          path: benji-report.json
```

Posting the comment is a **second workflow**, and that is deliberate. On a pull request from a fork,
`GITHUB_TOKEN` is read-only: the analysis succeeds and the comment fails with a 403. The fix is to
analyse without privileges and post from a `workflow_run` job the fork cannot reach. Copy
[`benji-comment.yml`](.github/workflows/benji-comment.yml) as-is.

Never use `pull_request_target` with a checkout of the head ref. That is the well-known way to hand
a stranger write access to your repository.

Locally:

```bash
pip install benji-bot          # `benji` on PyPI is an unrelated backup tool
benji diff HEAD~1 HEAD         # or: python -m benji.cli diff HEAD~1 HEAD
```

## Configure

Optional. Without it you still get cost per call and multiplicity; you do not get dollars per day,
and the comment says so rather than guessing.

```toml
# benji.toml
[frequency]
"api.tickets:TicketService.handle" = { low = 2000, expected = 5000, high = 20000 }
"workers.digest:run" = 1

[iterables]
"handle:orgs" = { low = 3, expected = 40, high = 500 }
```

Declare **entry points**, not call sites. Nobody knows how often `LLMClient.chat` runs; everybody
knows roughly how many tickets arrive in a day. The bot carries the number down the call graph to
every metered call it reaches.

See [`benji.example.toml`](benji.example.toml).

## What it detects

Python, OpenAI and Anthropic, in four shapes:

| shape | example | confidence |
|---|---|---|
| SDK | `client.chat.completions.create(model=...)` | exact |
| raw HTTP | `session.post(f"{url}/v1/chat/completions", json={"model": ...})` | exact |
| first-party wrapper | `self.llm.chat(...)` resolved to the SDK call underneath | decays with depth |
| framework | `llm.invoke(...)`, or `self.llm(messages)` — LangChain and friends | reduced |

Wrapper resolution follows attribute types from annotations, constructor assignments, module
singletons and local assignments, stopping at a metered call, a cycle, depth 5, or the repository
boundary. It never parses `site-packages`.

**Editing a wrapper is not a local change.** A reverse call graph finds every site that reaches the
function you touched, across files the diff never opened.

Measured on 11 real open-source Python applications: **78% of candidate call sites resolved**.

## Not in this version

Languages other than Python · own OTLP ingest · agent-loop analysis · merge gating · a dashboard.

Telemetry-backed frequency is the destination, not an alternative to the config file: the
`FrequencySource` interface exists so a Langfuse adapter is an addition rather than a rewrite.

## Release

Publishing runs on a GitHub Release via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — no API token is stored anywhere. The workflow runs the tests, checks the release tag matches the version in `pyproject.toml`, and only then uploads.

```bash
# bump version in pyproject.toml, commit, then
gh release create v0.1.0 --generate-notes
```

## Develop

```bash
pip install -e ".[dev]"
pytest -q
ruff check . && ruff format .
python -m corpus.score --sample 20   # coverage against pinned real repositories
```

`corpus/` is the fitness function, and it keeps its own copy of the call-detection logic on purpose:
a measuring instrument that shares code with the system it measures agrees with its own bugs.

## Licence

MIT.
