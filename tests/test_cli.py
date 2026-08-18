"""End to end: two commits in a real repository, one comment out.

Every other test exercises a stage in isolation. These drive `chihuahuabot diff`
against an actual git repository, because the failures worth catching here are
the ones that only appear when the stages meet.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from chihuahuabot.cli import build_report, main


def module(model: str, func: str = "handle") -> str:
    """A module with one metered call, kept short enough to read."""
    return (
        "import openai\n"
        "client = openai.OpenAI()\n\n\n"
        f"def {func}(t):\n"
        "    return client.chat.completions.create(\n"
        f'        model="{model}", max_tokens=400,\n'
        '        messages=[{"role": "user", "content": "Triage this ticket."}],\n'
        "    )\n"
    )


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "app"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")
    (root / "api.py").write_text(module("gpt-4o-mini"))
    (root / "chihuahuabot.toml").write_text('[frequency]\n"api:handle" = 5000\n')
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    return root


def commit(repo: Path, path: str, body: str, message: str = "change") -> None:
    (repo / path).write_text(body)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


# --- the happy path -------------------------------------------------------


def test_a_model_swap_produces_a_priced_comment(repo: Path):
    commit(repo, "api.py", module("gpt-4o"))
    report = build_report(repo, "HEAD~1", "HEAD")
    assert report.delta is not None
    assert any(c.kind == "edited" for c in report.changes)


def test_the_swap_is_described_in_the_comment(repo: Path):
    commit(repo, "api.py", module("gpt-4o"))
    edited = next(c for c in build_report(repo, "HEAD~1", "HEAD").changes if c.kind == "edited")
    assert "gpt-4o-mini" in edited.detail and "gpt-4o" in edited.detail


def test_a_more_expensive_model_costs_more(repo: Path):
    """The direction of the delta is the one thing a reader checks first."""
    commit(repo, "api.py", module("gpt-4o"))
    assert build_report(repo, "HEAD~1", "HEAD").delta.expected > 0


def test_a_new_call_site_is_reported_as_added(repo: Path):
    commit(repo, "worker.py", module("gpt-4o-mini", "digest"))
    assert any(c.kind == "added" for c in build_report(repo, "HEAD~1", "HEAD").changes)


def test_a_deleted_call_site_is_reported_as_removed(repo: Path):
    commit(repo, "api.py", "def handle(t):\n    return None\n")
    assert any(c.kind == "removed" for c in build_report(repo, "HEAD~1", "HEAD").changes)


# --- honesty --------------------------------------------------------------


def test_a_commit_touching_nothing_metered_is_empty(repo: Path):
    """A bot that comments on every pull request is a bot people mute."""
    commit(repo, "README.md", "# hello\n")
    assert build_report(repo, "HEAD~1", "HEAD").empty


def test_coverage_is_always_populated(repo: Path):
    commit(repo, "README.md", "# hello\n")
    coverage = build_report(repo, "HEAD~1", "HEAD").coverage
    assert coverage is not None and coverage.candidates >= coverage.analysed


def test_an_unresolvable_model_is_not_priced(repo: Path):
    """`model=config.MODEL` is unknowable, and the row says so rather than guessing."""
    commit(
        repo,
        "api.py",
        "import openai\nimport config\nclient = openai.OpenAI()\n\n\ndef handle(t):\n"
        "    return client.chat.completions.create(model=config.MODEL, messages=[])\n",
    )
    report = build_report(repo, "HEAD~1", "HEAD")
    edited = [c for c in report.changes if c.estimate and c.estimate.cost_per_call is None]
    assert edited


def test_analysis_covers_files_the_diff_never_touched(repo: Path):
    """Changed files are not the same set as affected call sites."""
    commit(repo, "unrelated.py", "def helper():\n    return 1\n")
    report = build_report(repo, "HEAD~1", "HEAD")
    assert report.coverage.analysed >= 1


# --- the command line -----------------------------------------------------


def test_markdown_is_the_default_output(repo: Path, capsys):
    commit(repo, "api.py", module("gpt-4o"))
    assert main(["diff", "HEAD~1", "HEAD", "--repo", str(repo)]) == 0
    assert "ChihuahuaBot" in capsys.readouterr().out


def test_json_output_carries_the_markdown_and_the_delta(repo: Path, tmp_path: Path):
    """The workflow handoff: analysis runs unprivileged, posting runs privileged."""
    commit(repo, "api.py", module("gpt-4o"))
    out = tmp_path / "report.json"
    main(["diff", "HEAD~1", "HEAD", "--repo", str(repo), "--format", "json", "--output", str(out)])
    payload = json.loads(out.read_text())
    assert "ChihuahuaBot" in payload["markdown"]
    assert payload["delta"]["high"] >= payload["delta"]["low"]
    assert payload["empty"] is False


def test_the_bot_does_not_fail_the_build_by_default(repo: Path, capsys):
    """A cost bot that breaks CI is a bot people uninstall."""
    commit(repo, "README.md", "# hello\n")
    assert main(["diff", "HEAD~1", "HEAD", "--repo", str(repo)]) == 0


def test_failing_on_empty_is_opt_in(repo: Path, capsys):
    commit(repo, "README.md", "# hello\n")
    assert main(["diff", "HEAD~1", "HEAD", "--repo", str(repo), "--fail-on-empty"]) == 1
