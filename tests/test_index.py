"""Wrapper resolution, call graph, and blast radius.

The fixture repository under ``fixtures/repo`` calls the OpenAI SDK in exactly
one file. Everything else reaches it through a wrapper, which is what real
application code looks like and what an inline-only detector cannot see.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chihuahuabot.index import MAX_DEPTH, RepoIndex

ROOT = Path(__file__).parent / "fixtures" / "repo"


@pytest.fixture(scope="module")
def index() -> RepoIndex:
    return RepoIndex.build(ROOT, sorted(ROOT.rglob("*.py")))


# --- resolution strategies -------------------------------------------------


def test_annotated_attribute_resolves_to_the_sdk(index: RepoIndex):
    found = index.resolve_function("app.service:AnnotatedService.handle")
    assert found is not None and found.provider == "openai"


def test_constructed_attribute_resolves_to_the_sdk(index: RepoIndex):
    found = index.resolve_function("app.service:ConstructedService.handle")
    assert found is not None and found.provider == "openai"


def test_module_singleton_resolves_to_the_sdk(index: RepoIndex):
    found = index.resolve_function("app.singleton:go")
    assert found is not None and found.provider == "openai"


def test_local_variable_resolves_to_the_sdk(index: RepoIndex):
    found = index.resolve_function("app.service:free_function")
    assert found is not None and found.provider == "openai"


def test_function_reaching_nothing_metered_does_not_resolve(index: RepoIndex):
    assert index.resolve_function("app.singleton:not_metered") is None


# --- depth and confidence --------------------------------------------------


def test_direct_sdk_call_is_depth_zero_and_certain(index: RepoIndex):
    found = index.resolve_function("app.clients:LLMClient.chat")
    assert (found.depth, found.confidence) == (0, 1.0)


def test_one_hop_wrapper_is_depth_one(index: RepoIndex):
    assert index.resolve_function("app.service:AnnotatedService.handle").depth == 1


def test_confidence_decays_with_depth(index: RepoIndex):
    shallow = index.resolve_function("app.deep:Layer3.run")
    deep = index.resolve_function("app.deep:Layer1.run")
    assert deep.depth > shallow.depth
    assert deep.confidence < shallow.confidence


def test_annotation_is_trusted_more_than_construction(index: RepoIndex):
    """Both reach the SDK in one hop; the evidence behind them differs."""
    annotated = index.resolve_function("app.service:AnnotatedService.handle")
    constructed = index.resolve_function("app.service:ConstructedService.handle")
    assert annotated.confidence > constructed.confidence


def test_resolution_records_the_chain_it_walked(index: RepoIndex):
    found = index.resolve_function("app.deep:Layer1.run")
    assert found.via[0] == "app.deep:Layer1.run"
    assert found.via[-1] == "app.clients:LLMClient.chat"


def test_mutual_recursion_terminates(index: RepoIndex):
    """A cycle with no metered call must return None rather than hang."""
    assert index.resolve_function("app.cycles:ping") is None


def test_depth_limit_is_five(index: RepoIndex):
    assert MAX_DEPTH == 5


# --- call sites ------------------------------------------------------------


def test_wrapper_call_becomes_a_call_site(index: RepoIndex):
    """`self.llm.chat(...)` is invisible to the extractor alone."""
    sites = index.call_sites("app/service.py")
    assert [s.shape for s in sites if s.qualname == "AnnotatedService.handle"] == ["wrapper"]


def test_wrapper_call_site_carries_confidence_below_one(index: RepoIndex):
    site = next(
        s for s in index.call_sites("app/service.py") if s.qualname == "AnnotatedService.handle"
    )
    assert 0.0 < site.confidence < 1.0


def test_direct_call_site_is_still_found_in_the_wrapper_module(index: RepoIndex):
    sites = index.call_sites("app/clients.py")
    assert {s.shape for s in sites} == {"sdk"}


def test_call_site_ids_stay_unique_once_wrappers_are_included(index: RepoIndex):
    ids = [s.id for s in index.call_sites("app/service.py")]
    assert len(ids) == len(set(ids))


def test_indirect_hop_through_self_method_resolves(index: RepoIndex):
    sites = index.call_sites("app/service.py")
    assert any(s.qualname == "ConstructedService.indirect" for s in sites)


# --- blast radius ----------------------------------------------------------


def test_editing_the_wrapper_reaches_callers_in_other_files(index: RepoIndex):
    """The headline output: a diff touching only clients.py is not local."""
    reached = index.blast_radius("app.clients:LLMClient.chat")
    assert "app.service:AnnotatedService.handle" in reached
    assert "app.singleton:go" in reached


def test_blast_radius_is_transitive(index: RepoIndex):
    reached = index.blast_radius("app.clients:LLMClient.chat")
    assert "app.deep:Layer1.run" in reached


def test_blast_radius_of_an_unreferenced_function_is_empty(index: RepoIndex):
    """`UnusedClient.chat` shares a method name with the real client and is
    reached by nobody. A name collision must not manufacture callers."""
    assert index.blast_radius("app.clients:UnusedClient.chat") == set()
