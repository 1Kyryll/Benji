"""Matching call sites across two versions.

Each test here is one way a diff can lie about what happened. The failure mode
being guarded against is always the same: telling the user a call became more
expensive when in truth an unrelated call was added next to it, or carrying a
site's cost history onto a call that has nothing to do with it.
"""

from __future__ import annotations

from benji.extract import extract
from benji.identity import content_hash, match

CALL = 'client.chat.completions.create(model="{model}", messages=[])'


def sites(body: str, file: str = "app.py"):
    return extract(body, file)


def one_call(model: str = "gpt-4o-mini", func: str = "handle", file: str = "app.py"):
    return sites(f"def {func}(x):\n    return {CALL.format(model=model)}\n", file)


# --- the stable cases -----------------------------------------------------


def test_identical_files_match_by_id():
    base = head = one_call()
    result = match(base, head)
    assert [m.method for m in result.matched] == ["id"]
    assert result.added == () and result.removed == ()


def test_inserting_an_import_changes_nothing():
    body = f"def handle(x):\n    return {CALL.format(model='gpt-4o')}\n"
    result = match(sites(body), sites("import os\n\n" + body))
    assert [m.method for m in result.matched] == ["id"]


def test_reformatting_does_not_break_the_match():
    """Content is normalised by ast.unparse, so whitespace is not evidence."""
    tight = 'def handle(x):\n    return client.chat.completions.create(model="gpt-4o")\n'
    loose = (
        "def handle(x):\n"
        "    return client.chat.completions.create(\n"
        '        model="gpt-4o",\n'
        "    )\n"
    )
    assert match(sites(tight), sites(loose)).matched[0].method == "id"


# --- the case the bot exists for ------------------------------------------


def test_model_swap_matches_and_is_reported_as_edited():
    result = match(one_call("gpt-4o-mini"), one_call("gpt-4o"))
    assert [m.method for m in result.matched] == ["id-edited"]
    assert result.edited and result.model_changes


def test_model_swap_is_not_reported_as_add_and_remove():
    """Losing the pairing would lose the delta, which is the whole product."""
    result = match(one_call("gpt-4o-mini"), one_call("gpt-4o"))
    assert result.added == () and result.removed == ()


# --- position breaks, content rescues -------------------------------------


def test_renamed_function_matches_on_content():
    result = match(one_call(func="handle"), one_call(func="triage"))
    assert [m.method for m in result.matched] == ["content-same-file"]


def test_call_inserted_above_does_not_steal_the_older_site():
    """Ordinals all shift, so the identifiers lie. The untouched call must not be
    reported as edited, and the genuinely new one must be reported as added."""
    base = sites(f"def handle(x):\n    return {CALL.format(model='gpt-4o')}\n")
    head = sites(
        "def handle(x):\n"
        f"    first = {CALL.format(model='gpt-4o-mini')}\n"
        f"    return {CALL.format(model='gpt-4o')}\n"
    )
    result = match(base, head)
    assert [m.method for m in result.matched] == ["content-same-file"]
    assert not result.edited
    assert [s.model for s in result.added] == ["gpt-4o-mini"]
    assert result.removed == ()


def test_moved_file_matches_with_lower_confidence():
    result = match(one_call(file="old.py"), one_call(file="new.py"))
    assert [m.method for m in result.matched] == ["content-moved"]
    assert result.matched[0].confidence < 1.0


def test_confidence_decays_down_the_match_chain():
    exact = match(one_call(), one_call()).matched[0]
    renamed = match(one_call(func="a"), one_call(func="b")).matched[0]
    moved = match(one_call(file="a.py"), one_call(file="b.py")).matched[0]
    assert exact.confidence > renamed.confidence > moved.confidence


# --- refusing to guess ----------------------------------------------------


def test_ambiguous_content_matches_nothing():
    """Two identical calls hash identically. Picking one would file real
    telemetry under the wrong site, and nothing downstream would ever show it."""
    body = "def a(x):\n    return {c}\n\n\ndef b(x):\n    return {c}\n"
    base = sites(body.format(c=CALL.format(model="gpt-4o")))
    head = sites(
        body.replace("def a", "def c")
        .replace("def b", "def d")
        .format(c=CALL.format(model="gpt-4o"))
    )
    result = match(base, head)
    assert result.matched == ()
    assert len(result.added) == 2 and len(result.removed) == 2


def test_a_genuinely_new_call_is_added_not_matched():
    base = []
    head = one_call()
    result = match(base, head)
    assert len(result.added) == 1 and result.matched == ()


def test_a_deleted_call_is_removed_not_matched():
    result = match(one_call(), [])
    assert len(result.removed) == 1 and result.matched == ()


def test_unrelated_call_replacing_another_is_not_treated_as_an_edit():
    """Same position, different provider. Reporting this as one site changing
    would carry the old site's history onto a call that never had it."""
    base = one_call()
    head = sites('def handle(x):\n    return c.messages.create(model="claude-sonnet-4")\n')
    result = match(base, head)
    # It matches positionally and is flagged edited, at reduced confidence —
    # visible, rather than silently confident.
    assert result.matched[0].method == "id-edited"
    assert result.matched[0].confidence < 1.0
    assert result.edited


# --- helpers --------------------------------------------------------------


def test_content_hash_ignores_formatting_but_not_arguments():
    same = one_call("gpt-4o")[0]
    reformatted = sites(
        "def handle(x):\n"
        "    return client.chat.completions.create(\n"
        '        model="gpt-4o", messages=[]\n'
        "    )\n"
    )[0]
    different = one_call("gpt-4o-mini")[0]
    assert content_hash(same) == content_hash(reformatted)
    assert content_hash(same) != content_hash(different)


def test_summary_counts_every_bucket():
    result = match(one_call("gpt-4o-mini"), one_call("gpt-4o"))
    assert result.summary() == "1 matched (1 edited), 0 added, 0 removed"
