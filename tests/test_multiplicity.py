"""How many times a call site fires per invocation of the function holding it.

The failures guarded against here are of two kinds: multiplying a cost that is
never paid, and inventing a bound for a loop whose size nobody knows.
"""

from __future__ import annotations

from chihuahuabot.estimate import Range
from chihuahuabot.multiplicity import Multiplicity, apply_declared, multiplicities

CALL = 'client.chat.completions.create(model="gpt-4o", messages=[])'


def only(source: str) -> Multiplicity:
    """The multiplicity of the single metered call in `source`."""
    found = multiplicities(source, "app.py")
    assert len(found) == 1, f"expected one call site, found {len(found)}"
    return next(iter(found.values()))


# --- the baseline ---------------------------------------------------------


def test_an_unwrapped_call_fires_once():
    assert only(f"def f():\n    {CALL}\n").range == Range.exact(1)


# --- loops we can count ---------------------------------------------------


def test_literal_list_gives_an_exact_count():
    assert only(f"def f():\n    for x in [1, 2, 3]:\n        {CALL}\n").range == Range.exact(3)


def test_literal_range_gives_an_exact_count():
    assert only(f"def f():\n    for i in range(5):\n        {CALL}\n").range == Range.exact(5)


def test_range_with_bounds_is_counted_correctly():
    assert only(f"def f():\n    for i in range(2, 7):\n        {CALL}\n").range == Range.exact(5)


def test_nested_loops_multiply():
    source = f"def f():\n    for a in [1, 2]:\n        for b in range(3):\n            {CALL}\n"
    assert only(source).range == Range.exact(6)


# --- loops we cannot count ------------------------------------------------


def test_unknown_iterable_does_not_produce_a_number():
    """A bounded-looking number invented from an unbounded loop is the failure
    this layer exists to avoid."""
    result = only(f"def f(orgs):\n    for org in orgs:\n        {CALL}\n")
    assert result.range is None and not result.resolved


def test_unknown_iterable_is_named_so_it_can_be_blamed():
    """'the range is driven by len(orgs)' needs the name."""
    result = only(f"def f(orgs):\n    for org in orgs:\n        {CALL}\n")
    assert [f.name for f in result.unknowns] == ["orgs"]


def test_unknown_iterable_carries_a_scoped_config_key():
    result = only(f"def handle(orgs):\n    for org in orgs:\n        {CALL}\n")
    assert result.unknowns[0].key == "handle:orgs"


def test_enumerate_is_seen_through_to_the_real_iterable():
    result = only(f"def f(orgs):\n    for i, o in enumerate(orgs):\n        {CALL}\n")
    assert result.unknowns[0].name == "orgs"


def test_while_loop_is_unresolved():
    assert only(f"def f():\n    while go():\n        {CALL}\n").range is None


def test_one_unknown_factor_poisons_the_whole_product():
    source = f"def f(orgs):\n    for a in [1, 2]:\n        for o in orgs:\n            {CALL}\n"
    assert only(source).range is None


# --- guards ---------------------------------------------------------------


def test_a_guarded_call_may_not_happen_at_all():
    result = only(f"def f(urgent):\n    if urgent:\n        {CALL}\n")
    assert result.range == Range(0, 0.5, 1)


def test_the_else_branch_is_guarded_too():
    result = only(f"def f(urgent):\n    if urgent:\n        pass\n    else:\n        {CALL}\n")
    assert result.range == Range(0, 0.5, 1)


def test_a_call_in_an_except_handler_is_guarded():
    source = f"def f():\n    try:\n        go()\n    except Exception:\n        {CALL}\n"
    assert only(source).range == Range(0, 0.5, 1)


def test_a_call_in_a_try_body_is_not_guarded():
    source = f"def f():\n    try:\n        {CALL}\n    except Exception:\n        pass\n"
    assert only(source).range == Range.exact(1)


# --- what does not multiply -----------------------------------------------


def test_the_loop_iterable_itself_runs_once():
    """`for x in llm(...)` evaluates the call once, not once per item."""
    source = f"def f():\n    for x in {CALL}:\n        pass\n"
    assert only(source).range == Range.exact(1)


def test_the_if_test_itself_is_not_guarded():
    source = f"def f():\n    if {CALL}:\n        pass\n"
    assert only(source).range == Range.exact(1)


def test_a_nested_function_resets_the_count():
    """Defining a function inside a loop does not run its body there."""
    source = f"def outer(orgs):\n    for o in orgs:\n        def inner():\n            {CALL}\n"
    assert only(source).range == Range.exact(1)


# --- comprehensions -------------------------------------------------------


def test_a_comprehension_is_a_loop():
    source = f"def f():\n    return [{CALL} for x in [1, 2, 3]]\n"
    assert only(source).range == Range.exact(3)


def test_a_comprehension_condition_is_a_guard():
    source = f"def f(items):\n    return [{CALL} for x in [1, 2] if x.ok]\n"
    assert only(source).range == Range(0, 1.0, 2)


# --- retries --------------------------------------------------------------


def test_a_retry_decorator_bounds_the_attempts():
    source = f"@retry(max_tries=3)\ndef f():\n    {CALL}\n"
    assert only(source).range == Range(1, 1.1, 3)


def test_tenacity_stop_after_attempt_is_understood():
    source = f"@retry(stop=stop_after_attempt(4))\ndef f():\n    {CALL}\n"
    assert only(source).range.high == 4


def test_backoff_decorator_is_understood():
    source = f"@backoff.on_exception(backoff.expo, Exception, max_tries=5)\ndef f():\n    {CALL}\n"
    assert only(source).range.high == 5


def test_a_retry_without_a_declared_limit_is_unresolved():
    """An unbounded retry is genuinely unbounded, and saying so beats guessing."""
    source = f"@retry\ndef f():\n    {CALL}\n"
    assert only(source).range is None


def test_retries_multiply_with_loops():
    source = f"@retry(max_tries=3)\ndef f():\n    for x in [1, 2]:\n        {CALL}\n"
    assert only(source).range == Range(2, 2.2, 6)


def test_an_unrelated_decorator_does_not_multiply():
    source = f"@app.route('/x')\ndef f():\n    {CALL}\n"
    assert only(source).range == Range.exact(1)


# --- declaring the unknowns -----------------------------------------------


def test_a_declared_size_resolves_the_loop():
    """This is how `for org in orgs` becomes a number."""
    result = only(f"def handle(orgs):\n    for org in orgs:\n        {CALL}\n")
    filled = apply_declared(result, {"handle:orgs": Range(3, 40, 500)})
    assert filled.range == Range(3, 40, 500)


def test_a_declaration_may_be_keyed_by_bare_name():
    result = only(f"def handle(orgs):\n    for org in orgs:\n        {CALL}\n")
    assert apply_declared(result, {"orgs": Range(1, 2, 3)}).resolved


def test_an_undeclared_loop_stays_unresolved():
    """Nothing is invented."""
    result = only(f"def handle(orgs):\n    for org in orgs:\n        {CALL}\n")
    assert apply_declared(result, {"something-else": Range(1, 2, 3)}).range is None


def test_describe_names_the_unknown_rather_than_hiding_it():
    result = only(f"def f(orgs):\n    for o in orgs:\n        {CALL}\n")
    assert "orgs" in result.describe()
