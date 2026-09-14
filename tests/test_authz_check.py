"""``check()`` is the authority. These are the tests that say what that means.

Four properties, in the order the module docstring of ``authz/check.py`` claims
them: it is correct over nested groups, it terminates on cycles, it is bounded,
and it shows its work. The boundary cases are written as exact equalities on
purpose — "roughly the right depth" is not a security bound.
"""

from __future__ import annotations

import time

import pytest

from sightline.types import Decision, ObjectRef, PrincipalRef, Tuple_

from _world import WORLD_EXPECTATIONS

authz = pytest.importorskip("sightline.authz", reason="sightline.authz is not present yet")

check = authz.check
expand = authz.expand
expand_leaves = authz.expand_leaves
explain = authz.explain
MemoryTupleStore = authz.MemoryTupleStore
MAX_USERSET_DEPTH = authz.MAX_USERSET_DEPTH
DEPTH_LIMIT_NOTE = authz.DEPTH_LIMIT_NOTE


def _check(store, principal: str, obj: str, relation: str = "viewer") -> Decision:
    return check(store, ObjectRef.parse(obj), relation, PrincipalRef.parse(principal))


# --------------------------------------------------------------------------
# The truth table
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("principal", "obj", "expected"), WORLD_EXPECTATIONS)
def test_world_matches_the_table_a_human_wrote(world, principal, obj, expected):
    """Every answer is checked against a hand-written table, not against code.

    A test that derives its expectation from the implementation proves only that
    the implementation is self-consistent, which is also true of a broken one.
    """
    assert _check(world, principal, obj).allowed is expected


def test_absence_is_denial_not_unknown(world):
    """No tuple means deny. There is no third state and no "unknown" to widen."""
    decision = _check(world, "user:dave", "doc:merger")
    assert decision.allowed is False
    assert decision.why, "a denial must name the edges it tried"


def test_nonexistent_object_denies_like_a_forbidden_one(world):
    """The evaluator gives the same answer for "no such doc" and "not yours".

    The difference is not expressible here, which is what lets the HTTP layer
    keep it unexpressible. ``tests/test_existence_protection.py`` carries the
    end-to-end half of this.
    """
    missing = _check(world, "user:alice", "doc:does-not-exist")
    forbidden = _check(world, "user:alice", "doc:bob-notes")
    assert missing.allowed is forbidden.allowed is False


# --------------------------------------------------------------------------
# Nested groups
# --------------------------------------------------------------------------


def test_nested_groups_transit_three_levels(world):
    """alice -> group:legal -> group:partners -> group:everyone -> doc:handbook."""
    decision = _check(world, "user:alice", "doc:handbook")
    assert decision.allowed is True
    # The path is the whole chain, not just the first and last edge.
    assert len(decision.why) == 4
    assert decision.why[0] == "doc:handbook#viewer@group:everyone#member"
    assert decision.why[-1] == "group:legal#member@user:alice"


def test_membership_is_not_transitive_the_wrong_way(world):
    """Being granted through a group does not make you that group's member.

    ``doc:merger`` is granted to ``group:legal#member``. Nothing in the store
    says a viewer of the merger memo is a member of legal, and an evaluator that
    walked the grant backwards would say otherwise.
    """
    assert _check(world, "user:carol", "doc:merger").allowed is False
    assert check(
        world, ObjectRef("group", "legal"), "member", PrincipalRef("user", "carol")
    ).allowed is False


def test_sibling_group_name_is_not_a_prefix_match(world):
    """``group:legal-interns`` must not inherit from ``group:legal`` (M9).

    Comparing unparsed subject strings is one line shorter and works on every
    fixture anybody writes by hand, right up until a real organisation creates
    the neighbouring group.
    """
    assert _check(world, "user:mallory", "doc:merger").allowed is False
    # And the reverse index itself must not match by prefix.
    legal = world.read_by_principal(PrincipalRef("group", "legal", "member"))
    assert all(t.principal.id == "legal" for t in legal)
    assert any(t.object == ObjectRef("doc", "merger") for t in legal)


def test_owner_implies_viewer_through_computed_userset(world):
    """``viewer <- computed(editor) <- computed(owner)``, two rewrites deep."""
    assert _check(world, "user:alice", "doc:draft", "owner").allowed is True
    assert _check(world, "user:alice", "doc:draft", "editor").allowed is True
    assert _check(world, "user:alice", "doc:draft", "viewer").allowed is True


def test_folder_inheritance_flows_to_the_document(world):
    """carol holds ``viewer`` on ``doc:payroll`` only through ``folder:hr``."""
    decision = _check(world, "user:carol", "doc:payroll")
    assert decision.allowed is True
    assert any("folder:hr" in edge for edge in decision.why)
    # Removing the containment tuple removes the access, with nothing else changed.
    world.delete("doc:payroll#parent@folder:hr")
    assert _check(world, "user:carol", "doc:payroll").allowed is False


# --------------------------------------------------------------------------
# Cycles
# --------------------------------------------------------------------------


def _cyclic_store():
    return MemoryTupleStore(
        (
            # Written two years apart by two admins, as the docstring says.
            "group:leads#member@group:seniors#member",
            "group:seniors#member@group:leads#member",
            "doc:x#viewer@group:leads#member",
            # A self-referencing group, because somebody always does this.
            "group:loop#member@group:loop#member",
            "doc:y#viewer@group:loop#member",
        )
    )


@pytest.mark.parametrize("obj", ["doc:x", "doc:y"])
def test_cycle_terminates_and_denies(obj):
    """A cycle ends the branch. It does not hang, raise, or grant.

    The wall-clock bound is deliberately loose: it is a hang detector, not a
    performance assertion, and a performance assertion on shared CI hardware is
    a test that fails for reasons unrelated to the code.
    """
    store = _cyclic_store()
    started = time.perf_counter()
    decision = _check(store, "user:zed", obj)
    elapsed = time.perf_counter() - started

    assert decision.allowed is False
    assert elapsed < 5.0, "cycle detection did not terminate"
    assert any(note.startswith("cycle:") for note in decision.why), decision.why


def test_a_cycle_does_not_hide_a_real_grant():
    """Termination must not cost the derivation that exists alongside the loop."""
    store = _cyclic_store()
    store.write("group:seniors#member@user:zed")
    assert _check(store, "user:zed", "doc:x").allowed is True


# --------------------------------------------------------------------------
# Depth bounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [1, 2, 4, 8])
def test_depth_bound_is_exact_on_both_sides(group_chain, limit):
    """``limit`` nested groups resolve; ``limit + 1`` do not. Off by one is M5.

    Both sides are asserted in one test because only the pair is meaningful: a
    bound that never allows passes the "too deep is denied" half perfectly.
    """
    principal = PrincipalRef("user", "deep-user")
    doc = ObjectRef("doc", "deep")

    at_bound = check(group_chain(limit), doc, "viewer", principal, max_depth=limit)
    assert at_bound.allowed is True, f"a chain of {limit} must resolve at max_depth={limit}"

    past_bound = check(group_chain(limit + 1), doc, "viewer", principal, max_depth=limit)
    assert past_bound.allowed is False, f"a chain of {limit + 1} must not resolve"
    assert any(note.startswith(DEPTH_LIMIT_NOTE) for note in past_bound.why), past_bound.why


def test_default_depth_bound_is_the_published_one(group_chain):
    """``MAX_USERSET_DEPTH`` is a number other modules budget against.

    ``check.py`` splits it into an object half and a subject half so a grant-token
    match can never imply a path the evaluator would refuse. If this constant
    moves, that arithmetic moves with it.
    """
    principal = PrincipalRef("user", "deep-user")
    doc = ObjectRef("doc", "deep")
    assert check(group_chain(MAX_USERSET_DEPTH), doc, "viewer", principal).allowed is True
    assert check(group_chain(MAX_USERSET_DEPTH + 1), doc, "viewer", principal).allowed is False


def test_depth_limit_resolves_toward_deny(group_chain):
    """The truncated walk denies. Truncating toward allow is the breach."""
    decision = check(
        group_chain(6), ObjectRef("doc", "deep"), "viewer",
        PrincipalRef("user", "deep-user"), max_depth=2,
    )
    assert decision.allowed is False


# --------------------------------------------------------------------------
# Derivations
# --------------------------------------------------------------------------


def test_allow_derivation_replays_tuple_by_tuple(world):
    """Every edge in ``why`` parses as a tuple and is present in the store.

    This is what makes the admin console's "why can they see this?" answerable
    without a second authorisation pass — and what makes a fabricated derivation
    a test failure rather than a plausible-looking string.
    """
    decision = _check(world, "user:alice", "doc:handbook")
    assert decision.allowed is True
    for edge in decision.why:
        parsed = Tuple_.parse(edge)
        assert parsed in world.read(parsed.object, parsed.relation), f"invented edge: {edge}"


def test_derivation_is_a_connected_chain(world):
    """Consecutive edges join: each tuple's principal is the next tuple's object."""
    why = _check(world, "user:alice", "doc:handbook").why
    edges = [Tuple_.parse(e) for e in why]
    for upper, lower in zip(edges, edges[1:]):
        assert upper.principal.namespace == lower.object.namespace
        assert upper.principal.id == lower.object.id
        assert upper.principal.relation == lower.relation


def test_deny_derivation_names_the_edges_it_tried(world):
    """"Denied" on its own is unactionable for the admin who has to fix it."""
    decision = _check(world, "user:mallory", "doc:merger")
    assert decision.allowed is False
    assert any("doc:merger#viewer" in note for note in decision.why)
    assert decision.checked_tuples > 0


def test_explain_carries_both_the_decision_and_the_tree(world):
    """``/v1/explain`` is admin-only precisely because this tree names groups."""
    result = explain(world, ObjectRef("doc", "merger"), "viewer", PrincipalRef("user", "alice"))
    assert result.allowed is True
    assert result.depth_reached >= 1
    blob = result.to_dict()
    assert blob["allowed"] is True
    assert blob["tree"]["object"] == "doc:merger"


# --------------------------------------------------------------------------
# expand(): the object side stops at the group edge
# --------------------------------------------------------------------------


def test_expand_leaves_stop_at_the_userset(world):
    """Ingest stamps groups, never people (ADR 0002).

    If this ever returns ``user:alice``, every membership change becomes a vector
    rewrite storm and the index starts carrying a list of who works here.
    """
    leaves = expand_leaves(world, ObjectRef("doc", "merger"), "viewer")
    assert PrincipalRef("group", "legal", "member") in leaves
    assert PrincipalRef("user", "alice") not in leaves
    assert all(leaf.relation is not None or leaf.namespace != "group" for leaf in leaves)


def test_expand_follows_folder_inheritance_but_not_membership(world):
    """A document's stamped set includes what it inherits, and nothing about people."""
    leaves = expand_leaves(world, ObjectRef("doc", "payroll"), "viewer")
    assert PrincipalRef("group", "hr", "member") in leaves
    assert PrincipalRef("user", "carol") not in leaves


def test_expand_terminates_on_a_cycle():
    """The object side has its own cycle guard, and it reports the truncation."""
    store = MemoryTupleStore(
        ("folder:a#parent@folder:b", "folder:b#parent@folder:a", "doc:z#parent@folder:a")
    )
    tree = expand(store, ObjectRef("doc", "z"), "viewer")
    assert tree.leaves() == frozenset()
