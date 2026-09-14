"""The tuple store: permission facts, the namespace configs that give them
meaning, and the monotonic policy epoch.

This module holds *storage*, not *semantics*. It knows how to put a tuple in and
get it back out quickly in both directions; it does not know what ``viewer``
means. Deciding whether a principal holds a relation is
:mod:`sightline.authz.check`, and it is deliberately a separate file: mixing the
evaluator into the index is how you end up with an index that quietly decides
things.

Two backends sit behind one Protocol:

* :class:`MemoryTupleStore` — dicts and sets, no dependencies at all. This is the
  one the whole test suite runs on, so the suite has no setup step.
* :class:`SQLiteTupleStore` — durable, and the epoch increment shares a
  transaction with the tuple write (planted mutant M15 moves it outside).
  ``sqlite3`` is in the standard library, so this backend costs nothing to
  install. Postgres would be the production choice; it is an extra and lives
  elsewhere.

TWO INDEXES, NOT ONE
--------------------
Both directions are needed and they have genuinely different access patterns:

* **Forward** — "who holds ``viewer`` on ``doc:42``". Always asked with a known
  relation, one object at a time, on the hot path of ``check()``. Keyed on
  ``(object, relation)``.
* **Reverse** — "what does ``user:alice`` hold". Asked *without* a relation,
  because plan compilation wants every grant a subject has, and then walks the
  result transitively. Keyed on the principal alone, filtered in memory when a
  relation is supplied.

There is also one derived view, :meth:`TupleStore.subject_keys`: the set of
usersets that appear on the principal side of any tuple — in practice, the
groups. It is what lets plan compilation tell "a group I am in" apart from "a
document I can read", and it is cached against the epoch, so a policy write
invalidates it and nothing else can.

Keying the reverse index on a prefix of the principal string instead of on its
parsed parts is planted mutant M9: ``group:legal`` would match
``group:legal-interns``. Everything here compares parsed fields.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from sightline.types import ObjectRef, PrincipalRef, Relation, Tuple_

__all__ = [
    "This",
    "ComputedUserset",
    "TupleToUserset",
    "Union",
    "Rewrite",
    "NamespaceConfig",
    "DEFAULT_NAMESPACES",
    "TupleStoreStats",
    "TupleStore",
    "MemoryTupleStore",
    "SQLiteTupleStore",
    "parse_tuples",
    "rewrite_to_json",
    "rewrite_from_json",
]


# --------------------------------------------------------------------------
# Rewrite rules
# --------------------------------------------------------------------------
#
# v1 supports three rewrite forms and no more. Intersection and exclusion
# ("viewer AND NOT denied") are not here, and their absence is a decision rather
# than an oversight: a half-implemented exclusion is a deny that silently does
# not deny, which is worse than no exclusion at all.


@dataclass(frozen=True, slots=True)
class This:
    """Direct tuples only. The base case of every relation."""


@dataclass(frozen=True, slots=True)
class ComputedUserset:
    """Another relation on the *same* object implies this one.

    ``viewer = union(this, computed(owner))`` reads as "an owner is also a
    viewer". This is the member of a union the FRD calls a "constituent
    relation"; it is named explicitly here because an unnamed concept is a
    concept nobody tests.
    """

    relation: Relation


@dataclass(frozen=True, slots=True)
class TupleToUserset:
    """Inheritance through another object: folder permissions flowing to docs.

    ``TupleToUserset("parent", "viewer")`` on ``doc.viewer`` reads as: for each
    object that is this document's ``parent``, whoever holds ``viewer`` there
    holds ``viewer`` here.

    ``both_directions`` exists because the containment tuple has two plausible
    spellings and the spec uses the second one:

    * ``doc:42#parent@folder:hr``  — the Zanzibar spelling.
    * ``folder:hr#parent@doc:42``  — how ``docs/FRD.md`` §1.1 writes it.

    Accepting both costs one extra index lookup per expansion step. Accepting
    only one of them costs silent recall loss on a corpus that used the other,
    and recall loss in the authority is not visible from the outside — it just
    looks like the document does not exist. Set this to ``False`` on a namespace
    where containment tuples are known to be one-directional.
    """

    tupleset: Relation
    computed_relation: Relation
    both_directions: bool = True


@dataclass(frozen=True, slots=True)
class Union:
    """Satisfied if any child is. The only combinator in v1."""

    children: tuple["Rewrite", ...]


Rewrite = This | ComputedUserset | TupleToUserset | Union


def rewrite_to_json(rw: Rewrite) -> dict[str, object]:
    """Serialise a rewrite so a durable backend can store namespace configs."""
    match rw:
        case This():
            return {"kind": "this"}
        case ComputedUserset(relation=rel):
            return {"kind": "computed_userset", "relation": rel}
        case TupleToUserset(tupleset=ts, computed_relation=cr, both_directions=bd):
            return {
                "kind": "tuple_to_userset",
                "tupleset": ts,
                "computed_relation": cr,
                "both_directions": bd,
            }
        case Union(children=kids):
            return {"kind": "union", "children": [rewrite_to_json(k) for k in kids]}
    raise TypeError(f"unserialisable rewrite: {rw!r}")


def rewrite_from_json(blob: Mapping[str, object]) -> Rewrite:
    """Inverse of :func:`rewrite_to_json`. Unknown kinds raise rather than
    degrading to ``This``: silently narrowing a rewrite would silently change
    who can see what."""
    kind = blob.get("kind")
    if kind == "this":
        return This()
    if kind == "computed_userset":
        return ComputedUserset(str(blob["relation"]))
    if kind == "tuple_to_userset":
        return TupleToUserset(
            str(blob["tupleset"]),
            str(blob["computed_relation"]),
            bool(blob.get("both_directions", True)),
        )
    if kind == "union":
        kids = blob.get("children") or []
        assert isinstance(kids, list)
        return Union(tuple(rewrite_from_json(k) for k in kids))
    raise ValueError(f"unknown rewrite kind: {kind!r}")


@dataclass(frozen=True, slots=True)
class NamespaceConfig:
    """Relation definitions for one namespace.

    A namespace change also bumps the epoch, because it changes what already
    stored tuples *mean*. That is a policy write even though no tuple moved.
    """

    name: str
    relations: Mapping[str, Rewrite] = field(default_factory=dict)

    def rewrite(self, relation: Relation) -> Rewrite:
        """Rewrite for ``relation``, defaulting to direct tuples only.

        An undeclared relation resolves to :class:`This` rather than raising.
        The default can only ever *lose* derivations, never invent one, so an
        unconfigured namespace fails closed and stays usable for ad-hoc objects
        (``tenant:acme#admin``) that nobody wrote a config for.
        """
        return self.relations.get(relation, This())

    def to_json(self) -> str:
        relations = {r: rewrite_to_json(v) for r, v in self.relations.items()}
        return json.dumps({"name": self.name, "relations": relations}, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "NamespaceConfig":
        blob = json.loads(raw)
        return cls(blob["name"], {r: rewrite_from_json(v) for r, v in blob["relations"].items()})


def _default_namespaces() -> dict[str, NamespaceConfig]:
    """The three namespaces every fixture in this repo uses.

    ``group.member`` is plain ``This``: nested groups are not a feature of the
    group namespace, they fall out of a userset principal pointing at another
    userset. There is no group table.
    """
    doc = NamespaceConfig(
        "doc",
        {
            "owner": This(),
            "editor": Union((This(), ComputedUserset("owner"))),
            "viewer": Union(
                (
                    This(),
                    ComputedUserset("editor"),
                    TupleToUserset("parent", "viewer"),
                )
            ),
            "parent": This(),
        },
    )
    folder = NamespaceConfig(
        "folder",
        {
            "owner": This(),
            "editor": Union((This(), ComputedUserset("owner"))),
            "viewer": Union(
                (
                    This(),
                    ComputedUserset("editor"),
                    TupleToUserset("parent", "viewer"),
                )
            ),
            "parent": This(),
        },
    )
    group = NamespaceConfig("group", {"member": This(), "owner": This()})
    return {"doc": doc, "folder": folder, "group": group}


DEFAULT_NAMESPACES: Mapping[str, NamespaceConfig] = _default_namespaces()


@dataclass(frozen=True, slots=True)
class TupleStoreStats:
    """What the store holds. Reported so capacity claims stay honest."""

    n_tuples: int
    n_objects: int
    n_principals: int
    epoch: int
    backend: str


def parse_tuples(raw: Iterable[str | Tuple_]) -> list[Tuple_]:
    """Accept tuples as strings or objects. Every write path takes both, because
    fixtures and the HTTP admin endpoint both speak the string form."""
    out: list[Tuple_] = []
    for item in raw:
        out.append(item if isinstance(item, Tuple_) else Tuple_.parse(item))
    return out


@runtime_checkable
class TupleStore(Protocol):
    """The authoritative store. Everything else in Sightline is a cache of it.

    Every method that changes policy returns the **new epoch**. Callers stamp
    that onto compiled plans, and a plan below the live epoch is never served.
    """

    backend: str

    # -- policy state ------------------------------------------------------
    def epoch(self) -> int:
        """Current policy epoch. The cheapest call in the system: every query
        makes it."""
        ...

    # -- writes ------------------------------------------------------------
    def write(self, *tuples: str | Tuple_) -> int:
        """Insert tuples. Idempotent on the whole tuple. Returns the new epoch."""
        ...

    def delete(self, *tuples: str | Tuple_) -> int:
        """Remove tuples. Deleting something that is not there is a no-op that
        **still** increments the epoch: over-incrementing costs a plan
        recompile, under-incrementing serves a stale plan."""
        ...

    def apply(
        self,
        writes: Sequence[str | Tuple_] = (),
        deletes: Sequence[str | Tuple_] = (),
    ) -> tuple[int, int]:
        """Apply deletes then writes in one transaction with one epoch bump.

        Returns ``(epoch, n_applied)``.
        """
        ...

    def define_namespace(self, config: NamespaceConfig) -> int:
        """Install a namespace config. Bumps the epoch."""
        ...

    # -- reads -------------------------------------------------------------
    def read(self, object: ObjectRef, relation: Relation | None = None) -> list[Tuple_]:
        """Forward index: tuples on ``object``, optionally one relation."""
        ...

    def read_by_principal(
        self, principal: PrincipalRef, relation: Relation | None = None
    ) -> list[Tuple_]:
        """Reverse index: tuples whose principal is exactly ``principal``.

        Exactly. ``group:legal`` does not match ``group:legal-interns``, and
        ``group:legal`` (no relation) does not match ``group:legal#member``.
        """
        ...

    def subject_keys(self) -> frozenset[tuple[str, str, str]]:
        """Every **userset** that appears on the principal side of some tuple.

        This is the node set of the subject graph — in practice, the groups.
        Plan compilation needs it to tell two things apart that look identical
        in the tuple store:

        * ``group:legal#member``, which somebody was granted *through*, and
        * ``doc:317#viewer``, which somebody simply holds.

        Walking the second kind is how a reverse closure ends up visiting one
        node per document in the corpus. Everything reachable *through* a
        userset is reachable only if that userset is named as a principal
        somewhere, so this set is exactly the part of the graph worth walking.

        Returned as ``(namespace, id, relation)`` string triples rather than
        :class:`~sightline.types.PrincipalRef` on purpose: this is tested once
        per candidate edge, of which a heavily-permissioned principal has
        hundreds of thousands, and constructing a frozen dataclass costs about
        7 microseconds on the reference machine. Implementations should cache
        the result against the epoch, since a policy write invalidates it and
        nothing else can.
        """
        ...

    def namespace_config(self, namespace: str) -> NamespaceConfig:
        ...

    def list_namespaces(self) -> list[str]:
        """Every configured namespace.

        Plan compilation needs it to walk folder inheritance backwards: the rule
        that says a document inherits from its folder is declared on the
        *document's* namespace, so finding what inherits from a folder means
        reading every namespace's rules rather than guessing the name of the
        containment relation.
        """
        ...

    def list_objects(self, namespace: str) -> list[ObjectRef]:
        """Every object in a namespace that appears on the object side of a
        tuple. Drives corpus-size estimates during plan compilation."""
        ...

    def count_objects(self, namespace: str) -> int:
        ...

    def iter_tuples(self) -> Iterator[Tuple_]:
        """Every tuple. For the oracle and for dumps, not for the query path."""
        ...

    def stats(self) -> TupleStoreStats:
        ...


class MemoryTupleStore:
    """In-process tuple store. No dependencies, no setup, no durability.

    The entire test suite runs on this, which is the point: a permission test
    that needs a database running is a permission test that gets skipped.
    """

    backend = "memory"

    def __init__(
        self,
        tuples: Iterable[str | Tuple_] = (),
        namespaces: Mapping[str, NamespaceConfig] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._epoch = 0
        self._tuples: set[Tuple_] = set()
        # Forward: (object, relation) -> tuples. Always queried with a relation.
        self._by_object: dict[tuple[str, str, str], set[Tuple_]] = {}
        # Which relations exist on an object, so read(object) needs no scan.
        self._relations: dict[tuple[str, str], set[str]] = {}
        # Reverse: principal -> tuples. Queried without a relation, then walked
        # transitively, so it is keyed on the parsed principal only.
        self._by_principal: dict[tuple[str, str, str | None], set[Tuple_]] = {}
        self._objects_by_ns: dict[str, set[str]] = {}
        # Derived view, rebuilt when the epoch moves. Caching store state
        # against the store's own epoch is safe in a way that caching *policy*
        # across an epoch change is not.
        self._subject_keys: tuple[int, frozenset[tuple[str, str, str]]] | None = None
        self._namespaces: dict[str, NamespaceConfig] = dict(namespaces or DEFAULT_NAMESPACES)
        if tuples:
            self.write(*tuples)

    # -- keys --------------------------------------------------------------
    @staticmethod
    def _okey(obj: ObjectRef, relation: str) -> tuple[str, str, str]:
        return (obj.namespace, obj.id, relation)

    @staticmethod
    def _pkey(p: PrincipalRef) -> tuple[str, str, str | None]:
        return (p.namespace, p.id, p.relation)

    # -- policy state ------------------------------------------------------
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def _bump(self) -> int:
        self._epoch += 1
        return self._epoch

    # -- writes ------------------------------------------------------------
    def write(self, *tuples: str | Tuple_) -> int:
        epoch, _ = self.apply(writes=parse_tuples(tuples))
        return epoch

    def delete(self, *tuples: str | Tuple_) -> int:
        epoch, _ = self.apply(deletes=parse_tuples(tuples))
        return epoch

    def apply(
        self,
        writes: Sequence[str | Tuple_] = (),
        deletes: Sequence[str | Tuple_] = (),
    ) -> tuple[int, int]:
        w = parse_tuples(writes)
        d = parse_tuples(deletes)
        with self._lock:
            applied = 0
            # Deletes first: a batch containing both spellings of the same fact
            # should end up with the fact present, not absent.
            for t in d:
                applied += int(self._remove(t))
            for t in w:
                applied += int(self._insert(t))
            return self._bump(), applied

    def _insert(self, t: Tuple_) -> bool:
        if t in self._tuples:
            return False
        self._tuples.add(t)
        self._by_object.setdefault(self._okey(t.object, t.relation), set()).add(t)
        self._relations.setdefault((t.object.namespace, t.object.id), set()).add(t.relation)
        self._by_principal.setdefault(self._pkey(t.principal), set()).add(t)
        self._objects_by_ns.setdefault(t.object.namespace, set()).add(t.object.id)
        return True

    def _remove(self, t: Tuple_) -> bool:
        if t not in self._tuples:
            return False
        self._tuples.discard(t)
        ok = self._okey(t.object, t.relation)
        bucket = self._by_object.get(ok)
        if bucket is not None:
            bucket.discard(t)
            if not bucket:
                del self._by_object[ok]
                rels = self._relations.get((t.object.namespace, t.object.id))
                if rels is not None:
                    rels.discard(t.relation)
                    if not rels:
                        del self._relations[(t.object.namespace, t.object.id)]
                        ids = self._objects_by_ns.get(t.object.namespace)
                        if ids is not None:
                            ids.discard(t.object.id)
        pk = self._pkey(t.principal)
        pbucket = self._by_principal.get(pk)
        if pbucket is not None:
            pbucket.discard(t)
            if not pbucket:
                del self._by_principal[pk]
        return True

    def define_namespace(self, config: NamespaceConfig) -> int:
        with self._lock:
            self._namespaces[config.name] = config
            return self._bump()

    # -- reads -------------------------------------------------------------
    def read(self, object: ObjectRef, relation: Relation | None = None) -> list[Tuple_]:
        with self._lock:
            if relation is not None:
                return list(self._by_object.get(self._okey(object, relation), ()))
            out: list[Tuple_] = []
            for rel in self._relations.get((object.namespace, object.id), ()):
                out.extend(self._by_object.get(self._okey(object, rel), ()))
            return out

    def read_by_principal(
        self, principal: PrincipalRef, relation: Relation | None = None
    ) -> list[Tuple_]:
        with self._lock:
            found = self._by_principal.get(self._pkey(principal), ())
            if relation is None:
                return list(found)
            return [t for t in found if t.relation == relation]

    def subject_keys(self) -> frozenset[tuple[str, str, str]]:
        with self._lock:
            cached = self._subject_keys
            if cached is not None and cached[0] == self._epoch:
                return cached[1]
            keys = frozenset(
                (ns, ident, rel)
                for ns, ident, rel in self._by_principal
                if rel is not None
            )
            self._subject_keys = (self._epoch, keys)
            return keys

    def namespace_config(self, namespace: str) -> NamespaceConfig:
        with self._lock:
            cfg = self._namespaces.get(namespace)
        return cfg if cfg is not None else NamespaceConfig(namespace)

    def list_namespaces(self) -> list[str]:
        with self._lock:
            return sorted(self._namespaces)

    def list_objects(self, namespace: str) -> list[ObjectRef]:
        with self._lock:
            return [ObjectRef(namespace, i) for i in self._objects_by_ns.get(namespace, ())]

    def count_objects(self, namespace: str) -> int:
        with self._lock:
            return len(self._objects_by_ns.get(namespace, ()))

    def iter_tuples(self) -> Iterator[Tuple_]:
        with self._lock:
            snapshot = list(self._tuples)
        return iter(snapshot)

    def stats(self) -> TupleStoreStats:
        with self._lock:
            return TupleStoreStats(
                n_tuples=len(self._tuples),
                n_objects=sum(len(v) for v in self._objects_by_ns.values()),
                n_principals=len(self._by_principal),
                epoch=self._epoch,
                backend=self.backend,
            )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tuples (
    object_ns     TEXT NOT NULL,
    object_id     TEXT NOT NULL,
    relation      TEXT NOT NULL,
    principal_ns  TEXT NOT NULL,
    principal_id  TEXT NOT NULL,
    principal_rel TEXT NOT NULL,   -- '' means "not a userset"; see the note below
    PRIMARY KEY (object_ns, object_id, relation, principal_ns, principal_id, principal_rel)
);
CREATE INDEX IF NOT EXISTS tuples_forward
    ON tuples (object_ns, object_id, relation);
CREATE INDEX IF NOT EXISTS tuples_reverse
    ON tuples (principal_ns, principal_id, principal_rel);
CREATE TABLE IF NOT EXISTS namespaces (
    name   TEXT PRIMARY KEY,
    config TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS policy (
    key   TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
INSERT OR IGNORE INTO policy (key, value) VALUES ('epoch', 0);
"""


class SQLiteTupleStore:
    """Durable tuple store on stdlib ``sqlite3``. No extra to install.

    Two things here are load-bearing rather than incidental:

    1. **The epoch increment is in the same transaction as the tuple write.**
       A crash between them would leave a policy change invisible to every
       cached plan, which is the leak this counter exists to prevent. Moving the
       increment out is planted mutant M15.
    2. **``principal_rel`` stores ``''`` rather than NULL** for a non-userset
       principal. NULL is not equal to NULL in SQL, so a nullable column here
       turns the primary key and the reverse-index lookup into ``IS`` special
       cases that are easy to get subtly wrong. An empty relation is not a legal
       tuple spelling, so the sentinel is unambiguous.
    """

    backend = "sqlite"

    def __init__(
        self,
        path: str | Path = ":memory:",
        namespaces: Mapping[str, NamespaceConfig] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._subject_keys: tuple[int, frozenset[tuple[str, str, str]]] | None = None
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._conn:
            self._conn.executescript(_SCHEMA)
        for cfg in (namespaces or DEFAULT_NAMESPACES).values():
            self._put_namespace_if_absent(cfg)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _row_to_tuple(row: Sequence[str]) -> Tuple_:
        return Tuple_(
            ObjectRef(row[0], row[1]),
            row[2],
            PrincipalRef(row[3], row[4], row[5] or None),
        )

    @staticmethod
    def _tuple_to_row(t: Tuple_) -> tuple[str, str, str, str, str, str]:
        return (
            t.object.namespace,
            t.object.id,
            t.relation,
            t.principal.namespace,
            t.principal.id,
            t.principal.relation or "",
        )

    def _put_namespace_if_absent(self, cfg: NamespaceConfig) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO namespaces (name, config) VALUES (?, ?)",
                (cfg.name, cfg.to_json()),
            )

    # -- policy state ------------------------------------------------------
    def epoch(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT value FROM policy WHERE key='epoch'").fetchone()
        return int(row[0]) if row else 0

    # -- writes ------------------------------------------------------------
    def write(self, *tuples: str | Tuple_) -> int:
        epoch, _ = self.apply(writes=parse_tuples(tuples))
        return epoch

    def delete(self, *tuples: str | Tuple_) -> int:
        epoch, _ = self.apply(deletes=parse_tuples(tuples))
        return epoch

    def apply(
        self,
        writes: Sequence[str | Tuple_] = (),
        deletes: Sequence[str | Tuple_] = (),
    ) -> tuple[int, int]:
        w = [self._tuple_to_row(t) for t in parse_tuples(writes)]
        d = [self._tuple_to_row(t) for t in parse_tuples(deletes)]
        with self._lock, self._conn:  # one transaction: rows and epoch together
            applied = 0
            for row in d:
                cur = self._conn.execute(
                    "DELETE FROM tuples WHERE object_ns=? AND object_id=? AND relation=?"
                    " AND principal_ns=? AND principal_id=? AND principal_rel=?",
                    row,
                )
                applied += max(0, cur.rowcount)
            for row in w:
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO tuples"
                    " (object_ns, object_id, relation, principal_ns, principal_id, principal_rel)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    row,
                )
                applied += max(0, cur.rowcount)
            self._conn.execute("UPDATE policy SET value = value + 1 WHERE key='epoch'")
            row2 = self._conn.execute("SELECT value FROM policy WHERE key='epoch'").fetchone()
            return int(row2[0]), applied

    def define_namespace(self, config: NamespaceConfig) -> int:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO namespaces (name, config) VALUES (?, ?)"
                " ON CONFLICT(name) DO UPDATE SET config=excluded.config",
                (config.name, config.to_json()),
            )
            self._conn.execute("UPDATE policy SET value = value + 1 WHERE key='epoch'")
            row = self._conn.execute("SELECT value FROM policy WHERE key='epoch'").fetchone()
            return int(row[0])

    # -- reads -------------------------------------------------------------
    def read(self, object: ObjectRef, relation: Relation | None = None) -> list[Tuple_]:
        sql = (
            "SELECT object_ns, object_id, relation, principal_ns, principal_id, principal_rel"
            " FROM tuples WHERE object_ns=? AND object_id=?"
        )
        args: list[str] = [object.namespace, object.id]
        if relation is not None:
            sql += " AND relation=?"
            args.append(relation)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._row_to_tuple(r) for r in rows]

    def read_by_principal(
        self, principal: PrincipalRef, relation: Relation | None = None
    ) -> list[Tuple_]:
        sql = (
            "SELECT object_ns, object_id, relation, principal_ns, principal_id, principal_rel"
            " FROM tuples WHERE principal_ns=? AND principal_id=? AND principal_rel=?"
        )
        args: list[str] = [principal.namespace, principal.id, principal.relation or ""]
        if relation is not None:
            sql += " AND relation=?"
            args.append(relation)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._row_to_tuple(r) for r in rows]

    def subject_keys(self) -> frozenset[tuple[str, str, str]]:
        epoch = self.epoch()
        with self._lock:
            cached = self._subject_keys
            if cached is not None and cached[0] == epoch:
                return cached[1]
            rows = self._conn.execute(
                "SELECT DISTINCT principal_ns, principal_id, principal_rel FROM tuples"
                " WHERE principal_rel != ''"
            ).fetchall()
            keys = frozenset((r[0], r[1], r[2]) for r in rows)
            self._subject_keys = (epoch, keys)
            return keys

    def namespace_config(self, namespace: str) -> NamespaceConfig:
        with self._lock:
            row = self._conn.execute(
                "SELECT config FROM namespaces WHERE name=?", (namespace,)
            ).fetchone()
        return NamespaceConfig.from_json(row[0]) if row else NamespaceConfig(namespace)

    def list_namespaces(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT name FROM namespaces ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def list_objects(self, namespace: str) -> list[ObjectRef]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT object_id FROM tuples WHERE object_ns=?", (namespace,)
            ).fetchall()
        return [ObjectRef(namespace, r[0]) for r in rows]

    def count_objects(self, namespace: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(DISTINCT object_id) FROM tuples WHERE object_ns=?", (namespace,)
            ).fetchone()
        return int(row[0]) if row else 0

    def iter_tuples(self) -> Iterator[Tuple_]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT object_ns, object_id, relation, principal_ns, principal_id, principal_rel"
                " FROM tuples"
            ).fetchall()
        return iter([self._row_to_tuple(r) for r in rows])

    def stats(self) -> TupleStoreStats:
        with self._lock:
            n_tuples = int(self._conn.execute("SELECT COUNT(*) FROM tuples").fetchone()[0])
            n_objects = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM (SELECT DISTINCT object_ns, object_id FROM tuples)"
                ).fetchone()[0]
            )
            n_principals = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM"
                    " (SELECT DISTINCT principal_ns, principal_id, principal_rel FROM tuples)"
                ).fetchone()[0]
            )
        return TupleStoreStats(n_tuples, n_objects, n_principals, self.epoch(), self.backend)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
