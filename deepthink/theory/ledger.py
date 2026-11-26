"""The body of work: one consistent ledger of results shared by the swarm.

Without it, an agent sees only its own thought stream: its persona, the cells
below it and its own past outputs. The ledger is where every QNN's work lands
and where every agent reads what the swarm has established.

**Writing.** Each cell a QNN solves or synthesizes becomes an entry: its claim,
its derivation, its answer, who wrote it (QNN, round, cell) and which entries
it was built on.

**Consistency.** The ledger keeps the body of work free of silent
contradictions:

* the same claim derived again (same statement, same answer) merges into one
  entry; a second QNN deriving it independently makes it *corroborated*;
* the same statement with a different answer, a declared ``same_as`` that
  disagrees, or a declared ``contradicts`` opens a **conflict**; both sides are
  *disputed* until an arbiter (or the verifier) settles it;
* the losing side is *refuted*, and every entry whose every derivation rests on
  a refuted or suspect entry becomes *suspect*; a suspect proposal loses vote
  weight;
* a verifier can *confirm* entries a proposal depends on (→ *verified*) or
  refute them.

**Reading.** Agents receive the entries most relevant to their cell (lexical
overlap, boosted by status), can cite them by id, and decomposers can mark a
subcell as already established so it is reused instead of solved again.
Proposals (root answers) are written but never shown to other agents, so QNNs
share lemmas, not verdicts. In ``private`` mode each QNN only reads its own
entries; conflicts are still detected across the swarm.

All of this is deterministic bookkeeping; only arbitration calls a model.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from deepthink.theory.answers import DISPROOF, PROOF, canonical_answer, same_answer

STATUSES = ("proposed", "corroborated", "verified", "disputed", "suspect", "refuted")
USABLE = frozenset({"proposed", "corroborated", "verified"})
TAINTED = frozenset({"refuted", "suspect"})
_STATUS_BOOST = {
    "verified": 0.35,
    "corroborated": 0.25,
    "proposed": 0.0,
    "disputed": -0.1,
    "suspect": -0.2,
    "refuted": -0.3,
}
_TRUE = {PROOF, "yes"}
_FALSE = {DISPROOF, "no"}

_STOP = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the this to was
    were will with show prove compute find determine let then there their these those which
    each every all any such given if we our cell""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: Any) -> frozenset[str]:
    """Content tokens for matching: lowercase words and numbers, no LaTeX commands."""
    s = re.sub(r"\\[a-zA-Z]+", " ", str(text or "")).lower()
    return frozenset(t for t in _TOKEN.findall(s) if (len(t) > 1 or t.isdigit()) and t not in _STOP)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _excerpt(text: str, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


@dataclass
class Entry:
    id: str
    kind: str
    statement: str
    result: str = ""
    answer: str = ""
    canon: str = ""
    status: str = "proposed"
    confidence: float = 0.0
    derivations: list[list[str]] = field(default_factory=list)
    uses: list[str] = field(default_factory=list)
    authors: list[dict[str, Any]] = field(default_factory=list)
    reused_by: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    round: int = 1
    proposal: bool = False
    held_status: str = ""

    @property
    def members(self) -> set[int]:
        return {int(a["member"]) for a in self.authors if "member" in a}

    @property
    def support(self) -> int:
        return len(self.members)

    @property
    def depends_on(self) -> list[str]:
        seen: list[str] = []
        for deriv in self.derivations:
            for d in deriv:
                if d not in seen:
                    seen.append(d)
        return seen

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("held_status", None)
        d["support"] = self.support
        d["depends_on"] = self.depends_on
        d["result"] = _excerpt(self.result, 4000)
        return d


@dataclass
class Conflict:
    """Two entries that cannot both hold, or (``b == ""``) one entry and a challenge to it."""

    id: str
    a: str
    b: str
    reason: str
    round: int = 1
    status: str = "open"  # open | resolved | dismissed
    verdict: str = ""  # a | b | neither | both
    note: str = ""
    challenge: str = ""  # the objection, when b is a challenge rather than an entry

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TheoryLedger:
    """The swarm's body of work (see module docstring)."""

    def __init__(
        self,
        *,
        tolerance: float = 0.0,
        answer_type: str = "",
        shared: bool = True,
        dedup: float = 0.8,
    ) -> None:
        self.tolerance = float(tolerance)
        self.answer_type = answer_type
        self.shared = bool(shared)
        self.dedup = float(dedup)
        self.entries: dict[str, Entry] = {}
        self.conflicts: dict[str, Conflict] = {}
        self.log: list[dict[str, Any]] = []
        self._tok: dict[str, frozenset[str]] = {}
        self._norm: dict[str, str] = {}
        self._seq = 0
        self._cseq = 0

    # ── basic access ────────────────────────────────────────────────
    def get(self, eid: str | None) -> Entry | None:
        return self.entries.get(eid) if eid else None

    def __len__(self) -> int:
        return len(self.entries)

    def _set_status(self, e: Entry, status: str, reason: str) -> None:
        if e.status == status:
            return
        self.log.append({"entry": e.id, "from": e.status, "to": status, "reason": reason})
        e.status = status
        if reason:
            e.notes.append(reason)

    def _agree(self, a: str, b: str) -> bool:
        if not a or not b:
            return True
        if a in _TRUE and b in _TRUE:
            return True
        if (a in _TRUE and b in _FALSE) or (a in _FALSE and b in _TRUE):
            return False
        return same_answer(a, b, self.tolerance) or a == b

    @staticmethod
    def _jaccard(x: frozenset[str], y: frozenset[str]) -> float:
        if not x or not y:
            return 0.0
        return len(x & y) / len(x | y)

    def _similar(
        self, toks: frozenset[str], norm: str, exclude: str = ""
    ) -> list[tuple[Entry, float]]:
        out: list[tuple[Entry, float]] = []
        for eid, e in self.entries.items():
            if e.proposal or eid == exclude:
                continue
            sim = (
                1.0 if norm and norm == self._norm.get(eid) else self._jaccard(toks, self._tok[eid])
            )
            if sim >= self.dedup:
                out.append((e, sim))
        out.sort(key=lambda t: (-t[1], t[0].id))
        return out

    # ── writing ─────────────────────────────────────────────────────
    def add(
        self,
        *,
        member: int,
        round_: int,
        cell: str,
        kind: str,
        statement: str,
        result: str = "",
        answer: str = "",
        confidence: float = 0.0,
        depends_on: list[str] | None = None,
        uses: list[str] | None = None,
        same_as: str | None = None,
        contradicts: list[str] | None = None,
        proposal: bool = False,
    ) -> tuple[Entry, list[Conflict]]:
        """Write one result. Returns the entry it landed in and any conflicts it opened."""
        statement = " ".join(str(statement or "").split())[:1500]
        canon = canonical_answer(answer, answer_type=self.answer_type if proposal else "")
        toks = tokens(statement)
        norm = _norm(statement)
        author = {"member": int(member), "round": int(round_), "cell": cell}
        cited = [u for u in dict.fromkeys(uses or []) if u in self.entries]
        deps = [d for d in dict.fromkeys((depends_on or []) + cited) if d in self.entries]
        conf = max(0.0, min(1.0, float(confidence or 0.0)))
        conflicts: list[Conflict] = []

        if not proposal:
            declared = self.get(same_as)
            if declared is not None and declared.proposal:
                declared = None
            candidates = [(declared, 1.0)] if declared is not None else self._similar(toks, norm)
            for match, _sim in candidates:
                if not self._agree(match.canon, canon):
                    continue
                self._merge(match, author, deps, result, conf, cited, member)
                return match, []

        self._seq += 1
        entry = Entry(
            id=f"T{self._seq}",
            kind=str(kind or "lemma")[:24],
            statement=statement,
            result=str(result or "")[:8000],
            answer=str(answer or "")[:300],
            canon=canon,
            confidence=conf,
            derivations=[deps],
            uses=cited,
            authors=[author],
            round=int(round_),
            proposal=bool(proposal),
        )
        self.entries[entry.id] = entry
        self._tok[entry.id] = toks
        self._norm[entry.id] = norm

        if not proposal:
            # A refuted entry is already settled: disagreeing with it is no conflict.
            for other, _sim in self._similar(toks, norm, exclude=entry.id):
                if other.status != "refuted" and not self._agree(other.canon, canon):
                    conflicts.append(
                        self._open(entry, other, "same statement, different answers", round_)
                    )
            declared = self.get(same_as)
            if (
                declared is not None
                and not declared.proposal
                and declared.id != entry.id
                and declared.status != "refuted"
            ):
                if not self._agree(declared.canon, canon):
                    conflicts.append(
                        self._open(entry, declared, f"QNN {member} declared it the same", round_)
                    )
            for cid in dict.fromkeys(contradicts or []):
                other_e = self.get(cid)
                if (
                    other_e is not None
                    and not other_e.proposal
                    and other_e.id != entry.id
                    and other_e.status != "refuted"
                ):
                    conflicts.append(
                        self._open(entry, other_e, f"QNN {member} reports a contradiction", round_)
                    )
        if deps and any(self.entries[d].status in TAINTED for d in deps):
            self._set_status(entry, "suspect", "built on a refuted or suspect entry")
        unique: dict[str, Conflict] = {c.id: c for c in conflicts}
        return entry, list(unique.values())

    def _merge(
        self,
        e: Entry,
        author: dict[str, Any],
        deps: list[str],
        result: str,
        conf: float,
        cited: list[str],
        member: int,
    ) -> None:
        new_member = int(member) not in e.members
        e.authors.append(author)
        if deps not in e.derivations:
            e.derivations.append(deps)
        for u in cited:
            if u not in e.uses:
                e.uses.append(u)
        if conf > e.confidence and result:
            e.result = str(result)[:8000]
        e.confidence = max(e.confidence, conf)
        if e.status == "refuted":
            e.notes.append(f"re-derived by QNN {member} after refutation")
        elif new_member and e.status == "proposed" and e.support >= 2:
            self._set_status(e, "corroborated", f"derived independently by {e.support} QNNs")
        elif e.status == "suspect" and not any(
            self.entries[d].status in TAINTED for d in deps if d in self.entries
        ):
            self._set_status(
                e, "corroborated" if e.support >= 2 else "proposed", "clean derivation"
            )

    def note_reuse(self, eid: str, *, member: int, round_: int, cell: str) -> Entry | None:
        e = self.get(eid)
        if e is None:
            return None
        e.reused_by.append({"member": int(member), "round": int(round_), "cell": cell})
        return e

    # ── conflicts ───────────────────────────────────────────────────
    def _open(self, a: Entry, b: Entry, reason: str, round_: int) -> Conflict:
        pair = tuple(sorted((a.id, b.id)))
        for c in self.conflicts.values():
            if c.status == "open" and tuple(sorted((c.a, c.b))) == pair:
                return c
        self._cseq += 1
        c = Conflict(id=f"C{self._cseq}", a=a.id, b=b.id, reason=reason, round=int(round_))
        self.conflicts[c.id] = c
        for e in (a, b):
            if e.status in {"proposed", "corroborated"}:
                e.held_status = e.status
                self._set_status(e, "disputed", f"conflict {c.id}: {reason}")
        return c

    def challenge(
        self, eid: str, objection: str, round_: int = 1
    ) -> tuple[list[str], Conflict | None]:
        """Object to an entry without a rival entry (for example, a verifier says it is wrong).

        A merely proposed entry is refuted at once. A corroborated, verified or disputed
        entry is too established for one objection: it opens a conflict the arbiter
        settles. Returns the tainted ids and the conflict opened, if any.
        """
        e = self.get(eid)
        if e is None or e.proposal or e.status in TAINTED:
            return [], None
        if e.status == "proposed":
            return self.refute(eid, objection), None
        for c in self.conflicts.values():
            if c.status == "open" and c.a == eid and not c.b:
                return [], c
        self._cseq += 1
        c = Conflict(
            id=f"C{self._cseq}",
            a=eid,
            b="",
            reason="challenged by a verifier",
            round=int(round_),
            challenge=_excerpt(objection, 600),
        )
        self.conflicts[c.id] = c
        if e.status != "disputed":
            e.held_status = e.status
            self._set_status(e, "disputed", f"conflict {c.id}: {_excerpt(objection, 200)}")
        return [], c

    def open_conflicts(self) -> list[Conflict]:
        return [c for c in self.conflicts.values() if c.status == "open"]

    def _restore(self, e: Entry) -> None:
        if e.status == "disputed" and not any(
            c.status == "open" and e.id in (c.a, c.b) for c in self.conflicts.values()
        ):
            self._set_status(e, e.held_status or "proposed", "dispute settled")

    def resolve(self, cid: str, verdict: str, note: str = "") -> list[str]:
        """Settle a conflict: ``a`` or ``b`` holds, ``neither`` holds, or ``both`` (no real clash)."""
        c = self.conflicts.get(cid)
        if c is None or c.status != "open":
            return []
        verdict = (verdict or "").strip().lower()
        if verdict not in {"a", "b", "neither", "both"}:
            verdict = "both"
        a = self.entries[c.a]
        b = self.entries.get(c.b) if c.b else None  # None: a challenge, not a rival entry
        c.verdict = verdict
        c.note = _excerpt(note, 400)
        c.status = "dismissed" if verdict == "both" else "resolved"
        changed: list[str] = []
        if verdict == "both":
            for e in (a, b):
                if e is not None:
                    self._restore(e)
            return changed
        sides: dict[str, tuple[list[Entry | None], list[Entry | None]]] = {
            "a": ([b], [a]),
            "b": ([a], [b]),
            "neither": ([a, b], []),
        }
        lose, win = sides[verdict]
        losers = [e for e in lose if e is not None]
        winners = [e for e in win if e is not None]
        for e in losers:
            changed += self.refute(e.id, f"lost arbitration {c.id}: {c.note or c.reason}")
        for e in winners:
            if e.status in {"disputed", "proposed", "corroborated"}:
                self._set_status(e, "verified", f"upheld by arbitration {c.id}")
        return changed

    # ── verification and refutation ─────────────────────────────────
    def confirm(self, eid: str, reason: str = "confirmed by a verifier") -> bool:
        e = self.get(eid)
        if e is None or e.proposal or e.status not in {"proposed", "corroborated"}:
            return False
        self._set_status(e, "verified", reason)
        return True

    def refute(self, eid: str, reason: str) -> list[str]:
        """Refute an entry and taint everything that rests only on refuted or suspect results."""
        e = self.get(eid)
        if e is None or e.status == "refuted":
            return []
        self._set_status(e, "refuted", reason)
        changed = [e.id] + self._propagate()
        for c in self.conflicts.values():
            if c.status == "open" and e.id in (c.a, c.b):
                c.status, c.verdict = "resolved", ("b" if c.a == e.id else "a")
                c.note = f"{e.id} refuted elsewhere"
                other = self.entries.get(c.b if c.a == e.id else c.a)
                if other is not None:
                    self._restore(other)
        return changed

    def _propagate(self) -> list[str]:
        changed: list[str] = []
        moved = True
        while moved:
            moved = False
            for e in self.entries.values():
                if e.status in TAINTED or not e.derivations:
                    continue
                if all(
                    deriv
                    and any(self.entries[d].status in TAINTED for d in deriv if d in self.entries)
                    for deriv in e.derivations
                ):
                    self._set_status(e, "suspect", "every derivation rests on a refuted result")
                    changed.append(e.id)
                    moved = True
        return changed

    def quality(self, eid: str | None, suspect_penalty: float = 0.5) -> float:
        """Vote multiplier for a proposal: 0 if refuted, ``suspect_penalty`` if suspect, else 1."""
        e = self.get(eid)
        if e is None:
            return 1.0
        if e.status == "refuted":
            return 0.0
        if e.status == "suspect":
            return max(0.0, min(1.0, suspect_penalty))
        return 1.0

    def closure(self, eid: str, limit: int = 64) -> list[str]:
        """Every entry ``eid`` depends on, nearest first."""
        out: list[str] = []
        frontier = list(self.entries[eid].depends_on) if eid in self.entries else []
        while frontier and len(out) < limit:
            nxt: list[str] = []
            for d in frontier:
                if d in out or d not in self.entries:
                    continue
                out.append(d)
                nxt.extend(self.entries[d].depends_on)
            frontier = nxt
        return out[:limit]

    # ── reading ─────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        *,
        member: int | None = None,
        k: int = 6,
        exclude: set[str] | None = None,
        fill: bool = True,
    ) -> list[tuple[Entry, float]]:
        """Entries most relevant to ``query`` (proposals never; refuted only when close).

        The ledger holds one problem's results, so with ``fill`` the lexical hits are
        topped up with the most established remaining entries (verified, then
        corroborated, by support): an agent always sees the core of the body of work.
        """
        if k <= 0:
            return []
        q = tokens(query)
        scored: list[tuple[Entry, float]] = []
        rest: list[Entry] = []
        for eid, e in self.entries.items():
            if e.proposal or (exclude and eid in exclude):
                continue
            if not self.shared and member is not None and int(member) not in e.members:
                continue
            t = self._tok.get(eid) or frozenset()
            inter = len(q & t) if q else 0
            if not inter:
                if e.status in {"verified", "corroborated"}:
                    rest.append(e)
                continue
            sim = inter / math.sqrt(len(q) * len(t))
            if e.status == "refuted" and sim < 0.3:
                continue
            score = sim + _STATUS_BOOST.get(e.status, 0.0) + 0.05 * min(e.support, 4)
            scored.append((e, score))
        scored.sort(key=lambda t: (-t[1], t[0].id))
        hits = scored[:k]
        if fill and len(hits) < k and rest:
            rank = {"verified": 0, "corroborated": 1}
            rest.sort(key=lambda e: (rank.get(e.status, 2), -e.support, e.id))
            hits += [(e, 0.0) for e in rest[: k - len(hits)]]
        return hits

    def context_block(self, hits: list[tuple[Entry, float]], *, excerpts: int = 2) -> str:
        if not hits:
            return ""
        lines = [
            "## Body of work (results the swarm has written so far)",
            "Cite entries by id. Verified and corroborated results need no new proof. Never build",
            "on refuted or suspect entries. If you find an entry is wrong, put its id in",
            '"contradicts"; if your cell asks the same thing as an entry, put its id in "same_as".',
            "",
        ]
        shown = 0
        for e, _score in hits:
            tag = f"{e.kind} · {e.status}"
            if e.support > 1:
                tag += f" · {e.support} QNNs"
            ans = f" → {e.answer}" if e.answer and e.canon not in _TRUE else ""
            lines.append(f"- {e.id} [{tag}] {_excerpt(e.statement, 320)}{ans}")
            if e.status in TAINTED and e.notes:
                lines.append(f"  ✗ {_excerpt(e.notes[-1], 200)}")
            elif shown < excerpts and e.result and e.status in USABLE:
                lines.append(f"  derivation: {_excerpt(e.result, 360)}")
                shown += 1
        return "\n".join(lines)

    def usable(
        self, eid: str | None, member: int | None = None, statement: str | None = None
    ) -> Entry | None:
        """An entry an agent may reuse: exists, not a proposal, usable status, visible to ``member``.

        With ``statement``, the entry must also be about the same thing (content-token
        overlap of at least 0.3), so a decomposer cannot pass off an unrelated result.
        """
        e = self.get(eid)
        if e is None or e.proposal or e.status not in USABLE:
            return None
        if not self.shared and member is not None and int(member) not in e.members:
            return None
        if statement is not None:
            want, have = tokens(statement), self._tok.get(e.id) or frozenset()
            if not want or not have:
                return None
            if len(want & have) / math.sqrt(len(want) * len(have)) < 0.3:
                return None
        return e

    # ── reporting ───────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        body = [e for e in self.entries.values() if not e.proposal]
        counts = {s: sum(1 for e in body if e.status == s) for s in STATUSES}
        return {
            "entries": len(body),
            "proposals": sum(1 for e in self.entries.values() if e.proposal),
            **counts,
            "conflicts_open": len(self.open_conflicts()),
            "conflicts_settled": sum(1 for c in self.conflicts.values() if c.status != "open"),
            "reuses": sum(len(e.reused_by) for e in body),
            "shared": self.shared,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared": self.shared,
            "stats": self.stats(),
            "entries": [e.to_dict() for e in self.entries.values()],
            "conflicts": [c.to_dict() for c in self.conflicts.values()],
            "log": list(self.log[-400:]),
        }

    def summary_markdown(self, limit: int = 24) -> str:
        body = [e for e in self.entries.values() if not e.proposal]
        order = {
            "verified": 0,
            "corroborated": 1,
            "proposed": 2,
            "disputed": 3,
            "suspect": 4,
            "refuted": 5,
        }
        body.sort(key=lambda e: (order.get(e.status, 9), -e.support, e.id))
        st = self.stats()
        lines = [
            f"{st['entries']} results: {st['verified']} verified, {st['corroborated']} corroborated, "
            f"{st['proposed']} proposed, {st['disputed']} disputed, {st['suspect']} suspect, "
            f"{st['refuted']} refuted. Conflicts settled: {st['conflicts_settled']}, "
            f"open: {st['conflicts_open']}. Reused: {st['reuses']}×."
        ]
        if body:
            lines.append("")
            lines.append("| id | kind | status | QNNs | result |")
            lines.append("|---|---|---|---|---|")
            for e in body[:limit]:
                ans = f" → {e.answer}" if e.answer and e.canon not in _TRUE else ""
                text = _excerpt(e.statement + ans, 160).replace("|", "\\|")
                lines.append(f"| {e.id} | {e.kind} | {e.status} | {e.support} | {text} |")
            if len(body) > limit:
                lines.append(f"\n…and {len(body) - limit} more in the JSON result.")
        return "\n".join(lines)
