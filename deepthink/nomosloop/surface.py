"""Code and social-engineering artifacts beside the policy: facts, links, register.

The written policy is not the only artifact an institution runs on. Code and
configuration are what actually executes, and procedures run through people.
This module carries both into the RSI loop as *facts*:

* a :class:`Fact` is one observation, such as a hard-coded credential at a
  file and line, or a help-desk reset that needs no strong verification
  (``repo.py`` and ``social.py`` produce them);
* :func:`link` reads each fact against the policy clauses on the same topic.
  A fact **contradicts** a clause that mandates the opposite, is
  **permitted** by a clause that allows it, and is **uncovered** when the
  policy says nothing about it;
* :func:`ax_nodes` turns facts into artifact nodes ``AX:code/…`` and
  ``AX:soc/…``. Their phased edges (π contradicts, 0 permitted, π/2
  uncovered) let the lattice see a code or procedure discrepancy the same way
  it sees two clauses that clash;
* :func:`register` is the deterministic discrepancy register in the report.

Facts are leads for a reviewer. Detection is pattern based and reads files and
role descriptions only: nothing is executed, nothing is probed on a network,
and nobody is contacted or tested.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from . import correspondence as tc
from .mechanics import SPECIES

KINDS = ("code", "social")
MIN_SCORE = 2  # topic score a clause needs to count as "on the same topic"
_MANDATE = (" must ", " shall ", " required", " every ", " only ", " are logged", " is logged")


@dataclass
class Fact:
    """One observation from a code or social-engineering artifact."""

    ref: str  # code/C3, soc/S2
    kind: str  # code | social
    rule: str  # detector id
    title: str
    subject: str  # where: a file path, a role
    statement: str  # one clause-like English sentence
    species: str  # the security function it touches (Agent Mechanics species)
    severity: float  # 0..1
    remediation: str
    topics: list[str] = field(default_factory=list)
    location: str = ""
    evidence: str = ""  # redacted where it could hold a secret
    count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Fact:
        keys = Fact.__dataclass_fields__
        return Fact(**{k: v for k, v in d.items() if k in keys})

    @property
    def text(self) -> str:
        """Artifact-node text, in the same 'name: subject => body' shape as a clause."""
        return f"{self.title}: {self.subject} => {self.statement}"


def _facts(items: list[Any]) -> list[Fact]:
    return [x if isinstance(x, Fact) else Fact.from_dict(x) for x in items or []]


def topic_score(topics: list[str], text: str) -> int:
    """Phrases (with a space) weigh 2, single stems 1."""
    low = " " + text.lower() + " "
    return sum((2 if " " in t else 1) for t in dict.fromkeys(topics) if t and t in low)


def mandates(text: str) -> bool:
    """A clause that requires or forbids (restrictive polarity, or 'must'/'every' with none)."""
    pol = tc.polarity(text)
    if pol < 0:
        return True
    low = " " + text.lower() + " "
    return pol == 0 and any(w in low for w in _MANDATE)


def _clauses(lattice: dict[str, Any]) -> list[tuple[str, str]]:
    out = []
    for lv in lattice.get("levels", []) or []:
        lid = str(lv.get("level_id"))
        if lid == "L0":  # the asset inventory names everything; it is not a rule
            continue
        for r in lv.get("rules", []) or []:
            out.append((f"pol/{lid}/{r.get('id')}", str(r.get("rewrite", ""))))
    return out


def link(items: list[Any], lattice: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each fact against the policy clauses on its topic."""
    clauses = _clauses(lattice)
    out: dict[str, dict[str, Any]] = {}
    for f in _facts(items):
        contra: list[dict[str, Any]] = []
        allowed: list[dict[str, Any]] = []
        for ref, text in clauses:
            s = topic_score(f.topics, text)
            if s < MIN_SCORE:
                continue
            row = {"clause": ref, "text": text[:200], "score": s}
            (contra if mandates(text) else allowed).append(row)
        best = max([r["score"] for r in contra + allowed], default=0)
        # keep clauses at least half as on-topic as the best one (drops incidental matches)
        contra = sorted(
            (r for r in contra if 2 * r["score"] >= best), key=lambda r: (-r["score"], r["clause"])
        )
        allowed = sorted(
            (r for r in allowed if 2 * r["score"] >= best), key=lambda r: (-r["score"], r["clause"])
        )
        status = "contradicts" if contra else "permitted" if allowed else "uncovered"
        out[f.ref] = {"status": status, "contradicts": contra[:3], "permitted": allowed[:2]}
    return out


def _modulus_ref(lattice: dict[str, Any]) -> str | None:
    ids = [str(lv.get("level_id")) for lv in lattice.get("levels", []) or []]
    for lid in ("L1", "L2", *ids):
        if lid in ids:
            return f"pol/{lid}#mod"
    return None


def ax_nodes(
    items: list[Any], links: dict[str, dict[str, Any]], lattice: dict[str, Any]
) -> tuple[list[tc.AXNode], list[tuple[str, str, float]]]:
    """Artifact nodes for the facts, with phased edges to the clauses they touch."""
    nodes: list[tc.AXNode] = []
    edges: list[tuple[str, str, float]] = []
    mod = _modulus_ref(lattice)
    for f in _facts(items):
        nodes.append(
            tc.AXNode(
                f.ref,
                f.kind,
                f.text,
                {"species": f.species, "rule": f.rule, "location": f.location, "surface": f.kind},
            )
        )
        lk = links.get(f.ref) or {}
        for row in lk.get("contradicts", [])[:2]:
            edges.append((f.ref, row["clause"], math.pi))
        for row in lk.get("permitted", [])[:1]:
            edges.append((f.ref, row["clause"], 0.0))
        if lk.get("status", "uncovered") == "uncovered" and mod:
            edges.append((f.ref, mod, math.pi / 2))  # the policy does not close it
    return nodes, edges


def register(items: list[Any], links: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The discrepancy register: every fact with its status against the policy."""
    rank = {"contradicts": 0, "uncovered": 1, "permitted": 2}
    rows = []
    for f in _facts(items):
        lk = links.get(f.ref) or {"status": "uncovered", "contradicts": [], "permitted": []}
        rows.append(
            {
                "ref": f.ref,
                "kind": f.kind,
                "rule": f.rule,
                "title": f.title,
                "location": f.location,
                "status": lk["status"],
                "contradicts": [r["clause"] for r in lk.get("contradicts", [])],
                "permitted_by": [r["clause"] for r in lk.get("permitted", [])],
                "statement": f.statement,
                "remediation": f.remediation,
                "severity": round(f.severity, 3),
                "species": f.species,
                "evidence": f.evidence,
                "count": f.count,
            }
        )
    rows.sort(key=lambda r: (rank[r["status"]], -r["severity"], r["ref"]))
    return rows


STATUS_TEXT = {
    "contradicts": "contradicts the policy",
    "uncovered": "not covered by the policy",
    "permitted": "permitted by the policy",
}


def report_lines(rows: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    """Markdown sections for the hardened-policy report."""
    out: list[str] = []
    for kind, heading in (
        ("code", "Code and configuration (repository analysis)"),
        ("social", "Social engineering (institutional procedures)"),
    ):
        part = [r for r in rows if r["kind"] == kind]
        info = meta.get(kind) or {}
        if not part and not info:
            continue
        out += [f"## {heading}", ""]
        if kind == "code" and info:
            out.append(
                f"Scanned {info.get('files_scanned', 0)} files under `{info.get('root', '')}` "
                f"({info.get('files_skipped', 0)} skipped){' · file cap reached' if info.get('truncated') else ''}. "
                "Secrets are redacted; nothing was executed."
            )
            out.append("")
        if kind == "social" and info:
            out.append(
                f"{info.get('roles', 0)} roles, assessed at role level. "
                f"Authorized by {info.get('authorized_by', '?')} ({info.get('scope', '')})."
            )
            if info.get("dropped"):
                out.append(
                    "Not admitted (personal or psychometric data): "
                    + ", ".join(f"`{d}`" for d in info["dropped"][:20])
                    + "."
                )
            out.append("")
        if not part:
            out += ["No discrepancies found.", ""]
            continue
        for r in part:
            where = ", ".join(f"`{c}`" for c in r["contradicts"] or r["permitted_by"])
            status = STATUS_TEXT[r["status"]] + (f" ({where})" if where else "")
            many = f" ×{r['count']}" if r.get("count", 1) > 1 else ""
            out.append(
                f"- **{r['ref']}** {r['title']}{many} · {status} · severity {r['severity']:.2f} · `{r['location']}`"
            )
            out.append(f"  {r['statement']} Remediation: {r['remediation']}")
            if r.get("evidence"):
                out.append(f"  Evidence: `{r['evidence']}`")
        out.append("")
    return out


def valid_species(sp: str) -> str:
    return sp if sp in SPECIES else "Ti"


__all__ = [
    "KINDS",
    "Fact",
    "ax_nodes",
    "link",
    "mandates",
    "register",
    "report_lines",
    "topic_score",
]
