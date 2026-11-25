"""Executable lattice semantics: instances are enumerated by *running* the
rewrite structure, never estimated from mutator-declared generative_power.

Semantics (see graph/v0 -> measured):
- expansion_map[rule] = [children]  explicit rewrite into those rules;
- expansion_map[rule] = []          explicit terminal macro (own ground space);
- rule absent from expansion_map    implicit downward pack into every rule of
  the immediately lower level; at the ground level (or when the lower level has
  no rules) it is terminal and contributes its own ground space.

Ground space per level is closed by the modulus:
- int modulus_spec.period -> that many distinct values (the quotient);
- int base_spec.radix     -> radix^2 two-digit window;
- otherwise               -> max(1, len(alphabet))^2.

Ground content is executable data: positional digit strings when a radix and
alphabet are present (binary/decimal/... evaluation), else chart slots.

Fitness is measured from enumerated instances + description cost, with a
holdout-motif coverage term. An agent cannot raise its own score by declaring
generative_power: the stock gate applies the edit and measures.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .state import Motif, ScaleLattice

MAX_INSTANCES = 10_000

_STOP = set(
    "the a an of to in on for with and or by as at from that this it is are was "
    "were be been being which who whom whose what where when how why not no yes "
    "if then than so such into over under between within without about per via".split()
)


def _levels(lattice: ScaleLattice) -> list[dict[str, Any]]:
    return list(lattice.get("levels", []) or [])


def _level_by_id(lattice: ScaleLattice) -> dict[str, dict[str, Any]]:
    return {str(lv.get("level_id")): lv for lv in _levels(lattice)}


def _level_below(lattice: ScaleLattice, level_id: str) -> dict[str, Any] | None:
    ids = [str(lv.get("level_id")) for lv in _levels(lattice)]
    if level_id not in ids:
        return None
    i = ids.index(level_id)
    return _levels(lattice)[i + 1] if i + 1 < len(ids) else None


def _rules_of(level: dict[str, Any] | None) -> list[dict[str, Any]]:
    return list(level.get("rules", []) or []) if level else []


def ground_fanout(level: dict[str, Any] | None) -> int:
    """Size of the ground instance space a terminal rule contributes."""
    if not level:
        return 1
    ms = level.get("modulus_spec") or {}
    period = ms.get("period")
    if isinstance(period, int) and period > 0:
        return min(MAX_INSTANCES, period)
    radix = (level.get("base_spec") or {}).get("radix")
    if isinstance(radix, int) and 1 < radix <= 1000:
        return min(4096, radix * radix)
    alpha = level.get("alphabet") or []
    return max(1, min(1024, len(alpha) * len(alpha)))


def _digits(value: int, radix: Any, alphabet: list[Any]) -> str | None:
    if not isinstance(radix, int) or radix < 2:
        return None
    table = "0123456789abcdefghijklmnopqrstuvwxyz"[:radix]
    if len(alphabet) >= radix:
        table = "".join(str(a)[:1] for a in alphabet[:radix])
    if value == 0:
        return table[0]
    out: list[str] = []
    v = value
    while v > 0:
        out.append(table[v % radix])
        v //= radix
    return "".join(reversed(out))


def enumerate_instances(
    lattice: ScaleLattice, max_instances: int = MAX_INSTANCES
) -> dict[str, Any]:
    """Execute the rewrite structure and return the distinct ground instances."""
    rules: dict[str, dict[str, Any]] = {}
    for lvl in _levels(lattice):
        for r in _rules_of(lvl):
            rid = str(r.get("id"))
            rules[rid] = {"rule": r, "level": lvl}
    expansion = lattice.get("expansion_map") or {}
    children_all = {str(c) for cs in expansion.values() for c in (cs or [])}
    roots = [rid for rid in rules if rid not in children_all]
    if not roots:
        roots = (
            [str(r.get("id")) for r in _rules_of(_levels(lattice)[0])] if _levels(lattice) else []
        )

    instances: list[dict[str, Any]] = []
    expanded: set = set()
    truncated = False

    def collect_ground(rid: str, lvl: dict[str, Any] | None) -> None:
        nonlocal truncated
        n = ground_fanout(lvl)
        radix = (lvl.get("base_spec") or {}).get("radix") if lvl else None
        alphabet = (lvl.get("alphabet") or []) if lvl else []
        lid = str(lvl.get("level_id")) if lvl else "?"
        for i in range(n):
            if len(instances) >= max_instances:
                truncated = True
                return
            digits = _digits(i, radix, alphabet)
            if digits is not None:
                content = {"radix": radix, "value": i, "digits": digits}
            else:
                symbol = str(alphabet[i % len(alphabet)]) if alphabet else f"slot{i}"
                content = {"slot": i, "symbol": symbol}
            instances.append(
                {"id": f"x-{rid}-{i:04d}", "rule": rid, "level": lid, "content": content}
            )

    def expand(rid: str) -> None:
        nonlocal truncated
        if rid in expanded or truncated:
            return
        expanded.add(rid)
        entry = rules.get(rid)
        lvl = entry["level"] if entry else None
        if rid in expansion:
            kids = [str(c) for c in (expansion.get(rid) or [])]
            if kids:
                for c in kids:
                    expand(c)
            else:  # explicit terminal macro: []; own ground space
                collect_ground(rid, lvl)
            return
        below = _level_below(lattice, str(lvl.get("level_id"))) if lvl else None
        below_rules = _rules_of(below)
        if below is None:
            collect_ground(rid, lvl)
        elif below_rules:
            for child in below_rules:
                expand(str(child.get("id")))
        # else: no lower rules -> dead end, contributes nothing

    for rid in roots:
        expand(str(rid))

    return {
        "instances": instances,
        "count": len(instances),
        "truncated": truncated,
        "roots": roots,
        "reached_rules": sorted(expanded),
    }


def description_cost(lattice: ScaleLattice) -> int:
    cost = 0
    for lvl in _levels(lattice):
        for r in _rules_of(lvl):
            cost += len(str(r.get("pattern", ""))) + len(str(r.get("rewrite", "")))
        cost += len(str(lvl.get("description", "")))
    return max(1, cost)


def measured_capacity(lattice: ScaleLattice) -> dict[str, float]:
    """Instantiation capacity from *executed* enumeration, not declarations."""
    enumerated = enumerate_instances(lattice)
    gen = math.log2(1 + enumerated["count"])
    cost = description_cost(lattice)
    score = gen * 10.0 / math.log2(10 + cost)
    return {
        "instances": enumerated["count"],
        "generative_power": round(gen, 4),
        "description_cost": float(cost),
        "capacity": score,
        "truncated": enumerated["truncated"],
    }


def _lattice_tokens(lattice: ScaleLattice) -> set:
    parts: list[str] = []
    for lvl in _levels(lattice):
        parts.extend(
            [
                str(lvl.get("name", "")),
                str(lvl.get("description", "")),
                str(lvl.get("base", "")),
                str(lvl.get("modulus", "")),
                " ".join(str(a) for a in (lvl.get("alphabet") or [])),
            ]
        )
        for r in _rules_of(lvl):
            parts.extend(
                [str(r.get("name", "")), str(r.get("pattern", "")), str(r.get("rewrite", ""))]
            )
    text = " ".join(parts).lower()
    return {t for t in re.findall(r"[a-z0-9]+", text) if len(t) >= 3 and t not in _STOP}


def holdout_coverage(lattice: ScaleLattice, motifs: list[Motif]) -> float:
    """Fraction of held-out motifs still generatable/covered by the lattice.

    A motif counts as covered when at least half of its (>=3 char) tokens
    appear in the lattice's vocabulary. Empty holdout -> 1.0 (vacuous).
    """
    if not motifs:
        return 1.0
    corpus = _lattice_tokens(lattice)
    covered = 0
    for m in motifs:
        toks = [
            t
            for t in re.findall(r"[a-z0-9]+", str(m.get("surface_form", "")).lower())
            if len(t) >= 3 and t not in _STOP
        ]
        if not toks:
            continue
        hit = sum(1 for t in toks if t in corpus)
        if hit * 2 >= len(toks):
            covered += 1
    return covered / len(motifs)
