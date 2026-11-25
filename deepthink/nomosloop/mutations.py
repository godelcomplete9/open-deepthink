"""Apply mutation proposals to a ScaleLattice. Pure: the input lattice is
never touched; a deep copy is returned.

Operations (graph/v0 + measured):
- add_rule        append new_rule at target level, declared terminal macro
                  (expansion_map[new_id] = []) so it grounds its own instances;
- refine_split    replace a rule with two copies (children inherited), so the
                  parent fan-out doubles where the split rule was terminal;
- elevate         promote a rule one level up (payload to/from honoured);
- merge_levels    merge the lower level's rules into the upper level;
- drop_rule       remove a rule and all expansion references to it;
- rewire_topology lattice-neutral here; the graph patch is applied (and
                  keep/rollback gated) in graph.rectify_node.
"""

from __future__ import annotations

import copy
from typing import Any

from .state import MutationProposal, ScaleLattice


def _levels(lattice: ScaleLattice) -> list[dict[str, Any]]:
    return lattice.setdefault("levels", [])


def _find_level(lattice: ScaleLattice, level_id: str) -> dict[str, Any] | None:
    for lvl in _levels(lattice):
        if str(lvl.get("level_id")) == str(level_id):
            return lvl
    return None


def _find_rule(
    lattice: ScaleLattice, rule_id: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for lvl in _levels(lattice):
        for r in lvl.get("rules", []) or []:
            if str(r.get("id")) == str(rule_id):
                return lvl, r
    return None, None


def _all_rule_ids(lattice: ScaleLattice) -> set:
    return {str(r.get("id")) for lvl in _levels(lattice) for r in (lvl.get("rules") or [])}


def _clean_refs(lattice: ScaleLattice) -> None:
    """Drop expansion references to rules that no longer exist.

    An entry whose non-empty child list becomes empty is removed entirely
    (implicit downward packing), so dangling parents do not silently become
    terminal macros.
    """
    expansion: dict[str, Any] = lattice.setdefault("expansion_map", {})
    alive = _all_rule_ids(lattice)
    for rid in list(expansion.keys()):
        kids = expansion.get(rid) or []
        kept = [str(c) for c in kids if str(c) in alive]
        if kids and not kept:
            del expansion[rid]
        elif kept != [str(c) for c in kids]:
            expansion[rid] = kept


def _add_rule(lattice: ScaleLattice, payload: dict[str, Any]) -> None:
    new_rule = dict(payload.get("new_rule") or {})
    if not new_rule:
        return
    target = _find_level(lattice, str(new_rule.get("level") or payload.get("target_level") or ""))
    if target is None:
        return
    rid = str(new_rule.get("id") or f"R-mut-{len(_all_rule_ids(lattice)):02d}")
    base_id, k = rid, 0
    while rid in _all_rule_ids(lattice):
        k += 1
        rid = f"{base_id}-{k}"
    new_rule["id"] = rid
    new_rule.setdefault("name", rid.upper())
    new_rule.setdefault("level", str(target.get("level_id")))
    new_rule.setdefault("pattern", str(target.get("level_id")))
    new_rule.setdefault("rewrite", "terminal macro grounding")
    target.setdefault("rules", []).append(new_rule)
    lattice.setdefault("expansion_map", {})[rid] = []  # explicit terminal macro


def _refine_split(lattice: ScaleLattice, payload: dict[str, Any]) -> None:
    rid = str(payload.get("split_rule") or "")
    lvl, rule = _find_rule(lattice, rid)
    if rule is None or lvl is None:
        return
    into = [str(x) for x in (payload.get("into") or [f"{rid}-a", f"{rid}-b"])][:2]
    expansion: dict[str, Any] = lattice.setdefault("expansion_map", {})
    children = expansion.get(rid)
    copies: list[dict[str, Any]] = []
    for nid in into:
        c = copy.deepcopy(rule)
        c["id"] = nid
        c["name"] = f"{rule.get('name', rid)}::{nid}"
        copies.append(c)
    lvl["rules"] = [r for r in lvl.get("rules", []) if str(r.get("id")) != rid]
    lvl.setdefault("rules", []).extend(copies)
    for parent, kids in list(expansion.items()):
        kids_s = [str(x) for x in kids]
        if rid in kids_s:
            replaced: list[str] = []
            for x in kids_s:
                if x == rid:
                    replaced.extend(into)
                else:
                    replaced.append(x)
            expansion[parent] = replaced
    if children is not None:
        for nid in into:
            expansion[nid] = list(children)
    elif rid in expansion:
        del expansion[rid]


def _elevate(lattice: ScaleLattice, payload: dict[str, Any]) -> None:
    rid = str(payload.get("promote") or "")
    lvl, rule = _find_rule(lattice, rid)
    if rule is None or lvl is None:
        return
    to = payload.get("to")
    target = _find_level(lattice, to) if to else None
    if target is None:
        ids = [str(lv.get("level_id")) for lv in _levels(lattice)]
        if str(lvl.get("level_id")) not in ids:
            return
        i = ids.index(str(lvl.get("level_id")))
        if i == 0:
            return
        target = _levels(lattice)[i - 1]
    lvl["rules"] = [r for r in lvl.get("rules", []) if str(r.get("id")) != rid]
    rule["level"] = str(target.get("level_id"))
    target.setdefault("rules", []).append(rule)


def _merge_levels(lattice: ScaleLattice, payload: dict[str, Any]) -> None:
    upper = _find_level(lattice, str(payload.get("upper") or ""))
    lower = _find_level(lattice, str(payload.get("lower") or ""))
    if upper is None or lower is None or upper is lower:
        return
    existing = {str(r.get("id")) for r in upper.get("rules", [])}
    for r in lower.get("rules", []) or []:
        if str(r.get("id")) not in existing:
            r["level"] = str(upper.get("level_id"))
            upper.setdefault("rules", []).append(r)
    merged_alpha = list(upper.get("alphabet") or [])
    for a in lower.get("alphabet") or []:
        if a not in merged_alpha:
            merged_alpha.append(a)
    upper["alphabet"] = merged_alpha
    _levels(lattice)[:] = [lv for lv in _levels(lattice) if lv is not lower]


def _drop_rule(lattice: ScaleLattice, payload: dict[str, Any]) -> None:
    rid = str(payload.get("drop") or "")
    lvl, rule = _find_rule(lattice, rid)
    if rule is None or lvl is None:
        return
    lvl["rules"] = [r for r in lvl.get("rules", []) if str(r.get("id")) != rid]
    lattice.get("expansion_map", {}).pop(rid, None)
    _clean_refs(lattice)


def apply_mutation(lattice: ScaleLattice, proposal: MutationProposal) -> ScaleLattice:
    """Return a new lattice with the proposal applied. Input is never mutated."""
    lat: ScaleLattice = copy.deepcopy(lattice)
    op = str(proposal.get("operation") or "")
    payload = dict(proposal.get("payload") or {})
    if op == "add_rule":
        _add_rule(lat, payload)
    elif op == "refine_split":
        _refine_split(lat, payload)
    elif op == "elevate":
        _elevate(lat, payload)
    elif op == "merge_levels":
        _merge_levels(lat, payload)
    elif op == "drop_rule":
        _drop_rule(lat, payload)
    elif op == "rewire_topology":
        pass  # graph-level patch, applied and gated in graph.rectify_node
    return lat
