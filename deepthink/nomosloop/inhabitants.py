"""Inhabitants: loop-carrying agents that instantiate peer and child agents.

Every mesh node gets one inhabitant. The inhabitant carries a minimal seed of
the RSI loop itself — law id, lattice version, its rule, and a spawn budget —
so it can instantiate:

- peers: same-level agents embodying a mutated variant of its rule
  (similar-level instantiation, e.g. a new Maxwell variant beside the old one);
- children: lower-level agents embodying the downward expansion of its rule
  (object instantiation, e.g. a concrete A_mu mode from POTENTIAL->FIELD).

Guards (non-negotiable):
1. Same-or-down only: an inhabitant may never spawn above its own level.
2. Budget: every spawn costs 1; at 0 the inhabitant is sterile.
3. Provenance: every spawn appends a spawn_log event and mesh edges, so the
   whole genealogy is inspectable from state alone.

Mechanics here are deterministic and cheap (no LLM calls). Novel content —
which mutation, which variant — arrives as arguments from the caller
(mutation_proposal node / future peer-to-peer negotiation), keeping the main
graph thin per the delegation constraint.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .mesh import MeshWorld
from .state import InhabitantRecord, ScaleLattice

PEER_RELATION = "spawned-peer"
CHILD_RELATION = "spawned-child"


def level_order(lattice: ScaleLattice) -> list[str]:
    """Top-down level ids, e.g. ['L3','L2','L1','L0']."""
    return [str(lv.get("level_id")) for lv in lattice.get("levels", [])]


def level_below(lattice: ScaleLattice, level: str) -> str | None:
    order = level_order(lattice)
    if level not in order or order.index(level) >= len(order) - 1:
        return None
    return order[order.index(level) + 1]


def populate(
    world: MeshWorld, lattice: ScaleLattice, law_id: str, budget: int = 2
) -> dict[str, InhabitantRecord]:
    """Attach one inhabitant per mesh node. Returns {inhabitant_id: record}."""
    version = str(lattice.get("version", "v0"))
    out: dict[str, InhabitantRecord] = {}
    for nid, node in world.nodes.items():
        iid = f"inh-{nid[5:]}" if nid.startswith("inst-") else f"inh-{nid}"
        out[iid] = InhabitantRecord(
            id=iid,
            node_id=nid,
            level=str(node.get("scale_level", "?")),
            rule_id=str(node.get("parent_rule", "?")),
            law_id=law_id,
            lattice_version=version,
            generation=0,
            budget=int(budget),
            offspring=[],
        )
    return out


def _register_node(
    world: MeshWorld,
    level: str,
    rule_id: str,
    law_hash: str,
    surface: str,
    expansion: str,
    fitness: float,
) -> str:
    seed = f"{law_hash}:{rule_id}:{surface}"
    nid = f"inst-{hashlib.sha256(seed.encode()).hexdigest()[:8]}"
    k = 0
    while nid in world.nodes:  # variant collision -> rehash with salt
        k += 1
        nid = f"inst-{hashlib.sha256((seed + str(k)).encode()).hexdigest()[:8]}"
    world.nodes[nid] = {
        "id": nid,
        "scale_level": level,
        "content": {"rule": rule_id, "surface": surface, "expansion": expansion},
        "parent_law_hash": law_hash,
        "parent_rule": rule_id,
        "fitness": fitness,
        "children_ids": [],
    }
    return nid


def spawn_peer(
    inhabitants: dict[str, InhabitantRecord],
    world: MeshWorld,
    lattice: ScaleLattice,
    parent_id: str,
    operation: str,
    law_hash: str,
    variant_note: str = "",
) -> tuple[InhabitantRecord, str]:
    """Same-level instantiation. Raises ValueError if budget is exhausted."""
    parent = inhabitants[parent_id]
    if int(parent.get("budget", 0)) <= 0:
        raise ValueError(f"{parent_id} has no spawn budget left")
    rule_id = f"{parent.get('rule_id')}-peer{len(parent.get('offspring', []))}"
    surface = f"{parent.get('rule_id')}⊕{operation}{(':' + variant_note) if variant_note else ''}"
    nid = _register_node(
        world,
        str(parent.get("level")),
        rule_id,
        law_hash,
        surface=surface,
        expansion=f"peer of {parent.get('node_id')}",
        fitness=1.0,
    )
    world.edges.append((str(parent.get("node_id")), nid, PEER_RELATION))
    parent["budget"] = int(parent.get("budget", 0)) - 1
    child = InhabitantRecord(
        id=f"{parent_id}p{len(parent.get('offspring', []))}",
        node_id=nid,
        level=str(parent.get("level")),
        rule_id=rule_id,
        law_id=str(parent.get("law_id", "")),
        lattice_version=str(lattice.get("version", "v0")),
        generation=int(parent.get("generation", 0)) + 1,
        budget=int(parent.get("budget", 0)),  # heirs inherit remaining budget
        offspring=[],
    )
    parent.setdefault("offspring", []).append(child["id"])
    inhabitants[child["id"]] = child
    return child, nid


def spawn_child(
    inhabitants: dict[str, InhabitantRecord],
    world: MeshWorld,
    lattice: ScaleLattice,
    parent_id: str,
    law_hash: str,
    variant: int = 0,
) -> tuple[InhabitantRecord, str]:
    """Lower-level instantiation via the lattice expansion map.

    Raises ValueError on budget exhaustion or when the parent sits at the
    ground base (no level below — nothing to instantiate downward).
    """
    parent = inhabitants[parent_id]
    if int(parent.get("budget", 0)) <= 0:
        raise ValueError(f"{parent_id} has no spawn budget left")
    below = level_below(lattice, str(parent.get("level")))
    if below is None:
        raise ValueError(f"{parent_id} at ground base {parent.get('level')}: no lower level")
    rule_id = f"{parent.get('rule_id')}-c{variant}"
    surface = f"{parent.get('rule_id')}#{variant}"
    nid = _register_node(
        world,
        below,
        rule_id,
        law_hash,
        surface=surface,
        expansion=f"downward instance of {parent.get('rule_id')}",
        fitness=1.0 + 0.1 * variant,
    )
    world.edges.append((str(parent.get("node_id")), nid, CHILD_RELATION))
    world.edges.append((str(parent.get("node_id")), nid, "expands-to"))
    pnode = world.nodes.get(str(parent.get("node_id")))
    if pnode is not None and nid not in pnode.get("children_ids", []):
        pnode.setdefault("children_ids", []).append(nid)
    parent["budget"] = int(parent.get("budget", 0)) - 1
    child = InhabitantRecord(
        id=f"{parent_id}c{len(parent.get('offspring', []))}",
        node_id=nid,
        level=below,
        rule_id=rule_id,
        law_id=str(parent.get("law_id", "")),
        lattice_version=str(lattice.get("version", "v0")),
        generation=int(parent.get("generation", 0)) + 1,
        budget=0,  # newborn objects act before they replicate; budget granted on elevation
        offspring=[],
    )
    parent.setdefault("offspring", []).append(child["id"])
    inhabitants[child["id"]] = child
    return child, nid


def log_event(parent_id: str, child_id: str, kind: str, level: str) -> dict[str, Any]:
    return {"parent": parent_id, "child": child_id, "kind": kind, "level": level}


def records(inhabitants: dict[str, InhabitantRecord]) -> list[InhabitantRecord]:
    return list(inhabitants.values())
