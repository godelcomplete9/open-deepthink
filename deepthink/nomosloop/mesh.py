"""Mesh world: the graph of instantiations agents inhabit.

Each instantiation is a lesser replica of the law at some scale level.
Agents live at nodes, sense neighbourhoods (lesser = more concrete,
higher = more abstract), and propose elevations that get tested here.

Fitness (instantiation capacity) is *measured* by executing the rewrite
structure (see rewrite.py), never estimated from declared generative_power:
  generative_power = log2(1 + #distinct enumerated instances)
  description_cost = total symbols needed to state the lattice (compression term)
Higher is better: compression + generative power over mere accuracy.
"""

from __future__ import annotations

import hashlib
import itertools

from . import rewrite
from .state import InstantiationRecord, ScaleLattice


class MeshWorld:
    def __init__(self) -> None:
        self.nodes: dict[str, InstantiationRecord] = {}
        self.edges: list[tuple[str, str, str]] = []  # (src, dst, relation)

    # -- construction -----------------------------------------------------
    def spawn_from_lattice(
        self, lattice: ScaleLattice, motifs, law_hash: str, fanout: int = 3
    ) -> MeshWorld:
        """Materialise `fanout` instantiations per rule per level."""
        for lvl in lattice.get("levels", []):
            lid = lvl.get("level_id", "?")
            for rule in lvl.get("rules", []):
                rid = rule.get("id", "r")
                for k in range(fanout):
                    seed = f"{law_hash}:{rid}:{k}"
                    nid = f"inst-{hashlib.sha256(seed.encode()).hexdigest()[:8]}"
                    if nid in self.nodes:
                        continue
                    self.nodes[nid] = InstantiationRecord(
                        id=nid,
                        scale_level=lid,
                        content={
                            "rule": rid,
                            "variant": k,
                            "surface": f"{rule.get('pattern', '?')}#{k}",
                            "expansion": rule.get("rewrite", ""),
                        },
                        parent_law_hash=law_hash,
                        parent_rule=rid,
                        fitness=1.0 + 0.1 * k,
                        children_ids=[],
                    )
        self._link_hierarchy(lattice)
        return self

    def _link_hierarchy(self, lattice: ScaleLattice) -> None:
        levels = [lv.get("level_id") for lv in lattice.get("levels", [])]
        # order assumed top-down L3..L0; link each node to one node of the level below
        by_level: dict[str, list[str]] = {}
        for nid, n in self.nodes.items():
            by_level.setdefault(n.get("scale_level", "?"), []).append(nid)
        for i in range(len(levels) - 1):
            hi, lo = levels[i], levels[i + 1]
            his, los = by_level.get(hi, []), by_level.get(lo, [])
            if not his or not los:
                continue
            for h, low in itertools.zip_longest(his, los, fillvalue=None):
                hh = h or his[0]
                ll = low or los[0]
                self.edges.append((hh, ll, "expands-to"))
                if ll not in self.nodes[hh].get("children_ids", []):
                    self.nodes[hh].setdefault("children_ids", []).append(ll)

    # -- agent interface ---------------------------------------------------
    def neighborhood(self, inst_id: str, radius: int = 1) -> list[str]:
        adj: dict[str, list[str]] = {}
        for s, d, _ in self.edges:
            adj.setdefault(s, []).append(d)
            adj.setdefault(d, []).append(s)
        seen, frontier = {inst_id}, [inst_id]
        for _ in range(radius):
            nxt = []
            for f in frontier:
                for nb in adj.get(f, []):
                    if nb not in seen:
                        seen.add(nb)
                        nxt.append(nb)
            frontier = nxt
        seen.discard(inst_id)
        return sorted(seen)

    def upward(self, inst_id: str) -> list[str]:
        return [s for s, d, r in self.edges if d == inst_id and r == "expands-to"]

    def downward(self, inst_id: str) -> list[str]:
        return [d for s, d, r in self.edges if s == inst_id and r == "expands-to"]

    # -- fitness (measured, see rewrite.py) ---------------------------------
    def expansion_count(self, lattice: ScaleLattice) -> int:
        """Distinct instances actually reachable by executing the rewrites."""
        return rewrite.enumerate_instances(lattice)["count"]

    def description_cost(self, lattice: ScaleLattice) -> int:
        return rewrite.description_cost(lattice)

    def instantiation_capacity(self, lattice: ScaleLattice) -> dict[str, float]:
        m = rewrite.measured_capacity(lattice)
        return {
            "generative_power": m["generative_power"],
            "description_cost": m["description_cost"],
            "capacity": m["capacity"],
            "instances": m["instances"],
        }

    # -- (de)serialisation for LangGraph state ------------------------------
    def snapshot(self):
        return list(self.nodes.values()), [[s, d, r] for s, d, r in self.edges]

    @staticmethod
    def from_snapshot(nodes, edges) -> MeshWorld:
        w = MeshWorld()
        for n in nodes or []:
            w.nodes[n["id"]] = n
        for e in edges or []:
            w.edges.append((e[0], e[1], e[2] if len(e) > 2 else "expands-to"))
        return w
