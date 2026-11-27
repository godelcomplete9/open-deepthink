"""The three-domain world of the RSI loop.

* **AM** (lattice): Agent Mechanics agents, attention edges with gauge phases,
  ticks of the tick law, typed edges, circuits, ranks, ensembles.
* **TC** (correspondence): one row per artifact element, with its species
  projection and the gauge link ``A`` to its agent.
* **AX** (artifact): the domain of origin, an institution's security policy
  (levels, clauses, moduli, viable control operations) and its phased
  relations; contradictory clauses carry phase π.

The loop is driven by **Leak and stock**. A leaking agent heals in the
lattice with a licensed operator, and the heal is pulled back to an artifact
inspection through the symbolic coderivative. A stocking agent spends its
surplus evolving the policy (a proposed control clause, kept only if the
measured posture improves).
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

from . import correspondence as tc
from . import mechanics as am
from .lora import DistilledMemory
from .mechanics import SIDES, SPECIES, Agent, Params

# Drive presets (NX-1 §8). Values are conventions; every one is overridable.
DRIVES: dict[str, dict[str, float]] = {
    "leak": {
        "s_min": 0.62,
        "zeta": 0.3,
        "heal_budget": 12,
        "stock_budget": 2,
        "theta_p": 0.2,
        "q": 0.03,
    },
    "stock": {
        "s_min": 0.38,
        "zeta": 0.08,
        "heal_budget": 3,
        "stock_budget": 8,
        "theta_p": 0.45,
        "q": 0.01,
    },
    "bal": {
        "s_min": 0.5,
        "zeta": 0.15,
        "heal_budget": 6,
        "stock_budget": 4,
        "theta_p": 0.3,
        "q": 0.02,
    },
}
BUDGET_KEYS = ("heal_budget", "stock_budget")
OP_COST = {"a": 0.0, "r": 0.0, "ra": 0.0, "sigma": 0.01, "arrow": 0.02, "cut": 0.03}


def drive_config(
    drive: str, overrides: dict[str, Any] | None = None
) -> tuple[Params, dict[str, int]]:
    """Params and budgets for a drive, with per-key overrides (``sensitivity``)."""
    preset = dict(DRIVES.get(drive, DRIVES["bal"]))
    preset.update({k: v for k, v in (overrides or {}).items() if v is not None})
    p = Params()
    for k, v in preset.items():
        if k in BUDGET_KEYS:
            continue
        if hasattr(p, k):
            setattr(p, k, float(v))
    budgets = {k: int(preset.get(k, DRIVES["bal"][k])) for k in BUDGET_KEYS}
    return p, budgets


@dataclass
class World:
    params: Params
    drive: str = "bal"
    lambda_am: float = 0.7
    mass: float = 3.0
    run_id: str = "run"
    agents: dict[str, Agent] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    rows: list[tc.TocRow] = field(default_factory=list)
    ax: dict[str, tc.AXNode] = field(default_factory=dict)
    ax_edges: list[tuple[str, str, float]] = field(default_factory=list)
    am_edges: dict[tuple[str, str], float] = field(default_factory=dict)
    pr_edges: list[tuple[str, str]] = field(default_factory=list)
    cut: set[tuple[str, str]] = field(default_factory=set)
    stems: set[tuple[str, str]] = field(default_factory=set)
    pending_parent: dict[str, str] = field(default_factory=dict)
    typed: list[tuple[str, str, str, str]] = field(default_factory=list)
    cycles: dict[tuple[str, ...], str] = field(default_factory=dict)
    ensembles: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    inspections: list[dict[str, Any]] = field(default_factory=list)
    intake: dict[str, list[float]] = field(default_factory=dict)
    prev_L: dict[str, float] = field(default_factory=dict)
    clash_cache: dict[str, float] = field(default_factory=dict)
    t: int = 0
    tombstones: int = 0
    memory: DistilledMemory = field(default_factory=DistilledMemory)
    coverage_history: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    solves: int = 0

    def __post_init__(self) -> None:
        self.R = am.channel_matrix(self.params)

    # ------------------------------------------------------------ helpers
    def new_id(self, cls: str) -> str:
        self.counters[cls] = self.counters.get(cls, 0) + 1
        return f"{cls}{self.counters[cls]}"

    def live(self) -> dict[str, Agent]:
        return {k: a for k, a in self.agents.items() if a.alive}

    def rows_by_ax(self) -> dict[str, list[tc.TocRow]]:
        out: dict[str, list[tc.TocRow]] = {}
        for r in self.rows:
            out.setdefault(r.ax_ref, []).append(r)
        return out

    def rows_of(self, agent_id: str) -> list[tc.TocRow]:
        return [r for r in self.rows if r.agent_id == agent_id and r.alive]

    def event(self, kind: str, agent: str, text: str, **extra: Any) -> dict[str, Any]:
        ev = {
            "t": self.t,
            "kind": kind,
            "agent": agent,
            "text": f"t={self.t} AM:{agent} {text}",
            **extra,
        }
        self.events.append(ev)
        return ev

    def fork(self) -> World:
        """An independent copy for an alternate universe; its logs start empty."""
        light = {"events": [], "inspections": [], "coverage_history": []}
        saved = {k: getattr(self, k) for k in light}
        try:
            for k, v in light.items():
                setattr(self, k, v)
            twin = copy.deepcopy(self)
        finally:
            for k, v in saved.items():
                setattr(self, k, v)
        return twin

    def reproject(self, node: tc.AXNode, mapper: str = "offline-lexicon") -> None:
        """Re-map an element whose text changed. The agent and its id persist (ev_i)."""
        self.ax[node.ref] = node
        data = tc.offline_row(node)
        for r in self.rows:
            if not r.alive or r.ax_ref != node.ref:
                continue
            r.p = list(data["p"])
            r.evidence = dict(data.get("evidence") or {})
            r.basis = list(data.get("basis") or [])
            r.conf = float(data.get("conf", 0.5))
            r.mapper = mapper
            ag = self.agents.get(r.agent_id)
            if ag is not None:
                r.A = tc.link_phase(r.p, ag.w_eff())

    # ------------------------------------------------------------ root
    def ensure_root(self, law_text: str) -> Agent:
        if "I1" in self.agents:
            return self.agents["I1"]
        p, _ = tc.project_text(law_text[:4000])
        aid = self.new_id("I")
        root = Agent(
            id=aid,
            cls="I",
            stack=(),
            w=[v * self.mass * 1.5 for v in p],
            end=1.0,
            aut=0,
            theta_p=self.params.theta_p,
            lam=1.0,
        )
        self.agents[aid] = root
        self.event("birth", aid, "born as the operator of the artifact (institution)")
        return root

    # ------------------------------------------------------------ sync
    def sync(
        self,
        nodes: list[tc.AXNode],
        edges: list[tuple[str, str, float]],
        mapped: dict[str, dict[str, Any]],
        mapper_name: str,
    ) -> dict[str, int]:
        """Bring TC and AM in line with the current artifact graph (NX-1 X1)."""
        current = {n.ref: n for n in nodes}
        born = 0
        dead = 0
        # tombstone rows/agents whose artifact element vanished
        for r in self.rows:
            if r.alive and r.ax_ref not in current:
                r.alive = False
                ag = self.agents.get(r.agent_id)
                if ag and ag.alive and not self.rows_of(ag.id):
                    ag.alive = False
                    self.tombstones += 1
                    dead += 1
                    self.event("tomb", ag.id, "!x  (artifact element removed)")
        known = {r.ax_ref for r in self.rows if r.alive}
        root = self.agents.get("I1")
        for ref, node in current.items():
            if ref in known:
                continue
            data = mapped.get(ref) or tc.offline_row(node)
            mapper = mapper_name if ref in mapped else "offline-lexicon"
            parent = self.agents.get(self.pending_parent.pop(ref, ""))
            operator = parent.id if parent else (root.id if root else "")
            self._instantiate(node, data, mapper, operator=operator, parent=parent)
            born += 1
        self.ax = current
        self.ax_edges = [e for e in edges if e[0] in current and e[1] in current]
        self._rebuild_edges()
        return {"born": born, "tombstoned": dead}

    def _instantiate(
        self,
        node: tc.AXNode,
        data: dict[str, Any],
        mapper: str,
        operator: str = "",
        parent: Agent | None = None,
    ) -> Agent:
        p = list(data["p"])
        order = sorted(range(8), key=lambda i: (-p[i], i))
        stack = tuple(SPECIES[i] for i in order if p[i] > 0.12)[:3] or (SPECIES[order[0]],)
        cls = data.get("spectrum", "A") or "A"
        if cls not in "BIEAMTQ":
            cls = "A"
        aid = self.new_id(cls)
        end = 0.0 if node.kind == "operation" else 0.25
        agent = Agent(
            id=aid,
            cls=cls,
            stack=stack,
            w=[max(0.0, v) * self.mass for v in p],
            end=end,
            aut=0,
            theta_p=self.params.theta_p,
            lam=0.6,
            op=str(node.attrs.get("op", "")),
            role=str(node.attrs.get("role", "")),
            operator=operator,
            born=self.t,
        )
        if node.kind == "operation":
            agent.energy["e"] = [0.6] * 4  # a prosthesis starts sub-active
        self.agents[aid] = agent
        row = tc.TocRow(
            n=len(self.rows) + 1,
            ax_ref=node.ref,
            agent_id=aid,
            basis=list(data.get("basis") or []),
            p=p,
            evidence=dict(data.get("evidence") or {}),
            spectrum=cls,
            why=str(data.get("why", "")),
            chain=f"{operator} -pr-> {aid}" if operator else "—",
            A=tc.link_phase(p, agent.w_eff()),
            conf=float(data.get("conf", 0.5)),
            mapper=mapper,
        )
        self.rows.append(row)
        agent.rows.append(row.n)
        if operator:
            self.pr_edges.append((operator, aid))
        if parent is not None:
            self.event("pr", parent.id, f"pr -> AM:{aid}", child=aid)
        return agent

    def _rebuild_edges(self) -> None:
        rba = self.rows_by_ax()
        edges: dict[tuple[str, str], float] = {}
        for a1, a2, A in self.ax_edges:
            for r1 in rba.get(a1, []):
                for r2 in rba.get(a2, []):
                    if not (r1.alive and r2.alive) or r1.agent_id == r2.agent_id:
                        continue
                    i, j = r1.agent_id, r2.agent_id
                    if not (self.agents[i].alive and self.agents[j].alive):
                        continue
                    for s, d in ((i, j), (j, i)):
                        edges[(s, d)] = max(edges.get((s, d), 0.0), A)
        for s, d in self.pr_edges:
            if (
                s in self.agents
                and d in self.agents
                and self.agents[s].alive
                and self.agents[d].alive
            ):
                edges.setdefault((s, d), 0.0)
                edges.setdefault((d, s), 0.0)
        for s, d in self.stems:  # stems opened by heal(sigma) persist
            if (
                s in self.agents
                and d in self.agents
                and self.agents[s].alive
                and self.agents[d].alive
            ):
                edges.setdefault((s, d), 0.0)
        for k in self.cut:
            edges.pop(k, None)
        self.am_edges = edges

    # ------------------------------------------------------------ ticks
    def run_ticks(self, n: int) -> None:
        p = self.params
        for _ in range(max(1, n)):
            self.t += 1
            for r in self.rows:
                if r.alive and r.agent_id in self.agents:
                    r.A = tc.link_phase(r.p, self.agents[r.agent_id].w_eff())
            for k, a in self.live().items():
                self.prev_L[k] = a.L
            report = am.tick(self.agents, self.am_edges, self.R, p, intake=self.intake, t=self.t)
            self.intake = {}
            self._type_and_name(report)
        self._classify()
        self._coverage()

    def _type_and_name(self, report: am.TickReport) -> None:
        p = self.params
        live = self.live()
        typed: dict[tuple[str, str], tuple[str, str, bool]] = {}
        for (s, d), phi in report.fluxes.items():
            typed[(s, d)] = am.type_edge(live[s], live[d], phi, report.contrib[(s, d)], p)
        out: list[tuple[str, str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for (s, d), (typ, sp, stores) in sorted(typed.items()):
            back = typed.get((d, s))
            if back and stores and back[2] and typ in ("st", "fl") and back[0] in ("st", "fl"):
                key = (min(s, d), max(s, d))
                if key in seen:
                    continue
                seen.add(key)
                out.append((key[0], key[1], "du", sp))
                continue
            out.append((s, d, typ, sp))
        pr = set(self.pr_edges)
        out += [(s, d, "pr", "") for s, d in pr if s in live and d in live]
        self.typed = out
        adj: dict[str, list[str]] = {}
        stores_of: dict[tuple[str, str], bool] = {}
        for s, d, typ, _sp in out:
            if typ in ("op", "pr"):
                continue
            pairs = [(s, d), (d, s)] if typ == "du" else [(s, d)]
            for a, b in pairs:
                adj.setdefault(a, []).append(b)
                stores_of[(a, b)] = typ in am.STORING_EDGES
        cycles = am.find_cycles(adj, max_len=5, limit=80)
        sigs: dict[tuple[str, ...], str] = {}
        member_cycle: dict[str, tuple[int, str]] = {}
        for cyc in cycles:
            st = [stores_of.get((cyc[i], cyc[(i + 1) % len(cyc)]), False) for i in range(len(cyc))]
            sig = am.signature(st)
            sigs[cyc] = sig
            prev = self.cycles.get(cyc)
            if prev == "store" and sig != "store":
                self.event("brk", cyc[0], f"Brk(Cyc{len(cyc)}/store->{sig})", cycle=list(cyc))
            if len(cyc) < 3:
                continue  # a 2-cycle is a dual (stem material, Vor), not a ring
            for m in cyc:
                best = member_cycle.get(m)
                rank_sig = {"store": 2, "mix": 1, "pol": 0}[sig]
                if best is None or rank_sig > {"store": 2, "mix": 1, "pol": 0}[best[1]]:
                    member_cycle[m] = (len(cyc), sig)
        self.cycles = sigs
        out_types: dict[str, set[str]] = {}
        for s, _d, typ, _sp in out:
            out_types.setdefault(s, set()).add(typ)
        for k, a in live.items():
            in_store = member_cycle.get(k, (0, ""))[1] == "store"
            op_rank = self.agents[a.operator].rank if a.operator in self.agents else None
            a.rank = am.next_rank(
                a, p, in_store_cycle=in_store, operator_rank=op_rank, prev_L=self.prev_L.get(k, 1.0)
            )
            n_c, sig = member_cycle.get(k, (0, ""))
            if a.rank == "Attr":
                a.regime = "hp"
            elif sig in ("store", "pol", "mix") and n_c:
                a.regime = (
                    ("store4" if sig == "store" else "pol4")
                    if n_c == 4 and sig != "mix"
                    else f"Cyc{n_c}/{sig}"
                )
            elif a.L >= 0.5 and sum(a.leak_op) >= max(sum(a.leak_ov), sum(a.leak_q)):
                a.regime = "leak"
            elif "st" in out_types.get(k, set()) or "du" in out_types.get(k, set()):
                a.regime = "stem"
            elif "or" in out_types.get(k, set()):
                a.regime = "orb"
            else:
                a.regime = "cond"
        self._ensembles(live)

    def _ensembles(self, live: dict[str, Agent]) -> None:
        parent = {k: k for k in live}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for s, d, typ, _sp in self.typed:
            if typ in ("st", "du") and s in parent and d in parent:
                ra, rb = find(s), find(d)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
        groups: dict[str, list[Agent]] = {}
        for k in live:
            groups.setdefault(find(k), []).append(live[k])
        ens = []
        idx = 0
        for _root, members in sorted(groups.items(), key=lambda t: (-len(t[1]), t[0])):
            ids = {a.id for a in members}
            rings = []
            for cyc, sig in self.cycles.items():
                if set(cyc) <= ids:
                    n = len(cyc)
                    name = (
                        f"Cyc4/{sig}4({','.join(cyc)})"
                        if n == 4 and sig != "mix"
                        else f"Cyc{n}/{sig}({','.join(cyc)})"
                    )
                    if n == 2 and sig == "store":
                        name = f"Vor({cyc[0]},{cyc[1]})"
                    rings.append(name)
            if len(members) < 2:
                a = members[0]
                a.climate = am.climate_of(a.w, False)
                continue
            idx += 1
            title = self._title(members, idx)
            e = am.ensemble_string(
                title, members, 0, self.typed, rings[:6], self.params, t=self.t, drive=self.drive
            )
            for a in members:
                a.climate = e["climate"]
            e["id"] = f"G{idx}"
            ens.append(e)
        self.ensembles = ens

    def _title(self, members: list[Agent], idx: int) -> str:
        refs = [r.ax_ref for a in members for r in self.rows_of(a.id)]
        heads = {}
        for ref in refs:
            parts = ref.split("/")
            key = "/".join(parts[:2]) if len(parts) > 1 else parts[0]
            heads[key] = heads.get(key, 0) + 1
        if heads:
            top = max(heads.items(), key=lambda t: t[1])[0]
            return (
                "Ens-"
                + "".join(ch if ch.isalnum() else "-" for ch in top).strip("-")[:28]
                + f"-{idx}"
            )
        return f"Ens-{idx}"

    # ------------------------------------------------------------ gauge / clash
    def am_phase(self, i: str, j: str) -> float | None:
        if (i, j) not in self.am_edges and (j, i) not in self.am_edges:
            return None
        a, b = self.agents.get(i), self.agents.get(j)
        if not a or not b:
            return None
        return 0.0 if am.encounter_mass(a, b, self.R) >= 0 else math.pi

    def faces(self) -> list[tc.Face]:
        return tc.faces(self.ax_edges, self.rows_by_ax(), self.am_phase)

    def clashes(self, threshold: float = 0.5) -> list[tc.Face]:
        fs = [f for f in self.faces() if f.f > threshold]
        fs.sort(key=lambda f: (-f.weight, -f.f, f.a1, f.a2))
        return fs

    # ------------------------------------------------------------ classify
    def _classify(self) -> None:
        p = self.params
        lam = max(0.0, min(1.0, self.lambda_am))
        clash_by_agent: dict[str, float] = {}
        for f in self.faces():
            for k in (f.i1, f.i2):
                clash_by_agent[k] = max(clash_by_agent.get(k, 0.0), f.weight)
        self.clash_cache = clash_by_agent
        for k, a in self.live().items():
            if a.inflow <= 0 and not clash_by_agent.get(k):
                a.state = "idle"
                continue
            lat_leak = (
                1.0
                if a.S < p.s_min
                else max(0.0, (p.s_min + 0.15 - a.S) / 0.15)
                if a.S < p.s_min + 0.15
                else 0.0
            )
            art_leak = clash_by_agent.get(k, 0.0)
            leak_sig = lam * lat_leak + (1 - lam) * art_leak
            lat_stock = 1.0 if (a.dw_hat >= 0 and a.x >= 1 and a.S >= p.s_min) else 0.0
            conf = max((r.conf for r in self.rows_of(k)), default=0.5)
            art_stock = conf * (1 - art_leak) * (1.0 if a.x >= 1 else 0.4)
            stock_sig = lam * lat_stock + (1 - lam) * art_stock
            prefer_leak = self.drive == "leak"
            if leak_sig >= 0.5 and (
                leak_sig > stock_sig or (prefer_leak and leak_sig >= stock_sig)
            ):
                a.state = "leak"
            elif stock_sig >= 0.5:
                a.state = "stock"
            else:
                a.state = "idle"

    # ------------------------------------------------------------ heal (AM)
    def heal(self, a: Agent) -> dict[str, Any]:
        """Self-heal a leaking agent with the licensed operator that most lowers its Leak."""
        p = self.params
        base = am.relative_leak(am.arrivals(a.w_eff(), a.incoming, p))
        cause = max(
            (("op", sum(a.leak_op)), ("ov", sum(a.leak_ov)), ("q", sum(a.leak_q))),
            key=lambda t: t[1],
        )[0]
        lk_sp = SPECIES[max(range(8), key=lambda i: a.leak_op[i] + a.leak_ov[i] + a.leak_q[i])]
        sig = f"{a.cls}|{a.hero()}|{cause}|{lk_sp}"
        cands: dict[str, tuple[float, Any]] = {}
        remembered = self.memory.recall(sig)
        order = [remembered] if remembered else ["a", "r", "ra", "cut", "sigma", "arrow"]
        for op in order:
            pred = self._predict(a, op)
            if pred is not None:
                cands[op] = pred
        if remembered and not cands:
            for op in ("a", "r", "ra", "cut", "sigma", "arrow"):
                pred = self._predict(a, op)
                if pred is not None:
                    cands[op] = pred
        trials = len(cands)
        self.rounds += trials
        if not cands:
            return {
                "op": "e",
                "before": base,
                "after": base,
                "trials": 0,
                "memory": bool(remembered),
                "sig": sig,
            }
        op, (after, payload) = min(
            cands.items(), key=lambda t: (t[1][0] + OP_COST.get(t[0], 0.0), t[0])
        )
        if after >= base - 1e-6:
            return {
                "op": "e",
                "before": base,
                "after": base,
                "trials": trials,
                "memory": bool(remembered),
                "sig": sig,
            }
        self._apply(a, op, payload)
        a.last_op = op
        self.memory.store(sig, op)
        self.solves += 1
        return {
            "op": op,
            "before": base,
            "after": after,
            "trials": trials,
            "memory": bool(remembered),
            "sig": sig,
            "cause": cause,
            "species": lk_sp,
        }

    def _predict(self, a: Agent, op: str) -> tuple[float, Any] | None:
        p = self.params
        inc = a.incoming
        if op in ("a", "r", "ra"):
            if a.side == op:
                return None
            return am.relative_leak(am.arrivals(a.w_eff(op), inc, p)), op
        if op == "cut":
            hostile = [(src, sum(-x for x in off if x < 0)) for src, off in inc]
            hostile = [h for h in hostile if h[1] > 0]
            if not hostile:
                return None
            src = max(hostile, key=lambda t: t[1])[0]
            rest = [(s, o) for s, o in inc if s != src]
            return am.relative_leak(am.arrivals(a.w_eff(), rest, p)), src
        if op == "sigma":
            neigh = {s for s, _ in inc}
            best, bv, boff = None, 0.0, None
            weff = a.w_eff()
            for k, b in self.live().items():
                if k == a.id or k in neigh:
                    continue
                off = am.offer(b, self.R, p)
                if any(x < 0 for x, wv in zip(off, weff) if wv > p.theta):
                    continue
                v = sum(max(0.0, x) * wv for x, wv in zip(off, weff))
                if v > bv:
                    best, bv, boff = k, v, off
            if best is None or boff is None:
                return None
            return am.relative_leak(am.arrivals(weff, inc + [(best, boff)], p)), best
        if op == "arrow":
            if sum(a.leak_ov) <= 0:
                return None
            weff = list(a.w_eff())
            budget = sum(a.energy[a.side]) * 0.5 / p.c
            plan = []
            for gi in sorted(range(8), key=lambda i: -a.leak_ov[i]):
                if a.leak_ov[gi] <= 0 or budget <= 0:
                    continue
                d_plus = sum(max(0.0, off[gi]) for _, off in inc)
                need = max(0.0, d_plus - p.delta - weff[gi])
                add = min(need, budget)
                weff[gi] += add
                budget -= add
                plan.append((gi, add))
            if not plan:
                return None
            return am.relative_leak(am.arrivals(weff, inc, p)), plan
        return None

    def _apply(self, a: Agent, op: str, payload: Any) -> None:
        p = self.params
        if op in ("a", "r", "ra"):
            step = p.eta / (1.0 + a.m_side.get(op, 0.0))
            a.pi[op] = a.pi.get(op, 0.0) + step
            tot = sum(a.pi.values())
            a.pi = {k: v / tot for k, v in a.pi.items() if v / tot > 0.01}
            if op not in a.pi:
                a.pi[op] = 0.02
        elif op == "cut":
            self.cut.add((payload, a.id))
            self.am_edges.pop((payload, a.id), None)
        elif op == "sigma":
            self.stems.add((payload, a.id))
            self.stems.add((a.id, payload))
            self.am_edges[(payload, a.id)] = 0.0
            self.am_edges[(a.id, payload)] = 0.0
        elif op == "arrow":
            spent = 0.0
            g = a.side
            for gi, add in payload:
                ego = am.IDX[am.act(g, SPECIES[gi])]
                a.w[ego] += add
                spent += add * p.c
            E = a.energy[g]
            tot = sum(E) or 1.0
            for k in range(4):
                E[k] = max(0.0, E[k] - spent * E[k] / tot)

    # ------------------------------------------------------------ stock helpers
    def feed(self, agent_id: str, gain: float) -> None:
        """Artifact success flows back as mass intake in the agent's species (AX -> AM)."""
        rows = self.rows_of(agent_id)
        if not rows:
            return
        vec = [0.0] * 8
        for r in rows:
            for i in range(8):
                vec[i] += r.p[i] * gain / len(rows)
        cur = self.intake.get(agent_id, [0.0] * 8)
        self.intake[agent_id] = [x + y for x, y in zip(cur, vec)]

    def spend(self, a: Agent, amount: float) -> None:
        E = a.energy[a.side]
        tot = sum(E) or 1.0
        for k in range(4):
            E[k] = max(0.0, E[k] - amount * E[k] / tot)

    def adopt_pending(self, lattice: dict[str, Any]) -> int:
        """Instantiate controls kept after the last correspond (pr from their payer)."""
        born = 0
        for lv in lattice.get("levels", []) or []:
            lid = str(lv.get("level_id"))
            for r in lv.get("rules", []) or []:
                ref = f"pol/{lid}/{r.get('id')}"
                parent = self.agents.get(self.pending_parent.get(ref, ""))
                if parent is None:
                    continue
                self.pending_parent.pop(ref, None)
                node = tc.AXNode(
                    ref,
                    "rule",
                    f"{r.get('name', '')}: {r.get('pattern', '')} => {r.get('rewrite', '')}",
                    {"level": lid},
                )
                self.ax[ref] = node
                self.ax_edges.append((ref, f"pol/{lid}#mod", 0.0))
                self._instantiate(
                    node, tc.offline_row(node), "offline-lexicon", operator=parent.id, parent=parent
                )
                born += 1
        if born:
            self._rebuild_edges()
        return born

    # ------------------------------------------------------------ coverage
    def _coverage(self) -> None:
        p = self.params
        cells = set()
        for r in self.rows:
            if not r.alive:
                continue
            a = self.agents.get(r.agent_id)
            if not a or not a.alive or a.S < p.s_min or a.x < 1:
                continue
            for i in range(8):
                if r.p[i] > 0.15:
                    cells.add((r.ax_ref, SPECIES[i]))
        possible = sum(1 for r in self.rows if r.alive for v in r.p if v > 0.15)
        self.coverage_history.append(
            {
                "t": self.t,
                "space": len(cells),
                "possible": possible,
                "rounds": self.rounds,
                "solves": self.solves,
                "agents": len(self.live()),
                "rate": round(len(cells) / max(1, self.t) / max(1.0, self.rounds_per_solve()), 4),
            }
        )

    def rounds_per_solve(self) -> float:
        return self.rounds / self.solves if self.solves else float(self.rounds or 0)

    def ensemble_leak(self) -> float:
        tot_leak = sum(a.leak_total() for a in self.live().values())
        tot_in = sum(a.inflow for a in self.live().values())
        return tot_leak / (tot_in + self.params.delta) if tot_in > 0 else 0.0

    # ------------------------------------------------------------ export
    def agent_view(self, a: Agent) -> dict[str, Any]:
        p = self.params
        rows = self.rows_of(a.id)
        return {
            "id": a.id,
            "name": am.agent_name(a, p),
            "cls": a.cls,
            "rank": a.rank,
            "regime": a.regime,
            "state": a.state,
            "hero": a.hero(),
            "side": a.side,
            "S": round(a.S, 3),
            "L": round(a.L, 3),
            "x": round(a.x, 3),
            "w": [round(v, 3) for v in a.w],
            "op": a.op,
            "role": a.role,
            "alive": a.alive,
            "ax": rows[0].ax_ref if rows else "",
            "operator": a.operator,
            "clash": round(self.clash_cache.get(a.id, 0.0), 3),
        }

    def snapshot(self, limit_agents: int = 220) -> dict[str, Any]:
        live = self.live()
        agents = [self.agent_view(a) for a in list(live.values())[:limit_agents]]
        keep = {a["id"] for a in agents}
        return {
            "t": self.t,
            "drive": self.drive,
            "lambda_am": self.lambda_am,
            "params": self.params.as_dict(),
            "agents": agents,
            "edges": [list(e) for e in self.typed if e[0] in keep and e[1] in keep][:900],
            "ensembles": [
                {
                    k: e[k]
                    for k in ("id", "title", "text", "climate", "s", "L", "lam", "ranks", "members")
                }
                for e in self.ensembles[:24]
            ],
            "ax": [
                {"ref": n.ref, "kind": n.kind, "text": n.text[:180]}
                for n in list(self.ax.values())[:400]
            ],
            "ax_edges": [[a, b, round(A, 3)] for a, b, A in self.ax_edges[:900]],
            "toc": [r.as_dict() for r in self.rows if r.alive][:400],
            "events": self.events[-80:],
            "inspections": self.inspections[-30:],
            "coverage": self.coverage_history[-60:],
            "tombstones": self.tombstones,
            "rounds_per_solve": round(self.rounds_per_solve(), 3),
            "memory_hits": self.memory.hits,
        }

    def nx1_report(self) -> str:
        lines = [
            "# NX-1 report",
            "",
            f"run: {self.run_id}  tick: {self.t}  drive: {self.drive}  λ_AM: {self.lambda_am}",
            "",
        ]
        lines.append("## Ensembles")
        for e in self.ensembles:
            lines += ["", "```text", e["text"], "```"]
        lines += ["", "## Table of correspondences", "", "```text"]
        lines += [r.nx1() for r in self.rows]
        lines += ["```", "", "## Event log", "", "```text"]
        lines += [e["text"] for e in self.events]
        lines += ["```", ""]
        return "\n".join(lines)


def viable_operations(lattice: dict[str, Any], cap: int) -> list[dict[str, Any]]:
    """Every viable control operation: one *add control* per under-covered function.

    A function is under-covered when its share of the policy is below 1/8
    (an even spread). Each operation becomes one agent (NX-1 §4.3).
    """
    from .security import FUNCTION, function_vector

    if cap <= 0:
        return []
    W = function_vector(lattice)
    tot = sum(W) or 1.0
    levels = [str(lv.get("level_id")) for lv in lattice.get("levels", []) or []]
    target_level = "L1" if "L1" in levels else (levels[-1] if levels else "L1")
    weak = sorted(range(8), key=lambda i: (W[i] / tot, i))
    out = []
    for i in weak:
        if W[i] / tot >= 1 / 8 or len(out) >= cap:
            continue
        sp = SPECIES[i]
        tag = hashlib.sha1(f"add_control:{sp}".encode()).hexdigest()[:4]
        out.append(
            {
                "id": f"add_control-{sp}-{tag}",
                "operation": "add_control",
                "function": sp,
                "target": f"pol/{target_level}#mod",
                "text": f"add a control for {FUNCTION[sp]}",
            }
        )
    return out


__all__ = ["DRIVES", "SIDES", "World", "drive_config", "viable_operations"]
