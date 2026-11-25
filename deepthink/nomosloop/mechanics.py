"""Agent Mechanics: the formal apparatus the RSI lattice runs on.

Implements the parts of the *Agent Mechanics* treatise (Formal Apparatus,
Parts I, IV, IV·bis, Volume II) that the RSI engine needs, plus the
conventions of ``docs/NOMENCLATURE_COMPLEMENT.md`` (NX-1):

* eight species ``F``, the involutions ``α`` (attitude) and ``β`` (Grant axis),
  the Klein group ``K = {e, r, a, ra}`` acting on species and stacks;
* the 16 valid stacks and their four-letter labels;
* the channel matrix ``R`` (schematic profile of Part I §4, scalable);
* offer, flux, voltage, Leak split by cause (hostile ``op``, overdrive ``ov``,
  sub-quantum ``q``), stability ``S``, the tick law with energy and
  transduction (Apparatus §8, §10, §14);
* variety rank, regime and the NX-1 agent name.

Everything numeric is a model output under uncalibrated parameters
(treatise [OG6]); nothing here measures a real system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

SPECIES: tuple[str, ...] = ("Ni", "Ne", "Si", "Se", "Ti", "Te", "Fi", "Fe")
IDX: dict[str, int] = {s: i for i, s in enumerate(SPECIES)}
PERCEPTION = frozenset({"Ni", "Ne", "Si", "Se"})
JUDGMENT = frozenset({"Ti", "Te", "Fi", "Fe"})
ALPHA = {
    "Ni": "Ne",
    "Ne": "Ni",
    "Si": "Se",
    "Se": "Si",
    "Ti": "Te",
    "Te": "Ti",
    "Fi": "Fe",
    "Fe": "Fi",
}
BETA = {
    "Ni": "Se",
    "Se": "Ni",
    "Ne": "Si",
    "Si": "Ne",
    "Te": "Fi",
    "Fi": "Te",
    "Ti": "Fe",
    "Fe": "Ti",
}
SIDES: tuple[str, ...] = ("e", "r", "a", "ra")
RANKS: tuple[str, ...] = ("Inr", "Seed", "Res", "Cyc", "Attr")
# R-I partition into rings (Volume II, Proposition 1; working choice [OG1]).
RINGS = (
    frozenset({"Ni", "Te"}),
    frozenset({"Fi", "Se"}),
    frozenset({"Ne", "Ti"}),
    frozenset({"Fe", "Si"}),
)
STORING_EDGES = frozenset({"st", "or", "fl", "du"})


def attitude(f: str) -> str:
    return f[1]


def act(side: str, f: str) -> str:
    """Action of a Klein element on a species (Apparatus §2)."""
    if side == "e":
        return f
    if side == "a":
        return ALPHA[f]
    if side == "r":
        return BETA[f]
    if side == "ra":
        return ALPHA[BETA[f]]
    raise ValueError(f"unknown side {side!r}")


def is_valid_stack(stack: tuple[str, ...] | list[str]) -> bool:
    """Valid iff s4 = β(s1), s3 = β(s2), s2 differs from s1 in class and attitude."""
    if len(stack) != 4 or len(set(stack)) != 4:
        return False
    s1, s2, s3, s4 = stack
    same_class = (s1 in PERCEPTION) == (s2 in PERCEPTION)
    return s4 == BETA[s1] and s3 == BETA[s2] and not same_class and attitude(s1) != attitude(s2)


def _all_valid_stacks() -> tuple[tuple[str, ...], ...]:
    out = []
    for s1 in SPECIES:
        for s2 in SPECIES:
            st = (s1, s2, BETA[s2], BETA[s1])
            if is_valid_stack(st):
                out.append(st)
    return tuple(out)


VALID_STACKS = _all_valid_stacks()


def stack_label(stack: tuple[str, ...]) -> str:
    """Four-letter label of a valid stack (e.g. Ni,Te,Fi,Se -> INTJ)."""
    s1, s2 = stack[0], stack[1]
    ie = "I" if attitude(s1) == "i" else "E"
    perc = s1 if s1 in PERCEPTION else s2
    judg = s1 if s1 in JUDGMENT else s2
    extraverted = s1 if attitude(s1) == "e" else s2
    jp = "J" if extraverted in JUDGMENT else "P"
    return f"{ie}{perc[0]}{judg[0]}{jp}"


LABEL_TO_STACK = {stack_label(s): s for s in VALID_STACKS}


def klein_stack(side: str, stack: tuple[str, ...]) -> tuple[str, ...]:
    """The writing of a stack on another side (elementwise action)."""
    return tuple(act(side, f) for f in stack)


def positions(hero: str) -> dict[str, int]:
    """Position k ∈ 1..4 of each species, relative to the hero (Volume II §1).

    Position 1 is the ring holding the hero, 2 the ring of β(hero), 3 of
    α(hero), 4 of αβ(hero). Rings are K-invariant, so the four are distinct.
    """
    anchors = (hero, BETA[hero], ALPHA[hero], ALPHA[BETA[hero]])
    pos: dict[str, int] = {}
    for k, anchor in enumerate(anchors, start=1):
        ring = next(r for r in RINGS if anchor in r)
        for f in ring:
            pos.setdefault(f, k)
    return pos


# ---------------------------------------------------------------- params ----


@dataclass
class Params:
    """Treatise parameters (all uncalibrated, [OG6]). Defaults are 'balanced'."""

    h: float = 0.25  # tick step
    delta: float = 0.05  # admittance floor δ
    E0: float = 1.0  # gate half-energy
    c: float = 0.03  # energy-to-stock conversion (units: flux is O(1) at r_scale 0.1)
    zeta: float = 0.15  # damage leak does to capacity ζ
    q: float = 0.02  # registration quantum q(g)
    theta: float = 0.35  # support threshold θ
    theta_k: float = 0.15  # online-writing threshold θ_K
    s_min: float = 0.5  # storing threshold S_min
    mu: float = 0.3  # side-mass gain μ
    rho_m: float = 0.02  # side-mass forgetting ρ_M
    eta: float = 0.6  # rewriting step η (divided by 1 + M_side)
    theta_p: float = 0.3  # default transduction θ
    energy_intake: float = 1.0  # nominal energy intake ι_E (compute budget)
    r_cover: float = 2.0
    r_orbit: float = 1.0
    r_diag: float = -2.0
    r_self: float = 0.0
    r_antisym: float = 0.0  # one-way conductivity P -> J (Apparatus §2, R_a)
    r_scale: float = 0.1

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}

    def profile(self) -> str:
        s = self.r_scale
        return f"schem({fmt(self.r_cover * s)},{fmt(self.r_orbit * s)},{fmt(self.r_diag * s)},{fmt(self.r_self * s)})"


def channel_matrix(p: Params) -> list[list[float]]:
    """Schematic R (Part I §4) with optional antisymmetric part."""
    R = [[0.0] * 8 for _ in SPECIES]
    for f in SPECIES:
        i = IDX[f]
        R[i][IDX[BETA[f]]] = p.r_cover * p.r_scale
        R[i][IDX[ALPHA[f]]] = p.r_orbit * p.r_scale
        R[i][IDX[ALPHA[BETA[f]]]] = p.r_diag * p.r_scale
        R[i][i] = p.r_self * p.r_scale
    if p.r_antisym:
        for f in PERCEPTION:
            for g in JUDGMENT:
                R[IDX[f]][IDX[g]] += p.r_antisym
                R[IDX[g]][IDX[f]] -= p.r_antisym
    return R


# ----------------------------------------------------------------- agent ----


def fmt(x: float) -> str:
    """Two decimals, no trailing zeros (IV·bis §7)."""
    s = f"{float(x):.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


@dataclass
class Agent:
    """Canonical tuple (M, K, w, M_side, E, λ, ψ, Z, a) plus RSI bookkeeping.

    ``w`` is the ego-frame stock. The admittance actually used in a tick is
    the one transported to the online side (Volume II, hypothesis HW).
    """

    id: str
    cls: str
    stack: tuple[str, ...]
    w: list[float]
    pi: dict[str, float] = field(default_factory=lambda: {"e": 1.0})
    m_side: dict[str, float] = field(default_factory=lambda: {s: 0.5 for s in SIDES})
    energy: dict[str, list[float]] = field(default_factory=dict)
    lam: float = 1.0
    theta_p: float = 0.3
    end: float = 0.25
    aut: int = 0
    op: str = ""
    role: str = ""
    rows: list[int] = field(default_factory=list)
    operator: str = ""
    alive: bool = True
    born: int = 0
    fatigue: float = 0.0
    # readings of the last tick
    G: list[float] = field(default_factory=lambda: [0.0] * 8)
    leak_op: list[float] = field(default_factory=lambda: [0.0] * 8)
    leak_ov: list[float] = field(default_factory=lambda: [0.0] * 8)
    leak_q: list[float] = field(default_factory=lambda: [0.0] * 8)
    inflow: float = 0.0
    S: float = 1.0
    L: float = 0.0
    emitted: float = 0.0
    dw_hat: float = 0.0
    x: float = 0.0
    own_intake: float = 0.0
    rank: str = "Seed"
    regime: str = "cond"
    climate: str = "TrC"
    state: str = "idle"  # idle | leak | stock
    last_op: str = ""
    incoming: list[tuple[str, list[float]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.energy:
            self.energy = {s: [1.1, 1.1, 1.1, 1.1] if s == "e" else [0.3] * 4 for s in SIDES}

    # -- writing / admittance
    @property
    def side(self) -> str:
        return max(SIDES, key=lambda s: (self.pi.get(s, 0.0), -SIDES.index(s)))

    def k_on(self, theta_k: float) -> list[str]:
        return [s for s in SIDES if self.pi.get(s, 0.0) > theta_k]

    def w_eff(self, side: str | None = None) -> list[float]:
        g = side or self.side
        return [self.w[IDX[act(g, f)]] for f in SPECIES]

    def hero(self, side: str | None = None) -> str:
        g = side or self.side
        if self.stack:
            return act(g, self.stack[0])
        weff = self.w_eff(g)
        return SPECIES[max(range(8), key=lambda i: weff[i])]

    def gate(self, p: Params, side: str | None = None) -> list[float]:
        g = side or self.side
        pos = positions(self.hero(g))
        E = self.energy[g]
        return [E[pos[f] - 1] / (E[pos[f] - 1] + p.E0) for f in SPECIES]

    def reserve(self, p: Params) -> float:
        """Total reserve ŵ = Σw + ΣE/c (Apparatus §9)."""
        return sum(self.w) + sum(sum(v) for v in self.energy.values()) / p.c

    def leak_total(self) -> float:
        return sum(self.leak_op) + sum(self.leak_ov) + sum(self.leak_q)

    def support(self, p: Params) -> int:
        return sum(1 for v in self.w if v > p.theta)


def offer(agent: Agent, R: list[list[float]], p: Params) -> list[float]:
    """e_i(g) = Σ_f γ_i(f) w_i(f) R_{f,g} on the online side (Apparatus §14.2)."""
    weff = agent.w_eff()
    gam = agent.gate(p)
    out = [0.0] * 8
    for fi in range(8):
        a = gam[fi] * weff[fi]
        if a == 0.0:
            continue
        row = R[fi]
        for gi in range(8):
            if row[gi]:
                out[gi] += a * row[gi]
    return out


def encounter_mass(i: Agent, j: Agent, R: list[list[float]]) -> float:
    """m(i, j) = w_iᵀ R w_j on the online sides (ungated)."""
    wi, wj = i.w_eff(), j.w_eff()
    return sum(wi[f] * R[f][g] * wj[g] for f in range(8) if wi[f] for g in range(8) if R[f][g])


def arrivals(
    weff: list[float], incoming: list[tuple[str, list[float]]], p: Params
) -> dict[str, list[float]]:
    """Apparatus §8 for one receiver given phase-adjusted incoming offers."""
    G = [0.0] * 8
    ov = [0.0] * 8
    subq = [0.0] * 8
    op = [0.0] * 8
    for gi in range(8):
        w = weff[gi]
        d_plus = 0.0
        in_q = 0.0
        in_sub = 0.0
        for _src, off in incoming:
            o = off[gi]
            phi = o * w
            if o > 0:
                d_plus += o
                if phi >= p.q:
                    in_q += phi
                else:
                    in_sub += phi
            elif o < 0:
                op[gi] += -phi
        v_plus = d_plus / (w + p.delta) if d_plus > 0 else 0.0
        chi = min(1.0, 1.0 / v_plus) if v_plus > 0 else 1.0
        G[gi] = chi * in_q
        ov[gi] = (1.0 - chi) * in_q
        subq[gi] = in_sub
    return {"G": G, "ov": ov, "q": subq, "op": op}


def relative_leak(res: dict[str, list[float]]) -> float:
    g = sum(res["G"])
    lk = sum(res["ov"]) + sum(res["q"]) + sum(res["op"])
    return lk / (g + lk) if (g + lk) > 0 else 0.0


@dataclass
class TickReport:
    fluxes: dict[tuple[str, str], list[float]]
    contrib: dict[tuple[str, str], list[int]]
    events: list[str]


def tick(
    agents: dict[str, Agent],
    edges: dict[tuple[str, str], float],
    R: list[list[float]],
    p: Params,
    *,
    intake: dict[str, list[float]] | None = None,
    t: int = 0,
) -> TickReport:
    """One canonical tick (Apparatus §10 with §14.3/§14.6).

    ``edges`` maps (src, dst) to the link phase A; matter couples through
    U = e^{iA}, so the real current carries cos A (Apparatus §7).
    ``intake`` is the mass intake ι(g) per agent (artifact feedback).
    """
    live = {k: a for k, a in agents.items() if a.alive}
    offers = {k: offer(a, R, p) for k, a in live.items()}
    incoming: dict[str, list[tuple[str, list[float]]]] = {k: [] for k in live}
    fluxes: dict[tuple[str, str], list[float]] = {}
    contrib: dict[tuple[str, str], list[int]] = {}
    emit_pos: dict[str, list[float]] = {k: [0.0] * 4 for k in live}
    emit: dict[str, float] = {k: 0.0 for k in live}

    for (src, dst), A in edges.items():
        if src not in live or dst not in live:
            continue
        ca = math.cos(A)
        eff = [o * ca for o in offers[src]]
        incoming[dst].append((src, eff))
        wj = live[dst].w_eff()
        phi = [eff[g] * wj[g] for g in range(8)]
        fluxes[(src, dst)] = phi
        # dominant contributing species per receiving species (for edge typing)
        a = live[src]
        weff, gam = a.w_eff(), a.gate(p)
        dom = []
        for g in range(8):
            best, bv = -1, 0.0
            for f in range(8):
                v = abs(gam[f] * weff[f] * R[f][g])
                if v > bv:
                    best, bv = f, v
            dom.append(best)
        contrib[(src, dst)] = dom
        pos = positions(a.hero())
        for f in range(8):
            base = gam[f] * weff[f]
            if not base:
                continue
            tot = sum(abs(base * R[f][g] * wj[g] * ca) for g in range(8))
            emit_pos[src][pos[SPECIES[f]] - 1] += tot
        emit[src] += sum(abs(x) for x in phi)

    events: list[str] = []
    intake = intake or {}
    for k, a in live.items():
        before = a.reserve(p)
        res = arrivals(a.w_eff(), incoming[k], p)
        a.incoming = incoming[k]
        a.G, a.leak_ov, a.leak_q, a.leak_op = res["G"], res["ov"], res["q"], res["op"]
        g_side = a.side
        pos = positions(a.hero())
        iota = intake.get(k, [0.0] * 8)
        # stock update on the online frame, written back to the ego frame
        for gi, f in enumerate(SPECIES):
            leak_g = res["ov"][gi] + res["q"][gi] + res["op"][gi]
            dw = p.h * ((1.0 - a.theta_p) * res["G"][gi] + iota[gi] - p.zeta * leak_g)
            ego = IDX[act(g_side, f)]
            a.w[ego] = max(0.0, a.w[ego] + dw)
        # energy update per position of the online side
        own = a.end * p.energy_intake * a.lam
        a.own_intake = own
        g_pos = [0.0] * 4
        for gi, f in enumerate(SPECIES):
            g_pos[pos[f] - 1] += res["G"][gi]
        tot_o = sum(emit_pos[k]) or 1.0
        E = a.energy[g_side]
        for kk in range(4):
            dE = p.h * (
                own / 4.0 + a.theta_p * p.c * g_pos[kk] - p.c * emit[k] * emit_pos[k][kk] / tot_o
            )
            E[kk] = max(0.0, E[kk] + dE)
        # side mass (Apparatus §14.4)
        stored = sum((1.0 - a.theta_p) * x for x in res["G"])
        for s in SIDES:
            if s == g_side:
                a.m_side[s] += p.h * (p.mu * stored - p.rho_m * a.m_side[s])
            else:
                a.m_side[s] -= p.h * p.rho_m * a.m_side[s]
        g_tot = sum(res["G"])
        lk = a.leak_total()
        a.inflow = g_tot + lk
        a.S = g_tot / (g_tot + lk) if (g_tot + lk) > 0 else 1.0
        a.L = lk / (g_tot + lk + p.delta) if (g_tot + lk) > 0 else 0.0
        a.emitted = emit[k]
        a.dw_hat = a.reserve(p) - before
        held = [pos[f] - 1 for f, v in zip(SPECIES, a.w_eff()) if v > p.theta] or [0, 1, 2, 3]
        a.x = min(E[i] for i in held) / p.E0
        a.fatigue = max(0.0, 0.9 * a.fatigue + 0.1 * min(1.0, a.emitted / 10.0))
    return TickReport(fluxes=fluxes, contrib=contrib, events=events)


# ------------------------------------------------------------- edges/circuits


def type_edge(
    src: Agent, dst: Agent, phi: list[float], dom: list[int], p: Params
) -> tuple[str, str, bool]:
    """Typed edge of IV·bis Table 2 for (src -> dst). Returns (type, species, stores)."""
    g = max(range(8), key=lambda i: abs(phi[i]))
    sp = SPECIES[g]
    if abs(phi[g]) < 1e-9:
        return ("fl", sp, False)
    if phi[g] < 0:
        return ("op", sp, False)
    G = dst.G[g]
    lk = dst.leak_op[g] + dst.leak_ov[g] + dst.leak_q[g]
    stores = (G / (G + lk) if (G + lk) > 0 else 0.0) >= p.s_min
    if not stores:
        return ("sv", sp, False)
    f = dom[g]
    if f >= 0 and SPECIES[f] == BETA[sp]:
        return ("st", sp, True)
    if f >= 0 and SPECIES[f] == ALPHA[sp]:
        return ("or", sp, True)
    return ("fl", sp, True)


def find_cycles(
    adj: dict[str, list[str]], max_len: int = 5, limit: int = 120
) -> list[tuple[str, ...]]:
    """Simple directed cycles up to ``max_len``, rotated to start at the lowest id."""
    out: set[tuple[str, ...]] = set()
    nodes = sorted(adj)
    for start in nodes:
        stack = [(start, (start,))]
        while stack and len(out) < limit:
            node, path = stack.pop()
            for nxt in adj.get(node, []):
                if nxt == start and len(path) >= 2:
                    out.add(path)
                elif nxt not in path and nxt > start and len(path) < max_len:
                    stack.append((nxt, path + (nxt,)))
    return sorted(out, key=lambda c: (len(c), c))


def signature(stores: list[bool]) -> str:
    if all(stores):
        return "store"
    if not any(stores):
        return "pol"
    return "mix"


# ------------------------------------------------------------ rank / name ----


def next_rank(
    a: Agent,
    p: Params,
    *,
    in_store_cycle: bool,
    operator_rank: str | None,
    prev_L: float,
) -> str:
    """Promotion laws of Part IV §5, one step per tick, with rank ceilings."""
    kon = len(a.k_on(p.theta_k))
    supp = a.support(p)
    pure = a.end <= 0.0 and a.aut == 0
    if a.S < 0.10 or kon == 0:
        target = "Inr"
    elif (
        kon == 4
        and supp == 8
        and a.L < 0.02
        and a.x >= 1
        and a.own_intake >= p.c * a.emitted * p.h
        and a.cls not in ("E", "T")
        and not pure
    ):
        target = "Attr"
    elif in_store_cycle and a.x >= 1:
        target = "Cyc"
    elif kon >= 2 and a.x >= 1 and (a.L < prev_L or a.L < 0.2):
        target = "Res"
    else:
        target = "Seed"
    cur = RANKS.index(a.rank) if a.rank in RANKS else 1
    tgt = RANKS.index(target)
    nxt = cur + (1 if tgt > cur else -1 if tgt < cur else 0)
    if pure and operator_rank in RANKS:
        nxt = min(nxt, RANKS.index(operator_rank))
    return RANKS[max(0, nxt)]


def orbit(a: Agent, p: Params) -> str:
    """Orbit field: label + flip for a valid stack, else stack by projection (IV·bis §9.4)."""
    side = a.side
    flip = "" if side == "e" else f".{side}"
    if a.stack and len(a.stack) == 4 and is_valid_stack(a.stack):
        return stack_label(a.stack) + flip
    weff = a.w_eff()
    order = sorted(range(8), key=lambda i: (-weff[i], i))
    top = [SPECIES[i] for i in order if weff[i] > p.theta][:4]
    if not top:
        top = [SPECIES[order[0]]]
    if len(top) == 4 and is_valid_stack(tuple(top)):
        return stack_label(tuple(top))
    return ">".join(top)


def climate_of(W: list[float], has_store: bool) -> str:
    """Climate reading convention of NX-1 §5.1 (pending treatise O9)."""
    g = SPECIES[max(range(8), key=lambda i: W[i])] if any(W) else "Si"
    dom = "Tr" if g in ("Ni", "Si") else "Fm" if g in ("Ne", "Se") else "Fd"
    return dom + ("O" if has_store else "C")


def l_band(L: float) -> str:
    return fmt(L)


def agent_name(a: Agent, p: Params) -> str:
    """NX-1 atom name with measures in canonical order (§2.1)."""
    head = f"{a.rank}-{orbit(a, p)}-{a.regime}-{a.climate}"
    m = [f"/s={a.support(p)}", f"/L={l_band(a.L)}", f"/S={fmt(a.S)}"]
    if sum(a.G) > 0:
        m.append(f"/G={fmt(sum(a.G))}")
    causes = []
    for gi in sorted(range(8), key=lambda i: -(a.leak_op[i] + a.leak_ov[i] + a.leak_q[i])):
        for cause, vec in (("op", a.leak_op), ("ov", a.leak_ov), ("q", a.leak_q)):
            if vec[gi] > 0.05:
                causes.append(f"{SPECIES[gi]}:{cause}{fmt(vec[gi])}")
    if causes:
        m.append("/Lk=" + "+".join(causes[:4]))
    ws = sorted(
        ((a.w[i], SPECIES[i]) for i in range(8) if a.w[i] > 0.05), key=lambda t: (-t[0], IDX[t[1]])
    )
    if ws:
        m.append("/w=" + ",".join(f"{s}{fmt(v)}" for v, s in ws))
    kon = [s for s in SIDES if a.pi.get(s, 0) > 0.05]
    m.append("/K=" + "+".join(f"{s}:{fmt(a.pi[s])}" for s in kon))
    m.append(f"/E={fmt(a.x)}")
    m.append(f"/Ms={fmt(a.m_side.get(a.side, 0.0))}")
    if a.cls != "B":
        m.append(f"/end={fmt(a.end)}")
        m.append(f"/aut={int(a.aut)}")
        m.append(f"/th={fmt(a.theta_p)}")
    if a.op:
        m.append(f"/op={a.op}")
    if a.role:
        m.append(f"/role={a.role}")
    return head + "".join(m)


def atom(a: Agent, p: Params) -> str:
    return f"{a.id}={agent_name(a, p)}@{fmt(a.lam)}"


def ensemble_string(
    title: str,
    members: list[Agent],
    tombstones: int,
    edges: list[tuple[str, str, str, str]],
    rings: list[str],
    p: Params,
    *,
    t: int,
    drive: str,
) -> dict[str, Any]:
    """Canonical NX-1 ensemble string (IV·bis §5–§7 with §2 extensions)."""
    order = {c: i for i, c in enumerate("BIEAMTQ")}

    def key(a: Agent) -> tuple:
        num = int("".join(ch for ch in a.id if ch.isdigit()) or 0)
        return (order.get(a.cls, 9), num)

    members = sorted(members, key=key)
    W = [0.0] * 8
    lam = 0.0
    leak = 0.0
    inflow = 0.0
    prof = [0] * 5
    for a in members:
        lam += a.lam
        for i in range(8):
            W[i] += a.lam * a.w[i]
        leak += a.leak_total()
        inflow += a.inflow
        prof[RANKS.index(a.rank) if a.rank in RANKS else 1] += 1
    s_g = sum(1 for v in W if v > p.theta)
    L_g = leak / (inflow + p.delta) if inflow > 0 else 0.0
    climate = climate_of(W, any("/store" in r or "store4" in r for r in rings))
    ids = {a.id for a in members}
    tord = {k: i for i, k in enumerate(["st", "fl", "du", "op", "ax", "sv", "pr", "at", "or"])}
    es = sorted(
        (e for e in edges if e[0] in ids and e[1] in ids),
        key=lambda e: (key(next(a for a in members if a.id == e[0])), tord.get(e[2], 9), e[1]),
    )
    head = (
        f"n={len(members)};x={tombstones};lam={fmt(lam)};r={'/'.join(str(x) for x in prof)};"
        f"s={s_g};L={fmt(L_g)};t={t};R={p.profile()};D={drive}"
    )
    atoms = ",\n  ".join(atom(a, p) for a in members)
    graph_parts = []
    for s, d, typ, sp in es:
        graph_parts.append(f"{s} -{typ}[{sp}]- {d}" if typ == "du" else f"{s} -{typ}[{sp}]-> {d}")
    graph_parts.extend(rings)
    text = f"{title}[{head}]{{\n  {atoms}\n  |\n  " + "; ".join(graph_parts) + f"\n  |{climate}}}"
    return {
        "title": title,
        "text": text,
        "header": head,
        "climate": climate,
        "W": [round(v, 3) for v in W],
        "s": s_g,
        "L": round(L_g, 4),
        "lam": round(lam, 3),
        "ranks": prof,
        "members": [a.id for a in members],
    }
