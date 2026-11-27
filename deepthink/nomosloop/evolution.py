"""Active evolution: alternate universes bred in the lattice, read as what-ifs on the policy.

When the phase is on, the loop branches the world into **alternate
universes**. Each universe is the current lattice plus one or more mutations
taken from the operators of the *Agent Mechanics* treatise and NX-1, applied
only where the lattice licenses them, and each mutation is also an edit of
the policy, so every universe reads as a plain "what if":

==============  =======================================  ===================================
operator        agent space (treatise)                   what if, in the policy
==============  =======================================  ===================================
ev_a ev_r ev_ra the online writing moves to g ∈ K; the   the clause is mirrored (ingress ↔
                identifier is kept (Klein group,         egress, read ↔ write), serves its
                Apparatus §2; evolution ev_i, IV·bis §8) complementary function, or becomes
                                                         institution-wide
σ stem          a storing edge to the agent whose hero   the clause is explicitly backed by
                covers this agent's weakest slot         the clause that covers what it
                (NX-1 §5; Part IV §5, I→II)              holds least
store4          three children close a four-cycle of     the loop is closed with three
                stems (Part IV §5, II→III)               controls: log, alert, review
η_g hairpin     a Klein-close child of its parent,       the clause and its mirror,
                paid with energy (Part IV §5, III→IV)    complement or norm form one standard
birth           a·(αX, βY): a child Klein-close to the   a new clause for what two linked
                winner, stocked with r of the pair's     clauses lose between them
                residual (Part I §9; IV·bis Table 4)
→ arrow         the parent dissolves and two children    the clause is split in two
                share its mass (NX-1 §6.4; restriction)
cut             the opponent's edge is cut, w → 0        the clause yields: "…, except
                (NX-1 §6.4, shielding)                   where §j applies"
†               dissolution: tombstone !x (IV·bis §8)    the clause is removed
ω               off-license emission, heat only          the clause is bypassed (a probe;
                (NX-1 §5)                                never adopted)
==============  =======================================  ===================================

Which operators an agent may use depends on its rank, as in Part IV: any
paid agent can flip or stem, a Res agent can close a ring, a Cyc agent can
hairpin, and a linked pair that leaks can give birth. Structures therefore
breed structures across generations: a ring bred in one generation lifts its
members toward Cyc, and in the next generation they can hairpin.

Every universe is ticked and measured in both spaces against a baseline
ticked the same number of times: posture and contradictions of its
hypothetical policy (artifact space), ensemble leak, clash mass, rings and
ranks (agent space). The fittest breed the next generation, by one more
mutation or by the union of two survivors' edits. Each universe carries its
diff, its consequences and the structures it bred.

Universes never touch the real policy unless ``adopt="gate"``. Then the best
universe whose posture rises with no new contradiction, and which neither
removes, bypasses nor weakens a restriction, is applied to the policy; its
agents are re-derived from the adopted text at the next correspond (the
winner pulled back through δ_A, NX-1 §6.4).

Weights, mass shares and licensing thresholds here are conventions of this
engine; the treatise leaves them open ([OG6]). The phase makes no model
calls.
"""

from __future__ import annotations

import copy
import math
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from . import correspondence as tc
from . import mechanics as am
from . import security, surface
from .mechanics import BETA, IDX, RANKS, SIDES, SPECIES, act, agent_name, klein_stack
from .world import World, viable_operations

TRIGGERS = ("always", "calm")
ADOPT = ("none", "gate")
BIRTH_MIN = 0.05  # residual a linked pair must leak before a birth is licensed [C]


@dataclass
class EvolveConfig:
    """Toggle and budget of the active-evolution phase."""

    enabled: bool = False
    universes: int = 6  # universes bred per generation
    generations: int = 2  # each generation breeds from the survivors of the last
    keep: int = 3  # survivors per generation
    ticks: int = 3  # ticks each universe runs before it is measured
    probes: int = 2  # ω probes per loop: what if a restriction were bypassed
    trigger: str = "always"  # always | calm (only when no agent leaks)
    adopt: str = "none"  # none | gate (apply the best licensed universe to the policy)
    seed: int | None = None  # None: derived from the policy and the loop

    @classmethod
    def from_dict(cls, d: Any) -> EvolveConfig:
        src = dict(d) if isinstance(d, dict) else {}

        def num(key: str, lo: int, hi: int, default: int) -> int:
            try:
                v = int(src.get(key, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        seed: int | None
        try:
            seed = None if src.get("seed") in (None, "") else int(src["seed"])
        except (TypeError, ValueError):
            seed = None
        trig = str(src.get("trigger") or "always")
        adopt = str(src.get("adopt") or "none")
        return cls(
            enabled=bool(src.get("enabled", False)),
            universes=num("universes", 1, 16, 6),
            generations=num("generations", 1, 4, 2),
            keep=num("keep", 1, 8, 3),
            ticks=num("ticks", 1, 8, 3),
            probes=num("probes", 0, 6, 2),
            trigger=trig if trig in TRIGGERS else "always",
            adopt=adopt if adopt in ADOPT else "none",
            seed=seed,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


LAW: dict[str, str] = {
    "ev": "Klein group K = {e, a, r, ra} (Apparatus §2); evolution ev_i keeps the identifier (IV·bis §8)",
    "sigma": "σ, stem licensing: a storing edge to the agent whose hero covers the weakest valued slot (NX-1 §5; Part IV §5, I→II)",
    "store4": "store4: a four-cycle of storing arrows (Part IV §5, II→III)",
    "hairpin": "hairpin η_g(M) = (M, g·M) with a Klein-close child (Part IV §5, III→IV)",
    "birth": "birth law: only a·(αX, βY) creates a terminal, by emitting r of the residual, Klein-close to the winner (Part I §9); birth is a pr edge plus a new atom (IV·bis Table 4)",
    "arrow": "→ on overload: split the overloaded scope (NX-1 §6.4); restriction G|S (IV·bis §8)",
    "cut": "cut, shielding: w_border → 0 on the clashing edge (NX-1 §6.4)",
    "dissolve": "dissolution †i: the atom becomes a tombstone !x and its id is not reused (IV·bis §8)",
    "omega": "ω, off-license emission: heat, never rank (NX-1 §5); a probe, never adopted",
}
KLEIN_NAME = {
    "a": "mirror (α, the opposite side)",
    "r": "complement (β, the complementary function)",
    "ra": "norm form (αβ, institution-wide)",
}

# One clause per security function, about an object of the policy ({obj}).
FUNCTION_CLAUSE: dict[str, str] = {
    "Ni": "Attack paths that reach {obj} are analysed before any change goes live.",
    "Ne": "{Obj} is tested by authorized red-team exercises at least twice a year.",
    "Si": "Every use of {obj} is logged and the logs are retained for review.",
    "Se": "Activity on {obj} is monitored in real time, and anomalies raise an alert.",
    "Ti": "Access to {obj} follows explicit rules, and every request is validated.",
    "Te": "Incidents involving {obj} have a tested response and patch procedure.",
    "Fi": "Only verified identities with least privilege can use {obj}, and its integrity is checked.",
    "Fe": "Owners of {obj} are notified of every change, and rollouts are coordinated.",
}
RING = (
    ("log", "Si", FUNCTION_CLAUSE["Si"]),
    ("alert", "Se", FUNCTION_CLAUSE["Se"]),
    ("review", "Te", "Alerts about {obj} are reviewed and acted on within one working day."),
)

# ------------------------------------------------------------------ text ---

_AUX = "is|are|must|shall|may|can|will|should"
_DET = "the|any|every|all|each|a|an"
_OBJ = re.compile(
    rf"\b(?:to|on|of|in|for|from|by)\s+((?:{_DET})\s+[^,.;:]+?)"
    rf"(?=\s+(?:at|from|to|only|outside|inside|during|through|{_AUX}|for|by|with|on|that|"
    rf"which|when|unless|except|and|or)\b|[,.;:]|$)",
    re.I,
)
_SUBJ = re.compile(rf"^\s*(.+?)\s+(?:{_AUX})\b", re.I)
_SWAP = {
    "inbound": "outbound",
    "outbound": "inbound",
    "ingress": "egress",
    "egress": "ingress",
    "incoming": "outgoing",
    "outgoing": "incoming",
    "upload": "download",
    "download": "upload",
    "import": "export",
    "export": "import",
    "read": "write",
    "write": "read",
    "receive": "send",
    "send": "receive",
}
_SWAP_RX = re.compile(r"\b(" + "|".join(sorted(_SWAP, key=len, reverse=True)) + r")(s?)\b", re.I)
_TOFROM = re.compile(
    rf"\bto ((?:(?:{_DET})\s+)?[^,.;]+?) from ((?:(?:{_DET})\s+)?[^,.;]+?)(?=[,.;]|$)", re.I
)
_NEGATE = (
    (r"\bmust never\b", "may"),
    (r"\bmust not\b", "may"),
    (r"\bshall not\b", "may"),
    (r"\bnever\b", "sometimes"),
    (r"\bis denied\b", "is granted"),
    (r"\bare denied\b", "are granted"),
    (r"\bdeny\b", "allow"),
    (r"\bblocks\b", "allows"),
    (r"\bblock\b", "allow"),
    (r"\bdisabled\b", "enabled"),
    (r"\bprohibited\b", "allowed"),
    (r"\bforbidden\b", "allowed"),
    (r"\bis logged\b", "is not logged"),
    (r"\bare logged\b", "are not logged"),
    (r"\bis encrypted\b", "is not encrypted"),
    (r"\bare encrypted\b", "are not encrypted"),
    (r"\bis rotated\b", "is not rotated"),
    (r"\bare rotated\b", "are not rotated"),
    (r"\bmust\b", "need not"),
)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _short(s: str, n: int = 110) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def object_phrase(text: str, rid: str = "") -> str:
    """What a clause is about: 'the /finance share', 'any server', 'service accounts'."""
    m = _OBJ.search(text)
    if m:
        return m.group(1).strip()
    m = _SUBJ.match(text)
    if m and len(m.group(1).split()) <= 6 and ":" not in m.group(1):
        subj = m.group(1).strip()
        first = subj.split()[0]
        if not first.isupper():
            subj = subj[:1].lower() + subj[1:]
        return subj
    return f"the scope of {rid}" if rid else "this scope"


def mirror(text: str) -> str | None:
    """The clause on the opposite side (α): inbound ↔ outbound, read ↔ write, to/from swapped."""

    def sub(m: re.Match[str]) -> str:
        word = m.group(1)
        rep = _SWAP[word.lower()]
        return (_cap(rep) if word[:1].isupper() else rep) + m.group(2)

    out = _SWAP_RX.sub(sub, text)
    out = _TOFROM.sub(lambda m: f"from {m.group(1).strip()} to {m.group(2).strip()}", out)
    return out if out != text else None


def negate(text: str) -> str | None:
    """The clause bypassed (ω): its first restriction flipped. None if it restricts nothing."""
    for rx, rep in _NEGATE:
        m = re.search(rx, text, re.I)
        if m:
            word = _cap(rep) if m.start() == 0 else rep
            return text[: m.start()] + word + text[m.end() :]
    return None


def split_clause(text: str) -> tuple[str, str] | None:
    """Split a clause at its first 'and', giving each half the shared subject (→)."""
    body = " ".join(text.split()).rstrip(".")
    if " and " not in body:
        return None
    left, right = body.split(" and ", 1)
    if len(left.split()) < 3 or len(right.split()) < 2:
        return None
    first = right.split()[0].lower()
    subj = re.match(rf"^(.*?)\s+(?:{_AUX})\b", left, re.I)
    head = re.match(rf"^(.*?\b(?:{_AUX})\b)", left, re.I)
    if first in _AUX.split("|"):
        if not subj:
            return None
        second = f"{subj.group(1)} {right}"
    elif head:
        second = f"{head.group(1)} {right}"
    elif ":" in left:
        second = f"{left[: left.index(':') + 1]} {right}"
    else:
        return None
    return left.strip() + ".", second.strip() + "."


def except_for(text: str, other_id: str) -> str:
    return " ".join(text.split()).rstrip(".") + f", except where {other_id} applies."


def image(g: str, text: str, hero: str, obj: str) -> str | None:
    """The clause's g-image in the policy: mirror (a), complement (r) or norm (ra)."""
    if g == "a":
        return mirror(text)
    if g == "r":
        return FUNCTION_CLAUSE[BETA[hero]].format(obj=obj, Obj=_cap(obj))
    if g == "ra":
        return None if text.startswith("Institution-wide") else f"Institution-wide: {text}"
    return None


# --------------------------------------------------------------- lattice ---


def _locate(lat: dict[str, Any], ref: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    parts = ref.split("/", 2)
    if len(parts) != 3 or parts[0] != "pol":
        return None, None
    for lv in lat.get("levels", []) or []:
        if str(lv.get("level_id")) == parts[1]:
            for r in lv.get("rules", []) or []:
                if str(r.get("id")) == parts[2]:
                    return lv, r
    return None, None


def _rule_ids(lat: dict[str, Any]) -> set[str]:
    return {
        str(r.get("id")) for lv in lat.get("levels", []) or [] for r in lv.get("rules", []) or []
    }


def _set_text(rule: dict[str, Any], text: str) -> None:
    rule["rewrite"] = text
    rule["name"] = " ".join(text.split()[:6])
    rule["pattern"] = security.subject(text)


def _add(lat: dict[str, Any], lid: str, rid: str, text: str, provenance: str) -> str | None:
    if rid in _rule_ids(lat):
        return None
    if lid == "L1" and not any(str(lv.get("level_id")) == "L1" for lv in lat.get("levels", [])):
        lat.update(security.ensure_controls_level(lat))
    lv = next((x for x in lat.get("levels", []) or [] if str(x.get("level_id")) == lid), None)
    if lv is None:
        return None
    lv.setdefault("rules", []).append(
        {
            "id": rid,
            "name": " ".join(text.split()[:6]),
            "level": lid,
            "pattern": security.subject(text),
            "rewrite": text,
            "examples": [],
            "generative_power": 1.0,
            "provenance": provenance,
        }
    )
    return f"pol/{lid}/{rid}"


def _remove(lat: dict[str, Any], ref: str) -> None:
    lv, rule = _locate(lat, ref)
    if lv is None or rule is None:
        return
    lv["rules"] = [r for r in lv.get("rules", []) if r is not rule]
    lat["links"] = [
        ln for ln in lat.get("links", []) or [] if ref not in (ln.get("from"), ln.get("to"))
    ]


def _link(lat: dict[str, Any], a: str, b: str, kind: str) -> None:
    lat.setdefault("links", []).append({"from": a, "to": b, "kind": kind})


def _level_above(lat: dict[str, Any], lid: str) -> str | None:
    order = ["L0", "L1", "L2", "L3"]
    have = {str(lv.get("level_id")) for lv in lat.get("levels", []) or []}
    if lid not in order:
        return None
    for up in order[order.index(lid) + 1 :]:
        if up in have:
            return up
    return None


def _edit(kind: str, ref: str, before: str = "", after: str = "", **extra: Any) -> dict[str, Any]:
    return {"kind": kind, "ref": ref, "before": before, "after": after, **extra}


# ----------------------------------------------------------------- agents --


def _agents_by_ref(world: World) -> dict[str, am.Agent]:
    out: dict[str, am.Agent] = {}
    for r in world.rows:
        if not r.alive or r.ax_ref in out:
            continue
        a = world.agents.get(r.agent_id)
        if a is not None and a.alive:
            out[r.ax_ref] = a
    return out


def _agent(world: World, ref: str) -> am.Agent | None:
    return _agents_by_ref(world).get(ref)


def _rank(a: am.Agent) -> int:
    return RANKS.index(a.rank) if a.rank in RANKS else 1


def _weakest(a: am.Agent, p: am.Params) -> int:
    """The weakest slot the agent values (online frame)."""
    weff = a.w_eff()
    valued = [i for i in range(8) if weff[i] > p.theta] or sorted(range(8), key=lambda i: -weff[i])[
        :2
    ]
    return min(valued, key=lambda i: (weff[i], i))


def _stem_partner(
    world: World, a: am.Agent, by_ref: dict[str, am.Agent]
) -> tuple[str, am.Agent, int] | None:
    weak = _weakest(a, world.params)
    adj = {d for (s, d) in world.am_edges if s == a.id} | {
        s for (s, d) in world.am_edges if d == a.id
    }
    best: tuple[float, str, am.Agent] | None = None
    for ref, b in sorted(by_ref.items()):
        if b.id == a.id or b.id in adj or not ref.startswith("pol/") or "#mod" in ref:
            continue
        h = b.hero()
        if world.R[IDX[h]][weak] <= 0:
            continue
        score = b.w_eff()[IDX[h]]
        if best is None or score > best[0]:
            best = (score, ref, b)
    return (best[1], best[2], weak) if best else None


def _residual(world: World, x: am.Agent, y: am.Agent) -> tuple[list[float], float, float]:
    """What a linked pair emits to each other and cannot store, and each side's flux."""
    p = world.params
    ex, ey = am.offer(x, world.R, p), am.offer(y, world.R, p)
    cxy = math.cos(world.am_edges.get((x.id, y.id), 0.0))
    cyx = math.cos(world.am_edges.get((y.id, x.id), 0.0))
    rx = am.arrivals(x.w_eff(), [(y.id, [o * cyx for o in ey])], p)
    ry = am.arrivals(y.w_eff(), [(x.id, [o * cxy for o in ex])], p)
    res = [
        rx["ov"][g] + rx["q"][g] + rx["op"][g] + ry["ov"][g] + ry["q"][g] + ry["op"][g]
        for g in range(8)
    ]
    fxy = sum(max(0.0, ex[g] * cxy) * y.w_eff()[g] for g in range(8))
    fyx = sum(max(0.0, ey[g] * cyx) * x.w_eff()[g] for g in range(8))
    return res, fxy, fyx


def klein_close(child: tuple[str, ...], parent: tuple[str, ...]) -> bool:
    """A child is Klein-close when its writing is g·M of the parent's for some g ∈ K."""
    return bool(child) and any(tuple(klein_stack(g, parent)) == tuple(child) for g in SIDES)


def _sync(
    world: World, lat: dict[str, Any], facts: list[dict[str, Any]], mutators: int
) -> dict[str, tc.AXNode]:
    ops = viable_operations(lat, mutators)
    nodes, edges = tc.lattice_ax(lat, ops)
    if facts:
        s_nodes, s_edges = surface.ax_nodes(facts, surface.link(facts, lat), lat)
        nodes += s_nodes
        edges += s_edges
    world.sync(nodes, edges, {}, "evolve-lexicon")
    return {n.ref: n for n in nodes}


def _scale(a: am.Agent, mass: float) -> None:
    tot = sum(a.w) or 1.0
    a.w = [v * mass / tot for v in a.w]


def _relink(world: World, ref: str, a: am.Agent) -> None:
    for r in world.rows:
        if r.alive and r.ax_ref == ref:
            r.A = tc.link_phase(r.p, a.w_eff())


# -------------------------------------------------------------- mutations --


def candidates(world: World, lat: dict[str, Any]) -> list[tuple[float, dict[str, Any]]]:
    """Every licensed mutation of this world, weighted by what the lattice is doing."""
    p = world.params
    by_ref = _agents_by_ref(world)
    live = world.live()
    out: list[tuple[float, dict[str, Any]]] = []
    for lid, rid, rule in security.clauses_of(lat):
        if lid == "L0":
            continue
        ref = f"pol/{lid}/{rid}"
        a = by_ref.get(ref)
        if a is None:
            continue
        text = str(rule.get("rewrite", ""))
        paid = a.x >= 1.0
        repair = 1.0 + 2.0 * a.L
        grow = 1.0 + (1.0 if a.state == "stock" else 0.0) + min(1.0, max(0.0, a.dw_hat))
        hero = a.hero()
        obj = object_phrase(text, rid)
        if paid and a.side != "a" and mirror(text):
            out.append((1.0 * repair, {"op": "ev", "g": "a", "ref": ref}))
        if (
            paid
            and a.side != "r"
            and any(
                b.hero() == BETA[hero] and b.S >= p.s_min for b in live.values() if b.id != a.id
            )
        ):
            out.append((1.0 * repair, {"op": "ev", "g": "r", "ref": ref}))
        if (
            paid
            and a.side != "ra"
            and a.state == "stock"
            and lid != "L3"  # a principle is already a norm
            and not text.startswith("Institution-wide")
        ):
            out.append((0.8 * grow, {"op": "ev", "g": "ra", "ref": ref}))
        if paid:
            partner = _stem_partner(world, a, by_ref)
            if partner:
                out.append((1.0 * grow, {"op": "sigma", "ref": ref, "other": partner[0]}))
        ring_ids = {f"{rid}·{tag}" for tag, _, _ in RING}
        if (
            paid
            and _rank(a) >= RANKS.index("Res")
            and lid in ("L2", "L1")
            and not ring_ids & _rule_ids(lat)
        ):
            out.append((1.2 * grow, {"op": "store4", "ref": ref}))
        if paid and _rank(a) >= RANKS.index("Cyc"):
            for g in ("a", "r", "ra"):
                if g == "ra" and not _level_above(lat, lid):
                    continue  # nowhere to publish the norm form
                if f"{rid}·{g}" not in _rule_ids(lat) and image(g, text, hero, obj):
                    out.append((0.6 * grow, {"op": "hairpin", "g": g, "ref": ref}))
        if lid in ("L2", "L1") and split_clause(text) and (a.support(p) >= 3 or sum(a.leak_ov) > 0):
            out.append((0.8 * repair, {"op": "arrow", "ref": ref}))
        out.append((0.4 * repair, {"op": "dissolve", "ref": ref}))
    for ra_ref, rb_ref, _c in security.contradictions(lat):
        for me, other in ((ra_ref, rb_ref), (rb_ref, ra_ref)):
            if me in by_ref and other in by_ref:
                out.append((1.5 * (1.0 + by_ref[me].L), {"op": "cut", "ref": me, "other": other}))
    seen: set[tuple[str, str]] = set()
    for s, d in sorted(world.am_edges):
        pair = (min(s, d), max(s, d))
        if pair in seen:
            continue
        seen.add(pair)
        xs = [ref for ref, a in by_ref.items() if a.id == pair[0] and ref.startswith("pol/")]
        ys = [ref for ref, a in by_ref.items() if a.id == pair[1] and ref.startswith("pol/")]
        if not xs or not ys or "#mod" in xs[0] + ys[0]:
            continue
        x, y = by_ref[xs[0]], by_ref[ys[0]]
        res, fxy, fyx = _residual(world, x, y)
        winner = x if (fxy, x.id) >= (fyx, y.id) else y
        if sum(res) >= BIRTH_MIN and winner.x >= 1.0:
            a_ref, b_ref = sorted((xs[0], ys[0]))
            out.append((1.5 * (1.0 + sum(res)), {"op": "birth", "ref": a_ref, "other": b_ref}))
    out.sort(
        key=lambda t: (t[1]["op"], t[1]["ref"], str(t[1].get("g", "")), str(t[1].get("other", "")))
    )
    return out


def probe_candidates(world: World, lat: dict[str, Any]) -> list[tuple[float, dict[str, Any]]]:
    """ω probes: every restriction that can be bypassed."""
    by_ref = _agents_by_ref(world)
    out = []
    for lid, rid, rule in security.clauses_of(lat):
        ref = f"pol/{lid}/{rid}"
        text = str(rule.get("rewrite", ""))
        if lid in ("L2", "L1") and ref in by_ref and surface.mandates(text) and negate(text):
            out.append(
                (1.0 + world.clash_cache.get(by_ref[ref].id, 0.0), {"op": "omega", "ref": ref})
            )
    return out


def spec_key(spec: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(spec.get("op")),
        str(spec.get("ref")),
        str(spec.get("g", "")),
        str(spec.get("other", "")),
    )


def sample(
    cands: list[tuple[float, dict[str, Any]]], n: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Weighted sampling without replacement, at most a third of the picks per operator."""
    if n <= 0 or not cands:
        return []
    keyed = sorted(
        ((rng.random() ** (1.0 / max(w, 1e-6)), i) for i, (w, _) in enumerate(cands)), reverse=True
    )
    cap = max(2, math.ceil(n / 3))
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    per_op: Counter[str] = Counter()
    for _, i in keyed:
        spec = cands[i][1]
        key = spec_key(spec)
        if key in seen or per_op[spec["op"]] >= cap:
            continue
        out.append(dict(spec))
        seen.add(key)
        per_op[spec["op"]] += 1
        if len(out) >= n:
            break
    return out


def apply(
    world: World,
    lat: dict[str, Any],
    spec: dict[str, Any],
    facts: list[dict[str, Any]] | None = None,
    mutators: int = 8,
) -> dict[str, Any] | None:
    """Apply one mutation in both spaces. Returns its record, or None if it does not apply.

    ``lat`` is edited in place only when the mutation applies.
    """
    op = str(spec.get("op", ""))
    ref = str(spec.get("ref", ""))
    work = copy.deepcopy(lat)
    lv, rule = _locate(work, ref)
    ag = _agent(world, ref)
    if lv is None or rule is None or ag is None or op not in LAW:
        return None
    p = world.params
    lid, rid = str(lv.get("level_id")), str(rule.get("id"))
    text = str(rule.get("rewrite", ""))
    obj = object_phrase(text, rid)
    hero = ag.hero()
    edits: list[dict[str, Any]] = []
    changed: list[str] = []
    pending: dict[str, str] = {}
    klein: dict[str, str] = {}
    what = ""
    weakens = False
    after: Callable[[World], tuple[list[str], str]] = lambda w: ([], "")  # noqa: E731
    g = str(spec.get("g") or "")
    other = str(spec.get("other") or "")

    if op == "ev":
        new = image(g, text, hero, obj)
        if not new or new == text:
            return None
        _set_text(rule, new)
        changed.append(ref)
        edits.append(_edit("replace", ref, text, new, level=lid))
        what = {
            "a": f"What if {rid} were mirrored to the opposite side: “{new}”?",
            "r": f"What if {rid} served its complement ({security.FUNCTION[BETA[hero]]}) instead: “{new}”?",
            "ra": f"What if {rid} were promoted to an institution-wide rule: “{new}”?",
        }[g]

        def after(w: World) -> tuple[list[str], str]:
            a = _agent(w, ref)
            if a is None:
                return [], ""
            before = a.hero()
            cost = p.c * sum(a.w) * (1.0 + a.m_side.get(g, 0.0))
            a.pi = {g: 1.0}
            w.spend(a, cost)
            _relink(w, ref, a)
            return [], (
                f"{a.id} is rewritten on side {g} (ev): hero {before} → {a.hero()}; the identifier is kept, "
                f"and the step costs energy {cost:.2f}, more with side mass"
            )

    elif op == "sigma":
        olv, orule = _locate(work, other)
        oag = _agent(world, other)
        if orule is None or oag is None or other == ref:
            return None
        oid = str(orule.get("id"))
        weak = SPECIES[_weakest(ag, p)]
        _link(work, ref, other, "stem")
        edits.append(
            _edit("link", ref, "", f"{rid} is backed by {oid}", other=other, kind_of="stem")
        )
        what = (
            f"What if {rid} were explicitly backed by {oid} (“{_short(str(orule.get('rewrite', '')))}”), "
            f"whose {security.FUNCTION[oag.hero()]} covers what {rid} holds least ({security.FUNCTION[weak]})?"
        )

        def after(w: World) -> tuple[list[str], str]:
            a, b = _agent(w, ref), _agent(w, other)
            if a is None or b is None:
                return [], ""
            w.stems.add((a.id, b.id))
            w.stems.add((b.id, a.id))
            return (
                [],
                f"{a.id} opens a storing stem with {b.id} (σ): {b.id}'s hero {b.hero()} covers {a.id}'s weakest slot {weak}",
            )

    elif op == "store4":
        if lid not in ("L2", "L1"):
            return None
        kids: list[str] = []
        texts: list[str] = []
        for tag, _sp, tmpl in RING:
            t = tmpl.format(obj=obj, Obj=_cap(obj))
            cref = _add(work, "L1", f"{rid}·{tag}", t, f"evolve:store4:{rid}")
            if cref is None:
                return None
            kids.append(cref)
            texts.append(t)
            pending[cref] = ag.id
            edits.append(_edit("add", cref, "", t, level="L1"))
        ring = [ref, *kids, ref]
        for s, d in zip(ring, ring[1:]):
            _link(work, s, d, "ring")
        edits.append(
            _edit("link", ref, "", " → ".join(x.split("/")[-1] for x in ring), kind_of="ring")
        )
        what = (
            f"What if {rid}'s loop were closed with three controls (log, alert, review): "
            + " ".join(f"“{t}”" for t in texts)
            + "?"
        )

        def after(w: World) -> tuple[list[str], str]:
            parent = _agent(w, ref)
            babies = [_agent(w, k) for k in kids]
            if parent is None or any(b is None for b in babies):
                return [], ""
            members = [b for b in babies if b is not None]
            each = 0.35 * sum(parent.w)
            for b in members:
                _scale(b, each)
            w.spend(parent, p.c * each * len(members))
            chain = [parent, *members, parent]
            for s_, d_ in zip(chain, chain[1:]):
                w.stems.add((s_.id, d_.id))
            ids = [b.id for b in members]
            return ids, (
                f"{parent.id} ({parent.rank}) pays for {', '.join(ids)} and closes the four-cycle "
                f"{' → '.join(c.id for c in chain)} (II→III: store4 if every arrow stores)"
            )

    elif op == "hairpin":
        img = image(g, text, hero, obj)
        if not img or img == text:
            return None
        target = (_level_above(work, lid) or lid) if g == "ra" else lid
        cref = _add(work, target, f"{rid}·{g}", img, f"evolve:hairpin:{rid}")
        if cref is None:
            return None
        _link(work, ref, cref, "standard")
        pending[cref] = ag.id
        edits.append(_edit("add", cref, "", img, level=target))
        edits.append(
            _edit("link", ref, "", f"{rid} and {rid}·{g} form one standard", kind_of="standard")
        )
        what = (
            f"What if {rid} and its {KLEIN_NAME[g]} were one standard, adding {rid}·{g}: “{img}”?"
        )

        def after(w: World) -> tuple[list[str], str]:
            parent, child = _agent(w, ref), _agent(w, cref)
            if parent is None or child is None:
                return [], ""
            mass = 0.5 * sum(parent.w)
            child.w = [parent.w[IDX[act(g, f)]] * 0.5 for f in SPECIES]
            child.stack = klein_stack(g, parent.stack) if parent.stack else (act(g, parent.hero()),)
            child.pi = {"e": 1.0}
            w.spend(parent, p.c * mass)
            _relink(w, cref, child)
            klein[child.id] = parent.id
            return [child.id], (
                f"{parent.id} ({parent.rank}) folds onto its {g}-writing (hairpin η_{g}): child {child.id} "
                f"with stack {'/'.join(child.stack)}, Klein-close to its parent, paid with energy {p.c * mass:.2f}"
            )

    elif op == "birth":
        olv, orule = _locate(work, other)
        y = _agent(world, other)
        if orule is None or y is None or other == ref:
            return None
        res, fxy, fyx = _residual(world, ag, y)
        total = sum(res)
        if total < BIRTH_MIN:
            return None
        oid = str(orule.get("id"))
        win, win_ref, lose_ref = (
            (ag, ref, other) if (fxy, ag.id) >= (fyx, y.id) else (y, other, ref)
        )
        w_child = [res[IDX[BETA[f]]] for f in SPECIES]  # r of the residual
        hw = win.hero()
        g_best = max(SIDES, key=lambda s: (w_child[IDX[act(s, hw)]], -SIDES.index(s)))
        h_child = act(g_best, hw)
        win_text = str((_locate(work, win_ref)[1] or {}).get("rewrite", ""))
        wobj = object_phrase(win_text, win_ref.split("/")[-1])
        t = FUNCTION_CLAUSE[h_child].format(obj=wobj, Obj=_cap(wobj))
        cref = _add(work, "L1", f"{rid}×{oid.lstrip('§')}", t, f"evolve:birth:{rid}+{oid}")
        if cref is None:
            return None
        _link(work, ref, cref, "birth")
        _link(work, other, cref, "birth")
        pending[cref] = win.id
        edits.append(_edit("add", cref, "", t, level="L1"))
        top = SPECIES[max(range(8), key=lambda i: res[i])]
        what = (
            f"What if {rid} and {oid} had a child clause for what they lose between them "
            f"({security.FUNCTION[h_child]}): “{t}”?"
        )
        win_stack = tuple(win.stack)

        def after(w: World) -> tuple[list[str], str]:
            child, wa, la = _agent(w, cref), _agent(w, win_ref), _agent(w, lose_ref)
            if child is None or wa is None or la is None:
                return [], ""
            mass = min(w.mass, max(0.6, 4.0 * total))
            tot = sum(w_child) or 1.0
            child.w = [v * mass / tot for v in w_child]
            child.stack = klein_stack(g_best, win_stack) if win_stack else (h_child,)
            child.pi = {"e": 1.0}
            w.pr_edges.append((la.id, child.id))
            w.spend(wa, p.c * mass)
            _relink(w, cref, child)
            klein[child.id] = wa.id
            return [child.id], (
                f"birth a·(α{ag.id}, β{y.id}): child {child.id} stores r of the pair's residual "
                f"({total:.2f}, mostly {top}) with stack {'/'.join(child.stack)}, Klein-close to the winner {wa.id}"
            )

    elif op == "arrow":
        if lid not in ("L2", "L1"):
            return None
        parts = split_clause(text)
        if not parts:
            return None
        _remove(work, ref)
        c1 = _add(work, lid, f"{rid}.1", parts[0], f"evolve:arrow:{rid}")
        c2 = _add(work, lid, f"{rid}.2", parts[1], f"evolve:arrow:{rid}")
        if not c1 or not c2:
            return None
        pending[c1] = ag.id
        pending[c2] = ag.id
        edits += [
            _edit("remove", ref, text, "", level=lid),
            _edit("add", c1, "", parts[0], level=lid),
            _edit("add", c2, "", parts[1], level=lid),
        ]
        what = f"What if {rid} were split in two: {rid}.1 “{parts[0]}” and {rid}.2 “{parts[1]}”?"
        pmass, ppi, pms = sum(ag.w), dict(ag.pi), dict(ag.m_side)
        pen = {s: list(v) for s, v in ag.energy.items()}
        pid = ag.id

        def after(w: World) -> tuple[list[str], str]:
            kids = [_agent(w, c) for c in (c1, c2)]
            if any(k is None for k in kids):
                return [], ""
            out = []
            for k in kids:
                if k is None:
                    continue
                _scale(k, 0.5 * pmass)
                k.pi = dict(ppi)
                k.m_side = dict(pms)
                k.energy = {s: [e * 0.5 for e in v] for s, v in pen.items()}
                out.append(k.id)
            return (
                out,
                f"{pid} dissolves; {out[0]} and {out[1]} share its mass ({pmass:.2f}) and its writing (→, restriction)",
            )

    elif op == "cut":
        olv, orule = _locate(work, other)
        oag = _agent(world, other)
        if orule is None or oag is None or other == ref:
            return None
        oid = str(orule.get("id"))
        new = except_for(text, oid)
        _set_text(rule, new)
        changed.append(ref)
        edits.append(_edit("replace", ref, text, new, level=lid))
        weakens = surface.mandates(text) and not surface.mandates(str(orule.get("rewrite", "")))
        what = f"What if {rid} yielded to {oid}: “{new}”?"

        def after(w: World) -> tuple[list[str], str]:
            a, b = _agent(w, ref), _agent(w, other)
            if a is None or b is None:
                return [], ""
            w.cut.add((b.id, a.id))
            return (
                [],
                f"{a.id} is shielded from {b.id}: the edge {b.id} → {a.id} is cut (w_border → 0)",
            )

    elif op == "dissolve":
        _remove(work, ref)
        edits.append(_edit("remove", ref, text, "", level=lid))
        weakens = surface.mandates(text)
        what = f"What if {rid} were removed: “{_short(text)}”?"
        gone = ag.id

        def after(w: World) -> tuple[list[str], str]:
            return [], f"{gone} dissolves (†): a tombstone !x whose id is not reused"

    elif op == "omega":
        new = negate(text)
        if not new:
            return None
        _set_text(rule, new)
        changed.append(ref)
        edits.append(_edit("replace", ref, text, new, level=lid))
        weakens = True
        what = f"What if {rid} were reversed (bypassed): “{new}”?"

        def after(w: World) -> tuple[list[str], str]:
            a = _agent(w, ref)
            name = a.id if a else ref
            return (
                [],
                f"{name} keeps its writing while its clause says the opposite: off-license emission (ω), recorded as heat",
            )

    lat.clear()
    lat.update(work)
    for cref, pid in pending.items():
        world.pending_parent[cref] = pid
    nodes = _sync(world, lat, list(facts or []), mutators)
    for r in changed:
        node = nodes.get(r)
        if node is not None:
            world.reproject(node, "evolve-lexicon")
    born, change = after(world)
    world._rebuild_edges()
    return {
        "op": op,
        "g": g,
        "spec": {k: v for k, v in spec.items() if k != "root"},
        "symbol": {
            "ev": f"ev_{g}",
            "hairpin": f"η_{g}",
            "sigma": "σ",
            "store4": "→×4",
            "birth": "a·(αX,βY)",
            "arrow": "→",
            "cut": "cut",
            "dissolve": "†",
            "omega": "ω",
        }[op],
        "law": LAW[op],
        "site": ref,
        "agent": ag.id,
        "rank": ag.rank,
        "what_if": what,
        "agent_change": change,
        "edits": edits,
        "born": born,
        "new_refs": [e["ref"] for e in edits if e["kind"] == "add"],
        "root": str(spec.get("root") or ref),
        "klein": dict(klein),
        "weakens": bool(weakens),
        "probe": op == "omega",
    }


# ---------------------------------------------------------------- measure --


def measure(world: World, lat: dict[str, Any]) -> dict[str, Any]:
    pos = security.posture(lat)
    pairs = sorted({(a, b) for a, b, _c in security.contradictions(lat)})
    faces: dict[tuple[str, str], tuple[float, float]] = {}
    for f in world.clashes():
        key = (min(f.a1, f.a2), max(f.a1, f.a2))
        if key not in faces or f.weight > faces[key][1]:
            faces[key] = (round(f.f, 3), round(f.weight, 3))
    live = world.live()
    return {
        "posture": pos["score"],
        "diversity": pos["diversity"],
        "missing": list(pos["missing"]),
        "contradictions": len(pairs),
        "pairs": pairs,
        "L_G": round(world.ensemble_leak(), 4),
        "leaking": sum(1 for a in live.values() if a.state == "leak"),
        "clash_mass": round(sum(f.weight for f in world.clashes()), 3),
        "faces": faces,
        "agents": len(live),
        "ranks": dict(Counter(a.rank for a in live.values())),
        "space": world.coverage_history[-1]["space"] if world.coverage_history else 0,
    }


def _summary(m: dict[str, Any]) -> dict[str, Any]:
    return {
        k: m[k]
        for k in (
            "posture",
            "contradictions",
            "L_G",
            "leaking",
            "clash_mass",
            "agents",
            "ranks",
            "space",
        )
    }


def _tail(ref: str) -> str:
    return ref.split("/")[-1]


def _universe(
    uid: str,
    loop: int,
    gen: int,
    parents: list[str],
    world: World,
    lat: dict[str, Any],
    records: list[dict[str, Any]],
    base: dict[str, Any],
    lam: float,
) -> dict[str, Any]:
    m = measure(world, lat)
    d = {
        "posture": round(m["posture"] - base["posture"], 4),
        "contradictions": m["contradictions"] - base["contradictions"],
        "L_G": round(m["L_G"] - base["L_G"], 4),
        "leaking": m["leaking"] - base["leaking"],
        "clash_mass": round(m["clash_mass"] - base["clash_mass"], 3),
        "space": m["space"] - base["space"],
        "agents": m["agents"] - base["agents"],
    }
    bp, mp = set(map(tuple, base["pairs"])), set(map(tuple, m["pairs"]))
    cons: list[str] = []
    if abs(d["posture"]) > 1e-4:
        cons.append(f"posture {base['posture']:.3f} → {m['posture']:.3f}")
    cons += [f"dissolves the contradiction {_tail(a)} × {_tail(b)}" for a, b in sorted(bp - mp)]
    cons += [f"adds the contradiction {_tail(a)} × {_tail(b)}" for a, b in sorted(mp - bp)]
    cons += [f"covers {security.FUNCTION[s]}" for s in base["missing"] if s not in m["missing"]]
    cons += [
        f"leaves {security.FUNCTION[s]} uncovered" for s in m["missing"] if s not in base["missing"]
    ]
    if any(r["weakens"] for r in records):
        cons.append("removes or weakens a restriction")
    new_faces = sorted(set(m["faces"]) - set(base["faces"]))
    gone_faces = sorted(set(base["faces"]) - set(m["faces"]))
    cons += [
        f"new clash in the lattice: {_tail(a)} × {_tail(b)} (f={m['faces'][(a, b)][0]:.2f}"
        + (")" if m["faces"][(a, b)][1] >= m["faces"][(a, b)][0] else ", a mapping mismatch)")
        for a, b in new_faces[:4]
    ]
    cons += [f"the clash {_tail(a)} × {_tail(b)} settles in the lattice" for a, b in gone_faces[:4]]
    if abs(d["L_G"]) >= 0.005:
        cons.append(f"ensemble leak {base['L_G']:.3f} → {m['L_G']:.3f}")
    born = [b for r in records for b in r["born"]]
    klein = {c: par for r in records for c, par in r["klein"].items()}
    structures = []
    for bid in born:
        a = world.agents.get(bid)
        if a is None:
            continue
        rows = world.rows_of(bid)
        par = klein.get(bid)
        structures.append(
            {
                "agent": bid,
                "ref": rows[0].ax_ref if rows else "",
                "name": agent_name(a, world.params) if a.alive else "!x",
                "rank": a.rank,
                "alive": a.alive,
                "parents": sorted({s for s, dd in world.pr_edges if dd == bid}),
                "klein_close": (
                    klein_close(tuple(a.stack), tuple(world.agents[par].stack))
                    if par and par in world.agents
                    else None
                ),
            }
        )
    sites = {r["agent"] for r in records}
    rings = [
        {"members": list(c), "signature": s}
        for c, s in world.cycles.items()
        if len(c) >= 3 and set(c) & (set(born) | sites)
    ]
    store_rings = sum(
        1 for r in rings if r["signature"] == "store" and set(r["members"]) & set(born)
    )
    # one point for any storing ring through a bred agent, plus each bred agent's rank (0..1)
    novelty = (1.0 if store_rings else 0.0) + sum(
        RANKS.index(s["rank"]) / 4 for s in structures if s["alive"] and s["rank"] in RANKS
    )
    probe = any(r["probe"] for r in records)
    heat = (
        max(0.0, d["L_G"]) * 5
        + max(0.0, d["clash_mass"]) * 0.5
        + max(0, d["leaking"]) * 0.2
        + max(0, d["contradictions"])
    )
    # A leak the policy text does not show: a strong new clash, a new leaking agent, or more L_G.
    lattice_leak = (
        any(m["faces"][k][1] >= 0.45 for k in new_faces) or d["leaking"] > 0 or d["L_G"] > 0.02
    )
    bred = store_rings > 0 or any(
        s["alive"] and s["rank"] in RANKS and RANKS.index(s["rank"]) >= 2 for s in structures
    )
    if probe:
        verdict = "load-bearing" if heat >= 0.25 else "slack"
    elif mp - bp or d["contradictions"] > 0:
        verdict = "breaks"
    elif d["contradictions"] < 0:
        verdict = "dissolves"
    elif lattice_leak:
        verdict = "leaks"
    elif bred:
        verdict = "breeds"
    elif d["posture"] > 1e-4:
        verdict = "improves"
    else:
        verdict = "neutral"
    art = 10.0 * d["posture"] - 1.0 * d["contradictions"]
    agent = -5.0 * d["L_G"] - 0.5 * d["clash_mass"] + 0.5 * novelty
    fitness = (1 - lam) * art + lam * agent - 0.05 * (len(records) - 1)
    diff = [e for r in records for e in r["edits"]]
    return {
        "id": uid,
        "loop": loop,
        "generation": gen,
        "parents": parents,
        "probe": probe,
        "what_if": " ".join(r["what_if"] for r in records),
        "mutations": [
            {
                k: r[k]
                for k in (
                    "op",
                    "g",
                    "symbol",
                    "law",
                    "site",
                    "agent",
                    "rank",
                    "what_if",
                    "agent_change",
                    "weakens",
                )
            }
            for r in records
        ],
        "diff": diff,
        "consequences": cons,
        "structures": structures,
        "rings": rings[:6],
        "metrics": _summary(m),
        "lattice_leak": bool(lattice_leak),
        "delta": d,
        "heat": round(heat, 3),
        "verdict": verdict,
        "fitness": round(fitness, 4),
        # Only additions, splits, links and explicit precedences can be adopted. Rewrites (ev),
        # removals (†), reversals (ω) and anything that weakens a restriction stay what-ifs.
        "adoptable": not probe
        and not any(r["weakens"] or r["op"] in ("ev", "dissolve", "omega") for r in records),
    }


def evolve(
    world: World,
    lattice: dict[str, Any],
    facts: list[dict[str, Any]] | None,
    cfg: EvolveConfig,
    *,
    loop: int = 1,
    seed: int = 0,
    mutators: int = 8,
) -> dict[str, Any]:
    """Branch the world into alternate universes; the real world and lattice are not touched."""
    rng = random.Random(seed)
    facts = list(facts or [])
    k_ticks = cfg.ticks
    lam = max(0.0, min(1.0, world.lambda_am))
    base = world.fork()
    base_lat = copy.deepcopy(dict(lattice))
    _sync(base, base_lat, facts, mutators)
    baseline = base.fork()
    baseline.run_ticks(k_ticks)
    bm = measure(baseline, base_lat)
    universes: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    lattices: dict[str, dict[str, Any]] = {}
    roots: dict[str, dict[str, str]] = {}
    alive: dict[str, tuple[World, dict[str, Any], list[dict[str, Any]]]] = {}
    survivors_log: dict[str, list[str]] = {}

    def keep(
        uid: str,
        w: World,
        lat: dict[str, Any],
        recs: list[dict[str, Any]],
        parent_roots: dict[str, str],
    ) -> None:
        rr = dict(parent_roots)
        for r in recs[-1:]:
            for nr in r["new_refs"]:
                rr[nr] = r["root"]
        roots[uid] = rr
        lattices[uid] = lat
        alive[uid] = (w, lat, recs)

    gen_us: list[dict[str, Any]] = []
    for k, spec in enumerate(sample(candidates(base, base_lat), cfg.universes, rng), 1):
        w, lat = base.fork(), copy.deepcopy(base_lat)
        rec = apply(w, lat, spec, facts, mutators)
        if rec is None:
            continue
        w.run_ticks(k_ticks)
        uid = f"U{loop}-1.{k}"
        u = _universe(uid, loop, 1, ["U0"], w, lat, [rec], bm, lam)
        gen_us.append(u)
        keep(uid, w, lat, [rec], {})
    for k, spec in enumerate(sample(probe_candidates(base, base_lat), cfg.probes, rng), 1):
        w, lat = base.fork(), copy.deepcopy(base_lat)
        rec = apply(w, lat, spec, facts, mutators)
        if rec is None:
            continue
        w.run_ticks(k_ticks)
        probes.append(_universe(f"Ω{loop}.{k}", loop, 1, ["U0"], w, lat, [rec], bm, lam))
    universes += gen_us
    survivors = [
        u["id"] for u in sorted(gen_us, key=lambda u: (-u["fitness"], u["id"]))[: cfg.keep]
    ]
    survivors_log["1"] = survivors
    for gen in range(2, cfg.generations + 1):
        if not survivors:
            break
        baseline = baseline.fork()
        baseline.run_ticks(k_ticks)
        bm_g = measure(baseline, base_lat)
        new_alive: dict[str, tuple[World, dict[str, Any], list[dict[str, Any]]]] = {}
        gen_us = []
        per = max(1, round(cfg.universes / len(survivors)))
        n = 0
        for sid in survivors:
            sw, slat, srecs = alive[sid]
            used = {spec_key(r["spec"]) for r in srecs}
            pool = [(wt, s) for wt, s in candidates(sw, slat) if spec_key(s) not in used]
            for spec in sample(pool, per, rng):
                spec["root"] = roots[sid].get(spec["ref"], spec["ref"])
                w, lat = sw.fork(), copy.deepcopy(slat)
                rec = apply(w, lat, spec, facts, mutators)
                if rec is None:
                    continue
                w.run_ticks(k_ticks)
                n += 1
                uid = f"U{loop}-{gen}.{n}"
                recs = [*srecs, rec]
                gen_us.append(_universe(uid, loop, gen, [sid], w, lat, recs, bm_g, lam))
                keep(uid, w, lat, recs, roots[sid])
                new_alive[uid] = alive.pop(uid)
        # crossover: the union of two survivors' edits, replayed from the base
        pair = None
        for i, a_id in enumerate(survivors):
            for b_id in survivors[i + 1 :]:
                sa = {r["site"] for r in alive[a_id][2]} | {
                    e["ref"] for r in alive[a_id][2] for e in r["edits"]
                }
                sb = {r["site"] for r in alive[b_id][2]} | {
                    e["ref"] for r in alive[b_id][2] for e in r["edits"]
                }
                if not sa & sb:
                    pair = (a_id, b_id)
                    break
            if pair:
                break
        if pair:
            w, lat = base.fork(), copy.deepcopy(base_lat)
            recs = []
            for r in alive[pair[0]][2] + alive[pair[1]][2]:
                rec = apply(w, lat, {**r["spec"], "root": r["root"]}, facts, mutators)
                if rec is not None:
                    recs.append(rec)
            if recs:
                w.run_ticks(k_ticks * gen)
                n += 1
                uid = f"U{loop}-{gen}.{n}"
                u = _universe(uid, loop, gen, list(pair), w, lat, recs, bm_g, lam)
                u["crossover"] = True
                gen_us.append(u)
                rr = {**roots[pair[0]], **roots[pair[1]]}
                for r in recs:
                    for nr in r["new_refs"]:
                        rr.setdefault(nr, r["root"])
                roots[uid] = rr
                lattices[uid] = lat
                new_alive[uid] = (w, lat, recs)
        universes += gen_us
        alive = new_alive
        survivors = [
            u["id"] for u in sorted(gen_us, key=lambda u: (-u["fitness"], u["id"]))[: cfg.keep]
        ]
        survivors_log[str(gen)] = survivors
    ranked = sorted(universes, key=lambda u: (-u["fitness"], u["id"]))
    gate = [
        u
        for u in ranked
        if u["adoptable"]
        and u["verdict"] in ("dissolves", "improves", "breeds")
        and u["delta"]["posture"] > 1e-6
        and u["delta"]["contradictions"] <= 0
    ]
    candidate = None
    if gate:
        best = gate[0]
        candidate = {
            "id": best["id"],
            "lattice": lattices[best["id"]],
            "roots": roots.get(best["id"], {}),
            "replaced": [e["ref"] for e in best["diff"] if e["kind"] == "replace"],
        }
    return {
        "loop": loop,
        "seed": seed,
        "config": cfg.as_dict(),
        "baseline": _summary(bm),
        "universes": ranked,
        "probes": sorted(probes, key=lambda u: (-u["heat"], u["id"])),
        "survivors": survivors_log,
        "best": ranked[0]["id"] if ranked else None,
        "candidate": candidate,
    }


# ----------------------------------------------------------------- output --

_EDIT_MARK = {"replace": "~", "add": "+", "remove": "−", "link": "↔"}


def diff_line(e: dict[str, Any]) -> str:
    """One artifact edit as a readable line."""
    rid = _tail(str(e.get("ref", "")))
    if e["kind"] == "replace":
        return f"~ {rid}: “{_short(e['before'], 90)}” → “{_short(e['after'], 140)}”"
    if e["kind"] == "add":
        return f"+ {rid} ({e.get('level', '')}): “{_short(e['after'], 140)}”"
    if e["kind"] == "remove":
        return f"− {rid}: “{_short(e['before'], 120)}”"
    return f"↔ {e.get('after', '')}"


def compact(history: list[dict[str, Any]], limit: int = 14) -> dict[str, Any]:
    """The last loop's universes, trimmed for the live view."""
    if not history:
        return {}
    last = history[-1]

    def trim(u: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": u["id"],
            "generation": u["generation"],
            "parents": u["parents"],
            "what_if": _short(u["what_if"], 600),
            "verdict": u["verdict"],
            "fitness": u["fitness"],
            "heat": u.get("heat", 0.0),
            "delta": u["delta"],
            "diff": [diff_line(e) for e in u["diff"][:8]],
            "consequences": u["consequences"][:6],
            "structures": [
                {k: s[k] for k in ("agent", "name", "rank", "klein_close")}
                for s in u["structures"][:4]
            ],
            "mutations": [
                {k: m[k] for k in ("symbol", "site", "agent_change")} for m in u["mutations"]
            ],
            "probe": u["probe"],
            "adoptable": u.get("adoptable", False),
            "crossover": bool(u.get("crossover")),
        }

    return {
        "loop": last.get("loop"),
        "seed": last.get("seed"),
        "skipped": last.get("skipped", ""),
        "adopted": last.get("adopted"),
        "baseline": last.get("baseline", {}),
        "survivors": last.get("survivors", {}),
        "universes": [trim(u) for u in last.get("universes", [])[:limit]],
        "probes": [trim(u) for u in last.get("probes", [])[:6]],
    }


def report_lines(history: list[dict[str, Any]], detail: int = 5) -> list[str]:
    """Markdown for the hardened-policy report."""
    out = [
        "## Alternate universes (active evolution)",
        "",
        "Each universe is the policy and its lattice with a few mutations bred by the treatise's "
        "operators. It is hypothetical: the policy changes only when adoption is on and the gate "
        "passes. Every universe reads as a what-if, with its edits and what they do in both spaces.",
        "",
    ]
    for rec in history:
        loop = rec.get("loop")
        if rec.get("skipped"):
            out += [f"### Loop {loop}: skipped ({rec['skipped']}; trigger is calm)", ""]
            continue
        us, probes = rec.get("universes", []), rec.get("probes", [])
        base = rec.get("baseline", {})
        out += [
            f"### Loop {loop} · seed {rec.get('seed')} · {len(us)} universes over "
            f"{len(rec.get('survivors', {}))} generations · adopted: {rec.get('adopted') or 'none'}",
            "",
            f"Baseline after the same ticks: posture {base.get('posture', 0):.3f}, "
            f"{base.get('contradictions', 0)} contradictions, ensemble leak {base.get('L_G', 0):.3f}.",
            "",
            "| Universe | From | Operators | Verdict | Δ posture | Δ contradictions | Δ L_G | Fitness |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for u in us:
            ops = " ".join(f"{m['symbol']}@{_tail(m['site'])}" for m in u["mutations"])
            d = u["delta"]
            out.append(
                f"| {u['id']} | {'+'.join(u['parents'])} | {ops} | {u['verdict']} | "
                f"{d['posture']:+.3f} | {d['contradictions']:+d} | {d['L_G']:+.3f} | {u['fitness']:.2f} |"
            )
        out.append("")
        for u in us[:detail]:
            only = "" if u.get("adoptable") or u.get("probe") else " (what-if only)"
            out.append(f"**{u['id']}** · {u['verdict']}{only}. {u['what_if']}")
            out += [f"- `{diff_line(e)}`" for e in u["diff"][:8]]
            if u["consequences"]:
                out.append("- Consequences: " + "; ".join(u["consequences"][:6]) + ".")
            for m in u["mutations"]:
                if m["agent_change"]:
                    out.append(f"- Lattice ({m['symbol']}): {m['agent_change']}.")
            for s in u["structures"][:4]:
                kc = " · Klein-close" if s.get("klein_close") else ""
                out.append(f"- Bred: `{s['agent']}={s['name']}`{kc}")
            out.append("")
        if probes:
            out += [
                "**ω probes: what if a restriction were reversed?** A load-bearing clause is one "
                "whose reversal heats the lattice or opens a contradiction.",
                "",
            ]
            for u in probes:
                out.append(
                    f"- **{u['id']}** {u['verdict']} (heat {u.get('heat', 0):.2f}). {u['what_if']} "
                    + ("; ".join(u["consequences"][:4]) + "." if u["consequences"] else "")
                )
            out.append("")
    return out


__all__ = [
    "ADOPT",
    "FUNCTION_CLAUSE",
    "LAW",
    "TRIGGERS",
    "EvolveConfig",
    "apply",
    "candidates",
    "compact",
    "diff_line",
    "evolve",
    "report_lines",
    "except_for",
    "image",
    "klein_close",
    "measure",
    "mirror",
    "negate",
    "object_phrase",
    "probe_candidates",
    "sample",
    "split_clause",
]
