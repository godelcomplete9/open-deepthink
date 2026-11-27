"""Table of correspondences (ToC) and the cross-domain gauge connection.

Implements NX-1 §3–§6 (``docs/NOMENCLATURE_COMPLEMENT.md``):

* artifact nodes ``AX:`` built from the policy lattice (levels, clauses,
  moduli) and the viable control operations;
* artifact relations with a phase (0 consistent, π contradictory);
* ToC rows mapping each artifact element onto the eight species (the
  functional basis read through the Agent Mechanics lexicon), by a mapper
  LLM when available and by the deterministic lexicon otherwise;
* link phases, faces, curvature ``B = d₁A``, frustration ``(1 − cos B)/2``,
  gauge transformations ``A → A + d₀χ`` (which leave ``B`` invariant);
* the symbolic gauge coderivative ``δ_A``: a licensed lattice operator at an
  agent, transported to artifact edit symbols along its rows.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .mechanics import SPECIES

# ------------------------------------------------------------- lexicon -----

LEXICON: dict[str, tuple[str, ...]] = {
    "Ni": (
        "converge",
        "imply",
        "implies",
        "forecast",
        "predict",
        "determine",
        "plan",
        "schedul",
        "path",
        "geodesic",
        "future",
        "strategy",
        "unif",
        "single",
        "vacuum",
        "principle",
        "threat",
    ),
    "Ne": (
        "branch",
        "generat",
        "variant",
        "explor",
        "fuzz",
        "search",
        "alternative",
        "option",
        "divers",
        "discover",
        "random",
        "spawn",
        "possib",
        "red-team",
        "novel",
        "expand",
    ),
    "Si": (
        "preserv",
        "record",
        "log",
        "audit",
        "history",
        "cache",
        "baseline",
        "backup",
        "retain",
        "retention",
        "conserv",
        "memory",
        "recur",
        "periodic",
        "invariant",
        "migration",
        "archiv",
    ),
    "Se": (
        "act",
        "apply",
        "now",
        "real-time",
        "realtime",
        "transmit",
        "input",
        "output",
        "i/o",
        "handler",
        "monitor",
        "telemetry",
        "detect",
        "sensor",
        "live",
        "request",
        "response",
        "traffic",
    ),
    "Ti": (
        "defin",
        "constrain",
        "constraint",
        "validat",
        "rule",
        "type",
        "pars",
        "check",
        "consisten",
        "closure",
        "close",
        "modulus",
        "condition",
        "firewall",
        "schema",
        "logic",
        "proof",
        "access rule",
    ),
    "Te": (
        "execut",
        "produc",
        "measur",
        "metric",
        "orchestrat",
        "run",
        "deploy",
        "patch",
        "remediat",
        "incident",
        "throughput",
        "efficien",
        "process",
        "build",
        "compute",
        "work",
        "operation",
    ),
    "Fi": (
        "protect",
        "bind",
        "must not",
        "integrity",
        "identity",
        "auth",
        "secret",
        "credential",
        "privilege",
        "permission",
        "encrypt",
        "crypto",
        "trust",
        "deny",
        "forbid",
        "confidential",
    ),
    "Fe": (
        "coordinat",
        "agree",
        "propagat",
        "consensus",
        "broadcast",
        "protocol",
        "sync",
        "share",
        "notify",
        "policy",
        "rollout",
        "communicat",
        "team",
        "gauge",
        "collective",
        "norm",
    ),
}

POSITIVE = (
    "allow",
    "permit",
    "grant",
    "enable",
    "public",
    "accept",
    "always",
    "open",
    "anonymous",
    "required",
    "expose",
    "unrestricted",
    "plaintext",
    "any server",
    "all traffic",
)
NEGATIVE = (
    "deny",
    "denied",
    "forbid",
    "prohibit",
    "disable",
    "private",
    "reject",
    "never",
    "block",
    "must not",
    "restricted",
    "encrypt",
    "only ",
    "outside",
    "except",
)
_STOP = set(
    "the a an of to in on for with and or by as at from that this it is are was were be been "
    "being which who whom whose what where when how why not no yes if then than so such into over "
    "under between within without about per via all any every each must may shall should can".split()
)

VERBS = {
    "Ni": "converges",
    "Ne": "branches",
    "Si": "preserves",
    "Se": "acts",
    "Ti": "constrains",
    "Te": "executes",
    "Fi": "protects",
    "Fe": "coordinates",
}


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z][a-z0-9_]+", text.lower()) if len(t) >= 3 and t not in _STOP
    }


def project_text(
    text: str, prior: dict[str, float] | None = None
) -> tuple[list[float], dict[str, str]]:
    """Species projection p (sums to 1) of a functional description, with evidence."""
    low = " " + text.lower() + " "
    scores = [0.0] * 8
    evidence: dict[str, str] = {}
    for i, sp in enumerate(SPECIES):
        for kw in LEXICON[sp]:
            idx = low.find(kw)
            if idx >= 0:
                scores[i] += 1.0
                if sp not in evidence:
                    a = max(0, idx - 18)
                    evidence[sp] = low[a : idx + len(kw) + 18].strip()
    for sp, v in (prior or {}).items():
        scores[SPECIES.index(sp)] += v
    tot = sum(scores)
    if tot <= 0:
        scores = [0.0] * 8
        scores[SPECIES.index("Ti")] = 0.5
        scores[SPECIES.index("Si")] = 0.5
        tot = 1.0
    return [s / tot for s in scores], evidence


def polarity(text: str) -> int:
    """+1 permissive, -1 restrictive, 0 neutral. Restrictive modals weigh 1.5."""
    low = " " + text.lower() + " "
    pos = sum(low.count(w) for w in POSITIVE)
    neg = 1.5 * sum(low.count(w) for w in NEGATIVE)
    return 1 if pos > neg else -1 if neg > pos else 0


def contradiction(a: str, b: str) -> float:
    """0..1 contradiction between two clauses: opposite polarity on a shared subject.

    A shared subject needs at least two shared content words that are not
    themselves polarity cues.
    """
    pa, pb = polarity(a), polarity(b)
    if pa == 0 or pb == 0 or pa == pb:
        return 0.0
    ta, tb = _tokens(a), _tokens(b)
    cues = tuple(w.strip() for w in POSITIVE + NEGATIVE)
    shared = {t for t in ta & tb if not any(t.startswith(c) or c.startswith(t) for c in cues)}
    if len(shared) < 2:
        return 0.0
    return min(1.0, 0.25 * len(shared) + 0.25)


# --------------------------------------------------------------- AX graph ---


@dataclass
class AXNode:
    ref: str
    kind: str  # law | level | rule | modulus | operation | code | social
    text: str
    attrs: dict[str, Any] = field(default_factory=dict)


def lattice_ax(
    lattice: dict[str, Any], ops: list[dict[str, Any]], prefix: str = "pol"
) -> tuple[list[AXNode], list[tuple[str, str, float]]]:
    """Artifact nodes and phased relations of a policy lattice and its viable operations."""
    nodes: list[AXNode] = []
    edges: list[tuple[str, str, float]] = []
    levels = list(lattice.get("levels", []) or [])
    prev_rules: list[str] = []
    for lv in levels:
        lid = str(lv.get("level_id"))
        mod_ref = f"{prefix}/{lid}#mod"
        mod_text = f"{lv.get('modulus', '')} {json.dumps(lv.get('modulus_spec', {}))} closes {lv.get('name', '')}"
        nodes.append(AXNode(mod_ref, "modulus", mod_text, {"level": lid}))
        rule_refs = []
        for r in lv.get("rules", []) or []:
            ref = f"{prefix}/{lid}/{r.get('id')}"
            text = f"{r.get('name', '')}: {r.get('pattern', '')} => {r.get('rewrite', '')}"
            nodes.append(AXNode(ref, "rule", text, {"level": lid, "rule": r.get("id")}))
            edges.append((ref, mod_ref, 0.0))  # the modulus closes its rules
            rule_refs.append(ref)
        for up in prev_rules:  # implicit downward expansion
            for down in rule_refs[:4]:
                edges.append((up, down, 0.0))
        prev_rules = rule_refs
    # contradictions between clauses that share a subject
    rule_nodes = [n for n in nodes if n.kind == "rule" and n.attrs.get("level") in ("L2", "L1")]
    for i, a in enumerate(rule_nodes):
        for b in rule_nodes[i + 1 :]:
            c = contradiction(a.text, b.text)
            if c > 0:
                edges.append((a.ref, b.ref, math.pi * c))
    for op in ops:
        ref = f"op/{op['id']}"
        nodes.append(
            AXNode(
                ref,
                "operation",
                op.get("text", op["id"]),
                {
                    "op": op["operation"],
                    "target": op.get("target", ""),
                    "function": op.get("function", ""),
                },
            )
        )
        if op.get("target"):
            edges.append((ref, op["target"], 0.0))
    return nodes, edges


# --------------------------------------------------------------- ToC rows ---


@dataclass
class TocRow:
    n: int
    ax_ref: str
    agent_id: str
    basis: list[str]
    p: list[float]
    evidence: dict[str, str]
    spectrum: str
    why: str
    chain: str
    A: float
    conf: float
    mapper: str
    alive: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "ax": self.ax_ref,
            "agent": self.agent_id,
            "basis": self.basis,
            "p": {SPECIES[i]: round(v, 3) for i, v in enumerate(self.p) if v > 0.005},
            "evidence": self.evidence,
            "spectrum": self.spectrum,
            "why": self.why,
            "chain": self.chain,
            "A": round(self.A, 4),
            "conf": round(self.conf, 3),
            "mapper": self.mapper,
            "alive": self.alive,
        }

    def nx1(self) -> str:
        p = ",".join(
            f"{SPECIES[i]}{v:.2f}".rstrip("0").rstrip(".")
            for i, v in sorted(enumerate(self.p), key=lambda t: -t[1])
            if v > 0.01
        )
        tomb = "" if self.alive else "!x"
        return (
            f"TC:{self.n}{tomb}  AX:{self.ax_ref} ⟼ AM:{self.agent_id}  basis={','.join(self.basis)}  p={p}  "
            f"spectrum={self.spectrum}  chain={self.chain}  A={self.A:.2f}  conf={self.conf:.2f}  mapper={self.mapper}"
        )


KIND_PRIOR: dict[str, dict[str, float]] = {
    "modulus": {"Ti": 1.0, "Si": 0.8},
    "rule": {},
    "operation": {},
    "law": {"Ni": 0.5},
}


def offline_row(node: AXNode) -> dict[str, Any]:
    prior = dict(KIND_PRIOR.get(node.kind) or {})
    sp = node.attrs.get("species")
    if sp in SPECIES:  # code and social facts carry the function they touch
        prior[sp] = prior.get(sp, 0.0) + 2.0
    p, ev = project_text(node.text, prior)
    top = sorted(range(8), key=lambda i: -p[i])[:3]
    basis = [VERBS[SPECIES[i]] for i in top if p[i] > 0.05]
    why = {
        "code": "what actually runs; a discrepancy with a clause is a leak in practice",
        "social": "a procedure run by people; a gap is a social-engineering path",
        "modulus": "closes a scale; energy only by traversal of its rules",
        "rule": "a clause entrusted by the institution; pattern without own intake",
        "operation": "a viable control operation; prosthesis energised by its licensing agent",
    }.get(node.kind, "artifact element")
    conf = 0.35 + 0.1 * min(4, len(ev))
    return {
        "p": p,
        "evidence": ev,
        "basis": basis,
        "spectrum": "A",
        "why": why,
        "conf": min(0.8, conf),
        "conflicts": [],
    }


TOC_SYSTEM = (
    "You build a table of correspondences between the clauses of a security policy and the eight "
    "species of Agent Mechanics, read as security functions: Ni threat forecasting and attack paths, "
    "Ne red-team testing and discovery, Si audit logging and baselines, Se real-time monitoring and "
    "alerting, Ti access rules and validation, Te incident response and patching, Fi identity, "
    "integrity and least privilege, Fe coordination and policy rollout. Read the FUNCTION of each "
    "clause, never its topic. List every clause it contradicts. Answer with JSON only."
)


def map_rows_llm(
    nodes: list[AXNode], call: Callable[..., dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Map nodes with the mapper LLM in one batch; returns (ref -> row data, routing record)."""
    items = [{"ref": n.ref, "kind": n.kind, "text": n.text[:240]} for n in nodes]
    prompt = (
        'For each element return {"ref", "p": {species: weight}, "evidence": {species: quote}, '
        '"basis": [verbs], "spectrum": one of B,I,E,A,M,T,Q, "why", "conf": 0..1, '
        '"conflicts": [refs this element contradicts]}. Output {"rows": [...]}.\n\n'
        + json.dumps(items, ensure_ascii=False)
    )
    r = call(prompt, task_kind="toc_map", system=TOC_SYSTEM, max_tokens=2500)
    out: dict[str, dict[str, Any]] = {}
    data = _json_block(r.get("text", ""))
    for row in (data or {}).get("rows", []) if isinstance(data, dict) else []:
        try:
            ref = str(row["ref"])
            pv = row.get("p") or {}
            vec = [max(0.0, float(pv.get(sp, 0.0))) for sp in SPECIES]
            tot = sum(vec)
            if tot <= 0:
                continue
            out[ref] = {
                "p": [v / tot for v in vec],
                "evidence": {
                    k: str(v)[:80] for k, v in (row.get("evidence") or {}).items() if k in SPECIES
                },
                "basis": [str(b)[:24] for b in (row.get("basis") or [])][:4],
                "spectrum": str(row.get("spectrum", "A"))[:1]
                if str(row.get("spectrum", "A"))[:1] in "BIEAMTQ"
                else "A",
                "why": str(row.get("why", ""))[:160],
                "conf": max(0.0, min(1.0, float(row.get("conf", 0.5)))),
                "conflicts": [str(c) for c in (row.get("conflicts") or [])][:6],
            }
        except (KeyError, TypeError, ValueError):
            continue
    return out, r


def _json_block(text: str) -> Any:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        end = max(text.rfind("}"), text.rfind("]"))
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


# ------------------------------------------------------------ gauge field ---


def link_phase(p: list[float], w: list[float]) -> float:
    """A(a ↔ i) = arccos cos(p_a, w_i): how faithfully the agent carries the function."""
    num = sum(x * y for x, y in zip(p, w))
    den = math.sqrt(sum(x * x for x in p)) * math.sqrt(sum(y * y for y in w))
    if den <= 0:
        return math.pi / 2
    return math.acos(max(-1.0, min(1.0, num / den)))


@dataclass
class Face:
    a1: str
    a2: str
    i1: str
    i2: str
    B: float
    f: float
    A_ax: float = 0.0

    @property
    def weight(self) -> float:
        """Findings rank real artifact contradictions above latent mapping mismatches."""
        if self.A_ax >= math.pi / 2:
            return self.f
        both_clauses = "/L" in self.a1 and "/L" in self.a2 and "#mod" not in self.a1 + self.a2
        return self.f * (0.5 if both_clauses else 0.25)


def curvature(A_ax: float, A_tc1: float, A_tc2: float, A_am: float) -> float:
    """B = d₁A around a1 → a2 → i2 → i1 → a1."""
    return A_ax + A_tc2 - A_am - A_tc1


def frustration(B: float) -> float:
    return (1.0 - math.cos(B)) / 2.0


def faces(
    ax_edges: list[tuple[str, str, float]],
    rows_by_ax: dict[str, list[TocRow]],
    am_phase: Callable[[str, str], float | None],
) -> list[Face]:
    """All squares (a1 → a2 ⇝ i2 ← i1 ⇜ a1) whose four edges exist (NX-1 §6.1)."""
    out: list[Face] = []
    for a1, a2, A_ax in ax_edges:
        for r1 in rows_by_ax.get(a1, []):
            for r2 in rows_by_ax.get(a2, []):
                if not (r1.alive and r2.alive) or r1.agent_id == r2.agent_id:
                    continue
                A_am = am_phase(r1.agent_id, r2.agent_id)
                if A_am is None:
                    continue
                B = curvature(A_ax, r1.A, r2.A, A_am)
                out.append(Face(a1, a2, r1.agent_id, r2.agent_id, B, frustration(B), A_ax))
    return out


def gauge_transform(
    links: dict[tuple[str, str], float], chi: dict[str, float]
) -> dict[tuple[str, str], float]:
    """A → A + d₀χ on every oriented edge (src, dst). B is invariant (d₁d₀ = 0)."""
    return {(s, d): A + chi.get(d, 0.0) - chi.get(s, 0.0) for (s, d), A in links.items()}


def square_curvature(
    links: dict[tuple[str, str], float], a1: str, a2: str, i1: str, i2: str
) -> float:
    """B computed from an oriented link table, for invariance checks."""
    return links[(a1, a2)] + links[(a2, i2)] - links[(i1, i2)] - links[(a1, i1)]


# -------------------------------------------------- symbolic coderivative ---

# 𝓜[o]: the edit symbol a licensed operator becomes in a security policy (NX-1 §6.4).
EDITS: dict[str, str] = {
    "a": "add the dual check on the opposite side (ingress and egress)",
    "r": "cover the complementary function",
    "ra": "promote the local control to policy",
    "sigma": "cover the weakest function",
    "eta": "consolidate overlapping controls into one standard",
    "arrow": "split the overloaded scope",
    "store4": "close the loop: log, alert, review",
    "cut": "isolate the clashing scope behind identity and least privilege",
    "omega": "none",
}


@dataclass
class ArtifactEdit:
    ax_ref: str
    symbol: str
    weight: float
    inverse: bool
    op: str
    agent: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ax": self.ax_ref,
            "edit": self.symbol,
            "weight": round(self.weight, 3),
            "inverse": self.inverse,
            "op": self.op,
            "agent": self.agent,
        }


def coderivative(op: str, agent_id: str, rows: list[TocRow]) -> list[ArtifactEdit]:
    """δ_A: transport a licensed lattice operator at an agent to policy edits.

    Aligned rows (A ≈ 0) carry 𝓜[o]; anti-aligned rows (A ≈ π) carry 𝓜[o]⁻¹,
    weighted by ½(1 ± cos A) (NX-1 §6.4).
    """
    sym = EDITS.get(op, EDITS["omega"])
    out: list[ArtifactEdit] = []
    for r in rows:
        if not r.alive or r.agent_id != agent_id:
            continue
        w_fwd = 0.5 * (1 + math.cos(r.A))
        w_inv = 0.5 * (1 - math.cos(r.A))
        if w_fwd >= w_inv:
            out.append(ArtifactEdit(r.ax_ref, sym, w_fwd, False, op, agent_id))
        else:
            out.append(ArtifactEdit(r.ax_ref, f"undo: {sym}", w_inv, True, op, agent_id))
    return out
