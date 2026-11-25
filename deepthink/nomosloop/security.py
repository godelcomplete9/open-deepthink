"""Cybersecurity domain of the RSI loop: a policy as a law, posture as fitness.

The artifact the loop defends and evolves is an institution's security
policy (or runbook, or control standard). This module turns it into a scale
lattice and gives the loop a measured fitness for it:

* ``parse_policy``: clauses become rules on four levels. They are
  **L3 principles** (purpose, scope), **L2 policies** (who may do what),
  **L1 controls** (prevent, detect, respond, recover) and **L0 assets and
  events**. Each level carries a base and a modulus, as every scale does.
* ``posture``: how evenly the policy covers the eight security functions,
  divided by one plus the number of contradictory clause pairs. The eight
  functions are the eight Agent Mechanics species read through the NX-1
  lexicon (§4.2).
* ``control_clause``: the control a stocking agent proposes, derived from
  the licensed operator it applies (NX-1 §6.4).

The posture score is bookkeeping over the policy text. It measures diversity
of covered functions and internal consistency. It does not measure whether
the institution is secure.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from . import correspondence as tc
from .mechanics import ALPHA, BETA, SPECIES

FUNCTION: dict[str, str] = {
    "Ni": "threat forecasting and attack-path analysis",
    "Ne": "red-team testing and discovery of unknown paths",
    "Si": "audit logging, retention and baselines",
    "Se": "real-time monitoring and alerting",
    "Ti": "access rules and input validation",
    "Te": "incident response and patch execution",
    "Fi": "identity, integrity and least privilege",
    "Fe": "coordination, notification and policy rollout",
}

LEVELS: list[dict[str, Any]] = [
    {
        "level_id": "L3",
        "name": "Principles",
        "keys": (
            "purpose",
            "principle",
            "objective",
            "mission",
            "applies to",
            "this policy",
            "this standard",
            "this runbook",
            "intent",
        ),
        "sections": ("principle", "purpose", "scope", "objective", "introduction", "overview"),
        "base": "principle radix: one objective packs into the policies below it",
        "modulus": "scope closes the policy: who and what it applies to",
    },
    {
        "level_id": "L2",
        "name": "Policies",
        "keys": (
            "must",
            "shall",
            "required",
            "prohibited",
            "allowed",
            "may ",
            "grant",
            "deny",
            "denied",
            "permit",
            "forbid",
            "only",
            "open to",
            "accept",
            "never",
            "always",
        ),
        "sections": (
            "access",
            "polic",
            "zone",
            "response",
            "recovery",
            "rule",
            "requirement",
            "role",
            "handling",
            "usage",
        ),
        "base": "policy radix: subject × action × condition",
        "modulus": "least privilege and separation of duties close the policies",
    },
    {
        "level_id": "L1",
        "name": "Controls",
        "keys": (
            "log",
            "monitor",
            "alert",
            "encrypt",
            "mfa",
            "multi-factor",
            "firewall",
            "patch",
            "backup",
            "scan",
            "rotate",
            "review",
            "audit",
            "detect",
            "segment",
            "vpn",
            "siem",
            "edr",
            "configure",
            "enable",
            "disable",
            "block",
            "quarantine",
            "notify",
            "escalat",
            "retain",
        ),
        "sections": ("control", "monitor", "logging", "technical", "detection"),
        "base": "control radix: prevent, detect, respond, recover",
        "modulus": "every control maps to a prevention or a detection",
    },
    {
        "level_id": "L0",
        "name": "Assets and events",
        "keys": ("assets", "inventory", "in scope:", "zones:", "systems in scope", "asset list"),
        "sections": ("asset", "inventor", "system", "definition", "glossary"),
        "base": "asset radix: systems × data × identities",
        "modulus": "the asset inventory closes the ground",
    },
]

_CLAUSE = re.compile(
    r"^\s*(§\s*\d+(?:\.\d+)*|\d+(?:\.\d+)*[.)]|[-*•]|\([a-z0-9]{1,3}\))\s+(.*)$", re.I
)
_HEADER = re.compile(r"^\s*(#{1,6}\s+.+|[A-Z][A-Z0-9 &/,\-]{3,}:?)\s*$")


def parse_clauses(text: str) -> list[dict[str, str]]:
    """Split a policy into clauses: numbered (§2, 2., 2.1)), bulleted, or paragraphs."""
    clauses: list[dict[str, str]] = []
    section = ""
    cur: dict[str, str] | None = None
    auto = 0

    def flush() -> None:
        nonlocal cur
        if cur and cur["text"].strip():
            cur["text"] = re.sub(r"\s+", " ", cur["text"]).strip()
            clauses.append(cur)
        cur = None

    lines = text.splitlines()
    first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first is not None:
        if re.match(r"^\s*[A-Z][A-Z0-9 &/,\-]{6,}", lines[first]) and not _CLAUSE.match(
            lines[first]
        ):
            lines = lines[first + 1 :]  # a document title, not a clause
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        m = _CLAUSE.match(line)
        if m:
            flush()
            tag = m.group(1).replace("§", "").strip().rstrip(".)")
            if not re.match(r"^\d", tag):
                auto += 1
                tag = f"c{auto}"
            cur = {"num": tag, "text": m.group(2), "section": section}
            continue
        if _HEADER.match(line) and len(line) < 80:
            flush()
            section = line.strip("#: ").strip()
            continue
        if cur is None:
            auto += 1
            cur = {"num": f"c{auto}", "text": line, "section": section}
        else:
            cur["text"] += " " + line.strip()
    flush()
    seen: Counter[str] = Counter()
    for c in clauses:
        seen[c["num"]] += 1
        if seen[c["num"]] > 1:
            c["num"] = f"{c['num']}-{seen[c['num']]}"
    return clauses


def _level_of(clause: str, section: str = "") -> str:
    """Level of a clause: from its section header if it names one, else by weighted keywords."""
    sec = section.lower()
    if sec:
        for lv in LEVELS:
            if any(k in sec for k in lv["sections"]):
                return str(lv["level_id"])
    low = " " + clause.lower() + " "
    weight = {"L3": 3, "L2": 2, "L1": 1, "L0": 3}
    scores = {
        lv["level_id"]: weight[lv["level_id"]] * sum(1 for k in lv["keys"] if k in low)
        for lv in LEVELS
    }
    best = max(
        ("L3", "L2", "L1", "L0"), key=lambda k: (scores[k], {"L2": 3, "L1": 2, "L3": 1, "L0": 0}[k])
    )
    return best if scores[best] else "L2"


_WEAK = set(
    "through using used before after unless upon also least most more less other times every must "
    "shall never always only outside inside within during while their there these those which "
    "access granted allowed denied staff institution".split()
)


def subject(text: str, n: int = 3) -> str:
    """A short subject phrase from a clause (its first distinct content words)."""
    toks = [
        t
        for t in re.findall(r"[a-z][a-z0-9/_-]+", text.lower())
        if len(t) > 3 and t not in tc._STOP and t not in _WEAK
    ]
    keep = [t for t in toks if not any(t.startswith(k.strip()) for k in tc.POSITIVE + tc.NEGATIVE)]
    return " ".join(list(dict.fromkeys(keep))[:n]) or "the institution"


def parse_policy(text: str, law_id: str) -> dict[str, Any] | None:
    """Policy text -> scale lattice (None if fewer than two clauses)."""
    clauses = parse_clauses(text)
    if len(clauses) < 2:
        return None
    by_level: dict[str, list[dict[str, Any]]] = {lv["level_id"]: [] for lv in LEVELS}
    for c in clauses:
        lid = _level_of(c["text"], c["section"])
        by_level[lid].append(
            {
                "id": f"§{c['num']}",
                "name": " ".join(c["text"].split()[:6]),
                "level": lid,
                "pattern": subject(c["text"]),
                "rewrite": c["text"][:400],
                "examples": [c["section"]] if c["section"] else [],
                "generative_power": 1.0,
                "provenance": "policy-clause",
            }
        )
    levels = []
    for lv in LEVELS:
        rules = by_level[lv["level_id"]]
        if not rules:
            continue
        levels.append(
            {
                "level_id": lv["level_id"],
                "name": lv["name"],
                "alphabet": [],
                "description": f"{lv['name']} of the policy",
                "rules": rules,
                "abstraction_score": {"L3": 0.9, "L2": 0.7, "L1": 0.4, "L0": 0.1}[lv["level_id"]],
                "base": lv["base"],
                "base_spec": {"radix": None, "alphabet": [], "encoding": "clauses"},
                "modulus": lv["modulus"],
                "modulus_spec": {
                    "period": None,
                    "constraint": lv["modulus"],
                    "operator": "closure",
                },
            }
        )
    return {
        "version": "v0",
        "law_id": law_id,
        "levels": levels,
        "order": "L3..L0 top-down",
        "expansion_map": {},
        "provenance": {"induction": "policy-parser"},
    }


def ensure_controls_level(lattice: dict[str, Any]) -> dict[str, Any]:
    """Add an empty L1 Controls level (before L0) so new controls have a home."""
    levels = list(lattice.get("levels", []) or [])
    if any(str(lv.get("level_id")) == "L1" for lv in levels):
        return lattice
    spec = next(lv for lv in LEVELS if lv["level_id"] == "L1")
    new = {
        "level_id": "L1",
        "name": spec["name"],
        "alphabet": [],
        "description": "Controls of the policy",
        "rules": [],
        "abstraction_score": 0.4,
        "base": spec["base"],
        "base_spec": {"radix": None, "alphabet": [], "encoding": "clauses"},
        "modulus": spec["modulus"],
        "modulus_spec": {"period": None, "constraint": spec["modulus"], "operator": "closure"},
    }
    idx = next((i for i, lv in enumerate(levels) if str(lv.get("level_id")) == "L0"), len(levels))
    levels.insert(idx, new)
    out = dict(lattice)
    out["levels"] = levels
    return out


def clauses_of(lattice: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    out = []
    for lv in lattice.get("levels", []) or []:
        for r in lv.get("rules", []) or []:
            out.append((str(lv.get("level_id")), str(r.get("id")), r))
    return out


def function_vector(lattice: dict[str, Any]) -> list[float]:
    W = [0.0] * 8
    for _lid, _rid, r in clauses_of(lattice):
        p, _ = tc.project_text(f"{r.get('name', '')} {r.get('rewrite', '')}")
        W = [a + b for a, b in zip(W, p)]
    return W


def contradictions(lattice: dict[str, Any]) -> list[tuple[str, str, float]]:
    """Contradictory pairs among policy and control clauses (L2, L1)."""
    cl = [c for c in clauses_of(lattice) if c[0] in ("L2", "L1")]
    out = []
    for i, (la, ra, a) in enumerate(cl):
        for lb, rb, b in cl[i + 1 :]:
            c = tc.contradiction(str(a.get("rewrite", "")), str(b.get("rewrite", "")))
            if c > 0:
                out.append((f"pol/{la}/{ra}", f"pol/{lb}/{rb}", c))
    return out


def posture(lattice: dict[str, Any]) -> dict[str, Any]:
    """Diversity of covered functions over (1 + contradictory clause pairs)."""
    W = function_vector(lattice)
    tot = sum(W)
    if tot <= 0:
        return {
            "score": 0.0,
            "diversity": 0.0,
            "contradictions": 0,
            "functions": {},
            "missing": list(SPECIES),
        }
    q = [w / tot for w in W]
    H = -sum(x * math.log(x) for x in q if x > 0)
    diversity = H / math.log(8)
    contra = contradictions(lattice)
    score = diversity / (1 + len(contra))
    return {
        "score": round(score, 4),
        "diversity": round(diversity, 4),
        "contradictions": len(contra),
        "contradiction_pairs": [[a, b, round(c, 2)] for a, b, c in contra[:12]],
        "functions": {SPECIES[i]: round(q[i], 3) for i in range(8)},
        "missing": [SPECIES[i] for i in range(8) if q[i] < 1 / 32],
    }


def target_function(op: str, hero: str, lattice: dict[str, Any], salt: str = "") -> str:
    """Which security function a licensed operator adds (NX-1 §6.4, security column)."""
    if op == "r":
        return BETA[hero]
    if op == "a":
        return ALPHA[hero]
    if op == "ra":
        return ALPHA[BETA[hero]]
    if op == "store4":
        return "Si"
    if op == "cut":
        return "Fi"
    W = function_vector(lattice)
    tot = sum(W) or 1.0
    weak = sorted((i for i in range(8) if W[i] / tot < 1 / 8), key=lambda i: (W[i], i)) or [
        min(range(8), key=lambda i: (W[i], i))
    ]
    pick = sum(ord(ch) for ch in salt) % min(3, len(weak)) if salt else 0
    return SPECIES[weak[pick]]  # sigma, arrow, eta: one of the weakest functions


EDIT_TEXT = {
    "a": "add the dual check on the opposite side (ingress and egress)",
    "r": "cover the complementary function",
    "ra": "promote the local control to policy",
    "sigma": "cover the weakest function",
    "eta": "consolidate overlapping controls into one standard",
    "arrow": "split the overloaded scope and cover the weakest function",
    "store4": "close the loop: log, alert, review",
    "cut": "isolate the clashing scope behind identity and least privilege",
}


def control_clause(
    op: str,
    hero: str,
    clause_text: str,
    lattice: dict[str, Any],
    objective: str = "",
    salt: str = "",
) -> tuple[str, str]:
    """Deterministic control text for an operator applied to a clause."""
    f = target_function(op, hero, lattice, salt)
    scope = subject(clause_text) if clause_text else subject(objective)
    return f, function_control(f, scope, objective)


VERB = {
    "Ni": "Analyse likely attack paths against",
    "Ne": "Run periodic red-team tests against",
    "Si": "Log, retain and baseline all activity on",
    "Se": "Monitor in real time and alert on anomalous activity on",
    "Ti": "Enforce explicit access rules and input validation for",
    "Te": "Maintain an incident-response and patch playbook for",
    "Fi": "Require strong identity, integrity checks and least privilege for",
    "Fe": "Notify owners and coordinate policy rollout for",
}


def function_control(f: str, scope: str, objective: str = "") -> str:
    """One enforceable sentence that adds security function ``f`` to ``scope``."""
    text = f"{VERB[f]} {scope}."
    if objective:
        text = text[:-1] + f", in service of: {objective.strip().rstrip('.')[:120]}."
    return text
