"""Specialized sub-agents. The main graph never does heavy lifting itself.

Tiers (see routing.ROUTING_POLICY):
  driver (local): tokenizer, motif_miner, consistency_checker
  mapper        : hierarchy_synthesizer (and the table of correspondences)

Each agent logs its routing decision so the run is auditable.
Offline mode uses deterministic heuristics so the MVP runs without API keys;
with an injected LLM bridge (open-deepthink) or OPENROUTER_API_KEY set, calls
go to the active model. Driver calls never leave the machine.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from .routing import call_llm
from .state import Motif, ScaleLattice, ScaleLevel, ScaleRule

AGENT_REGISTRY: dict[str, str] = {
    "tokenizer": "driver: regex+LLM tokenisation",
    "motif_miner": "driver: motif extraction",
    "consistency_checker": "driver: local consistency",
    "hierarchy_synthesizer": "mapper: multi-scale synthesis",
}

# -- cheap agents ------------------------------------------------------------

STOP = set(
    "the a an of to in on for with and or by as at from that this it is are was were be been being which who whom whose what where when how why not no yes if then than so such into over under between within without about into per via vs v s".split()
)


def tokenizer_agent(text: str) -> dict[str, Any]:
    r = call_llm(
        f"Tokenise the following formal text into atomic symbols/terms. Return space-separated tokens.\n\n{text[:3000]}",
        task_kind="tokenise",
        system="You are a precise tokenizer for formal scientific/legal text.",
    )
    toks = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|\\[a-zA-Z]+|d[A-Za-z]+|∂|μ|ν|ρ|σ|α|β|γ|Δ|∇|[0-9]+\.?[0-9]*|[+\-*/^=()\[\]{}_,.;:]",
        text,
    )
    toks = [t for t in toks if t.strip()]
    return {
        "tokens": toks[:2000],
        "llm_note": r["text"][:300],
        "tier": r["tier"],
        "model": r["model"],
        "offline": r["offline"],
    }


def motif_miner_agent(text: str, tokens: list[str]) -> dict[str, Any]:
    r = call_llm(
        f"Mine recurring motifs (symbols, objects, equations, principles) from these tokens:\n{' '.join(tokens[:400])}\n\nSource:\n{text[:3000]}",
        task_kind="motif_mine",
        system="You mine formal motifs: symbols, field objects, equations, variational principles.",
    )
    motifs: list[Motif] = []
    # heuristic motif mining (runs in both modes; LLM refines labels when online)
    eqs = re.findall(r".{0,40}=.{0,60}", text)
    caps = re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\b", text)
    greek = re.findall(
        r"\\[a-zA-Z]+|[A-Z]_\{[^{}]{1,12}\}|F_\{[^{}]{1,8}\}|g_\{[^{}]{1,8}\}|A_μ|∂_μ|Γ\^?\{?[^{}]*\}?",
        text,
    )
    cands = (
        [(e.strip(), "equation") for e in eqs[:12]]
        + [(c, "object") for c in caps[:12]]
        + [(g, "symbol") for g in greek[:12]]
    )
    seen: dict[str, int] = {}
    for surf, kind in cands:
        key = re.sub(r"\s+", " ", surf.strip().lower())[:80]
        if len(key) < 2:
            continue
        seen[key] = seen.get(key, 0) + 1
    for i, (key, freq) in enumerate(sorted(seen.items(), key=lambda kv: -kv[1])[:16]):
        lvl = (
            "L0"
            if len(key) < 12
            else ("L1" if len(key) < 30 else ("L2" if "=" in key or len(key) < 60 else "L3"))
        )
        if any(
            w in key
            for w in ("action", "principle", "unif", "kaluza", "variational", "einstein-hilbert")
        ):
            lvl, kind = "L3", "principle"
        elif "=" in key:
            kind = "equation"
        motifs.append(
            Motif(
                id=f"m{i:02d}",
                surface_form=key[:120],
                normalized=key[:120],
                frequency=freq,
                contexts=[key[:80]],
                candidate_level=lvl,
                kind=kind,
            )
        )
    # guarantee L0..L3 coverage for the lattice
    have = {m["candidate_level"] for m in motifs}
    filler = [
        ("L0", "symbol", "x^μ g_μν ∂_μ"),
        ("L1", "object", "A_μ F_μν"),
        ("L2", "equation", "∂_μ F^μν = J^ν"),
        ("L3", "principle", "5d vacuum einstein action"),
    ]
    for lvl, kind, surf in filler:
        if lvl not in have:
            motifs.append(
                Motif(
                    id=f"m{len(motifs):02d}",
                    surface_form=surf,
                    normalized=surf,
                    frequency=1,
                    contexts=["synthetic-backfill"],
                    candidate_level=lvl,
                    kind=kind,
                )
            )
    return {"motifs": motifs, "tier": r["tier"], "model": r["model"], "offline": r["offline"]}


def consistency_checker_agent(lattice: ScaleLattice) -> dict[str, Any]:
    r = call_llm(
        f"Check local consistency of this scale lattice (duplicate rules, orphan levels, missing base/modulus):\n{json.dumps(lattice)[:2500]}",
        task_kind="consistency_check",
        system="You are a strict local consistency checker.",
    )
    issues = []
    seen = set()
    for lvl in lattice.get("levels", []):
        if not lvl.get("base"):
            issues.append(f"{lvl.get('level_id')}: missing base (LLM must identify packing radix)")
        if not lvl.get("modulus"):
            issues.append(
                f"{lvl.get('level_id')}: missing modulus (LLM must identify close-out constraint)"
            )
        for rule in lvl.get("rules", []):
            if rule.get("id") in seen:
                issues.append(f"duplicate rule {rule.get('id')}")
            seen.add(rule.get("id"))
    if not lattice.get("levels"):
        issues.append("empty lattice")
    return {"issues": issues, "tier": r["tier"], "model": r["model"], "offline": r["offline"]}


# -- frontier agents ----------------------------------------------------------

FRONTIER_SYNTH_SYSTEM = (
    "You are RSI's frontier synthesizer. Invent coherent multi-scale rewrite grammars: "
    "compact high-level statements expanding into many consistent low-level instances. "
    "For EVERY scale level you must identify (1) its BASE — the rewrite alphabet/radix that "
    "packs this level into the one below (like binary packs decimal, like (g,A,phi) triples pack "
    "5D geometry), and (2) its MODULUS — the cycle/wrap/constraint that closes the scale "
    "(periodicity, gauge equivalence, cylinder condition, component partition, index symmetry). "
    "Prefer compression + generative power over descriptive accuracy. "
    "Always return strict JSON only."
)

BASE_MODULUS_SCHEMA_HINT = """{
  "levels": [
    {"level_id": "L3", "name": "...", "alphabet": ["..."],
     "description": "...", "abstraction_score": 0.95,
     "base": "how L3 packs into L2 (radix/encoding in words)",
     "base_spec": {"radix": null, "alphabet": ["..."], "encoding": "..."},
     "modulus": "what cycle/constraint closes L3 (in words)",
     "modulus_spec": {"period": null, "constraint": "...", "operator": "..."},
     "rules": [{"id": "R-x", "name": "...", "pattern": "...", "rewrite": "...",
                "examples": ["..."], "generative_power": 32.0}]},
    {"level_id": "L2", ...}, {"level_id": "L1", ...}, {"level_id": "L0", ...}
  ],
  "expansion_map": {"R-top": ["R-child-a", "R-child-b"]},
  "order": "L3(top,most-compressed) -> L0(bottom,ground-base)"
}"""


def _extract_json(text: str) -> dict | None:
    """Pull the first {...} JSON block out of LLM output. None if unparseable."""
    import json as _json
    import re as _re

    if not text:
        return None
    m = _re.search(r"\{.*\}", text, _re.DOTALL)
    if not m:
        return None
    cand = m.group(0)
    # try the whole block first, then a balanced-brace scan for fenced/trailing text
    try:
        return _json.loads(cand)
    except Exception:
        pass
    depth, start = 0, -1
    for i, ch in enumerate(cand):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return _json.loads(cand[start : i + 1])
                except Exception:
                    pass
    return None


def _validate_llm_lattice(data: dict, law_id: str, motifs: list[Motif]) -> ScaleLattice | None:
    """Accept LLM JSON only if it has 2-6 levels, each with a real base, a real
    modulus, and >=1 rule. A level whose base or modulus is absent/blank is
    rejected outright — the whole point is that the LLM identifies them, so a
    silent placeholder would defeat the contract. Returns None (caller falls
    back to the heuristic lattice) when anything is missing.
    """
    import time as _time

    if not isinstance(data, dict):
        return None
    levels = data.get("levels")
    if not isinstance(levels, list) or not (2 <= len(levels) <= 6):
        return None
    clean: list[ScaleLevel] = []
    for i, lv in enumerate(levels):
        if not isinstance(lv, dict):
            return None
        lid = str(lv.get("level_id") or f"L{i}")
        base = str(lv.get("base") or "").strip()
        modulus = str(lv.get("modulus") or "").strip()
        if not base or not modulus:
            return None  # base/modulus are mandatory, not defaulted
        rules = lv.get("rules") or []
        if not isinstance(rules, list) or not rules:
            return None
        c_rules: list[ScaleRule] = []
        for j, r in enumerate(rules):
            if not isinstance(r, dict):
                return None
            c_rules.append(
                {
                    "id": str(r.get("id") or f"R-{lid}-{j}"),
                    "name": str(r.get("name") or f"RULE-{j}"),
                    "level": lid,
                    "pattern": str(r.get("pattern") or ""),
                    "rewrite": str(r.get("rewrite") or ""),
                    "examples": list(r.get("examples") or [])[:4],
                    "generative_power": float(r.get("generative_power") or 8.0),
                    "provenance": "llm:base-modulus-synthesis",
                }
            )
        alpha = (
            lv.get("base_spec", {}).get("alphabet")
            if isinstance(lv.get("base_spec"), dict)
            else None
        )
        clean.append(
            {
                "level_id": lid,
                "name": str(lv.get("name") or lid),
                "alphabet": list(lv.get("alphabet") or alpha or []),
                "description": str(lv.get("description") or ""),
                "abstraction_score": float(
                    lv.get("abstraction_score") if lv.get("abstraction_score") is not None else 0.5
                ),
                "base": base,
                "base_spec": dict(
                    lv.get("base_spec")
                    or {"radix": None, "alphabet": list(lv.get("alphabet") or []), "encoding": base}
                ),
                "modulus": modulus,
                "modulus_spec": dict(
                    lv.get("modulus_spec")
                    or {"period": None, "constraint": modulus, "operator": ""}
                ),
                "rules": c_rules,
            }
        )
    exp = data.get("expansion_map") or {}
    if not isinstance(exp, dict):
        exp = {}
    return ScaleLattice(
        version="v0",
        law_id=law_id,
        levels=clean,
        order=str(data.get("order") or "LLM-ordered top-compressed -> bottom-base"),
        expansion_map={str(k): [str(x) for x in v] for k, v in exp.items() if isinstance(v, list)},
        provenance={"synthesizer": "frontier-llm:base-modulus", "motif_count": len(motifs)},
        created_at=_time.time(),
    )


def hierarchy_synthesizer_agent(text: str, motifs: list[Motif], law_id: str) -> dict[str, Any]:
    """Frontier synthesis where the LLM identifies each level's BASE and MODULUS.

    Primary path (online / injected LLM): LLM returns the full lattice JSON
    (levels + base / modulus + rules + expansion_map); we validate and accept
    it verbatim.
    Fallback (offline or invalid JSON): deterministic heuristic lattice with
    hand-filled base/modulus so the MVP still runs with zero keys.
    """
    motif_str = "\n".join(
        f"- {m['id']} [{m['candidate_level']}/{m['kind']}] {m['surface_form']} (x{m['frequency']})"
        for m in motifs[:20]
    )
    r = call_llm(
        f"Synthesize a scale lattice for this formal fragment.\n\n"
        f"Motifs:\n{motif_str}\n\nSource:\n{text[:4000]}\n\n"
        f"Requirements:\n"
        f"1. 3-5 levels, top = most compressed principle, bottom = ground symbol base.\n"
        f"2. For EACH level identify its BASE (rewrite radix/alphabet/encoding into the level below) "
        f"and its MODULUS (cycle/wrap/constraint closing the scale).\n"
        f"3. Each level needs >=1 executable rule {{id,name,pattern,rewrite,examples,generative_power}}.\n"
        f"4. Provide expansion_map {{parent_rule_id: [child_rule_ids]}} grounding the top rule to the base.\n"
        f"Return STRICT JSON ONLY matching this schema:\n{BASE_MODULUS_SCHEMA_HINT}",
        task_kind="hierarchy_synthesis",
        system=FRONTIER_SYNTH_SYSTEM,
        max_tokens=3000,
    )
    if not r.get("offline"):
        parsed = _extract_json(r["text"])
        valid = _validate_llm_lattice(parsed, law_id, motifs) if parsed else None
        if valid is not None:
            valid["provenance"]["llm_note"] = r["text"][:500]
            return {
                "lattice": valid,
                "tier": r["tier"],
                "model": r["model"],
                "offline": r["offline"],
            }
    lattice = _heuristic_lattice(text, motifs, law_id, llm_text=r["text"])
    return {"lattice": lattice, "tier": r["tier"], "model": r["model"], "offline": r["offline"]}


def _radix_levels(by_lvl: dict[str, list[Motif]], ex) -> tuple:
    """Offline radix lattice: value -> base-k digit expansions -> digit ground.

    Executable: R-digits is the terminal rule whose modulus period (2^8 = 256)
    closes the ground space; enumeration yields 256 distinct binary instances.
    """
    levels: list[ScaleLevel] = [
        {
            "level_id": "L2",
            "name": "Positional principle",
            "alphabet": ["v", "Σ", "d_k", "b^k"],
            "description": "One value statement expands into every base-b positional digit vector.",
            "abstraction_score": 0.9,
            "base": "radix pack: the value law packs into parallel base-k numeral systems",
            "base_spec": {
                "radix": None,
                "alphabet": ["v", "d_k", "b^k"],
                "encoding": "v = Σ_k d_k b^k -> [binary | decimal]",
            },
            "modulus": "digit constraint 0 <= d_k < b closes the expansion; carry wraps at b^k",
            "modulus_spec": {
                "period": None,
                "constraint": "digits below radix b",
                "operator": "mod b carry",
            },
            "rules": [
                {
                    "id": "R-value",
                    "name": "POSITIONAL_VALUE",
                    "level": "L2",
                    "pattern": "v = Σ_k d_k b^k",
                    "rewrite": "L1: base-2 and base-10 digit vectors",
                    "examples": ex("L2"),
                    "generative_power": 64.0,
                    "provenance": "synthesizer:L2",
                },
            ],
        },
        {
            "level_id": "L1",
            "name": "Numeral systems",
            "alphabet": ["0", "1", "2", "10"],
            "description": "Binary and decimal encodings of one value, each a positional digit string.",
            "abstraction_score": 0.5,
            "base": "bit radix: each numeral packs into L0 digit symbols",
            "base_spec": {"radix": 2, "alphabet": ["0", "1"], "encoding": "n -> positional digits"},
            "modulus": "byte wrap: the 8-bit window closes the ground space, 2^8 = 256",
            "modulus_spec": {"period": 256, "constraint": "0 <= n < 256", "operator": "mod 256"},
            "rules": [
                {
                    "id": "R-base2",
                    "name": "BASE2",
                    "level": "L1",
                    "pattern": "n -> Σ_k d_k 2^k",
                    "rewrite": "L0: binary digit strings",
                    "examples": ex("L1"),
                    "generative_power": 32.0,
                    "provenance": "synthesizer:L1",
                },
                {
                    "id": "R-base10",
                    "name": "BASE10",
                    "level": "L1",
                    "pattern": "n -> Σ_k d_k 10^k",
                    "rewrite": "L0: decimal digit strings",
                    "examples": ex("L1"),
                    "generative_power": 32.0,
                    "provenance": "synthesizer:L1",
                },
            ],
        },
        {
            "level_id": "L0",
            "name": "Ground digits",
            "alphabet": ["0", "1"],
            "description": "Ground rewrite base: digit symbols addressed by radix and position.",
            "abstraction_score": 0.1,
            "base": "digit radix 2: positional values 0..255 terminate expansion",
            "base_spec": {
                "radix": 2,
                "alphabet": ["0", "1"],
                "encoding": "terminal: (radix, value) -> digits",
            },
            "modulus": "mod 256 equivalence classes (byte wrap)",
            "modulus_spec": {"period": 256, "constraint": "value mod 256", "operator": "mod 256"},
            "rules": [
                {
                    "id": "R-digits",
                    "name": "DIGIT_BASE",
                    "level": "L0",
                    "pattern": "(d_k digits, radix b)",
                    "rewrite": "terminal: positional digit instance",
                    "examples": ex("L0"),
                    "generative_power": 256.0,
                    "provenance": "synthesizer:L0",
                },
            ],
        },
    ]
    expansion = {"R-value": ["R-base2", "R-base10"]}
    return levels, expansion


def _heuristic_lattice(
    text: str, motifs: list[Motif], law_id: str, llm_text: str = ""
) -> ScaleLattice:
    """Offline fallback only. Same shape the LLM must produce, with hand-filled base/modulus."""
    by_lvl: dict[str, list[Motif]] = {}
    for m in motifs:
        by_lvl.setdefault(m.get("candidate_level", "L1"), []).append(m)

    def ex(lvl: str, n: int = 2) -> list[str]:
        return [m["surface_form"] for m in by_lvl.get(lvl, [])[:n]] or [f"ex-{lvl}-{n}"]

    t = text.lower()
    words = set(re.findall(r"[a-z0-9]+", t))
    radix_terms = {
        "radix",
        "numeral",
        "numeral-system",
        "positional",
        "binary",
        "decimal",
        "digit",
        "digits",
        "bit",
        "bits",
        "byte",
        "hexadecimal",
    }
    if words & radix_terms:
        levels, expansion = _radix_levels(by_lvl, ex)
    else:  # default demo domain: 5D electromagnetism / Kaluza-Klein
        levels: list[ScaleLevel] = [
            {
                "level_id": "L3",
                "name": "Unification principle",
                "alphabet": ["S_5", "R̂", "G_AB", "δS=0"],
                "description": "5D vacuum geometry: one statement generating gravity+gauge+scalar.",
                "abstraction_score": 0.95,
                "base": "law-triple radix: one 5D vacuum statement packs into (gravity, gauge, scalar) L2 laws",
                "base_spec": {
                    "radix": 3,
                    "alphabet": ["S_5", "R̂", "G_AB"],
                    "encoding": "R-vac5 -> [R-einstein-4d, R-maxwell, R-dilaton]",
                },
                "modulus": "cylinder condition d5(.)=0 closes the 5th dimension; 15=10+4+1 partition",
                "modulus_spec": {
                    "period": "S1 compact y",
                    "constraint": "d5(.)=0",
                    "operator": "15=10+4+1",
                },
                "rules": [
                    {
                        "id": "R-vac5",
                        "name": "5D_VACUUM",
                        "level": "L3",
                        "pattern": "R̂_AB − ½ R̂ G_AB = 0  (S_5 = ∫ R̂ √−G d⁵x)",
                        "rewrite": "L2: [R-einstein-4d] + [R-maxwell] + [R-dilaton]  with cylinder condition ∂₅=0",
                        "examples": ex("L3"),
                        "generative_power": 64.0,
                        "provenance": "synthesizer:L3",
                    },
                ],
            },
            {
                "level_id": "L2",
                "name": "Field laws",
                "alphabet": ["G_μν", "T_μν", "F_μν", "J^ν", "φ"],
                "description": "4D Einstein + Maxwell + dilaton equations as parallel expansions.",
                "abstraction_score": 0.7,
                "base": "field-triple radix: each law packs into (potential, ansatz) L1 objects",
                "base_spec": {
                    "radix": 2,
                    "alphabet": ["G_μν", "F_μν", "φ"],
                    "encoding": "law -> [R-potential | R-kk-ansatz]",
                },
                "modulus": "Bianchi + conservation close the laws: d_[lF_mn]=0, nabla^mu G_mn=0",
                "modulus_spec": {
                    "period": None,
                    "constraint": "Bianchi + div-free",
                    "operator": "dF=0, nabla.G=0",
                },
                "rules": [
                    {
                        "id": "R-einstein-4d",
                        "name": "EINSTEIN_4D",
                        "level": "L2",
                        "pattern": "G_μν = 8πG T_μν^(EM+φ)",
                        "rewrite": "L1: metric modes g_μν + EM stress F_μαF_ν^α −¼g_μνF² + dilaton ∂φ∂φ",
                        "examples": ex("L2"),
                        "generative_power": 32.0,
                        "provenance": "synthesizer:L2",
                    },
                    {
                        "id": "R-maxwell",
                        "name": "MAXWELL",
                        "level": "L2",
                        "pattern": "∂_μ F^μν = J^ν ,  ∂_[λF_μν]=0",
                        "rewrite": "L1: A_μ plane-wave + Coulomb + dipole instances",
                        "examples": ex("L2"),
                        "generative_power": 32.0,
                        "provenance": "synthesizer:L2",
                    },
                    {
                        "id": "R-dilaton",
                        "name": "DILATON",
                        "level": "L2",
                        "pattern": "□φ = κ F² + ∂V/∂φ",
                        "rewrite": "L1: constant-φ + rolling-φ instances",
                        "examples": ex("L2"),
                        "generative_power": 8.0,
                        "provenance": "synthesizer:L2",
                    },
                ],
            },
            {
                "level_id": "L1",
                "name": "Field objects",
                "alphabet": ["g_μν", "A_μ", "F_μν", "φ", "J^ν"],
                "description": "Potentials, field strengths, currents: the composable middle scale.",
                "abstraction_score": 0.4,
                "base": "KK-triple radix: (g_mn coeffs, A_mu coeffs, phi value) packs geometry into L0 symbols",
                "base_spec": {
                    "radix": 3,
                    "alphabet": ["g_μν", "A_μ", "φ"],
                    "encoding": "G_AB = e^{2a.phi}g + e^{-4a.phi}(dy+kA)^2",
                },
                "modulus": "U(1) gauge periodicity A->A+d.lambda; KK circle y~y+2.pi.R",
                "modulus_spec": {
                    "period": "y ~ y+2piR; A ~ A + d.lambda",
                    "constraint": "gauge equivalence",
                    "operator": "U(1)",
                },
                "rules": [
                    {
                        "id": "R-potential",
                        "name": "POTENTIAL→FIELD",
                        "level": "L1",
                        "pattern": "A_μ  →  F_μν = ∂_μA_ν − ∂_νA_μ",
                        "rewrite": "L0: component matrix [E_i, B_i] wirings",
                        "examples": ex("L1"),
                        "generative_power": 16.0,
                        "provenance": "synthesizer:L1",
                    },
                    {
                        "id": "R-kk-ansatz",
                        "name": "KK_ANSATZ",
                        "level": "L1",
                        "pattern": "G_AB dx^A dx^B = e^{2αφ}g_μνdx^μdx^ν + e^{−4αφ}(dy + κA_μdx^μ)²",
                        "rewrite": "L0: (g_μν coeffs, A_μ coeffs, φ value) triples",
                        "examples": ex("L1"),
                        "generative_power": 24.0,
                        "provenance": "synthesizer:L1",
                    },
                ],
            },
            {
                "level_id": "L0",
                "name": "Symbol base",
                "alphabet": ["x^μ", "y", "∂_μ", "μνρσ", "coeffs", "0/1-like digits"],
                "description": "Ground rewrite base: coordinates, indices, numeric coefficients — the binary of this law.",
                "abstraction_score": 0.15,
                "base": "coordinate-digit radix: charts x (indices x coefficients) terminate expansion",
                "base_spec": {
                    "radix": None,
                    "alphabet": ["x^μ", "y", "coeffs"],
                    "encoding": "terminal: (chart, gauge, phi-value)",
                },
                "modulus": "index symmetries + chart overlap: mu<->nu symmetry, gauge-fix closes redundancy",
                "modulus_spec": {
                    "period": None,
                    "constraint": "index/gauge equivalence",
                    "operator": "symmetrize+fix",
                },
                "rules": [
                    {
                        "id": "R-coords",
                        "name": "COORD_BASE",
                        "level": "L0",
                        "pattern": "(x^μ, y) + index algebra",
                        "rewrite": "terminal: concrete chart instances {(t,x,y,z,y⁵) grid + gauge choice}",
                        "examples": ex("L0"),
                        "generative_power": 128.0,
                        "provenance": "synthesizer:L0",
                    },
                ],
            },
        ]
        expansion = {
            "R-vac5": ["R-einstein-4d", "R-maxwell", "R-dilaton"],
            "R-einstein-4d": ["R-kk-ansatz", "R-potential"],
            "R-maxwell": ["R-potential"],
            "R-dilaton": ["R-kk-ansatz"],
            "R-potential": ["R-coords"],
            "R-kk-ansatz": ["R-coords"],
        }
    return ScaleLattice(
        version="v0",
        law_id=law_id,
        levels=levels,
        order="top(most-compressed) → bottom(ground-base)",
        expansion_map=expansion,
        provenance={
            "synthesizer": r"frontier"
            if not llm_text.startswith("[offline")
            else "offline-heuristic",
            "llm_note": llm_text[:500],
            "motif_count": len(motifs),
        },
        created_at=time.time(),
    )
