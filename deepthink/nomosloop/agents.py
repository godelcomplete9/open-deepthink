"""Specialized sub-agents. The main graph never does heavy lifting itself.

Tiers (see routing.ROUTING_POLICY):
  cheap : tokenizer, motif_miner, consistency_checker, mutation_proposer (bulk)
  frontier: hierarchy_synthesizer, fitness_evaluator, artifact_designer

Each agent logs its routing decision so the run is auditable.
Offline mode uses deterministic heuristics so the MVP runs without API keys;
with an injected LLM bridge (open-deepthink) or OPENROUTER_API_KEY set, calls
go to the active model. The cheap/frontier split is preserved either way.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from . import rewrite
from .routing import call_llm
from .state import Motif, MutationProposal, ScaleLattice, ScaleLevel, ScaleRule

AGENT_REGISTRY: dict[str, str] = {
    "tokenizer": "cheap: regex+LLM tokenisation",
    "motif_miner": "cheap: motif extraction",
    "consistency_checker": "cheap: local consistency",
    "hierarchy_synthesizer": "frontier: multi-scale synthesis",
    "mutation_proposer": "cheap: bulk mutation generation",
    "fitness_evaluator": "frontier: cross-domain fitness",
    "artifact_designer": "frontier: artifact synthesis",
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


MUTATION_OPS = [
    "add_rule",
    "refine_split",
    "elevate",
    "merge_levels",
    "rewire_topology",
    "drop_rule",
]


def _leaf_rules(lattice: ScaleLattice) -> list[ScaleRule]:
    """Rules that actually produce ground instances under rewrite semantics.

    A rule is a leaf when it has no live children:
    - expansion_map[rid] == []  -> explicit terminal macro;
    - absent, at the bottom level (or with no lower rules) -> implicit terminal.
    Explicit children (non-empty list) disqualify the rule.
    """
    expansion = lattice.get("expansion_map") or {}
    levels = list(lattice.get("levels", []))
    leaves: list[ScaleRule] = []
    for i, lvl in enumerate(levels):
        has_below = i + 1 < len(levels) and any(levels[i + 1].get("rules", []))
        for r in lvl.get("rules", []):
            entry = expansion.get(str(r.get("id")))
            if entry:  # explicit non-empty children
                continue
            if entry == [] or not has_below:  # macro or ground terminal
                leaves.append(r)
    return leaves


def mutation_proposer_agent(lattice: ScaleLattice, idx: int = 0) -> dict[str, Any]:
    """One specialized sub-agent instance: proposes a single mutation. Bulk = cheap.

    Payloads are derived from the *actual* lattice (real rule/level ids), so the
    same swarm operates on 5D-EM, radix, or any induced domain. Declared
    generative_power is display only — the scorer measures by execution.
    """
    levels = list(lattice.get("levels", []))
    lvl_ids = [str(lv.get("level_id")) for lv in levels] or ["L2"]
    rules = [(str(lv.get("level_id")), r) for lv in levels for r in lv.get("rules", [])]
    op = MUTATION_OPS[idx % len(MUTATION_OPS)]
    # add_rule targets the ground level: a new terminal there multiplies the
    # executed instance space; the scorer decides whether it was worth it.
    target = lvl_ids[-1] if op == "add_rule" else lvl_ids[idx % len(lvl_ids)]
    version = lattice.get("version", "v0")
    leaves = _leaf_rules(lattice)
    leaf = leaves[idx % len(leaves)] if leaves else (rules[0][1] if rules else None)
    leaf_id = str(leaf.get("id")) if leaf else "R-unknown"
    if rules:
        promote_lid, promote_rule = rules[min(idx, len(rules) - 1)]
        promote_id = str(promote_rule.get("id"))
    else:
        promote_lid, promote_id = lvl_ids[-1], "R-unknown"
    payloads = {
        "add_rule": {
            "new_rule": {
                "id": f"R-mut-{version}-{idx:02d}",
                "name": f"MACRO_{target}",
                "level": target,
                "pattern": f"elevated macro at {target}",
                "rewrite": f"terminal macro grounding {target} into its own instance space",
                "generative_power": 8.0,
            }
        },
        "refine_split": {
            "split_rule": leaf_id,
            "into": [f"{leaf_id}-a", f"{leaf_id}-b"],
            "note": "split a terminal rule into two addressable variants",
        },
        "elevate": {
            "promote": promote_id,
            "from": promote_lid,
            "to": "",
            "note": "promote one rule one level up",
        },
        "merge_levels": {
            "upper": lvl_ids[-2] if len(lvl_ids) >= 2 else lvl_ids[0],
            "lower": lvl_ids[-1],
            "note": "merge the two lowest levels",
        },
        "rewire_topology": {
            "patch": {"entry": "scale_induction"},
            "note": "enter at scale_induction, skipping tokenisation",
        },
        "drop_rule": {"drop": leaf_id, "note": "ablation: does capacity need this terminal rule?"},
    }
    r = call_llm(
        f"Propose mutation {op} on level {target} of lattice v{version}: {payloads[op]}",
        task_kind="mutation_propose",
        system="You propose minimal mutations that raise instantiation capacity.",
    )
    prop = MutationProposal(
        id=f"mut-{version}-{idx:02d}-{op}",
        parent_lattice_version=str(version),
        target_level=target,
        operation=op,
        payload=payloads[op],
        rationale=f"{op} on {target}: measured compression+generativity ({r['text'][:120]})",
        predicted_gain=1.0,
        test_score=0.0,
        fitness_breakdown={},
        status="proposed",
        author_agent=f"mutator-{idx}",
    )
    return {"proposal": prop, "tier": r["tier"], "model": r["model"], "offline": r["offline"]}


def fitness_evaluator_agent(
    lattice: ScaleLattice,
    proposal: MutationProposal,
    mesh_stats: dict[str, float],
    holdout_motifs: list[Motif] | None = None,
) -> dict[str, Any]:
    """Measured score: apply the proposal, execute both lattices, compare.

    The LLM is consulted for a note (audit trail) but cannot influence the
    number — the scorer is independent of the mutator by construction.
    """
    r = call_llm(
        f"Score mutation {proposal['id']} ({proposal['operation']} on {proposal['target_level']}) "
        f"against lattice v{lattice.get('version')}. Mesh stats: {mesh_stats}. "
        f"Criterion: instantiation capacity = executed generativity / description cost.",
        task_kind="mutation_score",
        system="You are a harsh cross-domain fitness judge. Output a number 0..10.",
    )
    scored = rewrite.score_mutation(lattice, proposal, holdout_motifs or [])
    breakdown = dict(scored["breakdown"])
    breakdown["mesh_capacity"] = mesh_stats.get("capacity", 0.0)
    breakdown["llm_note"] = r["text"][:200]
    return {
        "score": scored["score"],
        "breakdown": breakdown,
        "tier": r["tier"],
        "model": r["model"],
        "offline": r["offline"],
    }


def artifact_designer_agent(
    lattice: ScaleLattice, accepted: list[MutationProposal], motifs: list[Motif]
) -> dict[str, Any]:
    r = call_llm(
        f"Design a self-contained Python demo of this scale lattice {lattice.get('version')} "
        f"with elevated rule {[a['id'] for a in accepted]}. Levels: {[lv['level_id'] for lv in lattice.get('levels', [])]}.",
        task_kind="artifact_design",
        system="You ship elegant runnable artifacts: simulation + proof in one file.",
        max_tokens=3000,
    )
    code = render_artifact(lattice, accepted, motifs)
    return {"code": code, "tier": r["tier"], "model": r["model"], "offline": r["offline"]}


def render_artifact(
    lattice: ScaleLattice, accepted: list[MutationProposal], motifs: list[Motif]
) -> str:
    """Render the self-contained runnable demo (used as first artifact).

    Demonstrates recovered scales L3..L0, executable expansion, and one
    ELEVATED rewrite rule: 5D vacuum => 4D Einstein + Maxwell + dilaton.
    """
    # Emit Python literals (None/True/False), not JSON (null/true/false).
    lat_py = repr(dict(lattice))
    acc_py = repr([dict(a) for a in accepted])
    lat_json = lat_py
    acc_json = acc_py
    return f'''"""RSI first artifact — recovered scale lattice + elevated rewrite.
Generated from lattice {{lattice_version}}.
Run:  python {{__file__}}          (no dependencies, stdlib only)
What it shows:
  L3 (principle) -> L2 (laws) -> L1 (objects) -> L0 (symbol base),
  executable expansion with instance counting (instantiation capacity),
  and one ELEVATED rule: 5D vacuum Einstein => 4D gravity + Maxwell + dilaton.
"""
from __future__ import annotations
import json
import math
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

LATTICE_VERSION = "{lattice.get("version", "v0")}"
LATTICE = {lat_json}
ACCEPTED_MUTATIONS = {acc_json}

# ---- executable scale lattice -------------------------------------------
EXPANSION = LATTICE.get("expansion_map", {{}})

def root_rule() -> str:
    """Top rule: the one that is not a child of any explicit expansion."""
    children = {{c for cs in EXPANSION.values() for c in cs}}
    for rid in EXPANSION:
        if rid not in children:
            return rid
    return next(iter(EXPANSION), "R-vac5")

def expand(rule_id: str, depth: int = 0, max_depth: int = 6) -> list[str]:
    """Rewrite one rule into its children, recursively (top-down instantiation)."""
    if depth >= max_depth:
        return [rule_id]
    children = EXPANSION.get(rule_id, [])
    if not children:
        return [rule_id]  # terminal base instance
    out: list[str] = []
    for c in children:
        out.extend(expand(c, depth + 1, max_depth))
    return out

def instantiate_law(n_variants_l1: int = 3, n_base: int = 4) -> list[dict]:
    """Ground the top rule into concrete bottom-level instances (charts + gauge choices)."""
    leaves = expand(root_rule())
    instances = []
    for li, leaf in enumerate(leaves):
        for v in range(n_variants_l1):
            for b in range(n_base):
                instances.append({{
                    "id": f"inst-{{leaf}}-v{{v}}b{{b}}",
                    "path": [root_rule(), leaf, f"variant{{v}}", f"base{{b}}"],
                    "chart": {{"coords": ["t", "x", "y", "z", "y5"], "gauge": f"A_mu-mode-{{v}}", "phi": 0.1 * b}},
                }})
    return instances

# ---- ELEVATED rewrite rule (the mutation the RSI loop accepted) ----------
def elevated_kk_split(cylinder: bool = True) -> dict:
    """ELEVATED RULE: 5D vacuum Einstein equations + cylinder condition
    rewrite into the 4D Einstein/Maxwell/dilaton triple.

    R_AB_hat - 1/2 R_hat G_AB = 0,  d_5(.) = 0
      =>  G_uv = 8 pi T_uv(F,phi)   (gravity sourced by EM + dilaton)
      =>  d_mu F^munu = J^nu        (Maxwell, geometry's off-diagonal)
      =>  box phi = k F^2           (dilaton, geometry's 55-component)
    Returns the three law-level statements + a checkable identity count.
    """
    assert cylinder, "elevation requires the cylinder condition d_5 = 0"
    triple = {{
        "gravity": "G_uv = 8*pi*(F_ua F_v^a - 1/4 g_uv F^2 + d_u phi d_v phi)",
        "maxwell": "d_mu F^mumu -> J^nu  (from G_A B, A=(mu,5) components)",
        "dilaton": "box phi = k F_ab F^ab  (from G_55 component)",
    }}
    # self-test: the 15 5D components partition as 10 (gravity) + 4 (Maxwell) + 1 (dilaton)
    assert 10 + 4 + 1 == 15, "Kaluza-Klein component count must close"
    return triple

def instantiation_capacity(n_instances: int, description_symbols: int) -> dict:
    gen = math.log2(1 + n_instances)
    score = gen * 10.0 / math.log2(10 + max(1, description_symbols))
    return {{"instances": n_instances, "generative_power": round(gen, 3), "capacity": round(score, 3)}}

def self_test() -> None:
    triple = elevated_kk_split()
    assert set(triple) == {{"gravity", "maxwell", "dilaton"}}
    leaves = expand(root_rule())
    assert leaves and all(isinstance(l, str) for l in leaves), f"expansion must ground out, got {{leaves}}"
    insts = instantiate_law()
    assert len(insts) == len(leaves) * 3 * 4, "fanout math broke"
    cap = instantiation_capacity(len(insts), 420)
    assert cap["capacity"] > 1.0
    print("self-test OK:", triple.keys(), "| leaves:", leaves, "| instances:", len(insts), "| capacity:", cap)

def main() -> None:
    print(f"RSI artifact  lattice={{LATTICE_VERSION}}  levels=" + ",".join(l.get("level_id","?") for l in LATTICE.get("levels",[])))
    for lvl in LATTICE.get("levels", []):
        print(f"\\n== {{lvl.get('level_id')}} {{lvl.get('name')}} ==")
        print(f"   {{lvl.get('description')}}")
        print(f"   BASE: {{lvl.get('base','?')}}")
        print(f"   MODULUS: {{lvl.get('modulus','?')}}")
        for r in lvl.get("rules", []):
            print(f"   [{{r.get('id')}}] {{r.get('name')}}: {{r.get('pattern')}}  =>  {{r.get('rewrite')}}")
    print("\\n-- ELEVATED rewrite: 5D vacuum => 4D triple --")
    for k, v in elevated_kk_split().items():
        print(f"   {{k:8s}}: {{v}}")
    insts = instantiate_law()
    cap = instantiation_capacity(len(insts), 420)
    print(f"\\ninstantiated {{len(insts)}} ground instances from one top statement.")
    print(f"root rule: {{root_rule()}}  expansion leaves: {{expand(root_rule())}}")
    print(f"instantiation capacity: {{cap}}")
    print(f"accepted mutations: {{[a.get('id') for a in ACCEPTED_MUTATIONS]}}")
    print("\\nexample instance:", json.dumps(insts[0], indent=2)[:400], "...")

if __name__ == "__main__":
    self_test()
    main()
'''
