"""Domain profiles for Theory Synthesis.

A profile tunes the prompts for one kind of problem: what a solver must state,
which checks the verifier runs, which seed words shape the personas, and how
close two numeric answers must be to count as one vote.
"""

from __future__ import annotations

import re
from typing import Any

DOMAINS: dict[str, dict[str, Any]] = {
    "math": {
        "label": "Mathematics",
        "brief": (
            "State the notation you use. Justify every step from definitions, given facts or "
            "results proved in earlier cells. Prefer exact values to decimals."
        ),
        "checks": [
            "each step follows from the previous ones",
            "edge and degenerate cases",
            "small cases or numeric spot checks",
            "the final answer answers the question that was asked",
        ],
        "seed_verbs": "induct, bound, telescope, factor, substitute, linearize, iterate",
        "seed_nouns": "invariant, lattice, residue, eigenvalue, recurrence, symmetry, extremum",
        "skill_examples": "induction, generating functions, inequalities, linear algebra, modular arithmetic",
        "framing_examples": (
            "algebraic versus geometric, generating functions, an invariant, the extremal case, "
            "a change of variables"
        ),
        "answer_hint": "a number, an expression, a set, yes/no, or 'proof'",
        "tolerance": 1e-9,
        "archetypes": [
            "Algebraist",
            "Analyst",
            "Combinatorialist",
            "Number theorist",
            "Geometer",
            "Probabilist",
        ],
        "skills": [
            "induction",
            "generating functions",
            "inequalities",
            "linear algebra",
            "modular arithmetic",
            "bijective counting",
            "telescoping sums",
            "case analysis",
        ],
    },
    "olympiad": {
        "label": "Olympiad (one IMO-style problem instance)",
        "brief": (
            "Treat this as one olympiad problem instance and write at contest rigor. Test small "
            "cases before generalizing. For 'find all' problems prove both directions: every "
            "listed answer works, and nothing else does. For inequalities, find the equality case."
        ),
        "checks": [
            "small cases agree with the claimed answer",
            "for 'find all': each answer is verified and all others are excluded",
            "for inequalities: the equality case is found and attained",
            "constructions are checked explicitly",
            "no step uses an unproven 'well-known' claim without stating it",
        ],
        "seed_verbs": "invert, color, pigeonhole, jump, telescope, induct, smooth, bound",
        "seed_nouns": "invariant, extremum, parity, residue, incircle, bijection, potential, monovariant",
        "skill_examples": (
            "invariants and monovariants, the extremal principle, pigeonhole, Vieta jumping, "
            "lifting the exponent, inversion, Cauchy-Schwarz, AM-GM"
        ),
        "framing_examples": (
            "an invariant or monovariant, the extremal principle, a coloring, complex numbers or "
            "coordinates, a bijection, working modulo a small prime"
        ),
        "answer_hint": "the answer in contest form (number, set, expression, yes/no), or 'proof'",
        "tolerance": 1e-9,
        "archetypes": [
            "Olympiad number theorist",
            "Olympiad combinatorialist",
            "Olympiad geometer",
            "Inequality specialist",
            "Functional-equation solver",
            "Invariant hunter",
        ],
        "skills": [
            "invariants and monovariants",
            "extremal principle",
            "pigeonhole principle",
            "Vieta jumping",
            "lifting the exponent",
            "inversion and projective tools",
            "AM-GM and Cauchy-Schwarz",
            "substitution in functional equations",
        ],
    },
    "algorithms": {
        "label": "Algorithm design",
        "brief": (
            "Give the algorithm precisely (pseudocode where useful), a correctness argument "
            "(invariant, exchange argument or reduction) and its time and space complexity. "
            "Consider worst-case inputs. Claim optimality only with a lower-bound argument."
        ),
        "checks": [
            "the correctness argument covers every case",
            "the complexity bound holds for the stated algorithm",
            "edge cases: empty input, duplicates, extremes, overflow",
            "a lower bound supports any optimality claim",
        ],
        "seed_verbs": "divide, memoize, sweep, relax, augment, amortize, prune, reduce",
        "seed_nouns": "heap, segment-tree, flow, matroid, invariant, frontier, potential, lattice",
        "skill_examples": (
            "dynamic programming, greedy exchange arguments, graph algorithms, data structures, "
            "amortized analysis, reductions"
        ),
        "framing_examples": (
            "a graph model, dynamic programming over a clever state, a greedy exchange argument, "
            "a reduction to a known problem, a potential function"
        ),
        "answer_hint": "the complexity of the best algorithm found, for example O(n log n)",
        "tolerance": 0.0,
        "archetypes": [
            "Graph algorithmist",
            "Dynamic-programming designer",
            "Data-structure engineer",
            "Complexity theorist",
            "Greedy-method analyst",
            "Randomized-algorithms designer",
        ],
        "skills": [
            "dynamic programming",
            "greedy exchange arguments",
            "shortest paths and flows",
            "segment trees and heaps",
            "amortized analysis",
            "reductions and lower bounds",
            "divide and conquer",
            "two pointers and sweeps",
        ],
    },
    "physics": {
        "label": "Physics",
        "brief": (
            "State the model and its assumptions. Use conservation laws and symmetry. Carry units "
            "through every step, check dimensions, and test limiting cases."
        ),
        "checks": [
            "dimensional analysis of every result",
            "limiting and special cases",
            "conservation laws",
            "order of magnitude",
            "sign conventions",
        ],
        "seed_verbs": "conserve, linearize, perturb, integrate, quantize, average, scale",
        "seed_nouns": "Lagrangian, flux, potential, boundary, symmetry, dimension, oscillator",
        "skill_examples": (
            "Lagrangian mechanics, conservation laws, dimensional analysis, perturbation theory, "
            "Gauss's law, statistical mechanics"
        ),
        "framing_examples": (
            "an energy method instead of forces, a symmetry argument, a change of reference frame, "
            "dimensional analysis first, a limiting case"
        ),
        "answer_hint": "a value with units, or a closed-form expression",
        "tolerance": 1e-3,
        "archetypes": [
            "Classical mechanician",
            "Electrodynamicist",
            "Statistical physicist",
            "Quantum theorist",
            "Fluid dynamicist",
            "Experimental estimator",
        ],
        "skills": [
            "Lagrangian mechanics",
            "conservation laws",
            "dimensional analysis",
            "perturbation theory",
            "Gauss's law",
            "statistical mechanics",
            "Fourier methods",
            "order-of-magnitude estimates",
        ],
    },
}

DOMAIN_NAMES = tuple(DOMAINS)

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "olympiad": (
        "imo",
        "olympiad",
        "find all",
        "prove that",
        "positive integers",
        "integers n",
        "triangle abc",
    ),
    "algorithms": (
        "algorithm",
        "complexity",
        "array",
        "o(n",
        "time limit",
        "data structure",
        "queries",
        "shortest path",
        "graph with",
    ),
    "physics": (
        "velocity",
        "acceleration",
        "mass",
        "energy",
        "momentum",
        "electric",
        "magnetic",
        "field",
        "particle",
        "friction",
        "temperature",
        "quantum",
        "m/s",
    ),
}


def guess_domain(problem: str) -> str:
    """Keyword guess used when ``domain="auto"`` and no sizer answer is available."""
    text = (problem or "").lower()
    best, best_hits = "math", 0
    for name, words in _KEYWORDS.items():
        hits = sum(1 for w in words if w in text)
        if hits > best_hits:
            best, best_hits = name, hits
    return best


def resolve_domain(name: Any, problem: str = "") -> str:
    """Return a known domain name; ``auto`` or unknown names fall back to a keyword guess."""
    key = str(name or "auto").strip().lower()
    aliases = {
        "imo": "olympiad",
        "competition": "olympiad",
        "algorithm": "algorithms",
        "cs": "algorithms",
    }
    key = aliases.get(key, key)
    if key in DOMAINS:
        return key
    return guess_domain(problem)


def domain_profile(name: str) -> dict[str, Any]:
    return DOMAINS.get(name, DOMAINS["math"])


_PROOF_RE = re.compile(r"\b(prove|show)\s+that\b", re.IGNORECASE)


def guess_answer_type(problem: str, domain: str) -> str:
    """Cheap answer-type guess: proofs vote by verification, everything else by value."""
    text = problem or ""
    asks_value = re.search(
        r"\b(find|determine|compute|calculate|evaluate|how many|what is)\b", text, re.IGNORECASE
    )
    if _PROOF_RE.search(text) and not asks_value:
        return "proof"
    if domain == "algorithms":
        return "complexity"
    return "other"
