"""Top-level LangGraph state schema for the RSI section.

Ontology:
- law: any generative formal system producing lesser replicas (instantiations).
- scale / base: layered rewrite grammar letting a compact high-level statement
  expand into many consistent low-level instances (binary->decimal, zodiac->charts).
- lattice: versioned, first-class, mutable ScaleLattice in state.
- mesh: world-graph of lesser/higher instantiations that agents inhabit.

The graph topology, routing policy, fitness fn, and artifact generators are
all legitimate RSI mutation targets — hence everything is explicit data.

This is the fourth open-deepthink section: RSI over formal artifacts (code
repos, math models, physics models, documentation) that follow formal
structures. It is API-only (no web UI wiring).
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, TypedDict

LATTICE_VERSION = "lattice/v0"

# ---------------------------------------------------------------- schemas ---


class ScaleRule(TypedDict, total=False):
    id: str
    name: str
    level: str  # level_id this rule lives at
    pattern: str  # what it matches (compact high-level form)
    rewrite: str  # what it expands into (lower-level form)
    examples: list[str]
    generative_power: float  # estimated # of distinct instances it can spawn
    provenance: str  # motif ids / agent id that produced it


class ScaleLevel(TypedDict, total=False):
    level_id: str  # e.g. "L0", "L1", ...
    name: str
    alphabet: list[str]  # symbols valid at this scale
    description: str
    rules: list[ScaleRule]
    abstraction_score: float  # 0..1, higher = more compressed
    # --- base / modulus (LLM-identified, see agents.hierarchy_synthesizer_agent)
    base: str  # rewrite radix: how this level packs/addresses the level below
    # e.g. "binary-like (x^mu,y)+coeffs", "KK-triple (g,A,phi)"
    base_spec: dict[str, Any]  # {radix: int|None, alphabet: [...], encoding: str}
    modulus: str  # cycle / wrap / constraint that closes this scale
    # e.g. "cylinder d5=0", "U(1) gauge periodicity", "15=10+4+1 partition"
    modulus_spec: dict[str, Any]  # {period: str|None, constraint: str, operator: str}


class ScaleLattice(TypedDict, total=False):
    version: str  # "v0", "v1", ... bumped on every elevation
    law_id: str
    levels: list[ScaleLevel]  # ordered high -> low or low -> high (see `order`)
    order: str  # "L3..L0 top-down" convention note
    expansion_map: dict[str, list[str]]  # rule_id -> [child rule_ids / level_ids]
    provenance: dict[str, Any]  # how each level was induced
    created_at: float


class Motif(TypedDict, total=False):
    id: str
    surface_form: str
    normalized: str
    frequency: int
    contexts: list[str]
    candidate_level: str  # provisional L0..L3 guess
    kind: str  # symbol | object | equation | principle | meta


class MutationProposal(TypedDict, total=False):
    id: str
    parent_lattice_version: str
    target_level: str
    operation: str  # add_rule | refine_split | merge_levels | elevate | drop_rule
    payload: dict[str, Any]
    rationale: str
    predicted_gain: float
    test_score: float  # filled by fitness node
    fitness_breakdown: dict[str, float]
    status: str  # proposed | tested | accepted | rejected
    author_agent: str


class InstantiationRecord(TypedDict, total=False):
    id: str
    scale_level: str
    content: dict[str, Any]  # concrete low-level instance data
    parent_law_hash: str
    parent_rule: str
    fitness: float
    children_ids: list[str]


class InhabitantRecord(TypedDict, total=False):
    id: str  # agent id, e.g. "inh-<node8>"
    node_id: str  # mesh node this agent inhabits
    level: str  # scale level of its node (L3..L0)
    rule_id: str  # rule it embodies
    law_id: str
    lattice_version: str  # loop seed: which lattice version it carries
    generation: int  # 0 = mesh-spawned, +1 per spawn hop
    budget: int  # remaining spawns it may perform
    offspring: list[str]  # child inhabitant ids it spawned


class NomosState(TypedDict, total=False):
    # identity
    run_id: str
    law_id: str
    law_title: str
    raw_text: str
    law_hash: str
    # pipeline products
    tokens: list[str]
    motifs: list[Motif]
    scale_lattice: ScaleLattice
    lattice_history: list[ScaleLattice]  # every version, inspectable
    holdout_motifs: list[Motif]  # held out from induction; coverage gate
    capacity_history: list[dict[str, Any]]  # measured capacity per lattice version
    baseline_capacity: float  # measured capacity of the induced v0
    rollbacks: list[dict[str, Any]]  # elevations/topology patches undone
    topology_history: list[dict[str, Any]]  # topology patches the loop attempted
    mutation_proposals: list[MutationProposal]
    accepted_mutations: list[MutationProposal]
    # mesh world (serializable snapshot; live MeshWorld rebuilt from it)
    mesh_nodes: list[InstantiationRecord]
    mesh_edges: list[list[str]]  # [src, dst, relation]
    inhabitants: list[InhabitantRecord]  # loop-carrying agents inhabiting mesh nodes
    spawn_log: list[dict[str, Any]]  # every peer/child instantiation event
    fitness_report: dict[str, Any]
    # artifact
    artifact_path: str
    artifact_code: str
    # loop control
    iteration: int
    max_iterations: int
    n_mutators: int  # gen-0 swarm size for this run (overrides env default)
    # inspectability
    routing_log: list[dict[str, Any]]
    topology_log: list[str]
    errors: list[str]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def new_run_state(
    raw_text: str, law_title: str = "untitled-law", max_iterations: int = 1
) -> NomosState:
    return NomosState(
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        law_id=f"law-{hash_text(raw_text)}",
        law_title=law_title,
        raw_text=raw_text,
        law_hash=hash_text(raw_text),
        tokens=[],
        motifs=[],
        scale_lattice=ScaleLattice(
            version="v-none",
            law_id="",
            levels=[],
            order="",
            expansion_map={},
            provenance={},
            created_at=time.time(),
        ),
        lattice_history=[],
        holdout_motifs=[],
        capacity_history=[],
        baseline_capacity=0.0,
        rollbacks=[],
        topology_history=[],
        mutation_proposals=[],
        accepted_mutations=[],
        mesh_nodes=[],
        mesh_edges=[],
        inhabitants=[],
        spawn_log=[],
        fitness_report={},
        artifact_path="",
        artifact_code="",
        iteration=0,
        max_iterations=max_iterations,
        n_mutators=-1,  # -1 => use graph.N_MUTATORS_GEN0 (env default)
        routing_log=[],
        topology_log=[],
        errors=[],
    )
