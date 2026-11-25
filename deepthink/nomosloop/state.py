"""Top-level LangGraph state schema for the RSI section.

Ontology (see docs/NOMENCLATURE_COMPLEMENT.md):
- law: any generative formal system producing lesser replicas (instantiations).
- scale / base: layered rewrite grammar letting a compact high-level statement
  expand into many consistent low-level instances (binary->decimal, zodiac->charts).
- three domains: AX (the artifact: the policy lattice and its control
  operations), TC (the table of correspondences) and AM (the Agent Mechanics
  lattice of agents).

The fourth open-deepthink section: RSI for cybersecurity, over an
institution's security policies. Library, CLI, and the web UI "RSI Loop" tab.
"""

from __future__ import annotations

import hashlib
import operator
import time
import uuid
from typing import Annotated, Any, TypedDict

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


class NomosState(TypedDict, total=False):
    # identity
    run_id: str
    law_id: str
    law_title: str
    raw_text: str
    law_hash: str
    # artifact domain (AX): the security policy and its scale lattice
    tokens: list[str]
    motifs: list[Motif]
    scale_lattice: ScaleLattice
    lattice_history: list[ScaleLattice]  # every policy version, inspectable
    posture_history: list[dict[str, Any]]  # measured posture per policy version
    controls: list[dict[str, Any]]  # controls kept by the posture gate
    rollbacks: list[dict[str, Any]]  # controls the gate refused
    ops_by_ref: dict[str, dict[str, Any]]  # viable control operations by AX ref
    # lattice (AM) + correspondence (TC): the live three-domain world
    world: Any
    # LoRA loop
    lora_cfg: dict[str, Any]
    lora_state: Any
    lora_buffer: list[dict[str, Any]]
    # artifact
    artifact_path: str
    artifact_code: str
    nx1_path: str
    # loop control
    iteration: int
    max_iterations: int
    rsi_cfg: dict[str, Any]  # drive, lambda_am, sensitivity, budgets, objective
    # inspectability
    routing_log: list[dict[str, Any]]
    topology_log: Annotated[list[str], operator.add]  # each node appends its line
    errors: list[str]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def new_run_state(
    raw_text: str, law_title: str = "untitled-policy", max_iterations: int = 1
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
        posture_history=[],
        controls=[],
        rollbacks=[],
        ops_by_ref={},
        world=None,
        lora_cfg={},
        lora_state=None,
        lora_buffer=[],
        artifact_path="",
        artifact_code="",
        nx1_path="",
        iteration=0,
        max_iterations=max_iterations,
        rsi_cfg={},
        routing_log=[],
        topology_log=[],
        errors=[],
    )
