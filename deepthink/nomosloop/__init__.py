"""
RSI — recursive self-improvement over formal artifacts.

The fourth open-deepthink section. A *law* is any generative formal system that
produces lesser replicas of itself (instantiations): a physics theory, a legal
code, a math model, a code repository, a documentation set. A *scale* (or
*base*) is a layered rewrite grammar that lets one compact high-level statement
expand into many consistent low-level instances — the way binary encodes
decimal, or a zodiac encodes individual charts.

The engine recovers those scales from raw text, exposes them as mutable state,
and evolves them so each generation spawns **more world per symbol** (higher
instantiation capacity).

Public library entry::

    from deepthink.rsi import run_rsi_pipeline, default_rsi_params
    # or: from deepthink import run_rsi

    result = await run_rsi_pipeline(llm, formal_text, iterations=2)

Library + CLI + web UI (the "RSI Loop" tab streams live node events via
``run_rsi_pipeline(on_event=...)``).
"""

from __future__ import annotations

from pathlib import Path

# Load .env BEFORE importing .graph, whose module-level constants
# (NOMOSLOOP_MUTATORS, NOMOSLOOP_ACCEPT, spawn budgets) read os.environ.
from .routing import ROUTING_POLICY, call_llm, clear_llm_bridge, load_dotenv, route, set_llm_bridge
from .state import (
    LATTICE_VERSION,
    InhabitantRecord,
    InstantiationRecord,
    Motif,
    MutationProposal,
    NomosState,
    ScaleLattice,
    ScaleLevel,
    ScaleRule,
    new_run_state,
)

load_dotenv()
_repo_env = Path(__file__).resolve().parents[2] / ".env"
if _repo_env.is_file():
    load_dotenv(str(_repo_env))

from .graph import (  # noqa: E402  (import after load_dotenv by design)
    TOPOLOGY,
    build_graph,
    describe_topology,
    mutate_topology,
    run_pipeline,
    set_observer,
    set_topology,
)
from .mesh import MeshWorld  # noqa: E402
from .pipeline import default_rsi_params, run_rsi_pipeline, snapshot_state  # noqa: E402
from .state import hash_text  # noqa: E402

__all__ = [
    # State
    "NomosState",
    "ScaleRule",
    "ScaleLevel",
    "ScaleLattice",
    "Motif",
    "MutationProposal",
    "InstantiationRecord",
    "InhabitantRecord",
    "LATTICE_VERSION",
    "new_run_state",
    "hash_text",
    # Routing
    "route",
    "call_llm",
    "load_dotenv",
    "ROUTING_POLICY",
    "set_llm_bridge",
    "clear_llm_bridge",
    # Engine
    "MeshWorld",
    "build_graph",
    "run_pipeline",
    "TOPOLOGY",
    "describe_topology",
    "mutate_topology",
    "set_topology",
    # Pipeline / API
    "run_rsi_pipeline",
    "default_rsi_params",
    "snapshot_state",
    "set_observer",
]
