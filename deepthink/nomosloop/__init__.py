"""
RSI: recursive self-improvement for cybersecurity.

The fourth open-deepthink section. The artifact is an institution's security
policy, read as a *law*: a generative formal system whose clauses expand from
principles into policies, controls and assets. The loop runs on **Agent
Mechanics** (the treatise in ``docs/``, with the NX-1 nomenclature complement
in ``docs/NOMENCLATURE_COMPLEMENT.md``):

* every clause, modulus and viable control operation becomes an agent,
  through a table of correspondences built by a mapper model;
* the agents tick under the treatise's tick law. Their clashes, ensembles
  and ranks are written in the NX-1 nomenclature;
* **Leak** (defensive): a leaking agent heals in the lattice with a licensed
  operator. The heal is pulled back through the symbolic gauge coderivative
  into an inspection finding;
* **Stock** (evolving): an agent with a surplus proposes a control clause,
  kept only when the measured posture improves;
* a **LoRA** loop distils the traces and clamps the adapter into the local
  driver model when the loop improves.

Public library entry::

    from deepthink.nomosloop import run_rsi_pipeline, default_rsi_params
    # or: from deepthink import run_rsi

    result = await run_rsi_pipeline(local_llm, policy_text, iterations=2,
                                    params={"drive": "leak"})
"""

from __future__ import annotations

from pathlib import Path

from .routing import (
    ROUTING_POLICY,
    call_llm,
    clear_llm_bridge,
    is_local_llm,
    load_dotenv,
    route,
    set_llm_bridge,
    set_tier_bridge,
)
from .state import (
    LATTICE_VERSION,
    Motif,
    NomosState,
    ScaleLattice,
    ScaleLevel,
    ScaleRule,
    hash_text,
    new_run_state,
)

load_dotenv()
_repo_env = Path(__file__).resolve().parents[2] / ".env"
if _repo_env.is_file():
    load_dotenv(str(_repo_env))

from .graph import TOPOLOGY, build_graph, describe_topology, run_pipeline, set_observer  # noqa: E402
from .pipeline import DRIVES, default_rsi_params, run_rsi_pipeline, snapshot_state  # noqa: E402
from .world import World, drive_config  # noqa: E402

__all__ = [
    # State
    "NomosState",
    "ScaleRule",
    "ScaleLevel",
    "ScaleLattice",
    "Motif",
    "LATTICE_VERSION",
    "new_run_state",
    "hash_text",
    # Routing
    "route",
    "call_llm",
    "load_dotenv",
    "ROUTING_POLICY",
    "set_llm_bridge",
    "set_tier_bridge",
    "clear_llm_bridge",
    "is_local_llm",
    # Engine
    "World",
    "drive_config",
    "DRIVES",
    "build_graph",
    "run_pipeline",
    "TOPOLOGY",
    "describe_topology",
    "set_observer",
    # Pipeline / API
    "run_rsi_pipeline",
    "default_rsi_params",
    "snapshot_state",
]
