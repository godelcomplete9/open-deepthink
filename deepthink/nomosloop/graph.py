"""LangGraph orchestration substrate for the RSI engine. Every
node/edge/state is explicit, inspectable, and RSI-mutable (see TOPOLOGY +
mutate_topology).

Nodes:
  ingest            (cheap tokenizer sub-agent)
  scale_induction   (cheap motif miner + frontier hierarchy synthesizer)
  mutation_proposal (spawn N cheap mutator sub-agents, gen-0)
  mesh_spawn        (materialise MeshWorld: the world agents inhabit)
  fitness           (frontier scoring of mutations by *measured* capacity)
  artifact_emit     (frontier artifact designer -> runnable .py demo)
  rectify           (apply best mutation -> lattice v1, with keep/rollback gate)

Edges (v0): ingest -> scale_induction -> mutation_proposal -> mesh_spawn
            -> fitness -> artifact_emit -> rectify -> END
  rectify may route back to mutation_proposal while iteration < max_iterations.

Keep/rollback gate (rectify):
- lattice mutations are applied, then measured (executed enumeration + holdout
  motif coverage); kept only if capacity does not regress and coverage holds;
- rewire_topology proposals patch TOPOLOGY at runtime, but only if the patch is
  acyclic, references known nodes, and the graph still compiles; otherwise the
  previous topology is restored. Patches take effect on the next build_graph().
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from . import agents, rewrite
from . import inhabitants as spawn
from .mesh import MeshWorld
from .mutations import apply_mutation
from .state import NomosState, new_run_state

# Explicit, inspectable topology table. RSI loop may patch this at runtime.
TOPOLOGY: dict[str, Any] = {
    "nodes": [
        "ingest",
        "scale_induction",
        "mutation_proposal",
        "mesh_spawn",
        "fitness",
        "artifact_emit",
        "rectify",
    ],
    "edges": [
        ["ingest", "scale_induction"],
        ["scale_induction", "mutation_proposal"],
        ["mutation_proposal", "mesh_spawn"],
        ["mesh_spawn", "fitness"],
        ["fitness", "artifact_emit"],
        ["artifact_emit", "rectify"],
    ],
    "entry": "ingest",
    "conditional": {"rectify": {"to_mutation_proposal_if": "iteration < max_iterations else END"}},
    "version": "graph/v0",
}

N_MUTATORS_GEN0 = int(os.environ.get("NOMOSLOOP_MUTATORS", "6"))
FITNESS_ACCEPT_THRESHOLD = float(os.environ.get("NOMOSLOOP_ACCEPT", "5.0"))
SPAWN_BUDGET = int(os.environ.get("NOMOSLOOP_SPAWN_BUDGET", "2"))
SPAWN_PEERS = int(os.environ.get("NOMOSLOOP_SPAWN_PEERS", "2"))
SPAWN_CHILDREN_CAP = int(os.environ.get("NOMOSLOOP_SPAWN_CHILDREN_CAP", "8"))


# Optional live observer (GUI / telemetry). Called as
#   observer(phase, node_name, state, update)
# with phase "enter" (update=None) or "exit" (update = the node's returned dict).
# Observers must never break the loop: exceptions are swallowed.
NodeObserver = Callable[[str, str, NomosState, "dict[str, Any] | None"], None]
_OBSERVER: NodeObserver | None = None


def set_observer(fn: NodeObserver | None) -> None:
    """Install (or clear with ``None``) the live node observer."""
    global _OBSERVER
    _OBSERVER = fn


def _notify(phase: str, name: str, state: NomosState, update: dict[str, Any] | None) -> None:
    obs = _OBSERVER
    if obs is None:
        return
    try:
        obs(phase, name, state, update)
    except Exception:  # noqa: BLE001 - observers are best-effort
        pass


def _observed(name: str, fn: Callable[[NomosState], dict[str, Any]]):
    def wrapped(state: NomosState) -> dict[str, Any]:
        _notify("enter", name, state, None)
        out = fn(state)
        _notify("exit", name, state, out)
        return out

    wrapped.__name__ = getattr(fn, "__name__", name)
    return wrapped


def _log(state: NomosState, tier: str, model: str, agent: str, task: str, offline: bool) -> None:
    state.setdefault("routing_log", []).append(
        {"agent": agent, "task": task, "tier": tier, "model": model, "offline": offline}
    )


# -- nodes (thin orchestrators; heavy lifting lives in agents.py) -----------


def ingest_node(state: NomosState) -> dict[str, Any]:
    """Node 1/3 (required): ingest arbitrary formal text -> tokens."""
    out = agents.tokenizer_agent(state.get("raw_text", ""))
    _log(state, out["tier"], out["model"], "tokenizer", "tokenise", out["offline"])
    return {
        "tokens": out["tokens"],
        "routing_log": state.get("routing_log", []),
        "topology_log": [f"ingest: {len(out['tokens'])} tokens via {out['model']}"],
    }


def scale_induction_node(state: NomosState) -> dict[str, Any]:
    """Node 2/3 (required): motif extraction (cheap) -> hierarchy (frontier).

    Real mined motifs are split deterministically: ~1 in 4 is held out of
    induction and used later as a generalization gate for mutations.
    """
    mined = agents.motif_miner_agent(state.get("raw_text", ""), state.get("tokens", []))
    _log(state, mined["tier"], mined["model"], "motif_miner", "motif_mine", mined["offline"])
    real = [m for m in mined["motifs"] if "synthetic-backfill" not in (m.get("contexts") or [])]
    holdout = [m for i, m in enumerate(real) if i % 4 == 3][:8] or real[:2]
    holdout_ids = {m["id"] for m in holdout}
    train = [m for m in mined["motifs"] if m["id"] not in holdout_ids]
    synth = agents.hierarchy_synthesizer_agent(
        state.get("raw_text", ""), train, state.get("law_id", "law-x")
    )
    _log(
        state,
        synth["tier"],
        synth["model"],
        "hierarchy_synthesizer",
        "hierarchy_synthesis",
        synth["offline"],
    )
    lattice = synth["lattice"]
    issues = agents.consistency_checker_agent(lattice)
    hist = list(state.get("lattice_history", [])) + [copy.deepcopy(lattice)]
    base = rewrite.measured_capacity(lattice)
    coverage = rewrite.holdout_coverage(lattice, holdout)
    return {
        "motifs": mined["motifs"],
        "holdout_motifs": holdout,
        "scale_lattice": lattice,
        "lattice_history": hist,
        "baseline_capacity": base["capacity"],
        "capacity_history": [
            {
                "version": lattice.get("version"),
                "iteration": 0,
                "source": "induction",
                "capacity": base["capacity"],
                "instances": base["instances"],
                "coverage": coverage,
            }
        ],
        "routing_log": state.get("routing_log", []),
        "topology_log": [
            f"scale_induction: lattice {lattice.get('version')} "
            f"{len(lattice.get('levels', []))} levels, "
            f"{sum(len(lv.get('rules', [])) for lv in lattice.get('levels', []))} rules; "
            f"holdout={len(holdout)} motifs; measured instances={base['instances']} "
            f"capacity={base['capacity']:.3f} coverage={coverage:.2f}; "
            f"consistency={issues['issues']}"
        ],
    }


def mutation_proposal_node(state: NomosState) -> dict[str, Any]:
    """Node 3/3 (required): spawn gen-0 mutator sub-agent swarm."""
    lattice = state.get("scale_lattice", {})
    proposals = list(state.get("mutation_proposals", []))
    start_idx = len(proposals)
    n = int(state.get("n_mutators", -1))
    if n < 0:
        n = N_MUTATORS_GEN0
    for i in range(n):
        out = agents.mutation_proposer_agent(lattice, idx=start_idx + i)
        _log(
            state,
            out["tier"],
            out["model"],
            out["proposal"]["author_agent"],
            "mutation_propose",
            out["offline"],
        )
        proposals.append(out["proposal"])
    return {
        "mutation_proposals": proposals,
        "routing_log": state.get("routing_log", []),
        "topology_log": [f"mutation_proposal: spawned gen-0 swarm of {n} (total {len(proposals)})"],
    }


def mesh_spawn_node(state: NomosState) -> dict[str, Any]:
    """Materialise the mesh world agents inhabit (lesser + higher instantiations).

    Each node gets one inhabitant carrying the loop seed (law, lattice
    version, rule, budget). Inhabitants then perform one governed spawn round:
    a few same-level peers plus downward children — the first recursive
    self-instantiation. Same-or-down only; budget-gated; fully logged.
    """
    lattice = state.get("scale_lattice", {})
    law_hash = state.get("law_hash", "h0")
    world = MeshWorld().spawn_from_lattice(lattice, state.get("motifs", []), law_hash, fanout=3)
    swarm = spawn.populate(world, lattice, state.get("law_id", "law-x"), budget=SPAWN_BUDGET)
    log: list[dict[str, Any]] = list(state.get("spawn_log", []))
    peers, children = 0, 0
    ops = ["refine_split", "elevate", "add_rule"]
    for iid in sorted(swarm):
        if peers >= SPAWN_PEERS:
            break
        try:
            child, _ = spawn.spawn_peer(swarm, world, lattice, iid, ops[peers % len(ops)], law_hash)
            log.append(spawn.log_event(iid, child["id"], "peer", str(child.get("level"))))
            peers += 1
        except ValueError:
            continue
    for iid in sorted(swarm):
        if children >= SPAWN_CHILDREN_CAP:
            break
        if iid not in swarm:  # newborns of this round do not spawn further this round
            continue
        try:
            child, _ = spawn.spawn_child(swarm, world, lattice, iid, law_hash)
            log.append(spawn.log_event(iid, child["id"], "child", str(child.get("level"))))
            children += 1
        except ValueError:
            continue
    nodes, edges = world.snapshot()
    stats = world.instantiation_capacity(lattice)
    return {
        "mesh_nodes": nodes,
        "mesh_edges": edges,
        "inhabitants": spawn.records(swarm),
        "spawn_log": log,
        "fitness_report": {
            **state.get("fitness_report", {}),
            "mesh_stats": stats,
            "mesh_size": len(nodes),
        },
        "topology_log": [
            f"mesh_spawn: {len(nodes)} instantiations ({peers} peers, {children} children), "
            f"{len(edges)} expansions, {len(swarm)} inhabitants, cap={stats}"
        ],
    }


def fitness_node(state: NomosState) -> dict[str, Any]:
    """Measured scoring of every proposed-but-untested mutation.

    Each proposal is applied in-memory and both lattices are executed; the
    score reflects the real capacity delta and holdout coverage delta, so
    declared generative_power cannot game the verdict.
    """
    lattice = state.get("scale_lattice", {})
    mesh_stats = state.get("fitness_report", {}).get("mesh_stats", {})
    holdout = state.get("holdout_motifs", []) or []
    proposals = list(state.get("mutation_proposals", []))
    accepted = list(state.get("accepted_mutations", []))
    for p in proposals:
        if p.get("status") != "proposed":
            continue
        scored = agents.fitness_evaluator_agent(lattice, p, mesh_stats, holdout)
        _log(
            state,
            scored["tier"],
            scored["model"],
            "fitness_evaluator",
            "mutation_score",
            scored["offline"],
        )
        p["test_score"] = scored["score"]
        p["fitness_breakdown"] = scored["breakdown"]
        p["status"] = "accepted" if scored["score"] >= FITNESS_ACCEPT_THRESHOLD else "rejected"
        if p["status"] == "accepted":
            accepted.append(copy.deepcopy(p))
    rep = {
        "mesh_stats": mesh_stats,
        "mesh_size": state.get("fitness_report", {}).get("mesh_size"),
        "accepted": [p["id"] for p in accepted],
        "rejected": [p["id"] for p in proposals if p.get("status") == "rejected"],
        "threshold": FITNESS_ACCEPT_THRESHOLD,
    }
    return {
        "mutation_proposals": proposals,
        "accepted_mutations": accepted,
        "fitness_report": rep,
        "routing_log": state.get("routing_log", []),
        "topology_log": [
            f"fitness: {len(accepted)} accepted / {len(proposals)} tested "
            f"(measured; holdout={len(holdout)})"
        ],
    }


def artifact_emit_node(state: NomosState) -> dict[str, Any]:
    """Frontier artifact designer -> self-contained runnable Python demo."""
    out = agents.artifact_designer_agent(
        state.get("scale_lattice", {}), state.get("accepted_mutations", []), state.get("motifs", [])
    )
    _log(state, out["tier"], out["model"], "artifact_designer", "artifact_design", out["offline"])
    path = state.get("artifact_path") or f"artifacts/artifact_{state.get('run_id', 'run-x')}.py"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out["code"])
    return {
        "artifact_code": out["code"],
        "artifact_path": path,
        "routing_log": state.get("routing_log", []),
        "topology_log": [
            f"artifact_emit: wrote {path} ({len(out['code'])} chars via {out['model']})"
        ],
    }


def _next_version(old_v: str, fallback: int) -> str:
    try:
        return f"v{int(str(old_v).lstrip('v')) + 1}"
    except Exception:
        return f"v{fallback}"


def _reachable(edges: list[list[str]], src: str, dst: str) -> bool:
    adj: dict[str, list[str]] = {}
    for s, d in edges:
        adj.setdefault(s, []).append(d)
    seen, stack = {src}, [src]
    while stack:
        u = stack.pop()
        for v in adj.get(u, []):
            if v == dst:
                return True
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def _topology_patch_valid(patch: dict[str, Any], topo: dict[str, Any]) -> tuple:
    """Static safety gate for rewire_topology patches.

    A patch is valid only if it references known nodes, does not duplicate an
    edge, and does not create an unconditional cycle (which would make
    execution non-terminating).
    """
    nodes = set(topo.get("nodes", []))
    edges = [list(e) for e in topo.get("edges", [])]
    if "entry" in patch and patch["entry"] not in nodes:
        return False, f"unknown entry {patch['entry']!r}"
    if "entry" in patch and patch["entry"] == topo.get("entry"):
        return False, f"entry already {patch['entry']!r} (no-op)"
    if "add_node" in patch and patch["add_node"] in nodes:
        return False, f"duplicate node {patch['add_node']!r}"
    if "add_edge" in patch:
        edge = list(patch["add_edge"])
        if len(edge) != 2:
            return False, "add_edge needs [src, dst]"
        s, d = edge
        if s not in nodes or d not in nodes:
            return False, f"unknown node in edge {s!r}->{d!r}"
        if [s, d] in edges:
            return False, f"duplicate edge {s!r}->{d!r}"
        if _reachable(edges, d, s):
            return False, f"edge {s!r}->{d!r} would create a cycle"
    return True, ""


def set_topology(topo: dict[str, Any]) -> dict[str, Any]:
    """Restore/replace the topology table wholesale (rollback support)."""
    TOPOLOGY.clear()
    TOPOLOGY.update(copy.deepcopy(topo))
    return describe_topology()


def rectify_node(state: NomosState) -> dict[str, Any]:
    """Apply accepted mutations in score order, each behind a keep/rollback gate.

    Lattice mutations: applied, then measured (executed enumeration + holdout
    coverage). Kept only if capacity does not regress and coverage does not
    drop; otherwise the lattice is restored and the rollback is logged.

    rewire_topology: the patch is validated (known nodes, no duplicate edge,
    acyclic), applied, and the graph must recompile. Any failure restores the
    previous topology. Kept patches are effective on the next build_graph().
    """
    lattice = copy.deepcopy(state.get("scale_lattice", {}))
    accepted = sorted(
        state.get("accepted_mutations", []) or [],
        key=lambda p: p.get("test_score", 0),
        reverse=True,
    )
    holdout = state.get("holdout_motifs", []) or []
    caps = list(state.get("capacity_history", []) or [])
    rollbacks = list(state.get("rollbacks", []) or [])
    topo_patches = list(state.get("topology_history", []) or [])
    it = int(state.get("iteration", 0)) + 1
    topo = [f"rectify: iteration {it}"]
    prev = (
        caps[-1]
        if caps
        else {
            "capacity": float(state.get("baseline_capacity", 0.0)),
            "coverage": 0.0,
            "version": lattice.get("version"),
        }
    )
    for mut in accepted:
        op = str(mut.get("operation", ""))
        if op == "rewire_topology":
            snapshot = describe_topology()
            patch = dict((mut.get("payload") or {}).get("patch") or {})
            ok, reason = _topology_patch_valid(patch, snapshot)
            if ok:
                try:
                    mutate_topology(patch)
                    build_graph()  # must still compile
                except Exception as e:  # noqa: BLE001 - any compile failure rolls back
                    set_topology(snapshot)
                    ok, reason = False, f"compile failed: {e}"
            else:
                set_topology(snapshot)
            topo_patches.append(
                {"iteration": it, "patch": patch, "applied": bool(ok), "reason": reason}
            )
            if ok:
                lattice.setdefault("provenance", {})["topology"] = (
                    f"{mut['id']} (effective on rebuild)"
                )
                topo.append(f"topology patched via {mut['id']} {patch} (effective on rebuild)")
            else:
                rollbacks.append(
                    {
                        "mutation": mut["id"],
                        "iteration": it,
                        "reason": f"topology patch rejected: {reason}",
                    }
                )
                topo.append(f"topology patch rolled back via {mut['id']}: {reason}")
            continue
        candidate = apply_mutation(lattice, mut)
        after = rewrite.measured_capacity(candidate)
        cov_before = rewrite.holdout_coverage(lattice, holdout)
        cov_after = rewrite.holdout_coverage(candidate, holdout)
        if (
            after["capacity"] + 1e-9 >= float(prev.get("capacity", -1e9))
            and cov_after + 1e-9 >= cov_before
        ):
            old_v = str(lattice.get("version", "v0"))
            lattice = candidate
            lattice["version"] = _next_version(old_v, len(state.get("lattice_history", [])))
            lattice.setdefault("provenance", {})["elevation"] = (
                f"{mut['id']} (score {mut.get('test_score')})"
            )
            caps.append(
                {
                    "version": lattice["version"],
                    "iteration": it,
                    "source": f"elevation:{mut['id']}",
                    "capacity": after["capacity"],
                    "instances": after["instances"],
                    "coverage": cov_after,
                }
            )
            prev = caps[-1]
            topo.append(
                f"elevated {old_v} -> {lattice['version']} via {mut['id']} "
                f"(capacity {after['capacity']:.3f}, instances {after['instances']}, "
                f"coverage {cov_after:.2f}, score {mut.get('test_score')})"
            )
        else:
            reason = (
                f"measured capacity {after['capacity']:.3f} < {float(prev.get('capacity', 0)):.3f}"
                if after["capacity"] + 1e-9 < float(prev.get("capacity", -1e9))
                else f"holdout coverage {cov_after:.2f} < {cov_before:.2f}"
            )
            rollbacks.append({"mutation": mut["id"], "iteration": it, "reason": reason})
            topo.append(f"rolled back {mut['id']}: {reason}")
    if not accepted:
        topo.append("no accepted mutations; lattice unchanged (descriptive bloat blocked)")
    hist = list(state.get("lattice_history", [])) + [copy.deepcopy(lattice)]
    return {
        "scale_lattice": lattice,
        "lattice_history": hist,
        "iteration": it,
        "capacity_history": caps,
        "rollbacks": rollbacks,
        "topology_history": topo_patches,
        "topology_log": topo,
    }


def _rectify_route(state: NomosState) -> str:
    if int(state.get("iteration", 0)) < int(state.get("max_iterations", 1)):
        return "mutation_proposal"
    return END


def build_graph():
    """Build the compiled LangGraph. Topology mirrors TOPOLOGY (inspectable)."""
    g = StateGraph(NomosState)
    g.add_node("ingest", _observed("ingest", ingest_node))
    g.add_node("scale_induction", _observed("scale_induction", scale_induction_node))
    g.add_node("mutation_proposal", _observed("mutation_proposal", mutation_proposal_node))
    g.add_node("mesh_spawn", _observed("mesh_spawn", mesh_spawn_node))
    g.add_node("fitness", _observed("fitness", fitness_node))
    g.add_node("artifact_emit", _observed("artifact_emit", artifact_emit_node))
    g.add_node("rectify", _observed("rectify", rectify_node))
    g.set_entry_point(TOPOLOGY["entry"])
    for s, d in TOPOLOGY["edges"]:
        g.add_edge(s, d)
    g.add_conditional_edges(
        "rectify", _rectify_route, {"mutation_proposal": "mutation_proposal", END: END}
    )
    return g.compile()


def describe_topology() -> dict[str, Any]:
    return copy.deepcopy(TOPOLOGY)


def mutate_topology(patch: dict[str, Any]) -> dict[str, Any]:
    """RSI self-modification hook: patch nodes/edges/entry at runtime.

    Example: mutate_topology({"add_edge": ["fitness", "mutation_proposal"]})
    Returns the new topology. Rebuild via build_graph().
    """
    if "add_node" in patch:
        TOPOLOGY["nodes"].append(patch["add_node"])
    if "add_edge" in patch:
        TOPOLOGY["edges"].append(list(patch["add_edge"]))
    if "entry" in patch:
        TOPOLOGY["entry"] = patch["entry"]
    TOPOLOGY["version"] += "+rsi"
    return describe_topology()


def run_pipeline(
    raw_text: str,
    law_title: str = "untitled-law",
    max_iterations: int = 1,
    artifact_path: str = "",
    mutators: int | None = None,
) -> NomosState:
    """Run the loop. `mutators=0` disables proposals (no-mutation baseline)."""
    state = new_run_state(raw_text, law_title=law_title, max_iterations=max_iterations)
    if mutators is not None:
        state["n_mutators"] = int(mutators)
    if artifact_path:
        state["artifact_path"] = artifact_path
    app = build_graph()
    return app.invoke(state)
