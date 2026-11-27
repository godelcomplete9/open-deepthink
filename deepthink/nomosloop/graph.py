"""LangGraph orchestration of the RSI loop for cybersecurity: Leak and stock.

The artifact is an institution's security policy. Agent Mechanics agents
stand for its clauses, the moduli that close each level, and every viable
control operation, plus the code and social-engineering facts read against
the policy (``surface.py``: repository analysis and role-level procedures). The loop alternates between two balances:

* **Leak (defensive).** An agent whose arriving current is not stored (its
  clauses clash, or its function is mis-mapped) heals in the lattice with a
  licensed operator. The heal is pulled back through the symbolic gauge
  coderivative into an inspection **finding** on the policy.
* **Stock (evolving).** An agent with a surplus spends it proposing a
  **control** clause. The control is kept only if the measured posture of
  the policy improves and no new contradiction appears.

Nodes:
  ingest           tokenise the policy                                (driver)
  scale_induction  clauses -> lattice: principles, policies, controls,
                   assets (deterministic parser; mapper fallback)     (mapper)
  correspond       table of correspondences; one agent per clause,
                   modulus and viable control operation               (mapper)
  tick             Agent Mechanics tick law; typed edges, circuits,
                   ranks, ensembles, clashes                            (none)
  heal             leaking agents heal; findings in the policy        (driver)
  stock            stocking agents propose controls (posture-gated)   (driver)
  evolve           active evolution (toggle): alternate universes bred
                   with the treatise's operators, read as what-ifs     (none)
  lora             distil traces; clamp the adapter into the driver     (none)
  artifact_emit    hardened policy + findings report + NX-1 report      (none)

Edges: ingest -> scale_induction -> correspond -> tick -> heal -> stock -> evolve -> lora
       lora -> correspond while iteration < max_iterations, else artifact_emit -> END.
       evolve passes straight through unless ``rsi_cfg["evolve"]["enabled"]``.
"""

from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from . import agents, evolution, security, surface, wsp
from . import correspondence as tc
from . import lora as lora_mod
from .mechanics import agent_name
from .mutations import apply_mutation
from .routing import call_llm
from .state import NomosState, new_run_state
from .world import World, drive_config, viable_operations

TOPOLOGY: dict[str, Any] = {
    "nodes": [
        "ingest",
        "scale_induction",
        "correspond",
        "tick",
        "heal",
        "stock",
        "evolve",
        "lora",
        "artifact_emit",
    ],
    "edges": [
        ["ingest", "scale_induction"],
        ["scale_induction", "correspond"],
        ["correspond", "tick"],
        ["tick", "heal"],
        ["heal", "stock"],
        ["stock", "evolve"],
        ["evolve", "lora"],
    ],
    "entry": "ingest",
    "conditional": {"lora": {"to_correspond_if": "iteration < max_iterations else artifact_emit"}},
    "version": "graph/v2-leak-stock-evolve-security",
}

# Optional live observer (GUI / telemetry): observer(phase, node, state, update).
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


def _log(state: NomosState, r: dict[str, Any], agent: str, task: str) -> list[dict[str, Any]]:
    rl = list(state.get("routing_log", []) or [])
    rl.append(
        {
            "agent": agent,
            "task": task,
            "tier": r["tier"],
            "model": r["model"],
            "offline": r["offline"],
        }
    )
    return rl


def _cfg(state: NomosState) -> dict[str, Any]:
    return dict(state.get("rsi_cfg") or {})


# ------------------------------------------------------------------ nodes --


def ingest_node(state: NomosState) -> dict[str, Any]:
    out = agents.tokenizer_agent(state.get("raw_text", ""))
    return {
        "tokens": out["tokens"],
        "routing_log": _log(state, out, "tokenizer", "tokenise"),
        "topology_log": [f"ingest: {len(out['tokens'])} tokens via {out['model']}"],
    }


def scale_induction_node(state: NomosState) -> dict[str, Any]:
    """Policy clauses -> lattice. Free-form text falls back to the mapper's synthesis."""
    raw = state.get("raw_text", "")
    rl = list(state.get("routing_log", []) or [])
    lattice = security.parse_policy(raw, state.get("law_id", "law-x"))
    source = "policy-parser"
    if lattice is None:
        mined = agents.motif_miner_agent(raw, state.get("tokens", []))
        rl.append(
            {
                "agent": "motif_miner",
                "task": "motif_mine",
                "tier": mined["tier"],
                "model": mined["model"],
                "offline": mined["offline"],
            }
        )
        synth = agents.hierarchy_synthesizer_agent(
            raw, mined["motifs"], state.get("law_id", "law-x")
        )
        rl.append(
            {
                "agent": "hierarchy_synthesizer",
                "task": "hierarchy_synthesis",
                "tier": synth["tier"],
                "model": synth["model"],
                "offline": synth["offline"],
            }
        )
        lattice = dict(synth["lattice"])
        source = "mapper-synthesis"
    pos = security.posture(lattice)
    n_rules = sum(len(lv.get("rules", [])) for lv in lattice.get("levels", []))
    return {
        "scale_lattice": lattice,  # type: ignore[typeddict-item]
        "lattice_history": [copy.deepcopy(lattice)],  # type: ignore[list-item]
        "posture_history": [
            {"version": lattice.get("version"), "iteration": 0, "source": "induction", **pos}
        ],
        "routing_log": rl,
        "topology_log": [
            f"scale_induction ({source}): {len(lattice.get('levels', []))} levels, {n_rules} clauses; "
            f"posture={pos['score']:.3f} diversity={pos['diversity']:.2f} contradictions={pos['contradictions']} "
            f"missing={','.join(pos['missing']) or '—'}"
        ],
    }


def _world(state: NomosState) -> World:
    w = state.get("world")
    if isinstance(w, World):
        return w
    cfg = _cfg(state)
    params, _ = drive_config(cfg.get("drive", "bal"), cfg.get("sensitivity"))
    w = World(
        params=params,
        drive=cfg.get("drive", "bal"),
        lambda_am=float(cfg.get("lambda_am", 0.7)),
        run_id=state.get("run_id", "run"),
    )
    w.ensure_root(state.get("raw_text", ""))
    return w


def correspond_node(state: NomosState) -> dict[str, Any]:
    """Build/refresh the table of correspondences and instantiate agents."""
    cfg = _cfg(state)
    world = _world(state)
    lattice = dict(state.get("scale_lattice", {}))
    ops = viable_operations(lattice, int(cfg.get("mutators", 8)))
    nodes, edges = tc.lattice_ax(lattice, ops)
    facts = list(state.get("surface_facts", []) or [])
    if facts:  # code and social facts, read against the current policy
        s_nodes, s_edges = surface.ax_nodes(facts, surface.link(facts, lattice), lattice)
        nodes += s_nodes
        edges += s_edges
    known = {r.ax_ref for r in world.rows if r.alive}
    fresh = [n for n in nodes if n.ref not in known]
    # Code and procedure facts stay with the offline lexicon unless the caller opts in,
    # so repository content and role descriptions are not sent to a remote mapper.
    remote_ok = bool(cfg.get("surface_remote_mapping"))
    to_map = [n for n in fresh if remote_ok or n.kind not in surface.KINDS]
    rl = list(state.get("routing_log", []) or [])
    mapped: dict[str, dict[str, Any]] = {}
    mapper_name = "offline-lexicon"
    for i in range(0, len(to_map), 16):
        got, r = tc.map_rows_llm(to_map[i : i + 16], call_llm)
        rl.append(
            {
                "agent": "toc_mapper",
                "task": "toc_map",
                "tier": r["tier"],
                "model": r["model"],
                "offline": r["offline"],
            }
        )
        mapped.update(got)
        if got:
            mapper_name = r["model"]
            for ref, data in got.items():
                for c in data.get("conflicts", []):
                    edges.append((ref, c, 3.141592653589793))
    res = world.sync(nodes, edges, mapped, mapper_name)
    declared = cfg.get("mapper_arc_agi2")
    note = ""
    if fresh and (declared is None or float(declared) <= 70):
        note = " (mapper recommendation ARC-AGI-2 > 70 not declared)"
    n_surf = sum(1 for n in nodes if n.kind in surface.KINDS)
    surf = f" + {n_surf} code/social facts" if n_surf else ""
    return {
        "world": world,
        "ops_by_ref": {f"op/{o['id']}": o for o in ops},
        "routing_log": rl,
        "topology_log": [
            f"correspond: {len(nodes) - n_surf} policy elements{surf}, {len(fresh)} new rows "
            f"({len(mapped)} by {mapper_name}), {res['born']} agents born, {res['tombstoned']} tombstoned{note}"
        ],
    }


def tick_node(state: NomosState) -> dict[str, Any]:
    cfg = _cfg(state)
    world = _world(state)
    world.run_ticks(int(cfg.get("ticks_per_loop", 3)))
    clashes = world.clashes()
    for f in clashes[:4]:
        world.event(
            "clash",
            f.i1,
            f"clash F(AX:{f.a1},AX:{f.a2};AM:{f.i1},AM:{f.i2}) f={f.f:.2f}",
            a1=f.a1,
            a2=f.a2,
            f=round(f.f, 3),
        )
    live = list(world.live().values())
    n_leak = sum(1 for a in live if a.state == "leak")
    n_stock = sum(1 for a in live if a.state == "stock")
    return {
        "world": world,
        "topology_log": [
            f"tick: t={world.t} agents={len(live)} leak={n_leak} stock={n_stock} "
            f"ensembles={len(world.ensembles)} clashes={len(clashes)} L_G={world.ensemble_leak():.3f}"
        ],
    }


def _surface_index(state: NomosState) -> dict[str, dict[str, Any]]:
    return {str(f.get("ref")): f for f in state.get("surface_facts", []) or []}


def _surface_ref(world: World, agent_id: str) -> str:
    """The code/social fact an agent stands for ('' for a policy agent)."""
    for r in world.rows_of(agent_id):
        if r.ax_ref.startswith(("code/", "soc/")):
            return r.ax_ref
    return ""


def heal_node(state: NomosState) -> dict[str, Any]:
    """Leak: agents heal in the lattice; each heal becomes a finding on the policy."""
    cfg = _cfg(state)
    world = _world(state)
    _, budgets = drive_config(cfg.get("drive", "bal"), cfg.get("sensitivity"))
    budget = int(cfg.get("heal_budget") or budgets["heal_budget"])
    pool = sorted(
        (a for a in world.live().values() if a.state == "leak"), key=lambda a: (-a.L, a.id)
    )
    # Code and social agents get their own budget, so they never crowd out the policy's clashes.
    leakers = [a for a in pool if not _surface_ref(world, a.id)][:budget]
    leakers += [a for a in pool if _surface_ref(world, a.id)][: max(2, budget)]
    facts = _surface_index(state)
    rl = list(state.get("routing_log", []) or [])
    buf = list(state.get("lora_buffer", []) or [])
    healed = 0
    clashes = world.clashes()
    objective = str(cfg.get("objective") or "")
    for a in leakers:
        before_name = agent_name(a, world.params)
        rec = world.heal(a)
        if rec["op"] == "e":
            world.event("heat", a.id, "leak with no licensed heal that lowers it (heat)")
            continue
        healed += 1
        edits = tc.coderivative(rec["op"], a.id, world.rows)
        mine = [f for f in clashes if a.id in (f.i1, f.i2)][:3]
        texts = {
            ref: (world.ax[ref].text if ref in world.ax else "")
            for f in mine
            for ref in (f.a1, f.a2)
        }
        prompt = (
            f"Security objective: {objective or 'protect the institution'}.\n"
            f"Agent {a.id} = {before_name} leaks ({rec.get('cause')}) on {rec.get('species')}.\n"
            f"Clashing clauses: {[(f.a1, texts.get(f.a1, '')[:160], f.a2, texts.get(f.a2, '')[:160], round(f.f, 2)) for f in mine]}.\n"
            f"Licensed heal chosen in the lattice: {rec['op']} -> {[e.symbol + ' @ ' + e.ax_ref for e in edits]}.\n"
            "In two sentences, state the security finding and the remediation."
        )
        note = ""
        if cfg.get("cot", True):
            r = call_llm(
                prompt,
                task_kind="heal_cot",
                system="You are a security inspection agent. Be concrete and brief.",
                max_tokens=220,
            )
            rl.append(
                {
                    "agent": f"heal:{a.id}",
                    "task": "heal_cot",
                    "tier": r["tier"],
                    "model": r["model"],
                    "offline": r["offline"],
                }
            )
            if not r["offline"]:
                note = r["text"].strip()[:500]
                world.rounds += 1
        fact = facts.get(_surface_ref(world, a.id))
        if not note and fact:
            linked = [
                x
                for f in mine
                for x in (f.a1, f.a2)
                if x != fact["ref"] and not x.startswith(("code/", "soc/"))
            ]
            clause = next((x for x in linked if "#mod" not in x), "")
            if clause:
                note = (
                    f"{fact['statement']} This contradicts {clause.split('/')[-1]}"
                    f"{_clause(world.ax[clause].text if clause in world.ax else '')}. "
                    f"Remediation: {fact['remediation']}"
                )
            else:
                note = f"{fact['statement']} No clause of the policy covers this. Remediation: {fact['remediation']}"
        if not note:
            if mine:
                f = mine[0]
                q1 = _clause(texts.get(f.a1, ""))
                q2 = _clause(texts.get(f.a2, ""))
                note = (
                    f"{f.a1.split('/')[-1]}{q1} and {f.a2.split('/')[-1]}{q2} cannot both hold "
                    f"(frustration {f.f:.2f}). Remediation ({rec['op']}): {tc.EDITS.get(rec['op'], '')}."
                )
            else:
                note = f"{a.id} stores too little of what reaches it on {rec.get('species')} ({rec.get('cause')}). Remediation ({rec['op']}): {tc.EDITS.get(rec['op'], '')}."
        lk = f"{rec.get('species')}:{rec.get('cause')}{rec['before']:.2f}"
        world.event(
            "leak",
            a.id,
            f"leak[{lk}] heal({rec['op']}) => {agent_name(a, world.params).split('/')[0]}",
            op=rec["op"],
            before=round(rec["before"], 3),
            after=round(rec["after"], 3),
            memory=rec["memory"],
        )
        severity = max([f.weight for f in mine], default=0.0) * 0.6 + rec["before"] * 0.4
        world.inspections.append(
            {
                "t": world.t,
                "agent": a.id,
                "name": before_name,
                "op": rec["op"],
                "cause": rec.get("cause"),
                "species": rec.get("species"),
                "severity": round(severity, 3),
                "leak_before": round(rec["before"], 3),
                "leak_after": round(rec["after"], 3),
                "trials": rec["trials"],
                "memory": rec["memory"],
                "edits": [e.as_dict() for e in edits],
                "clashes": [{"a1": f.a1, "a2": f.a2, "f": round(f.f, 3)} for f in mine],
                "finding": note,
            }
        )
        answer = f"heal({rec['op']}); " + "; ".join(f"{e.symbol} on {e.ax_ref}" for e in edits[:3])
        buf.append(lora_mod.example("heal", prompt, answer))
    return {
        "world": world,
        "routing_log": rl,
        "lora_buffer": buf,
        "topology_log": [
            f"heal: {healed}/{len(leakers)} leaking agents healed; "
            f"rounds/solve={world.rounds_per_solve():.2f}; L_G={world.ensemble_leak():.3f}"
        ],
    }


def _clause(text: str) -> str:
    """Quote the clause body of an AX node text ('name: pattern => rewrite')."""
    body = text.split("=>", 1)[-1].strip() if text else ""
    return f' ("{body[:110]}{"…" if len(body) > 110 else ""}")' if body else ""


def _next_version(v: str) -> str:
    try:
        return f"v{int(str(v).lstrip('v')) + 1}"
    except ValueError:
        return "v1"


def stock_node(state: NomosState) -> dict[str, Any]:
    """Stock: agents with a surplus propose controls; posture gates them."""
    cfg = _cfg(state)
    world = _world(state)
    _, budgets = drive_config(cfg.get("drive", "bal"), cfg.get("sensitivity"))
    budget = int(cfg.get("stock_budget") or budgets["stock_budget"])
    lattice: dict[str, Any] = copy.deepcopy(dict(state.get("scale_lattice", {})))
    post_hist = list(state.get("posture_history", []) or [])
    rollbacks = list(state.get("rollbacks", []) or [])
    controls = list(state.get("controls", []) or [])
    hist = list(state.get("lattice_history", []) or [])
    ops_by_ref = state.get("ops_by_ref") or {}
    rl = list(state.get("routing_log", []) or [])
    buf = list(state.get("lora_buffer", []) or [])
    objective = str(cfg.get("objective") or "")
    it = int(state.get("iteration", 0)) + 1
    levels = [str(lv.get("level_id")) for lv in lattice.get("levels", []) or []]
    if levels and "L1" not in levels:
        lattice = security.ensure_controls_level(lattice)
    target_level = "L1" if levels else ""
    facts = _surface_index(state)
    stockers = sorted(
        (a for a in world.live().values() if a.state == "stock"), key=lambda a: (-a.dw_hat, a.id)
    )[:budget]
    kept = 0
    for a in stockers:
        rows = world.rows_of(a.id)
        if not rows or not target_level:
            continue
        ax_ref = rows[0].ax_ref
        node = world.ax.get(ax_ref)
        before = security.posture(lattice)
        fact = facts.get(_surface_ref(world, a.id))
        if fact:  # a code or social gap: the repair becomes a control in the policy
            op = "sigma"
            fn = fact.get("species") if fact.get("species") in security.FUNCTION else "Ti"
            text = str(fact.get("remediation", "")).rstrip(".") + f" ({fact.get('title', '')})."
        elif a.op and ax_ref in ops_by_ref:  # an operation agent adds its own function
            fn = ops_by_ref[ax_ref]["function"]
            op = "sigma"
            text = security.function_control(
                fn, security.subject(objective) if objective else "all assets in scope", objective
            )
        else:
            op = a.last_op or ("sigma" if before["missing"] else "r")
            fn, text = security.control_clause(
                op, a.hero(), node.text if node else "", lattice, objective, salt=a.id
            )
        if cfg.get("cot", True):
            prompt = (
                f"Security objective: {objective or 'protect the institution'}.\n"
                f"Clause: {(node.text if node else ax_ref)[:300]}\n"
                f"Lattice operator {op}: {tc.EDITS.get(op, '')}. Function to add: {security.FUNCTION[fn]}.\n"
                f"Draft: {text}\nRewrite the draft as ONE enforceable control sentence. Output only the sentence."
            )
            r = call_llm(
                prompt,
                task_kind="stock_cot",
                system="You write precise, enforceable security controls.",
                max_tokens=120,
            )
            rl.append(
                {
                    "agent": f"stock:{a.id}",
                    "task": "stock_cot",
                    "tier": r["tier"],
                    "model": r["model"],
                    "offline": r["offline"],
                }
            )
            world.rounds += 1
            cand_text = r["text"].strip().strip('"').split("\n")[0][:300]
            if not r["offline"] and len(cand_text) > 12:
                text = cand_text
        rid = f"§R{len(controls) + 1}"
        mutation = {
            "operation": "add_rule",
            "payload": {
                "new_rule": {
                    "id": rid,
                    "name": " ".join(text.split()[:6]),
                    "level": target_level,
                    "pattern": security.subject(text),
                    "rewrite": text,
                    "provenance": f"rsi:{a.id}:{op}",
                }
            },
        }
        cand = dict(apply_mutation(lattice, mutation))  # type: ignore[arg-type]
        after = security.posture(cand)
        new_contra = after["contradictions"] > before["contradictions"]
        if after["score"] > before["score"] + 1e-6 and not new_contra:
            old_v = str(lattice.get("version", "v0"))
            lattice = cand
            lattice["version"] = _next_version(old_v)
            ref = f"pol/{target_level}/{rid}"
            world.pending_parent[ref] = a.id
            hist.append(copy.deepcopy(lattice))
            post_hist.append(
                {
                    "version": lattice["version"],
                    "iteration": it,
                    "source": f"stock:{a.id}:{op}",
                    **after,
                }
            )
            controls.append(
                {
                    "id": rid,
                    "ref": ref,
                    "text": text,
                    "function": fn,
                    "agent": a.id,
                    "op": op,
                    "clause": ax_ref,
                    "posture_before": before["score"],
                    "posture_after": after["score"],
                    "version": lattice["version"],
                }
            )
            gain = after["score"] - before["score"]
            world.feed(a.id, 1.0 + 20.0 * gain)
            world.solves += 1
            kept += 1
            world.event(
                "stock",
                a.id,
                f"stock[{a.dw_hat:.2f}] evolve({op}: {security.FUNCTION[fn]}) => AX:{ref}  posture {before['score']:.3f}->{after['score']:.3f}",
                op=op,
                ref=ref,
                version=lattice["version"],
            )
            buf.append(
                lora_mod.example(
                    "stock",
                    f"{agent_name(a, world.params)} stocks on {ax_ref}. Which control?",
                    text,
                )
            )
        else:
            reason = (
                "would add a contradiction"
                if new_contra
                else f"posture would not improve ({before['score']:.3f} -> {after['score']:.3f})"
            )
            rollbacks.append(
                {"agent": a.id, "op": op, "text": text, "iteration": it, "reason": reason}
            )
            world.event("rollback", a.id, f"stock[{a.dw_hat:.2f}] control rolled back: {reason}")
        world.spend(a, 0.3 * world.params.E0)
    return {
        "world": world,
        "scale_lattice": lattice,  # type: ignore[typeddict-item]
        "lattice_history": hist,
        "posture_history": post_hist,
        "rollbacks": rollbacks,
        "controls": controls,
        "routing_log": rl,
        "lora_buffer": buf,
        "topology_log": [
            f"stock: {len(stockers)} stocking agents; {kept} controls kept; policy {lattice.get('version')}"
        ],
    }


def evolve_node(state: NomosState) -> dict[str, Any]:
    """Active evolution (toggle): alternate universes bred in the lattice, read as what-ifs."""
    cfg = _cfg(state)
    ecfg = evolution.EvolveConfig.from_dict(cfg.get("evolve"))
    if not ecfg.enabled:
        return {"topology_log": ["evolve: off"]}
    world = _world(state)
    loop = int(state.get("iteration", 0)) + 1
    hist = list(state.get("evolution", []) or [])
    leaking = sum(1 for a in world.live().values() if a.state == "leak")
    if ecfg.trigger == "calm" and leaking:
        hist.append(
            {"loop": loop, "skipped": f"{leaking} agents leak", "universes": [], "probes": []}
        )
        return {
            "world": world,
            "evolution": hist,
            "topology_log": [f"evolve: skipped, {leaking} agents leak (trigger=calm)"],
        }
    lattice: dict[str, Any] = copy.deepcopy(dict(state.get("scale_lattice", {})))
    if ecfg.seed is not None:
        seed = ecfg.seed + loop - 1
    else:
        digest = hashlib.sha256(f"{state.get('law_hash', '')}:{loop}".encode()).hexdigest()
        seed = int(digest[:8], 16)
    out = evolution.evolve(
        world,
        lattice,
        list(state.get("surface_facts", []) or []),
        ecfg,
        loop=loop,
        seed=seed,
        mutators=int(cfg.get("mutators", 8)),
    )
    cand = out.pop("candidate", None)
    update: dict[str, Any] = {}
    adopted = None
    if ecfg.adopt == "gate" and cand:
        new_lat = copy.deepcopy(cand["lattice"])
        new_lat["version"] = _next_version(str(lattice.get("version", "v0")))
        tag = f"rsi:evolve:{cand['id']}"
        for lv in new_lat.get("levels", []) or []:
            for r in lv.get("rules", []) or []:
                ref = f"pol/{lv.get('level_id')}/{r.get('id')}"
                if str(r.get("provenance", "")).startswith("evolve:") or ref in cand["replaced"]:
                    r["provenance"] = tag
        by_ref = {r.ax_ref: r.agent_id for r in world.rows if r.alive}
        for new_ref, root in cand["roots"].items():
            if root in by_ref:
                world.pending_parent[new_ref] = by_ref[root]
        nodes, _ = tc.lattice_ax(new_lat, [])
        nmap = {n.ref: n for n in nodes}
        for ref in cand["replaced"]:
            if ref in nmap:
                world.reproject(nmap[ref], "evolve-lexicon")
        pos = security.posture(new_lat)
        update = {
            "scale_lattice": new_lat,
            "lattice_history": [
                *(state.get("lattice_history", []) or []),
                copy.deepcopy(new_lat),
            ],
            "posture_history": [
                *(state.get("posture_history", []) or []),
                {
                    "version": new_lat["version"],
                    "iteration": loop,
                    "source": f"evolve:{cand['id']}",
                    **pos,
                },
            ],
        }
        adopted = cand["id"]
        world.event(
            "adopt",
            by_ref.get(cand["roots"].get(next(iter(cand["roots"]), ""), ""), "I1"),
            f"adopt({cand['id']}) => policy {new_lat['version']} posture {pos['score']:.3f}",
            universe=cand["id"],
        )
    out["adopted"] = adopted
    for u in out["universes"][:6] + out["probes"]:
        ops = " ".join(f"{m['symbol']}@{m['site'].split('/')[-1]}" for m in u["mutations"])
        world.event(
            "branch",
            u["mutations"][0]["agent"],
            f"branch({u['id']}: {ops}) => {u['verdict']} "
            f"dP={u['delta']['posture']:+.3f} dL={u['delta']['L_G']:+.3f}",
            universe=u["id"],
        )
    buf = list(state.get("lora_buffer", []) or [])
    for u in out["universes"][:3] + out["probes"][:1]:
        answer = f"{u['verdict']}: " + ("; ".join(u["consequences"][:5]) or "no measurable change")
        buf.append(
            lora_mod.example("evolve", f"Policy {lattice.get('version')}. {u['what_if']}", answer)
        )
    hist.append(out)
    best = out["universes"][0] if out["universes"] else None
    line = (
        f"evolve: {len(out['universes'])} universes over {len(out['survivors'])} generations, "
        f"{len(out['probes'])} ω probes (seed {seed})"
    )
    if best:
        line += f"; best {best['id']} ({best['verdict']}, fitness {best['fitness']:.2f}): {best['what_if'][:90]}"
    line += f"; adopted: {adopted or 'none'}"
    return {"world": world, "evolution": hist, "lora_buffer": buf, "topology_log": [line], **update}


def lora_node(state: NomosState) -> dict[str, Any]:
    cfg = _cfg(state)
    world = _world(state)
    lora_in = dict(state.get("lora_cfg") or {})
    prog_in = lora_in.pop("program", None) or {}
    lcfg = lora_mod.LoraConfig(**lora_in)
    pcfg = wsp.ProgramConfig(**prog_in) if isinstance(prog_in, dict) else wsp.ProgramConfig()
    lstate = state.get("lora_state")
    if not isinstance(lstate, lora_mod.LoraState):
        lstate = lora_mod.LoraState()
    post = state.get("posture_history", []) or []
    metric = {
        "rounds_per_solve": round(world.rounds_per_solve(), 4),
        "leak": round(world.ensemble_leak(), 4),
        "capacity": round(float(post[-1]["score"]) if post else 0.0, 4),
        "solves": float(world.solves),
        "space": float(world.coverage_history[-1]["space"] if world.coverage_history else 0),
    }
    if not lcfg.out_dir:
        base = os.path.dirname(state.get("artifact_path") or "") or "artifacts"
        lcfg.out_dir = os.path.join(base, f"lora_{state.get('run_id', 'run')}")
    if cfg.get("lora_dataset", True):
        lora_mod.append_examples(lstate, lcfg.out_dir, list(state.get("lora_buffer", []) or []))
    rl = list(state.get("routing_log", []) or [])
    program_fn = None
    if pcfg.enabled:
        history = [h.get("metric", {}) for h in lstate.history] + [metric]

        def call(prompt: str, **kw: Any) -> dict[str, Any]:
            r = call_llm(prompt, **kw)
            rl.append(
                {
                    "agent": "weight_programmer",
                    "task": "weight_program",
                    "tier": r["tier"],
                    "model": r["model"],
                    "offline": r["offline"],
                }
            )
            return r

        def program_fn() -> dict[str, Any]:
            return wsp.program(lcfg, pcfg, lstate, history, call)

    rec = lora_mod.step(
        lcfg, lstate, metric, state.get("run_id", "run"), world.t, program_fn=program_fn
    )
    mix_txt = ",".join(f"{m['id']}:{m['scale']:.2f}" for m in lstate.mix)
    world.event(
        "clamp",
        "LoRA",
        f"clamp(lora=adapter{lcfg.adapter_id},scale={lstate.scale:.2f}{',mix=' + mix_txt if mix_txt else ''}) {rec['decision']}",
        decision=rec["decision"],
        scale=lstate.scale,
    )
    return {
        "world": world,
        "routing_log": rl,
        "lora_state": lstate,
        "lora_cfg": {**(state.get("lora_cfg") or {}), "out_dir": lcfg.out_dir},
        "lora_buffer": [],
        "iteration": int(state.get("iteration", 0)) + 1,
        "topology_log": [
            f"lora: {lstate.examples} examples, decision={rec['decision']} scale={lstate.scale:.2f} "
            f"pushed={rec['clamp'].get('pushed')}; metric={metric}"
        ],
    }


def artifact_emit_node(state: NomosState) -> dict[str, Any]:
    world = _world(state)
    world.adopt_pending(dict(state.get("scale_lattice", {})))
    text = security_report(state, world)
    path = state.get("artifact_path") or f"artifacts/rsi_{state.get('run_id', 'run-x')}.md"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    report = os.path.splitext(path)[0] + ".nx1.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write(world.nx1_report())
    return {
        "artifact_code": text,
        "artifact_path": path,
        "nx1_path": report,
        "topology_log": [f"artifact_emit: wrote {path} and {report}"],
    }


def security_report(state: NomosState, world: World) -> str:
    """Hardened policy, findings and coverage as one markdown document."""
    lat = state.get("scale_lattice", {}) or {}
    post = state.get("posture_history", []) or []
    first, last = (post[0], post[-1]) if post else ({}, {})
    controls = state.get("controls", []) or []
    findings = sorted(world.inspections, key=lambda x: -x.get("severity", 0.0))
    facts_n = len(state.get("surface_facts", []) or [])
    lines = [
        f"# Hardened policy: {state.get('law_title', 'policy')}",
        "",
        f"Run `{state.get('run_id')}` · drive `{world.drive}` · λ_AM {world.lambda_am} · policy {lat.get('version')}",
        "",
        f"- Posture: {first.get('score', 0):.3f} → **{last.get('score', 0):.3f}** "
        f"(function diversity {last.get('diversity', 0):.2f}, contradictory clause pairs {last.get('contradictions', 0)})",
        f"- Findings: {len(findings)} · controls kept: {len(controls)} · rolled back: {len(state.get('rollbacks', []) or [])}",
        *(
            [f"- Code and social-engineering facts read against the policy: {facts_n}"]
            if facts_n
            else []
        ),
        f"- Agents: {len(world.live())} alive, {world.tombstones} tombstoned · ensembles: {len(world.ensembles)} · rounds per solve: {world.rounds_per_solve():.2f}",
        "",
        "Posture is bookkeeping over the policy text (diversity of covered security functions over contradictions). It does not certify that the institution is secure.",
        "",
        "## Findings (leak-driven)",
        "",
    ]
    if not findings:
        lines.append("No agent leaked enough to need healing.")
    for i, fnd in enumerate(findings[:30], 1):
        cl = (
            "; ".join(f"`{c['a1']}` × `{c['a2']}` (f={c['f']:.2f})" for c in fnd.get("clashes", []))
            or "no clash face"
        )
        lines.append(
            f"{i}. **{fnd['agent']}** · severity {fnd.get('severity', 0):.2f} · heal `{fnd['op']}` · {cl}"
        )
        lines.append(f"   {fnd.get('finding', '')}")
    lines += ["", "## Controls added (stock-driven)", ""]
    if not controls:
        lines.append("No control improved the measured posture.")
    for c in controls:
        lines.append(
            f"- **{c['id']}** [{security.FUNCTION[c['function']]}] {c['text']} "
            f"_(by {c['agent']} via `{c['op']}`, posture {c['posture_before']:.3f}→{c['posture_after']:.3f})_"
        )
    lines += ["", "## Policy (lattice " + str(lat.get("version")) + ")", ""]
    for lv in lat.get("levels", []) or []:
        lines.append(f"### {lv.get('level_id')} · {lv.get('name')}")
        lines.append(f"_base: {lv.get('base', '')} · modulus: {lv.get('modulus', '')}_")
        lines.append("")
        for r in lv.get("rules", []) or []:
            tag = " **(RSI)**" if str(r.get("provenance", "")).startswith("rsi:") else ""
            lines.append(f"- {r.get('id')} {r.get('rewrite', '')}{tag}")
        lines.append("")
    if lat.get("links"):
        lines.append(
            "_Links: "
            + "; ".join(
                f"{ln.get('from', '').split('/')[-1]} → {ln.get('to', '').split('/')[-1]} ({ln.get('kind')})"
                for ln in lat["links"]
            )
            + "_"
        )
        lines.append("")
    facts = list(state.get("surface_facts", []) or [])
    if facts or state.get("surface_meta"):
        rows = surface.register(facts, surface.link(facts, lat))
        lines += surface.report_lines(rows, dict(state.get("surface_meta") or {}))
    if state.get("evolution"):
        lines += evolution.report_lines(list(state.get("evolution") or []))
    if last.get("functions"):
        lines += [
            "## Function coverage",
            "",
            "| Species | Security function | Share |",
            "|---|---|---|",
        ]
        for sp, share in last["functions"].items():
            lines.append(f"| {sp} | {security.FUNCTION[sp]} | {share:.3f} |")
        lines.append("")
    return "\n".join(lines)


def _route(state: NomosState) -> str:
    if int(state.get("iteration", 0)) < int(state.get("max_iterations", 1)):
        return "correspond"
    return "artifact_emit"


def build_graph():
    g = StateGraph(NomosState)
    for name, fn in (
        ("ingest", ingest_node),
        ("scale_induction", scale_induction_node),
        ("correspond", correspond_node),
        ("tick", tick_node),
        ("heal", heal_node),
        ("stock", stock_node),
        ("evolve", evolve_node),
        ("lora", lora_node),
        ("artifact_emit", artifact_emit_node),
    ):
        g.add_node(name, _observed(name, fn))
    g.set_entry_point(TOPOLOGY["entry"])
    for s, d in TOPOLOGY["edges"]:
        g.add_edge(s, d)
    g.add_conditional_edges(
        "lora", _route, {"correspond": "correspond", "artifact_emit": "artifact_emit"}
    )
    g.add_edge("artifact_emit", END)
    return g.compile()


def describe_topology() -> dict[str, Any]:
    return copy.deepcopy(TOPOLOGY)


def run_pipeline(
    raw_text: str,
    law_title: str = "untitled-policy",
    max_iterations: int = 1,
    artifact_path: str = "",
    mutators: int | None = None,
    rsi_cfg: dict[str, Any] | None = None,
    lora_cfg: dict[str, Any] | None = None,
    surface_facts: list[dict[str, Any]] | None = None,
    surface_meta: dict[str, Any] | None = None,
) -> NomosState:
    """Run the loop. ``mutators=0`` instantiates no operation agents (baseline).

    ``surface_facts`` are code and social-engineering facts (``repo.py``,
    ``social.py``); they become agents beside the policy's own.
    """
    state = new_run_state(raw_text, law_title=law_title, max_iterations=max_iterations)
    state["surface_facts"] = [dict(f) for f in surface_facts or []]
    state["surface_meta"] = dict(surface_meta or {})
    cfg = dict(rsi_cfg or {})
    if mutators is not None:
        cfg["mutators"] = int(mutators)
    state["rsi_cfg"] = cfg
    state["lora_cfg"] = dict(lora_cfg or {})
    if artifact_path:
        state["artifact_path"] = artifact_path
    app = build_graph()
    return app.invoke(state, {"recursion_limit": 25 + 8 * max(1, int(max_iterations))})


__all__ = [
    "TOPOLOGY",
    "build_graph",
    "describe_topology",
    "run_pipeline",
    "security_report",
    "set_observer",
]
