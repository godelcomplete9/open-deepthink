"""RSI pipeline entrypoint: params -> LangGraph leak/stock loop -> result.

The fourth open-deepthink section: **recursive self-improvement for
cybersecurity**. The artifact is an institution's security policy. Agent
Mechanics agents stand for its clauses, moduli and viable control
operations. Leaking agents produce inspection findings, and stocking agents
evolve the policy with posture-gated controls. A LoRA loop distils the traces
into the local driver model.

Model roles:

* ``llm``, the **driver**, must be local: in-process, or a llama.cpp /
  private-network endpoint. It drives the lattice, the heals and the control
  drafting. ``require_local_driver`` (default True) refuses a remote driver.
* ``editor_llm``, optional, is the editing model for weight-space programming
  of the LoRA adapter (``params["lora"]["program"]``). A powerful model such
  as Claude Fable 5.2 or GPT-6.2 Astra is the intended use. It sees trace
  text, which quotes the policy.
* ``mapper_llm``, optional, builds the table of correspondences. A model
  above 70 on ARC-AGI-2 is recommended. Declare its score with
  ``params["mapper_arc_agi2"]`` (it is recorded, not verified).

Artifacts beside the policy (optional):

* ``params["repo_path"]``: a local repository or file. Its code and
  configuration are scanned (``repo.py``) and every security fact is read
  against the policy, so code that does what a clause forbids becomes a
  finding.
* ``params["social_profile"]``: a role-level profile of the institution's
  procedures (``social.py``), for the social-engineering layer. It needs an
  ``authorization`` block; personal and psychometric data is not admitted.

Their facts are mapped offline unless ``params["surface_remote_mapping"]``
is true, so repository content does not reach a remote mapper by default.

With ``llm=None`` or ``offline=True`` everything runs on deterministic
offline heuristics, keyless.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from collections.abc import Callable
from typing import Any

from . import evolution, repo, social, surface
from .graph import describe_topology, run_pipeline, set_observer
from .routing import is_local_llm, llm_endpoint
from .world import World

LogFn = Callable[[str], Any] | None
EventFn = Callable[[dict[str, Any]], Any] | None

DRIVES = ("leak", "stock", "bal")


def default_rsi_params() -> dict[str, Any]:
    """Documented RSI defaults for harnesses / library callers."""
    return {
        "iterations": 1,
        "mutators": 8,  # cap on viable control operations (agents); 0 = none
        "law_title": "untitled-policy",
        "artifact_path": "",
        "offline": False,
        "drive": "bal",  # leak (defensive) | stock (evolving) | bal
        "lambda_am": 0.7,  # 0 = artifact signals only, 1 = Agent Mechanics balance only
        "sensitivity": {},  # overrides of mechanics.Params / drive presets
        "ticks_per_loop": 3,
        "heal_budget": None,  # None = drive preset
        "stock_budget": None,
        "objective": "",  # security objective that guides findings and controls
        "cot": True,  # ask the driver for findings/controls text
        "lora": {},  # LoraConfig fields + "program": {enabled, adapter_ids, allow_negative, sample}
        "mapper_arc_agi2": None,  # declared ARC-AGI-2 score of the mapper
        "require_local_driver": True,
        "repo_path": "",  # local repository or file to analyse beside the policy
        "social_profile": None,  # role-level profile (dict, JSON text or .json path)
        "surface_remote_mapping": False,  # send code/social facts to a remote mapper
        # active evolution (evolution.EvolveConfig): {enabled, universes, generations, keep,
        # ticks, probes, trigger: always|calm, adopt: none|gate, seed}
        "evolve": {},
    }


def collect_surface(
    raw_text: str, repo_path: str = "", social_profile: Any = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan the repository and analyse the social profile. Raises ValueError on bad input."""
    facts: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    if repo_path:
        try:
            scan = repo.scan_repository(str(repo_path))
        except FileNotFoundError as e:
            raise ValueError(f"repository not found: {repo_path!r}") from e
        facts += [f.as_dict() for f in scan.facts]
        meta["code"] = scan.meta()
    if social_profile not in (None, "", {}):
        try:
            soc_facts, soc_meta = social.analyze(social_profile, raw_text)
        except (social.SocialProfileError, json.JSONDecodeError, OSError) as e:
            raise ValueError(f"social profile: {e}") from e
        facts += [f.as_dict() for f in soc_facts]
        meta["social"] = soc_meta
    return facts, meta


def _make_bridge(llm):
    """Adapt a LangChain chat model to the engine's sync bridge."""

    def bridge(prompt: str, task_kind: str, system: str, max_tokens: int, tier: str):
        messages: list[tuple[str, str]] = []
        if system:
            messages.append(("system", system))
        messages.append(("user", prompt))
        resp = llm.invoke(messages)
        content = getattr(resp, "content", resp)
        if isinstance(content, list):
            content = "".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
            )
        return None if content is None else str(content)

    return bridge


def snapshot_state(state: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe, size-capped view of a run for live visualisation."""
    lat = state.get("scale_lattice", {}) or {}
    levels = [
        {
            "id": str(lv.get("level_id", "?")),
            "name": str(lv.get("name", "")),
            "base": str(lv.get("base", "")),
            "modulus": str(lv.get("modulus", "")),
            "rules": [
                {
                    "id": str(r.get("id", "")),
                    "name": str(r.get("name", "")),
                    "rsi": str(r.get("provenance", "")).startswith("rsi:"),
                }
                for r in lv.get("rules", []) or []
            ],
        }
        for lv in lat.get("levels", []) or []
    ]
    world = state.get("world")
    w = world.snapshot() if isinstance(world, World) else {}
    lst = state.get("lora_state")
    keys = (
        "version",
        "iteration",
        "score",
        "diversity",
        "contradictions",
        "source",
        "functions",
        "missing",
    )
    return {
        "lattice": {"version": lat.get("version"), "levels": levels},
        "posture": [{k: p.get(k) for k in keys} for p in state.get("posture_history", []) or []],
        "controls": (state.get("controls", []) or [])[-40:],
        "rollbacks": (state.get("rollbacks", []) or [])[-20:],
        "lora": lst.as_dict() if hasattr(lst, "as_dict") else {},
        "iteration": int(state.get("iteration", 0) or 0),
        "max_iterations": int(state.get("max_iterations", 1) or 1),
        "evolution": evolution.compact(list(state.get("evolution", []) or [])),
        **w,
    }


def _summary(state: dict[str, Any], world: World | None) -> str:
    post = state.get("posture_history", []) or []
    first, last = (post[0], post[-1]) if post else ({}, {})
    version = (state.get("scale_lattice") or {}).get("version", "?")
    lines = [
        f"# RSI security loop: {state.get('law_title', 'policy')}",
        "",
        f"- policy **{version}** · posture {first.get('score', 0):.3f} → **{last.get('score', 0):.3f}**",
        f"- findings: {len(world.inspections) if world else 0} · controls kept: "
        f"{len(state.get('controls', []) or [])} · rolled back: {len(state.get('rollbacks', []) or [])}",
    ]
    facts = list(state.get("surface_facts", []) or [])
    if facts:
        rows = surface.register(facts, surface.link(facts, state.get("scale_lattice") or {}))
        by = {k: sum(1 for r in rows if r["kind"] == k) for k in surface.KINDS}
        contra = sum(1 for r in rows if r["status"] == "contradicts")
        lines.append(
            f"- code facts: {by['code']} · social-engineering facts: {by['social']} · "
            f"contradicting the policy: {contra} · not covered: "
            f"{sum(1 for r in rows if r['status'] == 'uncovered')}"
        )
    evo = [r for r in state.get("evolution", []) or [] if r.get("universes")]
    if evo:
        n = sum(len(r["universes"]) for r in evo)
        best = max((u for r in evo for u in r["universes"]), key=lambda u: u["fitness"])
        adopted = [r["adopted"] for r in evo if r.get("adopted")]
        lines.append(
            f"- universes: {n} over {len(evo)} loops · best {best['id']} ({best['verdict']}) · "
            f"adopted: {', '.join(adopted) or 'none'}"
        )
    if world:
        lines.append(
            f"- agents: {len(world.live())} alive · ensembles: {len(world.ensembles)} · "
            f"L_G {world.ensemble_leak():.3f} · rounds/solve {world.rounds_per_solve():.2f}"
        )
    lines += [
        "",
        f"Full report: `{state.get('artifact_path', '')}` · NX-1 log: `{state.get('nx1_path', '')}`",
    ]
    return "\n".join(lines)


async def run_rsi_pipeline(
    llm,
    raw_text: str,
    *,
    law_title: str = "untitled-policy",
    iterations: int = 1,
    mutators: int | None = None,
    artifact_path: str = "",
    params: dict[str, Any] | None = None,
    log: LogFn = None,
    session_store: dict | None = None,
    session_id: str = "",
    on_event: EventFn = None,
    mapper_llm=None,
    editor_llm=None,
) -> dict[str, Any]:
    """Run the RSI security loop over a policy text.

    ``on_event`` (sync, called from the worker thread) receives
    ``{"phase": "enter"|"exit", "node", "iteration", "log", "state"}`` where
    ``state`` is :func:`snapshot_state`.
    """

    async def _log(msg: str) -> None:
        if log is None:
            return
        result = log(msg)
        if hasattr(result, "__await__"):
            await result

    given = dict(params or {})
    p = {**default_rsi_params(), **given}
    for key, val in (
        ("law_title", law_title),
        ("iterations", iterations),
        ("mutators", mutators),
        ("artifact_path", artifact_path),
    ):
        if key not in given and val is not None:
            p[key] = val
    if p.get("mutators") is None:
        p["mutators"] = 8
    offline = bool(p.get("offline", False))
    drive = str(p.get("drive") or "bal")
    if drive not in DRIVES:
        raise ValueError(f"drive must be one of {DRIVES}, got {drive!r}")
    if (
        llm is not None
        and not offline
        and p.get("require_local_driver", True)
        and not is_local_llm(llm)
    ):
        raise ValueError(
            "The RSI driver model must be local (in-process or llama.cpp on a local/private host); "
            f"got endpoint {llm_endpoint(llm)!r}. Pass a remote model as mapper_llm instead."
        )
    rsi_cfg = {
        "drive": drive,
        "lambda_am": float(p.get("lambda_am", 0.7)),
        "sensitivity": dict(p.get("sensitivity") or {}),
        "ticks_per_loop": int(p.get("ticks_per_loop", 3)),
        "heal_budget": p.get("heal_budget"),
        "stock_budget": p.get("stock_budget"),
        "objective": str(p.get("objective") or ""),
        "cot": bool(p.get("cot", True)),
        "mapper_arc_agi2": p.get("mapper_arc_agi2"),
        "surface_remote_mapping": bool(p.get("surface_remote_mapping", False)),
        "evolve": evolution.EvolveConfig.from_dict(p.get("evolve")).as_dict(),
    }
    surface_facts, surface_meta = await asyncio.to_thread(
        collect_surface, raw_text, str(p.get("repo_path") or ""), p.get("social_profile")
    )
    lora_cfg = dict(p.get("lora") or {})
    if lora_cfg.get("enabled") and not lora_cfg.get("base_url") and llm is not None:
        url = llm_endpoint(llm)
        if url:
            lora_cfg["base_url"] = url
    lora_cfg.setdefault("train_cmd", os.environ.get("RSI_LORA_TRAIN_CMD", ""))
    if llm is None or offline:
        driver_desc = "offline"
    else:
        driver_desc = llm_endpoint(llm) or "in-process"
    if offline:
        mapper_desc = "offline-lexicon"
    elif mapper_llm is not None:
        mapper_desc = llm_endpoint(mapper_llm) or "in-process"
    elif os.environ.get("OPENROUTER_API_KEY"):
        mapper_desc = "openrouter"
    else:
        mapper_desc = "driver-fallback"
    prog = dict(lora_cfg["program"]) if isinstance(lora_cfg.get("program"), dict) else {}
    if prog:
        lora_cfg["program"] = prog  # a copy: never mutate the caller's params
    if not prog.get("enabled"):
        editor_desc = "off"
    elif offline:
        editor_desc = "offline-program"
    elif editor_llm is not None:
        editor_desc = (
            getattr(editor_llm, "model_name", None)
            or getattr(editor_llm, "model", None)
            or "in-process"
        )
    elif os.environ.get("RSI_EDITOR_MODEL") and os.environ.get("OPENROUTER_API_KEY"):
        editor_desc = os.environ["RSI_EDITOR_MODEL"]
    else:
        editor_desc = "offline-program (no editing model configured)"
    if prog.get("enabled") and not prog.get("model"):
        prog["model"] = str(editor_desc)

    await _log(
        f"--- [RSI] security loop: title={p['law_title']!r} chars={len(raw_text)} iterations={p['iterations']} "
        f"drive={drive} λ_AM={rsi_cfg['lambda_am']} offline={offline} driver={driver_desc} mapper={mapper_desc} editor={editor_desc} ---"
    )
    if "code" in surface_meta:
        c = surface_meta["code"]
        await _log(
            f"  · repository: {c['files_scanned']} files scanned, {c['facts']} code facts "
            f"({', '.join(f'{k}×{v}' for k, v in sorted(c['by_rule'].items())) or 'none'})"
        )
    if "social" in surface_meta:
        s = surface_meta["social"]
        dropped = f"; not admitted: {len(s['dropped'])} fields" if s["dropped"] else ""
        await _log(
            f"  · social engineering: {s['roles']} roles, {s['facts']} facts "
            f"(authorized by {s['authorized_by']}){dropped}"
        )

    def _observer(phase: str, node: str, state: dict, update: dict | None) -> None:
        if on_event is None:
            return
        merged = {**state, **(update or {})}
        on_event(
            {
                "phase": phase,
                "node": node,
                "iteration": int(merged.get("iteration", 0) or 0),
                "log": list((update or {}).get("topology_log", []) or []),
                "state": snapshot_state(merged),
            }
        )

    def _run_sync() -> dict:
        from . import routing

        if on_event is not None:
            set_observer(_observer)
        old_offline = os.environ.get("NOMOSLOOP_OFFLINE")
        if offline:
            os.environ["NOMOSLOOP_OFFLINE"] = "1"
        if llm is not None and not offline:
            routing.set_llm_bridge(_make_bridge(llm))
        if mapper_llm is not None and not offline:
            routing.set_tier_bridge("mapper", _make_bridge(mapper_llm))
        if editor_llm is not None and not offline and prog.get("enabled"):
            routing.set_tier_bridge("editor", _make_bridge(editor_llm))
        try:
            return run_pipeline(
                raw_text,
                law_title=str(p["law_title"]),
                max_iterations=int(p["iterations"]),
                artifact_path=str(p.get("artifact_path") or ""),
                mutators=int(p["mutators"]),
                rsi_cfg=rsi_cfg,
                lora_cfg=lora_cfg,
                surface_facts=surface_facts,
                surface_meta=surface_meta,
            )
        finally:
            if on_event is not None:
                set_observer(None)
            routing.clear_llm_bridge()
            if offline:
                if old_offline is None:
                    os.environ.pop("NOMOSLOOP_OFFLINE", None)
                else:
                    os.environ["NOMOSLOOP_OFFLINE"] = old_offline

    try:
        state = await asyncio.to_thread(_run_sync)
        for line in state.get("topology_log", []):
            await _log(f"  · {line}")
        world = state.get("world") if isinstance(state.get("world"), World) else None
        lat = state.get("scale_lattice", {}) or {}
        post = state.get("posture_history", []) or []
        lst = state.get("lora_state")
        discrepancies = surface.register(surface_facts, surface.link(surface_facts, lat))
        result = {
            "mode": "rsi",
            "domain": "cybersecurity",
            "run_id": state.get("run_id"),
            "law_id": state.get("law_id"),
            "law_title": state.get("law_title"),
            "policy_version": lat.get("version"),
            "lattice_version": lat.get("version"),
            "levels": [str(lv.get("level_id")) for lv in lat.get("levels", [])],
            "num_levels": len(lat.get("levels", []) or []),
            "num_rules": sum(len(lv.get("rules", []) or []) for lv in lat.get("levels", []) or []),
            "drive": drive,
            "lambda_am": rsi_cfg["lambda_am"],
            "driver": {"endpoint": driver_desc, "local": True},
            "mapper": {"endpoint": mapper_desc, "arc_agi2_declared": rsi_cfg["mapper_arc_agi2"]},
            "editor": {"model": str(editor_desc), "programming": bool(prog.get("enabled"))},
            "posture_baseline": float(post[0]["score"]) if post else 0.0,
            "posture_final": float(post[-1]["score"]) if post else 0.0,
            "posture_history": post,
            "controls": state.get("controls", []) or [],
            "rollbacks": state.get("rollbacks", []) or [],
            "findings": sorted(world.inspections, key=lambda x: -x.get("severity", 0.0))
            if world
            else [],
            "ensembles": world.ensembles if world else [],
            "agents": [world.agent_view(a) for a in world.agents.values()] if world else [],
            "toc": [r.as_dict() for r in world.rows] if world else [],
            "events": world.events if world else [],
            "coverage": world.coverage_history if world else [],
            "rounds_per_solve": round(world.rounds_per_solve(), 4) if world else 0.0,
            "surface": surface_meta,
            "discrepancies": discrepancies,
            "evolution": state.get("evolution", []) or [],
            "lora": lst.as_dict() if hasattr(lst, "as_dict") else {},
            "iteration": int(state.get("iteration", 0) or 0),
            "artifact_path": state.get("artifact_path", ""),
            "artifact_code": state.get("artifact_code", ""),
            "nx1_path": state.get("nx1_path", ""),
            "scale_lattice": lat,
            "topology": describe_topology(),
            "topology_log": state.get("topology_log", []) or [],
            "routing_log": state.get("routing_log", []) or [],
            "proposed_solution": _summary(state, world),
            "snapshot": snapshot_state(state),
        }
        if session_store is not None and session_id in session_store:
            session_store[session_id]["rsi_result"] = result
            session_store[session_id]["final_solution"] = result
        await _log(f"FINAL_ANSWER: {json.dumps(result.get('proposed_solution'))}")
        await _log("SUCCESS: RSI LangGraph execution completed.")
        return result
    except Exception as e:
        await _log(f"RSI Pipeline Error: {e}")
        await _log(traceback.format_exc())
        err = {
            "mode": "rsi",
            "proposed_solution": f"RSI pipeline failed: {e}",
            "reasoning": str(e),
            "error": str(e),
        }
        if session_store is not None and session_id in session_store:
            session_store[session_id]["final_solution"] = err
        await _log(f"FINAL_ANSWER: {json.dumps(err)}")
        return err
