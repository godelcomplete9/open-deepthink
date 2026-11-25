"""
RSI pipeline entrypoint — wire params → LangGraph RSI loop → result.

This is the fourth open-deepthink section: **recursive self-improvement (RSI)
over formal artifacts** (code repositories, math models, physics models, a
collection of repos and documentation) that follow formal structures.

Model-agnostic: any LangChain-compatible chat model works (OpenRouter,
llama.cpp, mocks). When ``llm`` is provided, every cheap/frontier sub-agent
call is routed through it via an injected bridge; when ``llm`` is ``None`` (or
``offline=True``) the deterministic offline heuristics run keyless.
"""

from __future__ import annotations

import asyncio
import json
import os
import traceback
from collections.abc import Callable
from typing import Any

from .graph import describe_topology, run_pipeline

LogFn = Callable[[str], Any] | None


def default_rsi_params() -> dict[str, Any]:
    """Documented RSI defaults for harnesses / library callers."""
    return {
        "iterations": 1,
        "mutators": None,  # None -> env NOMOSLOOP_MUTATORS or 6; 0 = baseline
        "law_title": "untitled-law",
        "artifact_path": "",
        "offline": False,
    }


def _make_bridge(llm):
    """Adapt a LangChain chat model to the engine's sync ``call_llm`` bridge."""

    def bridge(prompt: str, task_kind: str, system: str, max_tokens: int, tier: str):
        messages: list[tuple[str, str]] = []
        if system:
            messages.append(("system", system))
        messages.append(("user", prompt))
        resp = llm.invoke(messages)
        content = getattr(resp, "content", resp)
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    parts.append(str(p.get("text", "")))
                else:
                    parts.append(str(p))
            content = "".join(parts)
        return None if content is None else str(content)

    return bridge


def _summarize(state: dict) -> str:
    """Compact markdown summary used as ``proposed_solution``."""
    lat = state.get("scale_lattice", {}) or {}
    caps = state.get("capacity_history", []) or []
    final = caps[-1] if caps else {}
    levels = [str(lv.get("level_id")) for lv in lat.get("levels", [])]
    lines = [
        f"# RSI — {state.get('law_title', 'untitled-law')}",
        "",
        f"- lattice: **{lat.get('version', '?')}**  levels: {', '.join(levels) or '—'}",
        f"- motifs: {len(state.get('motifs', []))}  holdout: {len(state.get('holdout_motifs', []))}",
        f"- proposals: {len(state.get('mutation_proposals', []))}  "
        f"accepted: {len(state.get('accepted_mutations', []))}  "
        f"rollbacks: {len(state.get('rollbacks', []))}",
        f"- measured capacity: baseline={state.get('baseline_capacity', 0.0):.3f} "
        f"→ final={final.get('capacity', 0.0):.3f}  "
        f"instances={final.get('instances', 0)}",
        f"- mesh: {state.get('fitness_report', {}).get('mesh_size', 0)} instantiations",
        f"- artifact: `{state.get('artifact_path', '')}`",
        "",
        "## Scale lattice",
    ]
    for lvl in lat.get("levels", []):
        lines.append(f"### {lvl.get('level_id')} — {lvl.get('name', '')}")
        lines.append(f"- BASE: {lvl.get('base', '?')}")
        lines.append(f"- MODULUS: {lvl.get('modulus', '?')}")
        for r in lvl.get("rules", []):
            lines.append(
                f"- `{r.get('id')}` {r.get('name', '')}: {r.get('pattern', '')} ⇒ {r.get('rewrite', '')}"
            )
    return "\n".join(lines)


async def run_rsi_pipeline(
    llm,
    raw_text: str,
    *,
    law_title: str = "untitled-law",
    iterations: int = 1,
    mutators: int | None = None,
    artifact_path: str = "",
    params: dict[str, Any] | None = None,
    log: LogFn = None,
    session_store: dict | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    """
    Run the full RSI pipeline over a formal artifact.

    Parameters
    ----------
    llm :
        LangChain-compatible chat model (or ``None`` for offline heuristics).
    raw_text :
        The formal artifact text (code repo docs, math/physics model, spec, …).
    law_title, iterations, mutators, artifact_path :
        Loop controls; ``params`` overrides them.
    params :
        Optional dict merged over :func:`default_rsi_params`.
    log :
        Optional sync/async callable receiving log lines.
    session_store, session_id :
        Optional disk-backed store to persist the final result.

    Returns
    -------
    dict with the final lattice, measured capacity history, artifact, and logs.
    """

    async def _log(msg: str):
        if log is None:
            return
        result = log(msg)
        if hasattr(result, "__await__"):
            await result

    p = {**default_rsi_params(), **(params or {})}
    # Explicit keyword arguments win unless ``params`` set the key itself.
    if "law_title" not in (params or {}):
        p["law_title"] = law_title
    if "iterations" not in (params or {}):
        p["iterations"] = iterations
    if "mutators" not in (params or {}):
        p["mutators"] = mutators
    if "artifact_path" not in (params or {}):
        p["artifact_path"] = artifact_path
    law_title = str(p.get("law_title") or "untitled-law")
    iterations = int(p.get("iterations", 1))
    mutators = p.get("mutators", None)
    artifact_path = str(p.get("artifact_path") or "")
    offline = bool(p.get("offline", False))

    await _log(
        f"--- [RSI] RSI over formal artifact: title={law_title!r} "
        f"chars={len(raw_text)} iterations={iterations} mutators={mutators} "
        f"offline={offline} ---"
    )
    await _log(
        "LOG: [RSI] Philosophy: compression + generativity beats description; "
        "every pattern is elevatable; no descriptive bloat without worldly capacity."
    )

    def _run_sync() -> dict:
        from . import routing

        bridge = None
        if llm is not None and not offline:
            bridge = _make_bridge(llm)
            routing.set_llm_bridge(bridge)
        old_offline = os.environ.get("NOMOSLOOP_OFFLINE")
        if offline:
            os.environ["NOMOSLOOP_OFFLINE"] = "1"
        try:
            return run_pipeline(
                raw_text,
                law_title=law_title,
                max_iterations=iterations,
                artifact_path=artifact_path,
                mutators=mutators,
            )
        finally:
            if bridge is not None:
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
        await _log(f"LOG: [RSI] artifact written to {state.get('artifact_path')}")

        lat = state.get("scale_lattice", {}) or {}
        caps = state.get("capacity_history", []) or []
        final = caps[-1] if caps else {}
        result = {
            "mode": "rsi",
            "run_id": state.get("run_id"),
            "law_id": state.get("law_id"),
            "law_title": state.get("law_title"),
            "lattice_version": lat.get("version"),
            "levels": [str(lv.get("level_id")) for lv in lat.get("levels", [])],
            "num_levels": len(lat.get("levels", []) or []),
            "num_rules": sum(len(lv.get("rules", []) or []) for lv in lat.get("levels", []) or []),
            "motifs": len(state.get("motifs", []) or []),
            "holdout_motifs": len(state.get("holdout_motifs", []) or []),
            "proposals": len(state.get("mutation_proposals", []) or []),
            "accepted_mutations": [m.get("id") for m in state.get("accepted_mutations", []) or []],
            "rollbacks": state.get("rollbacks", []) or [],
            "baseline_capacity": float(state.get("baseline_capacity", 0.0) or 0.0),
            "final_capacity": float(final.get("capacity", 0.0) or 0.0),
            "final_instances": int(final.get("instances", 0) or 0),
            "final_coverage": float(final.get("coverage", 0.0) or 0.0),
            "capacity_history": caps,
            "mesh_size": state.get("fitness_report", {}).get("mesh_size", 0),
            "mesh_stats": state.get("fitness_report", {}).get("mesh_stats", {}),
            "iteration": int(state.get("iteration", 0) or 0),
            "artifact_path": state.get("artifact_path", ""),
            "artifact_code": state.get("artifact_code", ""),
            "scale_lattice": lat,
            "topology": describe_topology(),
            "topology_log": state.get("topology_log", []) or [],
            "routing_log": state.get("routing_log", []) or [],
            "proposed_solution": _summarize(state),
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
