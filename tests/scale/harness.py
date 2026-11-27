"""Drivers and invariants: run any pipeline on a SyntheticLLM and judge the outcome."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.scale.synth import Runaway, SyntheticLLM  # noqa: E402

PIPELINES = ("theory", "qnn", "qdad", "distill", "rsi")


@dataclass
class Outcome:
    """What a driver saw."""

    result: Any = None
    logs: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    pipeline: str
    scenario: str
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    calls: int = 0
    faulted: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    peak: int = 0
    seconds: float = 0.0

    @property
    def status(self) -> str:
        return "FAIL" if self.problems else ("WARN" if self.warnings else "OK")


def _where(exc: BaseException) -> str:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if "deepthink" in f.filename]
    if not frames:
        return ""
    f = frames[-1]
    return f" @ {Path(f.filename).relative_to(ROOT).as_posix()}:{f.lineno}"


# ── drivers ──────────────────────────────────────────────────────────────

PROBLEM = "Let a_1 = 2 and a_(n+1) = a_n + 1. Find a_2025 and justify it."


async def drive_theory(llm: SyntheticLLM, p: dict[str, Any], out: Outcome) -> None:
    from deepthink.theory import run_theory_pipeline

    out.result = await run_theory_pipeline(llm, p.pop("problem", PROBLEM), p, log=out.logs.append)
    out.extra["params"] = out.result["params"]


async def drive_qnn(llm: SyntheticLLM, p: dict[str, Any], out: Outcome) -> None:
    from deepthink.qnn import run_qnn_pipeline

    out.result = await run_qnn_pipeline(llm, p.pop("problem", PROBLEM), p, log=out.logs.append)


async def drive_qdad(llm: SyntheticLLM, p: dict[str, Any], out: Outcome) -> None:
    from deepthink.qdad import run_qdad_pipeline

    out.result = await run_qdad_pipeline(
        llm=llm, params=p, user_prompt="a calm writing app", log=out.logs.append
    )


async def drive_distill(llm: SyntheticLLM, p: dict[str, Any], out: Outcome) -> None:
    from deepthink import knowledge_distillation as kd

    max_epochs = int(p.get("max_epochs", 3))
    budget = int(p.get("token_budget", 10**12))
    tmp = tempfile.mkdtemp(prefix="odt_distill_")
    graph = kd.DistillationGraph(
        llm, ["a", "b"], "Anchor question?", token_budget=budget, output_dir=tmp
    )
    epoch_seconds: list[float] = []
    epoch_tokens: list[int] = []
    written = 0
    more = True
    while more and graph.is_running and graph.epochs_run < max_epochs:
        t0, before = time.perf_counter(), graph.total_tokens
        more = await graph.run_epoch()
        epoch_seconds.append(time.perf_counter() - t0)
        epoch_tokens.append(graph.total_tokens - before)
        for path in (graph.dataset_path, graph.topology_archive_path):
            written += os.path.getsize(path) if os.path.isfile(path) else 0
        while not graph.log_queue.empty():
            out.logs.append(graph.log_queue.get_nowait())
    out.result = {
        "epochs": graph.epochs_run,
        "qa_pairs": len(graph.distilled_data),
        "total_tokens": graph.total_tokens,
        "budget": budget,
        "stopped": not more,
        "agents": len(graph._flat_agents()),
    }
    out.extra.update(
        graph=graph,
        max_epochs=max_epochs,
        epoch_seconds=epoch_seconds,
        epoch_tokens=epoch_tokens,
        bytes_written=written,
        archive_bytes=os.path.getsize(graph.topology_archive_path),
        dataset_bytes=os.path.getsize(graph.dataset_path),
    )


def make_policy(clauses: int) -> str:
    """A synthetic security policy with levels, numbered clauses and a few clashes."""
    verbs = ["log", "review", "encrypt", "approve", "monitor", "rotate", "revoke", "report"]
    nouns = ["admin access", "backups", "service accounts", "vendor links", "audit trails"]
    sections = ["PURPOSE", "POLICY", "CONTROLS", "ASSETS"]
    lines = ["# Synthetic Security Policy", ""]
    per = max(1, clauses // len(sections))
    n = 0
    for s, title in enumerate(sections):
        lines.append(f"## {s + 1}. {title}")
        count = per if s < len(sections) - 1 else max(1, clauses - n)
        for _ in range(count):
            n += 1
            v, o = verbs[n % len(verbs)], nouns[(n * 3) % len(nouns)]
            neg = "must not" if n % 11 == 0 else "must"
            lines.append(f"§{n} All staff {neg} {v} {o} for system {n % 17} every week.")
        lines.append("")
    return "\n".join(lines)


async def drive_rsi(llm: SyntheticLLM, p: dict[str, Any], out: Outcome) -> None:
    from deepthink.nomosloop import run_rsi_pipeline

    text = p.pop("text", None) or make_policy(int(p.pop("clauses", 24)))
    tmp = tempfile.mkdtemp(prefix="odt_rsi_")
    out.result = await run_rsi_pipeline(
        llm,
        text,
        params=p,
        artifact_path=os.path.join(tmp, "report.md"),
        log=out.logs.append,
        mapper_llm=llm,
        editor_llm=llm,
    )


DRIVERS = {
    "theory": drive_theory,
    "qnn": drive_qnn,
    "qdad": drive_qdad,
    "distill": drive_distill,
    "rsi": drive_rsi,
}

SMALL = {
    "theory": {
        "swarm_size": 4,
        "rounds": 2,
        "max_depth": 2,
        "quorum": 0.9,
        "seed": 1,
        "qnn_mode": "manual",
        "manual_layers": 3,
        "manual_width": 2,
        "polish": True,
    },
    "qnn": {"qnn_mode": "manual", "manual_layers": 2, "manual_width": 3, "num_epochs": 2},
    "qdad": {"grid_size": 3, "denoising_steps": 2},
    "distill": {"max_epochs": 3},
    "rsi": {
        "iterations": 2,
        "drive": "bal",
        "clauses": 24,
        "lora": {"enabled": False, "program": {"enabled": True, "adapter_ids": [0]}},
    },
}


# ── invariants ───────────────────────────────────────────────────────────


def _json_safe(value: Any, rep: Report, what: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as e:
        rep.problems.append(f"{what} is not strict-JSON serializable: {e}")


def _control_lines(logs: list[str], rep: Report) -> None:
    """The web UI treats any log line containing FINAL_ANSWER: as the final answer."""
    final = [x for x in logs if x.startswith("FINAL_ANSWER:")]
    forged = [x for x in logs if "FINAL_ANSWER:" in x and not x.startswith("FINAL_ANSWER:")]
    if forged:
        rep.problems.append(
            f"model text puts 'FINAL_ANSWER:' inside {len(forged)} progress line(s); "
            f"the UI would take it as the result: {forged[0][:90]!r}"
        )
    if len(final) > 1:
        rep.problems.append(f"{len(final)} FINAL_ANSWER lines in one run")


def check_theory(out: Outcome, llm: SyntheticLLM, rep: Report) -> None:
    from deepthink.cost import estimate_theory_cost

    r, p = out.result, out.extra["params"]
    _json_safe(r, rep, "theory result")
    _control_lines(out.logs, rep)
    if not any(x.startswith("FINAL_ANSWER:") for x in out.logs):
        rep.problems.append("no FINAL_ANSWER line: the UI would wait forever")
    est = estimate_theory_cost(
        p["swarm_size"],
        p["rounds"],
        max_depth=p["max_depth"],
        branching=p["branching"],
        max_cells=p["max_cells"],
        verify=p["verify"],
        max_arbitrations=p["max_arbitrations"],
        polish=p["polish"],
    )
    if r["calls"]["total"] > est.llm_calls:
        rep.problems.append(f"{r['calls']['total']} calls exceed the estimate {est.llm_calls}")
    if r["calls"]["total"] != llm.meter.calls:
        rep.problems.append(f"reported calls {r['calls']['total']} != actual {llm.meter.calls}")
    if llm.meter.peak > p["concurrency"]:
        rep.problems.append(f"concurrency {llm.meter.peak} exceeds cap {p['concurrency']}")
    if p["max_calls"] and llm.meter.calls > p["max_calls"]:
        rep.problems.append(f"call budget {p['max_calls']} exceeded: {llm.meter.calls}")
    if abs(sum(r["weights"]) - 1) > 1e-3:
        rep.problems.append(f"weights sum to {sum(r['weights'])}")
    for rec in r["vote_history"]:
        s = sum(c["support"] for c in rec["clusters"])
        if s > 1.0 + 1e-3:
            rep.problems.append(f"round {rec['round']} support sums to {s}")
    if r["converged"] and (r["support"] + 1e-6 < p["quorum"] or not r["final_answer"]):
        rep.problems.append("converged without quorum support or without an answer")
    if str(r["final_answer"]).startswith(("(cell failed", "(the call budget")):
        rep.problems.append(f"an error message is reported as the answer: {r['final_answer'][:80]}")
    bow = r["body_of_work"]
    entries = {e["id"]: e for e in bow["entries"]}
    for e in entries.values():
        missing = [d for d in e["depends_on"] if d not in entries]
        if missing:
            rep.problems.append(f"{e['id']} depends on unknown {missing}")
        bad = {"refuted", "suspect"}
        if (
            e["status"] not in bad
            and e["derivations"]
            and all(
                d and any(entries[x]["status"] in bad for x in d if x in entries)
                for d in e["derivations"]
            )
        ):
            rep.problems.append(
                f"{e['id']} is {e['status']} although every derivation rests on a refuted entry"
            )
    open_ids = {c["a"] for c in bow["conflicts"] if c["status"] == "open"} | {
        c["b"] for c in bow["conflicts"] if c["status"] == "open" and c["b"]
    }
    for e in entries.values():
        if e["status"] == "disputed" and e["id"] not in open_ids:
            rep.problems.append(f"{e['id']} stays disputed with no open conflict")
    for m in r["members"]:
        tree = m["tree"]
        if len(tree) > p["max_cells"]:
            rep.problems.append(f"QNN {m['index']} grew {len(tree)} cells > {p['max_cells']}")
        if tree and max(c["depth"] for c in tree) > p["max_depth"]:
            rep.problems.append(f"QNN {m['index']} exceeded max depth")
    if llm.plan.rate >= 1 and llm.plan.only is None and r["final_answer"] and r["converged"]:
        if llm.plan.fault in ("raise", "empty", "garbage"):
            rep.warnings.append("claims a converged answer although every call failed")


def check_qnn(out: Outcome, llm: SyntheticLLM, rep: Report) -> None:
    from deepthink.cost import estimate_qnn_cost

    r = out.result
    _json_safe(r, rep, "qnn result")
    _control_lines(out.logs, rep)
    text = r.get("proposed_solution")
    if not isinstance(text, str) or not text.strip():
        rep.problems.append(f"empty or non-text report: {type(text).__name__}")
    if not any(x.startswith("FINAL_ANSWER:") for x in out.logs):
        rep.problems.append("no FINAL_ANSWER line")
    t = r.get("topology") or {}
    est = estimate_qnn_cost(
        t.get("layers", 1),
        t.get("width", 1),
        t.get("epochs", 1),
        qnn_mode=t.get("qnn_mode", "manual"),
    )
    if llm.meter.calls > est.llm_calls:
        rep.problems.append(f"{llm.meter.calls} calls exceed the estimate {est.llm_calls}")
    if llm.plan.fault in ("empty", "garbage") and llm.plan.rate >= 1 and llm.plan.only is None:
        rep.warnings.append("reports success although every completion was useless")


def check_qdad(out: Outcome, llm: SyntheticLLM, rep: Report) -> None:
    r = out.result
    _json_safe(r, rep, "qdad result")
    _control_lines(out.logs, rep)
    if not any(x.startswith("FINAL_ANSWER:") for x in out.logs):
        rep.problems.append("no FINAL_ANSWER line")
    text = str(r.get("proposed_solution") or "")
    failed = text.startswith("QDAD pipeline failed")
    if not text.strip():
        rep.problems.append("empty proposed_solution")
    if not failed:
        prompt = str(r.get("app_build_prompt") or "")
        cells = [c for row in r.get("feature_matrix") or [] for c in row]
        errors = sum(1 for c in cells if c.startswith("[noise error]") or not c.strip())
        if not prompt.strip():
            rep.warnings.append("success with an empty app build prompt")
        if cells and errors == len(cells):
            rep.warnings.append("success although every grid cell is an error or empty")
        n = r.get("grid_size", 0)
        expected = 2 + n * n * (1 + r.get("denoising_steps", 0))
        if llm.meter.calls != expected:
            rep.problems.append(f"{llm.meter.calls} calls, expected {expected}")


def check_distill(out: Outcome, llm: SyntheticLLM, rep: Report) -> None:
    r = out.result
    g = out.extra["graph"]
    if r["agents"] != 12:
        rep.problems.append(f"topology has {r['agents']} agents, not 12")
    if not r["stopped"] and r["epochs"] >= out.extra["max_epochs"] and r["total_tokens"] == 0:
        rep.problems.append(
            f"never reaches its budget: {r['epochs']} epochs, 0 tokens counted, "
            f"{llm.meter.calls} calls made — the run would loop forever"
        )
    for path in (g.dataset_path, g.topology_archive_path):
        try:
            with open(path, encoding="utf-8") as f:
                json.load(f)
        except (OSError, ValueError) as e:
            rep.problems.append(f"{Path(path).name} unreadable: {e}")
    for a in g._flat_agents():
        if not isinstance(a.system_prompt, str) or not a.system_prompt.strip():
            rep.problems.append(f"agent {a.id} system prompt became {a.system_prompt!r:.60}")
            break
        if not isinstance(a.current_question, str):
            rep.problems.append(f"agent {a.id} question became {type(a.current_question).__name__}")
            break
        if not isinstance(a.attributes, list) or not all(isinstance(x, str) for x in a.attributes):
            rep.problems.append(f"agent {a.id} attributes became {a.attributes!r:.60}")
            break
    bad_answers = [q for q in g.distilled_data if not isinstance(q["answer"], str)]
    if bad_answers:
        rep.problems.append(
            f"{len(bad_answers)} dataset answers are {type(bad_answers[0]['answer']).__name__}, not text"
        )
    if g.distilled_data and not all(str(q["question"]).strip() for q in g.distilled_data):
        rep.problems.append("dataset contains empty questions")
    empty = sum(1 for q in g.distilled_data if not str(q["answer"]).strip())
    if empty:
        rep.warnings.append(f"{empty}/{len(g.distilled_data)} dataset answers are empty")
    errs = sum(1 for x in out.logs if "Mirror Descent error" in x)
    if errs:
        rep.warnings.append(f"{errs} agent evaluations raised inside mirror descent")


def check_rsi(out: Outcome, llm: SyntheticLLM, rep: Report) -> None:
    r = out.result
    _json_safe(r, rep, "rsi result")
    _control_lines(out.logs, rep)
    if r.get("error"):
        rep.problems.append(f"run failed: {r['error'][:160]}")
        return
    if r["posture_final"] + 1e-9 < r["posture_baseline"]:
        rep.problems.append("posture regressed")
    p = r["posture_final"]
    if not math.isfinite(p):
        rep.problems.append("posture is not finite")


CHECKS = {
    "theory": check_theory,
    "qnn": check_qnn,
    "qdad": check_qdad,
    "distill": check_distill,
    "rsi": check_rsi,
}


async def run_scenario(
    pipeline: str,
    scenario: str,
    llm: SyntheticLLM,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 60.0,
    check: bool = True,
) -> tuple[Report, Outcome]:
    rep = Report(pipeline, scenario)
    out = Outcome()
    p = json.loads(json.dumps({**SMALL[pipeline], **(params or {})}))
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(DRIVERS[pipeline](llm, p, out), timeout)
        if check:
            CHECKS[pipeline](out, llm, rep)
    except Runaway as e:
        rep.problems.append(f"runaway: {e}")
    except asyncio.TimeoutError:
        rep.problems.append(f"did not finish within {timeout:.0f}s")
    except Exception as e:  # the pipeline let an exception escape
        rep.problems.append(f"crashed: {type(e).__name__}: {str(e)[:140]}{_where(e)}")
    rep.seconds = time.perf_counter() - t0
    m = llm.meter
    rep.calls, rep.faulted, rep.peak = m.calls, m.faulted, m.peak
    rep.in_tokens, rep.out_tokens = m.in_tokens, m.out_tokens
    return rep, out
