"""Scaling sweeps: grow one dimension, measure, and fit the growth exponent.

Cost is measured, not paid: every model call is synthetic. For each sweep the
table lists calls, virtual tokens, orchestration seconds (all model latency is
zero, so this is pure engine overhead) and peak memory or disk. ``slope`` is
the exponent k in ``y ~ x^k`` fitted on the upper half of the sweep: 1 is
linear, 2 is quadratic.
"""

from __future__ import annotations

import gc
import math
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any

from tests.scale.harness import run_scenario
from tests.scale.synth import SyntheticLLM


@dataclass
class Sweep:
    name: str
    dimension: str
    expect: dict[str, float]  # metric -> largest acceptable exponent
    rows: list[dict[str, Any]] = field(default_factory=list)
    slopes: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def slope(xs: list[float], ys: list[float]) -> float:
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    pts = pts[-max(3, (len(pts) + 1) // 2 + 1) :]
    if len(pts) < 2:
        return float("nan")
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / den if den else float("nan")


async def _point(pipeline: str, params: dict, memory: bool, **llm_kw) -> dict[str, Any]:
    gc.collect()
    llm = SyntheticLLM(1, max_calls=500_000, **llm_kw)
    t0 = time.process_time()
    rep, out = await run_scenario(pipeline, "scale", llm, params, timeout=900, check=False)
    row = {
        "calls": llm.meter.calls,
        "in_tokens": llm.meter.in_tokens,
        "out_tokens": llm.meter.out_tokens,
        "max_prompt_tokens": max(llm.meter.max_in_chars.values(), default=0) // 4,
        "peak_parallel": llm.meter.peak,
        "cpu_s": round(time.process_time() - t0, 3),
        "error": "; ".join(rep.problems)[:200],
    }
    if pipeline == "theory" and out.result:
        row["ledger_entries"] = len(out.result["body_of_work"]["entries"])
    if pipeline == "distill" and out.extra:
        row["disk_written_mb"] = round(out.extra.get("bytes_written", 0) / 1e6, 3)
        row["archive_mb"] = round(out.extra.get("archive_bytes", 0) / 1e6, 3)
        es = out.extra.get("epoch_seconds") or []
        row["last_epoch_s"] = round(es[-1], 4) if es else 0
    if memory:
        gc.collect()
        tracemalloc.start()
        llm2 = SyntheticLLM(1, max_calls=500_000, **llm_kw)
        await run_scenario(pipeline, "scale-mem", llm2, params, timeout=900, check=False)
        row["peak_mem_mb"] = round(tracemalloc.get_traced_memory()[1] / 1e6, 2)
        tracemalloc.stop()
    return row


async def _sweep(
    sweep: Sweep,
    pipeline: str,
    values: list[int],
    params_for,
    memory: bool = False,
    progress=None,
    **llm_kw,
) -> Sweep:
    for v in values:
        row = await _point(pipeline, params_for(v), memory, **llm_kw)
        sweep.rows.append({sweep.dimension: v, **row})
        if progress:
            progress(f"  {sweep.name} {sweep.dimension}={v}: {row['calls']} calls, {row['cpu_s']}s")
    xs = [r[sweep.dimension] for r in sweep.rows]
    for metric, limit in sweep.expect.items():
        ys = [r.get(metric, 0) for r in sweep.rows]
        k = slope(xs, ys)
        sweep.slopes[metric] = round(k, 2)
        if k == k and k > limit:
            sweep.flags.append(
                f"{metric} grows like {sweep.dimension}^{k:.2f} (expected at most ^{limit})"
            )
    errs = [
        f"{sweep.dimension}={r[sweep.dimension]}: {r['error']}" for r in sweep.rows if r["error"]
    ]
    sweep.flags += errs
    return sweep


def ledger_microbench(sizes: list[int]) -> Sweep:
    """Cost of one write and one read as the body of work grows (no model calls at all)."""
    import random

    from deepthink.theory.ledger import TheoryLedger

    from tests.scale.synth import WORDS

    sweep = Sweep("theory ledger (per operation)", "entries", {"add_us": 0.3, "retrieve_us": 0.3})
    rng = random.Random(1)
    ledger = TheoryLedger()

    def statement(i: int) -> str:
        return f"Claim {i} " + " ".join(rng.choices(WORDS, k=8))

    n = 0
    for size in sizes:
        while n < size:
            ledger.add(
                member=n % 8, round_=1, cell="r.0", kind="lemma", statement=statement(n), answer="7"
            )
            n += 1
        t0 = time.perf_counter()
        for i in range(40):
            ledger.add(
                member=1, round_=1, cell="r.0", kind="lemma", statement=statement(n + i), answer="7"
            )
        add = (time.perf_counter() - t0) / 40
        n += 40
        t0 = time.perf_counter()
        for i in range(40):
            ledger.retrieve(statement(i), member=1, k=6)
        read = (time.perf_counter() - t0) / 40
        sweep.rows.append(
            {"entries": size, "add_us": round(add * 1e6, 1), "retrieve_us": round(read * 1e6, 1)}
        )
    xs = [r["entries"] for r in sweep.rows]
    for metric, limit in sweep.expect.items():
        k = slope(xs, [r[metric] for r in sweep.rows])
        sweep.slopes[metric] = round(k, 2)
        if k > limit:
            sweep.flags.append(f"{metric} grows like entries^{k:.2f}: total cost is superlinear")
    return sweep


async def run_sweeps(tier: str = "smoke", progress=None) -> list[Sweep]:
    s = {"smoke": 0, "full": 1, "stress": 2}[tier]

    def pick(smoke, full, stress):
        return [smoke, full, stress][s]

    sweeps: list[Sweep] = []
    theory = {
        "rounds": 2,
        "quorum": 1.0,
        "max_depth": 2,
        "branching": 3,
        "max_cells": 10,
        "concurrency": 8,
        "polish": False,
    }

    sweeps.append(
        await _sweep(
            Sweep(
                "theory: swarm size",
                "swarm_size",
                {"calls": 1.15, "in_tokens": 1.2, "cpu_s": 1.35, "peak_mem_mb": 1.25},
            ),
            "theory",
            pick([2, 4, 8, 16, 32], [2, 4, 8, 16, 32, 64, 128], [4, 8, 16, 32, 64, 128]),
            lambda v: {**theory, "swarm_size": v},
            memory=True,
            progress=progress,
            unique=True,
            split=1.0,
            agree=0.3,
            pad=600,
        )
    )
    sweeps.append(
        await _sweep(
            Sweep(
                "theory: cells per QNN",
                "max_cells",
                {"calls": 1.15, "in_tokens": 1.25, "cpu_s": 1.5},
            ),
            "theory",
            pick([4, 8, 16, 32], [4, 8, 16, 32, 64], [8, 16, 32, 64]),
            lambda v: {**theory, "swarm_size": 8, "max_cells": v, "max_depth": 4, "branching": 6},
            progress=progress,
            unique=True,
            split=1.0,
            pad=600,
        )
    )
    sweeps.append(
        await _sweep(
            Sweep("theory: rounds", "rounds", {"calls": 1.15, "in_tokens": 1.25, "cpu_s": 1.5}),
            "theory",
            pick([1, 2, 4, 8], [1, 2, 4, 8, 10], [2, 4, 8, 10]),
            lambda v: {**theory, "swarm_size": 8, "rounds": v},
            progress=progress,
            unique=True,
            split=1.0,
            agree=0.2,
            pad=600,
        )
    )
    sweeps.append(
        ledger_microbench(
            pick(
                [500, 1000, 2000, 4000], [1000, 2000, 4000, 8000, 16000], [4000, 8000, 16000, 32000]
            )
        )
    )

    qnn = {"qnn_mode": "manual", "manual_layers": 3, "num_epochs": 2}
    sweeps.append(
        await _sweep(
            Sweep(
                "qnn: width",
                "manual_width",
                {"calls": 1.15, "in_tokens": 1.25, "max_prompt_tokens": 0.25, "cpu_s": 1.5},
            ),
            "qnn",
            pick([2, 4, 8, 16], [2, 4, 8, 16, 32, 64], [8, 16, 32, 64, 100]),
            lambda v: {**qnn, "manual_width": v},
            progress=progress,
            pad=1500,
        )
    )
    sweeps.append(
        await _sweep(
            Sweep(
                "qnn: epochs",
                "num_epochs",
                {"calls": 1.15, "in_tokens": 1.25, "max_prompt_tokens": 0.25},
            ),
            "qnn",
            pick([1, 2, 4, 8], [2, 4, 8, 16, 20], [4, 8, 16, 20]),
            lambda v: {**qnn, "manual_width": 4, "num_epochs": v},
            progress=progress,
            pad=1500,
        )
    )
    sweeps.append(
        await _sweep(
            Sweep(
                "qdad: grid size",
                "grid_size",
                {"calls": 2.15, "in_tokens": 2.3, "peak_parallel": 2.15},
            ),
            "qdad",
            pick([2, 4, 6, 8], [2, 3, 4, 6, 8], [2, 4, 6, 8]),
            lambda v: {"grid_size": v, "denoising_steps": 3},
            progress=progress,
            pad=600,
        )
    )
    sweeps.append(
        await _sweep(
            Sweep(
                "distillation: epochs",
                "max_epochs",
                {
                    "calls": 1.15,
                    "in_tokens": 1.25,
                    "cpu_s": 1.35,
                    "peak_mem_mb": 1.25,
                    "archive_mb": 1.25,
                    "disk_written_mb": 1.35,
                },
            ),
            "distill",
            pick([4, 8, 16, 32], [5, 10, 20, 40, 80], [10, 20, 40, 80, 120]),
            lambda v: {"max_epochs": v},
            memory=True,
            progress=progress,
            pad=1500,
        )
    )
    sweeps.append(
        await _sweep(
            Sweep(
                "rsi: policy clauses",
                "clauses",
                {"calls": 1.15, "in_tokens": 1.25, "cpu_s": 1.35, "peak_mem_mb": 1.25},
            ),
            "rsi",
            pick([25, 50, 100, 200], [25, 50, 100, 200, 400, 800], [100, 200, 400, 800]),
            lambda v: {"clauses": v, "iterations": 1, "lora": {}},
            memory=True,
            progress=progress,
        )
    )
    sweeps.append(
        await _sweep(
            Sweep("rsi: iterations", "iterations", {"calls": 1.15, "cpu_s": 1.5}),
            "rsi",
            pick([1, 2, 4, 8], [1, 2, 4, 8, 16], [2, 4, 8, 16]),
            lambda v: {"clauses": 40, "iterations": v, "lora": {}},
            progress=progress,
        )
    )
    return sweeps


async def token_calibration() -> list[dict[str, Any]]:
    """Measured input tokens per call with realistic answer sizes, against the estimator."""
    from deepthink.cost import (
        DEFAULT_TOKENS_IN,
        DEFAULT_TOKENS_OUT,
        estimate_qdad_cost,
        estimate_qnn_cost,
        estimate_rsi_cost,
        estimate_theory_cost,
    )

    configs = [
        (
            "theory (default)",
            "theory",
            {
                "swarm_size": 5,
                "rounds": 2,
                "quorum": 0.5,
                "max_depth": 2,
                "max_cells": 10,
                "qnn_mode": "auto",
                "polish": False,
            },
            estimate_theory_cost(5, 2),
        ),
        (
            "qnn 3x3x2",
            "qnn",
            {"qnn_mode": "manual", "manual_layers": 3, "manual_width": 3, "num_epochs": 2},
            estimate_qnn_cost(3, 3, 2),
        ),
        (
            "qdad 3x3, 2 steps",
            "qdad",
            {"grid_size": 3, "denoising_steps": 2},
            estimate_qdad_cost(3, 2),
        ),
        (
            "rsi 2 loops, 24 clauses",
            "rsi",
            {"clauses": 24, "iterations": 2, "lora": {}},
            estimate_rsi_cost(2, 8, clauses=24),
        ),
        ("distillation, 10 epochs", "distill", {"max_epochs": 10}, None),
    ]
    rows = []
    for label, pipeline, params, est in configs:
        llm = SyntheticLLM(3, pad=3200)  # about 800 output tokens per free-text field
        await run_scenario(pipeline, "calibrate", llm, params, timeout=300, check=False)
        m = llm.meter
        calls = max(1, m.calls)
        rows.append(
            {
                "config": label,
                "calls": m.calls,
                "estimated_calls": est.llm_calls if est else None,
                "in_tokens": m.in_tokens,
                "out_tokens": m.out_tokens,
                "in_per_call": m.in_tokens // calls,
                "estimator_in_per_call": DEFAULT_TOKENS_IN,
                "out_per_call": m.out_tokens // calls,
                "estimator_out_per_call": DEFAULT_TOKENS_OUT,
                "estimated_total": est.est_tokens_total if est else None,
                "measured_total": m.in_tokens + m.out_tokens,
            }
        )
    return rows
