"""Fault matrix: every pipeline × every fault, globally and per prompt kind."""

from __future__ import annotations

from tests.scale.harness import PIPELINES, Report, run_scenario
from tests.scale.synth import FAULTS, FaultPlan, SyntheticLLM

TARGETED = ("raise", "garbage", "wrong_types")


async def kinds_of(pipeline: str) -> list[str]:
    llm = SyntheticLLM(1)
    await run_scenario(pipeline, "probe", llm, check=False)
    return sorted(llm.meter.by_kind)


async def fault_matrix(tier: str = "smoke", progress=None) -> list[Report]:
    reports: list[Report] = []
    seeds = {"smoke": 1, "full": 5, "stress": 25}[tier]

    async def one(pipeline, scenario, plan, seed=1, params=None):
        llm = SyntheticLLM(seed, plan=plan, max_calls=20_000)
        extra = dict(params or {})
        if pipeline == "distill":
            extra.setdefault("max_epochs", 4)
        rep, _ = await run_scenario(pipeline, scenario, llm, extra, timeout=40)
        reports.append(rep)

    for pipeline in PIPELINES:
        await one(pipeline, "clean", FaultPlan())
        for fault in FAULTS:
            await one(pipeline, f"all:{fault}", FaultPlan(fault, 1.0))
            for seed in range(1, seeds + 1):
                await one(pipeline, f"30%:{fault} seed{seed}", FaultPlan(fault, 0.3), seed)
        for kind in await kinds_of(pipeline):
            for fault in TARGETED:
                await one(
                    pipeline,
                    f"only {kind}:{fault}",
                    FaultPlan(fault, 1.0, frozenset({kind})),
                )
        if progress:
            progress(f"  faults: {pipeline} done ({len(reports)} runs so far)")
    return reports
