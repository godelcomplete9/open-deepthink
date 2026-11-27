#!/usr/bin/env python3
"""Live tests on FREE OpenRouter models, with small topologies and a rotation.

    OPENROUTER_API_KEY=... python tests/scale/live.py --budget 150

Free models are rate limited, twice: per account (all ``:free`` models share a
per-minute and per-day allowance) and per model (the upstream provider's free
pool is often saturated). So the runner:

* paces every request to ``--rpm``, whatever the model;
* rotates over a pool of free models, least-used first;
* on a 429 or a server error puts that model on a growing cooldown (honouring
  ``Retry-After``) and moves the call to another model;
* drops models that are gone (404) or refused, and stops on a bad key or when
  the daily allowance is spent;
* caps the whole run at ``--budget`` requests.

Safety: the key is read from the environment only, never stored or printed;
every model must be priced at zero or the run refuses to start; only synthetic
problems are sent, with no custom headers; the web server never sees the key.
Raw responses go to ``--raw``; the report goes to artifacts/scale/.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
import sys
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.runnables import Runnable  # noqa: E402

from tests.scale import harness as H  # noqa: E402
from tests.scale.synth import FaultPlan, Meter, Runaway, classify, text_of  # noqa: E402

BASE = "https://openrouter.ai/api/v1"
# Preference order only; the pool is filtered against the live list of free models.
PREFERRED = (
    "nvidia/nemotron-3.5-lightning:free",
    "inclusionai/ling-3.2-flash:free",
    "baidu/ernie-5.5-flash-preview:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "particle-ai/particle-1:free",
    "recursive/flint-v1.0:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "liquid/lfm-3-1.7b:free",
)
JSON_KINDS = {
    "sizer",
    "spanner",
    "decompose",
    "solve",
    "synthesize",
    "verify",
    "arbitrate",
    "evolve",
    "reframe",
    "q_complexity",
    "q_spanner",
    "q_agent",
    "q_reframe",
    "d_foundation",
    "k_master",
    "k_seed",
    "k_mirror",
    "k_mix",
    "k_followup",
    "k_perplexity",
    "r_toc",
    "r_program",
}
PROBLEM = (
    "Find the number of positive integers n with n <= 1000 such that n^2 + n + 1 is divisible by 7."
)
TRUTH = "286"  # n mod 7 in {2, 4}: 142 * 2 + 2
POLICY = """ACCESS
§1 Staff are granted read access to the reports share at all times.
§2 All access to the reports share is denied outside business hours.
§3 Service accounts must use unique credentials and must not share passwords.
CONTROLS
§4 Every access to the reports share is logged and retained for 90 days.
§5 Passwords for service accounts are rotated every 90 days.
"""


class BudgetSpent(Runaway):
    pass


@dataclass
class ModelState:
    ready_at: float = 0.0
    strikes: int = 0
    used: int = 0
    dropped: str = ""


class Rotation:
    """Shared pacing, model rotation, cooldowns, budget and per-request records."""

    def __init__(self, pool: list[str], budget: int, rpm: float, raw_path: Path) -> None:
        self.models = {m: ModelState() for m in pool}
        self.budget = budget
        self.requests = 0
        self.gap = 60.0 / rpm
        self._next = 0.0
        self.records: list[dict] = []
        self.raw_path = raw_path
        self.events: list[str] = []

    def say(self, msg: str) -> None:
        self.events.append(msg)
        print(f"   [rotation] {msg}", file=sys.__stderr__, flush=True)

    def acquire(self, preferred: str | None, avoid: set[str]) -> tuple[str, float]:
        """Pick the model for the next request and return (model, seconds to wait)."""
        if self.requests >= self.budget:
            raise BudgetSpent(f"live budget of {self.budget} requests spent")
        alive = [m for m, s in self.models.items() if not s.dropped]
        if not alive:
            raise BudgetSpent("every model in the rotation was dropped")
        choices = [m for m in alive if m not in avoid] or alive
        now = time.monotonic()
        ready = [m for m in choices if self.models[m].ready_at <= now]
        if preferred in ready:
            model = preferred
        elif ready:
            order = list(self.models)
            model = min(ready, key=lambda m: (self.models[m].used, order.index(m)))
        else:
            model = min(choices, key=lambda m: self.models[m].ready_at)
        at = max(now, self._next, self.models[model].ready_at)
        self._next = at + self.gap
        self.requests += 1
        self.models[model].used += 1
        return model, at - now

    def success(self, model: str) -> None:
        self.models[model].strikes = 0

    def failure(self, model: str, error: Exception) -> None:
        code = getattr(error, "status_code", None)
        text = str(error)
        state = self.models[model]
        if code == 401:
            raise BudgetSpent("OpenRouter rejected the key (401); stopping")
        if "free-models-per-day" in text:
            raise BudgetSpent("the daily free-model allowance is spent; stopping")
        if code in (402, 403, 404):
            state.dropped = f"{code}"
            self.say(f"{model} dropped ({code}): {text[:110]}")
            return
        state.strikes += 1
        retry_after = None
        response = getattr(error, "response", None)
        if response is not None:
            with contextlib.suppress(TypeError, ValueError):
                retry_after = float(response.headers.get("retry-after"))
        cooldown = retry_after or min(300.0, 40.0 * 2 ** (state.strikes - 1))
        state.ready_at = time.monotonic() + cooldown
        if "free-models-per-min" in text:  # the shared account limit: everyone waits
            self._next = max(self._next, time.monotonic() + 65)
            self.say("account per-minute limit hit: pausing all requests for 65 s")
        if state.strikes >= 4:
            state.dropped = "repeated failures"
            self.say(f"{model} dropped after {state.strikes} failures in a row")
        else:
            self.say(f"{model} cooling {cooldown:.0f}s ({type(error).__name__} {code or ''})")

    def record(self, row: dict, text: str) -> None:
        self.records.append(row)
        with open(self.raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**row, "response": text[:6000]}, ensure_ascii=False) + "\n")


class LiveLLM(Runnable):
    """The app's ChatOpenAI client behind the rotation. A preferred model is tried first."""

    def __init__(
        self,
        rotation: Rotation,
        meter: Meter,
        *,
        max_tokens: int,
        scenario: str = "",
        preferred: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.rotation = rotation
        self.preferred = preferred
        self.meter = meter
        self.plan = FaultPlan()
        self.max_tokens = max_tokens
        self.scenario = scenario
        self.temperature = temperature

    def bind(self, **kwargs):
        return LiveLLM(
            self.rotation,
            self.meter,
            max_tokens=self.max_tokens,
            scenario=self.scenario,
            preferred=self.preferred,
            temperature=float(kwargs.get("temperature", self.temperature)),
        )

    def invoke(self, input_data, config=None, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._call(input_data))
        finally:
            loop.close()

    async def ainvoke(self, input_data, config=None, **kwargs):
        return await self._call(input_data)

    async def _call(self, input_data):
        m = self.meter
        prompt = text_of(input_data)
        kind = classify(prompt)
        m.calls += 1
        m.by_kind[kind] += 1
        m.in_chars += len(prompt)
        m.max_in_chars[kind] = max(m.max_in_chars[kind], len(prompt))
        m.inflight += 1
        m.peak = max(m.peak, m.inflight)
        try:
            return await self._rotate(input_data, prompt, kind)
        finally:
            m.inflight -= 1

    async def _rotate(self, input_data, prompt: str, kind: str):
        from langchain_openai import ChatOpenAI

        failed: set[str] = set()
        last: Exception | None = None
        for attempt in range(6):
            model, wait = self.rotation.acquire(self.preferred, failed)
            await asyncio.sleep(wait)
            row = {
                "scenario": self.scenario,
                "model": model,
                "kind": kind,
                "attempt": attempt,
                "prompt_chars": len(prompt),
            }
            t0 = time.monotonic()
            try:
                client = ChatOpenAI(
                    model=model,
                    openai_api_key=os.environ["OPENROUTER_API_KEY"],
                    openai_api_base=BASE,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=120,
                    max_retries=0,
                )
                msg = await client.ainvoke(input_data)
            except Exception as e:
                last = e
                code = getattr(e, "status_code", None)
                row.update(
                    seconds=round(time.monotonic() - t0, 1),
                    ok=False,
                    error=f"{type(e).__name__} {code or ''}".strip(),
                    detail=str(e)[:240],
                )
                self.rotation.record(row, "")
                self.rotation.failure(model, e)
                failed.add(model)
                continue
            self.rotation.success(model)
            content = msg.content
            text = content if isinstance(content, str) else json.dumps(content)
            usage = getattr(msg, "usage_metadata", None) or {}
            meta = getattr(msg, "response_metadata", None) or {}
            parsed = None
            if kind in JSON_KINDS:
                from deepthink.utils import clean_and_parse_json

                parsed = isinstance(clean_and_parse_json(text), dict)
            row.update(
                seconds=round(time.monotonic() - t0, 1),
                ok=True,
                finish=meta.get("finish_reason"),
                chars=len(text),
                content_type=type(content).__name__,
                in_tokens=usage.get("input_tokens"),
                out_tokens=usage.get("output_tokens"),
                reasoning_tokens=(usage.get("output_token_details") or {}).get("reasoning"),
                json_ok=parsed,
            )
            self.rotation.record(row, text)
            self.meter.out_chars += len(text)
            return AIMessage(content=content)
        assert last is not None
        raise last


def get_json(url: str, key: str | None = None) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed https host
        return json.load(r)


def free_models() -> list[str]:
    out = []
    for m in get_json(f"{BASE}/models")["data"]:
        price = m.get("pricing") or {}
        if all(float(price.get(k) or 0) == 0 for k in ("prompt", "completion", "request")):
            if m["id"].endswith(":free"):
                out.append(m["id"])
    return out


# ── scenarios ────────────────────────────────────────────────────────────

THEORY_TINY = {
    "domain": "olympiad",
    "answer_type": "number",
    "qnn_mode": "manual",
    "manual_layers": 2,
    "manual_width": 2,
    "swarm_size": 2,
    "rounds": 1,
    "max_depth": 1,
    "branching": 2,
    "max_cells": 3,
    "quorum": 0.6,
    "concurrency": 2,
    "max_calls": 20,
    "polish": False,
    "seed": 5,
}
SCENARIOS = {
    "theory": "theory",
    "theory-mixed": "theory",
    "qnn": "qnn",
    "qdad": "qdad",
    "rsi": "rsi",
    "distill": "distill",
}


async def scenario(rotation: Rotation, name: str, max_tokens: int, timeout: float = 1500) -> dict:
    pipeline = SCENARIOS[name]
    meter = Meter()
    alive = [m for m, s in rotation.models.items() if not s.dropped]
    llm = LiveLLM(rotation, meter, max_tokens=max_tokens, scenario=name)
    rep = H.Report(pipeline, name)
    out = H.Outcome()
    start = len(rotation.records)
    t0 = time.time()

    async def run() -> None:
        if pipeline == "theory":
            from deepthink.theory import run_theory_pipeline

            p = dict(THEORY_TINY)
            members = None
            if name == "theory-mixed":
                p.update(swarm_size=3, rounds=2, max_calls=36)
                members = [
                    LiveLLM(rotation, meter, max_tokens=max_tokens, scenario=name, preferred=m)
                    for m in sorted(alive, key=lambda m: rotation.models[m].used)[:3]
                ]
            out.result = await run_theory_pipeline(
                llm,
                PROBLEM,
                p,
                log=out.logs.append,
                member_llms=members,
                member_models=[x.preferred for x in members] if members else None,
            )
            out.extra["params"] = out.result["params"]
        elif pipeline == "qnn":
            await H.drive_qnn(
                llm,
                {
                    "qnn_mode": "manual",
                    "manual_layers": 1,
                    "manual_width": 2,
                    "num_epochs": 1,
                    "problem": PROBLEM,
                },
                out,
            )
        elif pipeline == "qdad":
            await H.drive_qdad(llm, {"grid_size": 2, "denoising_steps": 1}, out)
        elif pipeline == "distill":
            await H.drive_distill(llm, {"max_epochs": 1}, out)
        elif pipeline == "rsi":
            from deepthink.nomosloop import run_rsi_pipeline

            # The driver must be local: it stays offline. Only mapper and editor are remote.
            out.result = await run_rsi_pipeline(
                None,
                POLICY,
                params={
                    "iterations": 1,
                    "drive": "bal",
                    "lora": {"program": {"enabled": True, "adapter_ids": [0]}},
                },
                artifact_path=os.path.join(tempfile.mkdtemp(), "r.md"),
                log=out.logs.append,
                mapper_llm=llm,
                editor_llm=llm,
            )

    stopped = ""
    try:
        await asyncio.wait_for(run(), timeout)
        H.CHECKS[pipeline](out, llm, rep)
    except Runaway as e:
        stopped = str(e)
        rep.problems.append(stopped)
    except asyncio.TimeoutError:
        rep.problems.append(f"did not finish within {timeout:.0f}s")
    except Exception as e:
        rep.problems.append(f"crashed: {type(e).__name__}: {str(e)[:200]}{H._where(e)}")
    rows = rotation.records[start:]
    summary: dict = {
        "scenario": name,
        "seconds": round(time.time() - t0, 1),
        "calls": meter.calls,
        "requests": len(rows),
        "failed_requests": sum(not r["ok"] for r in rows),
        "models_used": sorted({r["model"] for r in rows if r["ok"]}),
        "problems": rep.problems,
        "warnings": rep.warnings,
        "stopped": stopped,
    }
    r = out.result or {}
    if pipeline == "theory" and r:
        summary.update(
            answer=str(r.get("final_answer", ""))[:160],
            canonical=r.get("canonical_answer", ""),
            correct=r.get("canonical_answer") == TRUTH,
            converged=r.get("converged"),
            support=r.get("support"),
            rounds=r.get("rounds_run"),
            ledger=r["body_of_work"]["stats"],
            proposals=[m["proposal"].get("canonical") for m in r["members"]],
            verify=[(m["proposal"].get("verify") or {}).get("score") for m in r["members"]],
            cells=[len(m["tree"]) for m in r["members"]],
            cell_status=dict(Counter(c["status"] for m in r["members"] for c in m["tree"])),
        )
    elif pipeline == "qnn" and r:
        summary["report_chars"] = len(str(r.get("proposed_solution") or ""))
    elif pipeline == "qdad" and r:
        summary["report_chars"] = len(str(r.get("app_build_prompt") or ""))
        cells = [c for row in r.get("feature_matrix") or [] for c in row]
        summary["empty_or_error_cells"] = sum(
            1 for c in cells if not c.strip() or c.startswith("[noise error]")
        )
    elif pipeline == "distill" and r:
        g = out.extra["graph"]
        summary.update(
            qa_pairs=r["qa_pairs"],
            empty_answers=sum(1 for q in g.distilled_data if not str(q["answer"]).strip()),
            hard=sum(a.difficulty_history.count("Hard") for a in g._flat_agents()),
            children=sum(1 for a in g._flat_agents() if a.inherited_from),
            mixing_errors=sum("Mirror Descent error" in x for x in out.logs),
            perplexity=g.last_perplexity,
        )
    elif pipeline == "rsi" and r:
        summary.update(
            error=r.get("error"),
            findings=len(r.get("findings", [])),
            controls=len(r.get("controls", [])),
            toc_sources=dict(Counter(str(t.get("source")) for t in r.get("toc", []))),
            programs=[p.get("source") for p in (r.get("lora") or {}).get("programs", [])],
        )
    return summary


def per_model(records: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["model"]].append(r)
    out = []
    for model, rows in sorted(groups.items()):
        ok = [r for r in rows if r["ok"]]
        js = [r for r in ok if r.get("json_ok") is not None]
        secs = sorted(r["seconds"] for r in ok)
        out.append(
            {
                "model": model,
                "requests": len(rows),
                "ok": len(ok),
                "errors": dict(Counter(r["error"] for r in rows if not r["ok"])),
                "empty": sum(1 for r in ok if not r["chars"]),
                "truncated": sum(1 for r in ok if r.get("finish") == "length"),
                "json_ok": f"{sum(r['json_ok'] for r in js)}/{len(js)}",
                "median_s": secs[len(secs) // 2] if secs else None,
                "max_s": secs[-1] if secs else None,
                "in_tokens": sum(r.get("in_tokens") or 0 for r in ok),
                "out_tokens": sum(r.get("out_tokens") or 0 for r in ok),
                "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in ok),
            }
        )
    return out


async def main_async(args) -> dict:
    global PROBLEM, TRUTH
    if args.problem:
        PROBLEM, TRUTH = args.problem, args.truth
    if args.easiness is not None:
        THEORY_TINY["easiness"] = args.easiness
    key = os.environ["OPENROUTER_API_KEY"]
    free = free_models()
    if args.pool:
        pool = [m for m in args.pool.split(",") if m]
        paid = [m for m in pool if m not in free]
        if paid:
            raise SystemExit(f"refusing to run: not a free model: {paid}")
    else:
        pool = [m for m in PREFERRED if m in free] + [m for m in free if m not in PREFERRED]
    raw = Path(args.raw)
    raw.parent.mkdir(parents=True, exist_ok=True)
    rotation = Rotation(pool, args.budget, args.rpm, raw)
    report: dict = {"problem": PROBLEM, "truth": TRUTH, "pool": pool, "scenarios": []}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"live_{args.tag}.json"

    def save() -> None:
        report["models"] = per_model(rotation.records)
        report["requests"] = rotation.requests
        report["rotation_events"] = rotation.events
        report["dropped"] = {m: s.dropped for m, s in rotation.models.items() if s.dropped}
        path.write_text(
            json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
        )

    for name in [n for n in args.scenarios.split(",") if n]:
        if rotation.requests >= args.budget:
            break
        print(f"→ {name}  (requests so far {rotation.requests})", file=sys.__stderr__, flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = await scenario(rotation, name, args.max_tokens)
        report["scenarios"].append(summary)
        save()
        print(
            f"   {json.dumps(summary, default=str, ensure_ascii=False)}",
            file=sys.__stderr__,
            flush=True,
        )
        if "stopping" in summary["stopped"]:
            break
    info = get_json(f"{BASE}/key", key)["data"]
    report["key_usage_usd"] = info.get("usage")
    report["free_requests"] = info.get("free_model_daily_requests")
    save()
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="", help="free models to rotate over (default: all)")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--budget", type=int, default=150, help="maximum requests for the run")
    ap.add_argument("--rpm", type=float, default=14.0, help="global requests per minute")
    ap.add_argument("--max-tokens", type=int, default=3500)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--problem", default="", help="override the theory problem")
    ap.add_argument("--truth", default="", help="expected answer for --problem")
    ap.add_argument("--easiness", type=float, default=None, help="theory leaf threshold")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "scale"))
    ap.add_argument("--raw", default=str(ROOT / "artifacts" / "scale" / "live_raw.jsonl"))
    args = ap.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("set OPENROUTER_API_KEY in the environment")
    for var in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGCHAIN_TRACING"):
        os.environ[var] = "false"
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.disable(logging.CRITICAL)
    report = asyncio.run(main_async(args))
    keys = ("requests", "key_usage_usd", "free_requests", "dropped")
    print(json.dumps({k: report.get(k) for k in keys}))
    print(json.dumps(report.get("models"), indent=1))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
