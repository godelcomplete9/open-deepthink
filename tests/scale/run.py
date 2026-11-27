#!/usr/bin/env python3
"""Run the scalable mock test harness and write a report.

    python tests/scale/run.py                    # smoke: about 2 minutes
    python tests/scale/run.py --tier full        # wider sweeps, 5 fault seeds
    python tests/scale/run.py --only faults      # one section

No API key is read, outbound network connections are blocked, and every model
call is synthetic. Exit code 1 when any check FAILs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import socket
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SECTIONS = ("faults", "scaling", "server", "cli")


def block_network() -> None:
    """Refuse any connection that is not loopback, so a run can never spend tokens."""
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise OSError(f"network blocked in scale tests: {host}")
        return real_connect(self, address)

    socket.socket.connect = guarded
    os.environ["OPENROUTER_API_KEY"] = ""


def _signature(text: str) -> str:
    text = re.sub(r"'[^']{0,200}'|\{[^}]*\}|\[[^\]]*\]", "…", text)
    return re.sub(r"\d+(\.\d+)?", "N", text)[:100]


def group_faults(reports) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in reports:
        for level, items in (("FAIL", r.problems), ("WARN", r.warnings)):
            for item in items:
                groups[(r.pipeline, level, _signature(item))].append((r.scenario, item))
    out = []
    for (pipeline, level, _), hits in sorted(groups.items()):
        out.append(
            {
                "pipeline": pipeline,
                "level": level,
                "count": len(hits),
                "example": hits[0][1][:400],
                "scenarios": sorted({s.split(" seed")[0] for s, _ in hits})[:10],
            }
        )
    return out


def markdown(data: dict) -> str:
    lines = [f"# Scale test report ({data['tier']} tier)", ""]
    lines.append(
        f"Synthetic model only: no API calls, no tokens billed. Ran in {data['seconds']} s."
    )
    lines.append("")
    if "faults" in data:
        f = data["faults"]
        lines += [
            "## Fault matrix",
            "",
            f"{f['runs']} runs: {f['status'].get('OK', 0)} OK, {f['status'].get('WARN', 0)} WARN, "
            f"{f['status'].get('FAIL', 0)} FAIL. Virtual tokens exercised: {f['tokens']:,}.",
            "",
            "| pipeline | runs | OK | WARN | FAIL |",
            "|---|---|---|---|---|",
        ]
        for name, c in f["by_pipeline"].items():
            lines.append(
                f"| {name} | {sum(c.values())} | {c.get('OK', 0)} | {c.get('WARN', 0)} | "
                f"{c.get('FAIL', 0)} |"
            )
        lines += ["", "### Findings", ""]
        for g in f["findings"]:
            lines.append(f"- **{g['level']} · {g['pipeline']}** ×{g['count']}: {g['example']}")
            lines.append(f"  - scenarios: {', '.join(g['scenarios'])}")
        lines.append("")
    if "scaling" in data:
        lines += ["## Scaling", ""]
        for s in data["scaling"]["sweeps"]:
            lines.append(f"### {s['name']}")
            lines.append("")
            keys = [k for k in s["rows"][0] if k != "error"]
            lines.append("| " + " | ".join(keys) + " |")
            lines.append("|" + "---|" * len(keys))
            for r in s["rows"]:
                lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
            lines.append("")
            lines.append(
                "Growth exponents: "
                + ", ".join(f"{k} ^{v}" for k, v in s["slopes"].items() if v == v)
            )
            for flag in s["flags"]:
                lines.append(f"- ⚠ {flag}")
            lines.append("")
        lines += ["## Token calibration (about 800 output tokens per free-text field)", ""]
        cal = data["scaling"]["calibration"]
        keys = list(cal[0])
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "---|" * len(keys))
        for r in cal:
            lines.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
        lines.append("")
    for section in ("server", "cli"):
        if section in data:
            c = Counter(x["status"] for x in data[section])
            lines += [
                f"## {section.title()}",
                "",
                f"{len(data[section])} checks: {c.get('OK', 0)} OK, {c.get('WARN', 0)} WARN, "
                f"{c.get('FAIL', 0)} FAIL.",
                "",
            ]
            for x in data[section]:
                if x["status"] != "OK":
                    lines.append(
                        f"- **{x['status']}** · {x['area']} · `{x['name']}`: {x['detail']}"
                    )
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=("smoke", "full", "stress"), default="smoke")
    ap.add_argument("--only", choices=SECTIONS, action="append")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "scale"))
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    block_network()
    logging.disable(logging.CRITICAL)
    import builtins
    import contextlib
    import dataclasses
    import io

    sections = args.only or list(SECTIONS)
    data: dict = {"tier": args.tier}
    real_print = builtins.print

    def say(msg: str) -> None:
        real_print(msg, file=sys.__stderr__, flush=True)

    t0 = time.time()
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):  # pipelines print provider errors
        if "faults" in sections:
            from tests.scale.faults import fault_matrix

            say("fault matrix…")
            reps = asyncio.run(fault_matrix(args.tier, progress=say))
            by_pipeline: dict[str, Counter] = defaultdict(Counter)
            for r in reps:
                by_pipeline[r.pipeline][r.status] += 1
            data["faults"] = {
                "runs": len(reps),
                "status": dict(Counter(r.status for r in reps)),
                "tokens": sum(r.in_tokens + r.out_tokens for r in reps),
                "by_pipeline": {k: dict(v) for k, v in by_pipeline.items()},
                "findings": group_faults(reps),
            }
        if "scaling" in sections:
            from tests.scale.scaling import run_sweeps, token_calibration

            say("scaling sweeps…")

            async def both():
                return await run_sweeps(args.tier, progress=say), await token_calibration()

            sweeps, cal = asyncio.run(both())
            data["scaling"] = {
                "sweeps": [dataclasses.asdict(s) for s in sweeps],
                "calibration": cal,
            }
        if "server" in sections:
            from tests.scale.server_cli import server_checks

            say("server checks…")
            data["server"] = [dataclasses.asdict(c) for c in server_checks(progress=say)]
        if "cli" in sections:
            from tests.scale.server_cli import cli_checks

            say("cli checks…")
            data["cli"] = [dataclasses.asdict(c) for c in cli_checks(progress=say)]
    data["seconds"] = round(time.time() - t0, 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"report_{args.tier}.json").write_text(
        json.dumps(data, indent=1, default=str), encoding="utf-8"
    )
    md = markdown(data)
    (out / f"report_{args.tier}.md").write_text(md, encoding="utf-8")

    fails = data.get("faults", {}).get("status", {}).get("FAIL", 0)
    fails += sum(x["status"] == "FAIL" for k in ("server", "cli") for x in data.get(k, []))
    flags = sum(len(s["flags"]) for s in data.get("scaling", {}).get("sweeps", []))
    say(
        f"done in {data['seconds']} s: {fails} failing checks, {flags} scaling flags. "
        f"Report: {out / f'report_{args.tier}.md'}"
    )
    return 1 if fails else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # stray background threads must not keep the process alive
