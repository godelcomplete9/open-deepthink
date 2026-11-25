"""RSI benchmark: the security loop under each drive, offline.

For every policy given (default: ``examples/security/*.txt``) it runs the
loop under the leak, balanced and stock drives and reports:

- posture before and after (function diversity over contradictions);
- contradictory clause pairs found in the policy;
- findings (leak-driven heals) and controls kept (stock-driven edits);
- ensemble leak ``L_G`` and coverage (covered clause × function cells);
- chain-of-thought rounds per solve.

Exits non-zero if any run ends with a lower posture than it started with:
the stock gate must make that impossible.

Run:  python -m deepthink.nomosloop.benchmark [policy.txt ...]
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile

from .graph import run_pipeline
from .world import World

DRIVES = ("leak", "bal", "stock")


def run_case(path: str, drive: str, iterations: int, workdir: str) -> dict:
    from .ingest import load_formal_text

    raw = load_formal_text(path)
    name = os.path.splitext(os.path.basename(path))[0]
    old = os.environ.get("NOMOSLOOP_OFFLINE")
    os.environ["NOMOSLOOP_OFFLINE"] = "1"
    try:
        final = run_pipeline(
            raw,
            law_title=name,
            max_iterations=iterations,
            artifact_path=os.path.join(workdir, f"{name}_{drive}.md"),
            rsi_cfg={"drive": drive, "cot": False},
        )
    finally:
        if old is None:
            os.environ.pop("NOMOSLOOP_OFFLINE", None)
        else:
            os.environ["NOMOSLOOP_OFFLINE"] = old
    post = final.get("posture_history", []) or [{}]
    world = final.get("world")
    w = world if isinstance(world, World) else None
    return {
        "case": f"{name[:22]}/{drive}",
        "posture_0": float(post[0].get("score", 0.0)),
        "posture": float(post[-1].get("score", 0.0)),
        "contra": int(post[0].get("contradictions", 0)),
        "findings": len(w.inspections) if w else 0,
        "controls": len(final.get("controls", []) or []),
        "L_G": round(w.ensemble_leak(), 3) if w else 0.0,
        "space": w.coverage_history[-1]["space"] if w and w.coverage_history else 0,
        "rps": round(w.rounds_per_solve(), 2) if w else 0.0,
    }


def format_table(rows: list) -> str:
    head = f"{'case':<32}{'posture':>16}{'contra':>8}{'finds':>7}{'ctrls':>7}{'L_G':>7}{'space':>7}{'rps':>7}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['case']:<32}{r['posture_0']:>7.3f} -> {r['posture']:<6.3f}{r['contra']:>8}{r['findings']:>7}"
            f"{r['controls']:>7}{r['L_G']:>7.3f}{r['space']:>7}{r['rps']:>7.2f}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    paths = argv or sorted(glob.glob(os.path.join("examples", "security", "*.txt")))
    if not paths:
        print("no policies given and none found in examples/security", file=sys.stderr)
        return 2
    rows = []
    with tempfile.TemporaryDirectory() as workdir:
        for path in paths:
            for drive in DRIVES:
                rows.append(run_case(path, drive, iterations=3, workdir=workdir))
    print(format_table(rows))
    bad = [r for r in rows if r["posture"] + 1e-9 < r["posture_0"]]
    if bad:
        print(f"REGRESSION: posture fell in {[r['case'] for r in bad]}", file=sys.stderr)
        return 1
    print("ok: no posture regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
