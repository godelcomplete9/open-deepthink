"""RSI benchmark: scoreboard vs a no-mutation baseline.

Runs the same formal text through the loop twice:
- baseline: mutators=0, one iteration  (induction only, lattice v0);
- evolved:  mutators=6, two iterations (proposal -> measured fitness -> gate).

Metrics per case: measured capacity (executed enumeration), distinct ground
instances, holdout-motif coverage, accepted mutations, rollbacks, iterations.
Exits non-zero if evolved capacity falls below the baseline (regression).

Run:  python -m deepthink.nomosloop.benchmark [input.txt]
"""

from __future__ import annotations

import os
import sys
import tempfile

from .graph import describe_topology, run_pipeline, set_topology


def run_case(input_path: str, mutators: int, iterations: int, workdir: str) -> dict:
    from .ingest import load_formal_text

    snapshot = describe_topology()
    try:
        raw = load_formal_text(input_path)
        artifact = os.path.join(workdir, f"bench_m{mutators}.py")
        final = run_pipeline(
            raw,
            law_title=f"bench-m{mutators}",
            max_iterations=iterations,
            artifact_path=artifact,
            mutators=mutators,
        )
    finally:
        set_topology(snapshot)
    caps = final.get("capacity_history", [])
    last = caps[-1] if caps else {}
    return {
        "case": f"m={mutators},i={iterations}",
        "capacity": float(last.get("capacity", 0.0)),
        "instances": int(last.get("instances", 0)),
        "coverage": float(last.get("coverage", 0.0)),
        "accepted": len(final.get("accepted_mutations", []) or []),
        "rollbacks": len(final.get("rollbacks", []) or []),
        "topology_patches": sum(
            1 for p in (final.get("topology_history") or []) if p.get("applied")
        ),
        "iterations": int(final.get("iteration", 0)),
        "lattice": final.get("scale_lattice", {}).get("version"),
        "artifact": artifact,
    }


def format_table(rows: list) -> str:
    head = f"{'case':<22}{'capacity':>10}{'instances':>11}{'coverage':>10}{'accepted':>10}{'rolled':>8}{'iter':>6}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['case']:<22}{r['capacity']:>10.3f}{r['instances']:>11}"
            f"{r['coverage']:>10.2f}{r['accepted']:>10}{r['rollbacks']:>8}{r['iterations']:>6}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    input_path = argv[0] if argv else os.path.join("examples", "binary_decimal_fragment.txt")
    if not os.path.isfile(input_path):
        print(f"[bench] input not found: {input_path}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="nomosloop_bench_") as workdir:
        baseline = run_case(input_path, mutators=0, iterations=1, workdir=workdir)
        evolved = run_case(input_path, mutators=6, iterations=2, workdir=workdir)
    print(f"[bench] input: {input_path}")
    print(format_table([baseline, evolved]))
    delta = evolved["capacity"] - baseline["capacity"]
    verdict = "PASS" if delta >= -1e-9 else "FAIL"
    print(
        f"[bench] {verdict}: measured capacity delta {delta:+.3f} "
        f"({baseline['capacity']:.3f} -> {evolved['capacity']:.3f}), "
        f"instances {baseline['instances']} -> {evolved['instances']}, "
        f"rollbacks={evolved['rollbacks']}, topology patches kept={evolved['topology_patches']}"
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
