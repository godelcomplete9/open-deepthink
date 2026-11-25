"""RSI (recursive self-improvement over formal artifacts) unit tests.

API-only section: package structure, LangGraph nodes, offline pipeline, the
injected-LLM bridge, and the cost estimator. Runs keyless (offline heuristics
or a tiny fake chat model).
"""

import asyncio
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

results = []


def chk(name, fn):
    try:
        fn()
        results.append((name, "OK", None))
    except AssertionError as e:
        results.append((name, "FAIL", f"AssertionError: {e}"))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


EXAMPLE = ROOT / "examples" / "binary_decimal_fragment.txt"


def t1():
    from deepthink.nomosloop import (
        LATTICE_VERSION,
        build_graph,
        default_rsi_params,
        describe_topology,
        run_rsi_pipeline,
    )

    assert callable(run_rsi_pipeline)
    assert LATTICE_VERSION.startswith("lattice/")
    p = default_rsi_params()
    assert p["iterations"] == 1 and p["offline"] is False

    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    for required in (
        "ingest",
        "scale_induction",
        "mutation_proposal",
        "mesh_spawn",
        "fitness",
        "artifact_emit",
        "rectify",
    ):
        assert required in nodes, f"missing node {required} in {nodes}"

    topo = describe_topology()
    assert topo["entry"] == "ingest"
    assert any(e[0] == "rectify" or e[1] == "rectify" for e in topo["edges"])


chk("rsi package imports + graph nodes", t1)


def t2():
    from deepthink import default_rsi_params, estimate_rsi_cost, run_rsi

    assert callable(run_rsi)
    base = estimate_rsi_cost(iterations=1, mutators=0)
    evolved = estimate_rsi_cost(iterations=2, mutators=6)
    # induction (4) + iterations * (2N + 1)
    assert base.llm_calls == 4 + 1 * 1
    assert evolved.llm_calls == 4 + 2 * 13
    assert evolved.llm_calls > base.llm_calls
    assert default_rsi_params()["iterations"] == 1


chk("rsi cost estimator + top-level API", t2)


def t3():
    from deepthink.nomosloop import run_rsi_pipeline

    raw = EXAMPLE.read_text(encoding="utf-8")

    async def run():
        with tempfile.TemporaryDirectory() as workdir:
            artifact = str(Path(workdir) / "artifact.py")
            res = await run_rsi_pipeline(
                None,
                raw,
                law_title="radix-test",
                params={"iterations": 1, "mutators": 2, "offline": True},
                artifact_path=artifact,
            )
            # artifact is written inside the temp dir; check before it is removed
            assert Path(res["artifact_path"]).is_file(), res["artifact_path"]
            return res

    res = asyncio.run(run())
    assert res["mode"] == "rsi"
    assert res["levels"], "offline induction must produce levels"
    assert res["num_rules"] >= 1
    assert res["final_capacity"] + 1e-6 >= res["baseline_capacity"]
    assert res["artifact_path"].endswith(".py")
    assert "BASE" in res["proposed_solution"] or "lattice" in res["proposed_solution"].lower()
    # offline routing is recorded
    assert all(r.get("offline") for r in res["routing_log"])


chk("offline RSI pipeline produces lattice + artifact", t3)


def t4():
    from deepthink.nomosloop import run_rsi_pipeline

    class FakeLLM:
        def invoke(self, messages):
            # Invalid lattice JSON -> engine falls back to heuristic lattice.
            return SimpleNamespace(content="{}")

    raw = EXAMPLE.read_text(encoding="utf-8")

    async def run():
        with tempfile.TemporaryDirectory() as workdir:
            artifact = str(Path(workdir) / "artifact.py")
            return await run_rsi_pipeline(
                FakeLLM(),
                raw,
                law_title="bridge-test",
                params={"iterations": 1, "mutators": 1},
                artifact_path=artifact,
            )

    res = asyncio.run(run())
    assert res["mode"] == "rsi"
    assert res["levels"]
    # bridge was used for every agent call (model tagged bridge-<tier>)
    assert res["routing_log"], "routing log must be populated"
    assert all(str(r.get("model", "")).startswith("bridge-") for r in res["routing_log"])


chk("injected-LLM bridge drives the RSI loop", t4)


def t5():
    from deepthink.nomosloop import rewrite
    from deepthink.nomosloop.state import ScaleLattice

    lat = ScaleLattice(
        version="v0",
        law_id="law-x",
        levels=[
            {
                "level_id": "L1",
                "name": "top",
                "alphabet": ["a", "b"],
                "base": "radix",
                "base_spec": {"radix": 2, "alphabet": ["0", "1"]},
                "modulus": "mod",
                "modulus_spec": {"period": 4},
                "rules": [
                    {"id": "R-top", "name": "TOP", "level": "L1", "pattern": "p", "rewrite": "r"}
                ],
            },
            {
                "level_id": "L0",
                "name": "base",
                "alphabet": ["0", "1"],
                "base": "radix",
                "base_spec": {"radix": 2, "alphabet": ["0", "1"]},
                "modulus": "mod",
                "modulus_spec": {"period": 256},
                "rules": [
                    {"id": "R-leaf", "name": "LEAF", "level": "L0", "pattern": "p", "rewrite": "r"}
                ],
            },
        ],
        order="top->base",
        expansion_map={"R-top": ["R-leaf"]},
        provenance={},
        created_at=0.0,
    )
    cap = rewrite.measured_capacity(lat)
    assert cap["instances"] == 256, cap
    assert cap["capacity"] > 0


chk("measured capacity executes the rewrite structure", t5)


ok = sum(1 for _, s, _ in results if s == "OK")
total = len(results)
for name, status, err in results:
    print(f"  [{status}] {name}" + (f" :: {err}" if err else ""))
print(f"\nPHASE RSI: {ok}/{total} OK")
sys.exit(0 if ok == total else 1)
