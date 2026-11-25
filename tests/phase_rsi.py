"""RSI security loop tests (keyless).

Treatise invariants (Agent Mechanics), the NX-1 gauge objects, the policy
parser and posture gate, the leak/stock drives, the local-driver rule, the
LoRA clamp, and the live GUI event stream. Runs offline or with in-process
fake chat models.
"""

import asyncio
import json
import math
import random
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


ACCESS = ROOT / "examples" / "security" / "institution_access_policy.txt"
RUNBOOK = ROOT / "examples" / "security" / "incident_response_runbook.txt"


def run(text, params, llm=None, mapper=None, events=None):
    async def go():
        with tempfile.TemporaryDirectory() as workdir:
            res = await asyncio.wait_for(
                __import__("deepthink.nomosloop", fromlist=["run_rsi_pipeline"]).run_rsi_pipeline(
                    llm,
                    text,
                    law_title="t",
                    params=params,
                    artifact_path=str(Path(workdir) / "report.md"),
                    on_event=events.append if events is not None else None,
                    mapper_llm=mapper,
                ),
                timeout=120,
            )
            if not res.get("error"):
                assert Path(res["artifact_path"]).is_file(), res["artifact_path"]
                assert Path(res["nx1_path"]).is_file(), res["nx1_path"]
            return res

    return asyncio.run(go())


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
    assert p["drive"] == "bal" and p["require_local_driver"] is True
    nodes = set(build_graph().get_graph().nodes)
    for required in (
        "ingest",
        "scale_induction",
        "correspond",
        "tick",
        "heal",
        "stock",
        "lora",
        "artifact_emit",
    ):
        assert required in nodes, f"missing node {required} in {nodes}"
    topo = describe_topology()
    assert topo["entry"] == "ingest" and ["stock", "lora"] in topo["edges"]


chk("rsi package imports + leak/stock graph nodes", t1)


def t2():
    from deepthink import default_rsi_params, estimate_rsi_cost, run_rsi
    from deepthink.cost import estimate_rsi_cost as e2

    assert callable(run_rsi) and default_rsi_params()["mutators"] == 8
    leak, stock = estimate_rsi_cost(2, 8, drive="leak"), e2(2, 8, drive="stock")
    assert leak.llm_calls > 0 and stock.llm_calls > 0
    assert leak.breakdown["driver_heal"] > stock.breakdown["driver_heal"]
    assert estimate_rsi_cost(2, 8, cot=False).breakdown["driver_stock"] == 0


chk("rsi cost estimator + top-level API", t2)


def t3():
    from deepthink.nomosloop.mechanics import (
        IDX,
        LABEL_TO_STACK,
        SIDES,
        VALID_STACKS,
        Agent,
        Params,
        channel_matrix,
        encounter_mass,
        klein_stack,
        stack_label,
    )

    assert len(VALID_STACKS) == 16
    families = {frozenset(stack_label(klein_stack(g, s)) for g in SIDES) for s in VALID_STACKS}
    assert len(families) == 4 and all(len(f) == 4 for f in families)
    intj = LABEL_TO_STACK["INTJ"]
    assert [stack_label(klein_stack(g, intj)) for g in SIDES] == ["INTJ", "ESFP", "ENTP", "ISFJ"]
    # Part I §4, Table 1: m(INTJ, ENFP) = +27 with the schematic R and ego weights (4, 3, 2, 1)
    R = channel_matrix(Params(r_scale=1.0))

    def ego(label):
        w = [0.0] * 8
        for v, f in zip((4, 3, 2, 1), LABEL_TO_STACK[label]):
            w[IDX[f]] = v
        return Agent("B1", "B", LABEL_TO_STACK[label], w)

    assert encounter_mass(ego("INTJ"), ego("ENFP"), R) == 27
    assert encounter_mass(ego("INTJ"), ego("ESFJ"), R) == -34


chk("Agent Mechanics: 16 stacks, Klein families, Part I encounter masses", t3)


def t4():
    from deepthink.nomosloop.mechanics import Agent, Params, arrivals, channel_matrix, tick

    # Proposition 3' (conservation): Σ Δŵ = h[Σι + Σι_E/c − (1+ζ) Σ Leak] when nothing clamps.
    rnd = random.Random(7)
    p = Params(h=0.01, zeta=0.2, theta_p=0.3, energy_intake=0.4, r_scale=0.1, q=0.0)
    R = channel_matrix(p)
    agents = {}
    for i in range(6):
        w = [rnd.uniform(0.5, 2.0) for _ in range(8)]
        agents[f"A{i}"] = Agent(f"A{i}", "A", (), w, end=0.5, theta_p=p.theta_p)
        agents[f"A{i}"].energy = {s: [5.0] * 4 for s in ("e", "r", "a", "ra")}
    edges = {
        (f"A{i}", f"A{j}"): (math.pi if (i + j) % 5 == 0 else 0.0)
        for i in range(6)
        for j in range(6)
        if i != j
    }
    before = sum(a.reserve(p) for a in agents.values())
    tick(agents, edges, R, p)
    after = sum(a.reserve(p) for a in agents.values())
    leak = sum(a.leak_total() for a in agents.values())
    own = sum(a.own_intake for a in agents.values())
    expected = p.h * (own / p.c - (1 + p.zeta) * leak)
    assert abs((after - before) - expected) < 1e-6, (after - before, expected)
    # Proposition 2 (ii): with no hostile arrival, Leak(w) = w (D+ − w − δ) on 0 < w < D+ − δ.
    D = 3.0
    for wv in (0.4, 1.0, 1.4, 2.5, 3.5):
        weff = [wv] + [0.0] * 7
        res = arrivals(weff, [("x", [D] + [0.0] * 7)], Params(q=0.0, delta=0.05))
        lk = res["ov"][0] + res["q"][0] + res["op"][0]
        want = wv * (D - wv - 0.05) if wv < D - 0.05 else 0.0
        assert abs(lk - want) < 1e-9, (wv, lk, want)


chk("tick law conserves the total reserve; Leak bound (Prop. 2)", t4)


def t5():
    from deepthink.nomosloop import correspondence as tc

    # gauge invariance of curvature on random squares
    rnd = random.Random(3)
    links = {}
    squares = []
    for k in range(20):
        a1, a2, i1, i2 = f"a{k}", f"b{k}", f"i{k}", f"j{k}"
        for e in ((a1, a2), (a2, i2), (i1, i2), (a1, i1)):
            links[e] = rnd.uniform(0, math.pi)
        squares.append((a1, a2, i1, i2))
    chi = {v: rnd.uniform(-3, 3) for e in links for v in e}
    moved = tc.gauge_transform(links, chi)
    for sq in squares:
        assert abs(tc.square_curvature(links, *sq) - tc.square_curvature(moved, *sq)) < 1e-9
    # a contradiction between cooperating agents is maximally frustrated
    assert tc.frustration(tc.curvature(math.pi, 0.0, 0.0, 0.0)) > 0.999
    assert tc.frustration(tc.curvature(0.0, 0.1, 0.1, 0.0)) < 1e-9
    # coderivative: aligned rows carry the edit, anti-aligned rows its inverse
    rows = [
        tc.TocRow(1, "pol/L2/§1", "A1", [], [1] + [0] * 7, {}, "A", "", "", 0.1, 0.8, "m"),
        tc.TocRow(2, "pol/L2/§2", "A1", [], [1] + [0] * 7, {}, "A", "", "", 3.0, 0.8, "m"),
    ]
    edits = tc.coderivative("r", "A1", rows)
    assert [e.inverse for e in edits] == [False, True]
    assert edits[0].symbol == tc.EDITS["r"]


chk("gauge: curvature invariant under A -> A + d0 chi; coderivative", t5)


def t6():
    from deepthink.nomosloop import security

    lat = security.parse_policy(ACCESS.read_text(encoding="utf-8"), "x")
    ids = {lv["level_id"]: [r["id"] for r in lv["rules"]] for lv in lat["levels"]}
    assert (
        ids["L3"] == ["§1"] and "§13" in ids["L0"] and "§7" in ids["L2"] and "§10" in ids["L1"]
    ), ids
    pairs = {(a.split("/")[-1], b.split("/")[-1]) for a, b, _ in security.contradictions(lat)}
    assert pairs == {("§2", "§7"), ("§5", "§8")}, pairs
    rb = security.parse_policy(RUNBOOK.read_text(encoding="utf-8"), "y")
    pos = security.posture(rb)
    assert pos["contradictions"] == 0 and "Ne" in pos["missing"]
    assert any(lv["level_id"] == "L1" for lv in security.ensure_controls_level(rb)["levels"])


chk("security parser: levels, the two seeded contradictions, gaps", t6)


def t7():
    text = ACCESS.read_text(encoding="utf-8")
    leak = run(text, {"offline": True, "drive": "leak", "iterations": 3})
    stock = run(text, {"offline": True, "drive": "stock", "iterations": 3})
    for res in (leak, stock):
        assert res["mode"] == "rsi" and res["domain"] == "cybersecurity", res.get("error")
        assert res["posture_final"] + 1e-9 >= res["posture_baseline"]
        assert all(r.get("offline") for r in res["routing_log"])
        assert res["ensembles"] and res["agents"] and res["toc"]
        for e in res["ensembles"]:
            assert (
                e["text"].startswith(e["title"] + "[n=") and f"n={len(e['members'])};" in e["text"]
            )
        for a in res["agents"]:
            assert not (a["name"].startswith("Attr-") and a["L"] > 0), a["name"]
    assert len(leak["findings"]) > len(stock["findings"])
    assert len(stock["controls"]) > len(leak["controls"])
    seeded = {("pol/L2/§2", "pol/L2/§7"), ("pol/L2/§5", "pol/L2/§8")}
    found = {(c["a1"], c["a2"]) for f in leak["findings"] for c in f["clashes"]}
    assert found & seeded, found
    for c in stock["controls"]:
        assert c["posture_after"] > c["posture_before"]


chk("offline loop: leak finds the seeded clashes, stock evolves the policy", t7)


def t8():
    calls = []

    class FakeLocal:
        def invoke(self, messages):
            calls.append(messages[-1][1][:40])
            return SimpleNamespace(
                content="Rotate the credentials and restrict the share to business hours."
            )

    res = run(
        ACCESS.read_text(encoding="utf-8"), {"drive": "leak", "iterations": 1}, llm=FakeLocal()
    )
    assert not res.get("error"), res.get("error")
    assert res["routing_log"] and all(
        str(r["model"]).startswith("bridge-") for r in res["routing_log"]
    )
    assert any(r["model"] == "bridge-driver-as-mapper" for r in res["routing_log"])
    assert calls, "driver bridge never called"

    class Remote(FakeLocal):
        base_url = "https://api.example.com/v1"

    from deepthink.nomosloop import run_rsi_pipeline

    try:
        asyncio.run(
            run_rsi_pipeline(
                Remote(), "§1 Staff must use MFA.\n§2 Logs are kept.", params={"iterations": 1}
            )
        )
    except ValueError as e:
        assert "must be local" in str(e)
    else:
        raise AssertionError("a remote driver must be refused")
    from deepthink.nomosloop.routing import is_local_url

    assert is_local_url("http://localhost:8080/v1") and is_local_url("http://192.168.1.20:8080")
    assert not is_local_url("https://openrouter.ai/api/v1")
    from deepthink.nomosloop.routing import is_local_llm

    # a networked client whose endpoint cannot be read fails closed
    assert not is_local_llm(SimpleNamespace(client=object(), invoke=None))
    assert is_local_llm(FakeLocal())


chk("driver bridge is local-only; mapper falls back to the driver", t8)


def t9():
    from deepthink.nomosloop import lora

    sent = []

    def fake_http(url, method="GET", body=None, timeout=5.0):
        sent.append((url, method, body))
        return True, {"ok": True}

    orig = lora._http
    lora._http = fake_http
    try:
        cfg = lora.LoraConfig(
            enabled=True, base_url="http://localhost:8080/v1", adapter_id=2, clamp_step=0.25
        )
        st = lora.LoraState()
        r1 = lora.step(
            cfg, st, {"rounds_per_solve": 3.0, "leak": 0.3, "capacity": 0.2, "solves": 2}, "run", 1
        )
        assert r1["decision"] == "raise" and st.scale == 0.25 and st.good_scale == 0.0
        r2 = lora.step(
            cfg, st, {"rounds_per_solve": 2.0, "leak": 0.2, "capacity": 0.3, "solves": 4}, "run", 2
        )
        assert r2["decision"] == "raise" and st.scale == 0.5 and st.good_scale == 0.25
        r3 = lora.step(
            cfg, st, {"rounds_per_solve": 4.0, "leak": 0.4, "capacity": 0.3, "solves": 5}, "run", 3
        )
        assert r3["decision"] == "rollback" and st.scale == 0.25
    finally:
        lora._http = orig
    assert sent[-1] == ("http://localhost:8080/lora-adapters", "POST", [{"id": 2, "scale": 0.25}])
    assert lora.train(lora.LoraConfig(), st, "run")["ran"] is False


chk("LoRA clamp: raise on improvement, roll back otherwise", t9)


def t10():
    from deepthink.nomosloop import graph

    events = []
    res = run(
        RUNBOOK.read_text(encoding="utf-8"),
        {"offline": True, "drive": "bal", "iterations": 2},
        events=events,
    )
    assert not res.get("error"), res.get("error")
    assert events[0]["phase"] == "enter" and events[0]["node"] == "ingest"
    assert events[-1]["phase"] == "exit" and events[-1]["node"] == "artifact_emit"
    entered = [e["node"] for e in events if e["phase"] == "enter"]
    assert entered.count("correspond") == 2 and entered.count("lora") == 2, entered
    json.dumps(events)
    last = events[-1]["state"]
    assert last["agents"] and last["toc"] and last["posture"] and "lora" in last
    assert graph._OBSERVER is None


chk("live on_event stream brackets every node with snapshots", t10)


def t11():
    from deepthink.nomosloop import wsp
    from deepthink.nomosloop.lora import example

    exs = [
        example("heal", "p1", "long answer one"),
        example("heal", "p1", "short"),
        example("stock", "p2", "a2"),
    ]
    clean, notes = wsp.validate(
        {
            "data": {
                "drop": [0, 99],
                "rewrite": [{"i": 2, "answer": "a2 shorter"}],
                "add": [{"prompt": "p3", "answer": "a3"}],
            },
            "train": {
                "rank": 999,
                "alpha": 10**6,
                "learning_rate": 5,
                "target_modules": ["q_proj", "bogus"],
            },
            "mix": [{"id": 0, "scale": 7}, {"id": 9, "scale": 0.5}, {"id": 1, "scale": -0.4}],
        },
        n_examples=3,
        adapter_ids=[1],
        main_id=0,
        clamp_max=1.0,
        allow_negative=False,
    )
    t = clean["train"]
    assert t["rank"] == 64 and t["alpha"] == 256 and t["learning_rate"] == 1e-3, t
    assert t["target_modules"] == ["q_proj"], t
    assert clean["mix"] == [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}], clean["mix"]
    assert clean["data"]["drop"] == [0]
    assert any("not a loaded adapter" in n for n in notes) and any("rank 999" in n for n in notes)
    out = wsp.apply_data(exs, clean["data"])
    assert (
        len(out) == 3
        and out[-1]["kind"] == "programmed"
        and out[1]["messages"][-1]["content"] == "a2 shorter"
    )
    assert len(exs) == 3 and exs[2]["messages"][-1]["content"] == "a2"  # input untouched
    off = wsp.offline_program(exs, 0, 0.25, 0.25, 1.0)
    assert off["data"]["drop"] == [0] and off["mix"] == [{"id": 0, "scale": 0.5}], off
    neg, _ = wsp.validate({"mix": [{"id": 1, "scale": -0.4}]}, 0, [1], 0, 1.0, True)
    assert neg["mix"] == [{"id": 1, "scale": -0.4}]
    # half the dataset at most can be dropped
    capped, notes = wsp.validate({"data": {"drop": [0, 1, 2]}}, 3, [], 0, 1.0, False)
    assert capped["data"]["drop"] == [0] and notes


chk("weight program: clamped spec, loaded adapters only, data edits", t11)


def t12():
    from deepthink.nomosloop import lora

    sent = []

    def fake_http(url, method="GET", body=None, timeout=5.0):
        sent.append((url, method, body))
        return True, {"ok": True}

    def prog(mix):
        return {
            "n": 1,
            "source": "bridge-editor",
            "examples_in": 4,
            "examples_out": 3,
            "corrections": [],
            "spec_path": "spec.json",
            "dataset_path": "data.jsonl",
            "program": {"mix": mix},
        }

    first = [{"id": 0, "scale": 0.4}, {"id": 1, "scale": 1.0}]
    second = [{"id": 0, "scale": 0.6}, {"id": 1, "scale": 0.5}]
    orig = lora._http
    lora._http = fake_http
    try:
        cfg = lora.LoraConfig(enabled=True, base_url="http://localhost:8080/v1", adapter_id=0)
        st = lora.LoraState()
        r1 = lora.step(
            cfg,
            st,
            {"rounds_per_solve": 3.0, "leak": 0.3, "capacity": 0.2, "solves": 2},
            "run",
            1,
            program_fn=lambda: prog(first),
        )
        assert r1["decision"] == "program" and st.mix == first and st.scale == 0.4, r1
        assert sent[-1] == ("http://localhost:8080/lora-adapters", "POST", first)
        assert r1["train"]["ran"] is False  # no trainer configured
        r2 = lora.step(
            cfg,
            st,
            {"rounds_per_solve": 2.0, "leak": 0.2, "capacity": 0.3, "solves": 4},
            "run",
            2,
            program_fn=lambda: prog(second),
        )
        assert r2["decision"] == "program" and st.good_mix == first and st.mix == second
        r3 = lora.step(
            cfg,
            st,
            {"rounds_per_solve": 4.0, "leak": 0.4, "capacity": 0.3, "solves": 5},
            "run",
            3,
            program_fn=lambda: prog([{"id": 1, "scale": 0.1}]),
        )
        assert r3["decision"] == "rollback" and st.mix == first, r3
        assert sent[-1][2] == first
    finally:
        lora._http = orig


chk("weight program: mix clamp, keep on improvement, roll back to last good mix", t12)


def t13():
    import os

    import openai
    from deepthink.nomosloop import routing, run_rsi_pipeline
    from deepthink.nomosloop.routing import RouterChat

    driver_calls = []
    routing.clear_llm_bridge()
    routing.set_llm_bridge(lambda p, k, s, m, t: driver_calls.append(k) or "driver text")
    old = os.environ.pop("RSI_EDITOR_MODEL", None)
    try:
        r = routing.call_llm("x", task_kind="weight_program")
        assert r["tier"] == "editor" and r["offline"] and r["model"] == "offline-editor", r
        assert not driver_calls, "the editor must never fall back to the driver"
    finally:
        routing.clear_llm_bridge()
        if old is not None:
            os.environ["RSI_EDITOR_MODEL"] = old

    seen = {}

    class FakeClient:
        def __init__(self, **kw):
            seen["client"] = kw
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kw):
            seen["create"] = kw
            msg = SimpleNamespace(content="{}")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    orig = openai.OpenAI
    openai.OpenAI = FakeClient
    try:
        out = RouterChat("vendor/editing-model", "test-key").invoke([("user", "hi")])
    finally:
        openai.OpenAI = orig
    assert out.content == "{}" and seen["create"]["model"] == "vendor/editing-model"
    assert "temperature" not in seen["create"] and "top_p" not in seen["create"]

    class FakeLocal:
        def invoke(self, messages):
            return SimpleNamespace(content="Rotate the credentials and log every access.")

    class FakeEditor:
        model_name = "fake-editor"

        def invoke(self, messages):
            assert "Return a JSON object" in messages[-1][1]
            body = {
                "train": {"rank": 999},
                "mix": [{"id": 0, "scale": 0.5}, {"id": 1, "scale": 7}],
                "rationale": "t",
            }
            return SimpleNamespace(content="```json\n" + json.dumps(body) + "\n```")

    with tempfile.TemporaryDirectory() as workdir:
        res = asyncio.run(
            run_rsi_pipeline(
                FakeLocal(),
                RUNBOOK.read_text(encoding="utf-8"),
                law_title="t",
                params={
                    "drive": "stock",
                    "iterations": 2,
                    "lora": {"out_dir": workdir, "program": {"enabled": True, "adapter_ids": [1]}},
                },
                artifact_path=str(Path(workdir) / "report.md"),
                editor_llm=FakeEditor(),
            )
        )
        assert not res.get("error"), res.get("error")
        progs = res["lora"]["programs"]
        assert progs and all(p["source"] == "bridge-editor" for p in progs), progs
        assert progs[0]["mix"] == [{"id": 0, "scale": 0.5}, {"id": 1, "scale": 1.0}], progs[0][
            "mix"
        ]
        assert any("rank 999" in c for p in progs for c in p["corrections"])
        assert Path(progs[0]["dataset_path"]).is_file() and Path(progs[0]["spec_path"]).is_file()
    assert any(
        r["agent"] == "weight_programmer" and r["model"] == "bridge-editor"
        for r in res["routing_log"]
    )
    assert res["editor"] == {"model": "fake-editor", "programming": True}, res["editor"]


chk("weight program: editor tier routing, no sampling params, pipeline wiring", t13)


ok = sum(1 for _, s, _ in results if s == "OK")
total = len(results)
for name, status, err in results:
    print(f"  [{status}] {name}" + (f" :: {err}" if err else ""))
print(f"\nPHASE RSI: {ok}/{total} OK")
sys.exit(0 if ok == total else 1)
