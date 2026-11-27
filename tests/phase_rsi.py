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

# Check names carry treatise glyphs (η, →, σ, †, ω); never crash on a non-UTF-8 console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

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
    assert topo["entry"] == "ingest"
    assert ["stock", "evolve"] in topo["edges"] and ["evolve", "lora"] in topo["edges"]
    assert "evolve" in nodes


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

SURFACE = ROOT / "examples" / "security_surface"


def t14():
    import os

    from deepthink.nomosloop import repo

    res = repo.scan_repository(str(SURFACE / "repo"))
    rules = {f.rule: f for f in res.facts}
    for r in (
        "hardcoded-secret",
        "tls-verify-off",
        "debug-on",
        "open-ingress",
        "public-storage",
        "remote-desktop",
        "sql-injection",
        "container-root",
    ):
        assert r in rules, (r, sorted(rules))
    secret = rules["hardcoded-secret"]
    assert "demo-not-a-real-secret" not in secret.evidence and "redacted" in secret.evidence
    assert all("demo-not-a-real-secret" not in str(f.as_dict()) for f in res.facts)
    assert [f.ref for f in res.facts] == [f"code/C{i}" for i in range(1, len(res.facts) + 1)]
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "node_modules", "x"))
        os.makedirs(os.path.join(d, ".git"))
        Path(d, "node_modules", "x", "a.js").write_text('var password = "abcdefgh";')
        Path(d, ".git", "config").write_text('password = "abcdefgh"')
        Path(d, "blob.bin").write_bytes(b'\0\0password = "abcdefgh"')
        Path(d, "big.py").write_text("x = 1\n" * 100_000)
        Path(d, "ok.py").write_text(
            '# password = "commented-out"\nPASSWORD = os.environ["PASSWORD"]\nkey = "${KEY}"\n'
        )
        empty = repo.scan_repository(d)
        assert empty.facts == [], [f.as_dict() for f in empty.facts]
        assert empty.files_scanned == 1, empty.meta()
    try:
        repo.scan_repository(str(SURFACE / "missing"))
        raise AssertionError("missing path accepted")
    except FileNotFoundError:
        pass


chk("repository analysis: planted defects found, secrets redacted, junk skipped", t14)


def t15():
    from deepthink.nomosloop import security, social, surface

    profile = json.loads((SURFACE / "social_profile.json").read_text(encoding="utf-8"))
    facts, meta = social.analyze(profile, ACCESS.read_text(encoding="utf-8"))
    rules = {f.rule for f in facts}
    assert {"pretext-reset", "payment-redirect", "urgency-bypass", "no-reporting"} <= rules, rules
    assert "people (individuals)" in meta["dropped"], meta
    blob = json.dumps([f.as_dict() for f in facts])
    assert "ENFP" not in blob and "openness" not in blob and "biography" not in blob
    # no authorization block -> refused
    for bad in ({"roles": profile["roles"]}, {"authorization": {"by": ""}, "roles": []}):
        try:
            social.analyze(bad)
            raise AssertionError("profile without authorization accepted")
        except social.SocialProfileError:
            pass
    # personal fields inside a role are dropped
    clean, dropped = social.sanitize(
        {
            "authorization": {"by": "x"},
            "roles": [{"role": "help desk", "mbti": "INTJ", "name": "A"}],
        }
    )
    assert clean["roles"] == [{"role": "help desk"}] and "roles[0].mbti" in dropped
    # a strong callback procedure removes the reset finding
    strong = json.loads(json.dumps(profile))
    strong["roles"][0]["verification"] = ["callback to the number on file"]
    assert "pretext-reset" not in {f.rule for f in social.analyze(strong)[0]}
    # the rule says callback, the routine does not: a contradiction with the clause
    policy = (
        "ACCESS\n§1 Staff may use the VPN.\n"
        "§2 The help desk must verify identity by a callback before any password reset.\n"
    )
    lat = security.parse_policy(policy, "z")
    reset = [f for f in facts if f.rule == "pretext-reset"]
    links = surface.link([f.as_dict() for f in reset], lat)
    assert links[reset[0].ref]["status"] == "contradicts", links
    assert links[reset[0].ref]["contradicts"][0]["clause"] == "pol/L2/§2", links
    # urgency written as an allowed shortcut does not count as a rule against it
    p2 = dict(profile, procedures=["Urgent executive requests must never skip verification."])
    assert "urgency-bypass" not in {f.rule for f in social.analyze(p2)[0]}


chk("social engineering: role level only, authorization required, rule vs routine", t15)


def t16():
    from deepthink.nomosloop import security, surface

    text = ACCESS.read_text(encoding="utf-8")
    res = run(
        text,
        {
            "offline": True,
            "drive": "leak",
            "iterations": 2,
            "repo_path": str(SURFACE / "repo"),
            "social_profile": str(SURFACE / "social_profile.json"),
        },
    )
    assert not res.get("error"), res.get("error")
    assert res["surface"]["code"]["facts"] >= 8 and res["surface"]["social"]["roles"] == 3
    status = {r["ref"]: r for r in res["discrepancies"]}
    by_rule = {r["rule"]: r for r in res["discrepancies"]}
    assert by_rule["remote-desktop"]["contradicts"] == ["pol/L2/§8"], by_rule["remote-desktop"]
    assert "pol/L1/§12" in by_rule["open-ingress"]["contradicts"]
    assert by_rule["sql-injection"]["status"] == "uncovered"
    assert any(t["ax"].startswith("code/") for t in res["toc"])
    assert any(t["ax"].startswith("soc/") for t in res["toc"])
    # facts were mapped offline (no remote mapper call for code or procedures)
    assert all(t["mapper"] == "offline-lexicon" for t in res["toc"] if t["ax"] in status)
    hit = [f for f in res["findings"] if any(c["a1"].startswith("code/") for c in f["clashes"])]
    assert hit, [f["clashes"] for f in res["findings"]]
    assert "Remediation:" in hit[0]["finding"]
    assert "## Code and configuration" in res["artifact_code"]
    assert "## Social engineering" in res["artifact_code"]
    assert "demo-not-a-real-secret" not in json.dumps(res, default=str)
    # the policy's own seeded clashes still surface beside the code findings
    seeded = {("pol/L2/§2", "pol/L2/§7"), ("pol/L2/§5", "pol/L2/§8")}
    found = {(c["a1"], c["a2"]) for f in res["findings"] for c in f["clashes"]}
    assert found & seeded, found
    # without surface inputs nothing changes
    plain = run(text, {"offline": True, "drive": "leak", "iterations": 1})
    assert plain["discrepancies"] == [] and plain["surface"] == {}
    assert security.parse_policy(text, "x") and surface.KINDS == ("code", "social")
    try:
        run(text, {"offline": True, "social_profile": {"roles": []}})
        raise AssertionError("a profile without authorization was accepted")
    except ValueError as e:
        assert "authorization" in str(e), e


chk("offline loop reads code and procedures against the policy", t16)


def t17():
    import importlib
    import os

    import app as app_mod
    from deepthink.config import get_settings
    from fastapi.testclient import TestClient

    client = TestClient(app_mod.app)
    base = {"text": ACCESS.read_text(encoding="utf-8"), "offline": True}
    old = os.environ.pop("RSI_REPO_ROOTS", None)
    try:
        get_settings.cache_clear()
        r = client.post("/start_rsi", json={**base, "repo_path": str(SURFACE / "repo")})
        assert r.status_code == 400 and "RSI_REPO_ROOTS" in r.json()["message"], r.text
        os.environ["RSI_REPO_ROOTS"] = str(SURFACE)
        get_settings.cache_clear()
        r = client.post("/start_rsi", json={**base, "repo_path": str(ROOT / "deepthink")})
        assert r.status_code == 400 and "outside" in r.json()["message"], r.text
        r = client.post(
            "/start_rsi", json={**base, "repo_path": str(SURFACE / "repo" / ".." / ".." / "..")}
        )
        assert r.status_code == 400, r.text
        r = client.post("/start_rsi", json={**base, "social_profile": "{not json"})
        assert r.status_code == 400 and "JSON" in r.json()["message"], r.text
        r = client.post("/start_rsi", json={**base, "social_profile": "[1, 2]"})
        assert r.status_code == 400, r.text
        assert app_mod._rsi_repo_path(str(SURFACE / "repo")).endswith("repo")
    finally:
        if old is None:
            os.environ.pop("RSI_REPO_ROOTS", None)
        else:
            os.environ["RSI_REPO_ROOTS"] = old
        get_settings.cache_clear()
        importlib.invalidate_caches()


chk("API: repository scans only under RSI_REPO_ROOTS; profile must be a JSON object", t17)


VERDICTS = {
    "dissolves",
    "improves",
    "breeds",
    "leaks",
    "breaks",
    "neutral",
    "load-bearing",
    "slack",
}


def _one_loop_world():
    """World and policy after one offline loop (evolution off)."""
    import os

    from deepthink.nomosloop.graph import run_pipeline

    old = os.environ.get("NOMOSLOOP_OFFLINE")
    os.environ["NOMOSLOOP_OFFLINE"] = "1"
    try:
        with tempfile.TemporaryDirectory() as d:
            st = run_pipeline(
                ACCESS.read_text(encoding="utf-8"),
                max_iterations=1,
                artifact_path=str(Path(d) / "r.md"),
                rsi_cfg={"drive": "bal", "cot": False},
            )
    finally:
        if old is None:
            os.environ.pop("NOMOSLOOP_OFFLINE", None)
        else:
            os.environ["NOMOSLOOP_OFFLINE"] = old
    return st["world"], st["scale_lattice"]


def t18():
    from deepthink.nomosloop import evolution as ev
    from deepthink.nomosloop import security

    assert (
        ev.mirror("Firewall rules block inbound traffic to the HR database from the internet.")
        == "Firewall rules block outbound traffic from the HR database to the internet."
    )
    assert (
        ev.mirror("Contractors are granted access to the HR database only through the VPN.") is None
    )
    assert ev.negate("Remote desktop must be disabled on every server.") == (
        "Remote desktop must be enabled on every server."
    )
    assert ev.negate("Personal data must never be stored in a public bucket.") == (
        "Personal data may be stored in a public bucket."
    )
    assert ev.negate("Staff are granted read access to the share.") is None
    assert ev.split_clause(
        "Every access to the /finance share is logged and retained for 365 days."
    ) == (
        "Every access to the /finance share is logged.",
        "Every access to the /finance share is retained for 365 days.",
    )
    assert ev.split_clause(
        "Service accounts must use unique credentials and must not share passwords."
    ) == (
        "Service accounts must use unique credentials.",
        "Service accounts must not share passwords.",
    )
    assert ev.object_phrase(
        "Staff are granted read access to the /finance share at all times."
    ) == ("the /finance share")
    assert ev.object_phrase("Service accounts must use unique credentials.") == "service accounts"
    # an explicit precedence settles a clash, and does not flip the clause's own polarity
    lat = security.parse_policy(ACCESS.read_text(encoding="utf-8"), "x")
    for lv in lat["levels"]:
        for r in lv["rules"]:
            if r["id"] == "§2":
                r["rewrite"] = ev.except_for(r["rewrite"], "§7")
    pairs = {(a.split("/")[-1], b.split("/")[-1]) for a, b, _ in security.contradictions(lat)}
    assert pairs == {("§5", "§8")}, pairs
    cfg = ev.EvolveConfig.from_dict(
        {
            "enabled": 1,
            "universes": 99,
            "generations": 0,
            "trigger": "x",
            "adopt": "gate",
            "seed": "5",
        }
    )
    assert (cfg.universes, cfg.generations, cfg.trigger, cfg.adopt, cfg.seed) == (
        16,
        1,
        "always",
        "gate",
        5,
    )
    assert ev.EvolveConfig.from_dict(None).enabled is False


chk("evolution: what-if text transforms, explicit precedence, config clamps", t18)


def t19():
    import copy

    from deepthink.nomosloop import evolution as ev
    from deepthink.nomosloop import security

    world, lat = _one_loop_world()
    frozen = (
        len(world.agents),
        json.dumps(lat, sort_keys=True, default=str),
        world.t,
        len(world.events),
    )

    def fresh():
        return world.fork(), copy.deepcopy(lat)

    # cut: §2 yields to §7, the contradiction dissolves and the edge from §7's agent is cut
    w, L = fresh()
    rec = ev.apply(w, L, {"op": "cut", "ref": "pol/L2/§2", "other": "pol/L2/§7"})
    assert rec and "except where §7 applies" in rec["edits"][0]["after"], rec
    assert not rec["weakens"]  # the permission yields to the restriction
    pairs = {(a.split("/")[-1], b.split("/")[-1]) for a, b, _ in security.contradictions(L)}
    assert ("§2", "§7") not in pairs, pairs
    a2, a7 = ev._agent(w, "pol/L2/§2"), ev._agent(w, "pol/L2/§7")
    assert (a7.id, a2.id) in w.cut
    # the restriction yielding to a permission weakens it (never adopted)
    w, L = fresh()
    assert ev.apply(w, L, {"op": "cut", "ref": "pol/L2/§7", "other": "pol/L2/§2"})["weakens"]

    # hairpin η_a: a Klein-close child of the §12 agent, paid with energy, and the mirrored clause
    w, L = fresh()
    parent = ev._agent(w, "pol/L1/§12")
    energy = sum(parent.energy[parent.side])
    rec = ev.apply(w, L, {"op": "hairpin", "g": "a", "ref": "pol/L1/§12"})
    child = w.agents[rec["born"][0]]
    assert ev.klein_close(tuple(child.stack), tuple(parent.stack)), (child.stack, parent.stack)
    assert (parent.id, child.id) in w.pr_edges
    assert "outbound traffic from the HR database to the internet" in rec["edits"][0]["after"]
    assert sum(parent.energy[parent.side]) < energy
    assert {"from": "pol/L1/§12", "to": "pol/L1/§12·a", "kind": "standard"} in L["links"]

    # birth a·(αX, βY): two pr parents, Klein-close to the winner
    births = [s for _, s in ev.candidates(world, lat) if s["op"] == "birth"]
    assert births, "no linked pair leaks enough for a birth"
    w, L = fresh()
    rec = ev.apply(w, L, births[0])
    child = w.agents[rec["born"][0]]
    winner = w.agents[rec["klein"][child.id]]
    assert ev.klein_close(tuple(child.stack), tuple(winner.stack))
    assert len({s for s, d in w.pr_edges if d == child.id}) == 2

    # store4: three children (log, alert, review) and a four-cycle of stems
    w, L = fresh()
    parent = ev._agent(w, "pol/L1/§10")
    rec = ev.apply(w, L, {"op": "store4", "ref": "pol/L1/§10"})
    assert len(rec["born"]) == 3 and len(rec["new_refs"]) == 3
    ring = [parent.id, *rec["born"], parent.id]
    assert all((s, d) in w.stems for s, d in zip(ring, ring[1:])), w.stems

    # arrow: §4 splits in two, the parent dissolves, its mass is conserved
    w, L = fresh()
    parent = ev._agent(w, "pol/L2/§4")
    mass = sum(parent.w)
    rec = ev.apply(w, L, {"op": "arrow", "ref": "pol/L2/§4"})
    kids = [w.agents[k] for k in rec["born"]]
    assert len(kids) == 2 and not parent.alive
    assert abs(sum(sum(k.w) for k in kids) - mass) < 1e-6
    assert (
        ev._locate(L, "pol/L2/§4.2")[1]["rewrite"] == "Service accounts must not share passwords."
    )

    # σ: a stem to the agent that covers the weakest slot; † and ω
    w, L = fresh()
    sig = next(s for _, s in ev.candidates(world, lat) if s["op"] == "sigma")
    rec = ev.apply(w, L, sig)
    a, b = ev._agent(w, sig["ref"]), ev._agent(w, sig["other"])
    assert (a.id, b.id) in w.stems and rec["edits"][0]["kind"] == "link"
    w, L = fresh()
    rec = ev.apply(w, L, {"op": "dissolve", "ref": "pol/L2/§8"})
    assert rec["weakens"] and ev._locate(L, "pol/L2/§8")[1] is None
    w, L = fresh()
    rec = ev.apply(w, L, {"op": "omega", "ref": "pol/L2/§9"})
    assert rec["probe"] and "may be stored in a public bucket" in rec["edits"][0]["after"]
    # a mutation that does not apply leaves the lattice alone
    w, L = fresh()
    assert ev.apply(w, L, {"op": "arrow", "ref": "pol/L2/§3"}) is None
    assert json.dumps(L, sort_keys=True, default=str) == frozen[1]
    # none of this touched the real world or policy
    after = (
        len(world.agents),
        json.dumps(lat, sort_keys=True, default=str),
        world.t,
        len(world.events),
    )
    assert after == frozen


chk("evolution: each operator acts in both spaces (cut, η, birth, store4, →, σ, †, ω)", t19)


def t20():
    text = ACCESS.read_text(encoding="utf-8")
    plain = run(text, {"offline": True, "drive": "bal", "iterations": 2})
    on = {"enabled": True, "seed": 11}
    evo = run(text, {"offline": True, "drive": "bal", "iterations": 2, "evolve": on})
    assert plain["evolution"] == [] and "## Alternate universes" not in plain["artifact_code"]
    # what-ifs only: the real loop is untouched
    assert [x["score"] for x in plain["posture_history"]] == [
        x["score"] for x in evo["posture_history"]
    ]
    assert [c["text"] for c in plain["controls"]] == [c["text"] for c in evo["controls"]]
    assert len(plain["findings"]) == len(evo["findings"])
    recs = evo["evolution"]
    assert [r["loop"] for r in recs] == [1, 2] and all(r["adopted"] is None for r in recs)
    for rec in recs:
        assert rec["universes"], rec
        assert rec["survivors"]["1"] and rec["survivors"]["2"]
        for u in rec["universes"] + rec["probes"]:
            assert u["what_if"].startswith("What if"), u["what_if"]
            assert u["diff"] and u["verdict"] in VERDICTS, u
            assert all(m["law"] and m["agent_change"] for m in u["mutations"]), u["mutations"]
        assert all(
            not u["adoptable"] and u["verdict"] in ("load-bearing", "slack") for u in rec["probes"]
        )
        gen2 = [u for u in rec["universes"] if u["generation"] == 2]
        assert gen2 and all(u["parents"] != ["U0"] for u in gen2)
    assert "## Alternate universes" in evo["artifact_code"]
    assert any(e["kind"] == "branch" for e in evo["events"])
    assert evo["snapshot"]["evolution"]["universes"]
    again = run(text, {"offline": True, "drive": "bal", "iterations": 2, "evolve": on})
    assert [u["what_if"] for u in again["evolution"][0]["universes"]] == [
        u["what_if"] for u in recs[0]["universes"]
    ]


chk("evolution in the loop: universes are what-ifs, reproducible, the real loop untouched", t20)


def t21():
    text = ACCESS.read_text(encoding="utf-8")
    res = run(
        text,
        {
            "offline": True,
            "drive": "bal",
            "iterations": 2,
            "evolve": {"enabled": True, "adopt": "gate", "seed": 1},
        },
    )
    adopted = [(r["loop"], r["adopted"]) for r in res["evolution"] if r.get("adopted")]
    assert adopted, [r.get("adopted") for r in res["evolution"]]
    sources = [x["source"] for x in res["posture_history"]]
    for loop, uid in adopted:
        assert f"evolve:{uid}" in sources, sources
        u = next(x for x in res["evolution"][loop - 1]["universes"] if x["id"] == uid)
        assert u["adoptable"] and u["delta"]["posture"] > 0 and u["delta"]["contradictions"] <= 0
        assert not any(
            m["weakens"] or m["op"] in ("ev", "dissolve", "omega") for m in u["mutations"]
        )
    i = sources.index(f"evolve:{adopted[0][1]}")
    assert res["posture_history"][i]["score"] > res["posture_history"][i - 1]["score"]
    tagged = [
        r
        for lv in res["scale_lattice"]["levels"]
        for r in lv["rules"]
        if str(r.get("provenance", "")).startswith("rsi:evolve:")
    ]
    assert tagged and "(RSI)" in res["artifact_code"]


chk("evolution adoption: the gate applies only additive, non-weakening universes", t21)


def t22():
    from deepthink.cli import build_parser

    text = ACCESS.read_text(encoding="utf-8")
    res = run(
        text,
        {
            "offline": True,
            "drive": "leak",
            "iterations": 1,
            "evolve": {"enabled": True, "trigger": "calm"},
        },
    )
    rec = res["evolution"][0]
    assert rec.get("skipped") and not rec["universes"], rec
    assert any(line.startswith("evolve: skipped") for line in res["topology_log"])
    args = build_parser().parse_args(
        [
            "rsi",
            "-i",
            "p.txt",
            "--evolve",
            "--universes",
            "4",
            "--evolve-adopt",
            "gate",
            "--evolve-trigger",
            "calm",
        ]
    )
    assert (
        args.evolve
        and args.universes == 4
        and args.evolve_adopt == "gate"
        and args.evolve_trigger == "calm"
    )
    import app as app_mod
    from fastapi.testclient import TestClient

    r = TestClient(app_mod.app).post(
        "/start_rsi", json={"text": text, "offline": True, "evolve": "yes"}
    )
    assert r.status_code == 400 and "evolve" in r.json()["message"], r.text


chk("evolution: branch only at equilibrium when asked; CLI flags; API validation", t22)


def t23():
    """The equilibrium case: no agent leaks, the policy is fixed, and new structures still breed."""
    import copy

    from deepthink.nomosloop import graph

    world, lat = _one_loop_world()
    for a in world.live().values():
        if a.state == "leak":
            a.state = "stock"  # equilibrium: nothing leaks
    policy = json.dumps(lat, sort_keys=True, default=str)
    state = {
        "rsi_cfg": {"mutators": 8, "evolve": {"enabled": True, "trigger": "calm", "seed": 2}},
        "world": world,
        "scale_lattice": copy.deepcopy(lat),
        "iteration": 0,
        "law_hash": "eq",
        "evolution": [],
        "surface_facts": [],
        "lora_buffer": [],
    }
    out = graph.evolve_node(state)
    rec = out["evolution"][-1]
    assert not rec.get("skipped") and rec["universes"], rec
    assert "scale_lattice" not in out  # adoption is off: the policy did not change
    assert json.dumps(lat, sort_keys=True, default=str) == policy
    bred = [s for u in rec["universes"] for s in u["structures"]]
    assert bred, "no structure was bred at equilibrium"
    assert any(s["klein_close"] for s in bred) or any(u["rings"] for u in rec["universes"])
    assert any(ex["kind"] == "evolve" for ex in out["lora_buffer"])


chk("evolution at equilibrium: no leak, fixed policy, new structures still breed", t23)


ok = sum(1 for _, s, _ in results if s == "OK")
total = len(results)
for name, status, err in results:
    print(f"  [{status}] {name}" + (f" :: {err}" if err else ""))
print(f"\nPHASE RSI: {ok}/{total} OK")
sys.exit(0 if ok == total else 1)
