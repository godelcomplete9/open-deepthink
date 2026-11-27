"""Regression tests for faults found by scale and live testing (keyless).

Child-agent mixing, credentials in stored and exported runs, the default model,
distillation runs that cannot progress or overlap, and policy numbering styles.
"""

import asyncio
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepthink.mocks import DistillationMockLLM, MockLLM
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

results = []
SECRET = "sk-test-regression-0000"
RETIRED_MODEL = "stepfun/step-3.5-flash:free"


def chk(name, fn):
    try:
        fn()
        results.append((name, "OK", None))
    except AssertionError as e:
        results.append((name, "FAIL", f"AssertionError: {e}"))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def drain(queue) -> list[str]:
    out = []
    while not queue.empty():
        out.append(str(queue.get_nowait()))
    return out


def failing_llm(calls: list):
    def boom(_):
        calls.append(1)
        raise RuntimeError("HTTP 404: no such model")

    return RunnableLambda(boom)


def graph(llm, **kwargs):
    from deepthink.knowledge_distillation import DistillationGraph

    return DistillationGraph(llm, ["t"], "anchor", output_dir=tempfile.mkdtemp(), **kwargs)


# ── Mixing: child agents ──────────────────────────────────────────────


def mixing_prompt_variables():
    from deepthink.chains import get_mixing_chain

    seen = []

    def capture(prompt):
        seen.append(prompt.to_string())
        return AIMessage(content="{}")

    chain = get_mixing_chain(RunnableLambda(capture), density=0.5)
    chain.invoke(
        {
            "parent_a_attributes": "ATTR-A",
            "parent_a_prompt": "PROMPT-A {braces}",
            "parent_b_attributes": "ATTR-B",
            "parent_b_prompt": "PROMPT-B",
        }
    )
    text = seen[0]
    for part in ("ATTR-A", "PROMPT-A {braces}", "ATTR-B", "PROMPT-B", "density parameter is 0.5"):
        assert part in text, part
    assert '"new_system_prompt"' in text and "{{" not in text


chk("mixing prompt takes both parents and keeps its JSON example", mixing_prompt_variables)


def child_from(reply: dict):
    g = graph(RunnableLambda(lambda _: AIMessage(content=json.dumps(reply))))
    child, helper = g._flat_agents()[:2]
    before = (child.system_prompt, list(child.attributes), list(child.skills))
    asyncio.run(g._handle_hard_agent(child, {"best_match_agent_id": helper.id}, g._flat_agents()))
    # The grid description joins these lists, so they must stay lists of strings.
    g._build_current_grid_description()
    return child, before


def child_spawns():
    child, before = child_from(
        {"new_system_prompt": "CHILD", "new_attributes": ["x"], "new_skills": ["y"]}
    )
    assert child.system_prompt == "CHILD" and child.attributes == ["x"] and child.skills == ["y"]
    assert child.inherited_from


chk("a hard agent is replaced by the mixed child", child_spawns)


def child_with_wrong_types():
    child, before = child_from(
        {"new_system_prompt": 7, "new_attributes": [{"a": 1}], "new_skills": "p, q"}
    )
    assert (child.system_prompt, child.attributes) == before[:2]
    assert child.skills == ["p", "q"]


chk("wrong value types in a mixing reply keep the parent's values", child_with_wrong_types)


# ── Credentials ──────────────────────────────────────────────────────


def strip_secrets_names():
    from deepthink.utils import strip_secrets

    secret = ["api_key", "llamacpp_api_key", "openrouterApiKey", "token", "hf_token", "password"]
    kept = ["max_tokens", "token_budget", "num_epochs", "prompt", "seed"]
    out = strip_secrets({**dict.fromkeys(secret, SECRET), **dict.fromkeys(kept, 1), "n": [{}]})
    assert sorted(out) == sorted(kept + ["n"]), out
    assert strip_secrets({"a": [{"api_key": SECRET, "b": 2}]}) == {"a": [{"b": 2}]}


chk("strip_secrets removes credential fields and nothing else", strip_secrets_names)


def qnn_keeps_no_key():
    from deepthink.qnn import run_qnn_pipeline

    logs, store = [], {"s": {}}
    result = asyncio.run(
        run_qnn_pipeline(
            llm=MockLLM(),
            params={"api_key": SECRET, "num_epochs": 1, "manual_layers": 1, "manual_width": 1},
            user_prompt="design a queue",
            session_id="s",
            session_store=store,
            log=logs.append,
        )
    )
    assert SECRET not in "\n".join(map(str, logs)), "key in log"
    assert SECRET not in json.dumps(result, default=str), "key in result"
    assert SECRET not in json.dumps(store, default=str), "key in session"


chk("QNN never logs, returns or stores the API key", qnn_keeps_no_key)


def session_file_keeps_no_key():
    from deepthink.sessions import SessionStore

    store = SessionStore(tempfile.mkdtemp())
    store["a"] = {"params": {"api_key": SECRET, "prompt": "p"}}
    text = (store.persist_dir / "a.json").read_text(encoding="utf-8")
    assert SECRET not in text and '"prompt"' in text


chk("session files never contain credentials", session_file_keeps_no_key)


def server_keeps_no_key():
    from fastapi.testclient import TestClient

    previous = os.getcwd()
    os.chdir(ROOT)
    try:
        import app as server
    finally:
        os.chdir(previous)

    server.sessions.persist_dir = Path(tempfile.mkdtemp())

    async def idle(**_):
        return None

    originals = (server.run_qnn_background, server.run_qdad_background)
    server.run_qnn_background = server.run_qdad_background = idle
    try:
        with TestClient(server.app) as client:
            for mode in ("brainstorm", "app_slot_machine"):
                params = {
                    "api_key": SECRET,
                    "llamacpp_api_key": SECRET,
                    "debug_mode": "true",
                    "prompt": "p",
                }
                r = client.post("/build_and_run_graph", json={"mode": mode, "params": params})
                assert r.status_code == 200, r.text
                sid = r.json()["session_id"]
                assert SECRET not in client.get(f"/export_qnn/{sid}").text, f"{mode} export"
            old_export = json.dumps({"params": {"api_key": SECRET, "prompt": "p"}})
            r = client.post("/import_qnn", files={"file": ("old.json", old_export)})
            assert r.status_code == 200 and SECRET not in r.text, r.text
            assert SECRET not in client.get(f"/export_qnn/{r.json()['session_id']}").text
        for path in server.sessions.persist_dir.glob("*.json"):
            assert SECRET not in path.read_text(encoding="utf-8"), path.name
    finally:
        server.run_qnn_background, server.run_qdad_background = originals


chk("server sessions, exports and imports drop the API key", server_keeps_no_key)


# ── Default model ────────────────────────────────────────────────────


def retired_model_gone():
    from deepthink.config import Settings

    default = Settings.model_fields["openrouter_model"].default
    assert default != RETIRED_MODEL and default.endswith(":free"), default
    for name in (
        ".env.example",
        "docker-compose.yml",
        "app.py",
        "skills/qnn/run_template.py",
        "skills/qdad/run_template.py",
    ):
        assert RETIRED_MODEL not in (ROOT / name).read_text(encoding="utf-8"), name
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    # The one remaining mention clears the value from browsers that saved it.
    assert page.count(RETIRED_MODEL) == 1 and default in page


chk("the retired default model is replaced everywhere", retired_model_gone)


# ── Distillation runs ────────────────────────────────────────────────


def dead_provider_stops():
    calls: list = []

    async def run():
        from deepthink.api import run_distillation

        return await asyncio.wait_for(
            run_distillation(failing_llm(calls), "anchor", output_dir=tempfile.mkdtemp()), 60
        )

    g = asyncio.run(run())
    assert g.epochs_run == g.MAX_STALLED_EPOCHS, g.epochs_run
    assert "No agent answered" in g.error and calls and len(calls) < 100, len(calls)
    assert Path(g.dataset_path).is_file(), "a failed run still leaves its dataset file"


chk("distillation stops when no agent can answer", dead_provider_stops)


def empty_answers_stop():
    g = graph(RunnableLambda(lambda _: AIMessage(content="  ")))

    async def run():
        while await g.run_epoch():
            assert g.epochs_run < 10

    asyncio.run(run())
    assert g.error and g.epochs_run == g.MAX_STALLED_EPOCHS


chk("distillation stops when every answer is empty", empty_answers_stop)


def recovery_resets_stall_count():
    g = graph(failing_llm([]))

    async def run():
        assert await g.run_epoch() and await g.run_epoch()
        g.llm = DistillationMockLLM()
        assert await g.run_epoch()

    asyncio.run(run())
    assert g.stalled_epochs == 0 and not g.error and g.distilled_data


chk("a recovered provider resets the stall count", recovery_resets_stall_count)


def server_distillation():
    import app as server

    async def run():
        drain(server.log_stream)
        payload = {"topics": "a", "anchor_question": "q", "debug_mode": True}
        gate = asyncio.Event()
        original = server.run_distillation_loop

        async def held(*_):
            await gate.wait()

        server.run_distillation_loop = held
        try:
            first = await server.start_distillation(dict(payload))
            second = await server.start_distillation(dict(payload))
            assert first["status"] == "started"
            assert getattr(second, "status_code", 200) == 409, second
            gate.set()
            await server.distillation_task
            third = await server.start_distillation(dict(payload))
            assert third["status"] == "started", "a finished run must not block the next"
            await server.distillation_task
        finally:
            server.run_distillation_loop = original

        drain(server.log_stream)
        anchors = [{"question": "q1", "budget": 10_000_000}, {"question": "q2", "budget": 1000}]
        cwd = os.getcwd()
        os.chdir(tempfile.mkdtemp())
        try:
            await asyncio.wait_for(original(failing_llm([]), ["a"], anchors, False), 60)
        finally:
            os.chdir(cwd)
        return drain(server.log_stream)

    lines = asyncio.run(run())
    done = [json.loads(x) for x in lines if x.startswith('{"type": "distillation_complete"')]
    assert done and "No agent answered" in done[0]["error"], done
    assert not any("Anchor 2/2" in x for x in lines), "the next anchor must not start"


chk("server rejects overlapping runs and reports a dead provider", server_distillation)


# ── Policy numbering ─────────────────────────────────────────────────


def numbering_styles():
    from deepthink.nomosloop.security import parse_clauses

    body = "Administrators must rotate credential set {n} every quarter."
    for fmt, last in (
        ("1.{n}", "1.40"),
        ("A.{n}", "A.40"),
        ("AC-{n}", "AC-40"),
        ("AC-{n}:", "AC-40"),
        ("Section {n}:", "40"),
        ("1.{n} —", "1.40"),
    ):
        text = "\n".join(f"{fmt.format(n=n)} {body.format(n=n)}" for n in range(1, 41))
        out = parse_clauses(text)
        assert len(out) == 40 and out[-1]["num"] == last, (fmt, len(out), out[-1]["num"])


chk("policy clauses are split for 1.1, A.1, AC-1 and Section numbering", numbering_styles)


def wrapped_prose_stays_whole():
    from deepthink.nomosloop.security import parse_clauses

    text = (
        "- All traffic must use TLS version\n1.2 or higher on every link.\n"
        "- Backups use at least\n2.5 GB of space, encrypted with\nAES-256 in transit.\n"
        "- Hosts must follow\nSection 4 of the standard.\n"
    )
    out = parse_clauses(text)
    assert [c["num"] for c in out] == ["c1", "c2", "c3"], out


chk("wrapped lines starting with numbers are not split", wrapped_prose_stays_whole)


def section_headings():
    from deepthink.nomosloop.security import parse_clauses

    nested = parse_clauses(
        "Section 1: Access Control\n1.1 All users must use MFA.\n1.2 No shared accounts.\n"
        "Section 2: Logging\n2.1 Logs are kept for one year.\n"
    )
    assert [(c["num"], c["section"]) for c in nested] == [
        ("1.1", "Access Control"),
        ("1.2", "Access Control"),
        ("2.1", "Logging"),
    ], nested
    flat = parse_clauses("Section 1: Passwords rotate yearly\nSection 2: Admins use MFA\n")
    assert [c["num"] for c in flat] == ["1", "2"], flat


chk("Section titles head numbered clauses, and stand alone otherwise", section_headings)


ok = sum(1 for _, s, _ in results if s == "OK")
for name, status, err in results:
    print(f"[{status}] {name}")
    if err:
        print(err)
print(f"PHASE REGRESSIONS: {ok}/{len(results)} OK")
sys.exit(0 if ok == len(results) else 1)
