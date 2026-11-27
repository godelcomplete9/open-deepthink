"""Server and CLI checks: fuzzed requests, failing background runs, concurrency, secrets.

Every provider class the server would call is replaced by the synthetic model,
so nothing leaves the machine and nothing is billed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from tests.scale.harness import ROOT
from tests.scale.synth import FaultPlan, SyntheticLLM

SECRET = "sk-or-v1-SCALETEST-SECRET-0123456789"

UI_TERMINAL = (
    "final_answer:",
    "brainstorm graph execution completed",
    "theory synthesis complete",
    "theory synthesis graph background error",
    "app slot machine (qdad) completed",
    "qdad) langgraph execution completed",
    "graph background error",
    "qdad pipeline error",
    "success: graph execution completed.",
)


@dataclass
class Check:
    area: str
    name: str
    status: str = "OK"  # OK | WARN | FAIL
    detail: str = ""


@dataclass(eq=False)  # kept in a set of log clients
class _Tap:
    lines: list[str] = field(default_factory=list)

    async def put(self, message):
        self.lines.append(str(message))


def _wait(predicate, seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


FUZZ = [
    ("/build_and_run_graph", {}),
    ("/build_and_run_graph", {"params": None}),
    ("/build_and_run_graph", {"params": "text"}),
    ("/build_and_run_graph", {"mode": 5, "params": {"coder_debug_mode": "true"}}),
    ("/build_and_run_graph", {"mode": "theory", "params": {"coder_debug_mode": "true"}}),
    (
        "/build_and_run_graph",
        {
            "mode": "theory",
            "params": {"coder_debug_mode": "true", "prompt": "x", "swarm_size": "inf"},
        },
    ),
    (
        "/build_and_run_graph",
        {"mode": "theory", "params": {"coder_debug_mode": "true", "prompt": "x", "seed": "inf"}},
    ),
    (
        "/build_and_run_graph",
        {"mode": "theory", "params": {"coder_debug_mode": "true", "prompt": "x", "quorum": "abc"}},
    ),
    (
        "/build_and_run_graph",
        {"mode": "theory", "params": {"coder_debug_mode": "true", "prompt": {"a": 1}}},
    ),
    (
        "/build_and_run_graph",
        {
            "mode": "theory",
            "chat_history": "x",
            "params": {"coder_debug_mode": "true", "prompt": "x"},
        },
    ),
    (
        "/build_and_run_graph",
        {
            "mode": "brainstorm",
            "params": {"coder_debug_mode": "true", "prompt": "x", "num_epochs": "two"},
        },
    ),
    (
        "/build_and_run_graph",
        {
            "mode": "app_slot_machine",
            "params": {"coder_debug_mode": "true", "prompt": "x", "grid_size": "big"},
        },
    ),
    ("/build_and_run_graph", {"mode": "theory", "params": {"provider": "nope", "prompt": "x"}}),
    ("/start_distillation", {"topics": 5, "debug_mode": True}),
    ("/start_distillation", {"anchors": "x", "debug_mode": True}),
    ("/start_rsi", {}),
    ("/start_rsi", {"text": "x", "iterations": "many"}),
    ("/start_rsi", {"text": "§1 a\n§2 b", "lora": "yes"}),
    ("/start_rsi", {"text": "§1 a\n§2 b", "sensitivity": {"S_min": "x"}}),
    ("/run_inference_from_state", {}),
    ("/run_inference_from_state", {"imported_state": "x", "prompt": 1}),
    ("/chat", {}),
    ("/chat", {"message": "hi", "session_id": "missing"}),
    ("/diagnostic_chat", {}),
    ("/harvest", {}),
    ("/harvest", {"session_id": "missing"}),
]


def server_checks(progress=None) -> list[Check]:
    cwd = os.getcwd()
    os.chdir(ROOT)  # app.py mounts its static folders relative to the repository
    import app as app_mod
    from fastapi.testclient import TestClient

    # Work in a scratch folder so runs never overwrite distillation_output/ or artifacts/.
    scratch = tempfile.mkdtemp(prefix="odt_server_")
    checks: list[Check] = []
    made: list[SyntheticLLM] = []
    factory_plan: dict[str, FaultPlan] = {"plan": FaultPlan()}
    delay = {"s": 0.0}

    def factory(*_a, **_k):
        llm = SyntheticLLM(len(made), plan=factory_plan["plan"], max_calls=4000, delay=delay["s"])
        made.append(llm)
        return llm

    saved = {
        n: getattr(app_mod, n)
        for n in ("ChatOpenAI", "ChatLlamaCpp", "CoderMockLLM", "DistillationMockLLM")
    }
    for name in saved:
        setattr(app_mod, name, factory)
    old_dir = app_mod.sessions.persist_dir
    state_dir = Path(tempfile.mkdtemp(prefix="odt_sessions_"))
    app_mod.sessions.persist_dir = state_dir
    os.chdir(scratch)
    tap = _Tap()
    try:
        with TestClient(app_mod.app, raise_server_exceptions=False) as client:
            app_mod.connected_log_clients.add(tap)

            # 1) fuzzed requests: nothing may end in an unhandled 500
            for path, payload in FUZZ:
                r = client.post(path, json=payload)
                label = f"{path} {json.dumps(payload)[:90]}"
                if r.status_code >= 500:
                    handled = r.headers.get("content-type", "").startswith("application/json")
                    checks.append(
                        Check(
                            "server fuzz",
                            label,
                            "WARN" if handled else "FAIL",
                            f"{r.status_code} {'JSON error' if handled else 'unhandled'}: "
                            f"{r.text[:140]}",
                        )
                    )
                else:
                    checks.append(Check("server fuzz", label, detail=str(r.status_code)))
                client.post("/stop_distillation")
            for path, files in (
                ("/import_qnn", {"file": ("x.json", b"not json", "application/json")}),
                ("/import_qnn", {"file": ("x.json", b"[1,2]", "application/json")}),
                ("/upload_documents", {"files": ("x.pdf", b"%PDF-broken", "application/pdf")}),
                ("/upload_code_files", {"files": ("x.py", b"\xff\xfe\x00", "text/plain")}),
            ):
                r = client.post(path, files=files)
                bad = r.status_code >= 500 and not r.headers.get("content-type", "").startswith(
                    "application/json"
                )
                checks.append(
                    Check(
                        "server fuzz",
                        f"{path} {files}"[:100],
                        "FAIL" if bad else "OK",
                        str(r.status_code),
                    )
                )
            _wait(lambda: app_mod.active_distillation_graph is None, 20)
            if progress:
                progress("  server: fuzz done")

            # 2) every model call fails: does the UI ever learn the run is over?
            factory_plan["plan"] = FaultPlan("raise", 1.0)
            for mode in ("theory", "brainstorm", "app_slot_machine"):
                time.sleep(0.5)
                tap.lines.clear()
                r = client.post(
                    "/build_and_run_graph",
                    json={
                        "mode": mode,
                        "params": {
                            "coder_debug_mode": "true",
                            "prompt": "x",
                            "qnn_mode": "manual",
                            "manual_layers": 2,
                            "manual_width": 2,
                            "swarm_size": 2,
                        },
                    },
                )
                ended = _wait(
                    lambda: any(t in ln.lower() for ln in tap.lines for t in UI_TERMINAL), 12
                )
                checks.append(
                    Check(
                        "server failures",
                        f"{mode}: provider fails on every call",
                        "OK" if ended else "FAIL",
                        "UI gets a terminal signal"
                        if ended
                        else f"HTTP {r.status_code}, then no terminal log line in 12 s: the UI spins forever",
                    )
                )

            # 3) distillation on a failing provider
            made.clear()
            client.post(
                "/start_distillation",
                json={
                    "topics": "a",
                    "anchor_question": "q",
                    "token_budget": 1000,
                    "debug_mode": True,
                },
            )
            time.sleep(3.0)
            calls = sum(m.meter.calls for m in made)
            running = app_mod.active_distillation_graph is not None
            checks.append(
                Check(
                    "server failures",
                    "distillation: provider fails on every call",
                    "FAIL" if running and calls > 200 else "OK",
                    f"after 3 s still running={running} with {calls} calls and rising "
                    "(budget 1000 tokens, 0 ever counted)",
                )
            )
            client.post("/stop_distillation")
            _wait(lambda: app_mod.active_distillation_graph is None, 20)
            factory_plan["plan"] = FaultPlan()

            # 4) two distillations at once
            made.clear()
            tap.lines.clear()
            delay["s"] = 0.02
            codes = []
            for q in ("first anchor", "second anchor"):
                codes.append(
                    client.post(
                        "/start_distillation",
                        json={
                            "topics": "a",
                            "anchor_question": q,
                            "token_budget": 10**9,
                            "debug_mode": True,
                        },
                    ).status_code
                )
                time.sleep(0.4)
            time.sleep(1.0)
            client.post("/stop_distillation")
            time.sleep(2.5)
            before = sum(m.meter.calls for m in made)
            time.sleep(1.0)
            after = sum(m.meter.calls for m in made)
            errors = [ln for ln in tap.lines if "Distillation Error" in ln or "NoneType" in ln]
            problems = []
            if all(c == 200 for c in codes):
                problems.append("a second run is accepted while one is active")
            if errors:
                problems.append(f"loop crashed: {errors[0][:120]}")
            if after > before:
                problems.append(f"calls keep rising after stop ({before} → {after})")
            checks.append(
                Check(
                    "server concurrency",
                    "two distillation runs",
                    "FAIL" if len(problems) > 1 or errors else ("WARN" if problems else "OK"),
                    "; ".join(problems),
                )
            )
            for _ in range(3):
                client.post("/stop_distillation")
                time.sleep(0.5)
            delay["s"] = 0.0

            # 5) two theory runs at once share one log stream
            tap.lines.clear()
            sids = []
            for n in (1, 2):
                r = client.post(
                    "/build_and_run_graph",
                    json={
                        "mode": "theory",
                        "params": {
                            "coder_debug_mode": "true",
                            "prompt": f"p{n}",
                            "swarm_size": 2,
                            "rounds": 1,
                        },
                    },
                )
                sids.append(r.json().get("session_id"))
            _wait(lambda: sum(ln.startswith("FINAL_ANSWER:") for ln in tap.lines) >= 2, 20)
            events = [ln for ln in tap.lines if ln.startswith("SWARM_EVENT:")]
            tagged = [ln for ln in events if any(s and s in ln for s in sids)]
            done = sum(ln.startswith("FINAL_ANSWER:") for ln in tap.lines)
            ok_state = all(
                isinstance(app_mod.sessions.get(s, {}).get("final_solution"), dict) for s in sids
            )
            checks.append(
                Check(
                    "server concurrency",
                    "two theory runs: results",
                    "OK" if done == 2 and ok_state else "FAIL",
                    f"{done}/2 finished, session state {'ok' if ok_state else 'missing'}",
                )
            )
            checks.append(
                Check(
                    "server concurrency",
                    "two theory runs: live events",
                    "WARN" if events and not tagged else "OK",
                    f"{len(events)} SWARM_EVENT lines carry no session id, so the "
                    "swarm panel and chat mix both runs"
                    if events and not tagged
                    else "",
                )
            )

            # 6) RSI refuses a second concurrent run
            policy = "\n".join(f"§{i} Staff must log access to system {i}." for i in range(1, 200))
            a = client.post("/start_rsi", json={"text": policy, "iterations": 6, "offline": True})
            b = client.post("/start_rsi", json={"text": policy, "offline": True})
            checks.append(
                Check(
                    "server concurrency",
                    "second RSI run while one is active",
                    "OK" if (a.status_code, b.status_code) == (200, 409) else "WARN",
                    f"{a.status_code} then {b.status_code}",
                )
            )
            _wait(lambda: app_mod.active_rsi_task is None, 60)

            # 7) the API key must not reach logs, exports or disk
            for mode in ("theory", "brainstorm", "app_slot_machine"):
                tap.lines.clear()
                r = client.post(
                    "/build_and_run_graph",
                    json={
                        "mode": mode,
                        "params": {
                            "provider": "openrouter",
                            "api_key": SECRET,
                            "prompt": "x",
                            "swarm_size": 2,
                            "rounds": 1,
                            "qnn_mode": "manual",
                            "manual_layers": 2,
                            "manual_width": 2,
                            "num_epochs": 1,
                        },
                    },
                )
                sid = r.json().get("session_id", "")
                _wait(lambda: any(ln.startswith("FINAL_ANSWER:") for ln in tap.lines), 20)
                time.sleep(0.3)
                leaks = []
                if any(SECRET in ln for ln in tap.lines):
                    leaks.append("log stream")
                if SECRET in client.get(f"/export_qnn/{sid}").text:
                    leaks.append("/export_qnn download")
                disk = state_dir / f"{sid}.json"
                if disk.is_file() and SECRET in disk.read_text(encoding="utf-8", errors="replace"):
                    leaks.append("session file on disk")
                checks.append(
                    Check(
                        "secrets",
                        f"{mode}: API key handling",
                        "FAIL" if leaks else "OK",
                        "key appears in: " + ", ".join(leaks) if leaks else "",
                    )
                )
            app_mod.connected_log_clients.discard(tap)
    finally:
        for name, value in saved.items():
            setattr(app_mod, name, value)
        app_mod.sessions.persist_dir = old_dir
        os.chdir(cwd)
    if progress:
        progress("  server: failure, concurrency and secret checks done")
    return checks


def cli_checks(progress=None) -> list[Check]:
    policy = str(ROOT / "examples" / "security" / "institution_access_policy.txt")
    tmp = Path(tempfile.mkdtemp(prefix="odt_cli_"))
    cases: list[tuple[list[str], set[int]]] = [
        (["version"], {0}),
        (["estimate", "theory"], {0}),
        (["estimate", "qnn"], {0}),
        (["estimate", "qdad"], {0}),
        (["estimate", "distill"], {0}),
        (["estimate", "rsi"], {0}),
        (["theory", "--debug", "-p", "Find x.", "--swarm", "3", "--seed", "1"], {0}),
        (["theory", "--debug"], {2}),
        (["theory", "--debug", "-i", str(tmp / "missing.txt")], {2}),
        (["theory", "--debug", "-p", "x", "--swarm", "0", "--rounds", "0", "--depth", "9"], {0}),
        (["qnn", "--debug", "-p", "x", "--layers", "2", "--width", "2", "--epochs", "1"], {0}),
        (["qdad", "--debug", "-p", "x", "--n", "2", "--steps", "1"], {0}),
        (["rsi", "--input", policy, "--offline", "--artifact", str(tmp / "r.md")], {0}),
        (["rsi", "--input", str(tmp / "missing.txt"), "--offline"], {2}),
        (["eval"], {0}),
    ]
    checks = []
    env = {**os.environ, "OPENROUTER_API_KEY": "", "PYTHONIOENCODING": "utf-8"}
    for args, expected in cases:
        try:
            p = subprocess.run(
                [sys.executable, "-m", "deepthink.cli", *args],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            checks.append(Check("cli", " ".join(args)[:80], "FAIL", "timed out"))
            continue
        crashed = "Traceback (most recent call last)" in p.stderr
        ok = p.returncode in expected and not crashed
        checks.append(
            Check(
                "cli",
                " ".join(args).replace(str(tmp), "<tmp>")[:80],
                "OK" if ok else "FAIL",
                "" if ok else f"exit {p.returncode}: {p.stderr.strip()[-200:]}",
            )
        )
    if progress:
        progress("  cli: done")
    return checks
