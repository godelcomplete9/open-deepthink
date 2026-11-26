"""Theory Synthesis tests (keyless).

Answer canonicalization, the swarm vote and entropic mirror descent, the shared
body of work (merge, conflicts, arbitration, taint), the descent/ascent
mechanics, concurrency and budget limits, the cost bound, and the CLI, server
and UI wiring. Runs with in-process fake chat models or the CoderMockLLM.
"""

import asyncio
import json
import math
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.runnables import Runnable

results = []


def chk(name, fn):
    try:
        fn()
        results.append((name, "OK", None))
    except AssertionError as e:
        results.append((name, "FAIL", f"AssertionError: {e}"))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def text_of(input_data) -> str:
    """The rendered prompt as plain text (ChatPromptValue → its message contents)."""
    msgs = getattr(input_data, "messages", None)
    if msgs:
        return "\n".join(str(getattr(m, "content", m)) for m in msgs)
    if hasattr(input_data, "to_string"):
        return input_data.to_string()
    return str(input_data)


class Fake(Runnable):
    """Scriptable chat model: ``handler(kind, text, state) -> str``."""

    def __init__(self, handler, delay: float = 0.0):
        self.handler = handler
        self.delay = delay
        self.state = {"calls": [], "inflight": 0, "peak": 0, "spans": 0}

    def invoke(self, input_data, config=None, **kwargs):
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(input_data, config))

    async def ainvoke(self, input_data, config=None, **kwargs):
        text = text_of(input_data)
        low = text.lower()
        kind = classify(low)
        st = self.state
        st["inflight"] += 1
        st["peak"] = max(st["peak"], st["inflight"])
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            st["calls"].append((kind, time.perf_counter(), text))
            return self.handler(kind, text, st)
        finally:
            st["inflight"] -= 1


def classify(low: str) -> str:
    table = [
        ("you size a theory synthesis swarm member", "sizer"),
        ("you are the theory seed generator", "seeds"),
        ("you are the theory brief writer", "brief"),
        ("you are the qnn node generator for a theory synthesis member", "spanner"),
        ("you are the cell decomposer of a theory synthesis qnn", "decompose"),
        ("you are a theory synthesis solver", "solve"),
        ("you are a theory synthesis synthesizer", "synthesize"),
        ("you are an independent verifier for theory synthesis", "verify"),
        ("you are the theory arbiter", "arbitrate"),
        ("you are the persona evolver for a theory synthesis member", "evolve"),
        ("you are the theory reframer", "reframe"),
        ("you write the final theory synthesis report", "report"),
    ]
    for marker, kind in table:
        if marker in low:
            return kind
    return "other"


def spanner_reply(text: str, st: dict) -> str:
    ids = list(dict.fromkeys(re.findall(r"agent_\d+_\d+", text.split("Agent ids to fill", 1)[-1])))
    m = st["spans"]
    st["spans"] += 1
    return json.dumps(
        {
            "agents": [
                {
                    "id": aid,
                    "name": f"Fake M{m} {aid}",
                    "specialty": "fake specialist in algebra and counting",
                    "skills": ["algebra", "counting", "induction", "bounds"],
                    "system_prompt": "You are a fake agent for tests. Justify every step.",
                }
                for aid in ids
            ]
        }
    )


def member_of(text: str) -> int:
    m = re.search(r"YOU ARE FAKE M(\d+)", text)
    return int(m.group(1)) if m else -1


def cell_id(text: str) -> str:
    m = re.search(r"\(id ([\w.]+)", text)
    return m.group(1) if m else ""


def base_reply(kind: str, text: str, st: dict) -> str:
    if kind == "sizer":
        return json.dumps({"domain": "math", "answer_type": "number"})
    if kind == "seeds":
        return "bound induct telescope lattice residue parity symmetry invariant extremum iterate"
    if kind == "spanner":
        return spanner_reply(text, st)
    if kind == "verify":
        return json.dumps(
            {"score": 0.9, "verdict": "valid", "issues": [], "confirmed": [], "refuted": []}
        )
    if kind == "arbitrate":
        return json.dumps({"verdict": "A", "reason": "A checks out."})
    if kind == "evolve":
        return json.dumps({"agents": []})
    if kind == "reframe":
        return json.dumps({"framings": ["Work modulo a small prime first."]})
    if kind == "report":
        return "## Answer\nx"
    return json.dumps({"result": "ok", "claim": "ok", "final_answer": "1", "confidence": 0.8})


def run(llm, problem="Find x.", params=None, **kw):
    from deepthink.theory import run_theory_pipeline

    p = {"seed": 7, "qnn_mode": "manual", "manual_layers": 3, "manual_width": 2, **(params or {})}
    logs = []
    events = []
    res = asyncio.run(
        run_theory_pipeline(llm, problem, p, log=logs.append, on_event=events.append, **kw)
    )
    return res, logs, events


# ── 1. Answers ──────────────────────────────────────────────────────────
def t1():
    from deepthink.theory.answers import canonical_answer as c
    from deepthink.theory.answers import same_answer

    same = [
        (r"\boxed{\frac{1}{2}}", "0.5", "1/2", r"$\dfrac{2}{4}$"),
        ("{3, 1, 2}", r"\{1,2,3\}", "n = 1, 2, 3", "n = 1 or n = 2 or n = 3"),
        ("Yes", "true", r"\text{Yes}"),
        ("2^{10}", "1024", "1,024"),
        (r"O(n \log n)", r"\Theta(n\log n)", "O(n log n)"),
        (r"\frac{\pi}{4}", "pi/4"),
        (r"\sqrt{2}", "√2"),
        ("The answer is 2026.", "2026", "x = 2026"),
        ("−7", "-7"),
        ("none", r"\emptyset", "no solutions"),
        ("Q.E.D.", "proof"),
    ]
    for group in same:
        canon = {c(x) for x in group}
        assert len(canon) == 1, (group, canon)
    assert c("1/2") != c("1/3")
    assert c("(1, 2)") != c("(2, 1)")  # tuples keep order
    assert c("{1, 2}") == c("{2, 1}")  # sets do not
    assert c("") == "" and c("   ") == ""
    assert c("anything at all", answer_type="proof") == "proof"
    assert c("counterexample n = 4", answer_type="proof") == "disproof"
    assert same_answer(c("9.81 m/s^2"), c("9.80 m/s^2"), 1e-2)
    assert not same_answer(c("9.81 m/s^2"), c("9.80 m/s^2"), 1e-4)
    assert not same_answer(c("9.81 m/s^2"), c("9.81 m/s"), 1e-2)


chk("answers: canonical forms merge equal answers and keep different ones apart", t1)


def t2():
    from deepthink.theory.answers import extract_final_answer as x

    assert x(r"... so the total is \boxed{\{1,2\}}. Done.") == r"\{1,2\}"
    assert x(r"\boxed{\frac{a}{b}} then \boxed{7}") == "7"
    assert x("Some work.\n\nAnswer: 17\n") == "17"
    assert x("**Final answer:** $n=3$") == "$n=3$"
    assert x("no marker here\nlast line 5") == "last line 5"
    assert x("") == ""


chk("answers: final answer extraction (boxed, Answer: lines, last line)", t2)


# ── 2. Vote and mirror descent ──────────────────────────────────────────
def t3():
    import random

    from deepthink.theory.consensus import Ballot, is_converged, tally

    rng = random.Random(0)
    answers = ["1"] * 47 + ["2"] * 31 + ["3"] * 22
    rng.shuffle(answers)
    ballots = [Ballot(member=i, answer=a, weight=0.01) for i, a in enumerate(answers)]
    for mode in ("plurality", "weighted"):
        clusters = tally(ballots, mode=mode)
        assert clusters[0].key == "1" and clusters[0].count == 47, (mode, clusters[0])
        assert abs(clusters[0].support - 0.47) < 1e-9
    assert not is_converged(tally(ballots), 0.5)
    assert is_converged(tally(ballots), 0.45)
    # Abstentions keep their weight in the denominator: 2 of 10 agreeing is 20 % support.
    few = [Ballot(member=i, answer="9" if i < 2 else "", weight=0.1) for i in range(10)]
    cl = tally(few)
    assert len(cl) == 1 and abs(cl[0].support - 0.2) < 1e-9 and not is_converged(cl, 0.5)
    # Weighted mode: quality breaks a count tie, plurality ignores it.
    tie = [
        Ballot(member=0, answer="a", weight=0.25, quality=0.9),
        Ballot(member=1, answer="a", weight=0.25, quality=0.9),
        Ballot(member=2, answer="b", weight=0.25, quality=0.2),
        Ballot(member=3, answer="b", weight=0.25, quality=0.2),
    ]
    assert tally(tie)[0].key == "a"
    assert tally(tie, mode="plurality")[0].support == 0.5


chk("vote: the most common of 100 proposals wins; abstentions count against support", t3)


def t4():
    from deepthink.theory.consensus import mirror_descent_step

    w = [0.25, 0.25, 0.25, 0.25]
    losses = [0.0, 0.5, 1.0, 0.2]
    assert mirror_descent_step(w, losses, 0.0) == w
    new = mirror_descent_step(w, losses, 2.0)
    assert abs(sum(new) - 1.0) < 1e-12 and all(x > 0 for x in new)
    order = sorted(range(4), key=lambda i: -new[i])
    assert order == [0, 3, 1, 2], new
    # closed form of the entropic mirror step
    z = sum(wi * math.exp(-2.0 * li) for wi, li in zip(w, losses))
    for wi, li, ni in zip(w, losses, new):
        assert abs(ni - wi * math.exp(-2.0 * li) / z) < 1e-12
    assert mirror_descent_step([0.0, 0.0], [1.0, 1.0], 1.0) == [0.5, 0.5]


chk("mirror descent: exponentiated-gradient step on the simplex (closed form)", t4)


def t5():
    from deepthink.theory.consensus import Ballot, member_losses, mirror_descent_step, tally

    ballots = [
        Ballot(member=i, answer=a, weight=0.1, quality=1.0) for i, a in enumerate("AAAABBBCCD")
    ]
    prev = 0.0
    for _ in range(8):
        cl = tally(ballots)
        assert cl[0].key == "A"
        assert cl[0].support >= prev - 1e-12
        prev = cl[0].support
        w = mirror_descent_step([b.weight for b in ballots], member_losses(ballots, cl), 1.5)
        for b, x in zip(ballots, w):
            b.weight = x
    assert prev > 0.75, prev


chk("convergence: leader support rises monotonically under mirror descent", t5)


# ── 3. The body of work ────────────────────────────────────────────────
def t6():
    from deepthink.theory.ledger import TheoryLedger

    L = TheoryLedger(tolerance=1e-9)
    stmt = "Compute the number of domino tilings of a 2 by 10 board."
    a, c1 = L.add(member=0, round_=1, cell="r.0", kind="computation", statement=stmt, answer="89")
    b, c2 = L.add(
        member=1, round_=1, cell="r.1", kind="computation", statement=stmt + " ", answer="89"
    )
    assert a.id == b.id and a.status == "corroborated" and a.support == 2 and not c1 and not c2
    wrong, conflicts = L.add(
        member=2, round_=1, cell="r.0", kind="computation", statement=stmt, answer="144"
    )
    assert len(conflicts) == 1 and wrong.status == "disputed" and a.status == "disputed"
    built, _ = L.add(
        member=2,
        round_=1,
        cell="r.1",
        kind="lemma",
        statement="Square the tiling count.",
        answer="20736",
        depends_on=[wrong.id],
    )
    prop, _ = L.add(
        member=2,
        round_=1,
        cell="r",
        kind="proposal",
        statement="the problem",
        answer="20736",
        depends_on=[built.id],
        proposal=True,
    )
    tainted = L.resolve(conflicts[0].id, "b" if conflicts[0].b == a.id else "a", "F(11) = 89")
    assert a.status == "verified" and wrong.status == "refuted"
    assert built.status == "suspect" and prop.status == "suspect", (built.status, prop.status)
    assert set(tainted) >= {wrong.id, built.id, prop.id}
    assert L.quality(prop.id) == 0.5 and L.quality(a.id) == 1.0
    # agents never see proposals; refuted entries show only when close
    hits = [e.id for e, _ in L.retrieve("number of tilings of the board", member=5)]
    assert prop.id not in hits and a.id in hits
    block = L.context_block(L.retrieve("tilings board", member=5))
    assert "refuted" in block and "Body of work" in block
    # reuse needs a usable entry about the same thing
    assert L.usable(a.id, 3, statement="How many domino tilings does a 2 by 10 board have?")
    assert L.usable(a.id, 3, statement="Prove the triangle inequality.") is None
    assert L.usable(wrong.id, 3) is None and L.usable(prop.id, 3) is None
    # private mode: a QNN reads only its own entries
    P = TheoryLedger(shared=False)
    P.add(
        member=0,
        round_=1,
        cell="r.0",
        kind="lemma",
        statement="Bound the sum by n squared.",
        answer="proof",
    )
    assert P.retrieve("bound the sum", member=1) == [] and P.retrieve("bound the sum", member=0)
    # declared contradictions open conflicts too
    Q = TheoryLedger()
    x, _ = Q.add(
        member=0, round_=1, cell="r.0", kind="lemma", statement="f is injective", answer="proof"
    )
    y, cs = Q.add(
        member=1,
        round_=1,
        cell="r.0",
        kind="lemma",
        statement="g has a fixed point",
        answer="proof",
        contradicts=[x.id],
    )
    assert len(cs) == 1 and x.status == "disputed"
    Q.resolve(cs[0].id, "both", "different claims")
    assert x.status == "proposed" and y.status == "proposed"
    st = L.stats()
    assert st["refuted"] == 1 and st["conflicts_settled"] == 1 and st["proposals"] == 1
    json.dumps(L.to_dict())


chk("body of work: merge, corroborate, conflict, arbitrate, taint, retrieve, reuse guard", t6)


# ── 4. Descent and ascent ───────────────────────────────────────────────
def t7():
    def handler(kind, text, st):
        if kind == "decompose":
            cid = cell_id(text)
            if cid == "r":
                return json.dumps(
                    {
                        "easiness": 0.2,
                        "best_solver": "agent_0_1",
                        "plan": "three parts",
                        "subcells": [
                            {"statement": "Part zero: count the base cases.", "easiness": 0.9},
                            {"statement": "Part one: bound the middle term.", "easiness": 0.4},
                            {
                                "statement": "Part two: combine both results.",
                                "easiness": 0.95,
                                "depends_on": [0, 1, 2, "x", 5],
                            },
                            {
                                "statement": "Part three: beyond the branching limit.",
                                "easiness": 0.9,
                            },
                        ],
                    }
                )
            if cid == "r.1":
                return json.dumps(
                    {
                        "easiness": 0.3,
                        "subcells": [
                            {
                                "statement": "Sub a: lower bound of the middle term.",
                                "easiness": 0.9,
                            },
                            {
                                "statement": "Sub b: upper bound of the middle term.",
                                "easiness": 0.9,
                            },
                        ],
                    }
                )
            return json.dumps({"easiness": 0.99, "atomic": True})
        if kind == "solve":
            return json.dumps(
                {"result": "done", "claim": cell_id(text), "final_answer": "1", "confidence": 0.9}
            )
        if kind == "synthesize":
            return json.dumps(
                {"result": "\\boxed{42}", "claim": "root", "final_answer": "42", "confidence": 0.9}
            )
        return base_reply(kind, text, st)

    fake = Fake(handler, delay=0.005)
    res, logs, events = run(
        fake, params={"swarm_size": 1, "rounds": 1, "branching": 3, "max_depth": 2}
    )
    tree = {c["id"]: c for c in res["members"][0]["tree"]}
    assert set(tree) == {"r", "r.0", "r.1", "r.2", "r.1.0", "r.1.1"}, sorted(tree)
    assert (
        tree["r"]["leaf"] is False and tree["r.1"]["leaf"] is False and tree["r.0"]["leaf"] is True
    )
    assert tree["r.2"]["depends_on"] == ["r.0", "r.1"], tree["r.2"]["depends_on"]
    # decomposer calls: root and r.1 only (r.0 and r.2 were easy enough; r.1.* at max depth)
    assert res["calls"]["by_kind"]["decompose"] == 2, res["calls"]
    # synthesizer layers follow tree height: r.1 (height 1) → layer 1, root (height 2) → layer 2
    assert tree["r.1"]["layer"] == 1 and tree["r"]["layer"] == 2, (tree["r.1"], tree["r"])
    # r.2 is solved only after r.0 and the whole r.1 subtree are done
    order = [(k, t, cell_id(x)) for k, t, x in fake.state["calls"] if k in ("solve", "synthesize")]
    when = {cid: t for k, t, cid in order}
    assert when["r.2"] > when["r.1"] and when["r.2"] > when["r.0"], when
    assert res["final_answer"] == "42" and res["converged"]
    assert tree["r"]["entry"] and tree["r.0"]["entry"]


chk("descent/ascent: easiness threshold, branching cap, depth cap, deps, synthesizer layers", t7)


def t8():
    def handler(kind, text, st):
        if kind == "decompose":
            return json.dumps(
                {
                    "easiness": 0.1,
                    "subcells": [
                        {"statement": f"piece {k} of {cell_id(text)}", "easiness": 0.1}
                        for k in range(3)
                    ],
                }
            )
        if kind in ("solve", "synthesize"):
            return json.dumps({"result": "\\boxed{7}", "final_answer": "7", "confidence": 0.9})
        return base_reply(kind, text, st)

    fake = Fake(handler, delay=0.01)
    res, _, _ = run(fake, params={"swarm_size": 4, "rounds": 1, "concurrency": 3, "max_cells": 7})
    assert fake.state["peak"] <= 3, fake.state["peak"]
    assert res["calls"]["peak_concurrency"] <= 3
    for m in res["members"]:
        assert len(m["tree"]) <= 7
    # a hard budget ends the run gracefully
    fake2 = Fake(handler)
    res2, logs2, _ = run(fake2, params={"swarm_size": 3, "rounds": 2, "max_calls": 12})
    assert res2["calls"]["total"] <= 12 and res2["calls"]["budget_exhausted"]
    assert any("FINAL_ANSWER" in line for line in logs2)


chk("harness: concurrency cap and call budget are respected", t8)


def t9():
    from deepthink.cost import estimate_theory_cost, theory_tree_bound

    # Worst case: every cell is hard, every QNN answers differently, every cell conflicts.
    def handler(kind, text, st):
        m = member_of(text)
        if kind == "decompose":
            return json.dumps(
                {
                    "easiness": 0.0,
                    "subcells": [
                        {"statement": f"Shared step {k} at {cell_id(text)}", "easiness": 0.0}
                        for k in range(2)
                    ],
                }
            )
        if kind == "solve":
            return json.dumps({"result": "x", "final_answer": str(100 + m), "confidence": 0.5})
        if kind == "synthesize":
            return json.dumps({"result": "y", "final_answer": str(1000 + m), "confidence": 0.5})
        if kind == "arbitrate":
            return json.dumps({"verdict": "both", "reason": "different"})
        return base_reply(kind, text, st)

    for S, R, D, b, N in ((3, 3, 2, 2, 7), (4, 2, 3, 3, 10), (2, 3, 1, 4, 5)):
        fake = Fake(handler)
        res, _, _ = run(
            fake,
            params={
                "swarm_size": S,
                "rounds": R,
                "max_depth": D,
                "branching": b,
                "max_cells": N,
                "quorum": 1.0,
            },
        )
        est = estimate_theory_cost(
            S, R, layers=3, width=2, max_depth=D, branching=b, max_cells=N, auto_size=True
        )
        assert res["calls"]["total"] <= est.llm_calls, (S, R, D, b, N, res["calls"], est.llm_calls)
        assert res["body_of_work"]["stats"]["conflicts_settled"] > 0
    assert theory_tree_bound(0, 3, 10) == {"decompose": 0, "cells": 1, "total": 1}
    assert theory_tree_bound(2, 3, 10) == {"decompose": 4, "cells": 10, "total": 14}


chk("cost: the estimate bounds worst-case runs (hard cells, dissent, conflicts)", t9)


# ── 5. Conflict → arbitration → taint → re-run → convergence ───────────
def t10():
    def handler(kind, text, st):
        m = member_of(text)
        framed = "suggested framing" in text.lower()
        if kind == "decompose":
            if cell_id(text) == "r":
                return json.dumps(
                    {
                        "easiness": 0.2,
                        "subcells": [
                            {"statement": "Compute X.", "kind": "computation", "easiness": 0.9},
                            {"statement": "Compute Y.", "kind": "computation", "easiness": 0.9},
                        ],
                    }
                )
            return json.dumps({"easiness": 0.9, "atomic": True})
        if kind == "solve":
            own = text.split("YOUR CELL", 1)[-1]
            if "Compute X." in own:
                ans = "6" if (m == 1 and not framed) else "5"
            else:
                ans = "10"
            return json.dumps(
                {
                    "result": f"value {ans}",
                    "claim": f"value {ans}",
                    "final_answer": ans,
                    "confidence": 0.8,
                }
            )
        if kind == "synthesize":
            ans = "60" if (m == 1 and not framed) else "50"
            return json.dumps(
                {"result": f"\\boxed{{{ans}}}", "final_answer": ans, "confidence": 0.8}
            )
        if kind == "arbitrate":
            a_part = text.split("ENTRY A", 1)[-1].split("ENTRY B", 1)[0]
            return json.dumps(
                {
                    "verdict": "A" if "Answer: 5" in a_part else "B",
                    "reason": "X = 5 by direct count.",
                }
            )
        return base_reply(kind, text, st)

    fake = Fake(handler)
    res, logs, events = run(
        fake, params={"swarm_size": 2, "rounds": 3, "quorum": 0.75, "concurrency": 1}
    )
    bow = res["body_of_work"]
    st = bow["stats"]
    assert st["conflicts_settled"] >= 1 and st["refuted"] >= 1, st
    refuted = [e for e in bow["entries"] if e["status"] == "refuted"]
    assert refuted and refuted[0]["answer"] == "6", refuted
    suspect_props = [e for e in bow["entries"] if e["proposal"] and e["status"] == "suspect"]
    assert suspect_props and suspect_props[0]["answer"] == "60", [
        e for e in bow["entries"] if e["proposal"]
    ]
    assert res["converged"] and res["final_answer"] == "50" and res["rounds_run"] == 2, (
        res["converged"],
        res["final_answer"],
        res["rounds_run"],
    )
    m1 = res["members"][1]
    assert m1["history"][0]["loss"] > res["members"][0]["history"][0]["loss"]
    assert m1["framing"], "the dissenting QNN got a new framing"
    votes = [e for e in events if e["type"] == "vote"]
    assert votes[0]["converged"] is False and votes[-1]["converged"] is True
    xs = [e for e in bow["entries"] if e["statement"] == "Compute X." and e["status"] != "refuted"]
    assert xs and xs[0]["support"] == 2 and xs[0]["status"] == "verified", xs


chk("consistency: a refuted lemma taints its proposal, the QNN re-runs, the swarm converges", t10)


# ── 6. Full simulated run, params, wiring ──────────────────────────────
def t11():
    from deepthink.mocks import CoderMockLLM

    reused = 0
    for seed in (2, 4, 5):
        res, logs, events = run(
            CoderMockLLM(),
            "Let a_1 = 2 and a_{n+1} = a_n + 1. Find a_2025.",
            params={"swarm_size": 6, "rounds": 3, "seed": seed, "quorum": 0.9, "qnn_mode": "auto"},
        )
        assert res["mode"] == "theory" and res["final_answer"] == "2026" and res["converged"]
        reused += res["body_of_work"]["stats"]["reuses"]
        rep = res["proposed_solution"]
        for section in (
            "## Solution",
            "## How the swarm agreed",
            "## Body of work",
            "## Checks and caveats",
        ):
            assert section in rep, section
        final = [line for line in logs if line.startswith("FINAL_ANSWER:")]
        assert len(final) == 1 and json.loads(final[0][len("FINAL_ANSWER:") :])["mode"] == "theory"
        swarm = [
            json.loads(line[len("SWARM_EVENT:") :])
            for line in logs
            if line.startswith("SWARM_EVENT:")
        ]
        assert {e["type"] for e in swarm} >= {"init", "member", "ledger", "vote", "done"}
        assert swarm == events
        json.dumps(res)
    assert reused > 0, "dissenting QNNs reuse results from the body of work"


chk("simulation: full swarm run with CoderMockLLM (report, events, reuse, JSON-safe)", t11)


def t12():
    from deepthink.theory import clamp_theory_params, default_theory_params

    p = clamp_theory_params(
        {
            "swarm_size": 999,
            "num_epochs": 99,
            "max_depth": -3,
            "branching": 1,
            "quorum": 7,
            "vote": "bogus",
            "ledger": "PRIVATE",
            "verify": "false",
            "temperature_min": "",
            "seed": "11",
            "max_cells": 0,
        }
    )
    assert p["swarm_size"] == 128 and p["rounds"] == 10 and p["max_depth"] == 0
    assert p["branching"] == 2 and p["quorum"] == 1.0 and p["vote"] == "weighted"
    assert p["ledger"] == "private" and p["verify"] is False and p["temperature_min"] is None
    assert p["seed"] == 11 and p["max_cells"] == 1
    d = default_theory_params()
    assert d["vote"] == "weighted" and d["ledger"] == "shared" and d["swarm_size"] == 5
    assert clamp_theory_params({"rounds": 4, "num_epochs": 9})["rounds"] == 4


chk("params: clamps, aliases and defaults", t12)


def t13():
    from deepthink import cli

    help_txt = cli.build_parser().format_help()
    assert "theory" in help_txt
    env_root = str(ROOT)
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "deepthink.cli",
            "theory",
            "--debug",
            "-p",
            "Find a_2025 if a_n = n + 1.",
            "--swarm",
            "3",
            "--rounds",
            "2",
            "--seed",
            "4",
        ],
        capture_output=True,
        text=True,
        cwd=env_root,
        encoding="utf-8",
        errors="replace",
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "Theory Synthesis" in out.stdout and "Answer:" in out.stderr
    big = subprocess.run(
        [
            sys.executable,
            "-m",
            "deepthink.cli",
            "theory",
            "-p",
            "x",
            "--swarm",
            "50",
            "--api-key",
            "k",
        ],
        capture_output=True,
        text=True,
        cwd=env_root,
        encoding="utf-8",
        errors="replace",
    )
    assert big.returncode == 2 and "Refusing" in big.stderr
    est = subprocess.run(
        [sys.executable, "-m", "deepthink.cli", "estimate", "theory", "--swarm", "5"],
        capture_output=True,
        text=True,
        cwd=env_root,
        encoding="utf-8",
        errors="replace",
    )
    assert est.returncode == 0 and est.stdout.startswith("theory:"), est.stdout


chk("CLI: theory runs in debug mode, refuses big runs, estimate theory", t13)


def t14():
    from app import app, sessions
    from fastapi.testclient import TestClient

    payload = {
        "mode": "theory",
        "params": {
            "prompt": "Let a_1 = 2 and a_{n+1} = a_n + 1. Find a_2025.",
            "coder_debug_mode": "true",
            "swarm_size": 3,
            "rounds": 2,
            "seed": 4,
        },
    }
    # The context manager keeps one event loop alive, so the background swarm can finish.
    with TestClient(app) as client:
        r = client.post("/build_and_run_graph", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["cost_estimate"]["kind"] == "theory" and body["session_id"] in sessions
        sid = body["session_id"]
        assert sessions[sid]["mode"] == "theory" and sessions[sid]["params"]["swarm_size"] == 3
        for _ in range(150):
            time.sleep(0.2)
            fs = sessions[sid].get("final_solution")
            if isinstance(fs, dict):
                break
        fs = sessions[sid].get("final_solution")
        assert isinstance(fs, dict), "the background swarm did not finish"
        assert fs["mode"] == "theory" and fs["final_answer"] in ("2025", "2026"), fs
        assert (
            fs["canonical_answer"] == fs["final_answer"]
            and "## Body of work" in fs["proposed_solution"]
        )
        assert sessions[sid]["theory"]["body_of_work"]["stats"]["entries"] >= 1
        exported = client.get(f"/export_qnn/{sid}")
        assert exported.status_code == 200 and exported.json()["mode"] == "theory"
        bad = client.post(
            "/build_and_run_graph", json={"mode": "nope", "params": {"coder_debug_mode": "true"}}
        )
        assert bad.status_code == 400 and "theory" in bad.json()["message"]


chk("server: /build_and_run_graph mode=theory (simulation) runs the swarm", t14)


def t15():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    assert 'data-mode="theory">∑ Theory Synthesis</button>' in html
    assert "Brainstorming Mode</button>" not in html
    for el_id in (
        "theory-mode",
        "theory_domain",
        "theory_swarm_size",
        "theory_rounds",
        "theory_quorum",
        "theory_max_depth",
        "theory_branching",
        "theory_easiness",
        "theory_max_cells",
        "theory_vote",
        "theory_lr",
        "theory_ledger",
        "theory_verify",
        "theory_concurrency",
        "theory_max_calls",
        "theory-swarm-panel",
        "theory-mem-calc",
        "theory-cost-estimate",
    ):
        assert f'id="{el_id}"' in html, el_id
    assert "js/components/theory-swarm.js" in html and "css/theory.css" in html
    assert "SWARM_EVENT:" in html and "mode: 'theory'" in html
    js = (ROOT / "js" / "components" / "theory-swarm.js").read_text(encoding="utf-8")
    assert "const TheorySwarm" in js and "handle" in js
    assert (ROOT / "css" / "theory.css").is_file()


chk("UI: Theory Synthesis tab, swarm controls and live swarm panel", t15)


ok = sum(1 for _, s, _ in results if s == "OK")
total = len(results)
for name, status, err in results:
    print(f"  [{status}] {name}" + (f" :: {err}" if err else ""))
print(f"\nPHASE THEORY: {ok}/{total} OK")
sys.exit(0 if ok == total else 1)
