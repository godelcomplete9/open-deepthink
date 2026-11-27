"""Theory Synthesis: a swarm of QNNs that builds one consistent body of work.

One member of the swarm is one QNN, modified for problem solving. It keeps the
QNN heuristic:

* **Seeds → personas.** Verbs and nouns from the problem's field and from far
  fields are sampled into per-column word-vectors; each column spans a persona
  per layer. Layer 0 agents are *solvers*; layers 1+ are *synthesizers*.
* **Descent.** The root cell (the problem) is split into a tree of cells until
  each cell is easy for one of the QNN's solvers (``easiness ≥ τ``), or the
  depth or cell budget runs out. A subcell the body of work already
  establishes is reused instead of solved again.
* **Ascent (the forward pass).** Leaves are solved in parallel by layer 0. Each
  tree height is collapsed by the next layer of synthesizers, so layer *k* sees
  layer *k−1*'s outputs. The root synthesis is the QNN's **proposal**.
* **Self-attention over the body of work.** Every agent reads the entries most
  relevant to its cell from the shared ledger, not just its own thought stream.
* **Mirror Descent.** QNNs that lose the vote rewrite their personas (scaled by
  their loss) or respawn from fresh seeds when their weight collapses.
* **Reframe.** Dissenting QNNs get a new representation of the same problem.

The swarm runs S members in parallel. Every cell lands in the shared
:class:`~deepthink.theory.ledger.TheoryLedger`, which merges repeated results,
detects contradictions, settles them by arbitration and taints whatever rests
on refuted results. Proposals are clustered by canonical answer and weighed by
entropic mirror descent (see ``consensus.py``). The run stops when the leading
answer's support reaches the quorum, or the rounds run out.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from deepthink.chains.theory_chains import (
    ROOT_RULE,
    SOLVER_PROMPT,
    SYNTHESIZER_PROMPT,
    get_theory_arbiter_chain,
    get_theory_brief_chain,
    get_theory_cell_chain,
    get_theory_decomposer_chain,
    get_theory_member_spanner_chain,
    get_theory_persona_evolver_chain,
    get_theory_reframer_chain,
    get_theory_report_chain,
    get_theory_seed_chain,
    get_theory_sizer_chain,
    get_theory_verifier_chain,
)
from deepthink.theory.answers import PROOF, canonical_answer, extract_final_answer
from deepthink.theory.consensus import (
    VOTE_MODES,
    Ballot,
    Cluster,
    entropy,
    is_converged,
    leader,
    member_losses,
    mirror_descent_step,
    tally,
)
from deepthink.theory.domains import domain_profile, guess_answer_type, resolve_domain
from deepthink.theory.ledger import TheoryLedger, tokens
from deepthink.utils import clean_and_parse_json, strip_secrets

LogFn = Callable[[str], Any] | None
EventFn = Callable[[dict[str, Any]], Any] | None

ANSWER_TYPES = (
    "number",
    "expression",
    "set",
    "tuple",
    "yesno",
    "interval",
    "formula",
    "complexity",
    "construction",
    "proof",
    "other",
)

_INT_LIMITS: dict[str, tuple[int, int]] = {
    "swarm_size": (1, 128),
    "rounds": (1, 10),
    "manual_layers": (1, 6),
    "manual_width": (1, 8),
    "max_depth": (0, 4),
    "branching": (2, 6),
    "max_cells": (1, 64),
    "vector_word_size": (2, 12),
    "ledger_top_k": (0, 16),
    "max_arbitrations": (0, 32),
    "concurrency": (1, 64),
    "max_calls": (0, 1_000_000),
}
_FLOAT_LIMITS: dict[str, tuple[float, float]] = {
    "quorum": (0.05, 1.0),
    "learning_rate": (0.0, 8.0),
    "easiness": (0.05, 1.0),
    "min_verify": (0.0, 1.0),
    "suspect_penalty": (0.0, 1.0),
    "respawn_floor": (0.0, 1.0),
}


def default_theory_params() -> dict[str, Any]:
    """Documented defaults for the UI, CLI and library."""
    return {
        "domain": "auto",  # auto | math | olympiad | algorithms | physics
        "answer_type": "auto",  # auto | proof | number | set | … (auto = sizer, then a guess)
        "swarm_size": 5,  # S: QNNs in the swarm
        "rounds": 2,  # R: swarm rounds (the QNN's epochs)
        "quorum": 0.5,  # leader support needed to stop early
        "vote": "weighted",  # weighted (mirror-descent weights × checks) | plurality
        "learning_rate": 1.0,  # η: weight step, and the scale of persona rewrites
        "qnn_mode": "auto",  # auto | manual: topology of each QNN
        "manual_layers": 3,  # L: layer 0 solvers + L−1 synthesizer layers
        "manual_width": 3,  # W: agents per layer
        "max_depth": 2,  # D: how many times a cell may be split
        "branching": 3,  # b: subcells per split, at most
        "easiness": 0.75,  # τ: a cell is a leaf once its solver rates it this easy
        "max_cells": 10,  # cells per QNN tree, root included
        "vector_word_size": 4,  # V: guiding words per column
        "verify": True,  # independent check of every proposal
        "min_verify": 0.2,  # proposals scoring below this abstain
        "ledger": "shared",  # shared | private body of work
        "ledger_top_k": 6,  # body-of-work entries shown to each agent
        "max_arbitrations": 4,  # conflicts settled per round (one call each)
        "suspect_penalty": 0.5,  # vote multiplier for proposals built on refuted results
        "respawn_floor": 0.35,  # respawn a QNN whose weight falls below this × 1/S
        "concurrency": 8,  # simultaneous LLM calls
        "max_calls": 0,  # hard call budget (0 = none)
        "temperature_min": None,  # optional temperature spread across QNNs
        "temperature_max": None,
        "tolerance": None,  # relative tolerance for numeric votes (None = domain default)
        "polish": False,  # one extra call to write up the report
        "seed": None,  # RNG seed for reproducible seed sampling
    }


def _num(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default  # NaN guard


def clamp_theory_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Merge with defaults, accept aliases, and bound every knob."""
    raw = dict(params or {})
    aliases = {
        "num_epochs": "rounds",
        "epochs": "rounds",
        "swarm": "swarm_size",
        "depth": "max_depth",
    }
    for alias, key in aliases.items():
        if alias in raw and key not in raw:
            raw[key] = raw[alias]
    # Request params from the web UI carry credentials; they never belong in a run record.
    p = strip_secrets({**default_theory_params(), **raw})
    for key, (lo, hi) in _INT_LIMITS.items():
        p[key] = int(max(lo, min(hi, int(_num(p.get(key), default_theory_params()[key] or 0)))))
    for key, (flo, fhi) in _FLOAT_LIMITS.items():
        p[key] = float(max(flo, min(fhi, _num(p.get(key), default_theory_params()[key]))))
    p["max_cells"] = max(1, p["max_cells"])
    p["qnn_mode"] = "manual" if str(p.get("qnn_mode", "auto")).lower() == "manual" else "auto"
    p["vote"] = str(p.get("vote", "weighted")).lower()
    if p["vote"] not in VOTE_MODES:
        p["vote"] = "weighted"
    p["ledger"] = "private" if str(p.get("ledger", "shared")).lower() == "private" else "shared"
    for key in ("verify", "polish"):
        val = p.get(key)
        p[key] = (
            val.strip().lower() in {"1", "true", "yes", "on"} if isinstance(val, str) else bool(val)
        )
    for key in ("temperature_min", "temperature_max"):
        val = p.get(key)
        p[key] = None if val in (None, "") else max(0.0, min(2.0, _num(val, 0.7)))
    tol = p.get("tolerance")
    p["tolerance"] = None if tol in (None, "") else max(0.0, min(0.5, _num(tol, 0.0)))
    seed = p.get("seed")
    p["seed"] = None if seed in (None, "") else int(_num(seed, 0))
    p["domain"] = str(p.get("domain") or "auto").lower()
    p["answer_type"] = str(p.get("answer_type") or "auto").lower()
    return p


# ── Data ────────────────────────────────────────────────────────────────


@dataclass
class Cell:
    id: str
    statement: str
    depth: int
    parent: str | None = None
    kind: str = "root"
    index: int = 0
    depends_on: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    leaf: bool = True
    easiness: float | None = None
    solver: str = ""
    agent: str = ""
    layer: int = 0
    height: int = 0
    plan: str = ""
    reuse: str = ""
    result: str = ""
    claim: str = ""
    final_answer: str = ""
    confidence: float = 0.0
    checks: str = ""
    gaps: list[str] = field(default_factory=list)
    uses: list[str] = field(default_factory=list)
    attended: list[str] = field(default_factory=list)
    entry: str = ""
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["result"] = _excerpt(self.result, 3000)
        return d


@dataclass
class Member:
    index: int
    columns: list[str]
    layers: int
    width: int
    personas: dict[str, dict[str, Any]] = field(default_factory=dict)
    temperature: float | None = None
    model: str = ""
    weight: float = 0.0
    generation: int = 0
    framing: str = ""
    cells: dict[str, Cell] = field(default_factory=dict)
    proposal: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    llm: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "columns": self.columns,
            "layers": self.layers,
            "width": self.width,
            "personas": self.personas,
            "temperature": self.temperature,
            "model": self.model,
            "weight": round(self.weight, 6),
            "generation": self.generation,
            "framing": self.framing,
            "proposal": self.proposal,
            "history": self.history,
            "tree": [c.to_dict() for c in self.cells.values()],
        }


@dataclass
class TheoryResult:
    mode: str = "theory"
    proposed_solution: str = ""
    final_answer: str = ""
    canonical_answer: str = ""
    converged: bool = False
    support: float = 0.0
    rounds_run: int = 0
    domain: str = ""
    answer_type: str = ""
    winner: dict[str, Any] = field(default_factory=dict)
    vote: list[dict[str, Any]] = field(default_factory=list)
    vote_history: list[dict[str, Any]] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    members: list[dict[str, Any]] = field(default_factory=list)
    body_of_work: dict[str, Any] = field(default_factory=dict)
    topology: dict[str, Any] = field(default_factory=dict)
    calls: dict[str, Any] = field(default_factory=dict)
    final_solution: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _BudgetExhausted(Exception):
    """Raised before a call once ``max_calls`` is spent."""


def _excerpt(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _clamp01(value: Any, default: float | None = 0.0) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:
        return default
    return max(0.0, min(1.0, out))


def _str_list(value: Any, limit: int = 12) -> list[str]:
    if isinstance(value, str):
        value = [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:limit]


# ── Run context ─────────────────────────────────────────────────────────


class _Run:
    def __init__(
        self,
        *,
        llm: Any,
        synth: Any,
        synth_override: Any,
        params: dict[str, Any],
        log: LogFn,
        on_event: EventFn,
        rng: random.Random,
    ) -> None:
        self.llm = llm
        self.synth = synth
        self.synth_override = synth_override
        self.p = params
        self.log_fn = log
        self.on_event = on_event
        self.rng = rng
        self.sem = asyncio.Semaphore(params["concurrency"])
        self.calls = 0
        self.call_kinds: dict[str, int] = {}
        self.max_calls = int(params["max_calls"])
        self.exhausted = False
        self.inflight = 0
        self.peak_inflight = 0
        self._chains: dict[tuple[str, int], Any] = {}
        # set after sizing
        self.problem = ""
        self.domain = "math"
        self.profile: dict[str, Any] = domain_profile("math")
        self.answer_type = "other"
        self.tolerance = 0.0
        self.ledger = TheoryLedger()
        self.verbose = True
        self.seed_pool: list[str] = []

    async def log(self, msg: str) -> None:
        if self.log_fn is None:
            return
        out = self.log_fn(msg)
        if hasattr(out, "__await__"):
            await out

    async def emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            out = self.on_event(event)
            if hasattr(out, "__await__"):
                await out
        await self.log("SWARM_EVENT: " + json.dumps(event, ensure_ascii=False, default=str))

    def chain(self, name: str, llm: Any) -> Any:
        key = (name, id(llm))
        if key not in self._chains:
            factory = {
                "brief": get_theory_brief_chain,
                "sizer": get_theory_sizer_chain,
                "seeds": get_theory_seed_chain,
                "spanner": get_theory_member_spanner_chain,
                "decomposer": get_theory_decomposer_chain,
                "cell": get_theory_cell_chain,
                "verifier": get_theory_verifier_chain,
                "arbiter": get_theory_arbiter_chain,
                "evolver": get_theory_persona_evolver_chain,
                "reframer": get_theory_reframer_chain,
                "report": get_theory_report_chain,
            }[name]
            self._chains[key] = factory(llm)
        return self._chains[key]

    async def ask(self, name: str, llm: Any, inputs: dict[str, Any], kind: str = "") -> str:
        if self.max_calls and self.calls >= self.max_calls:
            self.exhausted = True
            raise _BudgetExhausted(kind or name)
        self.calls += 1
        label = kind or name
        self.call_kinds[label] = self.call_kinds.get(label, 0) + 1
        chain = self.chain(name, llm)
        async with self.sem:
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)
            try:
                out = await chain.ainvoke(inputs)
            finally:
                self.inflight -= 1
        return out if isinstance(out, str) else str(getattr(out, "content", out))

    def framing_block(self, member: Member) -> str:
        if not member.framing:
            return ""
        return (
            "\nSuggested framing for this attempt (a representation of the SAME problem):\n"
            f"{member.framing}\n"
        )


# ── Personas ────────────────────────────────────────────────────────────


def _fallback_persona(
    aid: str, layer: int, col: int, words: str, profile: dict[str, Any], rng: random.Random
) -> dict[str, Any]:
    archetypes = profile.get("archetypes") or ["Mathematician"]
    arche = archetypes[(col + 2 * layer) % len(archetypes)]
    pool = list(profile.get("skills") or ["case analysis"])
    skills = rng.sample(pool, k=min(4, len(pool)))
    role = (
        "You solve small cells directly and justify every step."
        if layer == 0
        else "You collapse solved subcells into a complete solution of their parent cell, "
        "checking every case and every link."
    )
    return {
        "id": aid,
        "name": f"{arche} {col + 1}.{layer}",
        "specialty": f"{arche} shaped by {words}",
        "skills": skills,
        "guiding_words": words,
        "system_prompt": f"You are a {arche.lower()} whose thinking is shaped by: {words}. {role}",
    }


def _normalize_persona(
    raw: dict[str, Any], aid: str, layer: int, col: int, words: str, fallback: dict[str, Any]
) -> dict[str, Any]:
    skills = _str_list(raw.get("skills"), 8) or fallback["skills"]
    prompt = str(raw.get("system_prompt") or "").strip()
    return {
        "id": aid,
        "name": str(raw.get("name") or fallback["name"])[:80],
        "specialty": str(raw.get("specialty") or fallback["specialty"])[:160],
        "skills": skills,
        "guiding_words": words,
        "layer": layer,
        "column": col,
        "role": "solver" if layer == 0 else "synthesizer",
        "system_prompt": prompt[:2400] if len(prompt) > 30 else fallback["system_prompt"],
    }


def _persona_text(persona: dict[str, Any]) -> str:
    skills = ", ".join(persona.get("skills") or [])
    return (
        f"YOU ARE {str(persona.get('name', 'Agent')).upper()}, "
        f"{str(persona.get('specialty', 'a specialist'))}.\n"
        f"{persona.get('system_prompt', '')}\nYour skills: {skills}."
    )


def _agent_ids(member: Member) -> list[str]:
    return [f"agent_{i}_{j}" for i in range(member.layers) for j in range(member.width)]


async def _span_member(run: _Run, member: Member) -> None:
    ids = _agent_ids(member)
    columns = "\n".join(f"- column {j}: {w}" for j, w in enumerate(member.columns))
    agents: list[Any] = []
    try:
        raw = await run.ask(
            "spanner",
            member.llm,
            {
                "layers": member.layers,
                "width": member.width,
                "domain_label": run.profile["label"],
                "problem": run.problem[:12000],
                "skill_examples": run.profile["skill_examples"],
                "columns": columns,
                "agent_ids": ", ".join(ids),
            },
        )
        data = clean_and_parse_json(raw)
        if isinstance(data, dict):
            agents = data.get("agents") or []
        elif isinstance(data, list):
            agents = data
    except _BudgetExhausted:
        agents = []
    except Exception as e:  # never crash a member on a bad persona payload
        await run.log(
            f"WARNING: [THEORY Q{member.index}] persona spanning failed ({e}); templates."
        )
    by_id = {str(a.get("id")): a for a in agents if isinstance(a, dict) and str(a.get("id")) in ids}
    personas: dict[str, dict[str, Any]] = {}
    for aid in ids:
        layer, col = (int(x) for x in aid.split("_")[1:])
        words = member.columns[col % len(member.columns)] if member.columns else ""
        fb = _fallback_persona(aid, layer, col, words, run.profile, run.rng)
        personas[aid] = _normalize_persona(by_id.get(aid, {}), aid, layer, col, words, fb)
    member.personas = personas


def _match_agent(member: Member, layer: int, text: str, salt: str = "") -> str:
    """Lexical skill match between a cell and the agents of one layer (QNN column choice)."""
    layer = max(0, min(layer, member.layers - 1))
    ids = [f"agent_{layer}_{j}" for j in range(member.width)]
    want = tokens(text)
    offset = sum(ord(ch) for ch in salt) % max(1, len(ids))
    best, best_score = ids[offset], -1.0
    for k in range(len(ids)):
        aid = ids[(offset + k) % len(ids)]
        p = member.personas.get(aid) or {}
        have = tokens(" ".join([str(p.get("specialty", "")), " ".join(p.get("skills") or [])]))
        score = float(len(want & have))
        if score > best_score:
            best, best_score = aid, score
    return best


# ── Descent ─────────────────────────────────────────────────────────────


def _path_text(cells: dict[str, Cell], cell: Cell) -> str:
    chain: list[Cell] = []
    cur: Cell | None = cell
    while cur is not None:
        chain.append(cur)
        cur = cells.get(cur.parent) if cur.parent else None
    chain.reverse()
    lines = [f"{'  ' * c.depth}- {c.id} ({c.kind}): {_excerpt(c.statement, 260)}" for c in chain]
    return "\n".join(lines)


def _solver_cards(member: Member) -> str:
    lines = []
    for j in range(member.width):
        p = member.personas.get(f"agent_0_{j}") or {}
        lines.append(
            f"- agent_0_{j}: {p.get('specialty', 'solver')}; skills: "
            + ", ".join(p.get("skills") or [])
        )
    return "\n".join(lines)


def _as_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


async def _decompose_cell(run: _Run, member: Member, cells: dict[str, Cell], cell: Cell) -> dict:
    hits = run.ledger.retrieve(cell.statement, member=member.index, k=run.p["ledger_top_k"])
    cell.attended = [e.id for e, _ in hits]
    raw = await run.ask(
        "decomposer",
        member.llm,
        {
            "domain_label": run.profile["label"],
            "domain_brief": run.profile["brief"],
            "problem": run.problem[:12000],
            "framing": run.framing_block(member),
            "path": _path_text(cells, cell),
            "cell_id": cell.id,
            "depth": cell.depth,
            "max_depth": run.p["max_depth"],
            "statement": cell.statement[:6000],
            "solvers": _solver_cards(member),
            "ledger": run.ledger.context_block(hits) or "(The body of work is empty so far.)",
            "threshold": run.p["easiness"],
            "branching": run.p["branching"],
        },
        "decompose",
    )
    data = clean_and_parse_json(raw)
    return data if isinstance(data, dict) else {}


async def _descend(run: _Run, member: Member) -> dict[str, Cell]:
    root = Cell(id="r", statement=run.problem, depth=0, kind="root")
    cells: dict[str, Cell] = {"r": root}
    tau, max_depth = run.p["easiness"], run.p["max_depth"]
    branching, max_cells = run.p["branching"], run.p["max_cells"]
    valid_solvers = {f"agent_0_{j}" for j in range(member.width)}
    frontier: list[Cell] = [root] if max_depth > 0 and max_cells > 1 else []
    if not frontier:
        root.solver = _match_agent(member, 0, root.statement, root.id)
    while frontier:
        results = await asyncio.gather(
            *[_decompose_cell(run, member, cells, c) for c in frontier], return_exceptions=True
        )
        nxt: list[Cell] = []
        for cell, data in zip(frontier, results):
            if not isinstance(data, dict):
                data = {}
            cell.easiness = _clamp01(data.get("easiness"), None)
            best = str(data.get("best_solver") or "")
            cell.solver = (
                best if best in valid_solvers else _match_agent(member, 0, cell.statement, cell.id)
            )
            cell.plan = _excerpt(data.get("plan"), 1200)
            listed = data.get("subcells")
            subs_raw: list[Any] = listed if isinstance(listed, list) else []
            subs = [
                s for s in subs_raw if isinstance(s, dict) and str(s.get("statement") or "").strip()
            ][:branching]
            easy = cell.easiness is not None and cell.easiness >= tau
            if bool(data.get("atomic")) or easy or cell.depth >= max_depth or len(subs) < 2:
                continue
            if len(cells) + len(subs) > max_cells:
                continue
            cell.leaf = False
            for k, s in enumerate(subs):
                cid = f"{cell.id}.{k}"
                deps = []
                for d in s.get("depends_on") or []:
                    idx = _as_index(d)
                    if idx is not None and 0 <= idx < k:
                        deps.append(f"{cell.id}.{idx}")
                sb = str(s.get("best_solver") or "")
                child = Cell(
                    id=cid,
                    statement=str(s.get("statement")).strip()[:4000],
                    depth=cell.depth + 1,
                    parent=cell.id,
                    kind=str(s.get("kind") or "lemma").strip().lower()[:24] or "lemma",
                    index=k,
                    depends_on=deps,
                    easiness=_clamp01(s.get("easiness"), None),
                    solver=sb if sb in valid_solvers else "",
                )
                if not child.solver:
                    child.solver = _match_agent(member, 0, child.statement, cid)
                reuse = run.ledger.usable(
                    str(s.get("reuse") or "").strip(), member.index, statement=child.statement
                )
                if reuse is not None:
                    child.reuse = reuse.id
                cells[cid] = child
                cell.children.append(cid)
                hard = child.easiness is None or child.easiness < tau
                if not child.reuse and child.depth < max_depth and hard:
                    nxt.append(child)
        frontier = nxt if len(cells) + 2 <= max_cells else []
    _set_heights(cells)
    return cells


def _set_heights(cells: dict[str, Cell]) -> None:
    def height(cid: str) -> int:
        c = cells[cid]
        c.height = 0 if c.leaf or not c.children else 1 + max(height(ch) for ch in c.children)
        return c.height

    height("r")


# ── Ascent ──────────────────────────────────────────────────────────────


def _dep_closure(cells: dict[str, Cell], cell: Cell) -> list[str]:
    out: list[str] = []
    cur: Cell | None = cell
    while cur is not None:
        for d in cur.depends_on:
            if d in cells and d not in out:
                out.append(d)
        cur = cells.get(cur.parent) if cur.parent else None
    return out


def _deps_text(run: _Run, cells: dict[str, Cell], ids: list[str]) -> str:
    lines = []
    for cid in ids:
        c = cells[cid]
        flag = ""
        e = run.ledger.get(c.entry)
        if e is not None and e.status in {"refuted", "suspect", "disputed"}:
            flag = f" ⚠ the body of work marks this result {e.status}"
        ref = f" [{c.entry}]" if c.entry else ""
        lines.append(
            f"- {cid}{ref} ({c.kind}): {_excerpt(c.claim or c.statement, 400)}"
            f" → {c.final_answer or '—'}{flag}"
        )
    return "\n".join(lines) or "(none)"


def _parse_cell(raw: str) -> dict[str, Any]:
    data = clean_and_parse_json(raw)
    if not isinstance(data, dict):
        text = str(raw or "")
        return {
            "result": text[:8000],
            "claim": _excerpt(text, 240),
            "final_answer": extract_final_answer(text),
            "confidence": 0.3,
        }
    return data


def _apply(cell: Cell, data: dict[str, Any]) -> dict[str, Any]:
    cell.result = str(data.get("result") or data.get("solution") or "")[:12000]
    cell.claim = _excerpt(data.get("claim") or cell.result, 600)
    answer = data.get("final_answer")
    cell.final_answer = (
        str(answer).strip()[:400]
        if answer not in (None, "")
        else extract_final_answer(cell.result)[:400]
    )
    cell.confidence = float(_clamp01(data.get("confidence"), 0.5) or 0.0)
    cell.checks = _excerpt(data.get("checks"), 800)
    cell.gaps = _str_list(data.get("gaps"), 8)
    cell.uses = _str_list(data.get("uses"), 12)
    return {
        "same_as": str(data.get("same_as") or "").strip() or None,
        "contradicts": _str_list(data.get("contradicts"), 8),
    }


async def _write(
    run: _Run, member: Member, cell: Cell, round_: int, extras: dict[str, Any], dep_cells: list[str]
) -> None:
    cells = member.cells
    depends = [cells[c].entry for c in cell.children + dep_cells if cells[c].entry]
    is_root = cell.id == "r"
    entry, conflicts = run.ledger.add(
        member=member.index,
        round_=round_,
        cell=cell.id,
        kind="proposal" if is_root else cell.kind,
        statement=cell.statement,
        result=(f"Claim: {cell.claim}\n\n{cell.result}" if cell.claim else cell.result),
        answer=cell.final_answer,
        confidence=cell.confidence,
        depends_on=depends,
        uses=cell.uses,
        same_as=None if is_root else extras.get("same_as"),
        contradicts=None if is_root else extras.get("contradicts"),
        proposal=is_root,
    )
    cell.entry = entry.id
    for c in conflicts:
        await run.log(
            f"LOG: [BODY OF WORK] conflict {c.id}: {c.a} × {c.b} ({c.reason}) — QNN {member.index}"
        )


async def _solve(run: _Run, member: Member, cell: Cell, round_: int) -> None:
    cells = member.cells
    persona_id = cell.solver or _match_agent(member, 0, cell.statement, cell.id)
    persona = member.personas.get(persona_id) or next(iter(member.personas.values()), {})
    deps = _dep_closure(cells, cell)
    hits = run.ledger.retrieve(cell.statement, member=member.index, k=run.p["ledger_top_k"])
    cell.attended = [e.id for e, _ in hits]
    prompt = SOLVER_PROMPT.format(
        persona=_persona_text(persona),
        domain_label=run.profile["label"],
        domain_brief=run.profile["brief"],
        problem=run.problem[:12000],
        framing=run.framing_block(member),
        path=_path_text(cells, cell),
        dependencies=_deps_text(run, cells, deps),
        ledger=run.ledger.context_block(hits) or "(The body of work is empty so far.)",
        cell_id=cell.id,
        kind=cell.kind,
        statement=cell.statement[:6000],
        checks=", ".join(run.profile["checks"]),
    )
    raw = await run.ask("cell", member.llm, {"input": prompt}, "solve")
    extras = _apply(cell, _parse_cell(raw))
    cell.agent, cell.layer, cell.status = persona_id, 0, "solved"
    await _write(run, member, cell, round_, extras, deps)
    if run.verbose:
        await run.log(
            f"SUCCESS: [THEORY Q{member.index} {cell.id}] solved by {persona_id} → "
            f"{_excerpt(cell.final_answer, 60)} [{cell.entry}]"
        )


async def _synthesize(run: _Run, member: Member, cell: Cell, round_: int) -> None:
    cells = member.cells
    layer = min(cell.height, member.layers - 1) if member.layers > 1 else 0
    agent = _match_agent(member, layer, cell.statement, cell.id)
    persona = member.personas.get(agent) or {}
    parts = []
    for ch_id in cell.children:
        ch = cells[ch_id]
        e = run.ledger.get(ch.entry)
        flag = ""
        if e is not None and e.status in {"refuted", "suspect", "disputed"}:
            flag = f"\n⚠ The body of work marks this result {e.status}."
        parts.append(
            f"### {ch.id} [{ch.entry or 'no entry'}] ({ch.kind}) {_excerpt(ch.statement, 600)}\n"
            f"Claim: {ch.claim}\nAnswer: {ch.final_answer or '—'}\n"
            f"Confidence: {ch.confidence:.2f}{flag}\nResult:\n{_excerpt(ch.result, 2600)}"
        )
    deps = _dep_closure(cells, cell)
    hits = run.ledger.retrieve(
        cell.statement,
        member=member.index,
        k=run.p["ledger_top_k"],
        exclude={cells[c].entry for c in cell.children if cells[c].entry},
    )
    cell.attended = [e.id for e, _ in hits]
    prompt = SYNTHESIZER_PROMPT.format(
        persona=_persona_text(persona),
        layer=layer,
        domain_label=run.profile["label"],
        domain_brief=run.profile["brief"],
        problem=run.problem[:12000],
        framing=run.framing_block(member),
        path=_path_text(cells, cell),
        cell_id=cell.id,
        kind=cell.kind,
        statement=cell.statement[:6000],
        plan=cell.plan or "(not given)",
        children="\n\n".join(parts),
        dependencies=_deps_text(run, cells, deps),
        ledger=run.ledger.context_block(hits) or "(No other entries are relevant.)",
        root_rule=ROOT_RULE if cell.id == "r" else "",
    )
    llm = run.synth_override if run.synth_override is not None else member.llm
    raw = await run.ask("cell", llm, {"input": prompt}, "synthesize")
    extras = _apply(cell, _parse_cell(raw))
    cell.agent, cell.layer, cell.status = agent, layer, "synthesized"
    await _write(run, member, cell, round_, extras, deps)
    if run.verbose:
        await run.log(
            f"SUCCESS: [THEORY Q{member.index} {cell.id}] synthesized by {agent} (layer {layer}) → "
            f"{_excerpt(cell.final_answer, 60)} [{cell.entry}]"
        )


async def _reuse(run: _Run, member: Member, cell: Cell, round_: int) -> bool:
    e = run.ledger.usable(cell.reuse, member.index)
    if e is None:
        cell.reuse = ""
        return False
    cell.result = f"Established in the body of work as {e.id} ({e.status}).\n\n{e.result}"
    cell.claim = _excerpt(e.result.split("\n", 1)[0].replace("Claim:", "").strip(), 400)
    cell.final_answer = e.answer
    cell.confidence = e.confidence
    cell.entry, cell.status, cell.agent = e.id, "reused", "body-of-work"
    run.ledger.note_reuse(e.id, member=member.index, round_=round_, cell=cell.id)
    if run.verbose:
        await run.log(f"LOG: [THEORY Q{member.index} {cell.id}] reused {e.id} ({e.status})")
    return True


async def _ascend(run: _Run, member: Member, round_: int) -> None:
    cells = member.cells
    done = {cid: asyncio.Event() for cid in cells}

    async def run_cell(cell: Cell) -> None:
        try:
            for dep in _dep_closure(cells, cell):
                await done[dep].wait()
            if cell.reuse and await _reuse(run, member, cell, round_):
                return
            if cell.leaf or not cell.children:
                await _solve(run, member, cell, round_)
            else:
                for ch in cell.children:
                    await done[ch].wait()
                await _synthesize(run, member, cell, round_)
        except _BudgetExhausted:
            cell.status, cell.result = "failed", "(the call budget ran out)"
        except Exception as e:  # one bad cell never stalls the tree
            cell.status, cell.result = "failed", f"(cell failed: {e})"
            await run.log(f"WARNING: [THEORY Q{member.index} {cell.id}] {e}")
        finally:
            done[cell.id].set()

    await asyncio.gather(*[run_cell(c) for c in cells.values()])


# ── Verification ────────────────────────────────────────────────────────


async def _verify(run: _Run, member: Member, round_: int) -> None:
    prop = member.proposal
    if not prop.get("solution"):
        return
    closure = run.ledger.closure(prop["entry"], limit=12) if prop.get("entry") else []
    lines = []
    for eid in closure:
        e = run.ledger.get(eid)
        if e is not None:
            lines.append(
                f"- {eid} ({e.kind}, {e.status}): {_excerpt(e.statement, 220)} → {e.answer or '—'}"
            )
    raw = await run.ask(
        "verifier",
        run.synth,
        {
            "domain_label": run.profile["label"],
            "checks": "; ".join(run.profile["checks"]),
            "problem": run.problem[:12000],
            "solution": str(prop.get("solution", ""))[:14000],
            "final_answer": prop.get("final_answer") or "(none)",
            "entries": "\n".join(lines) or "(none)",
        },
        "verify",
    )
    data = clean_and_parse_json(raw)
    if not isinstance(data, dict):
        data = {}
    score = float(_clamp01(data.get("score"), 0.5) or 0.0)
    issues = _str_list(data.get("issues"), 8)
    prop["verify"] = {
        "score": score,
        "verdict": str(data.get("verdict") or "")[:24],
        "issues": issues,
        "answer_check": _excerpt(data.get("answer_check"), 600),
    }
    if score >= 0.6:
        for eid in _str_list(data.get("confirmed"), 12):
            if eid in closure:
                run.ledger.confirm(eid, f"confirmed by the verifier of QNN {member.index}")
    if score < 0.7:
        why = issues[0] if issues else "the verifier found it wrong"
        for eid in _str_list(data.get("refuted"), 12):
            if eid in closure:
                # One verifier call can refute a merely proposed entry. Corroborated or
                # verified entries only get challenged: the arbiter decides.
                tainted, opened = run.ledger.challenge(
                    eid, f"the verifier of QNN {member.index}: {why}", round_
                )
                if tainted:
                    await run.log(
                        f"LOG: [BODY OF WORK] {eid} refuted; tainted: {', '.join(tainted)}"
                    )
                elif opened is not None:
                    await run.log(
                        f"LOG: [BODY OF WORK] {eid} challenged ({opened.id}) by the verifier "
                        f"of QNN {member.index}; the arbiter will decide"
                    )


# ── One member, one round ───────────────────────────────────────────────


async def _run_member(run: _Run, member: Member, round_: int) -> None:
    q = f"Q{member.index}"
    await run.emit({"type": "member", "member": member.index, "round": round_, "status": "descent"})
    try:
        member.cells = await _descend(run, member)
    except _BudgetExhausted:
        member.cells = {"r": Cell(id="r", statement=run.problem, depth=0, kind="root")}
    leaves = [c for c in member.cells.values() if c.leaf or not c.children]
    reused = sum(1 for c in member.cells.values() if c.reuse)
    depth = max(c.depth for c in member.cells.values())
    await run.log(
        f"LOG: [THEORY {q}·r{round_}] descent: {len(member.cells)} cells, {len(leaves)} leaves, "
        f"depth {depth}, {reused} reused from the body of work"
    )
    await run.emit(
        {
            "type": "member",
            "member": member.index,
            "round": round_,
            "status": "ascent",
            "cells": len(member.cells),
            "leaves": len(leaves),
        }
    )
    await _ascend(run, member, round_)
    root = member.cells["r"]
    raw_answer = root.final_answer or extract_final_answer(root.result)
    member.proposal = {
        "round": round_,
        "solution": root.result,
        "final_answer": raw_answer,
        "canonical": canonical_answer(raw_answer, answer_type=run.answer_type)
        if root.status != "failed"
        else "",
        "confidence": root.confidence,
        "entry": root.entry,
        "gaps": root.gaps,
        "cells": len(member.cells),
        "leaves": len(leaves),
        "depth": depth,
        "reused": reused,
        "framing": member.framing,
    }
    if run.p["verify"] and root.status != "failed":
        await run.emit(
            {"type": "member", "member": member.index, "round": round_, "status": "verify"}
        )
        try:
            await _verify(run, member, round_)
        except _BudgetExhausted:
            pass
        except Exception as e:
            await run.log(f"WARNING: [THEORY {q}] verifier failed: {e}")
    v = (member.proposal.get("verify") or {}).get("score")
    await run.log(
        f"SUCCESS: [THEORY {q}·r{round_}] proposal → {_excerpt(raw_answer, 80) or '(no answer)'}"
        + (f" (check {v:.2f})" if isinstance(v, float) else "")
    )
    await run.emit(
        {
            "type": "member",
            "member": member.index,
            "round": round_,
            "status": "done",
            "answer": _excerpt(raw_answer, 80),
            "canonical": member.proposal.get("canonical", ""),
            "verify": v,
        }
    )
    await run.emit({"type": "ledger", "round": round_, "stats": run.ledger.stats()})


# ── Consistency: arbitration ────────────────────────────────────────────


async def _arbitrate(run: _Run, round_: int) -> int:
    pending = run.ledger.open_conflicts()[: run.p["max_arbitrations"]]
    if not pending:
        return 0

    async def settle(cid: str) -> None:
        c = run.ledger.conflicts[cid]
        a = run.ledger.entries[c.a]
        b = run.ledger.entries.get(c.b) if c.b else None
        rival: dict[str, Any] = (
            {
                "b_id": b.id,
                "b_kind": b.kind,
                "b_status": b.status,
                "b_support": b.support,
                "b_statement": b.statement[:1200],
                "b_answer": b.answer or "—",
                "b_result": _excerpt(b.result, 3000),
            }
            if b is not None
            else {
                # A challenge: B is the objection that entry A is wrong.
                "b_id": "challenge",
                "b_kind": "objection",
                "b_status": "open",
                "b_support": 1,
                "b_statement": f"Entry {a.id} is wrong.",
                "b_answer": "—",
                "b_result": c.challenge or c.reason,
            }
        )
        raw = await run.ask(
            "arbiter",
            run.synth,
            {
                "problem": run.problem[:8000],
                "reason": c.reason,
                "a_id": a.id,
                "a_kind": a.kind,
                "a_status": a.status,
                "a_support": a.support,
                "a_statement": a.statement[:1200],
                "a_answer": a.answer or "—",
                "a_result": _excerpt(a.result, 3000),
                **rival,
            },
            "arbitrate",
        )
        data = clean_and_parse_json(raw)
        if not isinstance(data, dict):
            data = {}
        verdict = str(data.get("verdict") or "both").strip().lower()
        verdict = {"a": "a", "b": "b", "neither": "neither", "both": "both"}.get(verdict, "both")
        tainted = run.ledger.resolve(cid, verdict, str(data.get("reason") or ""))
        await run.log(
            f"LOG: [BODY OF WORK] {cid} settled: {verdict} ({_excerpt(data.get('reason'), 120)})"
            + (f"; tainted {', '.join(tainted)}" if tainted else "")
        )

    results = await asyncio.gather(*[settle(c.id) for c in pending], return_exceptions=True)
    for r in results:
        if isinstance(r, Exception) and not isinstance(r, _BudgetExhausted):
            await run.log(f"WARNING: [BODY OF WORK] arbitration failed: {r}")
    await run.emit({"type": "ledger", "round": round_, "stats": run.ledger.stats()})
    return len(pending)


# ── Dissent: Mirror Descent on personas, respawn, reframe ──────────────


def _weak_cells(run: _Run, member: Member, limit: int = 4) -> str:
    def badness(c: Cell) -> float:
        e = run.ledger.get(c.entry)
        tainted = 1.0 if e is not None and e.status in {"refuted", "suspect"} else 0.0
        return tainted * 2 + (1 - c.confidence) + (1.0 if c.status == "failed" else 0.0)

    worst = sorted(member.cells.values(), key=badness, reverse=True)[:limit]
    lines = []
    for c in worst:
        e = run.ledger.get(c.entry)
        status = e.status if e is not None else c.status
        lines.append(
            f"- {c.id} ({c.kind}, {c.agent or c.solver}, confidence {c.confidence:.2f}, {status}): "
            f"{_excerpt(c.statement, 200)}"
        )
    return "\n".join(lines) or "(none)"


async def _evolve_member(run: _Run, member: Member, loss: float, lr: float) -> None:
    agents = [
        {
            "id": aid,
            "role": p.get("role"),
            "specialty": p.get("specialty"),
            "skills": p.get("skills"),
            "system_prompt": _excerpt(p.get("system_prompt"), 900),
        }
        for aid, p in member.personas.items()
    ]
    issues = (member.proposal.get("verify") or {}).get("issues") or []
    raw = await run.ask(
        "evolver",
        member.llm,
        {
            "loss": f"{loss:.2f}",
            "learning_rate": f"{lr:.2f}",
            "issues": "\n".join(f"- {i}" for i in issues) or "(none)",
            "weak_cells": _weak_cells(run, member),
            "agents": json.dumps(agents, ensure_ascii=False),
        },
        "evolve",
    )
    data = clean_and_parse_json(raw)
    rows = data.get("agents") if isinstance(data, dict) else data if isinstance(data, list) else []
    changed = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        aid = str(row.get("id") or "")
        if aid not in member.personas:
            continue
        prompt = str(row.get("system_prompt") or "").strip()
        if len(prompt) > 30:
            member.personas[aid]["system_prompt"] = prompt[:2400]
            changed += 1
        skills = _str_list(row.get("skills"), 8)
        if skills:
            member.personas[aid]["skills"] = skills
    await run.log(
        f"LOG: [THEORY Q{member.index}] Mirror Descent: rewrote {changed} persona(s) "
        f"(loss {loss:.2f}, lr {lr:.2f})"
    )


def _sample_columns(run: _Run, width: int) -> list[str]:
    pool = run.seed_pool or ["invariant", "bound", "induct", "symmetry"]
    v = run.p["vector_word_size"]
    return [" ".join(run.rng.sample(pool, k=min(v, len(pool)))) for _ in range(width)]


def _dispute_text(run: _Run, clusters: list[Cluster], members: list[Member]) -> str:
    by_index = {m.index: m for m in members}
    lines = []
    for n, c in enumerate(clusters[:5]):
        plans = []
        for idx in c.members[:3]:
            m = by_index.get(idx)
            if m is None or "r" not in m.cells:
                continue
            root = m.cells["r"]
            tops = [m.cells[ch].statement for ch in root.children if ch in m.cells]
            plan = root.plan or "; ".join(_excerpt(t, 120) for t in tops[:3]) or "solved directly"
            plans.append(_excerpt(plan, 260))
        lines.append(
            f"- Group {chr(65 + n)}: {c.count} QNN(s), support {c.support:.0%}. "
            f"Plans: {' | '.join(plans) or '(none)'}"
        )
    abstain = [m.index for m in members if not m.proposal.get("canonical")]
    if abstain:
        lines.append(f"- {len(abstain)} QNN(s) gave no usable answer.")
    return "\n".join(lines) or "(no proposals yet)"


def _open_questions(run: _Run) -> str:
    lines = []
    for c in run.ledger.open_conflicts()[:4]:
        a = run.ledger.entries[c.a]
        rival = c.b or "a verifier's objection"
        lines.append(f"- unresolved: {_excerpt(a.statement, 160)} ({a.id} vs {rival})")
    refuted = [e for e in run.ledger.entries.values() if e.status == "refuted" and not e.proposal]
    for e in refuted[:4]:
        lines.append(f"- refuted: {_excerpt(e.statement, 160)} ({e.id})")
    return "\n".join(lines) or "(none)"


async def _reframe(
    run: _Run, clusters: list[Cluster], members: list[Member], count: int
) -> list[str]:
    count = max(1, min(8, count))
    try:
        raw = await run.ask(
            "reframer",
            run.llm,
            {
                "count": count,
                "domain_label": run.profile["label"],
                "problem": run.problem[:12000],
                "dispute": _dispute_text(run, clusters, members),
                "open_questions": _open_questions(run),
                "framing_examples": run.profile["framing_examples"],
            },
            "reframe",
        )
        data = clean_and_parse_json(raw)
        rows = (
            data.get("framings")
            if isinstance(data, dict)
            else data
            if isinstance(data, list)
            else []
        )
        framings = [_excerpt(f, 700) for f in _str_list(rows, count)]
    except _BudgetExhausted:
        framings = []
    except Exception as e:
        await run.log(f"WARNING: [THEORY] reframer failed: {e}")
        framings = []
    if not framings:
        framings = [
            f"Approach it through {x.strip()}."
            for x in str(run.profile["framing_examples"]).split(",")
            if x.strip()
        ]
    return framings


async def _prepare_dissenter(
    run: _Run, member: Member, loss: float, framing: str, swarm_size: int
) -> None:
    member.framing = framing
    floor = run.p["respawn_floor"] / max(1, swarm_size)
    try:
        if member.weight < floor:
            member.generation += 1
            member.columns = _sample_columns(run, member.width)
            await run.emit({"type": "member", "member": member.index, "status": "respawn"})
            await run.log(
                f"LOG: [THEORY Q{member.index}] weight {member.weight:.3f} < {floor:.3f}: "
                f"respawn (generation {member.generation}) from new seeds"
            )
            await _span_member(run, member)
        else:
            lr = max(0.0, min(2.0, run.p["learning_rate"] * loss))
            await run.emit({"type": "member", "member": member.index, "status": "evolve"})
            await _evolve_member(run, member, loss, lr)
    except _BudgetExhausted:
        pass
    except Exception as e:
        await run.log(f"WARNING: [THEORY Q{member.index}] evolution failed: {e}")


# ── Voting ──────────────────────────────────────────────────────────────


def _ballots(run: _Run, members: list[Member]) -> list[Ballot]:
    out = []
    for m in members:
        prop = m.proposal
        canon = prop.get("canonical") or ""
        quality = 1.0
        if run.p["verify"] and prop.get("verify"):
            score = float(prop["verify"].get("score") or 0.0)
            quality = score
            if score < run.p["min_verify"]:
                canon = ""
        elif run.p["verify"]:
            # The check never ran (provider error, budget): count it as uncertain,
            # never as better than a proposal that passed its check.
            quality = 0.5
        factor = run.ledger.quality(prop.get("entry"), run.p["suspect_penalty"])
        quality *= factor
        if factor <= 0:
            canon = ""
        out.append(
            Ballot(
                member=m.index,
                answer=canon,
                label=_excerpt(prop.get("final_answer"), 120),
                weight=m.weight,
                quality=quality,
                confidence=float(prop.get("confidence") or 0.0),
            )
        )
    return out


def _vote_record(
    run: _Run, round_: int, clusters: list[Cluster], members: list[Member], converged: bool
) -> dict[str, Any]:
    weights = [round(m.weight, 4) for m in members]
    keys = {}
    for c in clusters:
        for idx in c.members:
            keys[str(idx)] = c.key
    return {
        "round": round_,
        "clusters": [c.to_dict() for c in clusters[:8]],
        "weights": weights,
        "entropy": round(entropy([m.weight for m in members]), 4),
        "assignments": keys,
        "converged": converged,
        "quorum": run.p["quorum"],
    }


# ── Report ──────────────────────────────────────────────────────────────


def _tree_outline(member: Member, limit: int = 40) -> str:
    lines: list[str] = []

    def walk(cid: str) -> None:
        if len(lines) >= limit or cid not in member.cells:
            return
        c = member.cells[cid]
        who = c.agent or c.solver or "—"
        how = {"reused": "reused", "synthesized": "synthesized", "solved": "solved"}.get(
            c.status, c.status
        )
        ease = f", easiness {c.easiness:.2f}" if isinstance(c.easiness, float) else ""
        ans = (
            f" → {c.final_answer}"
            if c.final_answer and canonical_answer(c.final_answer) != PROOF
            else ""
        )
        lines.append(
            f"{'  ' * c.depth}- **{c.id}** ({c.kind}, {how} by {who}{ease}) "
            f"[{c.entry or '—'}]: {_excerpt(c.claim or c.statement, 220)}{ans}"
        )
        for ch in c.children:
            walk(ch)

    walk("r")
    return "\n".join(lines)


def _report(
    run: _Run,
    winner: Member | None,
    clusters: list[Cluster],
    members: list[Member],
    converged: bool,
    rounds_run: int,
    vote_history: list[dict[str, Any]],
) -> str:
    top = leader(clusters)
    answer = winner.proposal.get("final_answer", "") if winner else ""
    lines = ["# ∑ Theory Synthesis", ""]
    lines.append(f"**Answer:** {answer or '(no answer)'}")
    lines.append("")
    if top is not None:
        lines.append(
            f"**Consensus:** {top.count} of {len(members)} QNNs, support {top.support:.0%} "
            f"(quorum {run.p['quorum']:.0%}), after {rounds_run} round(s): "
            + ("converged." if converged else "not converged; this is the leading answer.")
        )
    else:
        lines.append("**Consensus:** none. No QNN produced an answer that passed its checks.")
    lines += ["", f"Domain: {run.profile['label']} · answer type: {run.answer_type}", ""]
    lines += [
        "## Solution",
        "",
        (winner.proposal.get("solution") if winner else "") or "(none)",
        "",
    ]
    lines += ["## How the swarm agreed", ""]
    if clusters:
        lines.append("| answer | QNNs | support | mass | mean check |")
        lines.append("|---|---|---|---|---|")
        for c in clusters[:8]:
            label = _excerpt(c.label, 60).replace("|", "\\|")
            lines.append(
                f"| {label} | {c.count} | {c.support:.0%} | {c.mass:.3f} | {c.mean_quality:.2f} |"
            )
    for rec in vote_history:
        tops = ", ".join(
            f"{_excerpt(c['label'], 24)} {c['support']:.0%}" for c in rec["clusters"][:3]
        )
        when = f"Round {rec['round']}" + (
            " after the last weight update" if rec.get("final") else ""
        )
        lines.append(f"- {when}: {tops or 'no answers'} (weight entropy {rec['entropy']})")
    lines.append("")
    if winner is not None:
        lines += [
            f"## Decomposition of the winning QNN (Q{winner.index})",
            "",
            _tree_outline(winner),
            "",
        ]
    lines += ["## Body of work", "", run.ledger.summary_markdown(), ""]
    lines += ["## Checks and caveats", ""]
    ver = (winner.proposal.get("verify") or {}) if winner else {}
    if ver:
        lines.append(
            f"- Verifier: score {ver.get('score', 0):.2f}, verdict {ver.get('verdict') or '—'}."
        )
        for issue in ver.get("issues") or []:
            lines.append(f"  - {issue}")
        if ver.get("answer_check"):
            lines.append(f"- Independent answer check: {ver['answer_check']}")
    if winner is not None and winner.proposal.get("gaps"):
        lines.append("- Gaps the synthesizer left open: " + "; ".join(winner.proposal["gaps"]))
    if run.p["ledger"] == "shared":
        lines.append(
            "- QNNs shared one body of work, so their votes are not independent samples. "
            "The ledger's conflict checks, not independence, keep the body of work consistent."
        )
    lines.append("- Agreement between QNNs is not a proof. Check the argument.")
    return "\n".join(lines)


# ── Entry point ─────────────────────────────────────────────────────────


def _persist(session_store: dict | None, session_id: str, payload: dict[str, Any]) -> None:
    if session_store is None or not session_id or session_id not in session_store:
        return
    session_store[session_id].update(payload)


async def run_theory_pipeline(
    llm,
    problem: str,
    params: dict[str, Any] | None = None,
    *,
    synthesis_llm=None,
    member_llms: list[Any] | None = None,
    member_models: list[str] | None = None,
    document_context: str = "",
    chat_history: list[dict] | None = None,
    log: LogFn = None,
    on_event: EventFn = None,
    session_id: str = "",
    session_store: dict | None = None,
) -> dict[str, Any]:
    """
    Run Theory Synthesis: a swarm of QNNs that decomposes, solves, synthesizes and votes.

    Parameters
    ----------
    llm :
        LangChain chat model for seeds, personas, decomposition, solving and dissent.
    problem : str
        The problem: one math or olympiad instance, an algorithm to design, a physics question.
    params : dict, optional
        See :func:`default_theory_params`; bounded by :func:`clamp_theory_params`.
    synthesis_llm :
        Optional separate model for synthesizer cells, verification, arbitration and the report.
    member_llms / member_models :
        Optional models assigned to QNNs round-robin (a mixed-model swarm), with display names.
    document_context, chat_history :
        Optional attachments and prior turns.
    log / on_event :
        Progress lines, and structured events (``member``, ``ledger``, ``vote``, ``done``).
    session_id / session_store :
        Optional web-session persistence.

    Returns
    -------
    dict
        ``TheoryResult.to_dict()``: report in ``proposed_solution``, the winning answer, the vote,
        every QNN's tree, and the body of work.
    """
    p = clamp_theory_params(params)
    rng = random.Random(p["seed"]) if p["seed"] is not None else random.Random()
    synth = synthesis_llm or llm
    run = _Run(
        llm=llm,
        synth=synth,
        synth_override=synthesis_llm,
        params=p,
        log=log,
        on_event=on_event,
        rng=rng,
    )
    S, R = p["swarm_size"], p["rounds"]
    run.verbose = S <= 8

    await run.log("--- [THEORY] Theory Synthesis: a swarm of QNNs, one body of work ---")
    try:
        from deepthink.cost import estimate_theory_cost

        est = estimate_theory_cost(
            swarm_size=S,
            rounds=R,
            layers=p["manual_layers"],
            width=p["manual_width"],
            max_depth=p["max_depth"],
            branching=p["branching"],
            max_cells=p["max_cells"],
            verify=p["verify"],
            max_arbitrations=p["max_arbitrations"],
            auto_size=p["qnn_mode"] == "auto"
            or p["domain"] == "auto"
            or p["answer_type"] == "auto",
            has_document_context=bool(document_context),
            polish=p["polish"],
        )
        await run.log(f"LOG: [COST] {est.summary_line()} (upper bound)")
    except Exception:
        pass

    # ── Step 0: problem text (and brief from documents) ────────────
    text = str(problem or "").strip()
    history = [m for m in (chat_history or []) if isinstance(m, dict)]
    if history and str(history[-1].get("content", "")).strip() == text:
        history = history[:-1]
    # Only the user's earlier turns: earlier reports carry boxed answers that would
    # anchor every QNN on the previous result.
    asked = [str(m.get("content", "")) for m in history if m.get("role") == "user"]
    if asked:
        prior = "\n".join(f"- {q}" for q in asked)
        text = f"{text}\n\nEarlier questions in this conversation (context):\n{prior[-3000:]}"
    if document_context:
        try:
            brief = await run.ask(
                "brief",
                llm,
                {"user_input": text, "document_context": document_context[:50000]},
                "brief",
            )
            text = brief.strip() or text
        except Exception as e:
            await run.log(f"WARNING: [THEORY] brief failed ({e}); using the request as given.")
    run.problem = text

    # ── Step 1: sizing ──────────────────────────────────────────────
    sized: dict[str, Any] = {}
    if p["qnn_mode"] == "auto" or p["domain"] == "auto" or p["answer_type"] == "auto":
        try:
            raw = await run.ask(
                "sizer",
                llm,
                {
                    "problem": text[:12000],
                    "document_context": (document_context or "")[:6000],
                    "domain_hint": p["domain"],
                },
                "sizer",
            )
            data = clean_and_parse_json(raw)
            sized = data if isinstance(data, dict) else {}
        except Exception as e:
            await run.log(f"WARNING: [THEORY] sizer failed ({e}); defaults.")
    domain = resolve_domain(p["domain"] if p["domain"] != "auto" else sized.get("domain"), text)
    run.domain, run.profile = domain, domain_profile(domain)
    answer_type = p["answer_type"]
    if answer_type not in ANSWER_TYPES:
        guess = str(sized.get("answer_type") or "").lower()
        answer_type = guess if guess in ANSWER_TYPES else guess_answer_type(text, domain)
    run.answer_type = answer_type
    tol = p["tolerance"]
    run.tolerance = float(run.profile["tolerance"] if tol is None else tol)
    run.ledger = TheoryLedger(
        tolerance=run.tolerance, answer_type=answer_type, shared=p["ledger"] == "shared"
    )

    layers, width = p["manual_layers"], p["manual_width"]
    if p["qnn_mode"] == "auto":
        from deepthink.cost import AUTO_AGENT_CAP

        layers = int(max(2, min(4, _num(sized.get("recommended_layers"), 3))))
        width = int(max(2, min(5, _num(sized.get("recommended_width"), 3))))
        if layers * width > AUTO_AGENT_CAP:
            width = max(1, AUTO_AGENT_CAP // layers)
        rec_depth = int(max(1, min(4, _num(sized.get("recommended_depth"), p["max_depth"]))))
        rec_b = int(max(2, min(6, _num(sized.get("recommended_branching"), p["branching"]))))
        p["max_depth"] = min(p["max_depth"], rec_depth) if p["max_depth"] > 0 else 0
        p["branching"] = min(p["branching"], rec_b)
    topology = {
        "swarm_size": S,
        "rounds": R,
        "layers": layers,
        "width": width,
        "agents_per_qnn": layers * width,
        "agents_total": S * layers * width,
        "max_depth": p["max_depth"],
        "branching": p["branching"],
        "easiness": p["easiness"],
        "max_cells": p["max_cells"],
        "qnn_mode": p["qnn_mode"],
        "vote": p["vote"],
        "quorum": p["quorum"],
        "ledger": p["ledger"],
    }
    await run.log(
        f"LOG: [THEORY STEP 1] {run.profile['label']} · answer type {answer_type} · "
        f"{S} QNNs × ({layers}L × {width}W) · depth ≤ {p['max_depth']} · b ≤ {p['branching']} · "
        f"τ = {p['easiness']} · ≤ {p['max_cells']} cells · {R} round(s) · body of work: {p['ledger']}"
    )

    # ── Step 2: seeds ───────────────────────────────────────────────
    V = p["vector_word_size"]
    want = int(max(12, min(96, V * width * min(S, 6))))
    words: list[str] = []
    try:
        seeds = await run.ask(
            "seeds",
            llm,
            {
                "word_count": want,
                "domain_label": run.profile["label"],
                "problem": text[:8000],
                "seed_verbs": run.profile["seed_verbs"],
                "seed_nouns": run.profile["seed_nouns"],
            },
            "seeds",
        )
        for w in str(seeds).replace(",", " ").split():
            w = w.strip().strip(".;:\"'()[]").lower()
            if len(w) > 1 and w.replace("-", "").isalpha() and w not in words:
                words.append(w)
    except Exception as e:
        await run.log(f"WARNING: [THEORY] seed generation failed ({e}); domain seeds.")
    for w in (run.profile["seed_verbs"] + ", " + run.profile["seed_nouns"]).split(","):
        w = w.strip().lower()
        if len(words) >= max(V * 2, 8):
            break
        if w and w not in words:
            words.append(w)
    run.seed_pool = words
    await run.log(f"LOG: [THEORY STEP 2] {len(words)} seed words: {' '.join(words[:24])}")

    # ── Step 3: the swarm ───────────────────────────────────────────
    llms = [m for m in (member_llms or []) if m is not None]
    names = list(member_models or [])
    t_lo, t_hi = p["temperature_min"], p["temperature_max"]
    members: list[Member] = []
    for i in range(S):
        base = llms[i % len(llms)] if llms else llm
        temp = None
        if t_lo is not None and t_hi is not None:
            temp = t_lo if S == 1 else t_lo + (t_hi - t_lo) * i / (S - 1)
            from deepthink.qdad.utils import llm_with_temperature

            base = llm_with_temperature(base, temp)
        m = Member(
            index=i,
            columns=_sample_columns(run, width),
            layers=layers,
            width=width,
            temperature=temp,
            model=names[i % len(names)] if names else "",
            weight=1.0 / S,
            llm=base,
        )
        members.append(m)
    await run.emit(
        {
            "type": "init",
            "members": S,
            "rounds": R,
            "quorum": p["quorum"],
            "vote": p["vote"],
            "ledger": p["ledger"],
            "domain": run.profile["label"],
            "answer_type": answer_type,
            "topology": topology,
        }
    )
    await run.log(f"LOG: [THEORY STEP 3] spanning personas for {S} QNNs...")
    await asyncio.gather(*[_span_member(run, m) for m in members])

    # ── Step 4: rounds ──────────────────────────────────────────────
    vote_history: list[dict[str, Any]] = []
    clusters: list[Cluster] = []
    converged = False
    rounds_run = 0
    to_run = list(members)
    for r in range(1, R + 1):
        rounds_run = r
        await run.log(
            f"--- [THEORY STEP 4] Round {r}/{R}: {len(to_run)} QNN(s) descend, solve, synthesize ---"
        )
        await asyncio.gather(*[_run_member(run, m, r) for m in to_run])
        settled = await _arbitrate(run, r)
        if settled:
            await run.log(f"LOG: [BODY OF WORK] settled {settled} conflict(s) after round {r}")
        ballots = _ballots(run, members)
        clusters = tally(ballots, tolerance=run.tolerance, mode=p["vote"])
        converged = is_converged(clusters, p["quorum"])
        rec = _vote_record(run, r, clusters, members, converged)
        vote_history.append(rec)
        top = leader(clusters)
        await run.log(
            f"LOG: [SWARM VOTE] round {r}: "
            + (
                " | ".join(
                    f"{_excerpt(c.label, 30)} {c.support:.0%} ({c.count})" for c in clusters[:4]
                )
                or "no answers"
            )
            + (" → converged" if converged else "")
        )
        await run.emit({"type": "vote", **rec})
        _persist(
            session_store,
            session_id,
            {
                "epoch": r,
                "theory": {
                    "members": [m.to_dict() for m in members],
                    "vote_history": vote_history,
                    "body_of_work": run.ledger.to_dict(),
                },
            },
        )
        losses = member_losses(
            ballots,
            clusters,
            tolerance=run.tolerance,
            quality_weight=0.5 if p["verify"] else 0.0,
        )
        stop = converged or run.exhausted
        new_w = (
            [m.weight for m in members]
            if stop
            else mirror_descent_step([m.weight for m in members], losses, p["learning_rate"])
        )
        for m, weight in zip(members, new_w):
            m.weight = weight
            m.history.append(
                {
                    "round": r,
                    "answer": m.proposal.get("final_answer", ""),
                    "canonical": m.proposal.get("canonical", ""),
                    "verify": (m.proposal.get("verify") or {}).get("score"),
                    "loss": round(losses[m.index], 4),
                    "weight": round(weight, 6),
                    "generation": m.generation,
                }
            )
        if stop:
            break
        if r == R:
            # The last mirror-descent step decides the final weighted vote.
            ballots = _ballots(run, members)
            clusters = tally(ballots, tolerance=run.tolerance, mode=p["vote"])
            converged = is_converged(clusters, p["quorum"])
            final_rec = _vote_record(run, r, clusters, members, converged)
            final_rec["final"] = True
            vote_history.append(final_rec)
            await run.emit({"type": "vote", **final_rec})
            break
        top = leader(clusters)
        lead_key = top.key if top is not None else None
        assigned = {idx: c.key for c in clusters for idx in c.members}
        quality = {b.member: b.quality for b in ballots}
        dissent = [
            m
            for m in members
            if assigned.get(m.index) != lead_key
            or run.ledger.quality(m.proposal.get("entry"), p["suspect_penalty"]) < 1.0
            or quality.get(m.index, 0.0) <= 0.0
        ]
        if not dissent:
            break
        framings = await _reframe(run, clusters, members, len(dissent))
        await asyncio.gather(
            *[
                _prepare_dissenter(run, m, losses[m.index], framings[k % len(framings)], S)
                for k, m in enumerate(dissent)
            ]
        )
        to_run = dissent

    # ── Step 5: winner and report ───────────────────────────────────
    top = leader(clusters)
    by_index = {m.index: m for m in members}
    winner: Member | None = None
    if top is not None:
        quality = {b.member: b.quality for b in _ballots(run, members)}
        winner = max(
            (by_index[i] for i in top.members),
            key=lambda m: (
                m.weight
                * quality.get(m.index, 0.0)
                * (0.5 + 0.5 * float(m.proposal.get("confidence") or 0.0)),
                -m.index,
            ),
        )
    else:
        with_text = [m for m in members if m.proposal.get("solution")]
        if with_text:
            winner = max(
                with_text,
                key=lambda m: float((m.proposal.get("verify") or {}).get("score") or 0.0),
            )
    report = _report(run, winner, clusters, members, converged, rounds_run, vote_history)
    if p["polish"] and winner is not None:
        try:
            polished = await run.ask(
                "report",
                synth,
                {
                    "problem": run.problem[:12000],
                    "solution": str(winner.proposal.get("solution", ""))[:14000],
                    "final_answer": winner.proposal.get("final_answer", ""),
                    "consensus": "\n".join(
                        f"- {c.label}: {c.count} QNNs, support {c.support:.0%}"
                        for c in clusters[:6]
                    ),
                    "body": run.ledger.summary_markdown(16),
                    "verifier": json.dumps(winner.proposal.get("verify") or {}, ensure_ascii=False),
                },
                "report",
            )
            if polished.strip():
                report = polished.strip()
        except Exception as e:
            await run.log(f"WARNING: [THEORY] report write-up failed ({e}); deterministic report.")

    final_answer = winner.proposal.get("final_answer", "") if winner else ""
    support = top.support if top is not None else 0.0
    final_solution = {
        "mode": "theory",
        "proposed_solution": report,
        "final_answer": final_answer,
        "canonical_answer": top.key if top is not None else "",
        "converged": converged,
        "support": round(support, 4),
        "rounds": rounds_run,
        "domain": run.profile["label"],
        "reasoning": (
            f"{top.count if top else 0} of {S} QNNs agree (support {support:.0%}); "
            + ("converged." if converged else "not converged.")
        ),
        "epoch": rounds_run,
    }
    calls = {
        "total": run.calls,
        "by_kind": dict(run.call_kinds),
        "peak_concurrency": run.peak_inflight,
        "budget_exhausted": run.exhausted,
    }
    result = TheoryResult(
        proposed_solution=report,
        final_answer=final_answer,
        canonical_answer=top.key if top is not None else "",
        converged=converged,
        support=round(support, 4),
        rounds_run=rounds_run,
        domain=domain,
        answer_type=answer_type,
        winner=(
            {
                "member": winner.index,
                "proposal": winner.proposal,
                "weight": round(winner.weight, 6),
            }
            if winner
            else {}
        ),
        vote=[c.to_dict() for c in clusters],
        vote_history=vote_history,
        weights=[round(m.weight, 6) for m in members],
        members=[m.to_dict() for m in members],
        body_of_work=run.ledger.to_dict(),
        topology=topology,
        calls=calls,
        final_solution=final_solution,
        params=p,
    ).to_dict()
    _persist(
        session_store,
        session_id,
        {
            "final_solution": final_solution,
            "theory": {
                "members": result["members"],
                "vote_history": vote_history,
                "body_of_work": result["body_of_work"],
                "calls": calls,
            },
        },
    )
    await run.emit(
        {
            "type": "done",
            "final_answer": _excerpt(final_answer, 120),
            "converged": converged,
            "support": round(support, 4),
            "rounds": rounds_run,
            "calls": run.calls,
            "ledger": run.ledger.stats(),
        }
    )
    await run.log(f"FINAL_ANSWER: {json.dumps(final_solution, ensure_ascii=False)}")
    await run.log(
        f"SUCCESS: [THEORY] Theory Synthesis complete: {_excerpt(final_answer, 80) or '(no answer)'} "
        f"({'converged' if converged else 'not converged'}, {run.calls} LLM calls)"
    )
    return result
