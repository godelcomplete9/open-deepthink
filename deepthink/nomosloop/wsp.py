"""Weight-space programming: an editing model programs the LoRA adapter.

The default LoRA loop trains on whatever traces the run produced and moves one
adapter scale up or down. With weight-space programming switched on, a strong
*editing model* (for example Claude Fable 5.2 or GPT-6.2 Astra, through the
router you configure) is shown the loop's metrics, a sample of its traces and
the adapters loaded on the local llama.cpp server. It writes a **weight
program**, a JSON document with three parts:

* ``data``: which traces to drop, which answers to rewrite into shorter
  derivations, and a few new examples to add. This shapes *what* the adapter
  learns.
* ``train``: rank, alpha, dropout, learning rate, epochs and target modules
  for the trainer. This shapes *how* it is learned.
* ``mix``: a scale per loaded adapter. llama.cpp applies a weighted sum of
  adapters (``W + Σ sᵢ·BᵢAᵢ``), so this is a linear program over the weight
  deltas themselves: task arithmetic, applied live.

The editing model never writes tensor values. Every program is validated and
clamped here before anything uses it (allowed ranks and modules, bounded
learning rate and epochs, known adapter ids, scales within the clamp). The
original trace dataset is never modified: edits go to a separate file. The mix
goes through the same keep/rollback gate as the single-scale clamp.

Privacy: the program prompt contains trace text, which quotes your policy.
Using a remote editing model sends that text to it. Programming is off by
default and must be enabled explicitly.

Without an editing model (offline, or no model configured) a deterministic
program is produced instead: exact-duplicate traces are dropped (keeping the
shortest answer), the default spec is used, and the mix steps the main adapter
by one clamp step. It is labelled ``offline-program``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

RANKS = (4, 8, 16, 32, 64)
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEFAULT_SPEC: dict[str, Any] = {
    "rank": 8,
    "alpha": 16,
    "dropout": 0.05,
    "learning_rate": 2e-4,
    "epochs": 2,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}
MAX_ANSWER_CHARS = 2000
MAX_PROMPT_CHARS = 4000
MAX_ADDED = 16

SYSTEM = (
    "You are a weight-space programmer for a LoRA adapter on a small local model. The model "
    "drives an Agent Mechanics security loop: it writes inspection findings and security "
    "controls. Your program decides what the adapter learns (data), how (train) and how the "
    "loaded adapters combine (mix). Prefer short, correct, reusable answers: an answer that "
    "states the operator and the edit in one line saves the most reasoning rounds later. "
    "Answer with one JSON object only."
)


@dataclass
class ProgramConfig:
    enabled: bool = False
    adapter_ids: list[int] = field(default_factory=list)  # adapters loaded on the server
    allow_negative: bool = False  # task negation needs a server that accepts negative scales
    sample: int = 24
    model: str = ""  # informational: which editing model the operator configured


def load_examples(path: str) -> list[dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _pair(ex: dict[str, Any]) -> tuple[str, str]:
    msgs = ex.get("messages") or []
    user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
    ans = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
    return str(user), str(ans)


def _example(kind: str, prompt: str, answer: str) -> dict[str, Any]:
    from .lora import example

    return example(kind, prompt[:MAX_PROMPT_CHARS], answer[:MAX_ANSWER_CHARS])


def build_prompt(
    examples: list[dict[str, Any]],
    history: list[dict[str, Any]],
    adapters: list[int],
    mix: list[dict[str, Any]],
    clamp_max: float,
    allow_negative: bool,
    sample: int,
) -> str:
    shown = []
    for i, ex in enumerate(examples[-sample:]):
        idx = len(examples) - min(sample, len(examples)) + i
        user, ans = _pair(ex)
        shown.append(
            {"i": idx, "kind": ex.get("kind", ""), "prompt": user[:600], "answer": ans[:400]}
        )
    lo = -clamp_max if allow_negative else 0.0
    schema = {
        "data": {
            "drop": "[indices of traces to drop]",
            "rewrite": "[{i, answer}] shorter correct answers",
            "add": f"[{{prompt, answer}}] at most {MAX_ADDED}",
        },
        "train": {
            "rank": list(RANKS),
            "alpha": "integer, at most 4*rank",
            "dropout": "0..0.3",
            "learning_rate": "1e-6..1e-3",
            "epochs": "1..10",
            "target_modules": list(MODULES),
        },
        "mix": f"[{{id, scale}}] ids from {adapters or 'the main adapter'}, scale in [{lo}, {clamp_max}]",
        "rationale": "one paragraph",
    }
    return (
        f"Loop metrics, oldest first: {json.dumps(history[-8:])}\n"
        f"Loaded adapter ids: {adapters}. Current mix: {json.dumps(mix)}.\n"
        f"Trace sample ({len(shown)} of {len(examples)}):\n{json.dumps(shown, ensure_ascii=False)}\n\n"
        f"Return a JSON object with this shape: {json.dumps(schema)}"
    )


def _json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def validate(
    program: dict[str, Any],
    n_examples: int,
    adapter_ids: list[int],
    main_id: int,
    clamp_max: float,
    allow_negative: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Clamp a program to what is allowed. Returns (clean program, corrections)."""
    notes: list[str] = []
    data_in = program.get("data") if isinstance(program.get("data"), dict) else {}
    drop = sorted(
        {int(i) for i in data_in.get("drop", []) if isinstance(i, int) and 0 <= i < n_examples}
    )
    if len(drop) > n_examples // 2:
        notes.append(f"drop capped at half the dataset ({len(drop)} -> {n_examples // 2})")
        drop = drop[: n_examples // 2]
    rewrite = []
    for r in data_in.get("rewrite", []) or []:
        if (
            isinstance(r, dict)
            and isinstance(r.get("i"), int)
            and 0 <= r["i"] < n_examples
            and str(r.get("answer", "")).strip()
        ):
            rewrite.append({"i": r["i"], "answer": str(r["answer"])[:MAX_ANSWER_CHARS]})
    add = []
    for a in (data_in.get("add", []) or [])[:MAX_ADDED]:
        if (
            isinstance(a, dict)
            and str(a.get("prompt", "")).strip()
            and str(a.get("answer", "")).strip()
        ):
            add.append(
                {
                    "prompt": str(a["prompt"])[:MAX_PROMPT_CHARS],
                    "answer": str(a["answer"])[:MAX_ANSWER_CHARS],
                }
            )
    if len(data_in.get("add", []) or []) > MAX_ADDED:
        notes.append(f"add capped at {MAX_ADDED}")

    t_in = program.get("train") if isinstance(program.get("train"), dict) else {}
    spec = dict(DEFAULT_SPEC)

    def num(key: str, lo: float, hi: float, cast: type) -> None:
        if key not in t_in:
            return
        try:
            v = cast(t_in[key])
        except (TypeError, ValueError):
            notes.append(f"train.{key} unreadable; default kept")
            return
        clamped = min(hi, max(lo, v))
        if clamped != v:
            notes.append(f"train.{key} {v} clamped to {clamped}")
        spec[key] = clamped

    if "rank" in t_in:
        try:
            r = int(t_in["rank"])
            spec["rank"] = min(RANKS, key=lambda x: (abs(x - r), x))
            if spec["rank"] != r:
                notes.append(f"train.rank {r} -> {spec['rank']}")
        except (TypeError, ValueError):
            notes.append("train.rank unreadable; default kept")
    num("alpha", 1, 4 * spec["rank"], int)
    num("dropout", 0.0, 0.3, float)
    num("learning_rate", 1e-6, 1e-3, float)
    num("epochs", 1, 10, int)
    if "target_modules" in t_in:
        mods = [m for m in t_in.get("target_modules") or [] if m in MODULES]
        dropped = [m for m in t_in.get("target_modules") or [] if m not in MODULES]
        if dropped:
            notes.append(f"unknown target_modules dropped: {dropped}")
        spec["target_modules"] = mods or list(DEFAULT_SPEC["target_modules"])

    known = set(adapter_ids) | {main_id}
    lo = -clamp_max if allow_negative else 0.0
    mix: dict[int, float] = {}
    for m in program.get("mix", []) or []:
        if not isinstance(m, dict):
            continue
        try:
            aid, s = int(m.get("id")), float(m.get("scale"))
        except (TypeError, ValueError):
            continue
        if aid not in known:
            notes.append(f"mix id {aid} is not a loaded adapter; ignored")
            continue
        c = min(clamp_max, max(lo, s))
        if c != s:
            notes.append(f"mix scale for adapter {aid} {s} clamped to {c}")
        mix[aid] = c
    clean = {
        "data": {"drop": drop, "rewrite": rewrite, "add": add},
        "train": spec,
        "mix": [{"id": k, "scale": round(v, 4)} for k, v in sorted(mix.items())],
        "rationale": str(program.get("rationale", ""))[:1200],
    }
    return clean, notes


def apply_data(examples: list[dict[str, Any]], data: dict[str, Any]) -> list[dict[str, Any]]:
    """The programmed dataset: drops, rewrites, then additions. Input is not modified."""
    drop = set(data.get("drop", []))
    rewrites = {r["i"]: r["answer"] for r in data.get("rewrite", [])}
    out = []
    for i, ex in enumerate(examples):
        if i in drop:
            continue
        if i in rewrites:
            user, _ = _pair(ex)
            out.append(_example(str(ex.get("kind", "rewrite")), user, rewrites[i]))
        else:
            out.append(ex)
    out += [_example("programmed", a["prompt"], a["answer"]) for a in data.get("add", [])]
    return out


def offline_program(
    examples: list[dict[str, Any]], main_id: int, scale: float, step: float, clamp_max: float
) -> dict[str, Any]:
    """Deterministic program: dedupe identical prompts (keep the shortest answer), default spec."""
    best: dict[str, tuple[int, int]] = {}
    for i, ex in enumerate(examples):
        user, ans = _pair(ex)
        if user not in best or len(ans) < best[user][1]:
            best[user] = (i, len(ans))
    keep = {i for i, _ in best.values()}
    return {
        "data": {
            "drop": [i for i in range(len(examples)) if i not in keep],
            "rewrite": [],
            "add": [],
        },
        "train": dict(DEFAULT_SPEC),
        "mix": [{"id": main_id, "scale": min(clamp_max, scale + step)}],
        "rationale": "offline-program: exact-duplicate traces dropped (shortest answer kept); default spec; one clamp step.",
    }


def program(
    lcfg: Any,
    pcfg: ProgramConfig,
    state: Any,
    history: list[dict[str, Any]],
    call: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Ask the editing model for a weight program, validate it, write it to disk."""
    examples = load_examples(state.dataset_path)
    adapters = sorted(set(pcfg.adapter_ids) | {int(lcfg.adapter_id)})
    current = state.mix or [{"id": int(lcfg.adapter_id), "scale": state.scale}]
    r = call(
        build_prompt(
            examples, history, adapters, current, lcfg.clamp_max, pcfg.allow_negative, pcfg.sample
        ),
        task_kind="weight_program",
        system=SYSTEM,
        max_tokens=8000,
    )
    raw = None if r.get("offline") else _json_object(r.get("text", ""))
    source = r.get("model", "")
    if raw is None:
        raw = offline_program(
            examples, int(lcfg.adapter_id), state.scale, lcfg.clamp_step, lcfg.clamp_max
        )
        source = (
            "offline-program"
            if r.get("offline")
            else f"offline-program (unreadable reply from {source})"
        )
    clean, notes = validate(
        raw, len(examples), adapters, int(lcfg.adapter_id), lcfg.clamp_max, pcfg.allow_negative
    )
    edited = apply_data(examples, clean["data"])
    out_dir = lcfg.out_dir or os.path.dirname(state.dataset_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    n = len(state.programs) + 1
    data_path = os.path.join(out_dir, f"lora_dataset.programmed.{n}.jsonl")
    with open(data_path, "w", encoding="utf-8") as f:
        for ex in edited:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    spec_path = os.path.join(out_dir, f"lora_program.{n}.json")
    record = {
        "n": n,
        "source": source,
        "configured_model": pcfg.model,
        "examples_in": len(examples),
        "examples_out": len(edited),
        "program": clean,
        "corrections": notes,
        "dataset_path": data_path,
        "spec_path": spec_path,
        "routing": {k: r.get(k) for k in ("tier", "model", "offline")},
    }
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    state.programs.append(
        {k: v for k, v in record.items() if k != "program"}
        | {"mix": clean["mix"], "train": clean["train"], "rationale": clean["rationale"]}
    )
    return record
