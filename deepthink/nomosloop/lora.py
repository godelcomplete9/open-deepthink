"""LoRA loop: distil the loop's own traces into the local driver model.

NX-1 §9 reads a LoRA adapter as side mass: stock the loop paid for many times
(heal chains, accepted artifact edits) folded into the driver's weights, so
the routes used most get cheaper and a solve needs fewer chain-of-thought
rounds.

What this module does, concretely:

1. **Dataset.** Every successful heal or stock becomes one chat-format JSONL
   example (``prompt -> compressed answer``) in ``<out_dir>/lora_dataset.jsonl``.
2. **Train (optional).** If a training command is configured *server-side*
   (``RSI_LORA_TRAIN_CMD`` or the library ``train_cmd`` argument), it runs with
   ``{dataset}``, ``{out_dir}`` and ``{run_id}`` substituted. Nothing is
   trained otherwise; the engine never ships a trainer.
3. **Clamp.** When the loop measurably improves, the adapter scale is raised
   by one step (capped at ``clamp_max``) and pushed to a running llama.cpp
   server through its ``POST /lora-adapters`` endpoint. When the next window
   does not improve, the scale returns to the last good value.
4. **Distilled memory.** Offline (or before an adapter exists), the same
   traces act as a lookup that lets an agent reuse a known heal in one round.
   This is the measured stand-in for the LoRA effect, and it is labelled as such.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoraConfig:
    enabled: bool = False
    base_url: str = ""  # llama.cpp server root, e.g. http://localhost:8080
    adapter_id: int = 0
    clamp_max: float = 1.0
    clamp_step: float = 0.25
    train_cmd: str = ""
    out_dir: str = ""
    min_examples: int = 6
    timeout_s: int = 3600

    @staticmethod
    def server_root(url: str) -> str:
        """llama.cpp serves /lora-adapters at the root, not under /v1."""
        url = (url or "").rstrip("/")
        for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
            if url.endswith(suffix):
                url = url[: -len(suffix)]
        return url


@dataclass
class LoraState:
    examples: int = 0
    scale: float = 0.0
    good_scale: float = 0.0
    trained: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    last_metric: dict[str, float] = field(default_factory=dict)
    dataset_path: str = ""
    adapter_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "scale": round(self.scale, 3),
            "good_scale": round(self.good_scale, 3),
            "trained": self.trained,
            "history": self.history[-20:],
            "last_metric": self.last_metric,
            "dataset_path": self.dataset_path,
            "adapter_path": self.adapter_path,
        }


class DistilledMemory:
    """Signature -> known heal operator. The offline stand-in for the adapter."""

    def __init__(self) -> None:
        self.table: dict[str, str] = {}
        self.hits = 0

    def recall(self, sig: str) -> str | None:
        op = self.table.get(sig)
        if op:
            self.hits += 1
        return op

    def store(self, sig: str, op: str) -> None:
        self.table[sig] = op


def example(kind: str, prompt: str, answer: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "messages": [
            {
                "role": "system",
                "content": "You are an Agent Mechanics agent. Answer in the fewest steps.",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def append_examples(state: LoraState, out_dir: str, rows: list[dict[str, Any]]) -> None:
    if not rows or not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "lora_dataset.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    state.examples += len(rows)
    state.dataset_path = path


def improved(prev: dict[str, float], cur: dict[str, float]) -> bool:
    """Improvement = fewer rounds per solve, lower ensemble leak, or higher capacity."""
    if not prev:
        return cur.get("solves", 0) > 0
    better = 0
    if cur.get("rounds_per_solve", 1e9) < prev.get("rounds_per_solve", 1e9) - 1e-9:
        better += 1
    if cur.get("leak", 1.0) < prev.get("leak", 1.0) - 1e-4:
        better += 1
    if cur.get("capacity", 0.0) > prev.get("capacity", 0.0) + 1e-9:
        better += 1
    worse = cur.get("leak", 0.0) > prev.get("leak", 0.0) + 0.05
    return better >= 1 and not worse


def train(cfg: LoraConfig, state: LoraState, run_id: str) -> dict[str, Any]:
    """Run the configured trainer. Returns a record; never raises."""
    if not cfg.train_cmd:
        return {"ran": False, "reason": "no trainer configured (RSI_LORA_TRAIN_CMD)"}
    if state.examples < cfg.min_examples:
        return {"ran": False, "reason": f"{state.examples} < {cfg.min_examples} examples"}
    cmd = cfg.train_cmd.format(dataset=state.dataset_path, out_dir=cfg.out_dir, run_id=run_id)
    try:
        proc = subprocess.run(  # noqa: S603 - operator-configured command, shell=False
            shlex.split(cmd, posix=os.name != "nt"),
            capture_output=True,
            text=True,
            timeout=cfg.timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return {"ran": True, "ok": False, "error": str(e)}
    ok = proc.returncode == 0
    if ok:
        state.trained += 1
        state.adapter_path = os.path.join(cfg.out_dir, "adapter")
    return {
        "ran": True,
        "ok": ok,
        "returncode": proc.returncode,
        "tail": (proc.stdout or proc.stderr)[-400:],
    }


def _http(
    url: str, method: str = "GET", body: Any = None, timeout: float = 5.0
) -> tuple[bool, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local server URL
            raw = resp.read().decode("utf-8") or "null"
            try:
                return True, json.loads(raw)
            except json.JSONDecodeError:
                return True, raw
    except (urllib.error.URLError, OSError, ValueError) as e:
        return False, str(e)


def list_adapters(base_url: str) -> tuple[bool, Any]:
    return _http(LoraConfig.server_root(base_url) + "/lora-adapters")


def clamp(cfg: LoraConfig, scale: float) -> dict[str, Any]:
    """Set the adapter scale on a running llama.cpp server (POST /lora-adapters)."""
    scale = max(0.0, min(cfg.clamp_max, scale))
    if not cfg.base_url:
        return {"pushed": False, "scale": scale, "reason": "no llama.cpp server configured"}
    ok, resp = _http(
        LoraConfig.server_root(cfg.base_url) + "/lora-adapters",
        "POST",
        [{"id": int(cfg.adapter_id), "scale": scale}],
    )
    return {
        "pushed": ok,
        "scale": scale,
        "response": resp if ok else None,
        "error": None if ok else resp,
    }


def step(
    cfg: LoraConfig, state: LoraState, metric: dict[str, float], run_id: str, tick: int
) -> dict[str, Any]:
    """One keep/rollback decision of the clamp (NX-1 legality rule X10)."""
    rec: dict[str, Any] = {"t": tick, "metric": metric, "prev": dict(state.last_metric)}
    if improved(state.last_metric, metric):
        # The window under the current scale improved: confirm it, then try one step more.
        rec["train"] = (
            train(cfg, state, run_id) if cfg.enabled else {"ran": False, "reason": "lora disabled"}
        )
        state.good_scale = state.scale
        new_scale = min(cfg.clamp_max, state.scale + cfg.clamp_step)
        rec["decision"] = "raise" if new_scale > state.scale else "hold"
    else:
        # No measured improvement: the tentative step is reverted to the last good scale.
        new_scale = state.good_scale
        rec["decision"] = "rollback" if new_scale < state.scale else "hold"
    state.scale = new_scale
    rec["clamp"] = (
        clamp(cfg, new_scale)
        if cfg.enabled
        else {"pushed": False, "scale": new_scale, "reason": "lora disabled"}
    )
    state.last_metric = metric
    state.history.append({k: v for k, v in rec.items() if k != "prev"})
    return rec
