"""Model routing for the RSI engine. Cheap vs frontier split is non-negotiable.

Policy:
- cheap/fast (user-selectable): tokenisation, motif mining, local consistency
  checks, unit tests, simple rewrites.
- frontier (Claude Opus class): multi-scale synthesis, novel scale invention,
  cross-domain fitness evaluation, final artifact design.

Three backends, in priority order:

1. **Injected bridge** (``set_llm_bridge``) — when open-deepthink drives the
   loop with a LangChain chat model (``run_rsi(llm=...)``), every agent call is
   routed through that model. This keeps the engine model-agnostic (OpenRouter,
   llama.cpp, mocks) and lets both tiers share one injected LLM.
2. **OpenRouter** (``OPENROUTER_API_KEY`` set) — direct OpenAI-compatible calls
   to the cheap/frontier models, exactly like standalone use.
3. **Offline heuristics** (no key, or ``NOMOSLOOP_OFFLINE=1``) — deterministic
   local behaviour so the demo, artifact, and tests run keyless.

Every routing decision is appended to ``state['routing_log']`` by the caller.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Literal

Tier = Literal["cheap", "frontier"]

# task_kind -> tier. This table is itself RSI-mutable (see graph.mutate_topology).
ROUTING_POLICY: dict[str, Tier] = {
    "tokenise": "cheap",
    "motif_mine": "cheap",
    "consistency_check": "cheap",
    "unit_test": "cheap",
    "simple_rewrite": "cheap",
    "hierarchy_synthesis": "frontier",
    "scale_invention": "frontier",
    "fitness_eval": "frontier",
    "artifact_design": "frontier",
    "mutation_propose": "cheap",  # bulk generation cheap; scoring is frontier
    "mutation_score": "frontier",
}

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Optional injected bridge: (prompt, task_kind, system, max_tokens, tier) -> str|None.
# Returning None falls through to OpenRouter/offline. Set by the pipeline.
_LLM_BRIDGE: Callable[..., str | None] | None = None


def set_llm_bridge(fn: Callable[..., str | None] | None) -> None:
    """Install (or clear, with ``None``) the injected LLM bridge."""
    global _LLM_BRIDGE
    _LLM_BRIDGE = fn


def clear_llm_bridge() -> None:
    set_llm_bridge(None)


def load_dotenv(path: str = ".env") -> bool:
    """Minimal .env loader (no extra dependency).

    Parses KEY=VALUE lines, ignoring blanks, `#` comments and an optional
    `export ` prefix; strips surrounding quotes. Never overrides variables
    already present in the real environment. Returns True if the file was
    read. Called once at package import (see __init__.py) so that graph
    module-level constants see .env values.
    """
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
        return True
    except OSError:
        return False


def cheap_model() -> str:
    return os.environ.get("OPENROUTER_CHEAP_MODEL", "google/gemini-2.0-flash-001")


def frontier_model() -> str:
    return os.environ.get("OPENROUTER_FRONTIER_MODEL", "anthropic/claude-opus-4.5")


def route(task_kind: str) -> Tier:
    """Return the tier for a task_kind. Unknown kinds default to cheap."""
    return ROUTING_POLICY.get(task_kind, "cheap")


def is_offline() -> bool:
    if os.environ.get("NOMOSLOOP_OFFLINE", "").strip() in ("1", "true", "yes"):
        return True
    return not os.environ.get("OPENROUTER_API_KEY")


def model_for(task_kind: str) -> str:
    return frontier_model() if route(task_kind) == "frontier" else cheap_model()


def call_llm(
    prompt: str, task_kind: str, system: str = "", max_tokens: int = 1500
) -> dict[str, Any]:
    """Call the active backend and return a routing record.

    Returns {"text": str, "tier": tier, "model": model, "offline": bool}.
    The main graph never does heavy lifting itself: it always goes through here
    via a named sub-agent wrapper in agents.py.
    """
    tier = route(task_kind)
    model = model_for(task_kind)

    if _LLM_BRIDGE is not None:
        try:
            text = _LLM_BRIDGE(prompt, task_kind, system, max_tokens, tier)
        except Exception as e:  # bridge failure -> offline stub, never crash
            return {
                "text": _offline_stub(prompt, task_kind),
                "tier": tier,
                "model": f"bridge-fallback-{tier}",
                "offline": True,
                "error": str(e),
            }
        if text is not None:
            return {"text": str(text), "tier": tier, "model": f"bridge-{tier}", "offline": False}
        # None -> fall through to OpenRouter/offline

    if is_offline():
        return {
            "text": _offline_stub(prompt, task_kind),
            "tier": tier,
            "model": f"offline-{tier}",
            "offline": True,
        }
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=OPENROUTER_BASE,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7 if tier == "frontier" else 0.2,
            extra_headers={
                "HTTP-Referer": "https://github.com/iblameandrew/open-deepthink",
                "X-Title": "open-deepthink NomosLoop RSI engine",
            },
        )
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        return {"text": text, "tier": tier, "model": model, "offline": False}
    except Exception as e:  # never crash the loop on router failure
        return {
            "text": _offline_stub(prompt, task_kind),
            "tier": tier,
            "model": f"fallback-{tier}",
            "offline": True,
            "error": str(e),
        }


def _offline_stub(prompt: str, task_kind: str) -> str:
    # Deterministic placeholder so tests/demos pass without keys.
    # Real synthesis happens in agents.py heuristics, not here.
    head = prompt.strip().replace("\n", " ")[:220]
    return f"[offline-{task_kind}] stub synthesis for: {head} ..."
