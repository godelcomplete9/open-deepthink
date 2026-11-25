"""Model routing for the RSI engine: a local driver and a mapper.

Two tiers, and the split is a rule of the engine, not a preference:

- **driver** (must be local): drives the Agent Mechanics lattice and its
  agents. Tokenising, motif mining, consistency checks, heal and stock
  chain-of-thought, code generation and artifact design all run here. A
  driver call is never sent to a remote API. Without a local driver bridge
  it falls back to deterministic offline heuristics.
- **mapper** (recommended: a model scoring above 70 on ARC-AGI-2): builds the
  table of correspondences and the multi-scale synthesis, i.e. the
  abstraction-transfer steps. The score is declared by the operator, not
  verified here.

Backends, in priority order:

1. **Injected bridges** (``set_llm_bridge`` for the driver,
   ``set_tier_bridge("mapper", fn)`` for the mapper), set by the pipeline from
   LangChain chat models.
2. **OpenRouter** (``OPENROUTER_API_KEY`` set), for the mapper only.
3. The **driver bridge**, for mapper calls when no mapper is available
   (recorded as ``bridge-driver-as-mapper``, so the run shows that the
   recommendation was not met).
4. **Offline heuristics** (no backend, or ``NOMOSLOOP_OFFLINE=1``).

Every routing decision is appended to ``state['routing_log']`` by the caller.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import urlparse

Tier = Literal["driver", "mapper"]

# task_kind -> tier.
ROUTING_POLICY: dict[str, Tier] = {
    "tokenise": "driver",
    "motif_mine": "driver",
    "consistency_check": "driver",
    "hierarchy_synthesis": "mapper",
    "toc_map": "mapper",
    "heal_cot": "driver",
    "stock_cot": "driver",
    "code_gen": "driver",
    "inspect": "driver",
    "artifact_design": "driver",
}

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Bridge signature: (prompt, task_kind, system, max_tokens, tier) -> str | None.
# Returning None falls through to the next backend.
_LLM_BRIDGE: Callable[..., str | None] | None = None
_TIER_BRIDGES: dict[str, Callable[..., str | None]] = {}


def set_llm_bridge(fn: Callable[..., str | None] | None) -> None:
    """Install (or clear, with ``None``) the driver bridge."""
    global _LLM_BRIDGE
    _LLM_BRIDGE = fn


def set_tier_bridge(tier: str, fn: Callable[..., str | None] | None) -> None:
    """Install (or clear) a bridge for one tier, e.g. the mapper."""
    if fn is None:
        _TIER_BRIDGES.pop(tier, None)
    else:
        _TIER_BRIDGES[tier] = fn


def clear_llm_bridge() -> None:
    set_llm_bridge(None)
    _TIER_BRIDGES.clear()


_LOCAL_HOSTS = {"localhost", "host.docker.internal", "0.0.0.0"}  # noqa: S104 - a name, not a bind


def is_local_url(url: str) -> bool:
    """True for loopback, private-network and ``.local`` hosts."""
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if host in _LOCAL_HOSTS or host.endswith(".local") or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def llm_endpoint(llm: Any) -> str | None:
    """Best-effort base URL of a LangChain chat model (None = in-process)."""
    for attr in ("base_url", "openai_api_base", "api_base", "endpoint_url"):
        val = getattr(llm, attr, None)
        if val:
            return str(val)
    client = getattr(llm, "client", None)
    val = getattr(client, "base_url", None) if client is not None else None
    return str(val) if val else None


def is_local_llm(llm: Any) -> bool:
    """A driver is local if it runs in-process or talks to a local/private host."""
    if llm is None:
        return True
    url = llm_endpoint(llm)
    return True if url is None else is_local_url(url)


def load_dotenv(path: str = ".env") -> bool:
    """Minimal .env loader (no extra dependency).

    Parses KEY=VALUE lines, ignoring blanks, `#` comments and an optional
    `export ` prefix; strips surrounding quotes. Never overrides variables
    already present in the real environment. Returns True if the file was
    read. Called once at package import (see __init__.py).
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


def driver_model() -> str:
    return os.environ.get("RSI_DRIVER_MODEL", "local-driver")


def mapper_model() -> str:
    return os.environ.get(
        "OPENROUTER_MAPPER_MODEL",
        os.environ.get("OPENROUTER_FRONTIER_MODEL", "anthropic/claude-opus-4.5"),
    )


def route(task_kind: str) -> Tier:
    """Return the tier for a task_kind. Unknown kinds run on the driver."""
    return ROUTING_POLICY.get(task_kind, "driver")


def is_offline() -> bool:
    if os.environ.get("NOMOSLOOP_OFFLINE", "").strip() in ("1", "true", "yes"):
        return True
    return not os.environ.get("OPENROUTER_API_KEY")


def _bridge_call(
    fn: Callable[..., str | None],
    prompt: str,
    task_kind: str,
    system: str,
    max_tokens: int,
    tier: str,
) -> dict[str, Any] | None:
    try:
        text = fn(prompt, task_kind, system, max_tokens, tier)
    except Exception as e:  # bridge failure -> offline stub, never crash
        return {
            "text": _offline_stub(prompt, task_kind),
            "tier": tier,
            "model": f"bridge-fallback-{tier}",
            "offline": True,
            "error": str(e),
        }
    if text is None:
        return None
    return {"text": str(text), "tier": tier, "model": f"bridge-{tier}", "offline": False}


def _openrouter(prompt: str, system: str, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER_BASE, api_key=os.environ["OPENROUTER_API_KEY"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=mapper_model(),
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
        extra_headers={"X-Title": "open-deepthink RSI mapper"},
    )
    return (resp.choices[0].message.content or "") if resp.choices else ""


def call_llm(
    prompt: str, task_kind: str, system: str = "", max_tokens: int = 1500
) -> dict[str, Any]:
    """Call the backend for ``task_kind`` and return a routing record.

    Returns {"text", "tier", "model", "offline"}. Driver calls never leave the
    machine; mapper calls may use OpenRouter.
    """
    tier = route(task_kind)
    own = _TIER_BRIDGES.get(tier)
    if own is not None:
        rec = _bridge_call(own, prompt, task_kind, system, max_tokens, tier)
        if rec is not None:
            return rec
    if tier == "driver":
        if _LLM_BRIDGE is not None:
            rec = _bridge_call(_LLM_BRIDGE, prompt, task_kind, system, max_tokens, tier)
            if rec is not None:
                return rec
        return {
            "text": _offline_stub(prompt, task_kind),
            "tier": tier,
            "model": "offline-driver",
            "offline": True,
        }

    error = ""
    if not is_offline():
        try:
            text = _openrouter(prompt, system, max_tokens)
            return {"text": text, "tier": tier, "model": mapper_model(), "offline": False}
        except Exception as e:  # never crash the loop on router failure
            error = str(e)
    if _LLM_BRIDGE is not None:
        rec = _bridge_call(_LLM_BRIDGE, prompt, task_kind, system, max_tokens, tier)
        if rec is not None:
            rec["model"] = "bridge-driver-as-mapper"
            if error:
                rec["error"] = error
            return rec
    rec = {
        "text": _offline_stub(prompt, task_kind),
        "tier": tier,
        "model": "offline-mapper",
        "offline": True,
    }
    if error:
        rec["error"] = error
    return rec


def _offline_stub(prompt: str, task_kind: str) -> str:
    # Deterministic placeholder so tests/demos pass without keys.
    head = prompt.strip().replace("\n", " ")[:220]
    return f"[offline-{task_kind}] stub synthesis for: {head} ..."
