"""Swarm vote and entropic mirror descent for Theory Synthesis.

Each QNN in the swarm casts one ballot: the canonical final answer of its
proposal, its weight ``w_i`` on the probability simplex, and a quality ``q_i``
(the verifier score, discounted when the proposal rests on refuted results).

* **mass**: ``Σ w_i·q_i`` over a cluster's members. The heaviest cluster leads.
* **support**: ``Σ w_i`` over a cluster's members. Abstentions keep their
  weight in the denominator, so two agreeing QNNs out of ten are 20 % support,
  not 100 %.
* **convergence**: the leader's support reaches the quorum.
* **mirror descent**: ``w_i ← w_i · exp(−η ℓ_i) / Z`` with loss
  ``ℓ_i = (1−λ)(1 − support of i's cluster) + λ(1 − q_i)``. This is
  exponentiated gradient on the simplex (negative-entropy mirror map): weight
  flows to QNNs that agree with the swarm and pass verification.

In ``plurality`` mode weights and quality are ignored: support is the share of
QNNs per answer, and the most common answer wins.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from deepthink.theory.answers import cluster_keys

VOTE_MODES = ("weighted", "plurality")


@dataclass
class Ballot:
    member: int
    answer: str  # canonical; "" = abstain
    label: str = ""  # display form of the answer
    weight: float = 0.0
    quality: float = 1.0
    confidence: float = 0.0


@dataclass
class Cluster:
    key: str
    label: str
    members: list[int] = field(default_factory=list)
    count: int = 0
    mass: float = 0.0
    support: float = 0.0
    count_share: float = 0.0
    mean_quality: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("mass", "support", "count_share", "mean_quality"):
            d[k] = round(float(d[k]), 4)
        return d


def tally(
    ballots: list[Ballot],
    *,
    tolerance: float = 0.0,
    mode: str = "weighted",
) -> list[Cluster]:
    """Group ballots by answer and rank clusters (heaviest first). Abstentions are left out."""
    if not ballots:
        return []
    plurality = mode == "plurality"
    n = len(ballots)
    keys = cluster_keys([b.answer for b in ballots], tolerance)
    total_w = sum(max(0.0, b.weight) for b in ballots) or 1.0
    groups: dict[str, Cluster] = {}
    for b, key in zip(ballots, keys):
        if not key:
            continue
        c = groups.get(key)
        if c is None:
            c = groups[key] = Cluster(key=key, label=b.label or key)
        c.members.append(b.member)
        c.count += 1
        w = 1.0 / n if plurality else max(0.0, b.weight) / total_w
        q = 1.0 if plurality else max(0.0, min(1.0, b.quality))
        c.support += w
        c.mass += w * q
        c.mean_quality += q
    for c in groups.values():
        c.count_share = c.count / n
        c.mean_quality = c.mean_quality / c.count if c.count else 0.0
    return sorted(groups.values(), key=lambda c: (-c.mass, -c.count, -c.mean_quality, c.key))


def leader(clusters: list[Cluster]) -> Cluster | None:
    return clusters[0] if clusters else None


def is_converged(clusters: list[Cluster], quorum: float) -> bool:
    top = leader(clusters)
    return bool(top is not None and top.support + 1e-12 >= quorum)


def member_losses(
    ballots: list[Ballot],
    clusters: list[Cluster],
    *,
    tolerance: float = 0.0,
    quality_weight: float = 0.5,
) -> list[float]:
    """Per-ballot loss in [0, 1]: disagreement with the swarm blended with failed checks."""
    keys = cluster_keys([b.answer for b in ballots] + [c.key for c in clusters], tolerance)
    ballot_keys = keys[: len(ballots)]
    support = {c.key: c.support for c in clusters}
    lam = max(0.0, min(1.0, quality_weight))
    out: list[float] = []
    for b, key in zip(ballots, ballot_keys):
        if not key:
            out.append(1.0)
            continue
        s = support.get(key, 0.0)
        q = max(0.0, min(1.0, b.quality))
        out.append(max(0.0, min(1.0, (1 - lam) * (1 - s) + lam * (1 - q))))
    return out


def mirror_descent_step(weights: list[float], losses: list[float], eta: float) -> list[float]:
    """One entropic mirror-descent (exponentiated gradient) step on the probability simplex."""
    n = len(weights)
    if n == 0:
        return []
    if len(losses) != n:
        raise ValueError("weights and losses must have the same length")
    eta = max(0.0, float(eta))
    base = min(losses) if losses else 0.0
    raw = [max(0.0, w) * math.exp(-eta * (loss - base)) for w, loss in zip(weights, losses)]
    z = sum(raw)
    if not math.isfinite(z) or z <= 0:
        return [1.0 / n] * n
    return [r / z for r in raw]


def entropy(weights: list[float]) -> float:
    """Shannon entropy of the weights (nats); ``log n`` is uniform, 0 is one QNN."""
    return -sum(w * math.log(w) for w in weights if w > 0)
