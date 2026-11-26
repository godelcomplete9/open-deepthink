"""
Theory Synthesis: a swarm of QNNs for math, olympiad (IMO) instances, algorithm
design and physics, building one consistent body of work.

    from deepthink.theory import run_theory_pipeline

See ``pipeline.py`` for the mechanism, ``ledger.py`` for the body of work,
``consensus.py`` for the vote and mirror descent, and ``answers.py`` for how
answers are compared.
"""

from .answers import canonical_answer, extract_final_answer, same_answer
from .consensus import Ballot, Cluster, is_converged, member_losses, mirror_descent_step, tally
from .domains import DOMAIN_NAMES, DOMAINS, resolve_domain
from .ledger import Conflict, Entry, TheoryLedger
from .pipeline import (
    TheoryResult,
    clamp_theory_params,
    default_theory_params,
    run_theory_pipeline,
)

__all__ = [
    "run_theory_pipeline",
    "default_theory_params",
    "clamp_theory_params",
    "TheoryResult",
    "TheoryLedger",
    "Entry",
    "Conflict",
    "canonical_answer",
    "extract_final_answer",
    "same_answer",
    "Ballot",
    "Cluster",
    "tally",
    "member_losses",
    "mirror_descent_step",
    "is_converged",
    "DOMAINS",
    "DOMAIN_NAMES",
    "resolve_domain",
]
