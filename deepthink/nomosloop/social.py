"""Social-engineering layer: institutional procedures as artifacts of the RSI loop.

Social engineering attacks a procedure through the people who run it: a
caller who pretends to be staff gets a password reset, an urgent e-mail "from
the CEO" changes a supplier's bank details. This layer reads an institution's
**role profile** and finds where a procedure lets that happen. Each gap is a
:class:`~.surface.Fact` that ``surface.link`` reads against the written
policy: a policy that demands a callback while the help desk resets passwords
on an employee number is a discrepancy between the rule and the routine.

The profile is role level (JSON)::

    {
      "authorization": {"by": "CISO", "scope": "help desk and finance procedures"},
      "reporting_channel": "security@…" | false,
      "phishing_simulation": false,
      "procedures": ["Urgent requests from executives are handled the same day."],
      "roles": [
        {"role": "help desk", "headcount": 6, "channels": ["phone", "chat"],
         "can": ["reset passwords", "enroll MFA devices"],
         "verification": ["employee id"], "public_exposure": true, "training": "none"}
      ]
    }

Privacy by construction:

* an ``authorization`` block (who authorized the assessment, and its scope) is
  required;
* only role-level fields are admitted. Personal and psychometric data (names,
  contact details, biographies, Big Five, MBTI, behavioural reports, and any
  list of individual people) is dropped before analysis and listed in the
  report as not admitted. The layer finds weak procedures; it does not build
  profiles of people or write pretexts and lures.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .surface import Fact

ROLE_FIELDS = (
    "role",
    "headcount",
    "channels",
    "can",
    "verification",
    "approvals",
    "callback",
    "public_exposure",
    "training",
)
TOP_FIELDS = (
    "institution",
    "authorization",
    "reporting_channel",
    "phishing_simulation",
    "procedures",
    "roles",
)
PERSONAL_LISTS = ("people", "persons", "users", "employees", "staff", "profiles", "individuals")
REMOTE = ("phone", "call", "chat", "email", "e-mail", "sms", "text", "ticket", "teams", "slack")
STRONG = (
    "callback",
    "call back",
    "call-back",
    "number on file",
    "known number",
    "manager approval",
    "manager's approval",
    "in person",
    "in-person",
    "video",
    "hardware token",
    "security key",
    "push to registered device",
    "out-of-band",
    "out of band",
    "sso",
    "verified account",
)
RESET = ("reset", "unlock", "mfa", "password", "enrol", "enroll", "recovery", "2fa", "token")
MONEY = ("bank", "vendor", "supplier", "payment", "wire", "transfer", "invoice", "payroll")
PRIVILEGED = RESET + MONEY + ("admin", "grant access", "provision", "badge", "door", "data export")
NO_TRAINING = ("", "none", "never", "no", "absent", "n/a", "false")


class SocialProfileError(ValueError):
    """The profile is missing its authorization or is not a JSON object."""


def load_profile(source: Any) -> dict[str, Any]:
    """A dict, a JSON string, or a path to a .json file."""
    if isinstance(source, dict):
        return source
    if isinstance(source, str):
        s = source.strip()
        if s.startswith("{"):
            data = json.loads(s)
        elif os.path.isfile(s):
            with open(s, encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            raise SocialProfileError(
                "social profile must be a JSON object or a path to a .json file"
            )
        if not isinstance(data, dict):
            raise SocialProfileError("social profile must be a JSON object")
        return data
    raise SocialProfileError("social profile must be a JSON object or a path to a .json file")


def _strs(v: Any) -> list[str]:
    if v is None or v is False:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip()][:20]
    return [str(v)]


def sanitize(profile: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Keep role-level fields only; return (clean profile, dropped field paths)."""
    dropped: list[str] = []
    clean: dict[str, Any] = {}
    for k, v in profile.items():
        key = str(k)
        if key in TOP_FIELDS:
            clean[key] = v
        else:
            dropped.append(key if key.lower() not in PERSONAL_LISTS else f"{key} (individuals)")
    roles = []
    for i, r in enumerate(clean.get("roles") or []):
        if not isinstance(r, dict):
            dropped.append(f"roles[{i}]")
            continue
        keep = {k: r[k] for k in ROLE_FIELDS if k in r}
        dropped += [f"roles[{i}].{k}" for k in r if k not in ROLE_FIELDS]
        if keep.get("role"):
            keep["role"] = str(keep["role"])[:60]
            roles.append(keep)
    clean["roles"] = roles
    clean["procedures"] = [p[:300] for p in _strs(clean.get("procedures"))]
    auth = clean.get("authorization")
    if isinstance(auth, dict):
        clean["authorization"] = {
            "by": str(auth.get("by", ""))[:80],
            "scope": str(auth.get("scope", ""))[:160],
            "date": str(auth.get("date", ""))[:20],
        }
    return clean, dropped


def _has(items: list[str], words: tuple[str, ...]) -> list[str]:
    return [x for x in items if any(w in x.lower() for w in words)]


def _join(items: list[str]) -> str:
    items = [x.strip() for x in items if x.strip()]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _mentions(text: str, *groups: tuple[str, ...]) -> bool:
    """True if one sentence of ``text`` holds a word from every group."""
    for sentence in re.split(r"(?<=[.;!?])\s+|\n+", text.lower()):
        if all(any(w in sentence for w in g) for g in groups):
            return True
    return False


def analyze(source: Any, policy_text: str = "") -> tuple[list[Fact], dict[str, Any]]:
    """Role profile (+ the policy text) -> social-engineering facts and run metadata."""
    profile, dropped = sanitize(load_profile(source))
    auth = profile.get("authorization")
    if not isinstance(auth, dict) or not auth.get("by"):
        raise SocialProfileError(
            "the social profile needs an 'authorization' block naming who authorized this "
            'assessment, e.g. {"by": "CISO", "scope": "help desk procedures"}'
        )
    written = " ".join([policy_text, *profile["procedures"]])
    facts: list[Fact] = []

    def add(rule: str, title: str, sp: str, sev: float, where: str, stmt: str, fix: str, topics):
        facts.append(
            Fact(
                ref="",
                kind="social",
                rule=rule,
                title=title,
                subject=where,
                statement=stmt,
                species=sp,
                severity=sev,
                remediation=fix,
                topics=list(topics),
                location=where,
            )
        )

    sensitive_remote = []
    for r in profile["roles"]:
        role = r["role"]
        where = f"role: {role}"
        chans = _has(_strs(r.get("channels")), REMOTE)
        can = _strs(r.get("can"))
        ver = _strs(r.get("verification"))
        strong = _has(ver, STRONG) or ([str(r["callback"])] if r.get("callback") else [])
        resets = _has(can, RESET)
        money = _has(can, MONEY)
        if chans and (resets or money or _has(can, PRIVILEGED)):
            sensitive_remote.append(role)
        if resets and chans and not strong:
            add(
                "pretext-reset",
                "Account reset on weak verification",
                "Fi",
                0.85,
                where,
                f"The {role} role can {_join(resets)} over {_join(chans)} after checking only "
                f"{_join(ver) or 'nothing'}, so a caller who pretends to be staff can take over an account.",
                "Verify out of band before any reset (call back the number on file, or get the "
                "manager's approval), and log every reset for review.",
                (
                    "password reset",
                    "reset",
                    "help desk",
                    "identity",
                    "verif",
                    "caller",
                    "mfa",
                    "account recovery",
                ),
            )
        try:
            approvals = int(r.get("approvals", 1))
        except (TypeError, ValueError):
            approvals = 1
        if money and (approvals < 2 or not strong):
            gaps = []
            if approvals < 2:
                gaps.append("one person can approve it")
            if not strong:
                gaps.append("no callback to a known number is required")
            add(
                "payment-redirect",
                "Payment or bank-detail change without dual control",
                "Ti",
                0.9,
                where,
                f"The {role} role can {_join(money)} and {_join(gaps)}, so a forged request "
                "(business e-mail compromise) can redirect money.",
                "Require two approvers and a callback to a number already on file before bank "
                "details change or a payment is released.",
                (
                    "payment",
                    "vendor",
                    "bank",
                    "invoice",
                    "approv",
                    "finance",
                    "transfer",
                    "dual control",
                ),
            )
        exposed = r.get("public_exposure")
        if exposed and exposed not in ("no", "none", "false") and (resets or money):
            add(
                "public-exposure",
                "Privileged role is publicly reachable",
                "Ni",
                0.5,
                where,
                f"The {role} role is listed publicly and can {_join(resets + money)}, which tells "
                "an attacker whom to call and what to ask for.",
                "Publish a shared contact instead of the role's direct channels, and treat inbound "
                "requests to it as unverified.",
                ("staff directory", "public directory", "contact details", "exposure", "org chart"),
            )
        training = str(r.get("training", "")).strip().lower()
        if chans and (resets or money) and training in NO_TRAINING:
            add(
                "no-training",
                "Exposed role without social-engineering training",
                "Fe",
                0.5,
                where,
                f"The {role} role handles sensitive requests over {_join(chans)} without "
                "social-engineering training.",
                "Train the role on pretexting and payment fraud at least once a year and after "
                "every incident.",
                ("training", "awareness", "phishing", "social engineering"),
            )
    inst = "institution"
    if sensitive_remote and not _mentions(
        written,
        ("urgent", "urgency", "executive", "ceo", "pressure", "exception"),
        ("never", "not ", "cannot", "must"),
    ):
        add(
            "urgency-bypass",
            "No rule against urgency or authority overrides",
            "Fe",
            0.6,
            inst,
            f"Nothing written says that urgency or a request from an executive cannot skip "
            f"verification, and {_join(sensitive_remote)} handle sensitive requests remotely.",
            "State that no request, however urgent or senior, skips verification, and back the "
            "staff who refuse.",
            ("urgent", "executive", "exception", "verification", "escalat", "pressure"),
        )
    reporting = profile.get("reporting_channel")
    if (not reporting or reporting in ("none", "no")) and not _mentions(
        written, ("report",), ("suspicious", "phishing", "security team", "incident")
    ):
        add(
            "no-reporting",
            "No channel to report suspicious contacts",
            "Fe",
            0.6,
            inst,
            "Staff have no stated channel to report a suspicious call, message or visitor.",
            "Publish one reporting channel (an address or a button in the mail client) and "
            "acknowledge every report.",
            ("report", "suspicious", "phishing", "notify", "security team"),
        )
    if not profile.get("phishing_simulation") and profile["roles"]:
        add(
            "no-simulation",
            "Procedures are never tested",
            "Ne",
            0.4,
            inst,
            "Social-engineering exposure is never exercised, so nobody knows whether the "
            "procedures hold.",
            "Run authorized, announced-to-management exercises against procedures (not "
            "individuals) and share the results in aggregate.",
            ("phishing", "simulation", "exercise", "test", "red team"),
        )
    facts.sort(key=lambda f: (-f.severity, f.location, f.rule))
    for n, f in enumerate(facts, 1):
        f.ref = f"soc/S{n}"
    meta = {
        "roles": len(profile["roles"]),
        "authorized_by": auth.get("by", ""),
        "scope": auth.get("scope", ""),
        "date": auth.get("date", ""),
        "dropped": dropped,
        "facts": len(facts),
    }
    return facts, meta


__all__ = ["SocialProfileError", "analyze", "load_profile", "sanitize"]
