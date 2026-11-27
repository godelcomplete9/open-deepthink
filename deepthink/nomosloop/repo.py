"""Repository analysis: code and configuration as artifacts of the RSI loop.

Walks a local repository (or one file), reads text files and matches
security-relevant patterns: hard-coded credentials, private keys, TLS
verification turned off, debug mode, services open to every interface or to
the whole internet, public storage, remote desktop switched on, SQL or shell
commands built from strings, weak hashes, containers running as root,
world-writable permissions, wildcard CORS, authentication bypasses and
logging turned off.

Each match becomes a :class:`~.surface.Fact` that ``surface.link`` reads
against the policy: code that does what a clause forbids is a discrepancy
between the rule and what actually runs.

Safety: files are only read, never executed or imported. Symbolic links,
VCS and dependency folders, binaries and large files are skipped. Secret
values are redacted before they reach a fact, so they never reach a model
prompt or the report.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .surface import Fact

SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".env.d",
        "dist",
        "build",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "site-packages",
        ".idea",
        ".vscode",
        "target",
        "vendor",
        ".terraform",
        "coverage",
    }
)
TEXT_EXT = frozenset(
    ".py .pyw .js .jsx .mjs .cjs .ts .tsx .java .kt .kts .scala .go .rs .rb .php .cs .c .h .cpp .hpp "
    ".swift .m .sh .bash .zsh .ps1 .psm1 .bat .cmd .sql .tf .tfvars .hcl .yaml .yml .json .toml .ini "
    ".cfg .conf .properties .xml .env .gradle .dockerfile .nginx .htaccess .vue .svelte .lua .pl .r".split()
)
NAMED_FILES = frozenset(
    {"dockerfile", "makefile", ".env", "procfile", "jenkinsfile", "vagrantfile"}
)
MAX_FILE_BYTES = 400_000
MAX_FILES = 3000
MAX_FACTS = 60
_COMMENT = ("#", "//", "/*", "*", "--", ";", "<!--", "rem ")


@dataclass
class Detector:
    id: str
    title: str
    species: str
    severity: float
    statement: str  # {loc} is filled in
    remediation: str
    topics: tuple[str, ...]
    patterns: tuple[re.Pattern[str], ...]
    secret: bool = False  # group "v" holds a value to redact
    files: tuple[str, ...] = ()  # restrict to these extensions / names (lowercase)
    context: bool = False  # add identifier phrases around the match as topics


def _rx(*pats: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.I) for p in pats)


DETECTORS: tuple[Detector, ...] = (
    Detector(
        "hardcoded-secret",
        "Hard-coded credential",
        "Fi",
        0.9,
        "A credential is written into source code at {loc} (value redacted), so anyone who can read the repository can use it.",
        "Move it to a secret store or environment variable, rotate it, and remove it from the version history.",
        (
            "password",
            "credential",
            "secret",
            "token",
            "api key",
            "service account",
            "rotat",
            "share passwords",
        ),
        _rx(
            r"""\b[\w.-]*(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?key|auth[_-]?token|client[_-]?secret|private[_-]?key)[\w-]*["']?\s*[:=]\s*["'](?P<v>[^"'\s]{6,})["']""",
            r"""^\s*(?:export\s+)?[A-Z0-9_]*(?:PASSWORD|PASSWD|SECRET|TOKEN|API_KEY|ACCESS_KEY)[A-Z0-9_]*\s*=\s*(?P<v>[^\s"'#$]{6,})\s*$""",
        ),
        secret=True,
    ),
    Detector(
        "private-key",
        "Private key in the repository",
        "Fi",
        0.95,
        "A private key is stored in the repository at {loc}.",
        "Revoke and reissue the key, keep it in a secret store, and remove it from the version history.",
        ("private key", "key", "credential", "certificate", "secret"),
        _rx(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"),
    ),
    Detector(
        "tls-verify-off",
        "TLS certificate verification disabled",
        "Ti",
        0.7,
        "Certificate verification is turned off at {loc}, so connections can be intercepted.",
        "Turn verification back on and trust the internal CA explicitly instead.",
        ("tls", "certificate", "encrypt", "in transit", "https", "intercept"),
        _rx(
            r"\bverify\s*=\s*False\b",
            r"rejectUnauthorized\s*:\s*false",
            r"InsecureSkipVerify\s*:\s*true",
            r"CURLOPT_SSL_VERIFYPEER\s*,\s*(?:0|false)",
            r"ssl\._create_unverified_context",
            r"NODE_TLS_REJECT_UNAUTHORIZED\s*=\s*['\"]?0",
        ),
    ),
    Detector(
        "debug-on",
        "Debug mode enabled",
        "Ti",
        0.5,
        "Debug mode is enabled at {loc}; in production it exposes internals and stack traces.",
        "Disable debug mode outside development and load it from per-environment configuration.",
        ("debug", "production", "configur", "harden", "error message"),
        _rx(r"\bDEBUG\s*=\s*True\b", r"\bdebug\s*=\s*True\b", r'"debug"\s*:\s*true'),
    ),
    Detector(
        "open-ingress",
        "Inbound traffic open to the internet",
        "Ti",
        0.8,
        "A network rule at {loc} accepts inbound traffic from any address on the internet.",
        "Restrict the source range to the networks that need it and put the service behind the VPN or a gateway.",
        ("inbound", "internet", "firewall", "ingress", "traffic", "network", "security group"),
        _rx(r"""["']0\.0\.0\.0/0["']""", r"""["']::/0["']""", r"\bsource-address\s+any\b"),
        context=True,
    ),
    Detector(
        "bind-all",
        "Service listens on every interface",
        "Ti",
        0.4,
        "A service at {loc} listens on every network interface, not only the internal one.",
        "Bind to the internal interface or localhost and expose it through a reviewed gateway.",
        ("network", "interface", "inbound", "internet", "exposed", "listen"),
        _rx(
            r"""host\s*[:=]\s*["']0\.0\.0\.0["']""",
            r"--host[= ]0\.0\.0\.0",
            r"\blisten\s+0\.0\.0\.0",
        ),
    ),
    Detector(
        "public-storage",
        "Storage readable by the public",
        "Fi",
        0.8,
        "Storage configured at {loc} can be read by anyone on the internet.",
        "Make the bucket private, serve public files through a separate bucket or CDN, and block public ACLs.",
        ("public", "bucket", "anonymous", "storage", "personal data", "public bucket"),
        _rx(
            r"public-read(?:-write)?",
            r'"Principal"\s*:\s*"\*"',
            r"block_public_acls\s*=\s*false",
            r"\ballUsers\b",
        ),
        context=True,
    ),
    Detector(
        "remote-desktop",
        "Remote desktop enabled",
        "Ti",
        0.7,
        "A script at {loc} turns remote desktop on.",
        "Keep remote desktop off on servers with sensitive data; use a bastion with MFA when it is needed.",
        ("remote desktop", "rdp", "server"),
        _rx(
            r"fDenyTSConnections\W+(?:-Value\s+)?0\b",
            r"Remote Desktop['\"]?\s*(?:-Enabled\s+True|\|\s*Enable)",
            r"Enable-NetFirewallRule\s+-DisplayGroup\s+['\"]Remote Desktop",
        ),
        context=True,
    ),
    Detector(
        "sql-injection",
        "SQL built from strings",
        "Ti",
        0.7,
        "A database query at {loc} is assembled from strings, which allows SQL injection.",
        "Use parameterised queries or the ORM's bound parameters.",
        ("input", "validat", "inject", "database", "query"),
        _rx(
            r"""\b(?:execute|executemany|query|raw)\s*\(\s*f["']""",
            r"""\b(?:execute|query)\s*\(\s*["'][^"']*\b(?:select|insert|update|delete)\b[^"']*["']\s*(?:%|\+|\.format)""",
        ),
    ),
    Detector(
        "shell-injection",
        "Shell command built at run time",
        "Ti",
        0.6,
        "A shell command or code string is evaluated at {loc}; attacker-controlled input could run commands.",
        "Pass arguments as a list without a shell, and never evaluate strings that can carry input.",
        ("input", "validat", "command", "execution", "inject"),
        _rx(r"\bshell\s*=\s*True\b", r"\bos\.system\s*\(", r"(?<![\w.])eval\s*\("),
        files=(".py", ".pyw", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".php", ".rb"),
    ),
    Detector(
        "weak-hash",
        "Weak hash function",
        "Fi",
        0.5,
        "MD5 or SHA-1 is used at {loc}; both are broken for passwords and integrity checks.",
        "Use a password hash (Argon2, bcrypt, scrypt) for credentials and SHA-256 or better elsewhere.",
        ("hash", "password", "integrity", "crypt"),
        _rx(
            r"\bhashlib\.(?:md5|sha1)\s*\(",
            r"createHash\(\s*['\"](?:md5|sha1)['\"]",
            r"MessageDigest\.getInstance\(\s*\"(?:MD5|SHA-?1)\"",
        ),
    ),
    Detector(
        "container-root",
        "Container runs as root",
        "Fi",
        0.6,
        "The container defined at {loc} runs as root.",
        "Add a non-root USER to the image and drop unneeded capabilities.",
        ("least privilege", "privilege", "root", "container", "administrat"),
        _rx(r"^\s*USER\s+(?:root|0)\b"),
        files=("dockerfile", ".dockerfile"),
    ),
    Detector(
        "world-writable",
        "World-writable permissions",
        "Fi",
        0.6,
        "Permissions at {loc} let every user write to the file or folder.",
        "Grant write access only to the owning service account.",
        ("permission", "least privilege", "integrity", "file"),
        _rx(r"\bchmod\s+(?:-R\s+)?0?777\b", r"\b0o777\b"),
    ),
    Detector(
        "cors-wildcard",
        "CORS open to every origin",
        "Ti",
        0.5,
        "The web service at {loc} accepts requests from any origin.",
        "List the allowed origins explicitly.",
        ("origin", "cors", "browser", "web", "api"),
        _rx(
            r"""Access-Control-Allow-Origin["']?\s*[:,]\s*["']\*""",
            r"""allow_origins\s*=\s*\[\s*["']\*""",
            r"""\borigin\s*:\s*["']\*["']""",
        ),
    ),
    Detector(
        "auth-bypass",
        "Authentication or CSRF check disabled",
        "Fi",
        0.7,
        "An authentication or CSRF check is switched off at {loc}.",
        "Require authentication on the route, or document and review the exception.",
        ("authenticat", "identity", "login", "access", "csrf"),
        _rx(
            r"@csrf_exempt\b",
            r"\bAllowAny\b",
            r"\.permitAll\(\)",
            r"""\bverify_signature["']?\s*[:=]\s*False""",
            r"""algorithms\s*=\s*\[\s*["']none["']""",
        ),
    ),
    Detector(
        "logging-off",
        "Logging disabled",
        "Si",
        0.6,
        "Logging is turned off at {loc}, so access to this component leaves no audit trail.",
        "Keep access and audit logging on and ship it to the central log store.",
        ("log", "logged", "audit", "retain", "monitor"),
        _rx(
            r"\blogging\.disable\s*\(",
            r'"(?:logging|enable_logging|access_log)"\s*:\s*false',
            r"\benable_logging\s*=\s*false",
            r"\baccess_log\s+off\b",
        ),
    ),
)

_PLACEHOLDER = re.compile(r"^(?:\$|%\(|<|\{|\*+$|x+$|\.\.\.|none$|null$|true$|false$)", re.I)
_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+){1,3}")


@dataclass
class ScanResult:
    root: str
    files_scanned: int = 0
    files_skipped: int = 0
    truncated: bool = False
    facts: list[Fact] = field(default_factory=list)
    by_rule: dict[str, int] = field(default_factory=dict)

    def meta(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "files_scanned": self.files_scanned,
            "files_skipped": self.files_skipped,
            "truncated": self.truncated,
            "facts": len(self.facts),
            "by_rule": dict(self.by_rule),
        }


def _wanted(name: str) -> bool:
    low = name.lower()
    if low in NAMED_FILES or low.startswith((".env", "dockerfile")):
        return True
    return os.path.splitext(low)[1] in TEXT_EXT


def _applies(det: Detector, name: str) -> bool:
    if not det.files:
        return True
    low = name.lower()
    return low in det.files or low.startswith(det.files) or os.path.splitext(low)[1] in det.files


def _redact(line: str, value: str) -> str:
    shown = value[:2] + "…[redacted]"
    return line.replace(value, shown).strip()[:160]


def _context_topics(lines: list[str], i: int) -> list[str]:
    window = " ".join(lines[max(0, i - 8) : i + 2])
    out = []
    for ident in _IDENT.findall(window):
        words = re.split(r"[_-]", ident.lower())
        if 2 <= len(words) <= 3 and all(len(w) > 1 for w in words):
            out.append(" ".join(words))
    return list(dict.fromkeys(out))[:8]


def _iter_files(root: str):
    if os.path.isfile(root):
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in SKIP_DIRS and not os.path.islink(os.path.join(dirpath, d))
        )
        for name in sorted(filenames):
            yield os.path.join(dirpath, name)


def scan_repository(
    path: str, *, max_files: int = MAX_FILES, max_facts: int = MAX_FACTS
) -> ScanResult:
    """Read a local repository (or file) and return its security facts."""
    root = os.path.realpath(path)
    if not os.path.exists(root):
        raise FileNotFoundError(path)
    base = root if os.path.isdir(root) else os.path.dirname(root)
    res = ScanResult(root=root)
    hits: dict[tuple[str, str], Fact] = {}
    counts: Counter[str] = Counter()
    for fp in _iter_files(root):
        name = os.path.basename(fp)
        if not _wanted(name) or os.path.islink(fp):
            res.files_skipped += 1
            continue
        if res.files_scanned >= max_files:
            res.truncated = True
            break
        try:
            if os.path.getsize(fp) > MAX_FILE_BYTES:
                res.files_skipped += 1
                continue
            with open(fp, "rb") as fh:
                raw = fh.read()
        except OSError:
            res.files_skipped += 1
            continue
        if b"\0" in raw[:4096]:
            res.files_skipped += 1
            continue
        res.files_scanned += 1
        rel = os.path.relpath(fp, base).replace(os.sep, "/")
        lines = raw.decode("utf-8", errors="replace").splitlines()
        for det in DETECTORS:
            if not _applies(det, name):
                continue
            for i, line in enumerate(lines):
                if len(line) > 2000:
                    continue
                stripped = line.strip().lower()
                if det.id != "private-key" and stripped.startswith(_COMMENT):
                    continue
                m = next((m for rx in det.patterns if (m := rx.search(line))), None)
                if m is None:
                    continue
                value = m.groupdict().get("v") if det.secret else None
                if det.secret and (not value or _PLACEHOLDER.match(value)):
                    continue
                key = (det.id, rel)
                counts[det.id] += 1
                if key in hits:
                    hits[key].count += 1
                    continue
                loc = f"{rel}:{i + 1}"
                topics = list(det.topics)
                if det.context:
                    topics += _context_topics(lines, i)
                hits[key] = Fact(
                    ref="",
                    kind="code",
                    rule=det.id,
                    title=det.title,
                    subject=rel,
                    statement=det.statement.format(loc=loc),
                    species=det.species,
                    severity=det.severity,
                    remediation=det.remediation,
                    topics=topics,
                    location=loc,
                    evidence=_redact(line, value) if value else line.strip()[:160],
                )
    facts = sorted(hits.values(), key=lambda f: (-f.severity, f.location))[:max_facts]
    for n, f in enumerate(facts, 1):
        f.ref = f"code/C{n}"
    res.facts = facts
    res.by_rule = dict(counts)
    return res


__all__ = ["DETECTORS", "ScanResult", "scan_repository"]
