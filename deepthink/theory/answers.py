"""Final-answer extraction and canonical forms for the Theory Synthesis vote.

Two proposals vote for the same answer when their canonical forms match:

* ``\\boxed{\\frac{1}{2}}``, ``0.5`` and ``1/2`` are one number;
* ``{3, 1, 2}``, ``\\{1,2,3\\}`` and ``n = 1, 2, 3`` are one set;
* ``Yes`` and ``true`` are one verdict, ``2^{10}`` and ``1024`` one value;
* ``9.81 m/s^2`` and ``9.8100 m/s²`` cluster within a relative tolerance.

Everything here is deterministic and makes no model call. It is a normalizer,
not a computer algebra system: answers that are equal only after real algebra
(``sin^2 x + cos^2 x`` and ``1``) stay separate.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from fractions import Fraction
from typing import Any

PROOF = "proof"
DISPROOF = "disproof"
EMPTY_SET = "{}"

_YES = {
    "yes",
    "true",
    "possible",
    "exists",
    "it exists",
    "there exists",
    "always",
    "it is possible",
    "yes it is possible",
}
_NO = {
    "no",
    "false",
    "impossible",
    "does not exist",
    "doesn't exist",
    "never",
    "it is impossible",
    "not possible",
}
_PROOF_WORDS = {
    "proof",
    "proved",
    "proven",
    "qed",
    "q.e.d",
    "q.e.d.",
    "∎",
    "□",
    "shown",
    "see proof",
    "proof complete",
    "the statement holds",
    "statement proved",
    "the claim holds",
}
_DISPROOF_WORDS = {"counterexample", "disproved", "the statement is false", "the claim is false"}
_EMPTY_WORDS = {
    "none",
    "no solutions",
    "no solution",
    "no such",
    "no such numbers",
    "no such n",
    "∅",
    "empty set",
    "the empty set",
    "emptyset",
    "varnothing",
}

_UNICODE = {
    "−": "-",
    "–": "-",
    "×": "*",
    "·": "*",
    "÷": "/",
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
    "√": "sqrt",
    "π": "pi",
    "∞": "inf",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "∈": " in ",
    "µ": "u",
    "Θ": "O",
    "θ": "O",
}

_TEXT_COMMANDS = (
    "text",
    "mathrm",
    "mathbf",
    "textbf",
    "mathit",
    "operatorname",
    "boldsymbol",
    "mathcal",
    "mathbb",
    "textrm",
)


def _braced(s: str, start: int) -> tuple[str, int] | None:
    """Content of the brace group opening at ``s[start]`` and the index of its closer."""
    if start >= len(s) or s[start] != "{":
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i], i
    return None


def extract_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` (or ``\\fbox{...}``), braces balanced."""
    s = str(text or "")
    for cmd in ("\\boxed", "\\fbox"):
        idx = s.rfind(cmd)
        while idx != -1:
            j = idx + len(cmd)
            while j < len(s) and s[j] == " ":
                j += 1
            got = _braced(s, j)
            if got is not None:
                return got[0].strip()
            idx = s.rfind(cmd, 0, idx)
    return None


_ANSWER_LINE = re.compile(
    r"(?im)^[\s>*_#-]*(?:final\s+answer|answer)[\s*_]*[:：]\s*(.+?)\s*$",
)


def extract_final_answer(text: Any) -> str:
    """Pull the final answer out of free text: boxed, then an 'Answer:' line, then the last line."""
    s = str(text or "").strip()
    if not s:
        return ""
    boxed = extract_boxed(s)
    if boxed is not None:
        return boxed
    lines = _ANSWER_LINE.findall(s)
    if lines:
        return lines[-1].strip().strip("*").strip()
    tail = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return (tail[-1] if tail else s)[:200]


def _replace_command_args(s: str, name: str, fn) -> str:
    """Replace ``\\name{a}{b}...`` using ``fn(args) -> str``; arity from ``fn.__code__``."""
    arity = fn.__code__.co_argcount
    token = "\\" + name
    out: list[str] = []
    i = 0
    while True:
        idx = s.find(token, i)
        if idx == -1:
            out.append(s[i:])
            break
        nxt = idx + len(token)
        if nxt < len(s) and s[nxt].isalpha():
            out.append(s[i:nxt])
            i = nxt
            continue
        args: list[str] = []
        j = nxt
        ok = True
        for _ in range(arity):
            while j < len(s) and s[j] == " ":
                j += 1
            got = _braced(s, j)
            if got is None:
                if j < len(s) and (s[j].isalnum()):
                    args.append(s[j])
                    j += 1
                    continue
                ok = False
                break
            args.append(got[0])
            j = got[1] + 1
        if not ok:
            out.append(s[i:nxt])
            i = nxt
            continue
        out.append(s[i:idx])
        out.append(fn(*args))
        i = j
    return "".join(out)


def _strip_latex(s: str) -> str:
    for src, dst in _UNICODE.items():
        s = s.replace(src, dst)
    s = s.replace("\\{", "{").replace("\\}", "}")
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
    s = s.replace("\\[", "").replace("\\]", "")
    s = re.sub(r"\\(?:left|right|displaystyle|big|Big|bigg|Bigg)(?![a-zA-Z])", "", s)
    s = re.sub(r"\\[,;:! ]|\\q?quad(?![a-zA-Z])|~", " ", s)
    s = re.sub(r"\\[dt]frac(?![a-zA-Z])", r"\\frac", s)
    for cmd in _TEXT_COMMANDS:
        s = _replace_command_args(s, cmd, lambda a: a)
    for _ in range(4):
        s = _replace_command_args(s, "frac", lambda a, b: f"({a})/({b})")
    s = re.sub(r"\\sqrt\s*\[([^\]]+)\]\s*\{([^{}]*)\}", r"(\2)^(1/(\1))", s)
    s = _replace_command_args(s, "sqrt", lambda a: f"sqrt({a})")
    s = re.sub(r"\\(?:cdot|times|ast)(?![a-zA-Z])", "*", s)
    s = re.sub(r"\\(?:leq|le)(?![a-zA-Z])", "<=", s)
    s = re.sub(r"\\(?:geq|ge)(?![a-zA-Z])", ">=", s)
    s = re.sub(r"\\(?:neq|ne)(?![a-zA-Z])", "!=", s)
    s = re.sub(r"\\infty(?![a-zA-Z])", "inf", s)
    s = re.sub(r"\\in(?![a-zA-Z])", " in ", s)
    s = re.sub(r"\\(?:Theta|theta|mathcal\s*O)(?=\s*\()", "O", s)
    s = re.sub(r"\\(?:emptyset|varnothing)(?![a-zA-Z])", "{}", s)
    for _ in range(3):
        s = re.sub(r"\^\s*\{([^{}]*)\}", r"^(\1)", s)
        s = re.sub(r"_\s*\{([^{}]*)\}", r"_\1", s)
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)  # \pi → pi, \log → log
    return s


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _eval_node(node: ast.AST) -> Fraction | float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise ValueError("bool")
        if isinstance(node.value, int):
            return Fraction(node.value)
        return Fraction(repr(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _eval_node(node.operand)
        return -val if isinstance(node.op, ast.USub) else val
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if isinstance(right, Fraction) and right.denominator == 1 and abs(right) <= 512:
                if isinstance(left, Fraction) and left == 0 and right < 0:
                    raise ZeroDivisionError
                if isinstance(left, Fraction) and abs(left.numerator) > 10**12:
                    raise ValueError("too large")
                return left ** int(right)
            if float(left) < 0:
                raise ValueError("complex")
            return float(left) ** float(right)
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError("operator")
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError
        return op(left, right)
    raise ValueError("unsupported")


def _as_number(s: str) -> Fraction | float | None:
    """Evaluate a plain arithmetic expression exactly when possible."""
    expr = s.replace("^", "**").strip()
    if not expr or len(expr) > 120 or not re.fullmatch(r"[\d\s.+\-*/()eE]+", expr):
        return None
    if re.search(r"\d\s+\d", expr):  # "1 2" is not a number
        return None
    if "," in s:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        val = _eval_node(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def _fmt_number(val: Fraction | float) -> str:
    if isinstance(val, Fraction):
        return str(val.numerator) if val.denominator == 1 else f"{val.numerator}/{val.denominator}"
    if float(val).is_integer() and abs(val) < 1e15:
        return str(int(val))
    return format(float(val), ".12g")


_QTY = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s*(?:\*|x)\s*10\s*\^\s*\(?\s*([-+]?\d+)\s*\)?)?"
    r"\s*([A-Za-z°Ω][A-Za-z°Ω0-9/^.*\-() ]*)?\s*$"
)


_UNIT_SYMBOLS = frozenset(
    """m s g kg mg cm mm km nm um A mA K mol cd N kN J kJ MJ eV keV MeV GeV W kW MW V mV kV
    C uC nC Pa kPa MPa atm bar Hz kHz MHz GHz T Wb H F uF nF pF ohm Ω L mL rad sr deg ° °C °F
    h min ms ns us ps ly AU pc yr""".split()
)


def _quantity(s: str) -> tuple[float, str] | None:
    """``9.81 m/s^2`` → (9.81, 'm/s^2'); ``3e8 m/s`` → (3e8, 'm/s'); no unit → None.

    Only physical units count: ``2n`` or ``3 pi`` is an expression, not a quantity, so
    it is normalized like ``2 \\cdot n`` and ``3\\pi``.
    """
    m = _QTY.match(s)
    if not m or not m.group(3):
        return None
    unit = m.group(3).strip()
    if not unit or unit.lower() in {"or", "and"}:
        return None
    symbols = re.findall(r"[A-Za-z°Ω]+", unit)
    if not symbols or any(sym not in _UNIT_SYMBOLS for sym in symbols):
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    if m.group(2):
        val *= 10 ** int(m.group(2))
    unit = re.sub(r"\s+", "", unit).replace("**", "^").replace("*", ".")
    unit = re.sub(r"\(([^()]*)\)", r"\1", unit)
    return val, unit


def _split_top(s: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


_ASSIGN = re.compile(r"^(?:[a-z](?:_\w+)?|[a-z]\(\w+\)|\w{1,3})\s*=\s*(?![=<>])")
_MEMBER = re.compile(r"^(?:[a-z](?:_\w+)?|\w{1,3})\s+in\s+(?=[\[{(])")
_LEADS = re.compile(
    r"^(?:the\s+)?(?:final\s+)?(?:answer|result)\s*(?:is|:)?\s*|"
    r"^(?:therefore|thus|hence|so)[,:]?\s+",
)


def _element(s: str) -> str:
    s = s.strip().rstrip(".;").strip()
    s = _ASSIGN.sub("", s, count=1).strip()
    num = _as_number(s)
    if num is not None:
        return _fmt_number(num)
    sym = s.replace(" ", "").replace("*", "")
    big_o = re.fullmatch(r"o\((.*)\)", sym)
    inner = big_o.group(1) if big_o else sym
    for _ in range(3):
        inner = re.sub(r"\(([A-Za-z0-9.]+)\)", r"\1", inner)
    return f"o({inner})" if big_o else inner


def _sort_key(item: str) -> tuple[int, float, str]:
    num = _as_number(item)
    if num is not None:
        return (0, float(num), item)
    return (1, 0.0, item)


def canonical_answer(raw: Any, *, answer_type: str = "") -> str:
    """Canonical string for voting. ``""`` means the proposal gave no usable answer."""
    s = str(raw or "").strip()
    if not s:
        return ""
    boxed = extract_boxed(s)
    if boxed is not None:
        s = boxed
    s = _strip_latex(s).strip()
    if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):  # 1,000 is a number, not a set
        s = s.replace(",", "")
    qty = _quantity(s)
    if qty is not None:
        return f"{_fmt_number(qty[0])} {qty[1]}"
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip().strip(".;").strip()
    while True:  # "Therefore, the answer is 5" carries two leading phrases
        stripped = _LEADS.sub("", s, count=1).strip()
        if stripped == s:
            break
        s = stripped
    words = s.strip(" .!")
    kind = (answer_type or "").lower()
    if kind == "proof":
        if words in _NO or words in _DISPROOF_WORDS or "counterexample" in words:
            return DISPROOF
        return PROOF
    if words in _PROOF_WORDS:
        return PROOF
    if words in _DISPROOF_WORDS:
        return DISPROOF
    if words in _YES:
        return "yes"
    if words in _NO:
        return "no"
    if words in _EMPTY_WORDS or s in {"{}", "{ }"}:
        return EMPTY_SET
    # "x = 1 or x = 2" / "n = 1 and n = 3" → a set of values
    alts = re.split(r"\s+(?:or|and)\s+", s)
    if len(alts) > 1 and all(_ASSIGN.match(a.strip()) for a in alts):
        items = sorted({_element(a) for a in alts}, key=_sort_key)
        return "{" + ",".join(items) + "}"
    s = _ASSIGN.sub("", s, count=1).strip()
    s = _MEMBER.sub("", s, count=1).strip()  # "n in {1, 2}" → "{1, 2}"
    # "n = 2 or 3" / "2 and 5" → a set of values
    alts = re.split(r"\s+(?:or|and)\s+", s)
    if len(alts) > 1 and all(0 < len(a) <= 40 for a in alts):
        items = sorted({_element(a) for a in alts}, key=_sort_key)
        return "{" + ",".join(items) + "}"
    if s.startswith("{") and s.endswith("}"):
        items = sorted({_element(x) for x in _split_top(s[1:-1])}, key=_sort_key)
        return "{" + ",".join(items) + "}"
    if s.startswith("[") and s.endswith("]") and "," in s:
        return "[" + ",".join(_element(x) for x in _split_top(s[1:-1])) + "]"
    if s.startswith("(") and s.endswith(")") and len(_split_top(s[1:-1])) > 1:
        return "(" + ",".join(_element(x) for x in _split_top(s[1:-1])) + ")"
    top = _split_top(s)
    if len(top) > 1 and all(len(t) <= 40 for t in top):
        items = sorted({_element(x) for x in top}, key=_sort_key)
        return "{" + ",".join(items) + "}"
    return _element(s)[:160]


def _numeric(canon: str) -> tuple[float, str] | None:
    if not canon:
        return None
    parts = canon.split(" ", 1)
    try:
        if "/" in parts[0] and parts[0].count("/") == 1:
            num, den = parts[0].split("/")
            val = float(Fraction(int(num), int(den)))
        else:
            val = float(parts[0])
    except (ValueError, ZeroDivisionError):
        return None
    return val, (parts[1] if len(parts) > 1 else "")


def same_answer(a: str, b: str, tolerance: float = 0.0) -> bool:
    """Exact canonical match, or numbers with the same unit within ``tolerance`` (relative)."""
    if a == b:
        return bool(a)
    if not a or not b or tolerance <= 0:
        return False
    na, nb = _numeric(a), _numeric(b)
    if na is None or nb is None or na[1] != nb[1]:
        return False
    scale = max(abs(na[0]), abs(nb[0]), 1e-12)
    return abs(na[0] - nb[0]) <= tolerance * scale


def cluster_keys(answers: list[str], tolerance: float = 0.0) -> list[str]:
    """Map each canonical answer to its cluster representative (first seen wins)."""
    reps: list[str] = []
    out: list[str] = []
    for ans in answers:
        if not ans:
            out.append("")
            continue
        for rep in reps:
            if same_answer(ans, rep, tolerance):
                out.append(rep)
                break
        else:
            reps.append(ans)
            out.append(ans)
    return out
