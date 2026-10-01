"""
lgr_core.py
===========
Núcleo matemático do Solver LGR (Lugar Geométrico das Raízes).

Tradução fiel da lógica do `index.html` original para Python/numpy, sem nenhuma
dependência de interface. Tudo aqui é testável isoladamente.

Convenções
----------
* Polinômios são arrays numpy 1-D com coeficientes em ordem decrescente de
  potência (mesma convenção de `np.polyval`): [1, 4, 4, 0] = s³ + 4s² + 4s.
* O "memorial" é uma lista de passos; cada passo é uma lista de blocos
  (tipo, conteúdo). A interface decide como desenhar cada tipo de bloco.
"""
from __future__ import annotations

import ast
import cmath
import math
import operator
import re
import uuid
from dataclasses import dataclass, field

import numpy as np

# ════════════════════════════════════════════════════════════════════════════
#  1. Avaliador seguro de expressões e parser de polinômios
# ════════════════════════════════════════════════════════════════════════════


class ExprError(ValueError):
    """Erro amigável de interpretação de expressão."""


class SymPoly:
    """Polinômio simbólico em 's' usado só para interpretar '10(s+2)(s^2+3s)'."""

    def __init__(self, c):
        self.c = np.atleast_1d(np.array(c, dtype=float))

    @staticmethod
    def _co(o):
        if isinstance(o, SymPoly):
            return o
        if isinstance(o, complex):
            if abs(o.imag) > 1e-12:
                raise ExprError("Coeficientes complexos não são suportados em polinômios.")
            o = o.real
        return SymPoly([float(o)])

    def __add__(self, o):
        return SymPoly(np.polyadd(self.c, self._co(o).c))

    __radd__ = __add__

    def __sub__(self, o):
        return SymPoly(np.polysub(self.c, self._co(o).c))

    def __rsub__(self, o):
        return SymPoly(np.polysub(self._co(o).c, self.c))

    def __mul__(self, o):
        return SymPoly(np.polymul(self.c, self._co(o).c))

    __rmul__ = __mul__

    def __neg__(self):
        return SymPoly(-self.c)

    def __pos__(self):
        return self

    def __truediv__(self, o):
        o = self._co(o)
        if len(trim(o.c)) != 1:
            raise ExprError("Divisão por polinômio não é suportada aqui (use numerador e denominador separados).")
        return SymPoly(self.c / trim(o.c)[0])

    def __pow__(self, n):
        if isinstance(n, SymPoly):
            n = trim(n.c)
            if len(n) != 1:
                raise ExprError("Expoente inválido.")
            n = n[0]
        if abs(n - round(n)) > 1e-12 or n < 0 or n > 64:
            raise ExprError("Só são aceitos expoentes inteiros entre 0 e 64.")
        out = SymPoly([1.0])
        for _ in range(int(round(n))):
            out = out * self
        return out


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": cmath.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
          "log": math.log, "ln": math.log, "exp": math.exp, "abs": abs, "atan": math.atan}
_CONSTS = {"pi": math.pi, "e": math.e}

_TOK = re.compile(
    r"\s*(?:(?P<num>\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"
    r"|(?P<id>[A-Za-z_]\w*)"
    r"|(?P<op>\*\*|[-+*/^(),])"
    r"|(?P<bad>\S))"
)


def _prepare(expr: str) -> str:
    """Troca ^ por **, vírgula decimal e insere multiplicação implícita: 10(s+2) → 10*(s+2)."""
    expr = expr.strip().replace(",", ".") if expr.count(",") and not re.search(r"[A-Za-z(]", expr) else expr.strip()
    toks = []
    for m in _TOK.finditer(expr):
        if m.group("bad"):
            raise ExprError(f"Caractere inválido: '{m.group('bad')}'")
        if m.group("num"):
            toks.append(("num", m.group("num")))
        elif m.group("id"):
            toks.append(("id", m.group("id")))
        else:
            op = m.group("op")
            toks.append(("op", "**" if op == "^" else op))
    out = []
    for i, (k, t) in enumerate(toks):
        if out:
            pk, pt = toks[i - 1]
            prev_value = pk == "num" or (pk == "op" and pt == ")") or (pk == "id" and pt not in _FUNCS)
            next_value = k in ("num", "id") or (k == "op" and t == "(")
            if prev_value and next_value:
                out.append("*")
        out.append(t)
    return " ".join(out)


def evaluate(expr: str, names: dict | None = None):
    """Avalia uma expressão aritmética com AST restrita (sem eval/exec)."""
    names = {**_CONSTS, **(names or {})}
    if expr is None or not str(expr).strip():
        raise ExprError("campo vazio")
    try:
        tree = ast.parse(_prepare(str(expr)), mode="eval")
    except SyntaxError:
        raise ExprError(f"expressão inválida: '{expr}'")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, complex)):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in names:
                return names[n.id]
            raise ExprError(f"nome desconhecido: '{n.id}'")
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and not isinstance(a, SymPoly) and not isinstance(b, SymPoly):
                if abs(b) > 64:
                    raise ExprError("expoente muito grande")
            try:
                return _BIN[type(n.op)](a, b)
            except ZeroDivisionError:
                raise ExprError("divisão por zero")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and len(n.args) == 1:
            arg = ev(n.args[0])
            if isinstance(arg, SymPoly):
                raise ExprError(f"{n.func.id}() não aceita 's'")
            try:
                return _FUNCS[n.func.id](arg)
            except (ValueError, OverflowError):
                raise ExprError(f"argumento inválido em {n.func.id}()")
        raise ExprError(f"expressão não suportada: '{expr}'")

    return ev(tree)


def parse_scalar(text: str, names: dict | None = None) -> float:
    """Número real a partir de texto (aceita sqrt(3), pi/2, 1e-3, ...)."""
    v = evaluate(text, names)
    if isinstance(v, SymPoly):
        v = trim(v.c)
        if len(v) != 1:
            raise ExprError("esperava um número, mas encontrou 's'")
        v = v[0]
    if isinstance(v, complex):
        if abs(v.imag) > 1e-12:
            raise ExprError("esperava um número real")
        v = v.real
    return float(v)


def parse_complex(text: str) -> complex:
    v = evaluate(text, {"j": 1j, "i": 1j})
    if isinstance(v, SymPoly):
        raise ExprError("raiz inválida")
    return complex(v)


def parse_array(text: str) -> np.ndarray:
    """
    Aceita três formatos:
      • lista de coeficientes:   [1, 4, 4, 0]
      • simbólico em s:          10(s+2)   |   (s+1)(s+3)^2   |   s^2+4s+4
      • ganho e raízes:          5: [-1, -2+3j, -2-3j]
    """
    if text is None or not str(text).strip():
        return np.array([1.0])
    s = str(text).strip()

    if ":" in s:
        gain_s, roots_s = s.split(":", 1)
        gain = parse_scalar(gain_s) if gain_s.strip() else 1.0
        roots = [parse_complex(r) for r in roots_s.replace("[", "").replace("]", "").split(",") if r.strip()]
        return trim(gain * poly_from_roots(roots))

    if "[" in s or ("," in s and "(" not in s and "s" not in s.lower()):
        items = [x for x in s.replace("[", "").replace("]", "").split(",") if x.strip()]
        if not items:
            raise ExprError("lista vazia")
        return trim(np.array([parse_scalar(x) for x in items], dtype=float))

    v = evaluate(s.replace("S", "s"), {"s": SymPoly([1.0, 0.0])})
    if isinstance(v, SymPoly):
        return trim(v.c)
    if isinstance(v, complex):
        if abs(v.imag) > 1e-12:
            raise ExprError("coeficiente complexo")
        v = v.real
    return np.array([float(v)])


# ════════════════════════════════════════════════════════════════════════════
#  2. Álgebra de polinômios e formatação
# ════════════════════════════════════════════════════════════════════════════


def trim(c, tol: float = 1e-12) -> np.ndarray:
    c = np.atleast_1d(np.asarray(c, dtype=float))
    nz = np.flatnonzero(np.abs(c) > tol)
    return c[nz[0]:].copy() if nz.size else np.array([0.0])


def pmul(a, b):
    return trim(np.polymul(trim(a), trim(b)))


def padd(a, b):
    return trim(np.polyadd(trim(a), trim(b)))


def pscale(a, k):
    return trim(np.asarray(a, dtype=float) * k)


def clean_roots(r) -> np.ndarray:
    r = np.array(r, dtype=complex)
    if r.size:
        small = np.abs(r.imag) < 1e-7 * np.maximum(1.0, np.abs(r))
        r.imag[small] = 0.0
    return r


def proots(c) -> np.ndarray:
    c = trim(c)
    return clean_roots(np.roots(c)) if len(c) > 1 else np.array([], dtype=complex)


def poly_from_roots(roots) -> np.ndarray:
    roots = list(roots)
    if not roots:
        return np.array([1.0])
    return np.real(np.poly(roots))


def cancel_roots(zs, ps, tol: float = 5e-3):
    """Cancela pares polo-zero próximos. Devolve (zeros, polos, cancelados)."""
    zs, ps, canceled = list(zs), list(ps), []
    for i in range(len(zs) - 1, -1, -1):
        for j in range(len(ps) - 1, -1, -1):
            if abs(zs[i] - ps[j]) < tol:
                canceled.append(zs[i])
                zs.pop(i)
                ps.pop(j)
                break
    return zs, ps, canceled


def reduce_tf(num, den, tol: float = 5e-3):
    """Remove fatores comuns (cancelamentos polo-zero) de num/den."""
    num, den = trim(num), trim(den)
    if len(num) < 2 or len(den) < 2:
        return num, den
    zs, ps, canceled = cancel_roots(proots(num), proots(den), tol)
    if not canceled:
        return num, den
    k = num[0] / den[0]
    return trim(k * poly_from_roots(zs)), trim(poly_from_roots(ps))


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def tex_num(x: float, nd: int = 4, strip: bool = False) -> str:
    if not math.isfinite(x):
        return r"\infty" if x > 0 else r"-\infty"
    if x == 0:
        return "0"
    ax = abs(x)
    if 1e-3 <= ax < 1e7:
        s = f"{x:.{nd}f}"
        if strip and "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    m, e = f"{x:.3e}".split("e")
    return rf"{m}\times 10^{{{int(e)}}}"


def cx(c, nd: int = 4) -> str:
    """Número complexo em texto: 1.0000 + 2.0000j"""
    c = complex(c)
    if abs(c.imag) < 1e-6:
        return f"{c.real:.{nd}f}"
    return f"{c.real:.{nd}f} {'+' if c.imag >= 0 else '-'} {abs(c.imag):.{nd}f}j"


def frac(n: str, d: str) -> str:
    return rf"\frac{{{n}}}{{{d}}}"


def poly_tex(c, var: str = "s", nd: int = 4) -> str:
    c = trim(c)
    n = len(c) - 1
    terms = []
    for i, a in enumerate(c):
        if abs(a) < 1e-9:
            continue
        p = n - i
        mag = abs(a)
        coef = "" if (abs(mag - 1) < 1e-9 and p > 0) else tex_num(mag, nd, strip=True)
        v = "" if p == 0 else (var if p == 1 else f"{var}^{{{p}}}")
        terms.append(("-" if a < 0 else "+", coef + v))
    if not terms:
        return "0"
    out = ("-" if terms[0][0] == "-" else "") + terms[0][1]
    for sign, t in terms[1:]:
        out += f" {sign} {t}"
    return out


def factored_tex(c, var: str = "s", nd: int = 4) -> str:
    """Forma fatorada: K s (s+a)(s²+bs+c)^k, agrupando raízes repetidas e pares conjugados."""
    c = trim(c)
    lead = c[0]
    roots = list(proots(c))
    if abs(lead) < 1e-12:
        return "0"
    reals, comps = [], []
    for r in roots:
        (reals if abs(r.imag) < 1e-9 else comps).append(r)

    def group(items, key_close):
        groups = []
        for r in items:
            for g in groups:
                if key_close(r, g[0]):
                    g[1] += 1
                    break
            else:
                groups.append([r, 1])
        return groups

    rg = group(reals, lambda a, b: abs(a - b) < 1e-3)
    rg.sort(key=lambda g: (abs(g[0]) > 1e-6, g[0].real))
    cg = group([r for r in comps if r.imag > 0], lambda a, b: abs(a - b) < 1e-3)
    cg.sort(key=lambda g: g[0].real)

    head = "" if abs(lead - 1) < 1e-6 else ("-" if abs(lead + 1) < 1e-6 else tex_num(lead, nd))
    s = head
    for r, k in rg:
        p = f"^{{{k}}}" if k > 1 else ""
        if abs(r) < 1e-6:
            s += f"{var}{p}"
        else:
            a = -r.real
            s += f"({var} {'+' if a >= 0 else '-'} {tex_num(abs(a), nd, strip=True)}){p}"
    for r, k in cg:
        p = f"^{{{k}}}" if k > 1 else ""
        s += f"({poly_tex([1, -2 * r.real, abs(r) ** 2], var, nd)}){p}"
    return s or tex_num(lead, nd)


def tf_tex(num, den, factored: bool = False) -> str:
    if factored:
        return frac(factored_tex(num), factored_tex(den))
    return frac(poly_tex(num), poly_tex(den))


def tf_both_tex(name: str, num, den) -> str:
    return rf"{name} = {tf_tex(num, den)} = {tf_tex(num, den, True)}"


def fmt_array(c) -> str:
    """[4, 16] — formato aceito de volta por parse_array."""
    def f(x):
        s = f"{x:.8f}".rstrip("0").rstrip(".")
        return "0" if s in ("", "-0") else s

    return "[" + ", ".join(f(x) for x in c) + "]"


# ════════════════════════════════════════════════════════════════════════════
#  3. Memorial (lista de passos com blocos tipados)
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class Step:
    title: str
    blocks: list = field(default_factory=list)


class Memorial:
    """
    Tipos de bloco: md · tex · warn · ok · info · metrics · plot · divider
    """

    def __init__(self):
        self.steps: list[Step] = []
        self._b: list = []

    def md(self, t): self._b.append(("md", t))
    def tex(self, t): self._b.append(("tex", t))
    def warn(self, t): self._b.append(("warn", t))
    def ok(self, t): self._b.append(("ok", t))
    def info(self, t): self._b.append(("info", t))
    def metrics(self, items): self._b.append(("metrics", items))
    def plot(self, spec): self._b.append(("plot", spec))
    def divider(self): self._b.append(("divider", None))

    def commit(self, title: str):
        self.steps.append(Step(title, self._b))
        self._b = []


def memorial_to_markdown(title: str, steps: list[Step]) -> str:
    out = [f"# {title}", ""]
    for i, st_ in enumerate(steps, 1):
        out += [f"## {i}. {st_.title}", ""]
        for kind, p in st_.blocks:
            if kind == "md":
                out += [p, ""]
            elif kind == "tex":
                out += ["$$", p, "$$", ""]
            elif kind == "warn":
                out += [f"> ⚠️ {p}", ""]
            elif kind == "ok":
                out += [f"> ✅ {p}", ""]
            elif kind == "info":
                out += [f"> ℹ️ {p}", ""]
            elif kind == "metrics":
                out += [f"- **{m[0]}:** {m[1]}" for m in p] + [""]
    return "\n".join(out)


# ════════════════════════════════════════════════════════════════════════════
#  4. Resposta ao degrau (espaço de estados, discretização exata ZOH)
# ════════════════════════════════════════════════════════════════════════════


def _expm(M: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(M, 1)
    s = max(0, int(math.ceil(math.log2(nrm))) + 1) if nrm > 0 else 0
    A = M / (2 ** s)
    E = np.eye(M.shape[0])
    term = np.eye(M.shape[0])
    for k in range(1, 22):
        term = term @ A / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E


def percent_overshoot(zeta: float) -> float:
    if zeta >= 1:
        return 0.0
    if zeta <= 0:
        return math.inf
    return math.exp(-zeta * math.pi / math.sqrt(1 - zeta * zeta)) * 100


def step_metrics(num, den, ts_factor: float = 4, target_ts: float = 0, samples: int = 5000) -> dict:
    """Resposta ao degrau unitário de num(s)/den(s) + MP, ts e valor final."""
    empty = dict(mp=0.0, ts=0.0, final=0.0, peak=0.0, t=np.array([0.0]), y=np.array([0.0]), stable=True)
    num, den = trim(num).copy(), trim(den).copy()
    if len(den) == 0 or abs(den[0]) < 1e-15:
        return empty
    lead = den[0]
    den, num = den / lead, num / lead
    n = len(den) - 1
    if n == 0:
        v = float(num[-1])
        return dict(mp=0.0, ts=0.0, final=v, peak=v, t=np.array([0.0, 1.0]), y=np.array([v, v]), stable=True)

    if len(num) < n + 1:
        num = np.concatenate([np.zeros(n + 1 - len(num)), num])
    elif len(num) > n + 1:
        num = num[len(num) - n - 1:]

    b0 = num[0]
    a_tail = den[1:]
    c_vec = num[1:] - b0 * a_tail
    final = num[-1] / den[-1] if abs(den[-1]) > 1e-12 else 1.0

    poles = proots(den)
    decay = [-p.real for p in poles if p.real < -1e-6]
    slowest = min(decay) if decay else math.inf
    t_end = max(5.0, target_ts * 3 if target_ts else 0.0, 10 / slowest if math.isfinite(slowest) else 10.0)
    t_end = min(t_end, 1000.0)
    samples = max(400, int(samples))
    dt = t_end / samples

    A = np.zeros((n, n))
    A[0, :] = -a_tail
    if n > 1:
        A[1:, :-1] = np.eye(n - 1)
    B = np.zeros(n)
    B[0] = 1.0
    aug = np.zeros((n + 1, n + 1))
    aug[:n, :n] = A * dt
    aug[:n, n] = B * dt
    with np.errstate(all="ignore"):
        E = _expm(aug)
        Ad, Bd = E[:n, :n], E[:n, n]
        X = np.zeros((samples + 1, n))
        x = np.zeros(n)
        for k in range(1, samples + 1):
            x = Ad @ x + Bd
            X[k] = x
        y = X @ c_vec + b0
    t = np.arange(samples + 1) * dt

    if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
        return dict(mp=math.inf, ts=math.inf, final=final, peak=math.inf, t=t,
                    y=np.where(np.isfinite(y), y, np.nan), stable=False)

    peak = float(np.max(y))
    scale = max(abs(final), 1e-9)
    mp = max(0.0, (peak - final) / scale * 100)
    band = 0.02 if ts_factor == 4 else 0.05
    outside = np.flatnonzero(np.abs(y - final) > band * scale)
    if outside.size == 0:
        ts = 0.0
    else:
        last = outside[-1]
        ts = float(t[last + 1]) if last + 1 < len(t) else math.inf
    return dict(mp=mp, ts=ts, final=final, peak=peak, t=t, y=y, stable=True)


# ════════════════════════════════════════════════════════════════════════════
#  5. Sistema em malha fechada, controladores e ajuste de ganho
# ════════════════════════════════════════════════════════════════════════════

MOD_NAMES = {"pd": "PD", "pi": "PI", "pid": "PID", "1pole": "1 Polo (Atraso)", "custom": "Custom Gc(s)"}


def closed_loop(gN, gD, hN, hD, gcN, gcD):
    """T(s) = G·Gc / (1 + G·H·Gc), já sem cancelamentos polo-zero."""
    num = pmul(pmul(gN, gcN), hD)
    den = padd(pmul(pmul(gD, hD), gcD), pmul(pmul(gN, hN), gcN))
    return reduce_tf(num, den)


def build_controller(mod: str, kc: float, z: float = 0.0, p: float = 0.0, custom=None):
    """Monta (num, den) de Gc(s) a partir dos parâmetros de ajuste."""
    if mod == "pd":
        return trim([kc, kc * z]), np.array([1.0])
    if mod == "pi":
        return trim([kc, kc * z]), np.array([1.0, 0.0])
    if mod == "pid":
        return trim([kc, 2 * kc * z, kc * z * z]), np.array([1.0, 0.0])
    if mod == "1pole":
        return trim([kc]), np.array([1.0, p])
    if mod == "custom" and custom is not None:
        num, den = custom
        k0 = num[0] / den[0] if abs(den[0]) > 1e-12 else 1.0
        f = kc / k0 if abs(k0) > 1e-12 else 1.0
        return pscale(num, f), np.array(den, dtype=float)
    raise ValueError(f"Controlador desconhecido: {mod}")


def _ts_factor(spec: dict) -> float:
    return spec.get("ts_factor", 4)


def find_gain_tuning(plant: dict, gc_num, gc_den, spec: dict, base_kc: float):
    """
    Procura o menor fator de escala de Gc(s) que atende MP e ts simultaneamente.
    Retorna dict(scale, kc, metrics) ou None.
    """
    gN, gD, hN, hD = plant["gN"], plant["gD"], plant["hN"], plant["hD"]

    def check(scale, samples):
        n, d = closed_loop(gN, gD, hN, hD, pscale(gc_num, scale), gc_den)
        m = step_metrics(n, d, _ts_factor(spec), spec["ts"], samples)
        ok = (math.isfinite(m["mp"]) and math.isfinite(m["ts"])
              and m["mp"] <= spec["mp"] + 1e-3 and m["ts"] <= spec["ts"] + 1e-3)
        return ok, scale, m

    best = None
    for s in np.arange(0.20, 6.00 + 1e-9, 0.05):
        r = check(round(float(s), 4), 1200)
        if r[0]:
            best = r
            break
    if best is None:
        for s in np.arange(6.10, 15.00 + 1e-9, 0.20):
            r = check(round(float(s), 4), 1000)
            if r[0]:
                best = r
                break
    if best is None:
        return None

    refined = best
    for s in np.arange(max(0.05, best[1] - 0.08), best[1] + 0.08 + 1e-9, 0.005):
        r = check(round(float(s), 5), 2000)
        if r[0]:
            refined = r
            break
    return dict(scale=refined[1], kc=base_kc * refined[1], metrics=refined[2])


def root_locus_points(total_num, total_den, s_max: float = 60.0, n_lin: int = 260, n_log: int = 140):
    """Pontos do LGR: raízes de den + s·num para s∈[0, s_max] (s=1 é o projeto atual)."""
    scales = np.concatenate([np.linspace(0, 4, n_lin), np.geomspace(4, s_max, n_log)[1:]])
    pts = []
    for s in scales:
        pts.extend(proots(padd(total_den, pscale(total_num, s))))
    return np.array(pts, dtype=complex)


# ════════════════════════════════════════════════════════════════════════════
#  6. Projeto do controlador por LGR (memorial passo a passo)
# ════════════════════════════════════════════════════════════════════════════


def _choose_branch(mem: Memorial, plant_angle, extra, double, sigma, wd):
    """Escolhe entre φ = ±180° - ∠GH + extra o ramo que dá zero/polo estável."""
    opt1 = 180 - plant_angle + extra
    opt2 = -180 - plant_angle + extra
    cands = sorted([opt1, opt2], key=abs)
    for cand in cands:
        test = cand / 2 if double else cand
        t = math.tan(math.radians(test))
        if abs(t) < 1e-8:
            continue
        z = sigma + wd / t
        if z >= 0:
            return dict(opt1=opt1, opt2=opt2, phi=cand, z=z, stable=True)
    cand = cands[0]
    test = cand / 2 if double else cand
    t = math.tan(math.radians(test))
    if abs(t) < 1e-8:
        mem.warn(r"Aviso: tan(φ) ≈ 0 — a raiz degenera (foi limitada a 10⁶).")
        z = 1e6
    else:
        z = sigma + wd / t
    return dict(opt1=opt1, opt2=opt2, phi=cand, z=z, stable=False)


def solve_design(cfg: dict):
    """
    cfg: mod, gN, gD, hN, hD, spec, [gcN, gcD], [gain_scale], [template_id]
    Retorna (steps, design, suggestion). Lança ValueError em dados inválidos.
    """
    mem = Memorial()
    mod = cfg["mod"]
    gN, gD, hN, hD = (np.array(cfg[k], dtype=float) for k in ("gN", "gD", "hN", "hD"))
    spec_in = cfg["spec"]
    gain_scale = cfg.get("gain_scale") or 1.0
    template_id = cfg.get("template_id")

    if len(gD) < 1 or abs(gD[0]) < 1e-15 or abs(hD[0]) < 1e-15:
        raise ValueError("O denominador de G(s) e de H(s) não pode ser zero.")
    if abs(gN[0]) < 1e-15 or abs(hN[0]) < 1e-15:
        raise ValueError("O numerador de G(s) e de H(s) não pode ser zero.")

    # ── Passo: Dados entendidos ────────────────────────────────────────────
    mem.md("**Planta** $G(s)$")
    mem.tex(rf"G(s) = {tf_tex(gN, gD)}")
    mem.md("**Realimentação** $H(s)$")
    mem.tex(rf"H(s) = {tf_tex(hN, hD)}")
    mem.md(f"**Controlador selecionado:** :blue[**{MOD_NAMES[mod]}**]")

    kind = spec_in["type"]
    if kind == "targets":
        mp, ts, tsf = spec_in["mp"], spec_in["ts"], spec_in["ts_factor"]
        if not 0 < mp < 100:
            raise ValueError("MP (%) deve estar entre 0 e 100 (exclusive).")
        if ts <= 0:
            raise ValueError("O tempo de acomodação ts deve ser positivo.")
        mem.md(f"**Especificações mapeadas:** overshoot **MP = {mp:g}%** · tempo de acomodação **ts = {ts:g} s** "
               f"(critério de {'2%' if tsf == 4 else '5%'})")
    elif kind == "params":
        zeta, wn = spec_in["zeta"], spec_in["wn"]
        if not 0 < zeta < 1:
            raise ValueError("ζ deve estar entre 0 e 1 (sistema subamortecido).")
        if wn <= 0:
            raise ValueError("ωn deve ser positivo.")
        mem.md(f"**Especificações mapeadas:** ζ = **{zeta:g}** · ωn = **{wn:g} rad/s**")
    else:
        if spec_in["im"] <= 0:
            raise ValueError("A parte imaginária do polo desejado deve ser positiva (ωd > 0).")
        if spec_in["re"] >= 0:
            raise ValueError("A parte real do polo desejado deve ser negativa (semiplano esquerdo).")
        mem.md(f"**Especificações mapeadas:** polo dominante direto = **{spec_in['re']:g} ± {spec_in['im']:g}j**")
    mem.commit("Dados entendidos pelo app")

    # ── Passo: Especificações → polo dominante ─────────────────────────────
    zeta_plot = wn_plot = None
    if kind == "targets":
        ln = math.log(mp / 100)
        zeta = math.sqrt(ln ** 2 / (math.pi ** 2 + ln ** 2))
        wn = tsf / (zeta * ts)
        wd = wn * math.sqrt(1 - zeta ** 2)
        sd = complex(-zeta * wn, wd)
        spec = dict(type="targets", mp=mp, ts=ts, ts_factor=tsf, zeta=zeta, wn=wn, wd=wd)
        mem.md(f"A partir de MP = {mp:g}% e ts = {ts:g} s (critério {'2%' if tsf == 4 else '5%'}):")
        mem.tex(rf"\zeta = \frac{{|\ln(MP/100)|}}{{\sqrt{{\pi^2 + \ln^2(MP/100)}}}} = {zeta:.4f}")
        mem.tex(rf"\omega_n = \frac{{{tsf:g}}}{{\zeta\, t_s}} = {wn:.4f}\ \text{{rad/s}}")
        mem.tex(rf"\omega_d = \omega_n\sqrt{{1-\zeta^2}} = {wd:.4f}\ \text{{rad/s}}")
        zeta_plot, wn_plot = zeta, wn
    elif kind == "params":
        zeta, wn = spec_in["zeta"], spec_in["wn"]
        wd = wn * math.sqrt(1 - zeta ** 2)
        sd = complex(-zeta * wn, wd)
        spec = dict(type="params", zeta=zeta, wn=wn, wd=wd)
        mem.tex(rf"\zeta = {zeta:.4f}, \qquad \omega_n = {wn:.4f}\ \text{{rad/s}}")
        mem.tex(rf"\omega_d = \omega_n\sqrt{{1-\zeta^2}} = {wd:.4f}\ \text{{rad/s}}")
        zeta_plot, wn_plot = zeta, wn
    else:
        sd = complex(spec_in["re"], spec_in["im"])
        spec = dict(type="poles", re=sd.real, im=sd.imag)
    mem.ok(f"Polo dominante desejado:  s_d = {cx(sd)}")
    mem.plot(dict(kind="splane", title="Polo dominante desejado no plano s", poles=[], zeros=[], sd=sd,
                  zeta=zeta_plot, wn=wn_plot))
    mem.commit("Especificações de desempenho")

    # ── Passo: Fatoração de polos e zeros ──────────────────────────────────
    plantN, plantD = pmul(gN, hN), pmul(gD, hD)
    mem.md("**Malha aberta** $G(s)H(s)$")
    mem.tex(tf_both_tex("G(s)H(s)", plantN, plantD))

    z_all, p_all = proots(plantN), proots(plantD)
    zR, pR, canceled = cancel_roots(z_all, p_all)
    if canceled:
        mem.warn("Cancelamento polo-zero detectado em: " + ", ".join(cx(c) for c in canceled))
        k_lead = plantN[0] / plantD[0]
        plantN = trim(k_lead * poly_from_roots(zR))
        plantD = trim(poly_from_roots(pR))
        mem.md("**G(s)H(s) simplificado**")
        mem.tex(tf_both_tex("G(s)H(s)", plantN, plantD))
        mem.info("Os modos cancelados continuam existindo internamente na planta; o LGR usa apenas a função simplificada.")

    mem.md("**Zeros de malha aberta**\n\n" + ("\n".join(f"- $z_{{{i + 1}}}$ = {cx(r)}" for i, r in enumerate(zR)) if zR else "Nenhum"))
    mem.md("**Polos de malha aberta**\n\n" + ("\n".join(f"- $p_{{{i + 1}}}$ = {cx(r)}" for i, r in enumerate(pR)) if pR else "Nenhum"))
    mem.plot(dict(kind="splane", title="Polos e zeros de malha aberta", poles=pR, zeros=zR, sd=None))
    mem.commit("Fatoração de polos e zeros")

    plant = dict(gN=gN, gD=gD, hN=hN, hD=hD)
    k_plant = (gN[0] * hN[0]) / (gD[0] * hD[0])

    # ── Controlador customizado: não há critério de ângulo ─────────────────
    if mod == "custom":
        gcN, gcD = trim(cfg["gcN"]), trim(cfg["gcD"])
        if abs(gcD[0]) < 1e-15:
            raise ValueError("O denominador de Gc(s) não pode ser zero.")
        mem.md("O controlador é fornecido diretamente por você:")
        mem.tex(tf_both_tex("G_c(s)", gcN, gcD))
        mem.commit("Definição do controlador")

        totN, totD = pmul(plantN, gcN), pmul(plantD, gcD)
        kc = float(gcN[0] / gcD[0])
        design = _make_design(cfg, plant, plantN, plantD, gcN, gcD, totN, totD, kc, kc, 0.0, 0.0, spec, sd,
                              f"Projeto customizado", template_id)
        _validate(mem, design, sd, spec, kc, gain_scale=1.0)
        return mem.steps, design, design.pop("_suggestion")

    # ── Passo: Critério de ângulo ──────────────────────────────────────────
    zero_angles = [math.degrees(cmath.phase(sd - r)) for r in zR]
    pole_angles = [math.degrees(cmath.phase(sd - r)) for r in pR]
    for i, a in enumerate(zero_angles):
        mem.tex(rf"\alpha_{{{i + 1}}} = \angle(s_d - z_{{{i + 1}}}) = {a:.2f}^\circ")
    if not zR:
        mem.md("Nenhum zero contribui angularmente.")
    for i, a in enumerate(pole_angles):
        mem.tex(rf"\theta_{{{i + 1}}} = \angle(s_d - p_{{{i + 1}}}) = {a:.2f}^\circ")
    if not pR:
        mem.md("Nenhum polo contribui angularmente.")
    plant_angle = sum(zero_angles) - sum(pole_angles)
    zs = " + ".join(f"{a:.2f}^\\circ" for a in zero_angles) or "0^\\circ"
    ps = " + ".join(f"{a:.2f}^\\circ" for a in pole_angles) or "0^\\circ"
    mem.tex(r"\angle G(s_d)H(s_d) = \Big(\sum \alpha_i\Big) - \Big(\sum \theta_j\Big)")
    mem.tex(rf"\angle G(s_d)H(s_d) = ({zs}) - ({ps}) = {plant_angle:.2f}^\circ")
    mem.plot(dict(kind="angles", title="Vetores de s_d até cada polo e zero", poles=pR, zeros=zR, sd=sd))
    mem.commit("Aplicação do critério de ângulo")

    # ── Passo: Cálculo do zero/polo do controlador ─────────────────────────
    sigma, wd_ = -sd.real, sd.imag
    c_zeros, c_poles = [], []
    z = p = 0.0
    sign_z = abs_z = sign_p = abs_p = None
    phi_txt = rf"z = \sigma_d + \frac{{\omega_d}}{{\tan\varphi}}"

    if mod in ("pd", "pi", "pid"):
        extra, double = 0.0, False
        if mod == "pd":
            mem.md("**Condição angular do LGR**")
            mem.tex(r"\angle G(s_d)H(s_d) + \varphi = \pm 180^\circ")
        else:
            th_c = math.degrees(cmath.phase(sd))
            extra = th_c
            if mod == "pi":
                mem.md("O PI adiciona um **polo na origem** ($s=0$).")
                mem.tex(rf"\theta_c = \angle(s_d - 0) = {th_c:.2f}^\circ")
                mem.tex(r"\angle G(s_d)H(s_d) - \theta_c + \varphi = \pm 180^\circ")
            else:
                double = True
                mem.md("O PID adiciona um **polo na origem** e **dois zeros idênticos** ($z_1 = z_2$).")
                mem.tex(rf"\theta_c = \angle(s_d - 0) = {th_c:.2f}^\circ")
                mem.tex(r"\sum\alpha_i - \sum\theta_j - \theta_c + 2\varphi = \pm 180^\circ")

        res = _choose_branch(mem, plant_angle, extra, double, sigma, wd_)
        z = res["z"]
        label = "2\\varphi" if double else "\\varphi"
        mem.md(f"Opções para ${label}$: **{res['opt1']:.2f}°** ou **{res['opt2']:.2f}°**")
        phi = res["phi"] / 2 if double else res["phi"]
        if double:
            mem.md(f"Ramo escolhido: $2\\varphi = {res['phi']:.2f}^\\circ \\Rightarrow$ :blue[**φ = {phi:.2f}°**]")
        else:
            mem.md(f"Ramo escolhido (estabilidade / menor módulo): :blue[**φ = {phi:.2f}°**]")
        if not res["stable"]:
            mem.warn("Nenhum ramo produziu um zero no semiplano esquerdo; o controlador resultante tem zero instável.")
        mem.tex(phi_txt)
        mem.tex(rf"z = {sigma:.4f} + \frac{{{wd_:.4f}}}{{\tan({phi:.2f}^\circ)}} = {z:.4f}")
        sign_z, abs_z = ("+" if z >= 0 else "-"), f"{abs(z):.4f}"
        c_zeros = [-z, -z] if mod == "pid" else [-z]
        c_poles = [0.0] if mod in ("pi", "pid") else []
        title = {"pd": "Controlador: cálculo do zero", "pi": "Controlador: cálculo do zero",
                 "pid": "Controlador: cálculo dos zeros"}[mod]
        mem.ok(("Zeros do PID: z₁ = z₂ = " if mod == "pid" else "Zero do controlador: z = ") + f"{z:.4f}")
        mem.commit(title)
    else:  # 1 polo
        mem.md("O controlador tem **um polo e nenhum zero** explícito:")
        mem.tex(r"G_c(s) = \frac{a}{s + p}")
        mem.tex(r"\angle G(s_d)H(s_d) - \theta_p = \pm 180^\circ")
        opt1, opt2 = plant_angle - 180, plant_angle + 180
        cands = sorted([opt1, opt2], key=abs)
        theta_p, found = cands[0], False
        for cand in cands:
            t = math.tan(math.radians(cand))
            if abs(t) < 1e-8:
                continue
            tp = sigma + wd_ / t
            if tp >= 0:
                theta_p, p, found = cand, tp, True
                break
        if not found:
            theta_p = cands[0]
            t = math.tan(math.radians(theta_p))
            if abs(t) < 1e-8:
                mem.warn("Aviso: tan(θp) ≈ 0 — o polo degenera (limitado a 10⁶).")
                p = 1e6
            else:
                p = sigma + wd_ / t
            mem.warn("Nenhum ramo produziu um polo no semiplano esquerdo; o controlador resultante é instável.")
        mem.md(f"Opções para $\\theta_p$: **{opt1:.2f}°** ou **{opt2:.2f}°** → :blue[**θp = {theta_p:.2f}°**]")
        mem.tex(r"p = \sigma_d + \frac{\omega_d}{\tan\theta_p}")
        mem.tex(rf"p = {sigma:.4f} + \frac{{{wd_:.4f}}}{{\tan({theta_p:.2f}^\circ)}} = {p:.4f}")
        sign_p, abs_p = ("+" if p >= 0 else "-"), f"{abs(p):.4f}"
        c_poles = [-p]
        mem.ok(f"Polo do controlador: p = {p:.4f}")
        mem.commit("Controlador: cálculo do polo")

    # ── Passo: Critério de módulo ──────────────────────────────────────────
    all_z = list(zR) + [complex(x) for x in c_zeros]
    all_p = list(pR) + [complex(x) for x in c_poles]
    mem.tex(r"|G(s_d)H(s_d)G_c(s_d)| = 1")
    prod_z = 1.0
    rows = []
    for i, r in enumerate(all_z):
        d = abs(sd - r)
        prod_z *= d
        rows.append(rf"|s_d - z_{{{i + 1}}}| = {d:.4f}")
    prod_p = 1.0
    for i, r in enumerate(all_p):
        d = abs(sd - r)
        prod_p *= d
        rows.append(rf"|s_d - p_{{{i + 1}}}| = {d:.4f}")
    if rows:
        mem.tex(r"\begin{aligned}" + r"\\".join(rows) + r"\end{aligned}")
    kt = prod_p / prod_z
    mem.tex(rf"K_t = \frac{{\prod |s_d - p_j|}}{{\prod |s_d - z_i|}} = \frac{{{prod_p:.4f}}}{{{prod_z:.4f}}} = {kt:.4f}")
    mem.md(f"Ganho estático (coef. líderes) da planta: **K = {k_plant:.4f}**")
    kc_base = kt / k_plant
    kc = kc_base * gain_scale
    mem.tex(rf"K_c = \frac{{K_t}}{{K}} = \frac{{{kt:.4f}}}{{{k_plant:.4f}}} = {kc_base:.4f}")
    if abs(gain_scale - 1) > 1e-9:
        mem.md(f"Ajuste fino de ganho aplicado: fator = **{gain_scale:.4f}**")
        mem.tex(rf"K_c^{{(ajustado)}} = {kc_base:.4f}\cdot {gain_scale:.4f} = {kc:.4f}")

    gcN = pscale(poly_from_roots(c_zeros), kc)
    gcD = poly_from_roots(c_poles)
    totN, totD = pmul(plantN, gcN), pmul(plantD, gcD)
    label_t = f"Q{template_id}" if template_id else "Projeto manual"

    if mod == "pd":
        gc_final = rf"G_c(s) = {kc:.4f}\,(s {sign_z} {abs_z})"
    elif mod == "pi":
        gc_final = rf"G_c(s) = {kc:.4f}\,\frac{{s {sign_z} {abs_z}}}{{s}}"
    elif mod == "pid":
        gc_final = rf"G_c(s) = {kc:.4f}\,\frac{{(s {sign_z} {abs_z})^2}}{{s}}"
    else:
        gc_final = rf"G_c(s) = \frac{{{kc:.4f}}}{{s {sign_p} {abs_p}}}"
    mem.md(f"**Controlador {MOD_NAMES[mod]} final**")
    mem.tex(gc_final)
    mem.tex(tf_both_tex("G_c(s)", gcN, gcD))
    mem.commit("Critério de módulo e Gc(s) final")

    design = _make_design(cfg, plant, plantN, plantD, gcN, gcD, totN, totD, kc_base, kc, z, p, spec, sd,
                          f"{label_t} - {MOD_NAMES[mod]}", template_id)
    _validate(mem, design, sd, spec, kc, gain_scale)
    return mem.steps, design, design.pop("_suggestion")


def _make_design(cfg, plant, plantN, plantD, gcN, gcD, totN, totD, kc_base, kc, z, p, spec, sd, label, template_id):
    d = dict(
        mod=cfg["mod"], plant=plant, plantN=plantN, plantD=plantD,
        gcN=gcN, gcD=gcD, totN=totN, totD=totD,
        kc=kc, kc_base=kc_base, z=z, p=p, spec=spec, sd=sd,
        label=label, template_id=template_id, tuned=False,
        calc=dict(kc=kc, z=z, p=p),
        custom=(np.array(cfg["gcN"], float), np.array(cfg["gcD"], float)) if cfg["mod"] == "custom" else None,
        gain_scale=cfg.get("gain_scale") or 1.0,
        id=uuid.uuid4().hex,
    )
    return d


def _validate(mem: Memorial, d: dict, sd: complex, spec: dict, kc: float, gain_scale: float):
    """Último passo: valida polos de malha fechada e resposta ao degrau; sugere ajuste de ganho."""
    totN, totD = d["totN"], d["totD"]
    plant = d["plant"]
    char = padd(totD, totN)
    cl_poles = proots(char)
    d["_suggestion"] = None
    if cl_poles.size == 0:
        mem.warn("Não foi possível calcular os polos de malha fechada para a validação.")
        mem.commit("Validação do projeto")
        return

    obtained = min(cl_poles, key=lambda r: abs(r - sd))
    err_re = abs(obtained.real - sd.real)
    err_im = abs(abs(obtained.imag) - abs(sd.imag))
    wn_o = abs(obtained)
    zeta_o = -obtained.real / wn_o if wn_o > 1e-12 else math.nan
    wd_o = abs(obtained.imag)
    mp_o = percent_overshoot(zeta_o) if not math.isnan(zeta_o) else math.nan
    ts2 = 4 / (zeta_o * wn_o) if zeta_o > 0 and wn_o > 0 else math.inf
    ts5 = 3 / (zeta_o * wn_o) if zeta_o > 0 and wn_o > 0 else math.inf
    ok = err_re <= 5e-2 and err_im <= 5e-2

    mem.md("**Equação característica de malha fechada**")
    mem.tex(r"1 + G(s)H(s)G_c(s) = 0")
    mem.tex(rf"{poly_tex(totD)} + \left({poly_tex(totN)}\right) = 0")
    mem.md("**Polos de malha fechada**\n\n" + "\n".join(f"- {cx(r)}" for r in cl_poles))
    mem.metrics([("Polo desejado", cx(sd)), ("Polo obtido", cx(obtained)),
                 ("Erro Re / Im", f"{err_re:.1e} / {err_im:.1e}")])

    closed_n, closed_d = closed_loop(plant["gN"], plant["gD"], plant["hN"], plant["hD"], d["gcN"], d["gcD"])
    sm = step_metrics(closed_n, closed_d, _ts_factor(spec), spec.get("ts", 0) or 0, 3000)
    step_spec = dict(kind="step", title="Resposta ao degrau em malha fechada", t=sm["t"], y=sm["y"],
                     final=sm["final"], mp=spec.get("mp"), ts=spec.get("ts"))
    suggestion = None

    if spec["type"] == "targets":
        tol = 1e-3
        mp_ok = sm["mp"] <= spec["mp"] + tol
        ts_ok = sm["ts"] <= spec["ts"] + tol
        ok = mp_ok and ts_ok
        crit = "2%" if spec["ts_factor"] == 4 else "5%"
        ts_o = ts2 if spec["ts_factor"] == 4 else ts5
        mem.md("**Parâmetros calculados a partir do polo obtido**")
        mem.tex(rf"\zeta = \frac{{-\mathrm{{Re}}(s)}}{{|s|}} = {zeta_o:.4f},\qquad \omega_n = |s| = {wn_o:.4f}\ \text{{rad/s}},"
                rf"\qquad \omega_d = |\mathrm{{Im}}(s)| = {wd_o:.4f}\ \text{{rad/s}}")
        mem.md(f"Estimativa de 2ª ordem pelo polo: MP = {mp_o:.4f}%, ts({crit}) = {ts_o:.4f} s")
        mem.md("**Validação pela resposta ao degrau simulada**")
        ts_txt = f"{sm['ts']:.3f} s" if math.isfinite(sm["ts"]) else "não acomodou"
        mp_txt = f"{sm['mp']:.3f}%" if math.isfinite(sm["mp"]) else "instável"
        mem.metrics([
            ("Overshoot simulado", mp_txt, f"limite {spec['mp']:g}%  →  {'✅' if mp_ok else '⚠️'}"),
            (f"ts ({crit}) simulado", ts_txt, f"limite {spec['ts']:g} s  →  {'✅' if ts_ok else '⚠️'}"),
            ("Valor final", f"{sm['final']:.4f}", ""),
        ])
        mem.plot(step_spec)
        if not ok:
            mem.warn("O polo foi posicionado pelo LGR, mas a resposta ao degrau não atende totalmente às especificações. "
                     "Isso é comum quando o controlador ou a planta introduzem zeros relevantes (a aproximação de "
                     "2ª ordem deixa de valer).")
            tuning = find_gain_tuning(plant, d["gcN"], d["gcD"], spec, kc)
            if tuning:
                abs_scale = gain_scale * tuning["scale"]
                suggestion = dict(scale=abs_scale, kc=tuning["kc"], metrics=tuning["metrics"])
                mem.md("**Sugestão de ajuste fino em ganho**")
                mem.metrics([("Kc atual", f"{kc:.4f}"), ("Kc recomendado", f"{tuning['kc']:.4f}"),
                             ("Fator sobre o atual", f"{tuning['scale']:.4f}")])
                mem.md(f"Com esse ajuste: MP ≈ **{tuning['metrics']['mp']:.3f}%** e ts({crit}) ≈ "
                       f"**{tuning['metrics']['ts']:.3f} s**.")
            else:
                mem.warn("Não foi encontrado Kc na faixa de busca automática que satisfaça MP e ts simultaneamente. "
                         "Tente outro tipo de controlador ou relaxe as especificações.")
    elif spec["type"] == "params":
        z_ok = abs(zeta_o - spec["zeta"]) <= 5e-2
        w_ok = abs(wn_o - spec["wn"]) <= 5e-2
        ok = ok and z_ok and w_ok
        mem.md("**Parâmetros calculados a partir do polo obtido**")
        mem.metrics([("ζ obtido", f"{zeta_o:.4f}", f"alvo {spec['zeta']:.4f}  →  {'✅' if z_ok else '⚠️'}"),
                     ("ωn obtido", f"{wn_o:.4f} rad/s", f"alvo {spec['wn']:.4f}  →  {'✅' if w_ok else '⚠️'}"),
                     ("ωd obtido", f"{wd_o:.4f} rad/s", "")])
        mem.plot(step_spec)
    else:
        pole_ok = err_re <= 5e-2 and err_im <= 5e-2
        ok = ok and pole_ok
        mem.md(f"Comparação direta com o polo solicitado: {'✅ atendido' if pole_ok else '⚠️ atenção'}")
        mem.plot(step_spec)

    mem.plot(dict(kind="locus", title="Lugar geométrico das raízes (G·H·Gc)", points=root_locus_points(totN, totD),
                  olp=proots(totD), olz=proots(totN), cl=cl_poles, sd=sd))
    (mem.ok if ok else mem.warn)("Status final da validação: " + ("especificações atendidas ✅" if ok else "atenção ⚠️ — veja os avisos acima"))
    mem.commit("Validação do projeto")
    d["_suggestion"] = suggestion


# ════════════════════════════════════════════════════════════════════════════
#  7. Ajuste fino (simulação com parâmetros livres)
# ════════════════════════════════════════════════════════════════════════════


def tuning_response(design: dict, kc: float, z: float, p: float, samples: int = 1500) -> dict:
    gcN, gcD = build_controller(design["mod"], kc, z, p, design.get("custom"))
    pl = design["plant"]
    n, dn = closed_loop(pl["gN"], pl["gD"], pl["hN"], pl["hD"], gcN, gcD)
    spec = design["spec"]
    m = step_metrics(n, dn, _ts_factor(spec), spec.get("ts", 0) or 0, samples)
    poles = proots(dn)
    m["poles"] = poles
    m["is_stable"] = bool(m["stable"] and (poles.size == 0 or np.all(poles.real < 1e-9)))
    m["gcN"], m["gcD"] = gcN, gcD
    return m


def apply_tuning(design: dict, kc: float, z: float, p: float) -> dict:
    gcN, gcD = build_controller(design["mod"], kc, z, p, design.get("custom"))
    design = dict(design)
    design.update(kc=kc, z=z, p=p, gcN=gcN, gcD=gcD,
                  totN=pmul(design["plantN"], gcN), totD=pmul(design["plantD"], gcD), tuned=True)
    return design


# ════════════════════════════════════════════════════════════════════════════
#  8. Discretização (Tustin / Euler forward)
# ════════════════════════════════════════════════════════════════════════════


def discretize(fN, fD, method: str, T: float):
    """Substitui s por (Az+B)/(Cz+D) e normaliza para denominador mônico."""
    if method == "tustin":
        A, B, C, D = 2.0, -2.0, T, T
    elif method == "euler":
        A, B, C, D = 1.0, -1.0, 0.0, T
    else:
        raise ValueError("Método desconhecido")
    fN, fD = trim(fN), trim(fD)
    n_num, n_den = len(fN) - 1, len(fD) - 1
    n_max = max(n_num, n_den)
    fa, fb = trim([A, B]), trim([C, D])

    def conv(f, n):
        res = np.array([0.0])
        for k in range(n + 1):
            t1 = np.array([1.0])
            for _ in range(k):
                t1 = pmul(t1, fa)
            t2 = np.array([1.0])
            for _ in range(n_max - k):
                t2 = pmul(t2, fb)
            res = padd(res, f[n - k] * pmul(t1, t2))
        return res

    nz, dz = conv(fN, n_num), conv(fD, n_den)
    if abs(dz[0]) < 1e-15:
        raise ValueError("Denominador discretizado nulo — verifique F(s).")
    lead = dz[0]
    return trim(nz / lead), trim(dz / lead)


def disc_defaults(template_id, source: str):
    """Defaults didáticos dos templates Q5 (controlador) e Q6 (sistema completo)."""
    if str(template_id) == "5" and source == "controller":
        return dict(method="tustin", T="2.0")
    if str(template_id) == "6" and source == "total":
        return dict(method="euler", T="1.0")
    return None


def solve_disc(fN, fD, method: str, T: float, origin: str):
    if T <= 0:
        raise ValueError("O período de amostragem T deve ser positivo.")
    mem = Memorial()
    mname = "Tustin (Bilinear)" if method == "tustin" else "Euler (Forward)"
    mem.md(f"**Origem da função:** {origin}")
    mem.md("**Função a ser discretizada**")
    mem.tex(rf"F(s) = {tf_tex(fN, fD)}")
    mem.metrics([("Método", mname), ("Período T", f"{T:g} s"), ("Taxa de amostragem", f"{1 / T:.4g} Hz")])
    mem.commit("Validação dos dados")

    if method == "tustin":
        mem.tex(rf"s \;\longrightarrow\; \frac{{2}}{{T}}\,\frac{{z-1}}{{z+1}} = \frac{{2}}{{{T:g}}}\,\frac{{z-1}}{{z+1}}")
        mem.info("A transformação bilinear preserva a estabilidade (mapeia o semiplano esquerdo para dentro do círculo unitário).")
    else:
        mem.tex(rf"s \;\longrightarrow\; \frac{{z-1}}{{T}} = \frac{{z-1}}{{{T:g}}}")
        mem.warn("Euler forward pode instabilizar sistemas estáveis se T for grande em relação às constantes de tempo.")
    mem.commit("Método e substituição")

    nz, dz = discretize(fN, fD, method, T)
    mem.md("Após substituir, multiplicar por $(Cz+D)^{n}$ e normalizar o denominador (mônico):")
    mem.tex(rf"F(z) = {frac(poly_tex(nz, 'z'), poly_tex(dz, 'z'))}")
    mem.tex(rf"F(z) = {frac(factored_tex(nz, 'z'), factored_tex(dz, 'z'))}")
    mem.commit("Normalização mônica e F(z) final")

    # Equação de diferenças
    m = len(dz) - 1
    if len(nz) - 1 > m:
        mem.warn("F(z) é imprópria (grau do numerador maior que o do denominador): não é realizável como equação de diferenças causal.")
    else:
        b = np.concatenate([np.zeros(m + 1 - len(nz)), nz])
        a = dz
        terms_u = [f"{tex_num(b[i], 4)}\\,u[k{'-' + str(i) if i else ''}]" for i in range(m + 1) if abs(b[i]) > 1e-12]
        terms_y = [f"{tex_num(-a[i], 4)}\\,y[k-{i}]" for i in range(1, m + 1) if abs(a[i]) > 1e-12]
        rhs = " + ".join(terms_u + terms_y).replace("+ -", "- ") or "0"
        mem.md("Com $F(z)=Y(z)/U(z)$ e $z^{-1}$ como atraso de uma amostra:")
        mem.tex(rf"y[k] = {rhs}")
        mem.info("Use esta equação diretamente no seu código (CLP, microcontrolador, Simulink...). Cada `u[k-i]` é a entrada i amostras atrás.")
    pz, zz = proots(dz), proots(nz)
    stable = bool(pz.size == 0 or np.all(np.abs(pz) < 1 - 1e-9))
    (mem.ok if stable else mem.warn)(
        "Polos de F(z): " + (", ".join(cx(r) for r in pz) if pz.size else "nenhum")
        + f"  →  {'todos dentro do círculo unitário (estável)' if stable else 'há polo fora/sobre o círculo unitário'}")
    mem.plot(dict(kind="zplane", title="Polos e zeros de F(z)", poles=pz, zeros=zz))
    mem.commit("Equação de diferenças e estabilidade")
    return mem.steps, dict(num=nz, den=dz, stable=stable)


# ════════════════════════════════════════════════════════════════════════════
#  9. Templates (Q1–Q6) do app original
# ════════════════════════════════════════════════════════════════════════════

TEMPLATES = {
    "1": dict(name="Q1 · PD — metas MP/ts", mod="pd", gN="[4, 16]", gD="[1, 4, 4, 0]", hN="[1]", hD="[1]",
              spec_type="targets", mp="10", crit=3, ts="4"),
    "2": dict(name="Q2 · PD — parâmetros ζ/ωn (planta instável)", mod="pd", gN="[1]", gD="[10000, 0, -11772]",
              hN="[1]", hD="[1]", spec_type="params", zeta="0.7", wn="0.5"),
    "3": dict(name="Q3 · PI — polos desejados", mod="pi", gN="[5, 25, 20]", gD="[1, 4, 4]", hN="[0.2]", hD="[1, 1]",
              spec_type="poles", re="-4", im="4"),
    "4": dict(name="Q4 · PID — metas MP/ts", mod="pid", gN="[5]", gD="[1, 12, 22, 20]", hN="[0.4]", hD="[1]",
              spec_type="targets", mp="20", crit=4, ts="5"),
    "5": dict(name="Q5 · PID — com discretização Tustin", mod="pid", gN="[5, 15]", gD="[1, 4, 0]", hN="[1]",
              hD="[1, 1]", spec_type="targets", mp="10", crit=3, ts="3"),
    "6": dict(name="Q6 · 1 Polo — com discretização Euler", mod="1pole", gN="[2, 2]", gD="[1, 2, 2]", hN="[1, 3]",
              hD="[1, 5]", spec_type="poles", re="-2.5", im="2"),
}
