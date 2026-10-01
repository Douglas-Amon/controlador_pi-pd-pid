"""
Solver LGR — Controle e Automação  ·  versão Streamlit
=======================================================
Tradução interativa do app HTML original.

    pip install streamlit numpy plotly
    streamlit run app.py

A matemática fica em `lgr_core.py` (numpy puro); este arquivo cuida só da interface.
"""
from __future__ import annotations

import html
import inspect
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import lgr_core as core

st.set_page_config(page_title="Solver LGR · Controle e Automação", page_icon="🎛️", layout="wide")


# ════════════════════════════════════════════════════════════════════════════
#  Compatibilidade entre versões do Streamlit (use_container_width → width)
# ════════════════════════════════════════════════════════════════════════════
def _stretch(fn) -> dict:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    return {"width": "stretch"} if "width" in params else {"use_container_width": True}


BTN = _stretch(st.button)
CHART = _stretch(st.plotly_chart)
DL = _stretch(st.download_button)

# ════════════════════════════════════════════════════════════════════════════
#  Estilo
# ════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
:root{--bg:#0f172a;--panel:#1e293b;--muted:#94a3b8;--primary:#3b82f6;--accent:#10b981;--border:#334155;}
.block-container{padding-top:2.2rem;max-width:1180px;}
.hero{background:linear-gradient(135deg,#1e293b 0%,#0f172a 60%,#172554 100%);border:1px solid var(--border);
      border-radius:18px;padding:22px 28px;margin-bottom:18px;}
.hero h1{margin:0;font-size:1.9rem;color:#f8fafc;letter-spacing:-.5px;}
.hero p{margin:6px 0 0;color:var(--muted);font-size:.98rem;}
.section-title{display:flex;align-items:center;gap:10px;margin:22px 0 6px;font-size:1.15rem;font-weight:700;color:#f8fafc;}
.section-title .num{width:28px;height:28px;border-radius:50%;background:var(--primary);color:#fff;display:flex;
      align-items:center;justify-content:center;font-size:.9rem;font-weight:800;}
.stepper{display:flex;align-items:center;margin:4px 0 16px;}
.stepper .dot{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;
      font-weight:700;border:2px solid var(--border);color:var(--muted);background:var(--bg);flex:0 0 auto;font-size:.9rem;}
.stepper .dot.done{background:var(--accent);border-color:var(--accent);color:#fff;}
.stepper .dot.cur{background:var(--primary);border-color:var(--primary);color:#fff;box-shadow:0 0 0 4px rgba(59,130,246,.25);}
.stepper .bar{height:3px;flex:1;background:var(--border);min-width:6px;}
.stepper .bar.done{background:var(--accent);}
.counter{text-align:right;color:var(--muted);font-weight:700;padding-top:14px;}
.mgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:8px 0 14px;}
.mcard{background:var(--bg);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:8px;padding:10px 14px;}
.mcard .ml{font-size:.74rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;}
.mcard .mv{font-size:1.2rem;font-weight:800;color:var(--accent);word-break:break-word;}
.mcard .mn{font-size:.78rem;color:var(--muted);margin-top:2px;}
.mcard.bad{border-left-color:#f87171;} .mcard.bad .mv{color:#f87171;}
.mcard.warn{border-left-color:#fbbf24;} .mcard.warn .mv{color:#fbbf24;}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.sel-marker){border-color:var(--primary)!important;
      box-shadow:0 0 0 1px var(--primary),0 8px 24px -12px rgba(59,130,246,.6);}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;background:rgba(59,130,246,.15);color:#93c5fd;
      font-size:.78rem;font-weight:700;border:1px solid rgba(59,130,246,.35);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  Estado
# ════════════════════════════════════════════════════════════════════════════
PAGES = ["🎯 Projeto LGR", "🎚️ Ajuste fino", "⏱️ Discretização"]

DEFAULTS = dict(
    page=PAGES[0], ctrl="pd",
    gN="[4, 16]", gD="[1, 4, 4, 0]", hN="1", hD="1",
    gcN="[1]", gcD="[1]",
    spec_type="targets", spec_mp="10", spec_crit=4, spec_ts="4",
    spec_zeta="0.5", spec_wn="2", spec_re="-2", spec_im="2",
    f_num="[10, 2]", f_den="[1, 2, 0]", disc_method="tustin", disc_T="1", disc_source="manual",
    phase="input", proj_idx=0, proj_steps=None,
    design=None, suggestion=None, gain_scale=None, template_id=None, cfg_snapshot=None,
    disc_phase="input", disc_idx=0, disc_steps=None,
    tune=None, tune_id=None, tune_ref=None,
)
# widgets que precisam sobreviver quando a tela deles não é renderizada
PERSIST = ["gN", "gD", "hN", "hD", "gcN", "gcD", "spec_type", "spec_mp", "spec_crit", "spec_ts", "spec_zeta",
           "spec_wn", "spec_re", "spec_im", "f_num", "f_den", "disc_method", "disc_T", "disc_source"]


def init_state():
    ss = st.session_state
    for k, v in DEFAULTS.items():
        ss.setdefault(k, v)
    for k in PERSIST:
        ss[k] = ss[k]


init_state()
ss = st.session_state

# ════════════════════════════════════════════════════════════════════════════
#  Callbacks
# ════════════════════════════════════════════════════════════════════════════


def invalidate():
    """Mudou algum dado do projeto → resultado contínuo anterior deixa de valer."""
    ss.design = None
    ss.suggestion = None
    ss.gain_scale = None
    ss.template_id = None
    ss.tune_id = None


def goto(page):
    ss.page = page


def pick_ctrl(key):
    ss.ctrl = key


def edit_project():
    ss.phase = "input"


def edit_disc():
    ss.disc_phase = "input"


def review_memorial():
    ss.phase = "wizard"


def move(idx_key, delta, n):
    ss[idx_key] = int(min(max(ss[idx_key] + delta, 0), n - 1))


def load_template():
    tid = ss.tpl_sel
    t = core.TEMPLATES[tid]
    ss.ctrl = t["mod"]
    ss.gN, ss.gD, ss.hN, ss.hD = t["gN"], t["gD"], t["hN"], t["hD"]
    ss.spec_type = t["spec_type"]
    for key, tk in (("spec_mp", "mp"), ("spec_ts", "ts"), ("spec_zeta", "zeta"), ("spec_wn", "wn"),
                    ("spec_re", "re"), ("spec_im", "im")):
        if tk in t:
            ss[key] = t[tk]
    if "crit" in t:
        ss.spec_crit = t["crit"]
    invalidate()
    ss.template_id = tid
    ss.phase = "input"
    ss.page = PAGES[0]
    st.toast(f"Template carregado: {t['name']}", icon="📚")


def _solve_and_store(cfg, jump_to_last=False):
    steps, design, suggestion = core.solve_design(cfg)
    ss.proj_steps = steps
    ss.design = design
    ss.suggestion = suggestion
    ss.cfg_snapshot = cfg
    ss.phase = "wizard"
    ss.proj_idx = len(steps) - 1 if jump_to_last else 0
    ss.tune_id = None


def apply_suggested():
    sug, cfg = ss.suggestion, ss.cfg_snapshot
    if not sug or not cfg:
        return
    try:
        _solve_and_store({**cfg, "gain_scale": sug["scale"]}, jump_to_last=True)
        ss.gain_scale = sug["scale"]
        st.toast("Ganho sugerido aplicado ao projeto.", icon="✅")
    except ValueError as e:
        st.toast(str(e), icon="⚠️")


def load_disc_fields(source):
    d = ss.design
    if not d or source == "manual":
        return
    num, den = (d["gcN"], d["gcD"]) if source == "controller" else (d["totN"], d["totD"])
    ss.f_num, ss.f_den = core.fmt_array(num), core.fmt_array(den)
    defaults = core.disc_defaults(d["template_id"], source)
    if defaults:
        ss.disc_method, ss.disc_T = defaults["method"], defaults["T"]
    ss.disc_source = source
    ss.disc_phase = "input"


def send_to_disc(source):
    load_disc_fields(source)
    ss.page = PAGES[2]


def on_disc_source():
    load_disc_fields(ss.disc_source)


def set_manual():
    ss.disc_source = "manual"


def on_tune(key, src):
    ss.tune[key] = float(ss[f"tn_{key}_{src}"])


def reset_tune():
    ss.tune = dict(ss.design["calc"])


def apply_tune():
    t = ss.tune
    ss.design = core.apply_tuning(ss.design, t["kc"], t["z"], t["p"])
    st.toast("Valores aplicados ao projeto — já valem para a discretização.", icon="✅")


# ════════════════════════════════════════════════════════════════════════════
#  Gráficos (plotly)
# ════════════════════════════════════════════════════════════════════════════
C_POLE, C_ZERO, C_SD, C_CL, C_LINE = "#f87171", "#38bdf8", "#fbbf24", "#34d399", "#60a5fa"


def _base(title, height=380):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)), template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a", margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(orientation="h", y=-0.22, x=0), font=dict(color="#e2e8f0"),
    )
    fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#475569")
    fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#475569")
    return fig


def _pz(fig, poles, zeros, pname="Polos", zname="Zeros"):
    poles, zeros = np.asarray(poles, complex), np.asarray(zeros, complex)
    if zeros.size:
        fig.add_trace(go.Scatter(x=zeros.real, y=zeros.imag, mode="markers", name=zname,
                                 marker=dict(symbol="circle-open", size=13, line=dict(width=2.5, color=C_ZERO))))
    if poles.size:
        fig.add_trace(go.Scatter(x=poles.real, y=poles.imag, mode="markers", name=pname,
                                 marker=dict(symbol="x", size=11, line=dict(width=2.5, color=C_POLE), color=C_POLE)))


def _sd(fig, sd):
    fig.add_trace(go.Scatter(x=[sd.real, sd.real], y=[sd.imag, -sd.imag], mode="markers", name="Polo desejado s_d",
                             marker=dict(symbol="diamond", size=12, color=C_SD, line=dict(width=1, color="#000"))))


def _fit(fig, pts, equal=True):
    pts = np.asarray([complex(p) for p in pts] + [0j])
    xs, ys = pts.real, np.abs(pts.imag)
    span = max(xs.max() - xs.min(), 2 * ys.max(), 2.0)
    m = 0.35 * span
    fig.update_xaxes(range=[xs.min() - m, xs.max() + m], title="Real")
    ymax = max(ys.max(), 0.35 * span) + m
    fig.update_yaxes(range=[-ymax, ymax], title="Imaginário")
    if equal:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)


def make_fig(spec):
    k = spec["kind"]
    if k in ("splane", "angles"):
        fig = _base(spec["title"], 400)
        poles, zeros, sd = spec["poles"], spec["zeros"], spec.get("sd")
        keypts = list(poles) + list(zeros) + ([sd] if sd is not None else [])
        if k == "angles" and sd is not None:
            for r in list(zeros) + list(poles):
                col = C_ZERO if r in list(zeros) else C_POLE
                fig.add_trace(go.Scatter(x=[r.real, sd.real], y=[r.imag, sd.imag], mode="lines", showlegend=False,
                                         line=dict(color=col, width=1.5, dash="dot")))
        if spec.get("zeta") and spec.get("wn") and sd is not None:
            wn, z = spec["wn"], spec["zeta"]
            ang = math.pi - math.acos(z)
            for sgn in (1, -1):
                fig.add_trace(go.Scatter(x=[0, 1.7 * wn * math.cos(ang)], y=[0, sgn * 1.7 * wn * math.sin(ang)],
                                         mode="lines", showlegend=False, line=dict(color="#64748b", dash="dash", width=1)))
            th = np.linspace(math.pi / 2, 3 * math.pi / 2, 100)
            fig.add_trace(go.Scatter(x=wn * np.cos(th), y=wn * np.sin(th), mode="lines", name=f"|s| = ωn = {wn:.3g}",
                                     line=dict(color="#64748b", dash="dot", width=1)))
            keypts += [complex(-1.7 * wn, 0), complex(0, 1.2 * wn)]
        _pz(fig, poles, zeros)
        if sd is not None:
            _sd(fig, sd)
        _fit(fig, keypts)
        return fig

    if k == "locus":
        fig = _base(spec["title"], 460)
        pts = np.asarray(spec["points"])
        fig.add_trace(go.Scattergl(x=pts.real, y=pts.imag, mode="markers", name="LGR",
                                   marker=dict(size=3, color="#64748b", opacity=0.75)))
        _pz(fig, spec["olp"], spec["olz"], "Polos de malha aberta", "Zeros de malha aberta")
        cl = np.asarray(spec["cl"], complex)
        fig.add_trace(go.Scatter(x=cl.real, y=cl.imag, mode="markers", name="Polos de malha fechada (projeto)",
                                 marker=dict(symbol="star", size=13, color=C_CL, line=dict(width=1, color="#000"))))
        _sd(fig, spec["sd"])
        _fit(fig, list(spec["olp"]) + list(spec["olz"]) + list(cl) + [spec["sd"]])
        return fig

    if k == "step":
        fig = _base(spec["title"], 380)
        t, y = np.asarray(spec["t"]), np.asarray(spec["y"], float)
        stride = max(1, len(t) // 1200)
        final = spec["final"]
        if spec.get("ref") is not None:
            rt, ry = spec["ref"]
            fig.add_trace(go.Scatter(x=rt[::stride], y=ry[::stride], mode="lines", name="Projeto original",
                                     line=dict(color="#64748b", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=t[::stride], y=y[::stride], mode="lines", name="y(t)", line=dict(color=C_LINE, width=3)))
        fig.add_hline(y=final, line=dict(color=C_CL, width=1.5), annotation_text="valor final", annotation_position="right")
        if spec.get("mp"):
            fig.add_hline(y=final * (1 + spec["mp"] / 100), line=dict(color=C_POLE, width=1.2, dash="dash"),
                          annotation_text=f"MP alvo {spec['mp']:g}%", annotation_position="top right")
        if spec.get("ts"):
            fig.add_vline(x=spec["ts"], line=dict(color=C_SD, width=1.2, dash="dash"),
                          annotation_text=f"ts alvo {spec['ts']:g}s", annotation_position="top")
        ymax = np.nanmax(y) if np.isfinite(y).any() else 1
        scale = max(abs(final), 1.0)
        top = 3 * scale if ymax > 10 * scale else max(ymax, final) * 1.15
        fig.update_yaxes(range=[min(0, np.nanmin(y) * 1.1 if np.isfinite(y).any() else 0), top], title="Amplitude")
        fig.update_xaxes(title="Tempo (s)")
        return fig

    if k == "zplane":
        fig = _base(spec["title"], 400)
        th = np.linspace(0, 2 * math.pi, 200)
        fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines", name="Círculo unitário",
                                 line=dict(color="#64748b", dash="dash")))
        _pz(fig, spec["poles"], spec["zeros"])
        _fit(fig, [1.3 + 0j, -1.3 + 1.3j] + list(spec["poles"]) + list(spec["zeros"]))
        return fig
    raise ValueError(k)


# ════════════════════════════════════════════════════════════════════════════
#  Componentes de UI
# ════════════════════════════════════════════════════════════════════════════
def section(num, text):
    st.markdown(f'<div class="section-title"><span class="num">{num}</span>{html.escape(text)}</div>', unsafe_allow_html=True)


def metric_cards(items):
    cards = []
    for it in items:
        label, value = it[0], it[1]
        note = it[2] if len(it) > 2 else ""
        cls = "bad" if "⚠️" in note else ""
        n = f'<div class="mn">{html.escape(note)}</div>' if note else ""
        cards.append(f'<div class="mcard {cls}"><div class="ml">{html.escape(str(label))}</div>'
                     f'<div class="mv">{html.escape(str(value))}</div>{n}</div>')
    st.markdown('<div class="mgrid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_blocks(blocks, prefix):
    for i, (kind, payload) in enumerate(blocks):
        if kind == "md":
            st.markdown(payload)
        elif kind == "tex":
            st.latex(payload)
        elif kind == "warn":
            st.warning(payload)
        elif kind == "ok":
            st.success(payload)
        elif kind == "info":
            st.info(payload)
        elif kind == "metrics":
            metric_cards(payload)
        elif kind == "plot":
            st.plotly_chart(make_fig(payload), key=f"{prefix}_plot_{i}", theme=None, **CHART)
        elif kind == "divider":
            st.divider()


def stepper_html(n, cur):
    parts = []
    for i in range(n):
        cls = "done" if i < cur else ("cur" if i == cur else "")
        parts.append(f'<div class="dot {cls}">{"✓" if i < cur else i + 1}</div>')
        if i < n - 1:
            parts.append(f'<div class="bar {"done" if i < cur else ""}"></div>')
    return '<div class="stepper">' + "".join(parts) + "</div>"


def render_wizard(ns, steps, idx_key, on_edit, on_finish, last_extra=None, doc_title="Memorial"):
    n = len(steps)
    idx = int(min(max(ss[idx_key], 0), n - 1))
    st.markdown(stepper_html(n, idx), unsafe_allow_html=True)
    step = steps[idx]
    with st.container(border=True):
        h1, h2 = st.columns([5, 1])
        h1.markdown(f"### {idx + 1}. {step.title}")
        h2.markdown(f'<div class="counter">{idx + 1} / {n}</div>', unsafe_allow_html=True)
        st.divider()
        render_blocks(step.blocks, f"{ns}_{idx}")
        if idx == n - 1:
            if last_extra:
                st.divider()
                last_extra()
            st.download_button("📄 Baixar memorial (.md)", core.memorial_to_markdown(doc_title, steps),
                               file_name=f"{ns}_memorial.md", mime="text/markdown", key=f"{ns}_dl", **DL)
    c1, c2, c3 = st.columns(3)
    c1.button("✏️ Editar dados", key=f"{ns}_edit", on_click=on_edit, **BTN)
    c2.button("◀ Anterior", key=f"{ns}_prev", disabled=idx == 0, on_click=move, args=(idx_key, -1, n), **BTN)
    if idx == n - 1:
        c3.button("Finalizar ✔", key=f"{ns}_next", type="primary", on_click=on_finish, **BTN)
    else:
        c3.button("Avançar ▶", key=f"{ns}_next", type="primary", on_click=move, args=(idx_key, 1, n), **BTN)


def gc_tex(d):
    return r"G_c(s) = " + core.tf_tex(d["gcN"], d["gcD"], factored=True)


# ════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎛️ Solver LGR")
    st.caption("Projeto de controladores pelo Lugar Geométrico das Raízes")
    st.radio("Módulo", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.markdown("**📚 Templates (Q1–Q6)**")
    st.selectbox("Template", list(core.TEMPLATES), format_func=lambda k: core.TEMPLATES[k]["name"],
                 key="tpl_sel", label_visibility="collapsed")
    st.button("Carregar template", key="btn_tpl", on_click=load_template, **BTN)
    d_ = ss.design
    if d_:
        st.divider()
        st.markdown("**Projeto atual**")
        st.markdown(f'<span class="pill">{html.escape(d_["label"])}</span>' + (' <span class="pill">ajustado</span>' if d_["tuned"] else ""),
                    unsafe_allow_html=True)
        st.latex(gc_tex(d_))
    with st.expander("ℹ️ Como usar"):
        st.markdown(
            "1. **Projeto LGR** — escolha o controlador, informe G(s), H(s) e as especificações.\n"
            "2. Navegue pelo **memorial passo a passo**.\n"
            "3. **Ajuste fino** — mexa em Kc, z, p e veja a resposta ao degrau em tempo real.\n"
            "4. **Discretização** — leve o controlador (ou o sistema todo) para o domínio z.\n\n"
            "**Formatos aceitos nos polinômios:**\n"
            "- Coeficientes: `[1, 4, 4, 0]`\n"
            "- Simbólico: `10(s+2)`, `(s+1)(s+3)^2`\n"
            "- Ganho e raízes: `5: [-1, -2+3j]`"
        )

# ════════════════════════════════════════════════════════════════════════════
#  Cabeçalho
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="hero"><h1>🎛️ Solver LGR — Controle e Automação</h1>'
    "<p>Projete PD, PI, PID, compensadores de 1 polo ou o seu próprio Gc(s) com memorial passo a passo, "
    "ajuste fino interativo e discretização.</p></div>",
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════
#  PÁGINA 1 — Projeto
# ════════════════════════════════════════════════════════════════════════════
CTRL_INFO = {
    "pd": ("PD", "⚡", "Adiciona um zero. Acelera e amortece a resposta transitória.", r"K_c\,(s+z)"),
    "pi": ("PI", "🎯", "Polo na origem + zero. Elimina o erro em regime permanente.", r"K_c\,\frac{s+z}{s}"),
    "pid": ("PID", "🧠", "Polo na origem + dois zeros iguais. O melhor dos dois mundos.", r"K_c\,\frac{(s+z)^2}{s}"),
    "1pole": ("1 Polo", "🐢", "Compensador de atraso com um único polo.", r"\frac{K_c}{s+p}"),
    "custom": ("Custom", "🛠️", "Você informa Gc(s); o app valida a malha fechada.", r"\frac{N(s)}{D(s)}"),
}
SPEC_LABELS = {"targets": "Metas (MP% e ts)", "params": "Parâmetros (ζ e ωn)", "poles": "Polos desejados (Re e Im)"}
CRIT_LABELS = {4: "2%  (ts = 4/ζωn)", 3: "5%  (ts = 3/ζωn)"}


def _p(label, fn, *a):
    try:
        return fn(*a)
    except core.ExprError as e:
        raise ValueError(f"**{label}:** {e}")


def collect_cfg():
    gN = _p("G(s) numerador", core.parse_array, ss.gN)
    gD = _p("G(s) denominador", core.parse_array, ss.gD)
    hN = _p("H(s) numerador", core.parse_array, ss.hN)
    hD = _p("H(s) denominador", core.parse_array, ss.hD)
    kind = ss.spec_type
    if kind == "targets":
        spec = dict(type="targets", mp=_p("MP (%)", core.parse_scalar, ss.spec_mp),
                    ts=_p("ts (s)", core.parse_scalar, ss.spec_ts), ts_factor=int(ss.spec_crit))
    elif kind == "params":
        spec = dict(type="params", zeta=_p("ζ", core.parse_scalar, ss.spec_zeta), wn=_p("ωn", core.parse_scalar, ss.spec_wn))
    else:
        spec = dict(type="poles", re=_p("Polo real (σ)", core.parse_scalar, ss.spec_re),
                    im=_p("Polo imaginário (ωd)", core.parse_scalar, ss.spec_im))
    cfg = dict(mod=ss.ctrl, gN=gN, gD=gD, hN=hN, hD=hD, spec=spec, template_id=ss.template_id)
    if ss.ctrl == "custom":
        cfg["gcN"] = _p("Gc(s) numerador", core.parse_array, ss.gcN)
        cfg["gcD"] = _p("Gc(s) denominador", core.parse_array, ss.gcD)
    return cfg


def tf_preview(name, nk, dk):
    try:
        n, d = core.parse_array(ss[nk]), core.parse_array(ss[dk])
        st.latex(f"{name} = " + core.tf_tex(n, d))
    except core.ExprError as e:
        st.caption(f"⚠️ {name}: {e}")


def spec_preview():
    try:
        if ss.spec_type == "targets":
            mp, ts, f = core.parse_scalar(ss.spec_mp), core.parse_scalar(ss.spec_ts), int(ss.spec_crit)
            if not (0 < mp < 100 and ts > 0):
                return
            ln = math.log(mp / 100)
            z = math.sqrt(ln ** 2 / (math.pi ** 2 + ln ** 2))
            wn = f / (z * ts)
            sd = complex(-z * wn, wn * math.sqrt(1 - z * z))
        elif ss.spec_type == "params":
            z, wn = core.parse_scalar(ss.spec_zeta), core.parse_scalar(ss.spec_wn)
            if not (0 < z < 1 and wn > 0):
                return
            sd = complex(-z * wn, wn * math.sqrt(1 - z * z))
        else:
            sd = complex(core.parse_scalar(ss.spec_re), core.parse_scalar(ss.spec_im))
            z, wn = (-sd.real / abs(sd), abs(sd)) if abs(sd) > 0 else (0, 0)
        metric_cards([("ζ", f"{z:.4f}"), ("ωn", f"{wn:.4f} rad/s"), ("Polo dominante s_d", core.cx(sd))])
    except (core.ExprError, ValueError):
        pass


def project_inputs():
    section(1, "Escolha o tipo de controlador")
    cols = st.columns(5)
    for col, (key, (name, icon, desc, tex)) in zip(cols, CTRL_INFO.items()):
        with col:
            with st.container(border=True):
                sel = ss.ctrl == key
                if sel:
                    st.markdown('<span class="sel-marker"></span>', unsafe_allow_html=True)
                st.markdown(f"#### {icon} {name}")
                st.caption(desc)
                st.latex(tex)
                st.button("✓ Selecionado" if sel else "Selecionar", key=f"pick_{key}", on_click=pick_ctrl, args=(key,),
                          type="primary" if sel else "secondary", **BTN)

    section(2, "Defina a planta G(s) e a realimentação H(s)")
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("G(s) — numerador", key="gN", on_change=invalidate, help="Ex.: [4, 16]  ou  4(s+4)")
        st.text_input("G(s) — denominador", key="gD", on_change=invalidate, help="Ex.: [1, 4, 4, 0]  ou  s(s+2)^2")
    with c2:
        st.text_input("H(s) — numerador", key="hN", on_change=invalidate)
        st.text_input("H(s) — denominador", key="hD", on_change=invalidate)
    p1, p2 = st.columns(2)
    with p1:
        tf_preview("G(s)", "gN", "gD")
    with p2:
        tf_preview("H(s)", "hN", "hD")

    if ss.ctrl == "custom":
        st.markdown("**Controlador customizado**")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Gc(s) — numerador", key="gcN", on_change=invalidate, help="Ex.: [10, 20]  ou  10(s+2)")
        with c2:
            st.text_input("Gc(s) — denominador", key="gcD", on_change=invalidate, help="Ex.: [1, 10]  ou  s+10")
        tf_preview("G_c(s)", "gcN", "gcD")
        st.info("O ganho deve estar incluso no numerador. O app valida o sistema completo em malha fechada.")

    section(3, "Especificações de desempenho")
    st.radio("Como você quer especificar?", list(SPEC_LABELS), key="spec_type", horizontal=True,
             format_func=SPEC_LABELS.get, on_change=invalidate)
    if ss.spec_type == "targets":
        c1, c2, c3 = st.columns(3)
        c1.text_input("MP — overshoot (%)", key="spec_mp", on_change=invalidate)
        c2.selectbox("Critério de ts", [4, 3], key="spec_crit", format_func=CRIT_LABELS.get, on_change=invalidate)
        c3.text_input("ts — tempo de acomodação (s)", key="spec_ts", on_change=invalidate)
    elif ss.spec_type == "params":
        c1, c2 = st.columns(2)
        c1.text_input("ζ (zeta)", key="spec_zeta", on_change=invalidate)
        c2.text_input("ωn (rad/s)", key="spec_wn", on_change=invalidate)
    else:
        c1, c2 = st.columns(2)
        c1.text_input("Polo real (σ)", key="spec_re", on_change=invalidate)
        c2.text_input("Polo imaginário (ωd)", key="spec_im", on_change=invalidate)
    spec_preview()

    st.write("")
    if st.button("🚀 Resolver e gerar memorial", type="primary", key="btn_solve", **BTN):
        try:
            _solve_and_store(collect_cfg())
        except ValueError as e:
            st.error(str(e))
        else:
            st.rerun()

    if ss.design is not None:
        d = ss.design
        st.success(f"Projeto disponível: **{d['label']}**. Continue para o ajuste fino ou a discretização.")
        a, b, c = st.columns(3)
        a.button("📖 Rever memorial", on_click=review_memorial, **BTN)
        b.button("🎚️ Ajuste fino", on_click=goto, args=(PAGES[1],), **BTN)
        c.button("⏱️ Discretizar", on_click=send_to_disc, args=("controller",), **BTN)


def project_last_extra():
    d, sug = ss.design, ss.suggestion
    if sug:
        st.warning(f"**Ajuste fino automático disponível:** Kc recomendado ≈ **{sug['kc']:.4f}** "
                   f"→ MP ≈ {sug['metrics']['mp']:.2f}% · ts ≈ {sug['metrics']['ts']:.2f}s.")
        st.button("⚙️ Aplicar Kc sugerido e recalcular", key="btn_apply_sug", on_click=apply_suggested, **BTN)
    st.markdown("#### E agora?")
    a, b, c = st.columns(3)
    a.button("🎚️ Ajuste fino interativo", key="x_tune", on_click=goto, args=(PAGES[1],), **BTN)
    b.button("⏱️ Discretizar controlador", key="x_dc", on_click=send_to_disc, args=("controller",), **BTN)
    c.button("⏱️ Discretizar sistema completo", key="x_dt", on_click=send_to_disc, args=("total",), **BTN)


def page_project():
    if ss.phase == "wizard" and ss.proj_steps:
        render_wizard("proj", ss.proj_steps, "proj_idx", edit_project, edit_project, project_last_extra,
                      doc_title=f"Memorial LGR — {ss.design['label'] if ss.design else ''}")
    else:
        project_inputs()


# ════════════════════════════════════════════════════════════════════════════
#  PÁGINA 2 — Ajuste fino
# ════════════════════════════════════════════════════════════════════════════
def _nice_step(hi):
    return 10.0 ** math.floor(math.log10(max(hi, 1e-9) / 1000.0))


def _range(v0):
    hi = max(20.0, abs(v0) * 3.0)
    return (-hi if v0 < 0 else 0.0), hi


def page_tuning():
    d = ss.design
    st.markdown("### 🎚️ Ajuste fino interativo")
    if not d:
        st.info("Projete um controlador primeiro para habilitar o ajuste fino com os valores calculados.")
        st.button("🎯 Ir para o projeto", on_click=goto, args=(PAGES[0],))
        return

    mod = d["mod"]
    if ss.tune_id != d["id"]:
        ss.tune = dict(d["calc"])
        ss.tune_id = d["id"]
        r = core.tuning_response(d, **d["calc"])
        ss.tune_ref = (r["t"], r["y"])

    params = [("kc", "Ganho Kc" if mod != "custom" else "Ganho equivalente Kc")]
    if mod in ("pd", "pi", "pid"):
        params.append(("z", "Zero z  (Gc ∝ s + z)"))
    if mod == "1pole":
        params.append(("p", "Polo p  (Gc ∝ 1/(s + p))"))

    st.caption(f"Controlador **{core.MOD_NAMES[mod]}** · base: {d['label']}")
    if mod == "custom":
        st.info("Para controladores customizados, o ajuste fino escala o ganho do Gc(s) informado.")

    t = ss.tune
    ranges = {k: _range(d["calc"][k]) for k, _ in params}
    for k, _ in params:
        lo, hi = ranges[k]
        t[k] = float(min(max(t[k], lo), hi))
        ss[f"tn_{k}_s"] = t[k]
        ss[f"tn_{k}_n"] = t[k]

    with st.container(border=True):
        for k, label in params:
            lo, hi = ranges[k]
            step = _nice_step(hi)
            c1, c2 = st.columns([4, 1])
            c1.slider(label, min_value=lo, max_value=hi, step=step, key=f"tn_{k}_s", on_change=on_tune, args=(k, "s"))
            c2.number_input("valor", min_value=lo, max_value=hi, step=step, format="%.4f", key=f"tn_{k}_n",
                            on_change=on_tune, args=(k, "n"), label_visibility="hidden")

    r = core.tuning_response(d, t["kc"], t["z"], t["p"])
    spec = d["spec"]
    st.latex(r"G_c(s) = " + core.tf_tex(r["gcN"], r["gcD"], factored=True))

    mp_txt = f"{r['mp']:.2f}%" if math.isfinite(r["mp"]) else "—"
    ts_txt = f"{r['ts']:.3f} s" if math.isfinite(r["ts"]) else "não acomodou"
    mp_note = ts_note = ""
    if spec["type"] == "targets":
        mp_note = f"meta ≤ {spec['mp']:g}%  " + ("✅" if r["mp"] <= spec["mp"] + 1e-3 else "⚠️")
        ts_note = f"meta ≤ {spec['ts']:g} s  " + ("✅" if r["ts"] <= spec["ts"] + 1e-3 else "⚠️")
    metric_cards([("Overshoot (MP)", mp_txt, mp_note), ("Tempo de acomodação (ts)", ts_txt, ts_note),
                  ("Valor final", f"{r['final']:.4f}"),
                  ("Malha fechada", "estável" if r["is_stable"] else "INSTÁVEL", "" if r["is_stable"] else "⚠️ polo no SPD")])
    if not r["is_stable"]:
        st.error("Com esses parâmetros a malha fechada é instável.")

    fig = make_fig(dict(kind="step", title="Resposta ao degrau — ajuste em tempo real", t=r["t"], y=r["y"],
                        final=r["final"], mp=spec.get("mp"), ts=spec.get("ts"), ref=ss.tune_ref))
    st.plotly_chart(fig, key="tune_plot", theme=None, **CHART)
    with st.expander("Polos de malha fechada"):
        st.markdown("\n".join(f"- {core.cx(p)}" for p in r["poles"]) or "Nenhum")

    c1, c2 = st.columns(2)
    c1.button("✅ Usar estes valores no projeto", type="primary", key="tn_apply", on_click=apply_tune, **BTN)
    c2.button("↩️ Resetar para o cálculo original", key="tn_reset", on_click=reset_tune, **BTN)


# ════════════════════════════════════════════════════════════════════════════
#  PÁGINA 3 — Discretização
# ════════════════════════════════════════════════════════════════════════════
SRC_LABELS = {"manual": "✍️ Manual", "controller": "🎛️ Controlador projetado Gc(s)",
              "total": "🔗 Sistema completo G·H·Gc"}
METHOD_LABELS = {"tustin": "Tustin (Bilinear)", "euler": "Euler (Forward)"}


def page_disc():
    if ss.disc_phase == "wizard" and ss.disc_steps:
        render_wizard("disc", ss.disc_steps, "disc_idx", edit_disc, edit_disc, doc_title="Memorial de discretização")
        return

    st.markdown("### ⏱️ Discretização de F(s)")
    d = ss.design
    opts = ["manual"] + (["controller", "total"] if d else [])
    if ss.disc_source not in opts:
        ss.disc_source = "manual"
    st.radio("Origem da função", opts, key="disc_source", horizontal=True, format_func=SRC_LABELS.get,
             on_change=on_disc_source)
    if not d:
        st.caption("💡 Projete um controlador para poder carregá-lo automaticamente aqui.")
    elif ss.disc_source != "manual":
        st.info(f"**Origem:** {SRC_LABELS[ss.disc_source]} carregado do projeto *{d['label']}*. Revise antes de discretizar.")

    c1, c2 = st.columns(2)
    c1.text_input("F(s) — numerador", key="f_num", on_change=set_manual)
    c2.text_input("F(s) — denominador", key="f_den", on_change=set_manual)
    try:
        st.latex("F(s) = " + core.tf_tex(core.parse_array(ss.f_num), core.parse_array(ss.f_den)))
    except core.ExprError as e:
        st.caption(f"⚠️ {e}")

    c1, c2 = st.columns(2)
    c1.selectbox("Método", list(METHOD_LABELS), key="disc_method", format_func=METHOD_LABELS.get)
    c2.text_input("Período de amostragem T (s)", key="disc_T")

    if st.button("⚡ Discretizar sistema", type="primary", key="btn_disc", **BTN):
        try:
            fN = _p("F(s) numerador", core.parse_array, ss.f_num)
            fD = _p("F(s) denominador", core.parse_array, ss.f_den)
            T = _p("Período T", core.parse_scalar, ss.disc_T)
            origin = (f"{SRC_LABELS[ss.disc_source]} (projeto {d['label']})" if d and ss.disc_source != "manual"
                      else "função informada manualmente")
            steps, _res = core.solve_disc(fN, fD, ss.disc_method, T, origin)
        except ValueError as e:
            st.error(str(e))
        else:
            ss.disc_steps, ss.disc_idx, ss.disc_phase = steps, 0, "wizard"
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  Roteamento
# ════════════════════════════════════════════════════════════════════════════
if ss.page == PAGES[0]:
    page_project()
elif ss.page == PAGES[1]:
    page_tuning()
else:
    page_disc()
