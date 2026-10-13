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

    st.latex(r"G_&#x63;*(*&#x73;*)* = " + core.tf_tex(r["gcN"], r["gcD"], factored=True))

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

METHOD_LABELS = {"tustin": "Tustin (Bilinear)", "euler": "Euler (Forward)", "backward": "Backward (s = (z-1)/(Tz))"}

def page_disc():

    if ss.disc_phase == "wizard" and ss.disc_steps:

        render_wizard("disc", ss.disc_steps, "disc_idx", edit_disc, edit_disc)

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
