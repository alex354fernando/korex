"""Apartado 3 · Comparar dos pacientes lado a lado."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402

P = ui.setup("⚖️ Comparar pacientes", "Evalúe dos pacientes a la vez y vea en qué se diferencian sus resultados.")
if P is None:
    st.stop()
ui.ensure("ca", P)
if "cb_age" not in st.session_state:
    ui.set_random("cb", P)

cols = st.columns(2)
for col, prefix, name in ((cols[0], "ca", "Paciente A"), (cols[1], "cb", "Paciente B")):
    with col:
        with st.container(border=True):
            st.subheader(name)
            c = st.columns(3)
            c[0].button("🧍 Base", on_click=ui.set_base, args=(prefix, P), key=f"{prefix}_b1", width="stretch")
            c[1].button("🎲 Azar", on_click=ui.set_random, args=(prefix, P), key=f"{prefix}_b2", width="stretch")
            if c[2].button("🩺 Último", key=f"{prefix}_b3", width="stretch", disabled="record" not in st.session_state):
                ui.load_record(prefix, st.session_state["record"], P)
                st.rerun()
            ui.patient_form(prefix, P, columns=2, compact=True)

ra, rb = ui.record_from("ca", P), ui.record_from("cb", P)
if st.button("Comparar", type="primary", width="stretch"):
    st.session_state["cmp"] = (P.predict(ra, ui.SCEN), P.predict(rb, ui.SCEN), ra, rb)

if "cmp" in st.session_state:
    a, b, ra, rb = st.session_state["cmp"]
    st.divider()
    c = st.columns(2)
    for col, res, name in ((c[0], a, "Paciente A"), (c[1], b, "Paciente B")):
        with col:
            st.markdown(f"#### {name}")
            st.markdown(ui.result_card(res), unsafe_allow_html=True)
            ui.formula_note(res)
    fig = go.Figure([go.Bar(name="Paciente A", x=ui.STAGES, y=a["probabilities"], marker_color="#0F766E"),
                     go.Bar(name="Paciente B", x=ui.STAGES, y=b["probabilities"], marker_color="#1E3A8A")])
    fig.update_layout(barmode="group", height=300, yaxis=dict(tickformat=".0%"), title="Probabilidades por estadio",
                      margin=dict(t=40), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    rows = []
    for f in P.features(ui.SCEN):
        va, vb = ra.get(f), rb.get(f)
        meta = ui.VARS.get(f, {"label": f, "unit": ""})
        rows.append({"Variable": meta["label"], "Paciente A": ui.show(va), "Paciente B": ui.show(vb),
                     "Unidad": meta.get("unit", ""), "¿Distinto?": "●" if va != vb else ""})
    st.markdown("**Datos de ambos pacientes**")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if st.button("🔍 Comparar los factores que más influyeron (SHAP)"):
        with st.spinner("Calculando explicaciones…"):
            ea, eb = P.explain(ra, ui.SCEN), P.explain(rb, ui.SCEN)
        c = st.columns(2)
        c[0].plotly_chart(ui.factor_chart(ea, 6), width="stretch"); c[0].caption(ea["reading"])
        c[1].plotly_chart(ui.factor_chart(eb, 6), width="stretch"); c[1].caption(eb["reading"])
