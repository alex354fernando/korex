"""Apartado 2 · Simulador «¿qué pasaría si…?»: modifica valores y observa el estadio en vivo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402

P = ui.setup("🎛️ Simulador ¿qué pasaría si…?",
             "Mueva los controles y vea en tiempo real cómo cambiaría el estadio estimado del paciente.")
if P is None:
    st.stop()

SLIDERS = {  # variable: (mín, máx, paso)
    "creatinine": (0.3, 15.0, 0.1), "bu": (10, 300, 1), "hemo": (4.0, 18.0, 0.1), "age": (18, 90, 1),
    "bp": (50, 140, 5), "pot": (2.5, 8.0, 0.1), "sod": (110, 160, 1), "bgr": (60, 500, 5),
}
TOGGLES = ["htn", "dm", "ane", "pe"]
screening = ui.has_screening(P) and st.toggle(
    "🔎 Simular sin creatinina (modo tamizaje)", key="sim_screening",
    help="Con creatinina, el estadio depende casi solo de ella (es la base de la fórmula KDIGO). "
         "Sin creatinina se ve cómo influyen urea, hemoglobina, albúmina y los demás análisis.")
SC = ui.SCREEN if screening else ui.SCEN
feats = [f for f in P.features(ui.SCEN) if not (screening and f == "creatinine")]


def load_source(rec: dict):
    st.session_state["sim_base"] = dict(rec)
    for f, (lo, hi, _) in SLIDERS.items():
        if f in feats and rec.get(f) is not None:
            v = float(np.clip(rec[f], lo, hi))
            st.session_state[f"s_{f}"] = int(v) if isinstance(lo, int) else v
    if "al" in feats:
        st.session_state["s_al"] = float(rec.get("al", 0) or 0)
    for f in TOGGLES:
        if f in feats:
            st.session_state[f"s_{f}"] = rec.get(f) == "yes"


if "sim_source" in st.session_state:                 # viene desde «Evaluar paciente»
    load_source(st.session_state.pop("sim_source"))
if "sim_base" not in st.session_state:
    load_source(st.session_state.get("record") or ui.CFG["base_patient"])


def _from_base():
    load_source(ui.CFG["base_patient"])


def _from_last():
    if st.session_state.get("record"):
        load_source(st.session_state["record"])


def _from_typical():
    ui.set_typical("tmp", P, "sim_typ")
    load_source(ui.record_from("tmp", P))


c = st.columns([1.2, 1.4, 0.7, 1.4])
c[0].button("🧍 Paciente base", on_click=_from_base, width="stretch")
c[1].button("🩺 Último paciente evaluado", on_click=_from_last, width="stretch", disabled="record" not in st.session_state)
c[2].selectbox("Estadio", ui.STAGES, key="sim_typ", label_visibility="collapsed")
c[3].button("🎯 Paciente típico", on_click=_from_typical, width="stretch", disabled="stage_ranges" not in P.metadata)

base = st.session_state["sim_base"]
left, right = st.columns([1, 1.25])
with left:
    with st.container(border=True):
        st.markdown("**Valores de laboratorio y datos del paciente**")
        for f, (lo, hi, step) in SLIDERS.items():
            if f in feats:
                meta = ui.VARS.get(f, {"label": f, "unit": ""})
                st.slider(f"{meta['label']} ({meta['unit']})" if meta.get("unit") else meta["label"],
                          lo, hi, step=step, key=f"s_{f}")
        if "al" in feats:
            st.select_slider("Albúmina en tira reactiva (0–5)", options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], key="s_al",
                             format_func=lambda v: f"{v:g}")
        tcols = st.columns(2)
        for i, f in enumerate(t for t in TOGGLES if t in feats):
            tcols[i % 2].toggle(ui.VARS[f]["label"], key=f"s_{f}")

record = dict(base)
for f in SLIDERS:
    if f in feats:
        record[f] = st.session_state[f"s_{f}"]
if "al" in feats:
    record["al"] = st.session_state["s_al"]
for f in TOGGLES:
    if f in feats:
        record[f] = "yes" if st.session_state[f"s_{f}"] else "no"
record["creatinine_unit"] = "mg/dL"
if screening:
    record["creatinine"] = None
    base = {**base, "creatinine": None}

res = P.predict(record, SC)
ref = P.predict(base, SC)
with right:
    st.markdown(ui.screening_card(res) if screening else ui.result_card(res), unsafe_allow_html=True)
    delta = res["risk_class"] - ref["risk_class"]
    msg = ("igual que el paciente de partida" if delta == 0 else
           f"{abs(delta)} estadio(s) {'más avanzado' if delta > 0 else 'menos avanzado'} que el paciente de partida "
           f"({ui.STAGES[ref['risk_class']]})")
    st.markdown(f"**Cambio:** {msg}.")
    if screening:
        ui.screening_note(res, P)
    else:
        ui.formula_note(res)
    st.plotly_chart(ui.prob_chart(res["probabilities"], height=250), width="stretch")

st.divider()
st.subheader("¿Cómo cambia el estadio al variar una sola variable?")
opts = [f for f in SLIDERS if f in feats]
var = st.selectbox("Variable", opts, format_func=lambda f: ui.VARS.get(f, {}).get("label", f))
lo, hi, _ = SLIDERS[var]
grid = np.linspace(lo, hi, 45)
Pg = P.predict_proba_records([{**record, var: float(v)} for v in grid], SC)
fig = go.Figure()
for k in range(Pg.shape[1]):
    fig.add_trace(go.Scatter(x=grid, y=Pg[:, k], stackgroup="one", name=ui.STAGES[k], mode="lines",
                             line=dict(width=0.5, color=ui.COLORS[k])))
fig.add_vline(x=record[var], line_dash="dash", line_color="#111827", annotation_text="valor actual")
meta = ui.VARS.get(var, {"label": var, "unit": ""})
fig.update_layout(height=360, yaxis=dict(tickformat=".0%", title="Probabilidad"),
                  xaxis_title=f"{meta['label']} ({meta.get('unit', '')})", margin=dict(t=20),
                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, width="stretch")
st.caption("El resto de variables se mantiene fijo. Herramienta educativa: no reemplaza la medición real.")
