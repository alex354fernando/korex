"""Apartado 4 · Historial de evaluaciones (registro de auditoría con hash encadenado)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402

P = ui.setup("🗂️ Historial", "Todas las evaluaciones realizadas en este equipo, con verificación de integridad.")
if P is None:
    st.stop()

if P.demo_mode():
    st.info("En la demostración pública el historial está desactivado: las evaluaciones no se guardan "
            "para que ningún visitante vea los datos de otro. En una instalación local cada evaluación "
            "queda registrada con verificación de integridad.")
    st.stop()

rows = P.read_audit()
if not rows:
    st.info("Aún no hay evaluaciones. Cada vez que pulse «Calcular estadio» quedará registrada aquí.")
    st.stop()

df = pd.DataFrame([{"fecha": pd.to_datetime(r["timestamp"], utc=True).tz_convert(None) + pd.Timedelta(hours=-5),
                    "estadio": r["stage"], "probabilidad": max(r["probabilities"]),
                    "edad": r["inputs"].get("age"), "sexo": {1: "Mujer", 0: "Hombre"}.get(r["inputs"].get("sex"), "—"),
                    "creatinina": r["inputs"].get("creatinine"), "modelo": r["model_version"],
                    "hash": r["record_hash"][:12], "_i": i} for i, r in enumerate(rows)])

check = P.verify_audit()
k = st.columns(4)
k[0].metric("Evaluaciones", len(df))
k[1].metric("Última", df["fecha"].max().strftime("%d/%m %H:%M"))
k[2].metric("Estadio más frecuente", df["estadio"].mode().iat[0])
k[3].metric("Integridad del registro", "✅ Íntegro" if check["ok"] else f"⚠️ Alterado (fila {check['broken_at']})")

with st.container(border=True):
    c = st.columns([2, 2])
    sel = c[0].multiselect("Filtrar por estadio", ui.STAGES, default=ui.STAGES)
    dmin, dmax = df["fecha"].min().date(), df["fecha"].max().date()
    rango = c[1].date_input("Rango de fechas", (dmin, dmax), min_value=dmin, max_value=dmax)
view = df[df["estadio"].isin(sel)]
if isinstance(rango, tuple) and len(rango) == 2:
    view = view[(view["fecha"].dt.date >= rango[0]) & (view["fecha"].dt.date <= rango[1])]

left, right = st.columns([2, 1])
with left:
    st.dataframe(view.drop(columns="_i").sort_values("fecha", ascending=False), width="stretch", hide_index=True,
                 column_config={"fecha": st.column_config.DatetimeColumn("Fecha", format="DD/MM/YYYY HH:mm"),
                                "probabilidad": st.column_config.NumberColumn("Probabilidad", format="percent")})
with right:
    counts = view["estadio"].value_counts().reindex(ui.STAGES).fillna(0)
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=ui.COLORS))
    fig.update_layout(height=280, title="Evaluaciones por estadio", margin=dict(t=40, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

st.download_button("⬇️ Exportar historial (CSV)", view.drop(columns="_i").to_csv(index=False, sep=";").encode("utf-8-sig"),
                   "korex_historial.csv", "text/csv")

st.divider()
st.subheader("Detalle de una evaluación")
if len(view):
    opts = view.sort_values("fecha", ascending=False)["_i"].tolist()
    i = st.selectbox("Evaluación", opts, format_func=lambda j: f"{df.loc[j, 'fecha']:%d/%m/%Y %H:%M} · "
                                                             f"{df.loc[j, 'estadio']} · {df.loc[j, 'hash']}")
    r = rows[i]
    c = st.columns([1, 1.3])
    with c[0]:
        st.plotly_chart(ui.prob_chart(r["probabilities"], height=250), width="stretch")
    with c[1]:
        st.dataframe(pd.DataFrame(P.patient_table(r["inputs"])).astype(str).rename(columns=str.capitalize), width="stretch",
                     hide_index=True, height=250)
    if st.button("🩺 Cargar este paciente en «Evaluar paciente»"):
        ui.load_record("p", r["inputs"], P)
        st.session_state.pop("result", None)
        st.switch_page(str(Path(__file__).with_name("evaluar.py")))
