"""Apartado 5 · Carga masiva de pacientes desde Excel/CSV/ARFF."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from korex.pipeline import read_table  # noqa: E402

P = ui.setup("📂 Carga masiva", "Clasifique muchos pacientes a la vez desde un archivo.")
if P is None:
    st.stop()

with st.container(border=True):
    st.markdown("**1 · Suba su archivo** (Excel, CSV o ARFF con los nombres de columna del dataset UCI: "
                "`age`, `sc`, `bu`, `hemo`, `htn`, …)")
    up = st.file_uploader("Arrastre aquí su archivo", type=["xlsx", "xls", "csv", "arff"], label_visibility="collapsed")
if up:
    name = up.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(up)
    elif name.endswith(".arff"):
        with tempfile.NamedTemporaryFile(suffix=".arff", delete=False) as tmp:
            tmp.write(up.getvalue())
        df = read_table(tmp.name, ui.CFG)
    else:
        df = pd.read_csv(up, sep=None, engine="python", encoding=ui.CFG["data"]["encoding"])
    with st.container(border=True):
        st.markdown(f"**2 · Revise la vista previa** ({len(df)} filas, {df.shape[1]} columnas)")
        st.dataframe(df.head(10).astype(str), width="stretch")
        if st.button("Clasificar todos", type="primary"):
            with st.spinner("Clasificando…"):
                try:
                    st.session_state["batch"] = P.predict_batch(df, ui.SCEN)
                except (ValueError, KeyError) as e:
                    st.error(f"No se pudo procesar el archivo: {e}")

out = st.session_state.get("batch")
if out is not None:
    st.markdown("**3 · Resultados**")
    k = st.columns(3)
    k[0].metric("Pacientes clasificados", len(out))
    k[1].metric("Estadio más frecuente", out["estadio"].mode().iat[0])
    valid = out["estadio_por_tfge"] != ""
    if valid.any():
        k[2].metric("Coincidencia con la fórmula", f"{(out['estadio'] == out['estadio_por_tfge'])[valid].mean() * 100:.1f} %")
    counts = out["estadio"].value_counts().reindex(ui.STAGES).fillna(0)
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=ui.COLORS, text=counts.values.astype(int),
                           textposition="outside"))
    fig.update_layout(height=280, margin=dict(t=20, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(out, width="stretch")
    st.download_button("⬇️ Descargar resultados (CSV)", out.to_csv(index=False, sep=";").encode("utf-8-sig"),
                       "korex_resultados.csv", "text/csv", type="primary")
