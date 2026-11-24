"""KORE-X · Dashboard clínico (Streamlit, multipágina). Ejecutar desde la raíz del proyecto:
    streamlit run dashboard/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from ui import LOGO  # noqa: E402

st.set_page_config(page_title="KORE-X · Estadio renal", page_icon="🩺", layout="wide")
if LOGO.exists():
    st.logo(str(LOGO), size="large")

V = Path(__file__).resolve().parent / "views"
pages = {
    "Atención clínica": [
        st.Page(V / "evaluar.py", title="Evaluar paciente", icon="🩺", default=True),
        st.Page(V / "simulador.py", title="Simulador ¿qué pasaría si…?", icon="🎛️"),
        st.Page(V / "comparar.py", title="Comparar pacientes", icon="⚖️"),
    ],
    "Datos": [
        st.Page(V / "historial.py", title="Historial", icon="🗂️"),
        st.Page(V / "carga.py", title="Carga masiva", icon="📂"),
    ],
    "Modelo": [
        st.Page(V / "desempeno.py", title="Desempeño del modelo", icon="📊"),
        st.Page(V / "biomarcadores.py", title="¿Qué aportan los biomarcadores?", icon="🧬"),
        st.Page(V / "acerca.py", title="Acerca de KORE-X", icon="ℹ️"),
    ],
}
nav = st.navigation(pages)
with st.sidebar:
    st.caption("Herramienta de apoyo para profesionales de salud. No sustituye el juicio clínico.")
nav.run()
