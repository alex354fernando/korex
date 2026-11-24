"""Apartado 7 · Acerca de KORE-X: qué hace, cómo funciona y sus límites."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

import ui  # noqa: E402

P = ui.setup("ℹ️ Acerca de KORE-X", "Qué hace el sistema, cómo funciona y qué límites tiene.")

st.markdown("### ¿Qué es?")
st.write("KORE-X estima el **estadio de filtrado glomerular KDIGO (G1–G5)** de un adulto a partir de sus datos "
         "clínicos y de laboratorio, explica qué variables influyeron y genera un informe descargable.")

st.markdown("### Estadios KDIGO")
rng = ["≥ 90", "60 – 89", "30 – 59", "15 – 29", "< 15"]
cols = st.columns(5)
for i, c in enumerate(cols):
    c.markdown(f'<div class="kx-card" style="border-top:6px solid {ui.COLORS[i]};min-height:150px">'
               f'<h3 style="margin:0;color:{ui.COLORS[i]}">{ui.STAGES[i]}</h3>'
               f'<div style="font-size:.9rem">{ui.K["class_names"][i].split("· ")[1]}</div>'
               f'<div style="margin-top:.4rem;font-weight:600">{rng[i]}</div>'
               f'<div style="font-size:.8rem;color:#6B7280">mL/min/1,73 m²</div></div>', unsafe_allow_html=True)

st.markdown("### ¿Cómo funciona?")
steps = [("1. Datos", "Dataset UCI Chronic Kidney Disease (400 pacientes; 356 adultos con creatinina)."),
         ("2. Etiqueta", "Estadio G calculado con la ecuación CKD-EPI 2021 (sin raza)."),
         ("3. Preparación", "Imputación de faltantes (MissForest), escalado y codificación, ajustados solo con datos de entrenamiento."),
         ("4. Balanceo", "OA-SMOTE-NC crea pacientes sintéticos solo en entrenamiento para equilibrar los estadios."),
         ("5. Ensamble", "Siete modelos (XGBoost, LightGBM, CatBoost, Extra Trees, regresión logística, SVM y un modelo "
                         "lineal clínico) combinados con un meta-modelo ordinal."),
         ("6. Explicación", "SHAP y LIME indican qué variables empujan hacia un estadio más o menos avanzado.")]
cols = st.columns(3)
for i, (t, d) in enumerate(steps):
    cols[i % 3].markdown(f'<div class="kx-card"><b>{t}</b><br><span style="font-size:.9rem">{d}</span></div>',
                         unsafe_allow_html=True)

st.markdown("### Limitaciones")
st.markdown("""
- El dataset **no registra el sexo**: el entrenamiento usa la TFGe más conservadora (supuesto «mujer»), por lo que en
  hombres el modelo puede indicar un estadio algo más avanzado que la fórmula.
- La etiqueta proviene de una fórmula, no de un diagnóstico clínico confirmado.
- Muestra pequeña (356 adultos) de un solo centro: se requiere validación externa antes de cualquier uso clínico.
- Las recomendaciones son orientativas y **no sustituyen el juicio del profesional**.
""")
if P:
    st.caption(f"Modelo {P.model_version} · {P.metadata.get('kdigo_guideline')} · {P.metadata.get('equation')}")
