"""Apartado · ¿Qué aportan los demás biomarcadores? (con y sin creatinina, frente a la literatura)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from korex.literature import COLUMNS, LITERATURE, korex_rows  # noqa: E402

P = ui.setup("🧬 ¿Qué aportan los demás biomarcadores?",
             "Por qué la creatinina domina cuando está disponible y qué información dan los demás análisis cuando falta.")
if P is None:
    st.stop()

meta, ev = P.metadata.get("scenarios", {}), P.evaluation()
G = P.global_shap() or {}
lab = lambda v: ui.VARS.get(v, {}).get("label", v)


def share_chart(share: dict, color: str, title: str, n: int = 12) -> go.Figure:
    items = list(share.items())[:n][::-1]
    fig = go.Figure(go.Bar(x=[100 * v for _, v in items], y=[lab(k) for k, _ in items], orientation="h",
                           marker_color=color, text=[f"{100 * v:.1f} %" for _, v in items], textposition="outside"))
    fig.update_layout(height=60 + 30 * len(items), title=title, title_font_size=14, xaxis_title="% de la explicación (SHAP)",
                      xaxis=dict(range=[0, 110]), margin=dict(l=10, r=10, t=45, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def beeswarm(g: dict, n: int = 10) -> go.Figure:
    """Cada punto es un paciente: posición = efecto SHAP, color = valor de la variable (azul bajo → rojo alto)."""
    S, names = np.asarray(g["shap"]), g["variables"]
    order = np.argsort(np.abs(S).mean(0))[::-1][:n][::-1]
    fig, rng = go.Figure(), np.random.default_rng(0)
    for row, j in enumerate(order):
        v = np.asarray(g["values"][names[j]], float)
        ok = ~np.isnan(v)
        c = np.full(len(v), 0.5)
        if ok.sum() > 1 and np.ptp(v[ok]) > 0:
            lo, hi = np.nanpercentile(v, [5, 95])
            c[ok] = np.clip((v[ok] - lo) / max(hi - lo, 1e-9), 0, 1)
        fig.add_trace(go.Scatter(x=S[:, j], y=row + rng.uniform(-.28, .28, len(v)), mode="markers", showlegend=False,
                                 marker=dict(size=6, color=c, colorscale="RdBu_r", cmin=0, cmax=1, opacity=.8,
                                             showscale=row == len(order) - 1,
                                             colorbar=dict(title="Valor", tickvals=[0, 1], ticktext=["bajo", "alto"])),
                                 hovertemplate=f"{lab(names[j])}<br>SHAP %{{x:.3f}}<extra></extra>"))
    fig.update_layout(height=80 + 34 * len(order), yaxis=dict(tickvals=list(range(len(order))),
                                                              ticktext=[lab(names[j]) for j in order]),
                      xaxis_title="Efecto sobre el estadio esperado (SHAP)", margin=dict(l=10, r=10, t=20, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.add_vline(x=0, line_color="#9CA3AF")
    return fig


# ---------------------------------------------------------------- 1. con creatinina
st.subheader("1 · Con creatinina: el modelo redescubre la definición KDIGO")
active = meta.get("A", {}).get("ensemble", {}).get("active_models", [])
names = {"lr_core": "regresión logística sobre creatinina y edad", "xgb": "XGBoost", "lgbm": "LightGBM",
         "cat": "CatBoost", "et": "Extra Trees", "lr": "regresión logística", "svm": "SVM"}
c = st.columns([1, 1.1])
with c[0]:
    st.markdown(
        "El estadio G se **define** con la TFGe, y la TFGe se calcula con **creatinina, edad y sexo** (CKD-EPI 2021). "
        "Por eso, cuando la creatinina está disponible, cualquier modelo bien entrenado debe apoyarse casi solo en ella: "
        "las demás variables no agregan información que la fórmula no tenga ya.")
    if active:
        st.markdown(f"La selección automática del ensamble se quedó con: **{', '.join(names.get(a, a) for a in active)}**. "
                    "Es decir, el sistema encontró por sí mismo la estructura de la fórmula, sin que se la diéramos.")
    st.markdown("Los estudios de estadificación de la literatura muestran el mismo patrón (por ejemplo, Hegde et al., 2025: "
                "TFGe con SHAP 1,16 frente a 0,02 de la siguiente variable).")
with c[1]:
    if meta.get("A", {}).get("global_share"):
        st.plotly_chart(share_chart(meta["A"]["global_share"], ui.BRAND, "Importancia con creatinina"), width="stretch")

# ---------------------------------------------------------------- 2. sin creatinina
st.subheader("2 · Sin creatinina: los demás biomarcadores sí informan")
if "B" not in meta:
    st.info("Vuelva a entrenar el modelo para generar el modo tamizaje: `python scripts/train.py --profile standard`")
    st.stop()
c = st.columns([1.1, 1])
with c[0]:
    if meta["B"].get("global_share"):
        st.plotly_chart(share_chart(meta["B"]["global_share"], "#EF6C00", "Importancia sin creatinina"), width="stretch")
with c[1]:
    st.markdown("Al quitar la creatinina, el modelo reparte la explicación entre marcadores con sentido fisiopatológico:")
    st.markdown("- **Urea**: también se acumula cuando cae el filtrado.\n"
                "- **Albúmina en orina** y **densidad urinaria**: daño glomerular y pérdida de la capacidad de concentrar.\n"
                "- **Hemoglobina / hematocrito**: anemia de la ERC (menor eritropoyetina).\n"
                "- **Edad**: el filtrado desciende con los años.")
    eb = ev.get("B", {})
    if eb:
        k = st.columns(3)
        k[0].metric("AUC TFGe<60", f"{eb.get('auc_lt60', {}).get('mean', np.nan):.2f}")
        k[1].metric("AUC TFGe<30", f"{eb.get('auc_lt30', {}).get('mean', np.nan):.2f}")
        k[2].metric("Estadio a ±1", f"{eb.get('accuracy_within1', {}).get('mean', np.nan):.0%}")
if G.get("B"):
    with st.expander("Ver cada paciente (beeswarm SHAP sin creatinina)", expanded=True):
        st.plotly_chart(beeswarm(G["B"]), width="stretch")
        st.caption("Cada punto es un paciente. A la derecha empuja hacia un estadio más avanzado. "
                   "Rojo = valor alto de la variable, azul = valor bajo (p. ej., urea alta y hemoglobina baja → estadio avanzado).")

# ---------------------------------------------------------------- 3. comparación
st.subheader("3 · Lo que cada modelo puede y no puede hacer")
rows = {"Exactitud (estadio exacto)": "accuracy", "Acierto a ±1 estadio": "accuracy_within1", "F1 macro": "f1_macro",
        "AUC (5 estadios)": "auc_ovr", "AUC TFGe < 60": "auc_lt60", "Sensibilidad TFGe < 60": "sens_lt60",
        "Especificidad TFGe < 60": "spec_lt60", "AUC TFGe < 30": "auc_lt30"}
cols = {"A": "Con creatinina", "B": "Sin creatinina (tamizaje)", "prior": "Al azar (referencia)"}
st.dataframe(pd.DataFrame({cols[g]: {r: (f"{ev[g][k]['mean']:.3f}" if k in ev[g] else "—") for r, k in rows.items()}
                           for g in cols if g in ev}), width="stretch")
st.markdown("**Lectura para la tesis:** con creatinina el sistema estadifica; sin ella no puede dar el estadio exacto "
            "(la información que lo define no está), pero sí distingue bien a quién tiene el filtrado disminuido. "
            "Esa es la utilidad de los demás biomarcadores: **priorizar a quién pedir creatinina** en lugares donde no "
            "está disponible de inmediato, como proponen los estudios de tamizaje (#3, #19).")
if "circularity" in ev:
    ic = ev["circularity"].get("f1_macro")
    if ic is not None:
        st.caption(f"Índice de circularidad IC_F1 = {ic:.2f}: fracción de la ventaja sobre el azar que depende de la "
                   "creatinina (1 = toda, 0 = nada).")

# ---------------------------------------------------------------- 4. literatura
st.subheader("4 · Comparación con la literatura")
lit = pd.DataFrame(korex_rows(P.metadata, ui.CFG) + LITERATURE).rename(columns=COLUMNS)
st.dataframe(lit, width="stretch", hide_index=True)
st.caption("Los estudios binarios (ERC sí/no) predicen el diagnóstico médico, no una fórmula; por eso sus variables "
           "importantes son otras y sus cifras no son comparables con la estadificación. Varios estudios de estadificación "
           "usan la TFGe como entrada, lo que KORE-X evita.")
