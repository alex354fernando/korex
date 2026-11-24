"""Apartado 6 · Desempeño del modelo (validación cruzada anidada)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402

P = ui.setup("📊 Desempeño del modelo",
             "Validación cruzada anidada: cada paciente se evalúa con un modelo que no lo vio al entrenar.")
if P is None:
    st.stop()
MAIN = {"precision_macro": "Precision", "recall_macro": "Recall", "f1_macro": "F1-Score", "auc_ovr": "AUC-ROC"}
ev = P.evaluation()
if not ev or ui.SCEN not in ev:
    st.info("El modelo se entrenó sin validación anidada (--no-eval).")
    st.stop()

m = ev[ui.SCEN]
cols = st.columns(4)
for col, (k, lbl) in zip(cols, MAIN.items()):
    if k in m:
        col.markdown(f'<div class="kx-kpi"><div class="l">{lbl} (macro)</div><div class="v">{m[k]["mean"]:.3f}</div>'
                     f'<div class="l">± {m[k]["std"]:.3f} entre pliegues</div></div>', unsafe_allow_html=True)
st.write("")

pooled = P.pooled_oof(ui.SCEN)
t1, t2, t2b, t3, t4 = st.tabs(["Por estadio", "Curvas ROC", "Curvas PR", "Matriz de confusión", "Comparación y técnica"])
if pooled:
    pm = pooled["metrics"]
    with t1:
        st.dataframe(pd.DataFrame({"Precision": pm["precision_per_class"], "Recall": pm["recall_per_class"],
                                   "F1-Score": pm["f1_per_class"],
                                   "AUC-ROC": [pooled["roc"].get(k, {}).get("auc", np.nan) for k in range(len(ui.STAGES))]},
                                  index=ui.STAGES).round(3), width="stretch")
        st.caption("Calculado sobre todas las predicciones fuera de muestra de la validación.")
    with t2:
        fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False)])
        for k, d in pooled["roc"].items():
            fig.add_trace(go.Scatter(x=d["fpr"], y=d["tpr"], mode="lines", line=dict(color=ui.COLORS[k], width=2.5),
                                     name=f"{ui.STAGES[k]} (AUC {d['auc']:.3f})"))
        fig.update_layout(height=430, xaxis_title="1 − especificidad", yaxis_title="Sensibilidad",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
    with t2b:
        fig = go.Figure()
        for k, d in pooled["pr"].items():
            fig.add_trace(go.Scatter(x=d["recall"], y=d["precision"], mode="lines", line=dict(color=ui.COLORS[k], width=2.5),
                                     name=f"{ui.STAGES[k]} (AP {d['ap']:.3f})"))
        fig.update_layout(height=430, xaxis_title="Recall (sensibilidad)", yaxis_title="Precision", yaxis_range=[0, 1.03],
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Precisión frente a exhaustividad por estadio (AP = área bajo la curva PR); complementa a la ROC.")
    with t3:
        cm = np.array(pm["confusion_matrix"])
        fig = go.Figure(go.Heatmap(z=cm, x=[f"Pred. {s}" for s in ui.STAGES], y=[f"Real {s}" for s in ui.STAGES],
                                   colorscale="Teal", text=cm, texttemplate="%{text}", showscale=False))
        fig.update_layout(height=430, yaxis_autorange="reversed", margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
with t4:
    names = {"A": "KORE-X", "B": "Tamizaje sin creatinina", "rule_tree": "Regla (árbol con TFGe)", "prior": "Sin información"}
    keys = {**MAIN, "accuracy": "Exactitud", "qwk": "κ ponderado", "mae_ordinal": "Error medio (estadios)",
            "brier": "Brier", "ece": "ECE (calibración)"}
    st.markdown("**Comparación de modelos**")
    st.dataframe(pd.DataFrame({names[g]: {lbl: f"{ev[g][k]['mean']:.3f} ± {ev[g][k]['std']:.3f}" for k, lbl in keys.items() if k in ev[g]}
                               for g in ("A", "B", "rule_tree", "prior") if g in ev}), width="stretch")
    cmp_ = ev.get("comparison", {}).get("f1_macro")
    if cmp_:
        from korex.stats import MODEL_NAMES
        st.markdown(f"**KORE-X frente a cada modelo por separado** · prueba de Friedman p = {cmp_['p_value']:.4f}; "
                    f"diferencia crítica de Nemenyi = {cmp_['cd']:.2f} rangos ({cmp_['n_folds']} pliegues)")
        st.dataframe(pd.DataFrame({
            "F1 macro": {MODEL_NAMES.get(m, m): f"{cmp_['mean'][m]:.3f} ± {cmp_['std'][m]:.3f}" for m in cmp_["models"]},
            "Rango medio (1 = mejor)": {MODEL_NAMES.get(m, m): round(cmp_["mean_rank"][m], 2) for m in cmp_["models"]},
            "Sin diferencia con el mejor": {MODEL_NAMES.get(m, m): m in cmp_["not_different_from_best"] for m in cmp_["models"]},
        }).sort_values("Rango medio (1 = mejor)"), width="stretch")
    if "circularity" in ev:
        st.markdown("**Índice de circularidad IC_M** · fracción de la ventaja sobre el azar que se debe a la creatinina")
        st.dataframe(pd.DataFrame({"IC_M": {MAIN.get(k, k): round(v, 3) for k, v in ev["circularity"].items()}}), width="stretch")
    if "ablation" in ev:
        lab = {"sin_balanceo": "Sin balanceo", "smote_nc_clasico": "SMOTE-NC clásico", "oa_smote_nc": "OA-SMOTE-NC"}
        st.markdown("**Ablación del balanceo (datos sintéticos)**")
        st.dataframe(pd.DataFrame({lab.get(g, g): {MAIN[k]: f"{v[k]['mean']:.3f} ± {v[k]['std']:.3f}" for k in MAIN if k in v}
                                   for g, v in ev["ablation"].items()}), width="stretch")
    cal = P.calibration(ui.SCEN)
    if cal:
        fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Perfecta")])
        for k, d in cal.items():
            fig.add_trace(go.Scatter(x=d["mean_predicted"], y=d["fraction_positive"], mode="lines+markers",
                                     name=ui.STAGES[int(k)], line=dict(color=ui.COLORS[int(k)])))
        fig.update_layout(height=340, title="Curvas de calibración", xaxis_title="Probabilidad predicha",
                          yaxis_title="Frecuencia observada", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
    meta = P.metadata["scenarios"].get(ui.SCEN, {})
    ens, rs = meta.get("ensemble", {}), meta.get("resampling", {})
    c = st.columns(2)
    if ens:
        c[0].markdown("**Ensamble** (Q de Yule medio; > 0,9 = redundante)")
        c[0].dataframe(pd.DataFrame({"Q medio": ens.get("yule_q_mean", {}),
                                     "Seleccionado": {k: k in ens.get("active_models", []) for k in ens.get("yule_q_mean", {})}}),
                       width="stretch")
    if rs:
        c[1].markdown("**Pacientes reales y sintéticos (OA-SMOTE-NC)**")
        real = [P.metadata["dataset"]["class_counts"].get(str(k)) for k in range(len(ui.STAGES))]
        c[1].dataframe(pd.DataFrame({"Reales": real, "Sintéticos": [a - r for a, r in zip(rs.get("counts_after"), real)],
                                     "Total": rs.get("counts_after")}, index=ui.STAGES), width="stretch")
    st.caption(f"Modelo {P.model_version} · {P.metadata.get('kdigo_guideline')} · {P.metadata.get('equation')} · "
               f"dataset {P.metadata['dataset']['path']} (n = {P.metadata['dataset']['n']}) · "
               f"entrenamiento {P.metadata.get('training_seconds', 0) / 60:.0f} min")
