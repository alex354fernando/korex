"""Apartado 1 · Evaluar paciente."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402

P = ui.setup("🩺 Evaluar paciente", "Ingrese los datos clínicos y obtenga el estadio KDIGO estimado (G1–G5).")
if P is None:
    st.stop()
ui.ensure("p", P)


def _clear():
    st.session_state.pop("result", None)
    st.session_state.pop("explanation", None)


ui.generator_bar("p", P, on_change=_clear)
screening = ui.has_screening(P) and st.toggle(
    "🔎 No tengo creatinina (modo tamizaje)", key="p_screening", on_change=_clear,
    help="Estima la probabilidad de filtrado disminuido con los demás análisis, para decidir a quién pedir creatinina.")
if screening:
    st.info("Modo tamizaje: sin creatinina el estadio exacto es incierto. El sistema estima la probabilidad de "
            "TFGe < 60 y < 30 a partir de urea, albúmina, hemoglobina, densidad urinaria y los demás datos.", icon="🔎")
scen = ui.SCREEN if screening else ui.SCEN
with st.container(border=True):
    ui.patient_form("p", P, columns=4, screening=screening)
record = ui.record_from("p", P, screening=screening)
ui.range_warnings(record)

if st.button("Calcular estadio", type="primary", width="stretch"):
    try:
        res = P.predict(record, scen)
        try:
            P.log_audit(record, res)
        except OSError as e:
            st.caption(f"Auditoría no disponible: {e}")
        st.session_state.update(result=res, record=record, explanation=None, result_record=dict(record))
    except ValueError as e:
        st.error(str(e))

res = st.session_state.get("result")
if res and st.session_state.get("result_record") != record:
    st.info("Los datos del paciente cambiaron desde el último cálculo. Pulse **Calcular estadio** para actualizar el resultado.",
            icon="🔄")
    res = None
if res:
    st.divider()
    left, right = st.columns([1.15, 1])
    with left:
        if res.get("screening"):
            st.markdown(ui.screening_card(res), unsafe_allow_html=True)
            st.write("")
            ui.screening_note(res, P)
        else:
            st.markdown(ui.result_card(res), unsafe_allow_html=True)
            st.write("")
            ui.formula_note(res)
        st.caption(res["recommendation"]["disclaimer"])
    with right:
        st.plotly_chart(ui.prob_chart(res["probabilities"]), width="stretch")

    t1, t2, t3 = st.tabs(["🔍 ¿Por qué este resultado?", "📄 Informe", "➡️ Siguientes pasos"])
    with t1:
        ex = st.session_state.get("explanation")
        if ex is None:
            st.write("Calcula qué variables influyeron más en la estimación de este paciente (SHAP + LIME).")
            if st.button("Calcular explicación"):
                with st.spinner("Calculando explicación…"):
                    st.session_state["explanation"] = ex = P.explain(st.session_state["record"], res["scenario"])
        if ex:
            st.plotly_chart(ui.factor_chart(ex), width="stretch")
            st.success(ex["reading"])
            with st.expander("Detalles técnicos (SHAP, LIME y convergencia)"):
                cv = ex["convergence"]
                m = st.columns(4)
                m[0].metric("ICSL", f"{cv['icsl']:.2f}"); m[1].metric("Spearman |φ|", f"{cv['spearman']:.2f}")
                m[2].metric("Jaccard top-k", f"{cv['jaccard_topk']:.2f}"); m[3].metric("Concordancia de signo", f"{cv['sign_agreement']:.2f}")
                st.dataframe(pd.DataFrame([{"Variable": f["label"], "Valor": ui.show(f["value"]), "Unidad": f["unit"],
                                            "SHAP": round(f["shap"], 4), "LIME": round(f["lime"], 4)} for f in ex["factors"]]),
                             width="stretch", hide_index=True)
    with t2:
        ex = st.session_state.get("explanation")
        rep = P.build_report(res, ex, st.session_state["record"])
        if ex is None:
            st.info("Tip: calcule la explicación para incluir los factores clave en el informe.")
        c = st.columns(3)
        c[0].download_button("⬇️ Informe PDF", P.to_pdf(rep, res["color"]), "informe_korex.pdf", "application/pdf",
                             width="stretch", type="primary")
        c[1].download_button("⬇️ Informe HTML", P.render_report_html(rep, res["color"]), "informe_korex.html", "text/html",
                             width="stretch")
        c[2].download_button("⬇️ Datos JSON", json.dumps(rep, ensure_ascii=False, indent=2, default=str), "informe_korex.json",
                             width="stretch")
        st.dataframe(pd.DataFrame(rep["datos_paciente"]).astype(str).rename(columns=str.capitalize), width="stretch",
                     hide_index=True)
    with t3:
        st.write("Explore cómo cambiaría el estadio si se modifican algunos valores, o compárelo con otro paciente.")
        c = st.columns(2)
        if c[0].button("🎛️ Abrir en el simulador", width="stretch"):
            st.session_state["sim_source"] = st.session_state["record"]
            st.switch_page(str(Path(__file__).with_name("simulador.py")))
        if c[1].button("⚖️ Comparar con otro paciente", width="stretch"):
            ui.load_record("ca", st.session_state["record"], P)
            st.switch_page(str(Path(__file__).with_name("comparar.py")))
