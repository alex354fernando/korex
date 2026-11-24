"""Componentes compartidos del dashboard KORE-X: estilo, formulario del paciente,
generadores (base, azar, típico por estadio) y tarjetas de resultado."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "dashboard"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from korex.pipeline import InferencePipeline, get_pipeline, load_config, validate_ranges  # noqa: E402

CFG = load_config()
K, VARS = CFG["kdigo"], CFG["variables"]
CATS, VLAB = CFG.get("categorical_values", {}), CFG.get("value_labels", {})
STAGES, COLORS = K["stage_labels"], K["class_colors"]
SCEN = "A"
SCREEN = "B"                      # modo tamizaje sin creatinina
INT_VARS = {"age", "bp", "bgr", "bu", "sod", "pcv", "wbcc"}
BRAND = "#0F766E"
SECTIONS = {
    "🧍 Datos básicos": ["age", "sex", "creatinine", "bp"],
    "🩸 Análisis de sangre": ["bgr", "bu", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"],
    "🧪 Análisis de orina": ["sg", "al", "su", "rbc", "pc", "pcc", "ba"],
    "📋 Antecedentes y síntomas": ["htn", "dm", "cad", "appet", "pe", "ane"],
}
LOGO = ROOT / "dashboard" / "assets" / "logo.svg"

CSS = f"""<style>
.block-container{{padding-top:1.6rem;max-width:1250px}}
.kx-hero{{background:linear-gradient(120deg,{BRAND} 0%,#134E4A 60%,#1E3A8A 100%);color:#fff;border-radius:18px;
  padding:1.2rem 1.6rem;margin-bottom:1.1rem;box-shadow:0 6px 22px rgba(15,118,110,.25)}}
.kx-hero h1{{color:#fff;font-size:1.65rem;margin:0}} .kx-hero p{{margin:.25rem 0 0;opacity:.9}}
.kx-card{{background:#fff;border:1px solid #E5E7EB;border-radius:16px;padding:1rem 1.2rem;
  box-shadow:0 2px 10px rgba(17,24,39,.05);margin-bottom:.8rem}}
.kx-kpi{{background:#fff;border:1px solid #E5E7EB;border-left:6px solid {BRAND};border-radius:14px;padding:.8rem 1rem}}
.kx-kpi .v{{font-size:1.7rem;font-weight:700;color:#111827}} .kx-kpi .l{{font-size:.85rem;color:#6B7280}}
.kx-result{{border-radius:18px;padding:1.2rem 1.5rem;color:#fff;box-shadow:0 6px 20px rgba(0,0,0,.15)}}
.kx-result h2{{color:#fff;margin:.5rem 0 0}}
.kx-lights{{background:#fff;border-radius:40px;padding:7px 4px 7px 12px;display:inline-block}}
.kx-light{{display:inline-block;width:30px;height:30px;border-radius:50%;margin-right:8px;opacity:.18;
  color:#fff;font-size:.7rem;font-weight:700;line-height:30px;text-align:center}}
.kx-light.on{{opacity:1;box-shadow:0 0 0 3px #111827}}
.kx-tag{{display:inline-block;padding:.15rem .6rem;border-radius:999px;font-size:.8rem;font-weight:600;color:#fff}}
div[data-testid="stMetric"]{{background:#fff;border:1px solid #E5E7EB;border-radius:14px;padding:.6rem .9rem}}
</style>"""


def setup(title: str, subtitle: str = "") -> InferencePipeline | None:
    """Estilo + encabezado + carga del modelo (cada apartado lo llama al inicio)."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="kx-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)
    if os.environ.get("KOREX_DEMO", "").strip() == "1":
        st.warning("Demostración pública con datos de ejemplo: no ingrese datos reales de pacientes. "
                   "Las evaluaciones no se guardan.")
    try:
        return get_pipeline()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("En una terminal ejecute:  `python scripts/train.py --profile quick`")
        return None


def show(v) -> str:
    """Valor legible en español (yes → Sí, 1.02 → 1.020)."""
    if v is None:
        return "—"
    if isinstance(v, str):
        return VLAB.get(v, v)
    if isinstance(v, float) and 1.0 < v < 1.1:
        return f"{v:.3f}"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def stage_tag(k: int) -> str:
    return f'<span class="kx-tag" style="background:{COLORS[k]}">{STAGES[k]}</span>'


# ------------------------------------------------------------------ estado del paciente
def options_for(f: str, P: InferencePipeline):
    if f in CATS:
        return CATS[f]
    rng = P.metadata.get("random_ranges", {}).get(f, {})
    if f in CFG["data"]["ordinal"] and rng.get("type") == "choice":
        return [float(v) for v in rng["values"]]
    return None


def _store(prefix: str, f: str, v, P):
    opts = options_for(f, P)
    if opts:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = float(v)
        v = v if v in opts else opts[0]
    elif f in INT_VARS:
        v = int(round(float(v)))
    else:
        v = float(round(float(v), 2))
    st.session_state[f"{prefix}_{f}"] = v


def load_record(prefix: str, record: dict, P):
    """Carga un registro (dict) en el formulario `prefix`."""
    st.session_state[f"{prefix}_sex"] = "Mujer" if record.get("sex", 1) == 1 else "Hombre"
    st.session_state[f"{prefix}_unit"] = "µmol/L" if record.get("creatinine_unit") == "umol/L" else "mg/dL"
    for f in P.features(SCEN):
        if f in record and f != "sex" and record[f] is not None:
            _store(prefix, f, record[f], P)


def set_base(prefix: str, P):
    load_record(prefix, CFG["base_patient"], P)


def _sample(prefix, P, ranges, rng):
    for f in P.features(SCEN):
        r = ranges.get(f)
        if f == "sex" or not r:
            continue
        if r["type"] == "choice":
            vals = r["values"]
            idx = rng.choice(len(vals), p=np.asarray(r["p"]) / np.sum(r["p"])) if "p" in r else rng.integers(len(vals))
            _store(prefix, f, vals[int(idx)], P)
        else:
            lo = max(r["min"], 18) if f == "age" else r["min"]
            _store(prefix, f, rng.uniform(lo, max(lo, r["max"])), P)


def set_random(prefix: str, P):
    rng = np.random.default_rng()
    st.session_state[f"{prefix}_sex"] = str(rng.choice(["Mujer", "Hombre"]))
    st.session_state[f"{prefix}_unit"] = "mg/dL"
    _sample(prefix, P, P.metadata.get("random_ranges", {}), rng)


def set_typical(prefix: str, P, stage_key: str):
    k = STAGES.index(st.session_state[stage_key])
    rng = np.random.default_rng()
    st.session_state[f"{prefix}_sex"] = str(rng.choice(["Mujer", "Hombre"]))
    st.session_state[f"{prefix}_unit"] = "mg/dL"
    _sample(prefix, P, P.metadata.get("stage_ranges", {}).get(str(k), {}), rng)


def ensure(prefix: str, P):
    if f"{prefix}_age" not in st.session_state:
        set_base(prefix, P)


def generator_bar(prefix: str, P, on_change=None):
    """Botones: paciente base, aleatorio y típico de un estadio."""
    def wrap(fn, *a):
        def _f():
            fn(prefix, P, *a)
            if on_change:
                on_change()
        return _f
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 0.7, 1.4])
    c1.button("🧍 Paciente base (sano)", on_click=wrap(set_base), width="stretch", key=f"{prefix}_bbase")
    c2.button("🎲 Aleatorizar todo", on_click=wrap(set_random), width="stretch", key=f"{prefix}_brand")
    c3.selectbox("Estadio", STAGES, key=f"{prefix}_typ", label_visibility="collapsed")
    c4.button("🎯 Paciente típico del estadio", on_click=wrap(set_typical, f"{prefix}_typ"), width="stretch",
              key=f"{prefix}_btyp", disabled="stage_ranges" not in P.metadata)


def _widget(prefix: str, f: str, P):
    meta = VARS.get(f, {"label": f, "unit": ""})
    key = f"{prefix}_{f}"
    if f == "sex":
        st.radio("Sexo", ["Mujer", "Hombre"], horizontal=True, key=key)
        return
    if f == "creatinine":
        unit = st.session_state.get(f"{prefix}_unit", "mg/dL")
        st.number_input(f"Creatinina en sangre ({unit})", min_value=0.05, step=0.1, format="%.2f", key=key)
        st.radio("Unidad de creatinina", ["mg/dL", "µmol/L"], horizontal=True, key=f"{prefix}_unit")
        return
    lbl = meta["label"] + (f" ({meta['unit']})" if meta.get("unit") else "")
    opts = options_for(f, P)
    if opts:
        st.selectbox(lbl, opts, key=key, format_func=show)
    elif f == "age":
        st.number_input(lbl, min_value=18, max_value=110, step=1, key=key)
    elif f in INT_VARS:
        st.number_input(lbl, step=1, key=key)
    else:
        st.number_input(lbl, step=0.1, format="%.2f", key=key)


def patient_form(prefix: str, P, columns: int = 4, compact: bool = False, screening: bool = False):
    """Formulario completo en una sola vista, agrupado por secciones con subtítulos."""
    feats = set(P.features(SCEN)) | {"sex"}
    if screening:
        feats.discard("creatinine")
    sections = {name: [f for f in fs if f in feats] for name, fs in SECTIONS.items()}
    known = {f for fs in SECTIONS.values() for f in fs}
    extra = [f for f in P.features(SCEN) if f not in known]
    if extra:
        sections["➕ Otros"] = extra
    for i, (name, fs) in enumerate(sections.items()):
        if not fs:
            continue
        if i:
            st.markdown('<hr style="margin:.4rem 0 .6rem;border:none;border-top:1px solid #E5E7EB">', unsafe_allow_html=True)
        st.markdown(f"**{name}**" if compact else f"##### {name}")
        cols = st.columns(columns)
        for j, f in enumerate(fs):
            with cols[j % columns]:
                _widget(prefix, f, P)


def record_from(prefix: str, P, screening: bool = False) -> dict:
    ss = st.session_state
    rec = {"sex": 1 if ss.get(f"{prefix}_sex", "Mujer") == "Mujer" else 0,
           "creatinine_unit": "umol/L" if ss.get(f"{prefix}_unit") == "µmol/L" else "mg/dL"}
    for f in P.features(SCEN):
        if f != "sex":
            rec[f] = ss.get(f"{prefix}_{f}")
    if screening:
        rec["creatinine"] = None
    return rec


def has_screening(P) -> bool:
    return SCREEN in P.models


def range_warnings(record: dict):
    check = dict(record)
    if check.get("creatinine") and check.get("creatinine_unit") == "umol/L":
        check["creatinine"] = check["creatinine"] / 88.4
    for w in validate_ranges({k: v for k, v in check.items() if v is not None}, CFG):
        st.warning(w, icon="⚠️")


# ------------------------------------------------------------------ resultados
def result_card(res: dict) -> str:
    lights = "".join(f'<span class="kx-light{" on" if i == res["risk_class"] else ""}" style="background:{c}">{STAGES[i]}</span>'
                     for i, c in enumerate(COLORS))
    return (f'<div class="kx-result" style="background:{res["color"]}"><span class="kx-lights">{lights}</span>'
            f'<h2>{res["risk_name"]}</h2>'
            f'<div style="opacity:.95">Estadio estimado {STAGES[res["risk_class"]]} · probabilidad {res["probability"] * 100:.0f} %</div>'
            f'<h4 style="margin:.8rem 0 .2rem;color:#fff">➜ {res["recommendation"]["level"]}</h4>'
            f'<div>{res["recommendation"]["text"]}</div></div>')


def screening_card(res: dict) -> str:
    sc = res["screening"]
    col = {"alto": "#6A1B9A", "moderado": "#EF6C00", "bajo": "#2E7D32"}[sc["level"]]
    bar = lambda lbl, p: (f'<div style="margin:.35rem 0"><div style="display:flex;justify-content:space-between">'
                          f'<span>{lbl}</span><b>{p * 100:.0f} %</b></div><div style="background:rgba(255,255,255,.3);'
                          f'border-radius:6px;height:10px"><div style="width:{p * 100:.0f}%;background:#fff;height:10px;'
                          f'border-radius:6px"></div></div></div>')
    return (f'<div class="kx-result" style="background:{col}"><span class="kx-tag" style="background:rgba(0,0,0,.25)">'
            f'MODO TAMIZAJE · SIN CREATININA</span><h2>{res["recommendation"]["level"]}</h2>'
            f'{bar("Probabilidad de TFGe &lt; 60 (G3–G5)", sc["p_lt60"])}{bar("Probabilidad de TFGe &lt; 30 (G4–G5)", sc["p_lt30"])}'
            f'<div style="margin-top:.5rem;opacity:.95">Estadio probable: <b>{sc["range"][0]} a {sc["range"][1]}</b></div>'
            f'<div style="margin-top:.6rem">{res["recommendation"]["text"]}</div></div>')


def screening_note(res: dict, P=None):
    """Confiabilidad para ESTE paciente: qué tenían en realidad los pacientes de validación con probabilidades parecidas."""
    sc = res["screening"]
    rel = P.screening_reliability(sc["p_lt60"], sc["p_lt30"]) if P is not None else None
    if rel:
        cnt = rel["stage_counts"]
        dist = ", ".join(f"{STAGES[i]}: {c}" for i, c in enumerate(cnt) if c)
        intro = f"**¿Qué tan confiable es?** De los {rel['n']} pacientes de validación con probabilidades parecidas a estas, "
        if sc["level"] == "bajo":
            msg = (f"el **{rel['frac_healthy']:.0%}** tenía función renal normal (G1–G2) y solo el "
                   f"{rel['frac_lt60']:.0%} tenía TFGe < 60.")
        else:
            main = (f"**{rel['frac_lt30']:.0%}** tenía realmente TFGe < 30" if sc["level"] == "alto" else
                    f"**{rel['frac_lt60']:.0%}** tenía realmente TFGe < 60")
            healthy = ("ninguno tenía función renal normal (G1–G2)" if rel["frac_healthy"] == 0 else
                       f"el {rel['frac_healthy']:.0%} tenía función renal normal (G1–G2)")
            msg = f"el {main} y {healthy}."
        st.markdown(intro + msg)
        st.caption(f"Estadio real de esos {rel['n']} pacientes → {dist}. El estadio KDIGO definitivo requiere creatinina sérica.")
    else:
        v = sc.get("validated", {})
        if v:
            st.caption(f"Desempeño validado sin creatinina: detecta TFGe < 60 con sensibilidad {v.get('sens_lt60', 0):.0%} y "
                       f"especificidad {v.get('spec_lt60', 0):.0%} (AUC {v.get('auc_lt60', 0):.2f}). "
                       "El estadio KDIGO definitivo requiere creatinina sérica.")


def prob_chart(probs, title="Probabilidad de cada estadio", height=290) -> go.Figure:
    fig = go.Figure(go.Bar(x=STAGES, y=probs, marker_color=COLORS, text=[f"{p * 100:.0f} %" for p in probs],
                           textposition="outside"))
    fig.update_layout(height=height, yaxis=dict(range=[0, 1.12], tickformat=".0%"), margin=dict(t=40, b=10, l=10, r=10),
                      title=title, title_font_size=14, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def factor_chart(explanation: dict, n: int = 8) -> go.Figure:
    f = explanation["factors"][:n][::-1]
    lab = [d["label"] + ("" if d["value"] is None else f" = {show(d['value'])} {d['unit']}".rstrip()) for d in f]
    fig = go.Figure(go.Bar(x=[d["shap"] for d in f], y=lab, orientation="h",
                           marker_color=["#C62828" if d["shap"] > 0 else "#2E7D32" for d in f]))
    fig.update_layout(height=70 + 38 * len(f), margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Efecto sobre el estadio",
                      title="🔴 empujan hacia un estadio más avanzado · 🟢 indican función renal conservada",
                      title_font_size=13, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def formula_note(res: dict):
    r = res.get("kdigo_rule")
    if not r:
        return
    st.markdown(f"**Referencia con la fórmula CKD-EPI 2021:** TFGe **{r['egfr']}** mL/min/1,73 m² → "
                f"{stage_tag(r['risk_class'])}", unsafe_allow_html=True)
    if r["risk_class"] != res["risk_class"]:
        st.warning("La estimación del modelo difiere del estadio calculado con la fórmula. "
                   "Revise los valores; ante la duda prevalece la TFGe medida.", icon="❗")
