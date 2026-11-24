"""Estudios del proyecto de tesis usados como referencia (tabla comparativa del artículo).
Valores tomados de cada artículo; «tarea» distingue estadificación (etiqueta = fórmula de TFGe)
de diagnóstico binario (etiqueta = juicio médico), porque no son comparables entre sí."""
from __future__ import annotations

LITERATURE = [
    # estadificación / multiclase
    {"ref": "#1 Hegde et al., 2025", "datos": "Hospital de Bangalore (aumentado a 5000)", "tarea": "Estadio G1–G5",
     "tfge_entrada": "Sí", "modelo": "CNN + selección XAI", "resultado": "Exactitud 99,5 %",
     "variable_dominante": "TFGe (SHAP 1,16 frente a 0,02 de la siguiente)"},
    {"ref": "#20 Akter et al., 2023 (CKD.Net)", "datos": "Corea, ~1 millón de registros", "tarea": "Estadio I–V",
     "tfge_entrada": "Sí", "modelo": "SimpleRNN + MLP", "resultado": "Exactitud 99,2–99,8 %",
     "variable_dominante": "TFGe y creatinina"},
    {"ref": "#4 Huang et al., 2025", "datos": "Shanghái, 1150 pacientes", "tarea": "Riesgo en 4 niveles",
     "tfge_entrada": "Sí (y el estadio)", "modelo": "XGBoost + Optuna", "resultado": "F1 91,9 % · AUC 97,7 %",
     "variable_dominante": "Estadio, ACR, TFGe"},
    {"ref": "#17 Huang et al., 2025", "datos": "Shanghái, 1150 pacientes", "tarea": "Riesgo en 4 niveles",
     "tfge_entrada": "Sí (y el estadio)", "modelo": "XGBoost + bandits", "resultado": "F1 91,0 % · AUC 97,8 %",
     "variable_dominante": "ACR, TFGe, estadio"},
    {"ref": "#6 Debal y Sitote, 2022", "datos": "Etiopía, 1718 registros", "tarea": "5 clases",
     "tfge_entrada": "No informado", "modelo": "XGBoost", "resultado": "Exactitud 82,6 %",
     "variable_dominante": "Creatinina, urea, hemoglobina, densidad urinaria"},
    {"ref": "#5 Ilyas et al., 2021", "datos": "UCI, estadio por CKD-EPI", "tarea": "Estadios",
     "tfge_entrada": "No", "modelo": "Árbol J48", "resultado": "Exactitud 85,5 %", "variable_dominante": "Sin XAI"},
    # diagnóstico binario (UCI)
    {"ref": "#14 Chouit et al., 2026", "datos": "UCI (n = 400)", "tarea": "Binaria (ERC sí/no)",
     "tfge_entrada": "No", "modelo": "XGBoost + SMOTE", "resultado": "Exactitud 94,6 % · AUC 0,948",
     "variable_dominante": "Densidad urinaria, hemoglobina, creatinina"},
    {"ref": "#7 Sci. Rep., 2026 (GWO)", "datos": "UCI", "tarea": "Binaria (ERC sí/no)", "tfge_entrada": "No",
     "modelo": "Ensamble ponderado por GWO", "resultado": "Exactitud 98,75 % · AUC 0,994",
     "variable_dominante": "Albúmina, glóbulos rojos, diabetes, creatinina"},
    {"ref": "#2 Chhabra et al., 2024", "datos": "UCI", "tarea": "Binaria (ERC sí/no)", "tfge_entrada": "No",
     "modelo": "Stacking (SVM, RF, AdaBoost)", "resultado": "Exactitud 99,5 %", "variable_dominante": "Sin XAI"},
    # sin creatinina (tamizaje)
    {"ref": "#3 Sci. Rep., 2026 (Buner)", "datos": "Pakistán, n = 650", "tarea": "Binaria, examen de orina",
     "tfge_entrada": "No", "modelo": "Ensamble ponderado", "resultado": "Exactitud 97,7 %", "variable_dominante": "Sin XAI"},
    {"ref": "#19 Qureshi et al., 2026", "datos": "380 pacientes, 20 variables", "tarea": "Binaria",
     "tfge_entrada": "No", "modelo": "KNN", "resultado": "Exactitud 94,7 %",
     "variable_dominante": "Albúmina, células de pus, bacterias"},
]


def korex_rows(metadata: dict, cfg: dict) -> list[dict]:
    """Filas de KORE-X (con y sin creatinina) a partir de los artefactos entrenados."""
    ev, sc, ds = metadata.get("evaluation", {}), metadata.get("scenarios", {}), metadata.get("dataset", {})
    label = lambda v: cfg["variables"].get(v, {}).get("label", v)
    rows = []
    for s, tarea, res_keys in (("A", "Estadio G1–G5", ("accuracy", "f1_macro", "auc_ovr")),
                               ("B", "Tamizaje TFGe < 60 sin creatinina", ("auc_lt60", "sens_lt60", "accuracy_within1"))):
        if s not in ev:
            continue
        names = {"accuracy": "Exactitud", "f1_macro": "F1 macro", "auc_ovr": "AUC", "auc_lt60": "AUC (<60)",
                 "sens_lt60": "Sensibilidad (<60)", "accuracy_within1": "±1 estadio"}
        res = " · ".join(f"{names[k]} {ev[s][k]['mean']:.3f}" for k in res_keys if k in ev[s])
        share = sc.get(s, {}).get("global_share", {})
        dom = ", ".join(f"{label(k)} {100 * v:.0f} %" for k, v in list(share.items())[:3]) if share else "—"
        rows.append({"ref": f"KORE-X ({'con' if s == 'A' else 'sin'} creatinina)",
                     "datos": f"UCI, n = {ds.get('n', '?')} adultos", "tarea": tarea, "tfge_entrada": "No",
                     "modelo": "Stacking ordinal + validación anidada", "resultado": res, "variable_dominante": dom})
    return rows


COLUMNS = {"ref": "Estudio", "datos": "Datos", "tarea": "Tarea", "tfge_entrada": "TFGe como entrada",
           "modelo": "Modelo", "resultado": "Resultado reportado", "variable_dominante": "Variable dominante (XAI)"}
