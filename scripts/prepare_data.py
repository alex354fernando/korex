"""Convierte el dataset UCI Chronic Kidney Disease (ARFF) a Excel y genera un resumen.

Uso:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --arff ruta/al/archivo.arff --out data/ckd_uci.xlsx

El Excel conserva los datos originales (sin limpiar ni etiquetar); la limpieza y el
etiquetado G1–G5 se hacen en el pipeline (C1). Incluye una hoja 'diccionario'.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from korex.pipeline import clean_dataframe, label_dataset, load_config, resolve_path  # noqa: E402
from korex.kdigo import read_arff  # noqa: E402

DICTIONARY = {
    "age": "Edad (años)", "bp": "Presión arterial (mmHg)", "sg": "Densidad urinaria (1,005–1,025)",
    "al": "Albúmina en orina, tira reactiva (0–5)", "su": "Azúcar en orina, tira reactiva (0–5)",
    "rbc": "Glóbulos rojos en orina (normal/abnormal)", "pc": "Células de pus (normal/abnormal)",
    "pcc": "Grumos de células de pus (present/notpresent)", "ba": "Bacterias (present/notpresent)",
    "bgr": "Glucosa en sangre aleatoria (mg/dL)", "bu": "Urea en sangre (mg/dL)", "sc": "Creatinina sérica (mg/dL)",
    "sod": "Sodio (mEq/L)", "pot": "Potasio (mEq/L)", "hemo": "Hemoglobina (g/dL)", "pcv": "Hematocrito (%)",
    "wc": "Leucocitos (células/µL)", "rc": "Glóbulos rojos (millones/µL)", "htn": "Hipertensión (yes/no)",
    "dm": "Diabetes mellitus (yes/no)", "cad": "Enfermedad coronaria (yes/no)", "appet": "Apetito (good/poor)",
    "pe": "Edema pedio (yes/no)", "ane": "Anemia (yes/no)",
    "class": "Diagnóstico original ckd/notckd (NO se usa como predictor: fuga)",
}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arff", default=str(resolve_path(cfg["paths"]["raw_data"])))
    ap.add_argument("--out", default=str(resolve_path(cfg["paths"]["data"])))
    a = ap.parse_args()
    raw = read_arff(a.arff)
    print(f"Leídas {len(raw)} filas y {raw.shape[1]} columnas · filas descartadas por formato: {raw.attrs['rejected_rows']}")
    raw = raw.replace({"?": np.nan, "": np.nan})
    for c in raw.columns:                                  # números como números en Excel
        num = pd.to_numeric(raw[c], errors="coerce")
        if num.notna().sum() >= 0.9 * raw[c].notna().sum():
            raw[c] = num
    data = label_dataset(clean_dataframe(raw, cfg), cfg)
    names = cfg["kdigo"]["stage_labels"]
    summary = pd.DataFrame({
        "estadio": names,
        "n (peor caso)": [data.info["class_counts"][k] for k in range(len(names))],
        "n (si todos hombres)": [data.info["class_counts_if_male"][k] for k in range(len(names))],
        "n (si todas mujeres)": [data.info["class_counts_if_female"][k] for k in range(len(names))],
    })
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as xw:
        raw.to_excel(xw, sheet_name="datos", index=False)
        pd.DataFrame({"variable": list(DICTIONARY), "descripcion": list(DICTIONARY.values()),
                      "faltantes_%": [round(100 * raw[c].isna().mean(), 1) if c in raw else None for c in DICTIONARY]}
                     ).to_excel(xw, sheet_name="diccionario", index=False)
        summary.to_excel(xw, sheet_name="estadios_G", index=False)
    print(f"Excel guardado en {out}")
    print(f"Adultos etiquetados: {data.info['n']} (excluidos sin edad/creatinina o <18: {len(raw) - data.info['n']})")
    print(summary.to_string(index=False))
    print(f"Discordancia de estadio hombre vs mujer: {100 * data.info['sex_discordance_rate']:.1f} %")


if __name__ == "__main__":
    main()
