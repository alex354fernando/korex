# KORE-X · Estadio KDIGO G1–G5 con stacking ordinal y explicabilidad SHAP+LIME

Estima el estadio de filtrado KDIGO (G1 ≥90 · G2 60–89 · G3 30–59 · G4 15–29 · G5 <15 mL/min/1,73 m²) con el
dataset UCI *Chronic Kidney Disease* (400 pacientes; 356 adultos etiquetables). **Un único modelo** usa todas las
variables clínicas (incluida la creatinina; la TFGe derivada no entra como predictor). El dataset no registra el
sexo: la etiqueta usa la TFGe CKD-EPI 2021 más conservadora (`kdigo.sex_policy`). Métricas principales (macro):
Precision, Recall, F1-Score y AUC-ROC. **Herramienta de apoyo: no sustituye el juicio clínico.**

```
ARFF UCI ─► Excel ─► C1 etiquetado G1–G5 ─► [validación anidada · C6 Optuna en el nivel interno]
     C2 MissForest + log + escalado (solo entrenamiento) ─► C3 Spearman/UPGMA ─► C4 OA-SMOTE-NC
     ─► C5 XGB·LGBM·CatBoost·ET·LR·SVM → odds proporcionales → mínimo costo ─► C7 métricas · C8 SHAP+LIME
     ─► korex/pipeline.py ─► dashboard/ (Streamlit multipágina)
```

## Instalación (Windows, PowerShell, Python 3.12 o 3.13)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Uso

| Paso | Comando | Tiempo aprox. |
|---|---|---|
| Convertir ARFF → Excel | `python scripts/prepare_data.py` | segundos |
| Prueba rápida | `python scripts/train.py --profile quick` | ~3–5 min |
| Modelo de la tesis | `python scripts/train.py --profile standard` | ~6–10 min |
| Corrida completa para la tesis (una vez) | `python scripts/train.py --profile standard --tesis` | ×2–3 |
| Validación 5×5 del documento | `--profile full` | varias horas |
| Figuras y tabla de literatura | `python scripts/figures.py` | ~1 min |
| Dashboard | `streamlit run dashboard/app.py` | — |
| Pruebas | `python -m pytest -q` | ~1 min |

Los pliegues se entrenan en paralelo (un proceso por núcleo, `project.parallel_workers`).

## Artefactos (`artifacts/`)

`korex_A.joblib` (modelo con creatinina), `korex_B.joblib` (modo tamizaje sin creatinina), `global_shap.joblib`
(SHAP global para beeswarm y dependencia), `metadata.json` (versión, semilla, variables, métricas, ablación, IC_M, rangos para
aleatorizar), `evaluation.joblib` (predicciones fuera de muestra), `figures/` y `auditoria.jsonl` (registro de
predicciones con hash encadenado, sin identificadores directos).

## Dashboard (apartados)

Atención clínica: **Evaluar paciente** · **Simulador ¿qué pasaría si…?** · **Comparar pacientes** ·
Datos: **Historial** (auditoría con verificación de integridad) · **Carga masiva** ·
Modelo: **Desempeño del modelo** · **Acerca de KORE-X**. Tema y colores en `.streamlit/config.toml`.

## Demostración en línea

Despliegue en Streamlit Community Cloud desde este repositorio: archivo principal `dashboard/app.py`,
Python 3.13 y dependencias de `requirements.txt` (más `packages.txt` para la biblioteca OpenMP de LightGBM).
Los artefactos de `artifacts/` se entrenaron con `python scripts/train.py --profile standard --tesis` y se cargan
sin reentrenar en Windows y en Linux con las versiones fijadas. **La demostración usa datos públicos y no debe
recibir datos reales de pacientes.**

## Licencia y datos

Código bajo licencia MIT (`LICENSE`). El conjunto *Chronic Kidney Disease* (Rubini, Soundarapandian y Eswaran, 2015)
procede del UCI Machine Learning Repository y se redistribuye en `data/raw/` bajo licencia CC BY 4.0,
https://doi.org/10.24432/C5G020.

## Cómo citar

Los metadatos de cita están en `CITATION.cff`. Cada versión publicada se archiva en Zenodo con un DOI propio.
