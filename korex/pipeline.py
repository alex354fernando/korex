"""Orquestación: KorexModel (C2→C5 entrenable) e InferencePipeline (artefactos,
predicción, explicación, reporte, FHIR y metadata.json). Única puerta de entrada
para api/ y dashboard/."""
from __future__ import annotations

import copy
import hashlib
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import __version__, load_config, resolve_path
from .ensemble import OrdinalStackingEnsemble, min_cost_decision
from .explain import Explainer
from .grouping import FeatureGrouper
from .kdigo import UMOL_PER_MGDL, ckd_epi_2021, clean_dataframe, egfr_by_policy, g_stage, read_table  # noqa: F401
from .labeling import EGFR_COL, label_dataset, scenario_features  # noqa: F401 (reexportadas)
from .metrics import calibration_curve_data, evaluate, pr_curves, roc_curves
from .preprocessing import Preprocessor
from .resampling import OASMOTENC, ordinal_cost_matrix

RECOMMENDATIONS = {
    0: ("Atención primaria", "Función renal normal o alta. Si hay albuminuria, hematuria u otro marcador de daño, "
                             "controlar en atención primaria al menos 1 vez al año y tratar factores de riesgo."),
    1: ("Atención primaria", "Filtrado levemente disminuido. Controlar creatinina y albuminuria 1 vez al año y optimizar "
                             "presión arterial, glucemia y lípidos."),
    2: ("Vigilancia", "Filtrado moderadamente disminuido (G3). Controlar 1–2 veces al año, ajustar dosis de fármacos "
                      "y valorar interconsulta con nefrología si progresa o hay albuminuria importante."),
    3: ("Nefrología", "Filtrado severamente disminuido (G4). Derivar a nefrología; controles cada 3–6 meses y "
                      "preparar la planificación de terapia de reemplazo renal."),
    4: ("Nefrología urgente", "Falla renal (G5). Derivación inmediata a nefrología para evaluar diálisis o trasplante."),
}
SCREENING_SCENARIO = "B"
SCREENING_RECS = {
    "alto": ("Prioridad alta", "Probable filtrado severamente disminuido (TFGe < 30). Solicite creatinina sérica de forma "
                               "prioritaria y valore derivación a nefrología."),
    "moderado": ("Solicitar creatinina", "Probable filtrado disminuido (TFGe < 60). Solicite creatinina sérica para "
                                         "estadificar según KDIGO."),
    "bajo": ("Riesgo bajo", "Baja probabilidad de TFGe < 60. Confirme con creatinina sérica en el control habitual."),
}
DISCLAIMER = "Herramienta de apoyo: no sustituye el juicio clínico. Confirme siempre con el profesional tratante."


def artifacts_dir(cfg: dict) -> Path:
    """Carpeta de artefactos (env KOREX_ARTIFACTS o paths.artifacts de config.yaml)."""
    return Path(os.environ.get("KOREX_ARTIFACTS") or resolve_path(cfg["paths"]["artifacts"]))


def default_params(cfg: dict, quick: bool = False) -> dict:
    """Hiperparámetros por defecto (config.yaml) + C4 + meta-aprendiz."""
    p = copy.deepcopy(cfg["tuning"]["defaults"])
    p["resampling"] = {k: cfg["resampling"][k] for k in ("r", "m", "k_s", "beta")}
    p["meta"] = {"meta_alpha": cfg["ensemble"]["meta_alpha"]}
    return apply_caps(p, cfg) if quick else p


def apply_caps(params: dict, cfg: dict) -> dict:
    caps = cfg["tuning"]["quick_caps"]
    for m in params.values():
        for k, v in caps.items():
            if k in m:
                m[k] = min(m[k], v)
    return params


def prepare_fold(cfg: dict, X_raw: pd.DataFrame, seed: int):
    """C2 + C3 ajustados solo con el pliegue de entrenamiento (cacheable en C6)."""
    pre = Preprocessor(cfg, seed)
    Xt = pre.fit_transform(X_raw)
    grouper = FeatureGrouper(cfg["grouping"]["distance_cut"], cfg["grouping"].get("always_keep", ())).fit(Xt)
    return pre, grouper, Xt


class KorexModel:
    """Cadena entrenable C2 → C3 → C4 → C5 para un escenario."""

    def __init__(self, cfg: dict, scenario: str, params: dict | None = None, seed: int | None = None,
                 stack_folds: int | None = None, fit_all: bool = False):
        self.cfg, self.scenario, self.fit_all = cfg, scenario, fit_all
        self.seed = cfg["project"]["seed"] if seed is None else seed
        self.params = params or default_params(cfg)
        self.stack_folds = stack_folds or cfg["ensemble"]["stack_folds"]
        c = cfg["cost"]
        self.C = ordinal_cost_matrix(len(cfg["kdigo"]["classes"]), c["p"], c["delta"])

    def fit(self, X_raw: pd.DataFrame, y: np.ndarray, prepared=None) -> "KorexModel":
        self.features_ = scenario_features(X_raw.columns, self.cfg, self.scenario)
        self.pre_, self.grouper_, Xt = prepared or prepare_fold(self.cfg, X_raw[self.features_], self.seed)
        rp = self.params["resampling"]
        self.res_ = OASMOTENC(self.pre_.out_types_, self.pre_.onehot_groups_, rp["r"], rp["m"], rp["k_s"],
                              rp["beta"], self.seed, self.cfg["resampling"]["max_attempts_factor"],
                              K=self.C.shape[0], mode=self.params.get("resampling_mode", "oa"))
        Xa, ya, wa, syn, par = self.res_.fit_resample(Xt, y)
        e = self.cfg["ensemble"]
        base = {k: v for k, v in self.params.items() if k not in ("resampling", "meta", "resampling_mode")}
        self.ens_ = OrdinalStackingEnsemble(
            base, self.grouper_.representatives_, self.C, self.stack_folds, e["q_threshold"],
            self.params["meta"]["meta_alpha"], e["calibration"], self.seed, self.cfg["project"]["n_jobs"],
            e["svm_max_samples"], self.cfg["tuning"]["objective_lambda"], rp["beta"],
            core_cols=self.cfg["grouping"].get("always_keep", []),
            greedy=e.get("greedy_selection", False), fit_all=self.fit_all).fit(Xa, ya, wa, syn, par)
        return self

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        return self.pre_.transform(X_raw.reindex(columns=self.features_))

    def predict_proba(self, X_raw: pd.DataFrame) -> np.ndarray:
        return self.ens_.predict_proba(self.transform(X_raw))

    def predict(self, X_raw: pd.DataFrame, C: np.ndarray | None = None) -> np.ndarray:
        return min_cost_decision(self.predict_proba(X_raw), self.C if C is None else C)

    def summary(self) -> dict:
        return {"features": self.features_, "preprocessing": self.pre_.type_record(),
                "grouping": self.grouper_.summary(), "resampling": self.res_.summary(),
                "ensemble": self.ens_.summary()}


# ===================================================================== inferencia
def payload_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def validate_ranges(record: dict, cfg: dict) -> list[str]:
    """Advertencias cuando un valor sale del rango clínico plausible."""
    out = []
    for k, v in record.items():
        meta = cfg["variables"].get(k)
        if meta is None or "min" not in meta or v is None or isinstance(v, str):
            continue
        if not (meta["min"] <= float(v) <= meta["max"]):
            out.append(f"{meta['label']}: {v} {meta['unit']} fuera del rango esperado ({meta['min']}–{meta['max']}).")
    return out


class InferencePipeline:
    """Carga los artefactos entrenados y expone predict / explain / report."""

    def __init__(self, models: dict, backgrounds: dict, metadata: dict, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.models, self.backgrounds, self.metadata = models, backgrounds, metadata
        self._explainers = {}

    # ------------------------------------------------------------ persistencia
    def save(self, directory: str | Path | None = None) -> Path:
        d = Path(directory or artifacts_dir(self.cfg))
        d.mkdir(parents=True, exist_ok=True)
        for s, m in self.models.items():
            joblib.dump({"model": m, "background": self.backgrounds[s]}, d / f"korex_{s}.joblib", compress=3)
        (d / "metadata.json").write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False, default=_json_default),
                                         encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | Path | None = None, cfg: dict | None = None) -> "InferencePipeline":
        cfg = cfg or load_config()
        d = Path(directory or artifacts_dir(cfg))
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No hay artefactos en {d}. Ejecute: python scripts/train.py")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        models, bgs = {}, {}
        for s in ("A", "B"):
            f = d / f"korex_{s}.joblib"
            if f.exists():
                try:
                    obj = joblib.load(f)
                except Exception as e:                  # p. ej., modelo entrenado en otro sistema operativo
                    raise FileNotFoundError(f"No se pudo abrir {f.name} ({type(e).__name__}). Los modelos deben "
                                            "entrenarse en este equipo: python scripts/train.py --profile quick") from e
                models[s], bgs[s] = obj["model"], obj["background"]
        return cls(models, bgs, metadata, cfg)

    @property
    def model_version(self) -> str:
        return f"{self.metadata.get('version', __version__)}+{self.metadata.get('trained_at', '')[:10]}"

    # ------------------------------------------------------------ entrada
    def prepare_record(self, record: dict) -> pd.DataFrame:
        r = {k: (v.strip().lower() if isinstance(v, str) else v) for k, v in record.items()
             if k not in ("patient_id", "creatinine_unit")}
        if str(record.get("creatinine_unit", "mg/dL")).lower().startswith("umol") and r.get("creatinine") is not None:
            r["creatinine"] = float(r["creatinine"]) / UMOL_PER_MGDL
        df = pd.DataFrame([r])
        return self._add_egfr(df)

    def _add_egfr(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in ("creatinine", "age", "sex"):
            if c not in df:
                df[c] = np.nan
        df["creatinine"] = pd.to_numeric(df["creatinine"], errors="coerce")
        df["sex"] = pd.to_numeric(df["sex"], errors="coerce")
        df[EGFR_COL] = egfr_by_policy(df["creatinine"], df["age"], df["sex"], self.cfg["kdigo"].get("sex_policy", "worst"))
        return df

    def features(self, scenario: str) -> list[str]:
        return self.models[scenario].features_

    # ------------------------------------------------------------ predicción
    def predict(self, record: dict, scenario: str | None = None, delta: float | None = None) -> dict:
        scenario = scenario or self.cfg["scenarios"]["default"]
        if scenario not in self.models:
            raise ValueError(f"Escenario {scenario} no entrenado.")
        df = self.prepare_record(record)
        if df["age"].isna().any():
            raise ValueError("La edad es obligatoria.")
        model = self.models[scenario]
        P = model.predict_proba(df)[0]
        C = model.C if delta is None else ordinal_cost_matrix(len(P), self.cfg["cost"]["p"], delta)
        cls = int(min_cost_decision(P[None, :], C)[0])
        k = self.cfg["kdigo"]
        title, text = RECOMMENDATIONS[cls]
        if pd.isna(df["creatinine"].iloc[0]) and scenario != SCREENING_SCENARIO:
            text += " Sin creatinina registrada: confirme con creatinina sérica para estadificar según KDIGO."
        screening = None
        if scenario == SCREENING_SCENARIO:              # modo tamizaje: el resultado útil es TFGe < 60 / < 30
            p60, p30 = float(P[2:].sum()), float(P[3:].sum())
            level = "alto" if p30 >= 0.5 else ("moderado" if p60 >= 0.5 else "bajo")
            title, text = SCREENING_RECS[level]
            lo, hi = max(0, cls - 1), min(len(P) - 1, cls + 1)
            ev = self.metadata.get("evaluation", {}).get(scenario, {})
            screening = {"p_lt60": round(p60, 4), "p_lt30": round(p30, 4), "level": level,
                         "range": [k["stage_labels"][lo], k["stage_labels"][hi]],
                         "validated": {m: round(ev[m]["mean"], 3) for m in
                                       ("accuracy", "accuracy_within1", "auc_lt60", "sens_lt60", "spec_lt60",
                                        "auc_lt30", "sens_lt30", "spec_lt30") if m in ev}}
        egfr = df[EGFR_COL].iloc[0]
        rule = None
        if not np.isnan(egfr):
            st_ = int(g_stage([egfr], k["g_cuts"])[0])
            rule = {"egfr": round(float(egfr), 1), "G": k["stage_labels"][st_], "risk_class": st_,
                    "sex_assumed": "registrado" if not pd.isna(df["sex"].iloc[0]) else k.get("sex_policy", "worst")}
        used = {f: (None if pd.isna(df[f].iloc[0]) else _py(df[f].iloc[0])) if f in df else None for f in model.features_}
        return {
            "risk_class": cls, "risk_name": k["class_names"][cls], "color": k["class_colors"][cls],
            "probabilities": [round(float(p), 4) for p in P], "probability": round(float(P[cls]), 4),
            "expected_risk": round(float(P @ np.arange(len(P))), 3),
            "recommendation": {"level": title, "text": text, "disclaimer": DISCLAIMER},
            "scenario": scenario, "scenario_name": self.cfg["scenarios"][scenario]["name"],
            "features_used": used, "imputed": [f for f, v in used.items() if v is None],
            "kdigo_rule": rule, "screening": screening, "warnings": validate_ranges(record, self.cfg),
            "model_version": self.model_version, "payload_hash": payload_hash(record),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def predict_proba_records(self, records: list[dict], scenario: str | None = None) -> np.ndarray:
        """Probabilidades para muchos registros a la vez (simulador, curvas)."""
        scenario = scenario or self.cfg["scenarios"]["default"]
        dfs = [self.prepare_record(r) for r in records]
        return self.models[scenario].predict_proba(pd.concat(dfs, ignore_index=True))

    @staticmethod
    def demo_mode() -> bool:
        """Demostración pública (KOREX_DEMO=1): no se guarda ni se muestra ninguna evaluación."""
        return os.environ.get("KOREX_DEMO", "").strip() == "1"

    def read_audit(self) -> list[dict]:
        if self.demo_mode():
            return []
        f = self._audit_path()
        if not f.exists():
            return []
        return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]

    def verify_audit(self) -> dict:
        """Comprueba la cadena de hashes del registro (detecta ediciones o borrados)."""
        prev, rows = "0" * 64, self.read_audit()
        for i, e in enumerate(rows):
            body = {k: v for k, v in e.items() if k != "record_hash"}
            if e.get("prev_hash") != prev or payload_hash(body) != e.get("record_hash"):
                return {"ok": False, "broken_at": i, "checked": i}
            prev = e["record_hash"]
        return {"ok": True, "checked": len(rows)}

    def _audit_path(self) -> Path:
        return (artifacts_dir(self.cfg) / "auditoria.jsonl") if os.environ.get("KOREX_ARTIFACTS") \
            else resolve_path(self.cfg["paths"]["audit_log"])

    def explainer(self, scenario: str) -> Explainer:
        if scenario not in self._explainers:
            self._explainers[scenario] = Explainer(self.models[scenario], self.backgrounds[scenario], self.cfg,
                                                   self.cfg["project"]["seed"])
        return self._explainers[scenario]

    def explain(self, record: dict, scenario: str | None = None, target="risk") -> dict:
        scenario = scenario or self.cfg["scenarios"]["default"]
        df = self.prepare_record(record)
        xt = self.models[scenario].transform(df)
        raw = {c: _py(df[c].iloc[0]) for c in df.columns}
        out = self.explainer(scenario).explain_instance(xt, raw, target)
        out["scenario"] = scenario
        return out

    def evaluation(self, scenario: str | None = None) -> dict:
        ev = self.metadata.get("evaluation", {})
        return ev if scenario is None else ev.get(scenario, {})

    def global_shap(self) -> dict | None:
        """Matrices SHAP globales guardadas al entrenar (barras, beeswarm, dependencia)."""
        f = artifacts_dir(self.cfg) / "global_shap.joblib"
        return joblib.load(f) if f.exists() else None

    def evaluation_records(self) -> dict | None:
        """Predicciones fuera de muestra de la validación anidada (si se guardaron)."""
        f = artifacts_dir(self.cfg) / "evaluation.joblib"
        return joblib.load(f) if f.exists() else None

    def pooled_oof(self, scenario: str = "A") -> dict | None:
        """Métricas y curvas ROC sobre todas las predicciones externas (fuera de muestra) agrupadas."""
        rec = self.evaluation_records()
        if not rec or scenario not in rec["records"]:
            return None
        r = rec["records"][scenario]["records"]
        idx = np.concatenate([np.asarray(x["test_idx"]) for x in r])
        P = np.vstack([np.asarray(x["proba"]) for x in r])
        pred = np.concatenate([np.asarray(x["pred"]) for x in r])
        y = rec["y"][idx]
        C = self.models[scenario].C if scenario in self.models else None
        return {"metrics": evaluate(y, P, pred, C, self.cfg["tuning"]["objective_lambda"]), "roc": roc_curves(y, P),
                "pr": pr_curves(y, P), "y": y, "P": P, "pred": pred}

    def screening_reliability(self, p_lt60: float, p_lt30: float, k: int = 30) -> dict | None:
        """¿Qué pasó en validación con los pacientes que recibieron probabilidades parecidas a estas?
        Toma los k pacientes externos más cercanos en (P<60, P<30) y cuenta su estadio real."""
        rec = self.evaluation_records()
        if not rec or SCREENING_SCENARIO not in rec["records"]:
            return None
        r = rec["records"][SCREENING_SCENARIO]["records"]
        idx = np.concatenate([np.asarray(x["test_idx"]) for x in r])
        P = np.vstack([np.asarray(x["proba"]) for x in r])
        y = np.asarray(rec["y"])[idx]
        d = np.hypot(P[:, 2:].sum(1) - p_lt60, P[:, 3:].sum(1) - p_lt30)
        near = np.argsort(d)[:min(k, len(d))]
        yn = y[near]
        return {"n": int(len(near)), "frac_lt60": float((yn >= 2).mean()), "frac_lt30": float((yn >= 3).mean()),
                "frac_healthy": float((yn <= 1).mean()),
                "stage_counts": np.bincount(yn, minlength=P.shape[1]).tolist(), "max_distance": float(d[near].max())}

    def calibration(self, scenario: str) -> dict | None:
        rec = self.evaluation_records()
        if not rec or scenario not in rec["records"]:
            return None
        r = rec["records"][scenario]["records"]
        idx = np.concatenate([np.asarray(x["test_idx"]) for x in r])
        P = np.vstack([np.asarray(x["proba"]) for x in r])
        return calibration_curve_data(rec["y"][idx], P)

    # ------------------------------------------------------------ lote / evaluación externa
    def predict_batch(self, df_raw: pd.DataFrame, scenario: str | None = None) -> pd.DataFrame:
        """Limpia (C1) y predice un lote; añade el estadio por TFGe si hay creatinina."""
        scenario = scenario or self.cfg["scenarios"]["default"]
        df = self._add_egfr(clean_dataframe(df_raw, self.cfg))
        model = self.models[scenario]
        P = model.predict_proba(df)
        pred = min_cost_decision(P, model.C)
        out = pd.DataFrame({"estadio": [self.cfg["kdigo"]["stage_labels"][k] for k in pred],
                            "descripcion": [self.cfg["kdigo"]["class_names"][k] for k in pred]})
        for k in range(P.shape[1]):
            out[f"p{k}"] = P[:, k].round(4)
        rule = g_stage(df[EGFR_COL], self.cfg["kdigo"]["g_cuts"])
        out["tfge"] = df[EGFR_COL].round(1)
        out["estadio_por_tfge"] = [self.cfg["kdigo"]["stage_labels"][int(v)] if not np.isnan(v) else "" for v in rule]
        return out

    def evaluate_dataset(self, df_raw: pd.DataFrame, scenario: str) -> dict:
        """Evaluación externa: etiqueta con C1 y mide el modelo desplegado (C7)."""
        data = label_dataset(clean_dataframe(df_raw, self.cfg), self.cfg)
        model = self.models[scenario]
        P = model.predict_proba(data.X)
        return {"n": data.info["n"], "class_counts": data.info["class_counts"],
                "metrics": evaluate(data.y, P, min_cost_decision(P, model.C), model.C, self.cfg["tuning"]["objective_lambda"])}

    # ------------------------------------------------------------ reporte / FHIR
    def patient_table(self, record: dict) -> list[dict]:
        """Datos ingresados con etiquetas en español (para el informe)."""
        vlab, rows = self.cfg.get("value_labels", {}), []
        sex = record.get("sex")
        rows.append({"variable": "Sexo", "valor": {1: "Mujer", 0: "Hombre"}.get(sex, "No registrado"), "unidad": ""})
        for k, v in record.items():
            if k in ("sex", "creatinine_unit", "patient_id") or v is None:
                continue
            meta = self.cfg["variables"].get(k, {"label": k, "unit": ""})
            unit = record.get("creatinine_unit", "mg/dL") if k == "creatinine" else meta.get("unit", "")
            if isinstance(v, str):
                val = vlab.get(v, v)
            elif 1.0 < float(v) < 1.1:
                val = f"{float(v):.3f}"                      # densidad urinaria
            else:
                val = int(v) if float(v).is_integer() else round(float(v), 2)
            rows.append({"variable": meta["label"], "valor": val, "unidad": unit})
        return rows

    def build_report(self, result: dict, explanation: dict | None = None, record: dict | None = None) -> dict:
        rep = {"titulo": "Informe KORE-X · Estadio de enfermedad renal (KDIGO G1–G5)", "fecha": result["timestamp"],
               "modelo": result["scenario_name"], "estadio": self.cfg["kdigo"]["stage_labels"][result["risk_class"]],
               "categoria": result["risk_class"],
               "nombre_riesgo": result["risk_name"], "probabilidad": result["probability"],
               "probabilidades": dict(zip(self.cfg["kdigo"]["stage_labels"], result["probabilities"])),
               "recomendacion": result["recommendation"], "regla_kdigo": result["kdigo_rule"],
               "advertencias": result["warnings"], "version_modelo": result["model_version"],
               "hash": result["payload_hash"]}
        if result.get("screening"):
            sc = result["screening"]
            rep["titulo"] = "Informe KORE-X · Tamizaje de función renal sin creatinina"
            rep["nombre_riesgo"] = f"{result['recommendation']['level']} · estadio probable {sc['range'][0]}–{sc['range'][1]}"
            rep["tamizaje"] = {"Probabilidad de TFGe < 60": f"{100 * sc['p_lt60']:.0f} %",
                               "Probabilidad de TFGe < 30": f"{100 * sc['p_lt30']:.0f} %",
                               "Estadio probable (±1)": f"{sc['range'][0]} a {sc['range'][1]}"}
        if record:
            rep["datos_paciente"] = self.patient_table(record)
        if explanation:
            rep["factores_clave"] = [{"factor": f["label"], "unidad": f["unit"] if f["value"] is not None else "",
                                      "valor": "no informado (estimado)" if f["value"] is None else f.get("value_label", f["value"]),
                                      "efecto": "aumenta" if f["shap"] > 0 else "protege"}
                                     for f in explanation["factors"][:self.cfg["explain"]["top_k"]]]
            rep["lectura"] = explanation["reading"]
        return rep

    def to_pdf(self, report: dict, color: str = "#333333") -> bytes:
        """Informe PDF A4 con encabezado, escala G1–G5, probabilidades, datos del paciente y firma."""
        from fpdf import FPDF
        t = lambda x: str(x).replace("–", "-").replace("≥", ">=").replace("➜", "->").replace("…", "...") \
                            .replace("²", "2").replace("\u03bc", "\u00b5").encode("latin-1", "replace").decode("latin-1")
        rgb = lambda h: tuple(int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        k = self.cfg["kdigo"]
        pdf = FPDF(format="A4")
        pdf.set_auto_page_break(True, 18)
        pdf.add_page()
        # encabezado
        pdf.set_fill_color(15, 118, 110); pdf.rect(0, 0, 210, 26, "F")
        pdf.set_text_color(255, 255, 255); pdf.set_xy(12, 7); pdf.set_font("Helvetica", "B", 18); pdf.cell(60, 8, "KORE-X")
        pdf.set_font("Helvetica", "", 10); pdf.set_xy(12, 15)
        pdf.cell(0, 6, t("Tamizaje de función renal sin creatinina" if report.get("tamizaje")
                         else "Informe de estadio de enfermedad renal crónica (KDIGO G1-G5)"))
        pdf.set_xy(140, 9); pdf.set_font("Helvetica", "", 9)
        pdf.cell(58, 5, t(report["fecha"][:16].replace("T", " ") + " UTC"), align="R")
        pdf.set_text_color(0, 0, 0); pdf.set_xy(12, 32)
        # resultado
        pdf.set_fill_color(*rgb(color)); pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", "B", 15)
        pdf.cell(186, 12, t(f"  {report['nombre_riesgo']}"), fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(186, 7, t(f"  Probabilidad estimada {report['probabilidad'] * 100:.0f} %"), fill=True,
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0); pdf.ln(3)
        # escala G1-G5 y barras de probabilidad
        x0, y0, w = 12, pdf.get_y(), 37.2
        probs = list(report.get("probabilidades", {}).values())
        for i, lab in enumerate(k["stage_labels"]):
            pdf.set_fill_color(*rgb(k["class_colors"][i])); pdf.rect(x0 + i * w, y0, w - 1, 7, "F")
            pdf.set_xy(x0 + i * w, y0 + 1); pdf.set_font("Helvetica", "B", 9); pdf.set_text_color(255, 255, 255)
            pdf.cell(w - 1, 5, lab, align="C")
            if i == report["categoria"]:
                pdf.set_draw_color(17, 24, 39); pdf.set_line_width(0.8); pdf.rect(x0 + i * w, y0, w - 1, 7)
            if probs:
                h = 22 * probs[i]
                pdf.set_fill_color(*rgb(k["class_colors"][i])); pdf.rect(x0 + i * w + 10, y0 + 32 - h, w - 21, h, "F")
                pdf.set_xy(x0 + i * w, y0 + 33); pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", "", 8)
                pdf.cell(w - 1, 4, f"{probs[i] * 100:.0f} %", align="C")
        pdf.set_text_color(0, 0, 0); pdf.set_xy(12, y0 + 40)
        # recomendación
        pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 6, t(f"Recomendación: {report['recomendacion']['level']}"),
                                                    new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10); pdf.multi_cell(0, 5, t(report["recomendacion"]["text"]), new_x="LMARGIN", new_y="NEXT")
        if report.get("tamizaje"):
            pdf.ln(1); pdf.set_font("Helvetica", "B", 10); pdf.set_fill_color(255, 243, 224)
            pdf.cell(0, 6, t("  Modo tamizaje (sin creatinina): resultado orientativo"), fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9.5)
            for kk, vv in report["tamizaje"].items():
                pdf.cell(70, 5, t(f"  {kk}")); pdf.set_font("Helvetica", "B", 9.5); pdf.cell(0, 5, t(vv), new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 9.5)
            pdf.set_font("Helvetica", "I", 8.5)
            pdf.multi_cell(0, 4.5, t("El estadio KDIGO exige creatinina sérica: este resultado solo indica a quién priorizar."),
                           new_x="LMARGIN", new_y="NEXT")
        if report.get("regla_kdigo"):
            rk = report["regla_kdigo"]
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 5, t(f"Referencia CKD-EPI 2021: TFGe {rk['egfr']} mL/min/1,73 m2 -> {rk['G']}"),
                           new_x="LMARGIN", new_y="NEXT")
        # datos del paciente
        if report.get("datos_paciente"):
            pdf.ln(2); pdf.set_font("Helvetica", "B", 11); pdf.set_fill_color(234, 241, 245)
            pdf.cell(0, 7, "  Datos del paciente", fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8.5)
            rows = report["datos_paciente"]
            half = (len(rows) + 1) // 2
            for i in range(half):
                for row in (rows[i], rows[i + half] if i + half < len(rows) else None):
                    if row is None:
                        pdf.cell(93, 4.6, "")
                        continue
                    pdf.set_font("Helvetica", "", 8.5); pdf.cell(55, 4.6, t(row["variable"])[:38])
                    pdf.set_font("Helvetica", "B", 8.5); pdf.cell(38, 4.6, t(f"{row['valor']} {row['unidad']}".strip())[:26])
                pdf.ln(4.6)
        # factores
        if report.get("factores_clave"):
            pdf.ln(2); pdf.set_font("Helvetica", "B", 11); pdf.set_fill_color(234, 241, 245)
            pdf.cell(0, 7, "  Factores que más influyeron", fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            for f in report["factores_clave"]:
                col = (198, 40, 40) if f["efecto"] == "aumenta" else (46, 125, 50)
                pdf.set_text_color(*col); pdf.cell(6, 5, "+" if f["efecto"] == "aumenta" else "-")
                pdf.set_text_color(0, 0, 0)
                pdf.cell(0, 5, t(f"{f['factor']} ({f['valor']} {f['unidad']})".replace(" )", ")")), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1); pdf.multi_cell(0, 5, t(report.get("lectura", "")), new_x="LMARGIN", new_y="NEXT")
        if report.get("advertencias"):
            pdf.ln(1); pdf.set_font("Helvetica", "B", 9); pdf.cell(0, 5, "Advertencias", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8.5)
            for wv in report["advertencias"]:
                pdf.multi_cell(0, 4.5, t(f"- {wv}"), new_x="LMARGIN", new_y="NEXT")
        # firma y pie
        if pdf.get_y() > 245:
            pdf.add_page()
        pdf.set_y(252); pdf.set_draw_color(120, 120, 120); pdf.set_line_width(0.3)
        pdf.line(120, 262, 195, 262); pdf.set_xy(120, 263); pdf.set_font("Helvetica", "", 8)
        pdf.cell(75, 4, t("Firma y sello del profesional"), align="C")
        pdf.set_xy(12, 270); pdf.set_font("Helvetica", "I", 7.5); pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(186, 3.6, t(f"{report['recomendacion']['disclaimer']} Modelo {report['version_modelo']} · "
                                   f"verificación {report['hash'][:16]}"), new_x="LMARGIN", new_y="NEXT")
        return bytes(pdf.output())

    # ------------------------------------------------------------ auditoría ligera
    def log_audit(self, record: dict, result: dict, role: str = "usuario") -> str:
        """Agrega la predicción a un registro JSONL con hash encadenado (sin identificadores directos).
        En modo demostración no escribe nada y devuelve una cadena vacía."""
        if self.demo_mode():
            return ""
        f = self._audit_path()
        f.parent.mkdir(parents=True, exist_ok=True)
        prev = "0" * 64
        if f.exists():
            lines = f.read_text(encoding="utf-8").strip().splitlines()
            prev = json.loads(lines[-1])["record_hash"] if lines else prev
        entry = {"timestamp": result["timestamp"], "payload_hash": result["payload_hash"], "role": role,
                 "model_version": result["model_version"], "inputs": {k: v for k, v in record.items() if k != "patient_id"},
                 "stage": self.cfg["kdigo"]["stage_labels"][result["risk_class"]], "probabilities": result["probabilities"],
                 "prev_hash": prev}
        entry["record_hash"] = payload_hash(entry)
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry["record_hash"]

    @staticmethod
    def to_fhir(result: dict, report: dict | None = None) -> dict:
        """Mapeo básico a FHIR R4 DiagnosticReport."""
        return {
            "resourceType": "DiagnosticReport", "id": result["payload_hash"][:16], "status": "preliminary",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "LAB"}]}],
            "code": {"text": "KORE-X: estadio de enfermedad renal crónica por filtrado (KDIGO G1–G5)"},
            "effectiveDateTime": result["timestamp"], "issued": result["timestamp"],
            "conclusion": f"{result['risk_name']} (p = {result['probability']:.2f}). "
                          f"{result['recommendation']['text']}",
            "conclusionCode": [{"text": result["risk_name"], "coding": [{"system": "urn:korex:kdigo-g-stage",
                                                                          "code": f"G{result['risk_class'] + 1}"}]}],
            "extension": [{"url": "urn:korex:scenario", "valueString": result["scenario"]},
                          {"url": "urn:korex:model-version", "valueString": result["model_version"]},
                          {"url": "urn:korex:probabilities", "valueString": json.dumps(result["probabilities"])}],
        }

    @staticmethod
    def render_report_html(report: dict, color: str = "#333") -> str:
        e = lambda v: html.escape(str(v))
        rows = "".join(f"<li><b>{e(f['factor'])}</b> ({e(f['valor'])} {e(f['unidad'])}): {e(f['efecto'])} el riesgo</li>"
                       for f in report.get("factores_clave", []))
        warn = "".join(f"<li>{e(w)}</li>" for w in report["advertencias"])
        pat = "".join(f"<tr><td style='padding:2px 12px 2px 0'>{e(r['variable'])}</td><td><b>{e(r['valor'])}</b> {e(r['unidad'])}</td></tr>"
                      for r in report.get("datos_paciente", []))
        return f"""<!DOCTYPE html><html lang="es"><meta charset="utf-8"><title>{e(report['titulo'])}</title>
<body style="font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;color:#222">
<h1>{e(report['titulo'])}</h1><p>{e(report['fecha'])} · {e(report['modelo'])}</p>
<div style="border-left:12px solid {e(color)};padding:1rem;background:#f7f7f7">
<h2 style="margin:0">{e(report['nombre_riesgo'])}</h2>
<p>Probabilidad estimada: {float(report['probabilidad']) * 100:.0f} %</p>
<p><b>{e(report['recomendacion']['level'])}:</b> {e(report['recomendacion']['text'])}</p></div>
<h3>Datos del paciente</h3><table style="border-collapse:collapse;font-size:.9rem">{pat}</table>
<h3>Factores clave</h3><ul>{rows or '<li>No solicitados.</li>'}</ul><p>{e(report.get('lectura', ''))}</p>
{f'<h3>Advertencias</h3><ul>{warn}</ul>' if warn else ''}
<p style="font-size:.85rem;color:#555">{e(report['recomendacion']['disclaimer'])}<br>
Modelo {e(report['version_modelo'])} · hash {e(report['hash'][:16])}…</p></body></html>"""


_CACHE: dict = {}


def get_pipeline(directory: str | Path | None = None) -> InferencePipeline:
    """Pipeline en caché; se recarga solo si metadata.json cambió (tras reentrenar)."""
    cfg = load_config()
    d = Path(directory or artifacts_dir(cfg))
    mtime = (d / "metadata.json").stat().st_mtime if (d / "metadata.json").exists() else None
    if _CACHE.get("key") != (str(d), mtime):
        _CACHE["pipe"], _CACHE["key"] = InferencePipeline.load(d, cfg), (str(d), mtime)
    return _CACHE["pipe"]


def _py(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
