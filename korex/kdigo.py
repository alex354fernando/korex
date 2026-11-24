"""C1 (parte 1) · Constantes KDIGO, ecuación CKD-EPI 2021, estadios G1–G5 y carga
limpia del dataset (UCI Chronic Kidney Disease en ARFF/Excel/CSV, u otro con creatinina)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

UMOL_PER_MGDL = 88.4
DEFAULT_CUTS = [90, 60, 30, 15]                   # G1 ≥90 · G2 60–89 · G3 30–59 · G4 15–29 · G5 <15


def ckd_epi_2021(scr_mg_dl, age, female) -> np.ndarray:
    """TFGe CKD-EPI 2021 sin raza (mL/min/1,73 m²). female: 1 mujer, 0 hombre."""
    scr = np.asarray(scr_mg_dl, dtype=float)
    age = np.asarray(age, dtype=float)
    fem = np.asarray(female, dtype=float) == 1
    kappa = np.where(fem, 0.7, 0.9)
    alpha = np.where(fem, -0.241, -0.302)
    ratio = scr / kappa
    egfr = (142.0 * np.minimum(ratio, 1.0) ** alpha * np.maximum(ratio, 1.0) ** -1.200
            * 0.9938 ** age * np.where(fem, 1.012, 1.0))
    return np.where(np.isfinite(scr) & (scr > 0) & np.isfinite(age), egfr, np.nan)


def egfr_by_policy(scr, age, sex=None, policy: str = "worst") -> np.ndarray:
    """TFGe cuando el sexo puede faltar. Con sexo conocido se usa ese valor; si falta:
    worst = mínimo entre el cálculo como hombre y como mujer (estadio más conservador),
    male / female = asumir ese sexo (análisis de sensibilidad)."""
    scr, age = np.asarray(scr, float), np.asarray(age, float)
    e_m, e_f = ckd_epi_2021(scr, age, np.zeros_like(scr)), ckd_epi_2021(scr, age, np.ones_like(scr))
    fallback = {"worst": np.fmin(e_m, e_f), "male": e_m, "female": e_f}[policy]
    fallback = np.where(np.isnan(e_m) | np.isnan(e_f), np.nan, fallback)
    if sex is None:
        return fallback
    sex = np.asarray(sex, float)
    known = ckd_epi_2021(scr, age, np.nan_to_num(sex))
    return np.where(np.isnan(sex), fallback, known)


def g_stage(egfr, cuts=None) -> np.ndarray:
    """Estadio ordinal 0..4 (G1..G5); NaN si falta la TFGe."""
    c = list(cuts or DEFAULT_CUTS)
    e = np.atleast_1d(np.asarray(egfr, dtype=float))
    stage = np.sum(e[:, None] < np.asarray(c, dtype=float)[None, :], axis=1).astype(float)
    return np.where(np.isnan(e), np.nan, stage)


# ---------------------------------------------------------------- lectura
def read_arff(path: str | Path) -> pd.DataFrame:
    """Lector tolerante de ARFF (el de UCI tiene filas con comas duplicadas y tabuladores)."""
    names, rows, in_data, bad = [], [], False, 0
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if not in_data:
            m = re.match(r"@attribute\s+'?([^'\s]+)'?\s", s, re.IGNORECASE)
            if m:
                names.append(m.group(1))
            in_data = s.lower().startswith("@data")
            continue
        fields = [f.strip() for f in s.split(",")]
        if len(fields) > len(names):
            fields = [f for f in fields if f != ""]          # comas sobrantes
        if len(fields) != len(names):
            bad += 1
            continue
        rows.append(fields)
    df = pd.DataFrame(rows, columns=names)
    df.attrs["rejected_rows"] = bad
    return df


def read_table(path: str | Path, cfg: dict) -> pd.DataFrame:
    """Lee ARFF, Excel o CSV (separador y codificación de config.yaml)."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".arff":
        return read_arff(path)
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, sep=cfg["data"]["sep"], encoding=cfg["data"]["encoding"], low_memory=False)


# ---------------------------------------------------------------- limpieza
def _coerce_numeric(df: pd.DataFrame, na_values) -> pd.DataFrame:
    """Normaliza texto (espacios, tabuladores, mayúsculas), marca faltantes y convierte a
    número las columnas que lo son en ≥90 % (admite coma decimal)."""
    na = {str(v).strip().lower() for v in na_values} | {"nan", "none", "na"}
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("str"):
            s = df[col].astype(str).str.strip().str.lower()
            s = s.where(~s.isin(na), np.nan)
            num = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
            df[col] = num if (s.notna().any() and num.notna().sum() >= 0.9 * s.notna().sum()) else s
    return df


def clean_dataframe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Faltantes ('?', centinela NHANES, códigos 7/9) → NaN, alias → nombres canónicos,
    conversión de unidades, sexo binario (1 = mujer) si existe y filtro de adultos."""
    d = cfg["data"]
    df = df.copy()
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    df = _coerce_numeric(df, d["na_values"])
    num = df.select_dtypes(include="number").columns
    if len(num):
        vals = df[num].to_numpy(dtype=float)
        df[num] = np.where(np.isclose(vals, d["na_sentinel"], rtol=1e-3, atol=0.0), np.nan, vals)
    q_re = re.compile(d["questionnaire_regex"])
    for col in num:
        if q_re.match(col):
            df.loc[df[col].isin(d["questionnaire_missing_codes"]), col] = np.nan
    for canon, aliases in d["aliases"].items():
        for a in aliases:
            if a in df.columns and canon not in df.columns:
                df = df.rename(columns={a: canon})
                break
    if "creatinine" not in df.columns and "creatinine_umol" in df.columns:
        df["creatinine"] = df["creatinine_umol"] / UMOL_PER_MGDL
    if "sex" in df.columns:
        fem = {str(v).lower() for v in d["female_values"]}
        raw = df["sex"]
        df["sex"] = np.where(raw.isna(), np.nan,
                             raw.astype(str).str.lower().str.replace(".0", "", regex=False).isin(fem).astype(float))
    if "age" in df.columns:
        df = df[df["age"] >= d["min_age"]]
    df = df.drop(columns=[c for c in d["id_columns"] if c in df.columns])
    return df.reset_index(drop=True)


def load_dataset(path: str | Path, cfg: dict) -> pd.DataFrame:
    """Carga y limpia el dataset base. Requiere creatinina y edad (el sexo es opcional)."""
    df = clean_dataframe(read_table(path, cfg), cfg)
    missing = [c for c in ("creatinine", "age") if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias {missing}. Ajuste data.aliases en config.yaml.")
    return df


# ------------------------------------------------- dataset sintético (demo/tests)
def make_synthetic_uci(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Dataset con el formato del UCI CKD (columnas, códigos de texto y '?'), con
    estructura fisiológica plausible. Solo para pruebas; no tiene validez clínica."""
    rng = np.random.default_rng(seed)
    ckd = rng.random(n) < 0.62
    age = np.clip(rng.normal(np.where(ckd, 55, 46), 15), 5, 90).round()
    dmg = np.where(ckd, rng.gamma(1.6, 0.9, n), np.abs(rng.normal(0, 0.12, n)))
    sc = np.exp(np.log(0.95) + 1.05 * dmg + rng.normal(0, 0.15, n))
    yn = lambda p: np.where(rng.random(n) < p, "yes", "no")
    df = pd.DataFrame({
        "age": age, "bp": (72 + 6 * dmg + rng.normal(0, 10, n)).round(-1),
        "sg": np.where(ckd, rng.choice([1.005, 1.010, 1.015, 1.020], n), rng.choice([1.020, 1.025], n)),
        "al": np.clip(np.round(dmg * 1.3 + rng.normal(0, 0.7, n)), 0, 5) * ckd,
        "su": np.clip(np.round(rng.exponential(0.4, n) * ckd), 0, 5),
        "rbc": np.where(rng.random(n) < 0.15 + 0.2 * ckd, "abnormal", "normal"),
        "pc": np.where(rng.random(n) < 0.1 + 0.25 * ckd, "abnormal", "normal"),
        "pcc": np.where(rng.random(n) < 0.1 * ckd, "present", "notpresent"),
        "ba": np.where(rng.random(n) < 0.05 * ckd + 0.01, "present", "notpresent"),
        "bgr": (110 + 60 * ckd * rng.random(n) + rng.normal(0, 25, n)).round(),
        "bu": (25 * sc ** 0.9 * rng.lognormal(0, 0.25, n)).round(),
        "sc": sc.round(1),
        "sod": (140 - 2.5 * dmg + rng.normal(0, 4, n)).round(),
        "pot": (4.3 + 0.35 * dmg + rng.normal(0, 0.5, n)).round(1),
        "hemo": (15 - 1.4 * dmg + rng.normal(0, 1.2, n)).round(1),
        "pcv": None, "wc": (rng.normal(8000, 2200, n)).round(-2), "rc": (5.2 - 0.45 * dmg + rng.normal(0, 0.5, n)).round(1),
        "htn": np.where(ckd, yn(0.55), "no"), "dm": np.where(ckd, yn(0.45), "no"), "cad": np.where(ckd, yn(0.1), "no"),
        "appet": np.where(rng.random(n) < 0.25 * ckd, "poor", "good"),
        "pe": np.where(rng.random(n) < 0.2 * ckd, "yes", "no"), "ane": np.where(rng.random(n) < 0.18 * ckd, "yes", "no"),
        "class": np.where(ckd, "ckd", "notckd"),
    })
    df["pcv"] = (df["hemo"] * 3 + rng.normal(0, 2, n)).round()
    for col, p in {"rbc": .35, "pc": .15, "sod": .2, "pot": .2, "wc": .25, "rc": .3, "bgr": .1, "bu": .05,
                   "hemo": .12, "pcv": .17, "sg": .1, "al": .1, "su": .1, "bp": .03, "age": .02, "sc": .03}.items():
        df[col] = df[col].astype(object)
        df.loc[rng.random(n) < p, col] = "?"
    return df


def write_synthetic(path: str | Path, cfg: dict, n: int = 400, seed: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_uci(n, seed).to_excel(path, index=False)
    return path
