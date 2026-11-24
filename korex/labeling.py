"""C1 (parte 2) · Etiquetado por estadio KDIGO G1–G5, escenarios A/B, línea base de
regla e índice de circularidad IC_M."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .kdigo import egfr_by_policy, g_stage

EGFR_COL = "egfr_ckdepi2021"
V_DEF = ["creatinine", EGFR_COL]               # variables que definen el estadio G


@dataclass
class LabeledData:
    """Conjunto supervisado D con trazabilidad del etiquetado."""
    X: pd.DataFrame                    # variables admisibles (sin fuga)
    y: np.ndarray                      # estadio 0..4 (G1..G5)
    egfr: np.ndarray                   # TFGe usada para etiquetar
    info: dict = field(default_factory=dict)


def imbalance_ratios(y: np.ndarray, classes) -> dict:
    """IR_k = max_c n_c / n_k."""
    counts = {int(k): int(np.sum(y == k)) for k in classes}
    n_max = max(counts.values())
    return {k: (n_max / v if v else float("inf")) for k, v in counts.items()}


def label_dataset(df: pd.DataFrame, cfg: dict) -> LabeledData:
    """Recalcula la TFGe (CKD-EPI 2021; sin sexo registrado → política de config),
    asigna el estadio G, elimina columnas de fuga y filas sin creatinina/edad."""
    k = cfg["kdigo"]
    d = df.copy()
    sex = d["sex"] if "sex" in d.columns else None
    policy = k.get("sex_policy", "worst")
    d[EGFR_COL] = egfr_by_policy(d["creatinine"], d["age"], sex, policy)
    y = g_stage(d[EGFR_COL], k["g_cuts"])
    # análisis de sensibilidad del supuesto de sexo (antes de filtrar)
    st_m = g_stage(egfr_by_policy(d["creatinine"], d["age"], sex, "male"), k["g_cuts"])
    st_f = g_stage(egfr_by_policy(d["creatinine"], d["age"], sex, "female"), k["g_cuts"])
    keep = ~np.isnan(y)
    disc = float(np.mean(st_m[keep] != st_f[keep])) if keep.any() else 0.0
    removed_leak = [c for c in d.columns if c.lower() in {x.lower() for x in cfg["data"]["leakage_columns"]}]
    diag = d[removed_leak[0]].loc[keep].value_counts().to_dict() if removed_leak else {}
    d, y = d.loc[keep].reset_index(drop=True), y[keep].astype(int)
    d = d.drop(columns=removed_leak)
    miss = d.isna().mean()
    protected = set(V_DEF) | {"age", "sex"}
    d = d.drop(columns=[c for c in d.columns if miss[c] > cfg["data"]["max_missing_fraction"] and c not in protected])
    d = d.drop(columns=[c for c in d.columns if d[c].nunique(dropna=True) <= 1 and c not in protected])
    counts = {int(c): int(np.sum(y == c)) for c in k["classes"]}
    info = {"n": int(len(y)), "n_dropped_unlabeled": int((~keep).sum()), "class_counts": counts,
            "imbalance_ratio": imbalance_ratios(y, k["classes"]), "guideline": k["guideline"],
            "equation": k["equation"], "sex_available": sex is not None, "sex_policy": policy,
            "sex_discordance_rate": disc,
            "class_counts_if_male": {int(c): int(np.sum(st_m[keep] == c)) for c in k["classes"]},
            "class_counts_if_female": {int(c): int(np.sum(st_f[keep] == c)) for c in k["classes"]},
            "removed_leakage_columns": removed_leak, "diagnosis_column_counts": {str(a): int(b) for a, b in diag.items()}}
    return LabeledData(X=d, y=y, egfr=d[EGFR_COL].to_numpy(), info=info)


def scenario_features(columns, cfg: dict, scenario: str) -> list[str]:
    """A (modelo KORE-X) = todas las variables salvo la TFGe derivada;
    B (control interno) = sin creatinina ni TFGe, para el índice de circularidad."""
    cols = list(columns)
    if scenario == "A":
        return [c for c in cols if c != EGFR_COL]
    if scenario == "B":
        excluded = set(V_DEF) | set(cfg["data"]["definitional_extra"])
        return [c for c in cols if c not in excluded]
    raise ValueError("scenario debe ser 'A' o 'B'")


# ------------------------------------------------------------ líneas base (M_0, regla)
def fit_rule_tree(X: pd.DataFrame, y: np.ndarray, seed: int) -> DecisionTreeClassifier:
    """Árbol de profundidad 3 entrenado solo con la TFGe (referencia mínima de la regla)."""
    Z = X[[EGFR_COL]].fillna(X[EGFR_COL].median())
    return DecisionTreeClassifier(max_depth=3, random_state=seed).fit(Z, y)


def rule_tree_proba(tree: DecisionTreeClassifier, X: pd.DataFrame, K: int) -> np.ndarray:
    Z = X[[EGFR_COL]].fillna(X[EGFR_COL].median())
    P = np.zeros((len(Z), K))
    P[:, tree.classes_] = tree.predict_proba(Z)
    return P


def prior_baseline(y_train: np.ndarray, n_test: int, seed: int, K: int):
    """Clasificador sin información: probabilidades = a priori; predicción ~ a priori."""
    prior = np.bincount(y_train, minlength=K) / len(y_train)
    rng = np.random.default_rng(seed)
    return np.tile(prior, (n_test, 1)), rng.choice(K, size=n_test, p=prior)


def circularity_index(m_a: float, m_b: float, m_0: float) -> float:
    """IC_M = (M_A − M_B) / (M_A − M_0). →1: el desempeño de A lo explica la regla."""
    den = m_a - m_0
    return float((m_a - m_b) / den) if abs(den) > 1e-12 else float("nan")


def circularity_table(metrics_a: dict, metrics_b: dict, metrics_0: dict,
                      keys=("f1_macro", "qwk", "accuracy", "auc_ovr")) -> dict:
    """IC_M para varias métricas de C7 (solo métricas «mayor es mejor»)."""
    return {k: circularity_index(metrics_a[k], metrics_b[k], metrics_0[k])
            for k in keys if k in metrics_a and k in metrics_b and k in metrics_0}
