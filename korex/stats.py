"""Comparación estadística de modelos sobre los mismos pliegues externos:
prueba de Friedman y diferencia crítica de Nemenyi (Demšar, 2006)."""
from __future__ import annotations

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, studentized_range

MODEL_NAMES = {"stack": "KORE-X (stacking ordinal)", "xgb": "XGBoost", "lgbm": "LightGBM", "cat": "CatBoost",
               "et": "Extra Trees", "lr": "Regresión logística", "svm": "SVM", "lr_core": "Logística (creatinina + edad)"}


def fold_table(records: list[dict], folds: list[dict], metric: str = "f1_macro") -> dict:
    """{modelo: [valor por pliegue]} con el stacking y cada modelo base."""
    out = {"stack": [f[metric] for f in folds]}
    names = sorted({n for r in records for n in r.get("base_models", {})})
    for n in names:
        vals = [r.get("base_models", {}).get(n, {}).get(metric) for r in records]
        if all(v is not None for v in vals):
            out[n] = vals
    return out


def friedman_nemenyi(table: dict, alpha: float = 0.05) -> dict:
    """Rangos medios (1 = mejor), estadístico de Friedman y CD de Nemenyi."""
    names = list(table)
    M = np.array([table[n] for n in names], dtype=float).T          # pliegues × modelos
    n, k = M.shape
    if k < 3 or n < 2:
        return {}
    ranks = np.vstack([rankdata(-row) for row in M])                 # mayor métrica → rango 1
    mean_rank = ranks.mean(0)
    try:
        stat, p = friedmanchisquare(*M.T)
    except ValueError:                                               # todos iguales
        stat, p = 0.0, 1.0
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = float(q * np.sqrt(k * (k + 1) / (6 * n)))
    order = np.argsort(mean_rank)
    best = names[order[0]]
    return {"models": names, "mean_rank": {nm: float(r) for nm, r in zip(names, mean_rank)},
            "mean": {nm: float(M[:, j].mean()) for j, nm in enumerate(names)},
            "std": {nm: float(M[:, j].std(ddof=1)) for j, nm in enumerate(names)},
            "friedman_stat": float(stat), "p_value": float(p), "n_folds": int(n), "alpha": alpha, "cd": cd,
            "best": best,
            "not_different_from_best": [nm for nm in names if mean_rank[names.index(nm)] - mean_rank[order[0]] < cd]}
