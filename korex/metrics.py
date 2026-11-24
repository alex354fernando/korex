"""C7 · Métricas nominales y ordinales, calibración y función objetivo J."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)


def expected_calibration_error(y: np.ndarray, P: np.ndarray, n_bins: int = 10) -> float:
    """ECE de la etiqueta de mayor probabilidad."""
    conf, pred = P.max(1), P.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs((pred[sel] == y[sel]).mean() - conf[sel].mean())
    return float(ece)


def brier_multiclass(y: np.ndarray, P: np.ndarray) -> float:
    return float(np.mean(np.sum((P - np.eye(P.shape[1])[y]) ** 2, axis=1)))


def mean_cost(y: np.ndarray, yhat: np.ndarray, C: np.ndarray) -> float:
    return float(np.mean(C[y, yhat]))


def ordinal_objective(y, yhat, C: np.ndarray, lam: float = 0.5) -> float:
    """J = F1 macro − λ · costo medio / costo máximo (mayor es mejor)."""
    y, yhat = np.asarray(y, int), np.asarray(yhat, int)
    f1 = f1_score(y, yhat, average="macro", labels=list(range(C.shape[0])), zero_division=0)
    return float(f1 - lam * mean_cost(y, yhat, C) / max(C.max(), 1e-12))


def calibration_curve_data(y: np.ndarray, P: np.ndarray, n_bins: int = 10) -> dict:
    """Curva de fiabilidad por clase (one-vs-rest)."""
    out, bins = {}, np.linspace(0, 1, n_bins + 1)
    for k in range(P.shape[1]):
        pk, yk = P[:, k], (y == k).astype(float)
        idx = np.clip(np.digitize(pk, bins) - 1, 0, n_bins - 1)
        mp, fo = [], []
        for b in range(n_bins):
            sel = idx == b
            if sel.sum() >= 5:
                mp.append(float(pk[sel].mean())); fo.append(float(yk[sel].mean()))
        out[k] = {"mean_predicted": mp, "fraction_positive": fo}
    return out


def evaluate(y, P: np.ndarray, yhat=None, C: np.ndarray | None = None, lam: float = 0.5) -> dict:
    """Conjunto completo de métricas C7."""
    y = np.asarray(y, int)
    K = P.shape[1]
    yhat = P.argmax(1) if yhat is None else np.asarray(yhat, int)
    C = np.abs(np.subtract.outer(np.arange(K), np.arange(K))) ** 2 if C is None else C
    labels = list(range(K))
    present = [k for k in labels if np.any(y == k)]
    try:
        Pn = P[:, present] / np.clip(P[:, present].sum(1, keepdims=True), 1e-12, None)
        auc = float(roc_auc_score(y, Pn, multi_class="ovr", average="macro", labels=present)) if len(present) > 2 else float("nan")
    except ValueError:
        auc = float("nan")
    kw = dict(labels=labels, zero_division=0)
    return {
        # métricas principales (macro: promedio simple entre estadios)
        "precision_macro": float(precision_score(y, yhat, average="macro", **kw)),
        "recall_macro": float(recall_score(y, yhat, average="macro", **kw)),
        "f1_macro": float(f1_score(y, yhat, average="macro", **kw)),
        "auc_ovr": auc,
        "accuracy": float(accuracy_score(y, yhat)),
        # métricas ordinales y de calibración (panel técnico)
        "mae_ordinal": float(np.mean(np.abs(y - yhat))),
        "qwk": float(cohen_kappa_score(y, yhat, weights="quadratic", labels=labels)),
        "lwk": float(cohen_kappa_score(y, yhat, weights="linear", labels=labels)),
        "brier": brier_multiclass(y, P),
        "ece": expected_calibration_error(y, P),
        "mean_cost": mean_cost(y, yhat, C),
        "objective_J": ordinal_objective(y, yhat, C, lam),
        **screening_metrics(y, P, yhat),
        "precision_per_class": [float(v) for v in precision_score(y, yhat, average=None, **kw)],
        "recall_per_class": [float(v) for v in recall_score(y, yhat, average=None, **kw)],
        "f1_per_class": [float(v) for v in f1_score(y, yhat, average=None, **kw)],
        "confusion_matrix": confusion_matrix(y, yhat, labels=labels).tolist(),
    }


def _safe_auc(t, s) -> float:
    return float(roc_auc_score(t, s)) if 0 < t.sum() < len(t) else float("nan")


def screening_metrics(y, P: np.ndarray, yhat, threshold: float = 0.5) -> dict:
    """Métricas de tamizaje: acierto a ±1 estadio y detección de TFGe < 60 (G3–G5) y < 30 (G4–G5).
    Útiles sobre todo para el modelo sin creatinina, cuyo objetivo es decidir a quién pedir creatinina."""
    y, yhat = np.asarray(y, int), np.asarray(yhat, int)
    out = {"accuracy_within1": float(np.mean(np.abs(y - yhat) <= 1))}
    if P.shape[1] < 4:
        return out
    for name, k in (("lt60", 2), ("lt30", 3)):
        t, s = (y >= k).astype(int), P[:, k:].sum(1)
        pred = s >= threshold
        out[f"auc_{name}"] = _safe_auc(t, s)
        out[f"sens_{name}"] = float(pred[t == 1].mean()) if t.any() else float("nan")
        out[f"spec_{name}"] = float((~pred[t == 0]).mean()) if (t == 0).any() else float("nan")
    return out


def pr_curves(y: np.ndarray, P: np.ndarray) -> dict:
    """Curvas precisión-exhaustividad one-vs-rest por estadio (AP = área bajo la curva PR)."""
    out = {}
    for k in range(P.shape[1]):
        t = (np.asarray(y) == k)
        if 0 < t.sum() < len(t):
            pr, rc, _ = precision_recall_curve(t, P[:, k])
            out[k] = {"precision": pr.tolist(), "recall": rc.tolist(), "ap": float(average_precision_score(t, P[:, k])),
                      "prevalence": float(t.mean())}
    return out


def roc_curves(y: np.ndarray, P: np.ndarray) -> dict:
    """Curvas ROC one-vs-rest por estadio (para figuras)."""
    out = {}
    for k in range(P.shape[1]):
        if 0 < np.sum(y == k) < len(y):
            fpr, tpr, _ = roc_curve(y == k, P[:, k])
            out[k] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                      "auc": float(roc_auc_score(y == k, P[:, k]))}
    return out


def summarize_folds(fold_metrics: list[dict]) -> dict:
    """Media, desviación estándar e IC95 % (percentiles) de las métricas escalares."""
    keys = [k for k, v in fold_metrics[0].items() if isinstance(v, float)]
    out = {}
    for k in keys:
        v = np.array([m[k] for m in fold_metrics], dtype=float)
        v = v[~np.isnan(v)]
        if v.size:
            out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                      "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    return out
