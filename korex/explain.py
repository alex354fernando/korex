"""C8 · Explicabilidad dual SHAP + LIME sobre el riesgo esperado E[y] = Σ k·p_k,
agregación por variable original y por grupo de C3, y convergencia SHAP–LIME (ICSL)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def icsl(phi: np.ndarray, lime_w: np.ndarray, top_k: int = 5) -> dict:
    """Índice de convergencia SHAP–LIME ∈ [0,1]: media de (ρ_Spearman de |atribuciones|+1)/2,
    Jaccard del top-k y concordancia de signo en el top-k de SHAP."""
    a, b = np.abs(phi), np.abs(lime_w)
    rho = spearmanr(a, b).statistic if a.size > 2 and a.std() > 0 and b.std() > 0 else 0.0
    rho = 0.0 if np.isnan(rho) else float(rho)
    k = min(top_k, a.size)
    ta, tb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    jac = len(ta & tb) / max(len(ta | tb), 1)
    top = np.argsort(-a)[:k]
    sign = float(np.mean(np.sign(phi[top]) == np.sign(lime_w[top]))) if k else 0.0
    return {"spearman": rho, "jaccard_topk": float(jac), "sign_agreement": sign,
            "icsl": float(np.mean([(rho + 1) / 2, jac, sign]))}


class Explainer:
    """Explicador modelo-agnóstico del ensamble en el espacio transformado por C2."""

    def __init__(self, model, background: pd.DataFrame, cfg: dict, seed: int = 42):
        self.model, self.cfg, self.seed = model, cfg, seed
        self.background = background.reset_index(drop=True)
        self.columns = list(background.columns)
        self.origin = model.pre_.origin_
        self.origins = list(dict.fromkeys(self.origin[c] for c in self.columns))
        self.K = len(cfg["kdigo"]["classes"])

    # ------------------------------------------------------------ funciones objetivo
    def _f(self, target):
        def f(X):
            P = self.model.ens_.predict_proba(pd.DataFrame(np.atleast_2d(X), columns=self.columns))
            return P @ np.arange(self.K) if target == "risk" else P[:, int(target)]
        return f

    def shap_values(self, xt: np.ndarray, target="risk") -> np.ndarray:
        import shap
        masker = shap.maskers.Independent(self.background.to_numpy(), max_samples=len(self.background))
        expl = shap.PermutationExplainer(self._f(target), masker, seed=self.seed)
        p = xt.shape[-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = expl(np.atleast_2d(xt), max_evals=max(self.cfg["explain"]["shap_max_evals"], 2 * p + 1), silent=True)
        return np.asarray(sv.values)[0]

    def lime_weights(self, xt: np.ndarray, target="risk") -> np.ndarray:
        from lime.lime_tabular import LimeTabularExplainer
        disc = self.cfg["explain"].get("lime_discretize", True)
        bg = self.background.to_numpy()
        lime = LimeTabularExplainer(bg, mode="regression", feature_names=self.columns, discretize_continuous=disc,
                                    sample_around_instance=not disc, random_state=self.seed)
        x = np.ravel(xt)
        exp = lime.explain_instance(x, self._f(target), num_features=len(self.columns),
                                    num_samples=self.cfg["explain"]["lime_num_samples"])
        w = np.zeros(len(self.columns))
        amap = exp.as_map()                       # en regresión LIME usa la clave 1
        for j, val in amap.get(1, next(iter(amap.values()))):
            w[j] = val
        if not disc:                               # pendiente → aporte local: β_j · (x_j − media del fondo)
            w = w * (x - bg.mean(0)) / np.where(bg.std(0) > 0, bg.std(0), 1.0)
        return w

    def _aggregate(self, v: np.ndarray) -> np.ndarray:
        """Suma de atribuciones por variable original (one-hot → nominal)."""
        s = pd.Series(v, index=[self.origin[c] for c in self.columns]).groupby(level=0).sum()
        return s.reindex(self.origins).to_numpy()

    # ------------------------------------------------------------ explicación local
    def explain_instance(self, xt_row: pd.DataFrame, raw_row: dict | None = None, target="risk") -> dict:
        xt = xt_row[self.columns].to_numpy(dtype=float)
        phi = self._aggregate(self.shap_values(xt, target))
        lw = self._aggregate(self.lime_weights(xt, target))
        conv = icsl(phi, lw, self.cfg["explain"]["top_k"])
        groups = {}
        gmap = self.model.grouper_.group_of()
        for name, val in zip(self.origins, phi):
            members = [c for c in self.columns if self.origin[c] == name]
            gid = gmap.get(members[0], -1)
            groups.setdefault(gid, {"variables": [], "shap": 0.0})
            groups[gid]["variables"].append(name); groups[gid]["shap"] += float(val)
        labels, vlab = self.cfg["variables"], self.cfg.get("value_labels", {})
        factors = []
        for name, s, l in zip(self.origins, phi, lw):
            meta = labels.get(name, {})
            raw = None if raw_row is None else raw_row.get(name)
            factors.append({"variable": name, "label": meta.get("label", name), "unit": meta.get("unit", ""),
                            "value": None if raw is None or (isinstance(raw, float) and np.isnan(raw)) else raw,
                            "value_label": vlab.get(raw, raw) if isinstance(raw, str) else raw,
                            "shap": float(s), "lime": float(l)})
        factors.sort(key=lambda d: -abs(d["shap"]))
        base = float(np.mean(self._f(target)(self.background.to_numpy())))
        return {"target": target, "base_value": base, "factors": factors,
                "groups": [g for g in groups.values() if len(g["variables"]) > 1],
                "convergence": conv, "reading": self.reading(factors)}

    @staticmethod
    def reading(factors: list[dict], n: int = 3) -> str:
        """Lectura textual sin tecnicismos."""
        def fmt(f):
            v = f["value"]
            if isinstance(v, (int, float)):
                v = round(v) if abs(v) >= 100 else round(v, 2)
                return f"{f['label']} ({v:g} {f['unit']})".replace(" )", ")")
            if isinstance(v, str) and v:
                return f"{f['label']} ({f.get('value_label', v)})"
            return f"{f['label']} (no informado, valor estimado)"
        up = [f for f in factors if f["shap"] > 0][:n]
        down = [f for f in factors if f["shap"] < 0][:n]
        parts = []
        if up:
            parts.append("Lo que más acerca a este paciente a un estadio más avanzado: " + ", ".join(fmt(f) for f in up) + ".")
        if down:
            parts.append("Lo que más indica una función renal conservada: " + ", ".join(fmt(f) for f in down) + ".")
        return " ".join(parts) or "Ningún factor destaca de forma clara."

    # ------------------------------------------------------------ análisis global
    def convergence_by_class(self, Xt: pd.DataFrame, y_pred: np.ndarray, n_per_class: int | None = None) -> dict:
        """ICSL medio por clase predicha (objetivo = probabilidad de esa clase)."""
        n_per_class = n_per_class or self.cfg["explain"]["convergence_per_class"]
        rng, out = np.random.default_rng(self.seed), {}
        for k in range(self.K):
            idx = np.where(y_pred == k)[0]
            if idx.size == 0:
                continue
            sel = rng.choice(idx, min(n_per_class, idx.size), replace=False)
            vals = [icsl(self._aggregate(self.shap_values(Xt.iloc[[i]].to_numpy(), k)),
                         self._aggregate(self.lime_weights(Xt.iloc[[i]].to_numpy(), k)),
                         self.cfg["explain"]["top_k"]) for i in sel]
            out[k] = {m: float(np.mean([v[m] for v in vals])) for m in vals[0]} | {"n": int(sel.size)}
        return out

    def shap_batch(self, Xt: pd.DataFrame, target="risk") -> np.ndarray:
        """SHAP agregados por variable original para varias filas (n × variables)."""
        import shap
        masker = shap.maskers.Independent(self.background.to_numpy(), max_samples=len(self.background))
        expl = shap.PermutationExplainer(self._f(target), masker, seed=self.seed)
        X = Xt[self.columns].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = expl(X, max_evals=max(self.cfg["explain"]["shap_max_evals"], 2 * X.shape[1] + 1), silent=True)
        return np.vstack([self._aggregate(v) for v in np.asarray(sv.values)])

    def global_importance(self, Xt: pd.DataFrame, n: int = 40) -> dict:
        """Media de |SHAP| por variable original sobre una muestra."""
        rng = np.random.default_rng(self.seed)
        sel = rng.choice(len(Xt), min(n, len(Xt)), replace=False)
        vals = np.array([self._aggregate(self.shap_values(Xt.iloc[[i]].to_numpy())) for i in sel])
        imp = np.abs(vals).mean(0)
        return dict(sorted(zip(self.origins, map(float, imp)), key=lambda t: -t[1]))


def _shap_chunk(model, background, cfg, seed, Xt):
    return Explainer(model, background, cfg, seed).shap_batch(Xt)


def global_shap(model, background: pd.DataFrame, cfg: dict, X_raw: pd.DataFrame, n: int, seed: int,
                n_jobs: int = 1) -> dict:
    """Explicación global: matriz SHAP (muestra × variable) con los valores originales de cada paciente.
    Sirve para el gráfico de barras, el beeswarm y los gráficos de dependencia (igual que la literatura)."""
    from joblib import Parallel, delayed
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X_raw), min(n, len(X_raw)), replace=False))
    raw = X_raw.iloc[idx].reset_index(drop=True)
    Xt = model.transform(raw)
    chunks = [c for c in np.array_split(np.arange(len(Xt)), max(1, n_jobs)) if len(c)]
    parts = Parallel(n_jobs=len(chunks))(delayed(_shap_chunk)(model, background, cfg, seed, Xt.iloc[c]) for c in chunks)
    S = np.vstack(parts)
    names = list(dict.fromkeys(model.pre_.origin_[c] for c in background.columns))
    cats = cfg.get("categorical_values", {})
    values = {}
    for v in names:                                     # valor original numérico (categorías → 0, 1, …)
        col = raw[v] if v in raw else pd.Series([np.nan] * len(raw))
        if v in cats:
            values[v] = col.map({c: i for i, c in enumerate(cats[v])}).astype(float).tolist()
        else:
            values[v] = pd.to_numeric(col, errors="coerce").astype(float).tolist()
    imp = np.abs(S).mean(0)
    total = imp.sum() or 1.0
    return {"variables": names, "shap": S.tolist(), "values": values, "index": idx.tolist(),
            "importance": dict(sorted(zip(names, map(float, imp)), key=lambda t: -t[1])),
            "share": dict(sorted(zip(names, map(float, imp / total)), key=lambda t: -t[1]))}
