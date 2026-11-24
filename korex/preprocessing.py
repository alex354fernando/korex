"""C2 · Preprocesamiento sin fuga: transformador T_θ (θ estimado solo con D_ent).
MissForest → log(1+x) en variables asimétricas → escalado robusto → codificación."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class MissForest:
    """Imputación iterativa con bosques aleatorios (Stekhoven & Bühlmann, 2012).
    Se detiene cuando la diferencia entre iteraciones deja de disminuir."""

    def __init__(self, cat_idx=(), n_estimators=50, max_depth=12, max_iter=6,
                 transform_iter=2, seed=42, n_jobs=-1):
        self.cat_idx = set(cat_idx)
        self.n_estimators, self.max_depth, self.max_iter = n_estimators, max_depth, max_iter
        self.transform_iter, self.seed, self.n_jobs = transform_iter, seed, n_jobs

    def _model(self, j):
        kw = dict(n_estimators=self.n_estimators, max_depth=self.max_depth,
                  random_state=self.seed, n_jobs=self.n_jobs)
        return RandomForestClassifier(**kw) if j in self.cat_idx else RandomForestRegressor(**kw)

    def _initial_fill(self, X, mask):
        X = X.copy()
        for j in range(X.shape[1]):
            X[mask[:, j], j] = self.init_[j]
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        mask = np.isnan(X)
        self.init_ = np.zeros(p)
        for j in range(p):
            col = X[~mask[:, j], j]
            if col.size == 0:
                self.init_[j] = 0.0
            elif j in self.cat_idx:
                vals, cnt = np.unique(col, return_counts=True)
                self.init_[j] = vals[np.argmax(cnt)]
            else:
                self.init_[j] = col.mean()
        X_old = self._initial_fill(X, mask)
        order = [j for j in np.argsort(mask.sum(0)) if mask[:, j].any()]
        cont = [j for j in range(p) if j not in self.cat_idx]
        cat = [j for j in range(p) if j in self.cat_idx]
        prev = (np.inf, np.inf)
        self.n_iter_ = 0
        for _ in range(self.max_iter if order else 0):
            X_new = X_old.copy()
            for j in order:
                obs, others = ~mask[:, j], [c for c in range(p) if c != j]
                mdl = self._model(j).fit(X_new[obs][:, others], X_new[obs, j])
                X_new[mask[:, j], j] = mdl.predict(X_new[mask[:, j]][:, others])
            d_cont = (np.sum((X_new[:, cont] - X_old[:, cont]) ** 2) / max(np.sum(X_new[:, cont] ** 2), 1e-12)
                      if cont else 0.0)
            n_cat_miss = mask[:, cat].sum() if cat else 0
            d_cat = (np.sum(X_new[:, cat] != X_old[:, cat]) / n_cat_miss) if n_cat_miss else 0.0
            self.n_iter_ += 1
            if d_cont >= prev[0] and d_cat >= prev[1]:
                break                                    # la diferencia dejó de disminuir
            X_old, prev = X_new, (d_cont, d_cat)
        # modelos finales para transformar datos nuevos (ajustados solo con entrenamiento)
        self.models_ = {}
        for j in range(p):
            obs, others = ~mask[:, j], [c for c in range(p) if c != j]
            if obs.sum() >= 5 and np.unique(X_old[obs, j]).size > 1:
                self.models_[j] = self._model(j).fit(X_old[obs][:, others], X_old[obs, j])
        self.miss_order_ = list(np.argsort(mask.sum(0)))
        return X_old

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        mask = np.isnan(X)
        Xf = self._initial_fill(X, mask)
        cols = [j for j in self.miss_order_ if mask[:, j].any() and j in self.models_]
        p = X.shape[1]
        for _ in range(self.transform_iter if cols else 0):
            for j in cols:
                others = [c for c in range(p) if c != j]
                Xf[mask[:, j], j] = self.models_[j].predict(Xf[mask[:, j]][:, others])
        return Xf


class Preprocessor:
    """T_θ: fit en D_ent, transform en D_pru sin reestimar parámetros."""

    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg, self.seed = cfg, seed

    # ---------------------------------------------------------------- tipos
    def _detect_types(self, df: pd.DataFrame) -> dict:
        nominal, ordinal = set(self.cfg["data"]["nominal"]), set(self.cfg["data"]["ordinal"])
        types = {}
        for c in df.columns:
            s = df[c]
            if c in nominal:
                types[c] = "nom"
            elif c in ordinal:
                types[c] = "ord"
            elif s.dropna().nunique() <= 2:
                types[c] = "bin"                       # sí/no, normal/anormal, 0/1
            elif not pd.api.types.is_numeric_dtype(s):
                types[c] = "nom"
            else:
                types[c] = "cont"
        return types

    def _encode(self, df: pd.DataFrame) -> np.ndarray:
        df = df.reindex(columns=self.columns_)
        out = np.full((len(df), len(self.columns_)), np.nan)
        for j, c in enumerate(self.columns_):
            t, s = self.types_[c], df[c]
            if t == "bin":
                out[:, j] = s.map(self.bin_maps_[c]).astype(float)
            elif t == "nom":
                out[:, j] = s.map(lambda v: self.nom_cats_[c].get(_norm_cat(v), np.nan)).astype(float)
            else:
                out[:, j] = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        return out

    # ---------------------------------------------------------------- fit / transform
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.fit_transform(df)
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.columns_ = list(df.columns)
        self.types_ = self._detect_types(df)
        self.bin_maps_, self.nom_cats_, self.ord_levels_ = {}, {}, {}
        for c, t in self.types_.items():
            if t == "bin":
                vals = sorted(df[c].dropna().unique().tolist())
                self.bin_maps_[c] = {v: float(i) for i, v in enumerate(vals)}
            elif t == "nom":
                cats = sorted({_norm_cat(v) for v in df[c].dropna()})
                self.nom_cats_[c] = {v: float(i) for i, v in enumerate(cats)}
            elif t == "ord":
                self.ord_levels_[c] = np.sort(pd.to_numeric(df[c], errors="coerce").dropna().unique())
        E = self._encode(df)
        mf = self.cfg["preprocessing"]["missforest"]
        cat_idx = [j for j, c in enumerate(self.columns_) if self.types_[c] in ("bin", "nom")]
        self.imputer_ = MissForest(cat_idx, mf["n_estimators"], mf["max_depth"], mf["max_iter"],
                                   mf["transform_iter"], self.seed, self.cfg["project"]["n_jobs"])
        I = self.imputer_.fit_transform(E)
        thr = self.cfg["preprocessing"]["skew_threshold"]
        self.log_cols_, self.median_, self.iqr_ = [], {}, {}
        for j, c in enumerate(self.columns_):
            if self.types_[c] != "cont":
                continue
            x = I[:, j]
            if x.min() >= 0 and skew(x) > thr:
                self.log_cols_.append(c)
                x = np.log1p(x)
            q1, med, q3 = np.percentile(x, [25, 50, 75])
            self.median_[c] = float(med)
            self.iqr_[c] = float(q3 - q1) if q3 > q1 else float(np.std(x) or 1.0)
        return self._finish(I)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._finish(self.imputer_.transform(self._encode(df)))

    def _finish(self, I: np.ndarray) -> pd.DataFrame:
        blocks, names, types, origin, groups = [], [], {}, {}, {}
        for j, c in enumerate(self.columns_):
            t, x = self.types_[c], I[:, j]
            if t == "cont":
                if c in self.log_cols_:
                    x = np.log1p(np.clip(x, 0, None))
                blocks.append((x - self.median_[c]) / self.iqr_[c]); names.append(c); types[c] = "cont"
            elif t == "ord":
                lv = self.ord_levels_[c]
                x = lv[np.abs(x[:, None] - lv[None, :]).argmin(1)] if lv.size else x
                blocks.append(x); names.append(c); types[c] = "ord"
            elif t == "bin":
                blocks.append(np.round(np.clip(x, 0, 1))); names.append(c); types[c] = "bin"
        for c, cats in self.nom_cats_.items():
            x, groups[c] = np.round(I[:, self.columns_.index(c)]), []
            for cat, code in cats.items():
                name = f"{c}={cat}"
                blocks.append((x == code).astype(float)); names.append(name)
                types[name] = "nom"; groups[c].append(name); origin[name] = c
        for n in names:
            origin.setdefault(n, n)
        self.out_columns_, self.out_types_, self.origin_, self.onehot_groups_ = names, types, origin, groups
        return pd.DataFrame(np.column_stack(blocks) if blocks else np.empty((len(I), 0)), columns=names)

    def type_record(self) -> dict:
        """Registro de tipos de variable (salida de C2)."""
        return {"raw": self.types_, "log_transformed": self.log_cols_, "onehot_groups": self.onehot_groups_}


def _norm_cat(v) -> str:
    """Normaliza categorías (1, 1.0 y '1' son la misma)."""
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return str(v).strip()
