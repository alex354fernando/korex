"""C4 · OA-SMOTE-NC: SMOTE-NC + Borderline-SMOTE con dos restricciones ordinales
(frontera adyacente y filtro de coherencia ordinal) y ponderación por costo."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def ordinal_cost_matrix(K: int = 4, p: int = 2, delta: float = 0.0) -> np.ndarray:
    """C_kj = |k − j|^p · (1 + δ·1[j < k]); k = clase real, j = clase decidida."""
    k, j = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    return np.abs(k - j) ** p * (1.0 + delta * (j < k))


def class_weights(y: np.ndarray, K: int, beta: float) -> np.ndarray:
    """w_k = (ñ / (K·ñ_k))^β."""
    counts = np.bincount(y, minlength=K).astype(float)
    w = np.where(counts > 0, (len(y) / (K * np.maximum(counts, 1))) ** beta, 0.0)
    return w


class OASMOTENC:
    """Sobremuestreo ordinal para variables mixtas (salida de C2).
    mode="oa": OA-SMOTE-NC completo · mode="classic": SMOTE-NC clásico (todas las semillas,
    sin frontera adyacente ni filtro ordinal) · mode="none": sin sintéticos (ablación)."""

    def __init__(self, types: dict, onehot_groups: dict, r=0.5, m=10, k_s=5, beta=0.5,
                 seed=42, max_attempts_factor=10, K=4, mode: str = "oa"):
        self.mode = mode
        self.types, self.onehot_groups = types, onehot_groups
        self.r, self.m, self.k_s, self.beta = r, int(m), int(k_s), beta
        self.seed, self.max_attempts_factor, self.K = seed, max_attempts_factor, K

    def _embedding(self, X: np.ndarray, med: float) -> np.ndarray:
        """Escala las nominales para que d² = Σ_cont (Δ)² + Med²·Σ_nom 1[≠]."""
        E = X.copy()
        E[:, self.bin_idx] *= med
        E[:, self.oh_idx] *= med / math.sqrt(2.0)          # un cambio de categoría mueve 2 columnas
        return E

    def fit_resample(self, X: pd.DataFrame, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        cols = list(X.columns)
        Xa, y = X.to_numpy(dtype=float), np.asarray(y, dtype=int)
        self.cont_idx = [i for i, c in enumerate(cols) if self.types.get(c) == "cont"]
        self.ord_idx = [i for i, c in enumerate(cols) if self.types.get(c) == "ord"]
        self.bin_idx = [i for i, c in enumerate(cols) if self.types.get(c) == "bin"]
        self.oh_idx = [i for i, c in enumerate(cols) if self.types.get(c) == "nom"]
        groups = [[cols.index(n) for n in g] for g in self.onehot_groups.values()]
        ord_levels = {i: np.unique(Xa[:, i]) for i in self.ord_idx}
        counts = np.bincount(y, minlength=self.K)
        n_max = counts.max()
        self.targets_ = {k: int(max(counts[k], math.ceil(self.r * n_max))) for k in range(self.K)}
        self.seed_types_, new_X, new_y, new_parent = {}, [], [], []

        for k in range(self.K):
            need = self.targets_[k] - counts[k]
            idx_k = np.where(y == k)[0]
            if need <= 0 or len(idx_k) < 2 or self.mode == "none":
                continue
            stds = Xa[idx_k][:, self.cont_idx].std(0) if self.cont_idx else np.array([1.0])
            med = float(np.median(stds)) or 1.0
            E = self._embedding(Xa, med)
            m = min(self.m, len(y) - 1)
            nn_all = NearestNeighbors(n_neighbors=m + 1).fit(E)
            nb = nn_all.kneighbors(E[idx_k], return_distance=False)[:, 1:]
            dist_cls = np.abs(y[nb] - k)
            a_i, b_i = (dist_cls == 1).sum(1), (dist_cls >= 2).sum(1)
            noise = b_i > m / 2
            border = ~noise & (a_i >= m / 2) & (a_i < m)
            safe = ~noise & ~border
            self.seed_types_[k] = {"ruido": int(noise.sum()), "frontera": int(border.sum()), "seguro": int(safe.sum())}
            seeds = idx_k if self.mode == "classic" else (idx_k[border] if border.any() else idx_k[safe])
            if seeds.size == 0:
                continue
            ks = min(self.k_s, len(idx_k) - 1)
            nn_k = NearestNeighbors(n_neighbors=ks + 1).fit(E[idx_k])
            nb_k = idx_k[nn_k.kneighbors(E[seeds], return_distance=False)[:, 1:]]   # (n_seeds, ks)
            accepted, attempts = 0, 0
            while accepted < need and attempts < need * self.max_attempts_factor:
                batch = min(max(2 * (need - accepted), 16), 20000)
                attempts += batch
                s_pos = rng.integers(0, seeds.size, batch)
                x_i = Xa[seeds[s_pos]]
                x_z = Xa[nb_k[s_pos, rng.integers(0, ks, batch)]]
                lam = rng.random((batch, 1))
                x_new = x_i.copy()
                num = self.cont_idx + self.ord_idx
                x_new[:, num] = x_i[:, num] + lam * (x_z[:, num] - x_i[:, num])
                for i in self.ord_idx:                      # interpolar y redondear al nivel próximo
                    lv = ord_levels[i]
                    x_new[:, i] = lv[np.abs(x_new[:, i][:, None] - lv[None, :]).argmin(1)]
                neigh = Xa[nb_k[s_pos]]                     # (batch, ks, p): moda entre los k_s vecinos
                if self.bin_idx:
                    c = neigh[:, :, self.bin_idx].sum(1) + 0.5 * x_i[:, self.bin_idx]
                    x_new[:, self.bin_idx] = (c > ks / 2).astype(float)
                for g in groups:
                    score = neigh[:, :, g].sum(1) + 0.5 * x_i[:, g]
                    x_new[:, g] = np.eye(len(g))[score.argmax(1)]
                # filtro de coherencia ordinal: vecino real más cercano en k o adyacente
                nearest = nn_all.kneighbors(self._embedding(x_new, med), n_neighbors=1, return_distance=False)[:, 0]
                ok = np.abs(y[nearest] - k) <= 1 if self.mode == "oa" else np.ones(batch, bool)
                take = np.where(ok)[0][: need - accepted]
                new_X.append(x_new[take]); new_y.append(np.full(take.size, k)); new_parent.append(seeds[s_pos[take]])
                accepted += take.size
            self.seed_types_[k]["sinteticos"] = int(accepted)

        X_out = np.vstack([Xa] + new_X) if new_X else Xa
        y_out = np.concatenate([y] + new_y) if new_y else y
        parent = np.concatenate([np.arange(len(y))] + new_parent) if new_parent else np.arange(len(y))
        is_syn = np.r_[np.zeros(len(y), bool), np.ones(len(y_out) - len(y), bool)]
        self.class_weights_ = class_weights(y_out, self.K, self.beta)
        weights = self.class_weights_[y_out]
        self.counts_after_ = np.bincount(y_out, minlength=self.K).tolist()
        return pd.DataFrame(X_out, columns=cols), y_out, weights, is_syn, parent

    def summary(self) -> dict:
        return {"targets": self.targets_, "counts_after": self.counts_after_,
                "class_weights": [float(w) for w in self.class_weights_], "seed_types": self.seed_types_}
