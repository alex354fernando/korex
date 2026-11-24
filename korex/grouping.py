"""C3 · Agrupación de variables correlacionadas: Spearman + UPGMA (δ = 1 − |ρ|)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


class FeatureGrouper:
    """Partición 𝒢 = {G_1..G_m} y representante por grupo (mayor |ρ| medio intra-grupo)."""

    def __init__(self, distance_cut: float = 0.3, always_keep=()):
        self.distance_cut, self.always_keep = distance_cut, list(always_keep)

    def fit(self, X: pd.DataFrame) -> "FeatureGrouper":
        cols = list(X.columns)
        self.columns_ = cols
        if len(cols) == 1:
            self.rho_ = pd.DataFrame([[1.0]], index=cols, columns=cols)
            self.groups_, self.representatives_, self.linkage_ = [cols], cols, None
            return self
        rho = spearmanr(X.to_numpy(), nan_policy="omit").statistic
        rho = np.nan_to_num(np.asarray(rho, dtype=float), nan=0.0)
        if rho.ndim == 0:                                   # con 2 variables scipy devuelve un escalar
            rho = np.array([[1.0, float(rho)], [float(rho), 1.0]])
        np.fill_diagonal(rho, 1.0)
        self.rho_ = pd.DataFrame(rho, index=cols, columns=cols)
        dist = np.clip(1.0 - np.abs(rho), 0.0, 1.0)
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0.0)
        self.linkage_ = linkage(squareform(dist, checks=False), method="average")
        labels = fcluster(self.linkage_, t=self.distance_cut, criterion="distance")
        self.groups_, self.representatives_ = [], []
        for g in np.unique(labels):
            members = [cols[i] for i in np.where(labels == g)[0]]
            self.groups_.append(members)
            if len(members) == 1:
                self.representatives_.append(members[0])
            else:
                sub = np.abs(self.rho_.loc[members, members].to_numpy())
                mean_corr = (sub.sum(1) - 1.0) / (len(members) - 1)
                self.representatives_.append(members[int(np.argmax(mean_corr))])
        # variables clínicas núcleo: siempre disponibles para los modelos lineales
        self.representatives_ += [c for c in self.always_keep if c in cols and c not in self.representatives_]
        return self

    def group_of(self) -> dict:
        """Mapa variable → índice de grupo (para explicaciones agrupadas en C8)."""
        return {c: i for i, g in enumerate(self.groups_) for c in g}

    def summary(self) -> dict:
        return {"n_features": len(self.columns_), "n_groups": len(self.groups_),
                "groups": [g for g in self.groups_ if len(g) > 1],
                "representatives": self.representatives_}
