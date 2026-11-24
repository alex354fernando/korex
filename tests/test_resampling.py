"""C4 · OA-SMOTE-NC: balance parcial, filtro de coherencia ordinal, tipos válidos y pesos."""
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors

from korex.resampling import OASMOTENC, class_weights, ordinal_cost_matrix


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n = [400, 120, 60, 30]
    y = np.repeat(np.arange(4), n)
    x1 = rng.normal(y * 1.5, 0.8)
    x2 = rng.normal(-y, 1.0)
    ordv = np.clip(np.round(y + rng.normal(0, 0.7, len(y))), 0, 4)
    binv = (rng.random(len(y)) < 0.2 + 0.2 * y).astype(float)
    cat = rng.integers(0, 3, len(y))
    X = pd.DataFrame({"x1": x1, "x2": x2, "ord": ordv, "bin": binv,
                      "c=a": (cat == 0) * 1.0, "c=b": (cat == 1) * 1.0, "c=c": (cat == 2) * 1.0})
    types = {"x1": "cont", "x2": "cont", "ord": "ord", "bin": "bin", "c=a": "nom", "c=b": "nom", "c=c": "nom"}
    return X, y, types, {"c": ["c=a", "c=b", "c=c"]}


def test_partial_balance_targets(toy):
    X, y, types, groups = toy
    r = 0.5
    res = OASMOTENC(types, groups, r=r, m=8, k_s=5, beta=0.5, seed=1)
    Xa, ya, w, syn, parent = res.fit_resample(X, y)
    counts = np.bincount(ya, minlength=4)
    n_max = np.bincount(y).max()
    assert counts[0] == 400                                         # mayoritaria intacta
    for k in range(1, 4):
        assert counts[k] <= max(np.sum(y == k), math.ceil(r * n_max))
        assert counts[k] >= np.sum(y == k)
    assert syn.sum() == len(ya) - len(y)
    assert np.all(y[parent[syn]] == ya[syn])                        # semilla de la misma clase


def test_ordinal_coherence_filter(toy):
    X, y, types, groups = toy
    res = OASMOTENC(types, groups, r=0.8, m=8, k_s=5, seed=2)
    Xa, ya, w, syn, _ = res.fit_resample(X, y)
    Xr, Xs = X.to_numpy(), Xa.to_numpy()[syn]
    for k in np.unique(ya[syn]):
        med = float(np.median(Xr[y == k][:, :2].std(0)))
        E = lambda A: np.c_[A[:, :3], A[:, 3:4] * med, A[:, 4:] * med / math.sqrt(2)]
        nn = NearestNeighbors(n_neighbors=1).fit(E(Xr))
        sel = ya[syn] == k
        nearest = nn.kneighbors(E(Xs[sel]), return_distance=False)[:, 0]
        assert np.all(np.abs(y[nearest] - k) <= 1)


def test_synthetic_types_valid(toy):
    X, y, types, groups = toy
    Xa, ya, *_ = OASMOTENC(types, groups, r=1.0, seed=3).fit_resample(X, y)
    assert set(np.unique(Xa["bin"])) <= {0.0, 1.0}
    assert set(np.unique(Xa["ord"])) <= set(np.unique(X["ord"]))
    assert np.allclose(Xa[groups["c"]].sum(axis=1), 1.0)


def test_class_weights_and_cost_matrix():
    y = np.array([0] * 6 + [1] * 2 + [2] * 1 + [3] * 1)
    w = class_weights(y, 4, beta=1.0)
    assert w[0] == pytest.approx(10 / (4 * 6)) and w[3] == pytest.approx(10 / 4)
    assert np.allclose(class_weights(y, 4, beta=0.0), 1.0)
    C = ordinal_cost_matrix(4, p=2, delta=1.0)
    assert C[0, 0] == 0 and C[0, 3] == 9 and C[3, 0] == 18          # subestimar cuesta el doble
    assert np.allclose(ordinal_cost_matrix(4, 1, 0.0), np.abs(np.subtract.outer(range(4), range(4))))
