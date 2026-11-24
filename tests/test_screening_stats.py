import numpy as np

from korex.metrics import evaluate, pr_curves, screening_metrics
from korex.stats import friedman_nemenyi


def test_screening_metrics_perfect():
    y = np.array([0, 1, 2, 3, 4, 2, 3])
    P = np.eye(5)[y]
    m = screening_metrics(y, P, y)
    assert m["accuracy_within1"] == 1.0
    assert m["auc_lt60"] == 1.0 and m["sens_lt60"] == 1.0 and m["spec_lt60"] == 1.0
    assert m["auc_lt30"] == 1.0


def test_within_one_counts_adjacent_stage():
    y = np.array([0, 1, 2, 3])
    yhat = np.array([1, 1, 4, 3])                       # 2 → 4 es un error de dos estadios
    assert screening_metrics(y, np.eye(5)[yhat], yhat)["accuracy_within1"] == 0.75


def test_evaluate_includes_screening_and_pr():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(5), 10)
    P = rng.dirichlet(np.ones(5), len(y))
    assert "auc_lt60" in evaluate(y, P)
    assert set(pr_curves(y, P)) == set(range(5))


def test_friedman_detects_consistent_winner():
    table = {"a": [.9, .91, .92, .9, .93], "b": [.8, .81, .79, .8, .82], "c": [.5, .52, .49, .51, .5]}
    r = friedman_nemenyi(table)
    assert r["best"] == "a" and r["mean_rank"]["a"] == 1.0 and r["p_value"] < 0.05 and r["cd"] > 0
