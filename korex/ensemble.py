"""C5 · Stacking ordinal heterogéneo: 6 modelos base → logits acumulados OOF →
meta-aprendiz de odds proporcionales (L2) → calibración → decisión de mínimo costo."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from .metrics import ordinal_objective
from .resampling import class_weights

BASE_MODELS = ["xgb", "lgbm", "cat", "et", "lr", "svm", "lr_core"]
LINEAR_MODELS = {"lr", "svm"}                     # usan solo los representantes de C3
CORE_MODELS = {"lr_core"}                         # modelo lineal sobre el núcleo clínico (creatinina + edad)


def make_base(name: str, params: dict, seed: int, n_jobs: int):
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="multi:softprob", tree_method="hist", random_state=seed,
                             n_jobs=n_jobs, verbosity=0, **params)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(objective="multiclass", random_state=seed, n_jobs=n_jobs, verbose=-1, **params)
    if name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(loss_function="MultiClass", random_seed=seed, verbose=0,
                                  allow_writing_files=False, thread_count=n_jobs, **params)
    if name == "et":
        return ExtraTreesClassifier(random_state=seed, n_jobs=n_jobs, **params)
    if name in ("lr", "lr_core"):
        return LogisticRegression(max_iter=5000, **params)       # multinomial, L2
    if name == "svm":                                             # escalado de Platt (sigmoide)
        return CalibratedClassifierCV(SVC(kernel="rbf", random_state=seed, **params), method="sigmoid",
                                      cv=3, ensemble=False)
    raise ValueError(name)


class BaseModel:
    """Envoltorio: subconjunto de columnas, pesos y salida siempre con K columnas."""

    def __init__(self, name, params, cols, K, seed, n_jobs, max_samples=None):
        self.name, self.params, self.cols, self.K = name, params, cols, K
        self.seed, self.n_jobs, self.max_samples = seed, n_jobs, max_samples

    def fit(self, X: pd.DataFrame, y, w):
        Xs, y, w = X[self.cols].to_numpy(), np.asarray(y), np.asarray(w)
        if self.max_samples and len(y) > self.max_samples:      # submuestreo estratificado (SVM)
            rng = np.random.default_rng(self.seed)
            idx = np.concatenate([rng.choice(np.where(y == k)[0], max(1, int(round(self.max_samples * np.mean(y == k)))),
                                             replace=False) for k in np.unique(y)])
            Xs, y, w = Xs[idx], y[idx], w[idx]
        self.classes_ = np.unique(y)
        if self.classes_.size == 1:
            self.model_ = None
            return self
        y_enc = np.searchsorted(self.classes_, y)
        self.model_ = make_base(self.name, self.params, self.seed, self.n_jobs).fit(Xs, y_enc, sample_weight=w)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        P = np.zeros((len(X), self.K))
        if self.model_ is None:
            P[:, self.classes_[0]] = 1.0
        else:
            P[:, self.classes_] = self.model_.predict_proba(X[self.cols].to_numpy())
        return P


def cumulative_logits(P: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """q_k = Σ_{c>k} p_c, k = 0..K−2, en escala logit."""
    q = np.clip(np.cumsum(P[:, ::-1], axis=1)[:, ::-1][:, 1:], eps, 1 - eps)
    return np.log(q / (1 - q))


def min_cost_decision(P: np.ndarray, C: np.ndarray) -> np.ndarray:
    """ŷ = argmin_j Σ_k p_k C_kj."""
    return np.argmin(P @ C, axis=1)


def yule_q(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    n11 = np.sum(correct_a & correct_b); n00 = np.sum(~correct_a & ~correct_b)
    n10 = np.sum(correct_a & ~correct_b); n01 = np.sum(~correct_a & correct_b)
    den = n11 * n00 + n01 * n10
    return float((n11 * n00 - n01 * n10) / den) if den else 1.0


class ProportionalOddsLR:
    """P(y ≤ k | z) = σ(ϑ_k − γᵀz), ϑ ordenados, penalización α‖γ‖², escala de temperatura T."""

    def __init__(self, alpha: float = 1.0, K: int = 4):
        self.alpha, self.K, self.T = alpha, K, 1.0

    def _theta(self, v):
        return np.cumsum(np.r_[v[0], np.exp(v[1:self.K - 1])])

    def _eta(self, Z):
        Zs = (Z - self.mu_) / self.sd_
        return self.theta_[None, :] - (Zs @ self.gamma_)[:, None]

    def _probs_from_eta(self, eta):
        F = np.c_[np.zeros(len(eta)), expit(eta), np.ones(len(eta))]
        return np.clip(np.diff(F, axis=1), 1e-12, 1.0)

    def fit(self, Z: np.ndarray, y: np.ndarray, sample_weight=None):
        K, y = self.K, np.asarray(y, int)
        self.mu_, self.sd_ = Z.mean(0), Z.std(0) + 1e-9
        Zs = (Z - self.mu_) / self.sd_
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        w = w / w.mean()
        cum = np.clip(np.cumsum(np.bincount(y, minlength=K))[:-1] / len(y), 0.02, 0.98)
        th0 = np.log(cum / (1 - cum))
        th0 = np.maximum.accumulate(th0 + np.arange(K - 1) * 1e-3)
        v0 = np.r_[th0[0], np.log(np.maximum(np.diff(th0), 1e-3)), np.zeros(Zs.shape[1])]
        rows = np.arange(len(y))

        def loss_grad(v):
            theta, gamma = self._theta(v), v[K - 1:]
            eta = theta[None, :] - (Zs @ gamma)[:, None]
            F = expit(eta)
            Fe = np.c_[np.zeros(len(y)), F, np.ones(len(y))]
            f = np.c_[np.zeros(len(y)), F * (1 - F), np.zeros(len(y))]
            p = np.clip(Fe[rows, y + 1] - Fe[rows, y], 1e-12, None)
            loss = -np.sum(w * np.log(p)) + self.alpha * gamma @ gamma
            G = np.zeros((len(y), K + 1))
            G[rows, y + 1] += f[rows, y + 1] / p
            G[rows, y] -= f[rows, y] / p
            g_theta = -(w[:, None] * G[:, 1:K]).sum(0)
            g_gamma = (w * (f[rows, y + 1] - f[rows, y]) / p) @ Zs + 2 * self.alpha * gamma
            ed = np.exp(v[1:K - 1])
            g_d = np.array([ed[i] * g_theta[i + 1:].sum() for i in range(K - 2)])
            return loss, np.r_[g_theta.sum(), g_d, g_gamma]

        res = minimize(loss_grad, v0, jac=True, method="L-BFGS-B", options={"maxiter": 200, "gtol": 1e-5})
        self.theta_, self.gamma_ = self._theta(res.x), res.x[K - 1:]
        return self

    def decision_eta(self, Z):
        return self._eta(Z)

    def fit_temperature(self, eta: np.ndarray, y: np.ndarray):
        """Calibración por temperatura sobre η fuera de muestra."""
        rows = np.arange(len(y))
        nll = lambda lt: -np.mean(np.log(self._probs_from_eta(eta / np.exp(lt))[rows, y]))
        self.T = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        P = self._probs_from_eta(self._eta(Z) / self.T)
        return P / P.sum(1, keepdims=True)


class OrdinalStackingEnsemble:
    def __init__(self, params: dict, lin_cols: list[str], C: np.ndarray, stack_folds=5, q_threshold=0.9,
                 meta_alpha=1.0, calibration="temperature", seed=42, n_jobs=-1, svm_max_samples=None,
                 lam=0.5, meta_beta=0.5, models=None, core_cols=None, greedy=False, fit_all=False):
        self.greedy, self.fit_all = greedy, fit_all
        self.params, self.lin_cols, self.C = params, lin_cols, C
        self.stack_folds, self.q_threshold, self.meta_alpha = stack_folds, q_threshold, meta_alpha
        self.calibration, self.seed, self.n_jobs = calibration, seed, n_jobs
        self.svm_max_samples, self.lam, self.meta_beta = svm_max_samples, lam, meta_beta
        self.core_cols = [c for c in (core_cols or []) if c in lin_cols]
        self.models = [m for m in (models or BASE_MODELS) if m not in CORE_MODELS or self.core_cols]
        self.K = C.shape[0]

    def _new_base(self, name, cols):
        use = self.core_cols if name in CORE_MODELS else (self.lin_cols if name in LINEAR_MODELS else cols)
        return BaseModel(name, self.params.get(name, {}), use,
                         self.K, self.seed, self.n_jobs, self.svm_max_samples if name == "svm" else None)

    def _z(self, P_by_model: dict, names) -> np.ndarray:
        return np.hstack([cumulative_logits(P_by_model[n]) for n in names])

    def _meta_cv(self, Z, y, w, folds: int = 5):
        """Predicciones del meta-aprendiz fuera de muestra (para poda y calibración)."""
        eta = np.zeros((len(y), self.K - 1))
        n_splits = max(2, min(folds, np.bincount(y).min()))
        for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=self.seed).split(Z, y):
            m = ProportionalOddsLR(self.meta_alpha, self.K).fit(Z[tr], y[tr], w[tr])
            eta[te] = m.decision_eta(Z[te])
        return eta

    def _objective(self, Z, y, w) -> float:
        eta = self._meta_cv(Z, y, w, folds=3)          # selección de modelos: 3 pliegues bastan
        P = ProportionalOddsLR(self.meta_alpha, self.K)._probs_from_eta(eta)
        return ordinal_objective(y, min_cost_decision(P / P.sum(1, keepdims=True), self.C), self.C, self.lam)

    def fit(self, X: pd.DataFrame, y, w, is_syn, parent):
        y, w, is_syn, parent = map(np.asarray, (y, w, is_syn, parent))
        self.columns_ = list(X.columns)
        real = np.where(~is_syn)[0]
        y_real = y[real]
        fold_real = np.zeros(len(real), int)
        n_splits = max(2, min(self.stack_folds, np.bincount(y_real).min()))
        for f, (_, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=self.seed).split(real, y_real)):
            fold_real[te] = f
        fold_all = fold_real[parent]                # cada sintético va al pliegue de su semilla real
        oof = {n: np.zeros((len(real), self.K)) for n in self.models}
        for f in range(n_splits):
            tr, te = np.where(fold_all != f)[0], real[fold_real == f]
            for n in self.models:
                m = self._new_base(n, self.columns_).fit(X.iloc[tr], y[tr], w[tr])
                oof[n][fold_real == f] = m.predict_proba(X.iloc[te])
        self.oof_base_ = oof
        w_meta = class_weights(y_real, self.K, self.meta_beta)[y_real]
        # poda por diversidad (Q de Yule)
        correct = {n: oof[n].argmax(1) == y_real for n in self.models}
        self.yule_q_ = {a: {b: yule_q(correct[a], correct[b]) for b in self.models if b != a} for a in self.models}
        active = list(self.models)
        self.pruned_ = []
        mean_q = {a: np.mean(list(v.values())) for a, v in self.yule_q_.items()}
        for cand in sorted(active, key=lambda a: -mean_q[a]):
            if mean_q[cand] <= self.q_threshold or len(active) <= 2:
                continue
            rest = [n for n in active if n != cand]
            if self._objective(self._z(oof, rest), y_real, w_meta) >= self._objective(self._z(oof, active), y_real, w_meta):
                active = rest
                self.pruned_.append(cand)
        if self.greedy:                            # selección hacia adelante (Caruana et al., 2004) por el objetivo J
            score = lambda subset: self._objective(self._z(oof, subset), y_real, w_meta)
            singles = {c: score([c]) for c in active}
            chosen = [max(singles, key=singles.get)]
            current = singles[chosen[0]]
            while True:
                rest = [c for c in active if c not in chosen]
                if not rest:
                    break
                trials = {c: score(chosen + [c]) for c in rest}
                best = max(trials, key=trials.get)
                if trials[best] <= current + 1e-4:
                    break
                chosen.append(best)
                current = trials[best]
            self.pruned_ += [c for c in active if c not in chosen]
            active = [c for c in self.models if c in chosen]
        self.active_models_ = active
        Z = self._z(oof, active)
        self.meta_ = ProportionalOddsLR(self.meta_alpha, self.K).fit(Z, y_real, w_meta)
        eta_cv = self._meta_cv(Z, y_real, w_meta)
        if self.calibration == "temperature":
            self.meta_.fit_temperature(eta_cv, y_real)
        self.oof_proba_ = self.meta_._probs_from_eta(eta_cv / self.meta_.T)
        self.oof_proba_ /= self.oof_proba_.sum(1, keepdims=True)
        self.oof_y_ = y_real
        # fit_all: también se ajustan los modelos podados (solo para compararlos en la validación externa)
        self.base_ = {n: self._new_base(n, self.columns_).fit(X, y, w) for n in (self.models if self.fit_all else active)}
        return self

    def _as_frame(self, X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.atleast_2d(X), columns=self.columns_)

    def base_proba(self, X) -> dict:
        X = self._as_frame(X)
        return {n: m.predict_proba(X) for n, m in self.base_.items()}

    def base_predictions(self, X) -> dict:
        """Probabilidades de cada modelo base por separado (comparación de modelos)."""
        return self.base_proba(X)

    def predict_proba(self, X) -> np.ndarray:
        X = self._as_frame(X)
        P = {n: self.base_[n].predict_proba(X) for n in self.active_models_}
        return self.meta_.predict_proba(self._z(P, self.active_models_))

    def predict(self, X, C: np.ndarray | None = None) -> np.ndarray:
        return min_cost_decision(self.predict_proba(X), self.C if C is None else C)

    def summary(self) -> dict:
        return {"active_models": self.active_models_, "pruned": self.pruned_,
                "yule_q_mean": {a: float(np.mean(list(v.values()))) for a, v in self.yule_q_.items()},
                "meta_thresholds": self.meta_.theta_.tolist(), "temperature": self.meta_.T}
