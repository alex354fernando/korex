"""C6 · Optimización bayesiana (Optuna/TPE + poda) con validación anidada:
externa 5×5 estratificada (solo estimación) e interna para la búsqueda."""
from __future__ import annotations

import logging
import time
from contextlib import nullcontext

import numpy as np
import optuna
from joblib import Parallel, cpu_count, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from .labeling import LabeledData, fit_rule_tree, prior_baseline, rule_tree_proba, scenario_features
from .metrics import evaluate, ordinal_objective, summarize_folds
from .ensemble import min_cost_decision
from .pipeline import KorexModel, apply_caps, default_params, prepare_fold

log = logging.getLogger("korex.tuning")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_params(trial: optuna.Trial, cfg: dict, quick: bool = False) -> dict:
    """Muestrea Θ (config.yaml → tuning.search_space)."""
    params = default_params(cfg)
    for group, space in cfg["tuning"]["search_space"].items():
        for name, (lo, hi, kind) in space.items():
            key = f"{group}.{name}"
            if kind == "int":
                v = trial.suggest_int(key, int(lo), int(hi))
            elif kind == "log":
                v = trial.suggest_float(key, float(lo), float(hi), log=True)
            else:
                v = trial.suggest_float(key, float(lo), float(hi))
            params.setdefault(group, {})[name] = v
    return apply_caps(params, cfg) if quick else params


def n_workers(cfg: dict) -> int:
    """Procesos paralelos (un pliegue por núcleo físico; config project.parallel_workers)."""
    w = cfg["project"].get("parallel_workers", -1)
    return max(1, cpu_count(only_physical_cores=True) if w in (-1, None) else int(w))


def _progress(msg: str):
    """Mensajes de avance visibles también desde los procesos paralelos."""
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _fit_score(cfg, scenario, params, seed, stack_folds, X_tr, y_tr, X_va, y_va, prep, lam) -> float:
    m = KorexModel(cfg, scenario, params, seed, stack_folds).fit(X_tr, y_tr, prepared=prep)
    return ordinal_objective(y_va, m.predict(X_va), m.C, lam)


def tune(cfg: dict, data: LabeledData, idx: np.ndarray, scenario: str, n_trials: int, inner_folds: int,
         stack_folds: int, seed: int, quick: bool = False, n_jobs: int = 1, label: str = ""):
    """Búsqueda bayesiana sobre las filas idx (pliegue externo de entrenamiento).
    n_jobs > 1 evalúa los pliegues internos de cada trial en paralelo (mismo resultado)."""
    if n_trials <= 0:
        return default_params(cfg, quick), {"n_trials": 0}
    X, y = data.X.iloc[idx].reset_index(drop=True), data.y[idx]
    feats = scenario_features(X.columns, cfg, scenario)
    lam = cfg["tuning"]["objective_lambda"]
    splits = list(StratifiedKFold(max(2, inner_folds), shuffle=True, random_state=seed).split(X, y))
    cache = [prepare_fold(cfg, X.iloc[tr][feats], seed) for tr, _ in splits]   # C2/C3 no dependen de Θ

    pool = Parallel(n_jobs=min(n_jobs, len(splits))) if n_jobs > 1 else None
    stack_folds = cfg["tuning"].get("tune_stack_folds", stack_folds)   # la búsqueda solo compara ajustes
    callbacks = [lambda st_, tr_: _progress(f"   {label} prueba {tr_.number + 1}/{n_trials} lista")] if label else []

    def objective(trial):
        params = suggest_params(trial, cfg, quick)
        if pool is not None:                         # pliegues internos en paralelo
            scores = pool(delayed(_fit_score)(cfg, scenario, params, seed, stack_folds, X.iloc[tr], y[tr],
                                              X.iloc[va], y[va], prep, lam) for (tr, va), prep in zip(splits, cache))
            return float(np.mean(scores))
        scores = []
        for step, ((tr, va), prep) in enumerate(zip(splits, cache)):
            scores.append(_fit_score(cfg, scenario, params, seed, stack_folds, X.iloc[tr], y[tr], X.iloc[va], y[va], prep, lam))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0))
    with pool if pool is not None else nullcontext():
        study.optimize(objective, n_trials=n_trials, catch=(ValueError,), callbacks=callbacks)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        return default_params(cfg, quick), {"n_trials": n_trials, "completed": 0}
    best = default_params(cfg)
    for key, v in study.best_params.items():
        g, n = key.split(".", 1)
        best.setdefault(g, {})[n] = v
    best = apply_caps(best, cfg) if quick else best
    return best, {"n_trials": n_trials, "completed": len(done),
                  "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
                  "best_value": float(study.best_value)}


def outer_splits(cfg: dict, y: np.ndarray, profile: dict, seed: int):
    cv = RepeatedStratifiedKFold(n_splits=profile["outer_folds"], n_repeats=profile["outer_repeats"], random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def _outer_fold(cfg, data, scenario, profile, seed, quick, i, tr, te):
    """Un pliegue externo completo: búsqueda interna + ajuste + evaluación (se ejecuta en paralelo)."""
    t0 = time.time()
    _progress(f"   pliegue {i + 1}: preparando datos y buscando ajustes…")
    params, info = tune(cfg, data, tr, scenario, profile["n_trials"], profile["inner_folds"],
                        profile["stack_folds"], seed + i, quick, label=f"pliegue {i + 1} ·")
    met, rec = _fit_eval(cfg, data, scenario, params, seed, profile, tr, te)
    rec.update(fold=i, tuning=info)
    return i, met, rec, time.time() - t0


def _fit_eval(cfg, data, scenario, params, seed, profile, tr, te):
    """Ajusta en `tr`, evalúa en `te` y guarda también el desempeño de cada modelo base por separado
    (mismos pliegues → comparación estadística de Friedman/Nemenyi sin entrenar nada aparte)."""
    X_tr = data.X.iloc[tr].reset_index(drop=True)
    m = KorexModel(cfg, scenario, params, seed, profile["stack_folds"], fit_all=True).fit(X_tr, data.y[tr])
    Xt = m.transform(data.X.iloc[te])
    P = m.ens_.predict_proba(Xt)
    yhat = min_cost_decision(P, m.C)
    lam = cfg["tuning"]["objective_lambda"]
    met = evaluate(data.y[te], P, yhat, m.C, lam)
    base = {}
    for name, Pb in m.ens_.base_predictions(Xt).items():
        mb = evaluate(data.y[te], Pb, min_cost_decision(Pb, m.C), m.C, lam)
        base[name] = {k: v for k, v in mb.items() if isinstance(v, float)}
    rec = {"test_idx": te.tolist(), "proba": P.tolist(), "pred": yhat.tolist(), "params": params,
           "base_models": base, "active_models": m.ens_.active_models_}
    return met, rec


SCENARIO_LABEL = {"A": "Modelo KORE-X", "B": "Tamizaje sin creatinina"}


def nested_cv(cfg: dict, data: LabeledData, scenario: str, profile: dict, seed: int, quick: bool = False) -> dict:
    """Validación anidada: estima el desempeño de todo el procedimiento (C2–C6).
    Los pliegues externos se procesan en paralelo; el resultado no depende del orden."""
    splits = outer_splits(cfg, data.y, profile, seed)
    jobs = (delayed(_outer_fold)(cfg, data, scenario, profile, seed, quick, i, tr, te) for i, (tr, te) in enumerate(splits))
    results, done = {}, 0
    for i, met, rec, secs in Parallel(n_jobs=n_workers(cfg), return_as="generator_unordered")(jobs):
        done += 1
        results[i] = (met, rec)
        log.info("%s · pliegue %d/%d listo · F1m=%.3f AUC=%.3f (%.0fs)", SCENARIO_LABEL.get(scenario, scenario),
                 done, len(splits), met["f1_macro"], met["auc_ovr"], secs)
    folds = [results[i][0] for i in sorted(results)]
    records = [results[i][1] for i in sorted(results)]
    return {"folds": folds, "summary": summarize_folds(folds), "records": records}


def _fixed_fold(cfg, data, scenario, profile, seed, params, i, tr, te):
    t0 = time.time()
    met, rec = _fit_eval(cfg, data, scenario, params, seed, profile, tr, te)
    rec.update(fold=i, tuning={"n_trials": 0, "fixed_params": True})
    return i, met, rec, time.time() - t0


def fixed_cv(cfg: dict, data: LabeledData, scenario: str, profile: dict, seed: int, params: dict) -> dict:
    """Validación externa con hiperparámetros fijos (los del modelo principal), sin búsqueda interna.
    Se usa para el modelo de tamizaje: mismos pliegues que la validación anidada, ~10 veces más rápido."""
    splits = outer_splits(cfg, data.y, profile, seed)
    jobs = (delayed(_fixed_fold)(cfg, data, scenario, profile, seed, params, i, tr, te) for i, (tr, te) in enumerate(splits))
    results, done = {}, 0
    for i, met, rec, secs in Parallel(n_jobs=n_workers(cfg), return_as="generator_unordered")(jobs):
        done += 1
        results[i] = (met, rec)
        log.info("%s · pliegue %d/%d listo · F1m=%.3f ±1=%.3f AUC(<60)=%.3f (%.0fs)", SCENARIO_LABEL.get(scenario, scenario),
                 done, len(splits), met["f1_macro"], met["accuracy_within1"], met.get("auc_lt60", float("nan")), secs)
    folds = [results[i][0] for i in sorted(results)]
    return {"folds": folds, "summary": summarize_folds(folds), "records": [results[i][1] for i in sorted(results)]}


def baseline_cv(cfg: dict, data: LabeledData, profile: dict, seed: int) -> dict:
    """M_0 (a priori) y regla (árbol prof. 3 solo con la TFGe) en los mismos pliegues externos."""
    lam, K = cfg["tuning"]["objective_lambda"], len(cfg["kdigo"]["classes"])
    from .resampling import ordinal_cost_matrix
    C = ordinal_cost_matrix(K, cfg["cost"]["p"], cfg["cost"]["delta"])
    prior, rule = [], []
    for i, (tr, te) in enumerate(outer_splits(cfg, data.y, profile, seed)):
        P0, y0 = prior_baseline(data.y[tr], len(te), seed + i, K)
        prior.append(evaluate(data.y[te], P0, y0, C, lam))
        tree = fit_rule_tree(data.X.iloc[tr], data.y[tr], seed)
        Pr = rule_tree_proba(tree, data.X.iloc[te], K)
        rule.append(evaluate(data.y[te], Pr, Pr.argmax(1), C, lam))
    return {"prior": {"folds": prior, "summary": summarize_folds(prior)},
            "rule_tree": {"folds": rule, "summary": summarize_folds(rule)}}


def best_fold_params(records: list[dict]):
    """Hiperparámetros del pliegue externo con mejor valor en su búsqueda interna."""
    scored = [r for r in records if r.get("params") and r["tuning"].get("best_value") is not None]
    if not scored:
        return None, None
    best = max(scored, key=lambda r: r["tuning"]["best_value"])
    return best["params"], {"reused_from_fold": best["fold"], "best_value": best["tuning"]["best_value"]}


def fit_final(cfg: dict, data: LabeledData, scenario: str, profile: dict, seed: int, quick: bool = False,
              params: dict | None = None):
    """Ajuste del modelo que se despliega. Si se pasan `params` (p. ej., los mejores de la validación
    anidada) se omite la búsqueda final; si no, se busca con todos los datos."""
    idx = np.arange(len(data.y))
    if params is not None:
        info = {"reused_params": True}
    else:
        params, info = tune(cfg, data, idx, scenario, profile["final_trials"], profile["inner_folds"],
                            profile["stack_folds"], seed, quick, n_jobs=n_workers(cfg))
    model = KorexModel(cfg, scenario, params, seed, profile["stack_folds"]).fit(data.X, data.y)
    return model, params, info


ABLATION_MODES = (("none", "sin_balanceo"), ("classic", "smote_nc_clasico"), ("oa", "oa_smote_nc"))


def _ablation_fold(cfg, data, scenario, profile, seed, quick, tr, te):
    """Mismo pliegue y mismo preprocesamiento para las tres variantes de balanceo."""
    X_tr = data.X.iloc[tr].reset_index(drop=True)
    prep = prepare_fold(cfg, X_tr[scenario_features(X_tr.columns, cfg, scenario)], seed)
    out = {}
    for mode, label in ABLATION_MODES:
        params = default_params(cfg, quick)
        params["resampling_mode"] = mode
        m = KorexModel(cfg, scenario, params, seed, profile["stack_folds"]).fit(X_tr, data.y[tr], prepared=prep)
        out[label] = evaluate(data.y[te], m.predict_proba(data.X.iloc[te]), m.predict(data.X.iloc[te]), m.C,
                              cfg["tuning"]["objective_lambda"])
    return out


def ablation_cv(cfg: dict, data: LabeledData, scenario: str, profile: dict, seed: int, quick: bool = False) -> dict:
    """Estudio de ablación del balanceo (hiperparámetros por defecto, mismos pliegues externos)."""
    res = Parallel(n_jobs=n_workers(cfg))(delayed(_ablation_fold)(cfg, data, scenario, profile, seed, quick, tr, te)
                                          for tr, te in outer_splits(cfg, data.y, profile, seed))
    out = {label: summarize_folds([r[label] for r in res]) for _, label in ABLATION_MODES}
    for label, v in out.items():
        log.info("Ablación %s · F1m=%.3f", label, v["f1_macro"]["mean"])
    return out
