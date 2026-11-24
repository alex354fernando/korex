"""Entrena, evalúa (validación anidada) y guarda los artefactos de KORE-X.

Uso:
    python scripts/train.py --profile quick            # prueba rápida (minutos)
    python scripts/train.py --profile standard         # recomendado para la tesis
    python scripts/train.py --profile full             # 5×5 anidada completa (horas)
    python scripts/train.py --synthetic --profile quick  # sin dataset real (demo)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from korex import __version__, load_config, resolve_path, set_seed  # noqa: E402
from korex.ensemble import min_cost_decision  # noqa: E402
from korex.kdigo import load_dataset, write_synthetic  # noqa: E402
from korex.labeling import circularity_table, label_dataset  # noqa: E402
from korex.metrics import evaluate  # noqa: E402
from korex.pipeline import InferencePipeline  # noqa: E402
from korex.resampling import ordinal_cost_matrix  # noqa: E402
from korex.explain import global_shap  # noqa: E402
from korex.stats import fold_table, friedman_nemenyi  # noqa: E402
from korex.tuning import ablation_cv, baseline_cv, best_fold_params, fit_final, fixed_cv, nested_cv, n_workers  # noqa: E402

log = logging.getLogger("korex.train")


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrenamiento KORE-X")
    ap.add_argument("--data", help="ruta al CSV/Excel (por defecto paths.data de config.yaml)")
    ap.add_argument("--synthetic", action="store_true", help="usar/generar dataset sintético de demostración")
    ap.add_argument("--profile", default="standard", choices=["quick", "standard", "full"])
    ap.add_argument("--circularidad", action="store_true",
                    help="validación anidada completa también para el modelo sin creatinina e índice IC_M (tesis)")
    ap.add_argument("--tesis", action="store_true",
                    help="todo lo necesario para la tesis: --circularidad --ablation --explain-global")
    ap.add_argument("--no-eval", action="store_true", help="omitir la validación anidada (solo modelo final)")
    ap.add_argument("--explain-global", action="store_true", help="importancia global SHAP y convergencia por clase")
    ap.add_argument("--ablation", action="store_true",
                    help="comparar sin balanceo / SMOTE-NC clásico / OA-SMOTE-NC (validación externa, sin Optuna)")
    ap.add_argument("--final-tuning", action="store_true",
                    help="repetir la búsqueda bayesiana con todos los datos para el modelo final (más lento)")
    ap.add_argument("--n-synthetic", type=int, default=400)
    args = ap.parse_args()
    if args.tesis:
        args.circularidad = args.ablation = args.explain_global = True
    args.scenarios = ["A", "B"]                    # B = modo tamizaje sin creatinina (siempre disponible)
    os.environ.setdefault("PYTHONWARNINGS", "ignore")          # también para los procesos paralelos
    warnings.filterwarnings("ignore")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config()
    seed = cfg["project"]["seed"]
    set_seed(seed)
    profile = cfg["tuning"]["profiles"][args.profile]
    quick = args.profile == "quick"

    path = Path(args.data) if args.data else resolve_path(cfg["paths"]["data"])
    if not args.synthetic and not path.exists() and resolve_path(cfg["paths"]["raw_data"]).exists():
        path = resolve_path(cfg["paths"]["raw_data"])          # ARFF original de UCI
    if args.synthetic or not path.exists():
        if not args.synthetic:
            log.warning("No se encontró %s → se usará el dataset SINTÉTICO de demostración.", path)
        path = resolve_path(cfg["paths"]["synthetic_data"])
        if not path.exists():
            write_synthetic(path, cfg, args.n_synthetic, seed)
    log.info("Cargando %s", path)
    data = label_dataset(load_dataset(path, cfg), cfg)
    log.info("D: n=%d · clases=%s · IR=%s", data.info["n"], data.info["class_counts"],
             {k: round(v, 1) for k, v in data.info["imbalance_ratio"].items()})

    log.info("Procesos en paralelo: %d", n_workers(cfg))
    t0 = time.time()
    evaluation, records = {}, {}
    if not args.no_eval:
        base = baseline_cv(cfg, data, profile, seed)
        evaluation["prior"], evaluation["rule_tree"] = base["prior"]["summary"], base["rule_tree"]["summary"]
        records["A"] = nested_cv(cfg, data, "A", profile, seed, quick)
        evaluation["A"] = records["A"]["summary"]
        main_params, _ = best_fold_params(records["A"]["records"])
        if args.circularidad:                       # sin creatinina con su propia búsqueda (estimación rigurosa)
            records["B"] = nested_cv(cfg, data, "B", profile, seed, quick)
        else:                                       # mismos pliegues y ajustes del modelo principal (rápido)
            log.info("Tamizaje sin creatinina · validación externa con los ajustes del modelo principal")
            records["B"] = fixed_cv(cfg, data, "B", profile, seed, main_params)
        evaluation["B"] = records["B"]["summary"]
        evaluation["B_protocol"] = "anidada" if args.circularidad else "ajustes_del_modelo_principal"
        for metric in ("f1_macro", "auc_ovr"):      # Friedman/Nemenyi: KORE-X frente a cada modelo base
            evaluation.setdefault("comparison", {})[metric] = friedman_nemenyi(
                fold_table(records["A"]["records"], records["A"]["folds"], metric))
        cmp_ = evaluation["comparison"]["f1_macro"]
        if cmp_:
            log.info("Friedman (F1 macro): p=%.4f · CD Nemenyi=%.2f · rangos %s", cmp_["p_value"], cmp_["cd"],
                     {k: round(v, 2) for k, v in cmp_["mean_rank"].items()})
        if True:                                    # IC_M (con --circularidad, B tiene su propia búsqueda)
            mean = lambda summ: {k: v["mean"] for k, v in summ.items()}
            evaluation["circularity"] = circularity_table(mean(evaluation["A"]), mean(evaluation["B"]), mean(evaluation["prior"]))
            per_fold = [circularity_table(a, b, p) for a, b, p in
                        zip(records["A"]["folds"], records["B"]["folds"], base["prior"]["folds"])]
            evaluation["circularity_per_fold"] = per_fold
            log.info("Índice de circularidad IC_M: %s", {k: round(v, 3) for k, v in evaluation["circularity"].items()})

    if args.ablation:
        evaluation["ablation"] = ablation_cv(cfg, data, "A", profile, seed, quick)
        log.info("Ablación (F1 macro): %s", {k: round(v["f1_macro"]["mean"], 3) for k, v in evaluation["ablation"].items()})

    # rangos realistas para el botón "Aleatorizar" del dashboard (95 % central de cada variable)
    lo_q, hi_q = cfg["random_quantiles"]
    ranges = {}
    for c in data.X.columns:
        col = data.X[c].dropna()
        if col.empty:
            continue
        if col.dtype == object or str(col.dtype).startswith("str") or col.nunique() <= 6:
            ranges[c] = {"type": "choice", "values": sorted(col.unique().tolist(), key=str)}
        else:
            ranges[c] = {"type": "range", "min": float(col.quantile(lo_q)), "max": float(col.quantile(hi_q))}
    # "paciente típico" por estadio: rango intercuartílico (numéricas) y frecuencias (categóricas)
    stage_ranges = {}
    for k in cfg["kdigo"]["classes"]:
        sub, stage_ranges[str(k)] = data.X[data.y == k], {}
        for c, r in ranges.items():
            col = sub[c].dropna()
            if col.empty:
                continue
            if r["type"] == "choice":
                vc = col.value_counts(normalize=True)
                stage_ranges[str(k)][c] = {"type": "choice", "values": vc.index.tolist(), "p": vc.values.round(4).tolist()}
            else:
                stage_ranges[str(k)][c] = {"type": "range", "min": float(col.quantile(.25)), "max": float(col.quantile(.75))}

    models, backgrounds, scen_meta = {}, {}, {}
    lam = cfg["tuning"]["objective_lambda"]
    rng = np.random.default_rng(seed)
    global_expl = {}
    for s in args.scenarios:
        log.info("Ajuste final · %s", {"A": "Modelo KORE-X", "B": "Tamizaje sin creatinina"}[s])
        src = s if (s == "A" or args.circularidad) else "A"          # el tamizaje rápido usa los ajustes de A
        reuse, reuse_info = (None, None) if args.final_tuning or src not in records else best_fold_params(records[src]["records"])
        if reuse_info and src != s:
            reuse_info["params_from"] = "A"
        model, params, info = fit_final(cfg, data, s, profile, seed, quick, params=reuse)
        if reuse_info:
            info.update(reuse_info)
            log.info("Modelo final con los ajustes del pliegue %d (sin repetir la búsqueda)", reuse_info["reused_from_fold"] + 1)
        bg_idx = rng.choice(len(data.y), min(cfg["explain"]["background_size"], len(data.y)), replace=False)
        backgrounds[s] = model.transform(data.X.iloc[bg_idx]).reset_index(drop=True)
        ens = model.ens_
        sens = {}
        for d in [cfg["cost"]["delta"]] + list(cfg["cost"]["delta_sensitivity"]):
            C = ordinal_cost_matrix(ens.K, cfg["cost"]["p"], d)
            sens[str(d)] = evaluate(ens.oof_y_, ens.oof_proba_, min_cost_decision(ens.oof_proba_, C), C, lam)
        scen_meta[s] = {"name": cfg["scenarios"][s]["name"], "best_params": params, "final_tuning": info,
                        "oof_metrics_final": sens[str(cfg["cost"]["delta"])],
                        "delta_sensitivity": sens, **model.summary()}
        t_ex = time.time()
        global_expl[s] = global_shap(model, backgrounds[s], cfg, data.X, cfg["explain"]["global_samples"], seed,
                                     n_workers(cfg))
        scen_meta[s]["global_importance"] = global_expl[s]["importance"]
        scen_meta[s]["global_share"] = global_expl[s]["share"]
        log.info("SHAP global · %s: %s (%.0f s)", s, {k: f"{100 * v:.1f} %" for k, v in list(global_expl[s]["share"].items())[:5]},
                 time.time() - t_ex)
        if args.explain_global:
            pipe_tmp = InferencePipeline({s: model}, {s: backgrounds[s]}, {}, cfg)
            sample = model.transform(data.X.iloc[rng.choice(len(data.y), min(200, len(data.y)), replace=False)])
            scen_meta[s]["convergence_by_class"] = pipe_tmp.explainer(s).convergence_by_class(sample, ens.predict(sample))
        models[s] = model

    metadata = {
        "name": cfg["project"]["name"], "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(), "seed": seed, "profile": args.profile,
        "profile_config": profile, "kdigo_guideline": cfg["kdigo"]["guideline"], "equation": cfg["kdigo"]["equation"],
        "dataset": {"path": str(path.name), "synthetic": "synthetic" in path.name, **data.info},
        "cost": cfg["cost"], "scenarios": scen_meta, "evaluation": evaluation, "random_ranges": ranges, "stage_ranges": stage_ranges,
        "training_seconds": round(time.time() - t0, 1),
    }
    pipe = InferencePipeline(models, backgrounds, metadata, cfg)
    out = pipe.save()
    if records:
        joblib.dump({"records": records, "y": data.y, "egfr": data.egfr}, out / "evaluation.joblib", compress=3)
    joblib.dump({"y": data.y, **global_expl}, out / "global_shap.joblib", compress=3)
    log.info("Artefactos guardados en %s (%.0f s)", out, time.time() - t0)


if __name__ == "__main__":
    main()
