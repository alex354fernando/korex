"""Genera las figuras del artículo a partir de los artefactos entrenados.

Uso:
    python scripts/figures.py            # figuras en artifacts/figures/
    python scripts/figures.py --shap     # además, importancia global SHAP y convergencia SHAP–LIME (lento)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402

from korex.pipeline import InferencePipeline, artifacts_dir, load_config  # noqa: E402
from korex.metrics import calibration_curve_data  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def fig_stage_sensitivity(meta, cfg, out):
    """Estadios G según el supuesto de sexo (el dataset UCI no lo registra)."""
    ds, labels = meta["dataset"], cfg["kdigo"]["stage_labels"]
    series = {"Peor caso (usado)": ds["class_counts"], "Si todos hombres": ds.get("class_counts_if_male"),
              "Si todas mujeres": ds.get("class_counts_if_female")}
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x, w = np.arange(len(labels)), 0.27
    for i, (name, c) in enumerate((k, v) for k, v in series.items() if v):
        ax.bar(x + (i - 1) * w, [c[str(k)] for k in range(len(labels))], w, label=name)
    ax.set_xticks(x, labels); ax.set_ylabel("Pacientes"); ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Estadio KDIGO por TFGe (discordancia H/M: {100 * ds.get('sex_discordance_rate', 0):.0f} %)")
    fig.savefig(out / "fig1_stage_sex_sensitivity.png"); plt.close(fig)


def fig_class_distribution(meta, cfg, out):
    K = len(cfg["kdigo"]["classes"])
    before = [meta["dataset"]["class_counts"][str(k)] for k in range(K)]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(K)
    ax.bar(x - .2, before, .4, label="Real", color="#90A4AE")
    after = meta["scenarios"].get("A", {}).get("resampling", {}).get("counts_after")
    if after:
        ax.bar(x + .2, after, .4, label="Tras OA-SMOTE-NC (real + sintético)", color="#1565C0")
    ax.set_xticks(x, cfg["kdigo"]["stage_labels"])
    ax.set_ylabel("Pacientes"); ax.legend(frameon=False); ax.set_title("Distribución de clases")
    fig.savefig(out / "fig2_class_distribution.png"); plt.close(fig)


def fig_performance(ev, out):
    keys = ["precision_macro", "recall_macro", "f1_macro", "auc_ovr"]
    groups = [("prior", "Sin información"), ("rule_tree", "Regla (árbol con TFGe)"),
              ("A", "KORE-X"), ("B", "Sin creatinina (tamizaje)")]
    groups = [g for g in groups if g[0] in ev]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = .8 / len(groups)
    for i, (g, name) in enumerate(groups):
        m = [ev[g][k]["mean"] for k in keys]; s = [ev[g][k]["std"] for k in keys]
        ax.bar(np.arange(4) + i * w, m, w, yerr=s, capsize=2, label=name)
    ax.set_xticks(np.arange(4) + .4 - w / 2, ["Precision", "Recall", "F1-Score", "AUC-ROC"])
    ax.set_ylim(-.1, 1.05); ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.15))
    if "circularity" in ev:
        ax.set_title(f"IC_F1 = {ev['circularity'].get('f1_macro', float('nan')):.2f}", loc="right", fontsize=9)
    fig.savefig(out / "fig3_performance_circularity.png"); plt.close(fig)


def fig_confusion_calibration(rec, cfg, out):
    names = cfg["kdigo"]["stage_labels"]
    K = len(names)
    for s, r in rec["records"].items():
        idx = np.concatenate([np.asarray(x["test_idx"]) for x in r["records"]])
        P = np.vstack([np.asarray(x["proba"]) for x in r["records"]])
        pred = np.concatenate([np.asarray(x["pred"]) for x in r["records"]])
        y = rec["y"][idx]
        cm = np.zeros((K, K), int)
        np.add.at(cm, (y, pred), 1)
        cmn = cm / cm.sum(1, keepdims=True).clip(1)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
        a1.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                a1.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                        color="white" if cmn[i, j] > .5 else "black", fontsize=8)
        a1.set_xticks(range(K), names, fontsize=9); a1.set_yticks(range(K), names, fontsize=9)
        a1.set_xlabel("Predicho"); a1.set_ylabel("Real (KDIGO)"); a1.set_title(f"Matriz de confusión · escenario {s}")
        a2.plot([0, 1], [0, 1], "--", color="gray")
        for k, d in calibration_curve_data(y, P).items():
            a2.plot(d["mean_predicted"], d["fraction_positive"], "o-", color=cfg["kdigo"]["class_colors"][k], label=names[k])
        a2.set_xlabel("Probabilidad predicha"); a2.set_ylabel("Frecuencia observada")
        a2.set_title(f"Calibración · escenario {s}"); a2.legend(frameon=False, fontsize=8)
        fig.savefig(out / f"fig4_confusion_calibration_{s}.png"); plt.close(fig)


def fig_roc(rec, cfg, out):
    """Curvas ROC one-vs-rest por estadio (predicciones externas de la validación anidada)."""
    from korex.metrics import roc_curves
    r = rec["records"].get("A")
    if not r:
        return
    idx = np.concatenate([np.asarray(x["test_idx"]) for x in r["records"]])
    P = np.vstack([np.asarray(x["proba"]) for x in r["records"]])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    for k, d in roc_curves(rec["y"][idx], P).items():
        ax.plot(d["fpr"], d["tpr"], color=cfg["kdigo"]["class_colors"][k],
                label=f"{cfg['kdigo']['stage_labels'][k]} (AUC = {d['auc']:.3f})")
    ax.set_xlabel("1 − especificidad"); ax.set_ylabel("Sensibilidad (recall)")
    ax.set_title("Curvas ROC por estadio · KORE-X"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(out / "fig8_roc_A.png"); plt.close(fig)


def fig_ablation(ev, out):
    ab = ev.get("ablation")
    if not ab:
        return
    keys, names = ["precision_macro", "recall_macro", "f1_macro", "auc_ovr"], ["Precision", "Recall", "F1-Score", "AUC-ROC"]
    labels = {"sin_balanceo": "Sin balanceo", "smote_nc_clasico": "SMOTE-NC clásico", "oa_smote_nc": "OA-SMOTE-NC"}
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    w = .8 / len(ab)
    for i, (g, m) in enumerate(ab.items()):
        ax.bar(np.arange(4) + i * w, [m[k]["mean"] for k in keys], w, yerr=[m[k]["std"] for k in keys],
               capsize=2, label=labels.get(g, g))
    ax.set_xticks(np.arange(4) + .4 - w / 2, names); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center"); ax.set_title("Ablación del balanceo")
    fig.savefig(out / "fig9_ablation.png"); plt.close(fig)


def fig_dendrogram(pipe, out):
    for s, m in pipe.models.items():
        g = m.grouper_
        if g.linkage_ is None:
            continue
        fig, ax = plt.subplots(figsize=(8, max(3, .22 * len(g.columns_))))
        dendrogram(g.linkage_, labels=g.columns_, orientation="right", color_threshold=g.distance_cut, ax=ax)
        ax.axvline(g.distance_cut, ls="--", color="red", lw=1)
        ax.set_xlabel("δ = 1 − |ρ Spearman|"); ax.set_title(f"Agrupación de variables (UPGMA) · escenario {s}")
        fig.savefig(out / f"fig5_dendrogram_{s}.png"); plt.close(fig)


def fig_shap(pipe, cfg, out, scenario="A"):
    meta = pipe.metadata["scenarios"].get(scenario, {})
    imp, conv = meta.get("global_importance"), meta.get("convergence_by_class")
    if imp is None:
        from korex.kdigo import load_dataset
        from korex.labeling import label_dataset
        from korex.pipeline import resolve_path
        p = resolve_path(cfg["paths"]["data"])
        p = p if p.exists() else resolve_path(cfg["paths"]["synthetic_data"])
        data = label_dataset(load_dataset(p, cfg), cfg)
        m = pipe.models[scenario]
        Xt = m.transform(data.X.sample(min(150, len(data.y)), random_state=cfg["project"]["seed"]))
        ex = pipe.explainer(scenario)
        imp = ex.global_importance(Xt, 30)
        conv = ex.convergence_by_class(Xt, m.ens_.predict(Xt), 5)
    top = list(imp.items())[:15][::-1]
    labels = [cfg["variables"].get(k, {}).get("label", k) for k, _ in top]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.barh(labels, [v for _, v in top], color="#1565C0")
    ax.set_xlabel("Media |SHAP| sobre el estadio esperado"); ax.set_title(f"Importancia global · escenario {scenario}")
    fig.savefig(out / f"fig6_shap_global_{scenario}.png"); plt.close(fig)
    if conv:
        ks = sorted(conv, key=int)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for i, met in enumerate(["spearman", "jaccard_topk", "sign_agreement", "icsl"]):
            ax.bar(np.arange(len(ks)) + i * .2, [conv[k][met] for k in ks], .2, label=met)
        ax.set_xticks(np.arange(len(ks)) + .3, [cfg["kdigo"]["stage_labels"][int(k)] for k in ks], fontsize=9)
        ax.set_ylim(-.2, 1.05); ax.legend(frameon=False, fontsize=8, ncol=4)
        ax.set_title(f"Convergencia SHAP–LIME por clase · escenario {scenario}")
        fig.savefig(out / f"fig7_convergence_{scenario}.png"); plt.close(fig)


def _label(cfg, v):
    return cfg["variables"].get(v, {}).get("label", v)


def fig_shap_compare(pipe, cfg, out):
    """Importancia global con y sin creatinina, en % de la explicación total (lado a lado)."""
    sc = pipe.metadata["scenarios"]
    pairs = [(s, sc[s]["global_share"]) for s in ("A", "B") if sc.get(s, {}).get("global_share")]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (s, share), col in zip(axes, pairs, ("#0F766E", "#EF6C00")):
        top = list(share.items())[:12][::-1]
        ax.barh([_label(cfg, k) for k, _ in top], [100 * v for _, v in top], color=col)
        for i, (_, v) in enumerate(top):
            ax.text(100 * v + 1, i, f"{100 * v:.1f} %", va="center", fontsize=8)
        ax.set_xlim(0, 110); ax.set_xlabel("% de la explicación (media |SHAP|)")
        ax.set_title("Con creatinina (KORE-X)" if s == "A" else "Sin creatinina (tamizaje)")
    fig.tight_layout(); fig.savefig(out / "fig6_shap_global_A_vs_B.png"); plt.close(fig)


def fig_beeswarm(G, cfg, out, scenario, n=12):
    """Beeswarm SHAP: un punto por paciente, color = valor de la variable."""
    g = G.get(scenario)
    if not g:
        return
    S, names = np.asarray(g["shap"]), g["variables"]
    order = np.argsort(np.abs(S).mean(0))[::-1][:n][::-1]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(order) + 1.2))
    for row, j in enumerate(order):
        v = np.asarray(g["values"][names[j]], float)
        c = np.full(len(v), .5)
        ok = ~np.isnan(v)
        if ok.sum() > 1 and np.ptp(v[ok]) > 0:
            lo, hi = np.nanpercentile(v, [5, 95])
            c[ok] = np.clip((v[ok] - lo) / max(hi - lo, 1e-9), 0, 1)
        sc_ = ax.scatter(S[:, j], row + rng.uniform(-.3, .3, len(v)), c=c, cmap="coolwarm", vmin=0, vmax=1, s=9, alpha=.85)
    ax.axvline(0, color="gray", lw=.8)
    ax.set_yticks(range(len(order)), [_label(cfg, names[j]) for j in order], fontsize=8.5)
    ax.set_xlabel("Valor SHAP (efecto sobre el estadio esperado)")
    cb = fig.colorbar(sc_, ax=ax, ticks=[0, 1], fraction=.04); cb.ax.set_yticklabels(["bajo", "alto"]); cb.set_label("Valor")
    ax.set_title("Beeswarm SHAP · " + ("con creatinina" if scenario == "A" else "sin creatinina (tamizaje)"))
    fig.savefig(out / f"fig10_shap_beeswarm_{scenario}.png"); plt.close(fig)


def fig_dependence(G, cfg, out):
    """Dependencia SHAP: creatinina (con) y la variable principal sin creatinina, coloreadas por estadio real."""
    panels = []
    if G.get("A") and "creatinine" in G["A"]["variables"]:
        panels.append(("A", "creatinine"))
    if G.get("B"):
        top = max(range(len(G["B"]["variables"])), key=lambda j: np.abs(np.asarray(G["B"]["shap"])[:, j]).mean())
        panels.append(("B", G["B"]["variables"][top]))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4))
    axes = np.atleast_1d(axes)
    for ax, (s, var) in zip(axes, panels):
        g = G[s]
        j = g["variables"].index(var)
        x, yv = np.asarray(g["values"][var], float), np.asarray(g["shap"])[:, j]
        stage = np.asarray(G["y"])[np.asarray(g["index"])]
        for k in range(len(cfg["kdigo"]["classes"])):
            m = stage == k
            ax.scatter(x[m], yv[m], s=14, color=cfg["kdigo"]["class_colors"][k], label=cfg["kdigo"]["stage_labels"][k])
        if var == "creatinine":
            ax.set_xscale("log")
        unit = cfg["variables"].get(var, {}).get("unit", "")
        ax.set_xlabel(f"{_label(cfg, var)} ({unit})" if unit else _label(cfg, var)); ax.set_ylabel("Valor SHAP")
        ax.axhline(0, color="gray", lw=.8)
        ax.set_title(("Con creatinina · " if s == "A" else "Sin creatinina · ") + _label(cfg, var))
        ax.legend(frameon=False, fontsize=7, title="Estadio real", title_fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig11_shap_dependence.png"); plt.close(fig)


def _pooled(rec, s):
    r = rec["records"].get(s)
    if not r:
        return None, None
    idx = np.concatenate([np.asarray(x["test_idx"]) for x in r["records"]])
    return rec["y"][idx], np.vstack([np.asarray(x["proba"]) for x in r["records"]])


def fig_pr(rec, cfg, out):
    from korex.metrics import pr_curves
    y, P = _pooled(rec, "A")
    if y is None:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for k, d in pr_curves(y, P).items():
        ax.plot(d["recall"], d["precision"], color=cfg["kdigo"]["class_colors"][k],
                label=f"{cfg['kdigo']['stage_labels'][k]} (AP = {d['ap']:.3f})")
    ax.set_xlabel("Recall (sensibilidad)"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.03)
    ax.set_title("Curvas precisión-recall por estadio · KORE-X"); ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(out / "fig12_pr_A.png"); plt.close(fig)


def fig_cd(ev, out):
    """Diagrama de diferencia crítica (Friedman + Nemenyi; Demšar, 2006)."""
    from korex.stats import MODEL_NAMES
    c = ev.get("comparison", {}).get("f1_macro")
    if not c:
        return
    ranks = sorted(c["mean_rank"].items(), key=lambda t: t[1])
    k = len(ranks)
    fig, ax = plt.subplots(figsize=(8, 0.32 * k + 1.8))
    ax.set_xlim(0.5, k + 0.5); ax.set_ylim(-k / 2 - 1, 1.6); ax.axis("off")
    ax.hlines(0, 1, k, color="black"); ax.invert_xaxis()
    for r in range(1, k + 1):
        ax.vlines(r, 0, .12, color="black"); ax.text(r, .25, str(r), ha="center", fontsize=8)
    half = (k + 1) // 2
    for i, (m, r) in enumerate(ranks):
        side_left = i < half
        yy = -(i if side_left else i - half) * .5 - .5
        xx = 0.7 if side_left else k + 0.3
        ax.plot([r, r, xx], [0, yy, yy], color="#374151", lw=.8)
        ax.text(xx + (-.05 if side_left else .05), yy, f"{MODEL_NAMES.get(m, m)} ({r:.2f})",
                ha="left" if side_left else "right", va="center", fontsize=8,
                fontweight="bold" if m == "stack" else "normal")
    ax.plot([1, 1 + c["cd"]], [1.15, 1.15], color="#C62828", lw=2)
    ax.text(1 + c["cd"] / 2, 1.3, f"CD = {c['cd']:.2f}", ha="center", color="#C62828", fontsize=8)
    ax.set_title(f"Rangos medios de F1 macro (1 = mejor) · Friedman p = {c['p_value']:.4f}", fontsize=10)
    fig.savefig(out / "fig13_friedman_nemenyi.png"); plt.close(fig)


def fig_screening(rec, ev, out):
    """ROC del tamizaje sin creatinina para TFGe < 60 y < 30."""
    from sklearn.metrics import roc_auc_score, roc_curve
    y, P = _pooled(rec, "B")
    if y is None:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    for k, name, col in ((2, "TFGe < 60 (G3–G5)", "#EF6C00"), (3, "TFGe < 30 (G4–G5)", "#6A1B9A")):
        t, sc_ = (y >= k).astype(int), P[:, k:].sum(1)
        fpr, tpr, _ = roc_curve(t, sc_)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} · AUC = {roc_auc_score(t, sc_):.3f}")
    ax.set_xlabel("1 − especificidad"); ax.set_ylabel("Sensibilidad")
    w1 = ev.get("B", {}).get("accuracy_within1", {}).get("mean")
    ax.set_title("Tamizaje sin creatinina" + (f" · ±1 estadio {100 * w1:.0f} %" if w1 else ""))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(out / "fig14_screening_roc_B.png"); plt.close(fig)


def table_literature(pipe, cfg, out):
    import pandas as pd
    from korex.literature import COLUMNS, LITERATURE, korex_rows
    df = pd.DataFrame(korex_rows(pipe.metadata, cfg) + LITERATURE).rename(columns=COLUMNS)
    df.to_csv(out / "tabla_comparacion_literatura.csv", index=False, encoding="utf-8-sig")
    with open(out / "tabla_comparacion_literatura.md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
        for _, r in df.iterrows():
            f.write("| " + " | ".join(str(v) for v in r) + " |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap", action="store_true", help="calcular importancia SHAP y convergencia si no están en metadata")
    args = ap.parse_args()
    cfg = load_config()
    pipe = InferencePipeline.load(cfg=cfg)
    out = artifacts_dir(cfg) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_stage_sensitivity(pipe.metadata, cfg, out)
    fig_class_distribution(pipe.metadata, cfg, out)
    ev = pipe.metadata.get("evaluation", {})
    if ev:
        fig_performance(ev, out)
    rec = pipe.evaluation_records()
    if rec:
        fig_confusion_calibration(rec, cfg, out)
        fig_roc(rec, cfg, out)
    fig_ablation(ev, out)
    fig_dendrogram(pipe, out)
    if args.shap and "A" in pipe.models and not pipe.metadata["scenarios"]["A"].get("global_importance"):
        fig_shap(pipe, cfg, out, "A")
    for s in ("A", "B"):                                   # convergencia SHAP–LIME (si se entrenó con --explain-global)
        if pipe.metadata["scenarios"].get(s, {}).get("convergence_by_class"):
            fig_shap(pipe, cfg, out, s)
    fig_shap_compare(pipe, cfg, out)
    G = pipe.global_shap()
    if G:
        for s in ("A", "B"):
            fig_beeswarm(G, cfg, out, s)
        fig_dependence(G, cfg, out)
    if rec:
        fig_pr(rec, cfg, out)
        fig_screening(rec, ev, out)
    fig_cd(ev, out)
    table_literature(pipe, cfg, out)
    print(f"Figuras guardadas en {out}:")
    for f in sorted(list(out.glob("*.png")) + list(out.glob("*.csv"))):
        print("  ·", f.name)


if __name__ == "__main__":
    main()
