"""Figuras del Capitulo 6 (§6.2). Lee data/results/ y escribe en memoria/figuras/cap6/.

Estilo comun sobrio para la memoria (serif, vectorial). Cada figura se guarda en PDF
(vectorial, lo que entra en el LaTeX) y PNG (previsualizacion). No recarga el modelo:
todo sale de los artefactos persistidos (scores, metrics, .pt). La unica excepcion del
catalogo (firmas topologicas re-extraidas) vive aparte.

Paleta por familia de metodo: topologicos (TOHA, HalluZig) en calidos, espectral
(LapEigvals) en azul, estados ocultos (SEPs) en verde, caja negra (SelfCheckGPT) en gris.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

FIG_DIR = config.PROJECT_ROOT / "memoria" / "figuras" / "cap6"

# --- Estilo comun -----------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "figure.dpi": 120,
})

METHOD_COLOR = {
    "toha": "#E8743B",        # topologico (calido)
    "halluzig": "#B5462E",    # topologico (calido oscuro)
    "lapeigvals": "#4C72B0",  # espectral (azul)
    "seps": "#55A868",        # estados ocultos (verde)
    "selfcheckgpt": "#8C8C8C",  # caja negra (gris)
}
METHOD_LABEL = {
    "toha": "TOHA", "halluzig": "HalluZig", "lapeigvals": "LapEigvals",
    "seps": "SEPs", "selfcheckgpt": "SelfCheckGPT",
}
BENCH_LABEL = {
    "halueval-qa-gen": "HaluEval-QA", "much-en": "MUCH-EN", "much-es": "MUCH-ES",
    "mushroom-es": "Mu-SHROOM-ES", "squad": "SQuAD", "xquad-en": "XQuAD-EN",
    "xquad-es": "XQuAD-ES", "mushroom-es-zeroshot": "Mu-SHROOM-ES (0-shot)",
    "memerag-es": "MEMERAG-ES (RAG)", "memerag-en": "MEMERAG-EN (RAG)",
}
MODEL_LABEL = {"llama32-3b": "Llama 3.2 3B", "phi35-mini": "Phi-3.5 Mini", "gemma3-4b": "Gemma 3 4B"}
LABEL_COLOR = {0: "#4C72B0", 1: "#C44E52"}  # factual azul, alucinada rojo
LABEL_NAME = {0: "Factual", 1: "Alucinada"}

METHOD_ORDER = ["toha", "halluzig", "lapeigvals", "seps"]


def save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return FIG_DIR / f"{name}.pdf"


def _detection() -> pd.DataFrame:
    return pd.read_parquet(config.RESULTS_DIR / "metrics" / "detection.parquet")


# --- F1: comparativa de AUROC (boceto: foco en Llama, 7 benchmarks x 4 metodos) ---
def fig_auroc_comparison(model: str = "llama32-3b",
                         benches=("halueval-qa-gen", "much-en", "much-es",
                                  "mushroom-es", "squad", "xquad-en", "xquad-es")):
    d = _detection()
    d = d[(d.model == model) & (d.method.isin(METHOD_ORDER)) & (d.benchmark.isin(benches))]
    agg = d.groupby(["benchmark", "method"])["auroc"].agg(["mean", "std"]).reset_index()
    benches = [b for b in benches if b in agg.benchmark.values]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    nb, nm = len(benches), len(METHOD_ORDER)
    w = 0.8 / nm
    x = np.arange(nb)
    for j, m in enumerate(METHOD_ORDER):
        means = [agg[(agg.benchmark == b) & (agg.method == m)]["mean"].values for b in benches]
        stds = [agg[(agg.benchmark == b) & (agg.method == m)]["std"].values for b in benches]
        means = [float(v[0]) if len(v) else np.nan for v in means]
        stds = [float(v[0]) if len(v) else 0.0 for v in stds]
        ax.bar(x + j * w - 0.4 + w / 2, means, w, yerr=stds, capsize=2,
               color=METHOD_COLOR[m], label=METHOD_LABEL[m], error_kw={"elinewidth": 0.7, "alpha": 0.6})
    ax.axhline(0.5, ls="--", lw=0.8, color="black", alpha=0.6)
    ax.text(nb - 0.5, 0.505, "azar", fontsize=8, color="black", alpha=0.6, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([BENCH_LABEL.get(b, b) for b in benches], rotation=20, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Capacidad de detección por método y benchmark ({MODEL_LABEL.get(model, model)})", y=1.14)
    return save(fig, "fig_f1_auroc_comparativa")


# --- E1: proyeccion de los hidden states sobre la direccion de la sonda lineal de SEPs ---
def fig_seps_projection(model: str = "llama32-3b", bench: str = "halueval-qa-gen", layer: int | None = None):
    """Eje x: puntuacion de la sonda lineal (regresion logistica) obtenida por validacion
    cruzada (out-of-fold, sin fuga), es decir, la direccion discriminante que SEPs explota.
    Eje y: primera componente principal (no supervisada) para dar dispersion. Ilustra que
    en la direccion de la sonda las clases se separan, aunque el espacio crudo no lo haga."""
    import torch
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler

    d = config.RESULTS_DIR / model / bench
    H = torch.load(d / "seps_hidden.pt", map_location="cpu").numpy()  # [n, L+1, d_model]
    samples = pd.read_parquet(d / "samples.parquet")
    y = samples["label"].to_numpy()
    if layer is None:
        layer = int(round(0.7 * (H.shape[1] - 1)))  # capa media-tardia
    Xs = StandardScaler().fit_transform(H[:, layer, :])
    xs = cross_val_predict(LogisticRegression(max_iter=1000), Xs, y, cv=5, method="decision_function")
    pc1 = PCA(n_components=1, random_state=config.SEED).fit_transform(Xs)[:, 0]

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for lab in (0, 1):
        m = y == lab
        ax.scatter(xs[m], pc1[m], s=8, alpha=0.55, color=LABEL_COLOR[lab],
                   label=f"{LABEL_NAME[lab]} (n={int(m.sum())})", edgecolors="none")
    ax.axvline(0, ls="--", lw=0.8, color="black", alpha=0.5)
    ax.set_xlabel("Proyección sobre la sonda lineal (validación cruzada)")
    ax.set_ylabel("1ª componente principal")
    ax.set_title(f"Estados ocultos de SEPs, capa {layer}\n{MODEL_LABEL.get(model, model)} · {BENCH_LABEL.get(bench, bench)}")
    ax.legend(loc="best", fontsize=9)
    return save(fig, "fig_e1_seps_proyeccion")


# --- F7: eficiencia (tiempo/muestra por regimen + nº de forward passes) ---
def fig_efficiency(model: str = "llama32-3b"):
    e = pd.read_parquet(config.RESULTS_DIR / "metrics" / "efficiency.parquet")
    ei = e[(e.model == model) & (e.method == "toha")].copy()  # tiempo = pasada compartida
    order = ["halueval-qa-gen", "much-es", "much-en", "mushroom-es", "squad", "xquad-en", "xquad-es"]
    ei = ei[ei.benchmark.isin(order)].set_index("benchmark").reindex([b for b in order if b in ei.benchmark.values]).reset_index()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    # (a) tiempo/muestra por benchmark, coloreado por regimen (con/sin contexto)
    ctx = {"squad", "xquad-en", "xquad-es"}
    colors = ["#55A868" if b in ctx else "#4C72B0" for b in ei.benchmark]
    a1.bar([BENCH_LABEL.get(b, b) for b in ei.benchmark], ei.seconds_per_sample, color=colors)
    a1.set_ylabel("Tiempo de extracción por muestra (s)")
    a1.set_title("Coste de la pasada de extracción")
    a1.tick_params(axis="x", rotation=25)
    for lbl in a1.get_xticklabels():
        lbl.set_ha("right")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#4C72B0"), plt.Rectangle((0, 0), 1, 1, color="#55A868")]
    a1.legend(handles, ["Sin contexto (cerrado)", "Con contexto (comprensión lectora)"], loc="upper left")
    # (b) nº de forward passes por metodo
    meth = ["toha", "lapeigvals", "halluzig", "seps", "selfcheckgpt"]
    nf = [1, 1, 1, 1, 6]
    a2.bar([METHOD_LABEL[m] for m in meth], nf, color=[METHOD_COLOR[m] for m in meth])
    a2.set_ylabel("\\textit{Forward passes} por muestra".replace("\\textit{", "").replace("}", ""))
    a2.set_title("Coste de inferencia")
    a2.tick_params(axis="x", rotation=30)
    for lbl in a2.get_xticklabels():
        lbl.set_ha("right")
    a2.set_yticks([1, 2, 3, 4, 5, 6])
    fig.suptitle(f"Eficiencia computacional ({MODEL_LABEL.get(model, model)})", y=1.02)
    return save(fig, "fig_f7_eficiencia")


def _scores(model: str, bench: str, method: str) -> pd.DataFrame:
    return pd.read_parquet(config.RESULTS_DIR / "scores" / f"{model}_{bench}_{method}.parquet")


def _violin(ax, data_by_label, colors=LABEL_COLOR):
    parts = ax.violinplot(data_by_label, positions=[1, 2], showmedians=True, widths=0.8)
    for pc, lab in zip(parts["bodies"], (0, 1)):
        pc.set_facecolor(colors[lab]); pc.set_alpha(0.55); pc.set_edgecolor("none")
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("black"); parts[key].set_alpha(0.6); parts[key].set_linewidth(0.8)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Factual", "Alucinada"])


# --- F2: curvas ROC y PR de los cuatro metodos en un benchmark ---
def fig_roc_pr(model: str = "llama32-3b", bench: str = "halueval-qa-gen"):
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
    base = None
    for m in METHOD_ORDER:
        s = _scores(model, bench, m)
        y, sc = s["label"].to_numpy(), s["score"].to_numpy()
        base = y.mean() if base is None else base
        fpr, tpr, _ = roc_curve(y, sc)
        a1.plot(fpr, tpr, color=METHOD_COLOR[m], lw=1.6, label=f"{METHOD_LABEL[m]} ({roc_auc_score(y, sc):.2f})")
        prec, rec, _ = precision_recall_curve(y, sc)
        a2.plot(rec, prec, color=METHOD_COLOR[m], lw=1.6, label=f"{METHOD_LABEL[m]} ({average_precision_score(y, sc):.2f})")
    a1.plot([0, 1], [0, 1], ls="--", lw=0.8, color="black", alpha=0.5)
    a1.set_xlabel("Tasa de falsos positivos"); a1.set_ylabel("Tasa de verdaderos positivos")
    a1.set_title("Curva ROC"); a1.legend(title="AUROC", fontsize=8)
    a2.axhline(base, ls="--", lw=0.8, color="black", alpha=0.5)
    a2.set_xlabel("Exhaustividad"); a2.set_ylabel("Precisión")
    a2.set_title("Curva precisión-exhaustividad"); a2.legend(title="AUPRC", fontsize=8)
    fig.suptitle(f"Curvas de detección · {MODEL_LABEL.get(model, model)} · {BENCH_LABEL.get(bench, bench)}", y=1.02)
    return save(fig, "fig_f2_roc_pr")


# --- F5: distribucion de scores por clase y metodo (violines) ---
def fig_score_violins(model: str = "llama32-3b", bench: str = "halueval-qa-gen"):
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6))
    for ax, m in zip(axes, METHOD_ORDER):
        s = _scores(model, bench, m)
        _violin(ax, [s[s.label == 0]["score"].to_numpy(), s[s.label == 1]["score"].to_numpy()])
        ax.set_title(METHOD_LABEL[m])
        if ax is axes[0]:
            ax.set_ylabel("Puntuación de alucinación")
    fig.suptitle(f"Distribución de puntuaciones por clase · {MODEL_LABEL.get(model, model)} · {BENCH_LABEL.get(bench, bench)}", y=1.03)
    return save(fig, "fig_f5_violines_scores")


# --- F6: metricas topologicas internas por grupo (foco en H1; H0 es trivial) ---
def fig_topological_groups(model: str = "llama32-3b", bench: str = "halueval-qa-gen"):
    t = pd.read_parquet(config.RESULTS_DIR / "metrics" / "topological.parquet")
    t = t[(t.model == model) & (t.benchmark == bench)]
    cols = [("ent_h1", "Entropía de persistencia (H1)"), ("betti1", "β₁ (máx. de la curva)"),
            ("totpers_h1", "Suma de persistencia (H1)"), ("toha_div", "Divergencia TOHA media")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6))
    for ax, (col, lab) in zip(axes, cols):
        _violin(ax, [t[t.label == 0][col].to_numpy(), t[t.label == 1][col].to_numpy()])
        ax.set_title(lab, fontsize=10)
    fig.suptitle(f"Firmas topológicas internas por grupo · {MODEL_LABEL.get(model, model)} · {BENCH_LABEL.get(bench, bench)}", y=1.03)
    return save(fig, "fig_f6_topologicas_grupo")


# --- E2: mapa de calor de la discriminatividad por cabeza (TOHA) ---
def fig_toha_heads(model: str = "llama32-3b", bench: str = "squad"):
    import torch
    from sklearn.metrics import roc_auc_score
    d = config.RESULTS_DIR / model / bench
    toha = torch.load(d / "toha_headdiv.pt", map_location="cpu").numpy()  # [n,L,H]
    y = pd.read_parquet(d / "samples.parquet")["label"].to_numpy()
    n, L, H = toha.shape
    auc = np.full((L, H), 0.5)
    for l in range(L):
        for h in range(H):
            col = toha[:, l, h]
            if np.unique(col).size > 1 and 0 < y.sum() < len(y):
                auc[l, h] = roc_auc_score(y, col)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(auc, aspect="auto", cmap="RdBu_r", vmin=0.3, vmax=0.7, origin="lower")
    ax.set_xlabel("Cabeza"); ax.set_ylabel("Capa")
    ax.set_title(f"Discriminatividad por cabeza (TOHA)\n{MODEL_LABEL.get(model, model)} · {BENCH_LABEL.get(bench, bench)}")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("AUROC de la cabeza aislada")
    return save(fig, "fig_e2_toha_cabezas")


# --- E4: forest plot de AUROC con IC95 (de los 5 pliegues) ---
def fig_forest(model: str = "llama32-3b",
               benches=("halueval-qa-gen", "squad", "xquad-en", "xquad-es")):
    d = _detection()
    d = d[(d.model == model) & (d.method.isin(METHOD_ORDER)) & (d.benchmark.isin(benches))]
    agg = d.groupby(["benchmark", "method"])["auroc"].agg(["mean", "std", "count"]).reset_index()
    agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    benches = [b for b in benches if b in agg.benchmark.values]

    rows = []
    for b in benches:
        for m in METHOD_ORDER:
            r = agg[(agg.benchmark == b) & (agg.method == m)]
            if len(r):
                rows.append((b, m, float(r["mean"].iloc[0]), float(r["ci"].iloc[0])))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1))
    yp = np.arange(len(rows))[::-1]
    for yi, (b, m, mu, ci) in zip(yp, rows):
        ax.errorbar(mu, yi, xerr=ci, fmt="o", color=METHOD_COLOR[m], capsize=3, ms=5, lw=1.4)
    ax.axvline(0.5, ls="--", lw=0.8, color="black", alpha=0.5)
    ax.set_yticks(yp)
    ax.set_yticklabels([f"{BENCH_LABEL.get(b, b)} · {METHOD_LABEL[m]}" for b, m, _, _ in rows], fontsize=8)
    ax.set_xlabel("AUROC (IC 95 %)")
    ax.set_xlim(0.4, 1.0)
    ax.set_title(f"AUROC con intervalo de confianza · {MODEL_LABEL.get(model, model)}")
    # separadores entre benchmarks
    for k in range(1, len(benches)):
        ax.axhline(len(rows) - k * len(METHOD_ORDER) - 0.5, color="black", alpha=0.12, lw=0.6)
    return save(fig, "fig_e4_forest_auroc")


# --- F3: transferibilidad (cross-language EN<->ES y cross-model con HalluZig) ---
def _cv_auroc(model: str, bench: str, method: str) -> float:
    d = _detection()
    r = d[(d.model == model) & (d.benchmark == bench) & (d.method == method)]
    return float(r["auroc"].mean()) if len(r) else float("nan")


def fig_transfer(model: str = "llama32-3b"):
    from sklearn.metrics import roc_auc_score

    from .evaluation.transfer import transfer_classifier, transfer_seps, transfer_toha
    from .io import load_array, load_samples
    from .methods.halluzig_score import make_halluzig_estimator
    from .methods.lapeigvals_score import make_lapeigvals_estimator

    # ---- Panel A: cross-language EN<->ES sobre XQuAD (mismo modelo) ----
    def xlang_score(method, src, dst):
        y_tr = load_samples(model, f"xquad-{src}")["label"].to_numpy()
        y_te = load_samples(model, f"xquad-{dst}")["label"].to_numpy()
        if method == "toha":
            a_tr = load_array(model, f"xquad-{src}", "toha_headdiv")
            a_te = load_array(model, f"xquad-{dst}", "toha_headdiv")
            s, _ = transfer_toha(a_tr, y_tr, a_te, config.TOHA_PROBE_SET_SIZE,
                                 config.TOHA_N_OPT_CANDIDATES, config.SEED)
        elif method == "lapeigvals":
            x_tr = load_array(model, f"xquad-{src}", "lapeigvals_feats").reshape(len(y_tr), -1)
            x_te = load_array(model, f"xquad-{dst}", "lapeigvals_feats").reshape(len(y_te), -1)
            s, _ = transfer_classifier(x_tr, y_tr, x_te, lambda n: make_lapeigvals_estimator(x_tr.shape[1], n))
        elif method == "halluzig":
            x_tr = load_array(model, f"xquad-{src}", "halluzig_feats")
            x_te = load_array(model, f"xquad-{dst}", "halluzig_feats")
            s, _ = transfer_classifier(x_tr, y_tr, x_te, make_halluzig_estimator)
        else:  # seps
            h_tr = load_array(model, f"xquad-{src}", "seps_hidden")
            h_te = load_array(model, f"xquad-{dst}", "seps_hidden")
            s, _ = transfer_seps(h_tr, y_tr, h_te, config.SEED, config.SEPS_VAL_FRAC)
        return roc_auc_score(y_te, s)

    methods = ["toha", "lapeigvals", "halluzig", "seps"]
    en2es = [xlang_score(m, "en", "es") for m in methods]
    es2en = [xlang_score(m, "es", "en") for m in methods]
    cv_es = [_cv_auroc(model, "xquad-es", m) for m in methods]
    cv_en = [_cv_auroc(model, "xquad-en", m) for m in methods]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(methods)); w = 0.38
    a1.bar(x - w / 2, en2es, w, color="#4C72B0", label="EN → ES")
    a1.bar(x + w / 2, es2en, w, color="#55A868", label="ES → EN")
    # referencia: CV intra-idioma (objetivo de la transferencia)
    for xi, (ce, cs) in enumerate(zip(cv_en, cv_es)):
        a1.plot([xi - w, xi], [ce, ce], color="black", lw=1.3, alpha=0.7)
        a1.plot([xi, xi + w], [cs, cs], color="black", lw=1.3, alpha=0.7)
    a1.plot([], [], color="black", lw=1.3, alpha=0.7, label="CV intra-idioma (objetivo)")
    a1.axhline(0.5, ls="--", lw=0.8, color="black", alpha=0.5)
    a1.set_xticks(x); a1.set_xticklabels([METHOD_LABEL[m] for m in methods])
    a1.set_ylabel("AUROC en el idioma destino"); a1.set_ylim(0.4, 1.0)
    a1.set_title("Transferencia cross-language (XQuAD, preguntas paralelas)")
    a1.legend(fontsize=8, loc="upper right")

    # ---- Panel B: cross-model con HalluZig (D=902 fija) sobre HaluEval-QA ----
    models = ["phi35-mini", "llama32-3b", "gemma3-4b"]
    bench = "halueval-qa-gen"
    feats = {m: load_array(m, bench, "halluzig_feats") for m in models}
    ys = {m: load_samples(m, bench)["label"].to_numpy() for m in models}
    M = np.zeros((len(models), len(models)))
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if i == j:
                M[i, j] = _cv_auroc(mi, bench, "halluzig")
            else:
                s, _ = transfer_classifier(feats[mi], ys[mi], feats[mj], make_halluzig_estimator)
                M[i, j] = roc_auc_score(ys[mj], s)
    im = a2.imshow(M, cmap="RdBu_r", vmin=0.4, vmax=0.7, aspect="auto")
    for i in range(len(models)):
        for j in range(len(models)):
            a2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(M[i, j] - 0.5) > 0.08 else "black")
    a2.set_xticks(range(len(models))); a2.set_yticks(range(len(models)))
    a2.set_xticklabels([MODEL_LABEL[m] for m in models], rotation=20, ha="right", fontsize=8)
    a2.set_yticklabels([MODEL_LABEL[m] for m in models], fontsize=8)
    a2.set_xlabel("Evaluación (test)"); a2.set_ylabel("Entrenamiento")
    a2.set_title("Cross-model · HalluZig\n(diagonal = CV intra-modelo)", fontsize=10)
    fig.colorbar(im, ax=a2, shrink=0.8, label="AUROC")
    fig.suptitle(f"Transferibilidad de las firmas · {MODEL_LABEL.get(model, model)}", y=1.02)
    return save(fig, "fig_f3_transferibilidad")


# --- F8: ablaciones (AUROC de TOHA vs nº de cabezas; importancia de features de HalluZig) ---
def fig_ablations(model: str = "llama32-3b", toha_bench: str = "squad", hz_bench: str = "halueval-qa-gen"):
    from sklearn.metrics import roc_auc_score

    from .io import load_array, load_samples
    from .methods.halluzig_score import make_halluzig_estimator
    from .methods.toha_score import _stratified_probe

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) TOHA: AUROC frente al numero de hallucination-aware heads
    a = load_array(model, toha_bench, "toha_headdiv")
    y = load_samples(model, toha_bench)["label"].to_numpy()
    n = len(a); div = a.reshape(n, -1)
    probe = _stratified_probe(y, np.arange(n), config.TOHA_PROBE_SET_SIZE, config.SEED)
    deltas = div[probe][y[probe] == 1].mean(0) - div[probe][y[probe] == 0].mean(0)
    ranked = np.argsort(-deltas)
    mask = np.ones(n, bool); mask[probe] = False
    Ns = [1, 2, 4, 6, 8, 10, 15, 20, 30]
    aucs = [roc_auc_score(y[mask], div[mask][:, ranked[:N]].mean(1)) for N in Ns]
    a1.plot(Ns, aucs, marker="o", color=METHOD_COLOR["toha"])
    a1.axhline(0.5, ls="--", lw=0.8, color="black", alpha=0.5)
    a1.set_xlabel("Nº de cabezas seleccionadas ($N_{\\mathrm{opt}}$)")
    a1.set_ylabel("AUROC (fuera del \\textit{probe})".replace("\\textit{", "").replace("}", ""))
    a1.set_title(f"TOHA · sensibilidad al nº de cabezas\n({BENCH_LABEL.get(toha_bench, toha_bench)})")

    # (b) HalluZig: importancia agregada por componente del vector D=902
    X = load_array(model, hz_bench, "halluzig_feats")
    yh = load_samples(model, hz_bench)["label"].to_numpy()
    est = make_halluzig_estimator(len(X))
    est.fit(X, yh)
    imp = getattr(est, "feature_importances_", None)
    if imp is None and hasattr(est, "steps"):
        imp = est.steps[-1][1].feature_importances_
    pi = config.HALLUZIG_PI_RES ** 2; blk = pi + 1 + config.HALLUZIG_BETTI_POINTS
    groups = {
        "PI H0": imp[:pi].sum(), "Entropía H0": imp[pi], "Betti H0": imp[pi + 1:blk].sum(),
        "PI H1": imp[blk:blk + pi].sum(), "Entropía H1": imp[blk + pi], "Betti H1": imp[blk + pi + 1:].sum(),
    }
    names = list(groups); vals = [groups[k] for k in names]
    cols = ["#B5462E" if "H1" in k else "#E8743B" for k in names]
    a2.barh(names[::-1], vals[::-1], color=cols[::-1])
    a2.set_xlabel("Importancia agregada (\\textit{Random Forest})".replace("\\textit{", "").replace("}", ""))
    a2.set_title(f"HalluZig · importancia por componente\n({BENCH_LABEL.get(hz_bench, hz_bench)})")
    fig.suptitle(f"Ablaciones de los métodos topológicos · {MODEL_LABEL.get(model, model)}", y=1.02)
    return save(fig, "fig_f8_ablaciones")


# =============================================================================
# Figuras del bloque RAG (Fase 5) — narrativa SEPARADA de las F/E del §6.2.
# Se generan con generate_rag(); no entran en generate_all().
# =============================================================================

# --- R1: mapa de regímenes (AUROC por método a lo largo del eje de contexto) ---
def fig_regime_map(model: str = "llama32-3b"):
    """AUROC de cada método ordenando los benchmarks por riqueza de contexto P:
    sin estructura para la respuesta (P = pregunta corta) frente a con contexto
    (P = pasaje/documento). Cuenta la tesis: TOHA necesita estructura, contexto rico
    o respuestas largas, LapEigvals es fuerte sin contexto y cae con RAG, SEPs es
    robusto en todo régimen. MUCH se omite (base rate ~4%, AUROC no fiable; se
    comenta aparte)."""
    no_ctx = ["halueval-qa-gen", "mushroom-es"]
    ctx = ["squad", "xquad-en", "xquad-es", "memerag-es"]
    benches = no_ctx + ctx
    x = np.arange(len(benches))
    split = len(no_ctx) - 0.5  # frontera entre regímenes

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    # bandas de régimen
    ax.axvspan(-0.5, split, color="#4C72B0", alpha=0.05)
    ax.axvspan(split, len(benches) - 0.5, color="#55A868", alpha=0.07)
    ax.axvline(split, ls="-", lw=1.0, color="black", alpha=0.3)
    ax.axhline(0.5, ls="--", lw=0.8, color="black", alpha=0.5)

    for m in METHOD_ORDER:
        ys = [_cv_auroc(model, b, m) for b in benches]
        ax.plot(x, ys, marker="o", ms=6, lw=1.6, color=METHOD_COLOR[m], label=METHOD_LABEL[m])

    # etiquetas; Mu-SHROOM se anota como respuestas largas (por qué TOHA rinde sin recuperación)
    labels = [BENCH_LABEL.get(b, b) for b in benches]
    labels[benches.index("mushroom-es")] = "Mu-SHROOM-ES\n(resp. largas)"
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    for tick, b in zip(ax.get_xticklabels(), benches):
        if b == "memerag-es":
            tick.set_fontweight("bold")  # ancla del RAG
    ax.set_ylim(0.35, 0.9)
    ax.set_ylabel("AUROC (CV 5-fold)")
    ax.text(0.5, 0.875, "Sin contexto recuperado\n(P = pregunta / prompt)", ha="center", va="top",
            fontsize=9, color="#3a567f", style="italic")
    ax.text(3.5, 0.875, "Con contexto\n(P = pasaje / documento)", ha="center", va="top",
            fontsize=9, color="#3d7a55", style="italic")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.36), fontsize=9)
    ax.set_title(f"Mapa de regímenes: detección por familia según el contexto · {MODEL_LABEL.get(model, model)}")
    return save(fig, "fig_r1_mapa_regimenes")


# --- R2: resultado del experimento RAG (MEMERAG-ES): AUROC + curvas ROC ---
def fig_rag_memerag(model: str = "llama32-3b", hz_corrected: float = 0.54):
    """Figura propia del experimento RAG: AUROC de los 4 métodos en MEMERAG-ES (con
    la corrección de HalluZig anotada) y sus curvas ROC. `hz_corrected` es el AUROC
    de HalluZig al re-vectorizar con el eje sin normalizar (ver Cap. 6 §HalluZig)."""
    from sklearn.metrics import roc_curve

    bench = "memerag-es"
    d = _detection()
    d = d[(d.model == model) & (d.benchmark == bench)]
    agg = d.groupby("method")["auroc"].agg(["mean", "std"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})

    # (a) barras de AUROC con IC (std de los folds) y corrección de HalluZig
    means = [agg.loc[m, "mean"] for m in METHOD_ORDER]
    stds = [agg.loc[m, "std"] for m in METHOD_ORDER]
    cols = [METHOD_COLOR[m] for m in METHOD_ORDER]
    xb = np.arange(len(METHOD_ORDER))
    a1.bar(xb, means, yerr=stds, color=cols, capsize=4, error_kw={"lw": 1, "alpha": 0.6})
    a1.axhline(0.5, ls="--", lw=0.8, color="black", alpha=0.5)
    # corrección de HalluZig (eje sin normalizar): barra fantasma + flecha
    hz = METHOD_ORDER.index("halluzig")
    a1.bar(hz, hz_corrected, color=METHOD_COLOR["halluzig"], alpha=0.28, hatch="//",
           edgecolor=METHOD_COLOR["halluzig"], lw=0.8)
    a1.annotate(f"{hz_corrected:.2f}\ncorregido", xy=(hz, hz_corrected), xytext=(hz + 0.05, hz_corrected + 0.07),
                fontsize=7.5, ha="center", color=METHOD_COLOR["halluzig"],
                arrowprops=dict(arrowstyle="->", color=METHOD_COLOR["halluzig"], lw=0.8))
    for xi, mu in zip(xb, means):
        a1.text(xi, mu + 0.015, f"{mu:.2f}", ha="center", va="bottom", fontsize=8.5)
    a1.set_xticks(xb); a1.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER])
    a1.set_ylim(0.0, 0.9); a1.set_ylabel("AUROC (CV 5-fold)")
    a1.set_title("Detección en MEMERAG-ES (RAG)")

    # (b) curvas ROC de los 4 métodos (scores out-of-fold)
    for m in METHOD_ORDER:
        s = _scores(model, bench, m)
        fpr, tpr, _ = roc_curve(s["label"], s["score"])
        a2.plot(fpr, tpr, lw=1.7, color=METHOD_COLOR[m],
                label=f"{METHOD_LABEL[m]} ({agg.loc[m, 'mean']:.2f})")
    a2.plot([0, 1], [0, 1], ls="--", lw=0.8, color="black", alpha=0.5)
    a2.set_xlabel("Tasa de falsos positivos"); a2.set_ylabel("Tasa de verdaderos positivos")
    a2.set_title("Curvas ROC"); a2.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"Experimento RAG · MEMERAG-ES · {MODEL_LABEL.get(model, model)} "
                 f"(detección sobre respuestas dadas, estilo RAGTruth)", y=1.03, fontsize=11)
    return save(fig, "fig_r2_rag_memerag")


# --- R3: transferencia cross-language EN<->ES en el régimen RAG (MEMERAG) ---
def fig_rag_crosslang(model: str = "llama32-3b"):
    """Reafirma en RAG (la cancha de TOHA) lo visto en XQuAD: las firmas transfieren
    entre idiomas. Barras EN→ES y ES→EN por método, con la CV intra-idioma del destino
    como referencia. Lee de detection.parquet (requiere haber corrido 20_evaluate sobre
    memerag-en y 23_transfer en ambos sentidos)."""
    methods = METHOD_ORDER
    # transferencia (fold=-1): out_bench = f"{test}-from-{train_lang}" en 23_transfer
    en2es = [_cv_auroc(model, "memerag-es-from-en", m) for m in methods]  # train en, test es
    es2en = [_cv_auroc(model, "memerag-en-from-es", m) for m in methods]  # train es, test en
    cv_es = [_cv_auroc(model, "memerag-es", m) for m in methods]
    cv_en = [_cv_auroc(model, "memerag-en", m) for m in methods]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(methods)); w = 0.38
    ax.bar(x - w / 2, en2es, w, color="#4C72B0", label="EN → ES")
    ax.bar(x + w / 2, es2en, w, color="#55A868", label="ES → EN")
    for xi, (ce, cs) in enumerate(zip(cv_en, cv_es)):
        ax.plot([xi - w, xi], [ce, ce], color="black", lw=1.3, alpha=0.7)
        ax.plot([xi, xi + w], [cs, cs], color="black", lw=1.3, alpha=0.7)
    ax.plot([], [], color="black", lw=1.3, alpha=0.7, label="CV intra-idioma (objetivo)")
    ax.axhline(0.5, ls="--", lw=0.8, color="black", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels([METHOD_LABEL[m] for m in methods])
    ax.set_ylabel("AUROC en el idioma destino"); ax.set_ylim(0.35, 0.85)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"Transferencia cross-language en RAG (MEMERAG) · {MODEL_LABEL.get(model, model)}")
    return save(fig, "fig_r3_rag_crosslang")


def generate_rag():
    print(fig_regime_map())
    print(fig_rag_memerag())
    print(fig_rag_crosslang())


def generate_all():
    print(fig_auroc_comparison())
    print(fig_roc_pr())
    print(fig_score_violins())
    print(fig_topological_groups())
    print(fig_efficiency())
    print(fig_toha_heads())
    print(fig_forest())
    print(fig_transfer())
    print(fig_ablations())
    print(fig_seps_projection())  # el mas lento (sonda con validacion cruzada)


if __name__ == "__main__":
    generate_all()
