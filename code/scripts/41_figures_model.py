"""41_figures_model.py - figuras del Cap. 6 que SI recargan el modelo (F4 y E3).

Re-extrae la atencion de dos muestras (una factual, una alucinada) de un benchmark
con contexto (SQuAD por defecto, donde TOHA rinde) y dibuja sus firmas topologicas.
Es la unica parte de §6.2 que vuelve a cargar el modelo.

  - F4: firma de TOHA. Para la cabeza mas discriminativa, el cross-barcode H0
    (coste de adherir cada token de la respuesta al bloque del prompt) y la
    distribucion de esos costes, factual vs alucinada. Es la divergencia que TOHA
    mide: si la respuesta se "despega" del contexto, los costes crecen.
  - E3: curva de Betti a lo largo de la filtracion zigzag de HalluZig (H0 y H1),
    factual vs alucinada.

    .venv/bin/python code/scripts/41_figures_model.py [MODEL] [BENCH]
"""

import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser

from tfm_hallu import config, viz
from tfm_hallu.data import load_squad, load_xquad
from tfm_hallu.extraction import encode_sample, extract_attention, load_model
from tfm_hallu.io import load_array, load_samples
from tfm_hallu.topology import betti_curve, build_layer_graphs, zigzag_persistence

LC, LN = viz.LABEL_COLOR, viz.LABEL_NAME


def _select_head(model_key, bench):
    """Cabeza (l,h) con mayor AUROC aislado sobre la divergencia TOHA ya persistida."""
    from sklearn.metrics import roc_auc_score
    a = load_array(model_key, bench, "toha_headdiv")
    y = load_samples(model_key, bench)["label"].to_numpy()
    best, best_auc = (0, 0), 0.5
    for l in range(a.shape[1]):
        for h in range(a.shape[2]):
            col = a[:, l, h]
            if np.unique(col).size > 1:
                auc = roc_auc_score(y, col)
                if auc > best_auc:
                    best, best_auc = (l, h), auc
    return best, best_auc


def _cross_barcode_h0(attn_2d, n_prompt):
    """Barras finitas del cross-barcode H0 (P colapsado); cada (0, death) es el coste
    de adherir un token de la respuesta al prompt. Replica topology.mtop_div_h0."""
    a = np.maximum(np.asarray(attn_2d, dtype=np.float64), np.asarray(attn_2d, dtype=np.float64).T)
    d = 1.0 - a
    d[:n_prompt, :n_prompt] = 0.0
    np.fill_diagonal(d, 0.0)
    dgm0 = ripser(d, maxdim=0, distance_matrix=True)["dgms"][0]
    return dgm0[np.isfinite(dgm0[:, 1])]


def _pick(samples):
    return {0: next(s for s in samples if s.label == 0), 1: next(s for s in samples if s.label == 1)}


def fig_f4(cb, head, auc, model, bench):
    xmax = max(float(cb[l][:, 1].max()) for l in (0, 1)) * 1.05
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for col, lab in enumerate((0, 1)):
        deaths = np.sort(cb[lab][:, 1])
        ax = axes[0, col]
        for k, dth in enumerate(deaths):
            ax.plot([0, dth], [k, k], color=LC[lab], lw=1.0, alpha=0.75)
        ax.set_xlim(0, xmax)
        ax.set_title(f"{LN[lab]} · cross-barcode H₀\nMTop-Div={cb[lab][:, 1].sum():.1f} · |R|={len(deaths)}")
        ax.set_xlabel("Coste de adhesión al contexto (1 − atención)"); ax.set_ylabel("Token de la respuesta")
        ax = axes[1, col]
        ax.hist(cb[lab][:, 1], bins=24, range=(0, xmax), color=LC[lab], alpha=0.75)
        ax.set_xlabel("Coste de adhesión"); ax.set_ylabel("Nº de tokens")
        ax.set_title(f"{LN[lab]} · distribución de costes")
    fig.suptitle(f"Firma topológica de TOHA · cabeza (capa {head[0]}, cabeza {head[1]}), AUROC {auc:.2f}\n"
                 f"{viz.MODEL_LABEL.get(model, model)} · {viz.BENCH_LABEL.get(bench, bench)}")
    fig.subplots_adjust(hspace=0.45, top=0.86)
    return viz.save(fig, "fig_f4_firmas_topologicas")


def fig_e3(zz, model, bench, points=50):
    xs = np.linspace(0, 1, points)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, dim in zip(axes, (0, 1)):
        for lab in (0, 1):
            bc = betti_curve(zz[lab]["bc"][dim], zz[lab]["max_level"], points)
            ax.plot(xs, bc, color=LC[lab], lw=1.8, label=LN[lab])
        ax.set_title(f"Curva de Betti · H{dim}")
        ax.set_xlabel("Filtración (normalizada)"); ax.set_ylabel(f"β{dim}"); ax.legend(fontsize=9)
    fig.suptitle(f"Evolución de los números de Betti (HalluZig) · {viz.MODEL_LABEL.get(model, model)} · {viz.BENCH_LABEL.get(bench, bench)}", y=1.03)
    return viz.save(fig, "fig_e3_betti_capas")


def main(model_key="llama32-3b", bench="squad"):
    spec = config.MODELS[model_key]
    samples = load_squad(model_key) if bench == "squad" else load_xquad(model_key, "es")
    chosen = _pick(samples)
    (l, h), auc = _select_head(model_key, bench)
    print(f"Cabeza más discriminativa: capa {l}, cabeza {h} (AUROC {auc:.3f})")
    print(f"Muestras: factual={chosen[0].id}  alucinada={chosen[1].id}")
    model, tok, device = load_model(spec)
    cb, zz = {}, {}
    for lab, s in chosen.items():
        enc = encode_sample(tok, s, spec.t_max)
        attn, _ = extract_attention(model, enc, device)
        cb[lab] = _cross_barcode_h0(attn[l, h], enc.n_prompt)
        graphs, n_nodes = build_layer_graphs(attn, config.HALLUZIG_PERCENTILE)
        barcode, n_levels = zigzag_persistence(graphs, n_nodes)
        zz[lab] = {"bc": barcode, "max_level": n_levels - 1}
        del attn
        print(f"  label {lab}: |R|={len(cb[lab])} barras H0, MTop-Div={cb[lab][:, 1].sum():.1f}")
    print(fig_f4(cb, (l, h), auc, model_key, bench))
    print(fig_e3(zz, model_key, bench))
    return 0


if __name__ == "__main__":
    mk = sys.argv[1] if len(sys.argv) > 1 else "llama32-3b"
    bk = sys.argv[2] if len(sys.argv) > 2 else "squad"
    sys.exit(main(mk, bk))
