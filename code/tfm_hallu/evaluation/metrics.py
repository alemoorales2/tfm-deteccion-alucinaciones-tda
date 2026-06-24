"""Métricas de detección (Cap. 5 §5.5.1).

AUROC (primaria), AUPRC, F1 al umbral óptimo y FPR@95%TPR, más intervalos de
confianza por bootstrap (Fase 4). El test de DeLong (comparación entre métodos)
se añade junto a las figuras comparativas.

Convención: `score` alto = mayor sospecha de alucinación; `y` con 1 = alucinada.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .. import config


def f1_at_optimal(y, score) -> float:
    """F1 máximo sobre todos los umbrales posibles."""
    precision, recall, _ = precision_recall_curve(y, score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(np.max(f1))


def fpr_at_tpr(y, score, target_tpr: float = 0.95) -> float:
    """Menor FPR alcanzable con TPR >= target_tpr."""
    fpr, tpr, _ = roc_curve(y, score)
    mask = tpr >= target_tpr
    return float(fpr[mask].min()) if mask.any() else 1.0


def detection_metrics(y, score) -> dict:
    y = np.asarray(y)
    score = np.asarray(score)
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "f1": f1_at_optimal(y, score),
        "fpr95": fpr_at_tpr(y, score, 0.95),
    }


def bootstrap_ci(
    y,
    score,
    metrics=("auroc", "auprc"),
    n_boot: int = config.N_BOOTSTRAP,
    ci: float = config.BOOTSTRAP_CI,
    seed: int = config.SEED,
) -> dict:
    """Intervalo de confianza por bootstrap (remuestreo de muestras con
    reemplazo) de las métricas indicadas. Devuelve, por métrica, un dict con
    `point` (sobre la muestra original), `lo`, `hi` y `n_boot_valid`.

    Las muestras con score NaN se descartan antes (p. ej. fuera del subconjunto
    de SelfCheckGPT). Cada réplica que pierda alguna de las dos clases se ignora
    (el AUROC no está definido); si quedan muy pocas réplicas válidas, el CI se
    marca con NaN. Pensado para clases desequilibradas (MUCH): el remuestreo no
    estratifica, de modo que el CI refleja también la incertidumbre del número de
    positivos."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    keep = ~np.isnan(score)
    y, score = y[keep], score[keep]
    n = len(y)
    funcs = {
        "auroc": lambda yy, ss: roc_auc_score(yy, ss),
        "auprc": lambda yy, ss: average_precision_score(yy, ss),
        "f1": f1_at_optimal,
        "fpr95": lambda yy, ss: fpr_at_tpr(yy, ss, 0.95),
    }
    out = {}
    point = {m: float(funcs[m](y, score)) for m in metrics if len(set(y.tolist())) > 1}
    rng = np.random.default_rng(seed)
    boot = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, ss = y[idx], score[idx]
        if len(set(yy.tolist())) < 2:
            continue
        for m in metrics:
            boot[m].append(funcs[m](yy, ss))
    alpha = (1 - ci) / 2
    for m in metrics:
        vals = np.asarray(boot[m], dtype=float)
        if len(vals) < max(20, n_boot // 20):
            out[m] = {"point": point.get(m, float("nan")), "lo": float("nan"),
                      "hi": float("nan"), "n_boot_valid": int(len(vals))}
        else:
            out[m] = {
                "point": point.get(m, float(np.median(vals))),
                "lo": float(np.quantile(vals, alpha)),
                "hi": float(np.quantile(vals, 1 - alpha)),
                "n_boot_valid": int(len(vals)),
            }
    return out
