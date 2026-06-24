"""TOHA: selección de cabezas y puntuación (etapa SIN modelo).

Implementa el protocolo acordado (Cap. 4 §4.5-4.6, sin fuga de información):

  Para cada fold de test, sobre el bloque de entrenamiento (los otros folds):
    1. Se toma un probe set estratificado de 50 muestras.
    2. Δ_ij = media(div alucinadas) - media(div factuales) del probe, por cabeza.
    3. Las cabezas se ordenan por Δ decreciente (hallucination-aware heads).
    4. N_opt ∈ {2,4,6,8,10} se elige maximizando el AUROC EN ENTRENAMIENTO.
  El fold de test se puntúa promediando la divergencia sobre las N_opt cabezas
  seleccionadas. La selección nunca ve el fold de test.

La divergencia ya viene normalizada por |R| de la etapa con modelo, así que aquí
solo hay selección y promediado, sin recálculo topológico.
"""

import numpy as np

from ..evaluation.metrics import detection_metrics
from sklearn.metrics import roc_auc_score


def _stratified_probe(y, train_idx, size, seed) -> np.ndarray:
    """Subconjunto estratificado de `size` índices tomados de `train_idx`."""
    rng = np.random.RandomState(seed)
    k = size // 2
    pos = train_idx[y[train_idx] == 1]
    neg = train_idx[y[train_idx] == 0]
    sel_pos = rng.choice(pos, min(k, len(pos)), replace=False)
    sel_neg = rng.choice(neg, min(size - k, len(neg)), replace=False)
    return np.concatenate([sel_pos, sel_neg])


def _select_n_opt(div_flat, y, train_idx, ranked, candidates):
    """N_opt que maximiza el AUROC en entrenamiento."""
    best_n, best_auc = candidates[0], -1.0
    for n in candidates:
        heads = ranked[:n]
        s = div_flat[np.ix_(train_idx, heads)].mean(axis=1)
        try:
            auc = roc_auc_score(y[train_idx], s)
        except ValueError:
            auc = 0.5
        if auc > best_auc:
            best_auc, best_n = auc, n
    return best_n, best_auc


def run_cv_toha(headdiv, y, folds, probe_size, candidates, seed):
    """Validación cruzada de TOHA. Devuelve (scores, fold_metrics, info).

    - `headdiv`: [n, L, H] divergencias normalizadas.
    - `scores`: [n], cada muestra puntuada cuando está en su fold de test.
    - `fold_metrics`: lista de dicts con métricas por fold (+ n_opt, n_eval).
    - `info`: lista de dicts con n_opt, auc_train y las cabezas top por fold.
    """
    n, L, H = headdiv.shape
    div_flat = headdiv.reshape(n, L * H)
    y = np.asarray(y)
    folds = np.asarray(folds)

    scores = np.full(n, np.nan, dtype=np.float64)
    fold_metrics, info = [], []

    for k in sorted(set(folds.tolist())):
        test_idx = np.where(folds == k)[0]
        train_idx = np.where(folds != k)[0]

        probe_idx = _stratified_probe(y, train_idx, probe_size, seed + k)
        p = div_flat[probe_idx]
        deltas = p[y[probe_idx] == 1].mean(axis=0) - p[y[probe_idx] == 0].mean(axis=0)
        ranked = np.argsort(-deltas)

        n_opt, auc_tr = _select_n_opt(div_flat, y, train_idx, ranked, candidates)
        heads = ranked[:n_opt]

        s_test = div_flat[np.ix_(test_idx, heads)].mean(axis=1)
        scores[test_idx] = s_test

        m = detection_metrics(y[test_idx], s_test)
        m["n_opt"] = int(n_opt)
        m["n_eval"] = int(len(test_idx))
        fold_metrics.append(m)
        info.append(
            {
                "fold": int(k),
                "n_opt": int(n_opt),
                "auc_train": float(auc_tr),
                "top_heads": [(int(h // H), int(h % H)) for h in heads],
            }
        )

    return scores, fold_metrics, info
