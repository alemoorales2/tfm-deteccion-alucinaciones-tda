"""SEPs: sonda lineal de estados ocultos + selección de capa (etapa SIN modelo).

Sonda lineal (StandardScaler + LogisticRegression) sobre el hidden state del
último token, una por capa (Cap. 5 §5.4). Para cada fold de test, la capa óptima
se elige por validación interna (holdout estratificado dentro del bloque de
entrenamiento), sin ver el fold de test; luego esa capa se reentrena sobre todo
el train del fold y puntúa el test. La selección de capa nunca ve el test.

Es la variante directa hidden->etiqueta (familia INSIDE/SAPLMA), no el SEPs con
entropía semántica del paper original: se eligió por eficiencia (un único forward,
compartido con la extracción de atención). Ver la nota de la memoria §5.4.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import config
from ..evaluation.metrics import detection_metrics


def _make_probe(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=config.SEPS_LR_MAX_ITER, random_state=seed
        ),
    )


def _select_layer(hidden, y, train_idx, seed, val_frac):
    """Capa con mayor AUROC en un holdout interno del train (sin tocar el test)."""
    sub_tr, sub_val = train_test_split(
        train_idx, test_size=val_frac, stratify=y[train_idx], random_state=seed
    )
    best_l, best_auc = 0, -1.0
    for l in range(hidden.shape[1]):
        probe = _make_probe(seed)
        probe.fit(hidden[sub_tr, l, :], y[sub_tr])
        s = probe.predict_proba(hidden[sub_val, l, :])[:, 1]
        try:
            auc = roc_auc_score(y[sub_val], s)
        except ValueError:
            auc = 0.5
        if auc > best_auc:
            best_auc, best_l = auc, l
    return best_l, best_auc


def run_cv_seps(hidden, y, folds, seed=config.SEED, val_frac=None):
    """Validación cruzada de SEPs. Devuelve (scores, fold_metrics, info).

    `hidden`: [n, L_h, d_model]. Por fold se elige la capa por validación interna
    y se puntúa el test con una sonda reentrenada en todo el train.
    """
    val_frac = config.SEPS_VAL_FRAC if val_frac is None else val_frac
    hidden = np.asarray(hidden, dtype=np.float32)
    y = np.asarray(y)
    folds = np.asarray(folds)

    scores = np.full(len(y), np.nan, dtype=np.float64)
    fold_metrics, info = [], []
    for k in sorted(set(folds.tolist())):
        test_idx = np.where(folds == k)[0]
        train_idx = np.where(folds != k)[0]

        l_opt, auc_val = _select_layer(hidden, y, train_idx, seed + k, val_frac)
        probe = _make_probe(seed)
        probe.fit(hidden[train_idx, l_opt, :], y[train_idx])
        s = probe.predict_proba(hidden[test_idx, l_opt, :])[:, 1]
        scores[test_idx] = s

        m = detection_metrics(y[test_idx], s)
        m["n_eval"] = int(len(test_idx))
        m["layer"] = int(l_opt)
        fold_metrics.append(m)
        info.append({"fold": int(k), "layer": int(l_opt), "auc_val": float(auc_val)})
    return scores, fold_metrics, info
