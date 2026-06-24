"""Entrenamiento y evaluación de los métodos con clasificador (etapa SIN modelo).

Para los métodos entrenables (LapEigvals, y en Fase 3 HalluZig y SEPs) el
protocolo del Cap. 5 §5.6.1 entrena sobre cuatro pliegues y evalúa sobre el
restante. `make_estimator(n_train)` devuelve un estimador sklearn fresco por
pliegue (recibe el tamaño de entrenamiento para dimensionar, p. ej., el PCA).
"""

import numpy as np

from .metrics import detection_metrics


def metrics_by_fold(y, scores, folds):
    """Métricas por fold a partir de un score por muestra ya calculado (útil para
    métodos sin entrenamiento, como la longitud o SelfCheckGPT). Ignora las muestras
    con score NaN (p. ej. las que quedan fuera del subconjunto de SelfCheckGPT) y los
    folds que queden sin ambas clases."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    folds = np.asarray(folds)
    out = []
    for k in sorted(set(folds.tolist())):
        idx = np.where(folds == k)[0]
        idx = idx[~np.isnan(scores[idx])]
        if len(idx) < 2 or len(set(y[idx].tolist())) < 2:
            continue
        m = detection_metrics(y[idx], scores[idx])
        m["n_eval"] = int(len(idx))
        out.append(m)
    return out


def run_cv_classifier(X, y, folds, make_estimator, seed=42):
    """Validación cruzada de un clasificador. Devuelve (scores, fold_metrics).

    `scores[i]` es la probabilidad de alucinación de la muestra i, obtenida
    cuando i estaba en el pliegue de test.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    folds = np.asarray(folds)
    scores = np.full(len(y), np.nan, dtype=np.float64)
    fold_metrics = []

    for k in sorted(set(folds.tolist())):
        test_idx = np.where(folds == k)[0]
        train_idx = np.where(folds != k)[0]

        est = make_estimator(len(train_idx))
        est.fit(X[train_idx], y[train_idx])
        s = est.predict_proba(X[test_idx])[:, 1]
        scores[test_idx] = s

        m = detection_metrics(y[test_idx], s)
        m["n_eval"] = int(len(test_idx))
        fold_metrics.append(m)

    return scores, fold_metrics
