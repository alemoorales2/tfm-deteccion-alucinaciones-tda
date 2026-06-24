"""Particionado 5-fold estratificado (Cap. 5 §5.6.1), semilla 42.

Si se pasan `groups` (p. ej. el ítem de origen, compartido por el par
right/hallucinated de HaluEval-QA), se usa StratifiedGroupKFold para que las
muestras de un mismo grupo no se repartan entre train y test (evita la fuga por
prompt compartido).
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .. import config


def assign_folds(labels, groups=None, n_folds: int = config.N_FOLDS, seed: int = config.SEED) -> np.ndarray:
    """Asigna a cada muestra un índice de fold (0..n_folds-1), estratificado por
    etiqueta y, si se da `groups`, manteniendo cada grupo en un único fold."""
    labels = np.asarray(labels)
    folds = np.full(len(labels), -1, dtype=int)
    if groups is None:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splitter = skf.split(np.zeros(len(labels)), labels)
    else:
        groups = np.asarray(groups)
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splitter = sgkf.split(np.zeros(len(labels)), labels, groups)
    for k, (_, test_idx) in enumerate(splitter):
        folds[test_idx] = k
    return folds
